"""Persistent conversational memory with rolling-summary checkpoints.

Technical specifications and project decisions live in ``ProjectContextManager`` and
have higher prompt priority. This module stores conversational history only.
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

MEMORY_DIR = Path(
    os.environ.get("AGENT_STATE_DIR", Path.home() / ".llm_agent_v3")
).expanduser().resolve() / "memory"
CONTEXT_FILE = MEMORY_DIR / "context.md"
HISTORY_FILE = MEMORY_DIR / "history.jsonl"
SUMMARY_FILE = MEMORY_DIR / "summary.md"
SUMMARY_STATE_FILE = MEMORY_DIR / "summary_state.json"


@dataclass
class Exchange:
    """One persisted user/assistant exchange."""

    timestamp: str
    user_query: str
    assistant_response: str
    model_used: str
    tags: list[str]

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "user_query": self.user_query,
                "assistant_response": self.assistant_response,
                "model_used": self.model_used,
                "tags": self.tags,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def from_dict(data: dict) -> "Exchange":
        return Exchange(
            timestamp=str(data["timestamp"]),
            user_query=str(data["user_query"]),
            assistant_response=str(data["assistant_response"]),
            model_used=str(data.get("model_used", "unknown")),
            tags=[str(item) for item in data.get("tags", [])],
        )


class MemoryManager:
    def __init__(self, memory_dir: Path = MEMORY_DIR) -> None:
        self.memory_dir = memory_dir
        self.context_file = memory_dir / "context.md"
        self.history_file = memory_dir / "history.jsonl"
        self.summary_file = memory_dir / "summary.md"
        self.summary_state_file = memory_dir / "summary_state.json"
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.context_file.touch(exist_ok=True)
        self.history_file.touch(exist_ok=True)
        self.summary_file.touch(exist_ok=True)
        if not self.summary_state_file.exists():
            self._write_summary_checkpoint(0)

    def load_context(self) -> str:
        return self.context_file.read_text(encoding="utf-8").strip()

    def load_summary(self) -> str:
        return self.summary_file.read_text(encoding="utf-8").strip()

    def _iter_messages(self) -> Iterator[Exchange]:
        if not self.history_file.exists():
            return
        with self.history_file.open("r", encoding="utf-8") as stream:
            for line in stream:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        continue
                    yield Exchange.from_dict(payload)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # One damaged line must not make all subsequent memory unusable.
                    continue

    def all_messages(self) -> list[Exchange]:
        return list(self._iter_messages())

    def recent_messages(self, n: int) -> list[Exchange]:
        if n <= 0:
            return []
        recent: deque[Exchange] = deque(maxlen=n)
        recent.extend(self._iter_messages())
        return list(recent)

    def messages_from(self, start_index: int) -> list[Exchange]:
        start = max(0, int(start_index))
        return [message for index, message in enumerate(self._iter_messages()) if index >= start]

    def messages_count(self) -> int:
        return sum(1 for _ in self._iter_messages())

    def append_exchange(
        self,
        user_query: str,
        assistant_response: str,
        model_used: str,
        tags: list[str] | None = None,
    ) -> Exchange:
        exchange = Exchange(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_query=user_query,
            assistant_response=assistant_response,
            model_used=model_used,
            tags=tags or [],
        )
        with self.history_file.open("a", encoding="utf-8") as stream:
            stream.write(exchange.to_json() + "\n")
        return exchange

    def overwrite_summary(self, summary_text: str) -> None:
        self.summary_file.write_text(summary_text.strip() + "\n", encoding="utf-8")

    def append_context_fact(self, fact: str) -> None:
        with self.context_file.open("a", encoding="utf-8") as stream:
            stream.write(f"\n- {fact}")

    def summary_checkpoint(self) -> int:
        try:
            payload = json.loads(self.summary_state_file.read_text(encoding="utf-8"))
            value = int(payload.get("last_summarized_count", 0))
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError):
            return 0
        return max(0, value)

    def _write_summary_checkpoint(self, count: int) -> None:
        payload = {"last_summarized_count": max(0, int(count))}
        temp = self.summary_state_file.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.summary_state_file)

    def mark_summary_checkpoint(self, count: int | None = None) -> None:
        self._write_summary_checkpoint(self.messages_count() if count is None else count)
