"""Persistent active-project context and technical-specification retrieval.

The chat history is intentionally *not* the source of truth for project requirements.
A technical specification is stored separately, versioned by SHA-256 and injected into
all planner/executor/reviewer calls with higher priority than conversational memory.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.workspace import WorkspaceError, WorkspaceManager


SPEC_CANDIDATES = (
    "TECHNICAL_SPECIFICATION.md",
    "technical_specification.md",
    "SPECIFICATION.md",
    "SPEC.md",
    "REQUIREMENTS.md",
    "SRS.md",
    "ТЕХНИЧЕСКОЕ_ЗАДАНИЕ.md",
    "ТЗ.md",
)


class ProjectContextError(RuntimeError):
    """Project-context configuration or persistence error."""


class ProjectContextState(BaseModel):
    active_project: str = "."
    specification_source: str | None = None
    specification_sha256: str | None = None
    specification_updated_at: str | None = None
    context_facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SpecificationSnapshot:
    text: str
    source: str
    sha256: str


class ProjectContextManager:
    """Keeps project contract independent from transient conversation history."""

    def __init__(self, memory_dir: Path, workspace: WorkspaceManager) -> None:
        self.workspace = workspace
        self.root = memory_dir / "project_context"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.root / "project.json"
        self.spec_file = self.root / "technical_specification.md"
        self._state = self._load_state()

    def _load_state(self) -> ProjectContextState:
        if not self.state_file.exists():
            state = ProjectContextState()
            self._save_state(state)
            return state
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
            return ProjectContextState.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProjectContextError(f"Повреждён {self.state_file}: {exc}") from exc

    def _save_state(self, state: ProjectContextState | None = None) -> None:
        current = state or self._state
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(current.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.state_file)

    @property
    def state(self) -> ProjectContextState:
        return self._state

    @property
    def active_project(self) -> str:
        return self._state.active_project

    def active_project_path(self) -> Path:
        return self.workspace.resolve_path(self.active_project, allow_root=True)

    def _candidate_paths(self) -> tuple[str, ...]:
        active = self.active_project
        return tuple(
            name if active == "." else f"{active}/{name}"
            for name in SPEC_CANDIDATES
        )

    @staticmethod
    def _workspace_source_path(source: str | None) -> str | None:
        if not source or not source.startswith("workspace:"):
            return None
        value = source.removeprefix("workspace:").strip()
        return value or None

    def set_active_project(self, path: str) -> str:
        target = self.workspace.resolve_path(path, allow_root=True)
        if not target.exists() or not target.is_dir():
            raise ProjectContextError(f"Каталог проекта не существует: {path}")
        relative = target.relative_to(self.workspace.root).as_posix() or "."
        if relative == "":
            relative = "."
        if relative != self._state.active_project:
            self._state.active_project = relative
            self._state.specification_source = None
            self._state.specification_sha256 = None
            self._state.specification_updated_at = None
            self._state.context_facts = []
            self._state.decisions = []
            if self.spec_file.exists():
                self.spec_file.unlink()
            self._save_state()
        self.auto_sync_specification()
        return relative

    def infer_project_from_query(self, query: str) -> str | None:
        """Select a top-level workspace directory when its name is mentioned verbatim."""
        lowered = query.casefold()
        matches: list[tuple[int, str]] = []
        listing = self.workspace.list_files(".", recursive=False)
        for item in listing.get("entries", []):
            if item.get("type") != "dir":
                continue
            name = Path(str(item.get("path", ""))).name
            if name and name.casefold() in lowered:
                matches.append((len(name), name))
        if not matches:
            return None
        _, chosen = max(matches)
        self.set_active_project(chosen)
        return chosen

    @staticmethod
    def _sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _store_spec(self, text: str, source: str) -> SpecificationSnapshot:
        normalized = text.strip() + "\n"
        sha = self._sha(normalized)
        self.spec_file.write_text(normalized, encoding="utf-8")
        self._state.specification_source = source
        self._state.specification_sha256 = sha
        self._state.specification_updated_at = datetime.now(timezone.utc).isoformat()
        self._save_state()
        return SpecificationSnapshot(normalized, source, sha)

    def set_specification(self, text: str, source: str = "cli:/spec") -> SpecificationSnapshot:
        if not text.strip():
            raise ProjectContextError("Техническое задание пустое")
        return self._store_spec(text, source)

    def import_specification(self, workspace_path: str) -> SpecificationSnapshot:
        try:
            payload = self.workspace.read_file(workspace_path)
        except WorkspaceError as exc:
            raise ProjectContextError(str(exc)) from exc
        return self._store_spec(
            str(payload["content"]),
            f"workspace:{payload['path']}",
        )

    def auto_sync_specification(self) -> SpecificationSnapshot | None:
        """Refresh the project contract from workspace sources before it is consumed.

        Resolution order is deterministic:
        1. conventional specification files in the active project root;
        2. an explicitly imported workspace specification (``/spec load``);
        3. a manually entered persistent specification from project context.

        A previously auto-discovered conventional file is not kept as a stale cache after
        it disappears from the active project. Manual CLI specifications remain persistent.
        """
        candidate_paths = self._candidate_paths()
        for workspace_path in candidate_paths:
            try:
                payload = self.workspace.read_file(workspace_path)
            except WorkspaceError:
                continue
            text = str(payload["content"])
            source = f"workspace:{payload['path']}"
            normalized = text.strip() + "\n"
            sha = self._sha(normalized)
            if (
                sha != self._state.specification_sha256
                or source != self._state.specification_source
            ):
                return self._store_spec(text, source)
            return SpecificationSnapshot(normalized, source, sha)

        current_source_path = self._workspace_source_path(
            self._state.specification_source
        )
        if current_source_path and current_source_path not in candidate_paths:
            try:
                payload = self.workspace.read_file(current_source_path)
            except WorkspaceError:
                return self.load_specification()
            text = str(payload["content"])
            source = f"workspace:{payload['path']}"
            normalized = text.strip() + "\n"
            sha = self._sha(normalized)
            if (
                sha != self._state.specification_sha256
                or source != self._state.specification_source
            ):
                return self._store_spec(text, source)
            return SpecificationSnapshot(normalized, source, sha)

        if current_source_path in candidate_paths:
            self.clear_specification()
            return None

        return self.load_specification()

    def load_specification(self) -> SpecificationSnapshot | None:
        if not self.spec_file.exists() or not self._state.specification_sha256:
            return None
        text = self.spec_file.read_text(encoding="utf-8")
        return SpecificationSnapshot(
            text=text,
            source=self._state.specification_source or "memory",
            sha256=self._state.specification_sha256,
        )

    def clear_specification(self) -> None:
        if self.spec_file.exists():
            self.spec_file.unlink()
        self._state.specification_source = None
        self._state.specification_sha256 = None
        self._state.specification_updated_at = None
        self._save_state()

    def add_context_fact(self, text: str) -> None:
        value = text.strip()
        if value and value not in self._state.context_facts:
            self._state.context_facts.append(value)
            self._state.context_facts = self._state.context_facts[-200:]
            self._save_state()

    def add_decision(self, text: str) -> None:
        value = text.strip()
        if value and value not in self._state.decisions:
            self._state.decisions.append(value)
            self._state.decisions = self._state.decisions[-200:]
            self._save_state()

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_\-]{4,}", text)
        }

    @staticmethod
    def _chunks(text: str, max_chunk_chars: int = 1800) -> list[str]:
        blocks = re.split(r"(?=^#{1,6}\s)|\n{2,}", text, flags=re.MULTILINE)
        chunks: list[str] = []
        buffer = ""
        for block in (item.strip() for item in blocks if item.strip()):
            if len(buffer) + len(block) + 2 <= max_chunk_chars:
                buffer = f"{buffer}\n\n{block}".strip()
                continue
            if buffer:
                chunks.append(buffer)
            if len(block) <= max_chunk_chars:
                buffer = block
            else:
                chunks.extend(
                    block[index : index + max_chunk_chars]
                    for index in range(0, len(block), max_chunk_chars)
                )
                buffer = ""
        if buffer:
            chunks.append(buffer)
        return chunks

    @staticmethod
    def _requirements_index(text: str, max_chars: int = 5000) -> str:
        selected: list[str] = []
        size = 0
        pattern = re.compile(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s)")
        for raw in text.splitlines():
            line = raw.strip()
            if not line or not pattern.match(line) or len(line) > 350:
                continue
            extra = len(line) + 1
            if size + extra > max_chars:
                break
            selected.append(line)
            size += extra
        return "\n".join(selected)

    def render_for_prompt(self, query: str, max_chars: int = 14000) -> str:
        """Build a high-priority project contract within a deterministic budget."""
        self.auto_sync_specification()
        header = [f"ACTIVE_PROJECT={self.active_project}"]
        spec = self.load_specification()
        if spec:
            header.extend(
                [
                    f"SPEC_SOURCE={spec.source}",
                    f"SPEC_SHA256={spec.sha256}",
                    (
                        "SPEC_POLICY=ТЗ является контрактом: не изменяй его смысл "
                        "и проверяй результат по нему."
                    ),
                ]
            )
        else:
            header.append("SPEC_SOURCE=(ТЗ не загружено)")

        if self._state.context_facts:
            header.append("PROJECT_FACTS:\n- " + "\n- ".join(self._state.context_facts[-30:]))
        if self._state.decisions:
            header.append("DECISIONS:\n- " + "\n- ".join(self._state.decisions[-30:]))

        base = "\n".join(header)
        if not spec:
            return base[:max_chars]

        remaining = max(0, max_chars - len(base) - 200)
        index_budget = min(5000, max(1200, remaining // 3))
        requirements_index = self._requirements_index(spec.text, index_budget)
        query_terms = self._terms(query)
        chunks = self._chunks(spec.text)
        scored: list[tuple[int, int, str]] = []
        for idx, chunk in enumerate(chunks):
            score = len(query_terms.intersection(self._terms(chunk)))
            if any(
                word in query.casefold()
                for word in ("тз", "техничес", "соответств", "requirements", "spec")
            ):
                score += 1
            scored.append((score, -idx, chunk))
        scored.sort(reverse=True)

        parts = [base]
        if requirements_index:
            parts.append("SPEC_REQUIREMENTS_INDEX:\n" + requirements_index)
        used = sum(len(item) for item in parts) + 20
        selected_indexes: set[str] = set()
        for _, _, chunk in scored:
            fingerprint = self._sha(chunk)[:12]
            if fingerprint in selected_indexes:
                continue
            addition = "\n\nSPEC_RELEVANT_SECTION:\n" + chunk
            if used + len(addition) > max_chars:
                continue
            parts.append(addition)
            selected_indexes.add(fingerprint)
            used += len(addition)
        return "\n\n".join(parts)[:max_chars]

    def describe(self) -> dict[str, Any]:
        spec = self.load_specification()
        return {
            "active_project": self.active_project,
            "specification_source": spec.source if spec else None,
            "specification_sha256": spec.sha256 if spec else None,
            "specification_chars": len(spec.text) if spec else 0,
            "context_facts": list(self._state.context_facts),
            "decisions": list(self._state.decisions),
        }
