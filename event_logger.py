"""Structured JSONL event log with aggressive secret/content redaction."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_SECRET_MARKERS = (
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
    "cookie",
)
_CONTENT_MARKERS = (
    "content",
    "old_text",
    "new_text",
    "text",
    "query",
    "prompt",
    "response",
    "output",
    "feedback",
    "tail",
    "source",
)
_URL_MARKERS = ("url", "uri", "endpoint")


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _content_fingerprint(value: str) -> dict[str, int | str]:
    return {"chars": len(value), "sha256_16": _hash_text(value)}


def _sanitize_url(value: str) -> str | dict[str, int | str]:
    """Remove credentials, query and fragment from URLs before logging."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _content_fingerprint(value)
    if not parsed.scheme or not parsed.netloc:
        return _content_fingerprint(value)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    safe_netloc = host + port
    return urlunsplit((parsed.scheme, safe_netloc, parsed.path, "", ""))


def sanitize(value: Any, *, key: str = "") -> Any:
    """Sanitize values before they reach the technical event log.

    Prompt/source/output bodies are never stored verbatim. Secret-looking fields are
    fully redacted. Long miscellaneous strings are fingerprinted without previews.
    """
    lowered = key.casefold()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): sanitize(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item, key=key) for item in value[:100]]
    if isinstance(value, str):
        if any(marker in lowered for marker in _URL_MARKERS):
            return _sanitize_url(value)
        if any(marker in lowered for marker in _CONTENT_MARKERS):
            return _content_fingerprint(value)
        if len(value) > 1000:
            return _content_fingerprint(value)
    return value


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **sanitize(data),
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
