"""Console and structured request logging for LLM Coding Agent V3.

Runtime logs are written to ``AGENT_STATE_DIR`` (``~/.llm_agent_v3`` by default),
never into the installed package directory. User prompts are represented only by
length and a short SHA-256 digest so secrets or source code are not copied into the
transport log.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone

from core.config import STATE_DIR

LOGS_DIR = STATE_DIR / "logs"
REQUESTS_LOG = LOGS_DIR / "requests.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
console_logger = logging.getLogger("llm_agent")


def _query_fingerprint(query: str) -> dict[str, int | str]:
    encoded = query.encode("utf-8", errors="replace")
    return {
        "chars": len(query),
        "sha256_16": hashlib.sha256(encoded).hexdigest()[:16],
    }


def log_request(
    *,
    model: str,
    query: str,
    success: bool,
    duration_seconds: float,
    error: str | None = None,
) -> None:
    """Append one transport-level request record without storing prompt contents."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "query": _query_fingerprint(query),
        "success": success,
        "duration_seconds": round(duration_seconds, 3),
        "error": error,
    }
    with REQUESTS_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if success:
        console_logger.info("OK  [%s] %.2fs", model, duration_seconds)
    else:
        console_logger.error("FAIL [%s] %.2fs — %s", model, duration_seconds, error)
