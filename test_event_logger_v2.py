import json
from pathlib import Path

from core.event_logger import EventLogger


def test_event_logger_redacts_secrets_and_file_content(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path)
    logger.log(
        "tool_call",
        arguments={"api_key": "secret", "content": "very long source code"},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["arguments"]["api_key"] == "<redacted>"
    assert record["arguments"]["content"]["chars"] == len("very long source code")
    assert "very long source code" not in path.read_text(encoding="utf-8")
