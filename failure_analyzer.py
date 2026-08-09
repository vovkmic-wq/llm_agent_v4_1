"""Deterministic test-output analysis used before asking an LLM for diagnosis."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


_FAILED_RE = re.compile(r"^FAILED\s+([^\s]+)", re.MULTILINE)
_FRAME_RE = re.compile(r"(?P<path>(?:[A-Za-z]:)?[^\s:]+\.py):(?P<line>\d+)")
_ERROR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b")


@dataclass(slots=True)
class FailureDigest:
    signature: str
    failed_tests: list[str] = field(default_factory=list)
    relevant_files: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    summary: str = ""


def analyze_test_output(output: str, project: str = ".") -> FailureDigest:
    normalized = output.replace("\\", "/")
    failed = list(dict.fromkeys(_FAILED_RE.findall(normalized)))[:20]
    files: list[str] = []
    project_prefix = project.strip("./").replace("\\", "/")
    for match in _FRAME_RE.finditer(normalized):
        raw = match.group("path").replace("\\", "/")
        if "/site-packages/" in raw or "/lib/python" in raw.casefold():
            continue
        if project_prefix and project_prefix != "." and raw.startswith(project_prefix + "/"):
            raw = raw[len(project_prefix) + 1 :]
        path = Path(raw).as_posix()
        if path not in files:
            files.append(path)
        if len(files) >= 20:
            break
    exceptions = list(dict.fromkeys(_ERROR_RE.findall(normalized)))[:10]
    basis = "\n".join([*failed, *files, *exceptions]) or normalized[-4000:]
    signature = hashlib.sha256(basis.encode("utf-8", errors="replace")).hexdigest()[:20]
    summary_lines = [
        f"failed_tests={len(failed)}",
        f"relevant_files={files[:10]}",
        f"exceptions={exceptions[:10]}",
    ]
    return FailureDigest(
        signature=signature,
        failed_tests=failed,
        relevant_files=files,
        exceptions=exceptions,
        summary="; ".join(summary_lines),
    )
