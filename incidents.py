"""ICS-inspired persistent incident control for Agent V4.

The module deliberately keeps critical workflow rules outside the LLM prompt:
plans are sealed by SHA-256, source mutations are gated by the seal, the activity
log is append-only, operational periods are persisted to disk, and stop/attempt
ceilings are enforced mechanically.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str, limit: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9а-яА-ЯёЁ_-]+", "-", text.strip()).strip("-")
    return (value or "incident")[:limit]


class IncidentKind(StrEnum):
    TRIVIAL = "trivial"
    NORMAL = "normal"
    ARCHITECTURAL = "architectural"


class IncidentPhase(StrEnum):
    INTAKE = "intake"
    OBJECTIVES = "objectives"
    PLANNING = "planning"
    APPROVED = "approved"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    SAFETY_REVIEW = "safety_review"
    CLOSED = "closed"
    BLOCKED = "blocked"


class IncidentState(BaseModel):
    incident_id: str
    goal: str
    project: str
    kind: IncidentKind = IncidentKind.NORMAL
    phase: IncidentPhase = IncidentPhase.INTAKE
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    operational_period: int = 1
    period_llm_rounds: int = 0
    total_llm_rounds: int = 0
    plan_revisions: int = 0
    safety_stops: int = 0
    mutation_count: int = 0
    verification_count: int = 0
    no_progress_rounds: int = 0
    consecutive_read_only_rounds: int = 0
    last_failure_signature: str | None = None
    approved_plan_sha256: str | None = None
    followups: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncidentManager:
    """Persist and enforce the lifecycle of one active engineering incident."""

    def __init__(
        self,
        root: Path,
        *,
        max_period_rounds: int = 12,
        max_periods: int = 4,
        max_plan_revisions: int = 4,
        max_safety_stops: int = 4,
        max_read_only_rounds: int = 4,
        max_no_progress_rounds: int = 3,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_period_rounds = max_period_rounds
        self.max_periods = max_periods
        self.max_plan_revisions = max_plan_revisions
        self.max_safety_stops = max_safety_stops
        self.max_read_only_rounds = max_read_only_rounds
        self.max_no_progress_rounds = max_no_progress_rounds
        self.state: IncidentState | None = None
        self.path: Path | None = None

    @staticmethod
    def classify(goal: str) -> IncidentKind:
        lowered = goal.casefold()
        architectural = (
            "архитектур",
            "перепиши весь",
            "полностью переработ",
            "миграц",
            "схем",
            "security",
            "безопасност",
        )
        trivial = ("опечат", "readme", "документац", "комментар", "переимен")
        if any(token in lowered for token in architectural):
            return IncidentKind.ARCHITECTURAL
        if len(goal) < 180 and any(token in lowered for token in trivial):
            return IncidentKind.TRIVIAL
        return IncidentKind.NORMAL

    def start(self, goal: str, project: str, *, kind: IncidentKind | None = None) -> IncidentState:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        incident_id = f"{stamp}-{_slug(goal)}-{uuid.uuid4().hex[:6]}"
        self.path = self.root / incident_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.state = IncidentState(
            incident_id=incident_id,
            goal=goal,
            project=project,
            kind=kind or self.classify(goal),
        )
        self._write_state()
        brief = (
            f"# Incident Brief\n\n- ID: {incident_id}\n"
            f"- Project: {project}\n"
            f"- Kind: {self.state.kind.value}\n"
            f"- Opened: {self.state.created_at}\n\n"
            f"## Goal\n\n{goal}\n"
        )
        self._write_text("201-BRIEF.md", brief)
        self._write_text(
            "202-OBJECTIVES.md",
            "# Objectives\n\nPending planner output.\n",
        )
        self._write_text("SAFETY.md", "# Safety Officer\n\nNo verdict yet.\n")
        self._write_text("FOLLOWUPS.md", "# Follow-up incidents\n\n")
        self._append_log(
            "INCIDENT_OPENED",
            {"goal": goal, "project": project, "kind": self.state.kind.value},
        )
        return self.state

    def load(self, incident_id: str) -> IncidentState:
        self.path = self.root / incident_id
        raw = (self.path / "STATE.json").read_text(encoding="utf-8")
        self.state = IncidentState.model_validate_json(raw)
        return self.state

    def set_objectives(self, goal: str, acceptance: list[str]) -> None:
        state = self._require_state()
        state.phase = IncidentPhase.OBJECTIVES
        body = [
            "# Operational Objectives",
            "",
            f"## Goal\n\n{goal}",
            "",
            "## Acceptance criteria",
            "",
        ]
        body.extend(f"- {item}" for item in acceptance)
        self._write_text("202-OBJECTIVES.md", "\n".join(body) + "\n")
        self._append_log("OBJECTIVES_SET", {"acceptance": acceptance})
        self._write_state()

    def write_iap(self, steps: list[str], acceptance: list[str]) -> str:
        state = self._require_state()
        state.phase = IncidentPhase.PLANNING
        state.plan_revisions += 1
        if state.plan_revisions > self.max_plan_revisions:
            self.block(f"plan revision ceiling exceeded ({self.max_plan_revisions})")
            raise RuntimeError("Превышен механический потолок ревизий плана")
        lines = [
            "# Incident Action Plan",
            "",
            f"Incident: {state.incident_id}",
            f"Operational period: {state.operational_period}",
            f"Revision: {state.plan_revisions}",
            "",
            "## Tactics",
            "",
        ]
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
        lines.extend(["", "## Acceptance criteria", ""])
        lines.extend(f"- {item}" for item in acceptance)
        text = "\n".join(lines) + "\n"
        self._write_text("IAP.md", text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        state.approved_plan_sha256 = None
        self._append_log("IAP_WRITTEN", {"sha256": digest, "revision": state.plan_revisions})
        self._write_state()
        return digest

    def approve_iap(self, *, actor: str = "kernel") -> str:
        state = self._require_state()
        text = self._read_text("IAP.md")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        approval = {
            "incident_id": state.incident_id,
            "sha256": digest,
            "approved_by": actor,
            "approved_at": _utc_now(),
        }
        approval_text = json.dumps(approval, ensure_ascii=False, indent=2) + "\n"
        self._write_text("IAP-APPROVED.json", approval_text)
        state.approved_plan_sha256 = digest
        state.phase = IncidentPhase.APPROVED
        self._append_log("IAP_APPROVED", approval)
        self._write_state()
        return digest


    def capture_original(self, relative_path: str, content: str) -> None:
        """Persist the pre-change file once so Safety Officer can inspect a real diff."""
        base = self._require_path() / "originals"
        base.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]
        meta = base / f"{key}.json"
        data = base / f"{key}.txt"
        if meta.exists():
            return
        meta.write_text(
            json.dumps({"path": relative_path}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        data.write_text(content, encoding="utf-8", newline="\n")
        self._append_log("ORIGINAL_CAPTURED", {"path": relative_path})

    def diff_bundle(
        self,
        workspace_root: Path,
        changed_files: list[str],
        max_chars: int = 24000,
    ) -> str:
        """Build deterministic unified diffs from captured pre-change files."""
        base = self._require_path() / "originals"
        chunks: list[str] = []
        for relative_path in changed_files:
            original = ""
            if base.exists():
                for meta in base.glob("*.json"):
                    try:
                        payload = json.loads(meta.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if payload.get("path") == relative_path:
                        data = meta.with_suffix(".txt")
                        if data.exists():
                            original = data.read_text(encoding="utf-8")
                        break
            target = workspace_root / relative_path
            current = (
                target.read_text(encoding="utf-8")
                if target.exists() and target.is_file()
                else ""
            )
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{relative_path}",
                    tofile=f"b/{relative_path}",
                )
            )
            chunks.append(diff or f"# {relative_path}: no textual diff available\n")
            if sum(len(item) for item in chunks) >= max_chars:
                chunks.append("\n...[diff bundle truncated]...\n")
                break
        return "\n".join(chunks)[:max_chars]

    def mutation_gate(self) -> tuple[bool, str]:
        state = self._require_state()
        if state.kind == IncidentKind.TRIVIAL:
            return True, "trivial incident: plan gate not required"
        approval_path = self._require_path() / "IAP-APPROVED.json"
        iap_path = self._require_path() / "IAP.md"
        if not approval_path.exists() or not iap_path.exists():
            return False, "IAP не утверждён: mutation gate закрыт"
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False, "IAP-APPROVED повреждён: mutation gate закрыт"
        current = hashlib.sha256(iap_path.read_bytes()).hexdigest()
        sealed = str(approval.get("sha256", ""))
        if not sealed or current != sealed or current != state.approved_plan_sha256:
            return False, "SHA-256 IAP не совпадает с утверждением: требуется повторное утверждение"
        return True, "approved"

    def begin_execution(self) -> None:
        state = self._require_state()
        ok, reason = self.mutation_gate()
        if state.kind != IncidentKind.TRIVIAL and not ok:
            raise RuntimeError(reason)
        state.phase = IncidentPhase.EXECUTING
        self._append_log("EXECUTION_STARTED", {"period": state.operational_period})
        self._write_state()

    def record_llm_round(self) -> bool:
        """Return True when a fresh operational period must begin."""
        state = self._require_state()
        state.period_llm_rounds += 1
        state.total_llm_rounds += 1
        self._write_state()
        return state.period_llm_rounds > self.max_period_rounds

    def record_action(self, *, mutated: bool, read_only: bool, progress: bool) -> None:
        state = self._require_state()
        if mutated:
            state.mutation_count += 1
            state.consecutive_read_only_rounds = 0
        elif read_only:
            state.consecutive_read_only_rounds += 1
        else:
            state.consecutive_read_only_rounds = 0
        state.no_progress_rounds = 0 if progress else state.no_progress_rounds + 1
        self._write_state()

    def progress_gate(self) -> tuple[bool, str | None]:
        state = self._require_state()
        if state.consecutive_read_only_rounds >= self.max_read_only_rounds:
            return False, (
                f"read-only ceiling reached ({self.max_read_only_rounds}); "
                "следующий раунд обязан изменить проект, запустить целевую "
                "диагностику или объявить blocker"
            )
        if state.no_progress_rounds >= self.max_no_progress_rounds:
            return False, (
                f"no-progress ceiling reached ({self.max_no_progress_rounds}); "
                "запрещено продолжать обзорные чтения"
            )
        return True, None

    def record_verification(self, *, signature: str | None, ok: bool) -> bool:
        """Return whether verification changed objective state."""
        state = self._require_state()
        state.phase = IncidentPhase.VERIFYING
        state.verification_count += 1
        changed = signature != state.last_failure_signature
        state.last_failure_signature = signature
        self._append_log("VERIFICATION", {"ok": ok, "signature": signature, "changed": changed})
        self._write_state()
        return changed

    def rotate_period(self, reason: str, handoff: dict[str, Any]) -> None:
        state = self._require_state()
        if state.operational_period >= self.max_periods:
            self.block(f"operational period ceiling exceeded ({self.max_periods})")
            raise RuntimeError("Превышен потолок оперативных периодов")
        filename = f"PERIOD-{state.operational_period:02d}-HANDOFF.md"
        text = [
            f"# Operational Period {state.operational_period} Handoff",
            "",
            f"Reason: {reason}",
            "",
            "```json",
            json.dumps(handoff, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
        self._write_text(filename, "\n".join(text))
        self._append_log("PERIOD_HANDOFF", {"period": state.operational_period, "reason": reason})
        state.operational_period += 1
        state.period_llm_rounds = 0
        state.consecutive_read_only_rounds = 0
        # no_progress_rounds intentionally survives the handoff: an operational
        # period must not reset an attempt/stop ceiling.
        self._write_state()

    def safety_verdict(self, *, clear: bool, findings: list[str], model: str) -> None:
        state = self._require_state()
        state.phase = IncidentPhase.SAFETY_REVIEW
        if not clear:
            state.safety_stops += 1
            if state.safety_stops > self.max_safety_stops:
                self.block(f"safety stop ceiling exceeded ({self.max_safety_stops})")
                raise RuntimeError("Превышен потолок остановок Safety Officer")
        body = [
            "# Safety Officer",
            "",
            f"Verdict: {'CLEAR' if clear else 'STOP'}",
            f"Model: {model}",
            f"Time: {_utc_now()}",
            "",
            "## Findings",
            "",
        ]
        body.extend(f"- {item}" for item in (findings or ["No findings"]))
        self._write_text("SAFETY.md", "\n".join(body) + "\n")
        event = "SAFETY_CLEAR" if clear else "SAFETY_STOP"
        self._append_log(event, {"findings": findings, "model": model})
        self._write_state()

    def add_followup(self, text: str) -> None:
        state = self._require_state()
        state.followups.append(text)
        path = self._require_path() / "FOLLOWUPS.md"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"- {text}\n")
        self._append_log("FOLLOWUP_REGISTERED", {"text": text})
        self._write_state()

    def close(self, summary: str, metrics: dict[str, Any]) -> None:
        state = self._require_state()
        state.phase = IncidentPhase.CLOSED
        aar = [
            "# After Action Review",
            "",
            f"Incident: {state.incident_id}",
            "",
            "## Summary",
            "",
            summary,
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ]
        self._write_text("AAR.md", "\n".join(aar))
        self._append_log("INCIDENT_CLOSED", metrics)
        self._write_state()

    def block(self, reason: str) -> None:
        state = self._require_state()
        state.phase = IncidentPhase.BLOCKED
        state.metadata["blocked_reason"] = reason
        self._append_log("INCIDENT_BLOCKED", {"reason": reason})
        self._write_state()

    def render_context(self, max_chars: int = 12000) -> str:
        state = self._require_state()
        files = ["201-BRIEF.md", "202-OBJECTIVES.md", "IAP.md", "SAFETY.md", "FOLLOWUPS.md"]
        chunks = ["=== INCIDENT STATE ===", state.model_dump_json(indent=2)]
        for name in files:
            path = self._require_path() / name
            if path.exists():
                chunks.extend([f"\n=== {name} ===", path.read_text(encoding="utf-8")])
        text = "\n".join(chunks)
        return text if len(text) <= max_chars else text[-max_chars:]

    def _append_log(self, event: str, payload: dict[str, Any]) -> None:
        path = self._require_path() / "214-LOG.jsonl"
        record = {"timestamp": _utc_now(), "event": event, "payload": payload}
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _write_state(self) -> None:
        state = self._require_state()
        state.updated_at = _utc_now()
        path = self._require_path() / "STATE.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temp.replace(path)

    def _write_text(self, name: str, text: str) -> None:
        (self._require_path() / name).write_text(text, encoding="utf-8", newline="\n")

    def _read_text(self, name: str) -> str:
        return (self._require_path() / name).read_text(encoding="utf-8")

    def _require_state(self) -> IncidentState:
        if self.state is None:
            raise RuntimeError("Нет активного инцидента")
        return self.state

    def _require_path(self) -> Path:
        if self.path is None:
            raise RuntimeError("Нет активного каталога инцидента")
        return self.path
