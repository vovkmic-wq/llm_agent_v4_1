"""Персистентное структурированное состояние текущей инженерной задачи."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    id: int
    description: str
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"


class TaskState(BaseModel):
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    goal: str
    status: Literal["planning", "executing", "reviewing", "done", "failed"] = "planning"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    plan: list[PlanStep] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    successful_evidence: list[str] = Field(default_factory=list)
    failed_tools: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStateManager:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.current: TaskState | None = None
        if path.exists() and path.stat().st_size:
            try:
                self.current = TaskState.model_validate_json(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.current = None

    def start(self, goal: str) -> TaskState:
        self.current = TaskState(goal=goal)
        self.save()
        return self.current

    def set_plan(self, steps: list[str], acceptance_criteria: list[str]) -> None:
        if self.current is None:
            return
        self.current.plan = [PlanStep(id=i + 1, description=step) for i, step in enumerate(steps)]
        self.current.acceptance_criteria = acceptance_criteria
        self.current.status = "executing"
        self.save()

    def record_tool(
        self,
        *,
        call_id: str,
        tool: str,
        ok: bool,
        changed_path: str | None = None,
    ) -> None:
        if self.current is None:
            return
        if ok:
            if call_id not in self.current.successful_evidence:
                self.current.successful_evidence.append(call_id)
            if changed_path and changed_path not in self.current.files_changed:
                self.current.files_changed.append(changed_path)
        else:
            self.current.failed_tools.append(tool)
        self.save()

    def finish(self, ok: bool) -> None:
        if self.current is None:
            return
        self.current.status = "done" if ok else "failed"
        self.save()

    def save(self) -> None:
        if self.current is None:
            return
        self.current.updated_at = datetime.now(timezone.utc).isoformat()
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.current.model_dump(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(self.path)
