from pathlib import Path

import pytest

from core.agent import Agent, AgentExecutionError
from core.memory import MemoryManager
from core.workspace import WorkspaceManager
from providers.base import CompletionResult


class ResumeRouter:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"resume.txt","content":"ok"},"call_id":"w1"}'
                ),
                raw_model_name=model_id,
            )
        return CompletionResult(
            text='{"type":"final","content":"Продолжено","evidence":["w1"]}',
            raw_model_name=model_id,
        )


def test_resume_reuses_existing_task_state(tmp_path: Path) -> None:
    router = ResumeRouter()
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=WorkspaceManager(tmp_path / "workspace"),
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    agent.task_state.start("Старая задача")
    agent.task_state.set_plan(["Сделать файл"], ["Файл существует"])
    old_task_id = agent.task_state.current.task_id
    answer = agent.resume("продолжай", model_id="test:model")
    assert answer == "Продолжено"
    assert agent.task_state.current.task_id == old_task_id
    assert agent.task_state.current.metadata["resume_count"] == 1
    assert (agent.workspace.root / "resume.txt").read_text(encoding="utf-8") == "ok"


def test_resume_requires_unfinished_task(tmp_path: Path) -> None:
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=ResumeRouter(),
        workspace=WorkspaceManager(tmp_path / "workspace"),
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    with pytest.raises(AgentExecutionError):
        agent.resume()
