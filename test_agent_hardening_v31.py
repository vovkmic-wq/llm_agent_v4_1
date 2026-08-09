from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.memory import MemoryManager
from core.workspace import WorkspaceManager
from providers.base import CompletionResult


class ScriptedRouter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append({"model_id": model_id, "messages": [dict(item) for item in messages]})
        return CompletionResult(text=next(self.responses), raw_model_name=model_id)


def _agent(tmp_path: Path, responses: list[str]) -> tuple[Agent, ScriptedRouter]:
    workspace = WorkspaceManager(tmp_path / "workspace")
    router = ScriptedRouter(responses)
    return (
        Agent(
            memory=MemoryManager(tmp_path / "memory"),
            router=router,
            workspace=workspace,
            planner_enabled=False,
            reviewer_enabled=False,
            execution_enabled=False,
            research_enabled=False,
        ),
        router,
    )


def test_duplicate_call_id_cannot_overwrite_successful_evidence(tmp_path: Path) -> None:
    agent, router = _agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.txt","content":"one"},"call_id":"same"}',
            '{"type":"tool","tool":"workspace.delete_file","arguments":'
            '{"path":"a.txt"},"call_id":"same"}',
            '{"type":"final","content":"Первое действие сохранено.","evidence":["same"]}',
        ],
    )
    answer = agent.ask("Создай a.txt", model_id="test:model", force_task=True)
    assert "сохранено" in answer
    assert (agent.workspace.root / "a.txt").read_text(encoding="utf-8") == "one"
    assert "уже использован" in router.calls[2]["messages"][-1]["content"]


def test_second_mutation_requires_fresh_read(tmp_path: Path) -> None:
    agent, router = _agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"a.txt"},"call_id":"r1"}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.txt","content":"two","overwrite":true},"call_id":"w1"}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.txt","content":"three","overwrite":true},"call_id":"w2"}',
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"a.txt"},"call_id":"r2"}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.txt","content":"three","overwrite":true},"call_id":"w3"}',
            '{"type":"final","content":"Готово","evidence":["w3"]}',
        ],
    )
    agent.workspace.write_file("a.txt", "one")
    answer = agent.ask("Исправь a.txt", model_id="test:model", force_task=True)
    assert answer == "Готово"
    assert (agent.workspace.root / "a.txt").read_text(encoding="utf-8") == "three"
    denied_message = router.calls[3]["messages"][-1]["content"]
    assert "сначала вызовите workspace.read_file" in denied_message


def test_large_tool_result_is_clipped_before_next_llm_call(tmp_path: Path) -> None:
    agent, router = _agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"big.txt"},"call_id":"r"}',
            '{"type":"final","content":"Прочитано","evidence":["r"]}',
        ],
    )
    agent.workspace.write_file("big.txt", "x" * 20000)
    answer = agent.ask("Проверь файл big.txt", model_id="test:model", force_task=True)
    assert answer == "Прочитано"
    tool_result = router.calls[1]["messages"][-1]["content"]
    assert "truncated_for_context" in tool_result
    assert len(tool_result) < 12000
