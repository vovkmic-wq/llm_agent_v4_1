from pathlib import Path

import pytest

from core.agent import Agent
from core.memory import MemoryManager
from core.workspace import WorkspaceManager
from providers.base import CompletionResult


class ScriptedRouter:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]]):
        self.calls.append(
            {"model_id": model_id, "system_prompt": system_prompt, "messages": messages}
        )
        return CompletionResult(text=next(self.responses), raw_model_name=model_id)


@pytest.mark.subprocess
def test_agent_executes_terminal_compat_in_inferred_project(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("ozon_market_analytics")
    workspace.write_file(
        "ozon_market_analytics/test_sample.py",
        "def test_ok():\n    assert True\n",
    )
    router = ScriptedRouter(
        [
            '{"type":"tool","tool":"terminal.execute","arguments":'
            '{"command":"pytest -q","cwd":"ozon_market_analytics"},"call_id":"test-run"}',
            '{"type":"final","content":"Тесты реально выполнены и прошли.",'
            '"evidence":["test-run"]}',
        ]
    )
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=True,
        research_enabled=False,
    )
    answer = agent.ask(
        "Протестируй ozon_market_analytics",
        model_id="test:model",
        force_task=True,
    )
    assert "реально выполнены" in answer
    assert agent.project_context.active_project == "ozon_market_analytics"


def test_spec_is_injected_into_agent_system_prompt(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo_project")
    workspace.write_file(
        "demo_project/TECHNICAL_SPECIFICATION.md",
        "# Требования\n\n- Храни данные только в SQLite.\n",
    )
    router = ScriptedRouter(["ТЗ требует SQLite."])
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Что требует demo_project?", model_id="test:model")
    assert answer == "ТЗ требует SQLite."
    assert "Храни данные только в SQLite" in router.calls[0]["system_prompt"]


def test_testing_task_cannot_finish_with_read_only_evidence(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("app")
    workspace.write_file("app/test_ok.py", "def test_ok():\n    assert True\n")
    router = ScriptedRouter(
        [
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"app/test_ok.py"},"call_id":"read1"}',
            '{"type":"final","content":"Всё работает.","evidence":["read1"]}',
            '{"type":"tool","tool":"execution.run_pytest","arguments":'
            '{"cwd":"app"},"call_id":"pytest1"}',
            '{"type":"final","content":"Проверено реальным pytest.",'
            '"evidence":["read1","pytest1"]}',
        ]
    )
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=True,
        research_enabled=False,
    )
    answer = agent.ask(
        "Протестируй app на работоспособность",
        model_id="test:model",
        force_task=True,
    )
    assert answer == "Проверено реальным pytest."
    assert any(
        "фактического запуска" in message["content"]
        for message in router.calls[2]["messages"]
    )


def test_testing_task_fails_fast_when_execution_disabled(tmp_path: Path) -> None:
    from core.agent import AgentExecutionError

    workspace = WorkspaceManager(tmp_path / "workspace")
    router = ScriptedRouter([])
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    import pytest

    with pytest.raises(AgentExecutionError, match="AGENT_ALLOW_CODE_EXECUTION"):
        agent.ask("Протестируй код на работоспособность", model_id="test:model", force_task=True)


def test_kernel_quality_gates_run_inside_active_project(tmp_path: Path, monkeypatch) -> None:
    import json

    from core.config import settings

    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo_project")

    class GateRouter:
        def __init__(self) -> None:
            self.index = 0

        def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]]):
            self.index += 1
            if self.index == 1:
                text = (
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"demo_project/calc.py",'
                    '"content":"def add(a, b):\\n    return a + b\\n"},'
                    '"call_id":"write1"}'
                )
            elif self.index == 2:
                text = (
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"demo_project/test_calc.py","content":"from calc import add\\n\\n'
                    'def test_add():\\n    assert add(2, 3) == 5\\n"},"call_id":"write2"}'
                )
            else:
                payload = json.loads(messages[-1]["content"])
                gate_ids = [item["call_id"] for item in payload["results"] if item["ok"]]
                text = json.dumps(
                    {
                        "type": "final",
                        "content": "готово и проверено в demo_project",
                        "evidence": ["write1", "write2", *gate_ids],
                    },
                    ensure_ascii=False,
                )
            return CompletionResult(text=text, raw_model_name=model_id)

    monkeypatch.setattr(settings.agent.quality_gates, "compileall", True)
    monkeypatch.setattr(settings.agent.quality_gates, "pytest", True)
    monkeypatch.setattr(settings.agent.quality_gates, "ruff", False)
    monkeypatch.setattr(settings.agent.quality_gates, "mypy", False)
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=GateRouter(),
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=True,
        research_enabled=False,
    )

    gate_counter = {"value": 0}
    def fake_gates(evidence):
        gate_counter["value"] += 1
        call_id = f"kernel-gate-fake-{gate_counter['value']}"
        result = {"cwd": "demo_project", "exit_code": 0, "ok": True, "output": "ok"}
        evidence[call_id] = {"tool": "execution.run_pytest", "ok": True, "result": result}
        agent.task_state.record_tool(call_id=call_id, tool="execution.run_pytest", ok=True)
        payload = {
            "type": "tool_result",
            "call_id": call_id,
            "tool": "execution.run_pytest",
            "ok": True,
            "result": result,
        }
        return ([payload], True)

    monkeypatch.setattr(agent, "_run_quality_gates", fake_gates)
    answer = agent.ask(
        "Создай и протестируй demo_project",
        model_id="test:model",
        force_task=True,
    )
    assert answer == "готово и проверено в demo_project"
    gate_results = [
        item
        for item in agent.task_state.current.successful_evidence
        if item.startswith("kernel-gate-")
    ]
    assert len(gate_results) >= 2


def test_agent_executes_multiple_tool_calls_from_one_model_response(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("a.txt", "A")
    workspace.write_file("b.txt", "B")
    router = ScriptedRouter(
        [
            '```json\n'
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"a.txt"},"call_id":"a"}\n'
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"b.txt"},"call_id":"b"}\n```',
            '{"type":"final","content":"Оба файла прочитаны.","evidence":["a","b"]}',
        ]
    )
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Прочитай a.txt и b.txt", model_id="test:model", force_task=True)
    assert answer == "Оба файла прочитаны."
    payload = router.calls[1]["messages"][-1]["content"]
    assert "tool_batch_results" in payload
