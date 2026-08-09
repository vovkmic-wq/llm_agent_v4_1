from pathlib import Path

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
            {
                "model_id": model_id,
                "system_prompt": system_prompt,
                "messages": list(messages),
            }
        )
        return CompletionResult(text=next(self.responses), raw_model_name=model_id)


def build_agent(tmp_path: Path, responses: list[str], **kwargs) -> tuple[Agent, ScriptedRouter]:
    workspace = WorkspaceManager(tmp_path / "workspace")
    memory = MemoryManager(tmp_path / "memory")
    router = ScriptedRouter(responses)
    agent = Agent(
        memory=memory,
        router=router,
        workspace=workspace,
        planner_enabled=kwargs.pop("planner_enabled", False),
        reviewer_enabled=kwargs.pop("reviewer_enabled", False),
        execution_enabled=kwargs.pop("execution_enabled", False),
        research_enabled=False,
        **kwargs,
    )
    return agent, router


def test_agent_executes_tool_and_requires_evidence(tmp_path: Path) -> None:
    agent, router = build_agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"hello.txt","content":"hello","overwrite":false},"call_id":"c1"}',
            '{"type":"final","content":"Файл создан.","evidence":[]}',
            '{"type":"final","content":"Файл создан.","evidence":["c1"]}',
        ],
    )
    answer = agent.ask("Создай hello.txt", model_id="test:model")
    assert answer == "Файл создан."
    assert (agent.workspace.root / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert len(router.calls) == 3
    assert "final_validation_error" in router.calls[2]["messages"][-1]["content"]


def test_agent_repairs_malformed_tool_protocol(tmp_path: Path) -> None:
    agent, router = build_agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.write_file","arguments":{"path":"a.txt"}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.txt","content":"ok"},"call_id":"c2"}',
            '{"type":"final","content":"Готово","evidence":["c2"]}',
        ],
    )
    answer = agent.ask("Создай файл a.txt", model_id="test:model")
    assert answer == "Готово"
    assert "protocol_error" in router.calls[1]["messages"][-1]["content"]
    assert (agent.workspace.root / "a.txt").exists()


def test_agent_reports_security_error_back_to_model(tmp_path: Path) -> None:
    agent, router = build_agent(
        tmp_path,
        [
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"../escape.txt","content":"bad"},"call_id":"bad1"}',
            '{"type":"final","status":"blocked",'
            '"content":"Операция отклонена политикой безопасности.",'
            '"evidence":["bad1"]}',
        ],
    )
    answer = agent.ask("Создай файл ../escape.txt", model_id="test:model")
    assert "отклонена" in answer
    assert not (tmp_path / "escape.txt").exists()
    assert '"ok": false' in router.calls[1]["messages"][-1]["content"]


def test_plain_chat_can_return_plain_text_without_protocol(tmp_path: Path) -> None:
    agent, _ = build_agent(tmp_path, ["Здравствуйте!"])
    answer = agent.ask("привет", model_id="test:model")
    assert answer == "Здравствуйте!"


def test_planner_executor_reviewer_roles(tmp_path: Path) -> None:
    agent, router = build_agent(
        tmp_path,
        [
            '{"goal":"создать note","steps":["создать файл"],'
            '"acceptance_criteria":["файл существует"]}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"note.txt","content":"v2"},"call_id":"n1"}',
            '{"type":"final","content":"note создан","evidence":["n1"]}',
            '{"clear":true,"findings":[]}',
            '{"approved":true,"feedback":"ok"}',
        ],
        planner_enabled=True,
        reviewer_enabled=True,
    )
    answer = agent.ask("Создай note.txt", model_id="test:model", force_task=True)
    assert answer == "note создан"
    assert len(router.calls) == 5
    assert agent.task_state.current is not None
    assert agent.task_state.current.status == "done"


def test_existing_file_must_be_read_before_mutation(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("existing.txt", "old")
    memory = MemoryManager(tmp_path / "memory")
    router = ScriptedRouter(
        [
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"existing.txt","content":"new","overwrite":true},"call_id":"w1"}',
            '{"type":"tool","tool":"workspace.read_file","arguments":'
            '{"path":"existing.txt"},"call_id":"r1"}',
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"existing.txt","content":"new","overwrite":true},"call_id":"w2"}',
            '{"type":"final","content":"изменено","evidence":["r1","w2"]}',
        ]
    )
    agent = Agent(
        memory=memory,
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Измени файл existing.txt", model_id="test:model")
    assert answer == "изменено"
    assert workspace.read_file("existing.txt")["content"] == "new"
    assert "сначала вызовите" in router.calls[1]["messages"][-1]["content"]


def test_agent_auto_runs_pytest_quality_gate(tmp_path: Path, monkeypatch) -> None:
    from core.config import settings

    workspace = WorkspaceManager(tmp_path / "workspace")
    memory = MemoryManager(tmp_path / "memory")

    class QualityRouter:
        def __init__(self) -> None:
            self.index = 0

        def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]]):
            self.index += 1
            if self.index == 1:
                text = (
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"calc.py","content":"def add(a, b):\\n    return a + b\\n"},'
                    '"call_id":"q1"}'
                )
            elif self.index == 2:
                text = (
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"test_calc.py","content":"from calc import add\\n\\n'
                    'def test_add():\\n    assert add(2, 3) == 5\\n"},"call_id":"q2"}'
                )
            else:
                payload = json.loads(messages[-1]["content"])
                gate_ids = [item["call_id"] for item in payload["results"] if item["ok"]]
                evidence = ["q1", "q2", *gate_ids]
                text = json.dumps(
                    {"type": "final", "content": "готово и проверено", "evidence": evidence},
                    ensure_ascii=False,
                )
            return CompletionResult(text=text, raw_model_name=model_id)

    import json

    monkeypatch.setattr(settings.agent.quality_gates, "pytest", True)
    monkeypatch.setattr(settings.agent.quality_gates, "ruff", False)
    monkeypatch.setattr(settings.agent.quality_gates, "mypy", False)
    agent = Agent(
        memory=memory,
        router=QualityRouter(),
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
        result = {"cwd": ".", "exit_code": 0, "ok": True, "output": "ok"}
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
    answer = agent.ask("Создай и протестируй calc", model_id="test:model", force_task=True)
    assert answer == "готово и проверено"
    assert agent.task_state.current is not None
    assert any(
        item.startswith("kernel-gate-")
        for item in agent.task_state.current.successful_evidence
    )
