from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.agent import Agent
from core.config import settings
from core.memory import MemoryManager
from core.protocol import ProtocolError, ToolCall, parse_agent_response
from core.workspace import WorkspaceManager
from providers.base import CompletionResult


def test_raw_payload_is_bound_without_json_escaping() -> None:
    text = (
        '{"type":"tool","tool":"workspace.write_file","arguments":'
        '{"path":"demo.py","overwrite":false},"call_id":"w1",'
        '"payload_id":"code1","payload_argument":"content"}\n'
        '<<<PAYLOAD:code1>>>\n'
        'def greet(name: str) -> str:\n'
        '    return f"Hello, {name}!"\n'
        '<<<END_PAYLOAD:code1>>>'
    )
    action = parse_agent_response(text)
    assert isinstance(action, ToolCall)
    assert action.arguments["content"] == (
        'def greet(name: str) -> str:\n    return f"Hello, {name}!"'
    )


def test_truncated_json_is_classified() -> None:
    with pytest.raises(ProtocolError) as caught:
        parse_agent_response(
            '{"type":"tool","tool":"workspace.write_file","arguments":'
            '{"path":"a.py","content":"def broken('
        )
    assert caught.value.code == "truncated_json"
    assert caught.value.likely_truncated is True


def test_replace_lines_checks_sha_and_validates_python(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("calc.py", "def add(a, b):\n    return a - b\n")
    read = workspace.read_file("calc.py")
    result = workspace.replace_lines(
        "calc.py",
        1,
        2,
        "def add(a, b):\n    return a + b",
        expected_sha256=read["sha256"],
    )
    assert result["verified"] is True
    assert "a + b" in workspace.read_file("calc.py")["content"]
    with pytest.raises(Exception):
        workspace.replace_lines(
            "calc.py",
            1,
            2,
            "def invalid(:",
            expected_sha256=result["sha256"],
        )
    assert "a + b" in workspace.read_file("calc.py")["content"]


class ProtocolFallbackRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.index = 0

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.calls.append(model_id)
        self.index += 1
        if self.index <= 2:
            return CompletionResult(
                text='{"type":"tool","tool":"workspace.write_file","arguments":{"path":"a.txt"',
                raw_model_name=model_id,
                finish_reason="length" if self.index == 1 else None,
            )
        if self.index == 3:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"a.txt","content":"ok"},"call_id":"w"}'
                ),
                raw_model_name=model_id,
            )
        return CompletionResult(
            text='{"type":"final","content":"Готово","evidence":["w"]}',
            raw_model_name=model_id,
        )


def test_protocol_errors_switch_executor_model(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings.agent.orchestration, "protocol_repairs_per_model", 1)
    monkeypatch.setattr(
        settings.agent.orchestration,
        "protocol_fallback_models",
        ["fallback:model"],
    )
    router = ProtocolFallbackRouter()
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=WorkspaceManager(tmp_path / "workspace"),
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Создай a.txt", model_id="primary:model", force_task=True)
    assert answer == "Готово"
    assert router.calls[:2] == ["primary:model", "primary:model"]
    assert router.calls[2] == "fallback:model"


class AutoFixRouter:
    def __init__(self, original_sha: str) -> None:
        self.index = 0
        self.original_sha = original_sha
        self.calls: list[dict] = []

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.index += 1
        self.calls.append({"model_id": model_id, "messages": list(messages)})
        if self.index == 1:
            # Baseline pytest failure should already be in context.
            assert "baseline_quality_gate_results" in messages[-1]["content"]
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.read_file","arguments":'
                    '{"path":"app/calc.py"},"call_id":"read"}'
                ),
                raw_model_name=model_id,
            )
        if self.index == 2:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.replace_lines","arguments":'
                    '{"path":"app/calc.py","start_line":1,"end_line":2,'
                    f'"expected_sha256":"{self.original_sha}"}},'
                    '"call_id":"patch","payload_id":"fix","payload_argument":"content"}\n'
                    '<<<PAYLOAD:fix>>>\n'
                    'def add(a, b):\n'
                    '    return a + b\n'
                    '<<<END_PAYLOAD:fix>>>'
                ),
                raw_model_name=model_id,
            )
        gate_payload = json.loads(messages[-1]["content"])
        assert gate_payload["type"] in {
            "post_mutation_quality_gate_results",
            "quality_gate_results",
        }
        assert gate_payload["all_ok"] is True
        gate_ids = [item["call_id"] for item in gate_payload["results"] if item["ok"]]
        return CompletionResult(
            text=json.dumps(
                {
                    "type": "final",
                    "content": "Исправлено и протестировано",
                    "evidence": ["read", "patch", *gate_ids],
                },
                ensure_ascii=False,
            ),
            raw_model_name=model_id,
        )


@pytest.mark.subprocess
def test_agent_baseline_failure_patch_and_retest(tmp_path: Path, monkeypatch) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("app")
    workspace.write_file("app/calc.py", "def add(a, b):\n    return a - b\n")
    workspace.write_file(
        "app/test_calc.py",
        "from calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    )
    original_sha = workspace.read_file("app/calc.py")["sha256"]
    monkeypatch.setattr(settings.agent.quality_gates, "compileall", True)
    monkeypatch.setattr(settings.agent.quality_gates, "pytest", True)
    monkeypatch.setattr(settings.agent.quality_gates, "ruff", False)
    monkeypatch.setattr(settings.agent.quality_gates, "mypy", False)
    router = AutoFixRouter(original_sha)
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
        "Протестируй app, исправляй код до успешного результата",
        model_id="test:model",
        force_task=True,
    )
    assert answer == "Исправлено и протестировано"
    assert "return a + b" in workspace.read_file("app/calc.py")["content"]
    assert any(
        item.startswith("kernel-gate-")
        for item in agent.task_state.current.successful_evidence
    )

class ProviderFallbackRouter:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        from providers.base import ProviderError

        self.calls.append(model_id)
        if model_id == "broken:model":
            raise ProviderError("network down")
        if len(self.calls) == 2:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"fallback.txt","content":"ok"},"call_id":"f1"}'
                ),
                raw_model_name=model_id,
            )
        return CompletionResult(
            text='{"type":"final","content":"fallback ok","evidence":["f1"]}',
            raw_model_name=model_id,
        )


def test_provider_error_switches_model_without_losing_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        settings.agent.orchestration,
        "protocol_fallback_models",
        ["healthy:model"],
    )
    router = ProviderFallbackRouter()
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=WorkspaceManager(tmp_path / "workspace"),
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Создай fallback.txt", model_id="broken:model", force_task=True)
    assert answer == "fallback ok"
    assert router.calls[:2] == ["broken:model", "healthy:model"]
    assert (agent.workspace.root / "fallback.txt").read_text(encoding="utf-8") == "ok"


def test_read_lines_returns_bounded_range_and_full_sha(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("many.py", "".join(f"line_{i} = {i}\n" for i in range(1, 101)))
    full = workspace.read_file("many.py")
    part = workspace.read_lines("many.py", start_line=10, end_line=12)
    assert part["sha256"] == full["sha256"]
    assert part["total_lines"] == 100
    assert part["content"].startswith("line_10")
    assert "line_12" in part["content"]
    assert "line_13" not in part["content"]
