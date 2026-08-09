from pathlib import Path

import pytest

from core.incidents import IncidentManager


def make_incident(tmp_path: Path, **kwargs) -> IncidentManager:
    manager = IncidentManager(tmp_path / "incidents", **kwargs)
    manager.start("Исправить дефект расчёта", "demo")
    manager.set_objectives("Исправить дефект расчёта", ["pytest проходит"])
    manager.write_iap(
        ["Прочитать failing file", "Внести patch", "Запустить pytest"],
        ["pytest проходит"],
    )
    manager.approve_iap()
    return manager


def test_iap_sha_gate_is_mechanical(tmp_path: Path) -> None:
    manager = make_incident(tmp_path)
    ok, _ = manager.mutation_gate()
    assert ok is True
    iap = manager.path / "IAP.md"
    iap.write_text(iap.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")
    ok, reason = manager.mutation_gate()
    assert ok is False
    assert "SHA-256" in reason


def test_activity_log_is_append_only_and_period_handoff_persists(tmp_path: Path) -> None:
    manager = make_incident(tmp_path, max_period_rounds=2)
    assert manager.record_llm_round() is False
    assert manager.record_llm_round() is False
    assert manager.record_llm_round() is True
    manager.rotate_period("context full", {"x": 1})
    assert manager.state.operational_period == 2
    assert (manager.path / "PERIOD-01-HANDOFF.md").exists()
    log_lines = (manager.path / "214-LOG.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) >= 5


def test_progress_ceiling_survives_period_handoff(tmp_path: Path) -> None:
    manager = make_incident(tmp_path, max_no_progress_rounds=3)
    for _ in range(3):
        manager.record_action(mutated=False, read_only=True, progress=False)
    ok, reason = manager.progress_gate()
    assert ok is False
    assert "no-progress" in reason
    manager.rotate_period("handoff", {})
    ok, _ = manager.progress_gate()
    assert ok is False
    manager.record_action(mutated=False, read_only=False, progress=True)
    ok, _ = manager.progress_gate()
    assert ok is True


def test_plan_revision_and_safety_stop_ceilings_are_mechanical(tmp_path: Path) -> None:
    manager = IncidentManager(
        tmp_path / "incidents",
        max_plan_revisions=1,
        max_safety_stops=1,
    )
    manager.start("Архитектурная правка", "demo")
    manager.write_iap(["one"], [])
    with pytest.raises(RuntimeError, match="ревизий плана"):
        manager.write_iap(["two"], [])

    manager = make_incident(tmp_path / "second", max_safety_stops=1)
    manager.safety_verdict(clear=False, findings=["x"], model="m")
    with pytest.raises(RuntimeError, match="Safety Officer"):
        manager.safety_verdict(clear=False, findings=["y"], model="m")


def test_incident_close_writes_aar(tmp_path: Path) -> None:
    manager = make_incident(tmp_path)
    manager.close("Исправлено", {"tests": "pass"})
    assert (manager.path / "AAR.md").exists()
    assert manager.state.phase.value == "closed"



def test_unavailable_safety_officer_blocks_incident(tmp_path: Path) -> None:
    from core.agent import Agent, AgentExecutionError
    from core.memory import MemoryManager
    from core.workspace import WorkspaceManager
    from providers.base import CompletionResult, ProviderError

    class SafetyFailureRouter:
        def __init__(self) -> None:
            self.calls = 0

        def call(
            self,
            *,
            model_id: str,
            system_prompt: str,
            messages: list[dict[str, str]],
        ) -> CompletionResult:
            self.calls += 1
            if self.calls == 1:
                return CompletionResult(
                    text=(
                        '{"type":"tool","tool":"workspace.read_file",'
                        '"arguments":{"path":"architecture.txt"},'
                        '"call_id":"read-architecture"}'
                    ),
                    raw_model_name=model_id,
                )
            if self.calls == 2:
                return CompletionResult(
                    text=(
                        '{"type":"final","content":"Анализ завершён",'
                        '"evidence":["read-architecture"]}'
                    ),
                    raw_model_name=model_id,
                )
            raise ProviderError("safety provider unavailable", retryable=False)

    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("architecture.txt", "architecture notes")
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=SafetyFailureRouter(),
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=True,
        execution_enabled=False,
        research_enabled=False,
    )

    try:
        agent.ask(
            "Проанализируй архитектурное решение и выдай итог",
            model_id="test:model",
            force_task=True,
        )
    except AgentExecutionError as exc:
        assert "safety-проверка недоступна" in str(exc)
    else:
        raise AssertionError("Safety provider failure must block the incident")

    assert agent.incident.state is not None
    assert agent.incident.state.phase.value == "blocked"
    assert "Safety Officer unavailable" in str(
        agent.incident.state.metadata.get("blocked_reason")
    )
