from pathlib import Path

from core.agent import Agent
from core.memory import MemoryManager
from core.workspace import WorkspaceManager
from providers.base import CompletionResult


class ReadLoopRouter:
    def __init__(self) -> None:
        self.index = 0
        self.calls = 0

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.calls += 1
        self.index += 1
        if self.index <= 4:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.read_file","arguments":'
                    f'{{"path":"f{self.index}.txt"}},"call_id":"r{self.index}"}}'
                ),
                raw_model_name=model_id,
            )
        return CompletionResult(
            text=(
                '{"type":"final","status":"blocked",'
                '"content":"Остановлено механическим progress gate.",'
                '"evidence":["r4"]}'
            ),
            raw_model_name=model_id,
        )


def test_read_only_loop_is_stopped_before_global_step_ceiling(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    for index in range(1, 5):
        workspace.write_file(f"f{index}.txt", str(index))
    router = ReadLoopRouter()
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=router,
        workspace=workspace,
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    answer = agent.ask("Проверь файлы и исправь проблему", model_id="test:model", force_task=True)
    assert "progress gate" in answer
    assert router.calls == 5
    assert agent.incident.state is not None
    assert agent.incident.state.total_llm_rounds == 5
    assert agent.incident.state.total_llm_rounds < 60
