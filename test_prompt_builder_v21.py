from pathlib import Path

from core.config import settings
from core.memory import MemoryManager
from core.prompt_builder import PromptBuilder


def test_prompt_builder_enforces_hard_budget_and_drops_old_history(
    tmp_path: Path, monkeypatch
) -> None:
    memory = MemoryManager(tmp_path / "memory")
    for index in range(12):
        memory.append_exchange(
            user_query=f"old question {index} " + "q" * 500,
            assistant_response="a" * 1000,
            model_used="test",
        )
    memory.overwrite_summary("summary " + "s" * 3000)
    monkeypatch.setattr(settings.agent.memory, "max_context_tokens", 1000)
    monkeypatch.setattr(settings.agent.memory, "last_messages", 12)
    builder = PromptBuilder(memory)
    built = builder.build(
        "current task",
        project_context="IMPORTANT_SPEC " + "x" * 5000,
        extra_system="TOOLS " + "t" * 3000,
    )
    assert built.estimated_tokens <= 1000
    assert built.dropped_history_exchanges > 0
    assert "IMPORTANT_SPEC" in built.system_prompt
