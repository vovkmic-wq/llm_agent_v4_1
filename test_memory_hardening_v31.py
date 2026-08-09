from pathlib import Path

from core.memory import MemoryManager


def test_corrupt_history_line_does_not_break_following_messages(tmp_path: Path) -> None:
    memory = MemoryManager(tmp_path / "memory")
    memory.append_exchange("one", "answer one", "test")
    with memory.history_file.open("a", encoding="utf-8") as stream:
        stream.write("{broken json\n")
    memory.append_exchange("two", "answer two", "test")
    assert [item.user_query for item in memory.all_messages()] == ["one", "two"]


def test_summary_checkpoint_is_persistent(tmp_path: Path) -> None:
    memory = MemoryManager(tmp_path / "memory")
    for index in range(3):
        memory.append_exchange(str(index), "answer", "test")
    memory.mark_summary_checkpoint(2)
    reloaded = MemoryManager(tmp_path / "memory")
    assert reloaded.summary_checkpoint() == 2
    assert [item.user_query for item in reloaded.messages_from(2)] == ["2"]
