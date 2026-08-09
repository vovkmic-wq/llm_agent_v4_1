from pathlib import Path

from core.task_state import TaskStateManager


def test_task_state_persists_plan_and_evidence(tmp_path: Path) -> None:
    path = tmp_path / "task_state.json"
    manager = TaskStateManager(path)
    manager.start("создать проект")
    manager.set_plan(["прочитать", "изменить"], ["tests pass"])
    manager.record_tool(call_id="c1", tool="workspace.write_file", ok=True, changed_path="a.py")
    manager.finish(True)

    restored = TaskStateManager(path).current
    assert restored is not None
    assert restored.status == "done"
    assert restored.successful_evidence == ["c1"]
    assert restored.files_changed == ["a.py"]
    assert len(restored.plan) == 2
