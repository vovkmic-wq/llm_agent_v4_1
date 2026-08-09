from pathlib import Path

import pytest

from core.execution import ExecutionError, ExecutionManager
from core.workspace import WorkspaceManager


def _manager(tmp_path: Path) -> ExecutionManager:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("app")
    workspace.write_file("app/test_ok.py", "def test_ok():\n    assert True\n")
    return ExecutionManager(workspace, enabled=True, allow_terminal_compat=True)


@pytest.mark.subprocess
def test_terminal_pytest_allows_common_safe_flags(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    result = manager.terminal_execute("pytest -q test_ok.py", cwd="app")
    assert result["ok"] is True


def test_terminal_pytest_rejects_unknown_option(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(ExecutionError):
        manager.terminal_execute("pytest --rootdir=/tmp", cwd="app")


def test_terminal_pytest_rejects_path_outside_workspace(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(ExecutionError):
        manager.terminal_execute("pytest ../../outside.py", cwd="app")


def test_terminal_compileall_rejects_external_path(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(ExecutionError):
        manager.terminal_execute("python -m compileall ../../outside", cwd="app")
