from pathlib import Path

import pytest

from core.execution import ExecutionError, ExecutionManager
from core.workspace import WorkspaceManager


@pytest.mark.subprocess
def test_terminal_execute_runs_pytest_in_project_cwd(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("project")
    workspace.write_file("project/test_ok.py", "def test_ok():\n    assert 3 * 3 == 9\n")
    runner = ExecutionManager(
        workspace,
        enabled=True,
        timeout_seconds=30,
        isolate_pytest_plugins=True,
    )
    result = runner.terminal_execute("pytest -q", cwd="project")
    assert result["ok"] is True
    assert result["cwd"] == "project"
    assert "passed" in result["output"]


def test_terminal_execute_rejects_general_shell(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    runner = ExecutionManager(workspace, enabled=True)
    with pytest.raises(ExecutionError):
        runner.terminal_execute("powershell Remove-Item important.txt")


def test_run_python_resolves_script_relative_to_active_cwd(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("project")
    workspace.write_file("project/main.py", "print('PROJECT_OK')\n")
    runner = ExecutionManager(workspace, enabled=True)
    result = runner.run_python("main.py", cwd="project")
    assert result["ok"] is True
    assert "PROJECT_OK" in result["output"]


def test_runner_prefers_project_virtualenv_python(tmp_path: Path) -> None:
    import os
    import sys

    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("project/.venv/bin")
    project_python = workspace.root / "project/.venv/bin/python"
    try:
        os.symlink(sys.executable, project_python)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    workspace.write_file("project/main.py", "print('VENV_OK')\n")
    runner = ExecutionManager(workspace, enabled=True)
    result = runner.run_python("main.py", cwd="project")
    assert result["ok"] is True
    assert result["command"][0] == str(project_python)
