import os
from pathlib import Path

import pytest

from core.execution import ExecutionManager
from core.workspace import WorkspaceManager


def test_run_python_uses_workspace_and_scrubs_secrets(tmp_path: Path, monkeypatch) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file(
        "probe.py",
        "import os\nprint(os.getcwd())\nprint(os.getenv('VERY_SECRET_API_KEY', 'MISSING'))\n",
    )
    monkeypatch.setenv("VERY_SECRET_API_KEY", "do-not-leak")
    runner = ExecutionManager(workspace, enabled=True, timeout_seconds=10)
    result = runner.run_python("probe.py")
    assert result["ok"] is True
    assert str(workspace.root) in result["output"]
    assert "MISSING" in result["output"]
    assert "do-not-leak" not in result["output"]


@pytest.mark.subprocess
def test_pytest_runner_executes_tests(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.write_file("test_sample.py", "def test_ok():\n    assert 2 + 2 == 4\n")
    runner = ExecutionManager(
        workspace,
        enabled=True,
        timeout_seconds=30,
        isolate_pytest_plugins=True,
    )
    result = runner.run_pytest(paths=["test_sample.py"])
    assert result["exit_code"] == 0
    assert result["ok"] is True
