from pathlib import Path
from types import SimpleNamespace

from core.project_context import ProjectContextManager
from core.workspace import WorkspaceManager
from main import _print_project, _print_specification


def _fake_agent(tmp_path: Path):
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("demo")
    return SimpleNamespace(project_context=manager), workspace


def test_spec_show_helper_auto_discovers_late_spec(tmp_path: Path, capsys) -> None:
    agent, workspace = _fake_agent(tmp_path)
    workspace.write_file(
        "demo/TECHNICAL_SPECIFICATION.md",
        "# Contract\n\nUse SQLite.\n",
    )

    _print_specification(agent)
    output = capsys.readouterr().out

    assert "workspace:demo/TECHNICAL_SPECIFICATION.md" in output
    assert "sha256=" in output
    assert "Use SQLite" in output


def test_spec_show_helper_reports_missing_contract(tmp_path: Path, capsys) -> None:
    agent, _ = _fake_agent(tmp_path)

    _print_specification(agent)
    output = capsys.readouterr().out

    assert "ТЗ не найдено" in output


def test_project_show_helper_refreshes_spec_metadata(tmp_path: Path, capsys) -> None:
    agent, workspace = _fake_agent(tmp_path)
    workspace.write_file("demo/TECHNICAL_SPECIFICATION.md", "# Contract\n")

    _print_project(agent)
    output = capsys.readouterr().out

    assert '"active_project": "demo"' in output
    assert '"specification_source": "workspace:demo/TECHNICAL_SPECIFICATION.md"' in output
