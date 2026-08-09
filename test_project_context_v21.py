from pathlib import Path

from core.project_context import ProjectContextManager
from core.workspace import WorkspaceManager


def test_project_context_auto_loads_spec_and_renders_contract(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("ozon_market_analytics")
    workspace.write_file(
        "ozon_market_analytics/TECHNICAL_SPECIFICATION.md",
        "# ТЗ\n\n- Использовать SQLite.\n- Все тесты pytest должны проходить.\n",
    )
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("ozon_market_analytics")
    contract = manager.render_for_prompt("проверь проект по ТЗ")
    assert "ACTIVE_PROJECT=ozon_market_analytics" in contract
    assert "Использовать SQLite" in contract
    assert "pytest" in contract
    assert manager.load_specification() is not None


def test_project_context_infers_top_level_project_from_query(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("alpha")
    workspace.make_directory("ozon_market_analytics")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    selected = manager.infer_project_from_query(
        "Проверь C:/workspace/ozon_market_analytics на работоспособность"
    )
    assert selected == "ozon_market_analytics"
    assert manager.active_project == "ozon_market_analytics"


def test_manual_spec_persists_independently_from_chat(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    snapshot = manager.set_specification("# Contract\n\nNever use CSV when SQLite is required.")
    reloaded = ProjectContextManager(tmp_path / "memory", workspace)
    loaded = reloaded.load_specification()
    assert loaded is not None
    assert loaded.sha256 == snapshot.sha256
    assert "SQLite" in loaded.text


def test_auto_sync_discovers_spec_created_after_project_selection(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("demo")
    assert manager.load_specification() is None

    workspace.write_file(
        "demo/TECHNICAL_SPECIFICATION.md",
        "# Late contract\n\n- Requirement A.\n",
    )

    snapshot = manager.auto_sync_specification()
    assert snapshot is not None
    assert snapshot.source == "workspace:demo/TECHNICAL_SPECIFICATION.md"
    assert "Requirement A" in snapshot.text


def test_auto_sync_refreshes_changed_conventional_spec(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo")
    workspace.write_file("demo/TECHNICAL_SPECIFICATION.md", "# Contract\n\nVersion one.\n")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("demo")
    first = manager.load_specification()
    assert first is not None

    workspace.write_file(
        "demo/TECHNICAL_SPECIFICATION.md",
        "# Contract\n\nVersion two.\n",
        overwrite=True,
    )
    second = manager.auto_sync_specification()
    assert second is not None
    assert second.sha256 != first.sha256
    assert "Version two" in second.text


def test_auto_sync_removes_stale_auto_discovered_spec(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo")
    workspace.write_file("demo/TECHNICAL_SPECIFICATION.md", "# Contract\n")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("demo")
    assert manager.load_specification() is not None

    workspace.delete_file("demo/TECHNICAL_SPECIFICATION.md")

    assert manager.auto_sync_specification() is None
    assert manager.load_specification() is None


def test_auto_sync_refreshes_explicit_workspace_spec(tmp_path: Path) -> None:
    workspace = WorkspaceManager(tmp_path / "workspace")
    workspace.make_directory("demo")
    workspace.make_directory("demo/docs")
    workspace.write_file("demo/docs/project_contract.md", "# Contract\n\nVersion one.\n")
    manager = ProjectContextManager(tmp_path / "memory", workspace)
    manager.set_active_project("demo")
    first = manager.import_specification("demo/docs/project_contract.md")

    workspace.write_file(
        "demo/docs/project_contract.md",
        "# Contract\n\nVersion two.\n",
        overwrite=True,
    )
    second = manager.auto_sync_specification()

    assert second is not None
    assert second.source == "workspace:demo/docs/project_contract.md"
    assert second.sha256 != first.sha256
    assert "Version two" in second.text
