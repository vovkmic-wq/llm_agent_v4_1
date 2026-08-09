import os
from pathlib import Path

from core.workspace import WorkspaceManager


def test_workspace_can_use_relative_default_without_env(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(agent_dir)
    monkeypatch.delenv("AGENT_WORKSPACE_DIR", raising=False)
    manager = WorkspaceManager.from_environment(
        root_env="AGENT_WORKSPACE_DIR",
        default_root="../AGENT_WORKSPACE",
    )
    assert manager.root == (tmp_path / "AGENT_WORKSPACE").resolve()
    assert manager.root.is_dir()


def test_workspace_env_can_be_relative(tmp_path: Path, monkeypatch) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    monkeypatch.chdir(agent_dir)
    monkeypatch.setenv("AGENT_WORKSPACE_DIR", "workspace")
    manager = WorkspaceManager.from_environment(root_env="AGENT_WORKSPACE_DIR")
    assert manager.root == (agent_dir / "workspace").resolve()
