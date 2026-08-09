import json
from pathlib import Path

import pytest

from core.workspace import WorkspaceError, WorkspaceManager, WorkspaceSecurityError


def make_workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        tmp_path / "workspace",
        protected_globs=[".env", ".env.*", "*.key", ".git/**"],
    )


def test_structured_json_patch_accumulates_values(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_json(
        "settings.json",
        {"periods": ["day"], "methods": ["reviews"]},
        overwrite=False,
    )
    workspace.patch_json(
        "settings.json",
        [
            {"op": "append_unique", "path": "/periods", "value": "week"},
            {"op": "append_unique", "path": "/periods", "value": "week"},
            {
                "op": "extend_unique",
                "path": "/methods",
                "value": ["position", "rating", "reviews"],
            },
        ],
    )
    data = workspace.read_json("settings.json")["data"]
    assert data["periods"] == ["day", "week"]
    assert data["methods"] == ["reviews", "position", "rating"]


def test_structured_yaml_merge_and_patch(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_yaml("config.yaml", {"agent": {"name": "v1", "tags": ["a"]}}, False)
    workspace.merge_yaml("config.yaml", {"agent": {"name": "v2"}})
    workspace.patch_yaml(
        "config.yaml",
        [{"op": "append_unique", "path": "/agent/tags", "value": "b"}],
    )
    data = workspace.read_yaml("config.yaml")["data"]
    assert data == {"agent": {"name": "v2", "tags": ["a", "b"]}}


def test_invalid_python_write_is_rejected_without_corrupting_file(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_file("app.py", "value = 1\n")
    before = workspace.read_file("app.py")["sha256"]
    with pytest.raises(WorkspaceError):
        workspace.write_file("app.py", "def broken(:\n", overwrite=True)
    assert workspace.read_file("app.py")["sha256"] == before


def test_invalid_json_write_is_rejected(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.write_file("bad.json", "{broken", overwrite=False)
    assert not (workspace.root / "bad.json").exists()


def test_apply_patch_verifies_result(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_file("a.py", "x = 1\n")
    result = workspace.apply_patch(
        "a.py",
        [{"op": "replace", "old_text": "x = 1", "new_text": "x = 2", "count": 1}],
    )
    assert result["verified"] is True
    assert workspace.read_file("a.py")["content"] == "x = 2\n"


@pytest.mark.parametrize("path", ["../escape.txt", "../../x", "/tmp/absolute.txt"])
def test_path_escape_blocked(tmp_path: Path, path: str) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        workspace.write_file(path, "blocked")


def test_secret_paths_blocked(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    for path in [".env", ".env.local", "private.key"]:
        with pytest.raises(WorkspaceSecurityError):
            workspace.write_file(path, "secret")


def test_symlink_escape_blocked(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace.root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink unavailable")
    with pytest.raises(WorkspaceSecurityError):
        workspace.write_file("link/escape.txt", "blocked")


def test_read_json_is_real_json(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_json("a.json", {"a": 1}, False)
    raw = workspace.read_file("a.json")["content"]
    assert json.loads(raw) == {"a": 1}


def test_recursive_listing_prunes_virtualenv_and_build_dirs(tmp_path: Path) -> None:
    workspace = WorkspaceManager(
        tmp_path / "workspace",
        protected_globs=[".venv", ".venv/*", ".venv/**"],
        ignored_globs=["build", "build/*", "build/**"],
    )
    workspace.make_directory("src")
    workspace.write_file("src/app.py", "print('ok')\n")
    (workspace.root / ".venv" / "Lib").mkdir(parents=True)
    (workspace.root / ".venv" / "Lib" / "huge.py").write_text("x", encoding="utf-8")
    (workspace.root / "build").mkdir()
    (workspace.root / "build" / "artifact.py").write_text("x", encoding="utf-8")
    listing = workspace.list_files(".", recursive=True)
    paths = {item["path"] for item in listing["entries"]}
    assert "src/app.py" in paths
    assert not any(path.startswith(".venv") for path in paths)
    assert not any(path.startswith("build") for path in paths)
