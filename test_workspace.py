from pathlib import Path

import pytest

from core.workspace import WorkspaceError, WorkspaceManager, WorkspaceSecurityError


def make_workspace(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(
        tmp_path / "workspace",
        protected_globs=[".env", ".env.*", "*.key", ".git/**"],
    )


def test_create_read_edit_list_delete(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)

    created = workspace.write_file("src/app.py", "print('old')\n")
    assert created["bytes_written"] > 0
    assert workspace.read_file("src/app.py")["content"] == "print('old')\n"

    edited = workspace.replace_text("src/app.py", "old", "new")
    assert edited["replacements"] == 1
    assert workspace.read_file("src/app.py")["content"] == "print('new')\n"

    listing = workspace.list_files(".", recursive=True)
    assert any(item["path"] == "src/app.py" for item in listing["entries"])

    deleted = workspace.delete_file("src/app.py")
    assert deleted["deleted"] is True
    assert not (workspace.root / "src/app.py").exists()


def test_overwrite_requires_explicit_flag(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    workspace.write_file("a.txt", "one")
    with pytest.raises(WorkspaceError):
        workspace.write_file("a.txt", "two")
    workspace.write_file("a.txt", "two", overwrite=True)
    assert workspace.read_file("a.txt")["content"] == "two"


@pytest.mark.parametrize("path", ["../escape.txt", "../../x", "/tmp/absolute.txt"])
def test_path_escape_is_blocked(tmp_path: Path, path: str) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        workspace.write_file(path, "blocked")


def test_protected_secrets_are_blocked(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    for path in (".env", ".env.local", "private.key"):
        with pytest.raises(WorkspaceSecurityError):
            workspace.write_file(path, "secret")


def test_unknown_tool_is_rejected(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    with pytest.raises(WorkspaceError):
        workspace.execute("workspace.shell", {"command": "whoami"})


def test_symlink_escape_is_blocked(tmp_path: Path) -> None:
    workspace = make_workspace(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace.root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are unavailable on this platform")

    with pytest.raises(WorkspaceSecurityError):
        workspace.write_file("link/escape.txt", "blocked")
