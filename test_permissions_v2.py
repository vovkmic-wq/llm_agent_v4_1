from core.permissions import PermissionManager


def test_permission_precedence_and_confirmation() -> None:
    calls: list[str] = []

    def confirm(tool: str, arguments: dict) -> bool:
        calls.append(tool)
        return arguments.get("path") == "ok.txt"

    manager = PermissionManager(
        {
            "workspace.*": "auto",
            "workspace.delete_file": "confirm",
            "execution.*": "deny",
        },
        confirmation_handler=confirm,
    )
    assert manager.authorize("workspace.write_file", {"path": "a.txt"}) == (True, None)
    assert manager.authorize("workspace.delete_file", {"path": "ok.txt"})[0] is True
    assert manager.authorize("workspace.delete_file", {"path": "no.txt"})[0] is False
    assert manager.authorize("execution.run_python", {"path": "x.py"})[0] is False
    assert calls == ["workspace.delete_file", "workspace.delete_file"]
