"""Ядерная политика разрешений: LLM не решает, когда спрашивать пользователя."""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

PermissionMode = Literal["auto", "confirm", "deny"]
ConfirmationHandler = Callable[[str, dict], bool]


class PermissionManager:
    def __init__(
        self,
        rules: dict[str, PermissionMode] | None = None,
        confirmation_handler: ConfirmationHandler | None = None,
    ) -> None:
        self.rules = rules or {}
        self.confirmation_handler = confirmation_handler

    def mode_for(self, tool_name: str) -> PermissionMode:
        if tool_name in self.rules:
            return self.rules[tool_name]
        category = tool_name.split(".", 1)[0] + ".*"
        return self.rules.get(category, "deny")

    def authorize(self, tool_name: str, arguments: dict) -> tuple[bool, str | None]:
        mode = self.mode_for(tool_name)
        if mode == "auto":
            return True, None
        if mode == "deny":
            return False, f"Инструмент запрещён политикой: {tool_name}"
        if self.confirmation_handler is None:
            return False, f"Для {tool_name} требуется подтверждение пользователя"
        if self.confirmation_handler(tool_name, arguments):
            return True, None
        return False, f"Пользователь отклонил выполнение {tool_name}"
