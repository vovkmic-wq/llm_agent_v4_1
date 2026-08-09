"""Unified registry of kernel-controlled tools."""
from __future__ import annotations

import json
from typing import Any

from core.execution import ExecutionError, ExecutionManager
from core.research import ResearchError, ResearchManager
from core.workspace import WorkspaceError, WorkspaceManager


class ToolRegistryError(RuntimeError):
    """Tool routing/execution error."""


class ToolRegistry:
    def __init__(
        self,
        workspace: WorkspaceManager,
        execution: ExecutionManager | None = None,
        research: ResearchManager | None = None,
    ) -> None:
        self.workspace = workspace
        self.execution = execution
        self.research = research

    def specs(self) -> list[dict[str, Any]]:
        specs = self.workspace.tool_specs()
        if self.execution is not None:
            specs.extend(self.execution.tool_specs())
        if self.research is not None:
            specs.extend(self.research.tool_specs())
        return specs

    def instructions(self) -> str:
        specs = json.dumps(self.specs(), ensure_ascii=False, separators=(",", ":"))
        return f"""
=== TOOL PROTOCOL V3 ===
Для действия верни небольшой JSON-envelope без пояснений:
{{"type":"tool","tool":"workspace.read_file",
 "arguments":{{"path":"src/app.py"}},"call_id":"read-app"}}

Для больших фрагментов кода НЕ помещай код в JSON. Используй raw PAYLOAD:
{{"type":"tool","tool":"workspace.replace_lines",
 "arguments":{{"path":"src/app.py","start_line":10,"end_line":25,
 "expected_sha256":"sha-from-read"}},"call_id":"patch-app",
 "payload_id":"patch1","payload_argument":"content"}}
<<<PAYLOAD:patch1>>>
raw Python code; кавычки и переносы не нужно JSON-экранировать
<<<END_PAYLOAD:patch1>>>

Для нового большого файла аналогично используй workspace.write_file +
payload_argument="content". Один PAYLOAD должен быть локальным и по возможности
не больше ~6000 символов. Большой рефакторинг разбивай на несколько небольших patch.

Финальный ответ:
{{"type":"final","status":"success","content":"...",
 "evidence":["успешный-call-id"]}}
Если проверка невозможна из-за инфраструктуры (например, отсутствует pytest),
используй status="blocked"/"partial" и приложи blocker call_id в evidence.

Обязательные правила:
- Не печатай tool-call как пример, если хочешь выполнить действие.
- Не утверждай об изменении файла/успехе тестов без реального tool_result.
- Для существующего файла сначала read_file/read_json/read_yaml, затем изменение.
- Для Python предпочитай replace_lines с expected_sha256 или небольшой apply_patch.
- Для JSON/YAML используй structured patch/merge tools.
- Если testing/build task: сам запускай execution.run_compileall/pytest/ruff/mypy.
- Если тест упал: прочитай относящийся файл, внеси минимальный patch и запусти снова.
- Не проси пользователя запускать команды, если execution включён.
- Никогда не используй произвольный shell; terminal.execute — ограниченный adapter.
- Если tool_result неуспешен, считай действие НЕ выполненным.

Доступные инструменты:
{specs}
""".strip()

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if tool_name.startswith("workspace."):
                return self.workspace.execute(tool_name, arguments)
            if tool_name.startswith("execution."):
                if self.execution is None:
                    raise ToolRegistryError("Execution tools не настроены")
                return self.execution.execute(tool_name, arguments)
            if tool_name == "terminal.execute":
                if self.execution is None:
                    raise ToolRegistryError("Execution tools не настроены")
                return self.execution.execute(tool_name, arguments)
            if tool_name.startswith("research."):
                if self.research is None:
                    raise ToolRegistryError("Research tools не настроены")
                return self.research.execute(tool_name, arguments)
            raise ToolRegistryError(f"Неизвестный инструмент: {tool_name}")
        except (WorkspaceError, ExecutionError, ResearchError) as exc:
            raise ToolRegistryError(str(exc)) from exc
