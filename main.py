"""CLI for LLM Coding Agent V4.1 ICS."""
from __future__ import annotations

import json
import sys

from core.agent import Agent, AgentExecutionError
from core.config import CONFIG_DIR, ENV_FILE, STATE_DIR, settings
from core.project_context import ProjectContextError
from core.router import clear_provider_cache, resolve_model_id
from core.workspace import WorkspaceError
from providers.base import ProviderError


def confirm_tool(tool_name: str, arguments: dict) -> bool:
    safe_args = dict(arguments)
    for key in list(safe_args):
        if key.lower() in {"content", "old_text", "new_text", "text"}:
            safe_args[key] = f"<скрыто, {len(str(safe_args[key]))} символов>"
    print(f"\n[Требуется подтверждение] {tool_name}")
    print(json.dumps(safe_args, ensure_ascii=False, indent=2))
    answer = input("Разрешить? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "д", "да"}


def _sync_specification(agent: Agent):
    """Refresh the active project specification before presenting or using it."""
    return agent.project_context.auto_sync_specification()


def _print_project(agent: Agent) -> None:
    _sync_specification(agent)
    print(json.dumps(agent.project_context.describe(), ensure_ascii=False, indent=2))


def _print_specification(agent: Agent) -> None:
    snapshot = _sync_specification(agent)
    if snapshot is None:
        print(
            "[ТЗ не найдено ни в project context, ни в активном проекте. "
            "Ожидается один из стандартных файлов ТЗ или используйте /spec load <path>.]"
        )
        return
    print(
        f"[source={snapshot.source}; sha256={snapshot.sha256}; "
        f"chars={len(snapshot.text)}]\n{snapshot.text}"
    )


def main() -> None:
    try:
        agent = Agent(confirmation_handler=confirm_tool)
    except WorkspaceError as exc:
        print(f"[Ошибка конфигурации workspace] {exc}")
        print(
            "Проверьте доступность workspace. По умолчанию используется "
            "../AGENT_WORKSPACE; AGENT_WORKSPACE_DIR в .env необязателен."
        )
        sys.exit(2)

    pending_model: str | None = None
    pending_tags: list[str] = []
    spec_mode = False
    spec_buffer: list[str] = []

    print(f"=== {settings.agent.name} ===")
    print(f"Workspace: {agent.workspace.root}")
    print(f"Active project: {agent.project_context.active_project}")
    print(f"Code execution: {'ON' if agent.execution.enabled else 'OFF'}")
    print(f"Research: {'ON' if agent.research.enabled else 'OFF'}")
    print(
        "Команды:\n"
        "  /project set <path> | /project show\n"
        "  /spec begin | /spec end | /spec load <path> | /spec show | /spec clear | /spec run\n"
        "  /context | /context add <fact> | /decision add <text>\n"
        "  /execution on|off | /task <text> | /model <provider:alias> | /tags <a,b>\n"
        "  /workspace | /state | /incident | /incident approve | /resume [note] "
        "| /doctor | /reload | /exit\n"
    )

    while True:
        try:
            user_input = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение.")
            return

        if not user_input:
            continue
        if user_input == "/exit":
            print("Завершение.")
            return
        if user_input == "/reload":
            execution_enabled = agent.execution.enabled
            research_enabled = agent.research.enabled
            settings.reload()
            clear_provider_cache()
            try:
                agent = Agent(
                    confirmation_handler=confirm_tool,
                    execution_enabled=execution_enabled,
                    research_enabled=research_enabled,
                )
            except WorkspaceError as exc:
                print(f"[Ошибка reload: {exc}]")
                continue
            print(
                "[YAML-конфиги и компоненты агента перечитаны; "
                ".env требует перезапуска процесса]"
            )
            continue
        if user_input == "/workspace":
            print(json.dumps(agent.workspace.describe(), ensure_ascii=False, indent=2))
            continue
        if user_input == "/state":
            state = agent.task_state.current
            print(state.model_dump_json(indent=2) if state else "[активной задачи нет]")
            continue
        if user_input == "/incident":
            state = agent.incident.state
            if state is None:
                print("[активного инцидента нет]")
            else:
                print(state.model_dump_json(indent=2))
                print(f"incident_dir={agent.incident.path}")
            continue
        if user_input == "/incident approve":
            state = agent.incident.state
            if state is None:
                print("[активного инцидента нет]")
                continue
            try:
                digest = agent.incident.approve_iap(actor="human-owner")
                print(f"[IAP approved: sha256={digest}]")
            except (OSError, RuntimeError) as exc:
                print(f"[Ошибка approval] {exc}")
            continue
        if user_input == "/resume" or user_input.startswith("/resume "):
            note = user_input.removeprefix("/resume").strip()
            try:
                response = agent.resume(
                    note,
                    tags=pending_tags,
                    model_id=pending_model,
                )
                print(f"\nАгент: {response}\n")
            except (ProviderError, AgentExecutionError) as exc:
                print(f"\n[Ошибка] {exc}\n")
            finally:
                pending_model = None
                pending_tags = []
            continue
        if user_input == "/doctor":
            providers = {}
            for name, entry in settings.models.providers.items():
                configured = False
                error: str | None = None
                if entry.enabled and entry.models:
                    alias = next(iter(entry.models))
                    try:
                        provider, _ = resolve_model_id(f"{name}:{alias}")
                        configured = provider.is_configured()
                    except ProviderError as exc:
                        error = str(exc)
                providers[name] = {
                    "enabled": entry.enabled,
                    "configured": configured,
                    "error": error,
                }
            print(
                json.dumps(
                    {
                        "python": sys.executable,
                        "env_file": str(ENV_FILE),
                        "config_dir": str(CONFIG_DIR),
                        "state_dir": str(STATE_DIR),
                        "workspace": str(agent.workspace.root),
                        "active_project": agent.project_context.active_project,
                        "project_context": agent.project_context.describe(),
                        "execution_enabled": agent.execution.enabled,
                        "research_enabled": agent.research.enabled,
                        "providers": providers,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            continue
        if user_input == "/project show":
            _print_project(agent)
            continue
        if user_input.startswith("/project set "):
            path = user_input.removeprefix("/project set ").strip()
            try:
                selected = agent.project_context.set_active_project(path)
                print(f"[Активный проект: {selected}]")
                _print_project(agent)
            except (ProjectContextError, WorkspaceError) as exc:
                print(f"[Ошибка project context] {exc}")
            continue
        if user_input == "/context":
            _print_project(agent)
            print("\nLegacy memory/context.md:\n" + (agent.memory.load_context() or "(пусто)"))
            continue
        if user_input.startswith("/context add "):
            agent.project_context.add_context_fact(user_input.removeprefix("/context add "))
            print("[Факт проекта сохранён]")
            continue
        if user_input.startswith("/decision add "):
            agent.project_context.add_decision(user_input.removeprefix("/decision add "))
            print("[Решение проекта сохранено]")
            continue
        if user_input == "/execution on":
            agent.execution.enabled = True
            print("[Выполнение кода включено для текущего процесса]")
            continue
        if user_input == "/execution off":
            agent.execution.enabled = False
            print("[Выполнение кода выключено для текущего процесса]")
            continue
        if user_input.startswith("/model "):
            pending_model = user_input.removeprefix("/model ").strip()
            print(f"[модель следующего запроса: {pending_model}]")
            continue
        if user_input.startswith("/tags "):
            pending_tags = [
                tag.strip()
                for tag in user_input.removeprefix("/tags ").split(",")
                if tag.strip()
            ]
            print(f"[теги следующего запроса: {pending_tags}]")
            continue

        if user_input == "/spec begin":
            spec_mode = True
            spec_buffer = []
            print(
                "[SPEC MODE: вставляйте ТЗ частями. /spec end сохранит его как "
                "постоянный PROJECT CONTRACT]"
            )
            continue
        if user_input == "/spec cancel":
            spec_mode = False
            spec_buffer = []
            print("[SPEC MODE отменён]")
            continue
        if user_input == "/spec end":
            if not spec_mode:
                print("[SPEC MODE не активен]")
                continue
            spec_mode = False
            specification = "\n".join(spec_buffer).strip()
            spec_buffer = []
            try:
                snapshot = agent.project_context.set_specification(specification)
                print(
                    f"[ТЗ сохранено: {len(snapshot.text)} символов, "
                    f"sha256={snapshot.sha256[:16]}…]"
                )
            except (ProjectContextError, WorkspaceError) as exc:
                print(f"[Ошибка ТЗ] {exc}")
            continue
        if user_input.startswith("/spec load "):
            path = user_input.removeprefix("/spec load ").strip()
            try:
                snapshot = agent.project_context.import_specification(path)
                print(
                    f"[ТЗ загружено из {snapshot.source}: {len(snapshot.text)} символов, "
                    f"sha256={snapshot.sha256[:16]}…]"
                )
            except (ProjectContextError, WorkspaceError) as exc:
                print(f"[Ошибка ТЗ] {exc}")
            continue
        if user_input == "/spec show":
            try:
                _print_specification(agent)
            except (ProjectContextError, WorkspaceError) as exc:
                print(f"[Ошибка ТЗ] {exc}")
            continue
        if user_input == "/spec clear":
            agent.project_context.clear_specification()
            print("[ТЗ удалено из project context]")
            continue
        if user_input == "/spec run":
            try:
                snapshot = _sync_specification(agent)
            except (ProjectContextError, WorkspaceError) as exc:
                print(f"[Ошибка ТЗ] {exc}")
                continue
            if snapshot is None:
                print(
                    "[ТЗ не найдено ни в project context, ни в активном проекте. "
                    "Используйте /spec load <path> или создайте стандартный файл ТЗ.]"
                )
                continue
            user_input = (
                "Выполни текущую задачу по активному техническому заданию. "
                "Проанализируй существующий проект, реализуй недостающее, запусти проверки "
                "и исправляй ошибки до прохождения доступных quality gates."
            )
            pending_tags = [*pending_tags, "code", "spec"]
        elif spec_mode:
            spec_buffer.append(user_input)
            print(f"[ТЗ: принято фрагментов: {len(spec_buffer)}]")
            continue

        force_task = False
        query = user_input
        if user_input.startswith("/task "):
            query = user_input.removeprefix("/task ").strip()
            force_task = True

        try:
            response = agent.ask(
                query,
                tags=pending_tags,
                model_id=pending_model,
                force_task=force_task or user_input == "/spec run",
            )
            print(f"\nАгент: {response}\n")
        except (ProviderError, AgentExecutionError) as exc:
            print(f"\n[Ошибка] {exc}\n")
        finally:
            pending_model = None
            pending_tags = []


if __name__ == "__main__":
    main()
