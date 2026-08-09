"""Controlled Python/test runners without ``shell=True``.

This is intentionally not advertised as an OS sandbox: executed Python inherits the
current user's OS permissions. The kernel therefore restricts commands, cwd, timeout,
output size and environment, and execution has a separate opt-in flag.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from core.workspace import WorkspaceError, WorkspaceManager


class ExecutionError(RuntimeError):
    """Execution policy or subprocess failure."""


class ExecutionManager:
    COMPILEALL_EXCLUDE_RE = (
        r"(^|[\\/])(\.venv|venv|\.git|__pycache__|\.pytest_cache|"
        r"\.mypy_cache|\.ruff_cache|\.tox|\.nox|node_modules|build|dist)([\\/]|$)"
    )

    def __init__(
        self,
        workspace: WorkspaceManager,
        *,
        enabled: bool,
        timeout_seconds: int = 120,
        max_output_chars: int = 40_000,
        allow_terminal_compat: bool = True,
        isolate_pytest_plugins: bool = False,
    ) -> None:
        self.workspace = workspace
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.allow_terminal_compat = allow_terminal_compat
        self.isolate_pytest_plugins = isolate_pytest_plugins

    @staticmethod
    def _scrubbed_environment() -> dict[str, str]:
        keep = {
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "HOME",
            "USERPROFILE",
            "LANG",
            "LC_ALL",
            "PYTHONIOENCODING",
        }
        env = {key: value for key, value in os.environ.items() if key in keep}
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        return env

    def _environment_for_command(self, command: list[str]) -> dict[str, str]:
        env = self._scrubbed_environment()
        if not command:
            return env
        executable = Path(command[0])
        if not executable.is_absolute():
            return env
        parent = executable.parent
        if parent.name.casefold() in {"scripts", "bin"}:
            venv_root = parent.parent
            if venv_root.name.casefold() in {".venv", "venv"}:
                env["VIRTUAL_ENV"] = str(venv_root)
                current_path = env.get("PATH", "")
                env["PATH"] = str(parent) + (os.pathsep + current_path if current_path else "")
        return env

    def _ensure_enabled(self) -> None:
        if not self.enabled:
            raise ExecutionError(
                "Исполнение кода отключено. Установите AGENT_ALLOW_CODE_EXECUTION=true."
            )

    def _resolve_cwd(self, cwd: str | None) -> Path:
        target = self.workspace.resolve_path(cwd or ".", allow_root=True)
        if not target.exists() or not target.is_dir():
            raise ExecutionError(f"Рабочий каталог не существует: {cwd or '.'}")
        return target

    @staticmethod
    def _python_for_cwd(cwd_path: Path) -> str:
        candidates = (
            cwd_path / ".venv" / "Scripts" / "python.exe",
            cwd_path / ".venv" / "bin" / "python",
            cwd_path / "venv" / "Scripts" / "python.exe",
            cwd_path / "venv" / "bin" / "python",
        )
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        return sys.executable

    def _resolve_path_for_cwd(
        self,
        path: str,
        cwd: str | None,
        *,
        suffix: str | None = None,
        must_exist: bool = True,
    ) -> tuple[Path, Path]:
        cwd_path = self._resolve_cwd(cwd)
        raw = Path(path)
        # Tool paths are workspace-relative by default. If a bare relative path exists
        # in cwd, prefer it, which is convenient for an active project.
        candidate_from_cwd = cwd_path / raw
        if not raw.is_absolute() and candidate_from_cwd.exists():
            target = candidate_from_cwd.resolve(strict=False)
            try:
                target.relative_to(self.workspace.root)
            except ValueError as exc:
                raise ExecutionError("Путь вышел за пределы workspace") from exc
        else:
            target = self.workspace.resolve_path(path)
        if must_exist and not target.exists():
            raise ExecutionError(f"Путь не существует: {path}")
        if suffix and target.suffix.lower() != suffix:
            raise ExecutionError(f"Ожидался файл {suffix}: {path}")
        return cwd_path, target

    def _run(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_enabled()
        cwd_path = self._resolve_cwd(cwd)
        requested_timeout = (
            self.timeout_seconds if timeout_seconds is None else int(timeout_seconds)
        )
        timeout = max(1, min(requested_timeout, self.timeout_seconds))
        start = time.monotonic()
        try:
            env = self._environment_for_command(command)
            if extra_env:
                env.update(extra_env)
            completed = subprocess.run(
                command,
                cwd=cwd_path,
                env=env,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExecutionError(f"Процесс превысил timeout {timeout} сек") from exc
        except OSError as exc:
            raise ExecutionError(f"Не удалось запустить процесс: {exc}") from exc
        output = (completed.stdout or "") + (completed.stderr or "")
        truncated = len(output) > self.max_output_chars
        if truncated:
            output = output[: self.max_output_chars] + "\n...[output truncated]"
        return {
            "command": command,
            "cwd": cwd_path.relative_to(self.workspace.root).as_posix() or ".",
            "exit_code": completed.returncode,
            "ok": completed.returncode == 0,
            "duration_seconds": round(time.monotonic() - start, 3),
            "output": output,
            "truncated": truncated,
        }

    def run_python(
        self,
        path: str,
        args: list[str] | None = None,
        timeout_seconds: int | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        cwd_path, target = self._resolve_path_for_cwd(path, cwd, suffix=".py")
        safe_args = [str(item) for item in (args or [])]
        if any(len(item) > 1000 for item in safe_args):
            raise ExecutionError("Аргумент run_python слишком длинный")
        relative = os.path.relpath(target, cwd_path)
        python_executable = self._python_for_cwd(cwd_path)
        return self._run(
            [python_executable, relative, *safe_args],
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )

    def run_compileall(self, path: str = ".", cwd: str | None = None) -> dict[str, Any]:
        cwd_path, target = self._resolve_path_for_cwd(path, cwd)
        relative = os.path.relpath(target, cwd_path)
        python_executable = self._python_for_cwd(cwd_path)
        return self._run(
            [
                python_executable,
                "-m",
                "compileall",
                "-q",
                "-x",
                self.COMPILEALL_EXCLUDE_RE,
                relative,
            ],
            cwd=cwd,
        )

    def _paths_for_command(self, paths: list[str] | None, cwd: str | None) -> list[str]:
        if not paths:
            return []
        cwd_path = self._resolve_cwd(cwd)
        result: list[str] = []
        for path in paths:
            _, target = self._resolve_path_for_cwd(path, cwd)
            result.append(os.path.relpath(target, cwd_path))
        return result

    def _safe_selector_path(self, token: str, cwd: str | None) -> str:
        """Validate a pytest-style path/selector and return a cwd-relative token."""
        path_part, separator, selector = token.partition("::")
        try:
            cwd_path, target = self._resolve_path_for_cwd(path_part or ".", cwd)
        except WorkspaceError as exc:
            raise ExecutionError(f"Недопустимый путь pytest: {token}") from exc
        relative = os.path.relpath(target, cwd_path)
        return relative + (separator + selector if separator else "")

    def _safe_pytest_args(self, args: list[str], cwd: str | None) -> list[str]:
        """Accept a deliberately small pytest CLI subset for terminal compatibility."""
        safe: list[str] = []
        no_value = {"-q", "-v", "-vv", "-x", "--disable-warnings", "--collect-only"}
        value_options = {"--maxfail", "--tb", "-k", "-m"}
        index = 0
        while index < len(args):
            token = args[index]
            if token in no_value:
                safe.append(token)
                index += 1
                continue
            if any(token.startswith(prefix) for prefix in ("--maxfail=", "--tb=")):
                safe.append(token)
                index += 1
                continue
            if token in value_options:
                if index + 1 >= len(args):
                    raise ExecutionError(f"Опция {token} требует значение")
                value = args[index + 1]
                if len(value) > 500:
                    raise ExecutionError(f"Значение {token} слишком длинное")
                safe.extend([token, value])
                index += 2
                continue
            if token.startswith("-"):
                raise ExecutionError(f"Опция pytest запрещена в terminal.execute: {token}")
            safe.append(self._safe_selector_path(token, cwd))
            index += 1
        return safe

    def _safe_path_tool_args(
        self,
        args: list[str],
        cwd: str | None,
        *,
        command: str,
        allowed_flags: set[str],
        allow_check_word: bool = False,
    ) -> list[str]:
        """Validate simple path-oriented ruff/mypy/compileall compatibility args."""
        safe: list[str] = []
        for index, token in enumerate(args):
            if allow_check_word and index == 0 and token == "check":
                safe.append(token)
                continue
            if token in allowed_flags:
                safe.append(token)
                continue
            if token.startswith("-"):
                raise ExecutionError(
                    f"Опция {command} запрещена в terminal.execute: {token}"
                )
            try:
                cwd_path, target = self._resolve_path_for_cwd(token, cwd)
            except WorkspaceError as exc:
                raise ExecutionError(
                    f"Недопустимый путь {command}: {token}"
                ) from exc
            safe.append(os.path.relpath(target, cwd_path))
        return safe

    def run_pytest(
        self,
        paths: list[str] | None = None,
        verbose: bool = False,
        maxfail: int = 1,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        python_executable = self._python_for_cwd(self._resolve_cwd(cwd))
        if not self._module_available(python_executable, "pytest", cwd):
            return self._unavailable_required_tool("pytest", cwd)
        command = [python_executable, "-m", "pytest", "-v" if verbose else "-q"]
        command.extend(["--maxfail", str(max(1, min(maxfail, 20)))])
        command.extend(self._paths_for_command(paths, cwd))
        extra_env = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"} if self.isolate_pytest_plugins else None
        return self._run(command, cwd=cwd, extra_env=extra_env)

    def _module_available(self, python_executable: str, module: str, cwd: str | None) -> bool:
        cwd_path = self._resolve_cwd(cwd)
        command = [
            python_executable,
            "-c",
            (
                "import importlib.util,sys;"
                f"sys.exit(0 if importlib.util.find_spec({module!r}) else 1)"
            ),
        ]
        try:
            probe = subprocess.run(
                command,
                cwd=cwd_path,
                env=self._environment_for_command(command),
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=min(self.timeout_seconds, 15),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0

    def _unavailable_required_tool(
        self,
        module: str,
        cwd: str | None,
    ) -> dict[str, Any]:
        cwd_path = self._resolve_cwd(cwd)
        return {
            "command": [module],
            "cwd": cwd_path.relative_to(self.workspace.root).as_posix() or ".",
            "exit_code": 127,
            "ok": False,
            "infrastructure_error": True,
            "reason": f"Python module '{module}' is not installed in project environment",
            "duration_seconds": 0.0,
            "output": "",
            "truncated": False,
        }

    def _skipped_optional_tool(
        self,
        module: str,
        cwd: str | None,
    ) -> dict[str, Any]:
        cwd_path = self._resolve_cwd(cwd)
        return {
            "command": [module],
            "cwd": cwd_path.relative_to(self.workspace.root).as_posix() or ".",
            "exit_code": 0,
            "ok": True,
            "skipped": True,
            "reason": f"Python module '{module}' is not installed in project environment",
            "duration_seconds": 0.0,
            "output": "",
            "truncated": False,
        }

    def run_ruff(
        self,
        paths: list[str] | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        python_executable = self._python_for_cwd(self._resolve_cwd(cwd))
        if not self._module_available(python_executable, "ruff", cwd):
            return self._skipped_optional_tool("ruff", cwd)
        command = [python_executable, "-m", "ruff", "check"]
        command.extend(self._paths_for_command(paths, cwd) or ["."])
        return self._run(command, cwd=cwd)

    def run_mypy(
        self,
        paths: list[str] | None = None,
        cwd: str | None = None,
    ) -> dict[str, Any]:
        python_executable = self._python_for_cwd(self._resolve_cwd(cwd))
        if not self._module_available(python_executable, "mypy", cwd):
            return self._skipped_optional_tool("mypy", cwd)
        command = [python_executable, "-m", "mypy"]
        command.extend(self._paths_for_command(paths, cwd) or ["."])
        return self._run(command, cwd=cwd)

    def terminal_execute(
        self,
        command: str,
        cwd: str | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Compatibility adapter for models that emit ``terminal.execute``.

        It is *not* a generic shell. Only Python script execution and pytest/ruff/mypy
        are accepted; pipes, redirections, PowerShell/cmd/bash and package managers are
        rejected. The resulting subprocess still uses shell=False.
        """
        if not self.allow_terminal_compat:
            raise ExecutionError("terminal.execute compatibility отключён")
        if not command.strip() or len(command) > 4000:
            raise ExecutionError("Некорректная команда")
        try:
            parts = shlex.split(command, posix=os.name != "nt")
        except ValueError as exc:
            raise ExecutionError(f"Не удалось разобрать команду: {exc}") from exc
        if not parts:
            raise ExecutionError("Пустая команда")

        first = Path(parts[0].strip('"')).name.casefold()
        args = [item.strip('"') for item in parts[1:]]
        python_executable = self._python_for_cwd(self._resolve_cwd(cwd))
        if first in {"pytest", "pytest.exe"}:
            if not self._module_available(python_executable, "pytest", cwd):
                return self._unavailable_required_tool("pytest", cwd)
            safe_args = self._safe_pytest_args(args, cwd)
            extra_env = (
                {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
                if self.isolate_pytest_plugins
                else None
            )
            return self._run(
                [python_executable, "-m", "pytest", *safe_args],
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                extra_env=extra_env,
            )
        if first in {"ruff", "ruff.exe"}:
            if not self._module_available(python_executable, "ruff", cwd):
                return self._skipped_optional_tool("ruff", cwd)
            safe_args = self._safe_path_tool_args(
                args, cwd, command="ruff", allowed_flags={"-q", "--quiet"},
                allow_check_word=True,
            )
            if not safe_args or safe_args == ["check"]:
                safe_args = ["check", "."]
            elif safe_args[0] != "check":
                safe_args.insert(0, "check")
            return self._run(
                [python_executable, "-m", "ruff", *safe_args],
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        if first in {"mypy", "mypy.exe"}:
            if not self._module_available(python_executable, "mypy", cwd):
                return self._skipped_optional_tool("mypy", cwd)
            safe_args = self._safe_path_tool_args(
                args,
                cwd,
                command="mypy",
                allowed_flags={"--strict", "--ignore-missing-imports"},
            )
            return self._run(
                [python_executable, "-m", "mypy", *(safe_args or ["."])],
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
        if first in {"python", "python.exe", Path(sys.executable).name.casefold()}:
            if not args:
                raise ExecutionError("Интерактивный Python через terminal.execute запрещён")
            if args[0] == "-c":
                raise ExecutionError("python -c запрещён; используйте execution.run_python")
            if args[0] == "-m":
                if len(args) < 2 or args[1] not in {"pytest", "ruff", "mypy", "compileall"}:
                    raise ExecutionError("Разрешены только python -m pytest/ruff/mypy/compileall")
                module = args[1]
                module_args = args[2:]
                if module == "pytest":
                    if not self._module_available(python_executable, "pytest", cwd):
                        return self._unavailable_required_tool("pytest", cwd)
                    safe_args = self._safe_pytest_args(module_args, cwd)
                elif module == "ruff":
                    if not self._module_available(python_executable, "ruff", cwd):
                        return self._skipped_optional_tool("ruff", cwd)
                    safe_args = self._safe_path_tool_args(
                        module_args, cwd, command="ruff",
                        allowed_flags={"-q", "--quiet"}, allow_check_word=True,
                    )
                    if not safe_args or safe_args == ["check"]:
                        safe_args = ["check", "."]
                    elif safe_args[0] != "check":
                        safe_args.insert(0, "check")
                elif module == "mypy":
                    if not self._module_available(python_executable, "mypy", cwd):
                        return self._skipped_optional_tool("mypy", cwd)
                    safe_args = self._safe_path_tool_args(
                        module_args, cwd, command="mypy",
                        allowed_flags={"--strict", "--ignore-missing-imports"},
                    ) or ["."]
                else:
                    safe_args = self._safe_path_tool_args(
                        module_args or ["."], cwd, command="compileall",
                        allowed_flags={"-q", "-f"},
                    )
                    safe_args = ["-x", self.COMPILEALL_EXCLUDE_RE, *safe_args]
                extra_env = (
                    {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
                    if module == "pytest" and self.isolate_pytest_plugins
                    else None
                )
                return self._run(
                    [python_executable, "-m", module, *safe_args],
                    cwd=cwd,
                    timeout_seconds=timeout_seconds,
                    extra_env=extra_env,
                )
            script = args[0]
            return self.run_python(
                script,
                args=args[1:],
                timeout_seconds=timeout_seconds,
                cwd=cwd,
            )
        raise ExecutionError(
            "terminal.execute разрешает только python <file.py>, pytest, ruff и mypy"
        )

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tools = {
            "execution.run_python": self.run_python,
            "execution.run_compileall": self.run_compileall,
            "execution.run_pytest": self.run_pytest,
            "execution.run_ruff": self.run_ruff,
            "execution.run_mypy": self.run_mypy,
            "terminal.execute": self.terminal_execute,
        }
        operation = tools.get(tool_name)
        if operation is None:
            raise ExecutionError(f"Неизвестный execution-инструмент: {tool_name}")
        try:
            return operation(**arguments)
        except (TypeError, WorkspaceError) as exc:
            raise ExecutionError(f"Некорректный вызов {tool_name}: {exc}") from exc

    @staticmethod
    def tool_specs() -> list[dict[str, Any]]:
        return [
            {
                "name": "execution.run_python",
                "arguments": {
                    "path": "str",
                    "args": "list[str]?",
                    "cwd": "str?",
                    "timeout_seconds": "int?",
                },
            },
            {
                "name": "execution.run_compileall",
                "arguments": {"path": "str", "cwd": "str?"},
            },
            {
                "name": "execution.run_pytest",
                "arguments": {
                    "paths": "list[str]?",
                    "verbose": "bool?",
                    "maxfail": "int?",
                    "cwd": "str?",
                },
            },
            {"name": "execution.run_ruff", "arguments": {"paths": "list[str]?", "cwd": "str?"}},
            {"name": "execution.run_mypy", "arguments": {"paths": "list[str]?", "cwd": "str?"}},
            {
                "name": "terminal.execute",
                "arguments": {"command": "str", "cwd": "str?", "timeout_seconds": "int?"},
                "note": (
                    "compatibility alias; only python file/pytest/ruff/mypy, "
                    "never a general shell"
                ),
            },
        ]
