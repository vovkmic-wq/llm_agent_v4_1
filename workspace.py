"""Безопасная файловая подсистема с атомарными и структурированными изменениями."""
from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


class WorkspaceError(RuntimeError):
    """Ошибка безопасной работы с workspace."""


class WorkspaceSecurityError(WorkspaceError):
    """Нарушение границы или политики защищённых файлов."""


@dataclass(frozen=True)
class WorkspaceLimits:
    max_file_size_bytes: int = 2_000_000
    max_list_entries: int = 500


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _pointer_parts(pointer: str) -> list[str]:
    if pointer in ("", "/"):
        return []
    if not pointer.startswith("/"):
        raise WorkspaceError("JSON/YAML pointer должен начинаться с '/'")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _resolve_container(document: Any, pointer: str, *, create: bool) -> tuple[Any, str | None]:
    parts = _pointer_parts(pointer)
    if not parts:
        return document, None
    current = document
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                if not create:
                    raise WorkspaceError(f"Путь не найден: {pointer}")
                current[part] = {}
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise WorkspaceError(f"Некорректный индекс в pointer: {pointer}") from exc
        else:
            raise WorkspaceError(f"Нельзя пройти по pointer: {pointer}")
    return current, parts[-1]


def _apply_structured_operations(document: Any, operations: list[dict[str, Any]]) -> Any:
    result = deepcopy(document)
    for operation in operations:
        op = str(operation.get("op", ""))
        pointer = str(operation.get("path", ""))
        value = deepcopy(operation.get("value"))
        container, key = _resolve_container(result, pointer, create=op in {"set", "merge"})
        if key is None:
            if op == "set":
                result = value
                continue
            if op == "merge" and isinstance(result, dict) and isinstance(value, dict):
                result = _deep_merge(result, value)
                continue
            raise WorkspaceError("Операция над корнем поддерживает только set/merge")

        if isinstance(container, dict):
            if op == "set":
                container[key] = value
            elif op == "delete":
                if key not in container:
                    raise WorkspaceError(f"Путь не найден: {pointer}")
                del container[key]
            elif op == "append_unique":
                target = container.setdefault(key, [])
                if not isinstance(target, list):
                    raise WorkspaceError(f"Ожидался список: {pointer}")
                if value not in target:
                    target.append(value)
            elif op == "extend_unique":
                target = container.setdefault(key, [])
                if not isinstance(target, list) or not isinstance(value, list):
                    raise WorkspaceError("extend_unique требует два списка")
                for item in value:
                    if item not in target:
                        target.append(item)
            elif op == "merge":
                target = container.get(key, {})
                if not isinstance(target, dict) or not isinstance(value, dict):
                    raise WorkspaceError("merge требует словари")
                container[key] = _deep_merge(target, value)
            else:
                raise WorkspaceError(f"Неизвестная structured operation: {op}")
        elif isinstance(container, list):
            try:
                index = int(key)
            except ValueError as exc:
                raise WorkspaceError(f"Ожидался индекс списка: {pointer}") from exc
            if op == "set":
                if index < 0 or index >= len(container):
                    raise WorkspaceError(f"Индекс вне диапазона: {pointer}")
                container[index] = value
            elif op == "delete":
                try:
                    del container[index]
                except IndexError as exc:
                    raise WorkspaceError(f"Индекс вне диапазона: {pointer}") from exc
            else:
                raise WorkspaceError(f"Операция {op} не поддерживается для элемента списка")
        else:
            raise WorkspaceError(f"Невозможно изменить pointer: {pointer}")
    return result


class WorkspaceManager:
    def __init__(
        self,
        root: Path,
        *,
        protected_globs: list[str] | None = None,
        ignored_globs: list[str] | None = None,
        limits: WorkspaceLimits | None = None,
        validate_on_write: bool = True,
    ) -> None:
        root = root.expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace не является директорией: {self.root}")
        self.protected_globs = protected_globs or []
        self.ignored_globs = ignored_globs or []
        self.limits = limits or WorkspaceLimits()
        self.validate_on_write = validate_on_write

    @classmethod
    def from_environment(
        cls,
        *,
        root_env: str,
        default_root: str | Path | None = None,
        protected_globs: list[str] | None = None,
        ignored_globs: list[str] | None = None,
        max_file_size_bytes: int = 2_000_000,
        max_list_entries: int = 500,
        validate_on_write: bool = True,
    ) -> "WorkspaceManager":
        raw_root = os.environ.get(root_env, "").strip()
        if raw_root:
            root = Path(raw_root).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
        elif default_root is not None:
            root = Path(default_root).expanduser()
            if not root.is_absolute():
                root = Path.cwd() / root
        else:
            raise WorkspaceError(
                f"Переменная {root_env} не задана и default_root не настроен."
            )
        return cls(
            root,
            protected_globs=protected_globs,
            ignored_globs=ignored_globs,
            limits=WorkspaceLimits(max_file_size_bytes, max_list_entries),
            validate_on_write=validate_on_write,
        )

    def _relative_posix(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    @staticmethod
    def _matches_globs(relative_path: str, patterns: list[str]) -> bool:
        normalized = relative_path[2:] if relative_path.startswith("./") else relative_path
        return any(
            fnmatch.fnmatch(normalized, pattern)
            or fnmatch.fnmatch(Path(normalized).name, pattern)
            for pattern in patterns
        )

    def _is_protected(self, relative_path: str) -> bool:
        return self._matches_globs(relative_path, self.protected_globs)

    def _is_ignored(self, relative_path: str) -> bool:
        return self._matches_globs(relative_path, self.ignored_globs)

    def resolve_path(self, relative_path: str, *, allow_root: bool = False) -> Path:
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise WorkspaceSecurityError("Путь должен быть непустой строкой")
        requested = Path(relative_path)
        if requested.is_absolute():
            raise WorkspaceSecurityError("Абсолютные пути запрещены")

        lexical = self.root
        for part in requested.parts:
            if part in ("", "."):
                continue
            lexical = lexical / part
            if lexical.exists() and lexical.is_symlink():
                raise WorkspaceSecurityError("Символические ссылки в workspace запрещены")

        candidate = (self.root / requested).resolve(strict=False)
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError("Выход за пределы workspace запрещён") from exc
        if candidate == self.root:
            if allow_root:
                return candidate
            raise WorkspaceSecurityError("Операция над корнем workspace запрещена")
        relative = self._relative_posix(candidate)
        if self._is_protected(relative):
            raise WorkspaceSecurityError(f"Путь защищён политикой workspace: {relative}")
        return candidate

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "validate_on_write": self.validate_on_write,
            "max_file_size_bytes": self.limits.max_file_size_bytes,
            "max_list_entries": self.limits.max_list_entries,
            "protected_globs": list(self.protected_globs),
            "ignored_globs": list(self.ignored_globs),
        }

    def _visible_listing_item(self, item: Path) -> tuple[str, bool] | None:
        if item.is_symlink():
            return None
        try:
            resolved = item.resolve(strict=False)
            relative = self._relative_posix(resolved)
        except (ValueError, OSError):
            return None
        if self._is_protected(relative) or self._is_ignored(relative):
            return None
        return relative, item.is_dir()

    def list_files(self, path: str = ".", recursive: bool = False) -> dict[str, Any]:
        """List visible project files while pruning caches and virtual environments."""
        target = self.resolve_path(path, allow_root=True)
        if not target.exists() or not target.is_dir():
            raise WorkspaceError(f"Директория не существует: {path}")

        entries: list[dict[str, Any]] = []
        truncated = False

        def append_item(item: Path) -> bool:
            nonlocal truncated
            visible = self._visible_listing_item(item)
            if visible is None:
                return False
            if len(entries) >= self.limits.max_list_entries:
                truncated = True
                return True
            relative, is_dir = visible
            try:
                size = item.stat().st_size if not is_dir else None
            except OSError:
                return False
            entries.append(
                {
                    "path": relative,
                    "type": "dir" if is_dir else "file",
                    "size": size,
                }
            )
            return len(entries) >= self.limits.max_list_entries

        if not recursive:
            for item in sorted(target.iterdir(), key=lambda p: p.name.casefold()):
                if append_item(item):
                    break
            return {"entries": entries, "truncated": truncated}

        for root, dirnames, filenames in os.walk(target, topdown=True, followlinks=False):
            root_path = Path(root)
            visible_dirs: list[str] = []
            for dirname in sorted(dirnames, key=str.casefold):
                directory = root_path / dirname
                if self._visible_listing_item(directory) is None:
                    continue
                visible_dirs.append(dirname)
                if append_item(directory):
                    break
            dirnames[:] = visible_dirs
            if truncated:
                break
            for filename in sorted(filenames, key=str.casefold):
                if append_item(root_path / filename):
                    break
            if truncated:
                break
        return {"entries": entries, "truncated": truncated}

    def read_file(self, path: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise WorkspaceError(f"Файл не существует: {path}")
        size = target.stat().st_size
        if size > self.limits.max_file_size_bytes:
            raise WorkspaceError(f"Файл слишком большой: {size} байт")
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("Поддерживаются только UTF-8 текстовые файлы") from exc
        return {
            "path": self._relative_posix(target),
            "content": content,
            "size": size,
            "sha256": _sha256(content.encode("utf-8")),
        }

    def read_lines(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 30000,
    ) -> dict[str, Any]:
        """Read a bounded 1-based line range while preserving full-file SHA."""
        raw = self.read_file(path)
        lines = raw["content"].splitlines(keepends=True)
        total = len(lines)
        if start_line < 1 or start_line > total + 1:
            raise WorkspaceError(f"start_line вне диапазона: {start_line}")
        resolved_end = total if end_line is None else end_line
        if resolved_end < start_line - 1 or resolved_end > total:
            raise WorkspaceError(f"end_line вне диапазона: {resolved_end}")
        selected = "".join(lines[start_line - 1 : resolved_end])
        clipped = selected[: max(1000, min(int(max_chars), 100000))]
        return {
            "path": raw["path"],
            "content": clipped,
            "start_line": start_line,
            "end_line": resolved_end,
            "total_lines": total,
            "sha256": raw["sha256"],
            "truncated": len(clipped) < len(selected),
        }

    def _validate_content(self, target: Path, content: str) -> None:
        if not self.validate_on_write:
            return
        suffix = target.suffix.lower()
        try:
            if suffix == ".py":
                ast.parse(content, filename=str(target))
            elif suffix == ".json":
                json.loads(content)
            elif suffix in {".yaml", ".yml"}:
                yaml.safe_load(content)
            elif suffix == ".toml":
                tomllib.loads(content)
        except (SyntaxError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
            raise WorkspaceError(f"Валидация {suffix or 'файла'} не пройдена: {exc}") from exc

    def _atomic_write(self, target: Path, content: str) -> None:
        encoded = content.encode("utf-8")
        if len(encoded) > self.limits.max_file_size_bytes:
            raise WorkspaceError("Содержимое превышает лимит workspace")
        self._validate_content(target, content)
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent.resolve(strict=True)
        try:
            parent.relative_to(self.root)
        except ValueError as exc:
            raise WorkspaceSecurityError("Родительская директория вышла за workspace") from exc
        fd, tmp_name = tempfile.mkstemp(prefix=".agent-", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def write_file(self, path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        target = self.resolve_path(path)
        if target.exists() and not overwrite:
            raise WorkspaceError(f"Файл уже существует: {path}; используйте overwrite=true")
        if target.exists() and not target.is_file():
            raise WorkspaceError(f"Путь не является файлом: {path}")
        self._atomic_write(target, content)
        verify = self.read_file(path)
        return {
            "path": verify["path"],
            "bytes_written": verify["size"],
            "sha256": verify["sha256"],
            "verified": True,
        }

    def replace_text(
        self,
        path: str,
        old_text: str,
        new_text: str,
        count: int = 0,
    ) -> dict[str, Any]:
        if not old_text:
            raise WorkspaceError("old_text не может быть пустым")
        current = self.read_file(path)["content"]
        occurrences = current.count(old_text)
        if occurrences == 0:
            raise WorkspaceError("Искомый текст не найден; файл не изменён")
        if count < 0:
            raise WorkspaceError("count не может быть отрицательным")
        replace_count = count if count > 0 else occurrences
        updated = current.replace(old_text, new_text, replace_count)
        written = self.write_file(path, updated, overwrite=True)
        return {**written, "replacements": min(occurrences, replace_count)}

    def apply_patch(self, path: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        """Структурированный текстовый patch без shell и внешней команды patch."""
        content = self.read_file(path)["content"]
        changes = 0
        for operation in operations:
            op = operation.get("op")
            if op == "replace":
                old = operation.get("old_text", "")
                new = operation.get("new_text", "")
                count = int(operation.get("count", 0))
                if not old or old not in content:
                    raise WorkspaceError("apply_patch: old_text не найден")
                actual = count if count > 0 else content.count(old)
                content = content.replace(old, new, actual)
                changes += actual
            elif op == "append":
                content += str(operation.get("text", ""))
                changes += 1
            elif op == "prepend":
                content = str(operation.get("text", "")) + content
                changes += 1
            else:
                raise WorkspaceError(f"Неизвестная patch operation: {op}")
        written = self.write_file(path, content, overwrite=True)
        return {**written, "changes": changes}


    def replace_lines(
        self,
        path: str,
        start_line: int,
        end_line: int,
        content: str,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Replace an inclusive 1-based line range with raw payload content.

        This is the preferred coding primitive for V3 because large source snippets
        travel outside JSON. ``expected_sha256`` prevents stale-model edits.
        """
        current = self.read_file(path)
        if expected_sha256 and current["sha256"] != expected_sha256:
            raise WorkspaceError(
                "Файл изменился после чтения; перечитайте его перед patch"
            )
        if start_line < 1 or end_line < start_line:
            raise WorkspaceError("Некорректный диапазон строк")
        lines = current["content"].splitlines(keepends=True)
        if start_line > len(lines) + 1 or end_line > max(len(lines), start_line - 1):
            raise WorkspaceError(
                f"Диапазон {start_line}:{end_line} выходит за файл из {len(lines)} строк"
            )
        replacement = content
        if replacement and not replacement.endswith(("\n", "\r")):
            replacement += "\n"
        replacement_lines = replacement.splitlines(keepends=True)
        before = lines[: start_line - 1]
        after = lines[end_line:]
        updated = "".join([*before, *replacement_lines, *after])
        written = self.write_file(path, updated, overwrite=True)
        return {
            **written,
            "start_line": start_line,
            "end_line": end_line,
            "inserted_lines": len(replacement_lines),
        }

    def read_json(self, path: str) -> dict[str, Any]:
        raw = self.read_file(path)
        try:
            data = json.loads(raw["content"])
        except json.JSONDecodeError as exc:
            raise WorkspaceError(f"Некорректный JSON: {exc}") from exc
        return {"path": raw["path"], "data": data, "sha256": raw["sha256"]}

    def write_json(self, path: str, data: Any, overwrite: bool = True) -> dict[str, Any]:
        content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        return self.write_file(path, content, overwrite=overwrite)

    def merge_json(self, path: str, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.read_json(path)["data"]
        if not isinstance(current, dict):
            raise WorkspaceError("merge_json поддерживает JSON-объект верхнего уровня")
        return self.write_json(path, _deep_merge(current, patch), overwrite=True)

    def patch_json(self, path: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        current = self.read_json(path)["data"]
        return self.write_json(path, _apply_structured_operations(current, operations), True)

    def read_yaml(self, path: str) -> dict[str, Any]:
        raw = self.read_file(path)
        try:
            data = yaml.safe_load(raw["content"])
        except yaml.YAMLError as exc:
            raise WorkspaceError(f"Некорректный YAML: {exc}") from exc
        return {"path": raw["path"], "data": data, "sha256": raw["sha256"]}

    def write_yaml(self, path: str, data: Any, overwrite: bool = True) -> dict[str, Any]:
        content = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        return self.write_file(path, content, overwrite=overwrite)

    def merge_yaml(self, path: str, patch: dict[str, Any]) -> dict[str, Any]:
        current = self.read_yaml(path)["data"] or {}
        if not isinstance(current, dict):
            raise WorkspaceError("merge_yaml поддерживает YAML-object верхнего уровня")
        return self.write_yaml(path, _deep_merge(current, patch), True)

    def patch_yaml(self, path: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        current = self.read_yaml(path)["data"]
        return self.write_yaml(path, _apply_structured_operations(current, operations), True)

    def delete_file(self, path: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists() or not target.is_file():
            raise WorkspaceError(f"Файл не существует: {path}")
        relative = self._relative_posix(target)
        target.unlink()
        return {"path": relative, "deleted": True}

    def make_directory(self, path: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": self._relative_posix(target), "created": True}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tools = {
            "workspace.list_files": self.list_files,
            "workspace.read_file": self.read_file,
            "workspace.read_lines": self.read_lines,
            "workspace.write_file": self.write_file,
            "workspace.replace_text": self.replace_text,
            "workspace.apply_patch": self.apply_patch,
            "workspace.replace_lines": self.replace_lines,
            "workspace.read_json": self.read_json,
            "workspace.write_json": self.write_json,
            "workspace.merge_json": self.merge_json,
            "workspace.patch_json": self.patch_json,
            "workspace.read_yaml": self.read_yaml,
            "workspace.write_yaml": self.write_yaml,
            "workspace.merge_yaml": self.merge_yaml,
            "workspace.patch_yaml": self.patch_yaml,
            "workspace.delete_file": self.delete_file,
            "workspace.make_directory": self.make_directory,
        }
        operation = tools.get(tool_name)
        if operation is None:
            raise WorkspaceError(f"Неизвестный workspace-инструмент: {tool_name}")
        try:
            return operation(**arguments)
        except TypeError as exc:
            raise WorkspaceError(f"Некорректные аргументы {tool_name}: {exc}") from exc

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {"name": "workspace.list_files", "arguments": {"path": "str", "recursive": "bool"}},
            {"name": "workspace.read_file", "arguments": {"path": "str"}},
            {
                "name": "workspace.read_lines",
                "arguments": {
                    "path": "str",
                    "start_line": "int?",
                    "end_line": "int?",
                    "max_chars": "int?",
                },
            },
            {
                "name": "workspace.write_file",
                "arguments": {"path": "str", "content": "str", "overwrite": "bool"},
            },
            {
                "name": "workspace.apply_patch",
                "arguments": {"path": "str", "operations": "list[patch-op]"},
            },
            {
                "name": "workspace.replace_lines",
                "arguments": {
                    "path": "str",
                    "start_line": "int",
                    "end_line": "int",
                    "content": "str (prefer raw PAYLOAD)",
                    "expected_sha256": "str?",
                },
            },
            {"name": "workspace.read_json", "arguments": {"path": "str"}},
            {
                "name": "workspace.write_json",
                "arguments": {"path": "str", "data": "any", "overwrite": "bool"},
            },
            {"name": "workspace.merge_json", "arguments": {"path": "str", "patch": "object"}},
            {
                "name": "workspace.patch_json",
                "arguments": {
                    "path": "str",
                    "operations": "list[set|delete|append_unique|extend_unique|merge]",
                },
            },
            {"name": "workspace.read_yaml", "arguments": {"path": "str"}},
            {
                "name": "workspace.write_yaml",
                "arguments": {"path": "str", "data": "any", "overwrite": "bool"},
            },
            {"name": "workspace.merge_yaml", "arguments": {"path": "str", "patch": "object"}},
            {
                "name": "workspace.patch_yaml",
                "arguments": {"path": "str", "operations": "list[structured-op]"},
            },
            {"name": "workspace.make_directory", "arguments": {"path": "str"}},
            {"name": "workspace.delete_file", "arguments": {"path": "str"}},
        ]
