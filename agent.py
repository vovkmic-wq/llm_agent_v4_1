"""Production orchestration kernel for LLM Coding Agent V4.1 ICS.

Pipeline: project contract -> plan -> baseline gates -> executor/tool loop ->
quality gates -> reviewer -> evidence-checked final. Protocol formatting failures are
repaired deterministically and can trigger model fallback without losing TaskState.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from core.config import GenerationProfileConfig, STATE_DIR, settings
from core.event_logger import EventLogger
from core.execution import ExecutionManager
from core.failure_analyzer import FailureDigest, analyze_test_output
from core.incidents import IncidentKind, IncidentManager
from core.logger import console_logger
from core.memory import MemoryManager
from core.permissions import ConfirmationHandler, PermissionManager
from core.project_context import ProjectContextManager
from core.prompt_builder import PromptBuilder, rough_token_estimate
from core.protocol import (
    FinalAnswer,
    ProtocolError,
    ToolBatch,
    ToolCall,
    format_tool_result,
    parse_agent_response,
    repair_instruction,
)
from core.research import ResearchManager
from core.router import Router, choose_model, resolve_model_id
from core.safety import SafetyVerdict, build_safety_prompt
from core.task_state import TaskStateManager
from core.tool_registry import ToolRegistry, ToolRegistryError
from core.workspace import WorkspaceError, WorkspaceManager
from providers.base import CompletionResult, ProviderError


class AgentExecutionError(RuntimeError):
    """The orchestration loop could not safely complete."""


class PlanResponse(BaseModel):
    goal: str
    steps: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class ReviewResponse(BaseModel):
    approved: bool
    feedback: str = ""


def _extract_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


class Agent:
    def __init__(
        self,
        *,
        memory: MemoryManager | None = None,
        router: Router | None = None,
        workspace: WorkspaceManager | None = None,
        confirmation_handler: ConfirmationHandler | None = None,
        planner_enabled: bool | None = None,
        reviewer_enabled: bool | None = None,
        execution_enabled: bool | None = None,
        research_enabled: bool | None = None,
    ) -> None:
        self.memory = memory or MemoryManager()
        self.prompt_builder = PromptBuilder(self.memory)
        self.router = router or Router()
        self.workspace = workspace or self._build_workspace()
        if self.workspace is None:
            raise WorkspaceError("Agent V3 требует включённый workspace")

        exec_enabled = (
            settings.code_execution_enabled
            if execution_enabled is None
            else execution_enabled
        )
        research_is_enabled = (
            settings.research_enabled
            if research_enabled is None
            else research_enabled
        )
        self.execution = ExecutionManager(
            self.workspace,
            enabled=exec_enabled,
            timeout_seconds=settings.agent.execution.timeout_seconds,
            max_output_chars=settings.agent.execution.max_output_chars,
            allow_terminal_compat=settings.agent.execution.allow_terminal_compat,
        )
        self.research = ResearchManager(
            enabled=research_is_enabled,
            allowed_domains=settings.agent.research.allowed_domains,
            timeout_seconds=settings.agent.research.timeout_seconds,
            max_response_bytes=settings.agent.research.max_response_bytes,
        )
        self.tools = ToolRegistry(self.workspace, self.execution, self.research)
        self.permissions = PermissionManager(
            settings.agent.permissions.rules,
            confirmation_handler=confirmation_handler,
        )
        self.events = EventLogger(STATE_DIR / "logs" / "events.jsonl")
        self.task_state = TaskStateManager(self.memory.memory_dir / "task_state.json")
        self.project_context = ProjectContextManager(self.memory.memory_dir, self.workspace)
        incident_cfg = settings.agent.incident
        self.incident = IncidentManager(
            self.memory.memory_dir / "incidents",
            max_period_rounds=incident_cfg.max_period_rounds,
            max_periods=incident_cfg.max_periods,
            max_plan_revisions=incident_cfg.max_plan_revisions,
            max_safety_stops=incident_cfg.max_safety_stops,
            max_read_only_rounds=incident_cfg.max_read_only_rounds,
            max_no_progress_rounds=incident_cfg.max_no_progress_rounds,
        )
        self.planner_enabled = (
            settings.agent.orchestration.planning_enabled
            if planner_enabled is None
            else planner_enabled
        )
        self.reviewer_enabled = (
            settings.agent.orchestration.review_enabled
            if reviewer_enabled is None
            else reviewer_enabled
        )

    @staticmethod
    def _build_workspace() -> WorkspaceManager | None:
        cfg = settings.agent.workspace
        if not cfg.enabled:
            return None
        return WorkspaceManager.from_environment(
            root_env=cfg.root_env,
            default_root="../AGENT_WORKSPACE",
            protected_globs=cfg.protected_globs,
            ignored_globs=cfg.ignored_globs,
            max_file_size_bytes=cfg.max_file_size_bytes,
            max_list_entries=cfg.max_list_entries,
            validate_on_write=cfg.validate_on_write,
        )

    def _router_call(
        self,
        *,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        profile: GenerationProfileConfig,
        allow_fallback: bool = True,
    ) -> CompletionResult:
        """Call modern Router while remaining compatible with simple test routers."""
        try:
            return self.router.call(
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
                allow_fallback=allow_fallback,
            )
        except TypeError as exc:
            # Test doubles and third-party routers built for V2 only expose three args.
            if "unexpected keyword" not in str(exc):
                raise
            return self.router.call(
                model_id=model_id,
                system_prompt=system_prompt,
                messages=messages,
            )

    def _is_task_request(
        self,
        user_query: str,
        tags: list[str] | None,
        force_task: bool,
    ) -> bool:
        if force_task:
            return True
        normalized_tags = {tag.casefold() for tag in (tags or [])}
        if normalized_tags & {"code", "программирование", "task", "spec"}:
            return True
        lowered = user_query.casefold()
        return any(
            keyword.casefold() in lowered
            for keyword in settings.agent.orchestration.task_keywords
        )

    def _workspace_snapshot(self) -> str:
        root = self.project_context.active_project
        listing = self.workspace.list_files(root, recursive=True)
        entries = listing["entries"][:400]
        lines = [
            (
                f"{item['type']}: {item['path']}"
                + (f" ({item.get('size')} bytes)" if item.get("size") is not None else "")
            )
            for item in entries
        ]
        if listing["truncated"] or len(listing["entries"]) > len(entries):
            lines.append("... snapshot truncated ...")
        return "\n".join(lines) or "(project workspace пуст)"

    def _create_plan(self, user_query: str, project_contract: str) -> PlanResponse:
        prompt = settings.prompts.planner_prompt.format(
            user_query=user_query,
            workspace_snapshot=self._workspace_snapshot(),
            project_context=project_contract,
        )
        model_id = settings.agent.roles.planner
        self.events.log("planner_request", model=model_id, query=user_query)
        try:
            result = self._router_call(
                model_id=model_id,
                system_prompt="Ты Planner. Верни только требуемый компактный JSON.",
                messages=[{"role": "user", "content": prompt}],
                profile=settings.agent.generation.planner,
            )
            payload = _extract_json_object(result.text)
            if payload is None:
                raise ValueError("Planner не вернул JSON")
            plan = PlanResponse.model_validate(payload)
            if not plan.steps:
                plan.steps = ["Выполнить задачу с проверкой результата"]
            return plan
        except (ProviderError, ValueError, ValidationError) as exc:
            console_logger.warning("Planner fallback: %s", exc)
            return PlanResponse(
                goal=user_query[:500],
                steps=[
                    "Проанализировать активный проект и ТЗ",
                    "Запустить доступные проверки и диагностировать ошибки",
                    "Внести минимальные изменения",
                    "Повторить проверки до подтверждённого результата",
                ],
                acceptance_criteria=[
                    "Результат соответствует запросу и PROJECT CONTRACT",
                    "Утверждения о тестах подтверждены execution evidence",
                ],
            )

    @staticmethod
    def _compact_evidence_for_review(
        evidence: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        compact: dict[str, dict[str, Any]] = {}
        for call_id, record in evidence.items():
            item: dict[str, Any] = {
                "tool": record.get("tool"),
                "ok": bool(record.get("ok")),
            }
            result = record.get("result")
            if isinstance(result, dict):
                for key in (
                    "path",
                    "cwd",
                    "exit_code",
                    "skipped",
                    "infrastructure_error",
                    "reason",
                    "verified",
                    "sha256",
                ):
                    if key in result:
                        item[key] = result[key]
                output = result.get("output")
                if isinstance(output, str) and output:
                    item["output_tail"] = output[-2500:]
            if record.get("error"):
                item["error"] = str(record["error"])[:1000]
            compact[call_id] = item
        return compact

    def _review(
        self,
        *,
        goal: str,
        plan: PlanResponse | None,
        answer: FinalAnswer,
        evidence: dict[str, dict[str, Any]],
        project_contract: str,
    ) -> ReviewResponse:
        prompt = settings.prompts.reviewer_prompt.format(
            goal=goal,
            plan=json.dumps(plan.model_dump() if plan else {}, ensure_ascii=False),
            answer=f"status={answer.status}\n{answer.content}",
            evidence=json.dumps(
                self._compact_evidence_for_review(evidence),
                ensure_ascii=False,
            )[:24000],
            project_context=project_contract,
        )
        model_id = settings.agent.roles.reviewer
        self.events.log("reviewer_request", model=model_id, task_id=self._task_id())
        try:
            result = self._router_call(
                model_id=model_id,
                system_prompt="Ты Reviewer. Верни только JSON approved/feedback.",
                messages=[{"role": "user", "content": prompt}],
                profile=settings.agent.generation.reviewer,
            )
            payload = _extract_json_object(result.text)
            if payload is None:
                raise ValueError("Reviewer не вернул JSON")
            return ReviewResponse.model_validate(payload)
        except (ProviderError, ValueError, ValidationError) as exc:
            console_logger.warning("Reviewer недоступен, пропуск review: %s", exc)
            return ReviewResponse(approved=True, feedback="Reviewer unavailable")

    def _task_id(self) -> str | None:
        return self.task_state.current.task_id if self.task_state.current else None

    @staticmethod
    def _is_mutating_tool(tool: str) -> bool:
        return tool in {
            "workspace.write_file",
            "workspace.replace_text",
            "workspace.apply_patch",
            "workspace.replace_lines",
            "workspace.write_json",
            "workspace.merge_json",
            "workspace.patch_json",
            "workspace.write_yaml",
            "workspace.merge_yaml",
            "workspace.patch_yaml",
            "workspace.delete_file",
            "workspace.make_directory",
        }

    @staticmethod
    def _changed_path(tool: str, result: dict[str, Any]) -> str | None:
        if tool.startswith("workspace.") and tool not in {
            "workspace.read_file",
            "workspace.read_lines",
            "workspace.read_json",
            "workspace.read_yaml",
            "workspace.list_files",
        }:
            path = result.get("path")
            return str(path) if path else None
        return None

    def _mutation_precondition_error(
        self,
        call: ToolCall,
        read_paths: set[str],
    ) -> str | None:
        if not self._is_mutating_tool(call.tool):
            return None
        if call.tool in {"workspace.make_directory", "workspace.delete_file"}:
            return None
        path = call.arguments.get("path")
        if not isinstance(path, str):
            return None
        try:
            target = self.workspace.resolve_path(path)
        except WorkspaceError:
            return None
        if target.exists() and target.is_file():
            relative = target.relative_to(self.workspace.root).as_posix()
            if relative not in read_paths:
                return (
                    f"Перед изменением существующего файла {relative} сначала вызовите "
                    "workspace.read_file/read_json/read_yaml."
                )
        return None

    @staticmethod
    def _action_for_history(call: ToolCall) -> str:
        arguments: dict[str, Any] = {}
        for key, value in call.arguments.items():
            if isinstance(value, str) and len(value) > 1000:
                arguments[key] = {
                    "redacted_chars": len(value),
                    "sha256_16": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
                }
            else:
                arguments[key] = value
        return json.dumps(
            {
                "type": "tool",
                "tool": call.tool,
                "arguments": arguments,
                "call_id": call.call_id,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _clip_for_context(text: str, limit: int = 8000) -> str:
        if len(text) <= limit:
            return text
        marker = "\n...[tool output clipped for context]...\n"
        head = max(1000, int((limit - len(marker)) * 0.30))
        tail = max(1000, limit - len(marker) - head)
        return text[:head] + marker + text[-tail:]

    @classmethod
    def _result_for_model(cls, tool: str, result: dict[str, Any]) -> dict[str, Any]:
        """Bound tool-result size before it is appended to the LLM conversation."""
        contextual = dict(result)
        if tool == "workspace.read_file":
            content = contextual.get("content")
            if isinstance(content, str) and len(content) > 8000:
                contextual["content"] = cls._clip_for_context(content)
                contextual["truncated_for_context"] = True
                contextual["instruction"] = (
                    "Файл обрезан только в LLM-контексте. Используй "
                    "workspace.read_lines для нужного диапазона; sha256 относится "
                    "к полному файлу."
                )
        elif tool in {"workspace.read_json", "workspace.read_yaml"}:
            data = contextual.get("data")
            if data is not None:
                serialized = json.dumps(data, ensure_ascii=False, default=str)
                if len(serialized) > 8000:
                    contextual.pop("data", None)
                    contextual["data_preview"] = cls._clip_for_context(serialized)
                    contextual["truncated_for_context"] = True
                    contextual["instruction"] = (
                        "Структурированные данные сокращены для контекста; "
                        "прочитай нужный диапазон через workspace.read_lines."
                    )
        elif tool.startswith("execution.") or tool == "terminal.execute":
            output = contextual.get("output")
            if isinstance(output, str) and len(output) > 8000:
                contextual["output"] = cls._clip_for_context(output)
                contextual["truncated_for_context"] = True
        elif tool.startswith("research."):
            text = contextual.get("text")
            if isinstance(text, str) and len(text) > 8000:
                contextual["text"] = cls._clip_for_context(text)
                contextual["truncated_for_context"] = True
        return contextual

    def _trim_loop_messages(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        history_message_count: int,
        evidence: dict[str, dict[str, Any]],
    ) -> int:
        """Keep a long tool loop inside the configured context window.

        Historical chat is removed first. If a coding loop is still too large,
        old tool exchanges are collapsed into a deterministic kernel trace while
        preserving the current task query and the newest interactions.
        """
        token_limit = settings.agent.memory.max_context_tokens

        def estimate() -> int:
            return rough_token_estimate(
                system_prompt + "".join(str(item.get("content", "")) for item in messages)
            )

        while estimate() > token_limit and history_message_count >= 2:
            del messages[:2]
            history_message_count -= 2

        if estimate() <= token_limit or len(messages) <= 8:
            return history_message_count

        # History is gone here in the common case: preserve current user task and
        # the newest tool interactions; summarize older evidence deterministically.
        anchor_index = history_message_count
        if anchor_index >= len(messages):
            anchor_index = 0
        anchor = messages[anchor_index]
        recent = messages[-6:]
        trace = json.dumps(
            {
                "type": "kernel_trace_summary",
                "evidence": self._compact_evidence_for_review(evidence),
                "instruction": "Продолжи текущую задачу; call_id остаются действительными.",
            },
            ensure_ascii=False,
            default=str,
        )
        trace = self._clip_for_context(trace, 6000)
        messages[:] = [anchor, {"role": "user", "content": trace}, *recent]
        history_message_count = 0

        # Last-resort clipping for unexpectedly huge third-party router messages.
        if estimate() > token_limit:
            for item in messages[1:]:
                content = str(item.get("content", ""))
                if len(content) > 5000:
                    item["content"] = self._clip_for_context(content, 5000)
        return history_message_count

    def _execute_tool(
        self,
        call: ToolCall,
        evidence: dict[str, dict[str, Any]],
        read_paths: set[str] | None = None,
    ) -> tuple[str, bool, bool]:
        if call.call_id in evidence:
            error = (
                f"call_id '{call.call_id}' уже использован в этой задаче; "
                "повторите действие с новым call_id"
            )
            self.events.log(
                "duplicate_call_id",
                task_id=self._task_id(),
                call_id=call.call_id,
                tool=call.tool,
            )
            return (
                format_tool_result(
                    call_id=call.call_id,
                    tool=call.tool,
                    error=error,
                ),
                False,
                False,
            )

        allowed, reason = self.permissions.authorize(call.tool, call.arguments)
        if not allowed:
            payload = format_tool_result(
                call_id=call.call_id,
                tool=call.tool,
                error=reason or "Операция запрещена",
            )
            self._record_tool_result(call, False, evidence, error=reason)
            return payload, False, False

        active_read_paths = read_paths if read_paths is not None else set()
        precondition_error = self._mutation_precondition_error(call, active_read_paths)
        if precondition_error:
            payload = format_tool_result(
                call_id=call.call_id,
                tool=call.tool,
                error=precondition_error,
            )
            self._record_tool_result(call, False, evidence, error=precondition_error)
            return payload, False, False

        if self.incident.state is not None:
            progress_ok, progress_reason = self.incident.progress_gate()
            is_read_only = call.tool in {
                "workspace.read_file",
                "workspace.read_lines",
                "workspace.read_json",
                "workspace.read_yaml",
                "workspace.list_files",
            }
            if is_read_only and not progress_ok:
                payload = format_tool_result(
                    call_id=call.call_id,
                    tool=call.tool,
                    error=progress_reason or "Механический progress gate запретил чтение",
                )
                self._record_tool_result(
                    call, False, evidence, error=progress_reason or "progress gate"
                )
                return payload, False, False
            if self._is_mutating_tool(call.tool):
                gate_ok, gate_reason = self.incident.mutation_gate()
                if not gate_ok:
                    payload = format_tool_result(
                        call_id=call.call_id,
                        tool=call.tool,
                        error=gate_reason,
                    )
                    self._record_tool_result(call, False, evidence, error=gate_reason)
                    return payload, False, False
                path_value = call.arguments.get("path")
                if isinstance(path_value, str):
                    try:
                        target = self.workspace.resolve_path(path_value)
                        if target.exists() and target.is_file():
                            relative = target.relative_to(self.workspace.root).as_posix()
                            self.incident.capture_original(
                                relative, target.read_text(encoding="utf-8")
                            )
                    except (OSError, UnicodeDecodeError, WorkspaceError):
                        pass

        console_logger.info("[TOOL] %s (%s)", call.tool, call.call_id)
        self.events.log(
            "tool_call",
            task_id=self._task_id(),
            call_id=call.call_id,
            tool=call.tool,
            arguments=call.arguments,
        )
        try:
            result = self.tools.execute(call.tool, call.arguments)
            semantic_ok = bool(result.get("ok", True))
            error = None if semantic_ok else (
                f"Команда завершилась с exit_code={result.get('exit_code')}"
            )
            payload = format_tool_result(
                call_id=call.call_id,
                tool=call.tool,
                result=self._result_for_model(call.tool, result),
                error=error,
                ok=semantic_ok,
            )
            evidence[call.call_id] = {
                "tool": call.tool,
                "ok": semantic_ok,
                "result": result,
            }
            if semantic_ok and read_paths is not None and call.tool in {
                "workspace.read_file",
                "workspace.read_lines",
                "workspace.read_json",
                "workspace.read_yaml",
            }:
                result_path = result.get("path")
                if isinstance(result_path, str):
                    read_paths.add(result_path)
            changed_path = self._changed_path(call.tool, result)
            self.task_state.record_tool(
                call_id=call.call_id,
                tool=call.tool,
                ok=semantic_ok,
                changed_path=changed_path,
            )
            if semantic_ok and changed_path and read_paths is not None:
                # A successful mutation invalidates the model's previous read snapshot.
                read_paths.discard(changed_path)
            self.events.log(
                "tool_result",
                task_id=self._task_id(),
                call_id=call.call_id,
                tool=call.tool,
                ok=semantic_ok,
                result=result,
            )
            return payload, semantic_ok, semantic_ok and self._is_mutating_tool(call.tool)
        except ToolRegistryError as exc:
            payload = format_tool_result(
                call_id=call.call_id,
                tool=call.tool,
                error=str(exc),
            )
            self._record_tool_result(call, False, evidence, error=str(exc))
            return payload, False, False

    def _record_tool_result(
        self,
        call: ToolCall,
        ok: bool,
        evidence: dict[str, dict[str, Any]],
        *,
        error: str | None,
    ) -> None:
        evidence[call.call_id] = {"tool": call.tool, "ok": ok, "error": error}
        self.task_state.record_tool(call_id=call.call_id, tool=call.tool, ok=ok)
        self.events.log(
            "tool_result",
            task_id=self._task_id(),
            call_id=call.call_id,
            tool=call.tool,
            ok=ok,
            error=error,
        )

    def _quality_gate_calls(self) -> list[ToolCall]:
        cfg = settings.agent.quality_gates
        if not cfg.auto_run or not self.execution.enabled:
            return []
        project = self.project_context.active_project
        listing = self.workspace.list_files(project, recursive=True)["entries"]
        python_files = [
            item["path"]
            for item in listing
            if item["type"] == "file" and item["path"].endswith(".py")
        ]
        tests_exist = any(
            "/tests/" in f"/{path}" or Path(path).name.startswith("test_")
            for path in python_files
        )
        calls: list[ToolCall] = []
        if cfg.compileall and python_files:
            calls.append(
                ToolCall(
                    tool="execution.run_compileall",
                    arguments={"path": ".", "cwd": project},
                )
            )
        if cfg.pytest and tests_exist:
            calls.append(
                ToolCall(
                    tool="execution.run_pytest",
                    arguments={"maxfail": 10, "cwd": project},
                )
            )
        if cfg.ruff and python_files:
            calls.append(ToolCall(tool="execution.run_ruff", arguments={"cwd": project}))
        if cfg.mypy and python_files:
            calls.append(ToolCall(tool="execution.run_mypy", arguments={"cwd": project}))
        return calls

    def _run_quality_gates(
        self,
        evidence: dict[str, dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        results: list[dict[str, Any]] = []
        all_ok = True
        for call in self._quality_gate_calls():
            call.call_id = f"kernel-gate-{uuid.uuid4().hex[:12]}"
            payload, ok, _ = self._execute_tool(call, evidence)
            decoded = json.loads(payload)
            results.append(decoded)
            all_ok = all_ok and ok
        return results, all_ok

    @staticmethod
    def _classify_gate_results(
        results: list[dict[str, Any]],
    ) -> tuple[set[str], set[str], set[str]]:
        successful: set[str] = set()
        code_failures: set[str] = set()
        infrastructure_failures: set[str] = set()
        for item in results:
            call_id = str(item.get("call_id", ""))
            if not call_id:
                continue
            if item.get("ok"):
                successful.add(call_id)
                continue
            result = item.get("result")
            if isinstance(result, dict) and result.get("infrastructure_error"):
                infrastructure_failures.add(call_id)
            else:
                code_failures.add(call_id)
        return successful, code_failures, infrastructure_failures

    @staticmethod
    def _requires_execution(user_query: str, task_mode: bool) -> bool:
        if not task_mode:
            return False
        lowered = user_query.casefold()
        markers = (
            "протест",
            "работоспособ",
            "запусти",
            "запуск",
            "pytest",
            "ruff",
            "mypy",
            "unit test",
            "integration test",
            "выполни тест",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _validate_final_evidence(
        answer: FinalAnswer,
        evidence: dict[str, dict[str, Any]],
        *,
        task_mode: bool,
        required_evidence: set[str] | None = None,
        required_blocker_evidence: set[str] | None = None,
        require_execution: bool = False,
    ) -> str | None:
        known = set(evidence)
        successful = {call_id for call_id, record in evidence.items() if record.get("ok")}
        failed = known.difference(successful)
        required_success = required_evidence or set()
        required_blockers = required_blocker_evidence or set()

        if task_mode and not evidence:
            return "Engineering task нельзя завершить без реальных tool_result."
        if task_mode and not answer.evidence:
            return "Финальный ответ engineering-task должен содержать evidence call_id."

        unknown = [item for item in answer.evidence if item not in known]
        if unknown:
            return f"Final evidence содержит неизвестные call_id: {unknown}"

        if answer.status == "success":
            unsuccessful_refs = [item for item in answer.evidence if item not in successful]
            if unsuccessful_refs:
                return (
                    "status=success не может ссылаться на неуспешные evidence: "
                    f"{unsuccessful_refs}"
                )
            if required_blockers:
                return (
                    "Нельзя вернуть status=success: остаются инфраструктурные блокеры "
                    f"{sorted(required_blockers)}. Верни status=blocked/partial."
                )
        elif answer.status == "blocked":
            blocker_refs = failed.intersection(answer.evidence)
            if not blocker_refs:
                return "status=blocked должен ссылаться хотя бы на один неуспешный call_id."
        elif answer.status == "partial" and not answer.evidence:
            return "status=partial должен содержать evidence выполненной части задачи."

        if require_execution:
            cited_execution = [
                record
                for call_id, record in evidence.items()
                if call_id in answer.evidence
                and (
                    str(record.get("tool", "")).startswith("execution.")
                    or record.get("tool") == "terminal.execute"
                )
            ]
            if answer.status == "success":
                if not any(record.get("ok") for record in cited_execution):
                    return (
                        "Задача требует фактического запуска/тестирования кода. "
                        "Сошлись в final.evidence на успешный execution tool_result."
                    )
            elif not cited_execution:
                return (
                    "Для blocked/partial результата задачи тестирования укажи evidence "
                    "фактической попытки execution."
                )

        missing = sorted(required_success.difference(answer.evidence))
        if missing:
            return f"Final evidence не содержит обязательные quality-gate call_id: {missing}"
        if answer.status in {"blocked", "partial"}:
            missing_blockers = sorted(required_blockers.difference(answer.evidence))
            if missing_blockers:
                return (
                    "Final evidence не содержит инфраструктурные blocker call_id: "
                    f"{missing_blockers}"
                )
        return None

    def _failure_digest_from_gate_results(
        self, results: list[dict[str, Any]]
    ) -> FailureDigest | None:
        outputs: list[str] = []
        for item in results:
            result = item.get("result")
            if not isinstance(result, dict):
                continue
            output = result.get("output")
            if isinstance(output, str) and output:
                outputs.append(output)
        if not outputs:
            return None
        return analyze_test_output(
            "\n".join(outputs), project=self.project_context.active_project
        )

    def _verification_payload(
        self,
        results: list[dict[str, Any]],
        *,
        label: str,
    ) -> tuple[dict[str, Any], bool]:
        successful, code_failures, infrastructure_failures = (
            self._classify_gate_results(results)
        )
        digest = self._failure_digest_from_gate_results(results)
        all_ok = not code_failures and not infrastructure_failures
        signature = digest.signature if digest is not None else ("green" if all_ok else None)
        if self.incident.state is not None:
            self.incident.record_verification(signature=signature, ok=all_ok)
        payload: dict[str, Any] = {
            "type": label,
            "all_ok": all_ok,
            "successful": sorted(successful),
            "code_failures": sorted(code_failures),
            "infrastructure_failures": sorted(infrastructure_failures),
            "results": results,
        }
        if digest is not None:
            payload["failure_digest"] = {
                "signature": digest.signature,
                "failed_tests": digest.failed_tests,
                "relevant_files": digest.relevant_files,
                "exceptions": digest.exceptions,
                "summary": digest.summary,
            }
        return payload, all_ok

    def _run_safety_officer(
        self,
        *,
        goal: str,
        project_contract: str,
        evidence: dict[str, dict[str, Any]],
    ) -> SafetyVerdict:
        if self.incident.state is None:
            return SafetyVerdict(clear=True, findings=[])
        changed_files = (
            list(self.task_state.current.files_changed)
            if self.task_state.current is not None
            else []
        )
        diff_text = self.incident.diff_bundle(self.workspace.root, changed_files)
        prompt = build_safety_prompt(
            goal=goal,
            contract=project_contract,
            incident_context=self.incident.render_context(max_chars=12000),
            evidence=evidence,
            changed_files=changed_files,
            diff_text=diff_text,
        )
        model_id = settings.agent.roles.safety_officer
        result = self._router_call(
            model_id=model_id,
            system_prompt=(
                "Ты независимый Safety Officer. Верни только JSON clear/findings. "
                "Не доверяй самоотчёту Executor."
            ),
            messages=[{"role": "user", "content": prompt}],
            profile=settings.agent.generation.reviewer,
        )
        payload = _extract_json_object(result.text)
        if payload is None:
            verdict = SafetyVerdict(
                clear=False, findings=["Safety Officer не вернул валидный JSON"]
            )
        else:
            try:
                verdict = SafetyVerdict.model_validate(payload)
            except ValidationError:
                verdict = SafetyVerdict(
                    clear=False, findings=["Safety Officer вернул невалидную схему"]
                )
        self.incident.safety_verdict(
            clear=verdict.clear, findings=verdict.findings, model=result.raw_model_name
        )
        return verdict

    def _protocol_candidates(self, primary: str) -> list[str]:
        configured = settings.agent.orchestration.protocol_fallback_models
        result: list[str] = []
        for model in [primary, *configured]:
            if model and model not in result:
                result.append(model)
        return result

    @staticmethod
    def _compact_malformed(text: str) -> dict[str, Any]:
        encoded = text.encode("utf-8", errors="replace")
        return {
            "chars": len(text),
            "sha256_16": hashlib.sha256(encoded).hexdigest()[:16],
            "tail": text[-1000:],
        }

    def _checkpoint_failure(
        self,
        *,
        reason: str,
        model: str,
        step: int,
    ) -> None:
        if self.task_state.current is None:
            return
        self.task_state.current.metadata["last_failure"] = {
            "reason": reason,
            "model": model,
            "step": step,
        }
        self.task_state.save()

    def ask(
        self,
        user_query: str,
        tags: list[str] | None = None,
        model_id: str | None = None,
        *,
        force_task: bool = False,
        resume_task: bool = False,
    ) -> str:
        self.project_context.infer_project_from_query(user_query)
        project_contract = self.project_context.render_for_prompt(
            user_query,
            max_chars=settings.agent.memory.project_context_chars,
        )
        task_mode = self._is_task_request(user_query, tags, force_task)
        require_execution = self._requires_execution(user_query, task_mode)
        if require_execution and not self.execution.enabled:
            raise AgentExecutionError(
                "Задача требует реального запуска/тестирования, но execution отключён. "
                "Установите AGENT_ALLOW_CODE_EXECUTION=true или /execution on."
            )

        plan: PlanResponse | None = None
        if task_mode:
            if resume_task and self.task_state.current is not None:
                current = self.task_state.current
                current.status = "executing"
                current.metadata["resume_count"] = int(
                    current.metadata.get("resume_count", 0)
                ) + 1
                current.metadata["resume_query"] = user_query[:2000]
                incident_id = current.metadata.get("incident_id")
                if isinstance(incident_id, str) and incident_id:
                    try:
                        self.incident.load(incident_id)
                    except (OSError, ValueError):
                        self.incident.start(current.goal, self.project_context.active_project)
                else:
                    self.incident.start(current.goal, self.project_context.active_project)
                current.metadata["incident_id"] = self.incident.state.incident_id
                self.task_state.save()
                plan = PlanResponse(
                    goal=current.goal,
                    steps=[step.description for step in current.plan]
                    or ["Продолжить незавершённую задачу"],
                    acceptance_criteria=list(current.acceptance_criteria),
                )
                # Legacy V3 checkpoints may not have an incident yet. Reconstruct
                # the mechanical plan gate from persisted TaskState before execution.
                gate_ok, _ = self.incident.mutation_gate()
                if self.incident.state.kind != IncidentKind.TRIVIAL and not gate_ok:
                    self.incident.set_objectives(plan.goal, plan.acceptance_criteria)
                    self.incident.write_iap(plan.steps, plan.acceptance_criteria)
                    if settings.agent.incident.approval_mode == "kernel":
                        self.incident.approve_iap(actor="kernel-resume")
                    else:
                        self.incident.block("IAP ожидает утверждения владельцем")
                        raise AgentExecutionError(
                            "Resume создал IAP, но approval_mode=human."
                        )
                self.incident.begin_execution()
            else:
                self.task_state.start(user_query)
                self.incident.start(user_query, self.project_context.active_project)
                if self.task_state.current is not None:
                    self.task_state.current.metadata["incident_id"] = (
                        self.incident.state.incident_id
                    )
                    self.task_state.save()
                if self.incident.state.kind == IncidentKind.TRIVIAL:
                    plan = PlanResponse(
                        goal=user_query,
                        steps=["Выполнить локальную правку и проверить результат"],
                        acceptance_criteria=["Изменение подтверждено tool_result"],
                    )
                else:
                    plan = (
                        self._create_plan(user_query, project_contract)
                        if self.planner_enabled
                        else PlanResponse(goal=user_query, steps=["Выполнить задачу"])
                    )
                self.task_state.set_plan(plan.steps, plan.acceptance_criteria)
                self.incident.set_objectives(plan.goal, plan.acceptance_criteria)
                if self.incident.state.kind != IncidentKind.TRIVIAL:
                    self.incident.write_iap(plan.steps, plan.acceptance_criteria)
                    if settings.agent.incident.approval_mode == "kernel":
                        self.incident.approve_iap(actor="kernel")
                    else:
                        self.incident.block("IAP ожидает утверждения владельцем")
                        raise AgentExecutionError(
                            "IAP создан, но approval_mode=human: требуется внешнее утверждение."
                        )
                self.incident.begin_execution()
                console_logger.info(
                    "[INCIDENT] %s | kind=%s | project=%s | IAP approved",
                    self.incident.state.incident_id,
                    self.incident.state.kind.value,
                    self.project_context.active_project,
                )

        task_state_text = (
            self.task_state.current.model_dump_json(indent=2)
            if self.task_state.current is not None
            else ""
        )
        extra_system = self.tools.instructions()
        extra_system += (
            "\n\n=== ACTIVE PROJECT FILE INDEX ===\n"
            + self._workspace_snapshot()[:8000]
        )
        if plan is not None:
            extra_system += "\n\n=== CURRENT TASK PLAN ===\n" + json.dumps(
                plan.model_dump(), ensure_ascii=False, indent=2
            )
        if task_mode and self.incident.state is not None:
            extra_system += "\n\n=== ICS INCIDENT CONTROL ===\n"
            extra_system += self.incident.render_context(max_chars=12000)
            extra_system += (
                "\n\nMECHANICAL RULES: source mutation is blocked unless IAP SHA is approved; "
                "independent read/list calls MUST be batched; after several read-only "
                "rounds the kernel will reject more reconnaissance. Prefer one small "
                "diagnostic batch -> one minimal patch -> automatic verification."
            )
        built = self.prompt_builder.build(
            user_query,
            project_context=project_contract,
            task_state=task_state_text,
            extra_system=extra_system,
        )
        primary_model = model_id or (
            settings.agent.roles.executor if task_mode else choose_model(tags)
        ) or settings.agent.default_model
        model_candidates = self._protocol_candidates(primary_model)
        candidate_index = 0
        current_model = model_candidates[candidate_index]
        system_prompt = built.system_prompt
        messages = list(built.messages)
        history_message_count = max(0, len(messages) - 1)

        evidence: dict[str, dict[str, Any]] = {}
        read_paths: set[str] = set()
        mutation_revision = 0
        gated_revision = 0
        required_evidence: set[str] = set()
        required_blocker_evidence: set[str] = set()
        gates_blocking_failure = False
        protocol_failures_for_model = 0
        reviewer_rejections = 0
        safety_cleared_revision = -1
        max_steps = settings.agent.orchestration.max_steps
        last_model_name = current_model

        if built.dropped_history_exchanges:
            self.events.log(
                "context_compaction",
                task_id=self._task_id(),
                estimated_tokens=built.estimated_tokens,
                dropped_history_exchanges=built.dropped_history_exchanges,
                active_project=self.project_context.active_project,
            )

        # For explicit testing requests, collect objective failures before the first LLM action.
        if (
            task_mode
            and require_execution
            and settings.agent.orchestration.baseline_quality_gates
        ):
            console_logger.info("[BASELINE] запускаю quality gates")
            baseline_results, baseline_ok = self._run_quality_gates(evidence)
            console_logger.info("[BASELINE] %s", "PASS" if baseline_ok else "FAIL")
            if baseline_results:
                gate_success, code_failures, infra_failures = (
                    self._classify_gate_results(baseline_results)
                )
                required_evidence = set()
                required_blocker_evidence = infra_failures
                gates_blocking_failure = bool(code_failures)
                verification_payload, _ = self._verification_payload(
                    baseline_results, label="baseline_quality_gate_results"
                )
                verification_payload["instruction"] = (
                    "Ядро уже выполнило baseline. Работай только по failure_digest и "
                    "релевантным файлам; независимые чтения объединяй в ToolBatch. "
                    "После mutation ядро само повторит проверки. Не делай обзор всего проекта."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(verification_payload, ensure_ascii=False),
                    }
                )

        self.events.log(
            "task_start" if task_mode else "chat_start",
            task_id=self._task_id(),
            model=current_model,
            query=user_query,
        )

        try:
            for step in range(1, max_steps + 1):
                history_message_count = self._trim_loop_messages(
                    system_prompt, messages, history_message_count, evidence
                )
                profile = (
                    settings.agent.generation.repair
                    if protocol_failures_for_model
                    else settings.agent.generation.executor
                )
                if task_mode and self.incident.state is not None:
                    if (
                        self.incident.state.total_llm_rounds
                        >= settings.agent.orchestration.max_total_llm_rounds
                    ):
                        reason = (
                            "Достигнут механический потолок LLM-раундов "
                            f"({settings.agent.orchestration.max_total_llm_rounds})"
                        )
                        self.incident.block(reason)
                        raise AgentExecutionError(reason)
                    rotate = self.incident.record_llm_round()
                    if rotate:
                        handoff = {
                            "evidence": self._compact_evidence_for_review(evidence),
                            "mutation_revision": mutation_revision,
                            "gated_revision": gated_revision,
                            "required_evidence": sorted(required_evidence),
                            "blocking": gates_blocking_failure,
                        }
                        self.incident.rotate_period(
                            "context-window operational period completed", handoff
                        )
                        console_logger.info(
                            "[HANDOFF] новый operational period=%s",
                            self.incident.state.operational_period,
                        )
                        messages = [
                            {"role": "user", "content": user_query},
                            {
                                "role": "user",
                                "content": self.incident.render_context(max_chars=12000)
                                + "\n\nKERNEL HANDOFF:\n"
                                + json.dumps(handoff, ensure_ascii=False, default=str)[:12000],
                            },
                        ]
                        history_message_count = 0
                self.events.log(
                    "llm_request",
                    task_id=self._task_id(),
                    step=step,
                    model=current_model,
                    temperature=profile.temperature,
                    max_tokens=profile.max_tokens,
                )
                try:
                    result = self._router_call(
                        model_id=current_model,
                        system_prompt=system_prompt,
                        messages=messages,
                        profile=profile,
                        allow_fallback=False,
                    )
                except ProviderError as exc:
                    candidate_index += 1
                    if candidate_index >= len(model_candidates):
                        raise
                    previous = current_model
                    current_model = model_candidates[candidate_index]
                    protocol_failures_for_model = 0
                    self.events.log(
                        "provider_model_fallback",
                        task_id=self._task_id(),
                        from_model=previous,
                        to_model=current_model,
                        error=str(exc),
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "type": "provider_fallback",
                                    "from": previous,
                                    "to": current_model,
                                    "error": str(exc),
                                    "instruction": "Продолжи текущий TaskState.",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                last_model_name = result.raw_model_name
                self.events.log(
                    "llm_response",
                    task_id=self._task_id(),
                    step=step,
                    model=current_model,
                    text=result.text,
                    usage_tokens=result.usage_tokens,
                    finish_reason=result.finish_reason,
                )

                try:
                    finish = (result.finish_reason or "").casefold()
                    truncation_markers = ("truncat", "max_token", "length", "token_limit")
                    if any(marker in finish for marker in truncation_markers):
                        raise ProtocolError(
                            f"Provider завершил ответ по лимиту: {result.finish_reason}",
                            code="truncated_json",
                            likely_truncated=True,
                        )
                    parsed = parse_agent_response(result.text)
                except ProtocolError as exc:
                    protocol_failures_for_model += 1
                    self.events.log(
                        "protocol_error",
                        task_id=self._task_id(),
                        step=step,
                        model=current_model,
                        code=exc.code,
                        likely_truncated=exc.likely_truncated,
                        malformed=self._compact_malformed(result.text),
                    )
                    limit = settings.agent.orchestration.protocol_repairs_per_model
                    if protocol_failures_for_model > limit:
                        candidate_index += 1
                        if candidate_index >= len(model_candidates):
                            reason = (
                                "Все protocol-fallback модели исчерпаны. "
                                f"Последняя ошибка: {exc}"
                            )
                            self._checkpoint_failure(
                                reason=reason,
                                model=current_model,
                                step=step,
                            )
                            raise AgentExecutionError(reason) from exc
                        previous = current_model
                        current_model = model_candidates[candidate_index]
                        protocol_failures_for_model = 0
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "type": "protocol_model_fallback",
                                        "from": previous,
                                        "to": current_model,
                                        "reason": exc.code,
                                        "instruction": (
                                            "Продолжи текущий TaskState. Не повторяй "
                                            "большой повреждённый ответ. Используй "
                                            "малые tool-call/PAYLOAD patch."
                                        ),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue

                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "type": "protocol_error",
                                    "code": exc.code,
                                    "error": str(exc),
                                    "malformed": self._compact_malformed(result.text),
                                    "instruction": repair_instruction(exc),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue

                if parsed is not None:
                    protocol_failures_for_model = 0

                if isinstance(parsed, ToolCall):
                    payload, tool_ok, mutated = self._execute_tool(
                        parsed, evidence, read_paths
                    )
                    read_only = parsed.tool in {
                        "workspace.read_file",
                        "workspace.read_lines",
                        "workspace.read_json",
                        "workspace.read_yaml",
                        "workspace.list_files",
                    }
                    objective_progress = mutated or (
                        tool_ok
                        and (
                            parsed.tool.startswith("execution.")
                            or parsed.tool == "terminal.execute"
                        )
                    )
                    if task_mode and self.incident.state is not None:
                        self.incident.record_action(
                            mutated=mutated,
                            read_only=read_only,
                            progress=objective_progress,
                        )
                    if mutated:
                        mutation_revision += 1
                    messages.append(
                        {"role": "assistant", "content": self._action_for_history(parsed)}
                    )
                    messages.append({"role": "user", "content": payload})
                    if (
                        task_mode
                        and mutated
                        and settings.agent.orchestration.auto_verify_after_mutation
                    ):
                        console_logger.info(
                            "[VERIFY] mutation_revision=%s -> quality gates", mutation_revision
                        )
                        console_logger.info(
                            "[VERIFY] mutation_revision=%s -> quality gates", mutation_revision
                        )
                        gate_results, _ = self._run_quality_gates(evidence)
                        gated_revision = mutation_revision
                        gate_success, code_failures, infra_failures = (
                            self._classify_gate_results(gate_results)
                        )
                        required_evidence = gate_success
                        required_blocker_evidence = infra_failures
                        gates_blocking_failure = bool(code_failures)
                        verification_payload, _ = self._verification_payload(
                            gate_results, label="post_mutation_quality_gate_results"
                        )
                        verification_payload["instruction"] = (
                            "Это автоматическая проверка текущей ревизии. Если есть "
                            "failure_digest — исправляй только указанную корневую причину. "
                            "Не делай обзорные чтения; следующий read набор объединяй в batch."
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    verification_payload, ensure_ascii=False
                                ),
                            }
                        )
                    continue

                if isinstance(parsed, ToolBatch):
                    batch_results: list[dict[str, Any]] = []
                    mutated_any = False
                    batch_all_read_only = bool(parsed.calls)
                    batch_objective_progress = False
                    for call in parsed.calls:
                        payload, tool_ok, mutated = self._execute_tool(
                            call, evidence, read_paths
                        )
                        if mutated:
                            mutation_revision += 1
                            mutated_any = True
                        read_only_call = call.tool in {
                            "workspace.read_file",
                            "workspace.read_lines",
                            "workspace.read_json",
                            "workspace.read_yaml",
                            "workspace.list_files",
                        }
                        batch_all_read_only = batch_all_read_only and read_only_call
                        batch_objective_progress = batch_objective_progress or mutated or (
                            tool_ok
                            and (
                                call.tool.startswith("execution.")
                                or call.tool == "terminal.execute"
                            )
                        )
                        batch_results.append(json.loads(payload))
                    if task_mode and self.incident.state is not None:
                        self.incident.record_action(
                            mutated=mutated_any,
                            read_only=batch_all_read_only,
                            progress=batch_objective_progress,
                        )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "type": "tools",
                                    "calls": [
                                        json.loads(self._action_for_history(call))
                                        for call in parsed.calls
                                    ],
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "type": "tool_batch_results",
                                    "results": batch_results,
                                    "instruction": (
                                        "Продолжай по реальным результатам; неуспешный "
                                        "вызов не считается выполненным. Независимые чтения "
                                        "всегда объединяй в batch."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    if (
                        task_mode
                        and mutated_any
                        and settings.agent.orchestration.auto_verify_after_mutation
                    ):
                        gate_results, _ = self._run_quality_gates(evidence)
                        gated_revision = mutation_revision
                        gate_success, code_failures, infra_failures = (
                            self._classify_gate_results(gate_results)
                        )
                        required_evidence = gate_success
                        required_blocker_evidence = infra_failures
                        gates_blocking_failure = bool(code_failures)
                        verification_payload, _ = self._verification_payload(
                            gate_results, label="post_mutation_quality_gate_results"
                        )
                        verification_payload["instruction"] = (
                            "Ядро автоматически проверило текущую ревизию. Работай "
                            "по failure_digest; не возвращайся к широкому обзору проекта."
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    verification_payload, ensure_ascii=False
                                ),
                            }
                        )
                    continue

                if isinstance(parsed, FinalAnswer):
                    if task_mode and gates_blocking_failure and mutation_revision <= gated_revision:
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "type": "quality_gate_block",
                                        "error": (
                                            "Quality gates всё ещё красные. Прочитай "
                                            "релевантные файлы, исправь код и повтори."
                                        ),
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue

                    if task_mode and mutation_revision > gated_revision:
                        gate_results, gates_ok = self._run_quality_gates(evidence)
                        gated_revision = mutation_revision
                        gate_success, code_failures, infra_failures = (
                            self._classify_gate_results(gate_results)
                        )
                        required_evidence = gate_success
                        required_blocker_evidence = infra_failures
                        gates_blocking_failure = bool(code_failures)
                        if gate_results:
                            if code_failures:
                                instruction = (
                                    "Есть ошибки кода. Исправь их, затем снова "
                                    "завершай задачу для повторного quality-gate run."
                                )
                            elif infra_failures:
                                instruction = (
                                    "Исходники не должны маскировать инфраструктурный "
                                    "блокер. Верни status=blocked/partial и сослаться "
                                    "на blocker evidence."
                                )
                            else:
                                instruction = (
                                    "Проверки пройдены. Верни status=success с evidence "
                                    "этих quality-gate call_id."
                                )
                            messages.append(
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "type": "quality_gate_results",
                                            "all_ok": gates_ok,
                                            "code_failures": sorted(code_failures),
                                            "infrastructure_failures": sorted(
                                                infra_failures
                                            ),
                                            "results": gate_results,
                                            "instruction": instruction,
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue

                    validation_error = self._validate_final_evidence(
                        parsed,
                        evidence,
                        task_mode=task_mode,
                        required_evidence=required_evidence,
                        required_blocker_evidence=required_blocker_evidence,
                        require_execution=require_execution,
                    )
                    if validation_error:
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "type": "final_validation_error",
                                        "error": validation_error,
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        continue

                    if (
                        task_mode
                        and self.reviewer_enabled
                        and self.incident.state is not None
                        and self.incident.state.kind != IncidentKind.TRIVIAL
                        and safety_cleared_revision != mutation_revision
                    ):
                        try:
                            safety = self._run_safety_officer(
                                goal=user_query,
                                project_contract=project_contract,
                                evidence=evidence,
                            )
                        except (ProviderError, RuntimeError) as exc:
                            reason = f"Safety Officer unavailable: {exc}"
                            self.incident.block(reason)
                            self._checkpoint_failure(
                                reason=reason,
                                model=current_model,
                                step=step,
                            )
                            raise AgentExecutionError(
                                "Независимая safety-проверка недоступна; "
                                "инцидент заблокирован до восстановления "
                                "проверяющего провайдера."
                            ) from exc
                        self.events.log(
                            "safety_result",
                            task_id=self._task_id(),
                            clear=safety.clear,
                            findings=safety.findings,
                        )
                        if not safety.clear:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "type": "safety_stop",
                                            "findings": safety.findings,
                                            "instruction": (
                                                "Safety Officer остановил закрытие. "
                                                "Исправь только подтверждённые замечания, "
                                                "повтори проверки и снова запроси закрытие."
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue
                        safety_cleared_revision = mutation_revision

                    if (
                        task_mode
                        and self.reviewer_enabled
                        and reviewer_rejections
                        < settings.agent.orchestration.max_reviewer_rejections
                    ):
                        review = self._review(
                            goal=user_query,
                            plan=plan,
                            answer=parsed,
                            evidence=evidence,
                            project_contract=project_contract,
                        )
                        self.events.log(
                            "review_result",
                            task_id=self._task_id(),
                            approved=review.approved,
                            feedback=review.feedback,
                        )
                        if not review.approved:
                            reviewer_rejections += 1
                            limit = settings.agent.orchestration.max_reviewer_rejections
                            if reviewer_rejections >= limit:
                                reason = (
                                    "Reviewer отклонил результат максимальное число раз: "
                                    + review.feedback[:2000]
                                )
                                self._checkpoint_failure(
                                    reason=reason,
                                    model=current_model,
                                    step=step,
                                )
                                if task_mode:
                                    self.task_state.finish(False)
                                raise AgentExecutionError(reason)
                            messages.append(
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "type": "review_feedback",
                                            "approved": False,
                                            "feedback": review.feedback,
                                            "instruction": (
                                                "Исправь замечания Reviewer и повтори "
                                                "необходимые проверки."
                                            ),
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            continue

                    self._save_exchange(user_query, parsed.content, last_model_name, tags)
                    self._maybe_summarize()
                    if task_mode:
                        self.task_state.finish(True)
                        if self.incident.state is not None:
                            self.incident.close(
                                parsed.content,
                                {
                                    "llm_rounds": self.incident.state.total_llm_rounds,
                                    "operational_periods": self.incident.state.operational_period,
                                    "mutations": self.incident.state.mutation_count,
                                    "verifications": self.incident.state.verification_count,
                                    "safety_stops": self.incident.state.safety_stops,
                                    "evidence_count": len(evidence),
                                },
                            )
                    self.events.log(
                        "task_done" if task_mode else "chat_done",
                        task_id=self._task_id(),
                        evidence=list(evidence),
                    )
                    return parsed.content

                if not task_mode and not evidence:
                    self._save_exchange(user_query, result.text, last_model_name, tags)
                    self._maybe_summarize()
                    return result.text

                protocol_failures_for_model += 1
                limit = settings.agent.orchestration.protocol_repairs_per_model
                if protocol_failures_for_model > limit:
                    candidate_index += 1
                    if candidate_index >= len(model_candidates):
                        reason = (
                            "Все protocol-fallback модели исчерпаны: модель "
                            "продолжает возвращать обычный текст вместо TOOL PROTOCOL V3."
                        )
                        self._checkpoint_failure(
                            reason=reason,
                            model=current_model,
                            step=step,
                        )
                        raise AgentExecutionError(reason)
                    previous = current_model
                    current_model = model_candidates[candidate_index]
                    protocol_failures_for_model = 0
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "type": "protocol_model_fallback",
                                    "from": previous,
                                    "to": current_model,
                                    "reason": "plain_text_in_task",
                                    "instruction": (
                                        "Продолжи текущий TaskState и верни только "
                                        "маленький tool/final action TOOL PROTOCOL V3."
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "type": "protocol_error",
                                "code": "invalid_json",
                                "error": "Engineering task требует TOOL PROTOCOL V3.",
                                "instruction": (
                                    "Верни один маленький tool/final action. Для кода "
                                    "используй raw PAYLOAD."
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
        except ProviderError as exc:
            if task_mode:
                self.task_state.finish(False)
            self._checkpoint_failure(reason=str(exc), model=current_model, step=step)
            self.events.log("provider_error", task_id=self._task_id(), error=str(exc))
            raise

        if task_mode:
            self.task_state.finish(False)
        error_text = f"Агент превысил лимит шагов ({max_steps}) и остановлен."
        self._checkpoint_failure(reason=error_text, model=current_model, step=max_steps)
        self._save_exchange(user_query, error_text, last_model_name, tags)
        raise AgentExecutionError(error_text)

    def resume(
        self,
        note: str = "",
        *,
        tags: list[str] | None = None,
        model_id: str | None = None,
    ) -> str:
        """Resume the persisted unfinished TaskState without discarding its contract."""
        current = self.task_state.current
        if current is None or current.status == "done":
            raise AgentExecutionError("Нет незавершённой задачи для /resume")
        continuation = current.goal
        if note.strip():
            continuation += "\n\nДополнительная инструкция при продолжении: " + note.strip()
        else:
            continuation += (
                "\n\nПродолжи с сохранённого checkpoint. Перепроверь текущее состояние "
                "файлов и quality gates перед следующими изменениями."
            )
        return self.ask(
            continuation,
            tags=tags,
            model_id=model_id,
            force_task=True,
            resume_task=True,
        )

    def _save_exchange(
        self,
        user_query: str,
        assistant_response: str,
        model_used: str,
        tags: list[str] | None,
    ) -> None:
        self.memory.append_exchange(
            user_query=user_query,
            assistant_response=assistant_response,
            model_used=model_used,
            tags=tags or [],
        )

    def _maybe_summarize(self) -> None:
        cfg = settings.agent.summarization
        if not cfg.enabled:
            return
        count = self.memory.messages_count()
        checkpoint = min(self.memory.summary_checkpoint(), count)
        if count - checkpoint >= cfg.trigger_after_messages:
            self._summarize_history(checkpoint=checkpoint, current_count=count)

    @staticmethod
    def _clip_summary_exchange(text: str, limit: int = 5000) -> str:
        if len(text) <= limit:
            return text
        head = int(limit * 0.75)
        tail = limit - head
        return text[:head] + "\n...[exchange clipped]...\n" + text[-tail:]

    def _summarize_history(
        self,
        *,
        checkpoint: int | None = None,
        current_count: int | None = None,
    ) -> None:
        start_index = self.memory.summary_checkpoint() if checkpoint is None else checkpoint
        count = self.memory.messages_count() if current_count is None else current_count
        new_messages = self.memory.messages_from(start_index)
        if not new_messages:
            self.memory.mark_summary_checkpoint(count)
            return

        previous_summary = self.memory.load_summary()
        chunks: list[str] = []
        if previous_summary:
            chunks.append("ПРЕДЫДУЩЕЕ РЕЗЮМЕ:\n" + previous_summary[:8000])
        for message in new_messages:
            exchange = (
                f"Пользователь: {message.user_query}\n"
                f"Ассистент: {message.assistant_response}"
            )
            chunks.append(self._clip_summary_exchange(exchange))
        history_text = "\n\n".join(chunks)
        if len(history_text) > 40000:
            history_text = history_text[-40000:]
            history_text = "...[older summary input clipped]...\n" + history_text

        prompt = settings.prompts.summarization_prompt.format(history_text=history_text)
        try:
            provider, real_model_name = resolve_model_id(
                settings.agent.summarization.summarizer_model
            )
            if not provider.is_configured():
                return
            profile = settings.agent.generation.summarizer
            result = provider.complete(
                model_name=real_model_name,
                system_prompt="Сделай краткое точное rolling-резюме.",
                messages=[{"role": "user", "content": prompt}],
                temperature=profile.temperature,
                max_tokens=profile.max_tokens,
            )
            self.memory.overwrite_summary(result.text)
            self.memory.mark_summary_checkpoint(count)
        except ProviderError:
            return

