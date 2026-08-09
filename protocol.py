"""Robust protocol codec for LLM -> agent-kernel actions.

V3 keeps control metadata in small JSON envelopes and optionally transports large
text/code outside JSON in raw PAYLOAD blocks. This avoids escaping failures and
output truncation when a model edits sizeable source files.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


ProtocolFaultCode = Literal[
    "truncated_json",
    "invalid_json",
    "invalid_schema",
    "missing_payload",
    "ambiguous_response",
]


class ProtocolError(ValueError):
    """A response looked like an agent action but could not be decoded safely."""

    def __init__(
        self,
        message: str,
        *,
        code: ProtocolFaultCode = "invalid_json",
        likely_truncated: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.likely_truncated = likely_truncated


class ToolCall(BaseModel):
    type: Literal["tool"] = "tool"
    tool: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    payload_id: str | None = Field(
        default=None,
        max_length=80,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    )
    payload_argument: str | None = Field(default=None, min_length=1, max_length=80)


class ToolBatch(BaseModel):
    type: Literal["tools"] = "tools"
    calls: list[ToolCall] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_unique_call_ids(self) -> "ToolBatch":
        call_ids = [call.call_id for call in self.calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("В batch каждый call_id должен быть уникальным")
        return self


class FinalAnswer(BaseModel):
    type: Literal["final"] = "final"
    status: Literal["success", "partial", "blocked"] = "success"
    content: str = Field(min_length=1, max_length=50000)
    evidence: list[str] = Field(default_factory=list, max_length=128)


AgentAction = ToolCall | ToolBatch | FinalAnswer


@dataclass(frozen=True)
class DecodedProtocol:
    action: AgentAction | None
    payloads: dict[str, str]


_PAYLOAD_RE = re.compile(
    r"<<<PAYLOAD:(?P<id>[A-Za-z0-9_.:-]{1,80})>>>\r?\n"
    r"(?P<body>.*?)"
    r"\r?\n<<<END_PAYLOAD:(?P=id)>>>",
    re.DOTALL,
)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.I | re.S)


def _extract_payloads(text: str) -> tuple[str, dict[str, str]]:
    payloads: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        payload_id = match.group("id")
        if payload_id in payloads:
            raise ProtocolError(
                f"Дублирующийся PAYLOAD id: {payload_id}",
                code="ambiguous_response",
            )
        payloads[payload_id] = match.group("body")
        return ""

    stripped = _PAYLOAD_RE.sub(replace, text)
    return stripped.strip(), payloads


def _balanced_json_objects(text: str) -> list[str]:
    """Extract complete JSON objects with string/escape awareness."""
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None

    return objects


def _looks_truncated(text: str) -> bool:
    """Heuristic for a model response cut in the middle of a JSON envelope."""
    stripped = text.rstrip()
    if not stripped:
        return False
    if '"type"' not in stripped and '"tool"' not in stripped:
        return False

    depth = 0
    in_string = False
    escaped = False
    saw_open = False
    for char in stripped:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
            saw_open = True
        elif char == "}":
            depth -= 1
    return saw_open and (depth > 0 or in_string)


def _candidate_json_objects(text: str) -> list[str]:
    candidates: list[str] = []
    candidates.extend(match.group(1).strip() for match in _FENCE_RE.finditer(text))
    candidates.extend(_balanced_json_objects(text))

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _bind_payload(call: ToolCall, payloads: dict[str, str]) -> ToolCall:
    if call.payload_id is None:
        return call
    if not call.payload_argument:
        raise ProtocolError(
            "Tool-call с payload_id должен содержать payload_argument",
            code="invalid_schema",
        )
    if call.payload_id not in payloads:
        raise ProtocolError(
            f"PAYLOAD '{call.payload_id}' не найден",
            code="missing_payload",
        )
    arguments = dict(call.arguments)
    if call.payload_argument in arguments:
        raise ProtocolError(
            f"Аргумент '{call.payload_argument}' задан и в JSON, и через PAYLOAD",
            code="ambiguous_response",
        )
    arguments[call.payload_argument] = payloads[call.payload_id]
    return call.model_copy(update={"arguments": arguments})


def decode_agent_response(text: str) -> DecodedProtocol:
    """Decode final/tool/batch actions and attach raw payloads when requested."""
    control_text, payloads = _extract_payloads(text)
    validation_errors: list[str] = []
    tools: list[ToolCall] = []
    finals: list[FinalAnswer] = []
    saw_protocol_type = False

    for candidate in _candidate_json_objects(control_text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            validation_errors.append(str(exc))
            continue
        if not isinstance(payload, dict):
            continue
        response_type = payload.get("type")
        if response_type not in {"tool", "tools", "final"}:
            continue
        saw_protocol_type = True
        try:
            if response_type == "tool":
                tools.append(_bind_payload(ToolCall.model_validate(payload), payloads))
            elif response_type == "tools":
                batch = ToolBatch.model_validate(payload)
                tools.extend(_bind_payload(call, payloads) for call in batch.calls)
            else:
                finals.append(FinalAnswer.model_validate(payload))
        except (ValidationError, ProtocolError) as exc:
            if isinstance(exc, ProtocolError):
                raise
            validation_errors.append(str(exc))

    if tools and finals:
        raise ProtocolError(
            "Ответ одновременно содержит tool-call и final",
            code="ambiguous_response",
        )
    if len(finals) > 1:
        raise ProtocolError(
            "Ответ содержит несколько final-объектов",
            code="ambiguous_response",
        )
    if tools:
        # Deduplicate repeated envelopes often produced around fenced + plain JSON.
        unique: list[ToolCall] = []
        fingerprints: set[str] = set()
        for call in tools:
            fingerprint = json.dumps(
                {"tool": call.tool, "arguments": call.arguments, "call_id": call.call_id},
                sort_keys=True,
                ensure_ascii=False,
            )
            if fingerprint not in fingerprints:
                fingerprints.add(fingerprint)
                unique.append(call)
        action: AgentAction = unique[0] if len(unique) == 1 else ToolBatch(calls=unique)
        return DecodedProtocol(action=action, payloads=payloads)
    if finals:
        return DecodedProtocol(action=finals[0], payloads=payloads)

    protocol_hint = (
        '"type"' in control_text
        and (
            '"tool"' in control_text
            or '"final"' in control_text
            or "workspace." in control_text
            or "execution." in control_text
            or "terminal.execute" in control_text
        )
    )
    if _looks_truncated(control_text):
        raise ProtocolError(
            "Tool JSON выглядит обрезанным до закрывающих кавычек/скобок",
            code="truncated_json",
            likely_truncated=True,
        )
    if saw_protocol_type:
        details = validation_errors[-1] if validation_errors else "schema validation failed"
        raise ProtocolError(
            f"Некорректная схема tool-протокола: {details}",
            code="invalid_schema",
        )
    if protocol_hint:
        details = validation_errors[-1] if validation_errors else "не удалось разобрать JSON"
        raise ProtocolError(
            f"Некорректный JSON tool-протокола: {details}",
            code="invalid_json",
        )
    if payloads:
        raise ProtocolError(
            "Получен PAYLOAD без валидного tool-call envelope",
            code="missing_payload",
        )
    return DecodedProtocol(action=None, payloads={})


def parse_agent_response(text: str) -> AgentAction | None:
    """Compatibility wrapper used by existing callers/tests."""
    return decode_agent_response(text).action


def repair_instruction(error: ProtocolError) -> str:
    """Return a concise deterministic recovery instruction for the model."""
    if error.code == "truncated_json":
        return (
            "Предыдущий action был обрезан. НЕ повторяй большой файл внутри JSON. "
            "Верни один маленький tool-call. Для большого кода используй raw PAYLOAD "
            "с workspace.replace_lines/write_file и payload_argument='content'. "
            "Размер одного изменения держи небольшим."
        )
    if error.code == "missing_payload":
        return (
            "Исправь ссылку на PAYLOAD: payload_id должен совпадать с блоком "
            "<<<PAYLOAD:id>>> ... <<<END_PAYLOAD:id>>>."
        )
    if error.code == "invalid_schema":
        return "Исправь поля envelope согласно TOOL PROTOCOL V3; верни только action."
    return (
        "Верни ровно один валидный action TOOL PROTOCOL V3 без пояснений. "
        "Не помещай большой исходный код внутрь JSON."
    )


def format_tool_result(
    *,
    call_id: str,
    tool: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    source: str = "tool",
    ok: bool | None = None,
) -> str:
    payload: dict[str, Any] = {
        "type": "tool_result",
        "call_id": call_id,
        "tool": tool,
        "ok": (error is None) if ok is None else ok,
        "source": source,
    }
    if result is not None:
        payload["result"] = result
    elif error is None:
        payload["result"] = {}
    if error is not None:
        payload["error"] = error
    return json.dumps(payload, ensure_ascii=False)
