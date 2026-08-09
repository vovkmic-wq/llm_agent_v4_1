"""Independent adversarial Safety Officer review."""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class SafetyVerdict(BaseModel):
    clear: bool
    findings: list[str] = Field(default_factory=list)


def build_safety_prompt(
    *,
    goal: str,
    contract: str,
    incident_context: str,
    evidence: dict[str, dict[str, Any]],
    changed_files: list[str],
    diff_text: str,
) -> str:
    compact_evidence: dict[str, Any] = {}
    for call_id, item in evidence.items():
        result = item.get("result") if isinstance(item, dict) else None
        compact: dict[str, Any] = {
            "tool": item.get("tool") if isinstance(item, dict) else None,
            "ok": bool(item.get("ok")) if isinstance(item, dict) else False,
        }
        if isinstance(result, dict):
            for key in ("path", "exit_code", "cwd", "sha256", "verified", "skipped"):
                if key in result:
                    compact[key] = result[key]
            output = result.get("output")
            if isinstance(output, str) and output:
                compact["output_tail"] = output[-2500:]
        compact_evidence[call_id] = compact
    return (
        "Ты независимый Safety Officer. Твоя задача — попытаться опровергнуть заявление "
        "о готовности. Не доверяй самоотчёту Executor. Проверяй только реальные evidence, "
        "последнюю ревизию проекта и критерии PROJECT CONTRACT. Если есть красные проверки, "
        "непроверенные изменения, расхождение с ТЗ или неразрешённый blocker — clear=false. "
        "Верни только JSON {\"clear\":true|false,\"findings\":[\"...\"]}.\n\n"
        f"GOAL:\n{goal}\n\nPROJECT CONTRACT:\n{contract[:16000]}\n\n"
        f"INCIDENT:\n{incident_context[:12000]}\n\n"
        f"CHANGED FILES:\n{json.dumps(changed_files, ensure_ascii=False)}\n\n"
        f"ACTUAL DIFF:\n{diff_text[:24000]}\n\n"
        f"EVIDENCE:\n{json.dumps(compact_evidence, ensure_ascii=False, default=str)[:24000]}"
    )
