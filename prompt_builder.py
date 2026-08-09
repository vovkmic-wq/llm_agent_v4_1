"""Priority-aware prompt assembly with a strict deterministic context budget."""
from __future__ import annotations

from dataclasses import dataclass

from core.config import settings
from core.memory import Exchange, MemoryManager


def rough_token_estimate(text: str) -> int:
    """Conservative tokenizer-free estimate for mixed Russian/English/code text.

    Cyrillic UTF-8 text is typically denser in tokens than plain English, so using
    only ``chars / 4`` systematically underestimates Russian prompts. The maximum of
    a character- and byte-based estimate intentionally leaves safety headroom.
    """
    char_estimate = (len(text) + 2) // 3
    byte_estimate = (len(text.encode("utf-8", errors="replace")) + 3) // 4
    return max(1, char_estimate, byte_estimate)


def _clip(text: str, limit: int, marker: str = "\n...[context clipped]...\n") -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(marker) + 40:
        return text[:limit]
    head = int((limit - len(marker)) * 0.72)
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


@dataclass
class BuiltPrompt:
    system_prompt: str
    messages: list[dict[str, str]]
    estimated_tokens: int
    dropped_history_exchanges: int = 0


class PromptBuilder:
    """Assemble context without letting chat history evict the project contract.

    Budget priority:
      1. kernel/agent rules;
      2. technical specification / active project contract;
      3. current TaskState and tool protocol/file index;
      4. durable decisions/context facts;
      5. summary;
      6. recent raw chat exchanges.
    """

    def __init__(self, memory: MemoryManager) -> None:
        self.memory = memory

    def build(
        self,
        user_query: str,
        *,
        project_context: str = "",
        task_state: str = "",
        extra_system: str = "",
    ) -> BuiltPrompt:
        cfg = settings.agent.memory
        reserve_tokens = min(
            cfg.loop_reserve_tokens,
            max(0, cfg.max_context_tokens // 4),
        )
        target_tokens = max(256, cfg.max_context_tokens - reserve_tokens)
        max_chars = max(4000, target_tokens * 4)
        query_budget = min(max(1200, max_chars // 6), 8000)
        prompt_query = _clip(user_query, query_budget)

        system_budget = max(2500, int(max_chars * 0.82))
        rules_text = "\n".join(f"- {rule}" for rule in settings.agent.rules)
        identity = (
            f"Имя агента: {settings.agent.name}\n"
            f"Роль: {settings.agent.role.strip()}\n\n"
            f"Правила:\n{rules_text}\n\n"
            "Приоритет: безопасность ядра > PROJECT CONTRACT/ТЗ > текущая задача > "
            "project decisions > история чата."
        )
        identity = _clip(identity, min(6500, int(system_budget * 0.20)))

        # Reserve explicit budgets instead of clipping one monolithic prompt at random.
        contract_budget = min(cfg.project_context_chars, int(system_budget * 0.43))
        task_budget = min(5000, int(system_budget * 0.14))
        tools_budget = min(13000, int(system_budget * 0.32))
        used = len(identity) + 100

        contract = _clip(
            project_context or "ACTIVE_PROJECT=(не задан); SPEC_SOURCE=(ТЗ не загружено)",
            max(1200, min(contract_budget, system_budget - used)),
        )
        used += len(contract) + 80
        task = _clip(
            task_state or "(нет активного TaskState)",
            max(500, min(task_budget, system_budget - used)),
        )
        used += len(task) + 80
        tools = _clip(
            extra_system or "(tools/execution context отсутствует)",
            max(800, min(tools_budget, system_budget - used)),
        )
        used += len(tools) + 80

        long_term = self.memory.load_context() if cfg.use_context else ""
        summary = self.memory.load_summary() if cfg.use_summary else ""
        low_priority_budget = max(0, system_budget - used)
        long_budget = min(cfg.long_term_context_chars, low_priority_budget // 2)
        summary_budget = min(cfg.summary_chars, low_priority_budget - long_budget)
        long_term = _clip(long_term, long_budget)
        summary = _clip(summary, summary_budget)

        parts = [
            identity,
            "=== ACTIVE PROJECT CONTRACT / TECHNICAL SPECIFICATION ===\n" + contract,
            "=== CURRENT TASK STATE ===\n" + task,
            "=== TOOL / EXECUTION CONTEXT ===\n" + tools,
        ]
        if long_term:
            parts.append("=== Долговременные заметки ===\n" + long_term)
        if summary:
            parts.append("=== Сжатая история ===\n" + summary)
        system_prompt = "\n\n".join(parts)
        if len(system_prompt) > system_budget:
            system_prompt = _clip(system_prompt, system_budget)

        remaining = max_chars - len(system_prompt) - len(prompt_query) - 100
        recent: list[Exchange] = self.memory.recent_messages(cfg.last_messages)
        kept: list[Exchange] = []
        for exchange in reversed(recent):
            pair_size = len(exchange.user_query) + len(exchange.assistant_response) + 50
            if pair_size > remaining:
                continue
            kept.append(exchange)
            remaining -= pair_size
        kept.reverse()
        messages: list[dict[str, str]] = []
        for exchange in kept:
            messages.append({"role": "user", "content": exchange.user_query})
            messages.append({"role": "assistant", "content": exchange.assistant_response})
        messages.append({"role": "user", "content": prompt_query})

        total = system_prompt + "".join(item["content"] for item in messages)
        # Final guard: raw history is always the first thing sacrificed.
        while rough_token_estimate(total) > target_tokens and len(messages) > 1:
            messages = messages[2:]
            total = system_prompt + "".join(item["content"] for item in messages)
        if rough_token_estimate(total) > target_tokens:
            overflow_chars = rough_token_estimate(total) * 4 - target_tokens * 4
            reduced_limit = max(1000, len(system_prompt) - overflow_chars - 16)
            system_prompt = _clip(system_prompt, reduced_limit)
            total = system_prompt + "".join(item["content"] for item in messages)

        return BuiltPrompt(
            system_prompt=system_prompt,
            messages=messages,
            estimated_tokens=rough_token_estimate(total),
            dropped_history_exchanges=max(
                0,
                len(recent) - max(0, (len(messages) - 1) // 2),
            ),
        )
