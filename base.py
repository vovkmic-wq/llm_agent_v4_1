"""Базовые типы LLM-провайдеров."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Ошибка вызова LLM-провайдера.

    ``retryable`` позволяет Router не тратить повторные попытки на постоянные
    ошибки конфигурации/авторизации/валидации запроса.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class CompletionResult:
    text: str
    raw_model_name: str
    usage_tokens: int | None = None
    finish_reason: str | None = None


class BaseProvider(ABC):
    provider_name: str = "base"

    def __init__(self, api_key: str | None, base_url: str | None) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None

    @abstractmethod
    def is_configured(self) -> bool:
        """True, если есть всё необходимое для вызова."""

    @abstractmethod
    def complete(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.15,
        max_tokens: int = 8192,
    ) -> CompletionResult:
        """Выполнить запрос и вернуть нормализованный результат."""
