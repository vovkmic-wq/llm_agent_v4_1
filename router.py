"""Model routing, provider retry and explicit protocol-fallback support."""
from __future__ import annotations

import time

from core.config import settings
from core.logger import console_logger, log_request
from providers.anthropic_provider import AnthropicProvider
from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.deepseek_provider import DeepSeekProvider
from providers.gemini_provider import GeminiProvider
from providers.gigachat_provider import GigaChatProvider
from providers.ollama_provider import OllamaProvider
from providers.openai_provider import OpenAIProvider
from providers.qwen_provider import QwenProvider
from providers.yandexgpt_provider import YandexGPTProvider

_PROVIDER_CLASSES: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "ollama": OllamaProvider,
    "qwen": QwenProvider,
}
_provider_instances: dict[str, BaseProvider] = {}


def clear_provider_cache() -> None:
    """Drop cached provider objects after configuration/environment reload."""
    _provider_instances.clear()


def _build_provider(provider_name: str) -> BaseProvider:
    if provider_name in _provider_instances:
        return _provider_instances[provider_name]
    entry = settings.models.providers.get(provider_name)
    if entry is None:
        raise ProviderError(f"Провайдер '{provider_name}' не описан в models.yaml")
    if not entry.enabled:
        raise ProviderError(f"Провайдер '{provider_name}' отключён в models.yaml")

    api_key = settings.provider_api_key(provider_name)
    base_url = settings.provider_base_url(provider_name)
    if provider_name == "gigachat":
        instance: BaseProvider = GigaChatProvider(api_key, base_url, entry.auth_url)
    elif provider_name == "yandexgpt":
        instance = YandexGPTProvider(api_key, base_url, entry.folder_id_env)
    elif provider_name == "ollama":
        instance = OllamaProvider(api_key, (base_url or "").rstrip("/") + "/v1")
    else:
        provider_cls = _PROVIDER_CLASSES.get(provider_name)
        if provider_cls is None:
            raise ProviderError(f"Нет реализации провайдера '{provider_name}'")
        instance = provider_cls(api_key, base_url)
    _provider_instances[provider_name] = instance
    return instance


def resolve_model_id(model_id: str) -> tuple[BaseProvider, str]:
    if ":" not in model_id:
        raise ProviderError(
            f"Некорректный формат модели: '{model_id}' (ожидается provider:alias)"
        )
    provider_name, alias = model_id.split(":", 1)
    entry = settings.models.providers.get(provider_name)
    if entry is None:
        raise ProviderError(f"Провайдер '{provider_name}' не найден в models.yaml")
    real_model_name = entry.models.get(alias)
    if real_model_name is None:
        raise ProviderError(f"Алиас '{alias}' не найден у провайдера '{provider_name}'")
    return _build_provider(provider_name), real_model_name


def choose_model(tags: list[str] | None = None) -> str:
    tags = tags or []
    for rule in settings.models.routing.rules:
        rule_tags = rule.match.get("tags", [])
        if any(tag in tags for tag in rule_tags):
            return rule.model
    return settings.models.routing.default


class Router:
    def __init__(self) -> None:
        self.routing_cfg = settings.models.routing

    def candidates(
        self,
        primary: str,
        *,
        protocol_chain: list[str] | None = None,
    ) -> list[str]:
        chain = protocol_chain or self.routing_cfg.fallback_chain
        result: list[str] = []
        for model in [primary, *chain]:
            if model and model not in result:
                result.append(model)
        return result

    def call(
        self,
        *,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.15,
        max_tokens: int = 8192,
        allow_fallback: bool = True,
    ) -> CompletionResult:
        candidates = (
            self.candidates(model_id) if allow_fallback else [model_id]
        )
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                return self._call_with_retry(
                    candidate,
                    system_prompt,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except ProviderError as exc:
                console_logger.warning(
                    "Модель '%s' недоступна (%s), пробуем следующую", candidate, exc
                )
                last_error = exc
        raise ProviderError(
            "Все модели из fallback-цепочки недоступны. "
            f"Последняя ошибка: {last_error}"
        )

    def _call_with_retry(
        self,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> CompletionResult:
        max_attempts = int(self.routing_cfg.retry.get("max_attempts", 3))
        backoff = float(self.routing_cfg.retry.get("backoff_seconds", 2))
        provider, real_model_name = resolve_model_id(model_id)
        if not provider.is_configured():
            raise ProviderError(
                f"Провайдер '{provider.provider_name}' не настроен (нет ключа/URL)"
            )

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            start = time.monotonic()
            try:
                result = provider.complete(
                    model_name=real_model_name,
                    system_prompt=system_prompt,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                log_request(
                    model=model_id,
                    query=messages[-1]["content"] if messages else "",
                    success=True,
                    duration_seconds=time.monotonic() - start,
                )
                return result
            except ProviderError as exc:
                last_error = exc
                log_request(
                    model=model_id,
                    query=messages[-1]["content"] if messages else "",
                    success=False,
                    duration_seconds=time.monotonic() - start,
                    error=str(exc),
                )
                if not exc.retryable:
                    raise
                if attempt < max_attempts:
                    delay = min(backoff * (2 ** (attempt - 1)), 30.0)
                    time.sleep(delay)
        raise ProviderError(
            f"'{model_id}' не ответил после {max_attempts} попыток: {last_error}",
            retryable=getattr(last_error, "retryable", True),
        )
