"""OpenAI-compatible transport for DeepSeek, Qwen and Ollama."""
from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error


class OpenAICompatibleProvider(BaseProvider):
    provider_name = "openai_compatible"
    requires_api_key = True

    def is_configured(self) -> bool:
        if self.requires_api_key:
            return bool(self.api_key and self.base_url)
        return bool(self.base_url)

    def complete(
        self,
        *,
        model_name: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        temperature: float = 0.15,
        max_tokens: int = 8192,
    ) -> CompletionResult:
        if not self.is_configured():
            raise ProviderError(
                f"{self.provider_name}: не настроен (ключ/base_url отсутствуют)",
                retryable=False,
            )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error(self.provider_name, exc) from exc

        data = json_object(response, self.provider_name)
        try:
            choice: Any = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"{self.provider_name}: неожиданный формат ответа",
                retryable=False,
            ) from exc
        if not isinstance(text, str):
            raise ProviderError(
                f"{self.provider_name}: content не является строкой",
                retryable=False,
            )
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return CompletionResult(
            text=text,
            raw_model_name=model_name,
            usage_tokens=as_optional_int(usage.get("total_tokens")),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )
