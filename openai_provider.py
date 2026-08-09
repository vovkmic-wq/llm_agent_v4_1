"""OpenAI Chat Completions provider."""
from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error


class OpenAIProvider(BaseProvider):
    provider_name = "openai"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

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
            raise ProviderError("OPENAI_API_KEY/base_url не заданы", retryable=False)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("OpenAI", exc) from exc

        data = json_object(response, "OpenAI")
        try:
            choice: Any = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "OpenAI: неожиданный формат ответа",
                retryable=False,
            ) from exc
        if not isinstance(text, str):
            raise ProviderError("OpenAI: content не является строкой", retryable=False)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return CompletionResult(
            text=text,
            raw_model_name=model_name,
            usage_tokens=as_optional_int(usage.get("total_tokens")),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )
