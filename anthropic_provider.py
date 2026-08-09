"""Anthropic Messages API provider."""
from __future__ import annotations

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error

API_VERSION = "2023-06-01"


class AnthropicProvider(BaseProvider):
    provider_name = "anthropic"

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
            raise ProviderError("ANTHROPIC_API_KEY/base_url не заданы", retryable=False)

        payload = {
            "model": model_name,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("Anthropic", exc) from exc

        data = json_object(response, "Anthropic")
        content = data.get("content")
        if not isinstance(content, list):
            raise ProviderError("Anthropic: отсутствует content[]", retryable=False)
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
        if not parts:
            raise ProviderError("Anthropic: текстовый content отсутствует", retryable=False)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        input_tokens = as_optional_int(usage.get("input_tokens")) or 0
        output_tokens = as_optional_int(usage.get("output_tokens")) or 0
        stop_reason = data.get("stop_reason")
        return CompletionResult(
            text="".join(parts),
            raw_model_name=model_name,
            usage_tokens=input_tokens + output_tokens,
            finish_reason=str(stop_reason) if stop_reason is not None else None,
        )
