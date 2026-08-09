"""Google Gemini generateContent provider."""
from __future__ import annotations

from typing import Any

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error


class GeminiProvider(BaseProvider):
    provider_name = "gemini"

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
            raise ProviderError("GEMINI_API_KEY/base_url не заданы", retryable=False)

        url = f"{self.base_url}/models/{model_name}:generateContent?key={self.api_key}"
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = "model" if message.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.get("content", "")} ]})
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            response = httpx.post(url, json=payload, timeout=60)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("Gemini", exc) from exc

        data = json_object(response, "Gemini")
        try:
            candidate: Any = data["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini: неожиданный формат ответа", retryable=False) from exc
        if not text:
            raise ProviderError("Gemini: пустой текст ответа", retryable=False)

        usage = data.get("usageMetadata") if isinstance(data.get("usageMetadata"), dict) else {}
        finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
        return CompletionResult(
            text=text,
            raw_model_name=model_name,
            usage_tokens=as_optional_int(usage.get("totalTokenCount")),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )
