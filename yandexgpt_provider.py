"""YandexGPT Foundation Models completion provider."""
from __future__ import annotations

import os
from typing import Any

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error


class YandexGPTProvider(BaseProvider):
    provider_name = "yandexgpt"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        folder_id_env: str | None = None,
    ) -> None:
        super().__init__(api_key, base_url)
        self.folder_id = os.environ.get(folder_id_env) if folder_id_env else None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.folder_id and self.base_url)

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
                "YandexGPT: YANDEX_API_KEY/YANDEX_FOLDER_ID/base_url не заданы",
                retryable=False,
            )

        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        yandex_messages = [{"role": "system", "text": system_prompt}]
        for message in messages:
            role = "assistant" if message.get("role") == "assistant" else "user"
            yandex_messages.append({"role": role, "text": message.get("content", "")})

        payload = {
            "modelUri": f"gpt://{self.folder_id}/{model_name}",
            "completionOptions": {
                "stream": False,
                "temperature": max(0.0, min(float(temperature), 1.0)),
                "maxTokens": max(256, min(int(max_tokens), 65536)),
            },
            "messages": yandex_messages,
        }
        try:
            response = httpx.post(
                f"{self.base_url}/completion",
                headers=headers,
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("YandexGPT", exc) from exc

        data = json_object(response, "YandexGPT")
        try:
            result: Any = data["result"]
            alternative: Any = result["alternatives"][0]
            text = alternative["message"]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "YandexGPT: неожиданный формат ответа",
                retryable=False,
            ) from exc
        if not isinstance(text, str):
            raise ProviderError("YandexGPT: message.text не строка", retryable=False)

        usage = result.get("usage") if isinstance(result, dict) else {}
        if not isinstance(usage, dict):
            usage = {}
        status = alternative.get("status") if isinstance(alternative, dict) else None
        return CompletionResult(
            text=text,
            raw_model_name=model_name,
            usage_tokens=as_optional_int(usage.get("totalTokens")),
            finish_reason=str(status) if status is not None else None,
        )
