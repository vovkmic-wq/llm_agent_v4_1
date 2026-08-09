"""GigaChat OAuth + chat/completions provider."""
from __future__ import annotations

import base64
import os
import ssl
import time
import uuid
from typing import Any

import httpx

from providers.base import BaseProvider, CompletionResult, ProviderError
from providers.http_utils import as_optional_int, json_object, provider_http_error


class GigaChatProvider(BaseProvider):
    provider_name = "gigachat"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None,
        auth_url: str | None = None,
    ) -> None:
        super().__init__(api_key, base_url)
        self.auth_url = auth_url.rstrip("/") if auth_url else None
        self.auth_key = os.environ.get("GIGACHAT_AUTH_KEY") or api_key
        self.client_id = os.environ.get("GIGACHAT_CLIENT_ID")
        self.client_secret = os.environ.get("GIGACHAT_CLIENT_SECRET")
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.ca_bundle = os.environ.get("GIGACHAT_CA_BUNDLE")
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    def _authorization_key(self) -> str | None:
        if self.auth_key:
            return self.auth_key.strip()
        if not self.client_id or not self.client_secret:
            return None
        raw = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        return base64.b64encode(raw).decode("ascii")

    def _verify(self) -> bool | ssl.SSLContext:
        if not self.ca_bundle:
            return True
        try:
            return ssl.create_default_context(cafile=self.ca_bundle)
        except (OSError, ssl.SSLError) as exc:
            raise ProviderError(
                f"GigaChat: не удалось загрузить CA bundle: {exc}",
                retryable=False,
            ) from exc

    def is_configured(self) -> bool:
        return bool(
            self.base_url
            and self.auth_url
            and self._authorization_key()
        )

    @staticmethod
    def _normalize_expiry(value: Any) -> float:
        now = time.time()
        if value is None:
            return now + 1800
        try:
            expiry = float(value)
        except (TypeError, ValueError):
            return now + 1800
        # Some API examples historically used milliseconds, current ones may use seconds.
        if expiry > 10_000_000_000:
            expiry /= 1000.0
        if expiry <= now:
            return now + 1800
        return expiry

    def _ensure_token(self) -> str:
        if self._access_token and time.time() < self._token_expires_at - 30:
            return self._access_token
        if not self.auth_url:
            raise ProviderError("GigaChat: auth_url не задан", retryable=False)
        authorization_key = self._authorization_key()
        if not authorization_key:
            raise ProviderError(
                "GigaChat: задайте GIGACHAT_AUTH_KEY либо CLIENT_ID/CLIENT_SECRET",
                retryable=False,
            )

        headers = {
            "Authorization": f"Basic {authorization_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            response = httpx.post(
                self.auth_url,
                headers=headers,
                data={"scope": self.scope},
                timeout=30,
                verify=self._verify(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("GigaChat OAuth", exc) from exc

        data = json_object(response, "GigaChat OAuth")
        token = data.get("access_token")
        if not isinstance(token, str) or not token:
            raise ProviderError(
                "GigaChat OAuth: access_token отсутствует",
                retryable=False,
            )
        self._access_token = token
        self._token_expires_at = self._normalize_expiry(data.get("expires_at"))
        return token

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
            raise ProviderError("GigaChat: OAuth/base_url не настроены", retryable=False)

        token = self._ensure_token()
        payload = {
            "model": model_name,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
                verify=self._verify(),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise provider_http_error("GigaChat", exc) from exc

        data = json_object(response, "GigaChat")
        try:
            choice: Any = data["choices"][0]
            text = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("GigaChat: неожиданный формат ответа", retryable=False) from exc
        if not isinstance(text, str):
            raise ProviderError("GigaChat: content не является строкой", retryable=False)

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        return CompletionResult(
            text=text,
            raw_model_name=model_name,
            usage_tokens=as_optional_int(usage.get("total_tokens")),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )
