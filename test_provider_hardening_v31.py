from __future__ import annotations

import base64
import time

import httpx
import pytest

from providers.gigachat_provider import GigaChatProvider
from providers.http_utils import provider_http_error


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_gigachat_builds_basic_key_and_accepts_seconds_expiry(monkeypatch) -> None:
    monkeypatch.delenv("GIGACHAT_AUTH_KEY", raising=False)
    monkeypatch.setenv("GIGACHAT_CLIENT_ID", "client")
    monkeypatch.setenv("GIGACHAT_CLIENT_SECRET", "secret")
    captured: list[dict] = []
    expires_at = int(time.time()) + 1800

    def fake_post(url, **kwargs):
        captured.append({"url": url, **kwargs})
        if "oauth" in url:
            return _Response({"access_token": "token", "expires_at": expires_at})
        return _Response(
            {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 12},
            }
        )

    monkeypatch.setattr("providers.gigachat_provider.httpx.post", fake_post)
    provider = GigaChatProvider(
        None,
        "https://api.giga.chat/v1",
        "https://ngw.example/api/v2/oauth",
    )
    result = provider.complete(
        model_name="GigaChat",
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
    )

    expected = base64.b64encode(b"client:secret").decode("ascii")
    assert captured[0]["headers"]["Authorization"] == f"Basic {expected}"
    assert provider._token_expires_at == float(expires_at)
    assert result.text == "ok"
    assert result.usage_tokens == 12


def test_gigachat_prefers_precomputed_authorization_key(monkeypatch) -> None:
    monkeypatch.setenv("GIGACHAT_AUTH_KEY", "precomputed")
    monkeypatch.delenv("GIGACHAT_CLIENT_ID", raising=False)
    monkeypatch.delenv("GIGACHAT_CLIENT_SECRET", raising=False)
    provider = GigaChatProvider(
        "precomputed",
        "https://api.giga.chat/v1",
        "https://ngw.example/api/v2/oauth",
    )
    assert provider.is_configured() is True
    assert provider._authorization_key() == "precomputed"


def test_http_401_is_nonretryable_but_429_is_retryable() -> None:
    request = httpx.Request("POST", "https://example.com")
    response_401 = httpx.Response(401, request=request)
    response_429 = httpx.Response(429, request=request)
    error_401 = httpx.HTTPStatusError("401", request=request, response=response_401)
    error_429 = httpx.HTTPStatusError("429", request=request, response=response_429)
    assert provider_http_error("test", error_401).retryable is False
    assert provider_http_error("test", error_429).retryable is True
