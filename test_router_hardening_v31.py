from __future__ import annotations

import pytest

from core import router as router_module
from core.config import settings
from core.router import Router
from providers.base import BaseProvider, CompletionResult, ProviderError


class _Provider(BaseProvider):
    provider_name = "test"

    def __init__(self, error: ProviderError) -> None:
        super().__init__("key", "https://example.com")
        self.error = error
        self.calls = 0

    def is_configured(self) -> bool:
        return True

    def complete(self, **kwargs) -> CompletionResult:
        self.calls += 1
        raise self.error


def test_nonretryable_provider_error_does_not_repeat(monkeypatch) -> None:
    provider = _Provider(ProviderError("bad credentials", retryable=False))
    monkeypatch.setattr(router_module, "resolve_model_id", lambda model_id: (provider, "model"))
    monkeypatch.setattr(settings.models.routing, "retry", {"max_attempts": 5, "backoff_seconds": 0})
    with pytest.raises(ProviderError, match="bad credentials"):
        Router()._call_with_retry(
            "test:model",
            "system",
            [{"role": "user", "content": "hello"}],
            temperature=0.1,
            max_tokens=1000,
        )
    assert provider.calls == 1
