from __future__ import annotations

from contextlib import AbstractContextManager

import pytest

from core.research import ResearchError, ResearchManager


class _FakeResponse(AbstractContextManager):
    def __init__(self, url: str, status_code: int, headers: dict[str, str], body: bytes = b""):
        self.url = url
        self.status_code = status_code
        self.headers = headers
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield self._body


class _RedirectClient(AbstractContextManager):
    calls: list[str] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def stream(self, method: str, url: str):
        type(self).calls.append(url)
        return _FakeResponse(
            url,
            302,
            {"location": "https://evil.example/steal"},
        )


def test_redirect_target_is_validated_before_second_request(monkeypatch) -> None:
    monkeypatch.setattr("core.research.httpx.Client", _RedirectClient)
    manager = ResearchManager(enabled=True, allowed_domains=["docs.example.com"])
    with pytest.raises(ResearchError, match="allow-list"):
        manager.fetch_url("https://docs.example.com/start")
    assert _RedirectClient.calls == ["https://docs.example.com/start"]


def test_research_rejects_credentials_and_nonstandard_port() -> None:
    manager = ResearchManager(enabled=True, allowed_domains=["example.com"])
    with pytest.raises(ResearchError, match="Credentials"):
        manager.fetch_url("https://user:pass@example.com/")
    with pytest.raises(ResearchError, match="443"):
        manager.fetch_url("https://example.com:444/")
