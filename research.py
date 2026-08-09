"""Ограниченный HTTPS-fetch для публичной документации и исследований."""
from __future__ import annotations

from typing import Any
from urllib.parse import urljoin, urlparse

import httpx


class ResearchError(RuntimeError):
    """Ошибка сетевой политики или HTTP-fetch."""


class ResearchManager:
    _REDIRECT_CODES = {301, 302, 303, 307, 308}

    def __init__(
        self,
        *,
        enabled: bool,
        allowed_domains: list[str] | None = None,
        timeout_seconds: int = 20,
        max_response_bytes: int = 500_000,
        user_agent: str = "LLM-Agent-V3/3.1",
        max_redirects: int = 5,
    ) -> None:
        self.enabled = enabled
        self.allowed_domains = [domain.lower() for domain in (allowed_domains or [])]
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_response_bytes = max(1, int(max_response_bytes))
        self.user_agent = user_agent
        self.max_redirects = max(0, int(max_redirects))

    def _allowed(self, host: str) -> bool:
        normalized = host.lower().rstrip(".")
        return any(
            normalized == domain or normalized.endswith("." + domain)
            for domain in self.allowed_domains
        )

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise ResearchError("Разрешены только HTTPS URL")
        if parsed.username or parsed.password:
            raise ResearchError("Credentials в URL запрещены")
        if parsed.port not in (None, 443):
            raise ResearchError("Разрешён только стандартный HTTPS-порт 443")
        if not self._allowed(parsed.hostname):
            raise ResearchError(f"Домен не входит в allow-list: {parsed.hostname}")

    @staticmethod
    def _decode(raw: bytes, content_type: str) -> str:
        # Keep decoding deterministic and never fail the entire agent on a bad page charset.
        charset = "utf-8"
        for part in content_type.split(";")[1:]:
            key, _, value = part.strip().partition("=")
            if key.casefold() == "charset" and value.strip():
                charset = value.strip().strip('"\'')
                break
        try:
            return raw.decode(charset, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")

    def fetch_url(self, url: str) -> dict[str, Any]:
        if not self.enabled:
            raise ResearchError("Research tools отключены конфигурацией")
        self._validate_url(url)

        current_url = url
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=self.timeout_seconds,
                headers={"User-Agent": self.user_agent},
            ) as client:
                for redirect_count in range(self.max_redirects + 1):
                    self._validate_url(current_url)
                    with client.stream("GET", current_url) as response:
                        if response.status_code in self._REDIRECT_CODES:
                            location = response.headers.get("location")
                            if not location:
                                raise ResearchError("HTTP redirect без Location")
                            if redirect_count >= self.max_redirects:
                                raise ResearchError("Превышен лимит HTTP redirects")
                            next_url = urljoin(str(response.url), location)
                            # Validate *before* making the redirected request.
                            self._validate_url(next_url)
                            current_url = next_url
                            continue

                        response.raise_for_status()
                        content_type = response.headers.get("content-type", "")
                        if not any(
                            kind in content_type.casefold()
                            for kind in ("text/", "json", "xml", "html")
                        ):
                            raise ResearchError(
                                f"Неподдерживаемый Content-Type: {content_type}"
                            )

                        raw = bytearray()
                        for chunk in response.iter_bytes():
                            raw.extend(chunk)
                            if len(raw) > self.max_response_bytes:
                                raise ResearchError("Ответ превышает лимит размера")
                        raw_bytes = bytes(raw)
                        return {
                            "url": str(response.url),
                            "status_code": response.status_code,
                            "content_type": content_type,
                            "text": self._decode(raw_bytes, content_type),
                            "bytes": len(raw_bytes),
                        }
        except ResearchError:
            raise
        except httpx.HTTPError as exc:
            raise ResearchError(f"HTTP ошибка: {exc}") from exc

        raise ResearchError("Не удалось получить URL")

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name != "research.fetch_url":
            raise ResearchError(f"Неизвестный research-инструмент: {tool_name}")
        try:
            return self.fetch_url(**arguments)
        except TypeError as exc:
            raise ResearchError(f"Некорректные аргументы {tool_name}: {exc}") from exc

    @staticmethod
    def tool_specs() -> list[dict[str, Any]]:
        return [{"name": "research.fetch_url", "arguments": {"url": "https URL"}}]
