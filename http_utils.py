"""Общие безопасные helpers для HTTP-провайдеров."""
from __future__ import annotations

from typing import Any

import httpx

from providers.base import ProviderError


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def provider_http_error(provider: str, exc: httpx.HTTPError) -> ProviderError:
    """Преобразовать HTTPX-ошибку в ProviderError с retryability."""
    status: int | None = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    retryable = status is None or status in _RETRYABLE_STATUS or status >= 500
    return ProviderError(f"{provider} API ошибка: {exc}", retryable=retryable)


def json_object(response: httpx.Response, provider: str) -> dict[str, Any]:
    """Разобрать JSON-объект и не выпускать ValueError/TypeError наружу."""
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(
            f"{provider}: API вернул невалидный JSON",
            retryable=False,
        ) from exc
    if not isinstance(data, dict):
        raise ProviderError(
            f"{provider}: ожидался JSON object, получен {type(data).__name__}",
            retryable=False,
        )
    return data


def as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
