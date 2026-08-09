"""providers/qwen_provider.py"""
from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class QwenProvider(OpenAICompatibleProvider):
    provider_name = "qwen"
    requires_api_key = True
