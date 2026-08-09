"""providers/deepseek_provider.py"""
from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    provider_name = "deepseek"
    requires_api_key = True
