"""providers/ollama_provider.py"""
from __future__ import annotations

from providers.openai_compatible import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    provider_name = "ollama"
    # Ollama отдаёт OpenAI-совместимый эндпоинт по адресу <base_url>/v1
    requires_api_key = False
