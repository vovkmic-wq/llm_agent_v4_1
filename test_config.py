from pathlib import Path

from core.config import PROJECT_DIR, settings


def _model_exists(model_id: str) -> bool:
    if ":" not in model_id:
        return False
    provider_name, alias = model_id.split(":", 1)
    provider = settings.models.providers.get(provider_name)
    return provider is not None and alias in provider.models


def test_all_configured_model_references_exist() -> None:
    references = [
        settings.models.routing.default,
        settings.agent.default_model,
        settings.agent.summarization.summarizer_model,
        settings.agent.roles.planner,
        settings.agent.roles.executor,
        settings.agent.roles.reviewer,
        *settings.models.routing.fallback_chain,
        *(rule.model for rule in settings.models.routing.rules),
    ]
    assert all(_model_exists(model_id) for model_id in references)


def test_env_example_contains_no_prefilled_api_secrets() -> None:
    env_example = (PROJECT_DIR / ".env.example").read_text(encoding="utf-8")
    sensitive_names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "GIGACHAT_CLIENT_ID",
        "GIGACHAT_CLIENT_SECRET",
        "QWEN_API_KEY",
        "YANDEX_API_KEY",
        "YANDEX_FOLDER_ID",
    ]
    values = {}
    for raw_line in env_example.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip()

    assert all(values.get(name, "") == "" for name in sensitive_names)


def test_gitignore_excludes_real_env_file() -> None:
    gitignore = Path(PROJECT_DIR / ".gitignore").read_text(encoding="utf-8")
    assert ".env\n" in gitignore
    assert "!.env.example" in gitignore
