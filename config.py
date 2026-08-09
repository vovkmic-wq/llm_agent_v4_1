"""Загрузка .env и строго типизированной YAML-конфигурации Agent V4."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("AGENT_CONFIG_DIR", PROJECT_DIR / "config")).expanduser()
_default_env = Path.cwd() / ".env"
ENV_FILE = Path(
    os.environ.get(
        "AGENT_ENV_FILE",
        _default_env if _default_env.exists() else PROJECT_DIR / ".env",
    )
).expanduser()
load_dotenv(ENV_FILE, override=False)
STATE_DIR = Path(
    os.environ.get("AGENT_STATE_DIR", Path.home() / ".llm_agent_v4")
).expanduser().resolve()


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "да"}


class MemoryConfig(BaseModel):
    use_context: bool = True
    use_summary: bool = True
    last_messages: int = 8
    max_context_tokens: int = 8000
    loop_reserve_tokens: int = 2000
    project_context_chars: int = 14000
    summary_chars: int = 4000
    long_term_context_chars: int = 5000


class SummarizationConfig(BaseModel):
    enabled: bool = True
    trigger_after_messages: int = 30
    summarizer_model: str = "yandexgpt:yandex-lite"


class WorkspaceConfig(BaseModel):
    enabled: bool = True
    root_env: str = "AGENT_WORKSPACE_DIR"
    max_file_size_bytes: int = 2_000_000
    max_list_entries: int = 1000
    validate_on_write: bool = True
    protected_globs: list[str] = Field(
        default_factory=lambda: [
            ".env",
            ".env.*",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "secrets.*",
            ".git",
            ".git/*",
            ".git/**",
            ".venv",
            ".venv/*",
            ".venv/**",
            "venv",
            "venv/*",
            "venv/**",
        ]
    )
    ignored_globs: list[str] = Field(
        default_factory=lambda: [
            "__pycache__",
            "__pycache__/*",
            "__pycache__/**",
            ".pytest_cache",
            ".pytest_cache/*",
            ".pytest_cache/**",
            ".mypy_cache",
            ".mypy_cache/*",
            ".mypy_cache/**",
            ".ruff_cache",
            ".ruff_cache/*",
            ".ruff_cache/**",
            ".tox",
            ".tox/*",
            ".tox/**",
            ".nox",
            ".nox/*",
            ".nox/**",
            "node_modules",
            "node_modules/*",
            "node_modules/**",
            "build",
            "build/*",
            "build/**",
            "dist",
            "dist/*",
            "dist/**",
        ]
    )


class ExecutionConfig(BaseModel):
    enabled_env: str = "AGENT_ALLOW_CODE_EXECUTION"
    timeout_seconds: int = 120
    max_output_chars: int = 40_000
    allow_terminal_compat: bool = True


class ResearchConfig(BaseModel):
    enabled_env: str = "AGENT_ALLOW_RESEARCH"
    allowed_domains: list[str] = Field(default_factory=list)
    timeout_seconds: int = 20
    max_response_bytes: int = 500_000


class IncidentConfig(BaseModel):
    enabled: bool = True
    approval_mode: Literal["kernel", "human"] = "kernel"
    max_period_rounds: int = 12
    max_periods: int = 4
    max_plan_revisions: int = 4
    max_safety_stops: int = 4
    max_read_only_rounds: int = 4
    max_no_progress_rounds: int = 3


class OrchestrationConfig(BaseModel):
    planning_enabled: bool = True
    review_enabled: bool = True
    max_steps: int = 60  # compatibility hard ceiling; periods should stop much earlier
    max_total_llm_rounds: int = 36
    auto_verify_after_mutation: bool = True
    batch_reads_required: bool = True
    protocol_repairs_per_model: int = 2
    protocol_fallback_models: list[str] = Field(default_factory=list)
    checkpoint_on_failure: bool = True
    baseline_quality_gates: bool = True
    max_reviewer_rejections: int = 2
    task_keywords: list[str] = Field(
        default_factory=lambda: [
            "создай",
            "разработай",
            "исправь",
            "переработай",
            "рефактор",
            "напиши код",
            "проект",
            "протестируй",
            "файл",
        ]
    )




class GenerationProfileConfig(BaseModel):
    temperature: float = Field(default=0.15, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8192, ge=256, le=65536)


class GenerationConfig(BaseModel):
    planner: GenerationProfileConfig = Field(
        default_factory=lambda: GenerationProfileConfig(temperature=0.1, max_tokens=3000)
    )
    executor: GenerationProfileConfig = Field(
        default_factory=lambda: GenerationProfileConfig(temperature=0.15, max_tokens=8192)
    )
    repair: GenerationProfileConfig = Field(
        default_factory=lambda: GenerationProfileConfig(temperature=0.0, max_tokens=4096)
    )
    reviewer: GenerationProfileConfig = Field(
        default_factory=lambda: GenerationProfileConfig(temperature=0.1, max_tokens=3000)
    )
    summarizer: GenerationProfileConfig = Field(
        default_factory=lambda: GenerationProfileConfig(temperature=0.1, max_tokens=2500)
    )


class QualityGateConfig(BaseModel):
    auto_run: bool = True
    compileall: bool = True
    pytest: bool = True
    ruff: bool = True
    mypy: bool = True


class RolesConfig(BaseModel):
    planner: str = "yandexgpt:yandex-pro"
    executor: str = "yandexgpt:yandex-pro"
    reviewer: str = "yandexgpt:yandex-pro"
    safety_officer: str = "yandexgpt:yandex-pro"


PermissionMode = Literal["auto", "confirm", "deny"]


class PermissionsConfig(BaseModel):
    rules: dict[str, PermissionMode] = Field(
        default_factory=lambda: {
            "workspace.*": "auto",
            "workspace.delete_file": "confirm",
            "execution.*": "auto",
            "terminal.execute": "auto",
            "research.*": "auto",
        }
    )


class AgentTask(BaseModel):
    name: str
    role: str
    rules: list[str] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    incident: IncidentConfig = Field(default_factory=IncidentConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    quality_gates: QualityGateConfig = Field(default_factory=QualityGateConfig)
    roles: RolesConfig = Field(default_factory=RolesConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    default_model: str = "yandexgpt:yandex-pro"
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)


class ModelEntry(BaseModel):
    enabled: bool = True
    env_key: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    auth_url: str | None = None
    folder_id_env: str | None = None
    models: dict[str, str] = Field(default_factory=dict)


class RoutingRule(BaseModel):
    name: str
    match: dict[str, Any]
    model: str


class RoutingConfig(BaseModel):
    rules: list[RoutingRule] = Field(default_factory=list)
    default: str
    fallback_chain: list[str] = Field(default_factory=list)
    retry: dict[str, int] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    providers: dict[str, ModelEntry]
    routing: RoutingConfig


class PromptsConfig(BaseModel):
    system_template: str
    planner_prompt: str
    reviewer_prompt: str
    summarization_prompt: str


class Settings:
    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        self.agent = AgentTask(**_load_yaml("agent.yaml")["agent"])
        self.models = ModelsConfig(**_load_yaml("models.yaml"))
        self.prompts = PromptsConfig(**_load_yaml("prompts.yaml"))

    def provider_api_key(self, provider_name: str) -> str | None:
        entry = self.models.providers.get(provider_name)
        if entry is None or entry.env_key is None:
            return None
        return os.environ.get(entry.env_key)

    def provider_base_url(self, provider_name: str) -> str | None:
        entry = self.models.providers.get(provider_name)
        if entry is None:
            return None
        if entry.base_url_env:
            return os.environ.get(entry.base_url_env, entry.base_url)
        return entry.base_url

    @property
    def code_execution_enabled(self) -> bool:
        return env_bool(self.agent.execution.enabled_env, False)

    @property
    def research_enabled(self) -> bool:
        return env_bool(self.agent.research.enabled_env, False)


settings = Settings()
