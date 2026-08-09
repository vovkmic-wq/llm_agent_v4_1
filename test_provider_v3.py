from pathlib import Path

from core.agent import Agent
from core.memory import MemoryManager
from core.workspace import WorkspaceManager
from providers.base import CompletionResult
from providers.yandexgpt_provider import YandexGPTProvider


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_yandex_provider_passes_generation_options_and_finish_reason(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setenv("YANDEX_FOLDER_ID", "folder")

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return _Response(
            {
                "result": {
                    "alternatives": [
                        {
                            "message": {"text": "{}"},
                            "status": "ALTERNATIVE_STATUS_TRUNCATED_FINAL",
                        }
                    ],
                    "usage": {"totalTokens": 42},
                }
            }
        )

    monkeypatch.setattr("providers.yandexgpt_provider.httpx.post", fake_post)
    provider = YandexGPTProvider(
        "secret",
        "https://llm.example/foundationModels/v1",
        "YANDEX_FOLDER_ID",
    )
    result = provider.complete(
        model_name="yandexgpt",
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.05,
        max_tokens=10000,
    )
    assert captured["json"]["completionOptions"]["temperature"] == 0.05
    assert captured["json"]["completionOptions"]["maxTokens"] == 10000
    assert result.finish_reason == "ALTERNATIVE_STATUS_TRUNCATED_FINAL"


class TruncatedStatusRouter:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *, model_id: str, system_prompt: str, messages: list[dict[str, str]], **kwargs):
        self.calls += 1
        if self.calls == 1:
            return CompletionResult(
                text='{"type":"tool","tool":"workspace.write_file","arguments":{"path":"a.txt"',
                raw_model_name=model_id,
                finish_reason="ALTERNATIVE_STATUS_TRUNCATED_FINAL",
            )
        if self.calls == 2:
            return CompletionResult(
                text=(
                    '{"type":"tool","tool":"workspace.write_file","arguments":'
                    '{"path":"a.txt","content":"ok"},"call_id":"w"}'
                ),
                raw_model_name=model_id,
            )
        return CompletionResult(
            text='{"type":"final","content":"ok","evidence":["w"]}',
            raw_model_name=model_id,
        )


def test_agent_recovers_from_yandex_truncated_status(tmp_path: Path) -> None:
    agent = Agent(
        memory=MemoryManager(tmp_path / "memory"),
        router=TruncatedStatusRouter(),
        workspace=WorkspaceManager(tmp_path / "workspace"),
        planner_enabled=False,
        reviewer_enabled=False,
        execution_enabled=False,
        research_enabled=False,
    )
    assert agent.ask("Создай a.txt", model_id="test:model", force_task=True) == "ok"
    assert (agent.workspace.root / "a.txt").read_text(encoding="utf-8") == "ok"
