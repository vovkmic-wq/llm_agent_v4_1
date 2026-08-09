import pytest

from core.protocol import FinalAnswer, ProtocolError, ToolCall, parse_agent_response


def test_parses_embedded_fenced_tool_call() -> None:
    text = (
        'Сейчас выполню:\n```json\n{"type":"tool","tool":"workspace.read_file",'
        '"arguments":{"path":"a.py"},"call_id":"c1"}\n```'
    )
    parsed = parse_agent_response(text)
    assert parsed == ToolCall(
        tool="workspace.read_file",
        arguments={"path": "a.py"},
        call_id="c1",
    )


def test_parses_final_with_evidence() -> None:
    parsed = parse_agent_response(
        '{"type":"final","content":"готово","evidence":["c1","c2"]}'
    )
    assert parsed == FinalAnswer(content="готово", evidence=["c1", "c2"])


def test_malformed_tool_like_response_raises_protocol_error() -> None:
    with pytest.raises(ProtocolError):
        parse_agent_response(
            '{"type":"tool","tool":"workspace.write_file","arguments":{"path":"a"}'
        )


def test_plain_text_remains_plain_text() -> None:
    assert parse_agent_response("Обычный ответ без инструментов") is None


def test_protocol_parses_multiple_consecutive_tool_calls() -> None:
    from core.protocol import ToolBatch, parse_agent_response

    text = '''
```json
{"type":"tool","tool":"workspace.read_file","arguments":{"path":"a.py"},"call_id":"a"}
{"type":"tool","tool":"workspace.read_file","arguments":{"path":"b.py"},"call_id":"b"}
```
'''
    parsed = parse_agent_response(text)
    assert isinstance(parsed, ToolBatch)
    assert [call.call_id for call in parsed.calls] == ["a", "b"]
