from core.tool_protocol import FinalAnswer, ToolCall, parse_agent_response


def test_parse_tool_call() -> None:
    parsed = parse_agent_response(
        '{"type":"tool","tool":"workspace.read_file",'
        '"arguments":{"path":"a.py"},"call_id":"x"}'
    )
    assert isinstance(parsed, ToolCall)
    assert parsed.tool == "workspace.read_file"
    assert parsed.arguments == {"path": "a.py"}
    assert parsed.call_id == "x"


def test_parse_fenced_final_answer() -> None:
    parsed = parse_agent_response(
        '```json\n{"type":"final","content":"done","evidence":[]}\n```'
    )
    assert parsed == FinalAnswer(content="done", evidence=[])


def test_plain_text_is_not_forced_into_protocol() -> None:
    assert parse_agent_response("обычный ответ") is None
