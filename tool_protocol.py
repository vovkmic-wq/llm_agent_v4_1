"""Совместимый импорт старого имени модуля протокола."""
from core.protocol import (  # noqa: F401
    FinalAnswer,
    ProtocolError,
    ToolBatch,
    ToolCall,
    format_tool_result,
    parse_agent_response,
)
