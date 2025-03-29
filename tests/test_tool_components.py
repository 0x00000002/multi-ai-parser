import pytest
from unittest.mock import Mock
from src.ai.tools.tool_executor import ToolExecutor
from src.ai.tools.tool_call_parser import ToolCallParser
from src.ai.tools.tool_prompt_builder import ToolPromptBuilder
from src.ai.tools.tools_list import Tool
from src.Logger import Logger

@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)

class TestToolExecutor:
    def test_tool_execution_success(self, mock_logger):
        executor = ToolExecutor(mock_logger)
        result = executor.execute("get_ticket_price", '{"destination_city": "New York"}')
        assert result is not None
        mock_logger.info.assert_called_once()

    def test_tool_execution_invalid_tool(self, mock_logger):
        executor = ToolExecutor(mock_logger)
        result = executor.execute("invalid_tool", "{}")
        assert result is None
        mock_logger.warning.assert_called_once()

    def test_tool_execution_error(self, mock_logger):
        executor = ToolExecutor(mock_logger)
        result = executor.execute("get_ticket_price", "invalid json")
        assert result is None
        mock_logger.error.assert_called_once()

class TestToolCallParser:
    def test_parse_tool_call_json(self, mock_logger):
        parser = ToolCallParser(mock_logger)
        content = 'TOOL_CALL: {"name": "get_ticket_price", "arguments": {"destination_city": "New York"}}'
        result = parser.parse_tool_call(content)
        assert result is not None
        assert result["name"] == "get_ticket_price"
        assert result["arguments"]["destination_city"] == "New York"

    def test_parse_tool_call_invalid_json(self, mock_logger):
        parser = ToolCallParser(mock_logger)
        content = 'TOOL_CALL: invalid json'
        result = parser.parse_tool_call(content)
        assert result is None
        mock_logger.error.assert_called_once()
        assert "Error processing tool call from text" in mock_logger.error.call_args[0][0]
        assert "JSONDecodeError" in mock_logger.error.call_args[0][0]

    def test_parse_tool_call_no_match(self, mock_logger):
        parser = ToolCallParser(mock_logger)
        content = "No tool call here"
        result = parser.parse_tool_call(content)
        assert result is None

class TestToolPromptBuilder:
    def test_build_tools_section(self):
        tools = [Tool.TICKET_ORACLE]
        section = ToolPromptBuilder.build_tools_section(tools)
        assert "Available tools for this request" in section
        assert "get_ticket_price" in section
        assert "TOOL_CALL:" in section

    def test_build_enhanced_prompt(self):
        tools = [Tool.TICKET_ORACLE]
        user_prompt = "How much is a ticket to New York?"
        enhanced = ToolPromptBuilder.build_enhanced_prompt(user_prompt, tools)
        assert user_prompt in enhanced
        assert "Available tools for this request" in enhanced
        assert "get_ticket_price" in enhanced 