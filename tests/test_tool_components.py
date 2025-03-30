import pytest
from unittest.mock import Mock, patch, MagicMock
from src.ai.tools.tool_executor import ToolExecutor
from src.ai.tools.tool_call_parser import ToolCallParser
from src.ai.tools.tool_prompt_builder import ToolPromptBuilder
from src.ai.tools.tools_list import Tool
from src.ai.tools.tools_manager import ToolManager
from src.ai.tools.tool_finder import ToolFinder
from src.logger import Logger
from src.ai.ai_config import Model, DEFAULT_TOOL_FINDER_MODEL

@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)

@pytest.fixture
def mock_tool_finder():
    return Mock(spec=ToolFinder)

@pytest.fixture
def tool_manager(mock_logger):
    with patch('src.ai.tools.tools_manager.ToolExecutor') as mock_executor, \
         patch('src.ai.tools.tools_manager.ToolFinder') as mock_finder, \
         patch('src.ai.tools.tools_manager.ToolCallParser') as mock_parser:
        mock_executor.return_value = MagicMock()
        mock_finder.return_value = MagicMock()
        mock_parser.return_value = MagicMock()
        manager = ToolManager(logger=mock_logger)
        yield manager

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

class TestToolManager:
    def test_initialization(self, tool_manager, mock_logger):
        assert tool_manager._logger == mock_logger
        assert tool_manager._tool_finder is None
        assert tool_manager._auto_find_tools is False

    def test_set_tool_finder(self, tool_manager, mock_tool_finder, mock_logger):
        tool_manager.set_tool_finder(mock_tool_finder)
        assert tool_manager._tool_finder == mock_tool_finder
        mock_logger.info.assert_called_once_with("ToolFinder has been set")

    @patch('src.ai.tools.tools_manager.ToolFinder')
    def test_create_tool_finder(self, mock_tool_finder_class, tool_manager, mock_logger):
        mock_tool_finder = MagicMock()
        mock_tool_finder_class.return_value = mock_tool_finder
        
        tool_manager.create_tool_finder(Model.CLAUDE_SONNET_3_5)
        assert tool_manager._tool_finder == mock_tool_finder
        mock_tool_finder_class.assert_called_once_with(
            model=Model.CLAUDE_SONNET_3_5,
            logger=mock_logger
        )
        mock_logger.info.assert_called_once_with(f"Created ToolFinder using {Model.CLAUDE_SONNET_3_5.name}")

    @patch('src.ai.tools.tools_manager.ToolFinder')
    def test_enable_auto_tool_finding(self, mock_tool_finder_class, tool_manager, mock_logger):
        # Reset the mock logger
        mock_logger.reset_mock()
        
        # Mock the ToolFinder creation
        mock_tool_finder = MagicMock()
        mock_tool_finder_class.return_value = mock_tool_finder
        
        # Prevent ToolFinder creation during initialization
        tool_manager._tool_finder = mock_tool_finder
        
        tool_manager.enable_auto_tool_finding(True)
        assert tool_manager._auto_find_tools is True
        mock_logger.info.assert_called_once_with("Auto tool finding enabled")

    def test_find_tools_with_existing_finder(self, tool_manager, mock_tool_finder):
        tool_manager._tool_finder = mock_tool_finder
        mock_tool_finder.find_tools.return_value = [Tool.TICKET_ORACLE]
        
        result = tool_manager.find_tools("test prompt")
        assert result == [Tool.TICKET_ORACLE]
        mock_tool_finder.find_tools.assert_called_once_with("test prompt", None)

    @patch('src.ai.tools.tools_manager.ToolFinder')
    def test_find_tools_without_finder(self, mock_tool_finder_class, tool_manager):
        mock_tool_finder = MagicMock()
        mock_tool_finder_class.return_value = mock_tool_finder
        mock_tool_finder.find_tools.return_value = [Tool.TICKET_ORACLE]
        
        result = tool_manager.find_tools("test prompt")
        assert result == [Tool.TICKET_ORACLE]
        mock_tool_finder.find_tools.assert_called_once_with("test prompt", None)

    def test_handle_tool_calls_with_tool_calls(self, tool_manager, mock_logger):
        mock_response = Mock()
        mock_response.tool_calls = [Mock(name="test_tool", arguments="{}")]
        mock_ai_requester = Mock()
        mock_ai_requester.add_tool_message.return_value = []
        mock_ai_requester.request.return_value = Mock(content="test response")
        
        result = tool_manager.handle_tool_calls([], mock_response, mock_ai_requester)
        assert result.content == "test response"
        mock_ai_requester.add_tool_message.assert_called_once()
        mock_ai_requester.request.assert_called_once()

    def test_handle_tool_calls_with_text_tool_call(self, tool_manager, mock_logger):
        # Set up the tool_manager with mock components
        tool_manager._tool_call_parser = Mock()
        tool_manager._tool_call_parser.parse_tool_call.return_value = {"name": "test_tool", "arguments": {}}

        # Mock the executor
        tool_manager._tool_executor = Mock()
        tool_manager._tool_executor.execute.return_value = "tool result"

        # Create a simple mock response that simulates the behavior of the real response object
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        # Mock the AI response with our custom mock
        mock_response = MockResponse('TOOL_CALL: {"name": "test_tool", "arguments": {}}')
        
        # Mock the AI requester and the follow-up response
        mock_follow_up_response = MockResponse("test response")
        
        mock_ai_requester = Mock()
        mock_ai_requester.add_tool_message.return_value = []
        mock_ai_requester.request.return_value = mock_follow_up_response

        # Call the method under test
        result = tool_manager.handle_tool_calls([], mock_response, mock_ai_requester)

        # The result should be the follow-up response
        assert result == mock_follow_up_response
        
        # Verify method calls
        tool_manager._tool_call_parser.parse_tool_call.assert_called_once_with(mock_response.content)
        tool_manager._tool_executor.execute.assert_called_once_with("test_tool", "{}")
        mock_ai_requester.add_tool_message.assert_called_once()
        mock_ai_requester.request.assert_called_once()

    def test_handle_tool_calls_error_handling(self, tool_manager, mock_logger):
        mock_response = Mock()
        mock_response.tool_calls = [Mock(name="test_tool", arguments="{}")]
        mock_ai_requester = Mock()
        mock_ai_requester.add_tool_message.side_effect = Exception("Test error")
        
        result = tool_manager.handle_tool_calls([], mock_response, mock_ai_requester)
        assert result == mock_response
        mock_logger.error.assert_called_once() 