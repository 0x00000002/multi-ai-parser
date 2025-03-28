import pytest
from src.ai.base_ai import AIBase as AI, Role
from src.ai.ModelSelector import ModelSelector, UseCase
import src.ai.AIConfig as config
from src.Logger import Logger, NullLogger
from unittest.mock import Mock, patch
import os
from dotenv import load_dotenv
from src.ai.modules.Anthropic import ClaudeAI
from src.ai.modules.Google import Gemini
from src.ai.modules.OpenAI import ChatGPT
from src.ai.tools.models import ToolCallRequest

# Load environment variables from .env file
load_dotenv()

# Test fixtures
@pytest.fixture
def mock_logger():
    return Mock(spec=Logger)

@pytest.fixture
def mock_api_keys(monkeypatch):
    # Get API keys from environment variables
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    # Set up API keys in test environment
    monkeypatch.setenv("ANTHROPIC_API_KEY", anthropic_key)
    monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    monkeypatch.setenv("GOOGLE_API_KEY", google_key)
    
    yield
    
    # Clean up
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

# AI Core Tests
class TestAI:
    @patch('src.ai.modules.Anthropic.ClaudeAI')
    def test_ai_initialization_with_model(self, mock_claude, mock_logger):
        mock_claude.return_value = Mock()
        ai = AI(config.Model.CLAUDE_SONNET_3_5, logger=mock_logger)
        assert ai.model == config.Model.CLAUDE_SONNET_3_5
        assert ai.system_prompt == ""

    @patch('src.ai.modules.OpenAI.ChatGPT')
    def test_ai_initialization_with_use_case(self, mock_chatgpt, mock_logger):
        mock_chatgpt.return_value = Mock()
        ai = AI(UseCase.CODING, logger=mock_logger)
        assert isinstance(ai.model, config.Model)
        assert len(ai.system_prompt) > 0

    @patch('src.ai.modules.OpenAI.ChatGPT')
    def test_ai_initialization_with_dict(self, mock_chatgpt, mock_logger):
        mock_chatgpt.return_value = Mock()
        ai = AI({"privacy": config.Privacy.EXTERNAL, "quality": config.Quality.HIGH, "speed": config.Speed.FAST}, logger=mock_logger)
        assert isinstance(ai.model, config.Model)

    @patch('src.ai.base_ai.ClaudeAI')
    def test_conversation_management(self, mock_claude, mock_logger):
        # Create mock instance with predefined response
        mock_claude_instance = Mock()
        mock_response = ToolCallRequest(
            tool_calls=[],
            content="Test response",
            finish_reason="stop"
        )
        mock_claude_instance.request.return_value = mock_response
        mock_claude.return_value = mock_claude_instance

        # Create AI instance with mocked ClaudeAI
        ai = AI(config.Model.CLAUDE_SONNET_3_5, logger=mock_logger)
        
        # Verify mock was used
        mock_claude.assert_called_once_with(config.Model.CLAUDE_SONNET_3_5, "", mock_logger)
        assert ai.ai == mock_claude_instance
        
        # Make request and verify response
        response = ai.request("Test prompt")
        assert response == "Test response"
        
        # Verify conversation was tracked
        assert len(ai.questions) == 1
        assert len(ai.responses) == 1
        assert ai.questions[0] == "Test prompt"
        assert ai.responses[0] == "Test response"
        
        # Verify mock was called correctly
        mock_claude_instance.request.assert_called_once()
        call_args = mock_claude_instance.request.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 1
        assert call_args[0]["role"] == "user"
        assert call_args[0]["content"] == "Test prompt"

        # Make another request
        response = ai.request("Test prompt 2")
        assert response == "Test response"

        # Verify conversation was tracked
        assert len(ai.questions) == 2
        assert len(ai.responses) == 2
        assert ai.questions[1] == "Test prompt 2"
        assert ai.responses[1] == "Test response"

        # Verify mock was called correctly for second request
        assert mock_claude_instance.request.call_count == 2
        call_args = mock_claude_instance.request.call_args[0][0]
        assert isinstance(call_args, list)
        assert len(call_args) == 3  # Previous user message, AI response, and new user message
        assert call_args[0]["role"] == "user"
        assert call_args[0]["content"] == "Test prompt"
        assert call_args[1]["role"] == "assistant"
        assert call_args[1]["content"] == "Test response"
        assert call_args[2]["role"] == "user"
        assert call_args[2]["content"] == "Test prompt 2"

    @patch('src.ai.modules.Anthropic.ClaudeAI')
    def test_system_prompt_management(self, mock_claude, mock_logger):
        mock_claude.return_value = Mock()
        ai = AI(config.Model.CLAUDE_SONNET_3_5, system_prompt="Test prompt", logger=mock_logger)
        assert ai.system_prompt == "Test prompt"
        
        ai.system_prompt = "New prompt"
        assert ai.system_prompt == "New prompt"

    @patch('src.ai.modules.Google.Gemini')
    def test_model_switching(self, mock_gemini, mock_logger):
        mock_gemini.return_value = Mock()
        ai = AI(config.Model.GEMINI_1_5_PRO, logger=mock_logger)
        assert ai.model == config.Model.GEMINI_1_5_PRO
        
        ai.model = config.Model.CLAUDE_SONNET_3_5
        assert ai.model == config.Model.CLAUDE_SONNET_3_5

    @patch('src.ai.base_ai.ClaudeAI')
    def test_streaming_response(self, mock_claude, mock_logger):
        # Create mock instance with predefined response
        mock_claude_instance = Mock()
        mock_claude_instance.stream.return_value = "Test streaming response"
        mock_claude.return_value = mock_claude_instance

        # Create AI instance with mocked ClaudeAI
        ai = AI(config.Model.CLAUDE_SONNET_3_5, logger=mock_logger)
        
        # Verify mock was used
        mock_claude.assert_called_once_with(config.Model.CLAUDE_SONNET_3_5, "", mock_logger)
        assert ai.ai == mock_claude_instance
        
        # Make request and verify response
        response = ai.stream("Test prompt")
        assert response == "Test streaming response"
        
        # Verify conversation was tracked
        assert len(ai.questions) == 1
        assert len(ai.responses) == 1
        assert ai.questions[0] == "Test prompt"
        assert ai.responses[0] == "Test streaming response"
        
        # Verify mock was called correctly
        mock_claude_instance.stream.assert_called_once()

    @patch('src.ai.modules.OpenAI.ChatGPT')
    def test_for_use_case_factory(self, mock_chatgpt, mock_logger):
        mock_chatgpt.return_value = Mock()
        ai = AI.for_use_case(UseCase.CODING, logger=mock_logger)
        assert isinstance(ai, AI)
        assert len(ai.system_prompt) > 0 