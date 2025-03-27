import pytest
from unittest.mock import Mock, patch
from src.ai.tools.tools_registry import ToolsRegistry
from src.ai.tools.tools_list import Tool
from src.ai.AIConfig import Provider
from src.ai.modules.Anthropic import ClaudeAI, AI_API_Key_Error as AnthropicAPIError
from src.ai.modules.OpenAI import ChatGPT, AI_API_Key_Error as OpenAIAPIError
from src.ai.modules.Google import Gemini, AI_API_Key_Error as GoogleAPIError
from src.ai.AIConfig import Model
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

class TestToolsRegistry:
    def test_tools_registration(self):
        tools = [Tool.SOME_TOOL]
        result = ToolsRegistry.get_tools(tools, Provider.OPENAI)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "function" in result[0]

    def test_tool_execution(self):
        tools = [Tool.SOME_TOOL]
        result = ToolsRegistry.get_tools(tools, Provider.ANTHROPIC)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "name" in result[0]
        assert "description" in result[0]
        assert "input_schema" in result[0]

    def test_invalid_tool(self):
        tools = []
        result = ToolsRegistry.get_tools(tools, Provider.GOOGLE)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_tool_with_context(self):
        tools = [Tool.SOME_TOOL]
        result = ToolsRegistry.get_tools(tools, Provider.OLLAMA)
        assert isinstance(result, list)
        assert len(result) > 0
        assert "function" in result[0]

    @patch('anthropic.Anthropic')
    def test_claude_initialization(self, mock_anthropic_class, mock_api_keys):
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        
        claude = ClaudeAI(Model.CLAUDE_SONNET_3_5)
        assert claude.client == mock_client

    @patch('src.ai.modules.OpenAI.OpenAI')
    def test_chatgpt_initialization(self, mock_openai_class, mock_api_keys):
        # Create mock client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Create ChatGPT instance
        chatgpt = ChatGPT(Model.CHATGPT_4O_MINI)
        
        # Verify that OpenAI client was created with correct API key
        mock_openai_class.assert_called_once_with(api_key=Model.CHATGPT_4O_MINI.api_key)
        assert chatgpt.model == Model.CHATGPT_4O_MINI
        assert chatgpt.client == mock_client

    def test_ollama_initialization(self):
        try:
            from ollama import Client
            assert True
        except ImportError:
            pytest.skip("Ollama not installed")

    def test_gemini_initialization(self):
        try:
            from google import genai
            assert True
        except ImportError:
            pytest.skip("Google AI not installed")

    @patch('anthropic.Anthropic')
    def test_claude_request(self, mock_anthropic_class, mock_api_keys):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        claude = ClaudeAI(Model.CLAUDE_SONNET_3_5)
        response = claude.request([{"role": "user", "content": "Test"}])
        assert response == "Test response"

    @patch('src.ai.modules.OpenAI.OpenAI')
    def test_chatgpt_request(self, mock_openai_class, mock_api_keys):
        # Create mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        
        # Create mock client
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create ChatGPT instance and make request
        chatgpt = ChatGPT(Model.CHATGPT_4O_MINI)
        
        # Verify mock was used with correct API key
        mock_openai_class.assert_called_once_with(api_key=Model.CHATGPT_4O_MINI.api_key)
        assert chatgpt.client == mock_client
        
        # Make request and verify response
        response = chatgpt.request([{"role": "user", "content": "Test"}])
        assert response == "Test response"
        
        # Verify mock was called correctly with expected parameters
        mock_client.chat.completions.create.assert_called_once_with(
            model=Model.CHATGPT_4O_MINI.model_id,
            messages=[{"role": "user", "content": "Test"}],
            max_completion_tokens=1024
        )

    @patch('google.genai.Client')
    def test_gemini_request(self, mock_genai_class, mock_api_keys):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_class.return_value = mock_client
        
        gemini = Gemini(Model.GEMINI_1_5_PRO)
        response = gemini.request("Test")
        assert response == "Test response"

    def test_error_handling(self, monkeypatch):
        # Remove API keys to test error handling
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Test each provider's API key validation
        with pytest.raises(AnthropicAPIError, match="No valid Anthropic API key found"):
            ClaudeAI(Model.CLAUDE_SONNET_3_5)
        with pytest.raises(OpenAIAPIError, match="No valid OpenAI API key found"):
            ChatGPT(Model.CHATGPT_4O_MINI)
        with pytest.raises(GoogleAPIError, match="No valid Google API key found"):
            # Mock dotenv.load_dotenv to do nothing
            with patch('src.ai.modules.Google.load_dotenv'):
                Gemini(Model.GEMINI_1_5_PRO) 