import pytest
from src.ai.AI import AI, Role
from src.ai.ModelSelector import ModelSelector, UseCase
import src.ai.AIConfig as config
from src.Logger import Logger, NullLogger
from unittest.mock import Mock, patch
import os
from dotenv import load_dotenv

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

# AI Configuration Tests
class TestAIConfig:
    def test_find_model_exact_match(self):
        model = config.Model.CLAUDE_SONNET_3_5
        assert model.api_key.startswith("sk-ant-api")

    def test_find_model_fallback(self):
        model = config.Model.CHATGPT_4O_MINI
        assert model.api_key.startswith("sk-proj")

    def test_model_api_key(self):
        model = config.Model.GEMINI_1_5_PRO
        assert model.api_key.startswith("AIza")

