import pytest
from src.ai.model_selector import ModelSelector, UseCase
import src.ai.ai_config as config
from src.logger import Logger
from unittest.mock import Mock
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

# ModelSelector Tests
class TestModelSelector:
    def test_get_model_params_basic(self):
        model = ModelSelector.get_model_params(UseCase.CODING)
        assert isinstance(model, config.Model)
        assert model.quality == config.Quality.HIGH
        assert model.speed == config.Speed.FAST

    def test_get_model_params_with_quality(self):
        model = ModelSelector.get_model_params(UseCase.CODING, quality=config.Quality.MEDIUM)
        assert model.quality == config.Quality.MEDIUM

    def test_get_model_params_local(self):
        model = ModelSelector.get_model_params(UseCase.CODING, use_local=True)
        assert model.privacy == config.Privacy.LOCAL

    def test_get_system_prompt(self):
        prompt = ModelSelector.get_system_prompt(UseCase.CODING)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
