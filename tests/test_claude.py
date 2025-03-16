import pytest
from src.ai.ClaudeAI import ClaudeAI

def test_claude_initialization():
    """Test that ClaudeAI can be initialized with a system prompt"""
    ai = ClaudeAI("")
    assert isinstance(ai, ClaudeAI)

def test_claude_stream():
    """Test that ClaudeAI can stream responses"""
    ai = ClaudeAI("Answer the question with exactly one word")
    response = ai.stream("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response == "Paris"

def test_claude_request():
    """Test that ClaudeAI can request responses"""
    ai = ClaudeAI("Answer the question with exactly one word")
    response = ai.request("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response == "Paris"

def test_claude_error_handling():
    """Test error handling in ClaudeAI"""
    with pytest.raises(Exception):
        # Test with invalid input or configuration that should raise an exception
        ai = ClaudeAI("")  # or some invalid configuration
        ai.request("Tell me 'Hello'", "")