import pytest
from src.ai.ChatGPT import ChatGPT

def test_openai_initialization():
    """Test that OpenAI can be initialized with a system prompt"""
    ai = ChatGPT("")
    assert isinstance(ai, ChatGPT)

def test_openai_stream():
    """Test that OpenAI can stream responses"""
    ai = ChatGPT("Answer the question with exactly one word")
    response = ai.stream("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response == "Paris"
    
def test_openai_request():
    """Test that OpenAI can request responses"""
    ai = ChatGPT("Answer the question with exactly one word")
    response = ai.request("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response == "Paris"


def test_openai_error_handling():
    """Test error handling in OpenAI"""
    with pytest.raises(Exception):
        # Test with invalid input or configuration that should raise an exception
        ai = ChatGPT("")  # or some invalid configuration
        ai.request("Tell me 'Hello'", "")