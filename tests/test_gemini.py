import pytest
from src.ai.Gemini import Gemini

def test_gemini_initialization():
    """Test that Gemini can be initialized with a system prompt"""
    ai = Gemini("")
    assert isinstance(ai, Gemini)

def test_gemini_stream():
    """Test that Gemini can stream responses"""
    ai = Gemini("Answer the question with exactly one word")
    response = ai.stream("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response.strip() == "Paris"

def test_gemini_request():
    """Test that Gemini can request responses"""
    ai = Gemini("Answer the question with exactly one word")
    response = ai.request("What is the capital of France?")
    assert response is not None
    assert isinstance(response, str)
    assert response.__contains__("Paris")
    assert response.strip() == "Paris"


def test_gemini_error_handling():
    """Test error handling in Gemini"""
    with pytest.raises(Exception):
        # Test with invalid input or configuration that should raise an exception
        ai = Gemini("")  # or some invalid configuration
        ai.request("Tell me 'Hello'", "")