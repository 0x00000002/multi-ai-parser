import pytest
from src.ai.Ollama import Ollama

def test_ollama_initialization():
    """Test that Ollama can be initialized with a system prompt"""
    ai = Ollama("")
    assert isinstance(ai, Ollama)

# @pytest.mark.skip(reason="This test is currently not working")
def test_ollama_stream():
    """Test that Ollama can stream responses"""
    ai = Ollama("Answer the question with exactly one word. No punctuation, no spaces, no extra words.")
    response = ai.stream("What is the capital of France?")
    assert response.strip() == "Paris"

def test_ollama_request():
    """Test that Ollama can request responses"""
    ai = Ollama("Answer the question with exactly one word. No punctuation, no spaces, no extra words.")
    response = ai.request("What is the capital of France?")
    assert response.strip() == "Paris"


# @pytest.mark.skip(reason="This test is currently not working")
def test_ollama_request():
    """Test that Ollama can request responses"""
    ai = Ollama("Answer the question with exactly one word. No punctuation, no spaces, no extra words.")
    response = ai.request("What is the capital of France?")
    assert response.strip() == "Paris"
