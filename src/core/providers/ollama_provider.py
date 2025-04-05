"""
Ollama provider implementation.
"""
from typing import List, Dict, Any, Optional, Union, BinaryIO
try:
    import ollama
except ImportError:
    ollama = None

from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config import get_config
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError, AISetupError
from .base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama local models."""
    
    def __init__(self, 
                 model_id: str,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            model_id: The model identifier
            logger: Logger instance
        """
        super().__init__(model_id, logger)
        
        # Check if ollama is installed
        if ollama is None:
            raise AISetupError(
                "Ollama SDK not installed. Please install with 'pip install ollama'.",
                component="ollama"
            )
            
        # Get provider configuration
        self.provider_config = self.config.get_provider_config("ollama") or {}
        
        self.logger.info(f"Initialized Ollama provider for model: {self.model_id}")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format messages for Ollama API."""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Map roles
            if role == "user":
                formatted.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted.append({"role": "assistant", "content": content})
            elif role == "system":
                formatted.append({"role": "system", "content": content})
            # Ollama doesn't support tool messages natively
        
        return formatted
    
    def request(self, messages: List[Dict[str, str]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Ollama API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Response content as a string
        """
        try:
            # Format messages for Ollama
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = {}
            for key, value in options.items():
                if key in ["temperature", "top_p", "top_k", "num_ctx", "num_predict"]:
                    params[key] = value
            
            # Make the API call using the Ollama SDK
            response = ollama.chat(
                model=self.model_id,
                messages=formatted_messages,
                options=params,
                stream=False
            )
            
            # Extract content from the response
            content = response.get("message", {}).get("content", "")
            
            return content
            
        except Exception as e:
            self.logger.error(f"Ollama request failed: {str(e)}")
            raise AIRequestError(f"Ollama request error: {str(e)}", provider="ollama")
    
    def stream(self, messages: List[Dict[str, str]], **options) -> str:
        """
        Stream a response from the Ollama API.
        
        Args:
            messages: List of message dictionaries
            **options: Additional options
            
        Returns:
            Aggregated response as a string
        """
        try:
            # Format messages for Ollama
            formatted_messages = self._format_messages(messages)
            
            # Merge model parameters with options
            params = {}
            for key, value in options.items():
                if key in ["temperature", "top_p", "top_k", "num_ctx", "num_predict"]:
                    params[key] = value
            
            # Make the streaming API call
            chunks = []
            for chunk in ollama.chat(
                model=self.model_id,
                messages=formatted_messages,
                options=params,
                stream=True
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    chunks.append(chunk["message"]["content"])
                    
            # Join all chunks
            return "".join(chunks)
            
        except Exception as e:
            self.logger.error(f"Ollama streaming failed: {str(e)}")
            raise AIRequestError(f"Ollama streaming error: {str(e)}", provider="ollama") 