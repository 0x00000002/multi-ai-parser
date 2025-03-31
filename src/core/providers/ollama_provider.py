"""
Ollama provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
import json
import requests
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config.config_manager import ConfigManager
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from .base_provider import BaseProvider


class OllamaProvider(BaseProvider):
    """Provider implementation for Ollama local models."""
    
    def __init__(self, 
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Ollama provider.
        
        Args:
            model_id: The model identifier
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        super().__init__(model_id, config_manager, logger)
        
        # Get provider configuration
        provider_config = self._config_manager.get_provider_config("ollama")
        self._base_url = provider_config.base_url or "http://localhost:11434"
        
        # Initialize session
        self._session = requests.Session()
        self._logger.info(f"Initialized Ollama provider with base URL: {self._base_url}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Ollama API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Either a string response (when no tools are needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        try:
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                prompt = messages
                formatted_messages = None
            else:
                # Format messages for Ollama API
                formatted_messages = self._format_messages(messages)
                # For single user message, use simple prompt
                if len(formatted_messages) == 1 and formatted_messages[0]["role"] == "user":
                    prompt = formatted_messages[0]["content"]
                    formatted_messages = None
                else:
                    prompt = None
            
            # Prepare request data
            api_url = f"{self._base_url}/api/chat"
            request_data = {
                "model": self._model_id,
                "stream": False,
                **{k: v for k, v in options.items() if v is not None}
            }
            
            # Add either prompt or messages
            if formatted_messages:
                request_data["messages"] = formatted_messages
            else:
                request_data["prompt"] = prompt
            
            # Make the request
            response = self._session.post(api_url, json=request_data)
            response.raise_for_status()
            data = response.json()
            
            # Extract content
            if "message" in data:
                content = data["message"]["content"]
            else:
                content = data.get("response", "")
            
            # Standardize the response format
            return self.standardize_response(content)
            
        except Exception as e:
            self._logger.error(f"Ollama request failed: {str(e)}")
            raise AIRequestError(
                f"Failed to make Ollama request: {str(e)}",
                provider="ollama",
                original_error=e
            )
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the Ollama API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Streamed response as a string
        """
        try:
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                prompt = messages
                formatted_messages = None
            else:
                # Format messages for Ollama API
                formatted_messages = self._format_messages(messages)
                # For single user message, use simple prompt
                if len(formatted_messages) == 1 and formatted_messages[0]["role"] == "user":
                    prompt = formatted_messages[0]["content"]
                    formatted_messages = None
                else:
                    prompt = None
            
            # Prepare request data
            api_url = f"{self._base_url}/api/chat"
            request_data = {
                "model": self._model_id,
                "stream": True,
                **{k: v for k, v in options.items() if v is not None}
            }
            
            # Add either prompt or messages
            if formatted_messages:
                request_data["messages"] = formatted_messages
            else:
                request_data["prompt"] = prompt
            
            # Make the streaming request
            response = self._session.post(api_url, json=request_data, stream=True)
            response.raise_for_status()
            
            # Process the stream
            content = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content += chunk["message"]["content"]
                        else:
                            content += chunk.get("response", "")
                    except json.JSONDecodeError:
                        pass
            
            # Standardize the response format
            return self.standardize_response(content)
            
        except Exception as e:
            self._logger.error(f"Ollama streaming failed: {str(e)}")
            raise AIRequestError(
                f"Failed to stream Ollama response: {str(e)}",
                provider="ollama",
                original_error=e
            ) 