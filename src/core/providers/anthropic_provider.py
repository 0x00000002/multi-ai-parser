"""
Anthropic provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config.config_manager import ConfigManager
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.models import ToolCall
from .base_provider import BaseProvider
import anthropic


class AnthropicProvider(BaseProvider):
    """Provider implementation for Anthropic's Claude AI."""
    
    # Role mapping for Anthropic API
    _ROLE_MAP = {
        "system": "assistant",
        "user": "user",
        "assistant": "assistant",
        "tool": "assistant"
    }
    
    def __init__(self, 
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_id: The model identifier
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        super().__init__(model_id, config_manager, logger)
        
        # Get provider configuration
        provider_config = self._config_manager.get_provider_config("anthropic")
        api_key = provider_config.api_key
        
        if not api_key or not (api_key.startswith("sk-ant-api") or api_key.startswith("sk-ant-api03")):
            raise AICredentialsError("No valid Anthropic API key found")
        
        try:
            self._client = anthropic.Anthropic(api_key=api_key)
            self._logger.info(f"Successfully initialized Anthropic client for {model_id}")
        except Exception as e:
            raise AICredentialsError(f"Failed to initialize Anthropic client: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for Anthropic API."""
        return [{
            "role": self._ROLE_MAP[msg["role"]],
            "content": f"Tool {msg.get('name', 'unknown')}: {msg['content']}" if msg["role"] == "tool" else msg["content"]
        } for msg in messages]
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the Anthropic API.
        
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
                formatted_messages = [{"role": "user", "content": messages}]
            else:
                # Format messages for Anthropic API
                formatted_messages = self._format_messages(messages)
            
            # Use max_tokens from options or default to 1000
            max_tokens = options.get('max_tokens', 1000)
            
            # Make the request
            response = self._client.messages.create(
                model=self._model_id,
                messages=formatted_messages,
                max_tokens=max_tokens,
                **{k: v for k, v in options.items() if v is not None and k != 'max_tokens'}
            )
            
            # Convert response to standardized format
            tool_calls = []
            content = response.content[0]
            if hasattr(content, 'tool_calls') and content.tool_calls:
                tool_calls = [ToolCall(name=t.name, arguments=t.arguments) for t in content.tool_calls]
            
            # Create response object
            result = {
                "content": content.text,
                "tool_calls": tool_calls,
                "finish_reason": response.stop_reason
            }
            
            # If there are no tool calls, just return the content as a string
            if not tool_calls:
                return self.standardize_response(content.text)
            
            # Otherwise return the full response object for tool handling
            return result
            
        except Exception as e:
            self._logger.error(f"Anthropic request failed: {str(e)}")
            raise AIRequestError(
                f"Failed to make Anthropic request: {str(e)}",
                provider="anthropic",
                original_error=e
            )
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the Anthropic API.
        
        Args:
            messages: User message string or list of conversation messages
            options: Additional request options
            
        Returns:
            Streamed response as a string
        """
        try:
            # Handle string input by converting to messages format
            if isinstance(messages, str):
                formatted_messages = [{"role": "user", "content": messages}]
            else:
                # Format messages for Anthropic API
                formatted_messages = self._format_messages(messages)
            
            # Use max_tokens from options or default to 1000
            max_tokens = options.get('max_tokens', 1000)
            
            # Stream the response
            response = self._client.messages.create(
                model=self._model_id,
                messages=formatted_messages,
                max_tokens=max_tokens,
                stream=True,
                **{k: v for k, v in options.items() if v is not None and k != 'max_tokens'}
            )
            
            return "".join(chunk.content[0].text for chunk in response if chunk.content)
            
        except Exception as e:
            self._logger.error(f"Anthropic streaming failed: {str(e)}")
            raise AIRequestError(
                f"Failed to stream Anthropic response: {str(e)}",
                provider="anthropic",
                original_error=e
            ) 