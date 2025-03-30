"""
OpenAI provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...config.config_manager import ConfigManager
from ...exceptions import AIRequestError, AICredentialsError, AIProviderError
from ...tools.models import ToolCall
from .base_provider import BaseProvider
import openai


class OpenAIProvider(BaseProvider):
    """Provider implementation for OpenAI models."""
    
    # Role mapping for OpenAI API
    _ROLE_MAP = {
        "system": "system",
        "user": "user",
        "assistant": "assistant",
        "tool": "tool" 
    }
    
    def __init__(self, 
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_id: The model identifier
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        super().__init__(model_id, config_manager, logger)
        
        # Get provider configuration
        provider_config = self._config_manager.get_provider_config("openai")
        api_key = provider_config.api_key
        
        if not api_key:
            raise AICredentialsError("No OpenAI API key found")
        
        try:
            self._client = openai.OpenAI(api_key=api_key)
            self._logger.info(f"Successfully initialized OpenAI client for {model_id}")
        except Exception as e:
            raise AICredentialsError(f"Failed to initialize OpenAI client: {str(e)}")
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI API."""
        formatted = []
        for msg in messages:
            role = self._ROLE_MAP.get(msg["role"], msg["role"])
            content = msg["content"]
            
            # Handle tool messages specially
            if msg["role"] == "tool":
                formatted.append({
                    "role": "tool",
                    "content": content,
                    "tool_call_id": msg.get("tool_call_id", "unknown"),
                    "name": msg.get("name", "unknown_tool")
                })
            else:
                formatted.append({
                    "role": role,
                    "content": content
                })
        
        return formatted
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the OpenAI API.
        
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
                # Format messages for OpenAI API
                formatted_messages = self._format_messages(messages)
            
            # Get function/tool definitions if provided
            tools = options.pop("tools", None)
            
            # Make the request
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=formatted_messages,
                tools=tools,
                **{k: v for k, v in options.items() if v is not None}
            )
            
            # Extract response message
            message = response.choices[0].message
            
            # Convert tool calls if present
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        id=tc.id
                    ))
            
            # Create response object
            result = {
                "content": message.content or "",
                "tool_calls": tool_calls,
                "finish_reason": response.choices[0].finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # If there are no tool calls, just return the content as a string
            if not tool_calls:
                return message.content or ""
            
            # Otherwise return the full response object for tool handling
            return result
            
        except Exception as e:
            self._logger.error(f"OpenAI request failed: {str(e)}")
            raise AIRequestError(
                f"Failed to make OpenAI request: {str(e)}",
                provider="openai",
                original_error=e
            )
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the OpenAI API.
        
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
                # Format messages for OpenAI API
                formatted_messages = self._format_messages(messages)
            
            # Stream the response
            stream = self._client.chat.completions.create(
                model=self._model_id,
                messages=formatted_messages,
                stream=True,
                **{k: v for k, v in options.items() if v is not None}
            )
            
            # Collect content chunks
            content_chunks = []
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content_chunks.append(chunk.choices[0].delta.content)
            
            return "".join(content_chunks)
            
        except Exception as e:
            self._logger.error(f"OpenAI streaming failed: {str(e)}")
            raise AIRequestError(
                f"Failed to stream OpenAI response: {str(e)}",
                provider="openai",
                original_error=e
            ) 