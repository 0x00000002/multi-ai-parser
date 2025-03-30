"""
Base provider implementation.
"""
from typing import List, Dict, Any, Optional, Union
from ..interfaces import ProviderInterface
from ...utils.logger import LoggerInterface, LoggerFactory
from ...exceptions import AIProviderError
from ...config.config_manager import ConfigManager


class BaseProvider(ProviderInterface):
    """Base implementation for AI providers with common message handling."""
    
    def __init__(self, 
                 model_id: str,
                 config_manager: ConfigManager,
                 logger: Optional[LoggerInterface] = None):
        """
        Initialize the base provider.
        
        Args:
            model_id: The model identifier
            config_manager: Configuration manager instance
            logger: Logger instance
        """
        self._model_id = model_id
        self._config_manager = config_manager
        self._logger = logger
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert messages to provider-specific format.
        Override this method in provider implementations.
        
        Args:
            messages: List of messages in standard format
            
        Returns:
            List of messages in provider-specific format
        """
        return messages
    
    def _convert_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert provider response to standard format.
        Override this method in provider implementations.
        
        Args:
            response: Provider-specific response
            
        Returns:
            Response in standard format
        """
        return response
    
    def _add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, 
                         content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message in provider-specific format.
        Override this method in provider implementations.
        
        Args:
            messages: Current conversation messages
            name: Tool name
            content: Tool response content
            
        Returns:
            Updated messages list
        """
        messages.append({
            "role": "tool",
            "name": name,
            "content": str(content)
        })
        return messages
        
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content from a provider response.
        Override this method in provider implementations if needed.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            Content string
        """
        return response.get('content', '')
    
    def _has_tool_calls(self, response: Dict[str, Any]) -> bool:
        """
        Check if response contains tool calls.
        Override this method in provider implementations if needed.
        
        Args:
            response: Provider response dictionary
            
        Returns:
            True if response contains tool calls
        """
        return bool(response.get('tool_calls', []))
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the AI model.
        Override this method in provider implementations, but maintain the return contract.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Either a string response (when no tool calls are needed) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        # This is a base implementation - providers should override this
        return "This is a base implementation. Override in provider classes."
        
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the AI model.
        Override this method in provider implementations.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        # This is a base implementation - providers should override this
        return "This is a base implementation. Override in provider classes." 