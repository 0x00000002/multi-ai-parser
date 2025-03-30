"""
Core interfaces for AI and provider implementations.
"""
from typing import Protocol, List, Dict, Any, Optional, Union, AsyncIterator
from typing_extensions import runtime_checkable


@runtime_checkable
class AIInterface(Protocol):
    """Interface for synchronous AI interactions."""
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response (after any tool calls have been resolved)
        """
        ...
    
    def stream(self, prompt: str, **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The complete streamed response
        """
        ...
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        ...
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages
        """
        ...


@runtime_checkable
class AsyncAIInterface(Protocol):
    """Interface for asynchronous AI interactions."""
    
    async def async_request(self, prompt: str, **options) -> str:
        """
        Make an asynchronous request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response as a string
        """
        ...
    
    async def async_stream(self, prompt: str, **options) -> AsyncIterator[str]:
        """
        Stream a response asynchronously from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            An async iterator yielding response chunks
        """
        ...
    
    async def async_reset_conversation(self) -> None:
        """Reset the conversation history asynchronously."""
        ...


@runtime_checkable
class ProviderInterface(Protocol):
    """Interface for AI providers."""
    
    def request(self, messages: Union[str, List[Dict[str, Any]]], **options) -> Union[str, Dict[str, Any]]:
        """
        Make a request to the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Either a string response (when no tools are called) or 
            a dictionary with 'content' and possibly 'tool_calls' for further processing
        """
        ...
    
    def stream(self, messages: Union[str, List[Dict[str, Any]]], **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            messages: The conversation messages or a simple string prompt
            options: Additional options for the request
            
        Returns:
            Streamed response as a string
        """
        ...


@runtime_checkable
class AsyncProviderInterface(Protocol):
    """Interface for asynchronous AI providers."""
    
    async def async_request(self, messages: List[Dict[str, Any]], **options) -> Dict[str, Any]:
        """
        Make an asynchronous request to the provider.
        
        Args:
            messages: List of conversation messages
            options: Additional options for the request
            
        Returns:
            Provider response object
        """
        ...
    
    async def async_stream(self, messages: List[Dict[str, Any]], **options) -> AsyncIterator[str]:
        """
        Stream a response asynchronously from the provider.
        
        Args:
            messages: List of conversation messages
            options: Additional options for the request
            
        Returns:
            An async iterator yielding response chunks
        """
        ...


@runtime_checkable
class ToolCapableProviderInterface(ProviderInterface, Protocol):
    """Interface for providers that support tools/functions."""
    
    def add_tool_message(self, messages: List[Dict[str, Any]], 
                         name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        ...


@runtime_checkable
class AsyncToolCapableProviderInterface(AsyncProviderInterface, Protocol):
    """Interface for async providers that support tools/functions."""
    
    async def async_add_tool_message(self, messages: List[Dict[str, Any]], 
                              name: str, content: str) -> List[Dict[str, Any]]:
        """
        Add a tool message to the conversation history asynchronously.
        
        Args:
            messages: The current conversation history
            name: The name of the tool
            content: The content/result of the tool call
            
        Returns:
            Updated conversation history
        """
        ... 