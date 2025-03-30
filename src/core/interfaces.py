"""
Core interfaces for the AI framework.
Defines protocols that different components must implement.
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
            The model's response as a string
        """
        ...
    
    def stream(self, prompt: str, **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The complete streamed response as a string
        """
        ...
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
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
    
    def request(self, messages: List[Dict[str, Any]], **options) -> Dict[str, Any]:
        """
        Make a request to the provider.
        
        Args:
            messages: List of conversation messages
            options: Additional options for the request
            
        Returns:
            Provider response object
        """
        ...
    
    def stream(self, messages: List[Dict[str, Any]], **options) -> str:
        """
        Stream a response from the provider.
        
        Args:
            messages: List of conversation messages
            options: Additional options for the request
            
        Returns:
            The complete streamed response as a string
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


@runtime_checkable
class ToolStrategy(Protocol):
    """Interface for tool implementations."""
    
    def execute(self, **args) -> Any:
        """
        Execute the tool with the provided arguments.
        
        Args:
            args: Tool-specific arguments
            
        Returns:
            Tool execution result
        """
        ...
    
    def get_description(self) -> str:
        """
        Get a description of the tool.
        
        Returns:
            Tool description string
        """
        ...
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for the tool parameters.
        
        Returns:
            JSON schema as a dictionary
        """
        ...


@runtime_checkable
class LoggerInterface(Protocol):
    """Interface for logging implementations."""
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        ...
    
    def info(self, message: str) -> None:
        """Log an info message."""
        ...
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        ...
    
    def error(self, message: str) -> None:
        """Log an error message."""
        ...
    
    def critical(self, message: str) -> None:
        """Log a critical message."""
        ...