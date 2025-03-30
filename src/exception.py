"""
Exception hierarchy for the AI framework.
Provides structured error handling and categorization.
"""
from typing import Optional, Any, Dict


class AIError(Exception):
    """Base exception for all AI-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize with a message and optional context.
        
        Args:
            message: Error message
            context: Additional context information
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)
    
    def with_context(self, **kwargs) -> 'AIError':
        """
        Add context information to the exception.
        
        Args:
            kwargs: Key-value pairs to add to context
            
        Returns:
            Self with updated context
        """
        self.context.update(kwargs)
        return self


class AISetupError(AIError):
    """Error during AI initialization or configuration."""
    pass


class AIConfigError(AISetupError):
    """Error in configuration settings."""
    pass


class AICredentialsError(AISetupError):
    """Error with API credentials."""
    pass


class AIProcessingError(AIError):
    """Error during AI request processing."""
    pass


class AIRequestError(AIProcessingError):
    """Error sending request to provider."""
    
    def __init__(self, 
                 message: str, 
                 provider: Optional[str] = None,
                 status_code: Optional[int] = None,
                 original_error: Optional[Exception] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize with request-specific information.
        
        Args:
            message: Error message
            provider: Provider name
            status_code: HTTP status code if applicable
            original_error: Original exception from the provider
            context: Additional context information
        """
        context = context or {}
        if provider:
            context['provider'] = provider
        if status_code:
            context['status_code'] = status_code
        if original_error:
            context['original_error'] = original_error
            context['original_error_type'] = type(original_error).__name__
        
        super().__init__(message, context)


class AIResponseError(AIProcessingError):
    """Error parsing or handling the response."""
    pass


class AIContentFilterError(AIResponseError):
    """Content was filtered by provider safety systems."""
    pass


class AIRateLimitError(AIRequestError):
    """Rate limit exceeded on provider API."""
    pass


class AIStreamingError(AIProcessingError):
    """Error during streaming response."""
    pass


class AIToolError(AIError):
    """Error related to tool operations."""
    pass


class AIToolNotFoundError(AIToolError):
    """Requested tool not found."""
    pass


class AIToolExecutionError(AIToolError):
    """Error executing a tool."""
    
    def __init__(self, 
                 message: str, 
                 tool_name: Optional[str] = None,
                 original_error: Optional[Exception] = None, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Initialize with tool-specific information.
        
        Args:
            message: Error message
            tool_name: Name of the tool
            original_error: Original exception from the tool
            context: Additional context information
        """
        context = context or {}
        if tool_name:
            context['tool_name'] = tool_name
        if original_error:
            context['original_error'] = original_error
            context['original_error_type'] = type(original_error).__name__
        
        super().__init__(message, context)