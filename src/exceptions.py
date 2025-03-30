"""
Custom exceptions for the AI framework.
Provides a standardized error handling hierarchy with rich context.
"""
from typing import Optional, Any, Dict, List, Union
from datetime import datetime


class AIError(Exception):
    """Base exception for all AI framework errors."""
    
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Initialize the AI error.
        
        Args:
            message: Error message
            provider: Name of the AI provider if applicable
            original_error: Original exception that caused this error
            context: Additional context data for debugging
            timestamp: When the error occurred (defaults to now)
        """
        self.message = message
        self.provider = provider
        self.original_error = original_error
        self.context = context or {}
        self.timestamp = timestamp or datetime.utcnow()
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for logging/debugging."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'provider': self.provider,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'original_error': str(self.original_error) if self.original_error else None
        }


class AIConfigError(AIError):
    """Base class for configuration-related errors."""
    pass


class AISetupError(AIConfigError):
    """Raised when there's an error during AI system setup."""
    pass


class AICredentialsError(AIConfigError):
    """Raised when there's an issue with AI provider credentials."""
    pass


class AIProviderError(AIError):
    """Base class for provider-related errors."""
    pass


class AIRequestError(AIProviderError):
    """Raised when there's an error making a request to an AI provider."""
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = {
            'request_data': request_data,
            'response_data': response_data,
            **kwargs
        }
        super().__init__(message, provider, original_error, context)


class AIResponseError(AIProviderError):
    """Raised when there's an error processing the provider's response."""
    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        original_error: Optional[Exception] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = {
            'response_data': response_data,
            **kwargs
        }
        super().__init__(message, provider, original_error, context)


class AIToolError(AIError):
    """Base class for tool-related errors."""
    def __init__(
        self,
        message: str,
        tool_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        context = {
            'tool_name': tool_name,
            'tool_args': tool_args,
            **kwargs
        }
        super().__init__(message, original_error=original_error, context=context)


class AIToolExecutionError(AIToolError):
    """Raised when there's an error executing an AI tool."""
    pass


class AIToolValidationError(AIToolError):
    """Raised when there's an error validating tool arguments."""
    pass


class AIToolNotFoundError(AIToolError):
    """Raised when a requested tool is not found."""
    pass


class ConversationError(AIError):
    """Base class for conversation-related errors."""
    def __init__(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
        **kwargs
    ):
        context = {
            'conversation_id': conversation_id,
            **kwargs
        }
        super().__init__(message, original_error=original_error, context=context)


class ConversationStateError(ConversationError):
    """Raised when there's an error with conversation state."""
    pass


class ConversationValidationError(ConversationError):
    """Raised when there's an error validating conversation data."""
    pass


class AIProcessingError(AIError):
    """Raised when there's an error during AI processing."""
    def __init__(
        self,
        message: str,
        processing_stage: Optional[str] = None,
        original_error: Optional[Exception] = None,
        **kwargs
    ):
        context = {
            'processing_stage': processing_stage,
            **kwargs
        }
        super().__init__(message, original_error=original_error, context=context) 