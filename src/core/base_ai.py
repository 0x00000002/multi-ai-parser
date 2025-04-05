"""
Base AI implementation that handles common functionality.
Implements the AIInterface and provides core conversation features.
"""
from typing import Dict, List, Any, Optional, Union
from .interfaces import AIInterface, ProviderInterface
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.unified_config import UnifiedConfig
from ..exceptions import AISetupError, AIProcessingError, ErrorHandler
from ..conversation.conversation_manager import ConversationManager, Message
from .provider_factory import ProviderFactory
from .providers.base_provider import BaseProvider
import uuid
from ..prompts.prompt_template import PromptTemplate

# Default system prompt if none provided
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions accurately and concisely."""


class AIBase(AIInterface):
    """
    Base implementation of the AI interface.
    Handles provider management, conversation history, and common operations.
    """
    
    def __init__(self, 
                 model: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None):
        """
        Initialize the base AI implementation.
        
        Args:
            model: The model to use (string ID, or None for default)
            system_prompt: Custom system prompt (or None for default)
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            prompt_template: PromptTemplate service for generating prompts
            
        Raises:
            AISetupError: If initialization fails
        """
        try:
            self._request_id = request_id or str(uuid.uuid4())
            # Create a logger with the appropriate name
            self._logger = logger or LoggerFactory.create(name=f"ai_framework.{self._request_id[:8]}")
            
            # Get unified configuration
            self._config = UnifiedConfig.get_instance()
            
            # Get model configuration
            model_id = model or self._config.get_default_model()
            self._model_config = self._config.get_model_config(model_id)
            
            # Set up prompt template
            self._prompt_template = prompt_template or PromptTemplate(logger=self._logger)
            
            # Initialize provider
            self._provider = ProviderFactory.create(
                provider_type=self._model_config.get("provider", "openai"),
                model_id=model_id,
                logger=self._logger
            )
            
            # Set up conversation manager
            self._conversation_manager = ConversationManager()
            
            # Set system prompt (config, parameter, or default)
            self._system_prompt = system_prompt or self._config.get_system_prompt() or self._get_default_system_prompt()
            
            # Add initial system message
            self._conversation_manager.add_message(
                role="system",
                content=self._system_prompt
            )
            
            self._logger.info(f"Initialized AI with model: {model_id}")
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AISetupError(f"Failed to initialize AI: {str(e)}", component="AIBase"),
                logger
            )
            self._logger.error(f"Initialization error: {error_response['message']}")
            raise
    
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt based on model configuration.
        Uses the template system if available.
        
        Returns:
            Default system prompt string
        """
        try:
            # Try to use template
            prompt, _ = self._prompt_template.render_prompt(
                template_id="base_ai",
                variables={"model_id": self._model_config.get("model_id", "")}
            )
            return prompt
        except (ValueError, AttributeError):
            # Fallback to hardcoded prompt
            self._logger.warning("System prompt template not found, using fallback")
            return f"You are a helpful AI assistant using the {self._model_config.get('model_id', 'default')} model."
    
    def request(self, prompt: str, **options) -> str:
        """
        Make a request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response as a string
            
        Raises:
            AIProcessingError: If the request fails
        """
        try:
            self._logger.info(f"Processing request: {prompt[:50]}...")
            
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Get response from provider
            response = self._provider.request(self._conversation_manager.get_messages(), **options)
            
            # Ensure response is properly formatted (should be a dict with 'content' key)
            if isinstance(response, str):
                response = {'content': response, 'tool_calls': []}
            
            # Get the content from the response
            content = response.get('content', '')
            
            # Add assistant message with thoughts handling
            self._conversation_manager.add_message(
                role="assistant",
                content=content,
                extract_thoughts=True,
                show_thinking=self._config.show_thinking
            )
            
            return content
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AIProcessingError(f"Request failed: {str(e)}", component="AIBase"),
                self._logger
            )
            self._logger.error(f"Request error: {error_response['message']}")
            raise
    
    def stream(self, prompt: str, **options) -> str:
        """
        Stream a response from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The complete streamed response as a string
            
        Raises:
            AIProcessingError: If streaming fails
        """
        try:
            self._logger.info(f"Processing streaming request: {prompt[:50]}...")
            
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Stream the response
            response = self._provider.stream(self._conversation_manager.get_messages(), **options)
            
            # Ensure response is properly formatted (should be a dict with 'content' key)
            if isinstance(response, str):
                response = {'content': response, 'tool_calls': []}
                
            # Get the content from the response
            content = response.get('content', '')
            
            # Add assistant message with thoughts handling
            self._conversation_manager.add_message(
                role="assistant", 
                content=content,
                extract_thoughts=True,
                show_thinking=self._config.show_thinking
            )
            
            return content
            
        except Exception as e:
            # Use error handler for standardized error handling
            error_response = ErrorHandler.handle_error(
                AIProcessingError(f"Streaming failed: {str(e)}", component="AIBase"),
                self._logger
            )
            self._logger.error(f"Streaming error: {error_response['message']}")
            raise
    
    def reset_conversation(self) -> None:
        """
        Reset the conversation history.
        
        Clears all messages and restores the system prompt.
        """
        self._conversation_manager.reset()
        self._logger.info("Conversation history reset")
        
        # Restore system prompt if it exists
        if self._system_prompt:
            self._conversation_manager.add_message(
                role="system",
                content=self._system_prompt
            )
    
    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of messages in the conversation
        """
        return self._conversation_manager.get_messages()
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set a new system prompt.
        
        Args:
            system_prompt: New system prompt
        """
        self._system_prompt = system_prompt
        self._conversation_manager.set_system_prompt(system_prompt)
        self._logger.info("System prompt updated")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_id": self._model_config.get("model_id", ""),
            "provider": self._model_config.get("provider", ""),
            "quality": self._model_config.get("quality", ""),
            "speed": self._model_config.get("speed", ""),
            "parameters": self._model_config.get("parameters", {}),
            "privacy": self._model_config.get("privacy", "")
        }