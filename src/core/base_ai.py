"""
Base AI implementation that handles common functionality.
Implements the AIInterface and provides core conversation features.
"""
from typing import Dict, List, Any, Optional, Union
from .interfaces import AIInterface, ProviderInterface
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.config_manager import ConfigManager
from ..config.models import Model
from ..exceptions import AISetupError, AIProcessingError
from ..conversation.conversation_manager import ConversationManager, Message
from .provider_factory import ProviderFactory
from .providers.base_provider import BaseProvider
import uuid


class AIBase(AIInterface):
    """
    Base implementation of the AI interface.
    Handles provider management, conversation history, and common operations.
    """
    
    def __init__(self, 
                 model: Optional[Union[Model, str]] = None, 
                 system_prompt: Optional[str] = None,
                 config_manager: Optional[ConfigManager] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None):
        """
        Initialize the base AI implementation.
        
        Args:
            model: The model to use (Model enum or string ID, or None for default)
            system_prompt: Custom system prompt (or None for default)
            config_manager: Configuration manager instance
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            
        Raises:
            AISetupError: If initialization fails
        """
        self._request_id = request_id or str(uuid.uuid4())
        # Create a logger with the appropriate name
        self._logger = logger or LoggerFactory.create(name=f"ai_framework.{self._request_id[:8]}")
        self._config_manager = config_manager or ConfigManager()
        
        try:
            # Get model configuration
            if model is None:
                # Use default model if none specified
                self._model_config = self._config_manager.get_model_config(self._config_manager.default_model)
                self._logger.info(f"Using default model: {self._model_config.model_id}")
            else:
                # Determine the model key to use
                model_key = model.value if isinstance(model, Model) else model
                self._model_config = self._config_manager.get_model_config(model_key)
                model_name = model.name if isinstance(model, Model) else model_key
                self._logger.info(f"Using model: {model_name} ({self._model_config.model_id})")
            
            # Set up the provider
            self._provider = ProviderFactory.create(
                provider_type=self._model_config.provider,
                model_id=model,  # Pass the original model (enum or string)
                config_manager=self._config_manager,
                logger=self._logger
            )
            
            # Set up conversation manager
            self._conversation_manager = ConversationManager(self._logger)
            
            # Set system prompt
            self._system_prompt = system_prompt or self._get_default_system_prompt()
            
            # Add system prompt if provided
            if self._system_prompt:
                self._conversation_manager.add_message(
                    role="system",
                    content=self._system_prompt
                )
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AI: {str(e)}")
            raise AISetupError(f"AI initialization failed: {str(e)}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on model configuration."""
        # This could be extended to load from a prompt library
        return f"You are a helpful AI assistant using the {self._model_config.model_id} model."
    
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
        self._logger.info(f"Processing request: {prompt[:50]}...")
        
        try:
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Get response from provider
            response = self._provider.request(self._conversation_manager.get_messages(), **options)
            
            # Add assistant message
            self._conversation_manager.add_message(
                role="assistant",
                content=response.get('content', '')
            )
            
            # Update conversation history
            self._conversation_manager.add_interaction(prompt, response.get('content', ''))
            
            return response.get('content', '')
            
        except Exception as e:
            self._logger.error(f"Request failed: {str(e)}")
            raise AIProcessingError(f"Request failed: {str(e)}")
    
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
        self._logger.info(f"Processing streaming request: {prompt[:50]}...")
        
        try:
            # Add user message
            self._conversation_manager.add_message(role="user", content=prompt)
            
            # Stream the response
            content = self._provider.stream(self._conversation_manager.get_messages(), **options)
            
            # Add assistant message
            self._conversation_manager.add_message(role="assistant", content=content)
            
            # Update conversation history
            self._conversation_manager.add_interaction(prompt, content)
            
            return content
            
        except Exception as e:
            self._logger.error(f"Streaming failed: {str(e)}")
            raise AIProcessingError(f"Streaming failed: {str(e)}")
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
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
            List of messages
        """
        return self._conversation_manager.get_messages()