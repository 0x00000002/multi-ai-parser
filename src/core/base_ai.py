"""
Base AI implementation that handles common functionality.
Implements the AIInterface and provides core conversation features.
"""
from typing import Dict, List, Any, Optional, Union
from .interfaces import AIInterface, ProviderInterface, LoggerInterface
from ..config.config_manager import ConfigManager
from ..exceptions import AISetupError, AIProcessingError
from ..conversation.conversation_manager import ConversationManager
from ..providers.provider_factory import ProviderFactory
from ..utils.logger import LoggerFactory
import uuid


class AIBase(AIInterface):
    """
    Base implementation of the AI interface.
    Handles provider management, conversation history, and common operations.
    """
    
    def __init__(self, 
                 model_id: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 config_manager: Optional[ConfigManager] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None):
        """
        Initialize the base AI implementation.
        
        Args:
            model_id: The model to use (or None for default)
            system_prompt: Custom system prompt (or None for default)
            config_manager: Configuration manager instance
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            
        Raises:
            AISetupError: If initialization fails
        """
        self._request_id = request_id or str(uuid.uuid4())
        self._logger = logger or LoggerFactory.create(request_id=self._request_id)
        self._config_manager = config_manager or ConfigManager()
        
        try:
            # Get model configuration
            self._model_config = self._config_manager.get_model_config(model_id)
            self._logger.info(f"Using model: {self._model_config.model_id}")
            
            # Set up the provider
            self._provider = ProviderFactory.create(
                provider_type=self._model_config.provider,
                model_id=self._model_config.model_id,
                config_manager=self._config_manager,
                logger=self._logger
            )
            
            # Set up conversation manager
            self._conversation_manager = ConversationManager(self._logger)
            
            # Set system prompt
            self._system_prompt = system_prompt or self._get_default_system_prompt()
            
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
            # Build messages with conversation history
            messages = self._build_messages(prompt)
            
            # Make the request to the provider
            response = self._provider.request(messages, **options)
            
            # Extract content from response
            content = self._extract_content(response)
            
            # Update conversation history
            self._conversation_manager.add_interaction(prompt, content)
            
            return content
            
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
            # Build messages with conversation history
            messages = self._build_messages(prompt)
            
            # Stream the response
            content = self._provider.stream(messages, **options)
            
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
    
    def _build_messages(self, prompt: str) -> List[Dict[str, Any]]:
        """
        Build the message list for the provider.
        
        Args:
            prompt: The current user prompt
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system prompt if set
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        
        # Add conversation history
        history = self._conversation_manager.get_messages()
        messages.extend(history)
        
        # Add the current user prompt
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """
        Extract content text from provider response.
        
        Args:
            response: Provider response object
            
        Returns:
            Content string
        """
        # This is a simple implementation
        # A more robust one would handle different response formats
        return response.get("content", "")