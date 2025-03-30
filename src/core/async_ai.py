"""
Async AI implementation that handles asynchronous operations.
"""
from typing import Dict, List, Any, Optional, Union, AsyncIterator
from .interfaces import AsyncAIInterface, AsyncProviderInterface
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.config_manager import ConfigManager
from ..exceptions import AISetupError, AIProcessingError
from ..conversation.conversation_manager import ConversationManager
from .provider_factory import ProviderFactory
from .base_ai import AIBase
import uuid


class AsyncAI(AIBase, AsyncAIInterface):
    """
    Asynchronous AI implementation.
    Extends the base AI with async capabilities.
    """
    
    def __init__(self, 
                 model_id: Optional[str] = None, 
                 system_prompt: Optional[str] = None,
                 config_manager: Optional[ConfigManager] = None,
                 logger: Optional[LoggerInterface] = None,
                 request_id: Optional[str] = None):
        """
        Initialize the async AI implementation.
        
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
            
            # Set up the async provider
            self._async_provider = AsyncProviderFactory.create(
                provider_type=self._model_config.provider,
                model_id=self._model_config.model_id,
                config_manager=self._config_manager,
                logger=self._logger
            )
            
            # Set up conversation manager
            self._conversation_manager = ConversationManager(self._logger)
            
            # Set system prompt
            self._system_prompt = system_prompt or self._get_default_system_prompt()
            
            # Initialize synchronous components for compatibility
            # This allows us to use the AsyncAI class with synchronous methods too
            super().__init__(
                model_id=model_id,
                system_prompt=system_prompt,
                config_manager=config_manager,
                logger=logger,
                request_id=request_id
            )
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AsyncAI: {str(e)}")
            raise AISetupError(f"AsyncAI initialization failed: {str(e)}")
    
    async def async_request(self, prompt: str, **options) -> str:
        """
        Make an asynchronous request to the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            The model's response as a string
            
        Raises:
            AIProcessingError: If the request fails
        """
        self._logger.info(f"Processing async request: {prompt[:50]}...")
        
        try:
            # Build messages with conversation history
            messages = self._build_messages(prompt)
            
            # Make the async request to the provider
            response = await self._async_provider.async_request(messages, **options)
            
            # Extract content from response
            content = self._extract_content(response)
            
            # Update conversation history
            self._conversation_manager.add_interaction(prompt, content)
            
            return content
            
        except Exception as e:
            self._logger.error(f"Async request failed: {str(e)}")
            raise AIProcessingError(f"Async request failed: {str(e)}")
    
    async def async_stream(self, prompt: str, **options) -> AsyncIterator[str]:
        """
        Stream a response asynchronously from the AI model.
        
        Args:
            prompt: The user prompt
            options: Additional options for the request
            
        Returns:
            An async iterator yielding response chunks
            
        Raises:
            AIProcessingError: If streaming fails
        """
        self._logger.info(f"Processing async streaming request: {prompt[:50]}...")
        
        try:
            # Build messages with conversation history
            messages = self._build_messages(prompt)
            
            # Stream the response asynchronously
            complete_content = ""
            async for chunk in self._async_provider.async_stream(messages, **options):
                complete_content += chunk
                yield chunk
            
            # Update conversation history with complete content
            self._conversation_manager.add_interaction(prompt, complete_content)
            
        except Exception as e:
            self._logger.error(f"Async streaming failed: {str(e)}")
            raise AIProcessingError(f"Async streaming failed: {str(e)}")
    
    async def async_reset_conversation(self) -> None:
        """Reset the conversation history asynchronously."""
        self._conversation_manager.reset()
        self._logger.info("Conversation history reset")