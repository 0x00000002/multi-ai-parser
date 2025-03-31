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
from ..prompts import PromptManager

# Default system prompt if none provided
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer the user's questions accurately and concisely."""


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
                 request_id: Optional[str] = None,
                 prompt_manager: Optional[PromptManager] = None):
        """
        Initialize the base AI implementation.
        
        Args:
            model: The model to use (Model enum or string ID, or None for default)
            system_prompt: Custom system prompt (or None for default)
            config_manager: Configuration manager instance
            logger: Logger instance
            request_id: Unique identifier for tracking this session
            prompt_manager: PromptManager instance for templated prompts
            
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
            self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            
            # Add system prompt if provided
            if self._system_prompt:
                self._conversation_manager.add_message(
                    role="system",
                    content=self._system_prompt
                )
            
            # Initialize prompt manager if provided
            self._prompt_manager = prompt_manager
            
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
                show_thinking=self._config_manager.show_thinking
            )
            
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
                show_thinking=self._config_manager.show_thinking
            )
            
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

    def request_with_template(self, 
                             template_id: str, 
                             variables: Optional[Dict[str, Any]] = None,
                             user_id: Optional[str] = None,
                             **options) -> str:
        """
        Make a request using a prompt template.
        
        Args:
            template_id: ID of the template to use
            variables: Variables for the template
            user_id: User ID for A/B testing
            options: Additional options for the request
            
        Returns:
            The model's response as a string
            
        Raises:
            ValueError: If prompt_manager is not set or template not found
        """
        if not self._prompt_manager:
            raise ValueError("No prompt manager set. Use set_prompt_manager() first.")
        
        # Start timing for metrics
        import time
        start_time = time.time()
        
        # Get rendered prompt from template
        rendered_prompt, usage_id = self._prompt_manager.render_prompt(
            template_id=template_id,
            variables=variables,
            user_id=user_id,
            context={"model": self._model_config.model_id}
        )
        
        self._logger.info(f"Using template {template_id} with variables: {variables}")
        
        # Make the request with the rendered prompt
        response = self.request(rendered_prompt, **options)
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        
        # Record metrics if we have a usage ID
        if usage_id:
            # Extract token counts if available in options
            metrics = {
                "latency": latency,
                "model": self._model_config.model_id
            }
            
            # Add token counts if response has them
            if hasattr(response, 'usage') and response.usage:
                metrics.update({
                    "prompt_tokens": response.usage.get("prompt_tokens", 0),
                    "completion_tokens": response.usage.get("completion_tokens", 0),
                    "total_tokens": response.usage.get("total_tokens", 0)
                })
            
            # Record performance
            self._prompt_manager.record_prompt_performance(
                usage_id=usage_id,
                metrics=metrics
            )
        
        return response
    
    def stream_with_template(self, 
                           template_id: str, 
                           variables: Optional[Dict[str, Any]] = None,
                           user_id: Optional[str] = None,
                           **options) -> str:
        """
        Stream a response using a prompt template.
        
        Args:
            template_id: ID of the template to use
            variables: Variables for the template
            user_id: User ID for A/B testing
            options: Additional options for the request
            
        Returns:
            The complete streamed response
            
        Raises:
            ValueError: If prompt_manager is not set or template not found
        """
        if not self._prompt_manager:
            raise ValueError("No prompt manager set. Use set_prompt_manager() first.")
        
        # Start timing for metrics
        import time
        start_time = time.time()
        
        # Get rendered prompt from template
        rendered_prompt, usage_id = self._prompt_manager.render_prompt(
            template_id=template_id,
            variables=variables,
            user_id=user_id,
            context={"model": self._model_config.model_id}
        )
        
        self._logger.info(f"Streaming with template {template_id} with variables: {variables}")
        
        # Make the streaming request with the rendered prompt
        response = self.stream(rendered_prompt, **options)
        
        # Calculate metrics
        end_time = time.time()
        latency = end_time - start_time
        
        # Record metrics if we have a usage ID
        if usage_id:
            metrics = {
                "latency": latency,
                "model": self._model_config.model_id,
                "streaming": True
            }
            
            # Record performance
            self._prompt_manager.record_prompt_performance(
                usage_id=usage_id,
                metrics=metrics
            )
        
        return response
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set a new system prompt.
        
        Args:
            system_prompt: New system prompt
        """
        self._system_prompt = system_prompt
        self._conversation_manager.set_system_prompt(system_prompt)
        self._logger.info("System prompt updated")
    
    def set_prompt_manager(self, prompt_manager: PromptManager) -> None:
        """
        Set the prompt manager for template-based prompts.
        
        Args:
            prompt_manager: PromptManager instance
        """
        self._prompt_manager = prompt_manager
        self._logger.info("Prompt manager set")