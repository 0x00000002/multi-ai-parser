"""
Factory for creating AI providers.
"""
from typing import Optional, Union, Dict, Any
from ..utils.logger import LoggerInterface, LoggerFactory
from ..config.config_manager import ConfigManager
from ..config.models import Model
from ..exceptions import AISetupError
from .providers.base_provider import BaseProvider
from .providers.anthropic_provider import AnthropicProvider
from .providers.openai_provider import OpenAIProvider
from .providers.ollama_provider import OllamaProvider
from .providers.gemini_provider import GeminiProvider


class ProviderFactory:
    """Factory for creating AI providers."""
    
    _providers = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider
    }
    
    @classmethod
    def create(cls,
               provider_type: Optional[str] = None,
               model_id: Optional[Union[Model, str]] = None,
               config_manager: Optional[ConfigManager] = None,
               logger: Optional[LoggerInterface] = None) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            provider_type: Type of provider to create (optional if model_id is a config key)
            model_id: Model to use (can be Model enum, config key, or model ID)
            config_manager: Configuration manager instance
            logger: Logger instance
            
        Returns:
            Provider instance
            
        Raises:
            AISetupError: If provider creation fails
        """
        try:
            # Create logger if not provided
            if logger is None:
                logger = LoggerFactory.create()
                
            # Create config manager if not provided
            if config_manager is None:
                config_manager = ConfigManager()
            
            # Handle the model_id parameter
            model_key = None
            if isinstance(model_id, Model):
                # If it's a Model enum, get its value (which is the config key)
                model_key = model_id.value
                logger.debug(f"Using Model enum value: {model_key}")
            else:
                # Assume it's a string model key/ID
                model_key = model_id
            
            # Get model configuration to resolve the actual model_id and provider
            model_config = config_manager.get_model_config(model_key)
            
            # Use provider from model config if not specified
            provider_type = provider_type or model_config.provider
            
            # Get provider class
            provider_class = cls._providers.get(provider_type.lower())
            if not provider_class:
                raise AISetupError(f"Unsupported provider type: {provider_type}")
            
            # Create and return provider instance with resolved model_id
            logger.info(f"Creating provider {provider_type} for model {model_key} ({model_config.model_id})")
            return provider_class(
                model_id=model_config.model_id,  # Use the actual model_id from config
                config_manager=config_manager,
                logger=logger
            )
            
        except Exception as e:
            raise AISetupError(f"Failed to create provider: {str(e)}")
    
    @staticmethod
    def get_content(response: Union[Dict[str, Any], str]) -> str:
        """
        Extract content from a provider response.
        
        Args:
            response: Provider response (dictionary or string)
            
        Returns:
            Content string
        """
        return BaseProvider.get_content(response) 