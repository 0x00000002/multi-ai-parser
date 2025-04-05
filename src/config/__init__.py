"""
Configuration System for Agentic AI

Provides a clean API for configuring the AI framework.
"""
from typing import Optional, Dict, Any, Union
from pathlib import Path
import os

from .user_config import UserConfig, UseCasePreset
from .unified_config import UnifiedConfig
from .dynamic_models import Model, get_available_models, is_valid_model, get_model_enum
from .provider import Provider, get_available_providers, get_provider_for_model

# Export key classes and functions for public API
__all__ = [
    'UserConfig',
    'UseCasePreset',
    'Model',
    'Provider',
    'get_config',
    'configure',
    'get_available_models',
    'is_valid_model',
    'get_model_enum',
    'get_available_providers',
    'get_provider_for_model',
]

def configure(model: Optional[str] = None,
              use_case: Optional[Union[str, UseCasePreset]] = None,
              temperature: Optional[float] = None,
              config_file: Optional[Union[str, Path]] = None,
              system_prompt: Optional[str] = None,
              show_thinking: Optional[bool] = None,
              **kwargs) -> None:
    """
    Configure the AI framework with user overrides.
    
    This function applies user overrides to the configuration system.
    It should be called before creating any AI or agent instances.
    
    Args:
        model: Model identifier (e.g., "claude-3-opus", "phi4")
        use_case: Predefined use case configuration (affects quality and speed)
        temperature: Model temperature (0.0 to 1.0)
        config_file: Path to external configuration file
        system_prompt: Custom system prompt
        show_thinking: Whether to include AI reasoning in responses
        **kwargs: Additional configuration overrides
    """
    user_config = UserConfig(
        model=model,
        use_case=use_case,
        temperature=temperature,
        config_file=config_file,
        system_prompt=system_prompt,
        show_thinking=show_thinking,
        **kwargs
    )
    
    # Get the UnifiedConfig instance with user overrides
    UnifiedConfig.get_instance(user_config=user_config)


def get_config() -> UnifiedConfig:
    """
    Get the current configuration instance.
    
    Returns:
        UnifiedConfig instance
    """
    return UnifiedConfig.get_instance()
