"""
Provider definitions for the AI framework.

This module provides dynamic provider enumeration based on the providers.yml file.
"""
from enum import Enum
from typing import Optional, Dict, Any
import os
import yaml
import logging


def _normalize_provider_name(name: str) -> str:
    """
    Normalize a provider name to a valid Python identifier.
    
    Args:
        name: Provider name to normalize
        
    Returns:
        Normalized name suitable for an enum
    """
    # Convert to uppercase and replace special characters with underscores
    normalized = name.upper().replace('-', '_').replace('.', '_')
    
    # Ensure it starts with a letter or underscore
    if not normalized[0].isalpha() and normalized[0] != '_':
        normalized = f"PROVIDER_{normalized}"
    
    # Replace any remaining invalid characters with underscores
    normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
    
    return normalized


def _load_providers_from_config(config_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Load provider configurations from YAML file.
    
    Args:
        config_dir: Optional directory containing configuration files
        
    Returns:
        Dictionary mapping enum names to provider identifiers
    """
    logger = logging.getLogger("dynamic_providers")
    
    # Determine the configuration directory
    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to providers configuration file
    providers_file = os.path.join(config_dir, "providers.yml")
    
    providers_dict = {}
    
    try:
        if os.path.exists(providers_file):
            with open(providers_file, 'r') as f:
                providers_data = yaml.safe_load(f)
                
            # Extract provider configurations
            if providers_data and "providers" in providers_data:
                for provider_id, _ in providers_data["providers"].items():
                    try:
                        # Create a normalized name for the enum
                        enum_name = _normalize_provider_name(provider_id)
                        
                        # Check for duplicates
                        if enum_name in providers_dict:
                            # If duplicate, append a suffix
                            enum_name = f"{enum_name}_{len([k for k in providers_dict if k.startswith(enum_name)])}"
                        
                        # Map enum name to provider ID
                        providers_dict[enum_name] = provider_id
                    except Exception as e:
                        logger.warning(f"Error processing provider {provider_id}: {str(e)}")
        else:
            logger.warning(f"Providers configuration file not found: {providers_file}")
            
            # Add common providers as fallback
            providers_dict = {
                "OPENAI": "openai",
                "ANTHROPIC": "anthropic",
                "GEMINI": "gemini",
                "OLLAMA": "ollama"
            }
            
    except Exception as e:
        logger.error(f"Error loading providers configuration: {str(e)}")
        
        # Add common providers as fallback
        providers_dict = {
            "OPENAI": "openai",
            "ANTHROPIC": "anthropic",
            "GEMINI": "gemini",
            "OLLAMA": "ollama"
        }
    
    return providers_dict


def create_provider_enum() -> type:
    """
    Create a dynamic Provider enumeration from configuration.
    
    Returns:
        Provider enum class with members generated from configuration
    """
    # Load provider configurations
    providers_dict = _load_providers_from_config()
    
    # Create the enum dynamically
    return Enum('Provider', providers_dict)


# Dynamically create the Provider enum
Provider = create_provider_enum()


def get_available_providers() -> list:
    """
    Get list of available provider identifiers.
    
    Returns:
        List of provider identifiers
    """
    return [provider.value for provider in Provider]


def get_provider_for_model(model_id: str, provider_type: str) -> Optional[Provider]:
    """
    Get the Provider enum value for a specific model and provider type.
    
    Args:
        model_id: Model identifier (e.g., "gpt-4o", "claude-3-opus")
        provider_type: Provider type (e.g., "openai", "anthropic")
        
    Returns:
        Provider enum value or None if not found
    """
    # Normalize the provider type
    provider_type = provider_type.lower()
    
    # Look for exact match
    for provider in Provider:
        if provider.value == provider_type:
            return provider
    
    # If not found, try to match by prefix
    for provider in Provider:
        if provider.value.startswith(provider_type):
            return provider
    
    return None 