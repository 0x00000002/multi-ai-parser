"""
Dynamic Model Generation for Agentic AI

This module provides a dynamically generated Model enumeration based on 
the models defined in the configuration files. It replaces the hardcoded
Model enumeration in models.py.
"""
from typing import Dict, List, Any, Optional, Set
from enum import Enum, auto
import logging
import os
import yaml


# Define quality levels
class Quality(Enum):
    """Quality levels for models."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Define speed levels
class Speed(Enum):
    """Speed levels for models."""
    FAST = "FAST"
    STANDARD = "STANDARD"
    SLOW = "SLOW"


# Define privacy levels
class Privacy(Enum):
    """Privacy levels for models."""
    LOCAL = "LOCAL"
    EXTERNAL = "EXTERNAL"


def _normalize_name(name: str) -> str:
    """
    Normalize a model name to a valid Python identifier suitable for an enum.
    
    Args:
        name: Model name to normalize
        
    Returns:
        Normalized name (uppercase, with dashes and dots replaced by underscores)
    """
    # Convert to uppercase and replace special characters with underscores
    normalized = name.upper().replace('-', '_').replace('.', '_')
    
    # Ensure it starts with a letter or underscore
    if not normalized[0].isalpha() and normalized[0] != '_':
        normalized = f"MODEL_{normalized}"
    
    # Replace any remaining invalid characters with underscores
    normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
    
    return normalized


def _load_models_from_config(config_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Load model configurations from YAML file.
    
    Args:
        config_dir: Optional directory containing configuration files
            
    Returns:
        Dictionary mapping enum names to model IDs
    """
    logger = logging.getLogger("dynamic_models")
    
    # Determine the configuration directory
    if config_dir is None:
        config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to models configuration file
    models_file = os.path.join(config_dir, "models.yml")
    
    models_dict = {}
    
    try:
        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                models_data = yaml.safe_load(f)
                
            # Extract model configurations
            if models_data and "models" in models_data:
                for model_id in models_data["models"]:
                    try:
                        # Create a normalized name for the enum
                        enum_name = _normalize_name(model_id)
                        
                        # Check for duplicates (unlikely but possible after normalization)
                        if enum_name in models_dict:
                            # If duplicate, append a suffix
                            enum_name = f"{enum_name}_{len([k for k in models_dict if k.startswith(enum_name)])}"
                        
                        # Map enum name to model ID
                        models_dict[enum_name] = model_id
                    except Exception as e:
                        logger.warning(f"Error processing model {model_id}: {str(e)}")
        else:
            logger.warning(f"Models configuration file not found: {models_file}")
            
    except Exception as e:
        logger.error(f"Error loading models configuration: {str(e)}")
    
    return models_dict


def create_model_enum() -> type:
    """
    Create a dynamic Model enumeration from configuration.
    
    Returns:
        Model enum class with members generated from configuration
    """
    # Load model configurations
    models_dict = _load_models_from_config()
    
    # Create the enum dynamically
    return Enum('Model', models_dict)


# Dynamically create the Model enum - this happens at import time
Model = create_model_enum()


def get_available_models() -> List[str]:
    """
    Get a list of available model IDs.
    
    Returns:
        List of model IDs
    """
    return [model.value for model in Model]


def is_valid_model(model_id: str) -> bool:
    """
    Check if a model ID is valid.
    
    Args:
        model_id: Model ID to check
        
    Returns:
        True if valid, False otherwise
    """
    return model_id in get_available_models()


def get_model_enum(model_id: str) -> Optional[Model]:
    """
    Get the Model enum member for a given model ID.
    
    Args:
        model_id: Model ID
        
    Returns:
        Model enum member or None if not found
    """
    for model in Model:
        if model.value == model_id:
            return model
    return None 