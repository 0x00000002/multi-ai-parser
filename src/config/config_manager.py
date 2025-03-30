"""
Configuration management module for AI framework.
Centralizes all configuration handling with validation and multiple source support.
"""
import os
from typing import Dict, Any, Optional, Union
from enum import Enum
import json
from pathlib import Path
from pydantic import BaseModel, Field, root_validator


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    api_key_env: str = Field(..., description="Environment variable name for the API key")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    timeout: int = Field(30, description="Request timeout in seconds")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key from environment variables."""
        return os.getenv(self.api_key_env)


class ModelConfig(BaseModel):
    """Configuration for an AI model."""
    model_id: str = Field(..., description="Model identifier used in API calls")
    provider: str = Field(..., description="Provider name for this model")
    privacy: str = Field(..., description="Privacy level (LOCAL/EXTERNAL)")
    quality: str = Field(..., description="Quality level (LOW/MEDIUM/HIGH)")
    speed: str = Field(..., description="Speed level (FAST/STANDARD/SLOW)")
    max_tokens: int = Field(1024, description="Default maximum tokens for generation")
    temperature: float = Field(0.7, description="Default temperature setting")
    
    class Config:
        validate_assignment = True


class FrameworkConfig(BaseModel):
    """Global configuration for the AI framework."""
    providers: Dict[str, ProviderConfig] = Field(..., description="Provider configurations")
    models: Dict[str, ModelConfig] = Field(..., description="Model configurations")
    default_model: str = Field(..., description="Default model identifier")
    log_level: str = Field("INFO", description="Default logging level")
    
    @root_validator
    def validate_default_model(cls, values):
        """Validate that the default model exists in the models dict."""
        if values.get('default_model') not in values.get('models', {}):
            raise ValueError(f"Default model '{values.get('default_model')}' not found in models.")
        return values


class ConfigManager:
    """
    Centralized configuration manager.
    Handles loading, validation, and access to configuration settings.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None, 
                 env_prefix: str = "AI_"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON/YAML)
            env_prefix: Prefix for environment variables
        """
        if self._initialized:
            return
            
        self._env_prefix = env_prefix
        self._config_path = config_path
        self._config = None
        self._load_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from file and/or environment variables."""
        config_data = {}
        
        # First try to load from file if specified
        if self._config_path:
            path = Path(self._config_path)
            if path.exists():
                with open(path, 'r') as f:
                    if path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    else:
                        # Add support for YAML if needed
                        pass
        
        # Override with environment variables
        env_config = {k[len(self._env_prefix):]: v 
                     for k, v in os.environ.items() 
                     if k.startswith(self._env_prefix)}
        
        # Merge configurations with environment taking precedence
        # This is a simple implementation; a more robust one would handle nested keys
        config_data.update(env_config)
        
        # Validate the configuration
        try:
            self._config = FrameworkConfig(**config_data)
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    def reload(self) -> None:
        """Reload configuration from sources."""
        self._load_config()
    
    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        """
        Get configuration for a specific model.
        
        Args:
            model_id: The model identifier, or None for default
            
        Returns:
            ModelConfig object for the specified model
            
        Raises:
            ValueError: If the model is not found
        """
        model_id = model_id or self._config.default_model
        if model_id not in self._config.models:
            raise ValueError(f"Model '{model_id}' not found in configuration")
        return self._config.models[model_id]
    
    def get_provider_config(self, provider_name: str) -> ProviderConfig:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: The provider name
            
        Returns:
            ProviderConfig object for the specified provider
            
        Raises:
            ValueError: If the provider is not found
        """
        if provider_name not in self._config.providers:
            raise ValueError(f"Provider '{provider_name}' not found in configuration")
        return self._config.providers[provider_name]
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider_name: The provider name
            
        Returns:
            API key string or None if not found
        """
        provider_config = self.get_provider_config(provider_name)
        return provider_config.api_key
    
    @property
    def log_level(self) -> str:
        """Get the configured log level."""
        return self._config.log_level
