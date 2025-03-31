"""
Configuration management module for AI framework.
Centralizes all configuration handling with validation and multiple source support.
"""
import os
from typing import Dict, Any, Optional, Union, List, get_type_hints
from enum import Enum
import json
import yaml
import sys
import importlib
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, ConfigDict
import logging
from dotenv import load_dotenv
from ..utils.logger import LoggerInterface, LoggerFactory
from ..exceptions import AIConfigError

# Import the Model enum from models module
from .models import Model

# Configure module-level logger
logger = logging.getLogger(__name__)


class Privacy(str, Enum):
    """Privacy level for AI models."""
    LOCAL = "LOCAL"
    EXTERNAL = "EXTERNAL"


class Quality(str, Enum):
    """Quality level for AI models."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Speed(str, Enum):
    """Speed level for AI models."""
    FAST = "FAST"
    STANDARD = "STANDARD"
    SLOW = "SLOW"


class Provider(str, Enum):
    """Available AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    api_key_env: str = Field(..., description="Environment variable name for the API key")
    base_url: Optional[str] = Field(None, description="Base URL for API calls")
    timeout: int = Field(30, description="Request timeout in seconds")
    
    @property
    def api_key(self) -> Optional[str]:
        """Get the API key from environment variables."""
        return os.getenv(self.api_key_env)


class CostConfig(BaseModel):
    """Configuration for model costs."""
    input_tokens: float
    output_tokens: float
    minimum_cost: float


class ModelConfig(BaseModel):
    """Configuration for an AI model."""
    model_config = ConfigDict(validate_assignment=True)
    
    name: Optional[str] = Field(None, description="User-friendly name for the model")
    model_id: str = Field(..., description="Model identifier used in API calls")
    provider: Provider = Field(..., description="Provider name for this model")
    privacy: Privacy = Field(..., description="Privacy level (LOCAL/EXTERNAL)")
    quality: Quality = Field(..., description="Quality level (LOW/MEDIUM/HIGH)")
    speed: Speed = Field(..., description="Speed level (FAST/STANDARD/SLOW)")
    max_tokens: int = Field(1024, description="Default maximum tokens for generation")
    temperature: float = Field(0.7, description="Default temperature setting")
    cost: CostConfig = Field(..., description="Cost configuration for the model")
    use_cases: List[str] = Field(default_factory=list, description="List of supported use cases")


class UseCaseConfig(BaseModel):
    """Configuration for a use case."""
    quality: Quality = Field(..., description="Required quality level (HIGH/MEDIUM/LOW)")
    speed: Speed = Field(..., description="Required speed level (FAST/STANDARD/SLOW)")


class FrameworkConfig(BaseModel):
    """Global configuration for the AI framework."""
    providers: Dict[str, ProviderConfig] = Field(..., description="Provider configurations")
    models: Dict[str, ModelConfig] = Field(..., description="Model configurations")
    default_model: str = Field(..., description="Default model identifier")
    log_level: str = Field("INFO", description="Default logging level")
    use_cases: Dict[str, UseCaseConfig] = Field(..., description="Use case configurations")
    show_thinking: bool = Field(False, description="Whether to include AI thoughts in responses")
    
    @model_validator(mode='after')
    def validate_default_model(self) -> 'FrameworkConfig':
        """Validate that the default model exists in the models dict."""
        if self.default_model not in self.models:
            raise ValueError(f"Default model '{self.default_model}' not found in models.")
        return self


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
                 env_prefix: str = "AI_",
                 show_thinking: bool = False,
                 logger: LoggerInterface = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (JSON/YAML)
            env_prefix: Prefix for environment variables
            show_thinking: Whether to include AI thoughts in responses
        """
        if self._initialized:
            return
            
        # Load environment variables first
        load_dotenv()
        
        # Set up logger - use NullLogger by default
        self._logger = logger or LoggerFactory.create("config_manager", use_real_logger=False)
        
        self._env_prefix = env_prefix
        self._config_path = config_path
        self._show_thinking = show_thinking
        self._config = None
        self._load_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        """Load configuration from file and/or environment variables."""
        config_data = {}
        
        # First try to load from file if specified
        if self._config_path:
            path = Path(self._config_path)
            self._logger.info(f"Loading config from: {path}")
            if path.exists():
                with open(path, 'r') as f:
                    if path.suffix.lower() == '.json':
                        config_data = json.load(f)
                    elif path.suffix.lower() in ['.yml', '.yaml']:
                        config_data = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {path.suffix}")
                self._logger.info(f"Loaded config data with keys: {list(config_data.keys())}")
                
            else:
                self._logger.warning(f"Config file not found at: {path}")
        
        # Override with environment variables
        env_config = {k[len(self._env_prefix):]: v 
                     for k, v in os.environ.items() 
                     if k.startswith(self._env_prefix)}
        self._logger.info(f"Environment config: {env_config}")
        
        # Merge configurations with environment taking precedence
        config_data.update(env_config)
        
        # Validate that all configured models exist in the static Model enum
        if 'models' in config_data:
            model_enum_values = {m.value for m in Model}
            configured_model_keys = set(config_data['models'].keys())
        
            missing_in_enum = configured_model_keys - model_enum_values
            if missing_in_enum:
                self._logger.error(f"Models found in configuration but not defined in Model enum: {missing_in_enum}. Please update src/config/models.py.")
                raise AIConfigError(f"Models found in configuration but not defined in Model enum: {missing_in_enum}. Please update src/config/models.py.")

            # Optional: Check if all enum models are in config (might be too strict, commenting out for now)
            # missing_in_config = model_enum_values - configured_model_keys
            # if missing_in_config:
            #     self._logger.warning(f"Models defined in Model enum but not found in configuration: {missing_in_config}")

        # Add show_thinking from initialization
        config_data['show_thinking'] = self._show_thinking
        
        # Validate the configuration using Pydantic
        try:
            self._config = FrameworkConfig(**config_data)
            self._logger.info("Configuration validated successfully")
        except Exception as e:
            self._logger.error(f"Configuration validation failed: {str(e)}")
            raise AIConfigError(f"Invalid configuration: {str(e)}")
    
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
    
    def get_model_by_name(self, name_or_id: str) -> ModelConfig:
        """
        Get configuration for a model by its key (ID) or user-friendly name.
        Prioritizes matching the key, then falls back to matching the name field.
        
        Args:
            name_or_id: The model key (ID) or user-friendly name
            
        Returns:
            ModelConfig object for the specified model
            
        Raises:
            ValueError: If no model with the given key or name is found
        """
        # First, try matching by key (model_id)
        if name_or_id in self._config.models:
            return self._config.models[name_or_id]
            
        # If not found by key, try matching by the 'name' field
        for model_id, model_config in self._config.models.items():
            if model_config.name and model_config.name == name_or_id:
                return model_config
                
        raise ValueError(f"No model found with key or name '{name_or_id}'")
    
    def get_all_models(self) -> Dict[str, ModelConfig]:
        """
        Get all model configurations.
        
        Returns:
            Dictionary of all model configurations
        """
        return self._config.models
    
    def get_use_case_config(self, use_case: str) -> Dict[str, str]:
        """
        Get configuration for a specific use case.
        
        Args:
            use_case: The use case name
            
        Returns:
            Dictionary containing quality and speed requirements
            
        Raises:
            ValueError: If the use case is not found
        """
        if use_case not in self._config.use_cases:
            raise ValueError(f"Use case '{use_case}' not found in configuration")
        use_case_config = self._config.use_cases[use_case]
        return {
            'quality': use_case_config.quality,
            'speed': use_case_config.speed
        }
    
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
        
    @property
    def show_thinking(self) -> bool:
        """Get the show_thinking setting."""
        return self._config.show_thinking
        
    @show_thinking.setter
    def show_thinking(self, value: bool) -> None:
        """Set the show_thinking setting."""
        self._config.show_thinking = value
        self._show_thinking = value
