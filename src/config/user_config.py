"""
User Configuration Interface for Agentic AI

This module provides a clean, user-facing interface for configuring the AI framework
without exposing implementation details.
"""
from typing import Optional, Dict, Any, Union
from pathlib import Path
from enum import Enum, auto


class UseCasePreset(str, Enum):
    """Predefined use case configurations."""
    CHAT = "chat"
    CODING = "coding"
    SOLIDITY_CODING = "solidity_coding"
    TRANSLATION = "translation"
    CONTENT_GENERATION = "content_generation"
    DATA_ANALYSIS = "data_analysis"
    WEB_ANALYSIS = "web_analysis"
    IMAGE_GENERATION = "image_generation"


class UserConfig:
    """
    User-facing configuration interface for the AI framework.
    
    This class provides a clean API for users to configure the AI framework
    without needing to understand implementation details.
    """
    
    def __init__(self,
                 model: Optional[str] = None,
                 use_case: Optional[Union[str, UseCasePreset]] = None,
                 temperature: Optional[float] = None,
                 config_file: Optional[Union[str, Path]] = None,
                 system_prompt: Optional[str] = None,
                 show_thinking: Optional[bool] = None,
                 **kwargs):
        """
        Initialize user configuration with optional overrides.
        
        Args:
            model: Model identifier (e.g., "claude-3-opus", "phi4")
            use_case: Predefined use case configuration (affects quality and speed)
            temperature: Model temperature (0.0 to 1.0)
            config_file: Path to external configuration file
            system_prompt: Custom system prompt
            show_thinking: Whether to include AI reasoning in responses
            **kwargs: Additional configuration overrides
        """
        self.model = model
        self.use_case = use_case
        self.temperature = temperature
        self.config_file = Path(config_file) if config_file else None
        self.system_prompt = system_prompt
        self.show_thinking = show_thinking
        self.overrides = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            "model": self.model,
            "use_case": self.use_case,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "show_thinking": self.show_thinking,
        }
        
        # Filter out None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Add additional overrides
        config_dict.update(self.overrides)
        
        return config_dict
    
    @classmethod
    def from_file(cls, config_file: Union[str, Path]) -> 'UserConfig':
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file (JSON or YAML)
            
        Returns:
            UserConfig instance
            
        Raises:
            ValueError: If file format is unsupported or file cannot be read
        """
        import yaml
        import json
        
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        try:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            return cls(**config_data)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {str(e)}") 