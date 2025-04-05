#!/usr/bin/env python
"""
Configuration Example for Agentic AI

This example demonstrates how to use the new configuration system.
"""
import os
import sys
import logging
from pathlib import Path

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("configuration_example")

def example_basic_usage():
    """Example of basic configuration usage."""
    from src.config import configure, get_config, UseCasePreset
    
    # Configure with specific model
    configure(model="phi4")
    
    # Get the configuration instance
    config = get_config()
    
    # Use the configuration
    print(f"Default model: {config.get_default_model()}")
    
    # Get model configuration
    model_config = config.get_model_config()
    print(f"Model details: {model_config['name']} ({model_config['provider']})")
    
    # Use a predefined use case
    configure(use_case=UseCasePreset.SOLIDITY_CODING)
    
    # Get use case configuration
    use_case_config = config.get_use_case_config()
    print(f"Use case configuration: {use_case_config}")


def example_use_cases():
    """Example of using different use cases."""
    from src.config import configure, get_config, UseCasePreset
    
    # Try different use cases
    use_cases = [
        UseCasePreset.CHAT,
        UseCasePreset.CODING,
        UseCasePreset.SOLIDITY_CODING,
        UseCasePreset.DATA_ANALYSIS
    ]
    
    for use_case in use_cases:
        configure(use_case=use_case)
        config = get_config()
        use_case_config = config.get_use_case_config()
        print(f"{use_case.value}: quality={use_case_config.get('quality')}, speed={use_case_config.get('speed')}")


def example_model_listing():
    """Example of listing available models."""
    from src.config import get_available_models, get_config
    
    # Get all available models
    models = get_available_models()
    config = get_config()
    
    print("Available models:")
    for model_id in models:
        try:
            model_config = config.get_model_config(model_id)
            print(f"- {model_id}: {model_config.get('name', 'Unnamed')} "
                  f"({model_config.get('privacy', 'unknown')} privacy, "
                  f"{model_config.get('quality', 'unknown')} quality)")
        except Exception as e:
            print(f"- {model_id}: Error retrieving details - {str(e)}")


def example_custom_settings():
    """Example of using custom configuration settings."""
    from src.config import configure, get_config
    
    # Configure with additional custom settings
    configure(
        model="claude-3-5-sonnet",
        temperature=0.8,
        system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
        show_thinking=True,
        max_tokens=2000,  # Custom setting passed through kwargs
        custom_setting="value"  # Custom setting passed through kwargs
    )
    
    # Use the configuration
    config = get_config()
    print(f"System prompt: {config.get_system_prompt()}")
    print(f"Show thinking: {config.show_thinking}")


def run_examples():
    """Run all configuration examples."""
    print("\n=== Basic Usage ===")
    example_basic_usage()
    
    print("\n=== Use Cases ===")
    example_use_cases()
    
    print("\n=== Model Listing ===")
    example_model_listing()
    
    print("\n=== Custom Settings ===")
    example_custom_settings()


if __name__ == "__main__":
    run_examples() 