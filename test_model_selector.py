#!/usr/bin/env python
import sys
sys.path.insert(0, '.')
from src.core.model_selector import ModelSelector, UseCase
from src.config import UnifiedConfig

def main():
    print("Testing ModelSelector with UnifiedConfig...")
    
    # Create ModelSelector
    selector = ModelSelector()
    print("ModelSelector initialized successfully")
    
    # Test getting a system prompt
    try:
        system_prompt = selector.get_system_prompt(UseCase.CHAT)
        print('Got system prompt:', system_prompt[:30] + '...')
    except Exception as e:
        print('Error getting system prompt:', e)
    
    # Test other methods
    try:
        # Get available models from UnifiedConfig
        config = UnifiedConfig.get_instance()
        models = config.get_all_models()
        print(f"Found {len(models)} models in configuration")
    except Exception as e:
        print('Error getting models:', e)

if __name__ == "__main__":
    main() 