"""
Example of using the AI system with different approaches.
"""
import os
import sys
import importlib
from pathlib import Path

# Add the src directory to the path so we can import from there
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)

# First load config and initialize the Model enum
from src.config.config_manager import ConfigManager
config_path = os.path.join(src_path, "src", "config", "config.yml")
config_manager = ConfigManager(config_path=config_path)

# Now import the Model enum after it's been updated
from src.config.models import Model
from src.core.model_selector import ModelSelector, UseCase
from src.core.tool_enabled_ai import ToolEnabledAI
from src.core.provider_factory import ProviderFactory
from src.utils.logger import LoggerFactory


def main():
    """Run various examples of using the AI system."""
    # Set up logger
    logger = LoggerFactory.create()
    
    # Print available models from the enum
    print("\nAvailable models:")
    for model in Model:
        print(f" - {model.name} ({model.value})")
    
    # For debugging, print the actual config data
    models = config_manager.get_all_models()
    print("\nModels from config:")
    for model_id, model_config in models.items():
        print(f" - {model_id}: {model_config.model_id} ({model_config.provider})")
    
    print(f"\n=== Example 1: Direct Model Usage with Claude ===")
    # Create AI instance with specific model
    ai = ToolEnabledAI(
        model=Model.CLAUDE_3_7_SONNET,  # Use Model enum
        config_manager=config_manager,
        logger=logger
    )
    response = ai.request("What is the capital of France?")
    print(f"Response: {response}")
    
    print("\n=== Example 2: Using Provider Factory with Claude ===")
    # Create provider directly using factory
    provider = ProviderFactory.create(
        model_id=Model.CLAUDE_3_7_SONNET,  # Use Model enum directly
        config_manager=config_manager,
        logger=logger
    )
    response = provider.request("What is the best programming language?")
    # Response is already a string now
    print(f"Response: {response}")
    
    print("\n=== Example 3: Using Tool-Enabled AI with Claude ===")
    # Create tool-enabled AI with Claude
    tool_ai = ToolEnabledAI(
        model=Model.CLAUDE_3_7_SONNET,
        config_manager=config_manager,
        logger=logger
    )
    
    # Register a simple calculator tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    tool_ai.register_tool(
        tool_name="add_numbers",
        tool_function=add_numbers,
        description="Add two numbers together and return the sum",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    )
    
    # Response already includes the tool execution results
    response = tool_ai.request("What is 25 + 17?")
    print(f"Response: {response}")
    
    # Example with OpenAI
    print("\n=== Example 4: Using OpenAI Model ===")
    try:
        # Create tool-enabled AI with GPT-4
        openai_ai = ToolEnabledAI(
            model=Model.GPT_4O_MINI,  # Use OpenAI model
            config_manager=config_manager,
            logger=logger
        )
        
        # Register the same calculator tool
        openai_ai.register_tool(
            tool_name="add_numbers",
            tool_function=add_numbers,
            description="Add two numbers together and return the sum",
            parameters_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        )
        
        # Try with a different math problem
        response = openai_ai.request("What is 43 + 57?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"OpenAI example failed: {e}")


if __name__ == "__main__":
    main() 