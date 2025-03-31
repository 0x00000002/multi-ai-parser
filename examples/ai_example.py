"""
Example of using the AI system with different approaches.
"""
import os
import sys
import importlib
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path so we can import from there
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)

# Import necessary components
from src.config.config_manager import ConfigManager
from src.config.models import Model  # Model enum is now static
from src.core.model_selector import ModelSelector, UseCase
from src.core.tool_enabled_ai import AI
from src.core.provider_factory import ProviderFactory
from src.utils.logger import LoggerFactory
from src.prompts import PromptManager


def main():
    """Run various examples of using the AI system."""
    # Set up logger
    logger = LoggerFactory.create()
    
    # Initialize ConfigManager (this will also validate models against the enum)
    config_path = os.path.join(src_path, "src", "config", "config.yml")
    config_manager = ConfigManager(config_path=config_path)
    
    # Print available models from the enum
    print("\nAvailable models (from static Enum):")
    for model in Model:
        print(f" - {model.name} ({model.value})")
    
    # For debugging, print the actual config data
    models = config_manager.get_all_models()
    print("\nModels from config:")
    for model_id, model_config in models.items():
        print(f" - {model_id}: {model_config.model_id} ({model_config.provider})")
    
    print(f"\n=== Example 1: Direct Model Usage with Claude ===")
    # Create AI instance with specific model
    ai = AI(
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
    tool_ai = AI(
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
        openai_ai = AI(
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
    
    # Example with prompt management
    print("\n=== Example 5: Using Prompt Management System ===")
    try:
        # Create prompt storage directory
        prompt_storage_dir = os.path.join(src_path, "data", "prompts")
        os.makedirs(prompt_storage_dir, exist_ok=True)
        
        # Initialize prompt manager
        prompt_manager = PromptManager(
            storage_dir=prompt_storage_dir,
            logger=logger
        )
        
        # Create a template for asking questions
        template_id = prompt_manager.create_template(
            name="Question Template",
            description="Template for asking questions about specific topics",
            template="Answer this question about {{topic}}: {{question}}",
            default_values={"topic": "general knowledge"}
        )
        
        # Create AI with prompt manager
        ai_with_templates = AI(
            model=Model.CLAUDE_3_7_SONNET,
            config_manager=config_manager,
            logger=logger,
            prompt_manager=prompt_manager
        )
        
        # Register a tool for looking up information
        def lookup_info(topic: str) -> str:
            """Look up information about a topic."""
            info = {
                "france": "France is a country in Western Europe known for Paris, cuisine, and art.",
                "python": "Python is a high-level programming language known for readability.",
                "climate": "Climate refers to weather conditions over an extended period."
            }
            return info.get(topic.lower(), f"No information available about {topic}")
        
        ai_with_templates.register_tool(
            tool_name="lookup_info",
            tool_function=lookup_info,
            description="Look up information about a specific topic"
        )
        
        # Use the template to ask a question
        response = ai_with_templates.request_with_template(
            template_id=template_id,
            variables={
                "topic": "geography",
                "question": "What is the capital of France?"
            },
            user_id="example-user-1"
        )
        print(f"Response using template: {response}")
        
        # Create a second version of the template with different wording
        version_id = prompt_manager.create_version(
            template_id=template_id,
            template_string="I need information about {{topic}}. Please answer: {{question}}",
            name="Alternative Wording",
            description="Different wording to test effectiveness",
            set_active=True
        )
        
        # Use the updated template
        response = ai_with_templates.request_with_template(
            template_id=template_id,
            variables={
                "topic": "programming",
                "question": "What are the benefits of Python?"
            },
            user_id="example-user-2"
        )
        print(f"Response using updated template: {response}")
        
        # Get template metrics
        metrics = prompt_manager.get_template_metrics(template_id)
        print(f"\nTemplate metrics: Used {metrics['usage_count']} times")
        
        if metrics["metrics"]:
            print("Performance metrics:")
            for metric_name, values in metrics["metrics"].items():
                print(f" - {metric_name}: avg={values['avg']:.2f}")
    
    except Exception as e:
        print(f"Prompt management example failed: {e}")


if __name__ == "__main__":
    main() 