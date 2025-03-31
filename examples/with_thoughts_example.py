"""
Example demonstrating the use of thoughts in AI responses.

This example shows:
1. How to configure whether AI thoughts are included in responses using the show_thinking parameter
2. How thoughts are automatically extracted and stored separately from the main content
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path so we can import from there
src_path = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, src_path)

from src.config.config_manager import ConfigManager
from src.config.models import Model
from src.core.tool_enabled_ai import AI
from src.utils.logger import LoggerFactory


def analyze_code(code: str) -> Dict[str, Any]:
    """Analyze code for potential issues."""
    return {
        "lines": len(code.splitlines()),
        "has_comments": "#" in code,
        "has_docstring": '"""' in code or "'''" in code
    }


def main():
    """Run examples demonstrating thoughts in AI responses."""
    # Ensure logging is disabled by default
    LoggerFactory.disable_real_loggers()
    
    # Example 1: Create a config manager with thoughts disabled (default behavior)
    print("\n=== Example 1: AI Response with Thoughts Hidden ===")
    config_path = os.path.join(src_path, "src", "config", "config.yml")
    config_without_thoughts = ConfigManager(
        config_path=config_path,
        show_thinking=False,  # Disable thoughts in responses (default)
        verbose=False  # Disable verbose output
    )
    
    # Create AI instance with thoughts disabled
    ai_without_thoughts = AI(
        model=Model.DEEPSEEK_7B,
        config_manager=config_without_thoughts
        # No logger needed - framework will use NullLogger by default
    )
    
    # Request with thoughts disabled
    response = ai_without_thoughts.request("What is the capital of France?")
    
    print(f"\nResponse (thoughts hidden):")
    print(f"----------------------------")
    print(response)
    print(f"----------------------------")
    
    # Example 2: Create a config manager with thoughts enabled
    print("\n=== Example 2: AI Response with Thoughts Visible ===")
    config_with_thoughts = ConfigManager(
        config_path=config_path,
        show_thinking=True,  # Enable thoughts in responses
        verbose=False  # Disable verbose output
    )
    
    # Create AI instance with thoughts enabled
    ai_with_thoughts = AI(
        model=Model.DEEPSEEK_7B,
        config_manager=config_with_thoughts
        # No logger needed - framework will use NullLogger by default
    )
    
    # Request with thoughts enabled
    response = ai_with_thoughts.request("What is 2 + 2?")
    print(f"\nResponse (including thoughts):")
    print(f"----------------------------")
    print(response)
    print(f"----------------------------")
    
    # Get the conversation history to inspect thoughts
    conversation = ai_with_thoughts.get_conversation()
    print("\nConversation history (last message):")
    last_msg = conversation[-1]  # Get the last message (assistant's response)
    print(f"Role: {last_msg['role']}")
    print(f"Content: {last_msg['content']}")
    if last_msg.get('thoughts'):
        print(f"Extracted thoughts: {last_msg['thoughts']}")


if __name__ == "__main__":
    main() 