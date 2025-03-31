# Example usage of enhanced AI with ToolFinder integration

from src.config.models import Model
from src.config.config_manager import ConfigManager
from src.core.model_selector import UseCase, ModelSelector
try:
    from src.config.models import Quality, Speed
except ImportError:
    from enum import Enum
    class Quality(Enum):
        HIGH = "high"
    class Speed(Enum):
        BALANCED = "balanced"
from src.utils.logger import LoggerFactory, LoggingLevel, LogFormat
import logging

# Import the enhanced AI class that inherits from AIBase
from src.core.tool_enabled_ai import AI

# Enable real logging
LoggerFactory.enable_real_loggers()

# Create a logger
logging.getLogger().handlers = []    
logger = LoggerFactory.create("logger", LoggingLevel.INFO, LogFormat.SIMPLE)
debugger = LoggerFactory.create("debugger", LoggingLevel.DEBUG, LogFormat.SIMPLE)

# Initialize ConfigManager (assuming default config path)
config_manager = ConfigManager(config_path="src/config/config.yml")

def get_ticket_price(destination: str = "New York") -> str:
    """Returns a fixed ticket price of $1000 USD for any destination."""
    return f"A ticket to {destination} costs $1000 USD"

# Example 1: Create AI with auto tool finding
def example_auto_tool_finding():
    debugger.debug("Starting example_auto_tool_finding")
    
    # Create main AI using the enhanced AI class
    debugger.debug("Creating AI instance with Gemini 2.5 Pro model")
    main_ai = AI(
        model=Model.GEMINI_2_5_PRO,
        system_prompt="You are a helpful assistant. Your goal is to help users find information about ticket prices. ALWAYS use the provided tools when available to answer questions about prices.",
        logger=debugger,
        config_manager=config_manager,
        auto_find_tools=True  # Enable auto tool finding through constructor
    )
    
    # Force tool support (fix for the issue)
    main_ai._supports_tools = True
    debugger.debug(f"Forcing tool support: {main_ai._supports_tools}")
    
    # Register the ticket price tool
    main_ai.register_tool(
        tool_name="get_ticket_price",
        tool_function=get_ticket_price,
        description="Returns the price of a ticket to a specified destination",
        parameters_schema={
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string",
                    "description": "The destination city"
                }
            }
        }
    )
    
    # Debug tool registration
    debugger.debug(f"Registered tools: {main_ai.get_tools()}")
    
    # Test with a user query about ticket prices
    debugger.debug("Sending query about ticket prices")
    response = main_ai.request("How much is a ticket to New York?")
    debugger.debug(f"Received response: {response}")
    print(response)
    
    # Try another query that likely doesn't need a tool
    response = main_ai.request("What's the weather like today there?")
    print(response)
    
    return main_ai


if __name__ == "__main__":
    # Choose which example to run
    debugger.debug("Starting main program execution")
    print("\n--- Running Example 1: Auto Tool Finding ---")
    ai = example_auto_tool_finding()
    debugger.debug("Example completed successfully")
