# Example usage of enhanced AI with ToolFinder integration

from src.ai.ai_config import Model, Quality, Speed
from src.ai.model_selector import UseCase
from src.Logger import LoggerFactory, LoggingLevel, LogFormat
import logging
from src.ai.tools.tool_finder import ToolFinder

# Import the enhanced AI class that inherits from AIBase
from src.ai.enhanced_ai import AI

# Create a logger
logging.getLogger().handlers = []    
logger = LoggerFactory.get_logger("logger", LoggingLevel.INFO, LogFormat.SIMPLE)
debugger = LoggerFactory.get_logger("debugger", LoggingLevel.DEBUG, LogFormat.SIMPLE)

# Example 1: Create AI with auto tool finding
def example_auto_tool_finding():
    # Create main AI using the enhanced AI class
    main_ai = AI(
        model_param=Model.GEMINI_2_5_PRO,
        system_prompt="You are a helpful assistant. Don't guess the answer, if you don't know the answer, say that you don't know.",
        logger=logger
    )
    
    # Enable automatic tool finding (automatically creates a ToolFinder)
    main_ai.enable_auto_tool_finding(True)
    
    # Test with a user query about ticket prices
    response = main_ai.request("How much is a ticket to New York?")
    print(response)
    
    # # Try another query that likely doesn't need a tool
    # response = main_ai.request("What's the weather like today there?")
    # print(response)
    
    return main_ai

# Example 2: Manual tool finding integration
def example_manual_integration():
    # Create main AI without auto tool finding
    main_ai = AI(
        model_param=Model.CHATGPT_4O_MINI,
        system_prompt="You are a helpful assistant.",
        logger=logger
    )
    
    # Create a dedicated tool finder with a specific model
    tool_finder = ToolFinder(
        model=Model.CHATGPT_4O_MINI,
        logger=logger
    )
    
    # Connect the tool finder
    main_ai.set_tool_finder(tool_finder)
    
    # Manually find tools before making a request
    user_prompt = "I need to book a flight to Paris, what's the price?"
    tools = main_ai.find_tools(user_prompt)
    
    # Log the tools found
    print(f"Found tools: {[tool.name for tool in tools]}")
    
    # Decide whether to enhance the prompt based on the found tools
    if tools:
        # Enable auto tool finding for this specific request
        main_ai.enable_auto_tool_finding(True)
        response = main_ai.request(user_prompt)
        # Turn it off after
        main_ai.enable_auto_tool_finding(False)
    else:
        # No tools needed, make a regular request
        response = main_ai.request(user_prompt)
    
    print(response)
    
    return main_ai

# Example 3: Use case specific initialization
def example_use_case():
    # Create AI for a specific use case
    main_ai = AI.for_use_case(  # This works because AI inherits for_use_case from AIBase
        use_case=UseCase.GENERAL_ASSISTANT,
        quality=Quality.HIGH,
        speed=Speed.BALANCED,
        logger=logger
    )
    
    # Create tool finder and enable
    main_ai.create_tool_finder(Model.CHATGPT_4O_MINI)
    main_ai.enable_auto_tool_finding(True)
    
    # Test with ticket price query
    response = main_ai.request("What's the cost of a ticket to Berlin?")
    print(response)
    
    return main_ai

if __name__ == "__main__":
    # Choose which example to run
    ai = example_auto_tool_finding()
    # ai = example_manual_integration()
    # ai = example_use_case()