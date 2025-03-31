# Simple test for debugging tool functionality

from src.config.models import Model
from src.config.config_manager import ConfigManager
from src.utils.logger import LoggerFactory, LoggingLevel, LogFormat
import logging
import json

# Import the enhanced AI class that inherits from AIBase
from src.core.tool_enabled_ai import AI
from src.tools.models import ToolCall

# Enable real logging
LoggerFactory.enable_real_loggers()

# Create a logger with DEBUG level
logging.getLogger().handlers = []    
logger = LoggerFactory.create("logger", LoggingLevel.DEBUG, LogFormat.SIMPLE)

# Initialize ConfigManager (assuming default config path)
config_manager = ConfigManager(config_path="src/config/config.yml")

def get_ticket_price(destination: str = "New York") -> str:
    """Returns a fixed ticket price of $1000 USD for any destination."""
    logger.debug(f"get_ticket_price called with destination: {destination}")
    return f"A ticket to {destination} costs $1000 USD"

def main():
    logger.debug("Starting test_tool main function")
    
    # Create AI instance with Gemini model
    ai = AI(
        model=Model.GEMINI_2_5_PRO,
        system_prompt="You are an assistant that helps with ticket pricing information. Always use tools when available.",
        logger=logger,
        config_manager=config_manager
    )
    
    # Force tool support
    ai._supports_tools = True
    logger.debug(f"Forced tool support: {ai._supports_tools}")
    
    # Register ticket price tool
    ai.register_tool(
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
    
    logger.debug(f"Registered tools: {ai.get_tools()}")
    
    # Test direct tool calling
    try:
        # Create a tool call manually
        tool_call = ToolCall(
            name="get_ticket_price",
            arguments={"destination": "Paris"}
        )
        
        # Call the tool directly
        logger.debug("Calling tool directly")
        result = ai._call_tool(tool_call)
        logger.debug(f"Direct tool call result: {result}")
    except Exception as e:
        logger.error(f"Direct tool call failed: {str(e)}")
    
    # Test tool via AI request
    try:
        # Make request to AI
        logger.debug("Making AI request with tool")
        response = ai.request("How much is a ticket to London?")
        logger.debug(f"AI response: {response}")
        
        # Check if tool was used
        if "costs $1000 USD" in response:
            logger.debug("Tool was successfully used in the response")
        else:
            logger.warning("Tool was not used in the response")
            
            # Debug response pattern
            logger.debug("Checking for JSON pattern in response")
            if "{" in response and "}" in response:
                # Parse JSON manually
                try:
                    import re
                    json_matches = re.findall(r'({[^{}]*})', response)
                    for json_str in json_matches:
                        logger.debug(f"Found JSON: {json_str}")
                        data = json.loads(json_str)
                        if "tool" in data and "parameters" in data:
                            tool_name = data["tool"]
                            params = data["parameters"]
                            logger.debug(f"Detected tool call: {tool_name} with params {params}")
                            
                            # Execute the tool manually
                            if tool_name == "get_ticket_price":
                                result = get_ticket_price(**params)
                                logger.debug(f"Manual tool execution result: {result}")
                except Exception as e:
                    logger.error(f"Error processing JSON in response: {str(e)}")
    except Exception as e:
        logger.error(f"AI request failed: {str(e)}")
    
    logger.debug("Test completed")

if __name__ == "__main__":
    main() 