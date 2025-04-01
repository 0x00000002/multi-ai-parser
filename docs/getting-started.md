# Getting Started

## Basic Usage

Here's a simple example to get started with Agentic-AI:

```python
from src.config.models import Model
from src.config.config_manager import ConfigManager
from src.core.tool_enabled_ai import AI
from src.utils.logger import LoggerFactory

# Set up logger
logger = LoggerFactory.create()

# Initialize ConfigManager
config_manager = ConfigManager()

# Create AI instance
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,  # Choose your model
    config_manager=config_manager,
    logger=logger
)

# Send a request
response = ai.request("What is the capital of France?")
print(response)
```

## Adding Tool Capabilities

Register tools to allow the AI to perform actions:

```python
# Create tool-enabled AI
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    config_manager=config_manager,
    logger=logger
)

# Define a function to use as a tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    # In a real app, you would call a weather API here
    return f"It's sunny in {location} today!"

# Register the tool
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get the current weather for a location",
    parameters_schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city or location"
            }
        },
        "required": ["location"]
    }
)

# The AI will now use the tool when appropriate
response = ai.request("What's the weather like in Tokyo today?")
print(response)
```

## Automatic Tool Finding

Enable the AI to automatically select relevant tools:

```python
# Create AI with auto tool finding
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    config_manager=config_manager,
    logger=logger,
    auto_find_tools=True  # Enable auto tool finding
)

# Register multiple tools
ai.register_tool("get_weather", get_weather, "Get weather for a location")
ai.register_tool("calculate", calculator_function, "Perform calculations")

# The AI will automatically select the appropriate tool
response = ai.request("What's the weather like in Paris today?")
```

See the [Tool Integration](tools/overview.md) section for more details.
