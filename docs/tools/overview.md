# Tool Integration

## Overview

Tools allow the AI to perform actions and retrieve information beyond its training data. The Agentic-AI framework includes a sophisticated tool management system that enables:

1. Registering Python functions as tools
2. Automatic discovery of relevant tools for user requests
3. Systematic execution of tool calls with proper error handling

## Tool Components

- **ToolManager**: Central service that coordinates all tool operations
- **AIToolFinder**: AI-powered component that selects relevant tools based on user input
- **ToolRegistry**: Stores tool definitions and makes them available to the AI
- **ToolExecutor**: Executes tool calls safely with error handling
- **ToolPromptBuilder**: Constructs appropriate prompts for tool usage

## Creating and Registering Tools

Tools are Python functions that can be registered with the AI. Each tool needs:

- A unique name
- The function implementing the tool's logic
- A description of what the tool does
- A schema defining the parameters the tool accepts

```python
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Implementation
    return f"Weather data for {location}"

ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get current weather for a location",
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
```

## Automatic Tool Finding

The AIToolFinder component uses an AI model to determine which tools are relevant for a given user query:

```python
# Enable auto tool finding when creating the AI
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    config_manager=config_manager,
    logger=logger,
    auto_find_tools=True
)

# Register multiple tools
ai.register_tool("get_weather", get_weather, "Get weather information")
ai.register_tool("calculate", calculator, "Perform calculations")
ai.register_tool("search_web", search_web, "Search the web for information")

# The AI will automatically select the appropriate tool
response = ai.request("What's the weather like in Paris?")
```

## Tool Selection Sequence

```
User Request
     |
     v
AI (Tool-Enabled)
     |
     v
Tool Manager ---> Available Tools
     |              |
     v              |
AIToolFinder <------+
     |
     v
Selected Tools
     |
     v
Tool Executor
     |
     v
Tool Results
     |
     v
Final AI Response
```

## Creating Custom Tool Strategies

You can create custom tool selection strategies by implementing the `ToolStrategy` interface:

```python
from src.tools.interfaces import ToolStrategy

class CustomToolStrategy(ToolStrategy):
    def select_tools(self, user_prompt, available_tools, context=None):
        # Your custom logic to select tools
        selected_tools = []
        # ...
        return selected_tools

# Use your custom strategy
tool_manager.set_tool_strategy(CustomToolStrategy())
```
