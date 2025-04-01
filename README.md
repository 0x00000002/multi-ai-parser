# Agentic-AI

A modular framework for building AI applications with tool integration capabilities.

## HTML documentation

[Documentation HTML website](./site/index.html)

## Overview

Agentic-AI is a Python library designed to create AI-powered applications that can:

- Use multiple AI model providers (OpenAI, Anthropic, Google, Ollama)
- Dynamically discover and call tools based on user input
- Manage conversations and maintain context
- Template and version prompts with metrics tracking

## Key Features

- **Multiple Provider Support**: Use models from OpenAI, Anthropic, Google, and Ollama seamlessly
- **Tool Integration**: Register Python functions as tools the AI can use
- **Automatic Tool Discovery**: AI-powered selection of relevant tools based on user queries
- **Prompt Management**: Create, version, and track performance of prompt templates
- **Conversation Management**: Maintain context across multiple interactions

## Installation

```bash
# With pip
pip install -r requirements.txt

# For development installation
pip install -e .
```

## Quick Example

```python
from src.config.models import Model
from src.core.tool_enabled_ai import AI

# Create an AI instance
ai = AI(model=Model.CLAUDE_3_7_SONNET)

# Define a tool function
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return f"It's sunny in {location} today!"

# Register the tool
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get current weather for a location"
)

# Use the AI with tools
response = ai.request("What's the weather like in Paris today?")
print(response)
```

## Documentation

Comprehensive documentation is available in the [docs](docs/) directory.

To build and view the documentation:

```bash
# Install MkDocs and the Material theme
pip install mkdocs mkdocs-material

# Serve the documentation locally
mkdocs serve
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/agentic-ai.git
cd agentic-ai

# Create and activate conda environment
conda env create -f environment.yml
conda activate agentic-ai

# Install in development mode
pip install -e .
```

### Testing

```bash
# Run all tests
python -m unittest discover -s tests

# Run specific module tests
python -m unittest discover -s tests/tools

# Run tools test suite
python tests/tools/run_tools_tests.py
```

## License

[MIT License](LICENSE)
