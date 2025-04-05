# Configuration Usage Examples

This document provides examples of how to use the new configuration system in various scenarios.

## Basic Usage

```python
# Import the configuration API
from src.config import configure
from src.core.tool_enabled_ai import AI

# Configure with default settings
configure()

# Create an AI instance using the default configuration
ai = AI()
response = ai.request("Hello, who are you?")
print(response)
```

## Selecting a Model

```python
from src.config import configure
from src.core.tool_enabled_ai import AI

# Configure with a specific model
configure(model="claude-3-5-sonnet")

# Create an AI instance with the configured model
ai = AI()
response = ai.request("Explain the concept of recursion")
print(response)
```

## Using a Specific Use Case

```python
from src.config import configure, UseCasePreset
from src.core.tool_enabled_ai import AI

# Configure for Solidity smart contract development
configure(use_case=UseCasePreset.SOLIDITY_CODING)

# Create an AI instance optimized for Solidity coding
ai = AI()
response = ai.request("Write a simple ERC20 token contract")
print(response)
```

## Custom Configuration with Multiple Options

```python
from src.config import configure
from src.core.tool_enabled_ai import AI

# Configure with multiple options
configure(
    model="claude-3-5-sonnet",
    temperature=0.8,
    system_prompt="You are a helpful assistant specialized in Solidity smart contract development.",
    show_thinking=True,
    max_tokens=2000
)

# Create an AI instance with the custom configuration
ai = AI()
response = ai.request("What is the best way to implement a token vesting mechanism?")
print(response)
```

## Loading Configuration from a File

Create a file named `solidity_config.yml`:

```yaml
model: claude-3-5-sonnet
use_case: solidity_coding
temperature: 0.8
show_thinking: true
system_prompt: You are a helpful assistant specialized in Solidity smart contract development.
```

Then use it in your code:

```python
from src.config import configure
from src.core.tool_enabled_ai import AI

# Load configuration from a file
configure(config_file="solidity_config.yml")

# Create an AI instance using the loaded configuration
ai = AI()
response = ai.request("Explain gas optimization techniques")
print(response)
```

## Accessing the Configuration API Directly

```python
from src.config import get_config, configure, get_available_models
from src.core.tool_enabled_ai import AI

# Configure the system
configure(model="claude-3-5-sonnet")

# Get the configuration instance
config = get_config()

# Get all available models
available_models = get_available_models()
print("Available models:", available_models)

# Get the current default model
default_model = config.get_default_model()
print("Default model:", default_model)

# Get the model configuration
model_config = config.get_model_config(default_model)
print("Model name:", model_config["name"])
print("Provider:", model_config["provider"])
print("Quality:", model_config["quality"])
print("Speed:", model_config["speed"])

# Create an AI instance
ai = AI()
response = ai.request("Hello!")
print(response)
```

## Using with the Orchestrator

```python
from src.config import configure, UseCasePreset
from src.agents.orchestrator import Orchestrator

# Configure for data analysis use case
configure(
    model="claude-3-5-sonnet",
    use_case=UseCasePreset.DATA_ANALYSIS,
    show_thinking=True
)

# Create an orchestrator (it will use the configured settings)
orchestrator = Orchestrator()

# Process a request
response = orchestrator.process_request({
    "prompt": "Analyze the trends in this data: [1, 3, 6, 10, 15, 21, 28]",
    "conversation_history": []
})

print(response["content"])
```
