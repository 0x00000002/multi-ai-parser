# Architecture

## Core Components

The Agentic-AI framework is built on several key components:

### AI Core

- **AIBase**: Base class implementing common functionality for all AI instances
- **AI**: Extension of AIBase with tool-calling capabilities
- **AsyncAI**: Asynchronous implementation for non-blocking operations

### Providers

- **BaseProvider**: Abstract base class for all model providers
- **Provider Implementations**: Specific implementations for OpenAI, Anthropic, Google, and Ollama
- **ProviderFactory**: Creates appropriate provider instances based on model ID

### Tool Management

- **ToolManager**: Central service for registering and executing tools
- **AIToolFinder**: Uses AI to select relevant tools based on user input
- **ToolRegistry**: Stores tool definitions and metadata
- **ToolExecutor**: Executes tools safely with proper error handling

### Conversation Management

- **ConversationManager**: Maintains conversation history and context
- **ResponseParser**: Processes AI responses to extract content and thoughts

### Prompt Management

- **PromptManager**: Manages prompt templates and versions
- **PromptTemplate**: Defines a parameterized prompt structure
- **PromptVersion**: Represents a specific version of a template

### Configuration

- **ConfigManager**: Loads and provides access to configuration settings
- **Model Definitions**: Enumerations of supported models

## Component Relationships

```
+-------------+     +-------------------+
| ConfigManager|<---| Provider Factory  |
+-------------+     +-------------------+
       |                   |
       |                   v
       |            +------------+
       |            | Providers  |
       |            +------------+
       |                   ^
       v                   |
+-------------+     +------------+     +----------------+
|  AIBase     |<----| AI         |---->| ToolManager   |
+-------------+     +------------+     +----------------+
                         |                    |
                         v                    v
                    +----------------+  +----------------+
                    | ConvManager   |  | AIToolFinder  |
                    +----------------+  +----------------+
                         |                    |
                         v                    v
                    +----------------+  +----------------+
                    | ResponseParser|  | ToolRegistry  |
                    +----------------+  +----------------+
                                           |
                                           v
                                     +----------------+
                                     | ToolExecutor  |
                                     +----------------+
```
