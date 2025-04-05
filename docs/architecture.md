# Agentic-AI Architecture

This document describes the architecture of the Agentic-AI framework, focusing on the core design principles and component relationships.

## Overview

Agentic-AI is a modular framework for building AI applications with tool integration capabilities. The architecture follows solid software engineering principles including:

- **Separation of Concerns**: Each component has a clear, focused responsibility
- **Dependency Injection**: Components receive their dependencies rather than creating them
- **Interface-Based Design**: Components interact through well-defined interfaces
- **Error Handling**: Standardized error handling across the framework
- **Configuration Management**: Modular, externalized configuration

## Core Components

### AI Core

The AI core provides the foundation for interacting with language models:

- **AIBase**: Base implementation of the AI interface
- **AI**: Extended implementation with tool integration capabilities
- **Providers**: Abstraction layer for different AI providers (OpenAI, Anthropic, etc.)
- **Interfaces**: Clear contracts for component interactions

### Multi-Agent System

The multi-agent system enables specialized processing of user requests:

- **Orchestrator**: Coordinates the workflow between specialized agents
- **RequestAnalyzer**: Analyzes requests to determine appropriate agents and tools
- **ResponseAggregator**: Combines responses from multiple agents
- **BaseAgent**: Common functionality for all agents
- **ToolFinderAgent**: Identifies relevant tools for user requests

### Configuration Management

Configuration is modularized for better maintainability:

- **ConfigFactory**: Central point for accessing configuration
- **Modular Config Files**: Separate files for models, providers, agents, and use cases

### Error Handling

The error handling system provides consistent error management:

- **ErrorHandler**: Centralized error handling with standardized responses
- **Exception Hierarchy**: Well-defined exception types for different error scenarios

## Dependency Management

The `AppContainer` class centralizes component creation and wiring:

- **Singleton Management**: Controls lifecycle of singleton components
- **Factory Methods**: Creates properly configured component instances
- **Dependency Resolution**: Automatically resolves dependencies between components

## Component Relationships

```
┌──────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                  │     │                   │     │                   │
│  ConfigFactory   │◄────┤   AppContainer    │────►│   ErrorHandler    │
│                  │     │                   │     │                   │
└───────┬──────────┘     └───────┬───────────┘     └───────────────────┘
        │                        │
        │                        │
┌───────▼──────────┐     ┌───────▼───────────┐
│                  │     │                   │
│  ProviderFactory │     │   AIBase / AI     │
│                  │     │                   │
└───────┬──────────┘     └───────┬───────────┘
        │                        │
        │                        │
┌───────▼──────────┐     ┌───────▼───────────┐     ┌───────────────────┐
│                  │     │                   │     │                   │
│    Providers     │     │   Orchestrator    │────►│  RequestAnalyzer  │
│                  │     │                   │     │                   │
└──────────────────┘     └───────┬───────────┘     └───────────────────┘
                                 │
                                 │
                         ┌───────▼───────────┐     ┌───────────────────┐
                         │                   │     │                   │
                         │  Specialized      │────►│ ResponseAggregator│
                         │  Agents           │     │                   │
                         └───────────────────┘     └───────────────────┘
```

## Key Architectural Improvements

### 1. Refactored Orchestrator

The Orchestrator has been split into three focused components:

- **Orchestrator**: Coordinates the overall workflow
- **RequestAnalyzer**: Specifically handles request analysis
- **ResponseAggregator**: Handles response aggregation

This separation improves maintainability by reducing class complexity and following the Single Responsibility Principle.

### 2. Modularized Configuration

Configuration has been separated into domain-specific files:

- **models.yml**: Model definitions and parameters
- **providers.yml**: Provider settings
- **agents.yml**: Agent configurations
- **use_cases.yml**: Use case specific settings

This makes configuration more manageable and easier to maintain.

### 3. Standardized Error Handling

A consistent error handling approach has been implemented:

- **AIFrameworkError**: Base exception type for all framework errors
- **ErrorHandler**: Centralized error processing and formatting
- **Error Response Format**: Standardized error response structure

This improves error visibility and provides better context for debugging.

### 4. Dependency Injection Container

The new `AppContainer` class centralizes component creation and wiring:

- **Explicit Dependencies**: Clear visibility of component dependencies
- **Lifecycle Management**: Controls singleton vs. factory instances
- **Simplified Testing**: Easier to mock components for testing

### 5. Enhanced Documentation

Improved documentation throughout the codebase:

- **Detailed Comments**: Clear explanations of component purpose and behavior
- **Architecture Documentation**: High-level explanation of system design
- **Type Annotations**: Better type safety and IDE support

## Usage Example

```python
from src.core.dependency_container import AppContainer

# Get a fully configured AI instance
ai = AppContainer.create_ai(model="claude-3-haiku")

# Register a tool
ai.register_tool(
    tool_name="get_weather",
    tool_function=get_weather,
    description="Get current weather for a location"
)

# Make a request
response = ai.request("What's the weather like in Paris today?")
print(response)

# Get a fully configured orchestrator for multi-agent processing
orchestrator = AppContainer.create_orchestrator()

# Process a request with the orchestrator
response = orchestrator.process_request({
    "prompt": "Translate this text to French and analyze the sentiment",
    "context": {"source_language": "en"}
})
print(response["content"])
```
