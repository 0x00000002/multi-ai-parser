# Agentic-AI

A modular framework for building AI applications with tool integration capabilities.

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
# With conda
conda env create -f environment.yml

# Or with pip
pip install -r requirements.txt

# For development installation
pip install -e .
```

## Quick Start

Check the [Getting Started](getting-started.md) guide to begin using Agentic-AI.
