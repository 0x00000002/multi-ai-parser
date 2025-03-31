# AI Response Thoughts Example

This example demonstrates how to handle AI responses that contain "thought" sections, which are useful for understanding the AI's reasoning process.

## Overview

The example in `with_thoughts_example.py` showcases:

1. How to configure whether AI thoughts are included in responses using the `show_thinking` parameter
2. How thoughts are automatically extracted and stored separately from the main content
3. How to inspect both the thoughts and content from conversation history

## How It Works

The AI framework handles "thoughts" in responses by:

1. **Extraction:** The response parser looks for content enclosed in `<think>...</think>` tags
2. **Storage:** Thoughts are extracted and stored separately from the main content
3. **Presentation:** Based on the `show_thinking` parameter, thoughts can be included or excluded from the final response

## Implementation Details

The framework includes several components that work together to handle thoughts properly:

1. **ConfigManager**: Controls whether thoughts are shown via the `show_thinking` parameter
2. **ResponseParser**: Extracts thoughts from responses and handles formatting based on the `show_thinking` setting
3. **ConversationManager**: Stores extracted thoughts separately from main content
4. **Provider Standardization**: All providers return responses in a consistent format

### Key Components

- **StandardizeResponse Method**: Ensures consistent response formats across all providers
- **Parse Response Method**: Robustly extracts and handles thought blocks
- **Exception Handling**: Gracefully handles cases where thought extraction might fail

## Configuration

To control whether thoughts are shown in responses, use the `show_thinking` parameter when creating a ConfigManager:

```python
# Hide thoughts in responses
config = ConfigManager(show_thinking=False)  # Default behavior

# Show thoughts in responses
config = ConfigManager(show_thinking=True)
```

## Best Practices

- **Development/Debugging:** Enable thoughts (`show_thinking=True`) to understand the AI's reasoning process
- **Production:** Disable thoughts (`show_thinking=False`) for cleaner, more concise responses
- **Analysis:** Use `get_conversation()` to access both content and thoughts separately for further processing

## How to Run

```bash
python examples/with_thoughts_example.py
```

## Expected Output

The example demonstrates two different configurations:

1. With `show_thinking=False` - thoughts are hidden from responses
2. With `show_thinking=True` - thoughts are included in responses

For each configuration, the example prints the response and shows how to access the thoughts from the conversation history.
