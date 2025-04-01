# Prompt Management

## Overview

The prompt management system in Agentic-AI allows you to:

- Create reusable prompt templates with variables
- Version prompts to test different variations
- Track performance metrics for prompts
- A/B test different prompt versions

## Creating Templates

Prompt templates are parameterized prompts with placeholders for variables:

```python
from src.prompts import PromptManager

# Initialize prompt manager
prompt_manager = PromptManager(storage_dir="data/prompts")

# Create a template
template_id = prompt_manager.create_template(
    name="Question Template",
    description="Template for asking questions about topics",
    template="Answer this question about {{topic}}: {{question}}",
    default_values={"topic": "general knowledge"}
)

# Use the template with the AI
ai = AI(
    model=Model.CLAUDE_3_7_SONNET,
    config_manager=config_manager,
    prompt_manager=prompt_manager
)

response = ai.request_with_template(
    template_id=template_id,
    variables={
        "topic": "history",
        "question": "When was the Declaration of Independence signed?"
    }
)
```

## Versioning Prompts

Create and test multiple versions of a prompt template:

```python
# Create an alternative version
version_id = prompt_manager.create_version(
    template_id=template_id,
    template_string="I need information about {{topic}}. Please answer: {{question}}",
    name="Alternative Wording",
    description="Different wording to test effectiveness"
)

# Set a version as active
prompt_manager.set_active_version(template_id, version_id)

# Or create and set as active in one step
version_id = prompt_manager.create_version(
    template_id=template_id,
    template_string="New template text with {{variables}}",
    set_active=True
)
```

## Tracking Metrics

The prompt management system automatically tracks usage and performance metrics:

```python
# Get metrics for a template
metrics = prompt_manager.get_template_metrics(template_id)

print(f"Template used {metrics['usage_count']} times")
for metric_name, values in metrics["metrics"].items():
    print(f"{metric_name}: avg={values['avg']}, min={values['min']}, max={values['max']}")

# You can also record custom metrics
prompt_manager.record_prompt_performance(
    usage_id="some-usage-id",
    metrics={
        "accuracy": 0.95,
        "relevance": 0.87
    }
)
```

## A/B Testing

Perform A/B testing by providing a user ID when using templates:

```python
# Different users will get different versions based on consistent hashing
response1 = ai.request_with_template(
    template_id=template_id,
    variables={"key": "value"},
    user_id="user-123"
)

response2 = ai.request_with_template(
    template_id=template_id,
    variables={"key": "value"},
    user_id="user-456"
)

# View metrics by version
metrics_by_version = prompt_manager.get_version_metrics(template_id)
for version_id, metrics in metrics_by_version.items():
    print(f"Version {version_id}: {metrics}")
```
