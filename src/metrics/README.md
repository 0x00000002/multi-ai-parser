# Metrics System

This module provides tools for tracking, analyzing, and visualizing metrics related to agents, tools, models, and requests in the Agentic-AI framework.

## Components

- **request_metrics.py**: Core service for tracking and storing metrics.
- **dashboard.py**: Visualization and reporting tools.
- **metrics_cli.py**: Command-line interface for metrics data.
- **decorators.py**: Decorators for easy integration with existing code.

## Quick Start

### Command-Line Usage

```bash
# View summary of all metrics
python src/metrics/metrics_cli.py summary

# View agent metrics
python src/metrics/metrics_cli.py agents

# View tool metrics
python src/metrics/metrics_cli.py tools

# View specific request details
python src/metrics/metrics_cli.py request REQUEST_ID

# Generate a performance report
python src/metrics/metrics_cli.py report
```

### Programmatic Usage

```python
from src.metrics.request_metrics import RequestMetricsService

# Initialize metrics service
metrics = RequestMetricsService()

# Start tracking a request
request_id = metrics.start_request_tracking(prompt="User query")

# Track an agent
metrics.track_agent_usage(
    request_id=request_id,
    agent_id="my_agent",
    success=True
)

# Track a tool
metrics.track_tool_usage(
    request_id=request_id,
    tool_id="my_tool",
    success=True
)

# End request tracking
metrics.end_request_tracking(
    request_id=request_id,
    success=True
)
```

### Visualization

```python
from src.metrics.dashboard import MetricsDashboard

# Create dashboard
dashboard = MetricsDashboard()

# Generate plots
dashboard.plot_agent_usage()
dashboard.plot_tool_usage()

# Generate report
report = dashboard.generate_performance_report()
```

## Documentation

For complete documentation, see `docs/metrics_system.md`.
