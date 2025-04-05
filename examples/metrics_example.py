#!/usr/bin/env python
"""
Metrics Example Script

This script demonstrates the basic usage of the metrics tracking system.
"""
import os
import sys
import time
import uuid
import random
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import metrics components
from src.metrics.request_metrics import RequestMetricsService
from src.metrics.dashboard import MetricsDashboard


def simulate_agent_activity(metrics, request_id, agent_id, success_rate=0.9):
    """Simulate activity for an agent."""
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))
    
    # Determine if successful based on success rate
    success = random.random() < success_rate
    
    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Track agent usage
    metrics.track_agent_usage(
        request_id=request_id,
        agent_id=agent_id,
        confidence=random.uniform(0.7, 1.0),
        duration_ms=duration_ms,
        success=success,
        metadata={"simulated": True}
    )
    
    return success


def simulate_tool_usage(metrics, request_id, tool_id, success_rate=0.85):
    """Simulate activity for a tool."""
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(random.uniform(0.05, 0.2))
    
    # Determine if successful based on success rate
    success = random.random() < success_rate
    
    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Track tool usage
    metrics.track_tool_usage(
        request_id=request_id,
        tool_id=tool_id,
        duration_ms=duration_ms,
        success=success,
        metadata={"simulated": True}
    )
    
    return success


def simulate_model_usage(metrics, request_id, model_id, tokens_range=(50, 500)):
    """Simulate activity for a model."""
    start_time = time.time()
    
    # Simulate processing time
    time.sleep(random.uniform(0.3, 1.0))
    
    # Simulate token counts
    tokens_in = random.randint(*tokens_range)
    tokens_out = random.randint(20, int(tokens_in / 2))
    
    # Calculate duration
    duration_ms = int((time.time() - start_time) * 1000)
    
    # Track model usage
    metrics.track_model_usage(
        request_id=request_id,
        model_id=model_id,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        duration_ms=duration_ms,
        success=True,
        metadata={"simulated": True}
    )


def simulate_request(metrics, prompt, agents, tools, model):
    """Simulate a complete request with agents and tools."""
    # Start request tracking
    request_id = metrics.start_request_tracking(
        prompt=prompt,
        metadata={"simulated": True}
    )
    
    print(f"Processing request: {request_id}")
    print(f"Prompt: {prompt}")
    
    # Simulate model usage
    simulate_model_usage(metrics, request_id, model)
    
    # Simulate agent usage
    all_successful = True
    for agent_id in agents:
        success = simulate_agent_activity(metrics, request_id, agent_id)
        if not success:
            all_successful = False
            print(f"Agent {agent_id} failed")
    
    # Simulate tool usage
    for tool_id in tools:
        success = simulate_tool_usage(metrics, request_id, tool_id)
        if not success:
            all_successful = False
            print(f"Tool {tool_id} failed")
    
    # End request tracking
    metrics.end_request_tracking(
        request_id=request_id,
        success=all_successful
    )
    
    return request_id, all_successful


def main():
    """Run the metrics example."""
    # Initialize metrics service
    metrics = RequestMetricsService()
    
    # Define some example agents, tools, and models
    agents = ["orchestrator", "coding_assistant", "request_analyzer", "response_aggregator"]
    tools = ["code_search", "file_reader", "code_writer", "terminal", "web_search"]
    models = ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
    
    # Example prompts
    prompts = [
        "How do I create a React component?",
        "Explain the metrics tracking system",
        "Write a Python function to calculate Fibonacci numbers",
        "Find all files in the project containing 'metrics'",
        "Generate documentation for the RequestMetricsService class"
    ]
    
    # Simulate multiple requests
    completed_requests = []
    for i in range(10):
        # Select random components for this request
        prompt = random.choice(prompts)
        request_agents = random.sample(agents, random.randint(1, len(agents)))
        request_tools = random.sample(tools, random.randint(1, 3))
        model = random.choice(models)
        
        # Simulate the request
        request_id, success = simulate_request(
            metrics=metrics,
            prompt=prompt,
            agents=request_agents,
            tools=request_tools,
            model=model
        )
        
        # Store successful request IDs
        if success:
            completed_requests.append(request_id)
        
        # Small delay between requests
        time.sleep(0.5)
    
    print("\nSimulation complete!")
    print(f"Simulated {len(completed_requests)} successful requests")
    
    # Display metrics for a specific request
    if completed_requests:
        sample_request = random.choice(completed_requests)
        print(f"\nSample request details ({sample_request}):")
        request_data = metrics._metrics_data["requests"].get(sample_request, {})
        print(f"Duration: {request_data.get('duration_ms', 0)} ms")
        print(f"Agents used: {request_data.get('agents_used', [])}")
        print(f"Tools used: {request_data.get('tools_used', [])}")
        print(f"Models used: {request_data.get('models_used', [])}")
    
    # Initialize dashboard
    dashboard = MetricsDashboard(metrics)
    
    # Generate a performance report
    print("\nGenerating performance report...")
    report = dashboard.generate_performance_report()
    
    # Display top agents and tools
    if report.get("top_performing_agents"):
        print("\nTop performing agents:")
        for agent in report["top_performing_agents"]:
            print(f"  {agent['agent_id']}: {agent['success_rate']:.2f} success rate")
    
    if report.get("top_performing_tools"):
        print("\nTop performing tools:")
        for tool in report["top_performing_tools"]:
            print(f"  {tool['tool_id']}: {tool['success_rate']:.2f} success rate")
    
    print("\nExample completed. You can now run the metrics CLI to explore the data:")
    print("  python src/metrics/metrics_cli.py summary")
    print("  python src/metrics/metrics_cli.py agents")
    print("  python src/metrics/metrics_cli.py tools")
    print(f"  python src/metrics/metrics_cli.py request {sample_request if completed_requests else 'REQUEST_ID'}")


if __name__ == "__main__":
    main() 