#!/usr/bin/env python
"""
Metrics CLI

Command-line interface for viewing and analyzing metrics data.
"""
import argparse
import json
from datetime import datetime, timedelta
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Use absolute imports
from src.metrics.request_metrics import RequestMetricsService
from src.metrics.dashboard import MetricsDashboard


def format_date(date_str):
    """Format a date string nicely."""
    if not date_str:
        return "N/A"
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return date_str


def main():
    """Run the metrics CLI."""
    parser = argparse.ArgumentParser(description="View and analyze metrics data")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show a summary of metrics")
    summary_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    
    # Agent command
    agent_parser = subparsers.add_parser("agents", help="Show agent metrics")
    agent_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    agent_parser.add_argument("--agent", type=str, help="Show metrics for a specific agent")
    agent_parser.add_argument("--plot", action="store_true", help="Generate a plot")
    agent_parser.add_argument("--output", type=str, help="Output file for plot")
    
    # Tool command
    tool_parser = subparsers.add_parser("tools", help="Show tool metrics")
    tool_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    tool_parser.add_argument("--tool", type=str, help="Show metrics for a specific tool")
    tool_parser.add_argument("--plot", action="store_true", help="Generate a plot")
    tool_parser.add_argument("--output", type=str, help="Output file for plot")
    
    # Request command
    request_parser = subparsers.add_parser("request", help="Show details for a specific request")
    request_parser.add_argument("request_id", type=str, help="Request ID to show details for")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a performance report")
    report_parser.add_argument("--days", type=int, default=30, help="Number of days to include")
    report_parser.add_argument("--output", type=str, help="Output file for report (JSON)")
    
    args = parser.parse_args()
    
    metrics_service = RequestMetricsService()
    dashboard = MetricsDashboard(metrics_service)
    
    if args.command == "summary":
        print(f"Summary for the last {args.days} days:")
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=args.days)
        
        # Get agent and tool metrics
        agent_metrics = metrics_service.get_agent_metrics(start_time=start_time, end_time=end_time)
        tool_metrics = metrics_service.get_tool_metrics(start_time=start_time, end_time=end_time)
        
        # Count total requests
        request_count = 0
        success_count = 0
        
        for request_id, request_data in metrics_service._metrics_data.get("requests", {}).items():
            request_start = request_data.get("start_time")
            if request_start and request_start >= start_time.isoformat() and request_start <= end_time.isoformat():
                request_count += 1
                if request_data.get("success", False):
                    success_count += 1
        
        print(f"Total requests: {request_count}")
        print(f"Successful requests: {success_count}")
        print(f"Success rate: {success_count / request_count * 100:.2f}%" if request_count else "Success rate: N/A")
        print()
        
        print(f"Top 5 agents by usage:")
        sorted_agents = sorted(agent_metrics.items(), key=lambda x: x[1]["total_requests"], reverse=True)[:5]
        for agent_id, metrics in sorted_agents:
            print(f"  {agent_id}: {metrics['total_requests']} requests, {metrics['successful_requests']} successful")
        print()
        
        print(f"Top 5 tools by usage:")
        sorted_tools = sorted(tool_metrics.items(), key=lambda x: x[1]["total_calls"], reverse=True)[:5]
        for tool_id, metrics in sorted_tools:
            print(f"  {tool_id}: {metrics['total_calls']} calls, {metrics['successful_calls']} successful")
    
    elif args.command == "agents":
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=args.days)
        agent_metrics = metrics_service.get_agent_metrics(args.agent, start_time, end_time)
        
        if args.plot:
            dashboard.plot_agent_usage(top_n=10, days=args.days, save_path=args.output)
            if not args.output:
                print("Plot displayed. Close the plot window to continue.")
        else:
            # Print agent metrics
            if args.agent:
                if args.agent not in agent_metrics:
                    print(f"No metrics found for agent: {args.agent}")
                else:
                    metrics = agent_metrics[args.agent]
                    print(f"Metrics for agent: {args.agent}")
                    print(f"Total requests: {metrics['total_requests']}")
                    print(f"Successful requests: {metrics['successful_requests']}")
                    print(f"Success rate: {metrics['successful_requests'] / metrics['total_requests'] * 100:.2f}%" if metrics['total_requests'] else "Success rate: N/A")
                    print(f"Average confidence: {metrics['avg_confidence']:.4f}")
                    print(f"Average duration: {metrics['avg_duration_ms']:.2f} ms")
                    print(f"Last used: {format_date(metrics['last_used'])}")
            else:
                print(f"Agent metrics for the last {args.days} days:")
                for agent_id, metrics in sorted(agent_metrics.items(), key=lambda x: x[1]["total_requests"], reverse=True):
                    success_rate = metrics['successful_requests'] / metrics['total_requests'] * 100 if metrics['total_requests'] else 0
                    print(f"  {agent_id}: {metrics['total_requests']} requests, {success_rate:.2f}% success rate")
    
    elif args.command == "tools":
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=args.days)
        tool_metrics = metrics_service.get_tool_metrics(args.tool, start_time, end_time)
        
        if args.plot:
            dashboard.plot_tool_usage(top_n=10, days=args.days, save_path=args.output)
            if not args.output:
                print("Plot displayed. Close the plot window to continue.")
        else:
            # Print tool metrics
            if args.tool:
                if args.tool not in tool_metrics:
                    print(f"No metrics found for tool: {args.tool}")
                else:
                    metrics = tool_metrics[args.tool]
                    print(f"Metrics for tool: {args.tool}")
                    print(f"Total calls: {metrics['total_calls']}")
                    print(f"Successful calls: {metrics['successful_calls']}")
                    print(f"Success rate: {metrics['successful_calls'] / metrics['total_calls'] * 100:.2f}%" if metrics['total_calls'] else "Success rate: N/A")
                    print(f"Average duration: {metrics['avg_duration_ms']:.2f} ms")
                    print(f"Last used: {format_date(metrics['last_used'])}")
            else:
                print(f"Tool metrics for the last {args.days} days:")
                for tool_id, metrics in sorted(tool_metrics.items(), key=lambda x: x[1]["total_calls"], reverse=True):
                    success_rate = metrics['successful_calls'] / metrics['total_calls'] * 100 if metrics['total_calls'] else 0
                    print(f"  {tool_id}: {metrics['total_calls']} calls, {success_rate:.2f}% success rate")
    
    elif args.command == "request":
        request_data = metrics_service._metrics_data.get("requests", {}).get(args.request_id)
        if not request_data:
            print(f"No data found for request: {args.request_id}")
        else:
            print(f"Request: {args.request_id}")
            print(f"Start time: {format_date(request_data.get('start_time'))}")
            print(f"End time: {format_date(request_data.get('end_time'))}")
            duration = request_data.get('duration_ms', 0)
            print(f"Duration: {duration} ms ({duration/1000:.2f} seconds)")
            print(f"Success: {request_data.get('success', False)}")
            if request_data.get('error'):
                print(f"Error: {request_data['error']}")
            
            print("\nPrompt:")
            print(request_data.get('prompt', 'No prompt available'))
            
            print("\nAgents used:")
            for agent_id in request_data.get('agents_used', []):
                print(f"  {agent_id}")
            
            print("\nTools used:")
            for tool_id in request_data.get('tools_used', []):
                print(f"  {tool_id}")
            
            print("\nModels used:")
            for model_id in request_data.get('models_used', []):
                print(f"  {model_id}")
    
    elif args.command == "report":
        report = dashboard.generate_performance_report(days=args.days)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2))
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 