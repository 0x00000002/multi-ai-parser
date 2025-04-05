"""
Metrics Dashboard Module.

This module provides visualization and analysis capabilities for request metrics data.
"""
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .request_metrics import RequestMetricsService


class MetricsDashboard:
    """
    Dashboard for visualizing and analyzing metrics data.
    """
    
    def __init__(self, metrics_service: Optional[RequestMetricsService] = None):
        """
        Initialize the metrics dashboard.
        
        Args:
            metrics_service: RequestMetricsService instance (or create new if None)
        """
        self.metrics_service = metrics_service or RequestMetricsService()
    
    def plot_agent_usage(self, 
                        top_n: int = 5, 
                        days: int = 30,
                        save_path: Optional[str] = None):
        """
        Plot agent usage metrics.
        
        Args:
            top_n: Number of top agents to include
            days: Number of days to include in the report
            save_path: Path to save the plot (or None to display)
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get agent metrics
        agent_metrics = self.metrics_service.get_agent_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert to DataFrame for easier plotting
        agents_df = pd.DataFrame.from_dict(agent_metrics, orient='index')
        
        # Sort and get top N agents by total requests
        top_agents = agents_df.sort_values('total_requests', ascending=False).head(top_n)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total requests
        top_agents['total_requests'].plot(
            kind='bar', 
            ax=axes[0, 0], 
            title=f'Total Requests by Agent (Last {days} Days)'
        )
        
        # Plot success rate
        top_agents['period_success_rate'].plot(
            kind='bar', 
            ax=axes[0, 1], 
            title=f'Success Rate by Agent (Last {days} Days)'
        )
        
        # Plot average duration
        top_agents['avg_duration_ms'].plot(
            kind='bar', 
            ax=axes[1, 0], 
            title=f'Average Duration (ms) by Agent'
        )
        
        # Plot average confidence
        top_agents['avg_confidence'].plot(
            kind='bar', 
            ax=axes[1, 1], 
            title=f'Average Confidence by Agent'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_tool_usage(self, 
                       top_n: int = 5, 
                       days: int = 30,
                       save_path: Optional[str] = None):
        """
        Plot tool usage metrics.
        
        Args:
            top_n: Number of top tools to include
            days: Number of days to include in the report
            save_path: Path to save the plot (or None to display)
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get tool metrics
        tool_metrics = self.metrics_service.get_tool_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Convert to DataFrame for easier plotting
        tools_df = pd.DataFrame.from_dict(tool_metrics, orient='index')
        
        # Sort and get top N tools by total calls
        top_tools = tools_df.sort_values('total_calls', ascending=False).head(top_n)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot total calls
        top_tools['total_calls'].plot(
            kind='bar', 
            ax=axes[0, 0], 
            title=f'Total Calls by Tool (Last {days} Days)'
        )
        
        # Plot success rate
        top_tools['period_success_rate'].plot(
            kind='bar', 
            ax=axes[0, 1], 
            title=f'Success Rate by Tool (Last {days} Days)'
        )
        
        # Plot average duration
        top_tools['avg_duration_ms'].plot(
            kind='bar', 
            ax=axes[1, 0], 
            title=f'Average Duration (ms) by Tool'
        )
        
        # Plot success ratio
        success_ratio = top_tools['successful_calls'] / top_tools['total_calls']
        success_ratio.plot(
            kind='bar', 
            ax=axes[1, 1], 
            title=f'Success Ratio by Tool'
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Args:
            days: Number of days to include in the report
            
        Returns:
            Dictionary with performance metrics
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Get agent and tool metrics
        agent_metrics = self.metrics_service.get_agent_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        tool_metrics = self.metrics_service.get_tool_metrics(
            start_time=start_time,
            end_time=end_time
        )
        
        # Compute top performers
        top_agents = sorted(
            agent_metrics.items(),
            key=lambda x: x[1].get('period_success_rate', 0),
            reverse=True
        )[:5]
        
        top_tools = sorted(
            tool_metrics.items(),
            key=lambda x: x[1].get('period_success_rate', 0),
            reverse=True
        )[:5]
        
        # Compute bottom performers
        bottom_agents = sorted(
            agent_metrics.items(),
            key=lambda x: x[1].get('period_success_rate', 0)
        )[:5]
        
        bottom_tools = sorted(
            tool_metrics.items(),
            key=lambda x: x[1].get('period_success_rate', 0)
        )[:5]
        
        # Compile report
        report = {
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days
            },
            "top_performing_agents": [
                {
                    "agent_id": agent_id,
                    "success_rate": metrics.get('period_success_rate', 0),
                    "total_requests": metrics.get('period_requests', 0)
                }
                for agent_id, metrics in top_agents
            ],
            "bottom_performing_agents": [
                {
                    "agent_id": agent_id,
                    "success_rate": metrics.get('period_success_rate', 0),
                    "total_requests": metrics.get('period_requests', 0)
                }
                for agent_id, metrics in bottom_agents
            ],
            "top_performing_tools": [
                {
                    "tool_id": tool_id,
                    "success_rate": metrics.get('period_success_rate', 0),
                    "total_calls": metrics.get('period_calls', 0)
                }
                for tool_id, metrics in top_tools
            ],
            "bottom_performing_tools": [
                {
                    "tool_id": tool_id,
                    "success_rate": metrics.get('period_success_rate', 0),
                    "total_calls": metrics.get('period_calls', 0)
                }
                for tool_id, metrics in bottom_tools
            ]
        }
        
        return report 