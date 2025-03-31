"""
Prompt metrics implementation.
Tracks usage and performance metrics for prompts to enable optimization.
"""
from typing import Dict, Any, Optional, List, Union
import uuid
import json
from datetime import datetime, timedelta


class PromptMetrics:
    """
    Tracks usage and performance metrics for prompt templates and versions.
    Enables A/B testing and optimization based on collected data.
    """
    
    def __init__(self):
        """Initialize the prompt metrics tracker."""
        self._usage_data = {}  # Template/version usage data
        self._performance_data = {}  # Performance metrics
        
    def record_usage(self, 
                     template_id: str, 
                     version_id: Optional[str] = None,
                     user_id: Optional[str] = None,
                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Record usage of a prompt template/version.
        
        Args:
            template_id: ID of the template used
            version_id: ID of the version used (if applicable)
            user_id: ID of the user (if applicable)
            context: Additional context information
            
        Returns:
            Generated usage record ID
        """
        usage_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        usage_record = {
            "usage_id": usage_id,
            "template_id": template_id,
            "version_id": version_id,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
            "context": context or {}
        }
        
        # Store by template ID
        if template_id not in self._usage_data:
            self._usage_data[template_id] = []
        self._usage_data[template_id].append(usage_record)
        
        return usage_id
    
    def record_performance(self, 
                          usage_id: str,
                          metrics: Dict[str, Union[float, int, str]],
                          feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Record performance metrics for a prompt usage.
        
        Args:
            usage_id: ID from record_usage
            metrics: Performance metrics (latency, tokens, etc.)
            feedback: User or system feedback (optional)
        """
        timestamp = datetime.utcnow()
        
        performance_record = {
            "usage_id": usage_id,
            "timestamp": timestamp.isoformat(),
            "metrics": metrics,
            "feedback": feedback or {}
        }
        
        # Store by usage ID
        self._performance_data[usage_id] = performance_record
    
    def get_metrics_for_template(self, 
                                template_id: str,
                                start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get aggregated metrics for a template.
        
        Args:
            template_id: ID of the template
            start_time: Start of time range (or None for all time)
            end_time: End of time range (or None for now)
            
        Returns:
            Aggregated metrics
        """
        if end_time is None:
            end_time = datetime.utcnow()
        
        # Default to 30 days ago if not specified
        if start_time is None:
            start_time = end_time - timedelta(days=30)
            
        # Collect usage records for this template
        usage_records = self._usage_data.get(template_id, [])
        
        # Filter by time range
        filtered_records = []
        for record in usage_records:
            record_time = datetime.fromisoformat(record["timestamp"])
            if start_time <= record_time <= end_time:
                filtered_records.append(record)
        
        # Count total usages
        usage_count = len(filtered_records)
        
        # Get performance data for these usages
        performance_metrics = {}
        for record in filtered_records:
            usage_id = record["usage_id"]
            if usage_id in self._performance_data:
                performance = self._performance_data[usage_id]
                
                # Collect metrics
                for key, value in performance.get("metrics", {}).items():
                    if key not in performance_metrics:
                        performance_metrics[key] = []
                    performance_metrics[key].append(value)
        
        # Calculate aggregates (min, max, avg) for each metric
        aggregated_metrics = {}
        for key, values in performance_metrics.items():
            # Only calculate aggregates for numeric values
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            if numeric_values:
                aggregated_metrics[key] = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "avg": sum(numeric_values) / len(numeric_values),
                    "count": len(numeric_values)
                }
        
        # Version distribution
        version_counts = {}
        for record in filtered_records:
            version_id = record.get("version_id", "unknown")
            if version_id not in version_counts:
                version_counts[version_id] = 0
            version_counts[version_id] += 1
        
        return {
            "template_id": template_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "usage_count": usage_count,
            "metrics": aggregated_metrics,
            "version_distribution": version_counts
        }
    
    def compare_versions(self,
                        template_id: str,
                        version_ids: List[str],
                        metric_keys: Optional[List[str]] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Compare metrics between different versions of a template.
        
        Args:
            template_id: ID of the template
            version_ids: List of version IDs to compare
            metric_keys: Specific metrics to compare (or None for all)
            start_time: Start of time range (or None for all time)
            end_time: End of time range (or None for now)
            
        Returns:
            Comparison results
        """
        if end_time is None:
            end_time = datetime.utcnow()
        
        # Default to 30 days ago if not specified
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # Results by version
        results = {}
        
        for version_id in version_ids:
            # Collect usage records for this version
            version_usage = []
            for usage_list in self._usage_data.values():
                for record in usage_list:
                    if record.get("version_id") == version_id:
                        record_time = datetime.fromisoformat(record["timestamp"])
                        if start_time <= record_time <= end_time:
                            version_usage.append(record)
            
            # Count usages
            usage_count = len(version_usage)
            
            # Get performance data
            version_metrics = {}
            for record in version_usage:
                usage_id = record["usage_id"]
                if usage_id in self._performance_data:
                    performance = self._performance_data[usage_id]
                    
                    # Filter to requested metrics if specified
                    for key, value in performance.get("metrics", {}).items():
                        if metric_keys is None or key in metric_keys:
                            if key not in version_metrics:
                                version_metrics[key] = []
                            version_metrics[key].append(value)
            
            # Calculate aggregates
            aggregated_metrics = {}
            for key, values in version_metrics.items():
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    aggregated_metrics[key] = {
                        "min": min(numeric_values),
                        "max": max(numeric_values),
                        "avg": sum(numeric_values) / len(numeric_values),
                        "count": len(numeric_values)
                    }
            
            results[version_id] = {
                "usage_count": usage_count,
                "metrics": aggregated_metrics
            }
        
        return {
            "template_id": template_id,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "versions": results
        }
    
    def recommend_version(self,
                         template_id: str,
                         optimization_metric: str,
                         higher_is_better: bool = True,
                         min_usage_count: int = 10) -> Optional[str]:
        """
        Recommend the best-performing version based on a metric.
        
        Args:
            template_id: ID of the template
            optimization_metric: Metric to optimize for
            higher_is_better: Whether higher values are better
            min_usage_count: Minimum usage count for statistical significance
            
        Returns:
            Recommended version ID or None if insufficient data
        """
        # Get metrics for all versions of this template
        all_metrics = self.get_metrics_for_template(template_id)
        
        # Check each version
        best_version = None
        best_value = None
        
        for version_id, count in all_metrics.get("version_distribution", {}).items():
            # Skip versions with insufficient data
            if count < min_usage_count:
                continue
                
            # Find the relevant metric
            for metric_key, metrics in all_metrics.get("metrics", {}).items():
                if metric_key == optimization_metric:
                    avg_value = metrics.get("avg")
                    
                    # Update best if this is better
                    if avg_value is not None:
                        if best_value is None:
                            best_value = avg_value
                            best_version = version_id
                        elif higher_is_better and avg_value > best_value:
                            best_value = avg_value
                            best_version = version_id
                        elif not higher_is_better and avg_value < best_value:
                            best_value = avg_value
                            best_version = version_id
        
        return best_version
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics data to a dictionary for serialization.
        
        Returns:
            Dictionary of all metrics data
        """
        return {
            "usage_data": self._usage_data,
            "performance_data": self._performance_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptMetrics':
        """
        Create metrics from a dictionary.
        
        Args:
            data: Dictionary of metrics data
            
        Returns:
            New metrics instance
        """
        metrics = cls()
        metrics._usage_data = data.get("usage_data", {})
        metrics._performance_data = data.get("performance_data", {})
        return metrics
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save metrics to a JSON file.
        
        Args:
            file_path: Path to save the metrics
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PromptMetrics':
        """
        Load metrics from a JSON file.
        
        Args:
            file_path: Path to load the metrics from
            
        Returns:
            New metrics instance
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data) 