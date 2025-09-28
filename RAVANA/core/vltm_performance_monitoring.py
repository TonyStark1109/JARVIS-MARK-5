"""
RAVANA VLTM Performance Monitoring
"""

import logging
import requests
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    ERROR_RATE = "error_rate"

class OperationType(Enum):
    """Types of operations."""
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    SEARCH = "search"

@dataclass
class PerformanceMetric:
    """Performance metric data."""
    metric_id: str
    metric_type: MetricType
    operation_type: OperationType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OperationProfile:
    """Profile of an operation."""
    operation_name: str
    operation_type: OperationType
    avg_latency: float
    max_latency: float
    min_latency: float
    success_rate: float
    total_operations: int
    error_count: int

class VLTMPerformanceMonitor:
    """Monitors VLTM performance metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = []
        self.operation_profiles = {}
        self.is_monitoring = False
        self.monitoring_start_time = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        try:
            self.is_monitoring = True
            self.monitoring_start_time = datetime.utcnow()
            self.logger.info("VLTM performance monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        try:
            self.is_monitoring = False
            self.logger.info("VLTM performance monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
    
    def record_metric(self, metric_id: str, metric_type: MetricType, 
                     operation_type: OperationType, value: float, 
                     metadata: Dict[str, Any] = None):
        """Record a performance metric."""
        try:
            if not self.is_monitoring:
                return
            
            metric = PerformanceMetric(
                metric_id=metric_id,
                metric_type=metric_type,
                operation_type=operation_type,
                value=value,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            self.metrics.append(metric)
            self.logger.debug(f"Recorded metric: {metric_id} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
    
    def record_operation(self, operation_name: str, operation_type: OperationType,
                        latency: float, success: bool, metadata: Dict[str, Any] = None):
        """Record an operation performance."""
        try:
            # Record latency metric
            self.record_metric(
                metric_id=f"{operation_name}_latency",
                metric_type=MetricType.LATENCY,
                operation_type=operation_type,
                value=latency,
                metadata=metadata
            )
            
            # Update operation profile
            if operation_name not in self.operation_profiles:
                self.operation_profiles[operation_name] = OperationProfile(
                    operation_name=operation_name,
                    operation_type=operation_type,
                    avg_latency=0.0,
                    max_latency=0.0,
                    min_latency=float('inf'),
                    success_rate=0.0,
                    total_operations=0,
                    error_count=0
                )
            
            profile = self.operation_profiles[operation_name]
            profile.total_operations += 1
            
            if not success:
                profile.error_count += 1
            
            # Update latency statistics
            if profile.total_operations == 1:
                profile.avg_latency = latency
                profile.max_latency = latency
                profile.min_latency = latency
        else:
                # Running average
                profile.avg_latency = (profile.avg_latency * (profile.total_operations - 1) + latency) / profile.total_operations
                profile.max_latency = max(profile.max_latency, latency)
                profile.min_latency = min(profile.min_latency, latency)
            
            # Update success rate
            profile.success_rate = (profile.total_operations - profile.error_count) / profile.total_operations
            
        except Exception as e:
            self.logger.error(f"Failed to record operation: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        try:
            if not self.metrics:
                return {"message": "No metrics recorded"}
            
            summary = {
                "total_metrics": len(self.metrics),
                "monitoring_duration": None,
                "metric_types": {},
                "operation_types": {},
                "recent_metrics": []
            }
            
            if self.monitoring_start_time:
                duration = datetime.utcnow() - self.monitoring_start_time
                summary["monitoring_duration"] = str(duration)
            
            # Count by metric type
            for metric in self.metrics:
                metric_type = metric.metric_type.value
                if metric_type not in summary["metric_types"]:
                    summary["metric_types"][metric_type] = 0
                summary["metric_types"][metric_type] += 1
                
                operation_type = metric.operation_type.value
                if operation_type not in summary["operation_types"]:
                    summary["operation_types"][operation_type] = 0
                summary["operation_types"][operation_type] += 1
            
            # Get recent metrics (last 10)
            recent_metrics = sorted(self.metrics, key=lambda x: x.timestamp, reverse=True)[:10]
            summary["recent_metrics"] = [
                {
                    "metric_id": m.metric_id,
                    "type": m.metric_type.value,
                    "value": m.value,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in recent_metrics
            ]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}
    
    def get_operation_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get profiles of all operations."""
        try:
            profiles = {}
            for name, profile in self.operation_profiles.items():
                profiles[name] = {
                    "operation_name": profile.operation_name,
                    "operation_type": profile.operation_type.value,
                    "avg_latency": profile.avg_latency,
                    "max_latency": profile.max_latency,
                    "min_latency": profile.min_latency,
                    "success_rate": profile.success_rate,
                    "total_operations": profile.total_operations,
                    "error_count": profile.error_count
                }
            return profiles
        except Exception as e:
            self.logger.error(f"Failed to get operation profiles: {e}")
            return {}
    
    def clear_metrics(self):
        """Clear all recorded metrics."""
        try:
            self.metrics.clear()
            self.operation_profiles.clear()
            self.logger.info("All metrics cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear metrics: {e}")
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        try:
            if format_type == "json":
                import json
                data = {
                    "metrics": [
                        {
                            "metric_id": m.metric_id,
                            "metric_type": m.metric_type.value,
                            "operation_type": m.operation_type.value,
                            "value": m.value,
                            "timestamp": m.timestamp.isoformat(),
                            "metadata": m.metadata
                        }
                        for m in self.metrics
                    ],
                    "operation_profiles": self.get_operation_profiles()
                }
                return json.dumps(data, indent=2)
            else:
                return "Unsupported format type"
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return "Export failed"

def main():
    """Main function."""
    monitor = VLTMPerformanceMonitor()
    monitor.start_monitoring()
    
    # Example usage
    monitor.record_operation("test_read", OperationType.READ, 0.1, True)
    monitor.record_operation("test_write", OperationType.WRITE, 0.2, True)
    
    print("Metrics summary:", monitor.get_metrics_summary())
    print("Operation profiles:", monitor.get_operation_profiles())

if __name__ == "__main__":
    main()