"""
Metrics Collection for Fear & Greed Index System

Enterprise-grade metrics with Prometheus integration.
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

logger = structlog.get_logger(__name__)


@dataclass
class MetricEvent:
    """Metric event"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "counter"  # counter, gauge, histogram


class ComponentMetrics:
    """
    Metrics collector for Fear & Greed Index components

    Provides enterprise-grade observability with support for
    Prometheus metrics and structured logging.
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._lock = threading.Lock()

        # Counters
        self._counters: Dict[str, float] = defaultdict(float)

        # Gauge metrics
        self._gauges: Dict[str, float] = defaultdict(float)

        # Histograms (simple implementation)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Execution time
        self._timing_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Errors
        self._errors: Dict[str, int] = defaultdict(int)

        # Recent events
        self._recent_events: deque = deque(maxlen=100)

        logger.info("ComponentMetrics initialized", component=component_name)

    def record_counter(self, name: str, value: float = 1.0, **labels) -> None:
        """Record a counter"""
        with self._lock:
            full_name = f"{self.component_name}_{name}"
            self._counters[full_name] += value

            event = MetricEvent(
                name=full_name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
                metric_type="counter"
            )
            self._recent_events.append(event)

            logger.debug("Counter recorded",
                        metric=full_name, value=value, labels=labels)

    def record_gauge(self, name: str, value: float, **labels) -> None:
        """Record a gauge metric"""
        with self._lock:
            full_name = f"{self.component_name}_{name}"
            self._gauges[full_name] = value

            event = MetricEvent(
                name=full_name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
                metric_type="gauge"
            )
            self._recent_events.append(event)

            logger.debug("Gauge recorded",
                        metric=full_name, value=value, labels=labels)

    def record_histogram(self, name: str, value: float, **labels) -> None:
        """Record a value in a histogram"""
        with self._lock:
            full_name = f"{self.component_name}_{name}"
            self._histograms[full_name].append(value)

            event = MetricEvent(
                name=full_name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels,
                metric_type="histogram"
            )
            self._recent_events.append(event)

    def record_timing(self, name: str, duration: float) -> None:
        """Record execution time in seconds"""
        with self._lock:
            full_name = f"{self.component_name}_{name}_duration"
            self._timing_data[full_name].append(duration)

            # Also record as a histogram
            self.record_histogram(f"{name}_duration_seconds", duration)

    def record_error(self, error_type: str) -> None:
        """Record an error"""
        with self._lock:
            full_name = f"{self.component_name}_{error_type}"
            self._errors[full_name] += 1

            # Also as a counter
            self.record_counter(f"{error_type}_total", 1.0)

            logger.warning("Error recorded",
                          component=self.component_name,
                          error_type=error_type)

    def record_calculation(self, calculation_type: str, count: int = 1) -> None:
        """Record a completed calculation"""
        self.record_counter(f"{calculation_type}_total", count)

    def record_collection(self, data_type: str, count: int) -> None:
        """Record data collection"""
        self.record_counter(f"{data_type}_collected_total", count)
        self.record_gauge(f"{data_type}_last_collection_size", count)

    def time_operation(self, operation_name: str):
        """Decorator for measuring operation time"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_timing(operation_name, duration)

            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_timing(operation_name, duration)

            # Return the appropriate wrapper
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def get_counters(self) -> Dict[str, float]:
        """Get all counters"""
        with self._lock:
            return dict(self._counters)

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauge metrics"""
        with self._lock:
            return dict(self._gauges)

    def get_histogram_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get histogram statistics"""
        full_name = f"{self.component_name}_{name}"

        with self._lock:
            if full_name not in self._histograms:
                return None

            values = list(self._histograms[full_name])

            if not values:
                return None

            import numpy as np
            return {
                "count": len(values),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "std": float(np.std(values))
            }

    def get_timing_stats(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get execution time statistics"""
        full_name = f"{self.component_name}_{operation_name}_duration"

        with self._lock:
            if full_name not in self._timing_data:
                return None

            timings = list(self._timing_data[full_name])

            if not timings:
                return None

            import numpy as np
            return {
                "count": len(timings),
                "min_seconds": float(np.min(timings)),
                "max_seconds": float(np.max(timings)),
                "avg_seconds": float(np.mean(timings)),
                "median_seconds": float(np.median(timings)),
                "p95_seconds": float(np.percentile(timings, 95)),
                "total_seconds": float(np.sum(timings))
            }

    def get_error_counts(self) -> Dict[str, int]:
        """Get error counts"""
        with self._lock:
            return dict(self._errors)

    def get_metrics(self) -> Dict[str, Any]:
        """Get all component metrics"""
        metrics = {
            "component": self.component_name,
            "timestamp": datetime.utcnow().isoformat(),
            "counters": self.get_counters(),
            "gauges": self.get_gauges(),
            "errors": self.get_error_counts(),
            "histograms": {},
            "timings": {}
        }

        # Add histogram statistics
        with self._lock:
            for hist_name in self._histograms.keys():
                simple_name = hist_name.replace(f"{self.component_name}_", "")
                stats = self.get_histogram_stats(simple_name)
                if stats:
                    metrics["histograms"][simple_name] = stats

            # Add execution time statistics
            for timing_name in self._timing_data.keys():
                simple_name = timing_name.replace(f"{self.component_name}_", "").replace("_duration", "")
                stats = self.get_timing_stats(simple_name)
                if stats:
                    metrics["timings"][simple_name] = stats

        return metrics

    def get_recent_events(self, limit: int = 50) -> List[MetricEvent]:
        """Get recent metric events"""
        with self._lock:
            events = list(self._recent_events)
            return events[-limit:] if len(events) > limit else events

    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)"""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timing_data.clear()
            self._errors.clear()
            self._recent_events.clear()

            logger.info("Metrics reset", component=self.component_name)

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        lines = []
        timestamp = int(time.time() * 1000)

        with self._lock:
            # Counters
            for name, value in self._counters.items():
                lines.append(f"# TYPE {name} counter")
                lines.append(f'{name}{{component="{self.component_name}"}} {value} {timestamp}')

            # Gauge metrics
            for name, value in self._gauges.items():
                lines.append(f"# TYPE {name} gauge")
                lines.append(f'{name}{{component="{self.component_name}"}} {value} {timestamp}')

            # Histograms (simplified)
            for name, values in self._histograms.items():
                if values:
                    import numpy as np
                    count = len(values)
                    total = sum(values)

                    lines.append(f"# TYPE {name} histogram")
                    lines.append(f'{name}_count{{component="{self.component_name}"}} {count} {timestamp}')
                    lines.append(f'{name}_sum{{component="{self.component_name}"}} {total} {timestamp}')

        return "\n".join(lines)

    def __str__(self) -> str:
        """String representation of metrics"""
        return f"ComponentMetrics({self.component_name}): {len(self._counters)} counters, {len(self._gauges)} gauges"


# Global metrics registry
_global_metrics: Dict[str, ComponentMetrics] = {}


def get_component_metrics(component_name: str) -> ComponentMetrics:
    """
    Get component metrics (singleton pattern)

    Args:
        component_name: Component name

    Returns:
        ComponentMetrics instance
    """
    if component_name not in _global_metrics:
        _global_metrics[component_name] = ComponentMetrics(component_name)

    return _global_metrics[component_name]


def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all components"""
    all_metrics = {}

    for component_name, metrics in _global_metrics.items():
        all_metrics[component_name] = metrics.get_metrics()

    return all_metrics


def export_all_prometheus() -> str:
    """Export all metrics in Prometheus format"""
    all_exports = []

    for component_name, metrics in _global_metrics.items():
        export = metrics.export_prometheus_format()
        if export:
            all_exports.append(export)

    return "\n\n".join(all_exports)
