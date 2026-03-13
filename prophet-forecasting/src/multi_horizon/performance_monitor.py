"""
Horizon Performance Monitor
ML-Framework-1329 - Multi-horizon forecasting performance monitoring

 2025: Real-time monitoring, alerting, performance analytics,
enterprise-grade observability for trading systems.
"""

import asyncio
import logging
import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import json
import numpy as np

from .horizon_config import HorizonConfig


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of monitored metrics"""
    EXECUTION_TIME = "execution_time"
    ACCURACY = "accuracy"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    CONFIDENCE_SCORE = "confidence_score"


@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    horizon_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'metric_type': self.metric_type.value,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'horizon_name': self.horizon_name,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged,
            'resolved': self.resolved
        }


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    timestamp: datetime
    horizon_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'horizon_name': self.horizon_name,
            'metadata': self.metadata
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance"""
    timestamp: datetime
    metrics: Dict[str, float]
    horizon_metrics: Dict[str, Dict[str, float]]
    system_metrics: Dict[str, float]
    alerts_active: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'horizon_metrics': self.horizon_metrics,
            'system_metrics': self.system_metrics,
            'alerts_active': self.alerts_active
        }


class HorizonPerformanceMonitor:
    """
    Comprehensive performance monitoring for multi-horizon forecasting.

    Provides real-time monitoring, alerting, and analytics for
    enterprise-grade trading system observability.
    """

    def __init__(
        self,
        monitoring_interval_seconds: int = 30,
        retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enable_system_monitoring: bool = True
    ):
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.retention_hours = retention_hours
        self.enable_system_monitoring = enable_system_monitoring

        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'execution_time_ms': 150.0,         # 150ms threshold
            'memory_usage_mb': 1024.0,          # 1GB threshold
            'cpu_usage_percent': 80.0,          # 80% CPU threshold
            'error_rate_percent': 10.0,         # 10% error rate threshold
            'accuracy_threshold': 0.6,          # 60% accuracy threshold
            'throughput_min': 0.5               # Minimum 0.5 predictions/second
        }

        # Metrics storage
        self.metrics_history: deque = deque(maxlen=int(retention_hours * 3600 / monitoring_interval_seconds))
        self.horizon_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.performance_snapshots: deque = deque(maxlen=2880)  # 24 hours at 30-second intervals

        # Alerting
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.start_time = datetime.now()

        # Statistics
        self.execution_count = 0
        self.total_execution_time_ms = 0.0
        self.error_count = 0

        self.logger = logging.getLogger("HorizonPerformanceMonitor")
        self.logger.info("HorizonPerformanceMonitor initialized")

    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Performance monitoring stopped")

    def record_execution(self, result: Any):
        """Record execution metrics from a multi-horizon result"""
        try:
            timestamp = datetime.now()

            # Extract metrics from result
            if hasattr(result, 'total_execution_time_ms'):
                execution_time = result.total_execution_time_ms
                self.record_metric('execution_time', execution_time, timestamp)

            if hasattr(result, 'success_rate'):
                self.record_metric('success_rate', result.success_rate, timestamp)

            if hasattr(result, 'average_confidence'):
                self.record_metric('confidence_score', result.average_confidence, timestamp)

            if hasattr(result, 'horizons'):
                # Record per-horizon metrics
                for horizon_name, horizon_result in result.horizons.items():
                    if hasattr(horizon_result, 'execution_time_ms'):
                        self.record_horizon_metric(
                            horizon_name, 'execution_time',
                            horizon_result.execution_time_ms, timestamp
                        )

                    if hasattr(horizon_result, 'confidence_score'):
                        self.record_horizon_metric(
                            horizon_name, 'confidence_score',
                            horizon_result.confidence_score, timestamp
                        )

                    if hasattr(horizon_result, 'error') and horizon_result.error:
                        self.record_horizon_metric(
                            horizon_name, 'error_occurred', 1.0, timestamp
                        )

            # Update statistics
            self.execution_count += 1
            if hasattr(result, 'total_execution_time_ms'):
                self.total_execution_time_ms += result.total_execution_time_ms

        except Exception as e:
            self.logger.error(f"Failed to record execution metrics: {e}")

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a general performance metric"""
        timestamp = timestamp or datetime.now()
        metadata = metadata or {}

        metric = PerformanceMetric(
            name=metric_name,
            value=value,
            timestamp=timestamp,
            metadata=metadata
        )

        self.metrics_history.append(metric)

        # Check for alert conditions
        self._check_alert_conditions(metric_name, value)

    def record_horizon_metric(
        self,
        horizon_name: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ):
        """Record a horizon-specific metric"""
        timestamp = timestamp or datetime.now()

        metric = PerformanceMetric(
            name=metric_name,
            value=value,
            timestamp=timestamp,
            horizon_name=horizon_name
        )

        self.horizon_metrics[horizon_name].append(metric)

        # Check for horizon-specific alert conditions
        self._check_horizon_alert_conditions(horizon_name, metric_name, value)

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Take performance snapshot
                snapshot = self._take_performance_snapshot()
                self.performance_snapshots.append(snapshot)

                # System monitoring
                if self.enable_system_monitoring:
                    self._monitor_system_resources()

                # Clean up old data
                self._cleanup_old_data()

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval_seconds)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval_seconds)

    def _take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take a snapshot of current performance"""
        timestamp = datetime.now()

        # Calculate recent metrics
        recent_metrics = self._calculate_recent_metrics()

        # Calculate per-horizon metrics
        horizon_metrics = {}
        for horizon_name, metrics in self.horizon_metrics.items():
            if metrics:
                recent_horizon_metrics = [m for m in metrics if
                                        (timestamp - m.timestamp).total_seconds() < 300]  # Last 5 minutes
                if recent_horizon_metrics:
                    horizon_metrics[horizon_name] = self._calculate_metrics_summary(recent_horizon_metrics)

        # System metrics
        system_metrics = {}
        if self.enable_system_monitoring:
            system_metrics = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': psutil.virtual_memory().used / (1024 * 1024),
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0.0
            }

        return PerformanceSnapshot(
            timestamp=timestamp,
            metrics=recent_metrics,
            horizon_metrics=horizon_metrics,
            system_metrics=system_metrics,
            alerts_active=len(self.active_alerts)
        )

    def _calculate_recent_metrics(self, window_minutes: int = 5) -> Dict[str, float]:
        """Calculate metrics for recent time window"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        if not recent_metrics:
            return {}

        # Group metrics by name
        grouped_metrics = defaultdict(list)
        for metric in recent_metrics:
            grouped_metrics[metric.name].append(metric.value)

        # Calculate summary statistics
        summary = {}
        for metric_name, values in grouped_metrics.items():
            if values:
                summary[f"{metric_name}_avg"] = np.mean(values)
                summary[f"{metric_name}_max"] = np.max(values)
                summary[f"{metric_name}_min"] = np.min(values)
                summary[f"{metric_name}_latest"] = values[-1]

        return summary

    def _calculate_metrics_summary(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """Calculate summary statistics for a list of metrics"""
        if not metrics:
            return {}

        # Group by metric name
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric.name].append(metric.value)

        summary = {}
        for metric_name, values in grouped.items():
            if values:
                summary[f"{metric_name}_avg"] = np.mean(values)
                summary[f"{metric_name}_count"] = len(values)

        return summary

    def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.record_metric('cpu_usage_percent', cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent

            self.record_metric('memory_usage_mb', memory_mb)
            self.record_metric('memory_usage_percent', memory_percent)

            # Disk usage (if available)
            try:
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.record_metric('disk_usage_percent', disk_percent)
            except:
                pass  # Disk monitoring not available

        except Exception as e:
            self.logger.warning(f"System monitoring error: {e}")

    def _check_alert_conditions(self, metric_name: str, value: float):
        """Check if metric value triggers an alert"""
        alert_key = f"global_{metric_name}"

        # Check specific thresholds
        if metric_name == 'execution_time' and value > self.alert_thresholds.get('execution_time_ms', 150):
            self._create_alert(
                alert_key, AlertSeverity.WARNING, MetricType.EXECUTION_TIME,
                f"Execution time ({value:.1f}ms) exceeded threshold", value,
                self.alert_thresholds['execution_time_ms']
            )

        elif metric_name == 'success_rate' and value < self.alert_thresholds.get('accuracy_threshold', 0.6):
            self._create_alert(
                alert_key, AlertSeverity.ERROR, MetricType.ACCURACY,
                f"Success rate ({value:.3f}) below threshold", value,
                self.alert_thresholds['accuracy_threshold']
            )

        elif metric_name == 'memory_usage_mb' and value > self.alert_thresholds.get('memory_usage_mb', 1024):
            self._create_alert(
                alert_key, AlertSeverity.WARNING, MetricType.MEMORY_USAGE,
                f"Memory usage ({value:.1f}MB) exceeded threshold", value,
                self.alert_thresholds['memory_usage_mb']
            )

        elif metric_name == 'cpu_usage_percent' and value > self.alert_thresholds.get('cpu_usage_percent', 80):
            self._create_alert(
                alert_key, AlertSeverity.WARNING, MetricType.CPU_USAGE,
                f"CPU usage ({value:.1f}%) exceeded threshold", value,
                self.alert_thresholds['cpu_usage_percent']
            )

    def _check_horizon_alert_conditions(self, horizon_name: str, metric_name: str, value: float):
        """Check if horizon-specific metric triggers an alert"""
        alert_key = f"{horizon_name}_{metric_name}"

        if metric_name == 'execution_time' and value > self.alert_thresholds.get('execution_time_ms', 150):
            self._create_alert(
                alert_key, AlertSeverity.WARNING, MetricType.EXECUTION_TIME,
                f"Horizon {horizon_name} execution time ({value:.1f}ms) exceeded threshold",
                value, self.alert_thresholds['execution_time_ms'], horizon_name
            )

        elif metric_name == 'error_occurred' and value > 0:
            self._create_alert(
                alert_key, AlertSeverity.ERROR, MetricType.ERROR_RATE,
                f"Error occurred in horizon {horizon_name}",
                value, 0.0, horizon_name
            )

    def _create_alert(
        self,
        alert_key: str,
        severity: AlertSeverity,
        metric_type: MetricType,
        message: str,
        value: float,
        threshold: float,
        horizon_name: Optional[str] = None
    ):
        """Create or update an alert"""
        alert = Alert(
            alert_id=alert_key,
            severity=severity,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold,
            horizon_name=horizon_name
        )

        # Store in active alerts
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)

        # Log alert
        self.logger.warning(f"ALERT [{severity.value.upper()}]: {message}")

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            self.logger.info(f"Alert {alert_id} resolved")
            return True
        return False

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert notification callback"""
        self.alert_callbacks.append(callback)

    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)

        # Clean up horizon metrics
        for horizon_name in self.horizon_metrics:
            metrics = self.horizon_metrics[horizon_name]
            # Remove old metrics
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        current_time = datetime.now()
        uptime_hours = (current_time - self.start_time).total_seconds() / 3600

        # Recent performance
        recent_metrics = self._calculate_recent_metrics(window_minutes=10)

        # Calculate rates
        avg_execution_time = (self.total_execution_time_ms / self.execution_count
                            if self.execution_count > 0 else 0)

        return {
            'uptime_hours': uptime_hours,
            'total_executions': self.execution_count,
            'error_count': self.error_count,
            'average_execution_time_ms': avg_execution_time,
            'recent_metrics': recent_metrics,
            'active_alerts_count': len(self.active_alerts),
            'total_alerts_generated': len(self.alert_history),
            'monitoring_active': self.is_monitoring,
            'horizons_monitored': len(self.horizon_metrics),
            'data_points_stored': len(self.metrics_history),
            'alert_thresholds': self.alert_thresholds.copy()
        }

    def get_performance_report(self, hours: int = 1) -> Dict[str, Any]:
        """Generate performance report for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Filter recent snapshots
        recent_snapshots = [s for s in self.performance_snapshots if s.timestamp >= cutoff_time]

        if not recent_snapshots:
            return {'error': 'No data available for specified time period'}

        # Analyze snapshots
        execution_times = []
        memory_usage = []
        cpu_usage = []

        for snapshot in recent_snapshots:
            if 'execution_time_avg' in snapshot.metrics:
                execution_times.append(snapshot.metrics['execution_time_avg'])
            if 'memory_usage_mb' in snapshot.system_metrics:
                memory_usage.append(snapshot.system_metrics['memory_usage_mb'])
            if 'cpu_percent' in snapshot.system_metrics:
                cpu_usage.append(snapshot.system_metrics['cpu_percent'])

        report = {
            'time_period_hours': hours,
            'snapshots_analyzed': len(recent_snapshots),
            'performance_summary': {},
            'horizon_analysis': {},
            'alerts_in_period': []
        }

        # Performance summary
        if execution_times:
            report['performance_summary']['execution_time'] = {
                'average_ms': np.mean(execution_times),
                'max_ms': np.max(execution_times),
                'min_ms': np.min(execution_times),
                'target_met_percentage': (np.sum(np.array(execution_times) <=
                                                self.alert_thresholds['execution_time_ms']) /
                                        len(execution_times)) * 100
            }

        if memory_usage:
            report['performance_summary']['memory_usage'] = {
                'average_mb': np.mean(memory_usage),
                'max_mb': np.max(memory_usage),
                'peak_usage_mb': np.max(memory_usage)
            }

        # Alerts in period
        recent_alerts = [alert for alert in self.alert_history
                        if alert.timestamp >= cutoff_time]
        report['alerts_in_period'] = [alert.to_dict() for alert in recent_alerts]

        return report

    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format"""
        if format.lower() == 'json':
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'monitoring_summary': self.get_metrics(),
                'recent_snapshots': [snapshot.to_dict() for snapshot in
                                   list(self.performance_snapshots)[-100:]],  # Last 100 snapshots
                'active_alerts': [alert.to_dict() for alert in self.active_alerts.values()],
                'alert_history': [alert.to_dict() for alert in list(self.alert_history)[-50:]]  # Last 50 alerts
            }
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup(self):
        """Clean up monitoring resources"""
        self.stop_monitoring()
        self.metrics_history.clear()
        self.horizon_metrics.clear()
        self.performance_snapshots.clear()
        self.active_alerts.clear()
        self.alert_history.clear()
        self.alert_callbacks.clear()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()