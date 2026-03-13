"""
Performance Profiler for NLP Sentiment Analysis

Enterprise-grade performance monitoring and profiling system with 
patterns for comprehensive system observability and optimization.

Author: ML-Framework Team
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import contextmanager
import statistics
import json
from pathlib import Path
import asyncio
from functools import wraps

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except ImportError:
    NVML_AVAILABLE = False

# Memory profiling
try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationProfile:
    """Profile data for an operation"""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    
    def add_measurement(self, duration: float, error: bool = False):
        """Add new measurement"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.count
        self.recent_times.append(duration)
        
        if error:
            self.error_count += 1
    
    def get_percentiles(self) -> Dict[str, float]:
        """Get percentile statistics"""
        if not self.recent_times:
            return {}
        
        times = list(self.recent_times)
        return {
            "p50": statistics.median(times),
            "p95": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "p99": statistics.quantiles(times, n=100)[98] if len(times) >= 100 else max(times),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = {
            "name": self.name,
            "count": self.count,
            "total_time": self.total_time,
            "min_time": self.min_time if self.min_time != float('inf') else 0.0,
            "max_time": self.max_time,
            "avg_time": self.avg_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.count if self.count > 0 else 0.0,
        }
        data.update(self.get_percentiles())
        return data


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def start(self):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                with self.lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception:
                # Continue monitoring even if collection fails
                pass
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        metrics = {
            "timestamp": timestamp,
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used,
                "free": memory.free,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            }
        }
        
        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_metrics = []
                for gpu in gpus:
                    gpu_metrics.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "temperature": gpu.temperature,
                        "load": gpu.load * 100,
                        "memory_total": gpu.memoryTotal,
                        "memory_used": gpu.memoryUsed,
                        "memory_free": gpu.memoryFree,
                        "memory_percent": (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    })
                metrics["gpus"] = gpu_metrics
            except Exception:
                pass
        
        # NVIDIA ML metrics
        if NVML_AVAILABLE:
            try:
                device_count = nvml.nvmlDeviceGetCount()
                nvml_metrics = []
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # Power usage
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    
                    # Memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    nvml_metrics.append({
                        "device_id": i,
                        "power_watts": power,
                        "memory_total": mem_info.total,
                        "memory_used": mem_info.used,
                        "memory_free": mem_info.free,
                    })
                
                metrics["nvml"] = nvml_metrics
            except Exception:
                pass
        
        return metrics
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current system metrics"""
        with self.lock:
            if self.metrics_history:
                return self.metrics_history[-1].copy()
        return None
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history"""
        with self.lock:
            history = list(self.metrics_history)
            return history[-limit:] if limit else history


class MemoryProfiler:
    """Memory usage profiling"""
    
    def __init__(self):
        self.enabled = TRACEMALLOC_AVAILABLE
        self.snapshots = []
        self.baseline = None
    
    def start(self):
        """Start memory profiling"""
        if not self.enabled:
            return
        
        tracemalloc.start()
        self.baseline = tracemalloc.take_snapshot()
    
    def stop(self):
        """Stop memory profiling"""
        if not self.enabled:
            return
        
        tracemalloc.stop()
    
    def take_snapshot(self, label: str = "snapshot"):
        """Take memory snapshot"""
        if not self.enabled:
            return None
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append({
            "label": label,
            "snapshot": snapshot,
            "timestamp": time.time(),
        })
        
        return snapshot
    
    def get_memory_diff(self, snapshot1, snapshot2) -> List[Dict[str, Any]]:
        """Get memory difference between snapshots"""
        if not self.enabled:
            return []
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        diff_data = []
        for stat in top_stats[:10]:  # Top 10 differences
            diff_data.append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_diff": stat.size_diff,
                "count_diff": stat.count_diff,
                "size": stat.size,
                "count": stat.count,
            })
        
        return diff_data
    
    def get_top_memory_usage(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top memory usage"""
        if not self.enabled or not self.snapshots:
            return []
        
        latest_snapshot = self.snapshots[-1]["snapshot"]
        top_stats = latest_snapshot.statistics('lineno')
        
        usage_data = []
        for stat in top_stats[:limit]:
            usage_data.append({
                "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size": stat.size,
                "count": stat.count,
                "size_mb": stat.size / 1024 / 1024,
            })
        
        return usage_data


class Profiler:
    """
    Enterprise Performance Profiler with enterprise integration
    
    Features:
    - Operation timing and profiling
    - System resource monitoring
    - Memory usage tracking
    - GPU utilization monitoring
    - Statistical analysis
    - Performance alerting
    - Export capabilities
    """
    
    def __init__(
        self,
        auto_start_monitoring: bool = True,
        monitoring_interval: float = 1.0,
        enable_memory_profiling: bool = True,
        max_history_size: int = 10000,
    ):
        self.operations = defaultdict(OperationProfile)
        self.metrics = deque(maxlen=max_history_size)
        self.lock = threading.Lock()
        
        # System monitoring
        self.system_monitor = SystemMonitor(interval=monitoring_interval)
        if auto_start_monitoring:
            self.system_monitor.start()
        
        # Memory profiling
        self.memory_profiler = MemoryProfiler() if enable_memory_profiling else None
        if self.memory_profiler:
            self.memory_profiler.start()
        
        # Performance thresholds
        self.thresholds = {
            "slow_operation": 1.0,  # seconds
            "memory_usage": 0.8,    # 80% of available memory
            "cpu_usage": 0.9,       # 90% CPU usage
            "gpu_usage": 0.9,       # 90% GPU usage
        }
        
        # Callbacks for alerts
        self.alert_callbacks = []
    
    def record(self, operation: str, duration: float, error: bool = False, **context):
        """Record operation performance"""
        with self.lock:
            profile = self.operations[operation]
            if not profile.name:
                profile.name = operation
            
            profile.add_measurement(duration, error)
            
            # Add metric
            metric = PerformanceMetric(
                name=operation,
                value=duration,
                unit="seconds",
                timestamp=time.time(),
                context=context
            )
            self.metrics.append(metric)
        
        # Check for performance alerts
        self._check_alerts(operation, duration, error, context)
    
    def _check_alerts(self, operation: str, duration: float, error: bool, context: Dict[str, Any]):
        """Check for performance alerts"""
        alerts = []
        
        # Slow operation alert
        if duration > self.thresholds["slow_operation"]:
            alerts.append({
                "type": "slow_operation",
                "operation": operation,
                "duration": duration,
                "threshold": self.thresholds["slow_operation"],
                "context": context,
            })
        
        # Error alert
        if error:
            alerts.append({
                "type": "operation_error",
                "operation": operation,
                "duration": duration,
                "context": context,
            })
        
        # System resource alerts
        current_metrics = self.system_monitor.get_current_metrics()
        if current_metrics:
            # Memory usage alert
            memory_percent = current_metrics.get("memory", {}).get("percent", 0) / 100
            if memory_percent > self.thresholds["memory_usage"]:
                alerts.append({
                    "type": "high_memory_usage",
                    "usage_percent": memory_percent * 100,
                    "threshold_percent": self.thresholds["memory_usage"] * 100,
                })
            
            # CPU usage alert
            cpu_percent = current_metrics.get("cpu", {}).get("percent", 0) / 100
            if cpu_percent > self.thresholds["cpu_usage"]:
                alerts.append({
                    "type": "high_cpu_usage",
                    "usage_percent": cpu_percent * 100,
                    "threshold_percent": self.thresholds["cpu_usage"] * 100,
                })
            
            # GPU usage alerts
            gpus = current_metrics.get("gpus", [])
            for gpu in gpus:
                gpu_load = gpu.get("load", 0) / 100
                if gpu_load > self.thresholds["gpu_usage"]:
                    alerts.append({
                        "type": "high_gpu_usage",
                        "gpu_id": gpu.get("id"),
                        "usage_percent": gpu_load * 100,
                        "threshold_percent": self.thresholds["gpu_usage"] * 100,
                    })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception:
                    pass  # Don't let callback errors break profiling
    
    @contextmanager
    def profile_operation(self, operation_name: str, **context):
        """Context manager for profiling operations"""
        start_time = time.time()
        error = False
        
        # Take memory snapshot if profiling enabled
        if self.memory_profiler:
            self.memory_profiler.take_snapshot(f"{operation_name}_start")
        
        try:
            yield
        except Exception as e:
            error = True
            context["error"] = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self.record(operation_name, duration, error, **context)
            
            # Take end memory snapshot
            if self.memory_profiler:
                self.memory_profiler.take_snapshot(f"{operation_name}_end")
    
    def timer(self, operation_name: str):
        """Decorator for timing functions"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_operation(operation_name, function=func.__name__):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def async_timer(self, operation_name: str):
        """Decorator for timing async functions"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.profile_operation(operation_name, function=func.__name__):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get operation statistics"""
        with self.lock:
            if operation_name:
                if operation_name in self.operations:
                    return self.operations[operation_name].to_dict()
                return {}
            else:
                return {name: profile.to_dict() for name, profile in self.operations.items()}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        current_metrics = self.system_monitor.get_current_metrics()
        metrics_history = self.system_monitor.get_metrics_history(limit=60)  # Last minute
        
        stats = {
            "current": current_metrics,
            "history_count": len(metrics_history),
        }
        
        # Calculate averages over last minute
        if metrics_history:
            cpu_values = [m.get("cpu", {}).get("percent", 0) for m in metrics_history]
            memory_values = [m.get("memory", {}).get("percent", 0) for m in metrics_history]
            
            stats["averages"] = {
                "cpu_percent": statistics.mean(cpu_values) if cpu_values else 0,
                "memory_percent": statistics.mean(memory_values) if memory_values else 0,
            }
            
            # GPU averages
            if metrics_history[0].get("gpus"):
                gpu_loads = []
                for metrics in metrics_history:
                    for gpu in metrics.get("gpus", []):
                        gpu_loads.append(gpu.get("load", 0))
                
                if gpu_loads:
                    stats["averages"]["gpu_load"] = statistics.mean(gpu_loads)
        
        return stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory profiling statistics"""
        if not self.memory_profiler:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "snapshots_count": len(self.memory_profiler.snapshots),
            "top_usage": self.memory_profiler.get_top_memory_usage(),
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            "operations": self.get_operation_stats(),
            "system": self.get_system_stats(),
            "memory": self.get_memory_stats(),
            "thresholds": self.thresholds.copy(),
            "total_operations": sum(p.count for p in self.operations.values()),
            "total_errors": sum(p.error_count for p in self.operations.values()),
        }
    
    def export_stats(self, file_path: Union[str, Path], format: str = "json"):
        """Export performance statistics to file"""
        file_path = Path(file_path)
        stats = self.get_performance_summary()
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_stats(self):
        """Reset all statistics"""
        with self.lock:
            self.operations.clear()
            self.metrics.clear()
    
    def set_threshold(self, metric: str, value: float):
        """Set performance threshold"""
        if metric in self.thresholds:
            self.thresholds[metric] = value
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert callback function"""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove alert callback function"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.system_monitor.stop()
            if self.memory_profiler:
                self.memory_profiler.stop()
        except Exception:
            pass


# Global profiler instance
_global_profiler: Optional[Profiler] = None


def get_profiler(**kwargs) -> Profiler:
    """Get global profiler instance"""
    global _global_profiler
    
    if _global_profiler is None:
        _global_profiler = Profiler(**kwargs)
    
    return _global_profiler


def set_profiler(profiler: Profiler):
    """Set global profiler instance"""
    global _global_profiler
    _global_profiler = profiler