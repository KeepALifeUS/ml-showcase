"""
Benchmark Runner - Enterprise Enterprise ML Benchmarking System
Comprehensive benchmarking framework for ML models crypto trading

Applies enterprise principles:
- Enterprise performance benchmarking
- Production-ready metrics
- Automated benchmark execution
- Comparative performance analysis
"""

import os
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
import hashlib
import psutil
import gc

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not set - tracking")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not set - GPU metrics unavailable")


class BenchmarkType(Enum):
    """Types benchmarks for Enterprise performance testing"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    SCALABILITY = "scalability"
    STRESS = "stress"
    ENDURANCE = "endurance"
    COMPARISON = "comparison"
    A_B_TEST = "a_b_test"
    REGRESSION = "regression"


class BenchmarkStatus(Enum):
    """Status benchmark for Enterprise tracking"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class BenchmarkConfig:
    """Configuration benchmark - Enterprise typed configuration"""
    # Basic benchmark settings
    benchmark_name: str
    benchmark_type: BenchmarkType
    duration_seconds: int = 300  # 5 minutes default
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    
    # Resource limits
    max_memory_mb: Optional[float] = None
    max_cpu_percent: Optional[float] = None
    timeout_seconds: int = 3600  # 1 hour default timeout
    
    # Concurrency settings
    max_concurrent_requests: int = 50
    thread_pool_size: int = 4
    process_pool_size: int = 2
    
    # Data settings
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 128, 256])
    sample_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])
    
    # Quality thresholds
    min_accuracy: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_throughput: Optional[float] = None
    
    # Monitoring settings
    monitor_resources: bool = True
    monitor_gpu: bool = GPU_AVAILABLE
    sampling_interval_seconds: float = 0.1
    
    # Enterprise enterprise settings
    enable_mlflow_tracking: bool = MLFLOW_AVAILABLE
    save_detailed_logs: bool = True
    generate_visualizations: bool = False
    
    # Crypto trading specific
    simulate_market_conditions: bool = False
    market_volatility_factor: float = 1.0
    trading_frequency_hz: float = 10.0  # 10 Hz trading frequency


@dataclass
class BenchmarkResult:
    """Result benchmark - Enterprise structured result"""
    benchmark_name: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    
    # Quality metrics
    accuracy_scores: List[float] = field(default_factory=list)
    latency_percentiles: Dict[str, float] = field(default_factory=dict)
    
    # Detailed results
    detailed_measurements: Dict[str, List[float]] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialization result for saving"""
        return {
            'benchmark_name': self.benchmark_name,
            'benchmark_type': self.benchmark_type.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'metrics': self.metrics,
            'resource_usage': self.resource_usage,
            'accuracy_scores': self.accuracy_scores,
            'latency_percentiles': self.latency_percentiles,
            'detailed_measurements': self.detailed_measurements,
            'error_log': self.error_log,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """serialization result from"""
        result = cls(
            benchmark_name=data['benchmark_name'],
            benchmark_type=BenchmarkType(data['benchmark_type']),
            status=BenchmarkStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time']),
            duration_seconds=data.get('duration_seconds'),
            metrics=data.get('metrics', {}),
            resource_usage=data.get('resource_usage', {}),
            accuracy_scores=data.get('accuracy_scores', []),
            latency_percentiles=data.get('latency_percentiles', {}),
            detailed_measurements=data.get('detailed_measurements', {}),
            error_log=data.get('error_log', []),
            metadata=data.get('metadata', {})
        )
        
        if data.get('end_time'):
            result.end_time = datetime.fromisoformat(data['end_time'])
        
        return result


class ResourceMonitor:
    """Enterprise resource monitoring for benchmark execution"""
    
    def __init__(self, sampling_interval: float = 0.1, monitor_gpu: bool = False):
        self.sampling_interval = sampling_interval
        self.monitor_gpu = monitor_gpu and GPU_AVAILABLE
        self.monitoring = False
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = {
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': [],
            'network_sent_mb': [],
            'network_recv_mb': []
        }
        
        if self.monitor_gpu:
            self.metrics.update({
                'gpu_utilization': [],
                'gpu_memory_used_mb': [],
                'gpu_memory_percent': [],
                'gpu_temperature': []
            })
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._process = psutil.Process()
    
    def start_monitoring(self) -> None:
        """Launch monitoring resources"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """Stopping monitoring metrics"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return self.metrics.copy()
    
    def _monitor_loop(self) -> None:
        """Main monitoring"""
        baseline_disk_io = self._process.io_counters()
        baseline_net_io = psutil.net_io_counters()
        
        while self.monitoring:
            timestamp = datetime.now()
            
            try:
                # CPU
                cpu_percent = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self._process.memory_percent()
                
                self.metrics['cpu_percent'].append((timestamp, cpu_percent))
                self.metrics['memory_mb'].append((timestamp, memory_mb))
                self.metrics['memory_percent'].append((timestamp, memory_percent))
                
                # Disk I/O
                try:
                    current_disk_io = self._process.io_counters()
                    disk_read_mb = (current_disk_io.read_bytes - baseline_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (current_disk_io.write_bytes - baseline_disk_io.write_bytes) / 1024 / 1024
                    
                    self.metrics['disk_io_read_mb'].append((timestamp, disk_read_mb))
                    self.metrics['disk_io_write_mb'].append((timestamp, disk_write_mb))
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                # Network I/O (system-wide)
                try:
                    current_net_io = psutil.net_io_counters()
                    if current_net_io and baseline_net_io:
                        net_sent_mb = (current_net_io.bytes_sent - baseline_net_io.bytes_sent) / 1024 / 1024
                        net_recv_mb = (current_net_io.bytes_recv - baseline_net_io.bytes_recv) / 1024 / 1024
                        
                        self.metrics['network_sent_mb'].append((timestamp, net_sent_mb))
                        self.metrics['network_recv_mb'].append((timestamp, net_recv_mb))
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                # GPU metrics
                if self.monitor_gpu:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Primary GPU
                            self.metrics['gpu_utilization'].append((timestamp, gpu.load * 100))
                            self.metrics['gpu_memory_used_mb'].append((timestamp, gpu.memoryUsed))
                            self.metrics['gpu_memory_percent'].append((timestamp, gpu.memoryUtil * 100))
                            self.metrics['gpu_temperature'].append((timestamp, gpu.temperature))
                    except Exception as e:
                        logger.debug(f"GPU monitoring error: {e}")
            
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
            
            time.sleep(self.sampling_interval)
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """Retrieval summary statistics by all metric"""
        summary = {}
        
        for metric_name, measurements in self.metrics.items():
            if not measurements:
                continue
            
            values = [value for _, value in measurements]
            
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'samples': len(values)
                }
        
        return summary


class BenchmarkRunner:
    """
    Enterprise Enterprise Benchmark Runner
    
 Comprehensive benchmarking framework for ML models crypto trading systems.
 Provides enterprise-grade performance testing comparative analysis.
    """
    
    def __init__(self, output_directory: str = "./benchmark_results"):
        """
        Initialization benchmark runner
        
        Args:
 output_directory: for saving results
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enterprise tracking
        self.benchmark_history: List[BenchmarkResult] = []
        self.active_benchmarks: Dict[str, BenchmarkResult] = {}
        
        # MLflow tracking
        if MLFLOW_AVAILABLE:
            self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow for experiment tracking"""
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("ML_Model_Benchmarks")
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {e}")
    
    def run_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        target_data: Optional[np.ndarray] = None
    ) -> BenchmarkResult:
        """
        Execution comprehensive benchmark models
        
        Args:
            model: ML model for testing
            test_data: Test data
            config: Configuration benchmark
            target_data: Target data (for accuracy testing)
        
        Returns:
            BenchmarkResult: Result benchmark
        """
        self.logger.info(f"Launch benchmark: {config.benchmark_name}")
        
        # Create benchmark result
        result = BenchmarkResult(
            benchmark_name=config.benchmark_name,
            benchmark_type=config.benchmark_type,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now(),
            metadata={
                'model_type': type(model).__name__,
                'test_data_shape': test_data.shape,
                'config': config.__dict__
            }
        )
        
        self.active_benchmarks[config.benchmark_name] = result
        
        # Resource monitoring
        resource_monitor = None
        if config.monitor_resources:
            resource_monitor = ResourceMonitor(
                sampling_interval=config.sampling_interval_seconds,
                monitor_gpu=config.monitor_gpu
            )
            resource_monitor.start_monitoring()
        
        # MLflow run
        mlflow_run = None
        if config.enable_mlflow_tracking and MLFLOW_AVAILABLE:
            mlflow_run = mlflow.start_run(run_name=config.benchmark_name)
            mlflow.log_params({
                'benchmark_type': config.benchmark_type.value,
                'model_type': type(model).__name__,
                'test_data_shape': str(test_data.shape)
            })
        
        try:
            # Execution benchmark by
            if config.benchmark_type == BenchmarkType.LATENCY:
                self._run_latency_benchmark(model, test_data, config, result)
            
            elif config.benchmark_type == BenchmarkType.THROUGHPUT:
                self._run_throughput_benchmark(model, test_data, config, result)
            
            elif config.benchmark_type == BenchmarkType.MEMORY:
                self._run_memory_benchmark(model, test_data, config, result)
            
            elif config.benchmark_type == BenchmarkType.ACCURACY:
                if target_data is not None:
                    self._run_accuracy_benchmark(model, test_data, target_data, config, result)
                else:
                    raise ValueError("target_data required for accuracy benchmark")
            
            elif config.benchmark_type == BenchmarkType.SCALABILITY:
                self._run_scalability_benchmark(model, test_data, config, result)
            
            elif config.benchmark_type == BenchmarkType.STRESS:
                self._run_stress_benchmark(model, test_data, config, result)
            
            elif config.benchmark_type == BenchmarkType.ENDURANCE:
                self._run_endurance_benchmark(model, test_data, config, result)
            
            else:
                raise ValueError(f"Unsupported benchmark type: {config.benchmark_type}")
            
            result.status = BenchmarkStatus.COMPLETED
            
        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.error_log.append(f"Benchmark failed: {str(e)}")
            self.logger.error(f"Benchmark {config.benchmark_name} failed: {e}")
        
        finally:
            # Stopping resource monitoring
            if resource_monitor:
                resource_usage = resource_monitor.stop_monitoring()
                result.resource_usage = {
                    metric: [value for _, value in measurements]
                    for metric, measurements in resource_usage.items()
                }
            
            # Completion benchmark
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            # MLflow logging
            if mlflow_run and MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics(result.metrics)
                    if result.accuracy_scores:
                        mlflow.log_metric("mean_accuracy", np.mean(result.accuracy_scores))
                    mlflow.end_run()
                except Exception as e:
                    self.logger.warning(f"MLflow logging failed: {e}")
            
            # Save result
            self._save_benchmark_result(result)
            self.benchmark_history.append(result)
            
            if config.benchmark_name in self.active_benchmarks:
                del self.active_benchmarks[config.benchmark_name]
        
        self.logger.info(f"Benchmark {config.benchmark_name} completed status: {result.status.value}")
        return result
    
    def _run_latency_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Benchmark latency models"""
        latencies = []
        
        # Warmup
        for _ in range(config.warmup_iterations):
            sample = test_data[:1]
            model.predict(sample)
        
        # Latency measurements
        for i in range(config.measurement_iterations):
            sample_idx = i % len(test_data)
            sample = test_data[sample_idx:sample_idx+1]
            
            start_time = time.perf_counter()
            _ = model.predict(sample)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Statistics
        result.metrics.update({
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies)
        })
        
        result.latency_percentiles = {
            'p50': np.percentile(latencies, 50),
            'p90': np.percentile(latencies, 90),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'p99.9': np.percentile(latencies, 99.9)
        }
        
        result.detailed_measurements['latencies_ms'] = latencies
        
        # Threshold validation
        if config.max_latency_ms and result.metrics['mean_latency_ms'] > config.max_latency_ms:
            result.error_log.append(f"Latency threshold exceeded: {result.metrics['mean_latency_ms']:.2f}ms > {config.max_latency_ms}ms")
    
    def _run_throughput_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Benchmark throughput models for different batch sizes"""
        throughput_results = {}
        
        for batch_size in config.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            batch_data = test_data[:batch_size]
            batch_throughputs = []
            
            # Warmup for each batch size
            for _ in range(3):
                model.predict(batch_data)
            
            # Throughput measurements
            for _ in range(config.measurement_iterations // 4):  # Fewer iterations for batch tests
                start_time = time.perf_counter()
                _ = model.predict(batch_data)
                end_time = time.perf_counter()
                
                duration = end_time - start_time
                throughput = batch_size / duration  # predictions per second
                batch_throughputs.append(throughput)
            
            mean_throughput = np.mean(batch_throughputs)
            throughput_results[f'batch_{batch_size}'] = {
                'mean_throughput': mean_throughput,
                'std_throughput': np.std(batch_throughputs),
                'measurements': batch_throughputs
            }
            
            result.metrics[f'throughput_batch_{batch_size}'] = mean_throughput
        
        # Overall throughput metrics
        all_throughputs = []
        for batch_info in throughput_results.values():
            all_throughputs.extend(batch_info['measurements'])
        
        if all_throughputs:
            result.metrics.update({
                'max_throughput': max(all_throughputs),
                'mean_throughput': np.mean(all_throughputs)
            })
        
        result.detailed_measurements['throughput_by_batch'] = throughput_results
        
        # Threshold validation
        if config.min_throughput and result.metrics.get('max_throughput', 0) < config.min_throughput:
            result.error_log.append(f"Throughput threshold not met: {result.metrics.get('max_throughput', 0):.2f} < {config.min_throughput}")
    
    def _run_memory_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Benchmark memory usage models"""
        import psutil
        import gc
        
        process = psutil.Process()
        memory_measurements = []
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory usage for different batch sizes
        for batch_size in config.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            batch_data = test_data[:batch_size]
            
            # Memory measurement
            gc.collect()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Prediction
            predictions = model.predict(batch_data)
            
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            memory_per_prediction = memory_used / batch_size if batch_size > 0 else memory_used
            
            memory_measurements.append({
                'batch_size': batch_size,
                'memory_used_mb': memory_used,
                'memory_per_prediction_mb': memory_per_prediction,
                'total_memory_mb': memory_after
            })
            
            result.metrics[f'memory_batch_{batch_size}_mb'] = memory_used
            
            # Cleanup
            del predictions
            gc.collect()
        
        # Summary memory metrics
        if memory_measurements:
            total_memories = [m['memory_used_mb'] for m in memory_measurements]
            per_prediction_memories = [m['memory_per_prediction_mb'] for m in memory_measurements]
            
            result.metrics.update({
                'baseline_memory_mb': baseline_memory,
                'max_memory_usage_mb': max(total_memories),
                'mean_memory_usage_mb': np.mean(total_memories),
                'max_memory_per_prediction_mb': max(per_prediction_memories),
                'mean_memory_per_prediction_mb': np.mean(per_prediction_memories)
            })
        
        result.detailed_measurements['memory_by_batch'] = memory_measurements
        
        # Threshold validation
        if config.max_memory_mb and result.metrics.get('max_memory_usage_mb', 0) > config.max_memory_mb:
            result.error_log.append(f"Memory threshold exceeded: {result.metrics['max_memory_usage_mb']:.1f}MB > {config.max_memory_mb}MB")
    
    def _run_accuracy_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        target_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Benchmark accuracy models on different sample sizes"""
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        
        accuracy_results = {}
        
        for sample_size in config.sample_sizes:
            if sample_size > len(test_data):
                sample_size = len(test_data)
            
            # Random sample
            indices = np.random.choice(len(test_data), size=sample_size, replace=False)
            X_sample = test_data[indices]
            y_sample = target_data[indices]
            
            # Predictions
            y_pred = model.predict(X_sample)
            
            # Metrics calculation
            if len(np.unique(y_sample)) <= 10:  # Classification
                accuracy = accuracy_score(y_sample, np.round(y_pred))
                result.accuracy_scores.append(accuracy)
                accuracy_results[f'sample_{sample_size}'] = {
                    'accuracy': accuracy,
                    'sample_size': sample_size
                }
            else:  # Regression
                mse = mean_squared_error(y_sample, y_pred)
                r2 = r2_score(y_sample, y_pred)
                accuracy_results[f'sample_{sample_size}'] = {
                    'mse': mse,
                    'r2_score': r2,
                    'rmse': np.sqrt(mse),
                    'sample_size': sample_size
                }
                result.accuracy_scores.append(r2)  # Use R² as accuracy proxy
        
        # Summary accuracy metrics
        if result.accuracy_scores:
            result.metrics.update({
                'mean_accuracy': np.mean(result.accuracy_scores),
                'std_accuracy': np.std(result.accuracy_scores),
                'min_accuracy': np.min(result.accuracy_scores),
                'max_accuracy': np.max(result.accuracy_scores)
            })
        
        result.detailed_measurements['accuracy_by_sample_size'] = accuracy_results
        
        # Threshold validation
        if config.min_accuracy and result.metrics.get('mean_accuracy', 0) < config.min_accuracy:
            result.error_log.append(f"Accuracy threshold not met: {result.metrics['mean_accuracy']:.3f} < {config.min_accuracy}")
    
    def _run_scalability_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Benchmark scalability models increase load"""
        scalability_results = {}
        concurrent_loads = [1, 2, 4, 8, 16, 32]
        
        def prediction_worker(data_batch: np.ndarray) -> Tuple[float, bool]:
            """Worker function for concurrent testing"""
            try:
                start_time = time.perf_counter()
                _ = model.predict(data_batch)
                end_time = time.perf_counter()
                return end_time - start_time, True
            except Exception:
                return 0.0, False
        
        for num_concurrent in concurrent_loads:
            if num_concurrent > config.max_concurrent_requests:
                break
            
            # Preparation data for concurrent requests
            batch_size = min(32, len(test_data) // num_concurrent)
            data_batches = []
            
            for i in range(num_concurrent):
                start_idx = (i * batch_size) % len(test_data)
                end_idx = start_idx + batch_size
                if end_idx > len(test_data):
                    end_idx = len(test_data)
                data_batches.append(test_data[start_idx:end_idx])
            
            # Concurrent execution
            response_times = []
            successful_requests = 0
            
            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=config.thread_pool_size) as executor:
                futures = [executor.submit(prediction_worker, batch) for batch in data_batches]
                
                for future in as_completed(futures):
                    duration, success = future.result()
                    if success:
                        response_times.append(duration)
                        successful_requests += 1
            
            total_time = time.perf_counter() - start_time
            
            # Scalability metrics
            if response_times:
                avg_response_time = np.mean(response_times)
                throughput = successful_requests / total_time
                
                scalability_results[f'concurrent_{num_concurrent}'] = {
                    'avg_response_time': avg_response_time,
                    'throughput': throughput,
                    'successful_requests': successful_requests,
                    'total_requests': num_concurrent,
                    'success_rate': successful_requests / num_concurrent
                }
                
                result.metrics[f'scalability_{num_concurrent}_response_time'] = avg_response_time
                result.metrics[f'scalability_{num_concurrent}_throughput'] = throughput
        
        result.detailed_measurements['scalability_results'] = scalability_results
        
        # Scalability analysis
        if len(scalability_results) >= 2:
            throughputs = [r['throughput'] for r in scalability_results.values()]
            response_times = [r['avg_response_time'] for r in scalability_results.values()]
            
            # Scalability efficiency (throughput should increase with load)
            throughput_trend = np.polyfit(range(len(throughputs)), throughputs, 1)[0]
            response_time_trend = np.polyfit(range(len(response_times)), response_times, 1)[0]
            
            result.metrics.update({
                'scalability_throughput_trend': throughput_trend,
                'scalability_response_time_trend': response_time_trend,
                'scalability_efficiency': throughput_trend / (response_time_trend + 1e-6)
            })
    
    def _run_stress_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Stress testing models at load"""
        stress_results = {
            'error_count': 0,
            'timeout_count': 0,
            'memory_spikes': 0,
            'response_times': [],
            'error_messages': []
        }
        
        import psutil
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # High-intensity stress test
        start_time = time.perf_counter()
        end_time = start_time + config.duration_seconds
        
        request_count = 0
        while time.perf_counter() < end_time:
            try:
                # Random batch size for unpredictable load
                batch_size = np.random.choice([1, 4, 8, 16, 32, 64, 128])
                batch_size = min(batch_size, len(test_data))
                
                sample_data = test_data[:batch_size]
                
                request_start = time.perf_counter()
                _ = model.predict(sample_data)
                request_end = time.perf_counter()
                
                response_time = request_end - request_start
                stress_results['response_times'].append(response_time)
                
                # Memory monitoring
                current_memory = process.memory_info().rss / 1024 / 1024
                if current_memory > baseline_memory * 2:  # 2x memory spike
                    stress_results['memory_spikes'] += 1
                
                request_count += 1
                
                # Brief pause to simulate real-world conditions
                time.sleep(0.001)  # 1ms
                
            except Exception as e:
                stress_results['error_count'] += 1
                stress_results['error_messages'].append(str(e))
            
            # Timeout check
            if time.perf_counter() - request_start > 1.0:  # 1 second timeout
                stress_results['timeout_count'] += 1
        
        # Stress test metrics
        total_duration = time.perf_counter() - start_time
        
        result.metrics.update({
            'stress_total_requests': request_count,
            'stress_requests_per_second': request_count / total_duration,
            'stress_error_rate': stress_results['error_count'] / request_count if request_count > 0 else 0,
            'stress_timeout_rate': stress_results['timeout_count'] / request_count if request_count > 0 else 0,
            'stress_memory_spikes': stress_results['memory_spikes']
        })
        
        if stress_results['response_times']:
            result.metrics.update({
                'stress_mean_response_time': np.mean(stress_results['response_times']),
                'stress_p95_response_time': np.percentile(stress_results['response_times'], 95),
                'stress_max_response_time': np.max(stress_results['response_times'])
            })
        
        result.detailed_measurements['stress_test_results'] = stress_results
        
        # Error logging
        if stress_results['error_count'] > 0:
            result.error_log.append(f"Stress test errors: {stress_results['error_count']}")
            result.error_log.extend(stress_results['error_messages'][:5])  # First 5 errors
    
    def _run_endurance_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        config: BenchmarkConfig,
        result: BenchmarkResult
    ) -> None:
        """Endurance testing models on"""
        endurance_results = {
            'time_windows': [],
            'performance_degradation': [],
            'memory_growth': [],
            'error_rates': []
        }
        
        import psutil
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Time windows for analysis (every 5% from total time)
        window_duration = config.duration_seconds / 20  # 20 windows
        total_windows = 20
        
        start_time = time.perf_counter()
        
        for window_idx in range(total_windows):
            window_start = time.perf_counter()
            window_end = window_start + window_duration
            
            window_response_times = []
            window_errors = 0
            window_requests = 0
            
            # Predictions window
            while time.perf_counter() < window_end:
                try:
                    batch_size = np.random.choice([1, 8, 32])
                    sample_data = test_data[:min(batch_size, len(test_data))]
                    
                    pred_start = time.perf_counter()
                    _ = model.predict(sample_data)
                    pred_end = time.perf_counter()
                    
                    window_response_times.append(pred_end - pred_start)
                    window_requests += 1
                    
                    time.sleep(0.01)  # 10ms between predictions
                    
                except Exception:
                    window_errors += 1
                    window_requests += 1
            
            # Window metrics
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - baseline_memory
            
            window_performance = np.mean(window_response_times) if window_response_times else float('inf')
            error_rate = window_errors / window_requests if window_requests > 0 else 0
            
            endurance_results['time_windows'].append(window_idx + 1)
            endurance_results['performance_degradation'].append(window_performance)
            endurance_results['memory_growth'].append(memory_growth)
            endurance_results['error_rates'].append(error_rate)
        
        # Endurance analysis
        if len(endurance_results['performance_degradation']) > 1:
            # Performance trend analysis
            performance_trend = np.polyfit(
                endurance_results['time_windows'],
                endurance_results['performance_degradation'],
                1
            )[0]
            
            # Memory leak detection
            memory_trend = np.polyfit(
                endurance_results['time_windows'],
                endurance_results['memory_growth'],
                1
            )[0]
            
            # Error rate trend
            error_trend = np.polyfit(
                endurance_results['time_windows'],
                endurance_results['error_rates'],
                1
            )[0]
            
            result.metrics.update({
                'endurance_performance_trend': performance_trend,
                'endurance_memory_trend': memory_trend,
                'endurance_error_trend': error_trend,
                'endurance_max_memory_growth': max(endurance_results['memory_growth']),
                'endurance_final_error_rate': endurance_results['error_rates'][-1] if endurance_results['error_rates'] else 0
            })
            
            # Endurance quality assessment
            performance_degradation = (
                endurance_results['performance_degradation'][-1] / endurance_results['performance_degradation'][0]
                if endurance_results['performance_degradation'][0] > 0 else 1.0
            )
            
            result.metrics['endurance_performance_degradation_ratio'] = performance_degradation
        
        result.detailed_measurements['endurance_results'] = endurance_results
        
        # Endurance warnings
        if result.metrics.get('endurance_memory_trend', 0) > 1.0:  # >1MB/window memory growth
            result.error_log.append("Potential memory leak detected during endurance test")
        
        if result.metrics.get('endurance_performance_degradation_ratio', 1.0) > 1.5:  # 50% performance degradation
            result.error_log.append("Significant performance degradation detected during endurance test")
    
    def _save_benchmark_result(self, result: BenchmarkResult) -> None:
        """Save result benchmark file"""
        timestamp = result.start_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{timestamp}.json"
        filepath = self.output_directory / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Benchmark result saved: {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error at saving benchmark result: {e}")
    
    def compare_benchmarks(
        self,
        benchmark_names: List[str],
        metrics_to_compare: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comparison results multiple benchmarks
        
        Args:
 benchmark_names: List benchmarks for comparison
            metrics_to_compare: List metrics for comparison
        
        Returns:
            Dict[str, Any]: Results comparison
        """
        # Search benchmark results
        benchmarks_to_compare = []
        for name in benchmark_names:
            matching_benchmarks = [b for b in self.benchmark_history if b.benchmark_name == name]
            if matching_benchmarks:
                benchmarks_to_compare.append(matching_benchmarks[-1])  # Latest result
        
        if len(benchmarks_to_compare) < 2:
            return {'error': 'Need at least 2 benchmarks for comparison'}
        
        # Default metrics for comparison
        if metrics_to_compare is None:
            metrics_to_compare = [
                'mean_latency_ms', 'max_throughput', 'mean_accuracy',
                'max_memory_usage_mb', 'mean_memory_usage_mb'
            ]
        
        comparison_result = {
            'benchmarks_compared': len(benchmarks_to_compare),
            'comparison_timestamp': datetime.now().isoformat(),
            'metric_comparisons': {},
            'winner_by_metric': {},
            'overall_ranking': []
        }
        
        # Comparison by each metrics
        for metric in metrics_to_compare:
            metric_values = {}
            
            for benchmark in benchmarks_to_compare:
                if metric in benchmark.metrics:
                    metric_values[benchmark.benchmark_name] = benchmark.metrics[metric]
            
            if len(metric_values) >= 2:
                # Detection values (lower is better for latency/memory, higher for others)
                is_lower_better = any(keyword in metric.lower() for keyword in ['latency', 'memory', 'error'])
                
                if is_lower_better:
                    best_benchmark = min(metric_values.items(), key=lambda x: x[1])
                    worst_benchmark = max(metric_values.items(), key=lambda x: x[1])
                else:
                    best_benchmark = max(metric_values.items(), key=lambda x: x[1])
                    worst_benchmark = min(metric_values.items(), key=lambda x: x[1])
                
                # Improvement calculation
                improvement_ratio = abs(best_benchmark[1] - worst_benchmark[1]) / abs(worst_benchmark[1]) if worst_benchmark[1] != 0 else 0
                
                comparison_result['metric_comparisons'][metric] = {
                    'values': metric_values,
                    'best': {'name': best_benchmark[0], 'value': best_benchmark[1]},
                    'worst': {'name': worst_benchmark[0], 'value': worst_benchmark[1]},
                    'improvement_ratio': improvement_ratio,
                    'lower_is_better': is_lower_better
                }
                
                comparison_result['winner_by_metric'][metric] = best_benchmark[0]
        
        # Overall ranking (simple scoring system)
        benchmark_scores = {b.benchmark_name: 0 for b in benchmarks_to_compare}
        
        for metric_comparison in comparison_result['metric_comparisons'].values():
            winner = metric_comparison['best']['name']
            benchmark_scores[winner] += 1
        
        # Ranking by scores
        ranking = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
        comparison_result['overall_ranking'] = [
            {'name': name, 'score': score, 'metrics_won': score}
            for name, score in ranking
        ]
        
        return comparison_result
    
    def load_benchmark_results(self, results_directory: Optional[str] = None) -> List[BenchmarkResult]:
        """Load benchmark results from files"""
        directory = Path(results_directory) if results_directory else self.output_directory
        loaded_results = []
        
        for result_file in directory.glob("*.json"):
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                benchmark_result = BenchmarkResult.from_dict(result_data)
                loaded_results.append(benchmark_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to load benchmark result {result_file}: {e}")
        
        # Addition history
        self.benchmark_history.extend(loaded_results)
        self.logger.info(f"{len(loaded_results)} benchmark results")
        
        return loaded_results
    
    def generate_benchmark_report(
        self,
        benchmark_names: Optional[List[str]] = None,
        include_detailed_results: bool = False
    ) -> Dict[str, Any]:
        """Generation comprehensive benchmark report"""
        results_to_include = self.benchmark_history
        
        if benchmark_names:
            results_to_include = [
                r for r in self.benchmark_history
                if r.benchmark_name in benchmark_names
            ]
        
        if not results_to_include:
            return {'status': 'no_benchmark_results'}
        
        # Summary statistics
        total_benchmarks = len(results_to_include)
        completed_benchmarks = sum(1 for r in results_to_include if r.status == BenchmarkStatus.COMPLETED)
        failed_benchmarks = sum(1 for r in results_to_include if r.status == BenchmarkStatus.FAILED)
        
        # Benchmark types distribution
        type_distribution = {}
        for result in results_to_include:
            benchmark_type = result.benchmark_type.value
            type_distribution[benchmark_type] = type_distribution.get(benchmark_type, 0) + 1
        
        # Performance summary
        performance_summary = {}
        for result in results_to_include:
            if result.status == BenchmarkStatus.COMPLETED:
                for metric, value in result.metrics.items():
                    if metric not in performance_summary:
                        performance_summary[metric] = []
                    performance_summary[metric].append(value)
        
        # Statistical summary
        performance_stats = {}
        for metric, values in performance_summary.items():
            if values:
                performance_stats[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_benchmarks': total_benchmarks,
                'completed_benchmarks': completed_benchmarks,
                'failed_benchmarks': failed_benchmarks,
                'success_rate': (completed_benchmarks / total_benchmarks) * 100 if total_benchmarks > 0 else 0
            },
            'benchmark_types': type_distribution,
            'performance_statistics': performance_stats,
            'benchmark_list': [
                {
                    'name': r.benchmark_name,
                    'type': r.benchmark_type.value,
                    'status': r.status.value,
                    'duration_seconds': r.duration_seconds,
                    'key_metrics': {k: v for k, v in r.metrics.items() if any(keyword in k for keyword in ['latency', 'throughput', 'accuracy', 'memory'])}
                }
                for r in results_to_include
            ]
        }
        
        if include_detailed_results:
            report['detailed_results'] = [r.to_dict() for r in results_to_include]
        
        return report
    
    def export_report(self, filepath: str, benchmark_names: Optional[List[str]] = None) -> None:
        """Export benchmark report file"""
        report = self.generate_benchmark_report(benchmark_names, include_detailed_results=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Benchmark report exported {filepath}")


def create_crypto_trading_benchmark_configs() -> Dict[str, BenchmarkConfig]:
    """
    Factory function for creation benchmark configurations for crypto trading models
    Enterprise pre-configured benchmarks for financial ML systems
    """
    configs = {}
    
    # Latency benchmark for HFT
    configs['crypto_hft_latency'] = BenchmarkConfig(
        benchmark_name='crypto_hft_latency',
        benchmark_type=BenchmarkType.LATENCY,
        duration_seconds=300,  # 5 minutes
        warmup_iterations=50,
 measurement_iterations=1000, # for accuracy
        max_latency_ms=10.0,  # 10ms maximum for HFT
        batch_sizes=[1],  # Single predictions for HFT
        monitor_resources=True,
        monitor_gpu=GPU_AVAILABLE,
        enable_mlflow_tracking=MLFLOW_AVAILABLE,
        simulate_market_conditions=True,
        trading_frequency_hz=100.0  # 100 Hz for HFT
    )
    
    # Throughput benchmark
    configs['crypto_throughput'] = BenchmarkConfig(
        benchmark_name='crypto_throughput',
        benchmark_type=BenchmarkType.THROUGHPUT,
        duration_seconds=600,  # 10 minutes
        warmup_iterations=20,
        measurement_iterations=200,
        min_throughput=1000.0,  # 1000 predictions/sec minimum
        batch_sizes=[1, 8, 16, 32, 64, 128, 256],
        max_concurrent_requests=100,
        thread_pool_size=8,
        monitor_resources=True,
        simulate_market_conditions=True,
        market_volatility_factor=1.5  # High volatility simulation
    )
    
    # Memory efficiency benchmark
    configs['crypto_memory'] = BenchmarkConfig(
        benchmark_name='crypto_memory',
        benchmark_type=BenchmarkType.MEMORY,
        duration_seconds=300,
        warmup_iterations=10,
        measurement_iterations=50,
        max_memory_mb=512.0,  # 512MB maximum
        batch_sizes=[1, 16, 64, 256, 1024],
        monitor_resources=True,
        monitor_gpu=GPU_AVAILABLE
    )
    
    # Accuracy benchmark
    configs['crypto_accuracy'] = BenchmarkConfig(
        benchmark_name='crypto_accuracy',
        benchmark_type=BenchmarkType.ACCURACY,
        duration_seconds=900,  # 15 minutes
        warmup_iterations=5,
        measurement_iterations=20,
        min_accuracy=0.65,  # 65% minimum for crypto predictions
        sample_sizes=[100, 500, 1000, 5000, 10000],
        monitor_resources=True
    )
    
    # Scalability benchmark
    configs['crypto_scalability'] = BenchmarkConfig(
        benchmark_name='crypto_scalability',
        benchmark_type=BenchmarkType.SCALABILITY,
        duration_seconds=1200,  # 20 minutes
        warmup_iterations=10,
        measurement_iterations=100,
        max_concurrent_requests=200,
        thread_pool_size=16,
        batch_sizes=[1, 8, 32, 128],
        monitor_resources=True,
        monitor_gpu=GPU_AVAILABLE
    )
    
    # Stress test
    configs['crypto_stress'] = BenchmarkConfig(
        benchmark_name='crypto_stress',
        benchmark_type=BenchmarkType.STRESS,
        duration_seconds=1800,  # 30 minutes stress test
        max_memory_mb=1024.0,  # 1GB memory limit
        max_cpu_percent=90.0,  # 90% CPU limit
        max_concurrent_requests=500,
        thread_pool_size=32,
        monitor_resources=True,
        monitor_gpu=GPU_AVAILABLE,
        simulate_market_conditions=True,
        market_volatility_factor=2.0  # Extreme volatility
    )
    
    # Endurance test
    configs['crypto_endurance'] = BenchmarkConfig(
        benchmark_name='crypto_endurance',
        benchmark_type=BenchmarkType.ENDURANCE,
        duration_seconds=7200,  # 2 hours endurance
        warmup_iterations=20,
        measurement_iterations=1000,
        max_memory_mb=1024.0,
        monitor_resources=True,
        monitor_gpu=GPU_AVAILABLE,
        trading_frequency_hz=1.0,  # 1 Hz for long-term testing
        simulate_market_conditions=True
    )
    
    return configs