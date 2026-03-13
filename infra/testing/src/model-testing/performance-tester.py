"""
Performance Tester - Enterprise Enterprise ML Performance Framework
Comprehensive performance testing for ML models crypto trading systems

Applies enterprise principles:
- Enterprise-grade benchmarking
- Production latency monitoring
- Resource utilization tracking
- Scalability validation
"""

import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
import gc
import warnings
from contextlib import contextmanager
from pathlib import Path
import logging

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning("GPUtil not set - GPU metrics unavailable")

try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    logger.warning("memory_profiler not set - detailed memory")


class PerformanceMetric(Enum):
    """Types metrics performance for Enterprise monitoring"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    GPU_MEMORY = "gpu_memory"
    BATCH_EFFICIENCY = "batch_efficiency"
    CONCURRENT_PERFORMANCE = "concurrent_performance"


class LoadPattern(Enum):
    """load for Enterprise stress testing"""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    BURST = "burst"
    SUSTAINED = "sustained"


@dataclass
class PerformanceResult:
    """Result testing performance - Enterprise structured result"""
    metric_type: PerformanceMetric
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"{self.metric_type.value}: {self.value:.4f} {self.unit}"


@dataclass
class BenchmarkConfig:
    """Configuration benchmark - Enterprise typed configuration"""
    # Basic settings
    num_iterations: int = 100
    warmup_iterations: int = 10
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 128])
    
    # Concurrency settings
    max_concurrent_requests: int = 50
    thread_pool_size: int = 4
    process_pool_size: int = 2
    
    # Load testing
    load_pattern: LoadPattern = LoadPattern.CONSTANT
    test_duration_seconds: int = 60
    ramp_up_duration_seconds: int = 10
    
    # Resource monitoring
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_gpu: bool = GPU_AVAILABLE
    sampling_interval: float = 0.1
    
    # Enterprise enterprise settings
    enable_detailed_profiling: bool = False
    save_intermediate_results: bool = True
    export_metrics_to_mlflow: bool = False
    
    # Crypto trading specific
    simulate_market_data: bool = True
    market_data_frequency: str = "1s"  # 1s, 5s, 1m, 5m, etc.
    max_acceptable_latency_ms: float = 50.0  # Critical for HFT


class ResourceMonitor:
    """Enterprise resource monitoring for ML performance testing"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self.monitoring = False
        self.metrics: Dict[str, List[Tuple[datetime, float]]] = {
            'cpu_percent': [],
            'memory_mb': [],
            'gpu_percent': [],
            'gpu_memory_mb': []
        }
        self._monitor_thread: Optional[threading.Thread] = None
    
    def start_monitoring(self):
        """Launch monitoring resources"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        self._monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, List[Tuple[datetime, float]]]:
        """Stopping monitoring results"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        return self.metrics.copy()
    
    def _monitor_resources(self):
        """Main monitoring resources"""
        process = psutil.Process()
        
        while self.monitoring:
            timestamp = datetime.now()
            
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.metrics['cpu_percent'].append((timestamp, cpu_percent))
                
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.metrics['memory_mb'].append((timestamp, memory_mb))
                
                # GPU metrics (if available)
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
 gpu = gpus[0] # Using GPU
                            self.metrics['gpu_percent'].append((timestamp, gpu.load * 100))
                            self.metrics['gpu_memory_mb'].append((timestamp, gpu.memoryUsed))
                    except Exception as e:
                        logger.debug(f"GPU monitoring : {e}")
                
            except Exception as e:
                logger.warning(f"Error monitoring resources: {e}")
            
            time.sleep(self.sampling_interval)


@contextmanager
def timing_context():
    """Context manager for time"""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        return end_time - start_time


class PerformanceTester:
    """
    Enterprise Enterprise Performance Tester
    
 Comprehensive performance testing framework for ML models crypto trading systems.
 Provides enterprise-grade monitoring validation performance.
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
 Initialization performance tester Enterprise configuration
        
        Args:
            config: Configuration benchmark (Enterprise typed)
        """
        self.config = config or BenchmarkConfig()
        self.results: List[PerformanceResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enterprise monitoring components
        self.resource_monitor = ResourceMonitor(self.config.sampling_interval)
        self._benchmark_start_time: Optional[datetime] = None
        
        # Performance baselines for Enterprise compliance
        self._baseline_metrics: Dict[str, float] = {}
    
    def run_comprehensive_benchmark(
        self,
        model: Any,
        test_data: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, List[PerformanceResult]]:
        """
 benchmark models - Enterprise enterprise testing pipeline
        
        Args:
            model: ML model for testing
            test_data: Test data
            model_name: Name models for reports
        
        Returns:
            Dict[str, List[PerformanceResult]]: Results all tests
        """
        self._benchmark_start_time = datetime.now()
        self.results.clear()
        
        self.logger.info(f"Starting Enterprise comprehensive benchmark for {model_name}")
        
        benchmark_results = {}
        
        try:
            # 1. Latency Testing
            self.logger.info("Testing latency...")
            latency_results = self._test_inference_latency(model, test_data)
            benchmark_results['latency'] = latency_results
            
            # 2. Throughput Testing
            self.logger.info("Testing throughput...")
            throughput_results = self._test_throughput(model, test_data)
            benchmark_results['throughput'] = throughput_results
            
            # 3. Batch Performance
            self.logger.info("Testing batch performance...")
            batch_results = self._test_batch_performance(model, test_data)
            benchmark_results['batch'] = batch_results
            
            # 4. Memory Usage Testing
            if self.config.monitor_memory:
                self.logger.info("Testing memory usage...")
                memory_results = self._test_memory_usage(model, test_data)
                benchmark_results['memory'] = memory_results
            
            # 5. Concurrent Performance
            self.logger.info("Testing concurrent performance...")
            concurrent_results = self._test_concurrent_performance(model, test_data)
            benchmark_results['concurrent'] = concurrent_results
            
            # 6. Load Testing
            self.logger.info("load testing...")
            load_results = self._test_load_performance(model, test_data)
            benchmark_results['load'] = load_results
            
            # 7. GPU Performance (if available)
            if GPU_AVAILABLE and self.config.monitor_gpu:
                self.logger.info("Testing GPU performance...")
                gpu_results = self._test_gpu_performance(model, test_data)
                benchmark_results['gpu'] = gpu_results
            
            # 8. Crypto Trading Specific Tests
            if self.config.simulate_market_data:
                self.logger.info("Testing crypto trading scenarios...")
                crypto_results = self._test_crypto_trading_scenarios(model, test_data)
                benchmark_results['crypto_trading'] = crypto_results
            
            benchmark_time = (datetime.now() - self._benchmark_start_time).total_seconds()
            self.logger.info(f"Enterprise benchmark completed in {benchmark_time:.2f} seconds")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Error at execution benchmark: {e}")
            raise
    
    def _test_inference_latency(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Testing latency inference - Enterprise real-time performance"""
        results = []
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            model.predict(test_data[:1])
        
        # Measurement latency for single predictions
        single_latencies = []
        for i in range(self.config.num_iterations):
            sample = test_data[i % len(test_data):i % len(test_data) + 1]
            
            start_time = time.perf_counter()
            _ = model.predict(sample)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            single_latencies.append(latency_ms)
        
        # latency
        mean_latency = np.mean(single_latencies)
        median_latency = np.median(single_latencies)
        p95_latency = np.percentile(single_latencies, 95)
        p99_latency = np.percentile(single_latencies, 99)
        max_latency = np.max(single_latencies)
        
        # Results
        results.extend([
            PerformanceResult(
                PerformanceMetric.LATENCY, mean_latency, "ms",
                metadata={'type': 'mean', 'samples': len(single_latencies)}
            ),
            PerformanceResult(
                PerformanceMetric.LATENCY, median_latency, "ms",
                metadata={'type': 'median', 'samples': len(single_latencies)}
            ),
            PerformanceResult(
                PerformanceMetric.LATENCY, p95_latency, "ms",
                metadata={'type': 'p95', 'samples': len(single_latencies)}
            ),
            PerformanceResult(
                PerformanceMetric.LATENCY, p99_latency, "ms",
                metadata={'type': 'p99', 'samples': len(single_latencies)}
            ),
            PerformanceResult(
                PerformanceMetric.LATENCY, max_latency, "ms",
                metadata={'type': 'max', 'samples': len(single_latencies)}
            )
        ])
        
        # Enterprise validation - verification critical for trading
        if mean_latency > self.config.max_acceptable_latency_ms:
            self.logger.warning(
                f"latency {mean_latency:.2f}ms critical"
                f"{self.config.max_acceptable_latency_ms}ms for crypto trading"
            )
        
        return results
    
    def _test_throughput(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Testing throughput - Enterprise scalability validation"""
        results = []
        
        for batch_size in self.config.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            batch_data = test_data[:batch_size]
            throughput_measurements = []
            
            # Warmup for batch size
            for _ in range(3):
                model.predict(batch_data)
            
            # Measurement throughput
 for _ in range(self.config.num_iterations // 4): # for batch tests
                start_time = time.perf_counter()
                _ = model.predict(batch_data)
                end_time = time.perf_counter()
                
                duration = end_time - start_time
                throughput = batch_size / duration  # predictions per second
                throughput_measurements.append(throughput)
            
            mean_throughput = np.mean(throughput_measurements)
            
            results.append(PerformanceResult(
                PerformanceMetric.THROUGHPUT, mean_throughput, "predictions/sec",
                metadata={
                    'batch_size': batch_size,
                    'samples': len(throughput_measurements),
                    'std': np.std(throughput_measurements)
                }
            ))
        
        return results
    
    def _test_batch_performance(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Testing batch efficiency - Enterprise optimization metrics"""
        results = []
        
        # Baseline - single prediction performance
        single_start = time.perf_counter()
        for i in range(10):
            model.predict(test_data[i:i+1])
        single_duration = (time.perf_counter() - single_start) / 10
        
        # Batch performance for different size
        for batch_size in self.config.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            batch_data = test_data[:batch_size]
            
            # Measurement batch prediction
            start_time = time.perf_counter()
            _ = model.predict(batch_data)
            batch_duration = time.perf_counter() - start_time
            
            expected_single_duration = single_duration * batch_size
            batch_efficiency = expected_single_duration / batch_duration
            
            results.append(PerformanceResult(
                PerformanceMetric.BATCH_EFFICIENCY, batch_efficiency, "efficiency_ratio",
                metadata={
                    'batch_size': batch_size,
                    'batch_duration': batch_duration,
                    'expected_single_duration': expected_single_duration,
                    'speedup': batch_efficiency
                }
            ))
        
        return results
    
    def _test_memory_usage(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Testing memory usage - Enterprise resource optimization"""
        results = []
        
        # memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Testing memory usage for different batch sizes
        for batch_size in self.config.batch_sizes:
            if batch_size > len(test_data):
                continue
            
            batch_data = test_data[:batch_size]
            
            # Measurement before prediction
            gc.collect()
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Prediction
            predictions = model.predict(batch_data)
            
            # Measurement after prediction
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            # Memory per prediction
            memory_per_prediction = memory_used / batch_size if batch_size > 0 else memory_used
            
            results.append(PerformanceResult(
                PerformanceMetric.MEMORY_USAGE, memory_used, "MB",
                metadata={
                    'batch_size': batch_size,
                    'memory_per_prediction_mb': memory_per_prediction,
                    'baseline_memory_mb': baseline_memory,
                    'total_memory_mb': memory_after
                }
            ))
            
            # Cleanup
            del predictions
            gc.collect()
        
        return results
    
    def _test_concurrent_performance(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Testing concurrent performance - Enterprise scalability testing"""
        results = []
        
        def predict_worker(data_batch: np.ndarray) -> Tuple[float, bool]:
            """Worker function for concurrent testing"""
            try:
                start_time = time.perf_counter()
                _ = model.predict(data_batch)
                end_time = time.perf_counter()
                return end_time - start_time, True
            except Exception as e:
                logger.warning(f"Concurrent prediction failed: {e}")
                return 0.0, False
        
        # Testing concurrent requests
        for num_concurrent in [1, 2, 4, 8, 16, 32]:
            if num_concurrent > self.config.max_concurrent_requests:
                break
            
            # Preparation data for concurrent requests
            data_batches = []
            batch_size = min(32, len(test_data) // num_concurrent)
            for i in range(num_concurrent):
                start_idx = (i * batch_size) % len(test_data)
                end_idx = start_idx + batch_size
                if end_idx > len(test_data):
                    end_idx = len(test_data)
                data_batches.append(test_data[start_idx:end_idx])
            
            # Concurrent execution ThreadPoolExecutor
            concurrent_times = []
            successful_requests = 0
            
            start_time = time.perf_counter()
            with ThreadPoolExecutor(max_workers=self.config.thread_pool_size) as executor:
                futures = [executor.submit(predict_worker, batch) for batch in data_batches]
                
                for future in as_completed(futures):
                    duration, success = future.result()
                    if success:
                        concurrent_times.append(duration)
                        successful_requests += 1
            
            total_time = time.perf_counter() - start_time
            
            if successful_requests > 0:
                avg_request_time = np.mean(concurrent_times)
                total_throughput = successful_requests / total_time
                
                results.extend([
                    PerformanceResult(
                        PerformanceMetric.CONCURRENT_PERFORMANCE, avg_request_time, "seconds",
                        metadata={
                            'concurrent_requests': num_concurrent,
                            'successful_requests': successful_requests,
                            'metric_type': 'avg_request_time'
                        }
                    ),
                    PerformanceResult(
                        PerformanceMetric.CONCURRENT_PERFORMANCE, total_throughput, "requests/sec",
                        metadata={
                            'concurrent_requests': num_concurrent,
                            'successful_requests': successful_requests,
                            'metric_type': 'throughput'
                        }
                    )
                ])
        
        return results
    
    def _test_load_performance(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Load testing - Enterprise sustained performance validation"""
        results = []
        
        # Launch resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Sustained load test
            start_time = time.perf_counter()
            end_time = start_time + self.config.test_duration_seconds
            
            request_times = []
            successful_requests = 0
            failed_requests = 0
            
            while time.perf_counter() < end_time:
                try:
                    # batch size for load
                    batch_size = np.random.choice([1, 4, 8, 16])
                    sample_data = test_data[:min(batch_size, len(test_data))]
                    
                    request_start = time.perf_counter()
                    _ = model.predict(sample_data)
                    request_end = time.perf_counter()
                    
                    request_times.append(request_end - request_start)
                    successful_requests += 1
                    
                    # for real-world conditions
                    time.sleep(0.001)  # 1ms
                    
                except Exception as e:
                    failed_requests += 1
                    logger.debug(f"Load test request failed: {e}")
            
            total_duration = time.perf_counter() - start_time
            
            # Stopping monitoring
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Analysis results
            if request_times:
                avg_latency = np.mean(request_times) * 1000  # ms
                throughput = successful_requests / total_duration
                error_rate = failed_requests / (successful_requests + failed_requests) * 100
                
                results.extend([
                    PerformanceResult(
                        PerformanceMetric.LATENCY, avg_latency, "ms",
                        metadata={
                            'test_type': 'sustained_load',
                            'duration_seconds': total_duration,
                            'successful_requests': successful_requests,
                            'failed_requests': failed_requests
                        }
                    ),
                    PerformanceResult(
                        PerformanceMetric.THROUGHPUT, throughput, "requests/sec",
                        metadata={
                            'test_type': 'sustained_load',
                            'duration_seconds': total_duration,
                            'error_rate_percent': error_rate
                        }
                    )
                ])
                
                # Resource utilization metrics
                if resource_metrics['cpu_percent']:
                    cpu_values = [val for _, val in resource_metrics['cpu_percent']]
                    avg_cpu = np.mean(cpu_values)
                    max_cpu = np.max(cpu_values)
                    
                    results.append(PerformanceResult(
                        PerformanceMetric.CPU_USAGE, avg_cpu, "percent",
                        metadata={
                            'test_type': 'sustained_load',
                            'max_cpu_percent': max_cpu,
                            'samples': len(cpu_values)
                        }
                    ))
                
                if resource_metrics['memory_mb']:
                    memory_values = [val for _, val in resource_metrics['memory_mb']]
                    avg_memory = np.mean(memory_values)
                    max_memory = np.max(memory_values)
                    
                    results.append(PerformanceResult(
                        PerformanceMetric.MEMORY_USAGE, avg_memory, "MB",
                        metadata={
                            'test_type': 'sustained_load',
                            'max_memory_mb': max_memory,
                            'samples': len(memory_values)
                        }
                    ))
        
        except Exception as e:
            self.resource_monitor.stop_monitoring()
            self.logger.error(f"Load test failed: {e}")
            raise
        
        return results
    
    def _test_gpu_performance(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """GPU performance testing - Enterprise hardware optimization"""
        if not GPU_AVAILABLE:
            return []
        
        results = []
        
        try:
            # Checking, model GPU
            # This framework-specific verification
            model_uses_gpu = False
            
            # PyTorch detection
            if hasattr(model, 'device'):
                model_uses_gpu = 'cuda' in str(model.device)
            
            # TensorFlow detection
            elif hasattr(model, 'get_config'):
                # Checking through TensorFlow GPU availability
                try:
                    import tensorflow as tf
                    model_uses_gpu = len(tf.config.list_physical_devices('GPU')) > 0
                except ImportError:
                    pass
            
            if not model_uses_gpu:
                self.logger.info("Model not GPU")
                return results
            
            # GPU utilization monitoring
            gpus = GPUtil.getGPUs()
            if not gpus:
                self.logger.warning("GPU not detected")
                return results
            
            gpu = gpus[0]
            initial_gpu_memory = gpu.memoryUsed
            
            # GPU performance test
            gpu_times = []
            for batch_size in [1, 16, 64, 256]:
                if batch_size > len(test_data):
                    continue
                
                batch_data = test_data[:batch_size]
                
                # Warmup
                for _ in range(3):
                    model.predict(batch_data)
                
                # Measurement
                start_time = time.perf_counter()
                _ = model.predict(batch_data)
                # GPU sync (if can)
                if hasattr(model, 'device') and 'cuda' in str(model.device):
                    try:
                        import torch
                        torch.cuda.synchronize()
                    except ImportError:
                        pass
                
                end_time = time.perf_counter()
                gpu_times.append((batch_size, end_time - start_time))
            
            # GPU memory usage
            final_gpu_memory = gpu.memoryUsed
            gpu_memory_used = final_gpu_memory - initial_gpu_memory
            
            results.extend([
                PerformanceResult(
                    PerformanceMetric.GPU_USAGE, gpu.load * 100, "percent",
                    metadata={'gpu_name': gpu.name, 'test_type': 'utilization'}
                ),
                PerformanceResult(
                    PerformanceMetric.GPU_MEMORY, gpu_memory_used, "MB",
                    metadata={'gpu_name': gpu.name, 'total_memory_mb': gpu.memoryTotal}
                )
            ])
            
            # GPU batch performance
            for batch_size, duration in gpu_times:
                gpu_throughput = batch_size / duration
                
                results.append(PerformanceResult(
                    PerformanceMetric.THROUGHPUT, gpu_throughput, "predictions/sec",
                    metadata={
                        'device': 'gpu',
                        'batch_size': batch_size,
                        'duration': duration
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"GPU performance test failed: {e}")
        
        return results
    
    def _test_crypto_trading_scenarios(self, model: Any, test_data: np.ndarray) -> List[PerformanceResult]:
        """Crypto trading specific performance scenarios - Enterprise domain testing"""
        results = []
        
        # Scenario 1: High-Frequency Trading (HFT) simulation
        hft_latencies = []
        for _ in range(1000):  # 1000 rapid-fire predictions
            sample = test_data[np.random.randint(0, len(test_data)):np.random.randint(0, len(test_data)) + 1]
            
            start_time = time.perf_counter()
            _ = model.predict(sample)
            end_time = time.perf_counter()
            
            hft_latencies.append((end_time - start_time) * 1000)  # ms
        
        hft_mean = np.mean(hft_latencies)
        hft_p99 = np.percentile(hft_latencies, 99)
        
        results.extend([
            PerformanceResult(
                PerformanceMetric.LATENCY, hft_mean, "ms",
                metadata={'scenario': 'hft_simulation', 'metric': 'mean', 'samples': 1000}
            ),
            PerformanceResult(
                PerformanceMetric.LATENCY, hft_p99, "ms",
                metadata={'scenario': 'hft_simulation', 'metric': 'p99', 'samples': 1000}
            )
        ])
        
        # Scenario 2: Market data burst processing
        burst_sizes = [100, 500, 1000, 2000]  # Simulating market data bursts
        
        for burst_size in burst_sizes:
            if burst_size > len(test_data):
                continue
            
            burst_data = test_data[:burst_size]
            
            start_time = time.perf_counter()
            predictions = model.predict(burst_data)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = burst_size / processing_time
            
            results.append(PerformanceResult(
                PerformanceMetric.THROUGHPUT, throughput, "market_updates/sec",
                metadata={
                    'scenario': 'market_burst',
                    'burst_size': burst_size,
                    'processing_time': processing_time
                }
            ))
        
        # Scenario 3: Real-time streaming simulation
        streaming_duration = 10  # seconds
        streaming_frequency = 10  # Hz (10 predictions per second)
        
        streaming_latencies = []
        start_time = time.perf_counter()
        next_prediction_time = start_time
        
        while time.perf_counter() < start_time + streaming_duration:
            current_time = time.perf_counter()
            
            if current_time >= next_prediction_time:
                sample = test_data[np.random.randint(0, len(test_data)):np.random.randint(0, len(test_data)) + 1]
                
                pred_start = time.perf_counter()
                _ = model.predict(sample)
                pred_end = time.perf_counter()
                
                streaming_latencies.append((pred_end - pred_start) * 1000)
                next_prediction_time = current_time + (1.0 / streaming_frequency)
            
            time.sleep(0.001)  # Small sleep to prevent busy waiting
        
        if streaming_latencies:
            streaming_mean = np.mean(streaming_latencies)
            streaming_std = np.std(streaming_latencies)
            
            results.extend([
                PerformanceResult(
                    PerformanceMetric.LATENCY, streaming_mean, "ms",
                    metadata={
                        'scenario': 'real_time_streaming',
                        'frequency_hz': streaming_frequency,
                        'std_latency': streaming_std,
                        'samples': len(streaming_latencies)
                    }
                )
            ])
        
        return results
    
    def generate_performance_report(self, results: Dict[str, List[PerformanceResult]]) -> Dict[str, Any]:
        """Generation comprehensive performance report - Enterprise reporting"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'benchmark_config': {
                'num_iterations': self.config.num_iterations,
                'batch_sizes': self.config.batch_sizes,
                'max_concurrent_requests': self.config.max_concurrent_requests,
                'test_duration_seconds': self.config.test_duration_seconds
            },
            'summary': {},
            'detailed_results': {},
            'recommendations': [],
            'enterprise_compliance': {}
        }
        
        # Summary statistics
        all_results = []
        for category_results in results.values():
            all_results.extend(category_results)
        
        # Latency summary
        latency_results = [r for r in all_results if r.metric_type == PerformanceMetric.LATENCY]
        if latency_results:
            latencies = [r.value for r in latency_results]
            report['summary']['latency'] = {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': np.max(latencies)
            }
        
        # Throughput summary
        throughput_results = [r for r in all_results if r.metric_type == PerformanceMetric.THROUGHPUT]
        if throughput_results:
            throughputs = [r.value for r in throughput_results]
            report['summary']['throughput'] = {
                'max_predictions_per_sec': np.max(throughputs),
                'mean_predictions_per_sec': np.mean(throughputs),
                'by_batch_size': {}
            }
            
            # Throughput by batch size
            for result in throughput_results:
                batch_size = result.metadata.get('batch_size')
                if batch_size:
                    report['summary']['throughput']['by_batch_size'][batch_size] = result.value
        
        # Memory summary
        memory_results = [r for r in all_results if r.metric_type == PerformanceMetric.MEMORY_USAGE]
        if memory_results:
            memory_values = [r.value for r in memory_results]
            report['summary']['memory'] = {
                'max_usage_mb': np.max(memory_values),
                'mean_usage_mb': np.mean(memory_values),
                'by_batch_size': {}
            }
            
            for result in memory_results:
                batch_size = result.metadata.get('batch_size')
                if batch_size:
                    report['summary']['memory']['by_batch_size'][batch_size] = result.value
        
        # Detailed results by category
        for category, category_results in results.items():
            report['detailed_results'][category] = [
                {
                    'metric_type': r.metric_type.value,
                    'value': r.value,
                    'unit': r.unit,
                    'timestamp': r.timestamp.isoformat(),
                    'metadata': r.metadata
                }
                for r in category_results
            ]
        
        # Enterprise compliance assessment
        compliance_score = self._assess_enterprise_compliance(results)
        report['enterprise_compliance'] = compliance_score
        
        # Performance recommendations
        recommendations = self._generate_recommendations(results)
        report['recommendations'] = recommendations
        
        return report
    
    def _assess_enterprise_compliance(self, results: Dict[str, List[PerformanceResult]]) -> Dict[str, Any]:
        """Enterprise performance standards"""
        compliance = {
            'overall_score': 0,
            'latency_compliance': False,
            'throughput_compliance': False,
            'resource_efficiency': False,
            'scalability': False
        }
        
        # Latency compliance (< 50ms for crypto trading)
        latency_results = []
        for category_results in results.values():
            latency_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.LATENCY])
        
        if latency_results:
            mean_latency = np.mean([r.value for r in latency_results])
            compliance['latency_compliance'] = mean_latency < self.config.max_acceptable_latency_ms
            if compliance['latency_compliance']:
                compliance['overall_score'] += 25
        
        # Throughput compliance (> 100 predictions/sec)
        throughput_results = []
        for category_results in results.values():
            throughput_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.THROUGHPUT])
        
        if throughput_results:
            max_throughput = np.max([r.value for r in throughput_results])
            compliance['throughput_compliance'] = max_throughput > 100
            if compliance['throughput_compliance']:
                compliance['overall_score'] += 25
        
        # Resource efficiency (memory usage reasonable)
        memory_results = []
        for category_results in results.values():
            memory_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.MEMORY_USAGE])
        
        if memory_results:
            max_memory = np.max([r.value for r in memory_results])
            compliance['resource_efficiency'] = max_memory < 1000  # < 1GB
            if compliance['resource_efficiency']:
                compliance['overall_score'] += 25
        
        # Scalability (batch efficiency > 1.5x)
        batch_results = []
        for category_results in results.values():
            batch_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.BATCH_EFFICIENCY])
        
        if batch_results:
            max_efficiency = np.max([r.value for r in batch_results])
            compliance['scalability'] = max_efficiency > 1.5
            if compliance['scalability']:
                compliance['overall_score'] += 25
        
        return compliance
    
    def _generate_recommendations(self, results: Dict[str, List[PerformanceResult]]) -> List[str]:
        """Generation Enterprise performance recommendations"""
        recommendations = []
        
        # Analyze latency results
        latency_results = []
        for category_results in results.values():
            latency_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.LATENCY])
        
        if latency_results:
            mean_latency = np.mean([r.value for r in latency_results])
            if mean_latency > self.config.max_acceptable_latency_ms:
                recommendations.append(
                    f"High latency ({mean_latency:.2f}ms) - consider models or usage GPU"
                )
        
        # Analyze batch efficiency
        batch_results = []
        for category_results in results.values():
            batch_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.BATCH_EFFICIENCY])
        
        if batch_results:
            max_efficiency = np.max([r.value for r in batch_results])
            if max_efficiency < 2.0:
                recommendations.append(
                    "Low - model may"
                )
        
        # Analyze memory usage
        memory_results = []
        for category_results in results.values():
            memory_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.MEMORY_USAGE])
        
        if memory_results:
            max_memory = np.max([r.value for r in memory_results])
            if max_memory > 500:  # MB
                recommendations.append(
                    f"memory ({max_memory:.1f}MB) - consider models or usage"
                )
        
        # Analyze concurrent performance
        concurrent_results = []
        for category_results in results.values():
            concurrent_results.extend([r for r in category_results if r.metric_type == PerformanceMetric.CONCURRENT_PERFORMANCE])
        
        if concurrent_results:
            # Checking at concurrent
            single_thread_latency = None
            multi_thread_latencies = []
            
            for result in concurrent_results:
                concurrent_reqs = result.metadata.get('concurrent_requests', 1)
                if concurrent_reqs == 1:
                    single_thread_latency = result.value
                elif concurrent_reqs > 1:
                    multi_thread_latencies.append((concurrent_reqs, result.value))
            
            if single_thread_latency and multi_thread_latencies:
                worst_degradation = max(
                    latency / single_thread_latency
                    for _, latency in multi_thread_latencies
                )
                if worst_degradation > 3.0:
                    recommendations.append(
                        f"at concurrent ({worst_degradation:.1f}x) -"
                        "consider usage model or thread safety"
                    )
        
        if not recommendations:
            recommendations.append("Performance models Enterprise enterprise")
        
        return recommendations
    
    def export_results(self, results: Dict[str, List[PerformanceResult]], filepath: str) -> None:
        """Export results testing - Enterprise reporting"""
        import json
        
        report = self.generate_performance_report(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Performance report exported {filepath}")


def create_crypto_trading_performance_tester() -> PerformanceTester:
    """
    Factory function for creation performance tester for crypto trading models
    Enterprise pre-configured tester for financial ML systems
    """
    config = BenchmarkConfig(
        # Crypto trading specific settings
 num_iterations=500, # for accuracy
 warmup_iterations=20, # warmup
 batch_sizes=[1, 4, 8, 16, 32, 64, 128], # Realistic size for trading
        
        # High concurrency for trading systems
        max_concurrent_requests=100,
        thread_pool_size=8,
        
        # Sustained performance testing
 test_duration_seconds=120, # 2 sustained load
        
        # Strict requirements for crypto trading
        max_acceptable_latency_ms=20.0,  # 20ms maximum for HFT
        
        # Comprehensive monitoring
        monitor_memory=True,
        monitor_cpu=True,
        monitor_gpu=GPU_AVAILABLE,
        sampling_interval=0.05,  # 50ms sampling for accuracy
        
        # Enterprise enterprise settings
        enable_detailed_profiling=True,
        save_intermediate_results=True,
 export_metrics_to_mlflow=False, # Can at MLflow
        
        # Crypto trading simulation
        simulate_market_data=True,
 market_data_frequency="100ms" # data
    )
    
    return PerformanceTester(config)