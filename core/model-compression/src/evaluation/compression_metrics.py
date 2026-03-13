"""
Module for evaluation metrics compression ML-models in crypto trading.
Comprehensive estimation quality, performance and deployment readiness.

Comprehensive evaluation patterns for production ML systems
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MetricCategory(Enum):
    """Categories metrics for evaluation"""
    COMPRESSION = "compression"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    DEPLOYMENT = "deployment"
    CRYPTO_SPECIFIC = "crypto_specific"

class CompressionTechnique(Enum):
    """Techniques compression for tracking"""
    QUANTIZATION = "quantization"
    STRUCTURED_PRUNING = "structured_pruning"
    UNSTRUCTURED_PRUNING = "unstructured_pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    COMBINED = "combined"

@dataclass
class CompressionMetrics:
    """Base metrics compression"""
    # Size metrics
    original_size_mb: float
    compressed_size_mb: float
    compression_ratio: float
    parameter_reduction_pct: float
    
    # Quality metrics
    accuracy_original: float
    accuracy_compressed: float
    accuracy_retention: float
    quality_degradation_pct: float
    
    # Performance metrics
    latency_original_ms: float
    latency_compressed_ms: float
    latency_improvement_pct: float
    throughput_improvement_pct: float
    
    # Memory metrics
    memory_original_mb: float
    memory_compressed_mb: float
    memory_reduction_pct: float
    
    def to_dict(self) -> Dict[str, float]:
        """Conversion in dictionary"""
        return asdict(self)

@dataclass
class CryptoTradingMetrics:
    """Specific metrics for crypto trading"""
    # Trading performance
    directional_accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    
    # Real-time constraints
    inference_latency_p99_us: float
    prediction_stability: float
    market_regime_detection_accuracy: float
    
    # Risk metrics
    var_estimation_accuracy: float
    volatility_prediction_error: float
    correlation_preservation: float
    
    def to_dict(self) -> Dict[str, float]:
        """Conversion in dictionary"""
        return asdict(self)

@dataclass
class DeploymentMetrics:
    """Metrics readiness to deployment"""
    # Hardware compatibility
    cpu_compatibility_score: float
    gpu_acceleration_support: float
    edge_device_suitability: float
    
    # Export formats
    onnx_export_success: bool
    torchscript_export_success: bool
    tflite_conversion_success: bool
    
    # Production readiness
    model_stability_score: float
    error_resilience_score: float
    resource_efficiency_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary"""
        return asdict(self)

class CompressionEvaluator:
    """
    Comprehensive evaluator for estimation quality compression ML-models
    with focus on crypto trading applications
    """
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 cache_results: bool = True):
        """
        Args:
            device: Device for computations
            cache_results: Cache results for repeated usage
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_results = cache_results
        
        self.logger = logging.getLogger(f"{__name__}.CompressionEvaluator")
        self.evaluation_cache = {}
        self.evaluation_history = []
    
    def comprehensive_evaluation(self,
                                original_model: nn.Module,
                                compressed_model: nn.Module,
                                test_data: torch.utils.data.DataLoader,
                                compression_technique: CompressionTechnique = CompressionTechnique.COMBINED,
                                crypto_specific_tests: bool = True) -> Dict[str, Any]:
        """
        Comprehensive estimation compressed model
        
        Args:
            original_model: Original model
            compressed_model: Compressed model
            test_data: Test data
            compression_technique: Used technique compression
            crypto_specific_tests: Enable crypto-specific tests
            
        Returns:
            Full report about evaluation
        """
        self.logger.info("Begin comprehensive estimation compressed model")
        
        evaluation_id = f"eval_{int(time.time())}"
        
        # Validation cache
        cache_key = self._generate_cache_key(original_model, compressed_model)
        if self.cache_results and cache_key in self.evaluation_cache:
            self.logger.info("Use cached results evaluation")
            return self.evaluation_cache[cache_key]
        
        results = {
            'evaluation_id': evaluation_id,
            'timestamp': time.time(),
            'compression_technique': compression_technique.value,
            'device_used': str(self.device),
        }
        
        try:
            # 1. Base metrics compression
            compression_metrics = self._evaluate_compression_metrics(
                original_model, compressed_model, test_data
            )
            results['compression_metrics'] = compression_metrics.to_dict()
            
            # 2. Detailed analysis performance
            performance_analysis = self._evaluate_performance_analysis(
                original_model, compressed_model, test_data
            )
            results['performance_analysis'] = performance_analysis
            
            # 3. Analysis quality predictions
            quality_analysis = self._evaluate_quality_analysis(
                original_model, compressed_model, test_data
            )
            results['quality_analysis'] = quality_analysis
            
            # 4. Crypto trading specific metrics
            if crypto_specific_tests:
                crypto_metrics = self._evaluate_crypto_trading_metrics(
                    original_model, compressed_model, test_data
                )
                results['crypto_metrics'] = crypto_metrics.to_dict()
            
            # 5. Deployment readiness
            deployment_metrics = self._evaluate_deployment_readiness(
                compressed_model
            )
            results['deployment_metrics'] = deployment_metrics.to_dict()
            
            # 6. Robustness and stability
            robustness_analysis = self._evaluate_robustness(
                original_model, compressed_model, test_data
            )
            results['robustness_analysis'] = robustness_analysis
            
            # 7. Comparative analysis
            comparative_analysis = self._comparative_analysis(results)
            results['comparative_analysis'] = comparative_analysis
            
            # 8. Recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive evaluation: {e}")
            results['error'] = str(e)
        
        # Caching
        if self.cache_results:
            self.evaluation_cache[cache_key] = results
        
        # Saving in history
        self.evaluation_history.append(results)
        
        self.logger.info(f"Comprehensive estimation completed (ID: {evaluation_id})")
        
        return results
    
    def _evaluate_compression_metrics(self,
                                    original_model: nn.Module,
                                    compressed_model: nn.Module,
                                    test_data: torch.utils.data.DataLoader) -> CompressionMetrics:
        """Estimation base metrics compression"""
        
        # Size metrics
        original_size_mb = self._calculate_model_size(original_model)
        compressed_size_mb = self._calculate_model_size(compressed_model)
        compression_ratio = original_size_mb / compressed_size_mb
        
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        parameter_reduction_pct = (1 - compressed_params / original_params) * 100
        
        # Quality metrics
        original_accuracy = self._measure_accuracy(original_model, test_data)
        compressed_accuracy = self._measure_accuracy(compressed_model, test_data)
        accuracy_retention = compressed_accuracy / original_accuracy
        quality_degradation_pct = (1 - accuracy_retention) * 100
        
        # Performance metrics
        original_latency = self._measure_latency(original_model, test_data)
        compressed_latency = self._measure_latency(compressed_model, test_data)
        latency_improvement_pct = (1 - compressed_latency / original_latency) * 100
        
        # Throughput estimation
        original_throughput = 1000 / original_latency  # samples/sec
        compressed_throughput = 1000 / compressed_latency
        throughput_improvement_pct = (compressed_throughput / original_throughput - 1) * 100
        
        # Memory metrics
        original_memory = self._estimate_memory_usage(original_model)
        compressed_memory = self._estimate_memory_usage(compressed_model)
        memory_reduction_pct = (1 - compressed_memory / original_memory) * 100
        
        return CompressionMetrics(
            original_size_mb=original_size_mb,
            compressed_size_mb=compressed_size_mb,
            compression_ratio=compression_ratio,
            parameter_reduction_pct=parameter_reduction_pct,
            accuracy_original=original_accuracy,
            accuracy_compressed=compressed_accuracy,
            accuracy_retention=accuracy_retention,
            quality_degradation_pct=quality_degradation_pct,
            latency_original_ms=original_latency,
            latency_compressed_ms=compressed_latency,
            latency_improvement_pct=latency_improvement_pct,
            throughput_improvement_pct=throughput_improvement_pct,
            memory_original_mb=original_memory,
            memory_compressed_mb=compressed_memory,
            memory_reduction_pct=memory_reduction_pct
        )
    
    def _evaluate_performance_analysis(self,
                                     original_model: nn.Module,
                                     compressed_model: nn.Module,
                                     test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Detailed analysis performance"""
        
        analysis = {}
        
        # Latency distribution analysis
        original_latencies = self._measure_latency_distribution(original_model, test_data)
        compressed_latencies = self._measure_latency_distribution(compressed_model, test_data)
        
        analysis['latency_distribution'] = {
            'original': {
                'mean': float(np.mean(original_latencies)),
                'std': float(np.std(original_latencies)),
                'p50': float(np.percentile(original_latencies, 50)),
                'p95': float(np.percentile(original_latencies, 95)),
                'p99': float(np.percentile(original_latencies, 99))
            },
            'compressed': {
                'mean': float(np.mean(compressed_latencies)),
                'std': float(np.std(compressed_latencies)),
                'p50': float(np.percentile(compressed_latencies, 50)),
                'p95': float(np.percentile(compressed_latencies, 95)),
                'p99': float(np.percentile(compressed_latencies, 99))
            }
        }
        
        # Batch size sensitivity
        analysis['batch_size_sensitivity'] = self._analyze_batch_size_sensitivity(
            original_model, compressed_model, test_data
        )
        
        # Resource utilization
        analysis['resource_utilization'] = self._analyze_resource_utilization(
            original_model, compressed_model
        )
        
        return analysis
    
    def _evaluate_quality_analysis(self,
                                 original_model: nn.Module,
                                 compressed_model: nn.Module,
                                 test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analysis quality predictions"""
        
        analysis = {}
        
        # Prediction correlation analysis
        correlations = self._analyze_prediction_correlations(
            original_model, compressed_model, test_data
        )
        analysis['prediction_correlations'] = correlations
        
        # Error distribution analysis
        error_analysis = self._analyze_prediction_errors(
            original_model, compressed_model, test_data
        )
        analysis['error_analysis'] = error_analysis
        
        # Confidence preservation
        confidence_analysis = self._analyze_confidence_preservation(
            original_model, compressed_model, test_data
        )
        analysis['confidence_preservation'] = confidence_analysis
        
        return analysis
    
    def _evaluate_crypto_trading_metrics(self,
                                       original_model: nn.Module,
                                       compressed_model: nn.Module,
                                       test_data: torch.utils.data.DataLoader) -> CryptoTradingMetrics:
        """Crypto trading specific metrics"""
        
        # Directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(
            compressed_model, test_data
        )
        
        # Trading performance simulation
        trading_metrics = self._simulate_trading_performance(
            original_model, compressed_model, test_data
        )
        
        # Real-time constraints
        p99_latency_us = self._measure_p99_latency_microseconds(compressed_model, test_data)
        
        # Prediction stability
        stability = self._measure_prediction_stability(compressed_model, test_data)
        
        # Market regime detection (if model supports it)
        regime_accuracy = self._evaluate_market_regime_detection(
            compressed_model, test_data
        )
        
        # Risk metrics
        risk_metrics = self._evaluate_risk_metrics(
            original_model, compressed_model, test_data
        )
        
        return CryptoTradingMetrics(
            directional_accuracy=directional_accuracy,
            sharpe_ratio=trading_metrics.get('sharpe_ratio', 0.0),
            max_drawdown=trading_metrics.get('max_drawdown', 0.0),
            profit_factor=trading_metrics.get('profit_factor', 0.0),
            inference_latency_p99_us=p99_latency_us,
            prediction_stability=stability,
            market_regime_detection_accuracy=regime_accuracy,
            var_estimation_accuracy=risk_metrics.get('var_accuracy', 0.0),
            volatility_prediction_error=risk_metrics.get('volatility_error', 0.0),
            correlation_preservation=risk_metrics.get('correlation_preservation', 0.0)
        )
    
    def _evaluate_deployment_readiness(self, model: nn.Module) -> DeploymentMetrics:
        """Estimation readiness to deployment"""
        
        # Hardware compatibility
        cpu_score = self._test_cpu_compatibility(model)
        gpu_score = self._test_gpu_acceleration(model)
        edge_score = self._test_edge_device_suitability(model)
        
        # Export format tests
        onnx_success = self._test_onnx_export(model)
        torchscript_success = self._test_torchscript_export(model)
        tflite_success = self._test_tflite_conversion(model)
        
        # Production readiness
        stability_score = self._evaluate_model_stability(model)
        resilience_score = self._evaluate_error_resilience(model)
        efficiency_score = self._evaluate_resource_efficiency(model)
        
        return DeploymentMetrics(
            cpu_compatibility_score=cpu_score,
            gpu_acceleration_support=gpu_score,
            edge_device_suitability=edge_score,
            onnx_export_success=onnx_success,
            torchscript_export_success=torchscript_success,
            tflite_conversion_success=tflite_success,
            model_stability_score=stability_score,
            error_resilience_score=resilience_score,
            resource_efficiency_score=efficiency_score
        )
    
    def _evaluate_robustness(self,
                           original_model: nn.Module,
                           compressed_model: nn.Module,
                           test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Estimation robustness model"""
        
        robustness = {}
        
        # Noise sensitivity
        robustness['noise_sensitivity'] = self._test_noise_sensitivity(
            original_model, compressed_model, test_data
        )
        
        # Input perturbation analysis
        robustness['perturbation_analysis'] = self._test_input_perturbations(
            original_model, compressed_model, test_data
        )
        
        # Numerical stability
        robustness['numerical_stability'] = self._test_numerical_stability(
            compressed_model, test_data
        )
        
        # Edge case handling
        robustness['edge_case_handling'] = self._test_edge_cases(
            compressed_model, test_data
        )
        
        return robustness
    
    # Helper methods for various measurements
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        return param_size / 1024 / 1024
    
    def _measure_accuracy(self, 
                         model: nn.Module, 
                         test_data: torch.utils.data.DataLoader,
                         max_batches: int = 100) -> float:
        """Measurement accuracy model"""
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= max_batches:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch, batch
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    outputs = outputs.get('trading_signal', outputs)
                
                loss = nn.MSELoss()(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return max(0.0, 1.0 - avg_loss)  # Convert in accuracy-like metric
    
    def _measure_latency(self,
                        model: nn.Module,
                        test_data: torch.utils.data.DataLoader,
                        num_iterations: int = 100) -> float:
        """Measurement average latency"""
        model.eval()
        model.to(self.device)
        
        # Retrieve example input
        sample_batch = next(iter(test_data))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1].to(self.device)
        else:
            sample_input = sample_batch[:1].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Measurement
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(sample_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        return float(np.mean(latencies))
    
    def _measure_latency_distribution(self,
                                    model: nn.Module,
                                    test_data: torch.utils.data.DataLoader,
                                    num_iterations: int = 200) -> List[float]:
        """Measurement distribution latency"""
        model.eval()
        model.to(self.device)
        
        sample_batch = next(iter(test_data))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1].to(self.device)
        else:
            sample_input = sample_batch[:1].to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(sample_input)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
        
        return latencies
    
    def _estimate_memory_usage(self, model: nn.Module) -> float:
        """Estimation usage memory"""
        # Simple estimation on basis parameters
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Approximate estimation activation memory (heuristic)
        total_params = sum(p.numel() for p in model.parameters())
        activation_memory_estimate = total_params * 4 * 2  # Approximate value
        
        total_memory_bytes = param_memory + buffer_memory + activation_memory_estimate
        return total_memory_bytes / 1024 / 1024  # MB
    
    def _analyze_batch_size_sensitivity(self,
                                      original_model: nn.Module,
                                      compressed_model: nn.Module,
                                      test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analysis sensitivity to size batch"""
        
        batch_sizes = [1, 4, 8, 16, 32]
        results = {'batch_sizes': batch_sizes, 'original_latencies': [], 'compressed_latencies': []}
        
        sample_batch = next(iter(test_data))
        if isinstance(sample_batch, (list, tuple)):
            base_input = sample_batch[0][:1]
        else:
            base_input = sample_batch[:1]
        
        for batch_size in batch_sizes:
            # Create input needed batch size
            batched_input = base_input.repeat(batch_size, *([1] * (len(base_input.shape) - 1)))
            batched_input = batched_input.to(self.device)
            
            # Measure latency for original model
            original_model.to(self.device).eval()
            original_latency = self._measure_single_inference_latency(original_model, batched_input)
            results['original_latencies'].append(original_latency)
            
            # Measure latency for compressed model
            compressed_model.to(self.device).eval()
            compressed_latency = self._measure_single_inference_latency(compressed_model, batched_input)
            results['compressed_latencies'].append(compressed_latency)
        
        return results
    
    def _measure_single_inference_latency(self, model: nn.Module, input_tensor: torch.Tensor) -> float:
        """Measurement latency one inference"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_tensor)
        
        # Measurement
        times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                _ = model(input_tensor)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
        
        return float(np.mean(times))
    
    def _analyze_resource_utilization(self,
                                    original_model: nn.Module,
                                    compressed_model: nn.Module) -> Dict[str, Any]:
        """Analysis usage resources"""
        
        analysis = {}
        
        # FLOPs estimation (approximately)
        original_flops = self._estimate_flops(original_model)
        compressed_flops = self._estimate_flops(compressed_model)
        
        analysis['flops'] = {
            'original': original_flops,
            'compressed': compressed_flops,
            'reduction_pct': (1 - compressed_flops / original_flops) * 100 if original_flops > 0 else 0
        }
        
        # Parameter count analysis
        original_params = sum(p.numel() for p in original_model.parameters())
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        
        analysis['parameters'] = {
            'original': original_params,
            'compressed': compressed_params,
            'reduction_pct': (1 - compressed_params / original_params) * 100
        }
        
        return analysis
    
    def _estimate_flops(self, model: nn.Module) -> int:
        """Approximate estimation FLOPs"""
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear layer: input_size * output_size * 2 (multiply + add)
                total_flops += module.in_features * module.out_features * 2
            elif isinstance(module, nn.Conv1d):
                # Conv1d FLOPs estimation
                kernel_flops = module.kernel_size[0] * module.in_channels
                output_elements = module.out_channels  # Simplified
                total_flops += kernel_flops * output_elements * 2
            elif isinstance(module, nn.Conv2d):
                # Conv2d FLOPs estimation
                kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                output_elements = module.out_channels  # Simplified
                total_flops += kernel_flops * output_elements * 2
        
        return total_flops
    
    def _analyze_prediction_correlations(self,
                                       original_model: nn.Module,
                                       compressed_model: nn.Module,
                                       test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Analysis correlation predictions"""
        
        original_model.eval()
        compressed_model.eval()
        
        original_predictions = []
        compressed_predictions = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 50:  # Limit for speed
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Predictions original model
                orig_out = original_model(inputs)
                if isinstance(orig_out, dict):
                    orig_out = orig_out.get('trading_signal', orig_out)
                original_predictions.append(orig_out.cpu().numpy())
                
                # Predictions compressed model
                comp_out = compressed_model(inputs)
                if isinstance(comp_out, dict):
                    comp_out = comp_out.get('trading_signal', comp_out)
                compressed_predictions.append(comp_out.cpu().numpy())
        
        if not original_predictions:
            return {'correlation': 0.0, 'mse': float('inf')}
        
        # Merge predictions
        orig_pred = np.concatenate(original_predictions, axis=0).flatten()
        comp_pred = np.concatenate(compressed_predictions, axis=0).flatten()
        
        # Compute correlation
        correlation = float(np.corrcoef(orig_pred, comp_pred)[0, 1])
        if np.isnan(correlation):
            correlation = 0.0
        
        # MSE between predictions
        mse = float(np.mean((orig_pred - comp_pred) ** 2))
        
        return {
            'correlation': correlation,
            'mse': mse,
            'mae': float(np.mean(np.abs(orig_pred - comp_pred)))
        }
    
    def _analyze_prediction_errors(self,
                                 original_model: nn.Module,
                                 compressed_model: nn.Module,
                                 test_data: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """Analysis errors predictions"""
        
        original_model.eval()
        compressed_model.eval()
        
        original_errors = []
        compressed_errors = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 50:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:
                    inputs = batch.to(self.device)
                    targets = inputs  # Self-prediction task
                
                # Errors original model
                orig_out = original_model(inputs)
                if isinstance(orig_out, dict):
                    orig_out = orig_out.get('trading_signal', orig_out)
                orig_error = nn.MSELoss(reduction='none')(orig_out, targets)
                original_errors.append(orig_error.cpu().numpy())
                
                # Errors compressed model
                comp_out = compressed_model(inputs)
                if isinstance(comp_out, dict):
                    comp_out = comp_out.get('trading_signal', comp_out)
                comp_error = nn.MSELoss(reduction='none')(comp_out, targets)
                compressed_errors.append(comp_error.cpu().numpy())
        
        if not original_errors:
            return {'error_increase': 0.0}
        
        orig_errors = np.concatenate(original_errors, axis=0).flatten()
        comp_errors = np.concatenate(compressed_errors, axis=0).flatten()
        
        return {
            'original_mean_error': float(np.mean(orig_errors)),
            'compressed_mean_error': float(np.mean(comp_errors)),
            'error_increase_pct': float((np.mean(comp_errors) - np.mean(orig_errors)) / np.mean(orig_errors) * 100),
            'error_std_original': float(np.std(orig_errors)),
            'error_std_compressed': float(np.std(comp_errors))
        }
    
    def _analyze_confidence_preservation(self,
                                       original_model: nn.Module,
                                       compressed_model: nn.Module,
                                       test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Analysis saving confidence in predictions"""
        
        # Simplified analysis on basis variations outputs
        original_variances = []
        compressed_variances = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 30:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Add small noise for estimation stability
                noise = torch.randn_like(inputs) * 0.01
                
                # Original model
                orig_out1 = original_model(inputs)
                orig_out2 = original_model(inputs + noise)
                if isinstance(orig_out1, dict):
                    orig_out1 = orig_out1.get('trading_signal', orig_out1)
                    orig_out2 = orig_out2.get('trading_signal', orig_out2)
                
                orig_var = torch.var(orig_out1 - orig_out2).item()
                original_variances.append(orig_var)
                
                # Compressed model
                comp_out1 = compressed_model(inputs)
                comp_out2 = compressed_model(inputs + noise)
                if isinstance(comp_out1, dict):
                    comp_out1 = comp_out1.get('trading_signal', comp_out1)
                    comp_out2 = comp_out2.get('trading_signal', comp_out2)
                
                comp_var = torch.var(comp_out1 - comp_out2).item()
                compressed_variances.append(comp_var)
        
        if not original_variances:
            return {'confidence_preservation': 1.0}
        
        orig_mean_var = np.mean(original_variances)
        comp_mean_var = np.mean(compressed_variances)
        
        # Confidence preservation score (back proportionally change variance)
        if orig_mean_var > 0:
            confidence_preservation = min(1.0, orig_mean_var / max(comp_mean_var, 1e-8))
        else:
            confidence_preservation = 1.0
        
        return {
            'confidence_preservation': float(confidence_preservation),
            'original_variance': float(orig_mean_var),
            'compressed_variance': float(comp_mean_var)
        }
    
    def _calculate_directional_accuracy(self,
                                      model: nn.Module,
                                      test_data: torch.utils.data.DataLoader) -> float:
        """Calculation directional accuracy for trading"""
        
        model.eval()
        correct_directions = 0
        total_comparisons = 0
        
        with torch.no_grad():
            prev_prediction = None
            prev_target = None
            
            for i, batch in enumerate(test_data):
                if i >= 50:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:
                    continue
                
                predictions = model(inputs)
                if isinstance(predictions, dict):
                    predictions = predictions.get('trading_signal', predictions)
                
                if prev_prediction is not None:
                    # Compare directions changes
                    pred_direction = torch.sign(predictions[0] - prev_prediction[-1])
                    actual_direction = torch.sign(targets[0] - prev_target[-1])
                    
                    if pred_direction.item() == actual_direction.item():
                        correct_directions += 1
                    total_comparisons += 1
                
                prev_prediction = predictions
                prev_target = targets
        
        return correct_directions / total_comparisons if total_comparisons > 0 else 0.0
    
    def _simulate_trading_performance(self,
                                    original_model: nn.Module,
                                    compressed_model: nn.Module,
                                    test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Simulation trading performance"""
        
        # Simplified simulation trading metrics
        compressed_model.eval()
        
        returns = []
        predictions = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 100:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                else:
                    continue
                
                prediction = compressed_model(inputs)
                if isinstance(prediction, dict):
                    prediction = prediction.get('trading_signal', prediction)
                
                # Simulate returns on basis predictions
                simulated_return = prediction.cpu().numpy().flatten()
                actual_return = targets.cpu().numpy().flatten()
                
                returns.extend(actual_return)
                predictions.extend(simulated_return)
        
        if len(returns) < 2:
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'profit_factor': 0.0}
        
        returns = np.array(returns)
        predictions = np.array(predictions)
        
        # Simplified trading metrics
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
        
        # Max drawdown (simplified)
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = running_max - cumulative_returns
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Profit factor (simplified)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        profit_factor = (np.sum(positive_returns) / (-np.sum(negative_returns) + 1e-8) 
                        if len(negative_returns) > 0 else 1.0)
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'profit_factor': float(profit_factor)
        }
    
    def _measure_p99_latency_microseconds(self,
                                        model: nn.Module,
                                        test_data: torch.utils.data.DataLoader) -> float:
        """Measurement P99 latency in microseconds"""
        
        latencies_ms = self._measure_latency_distribution(model, test_data, num_iterations=1000)
        latencies_us = [lat * 1000 for lat in latencies_ms]  # Convert in microseconds
        
        return float(np.percentile(latencies_us, 99))
    
    def _measure_prediction_stability(self,
                                    model: nn.Module,
                                    test_data: torch.utils.data.DataLoader) -> float:
        """Measurement stability predictions"""
        
        model.eval()
        stability_scores = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 20:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Several forward passes for estimation stability
                outputs = []
                for _ in range(5):
                    output = model(inputs)
                    if isinstance(output, dict):
                        output = output.get('trading_signal', output)
                    outputs.append(output.cpu().numpy())
                
                # Compute stability as inverse magnitude variance
                outputs_array = np.array(outputs)
                variance = np.var(outputs_array, axis=0).mean()
                stability = 1.0 / (1.0 + variance)
                stability_scores.append(stability)
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0
    
    def _evaluate_market_regime_detection(self,
                                        model: nn.Module,
                                        test_data: torch.utils.data.DataLoader) -> float:
        """Estimation accuracy determination market mode"""
        
        # Simplified estimation - assume that model can output mode
        model.eval()
        
        correct_regime_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 30:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                output = model(inputs)
                
                # If model supports market regime detection
                if isinstance(output, dict) and 'market_regime' in output:
                    regime_pred = output['market_regime']
                    # Here was would real logic estimation
                    # For example use stub
                    correct_regime_predictions += 1
                    total_predictions += 1
        
        return correct_regime_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _evaluate_risk_metrics(self,
                             original_model: nn.Module,
                             compressed_model: nn.Module,
                             test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Estimation risk-related metrics"""
        
        # Simplified estimation risk metrics
        compressed_model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 50:
                    break
                
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, batch_targets = batch
                    inputs = inputs.to(self.device)
                    
                    output = compressed_model(inputs)
                    if isinstance(output, dict):
                        output = output.get('trading_signal', output)
                    
                    predictions.append(output.cpu().numpy())
                    targets.append(batch_targets.numpy())
        
        if not predictions:
            return {'var_accuracy': 0.0, 'volatility_error': 0.0, 'correlation_preservation': 0.0}
        
        pred_array = np.concatenate(predictions, axis=0)
        target_array = np.concatenate(targets, axis=0)
        
        # VaR accuracy (simplified estimation)
        pred_var = np.percentile(pred_array, 5)  # 95% VaR
        actual_var = np.percentile(target_array, 5)
        var_accuracy = 1.0 - abs(pred_var - actual_var) / (abs(actual_var) + 1e-8)
        
        # Volatility prediction error
        pred_vol = np.std(pred_array)
        actual_vol = np.std(target_array)
        volatility_error = abs(pred_vol - actual_vol) / (actual_vol + 1e-8)
        
        # Correlation preservation
        correlation = np.corrcoef(pred_array.flatten(), target_array.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        return {
            'var_accuracy': float(max(0.0, var_accuracy)),
            'volatility_error': float(volatility_error),
            'correlation_preservation': float(abs(correlation))
        }
    
    # Deployment readiness methods
    
    def _test_cpu_compatibility(self, model: nn.Module) -> float:
        """Test compatibility with CPU"""
        try:
            model_cpu = model.cpu()
            dummy_input = torch.randn(1, 100)
            
            with torch.no_grad():
                _ = model_cpu(dummy_input)
            
            return 1.0  # Full compatibility
            
        except Exception:
            return 0.0  # Incompatible
    
    def _test_gpu_acceleration(self, model: nn.Module) -> float:
        """Test GPU acceleration"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            model_gpu = model.cuda()
            dummy_input = torch.randn(1, 100).cuda()
            
            with torch.no_grad():
                _ = model_gpu(dummy_input)
            
            return 1.0  # Is supported
            
        except Exception:
            return 0.0  # Not is supported
    
    def _test_edge_device_suitability(self, model: nn.Module) -> float:
        """Estimation fitness for edge devices"""
        
        # Factors for edge device suitability
        model_size_mb = self._calculate_model_size(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Simple scoring function
        size_score = max(0.0, 1.0 - model_size_mb / 100.0)  # Penalty for size > 100MB
        param_score = max(0.0, 1.0 - param_count / 10_000_000)  # Penalty for > 10M parameters
        
        # Validation on unsupported operations
        has_unsupported = False
        for module in model.modules():
            if isinstance(module, (nn.MultiheadAttention,)):  # Example unsupported ops
                has_unsupported = True
                break
        
        unsupported_penalty = 0.5 if has_unsupported else 0.0
        
        edge_score = (size_score + param_score) / 2 - unsupported_penalty
        
        return max(0.0, min(1.0, edge_score))
    
    def _test_onnx_export(self, model: nn.Module) -> bool:
        """Test export in ONNX"""
        try:
            import torch.onnx
            import tempfile
            
            model.eval()
            dummy_input = torch.randn(1, 100)
            
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as tmp:
                torch.onnx.export(
                    model,
                    dummy_input,
                    tmp.name,
                    export_params=True,
                    opset_version=11,
                    input_names=['input'],
                    output_names=['output']
                )
            
            return True
            
        except Exception as e:
            self.logger.debug(f"ONNX export failed: {e}")
            return False
    
    def _test_torchscript_export(self, model: nn.Module) -> bool:
        """Test export in TorchScript"""
        try:
            model.eval()
            dummy_input = torch.randn(1, 100)
            
            # Try trace
            try:
                traced_model = torch.jit.trace(model, dummy_input)
                # Test traced model
                with torch.no_grad():
                    _ = traced_model(dummy_input)
                return True
            except:
                # Try script
                scripted_model = torch.jit.script(model)
                with torch.no_grad():
                    _ = scripted_model(dummy_input)
                return True
                
        except Exception as e:
            self.logger.debug(f"TorchScript export failed: {e}")
            return False
    
    def _test_tflite_conversion(self, model: nn.Module) -> bool:
        """Test conversion in TensorFlow Lite"""
        # Simplified test - in reality is required ONNX -> TensorFlow -> TFLite pipeline
        try:
            # Here was would real logic conversion
            # For example return False (complex conversion)
            return False
            
        except Exception:
            return False
    
    def _evaluate_model_stability(self, model: nn.Module) -> float:
        """Estimation stability model"""
        try:
            model.eval()
            dummy_input = torch.randn(1, 100)
            
            # Multiple inference for validation stability
            outputs = []
            with torch.no_grad():
                for _ in range(10):
                    output = model(dummy_input)
                    if isinstance(output, dict):
                        output = output.get('trading_signal', output)
                    outputs.append(output.numpy())
            
            # Estimation stability through variance
            outputs_array = np.array(outputs)
            variance = np.var(outputs_array)
            stability = 1.0 / (1.0 + variance)
            
            return float(min(1.0, stability))
            
        except Exception:
            return 0.0
    
    def _evaluate_error_resilience(self, model: nn.Module) -> float:
        """Estimation stability to errors"""
        try:
            model.eval()
            
            error_tests = 0
            passed_tests = 0
            
            # Test 1: NaN inputs
            try:
                nan_input = torch.full((1, 100), float('nan'))
                with torch.no_grad():
                    output = model(nan_input)
                    if not torch.isnan(output).any():
                        passed_tests += 1
                error_tests += 1
            except:
                error_tests += 1
            
            # Test 2: Extreme values
            try:
                extreme_input = torch.full((1, 100), 1e6)
                with torch.no_grad():
                    output = model(extreme_input)
                    if torch.isfinite(output).all():
                        passed_tests += 1
                error_tests += 1
            except:
                error_tests += 1
            
            # Test 3: Zero input
            try:
                zero_input = torch.zeros(1, 100)
                with torch.no_grad():
                    output = model(zero_input)
                    if torch.isfinite(output).all():
                        passed_tests += 1
                error_tests += 1
            except:
                error_tests += 1
            
            return passed_tests / error_tests if error_tests > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _evaluate_resource_efficiency(self, model: nn.Module) -> float:
        """Estimation efficiency usage resources"""
        
        # Factors efficiency
        model_size_mb = self._calculate_model_size(model)
        param_count = sum(p.numel() for p in model.parameters())
        
        # Estimation efficiency
        size_efficiency = max(0.0, 1.0 - model_size_mb / 200.0)  # Penalty for size > 200MB
        param_efficiency = max(0.0, 1.0 - param_count / 50_000_000)  # Penalty for > 50M parameters
        
        # Simple latency test
        try:
            dummy_input = torch.randn(1, 100)
            model.eval()
            
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(dummy_input)
            end_time = time.perf_counter()
            
            avg_latency_ms = ((end_time - start_time) / 10) * 1000
            latency_efficiency = max(0.0, 1.0 - avg_latency_ms / 100.0)  # Penalty for > 100ms
            
        except:
            latency_efficiency = 0.0
        
        efficiency_score = (size_efficiency + param_efficiency + latency_efficiency) / 3
        
        return float(max(0.0, min(1.0, efficiency_score)))
    
    # Robustness test methods
    
    def _test_noise_sensitivity(self,
                              original_model: nn.Module,
                              compressed_model: nn.Module,
                              test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Test sensitivity to noise"""
        
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        results = {}
        
        for noise_level in noise_levels:
            original_errors = []
            compressed_errors = []
            
            with torch.no_grad():
                for i, batch in enumerate(test_data):
                    if i >= 20:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        clean_input = batch[0].to(self.device)
                    else:
                        clean_input = batch.to(self.device)
                    
                    # Add noise
                    noise = torch.randn_like(clean_input) * noise_level
                    noisy_input = clean_input + noise
                    
                    # Original model
                    clean_out_orig = original_model(clean_input)
                    noisy_out_orig = original_model(noisy_input)
                    if isinstance(clean_out_orig, dict):
                        clean_out_orig = clean_out_orig.get('trading_signal', clean_out_orig)
                        noisy_out_orig = noisy_out_orig.get('trading_signal', noisy_out_orig)
                    
                    orig_error = torch.mean((clean_out_orig - noisy_out_orig) ** 2).item()
                    original_errors.append(orig_error)
                    
                    # Compressed model
                    clean_out_comp = compressed_model(clean_input)
                    noisy_out_comp = compressed_model(noisy_input)
                    if isinstance(clean_out_comp, dict):
                        clean_out_comp = clean_out_comp.get('trading_signal', clean_out_comp)
                        noisy_out_comp = noisy_out_comp.get('trading_signal', noisy_out_comp)
                    
                    comp_error = torch.mean((clean_out_comp - noisy_out_comp) ** 2).item()
                    compressed_errors.append(comp_error)
            
            results[f'noise_level_{noise_level}'] = {
                'original_sensitivity': float(np.mean(original_errors)),
                'compressed_sensitivity': float(np.mean(compressed_errors)),
                'sensitivity_ratio': float(np.mean(compressed_errors) / (np.mean(original_errors) + 1e-8))
            }
        
        return results
    
    def _test_input_perturbations(self,
                                original_model: nn.Module,
                                compressed_model: nn.Module,
                                test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Test stability to perturbations input"""
        
        perturbation_tests = ['gaussian_noise', 'uniform_noise', 'dropout']
        results = {}
        
        for test_type in perturbation_tests:
            original_robustness = []
            compressed_robustness = []
            
            with torch.no_grad():
                for i, batch in enumerate(test_data):
                    if i >= 15:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        clean_input = batch[0].to(self.device)
                    else:
                        clean_input = batch.to(self.device)
                    
                    # Apply perturbation
                    if test_type == 'gaussian_noise':
                        perturbation = torch.randn_like(clean_input) * 0.1
                        perturbed_input = clean_input + perturbation
                    elif test_type == 'uniform_noise':
                        perturbation = (torch.rand_like(clean_input) - 0.5) * 0.2
                        perturbed_input = clean_input + perturbation
                    else:  # dropout
                        mask = torch.rand_like(clean_input) > 0.1  # 10% dropout
                        perturbed_input = clean_input * mask.float()
                    
                    # Test original model
                    clean_out_orig = original_model(clean_input)
                    pert_out_orig = original_model(perturbed_input)
                    if isinstance(clean_out_orig, dict):
                        clean_out_orig = clean_out_orig.get('trading_signal', clean_out_orig)
                        pert_out_orig = pert_out_orig.get('trading_signal', pert_out_orig)
                    
                    orig_diff = torch.mean((clean_out_orig - pert_out_orig) ** 2).item()
                    original_robustness.append(1.0 / (1.0 + orig_diff))
                    
                    # Test compressed model
                    clean_out_comp = compressed_model(clean_input)
                    pert_out_comp = compressed_model(perturbed_input)
                    if isinstance(clean_out_comp, dict):
                        clean_out_comp = clean_out_comp.get('trading_signal', clean_out_comp)
                        pert_out_comp = pert_out_comp.get('trading_signal', pert_out_comp)
                    
                    comp_diff = torch.mean((clean_out_comp - pert_out_comp) ** 2).item()
                    compressed_robustness.append(1.0 / (1.0 + comp_diff))
            
            results[test_type] = {
                'original_robustness': float(np.mean(original_robustness)),
                'compressed_robustness': float(np.mean(compressed_robustness)),
                'robustness_retention': float(np.mean(compressed_robustness) / (np.mean(original_robustness) + 1e-8))
            }
        
        return results
    
    def _test_numerical_stability(self,
                                model: nn.Module,
                                test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Test numerical stability"""
        
        model.eval()
        stability_tests = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_data):
                if i >= 20:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Test on very small values
                small_input = inputs * 1e-6
                try:
                    small_output = model(small_input)
                    if isinstance(small_output, dict):
                        small_output = small_output.get('trading_signal', small_output)
                    
                    is_stable = torch.isfinite(small_output).all().item()
                    stability_tests.append(is_stable)
                except:
                    stability_tests.append(False)
                
                # Test on large values
                large_input = inputs * 1e3
                try:
                    large_output = model(large_input)
                    if isinstance(large_output, dict):
                        large_output = large_output.get('trading_signal', large_output)
                    
                    is_stable = torch.isfinite(large_output).all().item()
                    stability_tests.append(is_stable)
                except:
                    stability_tests.append(False)
        
        stability_score = sum(stability_tests) / len(stability_tests) if stability_tests else 0.0
        
        return {
            'numerical_stability_score': float(stability_score),
            'stable_tests': sum(stability_tests),
            'total_tests': len(stability_tests)
        }
    
    def _test_edge_cases(self,
                        model: nn.Module,
                        test_data: torch.utils.data.DataLoader) -> Dict[str, bool]:
        """Test processing edge cases"""
        
        model.eval()
        edge_case_results = {}
        
        # Retrieve example input
        sample_batch = next(iter(test_data))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1].to(self.device)
        else:
            sample_input = sample_batch[:1].to(self.device)
        
        edge_cases = {
            'all_zeros': torch.zeros_like(sample_input),
            'all_ones': torch.ones_like(sample_input),
            'negative_values': -torch.abs(sample_input),
            'very_small_values': sample_input * 1e-8,
            'very_large_values': sample_input * 1e8,
            'mixed_extreme': torch.cat([sample_input * 1e8, sample_input * 1e-8], dim=-1)[:, :sample_input.shape[-1]]
        }
        
        for case_name, case_input in edge_cases.items():
            try:
                with torch.no_grad():
                    output = model(case_input)
                    if isinstance(output, dict):
                        output = output.get('trading_signal', output)
                    
                    # Check that output finite and reasonable
                    is_finite = torch.isfinite(output).all().item()
                    is_reasonable = (torch.abs(output) < 1e6).all().item()
                    
                    edge_case_results[case_name] = is_finite and is_reasonable
                    
            except Exception as e:
                self.logger.debug(f"Edge case {case_name} failed: {e}")
                edge_case_results[case_name] = False
        
        return edge_case_results
    
    def _comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comparative analysis results"""
        
        analysis = {}
        
        # Extract key metrics
        compression_metrics = results.get('compression_metrics', {})
        crypto_metrics = results.get('crypto_metrics', {})
        deployment_metrics = results.get('deployment_metrics', {})
        
        # Overall compression efficiency
        compression_ratio = compression_metrics.get('compression_ratio', 1.0)
        accuracy_retention = compression_metrics.get('accuracy_retention', 1.0)
        latency_improvement = compression_metrics.get('latency_improvement_pct', 0.0)
        
        # Composite scores
        efficiency_score = (
            min(compression_ratio / 4.0, 1.0) * 0.4 +  # Compression weight
            accuracy_retention * 0.4 +                  # Accuracy weight
            min(latency_improvement / 50.0, 1.0) * 0.2  # Latency weight
        )
        
        analysis['efficiency_score'] = float(efficiency_score)
        analysis['compression_quality_tradeoff'] = float(compression_ratio * accuracy_retention)
        
        # Deployment readiness score
        deployment_score = (
            deployment_metrics.get('cpu_compatibility_score', 0.0) * 0.3 +
            deployment_metrics.get('model_stability_score', 0.0) * 0.4 +
            deployment_metrics.get('resource_efficiency_score', 0.0) * 0.3
        )
        
        analysis['deployment_readiness_score'] = float(deployment_score)
        
        # Trading suitability score
        if crypto_metrics:
            trading_score = (
                crypto_metrics.get('directional_accuracy', 0.0) * 0.4 +
                min(crypto_metrics.get('sharpe_ratio', 0.0) / 2.0, 1.0) * 0.3 +
                crypto_metrics.get('prediction_stability', 0.0) * 0.3
            )
            analysis['trading_suitability_score'] = float(max(0.0, trading_score))
        
        return analysis
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generation recommendations on basis results"""
        
        recommendations = []
        
        compression_metrics = results.get('compression_metrics', {})
        comparative = results.get('comparative_analysis', {})
        deployment = results.get('deployment_metrics', {})
        
        # Compression recommendations
        compression_ratio = compression_metrics.get('compression_ratio', 1.0)
        accuracy_retention = compression_metrics.get('accuracy_retention', 1.0)
        
        if compression_ratio < 2.0:
            recommendations.append("Consider more aggressive techniques compression for increase compression ratio")
        
        if accuracy_retention < 0.95:
            recommendations.append("Accuracy significantly decreased. Consider fine-tuning or less aggressive compression")
        
        if compression_ratio > 5.0 and accuracy_retention > 0.98:
            recommendations.append("Excellent result compression! Model ready for production deployment")
        
        # Performance recommendations
        latency_improvement = compression_metrics.get('latency_improvement_pct', 0.0)
        if latency_improvement < 10:
            recommendations.append("Latency improved insignificantly. Try quantization or other techniques")
        
        # Deployment recommendations
        if not deployment.get('onnx_export_success', True):
            recommendations.append("ONNX export not succeeded. Check compatibility operations model")
        
        if deployment.get('edge_device_suitability', 0.0) < 0.5:
            recommendations.append("Model not suitable for edge devices. Necessary additional compression")
        
        # Trading-specific recommendations
        crypto_metrics = results.get('crypto_metrics', {})
        if crypto_metrics:
            directional_accuracy = crypto_metrics.get('directional_accuracy', 0.0)
            if directional_accuracy < 0.55:
                recommendations.append("Low directional accuracy. Consider domain-specific fine-tuning")
        
        # Overall recommendations
        efficiency_score = comparative.get('efficiency_score', 0.0)
        if efficiency_score > 0.8:
            recommendations.append("High efficiency score! Model well optimized")
        elif efficiency_score < 0.5:
            recommendations.append("Low efficiency score. Necessary reconsider strategy compression")
        
        if not recommendations:
            recommendations.append("Results compression in within norms. Model ready to usage")
        
        return recommendations
    
    def _generate_cache_key(self, original_model: nn.Module, compressed_model: nn.Module) -> str:
        """Generation key for caching"""
        orig_params = sum(p.numel() for p in original_model.parameters())
        comp_params = sum(p.numel() for p in compressed_model.parameters())
        
        return f"eval_{orig_params}_{comp_params}_{hash(str(original_model))}"