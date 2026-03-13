"""
Module dynamic quantization for crypto trading models.
Optimized for minimum latency in real-time inference.

High-frequency trading optimization patterns
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import time
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, default_dynamic_qconfig
import numpy as np
from enum import Enum

from .quantizer import BaseQuantizer, PrecisionLevel, CryptoModelQuantizer

logger = logging.getLogger(__name__)

class DynamicQuantizationMode(Enum):
    """Modes dynamic quantization"""
    AGGRESSIVE = "aggressive"    # Maximum compression
    BALANCED = "balanced"       # Balance compression/accuracy
    CONSERVATIVE = "conservative"  # Minimum loss accuracy

class LatencyOptimizer:
    """Optimizer latency for HFT scenarios"""
    
    def __init__(self, target_latency_us: float = 100.0):
        """
        Args:
            target_latency_us: Target latency in microseconds
        """
        self.target_latency_us = target_latency_us
        self.benchmark_results = {}
        self.logger = logging.getLogger(f"{__name__}.LatencyOptimizer")
    
    def benchmark_model(self, 
                       model: nn.Module, 
                       input_shape: Tuple[int, ...], 
                       num_iterations: int = 1000) -> Dict[str, float]:
        """
        Benchmark latency model
        
        Args:
            model: Model for testing
            input_shape: Size input data
            num_iterations: Number iterations for averaging
        
        Returns:
            Statistics latency
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Warmup model
        dummy_input = torch.randn(1, *input_shape).to(device)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measurement latency
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(dummy_input)
                end_time = time.perf_counter()
                
                latency_us = (end_time - start_time) * 1_000_000
                latencies.append(latency_us)
        
        # Statistics
        stats = {
            "mean_latency_us": np.mean(latencies),
            "median_latency_us": np.median(latencies),
            "p95_latency_us": np.percentile(latencies, 95),
            "p99_latency_us": np.percentile(latencies, 99),
            "std_latency_us": np.std(latencies),
            "min_latency_us": np.min(latencies),
            "max_latency_us": np.max(latencies)
        }
        
        self.benchmark_results = stats
        
        self.logger.info(f"Benchmark latency: "
                        f"average={stats['mean_latency_us']:.1f}μs, "
                        f"p95={stats['p95_latency_us']:.1f}μs")
        
        return stats
    
    def meets_latency_target(self) -> bool:
        """Validation compliance target latency"""
        if not self.benchmark_results:
            return False
        
        p95_latency = self.benchmark_results.get("p95_latency_us", float('inf'))
        return p95_latency <= self.target_latency_us

class DynamicQuantizer(CryptoModelQuantizer):
    """
    Specialized dynamic quantizer for crypto trading models
    with optimization for microsecond latency
    """
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 mode: DynamicQuantizationMode = DynamicQuantizationMode.BALANCED,
                 target_latency_us: float = 100.0):
        """
        Args:
            precision: Level accuracy quantization
            mode: Mode quantization (aggressive/balanced/conservative)
            target_latency_us: Target latency in microseconds
        """
        super().__init__(precision)
        self.mode = mode
        self.latency_optimizer = LatencyOptimizer(target_latency_us)
        self.quantization_config = self._get_dynamic_config()
        
    def _get_dynamic_config(self) -> Dict[str, Any]:
        """Retrieval configuration dynamic quantization"""
        base_config = {
            "dtype": torch.qint8 if self.precision == PrecisionLevel.INT8 else torch.qint8,
            "reduce_range": False
        }
        
        if self.mode == DynamicQuantizationMode.AGGRESSIVE:
            # Quantize maximum layers for maximum compression
            base_config["qconfig_spec"] = {
                nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU
            }
        elif self.mode == DynamicQuantizationMode.BALANCED:
            # Quantize main layers
            base_config["qconfig_spec"] = {nn.Linear, nn.Conv1d, nn.Conv2d}
        else:  # CONSERVATIVE
            # Quantize only Linear layers
            base_config["qconfig_spec"] = {nn.Linear}
            
        return base_config
    
    def quantize_for_hft(self, 
                        model: nn.Module,
                        input_shape: Tuple[int, ...],
                        calibration_data: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Special quantization for High-Frequency Trading
        with guarantee latency
        
        Args:
            model: Original model
            input_shape: Size input data
            calibration_data: Data for validation accuracy
            
        Returns:
            Quantized model with optimal latency
        """
        self.logger.info(f"Begin HFT quantization in mode {self.mode.value}")
        
        # Original benchmark
        original_stats = self.latency_optimizer.benchmark_model(model, input_shape)
        
        # Try different configuration quantization
        best_model = None
        best_config = None
        best_latency = float('inf')
        
        configs_to_try = self._generate_quantization_configs()
        
        for config_name, config in configs_to_try.items():
            try:
                # Quantization with current configuration
                quantized_model = self._apply_dynamic_quantization(model, config)
                
                # Benchmark quantized model
                quantized_stats = self.latency_optimizer.benchmark_model(
                    quantized_model, input_shape
                )
                
                current_latency = quantized_stats["p95_latency_us"]
                
                # Validation improvements latency and compliance goals
                if (current_latency < best_latency and 
                    current_latency <= self.latency_optimizer.target_latency_us):
                    
                    # Additional validation accuracy if exists data
                    if calibration_data is not None:
                        accuracy_ok = self._validate_accuracy(
                            model, quantized_model, calibration_data
                        )
                        if not accuracy_ok:
                            self.logger.warning(f"Configuration {config_name} not passed check accuracy")
                            continue
                    
                    best_model = quantized_model
                    best_config = config_name
                    best_latency = current_latency
                    
                self.logger.info(f"Configuration {config_name}: "
                               f"latency {current_latency:.1f}μs")
                
            except Exception as e:
                self.logger.warning(f"Error with configuration {config_name}: {e}")
                continue
        
        if best_model is None:
            self.logger.warning("Not succeeded find suitable configuration quantization")
            return model
        
        # Additional optimization for best model
        optimized_model = self._apply_hft_optimizations(best_model)
        
        # Final benchmark
        final_stats = self.latency_optimizer.benchmark_model(
            optimized_model, input_shape
        )
        
        improvement = (original_stats["p95_latency_us"] - 
                      final_stats["p95_latency_us"]) / original_stats["p95_latency_us"] * 100
        
        self.logger.info(f"HFT quantization completed. Best configuration: {best_config}")
        self.logger.info(f"Improvement latency: {improvement:.1f}%")
        
        # Save statistics
        self.compression_stats.update({
            "hft_optimization": {
                "original_p95_latency_us": original_stats["p95_latency_us"],
                "final_p95_latency_us": final_stats["p95_latency_us"],
                "latency_improvement_pct": improvement,
                "best_config": best_config,
                "meets_target": final_stats["p95_latency_us"] <= self.latency_optimizer.target_latency_us
            }
        })
        
        return optimized_model
    
    def _generate_quantization_configs(self) -> Dict[str, Dict[str, Any]]:
        """Generation various configurations for testing"""
        configs = {}
        
        base_layers = [nn.Linear]
        extended_layers = [nn.Linear, nn.Conv1d, nn.Conv2d]
        aggressive_layers = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU]
        
        # Base configuration
        configs["basic"] = {
            "qconfig_spec": set(base_layers),
            "dtype": torch.qint8
        }
        
        # Extended configuration
        configs["extended"] = {
            "qconfig_spec": set(extended_layers),
            "dtype": torch.qint8
        }
        
        # Aggressive configuration
        if self.mode == DynamicQuantizationMode.AGGRESSIVE:
            configs["aggressive"] = {
                "qconfig_spec": set(aggressive_layers),
                "dtype": torch.qint8
            }
        
        # Configuration with different settings reduce_range
        for name, base_config in list(configs.items()):
            reduced_config = base_config.copy()
            reduced_config["reduce_range"] = True
            configs[f"{name}_reduced"] = reduced_config
            
        return configs
    
    def _apply_dynamic_quantization(self, 
                                  model: nn.Module, 
                                  config: Dict[str, Any]) -> nn.Module:
        """Application dynamic quantization with configuration"""
        quantized_model = quantize_dynamic(
            model=model,
            qconfig_spec=config["qconfig_spec"],
            dtype=config["dtype"],
            mapping=None,
            inplace=False
        )
        
        return quantized_model
    
    def _validate_accuracy(self, 
                          original_model: nn.Module,
                          quantized_model: nn.Module,
                          validation_data: torch.Tensor,
                          threshold: float = 0.95) -> bool:
        """
        Validation accuracy quantized model
        
        Args:
            original_model: Original model
            quantized_model: Quantized model
            validation_data: Data for validation
            threshold: Minimum threshold compliance
            
        Returns:
            True if accuracy acceptable
        """
        try:
            original_model.eval()
            quantized_model.eval()
            
            with torch.no_grad():
                original_output = original_model(validation_data)
                quantized_output = quantized_model(validation_data)
                
                # Compute correlation between outputs
                original_flat = original_output.flatten().cpu().numpy()
                quantized_flat = quantized_output.flatten().cpu().numpy()
                
                correlation = np.corrcoef(original_flat, quantized_flat)[0, 1]
                
                # Check average relative error
                relative_error = np.mean(np.abs(original_flat - quantized_flat) / 
                                       (np.abs(original_flat) + 1e-8))
                
                accuracy_ok = (correlation >= threshold and 
                             relative_error <= (1.0 - threshold))
                
                self.logger.debug(f"Validation accuracy: correlation={correlation:.3f}, "
                                f"relative error={relative_error:.3f}")
                
                return accuracy_ok
                
        except Exception as e:
            self.logger.error(f"Error validation accuracy: {e}")
            return False
    
    def _apply_hft_optimizations(self, model: nn.Module) -> nn.Module:
        """Additional optimization for HFT"""
        optimized_model = model
        
        try:
            # 1. Fusing operations
            optimized_model = self._fuse_operations(optimized_model)
            
            # 2. JIT compilation if possibly
            if hasattr(torch, 'jit'):
                try:
                    # Create example input for tracing
                    input_shape = self._get_input_shape(model)
                    dummy_input = torch.randn(1, *input_shape)
                    
                    traced_model = torch.jit.trace(optimized_model, dummy_input)
                    optimized_model = torch.jit.optimize_for_inference(traced_model)
                    
                    self.logger.info("Applied JIT optimization")
                    
                except Exception as e:
                    self.logger.warning(f"JIT optimization not succeeded: {e}")
            
            # 3. Memory layout optimization
            optimized_model = self._optimize_memory_layout(optimized_model)
            
        except Exception as e:
            self.logger.warning(f"Some HFT optimization not succeeded: {e}")
        
        return optimized_model
    
    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimization memory layout for best cache locality"""
        try:
            # Ensure contiguous memory layout for parameters
            for param in model.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
            
            self.logger.debug("Applied optimization memory layout")
            
        except Exception as e:
            self.logger.warning(f"Error optimization memory layout: {e}")
        
        return model
    
    def create_inference_engine(self, 
                               model: nn.Module, 
                               input_shape: Tuple[int, ...]) -> 'HFTInferenceEngine':
        """Creation specialized inference engine for HFT"""
        return HFTInferenceEngine(
            model=model,
            input_shape=input_shape,
            target_latency_us=self.latency_optimizer.target_latency_us
        )

class HFTInferenceEngine:
    """
    High-Frequency Trading Inference Engine
    Optimized for microsecond latency inference
    """
    
    def __init__(self, 
                 model: nn.Module,
                 input_shape: Tuple[int, ...],
                 target_latency_us: float = 100.0):
        """
        Args:
            model: Quantized model
            input_shape: Size input data
            target_latency_us: Target latency
        """
        self.model = model
        self.input_shape = input_shape
        self.target_latency_us = target_latency_us
        
        # Warmup and optimization
        self._warmup_model()
        self._setup_input_cache()
        
        self.logger = logging.getLogger(f"{__name__}.HFTInferenceEngine")
    
    def _warmup_model(self, warmup_iterations: int = 100) -> None:
        """Warmup model for stable latency"""
        self.model.eval()
        dummy_input = torch.randn(1, *self.input_shape)
        
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(dummy_input)
        
        self.logger.info(f"Model warmed up {warmup_iterations} iterations")
    
    def _setup_input_cache(self) -> None:
        """Configuration cache input tensors for reuse"""
        # Pre-allocate input tensor for avoidance memory allocation overhead
        self._input_cache = torch.empty(1, *self.input_shape)
        
    def predict(self, input_data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Fast prediction with minimum latency
        
        Args:
            input_data: Input data
            
        Returns:
            Result predictions
        """
        # Use pre-allocated tensor
        if isinstance(input_data, np.ndarray):
            self._input_cache[0] = torch.from_numpy(input_data)
        else:
            self._input_cache[0] = input_data
        
        with torch.no_grad():
            return self.model(self._input_cache)
    
    def predict_batch(self, 
                     batch_data: Union[np.ndarray, torch.Tensor], 
                     max_batch_size: int = 32) -> torch.Tensor:
        """
        Batch prediction with control latency
        
        Args:
            batch_data: Batch input data
            max_batch_size: Maximum size batch for control latency
            
        Returns:
            Batch results
        """
        if isinstance(batch_data, np.ndarray):
            batch_data = torch.from_numpy(batch_data)
        
        batch_size = batch_data.size(0)
        
        if batch_size <= max_batch_size:
            with torch.no_grad():
                return self.model(batch_data)
        else:
            # Split on smaller batches for control latency
            results = []
            
            for i in range(0, batch_size, max_batch_size):
                batch_slice = batch_data[i:i + max_batch_size]
                with torch.no_grad():
                    batch_result = self.model(batch_slice)
                results.append(batch_result)
            
            return torch.cat(results, dim=0)
    
    def get_latency_stats(self, num_iterations: int = 1000) -> Dict[str, float]:
        """Retrieval statistics latency engine"""
        latency_optimizer = LatencyOptimizer(self.target_latency_us)
        return latency_optimizer.benchmark_model(
            self.model, self.input_shape, num_iterations
        )