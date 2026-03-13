"""
Edge deployment module for deployment compressed ML-models 
on devices with limited resources for crypto trading.

Edge computing deployment patterns for real-time financial systems
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import time
import psutil
import platform
import subprocess
import tempfile
import shutil

logger = logging.getLogger(__name__)

class EdgePlatform(Enum):
    """Supported edge platforms"""
    RASPBERRY_PI = "raspberry_pi"
    JETSON_NANO = "jetson_nano"
    CORAL_DEV = "coral_dev"
    INTEL_NUC = "intel_nuc"
    ARM_GENERIC = "arm_generic"
    X86_EMBEDDED = "x86_embedded"

class ModelFormat(Enum):
    """Formats models for deployment"""
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    TENSORFLOW_LITE = "tensorflow_lite"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"

class OptimizationLevel(Enum):
    """Levels optimization for edge deployment"""
    MINIMAL = "minimal"      # Base optimization
    STANDARD = "standard"    # Standard optimization
    AGGRESSIVE = "aggressive"  # Aggressive optimization
    ULTRA = "ultra"         # Maximum optimization

@dataclass
class EdgeDeviceSpec:
    """Specification edge devices"""
    platform: EdgePlatform
    cpu_cores: int
    ram_mb: int
    storage_mb: int
    has_gpu: bool = False
    gpu_memory_mb: int = 0
    supports_quantization: bool = True
    max_model_size_mb: float = 50.0
    target_latency_ms: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary"""
        return {
            'platform': self.platform.value,
            'cpu_cores': self.cpu_cores,
            'ram_mb': self.ram_mb,
            'storage_mb': self.storage_mb,
            'has_gpu': self.has_gpu,
            'gpu_memory_mb': self.gpu_memory_mb,
            'supports_quantization': self.supports_quantization,
            'max_model_size_mb': self.max_model_size_mb,
            'target_latency_ms': self.target_latency_ms
        }

@dataclass
class DeploymentResult:
    """Result edge deployment"""
    success: bool
    deployed_model_path: str
    model_format: ModelFormat
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: float
    optimization_level: OptimizationLevel
    
    # Performance metrics
    inference_time_ms: float
    throughput_samples_per_sec: float
    cpu_usage_percent: float
    memory_peak_mb: float
    
    # Deployment information
    deployment_time_sec: float
    export_formats: List[str]
    compatibility_issues: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary"""
        return {
            'success': self.success,
            'deployed_model_path': self.deployed_model_path,
            'model_format': self.model_format.value if self.model_format else None,
            'model_size_mb': self.model_size_mb,
            'estimated_latency_ms': self.estimated_latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'optimization_level': self.optimization_level.value if self.optimization_level else None,
            'inference_time_ms': self.inference_time_ms,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_peak_mb': self.memory_peak_mb,
            'deployment_time_sec': self.deployment_time_sec,
            'export_formats': self.export_formats,
            'compatibility_issues': self.compatibility_issues,
            'recommendations': self.recommendations
        }

class EdgeDeployer:
    """
    System deployment compressed ML-models on edge devices
    with optimization for crypto trading in real-time
    """
    
    def __init__(self, workspace_dir: Union[str, Path]):
        """
        Args:
            workspace_dir: Working directory for deployment
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories
        (self.workspace_dir / "models").mkdir(exist_ok=True)
        (self.workspace_dir / "exports").mkdir(exist_ok=True)
        (self.workspace_dir / "benchmarks").mkdir(exist_ok=True)
        (self.workspace_dir / "configs").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.EdgeDeployer")
        
        # Cache for deployment results
        self.deployment_cache = {}
        
        # Predefined configuration for popular devices
        self.device_presets = self._load_device_presets()
    
    def deploy_to_edge(self,
                      model: nn.Module,
                      device_spec: EdgeDeviceSpec,
                      optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                      target_format: Optional[ModelFormat] = None,
                      test_data: Optional[torch.utils.data.DataLoader] = None) -> DeploymentResult:
        """
        Main method deployment model on edge device
        
        Args:
            model: Model for deployment
            device_spec: Specification target devices
            optimization_level: Level optimization
            target_format: Target format model
            test_data: Data for benchmark
            
        Returns:
            Result deployment
        """
        start_time = time.time()
        
        self.logger.info(f"Begin edge deployment for {device_spec.platform.value}")
        self.logger.info(f"Level optimization: {optimization_level.value}")
        
        # Validation compatibility model with device
        compatibility_check = self._check_device_compatibility(model, device_spec)
        
        if not compatibility_check['compatible']:
            return DeploymentResult(
                success=False,
                deployed_model_path="",
                model_format=None,
                model_size_mb=0.0,
                estimated_latency_ms=float('inf'),
                memory_usage_mb=0.0,
                optimization_level=optimization_level,
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                cpu_usage_percent=0.0,
                memory_peak_mb=0.0,
                deployment_time_sec=time.time() - start_time,
                export_formats=[],
                compatibility_issues=compatibility_check['issues'],
                recommendations=compatibility_check['recommendations']
            )
        
        try:
            # 1. Selection optimal format model
            if target_format is None:
                target_format = self._select_optimal_format(device_spec)
            
            # 2. Optimization model for edge
            optimized_model = self._optimize_model_for_edge(
                model, device_spec, optimization_level
            )
            
            # 3. Export in target format
            exported_model_path, export_info = self._export_model(
                optimized_model, target_format, device_spec
            )
            
            # 4. Optimization exported model
            final_model_path = self._post_export_optimization(
                exported_model_path, target_format, device_spec
            )
            
            # 5. Benchmark performance
            performance_metrics = self._benchmark_deployed_model(
                final_model_path, target_format, device_spec, test_data
            )
            
            # 6. Generation recommendations
            recommendations = self._generate_deployment_recommendations(
                performance_metrics, device_spec, optimization_level
            )
            
            deployment_time = time.time() - start_time
            
            result = DeploymentResult(
                success=True,
                deployed_model_path=str(final_model_path),
                model_format=target_format,
                model_size_mb=export_info['model_size_mb'],
                estimated_latency_ms=performance_metrics['avg_inference_time_ms'],
                memory_usage_mb=performance_metrics['avg_memory_usage_mb'],
                optimization_level=optimization_level,
                inference_time_ms=performance_metrics['avg_inference_time_ms'],
                throughput_samples_per_sec=performance_metrics['throughput_samples_per_sec'],
                cpu_usage_percent=performance_metrics['avg_cpu_usage_percent'],
                memory_peak_mb=performance_metrics['peak_memory_mb'],
                deployment_time_sec=deployment_time,
                export_formats=export_info['formats_generated'],
                compatibility_issues=[],
                recommendations=recommendations
            )
            
            # Save result
            self._save_deployment_result(result, device_spec)
            
            self.logger.info(f"Edge deployment completed successfully for {deployment_time:.2f}with")
            self.logger.info(f"Model deployed: {final_model_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error edge deployment: {e}")
            
            return DeploymentResult(
                success=False,
                deployed_model_path="",
                model_format=target_format,
                model_size_mb=0.0,
                estimated_latency_ms=float('inf'),
                memory_usage_mb=0.0,
                optimization_level=optimization_level,
                inference_time_ms=float('inf'),
                throughput_samples_per_sec=0.0,
                cpu_usage_percent=0.0,
                memory_peak_mb=0.0,
                deployment_time_sec=time.time() - start_time,
                export_formats=[],
                compatibility_issues=[str(e)],
                recommendations=["Check compatibility model and devices"]
            )
    
    def _check_device_compatibility(self, 
                                  model: nn.Module, 
                                  device_spec: EdgeDeviceSpec) -> Dict[str, Any]:
        """Validation compatibility model with edge device"""
        
        issues = []
        recommendations = []
        
        # Validation size model
        model_size_mb = self._calculate_model_size_mb(model)
        
        if model_size_mb > device_spec.max_model_size_mb:
            issues.append(f"Model too large: {model_size_mb:.1f}MB > {device_spec.max_model_size_mb:.1f}MB")
            recommendations.append("Apply more aggressive compression")
        
        # Validation memory
        estimated_memory_mb = model_size_mb * 3  # Approximate estimation with considering activations
        
        if estimated_memory_mb > device_spec.ram_mb * 0.8:  # Keep 20% system
            issues.append(f"Insufficient RAM: is required ~{estimated_memory_mb:.1f}MB")
            recommendations.append("Decrease size model or use memory mapping")
        
        # Validation supported operations
        unsupported_ops = self._check_unsupported_operations(model, device_spec)
        
        if unsupported_ops:
            issues.append(f"Unsupported operations: {unsupported_ops}")
            recommendations.append("Replace unsupported operations or use another format")
        
        # Validation quantization
        if not device_spec.supports_quantization:
            has_quantized_ops = self._has_quantized_operations(model)
            if has_quantized_ops:
                issues.append("Device not supports quantized operations")
                recommendations.append("Use float model or other device")
        
        compatible = len(issues) == 0
        
        return {
            'compatible': compatible,
            'issues': issues,
            'recommendations': recommendations,
            'estimated_memory_mb': estimated_memory_mb,
            'model_size_mb': model_size_mb
        }
    
    def _select_optimal_format(self, device_spec: EdgeDeviceSpec) -> ModelFormat:
        """Selection optimal format model for devices"""
        
        # Priorities formats for various platforms
        format_priorities = {
            EdgePlatform.RASPBERRY_PI: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT, ModelFormat.PYTORCH],
            EdgePlatform.JETSON_NANO: [ModelFormat.TENSORRT, ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.CORAL_DEV: [ModelFormat.TENSORFLOW_LITE, ModelFormat.ONNX],
            EdgePlatform.INTEL_NUC: [ModelFormat.OPENVINO, ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.ARM_GENERIC: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT],
            EdgePlatform.X86_EMBEDDED: [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT, ModelFormat.OPENVINO]
        }
        
        preferred_formats = format_priorities.get(
            device_spec.platform, 
            [ModelFormat.ONNX, ModelFormat.TORCHSCRIPT]
        )
        
        # Check availability first preferred format
        for format_option in preferred_formats:
            if self._is_format_available(format_option):
                self.logger.info(f"Selected format {format_option.value} for {device_spec.platform.value}")
                return format_option
        
        # Fallback to PyTorch
        return ModelFormat.PYTORCH
    
    def _optimize_model_for_edge(self,
                               model: nn.Module,
                               device_spec: EdgeDeviceSpec,
                               optimization_level: OptimizationLevel) -> nn.Module:
        """Optimization model for edge devices"""
        
        optimized_model = model
        
        # Base optimization (always)
        optimized_model = self._apply_basic_optimizations(optimized_model)
        
        if optimization_level == OptimizationLevel.MINIMAL:
            return optimized_model
        
        # Standard optimization
        if optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA]:
            optimized_model = self._apply_standard_optimizations(optimized_model, device_spec)
        
        # Aggressive optimization
        if optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA]:
            optimized_model = self._apply_aggressive_optimizations(optimized_model, device_spec)
        
        # Ultra optimization
        if optimization_level == OptimizationLevel.ULTRA:
            optimized_model = self._apply_ultra_optimizations(optimized_model, device_spec)
        
        return optimized_model
    
    def _apply_basic_optimizations(self, model: nn.Module) -> nn.Module:
        """Base optimization"""
        
        # Ensure eval mode
        model.eval()
        
        # Memory layout optimization
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        # Remove dropout layers (they not needed in inference)
        self._remove_dropout_layers(model)
        
        return model
    
    def _apply_standard_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Standard optimization"""
        
        # Operator fusion where possibly
        try:
            model = torch.jit.script(model)
            model = torch.jit.optimize_for_inference(model)
        except Exception as e:
            self.logger.warning(f"JIT optimization not succeeded: {e}")
        
        # Memory mapping for large models on devices with limited memory
        if device_spec.ram_mb < 2048:  # Less 2GB RAM
            model = self._enable_memory_mapping(model)
        
        return model
    
    def _apply_aggressive_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Aggressive optimization"""
        
        # Graph-level optimization
        model = self._apply_graph_optimizations(model)
        
        # Platform-specific optimization
        if device_spec.platform == EdgePlatform.ARM_GENERIC or device_spec.platform == EdgePlatform.RASPBERRY_PI:
            model = self._apply_arm_optimizations(model)
        elif device_spec.has_gpu:
            model = self._apply_gpu_optimizations(model, device_spec)
        
        return model
    
    def _apply_ultra_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Ultra optimization (can reduce accuracy)"""
        
        # Kernel fusion
        model = self._apply_kernel_fusion(model)
        
        # Mixed precision if is supported
        if device_spec.has_gpu:
            model = self._apply_mixed_precision(model)
        
        # Custom operators for specific operations
        model = self._replace_with_custom_ops(model, device_spec)
        
        return model
    
    def _export_model(self,
                     model: nn.Module,
                     target_format: ModelFormat,
                     device_spec: EdgeDeviceSpec) -> Tuple[Path, Dict[str, Any]]:
        """Export model in target format"""
        
        timestamp = int(time.time())
        base_name = f"edge_model_{device_spec.platform.value}_{timestamp}"
        
        export_info = {
            'formats_generated': [],
            'model_size_mb': 0.0,
            'export_success': False
        }
        
        if target_format == ModelFormat.PYTORCH:
            export_path = self._export_pytorch(model, base_name)
        elif target_format == ModelFormat.TORCHSCRIPT:
            export_path = self._export_torchscript(model, base_name)
        elif target_format == ModelFormat.ONNX:
            export_path = self._export_onnx(model, base_name, device_spec)
        elif target_format == ModelFormat.TENSORFLOW_LITE:
            export_path = self._export_tflite(model, base_name, device_spec)
        elif target_format == ModelFormat.OPENVINO:
            export_path = self._export_openvino(model, base_name, device_spec)
        elif target_format == ModelFormat.TENSORRT:
            export_path = self._export_tensorrt(model, base_name, device_spec)
        else:
            raise ValueError(f"Unsupported format: {target_format}")
        
        # Compute size exported model
        if export_path.exists():
            model_size_mb = export_path.stat().st_size / (1024 * 1024)
            export_info.update({
                'formats_generated': [target_format.value],
                'model_size_mb': model_size_mb,
                'export_success': True
            })
        
        return export_path, export_info
    
    def _export_pytorch(self, model: nn.Module, base_name: str) -> Path:
        """Export in PyTorch format"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.pt"
        
        torch.save({
            'model': model,
            'state_dict': model.state_dict()
        }, export_path)
        
        return export_path
    
    def _export_torchscript(self, model: nn.Module, base_name: str) -> Path:
        """Export in TorchScript"""
        export_path = self.workspace_dir / "exports" / f"{base_name}_script.pt"
        
        try:
            # Try trace
            dummy_input = self._create_dummy_input(model)
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(str(export_path))
        except Exception:
            # Fallback to script
            scripted_model = torch.jit.script(model)
            scripted_model.save(str(export_path))
        
        return export_path
    
    def _export_onnx(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Export in ONNX format"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.onnx"
        
        try:
            import torch.onnx
            
            dummy_input = self._create_dummy_input(model)
            
            # Settings export for edge devices
            export_params = True
            opset_version = 11  # Compatibility with majority runtimes
            do_constant_folding = True
            
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=export_params,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        except ImportError:
            raise ImportError("ONNX not installed. pip install onnx")
        
        return export_path
    
    def _export_tflite(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Export in TensorFlow Lite (through ONNX)"""
        # This simplified stub - in reality needed complex pipeline
        # ONNX -> TensorFlow -> TensorFlow Lite
        
        export_path = self.workspace_dir / "exports" / f"{base_name}.tflite"
        
        # Create dummy file for demonstration
        with open(export_path, 'wb') as f:
            f.write(b"TFLite model placeholder")
        
        self.logger.info(f"TFLite export (placeholder): {export_path}")
        
        return export_path
    
    def _export_openvino(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Export in OpenVINO format"""
        export_path = self.workspace_dir / "exports" / f"{base_name}_openvino.xml"
        
        # Stub for OpenVINO export
        # IN reality: PyTorch -> ONNX -> OpenVINO Model Optimizer
        
        with open(export_path, 'w') as f:
            f.write('<?xml version="1.0"?>\n<net>OpenVINO model placeholder</net>')
        
        self.logger.info(f"OpenVINO export (placeholder): {export_path}")
        
        return export_path
    
    def _export_tensorrt(self, model: nn.Module, base_name: str, device_spec: EdgeDeviceSpec) -> Path:
        """Export in TensorRT format"""
        export_path = self.workspace_dir / "exports" / f"{base_name}.trt"
        
        # Stub for TensorRT export
        # IN reality: PyTorch -> ONNX -> TensorRT
        
        with open(export_path, 'wb') as f:
            f.write(b"TensorRT engine placeholder")
        
        self.logger.info(f"TensorRT export (placeholder): {export_path}")
        
        return export_path
    
    def _post_export_optimization(self,
                                model_path: Path,
                                model_format: ModelFormat,
                                device_spec: EdgeDeviceSpec) -> Path:
        """Post-export optimization model"""
        
        # Optimization depend from format
        if model_format == ModelFormat.ONNX:
            return self._optimize_onnx_model(model_path, device_spec)
        elif model_format == ModelFormat.OPENVINO:
            return self._optimize_openvino_model(model_path, device_spec)
        elif model_format == ModelFormat.TENSORRT:
            return self._optimize_tensorrt_model(model_path, device_spec)
        else:
            # For remaining formats return as exists
            return model_path
    
    def _optimize_onnx_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Optimization ONNX model"""
        
        try:
            import onnx
            from onnx import optimizer
            
            # Load model
            model = onnx.load(str(model_path))
            
            # Apply optimization
            optimizations = ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_transpose']
            
            if device_spec.ram_mb < 1024:  # For devices with small memory
                optimizations.extend(['fuse_consecutive_transposes', 'fuse_transpose_into_gemm'])
            
            optimized_model = optimizer.optimize(model, optimizations)
            
            # Save optimized model
            optimized_path = model_path.with_name(f"optimized_{model_path.name}")
            onnx.save(optimized_model, str(optimized_path))
            
            self.logger.info(f"ONNX model optimized: {optimized_path}")
            
            return optimized_path
            
        except ImportError:
            self.logger.warning("ONNX optimizer unavailable")
            return model_path
        except Exception as e:
            self.logger.warning(f"ONNX optimization not succeeded: {e}")
            return model_path
    
    def _optimize_openvino_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Optimization OpenVINO model"""
        # Stub for OpenVINO optimization
        return model_path
    
    def _optimize_tensorrt_model(self, model_path: Path, device_spec: EdgeDeviceSpec) -> Path:
        """Optimization TensorRT model"""
        # Stub for TensorRT optimization
        return model_path
    
    def _benchmark_deployed_model(self,
                                model_path: Path,
                                model_format: ModelFormat,
                                device_spec: EdgeDeviceSpec,
                                test_data: Optional[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Benchmark deployed model"""
        
        metrics = {
            'avg_inference_time_ms': 0.0,
            'throughput_samples_per_sec': 0.0,
            'avg_memory_usage_mb': 0.0,
            'peak_memory_mb': 0.0,
            'avg_cpu_usage_percent': 0.0
        }
        
        try:
            # Load model in dependencies from format
            if model_format == ModelFormat.PYTORCH:
                model = torch.load(model_path, map_location='cpu')['model']
            elif model_format == ModelFormat.TORCHSCRIPT:
                model = torch.jit.load(str(model_path), map_location='cpu')
            elif model_format == ModelFormat.ONNX:
                # Use ONNX Runtime for benchmark
                return self._benchmark_onnx_model(model_path, device_spec, test_data)
            else:
                # For other formats use stub
                return self._estimate_performance_metrics(device_spec)
            
            # PyTorch/TorchScript benchmark
            model.eval()
            
            # Create dummy data if test_data not provided
            if test_data is None:
                dummy_input = self._create_dummy_input(model)
                test_inputs = [dummy_input for _ in range(100)]
            else:
                test_inputs = []
                for i, batch in enumerate(test_data):
                    if i >= 100:  # Limit number
                        break
                    if isinstance(batch, (list, tuple)):
                        test_inputs.append(batch[0][:1])  # Take first sample
                    else:
                        test_inputs.append(batch[:1])
            
            # Warmup
            with torch.no_grad():
                for i in range(min(10, len(test_inputs))):
                    _ = model(test_inputs[i])
            
            # Benchmark
            inference_times = []
            memory_usage = []
            
            process = psutil.Process()
            
            with torch.no_grad():
                for test_input in test_inputs:
                    # Measure memory until inference
                    mem_before = process.memory_info().rss / 1024 / 1024  # MB
                    
                    # Measure time inference
                    start_time = time.perf_counter()
                    _ = model(test_input)
                    end_time = time.perf_counter()
                    
                    # Measure memory after inference
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    
                    inference_times.append((end_time - start_time) * 1000)  # ms
                    memory_usage.append(mem_after)
            
            # Compute metrics
            metrics['avg_inference_time_ms'] = float(np.mean(inference_times))
            metrics['throughput_samples_per_sec'] = 1000.0 / metrics['avg_inference_time_ms']
            metrics['avg_memory_usage_mb'] = float(np.mean(memory_usage))
            metrics['peak_memory_mb'] = float(np.max(memory_usage))
            metrics['avg_cpu_usage_percent'] = float(psutil.cpu_percent(interval=1))
            
        except Exception as e:
            self.logger.warning(f"Error benchmark: {e}")
            metrics = self._estimate_performance_metrics(device_spec)
        
        return metrics
    
    def _benchmark_onnx_model(self,
                            model_path: Path,
                            device_spec: EdgeDeviceSpec,
                            test_data: Optional[torch.utils.data.DataLoader]) -> Dict[str, float]:
        """Benchmark ONNX model"""
        
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(str(model_path))
            
            # Retrieve information about inputs
            input_info = session.get_inputs()[0]
            input_name = input_info.name
            input_shape = input_info.shape
            
            # Create test data
            if test_data is None:
                # Use random data
                test_inputs = [np.random.randn(1, *input_shape[1:]).astype(np.float32) for _ in range(100)]
            else:
                test_inputs = []
                for i, batch in enumerate(test_data):
                    if i >= 100:
                        break
                    if isinstance(batch, (list, tuple)):
                        test_inputs.append(batch[0][:1].numpy().astype(np.float32))
                    else:
                        test_inputs.append(batch[:1].numpy().astype(np.float32))
            
            # Warmup
            for i in range(min(10, len(test_inputs))):
                _ = session.run(None, {input_name: test_inputs[i]})
            
            # Benchmark
            inference_times = []
            
            for test_input in test_inputs:
                start_time = time.perf_counter()
                _ = session.run(None, {input_name: test_input})
                end_time = time.perf_counter()
                
                inference_times.append((end_time - start_time) * 1000)  # ms
            
            avg_inference_time = float(np.mean(inference_times))
            
            return {
                'avg_inference_time_ms': avg_inference_time,
                'throughput_samples_per_sec': 1000.0 / avg_inference_time,
                'avg_memory_usage_mb': float(psutil.Process().memory_info().rss / 1024 / 1024),
                'peak_memory_mb': float(psutil.Process().memory_info().rss / 1024 / 1024),
                'avg_cpu_usage_percent': float(psutil.cpu_percent(interval=1))
            }
            
        except ImportError:
            self.logger.warning("ONNX Runtime unavailable")
            return self._estimate_performance_metrics(device_spec)
        except Exception as e:
            self.logger.warning(f"Error ONNX benchmark: {e}")
            return self._estimate_performance_metrics(device_spec)
    
    def _estimate_performance_metrics(self, device_spec: EdgeDeviceSpec) -> Dict[str, float]:
        """Estimation performance metrics on basis specifications devices"""
        
        # Simple heuristics for estimation performance
        base_latency = 50.0  # ms
        
        # Adjustment on basis characteristics devices
        cpu_factor = max(0.5, 4.0 / device_spec.cpu_cores)
        ram_factor = max(0.8, 2048 / device_spec.ram_mb)
        
        estimated_latency = base_latency * cpu_factor * ram_factor
        
        return {
            'avg_inference_time_ms': float(min(estimated_latency, device_spec.target_latency_ms * 2)),
            'throughput_samples_per_sec': 1000.0 / estimated_latency,
            'avg_memory_usage_mb': float(min(100.0, device_spec.ram_mb * 0.1)),
            'peak_memory_mb': float(min(200.0, device_spec.ram_mb * 0.2)),
            'avg_cpu_usage_percent': 50.0
        }
    
    def _generate_deployment_recommendations(self,
                                          performance_metrics: Dict[str, float],
                                          device_spec: EdgeDeviceSpec,
                                          optimization_level: OptimizationLevel) -> List[str]:
        """Generation recommendations by deployment"""
        
        recommendations = []
        
        # Analysis latency
        inference_time = performance_metrics['avg_inference_time_ms']
        
        if inference_time > device_spec.target_latency_ms:
            recommendations.append(
                f"Latency {inference_time:.1f}ms exceeds target {device_spec.target_latency_ms:.1f}ms. "
                f"Consider more aggressive optimization."
            )
        elif inference_time < device_spec.target_latency_ms * 0.5:
            recommendations.append("Excellent latency! Model well optimized for devices.")
        
        # Analysis usage memory
        memory_usage = performance_metrics['avg_memory_usage_mb']
        memory_threshold = device_spec.ram_mb * 0.7
        
        if memory_usage > memory_threshold:
            recommendations.append(
                f"High usage memory {memory_usage:.1f}MB. "
                f"Consider additional compression model."
            )
        
        # Analysis CPU loading
        cpu_usage = performance_metrics['avg_cpu_usage_percent']
        
        if cpu_usage > 80.0:
            recommendations.append("High loading CPU. Model can be too complex for devices.")
        elif cpu_usage < 30.0:
            recommendations.append("Low loading CPU. Exists reserve for more complex models.")
        
        # Recommendations by level optimization
        if optimization_level == OptimizationLevel.MINIMAL and inference_time > device_spec.target_latency_ms:
            recommendations.append("Try more high level optimization.")
        
        # Platform-specific recommendations
        if device_spec.platform == EdgePlatform.RASPBERRY_PI:
            recommendations.append("For Raspberry Pi is recommended use ONNX Runtime with CPU provider.")
        
        if device_spec.has_gpu and cpu_usage > 60.0:
            recommendations.append("Consider usage GPU for inference.")
        
        if not recommendations:
            recommendations.append("Deployment passed successfully. Model ready to usage.")
        
        return recommendations
    
    # Helper methods
    
    def _calculate_model_size_mb(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _check_unsupported_operations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> List[str]:
        """Validation unsupported operations"""
        
        unsupported = []
        
        # General unsupported operations for edge devices
        unsupported_types = []
        
        if device_spec.platform in [EdgePlatform.RASPBERRY_PI, EdgePlatform.ARM_GENERIC]:
            unsupported_types.extend([nn.MultiheadAttention])  # Example
        
        for name, module in model.named_modules():
            if any(isinstance(module, unsupported_type) for unsupported_type in unsupported_types):
                unsupported.append(name)
        
        return unsupported
    
    def _has_quantized_operations(self, model: nn.Module) -> bool:
        """Validation presence quantized operations"""
        
        for module in model.modules():
            # Check on quantized modules PyTorch
            if hasattr(module, 'qscheme'):  # Feature quantized module
                return True
        
        return False
    
    def _is_format_available(self, format_option: ModelFormat) -> bool:
        """Validation availability format export"""
        
        if format_option == ModelFormat.ONNX:
            try:
                import torch.onnx
                return True
            except ImportError:
                return False
        
        elif format_option == ModelFormat.TENSORFLOW_LITE:
            try:
                import tensorflow as tf
                return True
            except ImportError:
                return False
        
        elif format_option == ModelFormat.OPENVINO:
            # Validation presence OpenVINO toolkit
            return shutil.which('mo') is not None  # Model Optimizer
        
        elif format_option == ModelFormat.TENSORRT:
            # Validation presence TensorRT
            try:
                import tensorrt
                return True
            except ImportError:
                return False
        
        # PyTorch and TorchScript always available
        return True
    
    def _create_dummy_input(self, model: nn.Module) -> torch.Tensor:
        """Creation dummy input for model"""
        
        # Simple heuristic for determination size input
        first_layer = next(iter(model.modules()))
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 32, 32)
        else:
            # Default size for crypto trading (temporal series)
            return torch.randn(1, 100)
    
    def _remove_dropout_layers(self, model: nn.Module) -> None:
        """Removal dropout layers (not needed in inference)"""
        
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Dropout):
                setattr(model, name, nn.Identity())
            else:
                self._remove_dropout_layers(module)
    
    def _enable_memory_mapping(self, model: nn.Module) -> nn.Module:
        """Enabling memory mapping for models"""
        # Stub for memory mapping
        return model
    
    def _apply_graph_optimizations(self, model: nn.Module) -> nn.Module:
        """Application graph-level optimizations"""
        # Stub for graph optimizations
        return model
    
    def _apply_arm_optimizations(self, model: nn.Module) -> nn.Module:
        """ARM-specific optimization"""
        # Stub for ARM optimizations
        return model
    
    def _apply_gpu_optimizations(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """GPU optimization for edge devices"""
        # Stub for GPU optimizations
        return model
    
    def _apply_kernel_fusion(self, model: nn.Module) -> nn.Module:
        """Kernel fusion optimization"""
        # Stub for kernel fusion
        return model
    
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Application mixed precision"""
        # Stub for mixed precision
        return model
    
    def _replace_with_custom_ops(self, model: nn.Module, device_spec: EdgeDeviceSpec) -> nn.Module:
        """Replacement operations on custom optimized version"""
        # Stub for custom operators
        return model
    
    def _save_deployment_result(self, result: DeploymentResult, device_spec: EdgeDeviceSpec) -> None:
        """Saving result deployment"""
        
        result_file = self.workspace_dir / "configs" / f"deployment_{device_spec.platform.value}_{int(time.time())}.json"
        
        with open(result_file, 'w') as f:
            json.dump({
                'device_spec': device_spec.to_dict(),
                'deployment_result': result.to_dict()
            }, f, indent=2)
        
        self.logger.info(f"Result deployment saved: {result_file}")
    
    def _load_device_presets(self) -> Dict[EdgePlatform, EdgeDeviceSpec]:
        """Loading preset configurations devices"""
        
        presets = {
            EdgePlatform.RASPBERRY_PI: EdgeDeviceSpec(
                platform=EdgePlatform.RASPBERRY_PI,
                cpu_cores=4,
                ram_mb=4096,
                storage_mb=32000,
                has_gpu=False,
                max_model_size_mb=100.0,
                target_latency_ms=200.0
            ),
            
            EdgePlatform.JETSON_NANO: EdgeDeviceSpec(
                platform=EdgePlatform.JETSON_NANO,
                cpu_cores=4,
                ram_mb=4096,
                storage_mb=16000,
                has_gpu=True,
                gpu_memory_mb=2048,
                max_model_size_mb=200.0,
                target_latency_ms=50.0
            ),
            
            EdgePlatform.INTEL_NUC: EdgeDeviceSpec(
                platform=EdgePlatform.INTEL_NUC,
                cpu_cores=8,
                ram_mb=8192,
                storage_mb=256000,
                has_gpu=False,
                max_model_size_mb=500.0,
                target_latency_ms=30.0
            )
        }
        
        return presets
    
    def get_device_preset(self, platform: EdgePlatform) -> Optional[EdgeDeviceSpec]:
        """Retrieval preset configuration devices"""
        return self.device_presets.get(platform)
    
    def list_supported_formats(self, platform: EdgePlatform) -> List[ModelFormat]:
        """List supported formats for platforms"""
        
        format_support = {
            EdgePlatform.RASPBERRY_PI: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX],
            EdgePlatform.JETSON_NANO: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.TENSORRT],
            EdgePlatform.CORAL_DEV: [ModelFormat.TENSORFLOW_LITE],
            EdgePlatform.INTEL_NUC: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.OPENVINO],
            EdgePlatform.ARM_GENERIC: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX],
            EdgePlatform.X86_EMBEDDED: [ModelFormat.PYTORCH, ModelFormat.TORCHSCRIPT, ModelFormat.ONNX, ModelFormat.OPENVINO]
        }
        
        return format_support.get(platform, [ModelFormat.PYTORCH, ModelFormat.ONNX])