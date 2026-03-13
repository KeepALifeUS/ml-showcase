"""
Base module for quantization models in Crypto Trading Bot.
Supports INT8, INT4 and mixed accuracy for optimization deployment.

Edge computing deployment patterns for high-frequency trading
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Tuple
import logging
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_qconfig
import torch.quantization.quantize_fx as quantize_fx
from pathlib import Path

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Types quantization for optimization models"""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "quantization_aware_training"  # Quantization with considering training
    MIXED_PRECISION = "mixed_precision"

class PrecisionLevel(Enum):
    """Levels accuracy for quantization"""
    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"
    BF16 = "bf16"

class BaseQuantizer(ABC):
    """Base class for all quantizers with enterprise patterns"""
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 backend: str = "fbgemm"):
        """
        Args:
            precision: Level accuracy (INT8/INT4/FP16)
            backend: Backend for quantization (fbgemm for CPU, qnnpack for mobile)
        """
        self.precision = precision
        self.backend = backend
        self.config = self._get_quantization_config()
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configuration logging for monitoring process"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _get_quantization_config(self) -> QConfig:
        """Retrieval configuration quantization"""
        if self.precision == PrecisionLevel.INT8:
            return default_qconfig
        elif self.precision == PrecisionLevel.INT4:
            # User configuration for INT4
            return QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8),
                weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8)
            )
        else:
            return default_qconfig
    
    @abstractmethod
    def quantize_model(self, model: nn.Module, **kwargs) -> nn.Module:
        """Abstract method for quantization model"""
        pass
    
    def validate_model(self, model: nn.Module) -> bool:
        """Validation model before quantization"""
        try:
            # Validation on presence unsupported layers
            unsupported_layers = self._find_unsupported_layers(model)
            if unsupported_layers:
                self.logger.warning(f"Found unsupported layers: {unsupported_layers}")
                return False
            
            # Validation size model
            model_size = self._calculate_model_size(model)
            if model_size > 500:  # MB
                self.logger.warning(f"Model too large: {model_size} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validation model: {e}")
            return False
    
    def _find_unsupported_layers(self, model: nn.Module) -> list:
        """Search unsupported for quantization layers"""
        unsupported = []
        unsupported_types = [nn.EmbeddingBag, nn.MultiheadAttention]
        
        for name, module in model.named_modules():
            if any(isinstance(module, unsupported_type) for unsupported_type in unsupported_types):
                unsupported.append(name)
        
        return unsupported
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb

class CryptoModelQuantizer(BaseQuantizer):
    """
    Specialized quantizer for models crypto trading
    with optimization for real-time inference
    """
    
    def __init__(self, 
                 precision: PrecisionLevel = PrecisionLevel.INT8,
                 latency_target: float = 1.0,  # ms
                 accuracy_threshold: float = 0.95):
        """
        Args:
            precision: Level accuracy
            latency_target: Target latency in ms
            accuracy_threshold: Minimum threshold accuracy
        """
        super().__init__(precision)
        self.latency_target = latency_target
        self.accuracy_threshold = accuracy_threshold
        self.compression_stats = {}
    
    def quantize_model(self, 
                      model: nn.Module, 
                      calibration_data: Optional[torch.Tensor] = None,
                      quantization_type: QuantizationType = QuantizationType.DYNAMIC) -> nn.Module:
        """
        Quantization model with considering specifics crypto trading
        
        Args:
            model: PyTorch model for quantization
            calibration_data: Data for calibration (for static quantization)
            quantization_type: Type quantization
            
        Returns:
            Quantized model
        """
        if not self.validate_model(model):
            raise ValueError("Model not passed validation")
        
        original_size = self._calculate_model_size(model)
        self.logger.info(f"Begin quantization model. Original size: {original_size:.2f} MB")
        
        try:
            if quantization_type == QuantizationType.DYNAMIC:
                quantized_model = self._dynamic_quantization(model)
            elif quantization_type == QuantizationType.STATIC:
                quantized_model = self._static_quantization(model, calibration_data)
            elif quantization_type == QuantizationType.QAT:
                quantized_model = self._quantization_aware_training(model)
            else:
                raise ValueError(f"Unsupported type quantization: {quantization_type}")
            
            # Analysis results compression
            compressed_size = self._calculate_model_size(quantized_model)
            compression_ratio = original_size / compressed_size
            
            self.compression_stats = {
                "original_size_mb": original_size,
                "compressed_size_mb": compressed_size,
                "compression_ratio": compression_ratio,
                "precision": self.precision.value,
                "quantization_type": quantization_type.value
            }
            
            self.logger.info(f"Quantization completed. Coefficient compression: {compression_ratio:.2f}x")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error when quantization model: {e}")
            raise
    
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization for fast inference"""
        # Define layers for quantization (usually Linear and Conv)
        layers_to_quantize = {nn.Linear, nn.Conv2d, nn.Conv1d}
        
        if self.precision == PrecisionLevel.INT8:
            dtype = torch.qint8
        elif self.precision == PrecisionLevel.INT4:
            # PyTorch while not supports INT4 natively, emulate through INT8
            dtype = torch.qint8
        else:
            dtype = torch.qint8
        
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=layers_to_quantize,
            dtype=dtype
        )
        
        return quantized_model
    
    def _static_quantization(self, model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """Static quantization with calibration"""
        if calibration_data is None:
            raise ValueError("For static quantization needed data calibration")
        
        # Preparation model to quantization
        model.eval()
        model.qconfig = self.config
        
        # Preparation model
        prepared_model = torch.quantization.prepare(model)
        
        # Calibration on representative data
        self.logger.info("Execute calibration model...")
        with torch.no_grad():
            for batch in self._create_calibration_batches(calibration_data):
                prepared_model(batch)
        
        # Conversion in quantized model
        quantized_model = torch.quantization.convert(prepared_model)
        
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Quantization with considering training (requires fine-tuning)"""
        model.train()
        model.qconfig = self.config
        
        # Preparation to QAT
        prepared_model = torch.quantization.prepare_qat(model)
        
        self.logger.info("Model prepared to QAT. Necessary fine-tuning.")
        
        return prepared_model
    
    def _create_calibration_batches(self, calibration_data: torch.Tensor, batch_size: int = 32):
        """Creation batches for calibration"""
        for i in range(0, len(calibration_data), batch_size):
            yield calibration_data[i:i + batch_size]
    
    def optimize_for_trading(self, model: nn.Module) -> nn.Module:
        """
        Special optimization for tasks crypto trading
        with focus on minimization latency
        """
        # Fusing operations for decrease latency
        fused_model = self._fuse_operations(model)
        
        # Optimization for ONNX Runtime (for production deployment)
        if hasattr(torch, 'jit'):
            try:
                traced_model = torch.jit.trace(fused_model, torch.randn(1, *self._get_input_shape(model)))
                optimized_model = torch.jit.optimize_for_inference(traced_model)
                return optimized_model
            except Exception as e:
                self.logger.warning(f"JIT optimization not succeeded: {e}")
                return fused_model
        
        return fused_model
    
    def _fuse_operations(self, model: nn.Module) -> nn.Module:
        """Fusing operations for acceleration inference"""
        try:
            # Standard fusions: Conv+ReLU, Linear+ReLU
            modules_to_fuse = []
            
            # Search sequences for fusing
            module_list = list(model.named_modules())
            for i, (name, module) in enumerate(module_list[:-1]):
                next_name, next_module = module_list[i + 1]
                
                # Conv2d + ReLU
                if isinstance(module, nn.Conv2d) and isinstance(next_module, nn.ReLU):
                    modules_to_fuse.append([name, next_name])
                
                # Linear + ReLU
                elif isinstance(module, nn.Linear) and isinstance(next_module, nn.ReLU):
                    modules_to_fuse.append([name, next_name])
            
            if modules_to_fuse:
                fused_model = torch.quantization.fuse_modules(model, modules_to_fuse)
                self.logger.info(f"Completed fusing {len(modules_to_fuse)} pairs modules")
                return fused_model
            
        except Exception as e:
            self.logger.warning(f"Fusing operations not succeeded: {e}")
        
        return model
    
    def _get_input_shape(self, model: nn.Module) -> Tuple[int, ...]:
        """Determination size input data model"""
        # Simple heuristic for determination size input
        first_layer = next(iter(model.modules()))
        if isinstance(first_layer, nn.Conv2d):
            # Assume standard size for crypto data
            return (first_layer.in_channels, 64, 64)  # Example for 2D data
        elif isinstance(first_layer, nn.Linear):
            return (first_layer.in_features,)
        else:
            # Default size for temporal series
            return (100,)  # 100 temporal steps
    
    def export_model(self, model: nn.Module, export_path: str, format: str = "onnx") -> bool:
        """
        Export quantized model for deployment
        
        Args:
            model: Quantized model
            export_path: Path for saving
            format: Format export (onnx, torchscript, tflite)
        """
        try:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "onnx":
                self._export_to_onnx(model, export_path)
            elif format.lower() == "torchscript":
                self._export_to_torchscript(model, export_path)
            else:
                raise ValueError(f"Unsupported format export: {format}")
            
            self.logger.info(f"Model exported in {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error export model: {e}")
            return False
    
    def _export_to_onnx(self, model: nn.Module, export_path: Path) -> None:
        """Export in ONNX format for cross-platform deployment"""
        try:
            import torch.onnx
            
            model.eval()
            dummy_input = torch.randn(1, *self._get_input_shape(model))
            
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path.with_suffix('.onnx')),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        except ImportError:
            self.logger.error("ONNX not installed. Set: pip install onnx")
            raise
    
    def _export_to_torchscript(self, model: nn.Module, export_path: Path) -> None:
        """Export in TorchScript for production inference"""
        model.eval()
        dummy_input = torch.randn(1, *self._get_input_shape(model))
        
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(export_path.with_suffix('.pt')))
    
    def get_compression_report(self) -> Dict[str, Any]:
        """Retrieval report about compression model"""
        return {
            "compression_stats": self.compression_stats,
            "config": {
                "precision": self.precision.value,
                "backend": self.backend,
                "latency_target_ms": self.latency_target,
                "accuracy_threshold": self.accuracy_threshold
            },
            "optimization_recommendations": self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> Dict[str, str]:
        """Recommendations by further optimization"""
        recommendations = {}
        
        if self.compression_stats.get("compression_ratio", 0) < 2:
            recommendations["compression"] = "Try more aggressive quantization or pruning"
        
        if self.precision == PrecisionLevel.INT8:
            recommendations["precision"] = "Consider INT4 for larger compression"
        
        recommendations["deployment"] = "Use ONNX Runtime for production inference"
        
        return recommendations