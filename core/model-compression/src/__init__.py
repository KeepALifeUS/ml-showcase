"""
ML-Framework ML Model Compression System

Comprehensive ML model compression for crypto trading applications with 
quantization, pruning, knowledge distillation, and edge deployment capabilities.

Author: ML-Framework Development Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ML-Framework Development Team"
__email__ = "dev@ml-framework.ai"
__license__ = "MIT"

# Core compression techniques
from src.quantization.quantizer import CryptoModelQuantizer
from src.quantization.dynamic_quantization import DynamicQuantizer, HFTInferenceEngine

from src.pruning.structured_pruning import CryptoTradingStructuredPruner
from src.pruning.unstructured_pruning import CryptoTradingUnstructuredPruner

from src.distillation.knowledge_distiller import CryptoKnowledgeDistiller
from src.distillation.teacher_student import TeacherStudentFramework

# Optimization and pipeline
from src.optimization.model_optimizer import CryptoModelOptimizer
from src.optimization.compression_pipeline import CryptoCompressionPipeline

# Evaluation and metrics
from src.evaluation.compression_metrics import CryptoCompressionEvaluator
from src.evaluation.accuracy_validator import CryptoTradingValidator

# Deployment
from src.deployment.edge_deployer import EdgeDeployer

# Utilities
from src.utils.model_analyzer import CryptoModelAnalyzer

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # Quantization
    "CryptoModelQuantizer",
    "DynamicQuantizer", 
    "HFTInferenceEngine",
    
    # Pruning
    "CryptoTradingStructuredPruner",
    "CryptoTradingUnstructuredPruner",
    
    # Knowledge Distillation
    "CryptoKnowledgeDistiller",
    "TeacherStudentFramework",
    
    # Optimization
    "CryptoModelOptimizer",
    "CryptoCompressionPipeline",
    
    # Evaluation
    "CryptoCompressionEvaluator",
    "CryptoTradingValidator",
    
    # Deployment
    "EdgeDeployer",
    
    # Utilities
    "CryptoModelAnalyzer",
]

# Quick access functions
def get_version() -> str:
    """Get the current package version."""
    return __version__

def get_supported_techniques() -> list[str]:
    """Get list of supported compression techniques."""
    return [
        "quantization",
        "pruning", 
        "knowledge_distillation",
        "multi_technique"
    ]

def get_supported_formats() -> list[str]:
    """Get list of supported export formats."""
    return [
        "pytorch",
        "onnx",
        "torchscript", 
        "tensorrt",
        "tensorflow_lite",
        "core_ml",
        "openvino"
    ]

def get_supported_devices() -> list[str]:
    """Get list of supported target devices."""
    return [
        "cpu",
        "cuda",
        "raspberry_pi",
        "jetson_nano", 
        "jetson_xavier",
        "intel_nuc",
        "aws_inferentia",
        "mobile"
    ]

# Package configuration
COMPRESSION_CONFIG = {
    "default_quantization": "int8",
    "default_pruning": "structured",
    "default_distillation": "response_based",
    "target_compression_ratio": 0.8,
    "accuracy_threshold": 0.95,
    "hft_latency_target_ms": 0.1,
}

# Framework requirements
FRAMEWORK_REQUIREMENTS = {
    "pytorch_min_version": "2.0.0",
    "tensorflow_min_version": "2.13.0", 
    "onnx_min_version": "1.14.0",
    "python_min_version": "3.9.0",
}

def check_dependencies() -> dict[str, bool]:
    """Check if required dependencies are available."""
    import importlib.util
    
    dependencies = {
        "torch": False,
        "tensorflow": False,
        "onnx": False,
        "numpy": False,
        "pandas": False,
        "sklearn": False,
    }
    
    for dep in dependencies:
        spec = importlib.util.find_spec(dep)
        dependencies[dep] = spec is not None
        
    return dependencies

def print_system_info():
    """Print system information and dependency status."""
    import sys
    import platform
    
    print(f"ML-Framework ML Model Compression System v{__version__}")
    print(f"Python {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()
    
    deps = check_dependencies()
    print("Dependencies:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    print()
    print("Supported techniques:", ", ".join(get_supported_techniques()))
    print("Supported formats:", ", ".join(get_supported_formats()))
    print("Supported devices:", ", ".join(get_supported_devices()))