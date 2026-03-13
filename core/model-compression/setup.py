#!/usr/bin/env python3
"""
Setup script for ML-Framework ML Model Compression System.
This setup.py provides compatibility with older Python tooling while
the main configuration is in pyproject.toml.
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're using Python 3.9+
if sys.version_info < (3, 9):
    sys.exit("ML-Framework ML Model Compression requires Python 3.9 or higher")

# Get the long description from the README file
HERE = Path(__file__).parent.resolve()
long_description = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

# Load requirements from requirements.txt
def load_requirements(filename: str = "requirements.txt") -> list[str]:
    """Load requirements from file."""
    req_file = HERE / filename
    if not req_file.exists():
        return []
    
    requirements = []
    with open(req_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
    return requirements

# Load version from __init__.py or environment
def get_version() -> str:
    """Get version from environment or default."""
    version = os.getenv("ML-Framework_VERSION", "1.0.0")
    
    # Try to load from package __init__.py if it exists
    init_file = HERE / "src" / "__init__.py"
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
    
    return version

# Package configuration
setup(
    name="ml-framework-ml-model-compression",
    version=get_version(),
    author="ML-Framework Development Team",
    author_email="dev@ml-framework.ai",
    description="Comprehensive ML Model Compression System for Crypto Trading Bot v5.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-framework/crypto-trading-bot-v5",
    project_urls={
        "Bug Reports": "https://github.com/ml-framework/crypto-trading-bot-v5/issues",
        "Source": "https://github.com/ml-framework/crypto-trading-bot-v5",
        "Documentation": "https://ml-framework.readthedocs.io/en/latest/packages/ml-model-compression/",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.9",
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "dev": load_requirements("requirements-dev.txt"),
        "tensorrt": ["tensorrt>=8.6.0", "pycuda>=2022.2"],
        "openvino": ["openvino>=2023.0.0", "openvino-dev>=2023.0.0"],
        "edge": ["tflite-runtime>=2.13.0", "onnxruntime-gpu>=1.15.0"],
        "visualization": ["graphviz>=0.20.0", "netron>=6.9.0", "torchinfo>=1.8.0"],
    },
    entry_points={
        "console_scripts": [
            "ml-framework-compress=src.cli:main",
            "ml-framework-quantize=src.quantization.quantizer:cli_main",
            "ml-framework-prune=src.pruning.structured_pruning:cli_main", 
            "ml-framework-distill=src.distillation.knowledge_distiller:cli_main",
            "ml-framework-deploy-edge=src.deployment.edge_deployer:cli_main",
            "ml-framework-analyze-model=src.utils.model_analyzer:cli_main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords=[
        "machine-learning", "model-compression", "quantization", "pruning",
        "knowledge-distillation", "crypto-trading", "edge-deployment", "pytorch",
        "tensorflow", "high-frequency-trading", "hft", "ml-framework", "enterprise"
    ],
    zip_safe=False,
)