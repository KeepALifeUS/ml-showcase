"""
Setup configuration for ML Common package
Python Package Setup
"""

from setuptools import setup, find_packages
import os

# Read long description from README
def read_long_description:
 try:
 with open("README.md", "r", encoding="utf-8") as fh:
 return fh.read
 except FileNotFoundError:
 return "Consolidated ML utilities for Crypto Trading Bot v5.0"

# Read requirements
def read_requirements(filename):
 try:
 with open(filename, "r", encoding="utf-8") as f:
 return [line.strip for line in f if line.strip and not line.startswith("#")]
 except FileNotFoundError:
 return []

# Create package list with proper namespace
def get_packages:
 """
 Create DUAL package structure:
 1. ml_common.* (standard namespace for Week 3+)
 2. Top-level packages (legacy compatibility for Week 1-2)
 """
 base_packages = find_packages(where="src", exclude=["*.egg-info", "__pycache__"])

 # Create ml_common namespace packages
 ml_common_packages = [f"ml_common.{pkg}" for pkg in base_packages]

 # Add root ml_common package
 ml_common_packages.insert(0, "ml_common")

 # Return both for dual compatibility
 return ml_common_packages + base_packages

setup(
 name="ml-framework-ml-common",
 version="1.0.0",
 author="ML-Framework Team",
 author_email="dev@ml-framework.dev",
 description="Consolidated ML utilities for Crypto Trading Bot v5.0 - unified mathematical functions",
 long_description=read_long_description,
 long_description_content_type="text/markdown",
 url="https://github.com/ml-framework/crypto-trading-bot",
 project_urls={
 "Bug Tracker": "https://github.com/ml-framework/crypto-trading-bot/issues",
 "Documentation": "https://ml-framework.dev/docs/ml-common",
 "Source Code": "https://github.com/ml-framework/crypto-trading-bot/tree/main/packages/ml-common",
 },
 packages=get_packages,
 package_dir={
 "": "src",
 "ml_common": "src",
 },
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
 install_requires=[
 "numpy>=1.21.0",
 "pandas>=1.3.0",
 "scikit-learn>=1.0.0",
 "ta-lib>=0.4.0",
 "numba>=0.56.0",
 "scipy>=1.7.0",
 "typing-extensions>=4.0.0",
 ],
 extras_require={
 "dev": [
 "pytest>=7.0.0",
 "pytest-cov>=4.0.0",
 "pytest-benchmark>=4.0.0",
 "black>=22.0.0",
 "flake8>=5.0.0",
 "mypy>=0.991",
 "sphinx>=5.0.0",
 "jupyter>=1.0.0",
 "pre-commit>=2.20.0",
 ],
 "performance": [
 "cupy-cuda11x>=11.0.0;platform_machine=='x86_64'",
 "numexpr>=2.8.0",
 "bottleneck>=1.3.0",
 ],
 "visualization": [
 "matplotlib>=3.5.0",
 "seaborn>=0.11.0",
 "plotly>=5.0.0",
 ],
 "full": [
 "ml-framework-ml-common[dev,performance,visualization]",
 ],
 },
 include_package_data=True,
 package_data={
 "ml_common": ["py.typed"],
 },
 entry_points={
 "console_scripts": [
 "ml-common-benchmark=ml_common.cli:benchmark",
 "ml-common-validate=ml_common.cli:validate",
 ],
 },
 zip_safe=False,
 keywords=[
 "crypto", "trading", "machine-learning", "technical-analysis",
 "indicators", "ml-common", "fintech", "quantitative-finance",
 "numpy", "pandas", "scikit-learn", "ta-lib"
 ],
)