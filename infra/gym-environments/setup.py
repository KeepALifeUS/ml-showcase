"""
Setup configuration for ML Gym Environments package
enterprise patterns for crypto trading environments
"""

from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Package requirements
install_requires = [
    # Core ML/RL dependencies
    "gymnasium>=0.26.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    
    # Utilities
    "dataclasses-json>=0.5.0",
    "typing-extensions>=4.0.0",
    
    # Optional ML dependencies (commonly used)
    "torch>=1.11.0",
    "stable-baselines3>=1.6.0",
]

# Development dependencies
dev_requires = [
    "pytest>=6.0.0",
    "pytest-cov>=3.0.0",
    "pytest-asyncio>=0.18.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pre-commit>=2.17.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.17.0",
]

# All extras
extras_require = {
    "dev": dev_requires,
    "docs": docs_requires,
    "all": dev_requires + docs_requires,
}

setup(
    name="ml-gym-environments",
    version="1.0.0",
    author="Crypto Trading Bot v5.0 Team",
    author_email="dev@cryptotradingbot.com",
    description="Enterprise-grade OpenAI Gym environments for cryptocurrency trading with sentiment analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cryptotradingbot/ml-gym-environments",
    project_urls={
        "Bug Tracker": "https://github.com/cryptotradingbot/ml-gym-environments/issues",
        "Documentation": "https://cryptotradingbot.com/docs/ml-gym-environments",
        "Source Code": "https://github.com/cryptotradingbot/ml-gym-environments",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "ml_gym_environments": [
            "data/*.json",
            "data/*.csv",
            "config/*.yaml",
            "config/*.json",
        ]
    },
    entry_points={
        "console_scripts": [
            "ml-gym-test=ml_gym_environments.scripts.test_runner:main",
            "ml-gym-benchmark=ml_gym_environments.scripts.benchmark:main",
        ],
    },
    zip_safe=False,
    keywords=[
        "cryptocurrency", "trading", "reinforcement-learning", 
        "gym", "gymnasium", "sentiment-analysis", "fintech",
        "machine-learning", "ai", "quantitative-finance",
        "algorithmic-trading", "ml-framework", "enterprise"
    ],
    # Platform specific requirements
    extras_require_platform={
        # Windows specific
        ':sys_platform=="win32"': [],
        # macOS specific
        ':sys_platform=="darwin"': [],
        # Linux specific
        ':sys_platform=="linux"': [],
    },
)