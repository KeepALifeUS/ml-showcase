"""
ML XGBoost - Enterprise XGBoost Implementation
===============================================

Advanced XGBoost package for time series prediction with
SHAP/LIME interpretability, feature engineering, and production-ready features.

Key Features:
- Time series optimized XGBoost configuration
- Automated hyperparameter tuning with Optuna
- SHAP values and model interpretability
- Feature importance analysis
- Cross-validation for time series
- FastAPI integration for model serving
- Real-time prediction capabilities

Modules:
--------
models: Core XGBoost models and predictors
features: Feature engineering for tree models
tuning: Hyperparameter optimization
validation: Cross-validation and backtesting
api: REST API endpoints
utils: Utility functions and helpers

Example Usage:
--------------
>>> from ml_xgboost import XGBoostTimeSeriesModel, FeatureEngineer
>>>
>>> # Create and configure model
>>> model = XGBoostTimeSeriesModel(
...     objective='reg:squarederror',
...     enable_categorical=True,
...     early_stopping_rounds=50
... )
>>>
>>> # Engineer features
>>> engineer = FeatureEngineer()
>>> X_features = engineer.create_features(price_data)
>>>
>>> # Train model
>>> model.fit(X_features, target, enable_shap=True)
>>>
>>> # Generate predictions
>>> predictions = model.predict(X_test)
>>>
>>> # Get feature importance and SHAP values
>>> importance = model.get_feature_importance()
>>> shap_values = model.get_shap_values(X_test)
"""

__version__ = "1.0.0"
__author__ = "ML XGBoost Contributors"
__email__ = ""
__license__ = "MIT"

import sys
import warnings
from typing import Dict, List, Optional, Any

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Core model exports
from .models.xgb_time_series import (
    XGBoostTimeSeriesModel,
    XGBoostRegressor,
    XGBoostClassifier,
    XGBoostRanker,
)

# Feature engineering exports  
from .features.feature_engineer import (
    FeatureEngineer,
    TechnicalIndicators,
    StatisticalFeatures,
    LaggingFeatures,
)

# Hyperparameter tuning exports
from .tuning.hyperparameter_tuner import (
    HyperparameterTuner,
    OptunaTuner,
    BayesianOptimizer,
    GridSearchTuner,
)

# Validation and backtesting exports
from .validation.cross_validator import (
    TimeSeriesCrossValidator,
    WalkForwardValidator,
    ModelValidator,
    BacktestEngine,
)

# API exports
from .api.prediction_service import (
    PredictionService,
    ModelServer,
    BatchPredictor,
)

# Utility exports
from .utils.model_utils import (
    ModelSaver,
    ModelLoader,
    ModelMetrics,
    PerformanceAnalyzer,
)

from .utils.interpretability import (
    SHAPExplainer,
    FeatureImportanceAnalyzer,
    ModelInterpreter,
    ExplanationGenerator,
)

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    
    # Core models
    "XGBoostTimeSeriesModel",
    "XGBoostRegressor", 
    "XGBoostClassifier",
    "XGBoostRanker",
    
    # Feature engineering
    "FeatureEngineer",
    "TechnicalIndicators",
    "StatisticalFeatures", 
    "LaggingFeatures",
    
    # Hyperparameter tuning
    "HyperparameterTuner",
    "OptunaTuner",
    "BayesianOptimizer",
    "GridSearchTuner",
    
    # Validation
    "TimeSeriesCrossValidator",
    "WalkForwardValidator", 
    "ModelValidator",
    "BacktestEngine",
    
    # API services
    "PredictionService",
    "ModelServer",
    "BatchPredictor",
    
    # Utilities
    "ModelSaver",
    "ModelLoader", 
    "ModelMetrics",
    "PerformanceAnalyzer",
    
    # Interpretability
    "SHAPExplainer",
    "FeatureImportanceAnalyzer",
    "ModelInterpreter", 
    "ExplanationGenerator",
    
    # Package functions
    "get_package_info",
    "check_dependencies",
    "setup_logging",
]


def get_package_info() -> Dict[str, Any]:
    """
    Get package information and metadata.
    
    Returns:
        Dict containing package info including version, dependencies, etc.
    """
    info = {
        "name": "ml-xgboost",
        "version": __version__,
        "description": "Enterprise XGBoost with SHAP/LIME interpretability and feature engineering",
        "author": __author__,
        "license": __license__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "features": [
            "Time series optimized XGBoost",
            "Automated hyperparameter tuning", 
            "SHAP interpretability",
            "Feature engineering pipeline",
            "Cross-validation framework",
            "FastAPI integration",
            "Real-time predictions",
            "Trading signal generation",
        ],
        "enterprise_ready": True,
    }
    
    return info


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dict mapping dependency names to availability status.
    """
    dependencies = {
        "xgboost": False,
        "sklearn": False,
        "pandas": False,
        "numpy": False,
        "optuna": False,
        "shap": False,
        "fastapi": False,
        "uvicorn": False,
        "loguru": False,
        "rich": False,
    }
    
    # Check each dependency
    for dep_name in dependencies:
        try:
            if dep_name == "sklearn":
                import sklearn
                dependencies[dep_name] = True
            else:
                __import__(dep_name)
                dependencies[dep_name] = True
        except ImportError:
            dependencies[dep_name] = False
    
    return dependencies


def setup_logging(level: str = "INFO", format_type: str = "rich") -> None:
    """
    Setup logging configuration for the package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('rich', 'simple', 'json')
    """
    from loguru import logger
    import sys
    
    # Remove default handler
    logger.remove()
    
    if format_type == "rich":
        logger.add(
            sys.stdout,
            level=level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                   "<level>{message}</level>",
            colorize=True
        )
    elif format_type == "json":
        logger.add(
            sys.stdout,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            serialize=True
        )
    else:  # simple
        logger.add(
            sys.stdout,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            colorize=False
        )
    
    logger.info(f"🚀 ML-XGBoost package initialized (v{__version__})")


# Initialize logging on import
try:
    setup_logging(level="INFO", format_type="rich")
except Exception:
    # Fallback to basic logging if rich is not available
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


# Dependency check on import
if __name__ != "__main__":
    missing_deps = [
        dep for dep, available in check_dependencies().items() 
        if not available
    ]
    
    if missing_deps:
        warnings.warn(
            f"Missing dependencies: {', '.join(missing_deps)}. "
            f"Run: pip install -e .[dev] to install all dependencies.",
            ImportWarning
        )