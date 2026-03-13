"""
Core XGBoost Models Module
==========================

Specialized XGBoost implementations optimized for time series prediction
with enterprise patterns for production deployment.

Models:
-------
- XGBoostTimeSeriesModel: Main time series prediction model
- XGBoostRegressor: Regression variant with time series optimizations
- XGBoostClassifier: Classification for trend/signal detection
- XGBoostRanker: Ranking model for portfolio selection

Key Features:
- Time series specific configurations
- Financial market optimizations
- Early stopping and overfitting prevention
- Feature importance tracking
- SHAP integration
- Model persistence and versioning
"""

from .xgb_time_series import (
    XGBoostTimeSeriesModel,
    XGBoostRegressor,
    XGBoostClassifier,
    XGBoostRanker,
)

from .ensemble_models import (
    XGBoostEnsemble,
    AdaptiveXGBoost,
    HierarchicalXGBoost,
)

from .specialized_models import (
    XGBoostVolatilityModel,
    XGBoostTrendModel,
    XGBoostAnomalyModel,
)

__all__ = [
    # Core models
    "XGBoostTimeSeriesModel",
    "XGBoostRegressor",
    "XGBoostClassifier", 
    "XGBoostRanker",
    
    # Ensemble models
    "XGBoostEnsemble",
    "AdaptiveXGBoost",
    "HierarchicalXGBoost",
    
    # Specialized models
    "XGBoostVolatilityModel",
    "XGBoostTrendModel",
    "XGBoostAnomalyModel",
]