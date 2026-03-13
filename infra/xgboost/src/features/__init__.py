"""
Feature Engineering Module for XGBoost Models
=============================================

Advanced feature engineering specifically optimized for tree-based models
and time series data with enterprise patterns.

Key Components:
--------------
- FeatureEngineer: Main feature engineering pipeline
- TechnicalIndicators: Technical analysis features for financial
- StatisticalFeatures: Statistical and mathematical features  
- LaggingFeatures: Time-based lagged features
- InteractionFeatures: Feature interactions and combinations
- VolatilityFeatures: Volatility and risk-based features

Tree Model Optimizations:
------------------------
- Categorical feature encoding optimized for tree splits
- Feature binning and discretization
- Interaction feature creation
- Missing value indicators
- Outlier-resistant transformations
"""

from .feature_engineer import (
    FeatureEngineer,
    FeatureConfig,
    FeatureMetadata,
)

from .technical_indicators import (
    TechnicalIndicators,
    TechnicalConfig,
    IndicatorType,
)

from .statistical_features import (
    StatisticalFeatures,
    StatisticalConfig,
    StatFeatureType,
)

from .lagging_features import (
    LaggingFeatures,
    LagConfig,
    LagType,
)

from .interaction_features import (
    InteractionFeatures,
    InteractionConfig,
    InteractionType,
)

from .volatility_features import (
    VolatilityFeatures,
    VolatilityConfig,
    VolatilityType,
)

__all__ = [
    # Main feature engineer
    "FeatureEngineer",
    "FeatureConfig", 
    "FeatureMetadata",
    
    # Technical indicators
    "TechnicalIndicators",
    "TechnicalConfig",
    "IndicatorType",
    
    # Statistical features
    "StatisticalFeatures", 
    "StatisticalConfig",
    "StatFeatureType",
    
    # Lagging features
    "LaggingFeatures",
    "LagConfig",
    "LagType",
    
    # Interaction features
    "InteractionFeatures",
    "InteractionConfig", 
    "InteractionType",
    
    # Volatility features
    "VolatilityFeatures",
    "VolatilityConfig",
    "VolatilityType",
]