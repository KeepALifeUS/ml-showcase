"""
Data Preprocessing Module for ML Common

High-performance data preprocessing utilities implementing .
Consolidates preprocessing logic from multiple ML packages.

Available modules:
- normalization: Scaling, normalization, standardization
- feature_engineering: Feature creation, selection, transformation
- data_cleaning: Outlier detection, missing value handling
"""

from .normalization import (
 # Core normalization functions
 normalize_data,
 standardize_data,
 robust_scale_data,
 minmax_scale_data,
 quantile_uniform_transform,

 # Main preprocessor class
 DataNormalizer,
 NormalizationConfig,

 # Utility functions
 inverse_transform,
 check_normalization_params
)

# TODO: Re-enable when feature_engineering.py and data_cleaning.py are available
# from .feature_engineering import (
# # Feature creation
# create_lag_features,
# create_rolling_features,
# create_technical_features,
# create_polynomial_features,
# create_interaction_features,
#
# # Feature selection
# select_features_variance,
# select_features_correlation,
# select_features_mutual_info,
#
# # Main feature engineer class
# FeatureEngineer,
# FeatureConfig
# )
#
# from .data_cleaning import (
# # Outlier detection
# detect_outliers_iqr,
# detect_outliers_zscore,
# detect_outliers_isolation_forest,
# handle_outliers,
#
# # Missing value handling
# handle_missing_values,
# impute_forward_fill,
# impute_backward_fill,
# impute_interpolate,
#
# # Data validation
# validate_data_quality,
# DataQualityReport,
#
# # Main data cleaner class
# DataCleaner,
# CleaningConfig
# )

__all__ = [
 # Normalization
 "normalize_data",
 "standardize_data",
 "robust_scale_data",
 "minmax_scale_data",
 "quantile_uniform_transform",
 "DataNormalizer",
 "NormalizationConfig",
 "inverse_transform",
 "check_normalization_params",

 # Feature engineering - TODO: Re-enable when modules are available
 # "create_lag_features",
 # "create_rolling_features",
 # "create_technical_features",
 # "create_polynomial_features",
 # "create_interaction_features",
 # "select_features_variance",
 # "select_features_correlation",
 # "select_features_mutual_info",
 # "FeatureEngineer",
 # "FeatureConfig",
 #
 # # Data cleaning
 # "detect_outliers_iqr",
 # "detect_outliers_zscore",
 # "detect_outliers_isolation_forest",
 # "handle_outliers",
 # "handle_missing_values",
 # "impute_forward_fill",
 # "impute_backward_fill",
 # "impute_interpolate",
 # "validate_data_quality",
 # "DataQualityReport",
 # "DataCleaner",
 # "CleaningConfig"
]