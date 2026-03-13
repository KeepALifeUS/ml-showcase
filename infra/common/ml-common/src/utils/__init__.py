"""
Utilities Module for ML Common

High-performance utility functions implementing .
Common mathematical and data processing utilities used across ML packages.

Available modules:
- math_utils: Mathematical utility functions
- time_series: Time series processing utilities
- data_loader: Data loading and validation utilities
"""

from .math_utils import (
 # Core math functions
 safe_divide,
 safe_log,
 safe_sqrt,
 clip_values,

 # Array operations
 rolling_window,
 exponential_smoothing,
 weighted_average,

 # Statistical functions
 detect_outliers,
 robust_mean,
 robust_std,
 calculate_correlation,

 # Signal processing
 smooth_signal,
 detrend_signal,
 normalize_signal,

 # Performance utilities
 batch_process,
 parallel_apply
)

# TODO: Re-enable when time_series.py is available
# from .time_series import (
# # Time series analysis
# detect_seasonality,
# decompose_time_series,
# calculate_autocorrelation,
#
# # Trend analysis
# detect_trend,
# calculate_trend_strength,
# fit_trend_line,
#
# # Change point detection
# detect_change_points,
# segment_time_series,
#
# # Forecasting utilities
# simple_forecast,
# exponential_smoothing_forecast,
#
# # Main time series class
# TimeSeriesAnalyzer,
# TimeSeriesConfig
# )
#
# # TODO: Re-enable when data_loader.py is available
# from .data_loader import (
# # Data validation
# validate_dataframe,
# check_data_quality,
# DataQualityReport,
#
# # Data loading
# load_csv_data,
# load_json_data,
# save_data,
#
# # Data processing
# resample_data,
# align_data,
# merge_datasets,
#
# # Main data loader class
# DataLoader,
# DataLoaderConfig
# )

__all__ = [
 # Math utilities
 "safe_divide",
 "safe_log",
 "safe_sqrt",
 "clip_values",
 "rolling_window",
 "exponential_smoothing",
 "weighted_average",
 "detect_outliers",
 "robust_mean",
 "robust_std",
 "calculate_correlation",
 "smooth_signal",
 "detrend_signal",
 "normalize_signal",
 "batch_process",
 "parallel_apply",

 # Time series utilities - TODO: Re-enable when available
 # "detect_seasonality",
 # "decompose_time_series",
 # "calculate_autocorrelation",
 # "detect_trend",
 # "calculate_trend_strength",
 # "fit_trend_line",
 # "detect_change_points",
 # "segment_time_series",
 # "simple_forecast",
 # "exponential_smoothing_forecast",
 # "TimeSeriesAnalyzer",
 # "TimeSeriesConfig",
 #
 # # Data loading utilities
 # "validate_dataframe",
 # "check_data_quality",
 # "DataQualityReport",
 # "load_csv_data",
 # "load_json_data",
 # "save_data",
 # "resample_data",
 # "align_data",
 # "merge_datasets",
 # "DataLoader",
 # "DataLoaderConfig"
]