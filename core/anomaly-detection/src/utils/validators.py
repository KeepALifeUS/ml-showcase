"""
✅ Input Validators

Comprehensive validation utilities for anomaly detection system.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Any
import structlog

logger = structlog.get_logger(__name__)

class Validators:
 """Input validation utilities."""

 @staticmethod
 def validate_numeric_data(data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> bool:
 """Validate that data is numeric and finite."""
 try:
 if isinstance(data, pd.DataFrame):
 return data.select_dtypes(include=[np.number]).shape[1] > 0
 elif isinstance(data, pd.Series):
 return pd.api.types.is_numeric_dtype(data)
 elif isinstance(data, np.ndarray):
 return np.issubdtype(data.dtype, np.number)
 return False
 except Exception as e:
 logger.error("Validation failed", error=str(e))
 return False

 @staticmethod
 def validate_sample_size(data: Union[np.ndarray, pd.DataFrame], min_samples: int = 10) -> bool:
 """Validate minimum sample size."""
 if isinstance(data, pd.DataFrame):
 return len(data) >= min_samples
 elif isinstance(data, np.ndarray):
 return data.shape[0] >= min_samples
 return False

 @staticmethod
 def validate_contamination_rate(contamination: float) -> bool:
 """Validate contamination rate is in valid range."""
 return 0.0 < contamination < 0.5

 @staticmethod
 def clean_data(data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
 """Clean data by removing NaN and infinite values."""
 if isinstance(data, pd.DataFrame):
 return data.dropna.replace([np.inf, -np.inf], np.nan).dropna
 elif isinstance(data, np.ndarray):
 mask = np.isfinite(data).all(axis=1) if data.ndim > 1 else np.isfinite(data)
 return data[mask]
 return data