"""
📦 Interquartile Range (IQR) Anomaly Detector

Implements IQR-based anomaly detection using quartiles and outlier fences.
Simple and intuitive method that works well for skewed distributions.

Formula:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Lower fence = Q1 - k*IQR
- Upper fence = Q3 + k*IQR

Features:
- Distribution-agnostic detection
- Tunable sensitivity (k parameter)
- Multi-dimensional support
- Real-time processing
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class IQRConfig:
 """Configuration for IQR detector."""
 k_factor: float = 1.5 # Multiplier for IQR (1.5 = standard, 3.0 = more strict)
 window_size: Optional[int] = None
 rolling_window: bool = False
 min_samples: int = 20
 bilateral: bool = True # True = check both directions
 auto_k_factor: bool = False # Automatically tune k
 contamination: float = 0.1
 quantile_method: str = 'linear' # Quantile interpolation method

class IQRDetector:
 """
 Interquartile Range Anomaly Detector.

 Uses quartiles for determining outliers:
 - Q1 (25th percentile)
 - Q3 (75th percentile)
 - IQR = Q3 - Q1
 - Anomalies: values outside [Q1 - k*IQR, Q3 + k*IQR]

 Features:
 - Simple and interpretable
 - Works with any distribution
 - Configurable sensitivity
 - Enterprise monitoring ready
 """

 def __init__(self, config: Optional[IQRConfig] = None):
 """
 Initialize the IQR detector.

 Args:
 config: Detector configuration
 """
 self.config = config or IQRConfig
 self.fitted = False
 self._q1 = None # 25th percentile
 self._q3 = None # 75th percentile
 self._iqr = None # Interquartile range
 self._lower_fence = None # Lower fence
 self._upper_fence = None # Upper fence

 logger.info(
 "IQRDetector initialized",
 k_factor=self.config.k_factor,
 bilateral=self.config.bilateral,
 auto_k_factor=self.config.auto_k_factor
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'IQRDetector':
 """
 Train the detector on historical data.

 Args:
 X: Historical data for training

 Returns:
 self: Trained detector
 """
 try:
 X = self._validate_input(X)

 if len(X) < self.config.min_samples:
 raise ValueError(
 f"Insufficient samples: {len(X)} < {self.config.min_samples}"
 )

 # Compute quartiles
 self._q1 = np.percentile(
 X, 25, axis=0, method=self.config.quantile_method
 )
 self._q3 = np.percentile(
 X, 75, axis=0, method=self.config.quantile_method
 )

 # Interquartile range
 self._iqr = self._q3 - self._q1

 # Protection against division by zero (constant data)
 self._iqr = np.where(self._iqr == 0, np.finfo(float).eps, self._iqr)

 # Automatically tune k-factor
 if self.config.auto_k_factor:
 self._auto_tune_k_factor(X)

 # Compute fences
 self._lower_fence = self._q1 - self.config.k_factor * self._iqr
 self._upper_fence = self._q3 + self.config.k_factor * self._iqr

 self.fitted = True

 logger.info(
 "IQRDetector fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1] if X.ndim > 1 else 1,
 q1=self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 q3=self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3,
 iqr=self._iqr.tolist if isinstance(self._iqr, np.ndarray) else self._iqr,
 k_factor=self.config.k_factor
 )

 return self

 except Exception as e:
 logger.error("Failed to fit IQRDetector", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
 """
 Detect anomalies in data.

 Args:
 X: Data to analyze

 Returns:
 Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before detecting anomalies")

 try:
 X = self._validate_input(X)

 if self.config.rolling_window and self.config.window_size:
 anomaly_labels, anomaly_scores = self._detect_rolling(X)
 else:
 anomaly_scores = self._calculate_iqr_scores(X)
 anomaly_labels = self._classify_anomalies(X)

 logger.debug(
 "IQR anomaly detection completed",
 n_samples=len(X),
 n_anomalies=np.sum(anomaly_labels),
 anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
 max_score=np.max(anomaly_scores)
 )

 return anomaly_labels, anomaly_scores

 except Exception as e:
 logger.error("Failed to detect anomalies with IQR", error=str(e))
 raise

 def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[bool, float]:
 """
 Real-time anomaly detection for a single data point.

 Args:
 value: Value to check

 Returns:
 Tuple[bool, float]: (is_anomaly, anomaly_score)
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before real-time detection")

 try:
 if isinstance(value, (int, float)):
 value = np.array([value])
 elif isinstance(value, list):
 value = np.array(value)

 value = value.reshape(1, -1)
 anomaly_score = self._calculate_iqr_scores(value)[0]
 is_anomaly = self._is_outlier(value[0])

 if is_anomaly:
 logger.warning(
 "Real-time IQR anomaly detected",
 value=value[0],
 anomaly_score=anomaly_score,
 lower_fence=self._lower_fence,
 upper_fence=self._upper_fence
 )

 return is_anomaly, anomaly_score

 except Exception as e:
 logger.error("Failed real-time IQR detection", error=str(e))
 raise

 def _validate_input(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
 """Validate and convert input data."""
 if isinstance(X, pd.DataFrame):
 X = X.values
 elif isinstance(X, pd.Series):
 X = X.values.reshape(-1, 1)
 elif isinstance(X, list):
 X = np.array(X)

 if X.ndim == 1:
 X = X.reshape(-1, 1)

 # Check for NaN and Inf
 if np.any(~np.isfinite(X)):
 warnings.warn("Input contains NaN or Inf values, removing them")
 X = X[np.isfinite(X).all(axis=1)]

 return X

 def _calculate_iqr_scores(self, X: np.ndarray) -> np.ndarray:
 """
 Compute anomaly scores based on IQR.

 Score = max(0, (value - upper_fence), (lower_fence - value)) / IQR
 """
 # Distance to fences
 upper_dist = np.maximum(0, X - self._upper_fence)
 lower_dist = np.maximum(0, self._lower_fence - X)

 # Maximum distance (normalized by IQR)
 distances = np.maximum(upper_dist, lower_dist) / self._iqr

 # For multivariate data, take the maximum across features
 if distances.ndim > 1 and distances.shape[1] > 1:
 scores = np.max(distances, axis=1)
 else:
 scores = distances.flatten

 return scores

 def _classify_anomalies(self, X: np.ndarray) -> np.ndarray:
 """Classify anomalies based on IQR fences."""
 if self.config.bilateral:
 # Two-sided check
 outliers = (X < self._lower_fence) | (X > self._upper_fence)
 else:
 # Only upper fence (high values)
 outliers = X > self._upper_fence

 # For multivariate data: anomaly if at least one feature falls outside the fences
 if outliers.ndim > 1 and outliers.shape[1] > 1:
 outliers = np.any(outliers, axis=1)
 else:
 outliers = outliers.flatten

 return outliers.astype(int)

 def _is_outlier(self, value: np.ndarray) -> bool:
 """Check whether the value is an outlier."""
 if self.config.bilateral:
 return bool(np.any((value < self._lower_fence) | (value > self._upper_fence)))
 else:
 return bool(np.any(value > self._upper_fence))

 def _detect_rolling(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """Detect anomalies using a sliding window."""
 window_size = self.config.window_size
 n_samples = len(X)

 anomaly_labels = np.zeros(n_samples)
 anomaly_scores = np.zeros(n_samples)

 for i in range(window_size, n_samples):
 window_data = X[i-window_size:i]

 # Compute quartiles for window
 q1_window = np.percentile(
 window_data, 25, axis=0, method=self.config.quantile_method
 )
 q3_window = np.percentile(
 window_data, 75, axis=0, method=self.config.quantile_method
 )
 iqr_window = q3_window - q1_window
 iqr_window = np.where(iqr_window == 0, np.finfo(float).eps, iqr_window)

 # Fences for window
 lower_fence_window = q1_window - self.config.k_factor * iqr_window
 upper_fence_window = q3_window + self.config.k_factor * iqr_window

 # Check current point
 current_value = X[i]

 # Compute distances to fences
 upper_dist = np.maximum(0, current_value - upper_fence_window)
 lower_dist = np.maximum(0, lower_fence_window - current_value)
 distances = np.maximum(upper_dist, lower_dist) / iqr_window

 # For multivariate data
 if isinstance(distances, np.ndarray) and len(distances) > 1:
 anomaly_score = np.max(distances)
 is_outlier = np.any((current_value < lower_fence_window) |
 (current_value > upper_fence_window))
 else:
 anomaly_score = float(distances)
 is_outlier = ((current_value < lower_fence_window) |
 (current_value > upper_fence_window)).any

 anomaly_scores[i] = anomaly_score
 anomaly_labels[i] = int(is_outlier)

 return anomaly_labels, anomaly_scores

 def _auto_tune_k_factor(self, X: np.ndarray) -> None:
 """Automatically tune k-factor based on the data."""
 # Try different k-factors
 k_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
 best_k = self.config.k_factor
 target_contamination = self.config.contamination

 best_diff = float('inf')

 for k in k_candidates:
 # Temporarily set k
 lower_fence = self._q1 - k * self._iqr
 upper_fence = self._q3 + k * self._iqr

 # Count outlier fraction
 if self.config.bilateral:
 outliers = (X < lower_fence) | (X > upper_fence)
 else:
 outliers = X > upper_fence

 if outliers.ndim > 1:
 outliers = np.any(outliers, axis=1)

 actual_contamination = np.mean(outliers)
 diff = abs(actual_contamination - target_contamination)

 if diff < best_diff:
 best_diff = diff
 best_k = k

 self.config.k_factor = best_k

 logger.info(
 "Auto-tuned IQR k-factor",
 old_k_factor=1.5,
 new_k_factor=self.config.k_factor,
 target_contamination=target_contamination
 )

 def get_statistics(self) -> Dict[str, Any]:
 """Get detector statistics."""
 if not self.fitted:
 return {"status": "not_fitted"}

 return {
 "fitted": True,
 "k_factor": self.config.k_factor,
 "bilateral": self.config.bilateral,
 "min_samples": self.config.min_samples,
 "q1": self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 "q3": self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3,
 "iqr": self._iqr.tolist if isinstance(self._iqr, np.ndarray) else self._iqr,
 "lower_fence": self._lower_fence.tolist if isinstance(self._lower_fence, np.ndarray) else self._lower_fence,
 "upper_fence": self._upper_fence.tolist if isinstance(self._upper_fence, np.ndarray) else self._upper_fence
 }

 def get_fence_values(self) -> Dict[str, Any]:
 """Get fence values for visualization."""
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 return {
 "lower_fence": self._lower_fence.tolist if isinstance(self._lower_fence, np.ndarray) else self._lower_fence,
 "upper_fence": self._upper_fence.tolist if isinstance(self._upper_fence, np.ndarray) else self._upper_fence,
 "q1": self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 "q3": self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3
 }

# Usage example for crypto trading
def create_crypto_iqr_detector(
 price_data: pd.DataFrame,
 k_factor: float = 1.5,
 features: Optional[List[str]] = None
) -> IQRDetector:
 """
 Create an IQR detector optimized for crypto data.

 Args:
 price_data: DataFrame with price data
 k_factor: IQR multiplier for determining fences
 features: List of features to analyze

 Returns:
 Configured IQRDetector
 """
 if features is None:
 features = ['close', 'volume']
 if 'returns' in price_data.columns:
 features.append('returns')

 config = IQRConfig(
 k_factor=k_factor,
 auto_k_factor=True,
 contamination=0.05, # 5% anomalies in crypto
 bilateral=True,
 quantile_method='linear'
 )

 detector = IQRDetector(config)
 detector.fit(price_data[features].values)

 return detector