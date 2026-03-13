"""
📈 Z-Score Anomaly Detector

Implements Z-score based anomaly detection for crypto trading data.
Z-score measures how many standard deviations away a data point is from the mean.

Features:
- Auto-scaling thresholds
- Real-time processing capability
- Multi-dimensional anomaly detection
- Enterprise monitoring integration
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
from scipy import stats
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class ZScoreConfig:
 """Configuration for Z-Score detector."""
 threshold: float = 3.0
 window_size: Optional[int] = None # None = use all data
 rolling_window: bool = False # True = sliding window
 min_samples: int = 30
 robust: bool = False # True = use median instead of mean
 bilateral: bool = True # True = two-sided check
 auto_threshold: bool = False # True = automatically tune threshold
 contamination: float = 0.1 # Expected anomaly fraction

class ZScoreDetector:
 """
 Z-Score Anomaly Detector for crypto trading.

 Detects anomalies based on standard Z-score:
 z = (x - μ) / σ

 Features:
 - Distributed processing support
 - Auto-scaling parameters
 - Real-time streaming capability
 - Enterprise monitoring
 """

 def __init__(self, config: Optional[ZScoreConfig] = None):
 """
 Initialize the Z-Score detector.

 Args:
 config: Detector configuration
 """
 self.config = config or ZScoreConfig
 self.stats_cache = {}
 self.fitted = False
 self._mean = None
 self._std = None
 self._median = None
 self._mad = None # Median Absolute Deviation

 logger.info(
 "ZScoreDetector initialized",
 threshold=self.config.threshold,
 robust=self.config.robust,
 auto_threshold=self.config.auto_threshold
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'ZScoreDetector':
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

 # Compute statistics
 if self.config.robust:
 # Use robust statistics
 self._median = np.median(X, axis=0)
 self._mad = np.median(np.abs(X - self._median), axis=0)
 # Scaling factor for MAD
 self._mad *= 1.4826 # Constant for normal distribution
 else:
 # Use standard statistics
 self._mean = np.mean(X, axis=0)
 self._std = np.std(X, axis=0, ddof=1)

 # Automatically tune threshold
 if self.config.auto_threshold:
 self._auto_tune_threshold(X)

 self.fitted = True

 logger.info(
 "ZScoreDetector fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1] if X.ndim > 1 else 1,
 robust=self.config.robust,
 threshold=self.config.threshold
 )

 return self

 except Exception as e:
 logger.error("Failed to fit ZScoreDetector", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
 """
 Detect anomalies in data.

 Args:
 X: Data to analyze

 Returns:
 Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
 anomaly_labels: 1 for anomalies, 0 for normal points
 anomaly_scores: Z-score values (higher values indicate more anomalous points)
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before detecting anomalies")

 try:
 X = self._validate_input(X)

 # Compute Z-scores
 if self.config.rolling_window and self.config.window_size:
 anomaly_labels, z_scores = self._detect_rolling(X)
 else:
 z_scores = self._calculate_zscore(X)
 anomaly_labels = self._classify_anomalies(z_scores)

 logger.debug(
 "Anomaly detection completed",
 n_samples=len(X),
 n_anomalies=np.sum(anomaly_labels),
 anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
 max_score=np.max(np.abs(z_scores))
 )

 return anomaly_labels, np.abs(z_scores)

 except Exception as e:
 logger.error("Failed to detect anomalies", error=str(e))
 raise

 def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[bool, float]:
 """
 Real-time anomaly detection for a single data point.

 Args:
 value: Value to check

 Returns:
 Tuple[bool, float]: (is_anomaly, z_score)
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before real-time detection")

 try:
 if isinstance(value, (int, float)):
 value = np.array([value])
 elif isinstance(value, list):
 value = np.array(value)

 z_score = self._calculate_zscore(value.reshape(1, -1))[0]
 is_anomaly = abs(z_score) > self.config.threshold

 if is_anomaly:
 logger.warning(
 "Real-time anomaly detected",
 value=value,
 z_score=z_score,
 threshold=self.config.threshold
 )

 return is_anomaly, abs(z_score)

 except Exception as e:
 logger.error("Failed real-time detection", error=str(e))
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

 def _calculate_zscore(self, X: np.ndarray) -> np.ndarray:
 """Compute Z-scores."""
 if self.config.robust:
 # Robust Z-score using median and MAD
 z_scores = (X - self._median) / self._mad
 else:
 # Standard Z-score
 z_scores = (X - self._mean) / self._std

 # For multivariate data, use the maximum Z-score
 if z_scores.ndim > 1 and z_scores.shape[1] > 1:
 z_scores = np.max(np.abs(z_scores), axis=1)
 else:
 z_scores = z_scores.flatten

 return z_scores

 def _classify_anomalies(self, z_scores: np.ndarray) -> np.ndarray:
 """Classify anomalies based on Z-scores."""
 if self.config.bilateral:
 # Two-sided check
 return (np.abs(z_scores) > self.config.threshold).astype(int)
 else:
 # One-sided check (high values only)
 return (z_scores > self.config.threshold).astype(int)

 def _detect_rolling(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """Detect anomalies using a sliding window."""
 window_size = self.config.window_size
 n_samples = len(X)

 anomaly_labels = np.zeros(n_samples)
 z_scores = np.zeros(n_samples)

 for i in range(window_size, n_samples):
 window_data = X[i-window_size:i]

 if self.config.robust:
 window_median = np.median(window_data, axis=0)
 window_mad = np.median(np.abs(window_data - window_median), axis=0) * 1.4826
 z_score = (X[i] - window_median) / window_mad
 else:
 window_mean = np.mean(window_data, axis=0)
 window_std = np.std(window_data, axis=0, ddof=1)
 z_score = (X[i] - window_mean) / window_std

 # For multivariate data
 if isinstance(z_score, np.ndarray) and len(z_score) > 1:
 z_score = np.max(np.abs(z_score))

 z_scores[i] = z_score
 anomaly_labels[i] = int(abs(z_score) > self.config.threshold)

 return anomaly_labels, z_scores

 def _auto_tune_threshold(self, X: np.ndarray) -> None:
 """Automatically tune the threshold based on data."""
 # Compute all Z-scores for training data
 z_scores = self._calculate_zscore(X)

 # Use percentiles to determine the threshold
 percentile = (1 - self.config.contamination) * 100
 threshold = np.percentile(np.abs(z_scores), percentile)

 # Minimum threshold = 2.0 (standard practice)
 self.config.threshold = max(threshold, 2.0)

 logger.info(
 "Auto-tuned threshold",
 old_threshold=3.0,
 new_threshold=self.config.threshold,
 contamination=self.config.contamination
 )

 def get_statistics(self) -> Dict[str, Any]:
 """Get detector statistics."""
 if not self.fitted:
 return {"status": "not_fitted"}

 stats = {
 "fitted": True,
 "threshold": self.config.threshold,
 "robust": self.config.robust,
 "bilateral": self.config.bilateral,
 "min_samples": self.config.min_samples
 }

 if self.config.robust:
 stats.update({
 "median": self._median.tolist if isinstance(self._median, np.ndarray) else self._median,
 "mad": self._mad.tolist if isinstance(self._mad, np.ndarray) else self._mad
 })
 else:
 stats.update({
 "mean": self._mean.tolist if isinstance(self._mean, np.ndarray) else self._mean,
 "std": self._std.tolist if isinstance(self._std, np.ndarray) else self._std
 })

 return stats

# Usage example for crypto trading
def create_crypto_zscore_detector(
 price_data: pd.DataFrame,
 threshold: float = 3.5,
 robust: bool = True
) -> ZScoreDetector:
 """
 Create a Z-Score detector optimized for crypto data.

 Args:
 price_data: DataFrame with price data
 threshold: Threshold for anomaly detection
 robust: Use robust statistics

 Returns:
 Configured ZScoreDetector
 """
 config = ZScoreConfig(
 threshold=threshold,
 robust=robust,
 auto_threshold=True,
 contamination=0.05, # 5% anomalies expected in crypto
 bilateral=True
 )

 detector = ZScoreDetector(config)
 detector.fit(price_data[['close', 'volume']].values)

 return detector