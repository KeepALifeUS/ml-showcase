"""
🎯 Tukey's Fences Anomaly Detector

Implements Tukey's method for outlier detection using fences based on quartiles.
Also known as the "boxplot method" - commonly used for exploratory data analysis.

Fences:
- Inner fences: Q1 - 1.5*IQR, Q3 + 1.5*IQR (mild outliers)
- Outer fences: Q1 - 3.0*IQR, Q3 + 3.0*IQR (extreme outliers)

Features:
- Two-tier outlier classification
- Visual interpretation support
- Robust quartile-based method
- Configurable fence multipliers
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import structlog
import warnings

logger = structlog.get_logger(__name__)

class OutlierSeverity(Enum):
 """Outlier severity classification."""
 NORMAL = 0
 MILD = 1 # Between inner and outer fences
 EXTREME = 2 # Beyond outer fences

@dataclass
class TukeyConfig:
 """Configuration for Tukey's method detector."""
 inner_fence_factor: float = 1.5 # Multiplier for inner fences
 outer_fence_factor: float = 3.0 # Multiplier for outer fences
 classify_severity: bool = True # Classify outlier severity
 bilateral: bool = True # Two-sided check
 quantile_method: str = 'linear' # Quantile interpolation method
 min_samples: int = 5

class TukeyMethodDetector:
 """
 Tukey's Fences Anomaly Detector.

 Classical method for outlier detection based on quartiles and IQR.
 Used in boxplots and widely applied in exploratory data analysis.

 Features:
 - Industry standard method
 - Visual interpretation ready
 - Two-tier classification
 - Robust to distribution shape
 """

 def __init__(self, config: Optional[TukeyConfig] = None):
 """
 Initialize the Tukey's method detector.

 Args:
 config: Detector configuration
 """
 self.config = config or TukeyConfig
 self.fitted = False
 self._q1 = None
 self._q3 = None
 self._iqr = None
 self._inner_lower = None
 self._inner_upper = None
 self._outer_lower = None
 self._outer_upper = None

 logger.info(
 "TukeyMethodDetector initialized",
 inner_factor=self.config.inner_fence_factor,
 outer_factor=self.config.outer_fence_factor,
 classify_severity=self.config.classify_severity
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'TukeyMethodDetector':
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

 # Protection against division by zero
 self._iqr = np.where(self._iqr == 0, np.finfo(float).eps, self._iqr)

 # Inner fences (mild outliers)
 self._inner_lower = self._q1 - self.config.inner_fence_factor * self._iqr
 self._inner_upper = self._q3 + self.config.inner_fence_factor * self._iqr

 # Outer fences (extreme outliers)
 self._outer_lower = self._q1 - self.config.outer_fence_factor * self._iqr
 self._outer_upper = self._q3 + self.config.outer_fence_factor * self._iqr

 self.fitted = True

 logger.info(
 "TukeyMethodDetector fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1] if X.ndim > 1 else 1,
 q1=self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 q3=self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3,
 iqr=self._iqr.tolist if isinstance(self._iqr, np.ndarray) else self._iqr
 )

 return self

 except Exception as e:
 logger.error("Failed to fit TukeyMethodDetector", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
 """
 Detect anomalies in data.

 Args:
 X: Data to analyze

 Returns:
 Tuple[np.ndarray, np.ndarray]: (anomaly_labels, severity_scores)
 anomaly_labels: 0=normal, 1=mild outlier, 2=extreme outlier
 severity_scores: Distance to nearest fence (normalized)
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before detecting anomalies")

 try:
 X = self._validate_input(X)

 # Classify anomalies
 anomaly_labels = self._classify_outliers(X)

 # Compute severity (distance to fences)
 severity_scores = self._calculate_severity_scores(X)

 n_mild = np.sum(anomaly_labels == 1)
 n_extreme = np.sum(anomaly_labels == 2)
 n_total_outliers = n_mild + n_extreme

 logger.debug(
 "Tukey's method detection completed",
 n_samples=len(X),
 n_mild_outliers=n_mild,
 n_extreme_outliers=n_extreme,
 total_outliers=n_total_outliers,
 outlier_rate=f"{n_total_outliers/len(X):.3%}",
 max_severity=np.max(severity_scores)
 )

 return anomaly_labels, severity_scores

 except Exception as e:
 logger.error("Failed to detect anomalies with Tukey's method", error=str(e))
 raise

 def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[int, float]:
 """
 Real-time anomaly classification.

 Args:
 value: Value to check

 Returns:
 Tuple[int, float]: (severity_level, severity_score)
 severity_level: 0=normal, 1=mild, 2=extreme
 severity_score: Numeric severity score
 """
 if not self.fitted:
 raise ValueError("Detector must be fitted before real-time detection")

 try:
 if isinstance(value, (int, float)):
 value = np.array([value])
 elif isinstance(value, list):
 value = np.array(value)

 value = value.reshape(1, -1)
 severity_level = self._classify_outliers(value)[0]
 severity_score = self._calculate_severity_scores(value)[0]

 if severity_level > 0:
 logger.warning(
 "Real-time Tukey outlier detected",
 value=value[0],
 severity_level=severity_level,
 severity_score=severity_score,
 classification=OutlierSeverity(severity_level).name
 )

 return int(severity_level), severity_score

 except Exception as e:
 logger.error("Failed real-time Tukey detection", error=str(e))
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

 def _classify_outliers(self, X: np.ndarray) -> np.ndarray:
 """Classify outliers by severity level."""
 n_samples = len(X)
 classifications = np.zeros(n_samples, dtype=int)

 for i in range(n_samples):
 sample = X[i]
 severity = self._classify_single_sample(sample)
 classifications[i] = severity.value

 return classifications

 def _classify_single_sample(self, sample: np.ndarray) -> OutlierSeverity:
 """Classify a single sample."""
 # For multivariate data, use the worst classification
 max_severity = OutlierSeverity.NORMAL

 if self.config.bilateral:
 # Two-sided check
 beyond_outer = ((sample < self._outer_lower) | (sample > self._outer_upper))
 beyond_inner = ((sample < self._inner_lower) | (sample > self._inner_upper))
 else:
 # Only upper fences
 beyond_outer = (sample > self._outer_upper)
 beyond_inner = (sample > self._inner_upper)

 # Check each feature
 if isinstance(beyond_outer, np.ndarray):
 if np.any(beyond_outer):
 max_severity = OutlierSeverity.EXTREME
 elif np.any(beyond_inner):
 max_severity = OutlierSeverity.MILD
 else:
 if beyond_outer:
 max_severity = OutlierSeverity.EXTREME
 elif beyond_inner:
 max_severity = OutlierSeverity.MILD

 return max_severity

 def _calculate_severity_scores(self, X: np.ndarray) -> np.ndarray:
 """
 Compute numeric severity scores.

 Score = max(0, distance_to_nearest_fences) / IQR
 """
 n_samples = len(X)
 severity_scores = np.zeros(n_samples)

 for i in range(n_samples):
 sample = X[i]

 if self.config.bilateral:
 # Distances to all fences
 dist_to_inner_lower = np.maximum(0, self._inner_lower - sample)
 dist_to_inner_upper = np.maximum(0, sample - self._inner_upper)

 # Maximum distance beyond inner fences
 inner_violation = np.maximum(dist_to_inner_lower, dist_to_inner_upper)
 else:
 # Only upper fence
 inner_violation = np.maximum(0, sample - self._inner_upper)

 # Normalize by IQR
 normalized_violation = inner_violation / self._iqr

 # For multivariate data, take the maximum
 if isinstance(normalized_violation, np.ndarray) and len(normalized_violation) > 1:
 severity_scores[i] = np.max(normalized_violation)
 else:
 severity_scores[i] = float(normalized_violation)

 return severity_scores

 def get_statistics(self) -> Dict[str, Any]:
 """Get detector statistics."""
 if not self.fitted:
 return {"status": "not_fitted"}

 return {
 "fitted": True,
 "inner_fence_factor": self.config.inner_fence_factor,
 "outer_fence_factor": self.config.outer_fence_factor,
 "bilateral": self.config.bilateral,
 "classify_severity": self.config.classify_severity,
 "q1": self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 "q3": self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3,
 "iqr": self._iqr.tolist if isinstance(self._iqr, np.ndarray) else self._iqr,
 "inner_fences": {
 "lower": self._inner_lower.tolist if isinstance(self._inner_lower, np.ndarray) else self._inner_lower,
 "upper": self._inner_upper.tolist if isinstance(self._inner_upper, np.ndarray) else self._inner_upper
 },
 "outer_fences": {
 "lower": self._outer_lower.tolist if isinstance(self._outer_lower, np.ndarray) else self._outer_lower,
 "upper": self._outer_upper.tolist if isinstance(self._outer_upper, np.ndarray) else self._outer_upper
 }
 }

 def get_boxplot_data(self) -> Dict[str, Any]:
 """Get data for building a boxplot."""
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 return {
 "q1": self._q1.tolist if isinstance(self._q1, np.ndarray) else self._q1,
 "q2_median": ((self._q1 + self._q3) / 2).tolist if isinstance(self._q1, np.ndarray) else (self._q1 + self._q3) / 2,
 "q3": self._q3.tolist if isinstance(self._q3, np.ndarray) else self._q3,
 "whiskers": {
 "lower": self._inner_lower.tolist if isinstance(self._inner_lower, np.ndarray) else self._inner_lower,
 "upper": self._inner_upper.tolist if isinstance(self._inner_upper, np.ndarray) else self._inner_upper
 },
 "outlier_thresholds": {
 "mild": {
 "lower": self._inner_lower.tolist if isinstance(self._inner_lower, np.ndarray) else self._inner_lower,
 "upper": self._inner_upper.tolist if isinstance(self._inner_upper, np.ndarray) else self._inner_upper
 },
 "extreme": {
 "lower": self._outer_lower.tolist if isinstance(self._outer_lower, np.ndarray) else self._outer_lower,
 "upper": self._outer_upper.tolist if isinstance(self._outer_upper, np.ndarray) else self._outer_upper
 }
 }
 }

# Usage example for crypto trading
def create_crypto_tukey_detector(
 price_data: pd.DataFrame,
 features: Optional[List[str]] = None,
 inner_factor: float = 1.5,
 outer_factor: float = 3.0
) -> TukeyMethodDetector:
 """
 Create a Tukey detector for crypto data.

 Args:
 price_data: DataFrame with price data
 features: List of features to analyze
 inner_factor: Multiplier for inner fences
 outer_factor: Multiplier for outer fences

 Returns:
 Configured TukeyMethodDetector
 """
 if features is None:
 features = ['close', 'volume']
 if 'returns' in price_data.columns:
 features.append('returns')
 if 'volatility' in price_data.columns:
 features.append('volatility')

 config = TukeyConfig(
 inner_fence_factor=inner_factor,
 outer_fence_factor=outer_factor,
 classify_severity=True,
 bilateral=True,
 quantile_method='linear'
 )

 detector = TukeyMethodDetector(config)
 detector.fit(price_data[features].dropna.values)

 return detector

def analyze_crypto_outliers(
 detector: TukeyMethodDetector,
 price_data: pd.DataFrame,
 features: List[str]
) -> Dict[str, Any]:
 """
 Analyze outliers in crypto data with detailed statistics.

 Args:
 detector: Trained TukeyMethodDetector
 price_data: DataFrame with data
 features: List of analyzed features

 Returns:
 Dict with analysis results
 """
 data = price_data[features].dropna.values
 anomaly_labels, severity_scores = detector.detect(data)

 # Statistics by outlier type
 n_normal = np.sum(anomaly_labels == 0)
 n_mild = np.sum(anomaly_labels == 1)
 n_extreme = np.sum(anomaly_labels == 2)
 n_total = len(anomaly_labels)

 # Outlier indices
 mild_indices = np.where(anomaly_labels == 1)[0]
 extreme_indices = np.where(anomaly_labels == 2)[0]

 return {
 "summary": {
 "total_samples": n_total,
 "normal": n_normal,
 "mild_outliers": n_mild,
 "extreme_outliers": n_extreme,
 "outlier_rate": (n_mild + n_extreme) / n_total
 },
 "outlier_indices": {
 "mild": mild_indices.tolist,
 "extreme": extreme_indices.tolist
 },
 "severity_stats": {
 "mean": np.mean(severity_scores),
 "std": np.std(severity_scores),
 "max": np.max(severity_scores),
 "percentiles": {
 "50": np.percentile(severity_scores, 50),
 "95": np.percentile(severity_scores, 95),
 "99": np.percentile(severity_scores, 99)
 }
 },
 "boxplot_data": detector.get_boxplot_data
 }