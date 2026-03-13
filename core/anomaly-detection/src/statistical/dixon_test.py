"""
🔬 Dixon's Q Test Anomaly Detector

Implements Dixon's Q test for detecting outliers in small samples (3-30 observations).
Simple ratio test that compares the gap containing the suspected outlier to the total range.

Q = gap / range
Where gap is the difference between the outlier and its nearest neighbor,
and range is the full range of the data.

Features:
- Small sample optimization
- Quick statistical test
- Multiple Q-test variants
- Iterative outlier removal
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class DixonConfig:
 """Configuration for Dixon Q test detector."""
 alpha: float = 0.05 # Significance level
 test_type: str = "r10" # Q-test type: r10, r11, r21, r22
 max_outliers: Optional[int] = None
 iterative: bool = True
 two_sided: bool = True
 min_samples: int = 3
 max_samples: int = 30 # Dixon test is effective for small samples

class DixonTestDetector:
 """
 Dixon's Q Test Anomaly Detector.

 Statistical test for small samples (3-30 observations).
 Uses gap-to-range ratio for outlier identification.

 Features:
 - Optimized for small samples
 - Fast computation
 - Multiple test variants
 - Statistical significance
 """

 # Critical values for Dixon Q-test
 CRITICAL_VALUES = {
 'r10': {
 3: {0.10: 0.886, 0.05: 0.941, 0.02: 0.976, 0.01: 0.988},
 4: {0.10: 0.679, 0.05: 0.765, 0.02: 0.846, 0.01: 0.889},
 5: {0.10: 0.557, 0.05: 0.642, 0.02: 0.729, 0.01: 0.780},
 6: {0.10: 0.482, 0.05: 0.560, 0.02: 0.644, 0.01: 0.698},
 7: {0.10: 0.434, 0.05: 0.507, 0.02: 0.586, 0.01: 0.637},
 8: {0.10: 0.399, 0.05: 0.468, 0.02: 0.543, 0.01: 0.590},
 9: {0.10: 0.370, 0.05: 0.437, 0.02: 0.510, 0.01: 0.555},
 10: {0.10: 0.349, 0.05: 0.412, 0.02: 0.483, 0.01: 0.527},
 },
 'r11': {
 # For larger samples (11-30)
 11: {0.10: 0.332, 0.05: 0.392, 0.02: 0.460, 0.01: 0.502},
 12: {0.10: 0.318, 0.05: 0.376, 0.02: 0.441, 0.01: 0.482},
 13: {0.10: 0.305, 0.05: 0.361, 0.02: 0.425, 0.01: 0.465},
 14: {0.10: 0.294, 0.05: 0.349, 0.02: 0.411, 0.01: 0.450},
 15: {0.10: 0.284, 0.05: 0.338, 0.02: 0.399, 0.01: 0.438},
 # ... can be extended up to n=30
 }
 }

 def __init__(self, config: Optional[DixonConfig] = None):
 """
 Initialize the Dixon Q test detector.

 Args:
 config: Detector configuration
 """
 self.config = config or DixonConfig
 self.fitted = False
 self._sample_size_range = (self.config.min_samples, self.config.max_samples)

 logger.info(
 "DixonTestDetector initialized",
 alpha=self.config.alpha,
 test_type=self.config.test_type,
 iterative=self.config.iterative,
 sample_range=self._sample_size_range
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> 'DixonTestDetector':
 """
 Train detector (Dixon test does not require training).

 Args:
 X: Historical data

 Returns:
 self: Detector
 """
 try:
 X = self._validate_input(X)

 if len(X) < self.config.min_samples:
 raise ValueError(
 f"Insufficient samples for Dixon test: {len(X)} < {self.config.min_samples}"
 )

 if len(X) > self.config.max_samples:
 logger.warning(
 "Sample size exceeds Dixon test recommendations",
 n_samples=len(X),
 max_recommended=self.config.max_samples,
 recommendation="Consider using other statistical tests"
 )

 self.fitted = True

 logger.info(
 "DixonTestDetector fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1] if X.ndim > 1 else 1
 )

 return self

 except Exception as e:
 logger.error("Failed to fit DixonTestDetector", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
 """
 Detect anomalies using Dixon Q test.

 Args:
 X: Data to analyze

 Returns:
 Tuple[np.ndarray, np.ndarray]: (anomaly_labels, q_statistics)
 """
 if not self.fitted:
 # Dixon test does not require training, but check size
 X_temp = self._validate_input(X)
 if len(X_temp) < self.config.min_samples:
 raise ValueError("Insufficient samples for Dixon test")
 self.fitted = True

 try:
 X = self._validate_input(X)

 if X.shape[1] == 1:
 # Univariate case
 anomaly_labels, q_stats = self._detect_univariate(X.flatten)
 else:
 # Multivariate case
 anomaly_labels, q_stats = self._detect_multivariate(X)

 logger.debug(
 "Dixon Q test completed",
 n_samples=len(X),
 n_anomalies=np.sum(anomaly_labels),
 anomaly_rate=f"{np.mean(anomaly_labels):.3%}",
 max_q_stat=np.max(q_stats)
 )

 return anomaly_labels, q_stats

 except Exception as e:
 logger.error("Failed to detect anomalies with Dixon Q test", error=str(e))
 raise

 def detect_single_outlier(self, X: np.ndarray) -> Tuple[bool, int, float]:
 """
 Single outlier detection using Dixon Q test.

 Args:
 X: Univariate data array

 Returns:
 Tuple[bool, int, float]: (is_outlier, outlier_index, q_statistic)
 """
 X = X.flatten
 n = len(X)

 if n < self.config.min_samples or n > self.config.max_samples:
 return False, -1, 0.0

 # Sort data
 sorted_indices = np.argsort(X)
 sorted_X = X[sorted_indices]

 # Compute Q-statistics for both sides
 q_low, q_high = self._calculate_q_statistics(sorted_X)

 # Determine maximum Q-statistic and corresponding index
 if q_low > q_high:
 q_max = q_low
 outlier_sorted_idx = 0 # Smallest value
 else:
 q_max = q_high
 outlier_sorted_idx = n - 1 # Largest value

 # Get critical value
 critical_value = self._get_critical_value(n)

 # Check significance
 is_outlier = q_max > critical_value

 # Convert back to original indices
 outlier_original_idx = sorted_indices[outlier_sorted_idx]

 if is_outlier:
 logger.debug(
 "Dixon Q test outlier detected",
 index=outlier_original_idx,
 value=X[outlier_original_idx],
 q_statistic=q_max,
 critical_value=critical_value
 )

 return is_outlier, outlier_original_idx, q_max

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

 def _detect_univariate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """Outlier detection in univariate data."""
 X_work = X.copy
 n_original = len(X)
 anomaly_labels = np.zeros(n_original, dtype=int)
 q_statistics = np.zeros(n_original)

 # Create index mapping
 remaining_indices = np.arange(n_original)

 max_outliers = self.config.max_outliers or min(3, len(X) // 5)
 outliers_found = 0

 while (len(X_work) >= self.config.min_samples and
 len(X_work) <= self.config.max_samples and
 outliers_found < max_outliers):

 is_outlier, local_idx, q_stat = self.detect_single_outlier(X_work)

 if not is_outlier or not self.config.iterative:
 break

 # Find global index
 global_idx = remaining_indices[local_idx]

 # Record result
 anomaly_labels[global_idx] = 1
 q_statistics[global_idx] = q_stat

 # Remove outlier for next iteration
 X_work = np.delete(X_work, local_idx)
 remaining_indices = np.delete(remaining_indices, local_idx)
 outliers_found += 1

 if not self.config.iterative:
 break

 return anomaly_labels, q_statistics

 def _detect_multivariate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """Outlier detection in multivariate data."""
 n_samples, n_features = X.shape
 anomaly_labels = np.zeros(n_samples, dtype=int)
 q_statistics = np.zeros(n_samples)

 for feature_idx in range(n_features):
 feature_data = X[:, feature_idx]
 feature_labels, feature_stats = self._detect_univariate(feature_data)

 # Combine results
 anomaly_labels = np.maximum(anomaly_labels, feature_labels)
 q_statistics = np.maximum(q_statistics, feature_stats)

 return anomaly_labels, q_statistics

 def _calculate_q_statistics(self, sorted_X: np.ndarray) -> Tuple[float, float]:
 """
 Compute Q-statistics for the lower and upper values.

 Args:
 sorted_X: Sorted data array

 Returns:
 Tuple[float, float]: (q_low, q_high)
 """
 n = len(sorted_X)

 if n < 3:
 return 0.0, 0.0

 # Total range
 total_range = sorted_X[-1] - sorted_X[0]

 if total_range == 0:
 return 0.0, 0.0

 # Q-statistic for lower value
 if self.config.test_type == "r10" or n <= 10:
 # r10: Q = (x2 - x1) / (xn - x1)
 q_low = (sorted_X[1] - sorted_X[0]) / total_range
 q_high = (sorted_X[-1] - sorted_X[-2]) / total_range
 else:
 # r11: Q = (x2 - x1) / (xn-1 - x1)
 q_low = (sorted_X[1] - sorted_X[0]) / (sorted_X[-2] - sorted_X[0])
 q_high = (sorted_X[-1] - sorted_X[-2]) / (sorted_X[-1] - sorted_X[1])

 return q_low, q_high

 def _get_critical_value(self, n: int) -> float:
 """Get critical value for sample size n."""
 # Determine test type
 if n <= 10:
 test_type = 'r10'
 else:
 test_type = 'r11'

 # Look up exact value
 if test_type in self.CRITICAL_VALUES and n in self.CRITICAL_VALUES[test_type]:
 alpha_values = self.CRITICAL_VALUES[test_type][n]

 # Find the nearest alpha
 available_alphas = sorted(alpha_values.keys, reverse=True)
 for alpha in available_alphas:
 if self.config.alpha >= alpha:
 return alpha_values[alpha]

 # If alpha is very small, use the strictest value
 return alpha_values[min(available_alphas)]

 # For n values not in the table, use interpolation or constant
 if test_type == 'r10':
 # For small samples
 return max(0.3, 0.9 - 0.05 * n) # Empirical formula
 else:
 # For larger samples
 return max(0.2, 0.5 - 0.015 * (n - 10))

 def get_statistics(self) -> Dict[str, Any]:
 """Get detector statistics."""
 if not self.fitted:
 return {"status": "not_fitted"}

 return {
 "fitted": True,
 "alpha": self.config.alpha,
 "test_type": self.config.test_type,
 "iterative": self.config.iterative,
 "two_sided": self.config.two_sided,
 "sample_size_range": self._sample_size_range,
 "critical_values_available": len(self.CRITICAL_VALUES.get(self.config.test_type, {}))
 }

# Usage example for crypto trading
def create_crypto_dixon_detector(
 price_data: pd.DataFrame,
 feature: str = 'returns',
 alpha: float = 0.05,
 window_size: int = 20
) -> DixonTestDetector:
 """
 Create Dixon detector for crypto data.

 Args:
 price_data: DataFrame with price data
 feature: Feature to analyze
 alpha: Significance level
 window_size: Analysis window size (must be ≤ 30)

 Returns:
 Configured DixonTestDetector
 """
 if window_size > 30:
 logger.warning(
 "Dixon test not recommended for samples > 30",
 window_size=window_size,
 recommendation="Consider using Grubbs test or other methods"
 )
 window_size = 30

 config = DixonConfig(
 alpha=alpha,
 test_type="r10" if window_size <= 10 else "r11",
 iterative=True,
 two_sided=True,
 max_samples=window_size
 )

 detector = DixonTestDetector(config)

 # Use sliding window for analysis
 if feature in price_data.columns:
 # Take last window_size observations for training
 recent_data = price_data[feature].dropna.tail(window_size).values.reshape(-1, 1)
 detector.fit(recent_data)
 else:
 raise ValueError(f"Feature '{feature}' not found in price_data")

 return detector