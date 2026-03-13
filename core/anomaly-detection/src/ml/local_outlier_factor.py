"""
🎯 Local Outlier Factor (LOF) Anomaly Detector

Implements LOF algorithm for density-based anomaly detection.
Identifies anomalies based on local density deviation compared to neighbors.

Features:
- Density-based anomaly scoring
- k-nearest neighbors optimization
- Real-time local density computation
- Enterprise monitoring integration
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class LOFConfig:
 """Configuration for LOF detector."""
 n_neighbors: int = 20
 algorithm: str = 'auto' # 'ball_tree', 'kd_tree', 'brute', 'auto'
 leaf_size: int = 30
 metric: str = 'minkowski'
 p: int = 2 # Parameter for minkowski metric
 contamination: float = 0.1
 novelty: bool = True # True for new data detection
 n_jobs: int = -1

 # Enterprise features
 auto_k_optimization: bool = True # Auto-optimize k
 scaler_type: str = 'robust'
 crypto_optimized: bool = True

class LocalOutlierFactorDetector:
 """
 Local Outlier Factor Anomaly Detector.

 Density-based anomaly detection method based on local density.
 """

 def __init__(self, config: Optional[LOFConfig] = None):
 self.config = config or LOFConfig
 self.fitted = False
 self.model = None
 self.scaler = None
 self._feature_names = None
 self._optimal_k = None

 self._create_scaler

 logger.info(
 "LocalOutlierFactorDetector initialized",
 n_neighbors=self.config.n_neighbors,
 contamination=self.config.contamination,
 metric=self.config.metric
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
 try:
 X, feature_names = self._validate_and_prepare_input(X)
 self._feature_names = feature_names

 # Scaling
 if self.scaler is not None:
 X_scaled = self.scaler.fit_transform(X)
 else:
 X_scaled = X

 # Auto-optimize k if enabled
 if self.config.auto_k_optimization:
 self._optimize_k(X_scaled)

 # Create and train the model
 self.model = LocalOutlierFactor(
 n_neighbors=self._optimal_k or self.config.n_neighbors,
 algorithm=self.config.algorithm,
 leaf_size=self.config.leaf_size,
 metric=self.config.metric,
 p=self.config.p,
 contamination=self.config.contamination,
 novelty=self.config.novelty,
 n_jobs=self.config.n_jobs
 )

 self.model.fit(X_scaled)
 self.fitted = True

 logger.info(
 "LOF detector fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1],
 optimal_k=self._optimal_k or self.config.n_neighbors
 )

 return self

 except Exception as e:
 logger.error("Failed to fit LOF detector", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 try:
 X, _ = self._validate_and_prepare_input(X)

 if self.scaler is not None:
 X_scaled = self.scaler.transform(X)
 else:
 X_scaled = X

 # Anomaly prediction
 if self.config.novelty:
 predictions = self.model.predict(X_scaled)
 scores = -self.model.decision_function(X_scaled)
 else:
 # For training data use negative_outlier_factor_
 predictions = self.model.fit_predict(X_scaled)
 scores = -self.model.negative_outlier_factor_

 anomaly_labels = np.where(predictions == -1, 1, 0)

 logger.debug(
 "LOF detection completed",
 n_samples=len(X),
 n_anomalies=np.sum(anomaly_labels),
 max_score=np.max(scores)
 )

 return anomaly_labels, scores

 except Exception as e:
 logger.error("Failed LOF detection", error=str(e))
 raise

 def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[bool, float]:
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 try:
 if isinstance(value, (int, float)):
 value = np.array([value])
 elif isinstance(value, list):
 value = np.array(value)

 value = value.reshape(1, -1)

 if self.scaler is not None:
 value_scaled = self.scaler.transform(value)
 else:
 value_scaled = value

 prediction = self.model.predict(value_scaled)[0]
 score = -self.model.decision_function(value_scaled)[0]

 is_anomaly = prediction == -1

 return is_anomaly, score

 except Exception as e:
 logger.error("Failed real-time LOF detection", error=str(e))
 raise

 def _validate_and_prepare_input(self, X):
 feature_names = None

 if isinstance(X, pd.DataFrame):
 feature_names = X.columns.tolist
 X = X.values
 elif isinstance(X, pd.Series):
 feature_names = [X.name] if X.name else ["feature_0"]
 X = X.values.reshape(-1, 1)
 elif isinstance(X, list):
 X = np.array(X)

 if X.ndim == 1:
 X = X.reshape(-1, 1)

 if np.any(~np.isfinite(X)):
 warnings.warn("Input contains NaN or Inf values, replacing with median")
 from sklearn.impute import SimpleImputer
 imputer = SimpleImputer(strategy='median')
 X = imputer.fit_transform(X)

 return X, feature_names

 def _create_scaler(self):
 if self.config.scaler_type == 'standard':
 self.scaler = StandardScaler
 elif self.config.scaler_type == 'robust':
 self.scaler = RobustScaler
 else:
 self.scaler = None

 def _optimize_k(self, X):
 """Automatic optimization of parameter k."""
 n_samples = X.shape[0]

 # Range of k for testing
 k_min = max(5, int(np.sqrt(n_samples) / 2))
 k_max = min(50, n_samples // 10)
 k_candidates = range(k_min, k_max, 2)

 best_k = self.config.n_neighbors
 best_score = float('-inf')

 for k in k_candidates:
 try:
 temp_model = LocalOutlierFactor(
 n_neighbors=k,
 contamination=self.config.contamination,
 novelty=False # For optimization use training data
 )

 temp_model.fit(X)
 # Use mean LOF value as quality metric
 avg_lof = np.mean(temp_model.negative_outlier_factor_)

 if avg_lof > best_score:
 best_score = avg_lof
 best_k = k

 except Exception:
 continue

 self._optimal_k = best_k

 logger.info(
 "Optimized k parameter for LOF",
 original_k=self.config.n_neighbors,
 optimal_k=self._optimal_k,
 tested_range=(k_min, k_max)
 )

 def get_statistics(self) -> Dict[str, Any]:
 if not self.fitted:
 return {"status": "not_fitted"}

 return {
 "fitted": True,
 "n_neighbors": self._optimal_k or self.config.n_neighbors,
 "contamination": self.config.contamination,
 "metric": self.config.metric,
 "feature_names": self._feature_names
 }

def create_crypto_lof_detector(price_data: pd.DataFrame, features: Optional[List[str]] = None) -> LocalOutlierFactorDetector:
 if features is None:
 features = ['close', 'volume']
 if 'returns' in price_data.columns:
 features.append('returns')

 config = LOFConfig(
 n_neighbors=20,
 contamination=0.05,
 auto_k_optimization=True,
 crypto_optimized=True,
 scaler_type='robust'
 )

 detector = LocalOutlierFactorDetector(config)
 detector.fit(price_data[features].dropna)

 return detector