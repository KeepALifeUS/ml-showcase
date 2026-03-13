"""
🛡️ One-Class SVM Anomaly Detector

Implements One-Class SVM for boundary-based anomaly detection.
Finds optimal hyperplane separating normal data from potential anomalies.

Features:
- Kernel-based boundary learning
- Non-linear decision boundaries
- Real-time classification
- Enterprise model versioning
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class OneClassSVMConfig:
 """Configuration for One-Class SVM detector."""
 kernel: str = 'rbf' # 'linear', 'poly', 'rbf', 'sigmoid'
 degree: int = 3 # Degree for poly kernel
 gamma: str = 'scale' # Kernel coefficient
 coef0: float = 0.0 # Independent term in kernel function
 tol: float = 1e-3 # Tolerance for stopping criterion
 nu: float = 0.1 # Upper bound on fraction of outliers
 shrinking: bool = True # Use shrinking heuristic
 cache_size: float = 200 # Kernel cache size in MB
 max_iter: int = -1 # Max iterations (-1 for no limit)

 # Enterprise features
 auto_gamma_optimization: bool = True
 scaler_type: str = 'standard'
 crypto_optimized: bool = True

class OneClassSVMDetector:
 """
 One-Class SVM Anomaly Detector.

 Boundary-based anomaly detection method using SVM.
 """

 def __init__(self, config: Optional[OneClassSVMConfig] = None):
 self.config = config or OneClassSVMConfig
 self.fitted = False
 self.model = None
 self.scaler = None
 self._feature_names = None

 self._create_scaler

 logger.info(
 "OneClassSVMDetector initialized",
 kernel=self.config.kernel,
 nu=self.config.nu,
 gamma=self.config.gamma
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
 try:
 X, feature_names = self._validate_and_prepare_input(X)
 self._feature_names = feature_names

 if self.scaler is not None:
 X_scaled = self.scaler.fit_transform(X)
 else:
 X_scaled = X

 # Auto-optimize gamma if enabled
 gamma = self.config.gamma
 if self.config.auto_gamma_optimization and self.config.kernel == 'rbf':
 gamma = self._optimize_gamma(X_scaled)

 self.model = OneClassSVM(
 kernel=self.config.kernel,
 degree=self.config.degree,
 gamma=gamma,
 coef0=self.config.coef0,
 tol=self.config.tol,
 nu=self.config.nu,
 shrinking=self.config.shrinking,
 cache_size=self.config.cache_size,
 max_iter=self.config.max_iter
 )

 self.model.fit(X_scaled)
 self.fitted = True

 logger.info(
 "One-Class SVM fitted successfully",
 n_samples=len(X),
 n_features=X.shape[1],
 n_support_vectors=self.model.support_.shape[0]
 )

 return self

 except Exception as e:
 logger.error("Failed to fit One-Class SVM", error=str(e))
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

 predictions = self.model.predict(X_scaled)
 scores = -self.model.decision_function(X_scaled)

 anomaly_labels = np.where(predictions == -1, 1, 0)

 logger.debug(
 "One-Class SVM detection completed",
 n_samples=len(X),
 n_anomalies=np.sum(anomaly_labels)
 )

 return anomaly_labels, scores

 except Exception as e:
 logger.error("Failed One-Class SVM detection", error=str(e))
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
 logger.error("Failed real-time One-Class SVM detection", error=str(e))
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

 def _optimize_gamma(self, X):
 """Automatic optimization of parameter gamma."""
 n_features = X.shape[1]

 # Heuristic for gamma based on data dimensionality
 gamma_candidates = [1/(n_features * X.var), 'scale', 'auto']

 best_gamma = 'scale'
 best_score = float('-inf')

 for gamma in gamma_candidates:
 try:
 temp_model = OneClassSVM(
 kernel=self.config.kernel,
 gamma=gamma,
 nu=self.config.nu
 )

 temp_model.fit(X)
 # Use number of support vectors as metric
 score = -len(temp_model.support_) # Fewer support vectors = better

 if score > best_score:
 best_score = score
 best_gamma = gamma

 except Exception:
 continue

 logger.info(
 "Optimized gamma parameter",
 original_gamma=self.config.gamma,
 optimal_gamma=best_gamma
 )

 return best_gamma

 def get_statistics(self) -> Dict[str, Any]:
 if not self.fitted:
 return {"status": "not_fitted"}

 return {
 "fitted": True,
 "kernel": self.config.kernel,
 "nu": self.config.nu,
 "gamma": self.model.gamma if hasattr(self.model, 'gamma') else self.config.gamma,
 "n_support_vectors": self.model.support_.shape[0] if hasattr(self.model, 'support_') else 0,
 "feature_names": self._feature_names
 }

def create_crypto_ocsvm_detector(price_data: pd.DataFrame, features: Optional[List[str]] = None) -> OneClassSVMDetector:
 if features is None:
 features = ['close', 'volume']
 if 'returns' in price_data.columns:
 features.append('returns')

 config = OneClassSVMConfig(
 kernel='rbf',
 nu=0.05, # 5% anomalies
 auto_gamma_optimization=True,
 crypto_optimized=True,
 scaler_type='standard'
 )

 detector = OneClassSVMDetector(config)
 detector.fit(price_data[features].dropna)

 return detector