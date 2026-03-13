"""
🤖 Machine Learning Anomaly Detection Module

Implements modern ML-based anomaly detection algorithms:
- Isolation Forest - tree-based ensemble method
- Local Outlier Factor - density-based method
- One-Class SVM - boundary-based method
- Autoencoder - neural network reconstruction
- LSTM Autoencoder - sequential reconstruction
- Variational Autoencoder - probabilistic reconstruction

All detectors follow enterprise patterns with proper model versioning,
monitoring, and distributed processing capabilities.
"""

from .isolation_forest import IsolationForestDetector
from .local_outlier_factor import LocalOutlierFactorDetector
from .one_class_svm import OneClassSVMDetector
from .autoencoder import AutoencoderDetector
from .lstm_autoencoder import LSTMAutoencoderDetector
from .vae_detector import VAEDetector

__all__ = [
 "IsolationForestDetector",
 "LocalOutlierFactorDetector",
 "OneClassSVMDetector",
 "AutoencoderDetector",
 "LSTMAutoencoderDetector",
 "VAEDetector"
]