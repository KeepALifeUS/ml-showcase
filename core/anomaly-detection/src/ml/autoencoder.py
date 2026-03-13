"""
🧠 Autoencoder Anomaly Detector

Implements neural network autoencoder for reconstruction-based anomaly detection.
Anomalies have higher reconstruction error compared to normal patterns.

Features:
- TensorFlow/Keras implementation
- Distributed training support
- Model versioning and checkpointing
- Real-time inference optimization
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class AutoencoderConfig:
 """Configuration for Autoencoder detector."""
 # Architecture
 hidden_layers: List[int] = None # [64, 32, 16, 32, 64]
 activation: str = 'relu'
 output_activation: str = 'linear'
 dropout_rate: float = 0.2

 # Training
 epochs: int = 100
 batch_size: int = 32
 learning_rate: float = 0.001
 validation_split: float = 0.2
 early_stopping_patience: int = 10

 # Anomaly detection
 threshold_percentile: float = 95 # Percentile for anomaly threshold
 scaler_type: str = 'minmax'

 # Enterprise features
 model_versioning: bool = True
 checkpointing: bool = True
 distributed_training: bool = False
 crypto_optimized: bool = True

class AutoencoderDetector:
 """
 Autoencoder Anomaly Detector.

 Neural network-based anomaly detection method based on reconstruction.
 """

 def __init__(self, config: Optional[AutoencoderConfig] = None):
 self.config = config or AutoencoderConfig
 if self.config.hidden_layers is None:
 self.config.hidden_layers = [64, 32, 16, 32, 64]

 self.fitted = False
 self.model = None
 self.scaler = None
 self.threshold = None
 self._feature_names = None
 self._training_history = None

 self._create_scaler

 logger.info(
 "AutoencoderDetector initialized",
 hidden_layers=self.config.hidden_layers,
 epochs=self.config.epochs,
 crypto_optimized=self.config.crypto_optimized
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
 try:
 X, feature_names = self._validate_and_prepare_input(X)
 self._feature_names = feature_names
 input_dim = X.shape[1]

 # Scaling
 if self.scaler is not None:
 X_scaled = self.scaler.fit_transform(X)
 else:
 X_scaled = X

 # Create model
 self.model = self._build_autoencoder(input_dim)

 # Callbacks
 callbacks = self._create_callbacks

 # Training
 logger.info("Training autoencoder model")\n history = self.model.fit(\n X_scaled, X_scaled,\n epochs=self.config.epochs,\n batch_size=self.config.batch_size,\n validation_split=self.config.validation_split,\n callbacks=callbacks,\n verbose=0\n )\n \n self._training_history = history.history\n \n # Compute threshold\n reconstructed = self.model.predict(X_scaled)\n mse = np.mean(np.square(X_scaled - reconstructed), axis=1)\n self.threshold = np.percentile(mse, self.config.threshold_percentile)\n \n self.fitted = True\n \n logger.info(\n "Autoencoder fitted successfully",\n n_samples=len(X),\n n_features=input_dim,\n threshold=self.threshold,\n final_loss=history.history['loss'][-1]\n )\n \n return self\n \n except Exception as e:\n logger.error("Failed to fit autoencoder", error=str(e))\n raise\n \n def detect(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:\n if not self.fitted:\n raise ValueError("Detector must be fitted")\n \n try:\n X, _ = self._validate_and_prepare_input(X)\n \n if self.scaler is not None:\n X_scaled = self.scaler.transform(X)\n else:\n X_scaled = X\n \n # Reconstruction\n reconstructed = self.model.predict(X_scaled)\n \n # Compute reconstruction error\n mse = np.mean(np.square(X_scaled - reconstructed), axis=1)\n \n # Anomaly labels\n anomaly_labels = (mse > self.threshold).astype(int)\n \n logger.debug(\n "Autoencoder detection completed",\n n_samples=len(X),\n n_anomalies=np.sum(anomaly_labels),\n max_mse=np.max(mse)\n )\n \n return anomaly_labels, mse\n \n except Exception as e:\n logger.error("Failed autoencoder detection", error=str(e))\n raise\n \n def detect_realtime(self, value: Union[float, np.ndarray]) -> Tuple[bool, float]:\n if not self.fitted:\n raise ValueError("Detector must be fitted")\n \n try:\n if isinstance(value, (int, float)):\n value = np.array([value])\n elif isinstance(value, list):\n value = np.array(value)\n \n value = value.reshape(1, -1)\n \n if self.scaler is not None:\n value_scaled = self.scaler.transform(value)\n else:\n value_scaled = value\n \n reconstructed = self.model.predict(value_scaled)\n mse = np.mean(np.square(value_scaled - reconstructed))\n \n is_anomaly = mse > self.threshold\n \n return bool(is_anomaly), float(mse)\n \n except Exception as e:\n logger.error("Failed real-time autoencoder detection", error=str(e))\n raise\n \n def _build_autoencoder(self, input_dim: int) -> keras.Model:\n """Build autoencoder architecture."""\n # Encoder\n input_layer = keras.layers.Input(shape=(input_dim,))\n x = input_layer\n \n # Encoder layers\n encoder_layers = self.config.hidden_layers[:len(self.config.hidden_layers)//2 + 1]\n for units in encoder_layers:\n x = keras.layers.Dense(units, activation=self.config.activation)(x)\n if self.config.dropout_rate > 0:\n x = keras.layers.Dropout(self.config.dropout_rate)(x)\n \n # Decoder layers\n decoder_layers = self.config.hidden_layers[len(self.config.hidden_layers)//2 + 1:]\n for units in decoder_layers:\n x = keras.layers.Dense(units, activation=self.config.activation)(x)\n if self.config.dropout_rate > 0:\n x = keras.layers.Dropout(self.config.dropout_rate)(x)\n \n # Output layer\n output = keras.layers.Dense(input_dim, activation=self.config.output_activation)(x)\n \n model = keras.Model(input_layer, output)\n \n # Compile\n optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)\n model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])\n \n return model\n \n def _create_callbacks(self) -> List:\n callbacks = []\n \n # Early stopping\n early_stopping = keras.callbacks.EarlyStopping(\n monitor='val_loss',\n patience=self.config.early_stopping_patience,\n restore_best_weights=True\n )\n callbacks.append(early_stopping)\n \n # Checkpointing\n if self.config.checkpointing:\n checkpoint = keras.callbacks.ModelCheckpoint(\n 'autoencoder_best.h5',\n monitor='val_loss',\n save_best_only=True,\n save_weights_only=True\n )\n callbacks.append(checkpoint)\n \n return callbacks\n \n def _validate_and_prepare_input(self, X):\n feature_names = None\n \n if isinstance(X, pd.DataFrame):\n feature_names = X.columns.tolist\n X = X.values\n elif isinstance(X, pd.Series):\n feature_names = [X.name] if X.name else [\"feature_0\"]\n X = X.values.reshape(-1, 1)\n elif isinstance(X, list):\n X = np.array(X)\n \n if X.ndim == 1:\n X = X.reshape(-1, 1)\n \n if np.any(~np.isfinite(X)):\n warnings.warn("Input contains NaN or Inf values, replacing with median")\n from sklearn.impute import SimpleImputer\n imputer = SimpleImputer(strategy='median')\n X = imputer.fit_transform(X)\n \n return X, feature_names\n \n def _create_scaler(self):\n if self.config.scaler_type == 'standard':\n self.scaler = StandardScaler\n elif self.config.scaler_type == 'minmax':\n self.scaler = MinMaxScaler\n else:\n self.scaler = None\n \n def get_statistics(self) -> Dict[str, Any]:\n if not self.fitted:\n return {\"status\": \"not_fitted\"}\n \n return {\n \"fitted\": True,\n \"hidden_layers\": self.config.hidden_layers,\n \"threshold\": float(self.threshold),\n \"n_parameters\": self.model.count_params if self.model else 0,\n \"feature_names\": self._feature_names,\n \"training_loss\": self._training_history['loss'][-1] if self._training_history else None\n }

def create_crypto_autoencoder_detector(price_data: pd.DataFrame, features: Optional[List[str]] = None) -> AutoencoderDetector:
 if features is None:\n features = ['close', 'volume']\n if 'returns' in price_data.columns:\n features.append('returns')\n \n config = AutoencoderConfig(\n hidden_layers=[32, 16, 8, 16, 32],\n epochs=50,\n batch_size=32,\n threshold_percentile=95,\n crypto_optimized=True,\n scaler_type='minmax'\n )\n \n detector = AutoencoderDetector(config)\n detector.fit(price_data[features].dropna)\n \n return detector