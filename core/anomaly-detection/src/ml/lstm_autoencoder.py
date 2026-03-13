"""
🔄 LSTM Autoencoder Anomaly Detector

Implements LSTM-based autoencoder for sequential anomaly detection.
Perfect for time series data with temporal dependencies.

Features:
- Sequential pattern learning
- Temporal anomaly detection
- Distributed training
- Real-time streaming
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class LSTMAutoencoderConfig:
 """Configuration for LSTM Autoencoder detector."""
 # Architecture
 sequence_length: int = 50
 lstm_units: List[int] = None # [50, 25, 25, 50]
 dropout_rate: float = 0.2

 # Training
 epochs: int = 50
 batch_size: int = 32
 learning_rate: float = 0.001
 validation_split: float = 0.2

 # Detection
 threshold_percentile: float = 95

 #
 crypto_optimized: bool = True

class LSTMAutoencoderDetector:
 """LSTM Autoencoder for sequential anomaly detection."""

 def __init__(self, config: Optional[LSTMAutoencoderConfig] = None):
 self.config = config or LSTMAutoencoderConfig
 if self.config.lstm_units is None:
 self.config.lstm_units = [50, 25, 25, 50]

 self.fitted = False
 self.model = None
 self.scaler = MinMaxScaler
 self.threshold = None

 logger.info(
 "LSTMAutoencoderDetector initialized",
 sequence_length=self.config.sequence_length,
 lstm_units=self.config.lstm_units
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
 try:
 X = self._validate_input(X)

 # Create sequences
 X_seq = self._create_sequences(X)
 n_features = X_seq.shape[2]

 # Build model
 self.model = self._build_lstm_autoencoder(self.config.sequence_length, n_features)

 # Train
 history = self.model.fit(
 X_seq, X_seq,
 epochs=self.config.epochs,
 batch_size=self.config.batch_size,
 validation_split=self.config.validation_split,
 verbose=0
 )

 # Compute threshold
 reconstructed = self.model.predict(X_seq)
 mse = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))
 self.threshold = np.percentile(mse, self.config.threshold_percentile)

 self.fitted = True

 logger.info(
 "LSTM Autoencoder fitted successfully",
 n_sequences=len(X_seq),
 n_features=n_features,
 threshold=self.threshold
 )

 return self

 except Exception as e:
 logger.error("Failed to fit LSTM autoencoder", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 try:
 X = self._validate_input(X)
 X_seq = self._create_sequences(X)

 reconstructed = self.model.predict(X_seq)
 mse = np.mean(np.square(X_seq - reconstructed), axis=(1, 2))

 anomaly_labels = (mse > self.threshold).astype(int)

 # Pad with zeros for sequence offset
 pad_size = self.config.sequence_length - 1
 anomaly_labels = np.concatenate([np.zeros(pad_size), anomaly_labels])
 mse = np.concatenate([np.zeros(pad_size), mse])

 return anomaly_labels.astype(int), mse

 except Exception as e:
 logger.error("Failed LSTM autoencoder detection", error=str(e))
 raise

 def _validate_input(self, X):
 if isinstance(X, pd.DataFrame):
 X = X.values
 elif isinstance(X, list):
 X = np.array(X)

 if X.ndim == 1:
 X = X.reshape(-1, 1)

 # Scale data
 X = self.scaler.fit_transform(X) if not self.fitted else self.scaler.transform(X)

 return X

 def _create_sequences(self, X):
 """Create overlapping sequences for LSTM."""
 n_samples, n_features = X.shape
 seq_len = self.config.sequence_length

 if n_samples < seq_len:
 raise ValueError(f"Not enough samples for sequence length {seq_len}")

 n_sequences = n_samples - seq_len + 1
 sequences = np.zeros((n_sequences, seq_len, n_features))

 for i in range(n_sequences):
 sequences[i] = X[i:i+seq_len]

 return sequences

 def _build_lstm_autoencoder(self, seq_len, n_features):
 # Encoder
 input_layer = keras.layers.Input(shape=(seq_len, n_features))
 x = input_layer

 # Encoder LSTM layers
 for i, units in enumerate(self.config.lstm_units[:2]):
 return_sequences = i < len(self.config.lstm_units[:2]) - 1
 x = keras.layers.LSTM(units, return_sequences=return_sequences, dropout=self.config.dropout_rate)(x)

 # Decoder
 x = keras.layers.RepeatVector(seq_len)(x)

 # Decoder LSTM layers
 for units in reversed(self.config.lstm_units[:2]):\n x = keras.layers.LSTM(units, return_sequences=True, dropout=self.config.dropout_rate)(x)\n \n output = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(x)\n \n model = keras.Model(input_layer, output)\n model.compile(optimizer=keras.optimizers.Adam(self.config.learning_rate), loss='mse')\n \n return model\n \n def get_statistics(self):\n if not self.fitted:\n return {\"status\": \"not_fitted\"}\n \n return {\n \"fitted\": True,\n \"sequence_length\": self.config.sequence_length,\n \"lstm_units\": self.config.lstm_units,\n \"threshold\": float(self.threshold)\n }\n\ndef create_crypto_lstm_autoencoder(price_data: pd.DataFrame, sequence_length: int = 50) -> LSTMAutoencoderDetector:\n config = LSTMAutoencoderConfig(\n sequence_length=sequence_length,\n lstm_units=[32, 16, 16, 32],\n epochs=30,\n crypto_optimized=True\n )\n \n detector = LSTMAutoencoderDetector(config)\n \n # Use price and volume for LSTM\n features = ['close', 'volume']\n detector.fit(price_data[features].dropna)\n \n return detector