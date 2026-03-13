"""
🎲 Variational Autoencoder (VAE) Anomaly Detector

Implements VAE for probabilistic anomaly detection.
Uses latent space probability distribution for anomaly scoring.

Features:
- Probabilistic latent representations
- Uncertainty quantification
- Generative anomaly modeling
- Distributed inference
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import structlog
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

logger = structlog.get_logger(__name__)

@dataclass
class VAEConfig:
 """Configuration for VAE detector."""
 latent_dim: int = 10
 encoder_layers: List[int] = None # [64, 32]
 decoder_layers: List[int] = None # [32, 64]
 beta: float = 1.0 # KL divergence weight

 epochs: int = 100
 batch_size: int = 32
 learning_rate: float = 0.001

 threshold_percentile: float = 95
 crypto_optimized: bool = True

class VAEDetector:
 """Variational Autoencoder Anomaly Detector."""

 def __init__(self, config: Optional[VAEConfig] = None):
 self.config = config or VAEConfig
 if self.config.encoder_layers is None:
 self.config.encoder_layers = [64, 32]
 if self.config.decoder_layers is None:
 self.config.decoder_layers = [32, 64]

 self.fitted = False
 self.vae = None
 self.encoder = None
 self.decoder = None
 self.scaler = StandardScaler
 self.threshold = None

 logger.info(
 "VAEDetector initialized",
 latent_dim=self.config.latent_dim,
 beta=self.config.beta
 )

 def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
 try:
 X = self._validate_input(X)
 input_dim = X.shape[1]

 # Build VAE
 self.vae, self.encoder, self.decoder = self._build_vae(input_dim)

 # Train
 history = self.vae.fit(
 X, X,
 epochs=self.config.epochs,
 batch_size=self.config.batch_size,
 verbose=0
 )

 # Compute threshold using reconstruction probability
 z_mean, z_log_var, z = self.encoder.predict(X)
 x_reconstructed = self.decoder.predict(z)

 # Reconstruction loss + KL divergence
 reconstruction_loss = np.mean(np.square(X - x_reconstructed), axis=1)
 kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
 total_loss = reconstruction_loss + self.config.beta * kl_loss

 self.threshold = np.percentile(total_loss, self.config.threshold_percentile)

 self.fitted = True

 logger.info(
 "VAE fitted successfully",
 n_samples=len(X),
 input_dim=input_dim,
 threshold=self.threshold
 )

 return self

 except Exception as e:
 logger.error("Failed to fit VAE", error=str(e))
 raise

 def detect(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
 if not self.fitted:
 raise ValueError("Detector must be fitted")

 try:
 X = self._validate_input(X)

 z_mean, z_log_var, z = self.encoder.predict(X)
 x_reconstructed = self.decoder.predict(z)

 reconstruction_loss = np.mean(np.square(X - x_reconstructed), axis=1)
 kl_loss = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
 anomaly_scores = reconstruction_loss + self.config.beta * kl_loss

 anomaly_labels = (anomaly_scores > self.threshold).astype(int)

 return anomaly_labels, anomaly_scores

 except Exception as e:
 logger.error("Failed VAE detection", error=str(e))
 raise

 def _validate_input(self, X):
 if isinstance(X, pd.DataFrame):
 X = X.values
 elif isinstance(X, list):
 X = np.array(X)

 if X.ndim == 1:
 X = X.reshape(-1, 1)

 if np.any(~np.isfinite(X)):
 warnings.warn("Input contains NaN or Inf values, replacing with mean")
 from sklearn.impute import SimpleImputer
 imputer = SimpleImputer(strategy='mean')
 X = imputer.fit_transform(X)

 # Scale data
 X = self.scaler.fit_transform(X) if not self.fitted else self.scaler.transform(X)

 return X

 def _sampling(self, args):
 \"\"\"Reparameterization trick.\"\"\"\n z_mean, z_log_var = args\n batch = tf.shape(z_mean)[0]\n dim = tf.shape(z_mean)[1]\n epsilon = tf.keras.backend.random_normal(shape=(batch, dim))\n return z_mean + tf.exp(0.5 * z_log_var) * epsilon\n \n def _build_vae(self, input_dim):\n # Encoder\n encoder_input = keras.layers.Input(shape=(input_dim,))\n x = encoder_input\n \n for units in self.config.encoder_layers:\n x = keras.layers.Dense(units, activation='relu')(x)\n \n z_mean = keras.layers.Dense(self.config.latent_dim, name='z_mean')(x)\n z_log_var = keras.layers.Dense(self.config.latent_dim, name='z_log_var')(x)\n z = keras.layers.Lambda(self._sampling, name='z')([z_mean, z_log_var])\n \n encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name='encoder')\n \n # Decoder\n decoder_input = keras.layers.Input(shape=(self.config.latent_dim,))\n x = decoder_input\n \n for units in self.config.decoder_layers:\n x = keras.layers.Dense(units, activation='relu')(x)\n \n decoder_output = keras.layers.Dense(input_dim, activation='linear')(x)\n decoder = keras.Model(decoder_input, decoder_output, name='decoder')\n \n # VAE\n vae_output = decoder(z)\n vae = keras.Model(encoder_input, vae_output, name='vae')\n \n # Loss\n def vae_loss(y_true, y_pred):\n reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)\n kl_loss = -0.5 * tf.reduce_mean(\n 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1\n )\n return reconstruction_loss + self.config.beta * kl_loss\n \n vae.compile(optimizer=keras.optimizers.Adam(self.config.learning_rate), loss=vae_loss)\n \n return vae, encoder, decoder\n \n def get_statistics(self):\n if not self.fitted:\n return {\"status\": \"not_fitted\"}\n \n return {\n \"fitted\": True,\n \"latent_dim\": self.config.latent_dim,\n \"beta\": self.config.beta,\n \"threshold\": float(self.threshold)\n }\n\ndef create_crypto_vae_detector(price_data: pd.DataFrame, features: Optional[List[str]] = None) -> VAEDetector:\n if features is None:\n features = ['close', 'volume']\n if 'returns' in price_data.columns:\n features.append('returns')\n \n config = VAEConfig(\n latent_dim=8,\n encoder_layers=[32, 16],\n decoder_layers=[16, 32],\n epochs=50,\n beta=1.0,\n crypto_optimized=True\n )\n \n detector = VAEDetector(config)\n detector.fit(price_data[features].dropna)\n \n return detector