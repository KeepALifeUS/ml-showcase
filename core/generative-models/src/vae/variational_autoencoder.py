"""
Variational Autoencoder (VAE) Implementation for Crypto Trading Data

Standard VAE implementation with crypto-specific adaptations for generating
realistic financial time series data with proper statistical properties.

Features:
- Standard VAE architecture with KL divergence regularization
- Crypto market-aware data preprocessing
- Temporal consistency preservation
- Latent space interpolation capabilities
- Production-ready monitoring and logging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import json

from loguru import logger
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Configure logging
logger.add(
    "logs/vae_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


@dataclass
class VAEConfig:
    """Configuration for Variational Autoencoder"""
    # Architecture
    input_dim: int = 5  # OHLCV
    sequence_length: int = 60
    latent_dim: int = 32 # Dimension latent space
    hidden_dims: List[int] = None
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 500
    beta: float = 1.0  # Weight KL divergence
    
    # Regularization
    dropout_rate: float = 0.2
    batch_norm: bool = True
    
    # VAE specific
    kl_warmup_epochs: int = 100 # beta
    reconstruction_loss_type: str = "mse"  # "mse", "bce", "smooth_l1"
    
    # Crypto-specific
    price_scaling: str = "log_return"
    volume_scaling: str = "log"
    use_temporal_embedding: bool = True
    
    # Monitor
    save_interval: int = 50
    log_interval: int = 10
    validate_interval: int = 25
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class CryptoVAEDataset(Dataset):
    """Dataset for VAE with data"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        price_columns: List[str] = None,
        volume_column: str = 'volume'
    ):
        self.sequence_length = sequence_length
        self.price_columns = price_columns or ['open', 'high', 'low', 'close']
        self.volume_column = volume_column
        
        self.data, self.scaler = self._preprocess_data(data)
        self.sequences = self._create_sequences()
        
        logger.info(f"VAE dataset created with {len(self.sequences)} sequences")
    
    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocessing data for VAE"""
        df = data.copy()
        
        # Log returns for prices
        for col in self.price_columns:
            if col in df.columns:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        # Log scaling for volume
        if self.volume_column in df.columns:
            df[f'{self.volume_column}_log'] = np.log1p(df[self.volume_column])
        
        # Additional features
        if 'close' in df.columns:
            # Simple volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Momentum
            df['momentum'] = df['close'].pct_change(10)
        
        # Select features for model
        feature_columns = [f'{col}_log_return' for col in self.price_columns]
        if f'{self.volume_column}_log' in df.columns:
            feature_columns.append(f'{self.volume_column}_log')
        
        # Add additional features
        for feature in ['volatility', 'momentum']:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # Removing NaN
        df_clean = df[feature_columns].dropna()
        
        # Standardization
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(df_clean.values)
        
        df_normalized = pd.DataFrame(data_normalized, columns=feature_columns)
        
        return df_normalized, scaler
    
    def _create_sequences(self) -> List[np.ndarray]:
        """Create sequences for VAE"""
        sequences = []
        data_values = self.data.values
        
        for i in range(len(data_values) - self.sequence_length + 1):
            sequence = data_values[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])


class VAEEncoder(nn.Module):
    """Encoder network for VAE"""
    
    def __init__(self, config: VAEConfig):
        super(VAEEncoder, self).__init__()
        self.config = config
        
        # Flatten input for fully connected layers
        input_size = config.sequence_length * config.input_dim
        
        layers = []
        
        # Main layers
        in_features = input_size
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        self.main = nn.Sequential(*layers)
        
        # Final layers for mu and log_var
        self.mu_layer = nn.Linear(in_features, config.latent_dim)
        self.logvar_layer = nn.Linear(in_features, config.latent_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"VAE Encoder initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        # Flatten input
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Main layers
        features = self.main(x_flat)
        
        # Getting parameters distribution
        mu = self.mu_layer(features)
        logvar = self.logvar_layer(features)
        
        return mu, logvar


class VAEDecoder(nn.Module):
    """Decoder network for VAE"""
    
    def __init__(self, config: VAEConfig):
        super(VAEDecoder, self).__init__()
        self.config = config
        
        # Output size
        output_size = config.sequence_length * config.input_dim
        
        layers = []
        
        # Starting with latent space
        in_features = config.latent_dim
        hidden_dims_reversed = list(reversed(config.hidden_dims))
        
        for hidden_dim in hidden_dims_reversed:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Final layer
        layers.extend([
            nn.Linear(in_features, output_size),
            # Without activation on for regression tasks
        ])
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"VAE Decoder initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, z):
        # Decoder forward pass
        output = self.main(z)
        
        # Reshape to original
        batch_size = z.size(0)
        output = output.view(batch_size, self.config.sequence_length, self.config.input_dim)
        
        return output


class VariationalAutoencoder:
    """
    Variational Autoencoder for cryptocurrency temporal series
    
    Implements:
    - Standard VAE with KL divergence regularization
    - Reparameterization trick
    - Beta scheduling for KL warmup
    - Crypto-specific data handling
    - Latent space exploration utilities
    """
    
    def __init__(self, config: VAEConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize encoder and decoder
        self.encoder = VAEEncoder(config).to(self.device)
        self.decoder = VAEDecoder(config).to(self.device)
        
        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(params, lr=config.learning_rate)
        
        # Scheduler learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Metrics
        self.training_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'kl_loss': [],
            'beta': [],
            'epoch': [],
            'timestamp': []
        }
        
        logger.info(f"VAE initialized on {self.device}")
        logger.info(f"Latent dimension: {config.latent_dim}")
    
    def _setup_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        
        return torch.device(device)
    
    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def _reconstruction_loss(self, recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Computation reconstruction loss"""
        if self.config.reconstruction_loss_type == "mse":
            return F.mse_loss(recon_x, x, reduction='sum')
        elif self.config.reconstruction_loss_type == "bce":
            return F.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='sum')
        elif self.config.reconstruction_loss_type == "smooth_l1":
            return F.smooth_l1_loss(recon_x, x, reduction='sum')
        else:
            return F.mse_loss(recon_x, x, reduction='sum')
    
    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence from distribution"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def _get_beta(self, epoch: int) -> float:
        """Beta scheduling for KL warmup"""
        if epoch < self.config.kl_warmup_epochs:
            # warmup
            return self.config.beta * (epoch / self.config.kl_warmup_epochs)
        else:
            return self.config.beta
    
    def train(
        self,
        dataloader: DataLoader,
        validation_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """Training VAE"""
        logger.info(f"Starting VAE training for {self.config.num_epochs} epochs")
        
        self.encoder.train()
        self.decoder.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            
            # Current beta for KL warmup
            beta = self._get_beta(epoch)
            
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Encoder
                mu, logvar = self.encoder(data)
                
                # Reparameterization
                z = self._reparameterize(mu, logvar)
                
                # Decoder
                recon_data = self.decoder(z)
                
                # Loss computation
                recon_loss = self._reconstruction_loss(recon_data, data)
                kl_loss = self._kl_divergence(mu, logvar)
                
                # Total loss with beta weighting
                total_loss = recon_loss + beta * kl_loss
                
                # Normalization by batch size
                total_loss = total_loss / batch_size
                recon_loss = recon_loss / batch_size
                kl_loss = kl_loss / batch_size
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=1.0
                )
                
                self.optimizer.step()
                
                # metrics
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {total_loss.item():.4f} "
                        f"Recon: {recon_loss.item():.4f} "
                        f"KL: {kl_loss.item():.4f} "
                        f"Beta: {beta:.4f}"
                    )
            
            # Averaging metrics for epoch
            num_batches = len(dataloader)
            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            
            # Save metrics
            self.training_history['total_loss'].append(avg_total_loss)
            self.training_history['reconstruction_loss'].append(avg_recon_loss)
            self.training_history['kl_loss'].append(avg_kl_loss)
            self.training_history['beta'].append(beta)
            self.training_history['epoch'].append(epoch)
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Learning rate scheduling
            self.scheduler.step(avg_total_loss)
            
            # Validation
            if validation_loader and epoch % self.config.validate_interval == 0:
                val_metrics = self._validate(validation_loader, beta)
                logger.info(f"Validation - Total: {val_metrics['total_loss']:.4f}, "
                           f"Recon: {val_metrics['recon_loss']:.4f}, "
                           f"KL: {val_metrics['kl_loss']:.4f}")
            
            # Save model
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"vae_checkpoint_epoch_{epoch}.pth")
            
            logger.info(
                f"Epoch [{epoch}/{self.config.num_epochs}] completed - "
                f"Total: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                f"KL: {avg_kl_loss:.4f}, Beta: {beta:.4f}"
            )
        
        logger.info("VAE training completed successfully")
        return self.training_history
    
    def _validate(self, validation_loader: DataLoader, beta: float) -> Dict[str, float]:
        """Validation model"""
        self.encoder.eval()
        self.decoder.eval()
        
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0
        
        with torch.no_grad():
            for data in validation_loader:
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Forward pass
                mu, logvar = self.encoder(data)
                z = self._reparameterize(mu, logvar)
                recon_data = self.decoder(z)
                
                # Loss computation
                recon_loss = self._reconstruction_loss(recon_data, data) / batch_size
                kl_loss = self._kl_divergence(mu, logvar) / batch_size
                total_loss = recon_loss + beta * kl_loss
                
                val_total_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()
        
        num_batches = len(validation_loader)
        val_total_loss /= num_batches
        val_recon_loss /= num_batches
        val_kl_loss /= num_batches
        
        self.encoder.train()
        self.decoder.train()
        
        return {
            'total_loss': val_total_loss,
            'recon_loss': val_recon_loss,
            'kl_loss': val_kl_loss
        }
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode data in latent space"""
        self.encoder.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decoding from latent space"""
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(z)
    
    def generate_samples(self, num_samples: int, z: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Generation new samples
        
        Args:
            num_samples: Number samples for generation
            z: vectors (if None, are generated from N(0,1))
        """
        self.decoder.eval()
        
        if z is None:
            z = torch.randn(num_samples, self.config.latent_dim).to(self.device)
        
        with torch.no_grad():
            generated_data = self.decoder(z)
        
        return generated_data.cpu().numpy()
    
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Interpolation between two in
        
        Args:
            x1, x2: Source data for interpolation
            num_steps: Number steps interpolation
        """
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # in latent space
            mu1, _ = self.encoder(x1)
            mu2, _ = self.encoder(x2)
            
            # Create
            interpolated_samples = []
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                x_interp = self.decoder(z_interp)
                interpolated_samples.append(x_interp.cpu().numpy())
        
        return np.array(interpolated_samples)
    
    def latent_space_analysis(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """Analysis latent space"""
        self.encoder.eval()
        
        latent_vectors = []
        reconstructions = []
        originals = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                mu, logvar = self.encoder(data)
                z = self._reparameterize(mu, logvar)
                recon = self.decoder(z)
                
                latent_vectors.append(mu.cpu().numpy())
                reconstructions.append(recon.cpu().numpy())
                originals.append(data.cpu().numpy())
        
        return {
            'latent_vectors': np.concatenate(latent_vectors, axis=0),
            'reconstructions': np.concatenate(reconstructions, axis=0),
            'originals': np.concatenate(originals, axis=0)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save checkpoint VAE"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"VAE checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint VAE"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"VAE checkpoint loaded from {filepath}")


def main():
    """Example use VAE"""
    
    # Configuration VAE
    config = VAEConfig(
        sequence_length=60,
        input_dim=5,  # OHLCV
        latent_dim=32,
        hidden_dims=[256, 128, 64],
        batch_size=32,
        num_epochs=200,
        learning_rate=0.001,
        beta=1.0,
        kl_warmup_epochs=50,
        reconstruction_loss_type="mse"
    )
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2000, freq='1H')
    
    # Simulation crypto data with different regimes
    prices = []
    base_price = 1000
    
    for i in range(2000):
        # Different market conditions
        if i < 500:
            # Trending market
            drift = 0.0005
            volatility = 0.01
        elif i < 1000:
            # High volatility
            drift = 0
            volatility = 0.03
        elif i < 1500:
            # Bear market
            drift = -0.0003
            volatility = 0.015
        else:
            # Recovery
            drift = 0.0008
            volatility = 0.02
        
        change = np.random.normal(drift, volatility)
        base_price *= (1 + change)
        prices.append(base_price)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
        'volume': np.random.lognormal(mean=12, sigma=1, size=2000)
    })
    
    # Create VAE dataset
    dataset = CryptoVAEDataset(data, sequence_length=config.sequence_length)
    
    # Split on train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Training VAE
    vae = VariationalAutoencoder(config)
    training_history = vae.train(train_dataloader, val_dataloader)
    
    # Generation new samples
    generated_samples = vae.generate_samples(num_samples=10)
    logger.info(f"Generated samples shape: {generated_samples.shape}")
    
    # Analysis latent space
    latent_analysis = vae.latent_space_analysis(val_dataloader)
    logger.info(f"Latent vectors shape: {latent_analysis['latent_vectors'].shape}")
    
    # Interpolation between samples
    if len(val_dataset) >= 2:
        sample1 = val_dataset[0].unsqueeze(0).to(vae.device)
        sample2 = val_dataset[1].unsqueeze(0).to(vae.device)
        interpolation = vae.interpolate(sample1, sample2, num_steps=5)
        logger.info(f"Interpolation shape: {interpolation.shape}")
    
    # Save model
    vae.save_checkpoint("models/vae_final.pth")
    
    # Analysis results
    final_metrics = training_history
    logger.info("VAE training metrics:")
    logger.info(f"Final Total Loss: {final_metrics['total_loss'][-1]:.4f}")
    logger.info(f"Final Reconstruction Loss: {final_metrics['reconstruction_loss'][-1]:.4f}")
    logger.info(f"Final KL Loss: {final_metrics['kl_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()