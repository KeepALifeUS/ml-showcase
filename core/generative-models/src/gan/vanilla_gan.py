"""
Vanilla GAN Implementation for Crypto Trading Data Generation

Enterprise-grade Generative Adversarial Network implementation specifically 
designed for crypto trading data synthesis. Follows enterprise patterns for 
scalability and production readiness.

Features:
- OHLCV data generation
- Market regime-aware synthesis
- Production-ready architecture
- Enterprise logging and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import pickle
from abc import ABC, abstractmethod

from loguru import logger
import wandb
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logger.add(
    "logs/vanilla_gan_{time}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)


@dataclass
class GANConfig:
    """Configuration for Vanilla GAN model"""
    # Architecture
    input_dim: int = 100  # Size noise vector
    hidden_dims: List[int] = None
    output_dim: int = 5  # OHLCV
    sequence_length: int = 60 # 60 temporal steps
    
    # Training
    batch_size: int = 64
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    num_epochs: int = 1000
    
    # Regularization
    dropout_rate: float = 0.2
    batch_norm: bool = True
    label_smoothing: float = 0.1
    
    # Crypto-specific
    price_scaling: str = "log_return"  # "minmax", "standard", "log_return"
    volume_scaling: str = "log"
    market_regime_conditioning: bool = True
    
    # Monitor
    save_interval: int = 100
    log_interval: int = 10
    validate_interval: int = 50
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 256]


class CryptoDataset(Dataset):
    """Dataset for cryptocurrency data"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        price_columns: List[str] = None,
        volume_column: str = 'volume',
        market_regime_column: str = None
    ):
        self.sequence_length = sequence_length
        self.price_columns = price_columns or ['open', 'high', 'low', 'close']
        self.volume_column = volume_column
        self.market_regime_column = market_regime_column
        
        self.data = self._preprocess_data(data)
        self.sequences = self._create_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocessing data"""
        # data
        df = data.copy()
        
        # Computing log returns for prices
        for col in self.price_columns:
            if col in df.columns:
                df[f'{col}_log_return'] = np.log(df[col] / df[col].shift(1))
        
        # Log scaling for volume
        if self.volume_column in df.columns:
            df[f'{self.volume_column}_log'] = np.log1p(df[self.volume_column])
        
        # Removing NaN values
        df = df.dropna()
        
        return df
    
    def _create_sequences(self) -> List[np.ndarray]:
        """Create sequences for training"""
        sequences = []
        
        # Select columns for model
        feature_columns = [f'{col}_log_return' for col in self.price_columns]
        if f'{self.volume_column}_log' in self.data.columns:
            feature_columns.append(f'{self.volume_column}_log')
        
        data_values = self.data[feature_columns].values
        
        # Normalization
        self.scaler = StandardScaler()
        data_normalized = self.scaler.fit_transform(data_values)
        
        # Create sequence
        for i in range(len(data_normalized) - self.sequence_length + 1):
            sequence = data_normalized[i:i + self.sequence_length]
            sequences.append(sequence)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx])


class Generator(nn.Module):
    """Generator network for Vanilla GAN"""
    
    def __init__(self, config: GANConfig):
        super(Generator, self).__init__()
        self.config = config
        
        layers = []
        
        # Input layer
        in_features = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(in_features, config.sequence_length * config.output_dim),
            nn.Tanh()
        ])
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Generator initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        """Count parameters model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, noise):
        output = self.main(noise)
        return output.view(-1, self.config.sequence_length, self.config.output_dim)


class Discriminator(nn.Module):
    """Discriminator network for Vanilla GAN"""
    
    def __init__(self, config: GANConfig):
        super(Discriminator, self).__init__()
        self.config = config
        
        layers = []
        
        # Input layer
        in_features = config.sequence_length * config.output_dim
        for hidden_dim in reversed(config.hidden_dims):
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout_rate)
            ])
            in_features = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        ])
        
        self.main = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Discriminator initialized with {self._count_parameters()} parameters")
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def _count_parameters(self):
        """Count parameters model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input):
        input_flat = input.view(-1, self.config.sequence_length * self.config.output_dim)
        return self.main(input_flat)


class VanillaGAN:
    """
    Enterprise Vanilla GAN for generation cryptocurrency data
    
    Implements enterprise patterns for production-ready deployment:
    - Comprehensive logging and monitoring
    - Robust training pipeline
    - Quality metrics evaluation
    - Model persistence and versioning
    """
    
    def __init__(self, config: GANConfig, device: str = "auto"):
        self.config = config
        self.device = self._setup_device(device)
        
        # Initialize models
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)
        
        # Optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Metrics
        self.training_history = {
            'g_loss': [],
            'd_loss': [],
            'epoch': [],
            'timestamp': []
        }
        
        logger.info(f"VanillaGAN initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Configure device for training"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("Using CPU")
        
        return torch.device(device)
    
    def train(
        self,
        dataloader: DataLoader,
        validation_loader: Optional[DataLoader] = None
    ) -> Dict[str, List[float]]:
        """
        Training GAN model
        
        Args:
            dataloader: DataLoader for training data
            validation_loader: DataLoader for validation data
            
        Returns:
            History training
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for batch_idx, real_data in enumerate(dataloader):
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Training Discriminator
                d_loss = self._train_discriminator(real_data, batch_size)
                epoch_d_loss += d_loss
                
                # Training Generator
                g_loss = self._train_generator(batch_size)
                epoch_g_loss += g_loss
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}] "
                        f"Batch [{batch_idx}/{len(dataloader)}] "
                        f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}"
                    )
            
            # Averaging loss for epoch
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            # Save metrics
            self.training_history['g_loss'].append(avg_g_loss)
            self.training_history['d_loss'].append(avg_d_loss)
            self.training_history['epoch'].append(epoch)
            self.training_history['timestamp'].append(datetime.now().isoformat())
            
            # Validation
            if validation_loader and epoch % self.config.validate_interval == 0:
                self._validate(validation_loader)
            
            # Save model
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            logger.info(
                f"Epoch [{epoch}/{self.config.num_epochs}] completed - "
                f"Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}"
            )
        
        logger.info("Training completed successfully")
        return self.training_history
    
    def _train_discriminator(self, real_data: torch.Tensor, batch_size: int) -> float:
        """Training discriminator"""
        self.d_optimizer.zero_grad()
        
        # Real data
        real_labels = torch.ones(batch_size, 1).to(self.device)
        if self.config.label_smoothing > 0:
            real_labels -= self.config.label_smoothing * torch.rand_like(real_labels)
        
        real_output = self.discriminator(real_data)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake data
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_data = self.generator(noise)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        fake_output = self.discriminator(fake_data.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        # Total loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        return d_loss.item()
    
    def _train_generator(self, batch_size: int) -> float:
        """Training generator"""
        self.g_optimizer.zero_grad()
        
        # Generation fake data
        noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
        fake_data = self.generator(noise)
        
        # Trying fool
        fake_labels = torch.ones(batch_size, 1).to(self.device)
        fake_output = self.discriminator(fake_data)
        
        g_loss = self.criterion(fake_output, fake_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item()
    
    def _validate(self, validation_loader: DataLoader) -> Dict[str, float]:
        """Validation model"""
        self.generator.eval()
        self.discriminator.eval()
        
        val_g_loss = 0.0
        val_d_loss = 0.0
        
        with torch.no_grad():
            for real_data in validation_loader:
                real_data = real_data.to(self.device)
                batch_size = real_data.size(0)
                
                # Generator loss
                noise = torch.randn(batch_size, self.config.input_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_output = self.discriminator(fake_data)
                fake_labels = torch.ones(batch_size, 1).to(self.device)
                g_loss = self.criterion(fake_output, fake_labels)
                val_g_loss += g_loss.item()
                
                # Discriminator loss
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_output = self.discriminator(real_data)
                real_loss = self.criterion(real_output, real_labels)
                
                fake_labels_d = torch.zeros(batch_size, 1).to(self.device)
                fake_output_d = self.discriminator(fake_data)
                fake_loss = self.criterion(fake_output_d, fake_labels_d)
                
                d_loss = real_loss + fake_loss
                val_d_loss += d_loss.item()
        
        val_g_loss /= len(validation_loader)
        val_d_loss /= len(validation_loader)
        
        logger.info(f"Validation - G_loss: {val_g_loss:.4f}, D_loss: {val_d_loss:.4f}")
        
        self.generator.train()
        self.discriminator.train()
        
        return {'val_g_loss': val_g_loss, 'val_d_loss': val_d_loss}
    
    def generate_samples(
        self,
        num_samples: int,
        noise: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generation synthetic samples
        
        Args:
            num_samples: Number samples for generation
            noise: Custom noise (optionally)
            
        Returns:
            Generated data
        """
        self.generator.eval()
        
        if noise is None:
            noise = torch.randn(num_samples, self.config.input_dim).to(self.device)
        
        with torch.no_grad():
            generated_data = self.generator(noise)
        
        return generated_data.cpu().numpy()
    
    def save_checkpoint(self, filepath: str):
        """Save checkpoint model"""
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {filepath}")


def main():
    """Example use VanillaGAN"""
    
    # Configuration
    config = GANConfig(
        sequence_length=60,
        output_dim=5,  # OHLCV
        batch_size=32,
        num_epochs=100,
        learning_rate=0.0002
    )
    
    # Create synthetic data for example
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    
    # Generation OHLCV data
    prices = np.cumsum(np.random.randn(1000) * 0.01) + 100
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.randn(1000) * 0.5),
        'low': prices - np.abs(np.random.randn(1000) * 0.5),
        'close': prices + np.random.randn(1000) * 0.2,
        'volume': np.random.lognormal(mean=10, sigma=1, size=1000)
    })
    
    # Create dataset
    dataset = CryptoDataset(data, sequence_length=config.sequence_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # Training model
    gan = VanillaGAN(config)
    training_history = gan.train(dataloader)
    
    # Generation samples
    samples = gan.generate_samples(10)
    logger.info(f"Generated samples shape: {samples.shape}")
    
    # Save model
    gan.save_checkpoint("models/vanilla_gan_final.pth")


if __name__ == "__main__":
    main()