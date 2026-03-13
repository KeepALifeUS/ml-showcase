"""
TimeGAN Implementation for Crypto Time Series Generation
Generates realistic synthetic crypto market data preserving temporal dynamics
Based on: https://proceedings.neurips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TimeGANConfig:
    """Configuration for TimeGAN"""
    # Data dimensions
    seq_len: int = 24  # Sequence length (e.g., 24 hours)
    n_features: int = 5  # Number of features (OHLCV)
    hidden_dim: int = 128  # Hidden dimension
    n_layers: int = 3  # Number of RNN layers
    
    # Training parameters
    batch_size: int = 128
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    gamma: float = 1.0  # Loss weight for supervised loss
    
    # Architecture choices
    module: str = 'gru'  # 'gru', 'lstm', 'rnn'
    bidirectional: bool = False
    
    # enterprise patterns
    enable_monitoring: bool = True
    checkpoint_frequency: int = 50
    use_cuda: bool = torch.cuda.is_available()
    mixed_precision: bool = True
    gradient_penalty: bool = True
    gradient_penalty_weight: float = 10.0


class EmbeddingNetwork(nn.Module):
    """Embedding network: maps input to latent space"""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # Choose RNN type
        if config.module == 'gru':
            self.rnn = nn.GRU(
                config.n_features,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.module == 'lstm':
            self.rnn = nn.LSTM(
                config.n_features,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                config.n_features,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        
        # Linear layer for output
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.linear = nn.Linear(rnn_output_dim, config.hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through embedding network
        
        Args:
            x: Input tensor [batch_size, seq_len, n_features]
        
        Returns:
            Tuple of (embedded sequence, final hidden state)
        """
        # RNN forward pass
        h, h_final = self.rnn(x)
        
        # Linear transformation
        h = self.linear(h)
        h = self.activation(h)
        
        return h, h_final


class RecoveryNetwork(nn.Module):
    """Recovery network: maps from latent space back to data space"""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # RNN for sequence generation
        if config.module == 'gru':
            self.rnn = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.module == 'lstm':
            self.rnn = nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        
        # Output layer
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.linear = nn.Linear(rnn_output_dim, config.n_features)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through recovery network
        
        Args:
            h: Latent representation [batch_size, seq_len, hidden_dim]
        
        Returns:
            Reconstructed data [batch_size, seq_len, n_features]
        """
        # RNN forward pass
        h_rec, _ = self.rnn(h)
        
        # Linear transformation to data space
        x_rec = self.linear(h_rec)
        
        return x_rec


class GeneratorNetwork(nn.Module):
    """Generator network: generates synthetic latent sequences"""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # RNN for sequence generation
        if config.module == 'gru':
            self.rnn = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.module == 'lstm':
            self.rnn = nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        
        # Output layer
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.linear = nn.Linear(rnn_output_dim, config.hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator network
        
        Args:
            z: Random noise [batch_size, seq_len, hidden_dim]
        
        Returns:
            Generated latent sequence [batch_size, seq_len, hidden_dim]
        """
        # RNN forward pass
        e_gen, _ = self.rnn(z)
        
        # Linear transformation
        e_gen = self.linear(e_gen)
        e_gen = self.activation(e_gen)
        
        return e_gen


class SupervisorNetwork(nn.Module):
    """Supervisor network: predicts next step in latent space"""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # RNN for supervision
        if config.module == 'gru':
            self.rnn = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers - 1,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.module == 'lstm':
            self.rnn = nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers - 1,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers - 1,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        
        # Output layer
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.linear = nn.Linear(rnn_output_dim, config.hidden_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through supervisor network
        
        Args:
            h: Latent sequence [batch_size, seq_len, hidden_dim]
        
        Returns:
            Supervised latent sequence [batch_size, seq_len, hidden_dim]
        """
        # RNN forward pass
        s, _ = self.rnn(h)
        
        # Linear transformation
        s = self.linear(s)
        s = self.activation(s)
        
        return s


class DiscriminatorNetwork(nn.Module):
    """Discriminator network: distinguishes real from synthetic sequences"""
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # RNN for discrimination
        if config.module == 'gru':
            self.rnn = nn.GRU(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        elif config.module == 'lstm':
            self.rnn = nn.LSTM(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        else:
            self.rnn = nn.RNN(
                config.hidden_dim,
                config.hidden_dim,
                config.n_layers,
                batch_first=True,
                bidirectional=config.bidirectional
            )
        
        # Output layer
        rnn_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.linear = nn.Linear(rnn_output_dim, 1)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator network
        
        Args:
            h: Latent sequence [batch_size, seq_len, hidden_dim]
        
        Returns:
            Discrimination scores [batch_size, seq_len, 1]
        """
        # RNN forward pass
        d, _ = self.rnn(h)
        
        # Linear transformation to scores
        y = self.linear(d)
        
        return y


class TimeGAN(nn.Module):
    """
    TimeGAN: Time-series Generative Adversarial Network
    Generates realistic synthetic crypto market data
    """
    
    def __init__(self, config: TimeGANConfig):
        super().__init__()
        self.config = config
        
        # Initialize networks
        self.embedding = EmbeddingNetwork(config)
        self.recovery = RecoveryNetwork(config)
        self.generator = GeneratorNetwork(config)
        self.supervisor = SupervisorNetwork(config)
        self.discriminator = DiscriminatorNetwork(config)
        
        # Move to device
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        self.to(self.device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Initialize optimizers (will be set during training)
        self.optimizer_embedding = None
        self.optimizer_generator = None
        self.optimizer_supervisor = None
        self.optimizer_discriminator = None
        
        # Training history
        self.history = {
            'embedding_loss': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'supervisor_loss': []
        }
    
    def _gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP
        
        Args:
            real_data: Real data samples
            fake_data: Generated fake samples
        
        Returns:
            Gradient penalty scalar
        """
        batch_size = real_data.size(0)
        
        # Random weight for interpolation
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)
        
        # Interpolate between real and fake
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Get discriminator output
        d_interpolated = self.discriminator(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_embedding(
        self,
        data_loader: DataLoader,
        n_epochs: int = 1000
    ):
        """
        Stage 1: Train embedding and recovery networks
        
        Args:
            data_loader: DataLoader for real data
            n_epochs: Number of training epochs
        """
        logger.info("Stage 1: Training Embedding and Recovery Networks")
        
        # Initialize optimizer
        self.optimizer_embedding = optim.Adam(
            list(self.embedding.parameters()) + list(self.recovery.parameters()),
            lr=self.config.learning_rate
        )
        
        for epoch in tqdm(range(n_epochs), desc="Embedding Training"):
            total_loss = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                batch_data = batch_data.to(self.device)
                
                # Forward pass
                h, _ = self.embedding(batch_data)
                x_rec = self.recovery(h)
                
                # Reconstruction loss
                loss = self.mse_loss(x_rec, batch_data)
                
                # Backward pass
                self.optimizer_embedding.zero_grad()
                loss.backward()
                self.optimizer_embedding.step()
                
                total_loss += loss.item()
            
            # Log progress
            avg_loss = total_loss / len(data_loader)
            self.history['embedding_loss'].append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    def train_supervisor(
        self,
        data_loader: DataLoader,
        n_epochs: int = 1000
    ):
        """
        Stage 2: Train supervisor network
        
        Args:
            data_loader: DataLoader for real data
            n_epochs: Number of training epochs
        """
        logger.info("Stage 2: Training Supervisor Network")
        
        # Initialize optimizer
        self.optimizer_supervisor = optim.Adam(
            self.supervisor.parameters(),
            lr=self.config.learning_rate
        )
        
        for epoch in tqdm(range(n_epochs), desc="Supervisor Training"):
            total_loss = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                batch_data = batch_data.to(self.device)
                
                # Get embeddings
                with torch.no_grad():
                    h, _ = self.embedding(batch_data)
                
                # Supervisor forward pass
                h_sup = self.supervisor(h)
                
                # Supervised loss (next-step prediction)
                loss = self.mse_loss(h_sup[:, :-1, :], h[:, 1:, :])
                
                # Backward pass
                self.optimizer_supervisor.zero_grad()
                loss.backward()
                self.optimizer_supervisor.step()
                
                total_loss += loss.item()
            
            # Log progress
            avg_loss = total_loss / len(data_loader)
            self.history['supervisor_loss'].append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    def train_joint(
        self,
        data_loader: DataLoader,
        n_epochs: int = 1000
    ):
        """
        Stage 3: Joint training of generator, discriminator, and supervisor
        
        Args:
            data_loader: DataLoader for real data
            n_epochs: Number of training epochs
        """
        logger.info("Stage 3: Joint Training")
        
        # Initialize optimizers
        self.optimizer_generator = optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.config.learning_rate
        )
        self.optimizer_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate
        )
        
        for epoch in tqdm(range(n_epochs), desc="Joint Training"):
            g_loss_total = 0
            d_loss_total = 0
            
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    batch_data = batch_data[0]
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)
                
                # Get real embeddings
                with torch.no_grad():
                    h_real, _ = self.embedding(batch_data)
                
                # Generate fake data
                z = torch.randn(batch_size, self.config.seq_len, self.config.hidden_dim).to(self.device)
                e_gen = self.generator(z)
                h_gen = self.supervisor(e_gen)
                
                # Train Discriminator
                self.optimizer_discriminator.zero_grad()
                
                # Real data discrimination
                y_real = self.discriminator(h_real)
                y_real_loss = self.bce_loss(y_real, torch.ones_like(y_real))
                
                # Fake data discrimination
                y_fake = self.discriminator(h_gen.detach())
                y_fake_loss = self.bce_loss(y_fake, torch.zeros_like(y_fake))
                
                # Total discriminator loss
                d_loss = y_real_loss + y_fake_loss
                
                # Add gradient penalty if enabled
                if self.config.gradient_penalty:
                    gp = self._gradient_penalty(h_real, h_gen.detach())
                    d_loss += self.config.gradient_penalty_weight * gp
                
                d_loss.backward()
                self.optimizer_discriminator.step()
                
                # Train Generator
                self.optimizer_generator.zero_grad()
                
                # Adversarial loss
                y_fake_gen = self.discriminator(h_gen)
                g_loss_adv = self.bce_loss(y_fake_gen, torch.ones_like(y_fake_gen))
                
                # Supervised loss
                g_loss_sup = self.mse_loss(h_gen[:, 1:, :], e_gen[:, :-1, :])
                
                # Moment matching loss (optional)
                g_loss_moment = torch.abs(h_real.mean() - h_gen.mean()) + \
                               torch.abs(h_real.std() - h_gen.std())
                
                # Total generator loss
                g_loss = g_loss_adv + self.config.gamma * g_loss_sup + 0.1 * g_loss_moment
                
                g_loss.backward()
                self.optimizer_generator.step()
                
                g_loss_total += g_loss.item()
                d_loss_total += d_loss.item()
            
            # Log progress
            avg_g_loss = g_loss_total / len(data_loader)
            avg_d_loss = d_loss_total / len(data_loader)
            self.history['generator_loss'].append(avg_g_loss)
            self.history['discriminator_loss'].append(avg_d_loss)
            
            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch {epoch+1}/{n_epochs}, G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")
    
    def fit(
        self,
        data: np.ndarray,
        embedding_epochs: int = 1000,
        supervisor_epochs: int = 1000,
        joint_epochs: int = 1000
    ):
        """
        Complete TimeGAN training pipeline
        
        Args:
            data: Training data [n_samples, seq_len, n_features]
            embedding_epochs: Epochs for embedding training
            supervisor_epochs: Epochs for supervisor training
            joint_epochs: Epochs for joint training
        """
        # Prepare data
        data_tensor = torch.FloatTensor(data)
        dataset = TensorDataset(data_tensor)
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Three-stage training
        self.train_embedding(data_loader, embedding_epochs)
        self.train_supervisor(data_loader, supervisor_epochs)
        self.train_joint(data_loader, joint_epochs)
        
        logger.info("TimeGAN training complete!")
    
    def generate(
        self,
        n_samples: int,
        seq_len: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic time series data
        
        Args:
            n_samples: Number of samples to generate
            seq_len: Sequence length (uses config default if None)
        
        Returns:
            Generated data [n_samples, seq_len, n_features]
        """
        self.eval()
        
        if seq_len is None:
            seq_len = self.config.seq_len
        
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(n_samples, seq_len, self.config.hidden_dim).to(self.device)
            
            # Generate synthetic embeddings
            e_gen = self.generator(z)
            h_gen = self.supervisor(e_gen)
            
            # Recover to data space
            x_gen = self.recovery(h_gen)
            
            # Convert to numpy
            synthetic_data = x_gen.cpu().numpy()
        
        return synthetic_data
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'config': self.config,
            'embedding_state': self.embedding.state_dict(),
            'recovery_state': self.recovery.state_dict(),
            'generator_state': self.generator.state_dict(),
            'supervisor_state': self.supervisor.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.config = checkpoint['config']
        self.embedding.load_state_dict(checkpoint['embedding_state'])
        self.recovery.load_state_dict(checkpoint['recovery_state'])
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.supervisor.load_state_dict(checkpoint['supervisor_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.history = checkpoint['history']
        
        logger.info(f"Model loaded from {path}")


def create_crypto_timegan(
    seq_len: int = 24,
    n_features: int = 5,
    hidden_dim: int = 128
) -> TimeGAN:
    """
    Factory function to create TimeGAN for crypto data
    
    Args:
        seq_len: Sequence length (time steps)
        n_features: Number of features (OHLCV)
        hidden_dim: Hidden dimension size
    
    Returns:
        Configured TimeGAN model
    """
    config = TimeGANConfig(
        seq_len=seq_len,
        n_features=n_features,
        hidden_dim=hidden_dim,
        n_layers=3,
        batch_size=128,
        module='gru',
        gradient_penalty=True
    )
    
    return TimeGAN(config)