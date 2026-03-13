"""
Graph Convolutional Network (GCN) Implementation for Crypto Trading
==================================================================

Enterprise-grade GCN implementation optimized for crypto market analysis
with cloud-native patterns and production-ready features.

Features:
- Spectral and spatial graph convolutions
- Multi-layer GCN with residual connections  
- Dropout and batch normalization for regularization
- Crypto-specific feature engineering
- Scalable distributed training support
- Real-time inference capabilities

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GCNConfig:
    """
    Configuration for Graph Convolutional Network
    
    Configuration as Code
    """
    input_dim: int = 64  # Dimension input features
    hidden_dims: List[int] = None  # Dimensions hidden layers
    output_dim: int = 1 # Dimension output (price, return)
    num_layers: int = 3  # Number GCN layers
    dropout_rate: float = 0.2 # Fraction dropout for regularization
    activation: str = 'relu'  # Function activation
    use_batch_norm: bool = True # Use BatchNorm
    use_residual: bool = True # Use residual connections
    use_edge_weights: bool = True  # Account for weights edges
    learning_rate: float = 0.001 # Speed training
    weight_decay: float = 1e-5  # L2 regularization
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Automatic architecture creation with decreasing dimensions
            self.hidden_dims = [
                max(32, self.input_dim // (2**i)) 
                for i in range(self.num_layers - 1)
            ]

class GraphConvolutionalNetwork(nn.Module):
    """
    Production-Ready Graph Convolutional Network
    
    Implements multi-layer GCN with regularization
    and enterprise patterns for crypto trading.
    """
    
    def __init__(self, config: GCNConfig):
        super(GraphConvolutionalNetwork, self).__init__()
        self.config = config
        
        # Validation configuration
        self._validate_config()
        
        # Initialize layers
        self._build_layers()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized GCN with {self.config.num_layers} layers")
    
    def _validate_config(self) -> None:
        """Validation configuration model"""
        if self.config.input_dim <= 0:
            raise ValueError("input_dim should be positive")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim should be positive")
        if not 0.0 <= self.config.dropout_rate <= 1.0:
            raise ValueError("dropout_rate should be in range [0, 1]")
        if self.config.num_layers < 1:
            raise ValueError("num_layers should be >= 1")
    
    def _build_layers(self) -> None:
        """Build architecture network"""
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Dimensions all layers
        all_dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        
        # Create GCN layers
        for i in range(len(all_dims) - 1):
            self.convs.append(
                GCNConv(
                    in_channels=all_dims[i],
                    out_channels=all_dims[i + 1],
                    improved=True, # normalization
                    cached=True, # Caching for
                    add_self_loops=True,
                    normalize=True
                )
            )
            
            # Batch Normalization for training
            if self.config.use_batch_norm and i < len(all_dims) - 2:
                self.batch_norms.append(BatchNorm(all_dims[i + 1]))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Function activation
        self.activation = self._get_activation()
        
        # Final layer for classification/regression
        self.final_layer = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim),
            nn.BatchNorm1d(self.config.output_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(self.config.output_dim, 1)
        )
    
    def _get_activation(self) -> nn.Module:
        """Get functions activation"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.01),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(self.config.activation, nn.ReLU())
    
    def _initialize_weights(self) -> None:
        """Initialize weights network"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, GCNConv)):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Direct pass through GCN
        
        Args:
            data: PyG Data object with node features, edge indices and edge weights
            
        Returns:
            torch.Tensor: Output predictions
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None) if self.config.use_edge_weights else None
        batch = getattr(data, 'batch', None)
        
        # Residual connections for networks
        residual_x = x if self.config.use_residual else None
        
        # Pass through GCN layers
        for i, conv in enumerate(self.convs[:-1]): # All layers except last
            x = conv(x, edge_index, edge_weight=edge_weight)
            
            # Batch normalization
            if self.config.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Residual connection
            if (self.config.use_residual and residual_x is not None 
                and x.shape == residual_x.shape):
                x = x + residual_x
                residual_x = x
            
            # Dropout
            x = self.dropout(x)
        
        # Last layer without activation
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        
        # Graph-level pooling for obtaining one predictions on graph
        if batch is not None:
            # Batch processing - averaging by in each
            x = global_mean_pool(x, batch)
        else:
            # Single graph - averaging by all
            x = torch.mean(x, dim=0, keepdim=True)
        
        # Final prediction
        output = self.final_layer(x)
        
        return output
    
    def get_embeddings(self, data: Data, layer_idx: int = -2) -> torch.Tensor:
        """
        Get embeddings from layers
        
        Args:
            data: Input data graph
            layer_idx: Index layers for embeddings
            
        Returns:
            torch.Tensor: Node embeddings
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None) if self.config.use_edge_weights else None
        
        # Pass up to specified layers
        for i in range(min(len(self.convs), layer_idx + 1)):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            
            if i < len(self.convs) - 1: # Not activation to layer
                if self.config.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        
        return x

class CryptoGCNTrainer:
    """
    Trainer for GCN model with
    
    Enterprise Training Pipeline
    """
    
    def __init__(self, model: GraphConvolutionalNetwork, config: GCNConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler learning rate
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,
            verbose=True
        )
        
        # History training
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
        self.model.to(self.device)
        logger.info(f"Model on : {self.device}")
    
    def train_step(self, batch: Data) -> Dict[str, float]:
        """One step training"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Direct pass
        predictions = self.model(batch)
        targets = batch.y.view(-1, 1).float()
        
        # loss
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        
        # pass
        mse_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item()
        }
    
    def validate_step(self, batch: Data) -> Dict[str, float]:
        """Validation model"""
        self.model.eval()
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch)
            targets = batch.y.view(-1, 1).float()
            
            mse_loss = F.mse_loss(predictions, targets)
            mae_loss = F.l1_loss(predictions, targets)
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item()
        }
    
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Training one epochs"""
        train_losses = []
        train_maes = []
        
        for batch in train_loader:
            metrics = self.train_step(batch)
            train_losses.append(metrics['loss'])
            train_maes.append(metrics['mae'])
        
        epoch_metrics = {
            'train_loss': np.mean(train_losses),
            'train_mae': np.mean(train_maes)
        }
        
        # Validation
        if val_loader is not None:
            val_losses = []
            val_maes = []
            
            for batch in val_loader:
                metrics = self.validate_step(batch)
                val_losses.append(metrics['loss'])
                val_maes.append(metrics['mae'])
            
            epoch_metrics.update({
                'val_loss': np.mean(val_losses),
                'val_mae': np.mean(val_maes)
            })
            
            # Update learning rate
            self.scheduler.step(epoch_metrics['val_loss'])
        
        # Save in history
        for key, value in epoch_metrics.items():
            self.history[key].append(value)
        
        return epoch_metrics
    
    def predict(self, data: Union[Data, List[Data]]) -> np.ndarray:
        """Prediction for new data"""
        self.model.eval()
        
        # Preparation data
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        return predictions.cpu().numpy()
    
    def save_model(self, filepath: str) -> None:
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"Model saved in {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        logger.info(f"Model loaded from {filepath}")

def create_crypto_gcn_model(
    input_dim: int,
    output_dim: int = 1,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> Tuple[GraphConvolutionalNetwork, CryptoGCNTrainer]:
    """
    Factory function for creation GCN model for crypto trading
    
    Args:
        input_dim: Dimension input features
        output_dim: Dimension output
        hidden_dims: Dimensions hidden layers
        **kwargs: Additional parameters configuration
        
    Returns:
        Tuple[GraphConvolutionalNetwork, CryptoGCNTrainer]: Model and trainer
    """
    config = GCNConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    model = GraphConvolutionalNetwork(config)
    trainer = CryptoGCNTrainer(model, config)
    
    return model, trainer

# Export main classes
__all__ = [
    'GraphConvolutionalNetwork',
    'GCNConfig',
    'CryptoGCNTrainer',
    'create_crypto_gcn_model'
]