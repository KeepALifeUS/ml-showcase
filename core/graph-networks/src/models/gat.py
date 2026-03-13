"""
Graph Attention Network (GAT) Implementation for Crypto Trading
================================================================

Enterprise-grade GAT implementation with multi-head attention mechanism
optimized for crypto market relationship modeling with enterprise patterns.

Features:
- Multi-head attention mechanisms
- Learnable attention weights
- Hierarchical attention pooling
- Crypto-specific attention patterns
- Production-ready scalability
- Real-time inference optimization

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, BatchNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GATConfig:
    """
    Configuration for Graph Attention Network
    
    Declarative Configuration
    """
    input_dim: int = 64
    hidden_dims: List[int] = None
    output_dim: int = 1
    num_layers: int = 3
    num_heads: List[int] = None # Number attention heads on each
    dropout_rate: float = 0.2
    attention_dropout: float = 0.1  # Dropout for attention weights
    activation: str = 'elu' # ELU better works with GAT
    use_batch_norm: bool = True
    use_residual: bool = True
    use_edge_weights: bool = True
    concat_heads: bool = True  # Concatenate or average attention heads
    use_gatv2: bool = False # Use GAT
    negative_slope: float = 0.2  # For LeakyReLU in attention
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    attention_regularization: float = 1e-4  # L2 regularization attention
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [
                max(32, self.input_dim // (2**i)) 
                for i in range(self.num_layers - 1)
            ]
        
        if self.num_heads is None:
            # More heads in , less in final
            self.num_heads = [8, 4, 2][:self.num_layers - 1] + [1]
            
        # Check
        if len(self.num_heads) != self.num_layers:
            raise ValueError("Number heads should layers")

class MultiHeadAttention(nn.Module):
    """
    Custom multi-head attention for crypto data
    
    Specifically optimized for time series and inter-asset correlations
    """
    
    def __init__(self, input_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        if input_dim % num_heads != 0:
            raise ValueError("input_dim should be divisible by num_heads")
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(input_dim, input_dim)
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        
        # Output projection
        self.output_proj = nn.Linear(input_dim, input_dim)
        
        # Dropout and normalization
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for attention weights"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with return attention weights for interpretation
        """
        batch_size, num_nodes, input_dim = x.size()
        
        # Linear projections
        Q = self.query_proj(x)  # [batch, nodes, dim]
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        
        # Mask for accounting only related nodes (according to edge_index)
        if edge_index is not None:
            mask = self._create_attention_mask(edge_index, num_nodes, batch_size)
            attention_scores = attention_scores.masked_fill(mask == 0, -float('inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, num_nodes, input_dim
        )
        
        # Final linear projection
        output = self.output_proj(attended_values)
        output = self.output_dropout(output)
        
        return output, attention_weights.mean(dim=1) # Averaging by heads for visualization
    
    def _create_attention_mask(self, edge_index: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
        """Create masks for attention on basis edge_index"""
        mask = torch.zeros(batch_size, num_nodes, num_nodes, device=edge_index.device)
        
        # Fill mask on basis edges graph
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            mask[:, src, dst] = 1
            mask[:, dst, src] = 1 # Symmetric mask for graphs
        
        # Self-attention always
        mask.fill_diagonal_(1)
        
        return mask.unsqueeze(1)  # [batch, 1, nodes, nodes] for broadcasting

class GraphAttentionNetwork(nn.Module):
    """
    Production-Ready Graph Attention Network for crypto trading
    
    Scalable Deep Learning Architecture
    """
    
    def __init__(self, config: GATConfig):
        super().__init__()
        self.config = config
        
        self._validate_config()
        self._build_layers()
        self._initialize_weights()
        
        logger.info(f"Initialized GAT with {config.num_layers} layers and {config.num_heads} ")
    
    def _validate_config(self) -> None:
        """Validation configuration"""
        if self.config.input_dim <= 0:
            raise ValueError("input_dim should be positive")
        if self.config.output_dim <= 0:
            raise ValueError("output_dim should be positive")
        if not 0.0 <= self.config.dropout_rate <= 1.0:
            raise ValueError("dropout_rate should be in [0, 1]")
        if not 0.0 <= self.config.attention_dropout <= 1.0:
            raise ValueError("attention_dropout should be in [0, 1]")
    
    def _build_layers(self) -> None:
        """Build GAT architecture"""
        self.attention_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Dimensions with taking into account concatenation heads
        all_dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.output_dim]
        
        for i in range(len(all_dims) - 1):
            num_heads = self.config.num_heads[i]
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            
            # Select between GAT and GATv2
            if self.config.use_gatv2:
                gat_layer = GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim // num_heads if self.config.concat_heads else out_dim,
                    heads=num_heads,
                    concat=self.config.concat_heads and i < len(all_dims) - 2,  # Last layer not concat
                    dropout=self.config.attention_dropout,
                    add_self_loops=True,
                    edge_dim=1 if self.config.use_edge_weights else None
                )
            else:
                gat_layer = GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim // num_heads if self.config.concat_heads else out_dim,
                    heads=num_heads,
                    concat=self.config.concat_heads and i < len(all_dims) - 2,
                    dropout=self.config.attention_dropout,
                    add_self_loops=True,
                    edge_dim=1 if self.config.use_edge_weights else None,
                    negative_slope=self.config.negative_slope
                )
            
            self.attention_layers.append(gat_layer)
            
            # Batch normalization
            if self.config.use_batch_norm and i < len(all_dims) - 2:
                effective_out_dim = out_dim if not self.config.concat_heads or i == len(all_dims) - 2 else out_dim * num_heads // num_heads
                self.batch_norms.append(BatchNorm(out_dim))
        
        # Regularization and activation
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.activation = self._get_activation()
        
        # Hierarchical attention pooling
        self.hierarchical_attention = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.output_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output layer with crypto-specific features
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim),
            nn.BatchNorm1d(self.config.output_dim),
            nn.ELU(),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(self.config.output_dim, 1),
            nn.Tanh() # Normalization output for stability in crypto trading
        )
        
        # Attention regularization
        self.attention_reg_loss = 0.0
    
    def _get_activation(self) -> nn.Module:
        """Get functions activation"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(self.config.negative_slope),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU()
        }
        return activations.get(self.config.activation, nn.ELU())
    
    def _initialize_weights(self) -> None:
        """Specialized initialization for GAT"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight'):
                    # Xavier initialization for layers
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (GATConv, GATv2Conv)):
                # initialization for attention layers
                if hasattr(module, 'lin_l') and hasattr(module.lin_l, 'weight'):
                    nn.init.xavier_uniform_(module.lin_l.weight)
                if hasattr(module, 'lin_r') and hasattr(module.lin_r, 'weight'):
                    nn.init.xavier_uniform_(module.lin_r.weight)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with return attention weights
        
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: (predictions, attention_weights)
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None) if self.config.use_edge_weights else None
        batch = getattr(data, 'batch', None)
        
        attention_weights = []
        residual_x = x if self.config.use_residual else None
        
        # Pass through attention layers
        for i, gat_layer in enumerate(self.attention_layers[:-1]):
            # GAT forward pass with attention weights
            if isinstance(gat_layer, (GATConv, GATv2Conv)):
                x_new, (edge_index_att, att_weights) = gat_layer(
                    x, edge_index, edge_attr=edge_weight, return_attention_weights=True
                )
                attention_weights.append(att_weights)
                x = x_new
            else:
                x = gat_layer(x, edge_index, edge_weight=edge_weight)
            
            # Batch normalization
            if self.config.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Residual connection
            if (self.config.use_residual and residual_x is not None 
                and x.shape[-1] == residual_x.shape[-1]):
                x = x + residual_x
                residual_x = x
            
            # Dropout
            x = self.dropout(x)
        
        # Last attention layer
        final_layer = self.attention_layers[-1]
        if isinstance(final_layer, (GATConv, GATv2Conv)):
            x, (_, final_att_weights) = final_layer(
                x, edge_index, edge_attr=edge_weight, return_attention_weights=True
            )
            attention_weights.append(final_att_weights)
        else:
            x = final_layer(x, edge_index, edge_weight=edge_weight)
        
        # Hierarchical attention pooling
        node_attention = self.hierarchical_attention(x)
        x_weighted = x * node_attention
        
        # Graph-level aggregation
        if batch is not None:
            # Weighted pooling for each graph in batch
            graph_features = []
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                node_feats = x_weighted[mask]
                weights = node_attention[mask]
                
                # Weighted mean pooling
                weighted_feat = torch.sum(node_feats * weights, dim=0) / torch.sum(weights)
                graph_features.append(weighted_feat)
            
            x = torch.stack(graph_features)
        else:
            # Single graph - weighted mean
            x = torch.sum(x_weighted * node_attention, dim=0) / torch.sum(node_attention)
            x = x.unsqueeze(0)
        
        # Final prediction
        output = self.output_layer(x)
        
        # Attention regularization loss
        self._compute_attention_regularization(attention_weights)
        
        return output, attention_weights
    
    def _compute_attention_regularization(self, attention_weights: List[torch.Tensor]) -> None:
        """Computation regularization loss for attention weights"""
        if not attention_weights:
            self.attention_reg_loss = 0.0
            return
        
        reg_loss = 0.0
        for att_weights in attention_weights:
            # L2 regularization for attention weights
            reg_loss += torch.sum(att_weights ** 2)
        
        self.attention_reg_loss = self.config.attention_regularization * reg_loss
    
    def get_attention_weights(self, data: Data) -> List[torch.Tensor]:
        """Get attention weights for interpretation"""
        _, attention_weights = self.forward(data)
        return attention_weights
    
    def get_node_embeddings(self, data: Data, layer_idx: int = -2) -> torch.Tensor:
        """Get node embeddings from specified layers"""
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_weight', None) if self.config.use_edge_weights else None
        
        # Pass up to specified layers
        for i in range(min(len(self.attention_layers), layer_idx + 1)):
            layer = self.attention_layers[i]
            if isinstance(layer, (GATConv, GATv2Conv)):
                x, _ = layer(x, edge_index, edge_attr=edge_weight, return_attention_weights=True)
            else:
                x = layer(x, edge_index, edge_weight=edge_weight)
            
            if i < len(self.attention_layers) - 1:
                if self.config.use_batch_norm and i < len(self.batch_norms):
                    x = self.batch_norms[i](x)
                x = self.activation(x)
                x = self.dropout(x)
        
        return x

class CryptoGATTrainer:
    """
    Specialized trainer for GAT model in crypto trading
    
    Production Training Pipeline
    """
    
    def __init__(self, model: GraphAttentionNetwork, config: GATConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer with attention-specific settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999), # for attention models
            eps=1e-8
        )
        
        # Scheduler with warm-up for attention models
        self.scheduler = self._create_scheduler()
        
        self.model.to(self.device)
        
        # Metrics and
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [],
            'attention_reg_loss': [], 'learning_rates': []
        }
        
        logger.info(f"GAT trainer ready on device: {self.device}")
    
    def _create_scheduler(self):
        """Create scheduler with warm-up"""
        def lr_lambda(current_step):
            warmup_steps = 100
            if current_step < warmup_steps:
                return current_step / warmup_steps
            return 0.95 ** (current_step - warmup_steps)
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Data) -> Dict[str, float]:
        """Step training with attention regularization"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass
        predictions, attention_weights = self.model(batch)
        targets = batch.y.view(-1, 1).float()
        
        # Main loss
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        
        # Total loss with attention regularization
        total_loss = mse_loss + self.model.attention_reg_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for attention models
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        
        self.optimizer.step()
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item(),
            'attention_reg': self.model.attention_reg_loss.item() if hasattr(self.model.attention_reg_loss, 'item') else 0.0,
            'total_loss': total_loss.item()
        }
    
    def validate_step(self, batch: Data) -> Dict[str, float]:
        """Validation with analysis attention patterns"""
        self.model.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions, attention_weights = self.model(batch)
            targets = batch.y.view(-1, 1).float()
            
            mse_loss = F.mse_loss(predictions, targets)
            mae_loss = F.l1_loss(predictions, targets)
            
            # Analysis attention entropy (diversity attention)
            attention_entropy = 0.0
            if attention_weights:
                for att_weights in attention_weights:
                    # Computing attention distribution
                    att_probs = F.softmax(att_weights, dim=-1)
                    entropy = -torch.sum(att_probs * torch.log(att_probs + 1e-8))
                    attention_entropy += entropy.item()
                attention_entropy /= len(attention_weights)
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item(),
            'attention_entropy': attention_entropy
        }
    
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Training epochs with attention analysis"""
        train_metrics = {'loss': [], 'mae': [], 'attention_reg': []}
        
        for batch in train_loader:
            metrics = self.train_step(batch)
            for key in train_metrics:
                if key in metrics:
                    train_metrics[key].append(metrics[key])
        
        epoch_metrics = {f'train_{k}': np.mean(v) for k, v in train_metrics.items()}
        
        # Validation
        if val_loader:
            val_metrics = {'loss': [], 'mae': [], 'attention_entropy': []}
            
            for batch in val_loader:
                metrics = self.validate_step(batch)
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
            
            epoch_metrics.update({f'val_{k}': np.mean(v) for k, v in val_metrics.items()})
        
        # Update learning rate
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        epoch_metrics['learning_rate'] = current_lr
        
        # Save in history
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        return epoch_metrics
    
    def predict_with_attention(self, data: Union[Data, List[Data]]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Prediction with attention weights"""
        self.model.eval()
        
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions, attention_weights = self.model(batch)
        
        # Convert attention weights in numpy
        attention_np = []
        for att_weights in attention_weights:
            attention_np.append(att_weights.cpu().numpy())
        
        return predictions.cpu().numpy(), attention_np
    
    def save_model(self, filepath: str) -> None:
        """Save GAT model with attention weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"GAT model saved in {filepath}")

def create_crypto_gat_model(
    input_dim: int,
    output_dim: int = 1,
    num_heads: Optional[List[int]] = None,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> Tuple[GraphAttentionNetwork, CryptoGATTrainer]:
    """
    Factory function for creation GAT model
    
    Factory Pattern for ML Models
    """
    config = GATConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    model = GraphAttentionNetwork(config)
    trainer = CryptoGATTrainer(model, config)
    
    return model, trainer

# Export for use
__all__ = [
    'GraphAttentionNetwork',
    'GATConfig', 
    'CryptoGATTrainer',
    'MultiHeadAttention',
    'create_crypto_gat_model'
]