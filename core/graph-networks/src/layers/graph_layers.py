"""
Custom Graph Neural Network Layers for Crypto Trading
======================================================

Enterprise-grade custom GNN layers optimized for cryptocurrency 
market analysis with  production patterns.

Features:
- Temporal graph convolution layers
- Multi-scale attention mechanisms
- Crypto-specific pooling operations
- Dynamic edge weight learning
- Production-ready implementations

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
import math
from typing import Optional, Tuple, Union

class TemporalGraphConv(MessagePassing):
    """
    Temporal Graph Convolution Layer for temporal series crypto data
    
    Temporal-Aware Graph Processing
    """
    
    def __init__(self, in_channels: int, out_channels: int, temporal_window: int = 3):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_window = temporal_window
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(in_channels, out_channels, temporal_window, padding=1)
        
        # Spatial graph convolution
        self.spatial_conv = nn.Linear(out_channels, out_channels)
        
        # Normalization
        self.batch_norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                temporal_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            temporal_data: Temporal features [num_nodes, temporal_window, in_channels]
        """
        # Temporal processing if available
        if temporal_data is not None:
            # [num_nodes, temporal_window, in_channels] -> [num_nodes, in_channels, temporal_window]
            temporal_data = temporal_data.transpose(1, 2)
            temporal_features = self.temporal_conv(temporal_data)
            # Take the last time step
            x_temporal = temporal_features[:, :, -1]  # [num_nodes, out_channels]
        else:
            x_temporal = x
        
        # Spatial graph convolution
        x_spatial = self.propagate(edge_index, x=x_temporal)
        
        # Normalization and activation
        x_out = self.batch_norm(x_spatial)
        return F.relu(x_out)
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.spatial_conv(x_j)

class CryptoAttentionLayer(nn.Module):
    """
    Specialized attention layer for crypto asset relationships
    
    Domain-Specific Attention Mechanisms
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention projections
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        
        # Crypto-specific attention biases
        self.price_correlation_bias = nn.Parameter(torch.randn(num_heads))
        self.volume_correlation_bias = nn.Parameter(torch.randn(num_heads))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                price_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Crypto-aware attention computation
        """
        batch_size, num_nodes, _ = x.size() if x.dim() == 3 else (1, x.size(0), x.size(1))
        x = x.view(batch_size, num_nodes, -1)
        
        # Multi-head projections
        Q = self.query_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add crypto-specific biases
        if price_features is not None:
            price_bias = torch.matmul(price_features, price_features.transpose(-2, -1))
            attention_scores += price_bias.unsqueeze(1) * self.price_correlation_bias.view(1, -1, 1, 1)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_values = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended_values = attended_values.view(batch_size, num_nodes, -1)
        output = self.output_proj(attended_values)
        
        return output.squeeze(0) if batch_size == 1 else output

class DynamicEdgeWeightLearner(nn.Module):
    """
    Learnable edge weights for adaptive graph relationships
    
    Adaptive Graph Structure Learning
    """
    
    def __init__(self, node_dim: int, edge_dim: int = 32):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # Edge weight prediction network
        self.edge_predictor = nn.Sequential(
            nn.Linear(node_dim * 2, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim // 2),
            nn.ReLU(),
            nn.Linear(edge_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temperature parameter for edge weight sharpening
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Learn dynamic edge weights based on node features
        """
        row, col = edge_index
        
        # Concatenate source and target node features
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        
        # Predict edge weights
        edge_weights = self.edge_predictor(edge_features).squeeze(-1)
        
        # Apply temperature scaling
        edge_weights = torch.sigmoid(edge_weights / self.temperature)
        
        return edge_weights

__all__ = [
    'TemporalGraphConv',
    'CryptoAttentionLayer', 
    'DynamicEdgeWeightLearner'
]