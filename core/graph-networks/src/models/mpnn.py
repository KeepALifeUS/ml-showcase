"""
Message Passing Neural Network (MPNN) Implementation for Crypto Trading
========================================================================

Enterprise-grade MPNN with customizable message and update functions
optimized for crypto market dynamics with cloud-native patterns.

Features:
- Custom message and update functions
- Edge feature integration
- Temporal message passing
- Crypto-specific message aggregation
- Multi-step reasoning capabilities
- Production-ready scalability

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MPNNConfig:
    """
    Configuration for Message Passing Neural Network
    
    Comprehensive Configuration Management
    """
    node_input_dim: int = 64
    edge_input_dim: int = 32
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 4 # Number steps message passing
    num_message_passing_steps: int = 3 # T in MPNN
    
    # Message function parameters
    message_function: str = 'neural'  # neural, linear, attention, crypto_correlation
    message_hidden_dims: List[int] = None
    
    # Update function parameters  
    update_function: str = 'gru'  # gru, lstm, neural, residual
    update_hidden_dims: List[int] = None
    
    # Readout function parameters
    readout_function: str = 'set2set'  # mean, max, add, attention, set2set, hierarchical
    readout_hidden_dim: int = 64
    
    # Regularization
    dropout_rate: float = 0.2
    message_dropout: float = 0.1
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    # Edge features
    use_edge_features: bool = True
    edge_hidden_dim: int = 32
    
    # Advanced features
    use_attention: bool = True
    attention_heads: int = 4
    use_residual: bool = True
    use_self_loops: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    def __post_init__(self):
        if self.message_hidden_dims is None:
            self.message_hidden_dims = [self.hidden_dim, self.hidden_dim // 2]
        
        if self.update_hidden_dims is None:
            self.update_hidden_dims = [self.hidden_dim, self.hidden_dim]

class MessageFunction(nn.Module, ABC):
    """
    Abstract base class for message functions
    
    Strategy Pattern for message passing
    """
    
    @abstractmethod
    def forward(
        self, 
        x_i: torch.Tensor,  # Source node features
        x_j: torch.Tensor,  # Target node features  
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computation from node j to i"""
        pass

class NeuralMessageFunction(MessageFunction):
    """Neural network message function"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        
        # Determining dimension
        input_dim = node_dim * 2  # x_i + x_j
        if edge_dim > 0:
            input_dim += edge_dim
        
        # Create multi-layer network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, node_dim))
        
        self.message_net = nn.Sequential(*layers)
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Concatenation features
        message_input = torch.cat([x_i, x_j], dim=-1)
        
        if edge_attr is not None:
            message_input = torch.cat([message_input, edge_attr], dim=-1)
        
        return self.message_net(message_input)

class AttentionMessageFunction(MessageFunction):
    """Attention-based message function"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Attention projections
        self.query_proj = nn.Linear(node_dim, hidden_dim)
        self.key_proj = nn.Linear(node_dim, hidden_dim)
        self.value_proj = nn.Linear(node_dim, hidden_dim)
        
        # Edge projection
        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_proj = None
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, node_dim)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x_i.size(0)
        
        # Attention computation
        Q = self.query_proj(x_i).view(batch_size, self.num_heads, self.head_dim)
        K = self.key_proj(x_j).view(batch_size, self.num_heads, self.head_dim)
        V = self.value_proj(x_j).view(batch_size, self.num_heads, self.head_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.sum(Q * K, dim=-1) / self.scale  # [batch_size, num_heads]
        
        # Edge attention bias
        if edge_attr is not None and self.edge_proj is not None:
            edge_bias = self.edge_proj(edge_attr).view(batch_size, self.num_heads, self.head_dim)
            edge_bias = torch.sum(edge_bias * K, dim=-1) / self.scale
            attention_scores = attention_scores + edge_bias
        
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads]
        
        # Apply attention
        attended_values = attention_weights.unsqueeze(-1) * V  # [batch_size, num_heads, head_dim]
        attended_values = attended_values.view(batch_size, -1)  # [batch_size, hidden_dim]
        
        # Output projection
        message = self.out_proj(attended_values)
        
        return message

class CryptoCorrelationMessageFunction(MessageFunction):
    """Crypto-specific message function based on market correlations"""
    
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        
        # Price correlation network
        self.price_correlation_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Volume correlation network  
        self.volume_correlation_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Message transformation
        self.message_transform = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Computing correlation
        correlation_input = torch.cat([x_i, x_j], dim=-1)
        price_corr = self.price_correlation_net(correlation_input)
        volume_corr = self.volume_correlation_net(correlation_input)
        
        # Weighted message on basis correlations
        correlation_weight = (price_corr + volume_corr) / 2.0
        
        # message
        raw_message = self.message_transform(x_j)
        weighted_message = raw_message * correlation_weight
        
        return weighted_message

class UpdateFunction(nn.Module, ABC):
    """Abstract base class for update functions"""
    
    @abstractmethod
    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Update hidden state node
        
        Args:
            h: state
            m: message
        """
        pass

class GRUUpdateFunction(UpdateFunction):
    """GRU-based update function"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.gru(m, h)

class NeuralUpdateFunction(UpdateFunction):
    """Neural network update function"""
    
    def __init__(self, hidden_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        
        input_dim = hidden_dim * 2  # h + m
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, hidden_dim))
        
        self.update_net = nn.Sequential(*layers)
        
    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        update_input = torch.cat([h, m], dim=-1)
        return self.update_net(update_input)

class ResidualUpdateFunction(UpdateFunction):
    """Residual update function"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return h + self.transform(m)

class MPNNLayer(MessagePassing):
    """
    One layer Message Passing Neural Network
    
    Composable Neural Network Layers
    """
    
    def __init__(self, config: MPNNConfig):
        super().__init__(aggr='add', node_dim=0) # aggr='add' for messages
        
        self.config = config
        
        # Create message function
        self.message_function = self._create_message_function()
        
        # Create update function
        self.update_function = self._create_update_function()
        
        # Edge embedding if edge features
        if config.use_edge_features and config.edge_input_dim > 0:
            self.edge_encoder = nn.Sequential(
                nn.Linear(config.edge_input_dim, config.edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.edge_hidden_dim, config.edge_hidden_dim)
            )
        else:
            self.edge_encoder = None
        
        # Normalization layers
        if config.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(config.hidden_dim)
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_dim)
            
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def _create_message_function(self) -> MessageFunction:
        """Factory for creation message function"""
        if self.config.message_function == 'neural':
            return NeuralMessageFunction(
                node_dim=self.config.hidden_dim,
                edge_dim=self.config.edge_hidden_dim if self.config.use_edge_features else 0,
                hidden_dims=self.config.message_hidden_dims,
                dropout=self.config.message_dropout
            )
        elif self.config.message_function == 'attention':
            return AttentionMessageFunction(
                node_dim=self.config.hidden_dim,
                edge_dim=self.config.edge_hidden_dim if self.config.use_edge_features else 0,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.attention_heads
            )
        elif self.config.message_function == 'crypto_correlation':
            return CryptoCorrelationMessageFunction(
                node_dim=self.config.hidden_dim,
                edge_dim=self.config.edge_hidden_dim if self.config.use_edge_features else 0,
                hidden_dim=self.config.hidden_dim
            )
        else:
            # Linear message function
            input_dim = self.config.hidden_dim * 2
            if self.config.use_edge_features:
                input_dim += self.config.edge_hidden_dim
            return nn.Sequential(
                nn.Linear(input_dim, self.config.hidden_dim),
                nn.ReLU()
            )
    
    def _create_update_function(self) -> UpdateFunction:
        """Factory for creation update function"""
        if self.config.update_function == 'gru':
            return GRUUpdateFunction(self.config.hidden_dim)
        elif self.config.update_function == 'lstm':
            return nn.LSTMCell(self.config.hidden_dim, self.config.hidden_dim)
        elif self.config.update_function == 'neural':
            return NeuralUpdateFunction(
                hidden_dim=self.config.hidden_dim,
                hidden_dims=self.config.update_hidden_dims,
                dropout=self.config.dropout_rate
            )
        elif self.config.update_function == 'residual':
            return ResidualUpdateFunction(self.config.hidden_dim)
        else:
            # Simple linear update
            return nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through MPNN layer"""
        
        # Add self-loops if necessary
        if self.config.use_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index, 
                edge_attr, 
                num_nodes=x.size(0),
                fill_value=1.0 if edge_attr is not None else None
            )
        
        # Encoding edge features
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Message passing
        messages = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
        # Update node representations
        if isinstance(self.update_function, nn.LSTMCell):
            # LSTM requires cell state
            h_new, _ = self.update_function(messages, (x, x))  # Using x as initial cell state
        elif hasattr(self.update_function, 'forward'):
            h_new = self.update_function(x, messages)
        else:
            # Simple linear update
            update_input = torch.cat([x, messages], dim=-1)
            h_new = self.update_function(update_input)
        
        # Residual connection
        if self.config.use_residual and h_new.shape == x.shape:
            h_new = h_new + x
        
        # Normalization
        if self.config.use_batch_norm and hasattr(self, 'batch_norm'):
            h_new = self.batch_norm(h_new)
        if self.config.use_layer_norm and hasattr(self, 'layer_norm'):
            h_new = self.layer_norm(h_new)
        
        # Dropout
        h_new = self.dropout(h_new)
        
        return h_new
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computation messages between """
        return self.message_function(x_i, x_j, edge_attr)

class ReadoutFunction(nn.Module):
    """
    Readout function for obtaining graph-level representations
    
    Flexible Aggregation Strategies
    """
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        self.readout_type = config.readout_function
        
        if self.readout_type == 'attention':
            self.attention_net = nn.Sequential(
                nn.Linear(config.hidden_dim, config.readout_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.readout_hidden_dim, 1),
                nn.Sigmoid()
            )
        elif self.readout_type == 'set2set':
            # Set2Set implementation
            self.set2set_layers = nn.LSTM(
                config.hidden_dim, 
                config.readout_hidden_dim, 
                batch_first=True
            )
            self.set2set_output = nn.Linear(config.readout_hidden_dim * 2, config.hidden_dim)
        elif self.readout_type == 'hierarchical':
            # Hierarchical pooling
            self.hierarchical_pools = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                    nn.ReLU()
                ),
                nn.Sequential(
                    nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
                    nn.ReLU()
                )
            ])
    
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Aggregation node features in graph-level representation
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch indices for separation graphs
            
        Returns:
            torch.Tensor: Graph-level features
        """
        if self.readout_type == 'mean':
            return global_mean_pool(x, batch)
        
        elif self.readout_type == 'max':
            return global_max_pool(x, batch)
        
        elif self.readout_type == 'add':
            return global_add_pool(x, batch)
        
        elif self.readout_type == 'attention':
            # Attention-based pooling
            attention_weights = self.attention_net(x)  # [num_nodes, 1]
            
            if batch is not None:
                # Weighted pooling for each graph in batch
                graph_features = []
                for i in range(batch.max().item() + 1):
                    mask = (batch == i)
                    node_feats = x[mask]
                    weights = attention_weights[mask]
                    
                    # Normalization weights graph
                    weights = F.softmax(weights, dim=0)
                    weighted_feat = torch.sum(node_feats * weights, dim=0)
                    graph_features.append(weighted_feat)
                
                return torch.stack(graph_features)
            else:
                # Single graph
                weights = F.softmax(attention_weights, dim=0)
                return torch.sum(x * weights, dim=0, keepdim=True)
        
        elif self.readout_type == 'set2set':
            # Set2Set pooling ( version)
            if batch is not None:
                graph_features = []
                for i in range(batch.max().item() + 1):
                    mask = (batch == i)
                    node_feats = x[mask].unsqueeze(0)  # [1, num_nodes_in_graph, hidden_dim]
                    
                    # LSTM for Set2Set
                    lstm_out, (h_n, c_n) = self.set2set_layers(node_feats)
                    
                    # Concatenation hidden and cell states
                    set2set_feat = torch.cat([h_n.squeeze(0), c_n.squeeze(0)], dim=-1)
                    graph_feat = self.set2set_output(set2set_feat)
                    graph_features.append(graph_feat.squeeze(0))
                
                return torch.stack(graph_features)
            else:
                # Single graph
                node_feats = x.unsqueeze(0)
                lstm_out, (h_n, c_n) = self.set2set_layers(node_feats)
                set2set_feat = torch.cat([h_n.squeeze(0), c_n.squeeze(0)], dim=-1)
                return self.set2set_output(set2set_feat)
        
        elif self.readout_type == 'hierarchical':
            # Hierarchical pooling
            current_x = x
            
            for pool_layer in self.hierarchical_pools:
                current_x = pool_layer(current_x)
                
                # Pooling on each
                if batch is not None:
                    current_x = global_mean_pool(current_x, batch)
                    # For next level new batch indices
                    batch = torch.arange(current_x.size(0), device=x.device)
                else:
                    current_x = torch.mean(current_x, dim=0, keepdim=True)
            
            return current_x
        
        else:
            # Default to mean pooling
            return global_mean_pool(x, batch)

class MessagePassingNeuralNetwork(nn.Module):
    """
    Complete Message Passing Neural Network for crypto trading
    
    Enterprise Deep Learning Architecture
    """
    
    def __init__(self, config: MPNNConfig):
        super().__init__()
        self.config = config
        
        self._validate_config()
        self._build_network()
        self._initialize_weights()
        
        logger.info(f"Initialized MPNN with {config.num_layers} layers, {config.num_message_passing_steps} steps")
    
    def _validate_config(self) -> None:
        """Validation configuration"""
        if self.config.node_input_dim <= 0:
            raise ValueError("node_input_dim should be positive")
        if self.config.hidden_dim <= 0:
            raise ValueError("hidden_dim should be positive")
        if self.config.num_layers <= 0:
            raise ValueError("num_layers should be positive")
    
    def _build_network(self) -> None:
        """Build MPNN architecture"""
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.node_input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
        # MPNN layers
        self.mpnn_layers = nn.ModuleList([
            MPNNLayer(self.config) 
            for _ in range(self.config.num_layers)
        ])
        
        # Readout function
        self.readout = ReadoutFunction(self.config)
        
        # Output prediction head
        self.output_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.BatchNorm1d(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate * 0.5),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, self.config.output_dim)
        )
    
    def _initialize_weights(self) -> None:
        """Initialize weights network"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through MPNN
        
        Args:
            data: PyG Data object with node features, edge indices and edge features
            
        Returns:
            torch.Tensor: Graph-level predictions
        """
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        # Encode node features
        h = self.node_encoder(x)  # [num_nodes, hidden_dim]
        
        # Message passing steps
        for t in range(self.config.num_message_passing_steps):
            h_prev = h
            
            # Pass through all MPNN layers
            for layer in self.mpnn_layers:
                h = layer(h, edge_index, edge_attr)
            
            # Residual connection between steps message passing
            if self.config.use_residual and h.shape == h_prev.shape:
                h = h + h_prev
        
        # Readout for obtaining graph-level representation
        graph_features = self.readout(h, batch)  # [batch_size, hidden_dim]
        
        # Final prediction
        output = self.output_head(graph_features)  # [batch_size, output_dim]
        
        return output
    
    def get_node_representations(self, data: Data) -> torch.Tensor:
        """Get final node representations"""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        
        # Encode node features  
        h = self.node_encoder(x)
        
        # Message passing
        for t in range(self.config.num_message_passing_steps):
            for layer in self.mpnn_layers:
                h = layer(h, edge_index, edge_attr)
        
        return h
    
    def get_graph_representation(self, data: Data) -> torch.Tensor:
        """Get graph-level representation without final predictions"""
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        # Encode and message passing
        h = self.node_encoder(x)
        for t in range(self.config.num_message_passing_steps):
            for layer in self.mpnn_layers:
                h = layer(h, edge_index, edge_attr)
        
        # Readout only
        return self.readout(h, batch)

class CryptoMPNNTrainer:
    """
    Specialized trainer for MPNN in crypto trading
    
    Production Training Infrastructure
    """
    
    def __init__(self, model: MessagePassingNeuralNetwork, config: MPNNConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer with settings for message passing
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Scheduler with warm-up for complex message passing models
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 3,
            total_steps=1000,  # Will be updated during training
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.model.to(self.device)
        
        # Metrics
        self.history = {
            'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': [],
            'message_passing_efficiency': [], 'convergence_steps': []
        }
        
        logger.info(f"MPNN trainer ready on device: {self.device}")
    
    def train_step(self, batch: Data) -> Dict[str, float]:
        """Step training with analysis message passing"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass
        predictions = self.model(batch)
        targets = batch.y.view(-1, 1).float()
        
        # Loss computation
        mse_loss = F.mse_loss(predictions, targets)
        mae_loss = F.l1_loss(predictions, targets)
        
        # Regularization on weights message functions
        message_reg = 0.0
        for layer in self.model.mpnn_layers:
            if hasattr(layer.message_function, 'parameters'):
                for param in layer.message_function.parameters():
                    message_reg += torch.sum(param ** 2)
        
        total_loss = mse_loss + 1e-6 * message_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping for stability message passing
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        
        self.optimizer.step()
        
        return {
            'loss': mse_loss.item(),
            'mae': mae_loss.item(),
            'message_reg': message_reg.item(),
            'total_loss': total_loss.item()
        }
    
    def validate_step(self, batch: Data) -> Dict[str, float]:
        """Validation with analysis message passing quality"""
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
    
    def predict(self, data: Union[Data, List[Data]]) -> np.ndarray:
        """Prediction with message passing"""
        self.model.eval()
        
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
        
        batch = batch.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        return predictions.cpu().numpy()
    
    def analyze_message_passing(self, data: Data) -> Dict[str, Any]:
        """Analysis quality message passing for interpretation"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            # Getting node representations on each
            x = self.model.node_encoder(data.x)
            
            step_representations = []
            step_representations.append(x.clone())
            
            for t in range(self.config.num_message_passing_steps):
                for layer in self.model.mpnn_layers:
                    x = layer(x, data.edge_index, getattr(data, 'edge_attr', None))
                step_representations.append(x.clone())
            
            # Analysis convergence
            convergence_metrics = []
            for i in range(1, len(step_representations)):
                prev_repr = step_representations[i-1]
                curr_repr = step_representations[i]
                
                # Cosine similarity between steps
                similarity = F.cosine_similarity(prev_repr.flatten(), curr_repr.flatten(), dim=0)
                convergence_metrics.append(similarity.item())
        
        return {
            'step_representations': [repr.cpu().numpy() for repr in step_representations],
            'convergence_metrics': convergence_metrics,
            'final_graph_repr': self.model.get_graph_representation(data).cpu().numpy()
        }
    
    def save_model(self, filepath: str) -> None:
        """Save MPNN model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        logger.info(f"MPNN model saved in {filepath}")

def create_crypto_mpnn_model(
    node_input_dim: int,
    edge_input_dim: int = 0,
    hidden_dim: int = 128,
    output_dim: int = 1,
    **kwargs
) -> Tuple[MessagePassingNeuralNetwork, CryptoMPNNTrainer]:
    """
    Factory function for creation MPNN model
    
    Factory with Dependency Injection
    """
    config = MPNNConfig(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        **kwargs
    )
    
    model = MessagePassingNeuralNetwork(config)
    trainer = CryptoMPNNTrainer(model, config)
    
    return model, trainer

# Export for use
__all__ = [
    'MessagePassingNeuralNetwork',
    'MPNNConfig',
    'CryptoMPNNTrainer',
    'MPNNLayer',
    'MessageFunction',
    'UpdateFunction', 
    'ReadoutFunction',
    'create_crypto_mpnn_model'
]