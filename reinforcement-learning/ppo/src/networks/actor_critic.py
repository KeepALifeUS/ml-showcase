"""
Actor-Critic Network Architectures for PPO
for crypto trading

Supports:
- Shared backbone architecture
- Separate networks for actor and critic
- CNN for price charts
- LSTM for temporal sequences 
- Attention mechanisms
- Multi-head outputs for different assets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import math

from .policy_network import PolicyNetwork
from .value_network import ValueNetwork


@dataclass
class ActorCriticConfig:
 """Configuration for actor-critic networks"""
 
 # Network architecture
 shared_backbone: bool = True
 hidden_dims: List[int] = None
 activation: str = "tanh" # relu, gelu, swish
 
 # Input specifications
 obs_dim: int = 64
 action_dim: int = 4
 action_type: str = "continuous" # discrete, continuous, mixed
 
 # CNN configuration (for price charts)
 use_cnn: bool = False
 cnn_channels: List[int] = None
 cnn_kernels: List[int] = None
 cnn_strides: List[int] = None
 
 # LSTM configuration (for sequences)
 use_lstm: bool = False
 lstm_hidden_size: int = 128
 lstm_num_layers: int = 2
 sequence_length: int = 50
 
 # Attention configuration
 use_attention: bool = False
 attention_heads: int = 8
 attention_dim: int = 128
 
 # Multi-asset support
 multi_asset: bool = False
 num_assets: int = 1
 asset_embedding_dim: int = 32
 
 # Regularization
 dropout_rate: float = 0.1
 layer_norm: bool = True
 spectral_norm: bool = False
 
 # Initialization
 orthogonal_init: bool = True
 gain_policy: float = 0.01
 gain_value: float = 1.0
 
 def __post_init__(self):
 if self.hidden_dims is None:
 self.hidden_dims = [256, 256]
 if self.cnn_channels is None:
 self.cnn_channels = [32, 64, 64]
 if self.cnn_kernels is None:
 self.cnn_kernels = [8, 4, 3]
 if self.cnn_strides is None:
 self.cnn_strides = [4, 2, 1]


class SharedBackbone(nn.Module):
 """Shared backbone network for feature extraction"""
 
 def __init__(self, config: ActorCriticConfig):
 super().__init__()
 self.config = config
 
 # Build feature extractor
 self.feature_extractor = self._build_feature_extractor()
 
 # Get output dimension
 with torch.no_grad():
 dummy_input = torch.zeros(1, config.obs_dim)
 dummy_output = self.feature_extractor(dummy_input)
 self.feature_dim = dummy_output.shape[-1]
 
 def _build_feature_extractor(self) -> nn.Module:
 """Build feature extraction layers"""
 layers = []
 
 if self.config.use_cnn:
 # CNN for price charts
 layers.extend(self._build_cnn_layers())
 
 if self.config.use_lstm:
 # LSTM for sequences
 layers.extend(self._build_lstm_layers())
 
 # Fully connected layers
 layers.extend(self._build_fc_layers())
 
 return nn.Sequential(*layers)
 
 def _build_cnn_layers(self) -> List[nn.Module]:
 """Build CNN layers for price chart processing"""
 layers = []
 in_channels = 1 # Assume single channel price data
 
 for out_channels, kernel_size, stride in zip(
 self.config.cnn_channels,
 self.config.cnn_kernels, 
 self.config.cnn_strides
 ):
 layers.extend([
 nn.Conv1d(in_channels, out_channels, kernel_size, stride),
 nn.ReLU(),
 nn.Dropout(self.config.dropout_rate) if self.config.dropout_rate > 0 else nn.Identity()
 ])
 in_channels = out_channels
 
 # Global pooling
 layers.append(nn.AdaptiveAvgPool1d(1))
 layers.append(nn.Flatten())
 
 return layers
 
 def _build_lstm_layers(self) -> List[nn.Module]:
 """Build LSTM layers for sequence processing"""
 return [
 nn.LSTM(
 input_size=self.config.obs_dim,
 hidden_size=self.config.lstm_hidden_size,
 num_layers=self.config.lstm_num_layers,
 batch_first=True,
 dropout=self.config.dropout_rate if self.config.lstm_num_layers > 1 else 0
 )
 ]
 
 def _build_fc_layers(self) -> List[nn.Module]:
 """Build fully connected layers"""
 layers = []
 
 # Determine input dimension
 if self.config.use_cnn:
 input_dim = self.config.cnn_channels[-1]
 elif self.config.use_lstm:
 input_dim = self.config.lstm_hidden_size
 else:
 input_dim = self.config.obs_dim
 
 # Add multi-asset embedding
 if self.config.multi_asset:
 input_dim += self.config.asset_embedding_dim
 
 # Hidden layers
 for hidden_dim in self.config.hidden_dims:
 layers.extend([
 nn.Linear(input_dim, hidden_dim),
 self._get_activation(),
 nn.LayerNorm(hidden_dim) if self.config.layer_norm else nn.Identity(),
 nn.Dropout(self.config.dropout_rate) if self.config.dropout_rate > 0 else nn.Identity()
 ])
 input_dim = hidden_dim
 
 return layers
 
 def _get_activation(self) -> nn.Module:
 """Get activation function"""
 activations = {
 "relu": nn.ReLU(),
 "tanh": nn.Tanh(),
 "gelu": nn.GELU(),
 "swish": nn.SiLU()
 }
 return activations.get(self.config.activation, nn.ReLU())
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """Forward pass through shared backbone"""
 return self.feature_extractor(x)


class MultiHeadAttention(nn.Module):
 """Multi-head attention for sequence modeling"""
 
 def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
 super().__init__()
 self.embed_dim = embed_dim
 self.num_heads = num_heads
 self.head_dim = embed_dim // num_heads
 
 assert self.head_dim * num_heads == embed_dim
 
 self.q_proj = nn.Linear(embed_dim, embed_dim)
 self.k_proj = nn.Linear(embed_dim, embed_dim) 
 self.v_proj = nn.Linear(embed_dim, embed_dim)
 self.out_proj = nn.Linear(embed_dim, embed_dim)
 
 self.dropout = nn.Dropout(dropout)
 self.scale = math.sqrt(self.head_dim)
 
 def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
 """Multi-head attention forward pass"""
 batch_size, seq_len, embed_dim = query.shape
 
 # Project to Q, K, V
 q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
 k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim) 
 v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
 
 # Transpose for attention computation
 q = q.transpose(1, 2) # [batch, heads, seq_len, head_dim]
 k = k.transpose(1, 2)
 v = v.transpose(1, 2)
 
 # Scaled dot-product attention
 scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
 attn_weights = F.softmax(scores, dim=-1)
 attn_weights = self.dropout(attn_weights)
 
 # Apply attention to values
 attn_output = torch.matmul(attn_weights, v)
 
 # Concatenate heads
 attn_output = attn_output.transpose(1, 2).contiguous()
 attn_output = attn_output.view(batch_size, seq_len, embed_dim)
 
 # Final projection
 output = self.out_proj(attn_output)
 
 return output


class ActorCriticNetwork(nn.Module):
 """
 Main Actor-Critic network with Supports:
 - Shared or separate architectures
 - Multiple action spaces (discrete, continuous, mixed)
 - CNN for chart data
 - LSTM for sequences
 - Attention mechanisms
 - Multi-asset trading
 """
 
 def __init__(self, config: ActorCriticConfig):
 super().__init__()
 self.config = config
 
 # Asset embeddings for multi-asset trading
 if config.multi_asset:
 self.asset_embedding = nn.Embedding(
 config.num_assets, 
 config.asset_embedding_dim
 )
 
 # Attention mechanism
 if config.use_attention:
 self.attention = MultiHeadAttention(
 embed_dim=config.attention_dim,
 num_heads=config.attention_heads,
 dropout=config.dropout_rate
 )
 
 if config.shared_backbone:
 # Shared backbone architecture
 self.shared_backbone = SharedBackbone(config)
 feature_dim = self.shared_backbone.feature_dim
 
 # Policy head
 self.policy_head = PolicyNetwork(
 input_dim=feature_dim,
 action_dim=config.action_dim,
 action_type=config.action_type,
 hidden_dims=[feature_dim // 2],
 activation=config.activation
 )
 
 # Value head
 self.value_head = ValueNetwork(
 input_dim=feature_dim,
 hidden_dims=[feature_dim // 2],
 activation=config.activation
 )
 else:
 # Separate networks
 self.actor = PolicyNetwork(
 input_dim=config.obs_dim,
 action_dim=config.action_dim,
 action_type=config.action_type,
 hidden_dims=config.hidden_dims,
 activation=config.activation
 )
 
 self.critic = ValueNetwork(
 input_dim=config.obs_dim,
 hidden_dims=config.hidden_dims,
 activation=config.activation
 )
 
 # Initialize weights
 if config.orthogonal_init:
 self._initialize_weights()
 
 def _initialize_weights(self):
 """Orthogonal initialization for stable training"""
 for module in self.modules():
 if isinstance(module, nn.Linear):
 # Policy layers get small initialization
 if any(name in str(module) for name in ["policy", "actor"]):
 nn.init.orthogonal_(module.weight, gain=self.config.gain_policy)
 else:
 nn.init.orthogonal_(module.weight, gain=self.config.gain_value)
 
 if module.bias is not None:
 nn.init.constant_(module.bias, 0)
 
 elif isinstance(module, nn.Conv1d):
 nn.init.orthogonal_(module.weight)
 if module.bias is not None:
 nn.init.constant_(module.bias, 0)
 
 def forward(
 self, 
 observations: torch.Tensor,
 asset_ids: Optional[torch.Tensor] = None
 ) -> Tuple[Union[Categorical, Normal], torch.Tensor]:
 """
 Forward pass through actor-critic
 
 Args:
 observations: State observations [batch_size, obs_dim]
 asset_ids: Asset IDs for multi-asset trading [batch_size]
 
 Returns:
 action_distribution, values
 """
 
 # Process asset embeddings
 if self.config.multi_asset and asset_ids is not None:
 asset_emb = self.asset_embedding(asset_ids) # [batch_size, embedding_dim]
 observations = torch.cat([observations, asset_emb], dim=-1)
 
 # Apply attention if used
 if self.config.use_attention:
 # Reshape for attention (assuming sequence dimension)
 batch_size = observations.shape[0]
 seq_len = observations.shape[1] if len(observations.shape) > 2 else 1
 
 if len(observations.shape) == 2:
 observations = observations.unsqueeze(1) # Add sequence dimension
 
 observations = self.attention(observations, observations, observations)
 
 if seq_len == 1:
 observations = observations.squeeze(1) # Remove sequence dimension
 
 if self.config.shared_backbone:
 # Shared feature extraction
 features = self.shared_backbone(observations)
 
 # Get policy distribution and value
 action_dist = self.policy_head(features)
 values = self.value_head(features)
 else:
 # Separate networks
 action_dist = self.actor(observations)
 values = self.critic(observations)
 
 return action_dist, values
 
 def get_value(self, observations: torch.Tensor) -> torch.Tensor:
 """Get value estimates for given observations"""
 with torch.no_grad():
 _, values = self.forward(observations)
 return values
 
 def evaluate_actions(
 self,
 observations: torch.Tensor,
 actions: torch.Tensor
 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
 """
 Evaluate actions for training
 
 Returns:
 log_probs, values, entropy
 """
 action_dist, values = self.forward(observations)
 
 # Compute log probabilities
 if isinstance(action_dist, Categorical):
 log_probs = action_dist.log_prob(actions.squeeze())
 else: # Normal distribution
 log_probs = action_dist.log_prob(actions).sum(dim=-1)
 
 # Compute entropy
 entropy = action_dist.entropy()
 if len(entropy.shape) > 1:
 entropy = entropy.sum(dim=-1)
 
 return log_probs, values.squeeze(), entropy


class CryptoActorCritic(ActorCriticNetwork):
 """
 Specialized Actor-Critic for crypto trading
 
 Additional features:
 - Price chart processing
 - Technical indicator integration 
 - Risk-aware action spaces
 - Multi-timeframe support
 """
 
 def __init__(
 self,
 config: ActorCriticConfig,
 price_history_length: int = 100,
 num_technical_indicators: int = 20
 ):
 self.price_history_length = price_history_length
 self.num_technical_indicators = num_technical_indicators
 
 # Adjust observation dimension
 config.obs_dim = price_history_length + num_technical_indicators
 
 super().__init__(config)
 
 # Price chart CNN
 self.price_cnn = nn.Sequential(
 nn.Conv1d(1, 32, kernel_size=8, stride=2),
 nn.ReLU(),
 nn.Conv1d(32, 64, kernel_size=4, stride=2),
 nn.ReLU(),
 nn.AdaptiveAvgPool1d(1),
 nn.Flatten()
 )
 
 # Technical indicators processor
 self.tech_indicators_fc = nn.Sequential(
 nn.Linear(num_technical_indicators, 64),
 nn.ReLU(),
 nn.Linear(64, 32)
 )
 
 # Risk assessment head
 self.risk_head = nn.Sequential(
 nn.Linear(self.shared_backbone.feature_dim, 64),
 nn.ReLU(),
 nn.Linear(64, 1),
 nn.Sigmoid() # Risk level [0, 1]
 )
 
 def forward(
 self,
 price_history: torch.Tensor, # [batch_size, price_history_length]
 tech_indicators: torch.Tensor, # [batch_size, num_tech_indicators]
 asset_ids: Optional[torch.Tensor] = None
 ) -> Tuple[Union[Categorical, Normal], torch.Tensor, torch.Tensor]:
 """
 Forward pass for crypto trading
 
 Returns:
 action_distribution, values, risk_estimates
 """
 
 # Process price history
 price_features = self.price_cnn(price_history.unsqueeze(1))
 
 # Process technical indicators
 tech_features = self.tech_indicators_fc(tech_indicators)
 
 # Combine features
 observations = torch.cat([price_features, tech_features], dim=-1)
 
 # Standard actor-critic forward
 action_dist, values = super().forward(observations, asset_ids)
 
 # Risk assessment
 if hasattr(self, 'shared_backbone'):
 features = self.shared_backbone(observations)
 risk_estimates = self.risk_head(features)
 else:
 # Use critic features for risk assessment
 critic_features = self.critic.get_features(observations)
 risk_estimates = self.risk_head(critic_features)
 
 return action_dist, values, risk_estimates


# Export main classes
__all__ = [
 "ActorCriticConfig",
 "ActorCriticNetwork", 
 "CryptoActorCritic",
 "SharedBackbone",
 "MultiHeadAttention"
]