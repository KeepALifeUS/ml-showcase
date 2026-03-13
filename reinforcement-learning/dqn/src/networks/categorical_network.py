"""
Categorical Network for Distributional DQN with .

Categorical DQN represents Q-values as full probability distributions
instead of scalar values:
- Value distributions over discrete support
- Categorical projection for Bellman updates
- Better uncertainty modeling
- Improved stability and sample efficiency
- Configurable number of atoms
"""

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel, Field, validator
import structlog

from .q_network import QNetworkConfig

logger = structlog.get_logger(__name__)


class CategoricalNetworkConfig(QNetworkConfig):
 """Configuration Categorical Network."""

 # Distributional parameters
 num_atoms: int = Field(default=51, description="Number of atoms in distribution", gt=1)
 v_min: float = Field(default=-10.0, description="Minimum value")
 v_max: float = Field(default=10.0, description="Maximum value")

 # Distribution specific
 support_type: str = Field(default="linear", description="Type support (linear/log)")

 @validator("v_max")
 def validate_v_max(cls, v, values):
 if "v_min" in values and v <= values["v_min"]:
 raise ValueError("v_max must be more v_min")
 return v

 @validator("num_atoms")
 def validate_num_atoms(cls, v):
 if v % 2 == 0:
 raise ValueError("num_atoms must be odd")
 return v

 @validator("support_type")
 def validate_support_type(cls, v):
 valid_types = ["linear", "log"]
 if v not in valid_types:
 raise ValueError(f"support_type must be one from: {valid_types}")
 return v


class CategoricalNetwork(nn.Module):
 """
 Categorical Network for Distributional DQN.

 Predicts probability distributions over value ranges instead of
 scalar Q-values. Each action has a categorical distribution over
 a discrete support set.

 Features:
 - Configurable number of atoms and value range
 - Linear or logarithmic support spacing
 - Proper categorical projection for Bellman updates
 - Temperature scaling for distribution sharpening
 - Enterprise monitoring and validation
 """

 def __init__(self, config: CategoricalNetworkConfig):
 """
 Initialization Categorical Network.

 Args:
 config: Configuration categorical network
 """
 super.__init__
 self.config = config
 self.state_size = config.state_size
 self.action_size = config.action_size
 self.num_atoms = config.num_atoms

 # Value support
 self.v_min = config.v_min
 self.v_max = config.v_max

 # Create support
 if config.support_type == "linear":
 support = torch.linspace(self.v_min, self.v_max, self.num_atoms)
 else: # log
 # Logarithmic spacing (useful for asymmetric value ranges)
 log_min = np.log(abs(self.v_min) + 1e-8)
 log_max = np.log(abs(self.v_max) + 1e-8)
 log_support = torch.linspace(log_min, log_max, self.num_atoms)
 support = torch.sign(torch.linspace(self.v_min, self.v_max, self.num_atoms)) * torch.exp(log_support)

 self.register_buffer('support', support)
 self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

 # Build network architecture
 self._build_network

 # Initialize weights
 self._initialize_weights

 self.logger = structlog.get_logger(__name__).bind(
 component="CategoricalNetwork",
 num_atoms=config.num_atoms,
 v_range=f"[{config.v_min}, {config.v_max}]"
 )

 self.logger.info("Categorical Network created")

 def _build_network(self) -> None:
 """Building network architecture."""
 layers = []
 layer_sizes = [self.state_size] + self.config.hidden_layers

 # Feature extraction layers
 for i in range(len(layer_sizes) - 1):
 in_size = layer_sizes[i]
 out_size = layer_sizes[i + 1]

 layers.append(nn.Linear(in_size, out_size))

 if self.config.use_batch_norm:
 layers.append(nn.BatchNorm1d(out_size))

 layers.append(self._get_activation)

 if self.config.dropout_rate > 0:
 layers.append(nn.Dropout(self.config.dropout_rate))

 self.feature_layers = nn.Sequential(*layers)

 # Distributional head
 # Output: [batch_size, action_size * num_atoms]
 feature_size = layer_sizes[-1]
 self.distributional_head = nn.Linear(feature_size, self.action_size * self.num_atoms)

 def _get_activation(self) -> nn.Module:
 """Get function activation."""
 activation_map = {
 "relu": nn.ReLU(inplace=True),
 "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
 "elu": nn.ELU(inplace=True),
 "selu": nn.SELU(inplace=True),
 "gelu": nn.GELU,
 "swish": nn.SiLU(inplace=True),
 }

 return activation_map[self.config.activation]

 def _initialize_weights(self) -> None:
 """Specialized weight initialization for categorical networks."""
 def init_layer(m):
 if isinstance(m, nn.Linear):
 if self.config.init_type == "xavier_uniform":
 nn.init.xavier_uniform_(m.weight)
 elif self.config.init_type == "kaiming_uniform":
 nn.init.kaiming_uniform_(m.weight, nonlinearity=self.config.activation)

 if m.bias is not None:
 nn.init.constant_(m.bias, 0.0)

 elif isinstance(m, nn.BatchNorm1d):
 nn.init.constant_(m.weight, 1.0)
 nn.init.constant_(m.bias, 0.0)

 self.apply(init_layer)

 # Special initialization for distributional head
 # Smaller weights for stable probability distributions
 nn.init.xavier_uniform_(self.distributional_head.weight, gain=0.1)
 nn.init.constant_(self.distributional_head.bias, 0.0)

 def forward(self, state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
 """
 Forward pass returning value distributions.

 Args:
 state: State tensor [batch_size, state_size]
 temperature: Temperature for softmax (lower = sharper distributions)

 Returns:
 Value distributions [batch_size, action_size, num_atoms]
 """
 if state.dim == 1:
 state = state.unsqueeze(0)

 batch_size = state.size(0)

 # Feature extraction
 features = self.feature_layers(state)

 # Distributional logits
 dist_logits = self.distributional_head(features)

 # Reshape: [batch_size, action_size, num_atoms]
 dist_logits = dist_logits.view(batch_size, self.action_size, self.num_atoms)

 # Apply temperature scaling
 if temperature != 1.0:
 dist_logits = dist_logits / temperature

 # Softmax for receiving probability distributions
 distributions = F.softmax(dist_logits, dim=2)

 return distributions

 def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
 """
 Get scalar Q-values from distributions.

 Args:
 state: State tensor

 Returns:
 Q-values [batch_size, action_size]
 """
 distributions = self.forward(state)

 # Expected value: Q(s,a) = sum(p_i * z_i)
 q_values = (distributions * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)

 return q_values

 def get_value_distribution(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
 """
 Get distribution for specific action.

 Args:
 state: State tensor [batch_size, state_size]
 action: Action indices [batch_size]

 Returns:
 Value distribution [batch_size, num_atoms]
 """
 distributions = self.forward(state)

 # Select distribution for beyonddata actions
 action_distributions = distributions.gather(
 1, action.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.num_atoms)
 ).squeeze(1)

 return action_distributions

 def project_distribution(self,
 target_support: torch.Tensor,
 target_probs: torch.Tensor) -> torch.Tensor:
 """
 Categorical projection for Bellman update.

 Projects target distribution onto current support.

 Args:
 target_support: Target support values [batch_size, num_atoms]
 target_probs: Target probabilities [batch_size, num_atoms]

 Returns:
 Projected distribution [batch_size, num_atoms]
 """
 batch_size = target_support.size(0)

 # Clamp target support to network support range
 target_support = torch.clamp(target_support, self.v_min, self.v_max)

 # Initialize projected distribution
 projected_dist = torch.zeros(batch_size, self.num_atoms, device=target_support.device)

 # Project each probability mass
 for i in range(self.num_atoms):
 # Find nearest support indices
 target_z = target_support[:, i]
 target_p = target_probs[:, i]

 # Compute projection indices
 b = (target_z - self.v_min) / self.delta_z
 l = torch.floor(b).long
 u = torch.ceil(b).long

 # Handle boundary cases
 l = torch.clamp(l, 0, self.num_atoms - 1)
 u = torch.clamp(u, 0, self.num_atoms - 1)

 # Distribute probability mass
 # Lower bound
 l_prob = target_p * (u.float - b)
 l_valid = (l >= 0) & (l < self.num_atoms)
 projected_dist.scatter_add_(1, l.unsqueeze(1), (l_prob * l_valid.float).unsqueeze(1))

 # Upper bound
 u_prob = target_p * (b - l.float)
 u_valid = (u >= 0) & (u < self.num_atoms) & (u != l)
 projected_dist.scatter_add_(1, u.unsqueeze(1), (u_prob * u_valid.float).unsqueeze(1))

 return projected_dist

 def compute_distributional_loss(self,
 pred_dist: torch.Tensor,
 target_dist: torch.Tensor,
 reduction: str = 'mean') -> torch.Tensor:
 """
 Compute distributional loss (cross-entropy).

 Args:
 pred_dist: Predicted distributions [batch_size, num_atoms]
 target_dist: Target distributions [batch_size, num_atoms]
 reduction: Loss reduction method

 Returns:
 Distributional loss
 """
 # Cross-entropy loss
 log_pred = torch.log(pred_dist + 1e-8) # Numerical stability
 loss = -(target_dist * log_pred).sum(dim=1)

 if reduction == 'mean':
 return loss.mean
 elif reduction == 'sum':
 return loss.sum
 else:
 return loss

 def get_distribution_statistics(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
 """
 Analysis predicted distributions for monitoring.

 Args:
 state: State tensor

 Returns:
 Distribution statistics
 """
 with torch.no_grad:
 distributions = self.forward(state)

 # Entropy
 entropy = -(distributions * torch.log(distributions + 1e-8)).sum(dim=2)

 # Mean and std each distribution
 support_expanded = self.support.unsqueeze(0).unsqueeze(0) # [1, 1, num_atoms]
 mean_values = (distributions * support_expanded).sum(dim=2)

 variance = (distributions * (support_expanded - mean_values.unsqueeze(2)) ** 2).sum(dim=2)
 std_values = torch.sqrt(variance + 1e-8)

 # Confidence (max probability)
 max_probs = distributions.max(dim=2)[0]

 return {
 'entropy': entropy,
 'mean': mean_values,
 'std': std_values,
 'confidence': max_probs,
 'q_values': self.get_q_values(state),
 }

 def visualize_distribution(self, state: torch.Tensor, action: int = 0) -> Dict[str, Any]:
 """
 Visualization data for specifically action distribution.

 Args:
 state: Single state [state_size]
 action: Action index

 Returns:
 Visualization data
 """
 if state.dim == 1:
 state = state.unsqueeze(0)

 with torch.no_grad:
 distributions = self.forward(state)
 action_dist = distributions[0, action, :].cpu.numpy
 support_vals = self.support.cpu.numpy

 return {
 'support': support_vals,
 'probabilities': action_dist,
 'mean': np.sum(action_dist * support_vals),
 'std': np.sqrt(np.sum(action_dist * (support_vals - np.sum(action_dist * support_vals)) ** 2)),
 'action': action,
 }

 def get_network_stats(self) -> Dict[str, Any]:
 """Network statistics with categorical-specific info."""
 total_params = sum(p.numel for p in self.parameters)
 distributional_params = sum(p.numel for p in self.distributional_head.parameters)

 return {
 "total_parameters": total_params,
 "distributional_parameters": distributional_params,
 "feature_parameters": total_params - distributional_params,

 "num_atoms": self.num_atoms,
 "support_range": [self.v_min, self.v_max],
 "delta_z": self.delta_z,
 "support_type": self.config.support_type,

 "memory_mb": total_params * 4 / (1024 * 1024),
 "config": self.config.dict,
 }

 def __repr__(self) -> str:
 """String representation."""
 return (
 f"CategoricalNetwork(state_size={self.state_size}, "
 f"action_size={self.action_size}, "
 f"num_atoms={self.num_atoms}, "
 f"support_range=[{self.v_min}, {self.v_max}])"
 )