"""
Value Network Implementation for PPO
for value function estimation

Features:
- State value estimation V(s)
- Multi-head value functions
- Ensemble value networks
- Uncertainty quantification
- Distributional value functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import math

from ..utils.normalization import RunningMeanStd


class BaseValueNetwork(nn.Module, ABC):
 """Base class for value networks"""
 
 def __init__(self, input_dim: int):
 super().__init__()
 self.input_dim = input_dim
 
 @abstractmethod
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """Forward pass returning value estimates"""
 pass
 
 def get_features(self, x: torch.Tensor) -> torch.Tensor:
 """Get intermediate features for external use"""
 return x


class StandardValueNetwork(BaseValueNetwork):
 """
 Standard value network for state value estimation
 
 Estimates V(s) - expected cumulative reward from state s
 """
 
 def __init__(
 self,
 input_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 layer_norm: bool = True,
 dropout_rate: float = 0.0
 ):
 super().__init__(input_dim)
 
 if hidden_dims is None:
 hidden_dims = [64, 64]
 
 # Build network layers
 layers = []
 prev_dim = input_dim
 
 for i, hidden_dim in enumerate(hidden_dims):
 layers.extend([
 nn.Linear(prev_dim, hidden_dim),
 self._get_activation(activation)
 ])
 
 if layer_norm:
 layers.append(nn.LayerNorm(hidden_dim))
 
 if dropout_rate > 0:
 layers.append(nn.Dropout(dropout_rate))
 
 prev_dim = hidden_dim
 
 # Value head - single output
 layers.append(nn.Linear(prev_dim, 1))
 
 self.network = nn.Sequential(*layers)
 
 # Store feature extraction part
 self.feature_extractor = nn.Sequential(*layers[:-1])
 
 # Initialize weights
 self._initialize_weights()
 
 def _get_activation(self, activation: str) -> nn.Module:
 """Get activation function"""
 activations = {
 "relu": nn.ReLU(),
 "tanh": nn.Tanh(), 
 "gelu": nn.GELU(),
 "swish": nn.SiLU()
 }
 return activations.get(activation, nn.Tanh())
 
 def _initialize_weights(self):
 """Initialize weights for stable training"""
 for module in self.modules():
 if isinstance(module, nn.Linear):
 nn.init.orthogonal_(module.weight, gain=1.0)
 nn.init.constant_(module.bias, 0)
 
 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """Forward pass returning value estimates"""
 return self.network(x)
 
 def get_features(self, x: torch.Tensor) -> torch.Tensor:
 """Get features before final value head"""
 return self.feature_extractor(x)


class MultiHeadValueNetwork(BaseValueNetwork):
 """
 Multi-head value network for different value types
 
 Heads:
 - Main value V(s)
 - Risk-adjusted value
 - Short-term value (immediate rewards)
 - Long-term value (discounted future)
 """
 
 def __init__(
 self,
 input_dim: int,
 num_heads: int = 3,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 head_names: Optional[List[str]] = None
 ):
 super().__init__(input_dim)
 
 if hidden_dims is None:
 hidden_dims = [128, 128]
 
 if head_names is None:
 head_names = [f"head_{i}" for i in range(num_heads)]
 
 self.num_heads = num_heads
 self.head_names = head_names
 
 # Shared feature extractor
 layers = []
 prev_dim = input_dim
 
 for hidden_dim in hidden_dims:
 layers.extend([
 nn.Linear(prev_dim, hidden_dim),
 self._get_activation(activation),
 nn.LayerNorm(hidden_dim)
 ])
 prev_dim = hidden_dim
 
 self.shared_features = nn.Sequential(*layers)
 
 # Multiple value heads
 self.value_heads = nn.ModuleDict()
 for head_name in head_names:
 self.value_heads[head_name] = nn.Sequential(
 nn.Linear(prev_dim, 64),
 self._get_activation(activation),
 nn.Linear(64, 1)
 )
 
 self._initialize_weights()
 
 def _get_activation(self, activation: str) -> nn.Module:
 activations = {
 "relu": nn.ReLU(),
 "tanh": nn.Tanh(),
 "gelu": nn.GELU(),
 "swish": nn.SiLU()
 }
 return activations.get(activation, nn.Tanh())
 
 def _initialize_weights(self):
 for module in self.modules():
 if isinstance(module, nn.Linear):
 nn.init.orthogonal_(module.weight, gain=1.0)
 nn.init.constant_(module.bias, 0)
 
 def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
 """Forward pass returning multiple value estimates"""
 features = self.shared_features(x)
 
 values = {}
 for head_name, head in self.value_heads.items():
 values[head_name] = head(features)
 
 return values
 
 def get_main_value(self, x: torch.Tensor) -> torch.Tensor:
 """Get main value estimate (first head)"""
 values = self.forward(x)
 main_head = self.head_names[0]
 return values[main_head]


class EnsembleValueNetwork(BaseValueNetwork):
 """
 Ensemble of value networks for uncertainty quantification
 
 Benefits:
 - Uncertainty estimation
 - Robust value estimates
 - Better generalization
 """
 
 def __init__(
 self,
 input_dim: int,
 ensemble_size: int = 5,
 hidden_dims: List[int] = None,
 activation: str = "tanh"
 ):
 super().__init__(input_dim)
 
 self.ensemble_size = ensemble_size
 
 # Create ensemble of networks
 self.networks = nn.ModuleList()
 for _ in range(ensemble_size):
 network = StandardValueNetwork(
 input_dim=input_dim,
 hidden_dims=hidden_dims,
 activation=activation
 )
 self.networks.append(network)
 
 def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Forward pass returning mean and uncertainty
 
 Returns:
 mean_values: [batch_size, 1]
 std_values: [batch_size, 1] - uncertainty estimate
 """
 # Get predictions from all networks
 predictions = []
 for network in self.networks:
 pred = network(x)
 predictions.append(pred)
 
 # Stack predictions
 predictions = torch.stack(predictions, dim=0) # [ensemble_size, batch_size, 1]
 
 # Compute statistics
 mean_values = predictions.mean(dim=0)
 std_values = predictions.std(dim=0)
 
 return mean_values, std_values
 
 def get_value_with_uncertainty(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
 """Get value estimate with uncertainty metrics"""
 mean_values, std_values = self.forward(x)
 
 return {
 "value": mean_values,
 "uncertainty": std_values,
 "confidence": 1.0 / (1.0 + std_values) # Higher = more confident
 }


class DistributionalValueNetwork(BaseValueNetwork):
 """
 Distributional value network (C51 style)
 
 Models value distribution instead of scalar expectation
 Better for handling reward uncertainty
 """
 
 def __init__(
 self,
 input_dim: int,
 num_atoms: int = 51,
 v_min: float = -10.0,
 v_max: float = 10.0,
 hidden_dims: List[int] = None,
 activation: str = "tanh"
 ):
 super().__init__(input_dim)
 
 self.num_atoms = num_atoms
 self.v_min = v_min
 self.v_max = v_max
 
 # Value support
 self.register_buffer(
 'support',
 torch.linspace(v_min, v_max, num_atoms)
 )
 
 if hidden_dims is None:
 hidden_dims = [128, 128]
 
 # Build network
 layers = []
 prev_dim = input_dim
 
 for hidden_dim in hidden_dims:
 layers.extend([
 nn.Linear(prev_dim, hidden_dim),
 self._get_activation(activation),
 nn.LayerNorm(hidden_dim)
 ])
 prev_dim = hidden_dim
 
 # Output logits for distribution over atoms
 layers.append(nn.Linear(prev_dim, num_atoms))
 
 self.network = nn.Sequential(*layers)
 self._initialize_weights()
 
 def _get_activation(self, activation: str) -> nn.Module:
 activations = {
 "relu": nn.ReLU(),
 "tanh": nn.Tanh(),
 "gelu": nn.GELU(),
 "swish": nn.SiLU()
 }
 return activations.get(activation, nn.Tanh())
 
 def _initialize_weights(self):
 for module in self.modules():
 if isinstance(module, nn.Linear):
 nn.init.orthogonal_(module.weight, gain=1.0)
 nn.init.constant_(module.bias, 0)
 
 def forward(self, x: torch.Tensor) -> Categorical:
 """Forward pass returning distribution over value atoms"""
 logits = self.network(x)
 return Categorical(logits=logits)
 
 def get_expected_value(self, x: torch.Tensor) -> torch.Tensor:
 """Get expected value from distribution"""
 dist = self.forward(x)
 probs = dist.probs # [batch_size, num_atoms]
 
 # Expected value = sum(prob * support_value)
 expected_value = (probs * self.support.unsqueeze(0)).sum(dim=-1, keepdim=True)
 
 return expected_value
 
 def get_value_distribution(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
 """Get full value distribution information"""
 dist = self.forward(x)
 probs = dist.probs
 
 expected_value = (probs * self.support.unsqueeze(0)).sum(dim=-1, keepdim=True)
 
 # Compute variance
 variance = (probs * (self.support.unsqueeze(0) - expected_value) ** 2).sum(dim=-1, keepdim=True)
 std = torch.sqrt(variance + 1e-8)
 
 return {
 "expected_value": expected_value,
 "std": std,
 "probabilities": probs,
 "support": self.support
 }


class CryptoValueNetwork(StandardValueNetwork):
 """
 Specialized value network for crypto trading
 
 Features:
 - Market regime awareness
 - Risk-adjusted valuation
 - Multi-timeframe value estimation
 """
 
 def __init__(
 self,
 input_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 market_regimes: int = 4 # Bull, Bear, Sideways, High Vol
 ):
 if hidden_dims is None:
 hidden_dims = [128, 128]
 
 super().__init__(input_dim, hidden_dims, activation)
 
 self.market_regimes = market_regimes
 
 # Market regime classifier
 self.regime_classifier = nn.Sequential(
 nn.Linear(hidden_dims[-1], 64),
 nn.ReLU(),
 nn.Linear(64, market_regimes),
 nn.Softmax(dim=-1)
 )
 
 # Regime-specific value heads
 self.regime_values = nn.ModuleList([
 nn.Linear(hidden_dims[-1], 1)
 for _ in range(market_regimes)
 ])
 
 # Risk assessment head
 self.risk_head = nn.Sequential(
 nn.Linear(hidden_dims[-1], 32),
 nn.ReLU(),
 nn.Linear(32, 1),
 nn.Sigmoid()
 )
 
 def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
 """Forward pass with regime-aware valuation"""
 features = self.feature_extractor(x)
 
 # Base value estimate
 base_value = self.network[-1](features) # Final linear layer
 
 # Market regime classification
 regime_probs = self.regime_classifier(features)
 
 # Regime-specific values
 regime_values = []
 for regime_head in self.regime_values:
 regime_value = regime_head(features)
 regime_values.append(regime_value)
 
 regime_values = torch.stack(regime_values, dim=-1) # [batch_size, 1, num_regimes]
 
 # Weighted value by regime probabilities
 weighted_value = (regime_values * regime_probs.unsqueeze(1)).sum(dim=-1)
 
 # Risk assessment
 risk_level = self.risk_head(features)
 
 return {
 "base_value": base_value,
 "regime_aware_value": weighted_value,
 "regime_probs": regime_probs,
 "risk_level": risk_level
 }
 
 def get_trading_value(self, x: torch.Tensor) -> torch.Tensor:
 """Get main trading value (regime-aware)"""
 outputs = self.forward(x)
 return outputs["regime_aware_value"]


class ValueNetwork:
 """
 Factory class for creating value networks
 
 Automatically selects appropriate network type
 """
 
 def __new__(
 cls,
 input_dim: int,
 network_type: str = "standard",
 **kwargs
 ) -> BaseValueNetwork:
 """Create appropriate value network"""
 
 if network_type == "standard":
 return StandardValueNetwork(input_dim=input_dim, **kwargs)
 
 elif network_type == "multi_head":
 return MultiHeadValueNetwork(input_dim=input_dim, **kwargs)
 
 elif network_type == "ensemble":
 return EnsembleValueNetwork(input_dim=input_dim, **kwargs)
 
 elif network_type == "distributional":
 return DistributionalValueNetwork(input_dim=input_dim, **kwargs)
 
 elif network_type == "crypto":
 return CryptoValueNetwork(input_dim=input_dim, **kwargs)
 
 else:
 raise ValueError(f"Unknown network type: {network_type}")


# Export classes
__all__ = [
 "BaseValueNetwork",
 "StandardValueNetwork",
 "MultiHeadValueNetwork",
 "EnsembleValueNetwork", 
 "DistributionalValueNetwork",
 "CryptoValueNetwork",
 "ValueNetwork"
]