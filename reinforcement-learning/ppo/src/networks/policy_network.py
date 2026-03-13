"""
Policy Network Implementation for PPO
for action distribution modeling

Supports:
- Discrete action spaces (Categorical distribution)
- Continuous action spaces (Normal distribution)
- Mixed action spaces
- Multi-modal distributions
- Constrained action spaces for trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MixtureSameFamily, Independent
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import numpy as np
import math

from ..utils.normalization import RunningMeanStd


class BasePolicyNetwork(nn.Module, ABC):
 """Base class for policy networks"""
 
 def __init__(self, input_dim: int, action_dim: int):
 super().__init__()
 self.input_dim = input_dim
 self.action_dim = action_dim
 
 @abstractmethod
 def forward(self, x: torch.Tensor) -> Union[Categorical, Normal]:
 """Forward pass returning action distribution"""
 pass
 
 @abstractmethod
 def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
 """Sample action from policy"""
 pass


class DiscretePolicyNetwork(BasePolicyNetwork):
 """
 Discrete action space policy network
 
 Uses Categorical distribution for discrete actions:
 - Buy/Sell/Hold decisions
 - Asset selection
 - Order types
 """
 
 def __init__(
 self,
 input_dim: int,
 action_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 temperature: float = 1.0
 ):
 super().__init__(input_dim, action_dim)
 
 if hidden_dims is None:
 hidden_dims = [64, 64]
 
 self.temperature = temperature
 
 # Build network layers
 layers = []
 prev_dim = input_dim
 
 for hidden_dim in hidden_dims:
 layers.extend([
 nn.Linear(prev_dim, hidden_dim),
 self._get_activation(activation),
 nn.LayerNorm(hidden_dim)
 ])
 prev_dim = hidden_dim
 
 # Output layer for action logits
 layers.append(nn.Linear(prev_dim, action_dim))
 
 self.network = nn.Sequential(*layers)
 
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
 nn.init.orthogonal_(module.weight, gain=0.01)
 nn.init.constant_(module.bias, 0)
 
 def forward(self, x: torch.Tensor) -> Categorical:
 """Forward pass returning Categorical distribution"""
 logits = self.network(x)
 
 # Apply temperature scaling
 if self.temperature != 1.0:
 logits = logits / self.temperature
 
 return Categorical(logits=logits)
 
 def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
 """Sample action from policy"""
 action_dist = self.forward(x)
 
 if deterministic:
 action = action_dist.logits.argmax(dim=-1)
 else:
 action = action_dist.sample()
 
 return action


class ContinuousPolicyNetwork(BasePolicyNetwork):
 """
 Continuous action space policy network
 
 Uses Normal distribution for continuous actions:
 - Position sizing
 - Price targets
 - Stop-loss levels
 """
 
 def __init__(
 self,
 input_dim: int,
 action_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 log_std_init: float = 0.0,
 std_bound: Tuple[float, float] = (1e-3, 1.0),
 action_bound: Optional[Tuple[float, float]] = None
 ):
 super().__init__(input_dim, action_dim)
 
 if hidden_dims is None:
 hidden_dims = [64, 64]
 
 self.std_bound = std_bound
 self.action_bound = action_bound
 
 # Build shared network
 layers = []
 prev_dim = input_dim
 
 for hidden_dim in hidden_dims:
 layers.extend([
 nn.Linear(prev_dim, hidden_dim),
 self._get_activation(activation),
 nn.LayerNorm(hidden_dim)
 ])
 prev_dim = hidden_dim
 
 self.shared_network = nn.Sequential(*layers)
 
 # Mean head
 self.mean_head = nn.Linear(prev_dim, action_dim)
 
 # Log std head (learned per action dimension)
 self.log_std_head = nn.Linear(prev_dim, action_dim)
 
 # Initialize log std
 nn.init.constant_(self.log_std_head.weight, 0)
 nn.init.constant_(self.log_std_head.bias, log_std_init)
 
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
 """Initialize weights"""
 for module in self.modules():
 if isinstance(module, nn.Linear) and module != self.log_std_head:
 nn.init.orthogonal_(module.weight, gain=0.01)
 nn.init.constant_(module.bias, 0)
 
 def forward(self, x: torch.Tensor) -> Normal:
 """Forward pass returning Normal distribution"""
 shared_features = self.shared_network(x)
 
 # Get mean and log std
 mean = self.mean_head(shared_features)
 log_std = self.log_std_head(shared_features)
 
 # Clamp log std for numerical stability
 log_std = torch.clamp(
 log_std,
 math.log(self.std_bound[0]),
 math.log(self.std_bound[1])
 )
 
 std = torch.exp(log_std)
 
 # Apply action bounds if specified
 if self.action_bound is not None:
 mean = torch.tanh(mean)
 mean = (mean + 1) / 2 # [0, 1]
 mean = mean * (self.action_bound[1] - self.action_bound[0]) + self.action_bound[0]
 
 return Normal(mean, std)
 
 def get_action(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
 """Sample action from policy"""
 action_dist = self.forward(x)
 
 if deterministic:
 action = action_dist.mean
 else:
 action = action_dist.sample()
 
 return action


class MixedPolicyNetwork(BasePolicyNetwork):
 """
 Mixed action space policy network
 
 Combines discrete and continuous actions:
 - Discrete: Action type (buy/sell/hold)
 - Continuous: Position size, price levels
 """
 
 def __init__(
 self,
 input_dim: int,
 discrete_action_dim: int,
 continuous_action_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh"
 ):
 super().__init__(input_dim, discrete_action_dim + continuous_action_dim)
 
 self.discrete_action_dim = discrete_action_dim
 self.continuous_action_dim = continuous_action_dim
 
 # Discrete policy
 self.discrete_policy = DiscretePolicyNetwork(
 input_dim=input_dim,
 action_dim=discrete_action_dim,
 hidden_dims=hidden_dims,
 activation=activation
 )
 
 # Continuous policy
 self.continuous_policy = ContinuousPolicyNetwork(
 input_dim=input_dim,
 action_dim=continuous_action_dim,
 hidden_dims=hidden_dims,
 activation=activation
 )
 
 def forward(self, x: torch.Tensor) -> Tuple[Categorical, Normal]:
 """Forward pass returning both distributions"""
 discrete_dist = self.discrete_policy(x)
 continuous_dist = self.continuous_policy(x)
 
 return discrete_dist, continuous_dist
 
 def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
 """Sample mixed actions"""
 discrete_dist, continuous_dist = self.forward(x)
 
 if deterministic:
 discrete_action = discrete_dist.logits.argmax(dim=-1)
 continuous_action = continuous_dist.mean
 else:
 discrete_action = discrete_dist.sample()
 continuous_action = continuous_dist.sample()
 
 return {
 "discrete": discrete_action,
 "continuous": continuous_action
 }


class CryptoTradingPolicy(ContinuousPolicyNetwork):
 """
 Specialized policy network for crypto trading
 
 Features:
 - Position sizing in [-1, 1] range
 - Risk-aware action scaling
 - Market regime conditioning
 """
 
 def __init__(
 self,
 input_dim: int,
 hidden_dims: List[int] = None,
 activation: str = "tanh",
 max_position: float = 1.0,
 risk_scaling: bool = True
 ):
 # 3 actions: position_size, entry_price_offset, stop_loss_ratio
 action_dim = 3
 
 super().__init__(
 input_dim=input_dim,
 action_dim=action_dim,
 hidden_dims=hidden_dims or [128, 128],
 activation=activation,
 action_bound=(-max_position, max_position)
 )
 
 self.max_position = max_position
 self.risk_scaling = risk_scaling
 
 # Risk conditioning layer
 if risk_scaling:
 self.risk_head = nn.Sequential(
 nn.Linear(hidden_dims[-1] if hidden_dims else 64, 32),
 nn.ReLU(),
 nn.Linear(32, 1),
 nn.Sigmoid() # Risk factor [0, 1]
 )
 
 def forward(self, x: torch.Tensor) -> Normal:
 """Forward pass with risk conditioning"""
 shared_features = self.shared_network(x)
 
 # Get base mean and std
 mean = self.mean_head(shared_features)
 log_std = self.log_std_head(shared_features)
 
 # Risk conditioning
 if self.risk_scaling:
 risk_factor = self.risk_head(shared_features)
 # Scale position size by risk
 mean = mean * risk_factor.repeat(1, self.action_dim)
 
 # Clamp log std
 log_std = torch.clamp(
 log_std,
 math.log(self.std_bound[0]),
 math.log(self.std_bound[1])
 )
 
 std = torch.exp(log_std)
 
 # Position bounds
 mean = torch.tanh(mean) * self.max_position
 
 return Normal(mean, std)
 
 def get_trading_action(
 self, 
 x: torch.Tensor, 
 current_price: float,
 deterministic: bool = False
 ) -> Dict[str, float]:
 """Get structured trading action"""
 action_dist = self.forward(x)
 
 if deterministic:
 raw_action = action_dist.mean.squeeze().cpu().numpy()
 else:
 raw_action = action_dist.sample().squeeze().cpu().numpy()
 
 return {
 "position_size": float(raw_action[0]), # [-max_position, max_position]
 "entry_price": current_price * (1 + raw_action[1] * 0.01), # ±1% price offset
 "stop_loss": current_price * (1 - abs(raw_action[2]) * 0.05) # Up to 5% stop loss
 }


class PolicyNetwork:
 """
 Factory class for creating policy networks
 
 Automatically selects appropriate network type based on action space
 """
 
 def __new__(
 cls,
 input_dim: int,
 action_dim: int,
 action_type: str = "continuous",
 **kwargs
 ) -> BasePolicyNetwork:
 """Create appropriate policy network"""
 
 if action_type == "discrete":
 return DiscretePolicyNetwork(
 input_dim=input_dim,
 action_dim=action_dim,
 **kwargs
 )
 
 elif action_type == "continuous":
 return ContinuousPolicyNetwork(
 input_dim=input_dim,
 action_dim=action_dim,
 **kwargs
 )
 
 elif action_type == "mixed":
 # Assume equal split for mixed actions
 discrete_dim = action_dim // 2
 continuous_dim = action_dim - discrete_dim
 
 return MixedPolicyNetwork(
 input_dim=input_dim,
 discrete_action_dim=discrete_dim,
 continuous_action_dim=continuous_dim,
 **kwargs
 )
 
 elif action_type == "crypto_trading":
 return CryptoTradingPolicy(
 input_dim=input_dim,
 **kwargs
 )
 
 else:
 raise ValueError(f"Unknown action type: {action_type}")


# Export classes
__all__ = [
 "BasePolicyNetwork",
 "DiscretePolicyNetwork",
 "ContinuousPolicyNetwork", 
 "MixedPolicyNetwork",
 "CryptoTradingPolicy",
 "PolicyNetwork"
]