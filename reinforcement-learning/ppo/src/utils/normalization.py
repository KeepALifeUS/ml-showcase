"""
Normalization Utilities for PPO
for stable training

Provides various normalization techniques:
- Running mean/std normalization
- Advantage normalization
- Observation normalization
- Reward normalization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import pickle
from abc import ABC, abstractmethod


@dataclass
class NormalizationConfig:
 """Configuration for normalization utilities"""
 
 # Running statistics
 epsilon: float = 1e-8 # Numerical stability
 momentum: float = 0.99 # Momentum for running stats
 clip_range: float = 10.0 # Clipping range
 
 # Observation normalization
 normalize_observations: bool = True
 obs_clip_range: float = 10.0
 
 # Reward normalization
 normalize_rewards: bool = False
 reward_clip_range: float = 10.0
 discount_factor: float = 0.99
 
 # Advantage normalization
 normalize_advantages: bool = True
 advantage_clip_range: float = 5.0
 
 # Batch normalization
 use_batch_norm: bool = False
 batch_norm_momentum: float = 0.1


class RunningMeanStd:
 """
 Running mean and standard deviation computation
 
 Efficiently tracks statistics over streaming data
 using Welford's online algorithm
 """
 
 def __init__(
 self,
 shape: Tuple[int, ...] = (),
 epsilon: float = 1e-8,
 momentum: Optional[float] = None
 ):
 self.shape = shape
 self.epsilon = epsilon
 self.momentum = momentum
 
 # Statistics
 self.count = 0
 self.mean = np.zeros(shape, dtype=np.float64)
 self.var = np.ones(shape, dtype=np.float64)
 
 # For momentum-based updates
 if momentum is not None:
 self.running_mean = np.zeros(shape, dtype=np.float64)
 self.running_var = np.ones(shape, dtype=np.float64)
 
 def update(self, x: Union[np.ndarray, torch.Tensor]):
 """Update statistics with new data"""
 
 if isinstance(x, torch.Tensor):
 x = x.detach().cpu().numpy()
 
 x = x.astype(np.float64)
 
 if self.momentum is None:
 # Welford's online algorithm
 self.count += 1
 delta = x - self.mean
 self.mean += delta / self.count
 delta2 = x - self.mean
 self.var += delta * delta2
 else:
 # Momentum-based update
 self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x
 self.running_var = self.momentum * self.running_var + (1 - self.momentum) * (x - self.running_mean) ** 2
 
 self.mean = self.running_mean
 self.var = self.running_var
 self.count += 1
 
 def update_batch(self, x: Union[np.ndarray, torch.Tensor]):
 """Update statistics with batch of data"""
 
 if isinstance(x, torch.Tensor):
 x = x.detach().cpu().numpy()
 
 x = x.astype(np.float64)
 batch_size = x.shape[0]
 
 if self.momentum is None:
 # Batch Welford update
 batch_mean = np.mean(x, axis=0)
 batch_var = np.var(x, axis=0, ddof=0)
 
 old_count = self.count
 self.count += batch_size
 
 # Update mean
 delta_mean = batch_mean - self.mean
 self.mean += delta_mean * batch_size / self.count
 
 # Update variance
 if old_count > 0:
 self.var = (
 old_count * self.var +
 batch_size * batch_var +
 delta_mean ** 2 * old_count * batch_size / self.count
 ) / (self.count - 1) if self.count > 1 else self.var
 else:
 self.var = batch_var
 else:
 # Momentum-based batch update
 batch_mean = np.mean(x, axis=0)
 batch_var = np.var(x, axis=0, ddof=0)
 
 if self.count == 0:
 self.running_mean = batch_mean
 self.running_var = batch_var
 else:
 self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
 self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
 
 self.mean = self.running_mean
 self.var = self.running_var
 self.count += batch_size
 
 def normalize(
 self,
 x: Union[np.ndarray, torch.Tensor],
 clip_range: Optional[float] = None
 ) -> Union[np.ndarray, torch.Tensor]:
 """Normalize data using current statistics"""
 
 was_tensor = isinstance(x, torch.Tensor)
 device = x.device if was_tensor else None
 
 if was_tensor:
 x_np = x.detach().cpu().numpy()
 else:
 x_np = x
 
 # Normalize
 std = np.sqrt(self.var + self.epsilon)
 normalized = (x_np - self.mean) / std
 
 # Clipping
 if clip_range is not None:
 normalized = np.clip(normalized, -clip_range, clip_range)
 
 # Convert back to tensor if needed
 if was_tensor:
 return torch.tensor(normalized, dtype=x.dtype, device=device)
 else:
 return normalized.astype(x.dtype)
 
 def denormalize(
 self,
 x: Union[np.ndarray, torch.Tensor]
 ) -> Union[np.ndarray, torch.Tensor]:
 """Denormalize data back to original scale"""
 
 was_tensor = isinstance(x, torch.Tensor)
 device = x.device if was_tensor else None
 
 if was_tensor:
 x_np = x.detach().cpu().numpy()
 else:
 x_np = x
 
 # Denormalize
 std = np.sqrt(self.var + self.epsilon)
 denormalized = x_np * std + self.mean
 
 # Convert back to tensor if needed
 if was_tensor:
 return torch.tensor(denormalized, dtype=x.dtype, device=device)
 else:
 return denormalized.astype(x.dtype)
 
 @property
 def std(self) -> np.ndarray:
 """Get standard deviation"""
 return np.sqrt(self.var + self.epsilon)
 
 def get_state(self) -> Dict[str, Any]:
 """Get serializable state"""
 return {
 "shape": self.shape,
 "epsilon": self.epsilon,
 "momentum": self.momentum,
 "count": self.count,
 "mean": self.mean.copy(),
 "var": self.var.copy()
 }
 
 def set_state(self, state: Dict[str, Any]):
 """Set state from serializable dict"""
 self.shape = state["shape"]
 self.epsilon = state["epsilon"]
 self.momentum = state["momentum"]
 self.count = state["count"]
 self.mean = state["mean"].copy()
 self.var = state["var"].copy()


class ObservationNormalizer:
 """
 Observation normalization for stable RL training
 
 Maintains running statistics of observations
 and normalizes them online
 """
 
 def __init__(
 self,
 observation_space: Tuple[int, ...],
 config: Optional[NormalizationConfig] = None
 ):
 self.config = config or NormalizationConfig()
 
 self.running_stats = RunningMeanStd(
 shape=observation_space,
 epsilon=self.config.epsilon,
 momentum=self.config.momentum if self.config.momentum < 1.0 else None
 )
 
 self.enabled = self.config.normalize_observations
 
 def __call__(
 self,
 observations: torch.Tensor,
 update_stats: bool = True
 ) -> torch.Tensor:
 """Normalize observations"""
 
 if not self.enabled:
 return observations
 
 # Update statistics
 if update_stats:
 if observations.ndim > len(self.running_stats.shape):
 # Batch of observations
 self.running_stats.update_batch(observations)
 else:
 # Single observation
 self.running_stats.update(observations)
 
 # Normalize
 return self.running_stats.normalize(
 observations,
 clip_range=self.config.obs_clip_range
 )
 
 def reset_stats(self):
 """Reset normalization statistics"""
 self.running_stats = RunningMeanStd(
 shape=self.running_stats.shape,
 epsilon=self.config.epsilon,
 momentum=self.config.momentum if self.config.momentum < 1.0 else None
 )


class RewardNormalizer:
 """
 Reward normalization for stable training
 
 Normalizes rewards using discounted return statistics
 """
 
 def __init__(self, config: Optional[NormalizationConfig] = None):
 self.config = config or NormalizationConfig()
 
 self.running_stats = RunningMeanStd(
 epsilon=self.config.epsilon,
 momentum=self.config.momentum if self.config.momentum < 1.0 else None
 )
 
 self.returns = 0.0
 self.enabled = self.config.normalize_rewards
 self.gamma = self.config.discount_factor
 
 def __call__(
 self,
 reward: float,
 done: bool = False
 ) -> float:
 """Normalize single reward"""
 
 if not self.enabled:
 return reward
 
 # Update discounted return
 self.returns = self.returns * self.gamma + reward
 
 # Update statistics
 self.running_stats.update(np.array([self.returns]))
 
 # Reset return if episode ended
 if done:
 self.returns = 0.0
 
 # Normalize reward
 normalized = self.running_stats.normalize(
 np.array([reward]),
 clip_range=self.config.reward_clip_range
 )[0]
 
 return float(normalized)
 
 def normalize_batch(
 self,
 rewards: torch.Tensor,
 dones: torch.Tensor
 ) -> torch.Tensor:
 """Normalize batch of rewards"""
 
 if not self.enabled:
 return rewards
 
 # Process batch sequentially for proper return tracking
 normalized_rewards = torch.zeros_like(rewards)
 
 for i in range(len(rewards)):
 normalized_rewards[i] = self.__call__(
 rewards[i].item(),
 dones[i].item()
 )
 
 return normalized_rewards


def normalize_advantages(
 advantages: torch.Tensor,
 epsilon: float = 1e-8
) -> torch.Tensor:
 """
 Normalize advantages for stable training
 
 Args:
 advantages: Advantage estimates [batch_size] or [seq_len, batch_size]
 epsilon: Numerical stability constant
 
 Returns:
 Normalized advantages
 """
 
 if advantages.numel() <= 1:
 return advantages
 
 # Compute mean and std
 mean = advantages.mean()
 std = advantages.std()
 
 # Normalize
 normalized = (advantages - mean) / (std + epsilon)
 
 return normalized


def normalize_returns(
 returns: torch.Tensor,
 epsilon: float = 1e-8,
 clip_range: Optional[float] = None
) -> torch.Tensor:
 """
 Normalize returns for value function training
 
 Args:
 returns: Return estimates
 epsilon: Numerical stability
 clip_range: Optional clipping range
 
 Returns:
 Normalized returns
 """
 
 if returns.numel() <= 1:
 return returns
 
 # Normalize
 mean = returns.mean()
 std = returns.std()
 
 normalized = (returns - mean) / (std + epsilon)
 
 # Optional clipping
 if clip_range is not None:
 normalized = torch.clamp(normalized, -clip_range, clip_range)
 
 return normalized


class VectorNormalizer:
 """
 Multi-dimensional vector normalization
 
 Supports different normalization strategies for
 different dimensions of vector inputs
 """
 
 def __init__(
 self,
 shape: Tuple[int, ...],
 normalize_dims: Optional[List[int]] = None,
 strategies: Optional[Dict[int, str]] = None,
 config: Optional[NormalizationConfig] = None
 ):
 self.shape = shape
 self.config = config or NormalizationConfig()
 
 # Determine which dimensions to normalize
 if normalize_dims is None:
 self.normalize_dims = list(range(len(shape)))
 else:
 self.normalize_dims = normalize_dims
 
 # Normalization strategies for each dimension
 if strategies is None:
 self.strategies = {dim: "standard" for dim in self.normalize_dims}
 else:
 self.strategies = strategies
 
 # Initialize normalizers for each dimension
 self.normalizers = {}
 for dim in self.normalize_dims:
 if dim < len(shape):
 dim_shape = (shape[dim],) if len(shape) > 1 else ()
 self.normalizers[dim] = RunningMeanStd(
 shape=dim_shape,
 epsilon=self.config.epsilon,
 momentum=self.config.momentum if self.config.momentum < 1.0 else None
 )
 
 def __call__(
 self,
 x: torch.Tensor,
 update_stats: bool = True
 ) -> torch.Tensor:
 """Normalize vector input"""
 
 normalized = x.clone()
 
 for dim in self.normalize_dims:
 if dim >= x.shape[-1]:
 continue
 
 # Extract dimension data
 if len(x.shape) == 1:
 dim_data = x[dim:dim+1]
 else:
 dim_data = x[..., dim]
 
 # Update statistics
 if update_stats:
 self.normalizers[dim].update_batch(dim_data)
 
 # Normalize based on strategy
 strategy = self.strategies.get(dim, "standard")
 
 if strategy == "standard":
 normalized_dim = self.normalizers[dim].normalize(dim_data)
 elif strategy == "minmax":
 # Min-max normalization
 min_val = self.normalizers[dim].mean - 2 * self.normalizers[dim].std
 max_val = self.normalizers[dim].mean + 2 * self.normalizers[dim].std
 normalized_dim = (dim_data - min_val) / (max_val - min_val + self.config.epsilon)
 normalized_dim = torch.clamp(normalized_dim, 0, 1)
 else:
 normalized_dim = dim_data
 
 # Update normalized tensor
 if len(x.shape) == 1:
 normalized[dim] = normalized_dim
 else:
 normalized[..., dim] = normalized_dim
 
 return normalized


class AdaptiveNormalizer:
 """
 Adaptive normalizer that adjusts normalization parameters
 based on training progress and data characteristics
 """
 
 def __init__(
 self,
 shape: Tuple[int, ...],
 adaptation_rate: float = 0.01,
 min_samples: int = 100,
 config: Optional[NormalizationConfig] = None
 ):
 self.config = config or NormalizationConfig()
 self.adaptation_rate = adaptation_rate
 self.min_samples = min_samples
 
 self.base_normalizer = RunningMeanStd(
 shape=shape,
 epsilon=self.config.epsilon
 )
 
 # Adaptive parameters
 self.adaptive_epsilon = self.config.epsilon
 self.adaptive_clip_range = self.config.clip_range
 
 # Tracking
 self.stability_history = []
 self.variance_history = []
 
 def __call__(
 self,
 x: torch.Tensor,
 update_stats: bool = True
 ) -> torch.Tensor:
 """Adaptive normalization"""
 
 # Update base statistics
 if update_stats:
 self.base_normalizer.update_batch(x)
 
 # Update adaptive parameters
 if self.base_normalizer.count >= self.min_samples:
 self._update_adaptive_parameters()
 
 # Normalize with adaptive parameters
 if isinstance(x, torch.Tensor):
 x_np = x.detach().cpu().numpy()
 else:
 x_np = x
 
 std = np.sqrt(self.base_normalizer.var + self.adaptive_epsilon)
 normalized = (x_np - self.base_normalizer.mean) / std
 
 # Adaptive clipping
 normalized = np.clip(normalized, -self.adaptive_clip_range, self.adaptive_clip_range)
 
 # Convert back
 if isinstance(x, torch.Tensor):
 return torch.tensor(normalized, dtype=x.dtype, device=x.device)
 else:
 return normalized
 
 def _update_adaptive_parameters(self):
 """Update adaptive parameters based on data characteristics"""
 
 # Compute stability metric
 current_var = np.mean(self.base_normalizer.var)
 self.variance_history.append(current_var)
 
 if len(self.variance_history) >= 10:
 # Variance stability
 recent_vars = self.variance_history[-10:]
 stability = 1.0 / (1.0 + np.std(recent_vars))
 self.stability_history.append(stability)
 
 # Adapt epsilon based on stability
 if stability < 0.5: # Unstable
 self.adaptive_epsilon = min(
 self.config.epsilon * 10,
 self.adaptive_epsilon * (1 + self.adaptation_rate)
 )
 else: # Stable
 self.adaptive_epsilon = max(
 self.config.epsilon,
 self.adaptive_epsilon * (1 - self.adaptation_rate)
 )
 
 # Adapt clipping based on variance
 if current_var > 1.0: # High variance
 self.adaptive_clip_range = min(
 self.config.clip_range * 2,
 self.adaptive_clip_range * (1 + self.adaptation_rate)
 )
 else: # Low variance
 self.adaptive_clip_range = max(
 self.config.clip_range * 0.5,
 self.adaptive_clip_range * (1 - self.adaptation_rate)
 )
 
 # Trim history
 if len(self.variance_history) > 100:
 self.variance_history = self.variance_history[-100:]
 if len(self.stability_history) > 100:
 self.stability_history = self.stability_history[-100:]


# Factory functions
def create_observation_normalizer(
 observation_space: Tuple[int, ...],
 **kwargs
) -> ObservationNormalizer:
 """Create observation normalizer"""
 return ObservationNormalizer(observation_space, **kwargs)


def create_reward_normalizer(**kwargs) -> RewardNormalizer:
 """Create reward normalizer"""
 return RewardNormalizer(**kwargs)


# Export functions and classes
__all__ = [
 "NormalizationConfig",
 "RunningMeanStd",
 "ObservationNormalizer",
 "RewardNormalizer",
 "VectorNormalizer",
 "AdaptiveNormalizer",
 "normalize_advantages",
 "normalize_returns",
 "create_observation_normalizer",
 "create_reward_normalizer"
]