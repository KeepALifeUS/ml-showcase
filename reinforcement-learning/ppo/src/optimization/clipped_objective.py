"""
Clipped Surrogate Objective Implementation for PPO
for stable policy optimization

Clipped objective prevents large policy updates by limiting
the ratio between new and old policies, ensuring training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
import math
from abc import ABC, abstractmethod

from ..utils.normalization import normalize_advantages


@dataclass
class ClippedObjectiveConfig:
 """Configuration for clipped objective"""
 
 # Clipping parameters
 clip_range: float = 0.2 # PPO clipping parameter
 clip_range_vf: Optional[float] = None # Value function clipping
 
 # Adaptive clipping
 adaptive_clipping: bool = False
 clip_decay_rate: float = 0.99
 min_clip_range: float = 0.05
 max_clip_range: float = 0.5
 target_kl: float = 0.01
 
 # Advanced clipping techniques
 use_kl_clip: bool = False # KL-based clipping
 kl_clip_threshold: float = 0.015
 use_double_clipping: bool = False # Double clipping
 
 # Regularization
 entropy_coef: float = 0.01 # Entropy regularization
 value_loss_coef: float = 0.5 # Value loss coefficient
 
 # Numerical stability
 eps: float = 1e-8
 max_ratio: float = 10.0 # Maximum policy ratio
 
 # Monitoring
 track_ratios: bool = True
 track_kl_divergence: bool = True


class BaseClippedObjective(ABC):
 """Base class for clipped objective implementations"""
 
 def __init__(self, config: ClippedObjectiveConfig):
 self.config = config
 self.clip_range = config.clip_range
 
 @abstractmethod
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Compute clipped policy loss"""
 pass
 
 @abstractmethod
 def compute_value_loss(
 self,
 values: torch.Tensor,
 old_values: torch.Tensor,
 returns: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Compute value function loss"""
 pass


class StandardClippedObjective(BaseClippedObjective):
 """
 Standard PPO clipped objective implementation
 
 L^CLIP(θ) = E_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
 where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
 """
 
 def __init__(self, config: ClippedObjectiveConfig):
 super().__init__(config)
 self.kl_history = []
 
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute standard clipped surrogate objective
 
 Args:
 log_probs: New policy log probabilities [batch_size]
 old_log_probs: Old policy log probabilities [batch_size] 
 advantages: Advantage estimates [batch_size]
 
 Returns:
 policy_loss, metrics_dict
 """
 
 # Normalize advantages
 if advantages.numel() > 1:
 advantages = normalize_advantages(advantages)
 
 # Compute probability ratio
 ratio = torch.exp(log_probs - old_log_probs)
 
 # Clamp ratio for numerical stability
 ratio = torch.clamp(ratio, 1 / self.config.max_ratio, self.config.max_ratio)
 
 # Clipped surrogate objective
 surr1 = ratio * advantages
 surr2 = torch.clamp(
 ratio, 
 1.0 - self.clip_range, 
 1.0 + self.clip_range
 ) * advantages
 
 # Take minimum (most conservative estimate)
 policy_loss = -torch.min(surr1, surr2).mean()
 
 # Compute metrics
 with torch.no_grad():
 # KL divergence approximation
 approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean()
 
 # Clipping metrics
 clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean()
 
 # Ratio statistics
 ratio_mean = ratio.mean()
 ratio_std = ratio.std()
 ratio_min = ratio.min()
 ratio_max = ratio.max()
 
 # Advantage statistics
 adv_mean = advantages.mean()
 adv_std = advantages.std()
 
 metrics = {
 "policy_loss": policy_loss.item(),
 "approx_kl": approx_kl.item(),
 "clip_fraction": clip_fraction.item(),
 "ratio_mean": ratio_mean.item(),
 "ratio_std": ratio_std.item(),
 "ratio_min": ratio_min.item(),
 "ratio_max": ratio_max.item(),
 "advantage_mean": adv_mean.item(),
 "advantage_std": adv_std.item(),
 "current_clip_range": self.clip_range
 }
 
 # Update adaptive clipping
 if self.config.adaptive_clipping:
 self._update_clip_range(approx_kl.item())
 metrics["adaptive_clip_range"] = self.clip_range
 
 return policy_loss, metrics
 
 def compute_value_loss(
 self,
 values: torch.Tensor,
 old_values: torch.Tensor,
 returns: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute value function loss with optional clipping
 
 Args:
 values: New value estimates [batch_size]
 old_values: Old value estimates [batch_size]
 returns: Target returns [batch_size]
 
 Returns:
 value_loss, metrics_dict
 """
 
 if self.config.clip_range_vf is None:
 # Standard MSE loss
 value_loss = F.mse_loss(values, returns)
 clipped_fraction = torch.tensor(0.0)
 else:
 # Clipped value loss
 values_clipped = old_values + torch.clamp(
 values - old_values,
 -self.config.clip_range_vf,
 self.config.clip_range_vf
 )
 
 loss_unclipped = F.mse_loss(values, returns, reduction='none')
 loss_clipped = F.mse_loss(values_clipped, returns, reduction='none')
 
 # Take maximum loss (most conservative)
 value_loss = torch.max(loss_unclipped, loss_clipped).mean()
 
 # Clipping fraction
 with torch.no_grad():
 clipped_fraction = (
 torch.abs(values - old_values) > self.config.clip_range_vf
 ).float().mean()
 
 # Value metrics
 with torch.no_grad():
 value_mean = values.mean()
 value_std = values.std()
 return_mean = returns.mean()
 return_std = returns.std()
 explained_variance = self._compute_explained_variance(values, returns)
 
 metrics = {
 "value_loss": value_loss.item(),
 "value_mean": value_mean.item(),
 "value_std": value_std.item(),
 "return_mean": return_mean.item(),
 "return_std": return_std.item(),
 "explained_variance": explained_variance.item(),
 "value_clip_fraction": clipped_fraction.item() if isinstance(clipped_fraction, torch.Tensor) else clipped_fraction
 }
 
 return value_loss, metrics
 
 def _update_clip_range(self, kl_divergence: float):
 """Update clip range based on KL divergence"""
 
 self.kl_history.append(kl_divergence)
 
 # Keep only recent history
 if len(self.kl_history) > 100:
 self.kl_history = self.kl_history[-100:]
 
 if len(self.kl_history) < 10:
 return
 
 # Compute recent KL trend
 recent_kl = np.mean(self.kl_history[-10:])
 
 # Adapt clip range
 if recent_kl > self.config.target_kl * 1.5:
 # KL too high, reduce clipping (more conservative)
 self.clip_range = max(
 self.config.min_clip_range,
 self.clip_range * self.config.clip_decay_rate
 )
 elif recent_kl < self.config.target_kl * 0.5:
 # KL too low, increase clipping (less conservative)
 self.clip_range = min(
 self.config.max_clip_range,
 self.clip_range / self.config.clip_decay_rate
 )
 
 def _compute_explained_variance(
 self, 
 predictions: torch.Tensor, 
 targets: torch.Tensor
 ) -> torch.Tensor:
 """Compute explained variance"""
 var_target = targets.var()
 return 1 - (targets - predictions).var() / (var_target + self.config.eps)


class KLClippedObjective(BaseClippedObjective):
 """
 KL-divergence based clipped objective
 
 Uses KL divergence instead of ratio clipping for more principled updates
 """
 
 def __init__(self, config: ClippedObjectiveConfig):
 super().__init__(config)
 self.kl_history = []
 
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 action_dist: Optional[Union[Categorical, Normal]] = None,
 old_action_dist: Optional[Union[Categorical, Normal]] = None,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute KL-clipped policy objective
 
 Clips based on KL divergence instead of probability ratios
 """
 
 # Standard ratio computation
 ratio = torch.exp(log_probs - old_log_probs)
 
 # Normalize advantages
 if advantages.numel() > 1:
 advantages = normalize_advantages(advantages)
 
 # Basic surrogate objective
 surrogate_loss = -(ratio * advantages).mean()
 
 # KL divergence computation
 if action_dist is not None and old_action_dist is not None:
 if isinstance(action_dist, Categorical):
 kl_div = torch.distributions.kl.kl_divergence(
 old_action_dist, action_dist
 ).mean()
 elif isinstance(action_dist, Normal):
 kl_div = torch.distributions.kl.kl_divergence(
 old_action_dist, action_dist
 ).sum(dim=-1).mean()
 else:
 # Approximation using log probs
 kl_div = ((ratio - 1) - (log_probs - old_log_probs)).mean()
 else:
 # Approximate KL using log prob difference
 kl_div = ((ratio - 1) - (log_probs - old_log_probs)).mean()
 
 # Apply KL clipping
 if kl_div > self.config.kl_clip_threshold:
 # Reduce update magnitude
 clip_factor = self.config.kl_clip_threshold / (kl_div + self.config.eps)
 surrogate_loss = surrogate_loss * clip_factor
 
 policy_loss = surrogate_loss
 
 # Metrics
 with torch.no_grad():
 clip_fraction = (kl_div > self.config.kl_clip_threshold).float()
 
 metrics = {
 "policy_loss": policy_loss.item(),
 "kl_divergence": kl_div.item(),
 "kl_clip_fraction": clip_fraction.item(),
 "ratio_mean": ratio.mean().item(),
 "ratio_std": ratio.std().item()
 }
 
 return policy_loss, metrics
 
 def compute_value_loss(self, *args, **kwargs):
 """Reuse standard value loss"""
 return super().compute_value_loss(*args, **kwargs)


class DoubleClippedObjective(BaseClippedObjective):
 """
 Double clipped objective for additional stability
 
 Applies clipping to both positive and negative advantages
 Provides extra protection against large policy updates
 """
 
 def __init__(self, config: ClippedObjectiveConfig):
 super().__init__(config)
 
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute double clipped objective
 
 Clips both surr1 and surr2 separately for positive/negative advantages
 """
 
 # Normalize advantages
 if advantages.numel() > 1:
 advantages = normalize_advantages(advantages)
 
 # Compute ratio
 ratio = torch.exp(log_probs - old_log_probs)
 ratio = torch.clamp(ratio, 1 / self.config.max_ratio, self.config.max_ratio)
 
 # Standard clipping
 surr1 = ratio * advantages
 surr2 = torch.clamp(
 ratio,
 1.0 - self.clip_range,
 1.0 + self.clip_range
 ) * advantages
 
 # Double clipping - separate handling for positive/negative advantages
 positive_mask = advantages > 0
 negative_mask = advantages <= 0
 
 # For positive advantages, take minimum (conservative)
 positive_loss = torch.where(
 positive_mask,
 torch.min(surr1, surr2),
 torch.zeros_like(surr1)
 )
 
 # For negative advantages, take maximum (less conservative on punishment)
 negative_loss = torch.where(
 negative_mask,
 torch.max(surr1, surr2),
 torch.zeros_like(surr1)
 )
 
 # Combine losses
 policy_loss = -(positive_loss + negative_loss).mean()
 
 # Metrics
 with torch.no_grad():
 approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean()
 clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean()
 
 metrics = {
 "policy_loss": policy_loss.item(),
 "approx_kl": approx_kl.item(),
 "clip_fraction": clip_fraction.item(),
 "ratio_mean": ratio.mean().item(),
 "ratio_std": ratio.std().item(),
 "positive_advantages": positive_mask.float().mean().item(),
 "negative_advantages": negative_mask.float().mean().item()
 }
 
 return policy_loss, metrics
 
 def compute_value_loss(self, *args, **kwargs):
 """Reuse standard value loss"""
 return super().compute_value_loss(*args, **kwargs)


class AdaptiveClippedObjective(StandardClippedObjective):
 """
 Adaptive clipped objective with dynamic parameter adjustment
 
 Automatically adjusts clipping parameters based on:
 - KL divergence trends
 - Training stability metrics
 - Performance indicators
 """
 
 def __init__(
 self,
 config: ClippedObjectiveConfig,
 adaptation_window: int = 100,
 stability_threshold: float = 0.1
 ):
 super().__init__(config)
 self.adaptation_window = adaptation_window
 self.stability_threshold = stability_threshold
 
 # Tracking metrics
 self.loss_history = []
 self.kl_history = []
 self.ratio_history = []
 
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Compute adaptive clipped objective"""
 
 # Standard computation
 policy_loss, metrics = super().compute_policy_loss(
 log_probs, old_log_probs, advantages, **kwargs
 )
 
 # Update tracking metrics
 self.loss_history.append(policy_loss.item())
 self.kl_history.append(metrics["approx_kl"])
 self.ratio_history.append(metrics["ratio_std"])
 
 # Adaptive adjustment
 if len(self.loss_history) >= self.adaptation_window:
 self._adapt_parameters()
 
 # Update metrics
 metrics.update({
 "adapted_clip_range": self.clip_range,
 "loss_stability": self._compute_stability_metric(self.loss_history),
 "kl_stability": self._compute_stability_metric(self.kl_history)
 })
 
 return policy_loss, metrics
 
 def _adapt_parameters(self):
 """Adapt clipping parameters based on training history"""
 
 # Get recent metrics
 recent_kl = self.kl_history[-self.adaptation_window:]
 recent_ratios = self.ratio_history[-self.adaptation_window:]
 recent_losses = self.loss_history[-self.adaptation_window:]
 
 # Stability metrics
 kl_stability = self._compute_stability_metric(recent_kl)
 ratio_stability = self._compute_stability_metric(recent_ratios)
 loss_stability = self._compute_stability_metric(recent_losses)
 
 # Adaptation logic
 if kl_stability < self.stability_threshold:
 # High KL instability - reduce clipping
 self.clip_range = max(
 self.config.min_clip_range,
 self.clip_range * 0.95
 )
 elif ratio_stability > self.stability_threshold * 2:
 # Stable ratios - can increase clipping slightly
 self.clip_range = min(
 self.config.max_clip_range,
 self.clip_range * 1.02
 )
 
 def _compute_stability_metric(self, values: List[float]) -> float:
 """Compute stability metric (inverse of coefficient of variation)"""
 if len(values) < 2:
 return 1.0
 
 values_array = np.array(values)
 mean_val = np.mean(values_array)
 std_val = np.std(values_array)
 
 if abs(mean_val) < 1e-8:
 return 1.0
 
 cv = std_val / abs(mean_val) # Coefficient of variation
 return 1.0 / (1.0 + cv) # Stability metric


# Factory function for creating clipped objectives
def create_clipped_objective(
 objective_type: str = "standard",
 config: Optional[ClippedObjectiveConfig] = None,
 **kwargs
) -> BaseClippedObjective:
 """Create clipped objective of specified type"""
 
 if config is None:
 config = ClippedObjectiveConfig()
 
 if objective_type == "standard":
 return StandardClippedObjective(config)
 elif objective_type == "kl_clipped":
 return KLClippedObjective(config)
 elif objective_type == "double_clipped":
 return DoubleClippedObjective(config)
 elif objective_type == "adaptive":
 return AdaptiveClippedObjective(config, **kwargs)
 else:
 raise ValueError(f"Unknown objective type: {objective_type}")


# Export classes and functions
__all__ = [
 "ClippedObjectiveConfig",
 "BaseClippedObjective",
 "StandardClippedObjective",
 "KLClippedObjective",
 "DoubleClippedObjective",
 "AdaptiveClippedObjective",
 "create_clipped_objective"
]