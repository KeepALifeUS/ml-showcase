"""
KL Penalty Implementation for PPO
for principled policy updates

KL penalty provides alternative to clipping by adding
KL divergence penalty to objective function:
L = L_SURR - β * KL(π_old, π_new)

Adaptive β ensures stable training without hard clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
import numpy as np
import math
from abc import ABC, abstractmethod

from ..utils.normalization import normalize_advantages


@dataclass
class KLPenaltyConfig:
 """Configuration for KL penalty methods"""
 
 # KL penalty parameters
 kl_coef: float = 0.2 # Initial KL coefficient
 target_kl: float = 0.01 # Target KL divergence
 
 # Adaptive KL adjustment
 adaptive_kl: bool = True
 kl_adjustment_factor: float = 2.0 # Adjustment multiplier
 min_kl_coef: float = 0.001 # Minimum KL coefficient
 max_kl_coef: float = 10.0 # Maximum KL coefficient
 
 # KL penalty variants
 penalty_type: str = "standard" # standard, squared, huber
 huber_delta: float = 1.0 # Huber loss delta
 
 # Regularization
 entropy_coef: float = 0.01 # Entropy regularization
 value_loss_coef: float = 0.5 # Value loss coefficient
 
 # Advanced features
 use_natural_gradient: bool = False # Natural policy gradient
 use_trust_region: bool = False # Trust region constraints
 trust_region_bound: float = 0.01 # Trust region KL bound
 
 # Monitoring
 kl_tolerance: float = 1.5 # KL tolerance factor
 early_stopping_kl: float = 0.05 # Early stopping threshold
 
 # Numerical stability
 eps: float = 1e-8
 max_kl: float = 100.0 # Maximum KL for clipping


class BaseKLPenalty(ABC):
 """Base class for KL penalty implementations"""
 
 def __init__(self, config: KLPenaltyConfig):
 self.config = config
 self.kl_coef = config.kl_coef
 
 @abstractmethod
 def compute_kl_penalty(
 self,
 action_dist: Union[Categorical, Normal],
 old_action_dist: Union[Categorical, Normal],
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Compute KL penalty term"""
 pass
 
 @abstractmethod
 def update_kl_coef(self, kl_divergence: float) -> float:
 """Update KL coefficient based on measured KL"""
 pass


class StandardKLPenalty(BaseKLPenalty):
 """
 Standard KL penalty implementation
 
 L = L_SURR - β * KL(π_old, π_new)
 
 Adaptively adjusts β to maintain target KL divergence
 """
 
 def __init__(self, config: KLPenaltyConfig):
 super().__init__(config)
 self.kl_history = []
 
 def compute_policy_loss_with_kl_penalty(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 action_dist: Union[Categorical, Normal],
 old_action_dist: Union[Categorical, Normal],
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute policy loss with KL penalty
 
 Args:
 log_probs: New policy log probabilities
 old_log_probs: Old policy log probabilities
 advantages: Advantage estimates
 action_dist: New action distribution
 old_action_dist: Old action distribution
 
 Returns:
 total_loss, metrics_dict
 """
 
 # Normalize advantages
 if advantages.numel() > 1:
 advantages = normalize_advantages(advantages)
 
 # Compute probability ratio
 ratio = torch.exp(log_probs - old_log_probs)
 
 # Surrogate loss
 surrogate_loss = -(ratio * advantages).mean()
 
 # KL penalty
 kl_penalty, kl_metrics = self.compute_kl_penalty(
 action_dist, old_action_dist
 )
 
 # Total policy loss
 policy_loss = surrogate_loss + self.kl_coef * kl_penalty
 
 # Update adaptive KL coefficient
 if self.config.adaptive_kl:
 self.update_kl_coef(kl_metrics["kl_divergence"])
 
 # Metrics
 metrics = {
 "policy_loss": policy_loss.item(),
 "surrogate_loss": surrogate_loss.item(),
 "kl_penalty": kl_penalty.item(),
 "kl_coefficient": self.kl_coef,
 "ratio_mean": ratio.mean().item(),
 "ratio_std": ratio.std().item(),
 **kl_metrics
 }
 
 return policy_loss, metrics
 
 def compute_kl_penalty(
 self,
 action_dist: Union[Categorical, Normal],
 old_action_dist: Union[Categorical, Normal],
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Compute KL penalty term"""
 
 # Compute KL divergence
 if isinstance(action_dist, Categorical) and isinstance(old_action_dist, Categorical):
 kl_div = kl_divergence(old_action_dist, action_dist)
 elif isinstance(action_dist, Normal) and isinstance(old_action_dist, Normal):
 kl_div = kl_divergence(old_action_dist, action_dist)
 if len(kl_div.shape) > 1:
 kl_div = kl_div.sum(dim=-1) # Sum over action dimensions
 else:
 raise ValueError("Action distributions must be of same type")
 
 # Clamp KL for numerical stability
 kl_div = torch.clamp(kl_div, 0, self.config.max_kl)
 
 # Apply penalty function
 if self.config.penalty_type == "standard":
 kl_penalty = kl_div.mean()
 elif self.config.penalty_type == "squared":
 kl_penalty = (kl_div ** 2).mean()
 elif self.config.penalty_type == "huber":
 kl_penalty = self._huber_loss(kl_div, self.config.huber_delta).mean()
 else:
 raise ValueError(f"Unknown penalty type: {self.config.penalty_type}")
 
 # Metrics
 with torch.no_grad():
 kl_mean = kl_div.mean()
 kl_std = kl_div.std()
 kl_max = kl_div.max()
 
 metrics = {
 "kl_divergence": kl_mean.item(),
 "kl_std": kl_std.item(),
 "kl_max": kl_max.item(),
 "kl_penalty_value": kl_penalty.item()
 }
 
 return kl_penalty, metrics
 
 def update_kl_coef(self, kl_divergence: float) -> float:
 """
 Adaptively update KL coefficient
 
 If KL > target: increase β (more penalty)
 If KL < target: decrease β (less penalty)
 """
 
 self.kl_history.append(kl_divergence)
 
 # Keep only recent history
 if len(self.kl_history) > 100:
 self.kl_history = self.kl_history[-100:]
 
 target_kl = self.config.target_kl
 
 if kl_divergence > target_kl * self.config.kl_tolerance:
 # KL too high, increase penalty
 self.kl_coef = min(
 self.config.max_kl_coef,
 self.kl_coef * self.config.kl_adjustment_factor
 )
 elif kl_divergence < target_kl / self.config.kl_tolerance:
 # KL too low, decrease penalty
 self.kl_coef = max(
 self.config.min_kl_coef,
 self.kl_coef / self.config.kl_adjustment_factor
 )
 
 return self.kl_coef
 
 def _huber_loss(self, x: torch.Tensor, delta: float) -> torch.Tensor:
 """Huber loss for robust KL penalty"""
 abs_x = torch.abs(x)
 return torch.where(
 abs_x <= delta,
 0.5 * x ** 2,
 delta * abs_x - 0.5 * delta ** 2
 )
 
 def should_early_stop(self) -> bool:
 """Check if KL divergence too high"""
 if len(self.kl_history) == 0:
 return False
 
 recent_kl = self.kl_history[-1]
 return recent_kl > self.config.early_stopping_kl


class NaturalPolicyGradientKL(BaseKLPenalty):
 """
 Natural Policy Gradient with KL penalty
 
 Uses Fisher Information Matrix for natural gradient updates
 More principled than standard gradient descent
 """
 
 def __init__(self, config: KLPenaltyConfig):
 super().__init__(config)
 self.fisher_matrix = None
 
 def compute_natural_policy_gradient(
 self,
 policy_gradient: torch.Tensor,
 action_dist: Union[Categorical, Normal],
 **kwargs
 ) -> torch.Tensor:
 """
 Compute natural policy gradient using Fisher Information Matrix
 
 Natural gradient = F^(-1) * ∇_θ L
 where F is Fisher Information Matrix
 """
 
 # Compute Fisher Information Matrix approximation
 fisher_matrix = self._compute_fisher_information(action_dist)
 
 # Regularize Fisher matrix
 regularization = 1e-4 * torch.eye(
 fisher_matrix.shape[0],
 device=fisher_matrix.device
 )
 fisher_regularized = fisher_matrix + regularization
 
 # Compute natural gradient
 try:
 natural_gradient = torch.linalg.solve(fisher_regularized, policy_gradient)
 except torch.linalg.LinAlgError:
 # Fallback to standard gradient if matrix singular
 natural_gradient = policy_gradient
 
 return natural_gradient
 
 def _compute_fisher_information(
 self,
 action_dist: Union[Categorical, Normal]
 ) -> torch.Tensor:
 """
 Compute Fisher Information Matrix approximation
 
 F_ij = E[∇_θ log π(a|s) * ∇_θ log π(a|s)^T]
 """
 
 if isinstance(action_dist, Categorical):
 # For discrete actions
 probs = action_dist.probs
 fisher = torch.diag(probs) - torch.outer(probs, probs)
 elif isinstance(action_dist, Normal):
 # For continuous actions (simplified approximation)
 mean = action_dist.mean
 std = action_dist.scale
 
 # Approximate Fisher matrix
 fisher_mean = 1 / (std ** 2)
 fisher_std = 2 / (std ** 2)
 
 fisher = torch.diag(torch.cat([fisher_mean.flatten(), fisher_std.flatten()]))
 else:
 raise NotImplementedError(f"Fisher matrix for {type(action_dist)} not implemented")
 
 return fisher
 
 def compute_kl_penalty(self, *args, **kwargs):
 """Reuse standard KL penalty computation"""
 # Implementation would be similar to StandardKLPenalty
 # but with natural gradient corrections
 pass
 
 def update_kl_coef(self, kl_divergence: float) -> float:
 """Update KL coefficient with natural gradient considerations"""
 # More conservative updates for natural gradients
 target_kl = self.config.target_kl
 
 if kl_divergence > target_kl * 1.2:
 self.kl_coef = min(
 self.config.max_kl_coef,
 self.kl_coef * 1.5
 )
 elif kl_divergence < target_kl * 0.8:
 self.kl_coef = max(
 self.config.min_kl_coef,
 self.kl_coef / 1.5
 )
 
 return self.kl_coef


class TrustRegionKLPenalty(BaseKLPenalty):
 """
 Trust Region KL penalty
 
 Implements hard constraint on KL divergence:
 L = L_SURR subject to KL(π_old, π_new) ≤ δ
 
 Uses Lagrange multipliers for constraint optimization
 """
 
 def __init__(self, config: KLPenaltyConfig):
 super().__init__(config)
 self.lagrange_multiplier = 1.0
 
 def compute_policy_loss_with_trust_region(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor,
 action_dist: Union[Categorical, Normal],
 old_action_dist: Union[Categorical, Normal],
 **kwargs
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Compute policy loss with trust region constraint
 
 Uses quadratic approximation for constraint handling
 """
 
 # Normalize advantages
 if advantages.numel() > 1:
 advantages = normalize_advantages(advantages)
 
 # Compute probability ratio
 ratio = torch.exp(log_probs - old_log_probs)
 
 # Surrogate loss
 surrogate_loss = -(ratio * advantages).mean()
 
 # KL divergence
 if isinstance(action_dist, Categorical):
 kl_div = kl_divergence(old_action_dist, action_dist).mean()
 else:
 kl_div = kl_divergence(old_action_dist, action_dist).sum(dim=-1).mean()
 
 # Trust region constraint violation
 constraint_violation = torch.clamp(
 kl_div - self.config.trust_region_bound,
 min=0.0
 )
 
 # Lagrangian: L + λ * constraint_violation
 policy_loss = surrogate_loss + self.lagrange_multiplier * constraint_violation
 
 # Update Lagrange multiplier
 self._update_lagrange_multiplier(constraint_violation.item())
 
 metrics = {
 "policy_loss": policy_loss.item(),
 "surrogate_loss": surrogate_loss.item(),
 "kl_divergence": kl_div.item(),
 "constraint_violation": constraint_violation.item(),
 "lagrange_multiplier": self.lagrange_multiplier,
 "trust_region_bound": self.config.trust_region_bound
 }
 
 return policy_loss, metrics
 
 def _update_lagrange_multiplier(self, constraint_violation: float):
 """Update Lagrange multiplier based on constraint violation"""
 
 learning_rate = 0.01
 
 if constraint_violation > 0:
 # Constraint violated, increase multiplier
 self.lagrange_multiplier += learning_rate * constraint_violation
 else:
 # Constraint satisfied, decrease multiplier
 self.lagrange_multiplier = max(
 0.0,
 self.lagrange_multiplier - learning_rate * abs(constraint_violation)
 )
 
 # Clamp multiplier
 self.lagrange_multiplier = np.clip(self.lagrange_multiplier, 0.0, 100.0)
 
 def compute_kl_penalty(self, *args, **kwargs):
 """Trust region uses hard constraints instead of soft penalties"""
 pass
 
 def update_kl_coef(self, kl_divergence: float) -> float:
 """Trust region doesn't use adaptive KL coefficient"""
 return self.kl_coef


class AdaptiveKLScheduler:
 """
 Advanced KL coefficient scheduler
 
 Adapts KL coefficient based on:
 - Training progress
 - Performance metrics
 - Stability indicators
 """
 
 def __init__(
 self,
 initial_kl_coef: float = 0.2,
 target_kl: float = 0.01,
 schedule_type: str = "adaptive", # adaptive, linear, exponential
 adaptation_window: int = 100
 ):
 self.initial_kl_coef = initial_kl_coef
 self.current_kl_coef = initial_kl_coef
 self.target_kl = target_kl
 self.schedule_type = schedule_type
 self.adaptation_window = adaptation_window
 
 # Tracking history
 self.kl_history = []
 self.loss_history = []
 self.performance_history = []
 
 def update_kl_coef(
 self,
 kl_divergence: float,
 policy_loss: float,
 performance_metric: Optional[float] = None
 ) -> float:
 """Update KL coefficient based on multiple signals"""
 
 # Update history
 self.kl_history.append(kl_divergence)
 self.loss_history.append(policy_loss)
 if performance_metric is not None:
 self.performance_history.append(performance_metric)
 
 # Trim history
 if len(self.kl_history) > self.adaptation_window:
 self.kl_history = self.kl_history[-self.adaptation_window:]
 self.loss_history = self.loss_history[-self.adaptation_window:]
 if self.performance_history:
 self.performance_history = self.performance_history[-self.adaptation_window:]
 
 if self.schedule_type == "adaptive":
 return self._adaptive_update(kl_divergence, policy_loss, performance_metric)
 elif self.schedule_type == "linear":
 return self._linear_decay()
 elif self.schedule_type == "exponential":
 return self._exponential_decay()
 else:
 return self.current_kl_coef
 
 def _adaptive_update(
 self,
 kl_divergence: float,
 policy_loss: float,
 performance_metric: Optional[float]
 ) -> float:
 """Adaptive KL coefficient update"""
 
 # KL-based adjustment
 kl_ratio = kl_divergence / self.target_kl
 
 if kl_ratio > 1.5:
 # Too much divergence
 kl_adjustment = 1.5
 elif kl_ratio < 0.5:
 # Too little divergence
 kl_adjustment = 0.8
 else:
 # Good range
 kl_adjustment = 1.0
 
 # Performance-based adjustment
 if performance_metric is not None and len(self.performance_history) >= 10:
 recent_performance = np.mean(self.performance_history[-10:])
 older_performance = np.mean(self.performance_history[-20:-10]) if len(self.performance_history) >= 20 else recent_performance
 
 if recent_performance > older_performance:
 # Improving performance
 performance_adjustment = 0.95
 else:
 # Degrading performance
 performance_adjustment = 1.05
 else:
 performance_adjustment = 1.0
 
 # Combined adjustment
 total_adjustment = kl_adjustment * performance_adjustment
 
 # Update coefficient
 self.current_kl_coef *= total_adjustment
 self.current_kl_coef = np.clip(self.current_kl_coef, 0.001, 10.0)
 
 return self.current_kl_coef
 
 def _linear_decay(self) -> float:
 """Linear decay schedule"""
 decay_rate = 0.999
 self.current_kl_coef = max(0.001, self.current_kl_coef * decay_rate)
 return self.current_kl_coef
 
 def _exponential_decay(self) -> float:
 """Exponential decay schedule"""
 decay_rate = 0.995
 self.current_kl_coef = max(0.001, self.current_kl_coef * decay_rate)
 return self.current_kl_coef


# Factory function for creating KL penalty objects
def create_kl_penalty(
 penalty_type: str = "standard",
 config: Optional[KLPenaltyConfig] = None,
 **kwargs
) -> BaseKLPenalty:
 """Create KL penalty object of specified type"""
 
 if config is None:
 config = KLPenaltyConfig()
 
 if penalty_type == "standard":
 return StandardKLPenalty(config)
 elif penalty_type == "natural_gradient":
 return NaturalPolicyGradientKL(config)
 elif penalty_type == "trust_region":
 return TrustRegionKLPenalty(config)
 else:
 raise ValueError(f"Unknown penalty type: {penalty_type}")


# Export classes and functions
__all__ = [
 "KLPenaltyConfig",
 "BaseKLPenalty",
 "StandardKLPenalty",
 "NaturalPolicyGradientKL",
 "TrustRegionKLPenalty",
 "AdaptiveKLScheduler",
 "create_kl_penalty"
]