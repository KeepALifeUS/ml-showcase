"""
Proximal Policy Optimization (PPO) Core Implementation
for stable RL training in crypto trading

Implements the main PPO algorithm with:
- Clipped surrogate objective for stability
- Entropy regularization for exploration
- Value function learning with optional clipping
- Adaptive KL coefficient for controlling policy updates
- Production-ready logging and monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import logging
from collections import defaultdict

from ..utils.normalization import RunningMeanStd, normalize_advantages
from ..utils.scheduling import LinearSchedule, ExponentialSchedule


@dataclass
class PPOConfig:
 """PPO Configuration with """
 # Core PPO hyperparameters
 learning_rate: float = 3e-4
 gamma: float = 0.99 # Discount factor
 gae_lambda: float = 0.95 # GAE lambda parameter
 clip_range: float = 0.2 # PPO clipping parameter
 clip_range_vf: Optional[float] = None # Value function clipping
 
 # Training parameters
 n_epochs: int = 10 # Number of epochs per update
 batch_size: int = 64 # Mini-batch size
 max_grad_norm: float = 0.5 # Gradient clipping
 
 # Regularization
 ent_coef: float = 0.01 # Entropy regularization coefficient
 vf_coef: float = 0.5 # Value function loss coefficient
 
 # KL divergence monitoring
 target_kl: Optional[float] = 0.01 # Target KL divergence
 kl_coef: float = 0.0 # Adaptive KL penalty coefficient
 
 # Normalization
 normalize_advantage: bool = True
 normalize_returns: bool = False
 
 # Logging and monitoring
 log_interval: int = 100
 save_interval: int = 1000
 eval_interval: int = 500
 
 # Performance optimization
 device: str = "cuda" if torch.cuda.is_available() else "cpu"
 num_workers: int = 4
 pin_memory: bool = True


class PPOLoss:
 """PPO Loss Functions performance patterns"""
 
 def __init__(self, config: PPOConfig):
 self.config = config
 self.clip_range = config.clip_range
 self.vf_coef = config.vf_coef
 self.ent_coef = config.ent_coef
 
 def compute_policy_loss(
 self,
 log_probs: torch.Tensor,
 old_log_probs: torch.Tensor,
 advantages: torch.Tensor
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Computes clipped surrogate objective
 
 Args:
 log_probs: New log probabilities
 old_log_probs: Old log probabilities
 advantages: Estimated advantages
 
 Returns:
 policy_loss, metrics_dict
 """
 # Ratio between old and new policy
 ratio = torch.exp(log_probs - old_log_probs)
 
 # Clipped surrogate objective
 surr1 = ratio * advantages
 surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
 
 policy_loss = -torch.min(surr1, surr2).mean()
 
 # Metrics for monitoring
 with torch.no_grad():
 approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean().item()
 clip_fraction = (torch.abs(ratio - 1) > self.clip_range).float().mean().item()
 
 metrics = {
 "policy_loss": policy_loss.item(),
 "approx_kl": approx_kl,
 "clip_fraction": clip_fraction,
 "ratio_mean": ratio.mean().item(),
 "ratio_std": ratio.std().item()
 }
 
 return policy_loss, metrics
 
 def compute_value_loss(
 self,
 values: torch.Tensor,
 old_values: torch.Tensor,
 returns: torch.Tensor
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """
 Computes value function loss with optional clipping
 """
 if self.config.clip_range_vf is None:
 # Simple MSE loss
 value_loss = F.mse_loss(values, returns)
 else:
 # Clipped value loss for stability
 values_clipped = old_values + torch.clamp(
 values - old_values,
 -self.config.clip_range_vf,
 self.config.clip_range_vf
 )
 
 value_loss_unclipped = F.mse_loss(values, returns)
 value_loss_clipped = F.mse_loss(values_clipped, returns)
 value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
 
 metrics = {
 "value_loss": value_loss.item(),
 "value_mean": values.mean().item(),
 "return_mean": returns.mean().item(),
 "value_std": values.std().item()
 }
 
 return value_loss, metrics
 
 def compute_entropy_loss(
 self,
 action_dist: Union[Categorical, Normal]
 ) -> Tuple[torch.Tensor, Dict[str, float]]:
 """Computes entropy loss for exploration"""
 entropy = action_dist.entropy().mean()
 entropy_loss = -entropy # Negative because we want to maximize entropy
 
 metrics = {
 "entropy": entropy.item(),
 "entropy_loss": entropy_loss.item()
 }
 
 return entropy_loss, metrics


class PPOAlgorithm:
 """
 Main PPO algorithm with Features:
 - Clipped surrogate objective
 - GAE for advantage estimation 
 - Value function learning
 - Entropy regularization
 - Adaptive KL penalty
 - Production monitoring
 """
 
 def __init__(
 self,
 actor_critic: nn.Module,
 config: PPOConfig,
 optimizer: Optional[torch.optim.Optimizer] = None
 ):
 self.config = config
 self.actor_critic = actor_critic.to(config.device)
 
 # Optimizer setup
 if optimizer is None:
 self.optimizer = torch.optim.Adam(
 self.actor_critic.parameters(),
 lr=config.learning_rate,
 eps=1e-8
 )
 else:
 self.optimizer = optimizer
 
 # Loss computation
 self.loss_fn = PPOLoss(config)
 
 # Learning rate scheduling
 self.lr_scheduler = LinearSchedule(
 initial_value=config.learning_rate,
 final_value=config.learning_rate * 0.1
 )
 
 # Adaptive KL coefficient
 self.kl_coef = config.kl_coef
 self.target_kl = config.target_kl
 
 # Statistics tracking
 self.running_stats = defaultdict(list)
 self.update_count = 0
 
 # Logger setup
 self.logger = logging.getLogger(__name__)
 
 def update(
 self,
 rollout_buffer,
 progress_remaining: float = 1.0
 ) -> Dict[str, float]:
 """
 Performs PPO update on data from rollout buffer
 
 Args:
 rollout_buffer: Buffer with collected trajectories
 progress_remaining: Training progress (1.0 to 0.0)
 
 Returns:
 Dictionary with training metrics
 """
 # Update learning rate
 current_lr = self.lr_scheduler.value(1.0 - progress_remaining)
 for param_group in self.optimizer.param_groups:
 param_group['lr'] = current_lr
 
 # Get data from buffer
 batch_data = rollout_buffer.get()
 
 # Accumulate metrics
 all_metrics = defaultdict(list)
 
 # Multiple training epochs
 for epoch in range(self.config.n_epochs):
 # Early stopping by KL divergence
 if self.target_kl is not None:
 if len(all_metrics["approx_kl"]) > 0:
 mean_kl = np.mean(all_metrics["approx_kl"][-10:]) # Last 10 batches
 if mean_kl > 1.5 * self.target_kl:
 self.logger.info(f"Early stopping at epoch {epoch} due to high KL: {mean_kl:.4f}")
 break
 
 # Training on mini-batches
 for batch_indices in rollout_buffer.get_minibatch_indices(self.config.batch_size):
 batch_metrics = self._update_step(batch_data, batch_indices)
 
 # Accumulate metrics
 for key, value in batch_metrics.items():
 all_metrics[key].append(value)
 
 # Compute final metrics
 final_metrics = {}
 for key, values in all_metrics.items():
 final_metrics[key] = np.mean(values)
 
 # Update adaptive KL coefficient
 if self.target_kl is not None and final_metrics.get("approx_kl"):
 self._update_kl_coef(final_metrics["approx_kl"])
 
 final_metrics.update({
 "learning_rate": current_lr,
 "kl_coef": self.kl_coef,
 "update_count": self.update_count
 })
 
 self.update_count += 1
 
 # Log metrics periodically
 if self.update_count % self.config.log_interval == 0:
 self._log_metrics(final_metrics)
 
 return final_metrics
 
 def _update_step(
 self,
 batch_data: Dict[str, torch.Tensor],
 batch_indices: np.ndarray
 ) -> Dict[str, float]:
 """Performs one step training on mini-batch"""
 
 # Extract data for current batch
 observations = batch_data["observations"][batch_indices]
 actions = batch_data["actions"][batch_indices]
 old_log_probs = batch_data["log_probs"][batch_indices]
 old_values = batch_data["values"][batch_indices]
 advantages = batch_data["advantages"][batch_indices]
 returns = batch_data["returns"][batch_indices]
 
 # Normalize advantages
 if self.config.normalize_advantage:
 advantages = normalize_advantages(advantages)
 
 # Forward pass through actor-critic
 action_dist, values = self.actor_critic(observations)
 
 # Compute log probabilities for selected actions
 if isinstance(action_dist, Categorical):
 log_probs = action_dist.log_prob(actions.squeeze())
 elif isinstance(action_dist, Normal):
 log_probs = action_dist.log_prob(actions).sum(dim=-1)
 else:
 raise NotImplementedError(f"Action distribution {type(action_dist)} not supported")
 
 # Compute losses
 policy_loss, policy_metrics = self.loss_fn.compute_policy_loss(
 log_probs, old_log_probs, advantages
 )
 
 value_loss, value_metrics = self.loss_fn.compute_value_loss(
 values.squeeze(), old_values, returns
 )
 
 entropy_loss, entropy_metrics = self.loss_fn.compute_entropy_loss(action_dist)
 
 # KL penalty (if used)
 kl_penalty = 0.0
 if self.kl_coef > 0:
 kl_penalty = self.kl_coef * policy_metrics["approx_kl"]
 
 # Total loss
 total_loss = (
 policy_loss +
 self.config.vf_coef * value_loss +
 self.config.ent_coef * entropy_loss +
 kl_penalty
 )
 
 # Backward pass
 self.optimizer.zero_grad()
 total_loss.backward()
 
 # Gradient clipping
 if self.config.max_grad_norm > 0:
 grad_norm = torch.nn.utils.clip_grad_norm_(
 self.actor_critic.parameters(),
 self.config.max_grad_norm
 )
 else:
 grad_norm = 0.0
 
 self.optimizer.step()
 
 # Combine all metrics
 metrics = {
 "total_loss": total_loss.item(),
 "grad_norm": grad_norm,
 "kl_penalty": kl_penalty,
 **policy_metrics,
 **value_metrics,
 **entropy_metrics
 }
 
 return metrics
 
 def _update_kl_coef(self, approx_kl: float):
 """Updates adaptive KL coefficient"""
 if approx_kl < self.target_kl / 1.5:
 self.kl_coef /= 2
 elif approx_kl > self.target_kl * 1.5:
 self.kl_coef *= 2
 
 # Clamp KL coefficient
 self.kl_coef = np.clip(self.kl_coef, 0.0, 1.0)
 
 def _log_metrics(self, metrics: Dict[str, float]):
 """Logs training metrics"""
 log_msg = f"Update {self.update_count}: "
 log_msg += f"Policy Loss: {metrics['policy_loss']:.4f}, "
 log_msg += f"Value Loss: {metrics['value_loss']:.4f}, "
 log_msg += f"Entropy: {metrics['entropy']:.4f}, "
 log_msg += f"KL: {metrics['approx_kl']:.4f}, "
 log_msg += f"Clip Fraction: {metrics['clip_fraction']:.3f}"
 
 self.logger.info(log_msg)
 
 def save_checkpoint(self, filepath: str):
 """Saves checkpoint for recovery training"""
 checkpoint = {
 "actor_critic": self.actor_critic.state_dict(),
 "optimizer": self.optimizer.state_dict(),
 "config": self.config,
 "update_count": self.update_count,
 "kl_coef": self.kl_coef
 }
 torch.save(checkpoint, filepath)
 self.logger.info(f"Checkpoint saved to {filepath}")
 
 def load_checkpoint(self, filepath: str):
 """Loads checkpoint for continuing training"""
 checkpoint = torch.load(filepath, map_location=self.config.device)
 
 self.actor_critic.load_state_dict(checkpoint["actor_critic"])
 self.optimizer.load_state_dict(checkpoint["optimizer"])
 self.update_count = checkpoint["update_count"]
 self.kl_coef = checkpoint["kl_coef"]
 
 self.logger.info(f"Checkpoint loaded from {filepath}")


# Export main classes
__all__ = [
 "PPOConfig",
 "PPOLoss", 
 "PPOAlgorithm"
]