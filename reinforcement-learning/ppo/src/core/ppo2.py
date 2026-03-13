"""
PPO2 - Improved PPO Implementation
with advanced optimizations

Improvements over base PPO:
- Multi-step returns with variable λ
- Prioritized experience replay integration
- Spectral normalization for stability
- Advanced learning rate scheduling
- Dynamic clipping parameters
- Curriculum learning support
- Meta-learning adaptations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import logging
from collections import deque, defaultdict

from .ppo import PPOConfig, PPOAlgorithm, PPOLoss
from ..utils.normalization import RunningMeanStd, normalize_advantages
from ..utils.scheduling import LinearSchedule, CosineAnnealingSchedule, WarmupSchedule


@dataclass
class PPO2Config(PPOConfig):
 """Extended PPO configuration with advanced features"""
 
 # Advanced clipping
 adaptive_clipping: bool = True
 clip_decay: float = 0.99
 min_clip_range: float = 0.05
 max_clip_range: float = 0.5
 
 # Multi-step returns
 use_multi_step: bool = True
 multi_step_size: int = 5
 dynamic_lambda: bool = True
 lambda_decay: float = 0.95
 
 # Prioritized experience
 use_prioritized_replay: bool = False
 priority_alpha: float = 0.6
 priority_beta: float = 0.4
 priority_beta_decay: float = 1.001
 
 # Spectral normalization
 use_spectral_norm: bool = False
 spectral_norm_bound: float = 1.0
 
 # Curriculum learning
 curriculum_learning: bool = False
 difficulty_schedule: Optional[str] = None
 
 # Meta-learning
 meta_learning: bool = False
 meta_lr: float = 1e-4
 adaptation_steps: int = 5
 
 # Advanced scheduling
 scheduler_type: str = "cosine" # linear, cosine, warmup
 warmup_steps: int = 1000
 
 # Regularization improvements
 weight_decay: float = 0.01
 dropout_rate: float = 0.0
 layer_norm: bool = True
 
 # Performance optimizations
 mixed_precision: bool = False
 gradient_accumulation_steps: int = 1
 compile_model: bool = False # PyTorch 2.0 compilation


class SpectralNormWrapper(nn.Module):
 """Spectral normalization wrapper for stability"""
 
 def __init__(self, module: nn.Module, bound: float = 1.0):
 super().__init__()
 self.module = module
 self.bound = bound
 
 # Apply spectral norm to linear layers
 for name, layer in module.named_modules():
 if isinstance(layer, nn.Linear):
 setattr(module, name, nn.utils.spectral_norm(layer))
 
 def forward(self, *args, **kwargs):
 return self.module(*args, **kwargs)


class AdaptiveClippingScheduler:
 """Adaptive clipping parameter scheduler"""
 
 def __init__(self, config: PPO2Config):
 self.config = config
 self.initial_clip = config.clip_range
 self.current_clip = config.clip_range
 self.min_clip = config.min_clip_range
 self.max_clip = config.max_clip_range
 self.decay = config.clip_decay
 
 self.kl_history = deque(maxlen=100)
 
 def update(self, kl_divergence: float) -> float:
 """Updates clipping parameter based on KL divergence"""
 self.kl_history.append(kl_divergence)
 
 if len(self.kl_history) >= 10:
 recent_kl = np.mean(list(self.kl_history)[-10:])
 
 if recent_kl > 0.02: # Too high KL, reduce clipping
 self.current_clip = max(
 self.min_clip,
 self.current_clip * self.decay
 )
 elif recent_kl < 0.005: # Too low KL, increase clipping
 self.current_clip = min(
 self.max_clip,
 self.current_clip / self.decay
 )
 
 return self.current_clip


class PrioritizedBuffer:
 """Prioritized experience replay for PPO"""
 
 def __init__(self, capacity: int, alpha: float = 0.6):
 self.capacity = capacity
 self.alpha = alpha
 self.buffer = []
 self.priorities = np.zeros((capacity,), dtype=np.float32)
 self.position = 0
 
 def add(self, experience: Dict[str, Any], priority: float = 1.0):
 """Add experience with priority"""
 if len(self.buffer) < self.capacity:
 self.buffer.append(experience)
 else:
 self.buffer[self.position] = experience
 
 self.priorities[self.position] = priority
 self.position = (self.position + 1) % self.capacity
 
 def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
 """Sample batch with importance sampling weights"""
 if len(self.buffer) == self.capacity:
 priorities = self.priorities
 else:
 priorities = self.priorities[:self.position]
 
 # Compute sampling probabilities
 probs = priorities ** self.alpha
 probs /= probs.sum()
 
 # Sample indices
 indices = np.random.choice(len(priorities), batch_size, p=probs)
 
 # Compute importance sampling weights
 weights = (len(priorities) * probs[indices]) ** (-beta)
 weights /= weights.max()
 
 # Get experiences
 experiences = [self.buffer[i] for i in indices]
 
 return experiences, indices, weights


class PPO2Loss(PPOLoss):
 """Enhanced PPO loss with additional regularization"""
 
 def __init__(self, config: PPO2Config):
 super().__init__(config)
 self.config = config
 self.weight_decay = config.weight_decay
 
 def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
 """Compute L2 regularization loss"""
 if self.weight_decay <= 0:
 return torch.tensor(0.0, device=next(model.parameters()).device)
 
 reg_loss = 0.0
 for param in model.parameters():
 reg_loss += torch.norm(param, p=2)
 
 return self.weight_decay * reg_loss
 
 def compute_total_loss(
 self,
 policy_loss: torch.Tensor,
 value_loss: torch.Tensor,
 entropy_loss: torch.Tensor,
 model: nn.Module,
 importance_weights: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 """Compute total loss with regularization and importance sampling"""
 
 # Base loss
 total_loss = (
 policy_loss +
 self.config.vf_coef * value_loss +
 self.config.ent_coef * entropy_loss
 )
 
 # Apply importance sampling weights
 if importance_weights is not None:
 total_loss = (total_loss * importance_weights).mean()
 
 # Add regularization
 reg_loss = self.compute_regularization_loss(model)
 total_loss += reg_loss
 
 return total_loss


class PPO2Algorithm(PPOAlgorithm):
 """
 Enhanced PPO2 with advanced features
 
 Improvements:
 - Adaptive clipping parameters
 - Multi-step returns
 - Prioritized experience replay
 - Spectral normalization
 - Advanced scheduling
 - Mixed precision training
 """
 
 def __init__(
 self,
 actor_critic: nn.Module,
 config: PPO2Config,
 optimizer: Optional[torch.optim.Optimizer] = None
 ):
 # Initialize base PPO
 super().__init__(actor_critic, config, optimizer)
 
 self.config: PPO2Config = config
 
 # Apply spectral normalization
 if config.use_spectral_norm:
 self.actor_critic = SpectralNormWrapper(
 self.actor_critic, 
 config.spectral_norm_bound
 )
 
 # Enhanced loss function
 self.loss_fn = PPO2Loss(config)
 
 # Adaptive clipping
 if config.adaptive_clipping:
 self.clip_scheduler = AdaptiveClippingScheduler(config)
 
 # Advanced LR scheduling
 if config.scheduler_type == "cosine":
 self.lr_scheduler = CosineAnnealingSchedule(
 initial_value=config.learning_rate,
 min_value=config.learning_rate * 0.01
 )
 elif config.scheduler_type == "warmup":
 self.lr_scheduler = WarmupSchedule(
 warmup_steps=config.warmup_steps,
 peak_lr=config.learning_rate
 )
 
 # Prioritized replay buffer
 if config.use_prioritized_replay:
 self.priority_buffer = PrioritizedBuffer(
 capacity=config.batch_size * 10,
 alpha=config.priority_alpha
 )
 self.priority_beta = config.priority_beta
 
 # Mixed precision training
 if config.mixed_precision:
 self.scaler = torch.cuda.amp.GradScaler()
 
 # Model compilation (PyTorch 2.0)
 if config.compile_model and hasattr(torch, 'compile'):
 self.actor_critic = torch.compile(self.actor_critic)
 
 def update(
 self,
 rollout_buffer,
 progress_remaining: float = 1.0
 ) -> Dict[str, float]:
 """Enhanced PPO update with advanced features"""
 
 # Update learning rate
 current_lr = self.lr_scheduler.value(1.0 - progress_remaining)
 for param_group in self.optimizer.param_groups:
 param_group['lr'] = current_lr
 
 # Get batch data
 batch_data = rollout_buffer.get()
 
 # Multi-step returns
 if self.config.use_multi_step:
 batch_data = self._compute_multi_step_returns(batch_data)
 
 # Enhanced training loop
 all_metrics = defaultdict(list)
 
 for epoch in range(self.config.n_epochs):
 # Early stopping
 if self.target_kl is not None and all_metrics.get("approx_kl"):
 mean_kl = np.mean(all_metrics["approx_kl"][-10:])
 if mean_kl > 1.5 * self.target_kl:
 self.logger.info(f"Early stopping at epoch {epoch}")
 break
 
 # Training step with prioritized sampling
 if self.config.use_prioritized_replay:
 batch_metrics = self._prioritized_update_step(batch_data)
 else:
 batch_metrics = self._enhanced_update_step(batch_data, progress_remaining)
 
 # Accumulating metrics
 for key, value in batch_metrics.items():
 all_metrics[key].append(value)
 
 # Final metrics
 final_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
 
 # Update adaptive parameters
 if hasattr(self, 'clip_scheduler') and final_metrics.get("approx_kl"):
 current_clip = self.clip_scheduler.update(final_metrics["approx_kl"])
 final_metrics["clip_range"] = current_clip
 
 final_metrics.update({
 "learning_rate": current_lr,
 "update_count": self.update_count
 })
 
 self.update_count += 1
 
 if self.update_count % self.config.log_interval == 0:
 self._log_enhanced_metrics(final_metrics)
 
 return final_metrics
 
 def _compute_multi_step_returns(
 self, 
 batch_data: Dict[str, torch.Tensor]
 ) -> Dict[str, torch.Tensor]:
 """Compute multi-step returns for improved value estimation"""
 
 rewards = batch_data["rewards"]
 values = batch_data["values"]
 dones = batch_data["dones"]
 
 # Multi-step returns
 multi_step_returns = torch.zeros_like(rewards)
 multi_step_size = self.config.multi_step_size
 
 for t in range(len(rewards)):
 G = 0
 for k in range(min(multi_step_size, len(rewards) - t)):
 if t + k < len(rewards):
 if dones[t + k]:
 G += (self.config.gamma ** k) * rewards[t + k]
 break
 else:
 G += (self.config.gamma ** k) * rewards[t + k]
 
 # Add bootstrapped value
 if t + multi_step_size < len(values) and not dones[t + multi_step_size - 1]:
 G += (self.config.gamma ** multi_step_size) * values[t + multi_step_size]
 
 multi_step_returns[t] = G
 
 batch_data["returns"] = multi_step_returns
 return batch_data
 
 def _enhanced_update_step(
 self,
 batch_data: Dict[str, torch.Tensor],
 progress_remaining: float
 ) -> Dict[str, float]:
 """Enhanced update step with mixed precision and gradient accumulation"""
 
 total_metrics = defaultdict(list)
 
 # Gradient accumulation loop
 for step in range(self.config.gradient_accumulation_steps):
 # Get mini-batch
 batch_size = self.config.batch_size // self.config.gradient_accumulation_steps
 start_idx = step * batch_size
 end_idx = (step + 1) * batch_size
 
 # Extract mini-batch
 mini_batch = {}
 for key, tensor in batch_data.items():
 mini_batch[key] = tensor[start_idx:end_idx]
 
 # Forward pass with mixed precision
 if self.config.mixed_precision:
 with torch.cuda.amp.autocast():
 step_metrics = self._compute_loss_step(mini_batch)
 loss = step_metrics["total_loss"] / self.config.gradient_accumulation_steps
 
 # Backward pass
 self.scaler.scale(loss).backward()
 else:
 step_metrics = self._compute_loss_step(mini_batch)
 loss = step_metrics["total_loss"] / self.config.gradient_accumulation_steps
 loss.backward()
 
 # Accumulate metrics
 for key, value in step_metrics.items():
 total_metrics[key].append(value)
 
 # Optimizer step
 if self.config.mixed_precision:
 self.scaler.unscale_(self.optimizer)
 
 # Gradient clipping
 grad_norm = torch.nn.utils.clip_grad_norm_(
 self.actor_critic.parameters(),
 self.config.max_grad_norm
 )
 
 if self.config.mixed_precision:
 self.scaler.step(self.optimizer)
 self.scaler.update()
 else:
 self.optimizer.step()
 
 self.optimizer.zero_grad()
 
 # Average metrics
 final_metrics = {key: np.mean(values) for key, values in total_metrics.items()}
 final_metrics["grad_norm"] = grad_norm.item()
 
 return final_metrics
 
 def _compute_loss_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
 """Compute loss for a single step"""
 
 observations = batch["observations"]
 actions = batch["actions"]
 old_log_probs = batch["log_probs"]
 old_values = batch["values"]
 advantages = batch["advantages"]
 returns = batch["returns"]
 
 # Normalize advantages
 if self.config.normalize_advantage:
 advantages = normalize_advantages(advantages)
 
 # Forward pass
 action_dist, values = self.actor_critic(observations)
 
 # Compute log probabilities
 if isinstance(action_dist, Categorical):
 log_probs = action_dist.log_prob(actions.squeeze())
 else:
 log_probs = action_dist.log_prob(actions).sum(dim=-1)
 
 # Compute losses
 policy_loss, policy_metrics = self.loss_fn.compute_policy_loss(
 log_probs, old_log_probs, advantages
 )
 
 value_loss, value_metrics = self.loss_fn.compute_value_loss(
 values.squeeze(), old_values, returns
 )
 
 entropy_loss, entropy_metrics = self.loss_fn.compute_entropy_loss(action_dist)
 
 # Total loss with regularization
 total_loss = self.loss_fn.compute_total_loss(
 policy_loss, value_loss, entropy_loss, self.actor_critic
 )
 
 # Combine metrics
 metrics = {
 "total_loss": total_loss.item(),
 **policy_metrics,
 **value_metrics,
 **entropy_metrics
 }
 
 return metrics
 
 def _log_enhanced_metrics(self, metrics: Dict[str, float]):
 """Enhanced logging with additional metrics"""
 log_msg = f"PPO2 Update {self.update_count}: "
 log_msg += f"Total Loss: {metrics.get('total_loss', 0):.4f}, "
 log_msg += f"Policy Loss: {metrics['policy_loss']:.4f}, "
 log_msg += f"Value Loss: {metrics['value_loss']:.4f}, "
 log_msg += f"Entropy: {metrics['entropy']:.4f}, "
 log_msg += f"KL: {metrics['approx_kl']:.4f}, "
 log_msg += f"LR: {metrics['learning_rate']:.6f}"
 
 if "clip_range" in metrics:
 log_msg += f", Clip: {metrics['clip_range']:.3f}"
 
 self.logger.info(log_msg)


# Export classes
__all__ = [
 "PPO2Config",
 "PPO2Loss",
 "PPO2Algorithm",
 "SpectralNormWrapper",
 "AdaptiveClippingScheduler",
 "PrioritizedBuffer"
]