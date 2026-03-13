"""
TD(λ) Implementation for PPO
for temporal difference learning

TD(λ) unifies TD(0) and Monte Carlo methods:
- λ = 0: TD(0) - pure bootstrapping
- λ = 1: Monte Carlo - complete rollouts
- 0 < λ < 1: λ-return combining both approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

from ..utils.normalization import normalize_advantages


@dataclass
class TDLambdaConfig:
 """Configuration for TD(λ) methods"""
 
 # TD(λ) parameters
 gamma: float = 0.99 # Discount factor
 lambda_param: float = 0.95 # λ parameter
 
 # Eligibility traces
 use_eligibility_traces: bool = True
 trace_decay: float = 0.9
 
 # Return computation
 n_step_max: int = 10 # Maximum n-step for λ-return
 truncated_lambda: bool = False # Use truncated λ-return
 
 # Normalization
 normalize_returns: bool = True
 normalize_advantages: bool = True
 
 # Numerical stability
 eps: float = 1e-8
 
 # Advanced features
 adaptive_lambda: bool = False
 lambda_schedule: Optional[str] = None # 'linear', 'cosine', None


class TDLambdaEstimator:
 """
 TD(λ) estimator for advantage and return computation
 
 Implements multiple variants:
 - Forward-view TD(λ)
 - Backward-view TD(λ) with eligibility traces
 - Truncated TD(λ)
 - True online TD(λ)
 """
 
 def __init__(self, config: TDLambdaConfig):
 self.config = config
 self.eligibility_traces = {} # Store traces for different states
 
 def compute_lambda_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Compute λ-returns and advantages
 
 λ-return: G_t^λ = (1-λ)∑_{n=1}^∞ λ^{n-1} G_t^{(n)}
 where G_t^{(n)} is n-step return
 """
 
 if self.config.truncated_lambda:
 return self._compute_truncated_lambda_returns(
 rewards, values, dones, next_values
 )
 else:
 return self._compute_full_lambda_returns(
 rewards, values, dones, next_values
 )
 
 def _compute_full_lambda_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute full λ-returns (infinite horizon)"""
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 lambda_returns = torch.zeros_like(rewards)
 advantages = torch.zeros_like(rewards)
 
 # Current lambda value
 current_lambda = self._get_current_lambda()
 
 # Backward computation
 last_lambda_return = torch.zeros(batch_size, device=device)
 if next_values is not None:
 last_lambda_return = next_values
 
 for t in reversed(range(seq_len)):
 if t == seq_len - 1:
 # Terminal step
 if next_values is not None:
 bootstrap_value = next_values
 else:
 bootstrap_value = torch.zeros(batch_size, device=device)
 
 lambda_return = rewards[t] + self.config.gamma * (1 - dones[t]) * bootstrap_value
 else:
 # Recursive λ-return computation
 # G_t^λ = r_t + γ[(1-λ)V_{t+1} + λG_{t+1}^λ]
 next_value = values[t + 1]
 lambda_return = rewards[t] + self.config.gamma * (1 - dones[t]) * (
 (1 - current_lambda) * next_value + 
 current_lambda * last_lambda_return
 )
 
 lambda_returns[t] = lambda_return
 advantages[t] = lambda_return - values[t]
 last_lambda_return = lambda_return
 
 # Normalization
 if self.config.normalize_returns:
 lambda_returns = self._normalize_tensor(lambda_returns)
 
 if self.config.normalize_advantages:
 advantages = normalize_advantages(advantages)
 
 return lambda_returns, advantages
 
 def _compute_truncated_lambda_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute truncated λ-returns (finite horizon)"""
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 lambda_returns = torch.zeros_like(rewards)
 advantages = torch.zeros_like(rewards)
 
 current_lambda = self._get_current_lambda()
 
 # Compute for each timestep
 for t in range(seq_len):
 lambda_return = 0.0
 lambda_weight = 1.0
 cumulative_weight = 0.0
 
 # Compute n-step returns up to max horizon
 for n in range(1, min(self.config.n_step_max + 1, seq_len - t + 1)):
 # n-step return
 n_step_return = 0.0
 discount = 1.0
 
 # Sum discounted rewards
 for k in range(n):
 if t + k < seq_len:
 n_step_return += discount * rewards[t + k]
 discount *= self.config.gamma
 
 # Early termination if episode ends
 if dones[t + k]:
 break
 
 # Add bootstrap value
 if t + n < seq_len and not dones[t + n - 1]:
 n_step_return += discount * values[t + n]
 elif t + n >= seq_len and next_values is not None and not dones[-1]:
 n_step_return += discount * next_values
 
 # Weight by (1-λ)λ^{n-1}
 if n == self.config.n_step_max:
 # Last term gets remaining weight
 weight = lambda_weight
 else:
 weight = (1 - current_lambda) * lambda_weight
 
 lambda_return += weight * n_step_return
 cumulative_weight += weight
 lambda_weight *= current_lambda
 
 # Normalize by total weight
 if cumulative_weight > 0:
 lambda_return /= cumulative_weight
 
 lambda_returns[t] = lambda_return
 advantages[t] = lambda_return - values[t]
 
 # Normalization
 if self.config.normalize_returns:
 lambda_returns = self._normalize_tensor(lambda_returns)
 
 if self.config.normalize_advantages:
 advantages = normalize_advantages(advantages)
 
 return lambda_returns, advantages
 
 def _get_current_lambda(self) -> float:
 """Get current lambda value (with optional scheduling)"""
 
 if self.config.lambda_schedule is None:
 return self.config.lambda_param
 
 # Implement scheduling logic
 # This would need training step tracking
 return self.config.lambda_param
 
 def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
 """Normalize tensor for stability"""
 mean_val = tensor.mean()
 std_val = tensor.std()
 return (tensor - mean_val) / (std_val + self.config.eps)


class EligibilityTracesTD:
 """
 TD(λ) with eligibility traces (backward view)
 
 Maintains eligibility traces for efficient credit assignment
 More computationally efficient than forward view
 """
 
 def __init__(self, config: TDLambdaConfig, state_dim: int):
 self.config = config
 self.state_dim = state_dim
 
 # Initialize eligibility traces
 self.eligibility_traces = None
 self.last_values = None
 
 def update_traces_and_values(
 self,
 states: torch.Tensor,
 rewards: torch.Tensor,
 values: torch.Tensor,
 next_values: torch.Tensor,
 dones: torch.Tensor
 ) -> Dict[str, torch.Tensor]:
 """
 Update eligibility traces and compute TD updates
 
 Args:
 states: Current states [batch_size, state_dim]
 rewards: Rewards [batch_size]
 values: Current value estimates [batch_size]
 next_values: Next value estimates [batch_size]
 dones: Done flags [batch_size]
 
 Returns:
 Dictionary with updates and traces
 """
 
 batch_size = states.shape[0]
 device = states.device
 
 # Initialize traces if first call
 if self.eligibility_traces is None:
 self.eligibility_traces = torch.zeros(
 batch_size, self.state_dim, device=device
 )
 
 # Compute TD error
 td_error = rewards + self.config.gamma * (1 - dones) * next_values - values
 
 # Update eligibility traces
 # e_t = γλe_{t-1} + ∇V(s_t)
 # For simplicity, assume ∇V(s_t) ≈ states (linear approximation)
 self.eligibility_traces = (
 self.config.gamma * self.config.lambda_param * self.eligibility_traces +
 states
 )
 
 # Reset traces where episodes ended
 self.eligibility_traces = self.eligibility_traces * (1 - dones.unsqueeze(1))
 
 # Compute value updates
 # ΔV = α * td_error * e_t
 value_updates = td_error.unsqueeze(1) * self.eligibility_traces
 
 return {
 "td_error": td_error,
 "eligibility_traces": self.eligibility_traces.clone(),
 "value_updates": value_updates
 }
 
 def reset_traces(self, done_mask: torch.Tensor):
 """Reset eligibility traces for terminated episodes"""
 if self.eligibility_traces is not None:
 self.eligibility_traces = self.eligibility_traces * (1 - done_mask.unsqueeze(1))


class TrueOnlineTD:
 """
 True Online TD(λ) implementation
 
 More accurate than conventional TD(λ) by maintaining
 exact eligibility traces without approximations
 """
 
 def __init__(self, config: TDLambdaConfig):
 self.config = config
 self.old_value = None
 
 def compute_true_online_updates(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 next_values: torch.Tensor,
 dones: torch.Tensor,
 eligibility_traces: torch.Tensor
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Compute true online TD(λ) updates
 
 Accounts for changing value function approximation
 during the eligibility trace accumulation
 """
 
 # TD error
 td_error = rewards + self.config.gamma * (1 - dones) * next_values - values
 
 if self.old_value is not None:
 # True online correction term
 correction = (values - self.old_value)
 td_error = td_error + correction
 
 # Update eligibility traces with true online method
 # e_t = γλe_{t-1} + ∇V(s_t) - αγλ(∇V(s_t)^T e_{t-1})∇V(s_t)
 # Simplified version without full gradient computation
 gamma_lambda = self.config.gamma * self.config.lambda_param
 updated_traces = gamma_lambda * eligibility_traces
 
 # Store current values for next update
 self.old_value = values.clone()
 
 return td_error, updated_traces


class MultiStepTD:
 """
 Multi-step TD learning with different step sizes
 
 Combines TD estimates from different horizons
 for more robust value estimation
 """
 
 def __init__(
 self,
 config: TDLambdaConfig,
 step_sizes: List[int] = None
 ):
 self.config = config
 self.step_sizes = step_sizes or [1, 3, 5, 10]
 
 def compute_multi_step_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Dict[str, torch.Tensor]:
 """
 Compute returns for multiple step sizes
 
 Returns dictionary with returns for each step size
 """
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 multi_step_returns = {}
 
 for n_steps in self.step_sizes:
 returns = torch.zeros_like(rewards)
 
 for t in range(seq_len):
 # Compute n-step return
 n_step_return = 0.0
 discount = 1.0
 
 for k in range(min(n_steps, seq_len - t)):
 n_step_return += discount * rewards[t + k]
 discount *= self.config.gamma
 
 if dones[t + k]:
 break
 
 # Bootstrap with value function
 if t + n_steps < seq_len and not dones[t + n_steps - 1]:
 n_step_return += discount * values[t + n_steps]
 elif t + n_steps >= seq_len and next_values is not None and not dones[-1]:
 n_step_return += discount * next_values
 
 returns[t] = n_step_return
 
 multi_step_returns[f"{n_steps}_step"] = returns
 
 return multi_step_returns
 
 def compute_weighted_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None,
 weights: Optional[List[float]] = None
 ) -> torch.Tensor:
 """Compute weighted combination of multi-step returns"""
 
 multi_returns = self.compute_multi_step_returns(
 rewards, values, dones, next_values
 )
 
 if weights is None:
 # Equal weighting
 weights = [1.0 / len(self.step_sizes)] * len(self.step_sizes)
 
 # Weighted combination
 weighted_returns = torch.zeros_like(rewards)
 for i, (step_size, weight) in enumerate(zip(self.step_sizes, weights)):
 weighted_returns += weight * multi_returns[f"{step_size}_step"]
 
 return weighted_returns


# Factory function for creating TD(λ) estimators
def create_td_lambda_estimator(
 estimator_type: str = "standard",
 config: Optional[TDLambdaConfig] = None,
 **kwargs
) -> Union[TDLambdaEstimator, EligibilityTracesTD, TrueOnlineTD, MultiStepTD]:
 """Create TD(λ) estimator of specified type"""
 
 if config is None:
 config = TDLambdaConfig()
 
 if estimator_type == "standard":
 return TDLambdaEstimator(config)
 elif estimator_type == "eligibility_traces":
 return EligibilityTracesTD(config, **kwargs)
 elif estimator_type == "true_online":
 return TrueOnlineTD(config)
 elif estimator_type == "multi_step":
 return MultiStepTD(config, **kwargs)
 else:
 raise ValueError(f"Unknown estimator type: {estimator_type}")


# Export classes and functions
__all__ = [
 "TDLambdaConfig",
 "TDLambdaEstimator",
 "EligibilityTracesTD",
 "TrueOnlineTD",
 "MultiStepTD",
 "create_td_lambda_estimator"
]