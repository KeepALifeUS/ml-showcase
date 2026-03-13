"""
Generalized Advantage Estimation (GAE) Implementation
for stable advantage estimation

GAE combines bias and variance tradeoffs:
- λ = 0: high bias, low variance (TD error)
- λ = 1: low bias, high variance (Monte Carlo)
- 0 < λ < 1: balanced bias-variance tradeoff
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
class GAEConfig:
 """Configuration for GAE computation"""
 
 # GAE parameters
 gamma: float = 0.99 # Discount factor
 gae_lambda: float = 0.95 # GAE lambda parameter
 
 # Normalization
 normalize_advantages: bool = True
 normalize_returns: bool = False
 
 # Clipping
 advantage_clipping: Optional[float] = None # Clip advantages
 return_clipping: Optional[float] = None # Clip returns
 
 # Numerical stability
 eps: float = 1e-8
 
 # Advanced features
 adaptive_lambda: bool = False # Adapt lambda based on TD error
 lambda_decay: float = 0.99 # Decay rate for adaptive lambda
 use_reward_prediction: bool = False # Use reward prediction error


class AdvantageEstimator(ABC):
 """Base class for advantage estimation methods"""
 
 def __init__(self, config: GAEConfig):
 self.config = config
 
 @abstractmethod
 def compute_advantages_and_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Compute advantages and returns
 
 Args:
 rewards: Rewards [seq_len, batch_size]
 values: Value estimates [seq_len, batch_size] 
 dones: Done flags [seq_len, batch_size]
 next_values: Next step values [batch_size] (optional)
 
 Returns:
 advantages, returns
 """
 pass


class GAE(AdvantageEstimator):
 """
 Generalized Advantage Estimation implementation
 
 Computes λ-return advantages using exponentially weighted
 combination of n-step advantages
 """
 
 def __init__(self, config: GAEConfig):
 super().__init__(config)
 self.lambda_history = []
 
 def compute_advantages_and_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor, 
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Compute GAE advantages and λ-returns
 
 GAE formula:
 Â_t = δ_t + (γλ)(1-done_t+1)δ_t+1 + ... 
 
 where δ_t = r_t + γ(1-done_t+1)V(s_t+1) - V(s_t)
 """
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 # Initialize advantages and returns
 advantages = torch.zeros_like(rewards)
 returns = torch.zeros_like(rewards)
 
 # Get current lambda
 current_lambda = self._get_current_lambda()
 
 # Compute TD errors (deltas)
 deltas = self._compute_td_errors(rewards, values, dones, next_values)
 
 # Backward computation of GAE
 gae = torch.zeros(batch_size, device=device)
 
 for t in reversed(range(seq_len)):
 # GAE recursive formula
 if t == seq_len - 1:
 # Last step - use next_values if available
 if next_values is not None:
 delta = rewards[t] + self.config.gamma * (1 - dones[t]) * next_values - values[t]
 else:
 delta = deltas[t]
 gae = delta
 else:
 delta = deltas[t]
 gae = delta + self.config.gamma * current_lambda * (1 - dones[t]) * gae
 
 advantages[t] = gae
 returns[t] = advantages[t] + values[t]
 
 # Update lambda if adaptive
 if self.config.adaptive_lambda:
 self._update_adaptive_lambda(deltas)
 
 # Apply normalization
 if self.config.normalize_advantages:
 advantages = normalize_advantages(advantages)
 
 if self.config.normalize_returns:
 returns = self._normalize_returns(returns)
 
 # Apply clipping
 if self.config.advantage_clipping is not None:
 advantages = torch.clamp(
 advantages,
 -self.config.advantage_clipping,
 self.config.advantage_clipping
 )
 
 if self.config.return_clipping is not None:
 returns = torch.clamp(
 returns,
 -self.config.return_clipping,
 self.config.return_clipping
 )
 
 return advantages, returns
 
 def _compute_td_errors(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor, 
 next_values: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 """Compute TD errors (temporal difference errors)"""
 
 seq_len, batch_size = rewards.shape
 deltas = torch.zeros_like(rewards)
 
 for t in range(seq_len):
 if t == seq_len - 1:
 # Last step
 if next_values is not None:
 next_val = next_values
 else:
 next_val = torch.zeros(batch_size, device=rewards.device)
 else:
 next_val = values[t + 1]
 
 # TD error: r_t + γ(1-done)V(s_t+1) - V(s_t)
 deltas[t] = (
 rewards[t] + 
 self.config.gamma * (1 - dones[t]) * next_val - 
 values[t]
 )
 
 return deltas
 
 def _get_current_lambda(self) -> float:
 """Get current lambda value (adaptive or fixed)"""
 if self.config.adaptive_lambda:
 if len(self.lambda_history) == 0:
 return self.config.gae_lambda
 else:
 # Adaptive lambda based on recent performance
 return self.lambda_history[-1]
 else:
 return self.config.gae_lambda
 
 def _update_adaptive_lambda(self, td_errors: torch.Tensor):
 """Update lambda based on TD error magnitude"""
 
 # Compute mean absolute TD error
 mean_td_error = torch.abs(td_errors).mean().item()
 
 # If TD errors are high, increase lambda (more Monte Carlo)
 # If TD errors are low, decrease lambda (more TD learning)
 if mean_td_error > 1.0: # High TD error
 new_lambda = min(0.99, self.config.gae_lambda * 1.01)
 else: # Low TD error
 new_lambda = max(0.80, self.config.gae_lambda * 0.999)
 
 self.lambda_history.append(new_lambda)
 
 # Keep only recent history
 if len(self.lambda_history) > 100:
 self.lambda_history = self.lambda_history[-100:]
 
 def _normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
 """Normalize returns"""
 mean_return = returns.mean()
 std_return = returns.std()
 
 return (returns - mean_return) / (std_return + self.config.eps)


class MultiStepAdvantages(AdvantageEstimator):
 """
 Multi-step advantage estimation
 
 Combines advantages from different step sizes for robust estimation
 """
 
 def __init__(self, config: GAEConfig, step_sizes: List[int] = None):
 super().__init__(config)
 self.step_sizes = step_sizes or [1, 3, 5, 10]
 
 def compute_advantages_and_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute multi-step advantages"""
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 # Compute advantages for different step sizes
 all_advantages = []
 all_returns = []
 
 for n_steps in self.step_sizes:
 advantages, returns = self._compute_n_step_advantages(
 rewards, values, dones, n_steps, next_values
 )
 all_advantages.append(advantages)
 all_returns.append(returns)
 
 # Weighted combination of advantages
 weights = torch.softmax(
 torch.tensor([1.0 / n for n in self.step_sizes], device=device),
 dim=0
 )
 
 final_advantages = sum(
 w * adv for w, adv in zip(weights, all_advantages)
 )
 final_returns = sum(
 w * ret for w, ret in zip(weights, all_returns)
 )
 
 # Normalization
 if self.config.normalize_advantages:
 final_advantages = normalize_advantages(final_advantages)
 
 return final_advantages, final_returns
 
 def _compute_n_step_advantages(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 n_steps: int,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute n-step advantages"""
 
 seq_len, batch_size = rewards.shape
 advantages = torch.zeros_like(rewards)
 returns = torch.zeros_like(rewards)
 
 for t in range(seq_len):
 # Compute n-step return
 n_step_return = 0.0
 discount = 1.0
 
 for k in range(min(n_steps, seq_len - t)):
 n_step_return += discount * rewards[t + k]
 discount *= self.config.gamma
 
 # Early termination if episode ends
 if dones[t + k]:
 break
 
 # Add bootstrapped value
 if t + n_steps < seq_len and not dones[t + n_steps - 1]:
 n_step_return += discount * values[t + n_steps]
 elif t + n_steps >= seq_len and next_values is not None and not dones[-1]:
 n_step_return += discount * next_values
 
 advantages[t] = n_step_return - values[t]
 returns[t] = n_step_return
 
 return advantages, returns


class RiskAdjustedGAE(GAE):
 """
 Risk-adjusted GAE for crypto trading
 
 Adjusts advantages based on:
 - Market volatility
 - Portfolio risk
 - Drawdown potential
 """
 
 def __init__(
 self,
 config: GAEConfig,
 risk_adjustment_factor: float = 0.1,
 volatility_window: int = 20
 ):
 super().__init__(config)
 self.risk_adjustment = risk_adjustment_factor
 self.volatility_window = volatility_window
 self.price_history = []
 
 def compute_advantages_and_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None,
 prices: Optional[torch.Tensor] = None,
 positions: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute risk-adjusted GAE advantages"""
 
 # Standard GAE computation
 advantages, returns = super().compute_advantages_and_returns(
 rewards, values, dones, next_values
 )
 
 # Risk adjustment
 if prices is not None:
 risk_factors = self._compute_risk_factors(prices, positions)
 advantages = advantages * (1 - self.risk_adjustment * risk_factors)
 
 return advantages, returns
 
 def _compute_risk_factors(
 self,
 prices: torch.Tensor,
 positions: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 """Compute risk adjustment factors"""
 
 seq_len, batch_size = prices.shape
 risk_factors = torch.ones_like(prices)
 
 # Compute rolling volatility
 for t in range(self.volatility_window, seq_len):
 price_window = prices[t-self.volatility_window:t]
 returns_window = torch.diff(torch.log(price_window + 1e-8), dim=0)
 volatility = returns_window.std(dim=0)
 
 # Higher volatility = higher risk factor
 risk_factors[t] = torch.clamp(volatility, 0.0, 1.0)
 
 # Adjust for position size if available
 if positions is not None:
 position_risk = torch.abs(positions)
 risk_factors = risk_factors * (1 + position_risk)
 
 return risk_factors


class AdaptiveGAE(GAE):
 """
 Adaptive GAE with dynamic parameter adjustment
 
 Automatically adjusts λ and γ based on:
 - Learning progress
 - Environment characteristics
 - Reward sparsity
 """
 
 def __init__(
 self,
 config: GAEConfig,
 adaptation_rate: float = 0.01,
 min_lambda: float = 0.8,
 max_lambda: float = 0.99
 ):
 super().__init__(config)
 self.adaptation_rate = adaptation_rate
 self.min_lambda = min_lambda
 self.max_lambda = max_lambda
 
 # Tracking variables
 self.td_error_history = []
 self.advantage_variance_history = []
 self.current_lambda = config.gae_lambda
 
 def compute_advantages_and_returns(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor] = None
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute adaptive GAE advantages"""
 
 # Compute TD errors for adaptation
 td_errors = self._compute_td_errors(rewards, values, dones, next_values)
 
 # Update lambda based on TD error characteristics
 self._adapt_lambda(td_errors)
 
 # Compute advantages with adapted lambda
 advantages, returns = self._compute_gae_with_lambda(
 rewards, values, dones, next_values, self.current_lambda
 )
 
 return advantages, returns
 
 def _adapt_lambda(self, td_errors: torch.Tensor):
 """Adapt lambda based on TD error characteristics"""
 
 # Compute TD error statistics
 mean_abs_td_error = torch.abs(td_errors).mean().item()
 td_error_std = td_errors.std().item()
 
 self.td_error_history.append(mean_abs_td_error)
 
 if len(self.td_error_history) < 10:
 return
 
 # Compute trend in TD errors
 recent_errors = self.td_error_history[-10:]
 error_trend = (recent_errors[-1] - recent_errors[0]) / 9
 
 # Adaptation logic:
 # If TD errors increasing → increase lambda (more Monte Carlo)
 # If TD errors decreasing → decrease lambda (more bootstrapping)
 if error_trend > 0.01: # Errors increasing
 lambda_adjustment = self.adaptation_rate
 elif error_trend < -0.01: # Errors decreasing 
 lambda_adjustment = -self.adaptation_rate
 else:
 lambda_adjustment = 0.0
 
 # Update lambda
 self.current_lambda += lambda_adjustment
 self.current_lambda = np.clip(
 self.current_lambda,
 self.min_lambda,
 self.max_lambda
 )
 
 def _compute_gae_with_lambda(
 self,
 rewards: torch.Tensor,
 values: torch.Tensor,
 dones: torch.Tensor,
 next_values: Optional[torch.Tensor],
 lambda_param: float
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """Compute GAE with specific lambda parameter"""
 
 seq_len, batch_size = rewards.shape
 device = rewards.device
 
 advantages = torch.zeros_like(rewards)
 returns = torch.zeros_like(rewards)
 
 deltas = self._compute_td_errors(rewards, values, dones, next_values)
 
 gae = torch.zeros(batch_size, device=device)
 
 for t in reversed(range(seq_len)):
 delta = deltas[t]
 gae = delta + self.config.gamma * lambda_param * (1 - dones[t]) * gae
 
 advantages[t] = gae
 returns[t] = advantages[t] + values[t]
 
 return advantages, returns


# Factory function for creating advantage estimators
def create_advantage_estimator(
 estimator_type: str = "gae",
 config: Optional[GAEConfig] = None,
 **kwargs
) -> AdvantageEstimator:
 """Create advantage estimator of specified type"""
 
 if config is None:
 config = GAEConfig()
 
 if estimator_type == "gae":
 return GAE(config)
 elif estimator_type == "multi_step":
 return MultiStepAdvantages(config, **kwargs)
 elif estimator_type == "risk_adjusted":
 return RiskAdjustedGAE(config, **kwargs)
 elif estimator_type == "adaptive":
 return AdaptiveGAE(config, **kwargs)
 else:
 raise ValueError(f"Unknown estimator type: {estimator_type}")


# Export classes and functions
__all__ = [
 "GAEConfig",
 "AdvantageEstimator",
 "GAE",
 "MultiStepAdvantages",
 "RiskAdjustedGAE", 
 "AdaptiveGAE",
 "create_advantage_estimator"
]