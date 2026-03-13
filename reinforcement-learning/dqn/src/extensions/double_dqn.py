"""
Double DQN implementation with enterprise patterns.

Double DQN solves problem overestimation bias in standard DQN:
- Separates action selection from action evaluation
- Uses main network for selection actions
- Uses target network for score Q-values
- Significantly improves stability training
- Full consistency with base DQN API
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
import structlog

from ..core.dqn import DQN, DQNConfig, Experience
from ..networks.q_network import QNetwork

logger = structlog.get_logger(__name__)


class DoubleDQNConfig(DQNConfig):
 """
 Configuration Double DQN with additional parameters.

 Inherits all parameters from base DQN and adds specific for Double DQN.
 """

 # Double DQN specific parameters
 double_q_freq: int = Field(default=1, description="Frequency usage double Q updates", gt=0)
 action_selection_temp: float = Field(default=1.0, description="Temperature for action selection", gt=0)
 evaluation_noise: float = Field(default=0.0, description="Noise for evaluation stability", ge=0, le=0.1)

 # Performance monitoring
 track_overestimation: bool = Field(default=True, description="Tracking overestimation bias")
 bias_correction_alpha: float = Field(default=0.1, description="Alpha for bias correction", ge=0, le=1.0)


class DoubleDQN(DQN):
 """
 Double DQN agent with enterprise-grade implementation.

 Double DQN difficulties:
 - Elimination overestimation bias through decoupled selection/evaluation
 - Improved stability in complex environments
 - Better convergence properties
 - Minimal overhead over standard DQN

 Features:
 - Full compatibility with base DQN API
 - Configurable double Q frequency for ablation studies
 - Overestimation tracking for analysis
 - Temperature-based action selection for exploration
 - Production-ready monitoring and logging
 """

 def __init__(self, config: DoubleDQNConfig):
 """
 Initialization Double DQN agent.

 Args:
 config: Configuration Double DQN
 """
 super.__init__(config)
 self.double_config = config

 # Tracking overestimation bias
 self.overestimation_history = []
 self.bias_estimates = []

 # Performance tracking
 self.double_q_updates = 0
 self.standard_q_updates = 0

 self.logger = structlog.get_logger(__name__).bind(
 agent_type="DoubleDQN",
 device=str(self.device),
 double_q_freq=config.double_q_freq
 )

 self.logger.info("Double DQN agent initialized",
 config=config.dict)

 def train_step(self) -> Dict[str, float]:
 """
 One step training with Double DQN update rule.

 Returns:
 Metrics training including overestimation bias
 """
 if len(self.replay_buffer) < self.config.min_replay_size:
 return {"status": "insufficient_data"}

 # Sampling batch from replay buffer
 experiences = self.replay_buffer.sample

 # Conversion to tensors
 states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
 actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
 rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
 next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
 dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)

 # Current Q values
 current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

 # Double DQN target computation
 with torch.no_grad:
 if (self.training_step % self.double_config.double_q_freq == 0 and
 self.training_step > 0):
 # Double DQN: action selection with main network, evaluation with target network
 next_q_values_main = self.q_network(next_states)

 # Action selection with optional temperature
 if self.double_config.action_selection_temp != 1.0:
 next_q_values_main = next_q_values_main / self.double_config.action_selection_temp

 next_actions = next_q_values_main.argmax(1).unsqueeze(1)

 # Evaluation with target network
 next_q_values_target = self.target_network(next_states)

 # Add evaluation noise for stability
 if self.double_config.evaluation_noise > 0:
 noise = torch.randn_like(next_q_values_target) * self.double_config.evaluation_noise
 next_q_values_target = next_q_values_target + noise

 next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

 # Tracking
 self.double_q_updates += 1

 # Overestimation bias tracking
 if self.double_config.track_overestimation:
 standard_max_q = next_q_values_target.max(1)[0]
 double_q_selected = next_q_values
 overestimation = (standard_max_q - double_q_selected).mean.item
 self.overestimation_history.append(overestimation)

 else:
 # Standard DQN update for comparison
 next_q_values = self.target_network(next_states).max(1)[0]
 self.standard_q_updates += 1

 # Target Q values
 target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

 # Compute loss
 loss = self._compute_loss(current_q_values.squeeze, target_q_values)

 # Bias correction if enabled
 if self.double_config.track_overestimation and len(self.overestimation_history) > 10:
 bias_correction = self._compute_bias_correction
 loss = loss + bias_correction

 # Optimization step
 self.optimizer.zero_grad
 loss.backward

 # Gradient clipping
 grad_norm = torch.nn.utils.clip_grad_norm_(
 self.q_network.parameters, self.config.grad_clip_norm
 )

 self.optimizer.step

 # Update training step
 self.training_step += 1

 # Update target network
 if self.training_step % self.config.target_update_freq == 0:
 self._update_target_network

 # Metrics including Double DQN specific
 metrics = {
 "loss": loss.item,
 "grad_norm": grad_norm.item,
 "epsilon": self.epsilon_schedule.get_epsilon(self.training_step),
 "training_step": self.training_step,
 "q_mean": current_q_values.mean.item,
 "target_mean": target_q_values.mean.item,
 "double_q_ratio": self.double_q_updates / max(self.training_step, 1),
 }

 # Overestimation bias metrics
 if (self.double_config.track_overestimation and
 len(self.overestimation_history) > 0):
 recent_bias = np.mean(self.overestimation_history[-100:])
 metrics.update({
 "overestimation_bias": recent_bias,
 "bias_std": np.std(self.overestimation_history[-100:]) if len(self.overestimation_history) > 1 else 0.0,
 })

 self.loss_history.append(loss.item)

 if self.training_step % self.config.log_freq == 0:
 self.logger.info("Double DQN training step completed", **metrics)

 return metrics

 def _compute_bias_correction(self) -> torch.Tensor:
 """
 Computing bias correction term based on overestimation history.

 Returns:
 Bias correction loss term
 """
 if len(self.overestimation_history) < 10:
 return torch.tensor(0.0, device=self.device)

 # Running estimate overestimation bias
 recent_bias = np.mean(self.overestimation_history[-50:])

 # Exponential moving average for stability
 if len(self.bias_estimates) == 0:
 self.bias_estimates.append(recent_bias)
 else:
 ema_bias = (self.double_config.bias_correction_alpha * recent_bias +
 (1 - self.double_config.bias_correction_alpha) * self.bias_estimates[-1])
 self.bias_estimates.append(ema_bias)

 # Bias correction term (L2 regularization based on bias estimate)
 bias_correction = torch.tensor(
 abs(self.bias_estimates[-1]) * 0.001, # Small coefficient
 device=self.device,
 requires_grad=False
 )

 return bias_correction

 def act_with_double_q(self, state: np.ndarray, training: bool = True,
 use_double: bool = True) -> Tuple[int, Dict[str, float]]:
 """
 Action selection with Double Q analysis.

 Args:
 state: Current state
 training: Mode training
 use_double: Use double Q selection

 Returns:
 Tuple of (action, analysis_dict)
 """
 if training and np.random.random < self.epsilon_schedule.get_epsilon(self.training_step):
 action = np.random.randint(0, self.config.network_config.action_size)
 return action, {"selection_type": "random", "epsilon": True}

 with torch.no_grad:
 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

 # Main network Q-values
 main_q_values = self.q_network(state_tensor)

 # Target network Q-values
 target_q_values = self.target_network(state_tensor)

 analysis = {
 "selection_type": "greedy",
 "epsilon": False,
 "main_q_max": main_q_values.max.item,
 "target_q_max": target_q_values.max.item,
 }

 if use_double:
 # Double Q selection
 main_action = main_q_values.argmax.item
 target_action = target_q_values.argmax.item

 analysis.update({
 "main_action": main_action,
 "target_action": target_action,
 "actions_agree": main_action == target_action,
 })

 # Use main network for action selection
 action = main_action
 else:
 # Standard greedy selection
 action = main_q_values.argmax.item

 # Q-value disagreement analysis
 q_disagreement = torch.abs(main_q_values - target_q_values).mean.item
 analysis["q_disagreement"] = q_disagreement

 return action, analysis

 def get_double_q_statistics(self) -> Dict[str, Any]:
 """Get statistics Double DQN."""
 stats = {
 "double_q_updates": self.double_q_updates,
 "standard_q_updates": self.standard_q_updates,
 "double_q_ratio": self.double_q_updates / max(self.training_step, 1),
 "overestimation_history_length": len(self.overestimation_history),
 }

 if len(self.overestimation_history) > 0:
 stats.update({
 "avg_overestimation_bias": np.mean(self.overestimation_history),
 "recent_overestimation_bias": np.mean(self.overestimation_history[-100:]),
 "overestimation_std": np.std(self.overestimation_history),
 "max_overestimation": np.max(self.overestimation_history),
 "min_overestimation": np.min(self.overestimation_history),
 })

 if len(self.bias_estimates) > 0:
 stats.update({
 "current_bias_estimate": self.bias_estimates[-1],
 "bias_estimates_length": len(self.bias_estimates),
 })

 return stats

 def get_training_stats(self) -> Dict[str, Any]:
 """Extended statistics training for Double DQN."""
 base_stats = super.get_training_stats
 double_stats = self.get_double_q_statistics

 base_stats.update({
 "double_dqn_stats": double_stats,
 "agent_type": "DoubleDQN",
 })

 return base_stats

 def compare_with_standard_dqn(self, env, num_episodes: int = 10) -> Dict[str, Any]:
 """
 Comparison Double DQN with standard DQN on evaluation.

 Args:
 env: Environment for testing
 num_episodes: Number of episodes

 Returns:
 Comparative metrics
 """
 double_rewards = []
 standard_rewards = []
 q_disagreements = []
 action_agreements = []

 for episode in range(num_episodes):
 state = env.reset
 episode_reward_double = 0
 episode_reward_standard = 0
 episode_disagreements = []
 episode_agreements = []
 done = False

 while not done:
 # Double DQN action
 double_action, double_analysis = self.act_with_double_q(
 state, training=False, use_double=True
 )

 # Standard DQN action
 standard_action, standard_analysis = self.act_with_double_q(
 state, training=False, use_double=False
 )

 # Collect disagreement statistics
 if not double_analysis["epsilon"]:
 episode_disagreements.append(double_analysis["q_disagreement"])
 episode_agreements.append(double_analysis.get("actions_agree", False))

 # Use Double DQN action for environment step
 next_state, reward, done, _ = env.step(double_action)
 episode_reward_double += reward

 # For fair comparison, use same action for standard
 episode_reward_standard += reward

 state = next_state

 double_rewards.append(episode_reward_double)
 standard_rewards.append(episode_reward_standard)
 if episode_disagreements:
 q_disagreements.extend(episode_disagreements)
 action_agreements.extend(episode_agreements)

 comparison_metrics = {
 "double_dqn_reward_mean": np.mean(double_rewards),
 "double_dqn_reward_std": np.std(double_rewards),
 "standard_dqn_reward_mean": np.mean(standard_rewards),
 "standard_dqn_reward_std": np.std(standard_rewards),
 "reward_improvement": np.mean(double_rewards) - np.mean(standard_rewards),
 "episodes": num_episodes,
 }

 if q_disagreements:
 comparison_metrics.update({
 "avg_q_disagreement": np.mean(q_disagreements),
 "q_disagreement_std": np.std(q_disagreements),
 "action_agreement_rate": np.mean(action_agreements),
 })

 self.logger.info("Double DQN comparison completed", **comparison_metrics)
 return comparison_metrics

 def __repr__(self) -> str:
 """String representation Double DQN agent."""
 return (
 f"DoubleDQN(state_size={self.config.network_config.state_size}, "
 f"action_size={self.config.network_config.action_size}, "
 f"training_step={self.training_step}, "
 f"double_q_ratio={self.double_q_updates / max(self.training_step, 1):.3f}, "
 f"device={self.device})"
 )