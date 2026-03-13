"""
Rainbow DQN - combination of all DQN improvements with .

Rainbow DQN combines:
- Double DQN (action selection/evaluation decoupling)
- Dueling DQN (value/advantage separation)
- Prioritized Experience Replay (importance sampling)
- Multi-step returns (n-step bootstrapping)
- Distributional DQN (categorical value distribution)
- Noisy Networks (parameter space exploration)

Production-ready implementation with full configurability.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator
import structlog

from ..core.dqn import DQNConfig
from ..extensions.double_dqn import DoubleDQN, DoubleDQNConfig
from ..networks.dueling_network import DuelingNetworkConfig
from ..networks.noisy_linear import NoisyLinearConfig
from ..buffers.prioritized_replay import PrioritizedReplayConfig

logger = structlog.get_logger(__name__)


class RainbowDQNConfig(DoubleDQNConfig):
 """Configuration Rainbow DQN with alland components."""

 # Component toggles
 use_double_dqn: bool = Field(default=True, description="Use Double DQN")
 use_dueling: bool = Field(default=True, description="Use Dueling architecture")
 use_prioritized_replay: bool = Field(default=True, description="Use PER")
 use_multi_step: bool = Field(default=True, description="Use multi-step returns")
 use_distributional: bool = Field(default=True, description="Use distributional DQN")
 use_noisy_networks: bool = Field(default=True, description="Use noisy networks")

 # Multi-step parameters
 n_step: int = Field(default=3, description="N-step returns", ge=1, le=10)

 # Distributional DQN parameters
 num_atoms: int = Field(default=51, description="Number of atoms for distribution", gt=1)
 v_min: float = Field(default=-10.0, description="Minimum value for distribution")
 v_max: float = Field(default=10.0, description="Maximum value for distribution")

 # Noisy networks config
 noisy_config: NoisyLinearConfig = Field(default_factory=NoisyLinearConfig)

 # Dueling config
 dueling_config: Optional[DuelingNetworkConfig] = Field(default=None, description="Dueling network config")

 # Advanced prioritized replay
 per_config: PrioritizedReplayConfig = Field(default_factory=PrioritizedReplayConfig)

 @validator("v_max")
 def validate_v_max(cls, v, values):
 if "v_min" in values and v <= values["v_min"]:
 raise ValueError("v_max must be more v_min")
 return v

 @validator("num_atoms")
 def validate_num_atoms(cls, v):
 if v % 2 == 0:
 raise ValueError("num_atoms must be odd for symmetric distribution")
 return v


class CategoricalDQNNetwork(nn.Module):
 """
 Distributional DQN network for categorical value distribution.

 Instead scalar Q-values predicts full distribution values.
 Allows model uncertainty and improve stability.
 """

 def __init__(self, config: RainbowDQNConfig):
 super.__init__
 self.config = config
 self.state_size = config.network_config.state_size
 self.action_size = config.network_config.action_size
 self.num_atoms = config.num_atoms

 # Value distribution support
 self.v_min = config.v_min
 self.v_max = config.v_max
 self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

 # Support values for distribution
 self.register_buffer(
 'support',
 torch.linspace(self.v_min, self.v_max, self.num_atoms)
 )

 # Determining architecture based on enabled components
 self._build_network

 def _build_network(self) -> None:
 """Building network architecture."""
 # Base feature layers
 layers = []
 layer_sizes = [self.state_size] + self.config.network_config.hidden_layers

 for i in range(len(layer_sizes) - 1):
 in_size = layer_sizes[i]
 out_size = layer_sizes[i + 1]

 # Noisy or standard linear layer
 if self.config.use_noisy_networks:
 from ..networks.noisy_linear import NoisyLinear
 layer = NoisyLinear(in_size, out_size, self.config.noisy_config)
 else:
 layer = nn.Linear(in_size, out_size)

 layers.append(layer)

 # Batch normalization
 if self.config.network_config.use_batch_norm:
 layers.append(nn.BatchNorm1d(out_size))

 # Activation
 layers.append(nn.ReLU(inplace=True))

 # Dropout
 if self.config.network_config.dropout_rate > 0:
 layers.append(nn.Dropout(self.config.network_config.dropout_rate))

 self.feature_layers = nn.Sequential(*layers)
 feature_size = layer_sizes[-1]

 if self.config.use_dueling:
 # Dueling architecture for distributional values

 # Value stream - outputs distribution over single value
 if self.config.use_noisy_networks:
 from ..networks.noisy_linear import NoisyLinear
 self.value_head = NoisyLinear(feature_size, self.num_atoms, self.config.noisy_config)
 else:
 self.value_head = nn.Linear(feature_size, self.num_atoms)

 # Advantage stream - outputs distribution for each action
 if self.config.use_noisy_networks:
 self.advantage_head = NoisyLinear(
 feature_size,
 self.action_size * self.num_atoms,
 self.config.noisy_config
 )
 else:
 self.advantage_head = nn.Linear(feature_size, self.action_size * self.num_atoms)

 else:
 # Standard distributional head
 output_size = self.action_size * self.num_atoms

 if self.config.use_noisy_networks:
 from ..networks.noisy_linear import NoisyLinear
 self.q_head = NoisyLinear(feature_size, output_size, self.config.noisy_config)
 else:
 self.q_head = nn.Linear(feature_size, output_size)

 def forward(self, state: torch.Tensor) -> torch.Tensor:
 """
 Forward pass returning Q-value distributions.

 Args:
 state: State tensor [batch_size, state_size]

 Returns:
 Q-value distributions [batch_size, action_size, num_atoms]
 """
 if state.dim == 1:
 state = state.unsqueeze(0)

 # Feature extraction
 features = self.feature_layers(state)
 batch_size = features.size(0)

 if self.config.use_dueling:
 # Dueling distributional architecture

 # Value distribution [batch_size, num_atoms]
 value_logits = self.value_head(features)
 value_dist = F.softmax(value_logits, dim=1)

 # Advantage distributions [batch_size, action_size, num_atoms]
 advantage_logits = self.advantage_head(features)
 advantage_logits = advantage_logits.view(batch_size, self.action_size, self.num_atoms)

 # Normalize advantages (mean subtraction)
 advantage_mean = advantage_logits.mean(dim=1, keepdim=True)
 advantage_logits = advantage_logits - advantage_mean

 # Combine value and advantages
 # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
 value_dist = value_dist.unsqueeze(1).expand(-1, self.action_size, -1)
 q_logits = value_dist + F.softmax(advantage_logits, dim=2)

 # Renormalize to valid probability distribution
 q_dist = q_logits / q_logits.sum(dim=2, keepdim=True)

 else:
 # Standard distributional head
 q_logits = self.q_head(features)
 q_logits = q_logits.view(batch_size, self.action_size, self.num_atoms)
 q_dist = F.softmax(q_logits, dim=2)

 return q_dist

 def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
 """
 Get scalar Q-values from distributions.

 Args:
 state: State tensor

 Returns:
 Q-values [batch_size, action_size]
 """
 q_dist = self.forward(state)
 q_values = (q_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)
 return q_values

 def reset_noise(self) -> None:
 """Reset noise for noisy networks."""
 if self.config.use_noisy_networks:
 for module in self.modules:
 if hasattr(module, 'reset_noise'):
 module.reset_noise


class RainbowDQN:
 """
 Rainbow DQN - state-of-the-art DQN with alland improvements.

 Combines:
 - Double DQN for reduced overestimation bias
 - Dueling networks for better value estimation
 - Prioritized Experience Replay for efficient learning
 - Multi-step returns for better bootstrap estimates
 - Distributional DQN for uncertainty modeling
 - Noisy networks for parameter space exploration

 Enterprise features:
 - Full configurability all components
 - Production-ready error handling
 - Comprehensive monitoring and logging
 - Efficient memory management
 - Distributed training support
 """

 def __init__(self, config: RainbowDQNConfig):
 """
 Initialization Rainbow DQN.

 Args:
 config: Rainbow DQN configuration
 """
 self.config = config
 self.device = self._setup_device

 # Initialize networks
 self.q_network = CategoricalDQNNetwork(config).to(self.device)
 self.target_network = CategoricalDQNNetwork(config).to(self.device)

 # Copy parameters to target network
 self.target_network.load_state_dict(self.q_network.state_dict)

 # Disable gradients for target network
 for param in self.target_network.parameters:
 param.requires_grad = False

 # Initialize optimizer
 self.optimizer = torch.optim.Adam(
 self.q_network.parameters,
 lr=config.learning_rate,
 weight_decay=config.weight_decay
 )

 # Initialize replay buffer
 if config.use_prioritized_replay:
 from ..buffers.prioritized_replay import PrioritizedReplayBuffer
 self.replay_buffer = PrioritizedReplayBuffer(
 config=config.per_config,
 device=self.device
 )
 else:
 from ..buffers.replay_buffer import ReplayBuffer
 self.replay_buffer = ReplayBuffer(
 capacity=config.buffer_size,
 batch_size=config.batch_size,
 device=self.device
 )

 # Multi-step returns buffer
 if config.use_multi_step:
 self.multi_step_buffer = []

 # Training state
 self.training_step = 0
 self.episode_rewards = []
 self.loss_history = []

 # Performance tracking
 self.component_usage = {
 "double_dqn": config.use_double_dqn,
 "dueling": config.use_dueling,
 "prioritized_replay": config.use_prioritized_replay,
 "multi_step": config.use_multi_step,
 "distributional": config.use_distributional,
 "noisy_networks": config.use_noisy_networks,
 }

 self.logger = structlog.get_logger(__name__).bind(
 component="RainbowDQN",
 components=sum(self.component_usage.values),
 device=str(self.device)
 )

 self.logger.info("Rainbow DQN initialized",
 active_components=self.component_usage,
 total_params=sum(p.numel for p in self.q_network.parameters))

 def _setup_device(self) -> torch.device:
 """Setup computation device."""
 if torch.cuda.is_available:
 device = torch.device("cuda")
 self.logger.info(f"Using CUDA: {torch.cuda.get_device_name}")
 elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available:
 device = torch.device("mps")
 self.logger.info("Using Apple Metal Performance Shaders (MPS)")
 else:
 device = torch.device("cpu")
 self.logger.info("Using CPU")
 return device

 def act(self, state: np.ndarray, training: bool = True) -> int:
 """
 Action selection with noisy networks or epsilon-greedy.

 Args:
 state: Current state
 training: Training mode

 Returns:
 Selected action
 """
 with torch.no_grad:
 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

 if self.config.use_noisy_networks:
 # Noisy networks handle exploration automatically
 if training:
 self.q_network.reset_noise

 q_values = self.q_network.get_q_values(state_tensor)
 action = q_values.argmax.item

 else:
 # Standard epsilon-greedy with base DQN logic
 if training and np.random.random < self._get_epsilon:
 action = np.random.randint(0, self.config.network_config.action_size)
 else:
 q_values = self.q_network.get_q_values(state_tensor)
 action = q_values.argmax.item

 return action

 def _get_epsilon(self) -> float:
 """Get current epsilon for epsilon-greedy (if not used noisy networks)."""
 if self.config.use_noisy_networks:
 return 0.0 # No epsilon needed with noisy networks

 # Simple linear decay
 epsilon = max(
 self.config.epsilon_end,
 self.config.epsilon_start * (self.config.epsilon_decay ** self.training_step)
 )
 return epsilon

 def store_experience(self,
 state: np.ndarray,
 action: int,
 reward: float,
 next_state: np.ndarray,
 done: bool) -> None:
 """
 Store experience with multi-step returns support.

 Args:
 state: Current state
 action: Action taken
 reward: Reward received
 next_state: Next state
 done: Episode done flag
 """
 if self.config.use_multi_step:
 # Add to multi-step buffer
 self.multi_step_buffer.append((state, action, reward, next_state, done))

 # Process multi-step returns
 if len(self.multi_step_buffer) >= self.config.n_step or done:
 self._process_multi_step_experience
 else:
 # Direct storage
 if self.config.use_prioritized_replay:
 # Initial high priority for new experiences
 self.replay_buffer.push(state, action, reward, next_state, done)
 else:
 self.replay_buffer.push(state, action, reward, next_state, done)

 def _process_multi_step_experience(self) -> None:
 """Process multi-step returns."""
 if not self.multi_step_buffer:
 return

 # Calculate n-step return
 multi_step_return = 0.0
 discount = 1.0

 for i, (_, _, reward, _, _) in enumerate(self.multi_step_buffer):
 multi_step_return += discount * reward
 discount *= self.config.gamma

 # Get initial state and final state
 initial_state, initial_action, _, _, _ = self.multi_step_buffer[0]
 _, _, _, final_next_state, final_done = self.multi_step_buffer[-1]

 # Store multi-step experience
 if self.config.use_prioritized_replay:
 self.replay_buffer.push(
 initial_state,
 initial_action,
 multi_step_return,
 final_next_state,
 final_done
 )
 else:
 self.replay_buffer.push(
 initial_state,
 initial_action,
 multi_step_return,
 final_next_state,
 final_done
 )

 # Clear buffer
 self.multi_step_buffer.clear

 def train_step(self) -> Dict[str, float]:
 """
 Training step with alland Rainbow improvements.

 Returns:
 Training metrics
 """
 if len(self.replay_buffer) < self.config.min_replay_size:
 return {"status": "insufficient_data"}

 # Sample batch
 if self.config.use_prioritized_replay:
 batch_data = self.replay_buffer.sample_tensors
 (states, actions, rewards, next_states, dones,
 is_weights, tree_indices) = batch_data
 else:
 states, actions, rewards, next_states, dones = self.replay_buffer.sample_tensors
 is_weights = torch.ones(states.size(0)).to(self.device)
 tree_indices = None

 # Current Q distributions
 current_q_dist = self.q_network(states)
 current_q_dist = current_q_dist.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms))
 current_q_dist = current_q_dist.squeeze(1)

 # Target Q distributions
 with torch.no_grad:
 if self.config.use_double_dqn:
 # Double DQN: action selection with main network
 next_q_values = self.q_network.get_q_values(next_states)
 next_actions = next_q_values.argmax(1)

 # Action evaluation with target network
 next_q_dist = self.target_network(next_states)
 next_q_dist = next_q_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms))
 else:
 # Standard DQN
 next_q_values = self.target_network.get_q_values(next_states)
 next_actions = next_q_values.argmax(1)
 next_q_dist = self.target_network(next_states)
 next_q_dist = next_q_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.num_atoms))

 next_q_dist = next_q_dist.squeeze(1)

 # Distributional Bellman update
 target_q_dist = self._distributional_bellman_update(
 rewards, next_q_dist, dones
 )

 # Compute distributional loss (cross-entropy)
 loss = -(target_q_dist * torch.log(current_q_dist + 1e-8)).sum(1)

 # Weighted loss with importance sampling
 loss = (loss * is_weights).mean

 # Optimization step
 self.optimizer.zero_grad
 loss.backward

 # Gradient clipping
 grad_norm = torch.nn.utils.clip_grad_norm_(
 self.q_network.parameters,
 self.config.grad_clip_norm
 )

 self.optimizer.step

 # Update priorities in PER
 if self.config.use_prioritized_replay and tree_indices is not None:
 # TD errors for priority update
 td_errors = (target_q_dist * self.q_network.support.unsqueeze(0)).sum(1) - \
 (current_q_dist * self.q_network.support.unsqueeze(0)).sum(1)
 td_errors = td_errors.abs.detach

 self.replay_buffer.update_priorities(tree_indices, td_errors)

 # Update target network
 self.training_step += 1
 if self.training_step % self.config.target_update_freq == 0:
 self.target_network.load_state_dict(self.q_network.state_dict)

 # Metrics
 metrics = {
 "loss": loss.item,
 "grad_norm": grad_norm.item,
 "epsilon": self._get_epsilon,
 "training_step": self.training_step,
 "replay_buffer_size": len(self.replay_buffer),
 }

 # Component-specific metrics
 if self.config.use_noisy_networks:
 noise_stats = {}
 for name, module in self.q_network.named_modules:
 if hasattr(module, 'get_noise_statistics'):
 noise_stats.update(module.get_noise_statistics)
 if noise_stats:
 metrics["noise_stats"] = noise_stats

 self.loss_history.append(loss.item)

 if self.training_step % self.config.log_freq == 0:
 self.logger.info("Rainbow training step", **metrics)

 return metrics

 def _distributional_bellman_update(self,
 rewards: torch.Tensor,
 next_q_dist: torch.Tensor,
 dones: torch.Tensor) -> torch.Tensor:
 """
 Distributional Bellman update for categorical DQN.

 Args:
 rewards: Rewards [batch_size]
 next_q_dist: Next Q distributions [batch_size, num_atoms]
 dones: Done flags [batch_size]

 Returns:
 Target Q distributions [batch_size, num_atoms]
 """
 batch_size = rewards.size(0)

 # Project rewards to support
 rewards = rewards.unsqueeze(1).expand(-1, self.config.num_atoms)
 support = self.q_network.support.unsqueeze(0).expand(batch_size, -1)

 # Bellman update: r + gamma * z
 dones = dones.unsqueeze(1).expand(-1, self.config.num_atoms)
 target_support = rewards + self.config.gamma * support * (1 - dones.float)

 # Clamp to distribution support
 target_support = torch.clamp(target_support, self.config.v_min, self.config.v_max)

 # Project to categorical distribution
 target_q_dist = torch.zeros_like(next_q_dist)

 # Distribute probability mass
 delta_z = (self.config.v_max - self.config.v_min) / (self.config.num_atoms - 1)

 for i in range(batch_size):
 for j in range(self.config.num_atoms):
 target_z = target_support[i, j].item

 # Find nearest support points
 b = (target_z - self.config.v_min) / delta_z
 l = int(np.floor(b))
 u = int(np.ceil(b))

 # Distribute probability
 if l == u:
 if 0 <= l < self.config.num_atoms:
 target_q_dist[i, l] += next_q_dist[i, j]
 else:
 if 0 <= l < self.config.num_atoms:
 target_q_dist[i, l] += next_q_dist[i, j] * (u - b)
 if 0 <= u < self.config.num_atoms:
 target_q_dist[i, u] += next_q_dist[i, j] * (b - l)

 return target_q_dist

 def get_rainbow_statistics(self) -> Dict[str, Any]:
 """Get Rainbow-specific statistics."""
 stats = {
 "training_step": self.training_step,
 "active_components": self.component_usage,
 "component_count": sum(self.component_usage.values),
 "replay_buffer_size": len(self.replay_buffer),
 "loss_history_length": len(self.loss_history),
 }

 # Component-specific stats
 if self.config.use_prioritized_replay:
 stats["per_stats"] = self.replay_buffer.get_priority_statistics

 if self.config.use_noisy_networks:
 stats["noisy_stats"] = {}
 for name, module in self.q_network.named_modules:
 if hasattr(module, 'get_noise_statistics'):
 module_stats = module.get_noise_statistics
 stats["noisy_stats"][name] = module_stats

 if self.config.use_multi_step:
 stats["multi_step_buffer_size"] = len(self.multi_step_buffer)
 stats["n_step"] = self.config.n_step

 return stats

 def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
 """Save Rainbow DQN checkpoint."""
 checkpoint = {
 "config": self.config.dict,
 "training_step": self.training_step,
 "q_network_state": self.q_network.state_dict,
 "target_network_state": self.target_network.state_dict,
 "optimizer_state": self.optimizer.state_dict,
 "loss_history": self.loss_history,
 "episode_rewards": self.episode_rewards,
 "component_usage": self.component_usage,
 "rainbow_statistics": self.get_rainbow_statistics,
 "metadata": metadata or {},
 "timestamp": datetime.now.isoformat,
 }

 torch.save(checkpoint, filepath)
 self.logger.info("Rainbow checkpoint saved", filepath=filepath)

 def __repr__(self) -> str:
 """String representation of Rainbow DQN."""
 active_components = [name for name, active in self.component_usage.items if active]
 return (
 f"RainbowDQN(components={active_components}, "
 f"training_step={self.training_step}, "
 f"device={self.device})"
 )