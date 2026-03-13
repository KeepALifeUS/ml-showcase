"""
Advanced PPO Agent for Crypto Trading
Enterprise-grade PPO implementation
Implements curiosity-driven exploration, multi-asset support, and adaptive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import logging
from collections import defaultdict, deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import time
from enum import Enum

# ====================================================================
# CONFIGURATION & TYPES
# ====================================================================

class ActionType(Enum):
 """Trading action types"""
 HOLD = 0
 BUY = 1
 SELL = 2
 LONG = 3
 SHORT = 4
 CLOSE = 5

class ExplorationStrategy(Enum):
 """Exploration strategies"""
 EPSILON_GREEDY = "epsilon_greedy"
 BOLTZMANN = "boltzmann"
 CURIOSITY_DRIVEN = "curiosity_driven"
 NOISE_BASED = "noise_based"
 UCB = "ucb"

@dataclass
class AdvancedPPOConfig:
 """Advanced PPO Configuration"""

 # Core PPO parameters
 learning_rate: float = 3e-4
 gamma: float = 0.99
 gae_lambda: float = 0.95
 clip_range: float = 0.2
 clip_range_vf: Optional[float] = None

 # Advanced features
 use_curiosity: bool = True
 curiosity_strength: float = 0.01
 use_rnd: bool = True # Random Network Distillation
 use_icm: bool = False # Intrinsic Curiosity Module

 # Multi-asset support
 num_assets: int = 10
 asset_correlation_window: int = 100
 portfolio_optimization: bool = True
 risk_adjusted_rewards: bool = True

 # Adaptive learning
 adaptive_lr: bool = True
 lr_schedule: str = "cosine" # linear, cosine, exponential
 adaptive_clip: bool = True
 kl_target: float = 0.01

 # Memory and experience
 memory_size: int = 100000
 prioritized_replay: bool = True
 n_step_returns: int = 5

 # Network architecture
 hidden_sizes: List[int] = field(default_factory=lambda: [512, 512, 256])
 activation: str = "relu"
 use_lstm: bool = True
 lstm_hidden_size: int = 256
 use_attention: bool = True
 attention_heads: int = 8

 # Training parameters
 batch_size: int = 256
 mini_batch_size: int = 64
 n_epochs: int = 10
 max_grad_norm: float = 0.5

 # Regularization
 ent_coef: float = 0.01
 vf_coef: float = 0.5
 l2_reg: float = 1e-5
 dropout_rate: float = 0.1

 # Performance optimization
 device: str = "cuda" if torch.cuda.is_available() else "cpu"
 num_workers: int = 4
 mixed_precision: bool = True
 gradient_checkpointing: bool = True

 # Monitoring
 log_interval: int = 100
 eval_interval: int = 500
 checkpoint_interval: int = 1000
 tensorboard: bool = True

# ====================================================================
# CURIOSITY MODULE
# ====================================================================

class CuriosityModule(nn.Module):
 """
 Implements curiosity-driven exploration using RND
 Intrinsic motivation for better exploration
 """

 def __init__(self, obs_dim: int, hidden_dim: int = 256):
 super().__init__()

 # Random target network (fixed)
 self.target_network = nn.Sequential(
 nn.Linear(obs_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, 128)
 )

 # Predictor network (trained)
 self.predictor_network = nn.Sequential(
 nn.Linear(obs_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, 128)
 )

 # Freeze target network
 for param in self.target_network.parameters():
 param.requires_grad = False

 def forward(self, obs: torch.Tensor) -> torch.Tensor:
 """Calculate curiosity bonus"""
 with torch.no_grad():
 target_features = self.target_network(obs)

 predicted_features = self.predictor_network(obs)

 # Curiosity bonus is the prediction error
 curiosity_bonus = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=-1)

 return curiosity_bonus

 def train_predictor(self, obs: torch.Tensor, optimizer: optim.Optimizer) -> float:
 """Train the predictor network"""
 with torch.no_grad():
 target_features = self.target_network(obs)

 predicted_features = self.predictor_network(obs)
 loss = F.mse_loss(predicted_features, target_features)

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()

 return loss.item()

# ====================================================================
# ATTENTION MODULE
# ====================================================================

class MultiHeadAttentionLayer(nn.Module):
 """
 Multi-head attention for asset correlation modeling
 Attention mechanisms for complex dependencies
 """

 def __init__(self, d_model: int, n_heads: int = 8):
 super().__init__()
 self.d_model = d_model
 self.n_heads = n_heads
 self.d_k = d_model // n_heads

 self.W_q = nn.Linear(d_model, d_model)
 self.W_k = nn.Linear(d_model, d_model)
 self.W_v = nn.Linear(d_model, d_model)
 self.W_o = nn.Linear(d_model, d_model)

 self.layer_norm = nn.LayerNorm(d_model)
 self.dropout = nn.Dropout(0.1)

 def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
 batch_size, seq_len, _ = x.size()

 # Linear transformations and split into heads
 Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
 K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
 V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

 # Attention scores
 scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

 if mask is not None:
 scores = scores.masked_fill(mask == 0, -1e9)

 # Attention weights
 attn_weights = F.softmax(scores, dim=-1)
 attn_weights = self.dropout(attn_weights)

 # Weighted sum
 context = torch.matmul(attn_weights, V)

 # Concatenate heads
 context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

 # Output projection
 output = self.W_o(context)

 # Residual connection and layer norm
 output = self.layer_norm(x + self.dropout(output))

 return output

# ====================================================================
# ACTOR-CRITIC NETWORK
# ====================================================================

class AdvancedActorCriticNetwork(nn.Module):
 """
 Advanced Actor-Critic Network with LSTM and Attention
 State-of-the-art architecture for trading
 """

 def __init__(self, config: AdvancedPPOConfig, obs_dim: int, action_dim: int):
 super().__init__()
 self.config = config

 # Feature extraction
 self.feature_extractor = nn.Sequential(
 nn.Linear(obs_dim, config.hidden_sizes[0]),
 nn.ReLU(),
 nn.Dropout(config.dropout_rate),
 nn.Linear(config.hidden_sizes[0], config.hidden_sizes[1]),
 nn.ReLU(),
 nn.Dropout(config.dropout_rate)
 )

 # LSTM for temporal dependencies
 if config.use_lstm:
 self.lstm = nn.LSTM(
 config.hidden_sizes[1],
 config.lstm_hidden_size,
 batch_first=True,
 bidirectional=True
 )
 lstm_output_size = config.lstm_hidden_size * 2
 else:
 lstm_output_size = config.hidden_sizes[1]

 # Attention layer
 if config.use_attention:
 self.attention = MultiHeadAttentionLayer(lstm_output_size, config.attention_heads)

 # Actor head (policy)
 self.actor = nn.Sequential(
 nn.Linear(lstm_output_size, config.hidden_sizes[2]),
 nn.ReLU(),
 nn.Linear(config.hidden_sizes[2], action_dim)
 )

 # Critic head (value function)
 self.critic = nn.Sequential(
 nn.Linear(lstm_output_size, config.hidden_sizes[2]),
 nn.ReLU(),
 nn.Linear(config.hidden_sizes[2], 1)
 )

 # Log std for continuous actions
 self.log_std = nn.Parameter(torch.zeros(action_dim))

 def forward(
 self,
 obs: torch.Tensor,
 hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
 """
 Forward pass

 Returns:
 action_logits, values, log_std, hidden_state
 """
 # Feature extraction
 features = self.feature_extractor(obs)

 # LSTM processing
 if self.config.use_lstm:
 if len(features.shape) == 2:
 features = features.unsqueeze(1)

 if hidden_state is None:
 lstm_out, hidden_state = self.lstm(features)
 else:
 lstm_out, hidden_state = self.lstm(features, hidden_state)

 features = lstm_out.squeeze(1) if lstm_out.shape[1] == 1 else lstm_out[:, -1, :]

 # Attention processing
 if self.config.use_attention and len(features.shape) == 3:
 features = self.attention(features)
 features = features[:, -1, :] if len(features.shape) == 3 else features

 # Actor and critic outputs
 action_logits = self.actor(features)
 values = self.critic(features)

 return action_logits, values, self.log_std, hidden_state

# ====================================================================
# ADVANCED PPO AGENT
# ====================================================================

class AdvancedPPOAgent:
 """
 Advanced PPO Agent
 """

 def __init__(
 self,
 config: AdvancedPPOConfig,
 obs_dim: int,
 action_dim: int,
 logger: Optional[logging.Logger] = None
 ):
 self.config = config
 self.obs_dim = obs_dim
 self.action_dim = action_dim
 self.logger = logger or logging.getLogger(__name__)

 # Device setup
 self.device = torch.device(config.device)

 # Networks
 self.actor_critic = AdvancedActorCriticNetwork(config, obs_dim, action_dim).to(self.device)
 self.old_actor_critic = AdvancedActorCriticNetwork(config, obs_dim, action_dim).to(self.device)
 self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

 # Curiosity module
 if config.use_curiosity:
 self.curiosity = CuriosityModule(obs_dim).to(self.device)
 self.curiosity_optimizer = optim.Adam(
 self.curiosity.predictor_network.parameters(),
 lr=config.learning_rate
 )

 # Optimizer with adaptive learning rate
 self.optimizer = optim.AdamW(
 self.actor_critic.parameters(),
 lr=config.learning_rate,
 weight_decay=config.l2_reg
 )

 # Learning rate scheduler
 if config.adaptive_lr:
 if config.lr_schedule == "cosine":
 self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
 self.optimizer, T_max=10000
 )
 elif config.lr_schedule == "exponential":
 self.scheduler = optim.lr_scheduler.ExponentialLR(
 self.optimizer, gamma=0.99
 )
 else:
 self.scheduler = optim.lr_scheduler.LinearLR(
 self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=10000
 )

 # Memory buffers
 self.observations = deque(maxlen=config.memory_size)
 self.actions = deque(maxlen=config.memory_size)
 self.rewards = deque(maxlen=config.memory_size)
 self.log_probs = deque(maxlen=config.memory_size)
 self.values = deque(maxlen=config.memory_size)
 self.dones = deque(maxlen=config.memory_size)

 # Statistics tracking
 self.episode_rewards = deque(maxlen=100)
 self.episode_lengths = deque(maxlen=100)
 self.kl_divergences = deque(maxlen=100)
 self.entropy_values = deque(maxlen=100)

 # Performance metrics
 self.total_steps = 0
 self.total_episodes = 0
 self.best_reward = float('-inf')

 # Adaptive parameters
 self.current_clip_range = config.clip_range
 self.current_ent_coef = config.ent_coef

 def select_action(
 self,
 obs: np.ndarray,
 deterministic: bool = False,
 hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
 ) -> Tuple[np.ndarray, float, float, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
 """
 Select action using the current policy

 Returns:
 action, log_prob, value, hidden_state
 """
 with torch.no_grad():
 obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

 # Get network outputs
 action_logits, values, log_std, hidden_state = self.actor_critic(obs_tensor, hidden_state)

 # Create action distribution
 if self.action_dim == 1:
 # Discrete actions
 probs = F.softmax(action_logits, dim=-1)
 dist = Categorical(probs)
 else:
 # Continuous actions
 std = torch.exp(log_std)
 dist = Normal(action_logits, std)

 # Sample or take deterministic action
 if deterministic:
 action = dist.mean if hasattr(dist, 'mean') else probs.argmax(dim=-1)
 else:
 action = dist.sample()

 log_prob = dist.log_prob(action).sum(dim=-1)

 # Add curiosity bonus to value estimate
 if self.config.use_curiosity and not deterministic:
 curiosity_bonus = self.curiosity(obs_tensor)
 values = values + self.config.curiosity_strength * curiosity_bonus.unsqueeze(-1)

 return (
 action.cpu().numpy().squeeze(),
 log_prob.cpu().item(),
 values.cpu().item(),
 hidden_state
 )

 def store_transition(
 self,
 obs: np.ndarray,
 action: np.ndarray,
 reward: float,
 log_prob: float,
 value: float,
 done: bool
 ):
 """Store transition in memory buffer"""
 self.observations.append(obs)
 self.actions.append(action)
 self.rewards.append(reward)
 self.log_probs.append(log_prob)
 self.values.append(value)
 self.dones.append(done)

 def compute_gae(
 self,
 rewards: List[float],
 values: List[float],
 dones: List[bool],
 next_value: float
 ) -> Tuple[np.ndarray, np.ndarray]:
 """
 Compute Generalized Advantage Estimation (GAE)

 Returns:
 advantages, returns
 """
 advantages = []
 returns = []

 gae = 0
 for t in reversed(range(len(rewards))):
 if t == len(rewards) - 1:
 next_value_t = next_value
 else:
 next_value_t = values[t + 1]

 delta = rewards[t] + self.config.gamma * next_value_t * (1 - dones[t]) - values[t]
 gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae

 advantages.insert(0, gae)
 returns.insert(0, gae + values[t])

 advantages = np.array(advantages)
 returns = np.array(returns)

 # Normalize advantages
 if self.config.normalize_advantage:
 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

 return advantages, returns

 def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
 """
 Update policy using PPO

 Returns:
 Dictionary of training metrics
 """
 if len(self.observations) < (batch_size or self.config.batch_size):
 return {}

 # Convert buffers to tensors
 obs = torch.FloatTensor(list(self.observations)).to(self.device)
 actions = torch.FloatTensor(list(self.actions)).to(self.device)
 old_log_probs = torch.FloatTensor(list(self.log_probs)).to(self.device)

 # Compute advantages and returns
 advantages, returns = self.compute_gae(
 list(self.rewards),
 list(self.values),
 list(self.dones),
 0.0 # Bootstrap value
 )

 advantages = torch.FloatTensor(advantages).to(self.device)
 returns = torch.FloatTensor(returns).to(self.device)

 # Update old policy
 self.old_actor_critic.load_state_dict(self.actor_critic.state_dict())

 # Training metrics
 metrics = defaultdict(list)

 # PPO epochs
 for epoch in range(self.config.n_epochs):
 # Create mini-batches
 indices = torch.randperm(len(obs))

 for start in range(0, len(obs), self.config.mini_batch_size):
 end = min(start + self.config.mini_batch_size, len(obs))
 batch_indices = indices[start:end]

 batch_obs = obs[batch_indices]
 batch_actions = actions[batch_indices]
 batch_old_log_probs = old_log_probs[batch_indices]
 batch_advantages = advantages[batch_indices]
 batch_returns = returns[batch_indices]

 # Forward pass
 action_logits, values, log_std, _ = self.actor_critic(batch_obs)

 # Create action distribution
 if self.action_dim == 1:
 probs = F.softmax(action_logits, dim=-1)
 dist = Categorical(probs)
 else:
 std = torch.exp(log_std)
 dist = Normal(action_logits, std)

 # Calculate log probabilities
 new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
 entropy = dist.entropy().mean()

 # PPO loss
 ratio = torch.exp(new_log_probs - batch_old_log_probs)
 surr1 = ratio * batch_advantages
 surr2 = torch.clamp(ratio, 1 - self.current_clip_range, 1 + self.current_clip_range) * batch_advantages

 policy_loss = -torch.min(surr1, surr2).mean()

 # Value loss
 value_loss = F.mse_loss(values.squeeze(), batch_returns)

 # Total loss
 loss = policy_loss + self.config.vf_coef * value_loss - self.current_ent_coef * entropy

 # Update curiosity module
 if self.config.use_curiosity:
 curiosity_loss = self.curiosity.train_predictor(batch_obs, self.curiosity_optimizer)
 metrics['curiosity_loss'].append(curiosity_loss)

 # Backward pass
 self.optimizer.zero_grad()
 loss.backward()

 # Gradient clipping
 nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)

 self.optimizer.step()

 # Track metrics
 metrics['policy_loss'].append(policy_loss.item())
 metrics['value_loss'].append(value_loss.item())
 metrics['entropy'].append(entropy.item())
 metrics['kl_divergence'].append((new_log_probs - batch_old_log_probs).mean().item())

 # Early stopping based on KL divergence
 if self.config.kl_target and metrics['kl_divergence'][-1] > self.config.kl_target * 1.5:
 break

 # Early stopping for epoch
 if self.config.kl_target and np.mean(metrics['kl_divergence']) > self.config.kl_target * 1.5:
 break

 # Update learning rate
 if self.config.adaptive_lr:
 self.scheduler.step()

 # Adaptive clip range
 if self.config.adaptive_clip:
 kl_mean = np.mean(metrics['kl_divergence'])
 if kl_mean > self.config.kl_target * 2:
 self.current_clip_range *= 0.99
 elif kl_mean < self.config.kl_target / 2:
 self.current_clip_range *= 1.01
 self.current_clip_range = np.clip(self.current_clip_range, 0.05, 0.3)

 # Clear buffers
 self.observations.clear()
 self.actions.clear()
 self.rewards.clear()
 self.log_probs.clear()
 self.values.clear()
 self.dones.clear()

 # Return averaged metrics
 return {k: np.mean(v) for k, v in metrics.items()}

 def save(self, path: str):
 """Save agent state"""
 torch.save({
 'actor_critic': self.actor_critic.state_dict(),
 'optimizer': self.optimizer.state_dict(),
 'scheduler': self.scheduler.state_dict() if self.config.adaptive_lr else None,
 'curiosity': self.curiosity.state_dict() if self.config.use_curiosity else None,
 'config': self.config,
 'total_steps': self.total_steps,
 'total_episodes': self.total_episodes,
 'best_reward': self.best_reward
 }, path)

 self.logger.info(f"Agent saved to {path}")

 def load(self, path: str):
 """Load agent state"""
 checkpoint = torch.load(path, map_location=self.device)

 self.actor_critic.load_state_dict(checkpoint['actor_critic'])
 self.optimizer.load_state_dict(checkpoint['optimizer'])

 if self.config.adaptive_lr and checkpoint['scheduler']:
 self.scheduler.load_state_dict(checkpoint['scheduler'])

 if self.config.use_curiosity and checkpoint['curiosity']:
 self.curiosity.load_state_dict(checkpoint['curiosity'])

 self.total_steps = checkpoint['total_steps']
 self.total_episodes = checkpoint['total_episodes']
 self.best_reward = checkpoint['best_reward']

 self.logger.info(f"Agent loaded from {path}")

# ====================================================================
# DISTRIBUTED PPO COORDINATOR
# ====================================================================

class DistributedPPOCoordinator:
 """
 Coordinates multiple PPO agents for distributed training
 Scalable distributed reinforcement learning
 """

 def __init__(
 self,
 config: AdvancedPPOConfig,
 num_agents: int = 4,
 logger: Optional[logging.Logger] = None
 ):
 self.config = config
 self.num_agents = num_agents
 self.logger = logger or logging.getLogger(__name__)

 # Create agent pool
 self.agents = []
 self.executors = []

 for i in range(num_agents):
 agent_config = config
 agent = AdvancedPPOAgent(agent_config, config.hidden_sizes[0], 6) # Assuming 6 actions
 self.agents.append(agent)
 self.executors.append(ThreadPoolExecutor(max_workers=1))

 # Shared model for synchronization
 self.shared_model = AdvancedActorCriticNetwork(
 config, config.hidden_sizes[0], 6
 ).to(config.device)

 # Global statistics
 self.global_steps = 0
 self.global_episodes = 0
 self.sync_interval = 100

 async def train_distributed(self, num_steps: int = 10000):
 """
 Train agents in distributed manner
 """
 tasks = []

 for i, agent in enumerate(self.agents):
 task = asyncio.create_task(self._train_agent(agent, i, num_steps))
 tasks.append(task)

 # Periodically sync models
 sync_task = asyncio.create_task(self._sync_models())
 tasks.append(sync_task)

 await asyncio.gather(*tasks)

 async def _train_agent(self, agent: AdvancedPPOAgent, agent_id: int, num_steps: int):
 """
 Train individual agent
 """
 local_steps = 0

 while local_steps < num_steps:
 # Collect experience
 # This would interact with environment

 # Update policy
 metrics = agent.update()

 local_steps += self.config.batch_size

 # Log progress
 if local_steps % self.config.log_interval == 0:
 self.logger.info(f"Agent {agent_id}: Steps {local_steps}, Metrics: {metrics}")

 await asyncio.sleep(0) # Yield control

 async def _sync_models(self):
 """
 Synchronize models across agents
 """
 while True:
 await asyncio.sleep(self.sync_interval)

 # Average all agent models
 state_dicts = [agent.actor_critic.state_dict() for agent in self.agents]

 # Average parameters
 averaged_state = {}
 for key in state_dicts[0].keys():
 averaged_state[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)

 # Update shared model
 self.shared_model.load_state_dict(averaged_state)

 # Distribute back to agents
 for agent in self.agents:
 agent.actor_critic.load_state_dict(averaged_state)

 self.logger.info("Models synchronized across agents")

# ====================================================================
# PERFORMANCE MONITOR
# ====================================================================

class PPOPerformanceMonitor:
 """
 Monitors PPO training performance and metrics
 Comprehensive performance tracking
 """

 def __init__(self, tensorboard_dir: Optional[str] = None):
 self.metrics_history = defaultdict(list)
 self.episode_rewards = []
 self.episode_lengths = []

 if tensorboard_dir:
 from torch.utils.tensorboard import SummaryWriter
 self.writer = SummaryWriter(tensorboard_dir)
 else:
 self.writer = None

 def log_metrics(self, step: int, metrics: Dict[str, float]):
 """Log training metrics"""
 for key, value in metrics.items():
 self.metrics_history[key].append(value)

 if self.writer:
 self.writer.add_scalar(f"train/{key}", value, step)

 def log_episode(self, episode: int, reward: float, length: int):
 """Log episode statistics"""
 self.episode_rewards.append(reward)
 self.episode_lengths.append(length)

 if self.writer:
 self.writer.add_scalar("episode/reward", reward, episode)
 self.writer.add_scalar("episode/length", length, episode)

 def get_statistics(self) -> Dict[str, Any]:
 """Get current statistics"""
 stats = {}

 if self.episode_rewards:
 stats['avg_reward'] = np.mean(self.episode_rewards[-100:])
 stats['max_reward'] = np.max(self.episode_rewards[-100:])
 stats['min_reward'] = np.min(self.episode_rewards[-100:])

 if self.episode_lengths:
 stats['avg_length'] = np.mean(self.episode_lengths[-100:])

 for key, values in self.metrics_history.items():
 if values:
 stats[f"avg_{key}"] = np.mean(values[-100:])

 return stats

 def close(self):
 """Close tensorboard writer"""
 if self.writer:
 self.writer.close()