"""
Deep Q-Network (DQN) implementation with enterprise patterns.

Complete DQN algorithm implementation with modern optimizations:
- Epsilon-greedy exploration with adaptive decay
- Target network for training stability
- Experience replay for efficient data usage
- Gradient clipping and regularization
- Comprehensive monitoring and logging
- Production-ready error handling
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel, Field, validator
import structlog
from dataclasses import dataclass
from datetime import datetime
import pickle
import json

from ..networks.q_network import QNetwork, QNetworkConfig
from ..buffers.replay_buffer import ReplayBuffer
from ..utils.epsilon_schedule import EpsilonSchedule

logger = structlog.get_logger(__name__)


@dataclass
class Experience:
 """Structure for storing experience (s, a, r, s', done)."""
 state: np.ndarray
 action: int
 reward: float
 next_state: np.ndarray
 done: bool

 def to_dict(self) -> Dict[str, Any]:
 """Convert to dictionary for serialization."""
 return {
 "state": self.state.tolist,
 "action": self.action,
 "reward": self.reward,
 "next_state": self.next_state.tolist,
 "done": self.done,
 }


class DQNConfig(BaseModel):
 """DQN agent configuration with validation."""

 # Network configuration
 network_config: QNetworkConfig = Field(..., description="Q-network configuration")

 # Training hyperparameters
 learning_rate: float = Field(default=1e-4, description="Learning rate", gt=0, le=1e-1)
 gamma: float = Field(default=0.99, description="Discount factor", ge=0, le=1.0)
 epsilon_start: float = Field(default=1.0, description="Initial epsilon", ge=0, le=1.0)
 epsilon_end: float = Field(default=0.01, description="Final epsilon", ge=0, le=1.0)
 epsilon_decay: float = Field(default=0.995, description="Decay rate epsilon", ge=0.9, le=1.0)

 # Experience replay
 buffer_size: int = Field(default=100000, description="Size replay buffer", gt=0)
 batch_size: int = Field(default=64, description="Batch size", gt=0, le=512)
 min_replay_size: int = Field(default=1000, description="Minimum experience for training", gt=0)

 # Target network updates
 target_update_freq: int = Field(default=1000, description="Frequency update target network", gt=0)
 soft_update_tau: Optional[float] = Field(default=None, description="Tau for soft updates", ge=0, le=1.0)

 # Optimization
 optimizer_type: str = Field(default="adam", description="Type optimizer")
 weight_decay: float = Field(default=1e-5, description="L2 regularization", ge=0)
 grad_clip_norm: float = Field(default=1.0, description="Gradient clipping norm", gt=0)

 # Loss function
 loss_type: str = Field(default="mse", description="Type loss function")
 huber_delta: float = Field(default=1.0, description="Delta for Huber loss", gt=0)

 # Monitoring
 log_freq: int = Field(default=1000, description="Frequency logging", gt=0)
 save_freq: int = Field(default=10000, description="Frequency saving", gt=0)
 eval_freq: int = Field(default=5000, description="Frequency evaluation", gt=0)

 # Device
 device: str = Field(default="auto", description="Device for computations")
 seed: Optional[int] = Field(default=None, description="Random seed")

 @validator("epsilon_end")
 def validate_epsilon_end(cls, v, values):
 if "epsilon_start" in values and v >= values["epsilon_start"]:
 raise ValueError("epsilon_end must be less than epsilon_start")
 return v

 @validator("min_replay_size")
 def validate_min_replay_size(cls, v, values):
 if "batch_size" in values and v < values["batch_size"]:
 raise ValueError("min_replay_size must be >= batch_size")
 return v

 @validator("optimizer_type")
 def validate_optimizer(cls, v):
 valid_optimizers = ["adam", "adamw", "rmsprop", "sgd"]
 if v not in valid_optimizers:
 raise ValueError(f"Optimizer must be one of: {valid_optimizers}")
 return v

 @validator("loss_type")
 def validate_loss(cls, v):
 valid_losses = ["mse", "huber", "smooth_l1"]
 if v not in valid_losses:
 raise ValueError(f"Loss must be one of: {valid_losses}")
 return v


class DQN:
 """
 Deep Q-Network (DQN) agent with enterprise-grade implementation.

 Features:
 - Epsilon-greedy exploration with adaptive strategies
 - Target network for training stability
 - Experience replay for efficient data usage
 - Configurable loss functions (MSE, Huber, Smooth L1)
 - Gradient clipping and regularization
 - Comprehensive logging and monitoring
 - Checkpointing and model persistence
 - Production-ready error handling
 """

 def __init__(self, config: DQNConfig):
 """
 DQN agent initialization.

 Args:
 config: Agent configuration
 """
 self.config = config
 self.training_step = 0
 self.episode_rewards = []
 self.loss_history = []

 # Setup device
 self.device = self._setup_device

 # Random seed setup
 if config.seed is not None:
 self._set_seed(config.seed)

 # Component initialization
 self._initialize_networks
 self._initialize_optimizer
 self._initialize_replay_buffer
 self._initialize_epsilon_schedule

 # Logging setup
 self.logger = structlog.get_logger(__name__).bind(
 agent_type="DQN",
 device=str(self.device)
 )

 self.logger.info("DQN agent initialized", config=config.dict)

 def _setup_device(self) -> torch.device:
 """Device setup for computations."""
 if self.config.device == "auto":
 if torch.cuda.is_available:
 device = torch.device("cuda")
 self.logger.info(f"CUDA available: {torch.cuda.get_device_name}")
 elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available:
 device = torch.device("mps")
 self.logger.info("Using Apple Metal Performance Shaders (MPS)")
 else:
 device = torch.device("cpu")
 self.logger.info("Using CPU")
 else:
 device = torch.device(self.config.device)

 return device

 def _set_seed(self, seed: int) -> None:
 """Setting random seed for reproducibility."""
 np.random.seed(seed)
 torch.manual_seed(seed)
 if torch.cuda.is_available:
 torch.cuda.manual_seed(seed)
 torch.cuda.manual_seed_all(seed)

 # For full determinism (may slow down training)
 torch.backends.cudnn.deterministic = True
 torch.backends.cudnn.benchmark = False

 self.logger.info("Random seed set", seed=seed)

 def _initialize_networks(self) -> None:
 """Initialization Q-networks."""
 # Main network
 self.q_network = QNetwork(self.config.network_config).to(self.device)

 # Target network (copy of main)
 self.target_network = QNetwork(self.config.network_config).to(self.device)
 self.target_network.load_state_dict(self.q_network.state_dict)

 # Freezing target network
 for param in self.target_network.parameters:
 param.requires_grad = False

 self.logger.info("Q-networks initialized",
 params=self.q_network.get_network_stats["total_parameters"])

 def _initialize_optimizer(self) -> None:
 """Initialization optimizer."""
 optimizer_map = {
 "adam": optim.Adam,
 "adamw": optim.AdamW,
 "rmsprop": optim.RMSprop,
 "sgd": optim.SGD,
 }

 optimizer_class = optimizer_map[self.config.optimizer_type]
 optimizer_kwargs = {
 "lr": self.config.learning_rate,
 "weight_decay": self.config.weight_decay,
 }

 if self.config.optimizer_type in ["adam", "adamw"]:
 optimizer_kwargs.update({"betas": (0.9, 0.999), "eps": 1e-8})
 elif self.config.optimizer_type == "sgd":
 optimizer_kwargs.update({"momentum": 0.9})

 self.optimizer = optimizer_class(self.q_network.parameters, **optimizer_kwargs)

 self.logger.info("Optimizer initialized",
 type=self.config.optimizer_type,
 lr=self.config.learning_rate)

 def _initialize_replay_buffer(self) -> None:
 """Initialization replay buffer."""
 self.replay_buffer = ReplayBuffer(
 capacity=self.config.buffer_size,
 batch_size=self.config.batch_size,
 device=self.device
 )

 self.logger.info("Replay buffer initialized", capacity=self.config.buffer_size)

 def _initialize_epsilon_schedule(self) -> None:
 """Epsilon schedule initialization."""
 self.epsilon_schedule = EpsilonSchedule(
 start_epsilon=self.config.epsilon_start,
 end_epsilon=self.config.epsilon_end,
 decay_rate=self.config.epsilon_decay
 )

 self.logger.info("Epsilon schedule initialized")

 def act(self, state: np.ndarray, training: bool = True) -> int:
 """
 Action selection with epsilon-greedy policy.

 Args:
 state: Current state
 training: Mode training (affects epsilon)

 Returns:
 Selected action
 """
 if training and np.random.random < self.epsilon_schedule.get_epsilon(self.training_step):
 # Random action (exploration)
 action = np.random.randint(0, self.config.network_config.action_size)
 else:
 # Greedy action (exploitation)
 with torch.no_grad:
 state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
 q_values = self.q_network(state_tensor)
 action = q_values.argmax.item

 return action

 def store_experience(self, state: np.ndarray, action: int, reward: float,
 next_state: np.ndarray, done: bool) -> None:
 """
 Saving experience in replay buffer.

 Args:
 state: Current state
 action: Performed action
 reward: Received reward
 next_state: Next state
 done: Episode completion flag
 """
 experience = Experience(state, action, reward, next_state, done)
 self.replay_buffer.push(experience)

 def train_step(self) -> Dict[str, float]:
 """
 One step training.

 Returns:
 Metrics training
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

 # Next Q values from target network
 with torch.no_grad:
 next_q_values = self.target_network(next_states).max(1)[0]
 target_q_values = rewards + (self.config.gamma * next_q_values * ~dones)

 # Compute loss
 loss = self._compute_loss(current_q_values.squeeze, target_q_values)

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

 # Logging
 metrics = {
 "loss": loss.item,
 "grad_norm": grad_norm.item,
 "epsilon": self.epsilon_schedule.get_epsilon(self.training_step),
 "training_step": self.training_step,
 "q_mean": current_q_values.mean.item,
 "target_mean": target_q_values.mean.item,
 }

 self.loss_history.append(loss.item)

 if self.training_step % self.config.log_freq == 0:
 self.logger.info("Training step completed", **metrics)

 return metrics

 def _compute_loss(self, current_q: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
 """
 Computing loss function.

 Args:
 current_q: Current Q-values
 target_q: Target Q-values

 Returns:
 Loss value
 """
 if self.config.loss_type == "mse":
 return F.mse_loss(current_q, target_q)
 elif self.config.loss_type == "huber":
 return F.huber_loss(current_q, target_q, delta=self.config.huber_delta)
 elif self.config.loss_type == "smooth_l1":
 return F.smooth_l1_loss(current_q, target_q)
 else:
 raise ValueError(f"Unknown type loss: {self.config.loss_type}")

 def _update_target_network(self) -> None:
 """Updating target network."""
 if self.config.soft_update_tau is not None:
 # Soft update
 self.q_network.soft_update(self.target_network, self.config.soft_update_tau)
 else:
 # Hard update
 self.q_network.hard_update(self.target_network)

 self.logger.debug("Target network updated", step=self.training_step)

 def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
 """
 Evaluation agent in environment.

 Args:
 env: Environment for testing
 num_episodes: Number of episodes for score

 Returns:
 Metrics performance
 """
 episode_rewards = []
 episode_lengths = []

 for episode in range(num_episodes):
 state = env.reset
 total_reward = 0
 steps = 0
 done = False

 while not done:
 action = self.act(state, training=False) # Greedy policy
 next_state, reward, done, _ = env.step(action)

 total_reward += reward
 steps += 1
 state = next_state

 episode_rewards.append(total_reward)
 episode_lengths.append(steps)

 metrics = {
 "eval_reward_mean": np.mean(episode_rewards),
 "eval_reward_std": np.std(episode_rewards),
 "eval_reward_min": np.min(episode_rewards),
 "eval_reward_max": np.max(episode_rewards),
 "eval_length_mean": np.mean(episode_lengths),
 "eval_episodes": num_episodes,
 }

 self.logger.info("Evaluation completed", **metrics)
 return metrics

 def get_training_stats(self) -> Dict[str, Any]:
 """Get statistics training."""
 return {
 "training_step": self.training_step,
 "epsilon": self.epsilon_schedule.get_epsilon(self.training_step),
 "replay_buffer_size": len(self.replay_buffer),
 "loss_history": self.loss_history[-1000:], # Last 1000 values
 "episode_rewards": self.episode_rewards[-100:], # Last 100 episodes
 "network_stats": self.q_network.get_network_stats,
 }

 def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
 """
 Saving checkpoint agent.

 Args:
 filepath: Path for saving
 metadata: Additional metadata
 """
 checkpoint = {
 "config": self.config.dict,
 "training_step": self.training_step,
 "q_network_state": self.q_network.state_dict,
 "target_network_state": self.target_network.state_dict,
 "optimizer_state": self.optimizer.state_dict,
 "loss_history": self.loss_history,
 "episode_rewards": self.episode_rewards,
 "epsilon_schedule_state": self.epsilon_schedule.get_state,
 "metadata": metadata or {},
 "timestamp": datetime.now.isoformat,
 }

 # Saving replay buffer separately (may be large)
 buffer_filepath = filepath.replace(".pth", "_buffer.pkl")
 with open(buffer_filepath, "wb") as f:
 pickle.dump(self.replay_buffer, f)

 torch.save(checkpoint, filepath)
 self.logger.info("Checkpoint saved", filepath=filepath)

 @classmethod
 def load_checkpoint(cls, filepath: str, load_buffer: bool = True) -> 'DQN':
 """
 Loading agent from checkpoint.

 Args:
 filepath: Path to checkpoint
 load_buffer: Load whether replay buffer

 Returns:
 Loaded agent
 """
 checkpoint = torch.load(filepath, map_location="cpu")

 # Creating agent with saved configuration
 config = DQNConfig(**checkpoint["config"])
 agent = cls(config)

 # Loading states
 agent.training_step = checkpoint["training_step"]
 agent.q_network.load_state_dict(checkpoint["q_network_state"])
 agent.target_network.load_state_dict(checkpoint["target_network_state"])
 agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
 agent.loss_history = checkpoint["loss_history"]
 agent.episode_rewards = checkpoint["episode_rewards"]
 agent.epsilon_schedule.load_state(checkpoint["epsilon_schedule_state"])

 # Loading replay buffer if needed
 if load_buffer:
 buffer_filepath = filepath.replace(".pth", "_buffer.pkl")
 if Path(buffer_filepath).exists:
 with open(buffer_filepath, "rb") as f:
 agent.replay_buffer = pickle.load(f)

 agent.logger.info("Checkpoint loaded", filepath=filepath)
 return agent

 def to(self, device: Union[str, torch.device]) -> 'DQN':
 """Moving agent on other device."""
 self.device = torch.device(device)
 self.q_network = self.q_network.to(self.device)
 self.target_network = self.target_network.to(self.device)
 return self

 def __repr__(self) -> str:
 """String representation agent."""
 return (
 f"DQN(state_size={self.config.network_config.state_size}, "
 f"action_size={self.config.network_config.action_size}, "
 f"training_step={self.training_step}, "
 f"device={self.device})"
 )