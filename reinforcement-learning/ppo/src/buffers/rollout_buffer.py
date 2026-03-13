"""
Rollout Buffer Implementation for PPO
for efficient data storage and sampling

Rollout buffer stores trajectories from policy rollouts:
- States, actions, rewards, values, log_probs
- Efficient memory management
- Mini-batch sampling
- GAE computation integration
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Generator
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import warnings

from ..advantages.gae import GAE, GAEConfig
from ..utils.normalization import normalize_advantages


@dataclass
class RolloutBufferConfig:
 """Configuration for rollout buffer"""
 
 # Buffer size
 buffer_size: int = 2048 # Number of steps to collect
 batch_size: int = 64 # Mini-batch size for training
 
 # Data dimensions
 observation_space: tuple = (64,) # Observation space shape
 action_space: tuple = (4,) # Action space shape
 
 # GAE configuration
 gae_config: Optional[GAEConfig] = None
 
 # Buffer management
 overlapping_buffers: bool = False # Allow overlapping data
 auto_reset: bool = True # Auto reset when full
 
 # Memory optimization
 use_shared_memory: bool = False # Shared memory for multiprocessing
 pin_memory: bool = True # Pin memory for GPU transfers
 device: str = "cpu" # Storage device
 
 # Advanced features
 priority_sampling: bool = False # Priority-based sampling
 recurrent_buffer: bool = False # Support for recurrent policies
 sequence_length: int = 1 # Sequence length for RNNs
 
 def __post_init__(self):
 if self.gae_config is None:
 self.gae_config = GAEConfig()


class RolloutBuffer:
 """
 Rollout buffer for PPO training
 
 Efficiently stores and manages trajectory data:
 - Observations, actions, rewards, values
 - Log probabilities, advantages, returns
 - Mini-batch sampling with shuffling
 - GAE computation integration
 """
 
 def __init__(self, config: RolloutBufferConfig):
 self.config = config
 self.buffer_size = config.buffer_size
 self.batch_size = config.batch_size
 self.device = config.device
 
 # Initialize GAE
 self.gae_estimator = GAE(config.gae_config) if config.gae_config else None
 
 # Buffer storage
 self.observations = torch.zeros(
 (self.buffer_size,) + config.observation_space,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.actions = torch.zeros(
 (self.buffer_size,) + config.action_space,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.rewards = torch.zeros(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.values = torch.zeros(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.log_probs = torch.zeros(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.dones = torch.zeros(
 self.buffer_size,
 dtype=torch.bool,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 # Computed later
 self.advantages = torch.zeros(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 self.returns = torch.zeros(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device,
 pin_memory=config.pin_memory
 )
 
 # Buffer state
 self.pos = 0
 self.full = False
 self.advantages_computed = False
 
 # Priority sampling
 if config.priority_sampling:
 self.priorities = torch.ones(
 self.buffer_size,
 dtype=torch.float32,
 device=config.device
 )
 
 # Recurrent support
 if config.recurrent_buffer:
 self.episode_starts = torch.zeros(
 self.buffer_size,
 dtype=torch.bool,
 device=config.device
 )
 
 def add(
 self,
 obs: torch.Tensor,
 action: torch.Tensor,
 reward: float,
 value: float,
 log_prob: float,
 done: bool,
 **kwargs
 ):
 """
 Add single step to buffer
 
 Args:
 obs: Observation [obs_dim]
 action: Action taken [action_dim]
 reward: Reward received
 value: Value estimate
 log_prob: Log probability of action
 done: Episode termination flag
 """
 
 if self.full and not self.config.overlapping_buffers:
 warnings.warn("Buffer is full and overlapping is disabled")
 return
 
 # Store data
 self.observations[self.pos] = obs
 self.actions[self.pos] = action
 self.rewards[self.pos] = reward
 self.values[self.pos] = value
 self.log_probs[self.pos] = log_prob
 self.dones[self.pos] = done
 
 # Priority sampling
 if self.config.priority_sampling:
 # Initialize with high priority for new experiences
 self.priorities[self.pos] = 1.0
 
 # Recurrent support
 if self.config.recurrent_buffer:
 episode_start = kwargs.get('episode_start', False)
 self.episode_starts[self.pos] = episode_start
 
 # Update position
 self.pos += 1
 if self.pos >= self.buffer_size:
 self.full = True
 if self.config.auto_reset:
 self.pos = 0
 
 # Reset advantages computation flag
 self.advantages_computed = False
 
 def add_batch(
 self,
 obs_batch: torch.Tensor,
 actions_batch: torch.Tensor,
 rewards_batch: torch.Tensor,
 values_batch: torch.Tensor,
 log_probs_batch: torch.Tensor,
 dones_batch: torch.Tensor
 ):
 """
 Add batch of experiences to buffer
 
 Args:
 obs_batch: Observations [batch_size, obs_dim]
 actions_batch: Actions [batch_size, action_dim]
 rewards_batch: Rewards [batch_size]
 values_batch: Values [batch_size]
 log_probs_batch: Log probabilities [batch_size]
 dones_batch: Done flags [batch_size]
 """
 
 batch_size = obs_batch.shape[0]
 
 # Check if enough space
 if self.pos + batch_size > self.buffer_size:
 if not self.config.overlapping_buffers:
 warnings.warn("Not enough space in buffer for batch")
 return
 else:
 # Wrap around
 remaining = self.buffer_size - self.pos
 self.add_batch(
 obs_batch[:remaining],
 actions_batch[:remaining],
 rewards_batch[:remaining],
 values_batch[:remaining],
 log_probs_batch[:remaining],
 dones_batch[:remaining]
 )
 
 self.pos = 0
 self.full = True
 
 if batch_size > remaining:
 self.add_batch(
 obs_batch[remaining:],
 actions_batch[remaining:],
 rewards_batch[remaining:],
 values_batch[remaining:],
 log_probs_batch[remaining:],
 dones_batch[remaining:]
 )
 return
 
 # Store batch
 end_pos = self.pos + batch_size
 self.observations[self.pos:end_pos] = obs_batch
 self.actions[self.pos:end_pos] = actions_batch
 self.rewards[self.pos:end_pos] = rewards_batch
 self.values[self.pos:end_pos] = values_batch
 self.log_probs[self.pos:end_pos] = log_probs_batch
 self.dones[self.pos:end_pos] = dones_batch
 
 # Update position
 self.pos = end_pos
 if self.pos >= self.buffer_size:
 self.full = True
 if self.config.auto_reset:
 self.pos = 0
 
 self.advantages_computed = False
 
 def compute_returns_and_advantages(
 self,
 last_values: Optional[torch.Tensor] = None
 ):
 """
 Compute returns and advantages using GAE
 
 Args:
 last_values: Value estimates for last observations [batch_size]
 Needed if trajectories are not complete
 """
 
 if not self.gae_estimator:
 raise ValueError("GAE estimator not configured")
 
 # Get valid data range
 if self.full:
 data_size = self.buffer_size
 else:
 data_size = self.pos
 
 if data_size == 0:
 return
 
 # Reshape data for GAE computation [seq_len, batch_size]
 rewards = self.rewards[:data_size].unsqueeze(1) # [seq_len, 1]
 values = self.values[:data_size].unsqueeze(1) # [seq_len, 1]
 dones = self.dones[:data_size].unsqueeze(1) # [seq_len, 1]
 
 # Compute advantages and returns
 advantages, returns = self.gae_estimator.compute_advantages_and_returns(
 rewards=rewards,
 values=values,
 dones=dones,
 next_values=last_values
 )
 
 # Store computed values
 self.advantages[:data_size] = advantages.squeeze(1)
 self.returns[:data_size] = returns.squeeze(1)
 
 self.advantages_computed = True
 
 def get(self) -> Dict[str, torch.Tensor]:
 """
 Get all buffer data
 
 Returns:
 Dictionary with all stored data
 """
 
 if not self.advantages_computed:
 warnings.warn("Advantages not computed. Call compute_returns_and_advantages first.")
 
 # Get valid data range
 if self.full:
 data_size = self.buffer_size
 else:
 data_size = self.pos
 
 if data_size == 0:
 return {}
 
 data = {
 "observations": self.observations[:data_size],
 "actions": self.actions[:data_size],
 "rewards": self.rewards[:data_size],
 "values": self.values[:data_size],
 "log_probs": self.log_probs[:data_size],
 "dones": self.dones[:data_size],
 "advantages": self.advantages[:data_size],
 "returns": self.returns[:data_size]
 }
 
 # Add priority sampling data
 if self.config.priority_sampling:
 data["priorities"] = self.priorities[:data_size]
 
 # Add recurrent data
 if self.config.recurrent_buffer:
 data["episode_starts"] = self.episode_starts[:data_size]
 
 return data
 
 def get_minibatch_indices(self, batch_size: Optional[int] = None) -> Generator[np.ndarray, None, None]:
 """
 Generate mini-batch indices for training
 
 Args:
 batch_size: Size of mini-batches (uses config.batch_size if None)
 
 Yields:
 Array of indices for each mini-batch
 """
 
 if batch_size is None:
 batch_size = self.batch_size
 
 # Get valid data size
 if self.full:
 data_size = self.buffer_size
 else:
 data_size = self.pos
 
 if data_size == 0:
 return
 
 # Generate shuffled indices
 if self.config.priority_sampling:
 # Priority-based sampling
 indices = self._priority_sample_indices(data_size, batch_size)
 else:
 # Uniform sampling
 indices = np.random.permutation(data_size)
 
 # Yield mini-batches
 for start_idx in range(0, len(indices), batch_size):
 end_idx = min(start_idx + batch_size, len(indices))
 yield indices[start_idx:end_idx]
 
 def _priority_sample_indices(self, data_size: int, total_samples: int) -> np.ndarray:
 """Sample indices based on priorities"""
 
 # Get priorities
 priorities = self.priorities[:data_size].cpu().numpy()
 
 # Compute sampling probabilities
 priorities = np.maximum(priorities, 1e-8) # Avoid zeros
 probs = priorities / priorities.sum()
 
 # Sample indices
 indices = np.random.choice(
 data_size,
 size=total_samples,
 p=probs,
 replace=True
 )
 
 return indices
 
 def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
 """Update priorities for priority sampling"""
 
 if not self.config.priority_sampling:
 return
 
 # Clamp priorities
 priorities = torch.clamp(priorities, min=1e-8, max=1e8)
 
 # Update
 self.priorities[indices] = priorities
 
 def reset(self):
 """Reset buffer to empty state"""
 
 self.pos = 0
 self.full = False
 self.advantages_computed = False
 
 # Reset data (optional, for memory efficiency)
 if hasattr(self, 'reset_data') and self.reset_data:
 self.observations.zero_()
 self.actions.zero_()
 self.rewards.zero_()
 self.values.zero_()
 self.log_probs.zero_()
 self.dones.zero_()
 self.advantages.zero_()
 self.returns.zero_()
 
 def size(self) -> int:
 """Get current buffer size"""
 return self.buffer_size if self.full else self.pos
 
 def is_full(self) -> bool:
 """Check if buffer is full"""
 return self.full
 
 def get_statistics(self) -> Dict[str, float]:
 """Get buffer statistics"""
 
 data_size = self.size()
 
 if data_size == 0:
 return {}
 
 stats = {
 "size": data_size,
 "is_full": self.full,
 "reward_mean": self.rewards[:data_size].mean().item(),
 "reward_std": self.rewards[:data_size].std().item(),
 "value_mean": self.values[:data_size].mean().item(),
 "value_std": self.values[:data_size].std().item(),
 "episode_ends": self.dones[:data_size].sum().item()
 }
 
 if self.advantages_computed:
 stats.update({
 "advantage_mean": self.advantages[:data_size].mean().item(),
 "advantage_std": self.advantages[:data_size].std().item(),
 "return_mean": self.returns[:data_size].mean().item(),
 "return_std": self.returns[:data_size].std().item()
 })
 
 return stats


class RecurrentRolloutBuffer(RolloutBuffer):
 """
 Specialized rollout buffer for recurrent policies
 
 Handles:
 - Episode boundaries
 - Hidden state management 
 - Sequence-based sampling
 """
 
 def __init__(self, config: RolloutBufferConfig):
 config.recurrent_buffer = True
 super().__init__(config)
 
 self.sequence_length = config.sequence_length
 
 # Episode tracking
 self.episode_lengths = []
 self.current_episode_length = 0
 
 def add(self, *args, episode_start: bool = False, **kwargs):
 """Add step with episode start tracking"""
 
 super().add(*args, episode_start=episode_start, **kwargs)
 
 # Track episode length
 if episode_start:
 if self.current_episode_length > 0:
 self.episode_lengths.append(self.current_episode_length)
 self.current_episode_length = 1
 else:
 self.current_episode_length += 1
 
 def get_sequences(self, sequence_length: Optional[int] = None) -> Generator[Dict[str, torch.Tensor], None, None]:
 """
 Generate sequences for recurrent training
 
 Args:
 sequence_length: Length of sequences (uses config if None)
 
 Yields:
 Dictionary with sequence data
 """
 
 if sequence_length is None:
 sequence_length = self.sequence_length
 
 data = self.get()
 data_size = self.size()
 
 if data_size < sequence_length:
 return
 
 # Find episode boundaries
 episode_starts = data["episode_starts"]
 
 # Generate sequences
 for start_idx in range(0, data_size - sequence_length + 1):
 # Check if sequence crosses episode boundary
 if episode_starts[start_idx + 1:start_idx + sequence_length].any():
 continue # Skip sequences that cross episodes
 
 # Extract sequence
 sequence_data = {}
 for key, tensor in data.items():
 if key != "episode_starts":
 sequence_data[key] = tensor[start_idx:start_idx + sequence_length]
 
 yield sequence_data


# Factory function for creating buffers
def create_rollout_buffer(
 buffer_type: str = "standard",
 config: Optional[RolloutBufferConfig] = None,
 **kwargs
) -> Union[RolloutBuffer, RecurrentRolloutBuffer]:
 """Create rollout buffer of specified type"""
 
 if config is None:
 config = RolloutBufferConfig()
 
 # Update config with kwargs
 for key, value in kwargs.items():
 if hasattr(config, key):
 setattr(config, key, value)
 
 if buffer_type == "standard":
 return RolloutBuffer(config)
 elif buffer_type == "recurrent":
 return RecurrentRolloutBuffer(config)
 else:
 raise ValueError(f"Unknown buffer type: {buffer_type}")


# Export classes and functions
__all__ = [
 "RolloutBufferConfig",
 "RolloutBuffer",
 "RecurrentRolloutBuffer",
 "create_rollout_buffer"
]