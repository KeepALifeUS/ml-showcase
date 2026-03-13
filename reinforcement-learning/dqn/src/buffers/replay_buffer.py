"""
Experience Replay Buffer for DQN with enterprise patterns.

Implements efficient circular buffer for storing experience transitions:
- Memory-efficient circular buffer implementation
- Thread-safe operations for production use
- Batch sampling with configurable batch size
- Optional data preprocessing and normalization
- Comprehensive monitoring and statistics
- Efficient memory management with automatic cleanup
"""

import logging
from typing import List, Optional, Union, Any, Dict, Tuple
from collections import deque, namedtuple
import numpy as np
import torch
import random
import threading
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import pickle
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

# Experience structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBufferConfig(BaseModel):
 """Replay Buffer configuration with validation."""

 capacity: int = Field(default=100000, description="Maximum buffer size", gt=0, le=1000000)
 batch_size: int = Field(default=64, description="Batch size for sampling", gt=0, le=1024)
 min_size: int = Field(default=1000, description="Minimum size before sampling", gt=0)

 # Memory management
 auto_cleanup: bool = Field(default=True, description="Automatic memory cleanup")
 cleanup_threshold: float = Field(default=0.9, description="Cleanup threshold", ge=0.5, le=1.0)

 # Data preprocessing
 normalize_states: bool = Field(default=False, description="State normalization")
 normalize_rewards: bool = Field(default=False, description="Reward normalization")
 reward_scaling: float = Field(default=1.0, description="Reward scaling", gt=0)

 # Performance
 pin_memory: bool = Field(default=True, description="Pin memory for GPU transfer")
 prefetch_batches: int = Field(default=0, description="Number of prefetch batches", ge=0, le=10)

 # Threading
 thread_safe: bool = Field(default=True, description="Thread-safe operations")

 @validator("min_size")
 def validate_min_size(cls, v, values):
 if "batch_size" in values and v < values["batch_size"]:
 raise ValueError("min_size must be >= batch_size")
 if "capacity" in values and v > values["capacity"]:
 raise ValueError("min_size must be <= capacity")
 return v


@dataclass
class BufferStatistics:
 """Replay Buffer statistics."""
 size: int
 capacity: int
 utilization: float
 total_added: int
 total_sampled: int
 avg_reward: float
 reward_std: float
 memory_usage_mb: float


class ReplayBuffer:
 """
 Experience Replay Buffer with enterprise-grade functionality.

 Features:
 - Efficient circular buffer with O(1) operations
 - Thread-safe operations for concurrent access
 - Batch sampling with optional preprocessing
 - Memory management and automatic cleanup
 - Comprehensive statistics and monitoring
 - GPU-optimized data transfer
 - Configurable normalization and scaling
 - Persistence support for checkpointing
 """

 def __init__(self,
 capacity: Optional[int] = None,
 batch_size: Optional[int] = None,
 config: Optional[ReplayBufferConfig] = None,
 device: Union[str, torch.device] = "cpu"):
 """
 Replay Buffer initialization.

 Args:
 capacity: Maximum buffer size (deprecated, use config)
 batch_size: Batch size (deprecated, use config)
 config: Full buffer configuration
 device: Device for tensor operations
 """
 # Backward compatibility
 if config is None:
 config = ReplayBufferConfig(
 capacity=capacity or 100000,
 batch_size=batch_size or 64
 )

 self.config = config
 self.device = torch.device(device)

 # Initialize storage
 self.buffer = deque(maxlen=config.capacity)
 self.position = 0
 self.size = 0
 self.total_added = 0
 self.total_sampled = 0

 # Threading support
 if config.thread_safe:
 self._lock = threading.RLock
 else:
 self._lock = None

 # Statistics tracking
 self.reward_sum = 0.0
 self.reward_sq_sum = 0.0

 # Normalization statistics
 self.state_mean = None
 self.state_std = None
 self.reward_mean = 0.0
 self.reward_std = 1.0

 # Prefetching
 self._prefetch_cache = []

 self.logger = structlog.get_logger(__name__).bind(
 component="ReplayBuffer",
 capacity=config.capacity,
 device=str(self.device)
 )

 self.logger.info("Replay Buffer initialized", config=config.dict)

 def _safe_operation(self, func, *args, **kwargs):
 """Thread-safe wrapper for operations."""
 if self._lock is not None:
 with self._lock:
 return func(*args, **kwargs)
 else:
 return func(*args, **kwargs)

 def push(self,
 state: np.ndarray,
 action: int,
 reward: float,
 next_state: np.ndarray,
 done: bool) -> None:
 """
 Adding experience to buffer.

 Args:
 state: Current state
 action: Performed action
 reward: Received reward
 next_state: Next state
 done: Episode completion flag
 """
 def _push:
 # Creating experience
 experience = Experience(
 state=state.copy,
 action=action,
 reward=reward,
 next_state=next_state.copy,
 done=done
 )

 # Adding in buffer
 self.buffer.append(experience)

 # Updating statistics
 self.total_added += 1
 self.reward_sum += reward
 self.reward_sq_sum += reward ** 2

 # Update size
 self.size = len(self.buffer)

 # Update normalization
 if self.config.normalize_states:
 self._update_state_normalization(state)

 if self.config.normalize_rewards:
 self._update_reward_normalization

 # Automatic cleanup if needed
 if (self.config.auto_cleanup and
 self.size / self.config.capacity > self.config.cleanup_threshold):
 self._cleanup_memory

 self._safe_operation(_push)

 if self.total_added % 10000 == 0:
 self.logger.debug("Buffer status",
 size=self.size,
 utilization=self.size/self.config.capacity)

 def sample(self, batch_size: Optional[int] = None) -> List[Experience]:
 """
 Sampling random batch from buffer.

 Args:
 batch_size: Batch size (defaults to config value)

 Returns:
 List of experiences
 """
 def _sample:
 if self.size < self.config.min_size:
 raise ValueError(f"Insufficient data in buffer: {self.size} < {self.config.min_size}")

 effective_batch_size = batch_size or self.config.batch_size

 # Uniform sampling
 batch_indices = random.sample(range(self.size), effective_batch_size)
 batch = [self.buffer[i] for i in batch_indices]

 # Preprocessing if enabled
 if self.config.normalize_states or self.config.normalize_rewards:
 batch = self._preprocess_batch(batch)

 self.total_sampled += effective_batch_size

 return batch

 return self._safe_operation(_sample)

 def sample_tensors(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
 """
 Sampling batch and conversion to tensors.

 Args:
 batch_size: Batch size

 Returns:
 Tuple of (states, actions, rewards, next_states, dones)
 """
 batch = self.sample(batch_size)

 # Conversion to numpy arrays
 states = np.array([e.state for e in batch])
 actions = np.array([e.action for e in batch])
 rewards = np.array([e.reward for e in batch])
 next_states = np.array([e.next_state for e in batch])
 dones = np.array([e.done for e in batch])

 # Conversion to tensors
 states_tensor = torch.FloatTensor(states).to(self.device)
 actions_tensor = torch.LongTensor(actions).to(self.device)
 rewards_tensor = torch.FloatTensor(rewards).to(self.device)
 next_states_tensor = torch.FloatTensor(next_states).to(self.device)
 dones_tensor = torch.BoolTensor(dones).to(self.device)

 # Pin memory for faster GPU transfer if enabled
 if self.config.pin_memory and self.device.type == 'cuda':
 states_tensor = states_tensor.pin_memory
 actions_tensor = actions_tensor.pin_memory
 rewards_tensor = rewards_tensor.pin_memory
 next_states_tensor = next_states_tensor.pin_memory
 dones_tensor = dones_tensor.pin_memory

 return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor

 def _preprocess_batch(self, batch: List[Experience]) -> List[Experience]:
 """
 Preprocessing batch data.

 Args:
 batch: Original batch

 Returns:
 Processed batch
 """
 processed_batch = []

 for experience in batch:
 state = experience.state
 next_state = experience.next_state
 reward = experience.reward

 # State normalization
 if self.config.normalize_states and self.state_std is not None:
 state = (state - self.state_mean) / (self.state_std + 1e-8)
 next_state = (next_state - self.state_mean) / (self.state_std + 1e-8)

 # Reward normalization
 if self.config.normalize_rewards:
 reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)

 # Reward scaling
 if self.config.reward_scaling != 1.0:
 reward = reward * self.config.reward_scaling

 processed_experience = Experience(
 state=state,
 action=experience.action,
 reward=reward,
 next_state=next_state,
 done=experience.done
 )
 processed_batch.append(processed_experience)

 return processed_batch

 def _update_state_normalization(self, state: np.ndarray) -> None:
 """Update state normalization statistics."""
 if self.state_mean is None:
 self.state_mean = state.copy
 self.state_std = np.zeros_like(state)
 else:
 # Running mean and std
 alpha = 1.0 / min(1000, self.total_added) # Decay factor
 delta = state - self.state_mean
 self.state_mean += alpha * delta
 self.state_std = (1 - alpha) * self.state_std + alpha * (delta ** 2)

 def _update_reward_normalization(self) -> None:
 """Update reward normalization statistics."""
 if self.total_added > 1:
 self.reward_mean = self.reward_sum / self.total_added
 self.reward_std = np.sqrt(max(
 self.reward_sq_sum / self.total_added - self.reward_mean ** 2,
 1e-8
 ))

 def _cleanup_memory(self) -> None:
 """Memory cleanup on overflow."""
 # Forced garbage collection may help
 import gc
 gc.collect

 self.logger.debug("Memory cleanup completed")

 def clear(self) -> None:
 """Full buffer cleanup."""
 def _clear:
 self.buffer.clear
 self.size = 0
 self.position = 0
 self.reward_sum = 0.0
 self.reward_sq_sum = 0.0
 self.state_mean = None
 self.state_std = None
 self.reward_mean = 0.0
 self.reward_std = 1.0

 self._safe_operation(_clear)
 self.logger.info("Buffer cleared")

 def get_statistics(self) -> BufferStatistics:
 """
 Get buffer statistics.

 Returns:
 Buffer statistics
 """
 def _get_stats:
 avg_reward = self.reward_mean if self.total_added > 0 else 0.0
 reward_std = self.reward_std if self.total_added > 1 else 0.0

 # Approximate estimate of memory usage
 memory_usage = 0.0
 if self.size > 0:
 sample_experience = self.buffer[0]
 state_size = sample_experience.state.nbytes
 next_state_size = sample_experience.next_state.nbytes
 experience_size = state_size + next_state_size + 16 # action, reward, done
 memory_usage = (experience_size * self.size) / (1024 * 1024) # MB

 return BufferStatistics(
 size=self.size,
 capacity=self.config.capacity,
 utilization=self.size / self.config.capacity,
 total_added=self.total_added,
 total_sampled=self.total_sampled,
 avg_reward=avg_reward,
 reward_std=reward_std,
 memory_usage_mb=memory_usage
 )

 return self._safe_operation(_get_stats)

 def save(self, filepath: str) -> None:
 """
 Save buffer to file.

 Args:
 filepath: Path for saving
 """
 def _save:
 save_data = {
 'config': self.config.dict,
 'buffer': list(self.buffer),
 'position': self.position,
 'size': self.size,
 'total_added': self.total_added,
 'total_sampled': self.total_sampled,
 'reward_sum': self.reward_sum,
 'reward_sq_sum': self.reward_sq_sum,
 'state_mean': self.state_mean,
 'state_std': self.state_std,
 'reward_mean': self.reward_mean,
 'reward_std': self.reward_std,
 }

 with open(filepath, 'wb') as f:
 pickle.dump(save_data, f)

 self._safe_operation(_save)
 self.logger.info("Buffer saved", filepath=filepath)

 @classmethod
 def load(cls, filepath: str, device: Union[str, torch.device] = "cpu") -> 'ReplayBuffer':
 """
 Load buffer from file.

 Args:
 filepath: Path to file
 device: Device for operations

 Returns:
 Loaded buffer
 """
 with open(filepath, 'rb') as f:
 save_data = pickle.load(f)

 # Creating buffer with saved configuration
 config = ReplayBufferConfig(**save_data['config'])
 buffer = cls(config=config, device=device)

 # Restoration state
 buffer.buffer = deque(save_data['buffer'], maxlen=config.capacity)
 buffer.position = save_data['position']
 buffer.size = save_data['size']
 buffer.total_added = save_data['total_added']
 buffer.total_sampled = save_data['total_sampled']
 buffer.reward_sum = save_data['reward_sum']
 buffer.reward_sq_sum = save_data['reward_sq_sum']
 buffer.state_mean = save_data['state_mean']
 buffer.state_std = save_data['state_std']
 buffer.reward_mean = save_data['reward_mean']
 buffer.reward_std = save_data['reward_std']

 buffer.logger.info("Buffer loaded", filepath=filepath, size=buffer.size)
 return buffer

 def __len__(self) -> int:
 """Buffer size."""
 return self.size

 def __repr__(self) -> str:
 """String representation buffer."""
 return (
 f"ReplayBuffer(size={self.size}, capacity={self.config.capacity}, "
 f"utilization={self.size/self.config.capacity:.1%})"
 )