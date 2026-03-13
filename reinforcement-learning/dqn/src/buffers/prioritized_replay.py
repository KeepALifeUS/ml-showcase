"""
Prioritized Experience Replay Buffer with enterprise patterns.

Implements PER (Prioritized Experience Replay) with sum-tree for efficient sampling:
- Sum-tree data structure for O(log n) sampling
- TD-error based priorities for better learning
- Importance sampling weights for bias correction
- Configurable alpha and beta parameters
- Production-ready thread safety
- Memory-efficient implementation
- Comprehensive monitoring
"""

import logging
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np
import torch
import threading
import random
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import structlog
from collections import namedtuple

from .replay_buffer import Experience, ReplayBuffer, ReplayBufferConfig

logger = structlog.get_logger(__name__)


class PrioritizedReplayConfig(ReplayBufferConfig):
 """Prioritized Experience Replay configuration with validation."""

 # PER specific parameters
 alpha: float = Field(default=0.6, description="Priority exponent", ge=0, le=1.0)
 beta_start: float = Field(default=0.4, description="Initial importance sampling exponent", ge=0, le=1.0)
 beta_end: float = Field(default=1.0, description="Final importance sampling exponent", ge=0, le=1.0)
 beta_frames: int = Field(default=100000, description="Frames to anneal beta", gt=0)

 # Priority management
 max_priority: float = Field(default=1.0, description="Maximum priority value", gt=0)
 min_priority: float = Field(default=1e-6, description="Minimum priority value", gt=0, le=1e-3)
 priority_epsilon: float = Field(default=1e-6, description="Small epsilon for numerical stability", gt=0)

 # Performance optimization
 tree_auto_rebalance: bool = Field(default=True, description="Auto rebalance sum tree")
 rebalance_threshold: int = Field(default=10000, description="Operations before rebalance", gt=0)

 @validator("beta_start")
 def validate_beta_start(cls, v, values):
 if "beta_end" in values and v > values["beta_end"]:
 raise ValueError("beta_start must be <= beta_end")
 return v

 @validator("min_priority")
 def validate_min_priority(cls, v, values):
 if "max_priority" in values and v >= values["max_priority"]:
 raise ValueError("min_priority must be < max_priority")
 return v


@dataclass
class PrioritizedExperience:
 """Experience with priority information."""
 experience: Experience
 priority: float
 tree_index: int


class SumTree:
 """
 Sum Tree data structure for efficient prioritized sampling.

 Binary tree where:
 - Leaves contain priorities
 - Internal nodes contain sums of children
 - Root contains the total sum of all priorities

 Operations:
 - Update: O(log n)
 - Sample: O(log n)
 - Total sum: O(1)
 """

 def __init__(self, capacity: int):
 """
 Initialization Sum Tree.

 Args:
 capacity: Maximum number of elements
 """
 self.capacity = capacity
 self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
 self.data = np.full(capacity, None)
 self.data_pointer = 0
 self.size = 0

 # Statistics for monitoring
 self.total_updates = 0
 self.total_samples = 0

 self.logger = structlog.get_logger(__name__).bind(component="SumTree")
 self.logger.debug("SumTree initialized", capacity=capacity)

 def add(self, priority: float, data: Any) -> int:
 """
 Adding element with priority.

 Args:
 priority: Priority value
 data: Data to store

 Returns:
 Tree index of added element
 """
 tree_index = self.data_pointer + self.capacity - 1

 # Save data
 self.data[self.data_pointer] = data

 # Update tree with new priority
 self.update(tree_index, priority)

 # Circular pointer
 self.data_pointer = (self.data_pointer + 1) % self.capacity

 # Increase size up to maximum
 if self.size < self.capacity:
 self.size += 1

 return tree_index

 def update(self, tree_index: int, priority: float) -> None:
 """
 Update element priority.

 Args:
 tree_index: Index in tree
 priority: New priority
 """
 change = priority - self.tree[tree_index]
 self.tree[tree_index] = priority

 # Update all parent nodes
 while tree_index != 0:
 tree_index = (tree_index - 1) // 2
 self.tree[tree_index] += change

 self.total_updates += 1

 def get_leaf(self, value: float) -> Tuple[int, float, Any]:
 """
 Get leaf node by cumulative value.

 Args:
 value: Cumulative value for search

 Returns:
 Tuple of (leaf_index, priority, data)
 """
 parent_index = 0

 while True:
 left_child_index = 2 * parent_index + 1
 right_child_index = left_child_index + 1

 # If reached leaf
 if left_child_index >= len(self.tree):
 leaf_index = parent_index
 break

 # Choose left or right child
 if value <= self.tree[left_child_index]:
 parent_index = left_child_index
 else:
 value -= self.tree[left_child_index]
 parent_index = right_child_index

 data_index = leaf_index - self.capacity + 1
 priority = self.tree[leaf_index]
 data = self.data[data_index]

 self.total_samples += 1

 return leaf_index, priority, data

 def total_priority(self) -> float:
 """Get total sum of priorities."""
 return self.tree[0]

 def max_priority(self) -> float:
 """Get maximum priority."""
 if self.size == 0:
 return 1.0

 # Maximum priority among used leaves
 start_idx = self.capacity - 1
 end_idx = start_idx + self.size
 return np.max(self.tree[start_idx:end_idx])

 def get_statistics(self) -> Dict[str, Any]:
 """Get Sum Tree statistics."""
 return {
 "capacity": self.capacity,
 "size": self.size,
 "total_priority": self.total_priority,
 "max_priority": self.max_priority,
 "avg_priority": self.total_priority / max(self.size, 1),
 "total_updates": self.total_updates,
 "total_samples": self.total_samples,
 "utilization": self.size / self.capacity,
 }


class PrioritizedReplayBuffer(ReplayBuffer):
 """
 Prioritized Experience Replay Buffer with enterprise functionality.

 Features:
 - Sum-tree based prioritized sampling for efficient O(log n) operations
 - TD-error based priorities for better learning efficiency
 - Importance sampling weights for bias correction
 - Configurable alpha/beta parameters with annealing
 - Thread-safe operations for production use
 - Memory-efficient implementation with auto-rebalancing
 - Comprehensive monitoring and statistics
 - Adaptive priority management
 """

 def __init__(self,
 capacity: Optional[int] = None,
 batch_size: Optional[int] = None,
 config: Optional[PrioritizedReplayConfig] = None,
 device: Union[str, torch.device] = "cpu"):
 """
 Prioritized Replay Buffer initialization.

 Args:
 capacity: Maximum buffer size (deprecated)
 batch_size: Batch size (deprecated)
 config: Full buffer configuration
 device: Device for tensor operations
 """
 # Backward compatibility
 if config is None:
 config = PrioritizedReplayConfig(
 capacity=capacity or 100000,
 batch_size=batch_size or 64
 )

 super.__init__(config=config, device=device)

 self.per_config = config

 # Initialize sum tree
 self.sum_tree = SumTree(config.capacity)

 # Priority management
 self.max_priority = config.max_priority
 self.priority_updates = 0

 # Beta annealing for importance sampling
 self.beta_frames_completed = 0

 # Performance tracking
 self.rebalance_operations = 0

 self.logger = structlog.get_logger(__name__).bind(
 component="PrioritizedReplayBuffer",
 capacity=config.capacity,
 alpha=config.alpha,
 device=str(self.device)
 )

 self.logger.info("Prioritized Replay Buffer initialized")

 def get_beta(self) -> float:
 """
 Get current beta for importance sampling.

 Returns:
 Current beta value
 """
 if self.beta_frames_completed >= self.per_config.beta_frames:
 return self.per_config.beta_end

 # Linear annealing
 progress = self.beta_frames_completed / self.per_config.beta_frames
 beta = self.per_config.beta_start + progress * (
 self.per_config.beta_end - self.per_config.beta_start
 )

 return beta

 def push(self,
 state: np.ndarray,
 action: int,
 reward: float,
 next_state: np.ndarray,
 done: bool,
 priority: Optional[float] = None) -> None:
 """
 Add experience with priority.

 Args:
 state: Current state
 action: Performed action
 reward: Received reward
 next_state: Next state
 done: Episode completion flag
 priority: Priority for experience (if None, max is used)
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

 # Determining priority
 if priority is None:
 priority = self.max_priority

 # Clamp priority to valid range
 priority = np.clip(priority, self.per_config.min_priority, self.per_config.max_priority)

 # Adding to sum tree
 tree_index = self.sum_tree.add(priority, experience)

 # Updating base statistics
 self.total_added += 1
 self.reward_sum += reward
 self.reward_sq_sum += reward ** 2
 self.size = min(self.total_added, self.per_config.capacity)

 # Update maximum priority
 if priority > self.max_priority:
 self.max_priority = priority

 # Auto-rebalancing if enabled
 if (self.per_config.tree_auto_rebalance and
 self.rebalance_operations % self.per_config.rebalance_threshold == 0):
 self._rebalance_tree

 self.rebalance_operations += 1

 self._safe_operation(_push)

 def sample(self, batch_size: Optional[int] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
 """
 Prioritized sampling from buffer.

 Args:
 batch_size: Batch size

 Returns:
 Tuple of (experiences, importance_weights, tree_indices)
 """
 def _sample:
 if self.size < self.per_config.min_size:
 raise ValueError(f"Insufficient data: {self.size} < {self.per_config.min_size}")

 effective_batch_size = batch_size or self.per_config.batch_size

 # Sampling with priorities
 experiences = []
 tree_indices = []
 priorities = []

 total_priority = self.sum_tree.total_priority
 segment_size = total_priority / effective_batch_size

 for i in range(effective_batch_size):
 # Uniform sampling in each segment
 segment_start = segment_size * i
 segment_end = segment_start + segment_size
 sample_value = np.random.uniform(segment_start, segment_end)

 # Getting experience from sum tree
 tree_index, priority, experience = self.sum_tree.get_leaf(sample_value)

 experiences.append(experience)
 tree_indices.append(tree_index)
 priorities.append(priority)

 # Computing importance sampling weights
 priorities = np.array(priorities, dtype=np.float32)
 sampling_probs = priorities / total_priority

 # IS weights with current beta
 beta = self.get_beta
 is_weights = (self.size * sampling_probs) ** (-beta)
 is_weights = is_weights / np.max(is_weights) # Normalize

 # Updating statistics
 self.total_sampled += effective_batch_size
 self.beta_frames_completed += effective_batch_size

 return experiences, is_weights, np.array(tree_indices)

 return self._safe_operation(_sample)

 def sample_tensors(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, ...]:
 """
 Prioritized sampling with tensor conversion.

 Args:
 batch_size: Batch size

 Returns:
 Tuple of (states, actions, rewards, next_states, dones, is_weights, tree_indices)
 """
 experiences, is_weights, tree_indices = self.sample(batch_size)

 # Convert experiences to tensors (as in basic class)
 states = np.array([e.state for e in experiences])
 actions = np.array([e.action for e in experiences])
 rewards = np.array([e.reward for e in experiences])
 next_states = np.array([e.next_state for e in experiences])
 dones = np.array([e.done for e in experiences])

 # Create tensors
 states_tensor = torch.FloatTensor(states).to(self.device)
 actions_tensor = torch.LongTensor(actions).to(self.device)
 rewards_tensor = torch.FloatTensor(rewards).to(self.device)
 next_states_tensor = torch.FloatTensor(next_states).to(self.device)
 dones_tensor = torch.BoolTensor(dones).to(self.device)
 is_weights_tensor = torch.FloatTensor(is_weights).to(self.device)
 tree_indices_tensor = torch.LongTensor(tree_indices).to(self.device)

 return (states_tensor, actions_tensor, rewards_tensor,
 next_states_tensor, dones_tensor, is_weights_tensor, tree_indices_tensor)

 def update_priorities(self, tree_indices: Union[List[int], np.ndarray],
 td_errors: Union[List[float], np.ndarray]) -> None:
 """
 Update priorities based on TD errors.

 Args:
 tree_indices: Indices in sum tree
 td_errors: TD errors for corresponding experiences
 """
 def _update:
 if isinstance(tree_indices, torch.Tensor):
 tree_indices_np = tree_indices.cpu.numpy
 else:
 tree_indices_np = np.array(tree_indices)

 if isinstance(td_errors, torch.Tensor):
 td_errors_np = td_errors.cpu.numpy
 else:
 td_errors_np = np.array(td_errors)

 # Computing new priorities
 priorities = (np.abs(td_errors_np) + self.per_config.priority_epsilon) ** self.per_config.alpha

 # Clamp priorities
 priorities = np.clip(priorities, self.per_config.min_priority, self.per_config.max_priority)

 # Update in sum tree
 for tree_index, priority in zip(tree_indices_np, priorities):
 self.sum_tree.update(tree_index, priority)

 # Update maximum priority
 if priority > self.max_priority:
 self.max_priority = priority

 self.priority_updates += len(tree_indices_np)

 self._safe_operation(_update)

 def _rebalance_tree(self) -> None:
 """Rebalancing sum tree for optimal performance."""
 # Simple implementation - periodic statistics recalculation
 self.max_priority = self.sum_tree.max_priority

 self.logger.debug("Sum tree rebalanced",
 max_priority=self.max_priority,
 total_priority=self.sum_tree.total_priority)

 def get_priority_statistics(self) -> Dict[str, Any]:
 """Get priority statistics."""
 tree_stats = self.sum_tree.get_statistics

 return {
 "max_priority": self.max_priority,
 "priority_updates": self.priority_updates,
 "current_beta": self.get_beta,
 "beta_progress": min(self.beta_frames_completed / self.per_config.beta_frames, 1.0),
 "tree_statistics": tree_stats,
 "alpha": self.per_config.alpha,
 }

 def get_statistics(self) -> Dict[str, Any]:
 """Extended statistics for PER buffer."""
 base_stats = super.get_statistics
 priority_stats = self.get_priority_statistics

 # Combine statistics
 stats_dict = base_stats.__dict__.copy
 stats_dict.update({
 "priority_stats": priority_stats,
 "buffer_type": "PrioritizedReplayBuffer"
 })

 return stats_dict

 def clear(self) -> None:
 """Full prioritized buffer cleanup."""
 super.clear

 # Cleanup PER-specific components
 self.sum_tree = SumTree(self.per_config.capacity)
 self.max_priority = self.per_config.max_priority
 self.priority_updates = 0
 self.beta_frames_completed = 0
 self.rebalance_operations = 0

 self.logger.info("Prioritized buffer cleared")

 def __repr__(self) -> str:
 """String representation prioritized buffer."""
 return (
 f"PrioritizedReplayBuffer(size={self.size}, capacity={self.per_config.capacity}, "
 f"alpha={self.per_config.alpha}, beta={self.get_beta:.3f}, "
 f"utilization={self.size/self.per_config.capacity:.1%})"
 )