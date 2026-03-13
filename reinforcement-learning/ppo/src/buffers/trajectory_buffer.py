"""
Trajectory Buffer Implementation for PPO
for trajectory-based learning

Trajectory buffer manages complete episodes:
- Episode-wise data storage
- Trajectory-level statistics
- Multi-environment support
- Hierarchical trajectory organization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from collections import deque, defaultdict
import pickle
import gzip

from ..advantages.gae import GAE, GAEConfig
from ..utils.normalization import normalize_advantages


class TrajectoryStep(NamedTuple):
 """Single step in trajectory"""
 observation: torch.Tensor
 action: torch.Tensor
 reward: float
 value: float
 log_prob: float
 done: bool
 info: Dict[str, Any] = {}


class Trajectory:
 """
 Single trajectory (episode) container
 
 Stores complete episode data and provides
 trajectory-level computations
 """
 
 def __init__(self, trajectory_id: Optional[str] = None):
 self.trajectory_id = trajectory_id or f"traj_{id(self)}"
 
 # Trajectory data
 self.steps: List[TrajectoryStep] = []
 self.episode_return = 0.0
 self.episode_length = 0
 self.is_complete = False
 
 # Computed values
 self.advantages: Optional[torch.Tensor] = None
 self.returns: Optional[torch.Tensor] = None
 self.discounted_return = 0.0
 
 # Metadata
 self.start_time = None
 self.end_time = None
 self.env_id = None
 
 def add_step(self, step: TrajectoryStep):
 """Add step to trajectory"""
 self.steps.append(step)
 self.episode_return += step.reward
 self.episode_length += 1
 
 if step.done:
 self.is_complete = True
 
 def add_transition(
 self,
 obs: torch.Tensor,
 action: torch.Tensor,
 reward: float,
 value: float,
 log_prob: float,
 done: bool,
 **info
 ):
 """Add transition data as trajectory step"""
 step = TrajectoryStep(
 observation=obs,
 action=action,
 reward=reward,
 value=value,
 log_prob=log_prob,
 done=done,
 info=info
 )
 self.add_step(step)
 
 def compute_returns_and_advantages(
 self,
 gae_estimator: GAE,
 next_value: Optional[float] = None
 ):
 """Compute returns and advantages for trajectory"""
 
 if len(self.steps) == 0:
 return
 
 # Extract data
 rewards = torch.tensor([step.reward for step in self.steps])
 values = torch.tensor([step.value for step in self.steps])
 dones = torch.tensor([step.done for step in self.steps])
 
 # Reshape for GAE [seq_len, batch_size=1]
 rewards = rewards.unsqueeze(1)
 values = values.unsqueeze(1)
 dones = dones.unsqueeze(1)
 
 # Compute advantages
 next_values = torch.tensor([[next_value]]) if next_value is not None else None
 
 advantages, returns = gae_estimator.compute_advantages_and_returns(
 rewards=rewards,
 values=values,
 dones=dones,
 next_values=next_values
 )
 
 # Store computed values
 self.advantages = advantages.squeeze(1)
 self.returns = returns.squeeze(1)
 self.discounted_return = returns[0].item()
 
 def get_data(self) -> Dict[str, torch.Tensor]:
 """Get trajectory data as tensors"""
 
 if len(self.steps) == 0:
 return {}
 
 # Stack observations and actions
 observations = torch.stack([step.observation for step in self.steps])
 actions = torch.stack([step.action for step in self.steps])
 
 # Convert other data
 rewards = torch.tensor([step.reward for step in self.steps])
 values = torch.tensor([step.value for step in self.steps])
 log_probs = torch.tensor([step.log_prob for step in self.steps])
 dones = torch.tensor([step.done for step in self.steps])
 
 data = {
 "observations": observations,
 "actions": actions,
 "rewards": rewards,
 "values": values,
 "log_probs": log_probs,
 "dones": dones
 }
 
 # Add computed values if available
 if self.advantages is not None:
 data["advantages"] = self.advantages
 if self.returns is not None:
 data["returns"] = self.returns
 
 return data
 
 def get_statistics(self) -> Dict[str, float]:
 """Get trajectory statistics"""
 
 return {
 "episode_return": self.episode_return,
 "episode_length": self.episode_length,
 "discounted_return": self.discounted_return,
 "is_complete": self.is_complete,
 "mean_reward": self.episode_return / max(self.episode_length, 1),
 "mean_value": np.mean([step.value for step in self.steps]) if self.steps else 0.0
 }


@dataclass
class TrajectoryBufferConfig:
 """Configuration for trajectory buffer"""
 
 # Buffer capacity
 max_trajectories: int = 1000 # Maximum number of trajectories
 max_steps_per_trajectory: int = 1000 # Maximum steps per trajectory
 
 # GAE configuration
 gae_config: Optional[GAEConfig] = None
 
 # Sampling configuration
 batch_size: int = 32 # Number of trajectories per batch
 shuffle_trajectories: bool = True # Shuffle trajectories
 
 # Multi-environment support
 num_environments: int = 1 # Number of parallel environments
 env_balancing: bool = True # Balance trajectories across envs
 
 # Storage options
 device: str = "cpu" # Storage device
 compression: bool = False # Compress stored trajectories
 
 # Quality filtering
 min_trajectory_length: int = 1 # Minimum trajectory length
 max_trajectory_length: Optional[int] = None # Maximum trajectory length
 reward_threshold: Optional[float] = None # Minimum reward threshold
 
 # Advanced features
 prioritize_trajectories: bool = False # Priority sampling
 diversity_sampling: bool = False # Diversity-based sampling
 curriculum_learning: bool = False # Curriculum learning support
 
 def __post_init__(self):
 if self.gae_config is None:
 self.gae_config = GAEConfig()


class TrajectoryBuffer:
 """
 Trajectory buffer for episode-based PPO training
 
 Features:
 - Complete episode storage
 - Trajectory-level statistics
 - Multi-environment support
 - Quality filtering
 - Priority sampling
 """
 
 def __init__(self, config: TrajectoryBufferConfig):
 self.config = config
 self.gae_estimator = GAE(config.gae_config)
 
 # Storage
 self.trajectories: List[Trajectory] = []
 self.incomplete_trajectories: Dict[str, Trajectory] = {}
 
 # Environment tracking
 self.env_trajectories: Dict[int, List[Trajectory]] = defaultdict(list)
 
 # Statistics
 self.total_steps = 0
 self.total_episodes = 0
 
 # Priority sampling
 if config.prioritize_trajectories:
 self.trajectory_priorities: List[float] = []
 
 def start_trajectory(
 self,
 trajectory_id: Optional[str] = None,
 env_id: int = 0
 ) -> str:
 """Start new trajectory"""
 
 if trajectory_id is None:
 trajectory_id = f"traj_{self.total_episodes}_{env_id}"
 
 trajectory = Trajectory(trajectory_id)
 trajectory.env_id = env_id
 
 self.incomplete_trajectories[trajectory_id] = trajectory
 
 return trajectory_id
 
 def add_step(
 self,
 trajectory_id: str,
 obs: torch.Tensor,
 action: torch.Tensor,
 reward: float,
 value: float,
 log_prob: float,
 done: bool,
 **info
 ):
 """Add step to specified trajectory"""
 
 if trajectory_id not in self.incomplete_trajectories:
 self.start_trajectory(trajectory_id)
 
 trajectory = self.incomplete_trajectories[trajectory_id]
 trajectory.add_transition(
 obs=obs,
 action=action,
 reward=reward,
 value=value,
 log_prob=log_prob,
 done=done,
 **info
 )
 
 self.total_steps += 1
 
 # Complete trajectory if done
 if done:
 self.complete_trajectory(trajectory_id)
 
 def complete_trajectory(
 self,
 trajectory_id: str,
 next_value: Optional[float] = None
 ):
 """Mark trajectory as complete and compute advantages"""
 
 if trajectory_id not in self.incomplete_trajectories:
 return
 
 trajectory = self.incomplete_trajectories.pop(trajectory_id)
 
 # Quality filtering
 if not self._passes_quality_filter(trajectory):
 return
 
 # Compute advantages and returns
 trajectory.compute_returns_and_advantages(
 self.gae_estimator,
 next_value=next_value
 )
 
 # Store trajectory
 self._store_trajectory(trajectory)
 
 self.total_episodes += 1
 
 def _passes_quality_filter(self, trajectory: Trajectory) -> bool:
 """Check if trajectory passes quality filters"""
 
 # Length filter
 if trajectory.episode_length < self.config.min_trajectory_length:
 return False
 
 if (self.config.max_trajectory_length is not None and
 trajectory.episode_length > self.config.max_trajectory_length):
 return False
 
 # Reward filter
 if (self.config.reward_threshold is not None and
 trajectory.episode_return < self.config.reward_threshold):
 return False
 
 return True
 
 def _store_trajectory(self, trajectory: Trajectory):
 """Store trajectory in buffer"""
 
 # Add to main storage
 self.trajectories.append(trajectory)
 
 # Add to environment-specific storage
 if trajectory.env_id is not None:
 self.env_trajectories[trajectory.env_id].append(trajectory)
 
 # Priority sampling
 if self.config.prioritize_trajectories:
 priority = self._compute_trajectory_priority(trajectory)
 self.trajectory_priorities.append(priority)
 
 # Maintain buffer size
 if len(self.trajectories) > self.config.max_trajectories:
 self._evict_trajectories()
 
 def _compute_trajectory_priority(self, trajectory: Trajectory) -> float:
 """Compute priority for trajectory"""
 
 # Simple priority based on return
 base_priority = abs(trajectory.episode_return)
 
 # Add diversity bonus if enabled
 if self.config.diversity_sampling:
 diversity_bonus = self._compute_diversity_bonus(trajectory)
 base_priority += diversity_bonus
 
 return base_priority
 
 def _compute_diversity_bonus(self, trajectory: Trajectory) -> float:
 """Compute diversity bonus for trajectory"""
 
 if len(self.trajectories) < 2:
 return 0.0
 
 # Simple diversity measure based on trajectory length and return
 current_stats = (trajectory.episode_length, trajectory.episode_return)
 
 # Find most similar trajectory
 min_distance = float('inf')
 for existing_traj in self.trajectories[-10:]: # Compare with recent trajectories
 existing_stats = (existing_traj.episode_length, existing_traj.episode_return)
 distance = np.linalg.norm(np.array(current_stats) - np.array(existing_stats))
 min_distance = min(min_distance, distance)
 
 # Higher diversity bonus for more different trajectories
 return min_distance / 100.0
 
 def _evict_trajectories(self):
 """Remove old trajectories to maintain buffer size"""
 
 excess = len(self.trajectories) - self.config.max_trajectories
 
 if excess <= 0:
 return
 
 # Remove oldest trajectories (FIFO)
 for _ in range(excess):
 evicted = self.trajectories.pop(0)
 
 # Remove from environment storage
 if evicted.env_id is not None:
 try:
 self.env_trajectories[evicted.env_id].remove(evicted)
 except ValueError:
 pass
 
 # Remove priority
 if self.config.prioritize_trajectories and self.trajectory_priorities:
 self.trajectory_priorities.pop(0)
 
 def sample_trajectories(
 self,
 num_trajectories: Optional[int] = None
 ) -> List[Trajectory]:
 """Sample trajectories for training"""
 
 if num_trajectories is None:
 num_trajectories = min(self.config.batch_size, len(self.trajectories))
 
 if len(self.trajectories) == 0:
 return []
 
 if self.config.prioritize_trajectories:
 return self._priority_sample_trajectories(num_trajectories)
 elif self.config.shuffle_trajectories:
 indices = np.random.choice(
 len(self.trajectories),
 size=min(num_trajectories, len(self.trajectories)),
 replace=False
 )
 return [self.trajectories[i] for i in indices]
 else:
 return self.trajectories[:num_trajectories]
 
 def _priority_sample_trajectories(self, num_trajectories: int) -> List[Trajectory]:
 """Sample trajectories based on priorities"""
 
 if not self.trajectory_priorities:
 return self.sample_trajectories(num_trajectories)
 
 # Compute sampling probabilities
 priorities = np.array(self.trajectory_priorities)
 priorities = np.maximum(priorities, 1e-8) # Avoid zeros
 probabilities = priorities / priorities.sum()
 
 # Sample indices
 indices = np.random.choice(
 len(self.trajectories),
 size=min(num_trajectories, len(self.trajectories)),
 p=probabilities,
 replace=False
 )
 
 return [self.trajectories[i] for i in indices]
 
 def get_all_data(self) -> Dict[str, torch.Tensor]:
 """Get all trajectory data concatenated"""
 
 if len(self.trajectories) == 0:
 return {}
 
 # Collect data from all trajectories
 all_observations = []
 all_actions = []
 all_rewards = []
 all_values = []
 all_log_probs = []
 all_dones = []
 all_advantages = []
 all_returns = []
 
 for trajectory in self.trajectories:
 data = trajectory.get_data()
 
 all_observations.append(data["observations"])
 all_actions.append(data["actions"])
 all_rewards.append(data["rewards"])
 all_values.append(data["values"])
 all_log_probs.append(data["log_probs"])
 all_dones.append(data["dones"])
 
 if "advantages" in data:
 all_advantages.append(data["advantages"])
 if "returns" in data:
 all_returns.append(data["returns"])
 
 # Concatenate all data
 combined_data = {
 "observations": torch.cat(all_observations, dim=0),
 "actions": torch.cat(all_actions, dim=0),
 "rewards": torch.cat(all_rewards, dim=0),
 "values": torch.cat(all_values, dim=0),
 "log_probs": torch.cat(all_log_probs, dim=0),
 "dones": torch.cat(all_dones, dim=0)
 }
 
 if all_advantages:
 combined_data["advantages"] = torch.cat(all_advantages, dim=0)
 if all_returns:
 combined_data["returns"] = torch.cat(all_returns, dim=0)
 
 return combined_data
 
 def get_statistics(self) -> Dict[str, Any]:
 """Get buffer statistics"""
 
 if len(self.trajectories) == 0:
 return {"num_trajectories": 0, "total_steps": 0}
 
 # Collect trajectory statistics
 episode_returns = [t.episode_return for t in self.trajectories]
 episode_lengths = [t.episode_length for t in self.trajectories]
 
 stats = {
 "num_trajectories": len(self.trajectories),
 "incomplete_trajectories": len(self.incomplete_trajectories),
 "total_steps": self.total_steps,
 "total_episodes": self.total_episodes,
 "mean_episode_return": np.mean(episode_returns),
 "std_episode_return": np.std(episode_returns),
 "mean_episode_length": np.mean(episode_lengths),
 "std_episode_length": np.std(episode_lengths),
 "min_episode_return": np.min(episode_returns),
 "max_episode_return": np.max(episode_returns)
 }
 
 # Environment-specific statistics
 if self.config.num_environments > 1:
 env_stats = {}
 for env_id, env_trajectories in self.env_trajectories.items():
 if env_trajectories:
 env_returns = [t.episode_return for t in env_trajectories]
 env_stats[f"env_{env_id}_mean_return"] = np.mean(env_returns)
 env_stats[f"env_{env_id}_num_trajectories"] = len(env_trajectories)
 
 stats.update(env_stats)
 
 return stats
 
 def clear(self):
 """Clear all trajectories"""
 
 self.trajectories.clear()
 self.incomplete_trajectories.clear()
 self.env_trajectories.clear()
 
 if self.config.prioritize_trajectories:
 self.trajectory_priorities.clear()
 
 self.total_steps = 0
 self.total_episodes = 0
 
 def save(self, filepath: str):
 """Save buffer to file"""
 
 data = {
 "trajectories": self.trajectories,
 "config": self.config,
 "statistics": self.get_statistics()
 }
 
 if self.config.compression:
 with gzip.open(filepath, 'wb') as f:
 pickle.dump(data, f)
 else:
 with open(filepath, 'wb') as f:
 pickle.dump(data, f)
 
 def load(self, filepath: str):
 """Load buffer from file"""
 
 if self.config.compression:
 with gzip.open(filepath, 'rb') as f:
 data = pickle.load(f)
 else:
 with open(filepath, 'rb') as f:
 data = pickle.load(f)
 
 self.trajectories = data["trajectories"]
 
 # Rebuild environment mapping
 self.env_trajectories.clear()
 for trajectory in self.trajectories:
 if trajectory.env_id is not None:
 self.env_trajectories[trajectory.env_id].append(trajectory)
 
 # Rebuild priorities if needed
 if self.config.prioritize_trajectories:
 self.trajectory_priorities = [
 self._compute_trajectory_priority(t) for t in self.trajectories
 ]


# Factory function
def create_trajectory_buffer(
 config: Optional[TrajectoryBufferConfig] = None,
 **kwargs
) -> TrajectoryBuffer:
 """Create trajectory buffer"""
 
 if config is None:
 config = TrajectoryBufferConfig()
 
 # Update config with kwargs
 for key, value in kwargs.items():
 if hasattr(config, key):
 setattr(config, key, value)
 
 return TrajectoryBuffer(config)


# Export classes and functions
__all__ = [
 "TrajectoryStep",
 "Trajectory",
 "TrajectoryBufferConfig",
 "TrajectoryBuffer",
 "create_trajectory_buffer"
]