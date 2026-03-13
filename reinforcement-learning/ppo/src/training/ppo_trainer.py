"""
PPO Trainer Implementation
for scalable RL training

Features:
- Complete PPO training loop
- Multi-environment support
- Checkpointing and resuming
- Comprehensive logging
- Performance monitoring
- Production patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import logging
import time
import os
from collections import defaultdict, deque
import json
import wandb
from pathlib import Path

from ..core.ppo import PPOAlgorithm, PPOConfig
from ..core.ppo2 import PPO2Algorithm, PPO2Config
from ..networks.actor_critic import ActorCriticNetwork, ActorCriticConfig
from ..buffers.rollout_buffer import RolloutBuffer, RolloutBufferConfig
from ..utils.normalization import ObservationNormalizer, RewardNormalizer
from ..utils.scheduling import PPOScheduler
from ..environments.crypto_env import CryptoTradingEnv


@dataclass
class PPOTrainerConfig:
 """Configuration for PPO trainer"""
 
 # Training parameters
 total_timesteps: int = 1_000_000 # Total training steps
 rollout_steps: int = 2048 # Steps per rollout
 batch_size: int = 64 # Mini-batch size
 n_epochs: int = 10 # Epochs per update
 
 # Environment settings
 num_envs: int = 1 # Number of parallel environments
 env_type: str = "crypto_trading" # Environment type
 
 # Algorithm configuration
 algorithm: str = "ppo" # ppo or ppo2
 ppo_config: Optional[PPOConfig] = None
 ppo2_config: Optional[PPO2Config] = None
 
 # Network configuration
 actor_critic_config: Optional[ActorCriticConfig] = None
 
 # Buffer configuration
 buffer_config: Optional[RolloutBufferConfig] = None
 
 # Optimization
 optimizer_type: str = "adam" # adam, sgd, rmsprop
 learning_rate: float = 3e-4
 lr_schedule: str = "linear" # linear, cosine, constant
 weight_decay: float = 0.0
 grad_clip_norm: float = 0.5
 
 # Normalization
 normalize_observations: bool = True
 normalize_rewards: bool = False
 normalize_advantages: bool = True
 
 # Logging and monitoring
 log_interval: int = 10 # Log every N updates
 eval_interval: int = 50 # Evaluate every N updates
 save_interval: int = 100 # Save checkpoint every N updates
 
 # Checkpointing
 checkpoint_dir: str = "./checkpoints"
 save_best_model: bool = True
 
 # Wandb logging
 use_wandb: bool = False
 wandb_project: str = "ppo_crypto_trading"
 wandb_entity: Optional[str] = None
 wandb_tags: List[str] = field(default_factory=list)
 
 # Performance monitoring
 target_fps: float = 1000.0 # Target training FPS
 early_stopping_patience: int = 0 # Early stopping (0 = disabled)
 early_stopping_threshold: float = 0.01
 
 # Production features
 performance_profiling: bool = True
 memory_monitoring: bool = True
 error_recovery: bool = True
 
 def __post_init__(self):
 if self.ppo_config is None:
 self.ppo_config = PPOConfig()
 if self.ppo2_config is None:
 self.ppo2_config = PPO2Config()
 if self.actor_critic_config is None:
 self.actor_critic_config = ActorCriticConfig()
 if self.buffer_config is None:
 self.buffer_config = RolloutBufferConfig(
 buffer_size=self.rollout_steps,
 batch_size=self.batch_size
 )


class PPOTrainer:
 """
 Complete PPO trainer with Features:
 - Multi-environment training
 - Automatic checkpointing
 - Performance monitoring
 - Comprehensive logging
 - Error recovery
 - Production-ready patterns
 """
 
 def __init__(
 self,
 config: PPOTrainerConfig,
 environments: Optional[List[Any]] = None,
 actor_critic: Optional[nn.Module] = None
 ):
 self.config = config
 self.logger = logging.getLogger(__name__)
 
 # Initialize environments
 if environments is None:
 self.envs = self._create_environments()
 else:
 self.envs = environments
 
 self.num_envs = len(self.envs)
 
 # Get environment specs
 obs_space = self.envs[0].observation_space.shape
 if hasattr(self.envs[0].action_space, 'n'):
 action_space = (self.envs[0].action_space.n,)
 action_type = "discrete"
 else:
 action_space = self.envs[0].action_space.shape
 action_type = "continuous"
 
 # Update network config
 self.config.actor_critic_config.obs_dim = obs_space[0]
 self.config.actor_critic_config.action_dim = action_space[0]
 self.config.actor_critic_config.action_type = action_type
 
 # Initialize actor-critic network
 if actor_critic is None:
 self.actor_critic = ActorCriticNetwork(self.config.actor_critic_config)
 else:
 self.actor_critic = actor_critic
 
 # Initialize optimizer
 self.optimizer = self._create_optimizer()
 
 # Initialize PPO algorithm
 if self.config.algorithm == "ppo":
 self.ppo = PPOAlgorithm(
 actor_critic=self.actor_critic,
 config=self.config.ppo_config,
 optimizer=self.optimizer
 )
 elif self.config.algorithm == "ppo2":
 self.ppo = PPO2Algorithm(
 actor_critic=self.actor_critic,
 config=self.config.ppo2_config,
 optimizer=self.optimizer
 )
 else:
 raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
 
 # Initialize buffer
 self.config.buffer_config.observation_space = obs_space
 self.config.buffer_config.action_space = action_space
 self.rollout_buffer = RolloutBuffer(self.config.buffer_config)
 
 # Initialize normalizers
 if self.config.normalize_observations:
 self.obs_normalizer = ObservationNormalizer(obs_space)
 else:
 self.obs_normalizer = None
 
 if self.config.normalize_rewards:
 self.reward_normalizer = RewardNormalizer()
 else:
 self.reward_normalizer = None
 
 # Initialize scheduler
 self.scheduler = PPOScheduler(
 total_steps=self.config.total_timesteps,
 lr_schedule=self.config.lr_schedule,
 initial_lr=self.config.learning_rate,
 final_lr=self.config.learning_rate * 0.1
 )
 
 # Training state
 self.current_timestep = 0
 self.current_update = 0
 self.best_reward = -float('inf')
 self.episode_rewards = deque(maxlen=100)
 self.episode_lengths = deque(maxlen=100)
 
 # Performance monitoring
 self.training_start_time = None
 self.fps_history = deque(maxlen=100)
 
 # Initialize logging
 self._setup_logging()
 
 # Create checkpoint directory
 os.makedirs(self.config.checkpoint_dir, exist_ok=True)
 
 self.logger.info(f"PPO Trainer initialized with {self.num_envs} environments")
 self.logger.info(f"Algorithm: {self.config.algorithm}")
 self.logger.info(f"Observation space: {obs_space}")
 self.logger.info(f"Action space: {action_space} ({action_type})")
 
 def _create_environments(self) -> List[Any]:
 """Create training environments"""
 
 envs = []
 for i in range(self.config.num_envs):
 if self.config.env_type == "crypto_trading":
 env = CryptoTradingEnv() # Default crypto trading environment
 else:
 raise ValueError(f"Unknown environment type: {self.config.env_type}")
 
 envs.append(env)
 
 return envs
 
 def _create_optimizer(self) -> optim.Optimizer:
 """Create optimizer"""
 
 if self.config.optimizer_type == "adam":
 return optim.Adam(
 self.actor_critic.parameters(),
 lr=self.config.learning_rate,
 weight_decay=self.config.weight_decay,
 eps=1e-8
 )
 elif self.config.optimizer_type == "sgd":
 return optim.SGD(
 self.actor_critic.parameters(),
 lr=self.config.learning_rate,
 weight_decay=self.config.weight_decay,
 momentum=0.9
 )
 elif self.config.optimizer_type == "rmsprop":
 return optim.RMSprop(
 self.actor_critic.parameters(),
 lr=self.config.learning_rate,
 weight_decay=self.config.weight_decay,
 eps=1e-8
 )
 else:
 raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
 
 def _setup_logging(self):
 """Setup logging systems"""
 
 # Wandb logging
 if self.config.use_wandb:
 wandb.init(
 project=self.config.wandb_project,
 entity=self.config.wandb_entity,
 tags=self.config.wandb_tags,
 config=self.config.__dict__
 )
 
 def train(self) -> Dict[str, Any]:
 """
 Main training loop
 
 Returns:
 Training statistics
 """
 
 self.training_start_time = time.time()
 self.logger.info(f"Starting training for {self.config.total_timesteps} timesteps")
 
 # Initialize environments
 observations = self._reset_environments()
 
 # Training loop
 while self.current_timestep < self.config.total_timesteps:
 # Collect rollouts
 rollout_start_time = time.time()
 
 try:
 observations = self._collect_rollouts(observations)
 rollout_time = time.time() - rollout_start_time
 
 # Update policy
 update_start_time = time.time()
 update_metrics = self._update_policy()
 update_time = time.time() - update_start_time
 
 # Update statistics
 self.current_update += 1
 
 # Compute FPS
 fps = self.config.rollout_steps * self.num_envs / rollout_time
 self.fps_history.append(fps)
 
 # Log progress
 if self.current_update % self.config.log_interval == 0:
 self._log_progress(update_metrics, fps, rollout_time, update_time)
 
 # Evaluate
 if self.config.eval_interval > 0 and self.current_update % self.config.eval_interval == 0:
 eval_metrics = self._evaluate()
 self._log_evaluation(eval_metrics)
 
 # Save checkpoint
 if self.current_update % self.config.save_interval == 0:
 self._save_checkpoint()
 
 # Check early stopping
 if self._should_stop_early():
 self.logger.info("Early stopping triggered")
 break
 
 except Exception as e:
 if self.config.error_recovery:
 self.logger.error(f"Training error: {e}, attempting recovery...")
 observations = self._reset_environments()
 continue
 else:
 raise e
 
 # Training completed
 total_time = time.time() - self.training_start_time
 final_metrics = self._get_final_metrics(total_time)
 
 # Save final model
 self._save_checkpoint(is_final=True)
 
 self.logger.info(f"Training completed in {total_time:.2f} seconds")
 
 return final_metrics
 
 def _reset_environments(self) -> np.ndarray:
 """Reset all environments"""
 
 observations = []
 for env in self.envs:
 obs = env.reset()
 observations.append(obs)
 
 return np.array(observations)
 
 def _collect_rollouts(self, initial_observations: np.ndarray) -> np.ndarray:
 """
 Collect rollout data from environments
 
 Args:
 initial_observations: Starting observations
 
 Returns:
 Final observations for next rollout
 """
 
 observations = initial_observations
 
 for step in range(self.config.rollout_steps):
 # Normalize observations
 if self.obs_normalizer is not None:
 normalized_obs = torch.stack([
 self.obs_normalizer(torch.tensor(obs, dtype=torch.float32))
 for obs in observations
 ])
 else:
 normalized_obs = torch.tensor(observations, dtype=torch.float32)
 
 # Get actions and values
 with torch.no_grad():
 action_dist, values = self.actor_critic(normalized_obs)
 actions = action_dist.sample()
 log_probs = action_dist.log_prob(actions)
 
 # Convert to numpy
 actions_np = actions.cpu().numpy()
 values_np = values.cpu().numpy().squeeze()
 log_probs_np = log_probs.cpu().numpy()
 
 # Step environments
 next_observations = []
 rewards = []
 dones = []
 
 for i, env in enumerate(self.envs):
 obs, reward, done, info = env.step(actions_np[i])
 
 # Normalize reward
 if self.reward_normalizer is not None:
 reward = self.reward_normalizer(reward, done)
 
 next_observations.append(obs)
 rewards.append(reward)
 dones.append(done)
 
 # Track episode statistics
 if done:
 if 'episode_reward' in info:
 self.episode_rewards.append(info['episode_reward'])
 if 'episode_length' in info:
 self.episode_lengths.append(info['episode_length'])
 
 # Reset environment
 obs = env.reset()
 next_observations[-1] = obs
 
 # Store in buffer
 for i in range(self.num_envs):
 self.rollout_buffer.add(
 obs=normalized_obs[i],
 action=actions[i],
 reward=rewards[i],
 value=values_np[i],
 log_prob=log_probs_np[i],
 done=dones[i]
 )
 
 observations = np.array(next_observations)
 self.current_timestep += self.num_envs
 
 # Compute final values for bootstrap
 if self.obs_normalizer is not None:
 normalized_final_obs = torch.stack([
 self.obs_normalizer(torch.tensor(obs, dtype=torch.float32))
 for obs in observations
 ])
 else:
 normalized_final_obs = torch.tensor(observations, dtype=torch.float32)
 
 with torch.no_grad():
 _, final_values = self.actor_critic(normalized_final_obs)
 final_values = final_values.cpu().numpy().squeeze()
 
 # Compute advantages and returns
 self.rollout_buffer.compute_returns_and_advantages(
 last_values=torch.tensor(final_values)
 )
 
 return observations
 
 def _update_policy(self) -> Dict[str, float]:
 """Update policy using PPO"""
 
 # Get scheduled parameters
 scheduled_params = self.scheduler.step()
 
 # Update optimizer learning rate
 for param_group in self.optimizer.param_groups:
 param_group['lr'] = scheduled_params['learning_rate']
 
 # PPO update
 progress_remaining = 1.0 - (self.current_timestep / self.config.total_timesteps)
 update_metrics = self.ppo.update(self.rollout_buffer, progress_remaining)
 
 # Reset buffer
 self.rollout_buffer.reset()
 
 # Add scheduled parameters to metrics
 update_metrics.update(scheduled_params)
 
 return update_metrics
 
 def _evaluate(self) -> Dict[str, float]:
 """Evaluate current policy"""
 
 eval_rewards = []
 eval_lengths = []
 
 # Run evaluation episodes
 for env in self.envs[:min(5, self.num_envs)]: # Evaluate on subset
 obs = env.reset()
 episode_reward = 0
 episode_length = 0
 done = False
 
 while not done:
 if self.obs_normalizer is not None:
 normalized_obs = self.obs_normalizer(
 torch.tensor(obs, dtype=torch.float32),
 update_stats=False
 )
 else:
 normalized_obs = torch.tensor(obs, dtype=torch.float32)
 
 with torch.no_grad():
 action_dist, _ = self.actor_critic(normalized_obs.unsqueeze(0))
 action = action_dist.mean # Deterministic evaluation
 
 obs, reward, done, _ = env.step(action.cpu().numpy().squeeze())
 episode_reward += reward
 episode_length += 1
 
 if episode_length > 1000: # Prevent infinite episodes
 break
 
 eval_rewards.append(episode_reward)
 eval_lengths.append(episode_length)
 
 return {
 "eval_mean_reward": np.mean(eval_rewards),
 "eval_std_reward": np.std(eval_rewards),
 "eval_mean_length": np.mean(eval_lengths),
 "eval_std_length": np.std(eval_lengths)
 }
 
 def _log_progress(
 self,
 update_metrics: Dict[str, float],
 fps: float,
 rollout_time: float,
 update_time: float
 ):
 """Log training progress"""
 
 # Compute statistics
 mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
 mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0.0
 mean_fps = np.mean(self.fps_history) if self.fps_history else 0.0
 
 # Progress metrics
 progress = self.current_timestep / self.config.total_timesteps
 time_elapsed = time.time() - self.training_start_time
 time_remaining = time_elapsed * (1 - progress) / progress if progress > 0 else 0
 
 # Log message
 log_msg = (
 f"Update {self.current_update:>6d} | "
 f"Timesteps: {self.current_timestep:>8d}/{self.config.total_timesteps} | "
 f"Progress: {progress:>6.1%} | "
 f"FPS: {fps:>6.0f} | "
 f"Reward: {mean_reward:>8.2f} | "
 f"Length: {mean_length:>6.1f} | "
 f"Policy Loss: {update_metrics.get('policy_loss', 0):>8.4f} | "
 f"Value Loss: {update_metrics.get('value_loss', 0):>8.4f}"
 )
 
 self.logger.info(log_msg)
 
 # Wandb logging
 if self.config.use_wandb:
 wandb_metrics = {
 "timesteps": self.current_timestep,
 "progress": progress,
 "fps": fps,
 "mean_fps": mean_fps,
 "mean_reward": mean_reward,
 "mean_episode_length": mean_length,
 "rollout_time": rollout_time,
 "update_time": update_time,
 "time_elapsed": time_elapsed,
 "time_remaining": time_remaining,
 **update_metrics
 }
 
 wandb.log(wandb_metrics, step=self.current_update)
 
 def _log_evaluation(self, eval_metrics: Dict[str, float]):
 """Log evaluation results"""
 
 self.logger.info(
 f"Evaluation | "
 f"Mean Reward: {eval_metrics['eval_mean_reward']:>8.2f} ± {eval_metrics['eval_std_reward']:>6.2f} | "
 f"Mean Length: {eval_metrics['eval_mean_length']:>6.1f} ± {eval_metrics['eval_std_length']:>5.1f}"
 )
 
 # Check for best model
 if eval_metrics['eval_mean_reward'] > self.best_reward:
 self.best_reward = eval_metrics['eval_mean_reward']
 if self.config.save_best_model:
 self._save_checkpoint(is_best=True)
 
 # Wandb logging
 if self.config.use_wandb:
 wandb.log(eval_metrics, step=self.current_update)
 
 def _should_stop_early(self) -> bool:
 """Check early stopping condition"""
 
 if self.config.early_stopping_patience <= 0:
 return False
 
 if len(self.episode_rewards) < self.config.early_stopping_patience:
 return False
 
 # Check if no improvement in recent episodes
 recent_rewards = list(self.episode_rewards)[-self.config.early_stopping_patience:]
 improvement = max(recent_rewards) - min(recent_rewards)
 
 return improvement < self.config.early_stopping_threshold
 
 def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
 """Save training checkpoint"""
 
 checkpoint = {
 "current_timestep": self.current_timestep,
 "current_update": self.current_update,
 "best_reward": self.best_reward,
 "actor_critic_state_dict": self.actor_critic.state_dict(),
 "optimizer_state_dict": self.optimizer.state_dict(),
 "config": self.config,
 "episode_rewards": list(self.episode_rewards),
 "episode_lengths": list(self.episode_lengths)
 }
 
 # Save normalizer states
 if self.obs_normalizer is not None:
 checkpoint["obs_normalizer_state"] = self.obs_normalizer.running_stats.get_state()
 if self.reward_normalizer is not None:
 checkpoint["reward_normalizer_state"] = self.reward_normalizer.running_stats.get_state()
 
 # Determine filename
 if is_final:
 filename = "final_checkpoint.pt"
 elif is_best:
 filename = "best_checkpoint.pt"
 else:
 filename = f"checkpoint_{self.current_update:06d}.pt"
 
 filepath = os.path.join(self.config.checkpoint_dir, filename)
 torch.save(checkpoint, filepath)
 
 self.logger.info(f"Checkpoint saved: {filepath}")
 
 def _get_final_metrics(self, total_time: float) -> Dict[str, Any]:
 """Get final training metrics"""
 
 return {
 "total_timesteps": self.current_timestep,
 "total_updates": self.current_update,
 "total_time": total_time,
 "mean_fps": np.mean(self.fps_history) if self.fps_history else 0.0,
 "final_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
 "best_reward": self.best_reward,
 "final_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0
 }
 
 def load_checkpoint(self, checkpoint_path: str):
 """Load training checkpoint"""
 
 checkpoint = torch.load(checkpoint_path, map_location='cpu')
 
 # Load states
 self.current_timestep = checkpoint["current_timestep"]
 self.current_update = checkpoint["current_update"]
 self.best_reward = checkpoint["best_reward"]
 
 self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
 self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
 
 self.episode_rewards.extend(checkpoint["episode_rewards"])
 self.episode_lengths.extend(checkpoint["episode_lengths"])
 
 # Load normalizer states
 if "obs_normalizer_state" in checkpoint and self.obs_normalizer is not None:
 self.obs_normalizer.running_stats.set_state(checkpoint["obs_normalizer_state"])
 if "reward_normalizer_state" in checkpoint and self.reward_normalizer is not None:
 self.reward_normalizer.running_stats.set_state(checkpoint["reward_normalizer_state"])
 
 self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
 self.logger.info(f"Resuming from timestep {self.current_timestep}")


# Export main class
__all__ = [
 "PPOTrainerConfig",
 "PPOTrainer"
]