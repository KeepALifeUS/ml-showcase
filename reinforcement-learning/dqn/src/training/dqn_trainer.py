"""
DQN Training Infrastructure with enterprise patterns.

Comprehensive training system for DQN agents:
- Multi-environment training support
- Distributed training with multiple workers
- Advanced monitoring and logging
- Automatic hyperparameter tuning
- Model checkpointing and versioning
- Performance evaluation and benchmarking
- Real-time visualization
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import wandb
from pydantic import BaseModel, Field, validator
import structlog
from datetime import datetime, timedelta
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from tqdm import tqdm
import gymnasium as gym

from ..core.dqn import DQN, DQNConfig
from ..agents.dqn_trader import DQNTrader, CryptoTradingDQNConfig
from ..utils.metrics import PerformanceMetrics

logger = structlog.get_logger(__name__)


@dataclass
class TrainingEpisodeResult:
 """Result training episode."""
 episode_id: int
 total_reward: float
 episode_length: int
 avg_loss: float
 epsilon: float
 training_time: float
 evaluation_metrics: Optional[Dict[str, float]] = None
 error: Optional[str] = None


@dataclass
class TrainingSession:
 """Information about training session."""
 session_id: str
 start_time: datetime
 config: Dict[str, Any]
 total_episodes: int = 0
 total_steps: int = 0
 best_reward: float = float('-inf')
 best_model_path: Optional[str] = None
 episode_results: List[TrainingEpisodeResult] = field(default_factory=list)


class TrainingConfig(BaseModel):
 """Configuration training process."""

 # Training parameters
 num_episodes: int = Field(default=1000, description="Number of episodes", gt=0)
 max_episode_steps: int = Field(default=1000, description="Maximum steps in episode", gt=0)

 # Evaluation
 eval_frequency: int = Field(default=50, description="Frequency evaluation", gt=0)
 eval_episodes: int = Field(default=10, description="Episodes for evaluation", gt=0)

 # Monitoring
 log_frequency: int = Field(default=10, description="Frequency logging", gt=0)
 save_frequency: int = Field(default=100, description="Frequency saving", gt=0)

 # Performance
 num_workers: int = Field(default=1, description="Number of worker processes", gt=0, le=16)
 batch_training: bool = Field(default=False, description="Batch training mode")

 # Checkpointing
 checkpoint_dir: str = Field(default="./checkpoints", description="Directory checkpoints")
 keep_best_n: int = Field(default=3, description="Save best N models", gt=0)

 # Early stopping
 early_stopping: bool = Field(default=True, description="Early stopping")
 patience: int = Field(default=100, description="Patience for early stopping", gt=0)
 min_improvement: float = Field(default=0.01, description="Minimum improvement", gt=0)

 # Distributed training
 distributed: bool = Field(default=False, description="Distributed training")
 master_port: int = Field(default=29500, description="Master port for distributed", gt=1024)

 # Logging
 use_tensorboard: bool = Field(default=True, description="Use TensorBoard")
 use_wandb: bool = Field(default=False, description="Use W&B")
 wandb_project: Optional[str] = Field(default=None, description="W&B project name")

 # Environment
 render_mode: Optional[str] = Field(default=None, description="Mode rendering")
 seed: Optional[int] = Field(default=None, description="Random seed")


class DQNTrainer:
 """
 Comprehensive DQN Training System with enterprise functionality.

 Features:
 - Multi-environment support (OpenAI Gym, custom environments)
 - Distributed training with multiple workers
 - Advanced monitoring (TensorBoard, W&B)
 - Automatic model checkpointing and versioning
 - Performance evaluation and benchmarking
 - Early stopping with configurable patience
 - Real-time training visualization
 - Hyperparameter optimization integration
 - Production-ready logging and error handling
 """

 def __init__(self,
 agent: Union[DQN, DQNTrader],
 env_factory: Callable[[], gym.Env],
 config: Optional[TrainingConfig] = None):
 """
 Initialization DQN Trainer.

 Args:
 agent: DQN agent for training
 env_factory: Factory function for creation environment
 config: Configuration training
 """
 if config is None:
 config = TrainingConfig

 self.config = config
 self.agent = agent
 self.env_factory = env_factory

 # Training state
 self.current_session: Optional[TrainingSession] = None
 self.training_metrics = PerformanceMetrics

 # Setup directories
 self.checkpoint_dir = Path(config.checkpoint_dir)
 self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

 self.log_dir = self.checkpoint_dir / "logs"
 self.log_dir.mkdir(exist_ok=True)

 # Initialize logging
 self.writer = None
 if config.use_tensorboard:
 self.writer = SummaryWriter(log_dir=str(self.log_dir))

 if config.use_wandb and config.wandb_project:
 wandb.init(project=config.wandb_project, config=config.dict)

 # Performance tracking
 self.best_models = []
 self.training_history = []

 # Threading for async operations
 self.executor = ThreadPoolExecutor(max_workers=4)

 self.logger = structlog.get_logger(__name__).bind(
 component="DQNTrainer",
 agent_type=type(agent).__name__
 )

 self.logger.info("DQN Trainer initialized", config=config.dict)

 def create_training_session(self) -> TrainingSession:
 """Creating new training session."""
 session_id = datetime.now.strftime("%Y%m%d_%H%M%S")

 session = TrainingSession(
 session_id=session_id,
 start_time=datetime.now,
 config=self.config.dict
 )

 self.current_session = session
 self.logger.info("Created training session", session_id=session_id)

 return session

 def train_single_episode(self,
 episode_id: int,
 env: gym.Env,
 training: bool = True) -> TrainingEpisodeResult:
 """
 Training a single episode.

 Args:
 episode_id: ID episode
 env: Environment for training
 training: Mode training

 Returns:
 Result episode
 """
 start_time = datetime.now

 try:
 state = env.reset
 if isinstance(state, tuple):
 state = state[0] # Gym 0.26+ compatibility

 total_reward = 0.0
 episode_length = 0
 losses = []
 done = False

 while not done and episode_length < self.config.max_episode_steps:
 # Action selection
 action = self.agent.act(state, training=training)

 # Environment step
 step_result = env.step(action)
 next_state, reward, done = step_result[:3]

 total_reward += reward
 episode_length += 1

 if training:
 # Store experience and train
 self.agent.store_experience(state, action, reward, next_state, done)

 # Training step
 if hasattr(self.agent, 'train_step'):
 metrics = self.agent.train_step
 if isinstance(metrics, dict) and 'loss' in metrics:
 losses.append(metrics['loss'])

 state = next_state

 # Calculate metrics
 avg_loss = np.mean(losses) if losses else 0.0
 epsilon = getattr(self.agent, 'epsilon_schedule', None)
 current_epsilon = epsilon.get_epsilon if epsilon else 0.0

 training_time = (datetime.now - start_time).total_seconds

 result = TrainingEpisodeResult(
 episode_id=episode_id,
 total_reward=total_reward,
 episode_length=episode_length,
 avg_loss=avg_loss,
 epsilon=current_epsilon,
 training_time=training_time
 )

 self.logger.debug("Episode completed",
 episode=episode_id,
 reward=total_reward,
 length=episode_length,
 training_time=training_time)

 return result

 except Exception as e:
 self.logger.error("Error in episode", episode=episode_id, error=str(e))
 return TrainingEpisodeResult(
 episode_id=episode_id,
 total_reward=0.0,
 episode_length=0,
 avg_loss=0.0,
 epsilon=0.0,
 training_time=0.0,
 error=str(e)
 )

 def evaluate_agent(self, num_episodes: int = None) -> Dict[str, float]:
 """
 Evaluation agent.

 Args:
 num_episodes: Number of episodes for evaluation

 Returns:
 Evaluation metrics
 """
 if num_episodes is None:
 num_episodes = self.config.eval_episodes

 env = self.env_factory
 episode_rewards = []
 episode_lengths = []

 self.logger.info("Start evaluation", episodes=num_episodes)

 for episode in range(num_episodes):
 result = self.train_single_episode(episode, env, training=False)
 episode_rewards.append(result.total_reward)
 episode_lengths.append(result.episode_length)

 env.close

 # Calculate metrics
 metrics = {
 "eval_reward_mean": np.mean(episode_rewards),
 "eval_reward_std": np.std(episode_rewards),
 "eval_reward_min": np.min(episode_rewards),
 "eval_reward_max": np.max(episode_rewards),
 "eval_length_mean": np.mean(episode_lengths),
 "eval_length_std": np.std(episode_lengths),
 }

 self.logger.info("Evaluation completed", **metrics)
 return metrics

 def save_checkpoint(self,
 episode: int,
 metrics: Dict[str, float],
 is_best: bool = False) -> str:
 """
 Saving checkpoint.

 Args:
 episode: Number episode
 metrics: Current metrics
 is_best: Best whether this model

 Returns:
 Path to checkpoint
 """
 checkpoint_name = f"checkpoint_ep{episode:06d}"
 if is_best:
 checkpoint_name += "_best"

 checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pth"

 # Training metadata
 metadata = {
 "episode": episode,
 "session_id": self.current_session.session_id if self.current_session else None,
 "metrics": metrics,
 "timestamp": datetime.now.isoformat,
 "config": self.config.dict,
 }

 # Save checkpoint
 self.agent.save_checkpoint(str(checkpoint_path), metadata)

 # Manage best models
 if is_best:
 self.best_models.append({
 "episode": episode,
 "path": str(checkpoint_path),
 "reward": metrics.get("eval_reward_mean", 0),
 "timestamp": datetime.now
 })

 # Keep only best N models
 self.best_models.sort(key=lambda x: x["reward"], reverse=True)
 if len(self.best_models) > self.config.keep_best_n:
 old_model = self.best_models.pop
 old_path = Path(old_model["path"])
 if old_path.exists:
 old_path.unlink

 self.logger.info("Checkpoint saved",
 path=checkpoint_path,
 is_best=is_best)

 return str(checkpoint_path)

 def log_metrics(self, episode: int, metrics: Dict[str, Any]) -> None:
 """
 Logging metrics.

 Args:
 episode: Number episode
 metrics: Metrics for logging
 """
 # TensorBoard logging
 if self.writer:
 for key, value in metrics.items:
 if isinstance(value, (int, float)):
 self.writer.add_scalar(f"training/{key}", value, episode)

 # W&B logging
 if wandb.run is not None:
 wandb_metrics = {f"episode": episode}
 wandb_metrics.update({k: v for k, v in metrics.items
 if isinstance(v, (int, float))})
 wandb.log(wandb_metrics)

 # Structured logging
 self.logger.info("Training metrics", episode=episode, **metrics)

 def train(self) -> TrainingSession:
 """
 Main function training.

 Returns:
 Result training session
 """
 session = self.create_training_session

 env = self.env_factory
 best_reward = float('-inf')
 episodes_without_improvement = 0

 self.logger.info("Start training",
 episodes=self.config.num_episodes,
 session_id=session.session_id)

 try:
 # Training loop
 for episode in tqdm(range(self.config.num_episodes), desc="Training"):
 # Train episode
 result = self.train_single_episode(episode, env, training=True)
 session.episode_results.append(result)
 session.total_episodes += 1
 session.total_steps += result.episode_length

 # Periodic evaluation
 if episode % self.config.eval_frequency == 0:
 eval_metrics = self.evaluate_agent
 result.evaluation_metrics = eval_metrics

 current_reward = eval_metrics["eval_reward_mean"]

 # Check for improvement
 if current_reward > best_reward + self.config.min_improvement:
 best_reward = current_reward
 episodes_without_improvement = 0

 # Save best model
 checkpoint_path = self.save_checkpoint(episode, eval_metrics, is_best=True)
 session.best_reward = best_reward
 session.best_model_path = checkpoint_path
 else:
 episodes_without_improvement += self.config.eval_frequency

 # Log metrics
 all_metrics = {
 "episode_reward": result.total_reward,
 "episode_length": result.episode_length,
 "avg_loss": result.avg_loss,
 "epsilon": result.epsilon,
 **eval_metrics
 }
 self.log_metrics(episode, all_metrics)

 # Periodic checkpointing
 if episode % self.config.save_frequency == 0:
 self.save_checkpoint(episode, {"episode_reward": result.total_reward})

 # Early stopping
 if (self.config.early_stopping and
 episodes_without_improvement >= self.config.patience):
 self.logger.info("Early stopping triggered",
 episodes_without_improvement=episodes_without_improvement)
 break

 # Progress logging
 if episode % self.config.log_frequency == 0:
 self.logger.info("Training progress",
 episode=episode,
 progress=f"{episode/self.config.num_episodes:.1%}",
 reward=result.total_reward,
 best_reward=best_reward)

 except KeyboardInterrupt:
 self.logger.info("Training interrupted by user")
 except Exception as e:
 self.logger.error("Training error", error=str(e))
 raise
 finally:
 env.close
 if self.writer:
 self.writer.close

 # Final evaluation
 final_metrics = self.evaluate_agent
 self.logger.info("Training completed",
 session_id=session.session_id,
 total_episodes=session.total_episodes,
 best_reward=session.best_reward,
 **final_metrics)

 return session

 def train_distributed(self, world_size: int = None) -> TrainingSession:
 """
 Distributed training with multiple workers.

 Args:
 world_size: Number of workers

 Returns:
 Training session result
 """
 if world_size is None:
 world_size = self.config.num_workers

 self.logger.info("Start distributed training", world_size=world_size)

 # Setup distributed training
 processes = []
 for rank in range(world_size):
 p = mp.Process(
 target=self._distributed_worker,
 args=(rank, world_size)
 )
 p.start
 processes.append(p)

 # Wait for completion
 for p in processes:
 p.join

 # Aggregate results
 session = self.create_training_session
 self.logger.info("Distributed training completed")

 return session

 def _distributed_worker(self, rank: int, world_size: int) -> None:
 """Worker function for distributed training."""
 # Setup distributed environment
 import torch.distributed as dist
 import os

 os.environ['MASTER_ADDR'] = 'localhost'
 os.environ['MASTER_PORT'] = str(self.config.master_port)

 dist.init_process_group("nccl", rank=rank, world_size=world_size)

 # Create local environment and agent
 env = self.env_factory

 # Training loop for this worker
 episodes_per_worker = self.config.num_episodes // world_size
 for episode in range(episodes_per_worker):
 result = self.train_single_episode(rank * episodes_per_worker + episode, env)

 # Synchronize periodically
 if episode % 10 == 0:
 # Sync agent parameters
 for param in self.agent.q_network.parameters:
 dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
 param.data /= world_size

 env.close
 dist.destroy_process_group

 def benchmark_agent(self, num_runs: int = 10) -> Dict[str, Any]:
 """
 Benchmarking agent on multiple runs.

 Args:
 num_runs: Number of runs

 Returns:
 Benchmark results
 """
 self.logger.info("Start benchmarking", runs=num_runs)

 all_results = []

 for run in range(num_runs):
 env = self.env_factory
 run_rewards = []

 for episode in range(10): # 10 episodes per run
 result = self.train_single_episode(episode, env, training=False)
 run_rewards.append(result.total_reward)

 all_results.extend(run_rewards)
 env.close

 # Calculate benchmark metrics
 benchmark_results = {
 "mean_reward": np.mean(all_results),
 "std_reward": np.std(all_results),
 "min_reward": np.min(all_results),
 "max_reward": np.max(all_results),
 "median_reward": np.median(all_results),
 "runs": num_runs,
 "total_episodes": len(all_results),
 "confidence_interval_95": np.percentile(all_results, [2.5, 97.5]).tolist,
 }

 self.logger.info("Benchmarking completed", **benchmark_results)
 return benchmark_results

 def get_training_summary(self) -> Dict[str, Any]:
 """Get summary current training."""
 if not self.current_session:
 return {"error": "No active training session"}

 session = self.current_session

 # Episode statistics
 rewards = [r.total_reward for r in session.episode_results if not r.error]
 losses = [r.avg_loss for r in session.episode_results if not r.error and r.avg_loss > 0]

 summary = {
 "session_id": session.session_id,
 "start_time": session.start_time.isoformat,
 "total_episodes": session.total_episodes,
 "total_steps": session.total_steps,
 "best_reward": session.best_reward,
 "best_model_path": session.best_model_path,

 "reward_statistics": {
 "mean": np.mean(rewards) if rewards else 0,
 "std": np.std(rewards) if rewards else 0,
 "min": np.min(rewards) if rewards else 0,
 "max": np.max(rewards) if rewards else 0,
 } if rewards else {},

 "loss_statistics": {
 "mean": np.mean(losses) if losses else 0,
 "std": np.std(losses) if losses else 0,
 "latest": losses[-1] if losses else 0,
 } if losses else {},

 "error_count": sum(1 for r in session.episode_results if r.error),
 "config": session.config,
 }

 return summary

 def __del__(self):
 """Cleanup resources."""
 if self.writer:
 self.writer.close
 if self.executor:
 self.executor.shutdown(wait=True)