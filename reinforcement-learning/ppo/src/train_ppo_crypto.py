"""
PPO Training Script for Crypto Trading
Production-ready training pipeline
Includes distributed training, hyperparameter optimization, and monitoring
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import asyncio
import wandb
from torch.utils.tensorboard import SummaryWriter

# Import our PPO components
from agents.advanced_ppo_agent import (
 AdvancedPPOAgent,
 AdvancedPPOConfig,
 DistributedPPOCoordinator,
 PPOPerformanceMonitor
)
from environments.advanced_crypto_env import (
 AdvancedCryptoTradingEnv,
 TradingEnvironmentConfig
)

# ====================================================================
# CONFIGURATION
# ====================================================================

class TrainingConfig:
 """Training configuration"""

 def __init__(self, config_path: Optional[str] = None):
 if config_path and Path(config_path).exists():
 with open(config_path, 'r') as f:
 config_dict = json.load(f)
 self.__dict__.update(config_dict)
 else:
 # Default configuration
 self.experiment_name = f"ppo_crypto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 self.seed = 42
 self.device = "cuda" if torch.cuda.is_available() else "cpu"

 # Training parameters
 self.num_episodes = 10000
 self.episode_length = 1000
 self.warmup_episodes = 100
 self.eval_interval = 100
 self.save_interval = 500

 # Distributed training
 self.use_distributed = False
 self.num_workers = 4

 # Hyperparameter optimization
 self.use_optuna = False
 self.n_trials = 100

 # Logging
 self.use_wandb = False
 self.use_tensorboard = True
 self.log_dir = "./logs"
 self.checkpoint_dir = "./checkpoints"

 # Early stopping
 self.early_stopping = True
 self.patience = 500
 self.min_improvement = 0.01

# ====================================================================
# TRAINING PIPELINE
# ====================================================================

class PPOTrainingPipeline:
 """
 Complete PPO training pipeline
 Production-ready ML training system
 """

 def __init__(self, training_config: TrainingConfig):
 self.config = training_config
 self.setup_logging()
 self.setup_directories()
 self.setup_seeds()

 # Initialize components
 self.ppo_config = self.create_ppo_config()
 self.env_config = self.create_env_config()

 # Create environment and agent
 self.env = AdvancedCryptoTradingEnv(self.env_config)
 obs_dim = self.env.observation_space.shape[0]
 action_dim = self.env.action_space.n

 self.agent = AdvancedPPOAgent(self.ppo_config, obs_dim, action_dim, self.logger)

 # Setup monitoring
 self.setup_monitoring()

 # Training statistics
 self.best_reward = float('-inf')
 self.patience_counter = 0
 self.training_history = []

 def setup_logging(self):
 """Setup logging configuration"""
 log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
 logging.basicConfig(level=logging.INFO, format=log_format)
 self.logger = logging.getLogger(__name__)

 # File handler
 if not Path(self.config.log_dir).exists():
 Path(self.config.log_dir).mkdir(parents=True)

 file_handler = logging.FileHandler(
 Path(self.config.log_dir) / f"{self.config.experiment_name}.log"
 )
 file_handler.setFormatter(logging.Formatter(log_format))
 self.logger.addHandler(file_handler)

 def setup_directories(self):
 """Create necessary directories"""
 directories = [
 self.config.log_dir,
 self.config.checkpoint_dir,
 Path(self.config.checkpoint_dir) / self.config.experiment_name
 ]

 for directory in directories:
 Path(directory).mkdir(parents=True, exist_ok=True)

 def setup_seeds(self):
 """Set random seeds for reproducibility"""
 np.random.seed(self.config.seed)
 torch.manual_seed(self.config.seed)
 if torch.cuda.is_available():
 torch.cuda.manual_seed(self.config.seed)

 def create_ppo_config(self) -> AdvancedPPOConfig:
 """Create PPO configuration"""
 return AdvancedPPOConfig(
 learning_rate=3e-4,
 gamma=0.99,
 gae_lambda=0.95,
 clip_range=0.2,
 use_curiosity=True,
 curiosity_strength=0.01,
 num_assets=5,
 adaptive_lr=True,
 batch_size=256,
 mini_batch_size=64,
 n_epochs=10,
 device=self.config.device,
 tensorboard=self.config.use_tensorboard
 )

 def create_env_config(self) -> TradingEnvironmentConfig:
 """Create environment configuration"""
 return TradingEnvironmentConfig(
 assets=["BTC", "ETH", "BNB", "SOL", "ADA"],
 initial_balance=10000.0,
 max_position_size=0.3,
 max_leverage=3.0,
 reward_type="sharpe",
 max_steps=self.config.episode_length,
 seed=self.config.seed
 )

 def setup_monitoring(self):
 """Setup monitoring tools"""
 # TensorBoard
 if self.config.use_tensorboard:
 tensorboard_dir = Path(self.config.log_dir) / "tensorboard" / self.config.experiment_name
 self.writer = SummaryWriter(tensorboard_dir)
 else:
 self.writer = None

 # Weights & Biases
 if self.config.use_wandb:
 wandb.init(
 project="ppo-crypto-trading",
 name=self.config.experiment_name,
 config={
 "ppo_config": self.ppo_config.__dict__,
 "env_config": self.env_config.__dict__,
 "training_config": self.config.__dict__
 }
 )

 # Performance monitor
 self.monitor = PPOPerformanceMonitor(
 tensorboard_dir=str(Path(self.config.log_dir) / "tensorboard" / self.config.experiment_name)
 if self.config.use_tensorboard else None
 )

 def train_episode(self, episode: int) -> Dict[str, float]:
 """Train one episode"""
 obs = self.env.reset()
 episode_reward = 0
 episode_length = 0
 hidden_state = None

 for step in range(self.config.episode_length):
 # Select action
 action, log_prob, value, hidden_state = self.agent.select_action(
 obs, deterministic=False, hidden_state=hidden_state
 )

 # Execute action
 next_obs, reward, done, info = self.env.step(action)

 # Store transition
 self.agent.store_transition(obs, action, reward, log_prob, value, done)

 # Update counters
 episode_reward += reward
 episode_length += 1
 obs = next_obs

 # Update policy
 if len(self.agent.observations) >= self.ppo_config.batch_size:
 update_metrics = self.agent.update()
 self.monitor.log_metrics(self.agent.total_steps, update_metrics)

 if done:
 break

 # Log episode metrics
 self.monitor.log_episode(episode, episode_reward, episode_length)

 return {
 "episode_reward": episode_reward,
 "episode_length": episode_length,
 "portfolio_value": info.get("portfolio_value", 0),
 "total_return": info.get("total_return", 0),
 "sharpe_ratio": info.get("sharpe_ratio", 0),
 "max_drawdown": info.get("current_drawdown", 0)
 }

 def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
 """Evaluate agent performance"""
 eval_rewards = []
 eval_returns = []
 eval_sharpe = []

 for _ in range(num_episodes):
 obs = self.env.reset()
 episode_reward = 0
 hidden_state = None

 for _ in range(self.config.episode_length):
 # Use deterministic policy for evaluation
 action, _, _, hidden_state = self.agent.select_action(
 obs, deterministic=True, hidden_state=hidden_state
 )

 obs, reward, done, info = self.env.step(action)
 episode_reward += reward

 if done:
 break

 eval_rewards.append(episode_reward)
 eval_returns.append(info.get("total_return", 0))
 eval_sharpe.append(info.get("sharpe_ratio", 0))

 return {
 "mean_reward": np.mean(eval_rewards),
 "std_reward": np.std(eval_rewards),
 "mean_return": np.mean(eval_returns),
 "mean_sharpe": np.mean(eval_sharpe)
 }

 def train(self):
 """Main training loop"""
 self.logger.info(f"Starting training: {self.config.experiment_name}")
 self.logger.info(f"Device: {self.config.device}")
 self.logger.info(f"Episodes: {self.config.num_episodes}")

 progress_bar = tqdm(range(self.config.num_episodes), desc="Training")

 for episode in progress_bar:
 # Train episode
 episode_metrics = self.train_episode(episode)
 self.training_history.append(episode_metrics)

 # Update progress bar
 progress_bar.set_postfix({
 "reward": f"{episode_metrics['episode_reward']:.2f}",
 "return": f"{episode_metrics['total_return']:.2%}",
 "sharpe": f"{episode_metrics['sharpe_ratio']:.2f}"
 })

 # Logging
 if self.config.use_wandb:
 wandb.log(episode_metrics, step=episode)

 if self.writer:
 for key, value in episode_metrics.items():
 self.writer.add_scalar(f"train/{key}", value, episode)

 # Evaluation
 if episode % self.config.eval_interval == 0 and episode > 0:
 eval_metrics = self.evaluate()
 self.logger.info(f"Episode {episode} - Evaluation: {eval_metrics}")

 if self.config.use_wandb:
 wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=episode)

 if self.writer:
 for key, value in eval_metrics.items():
 self.writer.add_scalar(f"eval/{key}", value, episode)

 # Early stopping check
 if self.config.early_stopping:
 if eval_metrics["mean_reward"] > self.best_reward + self.config.min_improvement:
 self.best_reward = eval_metrics["mean_reward"]
 self.patience_counter = 0
 self.save_checkpoint(episode, is_best=True)
 else:
 self.patience_counter += 1

 if self.patience_counter >= self.config.patience:
 self.logger.info(f"Early stopping triggered at episode {episode}")
 break

 # Save checkpoint
 if episode % self.config.save_interval == 0 and episode > 0:
 self.save_checkpoint(episode)

 # Final evaluation
 final_metrics = self.evaluate(num_episodes=50)
 self.logger.info(f"Final evaluation: {final_metrics}")

 # Save final model
 self.save_checkpoint(episode, is_final=True)

 # Close monitoring
 if self.writer:
 self.writer.close()

 if self.config.use_wandb:
 wandb.finish()

 self.monitor.close()

 return self.training_history

 def save_checkpoint(self, episode: int, is_best: bool = False, is_final: bool = False):
 """Save training checkpoint"""
 checkpoint_dir = Path(self.config.checkpoint_dir) / self.config.experiment_name

 if is_best:
 path = checkpoint_dir / "best_model.pt"
 elif is_final:
 path = checkpoint_dir / "final_model.pt"
 else:
 path = checkpoint_dir / f"checkpoint_episode_{episode}.pt"

 self.agent.save(str(path))

 # Save training history
 history_path = checkpoint_dir / "training_history.json"
 with open(history_path, 'w') as f:
 json.dump(self.training_history, f, indent=2, default=str)

 self.logger.info(f"Checkpoint saved: {path}")

 def load_checkpoint(self, path: str):
 """Load training checkpoint"""
 self.agent.load(path)
 self.logger.info(f"Checkpoint loaded: {path}")

# ====================================================================
# HYPERPARAMETER OPTIMIZATION
# ====================================================================

def optimize_hyperparameters(training_config: TrainingConfig):
 """
 Optimize hyperparameters using Optuna
 Automated hyperparameter tuning
 """
 import optuna

 def objective(trial):
 # Suggest hyperparameters
 ppo_config = AdvancedPPOConfig(
 learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
 gamma=trial.suggest_uniform("gamma", 0.9, 0.999),
 gae_lambda=trial.suggest_uniform("gae_lambda", 0.9, 0.99),
 clip_range=trial.suggest_uniform("clip_range", 0.1, 0.3),
 ent_coef=trial.suggest_loguniform("ent_coef", 1e-4, 0.1),
 vf_coef=trial.suggest_uniform("vf_coef", 0.1, 1.0),
 batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
 n_epochs=trial.suggest_int("n_epochs", 5, 20)
 )

 # Create and train agent
 env_config = TradingEnvironmentConfig()
 env = AdvancedCryptoTradingEnv(env_config)

 agent = AdvancedPPOAgent(
 ppo_config,
 env.observation_space.shape[0],
 env.action_space.n
 )

 # Train for limited episodes
 total_reward = 0
 for _ in range(100): # Quick evaluation
 obs = env.reset()
 episode_reward = 0

 for _ in range(500):
 action, log_prob, value, _ = agent.select_action(obs)
 obs, reward, done, _ = env.step(action)
 agent.store_transition(obs, action, reward, log_prob, value, done)
 episode_reward += reward

 if len(agent.observations) >= ppo_config.batch_size:
 agent.update()

 if done:
 break

 total_reward += episode_reward

 return total_reward / 100

 # Create study
 study = optuna.create_study(direction="maximize")
 study.optimize(objective, n_trials=training_config.n_trials)

 # Print best parameters
 print(f"Best parameters: {study.best_params}")
 print(f"Best value: {study.best_value}")

 return study.best_params

# ====================================================================
# MAIN EXECUTION
# ====================================================================

def main():
 """Main execution function"""
 parser = argparse.ArgumentParser(description="PPO Training for Crypto Trading")
 parser.add_argument("--config", type=str, help="Path to configuration file")
 parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "optimize"],
 help="Execution mode")
 parser.add_argument("--checkpoint", type=str, help="Path to checkpoint for evaluation")
 parser.add_argument("--distributed", action="store_true", help="Use distributed training")

 args = parser.parse_args()

 # Load configuration
 training_config = TrainingConfig(args.config)

 if args.distributed:
 training_config.use_distributed = True

 if args.mode == "train":
 # Standard training
 pipeline = PPOTrainingPipeline(training_config)
 history = pipeline.train()

 # Save final results
 results_path = Path(training_config.checkpoint_dir) / training_config.experiment_name / "results.json"
 with open(results_path, 'w') as f:
 json.dump({
 "config": training_config.__dict__,
 "final_metrics": pipeline.monitor.get_statistics(),
 "best_reward": pipeline.best_reward
 }, f, indent=2, default=str)

 elif args.mode == "evaluate":
 # Evaluation mode
 if not args.checkpoint:
 print("Error: Checkpoint path required for evaluation")
 return

 pipeline = PPOTrainingPipeline(training_config)
 pipeline.load_checkpoint(args.checkpoint)

 eval_metrics = pipeline.evaluate(num_episodes=100)
 print(f"Evaluation results: {eval_metrics}")

 elif args.mode == "optimize":
 # Hyperparameter optimization
 training_config.use_optuna = True
 best_params = optimize_hyperparameters(training_config)

 # Train with best parameters
 training_config.__dict__.update(best_params)
 pipeline = PPOTrainingPipeline(training_config)
 pipeline.train()

if __name__ == "__main__":
 main()