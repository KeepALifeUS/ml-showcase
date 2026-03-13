"""
Comprehensive Tests for PPO Implementation
 Enterprise Testing Patterns

Tests cover:
- Core PPO algorithm functionality
- Network architectures
- Buffer operations
- Training workflows
- Crypto trading integration
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import tempfile
import os
from unittest.mock import Mock, patch

from src.core.ppo import PPOAlgorithm, PPOConfig, PPOLoss
from src.core.ppo2 import PPO2Algorithm, PPO2Config
from src.networks.actor_critic import ActorCriticNetwork, ActorCriticConfig
from src.networks.policy_network import PolicyNetwork
from src.networks.value_network import ValueNetwork
from src.buffers.rollout_buffer import RolloutBuffer, RolloutBufferConfig
from src.buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferConfig
from src.advantages.gae import GAE, GAEConfig
from src.advantages.td_lambda import TDLambdaEstimator, TDLambdaConfig
from src.optimization.clipped_objective import StandardClippedObjective, ClippedObjectiveConfig
from src.optimization.kl_penalty import StandardKLPenalty, KLPenaltyConfig
from src.utils.normalization import RunningMeanStd, normalize_advantages
from src.utils.scheduling import LinearSchedule, ExponentialSchedule, PPOScheduler
from src.training.ppo_trainer import PPOTrainer, PPOTrainerConfig
from src.agents.ppo_trader import PPOTrader, PPOTraderConfig
from src.environments.crypto_env import CryptoTradingEnvironment, CryptoEnvConfig


class TestPPOCore:
 """Test core PPO algorithm components"""
 
 def test_ppo_config_creation(self):
 """Test PPO configuration creation"""
 config = PPOConfig()
 assert config.learning_rate == 3e-4
 assert config.gamma == 0.99
 assert config.gae_lambda == 0.95
 assert config.clip_range == 0.2
 
 def test_ppo_loss_computation(self):
 """Test PPO loss computation"""
 config = PPOConfig()
 loss_fn = PPOLoss(config)
 
 # Mock data
 batch_size = 32
 log_probs = torch.randn(batch_size)
 old_log_probs = torch.randn(batch_size)
 advantages = torch.randn(batch_size)
 
 # Compute loss
 policy_loss, metrics = loss_fn.compute_policy_loss(
 log_probs, old_log_probs, advantages
 )
 
 assert isinstance(policy_loss, torch.Tensor)
 assert policy_loss.requires_grad
 assert "approx_kl" in metrics
 assert "clip_fraction" in metrics
 assert "policy_loss" in metrics
 
 def test_ppo_value_loss(self):
 """Test value function loss"""
 config = PPOConfig(clip_range_vf=0.2)
 loss_fn = PPOLoss(config)
 
 batch_size = 32
 values = torch.randn(batch_size)
 old_values = torch.randn(batch_size)
 returns = torch.randn(batch_size)
 
 value_loss, metrics = loss_fn.compute_value_loss(
 values, old_values, returns
 )
 
 assert isinstance(value_loss, torch.Tensor)
 assert value_loss.requires_grad
 assert "value_loss" in metrics
 assert "explained_variance" in metrics
 
 @pytest.fixture
 def mock_actor_critic(self):
 """Create mock actor-critic network"""
 config = ActorCriticConfig(
 obs_dim=64,
 action_dim=4,
 action_type="continuous"
 )
 return ActorCriticNetwork(config)
 
 def test_ppo_algorithm_initialization(self, mock_actor_critic):
 """Test PPO algorithm initialization"""
 config = PPOConfig()
 ppo = PPOAlgorithm(mock_actor_critic, config)
 
 assert ppo.config == config
 assert ppo.actor_critic == mock_actor_critic
 assert ppo.optimizer is not None
 assert ppo.update_count == 0
 
 def test_ppo_update_step(self, mock_actor_critic):
 """Test PPO update step"""
 config = PPOConfig(n_epochs=2, batch_size=16)
 ppo = PPOAlgorithm(mock_actor_critic, config)
 
 # Create mock rollout buffer
 buffer_config = RolloutBufferConfig(
 buffer_size=32,
 batch_size=16,
 observation_space=(64,),
 action_space=(4,)
 )
 buffer = RolloutBuffer(buffer_config)
 
 # Fill buffer with mock data
 for _ in range(32):
 buffer.add(
 obs=torch.randn(64),
 action=torch.randn(4),
 reward=0.1,
 value=0.5,
 log_prob=-1.0,
 done=False
 )
 
 buffer.compute_returns_and_advantages()
 
 # Perform update
 metrics = ppo.update(buffer)
 
 assert isinstance(metrics, dict)
 assert "policy_loss" in metrics
 assert "value_loss" in metrics
 assert "entropy" in metrics
 assert ppo.update_count == 1


class TestPPO2:
 """Test PPO2 enhanced implementation"""
 
 def test_ppo2_config_creation(self):
 """Test PPO2 configuration"""
 config = PPO2Config()
 assert hasattr(config, 'adaptive_clipping')
 assert hasattr(config, 'use_multi_step')
 assert hasattr(config, 'mixed_precision')
 
 @pytest.fixture
 def mock_actor_critic_ppo2(self):
 """Create mock actor-critic for PPO2"""
 config = ActorCriticConfig(
 obs_dim=64,
 action_dim=4,
 action_type="continuous"
 )
 return ActorCriticNetwork(config)
 
 def test_ppo2_algorithm_initialization(self, mock_actor_critic_ppo2):
 """Test PPO2 initialization"""
 config = PPO2Config()
 ppo2 = PPO2Algorithm(mock_actor_critic_ppo2, config)
 
 assert isinstance(ppo2, PPO2Algorithm)
 assert ppo2.config == config


class TestNetworks:
 """Test network architectures"""
 
 def test_actor_critic_network_creation(self):
 """Test actor-critic network creation"""
 config = ActorCriticConfig(
 obs_dim=64,
 action_dim=4,
 action_type="continuous",
 shared_backbone=True
 )
 
 network = ActorCriticNetwork(config)
 
 assert hasattr(network, 'shared_backbone')
 assert hasattr(network, 'policy_head')
 assert hasattr(network, 'value_head')
 
 def test_actor_critic_forward_pass(self):
 """Test actor-critic forward pass"""
 config = ActorCriticConfig(
 obs_dim=64,
 action_dim=4,
 action_type="continuous"
 )
 
 network = ActorCriticNetwork(config)
 observations = torch.randn(32, 64)
 
 action_dist, values = network(observations)
 
 assert hasattr(action_dist, 'sample')
 assert values.shape == (32, 1)
 
 def test_policy_network_discrete(self):
 """Test discrete policy network"""
 network = PolicyNetwork(
 input_dim=64,
 action_dim=4,
 action_type="discrete"
 )
 
 observations = torch.randn(32, 64)
 action_dist = network(observations)
 
 assert hasattr(action_dist, 'sample')
 assert hasattr(action_dist, 'log_prob')
 
 def test_policy_network_continuous(self):
 """Test continuous policy network"""
 network = PolicyNetwork(
 input_dim=64,
 action_dim=4,
 action_type="continuous"
 )
 
 observations = torch.randn(32, 64)
 action_dist = network(observations)
 
 assert hasattr(action_dist, 'mean')
 assert hasattr(action_dist, 'scale')
 
 def test_value_network(self):
 """Test value network"""
 network = ValueNetwork(
 input_dim=64,
 network_type="standard"
 )
 
 observations = torch.randn(32, 64)
 values = network(observations)
 
 assert values.shape == (32, 1)


class TestBuffers:
 """Test buffer implementations"""
 
 def test_rollout_buffer_creation(self):
 """Test rollout buffer creation"""
 config = RolloutBufferConfig(
 buffer_size=128,
 batch_size=32,
 observation_space=(64,),
 action_space=(4,)
 )
 
 buffer = RolloutBuffer(config)
 
 assert buffer.buffer_size == 128
 assert buffer.batch_size == 32
 assert buffer.pos == 0
 assert not buffer.full
 
 def test_rollout_buffer_add_data(self):
 """Test adding data to rollout buffer"""
 config = RolloutBufferConfig(
 buffer_size=10,
 observation_space=(4,),
 action_space=(2,)
 )
 buffer = RolloutBuffer(config)
 
 for i in range(5):
 buffer.add(
 obs=torch.randn(4),
 action=torch.randn(2),
 reward=float(i),
 value=0.5,
 log_prob=-1.0,
 done=i == 4
 )
 
 assert buffer.pos == 5
 assert not buffer.full
 assert buffer.rewards[0] == 0.0
 assert buffer.rewards[4] == 4.0
 
 def test_rollout_buffer_gae_computation(self):
 """Test GAE computation in buffer"""
 config = RolloutBufferConfig(
 buffer_size=10,
 observation_space=(4,),
 action_space=(2,)
 )
 buffer = RolloutBuffer(config)
 
 # Fill buffer
 for i in range(10):
 buffer.add(
 obs=torch.randn(4),
 action=torch.randn(2),
 reward=0.1,
 value=0.5,
 log_prob=-1.0,
 done=False
 )
 
 # Compute advantages
 buffer.compute_returns_and_advantages()
 
 assert buffer.advantages_computed
 assert buffer.advantages.shape == (10,)
 assert buffer.returns.shape == (10,)
 
 def test_trajectory_buffer_creation(self):
 """Test trajectory buffer"""
 config = TrajectoryBufferConfig(max_trajectories=100)
 buffer = TrajectoryBuffer(config)
 
 assert len(buffer.trajectories) == 0
 assert len(buffer.incomplete_trajectories) == 0
 
 def test_trajectory_buffer_workflow(self):
 """Test complete trajectory workflow"""
 config = TrajectoryBufferConfig(max_trajectories=10)
 buffer = TrajectoryBuffer(config)
 
 # Start trajectory
 traj_id = buffer.start_trajectory()
 assert traj_id in buffer.incomplete_trajectories
 
 # Add steps
 for i in range(5):
 buffer.add_step(
 trajectory_id=traj_id,
 obs=torch.randn(4),
 action=torch.randn(2),
 reward=0.1 * i,
 value=0.5,
 log_prob=-1.0,
 done=i == 4
 )
 
 # Complete trajectory
 buffer.complete_trajectory(traj_id)
 
 assert len(buffer.trajectories) == 1
 assert traj_id not in buffer.incomplete_trajectories


class TestAdvantageEstimation:
 """Test advantage estimation methods"""
 
 def test_gae_config(self):
 """Test GAE configuration"""
 config = GAEConfig()
 assert config.gamma == 0.99
 assert config.gae_lambda == 0.95
 
 def test_gae_computation(self):
 """Test GAE advantage computation"""
 config = GAEConfig()
 gae = GAE(config)
 
 # Mock data
 seq_len, batch_size = 10, 1
 rewards = torch.randn(seq_len, batch_size)
 values = torch.randn(seq_len, batch_size)
 dones = torch.zeros(seq_len, batch_size, dtype=torch.bool)
 
 advantages, returns = gae.compute_advantages_and_returns(
 rewards, values, dones
 )
 
 assert advantages.shape == (seq_len, batch_size)
 assert returns.shape == (seq_len, batch_size)
 
 def test_td_lambda_estimator(self):
 """Test TD(λ) estimator"""
 config = TDLambdaConfig()
 estimator = TDLambdaEstimator(config)
 
 seq_len, batch_size = 10, 1
 rewards = torch.randn(seq_len, batch_size)
 values = torch.randn(seq_len, batch_size)
 dones = torch.zeros(seq_len, batch_size, dtype=torch.bool)
 
 returns, advantages = estimator.compute_lambda_returns(
 rewards, values, dones
 )
 
 assert returns.shape == (seq_len, batch_size)
 assert advantages.shape == (seq_len, batch_size)


class TestOptimization:
 """Test optimization components"""
 
 def test_clipped_objective(self):
 """Test clipped surrogate objective"""
 config = ClippedObjectiveConfig()
 objective = StandardClippedObjective(config)
 
 batch_size = 32
 log_probs = torch.randn(batch_size)
 old_log_probs = torch.randn(batch_size)
 advantages = torch.randn(batch_size)
 
 loss, metrics = objective.compute_policy_loss(
 log_probs, old_log_probs, advantages
 )
 
 assert isinstance(loss, torch.Tensor)
 assert "clip_fraction" in metrics
 
 def test_kl_penalty(self):
 """Test KL penalty method"""
 config = KLPenaltyConfig()
 kl_penalty = StandardKLPenalty(config)
 
 # Mock distributions
 from torch.distributions import Normal
 
 batch_size = 32
 action_dim = 4
 
 old_dist = Normal(
 torch.zeros(batch_size, action_dim),
 torch.ones(batch_size, action_dim)
 )
 new_dist = Normal(
 torch.randn(batch_size, action_dim) * 0.1,
 torch.ones(batch_size, action_dim)
 )
 
 penalty, metrics = kl_penalty.compute_kl_penalty(new_dist, old_dist)
 
 assert isinstance(penalty, torch.Tensor)
 assert "kl_divergence" in metrics


class TestUtilities:
 """Test utility functions"""
 
 def test_running_mean_std(self):
 """Test running mean/std computation"""
 rms = RunningMeanStd(shape=(4,))
 
 # Update with data
 for _ in range(100):
 data = torch.randn(10, 4)
 rms.update_batch(data)
 
 assert rms.count > 0
 assert rms.mean.shape == (4,)
 assert rms.var.shape == (4,)
 
 # Test normalization
 test_data = torch.randn(5, 4)
 normalized = rms.normalize(test_data)
 
 assert normalized.shape == test_data.shape
 
 def test_advantage_normalization(self):
 """Test advantage normalization"""
 advantages = torch.randn(100)
 normalized = normalize_advantages(advantages)
 
 assert normalized.shape == advantages.shape
 assert abs(normalized.mean().item()) < 1e-6
 assert abs(normalized.std().item() - 1.0) < 1e-6
 
 def test_linear_schedule(self):
 """Test linear parameter scheduling"""
 scheduler = LinearSchedule(1.0, 0.1)
 
 assert scheduler.value(0.0) == 1.0
 assert scheduler.value(1.0) == 0.1
 assert 0.1 < scheduler.value(0.5) < 1.0
 
 def test_exponential_schedule(self):
 """Test exponential scheduling"""
 scheduler = ExponentialSchedule(1.0, 0.99)
 
 val_0 = scheduler.value(0.0)
 val_1 = scheduler.value(1.0)
 
 assert val_0 > val_1
 assert val_0 == 1.0
 
 def test_ppo_scheduler(self):
 """Test PPO-specific scheduler"""
 scheduler = PPOScheduler(
 total_steps=1000,
 lr_schedule="linear",
 initial_lr=3e-4,
 final_lr=3e-5
 )
 
 values = scheduler.get_values(0.0)
 assert "learning_rate" in values
 assert "clip_range" in values
 assert "entropy_coef" in values
 
 assert values["learning_rate"] == 3e-4


class TestTraining:
 """Test training components"""
 
 @pytest.fixture
 def mock_env(self):
 """Create mock environment"""
 env = Mock()
 env.observation_space.shape = (64,)
 env.action_space.shape = (4,)
 env.reset.return_value = (np.random.randn(64), {})
 env.step.return_value = (
 np.random.randn(64), # obs
 0.1, # reward
 False, # done
 False, # truncated
 {} # info
 )
 return env
 
 def test_ppo_trainer_config(self):
 """Test PPO trainer configuration"""
 config = PPOTrainerConfig(
 total_timesteps=10000,
 rollout_steps=128,
 batch_size=32
 )
 
 assert config.total_timesteps == 10000
 assert config.rollout_steps == 128
 assert config.batch_size == 32
 
 @patch('src.training.ppo_trainer.PPOTrainer._create_environments')
 def test_ppo_trainer_initialization(self, mock_create_envs, mock_env):
 """Test PPO trainer initialization"""
 mock_create_envs.return_value = [mock_env]
 
 config = PPOTrainerConfig(
 total_timesteps=1000,
 num_envs=1
 )
 
 trainer = PPOTrainer(config)
 
 assert trainer.config == config
 assert trainer.num_envs == 1
 assert trainer.actor_critic is not None
 assert trainer.optimizer is not None


class TestCryptoTrading:
 """Test crypto trading components"""
 
 def test_crypto_env_config(self):
 """Test crypto environment configuration"""
 config = CryptoEnvConfig()
 assert config.initial_balance == 10000.0
 assert len(config.assets) > 0
 
 def test_crypto_env_creation(self):
 """Test crypto environment creation"""
 config = CryptoEnvConfig(
 assets=["BTC", "ETH"],
 max_steps=100
 )
 env = CryptoTradingEnvironment(config)
 
 assert len(env.config.assets) == 2
 assert env.config.max_steps == 100
 assert hasattr(env, 'action_space')
 assert hasattr(env, 'observation_space')
 
 def test_crypto_env_reset(self):
 """Test environment reset"""
 config = CryptoEnvConfig(assets=["BTC"], max_steps=50)
 env = CryptoTradingEnvironment(config)
 
 obs, info = env.reset()
 
 assert isinstance(obs, np.ndarray)
 assert obs.shape == env.observation_space.shape
 assert isinstance(info, dict)
 assert env.current_step == 0
 assert env.balance == config.initial_balance
 
 def test_crypto_env_step(self):
 """Test environment step"""
 config = CryptoEnvConfig(assets=["BTC"], max_steps=50)
 env = CryptoTradingEnvironment(config)
 
 env.reset()
 action = env.action_space.sample()
 
 obs, reward, done, truncated, info = env.step(action)
 
 assert isinstance(obs, np.ndarray)
 assert isinstance(reward, (int, float))
 assert isinstance(done, bool)
 assert isinstance(info, dict)
 assert env.current_step == 1
 
 def test_ppo_trader_config(self):
 """Test PPO trader configuration"""
 config = PPOTraderConfig(
 assets=["BTC", "ETH"],
 max_position_size=0.5
 )
 
 assert len(config.assets) == 2
 assert config.max_position_size == 0.5
 
 @patch('src.agents.ppo_trader.PPOTrader.load_model')
 def test_ppo_trader_initialization(self, mock_load):
 """Test PPO trader initialization"""
 config = PPOTraderConfig(assets=["BTC"])
 trader = PPOTrader(config)
 
 assert trader.config == config
 assert trader.actor_critic is not None
 assert trader.position_manager is not None
 
 def test_ppo_trader_market_data_update(self):
 """Test market data update"""
 config = PPOTraderConfig(assets=["BTC"])
 trader = PPOTrader(config)
 
 market_data = {
 "price": 50000.0,
 "volume": 1000.0,
 "timestamp": 1234567890
 }
 
 trader.update_market_data("BTC", market_data)
 
 assert len(trader.market_data["BTC"]) == 1
 assert trader.market_data["BTC"][-1]["price"] == 50000.0


class TestIntegration:
 """Integration tests"""
 
 def test_end_to_end_training_setup(self):
 """Test end-to-end training setup"""
 # Environment config
 env_config = CryptoEnvConfig(
 assets=["BTC"],
 max_steps=100
 )
 
 # Trainer config
 trainer_config = PPOTrainerConfig(
 total_timesteps=1000,
 rollout_steps=64,
 batch_size=16,
 num_envs=1
 )
 
 # Network config
 ac_config = ActorCriticConfig(
 obs_dim=100, # Estimated
 action_dim=1,
 action_type="continuous"
 )
 
 # Create components
 env = CryptoTradingEnvironment(env_config)
 network = ActorCriticNetwork(ac_config)
 
 # Basic functionality check
 obs, _ = env.reset()
 assert obs is not None
 
 with torch.no_grad():
 obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
 action_dist, values = network(obs_tensor)
 action = action_dist.sample()
 
 assert action is not None
 assert values is not None
 
 def test_model_save_load(self):
 """Test model saving and loading"""
 config = PPOTraderConfig(assets=["BTC"])
 trader = PPOTrader(config)
 
 # Save model
 with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
 model_path = f.name
 
 try:
 trader.save_model(model_path)
 assert os.path.exists(model_path)
 
 # Load model
 new_trader = PPOTrader(config)
 new_trader.load_model(model_path)
 
 # Basic check that models are similar
 # (More comprehensive checks would compare state dicts)
 assert isinstance(new_trader.actor_critic, nn.Module)
 
 finally:
 if os.path.exists(model_path):
 os.unlink(model_path)


# Performance benchmarks
class TestPerformance:
 """Performance and benchmarking tests"""
 
 def test_forward_pass_performance(self):
 """Test forward pass performance"""
 config = ActorCriticConfig(
 obs_dim=256,
 action_dim=32,
 action_type="continuous"
 )
 network = ActorCriticNetwork(config)
 
 batch_size = 1000
 observations = torch.randn(batch_size, 256)
 
 # Warmup
 for _ in range(10):
 with torch.no_grad():
 _, _ = network(observations)
 
 # Benchmark
 import time
 start_time = time.time()
 
 for _ in range(100):
 with torch.no_grad():
 _, _ = network(observations)
 
 elapsed = time.time() - start_time
 ops_per_second = (100 * batch_size) / elapsed
 
 # Should process at least 10k observations per second
 assert ops_per_second > 10000, f"Too slow: {ops_per_second:.0f} ops/sec"
 
 def test_buffer_memory_usage(self):
 """Test buffer memory efficiency"""
 config = RolloutBufferConfig(
 buffer_size=10000,
 observation_space=(128,),
 action_space=(8,)
 )
 
 import psutil
 process = psutil.Process()
 
 # Baseline memory
 baseline_memory = process.memory_info().rss
 
 # Create buffer
 buffer = RolloutBuffer(config)
 
 # Fill buffer
 for i in range(config.buffer_size):
 buffer.add(
 obs=torch.randn(128),
 action=torch.randn(8),
 reward=0.1,
 value=0.5,
 log_prob=-1.0,
 done=False
 )
 
 # Check memory usage
 final_memory = process.memory_info().rss
 memory_increase = final_memory - baseline_memory
 
 # Memory increase should be reasonable (less than 100MB)
 memory_mb = memory_increase / (1024 * 1024)
 assert memory_mb < 100, f"Memory usage too high: {memory_mb:.1f} MB"


if __name__ == "__main__":
 # Run tests
 pytest.main([__file__, "-v", "--tb=short"])