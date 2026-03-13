"""
Comprehensive tests for DQN implementation.

Covers:
- Core DQN functionality
- Network architectures
- Experience replay
- Training process
- Performance benchmarks
"""

import pytest
import numpy as np
import torch
import gymnasium as gym
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
from pathlib import Path

# Import DQN components
from ml_dqn import (
 DQN, DQNConfig, QNetworkConfig,
 DoubleDQN, DoubleDQNConfig,
 DuelingNetwork, DuelingNetworkConfig,
 ReplayBuffer, PrioritizedReplayBuffer,
 DQNTrader, CryptoTradingDQNConfig,
 PerformanceMetrics
)


class TestQNetwork:
 """Test Q-Network architecture."""

 def test_qnetwork_creation(self):
 """Test creation Q-Network."""
 config = QNetworkConfig(
 state_size=4,
 action_size=2,
 hidden_layers=[64, 32]
 )

 network = from ml_dqn.networks import QNetwork
 net = QNetwork(config)

 assert net.state_size == 4
 assert net.action_size == 2
 assert len(list(net.parameters)) > 0

 def test_qnetwork_forward(self):
 """Test forward pass."""
 config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[32])
 from ml_dqn.networks import QNetwork
 net = QNetwork(config)

 state = torch.randn(1, 4)
 q_values = net(state)

 assert q_values.shape == (1, 2)
 assert not torch.isnan(q_values).any
 assert not torch.isinf(q_values).any

 def test_qnetwork_batch_processing(self):
 """Test batch processing."""
 config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[32])
 from ml_dqn.networks import QNetwork
 net = QNetwork(config)

 # Batch input
 batch_state = torch.randn(10, 4)
 q_values = net(batch_state)

 assert q_values.shape == (10, 2)

 def test_qnetwork_gradient_flow(self):
 """Test gradient flow."""
 config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[32])
 from ml_dqn.networks import QNetwork
 net = QNetwork(config)

 state = torch.randn(1, 4, requires_grad=True)
 q_values = net(state)
 loss = q_values.sum
 loss.backward

 # Check gradients exist
 for param in net.parameters:
 if param.requires_grad:
 assert param.grad is not None


class TestReplayBuffer:
 """Test Experience Replay Buffer."""

 def test_replay_buffer_creation(self):
 """Test creation replay buffer."""
 buffer = ReplayBuffer(capacity=1000, batch_size=32)

 assert len(buffer) == 0
 assert buffer.config.capacity == 1000
 assert buffer.config.batch_size == 32

 def test_experience_storage(self):
 """Test saving experience."""
 buffer = ReplayBuffer(capacity=100, batch_size=10)

 state = np.random.randn(4)
 next_state = np.random.randn(4)

 buffer.push(state, 1, 1.0, next_state, False)

 assert len(buffer) == 1

 def test_batch_sampling(self):
 """Test sampling batches."""
 buffer = ReplayBuffer(capacity=100, batch_size=10)

 # Add experiences
 for i in range(50):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 buffer.push(state, i % 2, np.random.randn, next_state, i == 49)

 # Sample batch
 batch = buffer.sample
 assert len(batch) == 10

 # Check structure
 for exp in batch:
 assert hasattr(exp, 'state')
 assert hasattr(exp, 'action')
 assert hasattr(exp, 'reward')
 assert hasattr(exp, 'next_state')
 assert hasattr(exp, 'done')

 def test_circular_buffer_behavior(self):
 """Test circular buffer overflow."""
 buffer = ReplayBuffer(capacity=10, batch_size=5)

 # Fill beyond capacity
 for i in range(15):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 buffer.push(state, 0, 1.0, next_state, False)

 assert len(buffer) == 10 # Should not exceed capacity


class TestDQN:
 """Test core DQN algorithm."""

 @pytest.fixture
 def dqn_config(self):
 """DQN configuration fixture."""
 network_config = QNetworkConfig(
 state_size=4,
 action_size=2,
 hidden_layers=[32, 16]
 )

 return DQNConfig(
 network_config=network_config,
 learning_rate=1e-3,
 gamma=0.99,
 buffer_size=1000,
 batch_size=32,
 min_replay_size=100
 )

 def test_dqn_creation(self, dqn_config):
 """Test creation DQN agent."""
 agent = DQN(dqn_config)

 assert agent.config == dqn_config
 assert agent.training_step == 0
 assert len(agent.loss_history) == 0

 def test_action_selection(self, dqn_config):
 """Test selection actions."""
 agent = DQN(dqn_config)
 state = np.random.randn(4)

 # Training mode (epsilon-greedy)
 action = agent.act(state, training=True)
 assert isinstance(action, int)
 assert 0 <= action < 2

 # Evaluation mode (greedy)
 action = agent.act(state, training=False)
 assert isinstance(action, int)
 assert 0 <= action < 2

 def test_experience_storage(self, dqn_config):
 """Test saving experience."""
 agent = DQN(dqn_config)

 state = np.random.randn(4)
 next_state = np.random.randn(4)

 agent.store_experience(state, 1, 1.0, next_state, False)

 assert len(agent.replay_buffer) == 1

 def test_training_step(self, dqn_config):
 """Test training step."""
 agent = DQN(dqn_config)

 # Add sufficient experiences
 for i in range(dqn_config.min_replay_size + 10):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 agent.store_experience(state, i % 2, np.random.randn, next_state, i == dqn_config.min_replay_size + 9)

 # Training step
 metrics = agent.train_step

 assert isinstance(metrics, dict)
 assert 'loss' in metrics
 assert 'training_step' in metrics
 assert agent.training_step == 1

 def test_target_network_updates(self, dqn_config):
 """Test update target network."""
 dqn_config.target_update_freq = 5
 agent = DQN(dqn_config)

 # Fill buffer
 for i in range(dqn_config.min_replay_size + 10):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 agent.store_experience(state, i % 2, np.random.randn, next_state, False)

 # Initial target parameters
 initial_target_params = [p.clone for p in agent.target_network.parameters]

 # Train until target update
 for _ in range(6):
 agent.train_step

 # Check if target network updated
 updated_target_params = list(agent.target_network.parameters)

 # Some parameters should have changed
 params_changed = any(
 not torch.equal(init_p, upd_p)
 for init_p, upd_p in zip(initial_target_params, updated_target_params)
 )
 assert params_changed

 def test_model_saving_loading(self, dqn_config):
 """Test saving and loading model."""
 agent = DQN(dqn_config)

 # Train a bit
 for i in range(dqn_config.min_replay_size + 10):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 agent.store_experience(state, i % 2, np.random.randn, next_state, False)

 for _ in range(5):
 agent.train_step

 # Save model
 with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
 checkpoint_path = f.name

 agent.save_checkpoint(checkpoint_path)

 # Load model
 loaded_agent = DQN.load_checkpoint(checkpoint_path)

 assert loaded_agent.training_step == agent.training_step
 assert len(loaded_agent.loss_history) == len(agent.loss_history)

 # Cleanup
 Path(checkpoint_path).unlink


class TestDoubleDQN:
 """Test Double DQN implementation."""

 def test_double_dqn_creation(self):
 """Test creation Double DQN."""
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[32])
 config = DoubleDQNConfig(network_config=network_config, buffer_size=1000)

 agent = DoubleDQN(config)

 assert agent.double_config == config
 assert agent.double_q_updates == 0

 def test_double_q_training(self):
 """Test double Q training logic."""
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[32])
 config = DoubleDQNConfig(
 network_config=network_config,
 buffer_size=1000,
 min_replay_size=50,
 double_q_freq=1
 )

 agent = DoubleDQN(config)

 # Add experiences
 for i in range(60):
 state = np.random.randn(4)
 next_state = np.random.randn(4)
 agent.store_experience(state, i % 2, np.random.randn, next_state, False)

 # Training should use double Q
 metrics = agent.train_step

 assert 'double_q_ratio' in metrics
 assert agent.double_q_updates > 0


class TestCryptoTrading:
 """Test crypto trading specific functionality."""

 def test_trading_config_creation(self):
 """Test creation trading config."""
 from ml_dqn.agents.dqn_trader import TradingEnvironmentConfig

 config = TradingEnvironmentConfig(
 symbols=["BTCUSDT", "ETHUSDT"],
 initial_balance=10000.0
 )

 assert len(config.symbols) == 2
 assert config.initial_balance == 10000.0

 def test_market_data_structure(self):
 """Test structure market data."""
 from ml_dqn.agents.dqn_trader import MarketData

 data = MarketData(
 timestamp=datetime.now,
 symbol="BTCUSDT",
 open=45000, high=46000, low=44500, close=45500,
 volume=1000
 )

 assert data.symbol == "BTCUSDT"
 assert data.close == 45500
 assert isinstance(data.timestamp, datetime)

 def test_portfolio_state(self):
 """Test portfolio state."""
 from ml_dqn.agents.dqn_trader import PortfolioState

 portfolio = PortfolioState(
 cash_balance=10000.0,
 positions={"BTCUSDT": 0.1},
 total_value=10500.0,
 unrealized_pnl=500.0,
 realized_pnl=0.0
 )

 assert portfolio.cash_balance == 10000.0
 assert portfolio.positions["BTCUSDT"] == 0.1

 def test_trading_agent_creation(self):
 """Test creation trading agent."""
 network_config = QNetworkConfig(
 state_size=100, # Will be recalculated
 action_size=10,
 hidden_layers=[64, 32]
 )

 from ml_dqn.agents.dqn_trader import TradingEnvironmentConfig
 trading_config = TradingEnvironmentConfig(
 symbols=["BTCUSDT", "ETHUSDT"],
 initial_balance=10000.0
 )

 config = CryptoTradingDQNConfig(
 network_config=network_config,
 trading_config=trading_config
 )

 trader = DQNTrader(config)

 assert len(trader.trading_config.symbols) == 2
 assert trader.action_size == 10 # 5 actions × 2 symbols


class TestPerformanceMetrics:
 """Test performance metrics calculation."""

 def test_metrics_creation(self):
 """Test creation metrics."""
 metrics = PerformanceMetrics

 assert len(metrics.episodes) == 0
 assert len(metrics.returns) == 0

 def test_episode_addition(self):
 """Test adding episodes."""
 metrics = PerformanceMetrics

 metrics.add_episode(
 episode_id=1,
 total_reward=100.0,
 episode_length=50,
 actions_taken=[0, 1, 0, 1]
 )

 assert len(metrics.episodes) == 1
 assert metrics.episodes[0].total_reward == 100.0

 def test_basic_metrics_calculation(self):
 """Test calculation basic metrics."""
 metrics = PerformanceMetrics

 # Add some episodes
 rewards = [100, 150, 80, 200, 120]
 for i, reward in enumerate(rewards):
 metrics.add_episode(i, reward, 50)

 basic_metrics = metrics.get_basic_metrics

 assert basic_metrics['mean_reward'] == np.mean(rewards)
 assert basic_metrics['total_episodes'] == 5
 assert basic_metrics['success_rate'] == 1.0 # All positive rewards

 def test_financial_metrics(self):
 """Test financial metrics."""
 metrics = PerformanceMetrics

 # Add episodes with returns
 returns = [0.02, -0.01, 0.03, -0.005, 0.015, 0.01, -0.02]
 for i, ret in enumerate(returns):
 metrics.add_episode(i, ret * 100, 50)
 metrics.returns.append(ret)

 financial_metrics = metrics.get_financial_metrics

 assert 'sharpe_ratio' in financial_metrics
 assert 'max_drawdown' in financial_metrics
 assert 'volatility' in financial_metrics

 def test_report_generation(self):
 """Test generation report."""
 metrics = PerformanceMetrics

 # Add sample data
 for i in range(100):
 reward = np.random.randn * 10 + 50
 metrics.add_episode(i, reward, 50)

 report = metrics.generate_report

 assert 'basic_metrics' in report
 assert 'financial_metrics' in report
 assert 'generated_at' in report


class TestIntegration:
 """Integration tests."""

 @pytest.mark.slow
 def test_cartpole_training(self):
 """Test training on CartPole."""
 # Simplified CartPole training test
 network_config = QNetworkConfig(
 state_size=4,
 action_size=2,
 hidden_layers=[32]
 )

 config = DQNConfig(
 network_config=network_config,
 learning_rate=1e-3,
 buffer_size=1000,
 batch_size=16,
 min_replay_size=50
 )

 agent = DQN(config)

 # Mock environment
 env = Mock
 env.reset.return_value = np.random.randn(4)
 env.step.return_value = (np.random.randn(4), 1.0, False, {})

 # Training loop
 total_reward = 0
 state = env.reset

 for step in range(100):
 action = agent.act(state, training=True)
 next_state, reward, done, _ = env.step(action)

 agent.store_experience(state, action, reward, next_state, done)

 if step > config.min_replay_size:
 metrics = agent.train_step
 assert isinstance(metrics, dict)

 state = next_state
 total_reward += reward

 if done:
 state = env.reset
 total_reward = 0

 # Agent should have learned something
 assert agent.training_step > 0
 assert len(agent.loss_history) > 0

 def test_component_integration(self):
 """Test integration all components."""
 # Test that all major components work together
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[16])

 # Create different DQN variants
 dqn_config = DQNConfig(network_config=network_config, buffer_size=100, min_replay_size=20)
 double_config = DoubleDQNConfig(network_config=network_config, buffer_size=100, min_replay_size=20)

 dqn = DQN(dqn_config)
 double_dqn = DoubleDQN(double_config)

 # Both should work with same interface
 state = np.random.randn(4)

 action1 = dqn.act(state)
 action2 = double_dqn.act(state)

 assert isinstance(action1, int)
 assert isinstance(action2, int)

 # Both should store experiences
 next_state = np.random.randn(4)
 dqn.store_experience(state, action1, 1.0, next_state, False)
 double_dqn.store_experience(state, action2, 1.0, next_state, False)

 assert len(dqn.replay_buffer) == 1
 assert len(double_dqn.replay_buffer) == 1


class TestEdgeCases:
 """Test edge cases and error handling."""

 def test_invalid_config(self):
 """Test invalid configurations."""
 with pytest.raises(ValueError):
 # Invalid state size
 QNetworkConfig(state_size=0, action_size=2)

 with pytest.raises(ValueError):
 # Invalid action size
 QNetworkConfig(state_size=4, action_size=0)

 def test_empty_buffer_training(self):
 """Test training with empty buffer."""
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[16])
 config = DQNConfig(network_config=network_config, min_replay_size=100)

 agent = DQN(config)

 # Should return insufficient data
 metrics = agent.train_step
 assert metrics['status'] == 'insufficient_data'

 def test_nan_handling(self):
 """Test handling NaN values."""
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[16])
 config = DQNConfig(network_config=network_config)

 agent = DQN(config)

 # State with NaN
 state_with_nan = np.array([1.0, 2.0, np.nan, 4.0])

 # Should not crash
 action = agent.act(state_with_nan, training=False)
 assert isinstance(action, int)

 def test_extreme_values(self):
 """Test extreme input values."""
 network_config = QNetworkConfig(state_size=4, action_size=2, hidden_layers=[16])
 config = DQNConfig(network_config=network_config)

 agent = DQN(config)

 # Very large values
 large_state = np.array([1e6, -1e6, 1e10, -1e10])
 action = agent.act(large_state, training=False)
 assert isinstance(action, int)

 # Very small values
 small_state = np.array([1e-10, -1e-10, 1e-15, -1e-15])
 action = agent.act(small_state, training=False)
 assert isinstance(action, int)


@pytest.mark.benchmark
class TestPerformance:
 """Performance benchmarks."""

 def test_action_selection_speed(self, benchmark):
 """Benchmark action selection speed."""
 network_config = QNetworkConfig(state_size=100, action_size=10, hidden_layers=[128, 64])
 config = DQNConfig(network_config=network_config)
 agent = DQN(config)

 state = np.random.randn(100)

 def select_action:
 return agent.act(state, training=False)

 result = benchmark(select_action)
 assert isinstance(result, int)

 def test_training_speed(self, benchmark):
 """Benchmark training speed."""
 network_config = QNetworkConfig(state_size=20, action_size=4, hidden_layers=[64, 32])
 config = DQNConfig(
 network_config=network_config,
 buffer_size=1000,
 batch_size=32,
 min_replay_size=100
 )
 agent = DQN(config)

 # Fill buffer
 for i in range(150):
 state = np.random.randn(20)
 next_state = np.random.randn(20)
 agent.store_experience(state, i % 4, np.random.randn, next_state, False)

 def train_step:
 return agent.train_step

 metrics = benchmark(train_step)
 assert isinstance(metrics, dict)
 assert 'loss' in metrics

 def test_memory_usage(self):
 """Test memory usage patterns."""
 network_config = QNetworkConfig(state_size=50, action_size=5, hidden_layers=[128, 64])
 config = DQNConfig(network_config=network_config, buffer_size=10000)
 agent = DQN(config)

 # Measure memory before
 import psutil
 import os
 process = psutil.Process(os.getpid)
 memory_before = process.memory_info.rss

 # Add many experiences
 for i in range(5000):
 state = np.random.randn(50)
 next_state = np.random.randn(50)
 agent.store_experience(state, i % 5, np.random.randn, next_state, i % 100 == 0)

 # Memory after
 memory_after = process.memory_info.rss
 memory_increase = (memory_after - memory_before) / 1024 / 1024 # MB

 # Should not use excessive memory
 assert memory_increase < 500 # Less than 500MB
 assert len(agent.replay_buffer) <= config.buffer_size


if __name__ == "__main__":
 # Run specific test groups
 pytest.main([
 "test_dqn.py::TestQNetwork",
 "test_dqn.py::TestDQN",
 "-v"
 ])