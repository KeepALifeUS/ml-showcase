"""
Test suite for ML Gym Trading Environments
enterprise patterns for comprehensive testing

Tests cover:
- Environment initialization and configuration
- Observation space generation
- Action space functionality
- Reward calculation
- Market simulation
- Error handling and edge cases
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.crypto_trading_env import (
    CryptoTradingEnvironment,
    CryptoTradingConfig,
    MarketRegime
)
from spaces.observations import ObservationConfig, CryptoObservationSpace
from spaces.actions import ActionConfig, ActionMode, CryptoActionSpace
from rewards.profit_reward import ProfitReward, ProfitRewardConfig
from utils.logger import StructuredLogger, EventType


class TestCryptoTradingEnvironment:
    """Test suite for CryptoTradingEnvironment"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return CryptoTradingConfig(
            assets=["BTC", "ETH"],
            initial_balance=10000.0,
            max_steps=100,
            data_source="synthetic"
        )
    
    @pytest.fixture
    def environment(self, basic_config):
        """Create test environment"""
        return CryptoTradingEnvironment(basic_config)
    
    def test_environment_initialization(self, basic_config):
        """Test environment initialization"""
        env = CryptoTradingEnvironment(basic_config)
        
        assert env.crypto_config.assets == ["BTC", "ETH"]
        assert env.crypto_config.initial_balance == 10000.0
        assert env.crypto_config.max_steps == 100
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
    
    def test_environment_reset(self, environment):
        """Test environment reset functionality"""
        observation, info = environment.reset()
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert observation.shape == environment.observation_space.shape
        assert environment.current_step == 0
        assert environment.balance == environment.crypto_config.initial_balance
        assert isinstance(info, dict)
    
    def test_environment_step(self, environment):
        """Test environment step functionality"""
        environment.reset()
        
        # Sample random action
        action = environment.action_space.sample()
        observation, reward, terminated, truncated, info = environment.step(action)
        
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert environment.current_step == 1
    
    def test_observation_space_shape(self, environment):
        """Test observation space has correct shape"""
        observation, _ = environment.reset()
        
        expected_shape = environment.observation_space.shape
        actual_shape = observation.shape
        
        assert actual_shape == expected_shape, f"Expected {expected_shape}, got {actual_shape}"
    
    def test_action_space_validation(self, environment):
        """Test action space validation"""
        environment.reset()
        
        # Test valid action
        valid_action = environment.action_space.sample()
        assert environment.action_space.contains(valid_action)
        
        # Test step with valid action
        obs, reward, terminated, truncated, info = environment.step(valid_action)
        assert obs is not None
    
    def test_reward_calculation(self, environment):
        """Test reward calculation"""
        environment.reset()
        
        initial_portfolio_value = environment.portfolio_value
        
        # Execute step
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)
        
        # Reward should be calculated
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_market_regime_detection(self, environment):
        """Test market regime detection"""
        environment.reset()
        
        # Market regime should be initialized
        assert hasattr(environment, 'market_regime')
        assert isinstance(environment.market_regime, MarketRegime)
        
        # Step through environment
        for _ in range(10):
            action = environment.action_space.sample()
            environment.step(action)
        
        # Regime should still be valid
        assert isinstance(environment.market_regime, MarketRegime)
    
    def test_portfolio_tracking(self, environment):
        """Test portfolio value tracking"""
        environment.reset()
        
        initial_balance = environment.balance
        initial_portfolio_value = environment.portfolio_value
        
        assert initial_balance == environment.crypto_config.initial_balance
        assert initial_portfolio_value >= initial_balance
        
        # Execute trades
        for _ in range(5):
            action = environment.action_space.sample()
            environment.step(action)
        
        # Portfolio should be tracked
        assert hasattr(environment, 'portfolio_history')
        assert len(environment.portfolio_history) > 0
    
    def test_sentiment_integration(self):
        """Test sentiment analysis integration"""
        config = CryptoTradingConfig(
            assets=["BTC", "ETH"],
            enable_sentiment_signals=True,
            sentiment_sources=["twitter", "reddit", "news"]
        )
        
        env = CryptoTradingEnvironment(config)
        env.reset()
        
        # Should have sentiment analyzer
        assert hasattr(env, 'sentiment_analyzer')
        
        # Sentiment scores should be available
        sentiment_data = env.sentiment_analyzer.get_current_sentiment()
        assert isinstance(sentiment_data, dict)
        assert "BTC" in sentiment_data
        assert "ETH" in sentiment_data
    
    def test_order_book_simulation(self):
        """Test order book simulation"""
        config = CryptoTradingConfig(
            assets=["BTC", "ETH"],
            enable_order_book=True,
            order_book_depth=5
        )
        
        env = CryptoTradingEnvironment(config)
        env.reset()
        
        # Should have order book simulator
        assert hasattr(env, 'order_book_simulator')
        
        # Execute action and check order book data
        action = env.action_space.sample()
        env.step(action)
    
    def test_error_handling(self, environment):
        """Test error handling and edge cases"""
        environment.reset()
        
        # Test with invalid action (should be handled gracefully)
        if hasattr(environment.action_space, 'n'):
            # Discrete action space
            invalid_action = -1
        else:
            # Continuous action space
            invalid_action = np.array([float('inf')] * environment.action_space.shape[0])
        
        # Environment should handle invalid actions gracefully
        obs, reward, terminated, truncated, info = environment.step(invalid_action)
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
    
    def test_episode_termination(self, environment):
        """Test episode termination conditions"""
        environment.reset()
        
        # Run until termination
        max_steps = 200
        step_count = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and step_count < max_steps:
            action = environment.action_space.sample()
            obs, reward, terminated, truncated, info = environment.step(action)
            step_count += 1
        
        # Should terminate within reasonable time
        assert step_count <= environment.crypto_config.max_steps or terminated or truncated
    
    def test_info_dict_contents(self, environment):
        """Test info dictionary contents"""
        environment.reset()
        
        action = environment.action_space.sample()
        obs, reward, terminated, truncated, info = environment.step(action)
        
        # Check required info keys
        expected_keys = [
            "portfolio_value", "balance", "positions", 
            "total_trades", "max_drawdown"
        ]
        
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_market_data_updates(self, environment):
        """Test market data updates"""
        environment.reset()
        
        initial_prices = environment.current_prices.copy()
        
        # Step through environment
        for _ in range(5):
            action = environment.action_space.sample()
            environment.step(action)
        
        # Prices should have updated
        current_prices = environment.current_prices
        assert len(current_prices) == len(initial_prices)
        assert all(asset in current_prices for asset in initial_prices.keys())


class TestObservationSpace:
    """Test suite for observation space"""
    
    def test_observation_space_creation(self):
        """Test observation space creation"""
        config = ObservationConfig(
            include_price=True,
            include_volume=True,
            include_technical_indicators=True,
            include_sentiment=True
        )
        
        assets = ["BTC", "ETH"]
        obs_space = CryptoObservationSpace(config, assets)
        
        gym_space = obs_space.create_space()
        assert hasattr(gym_space, 'shape')
        assert len(gym_space.shape) == 1
        assert gym_space.shape[0] > 0
    
    def test_observation_building(self):
        """Test observation vector building"""
        config = ObservationConfig()
        assets = ["BTC", "ETH"]
        obs_space = CryptoObservationSpace(config, assets)
        
        # Mock data
        prices = {"BTC": 50000.0, "ETH": 3000.0}
        price_history = [prices] * 10
        volumes = {"BTC": 1000.0, "ETH": 500.0}
        
        observation = obs_space.build_observation(
            prices=prices,
            price_history=price_history,
            volumes=volumes
        )
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] > 0
        assert not np.any(np.isnan(observation))
        assert not np.any(np.isinf(observation))


class TestActionSpace:
    """Test suite for action space"""
    
    def test_continuous_action_space(self):
        """Test continuous action space"""
        config = ActionConfig(action_mode=ActionMode.CONTINUOUS)
        assets = ["BTC", "ETH"]
        
        action_space_builder = CryptoActionSpace(config, assets)
        gym_space = action_space_builder.create_space()
        
        # Should be Box space
        assert hasattr(gym_space, 'sample')
        assert hasattr(gym_space, 'shape')
        
        # Test action parsing
        action = gym_space.sample()
        parsed = action_space_builder.parse_action(action)
        
        assert isinstance(parsed, dict)
        assert "orders" in parsed
        assert "action_type" in parsed
    
    def test_discrete_action_space(self):
        """Test discrete action space"""
        config = ActionConfig(action_mode=ActionMode.DISCRETE)
        assets = ["BTC", "ETH"]
        
        action_space_builder = CryptoActionSpace(config, assets)
        gym_space = action_space_builder.create_space()
        
        # Should be Discrete space
        assert hasattr(gym_space, 'n')
        
        # Test action parsing
        action = gym_space.sample()
        parsed = action_space_builder.parse_action(action)
        
        assert isinstance(parsed, dict)
        assert "orders" in parsed
        assert parsed["action_type"] == "discrete"


class TestRewardFunctions:
    """Test suite for reward functions"""
    
    def test_profit_reward(self):
        """Test profit-based reward function"""
        config = ProfitRewardConfig(
            profit_scale=100.0,
            enable_risk_penalty=True
        )
        
        reward_fn = ProfitReward(config)
        
        # Test reward calculation
        trade_info = {
            "total_fees": 10.0,
            "total_slippage": 5.0,
            "trades_executed": 2
        }
        
        reward = reward_fn.calculate_reward(
            portfolio_value=11000.0,
            previous_portfolio_value=10000.0,
            trade_info=trade_info
        )
        
        assert isinstance(reward, float)
        assert not np.isnan(reward)
        assert not np.isinf(reward)
        
        # Profitable trade should give positive reward
        assert reward > 0  # 10% profit should be positive even with penalties
    
    def test_reward_statistics(self):
        """Test reward statistics"""
        config = ProfitRewardConfig()
        reward_fn = ProfitReward(config)
        
        # Calculate multiple rewards
        for i in range(10):
            reward_fn.calculate_reward(
                portfolio_value=10000 + i * 100,
                previous_portfolio_value=10000 + (i-1) * 100 if i > 0 else 10000,
                trade_info={"total_fees": 1.0, "total_slippage": 0.5, "trades_executed": 1}
            )
        
        stats = reward_fn.get_statistics()
        assert isinstance(stats, dict)
        assert "mean_reward" in stats
        assert "cumulative_reward" in stats


class TestUtilities:
    """Test suite for utility functions"""
    
    def test_structured_logger(self):
        """Test structured logger"""
        logger = StructuredLogger("test_env")
        
        # Test basic logging
        logger.info("Test message", extra={"test_key": "test_value"})
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Test event logging
        logger.log_trade_execution(
            asset="BTC",
            side="buy",
            quantity=0.1,
            price=50000.0,
            fees=50.0,
            step=1
        )
        
        # Check events were logged
        events = logger.get_events(EventType.TRADE_EXECUTION)
        assert len(events) > 0
        
        # Test performance metrics
        logger.log_performance_metric("test_metric", 1.5, step=1)
        
        summary = logger.get_performance_summary()
        assert "test_metric" in summary
    
    def test_logger_context(self):
        """Test logger context management"""
        logger = StructuredLogger("test_env")
        
        # Set context
        logger.set_context(environment_id="test_env_001", episode=1)
        
        logger.info("Test message with context")
        
        # Clear context
        logger.clear_context()
        
        # Should not raise errors
        assert True


# Integration tests
class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_full_trading_episode(self):
        """Test complete trading episode"""
        config = CryptoTradingConfig(
            assets=["BTC", "ETH"],
            initial_balance=10000.0,
            max_steps=50,
            enable_sentiment_signals=True
        )
        
        env = CryptoTradingEnvironment(config)
        
        # Run complete episode
        observation, info = env.reset()
        total_reward = 0.0
        step_count = 0
        
        while step_count < 50:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        # Verify episode completed successfully
        assert step_count > 0
        assert isinstance(total_reward, (int, float))
        assert not np.isnan(total_reward)
        
        # Check final info
        assert "portfolio_value" in info
        assert "total_trades" in info
        assert info["portfolio_value"] > 0
    
    def test_multiple_episodes(self):
        """Test multiple episodes"""
        config = CryptoTradingConfig(
            assets=["BTC", "ETH"],
            initial_balance=10000.0,
            max_steps=20
        )
        
        env = CryptoTradingEnvironment(config)
        
        episode_returns = []
        
        # Run multiple episodes
        for episode in range(3):
            observation, info = env.reset()
            episode_reward = 0.0
            
            for step in range(20):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            episode_returns.append(episode_reward)
        
        # All episodes should complete
        assert len(episode_returns) == 3
        assert all(isinstance(r, (int, float)) for r in episode_returns)
        assert all(not np.isnan(r) for r in episode_returns)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])