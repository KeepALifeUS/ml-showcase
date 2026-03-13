"""
Base Trading Environment for Crypto Trading Bot v5.0
enterprise patterns for production-ready trading environments

Foundation for all trading environments with enterprise-grade functionality:
- Async support for real-time trading
- Comprehensive logging and monitoring
- Fault tolerance and graceful degradation
- Performance optimization
- Type safety and validation
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import warnings

#  enterprise imports
from ..utils.logger import StructuredLogger, TradingEvent
from ..utils.performance import PerformanceMonitor
from ..utils.risk_metrics import RiskCalculator
from ..utils.portfolio import PortfolioManager


@dataclass
class BaseTradingConfig:
    """Base configuration for all trading environments"""
    
    # Environment parameters
    max_steps: int = 1000
    initial_balance: float = 10000.0
    random_seed: Optional[int] = None
    
    # Risk management
    max_position_size: float = 1.0
    max_leverage: float = 1.0
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.10
    max_drawdown: float = 0.20
    
    # Transaction costs
    maker_fee: float = 0.001
    taker_fee: float = 0.002
    slippage_model: str = "linear"  # linear, sqrt, impact
    
    # Performance tracking
    enable_performance_logging: bool = True
    performance_window: int = 100
    risk_calculation_freq: int = 10
    
    # Advanced features
    enable_portfolio_rebalancing: bool = True
    enable_sentiment_signals: bool = True
    enable_market_microstructure: bool = False
    
    # Data configuration
    normalize_observations: bool = True
    observation_window: int = 50
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "ema_12", "rsi_14", "bb_20", "macd"
    ])


class BaseTradingEnvironment(gym.Env, ABC):
    """
    Base trading environment with enterprise patterns
    
    Ensures:
    - Production-ready error handling
    - Comprehensive monitoring and logging
    - Performance optimization
    - Async support
    - Type safety
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "json"],
        "render_fps": 4,
    }
    
    def __init__(
        self,
        config: BaseTradingConfig,
        logger: Optional[StructuredLogger] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        super().__init__()
        
        self.config = config
        self.logger = logger or StructuredLogger(__name__)
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        
        # Core state
        self._setup_state()
        
        # Risk management
        self.risk_calculator = RiskCalculator()
        self.portfolio_manager = PortfolioManager(
            initial_balance=config.initial_balance,
            max_position_size=config.max_position_size,
            max_leverage=config.max_leverage
        )
        
        # Performance tracking
        self.episode_metrics = {}
        self.step_times = deque(maxlen=1000)
        self.reward_history = deque(maxlen=config.performance_window)
        
        # Setup spaces
        self._setup_spaces()
        
        # State validation
        self._validate_configuration()
        
        self.logger.info("Base trading environment initialized", extra={
            "config": self.config.__dict__,
            "observation_space": str(self.observation_space.shape),
            "action_space": str(self.action_space.shape)
        })
    
    def _setup_state(self) -> None:
        """Initialize environment state"""
        self.current_step = 0
        self.episode_start_time = time.time()
        self.is_done = False
        self.info = {}
        
        # Trading state
        self.balance = self.config.initial_balance
        self.positions = {}
        self.portfolio_value = self.config.initial_balance
        self.max_portfolio_value = self.config.initial_balance
        
        # Performance metrics
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
        # History tracking
        self.price_history = deque(maxlen=self.config.observation_window)
        self.action_history = deque(maxlen=100)
        self.portfolio_history = deque(maxlen=1000)
    
    @abstractmethod
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces"""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        pass
    
    @abstractmethod
    def _execute_action(self, action: Union[np.ndarray, int]) -> Dict[str, Any]:
        """Execute trading action"""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action_result: Dict[str, Any]) -> float:
        """Calculate step reward"""
        pass
    
    @abstractmethod
    def _update_market_data(self) -> Dict[str, Any]:
        """Update market data for current step"""
        pass
    
    def reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state"""
        
        with self.performance_monitor.measure_time("environment_reset"):
            try:
                super().reset(seed=seed)
                
                # Log episode end if not first reset
                if hasattr(self, 'current_step') and self.current_step > 0:
                    self._log_episode_end()
                
                # Reset state
                self._setup_state()
                
                # Reset portfolio manager
                self.portfolio_manager.reset()
                
                # Initialize market data
                self._initialize_market_data()
                
                # Get initial observation
                observation = self._get_observation()
                
                # Validate observation
                self._validate_observation(observation)
                
                self.logger.debug("Environment reset completed", extra={
                    "observation_shape": observation.shape,
                    "portfolio_value": self.portfolio_value,
                    "seed": seed
                })
                
                return observation, self.info
                
            except Exception as e:
                self.logger.error(f"Error during environment reset: {e}", exc_info=True)
                raise
    
    def step(
        self, 
        action: Union[np.ndarray, int, List[float]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step"""
        
        step_start_time = time.time()
        
        try:
            with self.performance_monitor.measure_time("environment_step"):
                
                # Validate action
                if not self.action_space.contains(action):
                    self.logger.warning(f"Invalid action: {action}")
                    action = self._sanitize_action(action)
                
                # Store action
                self.action_history.append(action)
                
                # Update market data
                market_update = self._update_market_data()
                
                # Execute action
                action_result = self._execute_action(action)
                
                # Update portfolio
                self._update_portfolio(market_update, action_result)
                
                # Calculate reward
                reward = self._calculate_reward(action_result)
                self.reward_history.append(reward)
                
                # Check termination conditions
                terminated = self._check_terminated()
                truncated = self._check_truncated()
                self.is_done = terminated or truncated
                
                # Update metrics
                self._update_metrics(reward, action_result)
                
                # Get next observation
                observation = self._get_observation()
                
                # Prepare info
                self.info = self._prepare_info(action_result, market_update)
                
                # Performance logging
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)
                
                if self.current_step % self.config.risk_calculation_freq == 0:
                    self._update_risk_metrics()
                
                self.current_step += 1
                
                # Log step if enabled
                if self.config.enable_performance_logging:
                    self._log_step(action, reward, step_time)
                
                return observation, reward, terminated, truncated, self.info
                
        except Exception as e:
            self.logger.error(f"Error during environment step: {e}", exc_info=True)
            # Graceful degradation - return safe values
            observation = self._get_safe_observation()
            return observation, -1.0, True, False, {"error": str(e)}
    
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
        
        try:
            if mode == "human":
                self._render_human()
            elif mode == "rgb_array":
                return self._render_rgb_array()
            elif mode == "json":
                return self._render_json()
            else:
                self.logger.warning(f"Unsupported render mode: {mode}")
                
        except Exception as e:
            self.logger.error(f"Error during rendering: {e}", exc_info=True)
    
    def close(self) -> None:
        """Clean up environment resources"""
        
        try:
            if hasattr(self, 'current_step') and self.current_step > 0:
                self._log_episode_end()
            
            # Cleanup resources
            if hasattr(self, 'portfolio_manager'):
                self.portfolio_manager.close()
                
            self.logger.info("Environment closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during environment cleanup: {e}", exc_info=True)
    
    def _validate_configuration(self) -> None:
        """Validate environment configuration"""
        
        if self.config.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        
        if self.config.max_steps <= 0:
            raise ValueError("Max steps must be positive")
        
        if not 0 < self.config.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        
        if self.config.max_drawdown <= 0 or self.config.max_drawdown >= 1:
            raise ValueError("Max drawdown must be between 0 and 1")
    
    def _validate_observation(self, observation: np.ndarray) -> None:
        """Validate observation shape and values"""
        
        if observation.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape {observation.shape} doesn't match "
                f"observation space {self.observation_space.shape}"
            )
        
        if not self.observation_space.contains(observation):
            self.logger.warning("Observation contains invalid values")
            # Clip values to valid range
            observation = np.clip(observation, 
                                self.observation_space.low, 
                                self.observation_space.high)
    
    def _sanitize_action(self, action: Union[np.ndarray, int, List[float]]) -> np.ndarray:
        """Sanitize invalid action"""
        
        if isinstance(action, (list, tuple)):
            action = np.array(action, dtype=np.float32)
        elif isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        
        # Clip to action space bounds
        if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
            action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def _get_safe_observation(self) -> np.ndarray:
        """Get safe observation in case of error"""
        
        try:
            return self._get_observation()
        except:
            # Return zeros matching observation space
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _update_portfolio(self, market_update: Dict, action_result: Dict) -> None:
        """Update portfolio state"""
        
        # Update portfolio manager
        self.portfolio_manager.update(market_update, action_result)
        
        # Update local state
        self.balance = self.portfolio_manager.balance
        self.positions = self.portfolio_manager.positions.copy()
        self.portfolio_value = self.portfolio_manager.total_value
        
        # Update maximum portfolio value
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        
        # Store portfolio history
        self.portfolio_history.append({
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy()
        })
    
    def _update_metrics(self, reward: float, action_result: Dict) -> None:
        """Update performance metrics"""
        
        # Basic metrics
        self.total_return = (self.portfolio_value - self.config.initial_balance) / self.config.initial_balance
        
        # Drawdown calculation
        current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Trading statistics
        if action_result.get("trades_executed", 0) > 0:
            self.total_trades += action_result["trades_executed"]
            if reward > 0:
                self.winning_trades += action_result["trades_executed"]
        
        # Calculate Sharpe ratio
        if len(self.reward_history) >= 20:
            returns = np.array(self.reward_history)
            if np.std(returns) > 0:
                self.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _update_risk_metrics(self) -> None:
        """Update risk metrics"""
        
        try:
            if len(self.portfolio_history) > 20:
                portfolio_values = [p["portfolio_value"] for p in self.portfolio_history]
                risk_metrics = self.risk_calculator.calculate_risk_metrics(portfolio_values)
                
                self.info.update({
                    "var_95": risk_metrics.get("var_95", 0.0),
                    "expected_shortfall": risk_metrics.get("expected_shortfall", 0.0),
                    "volatility": risk_metrics.get("volatility", 0.0)
                })
                
        except Exception as e:
            self.logger.warning(f"Error calculating risk metrics: {e}")
    
    def _check_terminated(self) -> bool:
        """Check if episode should terminate"""
        
        # Liquidation condition
        liquidation_threshold = self.config.initial_balance * (1 - self.config.max_drawdown)
        if self.portfolio_value <= liquidation_threshold:
            self.logger.warning("Episode terminated due to liquidation")
            return True
        
        # Maximum drawdown exceeded
        if self.max_drawdown >= self.config.max_drawdown:
            self.logger.warning("Episode terminated due to max drawdown")
            return True
        
        return False
    
    def _check_truncated(self) -> bool:
        """Check if episode should be truncated"""
        
        # Maximum steps reached
        return self.current_step >= self.config.max_steps
    
    def _prepare_info(self, action_result: Dict, market_update: Dict) -> Dict[str, Any]:
        """Prepare info dictionary"""
        
        info = {
            # Portfolio state
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "total_return": self.total_return,
            
            # Performance metrics
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            
            # Current step info
            "step": self.current_step,
            "action_result": action_result,
            "market_update": market_update,
            
            # Performance
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0.0
        }
        
        # Add episode info if done
        if self.is_done:
            episode_duration = time.time() - self.episode_start_time
            info.update({
                "episode_duration": episode_duration,
                "episode_return": self.total_return,
                "episode_length": self.current_step,
                "final_portfolio_value": self.portfolio_value
            })
        
        return info
    
    def _initialize_market_data(self) -> None:
        """Initialize market data - implemented by subclasses"""
        pass
    
    def _log_step(self, action: Any, reward: float, step_time: float) -> None:
        """Log step information"""
        
        if self.current_step % 100 == 0:  # Log every 100 steps
            self.logger.debug("Step completed", extra={
                "step": self.current_step,
                "portfolio_value": self.portfolio_value,
                "reward": reward,
                "step_time": step_time,
                "total_return": self.total_return,
                "max_drawdown": self.max_drawdown
            })
    
    def _log_episode_end(self) -> None:
        """Log episode end information"""
        
        episode_duration = time.time() - self.episode_start_time
        
        self.logger.info("Episode completed", extra={
            "episode_length": self.current_step,
            "episode_duration": episode_duration,
            "final_portfolio_value": self.portfolio_value,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "avg_step_time": np.mean(self.step_times) if self.step_times else 0.0
        })
    
    def _render_human(self) -> None:
        """Render for human viewing"""
        
        print(f"\n=== Trading Environment - Step {self.current_step} ===")
        print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"Balance: ${self.balance:,.2f}")
        print(f"Total Return: {self.total_return:.2%}")
        print(f"Max Drawdown: {self.max_drawdown:.2%}")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.winning_trades / max(self.total_trades, 1):.1%}")
        
        if self.positions:
            print("\nPositions:")
            for asset, position in self.positions.items():
                if abs(position) > 1e-6:
                    print(f"  {asset}: {position:.4f}")
        
        print("=" * 50)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for visualization"""
        # Implement visualization logic here
        # For now return placeholder
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def _render_json(self) -> Dict[str, Any]:
        """Render as JSON for API integration"""
        
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions,
            "total_return": self.total_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.winning_trades / max(self.total_trades, 1),
            "is_done": self.is_done
        }


# Async version for real-time trading
class AsyncBaseTradingEnvironment(BaseTradingEnvironment):
    """
    Async version of base trading environment
    for real-time trading integration
    """
    
    async def async_step(
        self, 
        action: Union[np.ndarray, int, List[float]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Async version of step method"""
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.step, action)
    
    async def async_reset(
        self, 
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Async version of reset method"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.reset, seed, options)


__all__ = [
    "BaseTradingConfig",
    "BaseTradingEnvironment", 
    "AsyncBaseTradingEnvironment"
]