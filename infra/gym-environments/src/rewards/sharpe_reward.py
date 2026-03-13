"""
Sharpe Ratio Based Reward Functions
enterprise patterns for risk-adjusted performance optimization

Advanced reward functions based on Sharpe ratio and related risk-adjusted metrics:
- Rolling Sharpe ratio rewards
- Information ratio rewards
- Sortino ratio rewards
- Calmar ratio rewards
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from collections import deque
import logging

from ..utils.risk_metrics import (
    calculate_sharpe_ratio, calculate_sortino_ratio, 
    calculate_information_ratio, calculate_calmar_ratio
)


@dataclass
class SharpeRewardConfig:
    """Configuration for Sharpe-based rewards"""
    
    # Sharpe calculation parameters
    lookback_window: int = 50           # Number of periods for calculation
    risk_free_rate: float = 0.02        # Annual risk-free rate
    target_sharpe: float = 1.0          # Target Sharpe ratio
    
    # Reward scaling
    sharpe_scale: float = 10.0          # Scale Sharpe rewards
    minimum_periods: int = 10           # Minimum periods before calculation
    
    # Rolling vs batch calculation
    use_rolling_calculation: bool = True
    update_frequency: int = 1           # How often to recalculate (steps)
    
    # Benchmark comparison
    use_benchmark: bool = False
    benchmark_returns: Optional[List[float]] = None
    
    # Advanced features
    use_downside_deviation: bool = False  # Use Sortino instead of Sharpe
    use_value_at_risk: bool = False      # Incorporate VaR adjustment
    var_confidence: float = 0.05         # VaR confidence level
    
    # Reward smoothing
    enable_reward_smoothing: bool = True
    smoothing_alpha: float = 0.1        # EMA smoothing factor


class SharpeReward:
    """
    Sharpe ratio based reward function
    
    Rewards agents based on risk-adjusted returns using rolling Sharpe ratio
    """
    
    def __init__(self, config: SharpeRewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Returns tracking
        self.returns = deque(maxlen=config.lookback_window)
        self.portfolio_values = deque(maxlen=config.lookback_window + 1)
        
        # Reward tracking
        self.sharpe_history = []
        self.reward_history = []
        self.smoothed_reward = 0.0
        
        # Benchmark tracking
        if config.use_benchmark and config.benchmark_returns:
            self.benchmark_returns = deque(config.benchmark_returns, maxlen=config.lookback_window)
        else:
            self.benchmark_returns = None
        
        # Step counter for update frequency
        self.step_count = 0
        self.last_sharpe = 0.0
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: Optional[float] = None,
        trade_info: Optional[Dict[str, Any]] = None,
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate Sharpe-based reward"""
        
        # Update portfolio values
        self.portfolio_values.append(portfolio_value)
        
        # Calculate return
        if len(self.portfolio_values) >= 2:
            prev_value = self.portfolio_values[-2]
            if prev_value > 0:
                period_return = (portfolio_value - prev_value) / prev_value
            else:
                period_return = 0.0
        else:
            period_return = 0.0
        
        self.returns.append(period_return)
        
        # Check if we should calculate reward
        self.step_count += 1
        if (self.step_count % self.config.update_frequency != 0 or 
            len(self.returns) < self.config.minimum_periods):
            
            # Return previous reward if not updating
            return self.smoothed_reward if self.config.enable_reward_smoothing else 0.0
        
        # Calculate Sharpe ratio
        current_sharpe = self._calculate_sharpe_ratio()
        self.sharpe_history.append(current_sharpe)
        self.last_sharpe = current_sharpe
        
        # Convert Sharpe to reward
        reward = self._sharpe_to_reward(current_sharpe)
        
        # Apply smoothing
        if self.config.enable_reward_smoothing:
            self.smoothed_reward = (
                self.config.smoothing_alpha * reward + 
                (1 - self.config.smoothing_alpha) * self.smoothed_reward
            )
            final_reward = self.smoothed_reward
        else:
            final_reward = reward
        
        self.reward_history.append(final_reward)
        
        self.logger.debug("Sharpe reward calculated", extra={
            "portfolio_value": portfolio_value,
            "period_return": period_return,
            "sharpe_ratio": current_sharpe,
            "reward": final_reward,
            "returns_count": len(self.returns)
        })
        
        return float(final_reward)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from returns"""
        
        if len(self.returns) < self.config.minimum_periods:
            return 0.0
        
        returns_array = np.array(self.returns)
        
        if self.config.use_downside_deviation:
            # Use Sortino ratio instead
            return calculate_sortino_ratio(returns_array, self.config.risk_free_rate)
        else:
            return calculate_sharpe_ratio(returns_array, self.config.risk_free_rate)
    
    def _sharpe_to_reward(self, sharpe_ratio: float) -> float:
        """Convert Sharpe ratio to reward signal"""
        
        # Base reward scaled by how much we exceed/fall short of target
        excess_sharpe = sharpe_ratio - self.config.target_sharpe
        base_reward = excess_sharpe * self.config.sharpe_scale
        
        # Apply non-linear scaling to encourage high Sharpe ratios
        if sharpe_ratio > 0:
            reward = base_reward * (1 + np.tanh(sharpe_ratio))
        else:
            reward = base_reward * (1 - np.tanh(abs(sharpe_ratio)))
        
        # Bonus for consistently high Sharpe
        if len(self.sharpe_history) >= 5:
            recent_sharpes = self.sharpe_history[-5:]
            if all(s > self.config.target_sharpe for s in recent_sharpes):
                reward *= 1.2  # 20% bonus for consistency
        
        return reward
    
    def reset(self) -> None:
        """Reset reward state"""
        self.returns.clear()
        self.portfolio_values.clear()
        self.sharpe_history.clear()
        self.reward_history.clear()
        self.smoothed_reward = 0.0
        self.step_count = 0
        self.last_sharpe = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics"""
        
        if not self.sharpe_history:
            return {}
        
        sharpes = np.array(self.sharpe_history)
        rewards = np.array(self.reward_history) if self.reward_history else np.array([])
        
        stats = {
            "sharpe_statistics": {
                "current_sharpe": self.last_sharpe,
                "mean_sharpe": float(np.mean(sharpes)),
                "std_sharpe": float(np.std(sharpes)),
                "min_sharpe": float(np.min(sharpes)),
                "max_sharpe": float(np.max(sharpes)),
                "sharpe_above_target": float(np.sum(sharpes > self.config.target_sharpe)),
                "sharpe_stability": float(1.0 / (1.0 + np.std(sharpes))) if np.std(sharpes) > 0 else 1.0
            }
        }
        
        if len(rewards) > 0:
            stats["reward_statistics"] = {
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "cumulative_reward": float(np.sum(rewards)),
                "reward_sharpe": float(np.mean(rewards) / np.std(rewards)) if np.std(rewards) > 0 else 0.0
            }
        
        return stats


class InformationRatioReward(SharpeReward):
    """
    Information ratio reward (excess return vs benchmark / tracking error)
    """
    
    def __init__(self, config: SharpeRewardConfig, benchmark_returns: List[float]):
        super().__init__(config)
        self.benchmark_returns = deque(benchmark_returns, maxlen=config.lookback_window)
        self.benchmark_idx = 0
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Information ratio instead of Sharpe"""
        
        if len(self.returns) < self.config.minimum_periods:
            return 0.0
        
        # Get corresponding benchmark returns
        portfolio_returns = list(self.returns)
        benchmark_slice = list(self.benchmark_returns)[-len(portfolio_returns):]
        
        if len(benchmark_slice) != len(portfolio_returns):
            return 0.0
        
        return calculate_information_ratio(portfolio_returns, benchmark_slice)


class SortinoReward(SharpeReward):
    """
    Sortino ratio reward (focuses on downside deviation)
    """
    
    def __init__(self, config: SharpeRewardConfig, minimum_acceptable_return: float = 0.0):
        config.use_downside_deviation = True
        super().__init__(config)
        self.minimum_acceptable_return = minimum_acceptable_return
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sortino ratio"""
        
        if len(self.returns) < self.config.minimum_periods:
            return 0.0
        
        returns_array = np.array(self.returns)
        return calculate_sortino_ratio(returns_array, self.minimum_acceptable_return)


class CalmarReward(SharpeReward):
    """
    Calmar ratio reward (annual return / maximum drawdown)
    """
    
    def __init__(self, config: SharpeRewardConfig):
        super().__init__(config)
        self.max_portfolio_value = 0.0
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: Optional[float] = None,
        trade_info: Optional[Dict[str, Any]] = None,
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate Calmar-based reward"""
        
        # Track maximum portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        return super().calculate_reward(
            portfolio_value, previous_portfolio_value, trade_info, market_info
        )
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Calmar ratio"""
        
        if len(self.portfolio_values) < self.config.minimum_periods:
            return 0.0
        
        portfolio_values = list(self.portfolio_values)
        return calculate_calmar_ratio(portfolio_values)


class AdaptiveSharpeReward(SharpeReward):
    """
    Adaptive Sharpe reward with dynamic parameters
    """
    
    def __init__(self, config: SharpeRewardConfig):
        super().__init__(config)
        
        # Adaptive parameters
        self.performance_history = deque(maxlen=100)
        self.adaptation_frequency = 20
        
        # Dynamic target adjustment
        self.dynamic_target = config.target_sharpe
        self.target_adjustment_rate = 0.1
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: Optional[float] = None,
        trade_info: Optional[Dict[str, Any]] = None,
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate adaptive Sharpe reward"""
        
        reward = super().calculate_reward(
            portfolio_value, previous_portfolio_value, trade_info, market_info
        )
        
        self.performance_history.append(self.last_sharpe)
        
        # Adapt target Sharpe based on recent performance
        if (len(self.performance_history) >= self.adaptation_frequency and
            self.step_count % self.adaptation_frequency == 0):
            
            recent_performance = np.mean(list(self.performance_history)[-self.adaptation_frequency:])
            
            # Adjust target towards recent performance
            target_adjustment = (recent_performance - self.dynamic_target) * self.target_adjustment_rate
            self.dynamic_target += target_adjustment
            
            # Clamp target to reasonable range
            self.dynamic_target = np.clip(self.dynamic_target, 0.1, 3.0)
            
            self.logger.debug(f"Adapted target Sharpe to: {self.dynamic_target:.3f}")
        
        return reward
    
    def _sharpe_to_reward(self, sharpe_ratio: float) -> float:
        """Use dynamic target in reward calculation"""
        
        # Use dynamic target instead of static one
        excess_sharpe = sharpe_ratio - self.dynamic_target
        base_reward = excess_sharpe * self.config.sharpe_scale
        
        # Apply scaling based on achievement relative to adaptive target
        if sharpe_ratio > self.dynamic_target:
            reward = base_reward * (1 + np.tanh(sharpe_ratio / self.dynamic_target))
        else:
            reward = base_reward * (1 - np.tanh(abs(sharpe_ratio - self.dynamic_target)))
        
        return reward


# Factory functions
def create_sharpe_reward(
    lookback_window: int = 50,
    target_sharpe: float = 1.0,
    risk_free_rate: float = 0.02
) -> SharpeReward:
    """Create standard Sharpe reward"""
    
    config = SharpeRewardConfig(
        lookback_window=lookback_window,
        target_sharpe=target_sharpe,
        risk_free_rate=risk_free_rate
    )
    
    return SharpeReward(config)


def create_sortino_reward(
    lookback_window: int = 50,
    minimum_acceptable_return: float = 0.0
) -> SortinoReward:
    """Create Sortino ratio reward"""
    
    config = SharpeRewardConfig(
        lookbook_window=lookback_window,
        use_downside_deviation=True
    )
    
    return SortinoReward(config, minimum_acceptable_return)


def create_information_ratio_reward(
    benchmark_returns: List[float],
    lookback_window: int = 50
) -> InformationRatioReward:
    """Create Information ratio reward"""
    
    config = SharpeRewardConfig(
        lookback_window=lookback_window,
        use_benchmark=True
    )
    
    return InformationRatioReward(config, benchmark_returns)


def create_adaptive_sharpe_reward(
    initial_target: float = 1.0,
    lookback_window: int = 50
) -> AdaptiveSharpeReward:
    """Create adaptive Sharpe reward"""
    
    config = SharpeRewardConfig(
        lookback_window=lookback_window,
        target_sharpe=initial_target,
        enable_reward_smoothing=True
    )
    
    return AdaptiveSharpeReward(config)


__all__ = [
    "SharpeRewardConfig",
    "SharpeReward",
    "InformationRatioReward",
    "SortinoReward", 
    "CalmarReward",
    "AdaptiveSharpeReward",
    "create_sharpe_reward",
    "create_sortino_reward",
    "create_information_ratio_reward",
    "create_adaptive_sharpe_reward"
]