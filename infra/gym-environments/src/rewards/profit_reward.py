"""
Profit-Based Reward Functions
enterprise patterns for robust profit optimization

Simple but effective profit-based rewards with various modifications:
- Raw profit/loss rewards
- Risk-adjusted profit rewards
- Logarithmic profit rewards
- Asymmetric profit/loss rewards
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ..utils.risk_metrics import calculate_sharpe_ratio, calculate_max_drawdown


@dataclass
class ProfitRewardConfig:
    """Configuration for profit-based rewards"""
    
    # Base reward scaling
    reward_scale: float = 1.0
    profit_scale: float = 100.0  # Scale portfolio returns to reasonable reward range
    
    # Risk adjustments
    enable_risk_penalty: bool = True
    drawdown_penalty_factor: float = 2.0
    volatility_penalty_factor: float = 0.1
    
    # Transaction cost penalties
    enable_transaction_penalty: bool = True
    transaction_penalty_factor: float = 1.0
    
    # Asymmetric rewards
    enable_asymmetric_rewards: bool = False
    loss_penalty_multiplier: float = 2.0  # Losses weighted higher than gains
    
    # Logarithmic scaling
    enable_log_scaling: bool = False
    log_base: float = np.e
    
    # Performance thresholds
    profit_threshold: float = 0.001  # Minimum profit to trigger positive reward
    loss_threshold: float = -0.001   # Maximum loss before penalty kicks in


class ProfitReward:
    """
    Simple profit-based reward function
    
    Rewards based on portfolio value changes with optional risk adjustments
    """
    
    def __init__(self, config: ProfitRewardConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Track reward history for analysis
        self.reward_history = []
        self.portfolio_history = []
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        trade_info: Dict[str, Any],
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate profit-based reward"""
        
        # Base profit calculation
        profit = portfolio_value - previous_portfolio_value
        profit_return = profit / previous_portfolio_value if previous_portfolio_value > 0 else 0.0
        
        # Scale to reasonable reward range
        base_reward = profit_return * self.config.profit_scale
        
        # Apply logarithmic scaling if enabled
        if self.config.enable_log_scaling:
            base_reward = self._apply_log_scaling(base_reward)
        
        # Apply asymmetric rewards if enabled
        if self.config.enable_asymmetric_rewards:
            base_reward = self._apply_asymmetric_scaling(base_reward)
        
        # Apply risk penalties
        risk_penalty = 0.0
        if self.config.enable_risk_penalty:
            risk_penalty = self._calculate_risk_penalty(
                portfolio_value, trade_info, market_info
            )
        
        # Apply transaction cost penalty
        transaction_penalty = 0.0
        if self.config.enable_transaction_penalty:
            transaction_penalty = self._calculate_transaction_penalty(trade_info)
        
        # Final reward
        total_reward = (base_reward - risk_penalty - transaction_penalty) * self.config.reward_scale
        
        # Track history
        self.reward_history.append(total_reward)
        self.portfolio_history.append(portfolio_value)
        
        self.logger.debug(f"Profit reward calculated", extra={
            "profit": profit,
            "profit_return": profit_return,
            "base_reward": base_reward,
            "risk_penalty": risk_penalty,
            "transaction_penalty": transaction_penalty,
            "total_reward": total_reward
        })
        
        return float(total_reward)
    
    def _apply_log_scaling(self, reward: float) -> float:
        """Apply logarithmic scaling to rewards"""
        
        if reward > 0:
            return np.log1p(reward) / np.log(self.config.log_base)
        elif reward < 0:
            return -np.log1p(-reward) / np.log(self.config.log_base)
        else:
            return 0.0
    
    def _apply_asymmetric_scaling(self, reward: float) -> float:
        """Apply asymmetric scaling (losses weighted more heavily)"""
        
        if reward < self.config.loss_threshold:
            return reward * self.config.loss_penalty_multiplier
        elif reward > self.config.profit_threshold:
            return reward
        else:
            return reward * 0.1  # Small rewards for neutral performance
    
    def _calculate_risk_penalty(
        self,
        portfolio_value: float,
        trade_info: Dict[str, Any],
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate risk-based penalty"""
        
        penalty = 0.0
        
        # Drawdown penalty
        if len(self.portfolio_history) > 1:
            max_value = max(self.portfolio_history)
            current_drawdown = (max_value - portfolio_value) / max_value
            penalty += current_drawdown * self.config.drawdown_penalty_factor
        
        # Volatility penalty (if enough history)
        if len(self.reward_history) >= 10:
            recent_rewards = self.reward_history[-10:]
            volatility = np.std(recent_rewards) if len(recent_rewards) > 1 else 0.0
            penalty += volatility * self.config.volatility_penalty_factor
        
        return penalty
    
    def _calculate_transaction_penalty(self, trade_info: Dict[str, Any]) -> float:
        """Calculate transaction cost penalty"""
        
        total_fees = trade_info.get("total_fees", 0.0)
        total_slippage = trade_info.get("total_slippage", 0.0)
        
        return (total_fees + total_slippage) * self.config.transaction_penalty_factor
    
    def reset(self) -> None:
        """Reset reward state"""
        self.reward_history.clear()
        self.portfolio_history.clear()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics"""
        
        if not self.reward_history:
            return {}
        
        rewards = np.array(self.reward_history)
        
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "cumulative_reward": float(np.sum(rewards)),
            "positive_rewards": float(np.sum(rewards > 0)),
            "negative_rewards": float(np.sum(rewards < 0)),
            "reward_sharpe": float(np.mean(rewards) / np.std(rewards)) if np.std(rewards) > 0 else 0.0
        }


class RiskAdjustedProfitReward(ProfitReward):
    """
    Risk-adjusted profit reward
    
    Incorporates Sharpe ratio and other risk metrics for better long-term performance
    """
    
    def __init__(self, config: ProfitRewardConfig, lookback_window: int = 50):
        super().__init__(config)
        self.lookback_window = lookback_window
        self.returns_history = []
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        trade_info: Dict[str, Any],
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate risk-adjusted reward"""
        
        # Calculate return
        portfolio_return = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
        self.returns_history.append(portfolio_return)
        
        # Keep only recent history
        if len(self.returns_history) > self.lookback_window:
            self.returns_history.pop(0)
        
        # Base profit reward
        base_reward = super().calculate_reward(
            portfolio_value, previous_portfolio_value, trade_info, market_info
        )
        
        # Risk adjustment
        if len(self.returns_history) >= 10:
            sharpe_ratio = calculate_sharpe_ratio(self.returns_history)
            risk_adjustment = np.tanh(sharpe_ratio)  # Smooth risk adjustment
            adjusted_reward = base_reward * (1 + risk_adjustment * 0.5)
        else:
            adjusted_reward = base_reward
        
        return float(adjusted_reward)
    
    def reset(self) -> None:
        """Reset state"""
        super().reset()
        self.returns_history.clear()


class CompoundReward:
    """
    Compound reward combining multiple profit-based strategies
    """
    
    def __init__(self, reward_configs: List[Tuple[ProfitReward, float]]):
        """
        Initialize with weighted list of reward functions
        
        Args:
            reward_configs: List of (reward_function, weight) tuples
        """
        self.reward_functions = reward_configs
        self.logger = logging.getLogger(__name__)
    
    def calculate_reward(
        self,
        portfolio_value: float,
        previous_portfolio_value: float,
        trade_info: Dict[str, Any],
        market_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate weighted compound reward"""
        
        total_reward = 0.0
        total_weight = 0.0
        
        for reward_fn, weight in self.reward_functions:
            component_reward = reward_fn.calculate_reward(
                portfolio_value, previous_portfolio_value, trade_info, market_info
            )
            
            total_reward += component_reward * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            final_reward = total_reward / total_weight
        else:
            final_reward = 0.0
        
        return float(final_reward)
    
    def reset(self) -> None:
        """Reset all component rewards"""
        for reward_fn, _ in self.reward_functions:
            reward_fn.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components"""
        
        stats = {"components": {}}
        
        for i, (reward_fn, weight) in enumerate(self.reward_functions):
            component_stats = reward_fn.get_statistics()
            stats["components"][f"component_{i}"] = {
                "weight": weight,
                "stats": component_stats
            }
        
        return stats


# Factory functions
def create_simple_profit_reward(
    profit_scale: float = 100.0,
    enable_risk_penalty: bool = True
) -> ProfitReward:
    """Create simple profit reward"""
    
    config = ProfitRewardConfig(
        profit_scale=profit_scale,
        enable_risk_penalty=enable_risk_penalty
    )
    
    return ProfitReward(config)


def create_risk_adjusted_profit_reward(
    profit_scale: float = 100.0,
    lookback_window: int = 50
) -> RiskAdjustedProfitReward:
    """Create risk-adjusted profit reward"""
    
    config = ProfitRewardConfig(
        profit_scale=profit_scale,
        enable_risk_penalty=True
    )
    
    return RiskAdjustedProfitReward(config, lookback_window)


def create_conservative_profit_reward() -> ProfitReward:
    """Create conservative profit reward with heavy risk penalties"""
    
    config = ProfitRewardConfig(
        profit_scale=50.0,
        enable_risk_penalty=True,
        drawdown_penalty_factor=3.0,
        enable_asymmetric_rewards=True,
        loss_penalty_multiplier=2.5,
        enable_transaction_penalty=True,
        transaction_penalty_factor=2.0
    )
    
    return ProfitReward(config)


def create_aggressive_profit_reward() -> ProfitReward:
    """Create aggressive profit reward with minimal penalties"""
    
    config = ProfitRewardConfig(
        profit_scale=200.0,
        enable_risk_penalty=False,
        enable_transaction_penalty=False,
        enable_asymmetric_rewards=False
    )
    
    return ProfitReward(config)


__all__ = [
    "ProfitRewardConfig",
    "ProfitReward",
    "RiskAdjustedProfitReward", 
    "CompoundReward",
    "create_simple_profit_reward",
    "create_risk_adjusted_profit_reward",
    "create_conservative_profit_reward",
    "create_aggressive_profit_reward"
]