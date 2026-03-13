"""
Reward Shaping utilities for curiosity-driven exploration.

Implements advanced reward shaping techniques with enterprise patterns
for optimal learning signal formation in crypto trading.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Any, Callable
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardShapingConfig:
    """Configuration for reward shaping."""
    
    # Reward components
    extrinsic_weight: float = 0.7
    intrinsic_weight: float = 0.3
    
    # Shaping parameters
    curiosity_bonus_scale: float = 1.0
    exploration_bonus_scale: float = 0.5
    risk_penalty_scale: float = 0.1
    
    # Normalization
    reward_normalization: bool = True
    reward_clipping: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)


class CryptoRewardShaper:
    """
    Advanced reward shaping for crypto trading with curiosity.
    
    Applies design pattern "Reward Engineering" for
    optimal learning signal formation.
    """
    
    def __init__(self, config: RewardShapingConfig):
        self.config = config
        
        # Running statistics for normalization
        self.reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        }
        
        # Component tracking
        self.component_history = {
            'extrinsic': [],
            'intrinsic': [],
            'total': []
        }
        
        logger.info("Crypto reward shaper initialized")
    
    def shape_reward(
        self,
        extrinsic_reward: float,
        intrinsic_reward: float,
        curiosity_bonus: float,
        exploration_bonus: float,
        risk_penalty: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Shape reward from multiple components.
        
        Args:
            extrinsic_reward: Base trading reward
            intrinsic_reward: Curiosity-driven intrinsic reward
            curiosity_bonus: Additional curiosity bonus
            exploration_bonus: Exploration bonus
            risk_penalty: Risk-based penalty
            context: Additional context
            
        Returns:
            Dictionary with shaped rewards and breakdown
        """
        if context is None:
            context = {}
        
        # Scale components
        scaled_curiosity = curiosity_bonus * self.config.curiosity_bonus_scale
        scaled_exploration = exploration_bonus * self.config.exploration_bonus_scale
        scaled_risk_penalty = risk_penalty * self.config.risk_penalty_scale
        
        # Market regime adjustment
        market_regime = context.get('market_regime', 'neutral')
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # Volatility adjustment
        volatility = context.get('volatility', 0.0)
        volatility_bonus = min(volatility * 0.1, 0.2)  # Cap bonus
        
        # Portfolio performance adjustment
        portfolio_return = context.get('portfolio_return', 0.0)
        performance_bonus = np.tanh(portfolio_return * 10) * 0.1  # Bounded bonus
        
        # Combine intrinsic components
        total_intrinsic = (
            intrinsic_reward + 
            scaled_curiosity + 
            scaled_exploration + 
            volatility_bonus + 
            performance_bonus - 
            scaled_risk_penalty
        )
        
        # Weight extrinsic vs intrinsic
        shaped_reward = (
            self.config.extrinsic_weight * extrinsic_reward + 
            self.config.intrinsic_weight * total_intrinsic
        ) * regime_multiplier
        
        # Normalization
        if self.config.reward_normalization:
            shaped_reward = self._normalize_reward(shaped_reward)
        
        # Clipping
        if self.config.reward_clipping:
            shaped_reward = np.clip(
                shaped_reward, 
                self.config.clip_range[0], 
                self.config.clip_range[1]
            )
        
        # Update statistics
        self._update_stats(extrinsic_reward, total_intrinsic, shaped_reward)
        
        # Reward breakdown
        breakdown = {
            'extrinsic_reward': extrinsic_reward,
            'intrinsic_reward': intrinsic_reward,
            'curiosity_bonus': scaled_curiosity,
            'exploration_bonus': scaled_exploration,
            'volatility_bonus': volatility_bonus,
            'performance_bonus': performance_bonus,
            'risk_penalty': scaled_risk_penalty,
            'total_intrinsic': total_intrinsic,
            'regime_multiplier': regime_multiplier,
            'shaped_reward': shaped_reward,
            'normalization_applied': self.config.reward_normalization,
            'clipping_applied': self.config.reward_clipping
        }
        
        return breakdown
    
    def _get_regime_multiplier(self, market_regime: str) -> float:
        """Multiplier on basis market regime."""
        regime_multipliers = {
            'bull': 1.0,
            'bear': 1.2,  # Higher rewards in bear market for exploration
            'sideways': 0.9,
            'volatile': 1.1,
            'neutral': 1.0
        }
        
        return regime_multipliers.get(market_regime, 1.0)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics."""
        # Update running statistics
        self.reward_stats['count'] += 1
        alpha = 1.0 / self.reward_stats['count']
        
        old_mean = self.reward_stats['mean']
        self.reward_stats['mean'] += alpha * (reward - old_mean)
        
        # Update variance (simplified)
        if self.reward_stats['count'] > 1:
            self.reward_stats['std'] = (
                (1 - alpha) * self.reward_stats['std']**2 + 
                alpha * (reward - old_mean)**2
            )**0.5
        
        # Normalize
        if self.reward_stats['std'] > 1e-8:
            normalized = (reward - self.reward_stats['mean']) / self.reward_stats['std']
        else:
            normalized = reward
        
        return normalized
    
    def _update_stats(
        self, 
        extrinsic: float, 
        intrinsic: float, 
        shaped: float
    ) -> None:
        """Update component statistics."""
        self.component_history['extrinsic'].append(extrinsic)
        self.component_history['intrinsic'].append(intrinsic)
        self.component_history['total'].append(shaped)
        
        # Keep only recent history
        max_history = 10000
        for key in self.component_history:
            if len(self.component_history[key]) > max_history:
                self.component_history[key] = self.component_history[key][-max_history//2:]
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get reward statistics."""
        stats = {
            'normalization_stats': self.reward_stats.copy(),
            'component_statistics': {}
        }
        
        for component, history in self.component_history.items():
            if history:
                stats['component_statistics'][component] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'min': min(history),
                    'max': max(history),
                    'count': len(history)
                }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset reward statistics."""
        self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        self.component_history = {
            'extrinsic': [],
            'intrinsic': [],
            'total': []
        }
        
        logger.info("Reward shaper statistics reset")


def create_reward_shaper(config: RewardShapingConfig) -> CryptoRewardShaper:
    """Factory function for creation reward shaper."""
    return CryptoRewardShaper(config)


if __name__ == "__main__":
    config = RewardShapingConfig(
        extrinsic_weight=0.7,
        intrinsic_weight=0.3,
        reward_normalization=True
    )
    
    shaper = create_reward_shaper(config)
    
    # Test reward shaping
    for i in range(100):
        extrinsic = np.random.randn() * 0.01
        intrinsic = np.random.exponential(0.1)
        curiosity = np.random.exponential(0.05)
        exploration = np.random.exponential(0.03)
        risk = np.random.uniform(0, 0.1)
        
        context = {
            'market_regime': np.random.choice(['bull', 'bear', 'sideways']),
            'volatility': np.random.uniform(0, 0.5),
            'portfolio_return': np.random.randn() * 0.02
        }
        
        breakdown = shaper.shape_reward(
            extrinsic_reward=extrinsic,
            intrinsic_reward=intrinsic,
            curiosity_bonus=curiosity,
            exploration_bonus=exploration,
            risk_penalty=risk,
            context=context
        )
        
        if i % 20 == 0:
            print(f"Step {i}: Shaped reward = {breakdown['shaped_reward']:.4f}")
    
    # Statistics
    stats = shaper.get_reward_statistics()
    print("\nReward Statistics:")
    for component, component_stats in stats['component_statistics'].items():
        print(f"{component}: mean={component_stats['mean']:.4f}, std={component_stats['std']:.4f}")