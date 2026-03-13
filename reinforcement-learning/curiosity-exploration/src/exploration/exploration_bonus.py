"""
Exploration Bonus System for coordination various exploration strategies.

Implements unified framework for multiple exploration signals
with enterprise patterns for intelligent bonus allocation.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplorationStrategy(Enum):
    """Types exploration strategies."""
    COUNT_BASED = "count_based"
    PREDICTION_BASED = "prediction_based"
    CURIOSITY_DRIVEN = "curiosity_driven"
    RANDOM_NETWORK = "random_network"
    NEVER_GIVE_UP = "never_give_up"
    INFORMATION_GAIN = "information_gain"
    ENTROPY_BASED = "entropy_based"


@dataclass
class ExplorationBonusConfig:
    """Configuration for exploration bonus system."""
    
    # Strategy weights
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        ExplorationStrategy.COUNT_BASED.value: 0.2,
        ExplorationStrategy.PREDICTION_BASED.value: 0.3,
        ExplorationStrategy.CURIOSITY_DRIVEN.value: 0.3,
        ExplorationStrategy.RANDOM_NETWORK.value: 0.2
    })
    
    # Bonus scaling
    max_total_bonus: float = 2.0
    min_total_bonus: float = 0.001
    bonus_normalization_method: str = "adaptive"  # "adaptive", "fixed", "percentile"
    normalization_window: int = 1000
    
    # Adaptive weighting
    adaptive_weights: bool = True
    weight_update_frequency: int = 500
    performance_window: int = 2000
    strategy_effectiveness_threshold: float = 0.1
    
    # Crypto-specific parameters
    market_regime_bonus_scaling: Dict[str, float] = field(default_factory=lambda: {
        "bull": 1.2,
        "bear": 1.5,
        "sideways": 1.0,
        "volatile": 1.8
    })
    
    # Risk-aware exploration
    risk_adjusted_exploration: bool = True
    max_risk_multiplier: float = 2.0
    risk_aversion_factor: float = 0.5
    
    # Temporal exploration patterns
    temporal_decay_factor: float = 0.99
    exploration_schedule: str = "constant"  # "constant", "decay", "cyclic"
    exploration_cycle_length: int = 10000
    
    # Performance optimization
    strategy_caching: bool = True
    parallel_computation: bool = True
    precision_mode: str = "float32"  # "float16", "float32", "float64"
    
    #  enterprise settings
    distributed_bonus_computation: bool = True
    real_time_adaptation: bool = True
    bonus_history_storage: bool = True
    metrics_collection: bool = True


class ExplorationBonusCalculator(ABC):
    """
    Abstract base class for exploration bonus calculation.
    
    Applies design pattern "Strategy Pattern" for
    flexible bonus computation methods.
    """
    
    @abstractmethod
    def compute_bonus(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Computation exploration bonus."""
        pass
    
    @abstractmethod
    def update(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Update internal state calculator."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics calculator."""
        pass


class CountBasedBonusCalculator(ExplorationBonusCalculator):
    """Count-based exploration bonus calculator."""
    
    def __init__(self, config: ExplorationBonusConfig):
        self.config = config
        self.state_counts = defaultdict(int)
        self.total_visits = 0
        self.bonus_coefficient = 0.1
        
    def compute_bonus(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute count-based bonus."""
        # Simple discretization for demonstration
        state_hash = hash(state.tobytes())
        count = self.state_counts[state_hash]
        
        if count == 0:
            return self.config.max_total_bonus * 0.5
        else:
            bonus = self.bonus_coefficient / np.sqrt(count)
            return min(bonus, self.config.max_total_bonus * 0.5)
    
    def update(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Update count statistics."""
        state_hash = hash(state.tobytes())
        old_count = self.state_counts[state_hash]
        self.state_counts[state_hash] += 1
        self.total_visits += 1
        
        return {
            'old_count': old_count,
            'new_count': self.state_counts[state_hash],
            'total_visits': self.total_visits
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get count-based statistics."""
        if not self.state_counts:
            return {}
        
        counts = list(self.state_counts.values())
        return {
            'unique_states': len(self.state_counts),
            'total_visits': self.total_visits,
            'mean_count': np.mean(counts),
            'max_count': max(counts),
            'min_count': min(counts)
        }


class PredictionErrorBonusCalculator(ExplorationBonusCalculator):
    """Prediction error-based exploration bonus calculator."""
    
    def __init__(self, config: ExplorationBonusConfig):
        self.config = config
        self.prediction_errors = deque(maxlen=1000)
        self.error_mean = 0.0
        self.error_std = 1.0
        
    def compute_bonus(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute prediction error-based bonus."""
        # Get prediction error from context
        prediction_error = context.get('prediction_error', 0.0) if context else 0.0
        
        # Normalization by running statistics
        if len(self.prediction_errors) > 10:
            normalized_error = (prediction_error - self.error_mean) / (self.error_std + 1e-8)
            bonus = 0.2 * np.clip(normalized_error, 0, 5)
        else:
            bonus = 0.2 * prediction_error
        
        return min(bonus, self.config.max_total_bonus * 0.4)
    
    def update(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Update prediction error statistics."""
        prediction_error = context.get('prediction_error', 0.0) if context else 0.0
        
        self.prediction_errors.append(prediction_error)
        
        # Update running statistics
        if len(self.prediction_errors) > 1:
            errors = list(self.prediction_errors)
            self.error_mean = np.mean(errors)
            self.error_std = np.std(errors)
        
        return {
            'prediction_error': prediction_error,
            'error_mean': self.error_mean,
            'error_std': self.error_std
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction error statistics."""
        if not self.prediction_errors:
            return {}
        
        errors = list(self.prediction_errors)
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': max(errors),
            'min_error': min(errors),
            'count': len(errors)
        }


class ExplorationBonusManager:
    """
    Manager for coordination multiple exploration bonus calculators.
    
    Uses design pattern "Composite Strategy" for
    intelligent combination exploration signals.
    """
    
    def __init__(self, config: ExplorationBonusConfig):
        self.config = config
        
        # Initialize calculators for each strategy
        self.calculators = {
            ExplorationStrategy.COUNT_BASED.value: CountBasedBonusCalculator(config),
            ExplorationStrategy.PREDICTION_BASED.value: PredictionErrorBonusCalculator(config)
        }
        
        # Strategy weights (can )
        self.current_weights = config.strategy_weights.copy()
        
        # Performance tracking for adaptive weighting
        self.strategy_performance = defaultdict(lambda: deque(maxlen=config.performance_window))
        self.strategy_effectiveness = defaultdict(float)
        
        # Bonus history for normalization
        self.bonus_history = deque(maxlen=config.normalization_window)
        self.component_bonus_history = defaultdict(lambda: deque(maxlen=config.normalization_window))
        
        # Normalization statistics
        self.bonus_stats = {'mean': 0.0, 'std': 1.0}
        self.update_counter = 0
        
        # Market regime tracking
        self.current_market_regime = "sideways"
        self.regime_history = deque(maxlen=100)
        
        # Risk tracking
        self.current_risk_level = 0.5
        self.risk_history = deque(maxlen=100)
        
        # Performance optimization
        self.bonus_cache = {} if config.strategy_caching else None
        self.cache_hit_count = 0
        self.cache_total_requests = 0
        
        logger.info(f"Exploration bonus manager initialized with {len(self.calculators)} strategies")
    
    def compute_exploration_bonus(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computation combined exploration bonus.
        
        Args:
            state: Current state
            action: Current action
            context: Additional context information
            
        Returns:
            Tuple (total_bonus, component_breakdown)
        """
        # Check cache
        cache_key = None
        if self.bonus_cache is not None:
            state_hash = hash(state.tobytes())
            action_hash = hash(action.tobytes()) if action is not None else 0
            cache_key = (state_hash, action_hash)
            
            self.cache_total_requests += 1
            if cache_key in self.bonus_cache:
                self.cache_hit_count += 1
                return self.bonus_cache[cache_key]
        
        # Get context information
        if context is None:
            context = {}
        
        # Update market regime and risk level
        self._update_market_context(context)
        
        # Computation bonus from each strategy
        component_bonuses = {}
        total_weighted_bonus = 0.0
        
        for strategy, calculator in self.calculators.items():
            try:
                bonus = calculator.compute_bonus(state, action, context)
                component_bonuses[strategy] = bonus
                
                # Weighted contribution
                weight = self.current_weights.get(strategy, 0.0)
                total_weighted_bonus += weight * bonus
                
                # Save for history
                self.component_bonus_history[strategy].append(bonus)
                
            except Exception as e:
                logger.warning(f"Error computing bonus for strategy {strategy}: {e}")
                component_bonuses[strategy] = 0.0
        
        # Market regime adjustment
        regime_multiplier = self.config.market_regime_bonus_scaling.get(
            self.current_market_regime, 1.0
        )
        
        # Risk adjustment
        risk_multiplier = self._compute_risk_multiplier()
        
        # Temporal decay
        temporal_multiplier = self._compute_temporal_multiplier()
        
        # Final bonus calculation
        adjusted_bonus = total_weighted_bonus * regime_multiplier * risk_multiplier * temporal_multiplier
        
        # Normalization
        normalized_bonus = self._normalize_bonus(adjusted_bonus)
        
        # in range
        final_bonus = np.clip(
            normalized_bonus,
            self.config.min_total_bonus,
            self.config.max_total_bonus
        )
        
        # Save for history
        self.bonus_history.append(final_bonus)
        
        # Component breakdown
        breakdown = {
            **component_bonuses,
            'total_weighted': total_weighted_bonus,
            'regime_multiplier': regime_multiplier,
            'risk_multiplier': risk_multiplier,
            'temporal_multiplier': temporal_multiplier,
            'adjusted_bonus': adjusted_bonus,
            'normalized_bonus': normalized_bonus,
            'final_bonus': final_bonus,
            'current_weights': self.current_weights.copy()
        }
        
        # Save in cache
        if cache_key is not None:
            self.bonus_cache[cache_key] = (final_bonus, breakdown)
            
            # Limitation size cache
            if len(self.bonus_cache) > 5000:
                keys_to_remove = list(self.bonus_cache.keys())[:1000]
                for key in keys_to_remove:
                    del self.bonus_cache[key]
        
        return final_bonus, breakdown
    
    def update_exploration_bonus(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        reward: Optional[float] = None,
        next_state: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Update exploration bonus calculators.
        
        Args:
            state: Current state
            action: Executed action
            reward: Received reward
            next_state: Next state
            context: Additional context
            
        Returns:
            Update statistics
        """
        if context is None:
            context = {}
        
        update_stats = {}
        
        # Update each calculator
        for strategy, calculator in self.calculators.items():
            try:
                stats = calculator.update(state, action, reward, next_state, context)
                update_stats[f'{strategy}_update'] = stats
                
                # Tracking performance for adaptive weights
                if reward is not None:
                    self.strategy_performance[strategy].append(reward)
                
            except Exception as e:
                logger.warning(f"Error updating calculator {strategy}: {e}")
                update_stats[f'{strategy}_update'] = {'error': str(e)}
        
        self.update_counter += 1
        
        # Periodic adaptive weight updates
        if (self.config.adaptive_weights and 
            self.update_counter % self.config.weight_update_frequency == 0):
            self._update_adaptive_weights()
        
        # Periodic normalization stats update
        if self.update_counter % 100 == 0:
            self._update_normalization_stats()
        
        # Clear cache periodically
        if self.bonus_cache is not None and self.update_counter % 1000 == 0:
            self.bonus_cache.clear()
        
        # General statistics
        update_stats.update({
            'update_counter': self.update_counter,
            'current_market_regime': self.current_market_regime,
            'current_risk_level': self.current_risk_level,
            'cache_hit_rate': self.cache_hit_count / max(1, self.cache_total_requests),
            'bonus_mean': self.bonus_stats['mean'],
            'bonus_std': self.bonus_stats['std']
        })
        
        return update_stats
    
    def _update_market_context(self, context: Dict[str, Any]) -> None:
        """Update market regime and risk context."""
        # Market regime detection
        if 'market_regime' in context:
            self.current_market_regime = context['market_regime']
            self.regime_history.append(self.current_market_regime)
        
        # Risk level tracking
        if 'risk_level' in context:
            self.current_risk_level = context['risk_level']
            self.risk_history.append(self.current_risk_level)
        elif 'portfolio_volatility' in context:
            # Approximate risk from volatility
            volatility = context['portfolio_volatility']
            self.current_risk_level = np.clip(volatility, 0.0, 1.0)
            self.risk_history.append(self.current_risk_level)
    
    def _compute_risk_multiplier(self) -> float:
        """Computation risk-adjusted multiplier."""
        if not self.config.risk_adjusted_exploration:
            return 1.0
        
        # Higher exploration in higher risk scenarios (with limits)
        risk_bonus = 1.0 + (self.current_risk_level - 0.5) * self.config.risk_aversion_factor
        risk_multiplier = np.clip(risk_bonus, 1.0 / self.config.max_risk_multiplier, self.config.max_risk_multiplier)
        
        return risk_multiplier
    
    def _compute_temporal_multiplier(self) -> float:
        """Computation temporal decay multiplier."""
        if self.config.exploration_schedule == "constant":
            return 1.0
        elif self.config.exploration_schedule == "decay":
            return self.config.temporal_decay_factor ** (self.update_counter / 1000)
        elif self.config.exploration_schedule == "cyclic":
            cycle_position = (self.update_counter % self.config.exploration_cycle_length) / self.config.exploration_cycle_length
            return 0.5 + 0.5 * np.sin(2 * np.pi * cycle_position)
        else:
            return 1.0
    
    def _normalize_bonus(self, bonus: float) -> float:
        """Normalization bonus value."""
        if self.config.bonus_normalization_method == "fixed":
            return bonus
        elif self.config.bonus_normalization_method == "adaptive":
            if len(self.bonus_history) > 10:
                # Z-score normalization
                normalized = (bonus - self.bonus_stats['mean']) / (self.bonus_stats['std'] + 1e-8)
                return normalized
            else:
                return bonus
        elif self.config.bonus_normalization_method == "percentile":
            if len(self.bonus_history) > 100:
                percentile_90 = np.percentile(list(self.bonus_history), 90)
                return bonus / (percentile_90 + 1e-8)
            else:
                return bonus
        else:
            return bonus
    
    def _update_normalization_stats(self) -> None:
        """Update normalization statistics."""
        if len(self.bonus_history) > 10:
            bonuses = list(self.bonus_history)
            self.bonus_stats = {
                'mean': np.mean(bonuses),
                'std': np.std(bonuses)
            }
    
    def _update_adaptive_weights(self) -> None:
        """Update adaptive strategy weights."""
        if not self.config.adaptive_weights:
            return
        
        # Computation effectiveness each strategy
        total_effectiveness = 0.0
        strategy_scores = {}
        
        for strategy in self.calculators.keys():
            if strategy in self.strategy_performance:
                performance_history = list(self.strategy_performance[strategy])
                if len(performance_history) > 50:
                    # Effectiveness based on recent performance
                    recent_performance = performance_history[-100:]
                    effectiveness = np.mean(recent_performance) + np.std(recent_performance)
                    strategy_scores[strategy] = max(effectiveness, 0.0)
                    total_effectiveness += strategy_scores[strategy]
                else:
                    strategy_scores[strategy] = 1.0
                    total_effectiveness += 1.0
            else:
                strategy_scores[strategy] = 1.0
                total_effectiveness += 1.0
        
        # Update weights on basis effectiveness
        if total_effectiveness > 0:
            for strategy in self.current_weights.keys():
                if strategy in strategy_scores:
                    new_weight = strategy_scores[strategy] / total_effectiveness
                    # Smooth update
                    alpha = 0.1
                    self.current_weights[strategy] = (
                        (1 - alpha) * self.current_weights[strategy] + 
                        alpha * new_weight
                    )
        
        # Normalization weights
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for strategy in self.current_weights:
                self.current_weights[strategy] /= total_weight
        
        logger.info(f"Updated adaptive weights: {self.current_weights}")
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics exploration."""
        stats = {
            'total_updates': self.update_counter,
            'current_weights': self.current_weights.copy(),
            'current_market_regime': self.current_market_regime,
            'current_risk_level': self.current_risk_level,
            'bonus_statistics': {
                'mean': self.bonus_stats['mean'],
                'std': self.bonus_stats['std'],
                'count': len(self.bonus_history)
            },
            'cache_performance': {
                'hit_rate': self.cache_hit_count / max(1, self.cache_total_requests),
                'total_requests': self.cache_total_requests,
                'cache_size': len(self.bonus_cache) if self.bonus_cache else 0
            }
        }
        
        # Strategy-specific statistics
        for strategy, calculator in self.calculators.items():
            strategy_stats = calculator.get_statistics()
            stats[f'{strategy}_statistics'] = strategy_stats
            
            # Performance statistics
            if strategy in self.strategy_performance:
                performance_history = list(self.strategy_performance[strategy])
                if performance_history:
                    stats[f'{strategy}_performance'] = {
                        'mean': np.mean(performance_history),
                        'std': np.std(performance_history),
                        'count': len(performance_history)
                    }
        
        # Component bonus statistics
        for strategy, history in self.component_bonus_history.items():
            if history:
                stats[f'{strategy}_bonus_stats'] = {
                    'mean': np.mean(list(history)),
                    'std': np.std(list(history)),
                    'count': len(history)
                }
        
        # Market regime statistics
        if self.regime_history:
            regime_counts = defaultdict(int)
            for regime in self.regime_history:
                regime_counts[regime] += 1
            stats['market_regime_distribution'] = dict(regime_counts)
        
        # Risk statistics
        if self.risk_history:
            stats['risk_statistics'] = {
                'mean': np.mean(list(self.risk_history)),
                'std': np.std(list(self.risk_history)),
                'min': min(self.risk_history),
                'max': max(self.risk_history)
            }
        
        return stats
    
    def set_strategy_weight(self, strategy: str, weight: float) -> None:
        """Setup weights for specific strategy."""
        if strategy in self.current_weights:
            self.current_weights[strategy] = weight
            
            # Renormalize weights
            total_weight = sum(self.current_weights.values())
            if total_weight > 0:
                for s in self.current_weights:
                    self.current_weights[s] /= total_weight
            
            logger.info(f"Updated weight for {strategy}: {weight}")
        else:
            logger.warning(f"Strategy {strategy} not found")
    
    def add_strategy_calculator(
        self,
        strategy: str,
        calculator: ExplorationBonusCalculator,
        weight: float = 0.1
    ) -> None:
        """Add new strategy calculator."""
        self.calculators[strategy] = calculator
        self.current_weights[strategy] = weight
        
        # Renormalize weights
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for s in self.current_weights:
                self.current_weights[s] /= total_weight
        
        logger.info(f"Added strategy calculator: {strategy} with weight {weight}")
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint exploration bonus manager."""
        checkpoint_data = {
            'config': self.config,
            'current_weights': self.current_weights,
            'bonus_stats': self.bonus_stats,
            'update_counter': self.update_counter,
            'current_market_regime': self.current_market_regime,
            'current_risk_level': self.current_risk_level,
            'cache_hit_count': self.cache_hit_count,
            'cache_total_requests': self.cache_total_requests
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Exploration bonus manager checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint exploration bonus manager."""
        import pickle
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.current_weights = checkpoint_data['current_weights']
        self.bonus_stats = checkpoint_data['bonus_stats']
        self.update_counter = checkpoint_data['update_counter']
        self.current_market_regime = checkpoint_data['current_market_regime']
        self.current_risk_level = checkpoint_data['current_risk_level']
        self.cache_hit_count = checkpoint_data['cache_hit_count']
        self.cache_total_requests = checkpoint_data['cache_total_requests']
        
        logger.info(f"Exploration bonus manager checkpoint loaded from {filepath}")


class ExplorationBonusEnvironment:
    """
    Environment wrapper with unified exploration bonus system.
    
    Integrates design pattern "Environment Enhancement" for
    comprehensive exploration in crypto trading.
    """
    
    def __init__(
        self,
        base_env,
        config: ExplorationBonusConfig,
        exploration_weight: float = 0.15
    ):
        self.base_env = base_env
        self.config = config
        self.exploration_weight = exploration_weight
        
        # Initialize exploration bonus manager
        self.bonus_manager = ExplorationBonusManager(config)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_bonuses = []
        self.episode_extrinsic_rewards = []
        
        # Performance tracking
        self.total_exploration_bonus = 0.0
        self.total_extrinsic_reward = 0.0
        
        logger.info(f"Exploration bonus environment initialized with weight: {exploration_weight}")
    
    def step(self, action):
        """Step with unified exploration bonus."""
        # Execute action in base environment
        next_state, extrinsic_reward, done, info = self.base_env.step(action)
        
        # Preparation context for exploration bonus
        context = {
            'market_regime': info.get('market_regime', 'sideways'),
            'risk_level': info.get('risk_level', 0.5),
            'portfolio_volatility': info.get('portfolio_volatility', 0.3),
            'prediction_error': info.get('prediction_error', 0.0),
            'episode_step': info.get('episode_step', 0)
        }
        
        # Get current state from info or use next_state
        current_state = info.get('current_state', next_state)
        
        # Computation exploration bonus
        exploration_bonus, bonus_breakdown = self.bonus_manager.compute_exploration_bonus(
            state=current_state,
            action=action,
            context=context
        )
        
        # Update exploration bonus manager
        update_stats = self.bonus_manager.update_exploration_bonus(
            state=current_state,
            action=action,
            reward=extrinsic_reward,
            next_state=next_state,
            context=context
        )
        
        # Merging rewards
        total_reward = extrinsic_reward + self.exploration_weight * exploration_bonus
        
        # Tracking
        self.episode_bonuses.append(exploration_bonus)
        self.episode_extrinsic_rewards.append(extrinsic_reward)
        self.total_exploration_bonus += exploration_bonus
        self.total_extrinsic_reward += extrinsic_reward
        
        # Update info
        info.update({
            'exploration_bonus': exploration_bonus,
            'exploration_weight': self.exploration_weight,
            'extrinsic_reward': extrinsic_reward,
            'total_reward': total_reward,
            'bonus_breakdown': bonus_breakdown,
            'update_stats': update_stats
        })
        
        if done:
            # Episode summary statistics
            info.update({
                'episode_exploration_bonus_sum': sum(self.episode_bonuses),
                'episode_exploration_bonus_mean': np.mean(self.episode_bonuses),
                'episode_extrinsic_reward_sum': sum(self.episode_extrinsic_rewards),
                'episode_extrinsic_reward_mean': np.mean(self.episode_extrinsic_rewards),
                'exploration_contribution': (
                    sum(self.episode_bonuses) * self.exploration_weight /
                    (sum(self.episode_extrinsic_rewards) + sum(self.episode_bonuses) * self.exploration_weight + 1e-8)
                )
            })
        
        return next_state, total_reward, done, info
    
    def reset(self):
        """Reset environment."""
        state = self.base_env.reset()
        
        # Reset episode tracking
        self.current_episode += 1
        self.episode_bonuses = []
        self.episode_extrinsic_rewards = []
        
        return state
    
    def get_exploration_report(self) -> Dict[str, Any]:
        """Get detailed report about exploration."""
        bonus_stats = self.bonus_manager.get_exploration_statistics()
        
        report = {
            'episode': self.current_episode,
            'exploration_weight': self.exploration_weight,
            'total_exploration_bonus': self.total_exploration_bonus,
            'total_extrinsic_reward': self.total_extrinsic_reward,
            'exploration_manager_stats': bonus_stats,
            'config': self.config
        }
        
        return report


def create_exploration_bonus_system(config: ExplorationBonusConfig) -> ExplorationBonusManager:
    """
    Factory function for creation exploration bonus system.
    
    Args:
        config: Exploration bonus configuration
        
    Returns:
        Configured exploration bonus manager
    """
    manager = ExplorationBonusManager(config)
    
    logger.info("Exploration bonus system created successfully")
    logger.info(f"Active strategies: {list(manager.calculators.keys())}")
    logger.info(f"Strategy weights: {manager.current_weights}")
    
    return manager


if __name__ == "__main__":
    # Example use exploration bonus system
    config = ExplorationBonusConfig(
        strategy_weights={
            ExplorationStrategy.COUNT_BASED.value: 0.3,
            ExplorationStrategy.PREDICTION_BASED.value: 0.7
        },
        adaptive_weights=True,
        max_total_bonus=1.5
    )
    
    manager = create_exploration_bonus_system(config)
    
    # Simulation exploration
    for episode in range(3):
        for step in range(50):
            # Random data
            state = np.random.randn(128)
            action = np.random.randn(5)
            reward = np.random.randn() * 0.1
            next_state = np.random.randn(128)
            
            context = {
                'market_regime': ['bull', 'bear', 'sideways'][episode],
                'risk_level': np.random.uniform(0.2, 0.8),
                'prediction_error': np.random.exponential(0.5)
            }
            
            # Get exploration bonus
            bonus, breakdown = manager.compute_exploration_bonus(state, action, context)
            
            # Update manager
            update_stats = manager.update_exploration_bonus(
                state, action, reward, next_state, context
            )
            
            if step % 25 == 0:
                print(f"Episode {episode}, Step {step}: "
                      f"Bonus={bonus:.4f}, Weights={manager.current_weights}")
    
    # Statistics
    stats = manager.get_exploration_statistics()
    print("\nExploration Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")