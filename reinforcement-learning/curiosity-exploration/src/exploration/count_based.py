"""
Count-Based Exploration for crypto trading environments.

Implements exploration strategies based on state visitation counts
with enterprise patterns for scalable exploration tracking.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
import hashlib
import pickle
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CountBasedConfig:
    """Configuration for count-based exploration."""
    
    # State discretization parameters
    state_discretization_bins: int = 100
    hash_state_dim: int = 32
    state_tolerance: float = 0.1
    
    # Exploration bonus parameters
    count_bonus_coefficient: float = 0.1
    count_bonus_power: float = 0.5
    max_count_bonus: float = 1.0
    min_count_bonus: float = 0.001
    
    # Adaptive parameters
    adaptive_discretization: bool = True
    discretization_update_frequency: int = 1000
    state_density_threshold: float = 0.01
    
    # Crypto-specific parameters
    market_regime_separation: bool = True
    portfolio_value_buckets: int = 20
    risk_level_buckets: int = 10
    temporal_state_window: int = 5
    
    # Performance optimization
    use_hash_table: bool = True
    hash_function: str = "md5"  # "md5", "sha256", "xxhash"
    max_states_memory: int = 1000000
    
    #  enterprise settings
    distributed_counting: bool = True
    persistent_storage: bool = True
    compression_enabled: bool = True


class StateDiscretizer(ABC):
    """
    Abstract base class for state discretization.
    
    Applies design pattern "Strategy Pattern" for
    flexible state representation strategies.
    """
    
    @abstractmethod
    def discretize(self, state: np.ndarray) -> Union[str, int, tuple]:
        """Discretization state."""
        pass
    
    @abstractmethod
    def update(self, states: np.ndarray) -> None:
        """Update discretizer parameters."""
        pass


class AdaptiveGridDiscretizer(StateDiscretizer):
    """
    Adaptive grid discretization for continuous states.
    
    Uses design pattern "Adaptive Systems" for
    dynamic adjustment discretization parameters.
    """
    
    def __init__(self, config: CountBasedConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.bins = config.state_discretization_bins
        
        # Adaptive for each dimension
        self.state_mins = np.full(state_dim, np.inf)
        self.state_maxs = np.full(state_dim, -np.inf)
        self.state_means = np.zeros(state_dim)
        self.state_stds = np.ones(state_dim)
        
        # Statistics for adaptation
        self.update_count = 0
        self.state_history = []
        
        logger.info(f"Adaptive grid discretizer initialized for {state_dim}D state space")
    
    def discretize(self, state: np.ndarray) -> tuple:
        """
        Discretization state in grid coordinates.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple with coordinates
        """
        # Normalization state
        normalized_state = (state - self.state_means) / (self.state_stds + 1e-8)
        
        # Limitation in reasonable range
        clipped_state = np.clip(normalized_state, -3, 3)
        
        # Discretization each dimension
        discrete_coords = []
        for i, value in enumerate(clipped_state):
            # Mapping [-3, 3] to [0, bins-1]
            bin_idx = int((value + 3) / 6 * (self.bins - 1))
            bin_idx = np.clip(bin_idx, 0, self.bins - 1)
            discrete_coords.append(bin_idx)
        
        return tuple(discrete_coords)
    
    def update(self, states: np.ndarray) -> None:
        """
        Update discretization parameters on basis new states.
        
        Args:
            states: Batch of states [batch_size, state_dim]
        """
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        
        # Update boundaries
        batch_mins = np.min(states, axis=0)
        batch_maxs = np.max(states, axis=0)
        
        self.state_mins = np.minimum(self.state_mins, batch_mins)
        self.state_maxs = np.maximum(self.state_maxs, batch_maxs)
        
        # Incremental update statistics
        alpha = 0.01  # Learning rate for statistics
        batch_means = np.mean(states, axis=0)
        batch_stds = np.std(states, axis=0)
        
        if self.update_count == 0:
            self.state_means = batch_means
            self.state_stds = batch_stds
        else:
            self.state_means = (1 - alpha) * self.state_means + alpha * batch_means
            self.state_stds = (1 - alpha) * self.state_stds + alpha * batch_stds
        
        self.update_count += 1
        
        # Save for adaptive binning
        if self.config.adaptive_discretization and len(self.state_history) < 10000:
            self.state_history.extend(states.tolist())


class HashBasedDiscretizer(StateDiscretizer):
    """
    Hash-based state discretization for high-dimensional states.
    
    Implements design pattern "Dimensionality Reduction" through
    locality-sensitive hashing.
    """
    
    def __init__(self, config: CountBasedConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.hash_dim = config.hash_state_dim
        self.tolerance = config.state_tolerance
        
        # Random for LSH
        self.random_projections = np.random.randn(state_dim, self.hash_dim)
        self.projection_biases = np.random.uniform(0, 2 * np.pi, self.hash_dim)
        
        # Normalization parameters
        self.state_mean = np.zeros(state_dim)
        self.state_std = np.ones(state_dim)
        self.normalization_count = 0
        
        logger.info(f"Hash-based discretizer initialized: {state_dim}D -> {self.hash_dim}D")
    
    def discretize(self, state: np.ndarray) -> str:
        """
        Hash-based discretization state.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Hash string discrete state
        """
        # Normalization state
        normalized_state = (state - self.state_mean) / (self.state_std + 1e-8)
        
        # LSH projection
        projections = np.dot(normalized_state, self.random_projections)
        
        # Create binary hash through sign function
        hash_bits = np.sign(np.sin(projections + self.projection_biases))
        
        # Convert in string hash
        hash_string = ''.join(['1' if bit > 0 else '0' for bit in hash_bits])
        
        return hash_string
    
    def update(self, states: np.ndarray) -> None:
        """Update normalization parameters."""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        
        # Incremental update statistics
        batch_size = states.shape[0]
        batch_mean = np.mean(states, axis=0)
        batch_std = np.std(states, axis=0)
        
        # Online mean and std update
        total_count = self.normalization_count + batch_size
        alpha = batch_size / total_count
        
        self.state_mean = (1 - alpha) * self.state_mean + alpha * batch_mean
        
        # Variance update
        if self.normalization_count > 0:
            self.state_std = np.sqrt(
                (1 - alpha) * self.state_std**2 + 
                alpha * batch_std**2 + 
                alpha * (1 - alpha) * (batch_mean - self.state_mean)**2
            )
        else:
            self.state_std = batch_std
        
        self.normalization_count = total_count


class CryptoStateDiscretizer(StateDiscretizer):
    """
    Specialized discretizer for crypto trading states.
    
    Applies design pattern "Domain-Specific Processing" for
    optimal representation financial data.
    """
    
    def __init__(self, config: CountBasedConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        
        # Split state dimensions on components
        self.market_dim = int(state_dim * 0.6)  # 60% - market data
        self.portfolio_dim = int(state_dim * 0.25)  # 25% - portfolio
        self.risk_dim = state_dim - self.market_dim - self.portfolio_dim  # Risk metrics
        
        # Discretizers for each component
        self.market_discretizer = AdaptiveGridDiscretizer(
            config, self.market_dim
        )
        self.portfolio_discretizer = HashBasedDiscretizer(
            config, self.portfolio_dim
        )
        self.risk_discretizer = AdaptiveGridDiscretizer(
            config, self.risk_dim
        )
        
        # Crypto-specific parameters
        self.portfolio_value_bins = config.portfolio_value_buckets
        self.risk_level_bins = config.risk_level_buckets
        
        logger.info(f"Crypto state discretizer: market={self.market_dim}, "
                   f"portfolio={self.portfolio_dim}, risk={self.risk_dim}")
    
    def discretize(self, state: np.ndarray) -> tuple:
        """
        Discretization with consideration crypto trading specifics.
        
        Args:
            state: Trading state vector
            
        Returns:
            Tuple (market_discrete, portfolio_hash, risk_discrete, meta_features)
        """
        # Split state on components
        market_data = state[:self.market_dim]
        portfolio_data = state[self.market_dim:self.market_dim + self.portfolio_dim]
        risk_data = state[-self.risk_dim:]
        
        # Discretization each component
        market_discrete = self.market_discretizer.discretize(market_data)
        portfolio_hash = self.portfolio_discretizer.discretize(portfolio_data)
        risk_discrete = self.risk_discretizer.discretize(risk_data)
        
        # Additional meta-features for crypto
        portfolio_value = np.sum(portfolio_data) if len(portfolio_data) > 0 else 0.0
        portfolio_bucket = min(int(portfolio_value * self.portfolio_value_bins), 
                              self.portfolio_value_bins - 1)
        
        risk_level = np.mean(risk_data) if len(risk_data) > 0 else 0.0
        risk_bucket = min(int(abs(risk_level) * self.risk_level_bins), 
                         self.risk_level_bins - 1)
        
        # Merging in composite key
        composite_key = (
            market_discrete,
            portfolio_hash,
            risk_discrete,
            portfolio_bucket,
            risk_bucket
        )
        
        return composite_key
    
    def update(self, states: np.ndarray) -> None:
        """Update all component discretizers."""
        if len(states.shape) == 1:
            states = states.reshape(1, -1)
        
        # Split states on components
        market_data = states[:, :self.market_dim]
        portfolio_data = states[:, self.market_dim:self.market_dim + self.portfolio_dim]
        risk_data = states[:, -self.risk_dim:]
        
        # Update each discretizer
        self.market_discretizer.update(market_data)
        self.portfolio_discretizer.update(portfolio_data)
        self.risk_discretizer.update(risk_data)


class CountBasedExplorer:
    """
    Count-based exploration system with advanced state tracking.
    
    Uses design pattern "Exploration Strategy" for
    intelligent exploration in crypto trading environments.
    """
    
    def __init__(self, config: CountBasedConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        
        # Select discretizer strategy
        if config.hash_state_dim < state_dim // 4:
            self.discretizer = HashBasedDiscretizer(config, state_dim)
            logger.info("Using hash-based discretization")
        else:
            self.discretizer = CryptoStateDiscretizer(config, state_dim)
            logger.info("Using crypto-specific discretization")
        
        # State visit counts
        self.state_counts = defaultdict(int)
        self.total_visits = 0
        
        # Market regime separation
        self.regime_counts = defaultdict(lambda: defaultdict(int))
        
        # Temporal tracking
        self.state_history = deque(maxlen=config.temporal_state_window)
        self.temporal_patterns = defaultdict(int)
        
        # Performance optimization
        self.update_frequency = config.discretization_update_frequency
        self.update_counter = 0
        
        # Statistics tracking
        self.exploration_coverage = 0.0
        self.unique_states_visited = set()
        self.state_densities = {}
        
        logger.info(f"Count-based explorer initialized for {state_dim}D state space")
    
    def get_count_bonus(
        self,
        state: np.ndarray,
        market_regime: Optional[int] = None
    ) -> float:
        """
        Computation count-based exploration bonus.
        
        Args:
            state: Current state
            market_regime: Optional market regime identifier
            
        Returns:
            Exploration bonus value
        """
        # Discretization state
        discrete_state = self.discretizer.discretize(state)
        
        # Get count (with consideration market regime if )
        if market_regime is not None and self.config.market_regime_separation:
            count = self.regime_counts[market_regime][discrete_state]
        else:
            count = self.state_counts[discrete_state]
        
        # Computation exploration bonus
        if count == 0:
            bonus = self.config.max_count_bonus
        else:
            # Usage -count
            bonus = self.config.count_bonus_coefficient / (count ** self.config.count_bonus_power)
            bonus = np.clip(bonus, self.config.min_count_bonus, self.config.max_count_bonus)
        
        return bonus
    
    def update_counts(
        self,
        state: np.ndarray,
        market_regime: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Update state visit counts.
        
        Args:
            state: Visited state
            market_regime: Market regime identifier
            
        Returns:
            Update statistics
        """
        # Discretization state
        discrete_state = self.discretizer.discretize(state)
        
        # Update counts
        old_count = self.state_counts[discrete_state]
        self.state_counts[discrete_state] += 1
        self.total_visits += 1
        
        # Market regime specific counting
        if market_regime is not None and self.config.market_regime_separation:
            self.regime_counts[market_regime][discrete_state] += 1
        
        # Tracking unique states
        self.unique_states_visited.add(discrete_state)
        
        # Temporal pattern tracking
        self.state_history.append(discrete_state)
        if len(self.state_history) >= 2:
            # Track state transitions
            transition = (self.state_history[-2], self.state_history[-1])
            self.temporal_patterns[transition] += 1
        
        # Periodic discretizer update
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:
            self.discretizer.update(state.reshape(1, -1))
        
        # Computation exploration coverage
        total_possible_states = len(self.state_counts)
        if total_possible_states > 0:
            self.exploration_coverage = len(self.unique_states_visited) / total_possible_states
        
        update_stats = {
            'discrete_state': str(discrete_state),
            'old_count': old_count,
            'new_count': self.state_counts[discrete_state],
            'total_visits': self.total_visits,
            'unique_states': len(self.unique_states_visited),
            'exploration_coverage': self.exploration_coverage,
            'market_regime': market_regime
        }
        
        return update_stats
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics exploration."""
        if len(self.state_counts) == 0:
            return {}
        
        counts_array = np.array(list(self.state_counts.values()))
        
        stats = {
            'total_states_visited': len(self.unique_states_visited),
            'total_visits': self.total_visits,
            'exploration_coverage': self.exploration_coverage,
            'count_statistics': {
                'mean': np.mean(counts_array),
                'std': np.std(counts_array),
                'min': np.min(counts_array),
                'max': np.max(counts_array),
                'median': np.median(counts_array)
            },
            'state_distribution': {
                'novel_states': np.sum(counts_array == 1),
                'rare_states': np.sum(counts_array <= 5),
                'common_states': np.sum(counts_array > 20),
                'very_common_states': np.sum(counts_array > 100)
            },
            'temporal_patterns': len(self.temporal_patterns),
            'market_regimes': len(self.regime_counts) if self.config.market_regime_separation else 0
        }
        
        # Market regime specific statistics
        if self.config.market_regime_separation:
            regime_stats = {}
            for regime, regime_counts in self.regime_counts.items():
                regime_array = np.array(list(regime_counts.values()))
                regime_stats[f'regime_{regime}'] = {
                    'unique_states': len(regime_counts),
                    'total_visits': np.sum(regime_array),
                    'mean_count': np.mean(regime_array),
                    'max_count': np.max(regime_array)
                }
            stats['regime_statistics'] = regime_stats
        
        return stats
    
    def get_state_density_map(self, top_k: int = 100) -> Dict[str, float]:
        """
        Get density map most visited states.
        
        Args:
            top_k: Number top states for return
            
        Returns:
            Dictionary {state: normalized_density}
        """
        if len(self.state_counts) == 0:
            return {}
        
        # Sort states by count
        sorted_states = sorted(
            self.state_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        # Normalization densities
        total_counts = sum(count for _, count in sorted_states)
        density_map = {
            str(state): count / total_counts 
            for state, count in sorted_states
        }
        
        return density_map
    
    def save_counts(self, filepath: str) -> None:
        """Save state counts in file."""
        data = {
            'state_counts': dict(self.state_counts),
            'regime_counts': {k: dict(v) for k, v in self.regime_counts.items()},
            'temporal_patterns': dict(self.temporal_patterns),
            'total_visits': self.total_visits,
            'unique_states_visited': list(self.unique_states_visited),
            'exploration_coverage': self.exploration_coverage,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Count data saved to {filepath}")
    
    def load_counts(self, filepath: str) -> None:
        """Load state counts from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.state_counts = defaultdict(int, data['state_counts'])
        self.regime_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data['regime_counts'].items()}
        )
        self.temporal_patterns = defaultdict(int, data['temporal_patterns'])
        self.total_visits = data['total_visits']
        self.unique_states_visited = set(data['unique_states_visited'])
        self.exploration_coverage = data['exploration_coverage']
        
        logger.info(f"Count data loaded from {filepath}")


class CountBasedEnvironment:
    """
    Environment wrapper with count-based exploration bonuses.
    
    Integrates design pattern "Environment Augmentation" for
    enhanced exploration in crypto trading.
    """
    
    def __init__(
        self,
        base_env,
        config: CountBasedConfig,
        exploration_weight: float = 0.1
    ):
        self.base_env = base_env
        self.config = config
        self.exploration_weight = exploration_weight
        
        # Get state dimension from environment
        if hasattr(base_env, 'observation_space'):
            if hasattr(base_env.observation_space, 'shape'):
                state_dim = base_env.observation_space.shape[0]
            else:
                state_dim = base_env.observation_space.n
        else:
            # Fallback for custom environments
            state_dim = 256
        
        self.explorer = CountBasedExplorer(config, state_dim)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_exploration_bonuses = []
        
        logger.info(f"Count-based environment initialized with exploration weight: {exploration_weight}")
    
    def step(self, action):
        """Step with count-based exploration bonus."""
        # Execute action in base environment
        next_state, extrinsic_reward, done, info = self.base_env.step(action)
        
        # Get market regime from info if available
        market_regime = info.get('market_regime', None)
        
        # Computation exploration bonus
        exploration_bonus = self.explorer.get_count_bonus(
            next_state, market_regime
        )
        
        # Update counts
        update_stats = self.explorer.update_counts(
            next_state, market_regime
        )
        
        # Total reward
        total_reward = extrinsic_reward + self.exploration_weight * exploration_bonus
        
        # Save exploration bonus
        self.episode_exploration_bonuses.append(exploration_bonus)
        
        # Update info
        info.update({
            'exploration_bonus': exploration_bonus,
            'extrinsic_reward': extrinsic_reward,
            'total_reward': total_reward,
            'state_count': update_stats['new_count'],
            'unique_states_visited': update_stats['unique_states'],
            'exploration_coverage': update_stats['exploration_coverage']
        })
        
        if done:
            info['episode_exploration_bonus_sum'] = sum(self.episode_exploration_bonuses)
            info['episode_exploration_bonus_mean'] = np.mean(self.episode_exploration_bonuses)
        
        return next_state, total_reward, done, info
    
    def reset(self):
        """Reset environment."""
        state = self.base_env.reset()
        
        # Reset episode tracking
        self.current_episode += 1
        self.episode_exploration_bonuses = []
        
        return state
    
    def get_exploration_report(self) -> Dict[str, Any]:
        """Get detailed report about exploration."""
        base_stats = self.explorer.get_exploration_statistics()
        density_map = self.explorer.get_state_density_map(50)
        
        report = {
            'episode': self.current_episode,
            'exploration_weight': self.exploration_weight,
            'statistics': base_stats,
            'top_states_density': density_map,
            'config': self.config
        }
        
        return report


def create_count_based_system(
    config: CountBasedConfig,
    state_dim: int
) -> CountBasedExplorer:
    """
    Factory function for creation count-based exploration system.
    
    Args:
        config: Count-based configuration
        state_dim: Dimensionality of state space
        
    Returns:
        Configured count-based explorer
    """
    explorer = CountBasedExplorer(config, state_dim)
    
    logger.info("Count-based exploration system created successfully")
    logger.info(f"State dimensionality: {state_dim}")
    logger.info(f"Discretization strategy: {type(explorer.discretizer).__name__}")
    
    return explorer


if __name__ == "__main__":
    # Example use count-based exploration
    config = CountBasedConfig(
        state_discretization_bins=50,
        hash_state_dim=16,
        count_bonus_coefficient=0.1,
        market_regime_separation=True
    )
    
    state_dim = 128
    explorer = create_count_based_system(config, state_dim)
    
    # Simulation exploration
    for episode in range(5):
        for step in range(100):
            # Random state
            state = np.random.randn(state_dim)
            market_regime = episode % 3 # 3 different regime
            
            # Get exploration bonus
            bonus = explorer.get_count_bonus(state, market_regime)
            
            # Update counts
            update_stats = explorer.update_counts(state, market_regime)
            
            if step % 50 == 0:
                print(f"Episode {episode}, Step {step}: "
                      f"Bonus={bonus:.4f}, Count={update_stats['new_count']}")
    
    # Statistics exploration
    stats = explorer.get_exploration_statistics()
    print("\nExploration Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Density map
    density_map = explorer.get_state_density_map(10)
    print("\nTop 10 visited states:")
    for state, density in density_map.items():
        print(f"State: {state[:50]}... Density: {density:.4f}")