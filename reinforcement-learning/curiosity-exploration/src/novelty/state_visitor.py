"""
State Visitor System for tracking exploration coverage.

Implements sophisticated tracking states with enterprise patterns
for comprehensive analysis exploration efficiency in crypto trading.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Any, Union, Set
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque, Counter
import time
import hashlib
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StateVisitorConfig:
    """Configuration for state visitor system."""
    
    # State representation
    state_representation: str = "hash"  # "hash", "cluster", "grid", "neural"
    hash_precision: int = 16
    grid_resolution: int = 50
    cluster_method: str = "kmeans"
    num_clusters: int = 1000
    
    # Visitation tracking
    track_temporal_patterns: bool = True
    track_transition_frequencies: bool = True
    track_return_times: bool = True
    max_history_length: int = 100000
    
    # Crypto-specific tracking
    market_regime_separation: bool = True
    portfolio_state_tracking: bool = True
    risk_level_tracking: bool = True
    temporal_context_length: int = 20
    
    # Analysis parameters
    coverage_analysis: bool = True
    density_estimation: bool = True
    exploration_efficiency_metrics: bool = True
    novelty_detection_integration: bool = True
    
    # Performance optimization
    batch_processing: bool = True
    parallel_computation: bool = True
    memory_efficient_storage: bool = True
    compression_enabled: bool = True
    
    # Visualization
    enable_visualization: bool = True
    plot_coverage_maps: bool = True
    plot_transition_graphs: bool = True
    save_visualizations: bool = True
    
    #  enterprise settings
    distributed_tracking: bool = True
    real_time_analysis: bool = True
    persistent_storage: bool = True
    metrics_export: bool = True


class StateRepresentation(ABC):
    """
    Abstract base class for state representation methods.
    
    Applies design pattern "Strategy Pattern" for
    flexible state encoding strategies.
    """
    
    @abstractmethod
    def encode_state(self, state: np.ndarray) -> Union[str, int, tuple]:
        """Encode state in compact representation."""
        pass
    
    @abstractmethod
    def decode_state(self, encoded_state: Union[str, int, tuple]) -> Optional[np.ndarray]:
        """Decoding state representation back (if )."""
        pass
    
    @abstractmethod
    def get_similarity(self, state1: Union[str, int, tuple], state2: Union[str, int, tuple]) -> float:
        """Computation similarity between encoded states."""
        pass


class HashStateRepresentation(StateRepresentation):
    """
    Hash-based state representation.
    
    Uses design pattern "Content Hashing" for
    efficient state identification.
    """
    
    def __init__(self, config: StateVisitorConfig):
        self.config = config
        self.precision = config.hash_precision
        
        # State normalization statistics
        self.state_mean = None
        self.state_std = None
        self.normalization_samples = 0
        
        logger.info(f"Hash state representation initialized with precision {self.precision}")
    
    def encode_state(self, state: np.ndarray) -> str:
        """Encode state through hashing."""
        # Normalization state
        normalized_state = self._normalize_state(state)
        
        # for consistent hashing
        quantized_state = np.round(normalized_state * (2 ** self.precision)).astype(np.int32)
        
        # MD5 hash
        state_bytes = quantized_state.tobytes()
        hash_object = hashlib.md5(state_bytes)
        state_hash = hash_object.hexdigest()
        
        return state_hash
    
    def decode_state(self, encoded_state: str) -> Optional[np.ndarray]:
        """Hash decoding ."""
        return None
    
    def get_similarity(self, state1: str, state2: str) -> float:
        """Hamming distance between hashes."""
        if state1 == state2:
            return 1.0
        
        # Simple similarity on basis shared prefixes
        common_prefix = 0
        for c1, c2 in zip(state1, state2):
            if c1 == c2:
                common_prefix += 1
            else:
                break
        
        similarity = common_prefix / max(len(state1), len(state2))
        return similarity
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalization state for consistent hashing."""
        if self.state_mean is None:
            # First initialization
            self.state_mean = state.copy()
            self.state_std = np.ones_like(state)
            self.normalization_samples = 1
        else:
            # Incremental update statistics
            self.normalization_samples += 1
            alpha = 1.0 / self.normalization_samples
            
            self.state_mean = (1 - alpha) * self.state_mean + alpha * state
            
            # Update variance (simplified)
            diff = state - self.state_mean
            self.state_std = np.sqrt(
                (1 - alpha) * self.state_std**2 + alpha * diff**2
            )
        
        # Normalization
        normalized = (state - self.state_mean) / (self.state_std + 1e-8)
        return np.clip(normalized, -5, 5)  # Clip for stability


class GridStateRepresentation(StateRepresentation):
    """
    Grid-based state representation for continuous spaces.
    
    Applies design pattern "Spatial Discretization" for
    structured state space representation.
    """
    
    def __init__(self, config: StateVisitorConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        self.resolution = config.grid_resolution
        
        # State bounds for grid construction
        self.state_mins = np.full(state_dim, np.inf)
        self.state_maxs = np.full(state_dim, -np.inf)
        self.bounds_initialized = False
        
        logger.info(f"Grid state representation initialized: {state_dim}D, resolution {self.resolution}")
    
    def encode_state(self, state: np.ndarray) -> tuple:
        """Encode state in grid coordinates."""
        # Update bounds
        self.state_mins = np.minimum(self.state_mins, state)
        self.state_maxs = np.maximum(self.state_maxs, state)
        self.bounds_initialized = True
        
        # Grid coordinates
        grid_coords = []
        for i, value in enumerate(state):
            if self.state_maxs[i] > self.state_mins[i]:
                # Normalize to [0, 1]
                normalized = (value - self.state_mins[i]) / (self.state_maxs[i] - self.state_mins[i])
                # Map to grid
                grid_coord = int(np.clip(normalized * self.resolution, 0, self.resolution - 1))
            else:
                grid_coord = 0
            
            grid_coords.append(grid_coord)
        
        return tuple(grid_coords)
    
    def decode_state(self, encoded_state: tuple) -> Optional[np.ndarray]:
        """Decode grid coordinates back in approximate state."""
        if not self.bounds_initialized:
            return None
        
        decoded_state = np.zeros(self.state_dim)
        for i, grid_coord in enumerate(encoded_state):
            if self.state_maxs[i] > self.state_mins[i]:
                # Map from grid back to original space
                normalized = (grid_coord + 0.5) / self.resolution  # Center of grid cell
                decoded_state[i] = self.state_mins[i] + normalized * (self.state_maxs[i] - self.state_mins[i])
            else:
                decoded_state[i] = self.state_mins[i]
        
        return decoded_state
    
    def get_similarity(self, state1: tuple, state2: tuple) -> float:
        """Euclidean distance in grid space."""
        if len(state1) != len(state2):
            return 0.0
        
        distance = np.sqrt(sum((c1 - c2)**2 for c1, c2 in zip(state1, state2)))
        max_distance = np.sqrt(len(state1) * (self.resolution - 1)**2)
        
        similarity = 1.0 - (distance / max_distance)
        return max(0.0, similarity)


class StateVisitor:
    """
    Comprehensive state visitation tracking system.
    
    Uses design pattern "Behavioral Analytics" for
    detailed analysis exploration patterns in crypto trading.
    """
    
    def __init__(self, config: StateVisitorConfig, state_dim: int):
        self.config = config
        self.state_dim = state_dim
        
        # State representation strategy
        if config.state_representation == "hash":
            self.state_encoder = HashStateRepresentation(config)
        elif config.state_representation == "grid":
            self.state_encoder = GridStateRepresentation(config, state_dim)
        else:
            raise ValueError(f"Unknown state representation: {config.state_representation}")
        
        # Visitation tracking
        self.state_visit_counts = defaultdict(int)
        self.state_first_visit_times = {}
        self.state_last_visit_times = {}
        self.total_visits = 0
        
        # Temporal patterns
        self.visit_history = deque(maxlen=config.max_history_length)
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.return_times = defaultdict(list)
        
        # Crypto-specific tracking
        self.market_regime_visits = defaultdict(lambda: defaultdict(int))
        self.portfolio_state_visits = defaultdict(lambda: defaultdict(float))
        self.risk_level_visits = defaultdict(lambda: defaultdict(float))
        
        # Coverage analysis
        self.unique_states_visited = set()
        self.coverage_over_time = []
        self.exploration_efficiency = []
        
        # Performance tracking
        self.visit_timestamps = deque(maxlen=10000)
        self.processing_times = deque(maxlen=1000)
        
        logger.info(f"State visitor initialized: {state_dim}D state space")
    
    def visit_state(
        self,
        state: np.ndarray,
        market_regime: Optional[str] = None,
        portfolio_value: Optional[float] = None,
        risk_level: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
          state.
        
        Args:
            state: State vector
            market_regime: Current market regime
            portfolio_value: Portfolio value
            risk_level: Risk level
            timestamp: Visit timestamp
            
        Returns:
            Visit statistics
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = time.time()
        
        # Encode state
        encoded_state = self.state_encoder.encode_state(state)
        
        # Update visit counts
        old_count = self.state_visit_counts[encoded_state]
        self.state_visit_counts[encoded_state] += 1
        self.total_visits += 1
        
        # Track timing
        if encoded_state not in self.state_first_visit_times:
            self.state_first_visit_times[encoded_state] = timestamp
        
        last_visit_time = self.state_last_visit_times.get(encoded_state, timestamp)
        self.state_last_visit_times[encoded_state] = timestamp
        
        # Return time tracking
        if old_count > 0:
            return_time = timestamp - last_visit_time
            self.return_times[encoded_state].append(return_time)
        
        # Update unique states
        self.unique_states_visited.add(encoded_state)
        
        # Add to visit history
        visit_record = {
            'encoded_state': encoded_state,
            'timestamp': timestamp,
            'market_regime': market_regime,
            'portfolio_value': portfolio_value,
            'risk_level': risk_level,
            'visit_count': self.state_visit_counts[encoded_state]
        }
        self.visit_history.append(visit_record)
        
        # Transition tracking
        if len(self.visit_history) >= 2:
            prev_state = self.visit_history[-2]['encoded_state']
            self.transition_counts[prev_state][encoded_state] += 1
        
        # Crypto-specific tracking
        if market_regime is not None:
            self.market_regime_visits[market_regime][encoded_state] += 1
        
        if portfolio_value is not None:
            self.portfolio_state_visits[encoded_state]['total_value'] = (
                self.portfolio_state_visits[encoded_state].get('total_value', 0.0) + portfolio_value
            )
            self.portfolio_state_visits[encoded_state]['visit_count'] = (
                self.portfolio_state_visits[encoded_state].get('visit_count', 0) + 1
            )
        
        if risk_level is not None:
            self.risk_level_visits[encoded_state]['total_risk'] = (
                self.risk_level_visits[encoded_state].get('total_risk', 0.0) + risk_level
            )
            self.risk_level_visits[encoded_state]['risk_count'] = (
                self.risk_level_visits[encoded_state].get('risk_count', 0) + 1
            )
        
        # Coverage tracking
        if self.config.coverage_analysis and self.total_visits % 100 == 0:
            coverage = len(self.unique_states_visited)
            self.coverage_over_time.append((self.total_visits, coverage))
            
            # Exploration efficiency
            if len(self.coverage_over_time) >= 2:
                prev_visits, prev_coverage = self.coverage_over_time[-2]
                efficiency = (coverage - prev_coverage) / (self.total_visits - prev_visits)
                self.exploration_efficiency.append(efficiency)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.visit_timestamps.append(timestamp)
        
        # Visit statistics
        visit_stats = {
            'encoded_state': str(encoded_state),
            'old_visit_count': old_count,
            'new_visit_count': self.state_visit_counts[encoded_state],
            'total_visits': self.total_visits,
            'unique_states': len(self.unique_states_visited),
            'is_first_visit': old_count == 0,
            'return_time': timestamp - last_visit_time if old_count > 0 else 0.0,
            'processing_time': processing_time
        }
        
        return visit_stats
    
    def get_state_statistics(self, encoded_state: Union[str, tuple]) -> Dict[str, Any]:
        """Get statistics for specific state."""
        if encoded_state not in self.state_visit_counts:
            return {}
        
        stats = {
            'visit_count': self.state_visit_counts[encoded_state],
            'first_visit_time': self.state_first_visit_times.get(encoded_state),
            'last_visit_time': self.state_last_visit_times.get(encoded_state),
            'return_times': self.return_times.get(encoded_state, [])
        }
        
        # Return time statistics
        if encoded_state in self.return_times and self.return_times[encoded_state]:
            return_times = self.return_times[encoded_state]
            stats['return_time_stats'] = {
                'mean': np.mean(return_times),
                'std': np.std(return_times),
                'min': min(return_times),
                'max': max(return_times),
                'count': len(return_times)
            }
        
        # Transition statistics
        if encoded_state in self.transition_counts:
            transitions = self.transition_counts[encoded_state]
            stats['outgoing_transitions'] = dict(transitions)
            stats['num_outgoing_transitions'] = len(transitions)
        
        # Incoming transitions
        incoming_count = 0
        for from_state, transitions in self.transition_counts.items():
            if encoded_state in transitions:
                incoming_count += transitions[encoded_state]
        stats['total_incoming_transitions'] = incoming_count
        
        # Portfolio and risk statistics
        if encoded_state in self.portfolio_state_visits:
            portfolio_data = self.portfolio_state_visits[encoded_state]
            if portfolio_data.get('visit_count', 0) > 0:
                stats['avg_portfolio_value'] = (
                    portfolio_data['total_value'] / portfolio_data['visit_count']
                )
        
        if encoded_state in self.risk_level_visits:
            risk_data = self.risk_level_visits[encoded_state]
            if risk_data.get('risk_count', 0) > 0:
                stats['avg_risk_level'] = (
                    risk_data['total_risk'] / risk_data['risk_count']
                )
        
        return stats
    
    def get_exploration_coverage_analysis(self) -> Dict[str, Any]:
        """Comprehensive analysis exploration coverage."""
        analysis = {
            'total_visits': self.total_visits,
            'unique_states': len(self.unique_states_visited),
            'coverage_ratio': len(self.unique_states_visited) / max(1, self.total_visits),
            'state_space_dimension': self.state_dim
        }
        
        # Visit distribution analysis
        if self.state_visit_counts:
            visit_counts = list(self.state_visit_counts.values())
            analysis['visit_distribution'] = {
                'mean_visits_per_state': np.mean(visit_counts),
                'std_visits_per_state': np.std(visit_counts),
                'min_visits': min(visit_counts),
                'max_visits': max(visit_counts),
                'median_visits': np.median(visit_counts)
            }
            
            # Frequency analysis
            visit_frequency = Counter(visit_counts)
            analysis['visit_frequency_distribution'] = dict(visit_frequency)
            
            # Exploration uniformity (Gini coefficient)
            sorted_counts = sorted(visit_counts)
            n = len(sorted_counts)
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
            analysis['exploration_uniformity_gini'] = gini
        
        # Coverage over time
        if self.coverage_over_time:
            visits, coverages = zip(*self.coverage_over_time)
            analysis['coverage_growth'] = {
                'visits': list(visits),
                'coverages': list(coverages),
                'final_coverage_rate': coverages[-1] / visits[-1] if visits[-1] > 0 else 0.0
            }
        
        # Exploration efficiency
        if self.exploration_efficiency:
            analysis['exploration_efficiency'] = {
                'mean': np.mean(self.exploration_efficiency),
                'std': np.std(self.exploration_efficiency),
                'trend': np.polyfit(range(len(self.exploration_efficiency)), self.exploration_efficiency, 1)[0]
            }
        
        # Transition analysis
        if self.transition_counts:
            total_transitions = sum(
                sum(transitions.values()) for transitions in self.transition_counts.values()
            )
            unique_transitions = sum(
                len(transitions) for transitions in self.transition_counts.values()
            )
            
            analysis['transition_statistics'] = {
                'total_transitions': total_transitions,
                'unique_transitions': unique_transitions,
                'avg_transitions_per_state': total_transitions / len(self.transition_counts),
                'transition_diversity': unique_transitions / len(self.transition_counts)
            }
        
        # Market regime analysis
        if self.config.market_regime_separation and self.market_regime_visits:
            regime_analysis = {}
            for regime, state_visits in self.market_regime_visits.items():
                regime_analysis[regime] = {
                    'unique_states': len(state_visits),
                    'total_visits': sum(state_visits.values()),
                    'avg_visits_per_state': np.mean(list(state_visits.values()))
                }
            analysis['market_regime_analysis'] = regime_analysis
        
        return analysis
    
    def get_most_visited_states(self, top_k: int = 10) -> List[Tuple[str, int, Dict[str, Any]]]:
        """Get top-k most visited states."""
        sorted_states = sorted(
            self.state_visit_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_states = []
        for encoded_state, count in sorted_states[:top_k]:
            state_stats = self.get_state_statistics(encoded_state)
            top_states.append((str(encoded_state), count, state_stats))
        
        return top_states
    
    def get_least_visited_states(self, top_k: int = 10) -> List[Tuple[str, int, Dict[str, Any]]]:
        """Get top-k visited states."""
        sorted_states = sorted(
            self.state_visit_counts.items(),
            key=lambda x: x[1]
        )
        
        least_states = []
        for encoded_state, count in sorted_states[:top_k]:
            state_stats = self.get_state_statistics(encoded_state)
            least_states.append((str(encoded_state), count, state_stats))
        
        return least_states
    
    def get_transition_matrix(self, top_states: Optional[int] = None) -> Dict[str, Any]:
        """Build transition matrix between states."""
        if not self.transition_counts:
            return {}
        
        # Select states for matrix
        if top_states is not None:
            # Top visited states
            sorted_states = sorted(
                self.state_visit_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            selected_states = [state for state, _ in sorted_states[:top_states]]
        else:
            selected_states = list(self.state_visit_counts.keys())
        
        # Build matrix
        state_to_idx = {state: i for i, state in enumerate(selected_states)}
        matrix_size = len(selected_states)
        transition_matrix = np.zeros((matrix_size, matrix_size))
        
        for from_state, transitions in self.transition_counts.items():
            if from_state in state_to_idx:
                from_idx = state_to_idx[from_state]
                total_transitions = sum(transitions.values())
                
                for to_state, count in transitions.items():
                    if to_state in state_to_idx:
                        to_idx = state_to_idx[to_state]
                        transition_matrix[from_idx, to_idx] = count / total_transitions
        
        return {
            'matrix': transition_matrix,
            'states': selected_states,
            'state_to_index': state_to_idx
        }
    
    def visualize_exploration_patterns(self, save_path: Optional[str] = None) -> None:
        """Visualization exploration patterns."""
        if not self.config.enable_visualization:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Coverage over time
            if self.coverage_over_time:
                visits, coverages = zip(*self.coverage_over_time)
                axes[0, 0].plot(visits, coverages, 'b-', linewidth=2)
                axes[0, 0].set_xlabel('Total Visits')
                axes[0, 0].set_ylabel('Unique States Discovered')
                axes[0, 0].set_title('Exploration Coverage Over Time')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Visit frequency distribution
            if self.state_visit_counts:
                visit_counts = list(self.state_visit_counts.values())
                axes[0, 1].hist(visit_counts, bins=50, edgecolor='black', alpha=0.7)
                axes[0, 1].set_xlabel('Visit Count per State')
                axes[0, 1].set_ylabel('Number of States')
                axes[0, 1].set_title('State Visit Frequency Distribution')
                axes[0, 1].set_yscale('log')
            
            # 3. Exploration efficiency
            if self.exploration_efficiency:
                axes[1, 0].plot(self.exploration_efficiency, 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Time Window')
                axes[1, 0].set_ylabel('New States / Visits')
                axes[1, 0].set_title('Exploration Efficiency Over Time')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Market regime distribution (if available)
            if self.market_regime_visits:
                regime_counts = {
                    regime: sum(states.values())
                    for regime, states in self.market_regime_visits.items()
                }
                
                regimes = list(regime_counts.keys())
                counts = list(regime_counts.values())
                
                axes[1, 1].bar(regimes, counts, color=['blue', 'red', 'green', 'orange'][:len(regimes)])
                axes[1, 1].set_xlabel('Market Regime')
                axes[1, 1].set_ylabel('Total Visits')
                axes[1, 1].set_title('Visits by Market Regime')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Exploration visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.warning(f"Error creating visualization: {e}")
    
    def save_state_visitor_data(self, filepath: str) -> None:
        """Save state visitor data."""
        data = {
            'config': self.config,
            'state_visit_counts': dict(self.state_visit_counts),
            'state_first_visit_times': self.state_first_visit_times,
            'state_last_visit_times': self.state_last_visit_times,
            'total_visits': self.total_visits,
            'unique_states_visited': list(self.unique_states_visited),
            'transition_counts': {
                k: dict(v) for k, v in self.transition_counts.items()
            },
            'return_times': {
                k: list(v) for k, v in self.return_times.items()
            },
            'coverage_over_time': self.coverage_over_time,
            'exploration_efficiency': self.exploration_efficiency,
            'market_regime_visits': {
                k: dict(v) for k, v in self.market_regime_visits.items()
            },
            'portfolio_state_visits': {
                k: dict(v) for k, v in self.portfolio_state_visits.items()
            },
            'risk_level_visits': {
                k: dict(v) for k, v in self.risk_level_visits.items()
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"State visitor data saved to {filepath}")
    
    def load_state_visitor_data(self, filepath: str) -> None:
        """Load state visitor data."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.state_visit_counts = defaultdict(int, data['state_visit_counts'])
        self.state_first_visit_times = data['state_first_visit_times']
        self.state_last_visit_times = data['state_last_visit_times']
        self.total_visits = data['total_visits']
        self.unique_states_visited = set(data['unique_states_visited'])
        
        self.transition_counts = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data['transition_counts'].items()}
        )
        self.return_times = defaultdict(
            list,
            {k: deque(v, maxlen=1000) for k, v in data['return_times'].items()}
        )
        
        self.coverage_over_time = data['coverage_over_time']
        self.exploration_efficiency = data['exploration_efficiency']
        
        self.market_regime_visits = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data['market_regime_visits'].items()}
        )
        
        self.portfolio_state_visits = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data['portfolio_state_visits'].items()}
        )
        
        self.risk_level_visits = defaultdict(
            lambda: defaultdict(float),
            {k: defaultdict(float, v) for k, v in data['risk_level_visits'].items()}
        )
        
        logger.info(f"State visitor data loaded from {filepath}")


def create_state_visitor_system(
    config: StateVisitorConfig,
    state_dim: int
) -> StateVisitor:
    """
    Factory function for creation state visitor system.
    
    Args:
        config: State visitor configuration
        state_dim: State space dimensionality
        
    Returns:
        Configured state visitor
    """
    visitor = StateVisitor(config, state_dim)
    
    logger.info("State visitor system created successfully")
    logger.info(f"State representation: {config.state_representation}")
    logger.info(f"State dimension: {state_dim}")
    
    return visitor


if __name__ == "__main__":
    # Example use state visitor
    config = StateVisitorConfig(
        state_representation="hash",
        track_temporal_patterns=True,
        market_regime_separation=True,
        coverage_analysis=True
    )
    
    state_dim = 128
    visitor = create_state_visitor_system(config, state_dim)
    
    # Simulation exploration
    market_regimes = ['bull', 'bear', 'sideways']
    
    for episode in range(5):
        for step in range(200):
            # Random state
            state = np.random.randn(state_dim)
            
            # Context
            market_regime = market_regimes[episode % 3]
            portfolio_value = np.random.uniform(1000, 10000)
            risk_level = np.random.uniform(0.1, 0.9)
            
            # Visit state
            visit_stats = visitor.visit_state(
                state=state,
                market_regime=market_regime,
                portfolio_value=portfolio_value,
                risk_level=risk_level
            )
            
            if step % 50 == 0:
                print(f"Episode {episode}, Step {step}: "
                      f"Unique states: {visit_stats['unique_states']}, "
                      f"Total visits: {visit_stats['total_visits']}")
    
    # Analysis
    coverage_analysis = visitor.get_exploration_coverage_analysis()
    print("\nCoverage Analysis:")
    for key, value in coverage_analysis.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")
    
    # Top visited states
    top_states = visitor.get_most_visited_states(5)
    print(f"\nTop 5 visited states:")
    for i, (state, count, stats) in enumerate(top_states):
        print(f"{i+1}. State: {state[:20]}... Visits: {count}")
    
    # Visualization
    visitor.visualize_exploration_patterns()