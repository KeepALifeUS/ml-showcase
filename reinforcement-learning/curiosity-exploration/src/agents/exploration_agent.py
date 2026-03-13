"""
Exploration Agent for systematic exploration crypto trading environments.

Implements specialized agent for pure exploration with enterprise patterns
for comprehensive strategy space exploration.

Production-ready with:
- Async/await support for non-blocking operations
- Graceful shutdown handling
- Auto device detection (CPU/CUDA)
- Comprehensive error handling
- Structured logging
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from collections import deque, defaultdict
import asyncio
import signal
import sys
from pathlib import Path

# Structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExplorationAgentConfig:
    """Configuration for exploration agent."""
    
    state_dim: int = 256
    action_dim: int = 10
    
    # Exploration strategies
    exploration_strategies: List[str] = None
    strategy_weights: Dict[str, float] = None
    
    # Exploration parameters
    epsilon: float = 0.9
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.1
    
    #  enterprise settings
    distributed_exploration: bool = True
    real_time_adaptation: bool = True
    
    def __post_init__(self):
        if self.exploration_strategies is None:
            self.exploration_strategies = ["random", "curiosity", "count_based"]
        if self.strategy_weights is None:
            self.strategy_weights = {
                "random": 0.3,
                "curiosity": 0.4,
                "count_based": 0.3
            }


class ExplorationAgent:
    """
    Specialized agent for systematic exploration.
    
    Applies design pattern "Exploration Strategy" for
    comprehensive coverage trading space strategies.
    """
    
    def __init__(self, config: ExplorationAgentConfig, device: Optional[str] = None):
        self.config = config

        # Auto-detect device with fallback
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Validate device availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = 'cpu'

        # Exploration tracking
        self.exploration_history = deque(maxlen=10000)
        self.strategy_performance = defaultdict(lambda: deque(maxlen=1000))
        self.action_space_coverage = defaultdict(int)

        # Current exploration state
        self.current_epsilon = config.epsilon
        self.exploration_step = 0

        # Shutdown flag for graceful termination
        self._shutdown_flag = False

        # Setup signal handlers
        self._setup_signal_handlers()

        logger.info(f"Exploration agent initialized on device: {self.device}")

    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self._shutdown_flag = True

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def select_exploration_action(
        self,
        state: np.ndarray,
        exploration_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Selection action for pure exploration (async version).

        Returns:
            action: Selected action
            exploration_info: Metadata about exploration

        Raises:
            RuntimeError: If agent is shutting down
        """
        if self._shutdown_flag:
            raise RuntimeError("Agent is shutting down, cannot select action")

        try:
            if exploration_context is None:
                exploration_context = {}

            # Choose exploration strategy
            strategy = self._select_exploration_strategy()

            # Generate action based on strategy (CPU-bound, might need yield)
            await asyncio.sleep(0)  # Yield to event loop

            if strategy == "random":
                action = self._random_action()
            elif strategy == "curiosity":
                action = self._curiosity_driven_action(state, exploration_context)
            elif strategy == "count_based":
                action = self._count_based_action(state)
            else:
                action = self._random_action()

            # Track exploration
            exploration_info = {
                'strategy': strategy,
                'epsilon': self.current_epsilon,
                'exploration_step': self.exploration_step,
                'action_entropy': self._calculate_action_entropy(action)
            }

            self.exploration_history.append({
                'state': state.copy(),
                'action': action.copy(),
                'strategy': strategy,
                'step': self.exploration_step
            })

            # Update action space coverage
            action_key = tuple(np.round(action, 2))
            self.action_space_coverage[action_key] += 1

            self.exploration_step += 1

            # Decay epsilon
            self.current_epsilon = max(
                self.config.min_epsilon,
                self.current_epsilon * self.config.epsilon_decay
            )

            return action, exploration_info

        except Exception as e:
            logger.error(f"Error selecting exploration action: {e}", exc_info=True)
            # Return safe default action
            return self._random_action(), {'strategy': 'fallback', 'error': str(e)}
    
    def _select_exploration_strategy(self) -> str:
        """Selection exploration strategy based on performance."""
        strategies = list(self.config.strategy_weights.keys())
        weights = list(self.config.strategy_weights.values())
        
        # Adaptive weighting based on performance
        if len(self.strategy_performance) > 0:
            adjusted_weights = []
            for strategy in strategies:
                if strategy in self.strategy_performance:
                    performance = list(self.strategy_performance[strategy])
                    if performance:
                        # Higher weight for better performing strategies
                        avg_performance = np.mean(performance)
                        weight = weights[strategies.index(strategy)] * (1 + avg_performance)
                    else:
                        weight = weights[strategies.index(strategy)]
                else:
                    weight = weights[strategies.index(strategy)]
                adjusted_weights.append(weight)
            
            # Normalize weights
            total_weight = sum(adjusted_weights)
            if total_weight > 0:
                weights = [w / total_weight for w in adjusted_weights]
        
        return np.random.choice(strategies, p=weights)
    
    def _random_action(self) -> np.ndarray:
        """Random action for exploration."""
        return np.random.uniform(-1, 1, self.config.action_dim)
    
    def _curiosity_driven_action(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Curiosity-driven action selection."""
        # Simplified curiosity-based exploration
        # In implementations was to curiosity system
        
        # Add noise to previous successful actions
        if len(self.exploration_history) > 0:
            recent_actions = [exp['action'] for exp in list(self.exploration_history)[-10:]]
            if recent_actions:
                base_action = np.mean(recent_actions, axis=0)
                noise = np.random.normal(0, 0.3, self.config.action_dim)
                action = base_action + noise
                return np.clip(action, -1, 1)
        
        return self._random_action()
    
    def _count_based_action(self, state: np.ndarray) -> np.ndarray:
        """Count-based exploration action."""
        # Select action with lowest visit count
        min_count = float('inf')
        best_action = None
        
        # Sample several random actions and choose least visited
        for _ in range(10):
            candidate_action = np.random.uniform(-1, 1, self.config.action_dim)
            action_key = tuple(np.round(candidate_action, 2))
            count = self.action_space_coverage.get(action_key, 0)
            
            if count < min_count:
                min_count = count
                best_action = candidate_action
        
        return best_action if best_action is not None else self._random_action()
    
    def _calculate_action_entropy(self, action: np.ndarray) -> float:
        """Computation entropy action for diversity measurement."""
        # Discretize action for entropy calculation
        discretized = np.round(action * 10) / 10
        unique_values = len(np.unique(discretized))
        max_unique = len(discretized)
        
        return unique_values / max_unique if max_unique > 0 else 0.0
    
    def update_exploration_performance(
        self,
        strategy: str,
        performance: float
    ) -> None:
        """Update performance metrics for exploration strategy."""
        self.strategy_performance[strategy].append(performance)
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive exploration statistics."""
        stats = {
            'exploration_step': self.exploration_step,
            'current_epsilon': self.current_epsilon,
            'total_unique_actions': len(self.action_space_coverage),
            'exploration_history_size': len(self.exploration_history)
        }
        
        # Action space coverage analysis
        if self.action_space_coverage:
            counts = list(self.action_space_coverage.values())
            stats['action_coverage'] = {
                'mean_visits': np.mean(counts),
                'std_visits': np.std(counts),
                'max_visits': max(counts),
                'min_visits': min(counts)
            }
        
        # Strategy performance
        strategy_stats = {}
        for strategy, performance_history in self.strategy_performance.items():
            if performance_history:
                performance_list = list(performance_history)
                strategy_stats[strategy] = {
                    'mean_performance': np.mean(performance_list),
                    'std_performance': np.std(performance_list),
                    'sample_count': len(performance_list)
                }
        stats['strategy_performance'] = strategy_stats
        
        # Recent exploration diversity
        if len(self.exploration_history) >= 100:
            recent_actions = [exp['action'] for exp in list(self.exploration_history)[-100:]]
            action_matrix = np.array(recent_actions)
            
            # Calculate diversity metrics
            action_std = np.std(action_matrix, axis=0)
            stats['recent_diversity'] = {
                'mean_action_std': np.mean(action_std),
                'action_range': np.max(action_matrix) - np.min(action_matrix)
            }
        
        return stats
    
    def reset_exploration(self) -> None:
        """Reset exploration state for new episode."""
        self.current_epsilon = self.config.epsilon
        # Not completely history for learning
    
    async def save_exploration_data(self, filepath: str) -> None:
        """
        Save exploration data (async version).

        Args:
            filepath: Path to save file

        Raises:
            IOError: If save fails
        """
        import pickle

        try:
            data = {
                'exploration_history': list(self.exploration_history),
                'strategy_performance': {
                    k: list(v) for k, v in self.strategy_performance.items()
                },
                'action_space_coverage': dict(self.action_space_coverage),
                'exploration_step': self.exploration_step,
                'current_epsilon': self.current_epsilon
            }

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Non-blocking file write
            await asyncio.to_thread(self._save_pickle, filepath, data)

            logger.info(f"Exploration data saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save exploration data: {e}", exc_info=True)
            raise IOError(f"Could not save to {filepath}") from e

    def _save_pickle(self, filepath: str, data: Dict[str, Any]) -> None:
        """Synchronous pickle save helper"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    async def load_exploration_data(self, filepath: str) -> None:
        """
        Load exploration data (async version).

        Args:
            filepath: Path to load file

        Raises:
            IOError: If load fails
        """
        import pickle

        try:
            # Non-blocking file read
            data = await asyncio.to_thread(self._load_pickle, filepath)

            self.exploration_history = deque(data['exploration_history'], maxlen=10000)
            self.strategy_performance = defaultdict(
                lambda: deque(maxlen=1000),
                {k: deque(v, maxlen=1000) for k, v in data['strategy_performance'].items()}
            )
            self.action_space_coverage = defaultdict(int, data['action_space_coverage'])
            self.exploration_step = data['exploration_step']
            self.current_epsilon = data['current_epsilon']

            logger.info(f"Exploration data loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load exploration data: {e}", exc_info=True)
            raise IOError(f"Could not load from {filepath}") from e

    def _load_pickle(self, filepath: str) -> Dict[str, Any]:
        """Synchronous pickle load helper"""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint for monitoring.

        Returns:
            Health status dictionary
        """
        try:
            return {
                'status': 'healthy' if not self._shutdown_flag else 'shutting_down',
                'device': self.device,
                'exploration_step': self.exploration_step,
                'current_epsilon': self.current_epsilon,
                'history_size': len(self.exploration_history),
                'unique_actions': len(self.action_space_coverage)
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                'status': 'unhealthy',
                'error': str(e)
            }

    async def shutdown(self) -> None:
        """
        Graceful shutdown - save state and cleanup.
        """
        logger.info("Starting graceful shutdown...")
        self._shutdown_flag = True

        try:
            # Auto-save exploration data
            save_path = Path("./data/exploration_checkpoint.pkl")
            await self.save_exploration_data(str(save_path))
            logger.info("Exploration data auto-saved on shutdown")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

        logger.info("Graceful shutdown completed")


async def main():
    """Production-ready async main with error handling"""
    config = ExplorationAgentConfig(
        state_dim=128,
        action_dim=5,
        exploration_strategies=["random", "curiosity", "count_based"]
    )

    agent = ExplorationAgent(config)

    try:
        # Simulation exploration
        for episode in range(5):
            if agent._shutdown_flag:
                logger.info("Shutdown requested, stopping exploration")
                break

            agent.reset_exploration()

            for step in range(200):
                if agent._shutdown_flag:
                    break

                state = np.random.randn(128)

                # Select exploration action (async)
                action, exploration_info = await agent.select_exploration_action(state)

                # Simulate performance
                performance = np.random.randn() * 0.1
                agent.update_exploration_performance(
                    exploration_info['strategy'], performance
                )

                if step % 50 == 0:
                    print(f"Episode {episode}, Step {step}: "
                          f"Strategy: {exploration_info['strategy']}, "
                          f"Epsilon: {exploration_info['epsilon']:.3f}")

                # Periodic health check
                if step % 100 == 0:
                    health = await agent.health_check()
                    logger.info(f"Health: {health}")

        # Final statistics
        stats = agent.get_exploration_statistics()
        print("\nExploration Statistics:")
        for key, value in stats.items():
            if not isinstance(value, dict):
                print(f"{key}: {value}")
            else:
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Graceful shutdown
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())