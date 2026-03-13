"""
Curiosity Buffer for prioritized replay curiosity-driven experiences.

Implements advanced replay buffer with enterprise patterns
for efficient storage and sampling curiosity experiences.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from collections import deque
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CuriosityBufferConfig:
    """Configuration for curiosity buffer."""
    
    buffer_capacity: int = 100000
    prioritization_method: str = "curiosity"  # "curiosity", "td_error", "random"
    alpha: float = 0.6  # Prioritization strength
    beta: float = 0.4  # Importance sampling correction
    
    #  settings
    distributed_sampling: bool = True
    compression_enabled: bool = True


class CuriosityReplayBuffer:
    """
    Prioritized replay buffer for curiosity experiences.
    
    Uses design pattern "Priority Queue" for
    intelligent sampling exploration experiences.
    """
    
    def __init__(self, config: CuriosityBufferConfig):
        self.config = config
        self.capacity = config.buffer_capacity
        
        # Buffer storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.curiosity_rewards = []
        self.priorities = np.zeros(config.buffer_capacity)
        
        self.size = 0
        self.current_index = 0
        
        logger.info(f"Curiosity replay buffer initialized with capacity {self.capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        curiosity_reward: float
    ) -> None:
        """Add experience in buffer."""
        # Calculate priority based on curiosity reward
        priority = abs(curiosity_reward) + 1e-6
        
        if self.size < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.curiosity_rewards.append(curiosity_reward)
            
            self.priorities[self.size] = priority
            self.size += 1
        else:
            # Replace oldest experience
            idx = self.current_index
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = done
            self.curiosity_rewards[idx] = curiosity_reward
            
            self.priorities[idx] = priority
            self.current_index = (self.current_index + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch with prioritization."""
        if self.size == 0:
            return {}
        
        # Prioritized sampling
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.config.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.config.beta)
        weights /= weights.max()
        
        # Extract batch
        batch = {
            'states': torch.FloatTensor([self.states[i] for i in indices]),
            'actions': torch.FloatTensor([self.actions[i] for i in indices]),
            'rewards': torch.FloatTensor([self.rewards[i] for i in indices]),
            'next_states': torch.FloatTensor([self.next_states[i] for i in indices]),
            'dones': torch.BoolTensor([self.dones[i] for i in indices]),
            'curiosity_rewards': torch.FloatTensor([self.curiosity_rewards[i] for i in indices]),
            'weights': torch.FloatTensor(weights),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) + 1e-6


if __name__ == "__main__":
    config = CuriosityBufferConfig(buffer_capacity=1000)
    buffer = CuriosityReplayBuffer(config)
    
    # Test addition experiences
    for i in range(100):
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = random.choice([True, False])
        curiosity_reward = np.random.exponential(0.5)
        
        buffer.add(state, action, reward, next_state, done, curiosity_reward)
    
    # Test sampling
    batch = buffer.sample(32)
    print(f"Buffer size: {buffer.size}")
    print(f"Sampled batch keys: {list(batch.keys())}")
    if batch:
        print(f"Batch shapes: states={batch['states'].shape}, actions={batch['actions'].shape}")