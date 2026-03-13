"""
Episodic Memory System for curiosity-driven exploration.

Implements sophisticated memory system for storage and retrieval exploration experiences
with enterprise patterns for scalable memory management.
"""

import numpy as np
import torch
import faiss
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque, defaultdict
import time
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemoryConfig:
    """Configuration for episodic memory system."""
    
    # Memory capacity
    memory_capacity: int = 100000
    embedding_dim: int = 64
    
    # Retrieval parameters
    num_neighbors: int = 10
    similarity_threshold: float = 0.1
    
    # Memory management
    replacement_strategy: str = "fifo"  # "fifo", "lru", "importance"
    compression_enabled: bool = True
    
    #  enterprise settings
    distributed_memory: bool = True
    persistent_storage: bool = True
    real_time_retrieval: bool = True


class EpisodicMemorySystem:
    """
    Enterprise-grade episodic memory for curiosity exploration.
    
    Applies design pattern "Memory Architecture" for
    efficient storage and retrieval exploration experiences.
    """
    
    def __init__(self, config: EpisodicMemoryConfig):
        self.config = config
        self.capacity = config.memory_capacity
        self.embedding_dim = config.embedding_dim
        
        # FAISS index for similarity search
        self.index = faiss.IndexFlatL2(config.embedding_dim)
        
        # Memory storage
        self.embeddings = np.zeros((config.memory_capacity, config.embedding_dim), dtype=np.float32)
        self.rewards = np.zeros(config.memory_capacity, dtype=np.float32)
        self.timestamps = np.zeros(config.memory_capacity, dtype=np.float64)
        self.episode_ids = np.zeros(config.memory_capacity, dtype=np.int32)
        
        # Memory management
        self.size = 0
        self.current_index = 0
        self.access_counts = np.zeros(config.memory_capacity, dtype=np.int32)
        
        logger.info(f"Episodic memory initialized with capacity {self.capacity}")
    
    def add_experience(
        self,
        embedding: np.ndarray,
        reward: float,
        episode_id: int,
        timestamp: Optional[float] = None
    ) -> None:
        """Add experience in memory."""
        if timestamp is None:
            timestamp = time.time()
        
        if self.size < self.capacity:
            index = self.size
            self.size += 1
        else:
            # Memory full, use replacement strategy
            index = self._get_replacement_index()
        
        # Store experience
        self.embeddings[index] = embedding.astype(np.float32)
        self.rewards[index] = reward
        self.timestamps[index] = timestamp
        self.episode_ids[index] = episode_id
        self.access_counts[index] = 0
        
        # Update FAISS index
        if index < self.index.ntotal:
            # Need to rebuild index
            self._rebuild_index()
        else:
            self.index.add(embedding.reshape(1, -1).astype(np.float32))
    
    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieval similar experiences."""
        if k is None:
            k = min(self.config.num_neighbors, self.size)
        
        if self.size == 0:
            return np.array([]), np.array([])
        
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), k
        )
        
        # Update access counts
        for idx in indices[0]:
            if idx < self.size:
                self.access_counts[idx] += 1
        
        return distances[0], indices[0]
    
    def _get_replacement_index(self) -> int:
        """Get index for replacement."""
        if self.config.replacement_strategy == "fifo":
            index = self.current_index
            self.current_index = (self.current_index + 1) % self.capacity
            return index
        elif self.config.replacement_strategy == "lru":
            return np.argmin(self.access_counts[:self.size])
        else:
            return np.random.randint(0, self.size)
    
    def _rebuild_index(self) -> None:
        """Rebuild FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        if self.size > 0:
            self.index.add(self.embeddings[:self.size].astype(np.float32))
    
    def save(self, filepath: str) -> None:
        """Save memory state."""
        data = {
            'embeddings': self.embeddings[:self.size],
            'rewards': self.rewards[:self.size],
            'timestamps': self.timestamps[:self.size],
            'episode_ids': self.episode_ids[:self.size],
            'size': self.size,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Episodic memory saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load memory state."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.size = data['size']
        self.embeddings[:self.size] = data['embeddings']
        self.rewards[:self.size] = data['rewards']
        self.timestamps[:self.size] = data['timestamps']
        self.episode_ids[:self.size] = data['episode_ids']
        
        self._rebuild_index()
        
        logger.info(f"Episodic memory loaded from {filepath}")


if __name__ == "__main__":
    config = EpisodicMemoryConfig(memory_capacity=1000, embedding_dim=32)
    memory = EpisodicMemorySystem(config)
    
    # Test addition experiences
    for i in range(100):
        embedding = np.random.randn(32)
        reward = np.random.randn()
        memory.add_experience(embedding, reward, episode_id=i//10)
    
    # Test retrieval
    query = np.random.randn(32)
    distances, indices = memory.retrieve_similar(query, k=5)
    
    print(f"Memory size: {memory.size}")
    print(f"Retrieved {len(indices)} similar experiences")
    print(f"Distances: {distances}")