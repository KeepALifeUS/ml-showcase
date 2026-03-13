"""
Memory Buffer System for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system management for prevention
catastrophic forgetting with integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
import random
from collections import defaultdict, deque
import heapq
import logging
from datetime import datetime, timedelta
import json
import pickle


class SamplingStrategy(Enum):
    """Strategies set from memory"""
    RANDOM = "random"
    RESERVOIR = "reservoir"
    K_CENTER = "k_center"
    GRADIENT_BASED = "gradient_based"
    HERDING = "herding"
    RING_BUFFER = "ring_buffer"
    IMPORTANCE_WEIGHTED = "importance_weighted"


class SelectionCriteria(Enum):
    """Criteria selection samples for memory"""
    UNCERTAINTY = "uncertainty"
    DIVERSITY = "diversity"
    REPRESENTATIVENESS = "representativeness"
    GRADIENT_MAGNITUDE = "gradient_magnitude"
    LOSS_MAGNITUDE = "loss_magnitude"
    TIME_WEIGHTED = "time_weighted"
    MARKET_REGIME = "market_regime"


@dataclass
class MemorySample:
    """Sample data in memory"""
    # Main data
    features: torch.Tensor
    target: torch.Tensor
    task_id: int
    
    # Metadata for crypto trading
    timestamp: datetime
    market_regime: str  # bull, bear, sideways, volatile
    asset: str  # BTC, ETH, etc.
    timeframe: str  # 1m, 5m, 1h, 1d
    
    # Metrics
    uncertainty_score: float = 0.0
    gradient_magnitude: float = 0.0
    loss_value: float = 0.0
    diversity_score: float = 0.0
    
    # Additional attributes
    prediction_confidence: float = 0.0
    market_volatility: float = 0.0
    trading_volume: float = 0.0
    
    #  enterprise metadata
    sample_id: str = field(default_factory=lambda: f"sample_{datetime.now().timestamp()}")
    quality_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def update_access(self) -> None:
        """Update information to """
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_staleness(self) -> float:
        """Calculation samples ( pattern)"""
        if self.last_accessed is None:
            return 0.0
        
        time_diff = datetime.now() - self.last_accessed
        return min(time_diff.total_seconds() / (24 * 3600), 1.0) # Normalization to
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert in dictionary for serialization"""
        return {
            "sample_id": self.sample_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "market_regime": self.market_regime,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "uncertainty_score": self.uncertainty_score,
            "gradient_magnitude": self.gradient_magnitude,
            "loss_value": self.loss_value,
            "diversity_score": self.diversity_score,
            "quality_score": self.quality_score,
            "access_count": self.access_count
        }


@dataclass
class BufferConfig:
    """Configuration buffer memory"""
    # Main parameters
    max_size: int = 1000
    sampling_strategy: SamplingStrategy = SamplingStrategy.RESERVOIR
    selection_criteria: List[SelectionCriteria] = field(default_factory=lambda: [
        SelectionCriteria.UNCERTAINTY,
        SelectionCriteria.DIVERSITY
    ])
    
    # enterprise parameters
    enable_quality_filtering: bool = True
    quality_threshold: float = 0.5
    enable_staleness_removal: bool = True
    max_staleness_days: int = 30
    
    # Crypto trading specific parameters
    market_regime_balance: bool = True # by market regimes
    asset_diversity: bool = True  # Diversity assets
    timeframe_coverage: bool = True # temporal
    
    # Performance settings
    batch_retrieval_size: int = 32
    enable_caching: bool = True
    cache_size: int = 100
    compression_enabled: bool = False
    
    # Monitor and logging
    enable_statistics: bool = True
    log_memory_usage: bool = True
    statistics_interval: int = 100


class BaseMemoryBuffer(ABC):
    """
    Base class for buffers memory continual training
    
    enterprise Features:
    - Adaptive memory management
    - Quality-aware sample selection
    - Performance monitoring
    - Automatic cleanup policies
    """
    
    def __init__(self, config: BufferConfig):
        self.config = config
        self.buffer: List[MemorySample] = []
        self.task_counters: Dict[int, int] = defaultdict(int)
        self.regime_counters: Dict[str, int] = defaultdict(int)
        
        # enterprise Components
        self.statistics = {
            "samples_added": 0,
            "samples_removed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "quality_filtered": 0,
            "staleness_removed": 0
        }
        
        # Caching for performance
        if self.config.enable_caching:
            self._sample_cache: Dict[str, MemorySample] = {}
            self._cache_order = deque(maxlen=self.config.cache_size)
        
        # Configure logging
        self.logger = logging.getLogger(f"MemoryBuffer-{self.__class__.__name__}")
        self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def add_samples(self, samples: List[MemorySample]) -> None:
        """Add samples in buffer memory"""
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int, exclude_task_ids: Optional[Set[int]] = None) -> List[MemorySample]:
        """Sample batch samples from buffer"""
        pass
    
    def get_samples_by_task(self, task_id: int) -> List[MemorySample]:
        """Get all samples for specific tasks"""
        return [sample for sample in self.buffer if sample.task_id == task_id]
    
    def get_samples_by_regime(self, market_regime: str) -> List[MemorySample]:
        """Get samples for specific market regime"""
        return [sample for sample in self.buffer if sample.market_regime == market_regime]
    
    def get_samples_by_asset(self, asset: str) -> List[MemorySample]:
        """Get samples for specific asset"""
        return [sample for sample in self.buffer if sample.asset == asset]
    
    def remove_stale_samples(self) -> int:
        """
        Remove obsolete samples ( pattern)
        
        Returns:
            Number removed samples
        """
        if not self.config.enable_staleness_removal:
            return 0
        
        initial_size = len(self.buffer)
        max_staleness = self.config.max_staleness_days
        
        # by
        fresh_samples = []
        for sample in self.buffer:
            staleness = sample.calculate_staleness()
            if staleness * 30 < max_staleness:  # Convert in days
                fresh_samples.append(sample)
        
        self.buffer = fresh_samples
        removed_count = initial_size - len(self.buffer)
        
        if removed_count > 0:
            self.statistics["staleness_removed"] += removed_count
            self.logger.info(f"Removed {removed_count} stale samples")
        
        return removed_count
    
    def filter_by_quality(self, samples: List[MemorySample]) -> List[MemorySample]:
        """
         samples by quality ( pattern)
        
        Args:
            samples: List samples for
            
        Returns:
             samples
        """
        if not self.config.enable_quality_filtering:
            return samples
        
        filtered = []
        for sample in samples:
            if sample.quality_score >= self.config.quality_threshold:
                filtered.append(sample)
            else:
                self.statistics["quality_filtered"] += 1
        
        return filtered
    
    def balance_market_regimes(self) -> None:
        """
        Balancing samples by market regimes
        
         pattern for adaptive training
        """
        if not self.config.market_regime_balance:
            return
        
        # Count samples by regimes
        regime_samples = defaultdict(list)
        for sample in self.buffer:
            regime_samples[sample.market_regime].append(sample)
        
        if len(regime_samples) <= 1:
            return
        
        # Finding minimum number samples
        min_samples = min(len(samples) for samples in regime_samples.values())
        target_per_regime = max(min_samples, self.config.max_size // len(regime_samples))
        
        # Balancing
        balanced_buffer = []
        for regime, samples in regime_samples.items():
            if len(samples) > target_per_regime:
                # Select best samples by
                samples.sort(key=lambda x: (
                    x.uncertainty_score + x.diversity_score + x.quality_score
                ), reverse=True)
                balanced_buffer.extend(samples[:target_per_regime])
            else:
                balanced_buffer.extend(samples)
        
        self.buffer = balanced_buffer
        self.logger.info(f"Balanced buffer across {len(regime_samples)} market regimes")
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics use memory
        
        Returns:
            Dictionary with statistics buffer
        """
        if not self.buffer:
            return {"status": "empty", "size": 0}
        
        # Main statistics
        stats = {
            "total_samples": len(self.buffer),
            "max_capacity": self.config.max_size,
            "utilization": len(self.buffer) / self.config.max_size,
            "task_distribution": dict(self.task_counters),
            "regime_distribution": dict(self.regime_counters)
        }
        
        # Statistics quality samples
        quality_scores = [sample.quality_score for sample in self.buffer]
        uncertainty_scores = [sample.uncertainty_score for sample in self.buffer]
        
        stats.update({
            "avg_quality": np.mean(quality_scores),
            "min_quality": np.min(quality_scores),
            "max_quality": np.max(quality_scores),
            "avg_uncertainty": np.mean(uncertainty_scores),
            "total_access_count": sum(sample.access_count for sample in self.buffer)
        })
        
        #  enterprise statistics
        stats.update(self.statistics)
        
        # Memory usage by
        asset_counts = defaultdict(int)
        for sample in self.buffer:
            asset_counts[sample.asset] += 1
        stats["asset_distribution"] = dict(asset_counts)
        
        return stats
    
    def save_buffer(self, filepath: str) -> bool:
        """
        Save buffer memory in file
        
        Args:
            filepath: Path to file for saving
            
        Returns:
            True if saving successfully
        """
        try:
            buffer_data = {
                "config": {
                    "max_size": self.config.max_size,
                    "sampling_strategy": self.config.sampling_strategy.value,
                    "selection_criteria": [criteria.value for criteria in self.config.selection_criteria]
                },
                "samples": [sample.to_dict() for sample in self.buffer],
                "statistics": self.statistics,
                "task_counters": dict(self.task_counters),
                "regime_counters": dict(self.regime_counters),
                "timestamp": datetime.now().isoformat()
            }
            
            if self.config.compression_enabled:
                import gzip
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(buffer_data, f, indent=2)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(buffer_data, f, indent=2)
            
            self.logger.info(f"Buffer saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving buffer: {e}")
            return False
    
    def load_buffer(self, filepath: str) -> bool:
        """
        Load buffer memory from file
        
        Args:
            filepath: Path to file for loading
            
        Returns:
            True if loading successful
        """
        try:
            if self.config.compression_enabled:
                import gzip
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    buffer_data = json.load(f)
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    buffer_data = json.load(f)
            
            # Restore statistics
            self.statistics = buffer_data.get("statistics", {})
            self.task_counters = defaultdict(int, buffer_data.get("task_counters", {}))
            self.regime_counters = defaultdict(int, buffer_data.get("regime_counters", {}))
            
            self.logger.info(f"Buffer loaded from {filepath}")
            self.logger.info(f"Loaded {len(buffer_data.get('samples', []))} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading buffer: {e}")
            return False
    
    def clear(self) -> None:
        """Cleanup buffer memory"""
        self.buffer.clear()
        self.task_counters.clear()
        self.regime_counters.clear()
        
        if self.config.enable_caching:
            self._sample_cache.clear()
            self._cache_order.clear()
        
        # Reset statistics
        self.statistics = {key: 0 for key in self.statistics}
        
        self.logger.info("Memory buffer cleared")
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"size={len(self.buffer)}/{self.config.max_size}, "
            f"strategy={self.config.sampling_strategy.value})"
        )


class ReservoirBuffer(BaseMemoryBuffer):
    """
    Reservoir Sampling Buffer for efficient management
    
     Implementation for set with
    """
    
    def __init__(self, config: BufferConfig):
        super().__init__(config)
        self.total_samples_seen = 0
    
    def add_samples(self, samples: List[MemorySample]) -> None:
        """Add samples with Reservoir Sampling"""
        # by quality
        filtered_samples = self.filter_by_quality(samples)
        
        for sample in filtered_samples:
            self.total_samples_seen += 1
            
            if len(self.buffer) < self.config.max_size:
                # Buffer not - add
                self.buffer.append(sample)
                self.task_counters[sample.task_id] += 1
                self.regime_counters[sample.market_regime] += 1
            else:
                # Reservoir sampling
                j = random.randint(1, self.total_samples_seen)
                if j <= self.config.max_size:
                    # Replacing random
                    old_sample = self.buffer[j - 1]
                    self.task_counters[old_sample.task_id] -= 1
                    self.regime_counters[old_sample.market_regime] -= 1
                    
                    self.buffer[j - 1] = sample
                    self.task_counters[sample.task_id] += 1
                    self.regime_counters[sample.market_regime] += 1
            
            self.statistics["samples_added"] += 1
        
        # cleanup obsolete samples
        if self.statistics["samples_added"] % 100 == 0:
            self.remove_stale_samples()
            self.balance_market_regimes()
    
    def sample_batch(self, batch_size: int, exclude_task_ids: Optional[Set[int]] = None) -> List[MemorySample]:
        """Random sample batch from buffer"""
        if not self.buffer:
            return []
        
        # by tasks
        available_samples = self.buffer
        if exclude_task_ids:
            available_samples = [
                sample for sample in self.buffer 
                if sample.task_id not in exclude_task_ids
            ]
        
        if not available_samples:
            return []
        
        # Random sample
        batch_size = min(batch_size, len(available_samples))
        selected_samples = random.sample(available_samples, batch_size)
        
        # Update statistics access
        for sample in selected_samples:
            sample.update_access()
        
        return selected_samples


class KCenterBuffer(BaseMemoryBuffer):
    """
    K-Center Coreset Buffer for maximum diversity samples
    
     Implementation for feature space
    """
    
    def __init__(self, config: BufferConfig):
        super().__init__(config)
        self.feature_cache: Dict[str, np.ndarray] = {}
    
    def add_samples(self, samples: List[MemorySample]) -> None:
        """Add samples with k-center selection"""
        filtered_samples = self.filter_by_quality(samples)
        
        for sample in filtered_samples:
            if len(self.buffer) < self.config.max_size:
                self.buffer.append(sample)
                self.task_counters[sample.task_id] += 1
                self.regime_counters[sample.market_regime] += 1
            else:
                # K-center replacement
                self._k_center_replacement(sample)
            
            self.statistics["samples_added"] += 1
    
    def _k_center_replacement(self, new_sample: MemorySample) -> None:
        """Replacement samples on basis k-center algorithm"""
        if not self.buffer:
            return
        
        # Convert features in numpy for
        new_features = new_sample.features.cpu().numpy().flatten()
        
        # Computation up to samples
        min_distance_to_existing = float('inf')
        for existing_sample in self.buffer:
            existing_features = existing_sample.features.cpu().numpy().flatten()
            distance = np.linalg.norm(new_features - existing_features)
            min_distance_to_existing = min(min_distance_to_existing, distance)
        
        # Search samples for
        replace_idx = -1
        min_contribution = float('inf')
        
        for i, candidate in enumerate(self.buffer):
            # Computation in coverage
            candidate_features = candidate.features.cpu().numpy().flatten()
            min_distance_without_candidate = float('inf')
            
            for j, other_sample in enumerate(self.buffer):
                if i == j:
                    continue
                other_features = other_sample.features.cpu().numpy().flatten()
                distance = np.linalg.norm(candidate_features - other_features)
                min_distance_without_candidate = min(min_distance_without_candidate, distance)
            
            if min_distance_without_candidate < min_contribution:
                min_contribution = min_distance_without_candidate
                replace_idx = i
        
        # Replacement if new sample coverage
        if min_distance_to_existing > min_contribution and replace_idx >= 0:
            old_sample = self.buffer[replace_idx]
            self.task_counters[old_sample.task_id] -= 1
            self.regime_counters[old_sample.market_regime] -= 1
            
            self.buffer[replace_idx] = new_sample
            self.task_counters[new_sample.task_id] += 1
            self.regime_counters[new_sample.market_regime] += 1
            
            self.statistics["samples_removed"] += 1
    
    def sample_batch(self, batch_size: int, exclude_task_ids: Optional[Set[int]] = None) -> List[MemorySample]:
        """Sample with consideration diversity"""
        if not self.buffer:
            return []
        
        available_samples = self.buffer
        if exclude_task_ids:
            available_samples = [
                sample for sample in self.buffer 
                if sample.task_id not in exclude_task_ids
            ]
        
        if not available_samples:
            return []
        
        batch_size = min(batch_size, len(available_samples))
        
        # Greedy selection for maximum diversity
        selected = []
        remaining = available_samples.copy()
        
        # First sample - random
        if remaining:
            first_sample = random.choice(remaining)
            selected.append(first_sample)
            remaining.remove(first_sample)
        
        # samples -
        while len(selected) < batch_size and remaining:
            best_candidate = None
            max_min_distance = 0
            
            for candidate in remaining:
                candidate_features = candidate.features.cpu().numpy().flatten()
                min_distance = float('inf')
                
                for selected_sample in selected:
                    selected_features = selected_sample.features.cpu().numpy().flatten()
                    distance = np.linalg.norm(candidate_features - selected_features)
                    min_distance = min(min_distance, distance)
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        # Update statistics access
        for sample in selected:
            sample.update_access()
        
        return selected


#  Production-Ready Factory
class MemoryBufferFactory:
    """
    Factory for creation buffers memory with enterprise patterns
    """
    
    @staticmethod
    def create_crypto_trading_buffer(
        max_size: int = 1000,
        strategy: SamplingStrategy = SamplingStrategy.RESERVOIR,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> BaseMemoryBuffer:
        """
        Create buffer memory for crypto trading
        
        Args:
            max_size: Maximum size buffer
            strategy: Strategy set
            config_overrides: Overrides configuration
            
        Returns:
            Configured buffer memory
        """
        # Base configuration for crypto trading
        config = BufferConfig(
            max_size=max_size,
            sampling_strategy=strategy,
            selection_criteria=[
                SelectionCriteria.UNCERTAINTY,
                SelectionCriteria.DIVERSITY,
                SelectionCriteria.MARKET_REGIME
            ],
            market_regime_balance=True,
            asset_diversity=True,
            timeframe_coverage=True,
            enable_quality_filtering=True,
            enable_staleness_removal=True,
            quality_threshold=0.6,
            max_staleness_days=7, # Crypto data
            enable_caching=True,
            enable_statistics=True
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create corresponding buffer
        if strategy == SamplingStrategy.RESERVOIR:
            return ReservoirBuffer(config)
        elif strategy == SamplingStrategy.K_CENTER:
            return KCenterBuffer(config)
        else:
            # Fallback to reservoir sampling
            return ReservoirBuffer(config)