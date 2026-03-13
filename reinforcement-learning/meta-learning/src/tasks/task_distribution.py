"""
Task Distribution System
Scalable Task Management for Meta-Learning

System distribution and management tasks for meta-training in context
cryptocurrency trading. Supports various types tasks and distributions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import random
from collections import defaultdict
import math

from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class TaskConfig:
    """Configuration tasks for meta-training"""
    
    # Main parameters
    num_classes: int = 5  # Number classes (for classification)
    num_support: int = 5  # Examples on class in support set
    num_query: int = 15   # Examples on class in query set
    
    # Type tasks
    task_type: str = "classification"  # classification, regression, ranking
    
    # Complexity tasks
    difficulty_level: str = "medium"  # easy, medium, hard
    min_difficulty: float = 0.1
    max_difficulty: float = 1.0
    
    # Temporal parameters
    time_horizon: int = 100  # Horizon forecasting for temporal series
    sequence_length: int = 50  # Length input sequences
    
    # Market parameters
    market_conditions: List[str] = field(default_factory=lambda: ["bull", "bear", "sideways"])
    volatility_range: Tuple[float, float] = (0.01, 0.1)  # Range volatility
    
    # Balancing
    class_balance: str = "balanced"  # balanced, imbalanced, natural
    imbalance_ratio: float = 0.1  # For imbalanced classes
    
    # Quality data
    noise_level: float = 0.05  # Level noise in data
    missing_data_ratio: float = 0.0  # Share missing data


@dataclass
class TaskMetadata:
    """Metadata tasks"""
    
    task_id: str
    task_type: str
    difficulty: float
    source_domain: str  # crypto_pairs, market_regimes, strategies
    target_variable: str
    feature_names: List[str]
    data_quality_score: float
    created_timestamp: float
    
    # Crypto-specific
    trading_pair: Optional[str] = None
    exchange: Optional[str] = None
    timeframe: Optional[str] = None
    market_cap_category: Optional[str] = None  # large, mid, small, micro


class BaseTaskDistribution(ABC):
    """
    Base class for distribution tasks
    
    Abstract Task Distribution
    - Pluggable task generation
    - Consistent interface
    - Extensible architecture
    """
    
    def __init__(self, config: TaskConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.task_registry = {}
        self.metadata_registry = {}
    
    @abstractmethod
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples one task"""
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Samples batch tasks"""
        pass
    
    @abstractmethod
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Evaluates complexity tasks"""
        pass
    
    def register_task(self, task_id: str, task_data: Dict[str, torch.Tensor], metadata: TaskMetadata):
        """Registers task in registry"""
        self.task_registry[task_id] = task_data
        self.metadata_registry[task_id] = metadata
        self.logger.debug(f"Registered task {task_id}")
    
    def get_task_by_id(self, task_id: str) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Gets task by ID"""
        if task_id not in self.task_registry:
            raise ValueError(f"Task {task_id} not found in registry")
        return self.task_registry[task_id], self.metadata_registry[task_id]
    
    def get_tasks_by_criteria(self, **criteria) -> List[Tuple[str, Dict[str, torch.Tensor], TaskMetadata]]:
        """Gets tasks by criteria"""
        matching_tasks = []
        
        for task_id, metadata in self.metadata_registry.items():
            match = True
            for key, value in criteria.items():
                if hasattr(metadata, key):
                    if isinstance(value, (list, tuple)):
                        if getattr(metadata, key) not in value:
                            match = False
                            break
                    else:
                        if getattr(metadata, key) != value:
                            match = False
                            break
                else:
                    match = False
                    break
            
            if match:
                matching_tasks.append((task_id, self.task_registry[task_id], metadata))
        
        return matching_tasks


class CryptoTaskDistribution(BaseTaskDistribution):
    """
    Distribution tasks for cryptocurrency trading
    
    Domain-Specific Task Distribution
    - Crypto market simulation
    - Realistic market conditions
    - Multi-asset scenarios
    """
    
    def __init__(
        self,
        config: TaskConfig,
        crypto_data: Optional[Dict[str, np.ndarray]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.crypto_data = crypto_data or self._generate_synthetic_data()
        self.available_pairs = list(self.crypto_data.keys())
        self.task_counter = 0
        
        # Statistics by tasks
        self.task_stats = defaultdict(int)
        
        self.logger.info(f"CryptoTaskDistribution initialized with {len(self.available_pairs)} trading pairs")
    
    def _generate_synthetic_data(self) -> Dict[str, np.ndarray]:
        """Generates synthetic data cryptocurrencies"""
        synthetic_data = {}
        
        # Main trading pairs
        pairs = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"
        ]
        
        for pair in pairs:
            # Generate OHLCV data
            length = 10000
            base_price = np.random.uniform(0.1, 50000)
            
            # Geometric Brownian movement with trend
            returns = np.random.normal(0.0001, 0.02, length)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # OHLCV
            open_prices = prices
            high_prices = prices * (1 + np.abs(np.random.normal(0, 0.01, length)))
            low_prices = prices * (1 - np.abs(np.random.normal(0, 0.01, length)))
            close_prices = prices
            volumes = np.random.lognormal(10, 1, length)
            
            # Technical indicators
            rsi = np.random.uniform(20, 80, length)
            macd = np.random.normal(0, 0.1, length)
            bb_upper = high_prices * 1.02
            bb_lower = low_prices * 0.98
            
            # Merge in one array
            data = np.column_stack([
                open_prices, high_prices, low_prices, close_prices, volumes,
                rsi, macd, bb_upper, bb_lower
            ])
            
            synthetic_data[pair] = data
        
        return synthetic_data
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples one cryptocurrency task"""
        task_id = f"crypto_task_{self.task_counter}"
        self.task_counter += 1
        
        # Select trading couple
        trading_pair = random.choice(self.available_pairs)
        data = self.crypto_data[trading_pair]
        
        # Define type tasks
        if self.config.task_type == "classification":
            task_data, metadata = self._create_classification_task(task_id, trading_pair, data)
        elif self.config.task_type == "regression":
            task_data, metadata = self._create_regression_task(task_id, trading_pair, data)
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        # Register task
        self.register_task(task_id, task_data, metadata)
        self.task_stats[self.config.task_type] += 1
        
        return task_data
    
    def _create_classification_task(
        self,
        task_id: str,
        trading_pair: str,
        data: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Creates task classification directions price"""
        
        # Select random period
        start_idx = random.randint(self.config.sequence_length, len(data) - self.config.time_horizon - 1000)
        end_idx = start_idx + 1000
        
        period_data = data[start_idx:end_idx]
        
        # Create features (sliding windows)
        features = []
        labels = []
        
        for i in range(self.config.sequence_length, len(period_data) - self.config.time_horizon):
            # Feature window
            feature_window = period_data[i-self.config.sequence_length:i, :]  # [seq_len, n_features]
            
            # Normalization
            feature_window = (feature_window - feature_window.mean(axis=0)) / (feature_window.std(axis=0) + 1e-8)
            
            # Label: direction price through time_horizon steps
            current_price = period_data[i, 3]  # close price
            future_price = period_data[i + self.config.time_horizon, 3]
            
            # Classes: 0 - drop, 1 - growth, 2 - sideways
            price_change = (future_price - current_price) / current_price
            
            if price_change < -0.02:
                label = 0  # Drop
            elif price_change > 0.02:
                label = 1  # Growth
            else:
                label = 2  # Sideways
            
            features.append(feature_window.flatten())
            labels.append(label)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Filter by available classes
        unique_labels = np.unique(labels)
        if len(unique_labels) < self.config.num_classes:
            # If not enough classes, supplement random
            while len(unique_labels) < self.config.num_classes:
                fake_label = len(unique_labels)
                labels = np.append(labels, fake_label)
                features = np.vstack([features, features[-1]])
                unique_labels = np.unique(labels)
        
        # Select needed number classes
        selected_classes = np.random.choice(unique_labels, self.config.num_classes, replace=False)
        
        # Support and query sets
        support_data, support_labels, query_data, query_labels = self._split_support_query(
            features, labels, selected_classes
        )
        
        task_data = {
            'support_data': torch.FloatTensor(support_data),
            'support_labels': torch.LongTensor(support_labels),
            'query_data': torch.FloatTensor(query_data),
            'query_labels': torch.LongTensor(query_labels)
        }
        
        # Metadata
        metadata = TaskMetadata(
            task_id=task_id,
            task_type="classification",
            difficulty=self._compute_classification_difficulty(support_labels, query_labels),
            source_domain="crypto_pairs",
            target_variable="price_direction",
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            data_quality_score=0.8,
            created_timestamp=0.0,
            trading_pair=trading_pair,
            timeframe="1h"
        )
        
        return task_data, metadata
    
    def _create_regression_task(
        self,
        task_id: str,
        trading_pair: str,
        data: np.ndarray
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Creates task regression predictions price"""
        
        # Similarly classification, but with continuous targets
        start_idx = random.randint(self.config.sequence_length, len(data) - self.config.time_horizon - 1000)
        end_idx = start_idx + 1000
        
        period_data = data[start_idx:end_idx]
        
        features = []
        targets = []
        
        for i in range(self.config.sequence_length, len(period_data) - self.config.time_horizon):
            # Feature window
            feature_window = period_data[i-self.config.sequence_length:i, :]
            feature_window = (feature_window - feature_window.mean(axis=0)) / (feature_window.std(axis=0) + 1e-8)
            
            # Target: relative price change
            current_price = period_data[i, 3]
            future_price = period_data[i + self.config.time_horizon, 3]
            price_change = (future_price - current_price) / current_price
            
            features.append(feature_window.flatten())
            targets.append(price_change)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Randomly select examples for support and query
        n_total = len(features)
        n_support = self.config.num_support * self.config.num_classes  # Reuse parameter
        n_query = self.config.num_query * self.config.num_classes
        
        indices = np.random.permutation(n_total)
        support_indices = indices[:n_support]
        query_indices = indices[n_support:n_support + n_query]
        
        task_data = {
            'support_data': torch.FloatTensor(features[support_indices]),
            'support_labels': torch.FloatTensor(targets[support_indices]),
            'query_data': torch.FloatTensor(features[query_indices]),
            'query_labels': torch.FloatTensor(targets[query_indices])
        }
        
        metadata = TaskMetadata(
            task_id=task_id,
            task_type="regression",
            difficulty=np.std(targets),  # Complexity = volatility
            source_domain="crypto_pairs",
            target_variable="price_change",
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            data_quality_score=0.8,
            created_timestamp=0.0,
            trading_pair=trading_pair,
            timeframe="1h"
        )
        
        return task_data, metadata
    
    def _split_support_query(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        selected_classes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separates data on support and query sets"""
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            # Find examples of this class
            class_mask = labels == class_label
            class_features = features[class_mask]
            class_labels = np.full(len(class_features), class_idx)  # Renumber classes
            
            if len(class_features) < self.config.num_support + self.config.num_query:
                # Duplicate examples if not enough
                needed = self.config.num_support + self.config.num_query
                indices = np.random.choice(len(class_features), needed, replace=True)
                class_features = class_features[indices]
                class_labels = np.full(needed, class_idx)
            
            # Randomly select examples
            indices = np.random.permutation(len(class_features))
            support_indices = indices[:self.config.num_support]
            query_indices = indices[self.config.num_support:self.config.num_support + self.config.num_query]
            
            support_data.append(class_features[support_indices])
            support_labels.append(class_labels[support_indices])
            query_data.append(class_features[query_indices])
            query_labels.append(class_labels[query_indices])
        
        return (
            np.vstack(support_data),
            np.concatenate(support_labels),
            np.vstack(query_data),
            np.concatenate(query_labels)
        )
    
    def _compute_classification_difficulty(
        self,
        support_labels: np.ndarray,
        query_labels: np.ndarray
    ) -> float:
        """Computes complexity tasks classification"""
        
        # Factors complexity:
        # 1. Balancing classes
        unique, counts = np.unique(support_labels, return_counts=True)
        balance_ratio = np.min(counts) / np.max(counts)
        
        # 2. Number classes
        num_classes = len(unique)
        class_complexity = num_classes / 10.0  # Normalize
        
        # 3. Size support set
        support_size_factor = 1.0 / len(support_labels) * 100
        
        # Total complexity
        difficulty = (1.0 - balance_ratio) * 0.4 + class_complexity * 0.3 + support_size_factor * 0.3
        return np.clip(difficulty, 0.0, 1.0)
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Samples batch tasks"""
        return [self.sample_task() for _ in range(batch_size)]
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Evaluates complexity tasks on basis data"""
        support_labels = task_data['support_labels']
        query_labels = task_data['query_labels']
        
        if task_data['support_labels'].dtype == torch.long:
            # Classification
            return self._compute_classification_difficulty(
                support_labels.numpy(), query_labels.numpy()
            )
        else:
            # Regression
            return float(torch.std(support_labels).item())
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Returns statistics by tasks"""
        return {
            'total_tasks': sum(self.task_stats.values()),
            'task_types': dict(self.task_stats),
            'available_pairs': self.available_pairs,
            'registered_tasks': len(self.task_registry)
        }


class CurriculumTaskDistribution(BaseTaskDistribution):
    """
    Distribution tasks with curriculum learning
    
    Progressive Learning System
    - Adaptive difficulty progression
    - Performance-based task selection
    - Multi-objective optimization
    """
    
    def __init__(
        self,
        base_distribution: BaseTaskDistribution,
        config: TaskConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.base_distribution = base_distribution
        self.current_difficulty = config.min_difficulty
        self.performance_history = []
        self.difficulty_schedule = self._create_difficulty_schedule()
        
        # Parameters curriculum
        self.difficulty_increase_threshold = 0.8  # Accuracy threshold for increase complexity
        self.difficulty_decrease_threshold = 0.5  # Accuracy threshold for decrease complexity
        self.difficulty_step = 0.1
        self.patience = 5  # Number epochs for changes complexity
        
        self.logger.info(f"CurriculumTaskDistribution initialized with difficulty range: "
                        f"{config.min_difficulty}-{config.max_difficulty}")
    
    def _create_difficulty_schedule(self) -> List[float]:
        """Creates schedule increase complexity"""
        num_steps = 20
        min_diff = self.config.min_difficulty
        max_diff = self.config.max_difficulty
        
        # Exponential increase complexity
        schedule = []
        for i in range(num_steps):
            progress = i / (num_steps - 1)
            difficulty = min_diff + (max_diff - min_diff) * (progress ** 2)
            schedule.append(difficulty)
        
        return schedule
    
    def update_performance(self, performance_metrics: Dict[str, float]) -> None:
        """Updates history performance and adapts complexity"""
        self.performance_history.append(performance_metrics)
        
        # Evaluate recent results
        if len(self.performance_history) >= self.patience:
            recent_performance = self.performance_history[-self.patience:]
            avg_accuracy = np.mean([p.get('accuracy', 0) for p in recent_performance])
            
            # Adapt complexity
            if avg_accuracy > self.difficulty_increase_threshold:
                # Increase complexity
                new_difficulty = min(
                    self.current_difficulty + self.difficulty_step,
                    self.config.max_difficulty
                )
                if new_difficulty > self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    self.logger.info(f"Increased difficulty to {self.current_difficulty:.2f}")
            
            elif avg_accuracy < self.difficulty_decrease_threshold:
                # Reduce complexity
                new_difficulty = max(
                    self.current_difficulty - self.difficulty_step,
                    self.config.min_difficulty
                )
                if new_difficulty < self.current_difficulty:
                    self.current_difficulty = new_difficulty
                    self.logger.info(f"Decreased difficulty to {self.current_difficulty:.2f}")
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples task with considering current complexity"""
        # Generate several candidates and select suitable by complexity
        candidates = []
        difficulties = []
        
        for _ in range(10):  # Generate 10 candidates
            task = self.base_distribution.sample_task()
            difficulty = self.base_distribution.get_task_difficulty(task)
            candidates.append(task)
            difficulties.append(difficulty)
        
        # Select task with nearest complexity
        target_difficulty = self.current_difficulty
        best_idx = np.argmin([abs(d - target_difficulty) for d in difficulties])
        
        selected_task = candidates[best_idx]
        selected_difficulty = difficulties[best_idx]
        
        self.logger.debug(f"Selected task with difficulty {selected_difficulty:.3f} "
                         f"(target: {target_difficulty:.3f})")
        
        return selected_task
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Samples batch tasks with considering curriculum"""
        return [self.sample_task() for _ in range(batch_size)]
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Delegates estimation complexity base distribution"""
        return self.base_distribution.get_task_difficulty(task_data)
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """Returns status curriculum learning"""
        return {
            'current_difficulty': self.current_difficulty,
            'min_difficulty': self.config.min_difficulty,
            'max_difficulty': self.config.max_difficulty,
            'performance_history_length': len(self.performance_history),
            'recent_avg_performance': (
                np.mean([p.get('accuracy', 0) for p in self.performance_history[-5:]])
                if len(self.performance_history) >= 5 else 0.0
            )
        }


class MultiDomainTaskDistribution(BaseTaskDistribution):
    """
    Distribution tasks from several domains
    
    Multi-Domain Meta-Learning
    - Cross-domain transfer
    - Domain adaptation
    - Balanced domain sampling
    """
    
    def __init__(
        self,
        domain_distributions: Dict[str, BaseTaskDistribution],
        config: TaskConfig,
        domain_weights: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.domain_distributions = domain_distributions
        self.domain_names = list(domain_distributions.keys())
        
        # Weights domains for sampling
        if domain_weights is None:
            self.domain_weights = {name: 1.0 for name in self.domain_names}
        else:
            self.domain_weights = domain_weights
        
        # Normalize weights
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            name: weight / total_weight
            for name, weight in self.domain_weights.items()
        }
        
        # Statistics
        self.domain_stats = defaultdict(int)
        
        self.logger.info(f"MultiDomainTaskDistribution initialized with domains: {self.domain_names}")
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples task from random domain"""
        # Select domain according to weights
        domain_name = np.random.choice(
            self.domain_names,
            p=list(self.domain_weights.values())
        )
        
        # Sample task from selected domain
        task = self.domain_distributions[domain_name].sample_task()
        
        # Add information about domain
        task['domain'] = domain_name
        
        self.domain_stats[domain_name] += 1
        return task
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Samples batch with balancing by domains"""
        batch = []
        
        # Distribute tasks by domains
        domain_counts = {}
        remaining = batch_size
        
        for domain_name in self.domain_names[:-1]:  # All except last
            count = int(batch_size * self.domain_weights[domain_name])
            domain_counts[domain_name] = count
            remaining -= count
        
        # Remainder give last domain
        domain_counts[self.domain_names[-1]] = remaining
        
        # Sample tasks
        for domain_name, count in domain_counts.items():
            if count > 0:
                domain_tasks = self.domain_distributions[domain_name].sample_batch(count)
                for task in domain_tasks:
                    task['domain'] = domain_name
                batch.extend(domain_tasks)
        
        # Shuffle batch
        random.shuffle(batch)
        return batch
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Evaluates complexity tasks through corresponding domain"""
        domain_name = task_data.get('domain', self.domain_names[0])
        return self.domain_distributions[domain_name].get_task_difficulty(task_data)
    
    def update_domain_weights(self, performance_by_domain: Dict[str, float]) -> None:
        """Updates weights domains on basis performance"""
        # Increase weights domains with low performance
        for domain_name in self.domain_names:
            performance = performance_by_domain.get(domain_name, 0.5)
            # Reverse dependency: than worse performance, the more weight
            self.domain_weights[domain_name] = 1.0 / (performance + 0.1)
        
        # Normalize weights
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            name: weight / total_weight
            for name, weight in self.domain_weights.items()
        }
        
        self.logger.info(f"Updated domain weights: {self.domain_weights}")
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Returns statistics by domains"""
        return {
            'domain_weights': self.domain_weights,
            'domain_stats': dict(self.domain_stats),
            'total_tasks': sum(self.domain_stats.values())
        }