"""
Task Sampler System
Efficient Task Sampling for Meta-Learning

System sampling tasks with optimization performance, caching
and intelligent preliminary loading for cryptocurrency trading.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Iterator, Callable
from dataclasses import dataclass
import logging
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
import pickle
import hashlib
from collections import deque, defaultdict
import asyncio
from pathlib import Path

from .task_distribution import BaseTaskDistribution, TaskConfig, TaskMetadata


@dataclass
class SamplerConfig:
    """Configuration for Task Sampler"""
    
    # Performance
    batch_size: int = 32  # Size batch for sampling
    prefetch_factor: int = 2  # How many batch preliminarily load
    num_workers: int = 4  # Number worker threads
    
    # Caching
    enable_cache: bool = True  # Enable caching tasks
    cache_size: int = 1000  # Maximum size cache
    cache_dir: Optional[str] = None  # Directory for persistent cache
    
    # Filtering tasks
    min_difficulty: Optional[float] = None
    max_difficulty: Optional[float] = None
    allowed_task_types: Optional[List[str]] = None
    required_domains: Optional[List[str]] = None
    
    # Balancing
    balance_by_difficulty: bool = True  # Balance by complexity
    balance_by_domain: bool = True  # Balance by domains
    difficulty_bins: int = 5  # Number bins for complexity
    
    # Quality tasks
    min_quality_score: float = 0.5  # Minimum score quality
    exclude_duplicate_tasks: bool = True  # Exclude duplicates
    
    # Async support
    async_mode: bool = False  # Asynchronous sampling
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TaskCache:
    """
    Cache tasks with LRU eviction and persistent storage
    
    High-Performance Caching
    - Memory-efficient storage
    - Persistent caching
    - Thread-safe operations
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.max_size = max_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.logger = logger or logging.getLogger(__name__)
        
        # In-memory cache (LRU)
        self.memory_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.access_order = deque()  # For LRU
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Persistent cache setup
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
    
    def _generate_task_key(self, task_config: Dict[str, Any]) -> str:
        """Generates unique key for tasks"""
        # Serialize configuration and take hash
        config_str = str(sorted(task_config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get(self, task_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Gets task from cache"""
        with self.lock:
            # Check memory cache
            if task_key in self.memory_cache:
                # Update LRU order
                self.access_order.remove(task_key)
                self.access_order.append(task_key)
                self.cache_stats['hits'] += 1
                return self.memory_cache[task_key]
            
            # Check persistent cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"{task_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            task_data = pickle.load(f)
                        
                        # Add in memory cache
                        self._add_to_memory_cache(task_key, task_data)
                        self.cache_stats['hits'] += 1
                        return task_data
                    except Exception as e:
                        self.logger.warning(f"Failed to load cached task {task_key}: {e}")
            
            self.cache_stats['misses'] += 1
            return None
    
    def put(self, task_key: str, task_data: Dict[str, torch.Tensor]) -> None:
        """Adds task in cache"""
        with self.lock:
            # Add in memory cache
            self._add_to_memory_cache(task_key, task_data)
            
            # Save in persistent cache
            if self.cache_dir:
                cache_file = self.cache_dir / f"{task_key}.pkl"
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(task_data, f)
                except Exception as e:
                    self.logger.warning(f"Failed to save task to cache {task_key}: {e}")
    
    def _add_to_memory_cache(self, task_key: str, task_data: Dict[str, torch.Tensor]) -> None:
        """Adds task in memory cache with LRU eviction"""
        # Remove old version if exists
        if task_key in self.memory_cache:
            self.access_order.remove(task_key)
        
        # Add new
        self.memory_cache[task_key] = task_data
        self.access_order.append(task_key)
        
        # LRU eviction if exceeded size
        while len(self.memory_cache) > self.max_size:
            oldest_key = self.access_order.popleft()
            del self.memory_cache[oldest_key]
            self.cache_stats['evictions'] += 1
    
    def _load_persistent_cache(self) -> None:
        """Loads existing persistent cache"""
        if not self.cache_dir.exists():
            return
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        self.logger.info(f"Found {len(cache_files)} cached tasks")
        
        # Load most new files in memory cache
        cache_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        loaded = 0
        for cache_file in cache_files[:self.max_size // 2]:  # Load half cache
            task_key = cache_file.stem
            try:
                with open(cache_file, 'rb') as f:
                    task_data = pickle.load(f)
                self._add_to_memory_cache(task_key, task_data)
                loaded += 1
            except Exception as e:
                self.logger.warning(f"Failed to load cached task {task_key}: {e}")
        
        self.logger.info(f"Loaded {loaded} tasks into memory cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics cache"""
        with self.lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'memory_cache_size': len(self.memory_cache),
                'hit_rate': hit_rate,
                'total_hits': self.cache_stats['hits'],
                'total_misses': self.cache_stats['misses'],
                'total_evictions': self.cache_stats['evictions']
            }


class TaskFilter:
    """
    Filter tasks by various criteria
    
    Configurable Task Filtering
    - Multi-criteria filtering
    - Performance optimization
    - Extensible filter system
    """
    
    def __init__(self, config: SamplerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics filtering
        self.filter_stats = defaultdict(int)
    
    def should_include_task(
        self,
        task_data: Dict[str, torch.Tensor],
        metadata: Optional[TaskMetadata] = None,
        difficulty: Optional[float] = None
    ) -> bool:
        """Checks, must whether task be enabled"""
        
        # Filter by complexity
        if difficulty is not None:
            if self.config.min_difficulty is not None and difficulty < self.config.min_difficulty:
                self.filter_stats['difficulty_too_low'] += 1
                return False
            
            if self.config.max_difficulty is not None and difficulty > self.config.max_difficulty:
                self.filter_stats['difficulty_too_high'] += 1
                return False
        
        # Filter by type tasks
        if metadata and self.config.allowed_task_types:
            if metadata.task_type not in self.config.allowed_task_types:
                self.filter_stats['wrong_task_type'] += 1
                return False
        
        # Filter by domain
        if metadata and self.config.required_domains:
            if metadata.source_domain not in self.config.required_domains:
                self.filter_stats['wrong_domain'] += 1
                return False
        
        # Filter by quality
        if metadata and metadata.data_quality_score < self.config.min_quality_score:
            self.filter_stats['low_quality'] += 1
            return False
        
        self.filter_stats['accepted'] += 1
        return True
    
    def get_filter_stats(self) -> Dict[str, int]:
        """Returns statistics filtering"""
        return dict(self.filter_stats)


class TaskBalancer:
    """
    Balancer tasks for uniform distribution
    
    Intelligent Task Balancing
    - Multi-dimensional balancing
    - Adaptive rebalancing
    - Performance monitoring
    """
    
    def __init__(self, config: SamplerConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics for balancing
        self.difficulty_distribution = defaultdict(int)
        self.domain_distribution = defaultdict(int)
        self.task_type_distribution = defaultdict(int)
    
    def balance_task_batch(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]
    ) -> List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]:
        """Balances batch tasks"""
        
        if not self.config.balance_by_difficulty and not self.config.balance_by_domain:
            return tasks
        
        balanced_tasks = []
        
        if self.config.balance_by_difficulty:
            tasks = self._balance_by_difficulty(tasks)
        
        if self.config.balance_by_domain:
            tasks = self._balance_by_domain(tasks)
        
        return tasks
    
    def _balance_by_difficulty(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]
    ) -> List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]:
        """Balances tasks by complexity"""
        
        if not tasks:
            return tasks
        
        # Group tasks by bins complexity
        difficulty_bins = {}
        min_difficulty = min(difficulty for _, _, difficulty in tasks)
        max_difficulty = max(difficulty for _, _, difficulty in tasks)
        
        bin_width = (max_difficulty - min_difficulty) / self.config.difficulty_bins
        
        for task_data, metadata, difficulty in tasks:
            if bin_width > 0:
                bin_idx = min(
                    int((difficulty - min_difficulty) / bin_width),
                    self.config.difficulty_bins - 1
                )
            else:
                bin_idx = 0
            
            if bin_idx not in difficulty_bins:
                difficulty_bins[bin_idx] = []
            difficulty_bins[bin_idx].append((task_data, metadata, difficulty))
        
        # Balance number tasks in in each bin
        target_per_bin = len(tasks) // len(difficulty_bins)
        balanced_tasks = []
        
        for bin_idx, bin_tasks in difficulty_bins.items():
            if len(bin_tasks) <= target_per_bin:
                balanced_tasks.extend(bin_tasks)
            else:
                # Randomly select target_per_bin tasks
                selected = np.random.choice(
                    len(bin_tasks), target_per_bin, replace=False
                )
                balanced_tasks.extend([bin_tasks[i] for i in selected])
        
        return balanced_tasks
    
    def _balance_by_domain(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]
    ) -> List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]:
        """Balances tasks by domains"""
        
        # Group by domains
        domain_tasks = defaultdict(list)
        for task_data, metadata, difficulty in tasks:
            domain = metadata.source_domain if metadata else "unknown"
            domain_tasks[domain].append((task_data, metadata, difficulty))
        
        # Balance
        min_domain_size = min(len(domain_list) for domain_list in domain_tasks.values())
        balanced_tasks = []
        
        for domain, domain_task_list in domain_tasks.items():
            if len(domain_task_list) <= min_domain_size:
                balanced_tasks.extend(domain_task_list)
            else:
                # Randomly select min_domain_size tasks
                selected = np.random.choice(
                    len(domain_task_list), min_domain_size, replace=False
                )
                balanced_tasks.extend([domain_task_list[i] for i in selected])
        
        return balanced_tasks
    
    def update_distributions(
        self,
        tasks: List[Tuple[Dict[str, torch.Tensor], TaskMetadata, float]]
    ) -> None:
        """Updates statistics distributions"""
        for task_data, metadata, difficulty in tasks:
            # Difficulty distribution
            difficulty_bin = int(difficulty * 10)  # 0.0-0.1 -> 0, 0.1-0.2 -> 1, etc.
            self.difficulty_distribution[difficulty_bin] += 1
            
            # Domain distribution
            if metadata:
                self.domain_distribution[metadata.source_domain] += 1
                self.task_type_distribution[metadata.task_type] += 1


class TaskSampler:
    """
    Main Task Sampler with performance and intelligence
    
    High-Performance Task Sampling
    - Async/sync operation modes
    - Intelligent prefetching
    - Multi-threaded processing
    - Advanced caching and filtering
    """
    
    def __init__(
        self,
        task_distribution: BaseTaskDistribution,
        config: SamplerConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.task_distribution = task_distribution
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Components
        self.cache = TaskCache(
            max_size=config.cache_size,
            cache_dir=config.cache_dir,
            logger=logger
        ) if config.enable_cache else None
        
        self.filter = TaskFilter(config, logger)
        self.balancer = TaskBalancer(config, logger)
        
        # Threading setup
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        self.prefetch_queue = queue.Queue(maxsize=config.prefetch_factor * config.batch_size)
        self.prefetch_thread = None
        self.should_stop_prefetch = threading.Event()
        
        # Statistics
        self.sampling_stats = {
            'total_sampled': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'filtered_out': 0,
            'sampling_time_total': 0.0,
            'average_sampling_time': 0.0
        }
        
        # Run prefetching if needed
        if config.prefetch_factor > 0:
            self._start_prefetching()
        
        self.logger.info(f"TaskSampler initialized with config: {config}")
    
    def _start_prefetching(self) -> None:
        """Launches prefetching in separate flow"""
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True
        )
        self.prefetch_thread.start()
        self.logger.info("Started prefetching thread")
    
    def _prefetch_worker(self) -> None:
        """Worker for prefetching tasks"""
        while not self.should_stop_prefetch.is_set():
            try:
                if self.prefetch_queue.qsize() < self.config.prefetch_factor * self.config.batch_size:
                    # Generate batch tasks
                    future = self.executor.submit(self._generate_filtered_batch, self.config.batch_size)
                    
                    # Wait result with timeout
                    batch = future.result(timeout=30)
                    
                    # Add in queue
                    for task in batch:
                        if not self.should_stop_prefetch.is_set():
                            self.prefetch_queue.put(task, timeout=1)
                
                # Small pause
                time.sleep(0.1)
                
            except queue.Full:
                # Queue full, skip
                time.sleep(0.5)
            except Exception as e:
                self.logger.warning(f"Error in prefetch worker: {e}")
                time.sleep(1)
    
    def _generate_filtered_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Generates batch tasks with filtering"""
        filtered_tasks = []
        attempts = 0
        max_attempts = batch_size * 3  # Maximum attempts
        
        while len(filtered_tasks) < batch_size and attempts < max_attempts:
            # Generate task
            task_data = self.task_distribution.sample_task()
            
            # Retrieve metadata and complexity
            metadata = None
            difficulty = None
            
            try:
                difficulty = self.task_distribution.get_task_difficulty(task_data)
                # Attempt get metadata (if available)
                if hasattr(self.task_distribution, 'metadata_registry'):
                    # Search metadata by last task
                    registry_keys = list(self.task_distribution.metadata_registry.keys())
                    if registry_keys:
                        last_key = registry_keys[-1]
                        metadata = self.task_distribution.metadata_registry[last_key]
            except Exception as e:
                self.logger.debug(f"Could not get task metadata/difficulty: {e}")
            
            # Filter
            if self.filter.should_include_task(task_data, metadata, difficulty):
                filtered_tasks.append(task_data)
            
            attempts += 1
        
        if len(filtered_tasks) < batch_size:
            self.logger.warning(f"Only generated {len(filtered_tasks)} tasks out of {batch_size} requested")
        
        return filtered_tasks
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples one task"""
        start_time = time.time()
        
        # Try get from prefetch queue
        if self.config.prefetch_factor > 0:
            try:
                task = self.prefetch_queue.get(timeout=1)
                self.sampling_stats['total_sampled'] += 1
                
                sampling_time = time.time() - start_time
                self._update_timing_stats(sampling_time)
                
                return task
            except queue.Empty:
                self.logger.debug("Prefetch queue empty, generating task directly")
        
        # Generate directly
        batch = self._generate_filtered_batch(1)
        if batch:
            task = batch[0]
            self.sampling_stats['total_sampled'] += 1
            
            sampling_time = time.time() - start_time
            self._update_timing_stats(sampling_time)
            
            return task
        else:
            raise RuntimeError("Failed to generate valid task")
    
    def sample_batch(self, batch_size: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Samples batch tasks"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        start_time = time.time()
        batch = []
        
        # Try get from prefetch queue
        if self.config.prefetch_factor > 0:
            queue_tasks = []
            for _ in range(min(batch_size, self.prefetch_queue.qsize())):
                try:
                    task = self.prefetch_queue.get_nowait()
                    queue_tasks.append(task)
                except queue.Empty:
                    break
            batch.extend(queue_tasks)
        
        # Supplement if needed
        remaining = batch_size - len(batch)
        if remaining > 0:
            additional_tasks = self._generate_filtered_batch(remaining)
            batch.extend(additional_tasks)
        
        # Limit size batch
        batch = batch[:batch_size]
        
        self.sampling_stats['total_sampled'] += len(batch)
        
        sampling_time = time.time() - start_time
        self._update_timing_stats(sampling_time)
        
        return batch
    
    async def sample_task_async(self) -> Dict[str, torch.Tensor]:
        """Asynchronous sampling tasks"""
        if not self.config.async_mode:
            raise RuntimeError("Async mode not enabled in config")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.sample_task)
    
    async def sample_batch_async(self, batch_size: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        """Asynchronous sampling batch"""
        if not self.config.async_mode:
            raise RuntimeError("Async mode not enabled in config")
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.sample_batch, batch_size)
    
    def _update_timing_stats(self, sampling_time: float) -> None:
        """Updates statistics time sampling"""
        self.sampling_stats['sampling_time_total'] += sampling_time
        
        if self.sampling_stats['total_sampled'] > 0:
            self.sampling_stats['average_sampling_time'] = (
                self.sampling_stats['sampling_time_total'] / 
                self.sampling_stats['total_sampled']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns full statistics sampler"""
        stats = {
            'sampling_stats': self.sampling_stats.copy(),
            'filter_stats': self.filter.get_filter_stats(),
        }
        
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        if self.config.prefetch_factor > 0:
            stats['prefetch_queue_size'] = self.prefetch_queue.qsize()
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.shutdown()
    
    def shutdown(self) -> None:
        """Completes work sampler and clears resources"""
        self.logger.info("Shutting down TaskSampler")
        
        # Stop prefetching
        if self.prefetch_thread:
            self.should_stop_prefetch.set()
            self.prefetch_thread.join(timeout=5)
        
        # Complete executor
        self.executor.shutdown(wait=True)
        
        # Final statistics
        final_stats = self.get_statistics()
        self.logger.info(f"TaskSampler final statistics: {final_stats}")


class DataLoader:
    """
    DataLoader for meta-training with support various sampling strategies
    
    Flexible Data Loading
    - Multiple iteration strategies
    - Memory optimization
    - Performance monitoring
    """
    
    def __init__(
        self,
        task_sampler: TaskSampler,
        num_iterations: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.task_sampler = task_sampler
        self.num_iterations = num_iterations
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.current_iteration = 0
    
    def __iter__(self) -> Iterator[List[Dict[str, torch.Tensor]]]:
        """Iterator by batch' tasks"""
        self.current_iteration = 0
        
        while True:
            if self.num_iterations and self.current_iteration >= self.num_iterations:
                break
            
            try:
                batch = self.task_sampler.sample_batch()
                
                if self.shuffle:
                    np.random.shuffle(batch)
                
                if self.drop_last and len(batch) < self.task_sampler.config.batch_size:
                    break
                
                yield batch
                self.current_iteration += 1
                
            except Exception as e:
                self.task_sampler.logger.error(f"Error during iteration {self.current_iteration}: {e}")
                break
    
    def __len__(self) -> int:
        """Returns number iterations"""
        return self.num_iterations if self.num_iterations else float('inf')