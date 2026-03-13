"""
Distributed PPO Training Implementation
for scalable distributed RL

Features:
- Multi-process data collection
- Centralized policy updates
- Gradient aggregation
- Asynchronous rollouts
- Fault tolerance
- Production-ready distributed training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import logging
import time
import os
import queue
import threading
from collections import defaultdict, deque
import psutil
import pickle

from .ppo_trainer import PPOTrainer, PPOTrainerConfig
from ..core.ppo import PPOAlgorithm, PPOConfig
from ..core.ppo2 import PPO2Algorithm, PPO2Config
from ..networks.actor_critic import ActorCriticNetwork
from ..buffers.rollout_buffer import RolloutBuffer
from ..utils.normalization import ObservationNormalizer


@dataclass
class DistributedPPOConfig(PPOTrainerConfig):
 """Configuration for distributed PPO training"""
 
 # Distributed settings
 world_size: int = 4 # Number of processes
 backend: str = "nccl" # Communication backend
 master_addr: str = "localhost" # Master node address
 master_port: str = "12355" # Master node port
 
 # Worker configuration
 workers_per_node: int = 1 # Workers per GPU/node
 use_gpu: bool = True # Use GPU acceleration
 gpu_ids: List[int] = field(default_factory=lambda: [0])
 
 # Data collection
 async_rollouts: bool = True # Asynchronous rollout collection
 rollout_workers: int = 2 # Number of rollout workers
 
 # Gradient synchronization
 gradient_compression: bool = False # Compress gradients
 gradient_clipping: str = "global" # global or local clipping
 sync_frequency: int = 1 # Sync every N updates
 
 # Load balancing
 dynamic_batching: bool = True # Dynamic batch size adjustment
 worker_timeout: float = 300.0 # Worker timeout (seconds)
 
 # Fault tolerance
 checkpoint_frequency: int = 100 # Checkpoint frequency
 auto_restart_failed_workers: bool = True
 max_worker_failures: int = 3
 
 # Performance optimization
 pin_memory: bool = True
 non_blocking_transfer: bool = True
 prefetch_factor: int = 2
 
 # Monitoring
 monitor_system_resources: bool = True
 log_distributed_metrics: bool = True


class WorkerState:
 """State tracking for distributed workers"""
 
 def __init__(self, worker_id: int):
 self.worker_id = worker_id
 self.is_active = True
 self.last_update_time = time.time()
 self.total_timesteps = 0
 self.total_episodes = 0
 self.failure_count = 0
 self.current_fps = 0.0
 self.memory_usage = 0.0
 self.gpu_usage = 0.0


class RolloutWorker:
 """
 Worker process for collecting rollouts
 
 Runs environments and collects experience data
 asynchronously from policy updates
 """
 
 def __init__(
 self,
 worker_id: int,
 config: DistributedPPOConfig,
 actor_critic: nn.Module,
 environments: List[Any],
 result_queue: mp.Queue,
 param_queue: mp.Queue
 ):
 self.worker_id = worker_id
 self.config = config
 self.result_queue = result_queue
 self.param_queue = param_queue
 
 # Setup device
 if config.use_gpu and torch.cuda.is_available():
 self.device = torch.device(f"cuda:{config.gpu_ids[worker_id % len(config.gpu_ids)]}")
 else:
 self.device = torch.device("cpu")
 
 # Initialize network
 self.actor_critic = actor_critic.to(self.device)
 self.environments = environments
 
 # Initialize normalizers
 if config.normalize_observations:
 obs_space = environments[0].observation_space.shape
 self.obs_normalizer = ObservationNormalizer(obs_space)
 else:
 self.obs_normalizer = None
 
 # Statistics
 self.episodes_collected = 0
 self.timesteps_collected = 0
 self.rollout_times = deque(maxlen=100)
 
 self.logger = logging.getLogger(f"Worker-{worker_id}")
 
 def run(self):
 """Main worker loop"""
 
 self.logger.info(f"Starting rollout worker {self.worker_id}")
 
 try:
 while True:
 # Check for new parameters
 self._check_parameter_updates()
 
 # Collect rollout
 start_time = time.time()
 rollout_data = self._collect_rollout()
 rollout_time = time.time() - start_time
 
 self.rollout_times.append(rollout_time)
 
 # Send results
 result = {
 "worker_id": self.worker_id,
 "rollout_data": rollout_data,
 "worker_stats": self._get_worker_stats(),
 "rollout_time": rollout_time
 }
 
 self.result_queue.put(result)
 
 # Update statistics
 self.timesteps_collected += len(rollout_data["rewards"])
 
 except KeyboardInterrupt:
 self.logger.info(f"Worker {self.worker_id} interrupted")
 except Exception as e:
 self.logger.error(f"Worker {self.worker_id} error: {e}")
 # Send error signal
 self.result_queue.put({
 "worker_id": self.worker_id,
 "error": str(e),
 "timestamp": time.time()
 })
 
 def _check_parameter_updates(self):
 """Check for new parameters from main process"""
 
 try:
 while not self.param_queue.empty():
 param_update = self.param_queue.get_nowait()
 
 if "actor_critic_params" in param_update:
 # Update network parameters
 self.actor_critic.load_state_dict(param_update["actor_critic_params"])
 
 if "normalizer_params" in param_update and self.obs_normalizer:
 # Update normalizer parameters
 self.obs_normalizer.running_stats.set_state(param_update["normalizer_params"])
 
 except queue.Empty:
 pass
 
 def _collect_rollout(self) -> Dict[str, Any]:
 """Collect single rollout"""
 
 # Initialize storage
 observations = []
 actions = []
 rewards = []
 values = []
 log_probs = []
 dones = []
 
 # Reset environments
 obs = [env.reset() for env in self.environments]
 
 for step in range(self.config.rollout_steps):
 # Normalize observations
 if self.obs_normalizer is not None:
 normalized_obs = [
 self.obs_normalizer(torch.tensor(o, dtype=torch.float32, device=self.device))
 for o in obs
 ]
 else:
 normalized_obs = [
 torch.tensor(o, dtype=torch.float32, device=self.device) 
 for o in obs
 ]
 
 obs_batch = torch.stack(normalized_obs)
 
 # Get actions and values
 with torch.no_grad():
 action_dist, values_pred = self.actor_critic(obs_batch)
 actions_pred = action_dist.sample()
 log_probs_pred = action_dist.log_prob(actions_pred)
 
 # Step environments
 next_obs = []
 step_rewards = []
 step_dones = []
 
 for i, env in enumerate(self.environments):
 o, r, d, info = env.step(actions_pred[i].cpu().numpy())
 
 next_obs.append(o)
 step_rewards.append(r)
 step_dones.append(d)
 
 if d:
 self.episodes_collected += 1
 o = env.reset()
 next_obs[-1] = o
 
 # Store data
 observations.extend([o.cpu() for o in normalized_obs])
 actions.extend([a.cpu() for a in actions_pred])
 rewards.extend(step_rewards)
 values.extend(values_pred.cpu().numpy())
 log_probs.extend(log_probs_pred.cpu().numpy())
 dones.extend(step_dones)
 
 obs = next_obs
 
 return {
 "observations": torch.stack(observations),
 "actions": torch.stack(actions),
 "rewards": torch.tensor(rewards, dtype=torch.float32),
 "values": torch.tensor(values, dtype=torch.float32),
 "log_probs": torch.tensor(log_probs, dtype=torch.float32),
 "dones": torch.tensor(dones, dtype=torch.bool)
 }
 
 def _get_worker_stats(self) -> Dict[str, Any]:
 """Get worker statistics"""
 
 # System resources
 process = psutil.Process()
 memory_usage = process.memory_info().rss / 1024 / 1024 # MB
 
 gpu_usage = 0.0
 if self.config.use_gpu and torch.cuda.is_available():
 gpu_usage = torch.cuda.memory_allocated(self.device) / 1024 / 1024 # MB
 
 fps = len(self.environments) * self.config.rollout_steps / np.mean(self.rollout_times) if self.rollout_times else 0.0
 
 return {
 "episodes_collected": self.episodes_collected,
 "timesteps_collected": self.timesteps_collected,
 "memory_usage_mb": memory_usage,
 "gpu_usage_mb": gpu_usage,
 "fps": fps,
 "avg_rollout_time": np.mean(self.rollout_times) if self.rollout_times else 0.0
 }


class DistributedPPOTrainer:
 """
 Distributed PPO trainer with Features:
 - Multi-process rollout collection
 - Centralized policy updates
 - Asynchronous training
 - Fault tolerance
 - Performance monitoring
 """
 
 def __init__(
 self,
 config: DistributedPPOConfig,
 actor_critic: Optional[nn.Module] = None,
 environments: Optional[List[Any]] = None
 ):
 self.config = config
 self.logger = logging.getLogger(__name__)
 
 # Initialize distributed training
 self._setup_distributed()
 
 # Initialize base trainer
 self.base_trainer = PPOTrainer(
 config=config,
 environments=environments,
 actor_critic=actor_critic
 )
 
 # Distributed components
 self.worker_states: Dict[int, WorkerState] = {}
 self.rollout_workers: List[mp.Process] = []
 self.result_queues: List[mp.Queue] = []
 self.param_queues: List[mp.Queue] = []
 
 # Aggregation buffers
 self.rollout_buffer_pool = []
 self.gradient_buffer = None
 
 # Performance monitoring
 self.distributed_metrics = defaultdict(list)
 self.last_sync_time = time.time()
 
 self.logger.info(f"Distributed PPO initialized with {config.world_size} processes")
 
 def _setup_distributed(self):
 """Setup distributed training environment"""
 
 # Set environment variables
 os.environ["MASTER_ADDR"] = self.config.master_addr
 os.environ["MASTER_PORT"] = self.config.master_port
 
 # Initialize process group
 if not dist.is_initialized():
 dist.init_process_group(
 backend=self.config.backend,
 world_size=self.config.world_size,
 rank=0 # Master process
 )
 
 self.logger.info(f"Distributed training setup complete")
 self.logger.info(f"World size: {self.config.world_size}")
 self.logger.info(f"Backend: {self.config.backend}")
 
 def train(self) -> Dict[str, Any]:
 """
 Distributed training loop
 
 Returns:
 Training statistics
 """
 
 try:
 # Start worker processes
 self._start_workers()
 
 # Main training loop
 training_start_time = time.time()
 
 while self.base_trainer.current_timestep < self.config.total_timesteps:
 # Collect rollouts from workers
 rollout_start_time = time.time()
 rollout_data = self._collect_distributed_rollouts()
 rollout_time = time.time() - rollout_start_time
 
 # Aggregate rollout data
 aggregated_buffer = self._aggregate_rollouts(rollout_data)
 
 # Update policy
 update_start_time = time.time()
 update_metrics = self._distributed_policy_update(aggregated_buffer)
 update_time = time.time() - update_start_time
 
 # Synchronize parameters
 if self.base_trainer.current_update % self.config.sync_frequency == 0:
 self._synchronize_parameters()
 
 # Monitor system
 if self.config.monitor_system_resources:
 self._monitor_workers()
 
 # Log distributed metrics
 if self.config.log_distributed_metrics:
 self._log_distributed_progress(update_metrics, rollout_time, update_time)
 
 # Handle failed workers
 if self.config.auto_restart_failed_workers:
 self._handle_failed_workers()
 
 self.base_trainer.current_update += 1
 
 # Training completed
 total_time = time.time() - training_start_time
 final_metrics = self._get_distributed_metrics(total_time)
 
 return final_metrics
 
 finally:
 # Cleanup
 self._cleanup_workers()
 self._cleanup_distributed()
 
 def _start_workers(self):
 """Start rollout worker processes"""
 
 self.logger.info(f"Starting {self.config.rollout_workers} rollout workers")
 
 for worker_id in range(self.config.rollout_workers):
 # Create queues
 result_queue = mp.Queue(maxsize=10)
 param_queue = mp.Queue(maxsize=5)
 
 self.result_queues.append(result_queue)
 self.param_queues.append(param_queue)
 
 # Create worker environments
 worker_envs = self._create_worker_environments(worker_id)
 
 # Create worker process
 worker = mp.Process(
 target=self._worker_main,
 args=(
 worker_id,
 self.config,
 self.base_trainer.actor_critic,
 worker_envs,
 result_queue,
 param_queue
 )
 )
 
 worker.start()
 self.rollout_workers.append(worker)
 self.worker_states[worker_id] = WorkerState(worker_id)
 
 self.logger.info(f"Started worker {worker_id}")
 
 def _worker_main(
 self,
 worker_id: int,
 config: DistributedPPOConfig,
 actor_critic: nn.Module,
 environments: List[Any],
 result_queue: mp.Queue,
 param_queue: mp.Queue
 ):
 """Worker process main function"""
 
 worker = RolloutWorker(
 worker_id=worker_id,
 config=config,
 actor_critic=actor_critic,
 environments=environments,
 result_queue=result_queue,
 param_queue=param_queue
 )
 
 worker.run()
 
 def _create_worker_environments(self, worker_id: int) -> List[Any]:
 """Create environments for specific worker"""
 
 # Calculate environments per worker
 envs_per_worker = self.config.num_envs // self.config.rollout_workers
 start_idx = worker_id * envs_per_worker
 end_idx = min(start_idx + envs_per_worker, self.config.num_envs)
 
 # Create environments
 worker_envs = []
 for i in range(start_idx, end_idx):
 env = self.base_trainer._create_environments()[0] # Create single env
 worker_envs.append(env)
 
 return worker_envs
 
 def _collect_distributed_rollouts(self) -> List[Dict[str, Any]]:
 """Collect rollouts from all workers"""
 
 rollout_data = []
 collected_workers = set()
 timeout_time = time.time() + self.config.worker_timeout
 
 while len(collected_workers) < len(self.rollout_workers):
 if time.time() > timeout_time:
 self.logger.warning("Worker timeout during rollout collection")
 break
 
 for i, result_queue in enumerate(self.result_queues):
 if i in collected_workers:
 continue
 
 try:
 result = result_queue.get(timeout=1.0)
 
 if "error" in result:
 self.logger.error(f"Worker {i} error: {result['error']}")
 self.worker_states[i].failure_count += 1
 collected_workers.add(i)
 continue
 
 rollout_data.append(result)
 collected_workers.add(i)
 
 # Update worker state
 if i in self.worker_states:
 self.worker_states[i].last_update_time = time.time()
 self.worker_states[i].total_timesteps += len(result["rollout_data"]["rewards"])
 
 except queue.Empty:
 continue
 
 return rollout_data
 
 def _aggregate_rollouts(self, rollout_data: List[Dict[str, Any]]) -> RolloutBuffer:
 """Aggregate rollouts from multiple workers"""
 
 if not rollout_data:
 raise RuntimeError("No rollout data collected")
 
 # Create aggregated buffer
 aggregated_buffer = RolloutBuffer(self.config.buffer_config)
 
 for data in rollout_data:
 rollout = data["rollout_data"]
 
 # Add to buffer
 batch_size = len(rollout["rewards"])
 for i in range(batch_size):
 aggregated_buffer.add(
 obs=rollout["observations"][i],
 action=rollout["actions"][i],
 reward=rollout["rewards"][i].item(),
 value=rollout["values"][i].item(),
 log_prob=rollout["log_probs"][i].item(),
 done=rollout["dones"][i].item()
 )
 
 # Compute advantages
 aggregated_buffer.compute_returns_and_advantages()
 
 return aggregated_buffer
 
 def _distributed_policy_update(self, rollout_buffer: RolloutBuffer) -> Dict[str, float]:
 """Perform distributed policy update"""
 
 # Standard PPO update
 progress_remaining = 1.0 - (self.base_trainer.current_timestep / self.config.total_timesteps)
 update_metrics = self.base_trainer.ppo.update(rollout_buffer, progress_remaining)
 
 # Distributed gradient aggregation
 if self.config.world_size > 1:
 self._aggregate_gradients()
 
 return update_metrics
 
 def _aggregate_gradients(self):
 """Aggregate gradients across distributed processes"""
 
 # Collect gradients
 for param in self.base_trainer.actor_critic.parameters():
 if param.grad is not None:
 # All-reduce gradients
 dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
 param.grad.data /= self.config.world_size
 
 def _synchronize_parameters(self):
 """Synchronize parameters across workers"""
 
 # Get current parameters
 actor_critic_params = self.base_trainer.actor_critic.state_dict()
 
 param_update = {
 "actor_critic_params": actor_critic_params,
 "timestamp": time.time()
 }
 
 # Add normalizer parameters
 if self.base_trainer.obs_normalizer is not None:
 param_update["normalizer_params"] = self.base_trainer.obs_normalizer.running_stats.get_state()
 
 # Send to all workers
 for param_queue in self.param_queues:
 try:
 param_queue.put(param_update, timeout=1.0)
 except queue.Full:
 self.logger.warning("Parameter queue full, skipping update")
 
 self.last_sync_time = time.time()
 
 def _monitor_workers(self):
 """Monitor worker health and performance"""
 
 current_time = time.time()
 
 for worker_id, state in self.worker_states.items():
 # Check for stale workers
 time_since_update = current_time - state.last_update_time
 
 if time_since_update > self.config.worker_timeout:
 self.logger.warning(f"Worker {worker_id} appears stale (last update {time_since_update:.1f}s ago)")
 state.is_active = False
 else:
 state.is_active = True
 
 def _handle_failed_workers(self):
 """Handle failed worker processes"""
 
 for i, (worker, state) in enumerate(zip(self.rollout_workers, self.worker_states.values())):
 if not worker.is_alive() and state.failure_count < self.config.max_worker_failures:
 self.logger.info(f"Restarting failed worker {i}")
 
 # Create new worker
 result_queue = mp.Queue(maxsize=10)
 param_queue = mp.Queue(maxsize=5)
 
 self.result_queues[i] = result_queue
 self.param_queues[i] = param_queue
 
 worker_envs = self._create_worker_environments(i)
 
 new_worker = mp.Process(
 target=self._worker_main,
 args=(
 i,
 self.config,
 self.base_trainer.actor_critic,
 worker_envs,
 result_queue,
 param_queue
 )
 )
 
 new_worker.start()
 self.rollout_workers[i] = new_worker
 state.failure_count += 1
 state.is_active = True
 state.last_update_time = time.time()
 
 def _log_distributed_progress(
 self,
 update_metrics: Dict[str, float],
 rollout_time: float,
 update_time: float
 ):
 """Log distributed training progress"""
 
 # Collect worker statistics
 active_workers = sum(1 for state in self.worker_states.values() if state.is_active)
 total_worker_timesteps = sum(state.total_timesteps for state in self.worker_states.values())
 avg_worker_fps = np.mean([state.current_fps for state in self.worker_states.values()])
 
 # Distributed metrics
 distributed_metrics = {
 "active_workers": active_workers,
 "total_worker_timesteps": total_worker_timesteps,
 "avg_worker_fps": avg_worker_fps,
 "rollout_time": rollout_time,
 "update_time": update_time,
 "sync_time": time.time() - self.last_sync_time,
 "distributed_efficiency": rollout_time / (rollout_time + update_time)
 }
 
 self.logger.info(
 f"Distributed Update {self.base_trainer.current_update} | "
 f"Active Workers: {active_workers}/{len(self.rollout_workers)} | "
 f"Worker FPS: {avg_worker_fps:.0f} | "
 f"Rollout Time: {rollout_time:.2f}s | "
 f"Update Time: {update_time:.2f}s"
 )
 
 # Store metrics
 for key, value in distributed_metrics.items():
 self.distributed_metrics[key].append(value)
 
 def _get_distributed_metrics(self, total_time: float) -> Dict[str, Any]:
 """Get final distributed training metrics"""
 
 base_metrics = self.base_trainer._get_final_metrics(total_time)
 
 distributed_metrics = {
 "total_workers": len(self.rollout_workers),
 "worker_failures": sum(state.failure_count for state in self.worker_states.values()),
 "avg_distributed_efficiency": np.mean(self.distributed_metrics.get("distributed_efficiency", [0])),
 "total_rollout_time": sum(self.distributed_metrics.get("rollout_time", [])),
 "total_update_time": sum(self.distributed_metrics.get("update_time", [])),
 "peak_worker_fps": max(self.distributed_metrics.get("avg_worker_fps", [0])) if self.distributed_metrics.get("avg_worker_fps") else 0
 }
 
 return {**base_metrics, **distributed_metrics}
 
 def _cleanup_workers(self):
 """Cleanup worker processes"""
 
 self.logger.info("Cleaning up worker processes")
 
 for worker in self.rollout_workers:
 if worker.is_alive():
 worker.terminate()
 worker.join(timeout=5.0)
 
 if worker.is_alive():
 self.logger.warning("Force killing worker process")
 worker.kill()
 
 # Clean up queues
 for queue in self.result_queues + self.param_queues:
 while not queue.empty():
 try:
 queue.get_nowait()
 except queue.Empty:
 break
 queue.close()
 
 def _cleanup_distributed(self):
 """Cleanup distributed training"""
 
 if dist.is_initialized():
 dist.destroy_process_group()
 
 self.logger.info("Distributed cleanup complete")


# Factory function
def create_distributed_trainer(
 config: DistributedPPOConfig,
 **kwargs
) -> DistributedPPOTrainer:
 """Create distributed PPO trainer"""
 
 return DistributedPPOTrainer(config=config, **kwargs)


# Export classes
__all__ = [
 "DistributedPPOConfig",
 "RolloutWorker",
 "DistributedPPOTrainer",
 "WorkerState",
 "create_distributed_trainer"
]