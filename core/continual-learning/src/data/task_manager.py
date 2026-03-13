"""
Task Manager for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system management tasks for
continuous training with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from queue import PriorityQueue
from threading import Lock
import asyncio

from .stream_generator import StreamSample, MarketRegime, BaseStreamGenerator
from ..core.continual_learner import TaskMetadata, TaskType


class TaskStatus(Enum):
    """ tasks"""
    PENDING = "pending" # execution
    READY = "ready" # to execution
    IN_PROGRESS = "in_progress" #
    COMPLETED = "completed" # Completed
    FAILED = "failed" #
    CANCELLED = "cancelled" #
    PAUSED = "paused" #


class TaskPriority(Enum):
    """ tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TaskSpec:
    """Specification tasks for continual training"""
    # Main parameters
    task_id: str
    name: str
    description: str
    task_type: TaskType
    priority: TaskPriority = TaskPriority.MEDIUM
    
    # Data tasks
    data_requirements: Dict[str, Any] = field(default_factory=dict)
    expected_samples: int = 1000
    max_samples: Optional[int] = None
    
    # Conditions execution
    market_regime_filter: Optional[List[MarketRegime]] = None
    asset_filter: Optional[List[str]] = None
    timeframe_filter: Optional[List[str]] = None
    
    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    start_after: Optional[datetime] = None
    deadline: Optional[datetime] = None
    max_duration: Optional[timedelta] = None
    
    #
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    blocks: List[str] = field(default_factory=list)     # Task IDs
    
    # enterprise settings
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    retry_config: Dict[str, Any] = field(default_factory=dict)
    
    # Callbacks
    on_start: Optional[Callable] = None
    on_complete: Optional[Callable] = None
    on_failure: Optional[Callable] = None


@dataclass
class TaskExecutionContext:
    """Context execution tasks"""
    task_spec: TaskSpec
    status: TaskStatus = TaskStatus.PENDING
    
    # Execute
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    # Results
    samples_collected: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    #
    allocated_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Retry information
    retry_count: int = 0
    max_retries: int = 3
    
    def is_ready_to_execute(self) -> bool:
        """Check readiness tasks to execution"""
        if self.status != TaskStatus.READY:
            return False
        
        # Check time
        if self.task_spec.start_after and datetime.now() < self.task_spec.start_after:
            return False
        
        # Check deadline
        if self.task_spec.deadline and datetime.now() > self.task_spec.deadline:
            return False
        
        return True
    
    def can_retry(self) -> bool:
        """Check capabilities repeated execution"""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED


class TaskManager:
    """
    Manager tasks for continual training
    
    enterprise Features:
    - Priority-based task scheduling
    - Dependency management
    - Resource allocation and limits
    - Performance monitoring
    - Automatic retry mechanisms
    - Market regime aware task filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("TaskManager")
        
        # tasks
        self.pending_tasks: PriorityQueue = PriorityQueue()
        self.active_tasks: Dict[str, TaskExecutionContext] = {}
        self.completed_tasks: Dict[str, TaskExecutionContext] = {}
        self.failed_tasks: Dict[str, TaskExecutionContext] = {}
        
        # Dependency graph
        self.dependency_graph: Dict[str, List[str]] = {}  # task_id -> [dependent_task_ids]
        
        #
        self.task_lock = Lock()
        
        # enterprise settings
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 5)
        self.task_timeout = self.config.get("task_timeout_hours", 24)
        self.enable_automatic_scheduling = self.config.get("auto_scheduling", True)
        self.performance_monitoring = self.config.get("performance_monitoring", True)
        
        # Statistics
        self.execution_stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0,
            "total_samples_collected": 0
        }
        
        # Scheduler
        self.scheduler_running = False
        
    def create_task(
        self,
        name: str,
        task_type: TaskType,
        market_regime: MarketRegime,
        assets: List[str],
        expected_samples: int = 1000,
        priority: TaskPriority = TaskPriority.MEDIUM,
        **kwargs
    ) -> str:
        """
        Create new tasks
        
        Args:
            name: Name tasks
            task_type: Type tasks
            market_regime: Market regime
            assets: List assets
            expected_samples: number samples
            priority: Priority tasks
            **kwargs: Additional parameters
            
        Returns:
            ID created tasks
        """
        task_id = str(uuid.uuid4())
        
        # Create metadata tasks
        task_metadata = TaskMetadata(
            task_id=len(self.active_tasks) + len(self.completed_tasks),  # Numeric ID
            name=name,
            task_type=task_type,
            description=kwargs.get("description", f"Task for {market_regime.value} market with {assets}"),
            market_regime=market_regime.value,
            assets=assets,
            timeframe=kwargs.get("timeframe", "1h"),
            start_time=datetime.now()
        )
        
        # Create tasks
        task_spec = TaskSpec(
            task_id=task_id,
            name=name,
            description=task_metadata.description,
            task_type=task_type,
            priority=priority,
            expected_samples=expected_samples,
            market_regime_filter=[market_regime] if market_regime else None,
            asset_filter=assets,
            timeframe_filter=kwargs.get("timeframe_filter"),
            start_after=kwargs.get("start_after"),
            deadline=kwargs.get("deadline"),
            max_duration=kwargs.get("max_duration"),
            depends_on=kwargs.get("depends_on", []),
            performance_requirements=kwargs.get("performance_requirements", {}),
            resource_limits=kwargs.get("resource_limits", {}),
            retry_config=kwargs.get("retry_config", {})
        )
        
        # Create context execution
        execution_context = TaskExecutionContext(
            task_spec=task_spec,
            status=TaskStatus.PENDING,
            max_retries=task_spec.retry_config.get("max_retries", 3)
        )
        
        with self.task_lock:
            # Add in queue with ( for PriorityQueue)
            priority_value = -priority.value # priority =
            self.pending_tasks.put((priority_value, datetime.now(), task_id, execution_context))
            
            # Update dependency graph
            self._update_dependency_graph(task_spec)
            
            # Statistics
            self.execution_stats["tasks_created"] += 1
        
        self.logger.info(f"Created task {task_id}: {name} with priority {priority.name}")
        return task_id
    
    def _update_dependency_graph(self, task_spec: TaskSpec) -> None:
        """
        Update graph dependencies
        
        Args:
            task_spec: Specification tasks
        """
        # Add dependencies
        for dependency_id in task_spec.depends_on:
            if dependency_id not in self.dependency_graph:
                self.dependency_graph[dependency_id] = []
            self.dependency_graph[dependency_id].append(task_spec.task_id)
        
        # Add
        if task_spec.task_id not in self.dependency_graph:
            self.dependency_graph[task_spec.task_id] = []
        
        for blocked_id in task_spec.blocks:
            self.dependency_graph[task_spec.task_id].append(blocked_id)
    
    def get_next_ready_task(self) -> Optional[TaskExecutionContext]:
        """
        Get next ready to execution tasks
        
        Returns:
            Context tasks or None
        """
        with self.task_lock:
            # Search tasks in queue
            temp_queue = []
            ready_task = None
            
            while not self.pending_tasks.empty():
                priority, created_time, task_id, context = self.pending_tasks.get()
                
                # Check dependencies
                if self._are_dependencies_satisfied(context.task_spec):
                    # Check readiness to execution
                    context.status = TaskStatus.READY
                    if context.is_ready_to_execute():
                        ready_task = context
                        break
                
                # Return in queue
                temp_queue.append((priority, created_time, task_id, context))
            
            # Return remaining tasks in queue
            for item in temp_queue:
                self.pending_tasks.put(item)
            
            return ready_task
    
    def _are_dependencies_satisfied(self, task_spec: TaskSpec) -> bool:
        """
        Check execution dependencies tasks
        
        Args:
            task_spec: Specification tasks
            
        Returns:
            True if all
        """
        for dependency_id in task_spec.depends_on:
            if dependency_id not in self.completed_tasks:
                return False
        
        return True
    
    def start_task_execution(self, context: TaskExecutionContext) -> bool:
        """
        Start execution tasks
        
        Args:
            context: Context tasks
            
        Returns:
            True if task successfully
        """
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return False
        
        with self.task_lock:
            context.status = TaskStatus.IN_PROGRESS
            context.started_at = datetime.now()
            self.active_tasks[context.task_spec.task_id] = context
        
        # Call callback
        if context.task_spec.on_start:
            try:
                context.task_spec.on_start(context)
            except Exception as e:
                self.logger.error(f"Error in on_start callback for task {context.task_spec.task_id}: {e}")
        
        self.logger.info(f"Started task execution: {context.task_spec.name}")
        return True
    
    def complete_task(
        self,
        task_id: str,
        performance_metrics: Dict[str, float],
        samples_collected: int = 0
    ) -> bool:
        """
        Completion execution tasks
        
        Args:
            task_id: ID tasks
            performance_metrics: Metrics performance
            samples_collected: Number samples
            
        Returns:
            True if task successfully completed
        """
        with self.task_lock:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task {task_id} not found in active tasks")
                return False
            
            context = self.active_tasks.pop(task_id)
            context.status = TaskStatus.COMPLETED
            context.completed_at = datetime.now()
            context.duration = context.completed_at - context.started_at
            context.performance_metrics = performance_metrics
            context.samples_collected = samples_collected
            
            self.completed_tasks[task_id] = context
            
            # Update statistics
            self.execution_stats["tasks_completed"] += 1
            self.execution_stats["total_samples_collected"] += samples_collected
            
            if context.duration:
                # Update time execution
                total_completed = self.execution_stats["tasks_completed"]
                old_avg = self.execution_stats["avg_execution_time"]
                new_time = context.duration.total_seconds()
                self.execution_stats["avg_execution_time"] = (
                    old_avg * (total_completed - 1) + new_time
                ) / total_completed
        
        # Call callback
        if context.task_spec.on_complete:
            try:
                context.task_spec.on_complete(context)
            except Exception as e:
                self.logger.error(f"Error in on_complete callback for task {task_id}: {e}")
        
        # Check tasks
        self._check_unblocked_tasks(task_id)
        
        self.logger.info(
            f"Completed task {context.task_spec.name} in {context.duration}. "
            f"Metrics: {performance_metrics}"
        )
        return True
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """
         tasks as
        
        Args:
            task_id: ID tasks
            error_message: about
            
        Returns:
            True if task as
        """
        with self.task_lock:
            if task_id not in self.active_tasks:
                self.logger.warning(f"Task {task_id} not found in active tasks")
                return False
            
            context = self.active_tasks.pop(task_id)
            context.status = TaskStatus.FAILED
            context.completed_at = datetime.now()
            context.duration = context.completed_at - context.started_at
            context.error_message = error_message
            
            # Check capabilities
            if context.can_retry():
                context.retry_count += 1
                context.status = TaskStatus.PENDING
                
                # in queue with
                retry_delay = context.task_spec.retry_config.get("delay_seconds", 300)
                context.task_spec.start_after = datetime.now() + timedelta(seconds=retry_delay)
                
                priority_value = -context.task_spec.priority.value
                self.pending_tasks.put((
                    priority_value,
                    datetime.now(),
                    task_id,
                    context
                ))
                
                self.logger.info(f"Task {task_id} scheduled for retry #{context.retry_count}")
            else:
                self.failed_tasks[task_id] = context
                self.execution_stats["tasks_failed"] += 1
                
                # Call callback
                if context.task_spec.on_failure:
                    try:
                        context.task_spec.on_failure(context)
                    except Exception as e:
                        self.logger.error(f"Error in on_failure callback for task {task_id}: {e}")
        
        self.logger.error(f"Task {task_id} failed: {error_message}")
        return True
    
    def _check_unblocked_tasks(self, completed_task_id: str) -> None:
        """
        Check tasks unblocked by completion of current task
        
        Args:
            completed_task_id: ID tasks
        """
        if completed_task_id in self.dependency_graph:
            dependent_tasks = self.dependency_graph[completed_task_id]
            
            for dependent_id in dependent_tasks:
                # Search tasks and update status
                self.logger.debug(f"Task {dependent_id} may be unblocked by completion of {completed_task_id}")
    
    def cancel_task(self, task_id: str) -> bool:
        """
         tasks
        
        Args:
            task_id: ID tasks
            
        Returns:
            True if task
        """
        with self.task_lock:
            # Search in active tasks
            if task_id in self.active_tasks:
                context = self.active_tasks.pop(task_id)
                context.status = TaskStatus.CANCELLED
                context.completed_at = datetime.now()
                self.failed_tasks[task_id] = context
                
                self.logger.info(f"Cancelled active task {task_id}")
                return True
            
            # Search in queue
            temp_queue = []
            found = False
            
            while not self.pending_tasks.empty():
                priority, created_time, tid, context = self.pending_tasks.get()
                
                if tid == task_id:
                    context.status = TaskStatus.CANCELLED
                    self.failed_tasks[task_id] = context
                    found = True
                    self.logger.info(f"Cancelled pending task {task_id}")
                else:
                    temp_queue.append((priority, created_time, tid, context))
            
            # Return remaining tasks in queue
            for item in temp_queue:
                self.pending_tasks.put(item)
            
            return found
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status tasks
        
        Args:
            task_id: ID tasks
            
        Returns:
            Information tasks
        """
        # Search in active tasks
        if task_id in self.active_tasks:
            context = self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            context = self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            context = self.failed_tasks[task_id]
        else:
            return None
        
        status_info = {
            "task_id": task_id,
            "name": context.task_spec.name,
            "status": context.status.value,
            "created_at": context.task_spec.created_at.isoformat(),
            "started_at": context.started_at.isoformat() if context.started_at else None,
            "completed_at": context.completed_at.isoformat() if context.completed_at else None,
            "duration_seconds": context.duration.total_seconds() if context.duration else None,
            "samples_collected": context.samples_collected,
            "performance_metrics": context.performance_metrics,
            "error_message": context.error_message,
            "retry_count": context.retry_count
        }
        
        return status_info
    
    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get status queue tasks
        
        Returns:
            Information about queue
        """
        with self.task_lock:
            queue_status = {
                "pending_tasks": self.pending_tasks.qsize(),
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "execution_stats": self.execution_stats.copy(),
                "max_concurrent_tasks": self.max_concurrent_tasks
            }
        
        return queue_status
    
    def filter_tasks_by_market_regime(
        self,
        market_regime: MarketRegime,
        include_pending: bool = True,
        include_active: bool = True,
        include_completed: bool = False
    ) -> List[Dict[str, Any]]:
        """
         tasks by market regime
        
        Args:
            market_regime: Market regime for
            include_pending: Enable tasks
            include_active: Enable tasks
            include_completed: Enable tasks
            
        Returns:
            List tasks
        """
        filtered_tasks = []
        
        # Check active tasks
        if include_active:
            for context in self.active_tasks.values():
                if (context.task_spec.market_regime_filter and 
                    market_regime in context.task_spec.market_regime_filter):
                    filtered_tasks.append(self.get_task_status(context.task_spec.task_id))
        
        # Check completed tasks
        if include_completed:
            for context in self.completed_tasks.values():
                if (context.task_spec.market_regime_filter and 
                    market_regime in context.task_spec.market_regime_filter):
                    filtered_tasks.append(self.get_task_status(context.task_spec.task_id))
        
        # TODO: Check pending tasks (requires iterations by PriorityQueue)
        
        return filtered_tasks
    
    def create_market_adaptation_task(
        self,
        current_regime: MarketRegime,
        new_regime: MarketRegime,
        assets: List[str],
        priority: TaskPriority = TaskPriority.HIGH
    ) -> str:
        """
        Create tasks adaptation to new market regime
        
        Market regime adaptation
        
        Args:
            current_regime: Current market regime
            new_regime: New market regime
            assets: List assets
            priority: Priority tasks
            
        Returns:
            ID created tasks
        """
        task_name = f"Market_Adaptation_{current_regime.value}_to_{new_regime.value}"
        
        # Determine number samples on basis complexity transition
        regime_transition_complexity = {
            (MarketRegime.BULL, MarketRegime.BEAR): 2000,
            (MarketRegime.BEAR, MarketRegime.BULL): 2000,
            (MarketRegime.VOLATILE, MarketRegime.SIDEWAYS): 1500,
            (MarketRegime.SIDEWAYS, MarketRegime.VOLATILE): 1800,
        }
        
        expected_samples = regime_transition_complexity.get(
            (current_regime, new_regime), 1200
        )
        
        # to performance
        performance_requirements = {
            "min_accuracy": 0.6,
            "max_adaptation_time_hours": 6,
            "stability_threshold": 0.05
        }
        
        task_id = self.create_task(
            name=task_name,
            task_type=TaskType.DOMAIN_INCREMENTAL,
            market_regime=new_regime,
            assets=assets,
            expected_samples=expected_samples,
            priority=priority,
            description=f"Adaptation from {current_regime.value} to {new_regime.value} market conditions",
            performance_requirements=performance_requirements,
            max_duration=timedelta(hours=8),
            deadline=datetime.now() + timedelta(hours=12)
        )
        
        self.logger.info(f"Created market adaptation task: {task_name}")
        return task_id
    
    def export_task_history(self, filepath: str) -> bool:
        """
        Export history tasks
        
        Args:
            filepath: Path for saving
            
        Returns:
            True if export successful
        """
        try:
            history = {
                "timestamp": datetime.now().isoformat(),
                "queue_status": self.get_queue_status(),
                "completed_tasks": [
                    self.get_task_status(task_id) 
                    for task_id in self.completed_tasks.keys()
                ],
                "failed_tasks": [
                    self.get_task_status(task_id) 
                    for task_id in self.failed_tasks.keys()
                ],
                "dependency_graph": self.dependency_graph
            }
            
            with open(filepath, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"Task history exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting task history: {e}")
            return False
    
    def cleanup_old_tasks(self, max_age_days: int = 30) -> int:
        """
        Cleanup old completed tasks
        
        Args:
            max_age_days: Maximum tasks in
            
        Returns:
            Number removed tasks
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0
        
        with self.task_lock:
            # Remove old completed tasks
            old_completed = [
                task_id for task_id, context in self.completed_tasks.items()
                if context.completed_at and context.completed_at < cutoff_date
            ]
            
            for task_id in old_completed:
                del self.completed_tasks[task_id]
                removed_count += 1
            
            # Remove old tasks
            old_failed = [
                task_id for task_id, context in self.failed_tasks.items()
                if context.completed_at and context.completed_at < cutoff_date
            ]
            
            for task_id in old_failed:
                del self.failed_tasks[task_id]
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old tasks")
        
        return removed_count
    
    def __repr__(self) -> str:
        return (
            f"TaskManager("
            f"pending={self.pending_tasks.qsize()}, "
            f"active={len(self.active_tasks)}, "
            f"completed={len(self.completed_tasks)}, "
            f"failed={len(self.failed_tasks)})"
        )