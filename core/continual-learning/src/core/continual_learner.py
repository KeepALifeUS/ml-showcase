"""
Continual Learning Framework for Crypto Trading Bot v5.0

Enterprise-grade system for continuous training trading models
with from catastrophic forgetting and integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Protocol, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json

# enterprise Imports
from ..utils.model_checkpoint import ModelCheckpointManager
from ..evaluation.forgetting_metrics import ForgettingMetrics
from ..evaluation.plasticity_metrics import PlasticityMetrics


class LearningStrategy(Enum):
    """Strategies continual training"""
    EWC = "elastic_weight_consolidation"
    REHEARSAL = "experience_replay"
    PROGRESSIVE = "progressive_neural_networks"
    PACKNET = "packnet_compression"
    LWF = "learning_without_forgetting"
    AGEM = "averaged_gradient_episodic_memory"


class TaskType(Enum):
    """Types tasks for continuous training"""
    TASK_INCREMENTAL = "task_incremental"  # New tasks trading
    CLASS_INCREMENTAL = "class_incremental" # New assets
    DOMAIN_INCREMENTAL = "domain_incremental" # New markets/
    ONLINE = "online_learning" # training


@dataclass
class TaskMetadata:
    """Metadata tasks for continual training"""
    task_id: int
    name: str
    task_type: TaskType
    description: str
    market_regime: str  # bull, bear, sideways, volatile
    assets: List[str]
    timeframe: str  # 1m, 5m, 1h, 1d
    start_time: datetime
    end_time: Optional[datetime] = None
    data_size: int = 0
    performance_baseline: Optional[float] = None


@dataclass
class LearnerConfig:
    """Configuration continual training"""
    # Main parameters
    strategy: LearningStrategy
    task_type: TaskType
    max_tasks: int = 10
    memory_budget: int = 1000
    
    # enterprise Settings
    enable_monitoring: bool = True
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100
    performance_threshold: float = 0.05
    
    # Specific for crypto trading
    market_adaptation_enabled: bool = True
    risk_aware_learning: bool = True
    feature_drift_detection: bool = True
    
    # Memory and performance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Regularization
    l1_lambda: float = 0.0001
    l2_lambda: float = 0.001
    dropout_rate: float = 0.1
    
    # Additional parameters
    verbose: bool = True
    log_level: str = "INFO"
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "backward_transfer", "forward_transfer", "memory_stability"
    ])


class ModelProtocol(Protocol):
    """Protocol for models compatible with continual learning"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass model"""
        ...
    
    def parameters(self):
        """Parameters model"""
        ...
    
    def state_dict(self) -> Dict[str, Any]:
        """State model"""
        ...
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state model"""
        ...


class ContinualLearner(ABC):
    """
    Base class for continual training in crypto trading.
    
    enterprise Features:
    - Adaptive learning with risk management
    - Production model versioning
    - Performance monitoring
    - Automatic rollback on degradation
    """
    
    def __init__(self, model: ModelProtocol, config: LearnerConfig):
        self.model = model
        self.config = config
        self.current_task = 0
        self.task_history: List[TaskMetadata] = []
        
        # enterprise Components
        self.checkpoint_manager = ModelCheckpointManager(
            checkpoint_dir=Path("checkpoints/continual_learning"),
            max_checkpoints=config.max_tasks * 2
        )
        self.forgetting_metrics = ForgettingMetrics()
        self.plasticity_metrics = PlasticityMetrics()
        
        # Monitor performance
        self.performance_history: Dict[int, Dict[str, float]] = {}
        self.memory_usage_history: List[Dict[str, Any]] = []
        
        # Configure logging
        self.logger = self._setup_logger()
        
        # Initialize metrics for 
        self._initialize__monitoring()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging for continual training"""
        logger = logging.getLogger(f"ContinualLearner-{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize__monitoring(self) -> None:
        """Initialize monitoring"""
        if self.config.enable_monitoring:
            self.monitoring_data = {
                "total_tasks": 0,
                "successful_adaptations": 0,
                "performance_degradations": 0,
                "memory_overflows": 0,
                "rollbacks": 0,
                "start_time": datetime.now().isoformat()
            }
            self.logger.info("monitoring initialized")
    
    @abstractmethod
    def learn_task(self, task_data: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Learn new task with catastrophic forgetting protection
        
        Args:
            task_data: Data for training new tasks
            task_metadata: Metadata tasks
            
        Returns:
            Dict with metrics performance
        """
        pass
    
    @abstractmethod
    def evaluate_task(self, task_id: int, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate performance on specific task
        
        Args:
            task_id: ID tasks for evaluation
            test_data: Test data
            
        Returns:
            Dict with metrics evaluation
        """
        pass
    
    def adapt_to_market_regime(self, new_regime: str, market_data: Dict[str, Any]) -> bool:
        """
        Adaptation to new market regime ( Adaptive ML)
        
        Args:
            new_regime: New market regime (bull/bear/sideways/volatile)
            market_data: Data market conditions
            
        Returns:
            True if successful, False
        """
        if not self.config.market_adaptation_enabled:
            return False
        
        self.logger.info(f"Adapting to new market regime: {new_regime}")
        
        try:
            # Create new tasks for adaptation to regime
            task_metadata = TaskMetadata(
                task_id=len(self.task_history),
                name=f"Market_Adaptation_{new_regime}",
                task_type=TaskType.DOMAIN_INCREMENTAL,
                description=f"Adaptation to {new_regime} market conditions",
                market_regime=new_regime,
                assets=market_data.get("assets", []),
                timeframe=market_data.get("timeframe", "1h"),
                start_time=datetime.now(),
                data_size=len(market_data.get("samples", []))
            )
            
            # Learning new tasks adaptation
            metrics = self.learn_task(market_data, task_metadata)
            
            # Check adaptation
            adaptation_threshold = self.config.performance_threshold
            success = metrics.get("accuracy", 0) > adaptation_threshold
            
            if success:
                self.monitoring_data["successful_adaptations"] += 1
                self.logger.info(f"Successfully adapted to {new_regime} regime")
            else:
                self.monitoring_data["performance_degradations"] += 1
                self.logger.warning(f"Failed to adapt to {new_regime} regime")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during market adaptation: {e}")
            return False
    
    def detect_feature_drift(self, new_data: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        """
        Detect drift features in new data
        
        Args:
            new_data: New data for analysis drift
            
        Returns:
            (drift_detected, drift_metrics)
        """
        if not self.config.feature_drift_detection:
            return False, {}
        
        # Simple implementation detection drift on basis statistics
        drift_metrics = {}
        drift_detected = False
        
        try:
            if hasattr(self, '_reference_stats') and self._reference_stats:
                current_stats = self._compute_data_statistics(new_data)
                
                for feature, current_stat in current_stats.items():
                    if feature in self._reference_stats:
                        ref_stat = self._reference_stats[feature]
                        drift_score = abs(current_stat - ref_stat) / (abs(ref_stat) + 1e-8)
                        drift_metrics[f"{feature}_drift"] = drift_score
                        
                        # Threshold drift (possible configure)
                        if drift_score > 0.1: # 10% change
                            drift_detected = True
            else:
                # First launch - creation reference statistics
                self._reference_stats = self._compute_data_statistics(new_data)
        
        except Exception as e:
            self.logger.error(f"Error in feature drift detection: {e}")
            return False, {}
        
        return drift_detected, drift_metrics
    
    def _compute_data_statistics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Computation statistics data for detection drift"""
        stats = {}
        
        if "features" in data:
            features = np.array(data["features"])
            if features.ndim == 2:
                stats.update({
                    "mean": float(np.mean(features)),
                    "std": float(np.std(features)),
                    "min": float(np.min(features)),
                    "max": float(np.max(features))
                })
        
        return stats
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save checkpoint model and state training
        
        Args:
            checkpoint_name: Name checkpoint' (optionally)
            
        Returns:
            Path to saved checkpoint'
        """
        if not self.config.enable_checkpointing:
            return ""
        
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "current_task": self.current_task,
            "task_history": [
                {
                    "task_id": task.task_id,
                    "name": task.name,
                    "task_type": task.task_type.value,
                    "market_regime": task.market_regime,
                    "start_time": task.start_time.isoformat()
                }
                for task in self.task_history
            ],
            "performance_history": self.performance_history,
            "config": {
                "strategy": self.config.strategy.value,
                "task_type": self.config.task_type.value,
                "max_tasks": self.config.max_tasks,
                "memory_budget": self.config.memory_budget
            }
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            checkpoint_data, checkpoint_name
        )
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load checkpoint model and state
        
        Args:
            checkpoint_path: Path to checkpoint'
            
        Returns:
            True if loading successful
        """
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # Load state model
            self.model.load_state_dict(checkpoint_data["model_state_dict"])
            
            # Restore state training
            self.current_task = checkpoint_data["current_task"]
            self.performance_history = checkpoint_data["performance_history"]
            
            # Restore history tasks
            self.task_history = []
            for task_data in checkpoint_data["task_history"]:
                task = TaskMetadata(
                    task_id=task_data["task_id"],
                    name=task_data["name"],
                    task_type=TaskType(task_data["task_type"]),
                    description="", # base
                    market_regime=task_data["market_regime"],
                    assets=[],
                    timeframe="1h",
                    start_time=datetime.fromisoformat(task_data["start_time"])
                )
                self.task_history.append(task)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary performance continual training
        
        Returns:
            Dict with metrics performance
        """
        if not self.performance_history:
            return {"status": "no_data", "tasks_completed": 0}
        
        # Computation main metrics
        all_accuracies = []
        for task_metrics in self.performance_history.values():
            if "accuracy" in task_metrics:
                all_accuracies.append(task_metrics["accuracy"])
        
        summary = {
            "tasks_completed": len(self.task_history),
            "current_task": self.current_task,
            "average_accuracy": np.mean(all_accuracies) if all_accuracies else 0.0,
            "latest_accuracy": all_accuracies[-1] if all_accuracies else 0.0,
            "memory_usage": len(getattr(self, 'memory_buffer', [])),
            "memory_budget": self.config.memory_budget,
            "strategy": self.config.strategy.value
        }
        
        #  monitoring data
        if hasattr(self, 'monitoring_data'):
            summary.update(self.monitoring_data)
        
        # Metrics forgetting and plasticity
        try:
            backward_transfer = self.forgetting_metrics.calculate_backward_transfer(
                self.performance_history
            )
            forward_transfer = self.plasticity_metrics.calculate_forward_transfer(
                self.performance_history
            )
            
            summary.update({
                "backward_transfer": backward_transfer,
                "forward_transfer": forward_transfer,
                "forgetting_measure": -backward_transfer, # BWT
                "learning_efficiency": (backward_transfer + forward_transfer) / 2
            })
        except Exception as e:
            self.logger.warning(f"Could not calculate transfer metrics: {e}")
        
        return summary
    
    def reset(self) -> None:
        """Reset state continual training"""
        self.current_task = 0
        self.task_history.clear()
        self.performance_history.clear()
        
        if hasattr(self, 'memory_buffer'):
            getattr(self, 'memory_buffer').clear()
        
        self.logger.info("Continual learner state reset")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"strategy={self.config.strategy.value}, "
            f"tasks_completed={len(self.task_history)}, "
            f"current_task={self.current_task})"
        )


#  Production-Ready Factory
class ContinualLearnerFactory:
    """
    Factory for creation training
    with enterprise patterns
    """
    
    @staticmethod
    def create_crypto_trader_learner(
        model: ModelProtocol,
        strategy: LearningStrategy = LearningStrategy.EWC,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> ContinualLearner:
        """
        Create continual training for
        
        Args:
            model: Model for training
            strategy: Strategy continual training
            config_overrides: Overrides configuration
            
        Returns:
            Configured system continual training
        """
        # Base configuration for crypto trading
        config = LearnerConfig(
            strategy=strategy,
            task_type=TaskType.DOMAIN_INCREMENTAL,
            max_tasks=50, # Many tasks for different market regimes
            memory_budget=5000, # Large buffer memory
            market_adaptation_enabled=True,
            risk_aware_learning=True,
            feature_drift_detection=True,
            enable_monitoring=True,
            enable_checkpointing=True,
            performance_threshold=0.60,  # 60% accuracy minimum
            batch_size=64,
            learning_rate=0.0005 # speed training
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create corresponding implementations
        if strategy == LearningStrategy.EWC:
            from ..strategies.ewc import EWCLearner
            return EWCLearner(model, config)
        elif strategy == LearningStrategy.REHEARSAL:
            from ..strategies.rehearsal import RehearsalLearner
            return RehearsalLearner(model, config)
        elif strategy == LearningStrategy.PROGRESSIVE:
            from ..strategies.progressive_neural import ProgressiveNetworkLearner
            return ProgressiveNetworkLearner(model, config)
        elif strategy == LearningStrategy.PACKNET:
            from ..strategies.packnet import PackNetLearner
            return PackNetLearner(model, config)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")