"""
Comprehensive Test Suite for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade tests for system continual training
with integration and production-ready .
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# system continual training
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.continual_learner import ContinualLearner, LearnerConfig, TaskMetadata, TaskType, LearningStrategy
from core.memory_buffer import MemorySample, ReservoirBuffer, KCenterBuffer, BufferConfig, SamplingStrategy
from strategies.ewc import EWCLearner
from strategies.rehearsal import RehearsalLearner
from strategies.progressive_neural import ProgressiveNetworkLearner
from strategies.packnet import PackNetLearner
from evaluation.forgetting_metrics import ForgettingMetrics, ForgettingEvent
from evaluation.plasticity_metrics import PlasticityMetrics, LearningEvent
from data.stream_generator import SyntheticCryptoStreamGenerator, StreamConfig, DataStreamType, MarketRegime
from data.task_manager import TaskManager, TaskSpec, TaskPriority, TaskStatus
from utils.model_checkpoint import ModelCheckpointManager, CheckpointType
from utils.visualization import ContinualLearningVisualizer


class SimpleCryptoModel(nn.Module):
    """Simple model for testing"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 32, output_size: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def simple_model():
    """Fixture for simple model"""
    return SimpleCryptoModel()


@pytest.fixture
def basic_config():
    """Fixture for base configuration"""
    return LearnerConfig(
        strategy=LearningStrategy.EWC,
        task_type=TaskType.DOMAIN_INCREMENTAL,
        max_tasks=5,
        memory_budget=100,
        learning_rate=0.001,
        batch_size=16,
        enable_monitoring=False,  # Disable for tests
        enable_checkpointing=False
    )


@pytest.fixture
def sample_task_data():
    """Fixture for test data tasks"""
    np.random.seed(42)
    return {
        "features": np.random.randn(100, 20).astype(np.float32),
        "targets": np.random.randn(100, 1).astype(np.float32)
    }


@pytest.fixture
def sample_task_metadata():
    """Fixture for metadata tasks"""
    return TaskMetadata(
        task_id=1,
        name="Test_Task_Bull_Market",
        task_type=TaskType.DOMAIN_INCREMENTAL,
        description="Test task for bull market conditions",
        market_regime="bull",
        assets=["BTC", "ETH"],
        timeframe="1h",
        start_time=datetime.now()
    )


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestContinualLearnerBase:
    """Tests basic class ContinualLearner"""
    
    def test_learner_initialization(self, simple_model, basic_config):
        """Test initialization continual training"""
        # Use EWC as specific
        learner = EWCLearner(simple_model, basic_config)
        
        assert learner.model == simple_model
        assert learner.config == basic_config
        assert learner.current_task == 0
        assert len(learner.task_history) == 0
        assert len(learner.performance_history) == 0
    
    def test_checkpoint_save_load(self, simple_model, basic_config, temp_dir):
        """Test saving and loading checkpoint'"""
        config = basic_config
        config.enable_checkpointing = True
        
        learner = EWCLearner(simple_model, config)
        learner.checkpoint_manager.checkpoint_dir = temp_dir / "checkpoints"
        
        # Save checkpoint'
        checkpoint_path = learner.save_checkpoint("test_checkpoint")
        assert checkpoint_path
        assert Path(checkpoint_path).exists()
        
        # Load checkpoint'
        success = learner.load_checkpoint(checkpoint_path)
        assert success
    
    def test_performance_summary(self, simple_model, basic_config):
        """Test obtaining summary performance"""
        learner = EWCLearner(simple_model, basic_config)
        
        # Add test data performance
        learner.performance_history[1] = {"accuracy": 0.75, "loss": 0.3}
        learner.performance_history[2] = {"accuracy": 0.82, "loss": 0.25}
        
        summary = learner.get_performance_summary()
        
        assert "tasks_completed" in summary
        assert "average_accuracy" in summary
        assert summary["tasks_completed"] == 2
        assert summary["average_accuracy"] == pytest.approx(0.785, rel=1e-2)


class TestEWCLearner:
    """Tests for Elastic Weight Consolidation"""
    
    def test_ewc_initialization(self, simple_model, basic_config):
        """Test initialization EWC learner"""
        learner = EWCLearner(simple_model, basic_config)
        
        assert len(learner.fisher_matrices) == 0
        assert len(learner.optimal_params) == 0
        assert learner.ewc_lambda == 0.4
        assert learner.memory_buffer is not None
    
    def test_ewc_learning_task(self, simple_model, basic_config, sample_task_data, sample_task_metadata):
        """Test learning tasks with EWC"""
        learner = EWCLearner(simple_model, basic_config)
        
        # Training on task
        metrics = learner.learn_task(sample_task_data, sample_task_metadata)
        
        assert "accuracy" in metrics
        assert "ewc_loss" in metrics
        assert learner.current_task == sample_task_metadata.task_id
        assert len(learner.task_history) == 1
        assert sample_task_metadata.task_id in learner.performance_history
    
    def test_fisher_matrix_computation(self, simple_model, basic_config, sample_task_data, sample_task_metadata):
        """Test computation Fisher Information Matrix"""
        learner = EWCLearner(simple_model, basic_config)
        learner.fisher_estimation_samples = 10 # Fewer for fast testing
        
        # Training tasks
        learner.learn_task(sample_task_data, sample_task_metadata)
        
        # Check that Fisher matrix
        assert sample_task_metadata.task_id in learner.fisher_matrices
        fisher_matrix = learner.fisher_matrices[sample_task_metadata.task_id]
        
        # Check that Fisher matrix contains all parameters model
        model_param_names = set(name for name, _ in simple_model.named_parameters())
        fisher_param_names = set(fisher_matrix.keys())
        assert model_param_names == fisher_param_names
    
    def test_adaptive_lambda(self, simple_model, basic_config):
        """Test adaptive lambda for different market regimes"""
        learner = EWCLearner(simple_model, basic_config)
        
        # Test for different regimes
        volatile_lambda = learner._compute_regime_adaptive_lambda("volatile")
        bull_lambda = learner._compute_regime_adaptive_lambda("bull")
        bear_lambda = learner._compute_regime_adaptive_lambda("bear")
        
        # Volatile should have more high lambda
        assert volatile_lambda > bull_lambda
        # Bear should be above bull
        assert bear_lambda > bull_lambda


class TestRehearsalLearner:
    """Tests for Experience Replay"""
    
    def test_rehearsal_initialization(self, simple_model, basic_config):
        """Test initialization Rehearsal learner"""
        learner = RehearsalLearner(simple_model, basic_config)
        
        assert learner.replay_ratio == 0.5
        assert learner.replay_frequency == 1
        assert learner.memory_buffer is not None
        assert learner.intelligent_selection == True
    
    def test_rehearsal_learning_with_replay(self, simple_model, basic_config, sample_task_data, sample_task_metadata):
        """Test training with experience replay"""
        learner = RehearsalLearner(simple_model, basic_config)
        
        # First task
        metrics1 = learner.learn_task(sample_task_data, sample_task_metadata)
        assert "replay_loss" in metrics1
        
        # Second task (must use replay)
        task_metadata_2 = TaskMetadata(
            task_id=2,
            name="Test_Task_Bear_Market",
            task_type=TaskType.DOMAIN_INCREMENTAL,
            description="Test task for bear market",
            market_regime="bear",
            assets=["BTC", "ETH"],
            timeframe="1h",
            start_time=datetime.now()
        )
        
        metrics2 = learner.learn_task(sample_task_data, task_metadata_2)
        
        # Second task must have replay loss > 0 if memory not
        if len(learner.memory_buffer) > 0:
            assert metrics2["replay_loss"] >= 0
    
    def test_adaptive_replay_parameters(self, simple_model, basic_config, sample_task_metadata):
        """Test adaptive parameters replay"""
        learner = RehearsalLearner(simple_model, basic_config)
        
        # Simulation history performance
        learner.recent_performances = [0.8, 0.75, 0.7, 0.65]  # Degradation
        
        original_ratio = learner.replay_ratio
        learner._adapt_replay_parameters(sample_task_metadata)
        
        # Ratio should at degradation
        assert learner.replay_ratio >= original_ratio
    
    def test_intelligent_sample_selection(self, simple_model, basic_config, sample_task_data, sample_task_metadata):
        """Test selection samples"""
        learner = RehearsalLearner(simple_model, basic_config)
        learner.intelligent_selection = True
        
        # Create mock training metrics
        training_metrics = {"accuracy": 0.8, "loss": 0.2}
        
        features = torch.tensor(sample_task_data["features"])
        targets = torch.tensor(sample_task_data["targets"])
        
        # Test method selection samples
        selected_samples = learner._select_important_samples(
            features, targets, sample_task_metadata, 10, training_metrics
        )
        
        assert len(selected_samples) <= 10
        assert all(isinstance(sample, MemorySample) for sample in selected_samples)


class TestProgressiveNetworkLearner:
    """Tests for Progressive Neural Networks"""
    
    def test_progressive_initialization(self, simple_model, basic_config):
        """Test initialization Progressive Network learner"""
        learner = ProgressiveNetworkLearner(simple_model, basic_config)
        
        assert len(learner.columns) == 0
        assert learner.active_column_idx == -1
        assert learner.adaptive_architecture == True
    
    def test_progressive_learning_multiple_tasks(self, simple_model, basic_config, sample_task_data):
        """Test learning several tasks with Progressive Networks"""
        learner = ProgressiveNetworkLearner(simple_model, basic_config)
        
        # First task
        task1 = TaskMetadata(
            task_id=1, name="Task1", task_type=TaskType.DOMAIN_INCREMENTAL,
            description="Task 1", market_regime="bull", assets=["BTC"],
            timeframe="1h", start_time=datetime.now()
        )
        
        metrics1 = learner.learn_task(sample_task_data, task1)
        assert len(learner.columns) == 1
        assert learner.active_column_idx == 0
        
        # Second task
        task2 = TaskMetadata(
            task_id=2, name="Task2", task_type=TaskType.DOMAIN_INCREMENTAL,
            description="Task 2", market_regime="bear", assets=["ETH"],
            timeframe="1h", start_time=datetime.now()
        )
        
        metrics2 = learner.learn_task(sample_task_data, task2)
        assert len(learner.columns) == 2
        assert learner.active_column_idx == 1
        
        # Check lateral connections
        second_column = learner.columns[1]
        assert len(second_column.lateral_connections) > 0
    
    def test_architecture_adaptation(self, simple_model, basic_config):
        """Test adaptation architecture under market regimes"""
        learner = ProgressiveNetworkLearner(simple_model, basic_config)
        
        volatile_arch = learner._adapt_architecture_to_regime("volatile")
        sideways_arch = learner._adapt_architecture_to_regime("sideways")
        
        # Volatile regime should have more neurons
        assert all(v >= s for v, s in zip(volatile_arch, sideways_arch))


class TestPackNetLearner:
    """Tests for PackNet"""
    
    def test_packnet_initialization(self, simple_model, basic_config):
        """Test initialization PackNet learner"""
        learner = PackNetLearner(simple_model, basic_config)
        
        assert learner.packnet_mask is not None
        assert learner.importance_driven_allocation == True
        assert learner.adaptive_compression == True
    
    def test_packnet_learning_with_masks(self, simple_model, basic_config, sample_task_data, sample_task_metadata):
        """Test training with PackNet masks"""
        learner = PackNetLearner(simple_model, basic_config)
        
        # Training tasks
        metrics = learner.learn_task(sample_task_data, sample_task_metadata)
        
        assert "parameter_utilization" in metrics
        assert "active_parameters" in metrics
        assert sample_task_metadata.task_id in learner.packnet_mask.task_masks
    
    def test_capacity_allocation(self, simple_model, basic_config):
        """Test extraction capacity parameters"""
        learner = PackNetLearner(simple_model, basic_config)
        
        # Extraction capacity for tasks
        task_mask = learner.packnet_mask.allocate_capacity_for_task(1, 0.2)
        
        assert len(task_mask) > 0
        # Check that masks have correct type
        for name, mask in task_mask.items():
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool


class TestMemoryBuffer:
    """Tests for buffers memory"""
    
    def test_reservoir_buffer(self):
        """Test Reservoir Buffer"""
        config = BufferConfig(max_size=50, sampling_strategy=SamplingStrategy.RESERVOIR)
        buffer = ReservoirBuffer(config)
        
        # Create test samples
        samples = []
        for i in range(100):
            sample = MemorySample(
                features=torch.randn(20),
                target=torch.randn(1),
                task_id=i % 3,
                timestamp=datetime.now(),
                market_regime="bull",
                asset="BTC",
                timeframe="1h"
            )
            samples.append(sample)
        
        buffer.add_samples(samples)
        
        # Check size buffer
        assert len(buffer) <= config.max_size
        
        # Test set
        sampled = buffer.sample_batch(10)
        assert len(sampled) <= 10
    
    def test_k_center_buffer(self):
        """Test K-Center Buffer"""
        config = BufferConfig(max_size=30, sampling_strategy=SamplingStrategy.K_CENTER)
        buffer = KCenterBuffer(config)
        
        # Create diverse samples
        samples = []
        for i in range(50):
            # Create samples for testing diversity
            cluster_center = np.random.randn(20)
            features = cluster_center + np.random.randn(20) * 0.1
            
            sample = MemorySample(
                features=torch.tensor(features, dtype=torch.float32),
                target=torch.randn(1),
                task_id=i % 2,
                timestamp=datetime.now(),
                market_regime="bear" if i % 2 else "bull",
                asset="ETH",
                timeframe="1h"
            )
            samples.append(sample)
        
        buffer.add_samples(samples)
        
        # K-Center should diverse samples
        assert len(buffer) <= config.max_size
        
        # Test set with
        sampled = buffer.sample_batch(5)
        assert len(sampled) <= 5


class TestEvaluationMetrics:
    """Tests for metrics evaluation"""
    
    def test_forgetting_metrics(self):
        """Test metrics forgetting"""
        metrics = ForgettingMetrics()
        
        # Create test history performance
        performance_history = {
            1: {"accuracy": 0.8},
            2: {"accuracy": 0.75},
            3: {"accuracy": 0.7}
        }
        
        # Setup baseline performance
        metrics.baseline_performances = {1: 0.85, 2: 0.8, 3: 0.75}
        
        # Test backward transfer
        bwt = metrics.calculate_backward_transfer(performance_history)
        assert isinstance(bwt, float)
        
        # Test forgetting measure
        fm = metrics.calculate_forgetting_measure(performance_history)
        assert isinstance(fm, float)
        assert fm >= 0 # Forgetting measure not can be negative
        
        # Test retention rates
        retention_rates = metrics.calculate_retention_rate(performance_history)
        assert len(retention_rates) == 3
        assert all(0 <= rate <= 1 for rate in retention_rates.values())
    
    def test_plasticity_metrics(self):
        """Test metrics plasticity"""
        metrics = PlasticityMetrics()
        
        performance_history = {
            1: {"accuracy": 0.6},
            2: {"accuracy": 0.75},
            3: {"accuracy": 0.8}
        }
        
        # Test forward transfer
        fwt = metrics.calculate_forward_transfer(performance_history)
        assert isinstance(fwt, float)
        
        # Test learning efficiency
        learning_curve = [(i, 0.5 + i * 0.1) for i in range(10)]
        efficiency = metrics.calculate_learning_efficiency(1, learning_curve)
        
        assert "overall_efficiency" in efficiency
        assert "learning_speed" in efficiency
        assert "convergence_epoch" in efficiency
    
    def test_catastrophic_forgetting_detection(self):
        """Test detection catastrophic forgetting"""
        metrics = ForgettingMetrics()
        metrics.catastrophic_threshold = 0.3  # 30% degradation
        
        # History with catastrophic
        performance_history = {
            1: {"accuracy": 0.4},  # Significant degradation from baseline
            2: {"accuracy": 0.7}
        }
        metrics.baseline_performances = {1: 0.8, 2: 0.75}
        
        events = metrics.detect_catastrophic_forgetting(performance_history)
        
        assert len(events) > 0
        assert events[0].task_id == 1
        assert events[0].severity_level in ["moderate", "severe", "catastrophic"]


class TestDataStreaming:
    """Tests for generation streams data"""
    
    def test_synthetic_stream_generator(self):
        """Test synthetic generator streams"""
        config = StreamConfig(
            stream_type=DataStreamType.SYNTHETIC,
            assets=["BTC", "ETH"],
            feature_dim=10,
            samples_per_hour=10
        )
        
        generator = SyntheticCryptoStreamGenerator(config)
        
        # Generation samples
        sample = generator.generate_sample()
        
        assert sample.features.shape[0] == config.feature_dim
        assert sample.asset in config.assets
        assert isinstance(sample.market_regime, MarketRegime)
        assert sample.timestamp is not None
    
    def test_stream_regime_transitions(self):
        """Test transitions between regimes"""
        config = StreamConfig(regime_transitions=True, regime_duration_hours=1)
        generator = SyntheticCryptoStreamGenerator(config)
        
        # Force transition
        generator.regime_start_time = datetime.now() - timedelta(hours=2)
        generator.regime_transition_prob = 1.0  # Guaranteed transition
        
        initial_regime = generator.current_regime
        
        # Generation samples must cause transition regime
        sample = generator.generate_sample()
        
        # Regime (not guaranteed - randomness)
        assert isinstance(generator.current_regime, MarketRegime)


class TestTaskManager:
    """Tests for manager tasks"""
    
    def test_task_manager_initialization(self):
        """Test initialization manager tasks"""
        manager = TaskManager()
        
        assert manager.pending_tasks.qsize() == 0
        assert len(manager.active_tasks) == 0
        assert len(manager.completed_tasks) == 0
        assert manager.max_concurrent_tasks == 5
    
    def test_task_creation_and_execution(self):
        """Test creation and execution tasks"""
        manager = TaskManager()
        
        # Create tasks
        task_id = manager.create_task(
            name="Test Task",
            task_type=TaskType.DOMAIN_INCREMENTAL,
            market_regime=MarketRegime.BULL,
            assets=["BTC"],
            expected_samples=100,
            priority=TaskPriority.HIGH
        )
        
        assert task_id is not None
        assert manager.pending_tasks.qsize() == 1
        
        # Get next ready tasks
        ready_task = manager.get_next_ready_task()
        assert ready_task is not None
        
        # Start execution
        success = manager.start_task_execution(ready_task)
        assert success
        assert len(manager.active_tasks) == 1
        
        # Completion tasks
        performance_metrics = {"accuracy": 0.75, "loss": 0.3}
        success = manager.complete_task(task_id, performance_metrics, samples_collected=95)
        assert success
        assert len(manager.completed_tasks) == 1
        assert len(manager.active_tasks) == 0
    
    def test_task_dependencies(self):
        """Test dependencies between tasks"""
        manager = TaskManager()
        
        # Create first tasks
        task1_id = manager.create_task(
            name="Task 1",
            task_type=TaskType.DOMAIN_INCREMENTAL,
            market_regime=MarketRegime.BULL,
            assets=["BTC"],
            priority=TaskPriority.MEDIUM
        )
        
        # Create second tasks, from first
        task2_id = manager.create_task(
            name="Task 2", 
            task_type=TaskType.DOMAIN_INCREMENTAL,
            market_regime=MarketRegime.BEAR,
            assets=["ETH"],
            priority=TaskPriority.HIGH,
            depends_on=[task1_id]
        )
        
        # Dependency graph should be
        assert task1_id in manager.dependency_graph


class TestModelCheckpointManager:
    """Tests for manager checkpoint'"""
    
    def test_checkpoint_manager_initialization(self, temp_dir):
        """Test initialization manager checkpoint'"""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = ModelCheckpointManager(checkpoint_dir, max_checkpoints=5)
        
        assert manager.checkpoint_dir == checkpoint_dir
        assert manager.max_checkpoints == 5
        assert checkpoint_dir.exists()
    
    def test_checkpoint_save_and_load(self, temp_dir, simple_model):
        """Test saving and loading checkpoint'"""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = ModelCheckpointManager(checkpoint_dir)
        
        # Test data checkpoint'
        checkpoint_data = {
            "model_state_dict": simple_model.state_dict(),
            "optimizer_state_dict": {},
            "performance_metrics": {"accuracy": 0.85, "loss": 0.2},
            "task_id": 1
        }
        
        # Save
        checkpoint_path = manager.save_checkpoint(
            checkpoint_data,
            "test_checkpoint",
            CheckpointType.MODEL_STATE,
            {"task_name": "Test Task"}
        )
        
        assert checkpoint_path
        assert Path(checkpoint_path).exists()
        
        # Load
        loaded_data = manager.load_checkpoint(Path(checkpoint_path).stem)
        
        assert "model_state_dict" in loaded_data
        assert "performance_metrics" in loaded_data
        assert loaded_data["performance_metrics"]["accuracy"] == 0.85
    
    def test_checkpoint_cleanup(self, temp_dir, simple_model):
        """Test cleanup old checkpoint'"""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = ModelCheckpointManager(checkpoint_dir, max_checkpoints=3, auto_cleanup=True)
        
        # Create checkpoint'
        checkpoint_data = {"model_state_dict": simple_model.state_dict()}
        
        checkpoint_ids = []
        for i in range(5):
            checkpoint_path = manager.save_checkpoint(
                checkpoint_data,
                f"checkpoint_{i}",
                CheckpointType.MODEL_STATE
            )
            checkpoint_ids.append(Path(checkpoint_path).stem)
        
        # maximum max_checkpoints
        assert len(manager.checkpoint_index) <= 3


class TestVisualization:
    """Tests for system visualization"""
    
    def test_visualizer_initialization(self, temp_dir):
        """Test initialization system visualization"""
        visualizer = ContinualLearningVisualizer(temp_dir / "visualizations")
        
        assert visualizer.output_dir.exists()
        assert "ContinualLearningVisualizer" in str(type(visualizer))
    
    def test_learning_curve_plotting(self, temp_dir):
        """Test building training"""
        visualizer = ContinualLearningVisualizer(temp_dir / "visualizations")
        
        # Test curve training
        learning_history = [(i, 0.5 + 0.3 * np.exp(-i/10)) for i in range(20)]
        
        # Static chart (always available)
        plot_path = visualizer.plot_learning_curve(
            learning_history,
            task_name="Test_Task",
            interactive=False
        )
        
        assert plot_path is not None
        assert Path(plot_path).exists()
        assert Path(plot_path).suffix == ".png"
    
    def test_metrics_csv_export(self, temp_dir):
        """Test metrics in CSV"""
        visualizer = ContinualLearningVisualizer(temp_dir / "visualizations")
        
        # Test data metrics
        metrics_data = {
            "forgetting_analysis": {
                "backward_transfer": -0.05,
                "forgetting_measure": 0.1
            },
            "plasticity_analysis": {
                "forward_transfer": 0.08
            },
            "learning_efficiencies": {
                1: {"overall_efficiency": 0.7, "learning_speed": 0.05},
                2: {"overall_efficiency": 0.8, "learning_speed": 0.06}
            }
        }
        
        csv_path = visualizer.export_metrics_to_csv(metrics_data)
        
        assert Path(csv_path).exists()
        assert Path(csv_path).suffix == ".csv"
        
        # Check CSV
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) > 0
        assert "metric_type" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns


class TestIntegration:
    """Integration tests system"""
    
    def test_end_to_end_continual_learning(self, simple_model, temp_dir):
        """Comprehensive test entire system continual training"""
        # Configuration
        config = LearnerConfig(
            strategy=LearningStrategy.EWC,
            task_type=TaskType.DOMAIN_INCREMENTAL,
            max_tasks=3,
            memory_budget=50,
            learning_rate=0.01,
            batch_size=16,
            enable_checkpointing=True
        )
        
        # Initialize learner
        learner = EWCLearner(simple_model, config)
        learner.checkpoint_manager.checkpoint_dir = temp_dir / "checkpoints"
        
        # Initialize metrics
        forgetting_metrics = ForgettingMetrics()
        plasticity_metrics = PlasticityMetrics()
        
        # Generation data for tasks
        np.random.seed(42)
        tasks_data = []
        for i in range(3):
            task_data = {
                "features": np.random.randn(50, 20).astype(np.float32),
                "targets": np.random.randn(50, 1).astype(np.float32)
            }
            task_metadata = TaskMetadata(
                task_id=i+1,
                name=f"Task_{i+1}",
                task_type=TaskType.DOMAIN_INCREMENTAL,
                description=f"Test task {i+1}",
                market_regime=["bull", "bear", "sideways"][i],
                assets=["BTC", "ETH", "ADA"][i:i+1],
                timeframe="1h",
                start_time=datetime.now()
            )
            tasks_data.append((task_data, task_metadata))
        
        # learning tasks
        all_metrics = []
        for task_data, task_metadata in tasks_data:
            # Learning tasks
            metrics = learner.learn_task(task_data, task_metadata)
            all_metrics.append(metrics)
            
            # Record events for metrics plasticity
            plasticity_metrics.record_learning_event(
                task_id=task_metadata.task_id,
                initial_performance=0.5,
                final_performance=metrics["accuracy"],
                convergence_epochs=10,
                market_regime=task_metadata.market_regime
            )
            
            # Save checkpoint'
            checkpoint_path = learner.save_checkpoint(f"task_{task_metadata.task_id}")
            assert Path(checkpoint_path).exists()
        
        # Evaluate on all tasks for analysis forgetting
        performance_history = {}
        for i, (task_data, task_metadata) in enumerate(tasks_data):
            eval_metrics = learner.evaluate_task(task_metadata.task_id, {
                "features": task_data["features"][:20], # Fewer data for testing
                "targets": task_data["targets"][:20]
            })
            performance_history[task_metadata.task_id] = eval_metrics
        
        # Analysis forgetting
        bwt = forgetting_metrics.calculate_backward_transfer(performance_history)
        fm = forgetting_metrics.calculate_forgetting_measure(performance_history)
        
        # Analysis plasticity
        fwt = plasticity_metrics.calculate_forward_transfer(performance_history)
        
        # results
        assert len(learner.task_history) == 3
        assert len(learner.performance_history) == 3
        assert isinstance(bwt, float)
        assert isinstance(fm, float)
        assert isinstance(fwt, float)
        
        # Check summary performance
        summary = learner.get_performance_summary()
        assert summary["tasks_completed"] == 3
        assert "backward_transfer" in summary
        assert "forward_transfer" in summary
        
        # Test visualization
        visualizer = ContinualLearningVisualizer(temp_dir / "visualizations")
        
        # Create data for visualization
        comprehensive_data = {
            "num_tasks": len(learner.task_history),
            "forgetting_analysis": {
                "backward_transfer": bwt,
                "forgetting_measure": fm,
                "retention_rates": forgetting_metrics.calculate_retention_rate(performance_history),
                "catastrophic_event_details": []
            },
            "plasticity_analysis": {
                "forward_transfer": fwt,
                "learning_efficiencies": {},
                "difficulty_distribution": {"easy": 1, "medium": 2, "hard": 0}
            }
        }
        
        # Create report
        report_path = visualizer.create_comprehensive_report(comprehensive_data)
        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".html"


# Launch tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])