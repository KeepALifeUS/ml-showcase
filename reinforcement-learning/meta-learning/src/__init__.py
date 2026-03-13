"""
Meta-Learning System for Crypto Trading Bot v5.0
Production-Ready Meta-Learning

Comprehensive meta-learning system for fast adaptation to new
cryptocurrency markets and trading strategies.
"""

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__email__ = "team@ml-framework.io"
__description__ = "Comprehensive Meta-Learning System for Crypto Trading"

# Main algorithms
from .algorithms.maml import MAML, MAMLConfig, MAMLTrainer
from .algorithms.reptile import Reptile, ReptileConfig, ReptileTrainer
from .algorithms.meta_sgd import MetaSGD, MetaSGDConfig, MetaSGDTrainer
from .algorithms.proto_net import PrototypicalNetworks, ProtoNetConfig, ProtoNetTrainer
from .algorithms.matching_net import MatchingNetworks, MatchingNetConfig, MatchingNetTrainer

# System tasks
from .tasks.task_distribution import (
    BaseTaskDistribution, CryptoTaskDistribution, 
    CurriculumTaskDistribution, MultiDomainTaskDistribution,
    TaskConfig, TaskMetadata
)
from .tasks.task_sampler import TaskSampler, SamplerConfig, TaskCache, DataLoader
from .tasks.crypto_tasks import (
    CryptoPriceDirectionTask, CryptoPortfolioOptimizationTask,
    CryptoMarketSimulator, CryptoTaskConfig, MarketRegime
)

# Optimization
from .optimization.meta_optimizer import (
    MetaOptimizerFactory, MetaOptimizerConfig,
    MAMLOptimizer, ReptileOptimizer, AdaptiveMetaOptimizer
)
from .optimization.inner_loop import (
    InnerLoopOptimizerFactory, InnerLoopConfig,
    SGDInnerLoopOptimizer, AdamInnerLoopOptimizer, 
    MetaInitializedInnerLoopOptimizer
)

# Estimation
from .evaluation.few_shot_evaluator import (
    ClassificationEvaluator, RegressionEvaluator,
    FewShotBenchmark, EvaluationConfig
)

# Utilities
from .utils.gradient_utils import (
    GradientManager, HigherOrderGradients,
    GradientAccumulator, GradientProfiler
)
from .utils.meta_utils import (
    MetaLearningMetrics, DataAnalyzer, 
    Visualizer, ModelSerializer
)

# Main components for fast start
__all__ = [
    # Algorithms
    "MAML", "MAMLConfig", "MAMLTrainer",
    "Reptile", "ReptileConfig", "ReptileTrainer", 
    "MetaSGD", "MetaSGDConfig", "MetaSGDTrainer",
    "PrototypicalNetworks", "ProtoNetConfig", "ProtoNetTrainer",
    "MatchingNetworks", "MatchingNetConfig", "MatchingNetTrainer",
    
    # Tasks
    "CryptoTaskDistribution", "CryptoTaskConfig",
    "TaskSampler", "SamplerConfig",
    "CryptoPriceDirectionTask", "CryptoPortfolioOptimizationTask",
    "CryptoMarketSimulator", "MarketRegime",
    
    # Optimization
    "MetaOptimizerFactory", "MetaOptimizerConfig",
    "InnerLoopOptimizerFactory", "InnerLoopConfig",
    
    # Estimation
    "FewShotBenchmark", "EvaluationConfig",
    "ClassificationEvaluator", "RegressionEvaluator",
    
    # Utilities
    "MetaLearningMetrics", "GradientManager", 
    "DataAnalyzer", "Visualizer"
]

# Versioning information
VERSION_INFO = {
    "version": __version__,
    "algorithms": ["MAML", "Reptile", "Meta-SGD", "ProtoNet", "MatchingNet"],
    "task_types": ["classification", "regression", "portfolio_optimization"],
    "crypto_support": True,
    "_patterns": True,
    "production_ready": True
}

def get_version_info():
    """Returns information about version system"""
    return VERSION_INFO

def create_quick_setup():
    """
    Fast configuration for beginning work
    
    Returns:
        Tuple with base components
    """
    # Create configuration by default
    crypto_config = CryptoTaskConfig(
        task_type="classification",
        trading_pairs=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        num_classes=3,
        num_support=5,
        num_query=15
    )
    
    maml_config = MAMLConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        num_inner_steps=5
    )
    
    eval_config = EvaluationConfig(
        num_episodes=50,
        num_runs=3,
        support_shots=[1, 5],
        adaptation_steps=[1, 5]
    )
    
    return crypto_config, maml_config, eval_config

# License information
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 ML-Framework Team"

#  compliance
ENTERPRISE_PATTERNS = [
    "Enterprise Meta-Learning Architecture",
    "Production-Ready Algorithm Implementation", 
    "Comprehensive Evaluation Framework",
    "High-Performance Task Sampling",
    "Advanced Gradient Management",
    "Scalable Optimization Pipeline",
    "Statistical Analysis & Reporting",
    "Real-Time Adaptation Support"
]

print(f"🚀 Meta-Learning System v{__version__} loaded successfully!")
print(f"📊 Available algorithms: {', '.join(VERSION_INFO['algorithms'])}")
print(f"🔧 enterprise patterns: {len(ENTERPRISE_PATTERNS)} implemented")