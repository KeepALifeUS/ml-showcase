"""
ML Curiosity Exploration System

Enterprise-grade curiosity-driven exploration for autonomous cryptocurrency trading
strategy discovery, built with enterprise patterns.

🚀 Core Features:
- Intrinsic Curiosity Module (ICM) for controllable exploration
- Random Network Distillation (RND) for uncertainty-based exploration
- Never Give Up (NGU) for persistent exploration with memory
- Advanced novelty detection and state visitation tracking
- Production-ready trading agents with curiosity integration

📦 Main Components:
- curiosity: Core curiosity mechanisms (ICM, RND, NGU)
- exploration: Exploration strategies (count-based, prediction-based)
- novelty: Novelty detection systems
- memory: Episodic memory and replay buffers
- agents: Curious trading and exploration agents
- utils: State encoding and reward shaping utilities

🏗️ enterprise patterns:
- Distributed training and inference
- Real-time processing capabilities
- Scalable memory management
- Production deployment ready
- Comprehensive monitoring and observability

Author: ML-Framework Team
Version: 1.0.0
License: MIT
"""

from .curiosity import (
    ICMConfig,
    ICMTrainer,
    RNDConfig, 
    RNDTrainer,
    NGUConfig,
    NGUTrainer,
    create_icm_system,
    create_rnd_system,
    create_ngu_system
)

from .exploration import (
    CountBasedConfig,
    CountBasedExplorer,
    PredictionBasedConfig,
    PredictionBasedExplorer,
    ExplorationBonusConfig,
    ExplorationBonusManager,
    create_count_based_system,
    create_prediction_based_system,
    create_exploration_bonus_system
)

from .novelty import (
    NoveltyDetectionConfig,
    CryptoNoveltyDetector,
    StateVisitorConfig,
    StateVisitor,
    create_novelty_detection_system,
    create_state_visitor_system
)

from .memory import (
    EpisodicMemoryConfig,
    EpisodicMemorySystem,
    CuriosityBufferConfig,
    CuriosityReplayBuffer
)

from .agents import (
    CuriousTraderConfig,
    CuriousTrader,
    ExplorationAgentConfig,
    ExplorationAgent
)

from .utils import (
    StateEncoderConfig,
    CryptoStateEncoder,
    RewardShapingConfig,
    CryptoRewardShaper,
    create_state_encoder,
    create_reward_shaper
)

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__license__ = "MIT"

__all__ = [
    # Curiosity systems
    "ICMConfig", "ICMTrainer", "create_icm_system",
    "RNDConfig", "RNDTrainer", "create_rnd_system", 
    "NGUConfig", "NGUTrainer", "create_ngu_system",
    
    # Exploration systems
    "CountBasedConfig", "CountBasedExplorer", "create_count_based_system",
    "PredictionBasedConfig", "PredictionBasedExplorer", "create_prediction_based_system",
    "ExplorationBonusConfig", "ExplorationBonusManager", "create_exploration_bonus_system",
    
    # Novelty detection
    "NoveltyDetectionConfig", "CryptoNoveltyDetector", "create_novelty_detection_system",
    "StateVisitorConfig", "StateVisitor", "create_state_visitor_system",
    
    # Memory systems
    "EpisodicMemoryConfig", "EpisodicMemorySystem",
    "CuriosityBufferConfig", "CuriosityReplayBuffer",
    
    # Trading agents
    "CuriousTraderConfig", "CuriousTrader",
    "ExplorationAgentConfig", "ExplorationAgent",
    
    # Utilities
    "StateEncoderConfig", "CryptoStateEncoder", "create_state_encoder",
    "RewardShapingConfig", "CryptoRewardShaper", "create_reward_shaper"
]