"""
ML-PPO: Enterprise Proximal Policy Optimization
for crypto trading

Main package exports for easy importing
"""

# Core PPO algorithms
from .core.ppo import PPOAlgorithm, PPOConfig, PPOLoss
from .core.ppo2 import PPO2Algorithm, PPO2Config

# Network architectures 
from .networks.actor_critic import ActorCriticNetwork, ActorCriticConfig, CryptoActorCritic
from .networks.policy_network import PolicyNetwork, CryptoTradingPolicy
from .networks.value_network import ValueNetwork, CryptoValueNetwork

# Advantage estimation
from .advantages.gae import GAE, GAEConfig, RiskAdjustedGAE, AdaptiveGAE
from .advantages.td_lambda import TDLambdaEstimator, TDLambdaConfig

# Optimization
from .optimization.clipped_objective import StandardClippedObjective, ClippedObjectiveConfig
from .optimization.kl_penalty import StandardKLPenalty, KLPenaltyConfig

# Buffers
from .buffers.rollout_buffer import RolloutBuffer, RolloutBufferConfig
from .buffers.trajectory_buffer import TrajectoryBuffer, TrajectoryBufferConfig

# Training
from .training.ppo_trainer import PPOTrainer, PPOTrainerConfig
from .training.distributed_ppo import DistributedPPOTrainer, DistributedPPOConfig

# Agents
from .agents.ppo_trader import PPOTrader, PPOTraderConfig

# Environments
from .environments.crypto_env import CryptoTradingEnvironment, CryptoEnvConfig

# Utils
from .utils.normalization import RunningMeanStd, ObservationNormalizer, normalize_advantages
from .utils.scheduling import LinearSchedule, ExponentialSchedule, PPOScheduler

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__description__ = "Enterprise PPO implementation for crypto trading"

# Main exports
__all__ = [
 # Core
 "PPOAlgorithm", "PPOConfig", "PPOLoss",
 "PPO2Algorithm", "PPO2Config",
 
 # Networks
 "ActorCriticNetwork", "ActorCriticConfig", "CryptoActorCritic",
 "PolicyNetwork", "CryptoTradingPolicy",
 "ValueNetwork", "CryptoValueNetwork",
 
 # Advantages
 "GAE", "GAEConfig", "RiskAdjustedGAE", "AdaptiveGAE",
 "TDLambdaEstimator", "TDLambdaConfig",
 
 # Optimization
 "StandardClippedObjective", "ClippedObjectiveConfig",
 "StandardKLPenalty", "KLPenaltyConfig",
 
 # Buffers
 "RolloutBuffer", "RolloutBufferConfig",
 "TrajectoryBuffer", "TrajectoryBufferConfig",
 
 # Training
 "PPOTrainer", "PPOTrainerConfig",
 "DistributedPPOTrainer", "DistributedPPOConfig",
 
 # Agents
 "PPOTrader", "PPOTraderConfig",
 
 # Environments
 "CryptoTradingEnvironment", "CryptoEnvConfig",
 
 # Utils
 "RunningMeanStd", "ObservationNormalizer", "normalize_advantages",
 "LinearSchedule", "ExponentialSchedule", "PPOScheduler"
]