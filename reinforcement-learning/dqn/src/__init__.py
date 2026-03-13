"""
Deep Q-Network (DQN) for crypto trading with enterprise patterns.

This package provides full implementation DQN and its attribution:
- Base DQN with epsilon-greedy exploration
- Double DQN for elimination overestimation bias
- Dueling DQN with separate value and advantage threads
- Prioritized Experience Replay for efficient training
- Rainbow DQN combining all difficulties
- Specialized integration for crypto trading

Enterprise patterns :
- Production-ready error handling and logging
- Comprehensive monitoring and metrics
- Scalable architecture with async support
- Type hints and strict validation
- Performance optimization
"""

from typing import Dict, Any
import logging
from pathlib import Path

# Version package
__version__ = "1.0.0"

# Export main components
from .core.dqn import DQN
from .agents.dqn_trader import DQNTrader
from .training.dqn_trainer import DQNTrainer
from .buffers.replay_buffer import ReplayBuffer
from .buffers.prioritized_replay import PrioritizedReplayBuffer
from .networks.q_network import QNetwork
from .extensions.double_dqn import DoubleDQN
from .extensions.dueling_dqn import DuelingDQN
from .extensions.rainbow_dqn import RainbowDQN

# Configuration logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Creating directory for logs if not exists
PACKAGE_ROOT = Path(__file__).parent.parent
LOG_DIR = PACKAGE_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Export all public components
__all__ = [
 # Core components
 "DQN",
 "DQNTrader",
 "DQNTrainer",

 # Buffers
 "ReplayBuffer",
 "PrioritizedReplayBuffer",

 # Networks
 "QNetwork",

 # Extensions
 "DoubleDQN",
 "DuelingDQN",
 "RainbowDQN",

 # Utilities
 "get_package_info",
 "setup_logging",
]


def get_package_info -> Dict[str, Any]:
 """Get information about package."""
 return {
 "name": "ml-dqn",
 "version": __version__,
 "description": "Deep Q-Network implementation for cryptocurrency trading",
 "root_path": str(PACKAGE_ROOT),
 "log_directory": str(LOG_DIR),
 }


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
 """
 Configure logging for package.

 Args:
 level: Level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
 log_file: Path to file logs (optionally)
 """
 import logging.config

 config = {
 "version": 1,
 "disable_existing_loggers": False,
 "formatters": {
 "standard": {
 "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
 },
 "detailed": {
 "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
 }
 },
 "handlers": {
 "console": {
 "level": level,
 "class": "logging.StreamHandler",
 "formatter": "standard",
 "stream": "ext://sys.stdout"
 }
 },
 "root": {
 "level": level,
 "handlers": ["console"]
 }
 }

 if log_file:
 config["handlers"]["file"] = {
 "level": level,
 "class": "logging.FileHandler",
 "formatter": "detailed",
 "filename": log_file,
 "mode": "a"
 }
 config["root"]["handlers"].append("file")

 logging.config.dictConfig(config)


# Initialization logging by default
setup_logging

# Information about package when import
logger = logging.getLogger(__name__)
logger.info(f"Loaded ML-DQN package v{__version__}")
logger.info(f"Package root: {PACKAGE_ROOT}")