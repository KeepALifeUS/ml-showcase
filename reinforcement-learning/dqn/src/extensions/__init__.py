"""DQN Extensions and difficulties."""

from .double_dqn import DoubleDQN
from .dueling_dqn import DuelingDQN
from .noisy_dqn import NoisyDQN
from .rainbow_dqn import RainbowDQN

__all__ = [
 "DoubleDQN",
 "DuelingDQN",
 "NoisyDQN",
 "RainbowDQN",
]