"""
Epsilon Scheduling for DQN exploration with enterprise patterns.

Implements various strategy epsilon decay for optimally exploration:
- Linear decay with configurable schedule
- Exponential decay with adaptive rates
- Step-wise decay with milestone-based reductions
- Cosine annealing with warm restarts
- Custom schedules through lambda functions
- Production monitoring and logging
"""

import logging
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import math
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, validator
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class ScheduleType(str, Enum):
 """Types epsilon schedules."""
 LINEAR = "linear"
 EXPONENTIAL = "exponential"
 COSINE = "cosine"
 STEP = "step"
 POLYNOMIAL = "polynomial"
 CUSTOM = "custom"


@dataclass
class ScheduleState:
 """State epsilon schedule for persistence."""
 current_step: int
 current_epsilon: float
 schedule_type: str
 parameters: Dict[str, Any]


class EpsilonScheduleConfig(BaseModel):
 """Configuration epsilon schedule with validation."""

 # Basic parameters
 start_epsilon: float = Field(default=1.0, description="Initial epsilon", ge=0, le=1.0)
 end_epsilon: float = Field(default=0.01, description="Final epsilon", ge=0, le=1.0)

 # Schedule type and parameters
 schedule_type: ScheduleType = Field(default=ScheduleType.EXPONENTIAL, description="Type schedule")

 # Linear schedule
 total_steps: Optional[int] = Field(default=100000, description="Total number of steps", gt=0)

 # Exponential schedule
 decay_rate: float = Field(default=0.995, description="Exponential decay rate", ge=0.9, le=1.0)

 # Step schedule
 step_sizes: Optional[List[int]] = Field(default=None, description="Sizes for step decay")
 step_gammas: Optional[List[float]] = Field(default=None, description="Multipliers for step decay")

 # Cosine schedule
 cosine_restarts: bool = Field(default=False, description="Warm restarts for cosine")
 restart_periods: Optional[List[int]] = Field(default=None, description="Periods for restarts")

 # Polynomial schedule
 power: float = Field(default=1.0, description="Power for polynomial decay", gt=0)

 # General parameters
 min_epsilon: float = Field(default=0.001, description="Absolute minimum epsilon", ge=0, le=0.1)
 warmup_steps: int = Field(default=0, description="Steps for warmup", ge=0)

 @validator("end_epsilon")
 def validate_end_epsilon(cls, v, values):
 if "start_epsilon" in values and v >= values["start_epsilon"]:
 raise ValueError("end_epsilon must be < start_epsilon")
 return v

 @validator("min_epsilon")
 def validate_min_epsilon(cls, v, values):
 if "end_epsilon" in values and v > values["end_epsilon"]:
 raise ValueError("min_epsilon must be <= end_epsilon")
 return v

 @validator("step_gammas")
 def validate_step_gammas(cls, v, values):
 if v is not None and "step_sizes" in values:
 if values["step_sizes"] is not None and len(v) != len(values["step_sizes"]):
 raise ValueError("step_gammas must have such also length as step_sizes")
 return v


class BaseEpsilonSchedule(ABC):
 """Base class for epsilon schedules."""

 def __init__(self, config: EpsilonScheduleConfig):
 self.config = config
 self.current_step = 0
 self.logger = structlog.get_logger(__name__).bind(
 component="EpsilonSchedule",
 schedule_type=config.schedule_type
 )

 @abstractmethod
 def get_epsilon(self, step: Optional[int] = None) -> float:
 """Get epsilon for given step."""
 pass

 def step(self) -> float:
 """Increment step and get new epsilon."""
 self.current_step += 1
 return self.get_epsilon

 def reset(self) -> None:
 """Reset schedule to initial state."""
 self.current_step = 0

 def get_state(self) -> ScheduleState:
 """Get current state for persistence."""
 return ScheduleState(
 current_step=self.current_step,
 current_epsilon=self.get_epsilon,
 schedule_type=self.config.schedule_type.value,
 parameters=self.config.dict
 )

 def load_state(self, state: ScheduleState) -> None:
 """Load state from persistence."""
 self.current_step = state.current_step


class LinearSchedule(BaseEpsilonSchedule):
 """Linear epsilon decay."""

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps
 total_steps = self.config.total_steps - self.config.warmup_steps

 # Linear interpolation
 progress = min(effective_step / total_steps, 1.0)
 epsilon = self.config.start_epsilon - progress * (
 self.config.start_epsilon - self.config.end_epsilon
 )

 return max(epsilon, self.config.min_epsilon)


class ExponentialSchedule(BaseEpsilonSchedule):
 """Exponential epsilon decay."""

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps

 # Exponential decay
 epsilon = self.config.end_epsilon + (
 self.config.start_epsilon - self.config.end_epsilon
 ) * (self.config.decay_rate ** effective_step)

 return max(epsilon, self.config.min_epsilon)


class CosineSchedule(BaseEpsilonSchedule):
 """Cosine annealing epsilon schedule with optional restarts."""

 def __init__(self, config: EpsilonScheduleConfig):
 super.__init__(config)
 self.restart_step = 0
 self.restart_count = 0

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps

 if self.config.cosine_restarts and self.config.restart_periods:
 # Determining current restart period
 period = self.config.restart_periods[
 min(self.restart_count, len(self.config.restart_periods) - 1)
 ]

 if effective_step >= self.restart_step + period:
 self.restart_step = effective_step
 self.restart_count += 1

 step_in_period = effective_step - self.restart_step
 progress = step_in_period / period
 else:
 # Standard cosine schedule
 total_steps = self.config.total_steps - self.config.warmup_steps
 progress = min(effective_step / total_steps, 1.0)

 # Cosine annealing
 epsilon = self.config.end_epsilon + 0.5 * (
 self.config.start_epsilon - self.config.end_epsilon
 ) * (1 + math.cos(math.pi * progress))

 return max(epsilon, self.config.min_epsilon)


class StepSchedule(BaseEpsilonSchedule):
 """Step-wise epsilon decay at milestones."""

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps

 # Determining current step multiplier
 epsilon = self.config.start_epsilon

 if self.config.step_sizes and self.config.step_gammas:
 cumulative_step = 0
 for step_size, gamma in zip(self.config.step_sizes, self.config.step_gammas):
 cumulative_step += step_size
 if effective_step >= cumulative_step:
 epsilon *= gamma
 else:
 break

 # Ensure not below end_epsilon
 epsilon = max(epsilon, self.config.end_epsilon)
 return max(epsilon, self.config.min_epsilon)


class PolynomialSchedule(BaseEpsilonSchedule):
 """Polynomial epsilon decay."""

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps
 total_steps = self.config.total_steps - self.config.warmup_steps

 # Polynomial decay
 progress = min(effective_step / total_steps, 1.0)
 epsilon = self.config.end_epsilon + (
 self.config.start_epsilon - self.config.end_epsilon
 ) * ((1 - progress) ** self.config.power)

 return max(epsilon, self.config.min_epsilon)


class CustomSchedule(BaseEpsilonSchedule):
 """Custom epsilon schedule through lambda function."""

 def __init__(self, config: EpsilonScheduleConfig, schedule_fn: Callable[[int], float]):
 super.__init__(config)
 self.schedule_fn = schedule_fn

 def get_epsilon(self, step: Optional[int] = None) -> float:
 current = step if step is not None else self.current_step

 # Warmup phase
 if current < self.config.warmup_steps:
 return self.config.start_epsilon

 effective_step = current - self.config.warmup_steps
 epsilon = self.schedule_fn(effective_step)

 return max(epsilon, self.config.min_epsilon)


class EpsilonSchedule:
 """
 Factory class for creation epsilon schedules with enterprise functionality.

 Features:
 - Multiple schedule types (linear, exponential, cosine, step, polynomial)
 - Configurable parameters through Pydantic validation
 - Warmup periods for stable start
 - Hard initial limits for safety
 - Comprehensive logging and monitoring
 - State persistence for checkpointing
 - Custom schedules through lambda functions
 """

 def __init__(self,
 config: Optional[EpsilonScheduleConfig] = None,
 start_epsilon: float = 1.0,
 end_epsilon: float = 0.01,
 decay_rate: float = 0.995,
 schedule_type: ScheduleType = ScheduleType.EXPONENTIAL,
 custom_schedule: Optional[Callable[[int], float]] = None):
 """
 Initialization epsilon schedule.

 Args:
 config: Full configuration (prior individual parameters)
 start_epsilon: Initial epsilon (backward compatibility)
 end_epsilon: Final epsilon (backward compatibility)
 decay_rate: Decay rate (backward compatibility)
 schedule_type: Type schedule
 custom_schedule: Custom schedule function
 """
 # Backward compatibility
 if config is None:
 config = EpsilonScheduleConfig(
 start_epsilon=start_epsilon,
 end_epsilon=end_epsilon,
 decay_rate=decay_rate,
 schedule_type=schedule_type
 )

 self.config = config
 self.current_step = 0

 # Creating corresponding schedule
 self.schedule = self._create_schedule(custom_schedule)

 self.logger = structlog.get_logger(__name__).bind(
 component="EpsilonScheduleFactory",
 schedule_type=config.schedule_type.value
 )

 self.logger.info("Epsilon schedule created", config=config.dict)

 def _create_schedule(self, custom_fn: Optional[Callable]) -> BaseEpsilonSchedule:
 """Creating specific schedule based on configuration."""
 if self.config.schedule_type == ScheduleType.LINEAR:
 return LinearSchedule(self.config)
 elif self.config.schedule_type == ScheduleType.EXPONENTIAL:
 return ExponentialSchedule(self.config)
 elif self.config.schedule_type == ScheduleType.COSINE:
 return CosineSchedule(self.config)
 elif self.config.schedule_type == ScheduleType.STEP:
 return StepSchedule(self.config)
 elif self.config.schedule_type == ScheduleType.POLYNOMIAL:
 return PolynomialSchedule(self.config)
 elif self.config.schedule_type == ScheduleType.CUSTOM:
 if custom_fn is None:
 raise ValueError("Custom schedule requires schedule function")
 return CustomSchedule(self.config, custom_fn)
 else:
 raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

 def get_epsilon(self, step: Optional[int] = None) -> float:
 """
 Get epsilon for given step.

 Args:
 step: Current step (if None, used internal counter)

 Returns:
 Epsilon value
 """
 if step is not None:
 self.current_step = step

 epsilon = self.schedule.get_epsilon(self.current_step)

 # Logging for monitoring (less often for performance)
 if self.current_step % 10000 == 0:
 self.logger.debug("Epsilon update",
 step=self.current_step,
 epsilon=epsilon)

 return epsilon

 def step(self) -> float:
 """Increment step and get epsilon."""
 self.current_step += 1
 return self.get_epsilon

 def reset(self) -> None:
 """Reset schedule to initial state."""
 self.current_step = 0
 self.schedule.reset
 self.logger.info("Epsilon schedule reset")

 def get_schedule_info(self) -> Dict[str, Any]:
 """Get information about schedule."""
 return {
 "schedule_type": self.config.schedule_type.value,
 "current_step": self.current_step,
 "current_epsilon": self.get_epsilon,
 "start_epsilon": self.config.start_epsilon,
 "end_epsilon": self.config.end_epsilon,
 "min_epsilon": self.config.min_epsilon,
 "config": self.config.dict,
 }

 def get_epsilon_trajectory(self, max_steps: int, step_size: int = 1000) -> Dict[str, List]:
 """
 Get trajectory epsilon for visualization.

 Args:
 max_steps: Maximum number of steps
 step_size: Step for sampling

 Returns:
 Dictionary with steps and epsilon values
 """
 steps = list(range(0, max_steps + 1, step_size))
 epsilons = [self.schedule.get_epsilon(step) for step in steps]

 return {"steps": steps, "epsilons": epsilons}

 def get_state(self) -> ScheduleState:
 """Get state for persistence."""
 return self.schedule.get_state

 def load_state(self, state: ScheduleState) -> None:
 """Load state from persistence."""
 self.current_step = state.current_step
 self.schedule.load_state(state)
 self.logger.info("State epsilon schedule loadedabout", step=self.current_step)

 @classmethod
 def create_linear(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01,
 total_steps: int = 100000) -> 'EpsilonSchedule':
 """Create linear schedule."""
 config = EpsilonScheduleConfig(
 start_epsilon=start_epsilon,
 end_epsilon=end_epsilon,
 total_steps=total_steps,
 schedule_type=ScheduleType.LINEAR
 )
 return cls(config=config)

 @classmethod
 def create_exponential(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01,
 decay_rate: float = 0.995) -> 'EpsilonSchedule':
 """Create exponential schedule."""
 config = EpsilonScheduleConfig(
 start_epsilon=start_epsilon,
 end_epsilon=end_epsilon,
 decay_rate=decay_rate,
 schedule_type=ScheduleType.EXPONENTIAL
 )
 return cls(config=config)

 @classmethod
 def create_cosine(cls, start_epsilon: float = 1.0, end_epsilon: float = 0.01,
 total_steps: int = 100000, restarts: bool = False) -> 'EpsilonSchedule':
 """Create cosine schedule."""
 config = EpsilonScheduleConfig(
 start_epsilon=start_epsilon,
 end_epsilon=end_epsilon,
 total_steps=total_steps,
 schedule_type=ScheduleType.COSINE,
 cosine_restarts=restarts
 )
 return cls(config=config)

 def __repr__(self) -> str:
 """String representation schedule."""
 return (
 f"EpsilonSchedule(type={self.config.schedule_type.value}, "
 f"step={self.current_step}, epsilon={self.get_epsilon:.4f})"
 )