"""
Parameter Scheduling Utilities for PPO
for adaptive training

Provides various scheduling strategies for:
- Learning rate scheduling
- Clipping parameter scheduling 
- KL coefficient scheduling
- Exploration parameter scheduling
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


@dataclass
class ScheduleConfig:
 """Configuration for parameter schedulers"""
 
 # Base parameters
 initial_value: float = 1.0
 final_value: float = 0.1
 
 # Schedule type specific
 schedule_type: str = "linear" # linear, exponential, cosine, polynomial
 total_steps: int = 1000000 # Total training steps
 
 # Polynomial schedule
 polynomial_power: float = 2.0
 
 # Exponential schedule 
 exponential_decay: float = 0.99
 
 # Step schedule
 step_size: int = 10000
 step_gamma: float = 0.5
 
 # Warmup
 warmup_steps: int = 0
 warmup_start_value: float = 0.0
 
 # Cyclical schedules
 cycle_length: int = 50000
 cycle_decay: float = 1.0
 
 # Adaptive scheduling
 patience: int = 10
 threshold: float = 1e-4
 cooldown: int = 5


class BaseSchedule(ABC):
 """Base class for parameter schedules"""
 
 def __init__(self, config: ScheduleConfig):
 self.config = config
 self.current_step = 0
 
 @abstractmethod
 def value(self, progress: float) -> float:
 """
 Get scheduled value
 
 Args:
 progress: Training progress [0.0, 1.0]
 
 Returns:
 Scheduled parameter value
 """
 pass
 
 def step(self) -> float:
 """Increment step and return current value"""
 self.current_step += 1
 progress = min(self.current_step / self.config.total_steps, 1.0)
 return self.value(progress)
 
 def reset(self):
 """Reset scheduler state"""
 self.current_step = 0


class LinearSchedule(BaseSchedule):
 """
 Linear parameter schedule
 
 value(t) = initial + (final - initial) * progress
 """
 
 def __init__(self, initial_value: float, final_value: float):
 config = ScheduleConfig(
 initial_value=initial_value,
 final_value=final_value,
 schedule_type="linear"
 )
 super().__init__(config)
 
 def value(self, progress: float) -> float:
 """Linear interpolation between initial and final values"""
 progress = np.clip(progress, 0.0, 1.0)
 
 return (
 self.config.initial_value + 
 (self.config.final_value - self.config.initial_value) * progress
 )


class ExponentialSchedule(BaseSchedule):
 """
 Exponential decay schedule
 
 value(t) = initial * decay^t
 """
 
 def __init__(
 self,
 initial_value: float,
 decay_rate: float = 0.99,
 min_value: Optional[float] = None
 ):
 config = ScheduleConfig(
 initial_value=initial_value,
 final_value=min_value or 0.0,
 exponential_decay=decay_rate,
 schedule_type="exponential"
 )
 super().__init__(config)
 self.min_value = min_value
 
 def value(self, progress: float) -> float:
 """Exponential decay"""
 steps = progress * self.config.total_steps
 
 decayed_value = (
 self.config.initial_value * 
 (self.config.exponential_decay ** steps)
 )
 
 if self.min_value is not None:
 decayed_value = max(decayed_value, self.min_value)
 
 return decayed_value


class CosineAnnealingSchedule(BaseSchedule):
 """
 Cosine annealing schedule
 
 value(t) = final + (initial - final) * 0.5 * (1 + cos(π * progress))
 """
 
 def __init__(
 self,
 initial_value: float,
 min_value: float = 0.0,
 restart_period: Optional[int] = None
 ):
 config = ScheduleConfig(
 initial_value=initial_value,
 final_value=min_value,
 schedule_type="cosine"
 )
 super().__init__(config)
 self.restart_period = restart_period
 
 def value(self, progress: float) -> float:
 """Cosine annealing"""
 progress = np.clip(progress, 0.0, 1.0)
 
 # Handle restarts
 if self.restart_period is not None:
 current_step = progress * self.config.total_steps
 cycle_progress = (current_step % self.restart_period) / self.restart_period
 progress = cycle_progress
 
 cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
 
 return (
 self.config.final_value + 
 (self.config.initial_value - self.config.final_value) * cosine_decay
 )


class PolynomialSchedule(BaseSchedule):
 """
 Polynomial decay schedule
 
 value(t) = (initial - final) * (1 - progress)^power + final
 """
 
 def __init__(
 self,
 initial_value: float,
 final_value: float,
 power: float = 2.0
 ):
 config = ScheduleConfig(
 initial_value=initial_value,
 final_value=final_value,
 polynomial_power=power,
 schedule_type="polynomial"
 )
 super().__init__(config)
 
 def value(self, progress: float) -> float:
 """Polynomial decay"""
 progress = np.clip(progress, 0.0, 1.0)
 
 decay_factor = (1 - progress) ** self.config.polynomial_power
 
 return (
 (self.config.initial_value - self.config.final_value) * decay_factor + 
 self.config.final_value
 )


class StepSchedule(BaseSchedule):
 """
 Step decay schedule
 
 Reduces value by factor every step_size steps
 """
 
 def __init__(
 self,
 initial_value: float,
 step_size: int,
 gamma: float = 0.1
 ):
 config = ScheduleConfig(
 initial_value=initial_value,
 step_size=step_size,
 step_gamma=gamma,
 schedule_type="step"
 )
 super().__init__(config)
 
 def value(self, progress: float) -> float:
 """Step decay"""
 current_step = progress * self.config.total_steps
 num_decays = int(current_step // self.config.step_size)
 
 return self.config.initial_value * (self.config.step_gamma ** num_decays)


class WarmupSchedule(BaseSchedule):
 """
 Warmup schedule followed by another schedule
 
 Linearly increases from warmup_start to initial_value over warmup_steps,
 then follows specified schedule
 """
 
 def __init__(
 self,
 warmup_steps: int,
 peak_lr: float,
 base_schedule: Optional[BaseSchedule] = None,
 warmup_start_value: float = 0.0
 ):
 config = ScheduleConfig(
 initial_value=peak_lr,
 warmup_steps=warmup_steps,
 warmup_start_value=warmup_start_value,
 schedule_type="warmup"
 )
 super().__init__(config)
 self.base_schedule = base_schedule
 
 def value(self, progress: float) -> float:
 """Warmup followed by base schedule"""
 current_step = progress * self.config.total_steps
 
 if current_step < self.config.warmup_steps:
 # Warmup phase
 warmup_progress = current_step / self.config.warmup_steps
 return (
 self.config.warmup_start_value + 
 (self.config.initial_value - self.config.warmup_start_value) * warmup_progress
 )
 else:
 # Main schedule phase
 if self.base_schedule is not None:
 # Adjust progress for base schedule
 remaining_steps = self.config.total_steps - self.config.warmup_steps
 adjusted_progress = (current_step - self.config.warmup_steps) / remaining_steps
 return self.base_schedule.value(adjusted_progress)
 else:
 return self.config.initial_value


class CyclicalSchedule(BaseSchedule):
 """
 Cyclical schedule (e.g., cyclical learning rates)
 
 Cycles between min and max values with optional decay
 """
 
 def __init__(
 self,
 min_value: float,
 max_value: float,
 cycle_length: int,
 decay_factor: float = 1.0,
 cycle_type: str = "triangular" # triangular, cosine
 ):
 config = ScheduleConfig(
 initial_value=max_value,
 final_value=min_value,
 cycle_length=cycle_length,
 cycle_decay=decay_factor,
 schedule_type="cyclical"
 )
 super().__init__(config)
 self.min_value = min_value
 self.max_value = max_value
 self.cycle_type = cycle_type
 
 def value(self, progress: float) -> float:
 """Cyclical value"""
 current_step = progress * self.config.total_steps
 
 # Current cycle
 cycle_num = int(current_step // self.config.cycle_length)
 cycle_progress = (current_step % self.config.cycle_length) / self.config.cycle_length
 
 # Apply decay
 cycle_amplitude = (self.max_value - self.min_value) * (self.config.cycle_decay ** cycle_num)
 
 if self.cycle_type == "triangular":
 # Triangular wave
 if cycle_progress <= 0.5:
 value = self.min_value + cycle_amplitude * (2 * cycle_progress)
 else:
 value = self.min_value + cycle_amplitude * (2 * (1 - cycle_progress))
 elif self.cycle_type == "cosine":
 # Cosine wave
 value = self.min_value + cycle_amplitude * 0.5 * (1 + math.cos(math.pi * cycle_progress))
 else:
 raise ValueError(f"Unknown cycle type: {self.cycle_type}")
 
 return value


class AdaptiveSchedule(BaseSchedule):
 """
 Adaptive schedule based on performance metrics
 
 Adjusts parameter based on improvement/degradation
 in performance metrics
 """
 
 def __init__(
 self,
 initial_value: float,
 patience: int = 10,
 threshold: float = 1e-4,
 factor: float = 0.5,
 cooldown: int = 5,
 min_value: Optional[float] = None,
 mode: str = "min" # min or max
 ):
 config = ScheduleConfig(
 initial_value=initial_value,
 patience=patience,
 threshold=threshold,
 cooldown=cooldown,
 schedule_type="adaptive"
 )
 super().__init__(config)
 
 self.factor = factor
 self.min_value = min_value or 0.0
 self.mode = mode
 
 # State tracking
 self.current_value = initial_value
 self.best_metric = None
 self.wait_counter = 0
 self.cooldown_counter = 0
 self.metric_history = []
 
 def value(self, progress: float) -> float:
 """Return current adaptive value"""
 return self.current_value
 
 def step_with_metric(self, metric: float) -> float:
 """Update schedule based on performance metric"""
 
 self.metric_history.append(metric)
 
 # Cooldown period
 if self.cooldown_counter > 0:
 self.cooldown_counter -= 1
 return self.current_value
 
 # Check for improvement
 improved = False
 if self.best_metric is None:
 self.best_metric = metric
 improved = True
 else:
 if self.mode == "min":
 if metric < self.best_metric - self.config.threshold:
 self.best_metric = metric
 improved = True
 else: # mode == "max"
 if metric > self.best_metric + self.config.threshold:
 self.best_metric = metric
 improved = True
 
 if improved:
 self.wait_counter = 0
 else:
 self.wait_counter += 1
 
 # Adjust parameter if no improvement
 if self.wait_counter >= self.config.patience:
 old_value = self.current_value
 self.current_value = max(self.min_value, self.current_value * self.factor)
 
 if self.current_value < old_value:
 self.wait_counter = 0
 self.cooldown_counter = self.config.cooldown
 
 return self.current_value


class MultiSchedule:
 """
 Combines multiple schedules for different parameters
 
 Useful for scheduling multiple hyperparameters simultaneously
 """
 
 def __init__(self, schedules: Dict[str, BaseSchedule]):
 self.schedules = schedules
 self.current_step = 0
 
 def value(self, progress: float) -> Dict[str, float]:
 """Get all scheduled values"""
 return {name: schedule.value(progress) for name, schedule in self.schedules.items()}
 
 def step(self) -> Dict[str, float]:
 """Step all schedules"""
 self.current_step += 1
 return {name: schedule.step() for name, schedule in self.schedules.items()}
 
 def reset(self):
 """Reset all schedules"""
 self.current_step = 0
 for schedule in self.schedules.values():
 schedule.reset()


class PPOScheduler:
 """
 Specialized scheduler for PPO hyperparameters
 
 Manages common PPO parameters:
 - Learning rate
 - Clipping range
 - KL coefficient
 - Entropy coefficient
 """
 
 def __init__(
 self,
 total_steps: int,
 lr_schedule: str = "linear",
 initial_lr: float = 3e-4,
 final_lr: float = 0.0,
 clip_schedule: str = "constant",
 initial_clip_range: float = 0.2,
 final_clip_range: float = 0.1,
 entropy_schedule: str = "constant",
 initial_entropy_coef: float = 0.01,
 final_entropy_coef: float = 0.001,
 kl_schedule: str = "constant",
 initial_kl_coef: float = 0.0,
 final_kl_coef: float = 0.0
 ):
 # Learning rate schedule
 if lr_schedule == "linear":
 self.lr_scheduler = LinearSchedule(initial_lr, final_lr)
 elif lr_schedule == "cosine":
 self.lr_scheduler = CosineAnnealingSchedule(initial_lr, final_lr)
 elif lr_schedule == "constant":
 self.lr_scheduler = LinearSchedule(initial_lr, initial_lr)
 else:
 raise ValueError(f"Unknown LR schedule: {lr_schedule}")
 
 # Clipping range schedule
 if clip_schedule == "linear":
 self.clip_scheduler = LinearSchedule(initial_clip_range, final_clip_range)
 elif clip_schedule == "constant":
 self.clip_scheduler = LinearSchedule(initial_clip_range, initial_clip_range)
 else:
 raise ValueError(f"Unknown clip schedule: {clip_schedule}")
 
 # Entropy coefficient schedule
 if entropy_schedule == "linear":
 self.entropy_scheduler = LinearSchedule(initial_entropy_coef, final_entropy_coef)
 elif entropy_schedule == "exponential":
 self.entropy_scheduler = ExponentialSchedule(initial_entropy_coef, 0.999, final_entropy_coef)
 elif entropy_schedule == "constant":
 self.entropy_scheduler = LinearSchedule(initial_entropy_coef, initial_entropy_coef)
 else:
 raise ValueError(f"Unknown entropy schedule: {entropy_schedule}")
 
 # KL coefficient schedule
 if kl_schedule == "adaptive":
 self.kl_scheduler = AdaptiveSchedule(initial_kl_coef, patience=5, factor=2.0)
 elif kl_schedule == "constant":
 self.kl_scheduler = LinearSchedule(initial_kl_coef, initial_kl_coef)
 else:
 raise ValueError(f"Unknown KL schedule: {kl_schedule}")
 
 self.total_steps = total_steps
 self.current_step = 0
 
 def get_values(self, progress: Optional[float] = None) -> Dict[str, float]:
 """Get all scheduled values"""
 
 if progress is None:
 progress = min(self.current_step / self.total_steps, 1.0)
 
 values = {
 "learning_rate": self.lr_scheduler.value(progress),
 "clip_range": self.clip_scheduler.value(progress),
 "entropy_coef": self.entropy_scheduler.value(progress)
 }
 
 # KL scheduler might be adaptive
 if isinstance(self.kl_scheduler, AdaptiveSchedule):
 values["kl_coef"] = self.kl_scheduler.value(progress)
 else:
 values["kl_coef"] = self.kl_scheduler.value(progress)
 
 return values
 
 def step(self, kl_divergence: Optional[float] = None) -> Dict[str, float]:
 """Step schedulers and return current values"""
 
 self.current_step += 1
 progress = min(self.current_step / self.total_steps, 1.0)
 
 # Regular schedulers
 values = {
 "learning_rate": self.lr_scheduler.value(progress),
 "clip_range": self.clip_scheduler.value(progress),
 "entropy_coef": self.entropy_scheduler.value(progress)
 }
 
 # Adaptive KL scheduler
 if isinstance(self.kl_scheduler, AdaptiveSchedule) and kl_divergence is not None:
 values["kl_coef"] = self.kl_scheduler.step_with_metric(kl_divergence)
 else:
 values["kl_coef"] = self.kl_scheduler.value(progress)
 
 return values


# Factory functions
def create_schedule(
 schedule_type: str,
 initial_value: float,
 final_value: Optional[float] = None,
 **kwargs
) -> BaseSchedule:
 """Create schedule of specified type"""
 
 if schedule_type == "linear":
 return LinearSchedule(initial_value, final_value or 0.0)
 elif schedule_type == "exponential":
 return ExponentialSchedule(initial_value, **kwargs)
 elif schedule_type == "cosine":
 return CosineAnnealingSchedule(initial_value, final_value or 0.0, **kwargs)
 elif schedule_type == "polynomial":
 return PolynomialSchedule(initial_value, final_value or 0.0, **kwargs)
 elif schedule_type == "step":
 return StepSchedule(initial_value, **kwargs)
 elif schedule_type == "warmup":
 return WarmupSchedule(initial_value, **kwargs)
 elif schedule_type == "cyclical":
 return CyclicalSchedule(final_value or 0.0, initial_value, **kwargs)
 elif schedule_type == "adaptive":
 return AdaptiveSchedule(initial_value, **kwargs)
 else:
 raise ValueError(f"Unknown schedule type: {schedule_type}")


# Export classes and functions
__all__ = [
 "ScheduleConfig",
 "BaseSchedule",
 "LinearSchedule",
 "ExponentialSchedule", 
 "CosineAnnealingSchedule",
 "PolynomialSchedule",
 "StepSchedule",
 "WarmupSchedule",
 "CyclicalSchedule",
 "AdaptiveSchedule",
 "MultiSchedule",
 "PPOScheduler",
 "create_schedule"
]