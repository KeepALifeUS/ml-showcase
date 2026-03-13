"""
Noisy Linear Layers for parameter space exploration with .

Noisy Networks replace epsilon-greedy exploration parameter noise:
- Factorized Gaussian noise for efficient computation
- Independent Gaussian noise for maximum expressiveness
- Learnable noise parameters σ for adaptive exploration
- Automatic noise scheduling without manual epsilon decay
- Better exploration in high-dimensional action spaces
"""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pydantic import BaseModel, Field, validator
import structlog

logger = structlog.get_logger(__name__)


class NoisyLinearConfig(BaseModel):
 """Configuration Noisy Linear layer."""

 # Noise type
 noise_type: str = Field(default="factorized", description="Type noise (factorized/independent)")

 # Initial noise parameters
 sigma_init: float = Field(default=0.5, description="Initial sigma", gt=0, le=1.0)
 sigma_min: float = Field(default=0.1, description="Minimum sigma", gt=0, le=1.0)

 # Factorized noise parameters
 mu_init_range: float = Field(default=0.1, description="Range initialization mu", gt=0, le=1.0)

 # Training dynamics
 noise_annealing: bool = Field(default=False, description="Annealing noise in time training")
 annealing_rate: float = Field(default=0.999, description="Rate for noise annealing", ge=0.9, le=1.0)

 @validator("sigma_min")
 def validate_sigma_min(cls, v, values):
 if "sigma_init" in values and v >= values["sigma_init"]:
 raise ValueError("sigma_min must be < sigma_init")
 return v

 @validator("noise_type")
 def validate_noise_type(cls, v):
 valid_types = ["factorized", "independent"]
 if v not in valid_types:
 raise ValueError(f"Noise type must be one from: {valid_types}")
 return v


class NoisyLinear(nn.Module):
 """
 Noisy Linear layer with enterprise-grade implementation.

 Implements parameter space exploration through learnable noise:
 - Factorized Gaussian noise: O(sqrt(in_features + out_features)) parameters
 - Independent Gaussian noise: O(in_features * out_features) parameters
 - Learnable sigma parameters for adaptive exploration
 - Proper initialization for stable training
 - Noise annealing support

 Forward pass: y = (μ_w + σ_w * ε_w) * x + (μ_b + σ_b * ε_b)
 where ε - Gaussian noise, μ - mean parameters, σ - noise parameters
 """

 def __init__(self,
 in_features: int,
 out_features: int,
 config: Optional[NoisyLinearConfig] = None,
 bias: bool = True):
 """
 Initialization Noisy Linear layer.

 Args:
 in_features: Number of input features
 out_features: Number of output features
 config: Configuration noisy layer
 bias: Use bias
 """
 super.__init__

 if config is None:
 config = NoisyLinearConfig

 self.config = config
 self.in_features = in_features
 self.out_features = out_features
 self.use_bias = bias

 # Mean parameters (trainable)
 self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
 if bias:
 self.bias_mu = nn.Parameter(torch.Tensor(out_features))

 # Sigma parameters (trainable noise scaling)
 self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
 if bias:
 self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

 # Noise buffers (not trainable, updated each forward)
 if config.noise_type == "factorized":
 self.register_buffer('epsilon_input', torch.zeros(in_features))
 self.register_buffer('epsilon_output', torch.zeros(out_features))
 else: # independent
 self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
 if bias:
 self.register_buffer('epsilon_bias', torch.zeros(out_features))

 # Training statistics
 self.noise_updates = 0
 self.current_sigma_scale = 1.0

 # Initialize parameters
 self._initialize_parameters

 self.logger = structlog.get_logger(__name__).bind(
 component="NoisyLinear",
 in_features=in_features,
 out_features=out_features,
 noise_type=config.noise_type
 )

 self.logger.debug("Noisy Linear layer created")

 def _initialize_parameters(self) -> None:
 """Initialization parameters."""
 # Mean parameters - standard uniform initialization
 mu_range = self.config.mu_init_range / math.sqrt(self.in_features)
 self.weight_mu.data.uniform_(-mu_range, mu_range)

 if self.use_bias:
 self.bias_mu.data.uniform_(-mu_range, mu_range)

 # Sigma parameters - constant initialization
 sigma_init = self.config.sigma_init / math.sqrt(self.in_features)
 self.weight_sigma.data.fill_(sigma_init)

 if self.use_bias:
 self.bias_sigma.data.fill_(sigma_init)

 def _sample_noise(self) -> None:
 """Sampling new noise for current forward pass."""
 if self.config.noise_type == "factorized":
 # Factorized Gaussian noise
 epsilon_input = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
 epsilon_output = self._f(torch.randn(self.out_features, device=self.weight_mu.device))

 self.epsilon_input.copy_(epsilon_input)
 self.epsilon_output.copy_(epsilon_output)

 else: # independent
 # Independent Gaussian noise
 self.epsilon_weight.copy_(torch.randn_like(self.epsilon_weight))
 if self.use_bias:
 self.epsilon_bias.copy_(torch.randn_like(self.epsilon_bias))

 def _f(self, x: torch.Tensor) -> torch.Tensor:
 """
 Factorization function: f(x) = sgn(x) * sqrt(|x|)

 Reduces variance factorized noise.

 Args:
 x: Input tensor

 Returns:
 Factorized noise
 """
 return torch.sign(x) * torch.sqrt(torch.abs(x))

 def forward(self, input: torch.Tensor) -> torch.Tensor:
 """
 Forward pass with noisy parameters.

 Args:
 input: Input tensor [batch_size, in_features]

 Returns:
 Output tensor [batch_size, out_features]
 """
 # Sample new noise each forward pass
 if self.training:
 self._sample_noise
 self.noise_updates += 1

 # Noise annealing if enabled
 if self.config.noise_annealing:
 self.current_sigma_scale *= self.config.annealing_rate
 self.current_sigma_scale = max(
 self.current_sigma_scale,
 self.config.sigma_min / self.config.sigma_init
 )

 # Compute noisy weights and biases
 if self.config.noise_type == "factorized":
 # Factorized noise: ε_w = ε_i ⊗ ε_j
 epsilon_weight = torch.outer(self.epsilon_output, self.epsilon_input)

 weight = self.weight_mu + self.weight_sigma * epsilon_weight * self.current_sigma_scale

 if self.use_bias:
 bias = self.bias_mu + self.bias_sigma * self.epsilon_output * self.current_sigma_scale
 else:
 bias = None

 else: # independent
 weight = self.weight_mu + self.weight_sigma * self.epsilon_weight * self.current_sigma_scale

 if self.use_bias:
 bias = self.bias_mu + self.bias_sigma * self.epsilon_bias * self.current_sigma_scale
 else:
 bias = None

 # Standard linear transformation
 return F.linear(input, weight, bias)

 def forward_without_noise(self, input: torch.Tensor) -> torch.Tensor:
 """
 Forward pass without noise (deterministic).

 Useful for evaluation or analysis.

 Args:
 input: Input tensor

 Returns:
 Deterministic output
 """
 return F.linear(input, self.weight_mu, self.bias_mu if self.use_bias else None)

 def get_noise_statistics(self) -> dict:
 """Get statistics noise."""
 stats = {
 "noise_updates": self.noise_updates,
 "current_sigma_scale": self.current_sigma_scale,
 "weight_sigma_mean": self.weight_sigma.mean.item,
 "weight_sigma_std": self.weight_sigma.std.item,
 "weight_sigma_min": self.weight_sigma.min.item,
 "weight_sigma_max": self.weight_sigma.max.item,
 }

 if self.use_bias:
 stats.update({
 "bias_sigma_mean": self.bias_sigma.mean.item,
 "bias_sigma_std": self.bias_sigma.std.item,
 })

 # Noise magnitude analysis
 if self.training:
 if self.config.noise_type == "factorized":
 noise_magnitude = (self.epsilon_input.std * self.epsilon_output.std).item
 else:
 noise_magnitude = self.epsilon_weight.std.item

 stats["current_noise_magnitude"] = noise_magnitude

 return stats

 def reset_noise(self) -> None:
 """Reset noise for new episode."""
 if self.config.noise_type == "factorized":
 self.epsilon_input.zero_
 self.epsilon_output.zero_
 else:
 self.epsilon_weight.zero_
 if self.use_bias:
 self.epsilon_bias.zero_

 def disable_noise(self) -> None:
 """Disabling noise (evaluation mode)."""
 self.current_sigma_scale = 0.0

 def enable_noise(self) -> None:
 """Enable noise (training mode)."""
 base_scale = self.config.sigma_min / self.config.sigma_init
 if self.config.noise_annealing:
 self.current_sigma_scale = max(self.current_sigma_scale, base_scale)
 else:
 self.current_sigma_scale = 1.0

 def get_effective_weights(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
 """
 Get current effective weights (μ + σ * ε).

 Returns:
 Tuple of (weight, bias)
 """
 if not self.training:
 return self.weight_mu, self.bias_mu if self.use_bias else None

 if self.config.noise_type == "factorized":
 epsilon_weight = torch.outer(self.epsilon_output, self.epsilon_input)
 weight = self.weight_mu + self.weight_sigma * epsilon_weight * self.current_sigma_scale

 if self.use_bias:
 bias = self.bias_mu + self.bias_sigma * self.epsilon_output * self.current_sigma_scale
 else:
 bias = None
 else:
 weight = self.weight_mu + self.weight_sigma * self.epsilon_weight * self.current_sigma_scale

 if self.use_bias:
 bias = self.bias_mu + self.bias_sigma * self.epsilon_bias * self.current_sigma_scale
 else:
 bias = None

 return weight, bias

 def extra_repr(self) -> str:
 """Additional information for repr."""
 return (
 f'in_features={self.in_features}, out_features={self.out_features}, '
 f'bias={self.use_bias}, noise_type={self.config.noise_type}, '
 f'sigma_scale={self.current_sigma_scale:.3f}'
 )


class NoisyNetwork(nn.Module):
 """
 Utility for creation fully noisy networks.

 Replaces last several layers usual network on noisy layers
 for parameter space exploration.
 """

 def __init__(self,
 base_network: nn.Module,
 noisy_layer_indices: list,
 config: Optional[NoisyLinearConfig] = None):
 """
 Conversion existing network in noisy network.

 Args:
 base_network: Base network
 noisy_layer_indices: Indices layers for replacement on noisy
 config: Configuration noisy layers
 """
 super.__init__

 if config is None:
 config = NoisyLinearConfig

 self.config = config
 self.base_network = base_network
 self.noisy_layers = nn.ModuleDict

 # Replacement specified layers on noisy
 self._convert_to_noisy(noisy_layer_indices)

 self.logger = structlog.get_logger(__name__).bind(
 component="NoisyNetwork",
 noisy_layers=len(noisy_layer_indices)
 )

 self.logger.info("Noisy Network created")

 def _convert_to_noisy(self, layer_indices: list) -> None:
 """Conversion layers in noisy."""
 modules = list(self.base_network.named_modules)

 for idx in layer_indices:
 if idx >= len(modules):
 continue

 name, module = modules[idx]
 if isinstance(module, nn.Linear):
 # Creating noisy replacement
 noisy_layer = NoisyLinear(
 module.in_features,
 module.out_features,
 self.config,
 bias=module.bias is not None
 )

 # Copying weights
 noisy_layer.weight_mu.data.copy_(module.weight.data)
 if module.bias is not None:
 noisy_layer.bias_mu.data.copy_(module.bias.data)

 # Replacement in network
 self._set_module_by_name(self.base_network, name, noisy_layer)
 self.noisy_layers[name] = noisy_layer

 self.logger.debug(f"Replaced number {name} on noisy",
 shape=f"{module.in_features}x{module.out_features}")

 def _set_module_by_name(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
 """Replacement module by name."""
 if '.' not in name:
 setattr(model, name, new_module)
 else:
 parent_name, child_name = name.rsplit('.', 1)
 parent = self._get_module_by_name(model, parent_name)
 setattr(parent, child_name, new_module)

 def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
 """Getting module by name."""
 for part in name.split('.'):
 model = getattr(model, part)
 return model

 def forward(self, x: torch.Tensor) -> torch.Tensor:
 """Forward pass through noisy network."""
 return self.base_network(x)

 def sample_noise(self) -> None:
 """Forced sampling new noise for all noisy layers."""
 for layer in self.noisy_layers.values:
 layer._sample_noise

 def disable_noise(self) -> None:
 """Disabling noise in all noisy layers."""
 for layer in self.noisy_layers.values:
 layer.disable_noise

 def enable_noise(self) -> None:
 """Enable noise in all noisy layers."""
 for layer in self.noisy_layers.values:
 layer.enable_noise

 def get_noise_statistics(self) -> dict:
 """Aggregated statistics noise."""
 all_stats = {}
 for name, layer in self.noisy_layers.items:
 stats = layer.get_noise_statistics
 all_stats[name] = stats

 # Total statisticsand
 if all_stats:
 all_sigma_means = [stats["weight_sigma_mean"] for stats in all_stats.values]
 aggregate_stats = {
 "total_noisy_layers": len(self.noisy_layers),
 "avg_sigma_mean": np.mean(all_sigma_means),
 "sigma_mean_std": np.std(all_sigma_means),
 "min_sigma_scale": min(stats["current_sigma_scale"] for stats in all_stats.values),
 "max_sigma_scale": max(stats["current_sigma_scale"] for stats in all_stats.values),
 }
 all_stats["aggregate"] = aggregate_stats

 return all_stats