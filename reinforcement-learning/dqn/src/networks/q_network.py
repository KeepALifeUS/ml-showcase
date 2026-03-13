"""
Q-Network architecture for DQN with enterprise patterns.

Implements standard Q-network architecture with optimizations for crypto trading:
- Dropout for regularization
- Batch normalization for stability
- Residual connections for deep network
- Configurable architecture through Pydantic
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field, validator
import numpy as np

logger = logging.getLogger(__name__)


class QNetworkConfig(BaseModel):
 """Configuration Q-Network with validation."""

 state_size: int = Field(..., description="Dimensionality space states", gt=0)
 action_size: int = Field(..., description="Dimensionality space actions", gt=0)
 hidden_layers: List[int] = Field(
 default=[512, 256, 128],
 description="Sizes hidden layers"
 )
 dropout_rate: float = Field(default=0.2, description="Probability dropout", ge=0.0, le=0.8)
 use_batch_norm: bool = Field(default=True, description="Use batch normalization")
 use_residual: bool = Field(default=True, description="Use residual connections")
 activation: str = Field(default="relu", description="Function activation")
 output_activation: Optional[str] = Field(default=None, description="Activation output single layer")
 init_type: str = Field(default="xavier_uniform", description="Type initialization weights")

 @validator("hidden_layers")
 def validate_hidden_layers(cls, v):
 if not v or len(v) == 0:
 raise ValueError("Must be minimum one hidden number")
 if any(size <= 0 for size in v):
 raise ValueError("All sizes layers must be positive")
 return v

 @validator("activation")
 def validate_activation(cls, v):
 valid_activations = ["relu", "leaky_relu", "elu", "selu", "gelu", "swish"]
 if v not in valid_activations:
 raise ValueError(f"Activation must be one from: {valid_activations}")
 return v

 @validator("init_type")
 def validate_init_type(cls, v):
 valid_inits = ["xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "orthogonal"]
 if v not in valid_inits:
 raise ValueError(f"Type initialization must be one from: {valid_inits}")
 return v


class QNetwork(nn.Module):
 """
 Q-Network for DQN with enterprise-grade architecture.

 Features:
 - Configurable architecture through Pydantic config
 - Dropout regularization for prevention overfitting
 - Batch normalization for stability training
 - Residual connections for deep network
 - Proper weight initialization
 - Gradient clipping support
 - Performance monitoring hooks
 """

 def __init__(self, config: QNetworkConfig):
 """
 Initialization Q-Network.

 Args:
 config: Configuration network
 """
 super.__init__
 self.config = config
 self.state_size = config.state_size
 self.action_size = config.action_size

 # Building architecture
 self._build_network

 # Initialization weights
 self._initialize_weights

 # Monitoring hooks for production
 self._register_hooks

 logger.info(f"Created Q-Network: {self._get_network_info}")

 def _build_network(self) -> None:
 """Building architecture network."""
 layers = []
 layer_sizes = [self.config.state_size] + self.config.hidden_layers

 # Hidden layers
 for i in range(len(layer_sizes) - 1):
 in_size = layer_sizes[i]
 out_size = layer_sizes[i + 1]

 # Linear layer
 layers.append(nn.Linear(in_size, out_size))

 # Batch normalization
 if self.config.use_batch_norm:
 layers.append(nn.BatchNorm1d(out_size))

 # Activation
 layers.append(self._get_activation)

 # Dropout
 if self.config.dropout_rate > 0:
 layers.append(nn.Dropout(self.config.dropout_rate))

 # Main network
 self.feature_layers = nn.Sequential(*layers)

 # Output number
 self.output_layer = nn.Linear(self.config.hidden_layers[-1], self.config.action_size)

 # Output activation if needed
 self.output_activation = None
 if self.config.output_activation:
 self.output_activation = self._get_activation(self.config.output_activation)

 # Residual connections if enabled
 if self.config.use_residual:
 self._setup_residual_connections

 def _setup_residual_connections(self) -> None:
 """Setup residual connections."""
 self.residual_layers = nn.ModuleList
 layer_sizes = [self.config.state_size] + self.config.hidden_layers

 for i in range(len(layer_sizes) - 1):
 in_size = layer_sizes[i]
 out_size = layer_sizes[i + 1]

 # Skip connection only if sizes match
 if in_size == out_size:
 self.residual_layers.append(nn.Identity)
 else:
 # Projection layer for changes sizes
 self.residual_layers.append(nn.Linear(in_size, out_size))

 def _get_activation(self, activation: Optional[str] = None) -> nn.Module:
 """Get function activation."""
 act_name = activation or self.config.activation

 activation_map = {
 "relu": nn.ReLU(inplace=True),
 "leaky_relu": nn.LeakyReLU(0.01, inplace=True),
 "elu": nn.ELU(inplace=True),
 "selu": nn.SELU(inplace=True),
 "gelu": nn.GELU,
 "swish": nn.SiLU(inplace=True),
 }

 return activation_map[act_name]

 def _initialize_weights(self) -> None:
 """Initialization weights network."""
 def init_layer(m):
 if isinstance(m, nn.Linear):
 if self.config.init_type == "xavier_uniform":
 nn.init.xavier_uniform_(m.weight)
 elif self.config.init_type == "xavier_normal":
 nn.init.xavier_normal_(m.weight)
 elif self.config.init_type == "kaiming_uniform":
 nn.init.kaiming_uniform_(m.weight, nonlinearity=self.config.activation)
 elif self.config.init_type == "kaiming_normal":
 nn.init.kaiming_normal_(m.weight, nonlinearity=self.config.activation)
 elif self.config.init_type == "orthogonal":
 nn.init.orthogonal_(m.weight)

 # Initialization bias
 if m.bias is not None:
 nn.init.constant_(m.bias, 0.0)

 elif isinstance(m, nn.BatchNorm1d):
 nn.init.constant_(m.weight, 1.0)
 nn.init.constant_(m.bias, 0.0)

 self.apply(init_layer)

 # Special initialization output single layer
 nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)
 nn.init.constant_(self.output_layer.bias, 0.0)

 logger.debug(f"Weights initialized method: {self.config.init_type}")

 def _register_hooks(self) -> None:
 """Registration hooks for monitoring."""
 def forward_hook(module, input, output):
 # Check for NaN and Inf
 if torch.isnan(output).any:
 logger.error(f"NaN detected in output module {module.__class__.__name__}")
 if torch.isinf(output).any:
 logger.error(f"Inf detected in output module {module.__class__.__name__}")

 def backward_hook(module, grad_input, grad_output):
 # Check gradients
 if grad_output[0] is not None:
 grad_norm = torch.norm(grad_output[0])
 if grad_norm > 10.0: # Threshold for gradient explosion
 logger.warning(f"Large gradients in {module.__class__.__name__}: {grad_norm:.4f}")

 # Registration hooks only in debug mode
 if logger.isEnabledFor(logging.DEBUG):
 for name, module in self.named_modules:
 if isinstance(module, nn.Linear):
 module.register_forward_hook(forward_hook)
 module.register_backward_hook(backward_hook)

 def forward(self, state: torch.Tensor) -> torch.Tensor:
 """
 Forward pass through network.

 Args:
 state: Tensor states [batch_size, state_size]

 Returns:
 Q-values for all actions [batch_size, action_size]
 """
 if state.dim == 1:
 state = state.unsqueeze(0) # Add batch dimension

 x = state

 # Residual connections if enabled
 if self.config.use_residual and hasattr(self, 'residual_layers'):
 residuals = []
 layer_idx = 0

 for i, layer in enumerate(self.feature_layers):
 if isinstance(layer, nn.Linear):
 # Save input data for residual connection
 if layer_idx < len(self.residual_layers):
 residual = self.residual_layers[layer_idx](x)
 residuals.append(residual)
 layer_idx += 1

 x = layer(x)

 # Add residual connection after activation
 if len(residuals) > 0 and not isinstance(self.feature_layers[i + 1], nn.BatchNorm1d):
 x = x + residuals.pop(0)

 else:
 x = layer(x)
 else:
 # Standard forward pass
 x = self.feature_layers(x)

 # Output number
 q_values = self.output_layer(x)

 # Output activation if is
 if self.output_activation:
 q_values = self.output_activation(q_values)

 return q_values

 def get_action_values(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
 """
 Get Q-values for states.

 Args:
 state: States [batch_size, state_size]
 action: Actions (if None, returns all Q-values) [batch_size]

 Returns:
 Q-values [batch_size] or [batch_size, action_size]
 """
 q_values = self.forward(state)

 if action is not None:
 # Choose Q-values for specific actions
 if action.dim == 1:
 q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
 else:
 q_values = q_values.gather(1, action)

 return q_values

 def get_best_actions(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Get best actions and their Q-values.

 Args:
 state: States [batch_size, state_size]

 Returns:
 Tuple of (actions, q_values)
 """
 q_values = self.forward(state)
 actions = torch.argmax(q_values, dim=1)
 max_q_values = torch.max(q_values, dim=1)[0]

 return actions, max_q_values

 def soft_update(self, target_network: 'QNetwork', tau: float = 0.005) -> None:
 """
 Soft updating target network.

 Args:
 target_network: Target network for update
 tau: Coefficient interpolation
 """
 for target_param, local_param in zip(target_network.parameters, self.parameters):
 target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

 def hard_update(self, target_network: 'QNetwork') -> None:
 """
 Hard updating target network.

 Args:
 target_network: Target network for update
 """
 target_network.load_state_dict(self.state_dict)

 def clip_gradients(self, max_norm: float = 1.0) -> float:
 """
 Processing gradients for stability training.

 Args:
 max_norm: Maximum norm gradients

 Returns:
 Norm gradients to processing
 """
 return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm)

 def freeze_layers(self, layer_names: List[str]) -> None:
 """
 Freezing certain layers.

 Args:
 layer_names: List name layers for freezing
 """
 for name, param in self.named_parameters:
 if any(layer_name in name for layer_name in layer_names):
 param.requires_grad = False
 logger.info(f"Expensive number: {name}")

 def unfreeze_all(self) -> None:
 """Unfreezing all layers."""
 for param in self.parameters:
 param.requires_grad = True
 logger.info("All layers unfrozen")

 def get_network_stats(self) -> Dict[str, Any]:
 """
 Get statisticsat network.

 Returns:
 Dictionary with statistical
 """
 total_params = sum(p.numel for p in self.parameters)
 trainable_params = sum(p.numel for p in self.parameters if p.requires_grad)

 return {
 "total_parameters": total_params,
 "trainable_parameters": trainable_params,
 "frozen_parameters": total_params - trainable_params,
 "memory_mb": total_params * 4 / (1024 * 1024), # Approximately for float32
 "config": self.config.dict,
 }

 def _get_network_info(self) -> str:
 """Get brief information about network."""
 stats = self.get_network_stats
 return (
 f"State:{self.state_size}, Action:{self.action_size}, "
 f"Layers:{self.config.hidden_layers}, "
 f"Params:{stats['total_parameters']:,}, "
 f"Memory:{stats['memory_mb']:.1f}MB"
 )

 def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
 """
 Saving checkpoint model.

 Args:
 filepath: Path for saving
 metadata: Additional metadata
 """
 checkpoint = {
 "model_state_dict": self.state_dict,
 "config": self.config.dict,
 "network_stats": self.get_network_stats,
 "metadata": metadata or {},
 }

 torch.save(checkpoint, filepath)
 logger.info(f"Checkpoint saved: {filepath}")

 @classmethod
 def load_checkpoint(cls, filepath: str) -> Tuple['QNetwork', Dict]:
 """
 Loading model from checkpoint.

 Args:
 filepath: Path to checkpoint

 Returns:
 Tuple of (model, metadata)
 """
 checkpoint = torch.load(filepath, map_location="cpu")

 # Creating model with saved configuration
 config = QNetworkConfig(**checkpoint["config"])
 model = cls(config)

 # Loading weights
 model.load_state_dict(checkpoint["model_state_dict"])

 logger.info(f"Checkpoint loaded: {filepath}")

 return model, checkpoint.get("metadata", {})