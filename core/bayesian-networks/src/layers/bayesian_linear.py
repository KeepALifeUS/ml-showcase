"""
Bayesian Linear Layer Implementation for Crypto Trading BNNs
Implements variational inference for weight uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BayesianLinearConfig:
    """Configuration for Bayesian Linear Layer"""
    in_features: int
    out_features: int
    prior_mean: float = 0.0
    prior_std: float = 1.0
    posterior_mu_init: float = 0.0
    posterior_rho_init: float = -3.0
    bias: bool = True
    local_reparam: bool = True
    
    # enterprise patterns
    enable_monitoring: bool = True
    cache_kl: bool = True
    numerical_stability_eps: float = 1e-8


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with weight uncertainty for crypto trading predictions
    Implements Bayes by Backprop with local reparameterization trick
    """
    
    def __init__(self, config: BayesianLinearConfig):
        super().__init__()
        self.config = config
        
        # Weight parameters (mean and log-std)
        self.weight_mu = nn.Parameter(
            torch.Tensor(config.out_features, config.in_features)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(config.out_features, config.in_features)
        )
        
        # Bias parameters
        if config.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(config.out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(config.out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # Prior distributions
        self.weight_prior = Normal(config.prior_mean, config.prior_std)
        if config.bias:
            self.bias_prior = Normal(config.prior_mean, config.prior_std)
        
        # Initialize parameters
        self.reset_parameters()
        
        # KL divergence cache for efficiency
        self._kl_cache = None
        
        # Monitoring metrics for 
        self.metrics = {
            'kl_divergence': [],
            'weight_uncertainty': [],
            'activation_stats': []
        }
    
    def reset_parameters(self):
        """Initialize parameters with proper scaling"""
        # Initialize weight mean with Xavier/He initialization
        fan_in = self.config.in_features
        fan_out = self.config.out_features
        
        # Xavier initialization for mean
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.weight_mu.data.normal_(self.config.posterior_mu_init, std)
        
        # Initialize log-std (rho)
        self.weight_rho.data.fill_(self.config.posterior_rho_init)
        
        if self.config.bias:
            self.bias_mu.data.zero_()
            self.bias_rho.data.fill_(self.config.posterior_rho_init)
    
    def forward(self, input: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with optional weight sampling
        
        Args:
            input: Input tensor [batch_size, in_features]
            sample: Whether to sample weights (training) or use mean (inference)
        
        Returns:
            Output tensor with uncertainty-aware predictions
        """
        if sample or self.training:
            # Sample weights using reparameterization trick
            weight = self._sample_weight()
            bias = self._sample_bias() if self.config.bias else None
        else:
            # Use mean weights for deterministic inference
            weight = self.weight_mu
            bias = self.bias_mu if self.config.bias else None
        
        # Compute output
        if self.config.local_reparam and sample:
            # Local reparameterization for reduced variance
            output = self._local_reparam_forward(input, weight, bias)
        else:
            output = F.linear(input, weight, bias)
        
        # Update monitoring metrics
        if self.config.enable_monitoring:
            self._update_metrics(weight, output)
        
        return output
    
    def _sample_weight(self) -> torch.Tensor:
        """Sample weight from posterior distribution"""
        weight_std = torch.log1p(torch.exp(self.weight_rho)) + self.config.numerical_stability_eps
        weight_eps = torch.randn_like(self.weight_mu)
        return self.weight_mu + weight_std * weight_eps
    
    def _sample_bias(self) -> Optional[torch.Tensor]:
        """Sample bias from posterior distribution"""
        if not self.config.bias:
            return None
        bias_std = torch.log1p(torch.exp(self.bias_rho)) + self.config.numerical_stability_eps
        bias_eps = torch.randn_like(self.bias_mu)
        return self.bias_mu + bias_std * bias_eps
    
    def _local_reparam_forward(
        self, 
        input: torch.Tensor, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Local reparameterization trick for reduced gradient variance
        Samples activations directly instead of weights
        """
        # Compute mean and variance of pre-activations
        mean = F.linear(input, self.weight_mu, self.bias_mu if self.config.bias else None)
        
        # Compute variance using weight uncertainty
        weight_var = torch.log1p(torch.exp(self.weight_rho)) ** 2
        var = F.linear(input ** 2, weight_var, None)
        
        # Add bias variance if applicable
        if self.config.bias:
            bias_var = torch.log1p(torch.exp(self.bias_rho)) ** 2
            var = var + bias_var
        
        # Sample from the distribution
        std = torch.sqrt(var + self.config.numerical_stability_eps)
        eps = torch.randn_like(mean)
        
        return mean + std * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior
        Used as regularization term in ELBO
        """
        if self.config.cache_kl and self._kl_cache is not None:
            return self._kl_cache
        
        # Weight KL divergence
        weight_posterior = Normal(
            self.weight_mu, 
            torch.log1p(torch.exp(self.weight_rho)) + self.config.numerical_stability_eps
        )
        weight_kl = kl_divergence(weight_posterior, self.weight_prior).sum()
        
        # Bias KL divergence
        bias_kl = 0
        if self.config.bias:
            bias_posterior = Normal(
                self.bias_mu,
                torch.log1p(torch.exp(self.bias_rho)) + self.config.numerical_stability_eps
            )
            bias_kl = kl_divergence(bias_posterior, self.bias_prior).sum()
        
        total_kl = weight_kl + bias_kl
        
        if self.config.cache_kl:
            self._kl_cache = total_kl
        
        return total_kl
    
    def _update_metrics(self, weight: torch.Tensor, output: torch.Tensor):
        """Update monitoring metrics for observability"""
        with torch.no_grad():
            # Weight uncertainty (std)
            weight_std = torch.log1p(torch.exp(self.weight_rho)).mean().item()
            self.metrics['weight_uncertainty'].append(weight_std)
            
            # KL divergence
            kl = self.kl_divergence().item()
            self.metrics['kl_divergence'].append(kl)
            
            # Activation statistics
            self.metrics['activation_stats'].append({
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item()
            })
    
    def get_weight_samples(self, n_samples: int = 100) -> torch.Tensor:
        """
        Generate multiple weight samples for uncertainty estimation
        
        Args:
            n_samples: Number of weight samples to generate
        
        Returns:
            Tensor of shape [n_samples, out_features, in_features]
        """
        samples = []
        for _ in range(n_samples):
            samples.append(self._sample_weight())
        return torch.stack(samples)
    
    def freeze_posterior(self):
        """Freeze posterior parameters for transfer learning"""
        self.weight_mu.requires_grad = False
        self.weight_rho.requires_grad = False
        if self.config.bias:
            self.bias_mu.requires_grad = False
            self.bias_rho.requires_grad = False
    
    def unfreeze_posterior(self):
        """Unfreeze posterior parameters"""
        self.weight_mu.requires_grad = True
        self.weight_rho.requires_grad = True
        if self.config.bias:
            self.bias_mu.requires_grad = True
            self.bias_rho.requires_grad = True
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (
            f'in_features={self.config.in_features}, '
            f'out_features={self.config.out_features}, '
            f'bias={self.config.bias}, '
            f'local_reparam={self.config.local_reparam}'
        )


class BayesianLinearWithDropout(BayesianLinear):
    """
    Enhanced Bayesian Linear layer with additional dropout for double uncertainty
    Combines epistemic uncertainty from BNN with aleatoric uncertainty from dropout
    """
    
    def __init__(self, config: BayesianLinearConfig, dropout_rate: float = 0.1):
        super().__init__(config)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
    
    def forward(self, input: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass with Bayesian weights and dropout"""
        # Apply Bayesian linear transformation
        output = super().forward(input, sample)
        
        # Apply dropout for additional uncertainty
        if self.training or sample:
            output = self.dropout(output)
        
        return output


def create_bayesian_mlp(
    input_dim: int,
    hidden_dims: list,
    output_dim: int,
    activation: str = 'relu',
    dropout_rate: float = 0.0
) -> nn.Module:
    """
    Factory function to create Bayesian MLP for crypto trading
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function ('relu', 'tanh', 'gelu')
        dropout_rate: Dropout rate for additional uncertainty
    
    Returns:
        Bayesian MLP model
    """
    layers = []
    prev_dim = input_dim
    
    # Hidden layers
    for hidden_dim in hidden_dims:
        config = BayesianLinearConfig(
            in_features=prev_dim,
            out_features=hidden_dim,
            bias=True
        )
        
        if dropout_rate > 0:
            layers.append(BayesianLinearWithDropout(config, dropout_rate))
        else:
            layers.append(BayesianLinear(config))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        prev_dim = hidden_dim
    
    # Output layer
    output_config = BayesianLinearConfig(
        in_features=prev_dim,
        out_features=output_dim,
        bias=True
    )
    layers.append(BayesianLinear(output_config))
    
    return nn.Sequential(*layers)