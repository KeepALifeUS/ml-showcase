"""
Bayesian Neural Network for Time Series Prediction in Crypto Trading
Specialized for price prediction with uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import logging

from ..layers.bayesian_linear import BayesianLinear, BayesianLinearConfig

logger = logging.getLogger(__name__)


@dataclass
class BNNTimeSeriesConfig:
    """Configuration for Time Series BNN"""
    input_features: int  # Number of input features
    sequence_length: int  # Length of input sequence
    hidden_dims: List[int] = None  # Hidden layer dimensions
    output_horizon: int = 1  # Prediction horizon
    
    # Architecture choices
    use_lstm: bool = True
    use_attention: bool = True
    dropout_rate: float = 0.1
    
    # Bayesian configuration
    prior_std: float = 1.0
    posterior_init_std: float = 0.1
    local_reparam: bool = True
    
    # enterprise patterns
    enable_residual: bool = True
    enable_layer_norm: bool = True
    enable_multi_scale: bool = True  # Multi-scale temporal features
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class BayesianLSTMCell(nn.Module):
    """Bayesian LSTM cell with weight uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int, config: BayesianLinearConfig):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input to hidden
        self.input_transform = BayesianLinear(
            BayesianLinearConfig(
                in_features=input_size,
                out_features=4 * hidden_size,
                prior_std=config.prior_std,
                local_reparam=config.local_reparam
            )
        )
        
        # Hidden to hidden
        self.hidden_transform = BayesianLinear(
            BayesianLinearConfig(
                in_features=hidden_size,
                out_features=4 * hidden_size,
                prior_std=config.prior_std,
                local_reparam=config.local_reparam
            )
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(
        self, 
        input: torch.Tensor, 
        hidden: Tuple[torch.Tensor, torch.Tensor],
        sample: bool = True
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through Bayesian LSTM cell
        
        Args:
            input: Input tensor [batch_size, input_size]
            hidden: Tuple of (h, c) hidden states
            sample: Whether to sample weights
        
        Returns:
            Output and new hidden states
        """
        h_prev, c_prev = hidden
        
        # Compute gates
        gates = self.input_transform(input, sample) + self.hidden_transform(h_prev, sample)
        
        # Split into individual gates
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)
        
        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # Update cell state
        c_new = forget_gate * c_prev + input_gate * cell_gate
        
        # Compute output
        h_new = output_gate * torch.tanh(self.layer_norm(c_new))
        
        return h_new, (h_new, c_new)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for the cell"""
        return self.input_transform.kl_divergence() + self.hidden_transform.kl_divergence()


class TemporalAttention(nn.Module):
    """Self-attention mechanism for temporal dependencies"""
    
    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal attention
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            mask: Optional attention mask
        
        Returns:
            Attended output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_dim)
        output = self.output(context)
        
        return output


class BNNTimeSeriesModel(nn.Module):
    """
    Bayesian Neural Network for Crypto Price Time Series Prediction
    Combines LSTM, Attention, and Bayesian layers for uncertainty-aware forecasting
    """
    
    def __init__(self, config: BNNTimeSeriesConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(
            config.input_features,
            config.hidden_dims[0]
        )
        
        # Bayesian LSTM layers
        if config.use_lstm:
            bayesian_config = BayesianLinearConfig(
                in_features=0,  # Will be set in LSTM cell
                out_features=0,  # Will be set in LSTM cell
                prior_std=config.prior_std,
                local_reparam=config.local_reparam
            )
            
            self.lstm_cells = nn.ModuleList([
                BayesianLSTMCell(
                    config.hidden_dims[0] if i == 0 else config.hidden_dims[i-1],
                    config.hidden_dims[i],
                    bayesian_config
                )
                for i in range(len(config.hidden_dims))
            ])
        
        # Temporal attention
        if config.use_attention:
            self.attention = TemporalAttention(config.hidden_dims[-1])
        
        # Multi-scale temporal features
        if config.enable_multi_scale:
            self.multi_scale_conv = nn.ModuleList([
                nn.Conv1d(
                    config.hidden_dims[-1],
                    config.hidden_dims[-1] // 3,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in [3, 5, 7]
            ])
        
        # Bayesian output layers
        final_hidden_dim = config.hidden_dims[-1]
        if config.enable_multi_scale:
            final_hidden_dim = config.hidden_dims[-1] + (config.hidden_dims[-1] // 3) * 3
        
        self.output_layers = nn.ModuleList([
            BayesianLinear(
                BayesianLinearConfig(
                    in_features=final_hidden_dim if i == 0 else config.hidden_dims[-1] // 2,
                    out_features=config.hidden_dims[-1] // 2 if i < 2 else config.output_horizon,
                    prior_std=config.prior_std,
                    local_reparam=config.local_reparam
                )
            )
            for i in range(3)  # 3-layer output head
        ])
        
        # Residual connections
        if config.enable_residual:
            self.residual_projection = nn.Linear(
                config.input_features,
                config.output_horizon
            )
        
        # Layer normalization
        if config.enable_layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(dim) for dim in config.hidden_dims
            ])
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Output activation
        self.output_activation = nn.Identity()  # Can be changed to Tanh() for bounded outputs
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through BNN time series model
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_features]
            sample: Whether to sample weights (training) or use mean (inference)
        
        Returns:
            Predictions with shape [batch_size, output_horizon]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        h = self.input_projection(x)
        h = self.dropout(h)
        
        # Process through Bayesian LSTM
        if self.config.use_lstm:
            lstm_outputs = []
            
            for t in range(seq_len):
                h_t = h[:, t, :]
                
                # Initialize hidden states for first timestep
                if t == 0:
                    hidden_states = [
                        (torch.zeros(batch_size, dim).to(x.device),
                         torch.zeros(batch_size, dim).to(x.device))
                        for dim in self.config.hidden_dims
                    ]
                
                # Process through LSTM layers
                for i, lstm_cell in enumerate(self.lstm_cells):
                    h_t, hidden_states[i] = lstm_cell(h_t, hidden_states[i], sample)
                    
                    if self.config.enable_layer_norm:
                        h_t = self.layer_norms[i](h_t)
                    
                    h_t = self.dropout(h_t)
                
                lstm_outputs.append(h_t)
            
            # Stack LSTM outputs
            h = torch.stack(lstm_outputs, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Apply temporal attention
        if self.config.use_attention:
            h_attention = self.attention(h)
            h = h + h_attention  # Residual connection
        
        # Take last hidden state for prediction
        h_final = h[:, -1, :]  # [batch, hidden_dim]
        
        # Multi-scale temporal features
        if self.config.enable_multi_scale:
            # Transpose for conv1d
            h_conv = h.transpose(1, 2)  # [batch, hidden_dim, seq_len]
            
            multi_scale_features = []
            for conv in self.multi_scale_conv:
                feature = conv(h_conv)
                feature = F.relu(feature)
                # Global average pooling
                feature = feature.mean(dim=-1)  # [batch, hidden_dim//3]
                multi_scale_features.append(feature)
            
            # Concatenate multi-scale features
            multi_scale = torch.cat(multi_scale_features, dim=1)
            h_final = torch.cat([h_final, multi_scale], dim=1)
        
        # Process through Bayesian output layers
        out = h_final
        for i, layer in enumerate(self.output_layers):
            out = layer(out, sample)
            if i < len(self.output_layers) - 1:
                out = F.relu(out)
                out = self.dropout(out)
        
        # Residual connection from input
        if self.config.enable_residual:
            # Use last input values as residual
            residual = self.residual_projection(x[:, -1, :])
            out = out + residual
        
        # Apply output activation
        out = self.output_activation(out)
        
        return out
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all Bayesian layers"""
        kl = 0
        
        # LSTM KL
        if self.config.use_lstm:
            for lstm_cell in self.lstm_cells:
                kl += lstm_cell.kl_divergence()
        
        # Output layers KL
        for layer in self.output_layers:
            kl += layer.kl_divergence()
        
        return kl
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_features]
            n_samples: Number of forward passes for uncertainty estimation
        
        Returns:
            Tuple of (mean_predictions, std_predictions, all_samples)
        """
        self.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                samples.append(pred)
        
        samples = torch.stack(samples)  # [n_samples, batch_size, output_horizon]
        
        mean_pred = samples.mean(dim=0)
        std_pred = samples.std(dim=0)
        
        # Calculate percentiles for prediction intervals
        lower_percentile = torch.quantile(samples, 0.025, dim=0)
        upper_percentile = torch.quantile(samples, 0.975, dim=0)
        
        return mean_pred, std_pred, (lower_percentile, upper_percentile)
    
    def forecast(
        self,
        x: torch.Tensor,
        n_steps: int,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-step ahead forecasting with uncertainty
        
        Args:
            x: Initial input sequence [batch_size, sequence_length, input_features]
            n_steps: Number of steps to forecast
            n_samples: Number of samples for uncertainty
        
        Returns:
            Tuple of (forecasts, uncertainties)
        """
        self.eval()
        forecasts = []
        uncertainties = []
        
        current_input = x.clone()
        
        for step in range(n_steps):
            # Get prediction with uncertainty
            mean_pred, std_pred, _ = self.predict_with_uncertainty(
                current_input, n_samples
            )
            
            forecasts.append(mean_pred)
            uncertainties.append(std_pred)
            
            # Update input for next step (assuming autoregressive)
            if self.config.output_horizon == 1:
                # Shift sequence and append prediction
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    mean_pred.unsqueeze(1).unsqueeze(2).expand(
                        -1, -1, self.config.input_features
                    )
                ], dim=1)
        
        forecasts = torch.stack(forecasts, dim=1)  # [batch, n_steps, output_horizon]
        uncertainties = torch.stack(uncertainties, dim=1)
        
        return forecasts, uncertainties


def create_crypto_price_bnn(
    input_features: int = 10,
    sequence_length: int = 60,
    output_horizon: int = 1,
    hidden_dims: Optional[List[int]] = None
) -> BNNTimeSeriesModel:
    """
    Factory function to create BNN for crypto price prediction
    
    Args:
        input_features: Number of input features (OHLCV + indicators)
        sequence_length: Length of input sequence
        output_horizon: Prediction horizon
        hidden_dims: Hidden layer dimensions
    
    Returns:
        Configured BNN time series model
    """
    config = BNNTimeSeriesConfig(
        input_features=input_features,
        sequence_length=sequence_length,
        hidden_dims=hidden_dims or [256, 128, 64],
        output_horizon=output_horizon,
        use_lstm=True,
        use_attention=True,
        enable_multi_scale=True,
        dropout_rate=0.1,
        prior_std=1.0
    )
    
    return BNNTimeSeriesModel(config)