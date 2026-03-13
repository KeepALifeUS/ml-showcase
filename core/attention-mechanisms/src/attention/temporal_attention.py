"""
Temporal Attention mechanism for analysis of temporal dependencies in crypto trading.
Specialized for processing time series with various temporal patterns.

Production-optimized temporal attention for real-time trading decisions.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass
import logging

from .multi_head_attention import MultiHeadAttention, AttentionConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalAttentionConfig(AttentionConfig):
    """Configuration for Temporal Attention mechanism."""
    # Temporal-specific parameters
    use_time_embeddings: bool = True  # Time-based embeddings
    time_embedding_dim: int = 64
    max_time_steps: int = 10000  # Maximum number of time steps
    
    # Multi-timeframe attention
    use_multi_timeframe: bool = True
    timeframe_windows: List[int] = None  # [1, 5, 15, 60, 240] minutes
    timeframe_weights: Optional[List[float]] = None
    
    # Temporal decay
    use_temporal_decay: bool = True
    decay_factor: float = 0.95  # Exponential decay for older timesteps
    decay_type: str = "exponential"  # "exponential", "linear", "learned"
    
    # Cyclical patterns
    use_cyclical_attention: bool = True
    daily_cycles: int = 24  # Hours in day
    weekly_cycles: int = 7   # Days in week
    monthly_cycles: int = 30 # Days in month
    
    # Trend awareness
    use_trend_attention: bool = True
    trend_window_size: int = 20  # Window for trend calculation
    
    # Seasonality
    use_seasonal_patterns: bool = True
    seasonal_periods: List[int] = None  # [24, 168, 720] hours
    
    def __post_init__(self):
        super().__post_init__()
        if self.timeframe_windows is None:
            self.timeframe_windows = [1, 5, 15, 60, 240]  # Standard crypto timeframes
        
        if self.timeframe_weights is None:
            self.timeframe_weights = [1.0] * len(self.timeframe_windows)
        
        if self.seasonal_periods is None:
            self.seasonal_periods = [24, 168, 720]  # 1 day, 1 week, 1 month in hours


class TemporalPositionalEncoding(nn.Module):
    """Temporal positional encoding with support for various temporal patterns."""
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        self.config = config
        
        # Standard sinusoidal encoding
        self.register_buffer(
            'positional_encoding',
            self._create_sinusoidal_encoding(config.max_time_steps, config.d_model)
        )
        
        # Learnable time embeddings
        if config.use_time_embeddings:
            self.time_embedding = nn.Embedding(config.max_time_steps, config.time_embedding_dim)
            self.time_proj = nn.Linear(config.time_embedding_dim, config.d_model)
        
        # Cyclical encodings
        if config.use_cyclical_attention:
            self.daily_encoding = self._create_cyclical_encoding(config.daily_cycles)
            self.weekly_encoding = self._create_cyclical_encoding(config.weekly_cycles)
            self.monthly_encoding = self._create_cyclical_encoding(config.monthly_cycles)
            
            # Projections for cyclical encodings
            self.cyclical_proj = nn.Linear(3 * 2, config.d_model)  # 3 cycles * 2 (sin, cos)
        
        # Seasonal encodings
        if config.use_seasonal_patterns:
            self.seasonal_projections = nn.ModuleDict({
                f'period_{period}': nn.Linear(2, config.d_model // len(config.seasonal_periods))
                for period in config.seasonal_periods
            })
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _create_cyclical_encoding(self, cycle_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create cyclical encoding (sin, cos) for a given cycle."""
        positions = torch.arange(cycle_length, dtype=torch.float)
        angles = 2 * math.pi * positions / cycle_length
        return torch.sin(angles), torch.cos(angles)
    
    def forward(
        self,
        sequence_length: int,
        timestamps: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply temporal positional encoding.
        
        Args:
            sequence_length: Length of the sequence
            timestamps: Actual timestamps [batch_size, seq_len] (Unix timestamps)
            time_indices: Time indices [batch_size, seq_len] (0 to max_time_steps-1)
        """
        batch_size = timestamps.shape[0] if timestamps is not None else 1
        device = timestamps.device if timestamps is not None else 'cpu'
        
        # Base positional encoding
        pos_encoding = self.positional_encoding[:sequence_length].to(device)
        pos_encoding = pos_encoding.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Time embeddings
        if self.config.use_time_embeddings and time_indices is not None:
            time_emb = self.time_embedding(time_indices)
            time_features = self.time_proj(time_emb)
            pos_encoding = pos_encoding + time_features
        
        # Cyclical patterns
        if self.config.use_cyclical_attention and timestamps is not None:
            cyclical_features = self._compute_cyclical_features(timestamps)
            cyclical_encoding = self.cyclical_proj(cyclical_features)
            pos_encoding = pos_encoding + cyclical_encoding
        
        # Seasonal patterns
        if self.config.use_seasonal_patterns and timestamps is not None:
            seasonal_features = self._compute_seasonal_features(timestamps)
            pos_encoding = pos_encoding + seasonal_features
        
        return pos_encoding
    
    def _compute_cyclical_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute cyclical features from timestamps."""
        # Convert Unix timestamps to hours, days, etc.
        hours = (timestamps / 3600) % self.config.daily_cycles
        days = (timestamps / (3600 * 24)) % self.config.weekly_cycles
        month_days = (timestamps / (3600 * 24)) % self.config.monthly_cycles
        
        # Cyclical encoding
        hour_sin = torch.sin(2 * math.pi * hours / self.config.daily_cycles)
        hour_cos = torch.cos(2 * math.pi * hours / self.config.daily_cycles)
        
        day_sin = torch.sin(2 * math.pi * days / self.config.weekly_cycles)
        day_cos = torch.cos(2 * math.pi * days / self.config.weekly_cycles)
        
        month_sin = torch.sin(2 * math.pi * month_days / self.config.monthly_cycles)
        month_cos = torch.cos(2 * math.pi * month_days / self.config.monthly_cycles)
        
        return torch.stack([hour_sin, hour_cos, day_sin, day_cos, month_sin, month_cos], dim=-1)
    
    def _compute_seasonal_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute seasonal features."""
        seasonal_outputs = []
        
        for period in self.config.seasonal_periods:
            period_phase = (timestamps / (3600 * period)) % 1
            sin_component = torch.sin(2 * math.pi * period_phase).unsqueeze(-1)
            cos_component = torch.cos(2 * math.pi * period_phase).unsqueeze(-1)
            
            seasonal_feature = torch.cat([sin_component, cos_component], dim=-1)
            projected_feature = self.seasonal_projections[f'period_{period}'](seasonal_feature)
            seasonal_outputs.append(projected_feature)
        
        return torch.cat(seasonal_outputs, dim=-1)


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism for temporal series in crypto trading.
    
    Features:
    - Multi-timeframe attention
    - Temporal decay for older observations
    - Cyclical pattern awareness (daily, weekly, monthly)
    - Trend-aware attention weights
    - Seasonal pattern recognition
    - Real-time inference optimization
    """
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__()
        self.config = config
        
        # Temporal positional encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(config)
        
        # Multi-timeframe attention heads
        if config.use_multi_timeframe:
            self.timeframe_attentions = nn.ModuleDict({
                f'tf_{window}': MultiHeadAttention(config)
                for window in config.timeframe_windows
            })
            
            # Learnable timeframe weights
            self.timeframe_weight_params = nn.Parameter(
                torch.tensor(config.timeframe_weights, dtype=torch.float)
            )
        else:
            self.base_attention = MultiHeadAttention(config)
        
        # Temporal decay weights
        if config.use_temporal_decay:
            if config.decay_type == "learned":
                self.decay_weights = nn.Parameter(torch.ones(config.max_seq_len))
            else:
                self.register_buffer(
                    'decay_weights',
                    self._create_decay_weights(config.max_seq_len, config.decay_factor, config.decay_type)
                )
        
        # Trend attention components
        if config.use_trend_attention:
            self.trend_detector = nn.Conv1d(
                in_channels=config.d_model,
                out_channels=config.d_model,
                kernel_size=config.trend_window_size,
                padding=config.trend_window_size // 2
            )
            self.trend_attention_weights = nn.Linear(config.d_model, 1)
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def _create_decay_weights(
        self, 
        max_len: int, 
        decay_factor: float, 
        decay_type: str
    ) -> torch.Tensor:
        """Create temporal decay weights."""
        positions = torch.arange(max_len, dtype=torch.float)
        
        if decay_type == "exponential":
            weights = decay_factor ** positions
        elif decay_type == "linear":
            weights = 1.0 - (positions / max_len) * (1 - decay_factor)
        else:
            weights = torch.ones(max_len)
        
        return weights.flip(0)  # Recent timesteps get higher weights
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        time_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Temporal attention forward pass.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
            timestamps: Unix timestamps [batch_size, seq_len]
            time_indices: Relative time indices [batch_size, seq_len]
            attention_mask: Attention mask
            need_weights: Return attention weights
        """
        batch_size, seq_len, d_model = x.shape
        residual = x
        
        # Apply temporal positional encoding
        pos_encoding = self.temporal_pos_encoding(seq_len, timestamps, time_indices)
        x_with_time = x + pos_encoding
        
        # Apply temporal decay
        if self.config.use_temporal_decay:
            decay_weights = self.decay_weights[:seq_len].to(x.device)
            x_with_time = x_with_time * decay_weights.view(1, -1, 1)
        
        # Multi-timeframe attention
        if self.config.use_multi_timeframe:
            timeframe_outputs = []
            attention_weights_dict = {}
            
            # Normalize timeframe weights
            tf_weights = F.softmax(self.timeframe_weight_params, dim=0)
            
            for i, window in enumerate(self.config.timeframe_windows):
                tf_attention = self.timeframe_attentions[f'tf_{window}']
                
                # Create timeframe-specific mask
                tf_mask = self._create_timeframe_mask(seq_len, window, x.device)
                combined_mask = tf_mask
                if attention_mask is not None:
                    combined_mask = tf_mask & attention_mask
                
                # Apply attention
                tf_output = tf_attention(
                    query=x_with_time,
                    attention_mask=combined_mask,
                    need_weights=need_weights
                )
                
                if need_weights:
                    tf_output, tf_attention_weights = tf_output
                    attention_weights_dict[f'tf_{window}'] = tf_attention_weights
                
                # Weight by timeframe importance
                weighted_output = tf_output * tf_weights[i]
                timeframe_outputs.append(weighted_output)
            
            # Combine timeframe outputs
            combined_output = torch.stack(timeframe_outputs, dim=0).sum(dim=0)
            
        else:
            # Single timeframe attention
            combined_output = self.base_attention(
                query=x_with_time,
                attention_mask=attention_mask,
                need_weights=need_weights
            )
            
            if need_weights:
                combined_output, attention_weights_dict = combined_output
        
        # Trend-aware weighting
        if self.config.use_trend_attention:
            trend_features = self.trend_detector(
                combined_output.transpose(1, 2)
            ).transpose(1, 2)
            
            trend_weights = torch.sigmoid(self.trend_attention_weights(trend_features))
            combined_output = combined_output * trend_weights
        
        # Output processing
        output = self.output_projection(combined_output)
        output = self.layer_norm(output + residual)  # Residual connection
        output = self.dropout(output)
        
        if need_weights:
            return output, attention_weights_dict
        return output
    
    def _create_timeframe_mask(
        self, 
        seq_len: int, 
        window_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Create attention mask for specific timeframe window."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            # Allow attention in within window
            start_idx = max(0, i - window_size + 1)
            end_idx = min(seq_len, i + window_size)
            mask[i, start_idx:end_idx] = True
        
        return mask


class CryptoTemporalAttention(TemporalAttention):
    """
    Specialized Temporal Attention for crypto trading patterns.
    
    Additional Features:
    - Market session awareness (Asian, European, US)
    - Crypto-specific cycles (4-year halving cycles)
    - Trading volume temporal patterns
    - Weekend effect modeling
    - News event temporal decay
    """
    
    def __init__(self, config: TemporalAttentionConfig):
        super().__init__(config)
        
        # Market session embeddings
        self.session_embedding = nn.Embedding(3, config.d_model // 8)  # 3 sessions
        
        # Crypto-specific cycle embeddings
        self.halving_cycle_proj = nn.Linear(2, config.d_model // 16)  # Sin/cos for 4-year cycle
        
        # Weekend effect modeling
        self.weekend_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid()
        )
        
        # Volume temporal pattern attention
        self.volume_temporal_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
        # News event decay modeling
        self.news_decay_proj = nn.Linear(1, config.d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        volume_data: Optional[torch.Tensor] = None,
        market_sessions: Optional[torch.Tensor] = None,
        news_timestamps: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward with crypto-specific temporal features.
        
        Args:
            x: Price features [batch_size, seq_len, d_model]
            timestamps: Unix timestamps [batch_size, seq_len]
            volume_data: Volume data [batch_size, seq_len, d_model]
            market_sessions: Session IDs [batch_size, seq_len]
            news_timestamps: News event timestamps [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Market session awareness
        if market_sessions is not None:
            session_emb = self.session_embedding(market_sessions)
            x = x + session_emb
        
        # Halving cycle effects
        if timestamps is not None:
            halving_features = self._compute_halving_cycle_features(timestamps)
            halving_emb = self.halving_cycle_proj(halving_features)
            x = x + halving_emb
        
        # Weekend effect
        if timestamps is not None:
            weekend_mask = self._compute_weekend_mask(timestamps)
            weekend_gate = self.weekend_gate(x)
            x = x * (1 - weekend_mask.unsqueeze(-1) * (1 - weekend_gate))
        
        # Volume temporal attention
        if volume_data is not None:
            volume_temporal_output, _ = self.volume_temporal_attention(
                query=x,
                key=volume_data,
                value=volume_data
            )
            x = x + volume_temporal_output * 0.3  # Moderate volume influence
        
        # News event temporal decay
        if news_timestamps is not None:
            news_decay_weights = self._compute_news_decay(timestamps, news_timestamps)
            news_features = self.news_decay_proj(news_decay_weights.unsqueeze(-1))
            x = x + news_features
        
        # Apply base temporal attention
        return super().forward(x, timestamps=timestamps, **kwargs)
    
    def _compute_halving_cycle_features(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute Bitcoin halving cycle features."""
        # Bitcoin halving approximately every 4 years (1461 days)
        halving_period = 4 * 365 * 24 * 3600  # 4 years in seconds
        
        cycle_position = (timestamps % halving_period) / halving_period
        
        sin_component = torch.sin(2 * math.pi * cycle_position).unsqueeze(-1)
        cos_component = torch.cos(2 * math.pi * cycle_position).unsqueeze(-1)
        
        return torch.cat([sin_component, cos_component], dim=-1)
    
    def _compute_weekend_mask(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute weekend mask (lower activity periods)."""
        # Convert to day of week (0=Monday, 6=Sunday)
        day_of_week = ((timestamps / (24 * 3600)) + 4) % 7  # +4 for correct offset
        
        # Weekend: Saturday (5) and Sunday (6)
        weekend_mask = (day_of_week >= 5).float()
        
        return weekend_mask
    
    def _compute_news_decay(
        self, 
        current_timestamps: torch.Tensor,
        news_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Compute temporal decay for news events."""
        # Time since news event (in hours)
        time_diff = (current_timestamps - news_timestamps) / 3600
        
        # Exponential decay (news impact decreases over time)
        decay_weights = torch.exp(-time_diff / 24)  # Decay over 24 hours
        
        return decay_weights.clamp(0, 1)


def create_temporal_attention_layer(
    d_model: int,
    num_heads: int,
    attention_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory for creation temporal attention layers.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_type: Type ("standard", "crypto")
        **kwargs: Additional configuration
    """
    base_config = TemporalAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        **kwargs
    )
    
    if attention_type == "standard":
        return TemporalAttention(base_config)
    elif attention_type == "crypto":
        return CryptoTemporalAttention(base_config)
    else:
        raise ValueError(f"Unknown temporal attention type: {attention_type}")


if __name__ == "__main__":
    # Test configurations
    config = TemporalAttentionConfig(
        d_model=512,
        num_heads=8,
        dropout=0.1,
        use_multi_timeframe=True,
        use_temporal_decay=True,
        use_cyclical_attention=True,
        use_trend_attention=True,
        timeframe_windows=[1, 5, 15, 60, 240]
    )
    
    batch_size, seq_len = 4, 256
    
    # Test standard temporal attention
    temporal_attn = TemporalAttention(config)
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))  # 2022-2023
    time_indices = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    output = temporal_attn(x, timestamps=timestamps, time_indices=time_indices)
    print(f"Standard Temporal Attention output: {output.shape}")
    
    # Test crypto temporal attention
    crypto_temporal = CryptoTemporalAttention(config)
    
    volume_data = torch.randn(batch_size, seq_len, config.d_model)
    market_sessions = torch.randint(0, 3, (batch_size, seq_len))  # 3 market sessions
    
    crypto_output = crypto_temporal(
        x,
        timestamps=timestamps,
        volume_data=volume_data,
        market_sessions=market_sessions,
        news_timestamps=timestamps - 3600  # News 1 hour ago
    )
    print(f"Crypto Temporal Attention output: {crypto_output.shape}")
    
    # Test with attention weights
    output_with_weights, attention_weights = temporal_attn(
        x, timestamps=timestamps, need_weights=True
    )
    print(f"Attention weights keys: {attention_weights.keys() if isinstance(attention_weights, dict) else 'Single tensor'}")
    
    print(f"\nModel Parameters:")
    print(f"Standard: {sum(p.numel() for p in temporal_attn.parameters())}")
    print(f"Crypto: {sum(p.numel() for p in crypto_temporal.parameters())}")
    
    # Memory usage test
    print(f"\nMemory efficient test with long sequence:")
    long_x = torch.randn(2, 2048, config.d_model)
    long_timestamps = torch.randint(1640995200, 1672531200, (2, 2048))
    
    try:
        long_output = temporal_attn(long_x, timestamps=long_timestamps)
        print(f"Long sequence output: {long_output.shape}")
    except RuntimeError as e:
        print(f"Memory error with long sequence: {e}")
        # Use smaller batch for testing
        small_long_output = temporal_attn(
            long_x[:1], timestamps=long_timestamps[:1]
        )
        print(f"Small batch long sequence: {small_long_output.shape}")