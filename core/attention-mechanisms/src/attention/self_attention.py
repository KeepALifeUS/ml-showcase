"""
Self-Attention mechanism for temporal series in crypto trading.
Optimized implementation for sequence modeling with support for temporal patterns.

Production-ready self-attention with memory efficiency and real-time inference.
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
class SelfAttentionConfig(AttentionConfig):
    """Configuration for Self-Attention with additional parameters."""
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    use_residual: bool = True
    use_pre_norm: bool = True  # Pre-norm vs post-norm
    use_gated_attention: bool = False  # Gated attention for controlling information stream
    use_sparse_attention: bool = False  # Sparse attention patterns
    sparse_block_size: int = 64
    use_local_attention: bool = False  # Local attention window
    local_window_size: int = 128
    
    # Crypto-specific parameters
    use_temporal_decay: bool = True  # Temporal decay for older timesteps
    temporal_decay_rate: float = 0.95
    use_volume_scaling: bool = True  # Volume-based attention scaling
    use_volatility_aware: bool = True  # Volatility-aware attention


class SelfAttention(nn.Module):
    """
    Self-Attention mechanism optimized for crypto temporal series.
    
    Features:
    - Standard self-attention with residual connections
    - Layer normalization (pre-norm or post-norm)
    - Gated attention for selective information flow
    - Sparse attention patterns for efficiency
    - Local attention window for long sequences
    - Temporal decay weights
    - Volume and volatility aware attention
    """
    
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention core
        self.attention = MultiHeadAttention(config)
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Gated attention components
        if config.use_gated_attention:
            self.gate_proj = nn.Linear(config.d_model, config.d_model)
            self.gate_activation = nn.Sigmoid()
        
        # Sparse attention mask computation
        if config.use_sparse_attention:
            self.sparse_attention_mask = self._create_sparse_mask(
                config.max_seq_len, config.sparse_block_size
            )
        
        # Local attention mask
        if config.use_local_attention:
            self.local_attention_mask = self._create_local_mask(
                config.max_seq_len, config.local_window_size
            )
        
        # Temporal decay weights
        if config.use_temporal_decay:
            self.register_buffer('temporal_weights', self._create_temporal_weights(
                config.max_seq_len, config.temporal_decay_rate
            ))
        
        # Crypto-specific projections
        if config.use_volume_scaling:
            self.volume_proj = nn.Linear(1, 1, bias=False)
        
        if config.use_volatility_aware:
            self.volatility_proj = nn.Linear(1, config.num_heads, bias=False)
    
    def _create_sparse_mask(self, seq_len: int, block_size: int) -> torch.Tensor:
        """Create sparse attention mask with block patterns."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        # Diagonal blocks
        for i in range(0, seq_len, block_size):
            end_i = min(i + block_size, seq_len)
            for j in range(0, seq_len, block_size):
                end_j = min(j + block_size, seq_len)
                mask[i:end_i, j:end_j] = True
        
        # Global tokens (first and last tokens attend to all)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        
        return mask
    
    def _create_local_mask(self, seq_len: int, window_size: int) -> torch.Tensor:
        """Create local attention mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = True
        
        return mask
    
    def _create_temporal_weights(self, seq_len: int, decay_rate: float) -> torch.Tensor:
        """Create temporal decay weights."""
        positions = torch.arange(seq_len, dtype=torch.float)
        weights = decay_rate ** positions
        return weights.flip(0)  # More recent timesteps get higher weight
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        volume: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Custom attention mask
            key_padding_mask: Padding mask
            volume: Volume data [batch_size, seq_len] (optional)
            volatility: Volatility data [batch_size, seq_len] (optional)
            need_weights: Return attention weights
            
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
            attention_weights: Attention weights (if need_weights=True)
        """
        residual = x
        batch_size, seq_len, d_model = x.shape
        
        # Pre-normalization
        if self.config.use_layer_norm and self.config.use_pre_norm:
            x = self.layer_norm(x)
        
        # Create attention mask
        combined_mask = self._create_combined_mask(
            seq_len, attention_mask, x.device
        )
        
        # Apply temporal decay (if enabled)
        if self.config.use_temporal_decay and hasattr(self, 'temporal_weights'):
            temporal_weights = self.temporal_weights[:seq_len].to(x.device)
            x = x * temporal_weights.view(1, -1, 1)
        
        # Self-attention
        attn_output = self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=combined_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights
        )
        
        if need_weights:
            attn_output, attention_weights = attn_output
        
        # Volume scaling (if available volume)
        if self.config.use_volume_scaling and volume is not None:
            volume_scale = torch.sigmoid(self.volume_proj(volume.unsqueeze(-1)))
            attn_output = attn_output * volume_scale
        
        # Volatility-aware weighting
        if self.config.use_volatility_aware and volatility is not None:
            volatility_weights = torch.softmax(
                self.volatility_proj(volatility.unsqueeze(-1)), dim=-1
            )  # [B, L, H]
            
            # Apply to each head separately
            attn_output = attn_output.view(batch_size, seq_len, self.config.num_heads, -1)
            attn_output = attn_output * volatility_weights.unsqueeze(-1)
            attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Gated attention
        if self.config.use_gated_attention:
            gate = self.gate_activation(self.gate_proj(x))
            attn_output = attn_output * gate
        
        # Residual connection
        if self.config.use_residual:
            attn_output = attn_output + residual
        
        # Post-normalization
        if self.config.use_layer_norm and not self.config.use_pre_norm:
            attn_output = self.layer_norm(attn_output)
        
        if need_weights:
            return attn_output, attention_weights
        return attn_output
    
    def _create_combined_mask(
        self,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """Create combined attention mask."""
        masks = []
        
        # Custom attention mask
        if attention_mask is not None:
            masks.append(attention_mask)
        
        # Sparse attention mask
        if self.config.use_sparse_attention and hasattr(self, 'sparse_attention_mask'):
            sparse_mask = self.sparse_attention_mask[:seq_len, :seq_len].to(device)
            masks.append(sparse_mask)
        
        # Local attention mask
        if self.config.use_local_attention and hasattr(self, 'local_attention_mask'):
            local_mask = self.local_attention_mask[:seq_len, :seq_len].to(device)
            masks.append(local_mask)
        
        # Causal mask
        if self.config.causal:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = ~causal_mask.to(device)
            masks.append(causal_mask)
        
        # Combine all masks
        if masks:
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
            return combined_mask
        
        return None


class CryptoSelfAttention(SelfAttention):
    """
    Specialized Self-Attention for crypto trading strategies.
    
    Additional Features:
    - Market microstructure awareness
    - Cross-timeframe attention
    - Order book level attention
    - News sentiment integration
    """
    
    def __init__(self, config: SelfAttentionConfig):
        super().__init__(config)
        
        # Market microstructure components
        self.bid_ask_proj = nn.Linear(2, config.d_model // 4, bias=False)
        
        # Cross-timeframe attention weights
        self.timeframe_weights = nn.Parameter(
            torch.ones(5) / 5  # 5 main timeframes
        )
        
        # Order book depth projection
        self.orderbook_proj = nn.Linear(10, config.d_model // 8, bias=False)  # Top 5 bids/asks
        
        # News sentiment projection
        self.sentiment_proj = nn.Linear(3, config.d_model // 8, bias=False)  # Positive/Negative/Neutral
    
    def forward(
        self,
        x: torch.Tensor,
        bid_ask: Optional[torch.Tensor] = None,
        orderbook: Optional[torch.Tensor] = None,
        sentiment: Optional[torch.Tensor] = None,
        timeframe_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward with consideration market microstructure.
        
        Args:
            x: Price/volume features [batch_size, seq_len, d_model]
            bid_ask: Bid-ask spread [batch_size, seq_len, 2]
            orderbook: Order book data [batch_size, seq_len, 10]
            sentiment: News sentiment [batch_size, seq_len, 3]
            timeframe_weights: Custom timeframe weights [5]
        """
        # Augment input with market microstructure
        augmented_features = []
        
        if bid_ask is not None:
            bid_ask_features = self.bid_ask_proj(bid_ask)
            augmented_features.append(bid_ask_features)
        
        if orderbook is not None:
            orderbook_features = self.orderbook_proj(orderbook)
            augmented_features.append(orderbook_features)
        
        if sentiment is not None:
            sentiment_features = self.sentiment_proj(sentiment)
            augmented_features.append(sentiment_features)
        
        # Concatenate augmented features
        if augmented_features:
            extra_features = torch.cat(augmented_features, dim=-1)
            # Pad to match d_model if needed
            if extra_features.size(-1) < x.size(-1):
                padding_size = x.size(-1) - extra_features.size(-1)
                padding = torch.zeros(*extra_features.shape[:-1], padding_size, device=x.device)
                extra_features = torch.cat([extra_features, padding], dim=-1)
            
            x = x + extra_features[:, :, :x.size(-1)]
        
        # Apply timeframe weighting
        if timeframe_weights is not None:
            self.timeframe_weights.data = timeframe_weights
        
        # Standard self-attention forward
        return super().forward(x, **kwargs)


class LinearSelfAttention(nn.Module):
    """
    Linear complexity self-attention for very long sequences.
    Based on kernel trick for approximation of softmax attention.
    """
    
    def __init__(self, config: SelfAttentionConfig):
        super().__init__()
        self.config = config
        
        # Linear projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Feature mapping dimension
        self.feature_dim = config.head_dim * 2  # Increased for better approximation
        
        # Random features for kernel approximation
        self.register_buffer(
            'random_features',
            torch.randn(config.head_dim, self.feature_dim) * (config.head_dim ** -0.5)
        )
    
    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """Feature mapping function for kernel approximation."""
        # x: [B, H, L, D_h]
        projections = torch.matmul(x, self.random_features)  # [B, H, L, feature_dim]
        
        # Split into cosine and sine parts
        cos_proj, sin_proj = torch.split(projections, self.feature_dim // 2, dim=-1)
        
        return torch.cat([
            torch.cos(cos_proj),
            torch.sin(sin_proj)
        ], dim=-1) * (self.feature_dim ** -0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, torch.Tensor]:
        """Linear attention forward pass."""
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim).transpose(1, 2)
        
        # Feature mapping
        Q_phi = self._phi(Q)  # [B, H, L, feature_dim]
        K_phi = self._phi(K)  # [B, H, L, feature_dim]
        
        # Linear attention computation
        # Avoid materializing full attention matrix
        KV = torch.matmul(K_phi.transpose(-2, -1), V)  # [B, H, feature_dim, D_h]
        QKV = torch.matmul(Q_phi, KV)  # [B, H, L, D_h]
        
        # Normalization
        Z = torch.matmul(Q_phi, K_phi.sum(dim=-2, keepdim=True).transpose(-2, -1))  # [B, H, L, 1]
        Z = Z.clamp(min=1e-6)  # Numerical stability
        
        output = QKV / Z
        
        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.o_proj(output)
        
        if need_weights:
            # Linear attention not provides explicit attention weights
            dummy_weights = torch.zeros(
                batch_size, self.config.num_heads, seq_len, seq_len,
                device=x.device
            )
            return output, dummy_weights
        
        return output


def create_self_attention_layer(
    d_model: int,
    num_heads: int,
    attention_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating various types of self-attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_type: Type of attention ("standard", "crypto", "linear")
        **kwargs: Additional configuration parameters
    """
    base_config = SelfAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        **kwargs
    )
    
    if attention_type == "standard":
        return SelfAttention(base_config)
    elif attention_type == "crypto":
        return CryptoSelfAttention(base_config)
    elif attention_type == "linear":
        return LinearSelfAttention(base_config)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


if __name__ == "__main__":
    # Test configurations
    config = SelfAttentionConfig(
        d_model=512,
        num_heads=8,
        dropout=0.1,
        use_layer_norm=True,
        use_gated_attention=True,
        use_temporal_decay=True,
        use_volume_scaling=True
    )
    
    # Test standard self-attention
    attention = SelfAttention(config)
    
    batch_size, seq_len = 4, 256
    x = torch.randn(batch_size, seq_len, config.d_model)
    volume = torch.rand(batch_size, seq_len)
    volatility = torch.rand(batch_size, seq_len)
    
    output = attention(x, volume=volume, volatility=volatility)
    print(f"Standard Self-Attention output: {output.shape}")
    
    # Test crypto self-attention
    crypto_attention = CryptoSelfAttention(config)
    
    # Market data
    bid_ask = torch.randn(batch_size, seq_len, 2)
    orderbook = torch.randn(batch_size, seq_len, 10)
    sentiment = torch.randn(batch_size, seq_len, 3)
    
    crypto_output = crypto_attention(
        x, 
        bid_ask=bid_ask,
        orderbook=orderbook,
        sentiment=sentiment,
        volume=volume
    )
    print(f"Crypto Self-Attention output: {crypto_output.shape}")
    
    # Test linear attention for long sequences
    linear_attention = LinearSelfAttention(config)
    
    long_x = torch.randn(2, 4096, config.d_model)  # Very long sequence
    linear_output = linear_attention(long_x)
    print(f"Linear Self-Attention output: {linear_output.shape}")
    
    # Performance comparison
    print(f"\nModel Parameters:")
    print(f"Standard: {sum(p.numel() for p in attention.parameters())}")
    print(f"Crypto: {sum(p.numel() for p in crypto_attention.parameters())}")
    print(f"Linear: {sum(p.numel() for p in linear_attention.parameters())}")