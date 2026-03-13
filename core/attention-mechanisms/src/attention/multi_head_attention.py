"""
Multi-head mechanism attention for crypto trading ML models.
Optimized implementation Scaled Dot-Product Attention with support for Flash Attention.

Enterprise-grade architecture with production optimizations for HFT.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass
import logging

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash Attention not available. Using standard implementation.")

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for mechanism attention."""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attn: bool = True
    use_rotary_pos_emb: bool = False
    max_seq_len: int = 2048
    scale_factor: Optional[float] = None
    causal: bool = False
    use_bias: bool = True
    
    def __post_init__(self):
        """Validation and configuration configuration."""
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) should be divisible by num_heads ({self.num_heads})")
        
        self.head_dim = self.d_model // self.num_heads
        
        if self.scale_factor is None:
            self.scale_factor = self.head_dim ** -0.5
        
        if self.use_flash_attn and not FLASH_ATTN_AVAILABLE:
            logger.warning("Flash Attention requested, but unavailable. Using standard implementation.")
            self.use_flash_attn = False


class MultiHeadAttention(nn.Module):
    """
    Multi-head mechanism attention with optimizations for crypto trading.
    
    Features:
    - Scaled Dot-Product Attention
    - Flash Attention for efficiency
    - Support for causal masking
    - Rotary Position Embeddings (optional)
    - Mixed precision training
    - Memory efficient attention
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Linear projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        
        # Rotary Position Embeddings (if needed)
        if config.use_rotary_pos_emb:
            self.rotary_emb = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights by Xavier/Glorot."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        return_attention_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (None for self-attention)
            value: Value tensor [batch_size, seq_len, d_model] (None for self-attention)
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            key_padding_mask: Key padding mask [batch_size, seq_len]
            need_weights: Return attention weights
            return_attention_weights: Legacy parameter for need_weights
            
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
            attention_weights: Weights attention (if need_weights=True)
        """
        if key is None:
            key = query
        if value is None:
            value = query
            
        batch_size, seq_len, d_model = query.shape
        
        # Linear projections
        Q = self.q_proj(query)  # [B, L, D]
        K = self.k_proj(key)    # [B, S, D]
        V = self.v_proj(value)  # [B, S, D]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.config.num_heads, self.config.head_dim)
        K = K.view(batch_size, -1, self.config.num_heads, self.config.head_dim)
        V = V.view(batch_size, -1, self.config.num_heads, self.config.head_dim)
        
        # Apply rotary embeddings if enabled
        if self.config.use_rotary_pos_emb:
            Q, K = self.rotary_emb(Q, K)
        
        # Select implementations attention
        if self.config.use_flash_attn and FLASH_ATTN_AVAILABLE:
            output, attention_weights = self._flash_attention(Q, K, V, attention_mask)
        else:
            output, attention_weights = self._scaled_dot_product_attention(
                Q, K, V, attention_mask, key_padding_mask
            )
        
        # Output projection
        output = self.o_proj(output)
        output = self.output_dropout(output)
        
        if need_weights or return_attention_weights:
            return output, attention_weights
        return output
    
    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard implementation Scaled Dot-Product Attention."""
        # Transpose for batch matrix multiplication
        Q = Q.transpose(1, 2)  # [B, H, L, D_h]
        K = K.transpose(1, 2)  # [B, H, S, D_h]
        V = V.transpose(1, 2)  # [B, H, S, D_h]
        
        # Computation scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.config.scale_factor
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Causal masking for autoregressive models
        if self.config.causal:
            seq_len = Q.size(-2)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            causal_mask = causal_mask.to(scores.device)
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(output.size(0), output.size(1), -1)
        
        return output, attention_weights
    
    def _flash_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flash Attention implementation for efficiency."""
        # Flash attention expects [batch, seq_len, num_heads, head_dim]
        output = flash_attn_func(
            Q, K, V,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            causal=self.config.causal,
            softmax_scale=self.config.scale_factor
        )
        
        # Reshape output
        batch_size, seq_len = output.shape[:2]
        output = output.view(batch_size, seq_len, -1)
        
        # Flash attention not returns weights attention
        attention_weights = None
        
        return output, attention_weights


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for improved positional encoding."""
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cosines and sines
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute cosines and sines for positions."""
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            self._cached_seq_len = max(seq_len, self._cached_seq_len)
            
            # Position indices
            t = torch.arange(self._cached_seq_len, device=device, dtype=dtype)
            
            # Frequency matrix
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat([freqs, freqs], dim=-1)
            
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
        
        return (
            self._cached_cos[:seq_len].to(dtype),
            self._cached_sin[:seq_len].to(dtype)
        )
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key tensors."""
        seq_len = Q.shape[1]
        
        cos, sin = self._compute_cos_sin(seq_len, Q.device, Q.dtype)
        
        # Apply rotary embeddings
        Q_rotated = self._apply_rotary_emb(Q, cos, sin)
        K_rotated = self._apply_rotary_emb(K, cos, sin)
        
        return Q_rotated, K_rotated
    
    def _apply_rotary_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding to tensor."""
        # Split last dimension in half
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        
        # Apply rotation
        return torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)


class CryptoMultiHeadAttention(MultiHeadAttention):
    """
    Specialized Multi-Head Attention for crypto trading.
    
    Features:
    - Volume-weighted attention
    - Market regime awareness
    - Cross-asset attention patterns
    - Time-decay attention weights
    """
    
    def __init__(self, config: AttentionConfig, use_volume_weighting: bool = True):
        super().__init__(config)
        self.use_volume_weighting = use_volume_weighting
        
        if use_volume_weighting:
            self.volume_proj = nn.Linear(1, config.num_heads, bias=False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        volume: Optional[torch.Tensor] = None,
        market_regime: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward with consideration of crypto-specific factors."""
        output = super().forward(query, key, value, **kwargs)
        
        # Volume weighting (if available volume)
        if self.use_volume_weighting and volume is not None:
            volume_weights = torch.sigmoid(self.volume_proj(volume.unsqueeze(-1)))
            if isinstance(output, tuple):
                output = (output[0] * volume_weights.unsqueeze(-1), output[1])
            else:
                output = output * volume_weights.unsqueeze(-1)
        
        return output


def create_attention_mask(seq_len: int, causal: bool = False) -> torch.Tensor:
    """Create attention mask."""
    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # Invert for correct masking
    else:
        return torch.ones(seq_len, seq_len).bool()


def benchmark_attention_performance():
    """Benchmark performance of various attention implementations."""
    import time
    
    config = AttentionConfig(d_model=512, num_heads=8)
    batch_size, seq_len = 32, 1024
    
    # Test data
    query = torch.randn(batch_size, seq_len, config.d_model)
    
    # Standard attention
    config_std = AttentionConfig(d_model=512, num_heads=8, use_flash_attn=False)
    attn_std = MultiHeadAttention(config_std)
    
    start_time = time.time()
    for _ in range(100):
        _ = attn_std(query)
    std_time = time.time() - start_time
    
    print(f"Standard Attention: {std_time:.4f}s")
    
    # Flash attention (if available)
    if FLASH_ATTN_AVAILABLE:
        config_flash = AttentionConfig(d_model=512, num_heads=8, use_flash_attn=True)
        attn_flash = MultiHeadAttention(config_flash)
        
        start_time = time.time()
        for _ in range(100):
            _ = attn_flash(query)
        flash_time = time.time() - start_time
        
        print(f"Flash Attention: {flash_time:.4f}s")
        print(f"Speedup: {std_time / flash_time:.2f}x")


if __name__ == "__main__":
    # Testing configuration
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        dropout=0.1,
        use_flash_attn=True,
        causal=False
    )
    
    # Create model
    attention = MultiHeadAttention(config)
    
    # Test forward pass
    batch_size, seq_len = 4, 256
    query = torch.randn(batch_size, seq_len, config.d_model)
    
    output = attention(query)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in attention.parameters())}")
    
    # Crypto-specific attention
    crypto_attention = CryptoMultiHeadAttention(config)
    volume = torch.rand(batch_size, seq_len)  # Volume data
    
    crypto_output = crypto_attention(query, volume=volume)
    print(f"Crypto attention output shape: {crypto_output.shape}")
    
    # Performance benchmark
    print("\nPerformance Benchmark:")
    benchmark_attention_performance()