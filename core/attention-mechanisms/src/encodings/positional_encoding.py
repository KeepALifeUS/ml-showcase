"""
Positional Encoding implementations for transformer architectures in crypto trading.
Includes sinusoidal, learned, relative and other variants of positional encoding.

Production-ready positional encodings with efficiency optimizations.
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

logger = logging.getLogger(__name__)


@dataclass
class PositionalEncodingConfig:
    """Configuration for various types positional encoding."""
    d_model: int = 512
    max_seq_len: int = 10000
    dropout: float = 0.1
    
    # Encoding type
    encoding_type: str = "sinusoidal"  # "sinusoidal", "learned", "relative", "rope", "alibi"
    
    # Sinusoidal encoding parameters
    base_freq: float = 10000.0
    temperature: float = 1.0
    
    # Learned encoding parameters
    use_layer_norm: bool = False
    init_std: float = 0.02
    
    # Relative positional encoding
    use_relative_attention_bias: bool = False
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    
    # RoPE (Rotary Position Embedding) parameters
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # ALiBi (Attention with Linear Biases) parameters
    alibi_max_bias: float = 8.0
    
    # Frequency modulation for crypto data
    use_frequency_modulation: bool = True
    frequency_bands: int = 8


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from original Transformer paper.
    
    Features:
    - Standard sinusoidal patterns
    - Configurable frequency base
    - Temperature scaling
    - Frequency modulation for crypto patterns
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        
        # Pre-compute sinusoidal encodings
        pe = self._create_sinusoidal_encoding(config.max_seq_len, config.d_model, config.base_freq)
        
        # Apply temperature scaling
        if config.temperature != 1.0:
            pe = pe / config.temperature
        
        self.register_buffer('pe', pe)
        
        # Frequency modulation for crypto patterns
        if config.use_frequency_modulation:
            self.freq_modulation = self._create_frequency_modulation(
                config.frequency_bands, config.d_model
            )
        else:
            self.freq_modulation = None
    
    def _create_sinusoidal_encoding(
        self, 
        max_len: int, 
        d_model: int, 
        base: float = 10000.0
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding matrix."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for frequency scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(base) / d_model)
        )
        
        # Apply sine and cosine
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _create_frequency_modulation(self, num_bands: int, d_model: int) -> nn.Module:
        """Create frequency modulation for crypto-specific patterns."""
        return nn.Sequential(
            nn.Linear(num_bands, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Tanh()
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply sinusoidal positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            position_ids: Custom position indices [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        if position_ids is not None:
            # Use custom positions
            positions = position_ids.unsqueeze(-1).float()  # [B, L, 1]
            
            # Compute encoding for custom positions
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=x.device).float() * 
                (-math.log(self.config.base_freq) / d_model)
            )
            
            # Expand dimensions for broadcasting
            div_term = div_term.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model//2]
            
            pos_encoding = torch.zeros(batch_size, seq_len, d_model, device=x.device)
            pos_encoding[:, :, 0::2] = torch.sin(positions * div_term)
            pos_encoding[:, :, 1::2] = torch.cos(positions * div_term)
        else:
            # Use pre-computed encoding
            pos_encoding = self.pe[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            pos_encoding = pos_encoding.to(x.device)
        
        # Apply frequency modulation if available
        if self.freq_modulation is not None:
            # Create frequency bands input (possible customized for crypto data)
            freq_input = torch.randn(batch_size, seq_len, self.config.frequency_bands, device=x.device)
            freq_mod = self.freq_modulation(freq_input)
            pos_encoding = pos_encoding + freq_mod * 0.1  # Small influence
        
        # Add positional encoding to input
        output = x + pos_encoding
        return self.dropout(output)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings.
    
    Features:
    - Trainable position embeddings
    - Optional layer normalization
    - Initialization strategies
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Learned position embeddings
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Layer normalization
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize position embeddings."""
        nn.init.normal_(self.position_embeddings.weight, std=self.config.init_std)
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply learned positional encoding."""
        batch_size, seq_len, d_model = x.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Get position embeddings
        pos_embeddings = self.position_embeddings(position_ids)
        
        # Add to input
        output = x + pos_embeddings
        
        # Apply layer normalization if enabled
        if hasattr(self, 'layer_norm'):
            output = self.layer_norm(output)
        
        return self.dropout(output)


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding (T5-style).
    
    Features:
    - Relative position biases
    - Bidirectional encoding
    - Bucket-based relative positions
    """
    
    def __init__(self, config: PositionalEncodingConfig, num_heads: int):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        
        # Relative attention bias
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets,
            num_heads
        )
        
        # Cache for relative position bucket mapping
        self.register_buffer(
            'relative_position_bucket_cache',
            torch.zeros(config.max_seq_len, config.max_seq_len, dtype=torch.long)
        )
        self._init_relative_position_cache()
    
    def _init_relative_position_cache(self):
        """Initialize relative position bucket cache."""
        max_seq_len = self.config.max_seq_len
        
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                relative_position = i - j
                bucket = self._relative_position_bucket(
                    relative_position,
                    bidirectional=True,
                    num_buckets=self.config.relative_attention_num_buckets,
                    max_distance=self.config.relative_attention_max_distance
                )
                self.relative_position_bucket_cache[i, j] = bucket
    
    def _relative_position_bucket(
        self,
        relative_position: int,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> int:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe7b40747ceb2b86b7bb6a7fd3f0fb80e6c/mesh_tensorflow/transformer/transformer_layers.py#L593
        """
        ret = 0
        n = -relative_position
        
        if bidirectional:
            num_buckets //= 2
            if n < 0:
                ret += num_buckets
            n = abs(n)
        else:
            n = max(n, 0)
        
        # Now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        if is_small:
            return ret + n
        else:
            return ret + max_exact + min(
                math.floor(math.log(n / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)),
                num_buckets - 1 - max_exact
            )
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Compute relative position bias.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            relative_bias: [num_heads, seq_len, seq_len]
        """
        # Get relative position buckets
        relative_position_buckets = self.relative_position_bucket_cache[:seq_len, :seq_len]
        
        # Get bias values
        relative_bias = self.relative_attention_bias(relative_position_buckets)  # [seq_len, seq_len, num_heads]
        relative_bias = relative_bias.permute(2, 0, 1)  # [num_heads, seq_len, seq_len]
        
        return relative_bias


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) from RoFormer.
    
    Features:
    - Rotary position embeddings
    - Configurable theta
    - Scaling support for longer sequences
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        head_dim = config.d_model // 8  # Assuming 8 heads, adjust accordingly
        
        # Compute frequency tensor
        inv_freq = 1.0 / (config.rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim
        ))
        self.register_buffer('inv_freq', inv_freq)
        
        # Scaling for longer sequences
        self.scaling_factor = 1.0
        if config.rope_scaling is not None:
            scaling_type = config.rope_scaling.get('type', 'linear')
            scaling_factor = config.rope_scaling.get('factor', 1.0)
            
            if scaling_type == 'linear':
                self.scaling_factor = scaling_factor
            elif scaling_type == 'dynamic':
                # Dynamic scaling based on sequence length
                self.dynamic_scaling = True
            else:
                self.scaling_factor = scaling_factor
        
        # Cache for cos and sin values
        self._cos_cache = None
        self._sin_cache = None
        self._cache_seq_len = 0
    
    def _compute_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Compute and cache cos/sin values."""
        if seq_len > self._cache_seq_len or self._cos_cache is None:
            self._cache_seq_len = max(seq_len, self._cache_seq_len)
            
            # Position indices
            t = torch.arange(self._cache_seq_len, device=device, dtype=dtype)
            t = t / self.scaling_factor
            
            # Frequency matrix
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat([freqs, freqs], dim=-1)
            
            self._cos_cache = emb.cos()
            self._sin_cache = emb.sin()
        
        return (
            self._cos_cache[:seq_len].to(dtype),
            self._sin_cache[:seq_len].to(dtype)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply RoPE to tensor x."""
        # x shape: [batch_size, seq_len, num_heads, head_dim]
        # or [batch_size, seq_len, d_model]
        
        if x.dim() == 3:
            # Reshape for multi-head format
            batch_size, seq_len, d_model = x.shape
            num_heads = 8  # Default, should be configured
            head_dim = d_model // num_heads
            x = x.view(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = x.shape
        
        # Get cos and sin values
        cos, sin = self._compute_cos_sin_cache(seq_len, x.device, x.dtype)
        
        # Apply rotary embedding
        x_rotated = self._apply_rope(x, cos, sin)
        
        # Reshape back if needed
        if x.dim() == 3:
            x_rotated = x_rotated.view(batch_size, seq_len, -1)
        
        return x_rotated
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary position embedding."""
        # Split x into two halves
        x1, x2 = x.chunk(2, dim=-1)
        
        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1),
            x1 * sin.unsqueeze(1) + x2 * cos.unsqueeze(1)
        ], dim=-1)
        
        return x_rotated


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    
    Features:
    - Linear bias penalties for distant tokens
    - No explicit positional embeddings needed
    - Extrapolates well to longer sequences
    """
    
    def __init__(self, config: PositionalEncodingConfig, num_heads: int):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        
        # Compute slopes for each attention head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """Get slopes for ALiBi bias."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(num_heads))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return torch.tensor(
                get_slopes_power_of_2(closest_power_of_2) + 
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:num_heads - closest_power_of_2]
            )
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create ALiBi bias matrix.
        
        Args:
            seq_len: Sequence length
            device: Device for tensor creation
            
        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # Create distance matrix
        arange_tensor = torch.arange(seq_len, device=device)
        distance_matrix = arange_tensor.unsqueeze(0) - arange_tensor.unsqueeze(1)
        
        # Apply slopes
        bias = distance_matrix.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        
        # Clamp maximum bias
        bias = torch.clamp(bias, max=0, min=-self.config.alibi_max_bias)
        
        return bias


class CryptoPositionalEncoding(nn.Module):
    """
    Crypto-specific positional encoding.
    
    Features:
    - Trading session awareness
    - Market cycle encoding
    - Time-of-day patterns
    - Day-of-week effects
    """
    
    def __init__(self, config: PositionalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Base positional encoding
        self.base_encoding = SinusoidalPositionalEncoding(config)
        
        # Trading session encoding (Asian, European, US)
        self.session_embedding = nn.Embedding(3, config.d_model // 8)
        
        # Time-of-day encoding (24 hours)
        self.hour_embedding = nn.Embedding(24, config.d_model // 16)
        
        # Day-of-week encoding (7 days)
        self.dow_embedding = nn.Embedding(7, config.d_model // 16)
        
        # Market cycle encoding
        self.cycle_proj = nn.Linear(4, config.d_model // 8)  # 4-year cycles
        
        # Combination layer
        self.combination = nn.Linear(config.d_model + config.d_model // 4, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply crypto-specific positional encoding.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            timestamps: Unix timestamps [batch_size, seq_len]
            position_ids: Position indices [batch_size, seq_len]
        """
        # Base positional encoding
        x_pos = self.base_encoding(x, position_ids)
        
        if timestamps is not None:
            # Extract time features
            batch_size, seq_len = timestamps.shape
            
            # Convert timestamps to time features
            hours = (timestamps // 3600) % 24  # Hour of day
            dow = ((timestamps // (24 * 3600)) + 4) % 7  # Day of week (Monday=0)
            
            # Trading sessions (simplified)
            # Asian: 0-8 UTC, European: 8-16 UTC, US: 16-24 UTC
            sessions = (hours // 8) % 3
            
            # Market cycles (4-year Bitcoin halving cycles)
            cycle_position = (timestamps % (4 * 365 * 24 * 3600)) / (4 * 365 * 24 * 3600)
            cycle_features = torch.stack([
                torch.sin(2 * math.pi * cycle_position),
                torch.cos(2 * math.pi * cycle_position),
                torch.sin(4 * math.pi * cycle_position),
                torch.cos(4 * math.pi * cycle_position)
            ], dim=-1)
            
            # Get embeddings
            session_emb = self.session_embedding(sessions.long())
            hour_emb = self.hour_embedding(hours.long())
            dow_emb = self.dow_embedding(dow.long())
            cycle_emb = self.cycle_proj(cycle_features)
            
            # Combine temporal features
            temporal_features = torch.cat([session_emb, hour_emb, dow_emb, cycle_emb], dim=-1)
            
            # Combine with positional encoding
            combined_features = torch.cat([x_pos, temporal_features], dim=-1)
            output = self.combination(combined_features)
        else:
            output = x_pos
        
        return self.dropout(output)


def create_positional_encoding(
    config: PositionalEncodingConfig,
    num_heads: Optional[int] = None
) -> nn.Module:
    """
    Factory function for creation positional encoding.
    
    Args:
        config: Configuration object
        num_heads: Number of attention heads (for relative/alibi)
        
    Returns:
        Positional encoding module
    """
    if config.encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(config)
    elif config.encoding_type == "learned":
        return LearnedPositionalEncoding(config)
    elif config.encoding_type == "relative":
        if num_heads is None:
            raise ValueError("num_heads required for relative positional encoding")
        return RelativePositionalEncoding(config, num_heads)
    elif config.encoding_type == "rope":
        return RoPE(config)
    elif config.encoding_type == "alibi":
        if num_heads is None:
            raise ValueError("num_heads required for ALiBi")
        return ALiBi(config, num_heads)
    elif config.encoding_type == "crypto":
        return CryptoPositionalEncoding(config)
    else:
        raise ValueError(f"Unknown encoding type: {config.encoding_type}")


if __name__ == "__main__":
    # Test various positional encodings
    config = PositionalEncodingConfig(
        d_model=512,
        max_seq_len=1000,
        dropout=0.1
    )
    
    batch_size, seq_len = 4, 256
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Test sinusoidal encoding
    sin_encoding = SinusoidalPositionalEncoding(config)
    sin_output = sin_encoding(x)
    print(f"Sinusoidal output: {sin_output.shape}")
    
    # Test learned encoding
    learned_encoding = LearnedPositionalEncoding(config)
    learned_output = learned_encoding(x)
    print(f"Learned output: {learned_output.shape}")
    
    # Test relative encoding
    num_heads = 8
    rel_encoding = RelativePositionalEncoding(config, num_heads)
    rel_bias = rel_encoding(seq_len)
    print(f"Relative bias: {rel_bias.shape}")
    
    # Test RoPE
    rope_encoding = RoPE(config)
    rope_output = rope_encoding(x)
    print(f"RoPE output: {rope_output.shape}")
    
    # Test ALiBi
    alibi_encoding = ALiBi(config, num_heads)
    alibi_bias = alibi_encoding(seq_len, x.device)
    print(f"ALiBi bias: {alibi_bias.shape}")
    
    # Test crypto encoding
    crypto_config = PositionalEncodingConfig(
        d_model=512,
        encoding_type="crypto",
        use_frequency_modulation=True
    )
    crypto_encoding = CryptoPositionalEncoding(crypto_config)
    
    timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))  # 2022-2023
    crypto_output = crypto_encoding(x, timestamps=timestamps)
    print(f"Crypto encoding output: {crypto_output.shape}")
    
    # Performance comparison
    print(f"\nModel parameters:")
    print(f"Sinusoidal: {sum(p.numel() for p in sin_encoding.parameters())}")
    print(f"Learned: {sum(p.numel() for p in learned_encoding.parameters())}")
    print(f"Relative: {sum(p.numel() for p in rel_encoding.parameters())}")
    print(f"RoPE: {sum(p.numel() for p in rope_encoding.parameters())}")
    print(f"ALiBi: {sum(p.numel() for p in alibi_encoding.parameters())}")
    print(f"Crypto: {sum(p.numel() for p in crypto_encoding.parameters())}")
    
    # Test extrapolation to longer sequences
    long_seq_len = 2048
    long_x = torch.randn(2, long_seq_len, config.d_model)
    
    print(f"\nExtrapolation test (seq_len={long_seq_len}):")
    
    try:
        sin_long = sin_encoding(long_x)
        print(f"Sinusoidal: ✓ {sin_long.shape}")
    except Exception as e:
        print(f"Sinusoidal: ✗ {e}")
    
    try:
        rope_long = rope_encoding(long_x)
        print(f"RoPE: ✓ {rope_long.shape}")
    except Exception as e:
        print(f"RoPE: ✗ {e}")
    
    try:
        alibi_long_bias = alibi_encoding(long_seq_len, long_x.device)
        print(f"ALiBi: ✓ {alibi_long_bias.shape}")
    except Exception as e:
        print(f"ALiBi: ✗ {e}")