"""
Causal Attention mechanism for autoregressive models in crypto trading.
Ensures compliance causal constraints for real-time inference.

Production causal attention for online learning in trading systems.
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
class CausalAttentionConfig(AttentionConfig):
    """Configuration for Causal Attention mechanism."""
    # Causal attention is always True
    causal: bool = True
    
    # Look-back window constraints
    use_look_back_limit: bool = False
    max_look_back: int = 512  # Maximum positions to look back
    
    # Sliding window attention
    use_sliding_window: bool = False
    window_size: int = 256
    
    # Blockwise causal attention for efficiency
    use_blockwise_causal: bool = False
    block_size: int = 64
    
    # Memory efficient causal attention
    use_memory_efficient: bool = True
    checkpoint_attention: bool = False  # Gradient checkpointing
    
    # Future masking with lookahead bias prevention
    use_strict_causality: bool = True  # Prevent any future information leakage
    
    # KV-cache for inference optimization
    use_kv_cache: bool = True
    cache_size: int = 1024
    
    # Causal regularization
    use_causal_regularization: bool = False
    causality_strength: float = 1.0


class CausalMask:
    """Utility class for creating various causal masks."""
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Standard lower triangular causal mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # True for allowed positions
    
    @staticmethod
    def create_sliding_window_mask(
        seq_len: int, 
        window_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Sliding window causal mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            start_pos = max(0, i - window_size + 1)
            end_pos = i + 1  # Causal constraint
            mask[i, start_pos:end_pos] = True
        
        return mask
    
    @staticmethod
    def create_blockwise_causal_mask(
        seq_len: int, 
        block_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Blockwise causal mask for efficiency."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        num_blocks = math.ceil(seq_len / block_size)
        
        for block_i in range(num_blocks):
            start_i = block_i * block_size
            end_i = min((block_i + 1) * block_size, seq_len)
            
            # Allow attention within current block and previous blocks
            for block_j in range(block_i + 1):
                start_j = block_j * block_size
                end_j = min((block_j + 1) * block_size, seq_len)
                
                if block_i == block_j:
                    # Within block - apply standard causal mask
                    block_mask = CausalMask.create_causal_mask(end_i - start_i, device)
                    mask[start_i:end_i, start_j:end_j] = block_mask
                else:
                    # Previous blocks - full attention allowed
                    mask[start_i:end_i, start_j:end_j] = True
        
        return mask
    
    @staticmethod
    def create_look_back_mask(
        seq_len: int, 
        max_look_back: int, 
        device: torch.device
    ) -> torch.Tensor:
        """Limited look-back causal mask."""
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        for i in range(seq_len):
            start_pos = max(0, i - max_look_back)
            end_pos = i + 1  # Causal constraint
            mask[i, start_pos:end_pos] = True
        
        return mask


class KVCache:
    """Key-Value cache for efficient causal attention inference."""
    
    def __init__(self, batch_size: int, num_heads: int, head_dim: int, max_len: int):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_len = max_len
        
        # Cache tensors
        self.keys = torch.zeros(batch_size, num_heads, max_len, head_dim)
        self.values = torch.zeros(batch_size, num_heads, max_len, head_dim)
        self.current_len = 0
    
    def update(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new keys/values and return full cached sequences."""
        seq_len = keys.shape[2]
        
        if self.current_len + seq_len > self.max_len:
            # Shift cache if exceeded limit
            shift_amount = self.current_len + seq_len - self.max_len
            self.keys[:, :, :-shift_amount] = self.keys[:, :, shift_amount:]
            self.values[:, :, :-shift_amount] = self.values[:, :, shift_amount:]
            self.current_len = self.max_len - seq_len
        
        # Add new keys/values
        self.keys[:, :, self.current_len:self.current_len + seq_len] = keys
        self.values[:, :, self.current_len:self.current_len + seq_len] = values
        self.current_len += seq_len
        
        return (
            self.keys[:, :, :self.current_len].clone(),
            self.values[:, :, :self.current_len].clone()
        )
    
    def reset(self):
        """Reset cache."""
        self.current_len = 0
        self.keys.zero_()
        self.values.zero_()
    
    def to(self, device: torch.device):
        """Move cache to device."""
        self.keys = self.keys.to(device)
        self.values = self.values.to(device)
        return self


class CausalAttention(nn.Module):
    """
    Causal Attention mechanism for autoregressive models.
    
    Features:
    - Strict causal masking (no future information leakage)
    - Sliding window attention for efficiency
    - Blockwise causal attention patterns
    - KV-cache for inference optimization
    - Memory efficient attention computation
    - Gradient checkpointing support
    """
    
    def __init__(self, config: CausalAttentionConfig):
        super().__init__()
        self.config = config
        
        # Base attention mechanism
        self.attention = MultiHeadAttention(config)
        
        # Pre-compute and register causal masks
        self._register_causal_masks()
        
        # KV-cache for inference
        if config.use_kv_cache:
            self.kv_cache: Optional[KVCache] = None
        
        # Causal regularization
        if config.use_causal_regularization:
            self.causality_loss = nn.Parameter(torch.tensor(0.0))
    
    def _register_causal_masks(self):
        """Pre-compute and register various causal masks."""
        max_seq_len = self.config.max_seq_len
        
        # Standard causal mask
        causal_mask = CausalMask.create_causal_mask(max_seq_len, torch.device('cpu'))
        self.register_buffer('causal_mask', causal_mask)
        
        # Sliding window mask
        if self.config.use_sliding_window:
            sliding_mask = CausalMask.create_sliding_window_mask(
                max_seq_len, self.config.window_size, torch.device('cpu')
            )
            self.register_buffer('sliding_window_mask', sliding_mask)
        
        # Blockwise causal mask
        if self.config.use_blockwise_causal:
            blockwise_mask = CausalMask.create_blockwise_causal_mask(
                max_seq_len, self.config.block_size, torch.device('cpu')
            )
            self.register_buffer('blockwise_causal_mask', blockwise_mask)
        
        # Look-back limited mask
        if self.config.use_look_back_limit:
            lookback_mask = CausalMask.create_look_back_mask(
                max_seq_len, self.config.max_look_back, torch.device('cpu')
            )
            self.register_buffer('lookback_mask', lookback_mask)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get appropriate causal mask based on configuration."""
        if self.config.use_sliding_window and hasattr(self, 'sliding_window_mask'):
            return self.sliding_window_mask[:seq_len, :seq_len].to(device)
        elif self.config.use_blockwise_causal and hasattr(self, 'blockwise_causal_mask'):
            return self.blockwise_causal_mask[:seq_len, :seq_len].to(device)
        elif self.config.use_look_back_limit and hasattr(self, 'lookback_mask'):
            return self.lookback_mask[:seq_len, :seq_len].to(device)
        else:
            return self.causal_mask[:seq_len, :seq_len].to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[KVCache] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, KVCache]]:
        """
        Causal attention forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Additional attention mask
            use_cache: Use KV-cache for inference
            past_key_values: Previous KV-cache state
            need_weights: Return attention weights
            
        Returns:
            output: Output tensor
            attention_weights: Attention weights (if requested)
            kv_cache: Updated KV-cache (if used)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)
        
        # Combine with additional mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            combined_mask = causal_mask & attention_mask.squeeze()
        else:
            combined_mask = causal_mask
        
        # Initialize or update KV-cache
        if use_cache:
            if past_key_values is None and self.config.use_kv_cache:
                self.kv_cache = KVCache(
                    batch_size, 
                    self.config.num_heads, 
                    self.config.head_dim, 
                    self.config.cache_size
                ).to(x.device)
            else:
                self.kv_cache = past_key_values
        
        # Attention computation with memory efficiency
        if self.config.use_memory_efficient and self.config.checkpoint_attention:
            # Gradient checkpointing for memory efficiency
            output = torch.utils.checkpoint.checkpoint(
                self._attention_forward,
                x, combined_mask, need_weights
            )
        else:
            output = self._attention_forward(x, combined_mask, need_weights)
        
        if need_weights:
            output, attention_weights = output
        
        # Causal regularization
        if self.config.use_causal_regularization and self.training:
            causality_penalty = self._compute_causality_penalty(attention_weights if need_weights else None)
            self.causality_loss.data = causality_penalty
        
        # Return results based on configuration
        if use_cache and need_weights:
            return output, attention_weights, self.kv_cache
        elif use_cache:
            return output, self.kv_cache
        elif need_weights:
            return output, attention_weights
        else:
            return output
    
    def _attention_forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor, 
        need_weights: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Internal attention forward computation."""
        return self.attention(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            need_weights=need_weights
        )
    
    def _compute_causality_penalty(self, attention_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute penalty for violations of causality."""
        if attention_weights is None:
            return torch.tensor(0.0, device=self.causality_loss.device)
        
        # Check for future information leakage
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Create upper triangular mask (future positions)
        future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        future_mask = future_mask.to(attention_weights.device)
        
        # Penalty for attention to future positions
        future_attention = attention_weights.masked_select(future_mask)
        causality_penalty = future_attention.sum() * self.config.causality_strength
        
        return causality_penalty
    
    def generate_next_token(
        self, 
        current_sequence: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate next token using causal attention (for inference).
        
        Args:
            current_sequence: Current sequence [batch_size, seq_len, d_model]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            
        Returns:
            next_token: Next token prediction [batch_size, 1, d_model]
        """
        with torch.no_grad():
            # Forward pass with cache
            output, self.kv_cache = self.forward(
                current_sequence[:, -1:],  # Only process last token
                use_cache=True,
                past_key_values=self.kv_cache
            )
            
            # Apply temperature
            if temperature != 1.0:
                output = output / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(output, top_k, dim=-1)
                output = torch.full_like(output, float('-inf'))
                output.scatter_(-1, top_k_indices, top_k_values)
            
            # Apply nucleus (top-p) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(output, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                output[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probabilities = F.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities.view(-1, probabilities.size(-1)), 1)
            next_token = next_token.view(output.shape[:-1] + (1,))
            
            return next_token


class CryptoCausalAttention(CausalAttention):
    """
    Specialized Causal Attention for crypto trading models.
    
    Additional Features:
    - Order execution causality (orders must be placed before execution)
    - Price movement causality constraints
    - Market data temporal ordering
    - Risk-aware causal constraints
    """
    
    def __init__(self, config: CausalAttentionConfig):
        super().__init__(config)
        
        # Order execution causality constraints
        self.order_causality_gate = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid()
        )
        
        # Price movement causality
        self.price_causality_weights = nn.Parameter(
            torch.linspace(1.0, 0.1, config.max_seq_len)  # Recent prices have more influence
        )
        
        # Risk-aware causal masking
        self.risk_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Market microstructure causality
        self.microstructure_mask_proj = nn.Linear(config.d_model, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        order_timestamps: Optional[torch.Tensor] = None,
        execution_timestamps: Optional[torch.Tensor] = None,
        risk_scores: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward with crypto-specific causal constraints.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            order_timestamps: Order placement timestamps
            execution_timestamps: Order execution timestamps  
            risk_scores: Risk scores for each timestep
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply price causality weighting
        price_weights = self.price_causality_weights[:seq_len].to(x.device)
        x = x * price_weights.view(1, -1, 1)
        
        # Order execution causality
        if order_timestamps is not None and execution_timestamps is not None:
            # Ensure orders precede executions
            order_execution_mask = self._create_order_execution_mask(
                order_timestamps, execution_timestamps
            )
            
            # Apply causality gate
            order_gate = self.order_causality_gate(x)
            x = x * order_gate * order_execution_mask.unsqueeze(-1)
        
        # Risk-aware causal masking
        if risk_scores is not None:
            risk_mask = (risk_scores < self.risk_threshold).float()
            x = x * risk_mask.unsqueeze(-1)
        
        # Microstructure causality constraints
        microstructure_weights = torch.sigmoid(self.microstructure_mask_proj(x))
        x = x * microstructure_weights
        
        # Apply base causal attention
        return super().forward(x, **kwargs)
    
    def _create_order_execution_mask(
        self, 
        order_timestamps: torch.Tensor,
        execution_timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Create mask ensuring order causality."""
        batch_size, seq_len = order_timestamps.shape
        
        # Create mask where order timestamp <= execution timestamp
        causality_mask = order_timestamps.unsqueeze(-1) <= execution_timestamps.unsqueeze(-2)
        
        return causality_mask.float()


def create_causal_attention_layer(
    d_model: int,
    num_heads: int,
    attention_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory for creation causal attention layers.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        attention_type: Type ("standard", "crypto")
        **kwargs: Additional configuration
    """
    base_config = CausalAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        **kwargs
    )
    
    if attention_type == "standard":
        return CausalAttention(base_config)
    elif attention_type == "crypto":
        return CryptoCausalAttention(base_config)
    else:
        raise ValueError(f"Unknown causal attention type: {attention_type}")


if __name__ == "__main__":
    # Test configurations
    config = CausalAttentionConfig(
        d_model=512,
        num_heads=8,
        dropout=0.1,
        use_sliding_window=True,
        window_size=128,
        use_kv_cache=True,
        use_memory_efficient=True
    )
    
    batch_size, seq_len = 4, 256
    
    # Test standard causal attention
    causal_attn = CausalAttention(config)
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Standard forward
    output = causal_attn(x)
    print(f"Standard Causal Attention output: {output.shape}")
    
    # Forward with KV-cache
    output_cached, kv_cache = causal_attn(x, use_cache=True)
    print(f"Cached Causal Attention output: {output_cached.shape}")
    print(f"KV-cache current length: {kv_cache.current_len}")
    
    # Test incremental generation
    print("\nTesting incremental generation:")
    for i in range(5):
        new_input = torch.randn(batch_size, 1, config.d_model)
        output_step, kv_cache = causal_attn(new_input, use_cache=True, past_key_values=kv_cache)
        print(f"Step {i+1}: output {output_step.shape}, cache length {kv_cache.current_len}")
    
    # Test crypto causal attention
    crypto_config = CausalAttentionConfig(
        d_model=512,
        num_heads=8,
        use_causal_regularization=True,
        causality_strength=0.1
    )
    crypto_causal = CryptoCausalAttention(crypto_config)
    
    order_timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
    execution_timestamps = order_timestamps + torch.randint(1, 3600, (batch_size, seq_len))  # 1 sec to 1 hour later
    risk_scores = torch.rand(batch_size, seq_len)
    
    crypto_output = crypto_causal(
        x,
        order_timestamps=order_timestamps,
        execution_timestamps=execution_timestamps,
        risk_scores=risk_scores
    )
    print(f"\nCrypto Causal Attention output: {crypto_output.shape}")
    
    # Test attention weights
    output_with_weights, attention_weights = causal_attn(x, need_weights=True)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Verify causality (no future information)
    upper_triangle = torch.triu(attention_weights[0, 0], diagonal=1)
    future_attention = upper_triangle.sum().item()
    print(f"Future attention sum (should be ~0): {future_attention:.6f}")
    
    print(f"\nModel Parameters:")
    print(f"Standard: {sum(p.numel() for p in causal_attn.parameters())}")
    print(f"Crypto: {sum(p.numel() for p in crypto_causal.parameters())}")
    
    # Performance test
    print(f"\nPerformance test:")
    import time
    
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    for _ in range(10):
        _ = causal_attn(x)
    
    start_time = time.time()
    for _ in range(100):
        _ = causal_attn(x)
    end_time = time.time()
    
    print(f"100 forward passes: {end_time - start_time:.4f}s")
    print(f"Average per forward: {(end_time - start_time) / 100 * 1000:.2f}ms")