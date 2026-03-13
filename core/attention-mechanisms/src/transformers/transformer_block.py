"""
Transformer Block implementation with optimizations for crypto trading ML models.
Includes encoder/decoder blocks with advanced normalization and residual connections.

Production-ready transformer blocks with memory efficiency and optimization.
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

# Import attention mechanisms
from ..attention.multi_head_attention import MultiHeadAttention, AttentionConfig
from ..attention.self_attention import SelfAttention, SelfAttentionConfig
from ..attention.cross_attention import CrossAttention, CrossAttentionConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerBlockConfig:
    """Configuration for transformer blocks."""
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 2048  # Feed-forward dimension
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Normalization
    norm_type: str = "layer_norm"  # "layer_norm", "rms_norm", "pre_norm", "post_norm"
    norm_eps: float = 1e-5
    
    # Activation functions
    activation: str = "relu"  # "relu", "gelu", "swish", "glu"
    
    # Advanced features
    use_residual: bool = True
    use_pre_norm: bool = True  # Pre-norm vs post-norm
    use_gated_mlp: bool = False  # Gated MLP like in GLU
    use_rotary_pos_emb: bool = False
    
    # Memory efficiency
    use_gradient_checkpointing: bool = False
    use_memory_efficient_attention: bool = True
    
    # Regularization
    use_dropout_schedule: bool = False  # Dynamic dropout scheduling
    stochastic_depth_rate: float = 0.0  # Stochastic depth for regularization
    
    # Crypto-specific features
    use_crypto_adaptations: bool = True
    use_market_aware_mlp: bool = False
    use_volatility_scaling: bool = False


class FeedForward(nn.Module):
    """
    Feed-forward network with various activation functions and optimizations.
    
    Features:
    - Multiple activation functions
    - Gated MLP support
    - Memory efficient implementation
    - Market-aware adaptations
    """
    
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config
        
        # Standard MLP layers
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        
        # Gated MLP support
        if config.use_gated_mlp:
            self.gate = nn.Linear(config.d_model, config.d_ff)
        
        # Activation function
        self.activation = self._get_activation_function(config.activation)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Market-aware MLP adaptations
        if config.use_market_aware_mlp:
            self.market_gate = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.Sigmoid()
            )
        
        # Volatility scaling
        if config.use_volatility_scaling:
            self.volatility_scale = nn.Parameter(torch.ones(1))
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish":
            return nn.SiLU()  # Swish/SiLU
        elif activation == "glu":
            return nn.GLU(dim=-1)
        else:
            return nn.ReLU()  # Default
    
    def forward(
        self, 
        x: torch.Tensor,
        volatility: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Feed-forward forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            volatility: Volatility data [batch_size, seq_len] (optional)
        """
        # First linear layer
        hidden = self.w1(x)
        
        # Gated MLP
        if self.config.use_gated_mlp:
            gate_values = self.gate(x)
            hidden = hidden * torch.sigmoid(gate_values)
        
        # Activation
        hidden = self.activation(hidden)
        
        # Dropout
        hidden = self.dropout(hidden)
        
        # Second linear layer
        output = self.w2(hidden)
        
        # Market-aware adaptations
        if self.config.use_market_aware_mlp:
            market_gate = self.market_gate(x)
            output = output * market_gate
        
        # Volatility scaling
        if self.config.use_volatility_scaling and volatility is not None:
            vol_scale = torch.sigmoid(self.volatility_scale * volatility.unsqueeze(-1))
            output = output * vol_scale
        
        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block with optimizations for crypto trading.
    
    Features:
    - Multi-head self-attention
    - Feed-forward network
    - Residual connections
    - Layer normalization
    - Gradient checkpointing support
    - Stochastic depth regularization
    """
    
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config
        
        # Self-attention
        attention_config = SelfAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            use_rotary_pos_emb=config.use_rotary_pos_emb
        )
        self.self_attention = SelfAttention(attention_config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
        # Normalization layers
        if config.norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        elif config.norm_type == "rms_norm":
            self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        
        # Dropout for stochastic depth
        if config.stochastic_depth_rate > 0:
            self.stochastic_depth = StochasticDepth(config.stochastic_depth_rate)
        
        # Dropout schedule
        if config.use_dropout_schedule:
            self.dropout_scheduler = DropoutScheduler(config.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        volatility: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encoder block forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            volatility: Volatility data [batch_size, seq_len]
            need_weights: Return attention weights
        """
        if self.config.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, attention_mask, volatility, need_weights
            )
        else:
            return self._forward_impl(x, attention_mask, volatility, need_weights)
    
    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        volatility: Optional[torch.Tensor],
        need_weights: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Internal forward implementation."""
        residual = x
        attention_weights = None
        
        # Pre-normalization
        if self.config.use_pre_norm:
            x = self.norm1(x)
        
        # Self-attention
        attn_output = self.self_attention(
            x, 
            attention_mask=attention_mask,
            volatility=volatility,
            need_weights=need_weights
        )
        
        if need_weights:
            attn_output, attention_weights = attn_output
        
        # Stochastic depth
        if hasattr(self, 'stochastic_depth') and self.training:
            attn_output = self.stochastic_depth(attn_output)
        
        # Residual connection
        if self.config.use_residual:
            x = residual + attn_output
        else:
            x = attn_output
        
        # Post-normalization
        if not self.config.use_pre_norm:
            x = self.norm1(x)
        
        # Feed-forward block
        residual = x
        
        if self.config.use_pre_norm:
            x = self.norm2(x)
        
        ff_output = self.feed_forward(x, volatility=volatility)
        
        # Stochastic depth
        if hasattr(self, 'stochastic_depth') and self.training:
            ff_output = self.stochastic_depth(ff_output)
        
        # Residual connection
        if self.config.use_residual:
            x = residual + ff_output
        else:
            x = ff_output
        
        # Post-normalization
        if not self.config.use_pre_norm:
            x = self.norm2(x)
        
        if need_weights:
            return x, attention_weights
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block with optimizations.
    
    Features:
    - Masked self-attention
    - Cross-attention
    - Feed-forward network
    - Causal masking support
    """
    
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.config = config
        
        # Masked self-attention
        self_attention_config = SelfAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            causal=True  # Causal for decoder
        )
        self.self_attention = SelfAttention(self_attention_config)
        
        # Cross-attention
        cross_attention_config = CrossAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        self.cross_attention = CrossAttention(cross_attention_config)
        
        # Feed-forward network
        self.feed_forward = FeedForward(config)
        
        # Normalization layers
        if config.norm_type == "layer_norm":
            self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        elif config.norm_type == "rms_norm":
            self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
            self.norm3 = RMSNorm(config.d_model, eps=config.norm_eps)
        else:
            self.norm1 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm2 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
            self.norm3 = nn.LayerNorm(config.d_model, eps=config.norm_eps)
        
        # Stochastic depth
        if config.stochastic_depth_rate > 0:
            self.stochastic_depth = StochasticDepth(config.stochastic_depth_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Decoder block forward pass.
        
        Args:
            x: Decoder input [batch_size, target_len, d_model]
            encoder_output: Encoder output [batch_size, source_len, d_model]
            self_attention_mask: Mask for self-attention
            cross_attention_mask: Mask for cross-attention
            need_weights: Return attention weights
        """
        all_attention_weights = {}
        
        # 1. Masked self-attention
        residual = x
        
        if self.config.use_pre_norm:
            x = self.norm1(x)
        
        self_attn_output = self.self_attention(
            x,
            attention_mask=self_attention_mask,
            need_weights=need_weights
        )
        
        if need_weights:
            self_attn_output, self_attn_weights = self_attn_output
            all_attention_weights['self_attention'] = self_attn_weights
        
        # Stochastic depth and residual
        if hasattr(self, 'stochastic_depth') and self.training:
            self_attn_output = self.stochastic_depth(self_attn_output)
        
        if self.config.use_residual:
            x = residual + self_attn_output
        else:
            x = self_attn_output
        
        if not self.config.use_pre_norm:
            x = self.norm1(x)
        
        # 2. Cross-attention
        residual = x
        
        if self.config.use_pre_norm:
            x = self.norm2(x)
        
        cross_attn_output = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=cross_attention_mask,
            need_weights=need_weights
        )
        
        if need_weights:
            cross_attn_output, cross_attn_weights = cross_attn_output
            all_attention_weights['cross_attention'] = cross_attn_weights
        
        # Stochastic depth and residual
        if hasattr(self, 'stochastic_depth') and self.training:
            cross_attn_output = self.stochastic_depth(cross_attn_output)
        
        if self.config.use_residual:
            x = residual + cross_attn_output
        else:
            x = cross_attn_output
        
        if not self.config.use_pre_norm:
            x = self.norm2(x)
        
        # 3. Feed-forward
        residual = x
        
        if self.config.use_pre_norm:
            x = self.norm3(x)
        
        ff_output = self.feed_forward(x)
        
        # Stochastic depth and residual
        if hasattr(self, 'stochastic_depth') and self.training:
            ff_output = self.stochastic_depth(ff_output)
        
        if self.config.use_residual:
            x = residual + ff_output
        else:
            x = ff_output
        
        if not self.config.use_pre_norm:
            x = self.norm3(x)
        
        if need_weights:
            return x, all_attention_weights
        return x


class StochasticDepth(nn.Module):
    """Stochastic Depth regularization."""
    
    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_rate == 0.0:
            return x
        
        keep_prob = 1 - self.drop_rate
        mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), keep_prob, device=x.device))
        return x * mask / keep_prob


class DropoutScheduler(nn.Module):
    """Dynamic dropout scheduling."""
    
    def __init__(self, base_dropout: float):
        super().__init__()
        self.base_dropout = base_dropout
        self.current_dropout = base_dropout
        self.step = 0
    
    def update_dropout(self, epoch: int, max_epochs: int):
        """Update dropout rate based on epoch."""
        # Linear decay
        self.current_dropout = self.base_dropout * (1 - epoch / max_epochs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, p=self.current_dropout, training=self.training)


class CryptoTransformerBlock(TransformerEncoderBlock):
    """
    Crypto-specific transformer block with trading optimizations.
    
    Additional Features:
    - Market regime aware processing
    - Volume-price attention coupling
    - Risk-aware feature scaling
    - Order book integration
    """
    
    def __init__(self, config: TransformerBlockConfig):
        super().__init__(config)
        
        # Market regime components
        self.regime_detector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 5),  # 5 market regimes
            nn.Softmax(dim=-1)
        )
        
        self.regime_adapters = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(5)
        ])
        
        # Volume-price coupling
        self.volume_price_fusion = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads // 2,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Risk scaling
        self.risk_gate = nn.Sequential(
            nn.Linear(config.d_model + 1, config.d_model),  # +1 for risk score
            nn.Sigmoid()
        )
        
        # Order book integration
        self.orderbook_proj = nn.Linear(20, config.d_model // 4)  # 10 bid + 10 ask levels
    
    def forward(
        self,
        x: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        risk_scores: Optional[torch.Tensor] = None,
        orderbook_data: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Crypto transformer block forward.
        
        Args:
            x: Price features [batch_size, seq_len, d_model]
            volume_data: Volume features [batch_size, seq_len, d_model]
            risk_scores: Risk scores [batch_size, seq_len]
            orderbook_data: Order book data [batch_size, seq_len, 20]
        """
        # Market regime detection
        regime_probs = self.regime_detector(x)
        regime_ids = torch.argmax(regime_probs, dim=-1)
        
        # Apply regime-specific adaptations
        regime_adapted = torch.zeros_like(x)
        for regime_id in range(5):
            mask = (regime_ids == regime_id).unsqueeze(-1).float()
            adapted = self.regime_adapters[regime_id](x)
            regime_adapted += adapted * mask
        
        x = x + regime_adapted * 0.1  # Small influence
        
        # Volume-price fusion
        if volume_data is not None:
            fused_features, _ = self.volume_price_fusion(
                query=x,
                key=volume_data,
                value=volume_data
            )
            x = x + fused_features * 0.3
        
        # Order book integration
        if orderbook_data is not None:
            ob_features = self.orderbook_proj(orderbook_data)
            # Pad to match d_model
            if ob_features.shape[-1] < x.shape[-1]:
                padding = torch.zeros(
                    *ob_features.shape[:-1],
                    x.shape[-1] - ob_features.shape[-1],
                    device=x.device
                )
                ob_features = torch.cat([ob_features, padding], dim=-1)
            x = x + ob_features * 0.2
        
        # Risk-aware scaling
        if risk_scores is not None:
            risk_input = torch.cat([x, risk_scores.unsqueeze(-1)], dim=-1)
            risk_gate = self.risk_gate(risk_input)
            x = x * risk_gate
        
        # Apply standard transformer block
        return super().forward(x, **kwargs)


def create_transformer_block(
    config: TransformerBlockConfig,
    block_type: str = "encoder"
) -> nn.Module:
    """
    Factory function for creation transformer blocks.
    
    Args:
        config: Block configuration
        block_type: "encoder", "decoder", "crypto"
    """
    if block_type == "encoder":
        return TransformerEncoderBlock(config)
    elif block_type == "decoder":
        return TransformerDecoderBlock(config)
    elif block_type == "crypto":
        return CryptoTransformerBlock(config)
    else:
        raise ValueError(f"Unknown block type: {block_type}")


if __name__ == "__main__":
    # Test transformer blocks
    config = TransformerBlockConfig(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        use_pre_norm=True,
        use_gated_mlp=True,
        stochastic_depth_rate=0.1,
        use_crypto_adaptations=True
    )
    
    batch_size, seq_len = 4, 256
    
    # Test encoder block
    encoder_block = TransformerEncoderBlock(config)
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    encoder_output = encoder_block(x)
    print(f"Encoder block output: {encoder_output.shape}")
    
    # Test with attention weights
    encoder_output_with_weights, attention_weights = encoder_block(x, need_weights=True)
    print(f"Encoder with weights: {encoder_output_with_weights.shape}, weights: {attention_weights.shape}")
    
    # Test decoder block
    decoder_block = TransformerDecoderBlock(config)
    
    target_x = torch.randn(batch_size, seq_len // 2, config.d_model)
    encoder_output_for_decoder = torch.randn(batch_size, seq_len, config.d_model)
    
    decoder_output = decoder_block(target_x, encoder_output_for_decoder)
    print(f"Decoder block output: {decoder_output.shape}")
    
    # Test crypto transformer block
    crypto_block = CryptoTransformerBlock(config)
    
    volume_data = torch.randn(batch_size, seq_len, config.d_model)
    risk_scores = torch.rand(batch_size, seq_len)
    orderbook_data = torch.randn(batch_size, seq_len, 20)
    
    crypto_output = crypto_block(
        x,
        volume_data=volume_data,
        risk_scores=risk_scores,
        orderbook_data=orderbook_data
    )
    print(f"Crypto transformer output: {crypto_output.shape}")
    
    # Test different normalization types
    print(f"\nTesting different normalization:")
    
    for norm_type in ["layer_norm", "rms_norm"]:
        norm_config = TransformerBlockConfig(
            d_model=512,
            num_heads=8,
            norm_type=norm_type
        )
        
        norm_block = TransformerEncoderBlock(norm_config)
        norm_output = norm_block(x)
        print(f"{norm_type}: {norm_output.shape}")
    
    # Test different activation functions
    print(f"\nTesting different activations:")
    
    for activation in ["relu", "gelu", "swish"]:
        act_config = TransformerBlockConfig(
            d_model=512,
            num_heads=8,
            activation=activation
        )
        
        act_block = TransformerEncoderBlock(act_config)
        act_output = act_block(x)
        print(f"{activation}: {act_output.shape}")
    
    # Model parameters
    print(f"\nModel parameters:")
    print(f"Encoder block: {sum(p.numel() for p in encoder_block.parameters())}")
    print(f"Decoder block: {sum(p.numel() for p in decoder_block.parameters())}")
    print(f"Crypto block: {sum(p.numel() for p in crypto_block.parameters())}")
    
    # Memory usage test
    print(f"\nMemory efficiency test:")
    
    # Standard forward
    mem_config = TransformerBlockConfig(
        d_model=512,
        num_heads=8,
        use_gradient_checkpointing=False
    )
    standard_block = TransformerEncoderBlock(mem_config)
    
    # Gradient checkpointing
    checkpoint_config = TransformerBlockConfig(
        d_model=512,
        num_heads=8,
        use_gradient_checkpointing=True
    )
    checkpoint_block = TransformerEncoderBlock(checkpoint_config)
    
    standard_output = standard_block(x)
    checkpoint_output = checkpoint_block(x)
    
    print(f"Standard forward: ✓")
    print(f"Gradient checkpointing: ✓")
    
    # Test stochastic depth
    print(f"\nTesting stochastic depth:")
    stoch_config = TransformerBlockConfig(
        d_model=512,
        num_heads=8,
        stochastic_depth_rate=0.3
    )
    
    stoch_block = TransformerEncoderBlock(stoch_config)
    stoch_block.train()  # Training mode for stochastic depth
    
    stoch_outputs = []
    for i in range(5):
        output = stoch_block(x)
        stoch_outputs.append(output)
    
    # Check variability (stochastic depth should create some variance)
    variance = torch.var(torch.stack(stoch_outputs), dim=0).mean().item()
    print(f"Stochastic depth variance: {variance:.6f} (should be > 0)")
    
    print(f"\n✅ Transformer blocks implementation complete!")