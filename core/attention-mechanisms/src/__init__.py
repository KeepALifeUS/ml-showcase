"""
ML Attention Mechanisms Package for Crypto Trading Bot v5.0

Comprehensive attention mechanisms library optimized for crypto trading applications.
Includes multi-head attention, self-attention, cross-attention, temporal attention,
positional encodings, transformer blocks, and crypto-specific models.

Enterprise-grade attention mechanisms with production optimizations.
"""

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__email__ = "team@ml-framework.io"

# Core attention mechanisms
from .attention.multi_head_attention import (
    MultiHeadAttention,
    CryptoMultiHeadAttention,
    AttentionConfig,
    create_attention_mask,
    benchmark_attention_performance
)

from .attention.self_attention import (
    SelfAttention,
    CryptoSelfAttention,
    LinearSelfAttention,
    SelfAttentionConfig,
    create_self_attention_layer
)

from .attention.cross_attention import (
    CrossAttention,
    CryptoCrossAttention,
    MultiModalCrossAttention,
    CrossAttentionConfig,
    create_cross_attention_layer
)

from .attention.temporal_attention import (
    TemporalAttention,
    CryptoTemporalAttention,
    TemporalAttentionConfig,
    create_temporal_attention_layer
)

from .attention.causal_attention import (
    CausalAttention,
    CryptoCausalAttention,
    CausalAttentionConfig,
    CausalMask,
    KVCache,
    create_causal_attention_layer
)

# Positional encodings
from .encodings.positional_encoding import (
    SinusoidalPositionalEncoding,
    LearnedPositionalEncoding,
    RelativePositionalEncoding,
    RoPE,
    ALiBi,
    CryptoPositionalEncoding,
    PositionalEncodingConfig,
    create_positional_encoding
)

from .encodings.temporal_encoding import (
    CyclicalTimeEncoding,
    MarketSessionEncoding,
    SeasonalityEncoding,
    CryptoTemporalEncoding,
    TemporalEncodingConfig,
    create_temporal_encoding
)

from .encodings.learnable_encoding import (
    LearnablePositionalEmbedding,
    CryptoLearnableEmbedding,
    AdaptiveEmbeddingLayer,
    LearnableEncodingConfig,
    create_learnable_encoding
)

# Transformer components
from .transformers.transformer_block import (
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    CryptoTransformerBlock,
    FeedForward,
    RMSNorm,
    TransformerBlockConfig,
    create_transformer_block
)

from .transformers.trading_transformer import (
    TradingTransformer,
    TradingTransformerConfig,
    InputEmbedding,
    MarketRegimeDetector,
    RiskAwareHead,
    create_trading_transformer
)

# Crypto-specific models
from .models.attention_models import (
    CryptoPricePredictionModel,
    CryptoSignalGeneratorModel,
    CryptoRiskAssessmentModel,
    CryptoPortfolioOptimizerModel,
    CryptoPredictionModelConfig,
    create_crypto_model
)

# Utilities
from .utils.attention_utils import (
    AttentionAnalyzer,
    AttentionMemoryOptimizer,
    AttentionPatternVisualizer,
    AttentionDebugger,
    AttentionStats,
    benchmark_attention_implementations
)

from .utils.visualization import (
    AttentionHeatmapVisualizer,
    InteractiveAttentionVisualizer,
    CryptoAttentionVisualizer,
    VisualizationConfig,
    create_visualization_report
)

# Package information
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Core attention mechanisms
    "MultiHeadAttention",
    "CryptoMultiHeadAttention",
    "AttentionConfig",
    "create_attention_mask",
    "benchmark_attention_performance",
    
    "SelfAttention",
    "CryptoSelfAttention", 
    "LinearSelfAttention",
    "SelfAttentionConfig",
    "create_self_attention_layer",
    
    "CrossAttention",
    "CryptoCrossAttention",
    "MultiModalCrossAttention",
    "CrossAttentionConfig",
    "create_cross_attention_layer",
    
    "TemporalAttention",
    "CryptoTemporalAttention",
    "TemporalAttentionConfig", 
    "create_temporal_attention_layer",
    
    "CausalAttention",
    "CryptoCausalAttention",
    "CausalAttentionConfig",
    "CausalMask",
    "KVCache",
    "create_causal_attention_layer",
    
    # Positional encodings
    "SinusoidalPositionalEncoding",
    "LearnedPositionalEncoding",
    "RelativePositionalEncoding",
    "RoPE",
    "ALiBi",
    "CryptoPositionalEncoding",
    "PositionalEncodingConfig",
    "create_positional_encoding",
    
    "CyclicalTimeEncoding",
    "MarketSessionEncoding",
    "SeasonalityEncoding",
    "CryptoTemporalEncoding",
    "TemporalEncodingConfig",
    "create_temporal_encoding",
    
    "LearnablePositionalEmbedding",
    "CryptoLearnableEmbedding", 
    "AdaptiveEmbeddingLayer",
    "LearnableEncodingConfig",
    "create_learnable_encoding",
    
    # Transformer components
    "TransformerEncoderBlock",
    "TransformerDecoderBlock",
    "CryptoTransformerBlock",
    "FeedForward",
    "RMSNorm",
    "TransformerBlockConfig",
    "create_transformer_block",
    
    "TradingTransformer",
    "TradingTransformerConfig",
    "InputEmbedding",
    "MarketRegimeDetector",
    "RiskAwareHead",
    "create_trading_transformer",
    
    # Crypto models
    "CryptoPricePredictionModel",
    "CryptoSignalGeneratorModel", 
    "CryptoRiskAssessmentModel",
    "CryptoPortfolioOptimizerModel",
    "CryptoPredictionModelConfig",
    "create_crypto_model",
    
    # Utilities
    "AttentionAnalyzer",
    "AttentionMemoryOptimizer",
    "AttentionPatternVisualizer",
    "AttentionDebugger",
    "AttentionStats",
    "benchmark_attention_implementations",
    
    "AttentionHeatmapVisualizer",
    "InteractiveAttentionVisualizer",
    "CryptoAttentionVisualizer",
    "VisualizationConfig", 
    "create_visualization_report"
]

# Package metadata
PACKAGE_INFO = {
    "name": "ml-attention-mechanisms",
    "version": __version__,
    "description": "Comprehensive attention mechanisms for crypto trading ML models",
    "author": __author__,
    "email": __email__,
    "license": "MIT",
    "python_requires": ">=3.10.0",
    "framework": "PyTorch",
    "domain": "Financial Machine Learning",
    "specialization": "Cryptocurrency Trading",
    "architecture": "enterprise"
}

def get_package_info():
    """Get package information."""
    return PACKAGE_INFO

def print_package_summary():
    """Print package summary."""
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    ML Attention Mechanisms v{__version__}                         ║
║                    Enterprise Crypto Trading Library                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🎯 Core Attention Mechanisms:                                               ║
║    • Multi-Head Attention (Scaled Dot-Product, Flash Attention)             ║
║    • Self-Attention (Standard, Linear, Crypto-optimized)                    ║
║    • Cross-Attention (Multi-modal, Cross-asset)                             ║
║    • Temporal Attention (Multi-timeframe, Market cycles)                    ║
║    • Causal Attention (Autoregressive, KV-cache)                           ║
║                                                                              ║
║  📍 Positional Encodings:                                                    ║
║    • Sinusoidal & Learned Embeddings                                        ║
║    • RoPE & ALiBi (Advanced position encoding)                              ║
║    • Temporal Encoding (Market sessions, Crypto cycles)                     ║
║    • Learnable Encoding (Adaptive, Context-aware)                           ║
║                                                                              ║
║  🏗️ Transformer Components:                                                  ║
║    • Encoder/Decoder Blocks (Pre/Post norm, Gated MLP)                      ║
║    • Trading Transformer (Complete architecture)                            ║
║    • Market Regime Detection                                                 ║
║    • Risk-Aware Predictions                                                  ║
║                                                                              ║
║  💰 Crypto-Specific Models:                                                  ║
║    • Price Prediction (Multi-step, Uncertainty)                             ║
║    • Signal Generation (Buy/Sell/Hold)                                       ║
║    • Risk Assessment (VaR, Drawdown)                                         ║
║    • Portfolio Optimization (Dynamic rebalancing)                           ║
║                                                                              ║
║  🛠️ Production Utilities:                                                     ║
║    • Performance Analysis & Benchmarking                                     ║
║    • Memory Optimization (Chunking, Checkpointing)                          ║
║    • Attention Visualization (Heatmaps, Interactive)                        ║
║    • Debugging & Monitoring Tools                                            ║
║                                                                              ║
║  📊 Features:                                                                ║
║    ✓ enterprise architecture                                        ║
║    ✓ Production-Ready Implementations                                        ║
║    ✓ Memory-Efficient Attention                                              ║
║    ✓ Real-Time Inference Optimized                                          ║
║    ✓ Comprehensive Test Coverage                                              ║
║    ✓ Multi-Modal Data Support                                                ║
║    ✓ Risk Management Integration                                              ║
║    ✓ Market Microstructure Awareness                                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

# Convenience imports for common usage patterns
def quick_setup():
    """Quick setup for common use cases."""
    examples = {
        'basic_attention': """
from ml_attention_mechanisms import MultiHeadAttention, AttentionConfig

# Basic multi-head attention
config = AttentionConfig(d_model=512, num_heads=8)
attention = MultiHeadAttention(config)

# Forward pass
import torch
x = torch.randn(4, 128, 512)  # batch, seq_len, d_model
output = attention(x)
        """,
        
        'crypto_model': """
from ml_attention_mechanisms import create_crypto_model

# Create crypto price prediction model
model = create_crypto_model(
    model_type="price_predictor",
    d_model=256,
    input_features=50,
    prediction_horizon=5
)

# Make predictions
price_data = torch.randn(4, 128, 50)
outputs = model(price_data=price_data, ...)
        """,
        
        'trading_transformer': """
from ml_attention_mechanisms import create_trading_transformer

# Complete trading transformer
model = create_trading_transformer(
    input_features=100,
    output_dim=1,
    use_risk_management=True,
    use_market_regime_detection=True
)

# Training/inference
x = torch.randn(4, 256, 100)
outputs = model(x=x, ...)
        """
    }
    
    print("Quick Setup Examples:")
    for name, code in examples.items():
        print(f"\n{name.upper()}:")
        print(code)

if __name__ == "__main__":
    print_package_summary()
    print("\n" + "="*80)
    quick_setup()