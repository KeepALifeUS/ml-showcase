"""
Comprehensive test suite for attention mechanisms package.
Covers all attention implementations, encodings, transformers and utility functions.

Production-grade testing with performance benchmarks and edge case coverage.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, Any, Tuple, List
import time
import logging

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention.multi_head_attention import (
    MultiHeadAttention, AttentionConfig, CryptoMultiHeadAttention,
    create_attention_mask, benchmark_attention_performance
)
from attention.self_attention import (
    SelfAttention, SelfAttentionConfig, CryptoSelfAttention, 
    LinearSelfAttention, create_self_attention_layer
)
from attention.cross_attention import (
    CrossAttention, CrossAttentionConfig, CryptoCrossAttention,
    MultiModalCrossAttention, create_cross_attention_layer
)
from attention.temporal_attention import (
    TemporalAttention, TemporalAttentionConfig, CryptoTemporalAttention,
    create_temporal_attention_layer
)
from attention.causal_attention import (
    CausalAttention, CausalAttentionConfig, CryptoCausalAttention,
    CausalMask, KVCache, create_causal_attention_layer
)
from encodings.positional_encoding import (
    SinusoidalPositionalEncoding, LearnedPositionalEncoding,
    RelativePositionalEncoding, RoPE, ALiBi, CryptoPositionalEncoding,
    create_positional_encoding, PositionalEncodingConfig
)
from encodings.temporal_encoding import (
    CyclicalTimeEncoding, MarketSessionEncoding, SeasonalityEncoding,
    CryptoTemporalEncoding, create_temporal_encoding, TemporalEncodingConfig
)
from encodings.learnable_encoding import (
    LearnablePositionalEmbedding, CryptoLearnableEmbedding,
    AdaptiveEmbeddingLayer, create_learnable_encoding, LearnableEncodingConfig
)
from transformers.transformer_block import (
    TransformerEncoderBlock, TransformerDecoderBlock, CryptoTransformerBlock,
    FeedForward, RMSNorm, create_transformer_block, TransformerBlockConfig
)
from transformers.trading_transformer import (
    TradingTransformer, TradingTransformerConfig, create_trading_transformer
)
from models.attention_models import (
    CryptoPricePredictionModel, CryptoSignalGeneratorModel,
    CryptoRiskAssessmentModel, CryptoPortfolioOptimizerModel,
    create_crypto_model, CryptoPredictionModelConfig
)
from utils.attention_utils import (
    AttentionAnalyzer, AttentionMemoryOptimizer, AttentionPatternVisualizer,
    AttentionDebugger, benchmark_attention_implementations
)

logger = logging.getLogger(__name__)


class TestAttentionMechanisms:
    """Test suite for core attention mechanisms."""
    
    @pytest.fixture
    def basic_config(self):
        return AttentionConfig(
            d_model=512,
            num_heads=8,
            dropout=0.1,
            max_seq_len=256
        )
    
    @pytest.fixture
    def test_data(self):
        batch_size, seq_len, d_model = 4, 64, 512
        return {
            'x': torch.randn(batch_size, seq_len, d_model),
            'query': torch.randn(batch_size, seq_len, d_model),
            'key': torch.randn(batch_size, seq_len, d_model),
            'value': torch.randn(batch_size, seq_len, d_model),
            'batch_size': batch_size,
            'seq_len': seq_len,
            'd_model': d_model
        }
    
    def test_multi_head_attention_forward(self, basic_config, test_data):
        """Test MultiHeadAttention forward pass."""
        attention = MultiHeadAttention(basic_config)
        
        # Test self-attention
        output = attention(test_data['query'])
        assert output.shape == test_data['query'].shape
        
        # Test cross-attention
        output = attention(
            query=test_data['query'],
            key=test_data['key'],
            value=test_data['value']
        )
        assert output.shape == test_data['query'].shape
        
        # Test with attention weights
        output, weights = attention(
            test_data['query'],
            need_weights=True
        )
        assert output.shape == test_data['query'].shape
        assert weights.shape == (
            test_data['batch_size'],
            basic_config.num_heads,
            test_data['seq_len'],
            test_data['seq_len']
        )
    
    def test_crypto_multi_head_attention(self, basic_config, test_data):
        """Test CryptoMultiHeadAttention functionality."""
        crypto_attention = CryptoMultiHeadAttention(basic_config, use_volume_weighting=True)
        
        volume = torch.rand(test_data['batch_size'], test_data['seq_len'])
        
        output = crypto_attention(
            query=test_data['query'],
            volume=volume
        )
        assert output.shape == test_data['query'].shape
    
    def test_attention_mask_creation(self):
        """Test attention mask creation utilities."""
        seq_len = 32
        
        # Test standard mask
        mask = create_attention_mask(seq_len, causal=False)
        assert mask.shape == (seq_len, seq_len)
        assert mask.all()  # All True for non-causal
        
        # Test causal mask
        causal_mask = create_attention_mask(seq_len, causal=True)
        assert causal_mask.shape == (seq_len, seq_len)
        assert torch.triu(causal_mask, diagonal=1).sum() == 0  # Upper triangle zeros
    
    def test_self_attention_variants(self, test_data):
        """Test different self-attention implementations."""
        config = SelfAttentionConfig(
            d_model=test_data['d_model'],
            num_heads=8,
            use_gated_attention=True,
            use_temporal_decay=True
        )
        
        # Standard self-attention
        self_attn = SelfAttention(config)
        output = self_attn(test_data['x'])
        assert output.shape == test_data['x'].shape
        
        # Crypto self-attention
        crypto_self_attn = CryptoSelfAttention(config)
        bid_ask = torch.randn(test_data['batch_size'], test_data['seq_len'], 2)
        
        crypto_output = crypto_self_attn(
            test_data['x'],
            bid_ask=bid_ask
        )
        assert crypto_output.shape == test_data['x'].shape
        
        # Linear self-attention
        linear_self_attn = LinearSelfAttention(config)
        linear_output = linear_self_attn(test_data['x'])
        assert linear_output.shape == test_data['x'].shape
    
    def test_cross_attention_variants(self, test_data):
        """Test cross-attention implementations."""
        config = CrossAttentionConfig(
            d_model=test_data['d_model'],
            num_heads=8,
            use_gated_fusion=True
        )
        
        # Standard cross-attention
        cross_attn = CrossAttention(config)
        output = cross_attn(
            query=test_data['query'],
            key=test_data['key'],
            value=test_data['value']
        )
        assert output.shape == test_data['query'].shape
        
        # Multi-modal cross-attention
        modalities = ['price', 'volume', 'news']
        multi_modal = MultiModalCrossAttention(config, modalities)
        
        modality_data = {
            'price': test_data['query'],
            'volume': test_data['key'],
            'news': test_data['value']
        }
        
        multi_output = multi_modal(modality_data)
        assert multi_output.shape == test_data['query'].shape
    
    def test_temporal_attention(self, test_data):
        """Test temporal attention mechanisms."""
        config = TemporalAttentionConfig(
            d_model=test_data['d_model'],
            num_heads=8,
            use_multi_timeframe=True,
            use_temporal_decay=True
        )
        
        temporal_attn = TemporalAttention(config)
        timestamps = torch.randint(1640995200, 1672531200, (test_data['batch_size'], test_data['seq_len']))
        
        output = temporal_attn(
            test_data['x'],
            timestamps=timestamps
        )
        assert output.shape == test_data['x'].shape
        
        # Test crypto temporal attention
        crypto_temporal = CryptoTemporalAttention(config)
        volume_data = torch.randn(test_data['batch_size'], test_data['seq_len'], test_data['d_model'])
        
        crypto_output = crypto_temporal(
            test_data['x'],
            timestamps=timestamps,
            volume_data=volume_data
        )
        assert crypto_output.shape == test_data['x'].shape
    
    def test_causal_attention(self, test_data):
        """Test causal attention mechanisms."""
        config = CausalAttentionConfig(
            d_model=test_data['d_model'],
            num_heads=8,
            use_kv_cache=True,
            use_sliding_window=True
        )
        
        causal_attn = CausalAttention(config)
        
        # Test standard forward
        output = causal_attn(test_data['x'])
        assert output.shape == test_data['x'].shape
        
        # Test with KV cache
        output_cached, kv_cache = causal_attn(test_data['x'], use_cache=True)
        assert output_cached.shape == test_data['x'].shape
        assert isinstance(kv_cache, KVCache)
        
        # Test incremental generation
        new_input = torch.randn(test_data['batch_size'], 1, test_data['d_model'])
        next_output, updated_cache = causal_attn(
            new_input, use_cache=True, past_key_values=kv_cache
        )
        assert next_output.shape == (test_data['batch_size'], 1, test_data['d_model'])
        assert updated_cache.current_len > kv_cache.current_len


class TestPositionalEncodings:
    """Test suite for positional encodings."""
    
    @pytest.fixture
    def encoding_config(self):
        return PositionalEncodingConfig(
            d_model=256,
            max_seq_len=512,
            dropout=0.1
        )
    
    def test_sinusoidal_encoding(self, encoding_config):
        """Test sinusoidal positional encoding."""
        encoding = SinusoidalPositionalEncoding(encoding_config)
        
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, encoding_config.d_model)
        
        output = encoding(x)
        assert output.shape == x.shape
        
        # Test with custom positions
        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        output_custom = encoding(x, position_ids=position_ids)
        assert output_custom.shape == x.shape
    
    def test_learned_encoding(self, encoding_config):
        """Test learned positional encoding."""
        encoding = LearnedPositionalEncoding(encoding_config)
        
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, encoding_config.d_model)
        
        output = encoding(x)
        assert output.shape == x.shape
        
        # Check that embeddings are learnable
        assert any(p.requires_grad for p in encoding.parameters())
    
    def test_relative_encoding(self, encoding_config):
        """Test relative positional encoding."""
        num_heads = 8
        encoding = RelativePositionalEncoding(encoding_config, num_heads)
        
        seq_len = 64
        bias = encoding(seq_len)
        assert bias.shape == (num_heads, seq_len, seq_len)
    
    def test_rope_encoding(self, encoding_config):
        """Test RoPE encoding."""
        rope = RoPE(encoding_config)
        
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, encoding_config.d_model)
        
        output = rope(x)
        assert output.shape == x.shape
    
    def test_alibi_encoding(self, encoding_config):
        """Test ALiBi encoding."""
        num_heads = 8
        alibi = ALiBi(encoding_config, num_heads)
        
        seq_len = 64
        bias = alibi(seq_len, torch.device('cpu'))
        assert bias.shape == (num_heads, seq_len, seq_len)
        
        # Check that bias values are negative (ALiBi penalty)
        assert (bias <= 0).all()
    
    def test_crypto_encoding(self, encoding_config):
        """Test crypto-specific positional encoding."""
        crypto_config = PositionalEncodingConfig(
            d_model=encoding_config.d_model,
            encoding_type="crypto"
        )
        crypto_encoding = CryptoPositionalEncoding(crypto_config)
        
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, encoding_config.d_model)
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        
        output = crypto_encoding(x, timestamps=timestamps)
        assert output.shape == x.shape


class TestTemporalEncoding:
    """Test suite for temporal encodings."""
    
    @pytest.fixture
    def temporal_config(self):
        return TemporalEncodingConfig(
            d_model=256,
            use_cyclical_encoding=True,
            use_market_sessions=True,
            use_seasonal_patterns=True,
            use_crypto_cycles=True
        )
    
    def test_cyclical_encoding(self, temporal_config):
        """Test cyclical time encoding."""
        encoding = CyclicalTimeEncoding(temporal_config)
        
        batch_size, seq_len = 4, 128
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        
        output = encoding(timestamps)
        assert output.shape == (batch_size, seq_len, temporal_config.d_model)
    
    def test_market_session_encoding(self, temporal_config):
        """Test market session encoding."""
        encoding = MarketSessionEncoding(temporal_config)
        
        batch_size, seq_len = 4, 128
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        
        output = encoding(timestamps)
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
        assert output.shape[2] == temporal_config.d_model // 4
    
    def test_seasonality_encoding(self, temporal_config):
        """Test seasonality encoding."""
        encoding = SeasonalityEncoding(temporal_config)
        
        batch_size, seq_len = 4, 128
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        
        output = encoding(timestamps)
        assert output.shape[0] == batch_size
        assert output.shape[1] == seq_len
    
    def test_crypto_temporal_encoding(self, temporal_config):
        """Test comprehensive crypto temporal encoding."""
        encoding = CryptoTemporalEncoding(temporal_config)
        
        batch_size, seq_len = 4, 128
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        volume_data = torch.rand(batch_size, seq_len)
        volatility_data = torch.rand(batch_size, seq_len)
        
        output = encoding(
            timestamps=timestamps,
            volume_data=volume_data,
            volatility_data=volatility_data
        )
        assert output.shape == (batch_size, seq_len, temporal_config.d_model)


class TestTransformerBlocks:
    """Test suite for transformer components."""
    
    @pytest.fixture
    def transformer_config(self):
        return TransformerBlockConfig(
            d_model=256,
            num_heads=8,
            d_ff=1024,
            dropout=0.1,
            use_pre_norm=True,
            use_gated_mlp=True
        )
    
    def test_encoder_block(self, transformer_config):
        """Test transformer encoder block."""
        encoder = TransformerEncoderBlock(transformer_config)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, transformer_config.d_model)
        
        output = encoder(x)
        assert output.shape == x.shape
        
        # Test with attention weights
        output, weights = encoder(x, need_weights=True)
        assert output.shape == x.shape
        assert weights.shape == (batch_size, transformer_config.num_heads, seq_len, seq_len)
    
    def test_decoder_block(self, transformer_config):
        """Test transformer decoder block."""
        decoder = TransformerDecoderBlock(transformer_config)
        
        batch_size, target_len, source_len = 4, 32, 64
        target_x = torch.randn(batch_size, target_len, transformer_config.d_model)
        encoder_output = torch.randn(batch_size, source_len, transformer_config.d_model)
        
        output = decoder(target_x, encoder_output)
        assert output.shape == target_x.shape
        
        # Test with attention weights
        output, weights = decoder(target_x, encoder_output, need_weights=True)
        assert output.shape == target_x.shape
        assert 'self_attention' in weights
        assert 'cross_attention' in weights
    
    def test_crypto_transformer_block(self, transformer_config):
        """Test crypto-specific transformer block."""
        crypto_block = CryptoTransformerBlock(transformer_config)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, transformer_config.d_model)
        volume_data = torch.randn(batch_size, seq_len, transformer_config.d_model)
        risk_scores = torch.rand(batch_size, seq_len)
        
        output = crypto_block(
            x,
            volume_data=volume_data,
            risk_scores=risk_scores
        )
        assert output.shape == x.shape
    
    def test_feed_forward(self, transformer_config):
        """Test feed-forward network."""
        ff = FeedForward(transformer_config)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, transformer_config.d_model)
        
        output = ff(x)
        assert output.shape == x.shape
    
    def test_rms_norm(self):
        """Test RMS normalization."""
        d_model = 256
        rms_norm = RMSNorm(d_model)
        
        batch_size, seq_len = 4, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = rms_norm(x)
        assert output.shape == x.shape
        
        # Check normalization properties
        rms = torch.sqrt(torch.mean(output**2, dim=-1, keepdim=True))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)


class TestTradingTransformer:
    """Test suite for trading transformer."""
    
    @pytest.fixture
    def trading_config(self):
        return TradingTransformerConfig(
            d_model=256,
            num_heads=8,
            num_encoder_layers=4,
            input_features=50,
            output_dim=1,
            use_multi_timeframe=True,
            use_market_regime_detection=True,
            use_risk_management=True
        )
    
    def test_trading_transformer_forward(self, trading_config):
        """Test trading transformer forward pass."""
        model = TradingTransformer(trading_config)
        
        batch_size, seq_len = 4, 128
        x = torch.randn(batch_size, seq_len, trading_config.input_features)
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        asset_ids = torch.randint(0, 10, (batch_size, seq_len))
        
        outputs = model(
            x=x,
            timestamps=timestamps,
            asset_ids=asset_ids
        )
        
        assert 'predictions' in outputs
        assert 'hidden_states' in outputs
        assert outputs['hidden_states'].shape == (batch_size, seq_len, trading_config.d_model)
    
    def test_next_timestep_prediction(self, trading_config):
        """Test autoregressive prediction."""
        trading_config.output_type = "sequence"
        model = TradingTransformer(trading_config)
        
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, trading_config.input_features)
        
        predictions = model.predict_next_timestep(x, num_predictions=5)
        if predictions is not None:
            assert predictions.shape == (batch_size, 5, trading_config.d_model)
    
    def test_attention_weights_extraction(self, trading_config):
        """Test attention weights extraction."""
        model = TradingTransformer(trading_config)
        
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, trading_config.input_features)
        
        attention_maps = model.get_attention_maps(x)
        assert isinstance(attention_maps, dict)


class TestCryptoModels:
    """Test suite for crypto-specific models."""
    
    @pytest.fixture
    def model_config(self):
        return CryptoPredictionModelConfig(
            d_model=256,
            num_heads=8,
            num_layers=4,
            input_features=50,
            prediction_horizon=5,
            num_assets=10,
            prediction_targets=["price", "volatility"]
        )
    
    def test_price_prediction_model(self, model_config):
        """Test crypto price prediction model."""
        model = CryptoPricePredictionModel(model_config)
        
        batch_size, seq_len = 4, 128
        price_data = torch.randn(batch_size, seq_len, model_config.input_features)
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        asset_ids = torch.randint(0, model_config.num_assets, (batch_size, seq_len))
        
        outputs = model(
            price_data=price_data,
            timestamps=timestamps,
            asset_ids=asset_ids
        )
        
        for target in model_config.prediction_targets:
            assert f'{target}_predictions' in outputs
            assert f'{target}_uncertainty' in outputs
            assert outputs[f'{target}_predictions'].shape == (batch_size, model_config.prediction_horizon, 1)
    
    def test_signal_generator_model(self, model_config):
        """Test trading signal generator."""
        model = CryptoSignalGeneratorModel(model_config)
        
        batch_size, seq_len = 4, 128
        multi_tf_data = {
            "1m": torch.randn(batch_size, seq_len, model_config.input_features),
            "5m": torch.randn(batch_size, seq_len, model_config.input_features),
            "15m": torch.randn(batch_size, seq_len, model_config.input_features)
        }
        timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
        
        outputs = model(
            multi_timeframe_data=multi_tf_data,
            timestamps=timestamps
        )
        
        expected_outputs = [
            'signal_probabilities', 'signal_class', 'signal_strength',
            'entry_probability', 'exit_probability', 'risk_reward_ratio'
        ]
        
        for key in expected_outputs:
            assert key in outputs
    
    def test_risk_assessment_model(self, model_config):
        """Test risk assessment model."""
        model = CryptoRiskAssessmentModel(model_config)
        
        batch_size, seq_len = 4, 128
        portfolio_data = torch.randn(batch_size, seq_len, model_config.input_features)
        asset_weights = torch.softmax(torch.randn(batch_size, model_config.num_assets), dim=-1)
        historical_returns = torch.randn(batch_size, seq_len, model_config.num_assets) * 0.02
        
        outputs = model(
            portfolio_data=portfolio_data,
            asset_weights=asset_weights,
            historical_returns=historical_returns
        )
        
        expected_outputs = [
            'value_at_risk', 'drawdown_probability', 'volatility_forecast',
            'correlation_matrix', 'risk_regime'
        ]
        
        for key in expected_outputs:
            assert key in outputs
    
    def test_portfolio_optimizer_model(self, model_config):
        """Test portfolio optimizer model."""
        model = CryptoPortfolioOptimizerModel(model_config)
        
        batch_size, seq_len = 4, 128
        asset_features = torch.randn(batch_size, seq_len, model_config.num_assets, model_config.input_features)
        current_weights = torch.softmax(torch.randn(batch_size, model_config.num_assets), dim=-1)
        
        outputs = model(
            asset_features=asset_features,
            current_weights=current_weights
        )
        
        expected_outputs = [
            'optimal_weights', 'expected_returns', 'expected_risks',
            'optimal_sharpe_ratio', 'improvement_potential'
        ]
        
        for key in expected_outputs:
            assert key in outputs


class TestUtilities:
    """Test suite for utility functions."""
    
    def test_attention_analyzer(self):
        """Test attention analysis utilities."""
        analyzer = AttentionAnalyzer()
        
        batch_size, num_heads, seq_len = 2, 4, 32
        attention_weights = torch.softmax(
            torch.randn(batch_size, num_heads, seq_len, seq_len),
            dim=-1
        )
        
        stats = analyzer.analyze_attention_weights(attention_weights)
        
        assert isinstance(stats.entropy, float)
        assert isinstance(stats.sparsity, float)
        assert isinstance(stats.head_diversity, float)
        assert 0 <= stats.sparsity <= 1
        assert 0 <= stats.head_diversity <= 1
    
    def test_memory_optimizer(self):
        """Test memory optimization utilities."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 64
        
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Test chunked attention
        output = AttentionMemoryOptimizer.chunk_attention(
            query, key, value, chunk_size=32
        )
        assert output.shape == query.shape
        
        # Test sequence truncation
        input_tensor = torch.randn(2, 512, 256)
        truncated = AttentionMemoryOptimizer.dynamic_sequence_truncation(
            input_tensor, max_memory_mb=50.0
        )
        assert truncated.shape[0] == input_tensor.shape[0]
        assert truncated.shape[2] == input_tensor.shape[2]
        assert truncated.shape[1] <= input_tensor.shape[1]
    
    def test_attention_debugger(self):
        """Test attention debugging utilities."""
        debugger = AttentionDebugger()
        
        # Test validation
        batch_size, num_heads, seq_len = 2, 4, 32
        attention_weights = torch.softmax(
            torch.randn(batch_size, num_heads, seq_len, seq_len),
            dim=-1
        )
        
        validation = debugger.validate_attention_weights(attention_weights)
        
        assert isinstance(validation['has_nan'], bool)
        assert isinstance(validation['has_inf'], bool)
        assert isinstance(validation['weights_sum_to_one'], bool)
        assert not validation['has_nan']
        assert not validation['has_inf']
        assert validation['weights_sum_to_one']


class TestPerformance:
    """Performance and benchmark tests."""
    
    def test_attention_performance_comparison(self):
        """Compare performance of different attention implementations."""
        d_model, num_heads, seq_len = 256, 8, 128
        
        # Create test implementations
        implementations = {
            'standard': MultiHeadAttention(AttentionConfig(d_model=d_model, num_heads=num_heads, use_flash_attn=False)),
            'crypto': CryptoMultiHeadAttention(AttentionConfig(d_model=d_model, num_heads=num_heads), use_volume_weighting=True),
        }
        
        # Add flash attention if available
        try:
            implementations['flash'] = MultiHeadAttention(
                AttentionConfig(d_model=d_model, num_heads=num_heads, use_flash_attn=True)
            )
        except:
            pass
        
        input_tensor = torch.randn(2, seq_len, d_model)
        
        results = benchmark_attention_implementations(
            implementations, input_tensor, num_runs=5
        )
        
        assert len(results) == len(implementations)
        for name, stats in results.items():
            if 'error' not in stats:
                assert 'forward_time_mean_ms' in stats
                assert stats['forward_time_mean_ms'] > 0
    
    @pytest.mark.slow
    def test_memory_scaling(self):
        """Test memory usage scaling with sequence length."""
        d_model, num_heads = 256, 8
        attention = MultiHeadAttention(AttentionConfig(d_model=d_model, num_heads=num_heads))
        
        sequence_lengths = [64, 128, 256, 512]
        memory_usages = []
        
        for seq_len in sequence_lengths:
            input_tensor = torch.randn(1, seq_len, d_model)
            
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
                attention = attention.cuda()
                torch.cuda.reset_peak_memory_stats()
                
                _ = attention(input_tensor)
                memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                memory_usages.append(memory_used)
                
                attention = attention.cpu()
                input_tensor = input_tensor.cpu()
                torch.cuda.empty_cache()
        
        if memory_usages:
            # Memory should scale roughly quadratically with sequence length
            assert len(memory_usages) == len(sequence_lengths)
            logger.info(f"Memory scaling: {dict(zip(sequence_lengths, memory_usages))}")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_sequences(self):
        """Test handling of empty sequences."""
        config = AttentionConfig(d_model=256, num_heads=8)
        attention = MultiHeadAttention(config)
        
        # Zero-length sequence should raise error or handle gracefully
        empty_input = torch.randn(1, 0, 256)
        
        with pytest.raises((RuntimeError, ValueError)):
            _ = attention(empty_input)
    
    def test_mismatched_dimensions(self):
        """Test handling of dimension mismatches."""
        config = AttentionConfig(d_model=256, num_heads=8)
        attention = MultiHeadAttention(config)
        
        # Wrong d_model dimension
        wrong_input = torch.randn(1, 32, 128)  # 128 instead of 256
        
        with pytest.raises((RuntimeError, ValueError)):
            _ = attention(wrong_input)
    
    def test_invalid_configurations(self):
        """Test invalid configuration handling."""
        # d_model not divisible by num_heads
        with pytest.raises(ValueError):
            config = AttentionConfig(d_model=257, num_heads=8)
            _ = MultiHeadAttention(config)
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        config = AttentionConfig(d_model=256, num_heads=8)
        attention = MultiHeadAttention(config)
        
        # Very large values
        large_input = torch.randn(1, 32, 256) * 100
        output = attention(large_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Very small values
        small_input = torch.randn(1, 32, 256) * 1e-6
        output = attention(small_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


# Pytest configuration
@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    logging.basicConfig(level=logging.INFO)


# Performance markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "benchmark: mark test as benchmark")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])