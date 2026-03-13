"""
Trading Transformer architecture specifically designed for crypto trading tasks.
Combines attention mechanisms with domain-specific optimizations for financial time series.

Production trading transformer with real-time inference and risk management.
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

# Import components
from .transformer_block import TransformerEncoderBlock, TransformerDecoderBlock, TransformerBlockConfig
from ..attention.temporal_attention import TemporalAttention, TemporalAttentionConfig
from ..attention.cross_attention import MultiModalCrossAttention, CrossAttentionConfig
from ..encodings.positional_encoding import create_positional_encoding, PositionalEncodingConfig
from ..encodings.temporal_encoding import CryptoTemporalEncoding, TemporalEncodingConfig

logger = logging.getLogger(__name__)


@dataclass
class TradingTransformerConfig:
    """Configuration for Trading Transformer."""
    # Model architecture
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Input specifications
    vocab_size: Optional[int] = None  # For tokenized inputs
    max_seq_len: int = 2048
    input_features: int = 100  # Number of input features (OHLCV, indicators, etc.)
    
    # Output specifications
    output_dim: int = 1  # Prediction dimension (price, return, signal)
    output_type: str = "regression"  # "regression", "classification", "sequence"
    num_classes: Optional[int] = None  # For classification
    
    # Trading-specific features
    use_multi_timeframe: bool = True
    timeframes: List[str] = None  # ["1m", "5m", "15m", "1h", "4h"]
    
    use_multi_asset: bool = True
    num_assets: int = 50
    
    use_market_regime_detection: bool = True
    num_regimes: int = 5
    
    use_risk_management: bool = True
    risk_features: int = 20
    
    # Advanced features
    use_temporal_attention: bool = True
    use_cross_modal_attention: bool = True
    use_causal_inference: bool = False  # For autoregressive generation
    
    # Positional encoding
    pos_encoding_type: str = "crypto"  # "sinusoidal", "learned", "crypto", "temporal"
    
    # Optimization
    use_gradient_checkpointing: bool = False
    use_mixed_precision: bool = True
    use_layer_scale: bool = False  # Layer scaling for training stability
    
    # Regularization
    stochastic_depth_rate: float = 0.0
    attention_dropout: float = 0.1
    path_dropout: float = 0.1
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h", "4h"]
        
        if self.output_type == "classification" and self.num_classes is None:
            self.num_classes = 3  # Buy/Hold/Sell


class InputEmbedding(nn.Module):
    """Input embedding layer for trading data."""
    
    def __init__(self, config: TradingTransformerConfig):
        super().__init__()
        self.config = config
        
        # Feature embedding
        self.feature_embedding = nn.Linear(config.input_features, config.d_model)
        
        # Multi-asset embeddings
        if config.use_multi_asset:
            self.asset_embedding = nn.Embedding(config.num_assets, config.d_model // 8)
        
        # Multi-timeframe embeddings
        if config.use_multi_timeframe:
            self.timeframe_embedding = nn.Embedding(len(config.timeframes), config.d_model // 16)
        
        # Feature type embeddings (price, volume, technical indicators, etc.)
        self.feature_type_embedding = nn.Embedding(10, config.d_model // 16)
        
        # Combination layer
        embed_dim = config.d_model
        if config.use_multi_asset:
            embed_dim += config.d_model // 8
        if config.use_multi_timeframe:
            embed_dim += config.d_model // 16
        embed_dim += config.d_model // 16  # feature type
        
        if embed_dim > config.d_model:
            self.projection = nn.Linear(embed_dim, config.d_model)
        else:
            self.projection = nn.Identity()
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        feature_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed input features.
        
        Args:
            x: Input features [batch_size, seq_len, input_features]
            asset_ids: Asset identifiers [batch_size, seq_len]
            timeframe_ids: Timeframe identifiers [batch_size, seq_len]
            feature_type_ids: Feature type identifiers [batch_size, seq_len]
        """
        # Feature embedding
        embedded = self.feature_embedding(x)
        embeddings = [embedded]
        
        # Asset embeddings
        if self.config.use_multi_asset and asset_ids is not None:
            asset_emb = self.asset_embedding(asset_ids)
            embeddings.append(asset_emb)
        
        # Timeframe embeddings
        if self.config.use_multi_timeframe and timeframe_ids is not None:
            timeframe_emb = self.timeframe_embedding(timeframe_ids)
            embeddings.append(timeframe_emb)
        
        # Feature type embeddings
        if feature_type_ids is not None:
            feature_type_emb = self.feature_type_embedding(feature_type_ids)
        else:
            # Default feature type
            feature_type_ids = torch.zeros(x.shape[:2], dtype=torch.long, device=x.device)
            feature_type_emb = self.feature_type_embedding(feature_type_ids)
        embeddings.append(feature_type_emb)
        
        # Combine embeddings
        if len(embeddings) > 1:
            combined = torch.cat(embeddings, dim=-1)
            embedded = self.projection(combined)
        else:
            embedded = embeddings[0]
        
        # Normalization and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class MarketRegimeDetector(nn.Module):
    """Market regime detection module."""
    
    def __init__(self, config: TradingTransformerConfig):
        super().__init__()
        self.config = config
        
        # Regime detection network
        self.detector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, config.num_regimes),
            nn.Softmax(dim=-1)
        )
        
        # Regime embeddings
        self.regime_embedding = nn.Embedding(config.num_regimes, config.d_model // 8)
        
        # Regime-specific transformations
        self.regime_transforms = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.num_regimes)
        ])
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.num_regimes, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect market regime and apply regime-specific processing.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            
        Returns:
            processed_x: Regime-adapted features [batch_size, seq_len, d_model]
            regime_probs: Regime probabilities [batch_size, seq_len, num_regimes]
            confidence: Detection confidence [batch_size, seq_len, 1]
        """
        # Detect regimes
        regime_probs = self.detector(x)
        
        # Get most likely regime
        regime_ids = torch.argmax(regime_probs, dim=-1)
        
        # Apply regime-specific transformations
        regime_adapted = torch.zeros_like(x)
        for regime_id in range(self.config.num_regimes):
            # Mask for current regime
            regime_mask = (regime_ids == regime_id).unsqueeze(-1).float()
            
            # Apply regime-specific transform
            transformed = self.regime_transforms[regime_id](x)
            
            # Accumulate weighted by regime probability
            regime_weight = regime_probs[:, :, regime_id:regime_id+1]
            regime_adapted += transformed * regime_mask * regime_weight
        
        # Blend with original features
        processed_x = x + regime_adapted * 0.2  # Moderate influence
        
        # Estimate confidence
        confidence = self.confidence_estimator(regime_probs)
        
        return processed_x, regime_probs, confidence


class RiskAwareHead(nn.Module):
    """Risk-aware output head for trading predictions."""
    
    def __init__(self, config: TradingTransformerConfig):
        super().__init__()
        self.config = config
        
        # Main prediction head
        if config.output_type == "regression":
            self.prediction_head = nn.Linear(config.d_model, config.output_dim)
        elif config.output_type == "classification":
            self.prediction_head = nn.Linear(config.d_model, config.num_classes)
        else:  # sequence
            self.prediction_head = nn.Linear(config.d_model, config.d_model)
        
        # Risk estimation
        if config.use_risk_management:
            self.risk_estimator = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, 1),
                nn.Sigmoid()  # Risk score 0-1
            )
            
            # Uncertainty estimation
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, config.output_dim),
                nn.Softplus()  # Always positive
            )
            
            # Risk-adjusted prediction
            self.risk_adjustment = nn.Sequential(
                nn.Linear(config.d_model + 1, config.d_model),  # +1 for risk score
                nn.Tanh()
            )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate risk-aware predictions.
        
        Args:
            x: Features [batch_size, seq_len, d_model]
            
        Returns:
            outputs: Dictionary with predictions, risk scores, uncertainty
        """
        outputs = {}
        
        # Base prediction
        if self.config.output_type == "sequence":
            # For sequence output, return full hidden states
            outputs['predictions'] = self.prediction_head(x)
        else:
            # For regression/classification, use last timestep or global pooling
            if x.dim() == 3:  # [batch, seq, dim]
                # Use last timestep
                last_hidden = x[:, -1, :]
            else:  # [batch, dim]
                last_hidden = x
            
            outputs['predictions'] = self.prediction_head(last_hidden)
        
        # Risk management
        if self.config.use_risk_management:
            if x.dim() == 3:
                risk_input = x[:, -1, :]  # Last timestep for risk
            else:
                risk_input = x
            
            # Risk score
            risk_score = self.risk_estimator(risk_input)
            outputs['risk_score'] = risk_score
            
            # Uncertainty estimation
            uncertainty = self.uncertainty_estimator(risk_input)
            outputs['uncertainty'] = uncertainty
            
            # Risk-adjusted prediction
            risk_adjusted_input = torch.cat([risk_input, risk_score], dim=-1)
            risk_adjustment = self.risk_adjustment(risk_adjusted_input)
            
            # Apply risk adjustment to predictions
            if self.config.output_type != "sequence":
                adjusted_features = risk_input + risk_adjustment * 0.1
                outputs['risk_adjusted_predictions'] = self.prediction_head(adjusted_features)
        
        return outputs


class TradingTransformer(nn.Module):
    """
    Complete Trading Transformer architecture.
    
    Features:
    - Multi-modal input processing
    - Temporal and cross-attention mechanisms
    - Market regime detection
    - Risk-aware predictions
    - Multi-timeframe processing
    - Real-time inference support
    """
    
    def __init__(self, config: TradingTransformerConfig):
        super().__init__()
        self.config = config
        
        # Input embedding
        self.input_embedding = InputEmbedding(config)
        
        # Positional encoding
        pos_config = PositionalEncodingConfig(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            encoding_type=config.pos_encoding_type,
            dropout=config.dropout
        )
        
        if config.pos_encoding_type == "crypto":
            temporal_config = TemporalEncodingConfig(
                d_model=config.d_model,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout
            )
            self.pos_encoding = CryptoTemporalEncoding(temporal_config)
        else:
            self.pos_encoding = create_positional_encoding(pos_config)
        
        # Market regime detection
        if config.use_market_regime_detection:
            self.regime_detector = MarketRegimeDetector(config)
        
        # Encoder layers
        encoder_config = TransformerBlockConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            stochastic_depth_rate=config.stochastic_depth_rate,
            use_gradient_checkpointing=config.use_gradient_checkpointing
        )
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(encoder_config)
            for _ in range(config.num_encoder_layers)
        ])
        
        # Temporal attention
        if config.use_temporal_attention:
            temporal_attention_config = TemporalAttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                use_multi_timeframe=config.use_multi_timeframe,
                timeframe_windows=[1, 5, 15, 60, 240]
            )
            self.temporal_attention = TemporalAttention(temporal_attention_config)
        
        # Cross-modal attention
        if config.use_cross_modal_attention:
            cross_attention_config = CrossAttentionConfig(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            modalities = ['price', 'volume', 'indicators', 'news']
            self.cross_modal_attention = MultiModalCrossAttention(
                cross_attention_config, modalities
            )
        
        # Decoder layers (for autoregressive tasks)
        if config.use_causal_inference:
            self.decoder_layers = nn.ModuleList([
                TransformerDecoderBlock(encoder_config)
                for _ in range(config.num_decoder_layers)
            ])
        
        # Output head
        self.output_head = RiskAwareHead(config)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Layer scaling for training stability
        if config.use_layer_scale:
            self.layer_scales = nn.ParameterList([
                nn.Parameter(torch.ones(config.d_model) * 0.1)
                for _ in range(config.num_encoder_layers)
            ])
        
        # Initialize weights
        self._init_weights()
        
        # Mixed precision support
        if config.use_mixed_precision:
            self.enable_mixed_precision()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def enable_mixed_precision(self):
        """Enable mixed precision training."""
        # This would be handled by training loop with GradScaler
        pass
    
    def forward(
        self,
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        asset_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None,
        modality_data: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        target_sequence: Optional[torch.Tensor] = None,  # For decoder
        need_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Trading Transformer forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_features]
            timestamps: Unix timestamps [batch_size, seq_len]
            asset_ids: Asset identifiers [batch_size, seq_len]
            timeframe_ids: Timeframe identifiers [batch_size, seq_len]
            modality_data: Multi-modal data dictionary
            attention_mask: Attention mask
            target_sequence: Target sequence for decoder
            need_attention_weights: Return attention weights
        """
        outputs = {}
        attention_weights = {}
        
        # Input embedding
        embedded = self.input_embedding(
            x, asset_ids=asset_ids, timeframe_ids=timeframe_ids
        )
        
        # Positional encoding
        if hasattr(self.pos_encoding, 'forward') and timestamps is not None:
            if isinstance(self.pos_encoding, CryptoTemporalEncoding):
                pos_encoded = embedded + self.pos_encoding(timestamps)
            else:
                pos_encoded = self.pos_encoding(embedded)
        else:
            pos_encoded = embedded
        
        # Market regime detection
        regime_info = None
        if self.config.use_market_regime_detection:
            pos_encoded, regime_probs, regime_confidence = self.regime_detector(pos_encoded)
            regime_info = {
                'regime_probabilities': regime_probs,
                'regime_confidence': regime_confidence
            }
            outputs.update(regime_info)
        
        # Encoder layers
        hidden_states = pos_encoded
        encoder_attention_weights = []
        
        for i, encoder_layer in enumerate(self.encoder_layers):
            layer_output = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                need_weights=need_attention_weights
            )
            
            if need_attention_weights:
                hidden_states, layer_attention = layer_output
                encoder_attention_weights.append(layer_attention)
            else:
                hidden_states = layer_output
            
            # Layer scaling
            if self.config.use_layer_scale:
                hidden_states = hidden_states * self.layer_scales[i]
        
        if need_attention_weights:
            attention_weights['encoder'] = encoder_attention_weights
        
        # Temporal attention
        if self.config.use_temporal_attention:
            temporal_output = self.temporal_attention(
                hidden_states,
                timestamps=timestamps,
                need_weights=need_attention_weights
            )
            
            if need_attention_weights:
                hidden_states, temporal_weights = temporal_output
                attention_weights['temporal'] = temporal_weights
            else:
                hidden_states = temporal_output
        
        # Cross-modal attention
        if self.config.use_cross_modal_attention and modality_data:
            # Prepare modality data with current hidden states as base
            enhanced_modality_data = {**modality_data, 'base': hidden_states}
            
            cross_modal_output = self.cross_modal_attention(
                enhanced_modality_data,
                need_weights=need_attention_weights
            )
            
            if need_attention_weights:
                hidden_states, cross_modal_weights = cross_modal_output
                attention_weights['cross_modal'] = cross_modal_weights
            else:
                hidden_states = cross_modal_output
        
        # Decoder (for autoregressive tasks)
        if self.config.use_causal_inference and target_sequence is not None:
            decoder_hidden = target_sequence
            
            for decoder_layer in self.decoder_layers:
                decoder_output = decoder_layer(
                    decoder_hidden,
                    encoder_output=hidden_states,
                    need_weights=need_attention_weights
                )
                
                if need_attention_weights:
                    decoder_hidden, decoder_attention = decoder_output
                    if 'decoder' not in attention_weights:
                        attention_weights['decoder'] = []
                    attention_weights['decoder'].append(decoder_attention)
                else:
                    decoder_hidden = decoder_output
            
            # Use decoder output for predictions
            final_features = self.final_norm(decoder_hidden)
        else:
            # Use encoder output
            final_features = self.final_norm(hidden_states)
        
        # Output predictions
        prediction_outputs = self.output_head(final_features)
        outputs.update(prediction_outputs)
        
        # Hidden states for analysis
        outputs['hidden_states'] = final_features
        
        if need_attention_weights:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def predict_next_timestep(
        self,
        x: torch.Tensor,
        num_predictions: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict following timesteps (for autoregressive generation).
        
        Args:
            x: Input sequence [batch_size, seq_len, input_features]
            num_predictions: Number of timesteps to predict
        """
        self.eval()
        predictions = []
        
        current_input = x
        
        with torch.no_grad():
            for _ in range(num_predictions):
                # Forward pass
                outputs = self.forward(current_input, **kwargs)
                
                # Get prediction (last timestep)
                if self.config.output_type == "sequence":
                    next_pred = outputs['predictions'][:, -1:, :]
                else:
                    # Convert to sequence format
                    next_pred = outputs['predictions'].unsqueeze(1)
                
                predictions.append(next_pred)
                
                # Update input for next prediction
                if self.config.output_type == "sequence":
                    # Append prediction to sequence
                    current_input = torch.cat([current_input[:, 1:, :], next_pred], dim=1)
                else:
                    # For non-sequence, this needs domain-specific logic
                    break
        
        return torch.cat(predictions, dim=1) if predictions else None
    
    def get_attention_maps(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Get attention maps for visualization."""
        outputs = self.forward(x, need_attention_weights=True, **kwargs)
        return outputs.get('attention_weights', {})


def create_trading_transformer(
    input_features: int,
    output_dim: int = 1,
    **kwargs
) -> TradingTransformer:
    """
    Factory function for creation trading transformer.
    
    Args:
        input_features: Number of input features
        output_dim: Output dimension
        **kwargs: Additional configuration parameters
    """
    config = TradingTransformerConfig(
        input_features=input_features,
        output_dim=output_dim,
        **kwargs
    )
    
    return TradingTransformer(config)


if __name__ == "__main__":
    # Test trading transformer
    config = TradingTransformerConfig(
        d_model=512,
        num_heads=8,
        num_encoder_layers=4,
        input_features=50,  # OHLCV + indicators
        output_dim=1,       # Price prediction
        max_seq_len=256,
        use_multi_timeframe=True,
        use_multi_asset=True,
        use_market_regime_detection=True,
        use_risk_management=True,
        use_temporal_attention=True,
        use_cross_modal_attention=True
    )
    
    # Create model
    model = TradingTransformer(config)
    
    batch_size, seq_len = 4, 128
    
    # Test data
    x = torch.randn(batch_size, seq_len, config.input_features)
    timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
    asset_ids = torch.randint(0, config.num_assets, (batch_size, seq_len))
    timeframe_ids = torch.randint(0, len(config.timeframes), (batch_size, seq_len))
    
    # Multi-modal data
    modality_data = {
        'price': torch.randn(batch_size, seq_len, config.d_model),
        'volume': torch.randn(batch_size, seq_len, config.d_model),
        'indicators': torch.randn(batch_size, seq_len, config.d_model)
    }
    
    # Forward pass
    outputs = model(
        x=x,
        timestamps=timestamps,
        asset_ids=asset_ids,
        timeframe_ids=timeframe_ids,
        modality_data=modality_data
    )
    
    print(f"Trading Transformer Results:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Test with attention weights
    print(f"\nTesting attention weights:")
    attention_outputs = model(
        x=x,
        timestamps=timestamps,
        need_attention_weights=True
    )
    
    if 'attention_weights' in attention_outputs:
        attn_weights = attention_outputs['attention_weights']
        for layer_type, weights in attn_weights.items():
            if isinstance(weights, list):
                print(f"  {layer_type}: {len(weights)} layers")
            else:
                print(f"  {layer_type}: {type(weights)}")
    
    # Test autoregressive prediction
    print(f"\nTesting next timestep prediction:")
    next_preds = model.predict_next_timestep(
        x[:, :64, :],  # Shorter sequence
        num_predictions=5,
        timestamps=timestamps[:, :64]
    )
    
    if next_preds is not None:
        print(f"Next timestep predictions: {next_preds.shape}")
    
    # Model statistics
    print(f"\nModel Statistics:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test different configurations
    print(f"\nTesting different configurations:")
    
    # Classification task
    cls_config = TradingTransformerConfig(
        input_features=50,
        output_type="classification",
        num_classes=3,
        d_model=256,
        num_heads=4,
        num_encoder_layers=3
    )
    
    cls_model = TradingTransformer(cls_config)
    cls_outputs = cls_model(x[:, :, :50])
    print(f"Classification model output: {cls_outputs['predictions'].shape}")
    
    # Sequence-to-sequence task
    seq_config = TradingTransformerConfig(
        input_features=50,
        output_type="sequence",
        d_model=256,
        num_heads=4,
        use_causal_inference=True,
        num_decoder_layers=2
    )
    
    seq_model = TradingTransformer(seq_config)
    seq_outputs = seq_model(
        x[:, :, :50],
        target_sequence=torch.randn(batch_size, seq_len//2, seq_config.d_model)
    )
    print(f"Sequence model output: {seq_outputs['predictions'].shape}")
    
    print(f"\n✅ Trading Transformer implementation complete!")