"""
Cross-Attention mechanism for multi-modal analysis in crypto trading.
Allows models to account for interactions between different data sources.

Enterprise cross-attention for fusion of multi-modal financial data.
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
class CrossAttentionConfig(AttentionConfig):
    """Configuration for Cross-Attention mechanism."""
    # Cross-attention specific parameters
    use_symmetric_attention: bool = False  # Symmetric attention between modalities
    use_gated_fusion: bool = True  # Gated fusion for controlling information stream
    fusion_dropout: float = 0.1  # Dropout for fusion layer
    use_temperature_scaling: bool = True  # Temperature scaling for attention scores
    temperature: float = 1.0
    
    # Multi-modal parameters
    use_modality_embeddings: bool = True  # Embeddings for various modalities
    num_modalities: int = 3  # Number modalities (price, volume, news, etc.)
    modality_dim: int = 64  # Dimension modality embeddings
    
    # Cross-asset attention
    use_cross_asset: bool = True  # Cross-asset attention patterns
    num_assets: int = 10  # Number assets in portfolio
    asset_embedding_dim: int = 32
    
    # Temporal cross-attention
    use_temporal_cross: bool = True  # Cross-attention between temporal periods
    temporal_window_sizes: List[int] = None  # Sizes temporal windows
    
    def __post_init__(self):
        super().__post_init__()
        if self.temporal_window_sizes is None:
            self.temporal_window_sizes = [5, 15, 60, 240]  # 5m, 15m, 1h, 4h


class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for multi-modal analysis.
    
    Features:
    - Cross-modal attention between various types of data
    - Gated fusion for selective information integration
    - Temperature scaling for attention sharpening/smoothing
    - Symmetric attention (optional)
    - Modality embeddings for consideration type of data
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention cores
        self.query_attention = MultiHeadAttention(config)
        
        if config.use_symmetric_attention:
            self.key_attention = MultiHeadAttention(config)
        
        # Gated fusion mechanism
        if config.use_gated_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.Sigmoid()
            )
            self.fusion_dropout = nn.Dropout(config.fusion_dropout)
        
        # Temperature parameter for attention scaling
        if config.use_temperature_scaling:
            self.temperature = nn.Parameter(torch.tensor(config.temperature))
        
        # Modality embeddings
        if config.use_modality_embeddings:
            self.modality_embeddings = nn.Embedding(
                config.num_modalities, config.modality_dim
            )
            # Project modality embeddings to d_model
            self.modality_proj = nn.Linear(config.modality_dim, config.d_model)
        
        # Cross-asset embeddings
        if config.use_cross_asset:
            self.asset_embeddings = nn.Embedding(
                config.num_assets, config.asset_embedding_dim
            )
            self.asset_proj = nn.Linear(config.asset_embedding_dim, config.d_model)
        
        # Temporal cross-attention components
        if config.use_temporal_cross:
            self.temporal_projections = nn.ModuleDict({
                f'window_{size}': nn.Linear(config.d_model, config.d_model)
                for size in config.temporal_window_sizes
            })
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        query_modality: Optional[torch.Tensor] = None,
        key_modality: Optional[torch.Tensor] = None,
        asset_ids: Optional[torch.Tensor] = None,
        temporal_windows: Optional[Dict[str, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query modality [batch_size, query_len, d_model]
            key: Key modality [batch_size, key_len, d_model]
            value: Value modality (if None, key is used)
            query_modality: ID modalities for query [batch_size, query_len]
            key_modality: ID modalities for key [batch_size, key_len]
            asset_ids: Asset IDs [batch_size, seq_len]
            temporal_windows: Various temporal windows data
            attention_mask: Attention mask
            need_weights: Return attention weights
        """
        if value is None:
            value = key
        
        batch_size, query_len, d_model = query.shape
        key_len = key.shape[1]
        
        # Augment with modality embeddings
        if self.config.use_modality_embeddings and query_modality is not None:
            query_mod_emb = self.modality_proj(
                self.modality_embeddings(query_modality)
            )
            query = query + query_mod_emb
        
        if self.config.use_modality_embeddings and key_modality is not None:
            key_mod_emb = self.modality_proj(
                self.modality_embeddings(key_modality)
            )
            key = key + key_mod_emb
            if torch.equal(key, value):  # Value also needs to be updated
                value = key
        
        # Cross-asset embeddings
        if self.config.use_cross_asset and asset_ids is not None:
            asset_emb = self.asset_proj(self.asset_embeddings(asset_ids))
            query = query + asset_emb
            if key.shape[1] == query.shape[1]:  # If key is the same length
                key = key + asset_emb
                if torch.equal(key, value):
                    value = key
        
        # Temporal cross-attention
        temporal_outputs = []
        if self.config.use_temporal_cross and temporal_windows:
            for window_name, window_data in temporal_windows.items():
                if window_name.replace('window_', '') in [str(s) for s in self.config.temporal_window_sizes]:
                    proj_name = f'window_{window_name.split("_")[-1]}'
                    if proj_name in self.temporal_projections:
                        projected_data = self.temporal_projections[proj_name](window_data)
                        temporal_outputs.append(projected_data)
        
        # Main cross-attention
        cross_attn_output = self.query_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            need_weights=need_weights
        )
        
        if need_weights:
            cross_attn_output, attention_weights = cross_attn_output
        
        # Symmetric attention (if enabled)
        if self.config.use_symmetric_attention:
            reverse_attn_output = self.key_attention(
                query=key,
                key=query,
                value=query,
                attention_mask=attention_mask.transpose(-2, -1) if attention_mask is not None else None
            )
            
            # Combine outputs
            if reverse_attn_output.shape[1] == cross_attn_output.shape[1]:
                cross_attn_output = (cross_attn_output + reverse_attn_output) / 2
        
        # Gated fusion
        if self.config.use_gated_fusion:
            # Combine query and cross-attention output
            combined = torch.cat([query, cross_attn_output], dim=-1)
            gate = self.fusion_gate(combined)
            cross_attn_output = gate * cross_attn_output + (1 - gate) * query
            cross_attn_output = self.fusion_dropout(cross_attn_output)
        
        # Incorporate temporal information
        if temporal_outputs:
            temporal_combined = torch.stack(temporal_outputs, dim=0).mean(dim=0)
            if temporal_combined.shape[1] == cross_attn_output.shape[1]:
                cross_attn_output = cross_attn_output + temporal_combined
        
        # Temperature scaling
        if self.config.use_temperature_scaling and hasattr(self, 'temperature'):
            cross_attn_output = cross_attn_output / self.temperature
        
        if need_weights:
            return cross_attn_output, attention_weights
        return cross_attn_output


class CryptoCrossAttention(CrossAttention):
    """
    Specialized Cross-Attention for crypto trading strategies.
    
    Additional Features:
    - Price-Volume cross-attention
    - News-Price sentiment fusion
    - Order book cross-modal attention
    - Cross-exchange arbitrage patterns
    - Social media sentiment integration
    """
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__(config)
        
        # Crypto-specific cross-attention components
        self.price_volume_fusion = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # News sentiment cross-attention
        self.news_price_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads // 2,  # Fewer heads for news
            dropout=config.dropout,
            batch_first=True
        )
        
        # Order book cross-attention
        self.orderbook_price_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Cross-exchange attention
        self.exchange_weights = nn.Parameter(
            torch.ones(5) / 5  # 5 main exchanges
        )
        
        # Social sentiment fusion
        self.social_sentiment_proj = nn.Linear(64, config.d_model)  # Twitter/Reddit embeddings
        
        # Market regime aware fusion
        self.regime_gate = nn.Sequential(
            nn.Linear(config.d_model + 10, config.d_model),  # +10 for regime features
            nn.Tanh(),
            nn.Linear(config.d_model, config.d_model),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        price_data: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        news_embeddings: Optional[torch.Tensor] = None,
        orderbook_data: Optional[torch.Tensor] = None,
        social_sentiment: Optional[torch.Tensor] = None,
        market_regime: Optional[torch.Tensor] = None,
        exchange_data: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Crypto-specific cross-attention forward.
        
        Args:
            price_data: OHLCV price data [batch, seq_len, d_model]
            volume_data: Volume patterns [batch, seq_len, d_model] 
            news_embeddings: News sentiment embeddings [batch, news_len, d_model]
            orderbook_data: Order book features [batch, seq_len, d_model]
            social_sentiment: Social media sentiment [batch, seq_len, 64]
            market_regime: Market regime indicators [batch, seq_len, 10]
            exchange_data: Data from multiple exchanges
        """
        batch_size, seq_len, d_model = price_data.shape
        outputs = [price_data]  # Base price data
        
        # Price-Volume cross-attention
        if volume_data is not None:
            pv_output, _ = self.price_volume_fusion(
                query=price_data,
                key=volume_data,
                value=volume_data
            )
            outputs.append(pv_output)
        
        # News-Price cross-attention
        if news_embeddings is not None:
            # Align news sequence length with price sequence
            if news_embeddings.shape[1] != seq_len:
                # Interpolate or repeat news embeddings
                news_embeddings = F.interpolate(
                    news_embeddings.transpose(1, 2),
                    size=seq_len,
                    mode='linear'
                ).transpose(1, 2)
            
            news_output, _ = self.news_price_attention(
                query=price_data,
                key=news_embeddings,
                value=news_embeddings
            )
            outputs.append(news_output * 0.5)  # Lower weight for news
        
        # Order book cross-attention
        if orderbook_data is not None:
            ob_output, _ = self.orderbook_price_attention(
                query=price_data,
                key=orderbook_data,
                value=orderbook_data
            )
            outputs.append(ob_output)
        
        # Social sentiment integration
        if social_sentiment is not None:
            social_features = self.social_sentiment_proj(social_sentiment)
            outputs.append(social_features * 0.3)  # Lower weight
        
        # Cross-exchange attention
        if exchange_data is not None:
            exchange_outputs = []
            for i, (exchange_name, exchange_features) in enumerate(exchange_data.items()):
                if i < len(self.exchange_weights):
                    weighted_features = exchange_features * self.exchange_weights[i]
                    exchange_outputs.append(weighted_features)
            
            if exchange_outputs:
                cross_exchange_output = torch.stack(exchange_outputs, dim=0).sum(dim=0)
                outputs.append(cross_exchange_output)
        
        # Combine all cross-attention outputs
        combined_output = torch.stack(outputs, dim=0).mean(dim=0)
        
        # Market regime aware gating
        if market_regime is not None:
            regime_input = torch.cat([combined_output, market_regime], dim=-1)
            regime_gate_weight = self.regime_gate(regime_input)
            combined_output = combined_output * regime_gate_weight
        
        # Apply base cross-attention
        final_output = super().forward(
            query=price_data,
            key=combined_output,
            value=combined_output,
            **kwargs
        )
        
        return final_output


class MultiModalCrossAttention(nn.Module):
    """
    Multi-modal cross-attention for fusion of multiple data types simultaneously.
    
    Features:
    - Simultaneous cross-attention between N modalities
    - Learnable fusion weights
    - Modality-specific transformations
    - Hierarchical attention (coarse -> fine)
    """
    
    def __init__(self, config: CrossAttentionConfig, modalities: List[str]):
        super().__init__()
        self.config = config
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        # Cross-attention for each pair of modalities
        self.cross_attentions = nn.ModuleDict()
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:  # Skip self-attention
                    self.cross_attentions[f"{mod1}_to_{mod2}"] = CrossAttention(config)
        
        # Fusion weights for combining results
        self.fusion_weights = nn.Parameter(
            torch.ones(self.num_modalities) / self.num_modalities
        )
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(config.d_model, config.d_model)
            for modality in modalities
        })
        
        # Final fusion layer
        self.final_fusion = nn.Sequential(
            nn.Linear(config.d_model * self.num_modalities, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.fusion_dropout)
        )
    
    def forward(
        self,
        modality_data: Dict[str, torch.Tensor],
        attention_masks: Optional[Dict[str, torch.Tensor]] = None,
        need_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Multi-modal cross-attention forward.
        
        Args:
            modality_data: Dictionary with data for each modality
                          {"price": tensor, "volume": tensor, "news": tensor, ...}
            attention_masks: Masks for each modalities (optional)
            need_weights: Return attention weights
        """
        if attention_masks is None:
            attention_masks = {}
        
        # Apply modality-specific projections
        projected_data = {}
        for modality, data in modality_data.items():
            if modality in self.modality_projections:
                projected_data[modality] = self.modality_projections[modality](data)
            else:
                projected_data[modality] = data
        
        # Cross-attention between all pairs of modalities
        cross_attention_outputs = {}
        attention_weights_dict = {}
        
        for i, query_mod in enumerate(self.modalities):
            if query_mod not in projected_data:
                continue
                
            query_data = projected_data[query_mod]
            modality_outputs = [query_data]  # Include self
            
            for j, key_mod in enumerate(self.modalities):
                if i != j and key_mod in projected_data:
                    attention_key = f"{query_mod}_to_{key_mod}"
                    
                    if attention_key in self.cross_attentions:
                        key_data = projected_data[key_mod]
                        mask = attention_masks.get(f"{query_mod}_{key_mod}")
                        
                        cross_output = self.cross_attentions[attention_key](
                            query=query_data,
                            key=key_data,
                            attention_mask=mask,
                            need_weights=need_weights
                        )
                        
                        if need_weights:
                            cross_output, attn_weights = cross_output
                            attention_weights_dict[attention_key] = attn_weights
                        
                        modality_outputs.append(cross_output)
            
            # Combine outputs for a given query modality
            if len(modality_outputs) > 1:
                combined = torch.stack(modality_outputs, dim=0)
                # Weighted combination
                weights = F.softmax(self.fusion_weights[:len(modality_outputs)], dim=0)
                cross_attention_outputs[query_mod] = torch.sum(
                    combined * weights.view(-1, 1, 1, 1), dim=0
                )
            else:
                cross_attention_outputs[query_mod] = modality_outputs[0]
        
        # Final fusion of all modalities
        if cross_attention_outputs:
            all_outputs = []
            for modality in self.modalities:
                if modality in cross_attention_outputs:
                    all_outputs.append(cross_attention_outputs[modality])
            
            if all_outputs:
                concatenated = torch.cat(all_outputs, dim=-1)
                final_output = self.final_fusion(concatenated)
            else:
                # Fallback to the first available modality
                final_output = next(iter(modality_data.values()))
        else:
            final_output = next(iter(modality_data.values()))
        
        if need_weights:
            return final_output, attention_weights_dict
        return final_output


def create_cross_attention_layer(
    d_model: int,
    num_heads: int,
    attention_type: str = "standard",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating various types of cross-attention.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads  
        attention_type: Type ("standard", "crypto", "multimodal")
        **kwargs: Additional configuration
    """
    base_config = CrossAttentionConfig(
        d_model=d_model,
        num_heads=num_heads,
        **kwargs
    )
    
    if attention_type == "standard":
        return CrossAttention(base_config)
    elif attention_type == "crypto":
        return CryptoCrossAttention(base_config)
    elif attention_type == "multimodal":
        modalities = kwargs.get('modalities', ['price', 'volume', 'news'])
        return MultiModalCrossAttention(base_config, modalities)
    else:
        raise ValueError(f"Unknown cross-attention type: {attention_type}")


if __name__ == "__main__":
    # Test configurations
    config = CrossAttentionConfig(
        d_model=512,
        num_heads=8,
        dropout=0.1,
        use_gated_fusion=True,
        use_symmetric_attention=True,
        use_modality_embeddings=True,
        num_modalities=3
    )
    
    batch_size, seq_len = 4, 256
    
    # Test standard cross-attention
    cross_attn = CrossAttention(config)
    
    query = torch.randn(batch_size, seq_len, config.d_model)  # Price data
    key = torch.randn(batch_size, seq_len, config.d_model)    # Volume data
    
    query_modality = torch.zeros(batch_size, seq_len, dtype=torch.long)  # Price modality
    key_modality = torch.ones(batch_size, seq_len, dtype=torch.long)     # Volume modality
    
    output = cross_attn(
        query=query,
        key=key,
        query_modality=query_modality,
        key_modality=key_modality
    )
    print(f"Standard Cross-Attention output: {output.shape}")
    
    # Test crypto cross-attention
    crypto_config = CrossAttentionConfig(
        d_model=512,
        num_heads=8,
        use_cross_asset=True,
        num_assets=5
    )
    crypto_cross_attn = CryptoCrossAttention(crypto_config)
    
    price_data = torch.randn(batch_size, seq_len, config.d_model)
    volume_data = torch.randn(batch_size, seq_len, config.d_model)
    news_embeddings = torch.randn(batch_size, 50, config.d_model)  # News sequence
    orderbook = torch.randn(batch_size, seq_len, config.d_model)
    
    crypto_output = crypto_cross_attn(
        price_data=price_data,
        volume_data=volume_data,
        news_embeddings=news_embeddings,
        orderbook_data=orderbook
    )
    print(f"Crypto Cross-Attention output: {crypto_output.shape}")
    
    # Test multi-modal cross-attention
    modalities = ['price', 'volume', 'news', 'sentiment']
    multi_modal = MultiModalCrossAttention(config, modalities)
    
    modality_data = {
        'price': torch.randn(batch_size, seq_len, config.d_model),
        'volume': torch.randn(batch_size, seq_len, config.d_model),
        'news': torch.randn(batch_size, seq_len, config.d_model),
        'sentiment': torch.randn(batch_size, seq_len, config.d_model)
    }
    
    multimodal_output = multi_modal(modality_data)
    print(f"Multi-modal Cross-Attention output: {multimodal_output.shape}")
    
    print(f"\nModel Parameters:")
    print(f"Standard: {sum(p.numel() for p in cross_attn.parameters())}")
    print(f"Crypto: {sum(p.numel() for p in crypto_cross_attn.parameters())}")
    print(f"Multi-modal: {sum(p.numel() for p in multi_modal.parameters())}")