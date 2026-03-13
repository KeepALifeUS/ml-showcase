"""
Learnable Encoding implementations for adaptive positional encoding.
Allows the model to learn optimal representations for crypto trading patterns.

Adaptive learnable encodings with curriculum learning for trading optimization.
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
class LearnableEncodingConfig:
    """Configuration for learnable encoding."""
    d_model: int = 512
    max_seq_len: int = 10000
    dropout: float = 0.1
    
    # Learnable embedding parameters
    use_learned_positions: bool = True
    use_learned_time_embeddings: bool = True
    use_adaptive_embeddings: bool = True
    
    # Initialization
    init_type: str = "normal"  # "normal", "uniform", "xavier", "kaiming"
    init_std: float = 0.02
    init_range: float = 0.1
    
    # Advanced learnable features
    use_hierarchical_embeddings: bool = False  # Multi-scale position embeddings
    hierarchy_levels: int = 3
    
    use_context_aware_embeddings: bool = False  # Context-dependent position embeddings
    context_dim: int = 64
    
    use_curriculum_learning: bool = False  # Curriculum learning for position complexity
    curriculum_stages: int = 5
    
    # Regularization
    use_position_dropout: bool = True
    position_dropout: float = 0.1
    
    use_embedding_regularization: bool = False
    reg_strength: float = 0.01
    
    # Optimization-specific
    use_embedding_scaling: bool = True
    scale_factor: Optional[float] = None
    
    # Crypto-specific learnable features
    use_market_regime_embeddings: bool = True
    num_market_regimes: int = 5  # Bull, Bear, Sideways, High Vol, Low Vol
    
    use_asset_embeddings: bool = True
    num_assets: int = 50  # Number of crypto assets
    asset_embedding_dim: int = 32


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embeddings with advanced features."""
    
    def __init__(self, config: LearnableEncodingConfig):
        super().__init__()
        self.config = config
        
        # Main position embeddings
        if config.use_learned_positions:
            self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Time-based learnable embeddings
        if config.use_learned_time_embeddings:
            # Different granularities
            self.minute_embeddings = nn.Embedding(60, config.d_model // 16)
            self.hour_embeddings = nn.Embedding(24, config.d_model // 8)
            self.day_embeddings = nn.Embedding(7, config.d_model // 8)
            self.month_embeddings = nn.Embedding(12, config.d_model // 16)
            
            # Time combination layer
            time_features_dim = config.d_model // 16 * 2 + config.d_model // 8 * 2
            self.time_combination = nn.Linear(time_features_dim, config.d_model // 4)
        
        # Adaptive embeddings
        if config.use_adaptive_embeddings:
            self.adaptive_layer = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.d_model)
            )
        
        # Hierarchical embeddings
        if config.use_hierarchical_embeddings:
            self.hierarchy_embeddings = nn.ModuleList([
                nn.Embedding(config.max_seq_len // (2**i), config.d_model // config.hierarchy_levels)
                for i in range(config.hierarchy_levels)
            ])
            self.hierarchy_combination = nn.Linear(config.d_model, config.d_model)
        
        # Context-aware embeddings
        if config.use_context_aware_embeddings:
            self.context_proj = nn.Linear(config.context_dim, config.d_model)
            self.context_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.Sigmoid()
            )
        
        # Regularization
        if config.use_position_dropout:
            self.position_dropout = nn.Dropout(config.position_dropout)
        
        # Scaling
        if config.use_embedding_scaling:
            if config.scale_factor is None:
                self.scale_factor = math.sqrt(config.d_model)
            else:
                self.scale_factor = config.scale_factor
        else:
            self.scale_factor = 1.0
        
        # Initialize weights
        self._init_weights()
        
        # Curriculum learning state
        if config.use_curriculum_learning:
            self.current_stage = 0
            self.stage_boundaries = torch.linspace(
                0, config.max_seq_len, config.curriculum_stages + 1
            ).long()
    
    def _init_weights(self):
        """Initialize embeddings based on configuration."""
        modules_to_init = []
        
        if hasattr(self, 'position_embeddings'):
            modules_to_init.append(self.position_embeddings)
        
        if self.config.use_learned_time_embeddings:
            modules_to_init.extend([
                self.minute_embeddings, self.hour_embeddings,
                self.day_embeddings, self.month_embeddings
            ])
        
        if self.config.use_hierarchical_embeddings:
            modules_to_init.extend(self.hierarchy_embeddings)
        
        for module in modules_to_init:
            if self.config.init_type == "normal":
                nn.init.normal_(module.weight, std=self.config.init_std)
            elif self.config.init_type == "uniform":
                nn.init.uniform_(module.weight, -self.config.init_range, self.config.init_range)
            elif self.config.init_type == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif self.config.init_type == "kaiming":
                nn.init.kaiming_uniform_(module.weight)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self,
        position_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply learnable positional encoding.
        
        Args:
            position_ids: Position indices [batch_size, seq_len]
            timestamps: Unix timestamps [batch_size, seq_len] (for time embeddings)
            context: Context information [batch_size, seq_len, context_dim]
            
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = position_ids.shape
        embeddings_list = []
        
        # Basic position embeddings
        if self.config.use_learned_positions:
            # Apply curriculum learning
            if self.config.use_curriculum_learning and self.training:
                max_pos = self.stage_boundaries[min(self.current_stage + 1, len(self.stage_boundaries) - 1)]
                position_ids = torch.clamp(position_ids, max=max_pos)
            
            pos_embeddings = self.position_embeddings(position_ids)
            embeddings_list.append(pos_embeddings)
        
        # Time-based learnable embeddings
        if self.config.use_learned_time_embeddings and timestamps is not None:
            time_features = self._extract_time_embeddings(timestamps)
            embeddings_list.append(time_features)
        
        # Hierarchical embeddings
        if self.config.use_hierarchical_embeddings:
            hierarchy_embeddings = self._compute_hierarchical_embeddings(position_ids)
            embeddings_list.append(hierarchy_embeddings)
        
        # Combine embeddings
        if len(embeddings_list) > 1:
            # Concatenate and project if different dimensions
            combined_embeddings = embeddings_list[0]
            for emb in embeddings_list[1:]:
                if emb.shape[-1] != combined_embeddings.shape[-1]:
                    # Pad smaller embedding
                    if emb.shape[-1] < combined_embeddings.shape[-1]:
                        padding = torch.zeros(
                            *emb.shape[:-1], 
                            combined_embeddings.shape[-1] - emb.shape[-1],
                            device=emb.device
                        )
                        emb = torch.cat([emb, padding], dim=-1)
                combined_embeddings = combined_embeddings + emb
        else:
            combined_embeddings = embeddings_list[0] if embeddings_list else torch.zeros(
                batch_size, seq_len, self.config.d_model, device=position_ids.device
            )
        
        # Apply scaling
        combined_embeddings = combined_embeddings * self.scale_factor
        
        # Adaptive embeddings
        if self.config.use_adaptive_embeddings:
            adaptive_adjustment = self.adaptive_layer(combined_embeddings)
            combined_embeddings = combined_embeddings + adaptive_adjustment
        
        # Context-aware embeddings
        if self.config.use_context_aware_embeddings and context is not None:
            context_features = self.context_proj(context)
            gate_input = torch.cat([combined_embeddings, context_features], dim=-1)
            gate = self.context_gate(gate_input)
            combined_embeddings = combined_embeddings * gate + context_features * (1 - gate)
        
        # Apply regularization
        if self.config.use_position_dropout and self.training:
            combined_embeddings = self.position_dropout(combined_embeddings)
        
        return combined_embeddings
    
    def _extract_time_embeddings(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Extract learnable time embeddings from timestamps."""
        # Convert timestamps to time components
        minutes = ((timestamps // 60) % 60).long()
        hours = ((timestamps // 3600) % 24).long()
        days = (((timestamps // (24 * 3600)) + 4) % 7).long()  # Monday = 0
        months = ((timestamps // (30 * 24 * 3600)) % 12).long()
        
        # Get embeddings
        minute_emb = self.minute_embeddings(minutes)
        hour_emb = self.hour_embeddings(hours)
        day_emb = self.day_embeddings(days)
        month_emb = self.month_embeddings(months)
        
        # Combine time features
        time_features = torch.cat([minute_emb, hour_emb, day_emb, month_emb], dim=-1)
        time_combined = self.time_combination(time_features)
        
        # Pad to match d_model
        if time_combined.shape[-1] < self.config.d_model:
            padding = torch.zeros(
                *time_combined.shape[:-1],
                self.config.d_model - time_combined.shape[-1],
                device=time_combined.device
            )
            time_combined = torch.cat([time_combined, padding], dim=-1)
        
        return time_combined
    
    def _compute_hierarchical_embeddings(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical position embeddings."""
        hierarchy_features = []
        
        for level, embedding_layer in enumerate(self.hierarchy_embeddings):
            # Different granularities for different levels
            scale = 2 ** level
            scaled_positions = (position_ids // scale).clamp(max=embedding_layer.num_embeddings - 1)
            level_embeddings = embedding_layer(scaled_positions)
            hierarchy_features.append(level_embeddings)
        
        # Combine hierarchical features
        combined_hierarchy = torch.cat(hierarchy_features, dim=-1)
        return self.hierarchy_combination(combined_hierarchy)
    
    def update_curriculum_stage(self, stage: int):
        """Update curriculum learning stage."""
        if self.config.use_curriculum_learning:
            self.current_stage = min(stage, self.config.curriculum_stages - 1)
    
    def get_embedding_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for embeddings."""
        if not self.config.use_embedding_regularization:
            return torch.tensor(0.0)
        
        reg_loss = 0.0
        num_params = 0
        
        # Regularize position embeddings
        if hasattr(self, 'position_embeddings'):
            reg_loss += torch.norm(self.position_embeddings.weight, p=2)
            num_params += 1
        
        # Regularize time embeddings
        if self.config.use_learned_time_embeddings:
            time_embeddings = [
                self.minute_embeddings, self.hour_embeddings,
                self.day_embeddings, self.month_embeddings
            ]
            for emb in time_embeddings:
                reg_loss += torch.norm(emb.weight, p=2)
                num_params += 1
        
        # Regularize hierarchical embeddings
        if self.config.use_hierarchical_embeddings:
            for emb in self.hierarchy_embeddings:
                reg_loss += torch.norm(emb.weight, p=2)
                num_params += 1
        
        return (reg_loss / max(num_params, 1)) * self.config.reg_strength


class CryptoLearnableEmbedding(nn.Module):
    """
    Crypto-specific learnable embeddings.
    
    Features:
    - Market regime embeddings
    - Asset-specific embeddings
    - Trading pattern embeddings
    - Volatility regime embeddings
    """
    
    def __init__(self, config: LearnableEncodingConfig):
        super().__init__()
        self.config = config
        
        # Market regime embeddings
        if config.use_market_regime_embeddings:
            self.regime_embeddings = nn.Embedding(config.num_market_regimes, config.d_model // 8)
            self.regime_detector = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, config.num_market_regimes),
                nn.Softmax(dim=-1)
            )
        
        # Asset embeddings
        if config.use_asset_embeddings:
            self.asset_embeddings = nn.Embedding(config.num_assets, config.asset_embedding_dim)
            self.asset_proj = nn.Linear(config.asset_embedding_dim, config.d_model // 8)
        
        # Trading pattern embeddings
        self.pattern_embeddings = nn.Embedding(10, config.d_model // 8)  # 10 common patterns
        
        # Volatility regime embeddings
        self.volatility_embeddings = nn.Embedding(3, config.d_model // 16)  # Low, Medium, High
        
        # Combination layer
        combination_dim = 0
        if config.use_market_regime_embeddings:
            combination_dim += config.d_model // 8
        if config.use_asset_embeddings:
            combination_dim += config.d_model // 8
        combination_dim += config.d_model // 8  # patterns
        combination_dim += config.d_model // 16  # volatility
        
        self.combination = nn.Linear(combination_dim, config.d_model // 4)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        input_features: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
        volatility_regime: Optional[torch.Tensor] = None,
        pattern_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply crypto-specific learnable embeddings.
        
        Args:
            input_features: Input features [batch_size, seq_len, d_model]
            asset_ids: Asset identifiers [batch_size, seq_len]
            volatility_regime: Volatility regime [batch_size, seq_len] (0=low, 1=med, 2=high)
            pattern_ids: Pattern identifiers [batch_size, seq_len]
            
        Returns:
            crypto_embeddings: [batch_size, seq_len, d_model//4]
        """
        batch_size, seq_len, d_model = input_features.shape
        crypto_features = []
        
        # Market regime embeddings
        if self.config.use_market_regime_embeddings:
            # Detect market regime from input features
            regime_probs = self.regime_detector(input_features)
            regime_ids = torch.argmax(regime_probs, dim=-1)
            regime_emb = self.regime_embeddings(regime_ids)
            crypto_features.append(regime_emb)
        
        # Asset embeddings
        if self.config.use_asset_embeddings and asset_ids is not None:
            asset_emb = self.asset_embeddings(asset_ids)
            asset_features = self.asset_proj(asset_emb)
            crypto_features.append(asset_features)
        
        # Pattern embeddings
        if pattern_ids is not None:
            pattern_emb = self.pattern_embeddings(pattern_ids)
        else:
            # Default to uniform pattern distribution
            pattern_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_features.device)
            pattern_emb = self.pattern_embeddings(pattern_ids)
        crypto_features.append(pattern_emb)
        
        # Volatility regime embeddings
        if volatility_regime is not None:
            vol_emb = self.volatility_embeddings(volatility_regime)
        else:
            # Default to medium volatility
            vol_regime = torch.ones(batch_size, seq_len, dtype=torch.long, device=input_features.device)
            vol_emb = self.volatility_embeddings(vol_regime)
        crypto_features.append(vol_emb)
        
        # Combine crypto features
        combined_features = torch.cat(crypto_features, dim=-1)
        crypto_embeddings = self.combination(combined_features)
        
        return self.dropout(crypto_embeddings)


class AdaptiveEmbeddingLayer(nn.Module):
    """
    Adaptive embedding layer which adjusts based on input characteristics.
    
    Features:
    - Input-dependent embedding selection
    - Dynamic embedding interpolation
    - Attention-based embedding weighting
    """
    
    def __init__(self, config: LearnableEncodingConfig):
        super().__init__()
        self.config = config
        
        # Multiple embedding banks
        self.num_embedding_banks = 4
        self.embedding_banks = nn.ModuleList([
            nn.Embedding(config.max_seq_len, config.d_model)
            for _ in range(self.num_embedding_banks)
        ])
        
        # Bank selection network
        self.bank_selector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, self.num_embedding_banks),
            nn.Softmax(dim=-1)
        )
        
        # Attention-based weighting
        self.attention_weights = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Interpolation network
        self.interpolation = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.Tanh(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Initialize embedding banks with different strategies
        self._init_embedding_banks()
    
    def _init_embedding_banks(self):
        """Initialize embedding banks with different strategies."""
        init_strategies = ["normal", "uniform", "xavier", "sinusoidal"]
        
        for i, (bank, strategy) in enumerate(zip(self.embedding_banks, init_strategies)):
            if strategy == "normal":
                nn.init.normal_(bank.weight, std=0.02)
            elif strategy == "uniform":
                nn.init.uniform_(bank.weight, -0.1, 0.1)
            elif strategy == "xavier":
                nn.init.xavier_uniform_(bank.weight)
            elif strategy == "sinusoidal":
                # Sinusoidal initialization
                pos_enc = self._create_sinusoidal_encoding(
                    self.config.max_seq_len, self.config.d_model
                )
                bank.weight.data = pos_enc
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal encoding for initialization."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(
        self,
        input_features: torch.Tensor,
        position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply adaptive embedding selection.
        
        Args:
            input_features: Input features [batch_size, seq_len, d_model]
            position_ids: Position indices [batch_size, seq_len]
            
        Returns:
            adaptive_embeddings: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = input_features.shape
        
        # Get embeddings from all banks
        bank_embeddings = []
        for bank in self.embedding_banks:
            emb = bank(position_ids)
            bank_embeddings.append(emb)
        
        # Select embedding bank weights based on input
        bank_weights = self.bank_selector(input_features)  # [B, L, num_banks]
        
        # Weighted combination of banks
        stacked_embeddings = torch.stack(bank_embeddings, dim=-1)  # [B, L, D, num_banks]
        weighted_embeddings = torch.sum(
            stacked_embeddings * bank_weights.unsqueeze(-2), dim=-1
        )  # [B, L, D]
        
        # Apply attention-based refinement
        refined_embeddings, _ = self.attention_weights(
            query=weighted_embeddings,
            key=weighted_embeddings,
            value=weighted_embeddings
        )
        
        # Interpolation with input features
        interpolation_input = torch.cat([input_features, refined_embeddings], dim=-1)
        final_embeddings = self.interpolation(interpolation_input)
        
        return final_embeddings + weighted_embeddings  # Residual connection


def create_learnable_encoding(config: LearnableEncodingConfig) -> nn.Module:
    """
    Factory function for creation learnable encoding.
    
    Args:
        config: Learnable encoding configuration
        
    Returns:
        Learnable encoding module
    """
    return LearnablePositionalEmbedding(config)


if __name__ == "__main__":
    # Test learnable encodings
    config = LearnableEncodingConfig(
        d_model=512,
        max_seq_len=1000,
        use_learned_positions=True,
        use_learned_time_embeddings=True,
        use_adaptive_embeddings=True,
        use_hierarchical_embeddings=True,
        use_context_aware_embeddings=True,
        use_curriculum_learning=True,
        use_market_regime_embeddings=True,
        use_asset_embeddings=True
    )
    
    batch_size, seq_len = 4, 256
    
    # Test basic learnable embedding
    learnable_emb = LearnablePositionalEmbedding(config)
    
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
    context = torch.randn(batch_size, seq_len, config.context_dim)
    
    learnable_output = learnable_emb(
        position_ids=position_ids,
        timestamps=timestamps,
        context=context
    )
    
    print(f"Learnable embedding output: {learnable_output.shape}")
    
    # Test crypto-specific embeddings
    crypto_emb = CryptoLearnableEmbedding(config)
    
    input_features = torch.randn(batch_size, seq_len, config.d_model)
    asset_ids = torch.randint(0, config.num_assets, (batch_size, seq_len))
    volatility_regime = torch.randint(0, 3, (batch_size, seq_len))
    pattern_ids = torch.randint(0, 10, (batch_size, seq_len))
    
    crypto_output = crypto_emb(
        input_features=input_features,
        asset_ids=asset_ids,
        volatility_regime=volatility_regime,
        pattern_ids=pattern_ids
    )
    
    print(f"Crypto embedding output: {crypto_output.shape}")
    
    # Test adaptive embedding
    adaptive_emb = AdaptiveEmbeddingLayer(config)
    
    adaptive_output = adaptive_emb(
        input_features=input_features,
        position_ids=position_ids
    )
    
    print(f"Adaptive embedding output: {adaptive_output.shape}")
    
    # Test curriculum learning
    print(f"\nTesting curriculum learning:")
    for stage in range(config.curriculum_stages):
        learnable_emb.update_curriculum_stage(stage)
        stage_output = learnable_emb(position_ids=position_ids)
        print(f"Stage {stage}: {stage_output.shape}, max pos: {learnable_emb.stage_boundaries[stage+1]}")
    
    # Test regularization
    reg_loss = learnable_emb.get_embedding_regularization_loss()
    print(f"Regularization loss: {reg_loss:.6f}")
    
    # Model parameters
    print(f"\nModel parameters:")
    print(f"Learnable embedding: {sum(p.numel() for p in learnable_emb.parameters())}")
    print(f"Crypto embedding: {sum(p.numel() for p in crypto_emb.parameters())}")
    print(f"Adaptive embedding: {sum(p.numel() for p in adaptive_emb.parameters())}")
    
    # Performance comparison with different init strategies
    print(f"\nTesting different initialization strategies:")
    
    for init_type in ["normal", "uniform", "xavier", "kaiming"]:
        test_config = LearnableEncodingConfig(
            d_model=512,
            max_seq_len=1000,
            init_type=init_type,
            use_learned_positions=True
        )
        
        test_emb = LearnablePositionalEmbedding(test_config)
        test_output = test_emb(position_ids=position_ids)
        
        # Measure embedding diversity
        embedding_std = test_emb.position_embeddings.weight.std().item()
        print(f"{init_type} init: std={embedding_std:.6f}")
    
    print(f"\n✅ Learnable encoding implementation complete!")