"""
Crypto-specific attention models for various trading tasks.
Combines attention mechanisms in ready-to-use model for price prediction, 
signal generation, risk assessment and portfolio optimization.

Production attention models with pre-trained checkpoints and deployment optimization.
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
from ..transformers.trading_transformer import TradingTransformer, TradingTransformerConfig
from ..attention.temporal_attention import CryptoTemporalAttention, TemporalAttentionConfig
from ..attention.cross_attention import CryptoCrossAttention, CrossAttentionConfig
from ..attention.multi_head_attention import CryptoMultiHeadAttention, AttentionConfig

logger = logging.getLogger(__name__)


@dataclass
class CryptoPredictionModelConfig:
    """Configuration for crypto prediction models."""
    # Model architecture
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    
    # Input/Output
    input_features: int = 100
    prediction_horizon: int = 1  # Number of timesteps to predict
    prediction_targets: List[str] = None  # ["price", "volume", "volatility"]
    
    # Crypto-specific
    num_assets: int = 50
    timeframes: List[str] = None
    use_orderbook: bool = True
    use_news_sentiment: bool = True
    use_social_sentiment: bool = False
    
    # Risk management
    use_risk_metrics: bool = True
    risk_lookback: int = 20
    max_drawdown_threshold: float = 0.05
    var_confidence: float = 0.05
    
    # Model variants
    model_type: str = "price_predictor"  # "price_predictor", "signal_generator", "risk_assessor", "portfolio_optimizer"
    
    def __post_init__(self):
        if self.prediction_targets is None:
            self.prediction_targets = ["price"]
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h", "4h"]


class CryptoPricePredictionModel(nn.Module):
    """
    Crypto price prediction model using attention mechanisms.
    
    Features:
    - Multi-step price forecasting
    - Uncertainty quantification
    - Regime-aware predictions
    - Risk-adjusted outputs
    """
    
    def __init__(self, config: CryptoPredictionModelConfig):
        super().__init__()
        self.config = config
        
        # Base transformer
        transformer_config = TradingTransformerConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_encoder_layers=config.num_layers,
            input_features=config.input_features,
            output_dim=len(config.prediction_targets),
            use_multi_asset=True,
            num_assets=config.num_assets,
            use_market_regime_detection=True,
            use_risk_management=config.use_risk_metrics
        )
        
        self.transformer = TradingTransformer(transformer_config)
        
        # Multi-step prediction heads
        self.prediction_heads = nn.ModuleDict({
            target: nn.ModuleList([
                nn.Linear(config.d_model, 1)
                for _ in range(config.prediction_horizon)
            ])
            for target in config.prediction_targets
        })
        
        # Uncertainty estimation
        self.uncertainty_heads = nn.ModuleDict({
            target: nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 4),
                nn.ReLU(),
                nn.Linear(config.d_model // 4, config.prediction_horizon),
                nn.Softplus()  # Positive uncertainty
            )
            for target in config.prediction_targets
        })
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Technical indicators integration
        if config.use_orderbook:
            self.orderbook_encoder = nn.Sequential(
                nn.Linear(20, config.d_model // 4),  # 10 bid + 10 ask levels
                nn.ReLU(),
                nn.Linear(config.d_model // 4, config.d_model // 8)
            )
        
        if config.use_news_sentiment:
            self.news_encoder = nn.Sequential(
                nn.Linear(768, config.d_model // 4),  # BERT embeddings
                nn.ReLU(),
                nn.Linear(config.d_model // 4, config.d_model // 8)
            )
    
    def forward(
        self,
        price_data: torch.Tensor,
        timestamps: torch.Tensor,
        asset_ids: torch.Tensor,
        orderbook_data: Optional[torch.Tensor] = None,
        news_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict crypto prices with uncertainty quantification.
        
        Args:
            price_data: OHLCV + technical indicators [batch, seq_len, features]
            timestamps: Unix timestamps [batch, seq_len]
            asset_ids: Asset identifiers [batch, seq_len]
            orderbook_data: Order book levels [batch, seq_len, 20]
            news_embeddings: News sentiment embeddings [batch, seq_len, 768]
        """
        batch_size, seq_len = price_data.shape[:2]
        
        # Enhance input with additional data
        enhanced_features = [price_data]
        
        if self.config.use_orderbook and orderbook_data is not None:
            ob_features = self.orderbook_encoder(orderbook_data)
            # Pad to match price_data features
            if ob_features.shape[-1] < price_data.shape[-1]:
                padding = torch.zeros(
                    batch_size, seq_len, 
                    price_data.shape[-1] - ob_features.shape[-1],
                    device=price_data.device
                )
                ob_features = torch.cat([ob_features, padding], dim=-1)
            enhanced_features.append(ob_features)
        
        if self.config.use_news_sentiment and news_embeddings is not None:
            news_features = self.news_encoder(news_embeddings)
            # Pad to match price_data features
            if news_features.shape[-1] < price_data.shape[-1]:
                padding = torch.zeros(
                    batch_size, seq_len,
                    price_data.shape[-1] - news_features.shape[-1],
                    device=price_data.device
                )
                news_features = torch.cat([news_features, padding], dim=-1)
            enhanced_features.append(news_features)
        
        # Combine all features
        if len(enhanced_features) > 1:
            # Weight different feature types
            weights = torch.softmax(torch.ones(len(enhanced_features)), dim=0)
            combined_features = sum(w * f for w, f in zip(weights, enhanced_features))
        else:
            combined_features = enhanced_features[0]
        
        # Forward through transformer
        transformer_outputs = self.transformer(
            x=combined_features,
            timestamps=timestamps,
            asset_ids=asset_ids,
            **kwargs
        )
        
        hidden_states = transformer_outputs['hidden_states']
        last_hidden = hidden_states[:, -1, :]  # Use last timestep
        
        outputs = {}
        
        # Multi-step predictions for each target
        for target in self.config.prediction_targets:
            target_predictions = []
            
            # Generate predictions for each step
            current_hidden = last_hidden
            for step in range(self.config.prediction_horizon):
                step_pred = self.prediction_heads[target][step](current_hidden)
                target_predictions.append(step_pred)
                
                # Update hidden state (simple approach)
                current_hidden = current_hidden + step_pred * 0.1
            
            outputs[f'{target}_predictions'] = torch.stack(target_predictions, dim=1)  # [batch, horizon, 1]
            
            # Uncertainty estimation
            uncertainty = self.uncertainty_heads[target](last_hidden)
            outputs[f'{target}_uncertainty'] = uncertainty  # [batch, horizon]
        
        # Confidence estimation
        confidence = self.confidence_estimator(last_hidden)
        outputs['prediction_confidence'] = confidence
        
        # Include regime information
        if 'regime_probabilities' in transformer_outputs:
            outputs['regime_probabilities'] = transformer_outputs['regime_probabilities']
        
        # Risk metrics
        if 'risk_score' in transformer_outputs:
            outputs['risk_score'] = transformer_outputs['risk_score']
            outputs['uncertainty_total'] = transformer_outputs['uncertainty']
        
        return outputs


class CryptoSignalGeneratorModel(nn.Module):
    """
    Trading signal generation model.
    
    Features:
    - Buy/Sell/Hold signal classification
    - Signal strength estimation
    - Entry/Exit point detection
    - Risk-reward ratio estimation
    """
    
    def __init__(self, config: CryptoPredictionModelConfig):
        super().__init__()
        self.config = config
        
        # Temporal attention for signal detection
        temporal_config = TemporalAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_multi_timeframe=True,
            use_temporal_decay=True,
            timeframe_windows=[1, 5, 15, 60, 240]
        )
        
        self.temporal_attention = CryptoTemporalAttention(temporal_config)
        
        # Multi-timeframe processing
        self.timeframe_encoders = nn.ModuleDict({
            tf: nn.Sequential(
                nn.Linear(config.input_features, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            for tf in config.timeframes
        })
        
        # Signal classification heads
        self.signal_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 3)  # Buy/Hold/Sell
        )
        
        # Signal strength estimation
        self.signal_strength = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()  # Strength 0-1
        )
        
        # Entry/Exit detection
        self.entry_detector = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        self.exit_detector = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # Risk-reward estimation
        self.risk_reward_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 2)  # Risk, Reward
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.d_model * len(config.timeframes), config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU()
        )
    
    def forward(
        self,
        multi_timeframe_data: Dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        market_sessions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate trading signals.
        
        Args:
            multi_timeframe_data: Data for each timeframe
            timestamps: Unix timestamps
            volume_data: Volume patterns
            market_sessions: Market session identifiers
        """
        # Process each timeframe
        timeframe_features = []
        
        for timeframe in self.config.timeframes:
            if timeframe in multi_timeframe_data:
                tf_data = multi_timeframe_data[timeframe]
                tf_features = self.timeframe_encoders[timeframe](tf_data)
                timeframe_features.append(tf_features)
        
        # Fuse multi-timeframe features
        if timeframe_features:
            combined_tf_features = torch.cat(timeframe_features, dim=-1)
            fused_features = self.feature_fusion(combined_tf_features)
        else:
            # Fallback if no multi-timeframe data
            main_data = next(iter(multi_timeframe_data.values()))
            fused_features = self.timeframe_encoders[self.config.timeframes[0]](main_data)
        
        # Apply temporal attention
        temporal_output = self.temporal_attention(
            fused_features,
            timestamps=timestamps,
            volume_data=volume_data,
            market_sessions=market_sessions
        )
        
        # Use last timestep for signal generation
        signal_features = temporal_output[:, -1, :]
        
        outputs = {}
        
        # Signal classification
        signal_logits = self.signal_classifier(signal_features)
        outputs['signal_probabilities'] = F.softmax(signal_logits, dim=-1)
        outputs['signal_class'] = torch.argmax(signal_logits, dim=-1)
        
        # Signal strength
        outputs['signal_strength'] = self.signal_strength(signal_features)
        
        # Entry/Exit detection
        outputs['entry_probability'] = self.entry_detector(signal_features)
        outputs['exit_probability'] = self.exit_detector(signal_features)
        
        # Risk-reward estimation
        risk_reward = self.risk_reward_estimator(signal_features)
        outputs['expected_risk'] = risk_reward[:, 0:1]
        outputs['expected_reward'] = risk_reward[:, 1:2]
        outputs['risk_reward_ratio'] = outputs['expected_reward'] / (outputs['expected_risk'] + 1e-6)
        
        return outputs


class CryptoRiskAssessmentModel(nn.Module):
    """
    Risk assessment model for crypto portfolios.
    
    Features:
    - Value at Risk (VaR) estimation
    - Drawdown prediction
    - Volatility forecasting
    - Correlation analysis
    """
    
    def __init__(self, config: CryptoPredictionModelConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention for asset correlation analysis
        cross_attention_config = CrossAttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            use_cross_asset=True,
            num_assets=config.num_assets
        )
        
        self.cross_asset_attention = CryptoCrossAttention(cross_attention_config)
        
        # Risk feature encoder
        self.risk_encoder = nn.Sequential(
            nn.Linear(config.input_features, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # VaR estimation network
        self.var_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.risk_lookback)  # VaR for multiple horizons
        )
        
        # Drawdown prediction
        self.drawdown_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()  # Probability of drawdown > threshold
        )
        
        # Volatility forecasting
        self.volatility_forecaster = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, config.prediction_horizon),
            nn.Softplus()  # Positive volatility
        )
        
        # Correlation matrix estimation
        self.correlation_estimator = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_assets * config.num_assets),
            nn.Tanh()  # Correlation values [-1, 1]
        )
        
        # Risk regime detection
        self.risk_regime_detector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 4),  # Low, Medium, High, Extreme
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        portfolio_data: torch.Tensor,
        asset_weights: torch.Tensor,
        historical_returns: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Assess portfolio risk.
        
        Args:
            portfolio_data: Portfolio features [batch, seq_len, features]
            asset_weights: Portfolio weights [batch, num_assets]
            historical_returns: Historical returns [batch, seq_len, num_assets]
        """
        batch_size, seq_len, _ = portfolio_data.shape
        
        # Encode risk features
        risk_features = self.risk_encoder(portfolio_data)
        
        # Cross-asset attention for correlation analysis
        cross_asset_features = self.cross_asset_attention(
            price_data=risk_features,
            exchange_data={"main": risk_features}
        )
        
        # Use last timestep for risk assessment
        current_risk_state = cross_asset_features[:, -1, :]
        
        outputs = {}
        
        # VaR estimation
        var_estimates = self.var_estimator(current_risk_state)
        outputs['value_at_risk'] = var_estimates  # [batch, risk_lookback]
        
        # Portfolio VaR (weighted average)
        if asset_weights.shape[1] == self.config.num_assets:
            # Weighted portfolio VaR
            portfolio_var = torch.sum(var_estimates * asset_weights.unsqueeze(-1), dim=1)
            outputs['portfolio_var'] = portfolio_var
        
        # Drawdown prediction
        outputs['drawdown_probability'] = self.drawdown_predictor(current_risk_state)
        
        # Volatility forecasting
        outputs['volatility_forecast'] = self.volatility_forecaster(current_risk_state)
        
        # Correlation matrix
        correlation_flat = self.correlation_estimator(current_risk_state)
        correlation_matrix = correlation_flat.view(batch_size, self.config.num_assets, self.config.num_assets)
        # Make symmetric
        correlation_matrix = (correlation_matrix + correlation_matrix.transpose(-1, -2)) / 2
        outputs['correlation_matrix'] = correlation_matrix
        
        # Risk regime detection
        outputs['risk_regime'] = self.risk_regime_detector(current_risk_state)
        
        # Additional risk metrics
        if historical_returns.shape[-1] == self.config.num_assets:
            # Compute historical volatility
            historical_vol = torch.std(historical_returns, dim=1)  # [batch, num_assets]
            outputs['historical_volatility'] = historical_vol
            
            # Portfolio volatility
            portfolio_returns = torch.sum(historical_returns * asset_weights.unsqueeze(1), dim=-1)
            portfolio_vol = torch.std(portfolio_returns, dim=1, keepdim=True)
            outputs['portfolio_volatility'] = portfolio_vol
        
        return outputs


class CryptoPortfolioOptimizerModel(nn.Module):
    """
    Portfolio optimization model using attention mechanisms.
    
    Features:
    - Optimal weight allocation
    - Risk-adjusted returns
    - Dynamic rebalancing signals
    - Constraint handling
    """
    
    def __init__(self, config: CryptoPredictionModelConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention for asset relationships
        attention_config = AttentionConfig(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.asset_attention = CryptoMultiHeadAttention(
            attention_config, use_volume_weighting=True
        )
        
        # Expected return estimation
        self.return_estimator = nn.Sequential(
            nn.Linear(config.input_features, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_assets)
        )
        
        # Risk estimation
        self.risk_estimator = nn.Sequential(
            nn.Linear(config.input_features, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.num_assets)
        )
        
        # Portfolio weight optimization
        self.weight_optimizer = nn.Sequential(
            nn.Linear(config.d_model + config.num_assets * 2, config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_assets),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        # Rebalancing signal
        self.rebalancing_detector = nn.Sequential(
            nn.Linear(config.d_model, 1),
            nn.Sigmoid()
        )
        
        # Risk tolerance adaptation
        self.risk_tolerance_adapter = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        asset_features: torch.Tensor,
        current_weights: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        risk_tolerance: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Optimize portfolio allocation.
        
        Args:
            asset_features: Features for each asset [batch, seq_len, num_assets, features]
            current_weights: Current portfolio weights [batch, num_assets]
            volume_data: Volume data for assets [batch, seq_len, num_assets]
            risk_tolerance: Risk tolerance level 0-1
        """
        batch_size, seq_len, num_assets, feature_dim = asset_features.shape
        
        # Reshape for processing
        reshaped_features = asset_features.view(batch_size * seq_len, num_assets, feature_dim)
        
        # Apply attention across assets
        attended_features = self.asset_attention(
            query=reshaped_features,
            volume=volume_data.view(-1, num_assets) if volume_data is not None else None
        )  # [batch*seq_len, num_assets, d_model]
        
        # Reshape back
        attended_features = attended_features.view(batch_size, seq_len, num_assets, -1)
        
        # Use last timestep
        current_features = attended_features[:, -1, :, :]  # [batch, num_assets, d_model]
        
        # Estimate expected returns and risks for each asset
        expected_returns = self.return_estimator(current_features.mean(dim=1))  # [batch, num_assets]
        expected_risks = self.risk_estimator(current_features.mean(dim=1))  # [batch, num_assets]
        
        outputs = {}
        outputs['expected_returns'] = expected_returns
        outputs['expected_risks'] = expected_risks
        
        # Portfolio optimization
        portfolio_features = current_features.mean(dim=1)  # Average across assets
        
        # Combine features for weight optimization
        optimization_input = torch.cat([
            portfolio_features,
            expected_returns,
            expected_risks
        ], dim=-1)
        
        # Generate optimal weights
        optimal_weights = self.weight_optimizer(optimization_input)
        outputs['optimal_weights'] = optimal_weights
        
        # Adaptive risk tolerance
        adaptive_risk_tolerance = self.risk_tolerance_adapter(portfolio_features)
        risk_adjusted_weights = optimal_weights * (1 - adaptive_risk_tolerance * expected_risks / expected_risks.max(dim=1, keepdim=True)[0])
        risk_adjusted_weights = F.softmax(risk_adjusted_weights, dim=-1)  # Renormalize
        outputs['risk_adjusted_weights'] = risk_adjusted_weights
        
        # Rebalancing signal
        rebalancing_signal = self.rebalancing_detector(portfolio_features)
        outputs['rebalancing_signal'] = rebalancing_signal
        
        # Portfolio metrics
        current_portfolio_return = torch.sum(expected_returns * current_weights, dim=-1, keepdim=True)
        optimal_portfolio_return = torch.sum(expected_returns * optimal_weights, dim=-1, keepdim=True)
        
        outputs['current_portfolio_return'] = current_portfolio_return
        outputs['optimal_portfolio_return'] = optimal_portfolio_return
        outputs['improvement_potential'] = optimal_portfolio_return - current_portfolio_return
        
        # Risk metrics
        current_portfolio_risk = torch.sqrt(torch.sum((current_weights * expected_risks) ** 2, dim=-1, keepdim=True))
        optimal_portfolio_risk = torch.sqrt(torch.sum((optimal_weights * expected_risks) ** 2, dim=-1, keepdim=True))
        
        outputs['current_portfolio_risk'] = current_portfolio_risk
        outputs['optimal_portfolio_risk'] = optimal_portfolio_risk
        
        # Sharpe ratio (simplified)
        outputs['current_sharpe_ratio'] = current_portfolio_return / (current_portfolio_risk + 1e-6)
        outputs['optimal_sharpe_ratio'] = optimal_portfolio_return / (optimal_portfolio_risk + 1e-6)
        
        return outputs


def create_crypto_model(
    model_type: str,
    config: Optional[CryptoPredictionModelConfig] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for creation crypto attention models.
    
    Args:
        model_type: Type of model ("price_predictor", "signal_generator", "risk_assessor", "portfolio_optimizer")
        config: Model configuration
        **kwargs: Additional configuration parameters
    """
    if config is None:
        config = CryptoPredictionModelConfig(model_type=model_type, **kwargs)
    
    if model_type == "price_predictor":
        return CryptoPricePredictionModel(config)
    elif model_type == "signal_generator":
        return CryptoSignalGeneratorModel(config)
    elif model_type == "risk_assessor":
        return CryptoRiskAssessmentModel(config)
    elif model_type == "portfolio_optimizer":
        return CryptoPortfolioOptimizerModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test crypto attention models
    config = CryptoPredictionModelConfig(
        d_model=256,
        num_heads=4,
        num_layers=3,
        input_features=50,
        prediction_horizon=5,
        num_assets=10,
        prediction_targets=["price", "volatility"]
    )
    
    batch_size, seq_len = 4, 128
    
    # Test price prediction model
    print("Testing Price Prediction Model:")
    price_model = CryptoPricePredictionModel(config)
    
    price_data = torch.randn(batch_size, seq_len, config.input_features)
    timestamps = torch.randint(1640995200, 1672531200, (batch_size, seq_len))
    asset_ids = torch.randint(0, config.num_assets, (batch_size, seq_len))
    orderbook_data = torch.randn(batch_size, seq_len, 20)
    
    price_outputs = price_model(
        price_data=price_data,
        timestamps=timestamps,
        asset_ids=asset_ids,
        orderbook_data=orderbook_data
    )
    
    for key, value in price_outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test signal generator
    print("\nTesting Signal Generator Model:")
    signal_model = CryptoSignalGeneratorModel(config)
    
    multi_tf_data = {
        "1m": torch.randn(batch_size, seq_len, config.input_features),
        "5m": torch.randn(batch_size, seq_len, config.input_features),
        "15m": torch.randn(batch_size, seq_len, config.input_features)
    }
    
    signal_outputs = signal_model(
        multi_timeframe_data=multi_tf_data,
        timestamps=timestamps
    )
    
    for key, value in signal_outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test risk assessment model
    print("\nTesting Risk Assessment Model:")
    risk_model = CryptoRiskAssessmentModel(config)
    
    portfolio_data = torch.randn(batch_size, seq_len, config.input_features)
    asset_weights = torch.softmax(torch.randn(batch_size, config.num_assets), dim=-1)
    historical_returns = torch.randn(batch_size, seq_len, config.num_assets) * 0.02
    
    risk_outputs = risk_model(
        portfolio_data=portfolio_data,
        asset_weights=asset_weights,
        historical_returns=historical_returns
    )
    
    for key, value in risk_outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test portfolio optimizer
    print("\nTesting Portfolio Optimizer Model:")
    optimizer_model = CryptoPortfolioOptimizerModel(config)
    
    asset_features = torch.randn(batch_size, seq_len, config.num_assets, config.input_features)
    current_weights = torch.softmax(torch.randn(batch_size, config.num_assets), dim=-1)
    
    optimizer_outputs = optimizer_model(
        asset_features=asset_features,
        current_weights=current_weights
    )
    
    for key, value in optimizer_outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Model parameters
    print(f"\nModel Parameters:")
    models = {
        "Price Predictor": price_model,
        "Signal Generator": signal_model,
        "Risk Assessor": risk_model,
        "Portfolio Optimizer": optimizer_model
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")
    
    # Test factory function
    print(f"\nTesting factory function:")
    factory_model = create_crypto_model(
        model_type="price_predictor",
        d_model=128,
        num_heads=4,
        input_features=30
    )
    
    factory_params = sum(p.numel() for p in factory_model.parameters())
    print(f"Factory model parameters: {factory_params:,}")
    
    print(f"\n✅ Crypto attention models implementation complete!")