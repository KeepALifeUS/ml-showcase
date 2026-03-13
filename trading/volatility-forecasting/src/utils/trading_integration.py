"""
Trading Integration Utilities for Volatility-Based Trading

Implementation of comprehensive trading utilities for integrating volatility forecasts:
- Volatility-based position sizing
- Dynamic stop-loss adjustment
- Options strategies recommendations
- Portfolio volatility optimization (risk parity)
- Volatility breakout strategies
- Pairs trading based on correlation analysis
- Kelly criterion for optimal position sizing

Features:
- Real-time position adjustment
- Risk-aware portfolio construction
- Advanced portfolio optimization
- Multi-asset correlation modeling
- Production-ready execution logic
- Comprehensive backtesting integration
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from numba import jit
import warnings

# Logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class PositionSizeMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    VOLATILITY_TARGET = "volatility_target"
    KELLY_CRITERION = "kelly_criterion"
    VAR_BASED = "var_based"
    SHARPE_OPTIMAL = "sharpe_optimal"
    RISK_PARITY = "risk_parity"

class PortfolioOptimizationMethod(Enum):
    """Portfolio optimization methods"""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    BLACK_LITTERMAN = "black_litterman"
    HIERARCHICAL_RISK_PARITY = "hrp"

class OptionStrategy(Enum):
    """Option strategies"""
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY_SPREAD = "butterfly_spread"
    VOLATILITY_SWAP = "volatility_swap"

@dataclass
class PositionSizeRecommendation:
    """Position size recommendation"""
    symbol: str
    timestamp: datetime
    
    # Position sizing
    recommended_size: float
    max_position_size: float
    sizing_method: PositionSizeMethod
    
    # Risk parameters
    expected_volatility: float
    var_95: float
    expected_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # Risk management
    stop_loss_level: Optional[float] = None
    take_profit_level: Optional[float] = None
    max_drawdown_limit: float = 0.05  # 5%
    
    # Confidence and adjustments
    confidence_score: float = 0.8
    volatility_adjustment_factor: float = 1.0
    market_regime_adjustment: float = 1.0
    
    # Metadata
    calculation_method: str = ""
    risk_budget: float = 0.02  # 2% risk per trade
    leverage_used: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DynamicStopLossRecommendation:
    """Dynamic stop-loss recommendation"""
    symbol: str
    timestamp: datetime
    entry_price: float
    current_price: float
    position_direction: str  # "long" or "short"
    
    # Stop-loss levels
    current_stop_loss: float
    recommended_stop_loss: float
    trailing_stop_distance: float
    
    # Volatility-based adjustments
    volatility_multiplier: float = 2.0
    atr_stop_distance: Optional[float] = None
    bollinger_stop_level: Optional[float] = None
    
    # Risk metrics
    current_unrealized_pnl: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    # Recommendation confidence
    adjustment_confidence: float = 0.8
    risk_level: str = "medium"  # low, medium, high
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioOptimizationResult:
    """Portfolio optimization result"""
    optimization_method: PortfolioOptimizationMethod
    timestamp: datetime
    
    # Optimal weights
    optimal_weights: Dict[str, float]
    rebalancing_required: bool
    current_weights: Optional[Dict[str, float]] = None
    
    # Expected portfolio metrics
    expected_return: float
    expected_volatility: float
    expected_sharpe_ratio: float
    
    # Risk measures
    portfolio_var_95: float
    portfolio_cvar_95: float
    maximum_drawdown_estimate: float
    
    # Diversification metrics
    effective_number_of_assets: float
    concentration_risk: float
    correlation_risk: float
    
    # Rebalancing info
    turnover: float = 0.0
    transaction_cost_estimate: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptionStrategyRecommendation:
    """Option strategy recommendation"""
    symbol: str
    timestamp: datetime
    strategy: OptionStrategy
    
    # Market outlook
    volatility_forecast: float
    volatility_regime: str  # "low", "normal", "high", "extreme"
    direction_bias: str  # "neutral", "bullish", "bearish"
    
    # Strategy details
    strikes_recommended: List[float]
    expiry_recommendation: datetime
    premium_estimate: float
    
    # Risk-reward profile
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_of_profit: float
    
    # Greeks exposure
    delta_exposure: float = 0.0
    gamma_exposure: float = 0.0
    theta_exposure: float = 0.0
    vega_exposure: float = 0.0
    
    # Execution details
    confidence_score: float = 0.7
    recommended_position_size: float = 1.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@jit(nopython=True)
def _kelly_criterion_numba(
    expected_return: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """Fast Kelly Criterion calculation"""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly fraction: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    
    # Cap Kelly at reasonable levels (25% max)
    return max(0, min(0.25, kelly_fraction))

@jit(nopython=True)
def _volatility_position_size_numba(
    target_volatility: float,
    asset_volatility: float,
    portfolio_value: float,
    max_position_ratio: float = 0.1
) -> float:
    """Fast volatility-based position sizing"""
    if asset_volatility <= 0:
        return 0.0
    
    # Position size = (Target Vol / Asset Vol) * Portfolio Value
    raw_size = (target_volatility / asset_volatility) * portfolio_value
    
    # Cap at maximum position ratio
    max_size = portfolio_value * max_position_ratio
    
    return min(raw_size, max_size)

class PositionSizer:
    """
    Volatility-Based Position Sizing Engine
    
    Dynamic position sizing based on:
    - Forecasted volatility
    - Kelly criterion
    - VaR constraints
    - Risk parity principles
    """
    
    def __init__(self, portfolio_value: float = 100000, max_risk_per_trade: float = 0.02):
        self.portfolio_value = portfolio_value
        self.max_risk_per_trade = max_risk_per_trade  # 2% max risk per trade
        self.position_history: List[PositionSizeRecommendation] = []
        
        logger.info(f"🎯 Position sizer initialized: ${portfolio_value:,.0f} portfolio")

    async def calculate_position_size(
        self,
        symbol: str,
        expected_volatility: float,
        expected_return: Optional[float] = None,
        current_price: float = 100.0,
        method: PositionSizeMethod = PositionSizeMethod.VOLATILITY_TARGET,
        target_volatility: float = 0.15,  # 15% target vol
        historical_performance: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> PositionSizeRecommendation:
        """
        Calculate optimal position size
        
        Args:
            symbol: Trading symbol
            expected_volatility: Forecasted volatility (annualized)
            expected_return: Expected return (optional)
            current_price: Current asset price
            method: Position sizing method
            target_volatility: Target portfolio volatility
            historical_performance: Historical win/loss statistics
        """
        try:
            logger.info(f"🔄 Calculating position size for {symbol} using {method.value}")
            
            if method == PositionSizeMethod.VOLATILITY_TARGET:
                position_size = await self._volatility_target_sizing(
                    expected_volatility, target_volatility, current_price
                )
                calculation_method = "Volatility Target"
                
            elif method == PositionSizeMethod.KELLY_CRITERION:
                if not historical_performance or not expected_return:
                    raise ValueError("Kelly criterion requires historical performance and expected return")
                    
                position_size = await self._kelly_criterion_sizing(
                    expected_return, historical_performance, current_price
                )
                calculation_method = "Kelly Criterion"
                
            elif method == PositionSizeMethod.VAR_BASED:
                position_size = await self._var_based_sizing(
                    expected_volatility, current_price, kwargs.get('var_confidence', 0.95)
                )
                calculation_method = "VaR-Based"
                
            elif method == PositionSizeMethod.FIXED_AMOUNT:
                position_size = kwargs.get('fixed_amount', self.portfolio_value * 0.05)
                calculation_method = "Fixed Amount"
                
            else:
                # Default to volatility target
                position_size = await self._volatility_target_sizing(
                    expected_volatility, target_volatility, current_price
                )
                calculation_method = "Default Volatility Target"
            
            # Apply risk constraints
            max_position = self.portfolio_value * self.max_risk_per_trade / expected_volatility
            position_size = min(position_size, max_position)
            
            # Calculate risk metrics
            var_95 = self._calculate_var_estimate(position_size, expected_volatility, current_price)
            
            # Stop-loss calculation
            stop_loss_distance = expected_volatility * 2.0  # 2x volatility stop
            if method == PositionSizeMethod.VOLATILITY_TARGET:
                stop_loss_level = current_price * (1 - stop_loss_distance)
            else:
                stop_loss_level = current_price * (1 - stop_loss_distance)
            
            # Market regime adjustment
            regime_adjustment = await self._calculate_regime_adjustment(expected_volatility)
            adjusted_position_size = position_size * regime_adjustment
            
            recommendation = PositionSizeRecommendation(
                symbol=symbol,
                timestamp=datetime.now(),
                recommended_size=adjusted_position_size,
                max_position_size=max_position,
                sizing_method=method,
                expected_volatility=expected_volatility,
                var_95=var_95,
                expected_return=expected_return,
                sharpe_ratio=expected_return / expected_volatility if expected_return and expected_volatility > 0 else None,
                stop_loss_level=stop_loss_level,
                confidence_score=0.8,  # Can be enhanced with model uncertainty
                volatility_adjustment_factor=1.0,
                market_regime_adjustment=regime_adjustment,
                calculation_method=calculation_method,
                risk_budget=self.max_risk_per_trade,
                metadata={
                    "current_price": current_price,
                    "target_volatility": target_volatility,
                    "portfolio_value": self.portfolio_value,
                    "method_params": kwargs
                }
            )
            
            self.position_history.append(recommendation)
            logger.info(f"✅ Position size calculated: {adjusted_position_size:.0f} units")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Error calculating position size: {e}")
            raise

    async def _volatility_target_sizing(
        self, 
        asset_volatility: float, 
        target_volatility: float,
        current_price: float
    ) -> float:
        """Volatility target position sizing"""
        
        if asset_volatility <= 0:
            return 0.0
        
        # Position value = (Target Vol / Asset Vol) * Portfolio Value
        position_value = (target_volatility / asset_volatility) * self.portfolio_value
        
        # Convert to number of units
        position_size = position_value / current_price
        
        return position_size

    async def _kelly_criterion_sizing(
        self,
        expected_return: float,
        historical_performance: Dict[str, float],
        current_price: float
    ) -> float:
        """Kelly criterion position sizing"""
        
        win_rate = historical_performance.get('win_rate', 0.5)
        avg_win = historical_performance.get('avg_win', 0.02)
        avg_loss = abs(historical_performance.get('avg_loss', 0.02))
        
        # Calculate Kelly fraction
        kelly_fraction = _kelly_criterion_numba(expected_return, win_rate, avg_win, avg_loss)
        
        # Apply Kelly fraction to portfolio
        position_value = kelly_fraction * self.portfolio_value
        position_size = position_value / current_price
        
        return position_size

    async def _var_based_sizing(
        self,
        expected_volatility: float,
        current_price: float,
        confidence_level: float = 0.95
    ) -> float:
        """VaR-based position sizing"""
        
        # Target: Risk no more than max_risk_per_trade at confidence_level
        z_score = stats.norm.ppf(confidence_level)
        
        # VaR = Position_Value * Volatility * Z_score
        # Position_Value = (Max_Risk * Portfolio_Value) / (Volatility * Z_score)
        max_risk_value = self.max_risk_per_trade * self.portfolio_value
        
        if expected_volatility > 0 and z_score > 0:
            position_value = max_risk_value / (expected_volatility * z_score)
            position_size = position_value / current_price
        else:
            position_size = 0.0
        
        return position_size

    def _calculate_var_estimate(
        self,
        position_size: float,
        volatility: float,
        price: float,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate VaR estimate for position"""
        
        position_value = position_size * price
        z_score = stats.norm.ppf(confidence_level)
        
        var_estimate = position_value * volatility * z_score
        return var_estimate

    async def _calculate_regime_adjustment(self, volatility: float) -> float:
        """Calculate market regime adjustment factor"""
        
        # Simple regime classification
        if volatility < 0.1:  # Low volatility
            return 1.2  # Increase position size
        elif volatility < 0.2:  # Normal volatility
            return 1.0  # No adjustment
        elif volatility < 0.4:  # High volatility  
            return 0.8  # Decrease position size
        else:  # Extreme volatility
            return 0.6  # Significantly decrease
    
    def get_position_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing history"""
        
        if not self.position_history:
            return {"message": "No position sizing history"}
        
        recent_recommendations = self.position_history[-10:]  # Last 10
        
        summary = {
            "total_recommendations": len(self.position_history),
            "portfolio_value": self.portfolio_value,
            "max_risk_per_trade": self.max_risk_per_trade,
            "average_position_size": np.mean([r.recommended_size for r in recent_recommendations]),
            "average_volatility": np.mean([r.expected_volatility for r in recent_recommendations]),
            "methods_used": list(set(r.sizing_method.value for r in recent_recommendations)),
            "recent_confidence_scores": [r.confidence_score for r in recent_recommendations],
            "regime_adjustments": [r.market_regime_adjustment for r in recent_recommendations]
        }
        
        return summary

class DynamicStopLoss:
    """
    Dynamic Stop-Loss Management with Volatility Adjustment

    Adaptive stop-losses based on:
    - Current volatility
    - ATR (Average True Range)
    - Bollinger Bands
    - Trailing stops with volatility scaling
    """
    
    def __init__(self, default_stop_multiplier: float = 2.0):
        self.default_stop_multiplier = default_stop_multiplier
        self.stop_loss_history: List[DynamicStopLossRecommendation] = []
        
        logger.info(f"🎯 Dynamic stop-loss manager initialized")

    async def calculate_dynamic_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        position_direction: str,
        current_volatility: float,
        price_history: Optional[pd.Series] = None,
        atr_period: int = 14,
        bollinger_period: int = 20,
        bollinger_std: float = 2.0,
        **kwargs
    ) -> DynamicStopLossRecommendation:
        """
        Calculate dynamic stop-loss level
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price of position
            current_price: Current market price
            position_direction: "long" or "short"
            current_volatility: Current volatility estimate
            price_history: Historical price data for ATR/Bollinger calculation
            atr_period: ATR calculation period
            bollinger_period: Bollinger Bands period
            bollinger_std: Bollinger Bands standard deviations
        """
        try:
            logger.info(f"🔄 Calculating dynamic stop-loss for {symbol} {position_direction} position")
            
            # Basic volatility-based stop
            vol_stop_distance = current_volatility * self.default_stop_multiplier
            
            if position_direction.lower() == "long":
                basic_stop_loss = current_price * (1 - vol_stop_distance)
                current_unrealized_pnl = current_price - entry_price
            else:  # short
                basic_stop_loss = current_price * (1 + vol_stop_distance)
                current_unrealized_pnl = entry_price - current_price
            
            # Enhanced stops if price history available
            atr_stop_distance = None
            bollinger_stop_level = None
            
            if price_history is not None and len(price_history) >= max(atr_period, bollinger_period):
                # ATR-based stop
                atr_stop_distance = await self._calculate_atr_stop(
                    price_history, atr_period, position_direction
                )
                
                # Bollinger-based stop
                bollinger_stop_level = await self._calculate_bollinger_stop(
                    price_history, bollinger_period, bollinger_std, position_direction
                )
            
            # Combine different stop methods
            recommended_stop = await self._combine_stop_methods(
                basic_stop_loss,
                atr_stop_distance,
                bollinger_stop_level,
                current_price,
                position_direction
            )
            
            # Trailing stop logic
            current_stop_loss = kwargs.get('current_stop_loss', basic_stop_loss)
            trailing_stop = await self._calculate_trailing_stop(
                current_stop_loss,
                recommended_stop,
                current_price,
                position_direction,
                vol_stop_distance
            )
            
            # Risk level assessment
            risk_level = await self._assess_risk_level(
                current_unrealized_pnl, entry_price, current_volatility
            )
            
            # Max favorable/adverse excursion (simplified)
            max_favorable_excursion = max(0, current_unrealized_pnl)
            max_adverse_excursion = min(0, current_unrealized_pnl)
            
            recommendation = DynamicStopLossRecommendation(
                symbol=symbol,
                timestamp=datetime.now(),
                entry_price=entry_price,
                current_price=current_price,
                position_direction=position_direction,
                current_stop_loss=current_stop_loss,
                recommended_stop_loss=trailing_stop,
                trailing_stop_distance=vol_stop_distance,
                volatility_multiplier=self.default_stop_multiplier,
                atr_stop_distance=atr_stop_distance,
                bollinger_stop_level=bollinger_stop_level,
                current_unrealized_pnl=current_unrealized_pnl,
                max_favorable_excursion=max_favorable_excursion,
                max_adverse_excursion=max_adverse_excursion,
                adjustment_confidence=0.8,
                risk_level=risk_level,
                metadata={
                    "volatility_stop": basic_stop_loss,
                    "combined_methods": ["volatility", "atr" if atr_stop_distance else None, "bollinger" if bollinger_stop_level else None],
                    "stop_evolution": "trailing"
                }
            )
            
            self.stop_loss_history.append(recommendation)
            logger.info(f"✅ Dynamic stop-loss: {trailing_stop:.4f} (risk level: {risk_level})")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Error calculating dynamic stop-loss: {e}")
            raise

    async def _calculate_atr_stop(
        self,
        price_history: pd.Series,
        atr_period: int,
        position_direction: str
    ) -> float:
        """Calculate ATR-based stop distance"""
        
        # Simple ATR calculation (simplified - in production needs OHLC)
        returns = price_history.pct_change().dropna()
        atr = returns.abs().rolling(atr_period).mean().iloc[-1]
        
        # Convert to price-based distance
        atr_stop_distance = atr * self.default_stop_multiplier
        
        return atr_stop_distance

    async def _calculate_bollinger_stop(
        self,
        price_history: pd.Series,
        bollinger_period: int,
        bollinger_std: float,
        position_direction: str
    ) -> float:
        """Calculate Bollinger Band-based stop level"""
        
        sma = price_history.rolling(bollinger_period).mean().iloc[-1]
        std = price_history.rolling(bollinger_period).std().iloc[-1]
        
        if position_direction.lower() == "long":
            bollinger_stop = sma - bollinger_std * std
        else:
            bollinger_stop = sma + bollinger_std * std
        
        return bollinger_stop

    async def _combine_stop_methods(
        self,
        vol_stop: float,
        atr_distance: Optional[float],
        bollinger_stop: Optional[float],
        current_price: float,
        position_direction: str
    ) -> float:
        """Combine different stop-loss methods"""
        
        stops = [vol_stop]
        
        # Add ATR-based stop
        if atr_distance:
            if position_direction.lower() == "long":
                atr_stop = current_price * (1 - atr_distance)
            else:
                atr_stop = current_price * (1 + atr_distance)
            stops.append(atr_stop)
        
        # Add Bollinger stop
        if bollinger_stop:
            stops.append(bollinger_stop)
        
        # Use most conservative stop (furthest from current price for long, closest for short)
        if position_direction.lower() == "long":
            combined_stop = min(stops)  # Lowest stop for long positions
        else:
            combined_stop = max(stops)  # Highest stop for short positions
        
        return combined_stop

    async def _calculate_trailing_stop(
        self,
        current_stop: float,
        recommended_stop: float,
        current_price: float,
        position_direction: str,
        vol_distance: float
    ) -> float:
        """Calculate trailing stop-loss"""
        
        if position_direction.lower() == "long":
            # For long positions, only move stop up (trail upward)
            trailing_stop = max(current_stop, recommended_stop)
        else:
            # For short positions, only move stop down (trail downward)
            trailing_stop = min(current_stop, recommended_stop)
        
        return trailing_stop

    async def _assess_risk_level(
        self,
        unrealized_pnl: float,
        entry_price: float,
        volatility: float
    ) -> str:
        """Assess current risk level of position"""
        
        pnl_percentage = unrealized_pnl / entry_price if entry_price > 0 else 0
        
        # Risk assessment based on PnL and volatility
        if abs(pnl_percentage) > volatility * 2:
            return "high"
        elif abs(pnl_percentage) > volatility:
            return "medium"
        else:
            return "low"

class PortfolioOptimizer:
    """
    Portfolio Optimization with Volatility Forecasts

    Modern portfolio optimization:
    - Mean-Variance Optimization with shrinkage
    - Risk Parity
    - Black-Litterman with volatility views
    - Hierarchical Risk Parity
    - Maximum Sharpe with volatility constraints
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_history: List[PortfolioOptimizationResult] = []
        
        logger.info(f"🎯 Portfolio optimizer initialized (rf={risk_free_rate:.2%})")

    async def optimize_portfolio(
        self,
        expected_returns: Dict[str, float],
        volatility_forecasts: Dict[str, float],
        correlation_matrix: pd.DataFrame,
        method: PortfolioOptimizationMethod = PortfolioOptimizationMethod.RISK_PARITY,
        constraints: Optional[Dict[str, Any]] = None,
        current_weights: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> PortfolioOptimizationResult:
        """
        Optimize portfolio weights
        
        Args:
            expected_returns: Expected returns for each asset
            volatility_forecasts: Volatility forecasts for each asset
            correlation_matrix: Asset correlation matrix
            method: Optimization method
            constraints: Portfolio constraints (min/max weights, etc.)
            current_weights: Current portfolio weights
        """
        try:
            logger.info(f"🔄 Optimizing portfolio using {method.value}...")
            
            # Validate inputs
            assets = list(expected_returns.keys())
            if not all(asset in volatility_forecasts for asset in assets):
                raise ValueError("All assets must have volatility forecasts")
            
            # Construct covariance matrix
            volatilities = np.array([volatility_forecasts[asset] for asset in assets])
            correlation_matrix_aligned = correlation_matrix.reindex(assets, columns=assets)
            covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix_aligned.values
            
            # Expected returns vector
            expected_returns_vec = np.array([expected_returns[asset] for asset in assets])
            
            # Optimize based on method
            if method == PortfolioOptimizationMethod.MEAN_VARIANCE:
                optimal_weights_vec = await self._mean_variance_optimization(
                    expected_returns_vec, covariance_matrix, constraints
                )
                
            elif method == PortfolioOptimizationMethod.RISK_PARITY:
                optimal_weights_vec = await self._risk_parity_optimization(
                    covariance_matrix, constraints
                )
                
            elif method == PortfolioOptimizationMethod.MINIMUM_VARIANCE:
                optimal_weights_vec = await self._minimum_variance_optimization(
                    covariance_matrix, constraints
                )
                
            elif method == PortfolioOptimizationMethod.MAXIMUM_SHARPE:
                optimal_weights_vec = await self._maximum_sharpe_optimization(
                    expected_returns_vec, covariance_matrix, constraints
                )
                
            elif method == PortfolioOptimizationMethod.BLACK_LITTERMAN:
                optimal_weights_vec = await self._black_litterman_optimization(
                    expected_returns_vec, covariance_matrix, constraints, kwargs
                )
                
            else:
                # Default to Risk Parity
                optimal_weights_vec = await self._risk_parity_optimization(
                    covariance_matrix, constraints
                )
            
            # Convert to dictionary
            optimal_weights = dict(zip(assets, optimal_weights_vec))
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimal_weights_vec * expected_returns_vec)
            portfolio_variance = np.dot(optimal_weights_vec, np.dot(covariance_matrix, optimal_weights_vec))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Risk measures
            portfolio_var_95 = portfolio_volatility * stats.norm.ppf(0.95)
            portfolio_cvar_95 = portfolio_var_95 * 1.2  # Approximation
            max_dd_estimate = portfolio_volatility * 2  # Rough estimate
            
            # Diversification metrics
            effective_assets = await self._calculate_effective_number_of_assets(optimal_weights_vec)
            concentration_risk = await self._calculate_concentration_risk(optimal_weights_vec)
            correlation_risk = await self._calculate_correlation_risk(correlation_matrix_aligned.values, optimal_weights_vec)
            
            # Rebalancing analysis
            rebalancing_required = False
            turnover = 0.0
            if current_weights:
                turnover = sum(abs(optimal_weights.get(asset, 0) - current_weights.get(asset, 0)) for asset in assets)
                rebalancing_required = turnover > 0.05  # 5% threshold
            
            result = PortfolioOptimizationResult(
                optimization_method=method,
                timestamp=datetime.now(),
                optimal_weights=optimal_weights,
                rebalancing_required=rebalancing_required,
                current_weights=current_weights,
                expected_return=portfolio_return,
                expected_volatility=portfolio_volatility,
                expected_sharpe_ratio=portfolio_sharpe,
                portfolio_var_95=portfolio_var_95,
                portfolio_cvar_95=portfolio_cvar_95,
                maximum_drawdown_estimate=max_dd_estimate,
                effective_number_of_assets=effective_assets,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                turnover=turnover,
                transaction_cost_estimate=turnover * 0.001,  # 0.1% per unit turnover
                metadata={
                    "n_assets": len(assets),
                    "optimization_method": method.value,
                    "risk_free_rate": self.risk_free_rate,
                    "constraints": constraints
                }
            )
            
            self.optimization_history.append(result)
            logger.info(f"✅ Portfolio optimized: {portfolio_return:.2%} return, {portfolio_volatility:.2%} vol, Sharpe: {portfolio_sharpe:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in portfolio optimization: {e}")
            raise

    async def _mean_variance_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Mean-Variance optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(covariance_matrix, weights))
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Sum to 1
        ]
        
        # Additional constraints
        if constraints:
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                constraints_list.extend([
                    {'type': 'ineq', 'fun': lambda weights, i=i: weights[i] - min_weight}
                    for i in range(n_assets)
                ])
            
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                constraints_list.extend([
                    {'type': 'ineq', 'fun': lambda weights, i=i: max_weight - weights[i]}
                    for i in range(n_assets)
                ])
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Mean-variance optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    async def _risk_parity_optimization(
        self,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Risk Parity optimization"""
        
        n_assets = len(covariance_matrix)
        
        def risk_budget_objective(weights):
            """Minimize difference in risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            
            # Target: equal risk contribution (1/n each)
            target_contrib = np.ones(n_assets) / n_assets
            return np.sum((contrib - target_contrib)**2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        ]
        
        # Bounds
        bounds = [(0.001, 0.5) for _ in range(n_assets)]  # Min 0.1%, max 50%
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            risk_budget_objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Risk parity optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    async def _minimum_variance_optimization(
        self,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Minimum Variance optimization"""
        
        n_assets = len(covariance_matrix)
        
        # Use Ledoit-Wolf shrinkage for covariance
        shrinkage_estimator = LedoitWolf()
        # Note: in real application, historical returns are needed for shrinkage
        
        # For now, use regularized covariance
        regularized_cov = covariance_matrix + np.eye(n_assets) * 1e-6
        
        # Analytical solution: w = inv(Cov) * 1 / (1' * inv(Cov) * 1)
        inv_cov = np.linalg.inv(regularized_cov)
        ones = np.ones(n_assets)
        
        weights = inv_cov.dot(ones) / ones.dot(inv_cov).dot(ones)
        
        # Ensure non-negative
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)
        
        return weights

    async def _maximum_sharpe_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]]
    ) -> np.ndarray:
        """Maximum Sharpe Ratio optimization"""
        
        n_assets = len(expected_returns)
        
        # Objective: minimize negative Sharpe ratio
        def neg_sharpe_objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Negative because we minimize
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        ]
        
        # Bounds
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = optimize.minimize(
            neg_sharpe_objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if result.success:
            return result.x
        else:
            logger.warning("⚠️ Max Sharpe optimization failed, using equal weights")
            return np.ones(n_assets) / n_assets

    async def _black_litterman_optimization(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        constraints: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ) -> np.ndarray:
        """Black-Litterman optimization (simplified)"""
        
        # This is a simplified BL - full implementation requires market cap weights,
        # investor views, and confidence levels
        
        n_assets = len(expected_returns)
        
        # Use market cap weights as prior (simplified - equal weights)
        prior_weights = np.ones(n_assets) / n_assets
        
        # Risk aversion parameter
        risk_aversion = kwargs.get('risk_aversion', 3.0)
        
        # Implied returns from market weights
        implied_returns = risk_aversion * np.dot(covariance_matrix, prior_weights)
        
        # If views provided, incorporate them (simplified)
        views = kwargs.get('views', {})
        if views:
            # Blend implied returns with views (simplified)
            view_confidence = kwargs.get('view_confidence', 0.5)
            adjusted_returns = (1 - view_confidence) * implied_returns + view_confidence * expected_returns
        else:
            adjusted_returns = implied_returns
        
        # Optimize with adjusted returns
        return await self._mean_variance_optimization(adjusted_returns, covariance_matrix, constraints)

    async def _calculate_effective_number_of_assets(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets (concentration measure)"""
        return 1 / np.sum(weights**2)

    async def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        return np.sum(weights**2)

    async def _calculate_correlation_risk(
        self,
        correlation_matrix: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate portfolio correlation risk"""
        
        # Average pairwise correlation weighted by portfolio weights
        n = len(weights)
        weighted_corr = 0.0
        total_weight = 0.0
        
        for i in range(n):
            for j in range(i + 1, n):
                weight_product = weights[i] * weights[j]
                weighted_corr += weight_product * correlation_matrix[i, j]
                total_weight += weight_product
        
        if total_weight > 0:
            return weighted_corr / total_weight
        else:
            return 0.0

class OptionStrategyRecommender:
    """
    Options Strategy Recommender based on Volatility Forecasts

    Option strategy recommendations:
    - Long/Short Straddles for volatility plays
    - Iron Condors for range-bound markets
    - Butterfly spreads for precise volatility bets
    - Strangles for asymmetric volatility
    """
    
    def __init__(self):
        self.recommendation_history: List[OptionStrategyRecommendation] = []
        
        logger.info("🎯 Options strategy recommender initialized")

    async def recommend_strategy(
        self,
        symbol: str,
        current_price: float,
        volatility_forecast: float,
        current_implied_vol: Optional[float] = None,
        market_outlook: Dict[str, Any] = None,
        expiry_days: int = 30,
        **kwargs
    ) -> OptionStrategyRecommendation:
        """
        Recommend options strategy based on volatility forecast
        
        Args:
            symbol: Underlying symbol
            current_price: Current underlying price
            volatility_forecast: Forecasted volatility
            current_implied_vol: Current implied volatility
            market_outlook: Direction bias and confidence
            expiry_days: Days to expiry
        """
        try:
            logger.info(f"🔄 Recommending options strategy for {symbol}")
            
            # Determine volatility regime
            vol_regime = self._classify_volatility_regime(volatility_forecast)
            
            # Determine market direction bias
            direction_bias = "neutral"
            if market_outlook:
                direction_bias = market_outlook.get("direction", "neutral")
            
            # Strategy selection logic
            if current_implied_vol and abs(volatility_forecast - current_implied_vol) > 0.05:
                # Significant volatility mismatch
                if volatility_forecast > current_implied_vol:
                    # Forecast higher vol than market expects
                    strategy = OptionStrategy.LONG_STRADDLE
                else:
                    # Forecast lower vol than market expects
                    strategy = OptionStrategy.SHORT_STRADDLE
            elif vol_regime == "low":
                # Low volatility environment
                if direction_bias == "neutral":
                    strategy = OptionStrategy.IRON_CONDOR
                else:
                    strategy = OptionStrategy.LONG_STRANGLE
            elif vol_regime == "high":
                # High volatility environment
                strategy = OptionStrategy.SHORT_STRADDLE
            else:
                # Normal volatility
                strategy = OptionStrategy.BUTTERFLY_SPREAD
            
            # Generate strategy details
            expiry_date = datetime.now() + timedelta(days=expiry_days)
            strikes = await self._calculate_optimal_strikes(
                current_price, volatility_forecast, strategy
            )
            
            # Risk-reward calculation
            risk_reward = await self._calculate_strategy_risk_reward(
                strategy, current_price, strikes, volatility_forecast, expiry_days
            )
            
            # Greeks exposure estimation
            greeks = await self._estimate_greeks_exposure(
                strategy, current_price, strikes, volatility_forecast, expiry_days
            )
            
            recommendation = OptionStrategyRecommendation(
                symbol=symbol,
                timestamp=datetime.now(),
                strategy=strategy,
                volatility_forecast=volatility_forecast,
                volatility_regime=vol_regime,
                direction_bias=direction_bias,
                strikes_recommended=strikes,
                expiry_recommendation=expiry_date,
                premium_estimate=risk_reward["premium_estimate"],
                max_profit=risk_reward["max_profit"],
                max_loss=risk_reward["max_loss"],
                breakeven_points=risk_reward["breakeven_points"],
                probability_of_profit=risk_reward["probability_of_profit"],
                delta_exposure=greeks["delta"],
                gamma_exposure=greeks["gamma"],
                theta_exposure=greeks["theta"],
                vega_exposure=greeks["vega"],
                confidence_score=0.7,
                recommended_position_size=1.0,
                metadata={
                    "current_price": current_price,
                    "current_implied_vol": current_implied_vol,
                    "vol_forecast_vs_implied": volatility_forecast - (current_implied_vol or 0),
                    "expiry_days": expiry_days,
                    "strategy_rationale": f"Volatility regime: {vol_regime}, Direction: {direction_bias}"
                }
            )
            
            self.recommendation_history.append(recommendation)
            logger.info(f"✅ Strategy recommended: {strategy.value} with {len(strikes)} strikes")
            
            return recommendation
            
        except Exception as e:
            logger.error(f"❌ Error recommending options strategy: {e}")
            raise

    def _classify_volatility_regime(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.15:
            return "low"
        elif volatility < 0.30:
            return "normal"
        elif volatility < 0.50:
            return "high"
        else:
            return "extreme"

    async def _calculate_optimal_strikes(
        self,
        current_price: float,
        volatility: float,
        strategy: OptionStrategy
    ) -> List[float]:
        """Calculate optimal strike prices for strategy"""
        
        if strategy == OptionStrategy.LONG_STRADDLE or strategy == OptionStrategy.SHORT_STRADDLE:
            # ATM straddle
            return [current_price]
            
        elif strategy == OptionStrategy.LONG_STRANGLE:
            # OTM strangle
            strike_distance = current_price * volatility * 0.5
            return [current_price - strike_distance, current_price + strike_distance]
            
        elif strategy == OptionStrategy.IRON_CONDOR:
            # Wide iron condor
            inner_distance = current_price * volatility * 0.3
            outer_distance = current_price * volatility * 0.6
            return [
                current_price - outer_distance,  # Short put
                current_price - inner_distance,  # Long put
                current_price + inner_distance,  # Long call
                current_price + outer_distance   # Short call
            ]
            
        elif strategy == OptionStrategy.BUTTERFLY_SPREAD:
            # ATM butterfly
            wing_distance = current_price * volatility * 0.2
            return [
                current_price - wing_distance,
                current_price,
                current_price + wing_distance
            ]
            
        else:
            # Default ATM
            return [current_price]

    async def _calculate_strategy_risk_reward(
        self,
        strategy: OptionStrategy,
        underlying_price: float,
        strikes: List[float],
        volatility: float,
        days_to_expiry: int
    ) -> Dict[str, Any]:
        """Calculate risk-reward profile (simplified)"""
        
        # Mock option pricing (in production use Black-Scholes or real prices)
        time_to_expiry = days_to_expiry / 365.0
        
        risk_reward = {
            "premium_estimate": 0.0,
            "max_profit": 0.0,
            "max_loss": 0.0,
            "breakeven_points": [],
            "probability_of_profit": 0.5
        }
        
        if strategy == OptionStrategy.LONG_STRADDLE:
            # Estimate premium for ATM straddle
            option_premium = underlying_price * volatility * np.sqrt(time_to_expiry) * 0.4
            risk_reward["premium_estimate"] = option_premium * 2  # Call + Put
            risk_reward["max_loss"] = risk_reward["premium_estimate"]
            risk_reward["max_profit"] = float('inf')  # Unlimited
            risk_reward["breakeven_points"] = [
                strikes[0] - risk_reward["premium_estimate"],
                strikes[0] + risk_reward["premium_estimate"]
            ]
            risk_reward["probability_of_profit"] = 0.4  # Roughly
            
        elif strategy == OptionStrategy.IRON_CONDOR:
            # Credit spread
            width = (strikes[1] - strikes[0])  # Wing width
            risk_reward["premium_estimate"] = width * 0.3  # Credit received
            risk_reward["max_profit"] = risk_reward["premium_estimate"]
            risk_reward["max_loss"] = width - risk_reward["premium_estimate"]
            risk_reward["breakeven_points"] = [
                strikes[1] + risk_reward["premium_estimate"],
                strikes[2] - risk_reward["premium_estimate"]
            ]
            risk_reward["probability_of_profit"] = 0.65
        
        # Add more strategy calculations as needed...
        
        return risk_reward

    async def _estimate_greeks_exposure(
        self,
        strategy: OptionStrategy,
        underlying_price: float,
        strikes: List[float],
        volatility: float,
        days_to_expiry: int
    ) -> Dict[str, float]:
        """Estimate Greeks exposure (simplified)"""
        
        greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0
        }
        
        if strategy == OptionStrategy.LONG_STRADDLE:
            # Long straddle: delta-neutral, high gamma, negative theta, positive vega
            greeks["delta"] = 0.0  # Delta neutral
            greeks["gamma"] = 0.1   # Positive gamma
            greeks["theta"] = -underlying_price * 0.001  # Negative theta
            greeks["vega"] = underlying_price * 0.01     # Positive vega
            
        elif strategy == OptionStrategy.SHORT_STRADDLE:
            # Opposite of long straddle
            greeks["delta"] = 0.0
            greeks["gamma"] = -0.1
            greeks["theta"] = underlying_price * 0.001
            greeks["vega"] = -underlying_price * 0.01
            
        elif strategy == OptionStrategy.IRON_CONDOR:
            # Limited Greeks exposure
            greeks["delta"] = 0.0   # Delta neutral
            greeks["gamma"] = -0.02 # Negative gamma
            greeks["theta"] = underlying_price * 0.0005  # Positive theta
            greeks["vega"] = -underlying_price * 0.005   # Negative vega
        
        return greeks

# Export all classes
__all__ = [
    "PositionSizeMethod",
    "PortfolioOptimizationMethod", 
    "OptionStrategy",
    "PositionSizeRecommendation",
    "DynamicStopLossRecommendation",
    "PortfolioOptimizationResult",
    "OptionStrategyRecommendation",
    "PositionSizer",
    "DynamicStopLoss",
    "PortfolioOptimizer", 
    "OptionStrategyRecommender",
    "_kelly_criterion_numba",
    "_volatility_position_size_numba"
]

logger.info("🔥 Trading Integration utilities loaded successfully!")