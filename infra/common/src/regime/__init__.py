"""
Market Regime Classification Module
Adaptive Trading

This module provides lightweight regime classification for the 768-dimensional state vector.
Enables autonomous AI to adapt trading strategies based on market conditions.

Key Features:
- Volatility regime classification (4 dims)
- Trend regime classification (4 dims)
- Time-based features (2 dims)
- <2ms latency for real-time trading
- Interpretable regime labels (for explainability)

Architecture:
- volatility.py: Low/Medium/High/Extreme volatility detection
- trend.py: Strong Up/Weak Up/Sideways/Weak Down/Strong Down classification
- market_hours.py: Trading session, day-of-week encoding

Use Cases:
- Circuit breakers (halt trading in extreme volatility)
- Position sizing (reduce size in high volatility)
- Strategy selection (momentum in trending, mean-reversion in ranging)
- Risk management (tighter stops in volatile regimes)

Usage:
 from ml_common.regime import classify_volatility_regime, classify_trend_regime

 # Historical prices (168h window)
 prices = [50000, 50100, 50200, ...]

 # Volatility regime (4 dims)
 vol_features = extract_volatility_features(prices, window=24)
 # Returns: [regime_id, volatility_percentile, regime_duration, regime_stability]

 # Trend regime (4 dims)
 trend_features = extract_trend_features(prices, window=24)
 # Returns: [trend_id, trend_strength, trend_duration, trend_acceleration]

 # Time features (2 dims)
 time_features = extract_time_features(timestamp)
 # Returns: [session_id, day_of_week_normalized]
"""

from .volatility import (
 classify_volatility_regime,
 calculate_volatility_percentile,
 calculate_regime_duration,
 calculate_regime_stability,
 extract_volatility_features,
)

from .trend import (
 classify_trend_regime,
 calculate_trend_strength,
 calculate_trend_duration,
 calculate_trend_acceleration,
 extract_trend_features,
)

from .market_hours import (
 classify_trading_session,
 normalize_day_of_week,
 extract_time_features,
)

__all__ = [
 # Volatility (4 functions)
 "classify_volatility_regime",
 "calculate_volatility_percentile",
 "calculate_regime_duration",
 "calculate_regime_stability",
 "extract_volatility_features",

 # Trend (4 functions)
 "classify_trend_regime",
 "calculate_trend_strength",
 "calculate_trend_duration",
 "calculate_trend_acceleration",
 "extract_trend_features",

 # Time (3 functions)
 "classify_trading_session",
 "normalize_day_of_week",
 "extract_time_features",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
__note__ = "Regime classification for adaptive trading strategies"
