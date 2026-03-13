"""
Technical Indicators Module for ML Common

High-performance technical analysis indicators implementing .
Consolidates and optimizes indicator calculations from multiple ML packages.

Available indicators:
- Moving averages: SMA, EMA, WMA
- Momentum: RSI, MACD, Stochastic, Williams %R
- Volatility: Bollinger Bands, ATR, Standard Deviation
- Volume: OBV, VWAP, MFI, A/D Line
- Trend: ADX, PSAR, Ichimoku components
"""

from .technical import (
 # Core calculation functions
 calculate_sma,
 calculate_ema,
 calculate_wma,
 calculate_rsi,
 calculate_macd,
 calculate_bollinger_bands,
 calculate_atr,
 calculate_stochastic,
 calculate_williams_r,

 # Main indicator class
 TechnicalIndicators,
 IndicatorConfig,

 # Result types
 MADCResult,
 BollingerBandsResult,
 StochasticResult
)

from .volatility import (
 calculate_standard_deviation,
 calculate_true_range,
 calculate_average_true_range,
 calculate_keltner_channels,
 calculate_donchian_channels,
 VolatilityIndicators
)

from .volume import (
 calculate_obv,
 calculate_vwap,
 calculate_mfi,
 calculate_ad_line,
 calculate_volume_profile,
 VolumeIndicators
)

__all__ = [
 # Technical indicators (moving averages, momentum)
 "calculate_sma",
 "calculate_ema",
 "calculate_wma",
 "calculate_rsi",
 "calculate_macd",
 "calculate_bollinger_bands",
 "calculate_atr",
 "calculate_stochastic",
 "calculate_williams_r",
 "TechnicalIndicators",
 "IndicatorConfig",
 "MADCResult",
 "BollingerBandsResult",
 "StochasticResult",

 # Volatility indicators
 "calculate_standard_deviation",
 "calculate_true_range",
 "calculate_average_true_range",
 "calculate_keltner_channels",
 "calculate_donchian_channels",
 "VolatilityIndicators",

 # Volume indicators
 "calculate_obv",
 "calculate_vwap",
 "calculate_mfi",
 "calculate_ad_line",
 "calculate_volume_profile",
 "VolumeIndicators"
]