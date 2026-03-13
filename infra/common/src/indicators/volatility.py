"""
Volatility Indicators Module

High-performance volatility indicators for crypto trading analysis.
Implements enterprise patterns with Numba optimization.

Available indicators:
- Standard Deviation
- True Range and Average True Range
- Keltner Channels
- Donchian Channels
- Historical Volatility
- Relative Volatility Index
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass
import logging

# Optional performance imports
try:
 import numba
 from numba import jit
 HAS_NUMBA = True
except ImportError:
 HAS_NUMBA = False
 def jit(*args, **kwargs):
 def decorator(func):
 return func
 return decorator

try:
 import talib
 HAS_TALIB = True
except ImportError:
 HAS_TALIB = False


# Result types
class KeltnerChannelsResult(NamedTuple):
 """Keltner Channels result"""
 upper_channel: float
 middle_line: float
 lower_channel: float


class DonchianChannelsResult(NamedTuple):
 """Donchian Channels result"""
 upper_channel: float
 middle_line: float
 lower_channel: float


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_std(values: np.ndarray, period: int) -> float:
 """Fast standard deviation calculation with Numba"""
 if len(values) < period:
 return 0.0
 recent_values = values[-period:]
 return np.std(recent_values)


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
 """Fast True Range calculation for multiple periods"""
 if len(high) < 2:
 return np.array([0.0])

 tr_values = np.zeros(len(high) - 1)
 for i in range(1, len(high)):
 tr1 = high[i] - low[i]
 tr2 = abs(high[i] - close[i-1])
 tr3 = abs(low[i] - close[i-1])
 tr_values[i-1] = max(tr1, max(tr2, tr3))

 return tr_values


def calculate_standard_deviation(
 prices: Union[List[float], np.ndarray],
 period: int = 20
) -> float:
 """
 Calculate standard deviation of price series

 Args:
 prices: Price series
 period: Number of periods

 Returns:
 Standard deviation value
 """
 if not prices or len(prices) < period:
 return 0.0

 prices_array = np.array(prices, dtype=np.float64)

 if HAS_TALIB and len(prices) >= period:
 result = talib.STDDEV(prices_array, timeperiod=period)
 return float(result[-1]) if not np.isnan(result[-1]) else 0.0

 return float(_fast_std(prices_array, period))


def calculate_true_range(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray]
) -> float:
 """
 Calculate True Range for current period

 Args:
 high: High prices
 low: Low prices
 close: Close prices

 Returns:
 True Range value for last period
 """
 if not all([high, low, close]) or len(high) < 2:
 return 0.0

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)
 close_array = np.array(close, dtype=np.float64)

 if HAS_TALIB and len(high) >= 2:
 result = talib.TRANGE(high_array, low_array, close_array)
 return float(result[-1]) if not np.isnan(result[-1]) else 0.0

 # Manual calculation for last period
 if len(high) >= 2:
 tr1 = high_array[-1] - low_array[-1]
 tr2 = abs(high_array[-1] - close_array[-2])
 tr3 = abs(low_array[-1] - close_array[-2])
 return float(max(tr1, tr2, tr3))

 return float(high_array[-1] - low_array[-1])


def calculate_average_true_range(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 period: int = 14
) -> float:
 """
 Calculate Average True Range

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 period: Number of periods

 Returns:
 ATR value
 """
 if not all([high, low, close]) or len(high) < period:
 return 0.0

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)
 close_array = np.array(close, dtype=np.float64)

 if HAS_TALIB and len(high) >= period:
 result = talib.ATR(high_array, low_array, close_array, timeperiod=period)
 return float(result[-1]) if not np.isnan(result[-1]) else 0.0

 # Manual calculation using Numba optimized function
 tr_values = _fast_true_range(high_array, low_array, close_array)

 if len(tr_values) >= period:
 return float(np.mean(tr_values[-period:]))
 else:
 return float(np.mean(tr_values)) if len(tr_values) > 0 else 0.0


def calculate_keltner_channels(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 period: int = 20,
 multiplier: float = 2.0
) -> KeltnerChannelsResult:
 """
 Calculate Keltner Channels

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 period: EMA period
 multiplier: ATR multiplier

 Returns:
 KeltnerChannelsResult with upper, middle, lower channels
 """
 if not all([high, low, close]) or len(close) < period:
 price = float(close[-1]) if close else 0.0
 return KeltnerChannelsResult(price, price, price)

 # Import here to avoid circular imports
 from .technical import calculate_ema

 # Calculate middle line (EMA)
 middle_line = calculate_ema(close, period)

 # Calculate ATR
 atr = calculate_average_true_range(high, low, close, period)

 # Calculate channels
 upper_channel = middle_line + (atr * multiplier)
 lower_channel = middle_line - (atr * multiplier)

 return KeltnerChannelsResult(
 float(upper_channel),
 float(middle_line),
 float(lower_channel)
 )


def calculate_donchian_channels(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 period: int = 20
) -> DonchianChannelsResult:
 """
 Calculate Donchian Channels

 Args:
 high: High prices
 low: Low prices
 period: Lookback period

 Returns:
 DonchianChannelsResult with upper, middle, lower channels
 """
 if not all([high, low]) or len(high) < period:
 high_val = float(high[-1]) if high else 0.0
 low_val = float(low[-1]) if low else 0.0
 middle = (high_val + low_val) / 2.0
 return DonchianChannelsResult(high_val, middle, low_val)

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)

 # Calculate channels
 upper_channel = float(np.max(high_array[-period:]))
 lower_channel = float(np.min(low_array[-period:]))
 middle_line = (upper_channel + lower_channel) / 2.0

 return DonchianChannelsResult(
 upper_channel,
 middle_line,
 lower_channel
 )


def calculate_historical_volatility(
 prices: Union[List[float], np.ndarray],
 period: int = 30,
 annualize: bool = True
) -> float:
 """
 Calculate Historical Volatility

 Args:
 prices: Price series
 period: Number of periods
 annualize: Whether to annualize the volatility

 Returns:
 Historical volatility value
 """
 if not prices or len(prices) < period + 1:
 return 0.0

 prices_array = np.array(prices, dtype=np.float64)

 # Calculate log returns
 log_returns = np.log(prices_array[1:] / prices_array[:-1])

 # Take recent returns
 recent_returns = log_returns[-period:] if len(log_returns) >= period else log_returns

 # Calculate standard deviation
 volatility = float(np.std(recent_returns))

 # Annualize if requested (assuming daily data)
 if annualize:
 volatility *= np.sqrt(365)

 return volatility


def calculate_relative_volatility_index(
 prices: Union[List[float], np.ndarray],
 period: int = 14,
 volatility_period: int = 10
) -> float:
 """
 Calculate Relative Volatility Index (RVI)

 Args:
 prices: Price series
 period: RSI period
 volatility_period: Standard deviation period

 Returns:
 RVI value (0-100)
 """
 if not prices or len(prices) < period + volatility_period:
 return 50.0

 prices_array = np.array(prices, dtype=np.float64)

 # Calculate standard deviations for up and down moves
 std_devs = []
 directions = []

 for i in range(volatility_period, len(prices_array)):
 current_std = np.std(prices_array[i-volatility_period:i])
 std_devs.append(current_std)

 if i > volatility_period:
 direction = 1 if prices_array[i] > prices_array[i-1] else 0
 directions.append(direction)

 if len(std_devs) < period or len(directions) < period:
 return 50.0

 # Separate volatilities for up and down days
 up_volatilities = []
 down_volatilities = []

 for i, direction in enumerate(directions[-period:]):
 if i < len(std_devs):
 if direction == 1:
 up_volatilities.append(std_devs[i])
 down_volatilities.append(0.0)
 else:
 up_volatilities.append(0.0)
 down_volatilities.append(std_devs[i])

 # Calculate average volatilities
 avg_up_vol = np.mean(up_volatilities) if up_volatilities else 0.0
 avg_down_vol = np.mean(down_volatilities) if down_volatilities else 0.0

 if avg_down_vol == 0:
 return 100.0

 # Calculate RVI
 rs = avg_up_vol / avg_down_vol
 rvi = 100.0 - (100.0 / (1.0 + rs))

 return float(rvi)


@dataclass
class VolatilityConfig:
 """Configuration for volatility indicators"""
 std_period: int = 20
 atr_period: int = 14
 keltner_period: int = 20
 keltner_multiplier: float = 2.0
 donchian_period: int = 20
 hv_period: int = 30
 hv_annualize: bool = True
 rvi_period: int = 14
 rvi_volatility_period: int = 10


class VolatilityIndicators:
 """
 High-performance volatility indicators calculator

 Calculates multiple volatility indicators efficiently using
 vectorized operations and optional Numba acceleration.
 """

 def __init__(self, config: Optional[VolatilityConfig] = None):
 self.config = config or VolatilityConfig
 self.logger = logging.getLogger(__name__)

 def calculate_all(
 self,
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Optional[Union[List[float], np.ndarray]] = None
 ) -> Dict[str, float]:
 """
 Calculate all volatility indicators

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume data (optional)

 Returns:
 Dictionary with all volatility indicator values
 """
 results = {}

 try:
 # Standard Deviation
 results["std"] = calculate_standard_deviation(close, self.config.std_period)

 # True Range and ATR
 results["tr"] = calculate_true_range(high, low, close)
 results["atr"] = calculate_average_true_range(high, low, close, self.config.atr_period)

 # Keltner Channels
 keltner = calculate_keltner_channels(
 high, low, close,
 self.config.keltner_period,
 self.config.keltner_multiplier
 )
 results["keltner_upper"] = keltner.upper_channel
 results["keltner_middle"] = keltner.middle_line
 results["keltner_lower"] = keltner.lower_channel

 # Donchian Channels
 donchian = calculate_donchian_channels(high, low, self.config.donchian_period)
 results["donchian_upper"] = donchian.upper_channel
 results["donchian_middle"] = donchian.middle_line
 results["donchian_lower"] = donchian.lower_channel

 # Historical Volatility
 results["hv"] = calculate_historical_volatility(
 close,
 self.config.hv_period,
 self.config.hv_annualize
 )

 # Relative Volatility Index
 results["rvi"] = calculate_relative_volatility_index(
 close,
 self.config.rvi_period,
 self.config.rvi_volatility_period
 )

 except Exception as e:
 self.logger.error(f"Error calculating volatility indicators: {e}")
 # Return zeros on error
 results = {
 "std": 0.0, "tr": 0.0, "atr": 0.0,
 "keltner_upper": 0.0, "keltner_middle": 0.0, "keltner_lower": 0.0,
 "donchian_upper": 0.0, "donchian_middle": 0.0, "donchian_lower": 0.0,
 "hv": 0.0, "rvi": 50.0
 }

 return results


# Export all functions and classes
__all__ = [
 # Core calculation functions
 "calculate_standard_deviation",
 "calculate_true_range",
 "calculate_average_true_range",
 "calculate_keltner_channels",
 "calculate_donchian_channels",
 "calculate_historical_volatility",
 "calculate_relative_volatility_index",

 # Result types
 "KeltnerChannelsResult",
 "DonchianChannelsResult",

 # Configuration and main class
 "VolatilityConfig",
 "VolatilityIndicators"
]