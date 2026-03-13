"""
Volume Indicators Module

High-performance volume-based indicators for crypto trading analysis.
Implements enterprise patterns with Numba optimization.

Available indicators:
- On-Balance Volume (OBV)
- Volume Weighted Average Price (VWAP)
- Money Flow Index (MFI)
- Accumulation/Distribution Line (A/D Line)
- Chaikin Money Flow (CMF)
- Volume Profile
- Volume Rate of Change
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
class VolumeProfileResult(NamedTuple):
 """Volume Profile result"""
 price_levels: List[float]
 volume_levels: List[float]
 poc: float # Point of Control
 value_area_high: float
 value_area_low: float


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_obv(prices: np.ndarray, volumes: np.ndarray) -> float:
 """Fast OBV calculation with Numba optimization"""
 if len(prices) < 2 or len(volumes) < 2:
 return 0.0

 obv = 0.0
 for i in range(1, min(len(prices), len(volumes))):
 if prices[i] > prices[i-1]:
 obv += volumes[i]
 elif prices[i] < prices[i-1]:
 obv -= volumes[i]
 # If prices[i] == prices[i-1], OBV unchanged

 return obv


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_vwap(prices: np.ndarray, volumes: np.ndarray, period: int) -> float:
 """Fast VWAP calculation with Numba optimization"""
 if len(prices) < period or len(volumes) < period:
 return 0.0

 recent_prices = prices[-period:]
 recent_volumes = volumes[-period:]

 total_pv = 0.0
 total_volume = 0.0

 for i in range(len(recent_prices)):
 pv = recent_prices[i] * recent_volumes[i]
 total_pv += pv
 total_volume += recent_volumes[i]

 if total_volume == 0:
 return 0.0

 return total_pv / total_volume


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
 volumes: np.ndarray, period: int) -> float:
 """Fast MFI calculation with Numba optimization"""
 if len(high) < period + 1:
 return 50.0

 # Calculate typical prices
 typical_prices = (high + low + close) / 3.0

 # Calculate raw money flow
 money_flows = typical_prices * volumes

 # Separate positive and negative money flows
 positive_flow = 0.0
 negative_flow = 0.0

 for i in range(1, min(len(typical_prices), period + 1)):
 if typical_prices[-(i)] > typical_prices[-(i+1)]:
 positive_flow += money_flows[-(i)]
 elif typical_prices[-(i)] < typical_prices[-(i+1)]:
 negative_flow += money_flows[-(i)]

 if negative_flow == 0:
 return 100.0

 money_ratio = positive_flow / negative_flow
 mfi = 100.0 - (100.0 / (1.0 + money_ratio))

 return mfi


def calculate_obv(
 prices: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray]
) -> float:
 """
 Calculate On-Balance Volume

 Args:
 prices: Price series
 volumes: Volume series

 Returns:
 OBV value
 """
 if not prices or not volumes or len(prices) < 2 or len(volumes) < 2:
 return 0.0

 prices_array = np.array(prices, dtype=np.float64)
 volumes_array = np.array(volumes, dtype=np.float64)

 if HAS_TALIB and len(prices) >= 2:
 result = talib.OBV(prices_array, volumes_array)
 return float(result[-1]) if not np.isnan(result[-1]) else 0.0

 return float(_fast_obv(prices_array, volumes_array))


def calculate_vwap(
 prices: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray],
 period: int = 20
) -> float:
 """
 Calculate Volume Weighted Average Price

 Args:
 prices: Price series
 volumes: Volume series
 period: Number of periods

 Returns:
 VWAP value
 """
 if not prices or not volumes or len(prices) < period or len(volumes) < period:
 return float(prices[-1]) if prices else 0.0

 prices_array = np.array(prices, dtype=np.float64)
 volumes_array = np.array(volumes, dtype=np.float64)

 return float(_fast_vwap(prices_array, volumes_array, period))


def calculate_mfi(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray],
 period: int = 14
) -> float:
 """
 Calculate Money Flow Index

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume series
 period: Number of periods

 Returns:
 MFI value (0-100)
 """
 if not all([high, low, close, volumes]) or len(high) < period + 1:
 return 50.0

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)
 close_array = np.array(close, dtype=np.float64)
 volumes_array = np.array(volumes, dtype=np.float64)

 if HAS_TALIB and len(high) > period:
 result = talib.MFI(high_array, low_array, close_array, volumes_array, timeperiod=period)
 return float(result[-1]) if not np.isnan(result[-1]) else 50.0

 return float(_fast_mfi(high_array, low_array, close_array, volumes_array, period))


def calculate_ad_line(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray]
) -> float:
 """
 Calculate Accumulation/Distribution Line

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume series

 Returns:
 A/D Line value
 """
 if not all([high, low, close, volumes]) or len(high) < 1:
 return 0.0

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)
 close_array = np.array(close, dtype=np.float64)
 volumes_array = np.array(volumes, dtype=np.float64)

 if HAS_TALIB and len(high) >= 1:
 result = talib.AD(high_array, low_array, close_array, volumes_array)
 return float(result[-1]) if not np.isnan(result[-1]) else 0.0

 # Manual calculation
 ad_line = 0.0
 for i in range(len(high_array)):
 if high_array[i] != low_array[i]:
 clv = ((close_array[i] - low_array[i]) - (high_array[i] - close_array[i])) / (high_array[i] - low_array[i])
 ad_line += clv * volumes_array[i]

 return float(ad_line)


def calculate_chaikin_money_flow(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray],
 period: int = 20
) -> float:
 """
 Calculate Chaikin Money Flow

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume series
 period: Number of periods

 Returns:
 CMF value (-1 to 1)
 """
 if not all([high, low, close, volumes]) or len(high) < period:
 return 0.0

 high_array = np.array(high, dtype=np.float64)
 low_array = np.array(low, dtype=np.float64)
 close_array = np.array(close, dtype=np.float64)
 volumes_array = np.array(volumes, dtype=np.float64)

 # Calculate money flow multiplier for each period
 money_flow_multipliers = []
 money_flow_volumes = []

 for i in range(len(high_array)):
 if high_array[i] != low_array[i]:
 mfm = ((close_array[i] - low_array[i]) - (high_array[i] - close_array[i])) / (high_array[i] - low_array[i])
 else:
 mfm = 0.0

 money_flow_multipliers.append(mfm)
 money_flow_volumes.append(mfm * volumes_array[i])

 # Calculate CMF for recent period
 recent_mfv = money_flow_volumes[-period:] if len(money_flow_volumes) >= period else money_flow_volumes
 recent_volumes = volumes_array[-period:] if len(volumes_array) >= period else volumes_array

 total_mfv = sum(recent_mfv)
 total_volume = sum(recent_volumes)

 if total_volume == 0:
 return 0.0

 cmf = total_mfv / total_volume
 return float(cmf)


def calculate_volume_profile(
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray],
 num_bins: int = 20,
 period: int = 100
) -> VolumeProfileResult:
 """
 Calculate Volume Profile

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume series
 num_bins: Number of price bins
 period: Lookback period

 Returns:
 VolumeProfileResult with profile data
 """
 if not all([high, low, close, volumes]) or len(high) < period:
 current_price = float(close[-1]) if close else 0.0
 return VolumeProfileResult(
 price_levels=[current_price],
 volume_levels=[0.0],
 poc=current_price,
 value_area_high=current_price,
 value_area_low=current_price
 )

 # Take recent data
 recent_high = np.array(high[-period:], dtype=np.float64)
 recent_low = np.array(low[-period:], dtype=np.float64)
 recent_close = np.array(close[-period:], dtype=np.float64)
 recent_volumes = np.array(volumes[-period:], dtype=np.float64)

 # Define price range
 min_price = np.min(recent_low)
 max_price = np.max(recent_high)
 price_range = max_price - min_price

 if price_range == 0:
 current_price = float(recent_close[-1])
 return VolumeProfileResult(
 price_levels=[current_price],
 volume_levels=[0.0],
 poc=current_price,
 value_area_high=current_price,
 value_area_low=current_price
 )

 # Create price bins
 bin_size = price_range / num_bins
 price_levels = [min_price + i * bin_size for i in range(num_bins + 1)]
 volume_levels = [0.0] * num_bins

 # Distribute volume across price levels
 for i in range(len(recent_high)):
 # Assume volume is distributed evenly across high-low range
 bar_min = recent_low[i]
 bar_max = recent_high[i]
 bar_volume = recent_volumes[i]

 # Find which bins this bar touches
 start_bin = max(0, int((bar_min - min_price) / bin_size))
 end_bin = min(num_bins - 1, int((bar_max - min_price) / bin_size))

 # Distribute volume proportionally
 bins_touched = end_bin - start_bin + 1
 volume_per_bin = bar_volume / bins_touched if bins_touched > 0 else bar_volume

 for bin_idx in range(start_bin, end_bin + 1):
 if 0 <= bin_idx < num_bins:
 volume_levels[bin_idx] += volume_per_bin

 # Find Point of Control (POC) - price level with highest volume
 max_volume_idx = np.argmax(volume_levels)
 poc = price_levels[max_volume_idx] + bin_size / 2 # Middle of bin

 # Calculate Value Area (70% of total volume)
 total_volume = sum(volume_levels)
 target_volume = total_volume * 0.70

 # Start from POC and expand outwards
 value_area_volume = volume_levels[max_volume_idx]
 left_idx = max_volume_idx
 right_idx = max_volume_idx

 while value_area_volume < target_volume and (left_idx > 0 or right_idx < num_bins - 1):
 left_volume = volume_levels[left_idx - 1] if left_idx > 0 else 0
 right_volume = volume_levels[right_idx + 1] if right_idx < num_bins - 1 else 0

 if left_volume >= right_volume and left_idx > 0:
 left_idx -= 1
 value_area_volume += volume_levels[left_idx]
 elif right_idx < num_bins - 1:
 right_idx += 1
 value_area_volume += volume_levels[right_idx]
 else:
 break

 value_area_high = price_levels[right_idx + 1]
 value_area_low = price_levels[left_idx]

 return VolumeProfileResult(
 price_levels=price_levels[:-1], # Remove last price level (it's just upper bound)
 volume_levels=volume_levels,
 poc=float(poc),
 value_area_high=float(value_area_high),
 value_area_low=float(value_area_low)
 )


def calculate_volume_roc(
 volumes: Union[List[float], np.ndarray],
 period: int = 12
) -> float:
 """
 Calculate Volume Rate of Change

 Args:
 volumes: Volume series
 period: Number of periods

 Returns:
 Volume ROC value (percentage)
 """
 if not volumes or len(volumes) < period + 1:
 return 0.0

 volumes_array = np.array(volumes, dtype=np.float64)

 current_volume = volumes_array[-1]
 past_volume = volumes_array[-(period + 1)]

 if past_volume == 0:
 return 0.0

 roc = ((current_volume - past_volume) / past_volume) * 100.0
 return float(roc)


@dataclass
class VolumeConfig:
 """Configuration for volume indicators"""
 vwap_period: int = 20
 mfi_period: int = 14
 cmf_period: int = 20
 volume_profile_bins: int = 20
 volume_profile_period: int = 100
 volume_roc_period: int = 12


class VolumeIndicators:
 """
 High-performance volume indicators calculator

 Calculates multiple volume-based indicators efficiently using
 vectorized operations and optional Numba acceleration.
 """

 def __init__(self, config: Optional[VolumeConfig] = None):
 self.config = config or VolumeConfig
 self.logger = logging.getLogger(__name__)

 def calculate_all(
 self,
 high: Union[List[float], np.ndarray],
 low: Union[List[float], np.ndarray],
 close: Union[List[float], np.ndarray],
 volumes: Union[List[float], np.ndarray]
 ) -> Dict[str, float]:
 """
 Calculate all volume indicators

 Args:
 high: High prices
 low: Low prices
 close: Close prices
 volumes: Volume data

 Returns:
 Dictionary with all volume indicator values
 """
 results = {}

 try:
 # On-Balance Volume
 results["obv"] = calculate_obv(close, volumes)

 # Volume Weighted Average Price
 results["vwap"] = calculate_vwap(close, volumes, self.config.vwap_period)

 # Money Flow Index
 results["mfi"] = calculate_mfi(high, low, close, volumes, self.config.mfi_period)

 # Accumulation/Distribution Line
 results["ad_line"] = calculate_ad_line(high, low, close, volumes)

 # Chaikin Money Flow
 results["cmf"] = calculate_chaikin_money_flow(
 high, low, close, volumes, self.config.cmf_period
 )

 # Volume Rate of Change
 results["volume_roc"] = calculate_volume_roc(volumes, self.config.volume_roc_period)

 # Volume Profile (simplified - just POC)
 volume_profile = calculate_volume_profile(
 high, low, close, volumes,
 self.config.volume_profile_bins,
 self.config.volume_profile_period
 )
 results["volume_poc"] = volume_profile.poc
 results["value_area_high"] = volume_profile.value_area_high
 results["value_area_low"] = volume_profile.value_area_low

 except Exception as e:
 self.logger.error(f"Error calculating volume indicators: {e}")
 # Return defaults on error
 results = {
 "obv": 0.0,
 "vwap": float(close[-1]) if close else 0.0,
 "mfi": 50.0,
 "ad_line": 0.0,
 "cmf": 0.0,
 "volume_roc": 0.0,
 "volume_poc": float(close[-1]) if close else 0.0,
 "value_area_high": float(close[-1]) if close else 0.0,
 "value_area_low": float(close[-1]) if close else 0.0
 }

 return results


# Export all functions and classes
__all__ = [
 # Core calculation functions
 "calculate_obv",
 "calculate_vwap",
 "calculate_mfi",
 "calculate_ad_line",
 "calculate_chaikin_money_flow",
 "calculate_volume_profile",
 "calculate_volume_roc",

 # Result types
 "VolumeProfileResult",

 # Configuration and main class
 "VolumeConfig",
 "VolumeIndicators"
]