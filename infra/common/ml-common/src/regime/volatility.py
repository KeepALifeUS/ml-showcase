"""
Volatility Regime Classification
Adaptive Risk Management

Calculates 4 volatility regime dimensions:
1. Regime ID (0=Low, 1=Medium, 2=High, 3=Extreme)
2. Volatility percentile (0-100, current vs historical)
3. Regime duration (hours in current regime)
4. Regime stability (how stable is current regime)

Performance: <1ms for 168h window with Numba optimization

Use Case: Circuit breakers, position sizing, risk limits
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from numba import jit
from datetime import datetime

# Optional Numba import
try:
 from numba import jit
 HAS_NUMBA = True
except ImportError:
 HAS_NUMBA = False
 def jit(*args, **kwargs):
 def decorator(func):
 return func
 return decorator


# Volatility regime thresholds (based on BTC historical data)
REGIME_THRESHOLDS = {
 'low': 0.015, # < 1.5% daily volatility
 'medium': 0.030, # 1.5% - 3.0%
 'high': 0.050, # 3.0% - 5.0%
 'extreme': 0.050 # > 5.0%
}


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_realized_volatility(returns: np.ndarray, window: int) -> float:
 """
 Fast realized volatility calculation with Numba

 Formula: RV = sqrt(Σ(r²)) × sqrt(periods_per_day)
 For hourly data: sqrt(24)
 """
 if len(returns) < window:
 return 0.0

 recent_returns = returns[-window:]

 # Realized volatility (sum of squared returns)
 rv = np.sqrt(np.sum(recent_returns ** 2))

 # Annualize to daily (for hourly data)
 rv_daily = rv * np.sqrt(24 / window)

 return rv_daily


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_percentile(values: np.ndarray, value: float) -> float:
 """
 Fast percentile calculation with Numba

 Returns: percentile rank of value in values (0-100)
 """
 if len(values) == 0:
 return 50.0

 count_below = 0
 for v in values:
 if v < value:
 count_below += 1

 percentile = (count_below / len(values)) * 100.0

 return percentile


def classify_volatility_regime(
 prices: Union[List[float], np.ndarray],
 window: int = 24,
 thresholds: Optional[dict] = None
) -> int:
 """
 Classify volatility regime

 Args:
 prices: Price series
 window: Lookback window for volatility calculation (default: 24 hours)
 thresholds: Custom thresholds (optional)

 Returns:
 Regime ID:
 0 = Low volatility (< 1.5% daily)
 1 = Medium volatility (1.5% - 3.0%)
 2 = High volatility (3.0% - 5.0%)
 3 = Extreme volatility (> 5.0%)

 Performance: ~0.1ms with Numba

 Use Case:
 Circuit breaker triggers:
 - Regime 0-1: Normal trading
 - Regime 2: Reduce position sizes by 50%
 - Regime 3: Halt new positions, reduce to minimum
 """
 # Convert to numpy array first for consistent handling
 prices_array = np.array(prices, dtype=np.float64)

 if len(prices_array) < window + 1:
 return 1 # Default to medium

 # Calculate log returns
 returns = np.diff(np.log(prices_array))

 if len(returns) < window:
 return 1

 # Calculate realized volatility
 volatility = _fast_realized_volatility(returns, window)

 # Use custom or default thresholds
 thresh = thresholds or REGIME_THRESHOLDS

 # Classify regime
 if volatility < thresh['low']:
 return 0 # Low
 elif volatility < thresh['medium']:
 return 1 # Medium
 elif volatility < thresh['high']:
 return 2 # High
 else:
 return 3 # Extreme


def calculate_volatility_percentile(
 prices: Union[List[float], np.ndarray],
 window: int = 24,
 lookback: int = 168 # 7 days
) -> float:
 """
 Calculate volatility percentile (current vs historical)

 Formula: Percentile rank of current volatility in recent history

 Args:
 prices: Price series
 window: Window for current volatility (default: 24 hours)
 lookback: Lookback for historical comparison (default: 168 hours = 7 days)

 Returns:
 Percentile (0-100)
 90+ = unusually high volatility (potential reversal)
 10- = unusually low volatility (potential breakout)

 Performance: ~0.15ms with Numba
 """
 prices_array = np.array(prices, dtype=np.float64)
 if len(prices_array) < lookback + window:
 return 50.0 # Neutral
 returns = np.diff(np.log(prices_array))

 if len(returns) < lookback:
 return 50.0

 # Calculate current volatility
 current_vol = _fast_realized_volatility(returns[-window:], window)

 # Calculate historical volatilities (rolling)
 historical_vols = []
 for i in range(window, lookback + 1, window // 4): # Every 6 hours
 if i <= len(returns):
 vol = _fast_realized_volatility(returns[-i:-i+window], window)
 historical_vols.append(vol)

 if len(historical_vols) < 2:
 return 50.0

 hist_array = np.array(historical_vols, dtype=np.float64)

 # Calculate percentile
 percentile = _fast_percentile(hist_array, current_vol)

 return float(percentile)


def calculate_regime_duration(
 prices: Union[List[float], np.ndarray],
 window: int = 24,
 thresholds: Optional[dict] = None
) -> int:
 """
 Calculate duration of current volatility regime (in hours)

 Args:
 prices: Price series
 window: Window for volatility calculation (default: 24 hours)
 thresholds: Custom thresholds (optional)

 Returns:
 Duration in hours (0-168 for 7-day lookback)

 Performance: ~0.2ms

 Use Case:
 Regime persistence indicator:
 - Short duration (<12h) = unstable regime, be cautious
 - Long duration (>48h) = stable regime, high confidence
 """
 prices_array = np.array(prices, dtype=np.float64)
 if len(prices_array) < window * 2:
 return 0
 returns = np.diff(np.log(prices_array))

 # Current regime
 current_regime = classify_volatility_regime(prices, window, thresholds)

 # Walk backward to find regime change
 duration = 0
 max_lookback = min(len(returns) - window, 168) # Max 7 days

 for i in range(0, max_lookback, 1): # Hourly steps
 # Calculate volatility at this point
 start_idx = len(returns) - window - i
 end_idx = len(returns) - i

 if start_idx < 0:
 break

 returns_window = returns[start_idx:end_idx]

 if len(returns_window) < window:
 break

 vol = _fast_realized_volatility(returns_window, window)

 # Classify regime at this point
 thresh = thresholds or REGIME_THRESHOLDS
 if vol < thresh['low']:
 regime = 0
 elif vol < thresh['medium']:
 regime = 1
 elif vol < thresh['high']:
 regime = 2
 else:
 regime = 3

 # Check if regime changed
 if regime != current_regime:
 break

 duration += 1

 return duration


def calculate_regime_stability(
 prices: Union[List[float], np.ndarray],
 window: int = 24
) -> float:
 """
 Calculate regime stability (inverse of regime transitions)

 Formula: 1 - (regime_transitions / max_possible_transitions)

 Args:
 prices: Price series
 window: Window for volatility calculation (default: 24 hours)

 Returns:
 Stability score (0-1)
 1.0 = very stable (no regime changes)
 0.0 = very unstable (frequent regime changes)

 Performance: ~0.2ms

 Use Case:
 Risk management:
 - High stability (>0.8) = predictable, safe to trade
 - Low stability (<0.4) = chaotic, reduce exposure
 """
 prices_array = np.array(prices, dtype=np.float64)
 if len(prices_array) < window * 3:
 return 0.5 # Neutral
 returns = np.diff(np.log(prices_array))

 # Calculate regime at multiple points
 regimes = []
 max_lookback = min(len(returns) - window, 72) # Last 3 days

 for i in range(0, max_lookback, 6): # Every 6 hours
 start_idx = len(returns) - window - i
 end_idx = len(returns) - i

 if start_idx < 0:
 break

 returns_window = returns[start_idx:end_idx]

 if len(returns_window) < window:
 break

 vol = _fast_realized_volatility(returns_window, window)

 # Classify
 if vol < REGIME_THRESHOLDS['low']:
 regime = 0
 elif vol < REGIME_THRESHOLDS['medium']:
 regime = 1
 elif vol < REGIME_THRESHOLDS['high']:
 regime = 2
 else:
 regime = 3

 regimes.append(regime)

 if len(regimes) < 2:
 return 0.5

 # Count transitions
 transitions = 0
 for i in range(1, len(regimes)):
 if regimes[i] != regimes[i-1]:
 transitions += 1

 max_transitions = len(regimes) - 1

 # Stability = 1 - (transitions / max_transitions)
 stability = 1.0 - (transitions / max_transitions)

 return float(stability)


# High-level API for 4-dimensional feature vector
def extract_volatility_features(
 prices: Union[List[float], np.ndarray],
 window: int = 24,
 lookback: int = 168
) -> np.ndarray:
 """
 Extract all 4 volatility regime features as a single vector

 Args:
 prices: Price series (MUST be at least 168h for reliable classification)
 window: Volatility calculation window (default: 24 hours)
 lookback: Historical comparison window (default: 168 hours = 7 days)

 Returns:
 4-dimensional feature vector:
 [0] regime_id (0=Low, 1=Medium, 2=High, 3=Extreme)
 [1] volatility_percentile (0-100, current vs historical)
 [2] regime_duration (hours in current regime)
 [3] regime_stability (0-1, inverse of transitions)

 Performance: <1ms total

 Example:
 prices = [50000, 50100, 50200, ...] # 168 hours
 features = extract_volatility_features(prices, window=24)
 # Returns: [1.0, 45.2, 36.0, 0.82]
 # Interpretation:
 # - Regime 1 (Medium volatility, normal trading)
 # - 45th percentile (below average volatility)
 # - 36 hours in current regime (stable)
 # - 0.82 stability (very stable regime)
 """
 features = np.zeros(4, dtype=np.float32)

 # Feature 0: Regime ID
 features[0] = float(classify_volatility_regime(prices, window))

 # Feature 1: Volatility percentile
 features[1] = calculate_volatility_percentile(prices, window, lookback)

 # Feature 2: Regime duration
 features[2] = float(calculate_regime_duration(prices, window))

 # Feature 3: Regime stability
 features[3] = calculate_regime_stability(prices, window)

 return features


__all__ = [
 "classify_volatility_regime",
 "calculate_volatility_percentile",
 "calculate_regime_duration",
 "calculate_regime_stability",
 "extract_volatility_features",
 "REGIME_THRESHOLDS",
]
