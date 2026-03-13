"""
Cross-Asset Spread Features
Inter-Asset Spread Analysis

Calculates 6 spread dimensions for pairs trading and convergence:
1-3. Normalized spreads (BTC-ETH, BTC-BNB, BTC-SOL)
4. Spread volatility (std of normalized spreads)
5. Spread convergence (mean reversion tendency)
6. Spread momentum (trend strength)

Performance: <1ms for 168h window

Use Case: Identifies arbitrage opportunities and relative value
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from numba import jit

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


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_z_score(values: np.ndarray, window: int) -> float:
 """
 Fast z-score calculation for spread

 Formula: z = (current - mean) / std
 """
 if len(values) < window:
 return 0.0

 recent = values[-window:]
 mean = np.mean(recent)
 std = np.std(recent)

 if std == 0:
 return 0.0

 current = values[-1]
 z_score = (current - mean) / std

 return z_score


def calculate_normalized_spread(
 prices_x: Union[List[float], np.ndarray],
 prices_y: Union[List[float], np.ndarray],
 method: str = 'log'
) -> float:
 """
 Calculate normalized spread between two assets

 Formula (log): spread = log(price_x / price_y)
 Formula (pct): spread = (price_x - price_y) / price_y × 100

 Args:
 prices_x: First asset prices
 prices_y: Second asset prices
 method: 'log' (default) or 'pct'

 Returns:
 Normalized spread (dimensionless)

 Performance: ~0.02ms
 """
 x_array = np.array(prices_x, dtype=np.float64)
 y_array = np.array(prices_y, dtype=np.float64)

 if len(x_array) == 0 or len(y_array) == 0:
 return 0.0

 # Take last values
 price_x = x_array[-1]
 price_y = y_array[-1]

 if price_y == 0:
 return 0.0

 if method == 'log':
 # Log spread (more stable for large price differences)
 spread = float(np.log(price_x / price_y))
 elif method == 'pct':
 # Percentage spread
 spread = float((price_x - price_y) / price_y * 100.0)
 else:
 raise ValueError(f"Unknown method: {method}")

 return spread


def calculate_spread_volatility(
 prices_x: Union[List[float], np.ndarray],
 prices_y: Union[List[float], np.ndarray],
 window: int = 24,
 method: str = 'log'
) -> float:
 """
 Calculate spread volatility (standard deviation of spreads)

 Args:
 prices_x: First asset prices
 prices_y: Second asset prices
 window: Rolling window (default: 24 hours)
 method: 'log' or 'pct'

 Returns:
 Spread volatility (std of spreads)

 Performance: ~0.1ms for 168 data points
 """
 x_array = np.array(prices_x, dtype=np.float64)
 y_array = np.array(prices_y, dtype=np.float64)

 min_len = min(len(x_array), len(y_array))

 if min_len < window:
 return 0.0

 # Calculate spreads over window
 spreads = []

 for i in range(max(0, min_len - window), min_len):
 if y_array[i] == 0:
 continue

 if method == 'log':
 spread = np.log(x_array[i] / y_array[i])
 else: # pct
 spread = (x_array[i] - y_array[i]) / y_array[i] * 100.0

 spreads.append(spread)

 if len(spreads) < 2:
 return 0.0

 spread_array = np.array(spreads, dtype=np.float64)
 volatility = float(np.std(spread_array))

 return volatility


def calculate_spread_convergence(
 prices_x: Union[List[float], np.ndarray],
 prices_y: Union[List[float], np.ndarray],
 window: int = 24
) -> float:
 """
 Calculate spread convergence (mean reversion tendency)

 Formula: -1 × autocorrelation(spread, lag=1)
 Positive = mean reverting, Negative = trending

 Args:
 prices_x: First asset prices
 prices_y: Second asset prices
 window: Rolling window (default: 24 hours)

 Returns:
 Convergence score (-1 to +1)
 +1 = strong mean reversion
 -1 = strong trending
 0 = no pattern

 Performance: ~0.15ms
 """
 x_array = np.array(prices_x, dtype=np.float64)
 y_array = np.array(prices_y, dtype=np.float64)

 min_len = min(len(x_array), len(y_array))

 if min_len < window + 1:
 return 0.0

 # Calculate log spreads
 spreads = []
 for i in range(max(0, min_len - window - 1), min_len):
 if y_array[i] == 0:
 continue
 spread = np.log(x_array[i] / y_array[i])
 spreads.append(spread)

 if len(spreads) < 2:
 return 0.0

 spread_array = np.array(spreads, dtype=np.float64)

 # Calculate autocorrelation at lag=1
 if len(spread_array) < 2:
 return 0.0

 # Lag-1 autocorrelation
 spread_lag0 = spread_array[1:]
 spread_lag1 = spread_array[:-1]

 # Normalize
 spread_lag0_norm = spread_lag0 - np.mean(spread_lag0)
 spread_lag1_norm = spread_lag1 - np.mean(spread_lag1)

 numerator = np.sum(spread_lag0_norm * spread_lag1_norm)
 denominator = np.sqrt(np.sum(spread_lag0_norm ** 2) * np.sum(spread_lag1_norm ** 2))

 if denominator == 0:
 return 0.0

 autocorr = numerator / denominator

 # Convergence = -autocorrelation (negative autocorr = mean reversion)
 convergence = -float(autocorr)

 return convergence


def calculate_spread_momentum(
 prices_x: Union[List[float], np.ndarray],
 prices_y: Union[List[float], np.ndarray],
 window: int = 24
) -> float:
 """
 Calculate spread momentum (trend strength)

 Formula: (current_spread - avg_spread) / std_spread

 Args:
 prices_x: First asset prices
 prices_y: Second asset prices
 window: Rolling window (default: 24 hours)

 Returns:
 Momentum z-score (typically -3 to +3)
 Positive = spread widening (x outperforming y)
 Negative = spread narrowing (y outperforming x)

 Performance: ~0.1ms
 """
 x_array = np.array(prices_x, dtype=np.float64)
 y_array = np.array(prices_y, dtype=np.float64)

 min_len = min(len(x_array), len(y_array))

 if min_len < window:
 return 0.0

 # Calculate log spreads
 spreads = []
 for i in range(max(0, min_len - window), min_len):
 if y_array[i] == 0:
 continue
 spread = np.log(x_array[i] / y_array[i])
 spreads.append(spread)

 if len(spreads) < 2:
 return 0.0

 spread_array = np.array(spreads, dtype=np.float64)

 # Z-score of current spread
 momentum = _fast_z_score(spread_array, len(spread_array))

 return float(momentum)


def calculate_spread_z_score(
 prices_x: Union[List[float], np.ndarray],
 prices_y: Union[List[float], np.ndarray],
 window: int = 24
) -> float:
 """
 Calculate spread z-score (for pairs trading signals)

 Formula: z = (current_spread - mean_spread) / std_spread

 Args:
 prices_x: First asset prices
 prices_y: Second asset prices
 window: Lookback window (default: 24 hours)

 Returns:
 Z-score (typically -3 to +3)
 z > +2: overbought (consider shorting spread)
 z < -2: oversold (consider buying spread)

 Performance: ~0.1ms with Numba
 """
 return calculate_spread_momentum(prices_x, prices_y, window)


# High-level API for 6-dimensional feature vector
def extract_spread_features(
 prices: Dict[str, List[float]],
 symbols: Optional[List[str]] = None,
 window: int = 24
) -> np.ndarray:
 """
 Extract all 6 spread features as a single vector

 Args:
 prices: Dictionary {symbol: [prices]} - MUST be time-aligned!
 symbols: List of 4 symbols (default: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
 window: Rolling window (default: 24 hours)

 Returns:
 6-dimensional feature vector:
 [0] spread_btc_eth (normalized log spread)
 [1] spread_btc_bnb (normalized log spread)
 [2] spread_btc_sol (normalized log spread)
 [3] spread_volatility (average std of 3 spreads)
 [4] spread_convergence (average mean reversion of 3 spreads)
 [5] spread_momentum (average momentum of 3 spreads)

 Performance: <1ms total

 Example:
 prices = {
 'BTCUSDT': [50000, 50100, 50200, ...],
 'ETHUSDT': [3000, 3010, 3020, ...],
 'BNBUSDT': [400, 401, 402, ...],
 'SOLUSDT': [100, 101, 102, ...]
 }
 features = extract_spread_features(prices, window=24)
 # Returns: [2.81, 4.83, 6.21, 0.05, 0.23, 0.15]
 """
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 if len(symbols) != 4:
 raise ValueError("Exactly 4 symbols required (BTC, ETH, BNB, SOL)")

 features = np.zeros(6, dtype=np.float32)

 # Base symbol (BTC)
 base_symbol = symbols[0]
 other_symbols = symbols[1:]

 if base_symbol not in prices:
 return features

 base_prices = prices[base_symbol]

 # Features 0-2: Normalized spreads
 for i, symbol in enumerate(other_symbols):
 if symbol in prices:
 spread = calculate_normalized_spread(
 base_prices,
 prices[symbol],
 method='log'
 )
 features[i] = spread

 # Feature 3: Average spread volatility
 volatilities = []
 for symbol in other_symbols:
 if symbol in prices:
 vol = calculate_spread_volatility(
 base_prices,
 prices[symbol],
 window=window
 )
 volatilities.append(vol)

 if volatilities:
 features[3] = float(np.mean(volatilities))

 # Feature 4: Average spread convergence
 convergences = []
 for symbol in other_symbols:
 if symbol in prices:
 conv = calculate_spread_convergence(
 base_prices,
 prices[symbol],
 window=window
 )
 convergences.append(conv)

 if convergences:
 features[4] = float(np.mean(convergences))

 # Feature 5: Average spread momentum
 momentums = []
 for symbol in other_symbols:
 if symbol in prices:
 mom = calculate_spread_momentum(
 base_prices,
 prices[symbol],
 window=window
 )
 momentums.append(mom)

 if momentums:
 features[5] = float(np.mean(momentums))

 return features


__all__ = [
 "calculate_normalized_spread",
 "calculate_spread_volatility",
 "calculate_spread_convergence",
 "calculate_spread_momentum",
 "calculate_spread_z_score",
 "extract_spread_features",
]
