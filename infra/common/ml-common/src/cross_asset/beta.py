"""
Cross-Asset Beta Features
Market Beta & Relative Strength

Calculates 4 beta dimensions:
1-3. Market beta for ETH, BNB, SOL (relative to BTC)
4. Average beta (sector beta)

Performance: <1ms for 168h window with Numba optimization

Use Case: Measures systematic risk and relative volatility
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
def _fast_beta(market_returns: np.ndarray, asset_returns: np.ndarray) -> float:
 """
 Fast beta calculation with Numba

 Formula: β = Cov(asset, market) / Var(market)
 """
 if len(market_returns) < 2 or len(asset_returns) < 2:
 return 1.0 # Neutral beta

 # Remove NaN
 mask = ~(np.isnan(market_returns) | np.isnan(asset_returns))
 market_clean = market_returns[mask]
 asset_clean = asset_returns[mask]

 if len(market_clean) < 2:
 return 1.0

 # Calculate covariance and variance
 mean_market = np.mean(market_clean)
 mean_asset = np.mean(asset_clean)

 cov = np.mean((market_clean - mean_market) * (asset_clean - mean_asset))
 var_market = np.var(market_clean)

 if var_market == 0:
 return 1.0

 beta = cov / var_market

 return beta


def calculate_beta(
 asset_returns: Union[List[float], np.ndarray],
 market_returns: Union[List[float], np.ndarray],
 window: Optional[int] = None
) -> float:
 """
 Calculate market beta (CAPM beta)

 Formula: β = Cov(R_asset, R_market) / Var(R_market)

 Args:
 asset_returns: Asset return series (already calculated returns, not prices)
 market_returns: Market (benchmark) return series (already calculated returns, not prices)
 window: Lookback window (None = use all data)

 Returns:
 Beta coefficient
 β > 1: more volatile than market (aggressive)
 β = 1: same volatility as market (neutral)
 β < 1: less volatile than market (defensive)
 β < 0: inverse relationship (rare in crypto)

 Performance: ~0.15ms with Numba for 168 data points
 """
 asset_array = np.array(asset_returns, dtype=np.float64)
 market_array = np.array(market_returns, dtype=np.float64)

 min_len = min(len(asset_array), len(market_array))

 if min_len < 2:
 return 1.0

 # Use window if specified
 if window is not None and min_len > window:
 asset_array = asset_array[-window:]
 market_array = market_array[-window:]

 # Input is already returns, use directly
 if len(asset_array) < 2:
 return 1.0

 return float(_fast_beta(market_array, asset_array))


def calculate_market_beta(
 prices: Dict[str, List[float]],
 asset_symbol: str,
 market_symbol: str = 'BTCUSDT',
 window: Optional[int] = None
) -> float:
 """
 Calculate market beta for an asset relative to market benchmark

 Args:
 prices: Dictionary {symbol: [prices]}
 asset_symbol: Symbol to calculate beta for
 market_symbol: Market benchmark (default: 'BTCUSDT')
 window: Lookback window (None = use all data)

 Returns:
 Beta coefficient

 Performance: ~0.15ms
 """
 if asset_symbol not in prices or market_symbol not in prices:
 return 1.0

 # Calculate log returns from prices
 asset_prices = np.array(prices[asset_symbol], dtype=np.float64)
 market_prices = np.array(prices[market_symbol], dtype=np.float64)

 if len(asset_prices) < 2 or len(market_prices) < 2:
 return 1.0

 asset_returns = np.diff(np.log(asset_prices))
 market_returns = np.diff(np.log(market_prices))

 return calculate_beta(
 asset_returns,
 market_returns,
 window=window
 )


def calculate_relative_strength(
 asset_prices: Union[List[float], np.ndarray],
 market_prices: Union[List[float], np.ndarray],
 window: int = 24
) -> float:
 """
 Calculate relative strength (price performance ratio)

 Formula: RS = (Asset Return / Market Return) over window

 Args:
 asset_prices: Asset price series
 market_prices: Market price series
 window: Lookback window (default: 24 hours)

 Returns:
 Relative strength ratio
 RS > 1: asset outperforming market
 RS = 1: equal performance
 RS < 1: asset underperforming market

 Performance: ~0.05ms
 """
 asset_array = np.array(asset_prices, dtype=np.float64)
 market_array = np.array(market_prices, dtype=np.float64)

 min_len = min(len(asset_array), len(market_array))

 if min_len < window + 1:
 return 1.0

 # Calculate returns over window
 asset_start = asset_array[-(window + 1)]
 asset_end = asset_array[-1]
 market_start = market_array[-(window + 1)]
 market_end = market_array[-1]

 if asset_start == 0 or market_start == 0:
 return 1.0

 asset_return = (asset_end - asset_start) / asset_start
 market_return = (market_end - market_start) / market_start

 if market_return == 0:
 return 1.0

 rs = (1.0 + asset_return) / (1.0 + market_return)

 return float(rs)


def calculate_momentum_divergence(
 asset_prices: Union[List[float], np.ndarray],
 market_prices: Union[List[float], np.ndarray],
 short_window: int = 12,
 long_window: int = 24
) -> float:
 """
 Calculate momentum divergence (relative momentum)

 Formula: (Asset RS_short - Asset RS_long) - (Market RS_short - Market RS_long)

 Args:
 asset_prices: Asset price series
 market_prices: Market price series
 short_window: Short momentum window (default: 12 hours)
 long_window: Long momentum window (default: 24 hours)

 Returns:
 Momentum divergence
 Positive = asset accelerating relative to market
 Negative = asset decelerating relative to market

 Performance: ~0.1ms
 """
 # Calculate short-term relative strength
 rs_short = calculate_relative_strength(asset_prices, market_prices, short_window)

 # Calculate long-term relative strength
 rs_long = calculate_relative_strength(asset_prices, market_prices, long_window)

 # Divergence = short RS - long RS
 divergence = rs_short - rs_long

 return float(divergence)


# High-level API for 4-dimensional feature vector
def extract_beta_features(
 prices: Dict[str, List[float]],
 symbols: Optional[List[str]] = None,
 market_symbol: str = 'BTCUSDT',
 window: int = 168 # 7 days
) -> np.ndarray:
 """
 Extract all 4 beta features as a single vector

 Args:
 prices: Dictionary {symbol: [prices]} - MUST be time-aligned!
 symbols: List of 4 symbols (default: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
 market_symbol: Market benchmark (default: 'BTCUSDT')
 window: Lookback window (default: 168 hours = 7 days)

 Returns:
 4-dimensional feature vector:
 [0] beta_eth (ETH beta relative to BTC)
 [1] beta_bnb (BNB beta relative to BTC)
 [2] beta_sol (SOL beta relative to BTC)
 [3] avg_beta (average sector beta)

 Performance: <1ms total

 Example:
 prices = {
 'BTCUSDT': [50000, 50100, 50200, ...],
 'ETHUSDT': [3000, 3010, 3020, ...],
 'BNBUSDT': [400, 401, 402, ...],
 'SOLUSDT': [100, 101, 102, ...]
 }
 features = extract_beta_features(prices, window=168)
 # Returns: [1.15, 0.95, 1.25, 1.12]
 # Interpretation:
 # - ETH has 15% more volatility than BTC
 # - BNB has 5% less volatility than BTC
 # - SOL has 25% more volatility than BTC
 # - Average sector beta is 1.12 (sector is 12% more volatile than BTC)
 """
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 if len(symbols) != 4:
 raise ValueError("Exactly 4 symbols required (BTC, ETH, BNB, SOL)")

 features = np.zeros(4, dtype=np.float32)

 # Base symbol (market benchmark)
 base_symbol = market_symbol
 other_symbols = [s for s in symbols if s != base_symbol]

 if base_symbol not in prices:
 # Default to neutral betas
 features[:] = 1.0
 return features

 # Features 0-2: Beta for each asset relative to BTC
 # Calculate base market returns once
 base_prices = np.array(prices[base_symbol], dtype=np.float64)
 if len(base_prices) < 2:
 features[:] = 1.0
 return features
 base_returns = np.diff(np.log(base_prices))

 betas = []
 for i, symbol in enumerate(other_symbols):
 if symbol in prices:
 asset_prices = np.array(prices[symbol], dtype=np.float64)
 if len(asset_prices) < 2:
 features[i] = 1.0
 betas.append(1.0)
 continue

 asset_returns = np.diff(np.log(asset_prices))
 beta = calculate_beta(
 asset_returns,
 base_returns,
 window=window
 )
 features[i] = beta
 betas.append(beta)
 else:
 features[i] = 1.0
 betas.append(1.0)

 # Feature 3: Average beta (sector beta)
 if betas:
 features[3] = float(np.mean(betas))
 else:
 features[3] = 1.0

 return features


__all__ = [
 "calculate_beta",
 "calculate_market_beta",
 "calculate_relative_strength",
 "calculate_momentum_divergence",
 "extract_beta_features",
]
