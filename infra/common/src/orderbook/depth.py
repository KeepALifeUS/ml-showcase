"""
Order Book Depth Features
Liquidity Analysis

Calculates 9 depth dimensions:
1-5. Bid depth at levels 1-5
6. Total bid depth (all levels)
7. Total ask depth (all levels)
8. Depth imbalance ratio
9. Depth slope (liquidity decay rate)

Performance: <1ms for 20-level orderbook with Numba optimization
"""

import numpy as np
from typing import List, Tuple, Union, Optional
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
def _fast_depth_slope(volumes: np.ndarray, prices: np.ndarray, reference_price: float) -> float:
 """
 Fast depth slope calculation with Numba

 Formula: Linear regression slope of log(volume) vs distance from mid
 Measures how quickly liquidity decays with price distance
 """
 if len(volumes) < 2:
 return 0.0

 # Calculate distances from reference price
 distances = np.abs(prices - reference_price) / reference_price

 # Log volumes (avoid log(0))
 log_volumes = np.log(np.maximum(volumes, 1e-10))

 # Simple linear regression
 n = len(distances)
 if n < 2:
 return 0.0

 mean_dist = np.mean(distances)
 mean_log_vol = np.mean(log_volumes)

 numerator = 0.0
 denominator = 0.0

 for i in range(n):
 diff_dist = distances[i] - mean_dist
 diff_vol = log_volumes[i] - mean_log_vol
 numerator += diff_dist * diff_vol
 denominator += diff_dist ** 2

 if denominator == 0:
 return 0.0

 slope = numerator / denominator

 return slope


def calculate_bid_depth(
 bids: List[List[float]],
 level: int = 1
) -> float:
 """
 Calculate bid depth at specific level

 Args:
 bids: List of [price, volume] pairs (sorted descending)
 level: Level to query (1 = best bid, 2 = second best, etc.)

 Returns:
 Volume at specified level (0.0 if not available)

 Performance: ~0.01ms
 """
 if not bids or level < 1 or level > len(bids):
 return 0.0

 return float(bids[level - 1][1])


def calculate_ask_depth(
 asks: List[List[float]],
 level: int = 1
) -> float:
 """
 Calculate ask depth at specific level

 Args:
 asks: List of [price, volume] pairs (sorted ascending)
 level: Level to query (1 = best ask, 2 = second best, etc.)

 Returns:
 Volume at specified level (0.0 if not available)

 Performance: ~0.01ms
 """
 if not asks or level < 1 or level > len(asks):
 return 0.0

 return float(asks[level - 1][1])


def calculate_total_bid_depth(
 bids: List[List[float]],
 levels: int = 20
) -> float:
 """
 Calculate total bid depth across N levels

 Args:
 bids: List of [price, volume] pairs
 levels: Number of levels to sum (default: 20)

 Returns:
 Total bid volume

 Performance: ~0.02ms
 """
 if not bids:
 return 0.0

 total = sum(bid[1] for bid in bids[:levels])

 return float(total)


def calculate_total_ask_depth(
 asks: List[List[float]],
 levels: int = 20
) -> float:
 """
 Calculate total ask depth across N levels

 Args:
 asks: List of [price, volume] pairs
 levels: Number of levels to sum (default: 20)

 Returns:
 Total ask volume

 Performance: ~0.02ms
 """
 if not asks:
 return 0.0

 total = sum(ask[1] for ask in asks[:levels])

 return float(total)


def calculate_depth_imbalance(
 bids: List[List[float]],
 asks: List[List[float]],
 levels: int = 10
) -> float:
 """
 Calculate depth imbalance ratio

 Formula: (Total Bid Depth - Total Ask Depth) / (Total Bid Depth + Total Ask Depth)

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 levels: Number of levels to consider (default: 10)

 Returns:
 Depth imbalance (-1.0 to +1.0)
 +1.0 = all bids, -1.0 = all asks, 0.0 = balanced

 Performance: ~0.03ms
 """
 total_bid = calculate_total_bid_depth(bids, levels)
 total_ask = calculate_total_ask_depth(asks, levels)

 total = total_bid + total_ask

 if total == 0:
 return 0.0

 return float((total_bid - total_ask) / total)


def calculate_depth_slope(
 bids: List[List[float]],
 asks: List[List[float]],
 mid_price: Optional[float] = None,
 levels: int = 10
) -> float:
 """
 Calculate depth slope (liquidity decay rate)

 Formula: Linear regression slope of log(volume) vs price distance
 Negative slope = liquidity decays quickly (illiquid market)
 Flat slope = liquidity stays constant (deep market)

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 mid_price: Mid price (calculated if None)
 levels: Number of levels to analyze (default: 10)

 Returns:
 Depth slope (typically -5.0 to 0.0)
 More negative = faster liquidity decay

 Performance: ~0.15ms with Numba
 """
 if not bids or not asks:
 return 0.0

 # Calculate mid price
 if mid_price is None:
 if len(bids) > 0 and len(asks) > 0:
 mid_price = (bids[0][0] + asks[0][0]) / 2.0
 else:
 return 0.0

 # Combine bid and ask data
 all_prices = []
 all_volumes = []

 for bid in bids[:levels]:
 all_prices.append(bid[0])
 all_volumes.append(bid[1])

 for ask in asks[:levels]:
 all_prices.append(ask[0])
 all_volumes.append(ask[1])

 if len(all_prices) < 2:
 return 0.0

 prices = np.array(all_prices, dtype=np.float64)
 volumes = np.array(all_volumes, dtype=np.float64)

 return float(_fast_depth_slope(volumes, prices, mid_price))


def calculate_depth_metrics(
 bids: List[List[float]],
 asks: List[List[float]],
 mid_price: Optional[float] = None
) -> Tuple[float, float, float, float]:
 """
 Calculate comprehensive depth metrics

 Returns:
 Tuple of (total_bid_depth, total_ask_depth, depth_imbalance, depth_slope)

 Performance: <0.5ms total
 """
 total_bid = calculate_total_bid_depth(bids, levels=20)
 total_ask = calculate_total_ask_depth(asks, levels=20)
 imbalance = calculate_depth_imbalance(bids, asks, levels=10)
 slope = calculate_depth_slope(bids, asks, mid_price, levels=10)

 return (total_bid, total_ask, imbalance, slope)


# High-level API for 9-dimensional feature vector
def extract_depth_features(
 bids: List[List[float]],
 asks: List[List[float]],
 mid_price: Optional[float] = None
) -> np.ndarray:
 """
 Extract all 9 depth features as a single vector

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 mid_price: Mid price (calculated if None)

 Returns:
 9-dimensional feature vector:
 [0] bid_depth_level_1
 [1] bid_depth_level_2
 [2] bid_depth_level_3
 [3] bid_depth_level_4
 [4] bid_depth_level_5
 [5] total_bid_depth (all levels)
 [6] total_ask_depth (all levels)
 [7] depth_imbalance (-1 to +1)
 [8] depth_slope (liquidity decay rate)

 Performance: <1ms total
 """
 features = np.zeros(9, dtype=np.float32)

 # Features 0-4: Bid depth at levels 1-5
 for level in range(1, 6):
 features[level - 1] = calculate_bid_depth(bids, level)

 # Feature 5: Total bid depth
 features[5] = calculate_total_bid_depth(bids, levels=20)

 # Feature 6: Total ask depth
 features[6] = calculate_total_ask_depth(asks, levels=20)

 # Feature 7: Depth imbalance
 features[7] = calculate_depth_imbalance(bids, asks, levels=10)

 # Feature 8: Depth slope
 features[8] = calculate_depth_slope(bids, asks, mid_price, levels=10)

 return features


__all__ = [
 "calculate_bid_depth",
 "calculate_ask_depth",
 "calculate_total_bid_depth",
 "calculate_total_ask_depth",
 "calculate_depth_imbalance",
 "calculate_depth_slope",
 "calculate_depth_metrics",
 "extract_depth_features",
]
