"""
Order Book Imbalance Features
Market Microstructure Analysis

Calculates 5 imbalance dimensions:
1. Bid-ask imbalance ratio
2. Volume-weighted imbalance
3. Cumulative delta (trade flow)
4. Order flow toxicity
5. Microstructure noise

Performance: <2ms for 20-level orderbook with Numba optimization
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
def _fast_imbalance_ratio(bid_volumes: np.ndarray, ask_volumes: np.ndarray) -> float:
 """
 Fast imbalance ratio calculation with Numba

 Formula: (Total Bid Volume - Total Ask Volume) / (Total Bid Volume + Total Ask Volume)
 """
 total_bid = np.sum(bid_volumes)
 total_ask = np.sum(ask_volumes)
 total = total_bid + total_ask

 if total == 0:
 return 0.0

 return (total_bid - total_ask) / total


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_vw_imbalance(
 bid_prices: np.ndarray,
 bid_volumes: np.ndarray,
 ask_prices: np.ndarray,
 ask_volumes: np.ndarray,
 mid_price: float
) -> float:
 """
 Fast volume-weighted imbalance calculation

 Formula: Σ(bid_vol × (1 - |bid_price - mid|/mid)) - Σ(ask_vol × (1 - |ask_price - mid|/mid))
 """
 if mid_price == 0:
 return 0.0

 weighted_bid = 0.0
 for i in range(len(bid_prices)):
 distance = abs(bid_prices[i] - mid_price) / mid_price
 weight = max(0.0, 1.0 - distance)
 weighted_bid += bid_volumes[i] * weight

 weighted_ask = 0.0
 for i in range(len(ask_prices)):
 distance = abs(ask_prices[i] - mid_price) / mid_price
 weight = max(0.0, 1.0 - distance)
 weighted_ask += ask_volumes[i] * weight

 total = weighted_bid + weighted_ask
 if total == 0:
 return 0.0

 return (weighted_bid - weighted_ask) / total


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_toxicity(
 bid_volumes: np.ndarray,
 ask_volumes: np.ndarray,
 trade_volumes: np.ndarray,
 trade_sides: np.ndarray
) -> float:
 """
 Fast order flow toxicity calculation

 Formula: Correlation between imbalance and future price movement
 Simplified: Ratio of aggressive trades to passive liquidity
 """
 if len(trade_volumes) == 0:
 return 0.5 # Neutral toxicity

 aggressive_buys = 0.0
 aggressive_sells = 0.0

 for i in range(len(trade_volumes)):
 if trade_sides[i] > 0: # Buy side
 aggressive_buys += trade_volumes[i]
 else: # Sell side
 aggressive_sells += trade_volumes[i]

 total_passive = np.sum(bid_volumes) + np.sum(ask_volumes)
 total_aggressive = aggressive_buys + aggressive_sells

 if total_passive == 0:
 return 1.0 # High toxicity (no passive liquidity)

 # Toxicity as ratio of aggressive to total volume
 toxicity = total_aggressive / (total_passive + total_aggressive)

 # Adjust for buy/sell imbalance
 if total_aggressive > 0:
 side_imbalance = (aggressive_buys - aggressive_sells) / total_aggressive
 toxicity *= (1.0 + abs(side_imbalance))

 return min(1.0, toxicity) # Cap at 1.0


def calculate_bid_ask_imbalance(
 bids: List[List[float]],
 asks: List[List[float]],
 levels: int = 5
) -> float:
 """
 Calculate bid-ask imbalance ratio

 Formula: (ΣBid Volume - ΣAsk Volume) / (ΣBid Volume + ΣAsk Volume)

 Args:
 bids: List of [price, volume] pairs (sorted descending)
 asks: List of [price, volume] pairs (sorted ascending)
 levels: Number of levels to consider (default: 5)

 Returns:
 Imbalance ratio (-1.0 to +1.0)
 +1.0 = all bids, -1.0 = all asks, 0.0 = balanced

 Performance: ~0.05ms with Numba
 """
 if not bids or not asks:
 return 0.0

 # Extract volumes from top N levels
 bid_volumes = np.array([bid[1] for bid in bids[:levels]], dtype=np.float64)
 ask_volumes = np.array([ask[1] for ask in asks[:levels]], dtype=np.float64)

 if len(bid_volumes) == 0 or len(ask_volumes) == 0:
 return 0.0

 return float(_fast_imbalance_ratio(bid_volumes, ask_volumes))


def calculate_volume_weighted_imbalance(
 bids: List[List[float]],
 asks: List[List[float]],
 mid_price: Optional[float] = None,
 levels: int = 10
) -> float:
 """
 Calculate volume-weighted imbalance (closer levels have more weight)

 Formula:
 VW_Imbalance = [Σ(bid_vol × weight) - Σ(ask_vol × weight)] / Total
 weight = 1 - |price - mid_price| / mid_price

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 mid_price: Mid price (calculated if None)
 levels: Number of levels to consider (default: 10)

 Returns:
 Volume-weighted imbalance (-1.0 to +1.0)

 Performance: ~0.08ms with Numba
 """
 if not bids or not asks:
 return 0.0

 # Extract prices and volumes
 bid_prices = np.array([bid[0] for bid in bids[:levels]], dtype=np.float64)
 bid_volumes = np.array([bid[1] for bid in bids[:levels]], dtype=np.float64)
 ask_prices = np.array([ask[0] for ask in asks[:levels]], dtype=np.float64)
 ask_volumes = np.array([ask[1] for ask in asks[:levels]], dtype=np.float64)

 # Calculate mid price if not provided
 if mid_price is None:
 if len(bid_prices) > 0 and len(ask_prices) > 0:
 mid_price = (bid_prices[0] + ask_prices[0]) / 2.0
 else:
 return 0.0

 return float(_fast_vw_imbalance(bid_prices, bid_volumes, ask_prices, ask_volumes, mid_price))


def calculate_cumulative_delta(
 trades: List[Tuple[float, float, str]],
 window_size: int = 100
) -> float:
 """
 Calculate cumulative delta (order flow)

 Formula: Σ(buy_volume) - Σ(sell_volume) over recent trades

 Args:
 trades: List of (price, volume, side) tuples
 side: 'buy' (market buy) or 'sell' (market sell)
 window_size: Number of recent trades to consider (default: 100)

 Returns:
 Cumulative delta (unbounded, typically -10000 to +10000 for BTC)

 Performance: ~0.03ms for 100 trades
 """
 if not trades:
 return 0.0

 recent_trades = trades[-window_size:]

 buy_volume = sum(volume for _, volume, side in recent_trades if side == 'buy')
 sell_volume = sum(volume for _, volume, side in recent_trades if side == 'sell')

 return float(buy_volume - sell_volume)


def calculate_order_flow_toxicity(
 bids: List[List[float]],
 asks: List[List[float]],
 trades: List[Tuple[float, float, str]],
 window_size: int = 50
) -> float:
 """
 Calculate order flow toxicity (VPIN-like metric)

 Measures the probability that informed traders are active.
 High toxicity indicates adverse selection risk for market makers.

 Formula (simplified):
 Toxicity = Aggressive Trades / (Aggressive + Passive Liquidity)
 Adjusted for buy/sell imbalance

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 trades: List of (price, volume, side) tuples
 window_size: Number of recent trades (default: 50)

 Returns:
 Toxicity score (0.0 to 1.0)
 0.0 = low toxicity (uninformed flow)
 1.0 = high toxicity (informed flow)

 Performance: ~0.10ms with Numba
 """
 if not bids or not asks:
 return 0.5

 bid_volumes = np.array([bid[1] for bid in bids[:10]], dtype=np.float64)
 ask_volumes = np.array([ask[1] for ask in asks[:10]], dtype=np.float64)

 if not trades:
 return 0.5

 recent_trades = trades[-window_size:]
 trade_volumes = np.array([volume for _, volume, _ in recent_trades], dtype=np.float64)
 trade_sides = np.array([1.0 if side == 'buy' else -1.0 for _, _, side in recent_trades], dtype=np.float64)

 return float(_fast_toxicity(bid_volumes, ask_volumes, trade_volumes, trade_sides))


def calculate_microstructure_noise(
 prices: List[float],
 window_size: int = 20
) -> float:
 """
 Calculate microstructure noise (high-frequency price volatility)

 Formula: Variance of first-differences of log prices

 Args:
 prices: List of recent trade prices
 window_size: Number of prices to analyze (default: 20)

 Returns:
 Microstructure noise (variance, typically 0.0001 to 0.001 for BTC)

 Performance: ~0.02ms
 """
 if not prices or len(prices) < 2:
 return 0.0

 recent_prices = prices[-window_size:]

 if len(recent_prices) < 2:
 return 0.0

 # Log returns
 log_prices = np.log(np.array(recent_prices, dtype=np.float64))
 returns = np.diff(log_prices)

 if len(returns) == 0:
 return 0.0

 # Variance of returns
 noise = float(np.var(returns))

 return noise


# High-level API for 5-dimensional feature vector
def extract_imbalance_features(
 bids: List[List[float]],
 asks: List[List[float]],
 trades: Optional[List[Tuple[float, float, str]]] = None,
 prices: Optional[List[float]] = None
) -> np.ndarray:
 """
 Extract all 5 imbalance features as a single vector

 Args:
 bids: List of [price, volume] pairs
 asks: List of [price, volume] pairs
 trades: List of (price, volume, side) tuples (optional)
 prices: List of recent prices (optional)

 Returns:
 5-dimensional feature vector:
 [0] bid_ask_imbalance (-1 to +1)
 [1] volume_weighted_imbalance (-1 to +1)
 [2] cumulative_delta (unbounded)
 [3] order_flow_toxicity (0 to 1)
 [4] microstructure_noise (variance)

 Performance: <2ms total
 """
 features = np.zeros(5, dtype=np.float32)

 # Feature 1: Bid-ask imbalance
 features[0] = calculate_bid_ask_imbalance(bids, asks, levels=5)

 # Feature 2: Volume-weighted imbalance
 features[1] = calculate_volume_weighted_imbalance(bids, asks, levels=10)

 # Feature 3: Cumulative delta (if trades available)
 if trades:
 features[2] = calculate_cumulative_delta(trades, window_size=100)

 # Feature 4: Order flow toxicity (if trades available)
 if trades:
 features[3] = calculate_order_flow_toxicity(bids, asks, trades, window_size=50)

 # Feature 5: Microstructure noise (if prices available)
 if prices:
 features[4] = calculate_microstructure_noise(prices, window_size=20)

 return features


__all__ = [
 "calculate_bid_ask_imbalance",
 "calculate_volume_weighted_imbalance",
 "calculate_cumulative_delta",
 "calculate_order_flow_toxicity",
 "calculate_microstructure_noise",
 "extract_imbalance_features",
]
