"""
Order Book Feature Engineering Module
Market Microstructure Analysis

This module provides lightweight, Numba-optimized functions for extracting
features from order book data for the 768-dimensional state vector.

Key Features:
- Bid-ask imbalance calculation (5 dims)
- Order book depth analysis (9 dims)
- Spread dynamics (6 dims)
- <10ms latency for real-time trading
- Zero duplication with TypeScript OrderBook entity

Architecture:
- imbalance.py: Volume imbalance, order flow toxicity, cumulative delta
- depth.py: Multi-level depth, depth slope, liquidity concentration
- spread.py: Absolute/relative spread, effective spread, volatility

Usage:
 from ml_common.orderbook import calculate_bid_ask_imbalance, calculate_depth_metrics

 # Extract from CCXT orderbook
 orderbook = exchange.fetch_order_book('BTC/USDT', limit=20)

 imbalance_features = calculate_bid_ask_imbalance(
 bids=orderbook['bids'],
 asks=orderbook['asks'],
 levels=5
 )
 # Returns: [imbalance_ratio, vw_imbalance, cumulative_delta, toxicity, noise]
"""

from .imbalance import (
 calculate_bid_ask_imbalance,
 calculate_volume_weighted_imbalance,
 calculate_cumulative_delta,
 calculate_order_flow_toxicity,
 calculate_microstructure_noise,
)

from .depth import (
 calculate_depth_metrics,
 calculate_bid_depth,
 calculate_ask_depth,
 calculate_depth_imbalance,
 calculate_depth_slope,
)

from .spread import (
 calculate_spread_metrics,
 calculate_absolute_spread,
 calculate_relative_spread,
 calculate_effective_spread,
 calculate_quoted_spread,
 calculate_realized_spread,
 calculate_spread_volatility,
)

from .orderbook_features import (
 OrderbookFeatureCalculator,
 OrderbookSnapshot,
 OrderbookWall,
)

from .orderbook_query import OrderbookQuery

__all__ = [
 # Imbalance (5 functions)
 "calculate_bid_ask_imbalance",
 "calculate_volume_weighted_imbalance",
 "calculate_cumulative_delta",
 "calculate_order_flow_toxicity",
 "calculate_microstructure_noise",

 # Depth (5 functions)
 "calculate_depth_metrics",
 "calculate_bid_depth",
 "calculate_ask_depth",
 "calculate_depth_imbalance",
 "calculate_depth_slope",

 # Spread (6 functions)
 "calculate_spread_metrics",
 "calculate_absolute_spread",
 "calculate_relative_spread",
 "calculate_effective_spread",
 "calculate_quoted_spread",
 "calculate_realized_spread",
 "calculate_spread_volatility",

 # OrderbookFeatureCalculator (3 classes, 20 features)
 "OrderbookFeatureCalculator",
 "OrderbookSnapshot",
 "OrderbookWall",

 # OrderbookQuery (Day 3.2: PostgreSQL integration)
 "OrderbookQuery",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
