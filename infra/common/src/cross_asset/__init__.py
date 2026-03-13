"""
Cross-Asset Feature Engineering Module
Multi-Symbol Analysis

This module provides lightweight, Numba-optimized functions for extracting
cross-asset features for the 768-dimensional state vector.

Key Features:
- Cross-symbol correlation analysis (10 dims)
- Inter-asset spread analysis (6 dims)
- Beta calculations (4 dims)
- <5ms latency for real-time trading
- Zero duplication with ml-graph-networks (correlation_graph.py)

Architecture:
- correlation.py: Pearson, Spearman, rolling correlation matrix (4x4)
- spreads.py: Normalized price differences, spread volatility, convergence
- beta.py: Market beta, sector beta, relative strength, momentum divergence

Important:
- Designed for 4 symbols: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT
- Look-ahead bias protection (uses only historical data)
- Numba-optimized for performance critical paths

Usage:
 from ml_common.cross_asset import calculate_correlation_matrix, calculate_normalized_spreads

 # Extract from synchronized price data
 prices = {
 'BTCUSDT': [50000, 50100, 50200, ...], # 168h history
 'ETHUSDT': [3000, 3010, 3020, ...],
 'BNBUSDT': [400, 401, 402, ...],
 'SOLUSDT': [100, 101, 102, ...]
 }

 # Correlation matrix (10 dims = 6 unique pairs + 4 rolling)
 corr_features = extract_correlation_features(prices, window=24)
 # Returns: [BTC-ETH, BTC-BNB, BTC-SOL, ETH-BNB, ETH-SOL, BNB-SOL,
 # rolling_avg, rolling_std, rolling_min, rolling_max]

 # Spreads (6 dims)
 spread_features = extract_spread_features(prices)
 # Returns: [spread_btc_eth, spread_btc_bnb, spread_btc_sol,
 # spread_volatility, spread_convergence, spread_momentum]

 # Beta (4 dims)
 beta_features = extract_beta_features(prices, market_symbol='BTCUSDT')
 # Returns: [beta_eth, beta_bnb, beta_sol, avg_beta]
"""

from .correlation import (
 calculate_correlation_matrix,
 calculate_rolling_correlation,
 calculate_pearson_correlation,
 calculate_spearman_correlation,
 extract_correlation_features,
)

from .spreads import (
 calculate_normalized_spread,
 calculate_spread_volatility,
 calculate_spread_convergence,
 calculate_spread_momentum,
 calculate_spread_z_score,
 extract_spread_features,
)

from .beta import (
 calculate_beta,
 calculate_market_beta,
 calculate_relative_strength,
 calculate_momentum_divergence,
 extract_beta_features,
)

__all__ = [
 # Correlation (5 functions)
 "calculate_correlation_matrix",
 "calculate_rolling_correlation",
 "calculate_pearson_correlation",
 "calculate_spearman_correlation",
 "extract_correlation_features",

 # Spreads (5 functions)
 "calculate_normalized_spread",
 "calculate_spread_volatility",
 "calculate_spread_convergence",
 "calculate_spread_momentum",
 "calculate_spread_z_score",
 "extract_spread_features",

 # Beta (4 functions)
 "calculate_beta",
 "calculate_market_beta",
 "calculate_relative_strength",
 "calculate_momentum_divergence",
 "extract_beta_features",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
__note__ = "Lightweight feature extraction, NOT neural network module"
