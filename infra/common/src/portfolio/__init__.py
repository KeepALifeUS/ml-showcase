"""
Portfolio Feature Engineering Module
Trading State Representation

This module provides features for representing current portfolio state
in the 768-dimensional state vector (50 dims).

Key Features:
- Position tracking (20 dims): Current holdings, exposure, allocation
- Performance metrics (30 dims): PnL, returns, Sharpe ratio, drawdown
- Risk metrics: Portfolio heat, correlation, concentration
- <1ms latency for feature extraction

Architecture:
- state.py: Current positions, cash, exposure (20 dims)
- performance.py: Historical PnL, returns, metrics (30 dims)
- risk.py: Circuit breaker integration (included in performance.py)

Usage:
 from ml_common.portfolio import extract_position_features, extract_performance_features

 # Current portfolio state
 portfolio = {
 'positions': {'BTCUSDT': 0.5, 'ETHUSDT': 2.0, 'BNBUSDT': 10.0, 'SOLUSDT': 50.0},
 'cash': 5000.0,
 'total_value': 100000.0
 }

 # Extract features
 position_features = extract_position_features(portfolio, current_prices)
 # Returns 20 dims: [btc_qty, eth_qty, bnb_qty, sol_qty, btc_value, eth_value, bnb_value, sol_value,
 # btc_weight, eth_weight, bnb_weight, sol_weight, total_exposure, cash_ratio,
 # long_exposure, short_exposure, net_exposure, gross_exposure, leverage, concentration]

 # Performance features (requires history)
 performance_features = extract_performance_features(portfolio_history, window_hours=168)
 # Returns 30 dims: [total_pnl, unrealized_pnl, realized_pnl, pnl_1h, pnl_24h, pnl_7d,
 # return_1h, return_24h, return_7d, sharpe_ratio, sortino_ratio,
 # max_drawdown, current_drawdown, win_rate, profit_factor, ...]
"""

from .state import (
 extract_position_features,
 calculate_position_weights,
 calculate_exposure_metrics,
 calculate_concentration_metrics,
)

from .performance import (
 extract_performance_features,
 calculate_pnl_metrics,
 calculate_return_metrics,
 calculate_risk_adjusted_metrics,
 calculate_drawdown_metrics,
)

__all__ = [
 # Position state (4 functions)
 "extract_position_features",
 "calculate_position_weights",
 "calculate_exposure_metrics",
 "calculate_concentration_metrics",

 # Performance (4 functions)
 "extract_performance_features",
 "calculate_pnl_metrics",
 "calculate_return_metrics",
 "calculate_risk_adjusted_metrics",
 "calculate_drawdown_metrics",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
__note__ = "Portfolio state representation for autonomous trading"
