"""
Evaluation Module for ML Common

High-performance evaluation metrics and backtesting utilities implementing .
Consolidates evaluation logic from multiple ML packages.

Available modules:
- metrics: Performance metrics (Sharpe, Sortino, Calmar, etc.)
- backtesting: Strategy backtesting framework
"""

from .metrics import (
 # Performance metrics
 calculate_sharpe_ratio,
 calculate_sortino_ratio,
 calculate_calmar_ratio,
 calculate_max_drawdown,
 calculate_win_rate,
 calculate_profit_factor,
 calculate_var,
 calculate_cvar,

 # Risk metrics
 calculate_beta,
 calculate_alpha,
 calculate_information_ratio,
 calculate_tracking_error,

 # Utility functions
 calculate_returns,
 calculate_cumulative_returns,
 calculate_rolling_metrics,

 # Main metrics calculator
 PerformanceMetrics,
 MetricsConfig
)

from .backtesting import (
 # Core backtesting
 backtest_strategy,
 BacktestResult,
 BacktestConfig,

 # Position management
 Position,
 Portfolio,
 Trade,
 OrderSide,
 PositionSide,

 # Strategy framework
 Strategy,
 BaseStrategy,

 # Performance analysis
 analyze_backtest_results,
 generate_performance_report,

 # Main backtesting engine
 BacktestEngine
)

__all__ = [
 # Performance metrics
 "calculate_sharpe_ratio",
 "calculate_sortino_ratio",
 "calculate_calmar_ratio",
 "calculate_max_drawdown",
 "calculate_win_rate",
 "calculate_profit_factor",
 "calculate_var",
 "calculate_cvar",

 # Risk metrics
 "calculate_beta",
 "calculate_alpha",
 "calculate_information_ratio",
 "calculate_tracking_error",

 # Utility functions
 "calculate_returns",
 "calculate_cumulative_returns",
 "calculate_rolling_metrics",

 # Metrics classes
 "PerformanceMetrics",
 "MetricsConfig",

 # Backtesting
 "backtest_strategy",
 "BacktestResult",
 "BacktestConfig",
 "Position",
 "Portfolio",
 "Trade",
 "OrderSide",
 "PositionSide",
 "Strategy",
 "BaseStrategy",
 "analyze_backtest_results",
 "generate_performance_report",
 "BacktestEngine"
]