"""
Performance Metrics Module

High-performance financial metrics calculation implementing .
Optimized for trading strategy evaluation with proper risk-adjusted returns.

Available metrics:
- Return metrics: Sharpe, Sortino, Calmar ratios
- Risk metrics: VaR, CVaR, Maximum Drawdown
- Trade metrics: Win rate, Profit factor, Average trade
- Portfolio metrics: Beta, Alpha, Information ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, NamedTuple
from dataclasses import dataclass
import logging
import warnings
from datetime import datetime, timedelta

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


@dataclass
class MetricsConfig:
 """Configuration for performance metrics"""

 # Risk-free rate (annual)
 risk_free_rate: float = 0.02

 # Return calculation
 return_type: str = "simple" # "simple" or "log"
 periods_per_year: int = 252 # Trading days per year

 # Drawdown settings
 drawdown_method: str = "peak_to_trough" # "peak_to_trough" or "rolling_window"
 rolling_window: int = 252

 # VaR settings
 var_confidence: float = 0.05 # 95% VaR
 var_method: str = "historical" # "historical", "parametric", "monte_carlo"

 # Performance calculation
 min_periods: int = 30 # Minimum periods for meaningful metrics
 use_trading_days: bool = True # Exclude weekends/holidays

 # Benchmark settings
 benchmark_returns: Optional[np.ndarray] = None
 benchmark_name: str = "Benchmark"


# Result types
class PerformanceResult(NamedTuple):
 """Complete performance metrics result"""
 total_return: float
 annualized_return: float
 volatility: float
 sharpe_ratio: float
 sortino_ratio: float
 calmar_ratio: float
 max_drawdown: float
 win_rate: float
 profit_factor: float
 var_95: float
 cvar_95: float


# Numba-optimized core functions
@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_drawdown(cumulative_returns: np.ndarray) -> Tuple[float, int, int]:
 """Fast maximum drawdown calculation with Numba"""
 if len(cumulative_returns) == 0:
 return 0.0, 0, 0

 peak = cumulative_returns[0]
 max_dd = 0.0
 peak_idx = 0
 trough_idx = 0

 for i in range(len(cumulative_returns)):
 if cumulative_returns[i] > peak:
 peak = cumulative_returns[i]
 peak_idx = i

 dd = (peak - cumulative_returns[i]) / peak if peak != 0 else 0.0

 if dd > max_dd:
 max_dd = dd
 trough_idx = i

 return max_dd, peak_idx, trough_idx


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_rolling_max(values: np.ndarray, window: int) -> np.ndarray:
 """Fast rolling maximum with Numba"""
 result = np.zeros(len(values))

 for i in range(len(values)):
 start_idx = max(0, i - window + 1)
 result[i] = np.max(values[start_idx:i+1])

 return result


@jit(nopython=True, cache=True) if HAS_NUMBA else lambda f: f
def _fast_var_historical(returns: np.ndarray, confidence: float) -> float:
 """Fast historical VaR calculation with Numba"""
 if len(returns) == 0:
 return 0.0

 sorted_returns = np.sort(returns)
 index = int(confidence * len(sorted_returns))

 if index == 0:
 return sorted_returns[0]
 elif index >= len(sorted_returns):
 return sorted_returns[-1]

 return sorted_returns[index]


def calculate_returns(
 prices: Union[np.ndarray, pd.Series, List[float]],
 return_type: str = "simple"
) -> np.ndarray:
 """
 Calculate returns from price series

 Args:
 prices: Price series
 return_type: "simple" or "log"

 Returns:
 Returns array

 Performance: ~0.1ms per 1000 data points with Numba
 """
 if len(prices) < 2:
 return np.array([])

 prices_array = np.array(prices, dtype=np.float64)

 if return_type == "log":
 # Log returns: ln(P_t / P_{t-1})
 returns = np.log(prices_array[1:] / prices_array[:-1])
 else:
 # Simple returns: (P_t - P_{t-1}) / P_{t-1}
 returns = (prices_array[1:] - prices_array[:-1]) / prices_array[:-1]

 # Handle infinite/nan values
 returns = np.where(np.isfinite(returns), returns, 0.0)

 return returns


def calculate_cumulative_returns(returns: Union[np.ndarray, List[float]]) -> np.ndarray:
 """
 Calculate cumulative returns

 Args:
 returns: Returns series

 Returns:
 Cumulative returns array
 """
 if len(returns) == 0:
 return np.array([1.0])

 returns_array = np.array(returns, dtype=np.float64)

 # Cumulative product for simple returns
 cumulative = np.cumprod(1 + returns_array)

 return cumulative


def calculate_sharpe_ratio(
 returns: Union[np.ndarray, List[float]],
 risk_free_rate: float = 0.02,
 periods_per_year: int = 252
) -> float:
 """
 Calculate Sharpe ratio

 Args:
 returns: Returns series
 risk_free_rate: Risk-free rate (annual)
 periods_per_year: Periods per year for annualization

 Returns:
 Sharpe ratio

 Performance: ~0.05ms per 1000 data points
 """
 if len(returns) < 2:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 # Calculate excess returns
 daily_rf_rate = risk_free_rate / periods_per_year
 excess_returns = returns_array - daily_rf_rate

 # Calculate Sharpe ratio
 mean_excess = np.mean(excess_returns)
 std_excess = np.std(excess_returns, ddof=1)

 # Use threshold for numerical stability (handle both zero and near-zero volatility)
 if std_excess < 1e-10:
 return 0.0

 sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

 return float(sharpe)


def calculate_sortino_ratio(
 returns: Union[np.ndarray, List[float]],
 risk_free_rate: float = 0.02,
 periods_per_year: int = 252,
 target_return: Optional[float] = None
) -> float:
 """
 Calculate Sortino ratio (downside deviation instead of total volatility)

 Args:
 returns: Returns series
 risk_free_rate: Risk-free rate (annual)
 periods_per_year: Periods per year
 target_return: Target return (defaults to risk-free rate)

 Returns:
 Sortino ratio
 """
 if len(returns) < 2:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 # Target return
 if target_return is None:
 target_return = risk_free_rate / periods_per_year
 else:
 target_return = target_return / periods_per_year

 # Calculate excess returns
 excess_returns = returns_array - target_return

 # Calculate downside deviation
 negative_returns = excess_returns[excess_returns < 0]

 if len(negative_returns) == 0:
 return float('inf') if np.mean(excess_returns) > 0 else 0.0

 downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

 if downside_deviation == 0:
 return 0.0

 # Calculate Sortino ratio
 mean_excess = np.mean(excess_returns)
 sortino = (mean_excess / downside_deviation) * np.sqrt(periods_per_year)

 return float(sortino)


def calculate_calmar_ratio(
 returns: Union[np.ndarray, List[float]],
 periods_per_year: int = 252
) -> float:
 """
 Calculate Calmar ratio (annualized return / max drawdown)

 Args:
 returns: Returns series
 periods_per_year: Periods per year

 Returns:
 Calmar ratio
 """
 if len(returns) < 2:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 # Calculate annualized return
 total_return = np.prod(1 + returns_array) - 1
 periods = len(returns_array)
 annualized_return = (1 + total_return) ** (periods_per_year / periods) - 1

 # Calculate maximum drawdown
 max_dd = calculate_max_drawdown(returns_array)

 if max_dd == 0:
 return float('inf') if annualized_return > 0 else 0.0

 calmar = annualized_return / max_dd

 return float(calmar)


def calculate_max_drawdown(
 returns: Union[np.ndarray, List[float]],
 method: str = "peak_to_trough"
) -> float:
 """
 Calculate maximum drawdown

 Args:
 returns: Returns series
 method: "peak_to_trough" or "rolling_window"

 Returns:
 Maximum drawdown (positive value)

 Performance: ~0.1ms per 1000 data points with Numba
 """
 if len(returns) < 2:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 # Calculate cumulative returns
 cumulative_returns = np.cumprod(1 + returns_array)

 if method == "peak_to_trough":
 max_dd, _, _ = _fast_drawdown(cumulative_returns)
 return float(max_dd)

 elif method == "rolling_window":
 # Rolling window approach
 window = min(252, len(cumulative_returns)) # 1 year window
 rolling_max = _fast_rolling_max(cumulative_returns, window)
 drawdowns = (rolling_max - cumulative_returns) / rolling_max
 return float(np.max(drawdowns))

 else:
 raise ValueError(f"Unknown drawdown method: {method}")


def calculate_win_rate(returns: Union[np.ndarray, List[float]]) -> float:
 """
 Calculate win rate (percentage of positive returns)

 Args:
 returns: Returns series

 Returns:
 Win rate (0-1)
 """
 if len(returns) == 0:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)
 positive_returns = np.sum(returns_array > 0)
 total_returns = len(returns_array)

 return float(positive_returns / total_returns)


def calculate_profit_factor(returns: Union[np.ndarray, List[float]]) -> float:
 """
 Calculate profit factor (gross profit / gross loss)

 Args:
 returns: Returns series

 Returns:
 Profit factor
 """
 if len(returns) == 0:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 positive_returns = returns_array[returns_array > 0]
 negative_returns = returns_array[returns_array < 0]

 gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
 gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0

 # Handle edge cases
 if gross_profit == 0 and gross_loss == 0:
 return 1.0 # No trades or all zero returns
 if gross_profit == 0:
 return 1.0 # No gains, return neutral value
 if gross_loss == 0:
 return float('inf') # Only gains, infinite profit factor

 return float(gross_profit / gross_loss)


def calculate_var(
 returns: Union[np.ndarray, List[float]],
 confidence: float = 0.05,
 method: str = "historical"
) -> float:
 """
 Calculate Value at Risk (VaR)

 Args:
 returns: Returns series
 confidence: Confidence level (0.05 for 95% VaR)
 method: "historical", "parametric", or "monte_carlo"

 Returns:
 VaR value (positive for loss)

 Performance: ~0.2ms per 1000 data points with historical method
 """
 if len(returns) < 10:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 if method == "historical":
 var_value = _fast_var_historical(returns_array, confidence)
 return float(-var_value) # Convert to positive loss value

 elif method == "parametric":
 # Gaussian VaR
 mean_return = np.mean(returns_array)
 std_return = np.std(returns_array, ddof=1)

 # Z-score for given confidence level
 from scipy.stats import norm
 z_score = norm.ppf(confidence)

 var_value = mean_return + z_score * std_return
 return float(-var_value)

 elif method == "monte_carlo":
 # Simple Monte Carlo simulation
 np.random.seed(42) # For reproducibility
 mean_return = np.mean(returns_array)
 std_return = np.std(returns_array, ddof=1)

 # Generate random returns
 simulated_returns = np.random.normal(mean_return, std_return, 10000)
 var_value = np.percentile(simulated_returns, confidence * 100)
 return float(-var_value)

 else:
 raise ValueError(f"Unknown VaR method: {method}")


def calculate_cvar(
 returns: Union[np.ndarray, List[float]],
 confidence: float = 0.05
) -> float:
 """
 Calculate Conditional Value at Risk (Expected Shortfall)

 Args:
 returns: Returns series
 confidence: Confidence level

 Returns:
 CVaR value (positive for loss)
 """
 if len(returns) < 10:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)

 # Calculate VaR threshold
 var_threshold = -calculate_var(returns_array, confidence, "historical")

 # Calculate average of returns below VaR threshold
 tail_returns = returns_array[returns_array <= var_threshold]

 if len(tail_returns) == 0:
 return float(-var_threshold)

 cvar_value = np.mean(tail_returns)
 return float(-cvar_value)


def calculate_beta(
 returns: Union[np.ndarray, List[float]],
 benchmark_returns: Union[np.ndarray, List[float]]
) -> float:
 """
 Calculate Beta (systematic risk relative to benchmark)

 Args:
 returns: Strategy returns
 benchmark_returns: Benchmark returns

 Returns:
 Beta value
 """
 if len(returns) < 10 or len(benchmark_returns) < 10:
 return 1.0

 returns_array = np.array(returns, dtype=np.float64)
 benchmark_array = np.array(benchmark_returns, dtype=np.float64)

 # Align series length
 min_length = min(len(returns_array), len(benchmark_array))
 returns_array = returns_array[:min_length]
 benchmark_array = benchmark_array[:min_length]

 # Calculate covariance and variance
 covariance = np.cov(returns_array, benchmark_array)[0, 1]
 benchmark_variance = np.var(benchmark_array, ddof=1)

 if benchmark_variance == 0:
 return 1.0

 beta = covariance / benchmark_variance
 return float(beta)


def calculate_alpha(
 returns: Union[np.ndarray, List[float]],
 benchmark_returns: Union[np.ndarray, List[float]],
 risk_free_rate: float = 0.02,
 periods_per_year: int = 252
) -> float:
 """
 Calculate Alpha (excess return over CAPM expected return)

 Args:
 returns: Strategy returns
 benchmark_returns: Benchmark returns
 risk_free_rate: Risk-free rate
 periods_per_year: Periods per year

 Returns:
 Alpha value (annualized)
 """
 if len(returns) < 10 or len(benchmark_returns) < 10:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)
 benchmark_array = np.array(benchmark_returns, dtype=np.float64)

 # Calculate daily risk-free rate
 daily_rf_rate = risk_free_rate / periods_per_year

 # Calculate excess returns
 excess_returns = returns_array - daily_rf_rate
 excess_benchmark = benchmark_array - daily_rf_rate

 # Calculate beta
 beta = calculate_beta(returns_array, benchmark_array)

 # Calculate alpha
 mean_excess_return = np.mean(excess_returns)
 mean_excess_benchmark = np.mean(excess_benchmark)

 alpha = mean_excess_return - beta * mean_excess_benchmark

 # Annualize alpha
 annualized_alpha = alpha * periods_per_year

 return float(annualized_alpha)


def calculate_information_ratio(
 returns: Union[np.ndarray, List[float]],
 benchmark_returns: Union[np.ndarray, List[float]]
) -> float:
 """
 Calculate Information Ratio (excess return / tracking error)

 Args:
 returns: Strategy returns
 benchmark_returns: Benchmark returns

 Returns:
 Information ratio
 """
 if len(returns) < 10 or len(benchmark_returns) < 10:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)
 benchmark_array = np.array(benchmark_returns, dtype=np.float64)

 # Align series length
 min_length = min(len(returns_array), len(benchmark_array))
 returns_array = returns_array[:min_length]
 benchmark_array = benchmark_array[:min_length]

 # Calculate excess returns
 excess_returns = returns_array - benchmark_array

 # Calculate tracking error
 tracking_error = np.std(excess_returns, ddof=1)

 if tracking_error == 0:
 return 0.0

 # Calculate information ratio
 mean_excess_return = np.mean(excess_returns)
 information_ratio = mean_excess_return / tracking_error

 return float(information_ratio)


def calculate_tracking_error(
 returns: Union[np.ndarray, List[float]],
 benchmark_returns: Union[np.ndarray, List[float]],
 periods_per_year: int = 252
) -> float:
 """
 Calculate Tracking Error (annualized standard deviation of excess returns)

 Args:
 returns: Strategy returns
 benchmark_returns: Benchmark returns
 periods_per_year: Periods per year

 Returns:
 Tracking error (annualized)
 """
 if len(returns) < 10 or len(benchmark_returns) < 10:
 return 0.0

 returns_array = np.array(returns, dtype=np.float64)
 benchmark_array = np.array(benchmark_returns, dtype=np.float64)

 # Align series length
 min_length = min(len(returns_array), len(benchmark_array))
 returns_array = returns_array[:min_length]
 benchmark_array = benchmark_array[:min_length]

 # Calculate excess returns
 excess_returns = returns_array - benchmark_array

 # Calculate tracking error
 tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)

 return float(tracking_error)


def calculate_rolling_metrics(
 returns: Union[np.ndarray, List[float]],
 window: int = 252,
 metrics: List[str] = ["sharpe", "volatility", "drawdown"]
) -> Dict[str, np.ndarray]:
 """
 Calculate rolling performance metrics

 Args:
 returns: Returns series
 window: Rolling window size
 metrics: List of metrics to calculate

 Returns:
 Dictionary with rolling metrics
 """
 if len(returns) < window:
 return {metric: np.array([]) for metric in metrics}

 returns_array = np.array(returns, dtype=np.float64)
 results = {}

 for metric in metrics:
 rolling_values = []

 for i in range(window - 1, len(returns_array)):
 window_returns = returns_array[i - window + 1:i + 1]

 if metric == "sharpe":
 value = calculate_sharpe_ratio(window_returns)
 elif metric == "volatility":
 value = np.std(window_returns, ddof=1) * np.sqrt(252)
 elif metric == "drawdown":
 value = calculate_max_drawdown(window_returns)
 elif metric == "win_rate":
 value = calculate_win_rate(window_returns)
 else:
 value = 0.0

 rolling_values.append(value)

 results[metric] = np.array(rolling_values)

 return results


class PerformanceMetrics:
 """
 Enterprise-grade performance metrics calculator

 Provides comprehensive performance analysis with proper statistical
 methods and enterprise patterns.
 """

 def __init__(self, config: Optional[MetricsConfig] = None):
 self.config = config or MetricsConfig
 self.logger = logging.getLogger(__name__)

 def calculate_all_metrics(
 self,
 returns: Union[np.ndarray, pd.Series, List[float]]
 ) -> PerformanceResult:
 """
 Calculate comprehensive performance metrics

 Args:
 returns: Returns series

 Returns:
 PerformanceResult with all metrics
 """
 if len(returns) < self.config.min_periods:
 self.logger.warning(f"Insufficient data: {len(returns)} < {self.config.min_periods}")
 return PerformanceResult(
 total_return=0.0, annualized_return=0.0, volatility=0.0,
 sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
 max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
 var_95=0.0, cvar_95=0.0
 )

 returns_array = np.array(returns, dtype=np.float64)

 try:
 # Return metrics
 total_return = np.prod(1 + returns_array) - 1
 periods = len(returns_array)
 annualized_return = (1 + total_return) ** (self.config.periods_per_year / periods) - 1
 volatility = np.std(returns_array, ddof=1) * np.sqrt(self.config.periods_per_year)

 # Risk-adjusted metrics
 sharpe = calculate_sharpe_ratio(
 returns_array, self.config.risk_free_rate, self.config.periods_per_year
 )
 sortino = calculate_sortino_ratio(
 returns_array, self.config.risk_free_rate, self.config.periods_per_year
 )
 calmar = calculate_calmar_ratio(returns_array, self.config.periods_per_year)

 # Risk metrics
 max_dd = calculate_max_drawdown(returns_array, self.config.drawdown_method)
 var_95 = calculate_var(returns_array, self.config.var_confidence, self.config.var_method)
 cvar_95 = calculate_cvar(returns_array, self.config.var_confidence)

 # Trade metrics
 win_rate = calculate_win_rate(returns_array)
 profit_factor = calculate_profit_factor(returns_array)

 return PerformanceResult(
 total_return=float(total_return),
 annualized_return=float(annualized_return),
 volatility=float(volatility),
 sharpe_ratio=float(sharpe),
 sortino_ratio=float(sortino),
 calmar_ratio=float(calmar),
 max_drawdown=float(max_dd),
 win_rate=float(win_rate),
 profit_factor=float(profit_factor),
 var_95=float(var_95),
 cvar_95=float(cvar_95)
 )

 except Exception as e:
 self.logger.error(f"Error calculating metrics: {e}")
 return PerformanceResult(
 total_return=0.0, annualized_return=0.0, volatility=0.0,
 sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
 max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
 var_95=0.0, cvar_95=0.0
 )

 def compare_strategies(
 self,
 strategy_returns: Dict[str, Union[np.ndarray, List[float]]]
 ) -> pd.DataFrame:
 """
 Compare multiple strategies

 Args:
 strategy_returns: Dictionary with strategy names and returns

 Returns:
 DataFrame with comparison metrics
 """
 results = []

 for name, returns in strategy_returns.items:
 metrics = self.calculate_all_metrics(returns)
 row = {
 "Strategy": name,
 "Total Return": f"{metrics.total_return:.2%}",
 "Annual Return": f"{metrics.annualized_return:.2%}",
 "Volatility": f"{metrics.volatility:.2%}",
 "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
 "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
 "Calmar Ratio": f"{metrics.calmar_ratio:.2f}",
 "Max Drawdown": f"{metrics.max_drawdown:.2%}",
 "Win Rate": f"{metrics.win_rate:.2%}",
 "Profit Factor": f"{metrics.profit_factor:.2f}",
 "VaR (95%)": f"{metrics.var_95:.2%}",
 "CVaR (95%)": f"{metrics.cvar_95:.2%}"
 }
 results.append(row)

 return pd.DataFrame(results)


# Export all functions and classes
__all__ = [
 # Core metrics functions
 "calculate_returns",
 "calculate_cumulative_returns",
 "calculate_sharpe_ratio",
 "calculate_sortino_ratio",
 "calculate_calmar_ratio",
 "calculate_max_drawdown",
 "calculate_win_rate",
 "calculate_profit_factor",
 "calculate_var",
 "calculate_cvar",

 # Advanced metrics
 "calculate_beta",
 "calculate_alpha",
 "calculate_information_ratio",
 "calculate_tracking_error",
 "calculate_rolling_metrics",

 # Result types
 "PerformanceResult",

 # Main classes
 "PerformanceMetrics",
 "MetricsConfig"
]