"""
Test Suite for Evaluation Module

Comprehensive tests for performance metrics and backtesting functionality
with accuracy verification and performance validation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple

from ml_common.evaluation import (
 # Metrics
 calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio,
 calculate_max_drawdown, calculate_win_rate, calculate_profit_factor,
 calculate_var, calculate_cvar, calculate_returns,
 PerformanceMetrics, MetricsConfig,

 # Backtesting
 backtest_strategy, BacktestResult, BacktestConfig,
 Position, Portfolio, Trade, BaseStrategy, Strategy,
 OrderSide, PositionSide, analyze_backtest_results,
 generate_performance_report
)

from . import (
 assert_array_almost_equal, assert_performance_acceptable, assert_no_warnings,
 SAMPLE_PRICES, SAMPLE_OHLCV, SAMPLE_RETURNS, PERFORMANCE_THRESHOLDS,
 generate_price_data, generate_ohlcv_data, generate_returns_data
)


class TestPerformanceMetrics:
 """Test performance metrics functions"""

 def test_calculate_returns_simple(self):
 """Test simple returns calculation"""
 prices = [100, 110, 105, 115]
 result = calculate_returns(prices, return_type="simple")

 expected = [(110-100)/100, (105-110)/110, (115-105)/105]
 assert_array_almost_equal(result, expected)

 def test_calculate_returns_log(self):
 """Test log returns calculation"""
 prices = [100, 110, 105, 115]
 result = calculate_returns(prices, return_type="log")

 expected = [np.log(110/100), np.log(105/110), np.log(115/105)]
 assert_array_almost_equal(result, expected)

 def test_calculate_returns_empty(self):
 """Test returns calculation with empty data"""
 result = calculate_returns([], return_type="simple")
 assert len(result) == 0

 def test_calculate_returns_single_price(self):
 """Test returns calculation with single price"""
 result = calculate_returns([100], return_type="simple")
 assert len(result) == 0

 def test_calculate_sharpe_ratio_basic(self):
 """Test basic Sharpe ratio calculation"""
 returns = [0.01, 0.02, -0.01, 0.03, 0.005, -0.005, 0.02, 0.01]
 result = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

 # Sharpe should be finite
 assert np.isfinite(result)

 def test_calculate_sharpe_ratio_no_volatility(self):
 """Test Sharpe ratio with zero volatility"""
 returns = [0.01] * 10 # Constant returns
 result = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

 # Should be 0 when volatility is zero
 assert result == 0.0

 def test_calculate_sharpe_ratio_insufficient_data(self):
 """Test Sharpe ratio with insufficient data"""
 returns = [0.01]
 result = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

 assert result == 0.0

 def test_calculate_sharpe_ratio_performance(self):
 """Test Sharpe ratio performance"""
 returns = generate_returns_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['evaluation']['calculate_sharpe_ratio']

 result = assert_performance_acceptable(
 calculate_sharpe_ratio, (returns,), threshold
 )
 assert np.isfinite(result)

 def test_calculate_sortino_ratio_basic(self):
 """Test basic Sortino ratio calculation"""
 returns = [0.01, 0.02, -0.01, 0.03, 0.005, -0.005, 0.02, 0.01]
 result = calculate_sortino_ratio(returns, risk_free_rate=0.02)

 # Sortino should be finite
 assert np.isfinite(result)

 def test_calculate_sortino_ratio_no_downside(self):
 """Test Sortino ratio with no negative returns"""
 returns = [0.01, 0.02, 0.03, 0.01, 0.02] # All positive
 result = calculate_sortino_ratio(returns, risk_free_rate=0.0)

 # Should be infinite when no downside deviation
 assert np.isinf(result) or result > 100 # Very high

 def test_calculate_max_drawdown_basic(self):
 """Test basic maximum drawdown calculation"""
 returns = [0.1, -0.05, -0.1, 0.2, -0.15, 0.1]
 result = calculate_max_drawdown(returns)

 # Drawdown should be positive
 assert result >= 0

 def test_calculate_max_drawdown_no_losses(self):
 """Test max drawdown with only gains"""
 returns = [0.01, 0.02, 0.03, 0.01, 0.02]
 result = calculate_max_drawdown(returns)

 # Should be very small or zero
 assert result < 0.01

 def test_calculate_max_drawdown_performance(self):
 """Test max drawdown performance"""
 returns = generate_returns_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['evaluation']['calculate_max_drawdown']

 result = assert_performance_acceptable(
 calculate_max_drawdown, (returns,), threshold
 )
 assert result >= 0

 def test_calculate_win_rate_basic(self):
 """Test win rate calculation"""
 returns = [0.01, -0.02, 0.03, -0.01, 0.02, -0.005, 0.01]
 result = calculate_win_rate(returns)

 # Count positive returns
 positive_count = sum(1 for r in returns if r > 0)
 expected = positive_count / len(returns)

 assert abs(result - expected) < 1e-10

 def test_calculate_win_rate_all_wins(self):
 """Test win rate with all positive returns"""
 returns = [0.01, 0.02, 0.03, 0.01, 0.02]
 result = calculate_win_rate(returns)

 assert result == 1.0

 def test_calculate_win_rate_all_losses(self):
 """Test win rate with all negative returns"""
 returns = [-0.01, -0.02, -0.03, -0.01, -0.02]
 result = calculate_win_rate(returns)

 assert result == 0.0

 def test_calculate_profit_factor_basic(self):
 """Test profit factor calculation"""
 returns = [0.1, -0.05, 0.08, -0.03, 0.06, -0.02]
 result = calculate_profit_factor(returns)

 # Calculate manually
 gross_profit = sum(r for r in returns if r > 0)
 gross_loss = abs(sum(r for r in returns if r < 0))
 expected = gross_profit / gross_loss if gross_loss > 0 else float('inf')

 if np.isfinite(expected):
 assert abs(result - expected) < 1e-10
 else:
 assert np.isinf(result)

 def test_calculate_profit_factor_no_losses(self):
 """Test profit factor with no losses"""
 returns = [0.01, 0.02, 0.03, 0.01, 0.02]
 result = calculate_profit_factor(returns)

 assert np.isinf(result)

 def test_calculate_profit_factor_no_gains(self):
 """Test profit factor with no gains"""
 returns = [-0.01, -0.02, -0.03, -0.01, -0.02]
 result = calculate_profit_factor(returns)

 assert result == 1.0 # As defined in implementation

 def test_calculate_var_historical(self):
 """Test VaR calculation with historical method"""
 returns = generate_returns_data(1000)
 result = calculate_var(returns, confidence=0.05, method="historical")

 # VaR should be positive (loss value)
 assert result >= 0

 def test_calculate_var_parametric(self):
 """Test VaR calculation with parametric method"""
 returns = generate_returns_data(1000)
 result = calculate_var(returns, confidence=0.05, method="parametric")

 # VaR should be positive
 assert result >= 0

 def test_calculate_cvar_basic(self):
 """Test CVaR calculation"""
 returns = generate_returns_data(1000)
 result = calculate_cvar(returns, confidence=0.05)

 # CVaR should be positive and >= VaR
 var_result = calculate_var(returns, confidence=0.05)
 assert result >= var_result

 def test_calculate_calmar_ratio_basic(self):
 """Test Calmar ratio calculation"""
 returns = [0.01, 0.02, -0.01, 0.03, -0.005, 0.01] * 50 # Enough data
 result = calculate_calmar_ratio(returns)

 # Calmar should be finite
 assert np.isfinite(result)


class TestPerformanceMetricsClass:
 """Test PerformanceMetrics class"""

 def test_initialization_default(self):
 """Test default initialization"""
 pm = PerformanceMetrics
 assert isinstance(pm.config, MetricsConfig)

 def test_initialization_with_config(self):
 """Test initialization with custom config"""
 config = MetricsConfig(
 risk_free_rate=0.03,
 periods_per_year=365
 )
 pm = PerformanceMetrics(config)

 assert pm.config.risk_free_rate == 0.03
 assert pm.config.periods_per_year == 365

 def test_calculate_all_metrics_basic(self):
 """Test comprehensive metrics calculation"""
 pm = PerformanceMetrics
 returns = generate_returns_data(100)

 result = pm.calculate_all_metrics(returns)

 # Check all required fields are present
 assert hasattr(result, 'total_return')
 assert hasattr(result, 'annualized_return')
 assert hasattr(result, 'volatility')
 assert hasattr(result, 'sharpe_ratio')
 assert hasattr(result, 'sortino_ratio')
 assert hasattr(result, 'max_drawdown')

 # Check values are reasonable
 assert np.isfinite(result.total_return)
 assert np.isfinite(result.volatility)
 assert result.max_drawdown >= 0

 def test_calculate_all_metrics_insufficient_data(self):
 """Test metrics calculation with insufficient data"""
 pm = PerformanceMetrics
 returns = [0.01, 0.02] # Very little data

 result = pm.calculate_all_metrics(returns)

 # Should return default values
 assert result.total_return == 0.0
 assert result.sharpe_ratio == 0.0

 def test_compare_strategies(self):
 """Test strategy comparison"""
 pm = PerformanceMetrics

 strategy_returns = {
 "Strategy A": generate_returns_data(100),
 "Strategy B": generate_returns_data(100),
 "Strategy C": generate_returns_data(100)
 }

 result = pm.compare_strategies(strategy_returns)

 # Should return DataFrame with comparison
 assert isinstance(result, pd.DataFrame)
 assert len(result) == 3 # Three strategies
 assert "Strategy" in result.columns
 assert "Sharpe Ratio" in result.columns


class TestTradeAndPosition:
 """Test Trade and Position classes"""

 def test_trade_creation(self):
 """Test trade creation"""
 entry_time = datetime.now
 trade = Trade(
 entry_time=entry_time,
 exit_time=None,
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 entry_price=50000.0,
 exit_price=None,
 quantity=0.1,
 entry_value=5000.0,
 exit_value=None,
 commission=5.0,
 slippage=2.0,
 pnl=None,
 return_pct=None,
 duration=None
 )

 assert trade.symbol == "BTCUSDT"
 assert trade.side == PositionSide.LONG
 assert trade.is_open is True

 def test_trade_close(self):
 """Test trade closing"""
 entry_time = datetime.now
 trade = Trade(
 entry_time=entry_time,
 exit_time=None,
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 entry_price=50000.0,
 exit_price=None,
 quantity=0.1,
 entry_value=5000.0,
 exit_value=None,
 commission=5.0,
 slippage=2.0,
 pnl=None,
 return_pct=None,
 duration=None
 )

 exit_time = entry_time + timedelta(hours=1)
 exit_price = 52000.0

 trade.close_trade(exit_price, exit_time, 5.0, 2.0)

 assert trade.is_open is False
 assert trade.exit_price == 52000.0
 assert trade.pnl is not None
 assert trade.pnl > 0 # Profitable trade

 def test_position_creation(self):
 """Test position creation"""
 position = Position(
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 quantity=0.1,
 avg_price=50000.0,
 market_value=5000.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )

 assert position.symbol == "BTCUSDT"
 assert position.quantity == 0.1

 def test_position_market_value_update(self):
 """Test position market value update"""
 position = Position(
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 quantity=0.1,
 avg_price=50000.0,
 market_value=5000.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )

 # Update with higher price
 position.update_market_value(52000.0)

 assert position.market_value == 5200.0 # 0.1 * 52000
 assert position.unrealized_pnl == 200.0 # Profit


class TestPortfolio:
 """Test Portfolio class"""

 def test_portfolio_creation(self):
 """Test portfolio creation"""
 portfolio = Portfolio(100000.0)

 assert portfolio.initial_capital == 100000.0
 assert portfolio.cash == 100000.0
 assert len(portfolio.positions) == 0
 assert len(portfolio.trades) == 0

 def test_portfolio_total_value(self):
 """Test portfolio total value calculation"""
 portfolio = Portfolio(100000.0)

 # Add a position
 position = Position(
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 quantity=0.1,
 avg_price=50000.0,
 market_value=5000.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )
 portfolio.positions["BTCUSDT"] = position

 # Update cash to simulate trade
 portfolio.cash = 95000.0

 total_value = portfolio.total_value
 assert total_value == 100000.0 # 95000 cash + 5000 position

 def test_portfolio_update_positions(self):
 """Test portfolio position updates"""
 portfolio = Portfolio(100000.0)

 # Add position
 position = Position(
 symbol="BTCUSDT",
 side=PositionSide.LONG,
 quantity=0.1,
 avg_price=50000.0,
 market_value=5000.0,
 unrealized_pnl=0.0,
 realized_pnl=0.0
 )
 portfolio.positions["BTCUSDT"] = position

 # Update with new prices
 prices = {"BTCUSDT": 52000.0}
 timestamp = datetime.now

 portfolio.update_positions(prices, timestamp)

 # Check position was updated
 assert portfolio.positions["BTCUSDT"].market_value == 5200.0
 assert len(portfolio.equity_curve) > 1


class TestSimpleStrategy(BaseStrategy):
 """Simple test strategy"""

 def generate_signals(self, data, timestamp, portfolio):
 """Generate simple buy-and-hold signal"""
 if len(data) == 1: # First day
 return [("main", OrderSide.BUY, 0.5)] # Buy 50% of portfolio
 return []


class TestBacktesting:
 """Test backtesting functionality"""

 def test_backtest_simple_strategy(self):
 """Test backtesting with simple strategy"""
 data = generate_ohlcv_data(100)
 strategy = TestSimpleStrategy
 config = BacktestConfig(initial_capital=100000.0)

 result = backtest_strategy(strategy, data, config)

 assert isinstance(result, BacktestResult)
 assert result.final_portfolio_value > 0
 assert result.total_trades >= 0
 assert result.total_days == 100

 def test_backtest_performance(self):
 """Test backtesting performance"""
 data = generate_ohlcv_data(1000)
 strategy = TestSimpleStrategy
 config = BacktestConfig(initial_capital=100000.0)

 threshold = PERFORMANCE_THRESHOLDS['evaluation']['backtest_strategy']

 result = assert_performance_acceptable(
 backtest_strategy, (strategy, data, config), threshold
 )

 assert isinstance(result, BacktestResult)

 def test_backtest_with_empty_data(self):
 """Test backtesting with empty data"""
 data = pd.DataFrame
 strategy = TestSimpleStrategy

 with pytest.raises(ValueError):
 backtest_strategy(strategy, data)

 def test_backtest_insufficient_data(self):
 """Test backtesting with insufficient data"""
 data = generate_ohlcv_data(1) # Only one day
 strategy = TestSimpleStrategy

 with pytest.raises(ValueError):
 backtest_strategy(strategy, data)

 def test_backtest_config_customization(self):
 """Test backtesting with custom configuration"""
 data = generate_ohlcv_data(50)
 strategy = TestSimpleStrategy

 config = BacktestConfig(
 initial_capital=50000.0,
 commission=0.002, # Higher commission
 position_size=0.2 # Smaller position size
 )

 result = backtest_strategy(strategy, data, config)

 assert isinstance(result, BacktestResult)

 def test_analyze_backtest_results(self):
 """Test backtest results analysis"""
 data = generate_ohlcv_data(50)
 strategy = TestSimpleStrategy
 result = backtest_strategy(strategy, data)

 analysis = analyze_backtest_results(result)

 assert isinstance(analysis, dict)
 assert "performance_summary" in analysis
 assert "trade_analysis" in analysis
 assert "risk_metrics" in analysis

 def test_generate_performance_report(self):
 """Test performance report generation"""
 data = generate_ohlcv_data(50)
 strategy = TestSimpleStrategy
 result = backtest_strategy(strategy, data)

 report = generate_performance_report(result)

 assert isinstance(report, str)
 assert "BACKTEST PERFORMANCE REPORT" in report
 assert "Total Return" in report
 assert "Sharpe Ratio" in report


class TestFunctionBasedStrategy:
 """Test function-based strategy wrapper"""

 def simple_signal_function(self, data, timestamp, portfolio):
 """Simple signal function"""
 if len(data) % 10 == 1: # Buy every 10 days
 return [("main", OrderSide.BUY, 0.1)]
 return []

 def test_strategy_wrapper(self):
 """Test Strategy wrapper class"""
 strategy = Strategy(self.simple_signal_function)
 data = generate_ohlcv_data(50)

 result = backtest_strategy(strategy, data)

 assert isinstance(result, BacktestResult)
 assert result.total_trades > 0


class TestEdgeCases:
 """Test edge cases and error conditions"""

 def test_zero_returns_metrics(self):
 """Test metrics with zero returns"""
 returns = [0.0] * 100

 sharpe = calculate_sharpe_ratio(returns)
 assert sharpe == 0.0 # No volatility

 max_dd = calculate_max_drawdown(returns)
 assert max_dd == 0.0 # No drawdown

 def test_extreme_returns_metrics(self):
 """Test metrics with extreme returns"""
 returns = [10.0, -0.5, 5.0, -0.8, 2.0] # Very high returns

 # Should not crash
 sharpe = calculate_sharpe_ratio(returns)
 assert np.isfinite(sharpe)

 var_result = calculate_var(returns)
 assert np.isfinite(var_result)

 def test_single_return_metrics(self):
 """Test metrics with single return"""
 returns = [0.05]

 sharpe = calculate_sharpe_ratio(returns)
 assert sharpe == 0.0

 win_rate = calculate_win_rate(returns)
 assert win_rate == 1.0

 def test_all_negative_returns(self):
 """Test metrics with all negative returns"""
 returns = [-0.01, -0.02, -0.03, -0.01, -0.02]

 win_rate = calculate_win_rate(returns)
 assert win_rate == 0.0

 profit_factor = calculate_profit_factor(returns)
 assert profit_factor == 1.0 # As defined


if __name__ == "__main__":
 pytest.main([__file__, "-v"])