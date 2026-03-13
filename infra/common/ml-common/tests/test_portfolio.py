"""
Test Suite for Portfolio Module
Testing Patterns

Comprehensive tests for portfolio feature extraction:
- Position features (20 dims): Quantities, values, weights, exposure
- Performance metrics (30 dims): PnL, returns, Sharpe, Sortino, drawdown
- Risk metrics: Concentration, leverage, exposure
- Performance validation (<3ms)
- Edge cases (empty portfolio, single position, 100% cash)
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from portfolio import (
 # Position state
 extract_position_features,
 calculate_position_weights,
 calculate_exposure_metrics,
 calculate_concentration_metrics,
 # Performance
 extract_performance_features,
 calculate_pnl_metrics,
 calculate_return_metrics,
 calculate_risk_adjusted_metrics,
 calculate_drawdown_metrics,
)

from . import (
 assert_performance_acceptable,
 generate_price_data,
)


def generate_portfolio_state(
 cash: float = 10000.0,
 positions: Dict[str, float] = None
) -> Dict[str, Any]:
 """Generate portfolio state for testing"""
 if positions is None:
 positions = {
 'BTCUSDT': 0.5, # 0.5 BTC
 'ETHUSDT': 2.0, # 2 ETH
 'BNBUSDT': 10.0, # 10 BNB
 'SOLUSDT': 50.0, # 50 SOL
 }

 # Calculate total value
 prices = {
 'BTCUSDT': 50000.0,
 'ETHUSDT': 3000.0,
 'BNBUSDT': 400.0,
 'SOLUSDT': 100.0,
 }

 position_value = sum(positions.get(s, 0) * prices[s] for s in prices.keys)
 total_value = cash + position_value

 return {
 'positions': positions,
 'cash': cash,
 'total_value': total_value,
 }


def generate_current_prices(
 base_prices: Dict[str, float] = None
) -> Dict[str, float]:
 """Generate current prices for testing"""
 if base_prices is None:
 return {
 'BTCUSDT': 50000.0,
 'ETHUSDT': 3000.0,
 'BNBUSDT': 400.0,
 'SOLUSDT': 100.0,
 }
 return base_prices


def generate_portfolio_history(
 n_hours: int = 168,
 initial_value: float = 100000.0,
 volatility: float = 0.01
) -> pd.DataFrame:
 """Generate portfolio history for testing"""
 np.random.seed(42)

 data = []
 current_value = initial_value

 for i in range(n_hours):
 # Random returns
 returns = np.random.normal(0.0005, volatility)
 current_value *= (1 + returns)

 # PnL
 pnl = current_value - initial_value
 pnl_1h = current_value * returns

 data.append({
 'timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i),
 'total_value': current_value,
 'cash': 10000.0,
 'pnl': pnl,
 'unrealized_pnl': pnl * 0.6,
 'realized_pnl': pnl * 0.4,
 'trades': 0 if i % 10 != 0 else 1,
 })

 return pd.DataFrame(data)


class TestPositionFeatures:
 """Test position state features"""

 def test_extract_position_features_basic(self):
 """Test basic position feature extraction"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Should return 20 dimensions
 assert isinstance(features, np.ndarray)
 assert features.shape == (20,), f"Expected shape (20,), got {features.shape}"

 # All should be finite
 assert np.all(np.isfinite(features)), "All features should be finite"

 def test_extract_position_features_quantities(self):
 """Test position quantities (first 4 dims)"""
 portfolio = generate_portfolio_state(
 positions={'BTCUSDT': 1.0, 'ETHUSDT': 5.0, 'BNBUSDT': 20.0, 'SOLUSDT': 100.0}
 )
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # First 4 dims should be quantities
 assert features[0] == 1.0 # BTC
 assert features[1] == 5.0 # ETH
 assert features[2] == 20.0 # BNB
 assert features[3] == 100.0 # SOL

 def test_extract_position_features_values(self):
 """Test position values (dims 4-7)"""
 portfolio = generate_portfolio_state(
 positions={'BTCUSDT': 1.0, 'ETHUSDT': 2.0, 'BNBUSDT': 10.0, 'SOLUSDT': 50.0}
 )
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Dims 4-7 should be position values
 expected_values = [
 1.0 * 50000.0, # BTC
 2.0 * 3000.0, # ETH
 10.0 * 400.0, # BNB
 50.0 * 100.0, # SOL
 ]

 for i, expected in enumerate(expected_values):
 assert abs(features[4 + i] - expected) < 1.0, f"Value mismatch at index {4+i}"

 def test_extract_position_features_weights(self):
 """Test position weights (dims 8-11)"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Dims 8-11 should be weights (sum should be close to 1.0 or less)
 weights = features[8:12]
 assert np.all(weights >= 0), "Weights should be non-negative"
 assert np.sum(weights) <= 1.0, "Weights should not exceed 1.0 (due to cash)"

 def test_extract_position_features_exposure(self):
 """Test exposure metrics (dims 12-18)"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Dim 12: Total exposure
 assert features[12] >= 0, "Total exposure should be non-negative"

 # Dim 13: Cash ratio
 assert 0 <= features[13] <= 1, "Cash ratio should be 0-1"

 # Dim 14-17: Long/short/net/gross exposure
 assert np.all(np.isfinite(features[14:18]))

 # Dim 18: Leverage
 assert features[18] >= 0, "Leverage should be non-negative"

 def test_extract_position_features_concentration(self):
 """Test concentration metric (dim 19)"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Dim 19: Herfindahl index (concentration)
 herfindahl = features[19]
 assert 0 <= herfindahl <= 1, "Herfindahl should be 0-1"

 def test_calculate_position_weights(self):
 """Test position weight calculation"""
 positions = {'BTCUSDT': 0.5, 'ETHUSDT': 2.0}
 prices = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}
 total_value = 50000.0

 weights = calculate_position_weights(positions, prices, total_value)

 # BTC weight: 0.5 * 50000 / 50000 = 0.5
 assert abs(weights['BTCUSDT'] - 0.5) < 0.01

 # ETH weight: 2.0 * 3000 / 50000 = 0.12
 assert abs(weights['ETHUSDT'] - 0.12) < 0.01

 def test_calculate_exposure_metrics(self):
 """Test exposure metrics calculation"""
 positions = {'BTCUSDT': 0.5, 'ETHUSDT': 2.0}
 prices = {'BTCUSDT': 50000.0, 'ETHUSDT': 3000.0}
 cash = 10000.0
 total_value = 41000.0

 metrics = calculate_exposure_metrics(positions, prices, cash, total_value)

 # Total exposure = 25000 + 6000 = 31000
 assert abs(metrics['total_exposure'] - 31000.0) < 1.0

 # Cash ratio = 10000 / 41000 ≈ 0.244
 assert abs(metrics['cash_ratio'] - (10000/41000)) < 0.01

 # Long exposure = 31000 (no shorts)
 assert abs(metrics['long_exposure'] - 31000.0) < 1.0

 # Leverage = 31000 / 41000 ≈ 0.756
 assert 0 <= metrics['leverage'] <= 1.0

 def test_calculate_concentration_metrics(self):
 """Test concentration metrics"""
 # Equal weights (4 positions)
 position_values = [10000.0, 10000.0, 10000.0, 10000.0]

 metrics = calculate_concentration_metrics(position_values)

 # HHI for equal weights: 1/4 = 0.25
 expected_hhi = 0.25
 assert abs(metrics['herfindahl_index'] - expected_hhi) < 0.01

 # Max weight should be 0.25
 assert abs(metrics['max_weight'] - 0.25) < 0.01

 # Diversification ratio: 1/0.25 = 4
 assert abs(metrics['diversification_ratio'] - 4.0) < 0.1


class TestPerformanceFeatures:
 """Test performance metrics"""

 def test_extract_performance_features_basic(self):
 """Test basic performance feature extraction"""
 history = generate_portfolio_history(168)

 features = extract_performance_features(history, window_hours=168)

 # Should return 30 dimensions
 assert isinstance(features, np.ndarray)
 assert features.shape == (30,), f"Expected shape (30,), got {features.shape}"

 # All should be finite
 assert np.all(np.isfinite(features)), "All features should be finite"

 def test_calculate_pnl_metrics(self):
 """Test PnL metrics calculation"""
 history = generate_portfolio_history(168, initial_value=100000.0)

 metrics = calculate_pnl_metrics(history)

 # Should have 6 keys
 assert len(metrics) == 6

 # Total PnL
 assert 'total_pnl' in metrics
 assert isinstance(metrics['total_pnl'], float)

 # Realized/unrealized
 assert 'realized_pnl' in metrics
 assert 'unrealized_pnl' in metrics

 # Time-based PnL
 assert 'pnl_1h' in metrics
 assert 'pnl_24h' in metrics
 assert 'pnl_7d' in metrics

 def test_calculate_return_metrics(self):
 """Test return metrics calculation"""
 history = generate_portfolio_history(168, initial_value=100000.0)

 metrics = calculate_return_metrics(history)

 # Should have 3 keys
 assert len(metrics) == 3

 # Returns
 assert 'return_1h' in metrics
 assert 'return_24h' in metrics
 assert 'return_7d' in metrics

 # Returns should be reasonable (-50% to +100%)
 for key, value in metrics.items:
 assert -0.5 <= value <= 1.0, f"{key} out of range: {value}"

 def test_calculate_risk_adjusted_metrics(self):
 """Test risk-adjusted metrics (Sharpe, Sortino, etc.)"""
 history = generate_portfolio_history(168)

 metrics = calculate_risk_adjusted_metrics(history, risk_free_rate=0.0)

 # Should have 5 keys
 assert len(metrics) == 5

 # Sharpe ratio
 assert 'sharpe_ratio' in metrics
 assert isinstance(metrics['sharpe_ratio'], float)

 # Sortino ratio
 assert 'sortino_ratio' in metrics

 # Calmar ratio
 assert 'calmar_ratio' in metrics

 # Information ratio
 assert 'information_ratio' in metrics

 # Treynor ratio
 assert 'treynor_ratio' in metrics

 # All should be finite
 for key, value in metrics.items:
 assert np.isfinite(value), f"{key} is not finite: {value}"

 def test_calculate_drawdown_metrics(self):
 """Test drawdown metrics"""
 history = generate_portfolio_history(168)

 metrics = calculate_drawdown_metrics(history)

 # Should have 4 keys
 assert len(metrics) == 4

 # Max drawdown (should be negative or zero)
 assert metrics['max_drawdown'] <= 0, "Max drawdown should be <= 0"

 # Current drawdown (should be negative or zero)
 assert metrics['current_drawdown'] <= 0

 # Drawdown duration (hours)
 assert 0 <= metrics['drawdown_duration'] <= 168

 # Recovery factor
 assert 'recovery_factor' in metrics

 def test_sharpe_ratio_calculation(self):
 """Test Sharpe ratio with known data"""
 # Create portfolio with steady positive returns
 data = []
 for i in range(100):
 value = 100000.0 * (1.001 ** i) # +0.1% per hour
 data.append({
 'timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i),
 'total_value': value,
 })

 history = pd.DataFrame(data)
 metrics = calculate_risk_adjusted_metrics(history)

 # Sharpe should be positive (positive returns)
 assert metrics['sharpe_ratio'] > 0, "Sharpe should be positive for positive returns"

 def test_drawdown_detection(self):
 """Test max drawdown detection"""
 # Create portfolio with known drawdown
 values = [100000, 105000, 110000, 100000, 95000, 100000, 105000] # -13.6% max DD
 data = []
 for i, value in enumerate(values):
 data.append({
 'timestamp': pd.Timestamp('2025-01-01') + pd.Timedelta(hours=i),
 'total_value': value,
 })

 history = pd.DataFrame(data)
 metrics = calculate_drawdown_metrics(history)

 # Max drawdown should be around -13.6%
 expected_dd = (95000 - 110000) / 110000 # -0.136
 assert abs(metrics['max_drawdown'] - expected_dd) < 0.01


class TestPortfolioEdgeCases:
 """Test edge cases and error handling"""

 def test_empty_portfolio(self):
 """Test with empty portfolio (100% cash)"""
 portfolio = generate_portfolio_state(cash=100000.0, positions={})
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Should return 20 dimensions
 assert features.shape == (20,)

 # First 4 dims (quantities) should be 0
 assert np.all(features[0:4] == 0)

 # Cash ratio should be 1.0
 assert abs(features[13] - 1.0) < 0.01

 def test_single_position(self):
 """Test with single position"""
 portfolio = generate_portfolio_state(
 cash=50000.0,
 positions={'BTCUSDT': 1.0, 'ETHUSDT': 0.0, 'BNBUSDT': 0.0, 'SOLUSDT': 0.0}
 )
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Should return 20 dimensions
 assert features.shape == (20,)

 # Only BTC should have quantity
 assert features[0] == 1.0
 assert np.all(features[1:4] == 0)

 # Herfindahl should be 1.0 (fully concentrated)
 assert abs(features[19] - 1.0) < 0.01

 def test_zero_total_value(self):
 """Test with zero total value"""
 portfolio = {
 'positions': {},
 'cash': 0.0,
 'total_value': 0.0,
 }
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Should return zeros
 assert features.shape == (20,)
 assert np.all(features == 0)

 def test_short_positions(self):
 """Test with short positions (negative quantities)"""
 # Pass sufficient cash to ensure positive total_value with short position
 # Short BTC: -0.5 * 50000 = -25000, Long ETH: 2.0 * 3000 = 6000
 # Need cash > 19000 to make total_value positive
 portfolio = generate_portfolio_state(
 cash=50000.0, # Sufficient cash to cover short position
 positions={'BTCUSDT': -0.5, 'ETHUSDT': 2.0, 'BNBUSDT': 0.0, 'SOLUSDT': 0.0}
 )
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Should handle shorts correctly
 assert features[0] == -0.5 # Negative quantity

 # Short exposure should be captured
 assert features[15] > 0 # Short exposure (dim 15)

 def test_insufficient_history(self):
 """Test performance with insufficient history"""
 history = generate_portfolio_history(n_hours=1) # Only 1 hour

 features = extract_performance_features(history, window_hours=168)

 # Should return zeros or defaults
 assert features.shape == (30,)

 def test_empty_history(self):
 """Test performance with empty history"""
 history = pd.DataFrame

 features = extract_performance_features(history, window_hours=168)

 # Should return zeros
 assert features.shape == (30,)
 assert np.all(features == 0)

 def test_high_leverage(self):
 """Test portfolio with high leverage"""
 # Large positions relative to total value
 portfolio = {
 'positions': {'BTCUSDT': 5.0, 'ETHUSDT': 100.0}, # $500k exposure
 'cash': 10000.0,
 'total_value': 100000.0, # But only $100k total
 }
 prices = generate_current_prices

 features = extract_position_features(portfolio, prices)

 # Leverage should be > 1
 leverage = features[18]
 assert leverage > 1.0, f"Expected leverage > 1, got {leverage}"


class TestPortfolioIntegration:
 """Integration tests for complete portfolio feature extraction"""

 def test_extract_all_portfolio_features(self):
 """Test extracting all 50 portfolio features"""
 # Position features (20 dims)
 portfolio = generate_portfolio_state
 prices = generate_current_prices
 position_features = extract_position_features(portfolio, prices)

 # Performance features (30 dims)
 history = generate_portfolio_history(168)
 performance_features = extract_performance_features(history, window_hours=168)

 # Combine
 all_features = np.concatenate([position_features, performance_features])

 # Should have 50 dimensions
 assert all_features.shape == (50,), f"Expected 50 dims, got {all_features.shape}"

 # All should be finite
 assert np.all(np.isfinite(all_features)), "All features should be finite"

 def test_feature_consistency(self):
 """Test feature extraction consistency"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 # Extract twice
 features1 = extract_position_features(portfolio, prices)
 features2 = extract_position_features(portfolio, prices)

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_performance_position_features(self):
 """Test position feature extraction performance"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices

 result = assert_performance_acceptable(
 extract_position_features,
 (portfolio, prices),
 max_time_ms=1.0 # <1ms for 20 features
 )

 assert result.shape == (20,)

 def test_performance_full_features(self):
 """Test performance feature extraction performance"""
 history = generate_portfolio_history(168)

 result = assert_performance_acceptable(
 extract_performance_features,
 (history, 168),
 max_time_ms=3.0 # <3ms for 30 features
 )

 assert result.shape == (30,)

 def test_performance_full_extraction(self):
 """Test performance of full portfolio extraction"""
 portfolio = generate_portfolio_state
 prices = generate_current_prices
 history = generate_portfolio_history(168)

 def extract_all:
 pos = extract_position_features(portfolio, prices)
 perf = extract_performance_features(history, 168)
 return np.concatenate([pos, perf])

 result = assert_performance_acceptable(
 extract_all,
 max_time_ms=5.0 # <5ms for all 50 features
 )

 assert result.shape == (50,)


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
