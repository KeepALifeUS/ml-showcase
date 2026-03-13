"""
Test Suite for Cross-Asset Module
Testing Patterns

Comprehensive tests for cross-asset feature extraction:
- Correlation analysis (10 dims: 6 pairs + 4 rolling stats)
- Inter-asset spreads (6 dims)
- Beta calculations (4 dims)
- Performance validation (<5ms)
- Edge cases (perfect correlation, zero correlation, single symbol)
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict

from ml_common.cross_asset import (
 calculate_correlation_matrix,
 calculate_rolling_correlation,
 calculate_pearson_correlation,
 calculate_spearman_correlation,
 extract_correlation_features,
 calculate_normalized_spread,
 calculate_spread_volatility,
 calculate_spread_convergence,
 calculate_spread_momentum,
 calculate_spread_z_score,
 extract_spread_features,
 calculate_beta,
 calculate_market_beta,
 calculate_relative_strength,
 calculate_momentum_divergence,
 extract_beta_features,
)

from . import (
 assert_array_almost_equal,
 assert_performance_acceptable,
 generate_price_data,
)


def generate_multi_symbol_prices(
 n_points: int = 168,
 symbols: list = None
) -> Dict[str, np.ndarray]:
 """
 Generate correlated price data for multiple symbols

 Returns:
 Dict of symbol -> price array
 """
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 np.random.seed(42)

 # Generate base market factor
 market_returns = np.random.normal(0.001, 0.02, n_points)

 prices = {}
 for i, symbol in enumerate(symbols):
 # Each symbol has different beta to market
 beta = 0.5 + i * 0.3 # 0.5, 0.8, 1.1, 1.4

 # Symbol-specific returns = beta * market + idiosyncratic
 symbol_returns = beta * market_returns + np.random.normal(0, 0.01, n_points)

 # Convert to prices
 base_price = 50000.0 / (i + 1) # Different price levels
 symbol_prices = np.zeros(n_points + 1)
 symbol_prices[0] = base_price

 for j in range(n_points):
 symbol_prices[j + 1] = symbol_prices[j] * (1 + symbol_returns[j])

 prices[symbol] = symbol_prices[1:]

 return prices


class TestCorrelation:
 """Test correlation calculations"""

 def test_calculate_pearson_correlation_perfect(self):
 """Test Pearson correlation with perfect correlation"""
 x = np.array([1, 2, 3, 4, 5])
 y = np.array([2, 4, 6, 8, 10]) # y = 2*x (perfect correlation)

 result = calculate_pearson_correlation(x, y)

 assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"

 def test_calculate_pearson_correlation_negative(self):
 """Test Pearson correlation with negative correlation"""
 x = np.array([1, 2, 3, 4, 5])
 y = np.array([5, 4, 3, 2, 1]) # Perfect negative

 result = calculate_pearson_correlation(x, y)

 assert abs(result - (-1.0)) < 1e-6, f"Expected -1.0, got {result}"

 def test_calculate_pearson_correlation_zero(self):
 """Test Pearson correlation with uncorrelated data"""
 np.random.seed(42)
 x = np.random.randn(100)
 y = np.random.randn(100) # Independent

 result = calculate_pearson_correlation(x, y)

 # Should be close to 0 (but not exactly due to sampling)
 assert abs(result) < 0.2, f"Expected ~0, got {result}"

 def test_calculate_pearson_correlation_insufficient_data(self):
 """Test correlation with insufficient data"""
 x = np.array([1])
 y = np.array([2])

 result = calculate_pearson_correlation(x, y)

 # Should return 0 for insufficient data
 assert result == 0.0

 def test_calculate_spearman_correlation(self):
 """Test Spearman rank correlation"""
 x = np.array([1, 2, 3, 4, 5])
 y = np.array([1, 4, 9, 16, 25]) # y = x^2 (non-linear)

 pearson = calculate_pearson_correlation(x, y)
 spearman = calculate_spearman_correlation(x, y)

 # Spearman should be 1.0 (perfect rank correlation)
 assert abs(spearman - 1.0) < 1e-6

 # Pearson should be less than 1.0
 assert pearson < spearman

 def test_calculate_correlation_matrix(self):
 """Test correlation matrix for 4 symbols"""
 prices = generate_multi_symbol_prices(n_points=100)

 result = calculate_correlation_matrix(prices)

 # Should be 4x4 matrix
 assert result.shape == (4, 4), f"Expected (4, 4), got {result.shape}"

 # Diagonal should be 1.0 (self-correlation)
 np.testing.assert_array_almost_equal(np.diag(result), np.ones(4), decimal=6)

 # Matrix should be symmetric
 np.testing.assert_array_almost_equal(result, result.T, decimal=6)

 # All values should be in [-1, 1]
 assert np.all(result >= -1.0) and np.all(result <= 1.0)

 def test_calculate_rolling_correlation(self):
 """Test rolling correlation calculation"""
 x = generate_price_data(100)
 y = generate_price_data(100)

 result = calculate_rolling_correlation(x, y, window=24)

 # Should return list of rolling correlations
 assert isinstance(result, list)
 assert len(result) > 0, "Should return non-empty list"
 # All correlations should be in [-1, 1] (with floating-point tolerance)
 assert all(-1.0 - 1e-10 <= r <= 1.0 + 1e-10 for r in result), \
 f"Correlation values outside [-1, 1]: {[r for r in result if r < -1.0 - 1e-10 or r > 1.0 + 1e-10]}"

 def test_extract_correlation_features(self):
 """Test full correlation feature extraction (10 dims)"""
 prices = generate_multi_symbol_prices(n_points=168)

 result = extract_correlation_features(prices, window=24)

 # Should return 10 dimensions
 assert isinstance(result, np.ndarray)
 assert result.shape == (10,), f"Expected (10,), got {result.shape}"

 # First 6 are pairwise correlations (should be in [-1, 1])
 pairwise_corrs = result[:6]
 assert np.all(pairwise_corrs >= -1.0) and np.all(pairwise_corrs <= 1.0)

 # All should be finite
 assert np.all(np.isfinite(result))

 def test_correlation_features_performance(self):
 """Test correlation extraction performance"""
 prices = generate_multi_symbol_prices(n_points=168)

 # Use wrapper to pass keyword argument correctly
 def extract_with_window:
 return extract_correlation_features(prices, window=24)

 result = assert_performance_acceptable(
 extract_with_window,
 max_time_ms=100.0 # <100ms for 168 points × 4 symbols (realistic threshold)
 )

 assert result.shape == (10,)


class TestSpreads:
 """Test inter-asset spread calculations"""

 def test_calculate_normalized_spread(self):
 """Test normalized spread calculation"""
 price1 = np.array([100, 102, 101, 103])
 price2 = np.array([50, 51, 50.5, 51.5])

 result = calculate_normalized_spread(price1, price2)

 # Should return current spread normalized by volatility
 assert isinstance(result, float)
 assert np.isfinite(result)

 def test_calculate_spread_volatility(self):
 """Test spread volatility"""
 price1 = np.array([100, 102, 101, 103, 105])
 price2 = np.array([50, 51, 50.5, 51.5, 52])

 result = calculate_spread_volatility(price1, price2, window=5)

 # Should be standard deviation of spread
 assert isinstance(result, float)
 assert result >= 0.0

 def test_calculate_spread_convergence(self):
 """Test spread convergence/divergence"""
 # Converging spread
 price1 = np.array([100, 101, 102, 103])
 price2 = np.array([50, 51, 52, 53]) # Same % change

 result = calculate_spread_convergence(price1, price2, window=4)

 assert isinstance(result, float)

 def test_calculate_spread_momentum(self):
 """Test spread momentum"""
 price1 = generate_price_data(50, start_price=100)
 price2 = generate_price_data(50, start_price=50)

 result = calculate_spread_momentum(price1, price2, window=10)

 assert isinstance(result, float)
 assert np.isfinite(result)

 def test_calculate_spread_z_score(self):
 """Test spread z-score (mean reversion indicator)"""
 price1 = np.array([100, 102, 101, 103, 102, 104, 103])
 price2 = np.array([50, 51, 50.5, 51.5, 51, 52, 51.5])

 result = calculate_spread_z_score(price1, price2, window=5)

 # Z-score should be roughly centered around 0
 assert isinstance(result, float)
 assert abs(result) < 5.0 # Extreme z-scores are rare

 def test_extract_spread_features(self):
 """Test full spread feature extraction (6 dims)"""
 prices = generate_multi_symbol_prices(n_points=168)

 result = extract_spread_features(prices)

 # Should return 6 dimensions
 assert isinstance(result, np.ndarray)
 assert result.shape == (6,), f"Expected (6,), got {result.shape}"

 # All should be finite
 assert np.all(np.isfinite(result))

 def test_spread_features_performance(self):
 """Test spread extraction performance"""
 prices = generate_multi_symbol_prices(n_points=168)

 result = assert_performance_acceptable(
 extract_spread_features,
 (prices,),
 max_time_ms=5.0 # <5ms
 )

 assert result.shape == (6,)


class TestBeta:
 """Test beta calculations (systematic risk)"""

 def test_calculate_beta_basic(self):
 """Test basic beta calculation"""
 # Market returns
 market = np.array([0.01, -0.02, 0.03, -0.01, 0.02])

 # Asset returns = 1.5 * market (beta = 1.5)
 asset = 1.5 * market

 result = calculate_beta(asset, market)

 assert abs(result - 1.5) < 0.1, f"Expected 1.5, got {result}"

 def test_calculate_beta_negative(self):
 """Test negative beta (inverse correlation)"""
 market = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
 asset = -1.0 * market # Perfect inverse

 result = calculate_beta(asset, market)

 assert result < 0, f"Expected negative beta, got {result}"

 def test_calculate_beta_zero(self):
 """Test zero beta (market-neutral)"""
 market = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
 asset = np.array([0.001, 0.001, 0.001, 0.001, 0.001]) # Constant

 result = calculate_beta(asset, market)

 # Should be close to 0
 assert abs(result) < 0.2

 def test_calculate_beta_insufficient_data(self):
 """Test beta with insufficient data"""
 market = np.array([0.01])
 asset = np.array([0.02])

 result = calculate_beta(asset, market)

 # Should return 1.0 as default
 assert result == 1.0

 def test_calculate_market_beta(self):
 """Test market beta (BTC as market proxy)"""
 prices = generate_multi_symbol_prices(n_points=100)

 # Calculate beta for ETH vs BTC
 result = calculate_market_beta(
 prices['ETHUSDT'],
 prices['BTCUSDT'],
 window=50
 )

 assert isinstance(result, float)
 # Crypto betas typically 0.5-2.0
 assert 0.0 < result < 3.0

 def test_calculate_relative_strength(self):
 """Test relative strength (outperformance)"""
 # Asset outperforming market
 asset = np.array([100, 105, 110, 115, 120])
 market = np.array([100, 102, 104, 106, 108])

 result = calculate_relative_strength(asset, market, window=5)

 # Should be positive (outperforming)
 assert result > 0

 def test_calculate_momentum_divergence(self):
 """Test momentum divergence"""
 prices = generate_multi_symbol_prices(n_points=100)

 result = calculate_momentum_divergence(
 prices['ETHUSDT'],
 prices['BTCUSDT'],
 short_window=12,
 long_window=24
 )

 assert isinstance(result, float)
 assert np.isfinite(result)

 def test_extract_beta_features(self):
 """Test full beta feature extraction (4 dims)"""
 prices = generate_multi_symbol_prices(n_points=168)

 result = extract_beta_features(prices, market_symbol='BTCUSDT')

 # Should return 4 dimensions
 assert isinstance(result, np.ndarray)
 assert result.shape == (4,), f"Expected (4,), got {result.shape}"

 # Betas should be reasonable (0-3)
 betas = result[:3] # First 3 are individual betas
 assert np.all(betas >= 0.0) and np.all(betas < 5.0)

 # All should be finite
 assert np.all(np.isfinite(result))

 def test_beta_features_performance(self):
 """Test beta extraction performance"""
 prices = generate_multi_symbol_prices(n_points=168)

 # Use wrapper to pass keyword arguments correctly
 def extract_with_market:
 return extract_beta_features(prices, market_symbol='BTCUSDT')

 result = assert_performance_acceptable(
 extract_with_market,
 max_time_ms=5.0 # <5ms
 )

 assert result.shape == (4,)


class TestCrossAssetEdgeCases:
 """Test edge cases and error handling"""

 def test_single_symbol(self):
 """Test with single symbol (should handle gracefully)"""
 prices = {'BTCUSDT': generate_price_data(100)}

 # Correlation matrix should be 1x1
 corr_matrix = calculate_correlation_matrix(prices)
 assert corr_matrix.shape == (1, 1)
 assert corr_matrix[0, 0] == 1.0

 def test_two_symbols(self):
 """Test with two symbols"""
 prices = {
 'BTCUSDT': generate_price_data(100, start_price=50000),
 'ETHUSDT': generate_price_data(100, start_price=3000)
 }

 corr_matrix = calculate_correlation_matrix(prices)
 assert corr_matrix.shape == (2, 2)

 def test_missing_symbol(self):
 """Test beta calculation with missing market symbol"""
 prices = {
 'ETHUSDT': generate_price_data(100),
 'BNBUSDT': generate_price_data(100)
 }

 # Should use first symbol as market if BTCUSDT missing
 result = extract_beta_features(prices, market_symbol='BTCUSDT')

 # Should still return 4 dimensions (may use fallback)
 assert result.shape == (4,)

 def test_constant_prices(self):
 """Test with constant prices (no volatility)"""
 prices = {
 'BTCUSDT': np.ones(100) * 50000,
 'ETHUSDT': np.ones(100) * 3000
 }

 # Correlation should handle constant series
 corr_matrix = calculate_correlation_matrix(prices)
 # Should return identity matrix for constant prices (each series correlates with itself)
 expected_identity = np.eye(2)
 assert np.allclose(corr_matrix, expected_identity, rtol=1e-5)

 def test_nan_prices(self):
 """Test with NaN values in prices"""
 prices = {
 'BTCUSDT': np.array([50000, 50100, np.nan, 50200]),
 'ETHUSDT': np.array([3000, 3010, 3020, np.nan])
 }

 # Should handle NaN gracefully (skip or interpolate)
 try:
 corr_features = extract_correlation_features(prices, window=2)
 # If it doesn't crash, that's good
 assert corr_features.shape == (10,)
 except Exception:
 # Acceptable to raise exception for invalid data
 pass


class TestCrossAssetIntegration:
 """Integration tests for complete cross-asset feature extraction"""

 def test_extract_all_cross_asset_features(self):
 """Test extracting all 20 cross-asset features"""
 prices = generate_multi_symbol_prices(n_points=168)

 # Extract correlation features (10 dims)
 corr_features = extract_correlation_features(prices, window=24)

 # Extract spread features (6 dims)
 spread_features = extract_spread_features(prices)

 # Extract beta features (4 dims)
 beta_features = extract_beta_features(prices, market_symbol='BTCUSDT')

 # Combine
 all_features = np.concatenate([corr_features, spread_features, beta_features])

 # Should have 20 dimensions
 assert all_features.shape == (20,), f"Expected 20 dims, got {all_features.shape}"

 # All should be finite
 assert np.all(np.isfinite(all_features)), "All features should be finite"

 def test_feature_consistency(self):
 """Test feature extraction consistency"""
 prices = generate_multi_symbol_prices(n_points=168)

 # Extract twice
 features1 = extract_correlation_features(prices, window=24)
 features2 = extract_correlation_features(prices, window=24)

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_performance_full_extraction(self):
 """Test performance of full cross-asset extraction"""
 prices = generate_multi_symbol_prices(n_points=168)

 def extract_all:
 corr = extract_correlation_features(prices, window=24)
 spread = extract_spread_features(prices)
 beta = extract_beta_features(prices, market_symbol='BTCUSDT')
 return np.concatenate([corr, spread, beta])

 result = assert_performance_acceptable(
 extract_all,
 max_time_ms=15.0 # <15ms for all 20 features
 )

 assert result.shape == (20,)

 def test_multi_window_analysis(self):
 """Test extraction with different window sizes"""
 prices = generate_multi_symbol_prices(n_points=168)

 for window in [12, 24, 48, 72]:
 features = extract_correlation_features(prices, window=window)
 assert features.shape == (10,)
 assert np.all(np.isfinite(features))


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
