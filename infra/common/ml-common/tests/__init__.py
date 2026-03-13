"""
Test Suite for ML Common Package

Comprehensive test suite implementing testing patterns.
Covers all modules with unit tests, integration tests, and performance benchmarks.

Test Structure:
- Unit tests for each module
- Integration tests for cross-module functionality
- Performance benchmarks
- Edge case validation
- Type safety tests
"""

import pytest
import numpy as np
import pandas as pd
import warnings

# Configure test environment
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Test data generators
def generate_price_data(n_points: int = 1000, start_price: float = 100.0) -> np.ndarray:
 """Generate realistic price data for testing"""
 np.random.seed(42) # For reproducible tests

 # Generate returns
 returns = np.random.normal(0.0005, 0.02, n_points) # Daily returns with drift

 # Add some volatility clustering
 volatility = np.random.exponential(0.01, n_points)
 returns = returns * (1 + volatility)

 # Convert to prices
 prices = np.zeros(n_points + 1)
 prices[0] = start_price

 for i in range(n_points):
 prices[i + 1] = prices[i] * (1 + returns[i])

 return prices[1:] # Return prices without initial value

def generate_ohlcv_data(n_points: int = 1000) -> pd.DataFrame:
 """Generate OHLCV data for testing"""
 np.random.seed(42)

 close_prices = generate_price_data(n_points)

 # Generate other OHLCV components
 data = []
 for i, close in enumerate(close_prices):
 # High and low around close
 volatility = 0.01 + np.random.exponential(0.005)
 high = close * (1 + np.random.uniform(0, volatility))
 low = close * (1 - np.random.uniform(0, volatility))

 # Open close to previous close
 if i == 0:
 open_price = close * (1 + np.random.normal(0, 0.001))
 else:
 open_price = close_prices[i-1] * (1 + np.random.normal(0, 0.005))

 # Volume
 volume = np.random.lognormal(10, 1) # Log-normal distribution

 data.append({
 'open': open_price,
 'high': high,
 'low': low,
 'close': close,
 'volume': volume
 })

 # Create DataFrame with datetime index
 dates = pd.date_range('2020-01-01', periods=n_points, freq='D')
 df = pd.DataFrame(data, index=dates)

 return df

def generate_returns_data(n_points: int = 1000) -> np.ndarray:
 """Generate returns data for testing"""
 np.random.seed(42)

 # Mix of normal returns with some outliers
 normal_returns = np.random.normal(0.001, 0.02, int(n_points * 0.95))
 outlier_returns = np.random.normal(0, 0.1, int(n_points * 0.05))

 all_returns = np.concatenate([normal_returns, outlier_returns])
 np.random.shuffle(all_returns)

 return all_returns[:n_points]

# Test utilities
def assert_array_almost_equal(actual, expected, decimal=7):
 """Assert arrays are almost equal with proper handling of NaN"""
 if isinstance(actual, (list, tuple)):
 actual = np.array(actual)
 if isinstance(expected, (list, tuple)):
 expected = np.array(expected)

 # Handle scalar case
 if np.isscalar(actual) and np.isscalar(expected):
 if np.isnan(actual) and np.isnan(expected):
 return True
 return abs(actual - expected) < 10**(-decimal)

 # Handle array case
 assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"

 # Check NaN positions match
 actual_nan = np.isnan(actual)
 expected_nan = np.isnan(expected)
 assert np.array_equal(actual_nan, expected_nan), "NaN positions don't match"

 # Check non-NaN values
 non_nan_mask = ~actual_nan
 if np.any(non_nan_mask):
 np.testing.assert_array_almost_equal(
 actual[non_nan_mask],
 expected[non_nan_mask],
 decimal=decimal
 )

def assert_performance_acceptable(func, args, max_time_ms=10.0):
 """Assert function executes within acceptable time"""
 import time

 start_time = time.time
 result = func(*args)
 end_time = time.time

 execution_time_ms = (end_time - start_time) * 1000
 assert execution_time_ms < max_time_ms, f"Function took {execution_time_ms:.2f}ms, expected < {max_time_ms}ms"

 return result

def assert_no_warnings(func, *args, **kwargs):
 """Assert function doesn't produce warnings"""
 with warnings.catch_warnings(record=True) as w:
 warnings.simplefilter("always")
 result = func(*args, **kwargs)
 assert len(w) == 0, f"Function produced {len(w)} warning(s): {[str(warning.message) for warning in w]}"
 return result

# Common test data
SAMPLE_PRICES = generate_price_data(100)
SAMPLE_OHLCV = generate_ohlcv_data(100)
SAMPLE_RETURNS = generate_returns_data(100)

# Performance test thresholds (in milliseconds)
PERFORMANCE_THRESHOLDS = {
 'technical_indicators': {
 'sma': 1.0,
 'ema': 1.5,
 'rsi': 2.0,
 'macd': 3.0,
 'bollinger_bands': 2.0,
 'atr': 2.0
 },
 'preprocessing': {
 'normalize_data': 5.0,
 'standardize_data': 3.0,
 'robust_scale': 4.0
 },
 'evaluation': {
 'calculate_sharpe_ratio': 1.0,
 'calculate_max_drawdown': 2.0,
 'backtest_strategy': 50.0
 },
 'math_utils': {
 'safe_divide': 0.1,
 'rolling_window': 2.0,
 'detect_outliers': 3.0
 }
}

__all__ = [
 "generate_price_data",
 "generate_ohlcv_data",
 "generate_returns_data",
 "assert_array_almost_equal",
 "assert_performance_acceptable",
 "assert_no_warnings",
 "SAMPLE_PRICES",
 "SAMPLE_OHLCV",
 "SAMPLE_RETURNS",
 "PERFORMANCE_THRESHOLDS"
]