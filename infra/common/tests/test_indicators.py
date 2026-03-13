"""
Test Suite for Technical Indicators Module

Comprehensive tests for all technical indicators with performance validation,
edge cases, and accuracy verification against known values.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

from ml_common.indicators import (
 calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
 calculate_bollinger_bands, calculate_atr, calculate_stochastic,
 TechnicalIndicators, IndicatorConfig,
 MADCResult, BollingerBandsResult, StochasticResult
)

from . import (
 assert_array_almost_equal, assert_performance_acceptable, assert_no_warnings,
 SAMPLE_PRICES, SAMPLE_OHLCV, PERFORMANCE_THRESHOLDS, generate_price_data
)


class TestTechnicalIndicatorsFunctions:
 """Test individual indicator calculation functions"""

 def test_calculate_sma_basic(self):
 """Test basic SMA calculation"""
 prices = [1, 2, 3, 4, 5]
 result = calculate_sma(prices, period=3)
 expected = (3 + 4 + 5) / 3 # Last 3 values
 assert abs(result - expected) < 1e-10

 def test_calculate_sma_insufficient_data(self):
 """Test SMA with insufficient data"""
 prices = [1, 2]
 result = calculate_sma(prices, period=5)
 assert np.isnan(result)

 def test_calculate_sma_empty_data(self):
 """Test SMA with empty data"""
 result = calculate_sma([], period=5)
 assert np.isnan(result)

 def test_calculate_sma_performance(self):
 """Test SMA performance"""
 prices = generate_price_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['technical_indicators']['sma']

 result = assert_performance_acceptable(
 calculate_sma, (prices, 20), threshold
 )
 assert not np.isnan(result)

 def test_calculate_ema_basic(self):
 """Test basic EMA calculation"""
 # Use volatile price sequence to ensure EMA != SMA
 prices = [10.0, 12.0, 9.0, 15.0, 8.0, 16.0, 7.0, 17.0, 6.0, 18.0]
 result = calculate_ema(prices, period=3)
 # EMA should be different from SMA with volatile data
 sma_result = calculate_sma(prices, period=3)
 assert abs(result - sma_result) > 0.01, f"EMA {result} should differ from SMA {sma_result}"

 def test_calculate_ema_single_value(self):
 """Test EMA with single value"""
 prices = [5.0]
 result = calculate_ema(prices, period=3)
 assert np.isnan(result) # Need at least 2 values

 def test_calculate_ema_performance(self):
 """Test EMA performance"""
 prices = generate_price_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['technical_indicators']['ema']

 result = assert_performance_acceptable(
 calculate_ema, (prices, 20), threshold
 )
 assert not np.isnan(result)

 def test_calculate_rsi_basic(self):
 """Test basic RSI calculation"""
 # Create data with clear up/down pattern
 prices = [10, 11, 12, 13, 14, 15, 14, 13, 12, 11, 10, 11, 12, 13, 14, 15]
 result = calculate_rsi(prices, period=14)

 # RSI should be between 0 and 100
 assert 0 <= result <= 100

 def test_calculate_rsi_all_up_trend(self):
 """Test RSI with only increasing prices"""
 prices = list(range(1, 16)) # 1, 2, 3, ..., 15
 result = calculate_rsi(prices, period=14)

 # Should be close to 100 (overbought)
 assert result > 80

 def test_calculate_rsi_all_down_trend(self):
 """Test RSI with only decreasing prices"""
 prices = list(range(15, 0, -1)) # 15, 14, 13, ..., 1
 result = calculate_rsi(prices, period=14)

 # Should be close to 0 (oversold)
 assert result < 20

 def test_calculate_rsi_insufficient_data(self):
 """Test RSI with insufficient data"""
 prices = [1, 2, 3]
 result = calculate_rsi(prices, period=14)
 assert result == 50.0 # Neutral RSI

 def test_calculate_rsi_performance(self):
 """Test RSI performance"""
 prices = generate_price_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['technical_indicators']['rsi']

 result = assert_performance_acceptable(
 calculate_rsi, (prices, 14), threshold
 )
 assert 0 <= result <= 100

 def test_calculate_macd_basic(self):
 """Test basic MACD calculation"""
 prices = generate_price_data(50)
 result = calculate_macd(prices)

 assert isinstance(result, MADCResult)
 assert hasattr(result, 'macd_line')
 assert hasattr(result, 'signal_line')
 assert hasattr(result, 'histogram')

 # Histogram should be macd_line - signal_line
 expected_histogram = result.macd_line - result.signal_line
 assert abs(result.histogram - expected_histogram) < 1e-10

 def test_calculate_macd_insufficient_data(self):
 """Test MACD with insufficient data"""
 prices = [1, 2, 3]
 result = calculate_macd(prices)

 assert result.macd_line == 0.0
 assert result.signal_line == 0.0
 assert result.histogram == 0.0

 def test_calculate_macd_performance(self):
 """Test MACD performance"""
 prices = generate_price_data(1000)
 threshold = PERFORMANCE_THRESHOLDS['technical_indicators']['macd']

 result = assert_performance_acceptable(
 calculate_macd, (prices,), threshold
 )
 assert isinstance(result, MADCResult)

 def test_calculate_bollinger_bands_basic(self):
 """Test basic Bollinger Bands calculation"""
 prices = generate_price_data(50)
 result = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

 assert isinstance(result, BollingerBandsResult)
 assert result.upper_band > result.middle_band > result.lower_band

 # Middle band should be SMA
 expected_middle = calculate_sma(prices, 20)
 assert abs(result.middle_band - expected_middle) < 1e-10

 def test_calculate_bollinger_bands_zero_std(self):
 """Test Bollinger Bands with constant prices (zero std)"""
 prices = [100.0] * 30
 result = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

 # All bands should be equal when std is zero
 assert abs(result.upper_band - result.middle_band) < 1e-10
 assert abs(result.middle_band - result.lower_band) < 1e-10

 def test_calculate_atr_basic(self):
 """Test basic ATR calculation"""
 high = [105, 108, 107, 110, 112]
 low = [95, 98, 97, 100, 102]
 close = [100, 103, 102, 105, 107]

 result = calculate_atr(high, low, close, period=3)
 assert result > 0 # ATR should be positive

 def test_calculate_atr_insufficient_data(self):
 """Test ATR with insufficient data"""
 high = [105, 108]
 low = [95, 98]
 close = [100, 103]

 result = calculate_atr(high, low, close, period=5)
 assert result == 0.0

 def test_calculate_stochastic_basic(self):
 """Test basic Stochastic calculation"""
 high = [110, 112, 115, 113, 118, 116, 120, 119, 121, 118, 115, 117, 119, 122, 120]
 low = [100, 102, 105, 103, 108, 106, 110, 109, 111, 108, 105, 107, 109, 112, 110]
 close = [105, 107, 110, 108, 113, 111, 115, 114, 116, 113, 110, 112, 114, 117, 115]

 result = calculate_stochastic(high, low, close, k_period=14)

 assert isinstance(result, StochasticResult)
 assert 0 <= result.percent_k <= 100
 assert 0 <= result.percent_d <= 100

 def test_calculate_stochastic_insufficient_data(self):
 """Test Stochastic with insufficient data"""
 high = [110, 112]
 low = [100, 102]
 close = [105, 107]

 result = calculate_stochastic(high, low, close, k_period=14)

 assert result.percent_k == 50.0
 assert result.percent_d == 50.0


class TestTechnicalIndicatorsClass:
 """Test TechnicalIndicators class"""

 def test_initialization_basic(self):
 """Test basic initialization"""
 indicators = ["sma_20", "ema_12", "rsi_14"]
 ti = TechnicalIndicators(indicators)

 assert ti.indicators == indicators
 assert isinstance(ti.config, IndicatorConfig)

 def test_initialization_with_config(self):
 """Test initialization with custom config"""
 config = IndicatorConfig(
 sma_periods=[10, 20, 50],
 rsi_period=21,
 use_cache=False
 )

 indicators = ["sma_20", "rsi_21"]
 ti = TechnicalIndicators(indicators, config)

 assert ti.config.rsi_period == 21
 assert ti.config.use_cache is False

 def test_calculate_single_indicator(self):
 """Test calculating single indicator"""
 ti = TechnicalIndicators(["sma_20"])
 prices = generate_price_data(50)

 result = ti.calculate(prices)

 assert "sma_20" in result
 assert isinstance(result["sma_20"], float)
 assert not np.isnan(result["sma_20"])

 def test_calculate_multiple_indicators(self):
 """Test calculating multiple indicators"""
 indicators = ["sma_20", "ema_12", "rsi_14"]
 ti = TechnicalIndicators(indicators)
 prices = generate_price_data(50)

 result = ti.calculate(prices)

 for indicator in indicators:
 assert indicator in result
 assert isinstance(result[indicator], float)

 def test_calculate_with_volume_data(self):
 """Test calculating with volume data"""
 ti = TechnicalIndicators(["sma_20", "obv"])
 data = generate_price_data(50)
 volumes = np.random.uniform(1000, 10000, 50)

 result = ti.calculate(data, volumes=volumes)

 assert "sma_20" in result
 # Note: OBV might not be available in basic implementation

 def test_calculate_insufficient_data(self):
 """Test calculation with insufficient data"""
 ti = TechnicalIndicators(["sma_20", "rsi_14"])
 prices = [100.0] # Only one price

 result = ti.calculate(prices)

 # Should return zero values for insufficient data
 for indicator in ti.indicators:
 assert result[indicator] == 0.0

 def test_calculate_empty_data(self):
 """Test calculation with empty data"""
 ti = TechnicalIndicators(["sma_20"])
 prices = []

 result = ti.calculate(prices)

 assert result["sma_20"] == 0.0

 def test_calculate_performance_single(self):
 """Test calculation performance for single indicator"""
 ti = TechnicalIndicators(["sma_20"])
 prices = generate_price_data(1000)

 result = assert_performance_acceptable(
 ti.calculate, (prices,), 5.0 # 5ms threshold for single indicator
 )

 assert "sma_20" in result

 def test_calculate_performance_multiple(self):
 """Test calculation performance for multiple indicators"""
 indicators = ["sma_20", "ema_12", "rsi_14", "macd"]
 ti = TechnicalIndicators(indicators)
 prices = generate_price_data(1000)

 result = assert_performance_acceptable(
 ti.calculate, (prices,), 15.0 # 15ms threshold for multiple indicators
 )

 for indicator in indicators:
 assert indicator in result

 def test_caching_functionality(self):
 """Test caching functionality"""
 config = IndicatorConfig(use_cache=True)
 ti = TechnicalIndicators(["sma_20"], config)
 prices = generate_price_data(50)

 # First calculation
 result1 = ti.calculate(prices)

 # Second calculation (should use cache)
 result2 = ti.calculate(prices)

 assert result1["sma_20"] == result2["sma_20"]

 # Check cache statistics
 stats = ti.get_performance_stats
 assert stats["cache_enabled"] is True
 assert stats["cache_hit_rate"] > 0

 def test_cache_disabled(self):
 """Test with caching disabled"""
 config = IndicatorConfig(use_cache=False)
 ti = TechnicalIndicators(["sma_20"], config)

 stats = ti.get_performance_stats
 assert stats["cache_enabled"] is False

 def test_unknown_indicator(self):
 """Test with unknown indicator"""
 ti = TechnicalIndicators(["unknown_indicator"])
 prices = generate_price_data(50)

 result = ti.calculate(prices)

 assert result["unknown_indicator"] == 0.0

 def test_get_performance_stats(self):
 """Test performance statistics"""
 ti = TechnicalIndicators(["sma_20"])
 prices = generate_price_data(50)

 # Make some calculations
 ti.calculate(prices)
 ti.calculate(prices)

 stats = ti.get_performance_stats

 assert "call_count" in stats
 assert stats["call_count"] == 2
 assert "has_numba" in stats
 assert "has_talib" in stats

 def test_clear_cache(self):
 """Test cache clearing"""
 config = IndicatorConfig(use_cache=True)
 ti = TechnicalIndicators(["sma_20"], config)
 prices = generate_price_data(50)

 # Make calculation to populate cache
 ti.calculate(prices)

 # Clear cache
 ti.clear_cache

 stats = ti.get_performance_stats
 assert stats["cache_size"] == 0
 assert stats["cache_hit_rate"] == 0

 def test_reset_state(self):
 """Test state reset"""
 ti = TechnicalIndicators(["sma_20"])
 prices = generate_price_data(50)

 # Make calculations
 ti.calculate(prices)
 ti.calculate(prices)

 # Reset state
 ti.reset

 stats = ti.get_performance_stats
 assert stats["call_count"] == 0


class TestIndicatorEdgeCases:
 """Test edge cases and error conditions"""

 def test_nan_input_handling(self):
 """Test handling of NaN inputs"""
 prices = [1.0, 2.0, np.nan, 4.0, 5.0]

 # Functions should handle NaN gracefully
 result = calculate_sma(prices, 3)
 # Result might be NaN or calculated ignoring NaN

 def test_infinite_input_handling(self):
 """Test handling of infinite inputs"""
 prices = [1.0, 2.0, np.inf, 4.0, 5.0]

 # Should not crash
 result = calculate_sma(prices, 3)

 def test_negative_prices(self):
 """Test with negative prices"""
 prices = [-1.0, -2.0, -3.0, -4.0, -5.0]

 result = calculate_sma(prices, 3)
 assert not np.isnan(result)

 def test_zero_prices(self):
 """Test with zero prices"""
 prices = [0.0, 0.0, 0.0, 0.0, 0.0]

 result = calculate_sma(prices, 3)
 assert result == 0.0

 def test_very_large_numbers(self):
 """Test with very large numbers"""
 prices = [1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4]

 result = calculate_sma(prices, 3)
 assert not np.isnan(result)
 assert not np.isinf(result)

 def test_very_small_numbers(self):
 """Test with very small numbers"""
 prices = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]

 result = calculate_sma(prices, 3)
 assert not np.isnan(result)
 assert result > 0

 def test_constant_prices_rsi(self):
 """Test RSI with constant prices"""
 prices = [100.0] * 20

 result = calculate_rsi(prices, 14)
 # RSI should be neutral (50) for constant prices
 assert abs(result - 50.0) < 1e-10


class TestIndicatorAccuracy:
 """Test accuracy against known values"""

 def test_sma_known_values(self):
 """Test SMA against manually calculated values"""
 prices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

 # SMA(5) for last 5 values: (6+7+8+9+10)/5 = 8
 result = calculate_sma(prices, 5)
 assert abs(result - 8.0) < 1e-10

 def test_ema_known_values(self):
 """Test EMA convergence properties"""
 # EMA should give more weight to recent values
 prices1 = [1, 2, 3, 4, 10] # Recent high value
 prices2 = [1, 2, 3, 4, 5] # Gradual increase

 ema1 = calculate_ema(prices1, 3)
 ema2 = calculate_ema(prices2, 3)

 # EMA1 should be higher due to recent high value
 assert ema1 > ema2

 def test_bollinger_bands_properties(self):
 """Test Bollinger Bands mathematical properties"""
 prices = generate_price_data(50)

 result = calculate_bollinger_bands(prices, period=20, std_dev=2.0)

 # Mathematical properties
 assert result.upper_band > result.middle_band
 assert result.middle_band > result.lower_band

 # Middle band should equal SMA
 sma_result = calculate_sma(prices, 20)
 assert abs(result.middle_band - sma_result) < 1e-10

 # Band width should be proportional to volatility
 std_dev = np.std(prices[-20:])
 expected_width = std_dev * 2.0 * 2 # 2 std dev on each side
 actual_width = result.upper_band - result.lower_band

 # Should be reasonably close (allowing for calculation differences)
 assert abs(actual_width - expected_width) / expected_width < 0.1


if __name__ == "__main__":
 pytest.main([__file__, "-v"])