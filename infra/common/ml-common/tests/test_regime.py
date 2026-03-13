"""
Test Suite for Regime Classification Module
Testing Patterns

Comprehensive tests for regime classification:
- Volatility regime classification (4 dims)
- Trend regime classification (4 dims)
- Time features (2 dims)
- Performance validation (<2ms)
- Edge cases (extreme volatility, sideways trend, weekend/weekday)
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List

from regime import (
 # Volatility
 classify_volatility_regime,
 calculate_volatility_percentile,
 calculate_regime_duration,
 calculate_regime_stability,
 extract_volatility_features,
 # Trend
 classify_trend_regime,
 calculate_trend_strength,
 calculate_trend_duration,
 calculate_trend_acceleration,
 extract_trend_features,
 # Time
 classify_trading_session,
 normalize_day_of_week,
 extract_time_features,
)

from . import (
 assert_performance_acceptable,
 generate_price_data,
)


def generate_volatile_prices(n_points: int = 168, volatility_level: str = 'medium') -> np.ndarray:
 """Generate price data with specific volatility level"""
 np.random.seed(42)

 if volatility_level == 'low':
 std = 0.005 # 0.5% daily vol
 elif volatility_level == 'medium':
 std = 0.020 # 2% daily vol
 elif volatility_level == 'high':
 std = 0.040 # 4% daily vol
 elif volatility_level == 'extreme':
 std = 0.080 # 8% daily vol
 else:
 std = 0.020

 returns = np.random.normal(0.0, std, n_points)
 prices = np.zeros(n_points)
 prices[0] = 50000.0

 for i in range(1, n_points):
 prices[i] = prices[i-1] * (1 + returns[i])

 return prices


def generate_trending_prices(n_points: int = 168, trend: str = 'up') -> np.ndarray:
 """Generate price data with specific trend"""
 np.random.seed(42)

 if trend == 'strong_up':
 drift = 0.002 # +0.2% per hour
 noise = 0.005
 elif trend == 'weak_up':
 drift = 0.0005 # +0.05% per hour
 noise = 0.01
 elif trend == 'sideways':
 drift = 0.0
 noise = 0.01
 elif trend == 'weak_down':
 drift = -0.0005
 noise = 0.01
 elif trend == 'strong_down':
 drift = -0.002
 noise = 0.005
 else:
 drift = 0.0
 noise = 0.01

 prices = np.zeros(n_points)
 prices[0] = 50000.0

 for i in range(1, n_points):
 returns = drift + np.random.normal(0, noise)
 prices[i] = prices[i-1] * (1 + returns)

 return prices


class TestVolatilityRegime:
 """Test volatility regime classification"""

 def test_classify_volatility_regime_low(self):
 """Test low volatility regime"""
 prices = generate_volatile_prices(168, 'low')

 regime = classify_volatility_regime(prices, window=24)

 # Should be 0 (Low) or 1 (Medium)
 assert regime in [0, 1], f"Expected low/medium regime, got {regime}"

 def test_classify_volatility_regime_medium(self):
 """Test medium volatility regime"""
 prices = generate_volatile_prices(168, 'medium')

 regime = classify_volatility_regime(prices, window=24)

 # Should be 2 (Medium-High) or 3 (High) for 2% daily volatility
 assert regime in [2, 3], f"Expected medium-high regime, got {regime}"

 def test_classify_volatility_regime_high(self):
 """Test high volatility regime"""
 prices = generate_volatile_prices(168, 'high')

 regime = classify_volatility_regime(prices, window=24)

 # Should be 2 (High) or 3 (Extreme)
 assert regime in [2, 3], f"Expected high regime, got {regime}"

 def test_classify_volatility_regime_extreme(self):
 """Test extreme volatility regime"""
 prices = generate_volatile_prices(168, 'extreme')

 regime = classify_volatility_regime(prices, window=24)

 # Should be 3 (Extreme)
 assert regime == 3, f"Expected extreme regime, got {regime}"

 def test_calculate_volatility_percentile(self):
 """Test volatility percentile calculation"""
 prices = generate_volatile_prices(200)

 percentile = calculate_volatility_percentile(prices, window=24, lookback=168)

 assert 0.0 <= percentile <= 100.0, f"Percentile should be 0-100, got {percentile}"
 assert isinstance(percentile, float)

 def test_calculate_regime_duration(self):
 """Test regime duration calculation"""
 prices = generate_volatile_prices(168, 'medium')

 duration = calculate_regime_duration(prices, window=24)

 # Duration should be 0-168 hours
 assert 0 <= duration <= 168, f"Duration should be 0-168, got {duration}"
 assert isinstance(duration, int)

 def test_calculate_regime_stability(self):
 """Test regime stability calculation"""
 prices = generate_volatile_prices(100, 'medium')

 stability = calculate_regime_stability(prices, window=24)

 # Stability should be 0-1
 assert 0.0 <= stability <= 1.0, f"Stability should be 0-1, got {stability}"
 assert isinstance(stability, float)

 def test_extract_volatility_features(self):
 """Test full volatility feature extraction"""
 prices = generate_volatile_prices(168)

 features = extract_volatility_features(prices, window=24)

 # Should return 4 dimensions
 assert isinstance(features, np.ndarray)
 assert features.shape == (4,), f"Expected shape (4,), got {features.shape}"

 # All should be finite
 assert np.all(np.isfinite(features)), "All features should be finite"

 # Validate ranges
 assert 0 <= features[0] <= 3, "Regime ID should be 0-3"
 assert 0 <= features[1] <= 100, "Percentile should be 0-100"
 assert 0 <= features[2] <= 168, "Duration should be 0-168"
 assert 0 <= features[3] <= 1, "Stability should be 0-1"

 def test_volatility_features_performance(self):
 """Test volatility extraction performance"""
 prices = generate_volatile_prices(168)

 result = assert_performance_acceptable(
 extract_volatility_features,
 (prices, 24),
 max_time_ms=2.0 # <2ms for all 4 features
 )

 assert result.shape == (4,)


class TestTrendRegime:
 """Test trend regime classification"""

 def test_classify_trend_regime_strong_up(self):
 """Test strong uptrend classification"""
 prices = generate_trending_prices(168, 'strong_up')

 regime = classify_trend_regime(prices, window=24)

 # Should be +2 (Strong Up) or +1 (Weak Up)
 assert regime > 0, f"Expected positive trend, got {regime}"

 def test_classify_trend_regime_weak_up(self):
 """Test weak uptrend classification"""
 prices = generate_trending_prices(168, 'weak_up')

 regime = classify_trend_regime(prices, window=24)

 # Should be +1 (Weak Up) or 0 (Sideways)
 assert regime >= 0, f"Expected non-negative trend, got {regime}"

 def test_classify_trend_regime_sideways(self):
 """Test sideways trend classification"""
 prices = generate_trending_prices(168, 'sideways')

 regime = classify_trend_regime(prices, window=24)

 # Should be 0 (Sideways)
 assert regime in [-1, 0, 1], f"Expected sideways trend, got {regime}"

 def test_classify_trend_regime_weak_down(self):
 """Test weak downtrend classification"""
 prices = generate_trending_prices(168, 'weak_down')

 regime = classify_trend_regime(prices, window=24)

 # Should be -1 (Weak Down) or 0 (Sideways)
 assert regime <= 0, f"Expected non-positive trend, got {regime}"

 def test_classify_trend_regime_strong_down(self):
 """Test strong downtrend classification"""
 prices = generate_trending_prices(168, 'strong_down')

 regime = classify_trend_regime(prices, window=24)

 # Should be -2 (Strong Down) or -1 (Weak Down)
 assert regime < 0, f"Expected negative trend, got {regime}"

 def test_calculate_trend_strength(self):
 """Test trend strength calculation"""
 prices = generate_trending_prices(168, 'strong_up')

 strength = calculate_trend_strength(prices, window=24)

 # Strength should be 0-1 (R²)
 assert 0.0 <= strength <= 1.0, f"Strength should be 0-1, got {strength}"
 assert isinstance(strength, float)

 def test_calculate_trend_duration(self):
 """Test trend duration calculation"""
 prices = generate_trending_prices(168, 'weak_up')

 duration = calculate_trend_duration(prices, window=24)

 # Duration should be 0-72 hours (max 3 days lookback)
 assert 0 <= duration <= 72, f"Duration should be 0-72, got {duration}"
 assert isinstance(duration, int)

 def test_calculate_trend_acceleration(self):
 """Test trend acceleration calculation"""
 prices = generate_trending_prices(168, 'strong_up')

 acceleration = calculate_trend_acceleration(prices, short_window=12, long_window=24)

 # Acceleration can be positive or negative
 assert isinstance(acceleration, float)
 assert np.isfinite(acceleration)

 def test_extract_trend_features(self):
 """Test full trend feature extraction"""
 prices = generate_trending_prices(168)

 features = extract_trend_features(prices, window=24)

 # Should return 4 dimensions
 assert isinstance(features, np.ndarray)
 assert features.shape == (4,), f"Expected shape (4,), got {features.shape}"

 # All should be finite
 assert np.all(np.isfinite(features)), "All features should be finite"

 # Validate ranges
 assert -2 <= features[0] <= 2, "Trend ID should be -2 to +2"
 assert 0 <= features[1] <= 1, "Strength should be 0-1"
 assert 0 <= features[2] <= 72, "Duration should be 0-72"

 def test_trend_features_performance(self):
 """Test trend extraction performance"""
 prices = generate_trending_prices(168)

 result = assert_performance_acceptable(
 extract_trend_features,
 (prices, 24),
 max_time_ms=1.0 # <1ms for all 4 features
 )

 assert result.shape == (4,)


class TestTimeFeatures:
 """Test time-based features"""

 def test_classify_trading_session_asian(self):
 """Test Asian session classification"""
 timestamp = datetime(2025, 10, 10, 4, 0, 0, tzinfo=timezone.utc) # 04:00 UTC

 session = classify_trading_session(timestamp)

 assert session == 0, f"Expected Asian session (0), got {session}"

 def test_classify_trading_session_european(self):
 """Test European session classification"""
 timestamp = datetime(2025, 10, 10, 12, 0, 0, tzinfo=timezone.utc) # 12:00 UTC

 session = classify_trading_session(timestamp)

 assert session == 1, f"Expected European session (1), got {session}"

 def test_classify_trading_session_us(self):
 """Test US session classification"""
 timestamp = datetime(2025, 10, 10, 18, 0, 0, tzinfo=timezone.utc) # 18:00 UTC

 session = classify_trading_session(timestamp)

 assert session == 2, f"Expected US session (2), got {session}"

 def test_classify_trading_session_overnight(self):
 """Test Overnight session classification"""
 timestamp = datetime(2025, 10, 10, 23, 0, 0, tzinfo=timezone.utc) # 23:00 UTC

 session = classify_trading_session(timestamp)

 assert session == 3, f"Expected Overnight session (3), got {session}"

 def test_normalize_day_of_week_monday(self):
 """Test Monday normalization"""
 timestamp = datetime(2025, 10, 6, 12, 0, 0, tzinfo=timezone.utc) # Monday

 normalized = normalize_day_of_week(timestamp)

 assert normalized == 0.0, f"Expected 0.0 for Monday, got {normalized}"

 def test_normalize_day_of_week_friday(self):
 """Test Friday normalization"""
 timestamp = datetime(2025, 10, 10, 12, 0, 0, tzinfo=timezone.utc) # Friday

 normalized = normalize_day_of_week(timestamp)

 expected = 4 / 7.0 # Friday is day 4
 assert abs(normalized - expected) < 0.01, f"Expected {expected}, got {normalized}"

 def test_normalize_day_of_week_sunday(self):
 """Test Sunday normalization"""
 timestamp = datetime(2025, 10, 12, 12, 0, 0, tzinfo=timezone.utc) # Sunday

 normalized = normalize_day_of_week(timestamp)

 expected = 6 / 7.0 # Sunday is day 6
 assert abs(normalized - expected) < 0.01, f"Expected {expected}, got {normalized}"

 def test_extract_time_features(self):
 """Test full time feature extraction"""
 timestamp = datetime(2025, 10, 10, 15, 30, 0, tzinfo=timezone.utc) # Friday 15:30 UTC

 features = extract_time_features(timestamp)

 # Should return 2 dimensions
 assert isinstance(features, np.ndarray)
 assert features.shape == (2,), f"Expected shape (2,), got {features.shape}"

 # All should be finite
 assert np.all(np.isfinite(features)), "All features should be finite"

 # Validate ranges
 assert 0 <= features[0] <= 3, "Session ID should be 0-3"
 assert 0 <= features[1] <= 1, "Day of week should be 0-1"

 def test_time_features_performance(self):
 """Test time extraction performance"""
 timestamp = datetime.now(timezone.utc)

 result = assert_performance_acceptable(
 extract_time_features,
 (timestamp,),
 max_time_ms=0.1 # <0.1ms for 2 features
 )

 assert result.shape == (2,)


class TestRegimeEdgeCases:
 """Test edge cases and error handling"""

 def test_constant_prices(self):
 """Test with constant prices (no volatility)"""
 prices = np.ones(168) * 50000.0

 # Volatility features
 vol_features = extract_volatility_features(prices)
 assert vol_features.shape == (4,)
 assert vol_features[0] == 0 # Low volatility regime

 # Trend features
 trend_features = extract_trend_features(prices)
 assert trend_features.shape == (4,)
 assert trend_features[0] == 0 # Sideways trend

 def test_insufficient_data_volatility(self):
 """Test volatility with insufficient data"""
 prices = np.array([50000.0, 50100.0]) # Only 2 points

 regime = classify_volatility_regime(prices, window=24)

 # Should return default (medium)
 assert regime == 1

 def test_insufficient_data_trend(self):
 """Test trend with insufficient data"""
 prices = np.array([50000.0, 50100.0])

 regime = classify_trend_regime(prices, window=24)

 # Should return default (sideways)
 assert regime == 0

 def test_extreme_volatility_spike(self):
 """Test with extreme volatility spike"""
 prices = generate_volatile_prices(168, 'low')
 # Add extreme spike at end
 prices[-1] = prices[-2] * 1.20 # +20% spike

 regime = classify_volatility_regime(prices, window=24)

 # Should detect high/extreme volatility
 assert regime >= 2, f"Expected high volatility, got regime {regime}"

 def test_weekend_classification(self):
 """Test weekend vs weekday classification"""
 # Saturday
 saturday = datetime(2025, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
 features_sat = extract_time_features(saturday)

 # Sunday
 sunday = datetime(2025, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
 features_sun = extract_time_features(sunday)

 # Weekday (Monday)
 monday = datetime(2025, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
 features_mon = extract_time_features(monday)

 # Weekend should have higher day_of_week values
 assert features_sat[1] > features_mon[1]
 assert features_sun[1] > features_mon[1]


class TestRegimeIntegration:
 """Integration tests for complete regime feature extraction"""

 def test_extract_all_regime_features(self):
 """Test extracting all 10 regime features"""
 prices = generate_price_data(168)
 timestamp = datetime.now(timezone.utc)

 # Extract all regime features
 vol_features = extract_volatility_features(prices, window=24)
 trend_features = extract_trend_features(prices, window=24)
 time_features = extract_time_features(timestamp)

 # Combine
 all_features = np.concatenate([vol_features, trend_features, time_features])

 # Should have 10 dimensions (4 + 4 + 2)
 assert all_features.shape == (10,), f"Expected 10 dims, got {all_features.shape}"

 # All should be finite
 assert np.all(np.isfinite(all_features)), "All features should be finite"

 def test_regime_consistency(self):
 """Test regime extraction consistency"""
 prices = generate_price_data(168)

 # Extract twice
 features1 = extract_volatility_features(prices, window=24)
 features2 = extract_volatility_features(prices, window=24)

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_performance_full_extraction(self):
 """Test performance of full regime extraction"""
 prices = generate_price_data(168)
 timestamp = datetime.now(timezone.utc)

 def extract_all:
 vol = extract_volatility_features(prices, window=24)
 trend = extract_trend_features(prices, window=24)
 time = extract_time_features(timestamp)
 return np.concatenate([vol, trend, time])

 result = assert_performance_acceptable(
 extract_all,
 max_time_ms=2.0 # <2ms for all 10 features
 )

 assert result.shape == (10,)


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
