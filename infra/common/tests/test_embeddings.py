"""
Test Suite for Embeddings Module
Testing Patterns

Comprehensive tests for embedding feature extraction:
- Symbol embeddings (16 dims = 4 symbols × 4-dim)
- Temporal embeddings (10 dims: hour/day/week/month sin/cos)
- Cyclic encoding correctness
- Performance validation (<0.5ms)
- Edge cases (midnight, New Year, weekend)
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List

from embeddings import (
 # Symbol
 extract_symbol_embeddings,
 get_symbol_embedding,
 initialize_symbol_embeddings,
 # Temporal
 extract_temporal_embeddings,
 encode_hour_of_day,
 encode_day_of_week,
 encode_week_of_year,
 encode_month_of_year,
)

from . import (
 assert_performance_acceptable,
)


@pytest.fixture(autouse=True)
def reset_symbol_embeddings:
 """Reset symbol embeddings to default 4-dim before each test"""
 from embeddings.symbol import DEFAULT_SYMBOLS, EMBEDDING_DIM
 initialize_symbol_embeddings(DEFAULT_SYMBOLS, EMBEDDING_DIM, seed=42)
 yield
 # Reset again after test to avoid side effects
 initialize_symbol_embeddings(DEFAULT_SYMBOLS, EMBEDDING_DIM, seed=42)


class TestSymbolEmbeddings:
 """Test symbol embedding extraction"""

 def test_extract_symbol_embeddings_basic(self):
 """Test basic symbol embedding extraction"""
 embeddings = extract_symbol_embeddings

 # Should return 16 dimensions (4 symbols × 4-dim)
 assert isinstance(embeddings, np.ndarray)
 assert embeddings.shape == (16,), f"Expected shape (16,), got {embeddings.shape}"

 # All should be finite
 assert np.all(np.isfinite(embeddings)), "All embeddings should be finite"

 def test_extract_symbol_embeddings_custom_symbols(self):
 """Test with custom symbol list"""
 symbols = ['BTCUSDT', 'ETHUSDT']

 embeddings = extract_symbol_embeddings(symbols=symbols)

 # Should return 8 dimensions (2 symbols × 4-dim)
 assert embeddings.shape == (8,)

 def test_get_symbol_embedding_btc(self):
 """Test getting BTC embedding"""
 embedding = get_symbol_embedding('BTCUSDT')

 # Should return 4-dimensional vector
 assert isinstance(embedding, np.ndarray)
 assert embedding.shape == (4,)
 assert np.all(np.isfinite(embedding))

 def test_get_symbol_embedding_all_symbols(self):
 """Test getting embeddings for all symbols"""
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 for symbol in symbols:
 embedding = get_symbol_embedding(symbol)
 assert embedding.shape == (4,)
 assert np.all(np.isfinite(embedding))

 def test_initialize_symbol_embeddings(self):
 """Test embedding initialization"""
 symbols = ['BTCUSDT', 'ETHUSDT']

 embeddings = initialize_symbol_embeddings(symbols=symbols, embedding_dim=4, seed=42)

 # Should return dict with 2 symbols
 assert len(embeddings) == 2
 assert 'BTCUSDT' in embeddings
 assert 'ETHUSDT' in embeddings

 # Each embedding should be 4-dimensional
 for symbol, emb in embeddings.items:
 assert emb.shape == (4,)
 assert np.all(np.isfinite(emb))

 def test_symbol_embeddings_reproducibility(self):
 """Test that embeddings are reproducible with same seed"""
 # Initialize with seed 42
 embeddings1 = initialize_symbol_embeddings(seed=42)

 # Initialize again with seed 42
 embeddings2 = initialize_symbol_embeddings(seed=42)

 # Should be identical
 for symbol in embeddings1.keys:
 np.testing.assert_array_equal(embeddings1[symbol], embeddings2[symbol])

 def test_symbol_embeddings_different_seeds(self):
 """Test that different seeds produce different embeddings"""
 embeddings1 = initialize_symbol_embeddings(seed=42)
 embeddings2 = initialize_symbol_embeddings(seed=123)

 # Should be different
 for symbol in embeddings1.keys:
 assert not np.array_equal(embeddings1[symbol], embeddings2[symbol])

 def test_symbol_embeddings_dimension(self):
 """Test custom embedding dimensions"""
 embeddings = initialize_symbol_embeddings(embedding_dim=8, seed=42)

 for symbol, emb in embeddings.items:
 assert emb.shape == (8,), f"Expected dim 8, got {emb.shape}"

 def test_symbol_embeddings_performance(self):
 """Test symbol embedding extraction performance"""
 result = assert_performance_acceptable(
 extract_symbol_embeddings,
 max_time_ms=0.1 # <0.1ms for 16 dims
 )

 assert result.shape == (16,)


class TestTemporalEmbeddings:
 """Test temporal embedding extraction"""

 def test_extract_temporal_embeddings_basic(self):
 """Test basic temporal embedding extraction"""
 timestamp = datetime(2025, 10, 10, 15, 30, 0, tzinfo=timezone.utc)

 embeddings = extract_temporal_embeddings(timestamp)

 # Should return 10 dimensions
 assert isinstance(embeddings, np.ndarray)
 assert embeddings.shape == (10,), f"Expected shape (10,), got {embeddings.shape}"

 # All should be finite
 assert np.all(np.isfinite(embeddings)), "All embeddings should be finite"

 def test_encode_hour_of_day_midnight(self):
 """Test hour encoding at midnight"""
 timestamp = datetime(2025, 10, 10, 0, 0, 0, tzinfo=timezone.utc)

 sin_val, cos_val = encode_hour_of_day(timestamp)

 # At 0:00, angle = 0, so sin=0, cos=1
 assert abs(sin_val - 0.0) < 0.01, f"Expected sin ≈ 0, got {sin_val}"
 assert abs(cos_val - 1.0) < 0.01, f"Expected cos ≈ 1, got {cos_val}"

 def test_encode_hour_of_day_noon(self):
 """Test hour encoding at noon"""
 timestamp = datetime(2025, 10, 10, 12, 0, 0, tzinfo=timezone.utc)

 sin_val, cos_val = encode_hour_of_day(timestamp)

 # At 12:00, angle = π, so sin≈0, cos≈-1
 assert abs(sin_val - 0.0) < 0.01, f"Expected sin ≈ 0, got {sin_val}"
 assert abs(cos_val - (-1.0)) < 0.01, f"Expected cos ≈ -1, got {cos_val}"

 def test_encode_hour_of_day_6am(self):
 """Test hour encoding at 6:00 AM"""
 timestamp = datetime(2025, 10, 10, 6, 0, 0, tzinfo=timezone.utc)

 sin_val, cos_val = encode_hour_of_day(timestamp)

 # At 6:00, angle = π/2, so sin≈1, cos≈0
 assert abs(sin_val - 1.0) < 0.01, f"Expected sin ≈ 1, got {sin_val}"
 assert abs(cos_val - 0.0) < 0.01, f"Expected cos ≈ 0, got {cos_val}"

 def test_encode_hour_of_day_continuity(self):
 """Test that hour encoding is continuous (23:59 → 00:00)"""
 timestamp1 = datetime(2025, 10, 10, 23, 59, 0, tzinfo=timezone.utc)
 timestamp2 = datetime(2025, 10, 11, 0, 1, 0, tzinfo=timezone.utc)

 sin1, cos1 = encode_hour_of_day(timestamp1)
 sin2, cos2 = encode_hour_of_day(timestamp2)

 # Should be close (cyclic continuity)
 distance = np.sqrt((sin1 - sin2)**2 + (cos1 - cos2)**2)
 assert distance < 0.1, f"Hour encoding not continuous: distance={distance}"

 def test_encode_day_of_week_monday(self):
 """Test day of week encoding for Monday"""
 timestamp = datetime(2025, 10, 6, 12, 0, 0, tzinfo=timezone.utc) # Monday

 sin_val, cos_val = encode_day_of_week(timestamp)

 # Monday = 0, angle = 0, so sin=0, cos=1
 assert abs(sin_val - 0.0) < 0.01
 assert abs(cos_val - 1.0) < 0.01

 def test_encode_day_of_week_thursday(self):
 """Test day of week encoding for Thursday"""
 timestamp = datetime(2025, 10, 9, 12, 0, 0, tzinfo=timezone.utc) # Thursday

 sin_val, cos_val = encode_day_of_week(timestamp)

 # Thursday = 3, angle ≈ 2.7 rad
 # Both sin and cos should be non-zero
 assert abs(sin_val) > 0.1
 assert abs(cos_val) > 0.1

 def test_encode_day_of_week_continuity(self):
 """Test day of week continuity (Sunday → Monday)"""
 sunday = datetime(2025, 10, 12, 12, 0, 0, tzinfo=timezone.utc) # Sunday
 monday = datetime(2025, 10, 13, 12, 0, 0, tzinfo=timezone.utc) # Monday

 sin_sun, cos_sun = encode_day_of_week(sunday)
 sin_mon, cos_mon = encode_day_of_week(monday)

 # Should be close (cyclic)
 distance = np.sqrt((sin_sun - sin_mon)**2 + (cos_sun - cos_mon)**2)
 assert distance < 1.0, f"Day encoding not continuous: distance={distance}"

 def test_encode_week_of_year_week1(self):
 """Test week of year encoding for week 1"""
 timestamp = datetime(2025, 1, 3, 12, 0, 0, tzinfo=timezone.utc) # Week 1 (Jan 1-5)

 sin_val, cos_val = encode_week_of_year(timestamp)

 # Week 1, small angle (week 1/53)
 assert abs(sin_val) < 0.13 # sin(2π*1/53) ≈ 0.118
 assert cos_val > 0.99 # cos(2π*1/53) ≈ 0.993

 def test_encode_week_of_year_mid_year(self):
 """Test week of year encoding for mid-year"""
 timestamp = datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc) # Week 14 (mid-year)

 sin_val, cos_val = encode_week_of_year(timestamp)

 # Week 14 out of 53 ≈ quarter way around circle
 # angle ≈ π/2, so sin≈1, cos≈0
 assert abs(sin_val) > 0.99 # sin(2π*14/53) ≈ 0.996
 assert abs(cos_val) < 0.1 # cos(2π*14/53) ≈ -0.089

 def test_encode_month_of_year_january(self):
 """Test month encoding for January"""
 timestamp = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

 month = encode_month_of_year(timestamp)

 # January = 1 → normalized to 0.0
 assert abs(month - 0.0) < 0.01

 def test_encode_month_of_year_december(self):
 """Test month encoding for December"""
 timestamp = datetime(2025, 12, 15, 12, 0, 0, tzinfo=timezone.utc)

 month = encode_month_of_year(timestamp)

 # December = 12 → normalized to 1.0
 assert abs(month - 1.0) < 0.01

 def test_encode_month_of_year_june(self):
 """Test month encoding for June"""
 timestamp = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

 month = encode_month_of_year(timestamp)

 # June = 6 → normalized to 5/11 ≈ 0.45
 expected = (6 - 1) / 11.0
 assert abs(month - expected) < 0.01

 def test_temporal_embeddings_structure(self):
 """Test temporal embeddings structure"""
 timestamp = datetime(2025, 10, 10, 15, 30, 0, tzinfo=timezone.utc)

 features = extract_temporal_embeddings(timestamp)

 # [0-1]: Hour sin/cos
 assert abs(features[0]**2 + features[1]**2 - 1.0) < 0.01, "Hour encoding not on unit circle"

 # [2-3]: Day sin/cos
 assert abs(features[2]**2 + features[3]**2 - 1.0) < 0.01, "Day encoding not on unit circle"

 # [4-5]: Week sin/cos
 assert abs(features[4]**2 + features[5]**2 - 1.0) < 0.01, "Week encoding not on unit circle"

 # [6]: Month (0-1)
 assert 0 <= features[6] <= 1, "Month should be 0-1"

 # [7]: Quarter (0, 0.33, 0.67, 1.0)
 assert features[7] in [0.0, 1/3, 2/3, 1.0] or abs(features[7] - 1/3) < 0.01 or abs(features[7] - 2/3) < 0.01

 # [8]: Year fraction (0-1)
 assert 0 <= features[8] <= 1, "Year fraction should be 0-1"

 # [9]: Is weekend (0 or 1)
 assert features[9] in [0.0, 1.0], "Is weekend should be 0 or 1"

 def test_temporal_embeddings_weekend(self):
 """Test weekend detection"""
 # Saturday
 saturday = datetime(2025, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
 features_sat = extract_temporal_embeddings(saturday)

 # Sunday
 sunday = datetime(2025, 10, 12, 12, 0, 0, tzinfo=timezone.utc)
 features_sun = extract_temporal_embeddings(sunday)

 # Weekday (Monday)
 monday = datetime(2025, 10, 6, 12, 0, 0, tzinfo=timezone.utc)
 features_mon = extract_temporal_embeddings(monday)

 # Weekend should be 1.0
 assert features_sat[9] == 1.0, "Saturday should be weekend"
 assert features_sun[9] == 1.0, "Sunday should be weekend"

 # Weekday should be 0.0
 assert features_mon[9] == 0.0, "Monday should not be weekend"

 def test_temporal_embeddings_performance(self):
 """Test temporal embedding extraction performance"""
 timestamp = datetime.now(timezone.utc)

 result = assert_performance_acceptable(
 extract_temporal_embeddings,
 (timestamp,),
 max_time_ms=0.5 # <0.5ms for 10 dims
 )

 assert result.shape == (10,)


class TestEmbeddingsEdgeCases:
 """Test edge cases and error handling"""

 def test_midnight_encoding(self):
 """Test encoding at midnight (boundary)"""
 timestamp = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

 features = extract_temporal_embeddings(timestamp)

 # Should not crash
 assert features.shape == (10,)
 assert np.all(np.isfinite(features))

 def test_new_year_encoding(self):
 """Test encoding at New Year transition"""
 nye = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
 new_year = datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

 features_nye = extract_temporal_embeddings(nye)
 features_new = extract_temporal_embeddings(new_year)

 # Both should be valid
 assert features_nye.shape == (10,)
 assert features_new.shape == (10,)
 assert np.all(np.isfinite(features_nye))
 assert np.all(np.isfinite(features_new))

 def test_leap_year_encoding(self):
 """Test encoding on leap year"""
 leap_day = datetime(2024, 2, 29, 12, 0, 0, tzinfo=timezone.utc)

 features = extract_temporal_embeddings(leap_day)

 # Should handle leap day correctly
 assert features.shape == (10,)
 assert np.all(np.isfinite(features))

 def test_unknown_symbol_embedding(self):
 """Test getting embedding for unknown symbol"""
 # Should create new embedding without crashing
 embedding = get_symbol_embedding('XYZUSDT')

 assert embedding.shape == (4,)
 assert np.all(np.isfinite(embedding))

 def test_empty_symbol_list(self):
 """Test with empty symbol list"""
 embeddings = extract_symbol_embeddings(symbols=[])

 # Should return empty array
 assert embeddings.shape == (0,)


class TestEmbeddingsIntegration:
 """Integration tests for complete embedding feature extraction"""

 def test_extract_all_embedding_features(self):
 """Test extracting all 26 embedding features (16 symbol + 10 temporal)"""
 # Symbol embeddings (16 dims)
 symbol_features = extract_symbol_embeddings

 # Temporal embeddings (10 dims)
 timestamp = datetime.now(timezone.utc)
 temporal_features = extract_temporal_embeddings(timestamp)

 # Combine
 all_features = np.concatenate([symbol_features, temporal_features])

 # Should have 26 dimensions
 assert all_features.shape == (26,), f"Expected 26 dims, got {all_features.shape}"

 # All should be finite
 assert np.all(np.isfinite(all_features)), "All features should be finite"

 def test_temporal_consistency(self):
 """Test temporal embedding consistency"""
 timestamp = datetime(2025, 10, 10, 15, 30, 0, tzinfo=timezone.utc)

 # Extract twice
 features1 = extract_temporal_embeddings(timestamp)
 features2 = extract_temporal_embeddings(timestamp)

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_symbol_consistency(self):
 """Test symbol embedding consistency"""
 # Extract twice
 features1 = extract_symbol_embeddings
 features2 = extract_symbol_embeddings

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_performance_full_extraction(self):
 """Test performance of full embedding extraction"""
 timestamp = datetime.now(timezone.utc)

 def extract_all:
 symbol = extract_symbol_embeddings
 temporal = extract_temporal_embeddings(timestamp)
 return np.concatenate([symbol, temporal])

 result = assert_performance_acceptable(
 extract_all,
 max_time_ms=0.5 # <0.5ms for all 26 features
 )

 assert result.shape == (26,)


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
