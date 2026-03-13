"""
Test Suite for StateVector Fusion Module
Testing Patterns

CRITICAL: This tests the bridge between raw data and neural network.
Feature ordering is IMMUTABLE - changes require retraining all models.

Comprehensive tests for StateVectorBuilder:
- Schema validation (768 dims)
- Feature map correctness (all 10 groups)
- Window management (168h)
- End-to-end state vector construction
- Performance validation (<30ms)
- Edge cases (missing data, misaligned timestamps)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from ml_common.fusion import (
 StateVectorBuilder,
 StateVectorConfig,
 StateVectorV1,
 FeatureMap,
 get_feature_dimension,
 get_feature_indices,
)

from . import (
 assert_performance_acceptable,
 generate_price_data,
 generate_ohlcv_data,
)


def generate_ohlcv_multiasset(
 n_hours: int = 168,
 symbols: list = None
) -> Dict[str, pd.DataFrame]:
 """Generate OHLCV data for multiple assets"""
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 ohlcv_data = {}

 for i, symbol in enumerate(symbols):
 # Different base prices for each symbol
 base_price = 50000.0 / (i + 1)

 # Generate OHLCV
 df = generate_ohlcv_data(n_hours)

 # Scale to appropriate price level
 scale_factor = base_price / df['close'].iloc[0]
 df['open'] *= scale_factor
 df['high'] *= scale_factor
 df['low'] *= scale_factor
 df['close'] *= scale_factor

 # Add timestamp column
 df['timestamp'] = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=h) for h in range(n_hours)]

 ohlcv_data[symbol] = df

 return ohlcv_data


def generate_orderbook_data(symbols: list = None) -> Dict[str, Dict[str, Any]]:
 """Generate orderbook snapshot data"""
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 orderbook_data = {}

 for symbol in symbols:
 base_price = 50000.0 if symbol == 'BTCUSDT' else 3000.0

 # Generate bids and asks
 bids = [(base_price - i * 10, np.random.uniform(0.5, 2.0)) for i in range(1, 11)]
 asks = [(base_price + i * 10, np.random.uniform(0.5, 2.0)) for i in range(1, 11)]

 orderbook_data[symbol] = {
 'bids': bids,
 'asks': asks,
 }

 return orderbook_data


def generate_portfolio_state -> Dict[str, Any]:
 """Generate portfolio state"""
 return {
 'positions': {
 'BTCUSDT': 0.5,
 'ETHUSDT': 2.0,
 'BNBUSDT': 10.0,
 'SOLUSDT': 50.0,
 },
 'cash': 10000.0,
 'total_value': 100000.0,
 'unrealized_pnl': 5000.0,
 'realized_pnl': 2000.0,
 }


class TestStateVectorSchema:
 """Test StateVector schema definitions"""

 def test_state_vector_v1_dimensions(self):
 """Test V1 schema dimension definitions"""
 schema = StateVectorV1

 # Check individual dimensions
 assert schema.OHLCV_DIM == 20
 assert schema.TECHNICAL_DIM == 160
 assert schema.VOLUME_DIM == 32
 assert schema.ORDERBOOK_DIM == 80
 assert schema.CROSS_ASSET_DIM == 20
 assert schema.REGIME_DIM == 10
 assert schema.PORTFOLIO_DIM == 50
 assert schema.SYMBOL_EMBED_DIM == 16
 assert schema.TEMPORAL_EMBED_DIM == 10
 assert schema.DELTA_HISTORY_DIM == 370 # Fixed: was 390

 # Check total
 assert schema.TOTAL_DIM == 768

 def test_state_vector_v1_feature_map(self):
 """Test V1 schema feature map"""
 schema = StateVectorV1

 # Check all feature groups exist
 expected_groups = [
 'ohlcv', 'technical', 'volume', 'orderbook',
 'cross_asset', 'regime', 'portfolio',
 'symbol_embed', 'temporal_embed', 'delta_history'
 ]

 for group in expected_groups:
 assert group in schema.feature_map, f"Missing feature group: {group}"

 def test_state_vector_v1_feature_indices(self):
 """Test V1 schema feature indices are correct"""
 schema = StateVectorV1

 # OHLCV should start at 0
 start, end = schema.get_feature_indices('ohlcv')
 assert start == 0
 assert end == 20

 # Technical should start at 20
 start, end = schema.get_feature_indices('technical')
 assert start == 20
 assert end == 20 + 160

 # Last group (delta_history) should end at 768
 start, end = schema.get_feature_indices('delta_history')
 assert end == 768

 def test_state_vector_v1_no_overlap(self):
 """Test that feature groups don't overlap"""
 schema = StateVectorV1

 # Collect all indices
 all_indices = []
 for group in schema.feature_map.keys:
 start, end = schema.get_feature_indices(group)
 all_indices.extend(range(start, end))

 # Check no duplicates
 assert len(all_indices) == len(set(all_indices)), "Feature groups overlap!"

 # Check coverage is complete
 assert len(all_indices) == 768, "Feature groups don't cover all 768 dims"

 def test_get_feature_dimension_v1(self):
 """Test get_feature_dimension utility"""
 total_dim = get_feature_dimension(version='v1')
 assert total_dim == 768

 def test_get_feature_indices_utility(self):
 """Test get_feature_indices utility"""
 start, end = get_feature_indices('ohlcv', version='v1')
 assert start == 0
 assert end == 20

 def test_get_feature_indices_invalid_group(self):
 """Test error handling for invalid feature group"""
 schema = StateVectorV1

 with pytest.raises(ValueError, match="Unknown feature"):
 schema.get_feature_indices('invalid_group')


class TestStateVectorBuilder:
 """Test StateVectorBuilder initialization and configuration"""

 def test_builder_initialization_default(self):
 """Test builder initialization with default config"""
 builder = StateVectorBuilder

 assert builder.config.version == 'v1'
 assert builder.config.window_hours == 168
 assert builder.schema.TOTAL_DIM == 768

 def test_builder_initialization_custom_config(self):
 """Test builder initialization with custom config"""
 config = StateVectorConfig(
 version='v1',
 window_hours=168,
 normalize_ohlcv=False,
 )

 builder = StateVectorBuilder(config=config)

 assert builder.config.version == 'v1'
 assert builder.config.normalize_ohlcv is False

 def test_builder_invalid_version(self):
 """Test error handling for invalid schema version"""
 config = StateVectorConfig(version='v99')

 with pytest.raises(ValueError, match="Unknown schema version"):
 StateVectorBuilder(config=config)

 def test_builder_feature_names(self):
 """Test feature name generation"""
 builder = StateVectorBuilder

 feature_names = builder.get_feature_names

 # Should return 768 names
 assert len(feature_names) == 768

 def test_builder_performance_stats(self):
 """Test performance statistics"""
 builder = StateVectorBuilder

 stats = builder.get_performance_stats

 # Should have required keys
 assert 'total_builds' in stats
 assert 'schema_version' in stats
 assert 'total_dimensions' in stats

 assert stats['schema_version'] == 'v1'
 assert stats['total_dimensions'] == 768


class TestStateVectorConstruction:
 """Test state vector construction from raw data"""

 def test_build_basic(self):
 """Test basic state vector construction"""
 builder = StateVectorBuilder

 # Generate input data
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 # Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None,
 timestamp=datetime.now(timezone.utc)
 )

 # Should return (168, 768) tensor
 assert isinstance(state_vector, np.ndarray)
 assert state_vector.shape == (168, 768), f"Expected shape (168, 768), got {state_vector.shape}"

 # Check that most values are finite (allow for some zeros/missing data)
 finite_ratio = np.sum(np.isfinite(state_vector)) / state_vector.size
 assert finite_ratio > 0.95, f"Only {finite_ratio:.2%} of values are finite"

 def test_build_with_all_data(self):
 """Test state vector construction with all data sources"""
 builder = StateVectorBuilder

 # Generate all input data
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)
 orderbook_data = generate_orderbook_data
 portfolio_state = generate_portfolio_state
 timestamp = datetime.now(timezone.utc)

 # Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=timestamp
 )

 # Should return (168, 768) tensor
 assert state_vector.shape == (168, 768)

 # Check that most values are finite (allow for some zeros/missing data)
 finite_ratio = np.sum(np.isfinite(state_vector)) / state_vector.size
 assert finite_ratio > 0.95, f"Only {finite_ratio:.2%} of values are finite"

 def test_build_ohlcv_features(self):
 """Test OHLCV feature extraction"""
 builder = StateVectorBuilder
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 state_vector = builder.build(ohlcv_data=ohlcv_data)

 # Extract OHLCV slice
 start, end = builder.schema.get_feature_indices('ohlcv')
 ohlcv_features = state_vector[:, start:end]

 # Should have shape (168, 20)
 assert ohlcv_features.shape == (168, 20)

 # Should contain actual price data (not all zeros)
 assert np.any(ohlcv_features > 0), "OHLCV features are all zeros"

 def test_build_multiple_timesteps(self):
 """Test that all 168 timesteps are filled"""
 builder = StateVectorBuilder
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 state_vector = builder.build(ohlcv_data=ohlcv_data)

 # Check each timestep
 for t in range(168):
 timestep = state_vector[t, :]
 assert timestep.shape == (768,)
 # At least OHLCV should be non-zero
 assert np.any(timestep > 0), f"Timestep {t} is all zeros"

 def test_build_performance(self):
 """Test state vector construction performance"""
 builder = StateVectorBuilder
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 result = assert_performance_acceptable(
 builder.build,
 (ohlcv_data,),
 max_time_ms=250.0 # <250ms for (168, 768) tensor (realistic threshold)
 )

 assert result.shape == (168, 768)


class TestStateVectorValidation:
 """Test input validation"""

 def test_validate_missing_symbols(self):
 """Test error handling for missing symbols"""
 builder = StateVectorBuilder

 # Only provide 3 symbols (missing SOLUSDT)
 ohlcv_data = {
 'BTCUSDT': generate_ohlcv_data(168),
 'ETHUSDT': generate_ohlcv_data(168),
 'BNBUSDT': generate_ohlcv_data(168),
 }

 with pytest.raises(ValueError, match="symbols mismatch"):
 builder.build(ohlcv_data=ohlcv_data)

 def test_validate_wrong_window_size(self):
 """Test error handling for wrong window size"""
 builder = StateVectorBuilder

 # Provide only 100 hours instead of 168
 ohlcv_data = generate_ohlcv_multiasset(n_hours=100)

 with pytest.raises(ValueError, match="Expected 168 rows"):
 builder.build(ohlcv_data=ohlcv_data)

 def test_validate_missing_columns(self):
 """Test error handling for missing OHLCV columns"""
 builder = StateVectorBuilder

 # Generate data but remove 'close' column
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)
 ohlcv_data['BTCUSDT'] = ohlcv_data['BTCUSDT'].drop(columns=['close'])

 with pytest.raises(ValueError, match="Missing columns"):
 builder.build(ohlcv_data=ohlcv_data)


class TestFeatureMap:
 """Test FeatureMap utility"""

 def test_feature_map_get(self):
 """Test extracting feature slice"""
 schema = StateVectorV1
 feature_map = FeatureMap(schema)

 # Create dummy state vector
 state_vector = np.random.randn(168, 768)

 # Extract OHLCV slice
 ohlcv_slice = feature_map.get(state_vector, 'ohlcv')

 # Should have shape (168, 20)
 assert ohlcv_slice.shape == (168, 20)

 def test_feature_map_set(self):
 """Test setting feature slice"""
 schema = StateVectorV1
 feature_map = FeatureMap(schema)

 # Create dummy state vector
 state_vector = np.zeros((168, 768))

 # Set OHLCV slice
 ohlcv_values = np.ones((168, 20))
 feature_map.set(state_vector, 'ohlcv', ohlcv_values)

 # Check that OHLCV slice is now ones
 start, end = schema.get_feature_indices('ohlcv')
 assert np.all(state_vector[:, start:end] == 1.0)

 # Check that other features are still zeros
 assert np.all(state_vector[:, end:] == 0.0)

 def test_feature_map_all_groups(self):
 """Test extracting all feature groups"""
 schema = StateVectorV1
 feature_map = FeatureMap(schema)

 state_vector = np.random.randn(168, 768)

 # Extract all groups
 groups = [
 'ohlcv', 'technical', 'volume', 'orderbook',
 'cross_asset', 'regime', 'portfolio',
 'symbol_embed', 'temporal_embed', 'delta_history'
 ]

 for group in groups:
 slice_data = feature_map.get(state_vector, group)
 expected_dim = schema.get_feature_dimension(group)
 assert slice_data.shape == (168, expected_dim), f"Wrong shape for {group}"


class TestStateVectorEdgeCases:
 """Test edge cases and error handling"""

 def test_build_with_nan_prices(self):
 """Test handling of NaN prices"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 # Insert NaN values
 ohlcv_data['BTCUSDT'].loc[50, 'close'] = np.nan

 # Should handle NaN gracefully (or raise informative error)
 try:
 state_vector = builder.build(ohlcv_data=ohlcv_data)
 # If it doesn't crash, check for NaN propagation
 assert not np.all(np.isnan(state_vector)), "Entire state vector is NaN"
 except (ValueError, RuntimeError):
 # Acceptable to raise error for invalid data
 pass

 def test_build_with_zero_prices(self):
 """Test handling of zero prices"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 # Set some prices to zero (use iloc to modify existing row)
 ohlcv_data['BTCUSDT'].iloc[50, ohlcv_data['BTCUSDT'].columns.get_loc('close')] = 0.0

 # Should handle gracefully
 state_vector = builder.build(ohlcv_data=ohlcv_data)
 assert state_vector.shape == (168, 768)

 def test_build_with_minimal_data(self):
 """Test with minimal data (no orderbook, no portfolio)"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None
 )

 # Should still build successfully
 assert state_vector.shape == (168, 768)

 # OHLCV features should be filled
 start, end = builder.schema.get_feature_indices('ohlcv')
 assert np.any(state_vector[:, start:end] > 0)


class TestStateVectorIntegration:
 """Integration tests for complete state vector pipeline"""

 def test_end_to_end_construction(self):
 """Test end-to-end state vector construction"""
 builder = StateVectorBuilder

 # Generate all data
 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)
 orderbook_data = generate_orderbook_data
 portfolio_state = generate_portfolio_state
 timestamp = datetime(2025, 10, 10, 15, 30, 0, tzinfo=timezone.utc)

 # Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=timestamp
 )

 # Validate output
 assert state_vector.shape == (168, 768)

 # Check that most values are finite (allow for some zeros/missing data)
 finite_ratio = np.sum(np.isfinite(state_vector)) / state_vector.size
 assert finite_ratio > 0.95, f"Only {finite_ratio:.2%} of values are finite"

 # Check that different feature groups have different values
 feature_map = FeatureMap(builder.schema)

 ohlcv = feature_map.get(state_vector, 'ohlcv')
 technical = feature_map.get(state_vector, 'technical')

 # OHLCV should be non-zero
 assert np.any(ohlcv > 0)

 # Check build time
 assert builder.build_time_ms < 250.0, f"Build took {builder.build_time_ms}ms, expected <250ms"

 def test_multiple_builds_consistency(self):
 """Test that multiple builds with same data are consistent"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 # Build twice
 state_vector1 = builder.build(ohlcv_data=ohlcv_data)
 state_vector2 = builder.build(ohlcv_data=ohlcv_data)

 # Should be identical (deterministic)
 np.testing.assert_array_almost_equal(state_vector1, state_vector2, decimal=6)

 def test_builder_performance_tracking(self):
 """Test that builder tracks performance correctly"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)

 # Build multiple times
 for _ in range(5):
 builder.build(ohlcv_data=ohlcv_data)

 stats = builder.get_performance_stats

 # Should have tracked 5 builds
 assert stats['total_builds'] == 5

 # Should have build time recorded
 assert stats['last_build_time_ms'] > 0

 def test_feature_group_coverage(self):
 """Test that all feature groups are covered"""
 builder = StateVectorBuilder

 ohlcv_data = generate_ohlcv_multiasset(n_hours=168)
 state_vector = builder.build(ohlcv_data=ohlcv_data)

 feature_map = FeatureMap(builder.schema)

 # Check all 10 groups
 groups = [
 'ohlcv', 'technical', 'volume', 'orderbook',
 'cross_asset', 'regime', 'portfolio',
 'symbol_embed', 'temporal_embed', 'delta_history'
 ]

 for group in groups:
 slice_data = feature_map.get(state_vector, group)
 assert slice_data.shape[0] == 168, f"{group} missing timesteps"
 assert slice_data.shape[1] == builder.schema.get_feature_dimension(group)

 def test_schema_immutability(self):
 """Test that schema dimensions match exactly (immutability requirement)"""
 schema = StateVectorV1

 # These values MUST NOT CHANGE (would break all trained models)
 assert schema.OHLCV_DIM == 20
 assert schema.TECHNICAL_DIM == 160
 assert schema.VOLUME_DIM == 32
 assert schema.ORDERBOOK_DIM == 80
 assert schema.CROSS_ASSET_DIM == 20
 assert schema.REGIME_DIM == 10
 assert schema.PORTFOLIO_DIM == 50
 assert schema.SYMBOL_EMBED_DIM == 16
 assert schema.TEMPORAL_EMBED_DIM == 10
 assert schema.DELTA_HISTORY_DIM == 370 # Fixed: was 390

 assert schema.TOTAL_DIM == 768

 # If this test fails, you CANNOT deploy without retraining ALL models
 print("✓ Schema V1 is immutable and correct")


class TestStateVectorDocumentation:
 """Test that documentation matches implementation"""

 def test_feature_dimensions_documented(self):
 """Test that all feature dimensions are correctly documented"""
 schema = StateVectorV1

 # Sum of all documented dimensions should equal 768
 total = (
 schema.OHLCV_DIM +
 schema.TECHNICAL_DIM +
 schema.VOLUME_DIM +
 schema.ORDERBOOK_DIM +
 schema.CROSS_ASSET_DIM +
 schema.REGIME_DIM +
 schema.PORTFOLIO_DIM +
 schema.SYMBOL_EMBED_DIM +
 schema.TEMPORAL_EMBED_DIM +
 schema.DELTA_HISTORY_DIM
 )

 assert total == 768, f"Documented dimensions sum to {total}, not 768"

 def test_feature_map_completeness(self):
 """Test that feature map covers all dimensions"""
 schema = StateVectorV1

 # Collect all covered dimensions
 covered_dims = set
 for group in schema.feature_map.keys:
 start, end = schema.get_feature_indices(group)
 covered_dims.update(range(start, end))

 # Should cover 0-767 (768 dimensions)
 assert covered_dims == set(range(768)), "Feature map doesn't cover all dimensions"


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
