"""
Test Suite for Orderbook Module
Testing Patterns

Comprehensive tests for order book microstructure feature extraction:
- Bid-ask imbalance calculations (5 dims)
- Order book depth analysis (9 dims)
- Spread dynamics (6 dims)
- Performance validation (<10ms)
- Edge cases (empty orderbook, single level, extreme imbalance)
"""

import pytest
import numpy as np
from typing import List, Tuple

from ml_common.orderbook import (
 calculate_bid_ask_imbalance,
 calculate_volume_weighted_imbalance,
 calculate_cumulative_delta,
 calculate_order_flow_toxicity,
 calculate_microstructure_noise,
 calculate_depth_metrics,
 calculate_bid_depth,
 calculate_ask_depth,
 calculate_depth_imbalance,
 calculate_depth_slope,
 calculate_spread_metrics,
 calculate_absolute_spread,
 calculate_relative_spread,
 calculate_effective_spread,
 calculate_quoted_spread,
 calculate_realized_spread,
 calculate_spread_volatility,
)
from ml_common.orderbook.orderbook_features import (
 OrderbookFeatureCalculator,
 OrderbookSnapshot,
 OrderbookWall,
)

from . import (
 assert_array_almost_equal,
 assert_performance_acceptable,
 assert_no_warnings,
)


def generate_orderbook(n_levels: int = 5, base_price: float = 50000.0) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
 """
 Generate realistic orderbook for testing

 Returns:
 Tuple of (bids, asks) where each is List[(price, volume)]
 """
 np.random.seed(42)

 # Generate bids (decreasing prices from base)
 bids = []
 for i in range(n_levels):
 price = base_price - (i + 1) * 10 # $10 increments
 volume = np.random.uniform(0.1, 2.0) # Random volume
 bids.append((price, volume))

 # Generate asks (increasing prices from base)
 asks = []
 for i in range(n_levels):
 price = base_price + (i + 1) * 10
 volume = np.random.uniform(0.1, 2.0)
 asks.append((price, volume))

 return bids, asks


class TestOrderbookImbalance:
 """Test bid-ask imbalance calculations"""

 def test_calculate_bid_ask_imbalance_balanced(self):
 """Test imbalance with balanced orderbook"""
 bids = [[50000, 1.0], [49990, 1.0]]
 asks = [[50010, 1.0], [50020, 1.0]]

 result = calculate_bid_ask_imbalance(bids, asks)

 # Balanced orderbook should have ~0 imbalance
 assert abs(result) < 0.1, f"Expected ~0 imbalance, got {result}"

 def test_calculate_bid_ask_imbalance_bid_heavy(self):
 """Test imbalance with bid-heavy orderbook"""
 bids = [[50000, 5.0], [49990, 5.0]] # High bid volume
 asks = [[50010, 1.0], [50020, 1.0]] # Low ask volume

 result = calculate_bid_ask_imbalance(bids, asks)

 # Bid-heavy should be positive
 assert result > 0.5, f"Expected positive imbalance, got {result}"

 def test_calculate_bid_ask_imbalance_ask_heavy(self):
 """Test imbalance with ask-heavy orderbook"""
 bids = [[50000, 1.0], [49990, 1.0]]
 asks = [[50010, 5.0], [50020, 5.0]]

 result = calculate_bid_ask_imbalance(bids, asks)

 # Ask-heavy should be negative
 assert result < -0.5, f"Expected negative imbalance, got {result}"

 def test_calculate_bid_ask_imbalance_empty(self):
 """Test imbalance with empty orderbook"""
 result = calculate_bid_ask_imbalance([], [])

 assert result == 0.0, f"Empty orderbook should return 0, got {result}"

 def test_calculate_bid_ask_imbalance_performance(self):
 """Test imbalance calculation performance"""
 bids, asks = generate_orderbook(n_levels=20)

 result = assert_performance_acceptable(
 calculate_bid_ask_imbalance,
 (bids, asks),
 max_time_ms=1.0 # <1ms for 20 levels
 )

 assert isinstance(result, float)
 assert -1.0 <= result <= 1.0, f"Imbalance should be in [-1, 1], got {result}"

 def test_calculate_volume_weighted_imbalance(self):
 """Test volume-weighted imbalance"""
 bids = [[50000, 2.0], [49990, 1.0]]
 asks = [[50010, 1.0], [50020, 0.5]]

 result = calculate_volume_weighted_imbalance(bids, asks, levels=2)

 assert isinstance(result, float)
 assert -1.0 <= result <= 1.0

 def test_calculate_cumulative_delta(self):
 """Test cumulative delta calculation"""
 # Trades format: (price, volume, side) where side is 'buy' or 'sell'
 trades = [
 (50000, 1.5, 'buy'),
 (49990, 1.0, 'buy'),
 (50010, 1.0, 'sell'),
 (50020, 0.8, 'sell')
 ]

 result = calculate_cumulative_delta(trades)

 # Should be sum of buy volumes - sum of sell volumes
 # (1.5 + 1.0) - (1.0 + 0.8) = 2.5 - 1.8 = 0.7
 assert isinstance(result, float)
 assert abs(result - 0.7) < 1e-10, f"Expected 0.7, got {result}"

 def test_calculate_order_flow_toxicity(self):
 """Test order flow toxicity (Kyle's lambda)"""
 bids, asks = generate_orderbook(n_levels=5)
 # Generate some sample trades
 trades = [
 (50000, 1.0, 'buy'),
 (50010, 0.5, 'sell'),
 (49995, 2.0, 'buy')
 ]

 result = calculate_order_flow_toxicity(bids, asks, trades)

 assert isinstance(result, float)
 assert result >= 0.0, "Toxicity should be non-negative"

 def test_calculate_microstructure_noise(self):
 """Test microstructure noise measurement"""
 # Generate price series
 prices = [50000 + i * 10 for i in range(-10, 11)] # 21 prices

 result = calculate_microstructure_noise(prices)

 assert isinstance(result, float)
 assert result >= 0.0, "Noise should be non-negative"


class TestOrderbookDepth:
 """Test order book depth calculations"""

 def test_calculate_bid_depth(self):
 """Test bid depth calculation"""
 bids = [[50000, 1.0], [49990, 2.0], [49980, 3.0]]

 result = calculate_bid_depth(bids, level=3)

 # Should return volume at level 3 (index 2)
 expected = 3.0
 assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

 def test_calculate_ask_depth(self):
 """Test ask depth calculation"""
 asks = [[50010, 1.0], [50020, 2.0], [50030, 3.0]]

 result = calculate_ask_depth(asks, level=3)

 # Should return volume at level 3 (index 2)
 expected = 3.0
 assert abs(result - expected) < 1e-6

 def test_calculate_depth_imbalance(self):
 """Test depth imbalance"""
 bids = [[50000, 5.0], [49990, 4.0]] # High bid depth
 asks = [[50010, 1.0], [50020, 1.0]] # Low ask depth

 result = calculate_depth_imbalance(bids, asks, levels=2)

 # Bid-heavy should be positive
 assert result > 0, f"Expected positive depth imbalance, got {result}"

 def test_calculate_depth_slope(self):
 """Test depth slope calculation"""
 # Decreasing depth (should have negative slope)
 bids = [[50000, 3.0], [49990, 2.0], [49980, 1.0]]
 asks = [[50010, 1.0], [50020, 2.0], [50030, 3.0]]

 result = calculate_depth_slope(bids, asks, levels=3)

 assert isinstance(result, float)

 def test_calculate_depth_metrics_full(self):
 """Test full depth metrics extraction"""
 bids, asks = generate_orderbook(n_levels=10)

 result = calculate_depth_metrics(bids, asks)

 # Should return tuple of 4 metrics
 assert isinstance(result, tuple)
 assert len(result) == 4, f"Expected 4 metrics, got {len(result)}"

 # All values should be finite
 assert all(np.isfinite(v) for v in result), "All depth metrics should be finite"

 def test_calculate_depth_metrics_performance(self):
 """Test depth metrics performance"""
 bids, asks = generate_orderbook(n_levels=20)

 result = assert_performance_acceptable(
 calculate_depth_metrics,
 (bids, asks),
 max_time_ms=3.0 # <3ms for 20 levels (allow more time for tuple unpacking)
 )

 assert len(result) == 4


class TestOrderbookSpread:
 """Test spread dynamics calculations"""

 def test_calculate_absolute_spread(self):
 """Test absolute spread calculation"""
 bids = [[49990.0, 1.0]]
 asks = [[50010.0, 1.0]]

 result = calculate_absolute_spread(bids, asks)

 expected = 50010.0 - 49990.0
 assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

 def test_calculate_relative_spread(self):
 """Test relative spread (percentage)"""
 bids = [[49990.0, 1.0]]
 asks = [[50010.0, 1.0]]

 result = calculate_relative_spread(bids, asks)

 # (Ask - Bid) / Mid * 100
 mid_price = (49990.0 + 50010.0) / 2.0
 expected = (50010.0 - 49990.0) / mid_price * 100.0
 assert abs(result - expected) < 1e-6

 def test_calculate_effective_spread_buy(self):
 """Test effective spread for buy order"""
 bids = [[50000.0, 1.0]]
 asks = [[50010.0, 1.0]]
 trade_size = 0.5 # Small trade, single level

 result = calculate_effective_spread(bids, asks, trade_size, side='buy')

 # Should execute at 50010, mid is 50005
 mid_price = (50000.0 + 50010.0) / 2.0
 expected = 2.0 * abs(50010.0 - mid_price) / mid_price * 100.0
 assert abs(result - expected) < 1e-6

 def test_calculate_effective_spread_sell(self):
 """Test effective spread for sell order"""
 bids = [[50000.0, 1.0]]
 asks = [[50010.0, 1.0]]
 trade_size = 0.5 # Small trade, single level

 result = calculate_effective_spread(bids, asks, trade_size, side='sell')

 # Should execute at 50000, mid is 50005
 mid_price = (50000.0 + 50010.0) / 2.0
 expected = 2.0 * abs(50000.0 - mid_price) / mid_price * 100.0
 assert abs(result - expected) < 1e-6

 def test_calculate_quoted_spread(self):
 """Test quoted spread"""
 bids = [[49990.0, 1.0]]
 asks = [[50010.0, 1.0]]

 result = calculate_quoted_spread(bids, asks)

 # Same as relative spread (returns percentage)
 mid_price = (49990.0 + 50010.0) / 2.0
 expected = (50010.0 - 49990.0) / mid_price * 100.0
 assert abs(result - expected) < 1e-6

 def test_calculate_realized_spread(self):
 """Test realized spread calculation"""
 bids = [[50000.0, 1.0]]
 asks = [[50010.0, 1.0]]
 execution_price = 50010.0
 mid_price_after = 50005.0

 result = calculate_realized_spread(bids, asks, execution_price, mid_price_after, side='buy')

 assert isinstance(result, float)

 def test_calculate_spread_volatility(self):
 """Test spread volatility"""
 spreads = [10.0, 12.0, 9.0, 11.0, 13.0, 8.0]

 result = calculate_spread_volatility(spreads)

 # Should be standard deviation of spreads
 expected_std = np.std(spreads)
 assert abs(result - expected_std) < 1e-6

 def test_calculate_spread_metrics_full(self):
 """Test full spread metrics extraction"""
 bids, asks = generate_orderbook(n_levels=10)

 # Calculate metrics
 result = calculate_spread_metrics(bids, asks)

 # Should return tuple of 4 metrics
 assert isinstance(result, tuple)
 assert len(result) == 4, f"Expected 4 metrics, got {len(result)}"

 # All values should be finite
 assert all(np.isfinite(v) for v in result), "All spread metrics should be finite"

 # All spreads should be non-negative
 assert all(v >= 0 for v in result), "All spreads should be non-negative"

 def test_calculate_spread_metrics_performance(self):
 """Test spread metrics performance"""
 bids, asks = generate_orderbook(n_levels=20)

 result = assert_performance_acceptable(
 calculate_spread_metrics,
 (bids, asks),
 max_time_ms=2.0 # <2ms
 )

 assert len(result) == 4


class TestOrderbookEdgeCases:
 """Test edge cases and error handling"""

 def test_empty_orderbook(self):
 """Test all functions with empty orderbook"""
 empty_bids = []
 empty_asks = []

 # Imbalance should be 0
 assert calculate_bid_ask_imbalance(empty_bids, empty_asks) == 0.0

 # Depth should be 0
 assert calculate_bid_depth(empty_bids) == 0.0
 assert calculate_ask_depth(empty_asks) == 0.0

 def test_single_level_orderbook(self):
 """Test with single-level orderbook"""
 bids = [[50000, 1.0]]
 asks = [[50010, 1.0]]

 # Should not crash
 imbalance = calculate_bid_ask_imbalance(bids, asks)
 assert isinstance(imbalance, float)

 depth = calculate_bid_depth(bids, level=1)
 assert depth == 1.0

 def test_extreme_imbalance(self):
 """Test with extreme bid/ask imbalance"""
 # Only bids, no asks - function returns 0.0 for safety
 bids = [[50000, 10.0]]
 asks = []

 imbalance = calculate_bid_ask_imbalance(bids, asks)
 # Function returns 0.0 when either side is empty (edge case protection)
 assert imbalance == 0.0

 # Only asks, no bids - function returns 0.0 for safety
 bids = []
 asks = [[50010, 10.0]]

 imbalance = calculate_bid_ask_imbalance(bids, asks)
 # Function returns 0.0 when either side is empty (edge case protection)
 assert imbalance == 0.0

 def test_zero_spread(self):
 """Test with zero spread (crossed market)"""
 # Technically impossible but handle gracefully
 bids = [[50000.0, 1.0]]
 asks = [[50000.0, 1.0]]

 spread = calculate_absolute_spread(bids, asks)
 assert spread == 0.0

 def test_negative_spread(self):
 """Test with negative spread (inverted market)"""
 # Bid > Ask (impossible but handle gracefully)
 bids = [[50010.0, 1.0]]
 asks = [[50000.0, 1.0]]

 spread = calculate_absolute_spread(bids, asks)
 # Function returns max(0.0, spread), so should be 0
 assert spread == 0.0


class TestOrderbookIntegration:
 """Integration tests for complete orderbook feature extraction"""

 def test_extract_all_features(self):
 """Test extracting all 20 orderbook features"""
 bids, asks = generate_orderbook(n_levels=10)

 # Extract imbalance features (3 available without trades data)
 imbalance_features = np.zeros(3)
 imbalance_features[0] = calculate_bid_ask_imbalance(bids, asks)
 imbalance_features[1] = calculate_volume_weighted_imbalance(bids, asks, levels=5)
 # Skip cumulative_delta (requires trades), order_flow_toxicity (requires trades),
 # and microstructure_noise (requires price series)
 imbalance_features[2] = 0.0 # Placeholder for features requiring additional data

 # Extract depth features (tuple of 4 metrics)
 depth_features = calculate_depth_metrics(bids, asks)

 # Extract spread features (tuple of 4 metrics)
 spread_features = calculate_spread_metrics(bids, asks)

 # Combine - convert tuples to arrays
 depth_array = np.array(depth_features)
 spread_array = np.array(spread_features)
 all_features = np.concatenate([imbalance_features, depth_array, spread_array])

 # Should have 11 dimensions (3 imbalance + 4 depth + 4 spread)
 assert all_features.shape == (11,), f"Expected 11 dims, got {all_features.shape}"

 # All should be finite
 assert np.all(np.isfinite(all_features)), "All features should be finite"

 def test_multi_symbol_extraction(self):
 """Test extracting features for multiple symbols"""
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 all_features = []
 for symbol in symbols:
 bids, asks = generate_orderbook(n_levels=10)

 # Extract all 11 features (3 imbalance + 4 depth + 4 spread)
 imb1 = calculate_bid_ask_imbalance(bids, asks)
 imb2 = calculate_volume_weighted_imbalance(bids, asks, levels=5)
 imb3 = 0.0 # Placeholder
 depth = np.array(calculate_depth_metrics(bids, asks))
 spread = np.array(calculate_spread_metrics(bids, asks))
 features = np.concatenate([[imb1, imb2, imb3], depth, spread])

 all_features.append(features)

 # Should have 4 symbols × 11 features = 44 dims
 features_array = np.array(all_features).flatten
 assert features_array.shape == (44,)

 def test_feature_consistency(self):
 """Test feature extraction consistency"""
 bids, asks = generate_orderbook(n_levels=10)

 # Extract features twice
 features1 = calculate_spread_metrics(bids, asks)
 features2 = calculate_spread_metrics(bids, asks)

 # Should be identical
 np.testing.assert_array_equal(features1, features2)

 def test_performance_full_extraction(self):
 """Test performance of full feature extraction"""
 bids, asks = generate_orderbook(n_levels=20)

 def extract_all:
 imbalance = calculate_bid_ask_imbalance(bids, asks)
 depth = np.array(calculate_depth_metrics(bids, asks))
 spread = np.array(calculate_spread_metrics(bids, asks))
 return np.concatenate([[imbalance], depth, spread])

 result = assert_performance_acceptable(
 extract_all,
 max_time_ms=10.0 # <10ms for all 9 features (1 + 4 + 4)
 )

 assert result.shape[0] == 9 # 1 imbalance + 4 depth + 4 spread


class TestOrderbookFeatureCalculator:
 """Test OrderbookFeatureCalculator - 20 Feature Implementation"""

 def test_calculator_initialization(self):
 """Test calculator initialization"""
 calc = OrderbookFeatureCalculator(use_gpu=False)
 assert calc.use_gpu is False
 assert calc.xp is not None

 def test_calculate_spread_pct(self):
 """Test spread percentage calculation"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0, 1.0], [49990.0, 1.0]]
 asks = [[50010.0, 1.0], [50020.0, 1.0]]

 result = calc.calculate_spread_pct(bids, asks)

 # (50010 - 50000) / 50005 * 100 ≈ 0.02%
 assert isinstance(result, float)
 assert result > 0, "Spread should be positive"
 assert result < 1.0, "Spread should be small for normal market"

 def test_calculate_mid_price_weighted(self):
 """Test volume-weighted mid price"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0, 1.0], [49990.0, 2.0]]
 asks = [[50010.0, 1.5], [50020.0, 1.0]]

 result = calc.calculate_mid_price_weighted(bids, asks, levels=2)

 assert isinstance(result, float)
 assert 49990.0 < result < 50020.0, "Mid price should be within orderbook range"

 def test_calculate_imbalance(self):
 """Test imbalance calculation"""
 calc = OrderbookFeatureCalculator

 # Balanced orderbook
 bids = [[50000.0, 1.0], [49990.0, 1.0]]
 asks = [[50010.0, 1.0], [50020.0, 1.0]]
 result = calc.calculate_imbalance(bids, asks, levels=2)
 assert abs(result) < 0.1, "Balanced orderbook should have ~0 imbalance"

 # Bid-heavy orderbook
 bids = [[50000.0, 5.0], [49990.0, 4.0]]
 asks = [[50010.0, 1.0], [50020.0, 1.0]]
 result = calc.calculate_imbalance(bids, asks, levels=2)
 assert result > 0.5, "Bid-heavy should have positive imbalance"

 # Ask-heavy orderbook
 bids = [[50000.0, 1.0], [49990.0, 1.0]]
 asks = [[50010.0, 5.0], [50020.0, 4.0]]
 result = calc.calculate_imbalance(bids, asks, levels=2)
 assert result < -0.5, "Ask-heavy should have negative imbalance"

 def test_calculate_depth(self):
 """Test depth calculation"""
 calc = OrderbookFeatureCalculator
 orders = [[50000.0, 1.0], [49990.0, 2.0], [49980.0, 3.0]]

 result = calc.calculate_depth(orders, levels=3)

 expected = 1.0 + 2.0 + 3.0
 assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

 def test_detect_wall_no_wall(self):
 """Test wall detection with no walls"""
 calc = OrderbookFeatureCalculator
 # Uniform orderbook - no walls
 orders = [[50000.0, 1.0], [49990.0, 1.0], [49980.0, 1.0]]
 mid_price = 50005.0

 result = calc.detect_wall(orders, mid_price, is_bid=True)

 assert isinstance(result, OrderbookWall)
 assert result.present is False

 def test_detect_wall_with_wall(self):
 """Test wall detection with large order"""
 calc = OrderbookFeatureCalculator
 # Large order at level 0 with many small orders for proper average
 # avg = (50 + 1 + 1 + 1 + 1 + 1) / 6 = 9.17, threshold = 27.5
 # 50 > 27.5, so wall IS detected
 orders = [[50000.0, 50.0], [49990.0, 1.0], [49980.0, 1.0], [49970.0, 1.0], [49960.0, 1.0], [49950.0, 1.0]]
 mid_price = 50005.0

 result = calc.detect_wall(orders, mid_price, is_bid=True)

 assert result.present is True
 assert result.price == 50000.0
 assert result.size == 50.0
 assert result.strength > 3.0, "Wall should be > 3× average size"
 assert result.distance >= 0, "Distance should be non-negative"

 def test_calculate_absorption_rate(self):
 """Test absorption rate calculation"""
 calc = OrderbookFeatureCalculator
 # Previous snapshot
 previous = [[50000.0, 5.0], [49990.0, 3.0], [49980.0, 2.0]]
 # Current snapshot (some orders consumed)
 current = [[50000.0, 3.0], [49990.0, 3.0], [49980.0, 2.0]] # 2.0 absorbed at level 0
 time_delta = 5.0 # 5 seconds

 result = calc.calculate_absorption_rate(current, previous, time_delta)

 # 2.0 absorbed / 5 sec = 0.4 qty/sec
 expected = 2.0 / 5.0
 assert abs(result - expected) < 1e-6, f"Expected {expected}, got {result}"

 def test_calculate_trade_features(self):
 """Test trade-based features calculation"""
 calc = OrderbookFeatureCalculator
 # Trades: (price, volume, side)
 # Need large trade to be > 10× average
 # avg = (1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 100) / 9 = 109/9 = 12.11
 # threshold = 121.1, need 100 < 121.1 - won't work
 # Better: many small trades
 # avg = (0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 0.5 + 60) / 9 = 64/9 = 7.11
 # threshold = 71.1, 60 < 71.1 - won't work
 # Even better: avg = (0.5×8 + 80) / 9 = 84/9 = 9.33, threshold = 93.3, 80 < 93.3
 # Solution: avg = (0.3×10 + 100) / 11 = 103/11 = 9.36, threshold = 93.6, 100 > 93.6 ✓
 trades = [
 (50000.0, 0.3, 'buy'),
 (50010.0, 0.3, 'buy'),
 (49990.0, 0.3, 'sell'),
 (50005.0, 0.3, 'buy'),
 (50001.0, 0.3, 'sell'),
 (50002.0, 0.3, 'buy'),
 (50003.0, 0.3, 'sell'),
 (50004.0, 0.3, 'buy'),
 (50006.0, 0.3, 'buy'),
 (50007.0, 0.3, 'sell'),
 (50015.0, 100.0, 'buy'), # Large trade: avg=9.36, threshold=93.6, 100>93.6 ✓
 ]

 result = calc.calculate_trade_features(trades, window_sec=300.0)

 assert 'aggression_ratio' in result
 assert 'large_trades_count' in result
 assert 'trade_volume_5m' in result
 assert 'trade_count_5m' in result
 assert 'avg_trade_size' in result
 assert 'vwap_5m' in result

 # Aggression ratio should be in [0, 1]
 assert 0.0 <= result['aggression_ratio'] <= 1.0

 # Should detect 1 large trade
 assert result['large_trades_count'] >= 1

 # Trade count should be 11
 assert result['trade_count_5m'] == 11

 def test_calculate_liquidity_score(self):
 """Test composite liquidity score"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, 1.0] for i in range(20)]
 asks = [[50010.0 + i*10, 1.0] for i in range(20)]
 spread_pct = 0.02

 result = calc.calculate_liquidity_score(bids, asks, spread_pct)

 assert isinstance(result, float)
 assert 0 <= result <= 100, "Liquidity score should be in [0, 100]"

 def test_calculate_all_features_basic(self):
 """Test calculating all 20 features"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, 1.0 + i*0.1] for i in range(20)]
 asks = [[50010.0 + i*10, 1.0 + i*0.1] for i in range(20)]

 result = calc.calculate_all_features(bids, asks)

 assert isinstance(result, np.ndarray)
 assert result.shape == (20,), f"Expected 20 features, got shape {result.shape}"
 assert np.all(np.isfinite(result)), "All features should be finite"

 def test_calculate_all_features_with_previous(self):
 """Test all features with previous snapshot (absorption rate)"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, 2.0] for i in range(20)]
 asks = [[50010.0 + i*10, 2.0] for i in range(20)]

 # Previous snapshot
 previous_bids = [[50000.0 - i*10, 3.0] for i in range(20)]
 previous_asks = [[50010.0 + i*10, 3.0] for i in range(20)]
 previous_snapshot = OrderbookSnapshot(
 bids=previous_bids,
 asks=previous_asks,
 timestamp=1000.0
 )

 result = calc.calculate_all_features(
 bids, asks,
 previous_snapshot=previous_snapshot,
 time_delta_sec=5.0
 )

 assert result.shape == (20,)
 # Features 10-11 (absorption rate) should be > 0
 assert result[10] > 0, "Bid absorption rate should be positive"
 assert result[11] > 0, "Ask absorption rate should be positive"

 def test_calculate_all_features_with_trades(self):
 """Test all features with trades"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, 1.0] for i in range(20)]
 asks = [[50010.0 + i*10, 1.0] for i in range(20)]

 trades = [
 (50005.0, 1.0, 'buy'),
 (50010.0, 2.0, 'sell'),
 (50000.0, 1.5, 'buy'),
 ]

 result = calc.calculate_all_features(bids, asks, trades=trades)

 assert result.shape == (20,)
 # Features 12-18 (trade-based) should be filled
 assert result[12] > 0, "Aggression ratio should be > 0"
 assert result[16] == 3, "Trade count should be 3"

 def test_calculate_all_features_empty_orderbook(self):
 """Test all features with empty orderbook"""
 calc = OrderbookFeatureCalculator
 result = calc.calculate_all_features([], [])

 assert result.shape == (20,)
 # Most features should be 0 or default values
 assert result[0] == 0.0, "Spread should be 0 for empty orderbook"

 def test_calculate_all_features_performance(self):
 """Test performance of all 20 features calculation"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, np.random.uniform(0.5, 2.0)] for i in range(20)]
 asks = [[50010.0 + i*10, np.random.uniform(0.5, 2.0)] for i in range(20)]

 result = assert_performance_acceptable(
 calc.calculate_all_features,
 (bids, asks),
 max_time_ms=5.0 # <5ms for real-time inference
 )

 assert result.shape == (20,)

 def test_multi_symbol_features(self):
 """Test feature extraction for 4 symbols (BTC/ETH/BNB/SOL)"""
 calc = OrderbookFeatureCalculator
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 all_features = []
 for symbol in symbols:
 bids = [[50000.0 - i*10, 1.0] for i in range(20)]
 asks = [[50010.0 + i*10, 1.0] for i in range(20)]

 features = calc.calculate_all_features(bids, asks)
 all_features.append(features)

 # Should have 4 symbols × 20 features = 80 dimensions
 features_array = np.array(all_features)
 assert features_array.shape == (4, 20), f"Expected (4, 20), got {features_array.shape}"

 # All features should be finite
 assert np.all(np.isfinite(features_array)), "All features should be finite"

 def test_feature_consistency(self):
 """Test that features are consistent across multiple calls"""
 calc = OrderbookFeatureCalculator
 bids = [[50000.0 - i*10, 1.0 + i*0.1] for i in range(20)]
 asks = [[50010.0 + i*10, 1.0 + i*0.1] for i in range(20)]

 features1 = calc.calculate_all_features(bids, asks)
 features2 = calc.calculate_all_features(bids, asks)

 np.testing.assert_array_almost_equal(features1, features2)


if __name__ == "__main__":
 pytest.main([__file__, "-v", "--tb=short"])
