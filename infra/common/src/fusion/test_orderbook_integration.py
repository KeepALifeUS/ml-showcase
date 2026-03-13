"""
Test script for OrderbookFeatureCalculator integration into StateVectorBuilder

Day 2.2: Verify orderbook features are correctly integrated into 768-dim state vector

Usage:
 cd /home/vlad/ML-Framework
 python -m packages.ml-common.src.fusion.test_orderbook_integration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from .state_vector import StateVectorBuilder, StateVectorConfig


def create_mock_ohlcv_data(symbols, window_hours=168):
 """Create mock OHLCV data for testing"""
 ohlcv_data = {}

 base_timestamp = datetime.now(timezone.utc) - timedelta(hours=window_hours)
 timestamps = [base_timestamp + timedelta(hours=i) for i in range(window_hours)]

 for i, symbol in enumerate(symbols):
 # Generate synthetic price data
 base_price = 50000.0 * (1.0 - i * 0.3) # BTC: 50k, ETH: 35k, BNB: 20k, SOL: 5k
 prices = base_price + np.cumsum(np.random.randn(window_hours) * 100)

 df = pd.DataFrame({
 'timestamp': timestamps,
 'open': prices + np.random.randn(window_hours) * 10,
 'high': prices + np.abs(np.random.randn(window_hours) * 20),
 'low': prices - np.abs(np.random.randn(window_hours) * 20),
 'close': prices,
 'volume': np.abs(np.random.randn(window_hours) * 1000) + 500,
 })

 ohlcv_data[symbol] = df

 return ohlcv_data


def create_mock_orderbook_data(symbols):
 """Create mock orderbook snapshot for testing"""
 orderbook_data = {}

 for i, symbol in enumerate(symbols):
 base_price = 50000.0 * (1.0 - i * 0.3)

 # Create 20-level orderbook
 bids = []
 asks = []
 for level in range(20):
 bid_price = base_price - (level + 1) * 5.0
 ask_price = base_price + (level + 1) * 5.0

 bid_size = 1.0 + np.random.rand * 10.0
 ask_size = 1.0 + np.random.rand * 10.0

 bids.append([bid_price, bid_size])
 asks.append([ask_price, ask_size])

 orderbook_data[symbol] = {
 'bids': bids,
 'asks': asks,
 'timestamp': datetime.now(timezone.utc),
 }

 return orderbook_data


def test_orderbook_integration:
 """Test OrderbookFeatureCalculator integration in StateVectorBuilder"""
 print("=" * 80)
 print("TEST: OrderbookFeatureCalculator Integration")
 print("=" * 80)

 # 1. Setup
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 print(f"\n1. Creating mock data for {len(symbols)} symbols...")

 ohlcv_data = create_mock_ohlcv_data(symbols, window_hours=168)
 orderbook_data = create_mock_orderbook_data(symbols)

 print(f" ✅ OHLCV data: {len(ohlcv_data)} symbols × 168 timesteps")
 print(f" ✅ Orderbook data: {len(orderbook_data)} symbols")

 # 2. Initialize StateVectorBuilder
 print("\n2. Initializing StateVectorBuilder...")
 config = StateVectorConfig(
 symbols=symbols,
 window_hours=168,
 log_build_time=True,
 )
 builder = StateVectorBuilder(config=config)
 print(f" ✅ StateVectorBuilder initialized (schema v1, {builder.schema.TOTAL_DIM} dims)")

 # 3. Build state vector
 print("\n3. Building state vector with orderbook features...")
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=None,
 timestamp=datetime.now(timezone.utc)
 )
 print(f" ✅ State vector built: shape={state_vector.shape}, dtype={state_vector.dtype}")
 print(f" ⏱️ Build time: {builder.build_time_ms:.2f}ms")

 # 4. Validate orderbook features
 print("\n4. Validating orderbook features (80 dims = 20 features × 4 symbols)...")
 start_idx, end_idx = builder.schema.get_feature_indices('orderbook')
 orderbook_features = state_vector[:, start_idx:end_idx]

 print(f" 📊 Orderbook slice: [{start_idx}:{end_idx}] = {end_idx - start_idx} dims")
 print(f" 📈 Shape: {orderbook_features.shape}")

 # Check each symbol's features
 for i, symbol in enumerate(symbols):
 symbol_start = i * 20
 symbol_end = symbol_start + 20
 symbol_features = orderbook_features[0, symbol_start:symbol_end] # First timestep

 # Feature validation
 spread_pct = symbol_features[0]
 imbalance_10 = symbol_features[2]
 bid_depth = symbol_features[4]
 ask_depth = symbol_features[5]

 print(f"\n {symbol}:")
 print(f" - spread_pct: {spread_pct:.4f}%")
 print(f" - imbalance_10: {imbalance_10:.4f}")
 print(f" - bid_depth_10: {bid_depth:.2f}")
 print(f" - ask_depth_10: {ask_depth:.2f}")

 # Basic sanity checks
 assert spread_pct >= 0, f"{symbol}: Invalid spread (negative)"
 assert -1.0 <= imbalance_10 <= 1.0, f"{symbol}: Imbalance out of range"
 assert bid_depth > 0, f"{symbol}: Invalid bid depth (non-positive)"
 assert ask_depth > 0, f"{symbol}: Invalid ask depth (non-positive)"

 print("\n ✅ All features validated successfully")

 # 5. Check broadcasting across timesteps
 print("\n5. Verifying feature broadcasting across 168 timesteps...")
 first_timestep = orderbook_features[0, :]
 last_timestep = orderbook_features[-1, :]

 if np.allclose(first_timestep, last_timestep):
 print(" ✅ Features correctly broadcasted across all timesteps")
 else:
 print(" ⚠️ Features vary across timesteps (expected for historical data)")

 # 6. Summary
 print("\n" + "=" * 80)
 print("TEST PASSED ✅")
 print("=" * 80)
 print(f"State Vector: {state_vector.shape[0]} timesteps × {state_vector.shape[1]} features")
 print(f"Orderbook Features: 80 dims = 4 symbols × 20 features")
 print(f"Non-zero features: {np.count_nonzero(orderbook_features)}/{orderbook_features.size}")
 print(f"Build time: {builder.build_time_ms:.2f}ms")
 print("=" * 80)

 return state_vector


if __name__ == '__main__':
 try:
 state_vector = test_orderbook_integration
 print("\n✅ Integration test completed successfully!")
 except Exception as e:
 print(f"\n❌ Integration test failed: {e}")
 import traceback
 traceback.print_exc
 exit(1)
