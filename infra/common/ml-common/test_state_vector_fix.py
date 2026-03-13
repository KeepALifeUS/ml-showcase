#!/usr/bin/env python3
"""
Test script for StateVectorBuilder non-zero features fix

Checks that after implementation technical/delta/portfolio features
percent nonzero elements increased with 2.6% to ≥70%
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# Add src to path
sys.path.insert(0, '/home/vlad/ML-Framework/packages/ml-common/src')

from fusion.state_vector import StateVectorBuilder, StateVectorConfig


def generate_synthetic_ohlcv(symbol: str, window_hours: int = 168) -> pd.DataFrame:
 """Generate synthetic OHLCV data for testing"""

 # Set different base prices for different symbols
 base_prices = {
 'BTCUSDT': 45000.0,
 'ETHUSDT': 2500.0,
 'BNBUSDT': 350.0,
 'SOLUSDT': 100.0,
 }

 base_price = base_prices.get(symbol, 1000.0)

 # Generate random walk with trend
 np.random.seed(hash(symbol) % 2**32)

 # Generate returns
 returns = np.random.normal(0.0001, 0.01, window_hours) # Slight upward drift

 # Generate prices
 prices = base_price * np.exp(np.cumsum(returns))

 # Generate OHLCV
 data = []
 for i in range(window_hours):
 price = prices[i]

 # High/Low with realistic spread
 hl_range = price * np.random.uniform(0.005, 0.02)
 high = price + hl_range / 2
 low = price - hl_range / 2

 # Open/Close within H/L
 open_price = np.random.uniform(low, high)
 close_price = price # Use price as close

 # Volume (realistic for each symbol)
 if 'BTC' in symbol:
 volume = np.random.uniform(100, 500)
 elif 'ETH' in symbol:
 volume = np.random.uniform(500, 2000)
 elif 'BNB' in symbol:
 volume = np.random.uniform(200, 1000)
 else: # SOL
 volume = np.random.uniform(1000, 5000)

 data.append({
 'timestamp': datetime.now(timezone.utc) - timedelta(hours=window_hours - i),
 'open': open_price,
 'high': high,
 'low': low,
 'close': close_price,
 'volume': volume,
 })

 return pd.DataFrame(data)


def test_state_vector_nonzero:
 """Test that state vector has ≥70% non-zero features"""

 print("=" * 80)
 print("STATE VECTOR NON-ZERO TEST")
 print("=" * 80)
 print

 # Create config
 config = StateVectorConfig(
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours=168
 )

 print(f"Config: {config.window_hours}h window, {len(config.symbols)} symbols")
 print

 # Generate synthetic OHLCV data
 print("Generating synthetic OHLCV data...")
 ohlcv_data = {}
 for symbol in config.symbols:
 df = generate_synthetic_ohlcv(symbol, config.window_hours)
 ohlcv_data[symbol] = df
 print(f" {symbol}: {len(df)} rows, price range ${df['close'].min:.2f}-${df['close'].max:.2f}")

 print

 # Create builder
 print("Creating StateVectorBuilder...")
 builder = StateVectorBuilder(config=config)
 print(f" Schema version: {builder.config.version}")
 print(f" Total dimensions: {builder.schema.TOTAL_DIM}")
 print

 # Build state vector (without portfolio state - training mode)
 print("Building state vector (training mode - no portfolio)...")
 state = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None
 )

 print(f" Shape: {state.shape}")
 print(f" Build time: {builder.build_time_ms:.2f}ms")
 print

 # Calculate statistics
 print("=" * 80)
 print("FEATURE STATISTICS")
 print("=" * 80)
 print

 total_elements = state.size
 non_zero_elements = np.count_nonzero(state)
 zero_elements = total_elements - non_zero_elements

 non_zero_pct = (non_zero_elements / total_elements) * 100

 print(f"Total elements: {total_elements:,}")
 print(f"Non-zero elements: {non_zero_elements:,} ({non_zero_pct:.1f}%)")
 print(f"Zero elements: {zero_elements:,} ({100 - non_zero_pct:.1f}%)")
 print

 # Per-feature group analysis
 print("=" * 80)
 print("PER-FEATURE GROUP ANALYSIS")
 print("=" * 80)
 print

 feature_groups = [
 'ohlcv',
 'technical',
 'volume',
 'orderbook',
 'cross_asset',
 'regime',
 'portfolio',
 'symbol_embed',
 'temporal_embed',
 'delta_history'
 ]

 for group_name in feature_groups:
 start_idx, end_idx = builder.schema.get_feature_indices(group_name)
 group_slice = state[:, start_idx:end_idx]

 group_total = group_slice.size
 group_nonzero = np.count_nonzero(group_slice)
 group_pct = (group_nonzero / group_total) * 100 if group_total > 0 else 0.0

 dim = end_idx - start_idx

 # Value statistics
 group_min = group_slice.min
 group_max = group_slice.max
 group_mean = group_slice.mean
 group_std = group_slice.std

 print(f"{group_name:20s} ({dim:3d} dims):")
 print(f" Non-zero: {group_nonzero:6,} / {group_total:6,} ({group_pct:5.1f}%)")
 print(f" Range: [{group_min:8.4f}, {group_max:8.4f}]")
 print(f" Mean: {group_mean:8.4f}")
 print(f" Std: {group_std:8.4f}")
 print

 # Result
 print("=" * 80)
 print("RESULT")
 print("=" * 80)
 print

 target_pct = 70.0

 if non_zero_pct >= target_pct:
 status = "✅ PASS"
 exit_code = 0
 else:
 status = "❌ FAIL"
 exit_code = 1

 print(f"Target: ≥{target_pct:.1f}% non-zero")
 print(f"Actual: {non_zero_pct:.1f}% non-zero")
 print(f"Status: {status}")
 print

 # Specific improvements
 print("=" * 80)
 print("IMPROVEMENTS")
 print("=" * 80)
 print

 print("Before fix: 2.6% non-zero (87% zeros)")
 print(f"After fix: {non_zero_pct:.1f}% non-zero ({100 - non_zero_pct:.1f}% zeros)")
 print(f"Improvement: {non_zero_pct - 2.6:.1f} percentage points")
 print

 return exit_code


if __name__ == '__main__':
 try:
 exit_code = test_state_vector_nonzero
 sys.exit(exit_code)
 except Exception as e:
 print(f"\n❌ ERROR: {e}")
 import traceback
 traceback.print_exc
 sys.exit(1)
