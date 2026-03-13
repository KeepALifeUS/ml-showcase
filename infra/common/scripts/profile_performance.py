"""
Performance Profiling Script for ML-Common Week 2 Modules
Performance Analysis

Profiles:
- StateVectorBuilder.build (CRITICAL - 30ms target)
- Individual feature extractors (orderbook, cross_asset, regime, portfolio, embeddings)
- Hot paths identification
- Memory usage analysis

Usage:
 python scripts/profile_performance.py
 python scripts/profile_performance.py --detailed
 python scripts/profile_performance.py --module state_vector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import cProfile
import pstats
import io
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta
import argparse

# Import modules to profile
from fusion.state_vector import StateVectorBuilder, StateVectorConfig
from orderbook import calculate_bid_ask_imbalance, calculate_depth_metrics, calculate_spread_metrics
from cross_asset import extract_correlation_features, extract_spread_features, extract_beta_features
from regime import extract_volatility_features, extract_trend_features, extract_time_features
from portfolio import extract_position_features, extract_performance_features
from embeddings import extract_symbol_embeddings, extract_temporal_embeddings


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def generate_test_ohlcv(n_hours: int = 168, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
 """Generate realistic OHLCV data for testing"""
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 np.random.seed(42)

 data = {}
 for symbol in symbols:
 base_price = 50000.0 if 'BTC' in symbol else (3000.0 if 'ETH' in symbol else (400.0 if 'BNB' in symbol else 100.0))

 # Generate price series
 returns = np.random.normal(0.0001, 0.02, n_hours)
 prices = np.zeros(n_hours)
 prices[0] = base_price

 for i in range(1, n_hours):
 prices[i] = prices[i-1] * (1 + returns[i])

 # Generate OHLCV
 df_data = []
 for i, close in enumerate(prices):
 volatility = 0.01 + np.random.exponential(0.005)
 high = close * (1 + np.random.uniform(0, volatility))
 low = close * (1 - np.random.uniform(0, volatility))
 open_price = prices[i-1] if i > 0 else close * (1 + np.random.normal(0, 0.001))
 volume = np.random.lognormal(10, 1)

 df_data.append({
 'open': open_price,
 'high': high,
 'low': low,
 'close': close,
 'volume': volume,
 'timestamp': datetime.now(timezone.utc) - timedelta(hours=n_hours-i)
 })

 data[symbol] = pd.DataFrame(df_data)

 return data


def generate_test_orderbook(n_levels: int = 10) -> Dict[str, Dict]:
 """Generate test orderbook data"""
 np.random.seed(42)

 bids = [(50000 - i*10, np.random.uniform(0.1, 2.0)) for i in range(1, n_levels+1)]
 asks = [(50000 + i*10, np.random.uniform(0.1, 2.0)) for i in range(1, n_levels+1)]

 return {'bids': bids, 'asks': asks}


def generate_test_portfolio -> Dict:
 """Generate test portfolio state"""
 return {
 'positions': {
 'BTCUSDT': 0.5,
 'ETHUSDT': 2.0,
 'BNBUSDT': 10.0,
 'SOLUSDT': 50.0
 },
 'cash': 5000.0,
 'total_value': 100000.0,
 'unrealized_pnl': 500.0,
 'realized_pnl': 1000.0
 }


def generate_test_portfolio_history(n_hours: int = 168) -> pd.DataFrame:
 """Generate portfolio history for performance metrics"""
 np.random.seed(42)

 base_value = 100000.0
 returns = np.random.normal(0.001, 0.02, n_hours)

 values = np.zeros(n_hours)
 values[0] = base_value

 for i in range(1, n_hours):
 values[i] = values[i-1] * (1 + returns[i])

 data = []
 for i, val in enumerate(values):
 data.append({
 'timestamp': datetime.now(timezone.utc) - timedelta(hours=n_hours-i),
 'total_value': val,
 'unrealized_pnl': (val - base_value) * 0.7,
 'realized_pnl': (val - base_value) * 0.3
 })

 return pd.DataFrame(data)


# ============================================================================
# PROFILING FUNCTIONS
# ============================================================================

def profile_state_vector_builder(n_iterations: int = 10) -> Dict[str, float]:
 """Profile StateVectorBuilder.build - CRITICAL"""
 print("\n" + "="*70)
 print("PROFILING: StateVectorBuilder (CRITICAL - 30ms target)")
 print("="*70)

 # Setup
 builder = StateVectorBuilder(config=StateVectorConfig(
 log_build_time=False,
 warn_slow_build=False
 ))

 ohlcv_data = generate_test_ohlcv(168)
 orderbook_data = {symbol: generate_test_orderbook for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']}
 portfolio_state = generate_test_portfolio

 # Warmup
 _ = builder.build(ohlcv_data, orderbook_data, portfolio_state)

 # Profile
 times = []
 for i in range(n_iterations):
 start = time.perf_counter
 result = builder.build(ohlcv_data, orderbook_data, portfolio_state)
 end = time.perf_counter

 elapsed_ms = (end - start) * 1000.0
 times.append(elapsed_ms)

 if i == 0:
 print(f"\nOutput shape: {result.shape}")
 print(f"Expected: (168, 768)")
 assert result.shape == (168, 768), f"Shape mismatch!"

 avg_time = np.mean(times)
 std_time = np.std(times)
 min_time = np.min(times)
 max_time = np.max(times)

 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.2f} ms")
 print(f" Std Dev: {std_time:.2f} ms")
 print(f" Min: {min_time:.2f} ms")
 print(f" Max: {max_time:.2f} ms")
 print(f" Target: 30.00 ms")

 status = "✅ PASS" if avg_time < 30.0 else "❌ FAIL"
 print(f"\nStatus: {status}")

 if avg_time >= 30.0:
 print(f"⚠️ Exceeds target by {avg_time - 30.0:.2f} ms")

 return {
 'avg_ms': avg_time,
 'std_ms': std_time,
 'min_ms': min_time,
 'max_ms': max_time,
 'target_ms': 30.0,
 'pass': avg_time < 30.0
 }


def profile_orderbook_features(n_iterations: int = 100) -> Dict[str, float]:
 """Profile orderbook feature extraction (target: <10ms)"""
 print("\n" + "="*70)
 print("PROFILING: Orderbook Features (20 dims, target <10ms)")
 print("="*70)

 bids, asks = [(50000-i*10, np.random.uniform(0.5, 2.0)) for i in range(1, 21)], \
 [(50000+i*10, np.random.uniform(0.5, 2.0)) for i in range(1, 21)]

 times = []
 for _ in range(n_iterations):
 start = time.perf_counter

 imbalance = calculate_bid_ask_imbalance(bids, asks)
 depth = calculate_depth_metrics(bids, asks, levels=10)
 spread = calculate_spread_metrics(bids, asks)

 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg_time = np.mean(times)
 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.3f} ms")
 print(f" Target: 10.000 ms")
 print(f" Status: {'✅ PASS' if avg_time < 10.0 else '❌ FAIL'}")

 return {'avg_ms': avg_time, 'target_ms': 10.0, 'pass': avg_time < 10.0}


def profile_cross_asset_features(n_iterations: int = 50) -> Dict[str, float]:
 """Profile cross-asset feature extraction (target: <5ms)"""
 print("\n" + "="*70)
 print("PROFILING: Cross-Asset Features (20 dims, target <5ms)")
 print("="*70)

 prices = {
 'BTCUSDT': np.random.randn(168) * 100 + 50000,
 'ETHUSDT': np.random.randn(168) * 50 + 3000,
 'BNBUSDT': np.random.randn(168) * 10 + 400,
 'SOLUSDT': np.random.randn(168) * 5 + 100
 }

 times = []
 for _ in range(n_iterations):
 start = time.perf_counter

 corr = extract_correlation_features(prices, window=24)
 spread = extract_spread_features(prices)
 beta = extract_beta_features(prices, market_symbol='BTCUSDT')

 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg_time = np.mean(times)
 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.3f} ms")
 print(f" Target: 5.000 ms")
 print(f" Status: {'✅ PASS' if avg_time < 5.0 else '❌ FAIL'}")

 return {'avg_ms': avg_time, 'target_ms': 5.0, 'pass': avg_time < 5.0}


def profile_regime_features(n_iterations: int = 100) -> Dict[str, float]:
 """Profile regime feature extraction (target: <2ms)"""
 print("\n" + "="*70)
 print("PROFILING: Regime Features (10 dims, target <2ms)")
 print("="*70)

 prices = np.random.randn(168) * 100 + 50000
 timestamp = datetime.now(timezone.utc)

 times = []
 for _ in range(n_iterations):
 start = time.perf_counter

 vol = extract_volatility_features(prices, window=24)
 trend = extract_trend_features(prices, window=24)
 time_feat = extract_time_features(timestamp)

 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg_time = np.mean(times)
 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.3f} ms")
 print(f" Target: 2.000 ms")
 print(f" Status: {'✅ PASS' if avg_time < 2.0 else '❌ FAIL'}")

 return {'avg_ms': avg_time, 'target_ms': 2.0, 'pass': avg_time < 2.0}


def profile_portfolio_features(n_iterations: int = 100) -> Dict[str, float]:
 """Profile portfolio feature extraction (target: <3ms)"""
 print("\n" + "="*70)
 print("PROFILING: Portfolio Features (50 dims, target <3ms)")
 print("="*70)

 portfolio = generate_test_portfolio
 prices = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'BNBUSDT': 400, 'SOLUSDT': 100}
 history = generate_test_portfolio_history(168)

 times = []
 for _ in range(n_iterations):
 start = time.perf_counter

 pos = extract_position_features(portfolio, prices)
 perf = extract_performance_features(history, window_hours=168)

 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg_time = np.mean(times)
 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.3f} ms")
 print(f" Target: 3.000 ms")
 print(f" Status: {'✅ PASS' if avg_time < 3.0 else '❌ FAIL'}")

 return {'avg_ms': avg_time, 'target_ms': 3.0, 'pass': avg_time < 3.0}


def profile_embeddings(n_iterations: int = 200) -> Dict[str, float]:
 """Profile embeddings extraction (target: <0.5ms)"""
 print("\n" + "="*70)
 print("PROFILING: Embeddings (26 dims, target <0.5ms)")
 print("="*70)

 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 timestamp = datetime.now(timezone.utc)

 times = []
 for _ in range(n_iterations):
 start = time.perf_counter

 sym_emb = extract_symbol_embeddings(symbols)
 temp_emb = extract_temporal_embeddings(timestamp)

 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg_time = np.mean(times)
 print(f"\nResults ({n_iterations} iterations):")
 print(f" Average: {avg_time:.3f} ms")
 print(f" Target: 0.500 ms")
 print(f" Status: {'✅ PASS' if avg_time < 0.5 else '❌ FAIL'}")

 return {'avg_ms': avg_time, 'target_ms': 0.5, 'pass': avg_time < 0.5}


# ============================================================================
# DETAILED PROFILING WITH CPROFILE
# ============================================================================

def detailed_profile_state_vector:
 """Detailed cProfile analysis of StateVectorBuilder"""
 print("\n" + "="*70)
 print("DETAILED PROFILING: StateVectorBuilder (cProfile)")
 print("="*70)

 builder = StateVectorBuilder(config=StateVectorConfig(
 log_build_time=False,
 warn_slow_build=False
 ))

 ohlcv_data = generate_test_ohlcv(168)
 orderbook_data = {symbol: generate_test_orderbook for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']}
 portfolio_state = generate_test_portfolio

 # Profile
 profiler = cProfile.Profile
 profiler.enable

 for _ in range(10):
 builder.build(ohlcv_data, orderbook_data, portfolio_state)

 profiler.disable

 # Print stats
 s = io.StringIO
 ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
 ps.print_stats(20) # Top 20 functions

 print(s.getvalue)


# ============================================================================
# MAIN
# ============================================================================

def main:
 parser = argparse.ArgumentParser(description='Profile ML-Common Performance')
 parser.add_argument('--detailed', action='store_true', help='Run detailed cProfile analysis')
 parser.add_argument('--module', type=str, choices=['state_vector', 'orderbook', 'cross_asset', 'regime', 'portfolio', 'embeddings', 'all'], default='all', help='Module to profile')
 parser.add_argument('--iterations', type=int, default=None, help='Number of iterations')

 args = parser.parse_args

 print("="*70)
 print("ML-COMMON PERFORMANCE PROFILING")
 print("Week 2 Day 5-6: Performance Optimization")
 print("Performance Analysis")
 print("="*70)

 results = {}

 if args.module in ['state_vector', 'all']:
 iters = args.iterations or 10
 results['state_vector'] = profile_state_vector_builder(iters)

 if args.detailed:
 detailed_profile_state_vector

 if args.module in ['orderbook', 'all']:
 iters = args.iterations or 100
 results['orderbook'] = profile_orderbook_features(iters)

 if args.module in ['cross_asset', 'all']:
 iters = args.iterations or 50
 results['cross_asset'] = profile_cross_asset_features(iters)

 if args.module in ['regime', 'all']:
 iters = args.iterations or 100
 results['regime'] = profile_regime_features(iters)

 if args.module in ['portfolio', 'all']:
 iters = args.iterations or 100
 results['portfolio'] = profile_portfolio_features(iters)

 if args.module in ['embeddings', 'all']:
 iters = args.iterations or 200
 results['embeddings'] = profile_embeddings(iters)

 # Summary
 print("\n" + "="*70)
 print("SUMMARY")
 print("="*70)

 total_pass = 0
 total_tests = 0

 for module, result in results.items:
 status = "✅ PASS" if result['pass'] else "❌ FAIL"
 print(f"{module:20s}: {result['avg_ms']:7.2f} ms / {result['target_ms']:6.2f} ms {status}")

 if result['pass']:
 total_pass += 1
 total_tests += 1

 print("\n" + "="*70)
 print(f"Overall: {total_pass}/{total_tests} tests passed")

 if total_pass == total_tests:
 print("✅ ALL PERFORMANCE TARGETS MET!")
 else:
 print(f"❌ {total_tests - total_pass} test(s) failed - optimization needed")

 print("="*70)


if __name__ == "__main__":
 main
