"""Quick performance profiling for ml-common modules"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict

# Imports
from embeddings import extract_symbol_embeddings, extract_temporal_embeddings
from regime import extract_volatility_features, extract_trend_features, extract_time_features
from portfolio import extract_position_features, extract_performance_features
from orderbook import calculate_bid_ask_imbalance, calculate_depth_metrics, calculate_spread_metrics
from cross_asset import extract_correlation_features, extract_spread_features, extract_beta_features

def benchmark(name: str, func, args, target_ms: float, iterations: int = 100):
 """Benchmark a function"""
 times = []
 for _ in range(iterations):
 start = time.perf_counter
 func(*args)
 end = time.perf_counter
 times.append((end - start) * 1000.0)

 avg = np.mean(times)
 status = "✅ PASS" if avg < target_ms else "❌ FAIL"
 print(f"{name:30s}: {avg:7.3f} ms / {target_ms:6.2f} ms {status}")
 return avg < target_ms

print("="*70)
print("ML-COMMON PERFORMANCE PROFILING")
print("="*70)

results = []

# 1. Embeddings (target: <0.5ms)
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
timestamp = datetime.now(timezone.utc)
results.append(benchmark("Embeddings", lambda: (extract_symbol_embeddings(symbols), extract_temporal_embeddings(timestamp)), [], 0.5, 200))

# 2. Regime (target: <2ms)
prices = np.random.randn(168) * 100 + 50000
results.append(benchmark("Regime", lambda: (extract_volatility_features(prices, 24), extract_trend_features(prices, 24), extract_time_features(timestamp)), [], 2.0, 100))

# 3. Portfolio (target: <3ms)
portfolio = {'positions': {'BTCUSDT': 0.5, 'ETHUSDT': 2.0}, 'cash': 5000.0, 'total_value': 100000.0}
price_dict = {'BTCUSDT': 50000, 'ETHUSDT': 3000}
history_data = []
for i in range(168):
 history_data.append({
 'timestamp': timestamp - timedelta(hours=168-i),
 'total_value': 100000 + np.random.randn * 1000,
 'unrealized_pnl': 0,
 'realized_pnl': 0
 })
history = pd.DataFrame(history_data)
results.append(benchmark("Portfolio", lambda: (extract_position_features(portfolio, price_dict), extract_performance_features(history, 168)), [], 3.0, 100))

# 4. Orderbook (target: <10ms)
bids = [(50000-i*10, np.random.uniform(0.5, 2.0)) for i in range(1, 21)]
asks = [(50000+i*10, np.random.uniform(0.5, 2.0)) for i in range(1, 21)]
results.append(benchmark("Orderbook", lambda: (calculate_bid_ask_imbalance(bids, asks), calculate_depth_metrics(bids, asks, 10), calculate_spread_metrics(bids, asks)), [], 10.0, 100))

# 5. Cross-Asset (target: <5ms)
price_dict_full = {
 'BTCUSDT': np.random.randn(168) * 100 + 50000,
 'ETHUSDT': np.random.randn(168) * 50 + 3000,
 'BNBUSDT': np.random.randn(168) * 10 + 400,
 'SOLUSDT': np.random.randn(168) * 5 + 100
}
results.append(benchmark("Cross-Asset", lambda: (extract_correlation_features(price_dict_full, window=24), extract_spread_features(price_dict_full), extract_beta_features(price_dict_full)), [], 5.0, 50))

print("="*70)
total_pass = sum(results)
total = len(results)
print(f"Overall: {total_pass}/{total} tests passed")
if total_pass == total:
 print("✅ ALL PERFORMANCE TARGETS MET!")
else:
 print(f"❌ {total - total_pass} test(s) failed - optimization needed")
print("="*70)
