"""Detailed performance profiling for regime module - identify regression"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from datetime import datetime, timezone

# Import individual functions
from regime.volatility import (
 classify_volatility_regime,
 calculate_volatility_percentile,
 calculate_regime_duration,
 calculate_regime_stability,
 extract_volatility_features,
 _fast_realized_volatility,
 _fast_percentile
)

from regime.trend import (
 classify_trend_regime,
 calculate_trend_strength,
 calculate_trend_duration,
 calculate_trend_acceleration,
 extract_trend_features,
 _fast_slope
)

from regime.market_hours import (
 classify_trading_session,
 normalize_day_of_week,
 extract_time_features
)

def micro_benchmark(name: str, func, iterations: int = 1000):
 """Micro-benchmark a function with high precision"""
 # Warmup (Numba JIT compilation)
 for _ in range(10):
 result = func

 # Actual benchmark
 times = []
 for _ in range(iterations):
 start = time.perf_counter
 result = func
 end = time.perf_counter
 times.append((end - start) * 1000.0)

 times_array = np.array(times)
 avg = np.mean(times_array)
 median = np.median(times_array)
 p95 = np.percentile(times_array, 95)
 std = np.std(times_array)

 print(f"{name:45s}: avg={avg:7.4f}ms, median={median:7.4f}ms, p95={p95:7.4f}ms, std={std:7.4f}ms")
 return avg

print("="*100)
print("DETAILED REGIME MODULE PROFILING - REGRESSION ANALYSIS")
print("="*100)

# Generate test data
np.random.seed(42)
prices_168h = np.random.randn(168) * 100 + 50000 # 168 hours
prices_24h = prices_168h[-24:]
timestamp = datetime.now(timezone.utc)

print("\n1. LOW-LEVEL NUMBA FUNCTIONS (should be <0.1ms each)")
print("-" * 100)

# Volatility low-level
returns_168h = np.diff(np.log(prices_168h))
returns_24h = returns_168h[-24:]

avg1 = micro_benchmark("_fast_realized_volatility(24)", lambda: _fast_realized_volatility(returns_24h, 24))
avg2 = micro_benchmark("_fast_percentile(array)", lambda: _fast_percentile(returns_168h, 0.5))

# Trend low-level
avg3 = micro_benchmark("_fast_slope(24)", lambda: _fast_slope(prices_24h))

print("\n2. VOLATILITY FUNCTIONS (individual)")
print("-" * 100)

avg4 = micro_benchmark("classify_volatility_regime(168h, w=24)", lambda: classify_volatility_regime(prices_168h, 24))
avg5 = micro_benchmark("calculate_volatility_percentile(168h)", lambda: calculate_volatility_percentile(prices_168h, 24, 168))
avg6 = micro_benchmark("calculate_regime_duration(168h)", lambda: calculate_regime_duration(prices_168h, 24))
avg7 = micro_benchmark("calculate_regime_stability(168h)", lambda: calculate_regime_stability(prices_168h, 24))
avg8 = micro_benchmark("extract_volatility_features(168h) [ALL 4]", lambda: extract_volatility_features(prices_168h, 24, 168), iterations=500)

vol_total = avg4 + avg5 + avg6 + avg7
print(f"{' → Volatility total (sum of 4)':45s}: {vol_total:7.4f}ms")

print("\n3. TREND FUNCTIONS (individual)")
print("-" * 100)

avg9 = micro_benchmark("classify_trend_regime(168h, w=24)", lambda: classify_trend_regime(prices_168h, 24))
avg10 = micro_benchmark("calculate_trend_strength(168h)", lambda: calculate_trend_strength(prices_168h, 24))
avg11 = micro_benchmark("calculate_trend_duration(168h)", lambda: calculate_trend_duration(prices_168h, 24))
avg12 = micro_benchmark("calculate_trend_acceleration(168h)", lambda: calculate_trend_acceleration(prices_168h, 12, 24))
avg13 = micro_benchmark("extract_trend_features(168h) [ALL 4]", lambda: extract_trend_features(prices_168h, 24), iterations=500)

trend_total = avg9 + avg10 + avg11 + avg12
print(f"{' → Trend total (sum of 4)':45s}: {trend_total:7.4f}ms")

print("\n4. TIME FUNCTIONS")
print("-" * 100)

avg14 = micro_benchmark("classify_trading_session", lambda: classify_trading_session(timestamp))
avg15 = micro_benchmark("normalize_day_of_week", lambda: normalize_day_of_week(timestamp))
avg16 = micro_benchmark("extract_time_features [ALL 2]", lambda: extract_time_features(timestamp))

print("\n5. COMBINED REGIME BENCHMARK (matches quick_profile.py)")
print("-" * 100)

def combined_regime:
 """Matches the exact code from quick_profile.py lines 65-73"""
 vol_regime = classify_volatility_regime(prices_168h, window=24)
 trend_regime = classify_trend_regime(prices_168h, window=24)
 vol_state = calculate_volatility_state(prices_168h, window=24)
 trend_strength = calculate_trend_strength(prices_168h, window=24)
 return (vol_regime, trend_regime, vol_state, trend_strength)

# WAIT - the quick_profile.py uses `calculate_volatility_state` which doesn't exist!
# Let me check what functions are actually called

def combined_regime_correct:
 """Correct version matching available functions"""
 vol_features = extract_volatility_features(prices_168h, 24, 168) # 4 dims
 trend_features = extract_trend_features(prices_168h, 24) # 4 dims
 time_features = extract_time_features(timestamp) # 2 dims
 return (vol_features, trend_features, time_features)

avg17 = micro_benchmark("COMBINED: All regime features (10 dims)", combined_regime_correct, iterations=200)

print("\n" + "="*100)
print("PERFORMANCE ANALYSIS")
print("="*100)

print(f"\nVolatility module total: {vol_total:7.4f}ms (target: <1.0ms)")
print(f"Trend module total: {trend_total:7.4f}ms (target: <1.0ms)")
print(f"Combined regime (10 dims): {avg17:7.4f}ms (target: <2.0ms)")

if avg17 < 2.0:
 print("\n✅ PASS: Performance meets <2ms target")
else:
 print(f"\n❌ FAIL: Performance is {avg17/2.0:.2f}x slower than target")
 print("\nBOTTLENECKS (slowest functions):")

 results = [
 (avg4, "classify_volatility_regime"),
 (avg5, "calculate_volatility_percentile"),
 (avg6, "calculate_regime_duration"),
 (avg7, "calculate_regime_stability"),
 (avg9, "classify_trend_regime"),
 (avg10, "calculate_trend_strength"),
 (avg11, "calculate_trend_duration"),
 (avg12, "calculate_trend_acceleration"),
 ]

 results.sort(reverse=True)
 for i, (time_ms, name) in enumerate(results[:5], 1):
 print(f" {i}. {name:35s}: {time_ms:7.4f}ms")

print("="*100)
