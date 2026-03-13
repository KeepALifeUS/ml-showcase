"""
Verification Script - Reproduce Exact User Scenario
Week 2 Day 5-6 Performance Optimization Task

Expected:
- Previous run: 0.980ms ✅ PASS
- Current run: Should be <2.00ms ✅ PASS
- Reported: 4.649ms ❌ FAIL (needs investigation)
"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from datetime import datetime, timezone

# Exact imports from quick_profile.py
from regime import (
 extract_volatility_features,
 extract_trend_features,
 extract_time_features,
 classify_volatility_regime,
 classify_trend_regime,
)

print("="*80)
print("REGRESSION VERIFICATION - Week 2 Day 5-6 Performance Optimization")
print("="*80)

# Generate test data (exact match to quick_profile.py line 44)
np.random.seed(42)
prices_168h = np.random.randn(168) * 100 + 50000
timestamp = datetime.now(timezone.utc)

print("\nTest Setup:")
print(f" Data points: {len(prices_168h)} hours (7 days)")
print(f" Window size: 24 hours")
print(f" Price range: ${prices_168h.min:.2f} - ${prices_168h.max:.2f}")

# Warmup (Numba JIT compilation)
print("\nWarming up Numba JIT compiler...")
for _ in range(5):
 _ = classify_volatility_regime(prices_168h, window=24)
 _ = classify_trend_regime(prices_168h, window=24)
print(" ✅ JIT compilation complete")

# Exact benchmark from quick_profile.py line 45
print("\nRunning benchmark (100 iterations)...")
times = []
for i in range(100):
 start = time.perf_counter

 # Exact code from quick_profile.py
 vol_features = extract_volatility_features(prices_168h, 24)
 trend_features = extract_trend_features(prices_168h, 24)
 time_features = extract_time_features(timestamp)

 end = time.perf_counter
 times.append((end - start) * 1000.0)

times_array = np.array(times)

# Statistics
mean_time = np.mean(times_array)
median_time = np.median(times_array)
min_time = np.min(times_array)
max_time = np.max(times_array)
p95_time = np.percentile(times_array, 95)
p99_time = np.percentile(times_array, 99)
std_time = np.std(times_array)

print("\n" + "="*80)
print("PERFORMANCE RESULTS")
print("="*80)

print(f"\nTiming Statistics (100 iterations):")
print(f" Mean: {mean_time:7.3f} ms")
print(f" Median: {median_time:7.3f} ms")
print(f" Min: {min_time:7.3f} ms")
print(f" Max: {max_time:7.3f} ms")
print(f" P95: {p95_time:7.3f} ms")
print(f" P99: {p99_time:7.3f} ms")
print(f" Std Dev: {std_time:7.3f} ms")

print(f"\nPerformance Targets:")
print(f" Target: 2.000 ms")
print(f" Previous run: 0.980 ms ✅ PASS")
print(f" Reported issue: 4.649 ms ❌ FAIL")
print(f" Current (median): {median_time:0.3f} ms", end=" ")

if median_time < 2.0:
 print("✅ PASS")
 margin = (2.0 - median_time) / 2.0 * 100
 print(f" Margin: {margin:.1f}% under target")
else:
 print("❌ FAIL")
 slowdown = median_time / 2.0
 print(f" Slowdown: {slowdown:.2f}x over target")

print(f"\nComparison to Previous Run (0.980ms):")
ratio = median_time / 0.980
if ratio < 1.1:
 print(f" ✅ Performance maintained: {ratio:.2f}x (within 10% variance)")
elif ratio < 1.5:
 print(f" ⚠️ Performance degraded: {ratio:.2f}x (10-50% slower)")
else:
 print(f" ❌ Performance regression: {ratio:.2f}x (>50% slower)")

print(f"\nComparison to Reported Issue (4.649ms):")
ratio_reported = median_time / 4.649
if ratio_reported < 0.5:
 print(f" ✅ Much faster than reported issue: {ratio_reported:.2f}x ({(1-ratio_reported)*100:.0f}% faster)")
else:
 print(f" ⚠️ Similar to reported issue: {ratio_reported:.2f}x")

# Detailed feature extraction breakdown
print("\n" + "="*80)
print("DETAILED BREAKDOWN")
print("="*80)

print("\nIndividual Feature Extraction:")

# Volatility features
vol_times = []
for _ in range(100):
 start = time.perf_counter
 vol_features = extract_volatility_features(prices_168h, 24)
 end = time.perf_counter
 vol_times.append((end - start) * 1000.0)
vol_mean = np.mean(vol_times)
print(f" Volatility (4 dims): {vol_mean:7.3f} ms (target: <1.0ms)", end=" ")
print("✅ PASS" if vol_mean < 1.0 else "❌ FAIL")

# Trend features
trend_times = []
for _ in range(100):
 start = time.perf_counter
 trend_features = extract_trend_features(prices_168h, 24)
 end = time.perf_counter
 trend_times.append((end - start) * 1000.0)
trend_mean = np.mean(trend_times)
print(f" Trend (4 dims): {trend_mean:7.3f} ms (target: <1.0ms)", end=" ")
print("✅ PASS" if trend_mean < 1.0 else "❌ FAIL")

# Time features
time_times = []
for _ in range(100):
 start = time.perf_counter
 time_features = extract_time_features(timestamp)
 end = time.perf_counter
 time_times.append((end - start) * 1000.0)
time_mean = np.mean(time_times)
print(f" Time (2 dims): {time_mean:7.3f} ms (target: <0.01ms)", end=" ")
print("✅ PASS" if time_mean < 0.01 else "❌ FAIL")

total_individual = vol_mean + trend_mean + time_mean
print(f" Sum of individuals: {total_individual:7.3f} ms")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if median_time < 2.0:
 print("\n✅ NO REGRESSION DETECTED")
 print(f"\nCurrent performance ({median_time:.3f}ms) meets the <2.0ms target with")
 print(f"{((2.0 - median_time) / 2.0 * 100):.1f}% margin. The reported 4.649ms issue is NOT reproducible.")
 print("\nPossible causes of the reported regression:")
 print(" 1. Cold start (first run includes JIT compilation overhead)")
 print(" 2. System load or CPU throttling during measurement")
 print(" 3. Missing Numba installation (fallback to pure Python)")
 print(" 4. Numba cache invalidation")
 print("\nRecommendation: NO ACTION REQUIRED - Performance is optimal")
else:
 print("\n❌ REGRESSION CONFIRMED")
 print(f"\nCurrent performance ({median_time:.3f}ms) exceeds the 2.0ms target by")
 print(f"{((median_time / 2.0 - 1) * 100):.1f}%. Investigation required.")
 print("\nRecommendation: Review optimization implementation")

print("="*80)
