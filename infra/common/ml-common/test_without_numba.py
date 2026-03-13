"""
Test performance WITHOUT Numba to understand the 4.649ms regression scenario
This simulates what happens if Numba is not installed or JIT fails
"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from datetime import datetime, timezone

# Mock Numba to test pure Python performance
import regime.volatility as vol_module
import regime.trend as trend_module

# Temporarily disable Numba
vol_module.HAS_NUMBA = False
trend_module.HAS_NUMBA = False

print("="*80)
print("PERFORMANCE TEST - PURE PYTHON MODE (Numba disabled)")
print("="*80)
print("\n⚠️ Testing fallback performance when Numba is NOT available")
print("This demonstrates the performance impact of missing optimization")

from regime import (
 extract_volatility_features,
 extract_trend_features,
 extract_time_features,
)

# Generate test data
np.random.seed(42)
prices_168h = np.random.randn(168) * 100 + 50000
timestamp = datetime.now(timezone.utc)

# Benchmark pure Python performance
print("\nRunning benchmark (50 iterations, slower than normal)...")
times = []
for i in range(50):
 start = time.perf_counter
 vol_features = extract_volatility_features(prices_168h, 24)
 trend_features = extract_trend_features(prices_168h, 24)
 time_features = extract_time_features(timestamp)
 end = time.perf_counter
 times.append((end - start) * 1000.0)

times_array = np.array(times)
mean_time = np.mean(times_array)
median_time = np.median(times_array)

print("\n" + "="*80)
print("PURE PYTHON PERFORMANCE (No Numba)")
print("="*80)

print(f"\nTiming Statistics:")
print(f" Mean: {mean_time:7.3f} ms")
print(f" Median: {median_time:7.3f} ms")

print(f"\nComparison:")
print(f" Target: 2.000 ms")
print(f" With Numba (optimal): 0.082 ms ✅")
print(f" Without Numba: {median_time:0.3f} ms", end=" ")

if median_time < 2.0:
 print("✅ PASS (still meets target)")
elif median_time < 5.0:
 print("⚠️ SLOW (near reported 4.649ms issue)")
else:
 print("❌ FAIL (exceeds target)")

print(f"\nSlowdown factor: {median_time / 0.082:.1f}x slower without Numba")

if 4.0 < median_time < 5.5:
 print("\n🎯 HYPOTHESIS CONFIRMED:")
 print(f"The reported 4.649ms matches pure Python performance ({median_time:.3f}ms)!")
 print("\nThis means the regression was likely caused by:")
 print(" 1. Numba not being installed (pip install numba)")
 print(" 2. Numba import failing silently (ImportError caught)")
 print(" 3. JIT compilation disabled or failed")
 print("\n✅ SOLUTION: Ensure Numba is installed and working")

print("="*80)
