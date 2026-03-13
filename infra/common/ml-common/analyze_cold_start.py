"""
Analyze cold start vs warm performance
Hypothesis: First few runs are much slower due to various factors
"""
import sys
sys.path.insert(0, 'src')

import time
import numpy as np
from datetime import datetime, timezone

from regime import (
 extract_volatility_features,
 extract_trend_features,
 extract_time_features,
)

print("="*80)
print("COLD START vs WARM PERFORMANCE ANALYSIS")
print("="*80)

# Generate test data
np.random.seed(42)
prices_168h = np.random.randn(168) * 100 + 50000
timestamp = datetime.now(timezone.utc)

# Measure each iteration individually
print("\nMeasuring 200 iterations (no warmup)...")
times = []
for i in range(200):
 start = time.perf_counter
 vol_features = extract_volatility_features(prices_168h, 24)
 trend_features = extract_trend_features(prices_168h, 24)
 time_features = extract_time_features(timestamp)
 end = time.perf_counter
 elapsed_ms = (end - start) * 1000.0
 times.append(elapsed_ms)

times_array = np.array(times)

# Analyze different phases
first_run = times[0]
first_5_mean = np.mean(times[:5])
first_10_mean = np.mean(times[:10])
runs_11_20 = np.mean(times[10:20])
runs_21_50 = np.mean(times[20:50])
runs_51_200 = np.mean(times[50:])
warm_phase = np.mean(times[10:]) # After first 10 runs

print("\n" + "="*80)
print("PHASE ANALYSIS")
print("="*80)

print(f"\nCold Start Phase:")
print(f" First run: {first_run:7.3f} ms", end="")
if first_run > 2.0:
 print(f" ⚠️ SLOW (includes JIT compilation)")
else:
 print

print(f" First 5 runs (avg): {first_5_mean:7.3f} ms")
print(f" First 10 runs (avg): {first_10_mean:7.3f} ms")

print(f"\nWarm Phase:")
print(f" Runs 11-20 (avg): {runs_11_20:7.3f} ms")
print(f" Runs 21-50 (avg): {runs_21_50:7.3f} ms")
print(f" Runs 51-200 (avg): {runs_51_200:7.3f} ms")
print(f" Warm average: {warm_phase:7.3f} ms")

# Find outliers
p99 = np.percentile(times, 99)
outliers = [t for t in times if t > p99]

print(f"\nOutlier Analysis:")
print(f" P99 threshold: {p99:7.3f} ms")
print(f" Outlier count: {len(outliers)} / 200")
print(f" Outlier times: ", end="")
if len(outliers) <= 10:
 print(", ".join([f"{t:.3f}ms" for t in outliers[:10]]))
else:
 print(f"[{len(outliers)} values > {p99:.3f}ms]")

# Histogram analysis
print(f"\nDistribution Histogram:")
bins = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]
labels = ['<0.1ms', '0.1-0.2ms', '0.2-0.5ms', '0.5-1.0ms', '1.0-2.0ms', '2.0-5.0ms', '5.0-10ms', '>10ms']

for i in range(len(bins) - 1):
 count = sum(1 for t in times if bins[i] <= t < bins[i+1])
 pct = count / len(times) * 100
 bar = '█' * int(pct / 2)
 print(f" {labels[i]:12s}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if first_run > 4.0:
 print(f"\n🎯 COLD START HYPOTHESIS CONFIRMED:")
 print(f"First run took {first_run:.3f}ms (matches reported 4.649ms regression!)")
 print(f"Warm runs average {warm_phase:.3f}ms (optimal performance)")
 print(f"\nSpeedup: {first_run / warm_phase:.1f}x faster after warmup")
 print("\n✅ ROOT CAUSE: First run includes Numba JIT compilation overhead")
 print("\n📋 RECOMMENDATION:")
 print(" 1. Always include warmup phase in benchmarks (5-10 iterations)")
 print(" 2. Report median/p95 instead of mean (more robust)")
 print(" 3. Numba cache persists across runs, so only first execution is slow")
elif first_5_mean > 2.0:
 print(f"\n⚠️ SLOW STARTUP DETECTED:")
 print(f"First 5 runs average {first_5_mean:.3f}ms")
 print(f"This explains the reported regression")
else:
 print(f"\n✅ NO SIGNIFICANT COLD START OVERHEAD")
 print(f"First run: {first_run:.3f}ms, Warm: {warm_phase:.3f}ms")

print("="*80)
