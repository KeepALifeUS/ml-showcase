"""
🧪 Test Suite for Anomaly Detectors

Comprehensive tests for all detection algorithms.
"""

import pytest
import numpy as np
import pandas as pd
from ml_anomaly_detection.statistical import ZScoreDetector, MADDetector, IQRDetector
from ml_anomaly_detection.ml import IsolationForestDetector

@pytest.fixture
def sample_crypto_data:
 """Generate sample cryptocurrency data for testing."""
 np.random.seed(42)
 n_samples = 1000

 # Normal price movements
 prices = np.cumsum(np.random.normal(0, 0.02, n_samples)) + 50000

 # Normal volume
 volumes = np.random.exponential(1000000, n_samples)

 # Inject some anomalies
 anomaly_indices = [100, 500, 800]
 prices[anomaly_indices] = prices[anomaly_indices] * 1.2 # 20% price spikes
 volumes[anomaly_indices] = volumes[anomaly_indices] * 5 # 5x volume spikes

 return pd.DataFrame({
 'close': prices,
 'volume': volumes,
 'returns': np.concatenate([[0], np.diff(np.log(prices))])
 })

class TestStatisticalDetectors:
 """Test statistical anomaly detectors."""

 def test_zscore_detector(self, sample_crypto_data):
 detector = ZScoreDetector
 detector.fit(sample_crypto_data[['close', 'volume']])

 labels, scores = detector.detect(sample_crypto_data[['close', 'volume']])

 assert len(labels) == len(sample_crypto_data)
 assert len(scores) == len(sample_crypto_data)
 assert np.sum(labels) > 0 # Should detect some anomalies

 def test_mad_detector(self, sample_crypto_data):
 detector = MADDetector
 detector.fit(sample_crypto_data[['close', 'volume']])

 labels, scores = detector.detect(sample_crypto_data[['close', 'volume']])

 assert len(labels) == len(sample_crypto_data)
 assert np.sum(labels) > 0

 def test_iqr_detector(self, sample_crypto_data):
 detector = IQRDetector
 detector.fit(sample_crypto_data[['close', 'volume']])

 labels, scores = detector.detect(sample_crypto_data[['close', 'volume']])

 assert len(labels) == len(sample_crypto_data)
 assert np.sum(labels) > 0

class TestMLDetectors:
 """Test machine learning anomaly detectors."""

 def test_isolation_forest(self, sample_crypto_data):
 detector = IsolationForestDetector
 detector.fit(sample_crypto_data[['close', 'volume']])

 labels, scores = detector.detect(sample_crypto_data[['close', 'volume']])

 assert len(labels) == len(sample_crypto_data)
 assert len(scores) == len(sample_crypto_data)
 assert np.sum(labels) > 0

 def test_realtime_detection(self, sample_crypto_data):
 detector = IsolationForestDetector
 detector.fit(sample_crypto_data[['close', 'volume']])

 # Test single point detection
 is_anomaly, score = detector.detect_realtime([60000, 5000000]) # High price & volume

 assert isinstance(is_anomaly, bool)
 assert isinstance(score, float)

class TestSystemIntegration:
 """Test system integration and performance."""

 def test_detector_statistics(self, sample_crypto_data):
 detector = ZScoreDetector

 # Before fitting
 stats_before = detector.get_statistics
 assert stats_before["fitted"] == False

 # After fitting
 detector.fit(sample_crypto_data[['close']])
 stats_after = detector.get_statistics
 assert stats_after["fitted"] == True
 assert "mean" in stats_after or "median" in stats_after

 def test_performance_benchmark(self, sample_crypto_data):
 """Basic performance test."""
 detector = IsolationForestDetector

 import time
 start_time = time.time
 detector.fit(sample_crypto_data[['close', 'volume']])
 fit_time = time.time - start_time

 start_time = time.time
 labels, scores = detector.detect(sample_crypto_data[['close', 'volume']])
 detect_time = time.time - start_time

 # Performance assertions (should be fast)
 assert fit_time < 10.0 # Fitting should take less than 10 seconds
 assert detect_time < 1.0 # Detection should take less than 1 second

 print(f"Performance: Fit={fit_time:.3f}s, Detect={detect_time:.3f}s")

if __name__ == "__main__":
 pytest.main([__file__])