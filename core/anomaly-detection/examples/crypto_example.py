"""
🪙 Crypto Trading Anomaly Detection Example

Demonstrates how to use the ML anomaly detection system
for cryptocurrency trading data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from statistical import ZScoreDetector, MADDetector, IQRDetector
from ml import IsolationForestDetector
# Note: In production, import would be:
# from ml_anomaly_detection.statistical import ZScoreDetector
# from ml_anomaly_detection.ml import IsolationForestDetector

def generate_crypto_data(n_days=365):
 """Generate synthetic cryptocurrency data with known anomalies."""
 print("📊 Generating synthetic crypto data...")

 np.random.seed(42)
 dates = pd.date_range(start='2024-01-01', periods=n_days*24, freq='H') # Hourly data

 # Base price trend with volatility
 base_trend = np.linspace(40000, 50000, len(dates))
 noise = np.random.normal(0, 1000, len(dates))
 price_changes = np.random.normal(0, 0.02, len(dates))

 prices = []
 current_price = 45000

 for i, change in enumerate(price_changes):
 current_price *= (1 + change)
 current_price += noise[i] * 0.1
 prices.append(current_price)

 prices = np.array(prices)

 # Generate volume data
 base_volume = 1000000
 volume_noise = np.random.exponential(0.5, len(dates))
 volumes = base_volume * (1 + volume_noise)

 # Inject known anomalies
 anomaly_times = [100, 1000, 2500, 5000, 7200] # Known anomaly indices

 for idx in anomaly_times:
 if idx < len(prices):
 # Price spike
 prices[idx:idx+3] *= 1.15 # 15% price increase
 # Volume spike
 volumes[idx:idx+3] *= 8 # 8x volume increase

 # Create DataFrame
 df = pd.DataFrame({
 'timestamp': dates,
 'open': prices * (1 + np.random.normal(0, 0.001, len(prices))),
 'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(prices)))),
 'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(prices)))),
 'close': prices,
 'volume': volumes
 })

 # Add technical indicators
 df['returns'] = df['close'].pct_change.fillna(0)
 df['volatility'] = df['returns'].rolling(24).std.fillna(0) # 24-hour volatility
 df['price_ma'] = df['close'].rolling(24).mean.fillna(df['close'])
 df['volume_ma'] = df['volume'].rolling(24).mean.fillna(df['volume'])

 print(f"✅ Generated {len(df)} data points with {len(anomaly_times)} known anomalies")
 return df, anomaly_times

def compare_detectors(data):
 """Compare different anomaly detection algorithms."""
 print("\\n🔍 Comparing anomaly detection algorithms...")

 # Features to use for detection
 features = ['close', 'volume', 'returns', 'volatility']
 X = data[features].dropna

 detectors = {
 'Z-Score': ZScoreDetector,
 'MAD': MADDetector,
 'IQR': IQRDetector,
 'Isolation Forest': IsolationForestDetector
 }

 results = {}

 for name, detector in detectors.items:
 print(f" 🔧 Training {name} detector...")

 try:
 # Fit detector
 detector.fit(X)

 # Detect anomalies
 labels, scores = detector.detect(X)

 n_anomalies = np.sum(labels)
 anomaly_rate = n_anomalies / len(labels) * 100

 results[name] = {
 'labels': labels,
 'scores': scores,
 'n_anomalies': n_anomalies,
 'anomaly_rate': anomaly_rate,
 'detector': detector
 }

 print(f" ✅ {name}: {n_anomalies} anomalies ({anomaly_rate:.2f}%)")

 except Exception as e:
 print(f" ❌ {name}: Error - {str(e)}")
 results[name] = None

 return results

def analyze_crypto_patterns(data, results):
 """Analyze detected anomaly patterns in crypto data."""
 print("\\n📈 Analyzing crypto anomaly patterns...")

 # Find consensus anomalies (detected by multiple algorithms)
 valid_results = {k: v for k, v in results.items if v is not None}

 if len(valid_results) < 2:
 print(" ⚠️ Need at least 2 working detectors for consensus analysis")
 return

 # Create consensus score
 all_labels = np.array([result['labels'] for result in valid_results.values])
 consensus_score = np.mean(all_labels, axis=0)

 # High consensus anomalies (detected by most algorithms)
 high_consensus = consensus_score >= 0.5
 n_consensus = np.sum(high_consensus)

 print(f" 🎯 Consensus anomalies: {n_consensus} ({n_consensus/len(consensus_score)*100:.2f}%)")

 if n_consensus > 0:
 # Analyze anomaly characteristics
 anomaly_indices = np.where(high_consensus)[0]

 print("\\n 📊 Anomaly Analysis:")
 for i, idx in enumerate(anomaly_indices[:10]): # Show first 10
 row = data.iloc[idx]
 print(f" 🚨 Anomaly {i+1}: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}")
 print(f" 💰 Price: ${row['close']:.2f}")
 print(f" 📊 Volume: {row['volume']:.0f}")
 print(f" 📈 Return: {row['returns']:.3f}")
 print(f" ⚡ Volatility: {row['volatility']:.4f}")

 return consensus_score, high_consensus

def real_time_demo(detector):
 """Demonstrate real-time anomaly detection."""
 print("\\n⚡ Real-time anomaly detection demo...")

 # Simulate real-time data points
 test_cases = [
 {"name": "Normal trading", "data": [48000, 1200000, 0.002, 0.02]},
 {"name": "Price spike", "data": [55000, 1500000, 0.12, 0.08]},
 {"name": "Volume anomaly", "data": [48500, 8000000, 0.01, 0.03]},
 {"name": "Flash crash", "data": [42000, 2000000, -0.15, 0.12]},
 {"name": "Normal trading 2", "data": [48200, 1100000, -0.003, 0.025]}
 ]

 for case in test_cases:
 is_anomaly, score = detector.detect_realtime(case["data"])
 status = "🚨 ANOMALY" if is_anomaly else "✅ NORMAL"
 print(f" {status} - {case['name']}: Score = {score:.3f}")

def visualize_results(data, results, consensus_score):
 """Create visualization of anomaly detection results."""
 print("\\n📊 Creating visualizations...")

 try:
 fig, axes = plt.subplots(3, 1, figsize=(15, 12))

 # Plot 1: Price with anomalies
 ax1 = axes[0]
 ax1.plot(data['timestamp'], data['close'], label='Price', alpha=0.7)

 # Highlight anomalies
 anomaly_mask = consensus_score >= 0.5
 if np.any(anomaly_mask):
 ax1.scatter(data['timestamp'][anomaly_mask], data['close'][anomaly_mask],
 color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)

 ax1.set_title('💰 Bitcoin Price with Detected Anomalies')
 ax1.set_ylabel('Price ($)')
 ax1.legend
 ax1.grid(True, alpha=0.3)

 # Plot 2: Volume with anomalies
 ax2 = axes[1]
 ax2.plot(data['timestamp'], data['volume'], label='Volume', alpha=0.7, color='orange')

 if np.any(anomaly_mask):
 ax2.scatter(data['timestamp'][anomaly_mask], data['volume'][anomaly_mask],
 color='red', s=50, alpha=0.8, label='Anomalies', zorder=5)

 ax2.set_title('📊 Trading Volume with Detected Anomalies')
 ax2.set_ylabel('Volume')
 ax2.legend
 ax2.grid(True, alpha=0.3)

 # Plot 3: Consensus anomaly score
 ax3 = axes[2]
 ax3.plot(data['timestamp'], consensus_score, label='Consensus Score', color='purple')
 ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')

 ax3.set_title('🎯 Consensus Anomaly Score')
 ax3.set_xlabel('Time')
 ax3.set_ylabel('Anomaly Score')
 ax3.legend
 ax3.grid(True, alpha=0.3)

 plt.tight_layout
 plt.savefig('/tmp/anomaly_detection_results.png', dpi=300, bbox_inches='tight')
 print(" 💾 Saved visualization to /tmp/anomaly_detection_results.png")

 except Exception as e:
 print(f" ⚠️ Visualization failed: {e}")

def main:
 """Main example execution."""
 print("🚀 ML Anomaly Detection System - Crypto Trading Example")
 print("=" * 60)

 # Generate sample data
 crypto_data, known_anomalies = generate_crypto_data(30) # 30 days of hourly data

 # Compare different detectors
 detection_results = compare_detectors(crypto_data)

 # Analyze patterns
 if any(r is not None for r in detection_results.values):
 consensus_score, high_consensus = analyze_crypto_patterns(crypto_data, detection_results)

 # Real-time demo with best performing detector
 best_detector = None
 for name, result in detection_results.items:
 if result is not None:
 best_detector = result['detector']
 print(f"\\n⚡ Using {name} for real-time demo...")
 break

 if best_detector:
 real_time_demo(best_detector)

 # Create visualizations
 if 'consensus_score' in locals:
 visualize_results(crypto_data, detection_results, consensus_score)

 print("\\n✅ Example completed successfully!")
 print("\\n💡 Next steps:")
 print(" - Integrate with real crypto exchange APIs (Binance, Coinbase)")
 print(" - Set up real-time alerting system")
 print(" - Deploy to production with monitoring")
 print(" - Tune detectors for specific trading strategies")

if __name__ == "__main__":
 main