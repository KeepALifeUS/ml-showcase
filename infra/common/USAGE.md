# 🚀 ML Common - Usage Guide

> **Complete usage examples for all module ML Common package**
> **🎉 NEW: Week 2 - State Vector Construction, Portfolio Tracking, Embeddings**

## 📦 Installation & Setup

```bash
# Install from source
cd packages/ml-common
pip install -e .[dev]

# Basic usage
pip install ml-framework-ml-common

# With all performance optimizations
pip install ml-framework-ml-common[performance]

# With visualization tools
pip install ml-framework-ml-common[visualization]

# Complete installation
pip install ml-framework-ml-common[full]

```

## 🎯 Quick Start Examples

### Technical Indicators

```python
from ml_common.indicators import calculate_sma, calculate_rsi, TechnicalIndicators
import numpy as np

# Sample price data
prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]

# Simple usage
sma_20 = calculate_sma(prices, period=5)
rsi_14 = calculate_rsi(prices, period=5)

print(f"SMA(5): {sma_20:.2f}")
print(f"RSI(5): {rsi_14:.2f}")

# Advanced usage with multiple indicators
indicators = TechnicalIndicators(
 indicators=["sma_5", "ema_5", "rsi_5", "macd"],
 config=IndicatorConfig(use_cache=True)
)

results = indicators.calculate(prices)
print("Multiple indicators:", results)

```

### Data Preprocessing

```python
from ml_common.preprocessing import normalize_data, DataNormalizer
import pandas as pd
import numpy as np

# Sample data
data = np.random.randn(100, 3) * 10 + 50

# Quick normalization
normalized = normalize_data(data, method="robust")

# Advanced preprocessing
normalizer = DataNormalizer(config=NormalizationConfig(
 method="robust",
 handle_outliers=True,
 outlier_method="clip"
))

# Fit and transform
processed_data = normalizer.fit_transform(data)

# Later transform new data
new_data = np.random.randn(20, 3) * 10 + 50
new_processed = normalizer.transform(new_data)

# Inverse transform
original_scale = normalizer.inverse_transform(processed_data)

```

### Performance Metrics

```python
from ml_common.evaluation import (
 calculate_sharpe_ratio, calculate_max_drawdown,
 PerformanceMetrics, backtest_strategy
)
import numpy as np

# Sample returns data
returns = np.random.normal(0.001, 0.02, 252) # Daily returns for 1 year

# Individual metrics
sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
max_dd = calculate_max_drawdown(returns)

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.2%}")

# Comprehensive metrics
metrics = PerformanceMetrics
result = metrics.calculate_all_metrics(returns)

print(f"Total Return: {result.total_return:.2%}")
print(f"Volatility: {result.volatility:.2%}")
print(f"Sortino Ratio: {result.sortino_ratio:.2f}")

```

### Backtesting

```python
from ml_common.evaluation import backtest_strategy, BaseStrategy, BacktestConfig
import pandas as pd
import numpy as np

# Create sample OHLCV data
dates = pd.date_range('2020-01-01', periods=252, freq='D')
np.random.seed(42)

prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 252)))
data = pd.DataFrame({
 'open': prices * (1 + np.random.normal(0, 0.001, 252)),
 'high': prices * (1 + np.random.uniform(0, 0.01, 252)),
 'low': prices * (1 - np.random.uniform(0, 0.01, 252)),
 'close': prices,
 'volume': np.random.lognormal(10, 1, 252)
}, index=dates)

# Simple buy-and-hold strategy
class BuyHoldStrategy(BaseStrategy):
 def generate_signals(self, data, timestamp, portfolio):
 if len(data) == 1: # First day
 return [("main", OrderSide.BUY, 0.8)] # Buy 80%
 return []

# Run backtest
strategy = BuyHoldStrategy
config = BacktestConfig(
 initial_capital=100000,
 commission=0.001,
 position_size=0.8
)

result = backtest_strategy(strategy, data, config)

print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Total Trades: {result.total_trades}")

```

## 📊 Advanced Examples

### High-Performance Technical Analysis

```python
from ml_common.indicators import TechnicalIndicators, IndicatorConfig
import pandas as pd
import time

# Large dataset
prices = generate_price_data(10000) # 10k data points

# High-performance configuration
config = IndicatorConfig(
 use_numba=True,
 use_cache=True,
 parallel_calculation=True,
 cache_size=5000
)

# Multiple indicators
indicators = TechnicalIndicators([
 "sma_20", "sma_50", "sma_200",
 "ema_12", "ema_26",
 "rsi_14", "macd", "bb_upper", "bb_lower",
 "atr_14", "stoch_k", "stoch_d"
], config)

# Benchmark calculation
start_time = time.time
results = indicators.calculate(prices)
end_time = time.time

print(f"Calculated {len(results)} indicators on {len(prices)} data points")
print(f"Execution time: {(end_time - start_time)*1000:.2f}ms")

# Performance stats
stats = indicators.get_performance_stats
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")

```

### Comprehensive Data Processing Pipeline

```python
from ml_common.preprocessing import DataNormalizer, NormalizationConfig
from ml_common.utils import detect_outliers, rolling_window
import numpy as np
import pandas as pd

# Create noisy financial data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=1000, freq='D')

# Price data with outliers
base_prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 1000)))
outlier_indices = np.random.choice(1000, 20)
base_prices[outlier_indices] *= np.random.uniform(1.5, 2.0, 20) # Add outliers

# Multiple features
data = pd.DataFrame({
 'price': base_prices,
 'volume': np.random.lognormal(10, 1, 1000),
 'volatility': rolling_window(np.diff(np.log(base_prices), prepend=base_prices[0]), 20, "std"),
 'momentum': np.concatenate([[0], np.diff(base_prices)])
}, index=dates)

print("Original data statistics:")
print(data.describe)

# Detect outliers
outliers = detect_outliers(data['price'], method="isolation_forest")
print(f"Detected {sum(outliers)} outliers")

# Configure advanced preprocessing
config = NormalizationConfig(
 method="robust",
 handle_outliers=True,
 outlier_method="clip",
 outlier_threshold=3.0,
 handle_missing=True,
 missing_strategy="interpolate"
)

# Process data
normalizer = DataNormalizer(config)
processed_data = normalizer.fit_transform(data)

print("\nProcessed data statistics:")
print(processed_data.describe)

# Get processing report
stats = normalizer.get_stats
print(f"\nProcessing stats:")
print(f"Transform count: {stats['transform_count']}")
print(f"Total samples processed: {stats['total_samples']}")

```

### Multi-Strategy Backtesting Comparison

```python
from ml_common.evaluation import (
 backtest_strategy, PerformanceMetrics, BaseStrategy,
 BacktestConfig, OrderSide
)
from ml_common.indicators import calculate_sma, calculate_rsi
import pandas as pd
import numpy as np

# Generate market data
data = generate_ohlcv_data(500)

# Strategy 1: Moving Average Crossover
class MAStrategy(BaseStrategy):
 def generate_signals(self, data, timestamp, portfolio):
 if len(data) < 50:
 return []

 prices = data['close'].values
 sma_20 = calculate_sma(prices, 20)
 sma_50 = calculate_sma(prices, 50)

 if sma_20 > sma_50 and len(portfolio.positions) == 0:
 return [("main", OrderSide.BUY, 0.5)]
 elif sma_20 < sma_50 and len(portfolio.positions) > 0:
 return [("main", OrderSide.SELL, 1.0)]

 return []

# Strategy 2: RSI Mean Reversion
class RSIStrategy(BaseStrategy):
 def generate_signals(self, data, timestamp, portfolio):
 if len(data) < 20:
 return []

 prices = data['close'].values
 rsi = calculate_rsi(prices, 14)

 if rsi < 30 and len(portfolio.positions) == 0: # Oversold
 return [("main", OrderSide.BUY, 0.3)]
 elif rsi > 70 and len(portfolio.positions) > 0: # Overbought
 return [("main", OrderSide.SELL, 1.0)]

 return []

# Strategy 3: Buy and Hold
class BuyHoldStrategy(BaseStrategy):
 def generate_signals(self, data, timestamp, portfolio):
 if len(data) == 1 and len(portfolio.positions) == 0:
 return [("main", OrderSide.BUY, 0.9)]
 return []

# Backtest configuration
config = BacktestConfig(
 initial_capital=100000,
 commission=0.001,
 position_size=0.5
)

# Run backtests
strategies = {
 "MA Crossover": MAStrategy,
 "RSI Mean Reversion": RSIStrategy,
 "Buy & Hold": BuyHoldStrategy
}

results = {}
for name, strategy in strategies.items:
 print(f"Backtesting {name}...")
 result = backtest_strategy(strategy, data, config)
 results[name] = result

# Compare strategies
metrics = PerformanceMetrics
strategy_returns = {
 name: result.returns
 for name, result in results.items
}

comparison = metrics.compare_strategies(strategy_returns)
print("\nStrategy Comparison:")
print(comparison)

# Detailed analysis
print("\nDetailed Results:")
for name, result in results.items:
 print(f"\n{name}:")
 print(f" Total Return: {result.total_return:.2%}")
 print(f" Sharpe Ratio: {result.sharpe_ratio:.2f}")
 print(f" Max Drawdown: {result.max_drawdown:.2%}")
 print(f" Win Rate: {result.win_rate:.2%}")
 print(f" Total Trades: {result.total_trades}")

```

### Mathematical Utilities

```python
from ml_common.utils import (
 safe_divide, rolling_window, detect_outliers,
 smooth_signal, normalize_signal, parallel_apply
)
import numpy as np

# Sample data with potential issues
prices = np.array([100, 105, 0, 110, np.inf, 95, 108]) # Contains 0 and inf

# Safe mathematical operations
returns = []
for i in range(1, len(prices)):
 # Safe division handles division by zero
 ret = safe_divide(prices[i] - prices[i-1], prices[i-1], default=0.0)
 returns.append(ret)

print("Safe returns:", returns)

# Signal processing
clean_prices = np.array([100, 105, 102, 110, 108, 95, 108, 112, 109])

# Rolling statistics
rolling_mean = rolling_window(clean_prices, window=3, operation="mean")
rolling_std = rolling_window(clean_prices, window=3, operation="std")

print("Rolling mean:", rolling_mean[-5:])
print("Rolling std:", rolling_std[-5:])

# Outlier detection
outliers = detect_outliers(clean_prices, method="zscore", threshold=2.0)
print("Outlier mask:", outliers)

# Signal smoothing
smoothed = smooth_signal(clean_prices, method="exponential", alpha=0.3)
print("Smoothed signal:", smoothed[-5:])

# Signal normalization
normalized = normalize_signal(clean_prices, method="zscore")
print("Normalized signal:", normalized[-5:])

# Parallel processing example
def process_chunk(data_chunk):
 """Process a chunk of data"""
 return np.mean(data_chunk), np.std(data_chunk)

# Large dataset
large_data = [np.random.randn(100) for _ in range(20)]

# Process in parallel
results = parallel_apply(
 large_data,
 process_chunk,
 n_jobs=4,
 use_processes=False
)

print(f"Processed {len(results)} chunks in parallel")

```

---

## 🎉 Week 2 NEW FEATURES

### State Vector Construction (⭐ CRITICAL)

```python
from ml_common.fusion import StateVectorBuilder, StateVectorConfig
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Initialize state vector builder
config = StateVectorConfig(
 version='v1',
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours=168, # 7 days
 use_cache=True,
 normalize_ohlcv=True,
 normalize_indicators=True,
 log_build_time=True
)

builder = StateVectorBuilder(config=config)

# Prepare market data (168 hours of 1h OHLCV data)
ohlcv_data = {
 'BTCUSDT': pd.DataFrame({
 'open': np.random.uniform(50000, 51000, 168),
 'high': np.random.uniform(50100, 51100, 168),
 'low': np.random.uniform(49900, 50900, 168),
 'close': np.random.uniform(50000, 51000, 168),
 'volume': np.random.uniform(100, 200, 168),
 'timestamp': pd.date_range(end=datetime.now(timezone.utc), periods=168, freq='1h')
 }),
 'ETHUSDT': pd.DataFrame({
 'open': np.random.uniform(3000, 3100, 168),
 'high': np.random.uniform(3010, 3110, 168),
 'low': np.random.uniform(2990, 3090, 168),
 'close': np.random.uniform(3000, 3100, 168),
 'volume': np.random.uniform(500, 1000, 168),
 'timestamp': pd.date_range(end=datetime.now(timezone.utc), periods=168, freq='1h')
 }),
 # Add BNBUSDT and SOLUSDT similarly...
}

# Optional: Orderbook data
orderbook_data = {
 'BTCUSDT': {
 'bids': [[50000.0, 1.5], [49999.0, 2.3], [49998.0, 1.2]],
 'asks': [[50001.0, 1.2], [50002.0, 1.8], [50003.0, 2.1]],
 'timestamp': datetime.now(timezone.utc)
 },
 # Add other symbols...
}

# Optional: Portfolio state
portfolio_state = {
 'positions': {
 'BTCUSDT': 0.5, # 0.5 BTC
 'ETHUSDT': 2.0, # 2.0 ETH
 'BNBUSDT': 10.0, # 10 BNB
 'SOLUSDT': 50.0 # 50 SOL
 },
 'cash': 5000.0, # USD
 'total_value': 100000.0 # USD
}

# Build 768-dimensional state vector
state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
)

# Validate output
print(f"✅ State vector shape: {state_vector.shape}") # (168, 768)
print(f"⏱️ Construction time: {builder.build_time_ms:.2f}ms")
print(f"📊 Data type: {state_vector.dtype}") # float32

# Validate dimensions
assert state_vector.shape == (168, 768), "Invalid shape"
assert state_vector.dtype == np.float32, "Invalid dtype"
assert np.isfinite(state_vector).all, "Contains NaN/Inf"

# Access feature groups
from ml_common.fusion.state_vector import StateVectorV1

schema = StateVectorV1

# Get OHLCV features (dims 0-19)
ohlcv_start, ohlcv_end = schema.get_feature_indices('ohlcv')
ohlcv_features = state_vector[:, ohlcv_start:ohlcv_end]
print(f"OHLCV features shape: {ohlcv_features.shape}") # (168, 20)

# Get portfolio features (dims 322-371)
portfolio_start, portfolio_end = schema.get_feature_indices('portfolio')
portfolio_features = state_vector[:, portfolio_start:portfolio_end]
print(f"Portfolio features shape: {portfolio_features.shape}") # (168, 50)

# Performance stats
stats = builder.get_performance_stats
print(f"Performance stats: {stats}")

```

### Portfolio State Tracking

```python
from ml_common.portfolio.state import (
 extract_position_features,
 calculate_position_weights,
 calculate_exposure_metrics,
 calculate_concentration_metrics
)
import numpy as np

# Define portfolio
portfolio = {
 'positions': {
 'BTCUSDT': 0.5, # 0.5 BTC
 'ETHUSDT': 2.0, # 2.0 ETH
 'BNBUSDT': 10.0, # 10 BNB
 'SOLUSDT': 50.0 # 50 SOL
 },
 'cash': 5000.0,
 'total_value': 100000.0
}

# Current market prices
current_prices = {
 'BTCUSDT': 50000.0,
 'ETHUSDT': 3000.0,
 'BNBUSDT': 400.0,
 'SOLUSDT': 100.0
}

# Extract 20-dimensional position features
position_features = extract_position_features(
 portfolio=portfolio,
 current_prices=current_prices,
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
)

print(f"Position features shape: {position_features.shape}") # (20,)
print(f"Position quantities: {position_features[0:4]}")
print(f"Position values: {position_features[4:8]}")
print(f"Position weights: {position_features[8:12]}")
print(f"Exposure metrics: {position_features[12:20]}")

# Calculate position weights
weights = calculate_position_weights(
 positions=portfolio['positions'],
 current_prices=current_prices,
 total_value=portfolio['total_value']
)
print(f"\nPosition weights:")
for symbol, weight in weights.items:
 print(f" {symbol}: {weight:.2%}")

# Calculate exposure metrics
exposure = calculate_exposure_metrics(
 positions=portfolio['positions'],
 current_prices=current_prices,
 cash=portfolio['cash'],
 total_value=portfolio['total_value']
)
print(f"\nExposure metrics:")
print(f" Total exposure: ${exposure['total_exposure']:,.2f}")
print(f" Cash ratio: {exposure['cash_ratio']:.2%}")
print(f" Long exposure: ${exposure['long_exposure']:,.2f}")
print(f" Leverage: {exposure['leverage']:.2f}x")

# Calculate concentration metrics
position_values = [
 portfolio['positions'].get(symbol, 0) * current_prices.get(symbol, 0)
 for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
]

concentration = calculate_concentration_metrics(position_values)
print(f"\nConcentration metrics:")
print(f" Herfindahl Index: {concentration['herfindahl_index']:.4f}")
print(f" Max weight: {concentration['max_weight']:.2%}")
print(f" Diversification ratio: {concentration['diversification_ratio']:.2f}")

```

### Symbol Embeddings

```python
from ml_common.embeddings.symbol import (
 initialize_symbol_embeddings,
 get_symbol_embedding,
 extract_symbol_embeddings,
 load_symbol_embeddings
)
import numpy as np

# Initialize symbol embeddings (random, will be learned during training)
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
embeddings = initialize_symbol_embeddings(
 symbols=symbols,
 embedding_dim=4,
 seed=42 # For reproducibility
)

print("Initialized symbol embeddings:")
for symbol, embedding in embeddings.items:
 print(f" {symbol}: {embedding}")

# Get embedding for a single symbol
btc_embedding = get_symbol_embedding('BTCUSDT', embedding_dim=4)
print(f"\nBTC embedding: {btc_embedding}")

# Extract all embeddings (concatenated for state vector)
all_embeddings = extract_symbol_embeddings(
 symbols=symbols,
 embedding_dim=4
)
print(f"\nConcatenated embeddings shape: {all_embeddings.shape}") # (16,)
print(f"Embeddings: {all_embeddings}")

# Load pre-trained embeddings (after model training)
trained_embeddings = {
 'BTCUSDT': np.array([0.521, -0.342, 0.123, 0.891], dtype=np.float32),
 'ETHUSDT': np.array([0.412, -0.231, 0.654, 0.342], dtype=np.float32),
 'BNBUSDT': np.array([0.234, 0.567, -0.432, 0.123], dtype=np.float32),
 'SOLUSDT': np.array([0.678, 0.123, 0.234, -0.456], dtype=np.float32),
}

load_symbol_embeddings(trained_embeddings)
print("\n✅ Loaded pre-trained symbol embeddings")

# Verify loaded embeddings
btc_embedding_trained = get_symbol_embedding('BTCUSDT')
print(f"Trained BTC embedding: {btc_embedding_trained}")

```

### Temporal Embeddings

```python
from ml_common.embeddings.temporal import (
 extract_temporal_embeddings,
 encode_hour_of_day,
 encode_day_of_week,
 encode_week_of_year
)
from datetime import datetime, timezone

# Extract temporal embeddings for current time
timestamp = datetime(2025, 10, 11, 15, 30, 0, tzinfo=timezone.utc) # Friday 3:30 PM

temporal_features = extract_temporal_embeddings(timestamp)

print(f"Temporal features shape: {temporal_features.shape}") # (10,)
print(f"\nTemporal features:")
print(f" [0-1] Hour (sin, cos): {temporal_features[0]:.3f}, {temporal_features[1]:.3f}")
print(f" [2-3] Day of week (sin, cos): {temporal_features[2]:.3f}, {temporal_features[3]:.3f}")
print(f" [4-5] Week of year (sin, cos): {temporal_features[4]:.3f}, {temporal_features[5]:.3f}")
print(f" [6] Month (normalized): {temporal_features[6]:.3f}")
print(f" [7] Quarter: {temporal_features[7]:.3f}")
print(f" [8] Year fraction: {temporal_features[8]:.3f}")
print(f" [9] Is weekend: {temporal_features[9]:.0f}")

# Individual encodings
hour_sin, hour_cos = encode_hour_of_day(timestamp)
print(f"\nHour encoding (15:30): sin={hour_sin:.3f}, cos={hour_cos:.3f}")

dow_sin, dow_cos = encode_day_of_week(timestamp)
print(f"Day of week encoding (Friday): sin={dow_sin:.3f}, cos={dow_cos:.3f}")

week_sin, week_cos = encode_week_of_year(timestamp)
print(f"Week of year encoding: sin={week_sin:.3f}, cos={week_cos:.3f}")

# Test multiple timestamps
timestamps = [
 datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc), # New Year midnight
 datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc), # Mid-year noon
 datetime(2025, 12, 31, 23, 59, 0, tzinfo=timezone.utc) # Year end
]

print("\n\nTemporal features for different timestamps:")
for ts in timestamps:
 features = extract_temporal_embeddings(ts)
 print(f"{ts.strftime('%Y-%m-%d %H:%M')} → {features}")

```

### Real-Time Trading Integration (Complete Example)

```python
"""
Complete example: Real-time trading bot with ML Common integration.

Demonstrates:
1. Data fetching (OHLCV, orderbook, portfolio)
2. State vector construction
3. AI model inference
4. Trading decision execution
"""

from ml_common.fusion import StateVectorBuilder, StateVectorConfig
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import time

class TradingBot:
 """Simple trading bot with ML Common integration"""

 def __init__(self):
 # Initialize state vector builder
 self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 self.state_builder = StateVectorBuilder(
 config=StateVectorConfig(
 version='v1',
 symbols=self.symbols,
 window_hours=168,
 use_cache=True,
 log_build_time=True
 )
 )

 # Load AI model (placeholder)
 self.ai_model = self._load_ai_model

 print("✅ Trading bot initialized")

 def run(self):
 """Main trading loop (runs every hour)"""

 while True:
 try:
 print(f"\n{'='*60}")
 print(f"🕐 {datetime.now(timezone.utc).isoformat}")

 # 1. Fetch market data
 ohlcv_data = self._fetch_ohlcv_168h
 orderbook_data = self._fetch_orderbook
 portfolio_state = self._get_portfolio_state

 # 2. Build state vector (<30ms)
 start_time = time.perf_counter
 state_vector = self.state_builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
 )
 build_time_ms = (time.perf_counter - start_time) * 1000

 print(f"✅ State vector: {state_vector.shape}")
 print(f"⏱️ Build time: {build_time_ms:.2f}ms")

 # 3. AI inference (<100ms)
 decision = self._ai_inference(state_vector)
 print(f"🤖 AI Decision: {decision}")

 # 4. Execute trade (if confident)
 if decision['confidence'] > 0.7:
 self._execute_trade(decision)
 else:
 print("⏸️ Low confidence, holding position")

 # Wait 1 hour for next iteration
 print("⏸️ Sleeping 1 hour...")
 time.sleep(3600)

 except KeyboardInterrupt:
 print("\n👋 Shutting down trading bot...")
 break

 except Exception as e:
 print(f"❌ Error: {e}")
 time.sleep(60) # Wait 1 minute on error

 def _fetch_ohlcv_168h(self) -> dict:
 """Fetch 168 hours of OHLCV data (STUB - replace with real implementation)"""
 # TODO: Replace with actual data fetching
 # Example: CCXT, Binance API, database query, etc.

 ohlcv_data = {}
 for symbol in self.symbols:
 # Placeholder: Generate random data
 ohlcv_data[symbol] = pd.DataFrame({
 'open': np.random.uniform(1000, 1100, 168),
 'high': np.random.uniform(1010, 1110, 168),
 'low': np.random.uniform(990, 1090, 168),
 'close': np.random.uniform(1000, 1100, 168),
 'volume': np.random.uniform(100, 200, 168),
 'timestamp': pd.date_range(end=datetime.now(timezone.utc), periods=168, freq='1h')
 })

 return ohlcv_data

 def _fetch_orderbook(self) -> dict:
 """Fetch orderbook snapshots (STUB)"""
 # TODO: Replace with WebSocket or REST API
 return None # Optional

 def _get_portfolio_state(self) -> dict:
 """Get current portfolio state (STUB)"""
 # TODO: Fetch from trading system
 return {
 'positions': {'BTCUSDT': 0.5, 'ETHUSDT': 2.0},
 'cash': 5000.0,
 'total_value': 100000.0
 }

 def _ai_inference(self, state_vector: np.ndarray) -> dict:
 """AI model inference (STUB - replace with trained model)"""
 # TODO: Load and run actual PyTorch/TensorFlow model
 # Example: prediction = model(torch.tensor(state_vector))

 # Placeholder decision
 return {
 'action': 'buy',
 'symbol': 'BTCUSDT',
 'quantity': 0.1,
 'confidence': 0.85
 }

 def _execute_trade(self, decision: dict):
 """Execute trade (STUB)"""
 print(f"🚀 Executing trade: {decision}")
 # TODO: Call exchange API
 # Example: exchange.create_order(...)

 def _load_ai_model(self):
 """Load pre-trained AI model (STUB)"""
 # TODO: Load PyTorch/TensorFlow model
 return None


# Run trading bot
if __name__ == '__main__':
 bot = TradingBot
 bot.run

```

## 🔧 Configuration & Optimization

### Performance Optimization

```python
from ml_common.indicators import IndicatorConfig
from ml_common.preprocessing import NormalizationConfig
from ml_common.evaluation import MetricsConfig
import numba

# Check available optimizations
print(f"Numba available: {numba.__version__ if 'numba' in locals else 'No'}")

# Optimal indicator configuration
indicator_config = IndicatorConfig(
 use_numba=True, # Enable JIT compilation
 use_talib=True, # Use TA-Lib if available
 use_cache=True, # Enable result caching
 cache_size=10000, # Large cache
 parallel_calculation=False, # Disable for small datasets
 precision="float64" # Use double precision
)

# Memory-efficient preprocessing
preprocessing_config = NormalizationConfig(
 method="robust", # Robust to outliers
 handle_outliers=True,
 outlier_method="clip",
 preserve_dtypes=True, # Maintain data types
 validate_finite=True # Check for NaN/inf
)

# Risk metrics configuration
metrics_config = MetricsConfig(
 risk_free_rate=0.02,
 periods_per_year=252, # Trading days
 var_confidence=0.05, # 95% VaR
 var_method="historical", # Fast method
 min_periods=30 # Minimum data requirement
)

```

### Error Handling & Logging

```python
import logging
from ml_common.indicators import TechnicalIndicators
from ml_common.preprocessing import DataNormalizer

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug logging for ML Common
logging.getLogger('ml_common').setLevel(logging.DEBUG)

# Robust calculation with error handling
def safe_calculate_indicators(prices, indicators_list):
 """Safely calculate indicators with error handling"""
 try:
 indicators = TechnicalIndicators(indicators_list)
 result = indicators.calculate(prices)

 # Validate results
 for indicator, value in result.items:
 if not np.isfinite(value):
 logging.warning(f"Invalid result for {indicator}: {value}")
 result[indicator] = 0.0

 return result

 except Exception as e:
 logging.error(f"Error calculating indicators: {e}")
 return {indicator: 0.0 for indicator in indicators_list}

# Example usage
prices = generate_price_data(100)
indicators = ["sma_20", "ema_12", "rsi_14"]

result = safe_calculate_indicators(prices, indicators)
print("Safe calculation result:", result)

```

## 📈 Real-World Integration Examples

### Integration with Existing Trading System

```python
# Example integration with crypto trading bot
from ml_common.indicators import TechnicalIndicators
from ml_common.evaluation import backtest_strategy, BaseStrategy
import pandas as pd

class MLEnhancedStrategy(BaseStrategy):
 """Trading strategy enhanced with ML Common indicators"""

 def __init__(self):
 self.indicators = TechnicalIndicators([
 "sma_20", "sma_50", "ema_12", "ema_26",
 "rsi_14", "macd", "bb_upper", "bb_lower", "atr_14"
 ])

 def generate_signals(self, data, timestamp, portfolio):
 if len(data) < 50:
 return []

 # Calculate all indicators
 prices = data['close'].values
 volumes = data['volume'].values
 high = data['high'].values
 low = data['low'].values

 indicators = self.indicators.calculate(
 prices=prices,
 volumes=volumes,
 high=high,
 low=low
 )

 # Multi-indicator strategy logic
 signals = []

 # Trend following: SMA crossover
 if indicators['sma_20'] > indicators['sma_50']:
 trend_score = 1
 else:
 trend_score = -1

 # Mean reversion: RSI
 if indicators['rsi_14'] < 30:
 rsi_score = 1 # Oversold
 elif indicators['rsi_14'] > 70:
 rsi_score = -1 # Overbought
 else:
 rsi_score = 0

 # Momentum: MACD
 macd_score = 1 if indicators['macd'] > 0 else -1

 # Volatility filter: ATR
 atr_filter = indicators['atr_14'] > np.mean(prices[-20:]) * 0.02

 # Combine signals
 total_score = trend_score + rsi_score + macd_score

 # Generate trading signals
 if total_score >= 2 and atr_filter and len(portfolio.positions) == 0:
 signals.append(("main", OrderSide.BUY, 0.5))
 elif total_score <= -2 and len(portfolio.positions) > 0:
 signals.append(("main", OrderSide.SELL, 1.0))

 return signals

# Backtest the enhanced strategy
data = generate_ohlcv_data(252) # 1 year of data
strategy = MLEnhancedStrategy

result = backtest_strategy(strategy, data)
print(f"ML Enhanced Strategy - Sharpe: {result.sharpe_ratio:.2f}")

```

### Data Pipeline Integration

```python
from ml_common.preprocessing import DataNormalizer
from ml_common.utils import detect_outliers, rolling_window
import pandas as pd

class DataProcessor:
 """Complete data processing pipeline"""

 def __init__(self):
 self.normalizer = DataNormalizer
 self.is_fitted = False

 def process_market_data(self, raw_data):
 """Process raw market data"""

 # Step 1: Basic cleaning
 data = raw_data.copy
 data = data.dropna

 # Step 2: Feature engineering
 data['returns'] = data['close'].pct_change
 data['volatility'] = rolling_window(data['returns'], 20, 'std')
 data['volume_ma'] = rolling_window(data['volume'], 10, 'mean')

 # Step 3: Outlier detection
 price_outliers = detect_outliers(data['close'], method="isolation_forest")
 volume_outliers = detect_outliers(data['volume'], method="zscore")

 # Step 4: Handle outliers
 data.loc[price_outliers, 'close'] = data['close'].clip(
 data['close'].quantile(0.01),
 data['close'].quantile(0.99)
 )[price_outliers]

 # Step 5: Normalize features
 feature_columns = ['close', 'volume', 'volatility', 'volume_ma']

 if not self.is_fitted:
 data[feature_columns] = self.normalizer.fit_transform(
 data[feature_columns]
 )
 self.is_fitted = True
 else:
 data[feature_columns] = self.normalizer.transform(
 data[feature_columns]
 )

 return data

 def get_processing_stats(self):
 """Get processing statistics"""
 return self.normalizer.get_stats

# Usage example
processor = DataProcessor

# Process training data
train_data = generate_ohlcv_data(500)
processed_train = processor.process_market_data(train_data)

# Process new data
new_data = generate_ohlcv_data(100)
processed_new = processor.process_market_data(new_data)

print("Processing completed successfully")
print("Stats:", processor.get_processing_stats)

```

## 🧪 Testing & Validation

```python
# Run comprehensive tests
import subprocess
import sys

def run_tests:
 """Run all tests with coverage"""

 # Unit tests
 result = subprocess.run([
 sys.executable, "-m", "pytest",
 "tests/",
 "--cov=src",
 "--cov-report=html",
 "--verbose"
 ], capture_output=True, text=True)

 print("Test results:")
 print(result.stdout)

 if result.returncode == 0:
 print("✅ All tests passed!")
 else:
 print("❌ Some tests failed:")
 print(result.stderr)

# Performance benchmarks
def benchmark_performance:
 """Benchmark key functions"""
 import time

 # Generate test data
 prices = generate_price_data(10000)

 # Benchmark indicators
 indicators = TechnicalIndicators(["sma_20", "ema_12", "rsi_14"])

 start_time = time.time
 for _ in range(100):
 result = indicators.calculate(prices)
 end_time = time.time

 avg_time = (end_time - start_time) / 100 * 1000
 print(f"Average indicator calculation time: {avg_time:.2f}ms")

 # Benchmark normalization
 data = np.random.randn(10000, 5)
 normalizer = DataNormalizer

 start_time = time.time
 for _ in range(10):
 normalizer.fit_transform(data)
 end_time = time.time

 avg_time = (end_time - start_time) / 10 * 1000
 print(f"Average normalization time: {avg_time:.2f}ms")

if __name__ == "__main__":
 print("🧪 Running ML Common tests...")
 run_tests

 print("\n📊 Running performance benchmarks...")
 benchmark_performance

```

## 🔗 Additional Resources

- **API Documentation**: `/docs/api/`
- **Performance Benchmarks**: `/docs/benchmarks/`
- **Example Notebooks**: `/examples/`
- **Contributing Guide**: `CONTRIBUTING.md`
- **Performance Optimization**: `/docs/optimization.md`

---

**Built with ❤️ for the ML-Framework Crypto Trading Bot v5.0**
**Enterprise Architecture • Production-Ready • High-Performance**
