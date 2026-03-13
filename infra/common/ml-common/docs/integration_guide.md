# ML Common Integration Guide

## Integrating 768-Dimensional State Vector Builder into Your Trading System

**Version:** 1.0
**Last Updated:** 2025-10-11
**Target Audience:** Backend Engineers, ML Engineers, Trading System Developers

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Integration](#basic-integration)
4. [Real-Time Data Pipeline](#real-time-data-pipeline)
5. [WebSocket Integration](#websocket-integration)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring & Observability](#monitoring--observability)
9. [Production Deployment](#production-deployment)
10. [Common Issues & Solutions](#common-issues--solutions)
11. [Complete Examples](#complete-examples)

---

## Quick Start

Get started with ML Common state vector construction in 5 minutes:

### Step 1: Install Package

```bash
cd /home/vlad/ML-Framework/packages/ml-common
pip install -e .
```

### Step 2: Basic Usage

```python
from ml_common.fusion import StateVectorBuilder, StateVectorConfig
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# Initialize builder
builder = StateVectorBuilder(
 config=StateVectorConfig(
 version='v1',
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours=168
 )
)

# Prepare market data (168 hours of 1h candles)
ohlcv_data = {
 'BTCUSDT': pd.DataFrame({
 'open': [...], # 168 values
 'high': [...],
 'low': [...],
 'close': [...],
 'volume': [...],
 'timestamp': [...]
 }),
 # Repeat for ETHUSDT, BNBUSDT, SOLUSDT
}

# Build state vector
state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None, # Optional
 portfolio_state=None, # Optional
 timestamp=datetime.now(timezone.utc)
)

# Output: (168, 768) numpy array
print(f"State vector shape: {state_vector.shape}")
print(f"Construction time: {builder.build_time_ms:.2f}ms")
```

### Step 3: Validate Output

```python
# Validate state vector
assert state_vector.shape == (168, 768), "Invalid shape"
assert state_vector.dtype == np.float32, "Invalid dtype"
assert np.isfinite(state_vector).all, "Contains NaN/Inf"

print("✅ State vector valid!")
```

---

## Installation

### Prerequisites

```bash
# Python 3.10+
python --version

# Required system libraries (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y python3-dev build-essential

# For macOS
brew install python@3.10
```

### Install ML Common

```bash
# From source (development)
cd /home/vlad/ML-Framework/packages/ml-common
pip install -e .[dev]

# Verify installation
python -c "from ml_common.fusion import StateVectorBuilder; print('✅ ML Common installed')"
```

### Install Dependencies

```bash
# Core dependencies
pip install numpy pandas numba scikit-learn

# Optional: Performance optimizations
pip install orjson # Fast JSON parsing

# Optional: Monitoring
pip install prometheus-client structlog
```

---

## Basic Integration

### Step-by-Step Integration

#### 1. Initialize StateVectorBuilder

```python
from ml_common.fusion import StateVectorBuilder, StateVectorConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create configuration
config = StateVectorConfig(
 version='v1',
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours=168,
 use_cache=True,
 cache_size=1000,
 normalize_ohlcv=True,
 normalize_indicators=True,
 fill_method='forward', # Handle missing data
 log_build_time=True,
 warn_slow_build=True,
 slow_threshold_ms=30.0
)

# Initialize builder (do this ONCE at startup)
builder = StateVectorBuilder(config=config)
```

#### 2. Fetch Market Data

```python
import pandas as pd
from datetime import datetime, timedelta, timezone

def fetch_ohlcv_168h(symbols: list[str]) -> dict[str, pd.DataFrame]:
 """
 Fetch 168 hours of 1h OHLCV data for multiple symbols.

 This is a STUB - replace with your actual data fetching logic.
 Examples:
 - CCXT Pro: exchange.fetch_ohlcv(symbol, '1h', limit=168)
 - Database: SELECT * FROM ohlcv WHERE symbol=? AND timestamp >= ?
 - REST API: GET /api/v3/klines?symbol=BTCUSDT&interval=1h&limit=168
 """
 ohlcv_data = {}

 for symbol in symbols:
 # Example: Fetch from your data source
 # df = your_data_source.get_ohlcv(symbol, timeframe='1h', limit=168)

 # Placeholder: Create empty DataFrame with correct structure
 df = pd.DataFrame({
 'open': [0.0] * 168,
 'high': [0.0] * 168,
 'low': [0.0] * 168,
 'close': [0.0] * 168,
 'volume': [0.0] * 168,
 'timestamp': [
 datetime.now(timezone.utc) - timedelta(hours=i)
 for i in range(167, -1, -1)
 ]
 })

 ohlcv_data[symbol] = df

 return ohlcv_data
```

#### 3. Fetch Orderbook Data (Optional)

```python
def fetch_orderbook(symbols: list[str]) -> dict[str, dict]:
 """
 Fetch current orderbook snapshots.

 This is OPTIONAL - if unavailable, state vector builder will use zeros.
 """
 orderbook_data = {}

 for symbol in symbols:
 # Example: Fetch from WebSocket or REST API
 # orderbook = exchange.fetch_order_book(symbol, limit=10)

 # Placeholder structure
 orderbook_data[symbol] = {
 'bids': [
 [50000.0, 1.5], # [price, quantity]
 [49999.0, 2.3],
 # ... up to 10 levels
 ],
 'asks': [
 [50001.0, 1.2],
 [50002.0, 1.8],
 # ... up to 10 levels
 ],
 'timestamp': datetime.now(timezone.utc)
 }

 return orderbook_data
```

#### 4. Get Portfolio State (Optional)

```python
def get_portfolio_state -> dict:
 """
 Get current portfolio state.

 This is OPTIONAL - if unavailable, state vector builder will use zeros.
 """
 # Example: Fetch from your trading system
 portfolio = {
 'positions': {
 'BTCUSDT': 0.5, # 0.5 BTC
 'ETHUSDT': 2.0, # 2.0 ETH
 'BNBUSDT': 10.0, # 10 BNB
 'SOLUSDT': 50.0 # 50 SOL
 },
 'cash': 5000.0, # USD
 'total_value': 100000.0 # USD
 }

 return portfolio
```

#### 5. Build State Vector

```python
def build_state_vector_for_decision:
 """Complete integration example"""

 # 1. Fetch market data
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 ohlcv_data = fetch_ohlcv_168h(symbols)

 # 2. Fetch optional data
 orderbook_data = fetch_orderbook(symbols)
 portfolio_state = get_portfolio_state

 # 3. Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
 )

 # 4. Validate
 assert state_vector.shape == (168, 768)
 assert np.isfinite(state_vector).all

 # 5. Log performance
 print(f"✅ State vector built in {builder.build_time_ms:.2f}ms")

 return state_vector


# Usage
state_vector = build_state_vector_for_decision
```

---

## Real-Time Data Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ REAL-TIME DATA PIPELINE │
├─────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────┐ ┌──────────────┐ │
│ │ Binance │ │ Coinbase │ │
│ │ WebSocket │ │ WebSocket │ Data Sources │
│ └───────┬──────┘ └───────┬──────┘ │
│ │ │ │
│ └──────────┬──────────┘ │
│ │ │
│ ┌──────────▼──────────┐ │
│ │ Data Aggregator │ Normalize & Store │
│ │ (Redis + TimescaleDB) │
│ └──────────┬──────────┘ │
│ │ │
│ ┌──────────▼──────────┐ │
│ │ StateVectorBuilder │ <30ms Construction │
│ └──────────┬──────────┘ │
│ │ │
│ ┌──────────▼──────────┐ │
│ │ Autonomous AI Model│ <100ms Inference │
│ └──────────┬──────────┘ │
│ │ │
│ ┌──────────▼──────────┐ │
│ │ Trading Execution │ Execute Trades │
│ └─────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

#### Data Aggregator Service

```python
import asyncio
import redis
from datetime import datetime, timezone
from collections import deque

class MarketDataAggregator:
 """
 Aggregates real-time market data from multiple exchanges.
 Maintains rolling 168-hour window in memory + Redis cache.
 """

 def __init__(self, symbols: list[str], window_hours: int = 168):
 self.symbols = symbols
 self.window_hours = window_hours

 # In-memory circular buffers (fast access)
 self.ohlcv_buffers = {
 symbol: deque(maxlen=window_hours)
 for symbol in symbols
 }

 # Redis for persistence
 self.redis_client = redis.Redis(
 host='localhost',
 port=6379,
 db=0,
 decode_responses=False
 )

 async def on_candle_update(self, symbol: str, candle: dict):
 """Handle new 1h candle (called every hour)"""

 # Add to in-memory buffer
 self.ohlcv_buffers[symbol].append({
 'open': candle['open'],
 'high': candle['high'],
 'low': candle['low'],
 'close': candle['close'],
 'volume': candle['volume'],
 'timestamp': candle['timestamp']
 })

 # Persist to Redis (for recovery)
 key = f"ohlcv:{symbol}:{candle['timestamp']}"
 self.redis_client.setex(
 key,
 86400 * 8, # 8 days TTL (168h + buffer)
 pickle.dumps(candle)
 )

 def get_ohlcv_168h(self) -> dict[str, pd.DataFrame]:
 """Get current 168-hour OHLCV data for all symbols"""
 import pandas as pd

 ohlcv_data = {}

 for symbol in self.symbols:
 buffer = list(self.ohlcv_buffers[symbol])

 # Ensure we have 168 hours
 if len(buffer) < self.window_hours:
 # Backfill from Redis if needed
 buffer = self._backfill_from_redis(symbol, self.window_hours)

 # Convert to DataFrame
 df = pd.DataFrame(buffer)
 ohlcv_data[symbol] = df

 return ohlcv_data

 def _backfill_from_redis(self, symbol: str, hours: int) -> list[dict]:
 """Backfill missing data from Redis"""
 # Implementation: Query Redis for historical candles
 # ... (omitted for brevity)
 pass
```

#### Trading Loop Integration

```python
import asyncio
from datetime import datetime, timezone

class TradingSystem:
 """Main trading system with ML Common integration"""

 def __init__(self):
 # Initialize components
 self.data_aggregator = MarketDataAggregator(
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 )

 self.state_vector_builder = StateVectorBuilder(
 config=StateVectorConfig(version='v1')
 )

 # Load AI model (placeholder)
 self.ai_model = self._load_ai_model

 # Performance tracking
 self.last_build_time_ms = 0.0
 self.last_inference_time_ms = 0.0

 async def trading_loop(self):
 """Main trading loop (runs every hour)"""

 while True:
 try:
 # 1. Wait for next hour boundary
 await self._wait_for_next_hour

 # 2. Build state vector
 start_time = asyncio.get_event_loop.time
 state_vector = await self._build_state_vector
 self.last_build_time_ms = (asyncio.get_event_loop.time - start_time) * 1000

 # 3. AI inference
 start_time = asyncio.get_event_loop.time
 decision = await self._ai_inference(state_vector)
 self.last_inference_time_ms = (asyncio.get_event_loop.time - start_time) * 1000

 # 4. Execute trade (if confident)
 if decision['confidence'] > 0.7:
 await self._execute_trade(decision)

 # 5. Log performance
 print(f"✅ Cycle complete - Build: {self.last_build_time_ms:.2f}ms, Inference: {self.last_inference_time_ms:.2f}ms")

 except Exception as e:
 print(f"❌ Error in trading loop: {e}")
 # Don't crash - continue to next iteration
 await asyncio.sleep(60)

 async def _build_state_vector(self):
 """Build state vector from current market data"""

 # Fetch market data
 ohlcv_data = self.data_aggregator.get_ohlcv_168h
 orderbook_data = await self._fetch_orderbook
 portfolio_state = await self._get_portfolio

 # Build state vector (synchronous, but fast <30ms)
 state_vector = self.state_vector_builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
 )

 return state_vector

 async def _ai_inference(self, state_vector):
 """Run AI model inference"""
 # Placeholder - replace with actual model
 return {
 'action': 'buy',
 'symbol': 'BTCUSDT',
 'quantity': 0.1,
 'confidence': 0.85
 }

 async def _execute_trade(self, decision):
 """Execute trade based on AI decision"""
 print(f"🚀 Executing trade: {decision}")
 # Implementation: Call exchange API
 pass

 async def _wait_for_next_hour(self):
 """Wait until the next hour boundary"""
 now = datetime.now(timezone.utc)
 next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
 wait_seconds = (next_hour - now).total_seconds
 await asyncio.sleep(wait_seconds)

 def _load_ai_model(self):
 """Load pre-trained AI model"""
 # Placeholder
 return None


# Run trading system
if __name__ == '__main__':
 system = TradingSystem
 asyncio.run(system.trading_loop)
```

---

## WebSocket Integration

### Binance WebSocket Example

```python
import websocket
import json
import threading
from datetime import datetime, timezone

class BinanceWebSocketClient:
 """
 Real-time Binance WebSocket integration for OHLCV and orderbook data.
 """

 def __init__(self, symbols: list[str]):
 self.symbols = symbols
 self.ws = None

 # Data storage
 self.current_candles = {symbol: {} for symbol in symbols}
 self.orderbooks = {symbol: {} for symbol in symbols}

 # Callbacks
 self.on_candle_callback = None
 self.on_orderbook_callback = None

 def connect(self):
 """Connect to Binance WebSocket"""

 # Subscribe to kline (candle) streams
 streams = [
 f"{symbol.lower}@kline_1h"
 for symbol in self.symbols
 ]

 # Add orderbook streams
 streams += [
 f"{symbol.lower}@depth10@100ms"
 for symbol in self.symbols
 ]

 # WebSocket URL
 ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

 # Connect
 self.ws = websocket.WebSocketApp(
 ws_url,
 on_message=self._on_message,
 on_error=self._on_error,
 on_close=self._on_close,
 on_open=self._on_open
 )

 # Run in background thread
 ws_thread = threading.Thread(target=self.ws.run_forever)
 ws_thread.daemon = True
 ws_thread.start

 def _on_message(self, ws, message):
 """Handle incoming WebSocket message"""
 data = json.loads(message)
 stream = data.get('stream', '')

 if '@kline_' in stream:
 self._handle_kline(data['data'])
 elif '@depth' in stream:
 self._handle_orderbook(data['data'])

 def _handle_kline(self, kline_data):
 """Handle kline (candle) update"""
 kline = kline_data['k']
 symbol = kline['s']

 # Update current candle
 self.current_candles[symbol] = {
 'open': float(kline['o']),
 'high': float(kline['h']),
 'low': float(kline['l']),
 'close': float(kline['c']),
 'volume': float(kline['v']),
 'timestamp': datetime.fromtimestamp(kline['t'] / 1000, tz=timezone.utc),
 'is_closed': kline['x'] # True if candle is closed
 }

 # If candle is closed, trigger callback
 if kline['x'] and self.on_candle_callback:
 self.on_candle_callback(symbol, self.current_candles[symbol])

 def _handle_orderbook(self, orderbook_data):
 """Handle orderbook update"""
 symbol = orderbook_data['s']

 # Update orderbook
 self.orderbooks[symbol] = {
 'bids': [
 [float(price), float(qty)]
 for price, qty in orderbook_data['bids']
 ],
 'asks': [
 [float(price), float(qty)]
 for price, qty in orderbook_data['asks']
 ],
 'timestamp': datetime.now(timezone.utc)
 }

 # Trigger callback
 if self.on_orderbook_callback:
 self.on_orderbook_callback(symbol, self.orderbooks[symbol])

 def _on_error(self, ws, error):
 """Handle WebSocket error"""
 print(f"❌ WebSocket error: {error}")

 def _on_close(self, ws, close_status_code, close_msg):
 """Handle WebSocket close"""
 print(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")

 def _on_open(self, ws):
 """Handle WebSocket open"""
 print(f"✅ WebSocket connected")

 def get_current_orderbooks(self) -> dict:
 """Get current orderbook snapshots"""
 return self.orderbooks.copy


# Usage example
if __name__ == '__main__':
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 ws_client = BinanceWebSocketClient(symbols)

 # Set callbacks
 data_aggregator = MarketDataAggregator(symbols)

 ws_client.on_candle_callback = data_aggregator.on_candle_update
 ws_client.on_orderbook_callback = lambda symbol, ob: print(f"Orderbook updated: {symbol}")

 # Connect
 ws_client.connect

 # Keep running
 import time
 while True:
 time.sleep(1)
```

---

## Error Handling

### Graceful Degradation

```python
from ml_common.fusion import StateVectorBuilder, StateVectorConfig
import logging

logger = logging.getLogger(__name__)

def build_state_vector_with_fallback(
 ohlcv_data: dict,
 orderbook_data: dict | None,
 portfolio_state: dict | None,
 builder: StateVectorBuilder
) -> np.ndarray | None:
 """
 Build state vector with comprehensive error handling.

 Returns None if construction fails completely.
 """

 try:
 # Attempt full construction
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
 )

 # Validate
 if not np.isfinite(state_vector).all:
 logger.warning("State vector contains NaN/Inf, replacing...")
 state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=0.0, neginf=0.0)

 return state_vector

 except ValueError as e:
 # Invalid input data
 logger.error(f"Invalid input data: {e}")

 # Try without optional data
 try:
 logger.info("Retrying without optional data...")
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None,
 timestamp=datetime.now(timezone.utc)
 )
 return state_vector

 except Exception as e2:
 logger.error(f"Fallback failed: {e2}")
 return None

 except Exception as e:
 # Unexpected error
 logger.error(f"Unexpected error in state vector construction: {e}", exc_info=True)
 return None


# Usage
state_vector = build_state_vector_with_fallback(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 builder=builder
)

if state_vector is None:
 print("❌ Failed to build state vector, skipping this cycle")
else:
 print(f"✅ State vector built successfully: {state_vector.shape}")
```

### Input Validation

```python
def validate_ohlcv_data(ohlcv_data: dict, expected_symbols: list[str], expected_hours: int) -> bool:
 """Validate OHLCV data structure"""

 # Check symbols
 if set(ohlcv_data.keys) != set(expected_symbols):
 logger.error(f"Symbol mismatch: {list(ohlcv_data.keys)} != {expected_symbols}")
 return False

 # Check each symbol
 for symbol, df in ohlcv_data.items:
 # Check length
 if len(df) != expected_hours:
 logger.error(f"{symbol}: Expected {expected_hours} rows, got {len(df)}")
 return False

 # Check columns
 required_cols = ['open', 'high', 'low', 'close', 'volume']
 missing = set(required_cols) - set(df.columns)
 if missing:
 logger.error(f"{symbol}: Missing columns {missing}")
 return False

 # Check for NaN
 if df[required_cols].isna.any.any:
 logger.warning(f"{symbol}: Contains NaN values")
 # Forward-fill NaN
 df.fillna(method='ffill', inplace=True)

 # Check for negative prices
 if (df[['open', 'high', 'low', 'close']] < 0).any.any:
 logger.error(f"{symbol}: Contains negative prices")
 return False

 return True


# Usage
if not validate_ohlcv_data(ohlcv_data, symbols, 168):
 logger.error("OHLCV data validation failed, aborting")
 return
```

---

## Performance Optimization

### Caching Strategy

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedStateVectorBuilder:
 """
 StateVectorBuilder with intelligent caching.

 Caches intermediate results (indicators, correlations) to speed up construction.
 """

 def __init__(self, config: StateVectorConfig):
 self.builder = StateVectorBuilder(config)

 # Cache settings
 self.cache_ttl_seconds = 60 # 1 minute TTL
 self.cache = {}

 def build_cached(self, ohlcv_data: dict, orderbook_data: dict | None, portfolio_state: dict | None):
 """Build state vector with caching"""

 # Generate cache key (based on latest timestamp)
 cache_key = self._generate_cache_key(ohlcv_data)

 # Check cache
 if cache_key in self.cache:
 cached_vector, cached_time = self.cache[cache_key]

 # Check if cache is still valid
 if datetime.now - cached_time < timedelta(seconds=self.cache_ttl_seconds):
 logger.debug(f"✅ Cache hit: {cache_key}")
 return cached_vector

 # Cache miss - build state vector
 state_vector = self.builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state
 )

 # Update cache
 self.cache[cache_key] = (state_vector, datetime.now)

 # Evict old entries (keep cache size manageable)
 if len(self.cache) > 100:
 self._evict_oldest

 return state_vector

 def _generate_cache_key(self, ohlcv_data: dict) -> str:
 """Generate cache key from OHLCV data"""
 # Use latest timestamp from each symbol
 timestamps = [
 df['timestamp'].iloc[-1].isoformat
 for df in ohlcv_data.values
 ]
 return ','.join(timestamps)

 def _evict_oldest(self):
 """Evict oldest cache entries"""
 # Sort by timestamp
 sorted_entries = sorted(self.cache.items, key=lambda x: x[1][1])

 # Keep only newest 50
 self.cache = dict(sorted_entries[-50:])
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def build_state_vectors_batch(
 ohlcv_data_list: list[dict],
 builder: StateVectorBuilder,
 max_workers: int = 4
) -> list[np.ndarray]:
 """
 Build multiple state vectors in parallel.

 Useful for backtesting or batch inference.
 """

 state_vectors = []

 with ThreadPoolExecutor(max_workers=max_workers) as executor:
 # Submit all tasks
 futures = [
 executor.submit(
 builder.build,
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None
 )
 for ohlcv_data in ohlcv_data_list
 ]

 # Collect results
 for future in as_completed(futures):
 try:
 state_vector = future.result
 state_vectors.append(state_vector)
 except Exception as e:
 logger.error(f"Error building state vector: {e}")
 state_vectors.append(None)

 return state_vectors


# Usage: Backtesting
ohlcv_data_list = [
 fetch_ohlcv_168h_at(timestamp)
 for timestamp in backtest_timestamps
]

state_vectors = build_state_vectors_batch(ohlcv_data_list, builder, max_workers=8)
print(f"✅ Built {len(state_vectors)} state vectors in parallel")
```

---

## Monitoring & Observability

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
state_vector_build_time = Histogram(
 'state_vector_build_duration_seconds',
 'Time to build state vector',
 buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
)

state_vector_build_errors = Counter(
 'state_vector_build_errors_total',
 'Total state vector build errors',
 ['error_type']
)

state_vector_dimension_gauge = Gauge(
 'state_vector_dimensions',
 'Number of dimensions in state vector'
)

orderbook_fallback_rate = Gauge(
 'orderbook_fallback_rate',
 'Rate of orderbook fallback (unavailable data)'
)


# Instrumented builder
class InstrumentedStateVectorBuilder:
 """StateVectorBuilder with Prometheus metrics"""

 def __init__(self, config: StateVectorConfig):
 self.builder = StateVectorBuilder(config)

 @state_vector_build_time.time
 def build(self, ohlcv_data, orderbook_data, portfolio_state):
 """Build state vector with metrics"""

 try:
 state_vector = self.builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state
 )

 # Update metrics
 state_vector_dimension_gauge.set(state_vector.shape[1])

 # Track orderbook fallback
 if orderbook_data is None:
 orderbook_fallback_rate.inc

 return state_vector

 except ValueError as e:
 state_vector_build_errors.labels(error_type='validation').inc
 raise

 except Exception as e:
 state_vector_build_errors.labels(error_type='unknown').inc
 raise


# Start Prometheus HTTP server
from prometheus_client import start_http_server
start_http_server(8000)
```

### Structured Logging

```python
import structlog

# Configure structured logging
structlog.configure(
 processors=[
 structlog.processors.TimeStamper(fmt="iso"),
 structlog.stdlib.add_log_level,
 structlog.processors.JSONRenderer
 ],
 logger_factory=structlog.stdlib.LoggerFactory,
)

logger = structlog.get_logger


# Logged builder
class LoggedStateVectorBuilder:
 """StateVectorBuilder with structured logging"""

 def __init__(self, config: StateVectorConfig):
 self.builder = StateVectorBuilder(config)

 def build(self, ohlcv_data, orderbook_data, portfolio_state):
 """Build state vector with logging"""

 logger.info(
 "state_vector_build_start",
 symbols=list(ohlcv_data.keys),
 has_orderbook=orderbook_data is not None,
 has_portfolio=portfolio_state is not None
 )

 try:
 state_vector = self.builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state
 )

 logger.info(
 "state_vector_build_success",
 shape=state_vector.shape,
 build_time_ms=self.builder.build_time_ms,
 dtype=str(state_vector.dtype)
 )

 return state_vector

 except Exception as e:
 logger.error(
 "state_vector_build_error",
 error=str(e),
 error_type=type(e).__name__
 )
 raise
```

---

## Production Deployment

### Deployment Checklist

```markdown
## Pre-Deployment

- [ ] All unit tests passing (100%)
- [ ] Integration tests passing
- [ ] Performance benchmarks met (<30ms P95)
- [ ] Load testing completed (1000+ req/s)
- [ ] Security audit passed
- [ ] Documentation up-to-date

## Deployment Steps

1. [ ] Deploy to staging environment
2. [ ] Run smoke tests
3. [ ] Monitor for 24 hours
4. [ ] A/B test with 10% traffic
5. [ ] Gradually increase to 100%
6. [ ] Monitor key metrics

## Post-Deployment

- [ ] Verify metrics (build time, errors)
- [ ] Check logs for warnings
- [ ] Test rollback procedure
- [ ] Document any issues
```

### Health Check Endpoint

```python
from flask import Flask, jsonify
import time

app = Flask(__name__)

@app.route('/health')
def health_check:
 """Health check endpoint for load balancer"""

 try:
 # Test state vector builder
 builder = StateVectorBuilder

 # Build test state vector
 test_ohlcv = generate_test_ohlcv_data
 start_time = time.perf_counter

 state_vector = builder.build(
 ohlcv_data=test_ohlcv,
 orderbook_data=None,
 portfolio_state=None
 )

 build_time_ms = (time.perf_counter - start_time) * 1000

 # Check performance
 if build_time_ms > 50: # 50ms threshold
 return jsonify({
 'status': 'degraded',
 'build_time_ms': build_time_ms,
 'message': 'Slow build detected'
 }), 503

 return jsonify({
 'status': 'healthy',
 'build_time_ms': build_time_ms,
 'version': 'v1'
 }), 200

 except Exception as e:
 return jsonify({
 'status': 'unhealthy',
 'error': str(e)
 }), 503


if __name__ == '__main__':
 app.run(host='0.0.0.0', port=8080)
```

### Docker Deployment

```dockerfile
# Dockerfile for ML Common service

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
 build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy ml-common package
COPY packages/ml-common /app/ml-common

# Install Python dependencies
RUN pip install --no-cache-dir -e /app/ml-common

# Copy service code
COPY services/state-vector-service /app/service

# Expose health check port
EXPOSE 8080

# Run service
CMD ["python", "/app/service/main.py"]
```

```bash
# Build and run Docker container
docker build -t ml-common-service:latest .
docker run -d -p 8080:8080 ml-common-service:latest

# Verify health
curl http://localhost:8080/health
```

---

## Common Issues & Solutions

### Issue 1: Slow Build Times (>30ms)

**Symptoms:**

- State vector construction takes >30ms
- P95 latency exceeds threshold

**Solutions:**

```python
# 1. Enable Numba JIT (if not already)
config = StateVectorConfig(use_numba=True) # Default: True

# 2. Enable caching
config = StateVectorConfig(use_cache=True, cache_size=1000)

# 3. Reduce feature complexity (last resort)
# Disable expensive features
config = StateVectorConfig(
 calculate_orderbook=False, # Skip if unavailable
 calculate_cross_asset=False # Skip if not needed
)

# 4. Pre-warm Numba JIT
for _ in range(10):
 builder.build(test_ohlcv_data, None, None)
```

### Issue 2: Missing OHLCV Data

**Symptoms:**

- ValueError: "Expected 168 rows, got 150"

**Solutions:**

```python
# 1. Forward-fill missing data
def forward_fill_ohlcv(df: pd.DataFrame, target_len: int) -> pd.DataFrame:
 """Forward-fill OHLCV data to target length"""
 if len(df) >= target_len:
 return df.iloc[-target_len:]

 # Create empty rows
 last_row = df.iloc[-1]
 missing_rows = target_len - len(df)

 # Forward-fill
 filled_rows = [last_row.copy for _ in range(missing_rows)]
 filled_df = pd.DataFrame(filled_rows)

 return pd.concat([df, filled_df], ignore_index=True)

# Apply
ohlcv_data[symbol] = forward_fill_ohlcv(ohlcv_data[symbol], 168)

# 2. Fallback to shorter window (less ideal)
config = StateVectorConfig(window_hours=100) # Use what's available
```

### Issue 3: NaN/Inf in State Vector

**Symptoms:**

- State vector contains NaN or Inf values
- Model inference fails

**Solutions:**

```python
# 1. Post-process state vector
state_vector = np.nan_to_num(
 state_vector,
 nan=0.0,
 posinf=1e6,
 neginf=-1e6
)

# 2. Check input data quality
for symbol, df in ohlcv_data.items:
 if df.isna.any.any:
 logger.warning(f"{symbol} has NaN values")
 df.fillna(method='ffill', inplace=True)

# 3. Enable validation
config = StateVectorConfig(validate_finite=True) # Raises error on NaN/Inf
```

### Issue 4: Memory Leaks

**Symptoms:**

- Memory usage grows over time
- OOM errors after hours of operation

**Solutions:**

```python
# 1. Clear cache periodically
builder.cache.clear

# 2. Limit cache size
config = StateVectorConfig(cache_size=100) # Smaller cache

# 3. Use context manager
with StateVectorBuilder(config) as builder:
 state_vector = builder.build(...)
# Automatically cleaned up

# 4. Monitor memory
import psutil
process = psutil.Process
print(f"Memory: {process.memory_info.rss / 1024 / 1024:.2f} MB")
```

---

## Complete Examples

### Example 1: Simple Trading Bot

```python
#!/usr/bin/env python3
"""
Simple trading bot with ML Common integration.

Runs every hour, builds state vector, makes trading decision.
"""

import time
from datetime import datetime, timezone
from ml_common.fusion import StateVectorBuilder, StateVectorConfig

def main:
 # Initialize
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 builder = StateVectorBuilder(
 config=StateVectorConfig(
 version='v1',
 symbols=symbols,
 window_hours=168
 )
 )

 print("✅ Trading bot initialized")

 # Trading loop
 while True:
 try:
 print(f"\n{'='*60}")
 print(f"🕐 {datetime.now(timezone.utc).isoformat}")

 # 1. Fetch data
 ohlcv_data = fetch_ohlcv_168h(symbols)
 orderbook_data = fetch_orderbook(symbols)
 portfolio_state = get_portfolio_state

 # 2. Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state
 )

 print(f"✅ State vector: {state_vector.shape}")
 print(f"⏱️ Build time: {builder.build_time_ms:.2f}ms")

 # 3. Make decision (placeholder)
 decision = make_trading_decision(state_vector)
 print(f"🤖 Decision: {decision}")

 # 4. Execute trade
 if decision['action'] != 'hold':
 execute_trade(decision)

 # Wait 1 hour
 print("⏸️ Sleeping 1 hour...")
 time.sleep(3600)

 except KeyboardInterrupt:
 print("\n👋 Shutting down...")
 break

 except Exception as e:
 print(f"❌ Error: {e}")
 time.sleep(60) # Wait 1 minute on error


if __name__ == '__main__':
 main
```

### Example 2: Backtesting Integration

```python
#!/usr/bin/env python3
"""
Backtest trading strategy with ML Common state vectors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ml_common.fusion import StateVectorBuilder, StateVectorConfig

def backtest_strategy(
 start_date: datetime,
 end_date: datetime,
 initial_capital: float = 100000.0
):
 """Run backtest with ML Common state vectors"""

 # Initialize
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
 builder = StateVectorBuilder(config=StateVectorConfig(version='v1', symbols=symbols))

 # Generate hourly timestamps
 timestamps = pd.date_range(start_date, end_date, freq='1h')

 # Portfolio tracking
 portfolio_value = initial_capital
 portfolio_history = []

 # Backtest loop
 for i, timestamp in enumerate(timestamps):
 if i < 168: # Skip first 168 hours (need 7 days of history)
 continue

 # Fetch historical data for this timestamp
 ohlcv_data = fetch_ohlcv_168h_at(timestamp, symbols)

 # Build state vector
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=None,
 portfolio_state=None
 )

 # Strategy decision (replace with your model)
 decision = your_strategy(state_vector)

 # Simulate trade execution
 trade_result = simulate_trade(decision, ohlcv_data, portfolio_value)
 portfolio_value = trade_result['new_portfolio_value']

 # Track performance
 portfolio_history.append({
 'timestamp': timestamp,
 'portfolio_value': portfolio_value,
 'decision': decision
 })

 # Progress
 if i % 168 == 0: # Every week
 print(f"Progress: {timestamp.date} - Portfolio: ${portfolio_value:,.2f}")

 # Calculate metrics
 returns = pd.Series([h['portfolio_value'] for h in portfolio_history]).pct_change
 sharpe_ratio = returns.mean / returns.std * np.sqrt(365 * 24) # Annualized
 total_return = (portfolio_value - initial_capital) / initial_capital

 print(f"\n{'='*60}")
 print(f"📊 Backtest Results")
 print(f"{'='*60}")
 print(f"Total Return: {total_return:.2%}")
 print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
 print(f"Final Portfolio: ${portfolio_value:,.2f}")

 return portfolio_history


if __name__ == '__main__':
 backtest_strategy(
 start_date=datetime(2023, 1, 1),
 end_date=datetime(2024, 1, 1),
 initial_capital=100000.0
 )
```

---

## Additional Resources

- **API Documentation:** `/home/vlad/ML-Framework/packages/ml-common/docs/state_vector_spec.md`
- **Performance Benchmarks:** `/home/vlad/ML-Framework/packages/ml-common/docs/benchmarks.md`
- **Package README:** `/home/vlad/ML-Framework/packages/ml-common/README.md`
- **Usage Examples:** `/home/vlad/ML-Framework/packages/ml-common/USAGE.md`

---

**Document Version:** 1.0
**Last Updated:** 2025-10-11
**Maintainer:** Autonomous AI Development Team
**Support:** <ai-team@ml-framework.dev>

---

**END OF INTEGRATION GUIDE**
