# State Vector Specification V1

## 768-Dimensional State Vector Contract for Autonomous AI Trading

**Version:** 1.0
**Status:** PRODUCTION
**Last Updated:** 2025-10-11
**Owner:** Autonomous AI Development Team

---

## Executive Summary

This document defines the **immutable contract** for the 768-dimensional state vector used by the Autonomous AI Crypto Trading System. The state vector is the singular interface between raw market data and the neural network - any change to this specification requires complete model retraining.

### Critical Characteristics

- **Dimensions:** Exactly 768 features
- **Temporal Window:** 168 timesteps (7 days of hourly data)
- **Tensor Shape:** `(168, 768)` float32
- **Performance Target:** <30ms construction time
- **Memory Footprint:** ~512KB per state vector
- **Versioning:** Schema V1 (immutable, use V2 for changes)

### Why 768 Dimensions?

The 768-dimensional vector is optimized for:

1. **Transformer Architecture Compatibility:** Aligns with BERT/GPT embedding dimensions
2. **Information Density:** Captures multi-scale market dynamics without redundancy
3. **Computational Efficiency:** Fits in GPU memory for batch processing
4. **Domain Coverage:** Comprehensive market state representation

---

## Feature Group Breakdown

The 768 dimensions are organized into 10 distinct feature groups, each serving a specific role in market state representation:

```
┌─────────────────────────────────────────────────────────────────┐
│ 768-DIMENSIONAL STATE VECTOR │
├─────────────────────────────────────────────────────────────────┤
│ │
│ [000-019] OHLCV Raw 20 dims 2.6% │
│ [020-179] Technical Indicators 160 dims 20.8% │
│ [180-211] Volume Features 32 dims 4.2% │
│ [212-291] Orderbook Microstructure 80 dims 10.4% │
│ [292-311] Cross-Asset Correlation 20 dims 2.6% │
│ [312-321] Market Regime 10 dims 1.3% │
│ [322-371] Portfolio State 50 dims 6.5% │
│ [372-387] Symbol Embeddings 16 dims 2.1% │
│ [388-397] Temporal Embeddings 10 dims 1.3% │
│ [398-767] Delta History 390 dims 50.8% │
│ │
│ TOTAL: 768 dims 100.0% │
│ │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Group Details

#### 1. OHLCV Raw Features (Dimensions 0-19)

**Purpose:** Current price levels and volume for 4 trading symbols

**Dimension:** 20 (4 symbols × 5 features)

**Features:**

```python
# For each symbol: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT
[0-4]: BTCUSDT [open, high, low, close, volume]
[5-9]: ETHUSDT [open, high, low, close, volume]
[10-14]: BNBUSDT [open, high, low, close, volume]
[15-19]: SOLUSDT [open, high, low, close, volume]
```

**Normalization:** Min-max scaling to [0, 1] range per symbol

**Look-ahead Bias:** NONE (current bar only)

**Update Frequency:** Every hour (aligned with 1h timeframe)

---

#### 2. Technical Indicators (Dimensions 20-179)

**Purpose:** Momentum, trend, volatility, and oscillator indicators

**Dimension:** 160 (40 indicators × 4 symbols)

**Indicators (per symbol):**

```python
# Moving Averages (8 indicators)
SMA(20), SMA(50), SMA(100), SMA(200)
EMA(12), EMA(26), EMA(50), EMA(100)

# Momentum Oscillators (10 indicators)
RSI(14), RSI(7), RSI(28)
MACD(12,26,9) - 3 components: MACD, Signal, Histogram
Stochastic(14,3) - 2 components: %K, %D
Williams %R(14)
CCI(20)
ROC(14)

# Volatility Indicators (8 indicators)
ATR(14), ATR(7)
Bollinger Bands(20,2) - 3 components: Upper, Middle, Lower
Keltner Channels(20) - 2 components: Upper, Lower

# Volume Indicators (6 indicators)
OBV (On-Balance Volume)
VWAP (Volume Weighted Average Price)
MFI(14) (Money Flow Index)
Force Index(13)
Volume SMA(20)
Volume Ratio (current / SMA)

# Trend Indicators (8 indicators)
ADX(14), +DI(14), -DI(14)
Aroon(25) - 2 components: Aroon Up, Aroon Down
Parabolic SAR
Supertrend(10,3)
Ichimoku - 2 components: Tenkan-sen, Kijun-sen

TOTAL: 40 indicators × 4 symbols = 160 dimensions
```

**Normalization:** Z-score normalization (mean=0, std=1)

**Look-ahead Bias:** PROTECTED (indicators use only historical data)

**Performance:** Each indicator <0.2ms calculation time

---

#### 3. Volume Features (Dimensions 180-211)

**Purpose:** Volume dynamics and trading activity patterns

**Dimension:** 32 (8 features × 4 symbols)

**Features (per symbol):**

```python
[0]: Volume change (1h)
[1]: Volume ratio vs. 24h average
[2]: Volume trend (5-period slope)
[3]: Volume acceleration (2nd derivative)
[4]: Cumulative volume delta (CVD)
[5]: Buy/sell volume imbalance
[6]: Large trade indicator (>2σ volume)
[7]: Volume concentration (top 10% bars)
```

**Normalization:** Robust scaling (median=0, IQR=1)

**Look-ahead Bias:** NONE (current and past data only)

**Update Frequency:** Every hour

---

#### 4. Orderbook Microstructure (Dimensions 212-291)

**Purpose:** Market depth and liquidity information

**Dimension:** 80 (20 features × 4 symbols)

**Features (per symbol):**

```python
# Bid-Ask Dynamics (6 features)
[0]: Bid-ask spread (absolute)
[1]: Bid-ask spread (relative %)
[2]: Mid-price
[3]: Weighted mid-price
[4]: Microprice
[5]: Effective spread

# Order Book Imbalance (5 features)
[6]: Level 1 imbalance (best bid/ask)
[7]: Level 5 imbalance (top 5 levels)
[8]: Level 10 imbalance (top 10 levels)
[9]: Volume-weighted imbalance
[10]: Order flow imbalance (1h)

# Depth Metrics (5 features)
[11]: Bid depth (sum of 10 levels)
[12]: Ask depth (sum of 10 levels)
[13]: Total depth
[14]: Depth ratio (bid/ask)
[15]: Depth concentration (top 3 / total)

# Liquidity Metrics (4 features)
[16]: Market impact (for $10K trade)
[17]: Slippage estimate (1%)
[18]: Order book resilience
[19]: Liquidity score (composite)
```

**Normalization:** Symbol-specific scaling (crypto-native scales)

**Look-ahead Bias:** NONE (snapshot data)

**Update Frequency:** Real-time (latest orderbook snapshot)

**Fallback:** If orderbook unavailable, use last valid snapshot

---

#### 5. Cross-Asset Correlation (Dimensions 292-311)

**Purpose:** Multi-symbol relationships and market coupling

**Dimension:** 20

**Features:**

```python
# Pairwise Correlation (10 features)
[0]: BTC-ETH correlation (24h rolling)
[1]: BTC-BNB correlation
[2]: BTC-SOL correlation
[3]: ETH-BNB correlation
[4]: ETH-SOL correlation
[5]: BNB-SOL correlation
[6]: BTC-USD correlation (if available)
[7]: Market-wide correlation (average)
[8]: Correlation dispersion (std)
[9]: Decorrelation events (count in 7d)

# Spreads & Ratios (6 features)
[10]: ETH/BTC ratio
[11]: ETH/BTC ratio change (24h)
[12]: BNB/BTC ratio
[13]: SOL/BTC ratio
[14]: Altcoin index (avg of 3 ratios)
[15]: Spread volatility

# Beta Coefficients (4 features)
[16]: ETH beta to BTC (30d)
[17]: BNB beta to BTC
[18]: SOL beta to BTC
[19]: Market beta (portfolio vs. BTC)
```

**Normalization:** Correlation [-1, 1], ratios log-scaled

**Look-ahead Bias:** PROTECTED (rolling windows, historical data)

**Performance:** <5ms calculation (optimized correlation)

---

#### 6. Market Regime (Dimensions 312-321)

**Purpose:** Current market condition classification

**Dimension:** 10

**Features:**

```python
# Volatility Regime (4 features)
[0]: Realized volatility (24h)
[1]: Volatility percentile (30d rank)
[2]: Volatility regime (low/med/high) one-hot → continuous [0,1]
[3]: Volatility trend (increasing/stable/decreasing)

# Trend Regime (4 features)
[4]: Trend strength (ADX-based)
[5]: Trend direction (bullish/bearish/neutral)
[6]: Trend duration (hours in current trend)
[7]: Trend consistency (% directional bars)

# Time-Based Regime (2 features)
[8]: Trading session (Asian/EU/US) → continuous [0,1]
[9]: Market phase (accumulation/markup/distribution/markdown)
```

**Normalization:** Mixed (continuous for volatility, categorical encoding for regimes)

**Look-ahead Bias:** NONE (current state classification)

**Update Frequency:** Every hour (regime detection)

---

#### 7. Portfolio State (Dimensions 322-371)

**Purpose:** Current positions, PnL, and risk exposure

**Dimension:** 50

**Features:**

```python
# Position Information (16 features)
[0-3]: Position quantities (BTC, ETH, BNB, SOL)
[4-7]: Position values (USD)
[8-11]: Position weights (% of portfolio)
[12-15]: Unrealized PnL per position (USD)

# Portfolio Metrics (14 features)
[16]: Total portfolio value (USD)
[17]: Cash balance (USD)
[18]: Cash ratio (%)
[19]: Total equity
[20]: Margin used
[21]: Available margin
[22]: Leverage ratio
[23]: Net liquidation value
[24]: Unrealized PnL (total)
[25]: Realized PnL (session)
[26]: Total PnL (session)
[27]: PnL percentage (%)
[28]: Return since inception (%)
[29]: Drawdown from peak (%)

# Risk Metrics (10 features)
[30]: Portfolio volatility (annualized)
[31]: Portfolio beta (vs. BTC)
[32]: Sharpe ratio (trailing 30d)
[33]: Sortino ratio
[34]: Maximum drawdown (30d)
[35]: Value at Risk (VaR 95%)
[36]: Expected shortfall (CVaR)
[37]: Risk-adjusted return
[38]: Concentration risk (HHI)
[39]: Correlation risk (avg pairwise)

# Exposure Metrics (10 features)
[40]: Long exposure (USD)
[41]: Short exposure (USD)
[42]: Net exposure (USD)
[43]: Gross exposure (USD)
[44]: Long exposure (%)
[45]: Short exposure (%)
[46]: Net exposure (%)
[47]: Gross exposure (%)
[48]: Sector exposure (if applicable)
[49]: Currency exposure
```

**Normalization:** USD values log-scaled, ratios [0,1], percentages [-1,1]

**Look-ahead Bias:** NONE (current portfolio state)

**Update Frequency:** Real-time (after each trade)

**Security:** Portfolio state is user-specific, isolated per account

---

#### 8. Symbol Embeddings (Dimensions 372-387)

**Purpose:** Learned symbol identity representations

**Dimension:** 16 (4 symbols × 4-dim embeddings)

**Features:**

```python
# Learnable embeddings (initialized randomly, trained end-to-end)
[0-3]: BTCUSDT embedding [btc_0, btc_1, btc_2, btc_3]
[4-7]: ETHUSDT embedding [eth_0, eth_1, eth_2, eth_3]
[8-11]: BNBUSDT embedding [bnb_0, bnb_1, bnb_2, bnb_3]
[12-15]: SOLUSDT embedding [sol_0, sol_1, sol_2, sol_3]
```

**Normalization:** Xavier/Glorot initialization, learned during training

**Look-ahead Bias:** NONE (static per symbol)

**Purpose:** Allow model to learn symbol-specific characteristics

- Market cap differences
- Volatility patterns
- Correlation behavior
- Liquidity profiles

**Training:** These embeddings are neural network parameters, updated via backpropagation

**Inference:** Use trained embeddings from model checkpoint

---

#### 9. Temporal Embeddings (Dimensions 388-397)

**Purpose:** Time-based cyclical features

**Dimension:** 10

**Features:**

```python
# Cyclical Time Features (6 features)
[0]: Hour of day (sin encoding) # 0-23 → sin(2π * hour/24)
[1]: Hour of day (cos encoding) # Preserves continuity
[2]: Day of week (sin encoding) # 0-6 (Monday=0)
[3]: Day of week (cos encoding)
[4]: Week of year (sin encoding) # 1-53 (ISO calendar)
[5]: Week of year (cos encoding)

# Linear Time Features (4 features)
[6]: Month of year (normalized) # 1-12 → 0.0-1.0
[7]: Quarter (categorical) # Q1=0, Q2=0.33, Q3=0.67, Q4=1.0
[8]: Year fraction # Day of year / 365
[9]: Is weekend (binary) # Sat/Sun = 1.0, else 0.0
```

**Normalization:** Sin/cos for cyclical, [0,1] for linear

**Look-ahead Bias:** NONE (current timestamp only)

**Why Cyclical Encoding?**

- Preserves temporal continuity (23:00 → 00:00 are close)
- Neural networks learn periodicities better with sin/cos
- Standard practice in time-series deep learning

**Update Frequency:** Every hour

---

#### 10. Delta History (Dimensions 398-767)

**Purpose:** Historical price movements and momentum

**Dimension:** 390 (78 lookback periods × 5 features)

**Features:**

```python
# 78 historical snapshots (at hours: -1, -2, -3, ..., -78)
# For each snapshot, 5 delta features:

For each of 78 lookback hours:
 [0]: BTC price change (log return)
 [1]: ETH price change (log return)
 [2]: BNB price change (log return)
 [3]: SOL price change (log return)
 [4]: Average volume change (log)

# Example: Most recent history
[398-402]: t-1 deltas (1 hour ago)
[403-407]: t-2 deltas (2 hours ago)
...
[763-767]: t-78 deltas (78 hours ago)
```

**Normalization:** Log returns (natural scale for price changes)

**Look-ahead Bias:** PROTECTED (strictly historical data)

**Why 78 Hours?**

- Covers 3+ days of market history
- Captures intraday and multi-day patterns
- Sufficient for LSTM/Transformer temporal modeling

**Performance:** Pre-computed rolling deltas (<2ms)

---

## Feature Ordering Contract

### Immutability Principle

> **CRITICAL:** Feature ordering is IMMUTABLE once deployed to production.
> Changing feature order = invalidating ALL trained models.

### Why Feature Order Matters

Neural networks learn **position-specific patterns**:

- Feature at index 0 is learned differently than feature at index 100
- Reordering features breaks learned weight matrices
- Even semantically identical reordering requires full retraining

### Schema Versioning Strategy

To evolve the state vector:

```python
# DO NOT modify StateVectorV1
class StateVectorV1:
 TOTAL_DIM = 768 # FROZEN
 # Feature ordering FROZEN

# Create new version for changes
class StateVectorV2:
 TOTAL_DIM = 1024 # NEW dimension
 # New feature ordering
 # Backward compatibility layer optional
```

### Version Migration

```python
# If migrating from V1 to V2:
1. Train new model on V2 schema (from scratch)
2. A/B test V1 vs V2 models
3. Gradual rollout if V2 performs better
4. Maintain V1 for 30 days (rollback capability)
```

### Dimension Validation

The state vector builder enforces exact dimensionality:

```python
assert state_vector.shape == (168, 768), \
 f"Invalid shape {state_vector.shape}, expected (168, 768)"

# Validate each feature group
schema = StateVectorV1
for group_name in schema.feature_map.keys:
 start, end = schema.get_feature_indices(group_name)
 assert end - start == schema.get_feature_dimension(group_name)
```

---

## Performance Characteristics

### Construction Time Budget

**Target:** <30ms per state vector construction

**Breakdown:**

```
Component Time (ms) % of Budget
─────────────────────────────────────────────────────
OHLCV Raw 0.5 1.7%
Technical Indicators 8.0 26.7%
Volume Features 1.2 4.0%
Orderbook Features 2.5 8.3%
Cross-Asset Correlation 3.5 11.7%
Market Regime 1.0 3.3%
Portfolio State 0.8 2.7%
Symbol Embeddings 0.1 0.3%
Temporal Embeddings 0.2 0.7%
Delta History 2.0 6.7%
Overhead (allocation, etc) 10.2 34.0%
─────────────────────────────────────────────────────
TOTAL 30.0 100.0%
```

### Performance Optimization Techniques

1. **Pre-computation:**
 - Technical indicators cached with 1-minute TTL
 - Rolling correlations updated incrementally
 - Delta history maintained in circular buffer

2. **Numba JIT Compilation:**
 - All numerical calculations JIT-compiled
 - 10-100x speedup vs pure Python
 - Minimal overhead after warm-up

3. **Vectorization:**
 - NumPy vectorized operations
 - SIMD instructions utilized
 - Batch processing for multiple symbols

4. **Memory Efficiency:**
 - Float32 precision (vs float64)
 - In-place operations where possible
 - Reuse allocated tensors

### Memory Footprint

**Per State Vector:**

```
Size: 168 timesteps × 768 dims × 4 bytes (float32) = 516,096 bytes ≈ 504 KB
```

**Batch Processing:**

```
Batch of 32 state vectors = 32 × 504 KB ≈ 16 MB
Easily fits in GPU memory for training
```

**Cache Overhead:**

```
Indicator cache (1000 entries): ~10 MB
Orderbook snapshots (4 symbols): ~2 MB
Total memory footprint: ~20-30 MB
```

---

## Integration with Autonomous AI Model

### Model Architecture Compatibility

The 768-dimensional state vector is designed for:

1. **Transformer Models:**
 - 768 dims aligns with BERT/GPT embedding dimension
 - Multi-head attention over 768 features
 - Positional encoding over 168 timesteps

2. **LSTM/GRU Models:**
 - Input: (batch, 168, 768)
 - Hidden state: typically 512-1024 dims
 - Output: trading decision (buy/sell/hold)

3. **CNN-LSTM Hybrids:**
 - 1D CNN over 168 timesteps
 - Feature extraction to 512 dims
 - LSTM for temporal modeling

### Data Pipeline

```python
# Real-time inference pipeline
from ml_common.fusion import StateVectorBuilder

# Initialize builder (once)
builder = StateVectorBuilder(
 config=StateVectorConfig(
 version='v1',
 symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours=168,
 use_cache=True
 )
)

# Every hour (or on-demand)
while True:
 # 1. Fetch market data (from WebSocket / REST API)
 ohlcv_data = fetch_ohlcv_168h # 168 hours of 1h candles
 orderbook_data = fetch_orderbook # Current snapshots
 portfolio_state = get_portfolio # Current positions

 # 2. Build state vector (<30ms)
 state_vector = builder.build(
 ohlcv_data=ohlcv_data,
 orderbook_data=orderbook_data,
 portfolio_state=portfolio_state,
 timestamp=datetime.now(timezone.utc)
 )

 # 3. Model inference (<100ms)
 decision = model.predict(state_vector)

 # 4. Execute trade (if confident)
 if decision['confidence'] > threshold:
 execute_trade(decision)

 # Wait for next signal
 time.sleep(3600) # 1 hour
```

### Training Data Format

```python
# Training dataset structure
training_data = {
 'state_vectors': np.ndarray, # Shape: (N, 168, 768)
 'actions': np.ndarray, # Shape: (N, 4) [buy/sell/hold/size]
 'rewards': np.ndarray, # Shape: (N,) [PnL]
 'timestamps': np.ndarray, # Shape: (N,) [datetime64]
}

# Example: Load training batch
batch_size = 32
indices = np.random.choice(len(training_data['state_vectors']), batch_size)

batch_states = training_data['state_vectors'][indices] # (32, 168, 768)
batch_actions = training_data['actions'][indices] # (32, 4)
batch_rewards = training_data['rewards'][indices] # (32,)
```

---

## Look-Ahead Bias Protection

### Critical Principle

> All features MUST use ONLY data available at decision time.
> Future data leakage = overfitted models = production failure.

### Protection Mechanisms

1. **Strict Timestamp Validation:**

 ```python
 # WRONG: Using future data
 future_return = (price[t+1] - price[t]) / price[t] # ❌

 # CORRECT: Using historical data
 past_return = (price[t] - price[t-1]) / price[t-1] # ✅
 ```

2. **Rolling Window Enforcement:**

 ```python
 # All indicators use strictly causal windows
 sma_20 = calculate_sma(prices[:t+1], period=20) # Up to current time
 ```

3. **Orderbook Snapshot Timing:**

 ```python
 # Orderbook snapshot MUST be from decision timestamp or earlier
 orderbook_snapshot = fetch_orderbook(timestamp=decision_time)
 ```

4. **Portfolio State Consistency:**

 ```python
 # Portfolio state reflects positions BEFORE decision
 # Not after hypothetical trade execution
 ```

### Audit Trail

Every state vector includes metadata:

```python
metadata = {
 'construction_timestamp': datetime, # When vector was built
 'data_cutoff_timestamp': datetime, # Latest data timestamp used
 'orderbook_timestamp': datetime, # Orderbook snapshot time
 'portfolio_timestamp': datetime, # Portfolio state time
}

# Validation
assert metadata['data_cutoff_timestamp'] <= metadata['construction_timestamp']
assert metadata['orderbook_timestamp'] <= metadata['construction_timestamp']
```

---

## Validation & Testing

### Dimension Validation

```python
def validate_state_vector(state_vector: np.ndarray) -> bool:
 """Validate state vector structure"""

 # Shape check
 if state_vector.shape != (168, 768):
 return False

 # Data type check
 if state_vector.dtype != np.float32:
 return False

 # Finite values check
 if not np.isfinite(state_vector).all:
 return False

 # Range checks (feature-specific)
 schema = StateVectorV1

 # OHLCV should be normalized [0, 1]
 ohlcv_start, ohlcv_end = schema.get_feature_indices('ohlcv')
 ohlcv = state_vector[:, ohlcv_start:ohlcv_end]
 if not (0 <= ohlcv).all or not (ohlcv <= 1).all:
 logger.warning("OHLCV out of range")

 # Correlation should be [-1, 1]
 corr_start, corr_end = schema.get_feature_indices('cross_asset')
 corr = state_vector[:, corr_start:corr_end]
 if not (-1 <= corr).all or not (corr <= 1).all:
 logger.warning("Correlation out of range")

 return True
```

### Integration Tests

```python
def test_state_vector_construction:
 """Test state vector builder"""

 # Generate synthetic market data
 ohlcv = generate_ohlcv_data(symbols=4, hours=168)
 orderbook = generate_orderbook_data(symbols=4)
 portfolio = generate_portfolio_state

 # Build state vector
 builder = StateVectorBuilder
 state_vector = builder.build(
 ohlcv_data=ohlcv,
 orderbook_data=orderbook,
 portfolio_state=portfolio
 )

 # Validate
 assert validate_state_vector(state_vector)
 assert builder.build_time_ms < 30.0 # Performance target

 # Check feature groups
 schema = StateVectorV1
 for group_name in schema.feature_map.keys:
 start, end = schema.get_feature_indices(group_name)
 features = state_vector[:, start:end]

 # Non-zero features (at least some data)
 assert (features != 0).any, f"Group {group_name} is all zeros"
```

### Performance Benchmarks

```python
def benchmark_state_vector_construction:
 """Benchmark construction performance"""

 import time

 builder = StateVectorBuilder
 ohlcv = generate_ohlcv_data(symbols=4, hours=168)
 orderbook = generate_orderbook_data(symbols=4)
 portfolio = generate_portfolio_state

 # Warm-up (Numba JIT compilation)
 for _ in range(10):
 builder.build(ohlcv, orderbook, portfolio)

 # Benchmark
 times = []
 for _ in range(100):
 start = time.perf_counter
 builder.build(ohlcv, orderbook, portfolio)
 end = time.perf_counter
 times.append((end - start) * 1000) # ms

 print(f"Mean: {np.mean(times):.2f}ms")
 print(f"P50: {np.percentile(times, 50):.2f}ms")
 print(f"P95: {np.percentile(times, 95):.2f}ms")
 print(f"P99: {np.percentile(times, 99):.2f}ms")

 # Assert performance target
 assert np.percentile(times, 95) < 30.0, "P95 exceeds 30ms target"
```

---

## Error Handling & Fallback Strategies

### Missing Data Handling

```python
# Orderbook unavailable
if orderbook_data is None:
 logger.warning("Orderbook unavailable, using last snapshot")
 orderbook_data = cache.get_last_orderbook

 # If still unavailable, use zeros (model should learn to handle)
 if orderbook_data is None:
 orderbook_features = np.zeros(80, dtype=np.float32)

# Incomplete OHLCV data
if len(ohlcv_data[symbol]) < 168:
 logger.warning(f"Incomplete OHLCV for {symbol}, padding with last value")
 # Forward-fill missing data
 ohlcv_data[symbol] = forward_fill(ohlcv_data[symbol], target_len=168)
```

### Invalid Feature Values

```python
# Handle NaN/Inf in indicators
features = calculate_indicators(prices)

# Replace invalid values
features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

# Clip to reasonable ranges
features = np.clip(features, -10, 10) # For z-scored features
```

### Degraded Mode

```python
# If performance budget exceeded, use simpler features
if builder.build_time_ms > 50:
 logger.warning("Slow build detected, enabling degraded mode")
 builder.config.use_cache = True
 builder.config.use_numba = True
 builder.config.parallel_calculation = True
```

---

## Deployment Considerations

### Production Checklist

- [ ] State vector builder performance <30ms (P95)
- [ ] All feature groups validated (non-zero, correct ranges)
- [ ] Look-ahead bias audit passed (100% compliant)
- [ ] Integration tests passed (100% pass rate)
- [ ] Backward compatibility verified (if updating from previous version)
- [ ] Model checkpoint compatible with V1 schema
- [ ] Monitoring & alerting configured
- [ ] Rollback plan documented

### Monitoring Metrics

```python
# Key metrics to monitor in production
metrics = {
 'state_vector_build_time_ms': histogram([5, 10, 20, 30, 50]),
 'state_vector_build_errors': counter,
 'feature_group_zeros': gauge, # Detect data issues
 'orderbook_fallback_rate': gauge, # Track data availability
 'cache_hit_rate': gauge,
 'model_inference_time_ms': histogram([10, 50, 100, 200]),
}
```

### Alerting Thresholds

```python
alerts = {
 'state_vector_build_time_p95': 40, # ms (30ms + 33% buffer)
 'state_vector_build_errors_rate': 0.01, # 1% error rate
 'orderbook_fallback_rate': 0.10, # 10% fallback rate
 'feature_group_all_zeros': 1, # Any group all zeros
}
```

---

## Appendix A: Full Feature Index

```python
# Complete index of all 768 features (reference only)

FEATURE_INDEX = {
 # OHLCV Raw (0-19)
 0: 'BTCUSDT_open',
 1: 'BTCUSDT_high',
 2: 'BTCUSDT_low',
 3: 'BTCUSDT_close',
 4: 'BTCUSDT_volume',
 # ... (abbreviated for brevity)

 # Technical Indicators (20-179)
 20: 'BTCUSDT_sma_20',
 21: 'BTCUSDT_sma_50',
 # ... (160 indicators)

 # Volume Features (180-211)
 180: 'BTCUSDT_volume_change',
 # ... (32 features)

 # Orderbook (212-291)
 212: 'BTCUSDT_bid_ask_spread',
 # ... (80 features)

 # Cross-Asset (292-311)
 292: 'BTC_ETH_correlation',
 # ... (20 features)

 # Market Regime (312-321)
 312: 'realized_volatility_24h',
 # ... (10 features)

 # Portfolio State (322-371)
 322: 'position_btc_quantity',
 # ... (50 features)

 # Symbol Embeddings (372-387)
 372: 'btc_embed_0',
 # ... (16 features)

 # Temporal Embeddings (388-397)
 388: 'hour_sin',
 # ... (10 features)

 # Delta History (398-767)
 398: 't_minus_1_btc_return',
 # ... (390 features)
}
```

---

## Appendix B: Change Log

### Version 1.0 (2025-10-11)

- Initial schema definition
- 768 dimensions finalized
- 10 feature groups defined
- Performance targets established
- Look-ahead bias protection documented

### Future Versions

**V2 (Planned):**

- Increase to 1024 dimensions
- Add options/derivatives features
- Add sentiment analysis features
- Add on-chain metrics (DeFi)

**Backward Compatibility:**

- V1 models continue to work with V1 state vectors
- V2 will require new model training
- Migration tool will convert V1 → V2 (if needed)

---

## Appendix C: References

**Academic Papers:**

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [Deep Learning for Trading](https://arxiv.org/abs/1811.07522) - Feature engineering for financial ML
- [Market Microstructure](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/market-microstructure) - Orderbook features

**Industry Standards:**

- Enterprise Architecture Patterns
- MLOps Best Practices for Financial Services
- CCXT Pro Market Data Standards

**Internal Documentation:**

- `/docs/integration_guide.md` - Integration instructions
- `/docs/performance_optimization.md` - Performance tuning
- `README.md` - Package overview
- `USAGE.md` - Code examples

---

**Document Status:** PRODUCTION
**Review Cycle:** Quarterly
**Next Review:** 2026-01-11
**Owner:** Autonomous AI Development Team
**Contact:** <ai-team@ml-framework.dev>

---

**END OF SPECIFICATION**
