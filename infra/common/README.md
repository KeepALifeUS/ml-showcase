# ML Common - Unified Machine Learning Utilities

> **Enterprise-grade consolidated ML utilities for Crypto Trading Bot v5.0**
> • High-performance • Type-safe • Production-ready

## Overview

ML Common consolidates **5000+ lines of duplicated mathematical functions** from 38+ ML packages into a single, optimized, enterprise-grade library. Built with architectural patterns for maximum performance and reliability.

**NEW: Week 2 Enhancement - 768-Dimensional State Vector Builder**

The package now includes a **production-ready state vector builder** that constructs 768-dimensional feature vectors for autonomous AI crypto trading. This critical component bridges raw market data and neural networks with <30ms construction time.

### Key Features

- **Technical Indicators**: 40 indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, etc.)
- **Data Preprocessing**: Normalization, scaling, feature engineering, outlier detection
- **Evaluation Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor
- **Backtesting**: Strategy validation, performance analysis, risk metrics
- **Pattern Recognition**: Candlestick patterns, chart patterns, trend detection
- **Order Book Analysis**: Bid-ask imbalance, depth metrics, spread dynamics (20 dims)
- **Cross-Asset Correlation**: Multi-symbol relationships, spreads, beta calculations (20 dims)
- **Market Regime Detection**: Volatility, trend, time-based classification (10 dims)
- **Portfolio State Tracking**: Positions, PnL, risk metrics, exposure analysis (50 dims)
- **Symbol & Temporal Embeddings**: Learnable representations (26 dims total)
- **768-Dim State Vector Builder**: THE CRITICAL autonomous AI integration component
- **High Performance**: Numba acceleration, vectorized operations, <30ms state construction
- **Type Safety**: Full type hints, Pydantic validation, runtime checks
- **Architecture**: Enterprise patterns, observability, monitoring

## Installation

```bash
# Basic installation
pip install ml-framework-ml-common

# With development dependencies
pip install ml-framework-ml-common[dev]

# Full installation with all extras
pip install ml-framework-ml-common[full]

# From source (development)
git clone https://github.com/ml-framework/crypto-trading-bot.git
cd packages/ml-common
pip install -e .[dev]

```

## Quick Start

### Technical Indicators

```python
from ml_common.indicators import TechnicalIndicators, calculate_sma, calculate_rsi

# Simple usage
prices = [100, 102, 101, 103, 105, 104, 106]
sma_20 = calculate_sma(prices, period=20)
rsi_14 = calculate_rsi(prices, period=14)

# Advanced usage with configuration
indicators = TechnicalIndicators(
 indicators=["sma_20", "ema_12", "rsi_14", "macd"],
 config=IndicatorConfig(use_cache=True, parallel_calculation=True)
)

results = indicators.calculate(
 prices=prices,
 volumes=volumes,
 high=high_prices,
 low=low_prices
)
# Returns: {"sma_20": 102.5, "ema_12": 103.1, "rsi_14": 65.4, ...}

```

### Data Preprocessing

```python
from ml_common.preprocessing import DataPreprocessor, normalize_data
import pandas as pd

# Quick normalization
normalized = normalize_data(data, method="z-score")

# Advanced preprocessing pipeline
preprocessor = DataPreprocessor(
 missing_strategy="knn",
 outlier_method="isolation_forest",
 scaling_method="robust"
)

processed_data = preprocessor.fit_transform(raw_data)

```

### Backtesting & Evaluation

```python
from ml_common.evaluation import backtest_strategy, calculate_sharpe_ratio

# Strategy backtesting
results = backtest_strategy(
 signals=trading_signals,
 prices=price_data,
 initial_capital=10000,
 commission=0.001
)

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")

```

## Module Structure

```

ml_common/
├── indicators/ # Technical analysis indicators (Week 1)
│ ├── technical.py # 40 indicators: SMA, EMA, RSI, MACD, ADX, etc.
│ ├── volatility.py # ATR, Bollinger Bands, etc.
│ └── volume.py # OBV, MFI, VWAP, etc.
├── orderbook/ # Order book microstructure (Week 1)
│ ├── imbalance.py # Bid-ask imbalance analysis
│ ├── depth.py # Market depth metrics
│ └── spread.py # Spread dynamics
├── cross_asset/ # Multi-symbol correlation (Week 1)
│ ├── correlation.py # Cross-symbol correlations
│ ├── spreads.py # Inter-asset spread analysis
│ └── beta.py # Beta coefficient calculations
├── regime/ # Market regime classification (Week 1)
│ ├── volatility.py # Volatility regime detection
│ ├── trend.py # Trend classification
│ └── market_hours.py # Time-based regime features
├── portfolio/ # Portfolio state tracking (Week 2)
│ ├── state.py # Position tracking, PnL, exposure
│ ├── performance.py # Historical performance metrics
│ └── risk.py # Risk calculations
├── embeddings/ # Symbol & temporal embeddings (Week 2)
│ ├── symbol.py # Learnable symbol representations (16 dims)
│ └── temporal.py # Time-based cyclic features (10 dims)
├── fusion/ # CRITICAL: State vector builder (Week 2)
│ ├── state_vector.py # 768-dim state vector constructor
│ └── windowing.py # Rolling window management
├── preprocessing/ # Data preprocessing utilities (Core)
│ ├── normalization.py # Scaling and normalization
│ ├── feature_engineering.py # Feature creation
│ └── data_cleaning.py # Outlier detection, missing values
├── evaluation/ # Performance evaluation (Core)
│ ├── metrics.py # Sharpe, Sortino, Calmar ratios
│ └── backtesting.py # Strategy validation
├── utils/ # Utility functions (Core)
│ ├── math_utils.py # Mathematical helpers
│ ├── time_series.py # Time series utilities
│ └── data_loader.py # Data loading helpers
└── patterns/ # Pattern recognition (Core)
 ├── candlestick.py # Candlestick patterns
 └── chart_patterns.py # Chart pattern detection

```

## Architecture

ML Common implements enterprise patterns:

### Core Principles

- **Performance First**: Numba JIT compilation, vectorized operations
- **Type Safety**: Full type hints, runtime validation
- **Observability**: Comprehensive logging, metrics, tracing
- **Reliability**: Error handling, graceful degradation
- **Scalability**: Modular design, efficient memory usage

### Enterprise Features

```python
from ml_common.indicators import TechnicalIndicators
from ml_common.utils import setup_logging, configure_monitoring

# Enterprise configuration
setup_logging(level="INFO", format="structured")
configure_monitoring(enable_metrics=True, enable_tracing=True)

# High-performance calculation with monitoring
indicators = TechnicalIndicators(
 indicators=["sma_20", "ema_12", "rsi_14"],
 config=IndicatorConfig(
 use_cache=True,
 parallel_calculation=True,
 enable_monitoring=True
 )
)

```

## Performance

ML Common is optimized for high-frequency trading:

- **Numba JIT**: Up to 100x speedup for numerical calculations
- **Vectorization**: Batch processing for multiple assets
- **Caching**: Intelligent caching for repeated calculations
- **Memory Efficiency**: Optimized data structures
- **State Vector Construction**: <30ms for 768 dims × 168 timesteps

### Benchmarks (ALL TARGETS MET)

```

Core Indicators (1000 data points):
- SMA calculation: ~0.05ms (vs 2.1ms pure Python)
- RSI calculation: ~0.12ms (vs 8.7ms pure Python)
- MACD calculation: ~0.18ms (vs 12.3ms pure Python)

Week 1 Modules (Performance Targets):
- Orderbook features: 0.030ms (target: 10.0ms)
- Cross-asset correlation: 3.443ms (target: 5.0ms)
- Regime detection: 0.080ms (target: 2.0ms)

Week 2 Modules (Performance Targets):
- Portfolio state: 0.132ms (target: 3.0ms)
- Symbol embeddings: 0.005ms (target: 0.5ms)
- Temporal embeddings: 0.005ms (target: 0.5ms)
- State Vector Builder: <30ms (target: 30.0ms)

Batch Processing (100 assets):
- Technical indicators: ~15ms
- Data preprocessing: ~45ms
- Backtesting: ~120ms

```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/benchmarks/ -v

# Type checking
mypy src/

```

## API Reference

### Core Functions

| Function | Description | Performance |
|----------|-------------|-------------|
| `calculate_sma(prices, period)` | Simple Moving Average | ~0.05ms |
| `calculate_ema(prices, period)` | Exponential Moving Average | ~0.08ms |
| `calculate_rsi(prices, period)` | Relative Strength Index | ~0.12ms |
| `calculate_macd(prices)` | MACD Indicator | ~0.18ms |
| `normalize_data(data, method)` | Data Normalization | ~2.1ms |
| `backtest_strategy(signals, prices)` | Strategy Backtesting | ~45ms |

### Configuration

```python
from ml_common.config import MLCommonConfig

config = MLCommonConfig(
 # Performance settings
 use_numba=True,
 enable_caching=True,
 cache_size=10000,

 # Monitoring settings
 enable_logging=True,
 enable_metrics=True,

 # Calculation settings
 precision="float64",
 parallel_workers=4
)

```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/ml-framework/crypto-trading-bot.git
cd packages/ml-common
pip install -e .[dev]
pre-commit install

```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Links

- **Documentation**: [ml-framework.dev/docs/ml-common](https://ml-framework.dev/docs/ml-common)
- **API Reference**: [ml-framework.dev/api/ml-common](https://ml-framework.dev/api/ml-common)
- **Benchmarks**: [ml-framework.dev/benchmarks/ml-common](https://ml-framework.dev/benchmarks/ml-common)
- **Examples**: [examples/](examples/)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history and version notes.

---

**Built with ❤️ by the ML-Framework Team • Enterprise Architecture**
