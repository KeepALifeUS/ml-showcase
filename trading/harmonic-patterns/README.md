# ML Harmonic Patterns - Professional Pattern Detection System

> **ML Harmonic Patterns** for high-performance detection of harmonic patterns in crypto trading systems.

## 🎯 Overview

Comprehensive harmonic pattern detection system with machine learning enhancement, real-time processing, and professional trading signal generation for cryptocurrency markets.

### ✅ Implemented Patterns

- **🟢 Gartley 222** - Classic harmonic pattern with advanced Fibonacci validation
- **🔄 Bat Pattern** - Coming soon
- **🔄 Butterfly Pattern** - Coming soon
- **🔄 Crab Pattern** - Coming soon
- **🔄 Shark Pattern** - Coming soon
- **🔄 Cypher Pattern** - Coming soon
- **🔄 ABCD Pattern** - Coming soon
- **🔄 Three Drives Pattern** - Coming soon

## 🚀 Features

### 🎯 Core Capabilities

- **High-precision pattern detection** with Fibonacci validation
- **Real-time pattern scanning** for live trading
- **Advanced confidence scoring** with ML-enhanced ranking
- **Professional risk management** with optimal position sizing
- **Multi-timeframe analysis** support
- **Volume confirmation** analysis
- **Interactive visualization** data preparation

### 🏗️ Enterprise Architecture

- **enterprise patterns** integration
- **Production-ready code** with comprehensive error handling
- **Memory-efficient caching** system
- **Scalable performance** for large datasets
- **Thread-safe operations** with immutable data structures
- **Comprehensive testing** suite

## 📦 Installation

```bash
# Install from project root
pip install -e .

# Or install dependencies directly
pip install numpy pandas typing-extensions dataclasses

```

## 🔧 Quick Start

### Basic Pattern Detection

```python
from ml_harmonic_patterns.patterns import GartleyPattern
import pandas as pd

# Initialize detector
detector = GartleyPattern(
    tolerance=0.05,        # 5% Fibonacci tolerance
    min_confidence=0.75,   # Minimum confidence threshold
    enable_volume_analysis=True,
    enable_ml_scoring=True
)

# Load your OHLCV data
# data = pd.DataFrame with columns: ['open', 'high', 'low', 'close', 'volume']

# Detect patterns
patterns = detector.detect_patterns(
    data,
    symbol="BTCUSDT",
    timeframe="1h"
)

print(f"Found {len(patterns)} valid Gartley patterns")

```

### Trading Signal Generation

```python
if patterns:
    # Get the highest confidence pattern
    best_pattern = patterns[0]

    # Generate entry signals
    signals = detector.get_entry_signals(best_pattern)

    print(f"Action: {signals['action']}")
    print(f"Entry: ${signals['entry_price']:,.2f}")
    print(f"Stop Loss: ${signals['stop_loss']:,.2f}")
    print(f"Take Profits: {signals['take_profit_levels']}")
    print(f"Confidence: {signals['confidence']:.1%}")
    print(f"Risk/Reward: {signals['risk_reward_ratio']:.2f}")

```

### Position Sizing

```python
# Calculate optimal position size
position_info = detector.calculate_position_size(
    pattern=best_pattern,
    account_balance=10000,    # $10k account
    max_risk_percent=2.0      # 2% risk per trade
)

print(f"Position Size: {position_info['position_size']:.4f}")
print(f"Risk Amount: ${position_info['risk_amount']:.2f}")
print(f"Leverage: {position_info['leverage']:.2f}x")
print(f"Potential Profit (TP1): ${position_info['potential_profit_tp1']:.2f}")

```

### Visualization Data

```python
# Prepare data for charting libraries
viz_data = detector.prepare_visualization_data(best_pattern)

# Use with your preferred charting library
pattern_points = viz_data['pattern_points']    # X, A, B, C, D points
pattern_lines = viz_data['pattern_lines']      # Pattern structure lines
fibonacci_levels = viz_data['fibonacci_levels'] # Fib ratios and accuracy
trading_levels = viz_data['trading_levels']    # Entry, SL, TP levels
metadata = viz_data['metadata']                # Pattern information

```

## 📊 Gartley Pattern Specification

### Fibonacci Ratios

The Gartley 222 pattern follows specific Fibonacci relationships:

```

Point Relationships:
├── XA: Initial impulse move (any length)
├── AB: 61.8% retracement of XA move
├── BC: 38.2% - 88.6% retracement of AB move
├── CD: 127.2% - 161.8% extension of BC move
└── AD: 78.6% retracement of XA move (critical validation)

Validation Tolerance: ±5% (configurable)

```

### Pattern Structure

#### Bullish Gartley

```

A (High)
├── X (Low) → A (High): Initial bullish impulse
├── A → B (Low): 61.8% bearish retracement
├── B → C (High): Partial bullish recovery
├── C → D (Low): Final bearish move to completion
└── D > X but D < A (78.6% retracement level)

```

#### Bearish Gartley

```

X (High)
├── X → A (Low): Initial bearish impulse
├── A → B (High): 61.8% bullish retracement
├── B → C (Low): Partial bearish recovery
├── C → D (High): Final bullish move to completion
└── D < X but D > A (78.6% retracement level)

```

## 🎯 Advanced Usage

### Pattern Performance Analysis

```python
from ml_harmonic_patterns.patterns import analyze_pattern_performance

# Analyze historical pattern performance
performance = analyze_pattern_performance(
    patterns=detected_patterns,
    actual_prices=future_price_data,
    lookforward_periods=50
)

print(f"Success Rate: {performance['success_rate']:.1%}")
print(f"TP1 Hit Rate: {performance['tp1_hit_rate']:.1%}")
print(f"Average Confidence: {performance['average_confidence']:.1%}")

```

### Quality Filtering

```python
from ml_harmonic_patterns.patterns import filter_patterns_by_quality

# Filter only high-quality patterns
high_quality_patterns = filter_patterns_by_quality(
    patterns,
    min_confidence=0.80,      # 80% minimum confidence
    min_risk_reward=2.0,      # 2:1 minimum R/R ratio
    max_risk_percent=3.0      # Maximum 3% account risk
)

print(f"High-quality patterns: {len(high_quality_patterns)}")

```

### Custom Configuration

```python
# Advanced detector configuration
detector = GartleyPattern(
    tolerance=0.03,                # Stricter Fibonacci tolerance
    min_confidence=0.85,           # Higher confidence threshold
    enable_volume_analysis=True,   # Volume confirmation
    enable_ml_scoring=True,        # ML-enhanced scoring
    min_pattern_bars=30,           # Minimum pattern duration
    max_pattern_bars=150           # Maximum pattern duration
)

# Enable caching for performance
patterns = detector.detect_patterns(data)

# Check cache statistics
cache_stats = detector.get_cache_stats()
print(f"Cache usage: {cache_stats}")

# Clear cache when needed
detector.clear_cache()

```

## 📈 Pattern Result Structure

```python
@dataclass
class PatternResult:
    # Pattern points (X-A-B-C-D)
    point_x: PatternPoint
    point_a: PatternPoint
    point_b: PatternPoint
    point_c: PatternPoint
    point_d: PatternPoint

    # Pattern metadata
    pattern_type: PatternType         # BULLISH/BEARISH
    validation_status: PatternValidation  # VALID/INVALID/MARGINAL
    confidence_score: float           # 0.0 - 1.0

    # Fibonacci calculations
    ab_ratio: float                   # AB/XA ratio
    bc_ratio: float                   # BC/AB ratio
    cd_ratio: float                   # CD/BC ratio
    ad_ratio: float                   # AD/XA ratio

    # Trading levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float

    # Risk management
    risk_reward_ratio: float
    max_risk_percent: float

    # Quality metrics
    pattern_strength: float           # Overall pattern quality
    fibonacci_confluence: float       # Fibonacci accuracy score
    volume_confirmation: float        # Volume analysis score

```

## 🧪 Testing

Comprehensive test suite with 100+ test cases:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_gartley_pattern.py::TestGartleyPatternInit -v
python -m pytest tests/test_gartley_pattern.py::TestFibonacciValidation -v
python -m pytest tests/test_gartley_pattern.py::TestPatternDetection -v

# Run performance tests
python -m pytest tests/test_gartley_pattern.py::TestPerformance -v

# Test coverage report
python -m pytest tests/ --cov=src --cov-report=html

```

## 🔧 Performance & Optimization

### Memory Management

- **Efficient caching** with automatic cleanup
- **Memory-optimized** data structures
- **Lazy evaluation** for large datasets
- **Connection pooling** for external services

### Scalability

- **Parallel processing** support
- **Batch pattern detection** capability
- **Streaming data** compatibility
- **Multi-timeframe** analysis

### Production Deployment

```python
# Production-ready configuration
detector = GartleyPattern(
    tolerance=0.05,
    min_confidence=0.75,
    enable_volume_analysis=True,
    enable_ml_scoring=True
)

# Monitor performance
import time
start = time.time()
patterns = detector.detect_patterns(large_dataset)
execution_time = time.time() - start

print(f"Processed {len(large_dataset)} bars in {execution_time:.2f}s")
print(f"Found {len(patterns)} patterns")

```

## 🚨 Error Handling

```python
try:
    patterns = detector.detect_patterns(data)
except ValueError as e:
    print(f"Data validation error: {e}")
except RuntimeError as e:
    print(f"Pattern detection error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

```

## 📝 Logging

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ml_harmonic_patterns')

# Detector automatically logs:
# - Pattern detection progress
# - Validation results
# - Performance metrics
# - Error conditions

```

## 🤝 Contributing

1. Follow enterprise patterns
2. Maintain 100% test coverage
3. Use type hints and docstrings
4. Optimize for performance
5. Add comprehensive error handling

## 📄 License

MIT License

## 🏗️ Architecture

### Enterprise Patterns Applied

1. **Immutable Data Structures** - Thread-safe pattern results
2. **Domain-Driven Design** - Clear separation of concerns
3. **SOLID Principles** - Maintainable and extensible code
4. **Performance Optimization** - Caching and lazy evaluation
5. **Comprehensive Testing** - Production-ready reliability
6. **Error Resilience** - Graceful degradation
7. **Observability** - Detailed logging and metrics

### Integration Points

- **Real-time Trading Engine** - Live pattern detection
- **Backtesting System** - Historical performance analysis
- **Risk Management** - Position sizing and risk controls
- **Visualization Engine** - Interactive chart overlays
- **ML Pipeline** - Pattern classification enhancement
- **Alert System** - Real-time pattern notifications

---

**Built with ML Harmonic Patterns v1.0**

## Support

For questions and support, please open an issue on GitHub.
