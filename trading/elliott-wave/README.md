# ML Elliott Wave - CNN-based Elliott Wave Pattern Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Professional Elliott Wave analysis system optimized for crypto markets with 24/7 trading, high volatility handling, and real-time pattern recognition.

## Overview

The Elliott Wave Analyzer is a comprehensive, production-ready system for automated Elliott Wave pattern detection and analysis, specifically optimized for cryptocurrency markets. It provides:

- **Advanced Pattern Recognition**: CNN-based ML models for high-accuracy wave detection
- **Complete Elliott Wave Analysis**: Impulse waves, corrective patterns, diagonal structures
- **Fibonacci Integration**: Retracements, extensions, time projections, and confluence analysis
- **Real-time Processing**: Async architecture for 24/7 crypto market monitoring
- **Multi-timeframe Analysis**: Synchronized analysis across multiple timeframes
- **Professional API**: RESTful endpoints with WebSocket real-time updates
- **Comprehensive Testing**: Test suite with crypto-specific scenarios

## Architecture

```

ml-elliott-wave/
├── src/
│   ├── patterns/           # Wave pattern detection
│   │   ├── impulse_wave.py      # 5-wave impulse patterns
│   │   ├── corrective_wave.py   # ABC corrective patterns
│   │   ├── diagonal_wave.py     # Leading/ending diagonals
│   │   ├── triangle_wave.py     # Triangle patterns
│   │   ├── flat_wave.py         # Flat corrections
│   │   └── zigzag_wave.py       # Zigzag corrections
│   │
│   ├── analysis/           # Wave analysis components
│   │   ├── wave_counter.py      # Automatic wave counting
│   │   ├── wave_labeler.py      # Wave degree labeling
│   │   ├── wave_validator.py    # Rule validation
│   │   ├── wave_projector.py    # Future projections
│   │   ├── alternative_counts.py # Alternative interpretations
│   │   └── confidence_scorer.py # Confidence scoring
│   │
│   ├── fibonacci/          # Fibonacci analysis tools
│   │   ├── fibonacci_retracement.py # Retracement levels
│   │   ├── fibonacci_extension.py   # Extension targets
│   │   ├── fibonacci_time.py        # Time projections
│   │   ├── fibonacci_clusters.py    # Cluster analysis
│   │   ├── golden_ratio.py          # Golden ratio calculations
│   │   └── fibonacci_channels.py    # Channel projections
│   │
│   ├── ml/                 # Machine learning models
│   │   ├── cnn_wave_detector.py     # CNN pattern recognition
│   │   ├── lstm_wave_predictor.py   # LSTM predictions
│   │   ├── transformer_analyzer.py  # Transformer models
│   │   ├── ensemble_model.py        # Ensemble methods
│   │   ├── reinforcement_learning.py# RL trading agents
│   │   └── genetic_optimizer.py     # Genetic algorithms
│   │
│   ├── api/                # API layer
│   │   ├── rest_api.py             # RESTful API
│   │   ├── websocket_server.py     # Real-time WebSocket
│   │   ├── graphql_schema.py       # GraphQL endpoints
│   │   └── authentication.py       # Security & auth
│   │
│   └── utils/              # Utilities & configuration
│       ├── config.py               # Configuration management
│       ├── logger.py               # Centralized logging
│       ├── validators.py           # Data validation
│       └── metrics.py              # Performance metrics
│
├── tests/                  # Comprehensive test suite
└── docs/                   # Documentation

```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/KeepALifeUS/ml-elliott-wave
cd ml-elliott-wave

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e .[dev,test,ml-extra]

```

### Basic Usage

```python
import asyncio
import pandas as pd
from src.patterns.impulse_wave import ImpulseWaveDetector
from src.patterns.corrective_wave import CorrectiveWaveDetector
from src.fibonacci.fibonacci_retracement import FibonacciRetracementCalculator

async def analyze_waves():
    # Load your OHLCV data
    data = pd.read_csv("btc_1h_data.csv")
    data.set_index('timestamp', inplace=True)

    # Initialize detectors
    impulse_detector = ImpulseWaveDetector(confidence_threshold=0.7)
    corrective_detector = CorrectiveWaveDetector()
    fibonacci_calc = FibonacciRetracementCalculator()

    # Detect impulse waves
    impulse_waves = await impulse_detector.detect_impulse_waves(
        data, "BTCUSDT", "1h"
    )

    # Detect corrective waves
    corrective_waves = await corrective_detector.detect_corrective_waves(
        data, "BTCUSDT", "1h"
    )

    # Analyze results
    print(f"Found {len(impulse_waves)} impulse waves")
    print(f"Found {len(corrective_waves)} corrective waves")

    for wave in impulse_waves:
        if wave.confidence > 0.8:
            print(f"High-confidence {wave.direction} impulse wave")
            print(f"Confidence: {wave.confidence:.2f}")
            print(f"Projection targets: {wave.get_projection_targets()}")

# Run analysis
asyncio.run(analyze_waves())

```

### API Server

```python
from src.api.rest_api import app
import uvicorn

# Start API server
uvicorn.run(app, host="0.0.0.0", port=8000)

```

#### API Usage

```bash
# Analyze waves via API
curl -X POST "http://your-server:8000/api/v1/analyze" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "4h",
    "price_data": [...],
    "analysis_types": ["impulse", "corrective", "fibonacci"],
    "confidence_threshold": 0.7
  }'

# WebSocket real-time updates
wscat -c ws://your-server:8000/ws/waves

```

## Pattern Detection Features

### Impulse Wave Detection

```python
from src.patterns.impulse_wave import ImpulseWaveDetector

detector = ImpulseWaveDetector(
    min_wave_length=5,
    max_wave_length=200,
    fibonacci_tolerance=0.15,
    confidence_threshold=0.7
)

waves = await detector.detect_impulse_waves(data, "BTCUSDT", "1h")

for wave in waves:
    print(f"Wave confidence: {wave.confidence}")
    print(f"Rules validation: {wave.rules_validation}")
    print(f"Fibonacci ratios: {wave.fibonacci_ratios}")
    print(f"Next targets: {wave.get_projection_targets()}")

```

### Corrective Wave Analysis

```python
from src.patterns.corrective_wave import CorrectiveWaveDetector

detector = CorrectiveWaveDetector()
waves = await detector.detect_corrective_waves(data, "BTCUSDT", "1h")

for wave in waves:
    print(f"Corrective type: {wave.corrective_type}")
    print(f"Pattern strength: {wave.confidence}")
    print(f"ABC structure: A={wave.wave_a_length:.2f}, "
          f"B={wave.wave_b_length:.2f}, C={wave.wave_c_length:.2f}")

```

### Machine Learning Integration

```python
from src.ml.cnn_wave_detector import CNNWaveDetector

# Load pre-trained model
cnn_detector = CNNWaveDetector(
    model_path="models/wave_cnn_v1.pth",
    architecture="custom_crypto"
)

# Detect patterns with ML
patterns = await cnn_detector.detect_patterns(
    data, "BTCUSDT", "1h", confidence_threshold=0.8
)

for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}")
    print(f"Confidence: {pattern.confidence}")
    print(f"Probabilities: {pattern.probability_distribution}")

```

## Fibonacci Analysis

```python
from src.fibonacci.fibonacci_retracement import FibonacciRetracementCalculator
from src.patterns.impulse_wave import WavePoint

calc = FibonacciRetracementCalculator(
    retracement_type="crypto_adapted",
    include_confluence=True
)

# Define swing points
swing_high = WavePoint(index=100, price=50000, timestamp=datetime.now())
swing_low = WavePoint(index=200, price=45000, timestamp=datetime.now())

# Calculate retracement
retracement = await calc.calculate_retracement(
    swing_high, swing_low, "BTCUSDT", "4h", data
)

print("Fibonacci Levels:")
for level in retracement.levels:
    print(f"  {level.percentage:.1f}% - ${level.price:.2f} "
          f"(Strength: {level.strength:.2f})")

print("Confluence Zones:")
for zone in retracement.get_confluence_zones():
    print(f"  Zone: {zone['level']:.2f} (Strength: {zone['strength']:.2f})")

```

## Multi-timeframe Analysis

```python
from src.analysis.wave_counter import WaveCounter

counter = WaveCounter(primary_degree="minor")

# Analyze multiple timeframes
timeframes = ["1h", "4h", "1d"]
all_counts = {}

for tf in timeframes:
    tf_data = get_data_for_timeframe(tf)  # Your data loading function
    counts = await counter.count_waves(tf_data, "BTCUSDT", tf)
    all_counts[tf] = counts

# Find confluence across timeframes
for tf, counts in all_counts.items():
    if counts:
        primary_count = counts[0]
        print(f"{tf}: {primary_count.wave_sequence} "
              f"({primary_count.confidence:.2f} confidence)")

```

## Configuration

```python
from src.utils.config import config

# Customize analysis parameters
config.wave_analysis.update({
    "min_wave_length": 10,
    "max_wave_length": 500,
    "fibonacci_tolerance": 0.1,
    "confidence_threshold": 0.75,
    "timeframes": ["15m", "1h", "4h", "1d"],
    "enable_multi_timeframe": True
})

# Crypto-specific settings
config.exchanges["binance"].update({
    "rate_limit": 1200,
    "timeout": 10.0
})

# ML model configuration
config.ml_config.update({
    "batch_size": 64,
    "learning_rate": 0.0001,
    "device": "cuda"
})

```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_patterns.py -v        # Pattern detection tests
pytest tests/test_fibonacci.py -v       # Fibonacci analysis tests
pytest tests/test_ml.py -v             # ML model tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Performance benchmarks
pytest tests/ -m slow -v               # Long-running performance tests

# Crypto-specific scenarios
pytest tests/ -k "crypto or btc" -v    # Crypto market tests

```

## Performance Optimization

### Crypto Market Adaptations

- **24/7 Trading**: Continuous pattern monitoring without market close gaps
- **High Volatility Handling**: Adaptive noise filtering and pattern validation
- **Multi-exchange Support**: Unified analysis across different crypto exchanges
- **Rapid Price Movement**: Optimized for crypto's fast-moving markets

### Performance Tuning

```python
# Enable performance optimizations
config.performance_config.update({
    "enable_caching": True,
    "cache_ttl": 300,
    "worker_pool_size": 8,
    "connection_pool_size": 50,
    "async_timeout": 60.0
})

# Monitor performance
from src.utils.metrics import performance_monitor

@performance_monitor
async def analyze_with_monitoring():
    # Your analysis code here
    pass

```

## Security & Authentication

```python
from src.api.authentication import AuthManager, JWTAuth

# Configure authentication
auth_manager = AuthManager()

# JWT-based authentication
jwt_auth = JWTAuth(
    secret_key=os.getenv("JWT_SECRET"),
    expiry_minutes=60
)

# API key authentication
api_key_auth = APIKeyAuth()

```

## Monitoring & Logging

```python
from src.utils.logger import get_logger, trading_logger

logger = get_logger(__name__)

# Structured trading logs
trading_logger.log_wave_detection(
    symbol="BTCUSDT",
    timeframe="4h",
    wave_type="impulse",
    confidence=0.85,
    detected_count=3
)

# Performance monitoring
trading_logger.log_fibonacci_level(
    symbol="BTCUSDT",
    timeframe="1h",
    level_type="retracement",
    price_level=45000,
    support_resistance="strong_support"
)

```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .[prod]

EXPOSE 8000

CMD ["uvicorn", "src.api.rest_api:app", "--host", "0.0.0.0", "--port", "8000"]

```

### Production Configuration

```python
# config/production.py
config.environment = "production"
config.debug = False
config.api_workers = 4
config.log_level = "INFO"

# Database configuration
config.database_url = os.getenv("DATABASE_URL", "postgresql://user:pass@db-host/elliott_waves")
config.redis_url = os.getenv("REDIS_URL", "redis://redis-host:6379/0")

# Security
config.jwt_secret = os.getenv("JWT_SECRET")
config.cors_origins = [os.getenv("CORS_ORIGIN", "https://your-app-domain.example.com")]

```

## Documentation

- **[API Documentation](docs/api.md)** - Complete API reference
- **[Pattern Theory](docs/theory.md)** - Elliott Wave theory implementation
- **[ML Models](docs/ml.md)** - Machine learning model details
- **[Crypto Adaptations](docs/crypto.md)** - Crypto-specific features
- **[Performance Guide](docs/performance.md)** - Optimization strategies

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -e .[dev,test,ml-extra]

# Setup pre-commit hooks
pre-commit install

# Run quality checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Elliott Wave Principle** by Robert Prechter Jr.
- **Crypto Trading Community** for market insights and feedback
- **Open Source Contributors** for tools and libraries

---

**Production-Ready | Crypto-Optimized**

## Support

For questions and support, please open an issue on GitHub.
