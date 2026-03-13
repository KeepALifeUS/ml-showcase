# ML Fear & Greed Index

Comprehensive Fear & Greed Index system for cryptocurrency markets with advanced ML capabilities.

## Overview

The ML Fear & Greed Index System provides a sophisticated, multi-component approach to measuring market sentiment in cryptocurrency markets. It combines traditional technical indicators with modern ML techniques, social sentiment analysis, and real-time data processing.

## Components

### Index Components

- **Volatility Component**: Market volatility measurement using multiple methodologies (Realized, Parkinson, GARCH)
- **Momentum Component**: Price momentum analysis with RSI, MACD, ROC, Williams %R, Stochastic RSI, CCI
- **Volume Component**: Trading volume trends analysis with OBV, A/D Line, MFI, VWAP, Volume RSI
- **Social Sentiment Component**: Social media sentiment analysis using transformer models
- **Dominance Component**: Bitcoin dominance and market cap distribution analysis
- **Search Trends Component**: Google Trends analysis for cryptocurrency interest
- **Surveys Component**: Market sentiment surveys and professional opinions

### Calculation Algorithms

- **Weighted Calculator**: Main algorithm combining all components with configurable weights
- **Adaptive Calculator**: Self-adjusting weights based on component reliability
- **Ensemble Calculator**: Multiple calculation methods with voting

### Data Collectors

- **Price Data Collector**: OHLCV data from multiple exchanges (CCXT integration)
- **Social Collector**: Social media data from Twitter, Reddit, Telegram
- **Google Trends Collector**: Search trends and interest data

### ML Models

- **Sentiment Transformer**: Advanced NLP for text sentiment analysis (FinBERT, RoBERTa)
- **Ensemble Models**: Multiple model voting for improved accuracy

## Architecture

```
ml-fear-greed-index/
├── src/
│   ├── components/          # Index calculation components
│   ├── calculators/         # Index calculation algorithms
│   ├── collectors/          # Data collection modules
│   ├── models/              # ML models and main index model
│   └── utils/               # Utilities and configuration
├── tests/                   # Test suite
└── pyproject.toml
```

## Quick Start

### Installation

```bash
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

The system uses environment variables with the `FEAR_GREED_` prefix for configuration via Pydantic:

```bash
# API Keys
export FEAR_GREED_TWITTER_BEARER_TOKEN="your_token"
export FEAR_GREED_REDDIT_CLIENT_ID="your_id"
export FEAR_GREED_REDDIT_CLIENT_SECRET="your_secret"
export FEAR_GREED_NEWS_API_KEY="your_key"
export FEAR_GREED_BINANCE_API_KEY="your_key"
export FEAR_GREED_BINANCE_SECRET_KEY="your_secret"
export FEAR_GREED_COINGECKO_API_KEY="your_key"

# Database
export DATABASE_HOST="localhost"
export DATABASE_PORT=5432
export DATABASE_NAME="fear_greed_index"
export DATABASE_USERNAME="postgres"
export DATABASE_PASSWORD="your_password"

# Cache
export REDIS_URL="redis://localhost:6379"
```

### Basic Usage

```python
from ml_fear_greed_index import FearGreedIndex, FearGreedConfig

# Initialize with configuration
config = FearGreedConfig()
fear_greed = FearGreedIndex(config)

# Calculate Fear & Greed Index for Bitcoin
result = await fear_greed.calculate("BTC")

print(f"Fear & Greed Score: {result.final_score}")
print(f"Interpretation: {result.interpretation}")
print(f"Confidence: {result.confidence:.2f}")
```

## Features

### Multi-Timeframe Analysis

- Support for 1h, 4h, 1d, 1w, 1M timeframes
- Cross-timeframe correlation analysis
- Adaptive weighting based on timeframe

### Multi-Exchange Support

- Binance, Coinbase Pro, Kraken, Huobi integration via CCXT
- Automatic data aggregation and validation
- Fallback mechanisms for data reliability

### Advanced ML Features

- BERT-based sentiment analysis for financial texts
- Ensemble models for improved accuracy
- Real-time model updates and retraining

### Production-Ready Features

- Redis caching for performance
- Comprehensive monitoring and alerting
- Circuit breakers and fault tolerance
- Rate limiting and API protection

## Testing

```bash
pytest
pytest --cov=src --cov-report=html
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For questions and support, please open an issue on GitHub.
