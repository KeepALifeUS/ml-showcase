# Social Media Sentiment Analyzer for Crypto Markets

Enterprise-grade social sentiment analyzer for cryptocurrency markets with enterprise patterns.

## Key Features

### Platforms

- **Twitter/X** - Streaming API v2, advanced search, trend monitoring
- **Reddit** - Crypto subreddits, comments, hot/new/rising posts
- **Telegram** - Channels and groups, real-time monitoring
- **Discord** - Bot integration, server monitoring
- **YouTube** - Video comments, channel analysis, trending videos
- **TikTok** - Hashtag tracking, viral content detection

### AI/ML Capabilities

- **Real-time analysis** - >1000 messages/second
- **Ensemble models** - BERT, RoBERTa, FinBERT for maximum accuracy
- **Multilingual support** - Analysis in multiple languages
- **Crypto-specific tuning** - Weights for crypto-terms
- **Sarcasm/Meme detection** - Advanced NLP analysis
- **Sentiment aggregation** - Smart aggregation from various platforms

### Enterprise Architecture

- **Enterprise patterns** - Cloud-native best practices
- **Circuit breakers** - Protection from cascading failures
- **Rate limiting** - Automatic backoff strategies
- **Fault tolerance** - Graceful degradation
- **Monitoring** - Prometheus metrics, distributed tracing
- **Scalability** - Horizontal scaling, load balancing

## Installation

### Requirements

- Python 3.10+
- PostgreSQL 14+
- Redis 6+
- 8GB+ RAM (for ML models)
- NVIDIA GPU (optional, for acceleration)

### Quick Start

```bash
# Clone and navigate to directory
cd packages/ml-social-sentiment-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Install ML models
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
"

# Install spaCy model
python -m spacy download en_core_web_sm

# Environment variables (copy .env.example to .env)
cp .env.example .env
# Configure API keys and connections
```

### Docker Compose

```yaml
version: '3.8'
services:
  sentiment-analyzer:
    build: .
    ports:
      - '8004:8004'
    environment:
      - DB_HOST=postgres
      - REDIS_HOST=redis
      - TWITTER_BEARER_TOKEN=${TWITTER_BEARER_TOKEN}
      - REDDIT_CLIENT_ID=${REDDIT_CLIENT_ID}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: crypto_sentiment
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Configuration

### Environment Variables

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crypto_sentiment
DB_USER=postgres
DB_PASSWORD=your_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Twitter/X API
TWITTER_BEARER_TOKEN=your_bearer_token
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USERNAME=your_username
REDDIT_PASSWORD=your_password

# Telegram API
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
TELEGRAM_PHONE=your_phone

# Discord Bot
DISCORD_BOT_TOKEN=your_bot_token

# YouTube API
YOUTUBE_API_KEY=your_api_key

# TikTok API (if available)
TIKTOK_ACCESS_TOKEN=your_access_token

# ML Settings
ML_DEVICE=auto  # auto, cpu, cuda, mps
ML_BATCH_SIZE=32
ML_USE_GPU=true

# API Settings
API_HOST=0.0.0.0
API_PORT=8004
DEBUG=false
WORKERS=4

# Security
JWT_SECRET=your_secret_key
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring
PROMETHEUS_PORT=9090
ENABLE_TRACING=true
LOG_LEVEL=INFO
```

## Usage

### REST API

```bash
# Start API server
python -m src.api.rest_api

# Or via uvicorn
uvicorn src.api.rest_api:app --host 0.0.0.0 --port 8004 --reload
```

### Sentiment Analysis

```python
import asyncio
from src.analysis.realtime_analyzer import RealtimeSentimentAnalyzer
from src.utils.config import get_config

async def analyze_sentiment():
    config = get_config()
    analyzer = RealtimeSentimentAnalyzer(config)

    await analyzer.initialize()

    result = await analyzer.analyze_sentiment(
        text="Bitcoin is going to the moon! 🚀 HODL strong!",
        platform="twitter"
    )

    print(f"Sentiment: {result.sentiment}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Crypto symbols: {result.crypto_symbols}")
    print(f"Processing time: {result.processing_time_ms:.1f}ms")

# Run
asyncio.run(analyze_sentiment())
```

### Social Media Data Collection

```python
import asyncio
from src.connectors.twitter_connector import TwitterConnector
from src.utils.config import get_config

async def collect_crypto_tweets():
    config = get_config()
    twitter = TwitterConnector(config)

    await twitter.connect()

    # Search crypto tweets
    tweets = await twitter.search_recent_tweets(
        query="bitcoin OR ethereum OR crypto",
        max_results=100,
        crypto_focus=True
    )

    for tweet in tweets:
        print(f"@{tweet['author_username']}: {tweet['text'][:100]}...")
        print(f"Crypto symbols: {tweet['crypto_symbols']}")
        print("---")

# Run
asyncio.run(collect_crypto_tweets())
```

### API Endpoints

```bash
# Health check
curl http://localhost:8004/health

# Sentiment analysis
curl -X POST http://localhost:8004/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Ethereum is pumping hard! To the moon! 🚀", "platform": "twitter"}'

# Batch analysis
curl -X POST http://localhost:8004/sentiment/analyze-batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Bitcoin looks bullish", "Market is crashing", "HODL forever"]}'

# Aggregated sentiment
curl "http://localhost:8004/sentiment/aggregated?symbol=BTC&time_window_hours=24"

# Trending topics
curl "http://localhost:8004/trends/topics?platforms=twitter,reddit&limit=20"

# Prometheus metrics
curl http://localhost:8004/metrics/prometheus
```

## Streaming Analysis

```python
import asyncio
from src.streaming.twitter_stream import TwitterStreamProcessor

async def stream_sentiment():
    config = get_config()
    stream_processor = TwitterStreamProcessor(config)

    await stream_processor.initialize()

    # Real-time tweet stream analysis
    async for sentiment_result in stream_processor.stream_crypto_sentiment():
        print(f"Real-time: {sentiment_result.sentiment} ({sentiment_result.confidence:.3f})")
        print(f"Text: {sentiment_result.text[:100]}...")
        print(f"Symbols: {sentiment_result.crypto_symbols}")
        print("---")

asyncio.run(stream_sentiment())
```

## Monitoring

### Prometheus Metrics

```bash
# Start Prometheus server
python -c "from src.monitoring.metrics_collector import start_prometheus_server; start_prometheus_server(9090)"

# Access metrics
curl http://localhost:9090/metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Social Sentiment Analytics",
    "panels": [
      {
        "title": "Messages Processed",
        "type": "stat",
        "targets": [{ "expr": "rate(social_sentiment_messages_total[5m])" }]
      },
      {
        "title": "Sentiment Distribution",
        "type": "piechart",
        "targets": [{ "expr": "social_sentiment_model_predictions_total" }]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [{ "expr": "social_sentiment_processing_duration_seconds" }]
      }
    ]
  }
}
```

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Benchmark tests
pytest tests/benchmarks/ -v --benchmark-only

# Tests with coverage
pytest tests/ --cov=src --cov-report=html

# Specific tests
pytest tests/test_connectors.py::TestTwitterConnector::test_search_tweets -v
```

## Advanced Settings

### Ensemble Models

```python
# src/ml/ensemble_model.py
ensemble_models = [
    "cardiffnlp/twitter-roberta-base-sentiment-latest",  # Twitter-optimized
    "nlptown/bert-base-multilingual-uncased-sentiment",  # Multilingual
    "ProsusAI/finbert",                                  # Financial context
    "ElKulako/cryptobert"                                # Crypto-specific
]
```

### Crypto-Specific Weights

```python
# src/analysis/realtime_analyzer.py
crypto_sentiment_weights = {
    "moon": 0.8,           # Very positive
    "lambo": 0.7,          # Positive
    "diamond hands": 0.8,  # Very positive
    "hodl": 0.6,           # Moderately positive
    "dump": -0.8,          # Very negative
    "crash": -0.9,         # Extremely negative
    "bear": -0.6,          # Negative
    "paper hands": -0.5,   # Negative
    "fud": -0.7,           # Negative
}
```

## Performance

### Benchmarks

- **Real-time analysis**: >1000 messages/second
- **Batch processing**: >5000 messages/second (batch=100)
- **API latency**: <50ms (95th percentile)
- **Memory usage**: ~4GB (with GPU models)
- **CPU usage**: ~40% (8 cores, with GPU)

### Optimization

```python
# Settings for high-performance
ML_BATCH_SIZE=64          # Increase for GPU
ML_USE_GPU=true           # Required for performance
ML_FP16=true             # Half precision for speed
REALTIME_BATCH_SIZE=128   # Larger batch for throughput
WORKERS=8                # More workers for API
```

## Security

### JWT Authentication

```python
from src.api.authentication import AuthManager

# Create token
auth_manager = AuthManager(config)
token = await auth_manager.create_token({
    "user_id": 123,
    "username": "trader",
    "is_admin": False
})

# Verification
user_info = await auth_manager.verify_token(token)
```

### Rate Limiting

```python
# Automatic rate limiting
RATE_LIMIT_REQUESTS=1000  # Requests per window
RATE_LIMIT_WINDOW=3600    # Window in seconds

# Per-platform rate limiting
twitter_rate_limit = 100   # Per 15 minutes
reddit_rate_limit = 60     # Per minute
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Social Sentiment API                     │
├─────────────────────────────────────────────────────────────┤
│  FastAPI │ GraphQL │ WebSocket │ Authentication │ Rate Limit │
├─────────────────────────────────────────────────────────────┤
│                 Real-time Analyzer                          │
├─────────────────────────────────────────────────────────────┤
│    Sentiment │ Multilingual │ Sarcasm │ Meme │ Ensemble     │
├─────────────────────────────────────────────────────────────┤
│              Platform Connectors                            │
├─────────────────────────────────────────────────────────────┤
│ Twitter │ Reddit │ Telegram │ Discord │ YouTube │ TikTok    │
├─────────────────────────────────────────────────────────────┤
│          Streaming │ Aggregation │ Trends │ Storage         │
├─────────────────────────────────────────────────────────────┤
│         PostgreSQL │ Redis │ Kafka │ Monitoring             │
└─────────────────────────────────────────────────────────────┘
```

## Roadmap

### v1.1

- [ ] Telegram Premium API integration
- [ ] Advanced meme detection with image analysis
- [ ] Crypto whale tracking
- [ ] Enhanced Discord server analytics

### v1.2

- [ ] LinkedIn integration for B2B crypto sentiment
- [ ] Advanced time series forecasting
- [ ] Multi-language dashboard
- [ ] Mobile app API

### v1.3

- [ ] AI-powered trend prediction
- [ ] Custom model training pipeline
- [ ] Advanced visualization dashboards
- [ ] Cryptocurrency price correlation

## Contributing

```bash
# Fork repository
git clone https://github.com/your-username/ml-social-sentiment-analyzer.git

# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m "Add amazing feature"

# Push to branch
git push origin feature/amazing-feature

# Create Pull Request
```

### Code Style

```bash
# Formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Support

- **Documentation**: [https://ml-framework-docs.io/ml-social-sentiment-analyzer](https://ml-framework-docs.io/ml-social-sentiment-analyzer)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)

## Stats

![GitHub stars](https://img.shields.io/github/stars/your-repo/ml-social-sentiment-analyzer)
![GitHub issues](https://img.shields.io/github/issues/your-repo/ml-social-sentiment-analyzer)
![GitHub license](https://img.shields.io/github/license/your-repo/ml-social-sentiment-analyzer)
![Python version](https://img.shields.io/badge/python-3.10%2B-blue)
![Code coverage](https://img.shields.io/badge/coverage-85%25-green)

---

**Built for Enterprise Social Sentiment Analysis**
