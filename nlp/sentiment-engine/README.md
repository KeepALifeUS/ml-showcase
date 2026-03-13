# ML-Framework ML Sentiment Engine

Enterprise-grade sentiment aggregation system for crypto trading.

## Features

### Data Sources

- **Twitter/X**: Real-time tweets with crypto-specific filtering
- **Reddit**: Multi-subreddit monitoring with comment analysis
- **News**: RSS aggregation from 20+ crypto news sources
- **Telegram**: Channel monitoring with real-time streaming
- **Discord**: Server monitoring with bot integration

### Sentiment Models

- **FinBERT**: Financial domain-specific transformer
- **VADER**: Social media optimized with crypto lexicon
- **Ensemble**: Adaptive weighted combination
- **Context-aware**: Source-specific model weighting

### Real-time Processing

- **Kafka Streaming**: Scalable message processing
- **WebSocket API**: Real-time sentiment feeds
- **Circuit Breakers**: Fault-tolerant external APIs
- **Rate Limiting**: Compliance with platform limits

### Enterprise Features

- **FastAPI**: Async REST API with OpenAPI docs
- **GraphQL**: Flexible data querying
- **Monitoring**: Prometheus metrics + OpenTelemetry
- **Caching**: Redis-based performance optimization
- **Storage**: TimescaleDB for time-series data

## Installation

```bash
# Install package
pip install -e .

# Install with GPU support
pip install -e ".[gpu]"

# Development dependencies
pip install -e ".[dev]"

```

## 🔧 Configuration

Create `.env` file:

```bash
# Database
DB_HOST=localhost
DB_PORT=5432
DB_USERNAME=ml-framework_user
DB_PASSWORD=your_password
DB_DATABASE=ml-framework_sentiment

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Social Media APIs
TWITTER_BEARER_TOKEN=your_bearer_token
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
DISCORD_TOKEN=your_bot_token

# ML Configuration
ML_USE_GPU=false
ML_BATCH_SIZE=32
ML_FINBERT_MODEL=ProsusAI/finbert

# API Configuration
API_HOST=0.0.0.0
API_PORT=8003
API_WORKERS=4

```

## 🚀 Quick Start

### 1. Start API Server

```bash
# Development
python -m src.api.sentiment_api

# Production
uvicorn src.api.sentiment_api:app --host 0.0.0.0 --port 8003 --workers 4

```

### 2. Basic Usage

```python
import asyncio
from src.models.ensemble_sentiment import create_ensemble_model

async def main:
 # Initialize model
 model = await create_ensemble_model

 # Analyze sentiment
 result = await model.predict(
 "Bitcoin is going to the moon! 🚀 HODL!",
 source="twitter"
 )

 print(f"Sentiment: {result.value}")
 print(f"Confidence: {result.confidence}")

asyncio.run(main)

```

### 3. Fetch Data from Sources

```python
from src.sources.twitter_source import create_twitter_source
from src.sources.reddit_source import create_reddit_source

async def fetch_crypto_sentiment:
 # Twitter data
 twitter = await create_twitter_source
 tweets = await twitter.search_tweets(
 symbols=["BTC", "ETH"],
 limit=100,
 hours_back=24
 )

 # Reddit data
 reddit = await create_reddit_source
 posts = await reddit.fetch_all_crypto_content(
 limit_per_subreddit=50
 )

 return tweets, posts

```

## 🌐 API Documentation

### Sentiment Analysis

```bash
# Single text analysis
curl -X POST "http://localhost:8003/sentiment" \
 -H "Content-Type: application/json" \
 -d '{
 "text": "Ethereum 2.0 upgrade looks promising!",
 "source": "news",
 "model": "ensemble"
 }'

# Batch analysis
curl -X POST "http://localhost:8003/sentiment/batch" \
 -H "Content-Type: application/json" \
 -d '{
 "texts": [
 "Bitcoin hitting new ATH!",
 "Market crash incoming?",
 "DeFi is the future"
 ],
 "source": "twitter"
 }'

```

### Data Sources

```bash
# Fetch Twitter data
curl "http://localhost:8003/data/twitter?symbol=BTC&limit=50&hours_back=24"

# Fetch Reddit data
curl "http://localhost:8003/data/reddit?limit=100"

# Fetch News data
curl "http://localhost:8003/data/news?limit=50&hours_back=12"

```

### Health & Stats

```bash
# Health check
curl "http://localhost:8003/health"

# API statistics
curl "http://localhost:8003/stats"

# Prometheus metrics
curl "http://localhost:8003/metrics"

```

## 🔄 Real-time Streaming

### WebSocket Sentiment Feed

```javascript
const ws = new WebSocket('ws://localhost:8003/ws/sentiment');

ws.onmessage = function (event) {
 const data = JSON.parse(event.data);
 console.log('New sentiment:', data);
};

```

### Kafka Integration

```python
from src.streaming.kafka_consumer import SentimentKafkaConsumer

consumer = SentimentKafkaConsumer
await consumer.start_consuming("crypto-sentiment")

```

## 📊 Monitoring

### Prometheus Metrics

- `sentiment_api_requests_total`: Total API requests
- `sentiment_predictions_total`: Total predictions by model
- `sentiment_api_request_duration_seconds`: Request latency
- `data_source_fetch_total`: Data source fetch counts

### Grafana Dashboard

Import the provided dashboard from `docs/grafana-dashboard.json`

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test suite
pytest tests/test_models.py -v

```

## 📚 Advanced Usage

### Custom Model Ensemble

```python
from src.models.ensemble_sentiment import EnsembleSentimentModel

# Create custom ensemble
ensemble = EnsembleSentimentModel
await ensemble.initialize

# Custom weights
result = await ensemble.predict(
 text="Crypto market analysis",
 source="news",
 use_adaptive_weighting=True
)

```

### Multi-source Aggregation

```python
from src.aggregation.weighted_aggregator import WeightedAggregator

aggregator = WeightedAggregator

# Aggregate sentiment from multiple sources
combined_sentiment = await aggregator.aggregate_multi_source([
 {"source": "twitter", "data": twitter_data},
 {"source": "reddit", "data": reddit_data},
 {"source": "news", "data": news_data}
])

```

## 🛠️ Development

### Code Quality

```bash
# Linting
flake8 src/ tests/

# Type checking
mypy src/

# Formatting
black src/ tests/
isort src/ tests/

```

### Adding New Data Sources

1. Create new source in `src/sources/`
2. Implement required interface methods
3. Add configuration in `src/utils/config.py`
4. Add tests in `tests/test_sources.py`

### Adding New Models

1. Create model in `src/models/`
2. Implement `predict` and `predict_batch` methods
3. Add to ensemble configuration
4. Add performance tests

## 🏗️ Architecture

```

ML-Framework ML Sentiment Engine
├── Data Sources Layer
│ ├── Twitter/X API
│ ├── Reddit API
│ ├── News RSS Feeds
│ ├── Telegram Client
│ └── Discord Bot
├── Processing Layer
│ ├── NLP Preprocessing
│ ├── Sentiment Models
│ ├── Ensemble Logic
│ └── Aggregation Algorithms
├── Storage Layer
│ ├── TimescaleDB (Time-series)
│ ├── Redis (Caching)
│ └── Vector Store (Embeddings)
├── API Layer
│ ├── REST API (FastAPI)
│ ├── GraphQL API
│ ├── WebSocket Streaming
│ └── Rate Limiting
└── Monitoring Layer
 ├── Prometheus Metrics
 ├── OpenTelemetry Tracing
 ├── Structured Logging
 └── Health Checks

```

## 📈 Performance

- **Throughput**: 1000+ predictions/second
- **Latency**: <100ms average response time
- **Scalability**: Horizontal scaling with Kafka
- **Reliability**: 99.9% uptime with circuit breakers

## 🔒 Security

- Input validation and sanitization
- Rate limiting per IP/API key
- SQL injection prevention
- XSS protection
- Secure credential storage

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: GitHub Issues
- Email: <ml-team@ml-framework.com>

---

**ML-Framework ML Sentiment Engine** - Enterprise-grade sentiment analysis for crypto trading.

## Support

For questions and support, please open an issue on GitHub.
