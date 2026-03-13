# ML Order Flow Detection System

Enterprise-grade order flow pattern detection system for cryptocurrency trading, built with enterprise patterns and designed for high-frequency, low-latency analysis.

## 🚀 Features

### Order Flow Analysis

- **Delta Analysis**: Buy/sell volume delta analysis with momentum detection
- **Cumulative Delta**: Advanced cumulative volume delta with divergence detection
- **Footprint Chart**: Price-level volume analysis with market structure identification
- **Imbalance Detection**: Real-time order imbalance pattern recognition
- **Absorption Detection**: Large order absorption pattern analysis
- **Exhaustion Detection**: Momentum exhaustion pattern identification

### Pattern Detection

- **Iceberg Orders**: Detection of hidden large orders using multiple algorithms
- **Spoofing Detection**: Comprehensive market manipulation detection
- **Layering Detection**: Multi-level fake order detection
- **Momentum Ignition**: False momentum creation detection
- **Stop Hunting**: Predatory trading pattern detection
- **Accumulation/Distribution**: Institutional flow pattern analysis

### Advanced Analytics

- **Volume Profile**: Multi-timeframe volume profile analysis
- **VWAP Analysis**: Volume-weighted average price with deviations
- **Point of Control**: Dynamic POC identification
- **Value Area**: Statistical value area calculations
- **Liquidity Zones**: Market liquidity analysis
- **Volume Clusters**: Significant volume cluster detection

### Machine Learning Models

- **LSTM Flow Predictor**: Deep learning for order flow prediction
- **Transformer Patterns**: Advanced pattern recognition
- **XGBoost Detection**: Gradient boosting for anomaly detection
- **Random Forest**: Ensemble classification
- **Neural Networks**: Deep neural networks for complex patterns
- **Ensemble Models**: Combined model approaches

### Real-time Processing

- **Stream Processing**: High-performance real-time analysis
- **WebSocket Handling**: Multi-exchange data streaming
- **Order Book Analysis**: Real-time order book processing
- **Trade Stream**: Live trade data analysis
- **Alert Generation**: Real-time pattern alerts

## 🏗️ Architecture

Built on enterprise patterns:

- **High-Performance Computing**: Optimized for microsecond latency
- **Stream Processing**: Real-time data processing
- **Event-Driven**: Asynchronous event processing
- **Microservices**: Modular, scalable architecture
- **Cloud-Native**: Container-ready with Kubernetes support

## 📦 Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis 6+
- PostgreSQL 13+

### Install Package

```bash
cd packages/ml-order-flow-detection
pip install -e .

```

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev,quality,performance]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=src

# Run quality checks
ruff check src/ tests/
mypy src/

```

## 🚀 Quick Start

### Basic Order Flow Analysis

```python
from src.order_flow.delta_analyzer import DeltaAnalyzer
from src.order_flow.cumulative_delta import CumulativeDeltaAnalyzer
from src.order_flow.footprint_chart import FootprintAnalyzer

# Initialize analyzers
delta_analyzer = DeltaAnalyzer("BTCUSDT")
cumulative_analyzer = CumulativeDeltaAnalyzer("BTCUSDT")
footprint_analyzer = FootprintAnalyzer("BTCUSDT")

# Process trade data
async def process_trade(price: float, volume: float, is_buy: bool):
    # Delta analysis
    delta_metrics = await delta_analyzer.add_trade(price, volume, is_buy)
    patterns = await delta_analyzer.detect_patterns()

    # Cumulative delta
    cum_bar = await cumulative_analyzer.add_trade(price, volume, is_buy)

    # Footprint analysis
    footprint_bar = await footprint_analyzer.add_trade(price, volume, is_buy)

    return {
        'delta': delta_metrics,
        'cumulative': cum_bar,
        'footprint': footprint_bar,
        'patterns': patterns
    }

```

### Pattern Detection

```python
from src.patterns.iceberg_detector import IcebergDetector
from src.patterns.spoofing_detector import SpoofingDetector

# Initialize detectors
iceberg_detector = IcebergDetector("BTCUSDT")
spoofing_detector = SpoofingDetector("BTCUSDT")

# Analyze order book for icebergs
order_book = {
    'bids': [(50000.0, 1.5), (49999.0, 2.0)],
    'asks': [(50001.0, 1.2), (50002.0, 1.8)]
}

icebergs = await iceberg_detector.analyze_order_book(order_book)

# Process order events for spoofing
from src.patterns.spoofing_detector import OrderEvent

order_event = OrderEvent(
    timestamp=time.time(),
    order_id="12345",
    price=50000.0,
    size=5.0,
    side="buy",
    event_type="place"
)

spoofing_signals = await spoofing_detector.process_order_event(order_event)

```

### Real-time Stream Processing

```python
from src.realtime.stream_processor import StreamProcessor
from src.realtime.websocket_handler import WebSocketHandler

# Initialize stream processor
processor = StreamProcessor(["BTCUSDT", "ETHUSDT"])

# Start processing
await processor.start()

# Process incoming data
async def on_trade(symbol, price, volume, side):
    analysis = await processor.analyze_trade(symbol, price, volume, side)
    if analysis.get('patterns'):
        print(f"Patterns detected: {analysis['patterns']}")

# WebSocket handling
ws_handler = WebSocketHandler("wss://api.exchange.com/ws")
await ws_handler.connect()
await ws_handler.subscribe_trades(["BTCUSDT"])

```

## 📊 Volume Profile Analysis

```python
from src.volume_profile.volume_profile_builder import VolumeProfileBuilder
from src.volume_profile.vwap_calculator import VWAPCalculator

# Build volume profile
profile_builder = VolumeProfileBuilder("BTCUSDT")
vwap_calc = VWAPCalculator("BTCUSDT")

# Process market data
for trade in historical_trades:
    await profile_builder.add_trade(
        trade['price'],
        trade['volume'],
        trade['timestamp']
    )

    vwap_data = await vwap_calc.update(
        trade['price'],
        trade['volume'],
        trade['timestamp']
    )

# Get volume profile
profile = profile_builder.get_profile()
poc = profile_builder.get_point_of_control()
value_area = profile_builder.get_value_area()

```

## 🤖 Machine Learning Integration

```python
from src.ml.lstm_flow_predictor import LSTMFlowPredictor
from src.ml.xgboost_detector import XGBoostDetector

# Train LSTM model
lstm_model = LSTMFlowPredictor()
await lstm_model.train(training_data)

# Make predictions
prediction = await lstm_model.predict(current_features)

# XGBoost anomaly detection
xgb_detector = XGBoostDetector()
await xgb_detector.fit(normal_patterns, anomalous_patterns)

anomaly_score = await xgb_detector.detect_anomaly(current_pattern)

```

## 🔧 Configuration

```python
from src.utils.config import get_settings, OrderFlowSettings

# Get global settings
settings = get_settings()

# Custom configuration
custom_settings = OrderFlowSettings(
    environment="production",
    order_flow=OrderFlowConfig(
        tick_size=0.01,
        imbalance_threshold=0.7,
        absorption_ratio=2.0
    ),
    ml_models=MLModelConfig(
        batch_size=1024,
        learning_rate=0.001
    )
)

```

## 📈 Backtesting

```python
from src.backtesting.flow_backtester import FlowBacktester

# Initialize backtester
backtester = FlowBacktester()

# Define strategy
async def order_flow_strategy(data):
    if data.delta_ratio > 0.7:
        return {'action': 'buy', 'confidence': data.confidence}
    elif data.delta_ratio < 0.3:
        return {'action': 'sell', 'confidence': data.confidence}
    return {'action': 'hold'}

# Run backtest
results = await backtester.run(
    strategy=order_flow_strategy,
    data=historical_data,
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")

```

## 🌐 API Endpoints

### REST API

```python
from src.api.rest_api import app

# Start REST API server
uvicorn.run(app, host="0.0.0.0", port=8000)

```

Available endpoints:

- `GET /api/v1/analysis/{symbol}/delta` - Delta analysis
- `GET /api/v1/analysis/{symbol}/patterns` - Detected patterns
- `GET /api/v1/volume-profile/{symbol}` - Volume profile
- `POST /api/v1/detect/iceberg` - Iceberg detection
- `POST /api/v1/detect/spoofing` - Spoofing detection

### WebSocket Server

```python
from src.api.websocket_server import WebSocketServer

# Start WebSocket server
server = WebSocketServer(port=8001)
await server.start()

```

WebSocket channels:

- `/ws/analysis/{symbol}` - Real-time analysis
- `/ws/patterns/{symbol}` - Pattern alerts
- `/ws/alerts` - System alerts

## 📊 Visualization

```python
from src.visualization.flow_visualizer import FlowVisualizer
from src.visualization.dashboard_api import DashboardAPI

# Create visualizations
visualizer = FlowVisualizer()
chart = visualizer.create_footprint_chart(footprint_data)
heatmap = visualizer.create_delta_heatmap(delta_data)

# Start dashboard
dashboard = DashboardAPI()
await dashboard.start(port=8050)

```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/test_ml_models.py  # ML model tests

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run performance tests
pytest tests/performance/ -v --benchmark-only

```

## 📋 Performance Metrics

### Latency Targets (performance standards)

- Order processing: < 100μs
- Pattern detection: < 1ms
- ML inference: < 10ms
- API response: < 50ms

### Throughput Targets

- Orders/second: 100,000+
- Trades/second: 50,000+
- WebSocket messages: 1M+/second
- Concurrent connections: 10,000+

## 🔒 Security Features

- Input validation and sanitization
- Rate limiting and DDoS protection
- Authentication and authorization
- Encrypted data transmission
- Audit logging and monitoring
- Compliance reporting

## 📚 Documentation

- [API Reference](docs/api.md)
- [Pattern Guide](docs/patterns.md)
- [ML Models Guide](docs/ml-models.md)
- [Configuration Reference](docs/configuration.md)
- [Performance Tuning](docs/performance.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] Options flow integration
- [ ] Cross-asset arbitrage detection
- [ ] Enhanced ML models (GPT, BERT)
- [ ] Real-time risk management
- [ ] Mobile dashboard
- [ ] Cloud deployment automation

## 📞 Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/ml-framework/ml-order-flow-detection/issues)
- Discussions: [GitHub Discussions](https://github.com/ml-framework/ml-order-flow-detection/discussions)

---

Built with ❤️ using enterprise patterns for Crypto Trading Bot v5.0

## Support

For questions and support, please open an issue on GitHub.
