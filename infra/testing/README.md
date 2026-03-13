# ML Testing Framework

**Enterprise-Grade Machine Learning Testing Suite with Enterprise Patterns**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://pytest.org/)
[![Enterprise](https://img.shields.io/badge/patterns-Enterprise-purple.svg)](#enterprise-patterns)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

Comprehensive ML Testing Framework specifically designed for cryptocurrency trading systems. Built with Enterprise Patterns for production-ready ML model validation, data quality assurance, and performance benchmarking.

### 🚀 Key Features

- **🔍 Model Validation**: Comprehensive testing of ML models (classification, regression, time series)
- **📊 Data Quality Testing**: Automated detection of data issues, drift, and anomalies
- **⚡ Performance Benchmarking**: Latency, throughput, memory usage, and scalability testing
- **📈 Drift Detection**: Statistical tests for data and concept drift detection
- **🎨 Advanced Visualization**: Interactive dashboards and publication-ready reports
- **🔧 Test Automation**: Comprehensive test harness with parallel execution
- **💰 Crypto-Specific**: Specialized features for cryptocurrency trading scenarios

## 📁 Project Structure

```

ml-testing/
├── src/
│   ├── model_testing/           # Model validation and testing
│   │   ├── model_validator.py   # Core model validation
│   │   ├── performance_tester.py # Performance testing
│   │   └── regression_tester.py  # Regression testing
│   ├── data_testing/           # Data quality and validation
│   │   ├── data_quality.py     # Data quality validator
│   │   ├── feature_validator.py # Feature engineering validation
│   │   └── drift_detector.py   # Data drift detection
│   ├── benchmarking/          # Performance benchmarking
│   │   ├── benchmark_runner.py # Benchmark execution
│   │   └── metrics_calculator.py # Metrics calculation
│   ├── integration/           # Integration components
│   │   ├── test_harness.py    # Main test orchestrator
│   │   └── pytest_fixtures.py # pytest fixtures
│   ├── utils/                 # Utilities
│   │   ├── test_data_generator.py # Test data generation
│   │   ├── visualization.py   # Visualization utilities
│   │   └── index.ts          # TypeScript exports
│   └── __init__.py
├── tests/
│   └── test_framework.py      # Comprehensive test suite
├── package.json              # Node.js configuration
├── pyproject.toml           # Python configuration
└── README.md               # This file

```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for TypeScript integration)
- Poetry or pip for dependency management

### Python Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using poetry
poetry install

```

### Node.js Dependencies

```bash
npm install
# or
yarn install

```

### Key Dependencies

- **ML Libraries**: scikit-learn, pandas, numpy
- **Deep Learning**: torch, tensorflow (optional)
- **Statistical Testing**: scipy, statsmodels
- **Visualization**: matplotlib, seaborn, plotly
- **Async/Performance**: asyncio, psutil
- **Testing**: pytest, pytest-asyncio
- **MLflow**: Experiment tracking and model management

## 🚀 Quick Start

### 1. Basic Model Validation

```python
from src.model_testing.model_validator import create_crypto_trading_validator
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Create validator
validator = create_crypto_trading_validator()

# Your model and data
model = RandomForestClassifier()
X = pd.DataFrame(...)  # Your features
y = pd.Series(...)     # Your targets

# Train model
model.fit(X, y)

# Validate
result = validator.validate_model(model, X, y)
print(f"Model is valid: {result.is_valid}")
print(f"Validation score: {result.validation_score:.3f}")

```

### 2. Data Quality Testing

```python
from src.data_testing.data_quality import create_crypto_data_validator

# Create data validator
data_validator = create_crypto_data_validator()

# Validate your data
quality_result = data_validator.validate_data(your_dataframe)

print(f"Overall quality score: {quality_result.overall_quality_score:.3f}")
print(f"Issues found: {len(quality_result.quality_issues)}")

```

### 3. Performance Benchmarking

```python
from src.benchmarking.benchmark_runner import create_crypto_benchmark_runner

# Create benchmark runner
benchmark_runner = create_crypto_benchmark_runner()

# Run benchmark
benchmark_result = benchmark_runner.run_benchmark(model, test_data)

print(f"Average latency: {benchmark_result.latency_metrics['mean_ms']:.2f}ms")
print(f"Throughput: {benchmark_result.throughput_samples_per_sec:.0f} samples/sec")

```

### 4. Complete Testing Suite

```python
import asyncio
from src.integration.test_harness import create_crypto_trading_harness

async def run_complete_tests():
    # Create test harness
    harness = create_crypto_trading_harness(
        environment="staging",
        risk_tolerance="conservative"
    )

    # Run complete test suite
    results = await harness.run_test_suite(
        model=your_model,
        data=your_data,
        target=your_target
    )

    print(f"Tests completed: {results.passed_count}/{len(results.test_results)}")
    return results

# Run tests
results = asyncio.run(run_complete_tests())

```

## 🔬 Advanced Features

### Data Drift Detection

```python
from src.data_testing.drift_detector import create_crypto_drift_detector

drift_detector = create_crypto_drift_detector()

# Compare reference data with current data
drift_result = drift_detector.detect_drift(
    reference_data=historical_data,
    current_data=new_data
)

if drift_result.drift_detected:
    print(f"⚠️ Data drift detected! Score: {drift_result.drift_score:.3f}")
    print(f"Features with drift: {list(drift_result.feature_drifts.keys())}")

```

### Crypto Market Data Generation

```python
from src.utils.test_data_generator import create_crypto_trading_data_generator
from src.utils.test_data_generator import MarketCondition

# Create data generator
generator = create_crypto_trading_data_generator()

# Generate crypto price data for different market conditions
bull_market_data = generator.generate_crypto_price_data(
    market_condition=MarketCondition.BULL_MARKET,
    symbol="BTCUSDT"
)

volatile_data = generator.generate_crypto_price_data(
    market_condition=MarketCondition.HIGH_VOLATILITY,
    symbol="ETHUSDT"
)

```

### Interactive Visualization

```python
from src.utils.visualization import create_crypto_trading_visualizer

# Create visualizer
visualizer = create_crypto_trading_visualizer(
    interactive=True,
    enterprise_theme=True
)

# Create performance dashboard
dashboard_path = visualizer.create_ml_testing_dashboard(
    test_results=your_test_results,
    title="Crypto Trading ML Dashboard"
)

print(f"Dashboard saved to: {dashboard_path}")

```

## 📊 Enterprise Patterns

This framework implements Enterprise Patterns for production ML systems:

### 1. **Governance & Compliance**

- Automated quality gates and compliance checks
- Audit logging for all testing activities
- Enterprise-grade reporting and documentation

### 2. **Observability & Monitoring**

- Real-time performance monitoring
- Distributed tracing integration
- Comprehensive metrics collection

### 3. **Reliability & Resilience**

- Circuit breaker patterns for external dependencies
- Graceful degradation and error recovery
- Automated rollback capabilities

### 4. **Security & Risk Management**

- Input validation and sanitization
- Secure credential management
- Risk assessment and mitigation

### 5. **Performance & Scalability**

- Parallel test execution
- Resource optimization
- Auto-scaling capabilities

## 🧪 Testing

### Run All Tests

```bash
# Run comprehensive test suite
python -m pytest tests/test_framework.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test categories
python -m pytest tests/test_framework.py::TestMLFrameworkIntegration::test_end_to_end_classification_workflow -v

```

### Test Categories

- **Integration Tests**: End-to-end workflow testing
- **Unit Tests**: Individual component testing
- **Performance Tests**: Benchmark and latency testing
- **Edge Case Tests**: Error handling and recovery
- **Crypto-Specific Tests**: Trading scenario validation

## 📈 Performance Metrics

The framework provides comprehensive metrics for ML model evaluation:

### Classification Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC AUC, PR AUC
- Confusion Matrix Analysis
- Class-specific metrics

### Regression Metrics

- MSE, MAE, RMSE, R²
- Mean Absolute Percentage Error (MAPE)
- Explained Variance Score
- Custom crypto trading metrics

### Performance Metrics

- Prediction Latency (mean, p95, p99)
- Throughput (samples per second)
- Memory Usage (peak, average)
- CPU Utilization

### Business Metrics (Crypto Trading)

- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Risk-Adjusted Returns

## 🔧 Configuration

### Environment Variables

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=crypto_ml_testing

# Testing Configuration
ML_TESTING_PARALLEL=true
ML_TESTING_WORKERS=4
ML_TESTING_TIMEOUT=3600

# Crypto Trading Configuration
TRADING_ENVIRONMENT=staging
RISK_TOLERANCE=conservative
MARKET_DATA_SOURCE=binance

```

### Configuration File

```python
# config.py
ML_TESTING_CONFIG = {
    "model_validation": {
        "strict_mode": True,
        "performance_threshold": 0.8,
        "drift_threshold": 0.1
    },
    "data_quality": {
        "missing_threshold": 0.05,
        "outlier_threshold": 0.02,
        "quality_score_min": 0.7
    },
    "benchmarking": {
        "latency_sla_ms": 100,
        "throughput_min": 1000,
        "memory_limit_mb": 512
    }
}

```

## 📚 API Reference

### Core Classes

#### ModelValidator

```python
class ModelValidator:
    def validate_model(self, model, data, target) -> ValidationResult
    def check_determinism(self, model, data) -> bool
    def detect_overfitting(self, model, train_data, val_data) -> OverfittingResult

```

#### DataQualityValidator

```python
class DataQualityValidator:
    def validate_data(self, data, target=None) -> QualityResult
    def check_missing_values(self, data) -> MissingValueResult
    def detect_outliers(self, data) -> OutlierResult

```

#### BenchmarkRunner

```python
class BenchmarkRunner:
    def run_benchmark(self, model, data) -> BenchmarkResult
    def test_latency(self, model, data) -> LatencyMetrics
    def test_throughput(self, model, data) -> ThroughputMetrics

```

### Factory Functions

All components can be created using factory functions with crypto trading defaults:

```python
# Model Testing
create_crypto_trading_validator()
create_crypto_performance_tester()
create_crypto_regression_tester()

# Data Testing
create_crypto_data_validator()
create_crypto_feature_validator()
create_crypto_drift_detector()

# Benchmarking
create_crypto_benchmark_runner()
create_crypto_metrics_calculator()

# Integration
create_crypto_trading_harness()
create_crypto_trading_data_generator()
create_crypto_trading_visualizer()

```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd ml-testing

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Enterprise Patterns for ML architecture guidance
- scikit-learn community for ML testing best practices
- Crypto trading community for domain-specific requirements
- MLflow for experiment tracking integration

## License

MIT

## Support

For questions and support, please open an issue on GitHub.
