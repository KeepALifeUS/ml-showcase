# ML AutoML Pipeline for Crypto Trading Bot

[![CI](https://github.com/KeepALifeUS/ml-automl-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/KeepALifeUS/ml-automl-pipeline/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Comprehensive automated machine learning pipeline specifically designed for cryptocurrency trading applications. Built with enterprise-grade **enterprise patterns** for production-ready scalability, reliability, and maintainability.

### Key Features

- **Automated Feature Engineering**: Technical indicators, statistical features, polynomial transformations, and TSFresh time series features
- **Advanced Feature Selection**: Multi-method ensemble selection with statistical, model-based, and correlation-based approaches
- **Hyperparameter Optimization**: Bayesian optimization with Optuna and scikit-optimize
- **Automated Model Selection**: Support for 15+ ML algorithms with automatic performance comparison
- **Ensemble Building**: Voting, stacking, and blending ensemble methods
- **Comprehensive Evaluation**: Multiple metrics, feature importance, and detailed reporting
- **Production Ready**: enterprise patterns, logging, monitoring, and error handling

## Architecture

```

ml-automl-pipeline/
├── src/
│   ├── feature_engineering/     # Automated feature generation & selection
│   │   ├── auto_feature_generator.py
│   │   ├── feature_selector.py
│   │   └── feature_transformer.py
│   ├── optimization/            # Hyperparameter optimization
│   │   ├── bayesian_optimizer.py
│   │   ├── hyperparameter_tuner.py
│   │   └── optuna_integration.py
│   ├── model_selection/         # Model selection & ensembling
│   │   ├── model_selector.py
│   │   ├── ensemble_builder.py
│   │   └── cross_validator.py
│   ├── pipeline/               # Main AutoML orchestration
│   │   ├── automl_pipeline.py
│   │   └── pipeline_orchestrator.py
│   ├── evaluation/             # Model evaluation & metrics
│   │   ├── model_evaluator.py
│   │   └── leaderboard.py
│   └── utils/                  # Configuration & utilities
│       ├── config_manager.py
│       └── data_preprocessor.py
├── tests/                      # Comprehensive test suite
└── examples/                   # Usage examples

```

## Quick Start

### Installation

```bash
# Install package dependencies
pnpm install

# Install Python dependencies
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml

```

### Basic Usage

```python
import pandas as pd
import numpy as np
from src.pipeline.automl_pipeline import AutoMLPipeline
from src.utils.config_manager import AutoMLConfig, PresetConfigs

# Load your crypto trading data
data = pd.read_csv('crypto_data.csv')  # OHLCV format expected

# Create AutoML pipeline with crypto-specific configuration
config = PresetConfigs.crypto_trading()
pipeline = AutoMLPipeline(config, output_dir="automl_results")

# Run complete AutoML pipeline
result = pipeline.run(
    data=data,
    target_column='future_return',  # Your target variable
    test_size=0.2,
    time_series_split=True  # Important for crypto data
)

# Access results
print(f"Best model: {result.best_model_name}")
print(f"Best score: {result.best_score:.4f}")
print(f"Processing time: {result.total_time:.2f}s")

# Use the trained model
predictions = result.best_model.predict(new_data)

```

### Advanced Configuration

```python
from src.utils.config_manager import AutoMLConfig

# Create custom configuration
config = AutoMLConfig()

# Feature engineering settings
config.feature_generation.enable_technical_indicators = True
config.feature_generation.technical_indicators_windows = [5, 10, 20, 50]
config.feature_generation.enable_polynomial_features = True
config.feature_generation.polynomial_degree = 2

# Hyperparameter optimization settings
config.hyperparameter_optimization.default_optimizer = "optuna_tpe"
config.hyperparameter_optimization.n_trials = 100

# Model selection settings
config.model_selection.sklearn_models = ['ridge', 'random_forest', 'gradient_boosting']
config.model_selection.gradient_boosting_models = ['xgboost', 'lightgbm']

# Save configuration for reproducibility
config.save_to_file("my_automl_config.json")

```

## Feature Engineering

### Technical Indicators for Crypto

- Moving Averages (SMA, EMA)
- MACD and Signal Lines
- Bollinger Bands and Band Position
- RSI and Stochastic Oscillators
- Volume indicators (VWAP, Volume SMA)
- ATR (Average True Range)
- Price change ratios and momentum

### Statistical Features

- Rolling statistics (mean, std, min, max, median)
- Quantiles and z-scores
- Skewness and kurtosis
- Lag features and differencing
- Position within rolling windows

### Advanced Features

- Polynomial feature interactions
- TSFresh time series features
- Feature selection based on importance
- Correlation-based feature filtering

## Hyperparameter Optimization

### Supported Optimizers

- **Optuna TPE**: Tree-structured Parzen Estimator (recommended)
- **Optuna Random**: Random sampling baseline
- **Gaussian Process**: Scikit-optimize GP-based optimization
- **Random Forest**: RF-based surrogate model
- **Gradient Boosting**: GBRT-based optimization

### Optimization Features

- Multi-objective optimization support
- Early stopping and pruning
- Parallel optimization
- Custom search spaces per model
- Convergence monitoring and visualization

## Model Selection

### Supported Models

#### Traditional ML

- Linear/Ridge/Lasso/ElasticNet Regression
- Random Forest and Extra Trees
- Gradient Boosting
- Support Vector Machines
- K-Nearest Neighbors
- Neural Networks (MLP)

#### Gradient Boosting

- **XGBoost**: eXtreme Gradient Boosting
- **LightGBM**: Microsoft's gradient boosting
- **CatBoost**: Yandex's categorical boosting

#### Model Selection Features

- Automated algorithm detection
- Time series cross-validation
- Performance-based ranking
- Resource usage monitoring

## Ensemble Methods

### Ensemble Types

- **Voting**: Weighted average of predictions
- **Stacking**: Meta-learner on base model predictions
- **Blending**: Holdout-based meta-learning
- **Dynamic Weighting**: Adaptive ensemble weights

### Ensemble Features

- Automatic diversity calculation
- Ensemble size optimization
- Cross-validation for meta-learning
- Ensemble performance comparison

## Model Evaluation

### Regression Metrics

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Mean Absolute Percentage Error (MAPE)
- Root Mean Squared Error (RMSE)

### Classification Metrics

- Accuracy and Balanced Accuracy
- Precision, Recall, F1-Score
- ROC AUC and PR AUC
- Confusion Matrix

### Crypto-Specific Metrics

- Risk-adjusted returns
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
- Win rate and profit factor

## Configuration Presets

### Available Presets

```python
from src.utils.config_manager import PresetConfigs

# Fast prototyping (reduced iterations)
config = PresetConfigs.fast_prototype()

# Production-ready (full optimization)
config = PresetConfigs.production_ready()

# Crypto trading specialized
config = PresetConfigs.crypto_trading()

# High-frequency trading optimized
config = PresetConfigs.high_frequency_trading()

```

### Preset Comparison

| Feature             | Fast Prototype | Production | Crypto Trading  | HFT     |
| ------------------- | -------------- | ---------- | --------------- | ------- |
| Optimization Trials | 20             | 200        | 100             | 50      |
| CV Folds            | 3              | 10         | 5               | 3       |
| Feature Generation  | Basic          | Full       | Crypto-specific | Minimal |
| Model Types         | 3              | 10+        | 7               | 2       |
| Ensemble Methods    | None           | All        | Voting+Stacking | None    |
| Processing Time     | ~5 min         | ~2 hours   | ~30 min         | ~10 min |

## Production Deployment

### Model Persistence

```python
# Models are automatically saved during pipeline execution
# Load saved model
import joblib

model = joblib.load("automl_results/best_model.pkl")
predictions = model.predict(new_data)

# Load complete pipeline state
from src.pipeline.automl_pipeline import AutoMLPipeline

pipeline = AutoMLPipeline.load_from_checkpoint("automl_results/")
result = pipeline.predict(new_data)

```

### Monitoring and Logging

```python
# Configure detailed logging
import logging
from loguru import logger

# Enable debug mode
config = AutoMLConfig()
config.debug_mode = True
config.verbose = True

# Custom log configuration
logger.add("automl_debug.log", level="DEBUG")
logger.add("automl_errors.log", level="ERROR")

```

### Performance Optimization

```python
# Resource management
config = AutoMLConfig()
config.max_memory_gb = 16.0  # Memory limit
config.max_training_time = 7200  # 2 hours max
config.n_jobs = 8  # Parallel processes

# Enable caching for faster reruns
config.enable_caching = True
config.cache_dir = "automl_cache"

```

## Crypto Trading Integration

### Data Format Requirements

```python
# Expected OHLCV format
data = pd.DataFrame({
    'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
    'future_return': [...]  # Target variable
})

```

### Time Series Considerations

- **Time Series Split**: Prevents data leakage
- **Walk-Forward Validation**: Simulates real trading conditions
- **Lag Features**: Incorporates historical patterns
- **Regime Detection**: Adapts to market conditions

### Risk Management Features

- **Volatility Features**: Market volatility indicators
- **Momentum Features**: Trend following signals
- **Risk-Adjusted Metrics**: Sharpe, Sortino ratios
- **Drawdown Analysis**: Maximum drawdown calculation

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_feature_engineering.py -v
python -m pytest tests/test_optimization.py -v
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

```

## Examples

### Complete Crypto Trading Example

```python
import pandas as pd
import numpy as np
from src.pipeline.automl_pipeline import AutoMLPipeline
from src.utils.config_manager import PresetConfigs

# Generate sample crypto data
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=2000, freq='1H')

data = pd.DataFrame({
    'open': np.random.randn(2000).cumsum() + 50000,
    'high': np.random.randn(2000).cumsum() + 50100,
    'low': np.random.randn(2000).cumsum() + 49900,
    'close': np.random.randn(2000).cumsum() + 50000,
    'volume': np.random.exponential(1000, 2000)
}, index=dates)

# Create target: next period return
data['future_return'] = data['close'].shift(-1) / data['close'] - 1
data = data.dropna()

# Configure for crypto trading
config = PresetConfigs.crypto_trading()
config.hyperparameter_optimization.n_trials = 50  # Reduce for demo

# Create and run pipeline
pipeline = AutoMLPipeline(config, output_dir="crypto_automl_results")
result = pipeline.run(
    data=data,
    target_column='future_return',
    test_size=0.2,
    time_series_split=True
)

print(f"Best model: {result.best_model_name}")
print(f"Test R²: {result.evaluation_result.test_r2:.4f}")
print(f"Total time: {result.total_time:.1f}s")

```

### Feature Engineering Only

```python
from src.feature_engineering.auto_feature_generator import AutoFeatureGenerator
from src.utils.config_manager import AutoMLConfig

# Configure feature generation
config = AutoMLConfig()
generator = AutoFeatureGenerator(config)

# Generate features
result = generator.generate_features(
    data[['open', 'high', 'low', 'close', 'volume']],
    parallel=True
)

print(f"Generated {len(result.feature_names)} features")
print(f"Processing time: {result.processing_time:.2f}s")

# View feature importance
importance_ranking = generator.get_feature_importance_ranking(result)
for feature, importance in importance_ranking[:10]:
    print(f"{feature}: {importance:.4f}")

```

### Model Selection and Optimization

```python
from src.model_selection.model_selector import ModelSelector
from src.optimization.bayesian_optimizer import CryptoMLHyperparameterOptimizer

# Select best models
selector = ModelSelector()
selection_result = selector.select_best_models(
    X, y,
    models=['xgboost', 'lightgbm', 'random_forest'],
    cv_folds=5,
    time_series_split=True
)

print(f"Best models: {selection_result.best_models}")

# Optimize hyperparameters
optimizer = CryptoMLHyperparameterOptimizer()
optimization_results = optimizer.optimize_multiple_models(
    X, y,
    models=selection_result.best_models[:3],
    n_calls=50
)

for model, result in optimization_results.items():
    print(f"{model}: {result.best_score:.4f}")

```

## Performance Tips

### Speed Optimization

- Use `PresetConfigs.fast_prototype()` for development
- Reduce `n_trials` for hyperparameter optimization
- Limit `cv_folds` to 3-5 for initial testing
- Enable parallel processing with `n_jobs=-1`
- Use caching for repeated experiments

### Memory Optimization

- Set appropriate `max_memory_gb` limits
- Reduce `polynomial_max_features` if needed
- Disable heavy features like TSFresh for large datasets
- Use data sampling for initial exploration

### Quality Optimization

- Use `PresetConfigs.production_ready()` for final models
- Increase `n_trials` to 200+ for production
- Use 10-fold cross-validation for stable estimates
- Enable all ensemble methods
- Validate with walk-forward analysis

## Troubleshooting

### Common Issues

**Memory Errors**

```python
# Reduce memory usage
config.feature_generation.polynomial_max_features = 50
config.feature_generation.enable_tsfresh_features = False
config.max_memory_gb = 4.0

```

**Long Processing Time**

```python
# Speed up processing
config = PresetConfigs.fast_prototype()
config.hyperparameter_optimization.n_trials = 20
config.model_selection.cv_folds = 3

```

**Model Performance Issues**

```python
# Improve model quality
config.feature_selection.ensemble_selection = True
config.ensemble.enable_stacking = True
config.hyperparameter_optimization.n_trials = 100

```

### Debug Mode

```python
# Enable comprehensive logging
config = AutoMLConfig()
config.debug_mode = True
config.verbose = True

# Check intermediate results
result = pipeline.run(data, target_column, stages=['data_preprocessing', 'feature_generation'])

```

## Integration with ML-Framework

This AutoML pipeline is designed to integrate seamlessly with the Crypto Trading Bot v5.0 ecosystem:

- **Market Data Integration**: Direct connection with `@ml-framework/market-data`
- **Strategy Integration**: Models can be used in `@ml-framework/trading-strategies`
- **Risk Management**: Outputs compatible with `@ml-framework/risk-management`
- **Backtesting**: Models integrate with `@ml-framework/backtesting-engine`

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For support and questions:

- Create an issue in the repository
- Check the documentation in `/docs`
- Review examples in `/examples`

---

**Built with enterprise patterns for Production-Ready ML Systems**

## Support

For questions and support, please open an issue on GitHub.
