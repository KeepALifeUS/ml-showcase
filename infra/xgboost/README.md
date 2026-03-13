# ML XGBoost

[![CI](https://github.com/KeepALifeUS/ml-xgboost/actions/workflows/ci.yml/badge.svg)](https://github.com/KeepALifeUS/ml-xgboost/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Enterprise XGBoost implementation with SHAP/LIME interpretability and advanced feature engineering for time series prediction.**

ML XGBoost provides a production-ready framework built on top of XGBoost, specifically optimized for time series forecasting. It combines advanced feature engineering pipelines, automated hyperparameter tuning, model interpretability via SHAP, and a FastAPI-based prediction service -- all in a single, well-structured package.

---

## Features

- **Time Series Optimized Models** -- XGBoost models pre-configured for temporal data with automatic lag feature generation, rolling window statistics, and volatility proxies
- **Comprehensive Feature Engineering** -- Modular pipeline with technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, etc.), statistical features, interaction features, lagging features, and volatility estimators
- **SHAP Interpretability** -- Built-in SHAP TreeExplainer integration for feature importance analysis, summary plots, and per-prediction explanations
- **Automated Hyperparameter Tuning** -- Multiple optimization strategies including Optuna TPE, CMA-ES, Bayesian optimization (Gaussian Process / Random Forest), and random search
- **FastAPI Prediction Service** -- REST API endpoints for real-time single and batch predictions with model management
- **Time Series Cross-Validation** -- Walk-forward validation, expanding window cross-validation, and backtesting engine
- **Ensemble Models** -- Ensemble, adaptive, and hierarchical XGBoost model variants
- **Specialized Models** -- Purpose-built models for volatility prediction, trend detection, and anomaly detection
- **GPU Support** -- Optional CUDA acceleration via RAPIDS (cuDF, cuML)
- **Numba Acceleration** -- JIT-compiled utility functions for fast rolling statistics and indicator calculations

---

## Architecture

```
ml-xgboost/
├── src/
│   ├── __init__.py              # Package exports and initialization
│   ├── models/
│   │   ├── xgb_time_series.py   # Core models: Regressor, Classifier, TimeSeries, Ranker
│   │   ├── ensemble_models.py   # Ensemble, Adaptive, Hierarchical XGBoost
│   │   └── specialized_models.py # Volatility, Trend, Anomaly models
│   ├── features/
│   │   ├── feature_engineer.py  # Main FeatureEngineer pipeline (sklearn-compatible)
│   │   ├── technical_indicators.py # 50+ technical analysis indicators
│   │   ├── statistical_features.py # Rolling statistics (skewness, kurtosis, quantiles)
│   │   ├── lagging_features.py  # Lag and difference features
│   │   ├── interaction_features.py # Feature products and ratios
│   │   └── volatility_features.py  # Realized, Parkinson, vol-of-vol
│   ├── tuning/
│   │   ├── hyperparameter_tuner.py # Main tuner orchestrator with multiple strategies
│   │   ├── optuna_tuner.py      # Optuna-specific tuner
│   │   ├── bayesian_optimizer.py # Bayesian optimization
│   │   └── grid_search_tuner.py # Exhaustive grid search
│   ├── validation/
│   │   └── cross_validator.py   # TimeSeriesCV, WalkForward, Backtest engine
│   ├── api/
│   │   └── prediction_service.py # FastAPI model serving (single + batch)
│   └── utils/
│       ├── model_utils.py       # Save/Load, Metrics, PerformanceAnalyzer
│       └── interpretability.py  # SHAP explainer, feature importance, reports
└── pyproject.toml               # Project configuration and dependencies
```

---

## Installation

### From Source

```bash
git clone https://github.com/KeepALifeUS/ml-xgboost.git
cd ml-xgboost
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### With GPU Support (CUDA)

```bash
pip install -e ".[gpu]"
```

### Core Dependencies

- Python >= 3.11
- xgboost >= 2.0.0
- scikit-learn >= 1.5.0
- pandas >= 2.1.0
- numpy >= 1.24.0
- shap >= 0.43.0
- optuna >= 3.4.0
- fastapi >= 0.104.0
- loguru >= 0.7.0

---

## Quick Start

### Training a Time Series Model

```python
from ml_xgboost import XGBoostTimeSeriesModel, FeatureEngineer

# Create model with time series defaults
model = XGBoostTimeSeriesModel(
    enable_shap=True,
    time_feature_engineering=True,
    lag_features=10,
    rolling_windows=[5, 10, 20, 50],
    volatility_features=True,
)

# Train (automatically creates lag, rolling, and volatility features)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance(importance_type="gain", top_k=20)
print(importance)

# Get SHAP values for interpretability
shap_values = model.get_shap_values(X_test)
model.plot_shap_summary(X_test, max_display=20)
```

### Feature Engineering Pipeline

```python
from ml_xgboost.features import FeatureEngineer, FeatureConfig

# Configure feature generation
config = FeatureConfig(
    enable_technical_indicators=True,
    enable_statistical_features=True,
    enable_lag_features=True,
    enable_interaction_features=True,
    enable_volatility_features=True,
    feature_selection_k=100,
    scaling_method="robust",
    parallel_processing=True,
    max_workers=4,
)

# Create pipeline (sklearn-compatible fit/transform)
engineer = FeatureEngineer(config)
X_features = engineer.fit_transform(ohlcv_data, target)

# Inspect results
importance = engineer.get_feature_importance(top_k=20)
engineer.plot_feature_importance(top_k=20, save_path="importance.png")
engineer.plot_correlation_heatmap(save_path="correlation.png")

# Save/load the fitted engineer
engineer.save_engineer("artifacts/feature_engineer")
loaded_engineer = FeatureEngineer.load_engineer("artifacts/feature_engineer")
```

### Hyperparameter Tuning

```python
from ml_xgboost.tuning import HyperparameterTuner, TuningConfig, OptimizationObjective

config = TuningConfig(
    strategy="optuna_tpe",
    objective=OptimizationObjective.MINIMIZE_RMSE,
    n_trials=200,
    cv_folds=5,
    cv_strategy="time_series",
    enable_pruning=True,
    timeout_seconds=3600,
)

tuner = HyperparameterTuner(config)
result = tuner.optimize(
    model_class=xgb.XGBRegressor,
    X=X_train,
    y=y_train,
)

print(f"Best score: {result.best_score:.4f}")
print(f"Best params: {result.best_params}")

# Create model with best params
best_model = tuner.get_best_model(xgb.XGBRegressor, result)

# Visualize optimization
tuner.plot_optimization_history(result, save_path="optimization.png")
tuner.plot_parameter_importance(result, save_path="param_importance.png")
```

### Model Serving with FastAPI

```python
from ml_xgboost.api import PredictionService, ModelServer

# Setup service
service = PredictionService()
service.load_model(trained_model, model_name="my_model")

# Create server
server = ModelServer(service)

# Run (available at http://localhost:8000)
server.run(host="0.0.0.0", port=8000)

# Endpoints:
#   GET  /health          - Health check
#   POST /predict         - Single prediction
#   POST /predict_batch   - Batch predictions
#   GET  /models          - List loaded models
```

### Cross-Validation and Backtesting

```python
from ml_xgboost.validation import (
    TimeSeriesCrossValidator,
    WalkForwardValidator,
    BacktestEngine,
)

# Time series cross-validation
cv = TimeSeriesCrossValidator(n_splits=5)
cv_results = cv.cross_validate(model, X, y, scoring="neg_mean_squared_error")
print(f"CV Mean: {cv_results['mean_score']:.4f} +/- {cv_results['std_score']:.4f}")

# Walk-forward validation
wf = WalkForwardValidator(initial_train_size=500, step_size=50)
wf_results = wf.validate(model, X, y)
print(f"Walk-Forward RMSE: {wf_results['overall_rmse']:.4f}")

# Backtesting
engine = BacktestEngine(initial_capital=10000)
bt_results = engine.backtest(predictions, actual_prices, transaction_cost=0.001)
print(f"Sharpe Ratio: {bt_results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {bt_results['max_drawdown']:.2%}")
```

### Model Interpretability

```python
from ml_xgboost.utils import SHAPExplainer, ModelInterpreter, ExplanationGenerator

# SHAP analysis
explainer = SHAPExplainer(trained_model)
shap_values = explainer.get_shap_values(X_test)
explainer.plot_summary(X_test, max_display=20, save_path="shap_summary.png")

# Per-prediction interpretation
interpreter = ModelInterpreter(trained_model)
interpretation = interpreter.interpret_prediction(X_test.iloc[:1], feature_names)
explanation = interpreter.generate_explanation(interpretation)
print(explanation)

# Full report
generator = ExplanationGenerator()
report = generator.generate_report(
    trained_model, X_test.iloc[:1], feature_names,
    save_path="explanation_report.txt"
)
```

---

## Configuration

### TrainingConfig (Model Training)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.1 | Step size shrinkage |
| `subsample` | 0.8 | Row subsampling ratio |
| `colsample_bytree` | 0.8 | Column subsampling per tree |
| `reg_alpha` | 0.0 | L1 regularization |
| `reg_lambda` | 1.0 | L2 regularization |
| `early_stopping_rounds` | 50 | Early stopping patience |
| `cv_folds` | 5 | Cross-validation folds |

### FeatureConfig (Feature Engineering)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_technical_indicators` | True | Generate technical analysis features |
| `enable_statistical_features` | True | Rolling statistical features |
| `enable_lag_features` | True | Time-lagged features |
| `enable_interaction_features` | True | Feature cross-products |
| `enable_volatility_features` | True | Volatility estimators |
| `feature_selection_k` | 100 | Top K features to select |
| `scaling_method` | robust | Data scaling (robust, standard, minmax) |
| `parallel_processing` | True | Parallel feature generation |

### TuningConfig (Hyperparameter Optimization)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | optuna_tpe | Optimization strategy |
| `n_trials` | 100 | Number of trials |
| `timeout_seconds` | 3600 | Maximum optimization time |
| `cv_folds` | 5 | Cross-validation folds |
| `enable_pruning` | True | Early trial pruning |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test markers
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
pytest -m "not gpu"      # Skip GPU tests
```

---

## CLI Commands

The package provides CLI entry points (when installed):

```bash
xgb-train      # Train a model
xgb-predict    # Generate predictions
xgb-tune       # Run hyperparameter tuning
xgb-explain    # Generate model explanations
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Support

For questions and support, please open an issue on GitHub.
