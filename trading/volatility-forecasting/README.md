# ML Volatility Forecasting

> **Enterprise-grade volatility forecasting system** with GARCH family models, realized volatility estimators, ML-based prediction, and real-time risk metrics.

## Overview

Production-ready volatility forecasting framework combining classical econometric models (GARCH family) with modern machine learning approaches for accurate volatility prediction and risk management.

### Key Capabilities

- **GARCH Family Models** — GARCH(1,1), EGARCH, GJR-GARCH, FIGARCH, DCC-GARCH with automatic model selection
- **Realized Volatility** — RV, Bipower Variation, Realized Kernels, Realized GARCH
- **Implied Volatility** — Black-Scholes IV extraction, volatility surface interpolation
- **ML Models** — LSTM networks, HAR-RV, Random Forest for volatility prediction
- **Risk Metrics** — VaR, CVaR, volatility cones, tail risk analysis
- **Real-Time API** — FastAPI with WebSocket streaming for live volatility updates
- **Trading Integration** — Position sizing, volatility-adjusted stop losses

## Architecture

```
src/
├── models/
│   ├── garch_models.py          # GARCH, EGARCH, GJR-GARCH, FIGARCH, DCC-GARCH
│   └── ml_volatility.py         # LSTM, HAR-RV, Random Forest models
├── estimators/
│   ├── realized_volatility.py   # RV, BPV, Realized Kernels
│   └── implied_volatility.py    # Black-Scholes IV, volatility surface
├── core/
│   └── volatility_system.py     # Unified forecasting system
├── api/
│   └── volatility_api.py        # FastAPI + WebSocket endpoints
├── utils/
│   ├── risk_metrics.py          # VaR, CVaR, volatility cones
│   └── trading_integration.py   # Position sizing, risk management
└── validation/
    └── volatility_validator.py  # Model validation & backtesting
```

## GARCH Models

The framework implements the complete GARCH family with automatic model selection:

| Model | Use Case | Key Feature |
|-------|----------|-------------|
| GARCH(1,1) | Baseline volatility clustering | Standard conditional heteroskedasticity |
| EGARCH | Asymmetric volatility | Leverage effect modeling |
| GJR-GARCH | News impact asymmetry | Threshold effects for positive/negative shocks |
| FIGARCH | Long memory | Fractionally integrated for persistent volatility |
| DCC-GARCH | Multi-asset | Dynamic conditional correlations |

```python
from ml_volatility_forecasting import GARCHModelSelector

selector = GARCHModelSelector()
best_model = selector.select_best_model(
    returns_data,
    candidates=["garch", "egarch", "gjr-garch", "figarch"],
    criterion="bic"  # AIC, BIC, or log-likelihood
)

forecast = best_model.forecast(horizon=10)
```

## Risk Metrics

Built-in risk analysis tools:

```python
from ml_volatility_forecasting import RiskMetrics

risk = RiskMetrics(returns_data)

# Value at Risk
var_95 = risk.calculate_var(confidence=0.95, method="historical")
var_99 = risk.calculate_var(confidence=0.99, method="parametric")

# Conditional VaR (Expected Shortfall)
cvar = risk.calculate_cvar(confidence=0.95)

# Volatility cones
cones = risk.volatility_cones(windows=[7, 14, 30, 60, 90])
```

## ML Volatility Models

Combine classical and ML approaches:

```python
from ml_volatility_forecasting import LSTMVolatilityModel, HARRVModel

# LSTM for sequence modeling
lstm = LSTMVolatilityModel(
    lookback=60,
    hidden_size=128,
    num_layers=2
)
lstm.fit(training_data)
prediction = lstm.predict(horizon=5)

# HAR-RV (Heterogeneous Autoregressive Realized Volatility)
har = HARRVModel()
har.fit(realized_vol_data)
```

## Real-Time API

FastAPI server with WebSocket streaming:

```python
# Start the API server
# uvicorn ml_volatility_forecasting.api.volatility_api:app --host 0.0.0.0 --port 8000

# REST endpoints:
# GET  /api/v1/forecast/{symbol}?horizon=10&model=egarch
# GET  /api/v1/risk-metrics/{symbol}
# GET  /api/v1/volatility-surface/{symbol}
# WS   /ws/volatility-stream
```

## Installation

```bash
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With quantitative finance tools
pip install -e ".[quant]"

# Development
pip install -e ".[dev]"
```

## Configuration

Set environment variables for external services:

```bash
export DATABASE_URL="postgresql://localhost/volatility"
export REDIS_URL="redis://localhost:6379"
```

## Tech Stack

- **Econometrics**: arch, statsmodels, pykalman
- **ML**: PyTorch, TensorFlow, scikit-learn
- **API**: FastAPI, WebSockets, uvicorn
- **Data**: pandas, NumPy, SciPy, Numba (JIT)
- **Storage**: PostgreSQL, Redis, ClickHouse
- **Monitoring**: Prometheus, structlog

## License

MIT License — see [LICENSE](LICENSE) for details.

## Support

For questions and support, please open an issue on GitHub.
