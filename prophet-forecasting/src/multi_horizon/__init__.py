"""
Multi-Horizon Forecasting Engine
ML-Framework-1329 - Enterprise-grade multi-horizon forecasting system

 2025: Concurrent prediction orchestration, adaptive model selection,
performance optimization with sub-100ms targets.

## Key Features

### 🎯 Multi-Horizon Engine
- Concurrent prediction orchestration across multiple time horizons
- Adaptive strategy selection (parallel, sequential, priority-based)
- Sub-100ms performance targets with enterprise patterns
- Circuit breaker and fault tolerance

### 🔧 Configuration Management
- Comprehensive horizon configuration system
- Prophet parameter optimization
- Performance constraints and validation settings
- Horizon-specific model parameters (ultra-short to strategic)

### 🎨 Ensemble Forecasting
- Multiple ensemble strategies (weighted, Bayesian, adaptive)
- Confidence-based weighting and time alignment
- Model disagreement analysis and uncertainty quantification

### 🏗️ Model Orchestration
- Lifecycle management for multiple forecasting models
- Health monitoring and automatic retraining
- Model caching and performance validation
- Cross-validation and backtesting support

### ⚡ Performance Optimization
- Machine learning-driven parameter optimization
- Adaptive performance tuning based on historical data
- Resource constraint optimization
- Strategy recommendation system

### 📊 Performance Monitoring
- Real-time performance monitoring and alerting
- Comprehensive metrics collection and analytics
- System resource monitoring
- Enterprise-grade observability

## Usage Example

```python
from multi_horizon import (
    MultiHorizonEngine, HorizonConfig, HorizonType,
    EnsembleForecaster, ModelOrchestrator
)

# Create horizon configurations
configs = [
    HorizonConfig.create_ultra_short_config("scalp", "1m"),
    HorizonConfig.create_short_config("day", "5m"),
    HorizonConfig.create_medium_config("swing", "1h"),
    HorizonConfig.create_long_config("position", "1d")
]

# Initialize multi-horizon engine
engine = MultiHorizonEngine(
    symbol="BTC",
    horizons_config=configs,
    performance_target_ms=85.0,
    enable_ensemble=True
)

# Initialize and predict
await engine.initialize()
result = await engine.predict(ohlcv_data, periods_ahead=30)

print(f"Ensemble confidence: {result.average_confidence:.3f}")
print(f"Execution time: {result.total_execution_time_ms:.1f}ms")
```

## Architecture

The multi-horizon system implements  2025 patterns:

1. **Concurrent Orchestration**: Parallel execution across horizons
2. **Adaptive Configuration**: Dynamic parameter optimization
3. **Circuit Breaker Pattern**: Fault tolerance and recovery
4. **Performance Monitoring**: Real-time observability
5. **Ensemble Intelligence**: Advanced model combination
"""

from .multi_horizon_engine import MultiHorizonEngine, PredictionStrategy, EngineState, HorizonPrediction, MultiHorizonResult
from .horizon_config import (
    HorizonConfig, HorizonType, ModelType, SeasonalityMode,
    ProphetParameters, PerformanceConstraints, ValidationSettings, DataProcessingConfig
)
from .ensemble_forecaster import (
    EnsembleForecaster, EnsembleStrategy, TimeAlignment,
    EnsembleWeights, EnsemblePrediction
)
from .model_orchestrator import (
    ModelOrchestrator, ModelState, ModelHealth,
    ModelMetadata, ModelInstance
)
from .horizon_optimizer import (
    HorizonOptimizer, OptimizationStrategy, OptimizationScope,
    OptimizationMetrics, OptimizationRecommendation, OptimizationResult
)
from .performance_monitor import (
    HorizonPerformanceMonitor, AlertSeverity, MetricType,
    Alert, PerformanceMetric, PerformanceSnapshot
)

__all__ = [
    # Core engine
    "MultiHorizonEngine",
    "PredictionStrategy",
    "EngineState",
    "HorizonPrediction",
    "MultiHorizonResult",

    # Configuration
    "HorizonConfig",
    "HorizonType",
    "ModelType",
    "SeasonalityMode",
    "ProphetParameters",
    "PerformanceConstraints",
    "ValidationSettings",
    "DataProcessingConfig",

    # Ensemble forecasting
    "EnsembleForecaster",
    "EnsembleStrategy",
    "TimeAlignment",
    "EnsembleWeights",
    "EnsemblePrediction",

    # Model orchestration
    "ModelOrchestrator",
    "ModelState",
    "ModelHealth",
    "ModelMetadata",
    "ModelInstance",

    # Performance optimization
    "HorizonOptimizer",
    "OptimizationStrategy",
    "OptimizationScope",
    "OptimizationMetrics",
    "OptimizationRecommendation",
    "OptimizationResult",

    # Performance monitoring
    "HorizonPerformanceMonitor",
    "AlertSeverity",
    "MetricType",
    "Alert",
    "PerformanceMetric",
    "PerformanceSnapshot",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__description__ = "Enterprise multi-horizon forecasting with  2025 patterns"