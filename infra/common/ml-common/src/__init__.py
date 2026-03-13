"""
ML Common - Unified Machine Learning Utilities for Crypto Trading Bot v5.0

Enterprise-grade consolidated ML utilities implementing .
Consolidates 5000+ lines of duplicated mathematical functions from 38+ ML packages.

Key Features:
- Technical indicators (40 indicators: SMA, EMA, RSI, MACD, ADX, Aroon, etc.)
- Order book microstructure (20 dims: imbalance, depth, spread)
- Cross-asset features (20 dims: correlation, spreads, beta)
- Market regime classification (10 dims: volatility, trend, time)
- Portfolio tracking (50 dims: positions, PnL, risk metrics)
- Symbol & temporal embeddings (26 dims: learned representations)
- StateVectorBuilder (768-dim integration - CRITICAL component)
- Data preprocessing and normalization
- Evaluation metrics and backtesting
- High-performance computation with Numba
- Type-safe interfaces with full validation

WEEK 2 ENHANCEMENTS (NEW):
- fusion/: StateVectorBuilder (768-dim), WindowManager (168h sliding windows)
- portfolio/: Position tracking, performance metrics, risk analysis
- embeddings/: Symbol embeddings (16 dims), temporal encodings (10 dims)

Usage:
 # Classic indicators
 from ml_common import calculate_sma, calculate_rsi, normalize_data
 from ml_common.indicators import TechnicalIndicators

 # NEW: StateVectorBuilder (CRITICAL)
 from ml_common.fusion import StateVectorBuilder, WindowManager
 builder = StateVectorBuilder(version='v1')
 state_vector = builder.build(ohlcv_data, orderbook_data, portfolio_state)
 # Output: (168, 768) numpy array

 # NEW: Week 1 feature modules
 from ml_common.orderbook import calculate_bid_ask_imbalance
 from ml_common.cross_asset import extract_correlation_features
 from ml_common.regime import classify_volatility_regime
 from ml_common.portfolio import extract_position_features
 from ml_common.embeddings import extract_symbol_embeddings
"""

__version__ = "2.0.0" # Major version bump for Week 2 features
__author__ = "ML-Framework Team"
__email__ = "dev@ml-framework.dev"
__license__ = "MIT"

# Core imports for ease of use
from .indicators.technical import (
 calculate_sma,
 calculate_ema,
 calculate_rsi,
 calculate_macd,
 calculate_bollinger_bands,
 calculate_atr,
 TechnicalIndicators,
 IndicatorConfig
)

from .preprocessing.normalization import (
 normalize_data,
 standardize_data,
 robust_scale_data,
 minmax_scale_data
)

from .evaluation.metrics import (
 calculate_sharpe_ratio,
 calculate_sortino_ratio,
 calculate_max_drawdown,
 calculate_calmar_ratio,
 calculate_win_rate
)

from .evaluation.backtesting import (
 backtest_strategy,
 BacktestResult,
 BacktestConfig
)

from .utils.math_utils import (
 rolling_window,
 exponential_smoothing,
 detect_outliers,
 safe_divide
)

# WEEK 2 ENHANCEMENTS - NEW MODULES
from .fusion import (
 StateVectorBuilder,
 StateVectorConfig,
 StateVectorV1,
 WindowManager,
 WindowConfig,
)

from .orderbook import (
 calculate_bid_ask_imbalance,
 calculate_depth_metrics,
 calculate_spread_metrics,
)

from .cross_asset import (
 extract_correlation_features,
 extract_spread_features,
 extract_beta_features,
)

from .regime import (
 extract_volatility_features,
 extract_trend_features,
 extract_time_features,
)

from .portfolio import (
 extract_position_features,
 extract_performance_features,
)

from .embeddings import (
 extract_symbol_embeddings,
 extract_temporal_embeddings,
)

# Version information
__all__ = [
 # Version info
 "__version__",
 "__author__",
 "__email__",
 "__license__",

 # Technical indicators
 "calculate_sma",
 "calculate_ema",
 "calculate_rsi",
 "calculate_macd",
 "calculate_bollinger_bands",
 "calculate_atr",
 "TechnicalIndicators",
 "IndicatorConfig",

 # Data preprocessing
 "normalize_data",
 "standardize_data",
 "robust_scale_data",
 "minmax_scale_data",

 # Evaluation metrics
 "calculate_sharpe_ratio",
 "calculate_sortino_ratio",
 "calculate_max_drawdown",
 "calculate_calmar_ratio",
 "calculate_win_rate",

 # Backtesting
 "backtest_strategy",
 "BacktestResult",
 "BacktestConfig",

 # Math utilities
 "rolling_window",
 "exponential_smoothing",
 "detect_outliers",
 "safe_divide",

 # WEEK 2 ENHANCEMENTS - Fusion module (CRITICAL)
 "StateVectorBuilder",
 "StateVectorConfig",
 "StateVectorV1",
 "WindowManager",
 "WindowConfig",

 # Week 1 - Orderbook features
 "calculate_bid_ask_imbalance",
 "calculate_depth_metrics",
 "calculate_spread_metrics",

 # Week 1 - Cross-asset features
 "extract_correlation_features",
 "extract_spread_features",
 "extract_beta_features",

 # Week 1 - Regime classification
 "extract_volatility_features",
 "extract_trend_features",
 "extract_time_features",

 # Week 2 - Portfolio tracking
 "extract_position_features",
 "extract_performance_features",

 # Week 2 - Embeddings
 "extract_symbol_embeddings",
 "extract_temporal_embeddings",
]

# Setup logging
import logging
import os

# Configure default logging
logging.basicConfig(
 level=logging.INFO if os.getenv("ML_COMMON_DEBUG") else logging.WARNING,
 format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info(f"ML Common v{__version__} initialized - Enterprise ML Utilities")

# Optional: Setup monitoring if enabled
if os.getenv("ML_COMMON_ENABLE_MONITORING", "false").lower == "true":
 try:
 from .utils.monitoring import setup_monitoring
 setup_monitoring
 logger.info("Monitoring enabled for ML Common")
 except ImportError:
 logger.warning("Monitoring dependencies not found, continuing without monitoring")

# Performance optimization hints
try:
 import numba
 logger.info(f"Numba {numba.__version__} available - JIT compilation enabled")
except ImportError:
 logger.warning("Numba not available - falling back to pure Python (slower)")

try:
 import numpy as np
 logger.info(f"NumPy {np.__version__} - vectorized operations enabled")
except ImportError:
 logger.error("NumPy is required for ML Common")
 raise ImportError("NumPy is required but not installed")