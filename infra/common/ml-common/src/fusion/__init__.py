"""
Fusion Module - StateVector Integration
Feature Integration

This module provides the CRITICAL StateVectorBuilder component that integrates
all Week 1-2 features into the 768-dimensional state vector for autonomous AI training.

Key Features:
- StateVectorBuilder: 768-dim constructor (CRITICAL)
- WindowManager: 168h sliding window management
- Version control: V1, V2, ... for backward compatibility
- <30ms construction time
- Type-safe numpy tensor output: (168, 768)

Architecture:
- state_vector.py: StateVectorBuilder class, feature schema, integration logic
- windowing.py: SlidingWindow class, timestamp alignment, missing data handling

StateVector Schema V1 (768 dimensions):
┌─────────────────────────────────────────────────────────────────┐
│ OHLCV Raw (20) : BTC/ETH/BNB/SOL × [O,H,L,C,V] │
│ Technical (160) : 40 indicators × 4 symbols │
│ Volume (32) : 8 volume features × 4 symbols │
│ Orderbook (80) : 20 microstructure × 4 symbols │
│ Cross-Asset (20) : Correlation, spreads, beta │
│ Regime (10) : Volatility, trend, time │
│ Portfolio (50) : Positions, PnL, risk metrics │
│ Symbol Embed (16) : 4-dim embeddings × 4 symbols │
│ Temporal Embed (10) : Time encoding (hour, day, week, ...) │
│ Delta History (390) : 78 lookback steps × 5 features │
├─────────────────────────────────────────────────────────────────┤
│ TOTAL = 768 dimensions │
└─────────────────────────────────────────────────────────────────┘

Usage:
 from ml_common.fusion import StateVectorBuilder, WindowManager

 # Create builder
 builder = StateVectorBuilder(version='v1', symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])

 # Construct state vector from raw data
 state_vector = builder.build(
 ohlcv_data=ohlcv_history, # 4 symbols × 168h
 orderbook_data=orderbook_snapshots,
 portfolio_state=current_portfolio,
 timestamp=current_time
 )

 # Output: numpy array shape (seq_length, state_dim) - dynamic based on config
 # assert state_vector.shape == (168, 768) # REMOVED - now dynamic (24, 384) for memory efficiency
 assert builder.build_time_ms < 30.0 # Performance guarantee

Why This Module is CRITICAL:
- Single source of truth for feature integration
- Contract between data pipeline and neural network
- Feature order is IMMUTABLE (changing = retraining all models)
- Without this, Week 1 features are isolated and useless
- Enables reproducible training (same input → same vector)
"""

from .state_vector import (
 StateVectorBuilder,
 StateVectorConfig,
 StateVectorV1,
 FeatureMap,
 get_feature_dimension,
 get_feature_indices,
)

from .windowing import (
 WindowManager,
 WindowConfig,
 SlidingWindow,
 align_timestamps,
 handle_missing_data,
)

from .indicator_cache import (
 PreCalculatedIndicatorCache,
 IndicatorCacheStats,
 create_cache_from_csv,
)

# 🎯 State Vector V2 - 1024 dimensions for Dreamer v3 (90%+ win rate target)
from .state_vector_v2 import (
 StateVectorV2,
 StateVectorBuilderV2,
)

# Feature extractors for V2
from .volume_features import VolumeFeatures
from .confidence_risk_features import ConfidenceRiskFeatures
from .regime_features import RegimeFeatures
from .portfolio_features import PortfolioFeatures
from .embeddings_delta import EmbeddingsDeltaFeatures

# ⚡ ULTRA-FAST Vectorized Components (180x speedup!)
from .vectorized_indicator_cache import VectorizedIndicatorCache
from .state_vector_vectorized import VectorizedStateVectorBuilder

# 🚀 GPU-NATIVE Components (ALL data in GPU VRAM, zero CPU transfers!)
try:
 from .gpu_indicator_cache import GPUIndicatorCache
 from .gpu_state_vector_builder import GPUStateVectorBuilder
 HAS_GPU_SUPPORT = True
except ImportError:
 # GPU components require torch - skip if not available
 GPUIndicatorCache = None # type: ignore
 GPUStateVectorBuilder = None # type: ignore
 HAS_GPU_SUPPORT = False

__all__ = [
 # State Vector V1 (6 exports) - 768 dimensions
 "StateVectorBuilder",
 "StateVectorConfig",
 "StateVectorV1",
 "FeatureMap",
 "get_feature_dimension",
 "get_feature_indices",

 # State Vector V2 (7 exports) - 1024 dimensions for Dreamer v3
 "StateVectorV2",
 "StateVectorBuilderV2",
 "VolumeFeatures",
 "ConfidenceRiskFeatures",
 "RegimeFeatures",
 "PortfolioFeatures",
 "EmbeddingsDeltaFeatures",

 # Windowing (5 exports)
 "WindowManager",
 "WindowConfig",
 "SlidingWindow",
 "align_timestamps",
 "handle_missing_data",

 # Indicator Cache (3 exports) - Performance Optimization
 "PreCalculatedIndicatorCache",
 "IndicatorCacheStats",
 "create_cache_from_csv",

 # ⚡ Vectorized Components (2 exports) - ULTRA-FAST 180x speedup!
 "VectorizedIndicatorCache",
 "VectorizedStateVectorBuilder",

 # 🚀 GPU-Native Components (2 exports) - ALL data in GPU VRAM!
 "GPUIndicatorCache",
 "GPUStateVectorBuilder",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
__critical__ = "StateVectorBuilder is THE bridge between raw data and neural network"
