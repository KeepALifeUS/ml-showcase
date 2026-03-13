"""
ML Fear & Greed Index System
=============================

Comprehensive Fear & Greed Index system for crypto markets.

Features:
- Multi-component index calculation (volatility, momentum, volume, sentiment, etc.)
- Real-time and historical data collection
- ML-based prediction and forecasting
- Enterprise-grade API with rate limiting
- Visualization and alerting system
- Production monitoring and logging

License: MIT
"""

__version__ = "1.0.0"

from .models.fear_greed_model import FearGreedIndex
from .calculators.weighted_calculator import WeightedCalculator
from .utils.config import FearGreedConfig

__all__ = [
    "FearGreedIndex",
    "WeightedCalculator",
    "FearGreedConfig",
]
