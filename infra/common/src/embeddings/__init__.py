"""
Embeddings Module - Symbol and Temporal Feature Encoding
Learned Representations

This module provides learnable embeddings for:
- Symbol embeddings (16 dims = 4 symbols × 4-dim): BTC, ETH, BNB, SOL
- Temporal embeddings (10 dims): Hour, day, week, month, quarter, year, ...

Key Features:
- Fixed embedding dimensions for v1 schema
- Learnable during neural network training
- Initialized with random values (will be trained)
- <0.1ms extraction time

Architecture:
- symbol.py: Symbol identity embeddings (16 dims)
- temporal.py: Time-based cyclic encodings (10 dims)

Usage:
 from ml_common.embeddings import extract_symbol_embeddings, extract_temporal_embeddings

 # Symbol embeddings (4 × 4-dim)
 symbol_features = extract_symbol_embeddings(symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
 # Returns: [btc_0, btc_1, btc_2, btc_3, eth_0, eth_1, eth_2, eth_3, ...]

 # Temporal embeddings (10 dims)
 from datetime import datetime
 temporal_features = extract_temporal_embeddings(timestamp=datetime.now)
 # Returns: [hour_sin, hour_cos, day_sin, day_cos, week_sin, week_cos, month, quarter, year_frac, weekend]
"""

from .symbol import (
 extract_symbol_embeddings,
 get_symbol_embedding,
 initialize_symbol_embeddings,
)

from .temporal import (
 extract_temporal_embeddings,
 encode_hour_of_day,
 encode_day_of_week,
 encode_week_of_year,
 encode_month_of_year,
)

__all__ = [
 # Symbol (3 functions)
 "extract_symbol_embeddings",
 "get_symbol_embedding",
 "initialize_symbol_embeddings",

 # Temporal (5 functions)
 "extract_temporal_embeddings",
 "encode_hour_of_day",
 "encode_day_of_week",
 "encode_week_of_year",
 "encode_month_of_year",
]

__version__ = "1.0.0"
__author__ = "ML-Framework Autonomous AI Team"
__note__ = "Learnable embeddings for symbols and time"
