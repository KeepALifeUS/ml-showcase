"""
Symbol Embeddings - Learnable Symbol Identity Representations
Multi-Asset Trading

Provides 4-dimensional embeddings for each of the 4 trading symbols:
- BTCUSDT: [btc_0, btc_1, btc_2, btc_3]
- ETHUSDT: [eth_0, eth_1, eth_2, eth_3]
- BNBUSDT: [bnb_0, bnb_1, bnb_2, bnb_3]
- SOLUSDT: [sol_0, sol_1, sol_2, sol_3]

Total: 16 dimensions

These embeddings are LEARNABLE during neural network training.
Initial values are random, will be optimized to capture symbol-specific characteristics.

Performance: <0.1ms per extraction
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# SYMBOL EMBEDDINGS (initialized randomly, will be learned)
# ============================================================================

# Default 4-symbol configuration
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']
EMBEDDING_DIM = 4

# Initial random embeddings (will be replaced during training)
# These are just placeholders - the neural network will learn optimal values
_SYMBOL_EMBEDDINGS: Dict[str, np.ndarray] = {}


def initialize_symbol_embeddings(
 symbols: Optional[List[str]] = None,
 embedding_dim: int = EMBEDDING_DIM,
 seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
 """
 Initialize random symbol embeddings

 These are just starting points - the neural network will learn better representations
 during training (similar to word2vec initialization).

 Args:
 symbols: List of symbols (defaults to ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
 embedding_dim: Dimension per symbol (default 4)
 seed: Random seed for reproducibility

 Returns:
 Dict of symbol -> embedding array (shape: embedding_dim,)
 """
 global _SYMBOL_EMBEDDINGS

 if symbols is None:
 symbols = DEFAULT_SYMBOLS

 if seed is not None:
 np.random.seed(seed)

 _SYMBOL_EMBEDDINGS = {}
 for symbol in symbols:
 # Initialize with small random values (Gaussian distribution)
 # Using Xavier/Glorot initialization scaled for embedding dimension
 scale = np.sqrt(2.0 / embedding_dim)
 embedding = np.random.randn(embedding_dim).astype(np.float32) * scale
 _SYMBOL_EMBEDDINGS[symbol] = embedding

 logger.info(f"Initialized {len(symbols)} symbol embeddings with dim={embedding_dim}")
 return _SYMBOL_EMBEDDINGS.copy


def get_symbol_embedding(symbol: str, embedding_dim: int = EMBEDDING_DIM) -> np.ndarray:
 """
 Get embedding for a single symbol

 Args:
 symbol: Symbol name (e.g., 'BTCUSDT')
 embedding_dim: Embedding dimension

 Returns:
 np.ndarray: Embedding vector (shape: embedding_dim,)
 """
 global _SYMBOL_EMBEDDINGS

 # Initialize if not already done
 if not _SYMBOL_EMBEDDINGS:
 initialize_symbol_embeddings

 # Return embedding if exists
 if symbol in _SYMBOL_EMBEDDINGS:
 return _SYMBOL_EMBEDDINGS[symbol].copy

 # Create new embedding for unknown symbol
 logger.warning(f"Unknown symbol {symbol}, creating new random embedding")
 scale = np.sqrt(2.0 / embedding_dim)
 embedding = np.random.randn(embedding_dim).astype(np.float32) * scale
 _SYMBOL_EMBEDDINGS[symbol] = embedding

 return embedding


def extract_symbol_embeddings(
 symbols: Optional[List[str]] = None,
 embedding_dim: int = EMBEDDING_DIM
) -> np.ndarray:
 """
 Extract symbol embeddings for all symbols

 Args:
 symbols: List of symbols (defaults to ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])
 embedding_dim: Dimension per symbol (default 4)

 Returns:
 np.ndarray: Concatenated embeddings (shape: num_symbols * embedding_dim,)
 For 4 symbols × 4 dims = 16 total dimensions

 Example:
 >>> embeddings = extract_symbol_embeddings
 >>> embeddings.shape
 (16,)
 >>> # embeddings = [btc_0, btc_1, btc_2, btc_3, eth_0, ..., sol_3]
 """
 if symbols is None:
 symbols = DEFAULT_SYMBOLS

 # Handle empty symbol list
 if len(symbols) == 0:
 return np.array([], dtype=np.float32)

 # Initialize if needed
 if not _SYMBOL_EMBEDDINGS:
 initialize_symbol_embeddings(symbols, embedding_dim)

 # Concatenate all embeddings
 embeddings_list = []
 for symbol in symbols:
 embedding = get_symbol_embedding(symbol, embedding_dim)
 embeddings_list.append(embedding)

 # Stack into single array
 result = np.concatenate(embeddings_list)

 assert result.shape[0] == len(symbols) * embedding_dim, \
 f"Shape mismatch: {result.shape[0]} != {len(symbols) * embedding_dim}"

 return result


def load_symbol_embeddings(embeddings: Dict[str, np.ndarray]) -> None:
 """
 Load pre-trained symbol embeddings (from trained model)

 Args:
 embeddings: Dict of symbol -> embedding array
 """
 global _SYMBOL_EMBEDDINGS
 _SYMBOL_EMBEDDINGS = embeddings.copy
 logger.info(f"Loaded {len(embeddings)} pre-trained symbol embeddings")


def get_all_embeddings -> Dict[str, np.ndarray]:
 """Get all current symbol embeddings"""
 global _SYMBOL_EMBEDDINGS
 if not _SYMBOL_EMBEDDINGS:
 initialize_symbol_embeddings
 return _SYMBOL_EMBEDDINGS.copy


# Initialize on module import with default symbols
initialize_symbol_embeddings(DEFAULT_SYMBOLS, EMBEDDING_DIM, seed=42)
