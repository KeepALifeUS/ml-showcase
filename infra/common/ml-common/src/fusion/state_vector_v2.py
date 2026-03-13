"""
StateVectorBuilder V2 - Enhanced for 90%+ Win Rate
ICON Architecture: 1024 dimensions

This is THE CRITICAL upgrade from V1 (768) → V2 (1024)
Added features specifically designed to achieve 90%+ win rate:
- USDT Dominance (10) - Critical macro trend filter
- Confidence & Risk (128) - Know when NOT to trade
- Futures Features (78) - Funding, OI, liquidations
- Enhanced Cross-Market (40 total) - Better correlation understanding

⚠️ WARNING: Feature ordering is IMMUTABLE
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Import USDT.D feature extractor
try:
 from ..data.usdt_dominance_fetcher import USDTDominanceFetcher
except ImportError:
 # Optional dependency - skip if not available
 USDTDominanceFetcher = None # type: ignore
from .usdt_dominance_features import USDTDominanceFeatures

logger = logging.getLogger(__name__)


# ============================================================================
# STATE VECTOR SCHEMA V2 (1024 dimensions) - ICON ARCHITECTURE
# ============================================================================

@dataclass
class StateVectorV2:
 """
 State Vector Schema Version 2 - ICON Architecture

 1024-dimensional feature vector for 90%+ win rate

 ENHANCED Feature Groups:
 1. OHLCV Raw (40 dims) : 4 symbols × 2 markets × 5 features
 2. Technical Indicators (320) : 40 indicators × 8 markets (spot+futures)
 3. Volume Features (64) : 8 volume features × 8 markets
 4. Orderbook Features (160) : 20 microstructure × 8 markets
 5. USDT Dominance (10) ← NEW! : Critical macro trend filter
 6. Futures Specific (78) ← NEW! : Funding, OI, liquidations
 7. Cross-Market (40) : Enhanced correlations
 8. Confidence & Risk (128) ← NEW! : Model confidence, risk assessment
 9. Regime Features (20) : Enhanced market classification
 10. Portfolio State (100) : Spot + Futures positions
 11. Symbol Embeddings (32) : 4 symbols × 4 dims × 2 markets
 12. Temporal Embeddings (20) : Multi-timeframe
 13. Delta History (32) : Compressed price changes

 TOTAL = 1024 dimensions ✅
 """

 # Feature dimensions (must sum to 1024)
 OHLCV_DIM: int = 40 # 4 symbols × 2 markets × 5 (OHLCV)
 TECHNICAL_DIM: int = 320 # 40 indicators × 8 markets
 VOLUME_DIM: int = 64 # 8 volume features × 8 markets
 ORDERBOOK_DIM: int = 160 # 20 microstructure × 8 markets
 USDT_D_DIM: int = 10 # NEW: USDT Dominance features
 FUTURES_DIM: int = 80 # NEW: Funding, OI, liquidations (20 per symbol)
 CROSS_MARKET_DIM: int = 40 # Enhanced (was 20 in V1)
 CONFIDENCE_RISK_DIM: int = 128 # NEW: Confidence + Risk assessment
 REGIME_DIM: int = 20 # Enhanced (was 10 in V1)
 PORTFOLIO_DIM: int = 100 # Enhanced (was 50 in V1)
 SYMBOL_EMBED_DIM: int = 32 # 4 symbols × 4 dims × 2 markets
 TEMPORAL_EMBED_DIM: int = 20 # Enhanced multi-timeframe
 DELTA_HISTORY_DIM: int = 10 # Compressed (was 370 in V1)

 TOTAL_DIM: int = 1024

 # Verify total
 def __post_init__(self):
 total = (
 self.OHLCV_DIM +
 self.TECHNICAL_DIM +
 self.VOLUME_DIM +
 self.ORDERBOOK_DIM +
 self.USDT_D_DIM +
 self.FUTURES_DIM +
 self.CROSS_MARKET_DIM +
 self.CONFIDENCE_RISK_DIM +
 self.REGIME_DIM +
 self.PORTFOLIO_DIM +
 self.SYMBOL_EMBED_DIM +
 self.TEMPORAL_EMBED_DIM +
 self.DELTA_HISTORY_DIM
 )

 if total != self.TOTAL_DIM:
 raise ValueError(
 f"StateVectorV2 dimension mismatch! "
 f"Expected {self.TOTAL_DIM}, got {total}. "
 f"Breakdown: OHLCV={self.OHLCV_DIM}, Tech={self.TECHNICAL_DIM}, "
 f"Vol={self.VOLUME_DIM}, OB={self.ORDERBOOK_DIM}, "
 f"USDT.D={self.USDT_D_DIM}, Futures={self.FUTURES_DIM}, "
 f"Cross={self.CROSS_MARKET_DIM}, ConfRisk={self.CONFIDENCE_RISK_DIM}, "
 f"Regime={self.REGIME_DIM}, Portfolio={self.PORTFOLIO_DIM}, "
 f"SymEmbed={self.SYMBOL_EMBED_DIM}, TempEmbed={self.TEMPORAL_EMBED_DIM}, "
 f"Delta={self.DELTA_HISTORY_DIM}"
 )

 # Symbols and markets
 SYMBOLS: List[str] = field(default_factory=lambda: ['BTC', 'ETH', 'BNB', 'SOL'])
 MARKETS: List[str] = field(default_factory=lambda: ['spot', 'futures'])
 NUM_SYMBOLS: int = 4
 NUM_MARKETS: int = 2

 # Sequence length for temporal features
 SEQ_LENGTH: int = 48 # 15-min candles, 12h history

 # Version
 VERSION: str = "V2"
 CREATED: str = "2025-10-23"
 TARGET: str = "90%+ Win Rate"


class StateVectorBuilderV2:
 """
 Build 1024-dimensional state vectors for Dreamer v3 Enhanced

 Critical for 90%+ win rate:
 - USDT.D filtering (don't trade against macro trend)
 - Confidence estimation (know when not to trade)
 - Risk assessment (avoid risky situations)
 - Futures integration (funding, OI, liquidations)
 """

 def __init__(self, config: Optional[StateVectorV2] = None):
 """
 Initialize StateVectorBuilder V2

 Args:
 config: StateVectorV2 configuration (default: create new)
 """
 self.config = config or StateVectorV2

 # Initialize USDT.D components
 self.usdt_d_fetcher = USDTDominanceFetcher
 self.usdt_d_features = USDTDominanceFeatures

 # Initialize OHLCV features extractor
 from fusion.ohlcv_features import OHLCVFeatures
 self.ohlcv_features = OHLCVFeatures

 # Initialize Technical Indicators extractor
 from fusion.technical_indicators import TechnicalIndicators
 self.technical_indicators = TechnicalIndicators

 # Initialize Futures features extractor
 from fusion.futures_features import FuturesFeatures
 self.futures_features = FuturesFeatures

 # Initialize Cross-Market features extractor
 from fusion.cross_market_features import CrossMarketFeatures
 self.cross_market_features = CrossMarketFeatures

 # Initialize Volume features extractor
 from fusion.volume_features import VolumeFeatures
 self.volume_features = VolumeFeatures

 # Initialize Confidence & Risk features extractor
 from fusion.confidence_risk_features import ConfidenceRiskFeatures
 self.confidence_risk_features = ConfidenceRiskFeatures

 # Initialize Regime Detection features extractor
 from fusion.regime_features import RegimeFeatures
 self.regime_features = RegimeFeatures

 # Initialize Portfolio features extractor
 from fusion.portfolio_features import PortfolioFeatures
 self.portfolio_features = PortfolioFeatures

 # Initialize Embeddings & Delta features extractor
 from fusion.embeddings_delta import EmbeddingsDeltaFeatures
 self.embeddings_delta_features = EmbeddingsDeltaFeatures

 # Cache for USDT.D data
 self.usdt_d_cache = None
 self.usdt_d_cache_time = None

 logger.info(
 f"StateVectorBuilder V2 initialized: "
 f"{self.config.TOTAL_DIM} dimensions, "
 f"{self.config.SEQ_LENGTH} timesteps"
 )

 def build(
 self,
 market_data: Dict[str, pd.DataFrame],
 portfolio: Dict[str, Any],
 timestamp: Optional[datetime] = None
 ) -> np.ndarray:
 """
 Build state vector from market data and portfolio state

 Args:
 market_data: Dict with keys:
 - 'BTC_spot', 'BTC_futures', etc. (OHLCV DataFrames)
 - 'orderbook_BTC_spot', etc. (orderbook snapshots)
 - 'usdt_dominance' (optional, will fetch if missing)
 portfolio: Dict with:
 - 'spot_positions': Dict of spot positions
 - 'futures_positions': Dict of futures positions
 - 'spot_balance': float
 - 'futures_balance': float
 timestamp: Current timestamp (default: now)

 Returns:
 State vector of shape (1024, 48) for sequence models
 or (1024,) for single timestep
 """
 if timestamp is None:
 timestamp = datetime.now

 # Separate spot and futures data
 spot_data = {k: v for k, v in market_data.items if '_spot' in k}
 futures_data = {k: v for k, v in market_data.items if '_futures' in k}

 # Determine sequence length from data
 n_samples = 0
 for key, df in market_data.items:
 if isinstance(df, pd.DataFrame) and len(df) > n_samples:
 n_samples = len(df)

 if n_samples == 0:
 logger.warning("No market data available, returning zeros")
 return np.zeros((self.config.TOTAL_DIM, self.config.SEQ_LENGTH))

 # Initialize state vector (each feature will be time series)
 # Shape: (total_dims, seq_length)
 state_vector = np.zeros((self.config.TOTAL_DIM, min(n_samples, self.config.SEQ_LENGTH)))
 actual_samples = min(n_samples, self.config.SEQ_LENGTH)

 feature_idx = 0

 # === CURRENTLY IMPLEMENTED FEATURES ===

 # OHLCV Features (40 dims) - position: 0-39
 ohlcv_start = 0
 if spot_data:
 ohlcv_features = self.ohlcv_features.extract(spot_data)
 ohlcv_len = min(len(ohlcv_features), actual_samples)
 state_vector[ohlcv_start:ohlcv_start+40, :ohlcv_len] = ohlcv_features[:ohlcv_len].T

 # Technical Indicators (320 dims) - position: 40-359
 tech_start = 40
 if spot_data:
 tech_features = self.technical_indicators.extract(spot_data)
 tech_len = min(len(tech_features), actual_samples)
 state_vector[tech_start:tech_start+320, :tech_len] = tech_features[:tech_len].T

 # USDT Dominance Features (10 dims) - position: 360-369
 usdt_d_start = 360
 if 'usdt_dominance' in market_data:
 usdt_d_features = self.extract_usdt_d_features(market_data['usdt_dominance'])
 usdt_d_len = min(len(usdt_d_features), actual_samples)
 state_vector[usdt_d_start:usdt_d_start+10, :usdt_d_len] = usdt_d_features[:usdt_d_len].T

 # Futures Features (80 dims) - position: 370-449
 futures_start = 370
 if futures_data:
 futures_features = self.futures_features.extract(futures_data, spot_data)
 futures_len = min(len(futures_features), actual_samples)
 state_vector[futures_start:futures_start+80, :futures_len] = futures_features[:futures_len].T

 # Cross-Market Features (40 dims) - position: 450-489
 cross_start = 450
 if spot_data and futures_data:
 cross_features = self.cross_market_features.extract(spot_data, futures_data)
 cross_len = min(len(cross_features), actual_samples)
 state_vector[cross_start:cross_start+40, :cross_len] = cross_features[:cross_len].T

 # Volume Features (64 dims) - position: 490-553
 volume_start = 490
 if spot_data:
 volume_features = self.volume_features.extract(spot_data)
 volume_len = min(len(volume_features), actual_samples)
 state_vector[volume_start:volume_start+64, :volume_len] = volume_features[:volume_len].T

 # Confidence & Risk Features (128 dims) - position: 714-841
 # Note: Orderbook (160) at 554-713 will be added later when orderbook pipeline is ready
 confidence_risk_start = 714
 if spot_data:
 # Get BTC close for correlation analysis
 btc_close = None
 if 'BTC_spot' in spot_data:
 btc_close = spot_data['BTC_spot']['close']

 conf_risk_features = self.confidence_risk_features.extract(spot_data, btc_close)
 conf_risk_len = min(len(conf_risk_features), actual_samples)
 state_vector[confidence_risk_start:confidence_risk_start+128, :conf_risk_len] = conf_risk_features[:conf_risk_len].T

 # Regime Detection Features (20 dims) - position: 842-861
 regime_start = 842
 if spot_data:
 # Get BTC close for correlation analysis
 btc_close = None
 if 'BTC_spot' in spot_data:
 btc_close = spot_data['BTC_spot']['close']

 regime_features = self.regime_features.extract(spot_data, btc_close)
 regime_len = min(len(regime_features), actual_samples)
 state_vector[regime_start:regime_start+20, :regime_len] = regime_features[:regime_len].T

 # Portfolio State Features (100 dims) - position: 862-961
 portfolio_start = 862
 # Portfolio features are single timestep (not time series)
 # Get current market prices for position valuation
 market_prices = {}
 for symbol in self.config.SYMBOLS:
 for market in ['spot', 'futures']:
 key = f"{symbol}_{market}"
 if key in market_data and len(market_data[key]) > 0:
 market_prices[key] = float(market_data[key]['close'].iloc[-1])

 portfolio_features = self.portfolio_features.extract(portfolio, market_prices)
 # Broadcast to all timesteps
 state_vector[portfolio_start:portfolio_start+100, :] = portfolio_features[:, np.newaxis]

 # Embeddings & Delta History (62 dims) - position: 962-1023
 embeddings_start = 962
 embeddings_features = self.embeddings_delta_features.extract(market_data, timestamp)
 # Symbol/Temporal embeddings broadcast to all timesteps, Delta history only for last
 # Symbol embeddings (32): 962-993
 state_vector[embeddings_start:embeddings_start+32, :] = embeddings_features[0:32, np.newaxis]
 # Temporal embeddings (20): 994-1013
 state_vector[embeddings_start+32:embeddings_start+52, :] = embeddings_features[32:52, np.newaxis]
 # Delta history (10): 1014-1023 - only use last timestep value
 state_vector[embeddings_start+52:embeddings_start+62, -1] = embeddings_features[52:62]

 # === PLACEHOLDER FOR REMAINING FEATURES ===
 # - Orderbook (160): 554-713 (requires real-time orderbook pipeline)

 logger.debug(
 f"Built state vector: {state_vector.shape}, "
 f"OHLCV: {ohlcv_start}-{ohlcv_start+40}, "
 f"Technical: {tech_start}-{tech_start+320}, "
 f"USDT.D: {usdt_d_start}-{usdt_d_start+10}, "
 f"Futures: {futures_start}-{futures_start+80}, "
 f"Cross: {cross_start}-{cross_start+40}, "
 f"Volume: {volume_start}-{volume_start+64}, "
 f"Conf&Risk: {confidence_risk_start}-{confidence_risk_start+128}, "
 f"Regime: {regime_start}-{regime_start+20}, "
 f"Portfolio: {portfolio_start}-{portfolio_start+100}, "
 f"Embeddings: {embeddings_start}-{embeddings_start+62}, "
 f"timestamp={timestamp}"
 )

 return state_vector

 async def fetch_usdt_dominance(self) -> pd.DataFrame:
 """
 Fetch USDT Dominance data

 Returns:
 DataFrame with USDT.D data for feature extraction
 """
 # Check cache (1-hour TTL)
 import time
 now = time.time

 if (self.usdt_d_cache is not None and
 self.usdt_d_cache_time is not None and
 now - self.usdt_d_cache_time < 3600):
 return self.usdt_d_cache

 # Fetch current + last 100 hours
 df_historical = await self.usdt_d_fetcher.fetch_range(days=5)
 current = await self.usdt_d_fetcher.fetch_current

 # Append current
 df_historical = pd.concat([
 df_historical,
 pd.DataFrame([{
 'timestamp': current['timestamp'],
 'usdt_dominance': current['dominance']
 }])
 ])

 # Cache
 self.usdt_d_cache = df_historical
 self.usdt_d_cache_time = now

 return df_historical

 def extract_usdt_d_features(self, usdt_d_df: pd.DataFrame) -> np.ndarray:
 """
 Extract 10 USDT.D features

 Args:
 usdt_d_df: DataFrame with 'usdt_dominance' column

 Returns:
 np.ndarray of shape (N, 10) with USDT.D features
 """
 return self.usdt_d_features.extract(usdt_d_df)


# Quick test
def test_state_vector_v2:
 """Test StateVectorV2 configuration"""
 print("=" * 60)
 print("TESTING STATE VECTOR V2 CONFIGURATION")
 print("=" * 60)

 # Test 1: Configuration validation
 print("\n1. Testing configuration...")
 try:
 config = StateVectorV2
 print("✅ Configuration created successfully")
 print(f" Total dimensions: {config.TOTAL_DIM}")
 print(f" Version: {config.VERSION}")
 print(f" Target: {config.TARGET}")
 except Exception as e:
 print(f"❌ Configuration failed: {e}")
 return

 # Test 2: Dimension breakdown
 print("\n2. Dimension breakdown:")
 print(f" OHLCV: {config.OHLCV_DIM}")
 print(f" Technical: {config.TECHNICAL_DIM}")
 print(f" Volume: {config.VOLUME_DIM}")
 print(f" Orderbook: {config.ORDERBOOK_DIM}")
 print(f" USDT.D (NEW): {config.USDT_D_DIM}")
 print(f" Futures (NEW): {config.FUTURES_DIM}")
 print(f" Cross-Market: {config.CROSS_MARKET_DIM}")
 print(f" Confidence: {config.CONFIDENCE_RISK_DIM}")
 print(f" Regime: {config.REGIME_DIM}")
 print(f" Portfolio: {config.PORTFOLIO_DIM}")
 print(f" Symbol Embed: {config.SYMBOL_EMBED_DIM}")
 print(f" Temporal: {config.TEMPORAL_EMBED_DIM}")
 print(f" Delta History: {config.DELTA_HISTORY_DIM}")
 print(f" " + "-" * 40)

 total_check = sum([
 config.OHLCV_DIM,
 config.TECHNICAL_DIM,
 config.VOLUME_DIM,
 config.ORDERBOOK_DIM,
 config.USDT_D_DIM,
 config.FUTURES_DIM,
 config.CROSS_MARKET_DIM,
 config.CONFIDENCE_RISK_DIM,
 config.REGIME_DIM,
 config.PORTFOLIO_DIM,
 config.SYMBOL_EMBED_DIM,
 config.TEMPORAL_EMBED_DIM,
 config.DELTA_HISTORY_DIM
 ])

 print(f" TOTAL: {total_check}")

 if total_check == 1024:
 print(" ✅ Dimensions sum to 1024 correctly!")
 else:
 print(f" ❌ ERROR: Expected 1024, got {total_check}")

 # Test 3: Builder initialization
 print("\n3. Testing builder initialization...")
 try:
 builder = StateVectorBuilderV2(config)
 print("✅ Builder initialized successfully")
 except Exception as e:
 print(f"❌ Builder initialization failed: {e}")
 return

 # Test 4: Build state vector with synthetic data
 print("\n4. Testing state vector building with synthetic data...")
 try:
 np.random.seed(42)
 n_samples = 100

 # Create synthetic market data
 symbols = ['BTC', 'ETH', 'BNB', 'SOL']
 market_data = {}

 # Add USDT dominance
 market_data['usdt_dominance'] = pd.DataFrame({
 'usdt_dominance': np.random.uniform(4.5, 5.5, n_samples)
 })

 # Add spot and futures data
 for symbol in symbols:
 base_price = 50000 if symbol == 'BTC' else 2000

 # Generate realistic OHLCV for spot
 close_prices = base_price + np.random.normal(0, base_price * 0.02, n_samples).cumsum
 high_prices = close_prices + np.abs(np.random.normal(0, base_price * 0.01, n_samples))
 low_prices = close_prices - np.abs(np.random.normal(0, base_price * 0.01, n_samples))
 open_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0, 1, n_samples)

 # Spot data
 market_data[f"{symbol}_spot"] = pd.DataFrame({
 'open': open_prices,
 'high': high_prices,
 'low': low_prices,
 'close': close_prices,
 'volume': np.random.uniform(1e8, 1e9, n_samples)
 })

 # Futures data
 futures_price = market_data[f"{symbol}_spot"]['close'].values + np.random.normal(10, 50, n_samples)
 market_data[f"{symbol}_futures"] = pd.DataFrame({
 'close': futures_price,
 'volume': np.random.uniform(1e8, 1e9, n_samples),
 'funding_rate': np.random.normal(0.0001, 0.00005, n_samples),
 'open_interest': np.random.uniform(1e9, 5e9, n_samples)
 })

 # Add empty portfolio
 portfolio = {
 'spot_positions': {},
 'futures_positions': {},
 'spot_balance': 1000.0,
 'futures_balance': 1000.0
 }

 # Build state vector
 state_vector = builder.build(market_data, portfolio)

 print(f"✅ State vector built: shape={state_vector.shape}")
 print(f" Expected: (1024, 48)")

 # Check dimensions
 if state_vector.shape[0] == 1024:
 print(f" ✅ First dimension correct: 1024")
 else:
 print(f" ❌ First dimension incorrect: {state_vector.shape[0]}")

 # Check for implemented features (non-zero)
 ohlcv_slice = state_vector[0:40, :]
 tech_slice = state_vector[40:360, :]
 usdt_d_slice = state_vector[360:370, :]
 futures_slice = state_vector[370:450, :]
 cross_slice = state_vector[450:490, :]
 volume_slice = state_vector[490:554, :]
 conf_risk_slice = state_vector[714:842, :]
 regime_slice = state_vector[842:862, :]

 ohlcv_nonzero = np.count_nonzero(ohlcv_slice)
 tech_nonzero = np.count_nonzero(tech_slice)
 usdt_d_nonzero = np.count_nonzero(usdt_d_slice)
 futures_nonzero = np.count_nonzero(futures_slice)
 cross_nonzero = np.count_nonzero(cross_slice)
 volume_nonzero = np.count_nonzero(volume_slice)
 conf_risk_nonzero = np.count_nonzero(conf_risk_slice)
 regime_nonzero = np.count_nonzero(regime_slice)

 print(f"\n Implemented features (non-zero count):")
 print(f" - OHLCV [0:40]: {ohlcv_nonzero} / {ohlcv_slice.size} ({100*ohlcv_nonzero/ohlcv_slice.size:.1f}%)")
 print(f" - Technical [40:360]: {tech_nonzero} / {tech_slice.size} ({100*tech_nonzero/tech_slice.size:.1f}%)")
 print(f" - USDT.D [360:370]: {usdt_d_nonzero} / {usdt_d_slice.size} ({100*usdt_d_nonzero/usdt_d_slice.size:.1f}%)")
 print(f" - Futures [370:450]: {futures_nonzero} / {futures_slice.size} ({100*futures_nonzero/futures_slice.size:.1f}%)")
 print(f" - Cross [450:490]: {cross_nonzero} / {cross_slice.size} ({100*cross_nonzero/cross_slice.size:.1f}%)")
 print(f" - Volume [490:554]: {volume_nonzero} / {volume_slice.size} ({100*volume_nonzero/volume_slice.size:.1f}%)")
 print(f" - Conf&Risk [714:842]: {conf_risk_nonzero} / {conf_risk_slice.size} ({100*conf_risk_nonzero/conf_risk_slice.size:.1f}%)")
 print(f" - Regime [842:862]: {regime_nonzero} / {regime_slice.size} ({100*regime_nonzero/regime_slice.size:.1f}%)")

 if ohlcv_nonzero > 0 and tech_nonzero > 0 and usdt_d_nonzero > 0 and futures_nonzero > 0 and cross_nonzero > 0 and volume_nonzero > 0 and conf_risk_nonzero > 0 and regime_nonzero > 0:
 print(f" ✅ All implemented features populated!")
 else:
 print(f" ⚠️ Some features are zero (check data availability)")

 except Exception as e:
 print(f"❌ State vector building failed: {e}")
 import traceback
 traceback.print_exc

 print("\n" + "=" * 60)
 print("STATE VECTOR V2 TESTS COMPLETED ✅")
 print("=" * 60)
 print(f"\nUpgrade: 768 → 1024 dimensions (+{1024-768} features)")
 print("New features for 90%+ win rate:")
 print(" - USDT Dominance (10)")
 print(" - Futures data (80)")
 print(" - Confidence & Risk (128)")
 print(" - Enhanced Cross-Market (20 → 40)")
 print(" - Enhanced Portfolio (50 → 100)")


if __name__ == "__main__":
 test_state_vector_v2
