"""
Observation Spaces for Crypto Trading Environments
enterprise patterns for sophisticated feature engineering

Features:
- Multi-modal observations (price, volume, sentiment, microstructure)
- Configurable feature engineering pipeline
- Real-time normalization and scaling
- Memory-efficient implementation
- Type-safe observation construction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from gymnasium import spaces
from enum import Enum
import warnings
from collections import deque

from ..utils.indicators import TechnicalIndicators
from ..utils.normalization import RunningMeanStd, MinMaxScaler


class ObservationMode(Enum):
    """Observation composition modes"""
    BASIC = "basic"              # Price + volume only
    TECHNICAL = "technical"      # + Technical indicators
    SENTIMENT = "sentiment"      # + Sentiment data
    MICROSTRUCTURE = "micro"     # + Order book data
    FULL = "full"               # All features


@dataclass
class ObservationConfig:
    """Configuration for observation space"""
    
    # Basic features
    include_price: bool = True
    include_volume: bool = True
    include_price_changes: bool = True
    price_history_length: int = 50
    
    # Technical indicators
    include_technical_indicators: bool = True
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "ema_12", "rsi_14", "bb_upper", "bb_lower", "macd", "macd_signal"
    ])
    
    # Sentiment features
    include_sentiment: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: [
        "overall", "twitter", "reddit", "news", "fear_greed"
    ])
    sentiment_history_length: int = 10
    
    # Market microstructure
    include_order_book: bool = False
    order_book_depth: int = 5
    include_trade_flow: bool = False
    trade_flow_window: int = 100
    
    # Portfolio state
    include_portfolio_state: bool = True
    include_position_history: bool = True
    position_history_length: int = 20
    
    # Market regime
    include_market_regime: bool = True
    regime_encoding: str = "onehot"  # onehot, ordinal, embedding
    
    # Time features
    include_time_features: bool = True
    time_features: List[str] = field(default_factory=lambda: [
        "hour", "day_of_week", "day_of_month", "quarter"
    ])
    
    # Normalization
    normalize_prices: bool = True
    normalize_volumes: bool = True
    normalize_indicators: bool = True
    normalization_method: str = "running_mean_std"  # running_mean_std, minmax, robust
    
    # Performance optimization
    use_float32: bool = True
    cache_observations: bool = True
    parallel_processing: bool = True


class CryptoObservationSpace:
    """
    Sophisticated observation space builder for crypto trading
    
    Creates comprehensive observations with multiple modalities:
    - Price/Volume data with history
    - Technical indicators
    - Sentiment analysis data
    - Market microstructure
    - Portfolio state
    - Market regime information
    """
    
    def __init__(self, config: ObservationConfig, assets: List[str]):
        self.config = config
        self.assets = assets
        self.num_assets = len(assets)
        
        # Initialize components
        self._setup_normalizers()
        self._setup_technical_indicators()
        self._calculate_observation_dimensions()
        
        # Cache for performance
        self.observation_cache = {} if config.cache_observations else None
        self.last_cache_step = -1
    
    def _setup_normalizers(self) -> None:
        """Setup normalization components"""
        
        self.normalizers = {}
        
        if self.config.normalization_method == "running_mean_std":
            for asset in self.assets:
                self.normalizers[f"{asset}_price"] = RunningMeanStd()
                self.normalizers[f"{asset}_volume"] = RunningMeanStd()
                
                # Normalizers for technical indicators
                if self.config.include_technical_indicators:
                    for indicator in self.config.technical_indicators:
                        self.normalizers[f"{asset}_{indicator}"] = RunningMeanStd()
        
        elif self.config.normalization_method == "minmax":
            self.normalizers["global"] = MinMaxScaler()
    
    def _setup_technical_indicators(self) -> None:
        """Setup technical indicators calculator"""
        
        if self.config.include_technical_indicators:
            self.technical_calculator = TechnicalIndicators(
                indicators=self.config.technical_indicators
            )
    
    def _calculate_observation_dimensions(self) -> None:
        """Calculate total observation dimensions"""
        
        self.observation_dim = 0
        self.feature_map = {}
        current_idx = 0
        
        # Price features
        if self.config.include_price:
            price_dim = self.num_assets * self.config.price_history_length
            self.feature_map["prices"] = (current_idx, current_idx + price_dim)
            self.observation_dim += price_dim
            current_idx += price_dim
        
        # Volume features  
        if self.config.include_volume:
            volume_dim = self.num_assets * self.config.price_history_length
            self.feature_map["volumes"] = (current_idx, current_idx + volume_dim)
            self.observation_dim += volume_dim
            current_idx += volume_dim
        
        # Price changes
        if self.config.include_price_changes:
            changes_dim = self.num_assets
            self.feature_map["price_changes"] = (current_idx, current_idx + changes_dim)
            self.observation_dim += changes_dim
            current_idx += changes_dim
        
        # Technical indicators
        if self.config.include_technical_indicators:
            indicators_dim = self.num_assets * len(self.config.technical_indicators)
            self.feature_map["technical_indicators"] = (current_idx, current_idx + indicators_dim)
            self.observation_dim += indicators_dim
            current_idx += indicators_dim
        
        # Sentiment features
        if self.config.include_sentiment:
            sentiment_dim = self.num_assets * len(self.config.sentiment_sources)
            self.feature_map["sentiment"] = (current_idx, current_idx + sentiment_dim)
            self.observation_dim += sentiment_dim
            current_idx += sentiment_dim
        
        # Order book features
        if self.config.include_order_book:
            # Bid/ask prices and volumes for each depth level
            orderbook_dim = self.num_assets * self.config.order_book_depth * 4  # bid_price, bid_vol, ask_price, ask_vol
            self.feature_map["order_book"] = (current_idx, current_idx + orderbook_dim)
            self.observation_dim += orderbook_dim
            current_idx += orderbook_dim
        
        # Portfolio state
        if self.config.include_portfolio_state:
            portfolio_dim = self.num_assets + 4  # positions + balance + portfolio_value + drawdown + step_ratio
            self.feature_map["portfolio"] = (current_idx, current_idx + portfolio_dim)
            self.observation_dim += portfolio_dim
            current_idx += portfolio_dim
        
        # Position history
        if self.config.include_position_history:
            pos_history_dim = self.num_assets * self.config.position_history_length
            self.feature_map["position_history"] = (current_idx, current_idx + pos_history_dim)
            self.observation_dim += pos_history_dim
            current_idx += pos_history_dim
        
        # Market regime
        if self.config.include_market_regime:
            if self.config.regime_encoding == "onehot":
                regime_dim = 5  # bull, bear, sideways, volatile, crisis
            elif self.config.regime_encoding == "ordinal":
                regime_dim = 1
            else:
                regime_dim = 8  # embedding dimension
            
            self.feature_map["market_regime"] = (current_idx, current_idx + regime_dim)
            self.observation_dim += regime_dim
            current_idx += regime_dim
        
        # Time features
        if self.config.include_time_features:
            time_dim = len(self.config.time_features)
            self.feature_map["time_features"] = (current_idx, current_idx + time_dim)
            self.observation_dim += time_dim
            current_idx += time_dim
    
    def create_space(self) -> spaces.Box:
        """Create gymnasium observation space"""
        
        # For most financial features, allow negative values but limit extremes
        low = -np.inf * np.ones(self.observation_dim, dtype=np.float32)
        high = np.inf * np.ones(self.observation_dim, dtype=np.float32)
        
        # Set reasonable bounds for some features
        if "sentiment" in self.feature_map:
            start, end = self.feature_map["sentiment"]
            low[start:end] = -1.0   # Sentiment typically [-1, 1]
            high[start:end] = 1.0
        
        if "market_regime" in self.feature_map and self.config.regime_encoding == "onehot":
            start, end = self.feature_map["market_regime"]
            low[start:end] = 0.0    # One-hot encoding [0, 1]
            high[start:end] = 1.0
        
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.observation_dim,),
            dtype=np.float32 if self.config.use_float32 else np.float64
        )
    
    def build_observation(
        self,
        prices: Dict[str, float],
        price_history: Optional[List[Dict[str, float]]] = None,
        volumes: Optional[Dict[str, float]] = None,
        price_changes: Optional[Dict[str, float]] = None,
        technical_indicators: Optional[Dict[str, Dict[str, float]]] = None,
        sentiment_scores: Optional[Dict[str, Dict[str, float]]] = None,
        order_books: Optional[Dict[str, Dict]] = None,
        portfolio_state: Optional[Dict[str, float]] = None,
        position_history: Optional[List[Dict[str, float]]] = None,
        market_regime: Optional[str] = None,
        timestamp: Optional[float] = None,
        step: Optional[int] = None
    ) -> np.ndarray:
        """Build complete observation vector"""
        
        # Check cache
        if self.observation_cache is not None and step == self.last_cache_step:
            return self.observation_cache["observation"]
        
        observation = np.zeros(self.observation_dim, dtype=np.float32)
        
        try:
            # Price features
            if self.config.include_price and price_history:
                self._add_price_features(observation, price_history)
            
            # Volume features
            if self.config.include_volume and volumes and price_history:
                self._add_volume_features(observation, volumes, price_history)
            
            # Price changes
            if self.config.include_price_changes and price_changes:
                self._add_price_change_features(observation, price_changes)
            
            # Technical indicators
            if self.config.include_technical_indicators and technical_indicators:
                self._add_technical_features(observation, technical_indicators)
            
            # Sentiment features
            if self.config.include_sentiment and sentiment_scores:
                self._add_sentiment_features(observation, sentiment_scores)
            
            # Order book features
            if self.config.include_order_book and order_books:
                self._add_orderbook_features(observation, order_books)
            
            # Portfolio state
            if self.config.include_portfolio_state and portfolio_state:
                self._add_portfolio_features(observation, portfolio_state)
            
            # Position history
            if self.config.include_position_history and position_history:
                self._add_position_history_features(observation, position_history)
            
            # Market regime
            if self.config.include_market_regime and market_regime:
                self._add_regime_features(observation, market_regime)
            
            # Time features
            if self.config.include_time_features and timestamp:
                self._add_time_features(observation, timestamp)
            
            # Cache observation
            if self.observation_cache is not None:
                self.observation_cache["observation"] = observation.copy()
                self.last_cache_step = step
            
            return observation
            
        except Exception as e:
            warnings.warn(f"Error building observation: {e}")
            return np.zeros(self.observation_dim, dtype=np.float32)
    
    def _add_price_features(self, observation: np.ndarray, price_history: List[Dict[str, float]]) -> None:
        """Add price history features"""
        
        start_idx, end_idx = self.feature_map["prices"]
        feature_idx = start_idx
        
        # Get recent price history
        recent_history = price_history[-self.config.price_history_length:] if len(price_history) >= self.config.price_history_length else price_history
        
        for asset in self.assets:
            asset_prices = []
            
            # Extract price series for asset
            for price_dict in recent_history:
                asset_prices.append(price_dict.get(asset, 0.0))
            
            # Pad if insufficient history
            while len(asset_prices) < self.config.price_history_length:
                asset_prices.insert(0, asset_prices[0] if asset_prices else 0.0)
            
            # Normalize prices
            if self.config.normalize_prices:
                asset_prices = self._normalize_series(f"{asset}_price", asset_prices)
            
            # Add to observation
            observation[feature_idx:feature_idx + len(asset_prices)] = asset_prices
            feature_idx += len(asset_prices)
    
    def _add_volume_features(self, observation: np.ndarray, volumes: Dict[str, float], price_history: List[Dict]) -> None:
        """Add volume features"""
        
        start_idx, end_idx = self.feature_map["volumes"]
        feature_idx = start_idx
        
        for asset in self.assets:
            # Create volume history (simplified - using current volume)
            current_volume = volumes.get(asset, 0.0)
            volume_series = [current_volume] * self.config.price_history_length
            
            # Normalize volumes
            if self.config.normalize_volumes:
                volume_series = self._normalize_series(f"{asset}_volume", volume_series)
            
            # Add to observation
            observation[feature_idx:feature_idx + len(volume_series)] = volume_series
            feature_idx += len(volume_series)
    
    def _add_price_change_features(self, observation: np.ndarray, price_changes: Dict[str, float]) -> None:
        """Add price change features"""
        
        start_idx, end_idx = self.feature_map["price_changes"]
        
        changes = [price_changes.get(asset, 0.0) for asset in self.assets]
        observation[start_idx:end_idx] = changes
    
    def _add_technical_features(self, observation: np.ndarray, technical_indicators: Dict[str, Dict[str, float]]) -> None:
        """Add technical indicator features"""
        
        start_idx, end_idx = self.feature_map["technical_indicators"]
        feature_idx = start_idx
        
        for asset in self.assets:
            asset_indicators = technical_indicators.get(asset, {})
            
            for indicator_name in self.config.technical_indicators:
                indicator_value = asset_indicators.get(indicator_name, 0.0)
                
                # Normalize indicator
                if self.config.normalize_indicators:
                    indicator_value = self._normalize_value(f"{asset}_{indicator_name}", indicator_value)
                
                observation[feature_idx] = indicator_value
                feature_idx += 1
    
    def _add_sentiment_features(self, observation: np.ndarray, sentiment_scores: Dict[str, Dict[str, float]]) -> None:
        """Add sentiment features"""
        
        start_idx, end_idx = self.feature_map["sentiment"]
        feature_idx = start_idx
        
        for asset in self.assets:
            asset_sentiment = sentiment_scores.get(asset, {})
            
            for source in self.config.sentiment_sources:
                sentiment_value = asset_sentiment.get(source, 0.0)
                
                # Sentiment scores are typically already normalized [-1, 1] or [0, 100]
                if source == "fear_greed":
                    sentiment_value = (sentiment_value - 50.0) / 50.0  # Normalize to [-1, 1]
                
                observation[feature_idx] = sentiment_value
                feature_idx += 1
    
    def _add_orderbook_features(self, observation: np.ndarray, order_books: Dict[str, Dict]) -> None:
        """Add order book features"""
        
        start_idx, end_idx = self.feature_map["order_book"]
        feature_idx = start_idx
        
        for asset in self.assets:
            order_book = order_books.get(asset, {"bids": [], "asks": []})
            
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            
            # Add bid/ask data for each depth level
            for level in range(self.config.order_book_depth):
                if level < len(bids):
                    bid_price, bid_volume = bids[level]
                else:
                    bid_price, bid_volume = 0.0, 0.0
                
                if level < len(asks):
                    ask_price, ask_volume = asks[level]
                else:
                    ask_price, ask_volume = 0.0, 0.0
                
                observation[feature_idx:feature_idx + 4] = [bid_price, bid_volume, ask_price, ask_volume]
                feature_idx += 4
    
    def _add_portfolio_features(self, observation: np.ndarray, portfolio_state: Dict[str, float]) -> None:
        """Add portfolio state features"""
        
        start_idx, end_idx = self.feature_map["portfolio"]
        feature_idx = start_idx
        
        # Current positions
        for asset in self.assets:
            position = portfolio_state.get("positions", {}).get(asset, 0.0)
            observation[feature_idx] = position
            feature_idx += 1
        
        # Portfolio metrics
        balance = portfolio_state.get("balance", 0.0)
        portfolio_value = portfolio_state.get("portfolio_value", 0.0)
        max_drawdown = portfolio_state.get("max_drawdown", 0.0)
        step_ratio = portfolio_state.get("step", 0) / portfolio_state.get("max_steps", 1000)
        
        observation[feature_idx:feature_idx + 4] = [balance, portfolio_value, max_drawdown, step_ratio]
    
    def _add_position_history_features(self, observation: np.ndarray, position_history: List[Dict[str, float]]) -> None:
        """Add position history features"""
        
        start_idx, end_idx = self.feature_map["position_history"]
        feature_idx = start_idx
        
        # Get recent position history
        recent_positions = position_history[-self.config.position_history_length:] if len(position_history) >= self.config.position_history_length else position_history
        
        for asset in self.assets:
            asset_positions = []
            
            for pos_dict in recent_positions:
                asset_positions.append(pos_dict.get(asset, 0.0))
            
            # Pad if insufficient history
            while len(asset_positions) < self.config.position_history_length:
                asset_positions.insert(0, 0.0)
            
            observation[feature_idx:feature_idx + len(asset_positions)] = asset_positions
            feature_idx += len(asset_positions)
    
    def _add_regime_features(self, observation: np.ndarray, market_regime: str) -> None:
        """Add market regime features"""
        
        start_idx, end_idx = self.feature_map["market_regime"]
        
        regime_map = {
            "bull": 0, "bear": 1, "sideways": 2, "volatile": 3, "crisis": 4
        }
        
        if self.config.regime_encoding == "onehot":
            regime_idx = regime_map.get(market_regime, 2)  # Default to sideways
            observation[start_idx + regime_idx] = 1.0
        
        elif self.config.regime_encoding == "ordinal":
            regime_value = regime_map.get(market_regime, 2) / 4.0  # Normalize to [0, 1]
            observation[start_idx] = regime_value
    
    def _add_time_features(self, observation: np.ndarray, timestamp: float) -> None:
        """Add time-based features"""
        
        start_idx, end_idx = self.feature_map["time_features"]
        feature_idx = start_idx
        
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        
        time_features_map = {
            "hour": dt.hour / 24.0,
            "day_of_week": dt.weekday() / 6.0,
            "day_of_month": dt.day / 31.0,
            "quarter": ((dt.month - 1) // 3) / 3.0
        }
        
        for time_feature in self.config.time_features:
            value = time_features_map.get(time_feature, 0.0)
            observation[feature_idx] = value
            feature_idx += 1
    
    def _normalize_series(self, key: str, values: List[float]) -> List[float]:
        """Normalize a series of values"""
        
        if key in self.normalizers:
            normalizer = self.normalizers[key]
            normalized = []
            
            for value in values:
                normalized_value = normalizer.normalize(value)
                normalized.append(normalized_value)
            
            return normalized
        
        return values
    
    def _normalize_value(self, key: str, value: float) -> float:
        """Normalize a single value"""
        
        if key in self.normalizers:
            return self.normalizers[key].normalize(value)
        
        return value
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in observation"""
        
        feature_names = []
        
        # Price features
        if "prices" in self.feature_map:
            for asset in self.assets:
                for i in range(self.config.price_history_length):
                    feature_names.append(f"{asset}_price_t-{self.config.price_history_length-i-1}")
        
        # Volume features
        if "volumes" in self.feature_map:
            for asset in self.assets:
                for i in range(self.config.price_history_length):
                    feature_names.append(f"{asset}_volume_t-{self.config.price_history_length-i-1}")
        
        # Price changes
        if "price_changes" in self.feature_map:
            for asset in self.assets:
                feature_names.append(f"{asset}_price_change")
        
        # Technical indicators
        if "technical_indicators" in self.feature_map:
            for asset in self.assets:
                for indicator in self.config.technical_indicators:
                    feature_names.append(f"{asset}_{indicator}")
        
        # Add other feature names...
        # (Implementation continues for all other feature types)
        
        return feature_names
    
    def get_observation_info(self) -> Dict[str, Any]:
        """Get information about observation space"""
        
        return {
            "total_dimension": self.observation_dim,
            "feature_map": self.feature_map.copy(),
            "assets": self.assets.copy(),
            "config": self.config.__dict__,
            "feature_names": self.get_feature_names()
        }


__all__ = [
    "ObservationMode",
    "ObservationConfig", 
    "CryptoObservationSpace"
]