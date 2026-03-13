"""
Data Stream Generator for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system generation streams data for simulation
continuous training in various market conditions with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Iterator, Generator
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
from abc import ABC, abstractmethod
import torch


class MarketRegime(Enum):
    """Market regimes for simulation"""
    BULL = "bull"           # Bull market - upward trend
    BEAR = "bear"           # Bear market - downward trend  
    SIDEWAYS = "sideways" #
    VOLATILE = "volatile"   # High volatility


class DataStreamType(Enum):
    """Types streams data"""
    SYNTHETIC = "synthetic"     # Synthetic data
    HISTORICAL = "historical" # data
    MIXED = "mixed" # data
    REAL_TIME = "real_time" # Real data in time


@dataclass
class StreamConfig:
    """Configuration stream data"""
    # Main parameters
    stream_type: DataStreamType = DataStreamType.SYNTHETIC
    regime_duration_hours: int = 48 # each regime
    samples_per_hour: int = 60  # Number samples in hour
    feature_dim: int = 20  # Dimension features
    
    # Market parameters
    volatility_base: float = 0.02  # Base volatility
    trend_strength: float = 0.005 # Strength trend
    noise_level: float = 0.01 # Level noise
    
    # enterprise parameters
    regime_transitions: bool = True # between regimes
    drift_simulation: bool = True # Simulation drift data
    anomaly_injection: bool = True # Integration anomalies
    correlation_patterns: bool = True # Patterns correlation
    
    # parameters
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "ADA"])
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    include_volume: bool = True
    include_social_sentiment: bool = True
    
    # Technical parameters
    random_seed: Optional[int] = None
    buffer_size: int = 1000
    streaming_mode: bool = True


@dataclass
class StreamSample:
    """Sample data from stream"""
    # Main data
    features: np.ndarray
    target: float
    timestamp: datetime
    
    # Metadata
    market_regime: MarketRegime
    asset: str
    timeframe: str
    sample_id: str
    
    # Additional information
    volatility: float = 0.0
    volume: float = 0.0
    sentiment: float = 0.0
    anomaly: bool = False
    
    # labels
    task_id: Optional[int] = None
    difficulty_level: str = "medium"
    noise_level: float = 0.0


class BaseStreamGenerator(ABC):
    """
    Base class for generators streams data
    
    enterprise Features:
    - Configurable market regime simulation
    - Realistic crypto market patterns
    - Anomaly detection and injection
    - Performance monitoring
    - Adaptive difficulty scaling
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        # Initialize generator random
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
        
        # State stream
        self.current_regime = MarketRegime.BULL
        self.regime_start_time = datetime.now()
        self.sample_counter = 0
        self.task_id_counter = 0
        
        # Buffer for streaming
        self.sample_buffer: List[StreamSample] = []
        
        # enterprise components
        self.regime_transition_prob = 0.1 # Probability switching regime
        self.anomaly_prob = 0.05 # Probability anomalies
        self.drift_rate = 0.001 # Speed drift
        
        # Statistics
        self.regime_statistics: Dict[MarketRegime, int] = {regime: 0 for regime in MarketRegime}
        self.anomaly_count = 0
        self.generated_samples = 0
    
    @abstractmethod
    def generate_sample(self) -> StreamSample:
        """Generation one samples data"""
        pass
    
    def generate_stream(self, num_samples: int) -> Iterator[StreamSample]:
        """
        Generation stream data
        
        Args:
            num_samples: Number samples for generation
            
        Yields:
            StreamSample: Samples data
        """
        for _ in range(num_samples):
            sample = self.generate_sample()
            self.sample_buffer.append(sample)
            
            # buffer
            if len(self.sample_buffer) > self.config.buffer_size:
                self.sample_buffer.pop(0)
            
            self.generated_samples += 1
            yield sample
    
    def generate_batch(self, batch_size: int) -> List[StreamSample]:
        """
        Generation batch samples
        
        Args:
            batch_size: Size batch
            
        Returns:
            List samples
        """
        return [self.generate_sample() for _ in range(batch_size)]
    
    def should_transition_regime(self) -> bool:
        """
        Determine necessity switching market regime
        
        Returns:
            True if need to regime
        """
        if not self.config.regime_transitions:
            return False
        
        # Check time in regime
        time_in_regime = datetime.now() - self.regime_start_time
        min_duration = timedelta(hours=self.config.regime_duration_hours)
        
        if time_in_regime < min_duration:
            return False
        
        # regime
        return random.random() < self.regime_transition_prob
    
    def transition_to_new_regime(self) -> None:
        """Transition to new market regime"""
        # Select new regime (excluding current)
        available_regimes = [r for r in MarketRegime if r != self.current_regime]
        new_regime = random.choice(available_regimes)
        
        self.logger.info(f"Transitioning from {self.current_regime.value} to {new_regime.value}")
        
        # Update state
        self.current_regime = new_regime
        self.regime_start_time = datetime.now()
        self.task_id_counter += 1
        
        # Statistics
        self.regime_statistics[new_regime] = self.regime_statistics.get(new_regime, 0) + 1
    
    def inject_anomaly(self, features: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Integration anomalies in data
        
        Args:
            features: Source features
            
        Returns:
            (_features, flag_anomalies)
        """
        if not self.config.anomaly_injection:
            return features, False
        
        if random.random() > self.anomaly_prob:
            return features, False
        
        # Types anomalies
        anomaly_type = random.choice(["spike", "shift", "noise"])
        modified_features = features.copy()
        
        if anomaly_type == "spike":
            # in
            spike_idx = random.randint(0, len(features) - 1)
            spike_magnitude = random.uniform(3.0, 10.0)
            modified_features[spike_idx] *= spike_magnitude
            
        elif anomaly_type == "shift":
            # all features
            shift_magnitude = random.uniform(1.5, 3.0)
            modified_features *= shift_magnitude
            
        elif anomaly_type == "noise":
            # Add level noise
            noise_magnitude = random.uniform(0.5, 2.0)
            noise = np.random.normal(0, noise_magnitude, features.shape)
            modified_features += noise
        
        self.anomaly_count += 1
        return modified_features, True
    
    def apply_concept_drift(self, features: np.ndarray) -> np.ndarray:
        """
        Apply concept drift to data
        
        Args:
            features: Source features
            
        Returns:
             features with
        """
        if not self.config.drift_simulation:
            return features
        
        # drift on basis time
        drift_factor = 1.0 + self.drift_rate * self.generated_samples
        drift_noise = np.random.normal(0, self.drift_rate, features.shape)
        
        drifted_features = features * drift_factor + drift_noise
        return drifted_features
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """
        Get statistics stream data
        
        Returns:
            Statistics stream
        """
        stats = {
            "total_samples_generated": self.generated_samples,
            "current_regime": self.current_regime.value,
            "regime_statistics": {regime.value: count for regime, count in self.regime_statistics.items()},
            "anomalies_injected": self.anomaly_count,
            "current_task_id": self.task_id_counter,
            "buffer_size": len(self.sample_buffer),
            "anomaly_rate": self.anomaly_count / max(1, self.generated_samples)
        }
        
        return stats


class SyntheticCryptoStreamGenerator(BaseStreamGenerator):
    """
    Generator synthetic cryptocurrency data
    
     Implementation for crypto patterns
    """
    
    def __init__(self, config: StreamConfig):
        super().__init__(config)
        
        # Parameters for each regime
        self.regime_params = {
            MarketRegime.BULL: {
                "trend": 0.01,
                "volatility": 0.015,
                "mean_reversion": 0.1
            },
            MarketRegime.BEAR: {
                "trend": -0.008,
                "volatility": 0.025,
                "mean_reversion": 0.15
            },
            MarketRegime.SIDEWAYS: {
                "trend": 0.0,
                "volatility": 0.012,
                "mean_reversion": 0.3
            },
            MarketRegime.VOLATILE: {
                "trend": 0.0,
                "volatility": 0.05,
                "mean_reversion": 0.05
            }
        }
        
        # State for each asset
        self.asset_states = {}
        for asset in config.assets:
            self.asset_states[asset] = {
                "price": 100.0, # Initial price
                "momentum": 0.0,
                "volatility_state": 1.0
            }
    
    def generate_sample(self) -> StreamSample:
        """Generation synthetic samples crypto data"""
        # Check necessity switching regime
        if self.should_transition_regime():
            self.transition_to_new_regime()
        
        # Select asset and
        asset = random.choice(self.config.assets)
        timeframe = random.choice(self.config.timeframes)
        
        # Generation features
        features = self._generate_technical_features(asset)
        
        # Generation goal (change prices)
        target = self._generate_price_target(asset)
        
        # Apply concept drift
        features = self.apply_concept_drift(features)
        
        # Integration anomalies
        features, is_anomaly = self.inject_anomaly(features)
        
        # Create samples
        sample = StreamSample(
            features=features,
            target=target,
            timestamp=datetime.now(),
            market_regime=self.current_regime,
            asset=asset,
            timeframe=timeframe,
            sample_id=f"{asset}_{timeframe}_{self.sample_counter}",
            volatility=self.regime_params[self.current_regime]["volatility"],
            volume=random.uniform(1000, 10000),
            sentiment=random.uniform(-1, 1),
            anomaly=is_anomaly,
            task_id=self.task_id_counter,
            difficulty_level=self._assess_difficulty(),
            noise_level=self.config.noise_level
        )
        
        self.sample_counter += 1
        return sample
    
    def _generate_technical_features(self, asset: str) -> np.ndarray:
        """
        Generation technical indicators
        
        Args:
            asset: Name asset
            
        Returns:
             technical features
        """
        state = self.asset_states[asset]
        regime_params = self.regime_params[self.current_regime]
        
        features = []
        
        # features
        features.extend([
            state["price"] / 100.0, # price
            state["momentum"], #
            regime_params["volatility"]  # Current volatility
        ])
        
        # Technical indicators ()
        # RSI
        rsi = 50 + 30 * np.sin(self.sample_counter * 0.1) + random.uniform(-10, 10)
        features.append(np.clip(rsi / 100.0, 0, 1))
        
        # MACD
        macd = regime_params["trend"] + random.uniform(-0.01, 0.01)
        features.append(macd)
        
        # Bollinger Bands position
        bb_position = random.uniform(0, 1)
        features.append(bb_position)
        
        # Volume indicators
        volume_sma = random.uniform(0.5, 2.0)
        features.append(volume_sma)
        
        # Volatility
        features.append(state["volatility_state"])
        
        # Correlation with other assets
        for other_asset in self.config.assets:
            if other_asset != asset:
                correlation = random.uniform(-0.5, 0.8)
                features.append(correlation)
        
        # Augmentation up to required dimensions
        while len(features) < self.config.feature_dim:
            features.append(random.uniform(-1, 1))
        
        # up to required dimensions
        features = features[:self.config.feature_dim]
        
        # Update state asset
        self._update_asset_state(asset)
        
        return np.array(features, dtype=np.float32)
    
    def _generate_price_target(self, asset: str) -> float:
        """
        Generation target prices
        
        Args:
            asset: Name asset
            
        Returns:
             change prices
        """
        regime_params = self.regime_params[self.current_regime]
        state = self.asset_states[asset]
        
        # Basic change on basis trend
        trend_component = regime_params["trend"]
        
        # Component mean reversion
        mean_reversion_component = -state["momentum"] * regime_params["mean_reversion"]
        
        # Random component
        random_component = random.gauss(0, regime_params["volatility"])
        
        # Total change
        price_change = trend_component + mean_reversion_component + random_component
        
        # Limitation values
        price_change = np.clip(price_change, -0.1, 0.1)
        
        return price_change
    
    def _update_asset_state(self, asset: str) -> None:
        """
        Update state asset
        
        Args:
            asset: Name asset
        """
        state = self.asset_states[asset]
        regime_params = self.regime_params[self.current_regime]
        
        # Update
        momentum_decay = 0.9
        new_momentum = state["momentum"] * momentum_decay + regime_params["trend"]
        state["momentum"] = new_momentum
        
        # Update prices
        price_change = self._generate_price_target(asset)
        state["price"] *= (1 + price_change)
        state["price"] = max(state["price"], 1.0)  # Minimal price
        
        # Update state volatility
        volatility_change = random.gauss(0, 0.001)
        state["volatility_state"] = np.clip(
            state["volatility_state"] + volatility_change,
            0.1, 3.0
        )
    
    def _assess_difficulty(self) -> str:
        """
        Evaluate complexity current conditions
        
        Returns:
            Level complexity: easy, medium, hard
        """
        regime_difficulty = {
            MarketRegime.BULL: "easy",
            MarketRegime.SIDEWAYS: "medium", 
            MarketRegime.BEAR: "medium",
            MarketRegime.VOLATILE: "hard"
        }
        
        base_difficulty = regime_difficulty.get(self.current_regime, "medium")
        
        # on basis anomalies
        if self.anomaly_count > 0 and self.anomaly_count / max(1, self.generated_samples) > 0.1:
            if base_difficulty == "easy":
                base_difficulty = "medium"
            elif base_difficulty == "medium":
                base_difficulty = "hard"
        
        return base_difficulty


class HistoricalDataStreamGenerator(BaseStreamGenerator):
    """
    Generator on basis historical data
    
     Implementation for work with crypto data
    """
    
    def __init__(self, config: StreamConfig, historical_data: Optional[pd.DataFrame] = None):
        super().__init__(config)
        
        if historical_data is not None:
            self.historical_data = historical_data
        else:
            # Create mock historical data
            self.historical_data = self._create_mock_historical_data()
        
        self.data_index = 0
        self.regime_detector = self._init_regime_detector()
    
    def _create_mock_historical_data(self) -> pd.DataFrame:
        """Create mock historical data"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='1H'
        )
        
        data = []
        for date in dates:
            row = {
                'timestamp': date,
                'open': random.uniform(95, 105),
                'high': random.uniform(100, 110),
                'low': random.uniform(90, 100),
                'close': random.uniform(95, 105),
                'volume': random.uniform(1000, 10000)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _init_regime_detector(self) -> Dict[str, Any]:
        """Initialize regimes"""
        return {
            "lookback_window": 24, # Window for analysis regime
            "volatility_threshold": 0.02,
            "trend_threshold": 0.01
        }
    
    def generate_sample(self) -> StreamSample:
        """Generation samples on basis historical data"""
        if self.data_index >= len(self.historical_data):
            # to data
            self.data_index = 0
        
        row = self.historical_data.iloc[self.data_index]
        
        # Determine market regime
        current_regime = self._detect_market_regime(self.data_index)
        
        # Generation features from historical data
        features = self._extract_features_from_historical(self.data_index)
        
        # Generation goal
        target = self._extract_target_from_historical(self.data_index)
        
        # Apply
        features = self.apply_concept_drift(features)
        features, is_anomaly = self.inject_anomaly(features)
        
        sample = StreamSample(
            features=features,
            target=target,
            timestamp=row['timestamp'],
            market_regime=current_regime,
            asset=random.choice(self.config.assets),
            timeframe="1h",
            sample_id=f"historical_{self.data_index}",
            volatility=self._calculate_volatility(self.data_index),
            volume=row['volume'],
            anomaly=is_anomaly,
            task_id=self.task_id_counter
        )
        
        self.data_index += 1
        return sample
    
    def _detect_market_regime(self, index: int) -> MarketRegime:
        """
        Determine market regime on basis historical data
        
        Args:
            index: Index in data
            
        Returns:
             market regime
        """
        window_size = min(self.regime_detector["lookback_window"], index + 1)
        start_idx = max(0, index - window_size + 1)
        
        window_data = self.historical_data.iloc[start_idx:index + 1]
        
        if len(window_data) < 2:
            return MarketRegime.SIDEWAYS
        
        # Calculation main
        returns = window_data['close'].pct_change().dropna()
        volatility = returns.std()
        mean_return = returns.mean()
        
        # Classification regime
        if volatility > self.regime_detector["volatility_threshold"]:
            return MarketRegime.VOLATILE
        elif mean_return > self.regime_detector["trend_threshold"]:
            return MarketRegime.BULL
        elif mean_return < -self.regime_detector["trend_threshold"]:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS
    
    def _extract_features_from_historical(self, index: int) -> np.ndarray:
        """Extraction features from historical data"""
        window_size = min(20, index + 1) # Window for calculation indicators
        start_idx = max(0, index - window_size + 1)
        
        window_data = self.historical_data.iloc[start_idx:index + 1]
        current_row = self.historical_data.iloc[index]
        
        features = []
        
        # Base features
        features.extend([
            current_row['open'] / 100.0,
            current_row['high'] / 100.0,
            current_row['low'] / 100.0,
            current_row['close'] / 100.0,
            current_row['volume'] / 5000.0
        ])
        
        # Technical indicators
        if len(window_data) >= 2:
            # Simple average
            sma = window_data['close'].rolling(min(10, len(window_data))).mean().iloc[-1]
            features.append(current_row['close'] / sma)
            
            # Returns
            returns = window_data['close'].pct_change().dropna()
            if len(returns) > 0:
                features.extend([
                    returns.iloc[-1],  # Last return
                    returns.mean(),    # Average return
                    returns.std()      # Volatility
                ])
            else:
                features.extend([0.0, 0.0, 0.01])
        else:
            features.extend([1.0, 0.0, 0.0, 0.01])
        
        # Augmentation up to required dimensions
        while len(features) < self.config.feature_dim:
            features.append(random.uniform(-1, 1))
        
        return np.array(features[:self.config.feature_dim], dtype=np.float32)
    
    def _extract_target_from_historical(self, index: int) -> float:
        """Extraction target values from historical data"""
        current_row = self.historical_data.iloc[index]
        
        # value for forecast (if available)
        if index + 1 < len(self.historical_data):
            next_row = self.historical_data.iloc[index + 1]
            return (next_row['close'] - current_row['close']) / current_row['close']
        else:
            # If no next values, use random change
            return random.gauss(0, 0.01)
    
    def _calculate_volatility(self, index: int) -> float:
        """Calculation volatility on basis historical data"""
        window_size = min(24, index + 1)
        start_idx = max(0, index - window_size + 1)
        
        window_data = self.historical_data.iloc[start_idx:index + 1]
        returns = window_data['close'].pct_change().dropna()
        
        return returns.std() if len(returns) > 0 else 0.01


#  Production-Ready Factory
class StreamGeneratorFactory:
    """
    Factory for creation generators streams data
    with enterprise patterns
    """
    
    @staticmethod
    def create_crypto_stream(
        stream_type: DataStreamType = DataStreamType.SYNTHETIC,
        market_regimes: Optional[List[MarketRegime]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> BaseStreamGenerator:
        """
        Create generator crypto stream
        
        Args:
            stream_type: Type stream data
            market_regimes: List market regimes for simulation
            config_overrides: Overrides configuration
            
        Returns:
            Configured generator stream
        """
        # Base configuration for crypto trading
        config = StreamConfig(
            stream_type=stream_type,
            regime_duration_hours=24,
            samples_per_hour=60,
            feature_dim=25,
            volatility_base=0.02,
            trend_strength=0.008,
            noise_level=0.005,
            assets=["BTC", "ETH", "ADA", "DOT", "SOL"],
            timeframes=["1h", "4h", "1d"],
            regime_transitions=True,
            drift_simulation=True,
            anomaly_injection=True,
            correlation_patterns=True
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Create corresponding generator
        if stream_type == DataStreamType.SYNTHETIC:
            return SyntheticCryptoStreamGenerator(config)
        elif stream_type == DataStreamType.HISTORICAL:
            return HistoricalDataStreamGenerator(config)
        else:
            # Default fallback
            return SyntheticCryptoStreamGenerator(config)