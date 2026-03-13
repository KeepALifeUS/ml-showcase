"""
Synthetic Data Generator for Crypto Trading
Orchestrates multiple generative models for comprehensive data augmentation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class GenerationMethod(Enum):
    """Available generation methods"""
    GAN = "gan"
    VAE = "vae"
    TIMEGAN = "timegan"
    DIFFUSION = "diffusion"
    STATISTICAL = "statistical"


@dataclass 
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    method: GenerationMethod = GenerationMethod.TIMEGAN
    n_samples: int = 1000
    seq_len: int = 24
    features: List[str] = None
    
    # Data characteristics
    preserve_statistics: bool = True
    preserve_correlations: bool = True
    preserve_temporal_dynamics: bool = True
    
    # Augmentation parameters
    noise_level: float = 0.01
    outlier_fraction: float = 0.05
    rare_event_probability: float = 0.01
    
    # enterprise patterns
    quality_threshold: float = 0.95
    statistical_tests: bool = True
    visual_validation: bool = True
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['open', 'high', 'low', 'close', 'volume']


class CryptoSyntheticGenerator:
    """
    Comprehensive synthetic data generator for crypto trading
    Combines multiple generative models with quality assurance
    """
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.models = {}
        self.generated_data = None
        self.quality_metrics = {}
        
    def generate_ohlcv(
        self,
        base_data: Optional[pd.DataFrame] = None,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for crypto trading
        
        Args:
            base_data: Optional base data for statistics
            symbol: Trading symbol
            interval: Time interval
        
        Returns:
            DataFrame with synthetic OHLCV data
        """
        if base_data is not None:
            # Extract statistics from base data
            stats = self._extract_statistics(base_data)
        else:
            # Use default crypto market statistics
            stats = self._get_default_crypto_stats(symbol)
        
        # Generate based on selected method
        if self.config.method == GenerationMethod.TIMEGAN:
            synthetic = self._generate_timegan(stats)
        elif self.config.method == GenerationMethod.STATISTICAL:
            synthetic = self._generate_statistical(stats)
        else:
            synthetic = self._generate_statistical(stats)  # Fallback
        
        # Post-process to ensure validity
        synthetic = self._postprocess_ohlcv(synthetic)
        
        # Add metadata
        synthetic['symbol'] = symbol
        synthetic['interval'] = interval
        
        return synthetic
    
    def _extract_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract statistical properties from real data"""
        stats = {
            'mean': data[self.config.features].mean().to_dict(),
            'std': data[self.config.features].std().to_dict(),
            'min': data[self.config.features].min().to_dict(),
            'max': data[self.config.features].max().to_dict(),
            'corr': data[self.config.features].corr().to_dict(),
            'autocorr': {}
        }
        
        # Calculate autocorrelations
        for feature in self.config.features:
            if feature in data.columns:
                stats['autocorr'][feature] = [
                    data[feature].autocorr(lag=i) 
                    for i in range(1, min(10, len(data)))
                ]
        
        # Returns statistics
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            stats['returns'] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skew': returns.skew(),
                'kurtosis': returns.kurtosis()
            }
        
        return stats
    
    def _get_default_crypto_stats(self, symbol: str) -> Dict[str, Any]:
        """Get default statistics for crypto symbols"""
        # Default statistics based on typical crypto behavior
        if "BTC" in symbol:
            return {
                'mean': {'open': 50000, 'high': 51000, 'low': 49000, 'close': 50000, 'volume': 1000},
                'std': {'open': 2000, 'high': 2100, 'low': 1900, 'close': 2000, 'volume': 200},
                'min': {'open': 40000, 'high': 40500, 'low': 39500, 'close': 40000, 'volume': 500},
                'max': {'open': 60000, 'high': 61000, 'low': 59000, 'close': 60000, 'volume': 2000},
                'returns': {'mean': 0.001, 'std': 0.02, 'skew': -0.5, 'kurtosis': 5}
            }
        else:
            return {
                'mean': {'open': 100, 'high': 102, 'low': 98, 'close': 100, 'volume': 1000},
                'std': {'open': 5, 'high': 5.2, 'low': 4.8, 'close': 5, 'volume': 200},
                'min': {'open': 80, 'high': 81, 'low': 79, 'close': 80, 'volume': 500},
                'max': {'open': 120, 'high': 122, 'low': 118, 'close': 120, 'volume': 2000},
                'returns': {'mean': 0.0005, 'std': 0.03, 'skew': -0.3, 'kurtosis': 4}
            }
    
    def _generate_statistical(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """Generate data using statistical methods"""
        n_samples = self.config.n_samples
        
        # Generate correlated returns
        mean_return = stats.get('returns', {}).get('mean', 0.001)
        std_return = stats.get('returns', {}).get('std', 0.02)
        
        # Generate returns with autocorrelation
        returns = self._generate_autocorrelated_returns(
            n_samples, mean_return, std_return
        )
        
        # Generate prices from returns
        initial_price = stats['mean'].get('close', 100)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = pd.DataFrame()
        
        # Close prices
        data['close'] = prices
        
        # Open prices (previous close with small gap)
        data['open'] = data['close'].shift(1)
        data['open'].iloc[0] = initial_price
        data['open'] *= np.random.uniform(0.995, 1.005, n_samples)
        
        # High prices (above open and close)
        high_factor = np.random.uniform(1.001, 1.02, n_samples)
        data['high'] = np.maximum(data['open'], data['close']) * high_factor
        
        # Low prices (below open and close)
        low_factor = np.random.uniform(0.98, 0.999, n_samples)
        data['low'] = np.minimum(data['open'], data['close']) * low_factor
        
        # Volume (correlated with price volatility)
        volatility = np.abs(returns)
        base_volume = stats['mean'].get('volume', 1000)
        data['volume'] = base_volume * (1 + volatility * 10) * np.random.lognormal(0, 0.3, n_samples)
        
        # Add timestamp
        data['timestamp'] = pd.date_range(
            start='2024-01-01', 
            periods=n_samples, 
            freq='1h'
        )
        
        return data
    
    def _generate_autocorrelated_returns(
        self,
        n_samples: int,
        mean: float,
        std: float,
        ar_coef: float = 0.1
    ) -> np.ndarray:
        """Generate autocorrelated returns"""
        returns = np.zeros(n_samples)
        noise = np.random.normal(mean, std, n_samples)
        
        returns[0] = noise[0]
        for i in range(1, n_samples):
            returns[i] = ar_coef * returns[i-1] + (1 - ar_coef) * noise[i]
        
        return returns
    
    def _generate_timegan(self, stats: Dict[str, Any]) -> pd.DataFrame:
        """Generate data using TimeGAN (placeholder)"""
        # This would use the actual TimeGAN model
        # For now, use statistical generation
        return self._generate_statistical(stats)
    
    def _postprocess_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Post-process to ensure OHLCV validity"""
        # Ensure OHLCV relationships
        data['high'] = np.maximum(data['high'], data['open'])
        data['high'] = np.maximum(data['high'], data['close'])
        data['low'] = np.minimum(data['low'], data['open'])
        data['low'] = np.minimum(data['low'], data['close'])
        
        # Ensure positive values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = np.abs(data[col])
                data[col] = data[col].replace(0, data[col].mean())
        
        # Round to reasonable precision
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col].round(2)
        
        if 'volume' in data.columns:
            data['volume'] = data['volume'].round(0)
        
        return data
    
    def add_market_events(
        self,
        data: pd.DataFrame,
        event_type: str = "crash",
        probability: float = 0.01
    ) -> pd.DataFrame:
        """Add synthetic market events to data"""
        n_samples = len(data)
        event_mask = np.random.random(n_samples) < probability
        
        if event_type == "crash":
            # Sudden price drop
            crash_factor = np.random.uniform(0.7, 0.9, n_samples)
            data.loc[event_mask, ['open', 'high', 'low', 'close']] *= crash_factor[event_mask].reshape(-1, 1)
            data.loc[event_mask, 'volume'] *= 3  # Spike in volume
            
        elif event_type == "pump":
            # Sudden price increase
            pump_factor = np.random.uniform(1.1, 1.3, n_samples)
            data.loc[event_mask, ['open', 'high', 'low', 'close']] *= pump_factor[event_mask].reshape(-1, 1)
            data.loc[event_mask, 'volume'] *= 2.5
            
        elif event_type == "volatility":
            # Increased volatility
            vol_factor = np.random.uniform(1.5, 3, n_samples)
            price_cols = ['open', 'high', 'low', 'close']
            mean_price = data[price_cols].mean(axis=1)
            
            for col in price_cols:
                deviation = data[col] - mean_price
                data.loc[event_mask, col] = mean_price[event_mask] + deviation[event_mask] * vol_factor[event_mask]
        
        return data
    
    def augment_with_noise(
        self,
        data: pd.DataFrame,
        noise_level: float = None
    ) -> pd.DataFrame:
        """Add controlled noise for augmentation"""
        if noise_level is None:
            noise_level = self.config.noise_level
        
        augmented = data.copy()
        
        # Add Gaussian noise to prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in augmented.columns:
                noise = np.random.normal(0, augmented[col].std() * noise_level, len(augmented))
                augmented[col] += noise
        
        # Add multiplicative noise to volume
        if 'volume' in augmented.columns:
            volume_noise = np.random.lognormal(0, noise_level, len(augmented))
            augmented['volume'] *= volume_noise
        
        # Ensure validity
        augmented = self._postprocess_ohlcv(augmented)
        
        return augmented
    
    def generate_order_book(
        self,
        mid_price: float,
        spread: float = 0.001,
        depth: int = 20,
        volume_range: Tuple[float, float] = (0.1, 10)
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic order book data"""
        bids = []
        asks = []
        
        # Generate bid side
        for i in range(depth):
            price = mid_price * (1 - spread * (i + 1))
            volume = np.random.uniform(*volume_range)
            bids.append({'price': price, 'volume': volume})
        
        # Generate ask side
        for i in range(depth):
            price = mid_price * (1 + spread * (i + 1))
            volume = np.random.uniform(*volume_range)
            asks.append({'price': price, 'volume': volume})
        
        return {
            'bids': pd.DataFrame(bids),
            'asks': pd.DataFrame(asks),
            'mid_price': mid_price,
            'spread': spread
        }
    
    def validate_quality(
        self,
        synthetic: pd.DataFrame,
        reference: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Validate quality of synthetic data"""
        metrics = {}
        
        # Basic statistics validation
        metrics['has_nulls'] = synthetic.isnull().any().any()
        metrics['has_negative_prices'] = (synthetic[['open', 'high', 'low', 'close']] < 0).any().any()
        metrics['ohlc_valid'] = ((synthetic['high'] >= synthetic['low']) & 
                                 (synthetic['high'] >= synthetic['open']) &
                                 (synthetic['high'] >= synthetic['close']) &
                                 (synthetic['low'] <= synthetic['open']) &
                                 (synthetic['low'] <= synthetic['close'])).all()
        
        if reference is not None:
            # Statistical similarity
            for col in ['close', 'volume']:
                if col in synthetic.columns and col in reference.columns:
                    metrics[f'{col}_mean_diff'] = abs(synthetic[col].mean() - reference[col].mean()) / reference[col].mean()
                    metrics[f'{col}_std_diff'] = abs(synthetic[col].std() - reference[col].std()) / reference[col].std()
            
            # Returns distribution similarity
            if 'close' in synthetic.columns and 'close' in reference.columns:
                synth_returns = synthetic['close'].pct_change().dropna()
                ref_returns = reference['close'].pct_change().dropna()
                
                from scipy import stats
                ks_stat, ks_pval = stats.ks_2samp(synth_returns, ref_returns)
                metrics['returns_ks_pvalue'] = ks_pval
                metrics['returns_similarity'] = 1 - ks_stat
        
        # Overall quality score
        quality_score = 1.0
        if metrics.get('has_nulls', False):
            quality_score *= 0.5
        if metrics.get('has_negative_prices', False):
            quality_score *= 0.5
        if not metrics.get('ohlc_valid', True):
            quality_score *= 0.7
        if 'returns_similarity' in metrics:
            quality_score *= metrics['returns_similarity']
        
        metrics['quality_score'] = quality_score
        
        self.quality_metrics = metrics
        return metrics