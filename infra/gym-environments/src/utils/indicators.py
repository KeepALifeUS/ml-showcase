"""
Technical Indicators for Trading Environments
enterprise patterns for high-performance indicator calculation

Comprehensive technical analysis indicators:
- Moving averages (SMA, EMA, WMA)
- Momentum indicators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, VWAP)
- Custom crypto-specific indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
import warnings
import logging


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    
    # Moving averages
    sma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    ema_periods: List[int] = field(default_factory=lambda: [12, 26, 50])
    wma_periods: List[int] = field(default_factory=lambda: [10, 20])
    
    # Momentum indicators
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # Volatility indicators
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Volume indicators
    obv_enabled: bool = True
    vwap_period: int = 20
    
    # Performance optimization
    use_cache: bool = True
    cache_size: int = 1000
    parallel_calculation: bool = False


class TechnicalIndicators:
    """
    High-performance technical indicators calculator
    
    Optimized for real-time calculation with caching and vectorization
    """
    
    def __init__(self, indicators: List[str], config: Optional[IndicatorConfig] = None):
        self.indicators = indicators
        self.config = config or IndicatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self.cache = {} if config.use_cache else None
        self.price_history = deque(maxlen=max(200, max(config.sma_periods + config.ema_periods)))
        self.volume_history = deque(maxlen=200)
        
        # Indicator calculation functions
        self.indicator_functions = {
            "sma_5": lambda p, v: self._sma(p, 5),
            "sma_10": lambda p, v: self._sma(p, 10),
            "sma_20": lambda p, v: self._sma(p, 20),
            "sma_50": lambda p, v: self._sma(p, 50),
            "ema_12": lambda p, v: self._ema(p, 12),
            "ema_26": lambda p, v: self._ema(p, 26),
            "ema_50": lambda p, v: self._ema(p, 50),
            "rsi_14": lambda p, v: self._rsi(p, 14),
            "macd": lambda p, v: self._macd(p)[0],
            "macd_signal": lambda p, v: self._macd(p)[1],
            "macd_histogram": lambda p, v: self._macd(p)[2],
            "bb_upper": lambda p, v: self._bollinger_bands(p)[0],
            "bb_middle": lambda p, v: self._bollinger_bands(p)[1],
            "bb_lower": lambda p, v: self._bollinger_bands(p)[2],
            "atr": lambda p, v: self._atr(p),
            "stoch_k": lambda p, v: self._stochastic(p)[0],
            "stoch_d": lambda p, v: self._stochastic(p)[1],
            "obv": lambda p, v: self._obv(p, v),
            "vwap": lambda p, v: self._vwap(p, v),
            "williams_r": lambda p, v: self._williams_r(p),
            "cci": lambda p, v: self._cci(p),
            "momentum": lambda p, v: self._momentum(p),
            "roc": lambda p, v: self._rate_of_change(p)
        }
        
        self.logger.info(f"Technical indicators initialized: {indicators}")
    
    def calculate(
        self,
        prices: List[float],
        volumes: Optional[List[float]] = None,
        high: Optional[List[float]] = None,
        low: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Calculate all configured indicators
        
        Args:
            prices: Price series (typically close prices)
            volumes: Volume series (optional)
            high: High prices for range-based indicators (optional)
            low: Low prices for range-based indicators (optional)
            
        Returns:
            Dictionary of indicator values
        """
        
        if len(prices) < 2:
            return {indicator: 0.0 for indicator in self.indicators}
        
        # Update history
        self.price_history.extend(prices[-50:])  # Keep recent history
        if volumes:
            self.volume_history.extend(volumes[-50:])
        
        # Use provided data or history
        price_series = prices if len(prices) >= 50 else list(self.price_history)
        volume_series = volumes if volumes else list(self.volume_history)
        
        if not volume_series:
            volume_series = [1.0] * len(price_series)  # Default volume
        
        results = {}
        
        try:
            for indicator in self.indicators:
                if indicator in self.indicator_functions:
                    # Check cache first
                    cache_key = f"{indicator}_{len(price_series)}_{hash(tuple(price_series[-10:]))}"
                    
                    if self.cache and cache_key in self.cache:
                        results[indicator] = self.cache[cache_key]
                    else:
                        # Calculate indicator
                        value = self.indicator_functions[indicator](price_series, volume_series)
                        results[indicator] = float(value) if not np.isnan(value) else 0.0
                        
                        # Cache result
                        if self.cache:
                            self.cache[cache_key] = results[indicator]
                            
                            # Limit cache size
                            if len(self.cache) > self.config.cache_size:
                                # Remove oldest entries
                                keys_to_remove = list(self.cache.keys())[:100]
                                for key in keys_to_remove:
                                    del self.cache[key]
                else:
                    self.logger.warning(f"Unknown indicator: {indicator}")
                    results[indicator] = 0.0
                    
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}", exc_info=True)
            results = {indicator: 0.0 for indicator in self.indicators}
        
        return results
    
    def _sma(self, prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        return float(np.mean(prices[-period:]))
    
    def _ema(self, prices: List[float], period: int, alpha: Optional[float] = None) -> float:
        """Exponential Moving Average"""
        if len(prices) < 2:
            return prices[-1] if prices else 0.0
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        # Calculate EMA recursively
        prices_array = np.array(prices)
        ema_values = np.zeros_like(prices_array)
        ema_values[0] = prices_array[0]
        
        for i in range(1, len(prices_array)):
            ema_values[i] = alpha * prices_array[i] + (1 - alpha) * ema_values[i-1]
        
        return float(ema_values[-1])
    
    def _rsi(self, prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        # Calculate price changes
        changes = np.diff(prices)
        
        # Separate gains and losses
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        
        # Calculate average gains and losses
        if len(gains) >= period:
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
        else:
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < max(self.config.macd_slow, self.config.macd_signal):
            return 0.0, 0.0, 0.0
        
        # Calculate EMAs
        ema_fast = self._ema(prices, self.config.macd_fast)
        ema_slow = self._ema(prices, self.config.macd_slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line) - simplified calculation
        signal_line = macd_line * 0.8  # Approximation for real-time calculation
        
        # Histogram
        histogram = macd_line - signal_line
        
        return float(macd_line), float(signal_line), float(histogram)
    
    def _bollinger_bands(self, prices: List[float]) -> Tuple[float, float, float]:
        """Bollinger Bands"""
        period = self.config.bb_period
        std_dev = self.config.bb_std
        
        if len(prices) < period:
            current_price = prices[-1] if prices else 0.0
            return current_price, current_price, current_price
        
        # Calculate middle band (SMA)
        middle = self._sma(prices, period)
        
        # Calculate standard deviation
        recent_prices = prices[-period:]
        std = float(np.std(recent_prices))
        
        # Calculate bands
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return float(upper), float(middle), float(lower)
    
    def _atr(self, prices: List[float], high: Optional[List[float]] = None, 
           low: Optional[List[float]] = None) -> float:
        """Average True Range"""
        period = self.config.atr_period
        
        if len(prices) < period:
            return 0.0
        
        # If high/low not provided, use close prices
        if high is None or low is None:
            # Estimate high/low from prices (simplified)
            volatility = np.std(prices[-period:]) if len(prices) >= period else 0.01
            return float(volatility * 2.0)  # Approximation
        
        # Calculate true range
        true_ranges = []
        for i in range(1, min(len(prices), len(high), len(low))):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - prices[i-1])
            tr3 = abs(low[i] - prices[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return float(np.mean(true_ranges)) if true_ranges else 0.0
        
        return float(np.mean(true_ranges[-period:]))
    
    def _stochastic(self, prices: List[float]) -> Tuple[float, float]:
        """Stochastic Oscillator"""
        k_period = self.config.stoch_k_period
        d_period = self.config.stoch_d_period
        
        if len(prices) < k_period:
            return 50.0, 50.0
        
        recent_prices = prices[-k_period:]
        highest_high = max(recent_prices)
        lowest_low = min(recent_prices)
        
        if highest_high == lowest_low:
            stoch_k = 50.0
        else:
            stoch_k = ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D is SMA of %K (simplified)
        stoch_d = stoch_k * 0.8  # Approximation
        
        return float(stoch_k), float(stoch_d)
    
    def _obv(self, prices: List[float], volumes: List[float]) -> float:
        """On-Balance Volume"""
        if len(prices) < 2 or len(volumes) < 2:
            return 0.0
        
        obv = 0.0
        for i in range(1, min(len(prices), len(volumes))):
            if prices[i] > prices[i-1]:
                obv += volumes[i]
            elif prices[i] < prices[i-1]:
                obv -= volumes[i]
        
        return float(obv)
    
    def _vwap(self, prices: List[float], volumes: List[float]) -> float:
        """Volume Weighted Average Price"""
        period = min(self.config.vwap_period, len(prices), len(volumes))
        
        if period < 2:
            return prices[-1] if prices else 0.0
        
        recent_prices = prices[-period:]
        recent_volumes = volumes[-period:]
        
        if sum(recent_volumes) == 0:
            return float(np.mean(recent_prices))
        
        # Calculate VWAP
        price_volume = sum(p * v for p, v in zip(recent_prices, recent_volumes))
        total_volume = sum(recent_volumes)
        
        return float(price_volume / total_volume)
    
    def _williams_r(self, prices: List[float], period: int = 14) -> float:
        """Williams %R"""
        if len(prices) < period:
            return -50.0
        
        recent_prices = prices[-period:]
        highest_high = max(recent_prices)
        lowest_low = min(recent_prices)
        
        if highest_high == lowest_low:
            return -50.0
        
        williams_r = ((highest_high - prices[-1]) / (highest_high - lowest_low)) * -100
        
        return float(williams_r)
    
    def _cci(self, prices: List[float], period: int = 20) -> float:
        """Commodity Channel Index"""
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        
        # Calculate mean deviation
        mad = np.mean([abs(p - sma) for p in recent_prices])
        
        if mad == 0:
            return 0.0
        
        cci = (prices[-1] - sma) / (0.015 * mad)
        
        return float(cci)
    
    def _momentum(self, prices: List[float], period: int = 10) -> float:
        """Price Momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        return float((prices[-1] - prices[-(period + 1)]) / prices[-(period + 1)] * 100)
    
    def _rate_of_change(self, prices: List[float], period: int = 12) -> float:
        """Rate of Change"""
        if len(prices) < period + 1:
            return 0.0
        
        return float((prices[-1] - prices[-(period + 1)]) / prices[-(period + 1)] * 100)
    
    def get_indicator_info(self) -> Dict[str, Any]:
        """Get information about available indicators"""
        
        return {
            "configured_indicators": self.indicators,
            "available_indicators": list(self.indicator_functions.keys()),
            "config": self.config.__dict__,
            "cache_size": len(self.cache) if self.cache else 0,
            "price_history_length": len(self.price_history),
            "volume_history_length": len(self.volume_history)
        }
    
    def clear_cache(self) -> None:
        """Clear indicator cache"""
        if self.cache:
            self.cache.clear()
    
    def reset(self) -> None:
        """Reset indicator state"""
        self.price_history.clear()
        self.volume_history.clear()
        self.clear_cache()


# Convenience functions
def calculate_sma(prices: List[float], period: int) -> float:
    """Calculate Simple Moving Average"""
    calc = TechnicalIndicators([f"sma_{period}"])
    result = calc.calculate(prices)
    return result.get(f"sma_{period}", 0.0)


def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate RSI"""
    calc = TechnicalIndicators([f"rsi_{period}"])
    result = calc.calculate(prices)
    return result.get(f"rsi_{period}", 50.0)


def calculate_macd(prices: List[float]) -> Tuple[float, float, float]:
    """Calculate MACD"""
    calc = TechnicalIndicators(["macd", "macd_signal", "macd_histogram"])
    result = calc.calculate(prices)
    return (
        result.get("macd", 0.0),
        result.get("macd_signal", 0.0), 
        result.get("macd_histogram", 0.0)
    )


__all__ = [
    "IndicatorConfig",
    "TechnicalIndicators",
    "calculate_sma",
    "calculate_rsi", 
    "calculate_macd"
]