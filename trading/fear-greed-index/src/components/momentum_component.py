"""
Momentum Component for Fear & Greed Index

Measures price momentum to determine the direction of market sentiment.
Positive momentum indicates greed, negative momentum indicates fear.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class MomentumType(Enum):
    """Types of momentum indicators"""
    RSI = "rsi"                # Relative Strength Index
    MACD = "macd"             # Moving Average Convergence Divergence
    ROC = "roc"               # Rate of Change
    WILLIAMS_R = "williams_r"  # Williams %R
    STOCH_RSI = "stoch_rsi"   # Stochastic RSI
    CCI = "cci"               # Commodity Channel Index


@dataclass
class MomentumData:
    """Momentum data"""
    timestamp: datetime
    symbol: str
    rsi_14: float           # RSI 14 periods
    rsi_7: float            # RSI 7 periods
    macd_line: float        # MACD line
    macd_signal: float      # MACD signal
    macd_histogram: float   # MACD histogram
    roc_1d: float          # Rate of change 1 day
    roc_7d: float          # Rate of change 7 days
    roc_30d: float         # Rate of change 30 days
    williams_r: float       # Williams %R
    stoch_rsi: float       # Stochastic RSI
    cci: float             # Commodity Channel Index
    momentum_score: float   # Combined score 0-100
    fear_greed_score: float # Fear & Greed score 0-100
    confidence: float       # Confidence 0-1


class MomentumComponent:
    """
    Momentum measurement component for Fear & Greed Index

    Uses multiple technical indicators to assess
    the strength and direction of price movement.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("momentum")
        self._momentum_cache: Dict[str, MomentumData] = {}

        # Indicator parameters
        self.rsi_periods = [7, 14, 21]
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.roc_periods = [1, 7, 30]  # days
        self.williams_period = 14
        self.cci_period = 20

        # Normalization bounds
        self.momentum_bounds = {
            "rsi": {"oversold": 30, "overbought": 70},
            "williams_r": {"oversold": -80, "overbought": -20},
            "cci": {"oversold": -100, "overbought": 100},
            "roc": {"extreme_negative": -50, "extreme_positive": 50}
        }

        # Component weights for final score
        self.component_weights = {
            "rsi": 0.3,
            "macd": 0.25,
            "roc": 0.2,
            "williams_r": 0.1,
            "stoch_rsi": 0.1,
            "cci": 0.05
        }

        logger.info("MomentumComponent initialized",
                   config=config.dict(), weights=self.component_weights)

    async def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Closing price series
            period: Calculation period

        Returns:
            Series of RSI values
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Exponential moving average
        avg_gain = gain.ewm(span=period).mean()
        avg_loss = loss.ewm(span=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator

        Args:
            prices: Closing price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple[macd_line, signal_line, histogram]
        """
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    async def calculate_rate_of_change(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Rate of Change (ROC)

        Args:
            prices: Closing price series
            period: Calculation period

        Returns:
            Series of ROC values in percentage
        """
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc

    async def calculate_williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R

        Args:
            high: High price series
            low: Low price series
            close: Closing price series
            period: Calculation period

        Returns:
            Series of Williams %R values
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r

    async def calculate_stochastic_rsi(
        self,
        prices: pd.Series,
        rsi_period: int = 14,
        stoch_period: int = 14
    ) -> pd.Series:
        """
        Calculate Stochastic RSI

        Args:
            prices: Closing price series
            rsi_period: RSI period
            stoch_period: Stochastic period

        Returns:
            Series of Stochastic RSI values
        """
        rsi = await self.calculate_rsi(prices, rsi_period)

        lowest_rsi = rsi.rolling(window=stoch_period).min()
        highest_rsi = rsi.rolling(window=stoch_period).max()

        stoch_rsi = (rsi - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
        return stoch_rsi

    async def calculate_cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI)

        Args:
            high: High price series
            low: Low price series
            close: Closing price series
            period: Calculation period

        Returns:
            Series of CCI values
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()

        # Mean absolute deviation
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        cci = (typical_price - sma) / (0.015 * mad)
        return cci

    def normalize_momentum_indicators(
        self,
        indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Normalize momentum indicators to 0-100 scale

        Args:
            indicators: Dict with indicator values

        Returns:
            Dict with normalized values
        """
        normalized = {}

        # RSI is already in 0-100 range
        normalized['rsi'] = max(0, min(100, indicators.get('rsi_14', 50)))

        # MACD histogram - normalize relative to historical values
        macd_hist = indicators.get('macd_histogram', 0)
        if macd_hist > 0:
            normalized['macd'] = 50 + min(50, abs(macd_hist) * 10)  # Bullish
        else:
            normalized['macd'] = 50 - min(50, abs(macd_hist) * 10)  # Bearish

        # ROC - normalize in -50% to +50% range
        roc_30d = indicators.get('roc_30d', 0)
        roc_normalized = 50 + (roc_30d / 100) * 50  # -50% ROC = 0, +50% ROC = 100
        normalized['roc'] = max(0, min(100, roc_normalized))

        # Williams %R is inverted (-100 to 0)
        williams = indicators.get('williams_r', -50)
        normalized['williams_r'] = 100 + williams  # Convert to 0-100

        # Stochastic RSI is already in 0-100 range
        normalized['stoch_rsi'] = max(0, min(100, indicators.get('stoch_rsi', 50)))

        # CCI normalization (-200 to +200 typical range)
        cci = indicators.get('cci', 0)
        cci_normalized = 50 + (cci / 400) * 100  # -200 CCI = 0, +200 CCI = 100
        normalized['cci'] = max(0, min(100, cci_normalized))

        return normalized

    def calculate_momentum_score(
        self,
        normalized_indicators: Dict[str, float]
    ) -> float:
        """
        Calculate combined momentum score

        Args:
            normalized_indicators: Normalized indicators

        Returns:
            Weighted momentum score (0-100)
        """
        total_score = 0.0
        total_weight = 0.0

        for indicator, weight in self.component_weights.items():
            if indicator in normalized_indicators:
                total_score += normalized_indicators[indicator] * weight
                total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 50.0  # Neutral if no indicators available

    async def get_fear_greed_component(
        self,
        symbol: str,
        price_data: pd.DataFrame
    ) -> MomentumData:
        """
        Get momentum component for Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol
            price_data: OHLCV data

        Returns:
            MomentumData object with results
        """
        try:
            self.validator.validate_price_data(price_data)

            if len(price_data) < 50:  # Minimum data for calculation
                raise ValueError(f"Insufficient data for momentum calculation: {len(price_data)} rows")

            close = price_data['close']
            high = price_data['high']
            low = price_data['low']

            # Calculate all indicators
            rsi_14 = await self.calculate_rsi(close, 14)
            rsi_7 = await self.calculate_rsi(close, 7)

            macd_line, macd_signal, macd_histogram = await self.calculate_macd(
                close, self.macd_fast, self.macd_slow, self.macd_signal
            )

            roc_1d = await self.calculate_rate_of_change(close, 1)
            roc_7d = await self.calculate_rate_of_change(close, 7)
            roc_30d = await self.calculate_rate_of_change(close, 30)

            williams_r = await self.calculate_williams_r(high, low, close, self.williams_period)
            stoch_rsi = await self.calculate_stochastic_rsi(close)
            cci = await self.calculate_cci(high, low, close, self.cci_period)

            # Get latest values
            current_indicators = {
                'rsi_14': float(rsi_14.iloc[-1]) if not pd.isna(rsi_14.iloc[-1]) else 50,
                'rsi_7': float(rsi_7.iloc[-1]) if not pd.isna(rsi_7.iloc[-1]) else 50,
                'macd_line': float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0,
                'macd_signal': float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0,
                'macd_histogram': float(macd_histogram.iloc[-1]) if not pd.isna(macd_histogram.iloc[-1]) else 0,
                'roc_1d': float(roc_1d.iloc[-1]) if not pd.isna(roc_1d.iloc[-1]) else 0,
                'roc_7d': float(roc_7d.iloc[-1]) if not pd.isna(roc_7d.iloc[-1]) else 0,
                'roc_30d': float(roc_30d.iloc[-1]) if not pd.isna(roc_30d.iloc[-1]) else 0,
                'williams_r': float(williams_r.iloc[-1]) if not pd.isna(williams_r.iloc[-1]) else -50,
                'stoch_rsi': float(stoch_rsi.iloc[-1]) if not pd.isna(stoch_rsi.iloc[-1]) else 50,
                'cci': float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0
            }

            # Normalize indicators
            normalized_indicators = self.normalize_momentum_indicators(current_indicators)

            # Calculate combined momentum score
            momentum_score = self.calculate_momentum_score(normalized_indicators)

            # Fear & Greed score (momentum score is already in 0-100 range)
            fear_greed_score = momentum_score

            # Confidence based on data availability and signal quality
            data_completeness = len(price_data) / (50 * 24)  # 50 days of hourly data
            indicator_quality = 1.0 - (sum(1 for v in current_indicators.values()
                                           if pd.isna(v) or v == 0) / len(current_indicators))
            confidence = min(1.0, (data_completeness + indicator_quality) / 2)

            result = MomentumData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                rsi_14=current_indicators['rsi_14'],
                rsi_7=current_indicators['rsi_7'],
                macd_line=current_indicators['macd_line'],
                macd_signal=current_indicators['macd_signal'],
                macd_histogram=current_indicators['macd_histogram'],
                roc_1d=current_indicators['roc_1d'],
                roc_7d=current_indicators['roc_7d'],
                roc_30d=current_indicators['roc_30d'],
                williams_r=current_indicators['williams_r'],
                stoch_rsi=current_indicators['stoch_rsi'],
                cci=current_indicators['cci'],
                momentum_score=momentum_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            # Caching
            self._momentum_cache[symbol] = result

            logger.info(
                "Momentum component calculated",
                symbol=symbol,
                rsi_14=current_indicators['rsi_14'],
                momentum_score=momentum_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating momentum component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_momentum(self, symbol: str) -> Optional[MomentumData]:
        """Get cached momentum data"""
        return self._momentum_cache.get(symbol)

    async def batch_calculate(
        self,
        symbols_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, MomentumData]:
        """
        Batch momentum calculation for multiple symbols

        Args:
            symbols_data: Dict with symbols and their price_data

        Returns:
            Dict with results for each symbol
        """
        tasks = []

        for symbol, price_data in symbols_data.items():
            task = self.get_fear_greed_component(symbol, price_data)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        momentum_results = {}
        for symbol, result in zip(symbols_data.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating momentum for {symbol}", error=str(result))
                continue
            momentum_results[symbol] = result

        return momentum_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
