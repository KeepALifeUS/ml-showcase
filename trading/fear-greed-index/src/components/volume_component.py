"""
Volume Component for Fear & Greed Index

Analyzes trading volumes to determine the strength of market sentiment.
High volumes during price increase = greed, high volumes during price decrease = fear.
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


class VolumeIndicator(Enum):
    """Types of volume indicators"""
    OBV = "obv"                    # On-Balance Volume
    AD_LINE = "ad_line"            # Accumulation/Distribution Line
    MFI = "mfi"                    # Money Flow Index
    VWAP = "vwap"                  # Volume Weighted Average Price
    VOLUME_RSI = "volume_rsi"      # Volume RSI
    VOLUME_SMA = "volume_sma"      # Volume Simple Moving Average


@dataclass
class VolumeData:
    """Volume analysis data"""
    timestamp: datetime
    symbol: str
    current_volume: float       # Current volume
    volume_sma_7d: float       # 7-day volume SMA
    volume_sma_30d: float      # 30-day volume SMA
    volume_ratio_7d: float     # Ratio to 7-day SMA
    volume_ratio_30d: float    # Ratio to 30-day SMA
    obv: float                 # On-Balance Volume
    obv_sma: float            # OBV Simple Moving Average
    ad_line: float            # Accumulation/Distribution Line
    mfi: float                # Money Flow Index
    vwap: float               # Volume Weighted Average Price
    volume_rsi: float         # Volume RSI
    price_volume_trend: float  # Price Volume Trend
    volume_score: float       # Volume score 0-100
    fear_greed_score: float   # Fear & Greed score 0-100
    confidence: float         # Confidence 0-1


class VolumeComponent:
    """
    Volume analysis component for Fear & Greed Index

    Uses various volume indicators to assess the strength
    and direction of market movement.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("volume")
        self._volume_cache: Dict[str, VolumeData] = {}

        # Indicator parameters
        self.obv_sma_period = 10
        self.mfi_period = 14
        self.volume_rsi_period = 14
        self.vwap_period = 20
        self.volume_sma_periods = [7, 30]

        # Interpretation thresholds
        self.volume_thresholds = {
            "high_volume_multiplier": 2.0,    # High volume = 2x of average
            "low_volume_multiplier": 0.5,     # Low volume = 0.5x of average
            "mfi_overbought": 80,
            "mfi_oversold": 20,
            "volume_rsi_overbought": 70,
            "volume_rsi_oversold": 30
        }

        # Component weights for the final score
        self.component_weights = {
            "volume_ratio": 0.25,    # Current volume to average ratio
            "obv_trend": 0.20,       # OBV trend
            "mfi": 0.20,            # Money Flow Index
            "ad_line": 0.15,        # A/D Line trend
            "volume_rsi": 0.10,     # Volume RSI
            "price_volume_confirm": 0.10  # Price-volume confirmation
        }

        logger.info("VolumeComponent initialized",
                   config=config.dict(), thresholds=self.volume_thresholds)

    async def calculate_obv(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            Series of OBV values
        """
        close = price_data['close']
        volume = price_data['volume']

        # OBV accumulates depending on price direction
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    async def calculate_ad_line(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            Series of A/D Line values
        """
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        volume = price_data['volume']

        # Money Flow Multiplier
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)  # Fill NaN with zeros (when high = low)

        # Money Flow Volume
        mfv = clv * volume

        # A/D Line - cumulative sum of MFV
        ad_line = mfv.cumsum()

        return ad_line

    async def calculate_mfi(
        self,
        price_data: pd.DataFrame,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate Money Flow Index (MFI)

        Args:
            price_data: DataFrame with OHLCV data
            period: Calculation period

        Returns:
            Series of MFI values
        """
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        volume = price_data['volume']

        # Typical Price
        typical_price = (high + low + close) / 3

        # Raw Money Flow
        money_flow = typical_price * volume

        # Positive and Negative Money Flow
        price_change = typical_price.diff()
        positive_mf = money_flow.where(price_change > 0, 0)
        negative_mf = money_flow.where(price_change < 0, 0)

        # Money Flow Ratio over period
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()

        money_flow_ratio = positive_mf_sum / negative_mf_sum
        money_flow_ratio = money_flow_ratio.replace([np.inf, -np.inf], 0)

        # Money Flow Index
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return mfi

    async def calculate_vwap(
        self,
        price_data: pd.DataFrame,
        period: int = 20
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)

        Args:
            price_data: DataFrame with OHLCV data
            period: Calculation period (None for cumulative VWAP)

        Returns:
            Series of VWAP values
        """
        typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
        volume = price_data['volume']

        if period is not None:
            # Rolling VWAP
            pv = (typical_price * volume).rolling(window=period).sum()
            v = volume.rolling(window=period).sum()
        else:
            # Cumulative VWAP
            pv = (typical_price * volume).cumsum()
            v = volume.cumsum()

        vwap = pv / v
        return vwap

    async def calculate_volume_rsi(
        self,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Calculate RSI based on volumes

        Args:
            volume: Volume series
            period: RSI period

        Returns:
            Series of Volume RSI values
        """
        # Volume changes
        volume_change = volume.diff()

        # Positive and negative changes
        volume_up = volume_change.where(volume_change > 0, 0)
        volume_down = -volume_change.where(volume_change < 0, 0)

        # Exponential moving averages
        avg_volume_up = volume_up.ewm(span=period).mean()
        avg_volume_down = volume_down.ewm(span=period).mean()

        # Volume RSI
        rs = avg_volume_up / avg_volume_down
        volume_rsi = 100 - (100 / (1 + rs))

        return volume_rsi

    async def calculate_price_volume_trend(
        self,
        price_data: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate Price Volume Trend (PVT)

        Args:
            price_data: DataFrame with OHLCV data

        Returns:
            Series of PVT values
        """
        close = price_data['close']
        volume = price_data['volume']

        # Percentage price change
        price_change_pct = close.pct_change()

        # PVT accumulates volume weighted by percentage price change
        pvt = (price_change_pct * volume).cumsum()

        return pvt

    def analyze_volume_price_relationship(
        self,
        price_data: pd.DataFrame,
        lookback_periods: int = 10
    ) -> float:
        """
        Analyze the relationship between price and volume

        Args:
            price_data: DataFrame with OHLCV data
            lookback_periods: Number of periods for analysis

        Returns:
            Score reflecting the strength of the price-volume relationship (0-100)
        """
        if len(price_data) < lookback_periods:
            return 50.0  # Neutral if insufficient data

        recent_data = price_data.tail(lookback_periods)

        price_change = recent_data['close'].pct_change()
        volume_change = recent_data['volume'].pct_change()

        # Count confirmation cases
        confirmations = 0
        total_signals = 0

        for i in range(1, len(recent_data)):
            price_dir = 1 if price_change.iloc[i] > 0 else -1 if price_change.iloc[i] < 0 else 0
            volume_dir = 1 if volume_change.iloc[i] > 0 else -1 if volume_change.iloc[i] < 0 else 0

            if price_dir != 0 and volume_dir != 0:
                total_signals += 1
                # Confirmation: price rise with volume rise or price drop with volume rise
                if (price_dir > 0 and volume_dir > 0) or (price_dir < 0 and volume_dir > 0):
                    confirmations += 1

        if total_signals == 0:
            return 50.0  # Neutral

        confirmation_ratio = confirmations / total_signals
        return confirmation_ratio * 100

    def normalize_volume_indicators(
        self,
        indicators: Dict[str, float],
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Normalize volume indicators to 0-100 scale

        Args:
            indicators: Dict with indicator values
            price_data: Source data for context

        Returns:
            Dict with normalized values
        """
        normalized = {}

        # Volume ratio normalization (logarithmic)
        volume_ratio_30d = indicators.get('volume_ratio_30d', 1.0)
        if volume_ratio_30d > 0:
            # Use logarithmic scaling for volume ratio
            log_ratio = np.log(volume_ratio_30d)
            # Normalize to 0-100 range, where 1.0 ratio = 50
            volume_score = 50 + (log_ratio / np.log(4)) * 50  # log(4) ~ maximum
            normalized['volume_ratio'] = max(0, min(100, volume_score))
        else:
            normalized['volume_ratio'] = 0

        # OBV trend - compare with OBV SMA
        obv_current = indicators.get('obv', 0)
        obv_sma = indicators.get('obv_sma', 0)
        if obv_sma != 0:
            obv_ratio = obv_current / obv_sma
            obv_score = 50 + (obv_ratio - 1) * 50
            normalized['obv_trend'] = max(0, min(100, obv_score))
        else:
            normalized['obv_trend'] = 50

        # MFI is already in 0-100 range
        normalized['mfi'] = max(0, min(100, indicators.get('mfi', 50)))

        # A/D Line trend - analyze recent values
        if len(price_data) >= 10:
            ad_values = price_data.tail(10)  # Get last 10 A/D values
            if not ad_values.empty and 'ad_line' in ad_values.columns:
                ad_trend = ad_values['ad_line'].iloc[-1] - ad_values['ad_line'].iloc[0]
                # Normalize A/D Line trend
                ad_score = 50 + np.sign(ad_trend) * min(50, abs(ad_trend) / ad_values['ad_line'].mean() * 100)
                normalized['ad_line'] = max(0, min(100, ad_score))
            else:
                normalized['ad_line'] = 50
        else:
            normalized['ad_line'] = 50

        # Volume RSI is already in 0-100 range
        normalized['volume_rsi'] = max(0, min(100, indicators.get('volume_rsi', 50)))

        # Price-Volume confirmation
        normalized['price_volume_confirm'] = indicators.get('price_volume_confirmation', 50)

        return normalized

    def calculate_volume_score(
        self,
        normalized_indicators: Dict[str, float]
    ) -> float:
        """
        Calculate combined volume score

        Args:
            normalized_indicators: Normalized indicators

        Returns:
            Weighted volume score (0-100)
        """
        total_score = 0.0
        total_weight = 0.0

        for component, weight in self.component_weights.items():
            if component in normalized_indicators:
                total_score += normalized_indicators[component] * weight
                total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 50.0  # Neutral if no indicators available

    async def get_fear_greed_component(
        self,
        symbol: str,
        price_data: pd.DataFrame
    ) -> VolumeData:
        """
        Get volume component for Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol
            price_data: OHLCV data

        Returns:
            VolumeData object with results
        """
        try:
            self.validator.validate_price_data(price_data)

            if len(price_data) < 30:  # Minimum data required
                raise ValueError(f"Insufficient data for volume calculation: {len(price_data)} rows")

            volume = price_data['volume']

            # Calculate all volume indicators
            obv = await self.calculate_obv(price_data)
            obv_sma = obv.rolling(window=self.obv_sma_period).mean()

            ad_line = await self.calculate_ad_line(price_data)
            mfi = await self.calculate_mfi(price_data, self.mfi_period)
            vwap = await self.calculate_vwap(price_data, self.vwap_period)
            volume_rsi = await self.calculate_volume_rsi(volume, self.volume_rsi_period)
            pvt = await self.calculate_price_volume_trend(price_data)

            # Volume SMA for different periods
            volume_sma_7d = volume.rolling(window=7*24).mean()  # 7 days (hourly data)
            volume_sma_30d = volume.rolling(window=30*24).mean()  # 30 days

            # Current values
            current_volume = float(volume.iloc[-1])
            current_volume_sma_7d = float(volume_sma_7d.iloc[-1]) if not pd.isna(volume_sma_7d.iloc[-1]) else current_volume
            current_volume_sma_30d = float(volume_sma_30d.iloc[-1]) if not pd.isna(volume_sma_30d.iloc[-1]) else current_volume

            # Volume ratios
            volume_ratio_7d = current_volume / current_volume_sma_7d if current_volume_sma_7d > 0 else 1.0
            volume_ratio_30d = current_volume / current_volume_sma_30d if current_volume_sma_30d > 0 else 1.0

            # Analyze price-volume relationship
            price_volume_confirmation = self.analyze_volume_price_relationship(price_data)

            # Current indicator values
            current_indicators = {
                'volume_ratio_7d': volume_ratio_7d,
                'volume_ratio_30d': volume_ratio_30d,
                'obv': float(obv.iloc[-1]) if not pd.isna(obv.iloc[-1]) else 0,
                'obv_sma': float(obv_sma.iloc[-1]) if not pd.isna(obv_sma.iloc[-1]) else 0,
                'ad_line': float(ad_line.iloc[-1]) if not pd.isna(ad_line.iloc[-1]) else 0,
                'mfi': float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50,
                'vwap': float(vwap.iloc[-1]) if not pd.isna(vwap.iloc[-1]) else 0,
                'volume_rsi': float(volume_rsi.iloc[-1]) if not pd.isna(volume_rsi.iloc[-1]) else 50,
                'pvt': float(pvt.iloc[-1]) if not pd.isna(pvt.iloc[-1]) else 0,
                'price_volume_confirmation': price_volume_confirmation
            }

            # Normalize indicators
            normalized_indicators = self.normalize_volume_indicators(current_indicators, price_data)

            # Calculate combined volume score
            volume_score = self.calculate_volume_score(normalized_indicators)

            # Fear & Greed score (volume score is already in 0-100 range)
            fear_greed_score = volume_score

            # Confidence based on data quality
            data_quality = 1.0 - (sum(1 for v in current_indicators.values()
                                     if pd.isna(v) or (isinstance(v, float) and v == 0)) / len(current_indicators))
            data_completeness = min(1.0, len(price_data) / (30 * 24))  # 30 days of data
            confidence = (data_quality + data_completeness) / 2

            result = VolumeData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                current_volume=current_volume,
                volume_sma_7d=current_volume_sma_7d,
                volume_sma_30d=current_volume_sma_30d,
                volume_ratio_7d=volume_ratio_7d,
                volume_ratio_30d=volume_ratio_30d,
                obv=current_indicators['obv'],
                obv_sma=current_indicators['obv_sma'],
                ad_line=current_indicators['ad_line'],
                mfi=current_indicators['mfi'],
                vwap=current_indicators['vwap'],
                volume_rsi=current_indicators['volume_rsi'],
                price_volume_trend=current_indicators['pvt'],
                volume_score=volume_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            # Caching
            self._volume_cache[symbol] = result

            logger.info(
                "Volume component calculated",
                symbol=symbol,
                volume_ratio_30d=volume_ratio_30d,
                mfi=current_indicators['mfi'],
                volume_score=volume_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating volume component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_volume(self, symbol: str) -> Optional[VolumeData]:
        """Get cached volume data"""
        return self._volume_cache.get(symbol)

    async def batch_calculate(
        self,
        symbols_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, VolumeData]:
        """
        Batch volume calculation for multiple symbols

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

        volume_results = {}
        for symbol, result in zip(symbols_data.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating volume for {symbol}", error=str(result))
                continue
            volume_results[symbol] = result

        return volume_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
