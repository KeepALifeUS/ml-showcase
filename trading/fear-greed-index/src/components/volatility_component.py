"""
Volatility Component for Fear & Greed Index

Measures market volatility to determine the level of fear/greed.
High volatility usually indicates fear, low volatility indicates complacency.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class VolatilityType(Enum):
    """Types of volatility for calculation"""
    REALIZED = "realized"  # Historical volatility
    IMPLIED = "implied"    # Implied volatility
    GARCH = "garch"       # GARCH model
    PARKINSON = "parkinson"  # Parkinson estimator


@dataclass
class VolatilityData:
    """Volatility data"""
    timestamp: datetime
    symbol: str
    volatility_1d: float
    volatility_7d: float
    volatility_30d: float
    volatility_90d: float
    normalized_score: float  # 0-100
    fear_greed_score: float  # 0-100
    confidence: float        # 0-1


class VolatilityComponent:
    """
    Volatility measurement component for Fear & Greed Index

    Uses multiple timeframes and various volatility calculation methods
    for a more accurate assessment of market sentiment.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("volatility")
        self._volatility_cache: Dict[str, VolatilityData] = {}

        # Volatility parameters
        self.timeframes = {
            "1d": 24,      # hours
            "7d": 168,     # hours
            "30d": 720,    # hours
            "90d": 2160    # hours
        }

        # Normalization bounds (annualized volatility in %)
        self.volatility_bounds = {
            "extreme_fear": 150,    # > 150% - extreme fear
            "fear": 100,           # 80-150% - fear
            "neutral": 50,         # 30-80% - neutral
            "greed": 30,          # 15-30% - greed
            "extreme_greed": 15    # < 15% - extreme greed
        }

        logger.info("VolatilityComponent initialized",
                   config=config.dict(), bounds=self.volatility_bounds)

    async def calculate_volatility(
        self,
        price_data: pd.DataFrame,
        method: VolatilityType = VolatilityType.REALIZED
    ) -> Dict[str, float]:
        """
        Calculate volatility using various methods

        Args:
            price_data: DataFrame with OHLCV data
            method: Volatility calculation method

        Returns:
            Dict with volatility for different timeframes
        """
        try:
            self.validator.validate_price_data(price_data)

            volatilities = {}

            if method == VolatilityType.REALIZED:
                volatilities = await self._calculate_realized_volatility(price_data)
            elif method == VolatilityType.PARKINSON:
                volatilities = await self._calculate_parkinson_volatility(price_data)
            elif method == VolatilityType.GARCH:
                volatilities = await self._calculate_garch_volatility(price_data)

            self.metrics.record_calculation("volatility_calculation", 1)

            return volatilities

        except Exception as e:
            logger.error("Error calculating volatility", error=str(e), method=method)
            self.metrics.record_error("volatility_calculation_error")
            raise

    async def _calculate_realized_volatility(
        self,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate historical (realized) volatility"""

        # Calculate logarithmic returns
        price_data = price_data.copy()
        price_data['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))

        volatilities = {}

        for timeframe, hours in self.timeframes.items():
            # Take data for the required period
            data_subset = price_data.tail(hours)

            if len(data_subset) < 10:  # Minimum data required
                volatilities[timeframe] = 0.0
                continue

            # Standard deviation of logarithmic returns
            returns_std = data_subset['log_returns'].std()

            # Annualization (assuming 24 hours per day, 365 days per year)
            annualized_vol = returns_std * np.sqrt(24 * 365) * 100

            volatilities[timeframe] = float(annualized_vol)

        return volatilities

    async def _calculate_parkinson_volatility(
        self,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate volatility using the Parkinson method (high-low)
        A more efficient estimator that uses high and low prices
        """

        volatilities = {}

        for timeframe, hours in self.timeframes.items():
            data_subset = price_data.tail(hours)

            if len(data_subset) < 10:
                volatilities[timeframe] = 0.0
                continue

            # Parkinson estimator
            hl_ratio = np.log(data_subset['high'] / data_subset['low'])
            parkinson_var = (hl_ratio ** 2).mean() / (4 * np.log(2))

            # Annualization
            annualized_vol = np.sqrt(parkinson_var * 24 * 365) * 100

            volatilities[timeframe] = float(annualized_vol)

        return volatilities

    async def _calculate_garch_volatility(
        self,
        price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        GARCH model for volatility forecasting
        Uses conditional heteroscedasticity
        """
        try:
            from arch import arch_model
        except ImportError:
            logger.warning("ARCH library not available, falling back to realized volatility")
            return await self._calculate_realized_volatility(price_data)

        price_data = price_data.copy()
        price_data['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
        price_data = price_data.dropna()

        volatilities = {}

        try:
            # GARCH(1,1) model
            model = arch_model(
                price_data['log_returns'] * 100,  # Scaling for better convergence
                vol='Garch',
                p=1,
                q=1
            )

            res = model.fit(disp='off')

            # Volatility forecast for different horizons
            for timeframe, hours in self.timeframes.items():
                forecast = res.forecast(horizon=min(hours, 30))  # Maximum 30 periods
                forecasted_vol = np.sqrt(forecast.variance.iloc[-1, :].mean())

                # Annualization
                annualized_vol = forecasted_vol * np.sqrt(24 * 365)
                volatilities[timeframe] = float(annualized_vol)

        except Exception as e:
            logger.warning("GARCH calculation failed", error=str(e))
            return await self._calculate_realized_volatility(price_data)

        return volatilities

    def normalize_volatility_score(
        self,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Normalize volatility to a 0-100 score for Fear & Greed Index

        Args:
            volatility: Volatility in annualized %

        Returns:
            Tuple[normalized_score, fear_greed_score]
        """

        # Invert the logic: high volatility = fear (low score)
        if volatility >= self.volatility_bounds["extreme_fear"]:
            fear_greed_score = 5    # Extreme fear
        elif volatility >= self.volatility_bounds["fear"]:
            # Linear interpolation between 5 and 25
            fear_greed_score = 5 + (25 - 5) * (
                (self.volatility_bounds["extreme_fear"] - volatility) /
                (self.volatility_bounds["extreme_fear"] - self.volatility_bounds["fear"])
            )
        elif volatility >= self.volatility_bounds["neutral"]:
            # Linear interpolation between 25 and 75
            fear_greed_score = 25 + (75 - 25) * (
                (self.volatility_bounds["fear"] - volatility) /
                (self.volatility_bounds["fear"] - self.volatility_bounds["neutral"])
            )
        elif volatility >= self.volatility_bounds["greed"]:
            # Linear interpolation between 75 and 95
            fear_greed_score = 75 + (95 - 75) * (
                (self.volatility_bounds["neutral"] - volatility) /
                (self.volatility_bounds["neutral"] - self.volatility_bounds["greed"])
            )
        else:
            fear_greed_score = 95   # Extreme greed

        # Normalized score (0-100)
        normalized_score = max(0, min(100, (volatility / 200) * 100))  # Cap at 200% volatility

        return float(normalized_score), float(fear_greed_score)

    async def get_fear_greed_component(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        method: VolatilityType = VolatilityType.REALIZED
    ) -> VolatilityData:
        """
        Get volatility component for Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC/USDT')
            price_data: OHLCV data
            method: Volatility calculation method

        Returns:
            VolatilityData object with results
        """
        try:
            # Calculate volatility
            volatilities = await self.calculate_volatility(price_data, method)

            # Use 30-day volatility as the primary metric
            main_volatility = volatilities.get("30d", 0.0)

            # Normalization
            normalized_score, fear_greed_score = self.normalize_volatility_score(main_volatility)

            # Calculate confidence based on data availability
            confidence = min(1.0, len(price_data) / (30 * 24))  # 30 days of hourly data

            result = VolatilityData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                volatility_1d=volatilities.get("1d", 0.0),
                volatility_7d=volatilities.get("7d", 0.0),
                volatility_30d=volatilities.get("30d", 0.0),
                volatility_90d=volatilities.get("90d", 0.0),
                normalized_score=normalized_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            # Caching
            self._volatility_cache[symbol] = result

            logger.info(
                "Volatility component calculated",
                symbol=symbol,
                volatility_30d=main_volatility,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating volatility component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_volatility(self, symbol: str) -> Optional[VolatilityData]:
        """Get cached volatility data"""
        return self._volatility_cache.get(symbol)

    async def batch_calculate(
        self,
        symbols_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, VolatilityData]:
        """
        Batch volatility calculation for multiple symbols

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

        volatility_results = {}
        for symbol, result in zip(symbols_data.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating volatility for {symbol}", error=str(result))
                continue
            volatility_results[symbol] = result

        return volatility_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
