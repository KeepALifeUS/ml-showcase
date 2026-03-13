"""
Price Data Collector for Fear & Greed Index

Collects OHLCV price data from various sources
with support for multiple exchanges.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog
import ccxt.async_support as ccxt

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class PriceDataCollector:
    """
    Price data collector for Fear & Greed Index

    Supports multiple exchanges and automatic
    data aggregation.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("price_data_collector")

        # Exchange configuration
        self.exchanges = {}
        self._initialize_exchanges()

        # Symbol mapping
        self.symbol_mapping = {
            'BTC': 'BTC/USDT',
            'ETH': 'ETH/USDT',
            'ADA': 'ADA/USDT',
            'DOT': 'DOT/USDT'
        }

        logger.info("PriceDataCollector initialized",
                   exchanges=list(self.exchanges.keys()))

    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        try:
            # Binance
            if self.config.binance_api_key:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.config.binance_api_key,
                    'secret': self.config.binance_secret_key,
                    'sandbox': not self.config.is_production(),
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })

            # Simulation of other exchanges for demonstration
            self.exchanges['demo'] = None  # Stub for demo data

        except Exception as e:
            logger.error("Error initializing exchanges", error=str(e))

    async def fetch_price_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        exchange: str = "binance"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data

        Args:
            symbol: Cryptocurrency symbol
            timeframe: Timeframe
            limit: Number of candles
            exchange: Exchange name

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert symbol
            trading_symbol = self.symbol_mapping.get(symbol, f"{symbol}/USDT")

            if exchange in self.exchanges and self.exchanges[exchange]:
                # Real data from exchange
                data = await self._fetch_from_exchange(
                    self.exchanges[exchange],
                    trading_symbol,
                    timeframe,
                    limit
                )
            else:
                # Simulated data for demonstration
                data = await self._simulate_price_data(symbol, timeframe, limit)

            # Data validation
            validation = self.validator.validate_price_data(data)
            if not validation.is_valid:
                logger.warning("Price data validation failed",
                             errors=validation.errors)

            self.metrics.record_collection("price_data", len(data))

            return data

        except Exception as e:
            logger.error("Error fetching price data",
                        symbol=symbol, exchange=exchange, error=str(e))
            self.metrics.record_error("price_fetch_error")
            raise

    async def _fetch_from_exchange(
        self,
        exchange: ccxt.Exchange,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Fetch data from a real exchange"""
        try:
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            return df

        except Exception as e:
            logger.error("Error fetching from exchange", error=str(e))
            raise

    async def _simulate_price_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Simulate price data for demonstration"""
        try:
            # Base prices for different cryptocurrencies
            base_prices = {
                'BTC': 45000,
                'ETH': 3000,
                'ADA': 0.5,
                'DOT': 25
            }

            base_price = base_prices.get(symbol, 100)

            # Generate timestamps
            end_time = datetime.utcnow()
            if timeframe == '1h':
                start_time = end_time - timedelta(hours=limit)
                freq = '1H'
            elif timeframe == '1d':
                start_time = end_time - timedelta(days=limit)
                freq = '1D'
            else:
                start_time = end_time - timedelta(hours=limit)
                freq = '1H'

            timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)[:limit]

            # Generate OHLCV data
            data = []
            current_price = base_price

            for timestamp in timestamps:
                # Random price change (-5% to +5%)
                price_change = np.random.uniform(-0.05, 0.05)
                current_price *= (1 + price_change)

                # OHLC with small volatility
                open_price = current_price
                high_price = open_price * (1 + abs(np.random.normal(0, 0.02)))
                low_price = open_price * (1 - abs(np.random.normal(0, 0.02)))
                close_price = np.random.uniform(low_price, high_price)

                # Volume (random)
                volume = np.random.uniform(1000000, 10000000)

                data.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })

                current_price = close_price

            df = pd.DataFrame(data, index=timestamps)

            logger.info(f"Simulated {len(df)} price data points for {symbol}")

            return df

        except Exception as e:
            logger.error("Error simulating price data", error=str(e))
            raise

    async def close(self):
        """Close exchange connections"""
        try:
            for exchange_name, exchange in self.exchanges.items():
                if exchange:
                    await exchange.close()
                    logger.info(f"Closed connection to {exchange_name}")
        except Exception as e:
            logger.error("Error closing exchanges", error=str(e))

    def get_metrics(self) -> Dict[str, float]:
        """Get collector metrics"""
        return self.metrics.get_metrics()
