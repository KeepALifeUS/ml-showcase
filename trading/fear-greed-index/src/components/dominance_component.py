"""
Dominance Component for Fear & Greed Index

Analyzes Bitcoin dominance and market cap distribution to assess market sentiment.
High BTC dominance = fear (flight to "safety"), low dominance = greed (altseason).
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


class DominanceMetric(Enum):
    """Types of dominance metrics"""
    BTC_DOMINANCE = "btc_dominance"
    ETH_DOMINANCE = "eth_dominance"
    TOP_10_DOMINANCE = "top_10_dominance"
    ALTCOIN_SEASON = "altcoin_season"
    DEFI_DOMINANCE = "defi_dominance"


@dataclass
class DominanceData:
    """Dominance data"""
    timestamp: datetime
    btc_dominance: float           # BTC market cap dominance %
    eth_dominance: float           # ETH market cap dominance %
    top_10_dominance: float        # Top 10 coins dominance %
    btc_dominance_change_7d: float # 7-day change in BTC dominance
    btc_dominance_change_30d: float # 30-day change in BTC dominance
    altcoin_season_index: float    # 0-100, 100 = full altseason
    dominance_volatility: float    # Volatility of BTC dominance
    market_concentration: float    # Herfindahl index of concentration
    fear_greed_score: float       # 0-100
    confidence: float             # 0-1


class DominanceComponent:
    """
    Dominance analysis component for Fear & Greed Index

    Analyzes market cap distribution among cryptocurrencies
    to assess market sentiment and cycle stage.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("dominance")
        self._dominance_cache: Dict[str, DominanceData] = {}

        # Historical dominance norms
        self.dominance_norms = {
            "btc_dominance": {
                "extreme_fear": 70,      # > 70% = extreme fear
                "fear": 60,              # 60-70% = fear
                "neutral": 50,           # 40-60% = neutral
                "greed": 40,            # 40-50% = greed
                "extreme_greed": 30     # < 30% = extreme greed (altseason)
            },
            "altcoin_season_threshold": 75,  # If altcoin season index > 75
            "volatility_threshold": 5.0      # High volatility threshold
        }

        # Weights for final score calculation
        self.component_weights = {
            "btc_dominance_level": 0.4,      # Absolute BTC dominance level
            "dominance_trend": 0.25,         # Dominance change trend
            "altcoin_season": 0.2,           # Altcoin season index
            "concentration": 0.1,            # Market concentration
            "volatility": 0.05               # Dominance volatility
        }

        logger.info("DominanceComponent initialized",
                   norms=self.dominance_norms, weights=self.component_weights)

    async def fetch_market_cap_data(self) -> pd.DataFrame:
        """
        Fetch market cap data (simulation - in reality via CoinGecko/CoinMarketCap API)

        Returns:
            DataFrame with market cap data for top cryptocurrencies
        """
        try:
            # Simulated market cap data
            # In a real implementation, this would query the CoinGecko API
            current_time = datetime.utcnow()

            simulated_data = []
            for i in range(30):  # 30 days of history
                timestamp = current_time - timedelta(days=i)

                # Simulated market cap data
                total_market_cap = 2_000_000_000_000 + np.random.normal(0, 200_000_000_000)  # $2T +/- $200B
                btc_market_cap = total_market_cap * (0.45 + np.random.normal(0, 0.05))  # ~45% +/- 5%
                eth_market_cap = total_market_cap * (0.18 + np.random.normal(0, 0.03))  # ~18% +/- 3%

                # Top 10 remaining coins
                remaining_top10 = total_market_cap * (0.25 + np.random.normal(0, 0.02))

                simulated_data.append({
                    'timestamp': timestamp,
                    'total_market_cap': total_market_cap,
                    'btc_market_cap': btc_market_cap,
                    'eth_market_cap': eth_market_cap,
                    'top_10_market_cap': btc_market_cap + eth_market_cap + remaining_top10,
                    'btc_dominance': (btc_market_cap / total_market_cap) * 100,
                    'eth_dominance': (eth_market_cap / total_market_cap) * 100,
                    'top_10_dominance': ((btc_market_cap + eth_market_cap + remaining_top10) / total_market_cap) * 100
                })

            df = pd.DataFrame(simulated_data)
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Fetched market cap data for {len(df)} days")
            self.metrics.record_collection("market_cap_data", len(df))

            return df

        except Exception as e:
            logger.error("Error fetching market cap data", error=str(e))
            self.metrics.record_error("market_cap_fetch_error")
            raise

    def calculate_altcoin_season_index(
        self,
        market_data: pd.DataFrame,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate Altcoin Season Index

        Altcoin season is defined as a period when altcoins
        outperform Bitcoin by market cap growth.

        Args:
            market_data: DataFrame with market cap data
            lookback_days: Number of days for analysis

        Returns:
            Altcoin season index (0-100)
        """
        if len(market_data) < lookback_days:
            return 50.0  # Neutral if insufficient data

        recent_data = market_data.tail(lookback_days)

        # BTC dominance change over the period
        btc_dominance_start = recent_data['btc_dominance'].iloc[0]
        btc_dominance_end = recent_data['btc_dominance'].iloc[-1]
        dominance_change = btc_dominance_end - btc_dominance_start

        # BTC dominance volatility
        dominance_volatility = recent_data['btc_dominance'].std()

        # Dominance change trend
        dominance_trend = np.polyfit(range(len(recent_data)), recent_data['btc_dominance'], 1)[0]

        # Altcoin season score
        # Negative BTC dominance trend = altcoin season
        trend_score = max(0, -dominance_trend * 20)  # Scaling
        change_score = max(0, -dominance_change * 2)   # Scaling

        # Combined index (0-100)
        altcoin_season_index = min(100, (trend_score + change_score) / 2)

        return altcoin_season_index

    def calculate_market_concentration(
        self,
        market_data: pd.DataFrame
    ) -> float:
        """
        Calculate market concentration index (Herfindahl Index)

        Args:
            market_data: DataFrame with market cap data

        Returns:
            Concentration index (0-1, where 1 = maximum concentration)
        """
        if market_data.empty:
            return 0.5

        latest_data = market_data.iloc[-1]

        # Market cap shares (in percentages)
        btc_share = latest_data['btc_dominance'] / 100
        eth_share = latest_data['eth_dominance'] / 100

        # Assume equal distribution of remaining top 10
        remaining_top10 = (latest_data['top_10_dominance'] - latest_data['btc_dominance'] - latest_data['eth_dominance']) / 100
        other_coins_in_top10 = 8  # Remaining 8 coins in top 10
        avg_other_share = remaining_top10 / other_coins_in_top10

        # Remaining coins outside top 10
        outside_top10_share = 1 - (latest_data['top_10_dominance'] / 100)

        # Herfindahl Index
        hhi = (btc_share ** 2 +
               eth_share ** 2 +
               other_coins_in_top10 * (avg_other_share ** 2) +
               outside_top10_share ** 2)

        return hhi

    def calculate_dominance_volatility(
        self,
        market_data: pd.DataFrame,
        lookback_days: int = 30
    ) -> float:
        """
        Calculate BTC dominance volatility

        Args:
            market_data: DataFrame with market cap data
            lookback_days: Number of days for calculation

        Returns:
            Dominance volatility (standard deviation)
        """
        if len(market_data) < lookback_days:
            return 0.0

        recent_data = market_data.tail(lookback_days)
        volatility = recent_data['btc_dominance'].std()

        return volatility

    def normalize_dominance_metrics(
        self,
        dominance_data: DominanceData
    ) -> Dict[str, float]:
        """
        Normalize dominance metrics to 0-100 scale for Fear & Greed calculation

        Args:
            dominance_data: Dominance data

        Returns:
            Dict with normalized values
        """
        normalized = {}

        # BTC Dominance level - invert the logic (high dominance = fear)
        btc_dom = dominance_data.btc_dominance
        if btc_dom >= self.dominance_norms["btc_dominance"]["extreme_fear"]:
            normalized['btc_dominance_level'] = 5  # Extreme fear
        elif btc_dom >= self.dominance_norms["btc_dominance"]["fear"]:
            # Linear interpolation between 5 and 25
            normalized['btc_dominance_level'] = 5 + (25 - 5) * (
                (self.dominance_norms["btc_dominance"]["extreme_fear"] - btc_dom) /
                (self.dominance_norms["btc_dominance"]["extreme_fear"] - self.dominance_norms["btc_dominance"]["fear"])
            )
        elif btc_dom >= self.dominance_norms["btc_dominance"]["greed"]:
            # Linear interpolation between 25 and 75
            normalized['btc_dominance_level'] = 25 + (75 - 25) * (
                (self.dominance_norms["btc_dominance"]["fear"] - btc_dom) /
                (self.dominance_norms["btc_dominance"]["fear"] - self.dominance_norms["btc_dominance"]["greed"])
            )
        elif btc_dom >= self.dominance_norms["btc_dominance"]["extreme_greed"]:
            # Linear interpolation between 75 and 95
            normalized['btc_dominance_level'] = 75 + (95 - 75) * (
                (self.dominance_norms["btc_dominance"]["greed"] - btc_dom) /
                (self.dominance_norms["btc_dominance"]["greed"] - self.dominance_norms["btc_dominance"]["extreme_greed"])
            )
        else:
            normalized['btc_dominance_level'] = 95  # Extreme greed (altseason)

        # Dominance trend - negative trend = growing greed
        dom_change_30d = dominance_data.btc_dominance_change_30d
        if dom_change_30d > 0:
            # Rising dominance = fear
            trend_score = max(0, 50 - dom_change_30d * 5)  # Decrease score
        else:
            # Falling dominance = greed
            trend_score = min(100, 50 + abs(dom_change_30d) * 5)  # Increase score
        normalized['dominance_trend'] = trend_score

        # Altcoin season index is already in 0-100
        normalized['altcoin_season'] = dominance_data.altcoin_season_index

        # Market concentration - high concentration = fear
        concentration = dominance_data.market_concentration
        concentration_score = max(0, 100 - concentration * 200)  # Invert
        normalized['concentration'] = concentration_score

        # Volatility - high volatility = uncertainty/fear
        volatility = dominance_data.dominance_volatility
        if volatility > self.dominance_norms["volatility_threshold"]:
            volatility_score = max(0, 50 - (volatility - 5) * 10)
        else:
            volatility_score = 50 + (5 - volatility) * 10
        normalized['volatility'] = min(100, volatility_score)

        return normalized

    def calculate_dominance_score(
        self,
        normalized_metrics: Dict[str, float]
    ) -> float:
        """
        Calculate combined dominance score

        Args:
            normalized_metrics: Normalized metrics

        Returns:
            Weighted dominance score (0-100)
        """
        total_score = 0.0
        total_weight = 0.0

        for component, weight in self.component_weights.items():
            if component in normalized_metrics:
                total_score += normalized_metrics[component] * weight
                total_weight += weight

        if total_weight > 0:
            return total_score / total_weight
        else:
            return 50.0  # Neutral if no metrics available

    async def get_fear_greed_component(self) -> DominanceData:
        """
        Get dominance component for Fear & Greed Index

        Returns:
            DominanceData object with results
        """
        try:
            # Fetch market cap data
            market_data = await self.fetch_market_cap_data()

            if market_data.empty:
                raise ValueError("No market cap data available")

            # Current values
            latest_data = market_data.iloc[-1]
            btc_dominance = latest_data['btc_dominance']
            eth_dominance = latest_data['eth_dominance']
            top_10_dominance = latest_data['top_10_dominance']

            # Changes over periods
            if len(market_data) >= 7:
                btc_dom_7d_ago = market_data.iloc[-7]['btc_dominance']
                btc_dominance_change_7d = btc_dominance - btc_dom_7d_ago
            else:
                btc_dominance_change_7d = 0.0

            if len(market_data) >= 30:
                btc_dom_30d_ago = market_data.iloc[-30]['btc_dominance']
                btc_dominance_change_30d = btc_dominance - btc_dom_30d_ago
            else:
                btc_dominance_change_30d = 0.0

            # Calculate additional metrics
            altcoin_season_index = self.calculate_altcoin_season_index(market_data)
            dominance_volatility = self.calculate_dominance_volatility(market_data)
            market_concentration = self.calculate_market_concentration(market_data)

            # Create DominanceData object
            dominance_data = DominanceData(
                timestamp=datetime.utcnow(),
                btc_dominance=btc_dominance,
                eth_dominance=eth_dominance,
                top_10_dominance=top_10_dominance,
                btc_dominance_change_7d=btc_dominance_change_7d,
                btc_dominance_change_30d=btc_dominance_change_30d,
                altcoin_season_index=altcoin_season_index,
                dominance_volatility=dominance_volatility,
                market_concentration=market_concentration,
                fear_greed_score=0.0,  # Will be calculated below
                confidence=1.0         # High confidence for market cap data
            )

            # Normalize metrics
            normalized_metrics = self.normalize_dominance_metrics(dominance_data)

            # Calculate final Fear & Greed score
            fear_greed_score = self.calculate_dominance_score(normalized_metrics)
            dominance_data.fear_greed_score = fear_greed_score

            # Caching
            self._dominance_cache["global"] = dominance_data

            logger.info(
                "Dominance component calculated",
                btc_dominance=btc_dominance,
                dominance_change_30d=btc_dominance_change_30d,
                altcoin_season_index=altcoin_season_index,
                fear_greed_score=fear_greed_score
            )

            self.metrics.record_calculation("component_calculation", 1)

            return dominance_data

        except Exception as e:
            logger.error("Error calculating dominance component", error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_dominance(self) -> Optional[DominanceData]:
        """Get cached dominance data"""
        return self._dominance_cache.get("global")

    def get_market_regime_analysis(
        self,
        dominance_data: DominanceData
    ) -> Dict[str, str]:
        """
        Analyze market regime based on dominance

        Args:
            dominance_data: Dominance data

        Returns:
            Dict with description of the current market regime
        """
        regime = {}

        btc_dom = dominance_data.btc_dominance
        altcoin_index = dominance_data.altcoin_season_index
        trend = dominance_data.btc_dominance_change_30d

        # Determine market phase
        if btc_dom > 65:
            if trend > 2:
                regime["phase"] = "Bear Market / Flight to Safety"
                regime["description"] = "Investors fleeing to Bitcoin as safe haven"
            else:
                regime["phase"] = "Bitcoin Accumulation"
                regime["description"] = "High Bitcoin dominance, market uncertainty"
        elif btc_dom < 40 and altcoin_index > 70:
            regime["phase"] = "Altcoin Season"
            regime["description"] = "Strong altcoin performance, market euphoria"
        elif trend < -3:
            regime["phase"] = "Early Altcoin Rally"
            regime["description"] = "Bitcoin dominance declining, altcoins gaining"
        else:
            regime["phase"] = "Balanced Market"
            regime["description"] = "Relatively balanced market conditions"

        # Recommendations
        if btc_dom > 60:
            regime["recommendation"] = "High fear levels, potential buying opportunity"
        elif btc_dom < 35:
            regime["recommendation"] = "High greed levels, consider profit taking"
        else:
            regime["recommendation"] = "Neutral conditions, monitor for changes"

        return regime

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
