"""
Search Trends Component for Fear & Greed Index

Analyzes Google search trends to determine public interest in cryptocurrencies.
High interest = greed/FOMO, low interest = fear/apathy.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not available, using simulated data")

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class TrendPeriod(Enum):
    """Periods for trend analysis"""
    REALTIME = "now 1-H"      # Last hour (real-time)
    TODAY = "now 1-d"         # Today
    WEEK = "now 7-d"          # Week
    MONTH = "today 1-m"       # Month
    THREE_MONTHS = "today 3-m" # Three months
    YEAR = "today 12-m"       # Year


class SearchCategory(Enum):
    """Search categories"""
    ALL = 0                   # All categories
    FINANCE = 7              # Finance
    TECHNOLOGY = 5           # Technology
    NEWS = 16                # News


@dataclass
class TrendData:
    """Search trend data"""
    timestamp: datetime
    keyword: str
    search_volume: int        # Relative search volume (0-100)
    trend_7d: float          # 7-day change (%)
    trend_30d: float         # 30-day change (%)
    volatility: float        # Trend volatility
    rising_queries: List[str] # Rising queries
    top_queries: List[str]   # Top related queries
    geographical_spread: int  # Number of countries with high interest
    fear_greed_score: float  # 0-100
    confidence: float        # 0-1


@dataclass
class AggregatedTrends:
    """Aggregated trend data"""
    timestamp: datetime
    symbol: str
    total_search_volume: float    # Total search volume
    trend_momentum: float         # Trend momentum
    fomo_index: float            # FOMO index (0-100)
    panic_index: float           # Panic index (0-100)
    mainstream_adoption: float   # Mainstream adoption index (0-100)
    fear_greed_score: float      # 0-100
    confidence: float            # 0-1
    keyword_breakdown: Dict[str, TrendData]


class SearchTrendsComponent:
    """
    Search trends analysis component for Fear & Greed Index

    Uses Google Trends to assess public interest
    and sentiment regarding cryptocurrencies.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("search_trends")
        self._trends_cache: Dict[str, AggregatedTrends] = {}

        # Initialize Google Trends API
        if PYTRENDS_AVAILABLE:
            self.pytrends = TrendReq(hl='en-US', tz=360)
        else:
            self.pytrends = None
            logger.warning("Google Trends API not available, using simulated data")

        # Keywords for different cryptocurrencies
        self.crypto_keywords = {
            'BTC': {
                'primary': ['Bitcoin', 'BTC'],
                'secondary': ['Bitcoin price', 'Bitcoin news', 'Bitcoin buy'],
                'fear': ['Bitcoin crash', 'Bitcoin bubble', 'Bitcoin scam'],
                'greed': ['Bitcoin moon', 'Bitcoin bull', 'Bitcoin ATH']
            },
            'ETH': {
                'primary': ['Ethereum', 'ETH'],
                'secondary': ['Ethereum price', 'Ethereum news', 'Ethereum 2.0'],
                'fear': ['Ethereum crash', 'Ethereum gas fees'],
                'greed': ['Ethereum bull', 'Ethereum DeFi', 'Ethereum NFT']
            },
            'CRYPTO': {
                'primary': ['Cryptocurrency', 'Crypto'],
                'secondary': ['Crypto news', 'Crypto price', 'Crypto market'],
                'fear': ['Crypto crash', 'Crypto bear market', 'Crypto regulation'],
                'greed': ['Crypto bull run', 'Crypto FOMO', 'Crypto to the moon']
            }
        }

        # Normalization settings
        self.trend_thresholds = {
            "high_interest": 80,      # High interest (greed)
            "medium_interest": 40,    # Medium interest
            "low_interest": 20,       # Low interest (fear/apathy)
            "volatility_threshold": 25 # High volatility threshold
        }

        # Weights for final score calculation
        self.component_weights = {
            "search_volume": 0.4,        # Absolute search volume
            "trend_momentum": 0.25,      # Change momentum
            "fomo_signals": 0.2,         # FOMO indicators
            "fear_signals": 0.1,         # Fear signals
            "mainstream_adoption": 0.05   # Mainstream adoption
        }

        logger.info("SearchTrendsComponent initialized",
                   keywords_available=list(self.crypto_keywords.keys()),
                   thresholds=self.trend_thresholds)

    async def fetch_google_trends(
        self,
        keywords: List[str],
        period: TrendPeriod = TrendPeriod.MONTH,
        category: SearchCategory = SearchCategory.FINANCE
    ) -> pd.DataFrame:
        """
        Fetch Google Trends data

        Args:
            keywords: List of keywords
            period: Analysis period
            category: Search category

        Returns:
            DataFrame with trend data
        """
        try:
            if not self.pytrends:
                return await self._simulate_trends_data(keywords, period)

            # Build Google Trends query
            self.pytrends.build_payload(
                kw_list=keywords,
                cat=category.value,
                timeframe=period.value,
                geo='',  # Worldwide
                gprop=''
            )

            # Get interest over time data
            interest_over_time = self.pytrends.interest_over_time()

            if interest_over_time.empty:
                logger.warning("No trends data received", keywords=keywords)
                return await self._simulate_trends_data(keywords, period)

            # Remove 'isPartial' column if present
            if 'isPartial' in interest_over_time.columns:
                interest_over_time = interest_over_time.drop(['isPartial'], axis=1)

            logger.info(f"Fetched Google Trends data for {len(keywords)} keywords")
            self.metrics.record_collection("trends_data", len(interest_over_time))

            return interest_over_time

        except Exception as e:
            logger.error("Error fetching Google Trends", keywords=keywords, error=str(e))
            self.metrics.record_error("trends_fetch_error")
            return await self._simulate_trends_data(keywords, period)

    async def _simulate_trends_data(
        self,
        keywords: List[str],
        period: TrendPeriod
    ) -> pd.DataFrame:
        """Simulate trend data for demonstration"""

        # Determine the number of data points based on period
        if period == TrendPeriod.REALTIME:
            periods = 24  # Hours
            freq = 'H'
        elif period == TrendPeriod.TODAY:
            periods = 24  # Hours
            freq = 'H'
        elif period == TrendPeriod.WEEK:
            periods = 7   # Days
            freq = 'D'
        elif period == TrendPeriod.MONTH:
            periods = 30  # Days
            freq = 'D'
        else:
            periods = 90  # Days for longer periods
            freq = 'D'

        # Generate timestamps
        end_time = datetime.utcnow()
        if freq == 'H':
            start_time = end_time - timedelta(hours=periods)
            date_range = pd.date_range(start=start_time, end=end_time, freq='H')
        else:
            start_time = end_time - timedelta(days=periods)
            date_range = pd.date_range(start=start_time, end=end_time, freq='D')

        # Simulate data for each keyword
        simulated_data = {}

        for keyword in keywords:
            # Base interest level (20-80)
            base_interest = np.random.uniform(20, 80)

            # Generate trend data with some volatility
            trend_data = []
            current_value = base_interest

            for _ in date_range:
                # Random change (-20% to +20%)
                change = np.random.normal(0, 0.1) * current_value
                current_value = max(0, min(100, current_value + change))
                trend_data.append(int(current_value))

            simulated_data[keyword] = trend_data

        # Create DataFrame
        df = pd.DataFrame(simulated_data, index=date_range)

        logger.info(f"Simulated trends data for {len(keywords)} keywords over {periods} periods")

        return df

    async def get_related_queries(
        self,
        keyword: str
    ) -> Tuple[List[str], List[str]]:
        """
        Get related queries for a keyword

        Args:
            keyword: Keyword

        Returns:
            Tuple[top_queries, rising_queries]
        """
        try:
            if not self.pytrends:
                return self._simulate_related_queries(keyword)

            # Get related queries
            related_queries = self.pytrends.related_queries()

            top_queries = []
            rising_queries = []

            if keyword in related_queries:
                # Top queries
                if 'top' in related_queries[keyword] and related_queries[keyword]['top'] is not None:
                    top_df = related_queries[keyword]['top']
                    top_queries = top_df['query'].head(10).tolist()

                # Rising queries
                if 'rising' in related_queries[keyword] and related_queries[keyword]['rising'] is not None:
                    rising_df = related_queries[keyword]['rising']
                    rising_queries = rising_df['query'].head(10).tolist()

            return top_queries, rising_queries

        except Exception as e:
            logger.error("Error fetching related queries", keyword=keyword, error=str(e))
            return self._simulate_related_queries(keyword)

    def _simulate_related_queries(self, keyword: str) -> Tuple[List[str], List[str]]:
        """Simulate related queries"""

        base_keyword = keyword.lower()

        # Simulated top queries
        top_queries = [
            f"{base_keyword} price",
            f"{base_keyword} news",
            f"{base_keyword} buy",
            f"{base_keyword} wallet",
            f"{base_keyword} forecast"
        ]

        # Simulated rising queries
        rising_queries = [
            f"{base_keyword} bull run",
            f"{base_keyword} prediction",
            f"{base_keyword} analysis",
            f"{base_keyword} investment"
        ]

        return top_queries, rising_queries

    def calculate_fomo_index(
        self,
        trends_data: pd.DataFrame,
        keywords: Dict[str, List[str]]
    ) -> float:
        """
        Calculate FOMO index based on search trends

        Args:
            trends_data: DataFrame with trend data
            keywords: Dict with keyword categories

        Returns:
            FOMO index (0-100)
        """
        fomo_signals = []

        # Analyze FOMO keywords
        if 'greed' in keywords:
            for greed_keyword in keywords['greed']:
                if greed_keyword in trends_data.columns:
                    recent_avg = trends_data[greed_keyword].tail(7).mean()
                    historical_avg = trends_data[greed_keyword].mean()

                    if historical_avg > 0:
                        fomo_ratio = recent_avg / historical_avg
                        fomo_signals.append(min(100, fomo_ratio * 50))  # Scaling

        # Analyze overall interest growth
        for col in trends_data.columns:
            recent_trend = trends_data[col].tail(7).mean()
            prev_trend = trends_data[col].iloc[-14:-7].mean() if len(trends_data) >= 14 else recent_trend

            if prev_trend > 0:
                growth_rate = (recent_trend - prev_trend) / prev_trend
                if growth_rate > 0.2:  # Growth over 20%
                    fomo_signals.append(min(100, growth_rate * 200))

        if fomo_signals:
            return np.mean(fomo_signals)
        else:
            return 0.0

    def calculate_panic_index(
        self,
        trends_data: pd.DataFrame,
        keywords: Dict[str, List[str]]
    ) -> float:
        """
        Calculate panic index based on search trends

        Args:
            trends_data: DataFrame with trend data
            keywords: Dict with keyword categories

        Returns:
            Panic index (0-100)
        """
        panic_signals = []

        # Analyze fear-related keywords
        if 'fear' in keywords:
            for fear_keyword in keywords['fear']:
                if fear_keyword in trends_data.columns:
                    recent_avg = trends_data[fear_keyword].tail(7).mean()
                    historical_avg = trends_data[fear_keyword].mean()

                    if historical_avg > 0:
                        panic_ratio = recent_avg / historical_avg
                        panic_signals.append(min(100, panic_ratio * 50))

        # Analyze sharp search spikes
        for col in trends_data.columns:
            if len(trends_data) > 1:
                volatility = trends_data[col].rolling(7).std().iloc[-1]
                if volatility > 20:  # High volatility
                    panic_signals.append(min(100, volatility * 2))

        if panic_signals:
            return np.mean(panic_signals)
        else:
            return 0.0

    def calculate_mainstream_adoption_index(
        self,
        trends_data: pd.DataFrame,
        keywords: Dict[str, List[str]]
    ) -> float:
        """
        Calculate mainstream adoption index

        Args:
            trends_data: DataFrame with trend data
            keywords: Dict with keyword categories

        Returns:
            Mainstream adoption index (0-100)
        """
        # Analyze stability of interest in primary terms
        adoption_signals = []

        if 'primary' in keywords:
            for primary_keyword in keywords['primary']:
                if primary_keyword in trends_data.columns:
                    # Consistently high interest = mainstream adoption
                    avg_interest = trends_data[primary_keyword].mean()
                    stability = 100 - trends_data[primary_keyword].std()  # Low volatility = stability

                    adoption_score = (avg_interest + stability) / 2
                    adoption_signals.append(min(100, adoption_score))

        if adoption_signals:
            return np.mean(adoption_signals)
        else:
            return 0.0

    async def analyze_keyword_trend(
        self,
        keyword: str,
        trends_data: pd.DataFrame
    ) -> TrendData:
        """
        Analyze trend for a specific keyword

        Args:
            keyword: Keyword
            trends_data: DataFrame with trend data

        Returns:
            TrendData object with results
        """
        try:
            if keyword not in trends_data.columns:
                # No data for the keyword
                return TrendData(
                    timestamp=datetime.utcnow(),
                    keyword=keyword,
                    search_volume=0,
                    trend_7d=0.0,
                    trend_30d=0.0,
                    volatility=0.0,
                    rising_queries=[],
                    top_queries=[],
                    geographical_spread=0,
                    fear_greed_score=50.0,
                    confidence=0.0
                )

            data_series = trends_data[keyword]

            # Current search volume (last value)
            current_volume = int(data_series.iloc[-1])

            # Changes over periods
            if len(data_series) >= 7:
                trend_7d = ((data_series.iloc[-1] - data_series.iloc[-7]) / data_series.iloc[-7]) * 100 if data_series.iloc[-7] > 0 else 0
            else:
                trend_7d = 0.0

            if len(data_series) >= 30:
                trend_30d = ((data_series.iloc[-1] - data_series.iloc[-30]) / data_series.iloc[-30]) * 100 if data_series.iloc[-30] > 0 else 0
            else:
                trend_30d = 0.0

            # Volatility
            volatility = float(data_series.std())

            # Related queries
            top_queries, rising_queries = await self.get_related_queries(keyword)

            # Fear & Greed score based on volume and trend
            volume_score = min(100, (current_volume / 100) * 100)  # Normalization
            trend_score = 50 + (trend_30d / 100) * 50  # Positive trend = higher score
            fear_greed_score = (volume_score + max(0, min(100, trend_score))) / 2

            # Confidence based on data volume and quality
            data_completeness = len(data_series) / 30  # 30 days of data
            volume_quality = min(1.0, current_volume / 50)  # Minimum volume for quality
            confidence = min(1.0, (data_completeness + volume_quality) / 2)

            result = TrendData(
                timestamp=datetime.utcnow(),
                keyword=keyword,
                search_volume=current_volume,
                trend_7d=trend_7d,
                trend_30d=trend_30d,
                volatility=volatility,
                rising_queries=rising_queries,
                top_queries=top_queries,
                geographical_spread=len(set(rising_queries + top_queries)),  # Simple approximation
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            return result

        except Exception as e:
            logger.error("Error analyzing keyword trend", keyword=keyword, error=str(e))
            raise

    async def get_fear_greed_component(
        self,
        symbol: str,
        period: TrendPeriod = TrendPeriod.MONTH
    ) -> AggregatedTrends:
        """
        Get search trends component for Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol (BTC, ETH, CRYPTO)
            period: Analysis period

        Returns:
            AggregatedTrends object with results
        """
        try:
            if symbol.upper() not in self.crypto_keywords:
                raise ValueError(f"Unsupported symbol: {symbol}")

            keywords_config = self.crypto_keywords[symbol.upper()]

            # Collect all keywords for analysis
            all_keywords = []
            for category in keywords_config.values():
                all_keywords.extend(category)

            # Fetch trend data
            trends_data = await self.fetch_google_trends(all_keywords[:5], period)  # Limit to 5 due to API limits

            if trends_data.empty:
                raise ValueError("No trends data available")

            # Analyze each keyword
            keyword_breakdown = {}
            for keyword in trends_data.columns:
                trend_data = await self.analyze_keyword_trend(keyword, trends_data)
                keyword_breakdown[keyword] = trend_data

            # Calculate aggregated metrics
            total_search_volume = sum(td.search_volume for td in keyword_breakdown.values())

            # Trend momentum - weighted average change
            total_weight = sum(td.search_volume for td in keyword_breakdown.values())
            if total_weight > 0:
                trend_momentum = sum(
                    td.trend_30d * td.search_volume
                    for td in keyword_breakdown.values()
                ) / total_weight
            else:
                trend_momentum = 0.0

            # FOMO and panic indices
            fomo_index = self.calculate_fomo_index(trends_data, keywords_config)
            panic_index = self.calculate_panic_index(trends_data, keywords_config)
            mainstream_adoption = self.calculate_mainstream_adoption_index(trends_data, keywords_config)

            # Final Fear & Greed score
            volume_component = min(100, (total_search_volume / 500) * 100)  # Normalization
            momentum_component = 50 + (trend_momentum / 100) * 50
            fomo_component = fomo_index
            fear_component = 100 - panic_index  # Invert panic
            adoption_component = mainstream_adoption

            weighted_score = (
                volume_component * self.component_weights["search_volume"] +
                max(0, min(100, momentum_component)) * self.component_weights["trend_momentum"] +
                fomo_component * self.component_weights["fomo_signals"] +
                fear_component * self.component_weights["fear_signals"] +
                adoption_component * self.component_weights["mainstream_adoption"]
            )

            # Confidence based on data quality
            avg_confidence = np.mean([td.confidence for td in keyword_breakdown.values()]) if keyword_breakdown else 0.0

            result = AggregatedTrends(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                total_search_volume=total_search_volume,
                trend_momentum=trend_momentum,
                fomo_index=fomo_index,
                panic_index=panic_index,
                mainstream_adoption=mainstream_adoption,
                fear_greed_score=weighted_score,
                confidence=avg_confidence,
                keyword_breakdown=keyword_breakdown
            )

            # Caching
            self._trends_cache[symbol] = result

            logger.info(
                "Search trends component calculated",
                symbol=symbol,
                total_search_volume=total_search_volume,
                trend_momentum=trend_momentum,
                fomo_index=fomo_index,
                fear_greed_score=weighted_score,
                confidence=avg_confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating search trends component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_trends(self, symbol: str) -> Optional[AggregatedTrends]:
        """Get cached trend data"""
        return self._trends_cache.get(symbol)

    async def batch_calculate(
        self,
        symbols: List[str],
        period: TrendPeriod = TrendPeriod.MONTH
    ) -> Dict[str, AggregatedTrends]:
        """
        Batch trend calculation for multiple symbols

        Args:
            symbols: List of symbols for analysis
            period: Analysis period

        Returns:
            Dict with results for each symbol
        """
        tasks = []

        for symbol in symbols:
            task = self.get_fear_greed_component(symbol, period)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        trends_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating trends for {symbol}", error=str(result))
                continue
            trends_results[symbol] = result

        return trends_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
