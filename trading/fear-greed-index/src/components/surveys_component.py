"""
Surveys Component for Fear & Greed Index

Analyzes market survey results and sentiment surveys to assess market participant sentiment.
Includes professional surveys, retail surveys, and sentiment indices.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum
import aiohttp
import json

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class SurveyType(Enum):
    """Types of market surveys"""
    INSTITUTIONAL = "institutional"  # Institutional investors
    RETAIL = "retail"               # Retail investors
    PROFESSIONAL = "professional"   # Professional traders
    SENTIMENT = "sentiment"         # General sentiment surveys
    FEAR_GREED = "fear_greed"      # Specialized F&G surveys


class SurveySource(Enum):
    """Survey sources"""
    AAII = "aaii"                   # American Association of Individual Investors
    II = "investors_intelligence"   # Investors Intelligence
    CRYPTO_FEAR_GREED = "crypto_fear_greed"  # Existing crypto Fear & Greed indices
    SANTIMENT = "santiment"         # Santiment sentiment data
    ALTERNATIVE_ME = "alternative_me"  # Alternative.me Fear & Greed
    CUSTOM_SURVEY = "custom_survey" # Custom surveys


@dataclass
class SurveyResult:
    """Survey result"""
    timestamp: datetime
    source: SurveySource
    survey_type: SurveyType
    total_responses: int
    bullish_percentage: float       # Percentage of bullish responses
    bearish_percentage: float       # Percentage of bearish responses
    neutral_percentage: float       # Percentage of neutral responses
    sentiment_score: float          # -100 to 100 (bearish to bullish)
    confidence_interval: float      # Statistical margin of error
    sample_quality: float          # Sample quality (0-1)
    fear_greed_score: float        # 0-100
    confidence: float              # 0-1


@dataclass
class AggregatedSurveyData:
    """Aggregated survey data"""
    timestamp: datetime
    total_surveys: int
    weighted_sentiment: float       # Weighted sentiment (-100 to 100)
    retail_sentiment: float        # Retail sentiment
    institutional_sentiment: float # Institutional sentiment
    professional_sentiment: float  # Professional sentiment
    sentiment_divergence: float    # Divergence between groups
    survey_momentum: float         # Sentiment change over time
    contrarian_signal: float       # Contrarian signal (0-100)
    fear_greed_score: float        # 0-100
    confidence: float              # 0-1
    source_breakdown: Dict[SurveySource, SurveyResult]


class SurveysComponent:
    """
    Market surveys analysis component for Fear & Greed Index

    Collects and analyzes data from various sentiment surveys
    to assess market participant sentiment.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("surveys")
        self._surveys_cache: Dict[str, AggregatedSurveyData] = {}

        # Data source settings
        self.source_weights = {
            SurveySource.ALTERNATIVE_ME: 0.3,      # Popular index
            SurveySource.CRYPTO_FEAR_GREED: 0.25,  # Other crypto F&G indices
            SurveySource.SANTIMENT: 0.2,           # Santiment data
            SurveySource.CUSTOM_SURVEY: 0.15,      # Custom surveys
            SurveySource.AAII: 0.1                 # Traditional surveys (adapted)
        }

        # Survey type settings
        self.survey_type_weights = {
            SurveyType.INSTITUTIONAL: 0.4,    # High weight for institutions
            SurveyType.PROFESSIONAL: 0.3,     # Professionals
            SurveyType.RETAIL: 0.2,          # Retail
            SurveyType.SENTIMENT: 0.1        # General sentiment
        }

        # Interpretation thresholds
        self.sentiment_thresholds = {
            "extreme_fear": -80,      # < -80 = extreme fear
            "fear": -40,              # -80 to -40 = fear
            "neutral": 0,             # -40 to 40 = neutral
            "greed": 40,              # 40 to 80 = greed
            "extreme_greed": 80       # > 80 = extreme greed
        }

        # Contrarian thresholds (when the crowd is wrong)
        self.contrarian_thresholds = {
            "extreme_bullish": 85,    # > 85% bullish = time to sell
            "extreme_bearish": 15     # < 15% bullish = time to buy
        }

        logger.info("SurveysComponent initialized",
                   sources=list(self.source_weights.keys()),
                   thresholds=self.sentiment_thresholds)

    async def fetch_alternative_me_index(self) -> Optional[SurveyResult]:
        """
        Fetch data from Alternative.me Fear & Greed Index

        Returns:
            SurveyResult with data or None on error
        """
        try:
            url = "https://api.alternative.me/fng/"

            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'data' in data and len(data['data']) > 0:
                            latest = data['data'][0]

                            # Convert to our format
                            fng_value = int(latest['value'])
                            fng_classification = latest['value_classification'].lower()

                            # Convert 0-100 to -100 to 100 sentiment
                            sentiment_score = (fng_value - 50) * 2

                            # Determine percentages based on classification
                            if fng_classification == "extreme fear":
                                bullish_pct, neutral_pct, bearish_pct = 10, 20, 70
                            elif fng_classification == "fear":
                                bullish_pct, neutral_pct, bearish_pct = 25, 30, 45
                            elif fng_classification == "neutral":
                                bullish_pct, neutral_pct, bearish_pct = 40, 40, 20
                            elif fng_classification == "greed":
                                bullish_pct, neutral_pct, bearish_pct = 60, 25, 15
                            else:  # extreme greed
                                bullish_pct, neutral_pct, bearish_pct = 80, 15, 5

                            result = SurveyResult(
                                timestamp=datetime.utcfromtimestamp(int(latest['timestamp'])),
                                source=SurveySource.ALTERNATIVE_ME,
                                survey_type=SurveyType.SENTIMENT,
                                total_responses=1000,  # Approximate value
                                bullish_percentage=bullish_pct,
                                bearish_percentage=bearish_pct,
                                neutral_percentage=neutral_pct,
                                sentiment_score=sentiment_score,
                                confidence_interval=5.0,  # +/-5%
                                sample_quality=0.8,
                                fear_greed_score=float(fng_value),
                                confidence=0.9
                            )

                            logger.info("Fetched Alternative.me F&G index",
                                       value=fng_value, classification=fng_classification)
                            self.metrics.record_collection("alternative_me_data", 1)

                            return result

                    logger.warning("Failed to fetch Alternative.me data", status=response.status)
                    return None

        except Exception as e:
            logger.error("Error fetching Alternative.me data", error=str(e))
            self.metrics.record_error("alternative_me_fetch_error")
            return None

    async def simulate_institutional_survey(self) -> SurveyResult:
        """Simulate an institutional survey"""

        # Simulate institutional survey results
        # In reality, this would use data from Bloomberg, Thomson Reuters, etc.

        bullish_pct = np.random.uniform(20, 70)
        bearish_pct = np.random.uniform(10, 40)
        neutral_pct = 100 - bullish_pct - bearish_pct

        if neutral_pct < 0:
            neutral_pct = 10
            total = bullish_pct + bearish_pct + neutral_pct
            bullish_pct = (bullish_pct / total) * 100
            bearish_pct = (bearish_pct / total) * 100
            neutral_pct = (neutral_pct / total) * 100

        # Sentiment score based on distribution
        sentiment_score = bullish_pct - bearish_pct  # -100 to 100 range

        # Fear & Greed score
        fear_greed_score = (sentiment_score + 100) / 2  # Convert to 0-100

        return SurveyResult(
            timestamp=datetime.utcnow(),
            source=SurveySource.CUSTOM_SURVEY,
            survey_type=SurveyType.INSTITUTIONAL,
            total_responses=150,  # Typical institutional sample size
            bullish_percentage=bullish_pct,
            bearish_percentage=bearish_pct,
            neutral_percentage=neutral_pct,
            sentiment_score=sentiment_score,
            confidence_interval=8.0,  # +/-8%
            sample_quality=0.9,  # High quality
            fear_greed_score=fear_greed_score,
            confidence=0.85
        )

    async def simulate_retail_survey(self) -> SurveyResult:
        """Simulate a retail survey"""

        # Retail investors are usually more emotional and prone to extremes
        sentiment_bias = np.random.choice([-1, 1])  # Random bias

        if sentiment_bias > 0:  # Bullish bias
            bullish_pct = np.random.uniform(55, 85)
            bearish_pct = np.random.uniform(5, 25)
        else:  # Bearish bias
            bullish_pct = np.random.uniform(10, 35)
            bearish_pct = np.random.uniform(45, 75)

        neutral_pct = 100 - bullish_pct - bearish_pct
        if neutral_pct < 0:
            neutral_pct = 10
            total = bullish_pct + bearish_pct + neutral_pct
            bullish_pct = (bullish_pct / total) * 100
            bearish_pct = (bearish_pct / total) * 100
            neutral_pct = (neutral_pct / total) * 100

        sentiment_score = bullish_pct - bearish_pct
        fear_greed_score = (sentiment_score + 100) / 2

        return SurveyResult(
            timestamp=datetime.utcnow(),
            source=SurveySource.CUSTOM_SURVEY,
            survey_type=SurveyType.RETAIL,
            total_responses=2500,  # Large retail sample
            bullish_percentage=bullish_pct,
            bearish_percentage=bearish_pct,
            neutral_percentage=neutral_pct,
            sentiment_score=sentiment_score,
            confidence_interval=3.0,  # +/-3% (large sample)
            sample_quality=0.7,  # Medium quality
            fear_greed_score=fear_greed_score,
            confidence=0.75
        )

    async def simulate_professional_survey(self) -> SurveyResult:
        """Simulate a professional traders survey"""

        # Professionals are usually more conservative and realistic
        bullish_pct = np.random.uniform(30, 60)
        bearish_pct = np.random.uniform(20, 50)
        neutral_pct = 100 - bullish_pct - bearish_pct

        if neutral_pct < 10:
            neutral_pct = 20
            total = bullish_pct + bearish_pct + neutral_pct
            bullish_pct = (bullish_pct / total) * 80
            bearish_pct = (bearish_pct / total) * 80
            neutral_pct = 20

        sentiment_score = bullish_pct - bearish_pct
        fear_greed_score = (sentiment_score + 100) / 2

        return SurveyResult(
            timestamp=datetime.utcnow(),
            source=SurveySource.CUSTOM_SURVEY,
            survey_type=SurveyType.PROFESSIONAL,
            total_responses=500,
            bullish_percentage=bullish_pct,
            bearish_percentage=bearish_pct,
            neutral_percentage=neutral_pct,
            sentiment_score=sentiment_score,
            confidence_interval=6.0,  # +/-6%
            sample_quality=0.95,  # High quality
            fear_greed_score=fear_greed_score,
            confidence=0.9
        )

    def calculate_contrarian_signal(
        self,
        survey_results: Dict[SurveySource, SurveyResult]
    ) -> float:
        """
        Calculate contrarian signal

        When too many people share the same sentiment,
        the market often moves in the opposite direction.

        Args:
            survey_results: Dict with survey results

        Returns:
            Contrarian signal (0-100, where 100 = strong contrarian signal)
        """
        contrarian_signals = []

        for survey_result in survey_results.values():
            bullish_pct = survey_result.bullish_percentage

            # Strong contrarian signal at extreme values
            if bullish_pct >= self.contrarian_thresholds["extreme_bullish"]:
                # Too many bulls = time to sell
                contrarian_signals.append(100)
            elif bullish_pct <= self.contrarian_thresholds["extreme_bearish"]:
                # Too many bears = time to buy
                contrarian_signals.append(100)
            elif bullish_pct >= 75:
                # Moderately high bullish sentiment
                signal = (bullish_pct - 75) * 4  # Scaling
                contrarian_signals.append(min(100, signal))
            elif bullish_pct <= 25:
                # Moderately high bearish sentiment
                signal = (25 - bullish_pct) * 4  # Scaling
                contrarian_signals.append(min(100, signal))
            else:
                # Normal range - no contrarian signal
                contrarian_signals.append(0)

        if contrarian_signals:
            return np.mean(contrarian_signals)
        else:
            return 0.0

    def calculate_sentiment_divergence(
        self,
        survey_results: Dict[SurveySource, SurveyResult]
    ) -> float:
        """
        Calculate sentiment divergence between different groups

        Args:
            survey_results: Dict with survey results

        Returns:
            Sentiment divergence (0-100, where 100 = maximum divergence)
        """
        sentiment_by_type = {}

        # Group by survey types
        for survey_result in survey_results.values():
            survey_type = survey_result.survey_type
            if survey_type not in sentiment_by_type:
                sentiment_by_type[survey_type] = []
            sentiment_by_type[survey_type].append(survey_result.sentiment_score)

        if len(sentiment_by_type) < 2:
            return 0.0  # No divergence with a single type

        # Calculate average sentiment by type
        avg_sentiments = {}
        for survey_type, sentiments in sentiment_by_type.items():
            avg_sentiments[survey_type] = np.mean(sentiments)

        # Calculate standard deviation between groups
        sentiment_values = list(avg_sentiments.values())
        divergence = np.std(sentiment_values)

        # Normalize to 0-100 range (max divergence = 100 points difference)
        normalized_divergence = min(100, (divergence / 100) * 100)

        return normalized_divergence

    def calculate_survey_momentum(
        self,
        symbol: str,
        current_sentiment: float
    ) -> float:
        """
        Calculate sentiment change over time (momentum)

        Args:
            symbol: Asset symbol
            current_sentiment: Current weighted sentiment

        Returns:
            Survey momentum (-100 to 100)
        """
        # Compare with previous data from cache
        if symbol in self._surveys_cache:
            prev_sentiment = self._surveys_cache[symbol].weighted_sentiment
            momentum = current_sentiment - prev_sentiment

            # Limit range
            momentum = max(-100, min(100, momentum))
        else:
            momentum = 0.0  # No previous data

        return momentum

    async def get_fear_greed_component(
        self,
        symbol: str = "CRYPTO"
    ) -> AggregatedSurveyData:
        """
        Get surveys component for Fear & Greed Index

        Args:
            symbol: Asset symbol (for caching)

        Returns:
            AggregatedSurveyData object with results
        """
        try:
            survey_results = {}

            # Collect data from various sources
            tasks = []

            # Alternative.me F&G index
            tasks.append(self.fetch_alternative_me_index())

            # Simulated surveys (in reality - real APIs)
            tasks.append(self.simulate_institutional_survey())
            tasks.append(self.simulate_retail_survey())
            tasks.append(self.simulate_professional_survey())

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            source_counter = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Survey collection failed", error=str(result))
                    continue

                if result is not None:
                    survey_results[result.source] = result
                    source_counter += 1

            if not survey_results:
                raise ValueError("No survey data available")

            # Calculate weighted sentiment
            total_weight = 0.0
            weighted_sentiment_sum = 0.0

            for source, survey_result in survey_results.items():
                weight = self.source_weights.get(source, 0.1)
                type_weight = self.survey_type_weights.get(survey_result.survey_type, 0.1)
                combined_weight = weight * type_weight * survey_result.confidence

                weighted_sentiment_sum += survey_result.sentiment_score * combined_weight
                total_weight += combined_weight

            if total_weight > 0:
                weighted_sentiment = weighted_sentiment_sum / total_weight
            else:
                weighted_sentiment = 0.0

            # Separate sentiment by type
            retail_sentiment = np.mean([
                sr.sentiment_score for sr in survey_results.values()
                if sr.survey_type == SurveyType.RETAIL
            ]) if any(sr.survey_type == SurveyType.RETAIL for sr in survey_results.values()) else 0.0

            institutional_sentiment = np.mean([
                sr.sentiment_score for sr in survey_results.values()
                if sr.survey_type == SurveyType.INSTITUTIONAL
            ]) if any(sr.survey_type == SurveyType.INSTITUTIONAL for sr in survey_results.values()) else 0.0

            professional_sentiment = np.mean([
                sr.sentiment_score for sr in survey_results.values()
                if sr.survey_type == SurveyType.PROFESSIONAL
            ]) if any(sr.survey_type == SurveyType.PROFESSIONAL for sr in survey_results.values()) else 0.0

            # Additional metrics
            sentiment_divergence = self.calculate_sentiment_divergence(survey_results)
            survey_momentum = self.calculate_survey_momentum(symbol, weighted_sentiment)
            contrarian_signal = self.calculate_contrarian_signal(survey_results)

            # Fear & Greed score (convert from -100/100 to 0/100)
            fear_greed_score = (weighted_sentiment + 100) / 2

            # Overall confidence
            avg_confidence = np.mean([sr.confidence for sr in survey_results.values()])

            result = AggregatedSurveyData(
                timestamp=datetime.utcnow(),
                total_surveys=len(survey_results),
                weighted_sentiment=weighted_sentiment,
                retail_sentiment=retail_sentiment,
                institutional_sentiment=institutional_sentiment,
                professional_sentiment=professional_sentiment,
                sentiment_divergence=sentiment_divergence,
                survey_momentum=survey_momentum,
                contrarian_signal=contrarian_signal,
                fear_greed_score=fear_greed_score,
                confidence=avg_confidence,
                source_breakdown=survey_results
            )

            # Caching
            self._surveys_cache[symbol] = result

            logger.info(
                "Surveys component calculated",
                symbol=symbol,
                total_surveys=len(survey_results),
                weighted_sentiment=weighted_sentiment,
                contrarian_signal=contrarian_signal,
                fear_greed_score=fear_greed_score,
                confidence=avg_confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating surveys component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    async def get_cached_surveys(self, symbol: str) -> Optional[AggregatedSurveyData]:
        """Get cached survey data"""
        return self._surveys_cache.get(symbol)

    def get_market_sentiment_analysis(
        self,
        survey_data: AggregatedSurveyData
    ) -> Dict[str, Union[str, float]]:
        """
        Analyze market sentiment based on surveys

        Args:
            survey_data: Aggregated survey data

        Returns:
            Dict with sentiment analysis
        """
        analysis = {}

        sentiment = survey_data.weighted_sentiment
        contrarian = survey_data.contrarian_signal
        divergence = survey_data.sentiment_divergence

        # Main interpretation
        if sentiment <= self.sentiment_thresholds["extreme_fear"]:
            analysis["interpretation"] = "Extreme Fear"
            analysis["description"] = "Market participants are extremely pessimistic"
            analysis["implication"] = "Potential buying opportunity (contrarian approach)"
        elif sentiment <= self.sentiment_thresholds["fear"]:
            analysis["interpretation"] = "Fear"
            analysis["description"] = "Market sentiment is significantly negative"
            analysis["implication"] = "Caution warranted, but may present opportunities"
        elif sentiment >= self.sentiment_thresholds["extreme_greed"]:
            analysis["interpretation"] = "Extreme Greed"
            analysis["description"] = "Market participants are extremely optimistic"
            analysis["implication"] = "High risk, consider profit taking"
        elif sentiment >= self.sentiment_thresholds["greed"]:
            analysis["interpretation"] = "Greed"
            analysis["description"] = "Market sentiment is significantly positive"
            analysis["implication"] = "Monitor for signs of overheating"
        else:
            analysis["interpretation"] = "Neutral"
            analysis["description"] = "Balanced market sentiment"
            analysis["implication"] = "Normal market conditions"

        # Contrarian analysis
        if contrarian > 70:
            analysis["contrarian_signal"] = "Strong"
            analysis["contrarian_note"] = "Extreme consensus suggests potential reversal"
        elif contrarian > 40:
            analysis["contrarian_signal"] = "Moderate"
            analysis["contrarian_note"] = "Some contrarian opportunities may exist"
        else:
            analysis["contrarian_signal"] = "Weak"
            analysis["contrarian_note"] = "No significant contrarian signals"

        # Divergence analysis
        if divergence > 50:
            analysis["consensus"] = "Low"
            analysis["consensus_note"] = "Significant disagreement between participant groups"
        else:
            analysis["consensus"] = "High"
            analysis["consensus_note"] = "General agreement among participant groups"

        # Numerical metrics
        analysis["fear_greed_score"] = survey_data.fear_greed_score
        analysis["confidence_level"] = survey_data.confidence
        analysis["momentum"] = survey_data.survey_momentum

        return analysis

    async def batch_calculate(
        self,
        symbols: List[str]
    ) -> Dict[str, AggregatedSurveyData]:
        """
        Batch survey calculation for multiple symbols

        Args:
            symbols: List of symbols for analysis

        Returns:
            Dict with results for each symbol
        """
        tasks = []

        for symbol in symbols:
            task = self.get_fear_greed_component(symbol)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        surveys_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating surveys for {symbol}", error=str(result))
                continue
            surveys_results[symbol] = result

        return surveys_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
