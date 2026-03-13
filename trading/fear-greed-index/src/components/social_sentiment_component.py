"""
Social Sentiment Component for Fear & Greed Index

Analyzes social media sentiment to determine market sentiment.
Positive sentiment = greed, negative sentiment = fear.
"""

import asyncio
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import structlog

import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import aiohttp
from bs4 import BeautifulSoup

from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


class SentimentSource(Enum):
    """Social data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    NEWS = "news"
    YOUTUBE = "youtube"


class SentimentModel(Enum):
    """Sentiment analysis models"""
    FINBERT = "finbert"
    CRYPTO_SENTIMENT = "crypto_sentiment"
    GENERAL_SENTIMENT = "general_sentiment"
    ENSEMBLE = "ensemble"


@dataclass
class SentimentData:
    """Sentiment analysis data"""
    timestamp: datetime
    symbol: str
    source: SentimentSource
    total_posts: int
    positive_posts: int
    negative_posts: int
    neutral_posts: int
    sentiment_score: float      # -1 to 1
    sentiment_strength: float   # 0 to 1
    engagement_score: float     # 0 to 100
    trending_score: float       # 0 to 100
    fear_greed_score: float    # 0 to 100
    confidence: float          # 0 to 1


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment data"""
    timestamp: datetime
    symbol: str
    sources_count: int
    total_posts: int
    weighted_sentiment: float     # -1 to 1
    sentiment_momentum: float     # Change over time
    viral_coefficient: float     # Viral spread metric
    fear_greed_score: float      # 0 to 100
    confidence: float            # 0 to 1
    source_breakdown: Dict[SentimentSource, SentimentData]


class SocialSentimentComponent:
    """
    Social sentiment analysis component for Fear & Greed Index

    Collects and analyzes data from various social platforms,
    using ML models to determine market sentiment.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("social_sentiment")
        self._sentiment_cache: Dict[str, AggregatedSentiment] = {}

        # Sentiment analysis model configuration
        self.sentiment_models = {}
        self._initialize_sentiment_models()

        # Data source settings
        self.source_weights = {
            SentimentSource.TWITTER: 0.3,
            SentimentSource.REDDIT: 0.25,
            SentimentSource.TELEGRAM: 0.15,
            SentimentSource.NEWS: 0.15,
            SentimentSource.DISCORD: 0.1,
            SentimentSource.YOUTUBE: 0.05
        }

        # Keywords for search
        self.crypto_keywords = {
            'BTC': ['bitcoin', 'btc', '#bitcoin', '$btc'],
            'ETH': ['ethereum', 'eth', '#ethereum', '$eth'],
            'CRYPTO': ['crypto', 'cryptocurrency', 'blockchain', 'defi']
        }

        # Analysis settings
        self.sentiment_thresholds = {
            "extreme_fear": -0.8,
            "fear": -0.4,
            "neutral": 0.0,
            "greed": 0.4,
            "extreme_greed": 0.8
        }

        logger.info("SocialSentimentComponent initialized",
                   models=list(self.sentiment_models.keys()),
                   sources=list(self.source_weights.keys()))

    def _initialize_sentiment_models(self):
        """Initialize ML models for sentiment analysis"""
        try:
            # FinBERT for financial texts
            self.sentiment_models[SentimentModel.FINBERT] = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
            logger.info("FinBERT model loaded successfully")

        except Exception as e:
            logger.warning("Failed to load FinBERT model", error=str(e))

        try:
            # General sentiment analysis model
            self.sentiment_models[SentimentModel.GENERAL_SENTIMENT] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            logger.info("General sentiment model loaded successfully")

        except Exception as e:
            logger.warning("Failed to load general sentiment model", error=str(e))

    async def analyze_text_sentiment(
        self,
        text: str,
        model: SentimentModel = SentimentModel.ENSEMBLE
    ) -> Tuple[float, float]:
        """
        Analyze text sentiment

        Args:
            text: Text to analyze
            model: Model to use

        Returns:
            Tuple[sentiment_score (-1 to 1), confidence (0 to 1)]
        """
        try:
            if not text or len(text.strip()) == 0:
                return 0.0, 0.0

            # Text preprocessing
            cleaned_text = self._preprocess_text(text)

            if model == SentimentModel.ENSEMBLE:
                return await self._ensemble_sentiment_analysis(cleaned_text)
            elif model in self.sentiment_models:
                return await self._single_model_analysis(cleaned_text, model)
            else:
                logger.warning(f"Model {model} not available, using general sentiment")
                return await self._single_model_analysis(cleaned_text, SentimentModel.GENERAL_SENTIMENT)

        except Exception as e:
            logger.error("Error analyzing text sentiment", error=str(e), text_length=len(text))
            return 0.0, 0.0

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove mentions (@username)
        text = re.sub(r'@[A-Za-z0-9_]+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Limit length (models have limitations)
        if len(text) > 512:
            text = text[:512]

        return text

    async def _single_model_analysis(
        self,
        text: str,
        model: SentimentModel
    ) -> Tuple[float, float]:
        """Analysis using a single model"""
        if model not in self.sentiment_models:
            return 0.0, 0.0

        try:
            result = self.sentiment_models[model](text)

            # Convert result to unified format
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                label = prediction['label'].upper()
                score = prediction['score']

                # Normalize to -1 to 1 range
                if 'POSITIVE' in label or 'POS' in label:
                    sentiment_score = score
                elif 'NEGATIVE' in label or 'NEG' in label:
                    sentiment_score = -score
                else:  # NEUTRAL
                    sentiment_score = 0.0

                return sentiment_score, score
            else:
                return 0.0, 0.0

        except Exception as e:
            logger.error("Error in single model analysis", model=model, error=str(e))
            return 0.0, 0.0

    async def _ensemble_sentiment_analysis(
        self,
        text: str
    ) -> Tuple[float, float]:
        """Ensemble analysis using multiple models"""
        sentiments = []
        confidences = []

        for model in self.sentiment_models.keys():
            sentiment, confidence = await self._single_model_analysis(text, model)
            if confidence > 0:
                sentiments.append(sentiment)
                confidences.append(confidence)

        if not sentiments:
            return 0.0, 0.0

        # Weighted average by confidence
        total_confidence = sum(confidences)
        if total_confidence > 0:
            weighted_sentiment = sum(s * c for s, c in zip(sentiments, confidences)) / total_confidence
            avg_confidence = total_confidence / len(confidences)
        else:
            weighted_sentiment = sum(sentiments) / len(sentiments)
            avg_confidence = 0.5

        return weighted_sentiment, avg_confidence

    async def collect_twitter_data(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> List[Dict]:
        """
        Collect data from Twitter (simulation - requires API keys)

        Args:
            symbol: Cryptocurrency symbol
            hours_back: Number of hours to look back

        Returns:
            List of tweets with metadata
        """
        try:
            # In a real implementation, this would use the Twitter API
            # For demonstration, we return simulated data

            keywords = self.crypto_keywords.get(symbol.upper(), [symbol.lower()])

            # Simulated tweets (in reality - Twitter API v2)
            simulated_tweets = [
                {
                    "id": f"tweet_{i}",
                    "text": f"Great news about {symbol}! Going to the moon! 🚀",
                    "created_at": datetime.utcnow() - timedelta(hours=i),
                    "public_metrics": {"retweet_count": 10, "like_count": 25, "reply_count": 5},
                    "lang": "en"
                }
                for i in range(10)
            ]

            logger.info(f"Collected {len(simulated_tweets)} tweets for {symbol}")
            self.metrics.record_collection("twitter_tweets", len(simulated_tweets))

            return simulated_tweets

        except Exception as e:
            logger.error("Error collecting Twitter data", symbol=symbol, error=str(e))
            self.metrics.record_error("twitter_collection_error")
            return []

    async def collect_reddit_data(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> List[Dict]:
        """
        Collect data from Reddit

        Args:
            symbol: Cryptocurrency symbol
            hours_back: Number of hours to look back

        Returns:
            List of Reddit posts with metadata
        """
        try:
            # Simulated Reddit posts (in reality - Reddit API)
            subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'cryptomarkets']

            simulated_posts = [
                {
                    "id": f"reddit_{i}",
                    "title": f"Discussion about {symbol} future prospects",
                    "selftext": f"I think {symbol} has great potential because of recent developments...",
                    "created_utc": datetime.utcnow() - timedelta(hours=i),
                    "score": 50 - i,
                    "num_comments": 15,
                    "subreddit": subreddits[i % len(subreddits)]
                }
                for i in range(15)
            ]

            logger.info(f"Collected {len(simulated_posts)} Reddit posts for {symbol}")
            self.metrics.record_collection("reddit_posts", len(simulated_posts))

            return simulated_posts

        except Exception as e:
            logger.error("Error collecting Reddit data", symbol=symbol, error=str(e))
            self.metrics.record_error("reddit_collection_error")
            return []

    async def collect_news_sentiment(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> List[Dict]:
        """
        Collect news articles

        Args:
            symbol: Cryptocurrency symbol
            hours_back: Number of hours to look back

        Returns:
            List of news articles with metadata
        """
        try:
            # Simulated news (in reality - News API, RSS feeds)
            simulated_news = [
                {
                    "id": f"news_{i}",
                    "title": f"{symbol} Shows Strong Performance in Recent Market Analysis",
                    "description": f"Market analysts are optimistic about {symbol} due to...",
                    "publishedAt": datetime.utcnow() - timedelta(hours=i*2),
                    "source": {"name": f"Crypto News {i}"},
                    "url": f"https://cryptonews{i}.com/article"
                }
                for i in range(8)
            ]

            logger.info(f"Collected {len(simulated_news)} news articles for {symbol}")
            self.metrics.record_collection("news_articles", len(simulated_news))

            return simulated_news

        except Exception as e:
            logger.error("Error collecting news data", symbol=symbol, error=str(e))
            self.metrics.record_error("news_collection_error")
            return []

    async def analyze_source_sentiment(
        self,
        symbol: str,
        source: SentimentSource,
        data: List[Dict]
    ) -> SentimentData:
        """
        Analyze sentiment for a specific source

        Args:
            symbol: Cryptocurrency symbol
            source: Data source
            data: Collected data

        Returns:
            SentimentData object with results
        """
        if not data:
            return SentimentData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                source=source,
                total_posts=0,
                positive_posts=0,
                negative_posts=0,
                neutral_posts=0,
                sentiment_score=0.0,
                sentiment_strength=0.0,
                engagement_score=0.0,
                trending_score=0.0,
                fear_greed_score=50.0,
                confidence=0.0
            )

        try:
            sentiments = []
            strengths = []
            engagements = []

            positive_count = 0
            negative_count = 0
            neutral_count = 0

            for item in data:
                # Extract text depending on source
                if source == SentimentSource.TWITTER:
                    text = item.get('text', '')
                    engagement = item.get('public_metrics', {})
                    engagement_value = (
                        engagement.get('like_count', 0) +
                        engagement.get('retweet_count', 0) * 2 +
                        engagement.get('reply_count', 0)
                    )
                elif source == SentimentSource.REDDIT:
                    text = f"{item.get('title', '')} {item.get('selftext', '')}"
                    engagement_value = item.get('score', 0) + item.get('num_comments', 0)
                elif source == SentimentSource.NEWS:
                    text = f"{item.get('title', '')} {item.get('description', '')}"
                    engagement_value = 1  # Base value for news
                else:
                    text = str(item)
                    engagement_value = 1

                # Analyze sentiment
                sentiment, strength = await self.analyze_text_sentiment(text)

                sentiments.append(sentiment)
                strengths.append(strength)
                engagements.append(engagement_value)

                # Count categories
                if sentiment > 0.1:
                    positive_count += 1
                elif sentiment < -0.1:
                    negative_count += 1
                else:
                    neutral_count += 1

            # Aggregated metrics
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            avg_strength = np.mean(strengths) if strengths else 0.0
            total_engagement = sum(engagements)

            # Normalized scores
            engagement_score = min(100, (total_engagement / len(data)) * 10) if data else 0
            trending_score = min(100, len(data) * 5)  # Simple trend metric

            # Fear & Greed score based on sentiment
            fear_greed_score = self._convert_sentiment_to_fear_greed(avg_sentiment)

            # Confidence based on data volume and signal strength
            confidence = min(1.0, (len(data) / 50) * avg_strength)

            result = SentimentData(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                source=source,
                total_posts=len(data),
                positive_posts=positive_count,
                negative_posts=negative_count,
                neutral_posts=neutral_count,
                sentiment_score=avg_sentiment,
                sentiment_strength=avg_strength,
                engagement_score=engagement_score,
                trending_score=trending_score,
                fear_greed_score=fear_greed_score,
                confidence=confidence
            )

            logger.info(
                "Source sentiment analyzed",
                symbol=symbol,
                source=source.value,
                total_posts=len(data),
                sentiment_score=avg_sentiment,
                fear_greed_score=fear_greed_score
            )

            return result

        except Exception as e:
            logger.error("Error analyzing source sentiment",
                        symbol=symbol, source=source.value, error=str(e))
            raise

    def _convert_sentiment_to_fear_greed(self, sentiment_score: float) -> float:
        """
        Convert sentiment score to Fear & Greed scale (0-100)

        Args:
            sentiment_score: Sentiment score (-1 to 1)

        Returns:
            Fear & Greed score (0-100)
        """
        # Linear conversion from [-1, 1] to [0, 100]
        return max(0, min(100, (sentiment_score + 1) * 50))

    async def get_fear_greed_component(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> AggregatedSentiment:
        """
        Get aggregated social sentiment component for Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol
            hours_back: Number of hours for analysis

        Returns:
            AggregatedSentiment object with results
        """
        try:
            source_data = {}
            source_sentiments = {}

            # Collect data from all sources in parallel
            tasks = [
                self.collect_twitter_data(symbol, hours_back),
                self.collect_reddit_data(symbol, hours_back),
                self.collect_news_sentiment(symbol, hours_back)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            sources = [SentimentSource.TWITTER, SentimentSource.REDDIT, SentimentSource.NEWS]

            for source, result in zip(sources, results):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to collect data from {source.value}", error=str(result))
                    source_data[source] = []
                else:
                    source_data[source] = result

            # Analyze sentiment for each source
            for source, data in source_data.items():
                if data:
                    sentiment_data = await self.analyze_source_sentiment(symbol, source, data)
                    source_sentiments[source] = sentiment_data

            # Aggregate results
            if not source_sentiments:
                # No data - return neutral values
                return AggregatedSentiment(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    sources_count=0,
                    total_posts=0,
                    weighted_sentiment=0.0,
                    sentiment_momentum=0.0,
                    viral_coefficient=0.0,
                    fear_greed_score=50.0,
                    confidence=0.0,
                    source_breakdown={}
                )

            # Weighted average across sources
            total_weight = 0.0
            weighted_sentiment_sum = 0.0
            total_posts = 0

            for source, sentiment_data in source_sentiments.items():
                weight = self.source_weights.get(source, 0.1)
                contribution = sentiment_data.sentiment_score * weight * sentiment_data.confidence

                weighted_sentiment_sum += contribution
                total_weight += weight * sentiment_data.confidence
                total_posts += sentiment_data.total_posts

            if total_weight > 0:
                weighted_sentiment = weighted_sentiment_sum / total_weight
            else:
                weighted_sentiment = 0.0

            # Calculate additional metrics
            viral_coefficient = self._calculate_viral_coefficient(source_sentiments)
            sentiment_momentum = self._calculate_sentiment_momentum(symbol, weighted_sentiment)

            # Final Fear & Greed score
            fear_greed_score = self._convert_sentiment_to_fear_greed(weighted_sentiment)

            # Overall confidence
            avg_confidence = np.mean([s.confidence for s in source_sentiments.values()])

            result = AggregatedSentiment(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                sources_count=len(source_sentiments),
                total_posts=total_posts,
                weighted_sentiment=weighted_sentiment,
                sentiment_momentum=sentiment_momentum,
                viral_coefficient=viral_coefficient,
                fear_greed_score=fear_greed_score,
                confidence=avg_confidence,
                source_breakdown=source_sentiments
            )

            # Caching
            self._sentiment_cache[symbol] = result

            logger.info(
                "Social sentiment component calculated",
                symbol=symbol,
                sources_count=len(source_sentiments),
                total_posts=total_posts,
                weighted_sentiment=weighted_sentiment,
                fear_greed_score=fear_greed_score,
                confidence=avg_confidence
            )

            self.metrics.record_calculation("component_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating social sentiment component",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("component_calculation_error")
            raise

    def _calculate_viral_coefficient(
        self,
        source_sentiments: Dict[SentimentSource, SentimentData]
    ) -> float:
        """Calculate content virality coefficient"""
        if not source_sentiments:
            return 0.0

        # Simple metric based on engagement and trending scores
        engagement_scores = [s.engagement_score for s in source_sentiments.values()]
        trending_scores = [s.trending_score for s in source_sentiments.values()]

        avg_engagement = np.mean(engagement_scores) if engagement_scores else 0
        avg_trending = np.mean(trending_scores) if trending_scores else 0

        # Viral coefficient (0-100)
        viral_coefficient = min(100, (avg_engagement + avg_trending) / 2)

        return viral_coefficient

    def _calculate_sentiment_momentum(
        self,
        symbol: str,
        current_sentiment: float
    ) -> float:
        """Calculate sentiment change over time"""
        # In a real implementation, this would compare with previous periods
        # For demonstration, we return the change from cache

        if symbol in self._sentiment_cache:
            prev_sentiment = self._sentiment_cache[symbol].weighted_sentiment
            momentum = current_sentiment - prev_sentiment
        else:
            momentum = 0.0

        return momentum

    async def get_cached_sentiment(self, symbol: str) -> Optional[AggregatedSentiment]:
        """Get cached sentiment data"""
        return self._sentiment_cache.get(symbol)

    async def batch_calculate(
        self,
        symbols: List[str],
        hours_back: int = 24
    ) -> Dict[str, AggregatedSentiment]:
        """
        Batch sentiment calculation for multiple symbols

        Args:
            symbols: List of symbols for analysis
            hours_back: Number of hours for analysis

        Returns:
            Dict with results for each symbol
        """
        tasks = []

        for symbol in symbols:
            task = self.get_fear_greed_component(symbol, hours_back)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        sentiment_results = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error calculating sentiment for {symbol}", error=str(result))
                continue
            sentiment_results[symbol] = result

        return sentiment_results

    def get_metrics(self) -> Dict[str, float]:
        """Get component performance metrics"""
        return self.metrics.get_metrics()
