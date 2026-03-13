"""
Sentiment Analysis Integration for Trading Environments
enterprise patterns for comprehensive market sentiment

Advanced sentiment analysis integration:
- Multi-source sentiment aggregation  
- Real-time sentiment streaming
- Historical sentiment data
- Sentiment-based signals generation
- Social media sentiment tracking
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import logging
import json
import warnings


class SentimentSource(Enum):
    """Sentiment data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"  
    NEWS = "news"
    FEAR_GREED_INDEX = "fear_greed"
    SOCIAL_VOLUME = "social_volume"
    WHALE_ALERTS = "whale_alerts"
    DERIVATIVES = "derivatives"
    ON_CHAIN = "on_chain"


@dataclass
class SentimentScore:
    """Individual sentiment score"""
    source: SentimentSource
    asset: str
    score: float          # Normalized [-1, 1] or [0, 100] depending on source
    confidence: float     # [0, 1] confidence in the score
    timestamp: float
    raw_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    
    # Data sources
    enabled_sources: List[SentimentSource] = field(default_factory=lambda: [
        SentimentSource.TWITTER,
        SentimentSource.REDDIT,
        SentimentSource.NEWS,
        SentimentSource.FEAR_GREED_INDEX
    ])
    
    # Update frequencies (seconds)
    twitter_update_freq: int = 60
    reddit_update_freq: int = 300  
    news_update_freq: int = 600
    fear_greed_update_freq: int = 3600
    
    # Aggregation settings
    aggregation_method: str = "weighted_average"  # weighted_average, simple_average, max, min
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "twitter": 0.3,
        "reddit": 0.25,
        "news": 0.35,
        "fear_greed": 0.1
    })
    
    # Historical data
    sentiment_history_length: int = 100
    enable_sentiment_history: bool = True
    
    # Signal generation
    enable_sentiment_signals: bool = True
    bullish_threshold: float = 0.6
    bearish_threshold: float = -0.6
    signal_smoothing_window: int = 5
    
    # Performance optimization
    cache_sentiment_data: bool = True
    async_updates: bool = True
    batch_processing: bool = True


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer for crypto trading
    
    Aggregates sentiment from multiple sources and generates trading signals
    """
    
    def __init__(
        self,
        assets: List[str],
        sources: Optional[List[str]] = None,
        config: Optional[SentimentConfig] = None
    ):
        self.assets = assets
        self.config = config or SentimentConfig()
        self.logger = logging.getLogger(__name__)
        
        # Configure sources
        if sources:
            source_enums = [SentimentSource(s) for s in sources if s in [e.value for e in SentimentSource]]
            self.config.enabled_sources = source_enums
        
        # Sentiment storage
        self.current_sentiment = {}  # asset -> {source -> score}
        self.sentiment_history = {}  # asset -> deque of historical scores
        self.sentiment_signals = {}  # asset -> current signal
        
        # Initialize storage
        for asset in assets:
            self.current_sentiment[asset] = {}
            self.sentiment_history[asset] = deque(maxlen=self.config.sentiment_history_length)
            self.sentiment_signals[asset] = 0.0
        
        # Source handlers
        self.source_handlers = {
            SentimentSource.TWITTER: self._get_twitter_sentiment,
            SentimentSource.REDDIT: self._get_reddit_sentiment,
            SentimentSource.NEWS: self._get_news_sentiment,
            SentimentSource.FEAR_GREED_INDEX: self._get_fear_greed_index,
            SentimentSource.SOCIAL_VOLUME: self._get_social_volume,
            SentimentSource.WHALE_ALERTS: self._get_whale_sentiment,
            SentimentSource.DERIVATIVES: self._get_derivatives_sentiment,
            SentimentSource.ON_CHAIN: self._get_onchain_sentiment
        }
        
        # Update tracking
        self.last_updates = {source: 0.0 for source in self.config.enabled_sources}
        
        # Synthetic data mode for development/testing
        self.synthetic_mode = True  # Will be overridden in production
        
        self.logger.info(f"Sentiment analyzer initialized for {len(assets)} assets with {len(self.config.enabled_sources)} sources")
    
    def get_current_sentiment(self) -> Dict[str, Dict[str, float]]:
        """Get current aggregated sentiment scores"""
        
        # Update sentiment data if needed
        self._update_sentiment_data()
        
        result = {}
        for asset in self.assets:
            # Aggregate scores from all sources
            aggregated_scores = self._aggregate_sentiment_scores(asset)
            result[asset] = aggregated_scores
        
        return result
    
    async def async_get_sentiment(self) -> Dict[str, Dict[str, float]]:
        """Async version of get_current_sentiment"""
        
        # Update asynchronously
        await self._async_update_sentiment_data()
        
        result = {}
        for asset in self.assets:
            aggregated_scores = self._aggregate_sentiment_scores(asset)
            result[asset] = aggregated_scores
        
        return result
    
    def _update_sentiment_data(self) -> None:
        """Update sentiment data from all sources"""
        
        current_time = time.time()
        
        for source in self.config.enabled_sources:
            # Check if update is needed
            last_update = self.last_updates[source]
            update_freq = self._get_update_frequency(source)
            
            if current_time - last_update >= update_freq:
                try:
                    self._update_source_sentiment(source)
                    self.last_updates[source] = current_time
                except Exception as e:
                    self.logger.error(f"Error updating {source.value} sentiment: {e}")
    
    async def _async_update_sentiment_data(self) -> None:
        """Async update of sentiment data"""
        
        current_time = time.time()
        update_tasks = []
        
        for source in self.config.enabled_sources:
            last_update = self.last_updates[source]
            update_freq = self._get_update_frequency(source)
            
            if current_time - last_update >= update_freq:
                task = asyncio.create_task(self._async_update_source_sentiment(source))
                update_tasks.append((source, task))
        
        # Wait for all updates
        for source, task in update_tasks:
            try:
                await task
                self.last_updates[source] = current_time
            except Exception as e:
                self.logger.error(f"Error updating {source.value} sentiment: {e}")
    
    def _update_source_sentiment(self, source: SentimentSource) -> None:
        """Update sentiment data from specific source"""
        
        if source in self.source_handlers:
            handler = self.source_handlers[source]
            
            for asset in self.assets:
                try:
                    sentiment_data = handler(asset)
                    if sentiment_data:
                        self._store_sentiment_score(asset, source, sentiment_data)
                except Exception as e:
                    self.logger.warning(f"Error getting {source.value} sentiment for {asset}: {e}")
    
    async def _async_update_source_sentiment(self, source: SentimentSource) -> None:
        """Async update sentiment from specific source"""
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._update_source_sentiment, source)
    
    def _store_sentiment_score(
        self,
        asset: str,
        source: SentimentSource,
        sentiment_data: Dict[str, Any]
    ) -> None:
        """Store sentiment score"""
        
        score = sentiment_data.get("score", 0.0)
        confidence = sentiment_data.get("confidence", 0.5)
        
        # Normalize score to [-1, 1] if needed
        if source == SentimentSource.FEAR_GREED_INDEX:
            # Fear & Greed is [0, 100], normalize to [-1, 1]
            score = (score - 50.0) / 50.0
        
        # Store current sentiment
        self.current_sentiment[asset][source.value] = {
            "score": score,
            "confidence": confidence,
            "timestamp": time.time(),
            "raw_data": sentiment_data
        }
        
        # Store in history if enabled
        if self.config.enable_sentiment_history:
            self.sentiment_history[asset].append({
                "source": source.value,
                "score": score,
                "confidence": confidence,
                "timestamp": time.time()
            })
    
    def _aggregate_sentiment_scores(self, asset: str) -> Dict[str, float]:
        """Aggregate sentiment scores from all sources"""
        
        asset_sentiment = self.current_sentiment.get(asset, {})
        
        if not asset_sentiment:
            return {
                "overall": 0.0,
                "twitter": 0.0,
                "reddit": 0.0,
                "news": 0.0,
                "fear_greed": 0.0,
                "confidence": 0.0
            }
        
        # Individual source scores
        individual_scores = {}
        weighted_scores = []
        total_weight = 0.0
        
        for source_name, data in asset_sentiment.items():
            score = data.get("score", 0.0)
            confidence = data.get("confidence", 0.5)
            
            individual_scores[source_name] = score
            
            # Weighted aggregation
            source_weight = self.config.source_weights.get(source_name, 0.25)
            weighted_score = score * confidence * source_weight
            weighted_scores.append(weighted_score)
            total_weight += confidence * source_weight
        
        # Calculate overall sentiment
        if total_weight > 0:
            overall_sentiment = sum(weighted_scores) / total_weight
        else:
            overall_sentiment = 0.0
        
        # Apply smoothing if enabled
        if self.config.enable_sentiment_signals:
            overall_sentiment = self._apply_sentiment_smoothing(asset, overall_sentiment)
            self.sentiment_signals[asset] = overall_sentiment
        
        # Build result
        result = {
            "overall": overall_sentiment,
            "confidence": min(total_weight, 1.0),
            **individual_scores
        }
        
        # Ensure all expected keys exist
        for source in ["twitter", "reddit", "news", "fear_greed"]:
            if source not in result:
                result[source] = 0.0
        
        return result
    
    def _apply_sentiment_smoothing(self, asset: str, new_score: float) -> float:
        """Apply smoothing to sentiment scores"""
        
        if len(self.sentiment_history[asset]) < self.config.signal_smoothing_window:
            return new_score
        
        # Get recent sentiment scores
        recent_scores = [
            entry["score"] for entry in list(self.sentiment_history[asset])[-self.config.signal_smoothing_window:]
        ]
        recent_scores.append(new_score)
        
        # Apply exponential moving average
        alpha = 2.0 / (len(recent_scores) + 1)
        smoothed_score = recent_scores[0]
        
        for score in recent_scores[1:]:
            smoothed_score = alpha * score + (1 - alpha) * smoothed_score
        
        return float(smoothed_score)
    
    def _get_update_frequency(self, source: SentimentSource) -> int:
        """Get update frequency for source"""
        
        freq_map = {
            SentimentSource.TWITTER: self.config.twitter_update_freq,
            SentimentSource.REDDIT: self.config.reddit_update_freq,
            SentimentSource.NEWS: self.config.news_update_freq,
            SentimentSource.FEAR_GREED_INDEX: self.config.fear_greed_update_freq
        }
        
        return freq_map.get(source, 300)  # Default 5 minutes
    
    # Source-specific handlers (synthetic data for development)
    def _get_twitter_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get Twitter sentiment (synthetic data)"""
        
        if self.synthetic_mode:
            # Generate realistic synthetic sentiment
            base_sentiment = np.random.normal(0.0, 0.3)
            noise = np.random.normal(0.0, 0.1)
            
            return {
                "score": np.clip(base_sentiment + noise, -1.0, 1.0),
                "confidence": np.random.uniform(0.6, 0.9),
                "tweet_count": np.random.randint(100, 1000),
                "engagement_rate": np.random.uniform(0.02, 0.08)
            }
        
        # Real Twitter API integration would go here
        return None
    
    def _get_reddit_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get Reddit sentiment (synthetic data)"""
        
        if self.synthetic_mode:
            # Reddit tends to be more extreme
            base_sentiment = np.random.normal(0.0, 0.4)
            
            return {
                "score": np.clip(base_sentiment, -1.0, 1.0),
                "confidence": np.random.uniform(0.5, 0.8),
                "post_count": np.random.randint(10, 100),
                "upvote_ratio": np.random.uniform(0.6, 0.95)
            }
        
        return None
    
    def _get_news_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get News sentiment (synthetic data)"""
        
        if self.synthetic_mode:
            # News sentiment tends to be more stable
            base_sentiment = np.random.normal(0.0, 0.25)
            
            return {
                "score": np.clip(base_sentiment, -1.0, 1.0),
                "confidence": np.random.uniform(0.7, 0.95),
                "article_count": np.random.randint(5, 50),
                "avg_sentiment": base_sentiment
            }
        
        return None
    
    def _get_fear_greed_index(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get Fear & Greed Index (synthetic data)"""
        
        if self.synthetic_mode:
            # Fear & Greed index [0, 100]
            base_index = np.random.uniform(20, 80)
            
            return {
                "score": base_index,
                "confidence": 0.9,  # High confidence in this metric
                "classification": self._classify_fear_greed(base_index)
            }
        
        return None
    
    def _get_social_volume(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get social volume metrics"""
        
        if self.synthetic_mode:
            return {
                "score": np.random.uniform(-0.2, 0.2),  # Volume doesn't directly indicate sentiment
                "confidence": 0.3,
                "volume": np.random.randint(1000, 50000)
            }
        
        return None
    
    def _get_whale_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get whale movement sentiment"""
        
        if self.synthetic_mode:
            # Whale movements can indicate sentiment
            whale_activity = np.random.uniform(-0.5, 0.5)
            
            return {
                "score": whale_activity,
                "confidence": 0.6,
                "large_transactions": np.random.randint(5, 50)
            }
        
        return None
    
    def _get_derivatives_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get derivatives market sentiment"""
        
        if self.synthetic_mode:
            # Options flow, futures positioning, etc.
            derivatives_sentiment = np.random.normal(0.0, 0.3)
            
            return {
                "score": np.clip(derivatives_sentiment, -1.0, 1.0),
                "confidence": 0.7,
                "put_call_ratio": np.random.uniform(0.5, 2.0)
            }
        
        return None
    
    def _get_onchain_sentiment(self, asset: str) -> Optional[Dict[str, Any]]:
        """Get on-chain sentiment indicators"""
        
        if self.synthetic_mode:
            # Network activity, holder behavior, etc.
            onchain_sentiment = np.random.normal(0.0, 0.2)
            
            return {
                "score": np.clip(onchain_sentiment, -1.0, 1.0),
                "confidence": 0.8,
                "active_addresses": np.random.randint(10000, 100000),
                "hodl_ratio": np.random.uniform(0.6, 0.9)
            }
        
        return None
    
    def _classify_fear_greed(self, index_value: float) -> str:
        """Classify Fear & Greed index value"""
        
        if index_value >= 75:
            return "Extreme Greed"
        elif index_value >= 55:
            return "Greed"
        elif index_value >= 45:
            return "Neutral"
        elif index_value >= 25:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def get_sentiment_signals(self) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals from sentiment"""
        
        signals = {}
        
        for asset in self.assets:
            current_sentiment = self.sentiment_signals.get(asset, 0.0)
            
            # Generate signal
            if current_sentiment >= self.config.bullish_threshold:
                signal_strength = min((current_sentiment - self.config.bullish_threshold) / (1.0 - self.config.bullish_threshold), 1.0)
                signals[asset] = {
                    "signal": "bullish",
                    "strength": signal_strength,
                    "sentiment_score": current_sentiment,
                    "confidence": self._get_signal_confidence(asset, current_sentiment)
                }
            elif current_sentiment <= self.config.bearish_threshold:
                signal_strength = min((self.config.bearish_threshold - current_sentiment) / (1.0 + self.config.bearish_threshold), 1.0)
                signals[asset] = {
                    "signal": "bearish", 
                    "strength": signal_strength,
                    "sentiment_score": current_sentiment,
                    "confidence": self._get_signal_confidence(asset, current_sentiment)
                }
            else:
                signals[asset] = {
                    "signal": "neutral",
                    "strength": 0.0,
                    "sentiment_score": current_sentiment,
                    "confidence": 0.5
                }
        
        return signals
    
    def _get_signal_confidence(self, asset: str, sentiment_score: float) -> float:
        """Calculate confidence in sentiment signal"""
        
        asset_data = self.current_sentiment.get(asset, {})
        
        if not asset_data:
            return 0.0
        
        # Average confidence from all sources
        confidences = [data.get("confidence", 0.0) for data in asset_data.values()]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Adjust based on sentiment extremity
        extremity_bonus = min(abs(sentiment_score), 0.2)  # Up to 20% bonus for extreme sentiment
        
        return min(avg_confidence + extremity_bonus, 1.0)
    
    def get_sentiment_statistics(self) -> Dict[str, Any]:
        """Get sentiment analysis statistics"""
        
        stats = {
            "assets_tracked": len(self.assets),
            "sources_enabled": [s.value for s in self.config.enabled_sources],
            "total_updates": sum(1 for t in self.last_updates.values() if t > 0),
            "avg_sentiment_by_asset": {},
            "source_reliability": {}
        }
        
        # Calculate average sentiment by asset
        for asset in self.assets:
            if self.sentiment_history[asset]:
                scores = [entry["score"] for entry in self.sentiment_history[asset]]
                stats["avg_sentiment_by_asset"][asset] = {
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "count": len(scores)
                }
        
        return stats
    
    def reset(self) -> None:
        """Reset sentiment analyzer state"""
        
        for asset in self.assets:
            self.current_sentiment[asset].clear()
            self.sentiment_history[asset].clear()
            self.sentiment_signals[asset] = 0.0
        
        self.last_updates = {source: 0.0 for source in self.config.enabled_sources}


__all__ = [
    "SentimentSource",
    "SentimentScore",
    "SentimentConfig", 
    "SentimentAnalyzer"
]