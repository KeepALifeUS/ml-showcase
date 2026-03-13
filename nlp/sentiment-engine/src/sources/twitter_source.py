"""
Twitter/X Sentiment Data Source for ML-Framework ML Sentiment Engine

Enterprise-grade Twitter data collection with and circuit breaker protection.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

import aiohttp
import tweepy
from tweepy.asynchronous import AsyncClient

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TwitterCircuitBreaker:
 """Circuit breaker for Twitter API"""

 def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
 self.failure_threshold = failure_threshold
 self.recovery_timeout = recovery_timeout
 self.failure_count = 0
 self.last_failure_time = None
 self.state = "closed" # closed, open, half-open

 def can_execute(self) -> bool:
 """Check if the request can be executed"""
 if self.state == "closed":
 return True
 elif self.state == "open":
 if time.time - self.last_failure_time > self.recovery_timeout:
 self.state = "half-open"
 return True
 return False
 else: # half-open
 return True

 def record_success(self):
 """Record a successful request"""
 self.failure_count = 0
 self.state = "closed"

 def record_failure(self):
 """Registration unsuccessful request"""
 self.failure_count += 1
 self.last_failure_time = time.time

 if self.failure_count >= self.failure_threshold:
 self.state = "open"
 logger.warning(
 "Twitter circuit breaker opened",
 failure_count=self.failure_count,
 threshold=self.failure_threshold
 )


class TwitterRateLimiter:
 """Rate limiter for Twitter API"""

 def __init__(self, requests_per_window: int = 300, window_minutes: int = 15):
 self.requests_per_window = requests_per_window
 self.window_seconds = window_minutes * 60
 self.requests = []

 async def acquire(self):
 """Acquire permission for the request"""
 now = time.time

 # Remove old requests
 self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]

 if len(self.requests) >= self.requests_per_window:
 # Waiting to end window
 oldest_request = min(self.requests)
 sleep_time = self.window_seconds - (now - oldest_request) + 1

 logger.warning(
 "Twitter rate limit reached, sleeping",
 sleep_time=sleep_time,
 requests_count=len(self.requests)
 )

 await asyncio.sleep(sleep_time)

 self.requests.append(now)


class TwitterSentimentSource:
 """
 Enterprise-grade Twitter data source for sentiment analysis

 Features:
 - Async streaming support
 - Circuit breaker protection
 - Rate limiting compliance
 - Real-time and historical data
 - Crypto-specific filtering
 - Text preprocessing
 """

 def __init__(self):
 """Initialize the Twitter data source"""
 config = get_config

 # Twitter API credentials
 self.bearer_token = config.social.twitter_bearer_token
 self.api_key = config.social.twitter_api_key
 self.api_secret = config.social.twitter_api_secret
 self.access_token = config.social.twitter_access_token
 self.access_token_secret = config.social.twitter_access_token_secret

 # Async client
 self.client: Optional[AsyncClient] = None

 # Protection mechanisms
 self.circuit_breaker = TwitterCircuitBreaker
 self.rate_limiter = TwitterRateLimiter

 # Crypto symbols and keywords
 self.crypto_symbols = set(get_crypto_symbols)
 self.crypto_keywords = set(get_crypto_keywords)

 # Performance metrics
 self.tweets_processed = 0
 self.api_calls_made = 0
 self.last_error = None

 async def initialize(self):
 """Initialize the Twitter API client"""
 if not self.bearer_token:
 raise ValueError("Twitter Bearer Token is required")

 self.client = AsyncClient(
 bearer_token=self.bearer_token,
 consumer_key=self.api_key,
 consumer_secret=self.api_secret,
 access_token=self.access_token,
 access_token_secret=self.access_token_secret,
 wait_on_rate_limit=True
 )

 logger.info("Twitter source initialized")

 async def cleanup(self):
 """Cleanup resources"""
 if self.client:
 await self.client.session.close
 logger.info("Twitter source cleaned up")

 def _extract_crypto_mentions(self, text: str) -> Set[str]:
 """
 Extract cryptocurrency mentions from text

 Args:
 text: Text to analyze

 Returns:
 Set[str]: Found cryptocurrency symbols
 """
 mentioned_symbols = set
 text_upper = text.upper

 # Search symbols cryptocurrencies
 for symbol in self.crypto_symbols:
 patterns = [
 rf'\b{symbol}\b', # Exact match
 rf'\${symbol}\b', # With prefix $
 rf'#{symbol}\b', # Hashtag
 ]

 for pattern in patterns:
 if re.search(pattern, text_upper):
 mentioned_symbols.add(symbol)
 break

 return mentioned_symbols

 def _is_crypto_relevant(self, text: str) -> bool:
 """
 Check text relevance for crypto analysis

 Args:
 text: Text to check

 Returns:
 bool: True if text relevant
 """
 text_lower = text.lower

 # Check for keywords
 for keyword in self.crypto_keywords:
 if keyword in text_lower:
 return True

 # Check for cryptocurrency symbols
 if self._extract_crypto_mentions(text):
 return True

 return False

 async def _make_api_call(self, api_call):
 """
 Execution API call with protection mechanisms

 Args:
 api_call: Function for call API

 Returns:
 Any: API call result
 """
 if not self.circuit_breaker.can_execute:
 raise Exception("Twitter circuit breaker is open")

 await self.rate_limiter.acquire

 try:
 start_time = time.time
 result = await api_call
 execution_time = (time.time - start_time) * 1000

 self.circuit_breaker.record_success
 self.api_calls_made += 1

 logger.debug(
 "Twitter API call successful",
 execution_time_ms=execution_time,
 total_calls=self.api_calls_made
 )

 return result

 except Exception as e:
 self.circuit_breaker.record_failure
 self.last_error = e

 logger.error(
 "Twitter API call failed",
 error=e,
 api_calls=self.api_calls_made
 )
 raise

 async def search_tweets(
 self,
 symbols: List[str] = None,
 limit: int = 100,
 hours_back: int = 24
 ) -> List[Dict[str, Any]]:
 """
 Search tweets by symbol cryptocurrencies

 Args:
 symbols: List symbols for search
 limit: Maximum number of tweets
 hours_back: Search period in hours

 Returns:
 List[Dict[str, Any]]: List processed tweets
 """
 if not self.client:
 await self.initialize

 if not symbols:
 symbols = list(self.crypto_symbols)[:5] # Top-5 by default

 all_tweets = []

 for symbol in symbols:
 try:
 # Building search request
 query_parts = [
 f"${symbol}",
 f"#{symbol}",
 f"{symbol} crypto",
 f"{symbol} bitcoin"
 ]
 query = " OR ".join(query_parts)
 query += " -is:retweet lang:en" # Excluding retweets, only English

 # Temporary marks
 end_time = datetime.utcnow
 start_time = end_time - timedelta(hours=hours_back)

 # API call
 async def api_call:
 return await self.client.search_recent_tweets(
 query=query,
 max_results=min(limit, 100), # Twitter API limit
 start_time=start_time,
 end_time=end_time,
 tweet_fields=["created_at", "author_id", "public_metrics", "context_annotations"],
 user_fields=["username", "verified", "public_metrics"]
 )

 response = await self._make_api_call(api_call)

 if not response.data:
 continue

 # Process received tweets
 for tweet in response.data:
 processed_tweet = await self._process_tweet(tweet, symbol)
 if processed_tweet and self._is_crypto_relevant(processed_tweet["text"]):
 all_tweets.append(processed_tweet)

 logger.info(
 "Tweets fetched for symbol",
 symbol=symbol,
 tweets_count=len(response.data),
 relevant_tweets=len([t for t in all_tweets if t.get("symbol") == symbol])
 )

 # Pause between symbols
 await asyncio.sleep(1)

 except Exception as e:
 logger.error(f"Error fetching tweets for symbol {symbol}", error=e)
 continue

 self.tweets_processed += len(all_tweets)

 logger.info(
 "Twitter search completed",
 symbols=symbols,
 total_tweets=len(all_tweets),
 processed_total=self.tweets_processed
 )

 return all_tweets

 async def _process_tweet(self, tweet, symbol: str) -> Optional[Dict[str, Any]]:
 """
 Process a single tweet

 Args:
 tweet: Object tweet from Twitter API
 symbol: Connected symbol cryptocurrency

 Returns:
 Optional[Dict[str, Any]]: Processed tweet or None
 """
 try:
 # Extract basic information
 text = tweet.text
 if not text:
 return None

 # Clean up text
 cleaned_text = sanitize_text(text)
 if not cleaned_text or len(cleaned_text) < 10:
 return None

 # Validate content
 if not validate_text_content(cleaned_text, "twitter"):
 return None

 # Extract metrics
 metrics = tweet.public_metrics or {}

 # Determining influence
 engagement_score = (
 metrics.get("like_count", 0) * 1 +
 metrics.get("retweet_count", 0) * 2 +
 metrics.get("reply_count", 0) * 1.5 +
 metrics.get("quote_count", 0) * 2
 )

 processed_tweet = {
 "id": tweet.id,
 "text": cleaned_text,
 "original_text": text,
 "symbol": symbol,
 "symbols_mentioned": list(self._extract_crypto_mentions(text)),
 "source": "twitter",
 "created_at": tweet.created_at.isoformat if tweet.created_at else datetime.utcnow.isoformat,
 "author_id": tweet.author_id,
 "metrics": {
 "likes": metrics.get("like_count", 0),
 "retweets": metrics.get("retweet_count", 0),
 "replies": metrics.get("reply_count", 0),
 "quotes": metrics.get("quote_count", 0),
 "engagement_score": engagement_score
 },
 "metadata": {
 "language": "en", # Filtering only English tweets
 "platform": "twitter",
 "content_type": "text",
 "is_verified": False, # Needed additional information about user
 "follower_count": 0 # Needed additional information about user
 }
 }

 return processed_tweet

 except Exception as e:
 logger.error("Error processing tweet", error=e, tweet_id=getattr(tweet, 'id', 'unknown'))
 return None

 async def stream_tweets(
 self,
 symbols: List[str] = None,
 callback=None
 ):
 """
 Stream tweets in real time

 Args:
 symbols: Symbols for monitoring
 callback: Function for handling each tweet
 """
 if not self.client:
 await self.initialize

 if not symbols:
 symbols = list(self.crypto_symbols)[:5]

 # Building filters
 track_terms = []
 for symbol in symbols:
 track_terms.extend([f"${symbol}", f"#{symbol}", f"{symbol} crypto"])

 logger.info(
 "Starting Twitter stream",
 symbols=symbols,
 track_terms_count=len(track_terms)
 )

 try:
 # This simplified version - for production needed TwitterStream
 while True:
 # In actually implementation here was would TwitterStream
 tweets = await self.search_tweets(symbols=symbols, limit=10, hours_back=1)

 for tweet in tweets:
 if callback:
 try:
 await callback(tweet)
 except Exception as e:
 logger.error("Error in stream callback", error=e)

 # Pause between iterations
 await asyncio.sleep(60)

 except Exception as e:
 logger.error("Error in Twitter stream", error=e)
 raise

 def get_stats(self) -> Dict[str, Any]:
 """
 Get source statistics

 Returns:
 Dict[str, Any]: Operational statistics
 """
 return {
 "source": "twitter",
 "tweets_processed": self.tweets_processed,
 "api_calls_made": self.api_calls_made,
 "circuit_breaker_state": self.circuit_breaker.state,
 "circuit_breaker_failures": self.circuit_breaker.failure_count,
 "last_error": str(self.last_error) if self.last_error else None,
 "crypto_symbols_tracked": len(self.crypto_symbols),
 "crypto_keywords_tracked": len(self.crypto_keywords),
 "initialized": self.client is not None
 }


async def create_twitter_source -> TwitterSentimentSource:
 """
 Factory function for creating a Twitter data source

 Returns:
 TwitterSentimentSource: Configured data source
 """
 source = TwitterSentimentSource
 await source.initialize
 return source