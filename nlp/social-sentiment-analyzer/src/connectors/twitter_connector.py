"""
Twitter/X API v2 Connector with enterprise patterns

Provides reliable connection to Twitter API with circuit breaker,
rate limiting, retry logic and full monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import tweepy
import aiohttp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import CircuitBreaker
from ratelimit import limits, sleep_and_retry
import json

from ..utils.config import Config
from ..utils.validators import validate_crypto_symbols
from ..monitoring.metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)

class TwitterConnector:
    """
    Enterprise Twitter/X API v2 connector with enterprise patterns
    
    Features:
    - Circuit breaker for protection from cascading failures
    - Rate limiting with automatic backoff
    - Retry logic with exponential delay
    - Crypto-specific search and filtering
    - Monitoring and metrics in real time
    - Fault tolerance and graceful degradation
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector("twitter_connector")
        self.logger = logger.bind(component="twitter_connector")
        
        # Twitter API v2 client
        self._client = tweepy.Client(
            bearer_token=config.twitter_bearer_token,
            consumer_key=config.twitter_api_key,
            consumer_secret=config.twitter_api_secret,
            access_token=config.twitter_access_token,
            access_token_secret=config.twitter_access_secret,
            wait_on_rate_limit=True
        )
        
        # Circuit breaker for protection API
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        # State connections
        self._connected = False
        self._last_request_time = 0
        self._request_count = 0
        
        # Crypto-specific search terms
        self.crypto_symbols = [
            "$BTC", "$ETH", "$ADA", "$SOL", "$DOT", "$LINK", "$UNI",
            "$MATIC", "$AVAX", "$ATOM", "$FTM", "$NEAR", "$ALGO",
            "Bitcoin", "Ethereum", "Cardano", "Solana", "Polkadot"
        ]
        
        # Search requests for crypto
        self.crypto_queries = [
            "crypto OR cryptocurrency OR blockchain OR DeFi OR NFT",
            "Bitcoin OR BTC OR #Bitcoin OR #BTC",
            "Ethereum OR ETH OR #Ethereum OR #ETH",
            "altcoin OR altcoins OR #altcoin",
            "HODL OR hodl OR #HODL",
            "moon OR #ToTheMoon OR #CryptoMoon",
            "diamond hands OR #DiamondHands OR #HODL",
            "paper hands OR #PaperHands OR sell",
            "bull market OR bear market OR #BullRun OR #BearMarket",
            "whale alert OR whale movement OR #WhaleAlert"
        ]
    
    async def connect(self) -> bool:
        """Install connection to Twitter API."""
        try:
            # Validation credentials
            me = await self._get_me()
            if me:
                self._connected = True
                self.logger.info("Twitter connection established", user_id=me.id)
                self.metrics.increment("connection_success")
                return True
            
            self.logger.error("Failed to authenticate with Twitter API")
            self.metrics.increment("connection_failure")
            return False
            
        except Exception as e:
            self.logger.error("Twitter connection failed", error=str(e))
            self.metrics.increment("connection_error")
            return False
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _get_me(self) -> Optional[Any]:
        """Get information about current user."""
        try:
            return self._client.get_me()
        except Exception as e:
            self.logger.error("Failed to get user info", error=str(e))
            raise
    
    @sleep_and_retry
    @limits(calls=100, period=900)  # 100 requests per 15 minutes
    async def search_recent_tweets(
        self,
        query: str = None,
        max_results: int = 100,
        crypto_focus: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search recent tweets with crypto focus
        
        Args:
            query: Search request (None = automatic crypto query)
            max_results: Maximum number results
            crypto_focus: Focus on cryptocurrencies
        """
        
        if not self._connected:
            await self.connect()
        
        try:
            # Automatic selection crypto query
            if not query and crypto_focus:
                query = " OR ".join(self.crypto_queries[:3])  # First 3 request
            elif not query:
                query = "crypto OR bitcoin OR ethereum"
            
            # Extended parameters search
            tweet_fields = [
                "created_at", "public_metrics", "context_annotations",
                "conversation_id", "in_reply_to_user_id", "referenced_tweets",
                "reply_settings", "source", "withheld", "geo", "lang"
            ]
            
            user_fields = [
                "created_at", "description", "entities", "location",
                "public_metrics", "url", "verified"
            ]
            
            self.logger.info("Searching tweets", query=query, max_results=max_results)
            
            response = self._client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=tweet_fields,
                user_fields=user_fields,
                expansions=["author_id", "referenced_tweets.id", "geo.place_id"]
            )
            
            if not response.data:
                self.logger.warning("No tweets found", query=query)
                return []
            
            tweets = []
            users_dict = {}
            
            # Create dictionary users for fast search
            if response.includes and "users" in response.includes:
                users_dict = {user.id: user for user in response.includes["users"]}
            
            for tweet in response.data:
                # Enrich data tweet
                author = users_dict.get(tweet.author_id, {})
                
                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat() if tweet.created_at else None,
                    "author_id": tweet.author_id,
                    "author_username": getattr(author, "username", "unknown"),
                    "author_name": getattr(author, "name", "unknown"),
                    "author_verified": getattr(author, "verified", False),
                    "author_followers": getattr(author, "public_metrics", {}).get("followers_count", 0) if hasattr(author, "public_metrics") else 0,
                    "public_metrics": {
                        "retweet_count": tweet.public_metrics["retweet_count"] if tweet.public_metrics else 0,
                        "like_count": tweet.public_metrics["like_count"] if tweet.public_metrics else 0,
                        "reply_count": tweet.public_metrics["reply_count"] if tweet.public_metrics else 0,
                        "quote_count": tweet.public_metrics["quote_count"] if tweet.public_metrics else 0,
                    } if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "source": tweet.source,
                    "conversation_id": tweet.conversation_id,
                    "context_annotations": tweet.context_annotations,
                    "crypto_symbols": self._extract_crypto_symbols(tweet.text),
                    "platform": "twitter",
                    "raw_data": tweet.data
                }
                tweets.append(tweet_data)
            
            self.metrics.increment("tweets_fetched", len(tweets))
            self.logger.info("Tweets fetched successfully", count=len(tweets))
            
            return tweets
            
        except Exception as e:
            self.logger.error("Tweet search failed", error=str(e), query=query)
            self.metrics.increment("search_error")
            raise
    
    async def get_user_tweets(
        self,
        username: str,
        max_results: int = 50,
        crypto_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """Get tweets specific user with crypto filter."""
        
        if not self._connected:
            await self.connect()
        
        try:
            # Get ID user
            user = self._client.get_user(username=username)
            if not user.data:
                self.logger.warning("User not found", username=username)
                return []
            
            user_id = user.data.id
            
            # Get tweets user
            response = self._client.get_users_tweets(
                id=user_id,
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "public_metrics", "lang", "context_annotations"],
                exclude=["retweets", "replies"] if crypto_filter else None
            )
            
            if not response.data:
                return []
            
            tweets = []
            for tweet in response.data:
                # Filtering by crypto if enabled
                if crypto_filter and not self._is_crypto_related(tweet.text):
                    continue
                
                tweet_data = {
                    "id": tweet.id,
                    "text": tweet.text,
                    "created_at": tweet.created_at.isoformat(),
                    "author_id": user_id,
                    "author_username": username,
                    "public_metrics": {
                        "retweet_count": tweet.public_metrics["retweet_count"],
                        "like_count": tweet.public_metrics["like_count"],
                        "reply_count": tweet.public_metrics["reply_count"],
                        "quote_count": tweet.public_metrics["quote_count"],
                    } if tweet.public_metrics else {},
                    "lang": tweet.lang,
                    "crypto_symbols": self._extract_crypto_symbols(tweet.text),
                    "platform": "twitter",
                }
                tweets.append(tweet_data)
            
            self.logger.info("User tweets fetched", username=username, count=len(tweets))
            return tweets
            
        except Exception as e:
            self.logger.error("Failed to fetch user tweets", username=username, error=str(e))
            raise
    
    async def get_trending_crypto_hashtags(self, woeid: int = 1) -> List[str]:
        """Get trends connected with cryptocurrencies."""
        
        if not self._connected:
            await self.connect()
        
        try:
            # Get current trends (Global WOEID = 1)
            trends = self._client.get_place_trends(woeid)[0]["trends"]
            
            crypto_trends = []
            for trend in trends:
                trend_name = trend["name"].lower()
                
                # Check on crypto-connectivity
                if any(symbol.lower()[1:] in trend_name for symbol in self.crypto_symbols):
                    crypto_trends.append(trend["name"])
                elif any(keyword in trend_name for keyword in [
                    "crypto", "bitcoin", "eth", "defi", "nft", "blockchain",
                    "hodl", "moon", "whale", "pump", "dump"
                ]):
                    crypto_trends.append(trend["name"])
            
            self.logger.info("Crypto trends fetched", count=len(crypto_trends))
            return crypto_trends
            
        except Exception as e:
            self.logger.error("Failed to fetch crypto trends", error=str(e))
            return []
    
    def _extract_crypto_symbols(self, text: str) -> List[str]:
        """Extract mentions cryptocurrencies from text."""
        symbols = []
        text_upper = text.upper()
        
        for symbol in self.crypto_symbols:
            if symbol.upper() in text_upper:
                symbols.append(symbol)
        
        # Search additional symbols in format $XXX
        import re
        dollar_symbols = re.findall(r'\$[A-Z]{2,6}', text_upper)
        symbols.extend(dollar_symbols)
        
        return list(set(symbols))  # Remove duplicates
    
    def _is_crypto_related(self, text: str) -> bool:
        """Check, connected whether text with cryptocurrencies."""
        text_lower = text.lower()
        
        # Direct mentions symbols
        if any(symbol.lower() in text_lower for symbol in self.crypto_symbols):
            return True
        
        # Crypto-terms
        crypto_terms = [
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain",
            "defi", "nft", "hodl", "moon", "whale", "satoshi", "altcoin",
            "bull market", "bear market", "diamond hands", "paper hands"
        ]
        
        return any(term in text_lower for term in crypto_terms)
    
    async def health_check(self) -> Dict[str, Any]:
        """Validation state connector."""
        try:
            if not self._connected:
                await self.connect()
            
            # Test request
            me = await self._get_me()
            
            return {
                "status": "healthy" if me else "unhealthy",
                "connected": self._connected,
                "user_id": me.id if me else None,
                "circuit_breaker_state": str(self._circuit_breaker.current_state),
                "metrics": self.metrics.get_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False,
                "circuit_breaker_state": str(self._circuit_breaker.current_state)
            }
    
    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        self.logger.info("Twitter connector disconnected")