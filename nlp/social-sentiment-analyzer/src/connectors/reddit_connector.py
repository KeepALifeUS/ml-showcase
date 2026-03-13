"""
Reddit API Connector with enterprise patterns

Integration with Reddit API through PRAW with full monitoring,
circuit breaker and crypto-focused collection data.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import praw
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import CircuitBreaker
from ratelimit import limits, sleep_and_retry
import re

from ..utils.config import Config
from ..monitoring.metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)

class RedditConnector:
    """
    Enterprise Reddit API connector with enterprise patterns
    
    Features:
    - PRAW integration with async wrapper
    - Circuit breaker for API protection
    - Crypto subreddits monitoring
    - Sentiment-aware comment parsing
    - Rate limiting and retry logic
    - Hot/New/Rising posts tracking
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector("reddit_connector")
        self.logger = logger.bind(component="reddit_connector")
        
        # Reddit API client
        self._client = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent=config.reddit_user_agent,
            username=config.reddit_username,
            password=config.reddit_password
        )
        
        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        self._connected = False
        
        # Crypto-focused subreddits
        self.crypto_subreddits = [
            "CryptoCurrency", "Bitcoin", "ethereum", "cardano", "solana",
            "DeFi", "NFTs", "altcoin", "CryptoMarkets", "CryptoTechnology",
            "BitcoinMarkets", "ethtrader", "SatoshiStreetBets", "CryptoMoonShots",
            "pancakeswap", "uniswap", "dogecoin", "shibainu", "polygon",
            "Chainlink", "dot", "CryptoCurrencyTrading", "CryptoNews",
            "binance", "coinbase", "kraken", "kucoin", "CryptoCurrencies"
        ]
        
        # Crypto keywords for filtering
        self.crypto_keywords = [
            "btc", "bitcoin", "eth", "ethereum", "crypto", "cryptocurrency",
            "blockchain", "defi", "nft", "altcoin", "hodl", "moon", "lambo",
            "diamond hands", "paper hands", "whale", "pump", "dump", "dip",
            "bull", "bear", "satoshi", "gwei", "gas", "staking", "yield",
            "liquidity", "apy", "apr", "dex", "cex", "dao", "smart contract"
        ]
    
    async def connect(self) -> bool:
        """Install connection to Reddit API."""
        try:
            # Validation connections
            user = self._client.user.me()
            if user:
                self._connected = True
                self.logger.info("Reddit connection established", username=user.name)
                self.metrics.increment("connection_success")
                return True
            
            self.logger.error("Failed to authenticate with Reddit API")
            self.metrics.increment("connection_failure")
            return False
            
        except Exception as e:
            self.logger.error("Reddit connection failed", error=str(e))
            self.metrics.increment("connection_error")
            return False
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    @sleep_and_retry
    @limits(calls=60, period=60)  # 60 requests per minute
    async def get_hot_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 100,
        crypto_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get hot posts from crypto subreddits
        
        Args:
            subreddits: List subreddits (None = all crypto)
            limit: Number posts on subreddit
            crypto_only: Filter only crypto-content
        """
        
        if not self._connected:
            await self.connect()
        
        if not subreddits:
            subreddits = self.crypto_subreddits[:10]  # Top-10 for performance
        
        all_posts = []
        
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self._client.subreddit(subreddit_name)
                    posts = list(subreddit.hot(limit=limit))
                    
                    for post in posts:
                        # Filtering by crypto if enabled
                        if crypto_only and not self._is_crypto_related(post.title + " " + post.selftext):
                            continue
                        
                        # Get top comments
                        post.comments.replace_more(limit=0)
                        top_comments = [
                            {
                                "id": comment.id,
                                "body": comment.body,
                                "score": comment.score,
                                "created_utc": comment.created_utc,
                                "author": str(comment.author) if comment.author else "[deleted]",
                                "crypto_symbols": self._extract_crypto_symbols(comment.body)
                            }
                            for comment in post.comments[:10]  # Top-10 comments
                            if hasattr(comment, 'body') and comment.body != "[removed]" and comment.body != "[deleted]"
                        ]
                        
                        post_data = {
                            "id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "url": post.url,
                            "created_utc": post.created_utc,
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "author": str(post.author) if post.author else "[deleted]",
                            "subreddit": subreddit_name,
                            "permalink": post.permalink,
                            "over_18": post.over_18,
                            "spoiler": post.spoiler,
                            "stickied": post.stickied,
                            "locked": post.locked,
                            "distinguished": post.distinguished,
                            "gilded": post.gilded,
                            "flair_text": post.link_flair_text,
                            "crypto_symbols": self._extract_crypto_symbols(post.title + " " + post.selftext),
                            "sentiment_indicators": self._extract_sentiment_indicators(post.title + " " + post.selftext),
                            "comments": top_comments,
                            "platform": "reddit"
                        }
                        
                        all_posts.append(post_data)
                    
                    self.logger.debug("Posts fetched from subreddit", 
                                    subreddit=subreddit_name, count=len(posts))
                    
                except Exception as e:
                    self.logger.error("Failed to fetch posts from subreddit", 
                                    subreddit=subreddit_name, error=str(e))
                    continue
            
            self.metrics.increment("posts_fetched", len(all_posts))
            self.logger.info("Hot posts fetched successfully", count=len(all_posts))
            
            return sorted(all_posts, key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            self.logger.error("Failed to fetch hot posts", error=str(e))
            self.metrics.increment("fetch_error")
            raise
    
    async def get_new_posts(
        self,
        subreddits: Optional[List[str]] = None,
        limit: int = 50,
        time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get new posts for specific period."""
        
        if not self._connected:
            await self.connect()
        
        if not subreddits:
            subreddits = self.crypto_subreddits[:5]  # Top-5 for new posts
        
        current_time = time.time()
        cutoff_time = current_time - (time_window_hours * 3600)
        
        all_posts = []
        
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self._client.subreddit(subreddit_name)
                    posts = list(subreddit.new(limit=limit))
                    
                    for post in posts:
                        # Filter by time
                        if post.created_utc < cutoff_time:
                            continue
                        
                        # Filter by crypto-content
                        if not self._is_crypto_related(post.title + " " + post.selftext):
                            continue
                        
                        post_data = {
                            "id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "created_utc": post.created_utc,
                            "score": post.score,
                            "upvote_ratio": post.upvote_ratio,
                            "num_comments": post.num_comments,
                            "author": str(post.author) if post.author else "[deleted]",
                            "subreddit": subreddit_name,
                            "crypto_symbols": self._extract_crypto_symbols(post.title + " " + post.selftext),
                            "platform": "reddit"
                        }
                        
                        all_posts.append(post_data)
                    
                except Exception as e:
                    self.logger.error("Failed to fetch new posts", 
                                    subreddit=subreddit_name, error=str(e))
                    continue
            
            self.logger.info("New posts fetched", count=len(all_posts))
            return sorted(all_posts, key=lambda x: x["created_utc"], reverse=True)
            
        except Exception as e:
            self.logger.error("Failed to fetch new posts", error=str(e))
            raise
    
    async def get_user_posts(self, username: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get posts specific user."""
        
        if not self._connected:
            await self.connect()
        
        try:
            redditor = self._client.redditor(username)
            posts = list(redditor.submissions.new(limit=limit))
            
            user_posts = []
            for post in posts:
                if self._is_crypto_related(post.title + " " + post.selftext):
                    post_data = {
                        "id": post.id,
                        "title": post.title,
                        "selftext": post.selftext,
                        "created_utc": post.created_utc,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "subreddit": str(post.subreddit),
                        "author": username,
                        "crypto_symbols": self._extract_crypto_symbols(post.title + " " + post.selftext),
                        "platform": "reddit"
                    }
                    user_posts.append(post_data)
            
            self.logger.info("User posts fetched", username=username, count=len(user_posts))
            return user_posts
            
        except Exception as e:
            self.logger.error("Failed to fetch user posts", username=username, error=str(e))
            return []
    
    async def search_posts(
        self,
        query: str,
        subreddits: Optional[List[str]] = None,
        time_filter: str = "week",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search posts by request in crypto subreddits."""
        
        if not self._connected:
            await self.connect()
        
        if not subreddits:
            subreddits = self.crypto_subreddits
        
        all_results = []
        
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self._client.subreddit(subreddit_name)
                    results = list(subreddit.search(query, time_filter=time_filter, limit=limit//len(subreddits)))
                    
                    for post in results:
                        post_data = {
                            "id": post.id,
                            "title": post.title,
                            "selftext": post.selftext,
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "created_utc": post.created_utc,
                            "subreddit": subreddit_name,
                            "author": str(post.author) if post.author else "[deleted]",
                            "relevance_score": self._calculate_relevance_score(post, query),
                            "crypto_symbols": self._extract_crypto_symbols(post.title + " " + post.selftext),
                            "platform": "reddit"
                        }
                        all_results.append(post_data)
                    
                except Exception as e:
                    self.logger.error("Search failed in subreddit", 
                                    subreddit=subreddit_name, error=str(e))
                    continue
            
            # Sorting by relevance
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            self.logger.info("Search completed", query=query, results=len(all_results))
            return all_results
            
        except Exception as e:
            self.logger.error("Search failed", query=query, error=str(e))
            return []
    
    def _extract_crypto_symbols(self, text: str) -> List[str]:
        """Extract mentions cryptocurrencies."""
        symbols = []
        text_upper = text.upper()
        
        # Main symbols
        crypto_symbols = [
            "BTC", "ETH", "ADA", "SOL", "DOT", "LINK", "UNI", "MATIC",
            "AVAX", "ATOM", "FTM", "NEAR", "ALGO", "XRP", "LTC", "BCH"
        ]
        
        for symbol in crypto_symbols:
            if symbol in text_upper or f"${symbol}" in text_upper:
                symbols.append(f"${symbol}")
        
        # Search additional symbols in format $XXX
        dollar_symbols = re.findall(r'\$[A-Z]{2,6}', text_upper)
        symbols.extend(dollar_symbols)
        
        return list(set(symbols))
    
    def _extract_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """Extract indicators sentiment from text."""
        text_lower = text.lower()
        
        bullish_terms = ["moon", "lambo", "diamond hands", "hodl", "bull", "pump", "rocket", "🚀"]
        bearish_terms = ["dump", "crash", "bear", "paper hands", "sell", "panic", "rekt"]
        neutral_terms = ["dip", "consolidation", "sideways", "stable"]
        
        return {
            "bullish_count": sum(1 for term in bullish_terms if term in text_lower),
            "bearish_count": sum(1 for term in bearish_terms if term in text_lower),
            "neutral_count": sum(1 for term in neutral_terms if term in text_lower)
        }
    
    def _is_crypto_related(self, text: str) -> bool:
        """Check connectivity with cryptocurrencies."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def _calculate_relevance_score(self, post, query: str) -> float:
        """Calculate relevance post to request."""
        title_score = post.title.lower().count(query.lower()) * 3
        text_score = post.selftext.lower().count(query.lower())
        engagement_score = (post.score + post.num_comments) / 100
        
        return title_score + text_score + engagement_score
    
    async def health_check(self) -> Dict[str, Any]:
        """Validation state connector."""
        try:
            if not self._connected:
                await self.connect()
            
            user = self._client.user.me()
            
            return {
                "status": "healthy" if user else "unhealthy",
                "connected": self._connected,
                "username": user.name if user else None,
                "circuit_breaker_state": str(self._circuit_breaker.current_state),
                "metrics": self.metrics.get_metrics()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connected": False
            }
    
    async def disconnect(self) -> None:
        """Close connection."""
        self._connected = False
        self.logger.info("Reddit connector disconnected")