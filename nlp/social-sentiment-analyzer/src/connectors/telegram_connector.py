"""
Telegram API Connector with enterprise patterns

Integration with Telegram API through Telethon for monitoring
crypto-channels and groups with enterprise-grade reliability.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import structlog
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.tl.types import User, Channel, Chat
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import CircuitBreaker
import re

from ..utils.config import Config
from ..monitoring.metrics_collector import MetricsCollector

logger = structlog.get_logger(__name__)

class TelegramConnector:
    """
    Enterprise Telegram API connector with enterprise patterns
    
    Features:
    - Telethon integration with async support
    - Monitoring crypto-channels and groups
    - Circuit breaker and rate limiting
    - Real-time message streaming
    - Crypto-focused filtering content
    - User activity tracking
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector("telegram_connector")
        self.logger = logger.bind(component="telegram_connector")
        
        # Telegram client
        self._client = TelegramClient(
            'crypto_sentiment_session',
            config.telegram_api_id,
            config.telegram_api_hash
        )
        
        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=Exception
        )
        
        self._connected = False
        self._monitored_channels = []
        
        # Crypto Telegram channels and groups
        self.crypto_channels = [
            "@bitcoin", "@ethereum", "@binance", "@cryptocom", 
            "@coinbase", "@kucoincom", "@okx", "@gate_io",
            "@CryptoNews", "@CryptoPanic", "@CryptoDaily",
            "@WhaleAlert", "@DefiPulse", "@UniswapProtocol",
            "@SushiSwap", "@CurveFinance", "@AaveAave",
            "@Coindesk", "@Cointelegraph", "@TheBlock__"
        ]
        
        # Crypto keywords for filtering
        self.crypto_keywords = [
            "btc", "bitcoin", "eth", "ethereum", "crypto", "cryptocurrency",
            "blockchain", "defi", "nft", "altcoin", "hodl", "moon", "lambo",
            "diamond hands", "paper hands", "whale", "pump", "dump", "dip",
            "bull", "bear", "satoshi", "gwei", "gas", "staking", "yield",
            "liquidity", "apy", "apr", "dex", "cex", "dao"
        ]
    
    async def connect(self) -> bool:
        """Install connection to Telegram API."""
        try:
            await self._client.start()
            
            if await self._client.is_user_authorized():
                me = await self._client.get_me()
                self._connected = True
                self.logger.info("Telegram connection established", 
                               user_id=me.id, username=me.username)
                self.metrics.increment("connection_success")
                return True
            else:
                # Is required authorization by number phone
                phone = self.config.telegram_phone
                await self._client.send_code_request(phone)
                self.logger.warning("Telegram requires phone verification", phone=phone)
                return False
                
        except SessionPasswordNeededError:
            self.logger.error("Telegram requires 2FA password")
            return False
        except Exception as e:
            self.logger.error("Telegram connection failed", error=str(e))
            self.metrics.increment("connection_error")
            return False
    
    @CircuitBreaker(failure_threshold=3, recovery_timeout=30)
    async def get_channel_messages(
        self,
        channel_username: str,
        limit: int = 100,
        hours_back: int = 24,
        crypto_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get messages from channel with crypto filtering
        
        Args:
            channel_username: Name channel (with @ or without)
            limit: Number messages
            hours_back: How many hours back search
            crypto_filter: Filter only crypto-content
        """
        
        if not self._connected:
            await self.connect()
        
        try:
            # Normalization name channel
            if not channel_username.startswith('@'):
                channel_username = '@' + channel_username
            
            entity = await self._client.get_entity(channel_username)
            
            # Temporary window for search
            offset_date = datetime.now() - timedelta(hours=hours_back)
            
            messages = []
            async for message in self._client.iter_messages(
                entity, 
                limit=limit,
                offset_date=offset_date
            ):
                if not message.text:
                    continue
                
                # Filtering by crypto content
                if crypto_filter and not self._is_crypto_related(message.text):
                    continue
                
                # Information about author
                sender = None
                if message.sender:
                    if isinstance(message.sender, User):
                        sender = {
                            "id": message.sender.id,
                            "username": message.sender.username,
                            "first_name": message.sender.first_name,
                            "is_bot": message.sender.bot,
                            "is_verified": message.sender.verified
                        }
                    elif isinstance(message.sender, Channel):
                        sender = {
                            "id": message.sender.id,
                            "title": message.sender.title,
                            "username": message.sender.username,
                            "is_channel": True,
                            "subscribers": message.sender.participants_count
                        }
                
                message_data = {
                    "id": message.id,
                    "text": message.text,
                    "date": message.date.isoformat(),
                    "views": message.views or 0,
                    "forwards": message.forwards or 0,
                    "replies": message.replies.replies if message.replies else 0,
                    "reactions": self._parse_reactions(message.reactions) if message.reactions else {},
                    "sender": sender,
                    "channel": channel_username,
                    "is_reply": message.is_reply,
                    "reply_to_msg_id": message.reply_to_msg_id,
                    "crypto_symbols": self._extract_crypto_symbols(message.text),
                    "sentiment_indicators": self._extract_sentiment_indicators(message.text),
                    "urls": self._extract_urls(message.text),
                    "platform": "telegram"
                }
                
                messages.append(message_data)
            
            self.metrics.increment("messages_fetched", len(messages))
            self.logger.info("Channel messages fetched", 
                           channel=channel_username, count=len(messages))
            
            return messages
            
        except FloodWaitError as e:
            self.logger.warning("Rate limit hit, waiting", wait_time=e.seconds)
            await asyncio.sleep(e.seconds)
            return []
        except Exception as e:
            self.logger.error("Failed to fetch channel messages", 
                           channel=channel_username, error=str(e))
            self.metrics.increment("fetch_error")
            return []
    
    async def monitor_channels_realtime(
        self,
        channels: Optional[List[str]] = None,
        crypto_filter: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Monitoring channels in real time
        
        Args:
            channels: List channels for monitoring
            crypto_filter: Filter only crypto-content
        """
        
        if not self._connected:
            await self.connect()
        
        if not channels:
            channels = self.crypto_channels[:10]  # Limitation for performance
        
        entities = []
        for channel in channels:
            try:
                if not channel.startswith('@'):
                    channel = '@' + channel
                entity = await self._client.get_entity(channel)
                entities.append(entity)
                self.logger.debug("Added channel to monitoring", channel=channel)
            except Exception as e:
                self.logger.warning("Failed to add channel", channel=channel, error=str(e))
                continue
        
        @self._client.on(events.NewMessage(chats=entities))
        async def handler(event):
            try:
                if not event.message.text:
                    return
                
                # Filtering by crypto content
                if crypto_filter and not self._is_crypto_related(event.message.text):
                    return
                
                # Information about sender
                sender = None
                if event.sender:
                    sender = {
                        "id": event.sender.id,
                        "username": getattr(event.sender, 'username', None),
                        "title": getattr(event.sender, 'title', None),
                        "first_name": getattr(event.sender, 'first_name', None),
                    }
                
                message_data = {
                    "id": event.message.id,
                    "text": event.message.text,
                    "date": event.message.date.isoformat(),
                    "chat_id": event.chat_id,
                    "sender": sender,
                    "crypto_symbols": self._extract_crypto_symbols(event.message.text),
                    "sentiment_indicators": self._extract_sentiment_indicators(event.message.text),
                    "platform": "telegram",
                    "real_time": True
                }
                
                self.metrics.increment("realtime_messages")
                yield message_data
                
            except Exception as e:
                self.logger.error("Error processing real-time message", error=str(e))
        
        self.logger.info("Started real-time monitoring", channels=len(entities))
        
        # Launch monitoring
        try:
            await self._client.run_until_disconnected()
        except KeyboardInterrupt:
            self.logger.info("Real-time monitoring stopped")
    
    async def search_in_channels(
        self,
        query: str,
        channels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search messages in channels by request."""
        
        if not self._connected:
            await self.connect()
        
        if not channels:
            channels = self.crypto_channels
        
        all_results = []
        
        for channel in channels:
            try:
                if not channel.startswith('@'):
                    channel = '@' + channel
                
                entity = await self._client.get_entity(channel)
                
                messages = []
                async for message in self._client.iter_messages(
                    entity, 
                    limit=limit//len(channels),
                    search=query
                ):
                    if message.text:
                        message_data = {
                            "id": message.id,
                            "text": message.text,
                            "date": message.date.isoformat(),
                            "views": message.views or 0,
                            "forwards": message.forwards or 0,
                            "channel": channel,
                            "relevance_score": self._calculate_relevance_score(message.text, query),
                            "crypto_symbols": self._extract_crypto_symbols(message.text),
                            "platform": "telegram"
                        }
                        messages.append(message_data)
                
                all_results.extend(messages)
                self.logger.debug("Search completed in channel", 
                                channel=channel, results=len(messages))
                
            except Exception as e:
                self.logger.error("Search failed in channel", 
                                channel=channel, error=str(e))
                continue
        
        # Sorting by relevance
        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        self.logger.info("Search completed", query=query, total_results=len(all_results))
        return all_results
    
    async def get_channel_info(self, channel_username: str) -> Dict[str, Any]:
        """Get information about channel."""
        
        if not self._connected:
            await self.connect()
        
        try:
            if not channel_username.startswith('@'):
                channel_username = '@' + channel_username
            
            entity = await self._client.get_entity(channel_username)
            
            if isinstance(entity, Channel):
                return {
                    "id": entity.id,
                    "title": entity.title,
                    "username": entity.username,
                    "description": entity.about or "",
                    "subscribers": entity.participants_count or 0,
                    "is_broadcast": entity.broadcast,
                    "is_megagroup": entity.megagroup,
                    "is_verified": entity.verified,
                    "created_date": entity.date.isoformat() if entity.date else None,
                    "platform": "telegram"
                }
            else:
                return {
                    "id": entity.id,
                    "title": entity.title,
                    "participants_count": 0,
                    "platform": "telegram"
                }
                
        except Exception as e:
            self.logger.error("Failed to get channel info", 
                           channel=channel_username, error=str(e))
            return {}
    
    def _parse_reactions(self, reactions) -> Dict[str, int]:
        """Parsing reactions on message."""
        if not reactions or not reactions.results:
            return {}
        
        parsed = {}
        for reaction in reactions.results:
            emoji = reaction.reaction.emoticon if hasattr(reaction.reaction, 'emoticon') else str(reaction.reaction)
            parsed[emoji] = reaction.count
        
        return parsed
    
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
        
        # Search additional symbols
        dollar_symbols = re.findall(r'\$[A-Z]{2,6}', text_upper)
        symbols.extend(dollar_symbols)
        
        return list(set(symbols))
    
    def _extract_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """Extract indicators sentiment."""
        text_lower = text.lower()
        
        bullish_terms = ["🚀", "🌙", "💎", "👐", "📈", "moon", "lambo", "diamond hands", "hodl", "pump"]
        bearish_terms = ["📉", "💸", "😭", "dump", "crash", "bear", "paper hands", "sell", "panic"]
        neutral_terms = ["📊", "📰", "dip", "consolidation", "analysis", "news"]
        
        return {
            "bullish_count": sum(1 for term in bullish_terms if term in text_lower),
            "bearish_count": sum(1 for term in bearish_terms if term in text_lower),
            "neutral_count": sum(1 for term in neutral_terms if term in text_lower)
        }
    
    def _extract_urls(self, text: str) -> List[str]:
        """Extract URL from text."""
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        return url_pattern.findall(text)
    
    def _is_crypto_related(self, text: str) -> bool:
        """Check connectivity with cryptocurrencies."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def _calculate_relevance_score(self, text: str, query: str) -> float:
        """Calculate relevance messages to request."""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Counting occurrences request
        count = text_lower.count(query_lower)
        
        # Bonus for crypto-terms
        crypto_bonus = sum(1 for keyword in self.crypto_keywords if keyword in text_lower)
        
        return count * 10 + crypto_bonus
    
    async def health_check(self) -> Dict[str, Any]:
        """Validation state connector."""
        try:
            if not self._connected:
                await self.connect()
            
            me = await self._client.get_me()
            
            return {
                "status": "healthy" if me else "unhealthy",
                "connected": self._connected,
                "user_id": me.id if me else None,
                "username": me.username if me else None,
                "circuit_breaker_state": str(self._circuit_breaker.current_state),
                "monitored_channels": len(self._monitored_channels),
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
        await self._client.disconnect()
        self._connected = False
        self.logger.info("Telegram connector disconnected")