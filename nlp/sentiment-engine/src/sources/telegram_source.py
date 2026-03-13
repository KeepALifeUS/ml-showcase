"""
Telegram Sentiment Data Source for ML-Framework ML Sentiment Engine

Enterprise-grade Telegram data collection with and async support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from telethon.tl.types import Channel, Chat, User

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TelegramRateLimiter:
 """Rate limiter for Telegram API"""

 def __init__(self, messages_per_second: float = 1.0):
 self.messages_per_second = messages_per_second
 self.min_interval = 1.0 / messages_per_second
 self.last_request_time = 0

 async def acquire(self):
 """Acquire permission for the request"""
 now = time.time
 time_since_last = now - self.last_request_time

 if time_since_last < self.min_interval:
 sleep_time = self.min_interval - time_since_last
 await asyncio.sleep(sleep_time)

 self.last_request_time = time.time


class TelegramSentimentSource:
 """
 Enterprise-grade Telegram data source for sentiment analysis

 Features:
 - Multi-channel monitoring
 - Real-time message streaming
 - Crypto-focused channels
 - Rate limiting compliance
 - Message deduplication
 - Channel metadata tracking
 """

 def __init__(self):
 """Initialize the Telegram data source"""
 config = get_config

 # Telegram API credentials
 self.api_id = config.social.telegram_api_id
 self.api_hash = config.social.telegram_api_hash
 self.phone = config.social.telegram_phone

 # Telegram client
 self.client: Optional[TelegramClient] = None

 # Rate limiting
 self.rate_limiter = TelegramRateLimiter

 # Crypto symbols and keywords
 self.crypto_symbols = set(get_crypto_symbols)
 self.crypto_keywords = set(get_crypto_keywords)

 # Crypto Telegram channels/groups
 self.crypto_channels = [
 # Public crypto channels
 "@bitcoin",
 "@ethereum",
 "@binance",
 "@CoinDesk",
 "@cointelegraph",
 "@cryptonews",

 # Trading channels
 "@cryptosignals",
 "@binancesignals",
 "@freecryptosignals",
 "@cryptowhales",
 "@whalewatching",

 # Analysis channels
 "@cryptoanalysis",
 "@bitcoinanalysis",
 "@technicalanalysis",
 "@cryptoTA",

 # News aggregators
 "@cryptonewsaggregator",
 "@dailycryptonews",
 "@cryptoupdates"
 ]

 # Performance metrics
 self.messages_processed = 0
 self.channels_monitored = 0
 self.api_calls_made = 0
 self.last_error = None

 # Message deduplication
 self.seen_messages = set

 async def initialize(self):
 """Initialize the Telegram client"""
 if not all([self.api_id, self.api_hash]):
 raise ValueError("Telegram API ID and Hash are required")

 self.client = TelegramClient(
 'ml-framework_sentiment_session',
 self.api_id,
 self.api_hash
 )

 try:
 await self.client.start(phone=self.phone)
 logger.info("Telegram client initialized successfully")

 # Check authorization
 me = await self.client.get_me
 logger.info(f"Telegram authenticated as: {me.username or me.phone}")

 except SessionPasswordNeededError:
 logger.error("Two-factor authentication required for Telegram")
 raise
 except Exception as e:
 logger.error("Failed to initialize Telegram client", error=e)
 raise

 async def cleanup(self):
 """Cleanup resources"""
 if self.client:
 await self.client.disconnect
 logger.info("Telegram source cleaned up")

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

 for symbol in self.crypto_symbols:
 patterns = [
 rf'\b{symbol}\b', # Exact match
 rf'\${symbol}\b', # With prefix $
 rf'#{symbol}\b', # Hashtag
 rf'{symbol}/USDT\b', # Trading pairs
 rf'{symbol}USDT\b',
 rf'{symbol}/BTC\b',
 rf'{symbol}BTC\b'
 ]

 for pattern in patterns:
 if re.search(pattern, text_upper):
 mentioned_symbols.add(symbol)
 break

 return mentioned_symbols

 def _is_crypto_relevant(self, text: str) -> bool:
 """
 Check message relevance for crypto analysis

 Args:
 text: Text to check

 Returns:
 bool: True if message relevant
 """
 text_lower = text.lower

 # Check for keywords
 for keyword in self.crypto_keywords:
 if keyword in text_lower:
 return True

 # Check for cryptocurrency symbols
 if self._extract_crypto_mentions(text):
 return True

 # Telegram-specific patterns
 telegram_patterns = [
 r'🚀', r'📈', r'📉', r'💎', r'🌙', # Crypto emoji
 r'\bto\s+the\s+moon\b', r'\bhodl\b', r'\bdip\b',
 r'\bpumping\b', r'\bdumping\b', r'\bwhales?\b',
 r'\bsignal\b', r'\bbuy\b', r'\bsell\b', r'\btarget\b'
 ]

 for pattern in telegram_patterns:
 if re.search(pattern, text_lower):
 return True

 return False

 async def _get_channel_info(self, channel_username: str) -> Optional[Dict[str, Any]]:
 """
 Get information about the channel

 Args:
 channel_username: Username channel

 Returns:
 Optional[Dict[str, Any]]: Information about channel
 """
 try:
 await self.rate_limiter.acquire

 entity = await self.client.get_entity(channel_username)

 channel_info = {
 "id": entity.id,
 "username": getattr(entity, 'username', None),
 "title": getattr(entity, 'title', ''),
 "participants_count": getattr(entity, 'participants_count', 0),
 "description": getattr(entity, 'about', ''),
 "type": "channel" if isinstance(entity, Channel) else "chat"
 }

 return channel_info

 except Exception as e:
 logger.error(f"Error getting channel info for {channel_username}", error=e)
 return None

 async def fetch_channel_messages(
 self,
 channel_username: str,
 limit: int = 100,
 hours_back: int = 24
 ) -> List[Dict[str, Any]]:
 """
 Get messages from a channel

 Args:
 channel_username: Username channel
 limit: Maximum number of messages
 hours_back: Time period in hours

 Returns:
 List[Dict[str, Any]]: List of processed messages
 """
 if not self.client:
 await self.initialize

 try:
 messages = []
 offset_date = datetime.utcnow - timedelta(hours=hours_back)

 await self.rate_limiter.acquire

 # Get messages
 async for message in self.client.iter_messages(
 channel_username,
 limit=limit,
 offset_date=offset_date
 ):
 processed_message = await self._process_message(message, channel_username)
 if processed_message and self._is_crypto_relevant(processed_message["text"]):
 messages.append(processed_message)

 # Check for duplicates
 message_id = f"{channel_username}_{message.id}"
 if message_id in self.seen_messages:
 continue

 self.seen_messages.add(message_id)

 self.messages_processed += len(messages)
 self.api_calls_made += 1

 logger.info(
 "Messages fetched from Telegram channel",
 channel=channel_username,
 messages_count=len(messages),
 total_processed=self.messages_processed
 )

 return messages

 except FloodWaitError as e:
 logger.warning(f"Telegram flood wait for {e.seconds} seconds", channel=channel_username)
 await asyncio.sleep(e.seconds)
 return []
 except Exception as e:
 self.last_error = e
 logger.error(f"Error fetching messages from {channel_username}", error=e)
 return []

 async def _process_message(self, message, channel_username: str) -> Optional[Dict[str, Any]]:
 """
 Process a single message

 Args:
 message: Object messages from Telegram API
 channel_username: Username channel

 Returns:
 Optional[Dict[str, Any]]: Processed message or None
 """
 try:
 # Check for text content
 if not message.text:
 return None

 text = message.text
 if len(text) < 5:
 return None

 # Clean up text
 cleaned_text = sanitize_text(text)
 if not cleaned_text:
 return None

 # Validate content
 if not validate_text_content(cleaned_text, "telegram"):
 return None

 # Extract author information
 sender = message.sender
 author_info = {
 "id": sender.id if sender else None,
 "username": getattr(sender, 'username', None),
 "first_name": getattr(sender, 'first_name', ''),
 "is_bot": getattr(sender, 'bot', False)
 }

 # Record metrics interactions
 views = getattr(message, 'views', 0)
 forwards = getattr(message, 'forwards', 0)
 replies = getattr(message, 'replies', None)
 reply_count = replies.replies if replies else 0

 processed_message = {
 "id": f"{channel_username}_{message.id}",
 "text": cleaned_text,
 "original_text": text,
 "symbols_mentioned": list(self._extract_crypto_mentions(text)),
 "source": "telegram",
 "channel": channel_username,
 "message_id": message.id,
 "created_at": message.date.isoformat if message.date else datetime.utcnow.isoformat,
 "author": author_info,
 "metrics": {
 "views": views,
 "forwards": forwards,
 "replies": reply_count,
 "engagement_score": views * 0.1 + forwards * 2 + reply_count * 1.5
 },
 "metadata": {
 "language": "en", # Primarily English content
 "platform": "telegram",
 "content_type": "message",
 "has_media": bool(message.media),
 "is_reply": bool(message.reply_to),
 "is_forwarded": bool(message.forward)
 }
 }

 return processed_message

 except Exception as e:
 logger.error("Error processing Telegram message", error=e, message_id=getattr(message, 'id', 'unknown'))
 return None

 async def fetch_all_channels(
 self,
 limit_per_channel: int = 50,
 hours_back: int = 24
 ) -> List[Dict[str, Any]]:
 """
 Get messages from all monitored channels

 Args:
 limit_per_channel: Message limit per channel
 hours_back: Time period in hours

 Returns:
 List[Dict[str, Any]]: List of all messages
 """
 all_messages = []

 for channel in self.crypto_channels:
 try:
 messages = await self.fetch_channel_messages(
 channel,
 limit=limit_per_channel,
 hours_back=hours_back
 )
 all_messages.extend(messages)
 self.channels_monitored += 1

 # Pause between channels for compliance rate limit
 await asyncio.sleep(2)

 except Exception as e:
 logger.error(f"Error processing Telegram channel {channel}", error=e)
 continue

 logger.info(
 "All Telegram channels processed",
 channels_processed=self.channels_monitored,
 total_messages=len(all_messages),
 total_processed=self.messages_processed
 )

 return all_messages

 async def start_real_time_monitoring(
 self,
 channels: List[str] = None,
 callback=None
 ):
 """
 Start real-time monitoring

 Args:
 channels: List channels for monitoring
 callback: Function for handling new messages
 """
 if not self.client:
 await self.initialize

 if not channels:
 channels = self.crypto_channels

 # Get entity for channels
 channel_entities = []
 for channel in channels:
 try:
 entity = await self.client.get_entity(channel)
 channel_entities.append(entity)
 except Exception as e:
 logger.error(f"Error getting entity for {channel}", error=e)
 continue

 logger.info(
 "Starting real-time Telegram monitoring",
 channels_count=len(channel_entities)
 )

 # Event handler for new message
 @self.client.on(events.NewMessage(chats=channel_entities))
 async def handle_new_message(event):
 try:
 channel_username = getattr(event.chat, 'username', f'id_{event.chat_id}')
 processed_message = await self._process_message(event.message, channel_username)

 if processed_message and self._is_crypto_relevant(processed_message["text"]):
 self.messages_processed += 1

 if callback:
 try:
 await callback(processed_message)
 except Exception as e:
 logger.error("Error in Telegram message callback", error=e)

 logger.debug(
 "New crypto message received",
 channel=channel_username,
 message_id=processed_message["id"]
 )

 except Exception as e:
 logger.error("Error handling new Telegram message", error=e)

 # Start client (blocking call)
 try:
 await self.client.run_until_disconnected
 except Exception as e:
 logger.error("Error in Telegram real-time monitoring", error=e)
 raise

 async def search_messages(
 self,
 query: str,
 channels: List[str] = None,
 limit: int = 100
 ) -> List[Dict[str, Any]]:
 """
 Search messages by keyword

 Args:
 query: Search query
 channels: Channels to search
 limit: Maximum number of results

 Returns:
 List[Dict[str, Any]]: Found messages
 """
 if not channels:
 channels = self.crypto_channels[:5] # Limits for search

 all_results = []

 for channel in channels:
 try:
 await self.rate_limiter.acquire

 results = []

 # Search in messages channel
 async for message in self.client.iter_messages(
 channel,
 search=query,
 limit=limit // len(channels)
 ):
 processed_message = await self._process_message(message, channel)
 if processed_message:
 results.append(processed_message)

 all_results.extend(results)

 logger.debug(
 "Telegram search completed for channel",
 channel=channel,
 query=query,
 results_count=len(results)
 )

 except Exception as e:
 logger.error(f"Error searching in Telegram channel {channel}", error=e)
 continue

 # Sort by time
 all_results.sort(
 key=lambda m: m.get('created_at', ''),
 reverse=True
 )

 logger.info(
 "Telegram search completed",
 query=query,
 channels_searched=len(channels),
 total_results=len(all_results)
 )

 return all_results[:limit]

 def get_stats(self) -> Dict[str, Any]:
 """
 Get source statistics

 Returns:
 Dict[str, Any]: Operational statistics
 """
 return {
 "source": "telegram",
 "messages_processed": self.messages_processed,
 "channels_monitored": self.channels_monitored,
 "api_calls_made": self.api_calls_made,
 "last_error": str(self.last_error) if self.last_error else None,
 "crypto_channels_tracked": len(self.crypto_channels),
 "crypto_symbols_tracked": len(self.crypto_symbols),
 "crypto_keywords_tracked": len(self.crypto_keywords),
 "initialized": self.client is not None,
 "seen_messages_count": len(self.seen_messages),
 "channels": self.crypto_channels
 }


async def create_telegram_source -> TelegramSentimentSource:
 """
 Factory function for creating a Telegram data source

 Returns:
 TelegramSentimentSource: Configured data source
 """
 source = TelegramSentimentSource
 await source.initialize
 return source