"""
Discord Sentiment Data Source for ML-Framework ML Sentiment Engine

Enterprise-grade Discord data collection with and bot support.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
import re

import discord
from discord.ext import tasks

from ..utils.logger import get_logger
from ..utils.config import get_config, get_crypto_symbols, get_crypto_keywords
from ..utils.validators import TextContent, CryptoSymbol, validate_text_content, sanitize_text

logger = get_logger(__name__)


class DiscordRateLimiter:
 """Rate limiter for Discord API"""

 def __init__(self, requests_per_second: float = 5.0):
 self.requests_per_second = requests_per_second
 self.min_interval = 1.0 / requests_per_second
 self.last_request_time = 0
 self.bucket_tokens = requests_per_second
 self.last_refill = time.time

 async def acquire(self):
 """Acquire permission for the request (token bucket)"""
 now = time.time

 # Refill bucket
 tokens_to_add = (now - self.last_refill) * self.requests_per_second
 self.bucket_tokens = min(self.requests_per_second, self.bucket_tokens + tokens_to_add)
 self.last_refill = now

 # Check token availability
 if self.bucket_tokens < 1:
 sleep_time = (1 - self.bucket_tokens) / self.requests_per_second
 await asyncio.sleep(sleep_time)
 self.bucket_tokens = 0
 else:
 self.bucket_tokens -= 1


class CryptoDiscordBot(discord.Client):
 """
 Custom Discord bot for monitoring crypto-servers
 """

 def __init__(self, sentiment_source, *args, **kwargs):
 super.__init__(*args, **kwargs)
 self.sentiment_source = sentiment_source
 self.monitored_channels = set
 self.message_callback = None

 async def on_ready(self):
 """Called when the bot is ready"""
 logger.info(f"Discord bot logged in as {self.user}")

 # Get list of available servers and channels
 for guild in self.guilds:
 logger.info(f"Connected to guild: {guild.name} (id: {guild.id})")

 for channel in guild.text_channels:
 if self._is_crypto_channel(channel.name):
 self.monitored_channels.add(channel.id)
 logger.debug(f"Monitoring channel: #{channel.name} in {guild.name}")

 async def on_message(self, message):
 """Process a new message"""
 # Ignore messages from bots
 if message.author.bot:
 return

 # Check that the channel is tracked
 if message.channel.id not in self.monitored_channels:
 return

 # Process messages
 try:
 processed_message = await self.sentiment_source._process_message(message)
 if processed_message and self.sentiment_source._is_crypto_relevant(processed_message["text"]):
 self.sentiment_source.messages_processed += 1

 if self.message_callback:
 try:
 await self.message_callback(processed_message)
 except Exception as e:
 logger.error("Error in Discord message callback", error=e)

 except Exception as e:
 logger.error("Error processing Discord message", error=e)

 def _is_crypto_channel(self, channel_name: str) -> bool:
 """Check whether the channel is crypto-related"""
 crypto_channel_keywords = [
 "crypto", "bitcoin", "btc", "ethereum", "eth", "trading", "signals",
 "market", "analysis", "price", "discussion", "general", "defi",
 "nft", "altcoin", "binance", "coinbase"
 ]

 channel_lower = channel_name.lower
 return any(keyword in channel_lower for keyword in crypto_channel_keywords)

 def set_message_callback(self, callback):
 """Set callback for handling new messages"""
 self.message_callback = callback


class DiscordSentimentSource:
 """
 Enterprise-grade Discord data source for sentiment analysis

 Features:
 - Multi-server monitoring
 - Real-time message tracking
 - Crypto-focused servers/channels
 - Bot-based data collection
 - Rate limiting compliance
 - Message history fetching
 """

 def __init__(self):
 """Initialize the Discord data source"""
 config = get_config

 # Discord Bot Token
 self.bot_token = config.social.discord_token
 self.target_channels = config.social.discord_channels

 # Discord bot
 self.bot: Optional[CryptoDiscordBot] = None

 # Rate limiting
 self.rate_limiter = DiscordRateLimiter

 # Crypto symbols and keywords
 self.crypto_symbols = set(get_crypto_symbols)
 self.crypto_keywords = set(get_crypto_keywords)

 # Popular crypto Discord servers (invite links/IDs)
 self.crypto_servers = [
 # Main crypto servers (needed invitation links or IDs)
 # Examples channels for monitoring
 "crypto-general",
 "bitcoin-discussion",
 "ethereum-talk",
 "altcoin-discussion",
 "trading-signals",
 "market-analysis",
 "defi-discussion",
 "nft-marketplace"
 ]

 # Performance metrics
 self.messages_processed = 0
 self.channels_monitored = 0
 self.servers_connected = 0
 self.api_calls_made = 0
 self.last_error = None

 # Message deduplication
 self.seen_messages = set

 async def initialize(self):
 """Initialize the Discord bot"""
 if not self.bot_token:
 raise ValueError("Discord Bot Token is required")

 # Create the bot with necessary intents
 intents = discord.Intents.default
 intents.message_content = True
 intents.guilds = True

 self.bot = CryptoDiscordBot(
 sentiment_source=self,
 intents=intents
 )

 # Start bot in background task
 self.bot_task = asyncio.create_task(self.bot.start(self.bot_token))

 # Wait for bot readiness
 await asyncio.sleep(3) # Allow time for connection

 if self.bot.is_ready:
 logger.info("Discord bot initialized successfully")
 self.servers_connected = len(self.bot.guilds)
 self.channels_monitored = len(self.bot.monitored_channels)
 else:
 logger.warning("Discord bot initialization may not be complete")

 async def cleanup(self):
 """Cleanup resources"""
 if self.bot:
 await self.bot.close

 if hasattr(self, 'bot_task'):
 self.bot_task.cancel
 try:
 await self.bot_task
 except asyncio.CancelledError:
 pass

 logger.info("Discord source cleaned up")

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
 rf'{symbol}/BTC\b'
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

 # Discord-specific patterns
 discord_patterns = [
 r'\bto\s+the\s+moon\b', r'\bhodl\b', r'\bdiamond\s+hands\b',
 r'\bpaper\s+hands\b', r'\bwhale\s+alert\b', r'\brug\s*pull\b',
 r'\bgm\b', r'\bgn\b', # Good morning/night in crypto community
 r'\blfg\b', r'\bwagmi\b', r'\bngmi\b', # Crypto slang
 r'\bairdrop\b', r'\bmint\b', r'\bfloor\b' # NFT terms
 ]

 for pattern in discord_patterns:
 if re.search(pattern, text_lower):
 return True

 return False

 async def _process_message(self, message) -> Optional[Dict[str, Any]]:
 """
 Process a Discord message

 Args:
 message: Discord message object

 Returns:
 Optional[Dict[str, Any]]: Processed message or None
 """
 try:
 # Check for text content
 if not message.content:
 return None

 text = message.content
 if len(text) < 5:
 return None

 # Clean up text from Discord-specific elements
 # Remove mentions (@user, @everyone, @here)
 text = re.sub(r'<@!?\d+>', '', text)
 text = re.sub(r'@everyone|@here', '', text)

 # Remove channel mentions (#channel)
 text = re.sub(r'<#\d+>', '', text)

 # Remove custom emojis (<:name:id>)
 text = re.sub(r'<a?:\w+:\d+>', '', text)

 cleaned_text = sanitize_text(text)
 if not cleaned_text:
 return None

 # Validate content
 if not validate_text_content(cleaned_text, "discord"):
 return None

 # Check for duplicates
 message_id = f"{message.guild.id}_{message.channel.id}_{message.id}"
 if message_id in self.seen_messages:
 return None

 self.seen_messages.add(message_id)

 # Author information
 author = message.author
 author_info = {
 "id": str(author.id),
 "username": author.name,
 "discriminator": author.discriminator,
 "display_name": author.display_name,
 "is_bot": author.bot,
 "avatar_url": str(author.avatar.url) if author.avatar else None
 }

 # Server and channel information
 guild = message.guild
 channel = message.channel

 # Record metrics reactions
 reactions_count = sum(reaction.count for reaction in message.reactions) if message.reactions else 0

 # Mentions and links
 mentions_count = len(message.mentions)
 has_links = bool(re.search(r'https?://', message.content))

 processed_message = {
 "id": message_id,
 "text": cleaned_text,
 "original_text": message.content,
 "symbols_mentioned": list(self._extract_crypto_mentions(text)),
 "source": "discord",
 "server_name": guild.name,
 "server_id": str(guild.id),
 "channel_name": channel.name,
 "channel_id": str(channel.id),
 "message_id": str(message.id),
 "created_at": message.created_at.isoformat,
 "author": author_info,
 "metrics": {
 "reactions_count": reactions_count,
 "mentions_count": mentions_count,
 "has_links": has_links,
 "message_length": len(cleaned_text),
 "engagement_score": reactions_count * 2 + mentions_count * 0.5
 },
 "metadata": {
 "language": "en", # Primarily English
 "platform": "discord",
 "content_type": "message",
 "has_attachments": bool(message.attachments),
 "has_embeds": bool(message.embeds),
 "is_reply": message.reference is not None,
 "is_pinned": message.pinned
 }
 }

 return processed_message

 except Exception as e:
 logger.error("Error processing Discord message", error=e, message_id=getattr(message, 'id', 'unknown'))
 return None

 async def fetch_channel_history(
 self,
 channel_id: int,
 limit: int = 100,
 hours_back: int = 24
 ) -> List[Dict[str, Any]]:
 """
 Get message history from a channel

 Args:
 channel_id: Discord channel ID
 limit: Maximum number of messages
 hours_back: Time period in hours

 Returns:
 List[Dict[str, Any]]: List of processed messages
 """
 if not self.bot or not self.bot.is_ready:
 await self.initialize

 try:
 channel = self.bot.get_channel(channel_id)
 if not channel:
 logger.warning(f"Channel {channel_id} not found or not accessible")
 return []

 messages = []
 after_time = datetime.utcnow - timedelta(hours=hours_back)

 await self.rate_limiter.acquire

 async for message in channel.history(
 limit=limit,
 after=after_time
 ):
 processed_message = await self._process_message(message)
 if processed_message and self._is_crypto_relevant(processed_message["text"]):
 messages.append(processed_message)

 self.messages_processed += len(messages)
 self.api_calls_made += 1

 logger.info(
 "Discord channel history fetched",
 channel_id=channel_id,
 channel_name=channel.name,
 messages_count=len(messages)
 )

 return messages

 except Exception as e:
 self.last_error = e
 logger.error(f"Error fetching Discord channel history {channel_id}", error=e)
 return []

 async def fetch_all_monitored_channels(
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
 if not self.bot or not self.bot.is_ready:
 await self.initialize

 all_messages = []

 # Process all monitored channels
 for channel_id in self.bot.monitored_channels:
 try:
 messages = await self.fetch_channel_history(
 channel_id,
 limit=limit_per_channel,
 hours_back=hours_back
 )
 all_messages.extend(messages)

 # Pause between channels
 await asyncio.sleep(1)

 except Exception as e:
 logger.error(f"Error processing Discord channel {channel_id}", error=e)
 continue

 logger.info(
 "All monitored Discord channels processed",
 channels_processed=len(self.bot.monitored_channels),
 total_messages=len(all_messages),
 servers_connected=self.servers_connected
 )

 return all_messages

 async def search_messages(
 self,
 query: str,
 channel_ids: List[int] = None,
 limit: int = 100
 ) -> List[Dict[str, Any]]:
 """
 Search for messages in channels

 Note: Discord API not supports search by content.
 This method receives recent messages and filters their.

 Args:
 query: Search query
 channel_ids: Channels to search
 limit: Maximum number of results

 Returns:
 List[Dict[str, Any]]: Found messages
 """
 if not channel_ids:
 if not self.bot or not self.bot.is_ready:
 await self.initialize
 channel_ids = list(self.bot.monitored_channels)[:5] # Limit the search scope

 all_results = []
 query_lower = query.lower

 for channel_id in channel_ids:
 try:
 # Get more messages for search
 messages = await self.fetch_channel_history(
 channel_id,
 limit=200, # Larger limit for search
 hours_back=168 # Week
 )

 # Filter by query
 matching_messages = [
 msg for msg in messages
 if query_lower in msg.get('text', '').lower
 ]

 all_results.extend(matching_messages)

 except Exception as e:
 logger.error(f"Error searching in Discord channel {channel_id}", error=e)
 continue

 # Sort by time
 all_results.sort(
 key=lambda m: m.get('created_at', ''),
 reverse=True
 )

 result = all_results[:limit]

 logger.info(
 "Discord search completed",
 query=query,
 channels_searched=len(channel_ids),
 total_results=len(result)
 )

 return result

 async def start_real_time_monitoring(self, callback=None):
 """
 Start real-time monitoring

 Args:
 callback: Function for handling new messages
 """
 if not self.bot:
 await self.initialize

 # Set callback for new messages
 self.bot.set_message_callback(callback)

 logger.info(
 "Real-time Discord monitoring started",
 channels_monitored=self.channels_monitored,
 servers_connected=self.servers_connected
 )

 # Bot already started in initialize, just wait
 try:
 await self.bot.wait_until_ready
 while not self.bot.is_closed:
 await asyncio.sleep(1)
 except Exception as e:
 logger.error("Error in Discord real-time monitoring", error=e)
 raise

 def get_stats(self) -> Dict[str, Any]:
 """
 Get source statistics

 Returns:
 Dict[str, Any]: Operational statistics
 """
 bot_ready = self.bot.is_ready if self.bot else False

 return {
 "source": "discord",
 "messages_processed": self.messages_processed,
 "channels_monitored": self.channels_monitored,
 "servers_connected": self.servers_connected,
 "api_calls_made": self.api_calls_made,
 "last_error": str(self.last_error) if self.last_error else None,
 "crypto_symbols_tracked": len(self.crypto_symbols),
 "crypto_keywords_tracked": len(self.crypto_keywords),
 "initialized": bot_ready,
 "seen_messages_count": len(self.seen_messages),
 "bot_latency": round(self.bot.latency * 1000, 2) if self.bot else None # ms
 }


async def create_discord_source -> DiscordSentimentSource:
 """
 Factory function for creating a Discord data source

 Returns:
 DiscordSentimentSource: Configured data source
 """
 source = DiscordSentimentSource
 await source.initialize
 return source