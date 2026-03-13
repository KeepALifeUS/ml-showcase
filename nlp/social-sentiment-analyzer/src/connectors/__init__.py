"""
Connectors to social platforms

Module contains connectors for integration with various social
platforms with application  enterprise-patterns.
"""

from .twitter_connector import TwitterConnector
from .reddit_connector import RedditConnector
from .telegram_connector import TelegramConnector
from .discord_connector import DiscordConnector
from .youtube_connector import YouTubeConnector
from .tiktok_connector import TikTokConnector

__all__ = [
    "TwitterConnector",
    "RedditConnector", 
    "TelegramConnector",
    "DiscordConnector",
    "YouTubeConnector",
    "TikTokConnector",
]