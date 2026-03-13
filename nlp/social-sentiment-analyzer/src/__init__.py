"""
Enterprise Social Media Sentiment Analyzer for Crypto Markets

This module provides full-featured system analysis sentiments
in social networks for cryptocurrency markets with
enterprise-patterns.

Main components:
- Connectors to social platforms (Twitter, Reddit, Telegram and etc.)
- System streaming processing data in real time
- ML-model for analysis sentiments with support many languages
- Aggregation data with various sources
- API for integration with trading systems
- System monitoring and alerts
"""

from typing import Dict, Any
import structlog

__version__ = "1.0.0"
__author__ = "ML-Framework ML Team"
__email__ = "ml-team@ml-framework.io"

# Configuration logging standard
logger = structlog.get_logger(__name__)

# Export main components
from .api.rest_api import SocialSentimentAPI
from .analysis.realtime_analyzer import RealtimeSentimentAnalyzer
from .streaming.twitter_stream import TwitterStreamProcessor
from .aggregation.sentiment_aggregator import SentimentAggregator
from .trends.trending_topics import TrendingTopicsDetector
from .monitoring.dashboard_backend import MonitoringDashboard

__all__ = [
    "SocialSentimentAPI",
    "RealtimeSentimentAnalyzer", 
    "TwitterStreamProcessor",
    "SentimentAggregator",
    "TrendingTopicsDetector",
    "MonitoringDashboard",
    "__version__",
]

def get_version() -> str:
    """Get version package."""
    return __version__

def get_system_info() -> Dict[str, Any]:
    """Get information about system."""
    import sys
    import platform
    
    return {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
    }

# Health Check
def health_check() -> Dict[str, Any]:
    """Validation state system."""
    try:
        import torch
        import transformers
        import fastapi
        
        return {
            "status": "healthy",
            "components": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "fastapi": fastapi.__version__,
            },
            "system_info": get_system_info(),
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "system_info": get_system_info(),
        }

logger.info("Social Sentiment Analyzer initialized", version=__version__)