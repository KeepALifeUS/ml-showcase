"""
Module analysis sentiments

Contains components for analysis sentiments in social networks
with using various approaches and models.
"""

from .realtime_analyzer import RealtimeSentimentAnalyzer
from .batch_analyzer import BatchSentimentAnalyzer
from .multilingual_analyzer import MultilingualSentimentAnalyzer
from .emoji_sentiment import EmojiSentimentAnalyzer
from .meme_analyzer import MemeAnalyzer
from .sarcasm_detector import SarcasmDetector

__all__ = [
    "RealtimeSentimentAnalyzer",
    "BatchSentimentAnalyzer",
    "MultilingualSentimentAnalyzer",
    "EmojiSentimentAnalyzer",
    "MemeAnalyzer",
    "SarcasmDetector",
]