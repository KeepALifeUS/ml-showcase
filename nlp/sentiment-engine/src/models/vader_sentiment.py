"""
VADER Sentiment Analysis for ML-Framework ML Sentiment Engine

Enterprise-grade VADER implementation with .
"""

import asyncio
import time
from typing import Any, Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ..utils.logger import get_logger
from ..utils.validators import SentimentScore, sanitize_text, validate_text_content

logger = get_logger(__name__)


class VADERSentimentAnalyzer:
 """VADER-based sentiment analyzer for social media content"""

 def __init__(self):
 """Initialize the VADER analyzer"""
 self.analyzer = SentimentIntensityAnalyzer
 self.predictions_made = 0
 self.total_inference_time = 0.0

 # Custom rules for crypto-terms
 crypto_booster_dict = {
 'moon': 2.0, 'lambo': 2.0, 'hodl': 1.5, 'diamond_hands': 2.0,
 'to_the_moon': 2.5, 'bullish': 2.0, 'bearish': -2.0, 'fud': -2.0,
 'fomo': 1.0, 'pump': 1.5, 'dump': -2.0, 'rug_pull': -3.0,
 'whale': 0.5, 'dip': -0.5, 'ath': 2.0, 'rekt': -2.5
 }

 # Update the VADER lexicon
 self.analyzer.lexicon.update(crypto_booster_dict)

 logger.info("VADER sentiment analyzer initialized with crypto lexicon")

 async def predict(self, text: str) -> SentimentScore:
 """Predict sentiment with VADER"""
 try:
 start_time = time.time

 cleaned_text = sanitize_text(text)
 if not validate_text_content(cleaned_text, "vader"):
 raise ValueError("Invalid text content")

 # VADER analysis in executor
 loop = asyncio.get_event_loop
 scores = await loop.run_in_executor(
 None,
 self.analyzer.polarity_scores,
 cleaned_text
 )

 # Convert compound score (-1 to 1)
 compound_score = scores['compound']

 # Confidence based on intensity
 confidence = abs(compound_score)

 inference_time = time.time - start_time
 self.predictions_made += 1
 self.total_inference_time += inference_time

 logger.debug(
 "VADER prediction completed",
 compound_score=compound_score,
 confidence=confidence,
 inference_time_ms=round(inference_time * 1000, 2)
 )

 return SentimentScore(
 value=compound_score,
 confidence=confidence,
 model_name="vader"
 )

 except Exception as e:
 logger.error("VADER prediction failed", error=e)
 raise

 async def predict_batch(self, texts: List[str]) -> List[SentimentScore]:
 """Batch prediction with VADER"""
 results = []

 for text in texts:
 try:
 result = await self.predict(text)
 results.append(result)
 except Exception:
 results.append(SentimentScore(value=0.0, confidence=0.0, model_name="vader"))

 return results

 def get_stats(self) -> Dict[str, Any]:
 """Get VADER statistics"""
 return {
 "model_type": "vader",
 "predictions_made": self.predictions_made,
 "avg_inference_time_ms": (
 (self.total_inference_time / max(self.predictions_made, 1)) * 1000
 ),
 "lexicon_size": len(self.analyzer.lexicon)
 }


async def create_vader_analyzer -> VADERSentimentAnalyzer:
 """Factory function for creating a VADER analyzer"""
 return VADERSentimentAnalyzer