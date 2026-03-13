"""
Ensemble Sentiment Model for ML-Framework ML Sentiment Engine

Enterprise-grade ensemble combining multiple sentiment analysis approaches.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .transformer_sentiment import TransformerSentimentEnsemble
from .vader_sentiment import VADERSentimentAnalyzer
from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.validators import SentimentScore, validate_text_content

logger = get_logger(__name__)


class EnsembleSentimentModel:
 """
 Advanced ensemble model combining multiple sentiment analyzers

 Features:
 - Multi-model ensemble (Transformers + VADER)
 - Adaptive weighting based on source type
 - Confidence-based model selection
 - Performance monitoring
 - Fallback mechanisms
 """

 def __init__(self):
 """Initialize the ensemble model"""
 config = get_config

 # Constituent models
 self.transformer_ensemble: Optional[TransformerSentimentEnsemble] = None
 self.vader_analyzer: Optional[VADERSentimentAnalyzer] = None

 # Ensemble configuration
 self.base_weights = config.ml.ensemble_weights
 self.min_confidence_threshold = 0.1

 # Performance metrics
 self.predictions_made = 0
 self.total_inference_time = 0.0
 self.model_failures = {
 "transformer": 0,
 "vader": 0
 }

 # Adaptive weighting history
 self.performance_history = {
 "transformer": [],
 "vader": []
 }

 logger.info("Ensemble sentiment model initialized")

 async def initialize(self):
 """Initialize all constituent models"""
 try:
 # Initialize transformer ensemble
 self.transformer_ensemble = TransformerSentimentEnsemble
 await self.transformer_ensemble.initialize

 # Initialize VADER
 self.vader_analyzer = VADERSentimentAnalyzer

 logger.info("All ensemble models initialized successfully")

 except Exception as e:
 logger.error("Failed to initialize ensemble models", error=e)
 raise

 async def predict(
 self,
 text: str,
 source: str = "unknown",
 use_adaptive_weighting: bool = True
 ) -> SentimentScore:
 """
 Ensemble prediction with adaptive weighting

 Args:
 text: Text to analyze
 source: Data source for adaptation
 use_adaptive_weighting: Use adaptive weights

 Returns:
 SentimentScore: Ensemble analysis result
 """
 if not self.transformer_ensemble or not self.vader_analyzer:
 await self.initialize

 try:
 start_time = time.time

 if not validate_text_content(text, "ensemble"):
 raise ValueError("Invalid text content")

 # Get predictions from all models
 model_results = await self._get_model_predictions(text, source)

 # Select ensemble strategy
 if use_adaptive_weighting:
 ensemble_result = self._adaptive_ensemble(model_results, source)
 else:
 ensemble_result = self._weighted_ensemble(model_results, source)

 # Record metrics
 inference_time = time.time - start_time
 self.predictions_made += 1
 self.total_inference_time += inference_time

 # Log the result
 logger.debug(
 "Ensemble prediction completed",
 source=source,
 transformer_sentiment=model_results.get("transformer", {}).get("value", 0),
 vader_sentiment=model_results.get("vader", {}).get("value", 0),
 ensemble_sentiment=ensemble_result.value,
 ensemble_confidence=ensemble_result.confidence,
 inference_time_ms=round(inference_time * 1000, 2)
 )

 return ensemble_result

 except Exception as e:
 logger.error("Ensemble prediction failed", error=e)
 raise

 async def _get_model_predictions(
 self,
 text: str,
 source: str
 ) -> Dict[str, SentimentScore]:
 """
 Get predictions from all models with error handling

 Args:
 text: Text to analyze
 source: Data source

 Returns:
 Dict[str, SentimentScore]: Results from each model
 """
 results = {}

 # Transformer ensemble prediction
 try:
 transformer_result = await self.transformer_ensemble.predict(text, source)
 results["transformer"] = transformer_result

 # Update performance history
 self.performance_history["transformer"].append(transformer_result.confidence)
 if len(self.performance_history["transformer"]) > 100:
 self.performance_history["transformer"].pop(0)

 except Exception as e:
 logger.warning("Transformer ensemble prediction failed", error=e)
 self.model_failures["transformer"] += 1
 results["transformer"] = SentimentScore(
 value=0.0, confidence=0.0, model_name="transformer_fallback"
 )

 # VADER prediction
 try:
 vader_result = await self.vader_analyzer.predict(text)
 results["vader"] = vader_result

 # Update performance history
 self.performance_history["vader"].append(vader_result.confidence)
 if len(self.performance_history["vader"]) > 100:
 self.performance_history["vader"].pop(0)

 except Exception as e:
 logger.warning("VADER prediction failed", error=e)
 self.model_failures["vader"] += 1
 results["vader"] = SentimentScore(
 value=0.0, confidence=0.0, model_name="vader_fallback"
 )

 return results

 def _adaptive_ensemble(
 self,
 model_results: Dict[str, SentimentScore],
 source: str
 ) -> SentimentScore:
 """
 Adaptive ensemble based on performance and confidence

 Args:
 model_results: Results from models
 source: Data source

 Returns:
 SentimentScore: Adaptive ensemble result
 """
 # Default weights for the source
 base_weights = self._get_source_weights(source)

 # Adapt weights based on confidence
 adaptive_weights = {}
 total_confidence = 0

 for model_name, result in model_results.items:
 if result.confidence > self.min_confidence_threshold:
 # Weight = base weight * confidence * performance_factor
 performance_factor = self._get_performance_factor(model_name)
 adaptive_weight = (
 base_weights.get(model_name, 0.5) *
 result.confidence *
 performance_factor
 )
 adaptive_weights[model_name] = adaptive_weight
 total_confidence += result.confidence
 else:
 adaptive_weights[model_name] = 0

 # Normalize weights
 if sum(adaptive_weights.values) > 0:
 total_weight = sum(adaptive_weights.values)
 adaptive_weights = {k: v / total_weight for k, v in adaptive_weights.items}
 else:
 # Fallback to equal weights if all confidence is low
 adaptive_weights = {k: 1.0 / len(model_results) for k in model_results.keys}

 # Compute ensemble sentiment
 ensemble_sentiment = sum(
 result.value * adaptive_weights[model_name]
 for model_name, result in model_results.items
 )

 # Ensemble confidence (weighted average)
 ensemble_confidence = sum(
 result.confidence * adaptive_weights[model_name]
 for model_name, result in model_results.items
 )

 return SentimentScore(
 value=ensemble_sentiment,
 confidence=ensemble_confidence,
 model_name="adaptive_ensemble"
 )

 def _weighted_ensemble(
 self,
 model_results: Dict[str, SentimentScore],
 source: str
 ) -> SentimentScore:
 """
 Simple weighted ensemble

 Args:
 model_results: Results from models
 source: Data source

 Returns:
 SentimentScore: Weighted ensemble result
 """
 weights = self._get_source_weights(source)

 # Weighted average sentiment
 ensemble_sentiment = sum(
 result.value * weights.get(self._map_model_name(model_name), 0.5)
 for model_name, result in model_results.items
 )

 # Weighted average confidence
 ensemble_confidence = sum(
 result.confidence * weights.get(self._map_model_name(model_name), 0.5)
 for model_name, result in model_results.items
 )

 return SentimentScore(
 value=ensemble_sentiment,
 confidence=ensemble_confidence,
 model_name="weighted_ensemble"
 )

 def _get_source_weights(self, source: str) -> Dict[str, float]:
 """
 Get weights based on source

 Args:
 source: Data source

 Returns:
 Dict[str, float]: Weights for each model
 """
 if source in ["twitter", "reddit", "telegram", "discord"]:
 # Social media - VADER works better
 return {
 "finbert": 0.3,
 "vader": 0.4,
 "textblob": 0.1,
 "crypto_specific": 0.2
 }
 elif source in ["news", "bloomberg", "reuters"]:
 # News - transformers better
 return {
 "finbert": 0.5,
 "vader": 0.2,
 "textblob": 0.1,
 "crypto_specific": 0.2
 }
 else:
 # Default weights
 return self.base_weights

 def _map_model_name(self, model_name: str) -> str:
 """Map model names"""
 mapping = {
 "transformer": "finbert",
 "vader": "vader"
 }
 return mapping.get(model_name, model_name)

 def _get_performance_factor(self, model_name: str) -> float:
 """
 Calculate performance factor for model

 Args:
 model_name: Model name

 Returns:
 float: Performance factor (0.5 - 1.5)
 """
 if model_name not in self.performance_history:
 return 1.0

 history = self.performance_history[model_name]
 if not history:
 return 1.0

 # Average confidence over recent predictions
 avg_confidence = np.mean(history)

 # Convert to performance factor
 # High confidence -> high factor
 return 0.5 + avg_confidence

 async def predict_batch(self, texts: List[str], source: str = "unknown") -> List[SentimentScore]:
 """
 Batch prediction for the ensemble

 Args:
 texts: List of texts
 source: Data source

 Returns:
 List[SentimentScore]: Ensemble results
 """
 results = []

 # Process small batches in parallel
 batch_size = 10
 for i in range(0, len(texts), batch_size):
 batch_texts = texts[i:i + batch_size]

 # Create tasks for parallel execution
 tasks = [self.predict(text, source) for text in batch_texts]
 batch_results = await asyncio.gather(*tasks, return_exceptions=True)

 # Handle exceptions
 for result in batch_results:
 if isinstance(result, Exception):
 logger.error("Batch prediction failed for text", error=result)
 results.append(SentimentScore(value=0.0, confidence=0.0, model_name="ensemble_error"))
 else:
 results.append(result)

 return results

 def get_stats(self) -> Dict[str, Any]:
 """Get ensemble statistics"""
 transformer_stats = self.transformer_ensemble.get_stats if self.transformer_ensemble else {}
 vader_stats = self.vader_analyzer.get_stats if self.vader_analyzer else {}

 return {
 "model_type": "ensemble",
 "predictions_made": self.predictions_made,
 "avg_inference_time_ms": (
 (self.total_inference_time / max(self.predictions_made, 1)) * 1000
 ),
 "model_failures": self.model_failures,
 "base_weights": self.base_weights,
 "performance_history_length": {
 k: len(v) for k, v in self.performance_history.items
 },
 "constituent_models": {
 "transformer": transformer_stats,
 "vader": vader_stats
 }
 }


async def create_ensemble_model -> EnsembleSentimentModel:
 """
 Factory function for creating an ensemble model

 Returns:
 EnsembleSentimentModel: Initialized ensemble model
 """
 model = EnsembleSentimentModel
 await model.initialize
 return model