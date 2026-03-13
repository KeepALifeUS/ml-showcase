"""
Transformer-based Sentiment Analysis for ML-Framework ML Sentiment Engine

Enterprise-grade BERT/FinBERT sentiment analysis with and async support.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re

import torch
import torch.nn.functional as F
from transformers import (
 AutoTokenizer, AutoModelForSequenceClassification,
 pipeline, BertTokenizer, BertForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import numpy as np

from ..utils.logger import get_logger
from ..utils.config import get_config
from ..utils.validators import SentimentScore, TextContent, validate_text_content, sanitize_text

logger = get_logger(__name__)


class TransformerModelManager:
 """Manager for transformer models"""

 def __init__(self):
 """Initialize the model manager"""
 config = get_config

 self.device = torch.device("cuda" if torch.cuda.is_available and config.ml.use_gpu else "cpu")
 self.models_cache = {}
 self.tokenizers_cache = {}
 self.pipelines_cache = {}

 # Model configuration
 self.model_configs = {
 "finbert": {
 "model_name": "ProsusAI/finbert",
 "labels": ["negative", "neutral", "positive"],
 "task": "sentiment-analysis"
 },
 "finbert_esg": {
 "model_name": "ProsusAI/finbert-esg",
 "labels": ["negative", "neutral", "positive"],
 "task": "sentiment-analysis"
 },
 "cardiffnlp_twitter": {
 "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
 "labels": ["negative", "neutral", "positive"],
 "task": "sentiment-analysis"
 },
 "distilbert_financetext": {
 "model_name": "ahmedrachid/FinanceInc-Sentiment-Analysis",
 "labels": ["negative", "neutral", "positive"],
 "task": "sentiment-analysis"
 }
 }

 # Performance metrics
 self.predictions_made = 0
 self.total_inference_time = 0.0
 self.model_loads = 0
 self.cache_hits = 0

 logger.info(f"Transformer models manager initialized on device: {self.device}")

 async def load_model(self, model_key: str) -> Tuple[Any, Any]:
 """
 Load a model and its tokenizer

 Args:
 model_key: Model key from configuration

 Returns:
 Tuple[Any, Any]: (model, tokenizer)
 """
 if model_key in self.models_cache:
 self.cache_hits += 1
 return self.models_cache[model_key], self.tokenizers_cache[model_key]

 if model_key not in self.model_configs:
 raise ValueError(f"Unknown model key: {model_key}")

 config = self.model_configs[model_key]
 model_name = config["model_name"]

 try:
 start_time = time.time

 # Load in a separate executor for blocking operations
 loop = asyncio.get_event_loop

 tokenizer = await loop.run_in_executor(
 None,
 lambda: AutoTokenizer.from_pretrained(model_name)
 )

 model = await loop.run_in_executor(
 None,
 lambda: AutoModelForSequenceClassification.from_pretrained(model_name)
 )

 # Transfer to device
 model = model.to(self.device)
 model.eval # Inference mode

 # Cache the model
 self.models_cache[model_key] = model
 self.tokenizers_cache[model_key] = tokenizer
 self.model_loads += 1

 load_time = time.time - start_time
 logger.info(
 f"Model loaded successfully",
 model_key=model_key,
 model_name=model_name,
 device=str(self.device),
 load_time_s=round(load_time, 2)
 )

 return model, tokenizer

 except Exception as e:
 logger.error(f"Failed to load model {model_key}", error=e)
 raise

 async def load_pipeline(self, model_key: str) -> Any:
 """
 Load a pipeline for the model

 Args:
 model_key: Model key

 Returns:
 Any: HuggingFace pipeline
 """
 if model_key in self.pipelines_cache:
 self.cache_hits += 1
 return self.pipelines_cache[model_key]

 if model_key not in self.model_configs:
 raise ValueError(f"Unknown model key: {model_key}")

 config = self.model_configs[model_key]

 try:
 start_time = time.time

 loop = asyncio.get_event_loop

 sentiment_pipeline = await loop.run_in_executor(
 None,
 lambda: pipeline(
 config["task"],
 model=config["model_name"],
 device=0 if self.device.type == "cuda" else -1
 )
 )

 self.pipelines_cache[model_key] = sentiment_pipeline

 load_time = time.time - start_time
 logger.info(
 f"Pipeline loaded successfully",
 model_key=model_key,
 load_time_s=round(load_time, 2)
 )

 return sentiment_pipeline

 except Exception as e:
 logger.error(f"Failed to load pipeline for {model_key}", error=e)
 raise

 def get_stats(self) -> Dict[str, Any]:
 """Get model manager statistics"""
 return {
 "device": str(self.device),
 "models_loaded": len(self.models_cache),
 "pipelines_loaded": len(self.pipelines_cache),
 "predictions_made": self.predictions_made,
 "avg_inference_time_ms": (
 (self.total_inference_time / max(self.predictions_made, 1)) * 1000
 ),
 "model_loads": self.model_loads,
 "cache_hits": self.cache_hits,
 "available_models": list(self.model_configs.keys)
 }


class FinBERTSentimentAnalyzer:
 """
 FinBERT-based sentiment analyzer for financial texts
 """

 def __init__(self, model_manager: TransformerModelManager):
 """
 Initialize the FinBERT analyzer

 Args:
 model_manager: Model manager
 """
 self.model_manager = model_manager
 self.model_key = "finbert"
 self.model = None
 self.tokenizer = None

 # Map labels to numeric values
 self.label_mapping = {
 "negative": -1.0,
 "neutral": 0.0,
 "positive": 1.0
 }

 async def initialize(self):
 """Initialize the model"""
 self.model, self.tokenizer = await self.model_manager.load_model(self.model_key)
 logger.info("FinBERT sentiment analyzer initialized")

 async def predict(self, text: str) -> SentimentScore:
 """
 Predict sentiment for text

 Args:
 text: Text to analyze

 Returns:
 SentimentScore: Analysis result
 """
 if not self.model or not self.tokenizer:
 await self.initialize

 try:
 start_time = time.time

 # Preprocess text
 cleaned_text = sanitize_text(text)
 if not validate_text_content(cleaned_text, "transformer"):
 raise ValueError("Invalid text content")

 # Tokenize
 inputs = self.tokenizer(
 cleaned_text,
 return_tensors="pt",
 truncation=True,
 padding=True,
 max_length=512
 )

 # Transfer to device
 inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items}

 # Inference
 with torch.no_grad:
 outputs = self.model(**inputs)
 predictions = F.softmax(outputs.logits, dim=-1)

 # Get probabilities
 probs = predictions.cpu.numpy[0]

 # Find the most probable class
 predicted_class_id = np.argmax(probs)
 predicted_label = self.model_manager.model_configs[self.model_key]["labels"][predicted_class_id]

 # Convert to sentiment score
 sentiment_value = self.label_mapping[predicted_label]
 confidence = float(probs[predicted_class_id])

 # Record metrics
 inference_time = time.time - start_time
 self.model_manager.predictions_made += 1
 self.model_manager.total_inference_time += inference_time

 logger.debug(
 "FinBERT prediction completed",
 text_length=len(cleaned_text),
 predicted_label=predicted_label,
 confidence=confidence,
 inference_time_ms=round(inference_time * 1000, 2)
 )

 return SentimentScore(
 value=sentiment_value,
 confidence=confidence,
 model_name="finbert"
 )

 except Exception as e:
 logger.error("FinBERT prediction failed", error=e)
 raise

 async def predict_batch(self, texts: List[str], batch_size: int = 8) -> List[SentimentScore]:
 """
 Batch prediction for a list of texts

 Args:
 texts: List of texts for analysis
 batch_size: Batch size

 Returns:
 List[SentimentScore]: Analysis results
 """
 if not self.model or not self.tokenizer:
 await self.initialize

 results = []

 # Process batches
 for i in range(0, len(texts), batch_size):
 batch_texts = texts[i:i + batch_size]

 try:
 start_time = time.time

 # Preprocess
 cleaned_texts = [sanitize_text(text) for text in batch_texts]
 valid_texts = [text for text in cleaned_texts if text and len(text) > 0]

 if not valid_texts:
 # Add empty results for invalid texts
 results.extend([
 SentimentScore(value=0.0, confidence=0.0, model_name="finbert")
 for _ in batch_texts
 ])
 continue

 # Tokenize batch
 inputs = self.tokenizer(
 valid_texts,
 return_tensors="pt",
 truncation=True,
 padding=True,
 max_length=512
 )

 # Transfer to device
 inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items}

 # Run batch inference
 with torch.no_grad:
 outputs = self.model(**inputs)
 predictions = F.softmax(outputs.logits, dim=-1)

 probs = predictions.cpu.numpy

 # Process results
 batch_results = []
 for j, prob in enumerate(probs):
 predicted_class_id = np.argmax(prob)
 predicted_label = self.model_manager.model_configs[self.model_key]["labels"][predicted_class_id]

 sentiment_value = self.label_mapping[predicted_label]
 confidence = float(prob[predicted_class_id])

 batch_results.append(SentimentScore(
 value=sentiment_value,
 confidence=confidence,
 model_name="finbert"
 ))

 results.extend(batch_results)

 # Record metrics
 inference_time = time.time - start_time
 self.model_manager.predictions_made += len(batch_texts)
 self.model_manager.total_inference_time += inference_time

 logger.debug(
 "FinBERT batch prediction completed",
 batch_size=len(batch_texts),
 inference_time_ms=round(inference_time * 1000, 2)
 )

 except Exception as e:
 logger.error("FinBERT batch prediction failed", error=e)
 # Add zero results for the failed batch
 results.extend([
 SentimentScore(value=0.0, confidence=0.0, model_name="finbert")
 for _ in batch_texts
 ])

 return results


class TwitterRoBERTaSentimentAnalyzer:
 """
 RoBERTa-based sentiment analyzer specially for Twitter/social media
 """

 def __init__(self, model_manager: TransformerModelManager):
 """Initialize the Twitter RoBERTa analyzer"""
 self.model_manager = model_manager
 self.model_key = "cardiffnlp_twitter"
 self.pipeline = None

 async def initialize(self):
 """Initialize the pipeline"""
 self.pipeline = await self.model_manager.load_pipeline(self.model_key)
 logger.info("Twitter RoBERTa sentiment analyzer initialized")

 async def predict(self, text: str) -> SentimentScore:
 """
 Predict sentiment for social media text

 Args:
 text: Text to analyze

 Returns:
 SentimentScore: Analysis result
 """
 if not self.pipeline:
 await self.initialize

 try:
 start_time = time.time

 # Preprocess for social media
 cleaned_text = self._preprocess_social_text(text)

 if not validate_text_content(cleaned_text, "twitter"):
 raise ValueError("Invalid social media text content")

 # Run inference through the pipeline
 loop = asyncio.get_event_loop
 result = await loop.run_in_executor(
 None,
 lambda: self.pipeline(cleaned_text)
 )

 # Process the result
 label = result["label"].lower
 confidence = result["score"]

 # Map labels
 label_mapping = {
 "negative": -1.0,
 "neutral": 0.0,
 "positive": 1.0
 }

 sentiment_value = label_mapping.get(label, 0.0)

 # Record metrics
 inference_time = time.time - start_time
 self.model_manager.predictions_made += 1
 self.model_manager.total_inference_time += inference_time

 logger.debug(
 "Twitter RoBERTa prediction completed",
 predicted_label=label,
 confidence=confidence,
 inference_time_ms=round(inference_time * 1000, 2)
 )

 return SentimentScore(
 value=sentiment_value,
 confidence=confidence,
 model_name="twitter_roberta"
 )

 except Exception as e:
 logger.error("Twitter RoBERTa prediction failed", error=e)
 raise

 def _preprocess_social_text(self, text: str) -> str:
 """
 Preprocess text for social media

 Args:
 text: Original text

 Returns:
 str: Processed text
 """
 # Basic cleanup
 cleaned = sanitize_text(text)

 # Social media specific handling
 # Normalize URLs
 cleaned = re.sub(r'http[s]?://[^\s]+', '[URL]', cleaned)

 # Normalize mentions
 cleaned = re.sub(r'@\w+', '[USER]', cleaned)

 # Preserve emoji (important for sentiment)
 # Normalize recurring characters
 cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)

 return cleaned


class TransformerSentimentEnsemble:
 """
 Ensemble transformer models for robust sentiment analysis
 """

 def __init__(self):
 """Initialize the ensemble"""
 self.model_manager = TransformerModelManager
 self.finbert_analyzer = FinBERTSentimentAnalyzer(self.model_manager)
 self.twitter_analyzer = TwitterRoBERTaSentimentAnalyzer(self.model_manager)

 # Ensemble weights (configurable)
 self.weights = {
 "finbert": 0.6, # Large weight for financial texts
 "twitter_roberta": 0.4 # Smaller weight for social media
 }

 async def initialize(self):
 """Initialize all analyzers"""
 await self.finbert_analyzer.initialize
 await self.twitter_analyzer.initialize
 logger.info("Transformer sentiment ensemble initialized")

 async def predict(
 self,
 text: str,
 source: str = "unknown",
 use_weighted_ensemble: bool = True
 ) -> SentimentScore:
 """
 Ensemble prediction with adaptive weights

 Args:
 text: Text to analyze
 source: Text source for weight adaptation
 use_weighted_ensemble: Whether to use weighted averaging

 Returns:
 SentimentScore: Ensemble analysis result
 """
 try:
 start_time = time.time

 # Get predictions from all models
 finbert_result = await self.finbert_analyzer.predict(text)
 twitter_result = await self.twitter_analyzer.predict(text)

 if not use_weighted_ensemble:
 # Simple averaging
 avg_sentiment = (finbert_result.value + twitter_result.value) / 2
 avg_confidence = (finbert_result.confidence + twitter_result.confidence) / 2
 else:
 # Adapt weights based on source
 weights = self._adapt_weights_for_source(source)

 # Weighted average
 avg_sentiment = (
 finbert_result.value * weights["finbert"] +
 twitter_result.value * weights["twitter_roberta"]
 )

 avg_confidence = (
 finbert_result.confidence * weights["finbert"] +
 twitter_result.confidence * weights["twitter_roberta"]
 )

 inference_time = time.time - start_time

 logger.debug(
 "Transformer ensemble prediction completed",
 source=source,
 finbert_sentiment=finbert_result.value,
 twitter_sentiment=twitter_result.value,
 ensemble_sentiment=avg_sentiment,
 ensemble_confidence=avg_confidence,
 inference_time_ms=round(inference_time * 1000, 2)
 )

 return SentimentScore(
 value=avg_sentiment,
 confidence=avg_confidence,
 model_name="transformer_ensemble"
 )

 except Exception as e:
 logger.error("Transformer ensemble prediction failed", error=e)
 raise

 def _adapt_weights_for_source(self, source: str) -> Dict[str, float]:
 """
 Adapt weights based on the data source

 Args:
 source: Data source

 Returns:
 Dict[str, float]: Adapted weights
 """
 if source in ["twitter", "reddit", "telegram", "discord"]:
 # Higher weight for social media model
 return {
 "finbert": 0.3,
 "twitter_roberta": 0.7
 }
 elif source in ["news", "bloomberg", "reuters"]:
 # Higher weight for financial model
 return {
 "finbert": 0.8,
 "twitter_roberta": 0.2
 }
 else:
 # Default weights
 return self.weights

 def get_stats(self) -> Dict[str, Any]:
 """Get ensemble statistics"""
 base_stats = self.model_manager.get_stats
 base_stats["ensemble_weights"] = self.weights
 base_stats["model_type"] = "transformer_ensemble"
 return base_stats


# Factory functions
async def create_finbert_analyzer -> FinBERTSentimentAnalyzer:
 """
 Factory function for creating a FinBERT analyzer

 Returns:
 FinBERTSentimentAnalyzer: Initialized analyzer
 """
 manager = TransformerModelManager
 analyzer = FinBERTSentimentAnalyzer(manager)
 await analyzer.initialize
 return analyzer


async def create_twitter_analyzer -> TwitterRoBERTaSentimentAnalyzer:
 """
 Factory function for creating a Twitter analyzer

 Returns:
 TwitterRoBERTaSentimentAnalyzer: Initialized analyzer
 """
 manager = TransformerModelManager
 analyzer = TwitterRoBERTaSentimentAnalyzer(manager)
 await analyzer.initialize
 return analyzer


async def create_transformer_ensemble -> TransformerSentimentEnsemble:
 """
 Factory function for creating a transformer ensemble

 Returns:
 TransformerSentimentEnsemble: Initialized ensemble
 """
 ensemble = TransformerSentimentEnsemble
 await ensemble.initialize
 return ensemble