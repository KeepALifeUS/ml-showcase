"""
Analyzer sentiments in real time

High-performance analyzer for processing flows social data
in real time with enterprise patterns.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import structlog
from circuitbreaker import CircuitBreaker
import aioredis

from ..utils.config import Config
from ..monitoring.metrics_collector import MetricsCollector
from ..nlp.preprocessor import TextPreprocessor
from ..ml.ensemble_model import EnsembleModel

logger = structlog.get_logger(__name__)

@dataclass
class SentimentResult:
    """Result analysis sentiment."""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]  # Detailed estimation
    crypto_symbols: List[str]
    processing_time_ms: float
    model_version: str
    timestamp: datetime

class RealtimeSentimentAnalyzer:
    """
    Enterprise analyzer sentiments in real time
    
    Features:
    - High performance (>1000 messages/sec)
    - Ensemble model for accuracy
    - Crypto-specific settings
    - Circuit breaker and fallback
    - Redis caching
    - Batch processing for optimization
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.metrics = MetricsCollector("realtime_sentiment_analyzer")
        self.logger = logger.bind(component="realtime_sentiment_analyzer")
        
        # Initialization components
        self.preprocessor = TextPreprocessor()
        self.ensemble_model: Optional[EnsembleModel] = None
        self._redis: Optional[aioredis.Redis] = None
        
        # Circuit breaker for protection from overloading
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30,
            expected_exception=Exception
        )
        
        # Settings performance
        self.batch_size = config.realtime_batch_size or 32
        self.max_queue_size = config.realtime_max_queue_size or 1000
        self.processing_timeout = config.realtime_timeout or 1.0  # seconds
        
        # Queue for batch processing
        self._input_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._result_cache = {}
        self._cache_ttl = 300  # 5 minutes caching
        
        # Statistics performance
        self._processed_count = 0
        self._error_count = 0
        self._start_time = time.time()
        
        # Crypto-specific weights
        self.crypto_sentiment_weights = {
            "moon": 0.8,      # Very positively
            "lambo": 0.7,     # Positively 
            "hodl": 0.6,      # Moderately positively
            "dump": -0.8,     # Very negatively
            "crash": -0.9,    # Extremely negatively
            "bear": -0.6,     # Negatively
            "bull": 0.7,      # Positively
            "diamond hands": 0.8,  # Very positively
            "paper hands": -0.5,   # Negatively
            "whale": 0.3,     # Weakly positively (can be and negatively)
            "fomo": -0.2,     # Weakly negatively
            "fud": -0.7,      # Negatively
        }
    
    async def initialize(self) -> None:
        """Initialization analyzer."""
        try:
            # Initialization Redis
            self._redis = aioredis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Initialization ensemble model
            self.ensemble_model = EnsembleModel(self.config)
            await self.ensemble_model.initialize()
            
            # Launch batch processor
            asyncio.create_task(self._batch_processor())
            
            self.logger.info("Real-time sentiment analyzer initialized",
                           batch_size=self.batch_size,
                           max_queue_size=self.max_queue_size)
            
        except Exception as e:
            self.logger.error("Failed to initialize analyzer", error=str(e))
            raise
    
    @CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    async def analyze_sentiment(
        self,
        text: str,
        platform: str = "unknown",
        priority: int = 1  # 1=high, 2=medium, 3=low
    ) -> SentimentResult:
        """
        Analysis sentiment one messages
        
        Args:
            text: Text for analysis
            platform: Platform source
            priority: Priority processing
        """
        
        start_time = time.time()
        
        try:
            # Validation cache
            cache_key = f"sentiment:{hash(text)}:{platform}"
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                self.metrics.increment("cache_hits")
                return cached_result
            
            # Preprocessing text
            preprocessed_text = self.preprocessor.preprocess(text)
            
            # Extraction crypto symbols
            crypto_symbols = self.preprocessor.extract_crypto_symbols(text)
            
            # Analysis through ensemble model
            sentiment_scores = await self.ensemble_model.predict(preprocessed_text)
            
            # Application crypto-specific weights
            adjusted_scores = self._apply_crypto_weights(
                sentiment_scores, preprocessed_text, crypto_symbols
            )
            
            # Determination final sentiment
            sentiment = self._determine_sentiment(adjusted_scores)
            confidence = max(adjusted_scores.values())
            
            # Creation result
            processing_time = (time.time() - start_time) * 1000
            
            result = SentimentResult(
                text=text,
                sentiment=sentiment,
                confidence=confidence,
                scores=adjusted_scores,
                crypto_symbols=crypto_symbols,
                processing_time_ms=processing_time,
                model_version=self.ensemble_model.version,
                timestamp=datetime.now()
            )
            
            # Saving in cache
            await self._save_to_cache(cache_key, result)
            
            # Metrics
            self.metrics.increment("messages_processed")
            self.metrics.histogram("processing_time_ms", processing_time)
            self._processed_count += 1
            
            return result
            
        except Exception as e:
            self.logger.error("Sentiment analysis failed", text=text[:100], error=str(e))
            self.metrics.increment("analysis_errors")
            self._error_count += 1
            
            # Fallback result
            return SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.5,
                scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                crypto_symbols=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version="fallback",
                timestamp=datetime.now()
            )
    
    async def analyze_batch(
        self,
        texts: List[str],
        platform: str = "unknown"
    ) -> List[SentimentResult]:
        """Batch analysis for optimization performance."""
        
        if not texts:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # Preprocessing total batch
            preprocessed_texts = [
                self.preprocessor.preprocess(text) for text in texts
            ]
            
            # Batch prediction through ensemble
            batch_scores = await self.ensemble_model.predict_batch(preprocessed_texts)
            
            # Processing results
            for i, (text, scores) in enumerate(zip(texts, batch_scores)):
                crypto_symbols = self.preprocessor.extract_crypto_symbols(text)
                
                # Application crypto weights
                adjusted_scores = self._apply_crypto_weights(
                    scores, preprocessed_texts[i], crypto_symbols
                )
                
                sentiment = self._determine_sentiment(adjusted_scores)
                confidence = max(adjusted_scores.values())
                
                result = SentimentResult(
                    text=text,
                    sentiment=sentiment,
                    confidence=confidence,
                    scores=adjusted_scores,
                    crypto_symbols=crypto_symbols,
                    processing_time_ms=0,  # Will be set later
                    model_version=self.ensemble_model.version,
                    timestamp=datetime.now()
                )
                
                results.append(result)
            
            # Update time processing
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(results)
            
            for result in results:
                result.processing_time_ms = avg_time
            
            self.metrics.increment("batch_processed", len(results))
            self.metrics.histogram("batch_processing_time_ms", total_time)
            
            return results
            
        except Exception as e:
            self.logger.error("Batch analysis failed", batch_size=len(texts), error=str(e))
            self.metrics.increment("batch_errors")
            
            # Fallback for total batch
            return [
                SentimentResult(
                    text=text,
                    sentiment="neutral",
                    confidence=0.5,
                    scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                    crypto_symbols=[],
                    processing_time_ms=0,
                    model_version="fallback",
                    timestamp=datetime.now()
                )
                for text in texts
            ]
    
    async def stream_analyze(
        self,
        message_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[SentimentResult, None]:
        """Streaming analysis with optimization performance."""
        
        batch_buffer = []
        last_flush_time = time.time()
        
        async for message in message_stream:
            try:
                text = message.get("text", "")
                platform = message.get("platform", "unknown")
                
                if not text:
                    continue
                
                # Addition in batch buffer
                batch_buffer.append((text, platform, message))
                
                # Conditions for flush batch
                should_flush = (
                    len(batch_buffer) >= self.batch_size or
                    (time.time() - last_flush_time) > self.processing_timeout
                )
                
                if should_flush:
                    # Processing batch
                    texts = [item[0] for item in batch_buffer]
                    platforms = [item[1] for item in batch_buffer]
                    messages = [item[2] for item in batch_buffer]
                    
                    # Analysis batch
                    results = await self.analyze_batch(texts, platforms[0])
                    
                    # Yield results with additional data
                    for result, original_msg in zip(results, messages):
                        # Addition metadata from original messages
                        enhanced_result = result
                        enhanced_result.platform = original_msg.get("platform")
                        enhanced_result.message_id = original_msg.get("id")
                        enhanced_result.author = original_msg.get("author")
                        
                        yield enhanced_result
                    
                    # Cleanup buffer
                    batch_buffer.clear()
                    last_flush_time = time.time()
                
            except Exception as e:
                self.logger.error("Stream processing error", error=str(e))
                continue
        
        # Flush remaining messages
        if batch_buffer:
            texts = [item[0] for item in batch_buffer]
            results = await self.analyze_batch(texts)
            
            for result in results:
                yield result
    
    def _apply_crypto_weights(
        self,
        scores: Dict[str, float],
        text: str,
        crypto_symbols: List[str]
    ) -> Dict[str, float]:
        """Application crypto-specific weights to estimates."""
        
        text_lower = text.lower()
        adjustment = 0.0
        
        # Application weights for crypto terms
        for term, weight in self.crypto_sentiment_weights.items():
            if term in text_lower:
                adjustment += weight * 0.1  # Limitation influence
        
        # Bonus for mention crypto symbols
        if crypto_symbols:
            symbol_bonus = min(len(crypto_symbols) * 0.05, 0.2)  # Max 20%
            adjustment += symbol_bonus if scores.get("positive", 0) > scores.get("negative", 0) else -symbol_bonus
        
        # Application adjustment to estimates
        adjusted_scores = scores.copy()
        
        if adjustment > 0:
            adjusted_scores["positive"] = min(1.0, adjusted_scores.get("positive", 0) + abs(adjustment))
            adjusted_scores["negative"] = max(0.0, adjusted_scores.get("negative", 0) - abs(adjustment) * 0.5)
        elif adjustment < 0:
            adjusted_scores["negative"] = min(1.0, adjusted_scores.get("negative", 0) + abs(adjustment))
            adjusted_scores["positive"] = max(0.0, adjusted_scores.get("positive", 0) - abs(adjustment) * 0.5)
        
        # Normalization
        total = sum(adjusted_scores.values())
        if total > 0:
            adjusted_scores = {k: v / total for k, v in adjusted_scores.items()}
        
        return adjusted_scores
    
    def _determine_sentiment(self, scores: Dict[str, float]) -> str:
        """Determination final sentiment."""
        
        # Minimum threshold confidence
        min_confidence = 0.4
        
        max_score = max(scores.values())
        if max_score < min_confidence:
            return "neutral"
        
        # Determination by maximum evaluation
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def _get_from_cache(self, key: str) -> Optional[SentimentResult]:
        """Retrieval result from cache."""
        
        if not self._redis:
            return None
        
        try:
            cached_data = await self._redis.get(key)
            if cached_data:
                # Deserialization (simplified for example)
                import json
                data = json.loads(cached_data)
                return SentimentResult(**data)
        except Exception as e:
            self.logger.debug("Cache read failed", key=key, error=str(e))
        
        return None
    
    async def _save_to_cache(self, key: str, result: SentimentResult) -> None:
        """Saving result in cache."""
        
        if not self._redis:
            return
        
        try:
            # Serialization (simplified)
            import json
            data = {
                "text": result.text,
                "sentiment": result.sentiment,
                "confidence": result.confidence,
                "scores": result.scores,
                "crypto_symbols": result.crypto_symbols,
                "processing_time_ms": result.processing_time_ms,
                "model_version": result.model_version,
                "timestamp": result.timestamp.isoformat()
            }
            
            await self._redis.setex(key, self._cache_ttl, json.dumps(data))
            
        except Exception as e:
            self.logger.debug("Cache write failed", key=key, error=str(e))
    
    async def _batch_processor(self) -> None:
        """Background task for processing queue."""
        
        while True:
            try:
                # Collection batch from queue
                batch = []
                deadline = time.time() + self.processing_timeout
                
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        item = await asyncio.wait_for(
                            self._input_queue.get(),
                            timeout=deadline - time.time()
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    # Processing batch
                    texts = [item["text"] for item in batch]
                    results = await self.analyze_batch(texts)
                    
                    # Return results
                    for item, result in zip(batch, results):
                        if "callback" in item:
                            asyncio.create_task(item["callback"](result))
                
                # Small pause
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error("Batch processor error", error=str(e))
                await asyncio.sleep(1)
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Retrieval statistics performance."""
        
        uptime = time.time() - self._start_time
        throughput = self._processed_count / uptime if uptime > 0 else 0
        error_rate = self._error_count / max(self._processed_count, 1)
        
        return {
            "uptime_seconds": uptime,
            "messages_processed": self._processed_count,
            "error_count": self._error_count,
            "error_rate": error_rate,
            "throughput_msg_per_sec": throughput,
            "queue_size": self._input_queue.qsize(),
            "circuit_breaker_state": str(self._circuit_breaker.current_state),
            "metrics": self.metrics.get_metrics()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Validation state analyzer."""
        
        try:
            # Test analysis
            test_result = await self.analyze_sentiment(
                "Bitcoin is going to the moon! 🚀",
                platform="test"
            )
            
            perf_stats = await self.get_performance_stats()
            
            return {
                "status": "healthy" if test_result.sentiment else "unhealthy",
                "model_loaded": self.ensemble_model is not None,
                "redis_connected": self._redis is not None,
                "test_analysis": {
                    "sentiment": test_result.sentiment,
                    "confidence": test_result.confidence,
                    "processing_time_ms": test_result.processing_time_ms
                },
                "performance": perf_stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self) -> None:
        """Graceful shutdown analyzer."""
        
        self.logger.info("Shutting down real-time sentiment analyzer")
        
        # Closing Redis connections
        if self._redis:
            await self._redis.close()
        
        # Stopping ensemble model
        if self.ensemble_model:
            await self.ensemble_model.shutdown()