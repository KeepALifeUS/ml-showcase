"""
FastAPI Sentiment Analysis API for ML-Framework ML Sentiment Engine

Enterprise-grade API with , rate limiting and async support.
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import uuid

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import aioredis
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from ..models.ensemble_sentiment import EnsembleSentimentModel, create_ensemble_model
from ..sources.twitter_source import TwitterSentimentSource, create_twitter_source
from ..sources.reddit_source import RedditSentimentSource, create_reddit_source
from ..sources.news_source import NewsSentimentSource, create_news_source
from ..utils.logger import get_logger, set_request_context, clear_request_context
from ..utils.config import get_config
from ..utils.validators import SentimentScore, CryptoSymbol, TextContent

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('sentiment_api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('sentiment_api_request_duration_seconds', 'Request duration')
PREDICTION_COUNT = Counter('sentiment_predictions_total', 'Total predictions', ['model', 'source'])


# Pydantic models for API
class SentimentRequest(BaseModel):
 """Sentiment analysis request"""
 text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")
 source: str = Field(default="api", description="Data source")
 model: str = Field(default="ensemble", description="Model for analysis")

 @validator("text")
 def validate_text(cls, v):
 if not v or not v.strip:
 raise ValueError("Text cannot be empty")
 return v.strip


class BatchSentimentRequest(BaseModel):
 """Batch request on analysis sentiment"""
 texts: List[str] = Field(..., max_items=100, description="List of texts")
 source: str = Field(default="api", description="Data source")
 model: str = Field(default="ensemble", description="Model for analysis")

 @validator("texts")
 def validate_texts(cls, v):
 if not v:
 raise ValueError("Texts list cannot be empty")
 return [text.strip for text in v if text and text.strip]


class SentimentResponse(BaseModel):
 """Response analysis sentiment"""
 sentiment: float = Field(..., description="Sentiment score (-1 to 1)")
 confidence: float = Field(..., description="Confidence score (0 to 1)")
 label: str = Field(..., description="Sentiment label")
 model: str = Field(..., description="Using model")
 processing_time_ms: float = Field(..., description="Time handling in with")
 request_id: str = Field(..., description="ID request")


class DataSourceRequest(BaseModel):
 """Request data from source"""
 source: str = Field(..., description="Data source")
 symbol: Optional[str] = Field(None, description="Cryptocurrency symbol")
 limit: int = Field(default=100, ge=1, le=1000, description="Result limit")
 hours_back: int = Field(default=24, ge=1, le=168, description="Time period in hours")


class HealthResponse(BaseModel):
 """Response health check"""
 status: str
 timestamp: datetime
 version: str
 models_loaded: int
 sources_available: int
 uptime_seconds: float


class SentimentAPI:
 """
 Enterprise-grade FastAPI application for sentiment analysis
 """

 def __init__(self):
 """Initialize the API"""
 self.config = get_config
 self.app = FastAPI(
 title="ML-Framework Sentiment Analysis API",
 description="Enterprise-grade sentiment analysis for crypto trading",
 version="1.0.0",
 docs_url="/docs" if self.config.is_development else None,
 redoc_url="/redoc" if self.config.is_development else None
 )

 # Models and sources
 self.ensemble_model: Optional[EnsembleSentimentModel] = None
 self.data_sources = {}

 # Redis for rate limiting
 self.redis: Optional[aioredis.Redis] = None

 # Record metrics
 self.start_time = time.time

 # Setup API
 self._setup_middleware
 self._setup_routes

 async def initialize(self):
 """Initialize the API components"""
 try:
 # Initialize the ensemble model
 logger.info("Initializing ensemble sentiment model...")
 self.ensemble_model = await create_ensemble_model

 # Initialize data sources
 logger.info("Initializing data sources...")
 await self._initialize_data_sources

 # Initialize Redis
 if self.config.redis.host:
 try:
 self.redis = aioredis.from_url(
 self.config.redis.url,
 encoding="utf-8",
 decode_responses=True
 )
 await self.redis.ping
 logger.info("Redis connection established")
 except Exception as e:
 logger.warning("Redis connection failed", error=e)

 logger.info("Sentiment API initialized successfully")

 except Exception as e:
 logger.error("Failed to initialize Sentiment API", error=e)
 raise

 async def _initialize_data_sources(self):
 """Initialize data sources"""
 try:
 # Twitter source
 if self.config.social.twitter_bearer_token:
 self.data_sources["twitter"] = await create_twitter_source

 # Reddit source
 if self.config.social.reddit_client_id:
 self.data_sources["reddit"] = await create_reddit_source

 # News source
 self.data_sources["news"] = await create_news_source

 logger.info(f"Initialized {len(self.data_sources)} data sources")

 except Exception as e:
 logger.warning("Some data sources failed to initialize", error=e)

 def _setup_middleware(self):
 """Setup middleware"""
 # CORS
 self.app.add_middleware(
 CORSMiddleware,
 allow_origins=self.config.api.cors_origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
 )

 # Compression
 self.app.add_middleware(GZipMiddleware, minimum_size=1000)

 # Request logging middleware
 @self.app.middleware("http")
 async def log_requests(request: Request, call_next):
 start_time = time.time
 request_id = str(uuid.uuid4)

 # Set request context
 set_request_context(request_id)

 try:
 response = await call_next(request)

 # Record metrics
 duration = time.time - start_time
 REQUEST_DURATION.observe(duration)
 REQUEST_COUNT.labels(
 method=request.method,
 endpoint=request.url.path,
 status=response.status_code
 ).inc

 # Logging
 logger.info(
 "API request completed",
 method=request.method,
 url=str(request.url),
 status_code=response.status_code,
 duration_ms=round(duration * 1000, 2),
 request_id=request_id
 )

 return response

 except Exception as e:
 # Error metrics
 REQUEST_COUNT.labels(
 method=request.method,
 endpoint=request.url.path,
 status=500
 ).inc

 logger.error("API request failed", error=e, request_id=request_id)
 raise
 finally:
 clear_request_context

 def _setup_routes(self):
 """Setup API routes"""

 @self.app.on_event("startup")
 async def startup_event:
 await self.initialize

 @self.app.on_event("shutdown")
 async def shutdown_event:
 # Cleanup data sources
 for source in self.data_sources.values:
 if hasattr(source, 'cleanup'):
 await source.cleanup

 # Close Redis
 if self.redis:
 await self.redis.close

 @self.app.get("/health", response_model=HealthResponse)
 async def health_check:
 """Health check endpoint"""
 uptime = time.time - self.start_time

 return HealthResponse(
 status="healthy",
 timestamp=datetime.utcnow,
 version="1.0.0",
 models_loaded=1 if self.ensemble_model else 0,
 sources_available=len(self.data_sources),
 uptime_seconds=uptime
 )

 @self.app.post("/sentiment", response_model=SentimentResponse)
 async def analyze_sentiment(request: SentimentRequest):
 """Analysis sentiment for a single text"""
 if not self.ensemble_model:
 raise HTTPException(status_code=503, detail="Sentiment model not ready")

 try:
 start_time = time.time
 request_id = str(uuid.uuid4)

 # Prediction
 result = await self.ensemble_model.predict(
 text=request.text,
 source=request.source
 )

 processing_time = (time.time - start_time) * 1000

 # Record metrics
 PREDICTION_COUNT.labels(
 model=request.model,
 source=request.source
 ).inc

 # Sentiment label
 label = "positive" if result.value > 0.1 else "negative" if result.value < -0.1 else "neutral"

 logger.info(
 "Sentiment analysis completed",
 sentiment=result.value,
 confidence=result.confidence,
 label=label,
 processing_time_ms=processing_time,
 request_id=request_id
 )

 return SentimentResponse(
 sentiment=result.value,
 confidence=result.confidence,
 label=label,
 model=result.model_name,
 processing_time_ms=processing_time,
 request_id=request_id
 )

 except Exception as e:
 logger.error("Sentiment analysis failed", error=e)
 raise HTTPException(status_code=500, detail=str(e))

 @self.app.post("/sentiment/batch", response_model=List[SentimentResponse])
 async def analyze_sentiment_batch(request: BatchSentimentRequest):
 """Batch analysis sentiment"""
 if not self.ensemble_model:
 raise HTTPException(status_code=503, detail="Sentiment model not ready")

 try:
 start_time = time.time

 # Batch prediction
 results = await self.ensemble_model.predict_batch(
 texts=request.texts,
 source=request.source
 )

 processing_time = (time.time - start_time) * 1000

 # Record metrics
 PREDICTION_COUNT.labels(
 model=request.model,
 source=request.source
 ).inc(len(request.texts))

 # Format responses
 responses = []
 for i, result in enumerate(results):
 label = "positive" if result.value > 0.1 else "negative" if result.value < -0.1 else "neutral"

 responses.append(SentimentResponse(
 sentiment=result.value,
 confidence=result.confidence,
 label=label,
 model=result.model_name,
 processing_time_ms=processing_time / len(request.texts),
 request_id=f"batch_{i}_{uuid.uuid4}"
 ))

 logger.info(
 "Batch sentiment analysis completed",
 batch_size=len(request.texts),
 processing_time_ms=processing_time
 )

 return responses

 except Exception as e:
 logger.error("Batch sentiment analysis failed", error=e)
 raise HTTPException(status_code=500, detail=str(e))

 @self.app.get("/data/{source}")
 async def fetch_data_from_source(
 source: str,
 symbol: Optional[str] = None,
 limit: int = 100,
 hours_back: int = 24
 ):
 """Get data from a source"""
 if source not in self.data_sources:
 raise HTTPException(status_code=404, detail=f"Data source '{source}' not found")

 try:
 data_source = self.data_sources[source]

 if source == "twitter" and symbol:
 data = await data_source.search_tweets(
 symbols=[symbol],
 limit=limit,
 hours_back=hours_back
 )
 elif source == "reddit":
 data = await data_source.fetch_all_crypto_content(
 limit_per_subreddit=limit // 10,
 include_comments=False
 )
 elif source == "news":
 data = await data_source.fetch_latest_news(
 limit=limit,
 hours_back=hours_back
 )
 else:
 raise HTTPException(status_code=400, detail="Invalid source or missing parameters")

 logger.info(
 "Data fetched from source",
 source=source,
 data_count=len(data),
 symbol=symbol
 )

 return {"source": source, "data_count": len(data), "data": data}

 except Exception as e:
 logger.error(f"Failed to fetch data from {source}", error=e)
 raise HTTPException(status_code=500, detail=str(e))

 @self.app.get("/stats")
 async def get_api_stats:
 """Get API statistics"""
 try:
 model_stats = self.ensemble_model.get_stats if self.ensemble_model else {}
 source_stats = {
 name: source.get_stats
 for name, source in self.data_sources.items
 }

 return {
 "api": {
 "uptime_seconds": time.time - self.start_time,
 "models_loaded": 1 if self.ensemble_model else 0,
 "sources_available": len(self.data_sources)
 },
 "model": model_stats,
 "sources": source_stats
 }

 except Exception as e:
 logger.error("Failed to get API stats", error=e)
 raise HTTPException(status_code=500, detail=str(e))

 @self.app.get("/metrics")
 async def get_prometheus_metrics:
 """Prometheus metrics endpoint"""
 return Response(generate_latest, media_type="text/plain")


# Global app instance
app = SentimentAPI.app