"""
REST API for system analysis sentiments

FastAPI-based REST API with enterprise patterns for
integration with trading systems and external applications.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog
import uvicorn

from ..utils.config import get_config
from ..analysis.realtime_analyzer import RealtimeSentimentAnalyzer, SentimentResult
from ..aggregation.sentiment_aggregator import SentimentAggregator
from ..trends.trending_topics import TrendingTopicsDetector
from ..monitoring.metrics_collector import get_metrics_collector
from .authentication import AuthManager

logger = structlog.get_logger(__name__)

# Pydantic model for API
class SentimentAnalysisRequest(BaseModel):
    """Request on analysis sentiment."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text for analysis")
    platform: Optional[str] = Field("api", description="Platform source")
    priority: Optional[int] = Field(1, ge=1, le=3, description="Priority processing")
    include_details: Optional[bool] = Field(False, description="Enable detailed information")

class BatchSentimentRequest(BaseModel):
    """Batch request on analysis sentiments."""
    texts: List[str] = Field(..., max_items=100, description="List texts for analysis")
    platform: Optional[str] = Field("api", description="Platform source")
    include_details: Optional[bool] = Field(False, description="Enable detailed information")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError('List texts not can be empty')
        for text in v:
            if not text or len(text.strip()) == 0:
                raise ValueError('Text not can be empty')
            if len(text) > 5000:
                raise ValueError('Text not can be longer 5000 symbols')
        return v

class SentimentResponse(BaseModel):
    """Response with result analysis."""
    sentiment: str = Field(..., description="Sentiment: positive, negative, neutral")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence model")
    scores: Dict[str, float] = Field(..., description="Detailed estimation")
    crypto_symbols: List[str] = Field(default_factory=list, description="Found crypto-symbols")
    processing_time_ms: float = Field(..., description="Time processing in milliseconds")
    model_version: str = Field(..., description="Version model")
    timestamp: datetime = Field(..., description="Time analysis")

class TrendingTopicsRequest(BaseModel):
    """Request trend the."""
    platforms: Optional[List[str]] = Field(None, description="List platforms")
    time_window_hours: Optional[int] = Field(24, ge=1, le=168, description="Temporary window in hours")
    limit: Optional[int] = Field(50, ge=1, le=200, description="Number results")
    crypto_only: Optional[bool] = Field(True, description="Only crypto-topics")

class AggregatedSentimentRequest(BaseModel):
    """Request aggregated sentiment."""
    symbol: Optional[str] = Field(None, description="Crypto-symbol for filtering")
    platforms: Optional[List[str]] = Field(None, description="List platforms")
    time_window_hours: Optional[int] = Field(24, ge=1, le=168, description="Temporary window")
    aggregation_method: Optional[str] = Field("weighted", description="Method aggregation")

class HealthResponse(BaseModel):
    """Response health check."""
    status: str = Field(..., description="State system")
    timestamp: datetime = Field(..., description="Time validation")
    components: Dict[str, Any] = Field(..., description="State components")
    metrics: Dict[str, Any] = Field(..., description="Main metrics")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup/shutdown."""
    
    # Startup
    logger.info("Starting Social Sentiment API")
    
    config = get_config()
    
    # Initialization components
    app.state.realtime_analyzer = RealtimeSentimentAnalyzer(config)
    await app.state.realtime_analyzer.initialize()
    
    app.state.sentiment_aggregator = SentimentAggregator(config)
    await app.state.sentiment_aggregator.initialize()
    
    app.state.trending_detector = TrendingTopicsDetector(config)
    await app.state.trending_detector.initialize()
    
    app.state.auth_manager = AuthManager(config)
    app.state.metrics = get_metrics_collector("api")
    
    logger.info("Social Sentiment API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Social Sentiment API")
    
    if hasattr(app.state, 'realtime_analyzer'):
        await app.state.realtime_analyzer.shutdown()
    
    if hasattr(app.state, 'sentiment_aggregator'):
        await app.state.sentiment_aggregator.shutdown()
    
    if hasattr(app.state, 'trending_detector'):
        await app.state.trending_detector.shutdown()
    
    logger.info("Social Sentiment API shutdown completed")

class SocialSentimentAPI:
    """
    Enterprise REST API for analysis sentiments
    
    Features:
    - FastAPI with async support
    - JWT authentication
    - Rate limiting
    - CORS support
    - Metrics and monitoring
    - Swagger/OpenAPI documentation
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logger.bind(component="rest_api")
        
        # Creation FastAPI application
        self.app = FastAPI(
            title="Social Sentiment Analyzer API",
            description="Enterprise API for analysis sentiments in social networks",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
            lifespan=lifespan
        )
        
        # Configuration middleware
        self._setup_middleware()
        
        # Registration routes
        self._setup_routes()
        
        # Security
        self.security = HTTPBearer(auto_error=False)
        
    def _setup_middleware(self):
        """Configuration middleware."""
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.security.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Gzip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request middleware for metrics
        @self.app.middleware("http")
        async def metrics_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Processing request
            response = await call_next(request)
            
            # Entry metrics
            processing_time = (time.time() - start_time) * 1000
            if hasattr(self.app.state, 'metrics'):
                self.app.state.metrics.record_api_request(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    duration_ms=processing_time
                )
            
            return response
    
    def _setup_routes(self):
        """Registration API routes."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Validation state system."""
            
            try:
                components = {}
                
                # Validation components
                if hasattr(self.app.state, 'realtime_analyzer'):
                    components['realtime_analyzer'] = await self.app.state.realtime_analyzer.health_check()
                
                if hasattr(self.app.state, 'sentiment_aggregator'):
                    components['sentiment_aggregator'] = await self.app.state.sentiment_aggregator.health_check()
                
                if hasattr(self.app.state, 'trending_detector'):
                    components['trending_detector'] = await self.app.state.trending_detector.health_check()
                
                # Collection metrics
                metrics = {}
                if hasattr(self.app.state, 'metrics'):
                    metrics = self.app.state.metrics.get_metrics()
                
                # Determination total state
                overall_status = "healthy"
                for component_status in components.values():
                    if component_status.get('status') != 'healthy':
                        overall_status = "degraded"
                        break
                
                return HealthResponse(
                    status=overall_status,
                    timestamp=datetime.now(),
                    components=components,
                    metrics=metrics
                )
                
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                raise HTTPException(status_code=500, detail="Health check failed")
        
        @self.app.post("/sentiment/analyze", response_model=SentimentResponse)
        async def analyze_sentiment(
            request: SentimentAnalysisRequest,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Analysis sentiment one text."""
            
            # Authentication
            await self._authenticate(credentials)
            
            try:
                # Analysis through realtime analyzer
                result = await self.app.state.realtime_analyzer.analyze_sentiment(
                    text=request.text,
                    platform=request.platform,
                    priority=request.priority
                )
                
                return SentimentResponse(
                    sentiment=result.sentiment,
                    confidence=result.confidence,
                    scores=result.scores,
                    crypto_symbols=result.crypto_symbols,
                    processing_time_ms=result.processing_time_ms,
                    model_version=result.model_version,
                    timestamp=result.timestamp
                )
                
            except Exception as e:
                logger.error("Sentiment analysis failed", text=request.text[:100], error=str(e))
                raise HTTPException(status_code=500, detail="Sentiment analysis failed")
        
        @self.app.post("/sentiment/analyze-batch", response_model=List[SentimentResponse])
        async def analyze_sentiment_batch(
            request: BatchSentimentRequest,
            background_tasks: BackgroundTasks,
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Batch analysis sentiments."""
            
            await self._authenticate(credentials)
            
            try:
                # Batch analysis
                results = await self.app.state.realtime_analyzer.analyze_batch(
                    texts=request.texts,
                    platform=request.platform
                )
                
                # Background task for additional processing
                if request.include_details:
                    background_tasks.add_task(
                        self._process_batch_details, 
                        results, 
                        request.platform
                    )
                
                return [
                    SentimentResponse(
                        sentiment=result.sentiment,
                        confidence=result.confidence,
                        scores=result.scores,
                        crypto_symbols=result.crypto_symbols,
                        processing_time_ms=result.processing_time_ms,
                        model_version=result.model_version,
                        timestamp=result.timestamp
                    )
                    for result in results
                ]
                
            except Exception as e:
                logger.error("Batch sentiment analysis failed", batch_size=len(request.texts), error=str(e))
                raise HTTPException(status_code=500, detail="Batch analysis failed")
        
        @self.app.get("/sentiment/aggregated")
        async def get_aggregated_sentiment(
            request: AggregatedSentimentRequest = Depends(),
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get aggregated sentiment."""
            
            await self._authenticate(credentials)
            
            try:
                result = await self.app.state.sentiment_aggregator.get_aggregated_sentiment(
                    symbol=request.symbol,
                    platforms=request.platforms,
                    time_window_hours=request.time_window_hours,
                    aggregation_method=request.aggregation_method
                )
                
                return result
                
            except Exception as e:
                logger.error("Aggregated sentiment failed", symbol=request.symbol, error=str(e))
                raise HTTPException(status_code=500, detail="Aggregated sentiment failed")
        
        @self.app.get("/trends/topics")
        async def get_trending_topics(
            request: TrendingTopicsRequest = Depends(),
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get trend topics."""
            
            await self._authenticate(credentials)
            
            try:
                topics = await self.app.state.trending_detector.get_trending_topics(
                    platforms=request.platforms,
                    time_window_hours=request.time_window_hours,
                    limit=request.limit,
                    crypto_only=request.crypto_only
                )
                
                return topics
                
            except Exception as e:
                logger.error("Trending topics failed", error=str(e))
                raise HTTPException(status_code=500, detail="Trending topics failed")
        
        @self.app.get("/metrics")
        async def get_metrics(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get metrics system."""
            
            await self._authenticate(credentials, admin_required=True)
            
            try:
                if hasattr(self.app.state, 'metrics'):
                    return self.app.state.metrics.get_metrics()
                else:
                    return {"error": "Metrics not available"}
                    
            except Exception as e:
                logger.error("Metrics retrieval failed", error=str(e))
                raise HTTPException(status_code=500, detail="Metrics retrieval failed")
        
        @self.app.get("/metrics/prometheus")
        async def get_prometheus_metrics(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Get metrics in format Prometheus."""
            
            await self._authenticate(credentials, admin_required=True)
            
            try:
                if hasattr(self.app.state, 'metrics'):
                    prometheus_data = self.app.state.metrics.get_prometheus_metrics()
                    return Response(
                        content=prometheus_data,
                        media_type="text/plain; version=0.0.4"
                    )
                else:
                    raise HTTPException(status_code=503, detail="Metrics not available")
                    
            except Exception as e:
                logger.error("Prometheus metrics failed", error=str(e))
                raise HTTPException(status_code=500, detail="Prometheus metrics failed")
        
        @self.app.get("/stream/sentiment")
        async def stream_sentiment_analysis(
            credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.security)
        ):
            """Streaming transmission analysis sentiments."""
            
            await self._authenticate(credentials)
            
            async def sentiment_stream():
                """Generator flow data."""
                
                try:
                    # Connection to flow real time
                    async for result in self.app.state.realtime_analyzer.stream_analyze():
                        yield f"data: {result.model_dump_json()}\n\n"
                        await asyncio.sleep(0.1)  # Small delay
                        
                except Exception as e:
                    logger.error("Stream analysis failed", error=str(e))
                    yield f"event: error\ndata: {{'error': '{str(e)}'}}\n\n"
            
            return StreamingResponse(
                sentiment_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                }
            )
    
    async def _authenticate(
        self,
        credentials: Optional[HTTPAuthorizationCredentials],
        admin_required: bool = False
    ) -> None:
        """Authentication user."""
        
        if not hasattr(self.app.state, 'auth_manager'):
            return  # Auth not configured
        
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        try:
            user_info = await self.app.state.auth_manager.verify_token(credentials.credentials)
            
            if admin_required and not user_info.get('is_admin', False):
                raise HTTPException(status_code=403, detail="Admin access required")
                
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid authentication")
    
    async def _process_batch_details(
        self,
        results: List[SentimentResult],
        platform: str
    ) -> None:
        """Background processing additional details."""
        
        try:
            # Saving results for analysis trends
            for result in results:
                await self.app.state.trending_detector.process_result(result, platform)
            
            logger.debug("Batch details processed", count=len(results), platform=platform)
            
        except Exception as e:
            logger.error("Batch details processing failed", error=str(e))
    
    def run(self, host: str = None, port: int = None) -> None:
        """Launch API server."""
        
        host = host or self.config.api_host
        port = port or self.config.api_port
        
        self.logger.info("Starting Social Sentiment API server", host=host, port=port)
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if not self.config.debug else "debug",
            reload=self.config.debug,
            workers=1 if self.config.debug else self.config.workers
        )

# Creation global instance API
app = SocialSentimentAPI().app

if __name__ == "__main__":
    api = SocialSentimentAPI()
    api.run()