"""
RESTful API for Elliott Wave Analyzer.

High-performance async API with comprehensive
wave analysis endpoints, real-time data, and crypto market integration.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import pandas as pd

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

from ..utils.logger import get_logger, trading_logger, performance_monitor
from ..utils.config import config
from ..patterns.impulse_wave import ImpulseWaveDetector, ImpulseWave
from ..patterns.corrective_wave import CorrectiveWaveDetector, CorrectiveWave
from ..analysis.wave_counter import WaveCounter, WaveCount
from ..fibonacci.fibonacci_retracement import FibonacciRetracementCalculator
from ..ml.cnn_wave_detector import CNNWaveDetector
from .authentication import AuthManager

logger = get_logger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Elliott Wave Analyzer API",
    description="Professional Elliott Wave analysis API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
auth_manager = AuthManager()
security = HTTPBearer()


# Pydantic models for API
class PriceData(BaseModel):
    """Price data input model."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Wave analysis request model."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(..., description="Timeframe (e.g., 1h, 4h, 1d)")
    price_data: List[PriceData] = Field(..., min_items=50, description="Historical price data")
    analysis_types: List[str] = Field(
        default=["impulse", "corrective", "fibonacci"],
        description="Types of analysis to perform"
    )
    confidence_threshold: float = Field(
        default=0.6, 
        ge=0.1, 
        le=1.0, 
        description="Minimum confidence threshold"
    )
    
    @validator('timeframe')
    def validate_timeframe(cls, v):
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]
        if v not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {valid_timeframes}")
        return v


class WaveDetectionResponse(BaseModel):
    """Wave detection response model."""
    wave_type: str
    direction: str
    confidence: float
    start_time: datetime
    end_time: datetime
    start_price: float
    end_price: float
    fibonacci_ratios: Dict[str, float]
    rules_validation: Dict[str, bool]
    projection_targets: Dict[str, float]


class FibonacciLevelResponse(BaseModel):
    """Fibonacci level response model."""
    ratio: float
    percentage: float
    price: float
    level_type: str
    support_resistance: str
    strength: float
    touches: int
    success_rate: float


class AnalysisResponse(BaseModel):
    """Complete analysis response model."""
    symbol: str
    timeframe: str
    analysis_timestamp: datetime
    processing_time_ms: float
    
    # Wave analysis results
    impulse_waves: List[WaveDetectionResponse] = []
    corrective_waves: List[WaveDetectionResponse] = []
    wave_count: Optional[Dict[str, Any]] = None
    
    # Fibonacci analysis
    fibonacci_levels: List[FibonacciLevelResponse] = []
    confluence_zones: List[Dict[str, Any]] = []
    
    # ML predictions
    ml_patterns: List[Dict[str, Any]] = []
    
    # Market analysis
    trend_direction: str
    volatility_score: float
    market_structure: Dict[str, Any]


class StreamSubscription(BaseModel):
    """Real-time stream subscription model."""
    symbols: List[str]
    timeframes: List[str]
    analysis_types: List[str] = ["impulse", "corrective", "fibonacci"]
    update_interval: int = Field(default=60, ge=5, le=300, description="Update interval in seconds")


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


class ElliottWaveAPI:
    """
    Elliott Wave Analysis API Server.
    
    
    - High-performance async processing
    - Comprehensive wave analysis
    - Real-time market integration
    - Professional-grade endpoints
    """
    
    def __init__(self):
        """Initialize API server with analysis components."""
        # Initialize analyzers
        self.impulse_detector = ImpulseWaveDetector()
        self.corrective_detector = CorrectiveWaveDetector()
        self.wave_counter = WaveCounter()
        self.fibonacci_calculator = FibonacciRetracementCalculator()
        self.cnn_detector = CNNWaveDetector()
        
        # Background task queue
        self.background_tasks = []
        
        # Active subscriptions for real-time updates
        self.active_subscriptions: Dict[str, StreamSubscription] = {}
        
    @performance_monitor
    async def analyze_waves(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Perform comprehensive Elliott Wave analysis.
        
        Multi-component analysis with performance tracking.
        """
        start_time = datetime.utcnow()
        
        # Convert price data to DataFrame
        price_df = self._convert_price_data(request.price_data)
        
        # Initialize response
        response = AnalysisResponse(
            symbol=request.symbol,
            timeframe=request.timeframe,
            analysis_timestamp=start_time,
            processing_time_ms=0.0,
            trend_direction="unknown",
            volatility_score=0.0,
            market_structure={}
        )
        
        try:
            # Impulse wave analysis
            if "impulse" in request.analysis_types:
                impulse_waves = await self.impulse_detector.detect_impulse_waves(
                    price_df, request.symbol, request.timeframe
                )
                response.impulse_waves = [
                    self._convert_impulse_wave(wave) for wave in impulse_waves
                    if wave.confidence >= request.confidence_threshold
                ]
                
            # Corrective wave analysis
            if "corrective" in request.analysis_types:
                corrective_waves = await self.corrective_detector.detect_corrective_waves(
                    price_df, request.symbol, request.timeframe
                )
                response.corrective_waves = [
                    self._convert_corrective_wave(wave) for wave in corrective_waves
                    if wave.confidence >= request.confidence_threshold
                ]
                
            # Wave counting
            if "count" in request.analysis_types:
                all_waves = []
                if response.impulse_waves:
                    # Convert back to wave objects for counting
                    pass  # Simplified for example
                    
                wave_counts = await self.wave_counter.count_waves(
                    price_df, request.symbol, request.timeframe
                )
                if wave_counts:
                    response.wave_count = self._convert_wave_count(wave_counts[0])
                    
            # Fibonacci analysis
            if "fibonacci" in request.analysis_types:
                fib_levels = await self._analyze_fibonacci_levels(price_df, request.symbol)
                response.fibonacci_levels = fib_levels
                
            # ML pattern detection
            if "ml" in request.analysis_types:
                ml_patterns = await self.cnn_detector.detect_patterns(
                    price_df, request.symbol, request.timeframe, request.confidence_threshold
                )
                response.ml_patterns = [
                    {
                        "pattern_type": pattern.pattern_type.value,
                        "confidence": pattern.confidence,
                        "probability_distribution": {k.value: v for k, v in pattern.probability_distribution.items()},
                        "bounding_box": pattern.bounding_box
                    } for pattern in ml_patterns
                ]
                
            # Market structure analysis
            response.trend_direction = await self._analyze_trend(price_df)
            response.volatility_score = await self._calculate_volatility(price_df)
            response.market_structure = await self._analyze_market_structure(price_df)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            response.processing_time_ms = processing_time
            
            # Log analysis
            trading_logger.log_wave_detection(
                symbol=request.symbol,
                timeframe=request.timeframe,
                wave_type="api_analysis",
                confidence=max([w.confidence for w in response.impulse_waves + response.corrective_waves], default=0),
                detected_count=len(response.impulse_waves) + len(response.corrective_waves),
                processing_time_ms=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Analysis error for {request.symbol}: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    def _convert_price_data(self, price_data: List[PriceData]) -> pd.DataFrame:
        """Convert API price data to DataFrame."""
        data = []
        for item in price_data:
            data.append({
                'timestamp': item.timestamp,
                'open': item.open,
                'high': item.high,
                'low': item.low,
                'close': item.close,
                'volume': item.volume or 0
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    def _convert_impulse_wave(self, wave: ImpulseWave) -> WaveDetectionResponse:
        """Convert impulse wave to API response."""
        return WaveDetectionResponse(
            wave_type="impulse",
            direction=wave.direction.value,
            confidence=wave.confidence,
            start_time=wave.wave_1[0].timestamp,
            end_time=wave.wave_5[1].timestamp,
            start_price=wave.wave_1[0].price,
            end_price=wave.wave_5[1].price,
            fibonacci_ratios=wave.fibonacci_ratios,
            rules_validation=wave.rules_validation,
            projection_targets=wave.get_projection_targets()
        )
        
    def _convert_corrective_wave(self, wave: CorrectiveWave) -> WaveDetectionResponse:
        """Convert corrective wave to API response."""
        return WaveDetectionResponse(
            wave_type=f"corrective_{wave.corrective_type.value}",
            direction=wave.direction.value,
            confidence=wave.confidence,
            start_time=wave.wave_a[0].timestamp,
            end_time=wave.wave_c[1].timestamp,
            start_price=wave.wave_a[0].price,
            end_price=wave.wave_c[1].price,
            fibonacci_ratios=wave.fibonacci_ratios,
            rules_validation=wave.rules_validation,
            projection_targets=wave.get_projection_targets()
        )
        
    def _convert_wave_count(self, wave_count: WaveCount) -> Dict[str, Any]:
        """Convert wave count to API response."""
        return {
            "count_type": wave_count.count_type.value,
            "confidence": wave_count.confidence,
            "confidence_level": wave_count.confidence_level.value,
            "wave_sequence": wave_count.wave_sequence,
            "completion_percentage": wave_count.completion_percentage,
            "next_expected_wave": wave_count.get_next_expected_wave(),
            "fibonacci_confluence": wave_count.fibonacci_confluence,
            "rules_compliance": wave_count.rules_compliance,
            "pattern_clarity": wave_count.pattern_clarity
        }
        
    async def _analyze_fibonacci_levels(self, price_df: pd.DataFrame, symbol: str) -> List[FibonacciLevelResponse]:
        """Analyze Fibonacci retracement levels."""
        try:
            # Find significant swing points (simplified)
            highs = price_df['high']
            lows = price_df['low']
            
            swing_high_idx = highs.idxmax()
            swing_low_idx = lows.idxmin()
            
            from ..patterns.impulse_wave import WavePoint
            
            swing_high = WavePoint(
                index=0,  # Simplified
                price=highs[swing_high_idx],
                timestamp=swing_high_idx
            )
            
            swing_low = WavePoint(
                index=0,  # Simplified
                price=lows[swing_low_idx],
                timestamp=swing_low_idx
            )
            
            # Calculate retracement
            retracement = await self.fibonacci_calculator.calculate_retracement(
                swing_high, swing_low, symbol, "1h", price_df
            )
            
            # Convert to API response
            return [
                FibonacciLevelResponse(
                    ratio=level.ratio,
                    percentage=level.percentage,
                    price=level.price,
                    level_type=level.level_type.value,
                    support_resistance=level.support_resistance.value,
                    strength=level.strength,
                    touches=level.touches,
                    success_rate=level.success_rate
                ) for level in retracement.levels
            ]
            
        except Exception as e:
            logger.error(f"Fibonacci analysis error: {e}")
            return []
            
    async def _analyze_trend(self, price_df: pd.DataFrame) -> str:
        """Analyze overall trend direction."""
        if len(price_df) < 20:
            return "unknown"
            
        closes = price_df['close'].values
        ma20 = closes[-20:].mean()
        ma50 = closes[-50:].mean() if len(closes) >= 50 else ma20
        
        current_price = closes[-1]
        
        if current_price > ma20 > ma50:
            return "bullish"
        elif current_price < ma20 < ma50:
            return "bearish"
        else:
            return "sideways"
            
    async def _calculate_volatility(self, price_df: pd.DataFrame) -> float:
        """Calculate volatility score."""
        if len(price_df) < 20:
            return 0.0
            
        returns = price_df['close'].pct_change().dropna()
        volatility = returns.std() * 100  # Percentage
        
        # Normalize to 0-1 scale
        return min(volatility / 10.0, 1.0)  # 10% daily volatility = 1.0
        
    async def _analyze_market_structure(self, price_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure."""
        if len(price_df) < 50:
            return {}
            
        highs = price_df['high']
        lows = price_df['low']
        
        # Find recent highs and lows
        recent_high = highs[-20:].max()
        recent_low = lows[-20:].max()
        
        return {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "support_level": lows[-50:].min(),
            "resistance_level": highs[-50:].max(),
            "range_percentage": ((recent_high - recent_low) / recent_low) * 100
        }


# Initialize API instance
wave_api = ElliottWaveAPI()


# API Routes
@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_waves(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user = Depends(get_current_user)
):
    """
    Perform comprehensive Elliott Wave analysis.
    
    Analyzes price data for Elliott Wave patterns including:
    - Impulse wave detection
    - Corrective wave patterns
    - Fibonacci retracements
    - ML-based pattern recognition
    - Market structure analysis
    """
    return await wave_api.analyze_waves(request)


@app.get("/api/v1/symbols/{symbol}/analysis")
async def get_symbol_analysis(
    symbol: str = Path(..., description="Trading symbol"),
    timeframe: str = Query("4h", description="Analysis timeframe"),
    days: int = Query(30, ge=1, le=365, description="Days of historical data"),
    user = Depends(get_current_user)
):
    """Get analysis for specific symbol with auto data fetching."""
    # This would integrate with market data providers
    raise HTTPException(status_code=501, detail="Auto data fetching not implemented")


@app.get("/api/v1/fibonacci/{symbol}")
async def get_fibonacci_analysis(
    symbol: str = Path(..., description="Trading symbol"),
    timeframe: str = Query("4h", description="Timeframe"),
    user = Depends(get_current_user)
):
    """Get Fibonacci analysis for symbol."""
    raise HTTPException(status_code=501, detail="Standalone Fibonacci analysis not implemented")


@app.get("/api/v1/patterns/ml/{symbol}")
async def get_ml_patterns(
    symbol: str = Path(..., description="Trading symbol"),
    confidence: float = Query(0.7, ge=0.1, le=1.0, description="Minimum confidence"),
    user = Depends(get_current_user)
):
    """Get ML-detected patterns for symbol."""
    raise HTTPException(status_code=501, detail="Standalone ML pattern detection not implemented")


@app.post("/api/v1/stream/subscribe")
async def subscribe_to_stream(
    subscription: StreamSubscription,
    user = Depends(get_current_user)
):
    """Subscribe to real-time wave analysis updates."""
    subscription_id = f"{user['user_id']}_{datetime.utcnow().timestamp()}"
    wave_api.active_subscriptions[subscription_id] = subscription
    
    return {
        "subscription_id": subscription_id,
        "status": "active",
        "symbols": subscription.symbols,
        "timeframes": subscription.timeframes,
        "update_interval": subscription.update_interval
    }


@app.delete("/api/v1/stream/{subscription_id}")
async def unsubscribe_from_stream(
    subscription_id: str = Path(..., description="Subscription ID"),
    user = Depends(get_current_user)
):
    """Unsubscribe from real-time updates."""
    if subscription_id in wave_api.active_subscriptions:
        del wave_api.active_subscriptions[subscription_id]
        return {"status": "unsubscribed", "subscription_id": subscription_id}
    else:
        raise HTTPException(status_code=404, detail="Subscription not found")


@app.get("/api/v1/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "components": {
            "impulse_detector": "operational",
            "corrective_detector": "operational",
            "fibonacci_calculator": "operational",
            "cnn_detector": "operational",
            "wave_counter": "operational"
        }
    }


@app.get("/api/v1/stats")
async def get_api_stats(user = Depends(get_current_user)):
    """Get API usage statistics."""
    return {
        "detectors": {
            "impulse": wave_api.impulse_detector.get_detection_statistics(),
            "corrective": wave_api.corrective_detector.detection_stats,
            "cnn": wave_api.cnn_detector.get_model_info()
        },
        "active_subscriptions": len(wave_api.active_subscriptions),
        "system_info": {
            "python_version": "3.11+",
            "fastapi_version": "0.101+",
            "torch_version": "2.0+"
        }
    }


# WebSocket endpoint for real-time updates
@app.websocket("/ws/waves")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time wave analysis updates."""
    # WebSocket implementation would go here
    # Placeholder for real-time functionality
    await websocket.accept()
    await websocket.send_text("WebSocket connection established")
    await websocket.close()


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "rest_api:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=config.debug
    )


# Export main classes
__all__ = [
    'app',
    'ElliottWaveAPI',
    'AnalysisRequest',
    'AnalysisResponse',
    'WaveDetectionResponse',
    'FibonacciLevelResponse'
]