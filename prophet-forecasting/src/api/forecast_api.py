"""
FastAPI REST API for Prophet forecasting system.

Enterprise-grade REST API with enterprise patterns for production deployment,
including async endpoints, WebSocket support, monitoring, and comprehensive error handling.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from ..models.prophet_model import ProphetForecaster, ForecastResult
from ..models.advanced_prophet import AdvancedProphetModel, AdvancedForecastResult
from ..preprocessing.data_processor import CryptoDataProcessor, ProcessedData
from ..validation.forecast_validator import ForecastValidator, ValidationStrategy, ValidationConfig
from ..config.prophet_config import get_config, ProphetConfig
from ..utils.logger import get_logger, get_request_logger
from ..utils.exceptions import (
    ProphetForecastingException,
    create_error_response,
    log_exception
)
from ..utils.helpers import validate_symbol, validate_timeframe

logger = get_logger(__name__)


# === Pydantic Models for API ===

class ForecastRequest(BaseModel):
    """Request on forecasting"""
    symbol: str = Field(..., description="Cryptocurrency symbol (e.g., BTC)")
    timeframe: str = Field(default="1h", description="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)")
    periods: Optional[int] = Field(default=None, ge=1, le=365, description="Number of periods to forecast")
    data: Optional[List[Dict]] = Field(default=None, description="OHLCV data for training")
    target_column: str = Field(default="close", description="Target column for prediction")
    include_features: bool = Field(default=True, description="Include additional features")
    model_type: str = Field(default="basic", description="Model type: 'basic' or 'advanced'")
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        if not validate_symbol(v, raise_error=False):
            raise ValueError(f"Invalid cryptocurrency symbol: {v}")
        return v.upper()
    
    @validator('timeframe')
    def validate_timeframe_format(cls, v):
        if not validate_timeframe(v, raise_error=False):
            raise ValueError(f"Invalid timeframe: {v}")
        return v.lower()


class TrainingRequest(BaseModel):
    """Request on training model"""
    symbol: str = Field(..., description="Cryptocurrency symbol")
    timeframe: str = Field(default="1h", description="Timeframe")
    data: List[Dict] = Field(..., description="Training data (OHLCV format)")
    target_column: str = Field(default="close", description="Target column")
    model_type: str = Field(default="basic", description="Model type")
    auto_optimize: bool = Field(default=False, description="Auto-optimize hyperparameters")
    validation_enabled: bool = Field(default=True, description="Enable validation")
    
    @validator('symbol')
    def validate_symbol_format(cls, v):
        return validate_symbol(v) and v.upper()
    
    @validator('timeframe') 
    def validate_timeframe_format(cls, v):
        return validate_timeframe(v) and v.lower()


class ValidationRequest(BaseModel):
    """Request on validation model"""
    symbol: str = Field(..., description="Cryptocurrency symbol")
    timeframe: str = Field(default="1h", description="Timeframe") 
    data: List[Dict] = Field(..., description="Validation data")
    strategy: ValidationStrategy = Field(default=ValidationStrategy.TIME_SERIES_SPLIT)
    n_splits: int = Field(default=5, ge=2, le=10)
    test_size_ratio: float = Field(default=0.2, ge=0.1, le=0.5)


class HealthResponse(BaseModel):
    """Response health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Current timestamp")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    models_loaded: int = Field(..., description="Number of loaded models")


class ModelInfoResponse(BaseModel):
    """Information about model"""
    symbol: str
    timeframe: str
    model_type: str
    is_trained: bool
    training_timestamp: Optional[datetime]
    training_samples: Optional[int]
    last_prediction: Optional[datetime]
    model_metrics: Dict[str, Any]


# === Global variables ===

# Cache models
MODEL_CACHE: Dict[str, Union[ProphetForecaster, AdvancedProphetModel]] = {}

# WebSocket connections
WEBSOCKET_CONNECTIONS: Dict[str, WebSocket] = {}

# Statistics API
API_STATS = {
    "requests_total": 0,
    "errors_total": 0,
    "models_trained": 0,
    "predictions_made": 0,
    "start_time": datetime.now()
}


# === Helper function ===

def get_model_key(symbol: str, timeframe: str) -> str:
    """Generation key model for cache"""
    return f"{symbol.upper()}_{timeframe.lower()}"


def get_or_create_model(
    symbol: str, 
    timeframe: str, 
    model_type: str = "basic"
) -> Union[ProphetForecaster, AdvancedProphetModel]:
    """Retrieval or creation model"""
    model_key = get_model_key(symbol, timeframe)
    
    if model_key not in MODEL_CACHE:
        if model_type == "advanced":
            MODEL_CACHE[model_key] = AdvancedProphetModel(symbol=symbol, timeframe=timeframe)
        else:
            MODEL_CACHE[model_key] = ProphetForecaster(symbol=symbol, timeframe=timeframe)
        
        logger.info(f"Created new {model_type} model: {model_key}")
    
    return MODEL_CACHE[model_key]


async def track_request():
    """Tracking requests"""
    API_STATS["requests_total"] += 1


async def handle_api_error(request_id: str, error: Exception) -> JSONResponse:
    """Processing errors API"""
    API_STATS["errors_total"] += 1
    
    if isinstance(error, ProphetForecastingException):
        error_response = create_error_response(error)
        log_exception(logger, error, {"request_id": request_id})
        return JSONResponse(
            status_code=400,
            content=error_response
        )
    else:
        logger.error(f"Unexpected API error: {error}", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": "InternalServerError",
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )


# === Lifespan Management ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Management life loop application"""
    # Startup
    logger.info("Starting Prophet Forecasting API")
    config = get_config()
    
    # Initialization components
    logger.info(f"Service configuration loaded: {config.service_name} v{config.version}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Prophet Forecasting API")
    
    # Cleanup resources
    MODEL_CACHE.clear()
    WEBSOCKET_CONNECTIONS.clear()
    
    logger.info("Cleanup completed")


# === Creation FastAPI application ===

def create_forecast_app(config: Optional[ProphetConfig] = None) -> FastAPI:
    """
    Creation FastAPI application for forecasting
    
    Args:
        config: Configuration application
        
    Returns:
        FastAPI application instance
    """
    if config is None:
        config = get_config()
    
    app = FastAPI(
        title="Prophet Forecasting API",
        description="Enterprise-grade cryptocurrency price forecasting using Facebook Prophet",
        version="5.0.0",
        docs_url="/docs" if config.api.debug else None,
        redoc_url="/redoc" if config.api.debug else None,
        lifespan=lifespan
    )
    
    # === Middleware ===
    
    # CORS
    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )
    
    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # === Main endpoints ===
    
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Root page"""
        return """
        <html>
            <head>
                <title>Prophet Forecasting API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                    .section { margin: 20px 0; }
                    .endpoint { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
                    .method { font-weight: bold; color: #27ae60; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🔮 Prophet Forecasting API</h1>
                    <p>Enterprise-grade cryptocurrency price forecasting using Facebook Prophet</p>
                </div>
                
                <div class="section">
                    <h2>Available Endpoints</h2>
                    <div class="endpoint">
                        <span class="method">GET</span> /health - Service health check
                    </div>
                    <div class="endpoint">
                        <span class="method">POST</span> /forecast - Create price forecast
                    </div>
                    <div class="endpoint">
                        <span class="method">POST</span> /train - Train Prophet model
                    </div>
                    <div class="endpoint">
                        <span class="method">POST</span> /validate - Validate model performance
                    </div>
                    <div class="endpoint">
                        <span class="method">GET</span> /models - List available models
                    </div>
                    <div class="endpoint">
                        <span class="method">GET</span> /stats - API statistics
                    </div>
                </div>
                
                <div class="section">
                    <h2>Documentation</h2>
                    <p><a href="/docs">Interactive API Documentation (Swagger UI)</a></p>
                    <p><a href="/redoc">Alternative Documentation (ReDoc)</a></p>
                </div>
            </body>
        </html>
        """
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        uptime = (datetime.now() - API_STATS["start_time"]).total_seconds()
        
        return HealthResponse(
            status="healthy",
            version="5.0.0",
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            models_loaded=len(MODEL_CACHE)
        )
    
    @app.post("/forecast")
    async def create_forecast(
        request: ForecastRequest,
        background_tasks: BackgroundTasks,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """
        Creation forecast price cryptocurrency
        
        - **symbol**: Symbol cryptocurrency (BTC, ETH, etc.)
        - **timeframe**: Timeframe data (1m, 5m, 15m, 30m, 1h, 4h, 1d)
        - **periods**: Number periods for forecast
        - **data**: OHLCV data for training (optionally)
        - **target_column**: Target column (open, high, low, close)
        - **model_type**: Type model (basic, advanced)
        """
        request_logger = get_request_logger(request_id, endpoint="/forecast")
        background_tasks.add_task(track_request)
        
        try:
            request_logger.info("Processing forecast request", extra={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "periods": request.periods,
                "model_type": request.model_type
            })
            
            # Retrieval or creation model
            model = get_or_create_model(
                request.symbol, 
                request.timeframe, 
                request.model_type
            )
            
            # Training model if provided data
            if request.data:
                request_logger.info("Training model with provided data")
                df = pd.DataFrame(request.data)
                
                if isinstance(model, AdvancedProphetModel):
                    await model.train_async(df)
                else:
                    training_result = model.train(df)
                    request_logger.info("Model training completed", extra=training_result)
            
            # Validation trainedness model
            if not hasattr(model, 'is_trained') or not model.is_trained:
                raise HTTPException(
                    status_code=400,
                    detail="Model is not trained. Please provide training data or train the model first."
                )
            
            # Creation forecast
            if isinstance(model, AdvancedProphetModel):
                forecast_result = await model.predict_async(
                    periods=request.periods,
                    include_history=False,
                    uncertainty_analysis=True
                )
            else:
                forecast_result = await model.predict_async(
                    periods=request.periods,
                    include_history=False
                )
            
            API_STATS["predictions_made"] += 1
            
            # Sending notifications through WebSocket
            background_tasks.add_task(
                broadcast_forecast_update, 
                request.symbol, 
                forecast_result
            )
            
            request_logger.info("Forecast completed successfully", extra={
                "forecast_points": len(forecast_result.forecast_df),
                "forecast_period": f"{forecast_result.forecast_df['ds'].min()} to {forecast_result.forecast_df['ds'].max()}"
            })
            
            return {
                "success": True,
                "request_id": request_id,
                "result": forecast_result.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return await handle_api_error(request_id, e)
    
    @app.post("/train")
    async def train_model(
        request: TrainingRequest,
        background_tasks: BackgroundTasks,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """
        Training Prophet model
        
        - **symbol**: Symbol cryptocurrency
        - **timeframe**: Timeframe data
        - **data**: Training data in format OHLCV
        - **target_column**: Target column for forecasting
        - **model_type**: Type model (basic, advanced)
        - **auto_optimize**: Automatic optimization hyperparameters
        - **validation_enabled**: Enable validation model
        """
        request_logger = get_request_logger(request_id, endpoint="/train")
        background_tasks.add_task(track_request)
        
        try:
            request_logger.info("Processing training request", extra={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "data_size": len(request.data),
                "model_type": request.model_type,
                "auto_optimize": request.auto_optimize
            })
            
            # Creation model
            model = get_or_create_model(
                request.symbol,
                request.timeframe, 
                request.model_type
            )
            
            # Preparation data
            df = pd.DataFrame(request.data)
            
            # Training model
            if isinstance(model, AdvancedProphetModel):
                training_result = await model.train_async(
                    data=df,
                    auto_optimize=request.auto_optimize,
                    feature_selection=True,
                    ensemble=False  # While disabled for API
                )
            else:
                training_result = model.train(
                    data=df,
                    validate=request.validation_enabled
                )
            
            API_STATS["models_trained"] += 1
            
            # Validation model (if enabled)
            validation_result = None
            if request.validation_enabled:
                request_logger.info("Running model validation")
                validator = ForecastValidator(request.symbol, request.timeframe)
                validation_result = await validator.backtest_model_async(
                    model=model,
                    data=df,
                    validation_config=ValidationConfig(
                        strategy=ValidationStrategy.TIME_SERIES_SPLIT,
                        n_splits=3,
                        test_size_ratio=0.2
                    )
                )
            
            request_logger.info("Model training completed successfully", extra={
                "training_time": training_result.get("training_time_seconds", 0),
                "training_samples": training_result.get("training_samples", 0)
            })
            
            response_data = {
                "success": True,
                "request_id": request_id,
                "training_result": training_result,
                "model_info": model.get_model_info(),
                "timestamp": datetime.now().isoformat()
            }
            
            if validation_result:
                response_data["validation_result"] = validation_result.to_dict()
            
            return response_data
            
        except Exception as e:
            return await handle_api_error(request_id, e)
    
    @app.post("/validate")
    async def validate_model(
        request: ValidationRequest,
        background_tasks: BackgroundTasks,
        request_id: str = Depends(lambda: str(uuid.uuid4()))
    ):
        """
        Validation model Prophet
        
        - **symbol**: Symbol cryptocurrency
        - **timeframe**: Timeframe data
        - **data**: Data for validation
        - **strategy**: Strategy validation
        - **n_splits**: Number splits for validation
        - **test_size_ratio**: Share test data
        """
        request_logger = get_request_logger(request_id, endpoint="/validate")
        background_tasks.add_task(track_request)
        
        try:
            request_logger.info("Processing validation request", extra={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "strategy": request.strategy.value,
                "n_splits": request.n_splits
            })
            
            # Retrieval model
            model_key = get_model_key(request.symbol, request.timeframe)
            if model_key not in MODEL_CACHE:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model not found for {request.symbol} {request.timeframe}. Please train the model first."
                )
            
            model = MODEL_CACHE[model_key]
            
            # Creation validator
            validator = ForecastValidator(request.symbol, request.timeframe)
            
            # Configuration validation
            validation_config = ValidationConfig(
                strategy=request.strategy,
                n_splits=request.n_splits,
                test_size_ratio=request.test_size_ratio
            )
            
            # Execution validation
            df = pd.DataFrame(request.data)
            validation_result = await validator.backtest_model_async(
                model=model,
                data=df,
                validation_config=validation_config
            )
            
            # Creation report
            report = validator.create_validation_report(
                validation_result,
                include_plots=False  # For API without charts
            )
            
            request_logger.info("Validation completed successfully", extra={
                "splits_count": validation_result.splits_count,
                "test_samples": validation_result.total_test_samples,
                "overall_mape": validation_result.overall_metrics.get('mape', {}).get('value', 0)
            })
            
            return {
                "success": True,
                "request_id": request_id,
                "validation_result": validation_result.to_dict(),
                "validation_report": report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return await handle_api_error(request_id, e)
    
    @app.get("/models")
    async def list_models():
        """List loaded models"""
        models = []
        
        for model_key, model in MODEL_CACHE.items():
            model_info = model.get_model_info()
            models.append({
                "key": model_key,
                "symbol": model_info["symbol"],
                "timeframe": model_info["timeframe"],
                "model_type": "advanced" if isinstance(model, AdvancedProphetModel) else "basic",
                "is_trained": model_info["is_trained"],
                "last_training_time": model_info.get("last_training_time"),
                "training_samples": model_info.get("training_data_size", 0)
            })
        
        return {
            "success": True,
            "models": models,
            "total_count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/models/{symbol}/{timeframe}", response_model=ModelInfoResponse)
    async def get_model_info(symbol: str, timeframe: str):
        """Information about specific model"""
        model_key = get_model_key(symbol, timeframe)
        
        if model_key not in MODEL_CACHE:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {symbol} {timeframe}"
            )
        
        model = MODEL_CACHE[model_key]
        info = model.get_model_info()
        
        return ModelInfoResponse(
            symbol=info["symbol"],
            timeframe=info["timeframe"], 
            model_type="advanced" if isinstance(model, AdvancedProphetModel) else "basic",
            is_trained=info["is_trained"],
            training_timestamp=datetime.fromisoformat(info["last_training_time"]) if info.get("last_training_time") else None,
            training_samples=info.get("training_data_size", 0),
            last_prediction=None,  # TODO: Add tracking last forecast
            model_metrics=info.get("training_metrics", {})
        )
    
    @app.delete("/models/{symbol}/{timeframe}")
    async def delete_model(symbol: str, timeframe: str):
        """Removal model from cache"""
        model_key = get_model_key(symbol, timeframe)
        
        if model_key not in MODEL_CACHE:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {symbol} {timeframe}"
            )
        
        del MODEL_CACHE[model_key]
        
        logger.info(f"Model deleted: {model_key}")
        
        return {
            "success": True,
            "message": f"Model {symbol} {timeframe} deleted successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    @app.get("/stats")
    async def get_api_stats():
        """Statistics API"""
        uptime = (datetime.now() - API_STATS["start_time"]).total_seconds()
        
        return {
            "success": True,
            "stats": {
                "requests_total": API_STATS["requests_total"],
                "errors_total": API_STATS["errors_total"],
                "models_trained": API_STATS["models_trained"],
                "predictions_made": API_STATS["predictions_made"],
                "models_cached": len(MODEL_CACHE),
                "websocket_connections": len(WEBSOCKET_CONNECTIONS),
                "uptime_seconds": uptime,
                "start_time": API_STATS["start_time"].isoformat(),
                "error_rate": API_STATS["errors_total"] / max(API_STATS["requests_total"], 1) * 100
            },
            "timestamp": datetime.now().isoformat()
        }
    
    # === WebSocket Support ===
    
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket endpoint for real-time notifications"""
        await websocket.accept()
        WEBSOCKET_CONNECTIONS[client_id] = websocket
        
        logger.info(f"WebSocket client connected: {client_id}")
        
        try:
            # Sending greetings
            await websocket.send_json({
                "type": "connection",
                "message": "Connected to Prophet Forecasting API",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Waiting messages from client
            while True:
                message = await websocket.receive_text()
                
                try:
                    data = json.loads(message)
                    await handle_websocket_message(client_id, data)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    })
                    
        except WebSocketDisconnect:
            logger.info(f"WebSocket client disconnected: {client_id}")
        finally:
            if client_id in WEBSOCKET_CONNECTIONS:
                del WEBSOCKET_CONNECTIONS[client_id]
    
    async def handle_websocket_message(client_id: str, message: Dict[str, Any]):
        """Processing messages from WebSocket client"""
        websocket = WEBSOCKET_CONNECTIONS.get(client_id)
        if not websocket:
            return
        
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Subscription on updates specific model
            symbol = message.get("symbol", "").upper()
            timeframe = message.get("timeframe", "").lower()
            
            await websocket.send_json({
                "type": "subscription",
                "message": f"Subscribed to {symbol} {timeframe} updates",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat()
            })
            
        elif message_type == "ping":
            # Response on ping for maintenance connections
            await websocket.send_json({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            })
    
    async def broadcast_forecast_update(symbol: str, forecast_result: Union[ForecastResult, AdvancedForecastResult]):
        """Mailing updates forecast through WebSocket"""
        if not WEBSOCKET_CONNECTIONS:
            return
        
        update_message = {
            "type": "forecast_update",
            "symbol": symbol,
            "timeframe": forecast_result.timeframe,
            "forecast_timestamp": forecast_result.forecast_timestamp.isoformat(),
            "summary": {
                "forecast_points": len(forecast_result.forecast_df),
                "first_prediction": forecast_result.forecast_df.iloc[0]["yhat"] if len(forecast_result.forecast_df) > 0 else None,
                "last_prediction": forecast_result.forecast_df.iloc[-1]["yhat"] if len(forecast_result.forecast_df) > 0 else None,
                "trend_direction": "up" if len(forecast_result.forecast_df) > 1 and forecast_result.forecast_df.iloc[-1]["yhat"] > forecast_result.forecast_df.iloc[0]["yhat"] else "down"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Sending all connected clients
        disconnected_clients = []
        for client_id, websocket in WEBSOCKET_CONNECTIONS.items():
            try:
                await websocket.send_json(update_message)
            except:
                disconnected_clients.append(client_id)
        
        # Removal disabled clients
        for client_id in disconnected_clients:
            del WEBSOCKET_CONNECTIONS[client_id]
    
    # === Error Handlers ===
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handler HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "type": "HTTPException",
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        """Total handler exceptions"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "type": "InternalServerError",
                    "code": "INTERNAL_ERROR", 
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    
    return app


# === Function launch ===

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    debug: bool = False,
    config: Optional[ProphetConfig] = None
):
    """
    Launch FastAPI server
    
    Args:
        host: Host for bindings
        port: Port for listening
        debug: Mode debugging
        config: Configuration application
    """
    if config is None:
        config = get_config()
    
    app = create_forecast_app(config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        access_log=debug,
        reload=debug
    )


if __name__ == "__main__":
    run_server(debug=True)