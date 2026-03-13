"""
ML-Framework ML Prophet Forecasting Package

Enterprise-grade Prophet forecasting system for cryptocurrency price predictions.
Implements Facebook Prophet with enterprise patterns for production-ready
time series forecasting with advanced features.

Key Features:
- Multi-cryptocurrency support with automated model selection
- Advanced seasonality detection (daily, weekly, monthly, yearly)
- Holiday and event effects for crypto markets
- Changepoint detection with uncertainty quantification
- Cross-validation and backtesting framework
- Real-time prediction API with WebSocket streaming
- Hyperparameter optimization with Bayesian methods
- Multi-variate forecasting with external regressors
- Production monitoring and observability

enterprise integration:
- Cloud-native architecture with containerized deployment
- Microservices design with API-first approach
- Event-driven processing with async/await patterns
- Comprehensive observability and monitoring
- Enterprise security and compliance
- Scalable data processing with optimized performance
"""

from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime, timedelta

# Version package
__version__ = "5.0.0"
__author__ = "ML-Framework Team"
__license__ = "MIT"

# Configuration logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Core imports for outer usage
try:
    from .models.prophet_model import ProphetForecaster
    from .models.advanced_prophet import AdvancedProphetModel
    from .preprocessing.data_processor import CryptoDataProcessor
    from .validation.forecast_validator import ForecastValidator
    from .config.prophet_config import ProphetConfig, ModelConfig
    from .utils.logger import get_logger
    from .utils.metrics import ForecastMetrics
    
    # API imports
    from .api.forecast_api import create_forecast_app
    
    __all__ = [
        # Core classes
        "ProphetForecaster",
        "AdvancedProphetModel", 
        "CryptoDataProcessor",
        "ForecastValidator",
        
        # Configuration
        "ProphetConfig",
        "ModelConfig",
        
        # Utilities
        "get_logger",
        "ForecastMetrics",
        
        # API
        "create_forecast_app",
        
        # Constants
        "__version__",
        "__author__",
        "__license__"
    ]
    
except ImportError as e:
    # IN mode development some modules can be unavailable
    import warnings
    warnings.warn(f"Some modules not available during development: {e}")
    
    __all__ = [
        "__version__",
        "__author__", 
        "__license__"
    ]

# Constants for system
SUPPORTED_CRYPTOCURRENCIES = [
    "BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "AVAX", "DOT", "MATIC", "LINK",
    "UNI", "AAVE", "SUSHI", "CRV", "YFI", "COMP", "MKR", "SNX", "1INCH", "ALPHA"
]

SUPPORTED_TIMEFRAMES = [
    "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"
]

DEFAULT_FORECAST_PERIODS = {
    "1m": 60,    # 1 hour forward
    "5m": 288,   # 1 day forward  
    "15m": 96,   # 1 day forward
    "30m": 48,   # 1 day forward
    "1h": 168,   # 1 week forward
    "4h": 42,    # 1 week forward
    "1d": 30,    # 1 month forward
    "1w": 12,    # 3 months forward
}

# Types for export
ForecastResult = Dict[str, Any]
ModelParams = Dict[str, Union[str, int, float, bool]]
TimeSeriesData = Dict[str, Union[datetime, float]]

def get_package_info() -> Dict[str, str]:
    """
    Get information about package
    
    Returns:
        Dict with information about version, author, license
    """
    return {
        "name": "ml-framework-ml-prophet-forecasting",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Enterprise Prophet forecasting for cryptocurrency predictions",
        "supported_cryptos": len(SUPPORTED_CRYPTOCURRENCIES),
        "supported_timeframes": len(SUPPORTED_TIMEFRAMES)
    }

def validate_cryptocurrency(symbol: str) -> bool:
    """
    Check support cryptocurrency
    
    Args:
        symbol: Symbol cryptocurrency (for example, "BTC")
        
    Returns:
        True if is supported
    """
    return symbol.upper() in SUPPORTED_CRYPTOCURRENCIES

def validate_timeframe(timeframe: str) -> bool:
    """
    Check support timeframe
    
    Args:
        timeframe: Timeframe (for example, "1h")
        
    Returns:
        True if is supported
    """
    return timeframe.lower() in SUPPORTED_TIMEFRAMES

def get_default_forecast_periods(timeframe: str) -> int:
    """
    Get number periods forecast by default
    
    Args:
        timeframe: Timeframe
        
    Returns:
        Number periods for forecast
    """
    return DEFAULT_FORECAST_PERIODS.get(timeframe.lower(), 30)

# Configuration for  integration
ENTERPRISE_CONFIG = {
    "service_name": "prophet-forecasting",
    "service_version": __version__,
    "api_prefix": "/api/v1",
    "health_check_endpoint": "/health",
    "metrics_endpoint": "/metrics",
    "docs_endpoint": "/docs",
    "openapi_endpoint": "/openapi.json"
}

def create__metadata() -> Dict[str, Any]:
    """
    Create metadata for  integration
    
    Returns:
        Dictionary with metadata service
    """
    return {
        **ENTERPRISE_CONFIG,
        "capabilities": [
            "cryptocurrency_forecasting",
            "multi_timeframe_prediction", 
            "seasonality_detection",
            "changepoint_analysis",
            "uncertainty_quantification",
            "real_time_prediction",
            "backtesting_validation",
            "hyperparameter_optimization"
        ],
        "integrations": [
            "fastapi",
            "websocket",
            "prometheus",
            "redis",
            "postgresql"
        ],
        "deployment": {
            "containerized": True,
            "scalable": True,
            "cloud_native": True,
            "monitoring": True
        }
    }