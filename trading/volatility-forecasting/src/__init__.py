"""
ML Volatility Forecasting v5.0

Enterprise-grade volatility forecasting system with GARCH models,
realized volatility estimators, and implied volatility.

Key features:
- GARCH family models (GARCH, EGARCH, GJR-GARCH, FIGARCH, DCC-GARCH)
- Realized volatility (RV, BPV, RK, Realized GARCH)
- ML models (LSTM, HAR-RV, Random Forest)
- Risk metrics (VaR, CVaR, volatility cones)
- Real-time API and WebSocket streaming
- Trading integration with position sizing

"""

from typing import Dict, Any, Optional, List
import sys
import os
import logging

# Package version
__version__ = "5.0.0"
__author__ = "ML Volatility Team"
__email__ = "example@example.com"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('volatility-forecasting.log')
    ]
)

logger = logging.getLogger(__name__)

# Core imports
try:
    from .models.garch_models import (
        GARCHModel,
        EGARCHModel, 
        GJRGARCHModel,
        FIGARCHModel,
        DCCGARCHModel,
        GARCHModelSelector
    )
    
    from .estimators.realized_volatility import (
        RealizedVolatilityEstimator,
        BipowerVariation,
        RealizedKernel,
        RealizedGARCH
    )
    
    from .estimators.implied_volatility import (
        ImpliedVolatilityExtractor,
        VolatilitySmileModel,
        CryptoVolatilityIndex
    )
    
    from .models.ml_volatility import (
        LSTMVolatilityModel,
        HARRVModel,
        RandomForestVolatility,
        EnsembleVolatilityModel
    )
    
    from .utils.risk_metrics import (
        RiskMetricsCalculator,
        VaRCalculator,
        VolatilityCone,
        DrawdownAnalyzer
    )
    
    from .api.volatility_api import (
        VolatilityAPI,
        create_volatility_app
    )
    
    from .utils.trading_integration import (
        PositionSizer,
        DynamicStopLoss,
        PortfolioOptimizer
    )
    
    logger.info(f"ML Volatility Forecasting v{__version__} initialized successfully")
    
except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    logger.warning("⚠️ Some components may not be available")

# Export main classes
__all__ = [
    # Version
    "__version__",
    "__author__", 
    "__email__",
    
    # GARCH Models
    "GARCHModel",
    "EGARCHModel", 
    "GJRGARCHModel",
    "FIGARCHModel",
    "DCCGARCHModel",
    "GARCHModelSelector",
    
    # Realized Volatility
    "RealizedVolatilityEstimator",
    "BipowerVariation",
    "RealizedKernel", 
    "RealizedGARCH",
    
    # Implied Volatility
    "ImpliedVolatilityExtractor",
    "VolatilitySmileModel",
    "CryptoVolatilityIndex",
    
    # ML Models
    "LSTMVolatilityModel",
    "HARRVModel",
    "RandomForestVolatility",
    "EnsembleVolatilityModel",
    
    # Risk Metrics
    "RiskMetricsCalculator",
    "VaRCalculator", 
    "VolatilityCone",
    "DrawdownAnalyzer",
    
    # API
    "VolatilityAPI",
    "create_volatility_app",
    
    # Trading Integration
    "PositionSizer",
    "DynamicStopLoss",
    "PortfolioOptimizer"
]

def get_package_info() -> Dict[str, Any]:
    """Get package information"""
    return {
        "name": "ml-volatility-forecasting",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Enterprise-grade Volatility Forecasting System",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "components": {
            "garch_models": True,
            "realized_volatility": True,
            "implied_volatility": True, 
            "ml_models": True,
            "risk_metrics": True,
            "api": True,
            "trading_integration": True
        }
    }

def check_dependencies() -> Dict[str, bool]:
    """Check availability of core dependencies"""
    dependencies = {
        "arch": False,
        "pandas": False,
        "numpy": False,
        "scipy": False,
        "sklearn": False,
        "tensorflow": False,
        "torch": False,
        "fastapi": False,
        "redis": False,
    }
    
    for dep in dependencies:
        try:
            if dep == "sklearn":
                import sklearn
            elif dep == "tensorflow":
                import tensorflow
            elif dep == "torch":  
                import torch
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            logger.warning(f"⚠️ Optional dependency '{dep}' not available")
    
    return dependencies

# Check dependencies on import
if __name__ != "__main__":
    deps = check_dependencies()
    missing_critical = [dep for dep, available in deps.items() 
                       if not available and dep in ["arch", "pandas", "numpy"]]
    
    if missing_critical:
        logger.error(f"❌ Critical dependencies missing: {missing_critical}")
        raise ImportError(f"Missing critical dependencies: {missing_critical}")

def create_volatility_forecasting_system(
    config: Optional[Dict[str, Any]] = None
) -> "VolatilityForecastingSystem":
    """
    Create a complete volatility forecasting system

    Args:
        config: System configuration

    Returns:
        VolatilityForecastingSystem: Configured system
    """
    from .core.volatility_system import VolatilityForecastingSystem
    
    default_config = {
        "garch_models": ["GARCH", "EGARCH", "GJR-GARCH"],
        "ml_models": ["LSTM", "HAR-RV", "RandomForest"],
        "risk_metrics": ["VaR", "CVaR", "VolatilityCone"],
        "data_frequency": "1h",
        "forecast_horizon": 24,
        "validation_window": 30,
        "cache_enabled": True,
        "redis_url": "redis://localhost:6379",
        "database_url": "postgresql://localhost/volatility",
    }
    
    if config:
        default_config.update(config)
    
    return VolatilityForecastingSystem(default_config)

logger.info("Volatility Forecasting System ready for production use!")