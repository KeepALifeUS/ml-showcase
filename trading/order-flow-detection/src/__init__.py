"""
ML Order Flow Detection Package
Enterprise-grade order flow pattern detection system with enterprise patterns
"""

from .utils.config import get_settings, OrderFlowSettings
from .utils.logger import get_logger, setup_logging, LoggerType

# Order Flow Analysis
from .order_flow.delta_analyzer import DeltaAnalyzer, DeltaMetrics, DeltaPattern
from .order_flow.cumulative_delta import CumulativeDeltaAnalyzer, CumulativeDeltaBar
from .order_flow.footprint_chart import FootprintAnalyzer, FootprintBar, PriceLevel

# Pattern Detection
from .patterns.iceberg_detector import IcebergDetector, IcebergSignature
from .patterns.spoofing_detector import SpoofingDetector, SpoofingSignal

__version__ = "1.0.0"
__author__ = "ML-Framework Team"

# Initialize logging on import
setup_logging()

# Export main components
__all__ = [
    # Configuration and utilities
    'get_settings',
    'OrderFlowSettings',
    'get_logger',
    'setup_logging',
    'LoggerType',
    
    # Order Flow Analysis
    'DeltaAnalyzer',
    'DeltaMetrics',
    'DeltaPattern',
    'CumulativeDeltaAnalyzer',
    'CumulativeDeltaBar',
    'FootprintAnalyzer',
    'FootprintBar',
    'PriceLevel',
    
    # Pattern Detection
    'IcebergDetector',
    'IcebergSignature',
    'SpoofingDetector',
    'SpoofingSignal',
    
    # Package metadata
    '__version__',
    '__author__'
]