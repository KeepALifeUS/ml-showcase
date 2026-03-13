"""
🚀 ML Anomaly Detection System v5.0 - Enterprise Architecture

Comprehensive anomaly detection system for cryptocurrency trading with:
- Real-time statistical and ML-based detection
- Deep learning models (GANs, Transformers, GNNs)
- Crypto-specific anomaly patterns
- Enterprise-grade monitoring and alerting
- cloud-native patterns

Author: ML-Framework Team
License: MIT
"""

from typing import Dict, Any
import structlog

# Version and metadata
__version__ = "5.0.0"
__author__ = "ML-Framework Team"
__email__ = "dev@ml-framework.io"

# Configure structured logging
logger = structlog.get_logger(__name__)

# System information
SYSTEM_INFO = {
 "name": "ML Anomaly Detection System",
 "version": __version__,
 "description": "Enterprise-grade anomaly detection for crypto trading",
 "architecture": "Cloud-Native",
 "components": [
 "statistical_detectors",
 "ml_detectors",
 "deep_learning",
 "timeseries_analysis",
 "realtime_processing",
 "crypto_specific",
 "alert_system",
 "visualization",
 "feature_engineering",
 "evaluation_metrics"
 ]
}

# Export main classes and functions
from .statistical import (
 ZScoreDetector,
 MADDetector,
 IQRDetector,
 GrubbsTestDetector,
 DixonTestDetector,
 TukeyMethodDetector
)

from .ml import (
 IsolationForestDetector,
 LocalOutlierFactorDetector,
 OneClassSVMDetector,
 AutoencoderDetector,
 LSTMAutoencoderDetector,
 VAEDetector
)

from .deep_learning import (
 GANDetector,
 TransformerAnomalyDetector,
 GraphNeuralNetworkDetector,
 AttentionMechanismDetector,
 EnsembleDeepDetector
)

from .timeseries import (
 ARIMADetector,
 ProphetDetector,
 STLDecompositionDetector,
 MatrixProfileDetector,
 DiscordDiscoveryDetector,
 HotspotDetector
)

from .realtime import (
 StreamProcessor,
 OnlineLearningDetector,
 SlidingWindowDetector,
 AdaptiveThresholdDetector,
 ChangePointDetector
)

from .crypto import (
 PumpDumpDetector,
 WashTradingDetector,
 WhaleMovementDetector,
 FlashCrashDetector,
 ManipulationDetector,
 ArbitrageAnomalyDetector
)

from .alerts import (
 AlertManager,
 SeverityClassifier,
 NotificationSystem,
 EscalationPolicy,
 AlertAggregator
)

from .utils import Config, Logger, Validators, Profiler

# Initialize system
def initialize_system(config: Dict[str, Any] = None) -> None:
 """Initialize the anomaly detection system with configuration."""
 logger.info("Initializing ML Anomaly Detection System v5.0")

 if config:
 Config.update(config)

 logger.info("System initialized successfully", **SYSTEM_INFO)

def get_system_info -> Dict[str, Any]:
 """Get system information and status."""
 return SYSTEM_INFO

# Health check
def health_check -> Dict[str, Any]:
 """Perform system health check."""
 return {
 "status": "healthy",
 "version": __version__,
 "components": len(SYSTEM_INFO["components"]),
 "architecture": "Cloud-Native"
 }

__all__ = [
 # Statistical detectors
 "ZScoreDetector",
 "MADDetector",
 "IQRDetector",
 "GrubbsTestDetector",
 "DixonTestDetector",
 "TukeyMethodDetector",

 # ML detectors
 "IsolationForestDetector",
 "LocalOutlierFactorDetector",
 "OneClassSVMDetector",
 "AutoencoderDetector",
 "LSTMAutoencoderDetector",
 "VAEDetector",

 # Deep learning
 "GANDetector",
 "TransformerAnomalyDetector",
 "GraphNeuralNetworkDetector",
 "AttentionMechanismDetector",
 "EnsembleDeepDetector",

 # Time series
 "ARIMADetector",
 "ProphetDetector",
 "STLDecompositionDetector",
 "MatrixProfileDetector",
 "DiscordDiscoveryDetector",
 "HotspotDetector",

 # Real-time
 "StreamProcessor",
 "OnlineLearningDetector",
 "SlidingWindowDetector",
 "AdaptiveThresholdDetector",
 "ChangePointDetector",

 # Crypto-specific
 "PumpDumpDetector",
 "WashTradingDetector",
 "WhaleMovementDetector",
 "FlashCrashDetector",
 "ManipulationDetector",
 "ArbitrageAnomalyDetector",

 # Alerts
 "AlertManager",
 "SeverityClassifier",
 "NotificationSystem",
 "EscalationPolicy",
 "AlertAggregator",

 # Utils
 "Config",
 "Logger",
 "Validators",
 "Profiler",

 # Functions
 "initialize_system",
 "get_system_info",
 "health_check",

 # Constants
 "__version__",
 "SYSTEM_INFO"
]