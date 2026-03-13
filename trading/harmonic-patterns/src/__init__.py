"""
ML Harmonic Patterns - Comprehensive Harmonic Pattern Detector for Crypto Trading.

High-precision pattern detection system with
real-time processing, machine learning classification, and advanced analytics.

This package provides:
- Comprehensive harmonic pattern detection (Gartley, Bat, Butterfly, Crab, Shark, Cypher)
- Advanced Fibonacci analysis and confluence detection
- Real-time pattern scanning and alerting
- Multi-timeframe pattern analysis
- Machine learning pattern classification
- Trading signal generation
- Performance backtesting
- Interactive visualization
"""

from .patterns import *
from .fibonacci import *
from .scanner import *
from .signals import *

__version__ = "1.0.0"
__author__ = "ML Harmonic Patterns Contributors"
__email__ = ""

__all__ = [
    # Pattern classes
    "GartleyPattern",
    "BatPattern", 
    "ButterflyPattern",
    "CrabPattern",
    "SharkPattern",
    "CypherPattern",
    "ABCDPattern",
    "ThreeDrivesPattern",
    
    # Fibonacci analysis
    "FibonacciAnalyzer",
    "RatioValidator",
    "RetracementCalculator",
    "ExtensionCalculator",
    "ClusterAnalyzer",
    "ConfluenceDetector",
    
    # Pattern scanning
    "MultiPatternScanner",
    "RealtimeScanner",
    "HistoricalScanner",
    "PatternRanker",
    "PatternFilter",
    "PatternValidator",
    
    # Trading signals
    "EntrySignals",
    "ExitSignals", 
    "StopLossCalculator",
    "TakeProfitCalculator",
    "RiskRewardAnalyzer",
    "PositionSizer"
]