"""
ML Elliott Wave Analyzer - Comprehensive Elliott Wave analysis for crypto trading.

Enterprise-grade Elliott Wave pattern recognition and analysis system with ,
designed specifically for 24/7 crypto markets with high volatility handling.
"""

__version__ = "1.0.0"
__author__ = "ML Elliott Wave Contributors"

from .patterns import *
from .analysis import *
from .fibonacci import *
from .degrees import *
from .rules import *
from .ml import *
from .technical import *
from .realtime import *
from .backtesting import *
from .visualization import *
from .api import *
from .utils import *

# Main classes for easy import
from .patterns.impulse_wave import ImpulseWaveDetector
from .patterns.corrective_wave import CorrectiveWaveDetector
from .analysis.wave_counter import WaveCounter
from .fibonacci.fibonacci_retracement import FibonacciRetracement
from .rules.elliott_rules import ElliottRulesEngine
from .ml.cnn_wave_detector import CNNWaveDetector
from .realtime.stream_analyzer import StreamAnalyzer
from .api.rest_api import ElliottWaveAPI

__all__ = [
    # Pattern Detection
    'ImpulseWaveDetector',
    'CorrectiveWaveDetector',
    
    # Analysis
    'WaveCounter',
    
    # Fibonacci
    'FibonacciRetracement',
    
    # Rules
    'ElliottRulesEngine',
    
    # ML
    'CNNWaveDetector',
    
    # Real-time
    'StreamAnalyzer',
    
    # API
    'ElliottWaveAPI',
]