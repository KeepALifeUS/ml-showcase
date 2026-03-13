"""
Harmonic Patterns Module - Professional Pattern Detection System.

Enterprise Implementation:
- Advanced harmonic pattern detection algorithms
- Real-time pattern scanning and validation
- Machine learning pattern classification
- Professional trading signal generation

Supported Patterns:
- Gartley 222 (Classic harmonic pattern) ✅ IMPLEMENTED
- Bat Pattern (Carney 2001) ✅ IMPLEMENTED
- Butterfly Pattern (Extended AD ratio) ✅ IMPLEMENTED
- Crab Pattern (Extreme extensions) ✅ IMPLEMENTED
- Shark Pattern (Unique O-X-A-B-C structure) ✅ IMPLEMENTED
- Cypher Pattern (BC extension rules) ✅ IMPLEMENTED

Example Usage:
    ```python
    from ml_harmonic_patterns.patterns import (
        GartleyPattern, BatPattern, ButterflyPattern, 
        CrabPattern, SharkPattern, CypherPattern
    )
    
    # Initialize different detectors
    gartley_detector = GartleyPattern(tolerance=0.05, min_confidence=0.75)
    bat_detector = BatPattern(tolerance=0.05, min_confidence=0.75)
    butterfly_detector = ButterflyPattern(tolerance=0.05, min_confidence=0.75)
    crab_detector = CrabPattern(tolerance=0.08, min_confidence=0.75)
    shark_detector = SharkPattern(tolerance=0.06, min_confidence=0.75)
    cypher_detector = CypherPattern(tolerance=0.06, min_confidence=0.75)
    
    # Detect patterns in OHLCV data
    gartley_patterns = gartley_detector.detect_patterns(ohlcv_data, symbol="BTCUSDT", timeframe="1h")
    bat_patterns = bat_detector.detect_patterns(ohlcv_data, symbol="BTCUSDT", timeframe="1h")
    
    # Generate trading signals
    if gartley_patterns:
        best_pattern = gartley_patterns[0]  # Highest confidence
        signals = gartley_detector.get_entry_signals(best_pattern)
        position_size = gartley_detector.calculate_position_size(best_pattern, account_balance=10000)
        viz_data = gartley_detector.prepare_visualization_data(best_pattern)
    ```
"""

from .gartley_pattern import (
    GartleyPattern,
    PatternResult,
    PatternType,
    PatternValidation,
    FibonacciRatios,
    PatternPoint,
    analyze_pattern_performance,
    filter_patterns_by_quality
)

from .bat_pattern import (
    BatPattern,
    BatFibonacciRatios
)

from .butterfly_pattern import (
    ButterflyPattern,
    ButterflyFibonacciRatios
)

from .crab_pattern import (
    CrabPattern,
    CrabFibonacciRatios
)

from .shark_pattern import (
    SharkPattern,
    SharkPatternResult,
    SharkFibonacciRatios
)

from .cypher_pattern import (
    CypherPattern,
    CypherFibonacciRatios
)

__all__ = [
    # Main pattern detectors
    "GartleyPattern",
    "BatPattern", 
    "ButterflyPattern",
    "CrabPattern",
    "SharkPattern",
    "CypherPattern",
    
    # Data structures
    "PatternResult",
    "SharkPatternResult",
    "PatternType", 
    "PatternValidation",
    "PatternPoint",
    
    # Fibonacci ratios
    "FibonacciRatios",
    "BatFibonacciRatios",
    "ButterflyFibonacciRatios", 
    "CrabFibonacciRatios",
    "SharkFibonacciRatios",
    "CypherFibonacciRatios",
    
    # Utility functions
    "analyze_pattern_performance",
    "filter_patterns_by_quality"
]