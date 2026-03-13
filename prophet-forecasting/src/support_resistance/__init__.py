"""
Support and Resistance Detection System
ML-Framework-1332 - Enterprise-grade support/resistance level detection

 2025: Advanced technical analysis, multiple detection algorithms,
performance-optimized with real-time updates and high accuracy.

## Key Features

### 🎯 Detection Algorithms
- Pivot points (Classical, Fibonacci, Camarilla, Woodie)
- Peak/Trough detection with fractal analysis
- Volume profile and volume-weighted levels
- Psychological levels and round numbers
- Machine learning-based detection

### 📊 Level Classification
- Strong vs Weak levels based on touch count
- Dynamic level strength calculation
- Time-decay for historical levels
- Confluence zone detection

### ⚡ Performance Optimization
- Caching and incremental updates
- Parallel processing for multiple timeframes
- Sub-50ms detection targets
- Memory-efficient algorithms

### 🔧 Integration Features
- Real-time level updates
- Multi-timeframe analysis
- Trading strategy integration
- Alert system for level breaks

## Usage Example

```python
from support_resistance import SupportResistanceDetector, DetectionMethod

# Initialize detector
detector = SupportResistanceDetector(
    symbol="BTC/USDT",
    timeframe="1h",
    methods=[DetectionMethod.PIVOT_POINTS, DetectionMethod.VOLUME_PROFILE]
)

# Detect levels
levels = detector.detect_levels(ohlcv_data)

# Get strongest levels
strong_levels = detector.get_strong_levels(min_strength=0.8)

# Check if price near level
is_near = detector.is_near_level(current_price, tolerance_percent=0.5)
```
"""

from .detector import (
    SupportResistanceDetector,
    DetectionMethod,
    LevelType,
    SupportResistanceLevel,
    DetectionResult
)
from .algorithms import (
    PivotPointCalculator,
    PeakTroughDetector,
    VolumeProfileAnalyzer,
    PsychologicalLevelDetector,
    MLLevelDetector
)
from .level_manager import (
    LevelManager,
    LevelStrength,
    ConfluenceZone,
    LevelUpdate
)
from .visualizer import (
    LevelVisualizer,
    ChartStyle,
    VisualizationConfig
)
from .validator import (
    LevelValidator,
    ValidationResult,
    BacktestResult
)

__all__ = [
    # Core detector
    "SupportResistanceDetector",
    "DetectionMethod",
    "LevelType",
    "SupportResistanceLevel",
    "DetectionResult",

    # Algorithms
    "PivotPointCalculator",
    "PeakTroughDetector",
    "VolumeProfileAnalyzer",
    "PsychologicalLevelDetector",
    "MLLevelDetector",

    # Level management
    "LevelManager",
    "LevelStrength",
    "ConfluenceZone",
    "LevelUpdate",

    # Visualization
    "LevelVisualizer",
    "ChartStyle",
    "VisualizationConfig",

    # Validation
    "LevelValidator",
    "ValidationResult",
    "BacktestResult"
]

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__description__ = "Enterprise support/resistance detection with  2025 patterns"