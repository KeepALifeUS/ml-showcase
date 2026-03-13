"""
Impulse Wave Pattern Detection for Elliott Wave Analysis.

5-wave impulse pattern detection,
optimized for crypto market volatility and 24/7 trading conditions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timezone

from ..utils.logger import get_logger, trading_logger, performance_monitor
from ..utils.config import config

logger = get_logger(__name__)


class WaveType(str, Enum):
    """Elliott Wave types in impulse pattern."""
    WAVE_1 = "wave_1"
    WAVE_2 = "wave_2"
    WAVE_3 = "wave_3"
    WAVE_4 = "wave_4"
    WAVE_5 = "wave_5"


class ImpulseDirection(str, Enum):
    """Direction of impulse wave."""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class WavePoint:
    """Represents a wave turning point."""
    index: int
    price: float
    timestamp: datetime
    volume: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)


@dataclass
class ImpulseWave:
    """
    Complete impulse wave structure.
    
    Rich domain model with validation.
    """
    wave_1: Tuple[WavePoint, WavePoint]  # Start, End
    wave_2: Tuple[WavePoint, WavePoint]  # Start, End
    wave_3: Tuple[WavePoint, WavePoint]  # Start, End
    wave_4: Tuple[WavePoint, WavePoint]  # Start, End
    wave_5: Tuple[WavePoint, WavePoint]  # Start, End
    
    direction: ImpulseDirection
    confidence: float
    timeframe: str
    symbol: str
    
    # Wave measurements
    wave_1_length: float
    wave_3_length: float
    wave_5_length: float
    wave_2_retracement: float
    wave_4_retracement: float
    
    # Fibonacci relationships
    fibonacci_ratios: Dict[str, float]
    
    # Validation results
    rules_validation: Dict[str, bool]
    guidelines_score: float
    
    # Performance metrics
    detected_at: datetime
    processing_time_ms: float
    
    def __post_init__(self):
        """Validate impulse wave structure."""
        self.validate_structure()
        self.calculate_fibonacci_relationships()
        
    def validate_structure(self) -> bool:
        """
        Validate Elliott Wave impulse rules.
        
        Business rule validation.
        """
        rules = {
            "wave_3_not_shortest": self.wave_3_length > min(self.wave_1_length, self.wave_5_length),
            "wave_2_doesnt_exceed_wave_1": self.wave_2_retracement < 1.0,
            "wave_4_doesnt_exceed_wave_1": self._validate_wave_4_overlap(),
            "alternation": self._validate_alternation(),
            "wave_degrees_consistent": self._validate_wave_degrees()
        }
        
        self.rules_validation = rules
        return all(rules.values())
        
    def _validate_wave_4_overlap(self) -> bool:
        """Validate that wave 4 doesn't overlap wave 1."""
        if self.direction == ImpulseDirection.BULLISH:
            wave_1_high = self.wave_1[1].price
            wave_4_low = min(self.wave_4[0].price, self.wave_4[1].price)
            return wave_4_low > wave_1_high
        else:
            wave_1_low = self.wave_1[1].price
            wave_4_high = max(self.wave_4[0].price, self.wave_4[1].price)
            return wave_4_high < wave_1_low
            
    def _validate_alternation(self) -> bool:
        """Validate rule of alternation between waves 2 and 4."""
        # Simple alternation check - more complex logic needed
        wave_2_depth = self.wave_2_retracement
        wave_4_depth = self.wave_4_retracement
        
        # Different retracement depths suggest alternation
        return abs(wave_2_depth - wave_4_depth) > 0.1
        
    def _validate_wave_degrees(self) -> bool:
        """Validate wave degree consistency."""
        # Placeholder - implement wave degree validation
        return True
        
    def calculate_fibonacci_relationships(self) -> Dict[str, float]:
        """Calculate Fibonacci relationships between waves."""
        ratios = {}
        
        # Wave 3 to Wave 1 ratios
        ratios["wave_3_to_1"] = self.wave_3_length / self.wave_1_length
        
        # Wave 5 to Wave 1 ratios  
        ratios["wave_5_to_1"] = self.wave_5_length / self.wave_1_length
        
        # Wave 5 to Wave 3 ratios
        ratios["wave_5_to_3"] = self.wave_5_length / self.wave_3_length
        
        # Common Fibonacci levels
        fib_levels = [0.382, 0.618, 1.0, 1.272, 1.618, 2.618]
        
        # Find closest Fibonacci ratios
        for key, value in ratios.items():
            closest_fib = min(fib_levels, key=lambda x: abs(x - value))
            ratios[f"{key}_fib_match"] = closest_fib
            ratios[f"{key}_fib_accuracy"] = 1 - abs(value - closest_fib) / closest_fib
            
        self.fibonacci_ratios = ratios
        return ratios
        
    def get_projection_targets(self) -> Dict[str, float]:
        """Get price projection targets for next move."""
        wave_5_end = self.wave_5[1].price
        
        if self.direction == ImpulseDirection.BULLISH:
            # Bullish projections beyond wave 5
            projections = {
                "fib_1272": wave_5_end * 1.272,
                "fib_1618": wave_5_end * 1.618,
                "fib_2618": wave_5_end * 2.618
            }
        else:
            # Bearish projections beyond wave 5
            projections = {
                "fib_1272": wave_5_end * 0.786,  # Inverse projection
                "fib_1618": wave_5_end * 0.618,
                "fib_2618": wave_5_end * 0.382
            }
            
        return projections
        
    @property
    def is_valid(self) -> bool:
        """Check if impulse wave passes all validation rules."""
        return all(self.rules_validation.values()) and self.confidence > 0.7
        
    @property
    def total_move(self) -> float:
        """Calculate total move from start to end of impulse."""
        start_price = self.wave_1[0].price
        end_price = self.wave_5[1].price
        return abs(end_price - start_price)


class ImpulseWaveDetector:
    """
    Advanced Impulse Wave Detection System.
    
    
    - High-performance pattern recognition
    - Multi-timeframe analysis
    - Crypto market optimization
    - Real-time processing capability
    """
    
    def __init__(self, 
                 min_wave_length: int = 5,
                 max_wave_length: int = 200,
                 fibonacci_tolerance: float = 0.15,
                 confidence_threshold: float = 0.7):
        """
        Initialize impulse wave detector.
        
        Args:
            min_wave_length: Minimum points in a wave
            max_wave_length: Maximum points in a wave
            fibonacci_tolerance: Tolerance for Fibonacci ratio matching
            confidence_threshold: Minimum confidence for valid detection
        """
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.fibonacci_tolerance = fibonacci_tolerance
        self.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.detection_stats = {
            "total_detections": 0,
            "valid_detections": 0,
            "false_positives": 0,
            "processing_times": []
        }
        
    @performance_monitor
    async def detect_impulse_waves(self, 
                                 data: pd.DataFrame, 
                                 symbol: str,
                                 timeframe: str) -> List[ImpulseWave]:
        """
        Detect impulse wave patterns in price data.
        
        Async processing with performance monitoring.
        
        Args:
            data: OHLCV price data
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            List[ImpulseWave]: Detected impulse waves
        """
        start_time = datetime.utcnow()
        
        # Validate input data
        if not self._validate_data(data):
            logger.warning(f"Invalid data for {symbol} {timeframe}")
            return []
            
        # Find pivot points
        pivots = await self._find_pivot_points(data)
        if len(pivots) < 6:  # Need at least 6 points for 5-wave structure
            return []
            
        # Detect potential impulse structures
        impulse_candidates = await self._scan_for_impulse_patterns(pivots, data)
        
        # Validate and score candidates
        valid_impulses = []
        for candidate in impulse_candidates:
            impulse_wave = await self._validate_impulse_candidate(
                candidate, symbol, timeframe, start_time
            )
            if impulse_wave and impulse_wave.confidence >= self.confidence_threshold:
                valid_impulses.append(impulse_wave)
                
        # Log detection results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="impulse",
            confidence=max([w.confidence for w in valid_impulses]) if valid_impulses else 0,
            detected_count=len(valid_impulses),
            candidates_scanned=len(impulse_candidates)
        )
        
        self.detection_stats["total_detections"] += len(impulse_candidates)
        self.detection_stats["valid_detections"] += len(valid_impulses)
        
        return valid_impulses
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input price data."""
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < self.min_wave_length * 5:  # Need data for 5 waves
            return False
            
        if data.isnull().any().any():
            return False
            
        return True
        
    async def _find_pivot_points(self, data: pd.DataFrame, 
                               window: int = 5) -> List[WavePoint]:
        """
        Find significant pivot points in price data.
        
        Efficient pivot detection with configurable sensitivity.
        """
        highs = data['high'].values
        lows = data['low'].values
        timestamps = pd.to_datetime(data.index)
        volumes = data.get('volume', [None] * len(data)).values
        
        pivots = []
        
        # Find local highs
        for i in range(window, len(highs) - window):
            if all(highs[i] > highs[j] for j in range(i - window, i)) and \
               all(highs[i] > highs[j] for j in range(i + 1, i + window + 1)):
                pivots.append(WavePoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    volume=volumes[i] if volumes[i] is not None else None
                ))
                
        # Find local lows
        for i in range(window, len(lows) - window):
            if all(lows[i] < lows[j] for j in range(i - window, i)) and \
               all(lows[i] < lows[j] for j in range(i + 1, i + window + 1)):
                pivots.append(WavePoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    volume=volumes[i] if volumes[i] is not None else None
                ))
                
        # Sort by timestamp
        pivots.sort(key=lambda x: x.index)
        
        return pivots
        
    async def _scan_for_impulse_patterns(self, pivots: List[WavePoint], 
                                       data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Scan pivot points for potential impulse wave patterns.
        
        Pattern matching with crypto market adaptations.
        """
        candidates = []
        
        # Need at least 6 pivots for 5-wave structure (start + 5 ends)
        for start_idx in range(len(pivots) - 5):
            # Try different combinations of 6 consecutive pivots
            candidate_pivots = pivots[start_idx:start_idx + 6]
            
            # Check if they form a valid 5-wave sequence
            if self._is_valid_impulse_sequence(candidate_pivots):
                candidate = self._build_impulse_candidate(candidate_pivots, data)
                if candidate:
                    candidates.append(candidate)
                    
        return candidates
        
    def _is_valid_impulse_sequence(self, pivots: List[WavePoint]) -> bool:
        """Check if pivot sequence could form valid impulse wave."""
        if len(pivots) != 6:
            return False
            
        # Check alternating direction pattern
        directions = []
        for i in range(len(pivots) - 1):
            if pivots[i + 1].price > pivots[i].price:
                directions.append(1)  # Up
            else:
                directions.append(-1)  # Down
                
        # For bullish impulse: up, down, up, down, up
        # For bearish impulse: down, up, down, up, down
        bullish_pattern = [1, -1, 1, -1, 1]
        bearish_pattern = [-1, 1, -1, 1, -1]
        
        return directions == bullish_pattern or directions == bearish_pattern
        
    def _build_impulse_candidate(self, pivots: List[WavePoint], 
                                data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Build impulse wave candidate from pivots."""
        try:
            # Determine direction
            if pivots[1].price > pivots[0].price:
                direction = ImpulseDirection.BULLISH
            else:
                direction = ImpulseDirection.BEARISH
                
            # Build wave structure
            waves = {
                'wave_1': (pivots[0], pivots[1]),
                'wave_2': (pivots[1], pivots[2]),
                'wave_3': (pivots[2], pivots[3]),
                'wave_4': (pivots[3], pivots[4]),
                'wave_5': (pivots[4], pivots[5]),
                'direction': direction,
                'pivots': pivots
            }
            
            return waves
            
        except Exception as e:
            logger.error(f"Error building impulse candidate: {e}")
            return None
            
    async def _validate_impulse_candidate(self, candidate: Dict[str, Any], 
                                        symbol: str, timeframe: str,
                                        start_time: datetime) -> Optional[ImpulseWave]:
        """Validate and score impulse wave candidate."""
        try:
            # Calculate wave measurements
            measurements = self._calculate_wave_measurements(candidate)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(candidate, measurements)
            
            if confidence < self.confidence_threshold:
                return None
                
            # Create ImpulseWave object
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            impulse_wave = ImpulseWave(
                wave_1=candidate['wave_1'],
                wave_2=candidate['wave_2'],
                wave_3=candidate['wave_3'],
                wave_4=candidate['wave_4'],
                wave_5=candidate['wave_5'],
                direction=candidate['direction'],
                confidence=confidence,
                timeframe=timeframe,
                symbol=symbol,
                wave_1_length=measurements['wave_1_length'],
                wave_3_length=measurements['wave_3_length'],
                wave_5_length=measurements['wave_5_length'],
                wave_2_retracement=measurements['wave_2_retracement'],
                wave_4_retracement=measurements['wave_4_retracement'],
                fibonacci_ratios={},  # Will be calculated in __post_init__
                rules_validation={},  # Will be calculated in __post_init__
                guidelines_score=confidence,
                detected_at=datetime.utcnow(),
                processing_time_ms=processing_time
            )
            
            return impulse_wave
            
        except Exception as e:
            logger.error(f"Error validating impulse candidate: {e}")
            return None
            
    def _calculate_wave_measurements(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Calculate wave measurements and retracements."""
        measurements = {}
        
        # Wave lengths (absolute price moves)
        measurements['wave_1_length'] = abs(
            candidate['wave_1'][1].price - candidate['wave_1'][0].price
        )
        measurements['wave_3_length'] = abs(
            candidate['wave_3'][1].price - candidate['wave_3'][0].price
        )
        measurements['wave_5_length'] = abs(
            candidate['wave_5'][1].price - candidate['wave_5'][0].price
        )
        
        # Retracement percentages
        measurements['wave_2_retracement'] = abs(
            (candidate['wave_2'][1].price - candidate['wave_2'][0].price) /
            (candidate['wave_1'][1].price - candidate['wave_1'][0].price)
        )
        measurements['wave_4_retracement'] = abs(
            (candidate['wave_4'][1].price - candidate['wave_4'][0].price) /
            (candidate['wave_3'][1].price - candidate['wave_3'][0].price)
        )
        
        return measurements
        
    def _calculate_confidence_score(self, candidate: Dict[str, Any], 
                                  measurements: Dict[str, float]) -> float:
        """
        Calculate confidence score for impulse wave candidate.
        
        Multi-factor scoring system.
        """
        score_components = []
        
        # Rule compliance scoring
        wave_3_length = measurements['wave_3_length']
        wave_1_length = measurements['wave_1_length']
        wave_5_length = measurements['wave_5_length']
        
        # Wave 3 not shortest rule (critical)
        if wave_3_length > max(wave_1_length, wave_5_length):
            score_components.append(0.3)  # 30% weight
        else:
            score_components.append(0.0)
            
        # Wave 2 retracement within bounds
        wave_2_ret = measurements['wave_2_retracement']
        if 0.382 <= wave_2_ret <= 0.786:
            score_components.append(0.2)  # 20% weight
        elif wave_2_ret < 1.0:  # At least doesn't exceed wave 1
            score_components.append(0.1)
        else:
            score_components.append(0.0)
            
        # Wave 4 retracement within bounds
        wave_4_ret = measurements['wave_4_retracement']
        if 0.236 <= wave_4_ret <= 0.618:
            score_components.append(0.2)  # 20% weight
        elif wave_4_ret < 1.0:
            score_components.append(0.1)
        else:
            score_components.append(0.0)
            
        # Fibonacci relationships
        fib_score = self._evaluate_fibonacci_relationships(measurements)
        score_components.append(fib_score * 0.2)  # 20% weight
        
        # Alternation between waves 2 and 4
        if abs(wave_2_ret - wave_4_ret) > 0.1:
            score_components.append(0.1)  # 10% weight
        else:
            score_components.append(0.05)
            
        return sum(score_components)
        
    def _evaluate_fibonacci_relationships(self, measurements: Dict[str, float]) -> float:
        """Evaluate Fibonacci relationships in wave measurements."""
        fib_levels = [0.382, 0.618, 1.0, 1.272, 1.618, 2.618]
        scores = []
        
        # Wave 3 to Wave 1 ratio
        ratio_3_1 = measurements['wave_3_length'] / measurements['wave_1_length']
        closest_fib = min(fib_levels, key=lambda x: abs(x - ratio_3_1))
        accuracy = 1 - abs(ratio_3_1 - closest_fib) / closest_fib
        if accuracy > (1 - self.fibonacci_tolerance):
            scores.append(0.5)
        else:
            scores.append(0.0)
            
        # Wave 5 to Wave 1 ratio
        ratio_5_1 = measurements['wave_5_length'] / measurements['wave_1_length']
        closest_fib = min(fib_levels, key=lambda x: abs(x - ratio_5_1))
        accuracy = 1 - abs(ratio_5_1 - closest_fib) / closest_fib
        if accuracy > (1 - self.fibonacci_tolerance):
            scores.append(0.5)
        else:
            scores.append(0.0)
            
        return sum(scores) / len(scores) if scores else 0.0
        
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detector performance statistics."""
        stats = self.detection_stats.copy()
        if stats["total_detections"] > 0:
            stats["accuracy"] = stats["valid_detections"] / stats["total_detections"]
        else:
            stats["accuracy"] = 0.0
            
        if stats["processing_times"]:
            stats["avg_processing_time_ms"] = np.mean(stats["processing_times"])
            stats["max_processing_time_ms"] = np.max(stats["processing_times"])
        else:
            stats["avg_processing_time_ms"] = 0.0
            stats["max_processing_time_ms"] = 0.0
            
        return stats


# Export main classes
__all__ = [
    'ImpulseWave',
    'ImpulseWaveDetector',
    'WaveType',
    'ImpulseDirection',
    'WavePoint'
]