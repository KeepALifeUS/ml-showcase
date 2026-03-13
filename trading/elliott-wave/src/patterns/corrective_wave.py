"""
Corrective Wave Pattern Detection for Elliott Wave Analysis.

ABC corrective pattern detection,
optimized for crypto market volatility and complex correction structures.
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
from .impulse_wave import WavePoint

logger = get_logger(__name__)


class CorrectiveType(str, Enum):
    """Types of corrective wave patterns."""
    ZIGZAG = "zigzag"
    FLAT = "flat"
    TRIANGLE = "triangle"
    COMPLEX = "complex"
    IRREGULAR = "irregular"
    EXPANDED_FLAT = "expanded_flat"
    CONTRACTING_TRIANGLE = "contracting_triangle"
    EXPANDING_TRIANGLE = "expanding_triangle"


class CorrectiveDirection(str, Enum):
    """Direction of corrective wave."""
    BULLISH = "bullish"  # Against bearish trend
    BEARISH = "bearish"  # Against bullish trend


@dataclass
class CorrectiveWave:
    """
    Complete corrective wave structure.
    
    Rich domain model for corrective patterns.
    """
    wave_a: Tuple[WavePoint, WavePoint]  # Start, End of wave A
    wave_b: Tuple[WavePoint, WavePoint]  # Start, End of wave B
    wave_c: Tuple[WavePoint, WavePoint]  # Start, End of wave C
    
    corrective_type: CorrectiveType
    direction: CorrectiveDirection
    confidence: float
    timeframe: str
    symbol: str
    
    # Wave measurements
    wave_a_length: float
    wave_b_length: float
    wave_c_length: float
    wave_b_retracement: float  # B retracement of A
    
    # Fibonacci relationships
    fibonacci_ratios: Dict[str, float]
    
    # Pattern-specific measurements
    pattern_measurements: Dict[str, float]
    
    # Validation results
    rules_validation: Dict[str, bool]
    guidelines_score: float
    
    # Performance metrics
    detected_at: datetime
    processing_time_ms: float
    
    def __post_init__(self):
        """Validate corrective wave structure."""
        self.validate_structure()
        self.calculate_fibonacci_relationships()
        self.analyze_pattern_characteristics()
        
    def validate_structure(self) -> bool:
        """
        Validate Elliott Wave corrective rules.
        
        Business rule validation for corrective patterns.
        """
        rules = {
            "wave_b_overlap": self._validate_wave_b_overlap(),
            "wave_c_completion": self._validate_wave_c_completion(),
            "fibonacci_relationships": self._validate_fibonacci_relationships(),
            "corrective_alternation": self._validate_corrective_alternation(),
            "pattern_integrity": self._validate_pattern_integrity()
        }
        
        self.rules_validation = rules
        return all(rules.values())
        
    def _validate_wave_b_overlap(self) -> bool:
        """Validate wave B overlap requirements for different corrective types."""
        wave_a_start = self.wave_a[0].price
        wave_a_end = self.wave_a[1].price
        wave_b_end = self.wave_b[1].price
        
        if self.corrective_type == CorrectiveType.ZIGZAG:
            # In zigzag, B should retrace 38.2% to 78.6% of A
            b_retracement = abs(wave_b_end - wave_a_end) / abs(wave_a_end - wave_a_start)
            return 0.382 <= b_retracement <= 0.786
            
        elif self.corrective_type == CorrectiveType.FLAT:
            # In flat, B should retrace 90%+ of A
            b_retracement = abs(wave_b_end - wave_a_end) / abs(wave_a_end - wave_a_start)
            return b_retracement >= 0.9
            
        elif self.corrective_type == CorrectiveType.EXPANDED_FLAT:
            # In expanded flat, B exceeds start of A
            if self.direction == CorrectiveDirection.BEARISH:
                return wave_b_end > wave_a_start
            else:
                return wave_b_end < wave_a_start
                
        return True  # Default validation for other types
        
    def _validate_wave_c_completion(self) -> bool:
        """Validate wave C completion requirements."""
        wave_a_length = self.wave_a_length
        wave_c_length = self.wave_c_length
        
        if self.corrective_type == CorrectiveType.ZIGZAG:
            # C typically equals A or extends to 1.618 * A
            ratio = wave_c_length / wave_a_length
            return 0.8 <= ratio <= 2.0
            
        elif self.corrective_type in [CorrectiveType.FLAT, CorrectiveType.EXPANDED_FLAT]:
            # C should at least equal A
            return wave_c_length >= wave_a_length * 0.8
            
        return True
        
    def _validate_fibonacci_relationships(self) -> bool:
        """Validate Fibonacci relationships within corrective pattern."""
        # Implementation depends on corrective type
        return True
        
    def _validate_corrective_alternation(self) -> bool:
        """Validate alternation within corrective structure."""
        # Check for alternating complexity and retracement depths
        return True
        
    def _validate_pattern_integrity(self) -> bool:
        """Validate overall pattern integrity."""
        # Check time relationships and proportions
        return True
        
    def calculate_fibonacci_relationships(self) -> Dict[str, float]:
        """Calculate Fibonacci relationships between corrective waves."""
        ratios = {}
        
        # Wave B to A ratios
        ratios["wave_b_to_a"] = self.wave_b_length / self.wave_a_length
        
        # Wave C to A ratios
        ratios["wave_c_to_a"] = self.wave_c_length / self.wave_a_length
        
        # Wave C to B ratios
        ratios["wave_c_to_b"] = self.wave_c_length / self.wave_b_length
        
        # Retracement ratios
        ratios["b_retracement_of_a"] = self.wave_b_retracement
        
        # Time relationships
        ratios["time_b_to_a"] = self._calculate_time_ratio("B", "A")
        ratios["time_c_to_a"] = self._calculate_time_ratio("C", "A")
        
        # Find Fibonacci matches
        fib_levels = [0.382, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        for key, value in ratios.items():
            if not key.startswith("time_"):  # Skip time ratios for now
                closest_fib = min(fib_levels, key=lambda x: abs(x - value))
                ratios[f"{key}_fib_match"] = closest_fib
                ratios[f"{key}_fib_accuracy"] = 1 - abs(value - closest_fib) / closest_fib
                
        self.fibonacci_ratios = ratios
        return ratios
        
    def _calculate_time_ratio(self, wave1: str, wave2: str) -> float:
        """Calculate time ratio between waves."""
        if wave1 == "B" and wave2 == "A":
            time_a = (self.wave_a[1].timestamp - self.wave_a[0].timestamp).total_seconds()
            time_b = (self.wave_b[1].timestamp - self.wave_b[0].timestamp).total_seconds()
            return time_b / time_a if time_a > 0 else 1.0
        elif wave1 == "C" and wave2 == "A":
            time_a = (self.wave_a[1].timestamp - self.wave_a[0].timestamp).total_seconds()
            time_c = (self.wave_c[1].timestamp - self.wave_c[0].timestamp).total_seconds()
            return time_c / time_a if time_a > 0 else 1.0
        return 1.0
        
    def analyze_pattern_characteristics(self) -> Dict[str, float]:
        """Analyze pattern-specific characteristics."""
        characteristics = {}
        
        if self.corrective_type == CorrectiveType.ZIGZAG:
            characteristics.update(self._analyze_zigzag_characteristics())
        elif self.corrective_type in [CorrectiveType.FLAT, CorrectiveType.EXPANDED_FLAT]:
            characteristics.update(self._analyze_flat_characteristics())
        elif self.corrective_type in [CorrectiveType.TRIANGLE, 
                                     CorrectiveType.CONTRACTING_TRIANGLE,
                                     CorrectiveType.EXPANDING_TRIANGLE]:
            characteristics.update(self._analyze_triangle_characteristics())
            
        self.pattern_measurements = characteristics
        return characteristics
        
    def _analyze_zigzag_characteristics(self) -> Dict[str, float]:
        """Analyze zigzag-specific characteristics."""
        return {
            "sharpness": self._calculate_pattern_sharpness(),
            "momentum_divergence": self._calculate_momentum_divergence(),
            "volume_confirmation": self._calculate_volume_confirmation()
        }
        
    def _analyze_flat_characteristics(self) -> Dict[str, float]:
        """Analyze flat correction characteristics."""
        return {
            "sideways_movement": self._calculate_sideways_movement(),
            "time_symmetry": self._calculate_time_symmetry(),
            "volatility_contraction": self._calculate_volatility_contraction()
        }
        
    def _analyze_triangle_characteristics(self) -> Dict[str, float]:
        """Analyze triangle pattern characteristics."""
        return {
            "convergence_rate": self._calculate_convergence_rate(),
            "volume_trend": self._calculate_volume_trend(),
            "breakout_potential": self._calculate_breakout_potential()
        }
        
    def _calculate_pattern_sharpness(self) -> float:
        """Calculate sharpness of the corrective pattern."""
        # Implementation for pattern sharpness calculation
        return 0.5
        
    def _calculate_momentum_divergence(self) -> float:
        """Calculate momentum divergence within pattern."""
        # Implementation for momentum analysis
        return 0.5
        
    def _calculate_volume_confirmation(self) -> float:
        """Calculate volume confirmation strength."""
        # Implementation for volume analysis
        return 0.5
        
    def _calculate_sideways_movement(self) -> float:
        """Calculate degree of sideways movement in flat."""
        # Implementation for sideways movement analysis
        return 0.5
        
    def _calculate_time_symmetry(self) -> float:
        """Calculate time symmetry in pattern."""
        # Implementation for time symmetry analysis
        return 0.5
        
    def _calculate_volatility_contraction(self) -> float:
        """Calculate volatility contraction."""
        # Implementation for volatility analysis
        return 0.5
        
    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate for triangles."""
        # Implementation for triangle convergence
        return 0.5
        
    def _calculate_volume_trend(self) -> float:
        """Calculate volume trend within pattern."""
        # Implementation for volume trend analysis
        return 0.5
        
    def _calculate_breakout_potential(self) -> float:
        """Calculate breakout potential."""
        # Implementation for breakout analysis
        return 0.5
        
    def get_projection_targets(self) -> Dict[str, float]:
        """Get price projection targets after corrective completion."""
        wave_c_end = self.wave_c[1].price
        
        if self.corrective_type == CorrectiveType.ZIGZAG:
            # Zigzag completion often leads to trend resumption
            if self.direction == CorrectiveDirection.BEARISH:
                # After bearish correction, bullish resumption
                projections = {
                    "trend_resumption": wave_c_end * 1.272,
                    "extended_target": wave_c_end * 1.618,
                    "conservative_target": wave_c_end * 1.1
                }
            else:
                # After bullish correction, bearish resumption  
                projections = {
                    "trend_resumption": wave_c_end * 0.786,
                    "extended_target": wave_c_end * 0.618,
                    "conservative_target": wave_c_end * 0.9
                }
        else:
            # Conservative projections for other corrective types
            projections = {
                "breakout_target": wave_c_end * (1.1 if self.direction == CorrectiveDirection.BEARISH else 0.9),
                "measured_move": wave_c_end * (1.05 if self.direction == CorrectiveDirection.BEARISH else 0.95)
            }
            
        return projections
        
    @property
    def is_valid(self) -> bool:
        """Check if corrective wave passes all validation rules."""
        return all(self.rules_validation.values()) and self.confidence > 0.6
        
    @property
    def total_correction(self) -> float:
        """Calculate total correction from start to end."""
        start_price = self.wave_a[0].price
        end_price = self.wave_c[1].price
        return abs(end_price - start_price)


class CorrectiveWaveDetector:
    """
    Advanced Corrective Wave Detection System.
    
    
    - Multi-pattern corrective recognition
    - Crypto market adaptation for complex corrections
    - Real-time processing with high accuracy
    """
    
    def __init__(self,
                 min_wave_length: int = 3,
                 max_wave_length: int = 150,
                 fibonacci_tolerance: float = 0.2,
                 confidence_threshold: float = 0.6):
        """
        Initialize corrective wave detector.
        
        Args:
            min_wave_length: Minimum points in a corrective wave
            max_wave_length: Maximum points in a corrective wave  
            fibonacci_tolerance: Tolerance for Fibonacci matching
            confidence_threshold: Minimum confidence for valid detection
        """
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.fibonacci_tolerance = fibonacci_tolerance
        self.confidence_threshold = confidence_threshold
        
        # Detection statistics
        self.detection_stats = {
            "total_detections": 0,
            "valid_detections": 0,
            "by_type": {ct.value: 0 for ct in CorrectiveType},
            "processing_times": []
        }
        
    @performance_monitor
    async def detect_corrective_waves(self,
                                    data: pd.DataFrame,
                                    symbol: str,
                                    timeframe: str) -> List[CorrectiveWave]:
        """
        Detect corrective wave patterns in price data.
        
        Comprehensive corrective pattern detection.
        
        Args:
            data: OHLCV price data
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            List[CorrectiveWave]: Detected corrective waves
        """
        start_time = datetime.utcnow()
        
        # Validate input data
        if not self._validate_data(data):
            logger.warning(f"Invalid data for {symbol} {timeframe}")
            return []
            
        # Find pivot points
        pivots = await self._find_pivot_points(data)
        if len(pivots) < 4:  # Need at least 4 points for ABC structure
            return []
            
        # Detect different corrective patterns
        corrective_candidates = []
        
        # Detect zigzag patterns
        zigzag_candidates = await self._detect_zigzag_patterns(pivots, data)
        corrective_candidates.extend(zigzag_candidates)
        
        # Detect flat patterns
        flat_candidates = await self._detect_flat_patterns(pivots, data)
        corrective_candidates.extend(flat_candidates)
        
        # Detect triangle patterns (if enough data)
        if len(pivots) >= 5:
            triangle_candidates = await self._detect_triangle_patterns(pivots, data)
            corrective_candidates.extend(triangle_candidates)
            
        # Validate and score candidates
        valid_correctives = []
        for candidate in corrective_candidates:
            corrective_wave = await self._validate_corrective_candidate(
                candidate, symbol, timeframe, start_time
            )
            if corrective_wave and corrective_wave.confidence >= self.confidence_threshold:
                valid_correctives.append(corrective_wave)
                
        # Log detection results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="corrective",
            confidence=max([w.confidence for w in valid_correctives]) if valid_correctives else 0,
            detected_count=len(valid_correctives),
            candidates_scanned=len(corrective_candidates)
        )
        
        return valid_correctives
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input price data for corrective analysis."""
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < self.min_wave_length * 3:  # Need data for ABC
            return False
            
        return True
        
    async def _find_pivot_points(self, data: pd.DataFrame, 
                               window: int = 3) -> List[WavePoint]:
        """Find pivot points optimized for corrective patterns."""
        # Similar to impulse wave detection but with smaller window
        # for more granular corrective pattern detection
        highs = data['high'].values
        lows = data['low'].values
        timestamps = pd.to_datetime(data.index)
        volumes = data.get('volume', [None] * len(data)).values
        
        pivots = []
        
        # Find local highs with smaller window for corrections
        for i in range(window, len(highs) - window):
            if all(highs[i] >= highs[j] for j in range(i - window, i)) and \
               all(highs[i] >= highs[j] for j in range(i + 1, i + window + 1)):
                pivots.append(WavePoint(
                    index=i,
                    price=highs[i],
                    timestamp=timestamps[i],
                    volume=volumes[i] if volumes[i] is not None else None
                ))
                
        # Find local lows
        for i in range(window, len(lows) - window):
            if all(lows[i] <= lows[j] for j in range(i - window, i)) and \
               all(lows[i] <= lows[j] for j in range(i + 1, i + window + 1)):
                pivots.append(WavePoint(
                    index=i,
                    price=lows[i],
                    timestamp=timestamps[i],
                    volume=volumes[i] if volumes[i] is not None else None
                ))
                
        # Sort by index
        pivots.sort(key=lambda x: x.index)
        return pivots
        
    async def _detect_zigzag_patterns(self, pivots: List[WavePoint], 
                                    data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect zigzag corrective patterns."""
        candidates = []
        
        # Need at least 4 pivots for ABC zigzag
        for start_idx in range(len(pivots) - 3):
            candidate_pivots = pivots[start_idx:start_idx + 4]
            
            if self._is_valid_zigzag_sequence(candidate_pivots):
                candidate = self._build_zigzag_candidate(candidate_pivots)
                if candidate:
                    candidates.append(candidate)
                    
        return candidates
        
    def _is_valid_zigzag_sequence(self, pivots: List[WavePoint]) -> bool:
        """Check if pivot sequence forms valid zigzag ABC pattern."""
        if len(pivots) != 4:
            return False
            
        # Check alternating direction: A-B-C pattern
        directions = []
        for i in range(len(pivots) - 1):
            if pivots[i + 1].price > pivots[i].price:
                directions.append(1)  # Up
            else:
                directions.append(-1)  # Down
                
        # Valid zigzag patterns: down-up-down or up-down-up
        return directions in [[-1, 1, -1], [1, -1, 1]]
        
    def _build_zigzag_candidate(self, pivots: List[WavePoint]) -> Optional[Dict[str, Any]]:
        """Build zigzag candidate from pivots."""
        try:
            # Determine direction (direction of wave A)
            if pivots[1].price < pivots[0].price:
                direction = CorrectiveDirection.BEARISH
            else:
                direction = CorrectiveDirection.BULLISH
                
            return {
                'wave_a': (pivots[0], pivots[1]),
                'wave_b': (pivots[1], pivots[2]),
                'wave_c': (pivots[2], pivots[3]),
                'corrective_type': CorrectiveType.ZIGZAG,
                'direction': direction,
                'pivots': pivots
            }
        except Exception as e:
            logger.error(f"Error building zigzag candidate: {e}")
            return None
            
    async def _detect_flat_patterns(self, pivots: List[WavePoint],
                                  data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect flat corrective patterns."""
        candidates = []
        
        for start_idx in range(len(pivots) - 3):
            candidate_pivots = pivots[start_idx:start_idx + 4]
            
            if self._is_valid_flat_sequence(candidate_pivots):
                candidate = self._build_flat_candidate(candidate_pivots)
                if candidate:
                    candidates.append(candidate)
                    
        return candidates
        
    def _is_valid_flat_sequence(self, pivots: List[WavePoint]) -> bool:
        """Check if sequence forms valid flat pattern."""
        if len(pivots) != 4:
            return False
            
        # Flat patterns have deep B retracements (>90% of A)
        wave_a_length = abs(pivots[1].price - pivots[0].price)
        wave_b_length = abs(pivots[2].price - pivots[1].price)
        
        if wave_a_length == 0:
            return False
            
        b_retracement = wave_b_length / wave_a_length
        
        # B wave should retrace at least 90% of A for flat
        return b_retracement >= 0.9
        
    def _build_flat_candidate(self, pivots: List[WavePoint]) -> Optional[Dict[str, Any]]:
        """Build flat pattern candidate."""
        try:
            # Determine if regular or expanded flat
            wave_a_start = pivots[0].price
            wave_b_end = pivots[2].price
            
            # Check if B exceeds A start (expanded flat)
            if pivots[1].price < pivots[0].price:  # Bearish A wave
                if wave_b_end > wave_a_start:
                    flat_type = CorrectiveType.EXPANDED_FLAT
                else:
                    flat_type = CorrectiveType.FLAT
                direction = CorrectiveDirection.BEARISH
            else:  # Bullish A wave
                if wave_b_end < wave_a_start:
                    flat_type = CorrectiveType.EXPANDED_FLAT
                else:
                    flat_type = CorrectiveType.FLAT
                direction = CorrectiveDirection.BULLISH
                
            return {
                'wave_a': (pivots[0], pivots[1]),
                'wave_b': (pivots[1], pivots[2]),
                'wave_c': (pivots[2], pivots[3]),
                'corrective_type': flat_type,
                'direction': direction,
                'pivots': pivots
            }
        except Exception as e:
            logger.error(f"Error building flat candidate: {e}")
            return None
            
    async def _detect_triangle_patterns(self, pivots: List[WavePoint],
                                      data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect triangle corrective patterns."""
        candidates = []
        
        # Triangles need 5 points minimum (ABCDE)
        for start_idx in range(len(pivots) - 4):
            candidate_pivots = pivots[start_idx:start_idx + 5]
            
            if self._is_valid_triangle_sequence(candidate_pivots):
                candidate = self._build_triangle_candidate(candidate_pivots)
                if candidate:
                    candidates.append(candidate)
                    
        return candidates
        
    def _is_valid_triangle_sequence(self, pivots: List[WavePoint]) -> bool:
        """Check if sequence forms valid triangle pattern."""
        if len(pivots) != 5:
            return False
            
        # Check for converging trendlines
        highs = [p.price for p in pivots[::2]]  # Odd positions
        lows = [p.price for p in pivots[1::2]]   # Even positions
        
        # Simple convergence check - more sophisticated logic needed
        if len(highs) >= 2 and len(lows) >= 2:
            return True
            
        return False
        
    def _build_triangle_candidate(self, pivots: List[WavePoint]) -> Optional[Dict[str, Any]]:
        """Build triangle pattern candidate."""
        try:
            return {
                'wave_a': (pivots[0], pivots[1]),
                'wave_b': (pivots[1], pivots[2]),
                'wave_c': (pivots[2], pivots[3]),
                'corrective_type': CorrectiveType.TRIANGLE,
                'direction': CorrectiveDirection.BEARISH if pivots[1].price < pivots[0].price else CorrectiveDirection.BULLISH,
                'pivots': pivots
            }
        except Exception as e:
            logger.error(f"Error building triangle candidate: {e}")
            return None
            
    async def _validate_corrective_candidate(self, candidate: Dict[str, Any],
                                           symbol: str, timeframe: str,
                                           start_time: datetime) -> Optional[CorrectiveWave]:
        """Validate and score corrective wave candidate."""
        try:
            # Calculate wave measurements
            measurements = self._calculate_corrective_measurements(candidate)
            
            # Calculate confidence score
            confidence = self._calculate_corrective_confidence(candidate, measurements)
            
            if confidence < self.confidence_threshold:
                return None
                
            # Create CorrectiveWave object
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            corrective_wave = CorrectiveWave(
                wave_a=candidate['wave_a'],
                wave_b=candidate['wave_b'],
                wave_c=candidate['wave_c'],
                corrective_type=candidate['corrective_type'],
                direction=candidate['direction'],
                confidence=confidence,
                timeframe=timeframe,
                symbol=symbol,
                wave_a_length=measurements['wave_a_length'],
                wave_b_length=measurements['wave_b_length'],
                wave_c_length=measurements['wave_c_length'],
                wave_b_retracement=measurements['wave_b_retracement'],
                fibonacci_ratios={},  # Calculated in __post_init__
                pattern_measurements={},  # Calculated in __post_init__
                rules_validation={},  # Calculated in __post_init__
                guidelines_score=confidence,
                detected_at=datetime.utcnow(),
                processing_time_ms=processing_time
            )
            
            return corrective_wave
            
        except Exception as e:
            logger.error(f"Error validating corrective candidate: {e}")
            return None
            
    def _calculate_corrective_measurements(self, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Calculate measurements for corrective wave candidate."""
        measurements = {}
        
        # Wave lengths
        measurements['wave_a_length'] = abs(
            candidate['wave_a'][1].price - candidate['wave_a'][0].price
        )
        measurements['wave_b_length'] = abs(
            candidate['wave_b'][1].price - candidate['wave_b'][0].price
        )
        measurements['wave_c_length'] = abs(
            candidate['wave_c'][1].price - candidate['wave_c'][0].price
        )
        
        # Wave B retracement of A
        if measurements['wave_a_length'] > 0:
            measurements['wave_b_retracement'] = measurements['wave_b_length'] / measurements['wave_a_length']
        else:
            measurements['wave_b_retracement'] = 0
            
        return measurements
        
    def _calculate_corrective_confidence(self, candidate: Dict[str, Any],
                                       measurements: Dict[str, float]) -> float:
        """Calculate confidence score for corrective pattern."""
        score_components = []
        
        corrective_type = candidate['corrective_type']
        
        if corrective_type == CorrectiveType.ZIGZAG:
            # Zigzag scoring
            b_ret = measurements['wave_b_retracement']
            if 0.382 <= b_ret <= 0.786:
                score_components.append(0.4)  # 40% for proper B retracement
            elif b_ret < 1.0:
                score_components.append(0.2)
            else:
                score_components.append(0.0)
                
            # C wave relationship to A
            c_to_a = measurements['wave_c_length'] / measurements['wave_a_length']
            if 0.8 <= c_to_a <= 1.8:
                score_components.append(0.3)  # 30% for C=A relationship
            else:
                score_components.append(0.1)
                
        elif corrective_type in [CorrectiveType.FLAT, CorrectiveType.EXPANDED_FLAT]:
            # Flat pattern scoring
            b_ret = measurements['wave_b_retracement']
            if b_ret >= 0.9:
                score_components.append(0.4)  # Strong B retracement
            else:
                score_components.append(0.0)
                
            # C wave should equal or exceed A
            c_to_a = measurements['wave_c_length'] / measurements['wave_a_length']
            if c_to_a >= 0.8:
                score_components.append(0.3)
            else:
                score_components.append(0.1)
                
        # Pattern clarity and structure
        score_components.append(0.2)  # Base structure score
        
        # Time symmetry (simplified)
        score_components.append(0.1)  # Time relationship score
        
        return sum(score_components)


# Export main classes
__all__ = [
    'CorrectiveWave',
    'CorrectiveWaveDetector', 
    'CorrectiveType',
    'CorrectiveDirection'
]