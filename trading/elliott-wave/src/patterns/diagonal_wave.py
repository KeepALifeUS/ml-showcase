"""
Diagonal Wave Pattern Detection for Elliott Wave Analysis.

Leading and ending diagonal detection,
optimized for crypto market conditions and wedge structures.
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


class DiagonalType(str, Enum):
    """Types of diagonal patterns."""
    LEADING_DIAGONAL = "leading_diagonal"
    ENDING_DIAGONAL = "ending_diagonal"
    EXPANDING_DIAGONAL = "expanding_diagonal"
    CONTRACTING_DIAGONAL = "contracting_diagonal"


class DiagonalDirection(str, Enum):
    """Direction of diagonal wave."""
    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass
class DiagonalWave:
    """
    Complete diagonal wave structure.
    
    Rich domain model for diagonal patterns.
    """
    wave_1: Tuple[WavePoint, WavePoint]  # Start, End
    wave_2: Tuple[WavePoint, WavePoint]  # Start, End  
    wave_3: Tuple[WavePoint, WavePoint]  # Start, End
    wave_4: Tuple[WavePoint, WavePoint]  # Start, End
    wave_5: Tuple[WavePoint, WavePoint]  # Start, End
    
    diagonal_type: DiagonalType
    direction: DiagonalDirection
    confidence: float
    timeframe: str
    symbol: str
    
    # Wave measurements
    wave_lengths: Dict[str, float]
    retracements: Dict[str, float]
    
    # Diagonal characteristics
    convergence_angle: float
    channel_slope: float
    overlap_percentage: float
    
    # Fibonacci relationships
    fibonacci_ratios: Dict[str, float]
    
    # Validation results
    rules_validation: Dict[str, bool]
    guidelines_score: float
    
    # Performance metrics
    detected_at: datetime
    processing_time_ms: float
    
    def __post_init__(self):
        """Initialize diagonal wave analysis."""
        self.validate_structure()
        self.calculate_fibonacci_relationships()
        self.analyze_convergence()
        
    def validate_structure(self) -> bool:
        """
        Validate Elliott Wave diagonal rules.
        
        Business rule validation for diagonal patterns.
        """
        rules = {
            "wave_overlap": self._validate_wave_overlap(),
            "wave_convergence": self._validate_convergence(),
            "fibonacci_relationships": self._validate_fibonacci_relationships(),
            "internal_structure": self._validate_internal_structure(),
            "momentum_characteristics": self._validate_momentum()
        }
        
        self.rules_validation = rules
        return sum(rules.values()) >= 3  # At least 3 out of 5 rules
        
    def _validate_wave_overlap(self) -> bool:
        """Validate that waves 1 and 4 overlap (diagonal requirement)."""
        wave_1_start = self.wave_1[0].price
        wave_1_end = self.wave_1[1].price
        wave_4_start = self.wave_4[0].price
        wave_4_end = self.wave_4[1].price
        
        if self.direction == DiagonalDirection.BULLISH:
            # In bullish diagonal, wave 4 should overlap wave 1
            wave_1_high = max(wave_1_start, wave_1_end)
            wave_1_low = min(wave_1_start, wave_1_end)
            wave_4_low = min(wave_4_start, wave_4_end)
            return wave_4_low < wave_1_high and wave_4_low > wave_1_low
        else:
            # In bearish diagonal, wave 4 should overlap wave 1
            wave_1_high = max(wave_1_start, wave_1_end)
            wave_1_low = min(wave_1_start, wave_1_end)
            wave_4_high = max(wave_4_start, wave_4_end)
            return wave_4_high > wave_1_low and wave_4_high < wave_1_high
            
    def _validate_convergence(self) -> bool:
        """Validate that the diagonal shows convergence."""
        # Check if trendlines converge
        return abs(self.convergence_angle) > 0.1  # Some convergence required
        
    def _validate_fibonacci_relationships(self) -> bool:
        """Validate Fibonacci relationships in diagonal."""
        # Diagonals often have specific Fibonacci relationships
        return True  # Placeholder
        
    def _validate_internal_structure(self) -> bool:
        """Validate internal corrective structure of waves 2 and 4."""
        # Waves 2 and 4 should be corrective in nature
        return True  # Placeholder
        
    def _validate_momentum(self) -> bool:
        """Validate momentum characteristics of diagonal."""
        # Diagonals often show momentum divergence
        return True  # Placeholder
        
    def calculate_fibonacci_relationships(self) -> Dict[str, float]:
        """Calculate Fibonacci relationships between diagonal waves."""
        ratios = {}
        
        wave_1_length = self.wave_lengths['wave_1']
        wave_3_length = self.wave_lengths['wave_3']
        wave_5_length = self.wave_lengths['wave_5']
        
        # Wave relationships
        if wave_1_length > 0:
            ratios["wave_3_to_1"] = wave_3_length / wave_1_length
            ratios["wave_5_to_1"] = wave_5_length / wave_1_length
            
        if wave_3_length > 0:
            ratios["wave_5_to_3"] = wave_5_length / wave_3_length
            
        # Common Fibonacci levels for diagonals
        fib_levels = [0.618, 0.786, 1.0, 1.272]
        
        # Find Fibonacci matches
        for key, value in ratios.items():
            closest_fib = min(fib_levels, key=lambda x: abs(x - value))
            ratios[f"{key}_fib_match"] = closest_fib
            ratios[f"{key}_fib_accuracy"] = 1 - abs(value - closest_fib) / closest_fib
            
        self.fibonacci_ratios = ratios
        return ratios
        
    def analyze_convergence(self) -> Dict[str, float]:
        """Analyze convergence characteristics of diagonal."""
        # Calculate convergence angle between trendlines
        self.convergence_angle = self._calculate_convergence_angle()
        self.channel_slope = self._calculate_channel_slope()
        self.overlap_percentage = self._calculate_overlap_percentage()
        
        return {
            "convergence_angle": self.convergence_angle,
            "channel_slope": self.channel_slope,
            "overlap_percentage": self.overlap_percentage
        }
        
    def _calculate_convergence_angle(self) -> float:
        """Calculate angle of convergence between trendlines."""
        # Simplified calculation - needs proper geometric analysis
        return 15.0  # Placeholder
        
    def _calculate_channel_slope(self) -> float:
        """Calculate overall slope of diagonal channel."""
        start_price = self.wave_1[0].price
        end_price = self.wave_5[1].price
        start_time = self.wave_1[0].timestamp.timestamp()
        end_time = self.wave_5[1].timestamp.timestamp()
        
        if end_time - start_time > 0:
            return (end_price - start_price) / (end_time - start_time)
        return 0.0
        
    def _calculate_overlap_percentage(self) -> float:
        """Calculate overlap percentage between waves."""
        # Calculate how much wave 4 overlaps with wave 1
        return 0.3  # Placeholder
        
    def get_projection_targets(self) -> Dict[str, float]:
        """Get price projection targets after diagonal completion."""
        wave_5_end = self.wave_5[1].price
        
        if self.diagonal_type == DiagonalType.ENDING_DIAGONAL:
            # After ending diagonal, expect sharp reversal
            if self.direction == DiagonalDirection.BULLISH:
                projections = {
                    "reversal_target": wave_5_end * 0.618,  # 38.2% retracement
                    "extended_reversal": wave_5_end * 0.5,   # 50% retracement
                    "deep_reversal": wave_5_end * 0.382      # 61.8% retracement
                }
            else:
                projections = {
                    "reversal_target": wave_5_end * 1.618,
                    "extended_reversal": wave_5_end * 2.0,
                    "deep_reversal": wave_5_end * 2.618
                }
        else:
            # Leading diagonal - expect continuation
            if self.direction == DiagonalDirection.BULLISH:
                projections = {
                    "continuation_target": wave_5_end * 1.272,
                    "extended_target": wave_5_end * 1.618,
                    "maximum_target": wave_5_end * 2.0
                }
            else:
                projections = {
                    "continuation_target": wave_5_end * 0.786,
                    "extended_target": wave_5_end * 0.618,
                    "maximum_target": wave_5_end * 0.5
                }
                
        return projections
        
    @property
    def is_valid(self) -> bool:
        """Check if diagonal wave passes validation."""
        return sum(self.rules_validation.values()) >= 3 and self.confidence > 0.65
        
    @property
    def total_move(self) -> float:
        """Calculate total move of diagonal."""
        start_price = self.wave_1[0].price
        end_price = self.wave_5[1].price
        return abs(end_price - start_price)


class DiagonalWaveDetector:
    """
    Advanced Diagonal Wave Detection System.
    
    
    - Leading/ending diagonal recognition
    - Crypto market wedge pattern optimization
    - Real-time diagonal tracking
    """
    
    def __init__(self,
                 min_wave_length: int = 4,
                 max_wave_length: int = 100,
                 overlap_threshold: float = 0.1,
                 confidence_threshold: float = 0.65):
        """
        Initialize diagonal wave detector.
        
        Args:
            min_wave_length: Minimum points in diagonal wave
            max_wave_length: Maximum points in diagonal wave
            overlap_threshold: Minimum overlap for diagonal validation
            confidence_threshold: Minimum confidence for valid detection
        """
        self.min_wave_length = min_wave_length
        self.max_wave_length = max_wave_length
        self.overlap_threshold = overlap_threshold
        self.confidence_threshold = confidence_threshold
        
        # Detection statistics
        self.detection_stats = {
            "total_detections": 0,
            "valid_detections": 0,
            "by_type": {dt.value: 0 for dt in DiagonalType},
            "processing_times": []
        }
        
    @performance_monitor
    async def detect_diagonal_waves(self,
                                  data: pd.DataFrame,
                                  symbol: str,
                                  timeframe: str) -> List[DiagonalWave]:
        """
        Detect diagonal wave patterns in price data.
        
        Comprehensive diagonal pattern detection.
        
        Args:
            data: OHLCV price data
            symbol: Trading symbol
            timeframe: Timeframe string
            
        Returns:
            List[DiagonalWave]: Detected diagonal waves
        """
        start_time = datetime.utcnow()
        
        # Validate input data
        if not self._validate_data(data):
            logger.warning(f"Invalid data for {symbol} {timeframe}")
            return []
            
        # Find pivot points with tighter parameters for diagonals
        pivots = await self._find_pivot_points(data, window=3)
        if len(pivots) < 6:  # Need 6 points for 5-wave diagonal
            return []
            
        # Detect potential diagonal structures
        diagonal_candidates = await self._scan_for_diagonal_patterns(pivots, data)
        
        # Validate and score candidates
        valid_diagonals = []
        for candidate in diagonal_candidates:
            diagonal_wave = await self._validate_diagonal_candidate(
                candidate, symbol, timeframe, start_time
            )
            if diagonal_wave and diagonal_wave.confidence >= self.confidence_threshold:
                valid_diagonals.append(diagonal_wave)
                
        # Log detection results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="diagonal",
            confidence=max([w.confidence for w in valid_diagonals]) if valid_diagonals else 0,
            detected_count=len(valid_diagonals),
            candidates_scanned=len(diagonal_candidates)
        )
        
        return valid_diagonals
        
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for diagonal analysis."""
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            return False
            
        if len(data) < self.min_wave_length * 5:
            return False
            
        return True
        
    async def _find_pivot_points(self, data: pd.DataFrame, 
                               window: int = 3) -> List[WavePoint]:
        """Find pivot points optimized for diagonal detection."""
        highs = data['high'].values
        lows = data['low'].values
        timestamps = pd.to_datetime(data.index)
        volumes = data.get('volume', [None] * len(data)).values
        
        pivots = []
        
        # Find local highs
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
                
        pivots.sort(key=lambda x: x.index)
        return pivots
        
    async def _scan_for_diagonal_patterns(self, pivots: List[WavePoint],
                                        data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Scan for potential diagonal patterns."""
        candidates = []
        
        # Need at least 6 pivots for diagonal
        for start_idx in range(len(pivots) - 5):
            candidate_pivots = pivots[start_idx:start_idx + 6]
            
            # Check if sequence could form diagonal
            if self._is_potential_diagonal_sequence(candidate_pivots):
                candidate = self._build_diagonal_candidate(candidate_pivots, data)
                if candidate:
                    candidates.append(candidate)
                    
        return candidates
        
    def _is_potential_diagonal_sequence(self, pivots: List[WavePoint]) -> bool:
        """Check if pivot sequence could form diagonal pattern."""
        if len(pivots) != 6:
            return False
            
        # Check alternating pattern like impulse
        directions = []
        for i in range(len(pivots) - 1):
            if pivots[i + 1].price > pivots[i].price:
                directions.append(1)
            else:
                directions.append(-1)
                
        # Should follow impulse pattern but with overlaps
        bullish_pattern = [1, -1, 1, -1, 1]
        bearish_pattern = [-1, 1, -1, 1, -1]
        
        if directions in [bullish_pattern, bearish_pattern]:
            # Check for overlap between waves 1 and 4
            return self._check_wave_overlap(pivots, directions == bullish_pattern)
            
        return False
        
    def _check_wave_overlap(self, pivots: List[WavePoint], is_bullish: bool) -> bool:
        """Check if waves 1 and 4 overlap (diagonal requirement)."""
        wave_1_start = pivots[0].price
        wave_1_end = pivots[1].price
        wave_4_start = pivots[3].price
        wave_4_end = pivots[4].price
        
        if is_bullish:
            wave_1_range = (min(wave_1_start, wave_1_end), max(wave_1_start, wave_1_end))
            wave_4_low = min(wave_4_start, wave_4_end)
            return wave_1_range[0] < wave_4_low < wave_1_range[1]
        else:
            wave_1_range = (min(wave_1_start, wave_1_end), max(wave_1_start, wave_1_end))
            wave_4_high = max(wave_4_start, wave_4_end)
            return wave_1_range[0] < wave_4_high < wave_1_range[1]
            
    def _build_diagonal_candidate(self, pivots: List[WavePoint],
                                data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Build diagonal candidate from pivot points."""
        try:
            # Determine direction
            if pivots[1].price > pivots[0].price:
                direction = DiagonalDirection.BULLISH
            else:
                direction = DiagonalDirection.BEARISH
                
            # Determine diagonal type (simplified logic)
            # More sophisticated analysis needed for proper classification
            diagonal_type = DiagonalType.CONTRACTING_DIAGONAL
            
            # Calculate wave lengths
            wave_lengths = {
                'wave_1': abs(pivots[1].price - pivots[0].price),
                'wave_2': abs(pivots[2].price - pivots[1].price),
                'wave_3': abs(pivots[3].price - pivots[2].price),
                'wave_4': abs(pivots[4].price - pivots[3].price),
                'wave_5': abs(pivots[5].price - pivots[4].price)
            }
            
            # Calculate retracements
            retracements = {
                'wave_2_ret': wave_lengths['wave_2'] / wave_lengths['wave_1'] if wave_lengths['wave_1'] > 0 else 0,
                'wave_4_ret': wave_lengths['wave_4'] / wave_lengths['wave_3'] if wave_lengths['wave_3'] > 0 else 0
            }
            
            return {
                'wave_1': (pivots[0], pivots[1]),
                'wave_2': (pivots[1], pivots[2]),
                'wave_3': (pivots[2], pivots[3]),
                'wave_4': (pivots[3], pivots[4]),
                'wave_5': (pivots[4], pivots[5]),
                'diagonal_type': diagonal_type,
                'direction': direction,
                'wave_lengths': wave_lengths,
                'retracements': retracements,
                'pivots': pivots
            }
            
        except Exception as e:
            logger.error(f"Error building diagonal candidate: {e}")
            return None
            
    async def _validate_diagonal_candidate(self, candidate: Dict[str, Any],
                                         symbol: str, timeframe: str,
                                         start_time: datetime) -> Optional[DiagonalWave]:
        """Validate and score diagonal candidate."""
        try:
            # Calculate confidence score
            confidence = self._calculate_diagonal_confidence(candidate)
            
            if confidence < self.confidence_threshold:
                return None
                
            # Create DiagonalWave object
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            diagonal_wave = DiagonalWave(
                wave_1=candidate['wave_1'],
                wave_2=candidate['wave_2'],
                wave_3=candidate['wave_3'],
                wave_4=candidate['wave_4'],
                wave_5=candidate['wave_5'],
                diagonal_type=candidate['diagonal_type'],
                direction=candidate['direction'],
                confidence=confidence,
                timeframe=timeframe,
                symbol=symbol,
                wave_lengths=candidate['wave_lengths'],
                retracements=candidate['retracements'],
                convergence_angle=0.0,  # Will be calculated
                channel_slope=0.0,      # Will be calculated
                overlap_percentage=0.0,  # Will be calculated
                fibonacci_ratios={},    # Will be calculated in __post_init__
                rules_validation={},    # Will be calculated in __post_init__
                guidelines_score=confidence,
                detected_at=datetime.utcnow(),
                processing_time_ms=processing_time
            )
            
            return diagonal_wave
            
        except Exception as e:
            logger.error(f"Error validating diagonal candidate: {e}")
            return None
            
    def _calculate_diagonal_confidence(self, candidate: Dict[str, Any]) -> float:
        """Calculate confidence score for diagonal pattern."""
        score_components = []
        
        wave_lengths = candidate['wave_lengths']
        retracements = candidate['retracements']
        
        # Wave overlap validation (critical for diagonals)
        if self._validate_overlap_in_candidate(candidate):
            score_components.append(0.3)  # 30% for proper overlap
        else:
            score_components.append(0.0)
            
        # Wave length relationships
        wave_3 = wave_lengths['wave_3']
        wave_1 = wave_lengths['wave_1'] 
        wave_5 = wave_lengths['wave_5']
        
        # In diagonals, wave 3 can be shortest
        if wave_3 > 0:
            score_components.append(0.2)  # 20% for wave 3 existence
            
        # Retracement quality
        wave_2_ret = retracements['wave_2_ret']
        wave_4_ret = retracements['wave_4_ret']
        
        if 0.5 <= wave_2_ret <= 0.9:  # Deeper retracements in diagonals
            score_components.append(0.15)
        else:
            score_components.append(0.05)
            
        if 0.5 <= wave_4_ret <= 0.9:
            score_components.append(0.15)
        else:
            score_components.append(0.05)
            
        # Convergence characteristics
        score_components.append(0.1)  # Base convergence score
        
        # Pattern clarity
        score_components.append(0.1)  # Pattern structure score
        
        return sum(score_components)
        
    def _validate_overlap_in_candidate(self, candidate: Dict[str, Any]) -> bool:
        """Validate overlap requirement for diagonal candidate."""
        pivots = candidate['pivots']
        return self._check_wave_overlap(pivots, candidate['direction'] == DiagonalDirection.BULLISH)


# Export main classes
__all__ = [
    'DiagonalWave',
    'DiagonalWaveDetector',
    'DiagonalType',
    'DiagonalDirection'
]