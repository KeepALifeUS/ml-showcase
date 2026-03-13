"""
Zigzag Wave Pattern Detection for Elliott Wave Analysis.

Specialized zigzag corrective pattern detection,
optimized for crypto market conditions and ABC zigzag structures.
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
from .corrective_wave import CorrectiveWave, CorrectiveType, CorrectiveDirection

logger = get_logger(__name__)


class ZigzagType(str, Enum):
    """Types of zigzag patterns."""
    SINGLE_ZIGZAG = "single_zigzag"
    DOUBLE_ZIGZAG = "double_zigzag"
    TRIPLE_ZIGZAG = "triple_zigzag"
    ELONGATED_ZIGZAG = "elongated_zigzag"


class ZigzagCharacteristic(str, Enum):
    """Zigzag pattern characteristics."""
    SHARP_A_WAVE = "sharp_a_wave"
    SHALLOW_B_WAVE = "shallow_b_wave"
    EXTENDING_C_WAVE = "extending_c_wave"
    MOMENTUM_DIVERGENCE = "momentum_divergence"
    VOLUME_CONFIRMATION = "volume_confirmation"


@dataclass
class ZigzagWave(CorrectiveWave):
    """
    Specialized Zigzag wave structure.
    
    Rich domain model for zigzag-specific analysis.
    """
    zigzag_type: ZigzagType
    characteristics: List[ZigzagCharacteristic]
    
    # Zigzag-specific measurements
    a_wave_steepness: float = 0.0
    b_wave_shallowness: float = 0.0
    c_wave_extension: float = 0.0
    
    # Market behavior
    momentum_divergence: float = 0.0
    volume_spike_at_c: bool = False
    breakout_confirmation: bool = False
    
    def __post_init__(self):
        """Initialize zigzag analysis."""
        super().__post_init__()
        self.analyze_zigzag_characteristics()
        self.validate_zigzag_rules()
        
    def analyze_zigzag_characteristics(self) -> None:
        """Analyze zigzag-specific characteristics."""
        self.characteristics = []
        
        # Analyze A wave steepness
        self.a_wave_steepness = self._calculate_wave_steepness('A')
        if self.a_wave_steepness > 0.7:
            self.characteristics.append(ZigzagCharacteristic.SHARP_A_WAVE)
            
        # Analyze B wave shallowness
        self.b_wave_shallowness = 1.0 - self.wave_b_retracement
        if self.wave_b_retracement < 0.618:  # Less than 61.8% retracement
            self.characteristics.append(ZigzagCharacteristic.SHALLOW_B_WAVE)
            
        # Analyze C wave extension
        c_to_a_ratio = self.wave_c_length / self.wave_a_length
        self.c_wave_extension = max(0, c_to_a_ratio - 1.0)
        if c_to_a_ratio > 1.272:
            self.characteristics.append(ZigzagCharacteristic.EXTENDING_C_WAVE)
            
        # Check momentum divergence
        self.momentum_divergence = self._calculate_momentum_divergence()
        if self.momentum_divergence > 0.3:
            self.characteristics.append(ZigzagCharacteristic.MOMENTUM_DIVERGENCE)
            
        # Check volume confirmation
        if self._check_volume_confirmation():
            self.characteristics.append(ZigzagCharacteristic.VOLUME_CONFIRMATION)
            
    def _calculate_wave_steepness(self, wave: str) -> float:
        """Calculate steepness of specific wave."""
        if wave == 'A':
            start_point = self.wave_a[0]
            end_point = self.wave_a[1]
        elif wave == 'B':
            start_point = self.wave_b[0]
            end_point = self.wave_b[1]
        elif wave == 'C':
            start_point = self.wave_c[0]
            end_point = self.wave_c[1]
        else:
            return 0.0
            
        # Calculate price change rate over time
        price_change = abs(end_point.price - start_point.price)
        time_duration = (end_point.timestamp - start_point.timestamp).total_seconds() / 3600  # hours
        
        if time_duration > 0:
            steepness = price_change / (start_point.price * time_duration)
            return min(steepness * 100, 1.0)  # Normalize to 0-1
        return 0.0
        
    def _calculate_momentum_divergence(self) -> float:
        """Calculate momentum divergence between waves A and C."""
        # Simplified momentum divergence calculation
        # In practice, this would use RSI, MACD, or similar indicators
        a_steepness = self.a_wave_steepness
        c_steepness = self._calculate_wave_steepness('C')
        
        if a_steepness > 0:
            divergence = abs(a_steepness - c_steepness) / a_steepness
            return min(divergence, 1.0)
        return 0.0
        
    def _check_volume_confirmation(self) -> bool:
        """Check for volume confirmation at wave endpoints."""
        # Placeholder - would analyze actual volume data
        # Look for volume spikes at wave C completion
        return False  # Simplified
        
    def validate_zigzag_rules(self) -> None:
        """Validate zigzag-specific Elliott Wave rules."""
        additional_rules = {
            "a_wave_impulse": self._validate_a_wave_impulse(),
            "b_wave_corrective": self._validate_b_wave_corrective(), 
            "c_wave_impulse": self._validate_c_wave_impulse(),
            "fibonacci_relationships": self._validate_zigzag_fibonacci(),
            "alternation": self._validate_abc_alternation()
        }
        
        self.rules_validation.update(additional_rules)
        
    def _validate_a_wave_impulse(self) -> bool:
        """Validate that wave A has impulsive characteristics."""
        # A wave should be sharp and decisive
        return self.a_wave_steepness > 0.4
        
    def _validate_b_wave_corrective(self) -> bool:
        """Validate that wave B is corrective and shallow."""
        # B wave should be corrective (not exceed 78.6% of A)
        return self.wave_b_retracement <= 0.786
        
    def _validate_c_wave_impulse(self) -> bool:
        """Validate that wave C has impulsive characteristics."""
        # C wave should at least equal A or extend beyond
        return self.wave_c_length >= self.wave_a_length * 0.8
        
    def _validate_zigzag_fibonacci(self) -> bool:
        """Validate Fibonacci relationships in zigzag."""
        # Common zigzag relationships
        c_to_a = self.wave_c_length / self.wave_a_length
        common_ratios = [1.0, 1.272, 1.618, 2.618]
        
        # Check if C-to-A ratio is close to common Fibonacci level
        for ratio in common_ratios:
            if abs(c_to_a - ratio) / ratio < 0.15:  # Within 15%
                return True
        return False
        
    def _validate_abc_alternation(self) -> bool:
        """Validate alternation between waves A and C vs wave B."""
        # A and C should be impulsive, B should be corrective
        # Simplified - would check internal structure
        return self.a_wave_steepness > 0.3 and self.b_wave_shallowness > 0.2
        
    def get_zigzag_targets(self) -> Dict[str, float]:
        """Get zigzag-specific price targets."""
        targets = {}
        
        # Standard zigzag projections
        a_wave_length = self.wave_a_length
        c_wave_end = self.wave_c[1].price
        
        # Target based on A wave projection
        if self.direction == CorrectiveDirection.BEARISH:
            targets["c_equals_a"] = self.wave_a[0].price - a_wave_length
            targets["c_127_of_a"] = self.wave_a[0].price - (a_wave_length * 1.272)
            targets["c_162_of_a"] = self.wave_a[0].price - (a_wave_length * 1.618)
        else:
            targets["c_equals_a"] = self.wave_a[0].price + a_wave_length
            targets["c_127_of_a"] = self.wave_a[0].price + (a_wave_length * 1.272)
            targets["c_162_of_a"] = self.wave_a[0].price + (a_wave_length * 1.618)
            
        # Post-zigzag reversal targets
        if self.is_complete:
            reversal_targets = self.get_reversal_targets()
            targets.update(reversal_targets)
            
        return targets
        
    def get_reversal_targets(self) -> Dict[str, float]:
        """Get reversal targets after zigzag completion."""
        targets = {}
        zigzag_range = abs(self.wave_c[1].price - self.wave_a[0].price)
        
        if self.direction == CorrectiveDirection.BEARISH:
            # After bearish zigzag, expect bullish reversal
            targets["reversal_38"] = self.wave_c[1].price + (zigzag_range * 0.382)
            targets["reversal_62"] = self.wave_c[1].price + (zigzag_range * 0.618)
            targets["reversal_100"] = self.wave_c[1].price + zigzag_range
        else:
            # After bullish zigzag, expect bearish reversal
            targets["reversal_38"] = self.wave_c[1].price - (zigzag_range * 0.382)
            targets["reversal_62"] = self.wave_c[1].price - (zigzag_range * 0.618)
            targets["reversal_100"] = self.wave_c[1].price - zigzag_range
            
        return targets
        
    @property
    def is_complete(self) -> bool:
        """Check if zigzag pattern is complete."""
        return (len(self.characteristics) >= 2 and 
                self.confidence > 0.6 and
                self.wave_c_length > 0)
                
    @property
    def strength_score(self) -> float:
        """Calculate overall strength score of zigzag."""
        strength_factors = []
        
        # Characteristic strength
        characteristic_score = len(self.characteristics) / 5.0  # Max 5 characteristics
        strength_factors.append(characteristic_score)
        
        # Fibonacci accuracy
        fib_accuracy = sum(
            v for k, v in self.fibonacci_ratios.items() 
            if k.endswith('_fib_accuracy')
        ) / max(1, len([k for k in self.fibonacci_ratios.keys() if k.endswith('_fib_accuracy')]))
        strength_factors.append(fib_accuracy)
        
        # Rule compliance
        rules_passed = sum(self.rules_validation.values())
        total_rules = len(self.rules_validation)
        rule_score = rules_passed / max(1, total_rules)
        strength_factors.append(rule_score)
        
        # Volume confirmation bonus
        if ZigzagCharacteristic.VOLUME_CONFIRMATION in self.characteristics:
            strength_factors.append(1.0)
        else:
            strength_factors.append(0.5)
            
        return np.mean(strength_factors)


class ZigzagWaveDetector:
    """
    Specialized Zigzag Wave Pattern Detector.
    
    
    - High-precision zigzag detection
    - Crypto market adaptations
    - Multi-wave zigzag structures
    """
    
    def __init__(self,
                 min_a_wave_steepness: float = 0.3,
                 max_b_retracement: float = 0.786,
                 min_c_to_a_ratio: float = 0.8,
                 confidence_threshold: float = 0.65):
        """
        Initialize zigzag detector.
        
        Args:
            min_a_wave_steepness: Minimum steepness for A wave
            max_b_retracement: Maximum B wave retracement
            min_c_to_a_ratio: Minimum C to A wave ratio
            confidence_threshold: Minimum confidence for detection
        """
        self.min_a_wave_steepness = min_a_wave_steepness
        self.max_b_retracement = max_b_retracement
        self.min_c_to_a_ratio = min_c_to_a_ratio
        self.confidence_threshold = confidence_threshold
        
        # Detection statistics
        self.detection_stats = {
            "total_detections": 0,
            "valid_zigzags": 0,
            "by_type": {zt.value: 0 for zt in ZigzagType}
        }
        
    @performance_monitor
    async def detect_zigzag_waves(self,
                                data: pd.DataFrame,
                                symbol: str,
                                timeframe: str) -> List[ZigzagWave]:
        """
        Detect zigzag wave patterns.
        
        Specialized zigzag detection with crypto adaptations.
        """
        start_time = datetime.utcnow()
        
        # Validate data
        if len(data) < 30:  # Minimum for zigzag
            return []
            
        # Find potential ABC sequences
        abc_candidates = await self._find_abc_sequences(data)
        
        # Filter for zigzag characteristics
        zigzag_candidates = []
        for candidate in abc_candidates:
            if self._passes_zigzag_filters(candidate):
                zigzag_candidates.append(candidate)
                
        # Create zigzag wave objects
        zigzag_waves = []
        for candidate in zigzag_candidates:
            zigzag = await self._create_zigzag_wave(candidate, symbol, timeframe)
            if zigzag and zigzag.confidence >= self.confidence_threshold:
                zigzag_waves.append(zigzag)
                
        # Update statistics
        self.detection_stats["total_detections"] += len(abc_candidates)
        self.detection_stats["valid_zigzags"] += len(zigzag_waves)
        
        # Log results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="zigzag",
            confidence=max([w.confidence for w in zigzag_waves]) if zigzag_waves else 0,
            detected_count=len(zigzag_waves)
        )
        
        return zigzag_waves
        
    async def _find_abc_sequences(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find potential ABC sequences in price data."""
        # Simplified pivot detection for ABC patterns
        highs = data['high'].values
        lows = data['low'].values
        
        candidates = []
        window = 5
        
        # Find significant pivots
        pivots = []
        for i in range(window, len(highs) - window):
            # High pivot
            if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
                pivots.append((i, highs[i], 'high'))
            # Low pivot  
            elif all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
                pivots.append((i, lows[i], 'low'))
                
        # Look for alternating high-low-high or low-high-low sequences
        for i in range(len(pivots) - 2):
            p1, p2, p3 = pivots[i], pivots[i+1], pivots[i+2]
            
            # Check for alternating pattern
            if ((p1[2] == 'high' and p2[2] == 'low' and p3[2] == 'high') or
                (p1[2] == 'low' and p2[2] == 'high' and p3[2] == 'low')):
                
                candidates.append({
                    'a_point': (p1[0], p1[1]),
                    'b_point': (p2[0], p2[1]),
                    'c_point': (p3[0], p3[1]),
                    'pattern_type': 'zigzag' if p1[2] != p2[2] else 'unknown'
                })
                
        return candidates
        
    def _passes_zigzag_filters(self, candidate: Dict[str, Any]) -> bool:
        """Check if candidate passes zigzag-specific filters."""
        a_idx, a_price = candidate['a_point']
        b_idx, b_price = candidate['b_point'] 
        c_idx, c_price = candidate['c_point']
        
        # Calculate basic measurements
        a_move = abs(b_price - a_price)
        b_retracement = abs(b_price - a_price) / abs(c_price - a_price) if c_price != a_price else 0
        c_to_a_ratio = abs(c_price - b_price) / a_move if a_move > 0 else 0
        
        # Apply zigzag filters
        if b_retracement > self.max_b_retracement:
            return False
            
        if c_to_a_ratio < self.min_c_to_a_ratio:
            return False
            
        # Time relationships (zigzags should be relatively swift)
        total_time = c_idx - a_idx
        if total_time > 200:  # Arbitrary limit for demonstration
            return False
            
        return True
        
    async def _create_zigzag_wave(self,
                                candidate: Dict[str, Any],
                                symbol: str,
                                timeframe: str) -> Optional[ZigzagWave]:
        """Create ZigzagWave object from candidate."""
        try:
            a_idx, a_price = candidate['a_point']
            b_idx, b_price = candidate['b_point']
            c_idx, c_price = candidate['c_point']
            
            # Create wave points (simplified timestamps)
            base_time = datetime.now(timezone.utc)
            wave_a_start = WavePoint(a_idx, a_price, base_time)
            wave_a_end = WavePoint(b_idx, b_price, base_time)
            wave_b_start = wave_a_end
            wave_b_end = WavePoint(c_idx, c_price, base_time)
            wave_c_start = wave_b_end
            wave_c_end = WavePoint(c_idx + 1, c_price, base_time)  # Simplified
            
            # Determine direction
            if c_price < a_price:
                direction = CorrectiveDirection.BEARISH
            else:
                direction = CorrectiveDirection.BULLISH
                
            # Calculate measurements
            wave_a_length = abs(b_price - a_price)
            wave_b_length = abs(c_price - b_price)
            wave_c_length = wave_b_length  # Simplified for ABC
            wave_b_retracement = wave_b_length / wave_a_length if wave_a_length > 0 else 0
            
            # Calculate confidence (simplified)
            confidence = self._calculate_zigzag_confidence(candidate)
            
            # Create ZigzagWave
            zigzag = ZigzagWave(
                wave_a=(wave_a_start, wave_a_end),
                wave_b=(wave_b_start, wave_b_end), 
                wave_c=(wave_c_start, wave_c_end),
                corrective_type=CorrectiveType.ZIGZAG,
                direction=direction,
                confidence=confidence,
                timeframe=timeframe,
                symbol=symbol,
                wave_a_length=wave_a_length,
                wave_b_length=wave_b_length,
                wave_c_length=wave_c_length,
                wave_b_retracement=wave_b_retracement,
                fibonacci_ratios={},
                pattern_measurements={},
                rules_validation={},
                guidelines_score=confidence,
                detected_at=datetime.utcnow(),
                processing_time_ms=0.0,
                zigzag_type=ZigzagType.SINGLE_ZIGZAG,
                characteristics=[]
            )
            
            return zigzag
            
        except Exception as e:
            logger.error(f"Error creating zigzag wave: {e}")
            return None
            
    def _calculate_zigzag_confidence(self, candidate: Dict[str, Any]) -> float:
        """Calculate confidence score for zigzag candidate."""
        confidence_factors = []
        
        a_idx, a_price = candidate['a_point']
        b_idx, b_price = candidate['b_point']
        c_idx, c_price = candidate['c_point']
        
        # Sharp A wave (steepness)
        a_steepness = abs(b_price - a_price) / max(b_idx - a_idx, 1)
        confidence_factors.append(min(a_steepness * 10, 1.0))
        
        # Shallow B retracement
        a_range = abs(b_price - a_price)
        total_range = abs(c_price - a_price)
        if total_range > 0:
            b_shallow = 1.0 - (abs(b_price - a_price) / total_range)
            confidence_factors.append(b_shallow)
        else:
            confidence_factors.append(0.5)
            
        # C wave extension
        c_extension = abs(c_price - b_price) / a_range if a_range > 0 else 0
        confidence_factors.append(min(c_extension, 1.0))
        
        # Pattern clarity
        clarity = 1.0 - (abs((b_idx - a_idx) - (c_idx - b_idx)) / max(c_idx - a_idx, 1))
        confidence_factors.append(clarity)
        
        return np.mean(confidence_factors)


# Export main classes
__all__ = [
    'ZigzagWave',
    'ZigzagWaveDetector',
    'ZigzagType',
    'ZigzagCharacteristic'
]