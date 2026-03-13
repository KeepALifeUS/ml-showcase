"""
Automatic Wave Counting System for Elliott Wave Analysis.

Intelligent wave counting with machine learning
assistance and crypto market adaptations for 24/7 trading conditions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone

from ..utils.logger import get_logger, trading_logger, performance_monitor
from ..utils.config import config, WaveDegree
from ..patterns.impulse_wave import ImpulseWave, WavePoint
from ..patterns.corrective_wave import CorrectiveWave

logger = get_logger(__name__)


class WaveCountType(str, Enum):
    """Types of wave counts."""
    IMPULSE_COUNT = "impulse_count"
    CORRECTIVE_COUNT = "corrective_count"
    COMPLEX_COUNT = "complex_count"
    MIXED_COUNT = "mixed_count"


class CountConfidence(str, Enum):
    """Confidence levels for wave counts."""
    HIGH = "high"          # 80%+
    MEDIUM = "medium"      # 60-79%
    LOW = "low"           # 40-59%
    SPECULATIVE = "speculative"  # <40%


@dataclass
class WaveCount:
    """
    Complete wave count structure.
    
    Rich domain model for wave counting results.
    """
    waves: List[Union[ImpulseWave, CorrectiveWave]]
    count_type: WaveCountType
    primary_degree: WaveDegree
    confidence: float
    
    # Count metadata
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    
    # Wave sequence information
    wave_sequence: List[str]  # e.g., ["1", "2", "3", "4", "5", "A", "B", "C"]
    degree_sequence: List[WaveDegree]
    
    # Pattern relationships
    fibonacci_confluence: Dict[str, float]
    time_relationships: Dict[str, float]
    
    # Validation scores
    rules_compliance: float
    guideline_adherence: float
    pattern_clarity: float
    
    # Alternative counts
    alternative_interpretations: List['WaveCount'] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: float = 0.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Initialize wave count analysis."""
        self.analyze_fibonacci_confluence()
        self.analyze_time_relationships()
        self.calculate_overall_scores()
        
    def analyze_fibonacci_confluence(self) -> Dict[str, float]:
        """Analyze Fibonacci confluence across waves."""
        confluence = {}
        
        # Collect all Fibonacci levels from waves
        all_fib_levels = []
        for wave in self.waves:
            if hasattr(wave, 'fibonacci_ratios'):
                for key, value in wave.fibonacci_ratios.items():
                    if isinstance(value, float) and 0.1 < value < 10.0:
                        all_fib_levels.append(value)
                        
        # Find confluence zones (levels within 5% of each other)
        confluence_zones = self._find_confluence_zones(all_fib_levels, tolerance=0.05)
        
        for i, zone in enumerate(confluence_zones):
            confluence[f"zone_{i}"] = {
                "center": np.mean(zone),
                "count": len(zone),
                "strength": len(zone) / len(all_fib_levels) if all_fib_levels else 0
            }
            
        self.fibonacci_confluence = confluence
        return confluence
        
    def _find_confluence_zones(self, levels: List[float], tolerance: float) -> List[List[float]]:
        """Find zones where multiple Fibonacci levels cluster."""
        if not levels:
            return []
            
        sorted_levels = sorted(levels)
        zones = []
        current_zone = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            if abs(level - current_zone[-1]) / current_zone[-1] <= tolerance:
                current_zone.append(level)
            else:
                if len(current_zone) >= 2:  # Only significant zones
                    zones.append(current_zone)
                current_zone = [level]
                
        if len(current_zone) >= 2:
            zones.append(current_zone)
            
        return zones
        
    def analyze_time_relationships(self) -> Dict[str, float]:
        """Analyze time relationships between waves."""
        relationships = {}
        
        if len(self.waves) >= 2:
            # Time equality and Fibonacci time ratios
            for i in range(len(self.waves) - 1):
                wave1 = self.waves[i]
                wave2 = self.waves[i + 1]
                
                # Calculate time durations
                if hasattr(wave1, 'detected_at') and hasattr(wave2, 'detected_at'):
                    duration1 = self._get_wave_duration(wave1)
                    duration2 = self._get_wave_duration(wave2)
                    
                    if duration1 > 0:
                        time_ratio = duration2 / duration1
                        relationships[f"wave_{i+1}_to_wave_{i}_time"] = time_ratio
                        
        self.time_relationships = relationships
        return relationships
        
    def _get_wave_duration(self, wave: Union[ImpulseWave, CorrectiveWave]) -> float:
        """Get duration of wave in seconds."""
        if isinstance(wave, ImpulseWave):
            start_time = wave.wave_1[0].timestamp
            end_time = wave.wave_5[1].timestamp
        elif isinstance(wave, CorrectiveWave):
            start_time = wave.wave_a[0].timestamp
            end_time = wave.wave_c[1].timestamp
        else:
            return 0.0
            
        return (end_time - start_time).total_seconds()
        
    def calculate_overall_scores(self) -> Dict[str, float]:
        """Calculate overall quality scores for wave count."""
        scores = {
            "rules_compliance": self._calculate_rules_compliance(),
            "guideline_adherence": self._calculate_guideline_adherence(),
            "pattern_clarity": self._calculate_pattern_clarity()
        }
        
        self.rules_compliance = scores["rules_compliance"]
        self.guideline_adherence = scores["guideline_adherence"]
        self.pattern_clarity = scores["pattern_clarity"]
        
        return scores
        
    def _calculate_rules_compliance(self) -> float:
        """Calculate Elliott Wave rules compliance score."""
        if not self.waves:
            return 0.0
            
        total_score = 0.0
        for wave in self.waves:
            if hasattr(wave, 'rules_validation'):
                wave_score = sum(wave.rules_validation.values()) / len(wave.rules_validation)
                total_score += wave_score
                
        return total_score / len(self.waves)
        
    def _calculate_guideline_adherence(self) -> float:
        """Calculate Elliott Wave guidelines adherence score."""
        if not self.waves:
            return 0.0
            
        total_score = 0.0
        for wave in self.waves:
            if hasattr(wave, 'guidelines_score'):
                total_score += wave.guidelines_score
                
        return total_score / len(self.waves)
        
    def _calculate_pattern_clarity(self) -> float:
        """Calculate pattern clarity score."""
        clarity_factors = []
        
        # Fibonacci confluence strength
        if self.fibonacci_confluence:
            confluence_strength = sum(
                zone.get("strength", 0) for zone in self.fibonacci_confluence.values()
                if isinstance(zone, dict)
            )
            clarity_factors.append(min(confluence_strength, 1.0))
            
        # Time relationship clarity
        if self.time_relationships:
            fib_time_ratios = [0.618, 1.0, 1.618]
            time_clarity = 0.0
            for ratio in self.time_relationships.values():
                closest_fib = min(fib_time_ratios, key=lambda x: abs(x - ratio))
                accuracy = 1 - abs(ratio - closest_fib) / closest_fib
                if accuracy > 0.8:  # Close to Fibonacci ratio
                    time_clarity += 0.2
            clarity_factors.append(min(time_clarity, 1.0))
            
        # Wave sequence logic
        clarity_factors.append(self._evaluate_sequence_logic())
        
        return np.mean(clarity_factors) if clarity_factors else 0.0
        
    def _evaluate_sequence_logic(self) -> float:
        """Evaluate logical consistency of wave sequence."""
        # Check for proper alternation and progression
        if not self.wave_sequence:
            return 0.0
            
        # Simple logic check - more sophisticated analysis needed
        expected_patterns = [
            ["1", "2", "3", "4", "5"],  # Impulse
            ["A", "B", "C"],            # Corrective  
            ["1", "2", "3", "4", "5", "A", "B", "C"]  # Impulse + Corrective
        ]
        
        for pattern in expected_patterns:
            if self._sequence_matches_pattern(self.wave_sequence, pattern):
                return 1.0
                
        return 0.5  # Partial match
        
    def _sequence_matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        """Check if sequence matches expected pattern."""
        return sequence == pattern or all(s in pattern for s in sequence)
        
    def get_next_expected_wave(self) -> Optional[str]:
        """Get the next expected wave in sequence."""
        if not self.wave_sequence:
            return "1"  # Start with wave 1
            
        last_wave = self.wave_sequence[-1]
        
        # Impulse sequence
        impulse_next = {
            "1": "2", "2": "3", "3": "4", "4": "5", "5": "A"
        }
        
        # Corrective sequence  
        corrective_next = {
            "A": "B", "B": "C", "C": "1"  # Next degree
        }
        
        return impulse_next.get(last_wave) or corrective_next.get(last_wave)
        
    def get_confluence_zones(self) -> List[Dict[str, Any]]:
        """Get significant Fibonacci confluence zones."""
        zones = []
        for zone_name, zone_data in self.fibonacci_confluence.items():
            if isinstance(zone_data, dict) and zone_data.get("count", 0) >= 3:
                zones.append({
                    "name": zone_name,
                    "level": zone_data["center"],
                    "strength": zone_data["strength"],
                    "count": zone_data["count"]
                })
        return sorted(zones, key=lambda x: x["strength"], reverse=True)
        
    @property
    def confidence_level(self) -> CountConfidence:
        """Get confidence level category."""
        if self.confidence >= 0.8:
            return CountConfidence.HIGH
        elif self.confidence >= 0.6:
            return CountConfidence.MEDIUM
        elif self.confidence >= 0.4:
            return CountConfidence.LOW
        else:
            return CountConfidence.SPECULATIVE
            
    @property
    def is_valid(self) -> bool:
        """Check if wave count is considered valid."""
        return (self.confidence >= 0.4 and 
                self.rules_compliance >= 0.6 and
                len(self.waves) > 0)
                
    @property
    def completion_percentage(self) -> float:
        """Estimate completion percentage of current pattern."""
        if not self.wave_sequence:
            return 0.0
            
        if self.count_type == WaveCountType.IMPULSE_COUNT:
            impulse_waves = ["1", "2", "3", "4", "5"]
            completed = len([w for w in self.wave_sequence if w in impulse_waves])
            return (completed / 5.0) * 100.0
        elif self.count_type == WaveCountType.CORRECTIVE_COUNT:
            corrective_waves = ["A", "B", "C"]
            completed = len([w for w in self.wave_sequence if w in corrective_waves])
            return (completed / 3.0) * 100.0
            
        return 0.0


class WaveCounter:
    """
    Advanced Automatic Wave Counting System.
    
    
    - ML-assisted wave recognition
    - Multi-timeframe counting synchronization
    - Real-time count updates
    - Crypto market adaptations
    """
    
    def __init__(self,
                 primary_degree: WaveDegree = WaveDegree.MINOR,
                 confidence_threshold: float = 0.4,
                 max_alternatives: int = 3):
        """
        Initialize wave counter.
        
        Args:
            primary_degree: Primary wave degree for analysis
            confidence_threshold: Minimum confidence for valid counts
            max_alternatives: Maximum alternative count interpretations
        """
        self.primary_degree = primary_degree
        self.confidence_threshold = confidence_threshold
        self.max_alternatives = max_alternatives
        
        # Counting statistics
        self.counting_stats = {
            "total_counts": 0,
            "valid_counts": 0,
            "by_type": {ct.value: 0 for ct in WaveCountType},
            "accuracy_metrics": []
        }
        
    @performance_monitor
    async def count_waves(self,
                         data: pd.DataFrame,
                         symbol: str,
                         timeframe: str,
                         detected_waves: List[Union[ImpulseWave, CorrectiveWave]] = None) -> List[WaveCount]:
        """
        Perform automatic wave counting.
        
        Comprehensive wave counting with alternatives.
        
        Args:
            data: Price data
            symbol: Trading symbol
            timeframe: Timeframe string
            detected_waves: Pre-detected waves (optional)
            
        Returns:
            List[WaveCount]: Possible wave counts ordered by confidence
        """
        start_time = datetime.utcnow()
        
        # Validate input
        if data.empty:
            return []
            
        # If no waves provided, detect them first
        if detected_waves is None:
            detected_waves = await self._detect_waves_for_counting(data, symbol, timeframe)
            
        if not detected_waves:
            return []
            
        # Generate primary wave count
        primary_count = await self._generate_primary_count(
            detected_waves, symbol, timeframe, start_time
        )
        
        wave_counts = []
        if primary_count and primary_count.confidence >= self.confidence_threshold:
            wave_counts.append(primary_count)
            
            # Generate alternative counts
            alternatives = await self._generate_alternative_counts(
                detected_waves, primary_count, symbol, timeframe
            )
            wave_counts.extend(alternatives)
            
        # Sort by confidence
        wave_counts.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to max alternatives
        wave_counts = wave_counts[:self.max_alternatives + 1]
        
        # Log counting results
        trading_logger.log_wave_detection(
            symbol=symbol,
            timeframe=timeframe,
            wave_type="count",
            confidence=wave_counts[0].confidence if wave_counts else 0,
            detected_count=len(wave_counts),
            primary_degree=self.primary_degree.value
        )
        
        # Update statistics
        self.counting_stats["total_counts"] += 1
        if wave_counts:
            self.counting_stats["valid_counts"] += 1
            count_type = wave_counts[0].count_type
            self.counting_stats["by_type"][count_type.value] += 1
            
        return wave_counts
        
    async def _detect_waves_for_counting(self,
                                       data: pd.DataFrame,
                                       symbol: str,
                                       timeframe: str) -> List[Union[ImpulseWave, CorrectiveWave]]:
        """Detect waves if not provided."""
        # Import here to avoid circular imports
        from ..patterns.impulse_wave import ImpulseWaveDetector
        from ..patterns.corrective_wave import CorrectiveWaveDetector
        
        waves = []
        
        # Detect impulse waves
        impulse_detector = ImpulseWaveDetector()
        impulse_waves = await impulse_detector.detect_impulse_waves(data, symbol, timeframe)
        waves.extend(impulse_waves)
        
        # Detect corrective waves
        corrective_detector = CorrectiveWaveDetector()
        corrective_waves = await corrective_detector.detect_corrective_waves(data, symbol, timeframe)
        waves.extend(corrective_waves)
        
        # Sort by start time
        waves.sort(key=lambda w: self._get_wave_start_time(w))
        
        return waves
        
    def _get_wave_start_time(self, wave: Union[ImpulseWave, CorrectiveWave]) -> datetime:
        """Get start time of wave."""
        if isinstance(wave, ImpulseWave):
            return wave.wave_1[0].timestamp
        elif isinstance(wave, CorrectiveWave):
            return wave.wave_a[0].timestamp
        else:
            return datetime.min.replace(tzinfo=timezone.utc)
            
    async def _generate_primary_count(self,
                                    waves: List[Union[ImpulseWave, CorrectiveWave]],
                                    symbol: str,
                                    timeframe: str,
                                    start_time: datetime) -> Optional[WaveCount]:
        """Generate primary wave count interpretation."""
        if not waves:
            return None
            
        try:
            # Determine count type based on wave patterns
            count_type = self._determine_count_type(waves)
            
            # Generate wave sequence labels
            wave_sequence = self._generate_wave_sequence(waves, count_type)
            
            # Generate degree sequence
            degree_sequence = [self.primary_degree] * len(waves)
            
            # Calculate confidence
            confidence = self._calculate_count_confidence(waves, count_type, wave_sequence)
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create wave count
            wave_count = WaveCount(
                waves=waves,
                count_type=count_type,
                primary_degree=self.primary_degree,
                confidence=confidence,
                symbol=symbol,
                timeframe=timeframe,
                start_time=self._get_wave_start_time(waves[0]),
                end_time=self._get_wave_end_time(waves[-1]),
                wave_sequence=wave_sequence,
                degree_sequence=degree_sequence,
                fibonacci_confluence={},  # Will be calculated in __post_init__
                time_relationships={},    # Will be calculated in __post_init__
                rules_compliance=0.0,     # Will be calculated in __post_init__
                guideline_adherence=0.0,  # Will be calculated in __post_init__
                pattern_clarity=0.0,      # Will be calculated in __post_init__
                processing_time_ms=processing_time
            )
            
            return wave_count
            
        except Exception as e:
            logger.error(f"Error generating primary count: {e}")
            return None
            
    def _get_wave_end_time(self, wave: Union[ImpulseWave, CorrectiveWave]) -> datetime:
        """Get end time of wave."""
        if isinstance(wave, ImpulseWave):
            return wave.wave_5[1].timestamp
        elif isinstance(wave, CorrectiveWave):
            return wave.wave_c[1].timestamp
        else:
            return datetime.max.replace(tzinfo=timezone.utc)
            
    def _determine_count_type(self, waves: List[Union[ImpulseWave, CorrectiveWave]]) -> WaveCountType:
        """Determine the type of wave count based on detected patterns."""
        impulse_count = len([w for w in waves if isinstance(w, ImpulseWave)])
        corrective_count = len([w for w in waves if isinstance(w, CorrectiveWave)])
        
        if impulse_count > corrective_count:
            return WaveCountType.IMPULSE_COUNT
        elif corrective_count > impulse_count:
            return WaveCountType.CORRECTIVE_COUNT
        elif impulse_count > 0 and corrective_count > 0:
            return WaveCountType.MIXED_COUNT
        else:
            return WaveCountType.COMPLEX_COUNT
            
    def _generate_wave_sequence(self,
                              waves: List[Union[ImpulseWave, CorrectiveWave]],
                              count_type: WaveCountType) -> List[str]:
        """Generate wave sequence labels."""
        sequence = []
        
        if count_type == WaveCountType.IMPULSE_COUNT:
            # Label as impulse sequence
            impulse_labels = ["1", "2", "3", "4", "5"]
            for i, wave in enumerate(waves[:5]):
                sequence.append(impulse_labels[i])
                
        elif count_type == WaveCountType.CORRECTIVE_COUNT:
            # Label as corrective sequence
            corrective_labels = ["A", "B", "C"]
            for i, wave in enumerate(waves[:3]):
                sequence.append(corrective_labels[i])
                
        elif count_type == WaveCountType.MIXED_COUNT:
            # Mixed labeling - need more sophisticated logic
            current_label = 1
            for wave in waves:
                if isinstance(wave, ImpulseWave):
                    sequence.append(str(current_label))
                    current_label += 1
                    if current_label > 5:
                        current_label = 1  # Reset for next cycle
                elif isinstance(wave, CorrectiveWave):
                    corrective_labels = ["A", "B", "C"]
                    label_idx = (len(sequence) - 5) % 3 if len(sequence) >= 5 else 0
                    if label_idx < len(corrective_labels):
                        sequence.append(corrective_labels[label_idx])
                        
        else:
            # Complex count - generic numbering
            for i in range(len(waves)):
                sequence.append(str(i + 1))
                
        return sequence
        
    def _calculate_count_confidence(self,
                                  waves: List[Union[ImpulseWave, CorrectiveWave]],
                                  count_type: WaveCountType,
                                  wave_sequence: List[str]) -> float:
        """Calculate confidence score for wave count."""
        confidence_factors = []
        
        # Individual wave confidence
        wave_confidences = [w.confidence for w in waves if hasattr(w, 'confidence')]
        if wave_confidences:
            confidence_factors.append(np.mean(wave_confidences))
            
        # Sequence logic
        sequence_score = self._evaluate_sequence_logic_score(wave_sequence, count_type)
        confidence_factors.append(sequence_score)
        
        # Pattern consistency
        consistency_score = self._evaluate_pattern_consistency(waves)
        confidence_factors.append(consistency_score)
        
        # Time and price relationships
        relationship_score = self._evaluate_wave_relationships(waves)
        confidence_factors.append(relationship_score)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
        
    def _evaluate_sequence_logic_score(self, sequence: List[str], count_type: WaveCountType) -> float:
        """Evaluate logical consistency of wave sequence."""
        if not sequence:
            return 0.0
            
        if count_type == WaveCountType.IMPULSE_COUNT:
            expected = ["1", "2", "3", "4", "5"]
            matches = sum(1 for i, s in enumerate(sequence[:5]) if i < len(expected) and s == expected[i])
            return matches / 5.0
            
        elif count_type == WaveCountType.CORRECTIVE_COUNT:
            expected = ["A", "B", "C"]
            matches = sum(1 for i, s in enumerate(sequence[:3]) if i < len(expected) and s == expected[i])
            return matches / 3.0
            
        return 0.5  # Default for mixed/complex
        
    def _evaluate_pattern_consistency(self, waves: List[Union[ImpulseWave, CorrectiveWave]]) -> float:
        """Evaluate consistency of wave patterns."""
        if not waves:
            return 0.0
            
        # Check for alternation in wave types
        if len(waves) > 1:
            alternation_score = 0.0
            for i in range(len(waves) - 1):
                if type(waves[i]) != type(waves[i + 1]):
                    alternation_score += 1.0
            return alternation_score / (len(waves) - 1)
            
        return 1.0
        
    def _evaluate_wave_relationships(self, waves: List[Union[ImpulseWave, CorrectiveWave]]) -> float:
        """Evaluate time and price relationships between waves."""
        if len(waves) < 2:
            return 1.0
            
        # Simple relationship evaluation - more sophisticated logic needed
        return 0.7  # Placeholder
        
    async def _generate_alternative_counts(self,
                                         waves: List[Union[ImpulseWave, CorrectiveWave]],
                                         primary_count: WaveCount,
                                         symbol: str,
                                         timeframe: str) -> List[WaveCount]:
        """Generate alternative wave count interpretations."""
        alternatives = []
        
        # Alternative count type interpretations
        if primary_count.count_type == WaveCountType.IMPULSE_COUNT:
            # Try corrective interpretation
            alt_count = await self._try_alternative_count_type(
                waves, WaveCountType.CORRECTIVE_COUNT, symbol, timeframe
            )
            if alt_count and alt_count.confidence >= self.confidence_threshold * 0.8:
                alternatives.append(alt_count)
                
        elif primary_count.count_type == WaveCountType.CORRECTIVE_COUNT:
            # Try impulse interpretation
            alt_count = await self._try_alternative_count_type(
                waves, WaveCountType.IMPULSE_COUNT, symbol, timeframe
            )
            if alt_count and alt_count.confidence >= self.confidence_threshold * 0.8:
                alternatives.append(alt_count)
                
        # Different degree interpretations
        for alt_degree in [WaveDegree.INTERMEDIATE, WaveDegree.PRIMARY, WaveDegree.MINUTE]:
            if alt_degree != self.primary_degree:
                alt_count = await self._try_alternative_degree(
                    waves, primary_count.count_type, alt_degree, symbol, timeframe
                )
                if alt_count and alt_count.confidence >= self.confidence_threshold * 0.7:
                    alternatives.append(alt_count)
                    
        return alternatives[:self.max_alternatives]
        
    async def _try_alternative_count_type(self,
                                        waves: List[Union[ImpulseWave, CorrectiveWave]],
                                        count_type: WaveCountType,
                                        symbol: str,
                                        timeframe: str) -> Optional[WaveCount]:
        """Try alternative count type interpretation."""
        try:
            wave_sequence = self._generate_wave_sequence(waves, count_type)
            degree_sequence = [self.primary_degree] * len(waves)
            confidence = self._calculate_count_confidence(waves, count_type, wave_sequence) * 0.9
            
            return WaveCount(
                waves=waves,
                count_type=count_type,
                primary_degree=self.primary_degree,
                confidence=confidence,
                symbol=symbol,
                timeframe=timeframe,
                start_time=self._get_wave_start_time(waves[0]),
                end_time=self._get_wave_end_time(waves[-1]),
                wave_sequence=wave_sequence,
                degree_sequence=degree_sequence,
                fibonacci_confluence={},
                time_relationships={},
                rules_compliance=0.0,
                guideline_adherence=0.0,
                pattern_clarity=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating alternative count type: {e}")
            return None
            
    async def _try_alternative_degree(self,
                                    waves: List[Union[ImpulseWave, CorrectiveWave]],
                                    count_type: WaveCountType,
                                    degree: WaveDegree,
                                    symbol: str,
                                    timeframe: str) -> Optional[WaveCount]:
        """Try alternative degree interpretation."""
        try:
            wave_sequence = self._generate_wave_sequence(waves, count_type)
            degree_sequence = [degree] * len(waves)
            confidence = self._calculate_count_confidence(waves, count_type, wave_sequence) * 0.8
            
            return WaveCount(
                waves=waves,
                count_type=count_type,
                primary_degree=degree,
                confidence=confidence,
                symbol=symbol,
                timeframe=timeframe,
                start_time=self._get_wave_start_time(waves[0]),
                end_time=self._get_wave_end_time(waves[-1]),
                wave_sequence=wave_sequence,
                degree_sequence=degree_sequence,
                fibonacci_confluence={},
                time_relationships={},
                rules_compliance=0.0,
                guideline_adherence=0.0,
                pattern_clarity=0.0
            )
            
        except Exception as e:
            logger.error(f"Error creating alternative degree count: {e}")
            return None


# Export main classes
__all__ = [
    'WaveCount',
    'WaveCounter',
    'WaveCountType',
    'CountConfidence'
]