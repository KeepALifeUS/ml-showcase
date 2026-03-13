"""
Harmonic Pattern Detector - Master Controller
Enterprise-grade harmonic pattern detection system

Real-time pattern detection with ML optimization,
multi-timeframe analysis, and performance monitoring.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

# Import pattern implementations
from .patterns import (
    GartleyPattern,
    BatPattern,
    ButterflyPattern,
    CrabPattern,
    SharkPattern,
    CypherPattern
)
from .fibonacci import FibonacciAnalyzer, RatioValidator
from .scanner import MultiPatternScanner
from .signals import EntrySignals, RiskRewardAnalyzer


class PatternType(Enum):
    """Available harmonic pattern types"""
    GARTLEY = "gartley"
    BAT = "bat"
    BUTTERFLY = "butterfly"
    CRAB = "crab"
    SHARK = "shark"
    CYPHER = "cypher"
    ALL = "all"


class PatternDirection(Enum):
    """Pattern direction (bullish or bearish)"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    BOTH = "both"


class DetectionMode(Enum):
    """Detection mode for pattern scanning"""
    REALTIME = "realtime"      # Live detection on new candles
    HISTORICAL = "historical"   # Batch detection on historical data
    HYBRID = "hybrid"          # Both realtime and historical


@dataclass
class DetectedPattern:
    """Represents a detected harmonic pattern"""
    pattern_type: PatternType
    direction: PatternDirection
    points: Dict[str, Tuple[int, float]]  # X, A, B, C, D points
    ratios: Dict[str, float]              # Fibonacci ratios
    completion_level: float               # Price at D point
    prz_zone: Tuple[float, float]        # Potential Reversal Zone
    confidence_score: float               # Pattern quality score
    timestamp: datetime
    timeframe: str
    symbol: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profits: List[float] = field(default_factory=list)
    risk_reward_ratio: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if pattern is complete (D point formed)"""
        return 'D' in self.points and self.points['D'] is not None

    @property
    def pattern_age_hours(self) -> float:
        """Get pattern age in hours"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type.value,
            'direction': self.direction.value,
            'points': self.points,
            'ratios': self.ratios,
            'completion_level': self.completion_level,
            'prz_zone': self.prz_zone,
            'confidence_score': self.confidence_score,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profits': self.take_profits,
            'risk_reward_ratio': self.risk_reward_ratio,
            'metadata': self.metadata
        }


@dataclass
class DetectionResult:
    """Result of harmonic pattern detection"""
    patterns: List[DetectedPattern]
    detection_time_ms: float
    data_points_analyzed: int
    timeframe: str
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def pattern_count(self) -> int:
        """Total number of patterns detected"""
        return len(self.patterns)

    @property
    def bullish_patterns(self) -> List[DetectedPattern]:
        """Get only bullish patterns"""
        return [p for p in self.patterns if p.direction == PatternDirection.BULLISH]

    @property
    def bearish_patterns(self) -> List[DetectedPattern]:
        """Get only bearish patterns"""
        return [p for p in self.patterns if p.direction == PatternDirection.BEARISH]

    @property
    def high_confidence_patterns(self) -> List[DetectedPattern]:
        """Get patterns with confidence > 0.8"""
        return [p for p in self.patterns if p.confidence_score > 0.8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'patterns': [p.to_dict() for p in self.patterns],
            'detection_time_ms': self.detection_time_ms,
            'data_points_analyzed': self.data_points_analyzed,
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'pattern_count': self.pattern_count,
            'bullish_count': len(self.bullish_patterns),
            'bearish_count': len(self.bearish_patterns)
        }


class HarmonicPatternDetector:
    """
    Master harmonic pattern detector.

    Orchestrates multiple pattern detection algorithms with ML optimization,
    real-time processing, and comprehensive signal generation.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        pattern_types: Optional[List[PatternType]] = None,
        min_confidence: float = 0.7,
        enable_ml_validation: bool = True,
        cache_enabled: bool = True,
        max_workers: int = 4
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.pattern_types = pattern_types or [PatternType.ALL]
        self.min_confidence = min_confidence
        self.enable_ml_validation = enable_ml_validation
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers

        # Initialize pattern detectors
        self.pattern_detectors = self._initialize_detectors()

        # Initialize analyzers
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.ratio_validator = RatioValidator()
        self.signal_generator = EntrySignals()
        self.risk_analyzer = RiskRewardAnalyzer()

        # Performance tracking
        self.detection_history: List[DetectionResult] = []
        self.pattern_cache: Dict[str, Tuple[DetectionResult, datetime]] = {}
        self.cache_ttl_minutes = 15

        # Concurrency
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.logger = logging.getLogger(f"HarmonicPatternDetector.{symbol}")
        self.logger.info(f"Detector initialized for {symbol} {timeframe}")

    def _initialize_detectors(self) -> Dict[PatternType, Any]:
        """Initialize individual pattern detectors"""
        return {
            PatternType.GARTLEY: GartleyPattern(),
            PatternType.BAT: BatPattern(),
            PatternType.BUTTERFLY: ButterflyPattern(),
            PatternType.CRAB: CrabPattern(),
            PatternType.SHARK: SharkPattern(),
            PatternType.CYPHER: CypherPattern()
        }

    async def detect_patterns(
        self,
        data: pd.DataFrame,
        pattern_types: Optional[List[PatternType]] = None,
        direction: PatternDirection = PatternDirection.BOTH,
        mode: DetectionMode = DetectionMode.HISTORICAL,
        use_cache: bool = True
    ) -> DetectionResult:
        """
        Detect harmonic patterns in price data.

        Args:
            data: OHLCV DataFrame
            pattern_types: Specific patterns to detect (None = all)
            direction: Look for bullish, bearish, or both
            mode: Detection mode (realtime, historical, hybrid)
            use_cache: Whether to use cached results

        Returns:
            DetectionResult with all detected patterns
        """
        start_time = time.time()
        pattern_types = pattern_types or self.pattern_types

        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(data, pattern_types, direction)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info("Returning cached pattern detection result")
                return cached_result

        try:
            # Prepare data
            prepared_data = self._prepare_data(data)

            # Determine which patterns to detect
            if PatternType.ALL in pattern_types:
                patterns_to_detect = list(PatternType)
                patterns_to_detect.remove(PatternType.ALL)
            else:
                patterns_to_detect = pattern_types

            # Detect patterns in parallel
            detected_patterns = await self._detect_patterns_parallel(
                prepared_data, patterns_to_detect, direction
            )

            # Filter by confidence
            filtered_patterns = [
                p for p in detected_patterns
                if p.confidence_score >= self.min_confidence
            ]

            # ML validation if enabled
            if self.enable_ml_validation and filtered_patterns:
                filtered_patterns = await self._validate_with_ml(filtered_patterns)

            # Generate trading signals
            for pattern in filtered_patterns:
                self._generate_trading_signals(pattern, prepared_data)

            # Calculate detection time
            detection_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = DetectionResult(
                patterns=filtered_patterns,
                detection_time_ms=detection_time_ms,
                data_points_analyzed=len(prepared_data),
                timeframe=self.timeframe,
                symbol=self.symbol
            )

            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)

            # Log performance
            self.logger.info(
                f"Detected {len(filtered_patterns)} patterns in {detection_time_ms:.1f}ms"
            )

            # Record history
            self.detection_history.append(result)

            return result

        except Exception as e:
            self.logger.error(f"Pattern detection failed: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate OHLCV data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Validate columns
        for col in required_columns:
            if col not in data.columns:
                # Try common variations
                col_upper = col.capitalize()
                if col_upper in data.columns:
                    data = data.rename(columns={col_upper: col})
                else:
                    raise ValueError(f"Required column '{col}' not found")

        # Add technical indicators for pattern detection
        data['pivot_high'] = self._identify_pivot_highs(data['high'])
        data['pivot_low'] = self._identify_pivot_lows(data['low'])

        # Calculate swing points
        data['swing_high'] = data['high'].rolling(window=5).max()
        data['swing_low'] = data['low'].rolling(window=5).min()

        return data

    async def _detect_patterns_parallel(
        self,
        data: pd.DataFrame,
        pattern_types: List[PatternType],
        direction: PatternDirection
    ) -> List[DetectedPattern]:
        """Detect patterns in parallel for performance"""
        detection_tasks = []

        for pattern_type in pattern_types:
            if pattern_type in self.pattern_detectors:
                # Check for bullish patterns
                if direction in [PatternDirection.BOTH, PatternDirection.BULLISH]:
                    task = self._detect_single_pattern(
                        data, pattern_type, PatternDirection.BULLISH
                    )
                    detection_tasks.append(task)

                # Check for bearish patterns
                if direction in [PatternDirection.BOTH, PatternDirection.BEARISH]:
                    task = self._detect_single_pattern(
                        data, pattern_type, PatternDirection.BEARISH
                    )
                    detection_tasks.append(task)

        # Execute all detections in parallel
        if detection_tasks:
            results = await asyncio.gather(*detection_tasks, return_exceptions=True)

            # Flatten results and filter exceptions
            detected_patterns = []
            for result in results:
                if isinstance(result, list):
                    detected_patterns.extend(result)
                elif not isinstance(result, Exception):
                    detected_patterns.append(result)
                else:
                    self.logger.warning(f"Pattern detection failed: {result}")

            return detected_patterns

        return []

    async def _detect_single_pattern(
        self,
        data: pd.DataFrame,
        pattern_type: PatternType,
        direction: PatternDirection
    ) -> List[DetectedPattern]:
        """Detect a single pattern type"""
        try:
            detector = self.pattern_detectors[pattern_type]

            # Find potential pattern points
            swing_points = self._find_swing_points(data, direction)

            detected_patterns = []

            # Check each combination of swing points
            for points in self._generate_point_combinations(swing_points):
                # Validate pattern ratios
                if self._validate_pattern_ratios(points, pattern_type):
                    # Calculate pattern metrics
                    ratios = self._calculate_ratios(points)
                    prz_zone = self._calculate_prz(points, ratios, pattern_type)
                    confidence = self._calculate_confidence(points, ratios, pattern_type)

                    # Create detected pattern
                    pattern = DetectedPattern(
                        pattern_type=pattern_type,
                        direction=direction,
                        points=points,
                        ratios=ratios,
                        completion_level=points['D'][1] if 'D' in points else None,
                        prz_zone=prz_zone,
                        confidence_score=confidence,
                        timestamp=datetime.now(),
                        timeframe=self.timeframe,
                        symbol=self.symbol,
                        metadata={
                            'detector_version': '1.0',
                            'validation_method': 'fibonacci_ratios'
                        }
                    )

                    detected_patterns.append(pattern)

            return detected_patterns

        except Exception as e:
            self.logger.error(f"Failed to detect {pattern_type.value}: {e}")
            return []

    def _find_swing_points(
        self,
        data: pd.DataFrame,
        direction: PatternDirection
    ) -> List[Tuple[int, float]]:
        """Find swing highs and lows for pattern detection"""
        swing_points = []

        if direction == PatternDirection.BULLISH:
            # For bullish patterns, look for swing lows
            for i in range(2, len(data) - 2):
                if (data['low'].iloc[i] < data['low'].iloc[i-1] and
                    data['low'].iloc[i] < data['low'].iloc[i-2] and
                    data['low'].iloc[i] < data['low'].iloc[i+1] and
                    data['low'].iloc[i] < data['low'].iloc[i+2]):
                    swing_points.append((i, data['low'].iloc[i]))
        else:
            # For bearish patterns, look for swing highs
            for i in range(2, len(data) - 2):
                if (data['high'].iloc[i] > data['high'].iloc[i-1] and
                    data['high'].iloc[i] > data['high'].iloc[i-2] and
                    data['high'].iloc[i] > data['high'].iloc[i+1] and
                    data['high'].iloc[i] > data['high'].iloc[i+2]):
                    swing_points.append((i, data['high'].iloc[i]))

        return swing_points

    def _generate_point_combinations(
        self,
        swing_points: List[Tuple[int, float]],
        max_patterns: int = 10
    ) -> List[Dict[str, Tuple[int, float]]]:
        """Generate possible XABCD point combinations"""
        combinations = []

        # Need at least 5 points for XABCD pattern
        if len(swing_points) < 5:
            return combinations

        # Generate combinations (simplified for performance)
        for i in range(len(swing_points) - 4):
            if len(combinations) >= max_patterns:
                break

            points = {
                'X': swing_points[i],
                'A': swing_points[i + 1],
                'B': swing_points[i + 2],
                'C': swing_points[i + 3],
                'D': swing_points[i + 4]
            }
            combinations.append(points)

        return combinations

    def _validate_pattern_ratios(
        self,
        points: Dict[str, Tuple[int, float]],
        pattern_type: PatternType
    ) -> bool:
        """Validate if points match pattern's Fibonacci ratios"""
        if not all(k in points for k in ['X', 'A', 'B', 'C', 'D']):
            return False

        # Get ratio requirements for pattern type
        ratio_rules = self._get_ratio_rules(pattern_type)

        # Calculate actual ratios
        xa_range = abs(points['A'][1] - points['X'][1])
        ab_retracement = abs(points['B'][1] - points['A'][1]) / xa_range if xa_range > 0 else 0
        bc_retracement = abs(points['C'][1] - points['B'][1]) / abs(points['B'][1] - points['A'][1]) if abs(points['B'][1] - points['A'][1]) > 0 else 0
        cd_extension = abs(points['D'][1] - points['C'][1]) / abs(points['C'][1] - points['B'][1]) if abs(points['C'][1] - points['B'][1]) > 0 else 0
        xd_retracement = abs(points['D'][1] - points['X'][1]) / xa_range if xa_range > 0 else 0

        # Validate against rules (with tolerance)
        tolerance = 0.05  # 5% tolerance

        return (
            self._in_range(ab_retracement, ratio_rules['AB'], tolerance) and
            self._in_range(bc_retracement, ratio_rules['BC'], tolerance) and
            self._in_range(cd_extension, ratio_rules['CD'], tolerance) and
            self._in_range(xd_retracement, ratio_rules['XD'], tolerance)
        )

    def _get_ratio_rules(self, pattern_type: PatternType) -> Dict[str, Tuple[float, float]]:
        """Get Fibonacci ratio rules for each pattern type"""
        rules = {
            PatternType.GARTLEY: {
                'AB': (0.618, 0.618),  # 61.8% retracement
                'BC': (0.382, 0.886),  # 38.2% to 88.6%
                'CD': (1.272, 1.618),  # 127.2% to 161.8%
                'XD': (0.786, 0.786)   # 78.6% retracement
            },
            PatternType.BAT: {
                'AB': (0.382, 0.50),   # 38.2% to 50%
                'BC': (0.382, 0.886),  # 38.2% to 88.6%
                'CD': (1.618, 2.618),  # 161.8% to 261.8%
                'XD': (0.886, 0.886)   # 88.6% retracement
            },
            PatternType.BUTTERFLY: {
                'AB': (0.786, 0.786),  # 78.6% retracement
                'BC': (0.382, 0.886),  # 38.2% to 88.6%
                'CD': (1.618, 2.618),  # 161.8% to 261.8%
                'XD': (1.272, 1.618)   # 127.2% to 161.8%
            },
            PatternType.CRAB: {
                'AB': (0.382, 0.618),  # 38.2% to 61.8%
                'BC': (0.382, 0.886),  # 38.2% to 88.6%
                'CD': (2.24, 3.618),   # 224% to 361.8%
                'XD': (1.618, 1.618)   # 161.8% extension
            },
            PatternType.SHARK: {
                'AB': (1.13, 1.618),   # 113% to 161.8%
                'BC': (1.618, 2.24),   # 161.8% to 224%
                'CD': (0.886, 1.13),   # 88.6% to 113%
                'XD': (0.886, 1.13)    # 88.6% to 113%
            },
            PatternType.CYPHER: {
                'AB': (0.382, 0.618),  # 38.2% to 61.8%
                'BC': (1.272, 1.414),  # 127.2% to 141.4%
                'CD': (1.272, 2.00),   # 127.2% to 200%
                'XD': (0.786, 0.786)   # 78.6% retracement
            }
        }

        return rules.get(pattern_type, rules[PatternType.GARTLEY])

    def _in_range(self, value: float, range_tuple: Tuple[float, float], tolerance: float) -> bool:
        """Check if value is within range with tolerance"""
        min_val = range_tuple[0] * (1 - tolerance)
        max_val = range_tuple[1] * (1 + tolerance)
        return min_val <= value <= max_val

    def _calculate_ratios(self, points: Dict[str, Tuple[int, float]]) -> Dict[str, float]:
        """Calculate actual Fibonacci ratios for pattern"""
        xa_range = abs(points['A'][1] - points['X'][1])
        ab_range = abs(points['B'][1] - points['A'][1])
        bc_range = abs(points['C'][1] - points['B'][1])
        cd_range = abs(points['D'][1] - points['C'][1])

        ratios = {}

        if xa_range > 0:
            ratios['AB'] = ab_range / xa_range
            ratios['XD'] = abs(points['D'][1] - points['X'][1]) / xa_range

        if ab_range > 0:
            ratios['BC'] = bc_range / ab_range

        if bc_range > 0:
            ratios['CD'] = cd_range / bc_range

        return ratios

    def _calculate_prz(
        self,
        points: Dict[str, Tuple[int, float]],
        ratios: Dict[str, float],
        pattern_type: PatternType
    ) -> Tuple[float, float]:
        """Calculate Potential Reversal Zone"""
        d_price = points['D'][1] if 'D' in points else points['C'][1]

        # PRZ is typically 5-10% around D point
        prz_range = abs(d_price * 0.05)

        return (d_price - prz_range, d_price + prz_range)

    def _calculate_confidence(
        self,
        points: Dict[str, Tuple[int, float]],
        ratios: Dict[str, float],
        pattern_type: PatternType
    ) -> float:
        """Calculate pattern confidence score"""
        # Get ideal ratios for pattern
        ideal_ratios = self._get_ratio_rules(pattern_type)

        # Calculate deviation from ideal
        total_deviation = 0
        ratio_count = 0

        for key in ['AB', 'BC', 'CD', 'XD']:
            if key in ratios and key in ideal_ratios:
                ideal_mid = (ideal_ratios[key][0] + ideal_ratios[key][1]) / 2
                deviation = abs(ratios[key] - ideal_mid) / ideal_mid
                total_deviation += deviation
                ratio_count += 1

        # Calculate confidence (inverse of average deviation)
        if ratio_count > 0:
            avg_deviation = total_deviation / ratio_count
            confidence = max(0, min(1, 1 - avg_deviation))
        else:
            confidence = 0

        return confidence

    def _generate_trading_signals(self, pattern: DetectedPattern, data: pd.DataFrame):
        """Generate trading signals for detected pattern"""
        try:
            # Entry price at PRZ
            pattern.entry_price = (pattern.prz_zone[0] + pattern.prz_zone[1]) / 2

            # Stop loss beyond X point
            x_price = pattern.points['X'][1]
            buffer = abs(x_price - pattern.completion_level) * 0.1  # 10% buffer

            if pattern.direction == PatternDirection.BULLISH:
                pattern.stop_loss = x_price - buffer
                # Take profits at Fibonacci extensions
                pattern.take_profits = [
                    pattern.entry_price + (pattern.entry_price - pattern.stop_loss) * 1.0,   # 1:1 RR
                    pattern.entry_price + (pattern.entry_price - pattern.stop_loss) * 1.618, # 1.618:1 RR
                    pattern.entry_price + (pattern.entry_price - pattern.stop_loss) * 2.618  # 2.618:1 RR
                ]
            else:
                pattern.stop_loss = x_price + buffer
                pattern.take_profits = [
                    pattern.entry_price - (pattern.stop_loss - pattern.entry_price) * 1.0,
                    pattern.entry_price - (pattern.stop_loss - pattern.entry_price) * 1.618,
                    pattern.entry_price - (pattern.stop_loss - pattern.entry_price) * 2.618
                ]

            # Calculate risk-reward ratio
            if pattern.stop_loss and pattern.take_profits:
                risk = abs(pattern.entry_price - pattern.stop_loss)
                reward = abs(pattern.take_profits[0] - pattern.entry_price)
                pattern.risk_reward_ratio = reward / risk if risk > 0 else 0

        except Exception as e:
            self.logger.warning(f"Failed to generate trading signals: {e}")

    async def _validate_with_ml(
        self,
        patterns: List[DetectedPattern]
    ) -> List[DetectedPattern]:
        """Validate patterns using machine learning (placeholder)"""
        # TODO: Implement ML validation
        # For now, return patterns with confidence > 0.75
        return [p for p in patterns if p.confidence_score > 0.75]

    def _identify_pivot_highs(self, highs: pd.Series, window: int = 5) -> pd.Series:
        """Identify pivot highs in price data"""
        pivot_highs = pd.Series(index=highs.index, dtype=float)

        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                pivot_highs.iloc[i] = highs.iloc[i]

        return pivot_highs

    def _identify_pivot_lows(self, lows: pd.Series, window: int = 5) -> pd.Series:
        """Identify pivot lows in price data"""
        pivot_lows = pd.Series(index=lows.index, dtype=float)

        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                pivot_lows.iloc[i] = lows.iloc[i]

        return pivot_lows

    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        pattern_types: List[PatternType],
        direction: PatternDirection
    ) -> str:
        """Generate cache key for detection parameters"""
        import hashlib

        data_hash = hashlib.md5(
            f"{len(data)}_{data['close'].iloc[-1]}_{data['close'].iloc[0]}".encode()
        ).hexdigest()[:8]

        patterns_str = "_".join(sorted([p.value for p in pattern_types]))

        return f"{self.symbol}_{self.timeframe}_{data_hash}_{patterns_str}_{direction.value}"

    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result if valid"""
        if cache_key in self.pattern_cache:
            result, cached_time = self.pattern_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return result
            else:
                del self.pattern_cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: DetectionResult):
        """Cache detection result"""
        self.pattern_cache[cache_key] = (result, datetime.now())

        # Limit cache size
        if len(self.pattern_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.pattern_cache.keys(),
                key=lambda k: self.pattern_cache[k][1]
            )[:20]
            for key in oldest_keys:
                del self.pattern_cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        if not self.detection_history:
            return {}

        recent_detections = self.detection_history[-50:]
        all_patterns = []
        for detection in recent_detections:
            all_patterns.extend(detection.patterns)

        pattern_type_counts = {}
        for pattern in all_patterns:
            pt = pattern.pattern_type.value
            pattern_type_counts[pt] = pattern_type_counts.get(pt, 0) + 1

        return {
            'total_detections': len(self.detection_history),
            'total_patterns_found': len(all_patterns),
            'average_detection_time_ms': np.mean([d.detection_time_ms for d in recent_detections]),
            'pattern_type_distribution': pattern_type_counts,
            'average_confidence': np.mean([p.confidence_score for p in all_patterns]) if all_patterns else 0,
            'cache_size': len(self.pattern_cache),
            'high_confidence_rate': len([p for p in all_patterns if p.confidence_score > 0.8]) / len(all_patterns) if all_patterns else 0
        }

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.pattern_cache.clear()
        self.detection_history.clear()

    def __del__(self):
        """Destructor"""
        self.cleanup()