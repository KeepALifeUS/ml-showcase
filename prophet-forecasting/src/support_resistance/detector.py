"""
Support and Resistance Detector
ML-Framework-1332 - Core detection engine for support/resistance levels

 2025: Multi-algorithm detection, adaptive level management,
performance-optimized with caching and real-time updates.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import hashlib
import json

import numpy as np
import pandas as pd

from .algorithms import (
    PivotPointCalculator,
    PeakTroughDetector,
    VolumeProfileAnalyzer,
    PsychologicalLevelDetector,
    MLLevelDetector
)
from .level_manager import LevelManager, LevelStrength


class DetectionMethod(Enum):
    """Available detection methods"""
    PIVOT_POINTS = "pivot_points"
    PEAK_TROUGH = "peak_trough"
    VOLUME_PROFILE = "volume_profile"
    PSYCHOLOGICAL = "psychological"
    ML_BASED = "ml_based"
    FIBONACCI = "fibonacci"
    CAMARILLA = "camarilla"
    WOODIE = "woodie"
    ALL = "all"


class LevelType(Enum):
    """Type of price level"""
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"
    CONFLUENCE = "confluence"


@dataclass
class SupportResistanceLevel:
    """Individual support/resistance level"""
    price: float
    level_type: LevelType
    strength: float  # 0.0 to 1.0
    method: DetectionMethod
    first_detected: datetime
    last_validated: datetime
    touch_count: int = 0
    break_count: int = 0
    timeframe: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_days(self) -> int:
        """Age of the level in days"""
        return (datetime.now() - self.first_detected).days

    @property
    def reliability_score(self) -> float:
        """Calculate reliability score based on touches and age"""
        touch_score = min(self.touch_count / 5, 1.0)  # Max score at 5 touches
        age_penalty = max(0, 1 - (self.age_days / 30))  # Decay over 30 days
        break_penalty = max(0, 1 - (self.break_count * 0.2))  # 20% penalty per break

        return (touch_score * 0.4 + self.strength * 0.4 + age_penalty * 0.1 + break_penalty * 0.1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'price': self.price,
            'level_type': self.level_type.value,
            'strength': self.strength,
            'method': self.method.value,
            'first_detected': self.first_detected.isoformat(),
            'last_validated': self.last_validated.isoformat(),
            'touch_count': self.touch_count,
            'break_count': self.break_count,
            'timeframe': self.timeframe,
            'reliability_score': self.reliability_score,
            'age_days': self.age_days,
            'metadata': self.metadata
        }


@dataclass
class DetectionResult:
    """Result of support/resistance detection"""
    levels: List[SupportResistanceLevel]
    confluence_zones: List[Tuple[float, float, float]]  # (min, max, strength)
    detection_time_ms: float
    methods_used: List[str]
    data_points_analyzed: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def support_levels(self) -> List[SupportResistanceLevel]:
        """Get only support levels"""
        return [l for l in self.levels if l.level_type == LevelType.SUPPORT]

    @property
    def resistance_levels(self) -> List[SupportResistanceLevel]:
        """Get only resistance levels"""
        return [l for l in self.levels if l.level_type == LevelType.RESISTANCE]

    @property
    def strongest_levels(self) -> List[SupportResistanceLevel]:
        """Get top 5 strongest levels"""
        return sorted(self.levels, key=lambda x: x.reliability_score, reverse=True)[:5]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'levels': [l.to_dict() for l in self.levels],
            'confluence_zones': self.confluence_zones,
            'detection_time_ms': self.detection_time_ms,
            'methods_used': self.methods_used,
            'data_points_analyzed': self.data_points_analyzed,
            'timestamp': self.timestamp.isoformat(),
            'support_count': len(self.support_levels),
            'resistance_count': len(self.resistance_levels)
        }


class SupportResistanceDetector:
    """
    Advanced support and resistance level detector with enterprise patterns.

    Combines multiple detection algorithms with adaptive level management
    and performance optimization for real-time trading applications.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        methods: Optional[List[DetectionMethod]] = None,
        lookback_periods: int = 100,
        min_level_strength: float = 0.3,
        confluence_threshold: float = 0.02,  # 2% price range for confluence
        cache_enabled: bool = True,
        max_workers: int = 2
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.methods = methods or [DetectionMethod.PIVOT_POINTS, DetectionMethod.PEAK_TROUGH]
        self.lookback_periods = lookback_periods
        self.min_level_strength = min_level_strength
        self.confluence_threshold = confluence_threshold
        self.cache_enabled = cache_enabled
        self.max_workers = max_workers

        # Initialize algorithm components
        self.pivot_calculator = PivotPointCalculator()
        self.peak_trough_detector = PeakTroughDetector()
        self.volume_analyzer = VolumeProfileAnalyzer()
        self.psychological_detector = PsychologicalLevelDetector()
        self.ml_detector = MLLevelDetector(symbol)

        # Level management
        self.level_manager = LevelManager(confluence_threshold)

        # Caching
        self.cache: Dict[str, Tuple[DetectionResult, datetime]] = {}
        self.cache_ttl_minutes = 5

        # Performance tracking
        self.detection_history: List[Dict] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.logger = logging.getLogger(f"SupportResistanceDetector.{symbol}")
        self.logger.info(f"Detector initialized for {symbol} {timeframe}")

    async def detect_levels(
        self,
        data: pd.DataFrame,
        methods: Optional[List[DetectionMethod]] = None,
        use_cache: bool = True
    ) -> DetectionResult:
        """
        Detect support and resistance levels using specified methods.

        Args:
            data: OHLCV DataFrame with columns: timestamp, open, high, low, close, volume
            methods: Override default detection methods
            use_cache: Whether to use cached results

        Returns:
            DetectionResult with detected levels and metadata
        """
        methods = methods or self.methods
        start_time = time.time()

        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(data, methods)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info("Returning cached detection result")
                return cached_result

        try:
            # Validate and prepare data
            prepared_data = self._prepare_data(data)

            # Run detection methods
            all_levels = []

            if DetectionMethod.ALL in methods:
                methods = [m for m in DetectionMethod if m != DetectionMethod.ALL]

            # Execute detection methods in parallel
            detection_tasks = []
            for method in methods:
                if method in [DetectionMethod.PIVOT_POINTS, DetectionMethod.FIBONACCI,
                            DetectionMethod.CAMARILLA, DetectionMethod.WOODIE]:
                    task = self._detect_pivot_levels(prepared_data, method)
                elif method == DetectionMethod.PEAK_TROUGH:
                    task = self._detect_peak_trough_levels(prepared_data)
                elif method == DetectionMethod.VOLUME_PROFILE:
                    task = self._detect_volume_levels(prepared_data)
                elif method == DetectionMethod.PSYCHOLOGICAL:
                    task = self._detect_psychological_levels(prepared_data)
                elif method == DetectionMethod.ML_BASED:
                    task = self._detect_ml_levels(prepared_data)
                else:
                    continue

                detection_tasks.append(task)

            # Wait for all detections to complete
            if detection_tasks:
                level_results = await asyncio.gather(*detection_tasks, return_exceptions=True)

                for result in level_results:
                    if isinstance(result, list):
                        all_levels.extend(result)
                    elif isinstance(result, Exception):
                        self.logger.warning(f"Detection method failed: {result}")

            # Filter by minimum strength
            filtered_levels = [l for l in all_levels if l.strength >= self.min_level_strength]

            # Detect confluence zones
            confluence_zones = self._detect_confluence_zones(filtered_levels)

            # Sort levels by price
            filtered_levels.sort(key=lambda x: x.price)

            # Calculate detection time
            detection_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = DetectionResult(
                levels=filtered_levels,
                confluence_zones=confluence_zones,
                detection_time_ms=detection_time_ms,
                methods_used=[m.value for m in methods],
                data_points_analyzed=len(prepared_data)
            )

            # Update level manager
            self.level_manager.update_levels(filtered_levels)

            # Cache result
            if self.cache_enabled:
                self._cache_result(cache_key, result)

            # Log performance
            self.logger.info(
                f"Detected {len(filtered_levels)} levels in {detection_time_ms:.1f}ms "
                f"({len(confluence_zones)} confluence zones)"
            )

            # Record detection history
            self.detection_history.append({
                'timestamp': datetime.now(),
                'levels_detected': len(filtered_levels),
                'detection_time_ms': detection_time_ms,
                'methods': [m.value for m in methods]
            })

            return result

        except Exception as e:
            self.logger.error(f"Level detection failed: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare and validate OHLCV data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # Check for required columns
        for col in required_columns:
            if col not in data.columns:
                # Try common variations
                if col == 'open' and 'Open' in data.columns:
                    data = data.rename(columns={'Open': 'open'})
                elif col == 'high' and 'High' in data.columns:
                    data = data.rename(columns={'High': 'high'})
                elif col == 'low' and 'Low' in data.columns:
                    data = data.rename(columns={'Low': 'low'})
                elif col == 'close' and 'Close' in data.columns:
                    data = data.rename(columns={'Close': 'close'})
                elif col == 'volume' and 'Volume' in data.columns:
                    data = data.rename(columns={'Volume': 'volume'})
                else:
                    raise ValueError(f"Required column '{col}' not found in data")

        # Ensure timestamp column
        if 'timestamp' not in data.columns:
            if 'Timestamp' in data.columns:
                data = data.rename(columns={'Timestamp': 'timestamp'})
            elif 'date' in data.columns:
                data = data.rename(columns={'date': 'timestamp'})
            elif 'Date' in data.columns:
                data = data.rename(columns={'Date': 'timestamp'})
            else:
                # Use index as timestamp if it's datetime
                if isinstance(data.index, pd.DatetimeIndex):
                    data['timestamp'] = data.index
                else:
                    # Create sequential timestamp
                    data['timestamp'] = pd.date_range(
                        start='2024-01-01',
                        periods=len(data),
                        freq=self.timeframe
                    )

        # Limit to lookback periods
        if len(data) > self.lookback_periods:
            data = data.tail(self.lookback_periods)

        # Remove any NaN values
        data = data.dropna()

        return data

    async def _detect_pivot_levels(
        self,
        data: pd.DataFrame,
        method: DetectionMethod
    ) -> List[SupportResistanceLevel]:
        """Detect pivot point levels"""
        try:
            if method == DetectionMethod.PIVOT_POINTS:
                pivot_type = "classical"
            elif method == DetectionMethod.FIBONACCI:
                pivot_type = "fibonacci"
            elif method == DetectionMethod.CAMARILLA:
                pivot_type = "camarilla"
            elif method == DetectionMethod.WOODIE:
                pivot_type = "woodie"
            else:
                pivot_type = "classical"

            levels = await self.pivot_calculator.calculate_pivots(data, pivot_type)

            # Convert to SupportResistanceLevel objects
            sr_levels = []
            for level_data in levels:
                level_type = LevelType.SUPPORT if level_data['type'] == 'support' else LevelType.RESISTANCE
                if level_data['type'] == 'pivot':
                    level_type = LevelType.PIVOT

                sr_level = SupportResistanceLevel(
                    price=level_data['price'],
                    level_type=level_type,
                    strength=level_data.get('strength', 0.7),
                    method=method,
                    first_detected=datetime.now(),
                    last_validated=datetime.now(),
                    timeframe=self.timeframe,
                    metadata={'pivot_type': pivot_type}
                )
                sr_levels.append(sr_level)

            return sr_levels

        except Exception as e:
            self.logger.error(f"Pivot detection failed: {e}")
            return []

    async def _detect_peak_trough_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect levels from peaks and troughs"""
        try:
            peaks_troughs = await self.peak_trough_detector.detect(data)

            sr_levels = []
            for pt_data in peaks_troughs:
                level_type = LevelType.RESISTANCE if pt_data['type'] == 'peak' else LevelType.SUPPORT

                sr_level = SupportResistanceLevel(
                    price=pt_data['price'],
                    level_type=level_type,
                    strength=pt_data.get('strength', 0.6),
                    method=DetectionMethod.PEAK_TROUGH,
                    first_detected=datetime.now(),
                    last_validated=datetime.now(),
                    touch_count=pt_data.get('touch_count', 1),
                    timeframe=self.timeframe,
                    metadata={'fractal_dimension': pt_data.get('fractal_dimension')}
                )
                sr_levels.append(sr_level)

            return sr_levels

        except Exception as e:
            self.logger.error(f"Peak/trough detection failed: {e}")
            return []

    async def _detect_volume_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect levels from volume profile"""
        try:
            volume_levels = await self.volume_analyzer.analyze_profile(data)

            sr_levels = []
            for vl_data in volume_levels:
                # Volume levels can be both support and resistance
                level_type = LevelType.SUPPORT if vl_data['position'] == 'below_price' else LevelType.RESISTANCE

                sr_level = SupportResistanceLevel(
                    price=vl_data['price'],
                    level_type=level_type,
                    strength=vl_data.get('strength', 0.8),
                    method=DetectionMethod.VOLUME_PROFILE,
                    first_detected=datetime.now(),
                    last_validated=datetime.now(),
                    timeframe=self.timeframe,
                    metadata={
                        'volume_concentration': vl_data.get('volume_concentration'),
                        'poc': vl_data.get('is_poc', False)  # Point of Control
                    }
                )
                sr_levels.append(sr_level)

            return sr_levels

        except Exception as e:
            self.logger.error(f"Volume profile detection failed: {e}")
            return []

    async def _detect_psychological_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect psychological price levels"""
        try:
            current_price = data['close'].iloc[-1]
            price_range = data['high'].max() - data['low'].min()

            psych_levels = await self.psychological_detector.detect_levels(
                current_price, price_range
            )

            sr_levels = []
            for pl_data in psych_levels:
                # Determine if support or resistance based on current price
                level_type = LevelType.SUPPORT if pl_data['price'] < current_price else LevelType.RESISTANCE

                sr_level = SupportResistanceLevel(
                    price=pl_data['price'],
                    level_type=level_type,
                    strength=pl_data.get('strength', 0.5),
                    method=DetectionMethod.PSYCHOLOGICAL,
                    first_detected=datetime.now(),
                    last_validated=datetime.now(),
                    timeframe=self.timeframe,
                    metadata={'round_number_type': pl_data.get('type')}
                )
                sr_levels.append(sr_level)

            return sr_levels

        except Exception as e:
            self.logger.error(f"Psychological level detection failed: {e}")
            return []

    async def _detect_ml_levels(self, data: pd.DataFrame) -> List[SupportResistanceLevel]:
        """Detect levels using machine learning"""
        try:
            ml_levels = await self.ml_detector.predict_levels(data)

            sr_levels = []
            for ml_data in ml_levels:
                level_type = LevelType.SUPPORT if ml_data['type'] == 'support' else LevelType.RESISTANCE

                sr_level = SupportResistanceLevel(
                    price=ml_data['price'],
                    level_type=level_type,
                    strength=ml_data.get('confidence', 0.7),
                    method=DetectionMethod.ML_BASED,
                    first_detected=datetime.now(),
                    last_validated=datetime.now(),
                    timeframe=self.timeframe,
                    metadata={
                        'model_confidence': ml_data.get('confidence'),
                        'feature_importance': ml_data.get('feature_importance')
                    }
                )
                sr_levels.append(sr_level)

            return sr_levels

        except Exception as e:
            self.logger.error(f"ML level detection failed: {e}")
            return []

    def _detect_confluence_zones(
        self,
        levels: List[SupportResistanceLevel]
    ) -> List[Tuple[float, float, float]]:
        """Detect confluence zones where multiple levels cluster"""
        if not levels:
            return []

        confluence_zones = []
        sorted_levels = sorted(levels, key=lambda x: x.price)

        i = 0
        while i < len(sorted_levels):
            zone_levels = [sorted_levels[i]]
            zone_min = sorted_levels[i].price
            zone_max = sorted_levels[i].price

            # Find all levels within confluence threshold
            j = i + 1
            while j < len(sorted_levels):
                price_diff = abs(sorted_levels[j].price - zone_min) / zone_min
                if price_diff <= self.confluence_threshold:
                    zone_levels.append(sorted_levels[j])
                    zone_max = max(zone_max, sorted_levels[j].price)
                    j += 1
                else:
                    break

            # Create confluence zone if multiple levels found
            if len(zone_levels) >= 2:
                # Calculate zone strength as average of level strengths
                zone_strength = np.mean([l.strength for l in zone_levels])
                # Boost strength based on number of levels
                zone_strength = min(1.0, zone_strength * (1 + 0.1 * len(zone_levels)))

                confluence_zones.append((zone_min, zone_max, zone_strength))

            i = j if j > i + 1 else i + 1

        return confluence_zones

    def get_strong_levels(
        self,
        min_strength: float = 0.7,
        max_levels: int = 10
    ) -> List[SupportResistanceLevel]:
        """Get strongest support/resistance levels"""
        all_levels = self.level_manager.get_all_levels()

        # Filter by strength
        strong_levels = [l for l in all_levels if l.reliability_score >= min_strength]

        # Sort by reliability score
        strong_levels.sort(key=lambda x: x.reliability_score, reverse=True)

        return strong_levels[:max_levels]

    def is_near_level(
        self,
        price: float,
        tolerance_percent: float = 1.0,
        level_type: Optional[LevelType] = None
    ) -> Tuple[bool, Optional[SupportResistanceLevel]]:
        """Check if price is near a support/resistance level"""
        all_levels = self.level_manager.get_all_levels()

        if level_type:
            all_levels = [l for l in all_levels if l.level_type == level_type]

        for level in all_levels:
            price_diff = abs(price - level.price) / level.price * 100
            if price_diff <= tolerance_percent:
                return True, level

        return False, None

    def get_nearest_levels(
        self,
        price: float,
        count: int = 2
    ) -> Dict[str, List[SupportResistanceLevel]]:
        """Get nearest support and resistance levels to given price"""
        all_levels = self.level_manager.get_all_levels()

        supports = [l for l in all_levels if l.level_type == LevelType.SUPPORT and l.price < price]
        resistances = [l for l in all_levels if l.level_type == LevelType.RESISTANCE and l.price > price]

        # Sort supports descending (closest first)
        supports.sort(key=lambda x: x.price, reverse=True)

        # Sort resistances ascending (closest first)
        resistances.sort(key=lambda x: x.price)

        return {
            'support': supports[:count],
            'resistance': resistances[:count]
        }

    def validate_level_break(
        self,
        level: SupportResistanceLevel,
        current_price: float,
        break_threshold_percent: float = 0.5
    ) -> bool:
        """Validate if a level has been broken"""
        if level.level_type == LevelType.SUPPORT:
            # Support is broken if price goes significantly below
            return current_price < level.price * (1 - break_threshold_percent / 100)
        elif level.level_type == LevelType.RESISTANCE:
            # Resistance is broken if price goes significantly above
            return current_price > level.price * (1 + break_threshold_percent / 100)

        return False

    def update_level_touches(self, price_data: pd.DataFrame):
        """Update touch counts for existing levels based on price data"""
        self.level_manager.update_touches(price_data)

    def _generate_cache_key(
        self,
        data: pd.DataFrame,
        methods: List[DetectionMethod]
    ) -> str:
        """Generate cache key for detection parameters"""
        data_hash = hashlib.md5(
            f"{len(data)}_{data['close'].iloc[-1]}_{data['close'].iloc[0]}".encode()
        ).hexdigest()[:8]

        methods_str = "_".join(sorted([m.value for m in methods]))

        return f"{self.symbol}_{self.timeframe}_{data_hash}_{methods_str}"

    def _get_cached_result(self, cache_key: str) -> Optional[DetectionResult]:
        """Get cached detection result if valid"""
        if cache_key in self.cache:
            result, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=self.cache_ttl_minutes):
                return result
            else:
                # Remove expired cache
                del self.cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, result: DetectionResult):
        """Cache detection result"""
        self.cache[cache_key] = (result, datetime.now())

        # Limit cache size
        if len(self.cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k][1]
            )[:20]
            for key in oldest_keys:
                del self.cache[key]

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        if not self.detection_history:
            return {}

        recent_detections = self.detection_history[-50:]

        return {
            'total_detections': len(self.detection_history),
            'average_detection_time_ms': np.mean([d['detection_time_ms'] for d in recent_detections]),
            'average_levels_detected': np.mean([d['levels_detected'] for d in recent_detections]),
            'cache_size': len(self.cache),
            'active_levels_count': len(self.level_manager.get_all_levels()),
            'methods_available': [m.value for m in DetectionMethod if m != DetectionMethod.ALL]
        }

    def export_levels(self, format: str = 'json') -> str:
        """Export detected levels in specified format"""
        all_levels = self.level_manager.get_all_levels()

        if format.lower() == 'json':
            export_data = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'timestamp': datetime.now().isoformat(),
                'levels': [l.to_dict() for l in all_levels],
                'statistics': self.get_detection_stats()
            }
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.cache.clear()
        self.detection_history.clear()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()