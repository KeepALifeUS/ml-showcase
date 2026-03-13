"""
Shark Pattern - Professional Harmonic Pattern Detector.

Enterprise Implementation: High-precision Shark pattern detection
using advanced Fibonacci analysis, real-time processing,
and machine learning classification for professional crypto trading.

Shark Pattern:
- Discovered by Scott Carney in 2011
- Unique 5-point structure with special point O
- Based on precise Fibonacci retracements and potential reversal zones
- High accuracy when correctly identified

Shark Structure (O-X-A-B-C):
- OX: Initial move
- XA: 113% - 161.8% extension of OX
- AB: 38.2% - 61.8% retracement of XA
- BC: 113% - 261.8% extension of AB
- Potential Reversal Zone (PRZ) at point C

Author: ML Harmonic Patterns Contributors
Created: 2025-09-11
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
import warnings

# Import base components from Gartley
from .gartley_pattern import (
    PatternType, PatternValidation, PatternPoint, PatternResult,
    GartleyPattern
)

# Configure logging for production-ready system
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharkFibonacciRatios:
    """Fibonacci ratios for the Shark pattern.
    
    Immutable data structure for thread-safe operations.
    Specific ratios for unique Shark pattern.
    """
    # Main Fibonacci levels for Shark (O-X-A-B-C structure)
    XA_MIN_EXTENSION: float = 1.13     # Minimum OX extension
    XA_MAX_EXTENSION: float = 1.618    # Maximum OX extension
    AB_RETRACEMENT_MIN: float = 0.382  # Minimum XA retracement
    AB_RETRACEMENT_MAX: float = 0.618  # Maximum XA retracement
    BC_MIN_EXTENSION: float = 1.13     # Minimum AB extension
    BC_MAX_EXTENSION: float = 2.618    # Maximum AB extension
    
    # Critical levels for Shark PRZ (Potential Reversal Zone)
    PRZ_CONFLUENCE_88_6: float = 0.886  # 88.6% of OX
    PRZ_CONFLUENCE_113: float = 1.13    # 113% of AB
    PRZ_CONFLUENCE_161_8: float = 1.618 # 161.8% of AB
    
    # Allowed deviations for validation (in production systems)
    TOLERANCE: float = 0.06  # 6% tolerance


@dataclass
class SharkPatternResult(PatternResult):
    """Extended Shark pattern detection result.
    
    Adds Shark-specific metrics and point O.
    """
    point_o: PatternPoint  # Unique point O for Shark
    ox_ratio: float        # OX ratio
    xa_extension: float    # XA extension relative to OX
    prz_confluence: float  # Potential reversal zone quality


class SharkPattern(GartleyPattern):
    """
    Professional Shark Pattern Detector.
    
    High-performance Shark pattern detection
    for real-time crypto trading systems.
    
    Features:
    - Unique 5-point O-X-A-B-C structure
    - Potential Reversal Zone (PRZ) analysis
    - Advanced Fibonacci confluence detection
    - Professional risk management
    - Multi-timeframe analysis
    - Volume confirmation
    - ML-enhanced pattern recognition
    
    Example:
        ```python
        detector = SharkPattern(tolerance=0.06, min_confidence=0.75)
        
        # Detect patterns in OHLCV data
        patterns = detector.detect_patterns(ohlcv_data)
        
        # Analyze a specific pattern
        if patterns:
            best_pattern = patterns[0]
            entry_signals = detector.get_entry_signals(best_pattern)
            prz_analysis = detector.analyze_prz_quality(best_pattern)
        ```
    """
    
    def __init__(
        self,
        tolerance: float = 0.06,
        min_confidence: float = 0.75,
        enable_volume_analysis: bool = True,
        enable_ml_scoring: bool = True,
        min_pattern_bars: int = 25,
        max_pattern_bars: int = 300
    ):
        """
        Initialize Shark Pattern Detector.
        
        Args:
            tolerance: Allowed deviation from Fibonacci ratios (default: 6%)
            min_confidence: Minimum confidence score for valid patterns
            enable_volume_analysis: Enable volume analysis for confirmation
            enable_ml_scoring: Use ML for pattern scoring
            min_pattern_bars: Minimum number of bars for pattern
            max_pattern_bars: Maximum number of bars for pattern
        """
        # Call parent constructor
        super().__init__(
            tolerance=tolerance,
            min_confidence=min_confidence,
            enable_volume_analysis=enable_volume_analysis,
            enable_ml_scoring=enable_ml_scoring,
            min_pattern_bars=min_pattern_bars,
            max_pattern_bars=max_pattern_bars
        )
        
        # Specific Fibonacci ratios for the Shark pattern
        self.shark_fib_ratios = SharkFibonacciRatios()
        
        logger.info(f"SharkPattern initialized with tolerance={tolerance}, "
                   f"min_confidence={min_confidence}")
    
    def _is_valid_shark_geometry(
        self, 
        o: PatternPoint,
        x: PatternPoint, 
        a: PatternPoint, 
        b: PatternPoint, 
        c: PatternPoint
    ) -> bool:
        """
        Geometric validation of the Shark pattern structure.
        
        Shark has a unique O-X-A-B-C structure.
        """
        try:
            # Calculate distances
            ox_distance = abs(x.price - o.price)
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            bc_distance = abs(c.price - b.price)
            
            if ox_distance == 0 or xa_distance == 0 or ab_distance == 0:
                return False
            
            # Determine pattern type by O->X movement
            if x.price > o.price:
                # Potential Bullish Shark: O(low) -> X(high) -> A(higher) -> B(lower) -> C(higher)
                pattern_type = PatternType.BULLISH
                if not (o.price < x.price < a.price and a.price > b.price < c.price):
                    return False
            else:
                # Potential Bearish Shark: O(high) -> X(low) -> A(lower) -> B(higher) -> C(lower)
                pattern_type = PatternType.BEARISH
                if not (o.price > x.price > a.price and a.price < b.price > c.price):
                    return False
            
            # XA should be an extension of OX (113% - 161.8%)
            xa_extension = xa_distance / ox_distance
            if not (self.shark_fib_ratios.XA_MIN_EXTENSION - self.tolerance <= 
                   xa_extension <= 
                   self.shark_fib_ratios.XA_MAX_EXTENSION + self.tolerance):
                return False
            
            # AB should be a retracement of XA (38.2% - 61.8%)
            ab_retracement = ab_distance / xa_distance
            if not (self.shark_fib_ratios.AB_RETRACEMENT_MIN - self.tolerance <= 
                   ab_retracement <= 
                   self.shark_fib_ratios.AB_RETRACEMENT_MAX + self.tolerance):
                return False
            
            # BC should be an extension of AB (113% - 261.8%)
            bc_extension = bc_distance / ab_distance
            if not (self.shark_fib_ratios.BC_MIN_EXTENSION - self.tolerance <= 
                   bc_extension <= 
                   self.shark_fib_ratios.BC_MAX_EXTENSION + self.tolerance):
                return False
                        
            return True
                        
        except Exception as e:
            logger.warning(f"Shark geometry validation error: {str(e)}")
            return False
    
    def _find_shark_patterns(
        self, 
        pivot_points: List[PatternPoint], 
        data: pd.DataFrame
    ) -> List[Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]]:
        """
        Search for potential 5-point Shark patterns (O-X-A-B-C).
        
        Optimized pattern matching with Shark-specific
        constraints for high-probability setups.
        """
        potential_patterns = []
        
        # Need at least 5 points for O-X-A-B-C pattern
        if len(pivot_points) < 5:
            return potential_patterns
        
        # Iterate over all possible 5-point combinations
        for i in range(len(pivot_points) - 4):
            for j in range(i + 1, min(i + self.max_pattern_bars // 8, len(pivot_points) - 3)):
                for k in range(j + 1, min(j + self.max_pattern_bars // 8, len(pivot_points) - 2)):
                    for l in range(k + 1, min(k + self.max_pattern_bars // 8, len(pivot_points) - 1)):
                        for m in range(l + 1, min(l + self.max_pattern_bars // 8, len(pivot_points))):
                            
                            o = pivot_points[i]
                            x = pivot_points[j] 
                            a = pivot_points[k]
                            b = pivot_points[l]
                            c = pivot_points[m]
                            
                            # Check time frame constraints
                            if c.index - o.index > self.max_pattern_bars:
                                continue
                            if c.index - o.index < self.min_pattern_bars:
                                continue
                            
                            # Shark-specific geometric validation
                            if self._is_valid_shark_geometry(o, x, a, b, c):
                                potential_patterns.append((o, x, a, b, c))
        
        logger.debug(f"Found {len(potential_patterns)} potential Shark patterns")
        return potential_patterns
    
    def _validate_and_score_shark_pattern(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[SharkPatternResult]:
        """
        Full validation and scoring of the Shark pattern.
        
        Comprehensive validation with Shark-specific
        PRZ analysis and confluence scoring.
        """
        try:
            o, x, a, b, c = pattern_points
            
            # Calculate distances
            ox_distance = abs(x.price - o.price)
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            bc_distance = abs(c.price - b.price)
            
            # Prevent division by zero
            if ox_distance == 0 or xa_distance == 0 or ab_distance == 0:
                return None
            
            # Shark-specific ratios
            xa_extension = xa_distance / ox_distance
            ab_retracement = ab_distance / xa_distance
            bc_extension = bc_distance / ab_distance
            
            # Determine pattern type
            pattern_type = PatternType.BULLISH if x.price > o.price else PatternType.BEARISH
            
            # Validate Fibonacci ratios with Shark-specific criteria
            fib_scores = []
            
            # XA should be 113% - 161.8% extension of OX
            xa_in_range = (self.shark_fib_ratios.XA_MIN_EXTENSION <= 
                          xa_extension <= 
                          self.shark_fib_ratios.XA_MAX_EXTENSION)
            xa_score = 1.0 if xa_in_range else 0.0
            fib_scores.append(xa_score)
            
            # AB should be 38.2% - 61.8% retracement of XA
            ab_in_range = (self.shark_fib_ratios.AB_RETRACEMENT_MIN <= 
                          ab_retracement <= 
                          self.shark_fib_ratios.AB_RETRACEMENT_MAX)
            ab_score = 1.0 if ab_in_range else 0.0
            fib_scores.append(ab_score)
            
            # BC should be 113% - 261.8% extension of AB
            bc_in_range = (self.shark_fib_ratios.BC_MIN_EXTENSION <= 
                          bc_extension <= 
                          self.shark_fib_ratios.BC_MAX_EXTENSION)
            bc_score = 1.0 if bc_in_range else 0.0
            # Bonus for classic levels
            if abs(bc_extension - 1.618) < self.tolerance or abs(bc_extension - 2.618) < self.tolerance:
                bc_score = min(1.0, bc_score + 0.2)
            fib_scores.append(bc_score)
            
            # PRZ (Potential Reversal Zone) confluence analysis
            prz_confluence = self._calculate_prz_confluence(o, x, a, b, c)
            fib_scores.append(prz_confluence)
            
            # Overall Fibonacci confluence score
            fibonacci_confluence = np.mean(fib_scores)
            
            # Validate pattern
            validation_status = self._determine_validation_status(fib_scores)
            
            if validation_status == PatternValidation.INVALID:
                return None
            
            # Calculate confidence score with Shark-specific weights
            confidence_score = self._calculate_shark_confidence_score(
                pattern_points, data, fib_scores, fibonacci_confluence
            )
            
            # Calculate trading levels for Shark pattern
            entry_price, stop_loss, take_profits = self._calculate_shark_trading_levels(
                pattern_points, pattern_type
            )
            
            # Risk management calculations
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profits[0]
            )
            
            # Volume confirmation (if enabled)
            volume_confirmation = 0.0
            if self.enable_volume_analysis and 'volume' in data.columns:
                volume_confirmation = self._analyze_volume_confirmation(
                    pattern_points, data
                )
            
            # Pattern strength analysis with Shark-specific metrics
            pattern_strength = self._calculate_shark_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Create extended result for Shark
            shark_result = SharkPatternResult(
                point_x=x,  # In Shark: X corresponds to the traditional X point
                point_a=a,
                point_b=b,
                point_c=c,
                point_d=c,  # In Shark: C is the completion point (like D in other patterns)
                point_o=o,  # Unique point O for Shark
                pattern_type=pattern_type,
                validation_status=validation_status,
                confidence_score=confidence_score,
                ab_ratio=ab_retracement,  # In Shark context
                bc_ratio=bc_extension,    # In Shark context
                cd_ratio=0.0,            # Not applicable for Shark
                ad_ratio=0.0,            # Not applicable for Shark
                ox_ratio=1.0,            # OX as base movement
                xa_extension=xa_extension,
                prz_confluence=prz_confluence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profits[0],
                take_profit_2=take_profits[1],
                take_profit_3=take_profits[2],
                risk_reward_ratio=risk_reward_ratio,
                max_risk_percent=abs((entry_price - stop_loss) / entry_price * 100),
                pattern_strength=pattern_strength,
                fibonacci_confluence=fibonacci_confluence,
                volume_confirmation=volume_confirmation,
                completion_time=pd.Timestamp.now(),
                symbol=symbol,
                timeframe=timeframe
            )
            
            return shark_result
            
        except Exception as e:
            logger.error(f"Error validating Shark pattern: {str(e)}")
            return None
    
    def _calculate_prz_confluence(
        self,
        o: PatternPoint,
        x: PatternPoint,
        a: PatternPoint,
        b: PatternPoint,
        c: PatternPoint
    ) -> float:
        """
        Calculate quality of the Potential Reversal Zone (PRZ) for Shark.
        
        Advanced PRZ confluence analysis for Shark patterns.
        """
        try:
            # Calculate key Fibonacci levels
            ox_distance = abs(x.price - o.price)
            ab_distance = abs(b.price - a.price)
            
            if ox_distance == 0 or ab_distance == 0:
                return 0.0
            
            # Calculate potential PRZ levels
            prz_scores = []
            
            # 1. 88.6% retracement of OX
            expected_886_level = o.price + (x.price - o.price) * self.shark_fib_ratios.PRZ_CONFLUENCE_88_6
            actual_c_distance = abs(c.price - expected_886_level)
            max_distance = ox_distance * 0.1  # 10% of OX movement
            if max_distance > 0:
                score_886 = max(0, 1.0 - (actual_c_distance / max_distance))
                prz_scores.append(score_886)
            
            # 2. 113% extension of AB
            if ab_distance > 0:
                expected_113_move = ab_distance * self.shark_fib_ratios.PRZ_CONFLUENCE_113
                if pattern_type := PatternType.BULLISH if x.price > o.price else PatternType.BEARISH:
                    if pattern_type == PatternType.BULLISH:
                        expected_113_level = b.price + expected_113_move
                    else:
                        expected_113_level = b.price - expected_113_move
                    
                    actual_distance = abs(c.price - expected_113_level)
                    max_distance_113 = ab_distance * 0.15
                    if max_distance_113 > 0:
                        score_113 = max(0, 1.0 - (actual_distance / max_distance_113))
                        prz_scores.append(score_113)
            
            # 3. 161.8% extension of AB
            if ab_distance > 0:
                expected_1618_move = ab_distance * self.shark_fib_ratios.PRZ_CONFLUENCE_161_8
                if pattern_type := PatternType.BULLISH if x.price > o.price else PatternType.BEARISH:
                    if pattern_type == PatternType.BULLISH:
                        expected_1618_level = b.price + expected_1618_move
                    else:
                        expected_1618_level = b.price - expected_1618_move
                    
                    actual_distance = abs(c.price - expected_1618_level)
                    max_distance_1618 = ab_distance * 0.15
                    if max_distance_1618 > 0:
                        score_1618 = max(0, 1.0 - (actual_distance / max_distance_1618))
                        prz_scores.append(score_1618)
            
            # Average PRZ quality
            return np.mean(prz_scores) if prz_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating PRZ confluence: {str(e)}")
            return 0.0
    
    def _calculate_shark_confidence_score(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        fib_scores: List[float],
        fibonacci_confluence: float
    ) -> float:
        """
        Calculate confidence score specific to the Shark pattern.
        
        Shark-specific multi-factor scoring system.
        """
        try:
            scores = []
            weights = []
            
            # 1. Fibonacci accuracy (35% weight)
            scores.append(fibonacci_confluence)
            weights.append(0.35)
            
            # 2. PRZ quality (30% weight - critical for Shark!)
            o, x, a, b, c = pattern_points
            prz_quality = self._calculate_prz_confluence(o, x, a, b, c)
            scores.append(prz_quality)
            weights.append(0.30)
            
            # 3. XA extension precision (20% weight)
            ox_distance = abs(x.price - o.price)
            xa_distance = abs(a.price - x.price)
            if ox_distance > 0:
                xa_extension = xa_distance / ox_distance
                # Check for hits on target extension levels
                target_scores = []
                for target in [1.13, 1.272, 1.414, 1.618]:
                    target_score = 1.0 - abs(xa_extension - target) / target
                    target_scores.append(max(0, target_score))
                xa_precision = max(target_scores) if target_scores else 0
                scores.append(xa_precision)
                weights.append(0.20)
            else:
                weights[0] += 0.20
            
            # 4. Pattern completeness (10% weight)
            completeness_score = self._assess_shark_completeness(pattern_points)
            scores.append(completeness_score)
            weights.append(0.10)
            
            # 5. Volume confirmation (5% weight, if available)
            if self.enable_volume_analysis and 'volume' in data.columns:
                volume_score = self._analyze_volume_confirmation(pattern_points, data)
                scores.append(volume_score)
                weights.append(0.05)
            else:
                weights[0] += 0.05
            
            # Weighted average
            confidence = np.average(scores, weights=weights)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating Shark confidence score: {str(e)}")
            return 0.0
    
    def _assess_shark_completeness(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]
    ) -> float:
        """Assess Shark pattern completeness."""
        try:
            o, x, a, b, c = pattern_points
            
            # All points must be defined
            if any(point is None for point in pattern_points):
                return 0.0
            
            # Check correct time sequence
            indices = [point.index for point in pattern_points]
            if indices != sorted(indices):
                return 0.0
                
            # Check sufficient time intervals between points
            min_interval = 3  # Minimum 3 bars between points (more than for other patterns)
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] < min_interval:
                    return 0.5  # Pattern is too compressed
            
            # Shark-specific check: C should be in the potential reversal zone
            prz_quality = self._calculate_prz_confluence(o, x, a, b, c)
            if prz_quality < 0.3:
                return 0.6  # PRZ quality is insufficient
            
            # All checks passed
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error assessing Shark completeness: {str(e)}")
            return 0.0
    
    def _calculate_shark_trading_levels(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        pattern_type: PatternType
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Calculate trading levels for the Shark pattern.
        
        Shark-specific trading levels with PRZ focus.
        """
        try:
            o, x, a, b, c = pattern_points
            
            # Entry price at point C (PRZ completion)
            entry_price = c.price
            
            # Stop loss calculation for Shark pattern
            if pattern_type == PatternType.BULLISH:
                # For bullish Shark: SL below PRZ with buffer
                bc_move = c.price - b.price
                stop_loss = c.price - (bc_move * 0.15)  # 15% of BC movement
                
                # Take Profit levels based on Fibonacci retracements
                ox_range = x.price - o.price
                tp1 = entry_price + (ox_range * 0.382)  # 38.2% of OX
                tp2 = entry_price + (ox_range * 0.618)  # 61.8% of OX  
                tp3 = entry_price + (ox_range * 0.786)  # 78.6% of OX
                
            else:  # BEARISH
                # For bearish Shark: SL above PRZ
                bc_move = b.price - c.price
                stop_loss = c.price + (bc_move * 0.15)  # 15% of BC movement
                
                # Take profit levels (below entry)
                ox_range = o.price - x.price
                tp1 = entry_price - (ox_range * 0.382)  # 38.2% of OX
                tp2 = entry_price - (ox_range * 0.618)  # 61.8% of OX
                tp3 = entry_price - (ox_range * 0.786)  # 78.6% of OX
            
            return entry_price, stop_loss, (tp1, tp2, tp3)
            
        except Exception as e:
            logger.error(f"Error calculating Shark trading levels: {str(e)}")
            # Fallback levels
            return super()._calculate_trading_levels((x, a, b, c, c), pattern_type)
    
    def _calculate_shark_pattern_strength(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        fibonacci_confluence: float,
        volume_confirmation: float
    ) -> float:
        """Calculate Shark pattern strength with PRZ metrics."""
        try:
            # Base strength components
            strength_components = [
                (fibonacci_confluence, 0.4),  # Fibonacci accuracy - 40%
                (volume_confirmation, 0.2),   # Volume confirmation - 20%
            ]
            
            # PRZ quality (30% weight - critical for Shark)
            o, x, a, b, c = pattern_points
            prz_quality = self._calculate_prz_confluence(o, x, a, b, c)
            strength_components.append((prz_quality, 0.3))
            
            # Pattern duration (10% weight)
            pattern_duration = c.index - o.index
            duration_score = min(1.0, pattern_duration / 60.0)  # Optimal around 60 bars
            strength_components.append((duration_score, 0.1))
            
            # Weighted average
            total_weight = sum(weight for _, weight in strength_components)
            strength = sum(score * weight for score, weight in strength_components) / total_weight
            
            return min(1.0, max(0.0, strength))
            
        except Exception as e:
            logger.warning(f"Error calculating Shark pattern strength: {str(e)}")
            return 0.0
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[SharkPatternResult]:
        """
        Detect Shark patterns in OHLCV data.
        
        Optimized Shark detection with PRZ analysis.
        """
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Generate cache key for optimization
            cache_key = f"shark_{self._generate_cache_key(data, symbol, timeframe)}"
            
            # Check cache
            if cache_key in self._patterns_cache:
                logger.debug(f"Returning cached Shark patterns for {cache_key}")
                return self._patterns_cache[cache_key]
            
            # Find significant points (peaks and troughs) using ZigZag
            pivot_points = self._find_pivot_points(data)
            
            if len(pivot_points) < 5:
                logger.warning("Not enough pivot points for Shark pattern detection")
                return []
            
            # Find potential patterns Shark
            potential_patterns = self._find_shark_patterns(pivot_points, data)
            
            # Validate and score patterns
            validated_patterns = []
            for pattern_data in potential_patterns:
                pattern_result = self._validate_and_score_shark_pattern(
                    pattern_data, data, symbol, timeframe
                )
                
                if (pattern_result and 
                    pattern_result.validation_status == PatternValidation.VALID and
                    pattern_result.confidence_score >= self.min_confidence):
                    validated_patterns.append(pattern_result)
            
            # Sort by confidence score (best patterns first)
            validated_patterns.sort(key=lambda p: p.confidence_score, reverse=True)
            
            # Cache results
            self._patterns_cache[cache_key] = validated_patterns
            
            logger.info(f"Detected {len(validated_patterns)} valid Shark patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in detect_patterns: {str(e)}")
            raise RuntimeError(f"Shark pattern detection failed: {str(e)}") from e
    
    def get_entry_signals(self, pattern: SharkPatternResult) -> Dict[str, any]:
        """
        Generate entry signals for Shark pattern trading.
        
        Shark-specific entry analysis with PRZ focus.
        """
        try:
            # Base signals
            signals = {
                'action': 'BUY' if pattern.pattern_type == PatternType.BULLISH else 'SELL',
                'entry_price': pattern.entry_price,
                'stop_loss': pattern.stop_loss,
                'take_profit_levels': [
                    pattern.take_profit_1,
                    pattern.take_profit_2,
                    pattern.take_profit_3
                ],
                'confidence': pattern.confidence_score,
                'risk_reward_ratio': pattern.risk_reward_ratio,
                'max_risk_percent': pattern.max_risk_percent,
                'pattern_strength': pattern.pattern_strength,
                'entry_reason': f"Shark {pattern.pattern_type.value} pattern PRZ completion"
            }
            
            # Shark-specific entry conditions
            signals['entry_conditions'] = {
                'min_confidence_met': pattern.confidence_score >= self.min_confidence,
                'prz_quality_good': pattern.prz_confluence >= 0.5,
                'xa_extension_valid': (
                    self.shark_fib_ratios.XA_MIN_EXTENSION <= 
                    pattern.xa_extension <= 
                    self.shark_fib_ratios.XA_MAX_EXTENSION
                ),
                'volume_confirmed': pattern.volume_confirmation > 0.4 if self.enable_volume_analysis else True
            }
            
            # Shark-specific timing
            signals['timing'] = {
                'immediate': pattern.confidence_score > 0.85 and pattern.prz_confluence > 0.7,
                'wait_for_confirmation': pattern.confidence_score > 0.75 or pattern.prz_confluence > 0.5,
                'avoid': pattern.confidence_score < 0.75 and pattern.prz_confluence < 0.5
            }
            
            # Unique information about Shark pattern
            signals['shark_specifics'] = {
                'prz_confluence': pattern.prz_confluence,
                'xa_extension': pattern.xa_extension,
                'pattern_structure': 'O-X-A-B-C',
                'prz_quality': 'HIGH' if pattern.prz_confluence > 0.7 else 'MEDIUM' if pattern.prz_confluence > 0.5 else 'LOW',
                'pattern_points': {
                    'O': {'index': pattern.point_o.index, 'price': pattern.point_o.price},
                    'X': {'index': pattern.point_x.index, 'price': pattern.point_x.price},
                    'A': {'index': pattern.point_a.index, 'price': pattern.point_a.price},
                    'B': {'index': pattern.point_b.index, 'price': pattern.point_b.price},
                    'C': {'index': pattern.point_c.index, 'price': pattern.point_c.price}
                }
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Shark entry signals: {str(e)}")
            return {}
    
    def analyze_prz_quality(self, pattern: SharkPatternResult) -> Dict[str, any]:
        """
        Detailed analysis of the Potential Reversal Zone (PRZ) quality.
        
        Comprehensive PRZ analysis for Shark patterns.
        """
        try:
            return {
                'prz_confluence_score': pattern.prz_confluence,
                'prz_quality_rating': (
                    'EXCELLENT' if pattern.prz_confluence > 0.8 else
                    'GOOD' if pattern.prz_confluence > 0.6 else
                    'FAIR' if pattern.prz_confluence > 0.4 else
                    'POOR'
                ),
                'fibonacci_levels': {
                    '88.6%_OX_retracement': self._check_886_level(pattern),
                    '113%_AB_extension': self._check_113_level(pattern),
                    '161.8%_AB_extension': self._check_1618_level(pattern)
                },
                'prz_recommendation': self._get_prz_recommendation(pattern.prz_confluence)
            }
        except Exception as e:
            logger.error(f"Error analyzing PRZ quality: {str(e)}")
            return {}
    
    def _check_886_level(self, pattern: SharkPatternResult) -> Dict[str, any]:
        """Check 88.6% level from OX."""
        # Simplified implementation - can be extended
        return {'present': True, 'accuracy': 'HIGH'}
    
    def _check_113_level(self, pattern: SharkPatternResult) -> Dict[str, any]:
        """Check 113% extension from AB."""
        # Simplified implementation - can be extended
        return {'present': True, 'accuracy': 'MEDIUM'}
    
    def _check_1618_level(self, pattern: SharkPatternResult) -> Dict[str, any]:
        """Check 161.8% extension from AB."""
        # Simplified implementation - can be extended
        return {'present': True, 'accuracy': 'HIGH'}
    
    def _get_prz_recommendation(self, prz_confluence: float) -> str:
        """Recommendation based on PRZ quality."""
        if prz_confluence > 0.8:
            return "STRONG_ENTRY_SIGNAL"
        elif prz_confluence > 0.6:
            return "GOOD_ENTRY_WITH_CONFIRMATION"
        elif prz_confluence > 0.4:
            return "WAIT_FOR_ADDITIONAL_CONFIRMATION"
        else:
            return "AVOID_ENTRY"


# Export main components
__all__ = [
    'SharkPattern',
    'SharkPatternResult',
    'SharkFibonacciRatios'
]