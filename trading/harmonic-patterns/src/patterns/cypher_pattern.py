"""
Cypher Pattern - Professional Harmonic Pattern Detector.

Enterprise Implementation: High-precision Cypher pattern detection
using advanced Fibonacci analysis, real-time processing,
and machine learning classification for professional crypto trading.

Cypher Pattern:
- Discovered by Scott Carney in 2011
- Unique structure with BC extension
- Based on precise Fibonacci retracements and extensions
- High accuracy when correctly identified

Fibonacci Ratios for Cypher:
- XA: Initial impulse (any length)
- AB: 38.2% - 61.8% retracement of XA
- BC: 113% - 141.4% extension of XA (unique!)
- CD: 78.6% retracement of XC

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
class CypherFibonacciRatios:
    """Fibonacci ratios for the Cypher pattern.
    
    Immutable data structure for thread-safe operations.
    Specific ratios for unique Cypher pattern.
    """
    # Main Fibonacci levels for Cypher
    AB_RETRACEMENT_MIN: float = 0.382  # Minimum XA retracement
    AB_RETRACEMENT_MAX: float = 0.618  # Maximum XA retracement
    BC_MIN_EXTENSION: float = 1.13     # Minimum XA extension (unique!)
    BC_MAX_EXTENSION: float = 1.414    # Maximum XA extension
    CD_RETRACEMENT: float = 0.786      # XC retracement (critical for Cypher)
    
    # Alternative levels
    BC_GOLDEN_EXTENSION: float = 1.272 # Golden extension
    
    # Allowed deviations for validation (in production systems)
    TOLERANCE: float = 0.06  # 6% tolerance


class CypherPattern(GartleyPattern):
    """
    Professional Cypher Pattern Detector.
    
    High-performance Cypher pattern detection
    for real-time crypto trading systems.
    
    Features:
    - Unique BC extension relative to XA
    - CD retracement from XC instead of BC
    - Advanced confidence scoring
    - Professional risk management
    - Multi-timeframe analysis
    - Volume confirmation
    - ML-enhanced pattern recognition
    
    Example:
        ```python
        detector = CypherPattern(tolerance=0.06, min_confidence=0.75)
        
        # Detect patterns in OHLCV data
        patterns = detector.detect_patterns(ohlcv_data)
        
        # Analyze a specific pattern
        if patterns:
            best_pattern = patterns[0]
            entry_signals = detector.get_entry_signals(best_pattern)
            risk_levels = detector.calculate_risk_levels(best_pattern)
        ```
    """
    
    def __init__(
        self,
        tolerance: float = 0.06,
        min_confidence: float = 0.75,
        enable_volume_analysis: bool = True,
        enable_ml_scoring: bool = True,
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 250
    ):
        """
        Initialize Cypher Pattern Detector.
        
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
        
        # Specific Fibonacci ratios for the Cypher pattern
        self.cypher_fib_ratios = CypherFibonacciRatios()
        
        logger.info(f"CypherPattern initialized with tolerance={tolerance}, "
                   f"min_confidence={min_confidence}")
    
    def _is_valid_cypher_geometry(
        self, 
        x: PatternPoint, 
        a: PatternPoint, 
        b: PatternPoint, 
        c: PatternPoint, 
        d: PatternPoint
    ) -> bool:
        """
        Geometric validation of the Cypher pattern structure.
        
        Cypher has unique characteristics: BC extension from XA, CD retracement from XC.
        """
        try:
            # Basic geometric check as in Gartley
            if not super()._is_valid_gartley_geometry(x, a, b, c, d):
                return False
            
            # Additional checks specific to Cypher
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            xc_distance = abs(c.price - x.price)
            cd_distance = abs(d.price - c.price)
            
            if xa_distance == 0 or xc_distance == 0:
                return False
            
            # AB must be in range 38.2% - 61.8% of XA
            ab_ratio = ab_distance / xa_distance
            if not (self.cypher_fib_ratios.AB_RETRACEMENT_MIN - self.tolerance <= 
                   ab_ratio <= 
                   self.cypher_fib_ratios.AB_RETRACEMENT_MAX + self.tolerance):
                return False
            
            # BC should be an extension of XA (113% - 141.4%) - UNIQUE to Cypher!
            bc_extension = xc_distance / xa_distance
            if not (self.cypher_fib_ratios.BC_MIN_EXTENSION - self.tolerance <= 
                   bc_extension <= 
                   self.cypher_fib_ratios.BC_MAX_EXTENSION + self.tolerance):
                # Check alternative level 127.2%
                if not (abs(bc_extension - self.cypher_fib_ratios.BC_GOLDEN_EXTENSION) <= self.tolerance):
                    return False
            
            # CD should be close to 78.6% retracement of XC (critical for Cypher)
            cd_retracement = cd_distance / xc_distance
            expected_cd_retracement = self.cypher_fib_ratios.CD_RETRACEMENT
            if abs(cd_retracement - expected_cd_retracement) > self.tolerance:
                return False
                        
            return True
                        
        except Exception as e:
            logger.warning(f"Cypher geometry validation error: {str(e)}")
            return False
    
    def _find_cypher_patterns(
        self, 
        pivot_points: List[PatternPoint], 
        data: pd.DataFrame
    ) -> List[Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]]:
        """
        Search for potential 5-point Cypher patterns (X-A-B-C-D).
        
        Optimized pattern matching with Cypher-specific
        constraints for high-probability setups.
        """
        potential_patterns = []
        
        # Need at least 5 points for X-A-B-C-D pattern
        if len(pivot_points) < 5:
            return potential_patterns
        
        # Iterate over all possible 5-point combinations
        for i in range(len(pivot_points) - 4):
            for j in range(i + 1, min(i + self.max_pattern_bars // 8, len(pivot_points) - 3)):
                for k in range(j + 1, min(j + self.max_pattern_bars // 8, len(pivot_points) - 2)):
                    for l in range(k + 1, min(k + self.max_pattern_bars // 8, len(pivot_points) - 1)):
                        for m in range(l + 1, min(l + self.max_pattern_bars // 8, len(pivot_points))):
                            
                            x = pivot_points[i]
                            a = pivot_points[j] 
                            b = pivot_points[k]
                            c = pivot_points[l]
                            d = pivot_points[m]
                            
                            # Check time frame constraints
                            if d.index - x.index > self.max_pattern_bars:
                                continue
                            if d.index - x.index < self.min_pattern_bars:
                                continue
                            
                            # Cypher-specific geometric validation
                            if self._is_valid_cypher_geometry(x, a, b, c, d):
                                potential_patterns.append((x, a, b, c, d))
        
        logger.debug(f"Found {len(potential_patterns)} potential Cypher patterns")
        return potential_patterns
    
    def _validate_and_score_cypher_pattern(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[PatternResult]:
        """
        Full validation and scoring of the Cypher pattern.
        
        Comprehensive validation with Cypher-specific
        scoring algorithms and unique ratio calculations.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Calculate distances
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            bc_distance = abs(c.price - b.price)
            xc_distance = abs(c.price - x.price)  # Important for Cypher!
            cd_distance = abs(d.price - c.price)
            
            # Prevent division by zero
            if xa_distance == 0 or xc_distance == 0 or ab_distance == 0:
                return None
            
            # Cypher-specific ratios
            ab_retracement = ab_distance / xa_distance
            bc_extension_from_xa = xc_distance / xa_distance  # BC as extension of XA
            cd_retracement_from_xc = cd_distance / xc_distance  # CD as retracement from XC
            
            # Standard ratios for compatibility
            bc_ratio = bc_distance / ab_distance if ab_distance > 0 else 0
            cd_ratio = cd_distance / bc_distance if bc_distance > 0 else 0
            ad_distance = abs(d.price - a.price)
            ad_ratio = ad_distance / xa_distance
            
            # Determine pattern type
            pattern_type = PatternType.BULLISH if a.price > x.price else PatternType.BEARISH
            
            # Validate Fibonacci ratios with Cypher-specific criteria
            fib_scores = []
            
            # AB should be 38.2% - 61.8% retracement of XA
            ab_in_range = (self.cypher_fib_ratios.AB_RETRACEMENT_MIN <= 
                          ab_retracement <= 
                          self.cypher_fib_ratios.AB_RETRACEMENT_MAX)
            ab_score = 1.0 if ab_in_range else 0.0
            fib_scores.append(ab_score)
            
            # BC should be 113% - 141.4% extension of XA (UNIQUE to Cypher)
            bc_in_range = (self.cypher_fib_ratios.BC_MIN_EXTENSION <= 
                          bc_extension_from_xa <= 
                          self.cypher_fib_ratios.BC_MAX_EXTENSION)
            # Also check alternative level 127.2%
            bc_golden = abs(bc_extension_from_xa - self.cypher_fib_ratios.BC_GOLDEN_EXTENSION) < self.tolerance
            bc_score = 1.0 if (bc_in_range or bc_golden) else 0.0
            # Bonus for exact hit on 127.2%
            if bc_golden:
                bc_score = min(1.0, bc_score + 0.2)
            fib_scores.append(bc_score)
            
            # CD should be ~78.6% retracement from XC (critical for Cypher!)
            cd_target = self.cypher_fib_ratios.CD_RETRACEMENT
            cd_score = 1.0 - abs(cd_retracement_from_xc - cd_target) / cd_target
            fib_scores.append(max(0, cd_score))
            
            # Additional overall structure assessment
            structure_score = self._assess_cypher_structure(x, a, b, c, d)
            fib_scores.append(structure_score)
            
            # Overall Fibonacci confluence score
            fibonacci_confluence = np.mean(fib_scores)
            
            # Validate pattern
            validation_status = self._determine_validation_status(fib_scores)
            
            if validation_status == PatternValidation.INVALID:
                return None
            
            # Calculate confidence score with Cypher-specific weights
            confidence_score = self._calculate_cypher_confidence_score(
                pattern_points, data, fib_scores, fibonacci_confluence
            )
            
            # Calculate trading levels for Cypher pattern
            entry_price, stop_loss, take_profits = self._calculate_cypher_trading_levels(
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
            
            # Pattern strength analysis with Cypher-specific metrics
            pattern_strength = self._calculate_cypher_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Create result
            pattern_result = PatternResult(
                point_x=x,
                point_a=a,
                point_b=b,
                point_c=c,
                point_d=d,
                pattern_type=pattern_type,
                validation_status=validation_status,
                confidence_score=confidence_score,
                ab_ratio=ab_retracement,
                bc_ratio=bc_extension_from_xa,  # BC as extension of XA
                cd_ratio=cd_retracement_from_xc,  # CD as retracement from XC
                ad_ratio=ad_ratio,
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
            
            return pattern_result
            
        except Exception as e:
            logger.error(f"Error validating Cypher pattern: {str(e)}")
            return None
    
    def _assess_cypher_structure(
        self,
        x: PatternPoint,
        a: PatternPoint,
        b: PatternPoint,
        c: PatternPoint,
        d: PatternPoint
    ) -> float:
        """
        Assess structural integrity of the Cypher pattern.
        
        Checks unique characteristics of the Cypher structure.
        """
        try:
            structure_score = 0.0
            
            # 1. C should be significantly beyond the XA range
            xa_distance = abs(a.price - x.price)
            xc_distance = abs(c.price - x.price)
            
            if xa_distance > 0:
                extension_ratio = xc_distance / xa_distance
                if extension_ratio > 1.1:  # C should be at least 10% further than A from X
                    structure_score += 0.4
                    
            # 2. D should be between X and C (for correct CD retracement)
            pattern_type = PatternType.BULLISH if a.price > x.price else PatternType.BEARISH
            
            if pattern_type == PatternType.BULLISH:
                if x.price < d.price < c.price:
                    structure_score += 0.3
            else:
                if c.price < d.price < x.price:
                    structure_score += 0.3
                    
            # 3. Time sequence should show correct development
            time_intervals = [
                a.index - x.index,
                b.index - a.index, 
                c.index - b.index,
                d.index - c.index
            ]
            
            # Check for reasonable time intervals
            if all(interval >= 2 for interval in time_intervals):  # Minimum 2 bars between points
                structure_score += 0.3
                
            return min(1.0, structure_score)
            
        except Exception as e:
            logger.warning(f"Error assessing Cypher structure: {str(e)}")
            return 0.5
    
    def _calculate_cypher_confidence_score(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        fib_scores: List[float],
        fibonacci_confluence: float
    ) -> float:
        """
        Calculate confidence score specific to the Cypher pattern.
        
        Cypher-specific multi-factor scoring system.
        """
        try:
            scores = []
            weights = []
            
            # 1. Fibonacci accuracy (40% weight)
            scores.append(fibonacci_confluence)
            weights.append(0.40)
            
            # 2. BC extension precision (25% weight - unique to Cypher)
            x, a, b, c, d = pattern_points
            xa_distance = abs(a.price - x.price)
            xc_distance = abs(c.price - x.price)
            if xa_distance > 0:
                bc_extension = xc_distance / xa_distance
                
                # Check for hits on target extension levels
                target_scores = []
                for target in [1.13, 1.272, 1.414]:  # Key levels for Cypher
                    target_score = 1.0 - abs(bc_extension - target) / target
                    target_scores.append(max(0, target_score))
                bc_precision = max(target_scores) if target_scores else 0
                
                scores.append(bc_precision)
                weights.append(0.25)
            else:
                weights[0] += 0.25
            
            # 3. CD retracement precision (20% weight)
            xc_distance = abs(c.price - x.price)
            cd_distance = abs(d.price - c.price)
            if xc_distance > 0:
                cd_retracement = cd_distance / xc_distance
                cd_precision = 1.0 - abs(cd_retracement - self.cypher_fib_ratios.CD_RETRACEMENT) / self.cypher_fib_ratios.CD_RETRACEMENT
                scores.append(max(0, cd_precision))
                weights.append(0.20)
            else:
                weights[0] += 0.20
            
            # 4. Pattern structure quality (10% weight)
            structure_quality = self._assess_cypher_structure(x, a, b, c, d)
            scores.append(structure_quality)
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
            logger.warning(f"Error calculating Cypher confidence score: {str(e)}")
            return 0.0
    
    def _calculate_cypher_trading_levels(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        pattern_type: PatternType
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Calculate trading levels for the Cypher pattern.
        
        Cypher-specific trading levels considering
        unique CD retracement structure.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Entry price at point D (completion)
            entry_price = d.price
            
            # Stop loss calculation for Cypher pattern
            if pattern_type == PatternType.BULLISH:
                # For bullish Cypher: SL below point X
                xa_range = a.price - x.price
                stop_loss = x.price - (xa_range * 0.1)  # 10% buffer below X
                
                # Take Profit levels based on XC retracement
                xc_range = c.price - x.price
                tp1 = entry_price + (xc_range * 0.382)  # 38.2% of XC movement
                tp2 = entry_price + (xc_range * 0.618)  # 61.8% of XC movement
                tp3 = c.price  # Full retracement to point C
                
            else:  # BEARISH
                # For bearish Cypher: SL above point X
                xa_range = x.price - a.price
                stop_loss = x.price + (xa_range * 0.1)  # 10% buffer above X
                
                # Take profit levels (below entry)
                xc_range = x.price - c.price
                tp1 = entry_price - (xc_range * 0.382)  # 38.2% of XC movement
                tp2 = entry_price - (xc_range * 0.618)  # 61.8% of XC movement
                tp3 = c.price  # Full retracement to point C
            
            return entry_price, stop_loss, (tp1, tp2, tp3)
            
        except Exception as e:
            logger.error(f"Error calculating Cypher trading levels: {str(e)}")
            # Fallback levels
            return super()._calculate_trading_levels(pattern_points, pattern_type)
    
    def _calculate_cypher_pattern_strength(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        fibonacci_confluence: float,
        volume_confirmation: float
    ) -> float:
        """Calculate Cypher pattern strength with unique metrics."""
        try:
            # Base calculation as in parent class
            base_strength = super()._calculate_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Cypher-specific additions
            x, a, b, c, d = pattern_points
            
            # BC extension quality bonus
            xa_distance = abs(a.price - x.price)
            xc_distance = abs(c.price - x.price)
            if xa_distance > 0:
                bc_extension = xc_distance / xa_distance
                
                # Bonus for exact hit on key extension levels
                extension_bonus = 0.0
                if abs(bc_extension - 1.272) < 0.05:  # Golden extension
                    extension_bonus = 0.15
                elif abs(bc_extension - 1.13) < 0.05 or abs(bc_extension - 1.414) < 0.05:
                    extension_bonus = 0.1
                    
                base_strength = min(1.0, base_strength + extension_bonus)
            
            # CD retracement precision bonus
            xc_distance = abs(c.price - x.price)
            cd_distance = abs(d.price - c.price)
            if xc_distance > 0:
                cd_retracement = cd_distance / xc_distance
                cd_precision = 1.0 - abs(cd_retracement - self.cypher_fib_ratios.CD_RETRACEMENT) / self.cypher_fib_ratios.CD_RETRACEMENT
                if cd_precision > 0.9:  # High precision
                    precision_bonus = (cd_precision - 0.9) * 0.5
                    base_strength = min(1.0, base_strength + precision_bonus)
            
            return base_strength
            
        except Exception as e:
            logger.warning(f"Error calculating Cypher pattern strength: {str(e)}")
            return 0.0
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[PatternResult]:
        """
        Detect Cypher patterns in OHLCV data.
        
        Optimized Cypher detection with unique structure analysis.
        """
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Generate cache key for optimization
            cache_key = f"cypher_{self._generate_cache_key(data, symbol, timeframe)}"
            
            # Check cache
            if cache_key in self._patterns_cache:
                logger.debug(f"Returning cached Cypher patterns for {cache_key}")
                return self._patterns_cache[cache_key]
            
            # Find significant points (peaks and troughs) using ZigZag
            pivot_points = self._find_pivot_points(data)
            
            if len(pivot_points) < 5:
                logger.warning("Not enough pivot points for Cypher pattern detection")
                return []
            
            # Find potential patterns Cypher
            potential_patterns = self._find_cypher_patterns(pivot_points, data)
            
            # Validate and score patterns
            validated_patterns = []
            for pattern_data in potential_patterns:
                pattern_result = self._validate_and_score_cypher_pattern(
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
            
            logger.info(f"Detected {len(validated_patterns)} valid Cypher patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in detect_patterns: {str(e)}")
            raise RuntimeError(f"Cypher pattern detection failed: {str(e)}") from e
    
    def get_entry_signals(self, pattern: PatternResult) -> Dict[str, any]:
        """
        Generate entry signals for Cypher pattern trading.
        
        Cypher-specific entry analysis with unique structure focus.
        """
        try:
            # Base signals from parent class
            signals = super().get_entry_signals(pattern)
            
            # Cypher-specific modifications
            signals['entry_reason'] = f"Cypher {pattern.pattern_type.value} pattern completion (BC ext: {pattern.bc_ratio:.3f}, CD ret: {pattern.cd_ratio:.3f})"
            
            # Cypher-specific entry conditions
            signals['entry_conditions']['bc_extension_valid'] = (
                self.cypher_fib_ratios.BC_MIN_EXTENSION <= pattern.bc_ratio <= self.cypher_fib_ratios.BC_MAX_EXTENSION or
                abs(pattern.bc_ratio - self.cypher_fib_ratios.BC_GOLDEN_EXTENSION) < self.tolerance
            )
            signals['entry_conditions']['cd_retracement_precise'] = (
                abs(pattern.cd_ratio - self.cypher_fib_ratios.CD_RETRACEMENT) < self.tolerance
            )
            
            # Cypher-specific timing
            signals['timing'] = {
                'immediate': pattern.confidence_score > 0.85,
                'wait_for_confirmation': 0.75 <= pattern.confidence_score <= 0.85,
                'avoid': pattern.confidence_score < 0.75
            }
            
            # Additional information about Cypher pattern
            signals['cypher_specifics'] = {
                'bc_extension_ratio': pattern.bc_ratio,
                'bc_extension_type': self._classify_bc_extension(pattern.bc_ratio),
                'cd_retracement_ratio': pattern.cd_ratio,
                'cd_retracement_precision': 1.0 - abs(pattern.cd_ratio - self.cypher_fib_ratios.CD_RETRACEMENT) / self.cypher_fib_ratios.CD_RETRACEMENT,
                'structure_quality': self._assess_cypher_structure(
                    pattern.point_x, pattern.point_a, pattern.point_b,
                    pattern.point_c, pattern.point_d
                ),
                'pattern_quality': 'HIGH' if pattern.confidence_score > 0.80 else 'MEDIUM'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Cypher entry signals: {str(e)}")
            return {}
    
    def _classify_bc_extension(self, bc_ratio: float) -> str:
        """Classify BC extension type for Cypher."""
        if abs(bc_ratio - 1.272) < 0.05:
            return "GOLDEN_127.2"
        elif abs(bc_ratio - 1.414) < 0.05:
            return "SQRT2_141.4"
        elif abs(bc_ratio - 1.13) < 0.05:
            return "MINIMUM_113"
        elif 1.13 <= bc_ratio <= 1.414:
            return "VALID_RANGE"
        else:
            return "OUT_OF_RANGE"


# Export main components
__all__ = [
    'CypherPattern',
    'CypherFibonacciRatios'
]