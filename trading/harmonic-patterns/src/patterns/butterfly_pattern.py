"""
Butterfly Pattern - Professional Harmonic Pattern Detector.

Enterprise Implementation: High-precision Butterfly pattern detection
using advanced Fibonacci analysis, real-time processing,
and machine learning classification for professional crypto trading.

Butterfly Pattern:
- Discovered by Scott Carney in 2001
- Aggressive pattern with AD extension
- Based on precise Fibonacci retracements and extensions
- High returns when correctly identified

Fibonacci Ratios for Butterfly:
- XA: Initial impulse (any length)
- AB: 78.6% retracement of XA
- BC: 38.2% - 88.6% retracement of AB
- CD: 161.8% - 261.8% extension of BC
- AD: 127.2% or 161.8% extension of XA (key distinguishing feature)

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
class ButterflyFibonacciRatios:
    """Fibonacci ratios for the Butterfly pattern.
    
    Immutable data structure for thread-safe operations.
    Specific ratios for aggressive Butterfly pattern.
    """
    # Main Fibonacci levels for Butterfly
    AB_RETRACEMENT: float = 0.786      # 78.6% retracement of XA (key level)
    BC_MIN_RETRACEMENT: float = 0.382  # Minimum AB retracement
    BC_MAX_RETRACEMENT: float = 0.886  # Maximum AB retracement
    CD_MIN_EXTENSION: float = 1.618    # Minimum BC extension
    CD_MAX_EXTENSION: float = 2.618    # Maximum BC extension
    AD_EXTENSION_1: float = 1.272      # First XA extension variant
    AD_EXTENSION_2: float = 1.618      # Second XA extension variant
    
    # Allowed deviations for validation (in production systems)
    TOLERANCE: float = 0.05  # 5% tolerance for each ratio


class ButterflyPattern(GartleyPattern):
    """
    Professional Butterfly Pattern Detector.
    
    High-performance Butterfly pattern detection
    for real-time crypto trading systems.
    
    Features:
    - High-precision Fibonacci validation specific to Butterfly
    - Real-time pattern scanning
    - Advanced confidence scoring with extended AD analysis
    - Professional risk management
    - Multi-timeframe analysis
    - Volume confirmation
    - ML-enhanced pattern recognition
    
    Example:
        ```python
        detector = ButterflyPattern(tolerance=0.05, min_confidence=0.75)
        
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
        tolerance: float = 0.05,
        min_confidence: float = 0.70,
        enable_volume_analysis: bool = True,
        enable_ml_scoring: bool = True,
        min_pattern_bars: int = 25,  # Increased for Butterfly
        max_pattern_bars: int = 300  # Increased for extended patterns
    ):
        """
        Initialize Butterfly Pattern Detector.
        
        Args:
            tolerance: Allowed deviation from Fibonacci ratios (default: 5%)
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
        
        # Specific Fibonacci ratios for the Butterfly pattern
        self.butterfly_fib_ratios = ButterflyFibonacciRatios()
        
        logger.info(f"ButterflyPattern initialized with tolerance={tolerance}, "
                   f"min_confidence={min_confidence}")
    
    def _is_valid_butterfly_geometry(
        self, 
        x: PatternPoint, 
        a: PatternPoint, 
        b: PatternPoint, 
        c: PatternPoint, 
        d: PatternPoint
    ) -> bool:
        """
        Geometric validation of the Butterfly pattern structure.
        
        Butterfly pattern has unique characteristics in AD extension.
        """
        try:
            # Basic geometric check as in Gartley
            if not super()._is_valid_gartley_geometry(x, a, b, c, d):
                return False
            
            # Additional checks specific to Butterfly
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            ad_distance = abs(d.price - a.price)
            
            if xa_distance == 0:
                return False
            
            # AB should be close to 78.6% of XA (critical for Butterfly)
            ab_ratio = ab_distance / xa_distance
            expected_ab_ratio = self.butterfly_fib_ratios.AB_RETRACEMENT
            if abs(ab_ratio - expected_ab_ratio) > self.tolerance:
                return False
            
            # AD should be an extension of XA (127.2% or 161.8%)
            ad_ratio = ad_distance / xa_distance
            target_1 = self.butterfly_fib_ratios.AD_EXTENSION_1
            target_2 = self.butterfly_fib_ratios.AD_EXTENSION_2
            
            if not (abs(ad_ratio - target_1) <= self.tolerance or 
                   abs(ad_ratio - target_2) <= self.tolerance):
                return False
            
            # Additional check: D must be OUTSIDE the X-A range
            if pattern_type := PatternType.BULLISH if a.price > x.price else PatternType.BEARISH:
                if pattern_type == PatternType.BULLISH:
                    # For bullish: D should be above A
                    if d.price <= a.price:
                        return False
                else:
                    # For bearish: D should be below A
                    if d.price >= a.price:
                        return False
                        
            return True
                        
        except Exception as e:
            logger.warning(f"Butterfly geometry validation error: {str(e)}")
            return False
    
    def _find_butterfly_patterns(
        self, 
        pivot_points: List[PatternPoint], 
        data: pd.DataFrame
    ) -> List[Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]]:
        """
        Search for potential 5-point Butterfly patterns (X-A-B-C-D).
        
        Optimized pattern matching with Butterfly-specific
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
                            
                            # Butterfly-specific geometric validation
                            if self._is_valid_butterfly_geometry(x, a, b, c, d):
                                potential_patterns.append((x, a, b, c, d))
        
        logger.debug(f"Found {len(potential_patterns)} potential Butterfly patterns")
        return potential_patterns
    
    def _validate_and_score_butterfly_pattern(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[PatternResult]:
        """
        Full validation and scoring of the Butterfly pattern.
        
        Comprehensive validation with Butterfly-specific
        scoring algorithms and extended AD analysis.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Calculate Fibonacci ratios
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            bc_distance = abs(c.price - b.price)
            cd_distance = abs(d.price - c.price)
            ad_distance = abs(d.price - a.price)
            
            # Prevent division by zero
            if xa_distance == 0 or ab_distance == 0:
                return None
            
            # Fibonacci ratios for validation
            ab_ratio = ab_distance / xa_distance
            bc_ratio = bc_distance / ab_distance if ab_distance > 0 else 0
            cd_ratio = cd_distance / bc_distance if bc_distance > 0 else 0
            ad_ratio = ad_distance / xa_distance
            
            # Determine pattern type
            pattern_type = PatternType.BULLISH if a.price > x.price else PatternType.BEARISH
            
            # Validate Fibonacci ratios with Butterfly-specific criteria
            fib_scores = []

            # AB should be ~78.6% of XA (critical for Butterfly)
            ab_target = self.butterfly_fib_ratios.AB_RETRACEMENT
            ab_score = 1.0 - abs(ab_ratio - ab_target) / ab_target
            fib_scores.append(max(0, ab_score))
            
            # BC should be 38.2% - 88.6% of AB
            bc_in_range = (self.butterfly_fib_ratios.BC_MIN_RETRACEMENT <= 
                          bc_ratio <= 
                          self.butterfly_fib_ratios.BC_MAX_RETRACEMENT)
            bc_score = 1.0 if bc_in_range else 0.0
            fib_scores.append(bc_score)
            
            # CD should be 161.8% - 261.8% of BC
            cd_in_range = (self.butterfly_fib_ratios.CD_MIN_EXTENSION <= 
                          cd_ratio <= 
                          self.butterfly_fib_ratios.CD_MAX_EXTENSION)
            cd_score = 1.0 if cd_in_range else 0.0
            fib_scores.append(cd_score)
            
            # AD should be 127.2% or 161.8% extension of XA (critical for Butterfly)
            target_1 = self.butterfly_fib_ratios.AD_EXTENSION_1
            target_2 = self.butterfly_fib_ratios.AD_EXTENSION_2
            score_1 = 1.0 - abs(ad_ratio - target_1) / target_1
            score_2 = 1.0 - abs(ad_ratio - target_2) / target_2
            ad_score = max(score_1, score_2)  # Take the best of two variants
            fib_scores.append(max(0, ad_score))
            
            # Overall Fibonacci confluence score
            fibonacci_confluence = np.mean(fib_scores)
            
            # Validate pattern
            validation_status = self._determine_validation_status(fib_scores)
            
            if validation_status == PatternValidation.INVALID:
                return None
            
            # Calculate confidence score with Butterfly-specific weights
            confidence_score = self._calculate_butterfly_confidence_score(
                pattern_points, data, fib_scores, fibonacci_confluence
            )
            
            # Calculate trading levels for Butterfly pattern
            entry_price, stop_loss, take_profits = self._calculate_butterfly_trading_levels(
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
            
            # Pattern strength analysis with Butterfly-specific metrics
            pattern_strength = self._calculate_butterfly_pattern_strength(
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
                ab_ratio=ab_ratio,
                bc_ratio=bc_ratio,
                cd_ratio=cd_ratio,
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
            logger.error(f"Error validating Butterfly pattern: {str(e)}")
            return None
    
    def _calculate_butterfly_confidence_score(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        fib_scores: List[float],
        fibonacci_confluence: float
    ) -> float:
        """
        Calculate confidence score specific to the Butterfly pattern.
        
        Butterfly-specific multi-factor scoring system.
        """
        try:
            scores = []
            weights = []
            
            # 1. Fibonacci accuracy (40% weight)
            scores.append(fibonacci_confluence)
            weights.append(0.40)
            
            # 2. AB ratio precision (25% weight - critical for Butterfly)
            x, a, b, c, d = pattern_points
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            if xa_distance > 0:
                ab_ratio = ab_distance / xa_distance
                ab_precision = 1.0 - abs(ab_ratio - self.butterfly_fib_ratios.AB_RETRACEMENT) / self.butterfly_fib_ratios.AB_RETRACEMENT
                scores.append(max(0, ab_precision))
                weights.append(0.25)
            else:
                weights[0] += 0.25
            
            # 3. AD extension quality (20% weight - unique Butterfly characteristic)
            if xa_distance > 0:
                ad_distance = abs(d.price - a.price)
                ad_ratio = ad_distance / xa_distance
                
                # Check both possible target levels
                target_1_score = 1.0 - abs(ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_1) / self.butterfly_fib_ratios.AD_EXTENSION_1
                target_2_score = 1.0 - abs(ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_2) / self.butterfly_fib_ratios.AD_EXTENSION_2
                ad_extension_score = max(target_1_score, target_2_score)
                
                scores.append(max(0, ad_extension_score))
                weights.append(0.20)
            else:
                weights[0] += 0.20
            
            # 4. Pattern symmetry (10% weight)
            symmetry_score = self._calculate_pattern_symmetry(pattern_points)
            scores.append(symmetry_score)
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
            logger.warning(f"Error calculating Butterfly confidence score: {str(e)}")
            return 0.0
    
    def _calculate_butterfly_trading_levels(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        pattern_type: PatternType
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Calculate trading levels for the Butterfly pattern.
        
        Butterfly-specific trading levels considering
        extended AD ratio and aggressive profit targets.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Entry price at point D (completion)
            entry_price = d.price
            
            # Stop loss calculation for Butterfly pattern
            if pattern_type == PatternType.BULLISH:
                # For bullish Butterfly: SL slightly above point C (more aggressive)
                stop_loss = c.price + abs(c.price - b.price) * 0.1  # 10% buffer above C

                # Take Profit levels considering extended movement
                xa_range = a.price - x.price
                tp1 = entry_price - (xa_range * 0.382)  # Retracement to 38.2% XA
                tp2 = entry_price - (xa_range * 0.618)  # Retracement to 61.8% XA
                tp3 = entry_price - (xa_range * 0.786)  # Retracement to 78.6% XA

            else:  # BEARISH
                # For bearish Butterfly: SL slightly below point C
                stop_loss = c.price - abs(b.price - c.price) * 0.1  # 10% buffer below C

                # Take profit levels (above entry)
                xa_range = x.price - a.price
                tp1 = entry_price + (xa_range * 0.382)  # Retracement to 38.2% XA
                tp2 = entry_price + (xa_range * 0.618)  # Retracement to 61.8% XA
                tp3 = entry_price + (xa_range * 0.786)  # Retracement to 78.6% XA
            
            return entry_price, stop_loss, (tp1, tp2, tp3)
            
        except Exception as e:
            logger.error(f"Error calculating Butterfly trading levels: {str(e)}")
            # Fallback levels
            return super()._calculate_trading_levels(pattern_points, pattern_type)
    
    def _calculate_butterfly_pattern_strength(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        fibonacci_confluence: float,
        volume_confirmation: float
    ) -> float:
        """Calculate Butterfly pattern strength with specific metrics."""
        try:
            # Base calculation as in parent class
            base_strength = super()._calculate_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Butterfly-specific additions
            x, a, b, c, d = pattern_points
            
            # Extension quality bonus (Butterfly is unique with AD extension)
            xa_distance = abs(a.price - x.price)
            ad_distance = abs(d.price - a.price)
            if xa_distance > 0:
                ad_ratio = ad_distance / xa_distance
                
                # Bonus for exact hit on target extension levels
                target_1_accuracy = 1.0 - abs(ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_1) / self.butterfly_fib_ratios.AD_EXTENSION_1
                target_2_accuracy = 1.0 - abs(ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_2) / self.butterfly_fib_ratios.AD_EXTENSION_2
                best_accuracy = max(target_1_accuracy, target_2_accuracy)
                
                if best_accuracy > 0.9:  # High precision
                    extension_bonus = (best_accuracy - 0.9) * 0.5
                    base_strength = min(1.0, base_strength + extension_bonus)
            
            return base_strength
            
        except Exception as e:
            logger.warning(f"Error calculating Butterfly pattern strength: {str(e)}")
            return 0.0
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[PatternResult]:
        """
        Detect Butterfly patterns in OHLCV data.
        
        Optimized Butterfly detection with caching and
        parallel processing for real-time trading systems.
        """
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Generate cache key for optimization
            cache_key = f"butterfly_{self._generate_cache_key(data, symbol, timeframe)}"
            
            # Check cache
            if cache_key in self._patterns_cache:
                logger.debug(f"Returning cached Butterfly patterns for {cache_key}")
                return self._patterns_cache[cache_key]
            
            # Find significant points (peaks and troughs) using ZigZag
            pivot_points = self._find_pivot_points(data)
            
            if len(pivot_points) < 5:
                logger.warning("Not enough pivot points for Butterfly pattern detection")
                return []
            
            # Find potential patterns Butterfly
            potential_patterns = self._find_butterfly_patterns(pivot_points, data)
            
            # Validate and score patterns
            validated_patterns = []
            for pattern_data in potential_patterns:
                pattern_result = self._validate_and_score_butterfly_pattern(
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
            
            logger.info(f"Detected {len(validated_patterns)} valid Butterfly patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in detect_patterns: {str(e)}")
            raise RuntimeError(f"Butterfly pattern detection failed: {str(e)}") from e
    
    def get_entry_signals(self, pattern: PatternResult) -> Dict[str, any]:
        """
        Generate entry signals for Butterfly pattern trading.
        
        Butterfly-specific entry analysis with enhanced extension focus.
        """
        try:
            # Base signals from parent class
            signals = super().get_entry_signals(pattern)

            # Butterfly-specific modifications
            signals['entry_reason'] = f"Butterfly {pattern.pattern_type.value} pattern completion (AD extension: {pattern.ad_ratio:.3f})"
            
            # More aggressive entry conditions for Butterfly (due to extension)
            signals['entry_conditions']['ab_ratio_precision'] = (
                abs(pattern.ab_ratio - self.butterfly_fib_ratios.AB_RETRACEMENT) < self.tolerance
            )
            
            # Check AD extension quality
            target_1_match = abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_1) < self.tolerance
            target_2_match = abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_2) < self.tolerance
            signals['entry_conditions']['ad_extension_valid'] = target_1_match or target_2_match
            
            # Butterfly-specific timing (more aggressive thresholds)
            signals['timing'] = {
                'immediate': pattern.confidence_score > 0.88,  # Highest threshold
                'wait_for_confirmation': 0.75 <= pattern.confidence_score <= 0.88,
                'avoid': pattern.confidence_score < 0.75
            }
            
            # Additional information about Butterfly pattern
            signals['butterfly_specifics'] = {
                'ad_ratio': pattern.ad_ratio,
                'ad_extension_1': self.butterfly_fib_ratios.AD_EXTENSION_1,
                'ad_extension_2': self.butterfly_fib_ratios.AD_EXTENSION_2,
                'closest_extension': (self.butterfly_fib_ratios.AD_EXTENSION_1 
                                    if abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_1) < 
                                       abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_2)
                                    else self.butterfly_fib_ratios.AD_EXTENSION_2),
                'extension_precision': min(
                    abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_1),
                    abs(pattern.ad_ratio - self.butterfly_fib_ratios.AD_EXTENSION_2)
                ) / pattern.ad_ratio,
                'pattern_quality': 'HIGH' if pattern.confidence_score > 0.85 else 'MEDIUM'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Butterfly entry signals: {str(e)}")
            return {}


# Export main components
__all__ = [
    'ButterflyPattern',
    'ButterflyFibonacciRatios'
]