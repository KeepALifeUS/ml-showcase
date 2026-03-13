"""
Crab Pattern - Professional Harmonic Pattern Detector.

Enterprise Implementation: High-precision Crab pattern detection
using advanced Fibonacci analysis, real-time processing,
and machine learning classification for professional crypto trading.

Crab Pattern:
- Discovered by Scott Carney in 2001
- The most extreme harmonic pattern
- Based on precise Fibonacci retracements and extreme extensions
- Very high accuracy when correctly identified

Fibonacci Ratios for Crab:
- XA: Initial impulse (any length)
- AB: 38.2% - 61.8% retracement of XA
- BC: 38.2% - 88.6% retracement of AB
- CD: 224% - 361.8% extension of BC (extreme!)
- AD: 161.8% extension of XA (critical for validation)

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
class CrabFibonacciRatios:
    """Fibonacci ratios for the Crab pattern.
    
    Immutable data structure for thread-safe operations.
    Specific ratios for extreme Crab pattern.
    """
    # Main Fibonacci levels for Crab
    AB_RETRACEMENT_MIN: float = 0.382  # Minimum XA retracement
    AB_RETRACEMENT_MAX: float = 0.618  # Maximum XA retracement
    BC_MIN_RETRACEMENT: float = 0.382  # Minimum AB retracement
    BC_MAX_RETRACEMENT: float = 0.886  # Maximum AB retracement
    CD_MIN_EXTENSION: float = 2.240    # Minimum BC extension (224%)
    CD_MAX_EXTENSION: float = 3.618    # Maximum BC extension (361.8%)
    AD_EXTENSION: float = 1.618        # XA extension (161.8%)
    
    # Allowed deviations for validation (in production systems)
    TOLERANCE: float = 0.08  # 8% tolerance (higher due to extreme nature)


class CrabPattern(GartleyPattern):
    """
    Professional Crab Pattern Detector.
    
    High-performance Crab pattern detection
    for real-time crypto trading systems.
    
    Features:
    - High-precision Fibonacci validation specific to Crab
    - Real-time pattern scanning with extreme extensions
    - Advanced confidence scoring
    - Professional risk management for high-risk patterns
    - Multi-timeframe analysis
    - Volume confirmation
    - ML-enhanced pattern recognition
    
    Example:
        ```python
        detector = CrabPattern(tolerance=0.08, min_confidence=0.80)
        
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
        tolerance: float = 0.08,  # Increased tolerance for Crab
        min_confidence: float = 0.75,  # Raised minimum threshold
        enable_volume_analysis: bool = True,
        enable_ml_scoring: bool = True,
        min_pattern_bars: int = 30,  # Increased for extreme patterns
        max_pattern_bars: int = 400  # Significantly increased
    ):
        """
        Initialize Crab Pattern Detector.
        
        Args:
            tolerance: Allowed deviation from Fibonacci ratios (default: 8%)
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
        
        # Specific Fibonacci ratios for the Crab pattern
        self.crab_fib_ratios = CrabFibonacciRatios()
        
        logger.info(f"CrabPattern initialized with tolerance={tolerance}, "
                   f"min_confidence={min_confidence}")
    
    def _is_valid_crab_geometry(
        self, 
        x: PatternPoint, 
        a: PatternPoint, 
        b: PatternPoint, 
        c: PatternPoint, 
        d: PatternPoint
    ) -> bool:
        """
        Geometric validation of the Crab pattern structure.
        
        Crab pattern has the most extreme CD extensions among all patterns.
        """
        try:
            # Basic geometric check as in Gartley
            if not super()._is_valid_gartley_geometry(x, a, b, c, d):
                return False
            
            # Additional checks specific to Crab
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            bc_distance = abs(c.price - b.price)
            cd_distance = abs(d.price - c.price)
            ad_distance = abs(d.price - a.price)
            
            if xa_distance == 0 or ab_distance == 0 or bc_distance == 0:
                return False
            
            # AB must be in range 38.2% - 61.8% of XA
            ab_ratio = ab_distance / xa_distance
            if not (self.crab_fib_ratios.AB_RETRACEMENT_MIN - self.tolerance <= 
                   ab_ratio <= 
                   self.crab_fib_ratios.AB_RETRACEMENT_MAX + self.tolerance):
                return False
            
            # CD should be an extreme extension 224% - 361.8% of BC
            cd_ratio = cd_distance / bc_distance
            if not (self.crab_fib_ratios.CD_MIN_EXTENSION - self.tolerance <= 
                   cd_ratio <= 
                   self.crab_fib_ratios.CD_MAX_EXTENSION + self.tolerance):
                return False
            
            # AD must be close to 161.8% extension of XA
            ad_ratio = ad_distance / xa_distance
            expected_ad_ratio = self.crab_fib_ratios.AD_EXTENSION
            if abs(ad_ratio - expected_ad_ratio) > self.tolerance:
                return False
            
            # Additional check: D must be significantly OUTSIDE the X-A range
            pattern_type = PatternType.BULLISH if a.price > x.price else PatternType.BEARISH
            if pattern_type == PatternType.BULLISH:
                # For bullish: D should be significantly above A
                if d.price <= a.price * 1.1:  # Minimum 10% above A
                    return False
            else:
                # For bearish: D should be significantly below A
                if d.price >= a.price * 0.9:  # Minimum 10% below A
                    return False
                        
            return True
                        
        except Exception as e:
            logger.warning(f"Crab geometry validation error: {str(e)}")
            return False
    
    def _find_crab_patterns(
        self, 
        pivot_points: List[PatternPoint], 
        data: pd.DataFrame
    ) -> List[Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]]:
        """
        Search for potential 5-point Crab patterns (X-A-B-C-D).
        
        Optimized pattern matching with Crab-specific
        extreme extension constraints for high-probability setups.
        """
        potential_patterns = []
        
        # Need at least 5 points for X-A-B-C-D pattern
        if len(pivot_points) < 5:
            return potential_patterns
        
        # Iterate over all possible 5-point combinations (increased range)
        for i in range(len(pivot_points) - 4):
            for j in range(i + 1, min(i + self.max_pattern_bars // 6, len(pivot_points) - 3)):
                for k in range(j + 1, min(j + self.max_pattern_bars // 6, len(pivot_points) - 2)):
                    for l in range(k + 1, min(k + self.max_pattern_bars // 6, len(pivot_points) - 1)):
                        for m in range(l + 1, min(l + self.max_pattern_bars // 6, len(pivot_points))):
                            
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
                            
                            # Crab-specific geometric validation
                            if self._is_valid_crab_geometry(x, a, b, c, d):
                                potential_patterns.append((x, a, b, c, d))
        
        logger.debug(f"Found {len(potential_patterns)} potential Crab patterns")
        return potential_patterns
    
    def _validate_and_score_crab_pattern(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[PatternResult]:
        """
        Full validation and scoring of the Crab pattern.
        
        Comprehensive validation with Crab-specific
        extreme extension scoring algorithms.
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
            if xa_distance == 0 or ab_distance == 0 or bc_distance == 0:
                return None
            
            # Fibonacci ratios for validation
            ab_ratio = ab_distance / xa_distance
            bc_ratio = bc_distance / ab_distance
            cd_ratio = cd_distance / bc_distance
            ad_ratio = ad_distance / xa_distance
            
            # Determine pattern type
            pattern_type = PatternType.BULLISH if a.price > x.price else PatternType.BEARISH
            
            # Validate Fibonacci ratios with Crab-specific criteria
            fib_scores = []
            
            # AB should be 38.2% - 61.8% of XA
            ab_in_range = (self.crab_fib_ratios.AB_RETRACEMENT_MIN <= 
                          ab_ratio <= 
                          self.crab_fib_ratios.AB_RETRACEMENT_MAX)
            ab_score = 1.0 if ab_in_range else 0.0
            fib_scores.append(ab_score)
            
            # BC should be 38.2% - 88.6% of AB
            bc_in_range = (self.crab_fib_ratios.BC_MIN_RETRACEMENT <= 
                          bc_ratio <= 
                          self.crab_fib_ratios.BC_MAX_RETRACEMENT)
            bc_score = 1.0 if bc_in_range else 0.0
            fib_scores.append(bc_score)
            
            # CD should be 224% - 361.8% of BC (extreme extension!)
            cd_in_range = (self.crab_fib_ratios.CD_MIN_EXTENSION <= 
                          cd_ratio <= 
                          self.crab_fib_ratios.CD_MAX_EXTENSION)
            cd_score = 1.0 if cd_in_range else 0.0
            # Bonus for hitting "golden" levels
            if abs(cd_ratio - 2.618) < self.tolerance or abs(cd_ratio - 3.618) < self.tolerance:
                cd_score = min(1.0, cd_score + 0.3)
            fib_scores.append(cd_score)
            
            # AD should be ~161.8% extension of XA (critical for Crab)
            ad_target = self.crab_fib_ratios.AD_EXTENSION
            ad_score = 1.0 - abs(ad_ratio - ad_target) / ad_target
            fib_scores.append(max(0, ad_score))
            
            # Overall Fibonacci confluence score
            fibonacci_confluence = np.mean(fib_scores)
            
            # Validate pattern (stricter criteria for Crab)
            validation_status = self._determine_crab_validation_status(fib_scores)
            
            if validation_status == PatternValidation.INVALID:
                return None
            
            # Calculate confidence score with Crab-specific weights
            confidence_score = self._calculate_crab_confidence_score(
                pattern_points, data, fib_scores, fibonacci_confluence
            )
            
            # Calculate trading levels for Crab pattern
            entry_price, stop_loss, take_profits = self._calculate_crab_trading_levels(
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
            
            # Pattern strength analysis with Crab-specific metrics
            pattern_strength = self._calculate_crab_pattern_strength(
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
            logger.error(f"Error validating Crab pattern: {str(e)}")
            return None
    
    def _determine_crab_validation_status(self, fib_scores: List[float]) -> PatternValidation:
        """Determine validation status specific to the Crab pattern."""
        avg_score = np.mean(fib_scores)
        min_score = min(fib_scores)
        
        # Stricter criteria for the extreme Crab pattern
        if avg_score >= 0.85 and min_score >= 0.7:  # Raised thresholds
            return PatternValidation.VALID
        elif avg_score >= 0.75 and min_score >= 0.5:
            return PatternValidation.MARGINAL
        else:
            return PatternValidation.INVALID
    
    def _calculate_crab_confidence_score(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        fib_scores: List[float],
        fibonacci_confluence: float
    ) -> float:
        """
        Calculate confidence score specific to the Crab pattern.
        
        Crab-specific multi-factor scoring system.
        """
        try:
            scores = []
            weights = []
            
            # 1. Fibonacci accuracy (35% weight)
            scores.append(fibonacci_confluence)
            weights.append(0.35)
            
            # 2. CD extension quality (30% weight - critical for Crab!)
            x, a, b, c, d = pattern_points
            bc_distance = abs(c.price - b.price)
            cd_distance = abs(d.price - c.price)
            if bc_distance > 0:
                cd_ratio = cd_distance / bc_distance
                
                # Check for hits on extreme extension levels
                extension_accuracy = 0.0
                for target in [2.240, 2.618, 3.618]:  # Key levels for Crab
                    accuracy = 1.0 - abs(cd_ratio - target) / target
                    extension_accuracy = max(extension_accuracy, accuracy)
                
                scores.append(max(0, extension_accuracy))
                weights.append(0.30)
            else:
                weights[0] += 0.30
            
            # 3. AD extension precision (20% weight)
            xa_distance = abs(a.price - x.price)
            ad_distance = abs(d.price - a.price)
            if xa_distance > 0:
                ad_ratio = ad_distance / xa_distance
                ad_precision = 1.0 - abs(ad_ratio - self.crab_fib_ratios.AD_EXTENSION) / self.crab_fib_ratios.AD_EXTENSION
                scores.append(max(0, ad_precision))
                weights.append(0.20)
            else:
                weights[0] += 0.20
            
            # 4. Pattern extremity assessment (10% weight - unique to Crab)
            extremity_score = self._assess_pattern_extremity(pattern_points)
            scores.append(extremity_score)
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
            logger.warning(f"Error calculating Crab confidence score: {str(e)}")
            return 0.0
    
    def _assess_pattern_extremity(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]
    ) -> float:
        """
        Assess pattern extremity (unique to Crab).
        
        Crab pattern should show extreme movements.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Calculate relative movements
            xa_move = abs(a.price - x.price) / x.price if x.price > 0 else 0
            bc_move = abs(c.price - b.price) / b.price if b.price > 0 else 0  
            cd_move = abs(d.price - c.price) / c.price if c.price > 0 else 0
            
            # Assess extremity
            extremity_score = 0.0
            
            # Bonus for strong XA and CD movements
            if xa_move > 0.10:  # Movement > 10%
                extremity_score += 0.3
            if cd_move > 0.15:  # CD movement should be even stronger
                extremity_score += 0.4
                
            # Bonus for large time span
            pattern_duration = d.index - x.index
            if pattern_duration > 50:  # Longer patterns are more reliable
                extremity_score += 0.3
                
            return min(1.0, extremity_score)
            
        except Exception as e:
            logger.warning(f"Error assessing pattern extremity: {str(e)}")
            return 0.5
    
    def _calculate_crab_trading_levels(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        pattern_type: PatternType
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Calculate trading levels for the Crab pattern.
        
        Crab-specific trading levels considering
        extreme extensions and conservative profit targets.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Entry price at point D (completion)
            entry_price = d.price
            
            # Stop loss calculation for Crab pattern (conservative)
            if pattern_type == PatternType.BULLISH:
                # For bullish Crab: SL below point C with large buffer
                bc_range = c.price - b.price
                stop_loss = c.price - (bc_range * 0.2)  # 20% buffer below C
                
                # Take Profit levels considering extreme movement (conservative)
                cd_range = d.price - c.price
                tp1 = entry_price - (cd_range * 0.382)  # Retracement to 38.2% CD
                tp2 = entry_price - (cd_range * 0.618)  # Retracement to 61.8% CD  
                tp3 = c.price  # Full retracement to point C
                
            else:  # BEARISH
                # For bearish Crab: SL above point C
                bc_range = b.price - c.price
                stop_loss = c.price + (bc_range * 0.2)  # 20% buffer above C
                
                # Take profit levels (above entry)
                cd_range = c.price - d.price
                tp1 = entry_price + (cd_range * 0.382)  # Retracement to 38.2% CD
                tp2 = entry_price + (cd_range * 0.618)  # Retracement to 61.8% CD
                tp3 = c.price  # Full retracement to point C
            
            return entry_price, stop_loss, (tp1, tp2, tp3)
            
        except Exception as e:
            logger.error(f"Error calculating Crab trading levels: {str(e)}")
            # Fallback levels
            return super()._calculate_trading_levels(pattern_points, pattern_type)
    
    def _calculate_crab_pattern_strength(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        fibonacci_confluence: float,
        volume_confirmation: float
    ) -> float:
        """Calculate Crab pattern strength with extreme metrics."""
        try:
            # Base calculation as in parent class
            base_strength = super()._calculate_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Crab-specific additions
            x, a, b, c, d = pattern_points
            
            # Extension extremity bonus
            bc_distance = abs(c.price - b.price)
            cd_distance = abs(d.price - c.price)
            if bc_distance > 0:
                cd_ratio = cd_distance / bc_distance
                
                # Bonus for hitting extreme levels
                extremity_bonus = 0.0
                if 2.5 <= cd_ratio <= 3.7:  # Extreme zone
                    extremity_bonus = 0.2
                elif 2.2 <= cd_ratio <= 2.5:  # Moderate zone
                    extremity_bonus = 0.1
                    
                base_strength = min(1.0, base_strength + extremity_bonus)
            
            # Volume intensity bonus (Crab requires strong volume at extremes)
            if volume_confirmation > 0.7:
                volume_bonus = (volume_confirmation - 0.7) * 0.5
                base_strength = min(1.0, base_strength + volume_bonus)
            
            return base_strength
            
        except Exception as e:
            logger.warning(f"Error calculating Crab pattern strength: {str(e)}")
            return 0.0
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[PatternResult]:
        """
        Detect Crab patterns in OHLCV data.
        
        Optimized Crab detection with caching and
        specialized processing for extreme patterns.
        """
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Generate cache key for optimization
            cache_key = f"crab_{self._generate_cache_key(data, symbol, timeframe)}"
            
            # Check cache
            if cache_key in self._patterns_cache:
                logger.debug(f"Returning cached Crab patterns for {cache_key}")
                return self._patterns_cache[cache_key]
            
            # Find significant points (peaks and troughs) using ZigZag
            pivot_points = self._find_pivot_points(data)
            
            if len(pivot_points) < 5:
                logger.warning("Not enough pivot points for Crab pattern detection")
                return []
            
            # Find potential patterns Crab
            potential_patterns = self._find_crab_patterns(pivot_points, data)
            
            # Validate and score patterns
            validated_patterns = []
            for pattern_data in potential_patterns:
                pattern_result = self._validate_and_score_crab_pattern(
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
            
            logger.info(f"Detected {len(validated_patterns)} valid Crab patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in detect_patterns: {str(e)}")
            raise RuntimeError(f"Crab pattern detection failed: {str(e)}") from e
    
    def get_entry_signals(self, pattern: PatternResult) -> Dict[str, any]:
        """
        Generate entry signals for Crab pattern trading.
        
        Crab-specific entry analysis with extreme extension focus.
        """
        try:
            # Base signals from parent class
            signals = super().get_entry_signals(pattern)
            
            # Crab-specific modifications
            signals['entry_reason'] = f"Crab {pattern.pattern_type.value} pattern completion (extreme CD: {pattern.cd_ratio:.3f}x)"
            
            # Very strict entry conditions for Crab
            signals['entry_conditions']['cd_extension_extreme'] = (
                self.crab_fib_ratios.CD_MIN_EXTENSION <= pattern.cd_ratio <= self.crab_fib_ratios.CD_MAX_EXTENSION
            )
            signals['entry_conditions']['ad_extension_valid'] = (
                abs(pattern.ad_ratio - self.crab_fib_ratios.AD_EXTENSION) < self.tolerance
            )
            signals['entry_conditions']['high_confidence_required'] = pattern.confidence_score >= 0.80
            
            # Crab-specific timing (strictest thresholds)
            signals['timing'] = {
                'immediate': pattern.confidence_score > 0.90,  # Very high threshold
                'wait_for_confirmation': 0.80 <= pattern.confidence_score <= 0.90,
                'avoid': pattern.confidence_score < 0.80
            }
            
            # Additional information about Crab pattern
            signals['crab_specifics'] = {
                'cd_ratio': pattern.cd_ratio,
                'cd_extension_level': self._classify_cd_extension(pattern.cd_ratio),
                'ad_ratio': pattern.ad_ratio,
                'ad_target': self.crab_fib_ratios.AD_EXTENSION,
                'pattern_extremity': self._assess_pattern_extremity((
                    pattern.point_x, pattern.point_a, pattern.point_b, 
                    pattern.point_c, pattern.point_d
                )),
                'risk_level': 'EXTREME',  # Crab is always high-risk
                'pattern_quality': 'EXTREME' if pattern.confidence_score > 0.85 else 'HIGH'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Crab entry signals: {str(e)}")
            return {}
    
    def _classify_cd_extension(self, cd_ratio: float) -> str:
        """Classify CD extension level for Crab."""
        if cd_ratio >= 3.5:
            return "EXTREME_361.8"
        elif cd_ratio >= 3.0:
            return "VERY_HIGH_300+"
        elif cd_ratio >= 2.6:
            return "HIGH_261.8"
        elif cd_ratio >= 2.2:
            return "MODERATE_224"
        else:
            return "BELOW_MINIMUM"


# Export main components
__all__ = [
    'CrabPattern',
    'CrabFibonacciRatios'
]