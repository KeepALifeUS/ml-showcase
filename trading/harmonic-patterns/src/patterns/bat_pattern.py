"""
Bat Pattern - Professional Harmonic Pattern Detector.

Enterprise Implementation: High-precision Bat pattern detection
using advanced Fibonacci analysis, real-time processing,
and machine learning classification for professional crypto trading.

Bat Pattern:
- Discovered by Scott Carney in 2001
- More precise than the Gartley pattern
- Based on strict Fibonacci retracements and extensions
- High accuracy in reversal prediction

Fibonacci Ratios for Bat:
- XA: Initial impulse (any length)
- AB: 38.2% or 50% retracement of XA
- BC: 38.2% - 88.6% retracement of AB
- CD: 161.8% - 261.8% extension of BC
- AD: 88.6% retracement of XA (critical for validation)

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
class BatFibonacciRatios:
    """Fibonacci ratios for the Bat pattern.
    
    Immutable data structure for thread-safe operations.
    Specific ratios for high-precision Bat pattern.
    """
    # Main Fibonacci levels for Bat
    AB_RETRACEMENT_MIN: float = 0.382  # 38.2% retracement of XA
    AB_RETRACEMENT_MAX: float = 0.500  # 50% retracement of XA  
    BC_MIN_RETRACEMENT: float = 0.382  # Minimum AB retracement
    BC_MAX_RETRACEMENT: float = 0.886  # Maximum AB retracement
    CD_MIN_EXTENSION: float = 1.618    # Minimum BC extension
    CD_MAX_EXTENSION: float = 2.618    # Maximum BC extension
    AD_RETRACEMENT: float = 0.886      # Critical XA retracement (differs from Gartley)
    
    # Allowed deviations for validation (in production systems)
    TOLERANCE: float = 0.05  # 5% tolerance for each ratio


class BatPattern(GartleyPattern):
    """
    Professional Bat Pattern Detector.
    
    High-performance Bat pattern detection
    for real-time crypto trading systems.
    
    Features:
    - High-precision Fibonacci validation specific to Bat
    - Real-time pattern scanning
    - Advanced confidence scoring
    - Professional risk management
    - Multi-timeframe analysis
    - Volume confirmation
    - ML-enhanced pattern recognition
    
    Example:
        ```python
        detector = BatPattern(tolerance=0.05, min_confidence=0.75)
        
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
        min_pattern_bars: int = 20,
        max_pattern_bars: int = 200
    ):
        """
        Initialize Bat Pattern Detector.
        
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
        
        # Specific Fibonacci ratios for the Bat pattern
        self.bat_fib_ratios = BatFibonacciRatios()
        
        logger.info(f"BatPattern initialized with tolerance={tolerance}, "
                   f"min_confidence={min_confidence}")
    
    def _is_valid_bat_geometry(
        self, 
        x: PatternPoint, 
        a: PatternPoint, 
        b: PatternPoint, 
        c: PatternPoint, 
        d: PatternPoint
    ) -> bool:
        """
        Geometric validation of the Bat pattern structure.
        
        Bat pattern has specific AD ratio (88.6% instead of 78.6% as in Gartley).
        """
        try:
            # Basic geometric check as in Gartley
            if not super()._is_valid_gartley_geometry(x, a, b, c, d):
                return False
            
            # Additional checks specific to Bat
            xa_distance = abs(a.price - x.price)
            ab_distance = abs(b.price - a.price)
            ad_distance = abs(d.price - a.price)
            
            if xa_distance == 0:
                return False
            
            # AB must be in range 38.2% - 50% of XA
            ab_ratio = ab_distance / xa_distance
            if not (self.bat_fib_ratios.AB_RETRACEMENT_MIN - self.tolerance <= 
                   ab_ratio <= 
                   self.bat_fib_ratios.AB_RETRACEMENT_MAX + self.tolerance):
                return False
            
            # AD must be close to 88.6% of XA (critical for Bat)
            ad_ratio = ad_distance / xa_distance
            expected_ad_ratio = self.bat_fib_ratios.AD_RETRACEMENT
            if abs(ad_ratio - expected_ad_ratio) > self.tolerance:
                return False
                        
            return True
                        
        except Exception as e:
            logger.warning(f"Bat geometry validation error: {str(e)}")
            return False
    
    def _find_bat_patterns(
        self, 
        pivot_points: List[PatternPoint], 
        data: pd.DataFrame
    ) -> List[Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint]]:
        """
        Search for potential 5-point Bat patterns (X-A-B-C-D).
        
        Optimized pattern matching with Bat-specific
        Fibonacci constraints for high-probability setups.
        """
        potential_patterns = []
        
        # Need at least 5 points for X-A-B-C-D pattern
        if len(pivot_points) < 5:
            return potential_patterns
        
        # Iterate over all possible 5-point combinations
        for i in range(len(pivot_points) - 4):
            for j in range(i + 1, min(i + self.max_pattern_bars // 10, len(pivot_points) - 3)):
                for k in range(j + 1, min(j + self.max_pattern_bars // 10, len(pivot_points) - 2)):
                    for l in range(k + 1, min(k + self.max_pattern_bars // 10, len(pivot_points) - 1)):
                        for m in range(l + 1, min(l + self.max_pattern_bars // 10, len(pivot_points))):
                            
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
                            
                            # Bat-specific geometric validation
                            if self._is_valid_bat_geometry(x, a, b, c, d):
                                potential_patterns.append((x, a, b, c, d))
        
        logger.debug(f"Found {len(potential_patterns)} potential Bat patterns")
        return potential_patterns
    
    def _validate_and_score_bat_pattern(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        symbol: Optional[str],
        timeframe: Optional[str]
    ) -> Optional[PatternResult]:
        """
        Full validation and scoring of the Bat pattern.
        
        Comprehensive validation with Bat-specific
        scoring algorithms, risk analysis, and ML-enhanced pattern recognition.
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

            # Validate Fibonacci ratios with Bat-specific criteria
            fib_scores = []

            # AB should be 38.2% - 50% of XA
            ab_in_range = (self.bat_fib_ratios.AB_RETRACEMENT_MIN <= 
                          ab_ratio <= 
                          self.bat_fib_ratios.AB_RETRACEMENT_MAX)
            ab_score = 1.0 if ab_in_range else 0.0
            # Additional bonus for exact hit on 38.2% or 50%
            if abs(ab_ratio - 0.382) < self.tolerance or abs(ab_ratio - 0.500) < self.tolerance:
                ab_score = min(1.0, ab_score + 0.2)
            fib_scores.append(ab_score)
            
            # BC should be 38.2% - 88.6% of AB
            bc_in_range = (self.bat_fib_ratios.BC_MIN_RETRACEMENT <= 
                          bc_ratio <= 
                          self.bat_fib_ratios.BC_MAX_RETRACEMENT)
            bc_score = 1.0 if bc_in_range else 0.0
            fib_scores.append(bc_score)
            
            # CD should be 161.8% - 261.8% of BC (extended range for Bat)
            cd_in_range = (self.bat_fib_ratios.CD_MIN_EXTENSION <= 
                          cd_ratio <= 
                          self.bat_fib_ratios.CD_MAX_EXTENSION)
            cd_score = 1.0 if cd_in_range else 0.0
            fib_scores.append(cd_score)
            
            # AD should be ~88.6% of XA (critical for Bat - differs from Gartley)
            ad_target = self.bat_fib_ratios.AD_RETRACEMENT
            ad_score = 1.0 - abs(ad_ratio - ad_target) / ad_target
            fib_scores.append(max(0, ad_score))
            
            # Overall Fibonacci confluence score
            fibonacci_confluence = np.mean(fib_scores)
            
            # Validate pattern
            validation_status = self._determine_validation_status(fib_scores)
            
            if validation_status == PatternValidation.INVALID:
                return None
            
            # Calculate confidence score with Bat-specific weights
            confidence_score = self._calculate_bat_confidence_score(
                pattern_points, data, fib_scores, fibonacci_confluence
            )
            
            # Calculate trading levels for Bat pattern
            entry_price, stop_loss, take_profits = self._calculate_bat_trading_levels(
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
            
            # Pattern strength analysis with Bat-specific metrics
            pattern_strength = self._calculate_bat_pattern_strength(
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
            logger.error(f"Error validating Bat pattern: {str(e)}")
            return None
    
    def _calculate_bat_confidence_score(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        data: pd.DataFrame,
        fib_scores: List[float],
        fibonacci_confluence: float
    ) -> float:
        """
        Calculate confidence score specific to the Bat pattern.
        
        Bat-specific multi-factor scoring system.
        """
        try:
            scores = []
            weights = []
            
            # 1. Fibonacci accuracy (45% weight - higher than Gartley due to Bat's strictness)
            scores.append(fibonacci_confluence)
            weights.append(0.45)
            
            # 2. AD ratio precision (20% weight - critical for Bat)
            x, a, b, c, d = pattern_points
            xa_distance = abs(a.price - x.price)
            ad_distance = abs(d.price - a.price)
            if xa_distance > 0:
                ad_ratio = ad_distance / xa_distance
                ad_precision = 1.0 - abs(ad_ratio - self.bat_fib_ratios.AD_RETRACEMENT) / self.bat_fib_ratios.AD_RETRACEMENT
                scores.append(max(0, ad_precision))
                weights.append(0.20)
            else:
                weights[0] += 0.20
            
            # 3. Pattern symmetry (15% weight)
            symmetry_score = self._calculate_pattern_symmetry(pattern_points)
            scores.append(symmetry_score)
            weights.append(0.15)
            
            # 4. Market context (10% weight)
            market_context_score = self._analyze_market_context(pattern_points, data)
            scores.append(market_context_score)
            weights.append(0.10)
            
            # 5. Volume confirmation (10% weight, if available)
            if self.enable_volume_analysis and 'volume' in data.columns:
                volume_score = self._analyze_volume_confirmation(pattern_points, data)
                scores.append(volume_score)
                weights.append(0.10)
            else:
                weights[0] += 0.10  # Redistribute weight to Fibonacci
            
            # Weighted average
            confidence = np.average(scores, weights=weights)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating Bat confidence score: {str(e)}")
            return 0.0
    
    def _calculate_bat_trading_levels(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        pattern_type: PatternType
    ) -> Tuple[float, float, Tuple[float, float, float]]:
        """
        Calculate trading levels for the Bat pattern.
        
        Bat-specific trading levels considering
        88.6% AD ratio and enhanced profit targets.
        """
        try:
            x, a, b, c, d = pattern_points
            
            # Entry price at point D (completion)
            entry_price = d.price
            
            # Stop loss calculation for Bat pattern
            if pattern_type == PatternType.BULLISH:
                # For bullish Bat: SL below point X with smaller buffer
                xa_range = a.price - x.price
                stop_loss = x.price - (xa_range * 0.05)  # 5% buffer (smaller than Gartley)

                # Take Profit levels considering Bat characteristics
                cd_range = d.price - c.price
                tp1 = entry_price + (cd_range * 0.382)  # 38.2% of CD movement
                tp2 = entry_price + (cd_range * 0.618)  # 61.8% of CD movement
                tp3 = entry_price + (cd_range * 1.272)  # 127.2% extension

            else:  # BEARISH
                # For bearish Bat: SL above point X
                xa_range = x.price - a.price
                stop_loss = x.price + (xa_range * 0.05)  # 5% buffer

                # Take profit levels (below entry)
                cd_range = c.price - d.price
                tp1 = entry_price - (cd_range * 0.382)  # 38.2% of CD movement
                tp2 = entry_price - (cd_range * 0.618)  # 61.8% of CD movement
                tp3 = entry_price - (cd_range * 1.272)  # 127.2% extension
            
            return entry_price, stop_loss, (tp1, tp2, tp3)
            
        except Exception as e:
            logger.error(f"Error calculating Bat trading levels: {str(e)}")
            # Fallback levels
            return super()._calculate_trading_levels(pattern_points, pattern_type)
    
    def _calculate_bat_pattern_strength(
        self,
        pattern_points: Tuple[PatternPoint, PatternPoint, PatternPoint, PatternPoint, PatternPoint],
        fibonacci_confluence: float,
        volume_confirmation: float
    ) -> float:
        """Calculate Bat pattern strength with specific metrics."""
        try:
            # Base calculation as in parent class
            base_strength = super()._calculate_pattern_strength(
                pattern_points, fibonacci_confluence, volume_confirmation
            )
            
            # Bat-specific additions
            x, a, b, c, d = pattern_points
            
            # AD ratio precision bonus (Bat critically depends on 88.6%)
            xa_distance = abs(a.price - x.price)
            ad_distance = abs(d.price - a.price)
            if xa_distance > 0:
                ad_ratio = ad_distance / xa_distance
                ad_precision = 1.0 - abs(ad_ratio - self.bat_fib_ratios.AD_RETRACEMENT) / self.bat_fib_ratios.AD_RETRACEMENT
                precision_bonus = max(0, ad_precision - 0.8) * 0.5  # Bonus for high precision
                base_strength = min(1.0, base_strength + precision_bonus)
            
            return base_strength
            
        except Exception as e:
            logger.warning(f"Error calculating Bat pattern strength: {str(e)}")
            return 0.0
    
    def detect_patterns(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[PatternResult]:
        """
        Detect Bat patterns in OHLCV data.
        
        Optimized Bat detection with caching and
        parallel processing for real-time trading systems.
        """
        try:
            # Validate input data
            self._validate_input_data(data)
            
            # Generate cache key for optimization
            cache_key = f"bat_{self._generate_cache_key(data, symbol, timeframe)}"
            
            # Check cache
            if cache_key in self._patterns_cache:
                logger.debug(f"Returning cached Bat patterns for {cache_key}")
                return self._patterns_cache[cache_key]
            
            # Find significant points (peaks and troughs) using ZigZag
            pivot_points = self._find_pivot_points(data)
            
            if len(pivot_points) < 5:
                logger.warning("Not enough pivot points for Bat pattern detection")
                return []
            
            # Find potential patterns Bat
            potential_patterns = self._find_bat_patterns(pivot_points, data)
            
            # Validate and score patterns
            validated_patterns = []
            for pattern_data in potential_patterns:
                pattern_result = self._validate_and_score_bat_pattern(
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
            
            logger.info(f"Detected {len(validated_patterns)} valid Bat patterns")
            return validated_patterns
            
        except Exception as e:
            logger.error(f"Error in detect_patterns: {str(e)}")
            raise RuntimeError(f"Bat pattern detection failed: {str(e)}") from e
    
    def get_entry_signals(self, pattern: PatternResult) -> Dict[str, any]:
        """
        Generate entry signals for Bat pattern trading.
        
        Bat-specific entry analysis with enhanced precision.
        """
        try:
            # Base signals from parent class
            signals = super().get_entry_signals(pattern)

            # Bat-specific modifications
            signals['entry_reason'] = f"Bat {pattern.pattern_type.value} pattern completion (88.6% AD ratio)"
            
            # Stricter entry conditions for Bat
            signals['entry_conditions']['ad_ratio_precision'] = (
                abs(pattern.ad_ratio - self.bat_fib_ratios.AD_RETRACEMENT) < self.tolerance
            )
            
            # Bat-specific timing
            signals['timing'] = {
                'immediate': pattern.confidence_score > 0.90,  # Higher threshold for Bat
                'wait_for_confirmation': 0.75 <= pattern.confidence_score <= 0.90,
                'avoid': pattern.confidence_score < 0.75
            }
            
            # Additional information about Bat pattern
            signals['bat_specifics'] = {
                'ad_ratio': pattern.ad_ratio,
                'ad_target': self.bat_fib_ratios.AD_RETRACEMENT,
                'ad_precision': 1.0 - abs(pattern.ad_ratio - self.bat_fib_ratios.AD_RETRACEMENT) / self.bat_fib_ratios.AD_RETRACEMENT,
                'pattern_quality': 'HIGH' if pattern.confidence_score > 0.85 else 'MEDIUM'
            }
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating Bat entry signals: {str(e)}")
            return {}


# Export main components
__all__ = [
    'BatPattern',
    'BatFibonacciRatios'
]