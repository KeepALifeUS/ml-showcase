"""
Fibonacci Retracement Analysis for Elliott Wave Trading.

High-precision Fibonacci analysis with crypto market
adaptations, dynamic level calculation, and confluence detection.
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
from ..patterns.impulse_wave import WavePoint

logger = get_logger(__name__)


class RetracementType(str, Enum):
    """Types of Fibonacci retracements."""
    STANDARD = "standard"           # 23.6%, 38.2%, 50%, 61.8%, 78.6%
    EXTENDED = "extended"           # Includes 88.6%, 91.4%, 94.1%
    CRYPTO_ADAPTED = "crypto_adapted"  # Adapted for crypto volatility
    CUSTOM = "custom"               # User-defined levels


class SupportResistance(str, Enum):
    """Support/Resistance classification."""
    STRONG_SUPPORT = "strong_support"
    MODERATE_SUPPORT = "moderate_support"
    WEAK_SUPPORT = "weak_support"
    STRONG_RESISTANCE = "strong_resistance"
    MODERATE_RESISTANCE = "moderate_resistance"
    WEAK_RESISTANCE = "weak_resistance"
    NEUTRAL = "neutral"


@dataclass
class RetracementLevel:
    """
    Individual Fibonacci retracement level.
    
    Rich domain model for retracement analysis.
    """
    ratio: float                    # Fibonacci ratio (e.g., 0.618)
    price: float                    # Price level
    percentage: float               # Percentage (e.g., 61.8)
    
    # Level characteristics
    level_type: RetracementType
    support_resistance: SupportResistance
    strength: float                 # 0-1 strength score
    
    # Market interaction
    touches: int = 0                # Number of times price touched this level
    bounces: int = 0                # Number of successful bounces
    breaks: int = 0                 # Number of breaks through level
    
    # Volume characteristics
    avg_volume_at_level: float = 0.0
    volume_spikes: List[float] = None
    
    # Time analysis
    first_touch: Optional[datetime] = None
    last_touch: Optional[datetime] = None
    avg_hold_time: float = 0.0      # Average time price spends near level
    
    # Confluence factors
    confluence_score: float = 0.0   # Score from other technical levels
    confluence_factors: List[str] = None
    
    def __post_init__(self):
        """Initialize calculated fields."""
        if self.volume_spikes is None:
            self.volume_spikes = []
        if self.confluence_factors is None:
            self.confluence_factors = []
            
    @property
    def success_rate(self) -> float:
        """Calculate success rate of level as support/resistance."""
        if self.touches == 0:
            return 0.0
        return self.bounces / self.touches
        
    @property
    def is_strong_level(self) -> bool:
        """Check if level is considered strong."""
        return (self.strength > 0.7 and 
                self.success_rate > 0.6 and
                self.touches >= 2)
        
    @property
    def age_days(self) -> float:
        """Calculate age of level in days."""
        if self.first_touch and self.last_touch:
            return (self.last_touch - self.first_touch).days
        return 0.0


@dataclass
class FibonacciRetracement:
    """
    Complete Fibonacci retracement analysis.
    
    Comprehensive retracement analysis with crypto adaptations.
    """
    # Source data
    swing_high: WavePoint
    swing_low: WavePoint
    direction: str                  # "bullish" or "bearish"
    
    # Retracement levels
    levels: List[RetracementLevel]
    retracement_type: RetracementType
    
    # Analysis metadata
    symbol: str
    timeframe: str
    calculated_at: datetime
    
    # Price range information
    total_range: float
    range_percentage: float         # As percentage of current price
    
    # Market context
    trend_direction: str            # Overall trend context
    volatility_adjustment: float    # Volatility-based adjustment factor
    
    # Validation and quality
    quality_score: float            # Overall quality of retracement analysis
    reliability_score: float        # Reliability based on historical performance
    
    # Performance tracking
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        """Initialize retracement analysis."""
        self.calculate_range_metrics()
        self.analyze_level_strengths()
        
    def calculate_range_metrics(self) -> None:
        """Calculate range and percentage metrics."""
        self.total_range = abs(self.swing_high.price - self.swing_low.price)
        
        current_price = max(self.swing_high.price, self.swing_low.price)
        if current_price > 0:
            self.range_percentage = (self.total_range / current_price) * 100
        else:
            self.range_percentage = 0.0
            
    def analyze_level_strengths(self) -> None:
        """Analyze strength of each retracement level."""
        for level in self.levels:
            level.strength = self._calculate_level_strength(level)
            level.support_resistance = self._classify_support_resistance(level)
            
    def _calculate_level_strength(self, level: RetracementLevel) -> float:
        """Calculate strength score for retracement level."""
        strength_factors = []
        
        # Fibonacci ratio importance (golden ratio is strongest)
        fib_importance = {
            0.236: 0.3, 0.382: 0.6, 0.5: 0.7, 
            0.618: 1.0, 0.786: 0.8, 0.886: 0.4
        }
        strength_factors.append(fib_importance.get(level.ratio, 0.2))
        
        # Historical performance
        if level.touches > 0:
            strength_factors.append(level.success_rate)
        else:
            strength_factors.append(0.5)  # Neutral for untested levels
            
        # Volume confirmation
        if level.avg_volume_at_level > 0:
            # Higher volume = stronger level
            strength_factors.append(min(level.avg_volume_at_level / 1000000, 1.0))
        else:
            strength_factors.append(0.5)
            
        # Confluence with other indicators
        strength_factors.append(min(level.confluence_score, 1.0))
        
        return np.mean(strength_factors)
        
    def _classify_support_resistance(self, level: RetracementLevel) -> SupportResistance:
        """Classify level as support or resistance."""
        current_price = max(self.swing_high.price, self.swing_low.price)
        
        if level.price < current_price:
            # Below current price = support
            if level.strength > 0.8:
                return SupportResistance.STRONG_SUPPORT
            elif level.strength > 0.6:
                return SupportResistance.MODERATE_SUPPORT
            else:
                return SupportResistance.WEAK_SUPPORT
        elif level.price > current_price:
            # Above current price = resistance
            if level.strength > 0.8:
                return SupportResistance.STRONG_RESISTANCE
            elif level.strength > 0.6:
                return SupportResistance.MODERATE_RESISTANCE
            else:
                return SupportResistance.WEAK_RESISTANCE
        else:
            return SupportResistance.NEUTRAL
            
    def get_nearest_level(self, price: float) -> Optional[RetracementLevel]:
        """Get nearest Fibonacci level to given price."""
        if not self.levels:
            return None
            
        return min(self.levels, key=lambda level: abs(level.price - price))
        
    def get_levels_in_range(self, min_price: float, max_price: float) -> List[RetracementLevel]:
        """Get all levels within price range."""
        return [level for level in self.levels 
                if min_price <= level.price <= max_price]
                
    def get_strong_levels(self) -> List[RetracementLevel]:
        """Get all strong Fibonacci levels."""
        return [level for level in self.levels if level.is_strong_level]
        
    def get_support_levels(self, current_price: float) -> List[RetracementLevel]:
        """Get support levels below current price."""
        return [level for level in self.levels 
                if level.price < current_price and 
                level.support_resistance in [
                    SupportResistance.STRONG_SUPPORT,
                    SupportResistance.MODERATE_SUPPORT,
                    SupportResistance.WEAK_SUPPORT
                ]]
                
    def get_resistance_levels(self, current_price: float) -> List[RetracementLevel]:
        """Get resistance levels above current price."""
        return [level for level in self.levels 
                if level.price > current_price and 
                level.support_resistance in [
                    SupportResistance.STRONG_RESISTANCE,
                    SupportResistance.MODERATE_RESISTANCE,
                    SupportResistance.WEAK_RESISTANCE
                ]]
                
    def calculate_target_probabilities(self) -> Dict[float, float]:
        """Calculate probability of reaching each Fibonacci level."""
        probabilities = {}
        
        for level in self.levels:
            # Base probability from Fibonacci ratio importance
            base_prob = {
                0.236: 0.8, 0.382: 0.9, 0.5: 0.75, 
                0.618: 0.85, 0.786: 0.7, 0.886: 0.5
            }.get(level.ratio, 0.3)
            
            # Adjust for level strength
            strength_adj = level.strength * 0.2
            
            # Adjust for historical success rate
            history_adj = level.success_rate * 0.1 if level.touches > 0 else 0
            
            # Adjust for confluence
            confluence_adj = level.confluence_score * 0.1
            
            probabilities[level.price] = min(
                base_prob + strength_adj + history_adj + confluence_adj, 1.0
            )
            
        return probabilities
        
    @property
    def is_valid(self) -> bool:
        """Check if retracement analysis is valid."""
        return (len(self.levels) > 0 and 
                self.total_range > 0 and
                self.quality_score > 0.5)


class FibonacciRetracementCalculator:
    """
    Advanced Fibonacci Retracement Calculator.
    
    
    - High-precision calculations
    - Crypto market adaptations
    - Dynamic level adjustment
    - Historical performance tracking
    """
    
    def __init__(self,
                 retracement_type: RetracementType = RetracementType.STANDARD,
                 include_confluence: bool = True,
                 crypto_volatility_adjustment: bool = True):
        """
        Initialize Fibonacci retracement calculator.
        
        Args:
            retracement_type: Type of retracement levels to calculate
            include_confluence: Whether to include confluence analysis
            crypto_volatility_adjustment: Apply crypto-specific adjustments
        """
        self.retracement_type = retracement_type
        self.include_confluence = include_confluence
        self.crypto_volatility_adjustment = crypto_volatility_adjustment
        
        # Performance statistics
        self.calculation_stats = {
            "total_calculations": 0,
            "valid_calculations": 0,
            "accuracy_scores": [],
            "processing_times": []
        }
        
    @performance_monitor
    async def calculate_retracement(self,
                                  swing_high: WavePoint,
                                  swing_low: WavePoint,
                                  symbol: str,
                                  timeframe: str,
                                  price_data: Optional[pd.DataFrame] = None) -> FibonacciRetracement:
        """
        Calculate Fibonacci retracement levels.
        
        Comprehensive retracement calculation with adaptations.
        
        Args:
            swing_high: Swing high point
            swing_low: Swing low point
            symbol: Trading symbol
            timeframe: Timeframe string
            price_data: Historical price data for analysis
            
        Returns:
            FibonacciRetracement: Complete retracement analysis
        """
        start_time = datetime.utcnow()
        
        # Validate inputs
        if not self._validate_swing_points(swing_high, swing_low):
            raise ValueError("Invalid swing points for retracement calculation")
            
        # Determine direction
        direction = "bullish" if swing_low.timestamp < swing_high.timestamp else "bearish"
        
        # Get Fibonacci ratios based on type
        ratios = self._get_fibonacci_ratios(self.retracement_type)
        
        # Apply crypto volatility adjustment if enabled
        if self.crypto_volatility_adjustment:
            ratios = await self._adjust_ratios_for_crypto_volatility(ratios, symbol, price_data)
            
        # Calculate retracement levels
        levels = await self._calculate_retracement_levels(
            swing_high, swing_low, ratios, direction
        )
        
        # Add confluence analysis if enabled
        if self.include_confluence and price_data is not None:
            await self._add_confluence_analysis(levels, price_data)
            
        # Add historical performance analysis
        if price_data is not None:
            await self._analyze_historical_performance(levels, price_data)
            
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Create retracement analysis
        retracement = FibonacciRetracement(
            swing_high=swing_high,
            swing_low=swing_low,
            direction=direction,
            levels=levels,
            retracement_type=self.retracement_type,
            symbol=symbol,
            timeframe=timeframe,
            calculated_at=datetime.utcnow(),
            total_range=abs(swing_high.price - swing_low.price),
            range_percentage=0.0,  # Will be calculated in __post_init__
            trend_direction=await self._determine_trend_direction(price_data) if price_data is not None else "unknown",
            volatility_adjustment=await self._calculate_volatility_adjustment(symbol, price_data) if price_data is not None else 1.0,
            quality_score=self._calculate_quality_score(levels),
            reliability_score=self._calculate_reliability_score(levels),
            processing_time_ms=processing_time
        )
        
        # Log calculation
        trading_logger.log_fibonacci_level(
            symbol=symbol,
            timeframe=timeframe,
            level_type="retracement",
            price_level=swing_high.price,  # Reference level
            support_resistance="analysis_complete"
        )
        
        # Update statistics
        self.calculation_stats["total_calculations"] += 1
        if retracement.is_valid:
            self.calculation_stats["valid_calculations"] += 1
        self.calculation_stats["processing_times"].append(processing_time)
        
        return retracement
        
    def _validate_swing_points(self, swing_high: WavePoint, swing_low: WavePoint) -> bool:
        """Validate swing points for calculation."""
        if swing_high.price <= swing_low.price:
            return False
        if swing_high.timestamp == swing_low.timestamp:
            return False
        return True
        
    def _get_fibonacci_ratios(self, retracement_type: RetracementType) -> List[float]:
        """Get Fibonacci ratios based on type."""
        ratios_map = {
            RetracementType.STANDARD: [0.236, 0.382, 0.5, 0.618, 0.786],
            RetracementType.EXTENDED: [0.236, 0.382, 0.5, 0.618, 0.786, 0.886, 0.914, 0.941],
            RetracementType.CRYPTO_ADAPTED: [0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.854],
            RetracementType.CUSTOM: config.fibonacci_config.get("retracement_levels", [0.236, 0.382, 0.5, 0.618, 0.786])
        }
        
        return ratios_map.get(retracement_type, ratios_map[RetracementType.STANDARD])
        
    async def _adjust_ratios_for_crypto_volatility(self,
                                                 ratios: List[float],
                                                 symbol: str,
                                                 price_data: Optional[pd.DataFrame]) -> List[float]:
        """Adjust Fibonacci ratios for crypto market volatility."""
        if price_data is None or len(price_data) < 20:
            return ratios
            
        # Calculate volatility
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Adjust ratios for high volatility crypto markets
        if volatility > 0.5:  # High volatility
            # Add intermediate levels for better precision
            adjusted_ratios = []
            for i, ratio in enumerate(ratios[:-1]):
                adjusted_ratios.append(ratio)
                # Add intermediate level
                intermediate = (ratio + ratios[i + 1]) / 2
                adjusted_ratios.append(intermediate)
            adjusted_ratios.append(ratios[-1])
            return adjusted_ratios
            
        return ratios
        
    async def _calculate_retracement_levels(self,
                                          swing_high: WavePoint,
                                          swing_low: WavePoint,
                                          ratios: List[float],
                                          direction: str) -> List[RetracementLevel]:
        """Calculate individual retracement levels."""
        levels = []
        price_range = swing_high.price - swing_low.price
        
        for ratio in ratios:
            # Calculate retracement price
            if direction == "bullish":
                # Retracing from high to low
                price = swing_high.price - (price_range * ratio)
            else:
                # Retracing from low to high
                price = swing_low.price + (price_range * ratio)
                
            level = RetracementLevel(
                ratio=ratio,
                price=price,
                percentage=ratio * 100,
                level_type=self.retracement_type,
                support_resistance=SupportResistance.NEUTRAL,  # Will be classified later
                strength=0.0  # Will be calculated later
            )
            
            levels.append(level)
            
        return levels
        
    async def _add_confluence_analysis(self,
                                     levels: List[RetracementLevel],
                                     price_data: pd.DataFrame) -> None:
        """Add confluence analysis with other technical levels."""
        if len(price_data) < 20:
            return
            
        # Calculate common technical levels
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        
        # Moving averages
        ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
        ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
        ma200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
        
        # Support/resistance levels from pivot points
        pivot_levels = self._calculate_pivot_levels(highs, lows, closes)
        
        # Check confluence for each Fibonacci level
        for level in levels:
            confluence_factors = []
            confluence_score = 0.0
            
            # Check MA confluence
            for ma_name, ma_value in [("MA20", ma20), ("MA50", ma50), ("MA200", ma200)]:
                if ma_value and abs(level.price - ma_value) / level.price < 0.02:  # Within 2%
                    confluence_factors.append(ma_name)
                    confluence_score += 0.2
                    
            # Check pivot level confluence
            for pivot_price in pivot_levels:
                if abs(level.price - pivot_price) / level.price < 0.01:  # Within 1%
                    confluence_factors.append("Pivot")
                    confluence_score += 0.3
                    
            level.confluence_factors = confluence_factors
            level.confluence_score = min(confluence_score, 1.0)
            
    def _calculate_pivot_levels(self,
                              highs: np.ndarray,
                              lows: np.ndarray,
                              closes: np.ndarray) -> List[float]:
        """Calculate pivot point levels."""
        if len(highs) < 3:
            return []
            
        # Simple pivot calculation
        high = np.max(highs[-20:])
        low = np.min(lows[-20:])
        close = closes[-1]
        
        pivot = (high + low + close) / 3
        
        return [
            pivot,
            pivot + (high - low) * 0.382,
            pivot + (high - low) * 0.618,
            pivot - (high - low) * 0.382,
            pivot - (high - low) * 0.618
        ]
        
    async def _analyze_historical_performance(self,
                                            levels: List[RetracementLevel],
                                            price_data: pd.DataFrame) -> None:
        """Analyze historical performance of Fibonacci levels."""
        if len(price_data) < 50:
            return
            
        closes = price_data['close'].values
        volumes = price_data.get('volume', np.zeros(len(closes))).values
        timestamps = pd.to_datetime(price_data.index)
        
        for level in levels:
            # Find touches and bounces
            touches = []
            for i, price in enumerate(closes):
                if abs(price - level.price) / level.price < 0.005:  # Within 0.5%
                    touches.append(i)
                    
            level.touches = len(touches)
            
            if touches:
                # Calculate bounces vs breaks
                bounces = 0
                breaks = 0
                
                for touch_idx in touches:
                    # Look at next 5 candles to see if it bounced or broke
                    if touch_idx + 5 < len(closes):
                        future_prices = closes[touch_idx:touch_idx + 5]
                        
                        # Determine if support or resistance held
                        if level.price < closes[touch_idx]:
                            # Acting as support
                            if np.min(future_prices) > level.price * 0.995:
                                bounces += 1
                            else:
                                breaks += 1
                        else:
                            # Acting as resistance
                            if np.max(future_prices) < level.price * 1.005:
                                bounces += 1
                            else:
                                breaks += 1
                                
                level.bounces = bounces
                level.breaks = breaks
                
                # Calculate volume at level
                touch_volumes = [volumes[i] for i in touches if i < len(volumes)]
                if touch_volumes:
                    level.avg_volume_at_level = np.mean(touch_volumes)
                    
                # Calculate time metrics
                touch_times = [timestamps[i] for i in touches if i < len(timestamps)]
                if len(touch_times) >= 2:
                    level.first_touch = touch_times[0]
                    level.last_touch = touch_times[-1]
                    
    async def _determine_trend_direction(self, price_data: pd.DataFrame) -> str:
        """Determine overall trend direction."""
        if len(price_data) < 20:
            return "unknown"
            
        closes = price_data['close'].values
        recent_trend = np.polyfit(range(len(closes[-20:])), closes[-20:], 1)[0]
        
        if recent_trend > 0:
            return "bullish"
        elif recent_trend < 0:
            return "bearish"
        else:
            return "sideways"
            
    async def _calculate_volatility_adjustment(self,
                                             symbol: str,
                                             price_data: pd.DataFrame) -> float:
        """Calculate volatility adjustment factor."""
        if len(price_data) < 20:
            return 1.0
            
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Normalize to typical crypto volatility (0.03 daily)
        adjustment = min(volatility / 0.03, 2.0)  # Cap at 2x
        return adjustment
        
    def _calculate_quality_score(self, levels: List[RetracementLevel]) -> float:
        """Calculate overall quality score for retracement analysis."""
        if not levels:
            return 0.0
            
        # Average strength of levels
        avg_strength = np.mean([level.strength for level in levels])
        
        # Confluence factor
        confluence_factor = np.mean([level.confluence_score for level in levels])
        
        # Level coverage (good spread of ratios)
        coverage_score = min(len(levels) / 5.0, 1.0)  # Optimal around 5 levels
        
        return (avg_strength * 0.5 + confluence_factor * 0.3 + coverage_score * 0.2)
        
    def _calculate_reliability_score(self, levels: List[RetracementLevel]) -> float:
        """Calculate reliability score based on historical performance."""
        if not levels:
            return 0.0
            
        # Average success rate
        success_rates = [level.success_rate for level in levels if level.touches > 0]
        if not success_rates:
            return 0.5  # Default for untested levels
            
        return np.mean(success_rates)


# Export main classes
__all__ = [
    'FibonacciRetracement',
    'FibonacciRetracementCalculator',
    'RetracementLevel',
    'RetracementType',
    'SupportResistance'
]