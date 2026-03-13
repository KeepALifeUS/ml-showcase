"""
Cumulative Delta - Cumulative analysis delta volume
enterprise patterns for analysis institutional flows
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import time
from collections import deque, defaultdict
import bisect

from ..utils.config import get_settings
from ..utils.logger import get_logger, LoggerType, performance_monitor

class CumulativeDeltaSignal(str, Enum):
    """Signals cumulative delta"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"

class TrendDirection(str, Enum):
    """Direction trend"""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

@dataclass
class CumulativeDeltaBar:
    """Bar cumulative delta"""
    timestamp: float
    open_delta: float
    high_delta: float
    low_delta: float
    close_delta: float
    volume: float
    buy_volume: float
    sell_volume: float
    net_delta: float
    
    def __post_init__(self):
        """Computation additional metrics"""
        total_volume = self.buy_volume + self.sell_volume
        self.delta_ratio = self.buy_volume / total_volume if total_volume > 0 else 0.5
        self.volume_imbalance = abs(self.buy_volume - self.sell_volume) / total_volume if total_volume > 0 else 0

@dataclass
class CumulativeDeltaDivergence:
    """Divergence cumulative delta"""
    divergence_type: str  # bullish, bearish, hidden_bullish, hidden_bearish
    price_points: List[Tuple[float, float]]  # (timestamp, price)
    delta_points: List[Tuple[float, float]]  # (timestamp, cumulative_delta)
    strength: float
    duration: int
    confidence: float

@dataclass
class DeltaProfile:
    """Profile delta by price"""
    price_level: float
    cumulative_delta: float
    volume: float
    buy_volume: float
    sell_volume: float
    trade_count: int
    avg_trade_size: float
    
    @property
    def delta_per_share(self) -> float:
        """Delta on unit volume"""
        return self.cumulative_delta / self.volume if self.volume > 0 else 0

class CumulativeDeltaAnalyzer:
    """
    Analyzer cumulative delta
    Stream Processing + Financial Analytics patterns
    """
    
    def __init__(self, symbol: str, timeframe_minutes: int = 1):
        self.symbol = symbol
        self.timeframe_minutes = timeframe_minutes
        self.settings = get_settings()
        self.logger = get_logger(LoggerType.ORDER_FLOW, f"cumulative_delta_{symbol}")
        
        # Settings analysis
        self.max_bars = 10000
        self.divergence_lookback = 100
        self.trend_periods = 20
        
        # Storage data
        self.delta_bars: deque = deque(maxlen=self.max_bars)
        self.current_bar: Optional[CumulativeDeltaBar] = None
        self.bar_start_time: float = 0
        
        # Cumulative values
        self.cumulative_delta: float = 0.0
        self.session_high_delta: float = 0.0
        self.session_low_delta: float = 0.0
        
        # Profiles delta
        self.delta_profiles: Dict[float, DeltaProfile] = {}
        self.tick_size: float = self.settings.order_flow.tick_size
        
        # Cache for optimization
        self._trend_cache: Dict[str, Any] = {}
        self._divergence_cache: List[CumulativeDeltaDivergence] = []
        
        self.logger.logger.info(
            "Cumulative delta analyzer initialized",
            symbol=symbol,
            timeframe_minutes=timeframe_minutes
        )
    
    @performance_monitor("cumulative_delta_update")
    async def add_trade(self, price: float, volume: float, is_buy: bool, timestamp: float = None) -> CumulativeDeltaBar:
        """
        Addition trades and update cumulative delta
        Real-time Stream Processing pattern
        """
        timestamp = timestamp or time.time()
        
        # Determination delta trades
        trade_delta = volume if is_buy else -volume
        
        # Update cumulative delta
        self.cumulative_delta += trade_delta
        
        # Update session extremes
        self.session_high_delta = max(self.session_high_delta, self.cumulative_delta)
        self.session_low_delta = min(self.session_low_delta, self.cumulative_delta)
        
        # Validation necessity creation new bar
        if await self._should_create_new_bar(timestamp):
            await self._finalize_current_bar()
            await self._start_new_bar(timestamp)
        
        # Update current bar
        await self._update_current_bar(price, volume, is_buy, timestamp, trade_delta)
        
        # Update profile delta
        await self._update_delta_profile(price, volume, is_buy, trade_delta)
        
        return self.current_bar
    
    async def _should_create_new_bar(self, timestamp: float) -> bool:
        """Validation necessity creation new bar"""
        if self.bar_start_time == 0:
            return True
        
        bar_duration = self.timeframe_minutes * 60  # in seconds
        return timestamp >= self.bar_start_time + bar_duration
    
    async def _start_new_bar(self, timestamp: float) -> None:
        """Start new bar"""
        self.bar_start_time = timestamp
        self.current_bar = CumulativeDeltaBar(
            timestamp=timestamp,
            open_delta=self.cumulative_delta,
            high_delta=self.cumulative_delta,
            low_delta=self.cumulative_delta,
            close_delta=self.cumulative_delta,
            volume=0.0,
            buy_volume=0.0,
            sell_volume=0.0,
            net_delta=0.0
        )
    
    async def _finalize_current_bar(self) -> None:
        """Completion current bar"""
        if self.current_bar is not None:
            self.current_bar.close_delta = self.cumulative_delta
            self.delta_bars.append(self.current_bar)
            
            # Logging completion bar
            self.logger.logger.debug(
                "Bar finalized",
                symbol=self.symbol,
                timestamp=self.current_bar.timestamp,
                delta_change=self.current_bar.close_delta - self.current_bar.open_delta,
                volume=self.current_bar.volume
            )
    
    async def _update_current_bar(self, price: float, volume: float, is_buy: bool, 
                                 timestamp: float, trade_delta: float) -> None:
        """Update current bar"""
        if self.current_bar is None:
            await self._start_new_bar(timestamp)
        
        # Update OHLC delta
        self.current_bar.high_delta = max(self.current_bar.high_delta, self.cumulative_delta)
        self.current_bar.low_delta = min(self.current_bar.low_delta, self.cumulative_delta)
        self.current_bar.close_delta = self.cumulative_delta
        
        # Update volumes
        self.current_bar.volume += volume
        if is_buy:
            self.current_bar.buy_volume += volume
        else:
            self.current_bar.sell_volume += volume
        
        self.current_bar.net_delta += trade_delta
    
    async def _update_delta_profile(self, price: float, volume: float, is_buy: bool, trade_delta: float) -> None:
        """Update profile delta by levels prices"""
        # Rounding price until tick size
        price_level = round(price / self.tick_size) * self.tick_size
        
        if price_level not in self.delta_profiles:
            self.delta_profiles[price_level] = DeltaProfile(
                price_level=price_level,
                cumulative_delta=0.0,
                volume=0.0,
                buy_volume=0.0,
                sell_volume=0.0,
                trade_count=0,
                avg_trade_size=0.0
            )
        
        profile = self.delta_profiles[price_level]
        
        # Update profile
        profile.cumulative_delta += trade_delta
        profile.volume += volume
        if is_buy:
            profile.buy_volume += volume
        else:
            profile.sell_volume += volume
        
        profile.trade_count += 1
        profile.avg_trade_size = profile.volume / profile.trade_count
    
    @performance_monitor("delta_divergence_detection")
    async def detect_divergences(self, price_data: List[Tuple[float, float]], lookback: int = None) -> List[CumulativeDeltaDivergence]:
        """
        Detection divergences between price and cumulative delta
        Pattern Recognition pattern
        """
        lookback = lookback or self.divergence_lookback
        
        if len(self.delta_bars) < lookback or len(price_data) < lookback:
            return []
        
        # Retrieval data for analysis
        recent_bars = list(self.delta_bars)[-lookback:]
        recent_prices = price_data[-lookback:] if len(price_data) >= lookback else price_data
        
        divergences = []
        
        # Detection various types divergences
        divergences.extend(await self._detect_regular_divergences(recent_bars, recent_prices))
        divergences.extend(await self._detect_hidden_divergences(recent_bars, recent_prices))
        
        # Update cache
        self._divergence_cache = divergences
        
        return divergences
    
    async def _detect_regular_divergences(self, bars: List[CumulativeDeltaBar], 
                                        prices: List[Tuple[float, float]]) -> List[CumulativeDeltaDivergence]:
        """Detection regular divergences"""
        divergences = []
        
        if len(bars) < 20 or len(prices) < 20:
            return divergences
        
        # Search local extremes price
        price_highs, price_lows = await self._find_price_extremes(prices)
        
        # Search corresponding extremes delta
        delta_values = [(bar.timestamp, bar.close_delta) for bar in bars]
        delta_highs, delta_lows = await self._find_delta_extremes(delta_values)
        
        # Analysis bullish divergences (price falls, delta grows)
        for i in range(1, len(price_lows)):
            if i >= len(delta_lows):
                break
                
            price_low_1 = price_lows[i-1]
            price_low_2 = price_lows[i]
            
            # Search corresponding points delta
            delta_low_1 = await self._find_closest_delta_point(delta_lows, price_low_1[0])
            delta_low_2 = await self._find_closest_delta_point(delta_lows, price_low_2[0])
            
            if delta_low_1 and delta_low_2:
                # Validation conditions divergences
                price_falling = price_low_2[1] < price_low_1[1]
                delta_rising = delta_low_2[1] > delta_low_1[1]
                
                if price_falling and delta_rising:
                    strength = abs(delta_low_2[1] - delta_low_1[1]) / max(abs(delta_low_1[1]), 1)
                    confidence = min(strength * 0.7, 0.9)
                    
                    divergences.append(CumulativeDeltaDivergence(
                        divergence_type="bullish",
                        price_points=[price_low_1, price_low_2],
                        delta_points=[delta_low_1, delta_low_2],
                        strength=strength,
                        duration=int(price_low_2[0] - price_low_1[0]),
                        confidence=confidence
                    ))
        
        # Analysis bearish divergences (price grows, delta falls)
        for i in range(1, len(price_highs)):
            if i >= len(delta_highs):
                break
                
            price_high_1 = price_highs[i-1]
            price_high_2 = price_highs[i]
            
            delta_high_1 = await self._find_closest_delta_point(delta_highs, price_high_1[0])
            delta_high_2 = await self._find_closest_delta_point(delta_highs, price_high_2[0])
            
            if delta_high_1 and delta_high_2:
                price_rising = price_high_2[1] > price_high_1[1]
                delta_falling = delta_high_2[1] < delta_high_1[1]
                
                if price_rising and delta_falling:
                    strength = abs(delta_high_1[1] - delta_high_2[1]) / max(abs(delta_high_1[1]), 1)
                    confidence = min(strength * 0.7, 0.9)
                    
                    divergences.append(CumulativeDeltaDivergence(
                        divergence_type="bearish",
                        price_points=[price_high_1, price_high_2],
                        delta_points=[delta_high_1, delta_high_2],
                        strength=strength,
                        duration=int(price_high_2[0] - price_high_1[0]),
                        confidence=confidence
                    ))
        
        return divergences
    
    async def _detect_hidden_divergences(self, bars: List[CumulativeDeltaBar], 
                                       prices: List[Tuple[float, float]]) -> List[CumulativeDeltaDivergence]:
        """Detection hidden divergences"""
        # Hidden divergences indicate on continuation trend
        divergences = []
        
        # Determination current trend
        trend = await self._determine_trend(prices)
        
        if trend == TrendDirection.SIDEWAYS:
            return divergences
        
        # Search hidden divergences in dependencies from trend
        if trend == TrendDirection.UP:
            # IN uptrend search hidden bullish divergence
            # Price does more high low, delta - more low low
            price_lows = await self._find_price_extremes(prices, extremes_type="lows")
            delta_values = [(bar.timestamp, bar.close_delta) for bar in bars]
            delta_lows = await self._find_delta_extremes(delta_values, extremes_type="lows")
            
            for i in range(1, min(len(price_lows), len(delta_lows))):
                price_low_1, price_low_2 = price_lows[i-1], price_lows[i]
                delta_low_1 = await self._find_closest_delta_point(delta_lows, price_low_1[0])
                delta_low_2 = await self._find_closest_delta_point(delta_lows, price_low_2[0])
                
                if delta_low_1 and delta_low_2:
                    higher_price_low = price_low_2[1] > price_low_1[1]
                    lower_delta_low = delta_low_2[1] < delta_low_1[1]
                    
                    if higher_price_low and lower_delta_low:
                        strength = abs(delta_low_1[1] - delta_low_2[1]) / max(abs(delta_low_1[1]), 1)
                        
                        divergences.append(CumulativeDeltaDivergence(
                            divergence_type="hidden_bullish",
                            price_points=[price_low_1, price_low_2],
                            delta_points=[delta_low_1, delta_low_2],
                            strength=strength,
                            duration=int(price_low_2[0] - price_low_1[0]),
                            confidence=min(strength * 0.6, 0.8)
                        ))
        
        return divergences
    
    async def _find_price_extremes(self, prices: List[Tuple[float, float]], extremes_type: str = "both") -> Union[Tuple[List, List], List]:
        """Search local extremes price"""
        if len(prices) < 5:
            return ([], []) if extremes_type == "both" else []
        
        price_values = [p[1] for p in prices]
        timestamps = [p[0] for p in prices]
        
        highs = []
        lows = []
        
        # Search local maximums and minimums
        for i in range(2, len(price_values) - 2):
            # Local maximum
            if (price_values[i] > price_values[i-1] and price_values[i] > price_values[i+1] and
                price_values[i] > price_values[i-2] and price_values[i] > price_values[i+2]):
                highs.append((timestamps[i], price_values[i]))
            
            # Local minimum
            if (price_values[i] < price_values[i-1] and price_values[i] < price_values[i+1] and
                price_values[i] < price_values[i-2] and price_values[i] < price_values[i+2]):
                lows.append((timestamps[i], price_values[i]))
        
        if extremes_type == "highs":
            return highs
        elif extremes_type == "lows":
            return lows
        else:
            return highs, lows
    
    async def _find_delta_extremes(self, delta_values: List[Tuple[float, float]], extremes_type: str = "both") -> Union[Tuple[List, List], List]:
        """Search extremes cumulative delta"""
        return await self._find_price_extremes(delta_values, extremes_type)
    
    async def _find_closest_delta_point(self, delta_points: List[Tuple[float, float]], target_timestamp: float) -> Optional[Tuple[float, float]]:
        """Search nearest points delta to specified time"""
        if not delta_points:
            return None
        
        # Search nearest points by time
        closest_point = min(delta_points, key=lambda x: abs(x[0] - target_timestamp))
        
        # Validation that point sufficient close (in within 5 minutes)
        if abs(closest_point[0] - target_timestamp) <= 300:  # 5 minutes
            return closest_point
        
        return None
    
    async def _determine_trend(self, prices: List[Tuple[float, float]], periods: int = None) -> TrendDirection:
        """Determination directions trend"""
        periods = periods or self.trend_periods
        
        if len(prices) < periods:
            return TrendDirection.SIDEWAYS
        
        recent_prices = prices[-periods:]
        price_values = [p[1] for p in recent_prices]
        
        # Linear regression for determination trend
        x = np.arange(len(price_values))
        slope = np.polyfit(x, price_values, 1)[0]
        
        # Threshold values for determination trend
        price_range = max(price_values) - min(price_values)
        trend_threshold = price_range * 0.02  # 2% from range prices
        
        if slope > trend_threshold:
            return TrendDirection.UP
        elif slope < -trend_threshold:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    async def get_delta_strength_signal(self) -> Dict[str, Any]:
        """
        Retrieval signal on basis force delta
        Signal Processing pattern
        """
        if len(self.delta_bars) < 10:
            return {'signal': CumulativeDeltaSignal.NEUTRAL, 'confidence': 0.0}
        
        recent_bars = list(self.delta_bars)[-10:]
        
        # Analysis trend delta
        delta_changes = [bar.close_delta - bar.open_delta for bar in recent_bars]
        positive_changes = sum(1 for change in delta_changes if change > 0)
        negative_changes = sum(1 for change in delta_changes if change < 0)
        
        # Analysis volumetric imbalance
        total_buy_volume = sum(bar.buy_volume for bar in recent_bars)
        total_sell_volume = sum(bar.sell_volume for bar in recent_bars)
        total_volume = total_buy_volume + total_sell_volume
        
        buy_ratio = total_buy_volume / total_volume if total_volume > 0 else 0.5
        
        # Determination signal
        signal = CumulativeDeltaSignal.NEUTRAL
        confidence = 0.0
        
        if positive_changes >= 7 and buy_ratio > 0.6:  # Strong growth delta
            signal = CumulativeDeltaSignal.STRONG_BUY
            confidence = 0.8
        elif positive_changes >= 6 and buy_ratio > 0.55:
            signal = CumulativeDeltaSignal.BUY
            confidence = 0.6
        elif negative_changes >= 7 and buy_ratio < 0.4:
            signal = CumulativeDeltaSignal.STRONG_SELL
            confidence = 0.8
        elif negative_changes >= 6 and buy_ratio < 0.45:
            signal = CumulativeDeltaSignal.SELL
            confidence = 0.6
        
        # Analysis accumulation/distribution
        delta_momentum = sum(delta_changes[-5:])  # Recent 5 bars
        if abs(delta_momentum) < sum(abs(change) for change in delta_changes[-5:]) * 0.2:
            if buy_ratio > 0.6:
                signal = CumulativeDeltaSignal.ACCUMULATION
                confidence = 0.7
            elif buy_ratio < 0.4:
                signal = CumulativeDeltaSignal.DISTRIBUTION
                confidence = 0.7
        
        return {
            'signal': signal,
            'confidence': confidence,
            'buy_ratio': buy_ratio,
            'delta_momentum': delta_momentum,
            'positive_changes': positive_changes,
            'negative_changes': negative_changes
        }
    
    def get_delta_profile_levels(self, top_n: int = 20) -> List[DeltaProfile]:
        """Retrieval top levels by delta"""
        profiles = list(self.delta_profiles.values())
        
        # Sorting by volume delta
        profiles.sort(key=lambda p: abs(p.cumulative_delta), reverse=True)
        
        return profiles[:top_n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieval statistics analyzer"""
        if not self.delta_bars:
            return {}
        
        recent_bars = list(self.delta_bars)[-100:] if len(self.delta_bars) >= 100 else list(self.delta_bars)
        
        return {
            'total_bars': len(self.delta_bars),
            'cumulative_delta': self.cumulative_delta,
            'session_high_delta': self.session_high_delta,
            'session_low_delta': self.session_low_delta,
            'avg_bar_volume': np.mean([bar.volume for bar in recent_bars]),
            'avg_buy_ratio': np.mean([bar.buy_volume / (bar.buy_volume + bar.sell_volume) 
                                    for bar in recent_bars if (bar.buy_volume + bar.sell_volume) > 0]),
            'delta_range': self.session_high_delta - self.session_low_delta,
            'profile_levels': len(self.delta_profiles),
            'active_divergences': len(self._divergence_cache)
        }
    
    async def reset_session(self) -> None:
        """Reset session data"""
        self.cumulative_delta = 0.0
        self.session_high_delta = 0.0
        self.session_low_delta = 0.0
        self.delta_profiles.clear()
        self._trend_cache.clear()
        self._divergence_cache.clear()
        
        self.logger.logger.info("Session reset", symbol=self.symbol)

# Utility functions
async def calculate_delta_momentum(bars: List[CumulativeDeltaBar], period: int = 14) -> float:
    """
    Calculation momentum cumulative delta
    Technical Analysis pattern
    """
    if len(bars) < period + 1:
        return 0.0
    
    current_delta = bars[-1].close_delta
    previous_delta = bars[-period-1].close_delta
    
    return current_delta - previous_delta

async def calculate_delta_rsi(bars: List[CumulativeDeltaBar], period: int = 14) -> float:
    """Calculation RSI for cumulative delta"""
    if len(bars) < period + 1:
        return 50.0
    
    delta_changes = []
    for i in range(1, len(bars)):
        delta_changes.append(bars[i].close_delta - bars[i-1].close_delta)
    
    if len(delta_changes) < period:
        return 50.0
    
    recent_changes = delta_changes[-period:]
    gains = [change for change in recent_changes if change > 0]
    losses = [-change for change in recent_changes if change < 0]
    
    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Export main components
__all__ = [
    'CumulativeDeltaAnalyzer',
    'CumulativeDeltaBar',
    'CumulativeDeltaDivergence',
    'DeltaProfile',
    'CumulativeDeltaSignal',
    'TrendDirection',
    'calculate_delta_momentum',
    'calculate_delta_rsi'
]