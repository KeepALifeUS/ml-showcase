"""
Footprint Chart Analysis - Analysis trace volume by levels prices
enterprise patterns for detailed analysis order flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import time
from collections import defaultdict, deque
import bisect

from ..utils.config import get_settings
from ..utils.logger import get_logger, LoggerType, performance_monitor

class FootprintType(str, Enum):
    """Types footprint analysis"""
    VOLUME = "volume"
    DELTA = "delta"
    BID_ASK = "bid_ask"
    COMPOSITE = "composite"

class LevelImbalance(str, Enum):
    """Types imbalance levels"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_TRADE = "no_trade"

@dataclass
class PriceLevel:
    """Level price in footprint"""
    price: float
    bid_volume: float = 0.0
    ask_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    total_volume: float = 0.0
    trade_count: int = 0
    avg_trade_size: float = 0.0
    first_trade_time: Optional[float] = None
    last_trade_time: Optional[float] = None
    
    @property
    def net_delta(self) -> float:
        """Net delta level"""
        return self.buy_volume - self.sell_volume
    
    @property
    def delta_percentage(self) -> float:
        """Percent delta purchases"""
        return (self.buy_volume / self.total_volume) if self.total_volume > 0 else 0.5
    
    @property
    def imbalance_ratio(self) -> float:
        """Coefficient imbalance"""
        if self.sell_volume == 0:
            return float('inf') if self.buy_volume > 0 else 0
        return self.buy_volume / self.sell_volume
    
    def get_imbalance_level(self, thresholds: Dict[str, float]) -> LevelImbalance:
        """Determination level imbalance"""
        if self.total_volume == 0:
            return LevelImbalance.NO_TRADE
        
        delta_pct = self.delta_percentage
        
        if delta_pct >= thresholds.get('strong_buy', 0.8):
            return LevelImbalance.STRONG_BUY
        elif delta_pct >= thresholds.get('buy', 0.65):
            return LevelImbalance.BUY
        elif delta_pct <= thresholds.get('strong_sell', 0.2):
            return LevelImbalance.STRONG_SELL
        elif delta_pct <= thresholds.get('sell', 0.35):
            return LevelImbalance.SELL
        else:
            return LevelImbalance.NEUTRAL

@dataclass
class FootprintBar:
    """Footprint bar"""
    timestamp: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    price_levels: Dict[float, PriceLevel] = field(default_factory=dict)
    
    @property
    def price_range(self) -> float:
        """Range prices bar"""
        return self.high_price - self.low_price
    
    @property
    def total_buy_volume(self) -> float:
        """Total volume purchases"""
        return sum(level.buy_volume for level in self.price_levels.values())
    
    @property
    def total_sell_volume(self) -> float:
        """Total volume sales"""
        return sum(level.sell_volume for level in self.price_levels.values())
    
    @property
    def net_delta(self) -> float:
        """Net delta bar"""
        return self.total_buy_volume - self.total_sell_volume
    
    def get_volume_at_price(self, price: float) -> float:
        """Retrieval volume on level price"""
        return self.price_levels.get(price, PriceLevel(price)).total_volume
    
    def get_delta_at_price(self, price: float) -> float:
        """Retrieval delta on level price"""
        return self.price_levels.get(price, PriceLevel(price)).net_delta

@dataclass
class FootprintPattern:
    """Pattern footprint"""
    pattern_name: str
    confidence: float
    price_levels: List[float]
    volume_profile: Dict[float, float]
    delta_profile: Dict[float, float]
    imbalance_levels: Dict[float, LevelImbalance]
    strength: float
    duration: int
    timestamp: float

class FootprintAnalyzer:
    """
    Analyzer Footprint Chart
    High-Resolution Market Analysis pattern
    """
    
    def __init__(self, symbol: str, timeframe_seconds: int = 60, tick_size: float = None):
        self.symbol = symbol
        self.timeframe_seconds = timeframe_seconds
        self.settings = get_settings()
        self.logger = get_logger(LoggerType.ORDER_FLOW, f"footprint_{symbol}")
        
        # Settings
        self.tick_size = tick_size or self.settings.order_flow.tick_size
        self.max_bars = 1000
        self.max_price_levels_per_bar = 200
        
        # Thresholds imbalance
        self.imbalance_thresholds = {
            'strong_buy': 0.8,
            'buy': 0.65,
            'sell': 0.35,
            'strong_sell': 0.2
        }
        
        # Storage data
        self.footprint_bars: deque = deque(maxlen=self.max_bars)
        self.current_bar: Optional[FootprintBar] = None
        self.bar_start_time: float = 0
        
        # Caches for optimization
        self._pattern_cache: List[FootprintPattern] = []
        self._volume_profile_cache: Dict[float, float] = {}
        
        self.logger.logger.info(
            "Footprint analyzer initialized",
            symbol=symbol,
            timeframe_seconds=timeframe_seconds,
            tick_size=self.tick_size
        )
    
    @performance_monitor("footprint_update")
    async def add_trade(self, price: float, volume: float, is_buy: bool, 
                       bid: float = None, ask: float = None, timestamp: float = None) -> FootprintBar:
        """
        Addition trades in footprint analysis
        High-Frequency Data Processing pattern
        """
        timestamp = timestamp or time.time()
        
        # Validation necessity creation new bar
        if await self._should_create_new_bar(timestamp):
            await self._finalize_current_bar()
            await self._start_new_bar(timestamp, price)
        
        # Update current bar
        await self._update_current_bar(price, volume, is_buy, bid, ask, timestamp)
        
        return self.current_bar
    
    async def _should_create_new_bar(self, timestamp: float) -> bool:
        """Validation necessity creation new bar"""
        if self.bar_start_time == 0:
            return True
        
        return timestamp >= self.bar_start_time + self.timeframe_seconds
    
    async def _start_new_bar(self, timestamp: float, price: float) -> None:
        """Creation new bar"""
        self.bar_start_time = timestamp
        self.current_bar = FootprintBar(
            timestamp=timestamp,
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
            volume=0.0
        )
    
    async def _finalize_current_bar(self) -> None:
        """Completion current bar"""
        if self.current_bar is not None:
            self.footprint_bars.append(self.current_bar)
            
            # Update caches
            await self._update_volume_profile_cache(self.current_bar)
            
            self.logger.logger.debug(
                "Footprint bar finalized",
                symbol=self.symbol,
                timestamp=self.current_bar.timestamp,
                price_levels=len(self.current_bar.price_levels),
                volume=self.current_bar.volume
            )
    
    async def _update_current_bar(self, price: float, volume: float, is_buy: bool,
                                 bid: float, ask: float, timestamp: float) -> None:
        """Update current bar"""
        if self.current_bar is None:
            await self._start_new_bar(timestamp, price)
        
        # Update OHLC
        self.current_bar.high_price = max(self.current_bar.high_price, price)
        self.current_bar.low_price = min(self.current_bar.low_price, price)
        self.current_bar.close_price = price
        self.current_bar.volume += volume
        
        # Rounding price until tick size
        price_level = round(price / self.tick_size) * self.tick_size
        
        # Creation or update level price
        if price_level not in self.current_bar.price_levels:
            if len(self.current_bar.price_levels) >= self.max_price_levels_per_bar:
                # Removal level with smallest volume
                min_volume_price = min(self.current_bar.price_levels.keys(),
                                     key=lambda p: self.current_bar.price_levels[p].total_volume)
                del self.current_bar.price_levels[min_volume_price]
            
            self.current_bar.price_levels[price_level] = PriceLevel(
                price=price_level,
                first_trade_time=timestamp
            )
        
        level = self.current_bar.price_levels[price_level]
        
        # Update level
        if is_buy:
            level.buy_volume += volume
            if ask is not None:
                level.ask_volume += volume
        else:
            level.sell_volume += volume
            if bid is not None:
                level.bid_volume += volume
        
        level.total_volume += volume
        level.trade_count += 1
        level.avg_trade_size = level.total_volume / level.trade_count
        level.last_trade_time = timestamp
    
    @performance_monitor("footprint_pattern_detection")
    async def detect_patterns(self, lookback_bars: int = 20) -> List[FootprintPattern]:
        """
        Detection footprint patterns
        Pattern Recognition pattern
        """
        if len(self.footprint_bars) < lookback_bars:
            return []
        
        recent_bars = list(self.footprint_bars)[-lookback_bars:]
        patterns = []
        
        # Various types patterns
        patterns.extend(await self._detect_absorption_patterns(recent_bars))
        patterns.extend(await self._detect_imbalance_clusters(recent_bars))
        patterns.extend(await self._detect_volume_nodes(recent_bars))
        patterns.extend(await self._detect_single_prints(recent_bars))
        patterns.extend(await self._detect_poor_highs_lows(recent_bars))
        
        self._pattern_cache = patterns
        
        # Logging patterns
        for pattern in patterns:
            self.logger.log_pattern_detected(
                pattern_type=pattern.pattern_name,
                symbol=self.symbol,
                confidence=pattern.confidence
            )
        
        return patterns
    
    async def _detect_absorption_patterns(self, bars: List[FootprintBar]) -> List[FootprintPattern]:
        """Detection patterns absorption"""
        patterns = []
        
        absorption_threshold = 2.0  # Minimum ratio volumes for absorption
        
        for i in range(1, len(bars)):
            current_bar = bars[i]
            previous_bar = bars[i-1]
            
            # Analysis of each level current bar
            for price_level, level in current_bar.price_levels.items():
                if level.total_volume == 0:
                    continue
                
                # Search absorption sales purchases
                if (level.imbalance_ratio >= absorption_threshold and 
                    level.buy_volume > level.sell_volume * absorption_threshold):
                    
                    # Validation context - was whether previous bar bearish
                    if previous_bar.net_delta < 0:
                        patterns.append(FootprintPattern(
                            pattern_name="buy_absorption",
                            confidence=min(level.imbalance_ratio / 5.0, 0.9),
                            price_levels=[price_level],
                            volume_profile={price_level: level.total_volume},
                            delta_profile={price_level: level.net_delta},
                            imbalance_levels={price_level: level.get_imbalance_level(self.imbalance_thresholds)},
                            strength=level.buy_volume / current_bar.volume,
                            duration=1,
                            timestamp=current_bar.timestamp
                        ))
                
                # Search absorption purchases sales
                elif (level.sell_volume > 0 and level.buy_volume > 0 and
                      level.sell_volume > level.buy_volume * absorption_threshold):
                    
                    if previous_bar.net_delta > 0:
                        patterns.append(FootprintPattern(
                            pattern_name="sell_absorption",
                            confidence=min((level.sell_volume / level.buy_volume) / 5.0, 0.9),
                            price_levels=[price_level],
                            volume_profile={price_level: level.total_volume},
                            delta_profile={price_level: level.net_delta},
                            imbalance_levels={price_level: level.get_imbalance_level(self.imbalance_thresholds)},
                            strength=level.sell_volume / current_bar.volume,
                            duration=1,
                            timestamp=current_bar.timestamp
                        ))
        
        return patterns
    
    async def _detect_imbalance_clusters(self, bars: List[FootprintBar]) -> List[FootprintPattern]:
        """Detection clusters imbalance"""
        patterns = []
        
        # Grouping neighboring levels with same imbalance
        for bar in bars[-5:]:  # Analysis recent 5 bars
            buy_clusters = []
            sell_clusters = []
            current_buy_cluster = []
            current_sell_cluster = []
            
            # Sorting levels by price
            sorted_levels = sorted(bar.price_levels.items())
            
            for price, level in sorted_levels:
                imbalance = level.get_imbalance_level(self.imbalance_thresholds)
                
                if imbalance in [LevelImbalance.BUY, LevelImbalance.STRONG_BUY]:
                    current_buy_cluster.append((price, level))
                    if current_sell_cluster and len(current_sell_cluster) >= 3:
                        sell_clusters.append(current_sell_cluster.copy())
                    current_sell_cluster.clear()
                    
                elif imbalance in [LevelImbalance.SELL, LevelImbalance.STRONG_SELL]:
                    current_sell_cluster.append((price, level))
                    if current_buy_cluster and len(current_buy_cluster) >= 3:
                        buy_clusters.append(current_buy_cluster.copy())
                    current_buy_cluster.clear()
                    
                else:
                    # Neutral level - completion clusters
                    if current_buy_cluster and len(current_buy_cluster) >= 3:
                        buy_clusters.append(current_buy_cluster.copy())
                    if current_sell_cluster and len(current_sell_cluster) >= 3:
                        sell_clusters.append(current_sell_cluster.copy())
                    current_buy_cluster.clear()
                    current_sell_cluster.clear()
            
            # Finalization remaining clusters
            if len(current_buy_cluster) >= 3:
                buy_clusters.append(current_buy_cluster)
            if len(current_sell_cluster) >= 3:
                sell_clusters.append(current_sell_cluster)
            
            # Creation patterns for clusters
            for cluster in buy_clusters:
                total_volume = sum(level.total_volume for _, level in cluster)
                total_delta = sum(level.net_delta for _, level in cluster)
                
                patterns.append(FootprintPattern(
                    pattern_name="buy_imbalance_cluster",
                    confidence=min(len(cluster) / 10.0, 0.85),
                    price_levels=[price for price, _ in cluster],
                    volume_profile={price: level.total_volume for price, level in cluster},
                    delta_profile={price: level.net_delta for price, level in cluster},
                    imbalance_levels={price: level.get_imbalance_level(self.imbalance_thresholds) 
                                    for price, level in cluster},
                    strength=total_volume / bar.volume,
                    duration=1,
                    timestamp=bar.timestamp
                ))
            
            for cluster in sell_clusters:
                total_volume = sum(level.total_volume for _, level in cluster)
                
                patterns.append(FootprintPattern(
                    pattern_name="sell_imbalance_cluster",
                    confidence=min(len(cluster) / 10.0, 0.85),
                    price_levels=[price for price, _ in cluster],
                    volume_profile={price: level.total_volume for price, level in cluster},
                    delta_profile={price: level.net_delta for price, level in cluster},
                    imbalance_levels={price: level.get_imbalance_level(self.imbalance_thresholds) 
                                    for price, level in cluster},
                    strength=total_volume / bar.volume,
                    duration=1,
                    timestamp=bar.timestamp
                ))
        
        return patterns
    
    async def _detect_volume_nodes(self, bars: List[FootprintBar]) -> List[FootprintPattern]:
        """Detection nodes volume"""
        patterns = []
        
        # Collection all levels for period
        all_levels = {}
        for bar in bars:
            for price, level in bar.price_levels.items():
                if price not in all_levels:
                    all_levels[price] = {'volume': 0, 'delta': 0, 'occurrences': 0}
                all_levels[price]['volume'] += level.total_volume
                all_levels[price]['delta'] += level.net_delta
                all_levels[price]['occurrences'] += 1
        
        if not all_levels:
            return patterns
        
        # Search significant nodes volume
        avg_volume = np.mean([data['volume'] for data in all_levels.values()])
        volume_threshold = avg_volume * 2  # Node must be in 2 times more average
        
        volume_nodes = []
        for price, data in all_levels.items():
            if (data['volume'] > volume_threshold and 
                data['occurrences'] >= 3):  # Level was traded minimum 3 times
                volume_nodes.append((price, data))
        
        # Creation patterns for nodes
        for price, data in volume_nodes:
            avg_delta_per_occurrence = data['delta'] / data['occurrences']
            
            pattern_name = "volume_node"
            if avg_delta_per_occurrence > 0:
                pattern_name = "bullish_volume_node"
            elif avg_delta_per_occurrence < 0:
                pattern_name = "bearish_volume_node"
            
            patterns.append(FootprintPattern(
                pattern_name=pattern_name,
                confidence=min(data['volume'] / (avg_volume * 10), 0.8),
                price_levels=[price],
                volume_profile={price: data['volume']},
                delta_profile={price: data['delta']},
                imbalance_levels={},
                strength=data['volume'] / sum(d['volume'] for d in all_levels.values()),
                duration=data['occurrences'],
                timestamp=bars[-1].timestamp
            ))
        
        return patterns
    
    async def _detect_single_prints(self, bars: List[FootprintBar]) -> List[FootprintPattern]:
        """Detection single prints (levels, which were traded only one times)"""
        patterns = []
        
        if len(bars) < 3:
            return patterns
        
        for i in range(1, len(bars) - 1):
            current_bar = bars[i]
            prev_bar = bars[i-1]
            next_bar = bars[i+1]
            
            # Search levels, which were traded only in current bar
            for price, level in current_bar.price_levels.items():
                # Validation, that level not was traded in neighboring bars
                prev_traded = price in prev_bar.price_levels
                next_traded = price in next_bar.price_levels
                
                if not prev_traded and not next_traded and level.total_volume > 0:
                    # Determination type single print
                    pattern_name = "single_print"
                    confidence = 0.6
                    
                    # If single print on extreme bar
                    if price == current_bar.high_price:
                        pattern_name = "single_print_high"
                        confidence = 0.7
                    elif price == current_bar.low_price:
                        pattern_name = "single_print_low"
                        confidence = 0.7
                    
                    patterns.append(FootprintPattern(
                        pattern_name=pattern_name,
                        confidence=confidence,
                        price_levels=[price],
                        volume_profile={price: level.total_volume},
                        delta_profile={price: level.net_delta},
                        imbalance_levels={price: level.get_imbalance_level(self.imbalance_thresholds)},
                        strength=level.total_volume / current_bar.volume,
                        duration=1,
                        timestamp=current_bar.timestamp
                    ))
        
        return patterns
    
    async def _detect_poor_highs_lows(self, bars: List[FootprintBar]) -> List[FootprintPattern]:
        """Detection poor highs/lows (extremes with low volume)"""
        patterns = []
        
        volume_threshold_percentile = 30  # Lower 30% by volume are considered "poor"
        
        for bar in bars[-10:]:  # Analysis recent 10 bars
            if len(bar.price_levels) < 5:
                continue
            
            # Retrieval volumes all levels
            volumes = [level.total_volume for level in bar.price_levels.values()]
            threshold_volume = np.percentile(volumes, volume_threshold_percentile)
            
            # Validation maximum
            high_level = bar.price_levels.get(bar.high_price)
            if high_level and high_level.total_volume <= threshold_volume:
                patterns.append(FootprintPattern(
                    pattern_name="poor_high",
                    confidence=0.7,
                    price_levels=[bar.high_price],
                    volume_profile={bar.high_price: high_level.total_volume},
                    delta_profile={bar.high_price: high_level.net_delta},
                    imbalance_levels={bar.high_price: high_level.get_imbalance_level(self.imbalance_thresholds)},
                    strength=1.0 - (high_level.total_volume / max(volumes)),
                    duration=1,
                    timestamp=bar.timestamp
                ))
            
            # Validation minimum
            low_level = bar.price_levels.get(bar.low_price)
            if low_level and low_level.total_volume <= threshold_volume:
                patterns.append(FootprintPattern(
                    pattern_name="poor_low",
                    confidence=0.7,
                    price_levels=[bar.low_price],
                    volume_profile={bar.low_price: low_level.total_volume},
                    delta_profile={bar.low_price: low_level.net_delta},
                    imbalance_levels={bar.low_price: low_level.get_imbalance_level(self.imbalance_thresholds)},
                    strength=1.0 - (low_level.total_volume / max(volumes)),
                    duration=1,
                    timestamp=bar.timestamp
                ))
        
        return patterns
    
    async def _update_volume_profile_cache(self, bar: FootprintBar) -> None:
        """Update cache volume profile"""
        for price, level in bar.price_levels.items():
            if price not in self._volume_profile_cache:
                self._volume_profile_cache[price] = 0
            self._volume_profile_cache[price] += level.total_volume
        
        # Limitation size cache
        if len(self._volume_profile_cache) > 1000:
            # Removal levels with smallest volume
            sorted_levels = sorted(self._volume_profile_cache.items(), key=lambda x: x[1])
            for price, _ in sorted_levels[:200]:  # Remove 200 least active levels
                del self._volume_profile_cache[price]
    
    def get_volume_profile(self, price_range: Tuple[float, float] = None) -> Dict[float, float]:
        """Retrieval volume profile"""
        if price_range is None:
            return self._volume_profile_cache.copy()
        
        min_price, max_price = price_range
        return {price: volume for price, volume in self._volume_profile_cache.items()
                if min_price <= price <= max_price}
    
    def get_market_structure_levels(self) -> Dict[str, List[float]]:
        """Retrieval key levels market structures"""
        if not self._volume_profile_cache:
            return {}
        
        # Sorting by volume
        sorted_levels = sorted(self._volume_profile_cache.items(), key=lambda x: x[1], reverse=True)
        
        # Determination key levels
        total_volume = sum(self._volume_profile_cache.values())
        
        high_volume_levels = []
        support_levels = []
        resistance_levels = []
        
        cumulative_volume = 0
        for price, volume in sorted_levels:
            volume_percentage = volume / total_volume
            cumulative_volume += volume
            
            # Levels with high volume (top 20%)
            if cumulative_volume / total_volume <= 0.2:
                high_volume_levels.append(price)
            
            # Determination support/resistance on basis recent price action
            if self.current_bar:
                current_price = self.current_bar.close_price
                if price < current_price and volume_percentage > 0.05:
                    support_levels.append(price)
                elif price > current_price and volume_percentage > 0.05:
                    resistance_levels.append(price)
        
        return {
            'high_volume_nodes': sorted(high_volume_levels),
            'support_levels': sorted(support_levels, reverse=True)[:5],  # Nearest 5 levels support
            'resistance_levels': sorted(resistance_levels)[:5]  # Nearest 5 levels resistance
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieval statistics analyzer"""
        if not self.footprint_bars:
            return {}
        
        recent_bars = list(self.footprint_bars)[-20:] if len(self.footprint_bars) >= 20 else list(self.footprint_bars)
        
        total_levels = sum(len(bar.price_levels) for bar in recent_bars)
        total_volume = sum(bar.volume for bar in recent_bars)
        
        return {
            'total_bars': len(self.footprint_bars),
            'avg_levels_per_bar': total_levels / len(recent_bars) if recent_bars else 0,
            'avg_bar_volume': total_volume / len(recent_bars) if recent_bars else 0,
            'volume_profile_levels': len(self._volume_profile_cache),
            'cached_patterns': len(self._pattern_cache),
            'price_range': {
                'min': min(bar.low_price for bar in recent_bars) if recent_bars else 0,
                'max': max(bar.high_price for bar in recent_bars) if recent_bars else 0
            }
        }

# Utility functions
def calculate_level_significance(level: PriceLevel, bar_volume: float) -> float:
    """Calculation significance price level"""
    if bar_volume == 0:
        return 0.0
    
    volume_ratio = level.total_volume / bar_volume
    imbalance_strength = abs(level.delta_percentage - 0.5) * 2  # 0 to 1
    trade_density = min(level.trade_count / 20, 1.0)  # Normalization to 20 trades
    
    significance = (volume_ratio * 0.5 + imbalance_strength * 0.3 + trade_density * 0.2)
    
    return min(significance, 1.0)

def merge_nearby_levels(levels: Dict[float, PriceLevel], merge_distance: float) -> Dict[float, PriceLevel]:
    """Merging close price levels"""
    if not levels:
        return levels
    
    sorted_prices = sorted(levels.keys())
    merged_levels = {}
    
    i = 0
    while i < len(sorted_prices):
        current_price = sorted_prices[i]
        current_level = levels[current_price]
        
        # Search close levels for merging
        merge_group = [current_level]
        j = i + 1
        
        while j < len(sorted_prices) and (sorted_prices[j] - current_price) <= merge_distance:
            merge_group.append(levels[sorted_prices[j]])
            j += 1
        
        # Creation unified level
        if len(merge_group) > 1:
            # Weighted average price
            total_volume = sum(level.total_volume for level in merge_group)
            if total_volume > 0:
                weighted_price = sum(level.price * level.total_volume for level in merge_group) / total_volume
            else:
                weighted_price = current_price
            
            merged_level = PriceLevel(price=weighted_price)
            merged_level.buy_volume = sum(level.buy_volume for level in merge_group)
            merged_level.sell_volume = sum(level.sell_volume for level in merge_group)
            merged_level.total_volume = sum(level.total_volume for level in merge_group)
            merged_level.trade_count = sum(level.trade_count for level in merge_group)
            merged_level.avg_trade_size = merged_level.total_volume / merged_level.trade_count if merged_level.trade_count > 0 else 0
            
            merged_levels[weighted_price] = merged_level
        else:
            merged_levels[current_price] = current_level
        
        i = j
    
    return merged_levels

# Export main components
__all__ = [
    'FootprintAnalyzer',
    'FootprintBar',
    'PriceLevel',
    'FootprintPattern',
    'FootprintType',
    'LevelImbalance',
    'calculate_level_significance',
    'merge_nearby_levels'
]