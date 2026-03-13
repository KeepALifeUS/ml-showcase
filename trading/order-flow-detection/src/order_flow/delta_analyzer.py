"""
Delta Analyzer - Analysis buying/selling delta
Uses enterprise patterns for high-frequency analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from decimal import Decimal
import time
from collections import deque

from ..utils.config import get_settings
from ..utils.logger import get_logger, LoggerType, performance_monitor

class DeltaType(str, Enum):
    """Types delta"""
    BUY_DELTA = "buy"
    SELL_DELTA = "sell"
    NET_DELTA = "net"
    CUMULATIVE_DELTA = "cumulative"

class DeltaSignal(str, Enum):
    """Signals delta"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    EXHAUSTION = "exhaustion"
    MOMENTUM = "momentum"

@dataclass
class DeltaMetrics:
    """Metrics delta"""
    buy_volume: float
    sell_volume: float
    net_delta: float
    cumulative_delta: float
    delta_ratio: float
    momentum: float
    strength: float
    timestamp: float

@dataclass
class DeltaPattern:
    """Pattern delta"""
    pattern_type: str
    signal: DeltaSignal
    confidence: float
    duration: int
    volume_profile: Dict[str, float]
    price_levels: List[Tuple[float, float]]  # price, delta
    timestamp: float

class DeltaAnalyzer:
    """
    Analyzer buying/selling delta volume
    High-Performance Computing pattern
    """
    
    def __init__(self, symbol: str, tick_size: float = None):
        self.symbol = symbol
        self.settings = get_settings()
        self.logger = get_logger(LoggerType.ORDER_FLOW, f"delta_analyzer_{symbol}")
        
        # Settings analysis
        self.tick_size = tick_size or self.settings.order_flow.tick_size
        self.analysis_window = 1000  # number ticks for analysis
        self.momentum_window = 50    # window for calculation momentum
        
        # Storage data
        self.trades = deque(maxlen=self.analysis_window)
        self.delta_history = deque(maxlen=self.analysis_window)
        self.cumulative_delta = 0.0
        
        # Cache for optimization
        self._delta_cache: Dict[str, Any] = {}
        self._last_calculation_time = 0
        
        # Threshold values
        self.imbalance_threshold = self.settings.order_flow.imbalance_threshold
        self.exhaustion_threshold = self.settings.order_flow.exhaustion_threshold
        
        self.logger.logger.info(
            "Delta analyzer initialized",
            symbol=symbol,
            tick_size=self.tick_size,
            analysis_window=self.analysis_window
        )
    
    @performance_monitor("delta_analysis")
    async def add_trade(self, price: float, volume: float, is_buy: bool, timestamp: float = None) -> DeltaMetrics:
        """
        Addition trades for analysis delta
        Stream Processing pattern
        """
        timestamp = timestamp or time.time()
        
        # Determination delta
        buy_delta = volume if is_buy else 0
        sell_delta = volume if not is_buy else 0
        net_delta = buy_delta - sell_delta
        
        # Update cumulative delta
        self.cumulative_delta += net_delta
        
        # Addition in history
        trade_data = {
            'price': price,
            'volume': volume,
            'buy_delta': buy_delta,
            'sell_delta': sell_delta,
            'net_delta': net_delta,
            'timestamp': timestamp
        }
        
        self.trades.append(trade_data)
        
        # Calculation metrics
        metrics = await self._calculate_delta_metrics()
        
        self.delta_history.append(metrics)
        
        # Update cache
        self._update_cache(metrics)
        
        return metrics
    
    async def _calculate_delta_metrics(self) -> DeltaMetrics:
        """Calculation metrics delta"""
        if len(self.trades) < 10:
            return DeltaMetrics(0, 0, 0, 0, 0, 0, 0, time.time())
        
        # Retrieval recent data
        recent_trades = list(self.trades)[-self.momentum_window:]
        
        # Summation volumes
        total_buy_volume = sum(t['buy_delta'] for t in recent_trades)
        total_sell_volume = sum(t['sell_delta'] for t in recent_trades)
        total_volume = total_buy_volume + total_sell_volume
        
        # Calculation delta
        net_delta = total_buy_volume - total_sell_volume
        delta_ratio = (total_buy_volume / total_volume) if total_volume > 0 else 0.5
        
        # Calculation momentum
        momentum = await self._calculate_momentum()
        
        # Calculation force signal
        strength = abs(net_delta) / total_volume if total_volume > 0 else 0
        
        return DeltaMetrics(
            buy_volume=total_buy_volume,
            sell_volume=total_sell_volume,
            net_delta=net_delta,
            cumulative_delta=self.cumulative_delta,
            delta_ratio=delta_ratio,
            momentum=momentum,
            strength=strength,
            timestamp=time.time()
        )
    
    async def _calculate_momentum(self) -> float:
        """Calculation momentum delta"""
        if len(self.delta_history) < 2:
            return 0.0
        
        # Retrieval recent values
        current_delta = self.delta_history[-1].net_delta if self.delta_history else 0
        previous_deltas = [d.net_delta for d in list(self.delta_history)[-10:]]
        
        if not previous_deltas:
            return 0.0
        
        # Calculation changes
        avg_previous = np.mean(previous_deltas)
        momentum = (current_delta - avg_previous) / (abs(avg_previous) + 1e-6)
        
        return momentum
    
    @performance_monitor("delta_pattern_detection")
    async def detect_patterns(self, lookback_periods: int = 100) -> List[DeltaPattern]:
        """
        Detection patterns delta
        Pattern Recognition pattern
        """
        if len(self.delta_history) < lookback_periods:
            return []
        
        patterns = []
        recent_metrics = list(self.delta_history)[-lookback_periods:]
        
        # Search various patterns
        patterns.extend(await self._detect_imbalance_patterns(recent_metrics))
        patterns.extend(await self._detect_exhaustion_patterns(recent_metrics))
        patterns.extend(await self._detect_momentum_patterns(recent_metrics))
        patterns.extend(await self._detect_divergence_patterns(recent_metrics))
        
        # Logging found patterns
        for pattern in patterns:
            self.logger.log_pattern_detected(
                pattern_type=pattern.pattern_type,
                symbol=self.symbol,
                confidence=pattern.confidence,
                signal=pattern.signal.value
            )
        
        return patterns
    
    async def _detect_imbalance_patterns(self, metrics: List[DeltaMetrics]) -> List[DeltaPattern]:
        """Detection patterns imbalance"""
        patterns = []
        
        # Analysis recent metrics
        for i in range(len(metrics) - 10, len(metrics)):
            if i < 10:
                continue
                
            current = metrics[i]
            window = metrics[i-10:i+1]
            
            # Validation stable imbalance
            buy_dominance = sum(1 for m in window if m.delta_ratio > self.imbalance_threshold)
            sell_dominance = sum(1 for m in window if m.delta_ratio < (1 - self.imbalance_threshold))
            
            if buy_dominance >= 7:  # 70% periods with predominance purchases
                confidence = min(buy_dominance / 10, 0.9)
                patterns.append(DeltaPattern(
                    pattern_type="buy_imbalance",
                    signal=DeltaSignal.BULLISH,
                    confidence=confidence,
                    duration=10,
                    volume_profile=self._get_volume_profile(window),
                    price_levels=self._get_price_levels(window),
                    timestamp=current.timestamp
                ))
            
            elif sell_dominance >= 7:  # 70% periods with predominance sales
                confidence = min(sell_dominance / 10, 0.9)
                patterns.append(DeltaPattern(
                    pattern_type="sell_imbalance",
                    signal=DeltaSignal.BEARISH,
                    confidence=confidence,
                    duration=10,
                    volume_profile=self._get_volume_profile(window),
                    price_levels=self._get_price_levels(window),
                    timestamp=current.timestamp
                ))
        
        return patterns
    
    async def _detect_exhaustion_patterns(self, metrics: List[DeltaMetrics]) -> List[DeltaPattern]:
        """Detection patterns exhaustion"""
        patterns = []
        
        if len(metrics) < 20:
            return patterns
        
        # Analysis reduction force momentum
        momentum_values = [m.momentum for m in metrics[-20:]]
        strength_values = [m.strength for m in metrics[-20:]]
        
        # Validation on exhaustion bullish momentum
        if (momentum_values[-1] > 0 and 
            np.mean(momentum_values[-5:]) < np.mean(momentum_values[-15:-10]) and
            np.mean(strength_values[-5:]) < self.exhaustion_threshold):
            
            patterns.append(DeltaPattern(
                pattern_type="bullish_exhaustion",
                signal=DeltaSignal.EXHAUSTION,
                confidence=0.7,
                duration=20,
                volume_profile=self._get_volume_profile(metrics[-20:]),
                price_levels=self._get_price_levels(metrics[-20:]),
                timestamp=metrics[-1].timestamp
            ))
        
        # Validation on exhaustion bearish momentum
        if (momentum_values[-1] < 0 and 
            abs(np.mean(momentum_values[-5:])) < abs(np.mean(momentum_values[-15:-10])) and
            np.mean(strength_values[-5:]) < self.exhaustion_threshold):
            
            patterns.append(DeltaPattern(
                pattern_type="bearish_exhaustion",
                signal=DeltaSignal.EXHAUSTION,
                confidence=0.7,
                duration=20,
                volume_profile=self._get_volume_profile(metrics[-20:]),
                price_levels=self._get_price_levels(metrics[-20:]),
                timestamp=metrics[-1].timestamp
            ))
        
        return patterns
    
    async def _detect_momentum_patterns(self, metrics: List[DeltaMetrics]) -> List[DeltaPattern]:
        """Detection momentum patterns"""
        patterns = []
        
        if len(metrics) < 15:
            return patterns
        
        momentum_values = [m.momentum for m in metrics[-15:]]
        
        # Acceleration momentum
        recent_momentum = np.mean(momentum_values[-5:])
        earlier_momentum = np.mean(momentum_values[-15:-10])
        
        momentum_acceleration = recent_momentum - earlier_momentum
        
        if momentum_acceleration > 0.1:  # Significant acceleration
            signal = DeltaSignal.BULLISH if recent_momentum > 0 else DeltaSignal.BEARISH
            patterns.append(DeltaPattern(
                pattern_type="momentum_acceleration",
                signal=signal,
                confidence=min(abs(momentum_acceleration), 0.9),
                duration=15,
                volume_profile=self._get_volume_profile(metrics[-15:]),
                price_levels=self._get_price_levels(metrics[-15:]),
                timestamp=metrics[-1].timestamp
            ))
        
        return patterns
    
    async def _detect_divergence_patterns(self, metrics: List[DeltaMetrics]) -> List[DeltaPattern]:
        """Detection patterns divergences"""
        patterns = []
        
        if len(metrics) < 30 or len(self.trades) < 30:
            return patterns
        
        # Retrieval price data
        recent_trades = list(self.trades)[-30:]
        recent_prices = [t['price'] for t in recent_trades]
        
        # Analysis divergences between price and delta
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        delta_values = [m.cumulative_delta for m in metrics[-30:]]
        delta_trend = np.polyfit(range(len(delta_values)), delta_values, 1)[0]
        
        # Determination divergences
        if price_trend > 0 and delta_trend < -0.1:  # Price grows, delta falls
            patterns.append(DeltaPattern(
                pattern_type="bearish_divergence",
                signal=DeltaSignal.BEARISH,
                confidence=0.6,
                duration=30,
                volume_profile=self._get_volume_profile(metrics[-30:]),
                price_levels=self._get_price_levels(metrics[-30:]),
                timestamp=metrics[-1].timestamp
            ))
        
        elif price_trend < 0 and delta_trend > 0.1:  # Price falls, delta grows
            patterns.append(DeltaPattern(
                pattern_type="bullish_divergence",
                signal=DeltaSignal.BULLISH,
                confidence=0.6,
                duration=30,
                volume_profile=self._get_volume_profile(metrics[-30:]),
                price_levels=self._get_price_levels(metrics[-30:]),
                timestamp=metrics[-1].timestamp
            ))
        
        return patterns
    
    def _get_volume_profile(self, metrics: List[DeltaMetrics]) -> Dict[str, float]:
        """Retrieval profile volume"""
        total_buy = sum(m.buy_volume for m in metrics)
        total_sell = sum(m.sell_volume for m in metrics)
        total_volume = total_buy + total_sell
        
        return {
            'buy_percentage': (total_buy / total_volume) if total_volume > 0 else 0,
            'sell_percentage': (total_sell / total_volume) if total_volume > 0 else 0,
            'total_volume': total_volume,
            'net_delta': total_buy - total_sell
        }
    
    def _get_price_levels(self, metrics: List[DeltaMetrics]) -> List[Tuple[float, float]]:
        """Retrieval levels price with delta"""
        if not self.trades:
            return []
        
        # Grouping by price levels
        price_delta_map = {}
        recent_trades = list(self.trades)[-len(metrics):]
        
        for trade in recent_trades:
            price = round(trade['price'] / self.tick_size) * self.tick_size
            if price not in price_delta_map:
                price_delta_map[price] = 0
            price_delta_map[price] += trade['net_delta']
        
        return [(price, delta) for price, delta in price_delta_map.items()]
    
    def _update_cache(self, metrics: DeltaMetrics) -> None:
        """Update cache for optimization"""
        self._delta_cache.update({
            'last_delta_ratio': metrics.delta_ratio,
            'last_momentum': metrics.momentum,
            'last_strength': metrics.strength,
            'last_update': time.time()
        })
    
    async def get_real_time_signal(self) -> Dict[str, Any]:
        """
        Retrieval real-time signal
        Real-time Processing pattern
        """
        if not self.delta_history:
            return {'signal': DeltaSignal.NEUTRAL, 'confidence': 0.0}
        
        latest_metrics = self.delta_history[-1]
        
        # Determination signal
        signal = DeltaSignal.NEUTRAL
        confidence = 0.0
        
        # Analysis delta
        if latest_metrics.delta_ratio > self.imbalance_threshold:
            signal = DeltaSignal.BULLISH
            confidence = min((latest_metrics.delta_ratio - 0.5) * 2, 0.9)
        elif latest_metrics.delta_ratio < (1 - self.imbalance_threshold):
            signal = DeltaSignal.BEARISH
            confidence = min((0.5 - latest_metrics.delta_ratio) * 2, 0.9)
        
        # Accounting momentum
        if abs(latest_metrics.momentum) > 0.1:
            if latest_metrics.momentum > 0:
                signal = DeltaSignal.MOMENTUM if signal == DeltaSignal.BULLISH else DeltaSignal.BULLISH
            else:
                signal = DeltaSignal.MOMENTUM if signal == DeltaSignal.BEARISH else DeltaSignal.BEARISH
            confidence = min(confidence + abs(latest_metrics.momentum) * 0.5, 0.95)
        
        # Accounting force signal
        confidence *= latest_metrics.strength
        
        return {
            'signal': signal,
            'confidence': confidence,
            'metrics': latest_metrics,
            'timestamp': time.time()
        }
    
    async def reset(self) -> None:
        """Reset state analyzer"""
        self.trades.clear()
        self.delta_history.clear()
        self.cumulative_delta = 0.0
        self._delta_cache.clear()
        self._last_calculation_time = 0
        
        self.logger.logger.info("Delta analyzer reset", symbol=self.symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieval statistics analyzer"""
        if not self.delta_history:
            return {}
        
        metrics_list = list(self.delta_history)
        
        return {
            'total_trades': len(self.trades),
            'analysis_period': len(self.delta_history),
            'cumulative_delta': self.cumulative_delta,
            'avg_delta_ratio': np.mean([m.delta_ratio for m in metrics_list]),
            'avg_momentum': np.mean([m.momentum for m in metrics_list]),
            'avg_strength': np.mean([m.strength for m in metrics_list]),
            'max_imbalance': max([max(m.delta_ratio, 1 - m.delta_ratio) for m in metrics_list]),
            'cache_size': len(self._delta_cache)
        }

# Utility functions for work with delta
def classify_trade_by_delta(price: float, bid: float, ask: float, volume: float) -> Tuple[bool, float]:
    """
    Classification trades as purchase/sale on basis delta
    Classification pattern
    """
    mid_price = (bid + ask) / 2
    
    # Simple classification by price execution
    if price >= mid_price:
        return True, volume  # Purchase
    else:
        return False, volume  # Sale

async def calculate_vpin(trades: List[Dict], window: int = 50) -> float:
    """
    Calculation Volume-synchronized Probability of Informed Trading (VPIN)
    Financial Metrics pattern
    """
    if len(trades) < window:
        return 0.0
    
    recent_trades = trades[-window:]
    
    total_buy_volume = sum(t.get('buy_delta', 0) for t in recent_trades)
    total_sell_volume = sum(t.get('sell_delta', 0) for t in recent_trades)
    total_volume = total_buy_volume + total_sell_volume
    
    if total_volume == 0:
        return 0.0
    
    # VPIN formula
    vpin = abs(total_buy_volume - total_sell_volume) / total_volume
    
    return vpin

# Export main components
__all__ = [
    'DeltaAnalyzer',
    'DeltaMetrics',
    'DeltaPattern',
    'DeltaType',
    'DeltaSignal',
    'classify_trade_by_delta',
    'calculate_vpin'
]