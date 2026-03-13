"""
Spoofing Detection - Detection spoofing (false orders)
enterprise patterns for detection manipulative practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import time
from collections import deque, defaultdict

from ..utils.config import get_settings
from ..utils.logger import get_logger, LoggerType, performance_monitor

class SpoofingType(str, Enum):
    """Types spoofing"""
    LAYERING = "layering"  # Setting multiple false orders
    QUOTE_STUFFING = "quote_stuffing"  # Fast placement and cancellation quotes
    MOMENTUM_IGNITION = "momentum_ignition"  # Creation false impulse
    PING_PONG = "ping_pong"  # Fast placement-cancellation orders
    CROSS_PRODUCT = "cross_product"  # Spoofing between connected tools

class SpoofingPattern(str, Enum):
    """Patterns spoofing"""
    LARGE_ORDER_PLACEMENT = "large_order_placement"
    QUICK_CANCELLATION = "quick_cancellation"
    VOLUME_IMBALANCE_CREATION = "volume_imbalance_creation"
    PRICE_MANIPULATION = "price_manipulation"
    SYSTEMATIC_PATTERN = "systematic_pattern"

@dataclass
class OrderEvent:
    """Event order"""
    timestamp: float
    order_id: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    event_type: str  # 'place', 'modify', 'cancel', 'fill'
    account_id: Optional[str] = None
    
    @property
    def is_aggressive(self) -> bool:
        """Validation aggressiveness order"""
        return self.event_type in ['fill', 'market_order']

@dataclass
class SpoofingSignal:
    """Signal spoofing"""
    spoofing_type: SpoofingType
    pattern: SpoofingPattern
    confidence: float
    severity: float
    start_time: float
    end_time: float
    affected_price_range: Tuple[float, float]
    suspicious_orders: List[OrderEvent]
    market_impact: float
    volume_involved: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration spoofing activity"""
        return self.end_time - self.start_time

@dataclass
class TradingBehaviorProfile:
    """Profile trading behavior"""
    account_id: str
    avg_order_size: float
    avg_hold_time: float
    cancellation_rate: float
    order_frequency: float
    price_levels_used: Set[float]
    typical_patterns: List[str]
    suspicion_score: float = 0.0
    
    def update_suspicion_score(self, new_evidence: float):
        """Update score suspiciousness"""
        self.suspicion_score = min(self.suspicion_score + new_evidence, 1.0)

class SpoofingDetector:
    """
    Detector spoofing activity
    Market Surveillance and Compliance pattern
    """
    
    def __init__(self, symbol: str, detection_window: int = 300):
        self.symbol = symbol
        self.detection_window = detection_window
        self.settings = get_settings()
        self.logger = get_logger(LoggerType.ORDER_FLOW, f"spoofing_detector_{symbol}")
        
        # Threshold values for detection
        self.quick_cancel_threshold = 5.0  # seconds
        self.large_order_multiplier = 5.0  # in how many times more average
        self.cancellation_rate_threshold = 0.8  # 80% cancellations
        self.quote_stuffing_threshold = 10  # orders in second
        self.min_confidence_threshold = 0.6
        
        # Storage data
        self.order_events: deque = deque(maxlen=10000)
        self.active_orders: Dict[str, OrderEvent] = {}
        self.canceled_orders: deque = deque(maxlen=5000)
        self.trading_profiles: Dict[str, TradingBehaviorProfile] = {}
        
        # Statistics for calibration
        self.order_size_stats = {'mean': 0, 'std': 0, 'samples': []}
        self.hold_time_stats = {'mean': 0, 'std': 0, 'samples': []}
        
        # Detected cases spoofing
        self.detected_spoofing: List[SpoofingSignal] = []
        
        self.logger.logger.info(
            "Spoofing detector initialized",
            symbol=symbol,
            detection_window=detection_window
        )
    
    @performance_monitor("spoofing_detection")
    async def process_order_event(self, event: OrderEvent) -> List[SpoofingSignal]:
        """
        Processing events order and detection spoofing
        Event-Driven Processing pattern
        """
        # Addition events in history
        self.order_events.append(event)
        
        # Update active orders
        await self._update_order_tracking(event)
        
        # Update profiles behavior
        if event.account_id:
            await self._update_behavior_profile(event)
        
        # Cleanup old data
        await self._cleanup_old_data(event.timestamp)
        
        # Update statistics
        await self._update_statistics(event)
        
        # Detection various types spoofing
        detected_signals = []
        
        # Detection fast cancellations
        quick_cancel_signals = await self._detect_quick_cancellations(event)
        detected_signals.extend(quick_cancel_signals)
        
        # Detection layering
        layering_signals = await self._detect_layering_patterns(event)
        detected_signals.extend(layering_signals)
        
        # Detection quote stuffing
        stuffing_signals = await self._detect_quote_stuffing(event)
        detected_signals.extend(stuffing_signals)
        
        # Detection momentum ignition
        momentum_signals = await self._detect_momentum_ignition(event)
        detected_signals.extend(momentum_signals)
        
        # Detection ping-pong patterns
        pingpong_signals = await self._detect_ping_pong_patterns(event)
        detected_signals.extend(pingpong_signals)
        
        # Saving detected signals
        self.detected_spoofing.extend(detected_signals)
        
        # Logging suspicious activity
        for signal in detected_signals:
            self.logger.log_market_manipulation(
                symbol=self.symbol,
                manipulation_type=signal.spoofing_type.value,
                evidence=signal.evidence
            )
        
        return detected_signals
    
    async def _update_order_tracking(self, event: OrderEvent):
        """Update tracking orders"""
        if event.event_type == 'place':
            self.active_orders[event.order_id] = event
        elif event.event_type in ['cancel', 'fill']:
            if event.order_id in self.active_orders:
                original_order = self.active_orders[event.order_id]
                
                # Calculation time life order
                hold_time = event.timestamp - original_order.timestamp
                
                if event.event_type == 'cancel':
                    # Addition in list cancelled
                    cancel_info = {
                        'original_order': original_order,
                        'cancel_event': event,
                        'hold_time': hold_time,
                        'timestamp': event.timestamp
                    }
                    self.canceled_orders.append(cancel_info)
                
                del self.active_orders[event.order_id]
    
    async def _update_behavior_profile(self, event: OrderEvent):
        """Update profile behavior participant"""
        account_id = event.account_id
        
        if account_id not in self.trading_profiles:
            self.trading_profiles[account_id] = TradingBehaviorProfile(
                account_id=account_id,
                avg_order_size=0,
                avg_hold_time=0,
                cancellation_rate=0,
                order_frequency=0,
                price_levels_used=set(),
                typical_patterns=[]
            )
        
        profile = self.trading_profiles[account_id]
        
        # Update statistics sizes orders
        if hasattr(profile, '_order_sizes'):
            profile._order_sizes.append(event.size)
        else:
            profile._order_sizes = [event.size]
        profile.avg_order_size = np.mean(profile._order_sizes[-100:])  # Recent 100 orders
        
        # Update used price levels
        profile.price_levels_used.add(event.price)
        
        # Calculation frequencies orders
        recent_events = [e for e in self.order_events if e.account_id == account_id and 
                        e.timestamp > event.timestamp - 3600]  # Last hour
        profile.order_frequency = len(recent_events) / 3600  # orders in second
        
        # Calculation coefficient cancellations
        if hasattr(profile, '_total_orders'):
            profile._total_orders += 1
        else:
            profile._total_orders = 1
            
        if hasattr(profile, '_canceled_orders'):
            if event.event_type == 'cancel':
                profile._canceled_orders += 1
        else:
            profile._canceled_orders = 1 if event.event_type == 'cancel' else 0
            
        profile.cancellation_rate = profile._canceled_orders / profile._total_orders
    
    async def _cleanup_old_data(self, current_time: float):
        """Cleanup obsolete data"""
        cutoff_time = current_time - self.detection_window
        
        # Cleanup old events cancellation
        self.canceled_orders = deque(
            [order for order in self.canceled_orders if order['timestamp'] > cutoff_time],
            maxlen=5000
        )
        
        # Cleanup old detected signals
        self.detected_spoofing = [
            signal for signal in self.detected_spoofing 
            if signal.end_time > cutoff_time
        ]
    
    async def _update_statistics(self, event: OrderEvent):
        """Update total statistics"""
        # Statistics sizes orders
        self.order_size_stats['samples'].append(event.size)
        if len(self.order_size_stats['samples']) > 1000:
            self.order_size_stats['samples'] = self.order_size_stats['samples'][-500:]
        
        self.order_size_stats['mean'] = np.mean(self.order_size_stats['samples'])
        self.order_size_stats['std'] = np.std(self.order_size_stats['samples'])
    
    async def _detect_quick_cancellations(self, event: OrderEvent) -> List[SpoofingSignal]:
        """Detection fast cancellations orders"""
        signals = []
        
        if event.event_type != 'cancel':
            return signals
        
        # Search corresponding placement order
        original_order = None
        for order_event in reversed(self.order_events):
            if (order_event.order_id == event.order_id and 
                order_event.event_type == 'place'):
                original_order = order_event
                break
        
        if not original_order:
            return signals
        
        hold_time = event.timestamp - original_order.timestamp
        
        # Validation fast cancellation
        if hold_time <= self.quick_cancel_threshold:
            # Validation size order
            is_large_order = (original_order.size > 
                            self.order_size_stats['mean'] * self.large_order_multiplier)
            
            # Calculation confidence
            time_factor = 1.0 - (hold_time / self.quick_cancel_threshold)
            size_factor = min(original_order.size / (self.order_size_stats['mean'] + 1e-6), 5) / 5
            
            confidence = time_factor * 0.6 + size_factor * 0.4
            
            if confidence >= self.min_confidence_threshold:
                # Estimation impact on market
                market_impact = self._estimate_market_impact(original_order, event)
                
                signal = SpoofingSignal(
                    spoofing_type=SpoofingType.LAYERING if is_large_order else SpoofingType.PING_PONG,
                    pattern=SpoofingPattern.QUICK_CANCELLATION,
                    confidence=confidence,
                    severity=confidence * (1.0 if is_large_order else 0.7),
                    start_time=original_order.timestamp,
                    end_time=event.timestamp,
                    affected_price_range=(original_order.price, original_order.price),
                    suspicious_orders=[original_order, event],
                    market_impact=market_impact,
                    volume_involved=original_order.size,
                    evidence={
                        'hold_time': hold_time,
                        'order_size': original_order.size,
                        'avg_order_size': self.order_size_stats['mean'],
                        'size_multiplier': original_order.size / (self.order_size_stats['mean'] + 1e-6)
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _detect_layering_patterns(self, event: OrderEvent) -> List[SpoofingSignal]:
        """Detection layering (placement multiple false orders)"""
        signals = []
        
        if event.event_type != 'place':
            return signals
        
        # Analysis recent orders that same participant
        if not event.account_id:
            return signals
        
        recent_window = 60  # Analyze recent 60 seconds
        recent_orders = [
            e for e in self.order_events
            if (e.account_id == event.account_id and
                e.event_type == 'place' and
                e.side == event.side and  # That same side
                e.timestamp > event.timestamp - recent_window)
        ]
        
        if len(recent_orders) < 3:  # Minimum 3 order for layering
            return signals
        
        # Analysis price levels
        prices = [order.price for order in recent_orders]
        price_spread = max(prices) - min(prices)
        
        # Analysis sizes
        sizes = [order.size for order in recent_orders]
        total_size = sum(sizes)
        
        # Validation on multiple levels
        unique_prices = len(set(prices))
        
        if unique_prices >= 3 and total_size > self.order_size_stats['mean'] * 3:
            # Validation fast placement
            time_span = max([order.timestamp for order in recent_orders]) - min([order.timestamp for order in recent_orders])
            
            if time_span < 30:  # Fast placement for 30 seconds
                confidence = min(
                    (unique_prices / 5) * 0.3 +  # Number levels
                    (total_size / (self.order_size_stats['mean'] * 10)) * 0.4 +  # Total size
                    (1.0 - time_span / 30) * 0.3,  # Speed placement
                    1.0
                )
                
                if confidence >= self.min_confidence_threshold:
                    signal = SpoofingSignal(
                        spoofing_type=SpoofingType.LAYERING,
                        pattern=SpoofingPattern.LARGE_ORDER_PLACEMENT,
                        confidence=confidence,
                        severity=confidence * 0.9,  # Layering serious violation
                        start_time=min([order.timestamp for order in recent_orders]),
                        end_time=max([order.timestamp for order in recent_orders]),
                        affected_price_range=(min(prices), max(prices)),
                        suspicious_orders=recent_orders,
                        market_impact=self._estimate_layering_impact(recent_orders),
                        volume_involved=total_size,
                        evidence={
                            'price_levels': unique_prices,
                            'total_volume': total_size,
                            'time_span': time_span,
                            'price_spread': price_spread
                        }
                    )
                    
                    signals.append(signal)
        
        return signals
    
    async def _detect_quote_stuffing(self, event: OrderEvent) -> List[SpoofingSignal]:
        """Detection quote stuffing (redundant quotes)"""
        signals = []
        
        # Counting events for recent seconds
        recent_window = 1.0  # 1 second
        recent_events = [
            e for e in self.order_events
            if e.timestamp > event.timestamp - recent_window
        ]
        
        events_per_second = len(recent_events)
        
        if events_per_second > self.quote_stuffing_threshold:
            # Analysis pattern placement/cancellation
            placements = [e for e in recent_events if e.event_type == 'place']
            cancellations = [e for e in recent_events if e.event_type == 'cancel']
            
            # High coefficient cancellations + high frequency = suspicion on stuffing
            if len(cancellations) > len(placements) * 0.7:
                confidence = min(events_per_second / 50, 1.0)  # Normalization to 50 events/sec
                
                # Analysis impact on spread
                spread_impact = self._analyze_spread_impact(recent_events)
                
                signal = SpoofingSignal(
                    spoofing_type=SpoofingType.QUOTE_STUFFING,
                    pattern=SpoofingPattern.SYSTEMATIC_PATTERN,
                    confidence=confidence,
                    severity=confidence * 0.8,
                    start_time=event.timestamp - recent_window,
                    end_time=event.timestamp,
                    affected_price_range=self._get_price_range(recent_events),
                    suspicious_orders=recent_events,
                    market_impact=spread_impact,
                    volume_involved=sum([e.size for e in placements]),
                    evidence={
                        'events_per_second': events_per_second,
                        'cancellation_ratio': len(cancellations) / max(len(placements), 1),
                        'spread_impact': spread_impact
                    }
                )
                
                signals.append(signal)
        
        return signals
    
    async def _detect_momentum_ignition(self, event: OrderEvent) -> List[SpoofingSignal]:
        """Detection momentum ignition (creation false impulse)"""
        signals = []
        
        # Search pattern: large order -> movement price -> fast cancellation
        if event.event_type != 'place':
            return signals
        
        # Validation size order
        if event.size < self.order_size_stats['mean'] * self.large_order_multiplier:
            return signals
        
        # Analysis recent movements price
        recent_trades = [
            e for e in self.order_events
            if (e.event_type == 'fill' and
                e.timestamp > event.timestamp - 30)  # Recent 30 seconds
        ]
        
        if len(recent_trades) >= 3:
            # Analysis price movements
            trade_prices = [trade.price for trade in recent_trades]
            price_movement = max(trade_prices) - min(trade_prices)
            
            # Determination directions movements
            if len(trade_prices) >= 2:
                price_trend = trade_prices[-1] - trade_prices[0]
                
                # Validation on creation impulse in that same direction
                order_direction = 1 if event.side == 'buy' else -1
                trend_direction = 1 if price_trend > 0 else -1
                
                if order_direction == trend_direction and abs(price_trend) > 0:
                    confidence = min(
                        (event.size / (self.order_size_stats['mean'] * 10)) * 0.5 +
                        (abs(price_trend) / price_movement if price_movement > 0 else 0) * 0.5,
                        0.9
                    )
                    
                    if confidence >= self.min_confidence_threshold:
                        signal = SpoofingSignal(
                            spoofing_type=SpoofingType.MOMENTUM_IGNITION,
                            pattern=SpoofingPattern.PRICE_MANIPULATION,
                            confidence=confidence,
                            severity=confidence * 0.95,  # High severity
                            start_time=event.timestamp,
                            end_time=event.timestamp,  # Will be updated when cancellation
                            affected_price_range=(min(trade_prices), max(trade_prices)),
                            suspicious_orders=[event],
                            market_impact=abs(price_trend),
                            volume_involved=event.size,
                            evidence={
                                'price_movement': price_movement,
                                'price_trend': price_trend,
                                'order_size_ratio': event.size / (self.order_size_stats['mean'] + 1e-6)
                            }
                        )
                        
                        signals.append(signal)
        
        return signals
    
    async def _detect_ping_pong_patterns(self, event: OrderEvent) -> List[SpoofingSignal]:
        """Detection ping-pong patterns"""
        signals = []
        
        if not event.account_id:
            return signals
        
        # Analysis pattern placement-cancellation for one participant
        recent_window = 30  # 30 seconds
        account_events = [
            e for e in self.order_events
            if (e.account_id == event.account_id and
                e.timestamp > event.timestamp - recent_window)
        ]
        
        # Counting loops placement-cancellation
        ping_pong_cycles = 0
        i = 0
        while i < len(account_events) - 1:
            if (account_events[i].event_type == 'place' and
                account_events[i+1].event_type == 'cancel' and
                account_events[i].order_id == account_events[i+1].order_id):
                ping_pong_cycles += 1
                i += 2
            else:
                i += 1
        
        if ping_pong_cycles >= 3:  # Minimum 3 loop
            # Analysis temporal intervals
            intervals = []
            for i in range(0, len(account_events) - 1, 2):
                if (i + 1 < len(account_events) and
                    account_events[i].event_type == 'place' and
                    account_events[i+1].event_type == 'cancel'):
                    intervals.append(account_events[i+1].timestamp - account_events[i].timestamp)
            
            if intervals:
                avg_interval = np.mean(intervals)
                interval_consistency = 1.0 - (np.std(intervals) / (avg_interval + 1e-6))
                
                # Fast and regular loops suspicious
                if avg_interval < 10 and interval_consistency > 0.5:
                    confidence = min(
                        (ping_pong_cycles / 10) * 0.4 +
                        (1.0 - avg_interval / 10) * 0.3 +
                        interval_consistency * 0.3,
                        0.9
                    )
                    
                    if confidence >= self.min_confidence_threshold:
                        signal = SpoofingSignal(
                            spoofing_type=SpoofingType.PING_PONG,
                            pattern=SpoofingPattern.SYSTEMATIC_PATTERN,
                            confidence=confidence,
                            severity=confidence * 0.7,
                            start_time=account_events[0].timestamp,
                            end_time=account_events[-1].timestamp,
                            affected_price_range=self._get_price_range(account_events),
                            suspicious_orders=account_events,
                            market_impact=self._estimate_ping_pong_impact(account_events),
                            volume_involved=sum([e.size for e in account_events if e.event_type == 'place']),
                            evidence={
                                'ping_pong_cycles': ping_pong_cycles,
                                'avg_interval': avg_interval,
                                'interval_consistency': interval_consistency
                            }
                        )
                        
                        signals.append(signal)
        
        return signals
    
    def _estimate_market_impact(self, order: OrderEvent, cancel: OrderEvent) -> float:
        """Estimation impact order on market"""
        # Simple estimation on basis size and time life
        size_impact = order.size / (self.order_size_stats['mean'] + 1e-6)
        time_impact = 1.0 - min((cancel.timestamp - order.timestamp) / 60, 1.0)
        
        return min(size_impact * time_impact * 0.1, 1.0)
    
    def _estimate_layering_impact(self, orders: List[OrderEvent]) -> float:
        """Estimation impact layering on market"""
        total_size = sum([order.size for order in orders])
        price_range = max([order.price for order in orders]) - min([order.price for order in orders])
        
        size_impact = total_size / (self.order_size_stats['mean'] * 10)
        spread_impact = price_range / min([order.price for order in orders])  # Relative spread
        
        return min(size_impact + spread_impact, 1.0)
    
    def _analyze_spread_impact(self, events: List[OrderEvent]) -> float:
        """Analysis impact on spread"""
        # Simplified analysis - in real system needed data about spread
        placements = [e for e in events if e.event_type == 'place']
        if len(placements) < 2:
            return 0.0
        
        prices = [e.price for e in placements]
        price_volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        return min(price_volatility * 10, 1.0)
    
    def _estimate_ping_pong_impact(self, events: List[OrderEvent]) -> float:
        """Estimation impact ping-pong activity"""
        frequency = len(events) / 30  # events for 30 seconds
        avg_size = np.mean([e.size for e in events if e.event_type == 'place'])
        
        frequency_impact = min(frequency / 10, 1.0)
        size_impact = avg_size / (self.order_size_stats['mean'] + 1e-6)
        
        return min((frequency_impact + size_impact) * 0.5, 1.0)
    
    def _get_price_range(self, events: List[OrderEvent]) -> Tuple[float, float]:
        """Retrieval price range events"""
        prices = [e.price for e in events]
        return (min(prices), max(prices)) if prices else (0.0, 0.0)
    
    def get_suspicious_accounts(self, min_suspicion_score: float = 0.5) -> List[TradingBehaviorProfile]:
        """Retrieval suspicious accounts"""
        return [
            profile for profile in self.trading_profiles.values()
            if profile.suspicion_score >= min_suspicion_score
        ]
    
    def get_recent_spoofing_activity(self, lookback_seconds: int = 3600) -> List[SpoofingSignal]:
        """Retrieval recent spoofing activity"""
        current_time = time.time()
        cutoff_time = current_time - lookback_seconds
        
        return [
            signal for signal in self.detected_spoofing
            if signal.end_time > cutoff_time
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieval statistics detector"""
        recent_activity = self.get_recent_spoofing_activity(3600)
        
        spoofing_by_type = defaultdict(int)
        for signal in recent_activity:
            spoofing_by_type[signal.spoofing_type.value] += 1
        
        return {
            'total_order_events': len(self.order_events),
            'active_orders': len(self.active_orders),
            'canceled_orders': len(self.canceled_orders),
            'tracked_accounts': len(self.trading_profiles),
            'suspicious_accounts': len(self.get_suspicious_accounts()),
            'detected_spoofing_1h': len(recent_activity),
            'spoofing_by_type': dict(spoofing_by_type),
            'avg_order_size': self.order_size_stats['mean'],
            'detection_settings': {
                'quick_cancel_threshold': self.quick_cancel_threshold,
                'large_order_multiplier': self.large_order_multiplier,
                'quote_stuffing_threshold': self.quote_stuffing_threshold
            }
        }
    
    async def generate_compliance_report(self) -> Dict[str, Any]:
        """Generation report for compliance"""
        recent_signals = self.get_recent_spoofing_activity(86400)  # 24 hours
        
        high_severity_cases = [s for s in recent_signals if s.severity > 0.8]
        
        report = {
            'report_timestamp': time.time(),
            'symbol': self.symbol,
            'reporting_period': '24 hours',
            'total_alerts': len(recent_signals),
            'high_severity_cases': len(high_severity_cases),
            'cases_by_type': {},
            'suspicious_accounts': [],
            'recommendations': []
        }
        
        # Grouping by types
        for signal in recent_signals:
            signal_type = signal.spoofing_type.value
            if signal_type not in report['cases_by_type']:
                report['cases_by_type'][signal_type] = []
            
            report['cases_by_type'][signal_type].append({
                'timestamp': signal.start_time,
                'confidence': signal.confidence,
                'severity': signal.severity,
                'market_impact': signal.market_impact,
                'volume_involved': signal.volume_involved
            })
        
        # Suspicious accounts
        suspicious = self.get_suspicious_accounts(0.7)
        for profile in suspicious:
            report['suspicious_accounts'].append({
                'account_id': profile.account_id,
                'suspicion_score': profile.suspicion_score,
                'cancellation_rate': profile.cancellation_rate,
                'order_frequency': profile.order_frequency
            })
        
        # Recommendations
        if high_severity_cases:
            report['recommendations'].append("Immediate investigation required for high severity cases")
        if len(suspicious) > 0:
            report['recommendations'].append(f"Review trading patterns of {len(suspicious)} suspicious accounts")
        
        return report

# Utility functions
def calculate_manipulation_severity(signal: SpoofingSignal) -> str:
    """Calculation severity manipulation"""
    if signal.severity >= 0.9:
        return "CRITICAL"
    elif signal.severity >= 0.7:
        return "HIGH"
    elif signal.severity >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def analyze_spoofing_trends(signals: List[SpoofingSignal], period_hours: int = 24) -> Dict[str, Any]:
    """Analysis trends spoofing for period"""
    current_time = time.time()
    cutoff_time = current_time - (period_hours * 3600)
    
    recent_signals = [s for s in signals if s.start_time > cutoff_time]
    
    if not recent_signals:
        return {}
    
    # Grouping by hours
    hourly_counts = defaultdict(int)
    for signal in recent_signals:
        hour = int((signal.start_time - cutoff_time) // 3600)
        hourly_counts[hour] += 1
    
    # Analysis by types
    type_distribution = defaultdict(int)
    for signal in recent_signals:
        type_distribution[signal.spoofing_type.value] += 1
    
    return {
        'total_cases': len(recent_signals),
        'avg_cases_per_hour': len(recent_signals) / period_hours,
        'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else 0,
        'type_distribution': dict(type_distribution),
        'avg_confidence': np.mean([s.confidence for s in recent_signals]),
        'avg_severity': np.mean([s.severity for s in recent_signals])
    }

# Export main components
__all__ = [
    'SpoofingDetector',
    'SpoofingSignal',
    'OrderEvent',
    'TradingBehaviorProfile',
    'SpoofingType',
    'SpoofingPattern',
    'calculate_manipulation_severity',
    'analyze_spoofing_trends'
]