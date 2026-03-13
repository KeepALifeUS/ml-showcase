"""
Harmonic Pattern Monitor - Real-time monitoring and alerting
Real-time pattern monitoring with alerts

WebSocket streaming, real-time alerts, and performance monitoring
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import json

import pandas as pd
import numpy as np

from .detector import (
    HarmonicPatternDetector,
    DetectedPattern,
    PatternType,
    PatternDirection,
    DetectionMode
)


class AlertType(Enum):
    """Types of pattern alerts"""
    PATTERN_FORMING = "pattern_forming"        # Pattern is forming (80% complete)
    PATTERN_COMPLETE = "pattern_complete"      # Pattern fully formed
    ENTRY_SIGNAL = "entry_signal"             # Entry point reached
    TARGET_HIT = "target_hit"                 # Take profit reached
    STOP_TRIGGERED = "stop_triggered"         # Stop loss triggered
    PATTERN_INVALIDATED = "pattern_invalidated"  # Pattern no longer valid


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PatternAlert:
    """Alert for harmonic pattern events"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    pattern: DetectedPattern
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'priority': self.priority.value,
            'pattern': self.pattern.to_dict(),
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'acknowledged': self.acknowledged
        }


@dataclass
class MonitoringConfig:
    """Configuration for pattern monitoring"""
    symbols: List[str]
    timeframes: List[str]
    pattern_types: List[PatternType]
    min_confidence: float = 0.7
    alert_on_forming: bool = True
    alert_on_complete: bool = True
    alert_on_entry: bool = True
    scan_interval_seconds: int = 60
    max_pattern_age_hours: int = 24


class HarmonicPatternMonitor:
    """
    Real-time harmonic pattern monitoring system.

    Continuously scans markets for patterns and generates alerts.
    """

    def __init__(
        self,
        config: MonitoringConfig,
        data_provider: Optional[Callable] = None
    ):
        self.config = config
        self.data_provider = data_provider

        # Pattern detectors for each symbol/timeframe combination
        self.detectors: Dict[str, HarmonicPatternDetector] = {}

        # Active patterns being monitored
        self.active_patterns: Dict[str, DetectedPattern] = {}

        # Alert management
        self.alerts: List[PatternAlert] = []
        self.alert_callbacks: List[Callable[[PatternAlert], None]] = []

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.scan_history: List[Dict] = []
        self.alert_history: List[PatternAlert] = []

        self.logger = logging.getLogger("HarmonicPatternMonitor")

        # Initialize detectors
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize pattern detectors for all symbol/timeframe combinations"""
        for symbol in self.config.symbols:
            for timeframe in self.config.timeframes:
                key = f"{symbol}_{timeframe}"
                self.detectors[key] = HarmonicPatternDetector(
                    symbol=symbol,
                    timeframe=timeframe,
                    pattern_types=self.config.pattern_types,
                    min_confidence=self.config.min_confidence
                )

        self.logger.info(f"Initialized {len(self.detectors)} pattern detectors")

    async def start_monitoring(self):
        """Start real-time pattern monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Pattern monitoring started")

    async def stop_monitoring(self):
        """Stop pattern monitoring"""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Pattern monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                scan_start = datetime.now()

                # Scan all configured markets
                await self._scan_markets()

                # Update active patterns
                await self._update_active_patterns()

                # Clean up old patterns
                self._cleanup_old_patterns()

                # Log scan performance
                scan_time = (datetime.now() - scan_start).total_seconds()
                self.scan_history.append({
                    'timestamp': scan_start,
                    'duration_seconds': scan_time,
                    'patterns_found': len(self.active_patterns),
                    'alerts_generated': len([a for a in self.alerts if a.timestamp >= scan_start])
                })

                # Wait for next scan
                await asyncio.sleep(self.config.scan_interval_seconds)

            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.scan_interval_seconds)

    async def _scan_markets(self):
        """Scan all configured markets for patterns"""
        scan_tasks = []

        for key, detector in self.detectors.items():
            symbol, timeframe = key.split('_')
            task = self._scan_single_market(symbol, timeframe, detector)
            scan_tasks.append(task)

        # Execute scans in parallel
        if scan_tasks:
            results = await asyncio.gather(*scan_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Market scan failed: {result}")

    async def _scan_single_market(
        self,
        symbol: str,
        timeframe: str,
        detector: HarmonicPatternDetector
    ):
        """Scan a single market for patterns"""
        try:
            # Get market data
            if self.data_provider:
                data = await self.data_provider(symbol, timeframe)
            else:
                # Placeholder - in production, connect to real data source
                data = self._generate_sample_data()

            # Detect patterns
            result = await detector.detect_patterns(
                data,
                mode=DetectionMode.REALTIME
            )

            # Process detected patterns
            for pattern in result.patterns:
                pattern_id = self._generate_pattern_id(pattern)

                # Check if this is a new pattern
                if pattern_id not in self.active_patterns:
                    self.active_patterns[pattern_id] = pattern

                    # Generate alerts for new patterns
                    await self._generate_pattern_alerts(pattern)

                    self.logger.info(
                        f"New {pattern.pattern_type.value} pattern detected on {symbol} {timeframe}"
                    )

        except Exception as e:
            self.logger.error(f"Failed to scan {symbol} {timeframe}: {e}")

    async def _update_active_patterns(self):
        """Update status of active patterns"""
        current_prices = await self._get_current_prices()

        for pattern_id, pattern in list(self.active_patterns.items()):
            try:
                # Get current price for pattern's symbol
                if pattern.symbol in current_prices:
                    current_price = current_prices[pattern.symbol]

                    # Check if pattern is still valid
                    if not self._is_pattern_valid(pattern, current_price):
                        await self._invalidate_pattern(pattern)
                        del self.active_patterns[pattern_id]
                        continue

                    # Check for entry signals
                    if self._check_entry_signal(pattern, current_price):
                        await self._generate_entry_alert(pattern, current_price)

                    # Check for target/stop hits
                    await self._check_exit_signals(pattern, current_price)

            except Exception as e:
                self.logger.error(f"Failed to update pattern {pattern_id}: {e}")

    def _is_pattern_valid(self, pattern: DetectedPattern, current_price: float) -> bool:
        """Check if pattern is still valid"""
        # Pattern is invalid if:
        # 1. Too old
        if pattern.pattern_age_hours > self.config.max_pattern_age_hours:
            return False

        # 2. Price has moved too far from PRZ
        prz_min, prz_max = pattern.prz_zone
        prz_range = prz_max - prz_min
        if abs(current_price - pattern.completion_level) > prz_range * 3:
            return False

        # 3. Stop loss has been hit
        if pattern.stop_loss:
            if pattern.direction == PatternDirection.BULLISH and current_price < pattern.stop_loss:
                return False
            elif pattern.direction == PatternDirection.BEARISH and current_price > pattern.stop_loss:
                return False

        return True

    def _check_entry_signal(self, pattern: DetectedPattern, current_price: float) -> bool:
        """Check if price has reached entry zone"""
        if not pattern.entry_price:
            return False

        # Check if price is within entry zone (2% tolerance)
        tolerance = pattern.entry_price * 0.02
        return abs(current_price - pattern.entry_price) <= tolerance

    async def _check_exit_signals(self, pattern: DetectedPattern, current_price: float):
        """Check for target or stop loss hits"""
        # Check stop loss
        if pattern.stop_loss:
            if pattern.direction == PatternDirection.BULLISH and current_price <= pattern.stop_loss:
                await self._generate_stop_alert(pattern, current_price)
            elif pattern.direction == PatternDirection.BEARISH and current_price >= pattern.stop_loss:
                await self._generate_stop_alert(pattern, current_price)

        # Check take profit levels
        for i, target in enumerate(pattern.take_profits):
            if pattern.direction == PatternDirection.BULLISH and current_price >= target:
                await self._generate_target_alert(pattern, i + 1, current_price)
            elif pattern.direction == PatternDirection.BEARISH and current_price <= target:
                await self._generate_target_alert(pattern, i + 1, current_price)

    async def _generate_pattern_alerts(self, pattern: DetectedPattern):
        """Generate alerts for newly detected pattern"""
        # Alert for pattern completion
        if self.config.alert_on_complete and pattern.is_complete:
            alert = PatternAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.PATTERN_COMPLETE,
                priority=self._determine_alert_priority(pattern),
                pattern=pattern,
                message=f"✅ {pattern.pattern_type.value.upper()} pattern completed on {pattern.symbol} {pattern.timeframe}",
                timestamp=datetime.now(),
                metadata={
                    'confidence': pattern.confidence_score,
                    'prz_zone': pattern.prz_zone,
                    'risk_reward': pattern.risk_reward_ratio
                }
            )
            await self._send_alert(alert)

        # Alert for pattern forming (if enabled)
        elif self.config.alert_on_forming and not pattern.is_complete:
            completion_percentage = self._calculate_pattern_completion(pattern)
            if completion_percentage >= 0.8:  # 80% complete
                alert = PatternAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.PATTERN_FORMING,
                    priority=AlertPriority.MEDIUM,
                    pattern=pattern,
                    message=f"⏳ {pattern.pattern_type.value.upper()} pattern {int(completion_percentage * 100)}% complete on {pattern.symbol}",
                    timestamp=datetime.now(),
                    metadata={'completion_percentage': completion_percentage}
                )
                await self._send_alert(alert)

    async def _generate_entry_alert(self, pattern: DetectedPattern, current_price: float):
        """Generate alert for entry signal"""
        if not self.config.alert_on_entry:
            return

        alert = PatternAlert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.ENTRY_SIGNAL,
            priority=AlertPriority.HIGH,
            pattern=pattern,
            message=f"🎯 Entry signal for {pattern.pattern_type.value} on {pattern.symbol} at {current_price:.4f}",
            timestamp=datetime.now(),
            metadata={
                'entry_price': pattern.entry_price,
                'current_price': current_price,
                'stop_loss': pattern.stop_loss,
                'targets': pattern.take_profits
            }
        )
        await self._send_alert(alert)

    async def _generate_target_alert(self, pattern: DetectedPattern, target_num: int, current_price: float):
        """Generate alert for target hit"""
        alert = PatternAlert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.TARGET_HIT,
            priority=AlertPriority.MEDIUM,
            pattern=pattern,
            message=f"🎉 Target {target_num} hit for {pattern.pattern_type.value} on {pattern.symbol} at {current_price:.4f}",
            timestamp=datetime.now(),
            metadata={
                'target_number': target_num,
                'target_price': pattern.take_profits[target_num - 1] if target_num <= len(pattern.take_profits) else None,
                'current_price': current_price
            }
        )
        await self._send_alert(alert)

    async def _generate_stop_alert(self, pattern: DetectedPattern, current_price: float):
        """Generate alert for stop loss hit"""
        alert = PatternAlert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.STOP_TRIGGERED,
            priority=AlertPriority.HIGH,
            pattern=pattern,
            message=f"⛔ Stop loss triggered for {pattern.pattern_type.value} on {pattern.symbol} at {current_price:.4f}",
            timestamp=datetime.now(),
            metadata={
                'stop_loss': pattern.stop_loss,
                'current_price': current_price
            }
        )
        await self._send_alert(alert)

    async def _invalidate_pattern(self, pattern: DetectedPattern):
        """Generate alert for pattern invalidation"""
        alert = PatternAlert(
            alert_id=self._generate_alert_id(),
            alert_type=AlertType.PATTERN_INVALIDATED,
            priority=AlertPriority.LOW,
            pattern=pattern,
            message=f"❌ {pattern.pattern_type.value} pattern invalidated on {pattern.symbol}",
            timestamp=datetime.now(),
            metadata={'reason': 'Pattern expired or price moved out of range'}
        )
        await self._send_alert(alert)

    async def _send_alert(self, alert: PatternAlert):
        """Send alert to all registered callbacks"""
        self.alerts.append(alert)
        self.alert_history.append(alert)

        # Limit alert history size
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-500:]

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await asyncio.create_task(asyncio.coroutine(callback)(alert))
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable[[PatternAlert], None]):
        """Register alert callback"""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable[[PatternAlert], None]):
        """Remove alert callback"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def _cleanup_old_patterns(self):
        """Remove old patterns from active monitoring"""
        cutoff_time = datetime.now() - timedelta(hours=self.config.max_pattern_age_hours)

        patterns_to_remove = []
        for pattern_id, pattern in self.active_patterns.items():
            if pattern.timestamp < cutoff_time:
                patterns_to_remove.append(pattern_id)

        for pattern_id in patterns_to_remove:
            del self.active_patterns[pattern_id]

        if patterns_to_remove:
            self.logger.info(f"Removed {len(patterns_to_remove)} expired patterns")

    def _generate_pattern_id(self, pattern: DetectedPattern) -> str:
        """Generate unique pattern ID"""
        return f"{pattern.symbol}_{pattern.timeframe}_{pattern.pattern_type.value}_{pattern.timestamp.timestamp()}"

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return str(uuid.uuid4())

    def _determine_alert_priority(self, pattern: DetectedPattern) -> AlertPriority:
        """Determine alert priority based on pattern confidence and RR ratio"""
        if pattern.confidence_score >= 0.9 and pattern.risk_reward_ratio and pattern.risk_reward_ratio >= 2:
            return AlertPriority.CRITICAL
        elif pattern.confidence_score >= 0.8:
            return AlertPriority.HIGH
        elif pattern.confidence_score >= 0.7:
            return AlertPriority.MEDIUM
        else:
            return AlertPriority.LOW

    def _calculate_pattern_completion(self, pattern: DetectedPattern) -> float:
        """Calculate pattern completion percentage"""
        required_points = ['X', 'A', 'B', 'C', 'D']
        completed_points = sum(1 for point in required_points if point in pattern.points and pattern.points[point])
        return completed_points / len(required_points)

    async def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all monitored symbols"""
        prices = {}

        for symbol in self.config.symbols:
            # In production, get real prices from data provider
            # For now, return placeholder
            prices[symbol] = np.random.uniform(40000, 50000)  # Placeholder

        return prices

    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample OHLCV data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=500, freq='1h')

        # Generate realistic price movement
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        prices = 45000 * np.exp(np.cumsum(returns))

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.randn(500) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(500) * 0.003)),
            'low': prices * (1 - np.abs(np.random.randn(500) * 0.003)),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 500)
        })

        return data

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        recent_scans = self.scan_history[-100:] if self.scan_history else []

        return {
            'is_monitoring': self.is_monitoring,
            'symbols_monitored': len(self.config.symbols),
            'timeframes_monitored': len(self.config.timeframes),
            'active_patterns': len(self.active_patterns),
            'total_alerts': len(self.alert_history),
            'recent_alerts': len([a for a in self.alert_history if
                                a.timestamp > datetime.now() - timedelta(hours=1)]),
            'average_scan_time': np.mean([s['duration_seconds'] for s in recent_scans]) if recent_scans else 0,
            'patterns_by_type': self._count_patterns_by_type(),
            'alerts_by_type': self._count_alerts_by_type()
        }

    def _count_patterns_by_type(self) -> Dict[str, int]:
        """Count active patterns by type"""
        counts = {}
        for pattern in self.active_patterns.values():
            pt = pattern.pattern_type.value
            counts[pt] = counts.get(pt, 0) + 1
        return counts

    def _count_alerts_by_type(self) -> Dict[str, int]:
        """Count alerts by type"""
        counts = {}
        for alert in self.alert_history:
            at = alert.alert_type.value
            counts[at] = counts.get(at, 0) + 1
        return counts

    def cleanup(self):
        """Clean up resources"""
        if self.is_monitoring:
            asyncio.create_task(self.stop_monitoring())

        self.active_patterns.clear()
        self.alerts.clear()
        self.alert_callbacks.clear()

    def __del__(self):
        """Destructor"""
        self.cleanup()