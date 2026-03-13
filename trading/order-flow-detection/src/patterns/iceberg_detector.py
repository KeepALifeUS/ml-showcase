"""
Iceberg Order Detection - Detection iceberg orders
enterprise patterns for detection hidden large orders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from decimal import Decimal
import time
from collections import deque, defaultdict
import statistics

from ..utils.config import get_settings
from ..utils.logger import get_logger, LoggerType, performance_monitor

class IcebergType(str, Enum):
    """Types iceberg orders"""
    BUY_ICEBERG = "buy_iceberg"
    SELL_ICEBERG = "sell_iceberg"
    HIDDEN_LIQUIDITY = "hidden_liquidity"
    DARK_POOL = "dark_pool"

class IcebergStrength(str, Enum):
    """Force iceberg signal"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

@dataclass
class OrderBookLevel:
    """Level order book orders"""
    price: float
    bid_size: float
    ask_size: float
    bid_orders: int
    ask_orders: int
    timestamp: float
    
    @property
    def spread(self) -> float:
        """Spread on level"""
        return abs(self.ask_size - self.bid_size)
    
    @property
    def imbalance(self) -> float:
        """Imbalance level"""
        total = self.bid_size + self.ask_size
        return (self.bid_size - self.ask_size) / total if total > 0 else 0

@dataclass
class IcebergSignature:
    """Signature iceberg order"""
    price_level: float
    side: str  # 'bid' or 'ask'
    total_estimated_size: float
    visible_size: float
    hidden_size_estimate: float
    refresh_count: int
    avg_refresh_interval: float
    detection_confidence: float
    strength: IcebergStrength
    first_detected: float
    last_activity: float
    
    @property
    def iceberg_ratio(self) -> float:
        """Coefficient hidden volume"""
        return self.hidden_size_estimate / self.total_estimated_size if self.total_estimated_size > 0 else 0
    
    @property
    def activity_duration(self) -> float:
        """Duration activity"""
        return self.last_activity - self.first_detected

@dataclass
class IcebergDetectionResult:
    """Result detection iceberg order"""
    iceberg_type: IcebergType
    signature: IcebergSignature
    market_impact: float
    price_influence: float
    volume_absorption: float
    liquidity_provision: float
    detection_algorithm: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class IcebergDetector:
    """
    Detector iceberg orders
    Advanced Market Microstructure Analysis pattern
    """
    
    def __init__(self, symbol: str, detection_window: int = 300):
        self.symbol = symbol
        self.detection_window = detection_window  # seconds
        self.settings = get_settings()
        self.logger = get_logger(LoggerType.ORDER_FLOW, f"iceberg_detector_{symbol}")
        
        # Settings detector
        self.min_refresh_count = 3  # Minimum updates for consider iceberg
        self.size_threshold_multiplier = 2.0  # In how many times size must exceed average
        self.refresh_interval_threshold = 10.0  # Maximum interval between updates (sec)
        self.confidence_threshold = 0.7  # Minimum confidence for detection
        
        # Storage data
        self.order_book_history: deque = deque(maxlen=1000)
        self.level_refreshes: Dict[float, List[Dict]] = defaultdict(list)
        self.detected_icebergs: Dict[float, IcebergSignature] = {}
        
        # Statistics for calibration
        self.size_statistics = {'mean': 0, 'std': 0, 'samples': []}
        self.refresh_statistics = {'intervals': [], 'counts': []}
        
        self.logger.logger.info(
            "Iceberg detector initialized",
            symbol=symbol,
            detection_window=detection_window
        )
    
    @performance_monitor("iceberg_detection")
    async def analyze_order_book(self, order_book: Dict[str, List[Tuple[float, float]]], 
                                timestamp: float = None) -> List[IcebergDetectionResult]:
        """
        Analysis order book orders on presence icebergs
        Real-time Market Analysis pattern
        """
        timestamp = timestamp or time.time()
        
        # Update history order book
        await self._update_order_book_history(order_book, timestamp)
        
        # Cleanup old data
        await self._cleanup_old_data(timestamp)
        
        # Update statistics
        await self._update_statistics(order_book)
        
        # Detection icebergs various methods
        results = []
        
        # Method 1: Analysis frequencies updates
        refresh_icebergs = await self._detect_by_refresh_pattern(timestamp)
        results.extend(refresh_icebergs)
        
        # Method 2: Analysis anomalous sizes
        size_icebergs = await self._detect_by_size_anomaly(order_book, timestamp)
        results.extend(size_icebergs)
        
        # Method 3: Analysis absorption volume
        absorption_icebergs = await self._detect_by_volume_absorption(timestamp)
        results.extend(absorption_icebergs)
        
        # Method 4: Analysis stability levels
        stability_icebergs = await self._detect_by_level_stability(timestamp)
        results.extend(stability_icebergs)
        
        # Update active icebergs
        await self._update_active_icebergs(results, timestamp)
        
        # Logging results
        if results:
            self.logger.logger.info(
                f"Detected {len(results)} iceberg patterns",
                symbol=self.symbol,
                patterns=[r.iceberg_type.value for r in results]
            )
        
        return results
    
    async def _update_order_book_history(self, order_book: Dict[str, List[Tuple[float, float]]], timestamp: float):
        """Update history order book"""
        # Transformation in structured format
        structured_book = {
            'timestamp': timestamp,
            'bids': {price: size for price, size in order_book.get('bids', [])},
            'asks': {price: size for price, size in order_book.get('asks', [])}
        }
        
        self.order_book_history.append(structured_book)
        
        # Tracking changes levels
        if len(self.order_book_history) >= 2:
            await self._track_level_changes(self.order_book_history[-2], structured_book, timestamp)
    
    async def _track_level_changes(self, prev_book: Dict, curr_book: Dict, timestamp: float):
        """Tracking changes levels order book"""
        # Analysis changes bid levels
        for price, size in curr_book['bids'].items():
            prev_size = prev_book['bids'].get(price, 0)
            
            if prev_size > 0 and size != prev_size:
                # Fixed changes level
                change_info = {
                    'timestamp': timestamp,
                    'prev_size': prev_size,
                    'curr_size': size,
                    'side': 'bid',
                    'change_type': 'refresh' if size > 0 else 'cancel'
                }
                self.level_refreshes[price].append(change_info)
        
        # Analysis changes ask levels
        for price, size in curr_book['asks'].items():
            prev_size = prev_book['asks'].get(price, 0)
            
            if prev_size > 0 and size != prev_size:
                change_info = {
                    'timestamp': timestamp,
                    'prev_size': prev_size,
                    'curr_size': size,
                    'side': 'ask',
                    'change_type': 'refresh' if size > 0 else 'cancel'
                }
                self.level_refreshes[price].append(change_info)
    
    async def _cleanup_old_data(self, current_time: float):
        """Cleanup obsolete data"""
        cutoff_time = current_time - self.detection_window
        
        # Cleanup history refreshes
        for price in list(self.level_refreshes.keys()):
            self.level_refreshes[price] = [
                refresh for refresh in self.level_refreshes[price]
                if refresh['timestamp'] > cutoff_time
            ]
            
            if not self.level_refreshes[price]:
                del self.level_refreshes[price]
        
        # Cleanup inactive icebergs
        for price in list(self.detected_icebergs.keys()):
            if self.detected_icebergs[price].last_activity < cutoff_time:
                del self.detected_icebergs[price]
    
    async def _update_statistics(self, order_book: Dict[str, List[Tuple[float, float]]]):
        """Update statistics for calibration"""
        # Statistics sizes orders
        all_sizes = []
        for price, size in order_book.get('bids', []):
            all_sizes.append(size)
        for price, size in order_book.get('asks', []):
            all_sizes.append(size)
        
        if all_sizes:
            self.size_statistics['samples'].extend(all_sizes)
            
            # Limitation number samples for performance
            if len(self.size_statistics['samples']) > 10000:
                self.size_statistics['samples'] = self.size_statistics['samples'][-5000:]
            
            self.size_statistics['mean'] = np.mean(self.size_statistics['samples'])
            self.size_statistics['std'] = np.std(self.size_statistics['samples'])
    
    async def _detect_by_refresh_pattern(self, timestamp: float) -> List[IcebergDetectionResult]:
        """Detection by pattern frequent updates"""
        results = []
        
        for price, refreshes in self.level_refreshes.items():
            if len(refreshes) < self.min_refresh_count:
                continue
            
            # Analysis intervals between updates
            refresh_times = [r['timestamp'] for r in refreshes]
            intervals = [refresh_times[i] - refresh_times[i-1] for i in range(1, len(refresh_times))]
            
            if not intervals:
                continue
            
            avg_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            # Validation on regularity updates (feature algorithmic trading)
            regularity_score = 1.0 - min(std_interval / (avg_interval + 1e-6), 1.0)
            
            # Validation frequencies updates
            if avg_interval <= self.refresh_interval_threshold and regularity_score > 0.5:
                
                # Estimation sizes
                sizes = [r['curr_size'] for r in refreshes if r['curr_size'] > 0]
                if not sizes:
                    continue
                
                avg_size = np.mean(sizes)
                total_volume = sum(sizes)
                
                # Estimation hidden volume
                hidden_size_estimate = total_volume * 0.7  # Empirical estimation
                
                # Calculation confidence
                confidence = min(
                    (len(refreshes) / 10.0) * 0.4 +  # Frequency updates
                    regularity_score * 0.3 +         # Regularity
                    min(avg_size / (self.size_statistics['mean'] + 1e-6), 3) / 3 * 0.3,  # Size
                    1.0
                )
                
                if confidence >= self.confidence_threshold:
                    # Determination side
                    side = refreshes[0]['side']
                    
                    signature = IcebergSignature(
                        price_level=price,
                        side=side,
                        total_estimated_size=total_volume + hidden_size_estimate,
                        visible_size=avg_size,
                        hidden_size_estimate=hidden_size_estimate,
                        refresh_count=len(refreshes),
                        avg_refresh_interval=avg_interval,
                        detection_confidence=confidence,
                        strength=self._calculate_strength(confidence, len(refreshes)),
                        first_detected=refresh_times[0],
                        last_activity=refresh_times[-1]
                    )
                    
                    iceberg_type = IcebergType.BUY_ICEBERG if side == 'bid' else IcebergType.SELL_ICEBERG
                    
                    results.append(IcebergDetectionResult(
                        iceberg_type=iceberg_type,
                        signature=signature,
                        market_impact=self._estimate_market_impact(signature),
                        price_influence=self._estimate_price_influence(signature),
                        volume_absorption=total_volume,
                        liquidity_provision=avg_size,
                        detection_algorithm="refresh_pattern",
                        metadata={
                            'regularity_score': regularity_score,
                            'avg_interval': avg_interval,
                            'std_interval': std_interval
                        }
                    ))
        
        return results
    
    async def _detect_by_size_anomaly(self, order_book: Dict[str, List[Tuple[float, float]]], 
                                    timestamp: float) -> List[IcebergDetectionResult]:
        """Detection by anomalous dimensions orders"""
        results = []
        
        if self.size_statistics['std'] == 0:
            return results
        
        # Analysis bid side
        for price, size in order_book.get('bids', []):
            anomaly_score = (size - self.size_statistics['mean']) / self.size_statistics['std']
            
            if anomaly_score > self.size_threshold_multiplier:
                confidence = min(anomaly_score / 5.0, 0.9)
                
                signature = IcebergSignature(
                    price_level=price,
                    side='bid',
                    total_estimated_size=size * 1.5,  # Estimation full size
                    visible_size=size,
                    hidden_size_estimate=size * 0.5,
                    refresh_count=1,
                    avg_refresh_interval=0,
                    detection_confidence=confidence,
                    strength=self._calculate_strength(confidence, 1),
                    first_detected=timestamp,
                    last_activity=timestamp
                )
                
                results.append(IcebergDetectionResult(
                    iceberg_type=IcebergType.BUY_ICEBERG,
                    signature=signature,
                    market_impact=self._estimate_market_impact(signature),
                    price_influence=self._estimate_price_influence(signature),
                    volume_absorption=size,
                    liquidity_provision=size,
                    detection_algorithm="size_anomaly",
                    metadata={'anomaly_score': anomaly_score}
                ))
        
        # Analysis ask side
        for price, size in order_book.get('asks', []):
            anomaly_score = (size - self.size_statistics['mean']) / self.size_statistics['std']
            
            if anomaly_score > self.size_threshold_multiplier:
                confidence = min(anomaly_score / 5.0, 0.9)
                
                signature = IcebergSignature(
                    price_level=price,
                    side='ask',
                    total_estimated_size=size * 1.5,
                    visible_size=size,
                    hidden_size_estimate=size * 0.5,
                    refresh_count=1,
                    avg_refresh_interval=0,
                    detection_confidence=confidence,
                    strength=self._calculate_strength(confidence, 1),
                    first_detected=timestamp,
                    last_activity=timestamp
                )
                
                results.append(IcebergDetectionResult(
                    iceberg_type=IcebergType.SELL_ICEBERG,
                    signature=signature,
                    market_impact=self._estimate_market_impact(signature),
                    price_influence=self._estimate_price_influence(signature),
                    volume_absorption=size,
                    liquidity_provision=size,
                    detection_algorithm="size_anomaly",
                    metadata={'anomaly_score': anomaly_score}
                ))
        
        return results
    
    async def _detect_by_volume_absorption(self, timestamp: float) -> List[IcebergDetectionResult]:
        """Detection by absorption volume"""
        results = []
        
        if len(self.order_book_history) < 5:
            return results
        
        # Analysis recent changes in order book
        recent_books = list(self.order_book_history)[-5:]
        
        # Search levels, which absorb many volume, but remain stable
        for book_idx in range(1, len(recent_books)):
            curr_book = recent_books[book_idx]
            prev_book = recent_books[book_idx - 1]
            
            # Analysis bid levels
            for price, size in curr_book['bids'].items():
                prev_size = prev_book['bids'].get(price, 0)
                
                # Search stable levels with high turnover
                if prev_size > 0 and abs(size - prev_size) < prev_size * 0.1:  # Size stable
                    # Validation activity trading on in this level
                    volume_activity = self._calculate_level_activity(price, recent_books)
                    
                    if volume_activity > size * 2:  # Turnover exceeds visible size
                        confidence = min(volume_activity / (size * 5), 0.8)
                        
                        signature = IcebergSignature(
                            price_level=price,
                            side='bid',
                            total_estimated_size=volume_activity,
                            visible_size=size,
                            hidden_size_estimate=volume_activity - size,
                            refresh_count=len(recent_books),
                            avg_refresh_interval=(curr_book['timestamp'] - recent_books[0]['timestamp']) / len(recent_books),
                            detection_confidence=confidence,
                            strength=self._calculate_strength(confidence, len(recent_books)),
                            first_detected=recent_books[0]['timestamp'],
                            last_activity=timestamp
                        )
                        
                        results.append(IcebergDetectionResult(
                            iceberg_type=IcebergType.BUY_ICEBERG,
                            signature=signature,
                            market_impact=self._estimate_market_impact(signature),
                            price_influence=self._estimate_price_influence(signature),
                            volume_absorption=volume_activity,
                            liquidity_provision=size,
                            detection_algorithm="volume_absorption",
                            metadata={'volume_activity': volume_activity}
                        ))
        
        return results
    
    async def _detect_by_level_stability(self, timestamp: float) -> List[IcebergDetectionResult]:
        """Detection by stability levels"""
        results = []
        
        if len(self.order_book_history) < 10:
            return results
        
        # Analysis stability levels for period
        level_stability = defaultdict(list)
        
        for book in self.order_book_history:
            for price, size in book['bids'].items():
                level_stability[('bid', price)].append(size)
            for price, size in book['asks'].items():
                level_stability[('ask', price)].append(size)
        
        # Search anomalously stable levels
        for (side, price), sizes in level_stability.items():
            if len(sizes) < 5:  # Insufficient data
                continue
            
            # Calculation coefficient variations
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            cv = std_size / mean_size if mean_size > 0 else float('inf')
            
            # Stable level has low coefficient variations
            if cv < 0.2 and mean_size > self.size_statistics['mean']:
                confidence = min((1.0 - cv) * 0.7, 0.8)
                
                signature = IcebergSignature(
                    price_level=price,
                    side=side,
                    total_estimated_size=mean_size * 2,  # Estimation full size
                    visible_size=mean_size,
                    hidden_size_estimate=mean_size,
                    refresh_count=len(sizes),
                    avg_refresh_interval=self.detection_window / len(sizes),
                    detection_confidence=confidence,
                    strength=self._calculate_strength(confidence, len(sizes)),
                    first_detected=timestamp - self.detection_window,
                    last_activity=timestamp
                )
                
                iceberg_type = IcebergType.BUY_ICEBERG if side == 'bid' else IcebergType.SELL_ICEBERG
                
                results.append(IcebergDetectionResult(
                    iceberg_type=iceberg_type,
                    signature=signature,
                    market_impact=self._estimate_market_impact(signature),
                    price_influence=self._estimate_price_influence(signature),
                    volume_absorption=mean_size * len(sizes),
                    liquidity_provision=mean_size,
                    detection_algorithm="level_stability",
                    metadata={
                        'coefficient_of_variation': cv,
                        'stability_periods': len(sizes)
                    }
                ))
        
        return results
    
    def _calculate_level_activity(self, price: float, books: List[Dict]) -> float:
        """Calculation activity trading on level"""
        total_activity = 0.0
        
        for i in range(1, len(books)):
            curr_book = books[i]
            prev_book = books[i-1]
            
            curr_size = curr_book['bids'].get(price, 0) + curr_book['asks'].get(price, 0)
            prev_size = prev_book['bids'].get(price, 0) + prev_book['asks'].get(price, 0)
            
            # Estimation turnover as absolute change size
            if prev_size > 0 and curr_size > 0:
                total_activity += abs(curr_size - prev_size)
        
        return total_activity
    
    def _calculate_strength(self, confidence: float, refresh_count: int) -> IcebergStrength:
        """Calculation force iceberg signal"""
        combined_score = confidence * 0.7 + min(refresh_count / 20, 1.0) * 0.3
        
        if combined_score >= 0.8:
            return IcebergStrength.VERY_STRONG
        elif combined_score >= 0.7:
            return IcebergStrength.STRONG
        elif combined_score >= 0.5:
            return IcebergStrength.MODERATE
        else:
            return IcebergStrength.WEAK
    
    def _estimate_market_impact(self, signature: IcebergSignature) -> float:
        """Estimation impact on market"""
        # Simple estimation on basis size relatively average liquidity
        if self.size_statistics['mean'] > 0:
            impact = signature.total_estimated_size / (self.size_statistics['mean'] * 10)
            return min(impact, 1.0)
        return 0.5
    
    def _estimate_price_influence(self, signature: IcebergSignature) -> float:
        """Estimation influence on price"""
        # Estimation on basis size and proximity to market
        base_influence = min(signature.total_estimated_size / (self.size_statistics['mean'] * 5), 1.0)
        
        # Additional weight for prolonged activity
        duration_factor = min(signature.activity_duration / 300, 1.0)  # 5 minutes = maximum
        
        return base_influence * (1 + duration_factor * 0.5)
    
    async def _update_active_icebergs(self, new_results: List[IcebergDetectionResult], timestamp: float):
        """Update list active icebergs"""
        # Update existing and addition new
        for result in new_results:
            price = result.signature.price_level
            
            if price in self.detected_icebergs:
                # Update existing
                existing = self.detected_icebergs[price]
                existing.last_activity = timestamp
                existing.refresh_count = result.signature.refresh_count
                existing.detection_confidence = max(existing.detection_confidence, result.signature.detection_confidence)
            else:
                # Addition new
                self.detected_icebergs[price] = result.signature
    
    def get_active_icebergs(self) -> List[IcebergSignature]:
        """Retrieval list active icebergs"""
        current_time = time.time()
        active_threshold = current_time - 60  # Active in last minute
        
        return [
            iceberg for iceberg in self.detected_icebergs.values()
            if iceberg.last_activity > active_threshold
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retrieval statistics detector"""
        return {
            'total_detected': len(self.detected_icebergs),
            'active_icebergs': len(self.get_active_icebergs()),
            'tracked_levels': len(self.level_refreshes),
            'order_book_samples': len(self.order_book_history),
            'avg_order_size': self.size_statistics['mean'],
            'size_std': self.size_statistics['std'],
            'detection_settings': {
                'min_refresh_count': self.min_refresh_count,
                'size_threshold_multiplier': self.size_threshold_multiplier,
                'confidence_threshold': self.confidence_threshold
            }
        }
    
    async def calibrate_thresholds(self, historical_data: List[Dict]) -> None:
        """Calibration thresholds on basis historical data"""
        # Analysis historical data for optimization thresholds
        if not historical_data:
            return
        
        # Collection statistics sizes
        all_sizes = []
        for data in historical_data:
            for price, size in data.get('bids', []):
                all_sizes.append(size)
            for price, size in data.get('asks', []):
                all_sizes.append(size)
        
        if all_sizes:
            mean_size = np.mean(all_sizes)
            std_size = np.std(all_sizes)
            
            # Adaptive configuration thresholds
            self.size_threshold_multiplier = max(2.0, 2 + std_size / mean_size)
            
            self.logger.logger.info(
                "Thresholds calibrated",
                mean_size=mean_size,
                std_size=std_size,
                new_size_threshold=self.size_threshold_multiplier
            )

# Utility functions
def calculate_iceberg_probability(signature: IcebergSignature) -> float:
    """Calculation probability presence iceberg order"""
    factors = {
        'refresh_frequency': min(signature.refresh_count / 10, 1.0) * 0.3,
        'size_consistency': (1.0 - abs(signature.visible_size - signature.total_estimated_size / signature.refresh_count) / signature.visible_size) * 0.2,
        'duration': min(signature.activity_duration / 600, 1.0) * 0.2,  # 10 minutes maximum
        'confidence': signature.detection_confidence * 0.3
    }
    
    return sum(factors.values())

def estimate_total_iceberg_size(visible_refreshes: List[float], time_intervals: List[float]) -> float:
    """Estimation total size iceberg order"""
    if not visible_refreshes or not time_intervals:
        return 0.0
    
    avg_visible_size = np.mean(visible_refreshes)
    total_time = sum(time_intervals)
    avg_interval = np.mean(time_intervals)
    
    # Estimation number hidden refreshes
    estimated_total_refreshes = total_time / avg_interval * 1.5  # 50% hidden
    
    return avg_visible_size * estimated_total_refreshes

# Export main components
__all__ = [
    'IcebergDetector',
    'IcebergSignature',
    'IcebergDetectionResult',
    'OrderBookLevel',
    'IcebergType',
    'IcebergStrength',
    'calculate_iceberg_probability',
    'estimate_total_iceberg_size'
]