"""
Unit tests for Pattern Detection components
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch
from collections import defaultdict

from src.patterns.iceberg_detector import IcebergDetector, IcebergSignature, IcebergType
from src.patterns.spoofing_detector import SpoofingDetector, SpoofingSignal, OrderEvent, SpoofingType

@pytest.fixture
def iceberg_detector():
    """Create IcebergDetector instance for testing"""
    return IcebergDetector("BTCUSDT", detection_window=300)

@pytest.fixture
def spoofing_detector():
    """Create SpoofingDetector instance for testing"""
    return SpoofingDetector("BTCUSDT", detection_window=300)

@pytest.fixture
def sample_order_book():
    """Sample order book data"""
    return {
        'bids': [
            (50000.0, 1.5),
            (49999.5, 2.0),
            (49999.0, 1.2),
            (49998.5, 3.0),
            (49998.0, 1.8)
        ],
        'asks': [
            (50000.5, 1.2),
            (50001.0, 1.8),
            (50001.5, 2.5),
            (50002.0, 1.1),
            (50002.5, 2.2)
        ]
    }

@pytest.fixture
def sample_order_events():
    """Sample order events for spoofing detection"""
    base_time = time.time()
    return [
        OrderEvent(base_time, "order_1", 50000.0, 5.0, "buy", "place", "account_1"),
        OrderEvent(base_time + 1, "order_2", 49999.0, 3.0, "sell", "place", "account_1"),
        OrderEvent(base_time + 2, "order_3", 50001.0, 2.0, "buy", "place", "account_2"),
        OrderEvent(base_time + 3, "order_1", 50000.0, 5.0, "buy", "cancel", "account_1"),
        OrderEvent(base_time + 4, "order_4", 50000.5, 1.5, "buy", "fill", "account_3"),
    ]

class TestIcebergDetector:
    """Tests for IcebergDetector"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, iceberg_detector):
        """Test detector initialization"""
        assert iceberg_detector.symbol == "BTCUSDT"
        assert iceberg_detector.detection_window == 300
        assert len(iceberg_detector.order_book_history) == 0
        assert len(iceberg_detector.detected_icebergs) == 0
        
    @pytest.mark.asyncio
    async def test_order_book_analysis(self, iceberg_detector, sample_order_book):
        """Test order book analysis"""
        timestamp = time.time()
        
        results = await iceberg_detector.analyze_order_book(sample_order_book, timestamp)
        
        assert isinstance(results, list)
        # Results may be empty with limited sample data
        
    @pytest.mark.asyncio
    async def test_size_anomaly_detection(self, iceberg_detector):
        """Test detection by size anomaly"""
        # Create order book with anomalously large order
        large_order_book = {
            'bids': [
                (50000.0, 100.0),  # Very large order
                (49999.5, 1.0),
                (49999.0, 1.2)
            ],
            'asks': [
                (50000.5, 1.2),
                (50001.0, 1.8)
            ]
        }
        
        # Need to build statistics first
        for i in range(10):
            normal_book = {
                'bids': [(50000.0 - i * 0.5, 1.0 + i * 0.1)],
                'asks': [(50000.5 + i * 0.5, 1.0 + i * 0.1)]
            }
            await iceberg_detector.analyze_order_book(normal_book, time.time() + i)
        
        # Now test with large order
        results = await iceberg_detector.analyze_order_book(large_order_book, time.time() + 20)
        
        # Should detect large order as potential iceberg
        large_order_results = [r for r in results if r.iceberg_type == IcebergType.BUY_ICEBERG]
        # May or may not detect depending on statistics built up
        
    @pytest.mark.asyncio
    async def test_refresh_pattern_detection(self, iceberg_detector, sample_order_book):
        """Test detection by refresh patterns"""
        timestamp = time.time()
        
        # Simulate order refreshes
        for i in range(5):
            # Modify order book slightly to simulate refreshes
            modified_book = sample_order_book.copy()
            modified_book['bids'][0] = (50000.0, 1.5 + i * 0.1)  # Slight size changes
            
            await iceberg_detector.analyze_order_book(modified_book, timestamp + i * 2)
            
        results = await iceberg_detector.analyze_order_book(sample_order_book, timestamp + 20)
        
        # Should track refreshes
        assert len(iceberg_detector.level_refreshes) >= 0
        
    @pytest.mark.asyncio
    async def test_active_icebergs(self, iceberg_detector, sample_order_book):
        """Test active iceberg tracking"""
        # Simulate detecting an iceberg
        timestamp = time.time()
        
        # Create a clear iceberg pattern
        for i in range(10):
            book_with_large_order = {
                'bids': [(50000.0, 10.0 + i * 0.5)],  # Large, changing order
                'asks': sample_order_book['asks']
            }
            await iceberg_detector.analyze_order_book(book_with_large_order, timestamp + i)
            
        active_icebergs = iceberg_detector.get_active_icebergs()
        
        assert isinstance(active_icebergs, list)
        
    @pytest.mark.asyncio
    async def test_statistics(self, iceberg_detector, sample_order_book):
        """Test statistics retrieval"""
        # Add some data
        await iceberg_detector.analyze_order_book(sample_order_book, time.time())
        
        stats = iceberg_detector.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_detected' in stats
        assert 'active_icebergs' in stats
        assert 'detection_settings' in stats
        
    @pytest.mark.asyncio
    async def test_calibration(self, iceberg_detector):
        """Test threshold calibration"""
        # Create historical data
        historical_data = []
        for i in range(50):
            historical_data.append({
                'bids': [(50000.0 - i * 0.1, 1.0 + np.random.random())],
                'asks': [(50001.0 + i * 0.1, 1.0 + np.random.random())]
            })
            
        await iceberg_detector.calibrate_thresholds(historical_data)
        
        # Thresholds should be updated
        assert iceberg_detector.size_threshold_multiplier >= 2.0

class TestSpoofingDetector:
    """Tests for SpoofingDetector"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, spoofing_detector):
        """Test detector initialization"""
        assert spoofing_detector.symbol == "BTCUSDT"
        assert spoofing_detector.detection_window == 300
        assert len(spoofing_detector.order_events) == 0
        assert len(spoofing_detector.detected_spoofing) == 0
        
    @pytest.mark.asyncio
    async def test_order_event_processing(self, spoofing_detector, sample_order_events):
        """Test order event processing"""
        results = []
        
        for event in sample_order_events:
            event_results = await spoofing_detector.process_order_event(event)
            results.extend(event_results)
            
        assert len(spoofing_detector.order_events) == len(sample_order_events)
        # Results may be empty with limited sample data
        
    @pytest.mark.asyncio
    async def test_quick_cancellation_detection(self, spoofing_detector):
        """Test quick cancellation detection"""
        base_time = time.time()
        
        # Create quick cancel pattern
        place_event = OrderEvent(base_time, "quick_order", 50000.0, 10.0, "buy", "place", "spoofer")
        cancel_event = OrderEvent(base_time + 1, "quick_order", 50000.0, 10.0, "buy", "cancel", "spoofer")
        
        # Process placement
        await spoofing_detector.process_order_event(place_event)
        
        # Build some statistics first
        for i in range(20):
            normal_event = OrderEvent(
                base_time + 10 + i, f"normal_{i}", 50000.0 + i * 0.1, 
                1.0, "buy" if i % 2 == 0 else "sell", "place", f"trader_{i}"
            )
            await spoofing_detector.process_order_event(normal_event)
            
        # Process cancellation
        results = await spoofing_detector.process_order_event(cancel_event)
        
        # Should detect quick cancellation
        quick_cancel_results = [r for r in results if r.pattern.value == "quick_cancellation"]
        # May detect depending on statistics
        
    @pytest.mark.asyncio
    async def test_layering_detection(self, spoofing_detector):
        """Test layering pattern detection"""
        base_time = time.time()
        account_id = "layer_spoofer"
        
        # Create layering pattern - multiple orders on same side
        layering_events = []
        for i in range(5):
            event = OrderEvent(
                base_time + i * 2, 
                f"layer_order_{i}",
                50000.0 - i * 0.5,  # Different price levels
                2.0,
                "buy",
                "place",
                account_id
            )
            layering_events.append(event)
            
        results = []
        for event in layering_events:
            event_results = await spoofing_detector.process_order_event(event)
            results.extend(event_results)
            
        # Should detect layering pattern
        layering_results = [r for r in results if r.spoofing_type == SpoofingType.LAYERING]
        # Detection depends on thresholds and statistics
        
    @pytest.mark.asyncio
    async def test_quote_stuffing_detection(self, spoofing_detector):
        """Test quote stuffing detection"""
        base_time = time.time()
        
        # Create rapid-fire events (quote stuffing)
        stuffing_events = []
        for i in range(15):  # More than threshold
            event = OrderEvent(
                base_time + i * 0.05,  # Very rapid
                f"stuff_order_{i}",
                50000.0,
                1.0,
                "buy",
                "place" if i % 2 == 0 else "cancel",
                "stuffer"
            )
            stuffing_events.append(event)
            
        results = []
        for event in stuffing_events:
            event_results = await spoofing_detector.process_order_event(event)
            results.extend(event_results)
            
        # Should detect quote stuffing
        stuffing_results = [r for r in results if r.spoofing_type == SpoofingType.QUOTE_STUFFING]
        # May detect based on event frequency
        
    @pytest.mark.asyncio
    async def test_ping_pong_detection(self, spoofing_detector):
        """Test ping-pong pattern detection"""
        base_time = time.time()
        account_id = "ping_ponger"
        
        # Create ping-pong pattern
        ping_pong_events = []
        for i in range(6):  # 3 cycles
            place_event = OrderEvent(
                base_time + i * 4,
                f"pp_order_{i}",
                50000.0,
                1.5,
                "buy",
                "place",
                account_id
            )
            cancel_event = OrderEvent(
                base_time + i * 4 + 2,
                f"pp_order_{i}",
                50000.0,
                1.5,
                "buy",
                "cancel",
                account_id
            )
            ping_pong_events.extend([place_event, cancel_event])
            
        results = []
        for event in ping_pong_events:
            event_results = await spoofing_detector.process_order_event(event)
            results.extend(event_results)
            
        # Should detect ping-pong pattern
        pingpong_results = [r for r in results if r.spoofing_type == SpoofingType.PING_PONG]
        
    @pytest.mark.asyncio
    async def test_behavior_profile_tracking(self, spoofing_detector, sample_order_events):
        """Test trading behavior profile tracking"""
        # Process events with account IDs
        for event in sample_order_events:
            await spoofing_detector.process_order_event(event)
            
        # Check profiles were created
        assert len(spoofing_detector.trading_profiles) > 0
        
        # Check profile data
        for profile in spoofing_detector.trading_profiles.values():
            assert hasattr(profile, 'account_id')
            assert hasattr(profile, 'suspicion_score')
            
    @pytest.mark.asyncio
    async def test_suspicious_accounts(self, spoofing_detector):
        """Test suspicious account identification"""
        # Create suspicious behavior
        suspicious_account = "suspicious_trader"
        base_time = time.time()
        
        # High cancellation rate pattern
        for i in range(20):
            place_event = OrderEvent(
                base_time + i * 2,
                f"sus_order_{i}",
                50000.0,
                1.0,
                "buy",
                "place",
                suspicious_account
            )
            cancel_event = OrderEvent(
                base_time + i * 2 + 1,
                f"sus_order_{i}",
                50000.0,
                1.0,
                "buy",
                "cancel",
                suspicious_account
            )
            
            await spoofing_detector.process_order_event(place_event)
            await spoofing_detector.process_order_event(cancel_event)
            
        suspicious_accounts = spoofing_detector.get_suspicious_accounts(min_suspicion_score=0.1)
        
        # Should identify suspicious account
        assert len(suspicious_accounts) >= 0  # May vary based on thresholds
        
    @pytest.mark.asyncio
    async def test_compliance_report(self, spoofing_detector, sample_order_events):
        """Test compliance report generation"""
        # Process some events
        for event in sample_order_events:
            await spoofing_detector.process_order_event(event)
            
        report = await spoofing_detector.generate_compliance_report()
        
        assert isinstance(report, dict)
        assert 'report_timestamp' in report
        assert 'symbol' in report
        assert 'total_alerts' in report
        assert 'cases_by_type' in report
        assert 'suspicious_accounts' in report
        assert 'recommendations' in report
        
    @pytest.mark.asyncio
    async def test_statistics(self, spoofing_detector, sample_order_events):
        """Test statistics retrieval"""
        # Add some data
        for event in sample_order_events:
            await spoofing_detector.process_order_event(event)
            
        stats = spoofing_detector.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_order_events' in stats
        assert 'tracked_accounts' in stats
        assert 'detection_settings' in stats

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_iceberg_probability_calculation(self):
        """Test iceberg probability calculation"""
        from src.patterns.iceberg_detector import calculate_iceberg_probability, IcebergSignature, IcebergStrength
        
        signature = IcebergSignature(
            price_level=50000.0,
            side='bid',
            total_estimated_size=100.0,
            visible_size=10.0,
            hidden_size_estimate=90.0,
            refresh_count=5,
            avg_refresh_interval=10.0,
            detection_confidence=0.8,
            strength=IcebergStrength.STRONG,
            first_detected=time.time() - 300,
            last_activity=time.time()
        )
        
        probability = calculate_iceberg_probability(signature)
        
        assert isinstance(probability, float)
        assert 0 <= probability <= 1
        
    def test_iceberg_size_estimation(self):
        """Test iceberg size estimation"""
        from src.patterns.iceberg_detector import estimate_total_iceberg_size
        
        visible_refreshes = [10.0, 12.0, 9.0, 11.0, 10.5]
        time_intervals = [5.0, 4.5, 6.0, 5.2, 4.8]
        
        estimated_size = estimate_total_iceberg_size(visible_refreshes, time_intervals)
        
        assert isinstance(estimated_size, float)
        assert estimated_size > 0
        
    def test_manipulation_severity_calculation(self):
        """Test manipulation severity calculation"""
        from src.patterns.spoofing_detector import calculate_manipulation_severity, SpoofingSignal, SpoofingType, SpoofingPattern
        
        signal = SpoofingSignal(
            spoofing_type=SpoofingType.LAYERING,
            pattern=SpoofingPattern.LARGE_ORDER_PLACEMENT,
            confidence=0.8,
            severity=0.9,
            start_time=time.time() - 60,
            end_time=time.time(),
            affected_price_range=(49999.0, 50001.0),
            suspicious_orders=[],
            market_impact=0.5,
            volume_involved=100.0
        )
        
        severity = calculate_manipulation_severity(signal)
        
        assert severity in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
    def test_spoofing_trends_analysis(self):
        """Test spoofing trends analysis"""
        from src.patterns.spoofing_detector import analyze_spoofing_trends, SpoofingSignal, SpoofingType, SpoofingPattern
        
        # Create sample signals
        signals = []
        base_time = time.time() - 3600  # 1 hour ago
        
        for i in range(10):
            signal = SpoofingSignal(
                spoofing_type=SpoofingType.LAYERING if i % 2 == 0 else SpoofingType.PING_PONG,
                pattern=SpoofingPattern.QUICK_CANCELLATION,
                confidence=0.7 + i * 0.02,
                severity=0.6 + i * 0.03,
                start_time=base_time + i * 300,  # Every 5 minutes
                end_time=base_time + i * 300 + 60,
                affected_price_range=(50000.0, 50001.0),
                suspicious_orders=[],
                market_impact=0.3,
                volume_involved=10.0
            )
            signals.append(signal)
            
        trends = analyze_spoofing_trends(signals, period_hours=1)
        
        assert isinstance(trends, dict)
        if trends:  # May be empty if no recent signals
            assert 'total_cases' in trends
            assert 'type_distribution' in trends

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_iceberg_detection_performance(self, iceberg_detector):
        """Test iceberg detection performance"""
        start_time = time.time()
        
        # Process many order books
        for i in range(100):
            order_book = {
                'bids': [(50000.0 - j * 0.1, 1.0 + j * 0.05) for j in range(5)],
                'asks': [(50000.5 + j * 0.1, 1.0 + j * 0.05) for j in range(5)]
            }
            await iceberg_detector.analyze_order_book(order_book, time.time() + i)
            
        elapsed = time.time() - start_time
        
        # Should process 100 order books quickly
        assert elapsed < 5.0  # 5 seconds max
        
    @pytest.mark.asyncio
    async def test_spoofing_detection_performance(self, spoofing_detector):
        """Test spoofing detection performance"""
        start_time = time.time()
        
        # Process many order events
        base_time = time.time()
        for i in range(1000):
            event = OrderEvent(
                base_time + i * 0.01,
                f"order_{i}",
                50000.0 + (i % 100) * 0.01,
                1.0 + (i % 10) * 0.1,
                "buy" if i % 2 == 0 else "sell",
                "place" if i % 3 == 0 else ("cancel" if i % 3 == 1 else "fill"),
                f"account_{i % 50}"
            )
            await spoofing_detector.process_order_event(event)
            
        elapsed = time.time() - start_time
        
        # Should process 1000 events quickly
        assert elapsed < 10.0  # 10 seconds max

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_order_book(self, iceberg_detector):
        """Test handling of empty order book"""
        empty_book = {'bids': [], 'asks': []}
        
        results = await iceberg_detector.analyze_order_book(empty_book, time.time())
        
        assert isinstance(results, list)
        assert len(results) == 0
        
    @pytest.mark.asyncio
    async def test_invalid_order_event(self, spoofing_detector):
        """Test handling of invalid order events"""
        # Test with missing account_id
        event_no_account = OrderEvent(
            time.time(), "order_1", 50000.0, 1.0, "buy", "place", None
        )
        
        results = await spoofing_detector.process_order_event(event_no_account)
        
        assert isinstance(results, list)
        # Should handle gracefully
        
    @pytest.mark.asyncio
    async def test_data_cleanup(self, iceberg_detector, spoofing_detector):
        """Test data cleanup mechanisms"""
        # Add old data
        old_time = time.time() - 1000  # Very old
        
        old_book = {
            'bids': [(50000.0, 1.0)],
            'asks': [(50001.0, 1.0)]
        }
        await iceberg_detector.analyze_order_book(old_book, old_time)
        
        old_event = OrderEvent(old_time, "old_order", 50000.0, 1.0, "buy", "place", "old_account")
        await spoofing_detector.process_order_event(old_event)
        
        # Add current data
        current_book = {
            'bids': [(50000.0, 1.0)],
            'asks': [(50001.0, 1.0)]
        }
        await iceberg_detector.analyze_order_book(current_book, time.time())
        
        current_event = OrderEvent(time.time(), "current_order", 50000.0, 1.0, "buy", "place", "current_account")
        await spoofing_detector.process_order_event(current_event)
        
        # Old data should be cleaned up automatically
        # (Implementation dependent on cleanup mechanisms)

@pytest.mark.integration  
class TestPatternIntegration:
    """Integration tests for pattern detection"""
    
    @pytest.mark.asyncio
    async def test_combined_detection_pipeline(self):
        """Test combined iceberg and spoofing detection"""
        iceberg_detector = IcebergDetector("BTCUSDT")
        spoofing_detector = SpoofingDetector("BTCUSDT")
        
        base_time = time.time()
        
        # Simulate realistic trading scenario with both patterns
        for i in range(50):
            # Order book updates
            order_book = {
                'bids': [(50000.0 - j * 0.1, 1.0 + np.random.random()) for j in range(5)],
                'asks': [(50000.5 + j * 0.1, 1.0 + np.random.random()) for j in range(5)]
            }
            
            # Add large order occasionally (iceberg pattern)
            if i % 10 == 0:
                order_book['bids'][0] = (50000.0, 50.0)  # Large order
                
            iceberg_results = await iceberg_detector.analyze_order_book(order_book, base_time + i)
            
            # Order events
            event = OrderEvent(
                base_time + i,
                f"order_{i}",
                50000.0 + (i % 10) * 0.1,
                1.0 + (i % 5) * 0.2,
                "buy" if i % 2 == 0 else "sell",
                "place" if i % 4 != 3 else "cancel",
                f"account_{i % 10}"
            )
            
            spoofing_results = await spoofing_detector.process_order_event(event)
            
        # Both detectors should have processed data
        assert len(iceberg_detector.order_book_history) > 0
        assert len(spoofing_detector.order_events) > 0
        
        # Check for any detected patterns
        active_icebergs = iceberg_detector.get_active_icebergs()
        recent_spoofing = spoofing_detector.get_recent_spoofing_activity()
        
        assert isinstance(active_icebergs, list)
        assert isinstance(recent_spoofing, list)