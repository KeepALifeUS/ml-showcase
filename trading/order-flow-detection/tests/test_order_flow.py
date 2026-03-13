"""
Unit tests for Order Flow Analysis components
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch

from src.order_flow.delta_analyzer import DeltaAnalyzer, DeltaMetrics, DeltaSignal
from src.order_flow.cumulative_delta import CumulativeDeltaAnalyzer, CumulativeDeltaBar
from src.order_flow.footprint_chart import FootprintAnalyzer, FootprintBar, PriceLevel

@pytest.fixture
def delta_analyzer():
    """Create DeltaAnalyzer instance for testing"""
    return DeltaAnalyzer("BTCUSDT", tick_size=0.01)

@pytest.fixture
def cumulative_analyzer():
    """Create CumulativeDeltaAnalyzer instance for testing"""
    return CumulativeDeltaAnalyzer("BTCUSDT", timeframe_minutes=1)

@pytest.fixture
def footprint_analyzer():
    """Create FootprintAnalyzer instance for testing"""
    return FootprintAnalyzer("BTCUSDT", timeframe_seconds=60)

@pytest.fixture
def sample_trades():
    """Sample trade data for testing"""
    return [
        (50000.0, 1.5, True, time.time()),   # Buy
        (49999.5, 2.0, False, time.time()+1), # Sell
        (50000.5, 1.8, True, time.time()+2),  # Buy
        (49999.0, 1.2, False, time.time()+3), # Sell
        (50001.0, 3.0, True, time.time()+4),  # Buy
    ]

class TestDeltaAnalyzer:
    """Tests for DeltaAnalyzer"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, delta_analyzer):
        """Test analyzer initialization"""
        assert delta_analyzer.symbol == "BTCUSDT"
        assert delta_analyzer.tick_size == 0.01
        assert delta_analyzer.cumulative_delta == 0.0
        assert len(delta_analyzer.trades) == 0
        
    @pytest.mark.asyncio
    async def test_add_trade_buy(self, delta_analyzer):
        """Test adding buy trade"""
        price, volume, is_buy, timestamp = 50000.0, 1.5, True, time.time()
        
        metrics = await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
        
        assert isinstance(metrics, DeltaMetrics)
        assert metrics.buy_volume >= volume
        assert metrics.sell_volume == 0
        assert metrics.net_delta > 0
        assert delta_analyzer.cumulative_delta == volume
        
    @pytest.mark.asyncio
    async def test_add_trade_sell(self, delta_analyzer):
        """Test adding sell trade"""
        price, volume, is_buy, timestamp = 50000.0, 1.5, False, time.time()
        
        metrics = await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
        
        assert metrics.buy_volume == 0
        assert metrics.sell_volume >= volume
        assert metrics.net_delta < 0
        assert delta_analyzer.cumulative_delta == -volume
        
    @pytest.mark.asyncio
    async def test_multiple_trades(self, delta_analyzer, sample_trades):
        """Test processing multiple trades"""
        total_buy = 0
        total_sell = 0
        
        for price, volume, is_buy, timestamp in sample_trades:
            metrics = await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            if is_buy:
                total_buy += volume
            else:
                total_sell += volume
                
        expected_cumulative = total_buy - total_sell
        assert abs(delta_analyzer.cumulative_delta - expected_cumulative) < 0.01
        
    @pytest.mark.asyncio
    async def test_pattern_detection(self, delta_analyzer, sample_trades):
        """Test pattern detection"""
        # Add trades to build history
        for price, volume, is_buy, timestamp in sample_trades:
            await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        patterns = await delta_analyzer.detect_patterns(lookback_periods=5)
        
        assert isinstance(patterns, list)
        # Patterns may or may not be found with sample data
        
    @pytest.mark.asyncio
    async def test_real_time_signal(self, delta_analyzer, sample_trades):
        """Test real-time signal generation"""
        # Add some trades
        for price, volume, is_buy, timestamp in sample_trades[:3]:
            await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        signal = await delta_analyzer.get_real_time_signal()
        
        assert 'signal' in signal
        assert 'confidence' in signal
        assert 'metrics' in signal
        assert isinstance(signal['signal'], DeltaSignal)
        assert 0 <= signal['confidence'] <= 1
        
    @pytest.mark.asyncio
    async def test_reset(self, delta_analyzer, sample_trades):
        """Test analyzer reset"""
        # Add trades
        for price, volume, is_buy, timestamp in sample_trades:
            await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        assert delta_analyzer.cumulative_delta != 0
        assert len(delta_analyzer.trades) > 0
        
        # Reset
        await delta_analyzer.reset()
        
        assert delta_analyzer.cumulative_delta == 0
        assert len(delta_analyzer.trades) == 0
        assert len(delta_analyzer.delta_history) == 0
        
    def test_statistics(self, delta_analyzer):
        """Test statistics retrieval"""
        stats = delta_analyzer.get_statistics()
        
        assert isinstance(stats, dict)
        # Empty analyzer should return empty stats or defaults

class TestCumulativeDeltaAnalyzer:
    """Tests for CumulativeDeltaAnalyzer"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, cumulative_analyzer):
        """Test analyzer initialization"""
        assert cumulative_analyzer.symbol == "BTCUSDT"
        assert cumulative_analyzer.timeframe_minutes == 1
        assert cumulative_analyzer.cumulative_delta == 0.0
        assert len(cumulative_analyzer.delta_bars) == 0
        
    @pytest.mark.asyncio
    async def test_bar_creation(self, cumulative_analyzer):
        """Test delta bar creation"""
        price, volume, is_buy, timestamp = 50000.0, 1.5, True, time.time()
        
        bar = await cumulative_analyzer.add_trade(price, volume, is_buy, timestamp)
        
        assert isinstance(bar, CumulativeDeltaBar)
        assert bar.volume == volume
        assert bar.buy_volume == volume if is_buy else 0
        assert bar.sell_volume == 0 if is_buy else volume
        
    @pytest.mark.asyncio
    async def test_divergence_detection(self, cumulative_analyzer):
        """Test divergence detection"""
        # Create sample price data
        price_data = [(time.time() + i, 50000 + i*10) for i in range(20)]
        
        # Add trades to build history
        for i, (timestamp, price) in enumerate(price_data):
            volume = 1.0
            is_buy = i % 2 == 0  # Alternate buy/sell
            await cumulative_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        divergences = await cumulative_analyzer.detect_divergences(price_data)
        
        assert isinstance(divergences, list)
        
    @pytest.mark.asyncio
    async def test_delta_strength_signal(self, cumulative_analyzer, sample_trades):
        """Test delta strength signal"""
        # Add trades
        for price, volume, is_buy, timestamp in sample_trades:
            await cumulative_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        signal = await cumulative_analyzer.get_delta_strength_signal()
        
        assert 'signal' in signal
        assert 'confidence' in signal
        assert 'buy_ratio' in signal

class TestFootprintAnalyzer:
    """Tests for FootprintAnalyzer"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, footprint_analyzer):
        """Test analyzer initialization"""
        assert footprint_analyzer.symbol == "BTCUSDT"
        assert footprint_analyzer.timeframe_seconds == 60
        assert len(footprint_analyzer.footprint_bars) == 0
        
    @pytest.mark.asyncio
    async def test_trade_processing(self, footprint_analyzer):
        """Test trade processing"""
        price = 50000.0
        volume = 1.5
        is_buy = True
        bid = 49999.5
        ask = 50000.5
        timestamp = time.time()
        
        bar = await footprint_analyzer.add_trade(price, volume, is_buy, bid, ask, timestamp)
        
        assert isinstance(bar, FootprintBar)
        assert bar.volume == volume
        assert price in [level.price for level in bar.price_levels.values()]
        
    @pytest.mark.asyncio
    async def test_pattern_detection(self, footprint_analyzer, sample_trades):
        """Test footprint pattern detection"""
        # Add trades
        for price, volume, is_buy, timestamp in sample_trades:
            await footprint_analyzer.add_trade(price, volume, is_buy, timestamp=timestamp)
            
        patterns = await footprint_analyzer.detect_patterns(lookback_bars=5)
        
        assert isinstance(patterns, list)
        
    @pytest.mark.asyncio
    async def test_volume_profile(self, footprint_analyzer, sample_trades):
        """Test volume profile generation"""
        # Add trades
        for price, volume, is_buy, timestamp in sample_trades:
            await footprint_analyzer.add_trade(price, volume, is_buy, timestamp=timestamp)
            
        volume_profile = footprint_analyzer.get_volume_profile()
        
        assert isinstance(volume_profile, dict)
        
    @pytest.mark.asyncio
    async def test_market_structure_levels(self, footprint_analyzer, sample_trades):
        """Test market structure level identification"""
        # Add sufficient trades
        for i in range(20):
            price = 50000 + (i % 10) * 0.5
            volume = 1.0 + (i % 3) * 0.5
            is_buy = i % 2 == 0
            timestamp = time.time() + i
            await footprint_analyzer.add_trade(price, volume, is_buy, timestamp=timestamp)
            
        levels = footprint_analyzer.get_market_structure_levels()
        
        assert isinstance(levels, dict)
        assert 'high_volume_nodes' in levels
        assert 'support_levels' in levels
        assert 'resistance_levels' in levels

class TestUtilityFunctions:
    """Tests for utility functions"""
    
    def test_classify_trade_by_delta(self):
        """Test trade classification"""
        from src.order_flow.delta_analyzer import classify_trade_by_delta
        
        # Test buy classification
        is_buy, volume = classify_trade_by_delta(50000.5, 50000.0, 50001.0, 1.5)
        assert is_buy == True
        assert volume == 1.5
        
        # Test sell classification
        is_buy, volume = classify_trade_by_delta(49999.5, 50000.0, 50001.0, 1.5)
        assert is_buy == False
        assert volume == 1.5
        
    @pytest.mark.asyncio
    async def test_calculate_vpin(self):
        """Test VPIN calculation"""
        from src.order_flow.delta_analyzer import calculate_vpin
        
        trades = [
            {'buy_delta': 1.0, 'sell_delta': 0.5},
            {'buy_delta': 0.8, 'sell_delta': 1.2},
            {'buy_delta': 1.5, 'sell_delta': 0.3},
        ]
        
        vpin = await calculate_vpin(trades)
        
        assert isinstance(vpin, float)
        assert 0 <= vpin <= 1
        
    @pytest.mark.asyncio
    async def test_delta_momentum(self):
        """Test delta momentum calculation"""
        from src.order_flow.cumulative_delta import calculate_delta_momentum, CumulativeDeltaBar
        
        bars = [
            CumulativeDeltaBar(time.time(), 0, 0, 0, i, 1.0, 0.5, 0.5, i-1)
            for i in range(20)
        ]
        
        momentum = await calculate_delta_momentum(bars, period=10)
        
        assert isinstance(momentum, (int, float))

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_delta_analyzer_performance(self, delta_analyzer):
        """Test delta analyzer performance with many trades"""
        start_time = time.time()
        
        # Process 1000 trades
        for i in range(1000):
            price = 50000 + (i % 100) * 0.01
            volume = 0.1 + (i % 10) * 0.05
            is_buy = i % 2 == 0
            timestamp = time.time() + i * 0.001
            
            await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        elapsed = time.time() - start_time
        
        # Should process 1000 trades in reasonable time
        assert elapsed < 5.0  # 5 seconds max
        assert len(delta_analyzer.trades) > 0
        
    @pytest.mark.asyncio
    async def test_pattern_detection_performance(self, delta_analyzer):
        """Test pattern detection performance"""
        # Add trades first
        for i in range(100):
            price = 50000 + np.sin(i * 0.1) * 100
            volume = 0.5 + np.random.random() * 0.5
            is_buy = np.random.random() > 0.5
            timestamp = time.time() + i * 0.1
            
            await delta_analyzer.add_trade(price, volume, is_buy, timestamp)
            
        start_time = time.time()
        patterns = await delta_analyzer.detect_patterns(lookback_periods=50)
        elapsed = time.time() - start_time
        
        # Pattern detection should be fast
        assert elapsed < 1.0  # 1 second max
        assert isinstance(patterns, list)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_invalid_trade_data(self, delta_analyzer):
        """Test handling of invalid trade data"""
        # Test negative volume
        with pytest.raises((ValueError, AssertionError)):
            await delta_analyzer.add_trade(50000.0, -1.0, True)
            
        # Test zero volume
        metrics = await delta_analyzer.add_trade(50000.0, 0.0, True)
        assert isinstance(metrics, DeltaMetrics)
        
    @pytest.mark.asyncio
    async def test_empty_data_patterns(self, delta_analyzer):
        """Test pattern detection with no data"""
        patterns = await delta_analyzer.detect_patterns()
        
        assert isinstance(patterns, list)
        assert len(patterns) == 0
        
    @pytest.mark.asyncio
    async def test_concurrent_access(self, delta_analyzer):
        """Test concurrent access to analyzer"""
        async def add_trades(start_idx):
            for i in range(start_idx, start_idx + 10):
                await delta_analyzer.add_trade(
                    50000.0 + i * 0.01, 
                    0.1, 
                    i % 2 == 0,
                    time.time() + i * 0.001
                )
        
        # Run concurrent tasks
        tasks = [add_trades(i * 10) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Should handle concurrent access gracefully
        assert len(delta_analyzer.trades) >= 50

@pytest.mark.integration
class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline"""
        # Initialize all analyzers
        delta_analyzer = DeltaAnalyzer("BTCUSDT")
        cumulative_analyzer = CumulativeDeltaAnalyzer("BTCUSDT")
        footprint_analyzer = FootprintAnalyzer("BTCUSDT")
        
        # Simulate realistic trading scenario
        base_price = 50000.0
        timestamp = time.time()
        
        for i in range(100):
            # Simulate price movement with some volatility
            price = base_price + np.sin(i * 0.1) * 50 + np.random.normal(0, 10)
            volume = 0.1 + np.random.exponential(0.3)
            is_buy = np.random.random() > 0.5
            
            # Process through all analyzers
            delta_metrics = await delta_analyzer.add_trade(price, volume, is_buy, timestamp + i)
            cum_bar = await cumulative_analyzer.add_trade(price, volume, is_buy, timestamp + i)
            footprint_bar = await footprint_analyzer.add_trade(price, volume, is_buy, timestamp=timestamp + i)
            
        # Check that all analyzers processed data
        assert len(delta_analyzer.trades) > 0
        assert len(cumulative_analyzer.delta_bars) > 0
        assert len(footprint_analyzer.footprint_bars) > 0
        
        # Test pattern detection
        delta_patterns = await delta_analyzer.detect_patterns()
        footprint_patterns = await footprint_analyzer.detect_patterns()
        
        assert isinstance(delta_patterns, list)
        assert isinstance(footprint_patterns, list)
        
        # Test signal generation
        real_time_signal = await delta_analyzer.get_real_time_signal()
        delta_strength_signal = await cumulative_analyzer.get_delta_strength_signal()
        
        assert 'signal' in real_time_signal
        assert 'signal' in delta_strength_signal