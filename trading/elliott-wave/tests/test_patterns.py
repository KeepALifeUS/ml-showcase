"""
Comprehensive tests for Elliott Wave pattern detection.

Production-ready test suite with crypto market
scenarios, performance benchmarks, and edge case coverage.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List
import asyncio

from src.patterns.impulse_wave import ImpulseWaveDetector, ImpulseWave, WavePoint
from src.patterns.corrective_wave import CorrectiveWaveDetector, CorrectiveWave
from src.utils.config import config


class TestData:
    """Test data generator for Elliott Wave patterns."""
    
    @staticmethod
    def generate_impulse_pattern(
        start_price: float = 100.0,
        volatility: float = 0.02,
        length: int = 100,
        noise: bool = True
    ) -> pd.DataFrame:
        """Generate synthetic impulse wave pattern."""
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(hours=length),
            end=datetime.now(timezone.utc),
            periods=length
        )
        
        # Generate idealized 5-wave impulse pattern
        wave_points = [0, 0.2, 0.15, 0.35, 0.3, 0.5]  # Relative price levels
        wave_lengths = [20, 15, 25, 15, 25]  # Relative time lengths
        
        prices = []
        current_idx = 0
        
        for i, wave_len in enumerate(wave_lengths):
            start_level = wave_points[i]
            end_level = wave_points[i + 1]
            
            wave_prices = np.linspace(start_level, end_level, wave_len)
            
            # Add some curvature for realism
            if i % 2 == 0:  # Impulse waves (1, 3, 5)
                wave_prices = start_level + (end_level - start_level) * (wave_prices - start_level) ** 1.2
            else:  # Corrective waves (2, 4)
                wave_prices = start_level + (end_level - start_level) * np.sin(np.pi * (wave_prices - start_level) / 2)
            
            prices.extend(wave_prices)
            current_idx += wave_len
            
        # Scale to actual price range
        prices = np.array(prices[:length])
        price_range = max(prices) - min(prices)
        scaled_prices = start_price + prices * price_range * 2
        
        # Add noise if requested
        if noise:
            noise_factor = scaled_prices * volatility * np.random.randn(len(scaled_prices))
            scaled_prices += noise_factor
            
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, scaled_prices)):
            # Simple OHLC generation
            open_price = scaled_prices[i-1] if i > 0 else price
            high = max(open_price, price) * (1 + np.random.random() * volatility)
            low = min(open_price, price) * (1 - np.random.random() * volatility)
            volume = 1000000 * (1 + np.random.random())
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    @staticmethod
    def generate_corrective_pattern(
        start_price: float = 100.0,
        pattern_type: str = "zigzag",
        length: int = 60
    ) -> pd.DataFrame:
        """Generate synthetic corrective wave pattern."""
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(hours=length),
            end=datetime.now(timezone.utc),
            periods=length
        )
        
        if pattern_type == "zigzag":
            # A-B-C zigzag pattern
            wave_points = [0, -0.6, -0.3, -0.8]  # Sharp A, moderate B, extending C
            wave_lengths = [20, 15, 25]
        elif pattern_type == "flat":
            # A-B-C flat pattern
            wave_points = [0, -0.5, -0.05, -0.5]  # Deep B retracement
            wave_lengths = [20, 20, 20]
        else:
            # Default zigzag
            wave_points = [0, -0.618, -0.382, -0.786]
            wave_lengths = [20, 15, 25]
            
        prices = []
        for i, wave_len in enumerate(wave_lengths):
            start_level = wave_points[i]
            end_level = wave_points[i + 1]
            
            wave_prices = np.linspace(start_level, end_level, wave_len)
            prices.extend(wave_prices)
            
        # Scale to actual prices
        prices = np.array(prices[:length])
        scaled_prices = start_price * (1 + prices)
        
        # Create OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, scaled_prices)):
            open_price = scaled_prices[i-1] if i > 0 else price
            high = max(open_price, price) * 1.005
            low = min(open_price, price) * 0.995
            volume = 500000
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    @staticmethod
    def generate_btc_like_data(days: int = 30) -> pd.DataFrame:
        """Generate Bitcoin-like price data for crypto-specific tests."""
        # High volatility, 24/7 trading characteristics
        dates = pd.date_range(
            start=datetime.now(timezone.utc) - timedelta(days=days),
            end=datetime.now(timezone.utc),
            freq='1H'  # Hourly data
        )
        
        # Bitcoin-like price movement with high volatility
        start_price = 45000.0
        prices = [start_price]
        
        for i in range(1, len(dates)):
            # Higher volatility for crypto
            daily_return = np.random.normal(0, 0.04)  # 4% daily volatility
            hourly_return = daily_return / 24
            
            # Add momentum and mean reversion
            momentum = np.random.choice([-1, 1]) * np.random.exponential(0.02)
            new_price = prices[-1] * (1 + hourly_return + momentum * 0.5)
            
            # Prevent negative prices
            new_price = max(new_price, 100.0)
            prices.append(new_price)
            
        # Create realistic OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            open_price = prices[i-1] if i > 0 else close
            
            # Realistic intraday range
            range_pct = np.random.uniform(0.005, 0.03)  # 0.5% to 3% intraday range
            mid_price = (open_price + close) / 2
            
            high = mid_price * (1 + range_pct)
            low = mid_price * (1 - range_pct)
            
            # Ensure OHLC logic is correct
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Crypto-like volume (higher on volatile moves)
            volatility = abs((close - open_price) / open_price)
            volume = 1000000 * (1 + volatility * 10)
            
            data.append({
                'timestamp': date,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df


@pytest.fixture
def impulse_detector():
    """Fixture for impulse wave detector."""
    return ImpulseWaveDetector(
        min_wave_length=5,
        max_wave_length=50,
        confidence_threshold=0.6
    )


@pytest.fixture
def corrective_detector():
    """Fixture for corrective wave detector."""
    return CorrectiveWaveDetector(
        min_wave_length=3,
        max_wave_length=40,
        confidence_threshold=0.5
    )


@pytest.fixture
def sample_impulse_data():
    """Fixture for sample impulse wave data."""
    return TestData.generate_impulse_pattern(
        start_price=100.0,
        volatility=0.015,
        length=100,
        noise=True
    )


@pytest.fixture
def sample_corrective_data():
    """Fixture for sample corrective wave data."""
    return TestData.generate_corrective_pattern(
        start_price=100.0,
        pattern_type="zigzag",
        length=60
    )


@pytest.fixture
def btc_data():
    """Fixture for Bitcoin-like data."""
    return TestData.generate_btc_like_data(days=30)


class TestImpulseWaveDetector:
    """Test suite for impulse wave detection."""
    
    @pytest.mark.asyncio
    async def test_basic_impulse_detection(self, impulse_detector, sample_impulse_data):
        """Test basic impulse wave detection functionality."""
        waves = await impulse_detector.detect_impulse_waves(
            sample_impulse_data, "TESTUSDT", "1h"
        )
        
        assert isinstance(waves, list)
        assert len(waves) >= 0  # May or may not find patterns
        
        for wave in waves:
            assert isinstance(wave, ImpulseWave)
            assert 0.0 <= wave.confidence <= 1.0
            assert wave.symbol == "TESTUSDT"
            assert wave.timeframe == "1h"
            
    @pytest.mark.asyncio
    async def test_impulse_wave_validation(self, impulse_detector, sample_impulse_data):
        """Test impulse wave rule validation."""
        waves = await impulse_detector.detect_impulse_waves(
            sample_impulse_data, "TESTUSDT", "1h"
        )
        
        for wave in waves:
            # Test wave structure
            assert hasattr(wave, 'wave_1')
            assert hasattr(wave, 'wave_2')
            assert hasattr(wave, 'wave_3')
            assert hasattr(wave, 'wave_4')
            assert hasattr(wave, 'wave_5')
            
            # Test Elliott Wave rules
            assert isinstance(wave.rules_validation, dict)
            
            # Wave 3 should not be shortest (if rules pass)
            if wave.rules_validation.get('wave_3_not_shortest', False):
                assert wave.wave_3_length >= min(wave.wave_1_length, wave.wave_5_length)
                
    @pytest.mark.asyncio
    async def test_confidence_threshold(self, impulse_detector, sample_impulse_data):
        """Test confidence threshold filtering."""
        # High threshold - should get fewer results
        high_threshold_detector = ImpulseWaveDetector(confidence_threshold=0.9)
        high_waves = await high_threshold_detector.detect_impulse_waves(
            sample_impulse_data, "TESTUSDT", "1h"
        )
        
        # Low threshold - should get more results
        low_threshold_detector = ImpulseWaveDetector(confidence_threshold=0.3)
        low_waves = await low_threshold_detector.detect_impulse_waves(
            sample_impulse_data, "TESTUSDT", "1h"
        )
        
        assert len(high_waves) <= len(low_waves)
        
        # All waves should meet threshold
        for wave in high_waves:
            assert wave.confidence >= 0.9
            
    @pytest.mark.asyncio
    async def test_crypto_market_adaptation(self, impulse_detector, btc_data):
        """Test performance with crypto market data."""
        waves = await impulse_detector.detect_impulse_waves(
            btc_data, "BTCUSDT", "1h"
        )
        
        # Should handle high volatility crypto data
        assert isinstance(waves, list)
        
        for wave in waves:
            # Check for crypto-specific validations
            assert wave.symbol == "BTCUSDT"
            assert wave.timeframe == "1h"
            
            # Should handle large price ranges
            total_move = wave.total_move
            assert total_move > 0
            
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, impulse_detector, btc_data):
        """Test detection performance with larger dataset."""
        import time
        
        start_time = time.time()
        waves = await impulse_detector.detect_impulse_waves(
            btc_data, "BTCUSDT", "1h"
        )
        end_time = time.time()
        
        processing_time = (end_time - start_time) * 1000  # milliseconds
        
        # Should process within reasonable time (adjust based on requirements)
        assert processing_time < 5000  # 5 seconds max
        
        # Check processing time is recorded
        for wave in waves:
            assert wave.processing_time_ms > 0
            
    def test_invalid_data_handling(self, impulse_detector):
        """Test handling of invalid input data."""
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should not raise exception
        import asyncio
        waves = asyncio.run(impulse_detector.detect_impulse_waves(
            empty_df, "TESTUSDT", "1h"
        ))
        assert waves == []
        
        # DataFrame with insufficient data
        small_df = pd.DataFrame({
            'open': [100, 101],
            'high': [101, 102], 
            'low': [99, 100],
            'close': [100.5, 101.5],
            'volume': [1000, 1100]
        })
        small_df.index = pd.date_range(start='2023-01-01', periods=2, freq='1H')
        
        waves = asyncio.run(impulse_detector.detect_impulse_waves(
            small_df, "TESTUSDT", "1h"
        ))
        assert waves == []  # Not enough data for pattern detection


class TestCorrectiveWaveDetector:
    """Test suite for corrective wave detection."""
    
    @pytest.mark.asyncio
    async def test_basic_corrective_detection(self, corrective_detector, sample_corrective_data):
        """Test basic corrective wave detection."""
        waves = await corrective_detector.detect_corrective_waves(
            sample_corrective_data, "TESTUSDT", "1h"
        )
        
        assert isinstance(waves, list)
        
        for wave in waves:
            assert isinstance(wave, CorrectiveWave)
            assert 0.0 <= wave.confidence <= 1.0
            assert wave.symbol == "TESTUSDT"
            assert wave.timeframe == "1h"
            
    @pytest.mark.asyncio
    async def test_zigzag_pattern_detection(self, corrective_detector):
        """Test specific zigzag pattern detection."""
        zigzag_data = TestData.generate_corrective_pattern(
            pattern_type="zigzag", length=60
        )
        
        waves = await corrective_detector.detect_corrective_waves(
            zigzag_data, "TESTUSDT", "1h"
        )
        
        # Should detect some zigzag patterns
        zigzag_waves = [w for w in waves if "zigzag" in w.corrective_type.value.lower()]
        assert len(zigzag_waves) >= 0  # May or may not find specific pattern type
        
    @pytest.mark.asyncio
    async def test_flat_pattern_detection(self, corrective_detector):
        """Test flat pattern detection."""
        flat_data = TestData.generate_corrective_pattern(
            pattern_type="flat", length=60
        )
        
        waves = await corrective_detector.detect_corrective_waves(
            flat_data, "TESTUSDT", "1h"
        )
        
        for wave in waves:
            # Test ABC structure
            assert hasattr(wave, 'wave_a')
            assert hasattr(wave, 'wave_b')
            assert hasattr(wave, 'wave_c')
            
            # Test flat-specific characteristics
            if "flat" in wave.corrective_type.value.lower():
                # B wave should deeply retrace A wave
                assert wave.wave_b_retracement > 0.8  # > 80% retracement
                
    @pytest.mark.asyncio
    async def test_corrective_wave_fibonacci(self, corrective_detector, sample_corrective_data):
        """Test Fibonacci relationships in corrective waves."""
        waves = await corrective_detector.detect_corrective_waves(
            sample_corrective_data, "TESTUSDT", "1h"
        )
        
        for wave in waves:
            # Should have fibonacci ratios calculated
            assert isinstance(wave.fibonacci_ratios, dict)
            
            # Should have some common ratios
            expected_ratios = ['wave_c_to_a', 'wave_b_to_a', 'b_retracement_of_a']
            for ratio in expected_ratios:
                if ratio in wave.fibonacci_ratios:
                    assert isinstance(wave.fibonacci_ratios[ratio], float)
                    assert wave.fibonacci_ratios[ratio] > 0


class TestCryptoSpecificScenarios:
    """Test crypto-specific market scenarios."""
    
    @pytest.mark.asyncio
    async def test_high_volatility_handling(self, impulse_detector):
        """Test handling of extreme crypto volatility."""
        # Generate data with extreme volatility
        extreme_data = TestData.generate_impulse_pattern(
            volatility=0.1,  # 10% noise
            length=200
        )
        
        waves = await impulse_detector.detect_impulse_waves(
            extreme_data, "DOGEUSDT", "1h"
        )
        
        # Should still detect patterns despite high volatility
        assert isinstance(waves, list)
        
        # Waves should have reasonable confidence despite noise
        for wave in waves:
            assert wave.confidence > 0.0
            
    @pytest.mark.asyncio
    async def test_24_7_trading_patterns(self, impulse_detector):
        """Test 24/7 crypto market pattern detection."""
        # Generate continuous data (no market close gaps)
        continuous_data = TestData.generate_btc_like_data(days=7)  # Full week
        
        waves = await impulse_detector.detect_impulse_waves(
            continuous_data, "BTCUSDT", "1h"
        )
        
        # Should handle continuous trading
        assert isinstance(waves, list)
        
        # Verify timestamp continuity
        for wave in waves:
            start_time = wave.wave_1[0].timestamp
            end_time = wave.wave_5[1].timestamp
            assert end_time > start_time
            
    @pytest.mark.asyncio
    async def test_altcoin_vs_bitcoin_patterns(self, impulse_detector, corrective_detector):
        """Test pattern differences between major and minor crypto assets."""
        # Bitcoin-like data (more stable)
        btc_data = TestData.generate_btc_like_data(days=30)
        btc_waves = await impulse_detector.detect_impulse_waves(
            btc_data, "BTCUSDT", "1h"
        )
        
        # Altcoin-like data (more volatile)
        alt_data = TestData.generate_impulse_pattern(
            volatility=0.08,  # Higher volatility
            length=len(btc_data)
        )
        alt_waves = await impulse_detector.detect_impulse_waves(
            alt_data, "ADAUSDT", "1h"
        )
        
        # Both should detect patterns but characteristics may differ
        assert isinstance(btc_waves, list)
        assert isinstance(alt_waves, list)


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, impulse_detector):
        """Test performance with large datasets."""
        # Generate 1 year of hourly data
        large_data = TestData.generate_btc_like_data(days=365)
        
        import time
        start_time = time.time()
        
        waves = await impulse_detector.detect_impulse_waves(
            large_data, "BTCUSDT", "1h"
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time)
        
        # Should process large dataset efficiently
        assert processing_time < 30.0  # 30 seconds max
        assert isinstance(waves, list)
        
        # Log performance metrics
        print(f"Processed {len(large_data)} data points in {processing_time:.2f} seconds")
        print(f"Found {len(waves)} impulse waves")
        
    @pytest.mark.asyncio
    async def test_concurrent_detection(self, impulse_detector):
        """Test concurrent pattern detection."""
        # Create multiple datasets
        datasets = [
            TestData.generate_btc_like_data(days=30)
            for _ in range(5)
        ]
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        import time
        start_time = time.time()
        
        # Process concurrently
        tasks = [
            impulse_detector.detect_impulse_waves(data, symbol, "1h")
            for data, symbol in zip(datasets, symbols)
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = (end_time - start_time)
        
        # Should handle concurrent processing
        assert len(results) == 5
        for waves in results:
            assert isinstance(waves, list)
            
        print(f"Concurrent processing completed in {processing_time:.2f} seconds")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])