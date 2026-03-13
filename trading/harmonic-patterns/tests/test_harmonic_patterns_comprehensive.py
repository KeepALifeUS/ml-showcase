"""
Comprehensive Test Suite for Harmonic Patterns Package.

Production-ready test suite for all harmonic patterns
with synthetic data, edge cases, and performance testing.

Author: ML Harmonic Patterns Contributors
Created: 2025-09-11
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from unittest.mock import Mock, patch
import warnings

# Import all patterns
from ml_harmonic_patterns.patterns import (
    GartleyPattern, BatPattern, ButterflyPattern,
    CrabPattern, SharkPattern, CypherPattern,
    PatternResult, SharkPatternResult,
    PatternType, PatternValidation, PatternPoint,
    FibonacciRatios, BatFibonacciRatios, ButterflyFibonacciRatios,
    CrabFibonacciRatios, SharkFibonacciRatios, CypherFibonacciRatios,
    analyze_pattern_performance, filter_patterns_by_quality
)


class TestDataGenerator:
    """Generator of synthetic OHLCV data for testing."""
    
    @staticmethod
    def generate_synthetic_ohlcv(
        start_price: float = 100.0,
        num_bars: int = 200,
        volatility: float = 0.02,
        trend: float = 0.0,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data."""
        np.random.seed(42)  # For test reproducibility
        
        # Generate base prices
        returns = np.random.normal(trend, volatility, num_bars)
        prices = start_price * np.cumprod(1 + returns)
        
        # Create OHLCV data
        data = []
        for i, price in enumerate(prices):
            daily_volatility = volatility * np.random.uniform(0.5, 2.0)
            high = price * (1 + np.random.uniform(0, daily_volatility))
            low = price * (1 - np.random.uniform(0, daily_volatility))
            close = price * (1 + np.random.uniform(-daily_volatility/2, daily_volatility/2))
            
            volume = np.random.uniform(1000, 10000) if include_volume else 0
            
            data.append({
                'open': price,
                'high': max(price, high, close),
                'low': min(price, low, close),
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.index = pd.date_range('2023-01-01', periods=num_bars, freq='1H')
        return df
    
    @staticmethod
    def generate_gartley_pattern_data() -> pd.DataFrame:
        """Generate data with ideal Gartley pattern."""
        # Create ideal points for Gartley
        pattern_points = [
            (0, 100.0),    # X
            (20, 120.0),   # A
            (40, 112.84),  # B (61.8% retracement of XA)
            (60, 118.0),   # C 
            (80, 115.28),  # D (78.6% retracement of XA)
            (100, 110.0)   # Continuation
        ]
        
        # Interpolate between points
        data = []
        for i in range(len(pattern_points) - 1):
            start_bar, start_price = pattern_points[i]
            end_bar, end_price = pattern_points[i + 1]
            
            bars_between = end_bar - start_bar
            for j in range(bars_between):
                ratio = j / bars_between
                interpolated_price = start_price + (end_price - start_price) * ratio
                
                # Add small noise
                noise = np.random.normal(0, 0.5)
                price = interpolated_price + noise
                
                data.append({
                    'open': price,
                    'high': price * 1.01,
                    'low': price * 0.99,
                    'close': price,
                    'volume': np.random.uniform(1000, 5000)
                })
        
        df = pd.DataFrame(data)
        df.index = pd.date_range('2023-01-01', periods=len(data), freq='1H')
        return df


class TestPatternDetectors:
    """Basic tests for all pattern detectors."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Fixture with synthetic data."""
        return TestDataGenerator.generate_synthetic_ohlcv()
    
    @pytest.fixture
    def gartley_data(self):
        """Fixture with data containing Gartley pattern."""
        return TestDataGenerator.generate_gartley_pattern_data()
    
    def test_gartley_pattern_initialization(self):
        """Test Gartley Pattern initialization."""
        detector = GartleyPattern(tolerance=0.05, min_confidence=0.75)
        
        assert detector.tolerance == 0.05
        assert detector.min_confidence == 0.75
        assert detector.enable_volume_analysis is True
        assert detector.enable_ml_scoring is True
        assert isinstance(detector.fib_ratios, FibonacciRatios)
    
    def test_bat_pattern_initialization(self):
        """Test Bat Pattern initialization."""
        detector = BatPattern(tolerance=0.05, min_confidence=0.70)
        
        assert detector.tolerance == 0.05
        assert detector.min_confidence == 0.70
        assert isinstance(detector.bat_fib_ratios, BatFibonacciRatios)
        # AD retracement should be 88.6% for Bat (differs from Gartley)
        assert detector.bat_fib_ratios.AD_RETRACEMENT == 0.886
    
    def test_butterfly_pattern_initialization(self):
        """Test Butterfly Pattern initialization."""
        detector = ButterflyPattern(tolerance=0.05, min_confidence=0.70)
        
        assert detector.tolerance == 0.05
        assert isinstance(detector.butterfly_fib_ratios, ButterflyFibonacciRatios)
        # AB retracement should be 78.6% for Butterfly
        assert detector.butterfly_fib_ratios.AB_RETRACEMENT == 0.786
        # AD extensions for Butterfly
        assert detector.butterfly_fib_ratios.AD_EXTENSION_1 == 1.272
        assert detector.butterfly_fib_ratios.AD_EXTENSION_2 == 1.618
    
    def test_crab_pattern_initialization(self):
        """Test Crab Pattern initialization."""
        detector = CrabPattern(tolerance=0.08, min_confidence=0.75)
        
        assert detector.tolerance == 0.08  # Increased tolerance for extreme patterns
        assert isinstance(detector.crab_fib_ratios, CrabFibonacciRatios)
        # CD extensions for Crab should be extreme
        assert detector.crab_fib_ratios.CD_MIN_EXTENSION == 2.240
        assert detector.crab_fib_ratios.CD_MAX_EXTENSION == 3.618
    
    def test_shark_pattern_initialization(self):
        """Test Shark Pattern initialization."""
        detector = SharkPattern(tolerance=0.06, min_confidence=0.75)
        
        assert detector.tolerance == 0.06
        assert isinstance(detector.shark_fib_ratios, SharkFibonacciRatios)
        # XA extension for Shark
        assert detector.shark_fib_ratios.XA_MIN_EXTENSION == 1.13
        assert detector.shark_fib_ratios.XA_MAX_EXTENSION == 1.618
    
    def test_cypher_pattern_initialization(self):
        """Test Cypher Pattern initialization."""
        detector = CypherPattern(tolerance=0.06, min_confidence=0.75)
        
        assert detector.tolerance == 0.06
        assert isinstance(detector.cypher_fib_ratios, CypherFibonacciRatios)
        # BC extension from XA for Cypher (unique characteristic)
        assert detector.cypher_fib_ratios.BC_MIN_EXTENSION == 1.13
        assert detector.cypher_fib_ratios.BC_MAX_EXTENSION == 1.414
        # CD retracement from XC
        assert detector.cypher_fib_ratios.CD_RETRACEMENT == 0.786
    
    def test_pattern_detection_with_insufficient_data(self, synthetic_data):
        """Test behavior with insufficient data."""
        short_data = synthetic_data.head(10)  # Only 10 bars
        
        detectors = [
            GartleyPattern(),
            BatPattern(), 
            ButterflyPattern(),
            CrabPattern(),
            SharkPattern(),
            CypherPattern()
        ]
        
        for detector in detectors:
            with pytest.raises(ValueError, match="Insufficient data"):
                detector.detect_patterns(short_data)
    
    def test_pattern_detection_with_missing_columns(self):
        """Test behavior with missing columns."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            # missing 'low' and 'close'
        })
        
        detector = GartleyPattern()
        with pytest.raises(ValueError, match="Missing required columns"):
            detector.detect_patterns(incomplete_data)
    
    def test_pattern_detection_with_nan_values(self, synthetic_data):
        """Test behavior with NaN values."""
        data_with_nan = synthetic_data.copy()
        data_with_nan.loc[data_with_nan.index[10], 'close'] = np.nan
        
        detector = GartleyPattern()
        # Should issue a warning and fill NaN values
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            patterns = detector.detect_patterns(data_with_nan)
            assert len(w) > 0
            assert "NaN values" in str(w[0].message)
    
    def test_fibonacci_ratios_immutability(self):
        """Test Fibonacci ratios immutability."""
        ratios = FibonacciRatios()
        
        # Attempting to change values should raise an error
        with pytest.raises(AttributeError):
            ratios.AB_RETRACEMENT = 0.5
    
    def test_pattern_point_creation(self):
        """Test pattern point creation."""
        point = PatternPoint(index=10, price=100.5, timestamp=pd.Timestamp.now())
        
        assert point.index == 10
        assert point.price == 100.5
        assert isinstance(point.timestamp, pd.Timestamp)
    
    def test_cache_functionality(self, synthetic_data):
        """Test result caching."""
        detector = GartleyPattern()
        
        # First call
        patterns1 = detector.detect_patterns(synthetic_data, symbol="TEST", timeframe="1h")
        
        # Second call should use cache
        with patch.object(detector, '_find_pivot_points') as mock_pivot:
            patterns2 = detector.detect_patterns(synthetic_data, symbol="TEST", timeframe="1h")
            # Method should not be called since cache is used
            mock_pivot.assert_not_called()
        
        assert patterns1 == patterns2
    
    def test_cache_clearing(self, synthetic_data):
        """Test cache clearing."""
        detector = GartleyPattern()
        
        # Create cache
        patterns = detector.detect_patterns(synthetic_data, symbol="TEST", timeframe="1h")
        assert detector.get_cache_stats()['patterns_cache_size'] > 0
        
        # Clear cache
        detector.clear_cache()
        stats = detector.get_cache_stats()
        assert stats['patterns_cache_size'] == 0
        assert stats['zigzag_cache_size'] == 0


class TestPatternSpecificBehavior:
    """Tests for specific behavior of each pattern."""
    
    @pytest.fixture
    def synthetic_data(self):
        return TestDataGenerator.generate_synthetic_ohlcv()
    
    def test_gartley_specific_validation(self, synthetic_data):
        """Test Gartley-specific validation."""
        detector = GartleyPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            # AD ratio should be close to 78.6% for a valid Gartley
            assert isinstance(pattern.ad_ratio, float)
            if pattern.validation_status == PatternValidation.VALID:
                # Check that AD ratio is within reasonable limits for Gartley
                assert 0.5 <= pattern.ad_ratio <= 1.2
    
    def test_bat_specific_validation(self, synthetic_data):
        """Test Bat-specific validation."""
        detector = BatPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            if pattern.validation_status == PatternValidation.VALID:
                # AB must be in range 38.2% - 50% for Bat
                assert 0.3 <= pattern.ab_ratio <= 0.6
    
    def test_butterfly_ad_extensions(self, synthetic_data):
        """Test AD extensions for Butterfly pattern."""
        detector = ButterflyPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            if pattern.validation_status == PatternValidation.VALID:
                # AD should be an extension (> 1.0) for Butterfly
                assert pattern.ad_ratio > 1.0
    
    def test_crab_extreme_extensions(self, synthetic_data):
        """Test extreme extensions for Crab pattern."""
        detector = CrabPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            if pattern.validation_status == PatternValidation.VALID:
                # CD ratio should be extreme (> 2.0) for Crab
                assert pattern.cd_ratio > 2.0
    
    def test_shark_prz_analysis(self, synthetic_data):
        """Test PRZ analysis for Shark pattern."""
        detector = SharkPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            if isinstance(pattern, SharkPatternResult):
                # Shark should have PRZ confluence
                assert hasattr(pattern, 'prz_confluence')
                assert 0.0 <= pattern.prz_confluence <= 1.0
                
                # Test PRZ analysis
                prz_analysis = detector.analyze_prz_quality(pattern)
                assert 'prz_confluence_score' in prz_analysis
                assert 'prz_quality_rating' in prz_analysis
    
    def test_cypher_bc_extension_structure(self, synthetic_data):
        """Test BC extension structure for Cypher pattern."""
        detector = CypherPattern()
        patterns = detector.detect_patterns(synthetic_data)
        
        for pattern in patterns:
            if pattern.validation_status == PatternValidation.VALID:
                # BC ratio in Cypher represents extension from XA
                assert pattern.bc_ratio > 1.0  # Should be an extension


class TestTradingSignals:
    """Tests for trading signal generation."""
    
    @pytest.fixture
    def mock_pattern(self):
        """Mock pattern for testing."""
        return PatternResult(
            point_x=PatternPoint(0, 100.0),
            point_a=PatternPoint(20, 120.0),
            point_b=PatternPoint(40, 112.84),
            point_c=PatternPoint(60, 118.0),
            point_d=PatternPoint(80, 115.28),
            pattern_type=PatternType.BULLISH,
            validation_status=PatternValidation.VALID,
            confidence_score=0.85,
            ab_ratio=0.618,
            bc_ratio=0.5,
            cd_ratio=1.4,
            ad_ratio=0.786,
            entry_price=115.28,
            stop_loss=98.0,
            take_profit_1=118.0,
            take_profit_2=122.0,
            take_profit_3=125.0,
            risk_reward_ratio=2.1,
            max_risk_percent=15.0,
            pattern_strength=0.78,
            fibonacci_confluence=0.82,
            volume_confirmation=0.65,
            completion_time=pd.Timestamp.now()
        )
    
    def test_entry_signals_generation(self, mock_pattern):
        """Test entry signal generation."""
        detector = GartleyPattern()
        signals = detector.get_entry_signals(mock_pattern)
        
        # Check main signal components
        assert 'action' in signals
        assert signals['action'] in ['BUY', 'SELL']
        assert 'entry_price' in signals
        assert 'stop_loss' in signals
        assert 'take_profit_levels' in signals
        assert 'confidence' in signals
        assert 'risk_reward_ratio' in signals
        assert 'entry_conditions' in signals
        assert 'timing' in signals
        
        # Check entry conditions
        entry_conditions = signals['entry_conditions']
        assert 'min_confidence_met' in entry_conditions
        assert 'risk_reward_acceptable' in entry_conditions
        assert 'pattern_valid' in entry_conditions
    
    def test_position_sizing(self, mock_pattern):
        """Test position sizing calculation."""
        detector = GartleyPattern()
        position_info = detector.calculate_position_size(
            mock_pattern, 
            account_balance=10000.0, 
            max_risk_percent=2.0
        )
        
        assert 'position_size' in position_info
        assert 'risk_amount' in position_info
        assert 'leverage' in position_info
        assert 'max_loss' in position_info
        
        # Risk amount should be 2% of balance
        assert position_info['risk_amount'] == 200.0
        assert position_info['max_loss'] == 200.0
    
    def test_visualization_data_preparation(self, mock_pattern):
        """Test visualization data preparation."""
        detector = GartleyPattern()
        viz_data = detector.prepare_visualization_data(mock_pattern)
        
        assert 'pattern_points' in viz_data
        assert 'pattern_lines' in viz_data
        assert 'fibonacci_levels' in viz_data
        assert 'trading_levels' in viz_data
        assert 'metadata' in viz_data
        assert 'chart_title' in viz_data
        
        # Check pattern points
        pattern_points = viz_data['pattern_points']
        assert len(pattern_points) == 5  # X, A, B, C, D
        
        # Check trading levels
        trading_levels = viz_data['trading_levels']
        level_names = [level['name'] for level in trading_levels]
        assert 'Entry' in level_names
        assert 'Stop Loss' in level_names
        assert 'TP1' in level_names


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.fixture
    def mock_patterns(self):
        """Mock patterns for testing."""
        patterns = []
        for i in range(5):
            pattern = PatternResult(
                point_x=PatternPoint(0, 100.0),
                point_a=PatternPoint(20, 120.0),
                point_b=PatternPoint(40, 112.84),
                point_c=PatternPoint(60, 118.0),
                point_d=PatternPoint(80, 115.28),
                pattern_type=PatternType.BULLISH,
                validation_status=PatternValidation.VALID,
                confidence_score=0.7 + i * 0.05,  # Varying confidence scores
                ab_ratio=0.618,
                bc_ratio=0.5,
                cd_ratio=1.4,
                ad_ratio=0.786,
                entry_price=115.28,
                stop_loss=98.0,
                take_profit_1=118.0,
                take_profit_2=122.0,
                take_profit_3=125.0,
                risk_reward_ratio=1.5 + i * 0.2,  # Varying R/R
                max_risk_percent=15.0 - i,  # Varying risk
                pattern_strength=0.78,
                fibonacci_confluence=0.82,
                volume_confirmation=0.65,
                completion_time=pd.Timestamp.now()
            )
            patterns.append(pattern)
        return patterns
    
    def test_pattern_filtering(self, mock_patterns):
        """Test pattern quality filtering."""
        # Filter with high requirements
        filtered = filter_patterns_by_quality(
            mock_patterns,
            min_confidence=0.80,
            min_risk_reward=2.0,
            max_risk_percent=12.0
        )
        
        # Only high-quality patterns should remain
        assert len(filtered) < len(mock_patterns)
        
        for pattern in filtered:
            assert pattern.confidence_score >= 0.80
            assert pattern.risk_reward_ratio >= 2.0
            assert pattern.max_risk_percent <= 12.0
    
    def test_performance_analysis_empty_patterns(self):
        """Test performance analysis with empty list."""
        empty_patterns = []
        mock_prices = pd.Series([100, 101, 102, 103])
        
        performance = analyze_pattern_performance(empty_patterns, mock_prices)
        assert performance == {}
    
    def test_performance_analysis_with_patterns(self, mock_patterns):
        """Test performance analysis with patterns."""
        # Create mock prices
        mock_prices = pd.Series(
            [115, 118, 120, 122, 125, 128, 130],  # Upward trend
            index=range(80, 87)  # Start from point D
        )
        
        performance = analyze_pattern_performance(mock_patterns, mock_prices, lookforward_periods=5)
        
        assert 'total_patterns' in performance
        assert 'successful_patterns' in performance
        assert 'success_rate' in performance
        assert 'tp1_hit_rate' in performance
        assert 'average_confidence' in performance
        
        assert performance['total_patterns'] == len(mock_patterns)


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_division_protection(self):
        """Test division by zero protection."""
        # Create data where all prices are identical
        flat_data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [100.0] * 50,
            'low': [100.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000] * 50
        })
        
        detector = GartleyPattern()
        # Should not crash with error, should return empty list
        patterns = detector.detect_patterns(flat_data)
        assert patterns == []
    
    def test_extreme_volatility_data(self):
        """Test with extremely volatile data."""
        volatile_data = TestDataGenerator.generate_synthetic_ohlcv(
            volatility=0.5  # 50% volatility
        )
        
        detector = GartleyPattern()
        # Should correctly handle even extreme data
        patterns = detector.detect_patterns(volatile_data)
        assert isinstance(patterns, list)
    
    def test_very_low_confidence_patterns(self):
        """Test with very low confidence threshold."""
        detector = GartleyPattern(min_confidence=0.1)  # Very low threshold
        data = TestDataGenerator.generate_synthetic_ohlcv()
        
        patterns = detector.detect_patterns(data)
        # May find more patterns, but all should be valid
        for pattern in patterns:
            assert pattern.confidence_score >= 0.1
    
    def test_memory_efficiency_large_dataset(self):
        """Test memory efficiency on large data."""
        large_data = TestDataGenerator.generate_synthetic_ohlcv(num_bars=2000)
        
        detector = GartleyPattern()
        patterns = detector.detect_patterns(large_data)
        
        # Check that cache does not grow uncontrollably
        cache_stats = detector.get_cache_stats()
        assert cache_stats['patterns_cache_size'] <= 10  # Reasonable limit
    
    def test_thread_safety_simulation(self, synthetic_data):
        """Thread safety simulation (basic check)."""
        detector = GartleyPattern()
        
        # Simulate parallel calls
        results = []
        for i in range(5):
            patterns = detector.detect_patterns(synthetic_data, symbol=f"TEST{i}")
            results.append(patterns)
        
        # All results should be correct
        for patterns in results:
            assert isinstance(patterns, list)


class TestPerformance:
    """Performance tests."""
    
    def test_detection_speed(self):
        """Test pattern detection speed."""
        import time
        
        data = TestDataGenerator.generate_synthetic_ohlcv(num_bars=500)
        detector = GartleyPattern()
        
        start_time = time.time()
        patterns = detector.detect_patterns(data)
        end_time = time.time()
        
        detection_time = end_time - start_time
        
        # Detection should complete in reasonable time (< 5 seconds for 500 bars)
        assert detection_time < 5.0, f"Detection took {detection_time:.2f}s, which is too slow"
    
    def test_memory_usage_stability(self):
        """Test memory usage stability."""
        detector = GartleyPattern()
        
        # Multiple detections to check for memory leaks
        for i in range(10):
            data = TestDataGenerator.generate_synthetic_ohlcv(num_bars=100)
            patterns = detector.detect_patterns(data, symbol=f"TEST{i}")
            
            # Clear cache periodically
            if i % 3 == 0:
                detector.clear_cache()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])