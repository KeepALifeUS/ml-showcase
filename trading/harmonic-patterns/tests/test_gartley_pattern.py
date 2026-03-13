"""
Comprehensive Tests for Gartley Pattern Detection.

Enterprise-grade test suite with extensive coverage,
performance testing, edge cases, and integration validation for production-ready
harmonic pattern detection system.

Test Categories:
1. Unit Tests - Core functionality validation
2. Integration Tests - End-to-end pattern detection
3. Performance Tests - Scalability and memory usage
4. Edge Cases - Error handling and boundary conditions
5. Validation Tests - Fibonacci accuracy and confidence scoring

Author: ML Harmonic Patterns Contributors
Created: 2025-09-11
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from typing import List, Dict
from unittest.mock import patch, MagicMock
import logging
from datetime import datetime, timedelta

# Import our module
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from patterns.gartley_pattern import (
    GartleyPattern,
    PatternResult,
    PatternType,
    PatternValidation,
    FibonacciRatios,
    PatternPoint,
    analyze_pattern_performance,
    filter_patterns_by_quality
)


class TestGartleyPatternInit:
    """Tests for GartleyPattern class initialization."""
    
    def test_default_initialization(self):
        """Test creating detector with default parameters."""
        detector = GartleyPattern()
        
        assert detector.tolerance == 0.05
        assert detector.min_confidence == 0.70
        assert detector.enable_volume_analysis == True
        assert detector.enable_ml_scoring == True
        assert detector.min_pattern_bars == 20
        assert detector.max_pattern_bars == 200
        assert isinstance(detector.fib_ratios, FibonacciRatios)
    
    def test_custom_initialization(self):
        """Test creating detector with custom parameters."""
        detector = GartleyPattern(
            tolerance=0.03,
            min_confidence=0.80,
            enable_volume_analysis=False,
            enable_ml_scoring=False,
            min_pattern_bars=30,
            max_pattern_bars=150
        )
        
        assert detector.tolerance == 0.03
        assert detector.min_confidence == 0.80
        assert detector.enable_volume_analysis == False
        assert detector.enable_ml_scoring == False
        assert detector.min_pattern_bars == 30
        assert detector.max_pattern_bars == 150


class TestFibonacciRatios:
    """Tests for Fibonacci ratios of the Gartley pattern."""
    
    def test_fibonacci_ratios_constants(self):
        """Test correctness of Fibonacci constants."""
        fib = FibonacciRatios()
        
        # Main Gartley ratios
        assert fib.AB_RETRACEMENT == 0.618
        assert fib.BC_MIN_RETRACEMENT == 0.382
        assert fib.BC_MAX_RETRACEMENT == 0.886
        assert fib.CD_MIN_EXTENSION == 1.272
        assert fib.CD_MAX_EXTENSION == 1.618
        assert fib.AD_RETRACEMENT == 0.786
        assert fib.TOLERANCE == 0.05
    
    def test_fibonacci_ratios_immutability(self):
        """Test immutability of Fibonacci ratios (frozen dataclass)."""
        fib = FibonacciRatios()
        
        with pytest.raises(AttributeError):
            fib.AB_RETRACEMENT = 0.5  # Should raise an error


class TestPatternPoint:
    """Tests for PatternPoint NamedTuple."""
    
    def test_pattern_point_creation(self):
        """Test PatternPoint creation."""
        point = PatternPoint(
            index=10,
            price=50000.0,
            timestamp=pd.Timestamp('2025-01-01'),
            volume=1000.0
        )
        
        assert point.index == 10
        assert point.price == 50000.0
        assert point.timestamp == pd.Timestamp('2025-01-01')
        assert point.volume == 1000.0
    
    def test_pattern_point_minimal(self):
        """Test PatternPoint creation with minimal parameters."""
        point = PatternPoint(index=5, price=45000.0)
        
        assert point.index == 5
        assert point.price == 45000.0
        assert point.timestamp is None
        assert point.volume is None


class TestDataValidation:
    """Tests for input data validation."""
    
    def test_valid_ohlcv_data(self):
        """Test validation of correct OHLCV data."""
        data = self._create_sample_ohlcv_data(100)
        detector = GartleyPattern()
        
        # Should not raise an exception
        detector._validate_input_data(data)
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            # Missing 'low' and 'close'
        })
        
        detector = GartleyPattern()
        
        with pytest.raises(ValueError, match="Missing required columns"):
            detector._validate_input_data(data)
    
    def test_insufficient_data_length(self):
        """Test handling of insufficient data."""
        data = self._create_sample_ohlcv_data(10)  # Less than min_pattern_bars
        detector = GartleyPattern(min_pattern_bars=20)
        
        with pytest.raises(ValueError, match="Insufficient data"):
            detector._validate_input_data(data)
    
    def test_nan_values_handling(self):
        """Test handling of NaN values in data."""
        data = self._create_sample_ohlcv_data(50)
        data.loc[10, 'close'] = np.nan
        data.loc[20, 'high'] = np.nan
        
        detector = GartleyPattern()
        
        # Should issue a warning, but not an exception
        with pytest.warns(UserWarning):
            detector._validate_input_data(data)
    
    def _create_sample_ohlcv_data(self, length: int) -> pd.DataFrame:
        """Create test OHLCV data."""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic crypto price data
        base_price = 50000.0
        dates = pd.date_range(start='2025-01-01', periods=length, freq='1H')
        
        # Random walk with some volatility
        price_changes = np.random.normal(0, 0.02, length)  # 2% std
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1000.0))  # Minimum price floor
        
        # Create OHLC based on prices
        ohlc_data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC
            volatility = abs(np.random.normal(0, 0.01))
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = prices[i-1] if i > 0 else price
            close = price
            volume = np.random.uniform(100, 1000)
            
            ohlc_data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(ohlc_data, index=dates)


class TestPivotPointDetection:
    """Tests for pivot point detection (ZigZag)."""
    
    def test_find_pivot_points_basic(self):
        """Test basic pivot point detection functionality."""
        data = self._create_trending_data()
        detector = GartleyPattern()
        
        pivot_points = detector._find_pivot_points(data)
        
        # Should find at least some pivot points
        assert len(pivot_points) >= 3
        
        # All pivot points should be PatternPoint objects
        for point in pivot_points:
            assert isinstance(point, PatternPoint)
            assert 0 <= point.index < len(data)
            assert point.price > 0
    
    def test_pivot_points_alternating(self):
        """Test that pivot points alternate (high->low->high->low)."""
        data = self._create_zigzag_data()
        detector = GartleyPattern()
        
        pivot_points = detector._find_pivot_points(data)
        
        if len(pivot_points) >= 3:
            # Check that points alternate between high and low
            for i in range(1, len(pivot_points) - 1):
                prev_point = pivot_points[i-1]
                current_point = pivot_points[i]
                next_point = pivot_points[i+1]
                
                # Current point should be either above or below neighbors
                is_peak = (current_point.price > prev_point.price and 
                          current_point.price > next_point.price)
                is_trough = (current_point.price < prev_point.price and 
                           current_point.price < next_point.price)
                
                assert is_peak or is_trough, f"Point {i} is not a proper pivot"
    
    def test_empty_pivot_points(self):
        """Test handling when no pivot points are found."""
        # Create flat data without significant movements
        data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [100.1] * 50,
            'low': [99.9] * 50,
            'close': [100.0] * 50,
            'volume': [1000] * 50
        })
        
        detector = GartleyPattern()
        pivot_points = detector._find_pivot_points(data)
        
        # May be empty or have very few points
        assert isinstance(pivot_points, list)
    
    def _create_trending_data(self) -> pd.DataFrame:
        """Create trending data for pivot point testing."""
        length = 100
        dates = pd.date_range(start='2025-01-01', periods=length, freq='1H')
        
        # Create uptrend with pullbacks
        trend = np.linspace(40000, 60000, length)
        noise = np.sin(np.linspace(0, 4*np.pi, length)) * 2000  # Sinusoidal oscillations
        prices = trend + noise
        
        ohlc_data = []
        for i, price in enumerate(prices):
            ohlc_data.append({
                'open': price * 0.999,
                'high': price * 1.002,
                'low': price * 0.998,
                'close': price,
                'volume': 1000
            })
        
        return pd.DataFrame(ohlc_data, index=dates)
    
    def _create_zigzag_data(self) -> pd.DataFrame:
        """Create ZigZag data for testing."""
        # Clear high/low alternating pattern
        prices = [40000, 45000, 42000, 48000, 44000, 50000, 46000, 52000]
        length = len(prices) * 10  # Stretch each point
        
        extended_prices = []
        for price in prices:
            extended_prices.extend([price] * 10)
        
        dates = pd.date_range(start='2025-01-01', periods=length, freq='1H')
        
        ohlc_data = []
        for price in extended_prices:
            ohlc_data.append({
                'open': price,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': 1000
            })
        
        return pd.DataFrame(ohlc_data, index=dates)


class TestGartleyGeometry:
    """Tests for Gartley pattern geometric validation."""
    
    def test_valid_bullish_gartley_geometry(self):
        """Test validation of correct bullish Gartley geometry."""
        detector = GartleyPattern()
        
        # Create correct bullish Gartley structure: X(low)->A(high)->B(low)->C(high)->D(low)
        x = PatternPoint(index=0, price=40000)   # Low
        a = PatternPoint(index=10, price=50000)  # High
        b = PatternPoint(index=20, price=43000)  # Low (retracement)
        c = PatternPoint(index=30, price=47000)  # High (correction)
        d = PatternPoint(index=40, price=42000)  # Low (completion, above X)
        
        is_valid = detector._is_valid_gartley_geometry(x, a, b, c, d)
        assert is_valid == True
    
    def test_valid_bearish_gartley_geometry(self):
        """Test validation of correct bearish Gartley geometry."""
        detector = GartleyPattern()
        
        # Create correct bearish Gartley structure: X(high)->A(low)->B(high)->C(low)->D(high)
        x = PatternPoint(index=0, price=50000)   # High
        a = PatternPoint(index=10, price=40000)  # Low
        b = PatternPoint(index=20, price=47000)  # High (retracement)
        c = PatternPoint(index=30, price=43000)  # Low (correction)
        d = PatternPoint(index=40, price=48000)  # High (completion, below X)
        
        is_valid = detector._is_valid_gartley_geometry(x, a, b, c, d)
        assert is_valid == True
    
    def test_invalid_gartley_geometry(self):
        """Test rejection of incorrect geometry."""
        detector = GartleyPattern()
        
        # Invalid structure - all points are increasing
        x = PatternPoint(index=0, price=40000)
        a = PatternPoint(index=10, price=41000)
        b = PatternPoint(index=20, price=42000)
        c = PatternPoint(index=30, price=43000)
        d = PatternPoint(index=40, price=44000)
        
        is_valid = detector._is_valid_gartley_geometry(x, a, b, c, d)
        assert is_valid == False
    
    def test_geometry_with_invalid_d_level(self):
        """Test rejection of pattern with incorrect D level."""
        detector = GartleyPattern()
        
        # D below X in bullish pattern (invalid)
        x = PatternPoint(index=0, price=40000)
        a = PatternPoint(index=10, price=50000)
        b = PatternPoint(index=20, price=43000)
        c = PatternPoint(index=30, price=47000)
        d = PatternPoint(index=40, price=38000)  # Below X
        
        is_valid = detector._is_valid_gartley_geometry(x, a, b, c, d)
        assert is_valid == False


class TestFibonacciValidation:
    """Tests for Fibonacci ratio validation."""
    
    def test_perfect_fibonacci_ratios(self):
        """Test pattern with ideal Fibonacci ratios."""
        detector = GartleyPattern(tolerance=0.01)
        
        # Create pattern with exact Fibonacci ratios
        x = PatternPoint(index=0, price=1000.0)
        a = PatternPoint(index=10, price=1500.0)  # XA = 500
        b = PatternPoint(index=20, price=1191.0)  # AB = 309 (61.8% of XA)
        c = PatternPoint(index=30, price=1309.0)  # BC = 118 (38.2% of AB)
        d = PatternPoint(index=40, price=1393.0)  # AD = 393 (78.6% of XA)
        
        data = self._create_pattern_data([x, a, b, c, d])
        pattern_result = detector._validate_and_score_pattern(
            (x, a, b, c, d), data, "BTCUSDT", "1h"
        )
        
        assert pattern_result is not None
        assert pattern_result.validation_status == PatternValidation.VALID
        assert pattern_result.confidence_score > 0.8
        
        # Check Fibonacci ratios
        assert abs(pattern_result.ab_ratio - 0.618) < 0.05
        assert abs(pattern_result.ad_ratio - 0.786) < 0.05
    
    def test_marginal_fibonacci_ratios(self):
        """Test pattern with acceptable but not ideal ratios."""
        detector = GartleyPattern(tolerance=0.1)
        
        # Create pattern with acceptable ratios
        x = PatternPoint(index=0, price=1000.0)
        a = PatternPoint(index=10, price=1500.0)
        b = PatternPoint(index=20, price=1200.0)  # ~60% instead of 61.8%
        c = PatternPoint(index=30, price=1320.0)  # ~40% of AB
        d = PatternPoint(index=40, price=1400.0)  # ~80% of XA
        
        data = self._create_pattern_data([x, a, b, c, d])
        pattern_result = detector._validate_and_score_pattern(
            (x, a, b, c, d), data, "BTCUSDT", "1h"
        )
        
        assert pattern_result is not None
        # May be VALID or MARGINAL depending on scoring
        assert pattern_result.validation_status in [PatternValidation.VALID, PatternValidation.MARGINAL]
    
    def test_invalid_fibonacci_ratios(self):
        """Test pattern with incorrect Fibonacci ratios."""
        detector = GartleyPattern(tolerance=0.05)
        
        # Create pattern with poor ratios
        x = PatternPoint(index=0, price=1000.0)
        a = PatternPoint(index=10, price=1500.0)
        b = PatternPoint(index=20, price=1400.0)  # Only 20% retracement
        c = PatternPoint(index=30, price=1450.0)
        d = PatternPoint(index=40, price=1300.0)  # 40% retracement instead of 78.6%
        
        data = self._create_pattern_data([x, a, b, c, d])
        pattern_result = detector._validate_and_score_pattern(
            (x, a, b, c, d), data, "BTCUSDT", "1h"
        )
        
        # Should be rejected or have very low confidence
        assert pattern_result is None or pattern_result.confidence_score < 0.3
    
    def _create_pattern_data(self, points: List[PatternPoint]) -> pd.DataFrame:
        """Create OHLCV data from pattern points."""
        max_index = max(point.index for point in points) + 10
        dates = pd.date_range(start='2025-01-01', periods=max_index, freq='1H')
        
        # Interpolate prices between points
        indices = [point.index for point in points]
        prices = [point.price for point in points]
        
        all_prices = np.interp(range(max_index), indices, prices)
        
        ohlc_data = []
        for i, price in enumerate(all_prices):
            ohlc_data.append({
                'open': price * 0.999,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': 1000
            })
        
        return pd.DataFrame(ohlc_data, index=dates[:max_index])


class TestPatternDetection:
    """Integration tests for full pattern detection."""
    
    def test_detect_patterns_with_valid_data(self):
        """Test pattern detection with valid data."""
        detector = GartleyPattern(min_confidence=0.5)  # Low threshold for tests
        data = self._create_realistic_crypto_data_with_pattern()
        
        patterns = detector.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
        
        # Should find at least one pattern or empty list (not an error)
        assert isinstance(patterns, list)
        
        # If patterns are found, check their structure
        for pattern in patterns:
            assert isinstance(pattern, PatternResult)
            assert pattern.pattern_type in [PatternType.BULLISH, PatternType.BEARISH]
            assert pattern.validation_status == PatternValidation.VALID
            assert pattern.confidence_score >= detector.min_confidence
            assert pattern.symbol == "BTCUSDT"
            assert pattern.timeframe == "1h"
    
    def test_detect_patterns_insufficient_data(self):
        """Test detection with insufficient data."""
        detector = GartleyPattern()
        data = TestDataValidation()._create_sample_ohlcv_data(10)  # Too little data
        
        with pytest.raises(ValueError):
            detector.detect_patterns(data)
    
    def test_detect_patterns_caching(self):
        """Test detection result caching."""
        detector = GartleyPattern()
        data = TestDataValidation()._create_sample_ohlcv_data(100)
        
        # First call
        patterns1 = detector.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
        cache_stats1 = detector.get_cache_stats()
        
        # Second call with the same data
        patterns2 = detector.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
        cache_stats2 = detector.get_cache_stats()
        
        # Results should be identical
        assert len(patterns1) == len(patterns2)
        
        # Cache should be used
        assert cache_stats2['patterns_cache_size'] >= cache_stats1['patterns_cache_size']
    
    def test_clear_cache(self):
        """Test cache clearing."""
        detector = GartleyPattern()
        data = TestDataValidation()._create_sample_ohlcv_data(100)
        
        # Create cached data
        detector.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
        
        # Check that cache is not empty
        cache_stats_before = detector.get_cache_stats()
        assert cache_stats_before['patterns_cache_size'] > 0
        
        # Clear cache
        detector.clear_cache()
        
        # Check that cache is cleared
        cache_stats_after = detector.get_cache_stats()
        assert cache_stats_after['patterns_cache_size'] == 0
        assert cache_stats_after['zigzag_cache_size'] == 0
    
    def _create_realistic_crypto_data_with_pattern(self) -> pd.DataFrame:
        """Create realistic crypto data with embedded Gartley pattern."""
        length = 200
        dates = pd.date_range(start='2025-01-01', periods=length, freq='1H')
        
        # Base trend
        base_trend = np.linspace(45000, 55000, length)
        
        # Add Gartley-like structure in the middle
        gartley_start = 50
        gartley_pattern = np.array([
            0,      # X
            5000,   # A (up move)
            -3000,  # B (61.8% retracement)
            2000,   # C (partial recovery)
            -4000   # D (78.6% retracement from XA)
        ])
        
        # Interpolate pattern over time
        pattern_indices = np.linspace(gartley_start, gartley_start + 80, len(gartley_pattern))
        pattern_overlay = np.interp(range(length), pattern_indices, gartley_pattern)
        
        # Combine trend and pattern
        prices = base_trend + pattern_overlay
        
        # Add realistic noise
        noise = np.random.normal(0, 200, length)
        prices += noise
        
        # Create OHLCV
        ohlc_data = []
        for i, price in enumerate(prices):
            volatility = abs(np.random.normal(0, 100))
            ohlc_data.append({
                'open': max(price - volatility/2, 1000),
                'high': price + volatility,
                'low': max(price - volatility, 1000),
                'close': price,
                'volume': np.random.uniform(500, 2000)
            })
        
        return pd.DataFrame(ohlc_data, index=dates)


class TestTradingLevels:
    """Tests for trading level calculation."""
    
    def test_bullish_trading_levels(self):
        """Test trading level calculation for bullish pattern."""
        detector = GartleyPattern()
        
        # Bullish Gartley pattern
        x = PatternPoint(index=0, price=40000)
        a = PatternPoint(index=10, price=50000)
        b = PatternPoint(index=20, price=43000)
        c = PatternPoint(index=30, price=47000)
        d = PatternPoint(index=40, price=42000)
        
        entry, stop_loss, (tp1, tp2, tp3) = detector._calculate_trading_levels(
            (x, a, b, c, d), PatternType.BULLISH
        )
        
        # Entry should be at point D
        assert entry == d.price
        
        # Stop loss should be below X for bullish
        assert stop_loss < x.price
        
        # Take profits should be above entry
        assert tp1 > entry
        assert tp2 > tp1
        assert tp3 > tp2
    
    def test_bearish_trading_levels(self):
        """Test trading level calculation for bearish pattern."""
        detector = GartleyPattern()
        
        # Bearish Gartley pattern
        x = PatternPoint(index=0, price=50000)
        a = PatternPoint(index=10, price=40000)
        b = PatternPoint(index=20, price=47000)
        c = PatternPoint(index=30, price=43000)
        d = PatternPoint(index=40, price=48000)
        
        entry, stop_loss, (tp1, tp2, tp3) = detector._calculate_trading_levels(
            (x, a, b, c, d), PatternType.BEARISH
        )
        
        # Entry should be at point D
        assert entry == d.price
        
        # Stop loss should be above X for bearish
        assert stop_loss > x.price
        
        # Take profits should be below entry
        assert tp1 < entry
        assert tp2 < tp1
        assert tp3 < tp2
    
    def test_risk_reward_calculation(self):
        """Test Risk/Reward ratio calculation."""
        detector = GartleyPattern()
        
        entry_price = 45000
        stop_loss = 42000  # Risk = 3000
        take_profit = 51000  # Reward = 6000
        
        rr_ratio = detector._calculate_risk_reward_ratio(entry_price, stop_loss, take_profit)
        
        expected_ratio = 6000 / 3000  # 2.0
        assert abs(rr_ratio - expected_ratio) < 0.01
    
    def test_zero_risk_handling(self):
        """Test zero risk handling."""
        detector = GartleyPattern()
        
        # Case where entry == stop_loss (zero risk)
        rr_ratio = detector._calculate_risk_reward_ratio(45000, 45000, 48000)
        
        assert rr_ratio == 0.0


class TestEntrySignals:
    """Tests for trading signal generation."""
    
    def test_bullish_entry_signals(self):
        """Test bullish entry signal generation."""
        detector = GartleyPattern()
        
        # Create a quality bullish pattern
        pattern = self._create_sample_pattern_result(PatternType.BULLISH, confidence=0.85)
        
        signals = detector.get_entry_signals(pattern)
        
        assert signals['action'] == 'BUY'
        assert signals['entry_price'] == pattern.entry_price
        assert signals['confidence'] == pattern.confidence_score
        assert len(signals['take_profit_levels']) == 3
        assert signals['entry_conditions']['min_confidence_met'] == True
        assert signals['timing']['immediate'] == True  # High confidence
    
    def test_bearish_entry_signals(self):
        """Test bearish entry signal generation."""
        detector = GartleyPattern()
        
        pattern = self._create_sample_pattern_result(PatternType.BEARISH, confidence=0.75)
        
        signals = detector.get_entry_signals(pattern)
        
        assert signals['action'] == 'SELL'
        assert signals['entry_price'] == pattern.entry_price
        assert signals['timing']['wait_for_confirmation'] == True  # Medium confidence
    
    def test_low_confidence_signals(self):
        """Test signals with low confidence."""
        detector = GartleyPattern()
        
        pattern = self._create_sample_pattern_result(PatternType.BULLISH, confidence=0.60)
        
        signals = detector.get_entry_signals(pattern)
        
        assert signals['timing']['avoid'] == True  # Low confidence
    
    def _create_sample_pattern_result(
        self, 
        pattern_type: PatternType, 
        confidence: float
    ) -> PatternResult:
        """Create sample PatternResult for tests."""
        return PatternResult(
            point_x=PatternPoint(0, 40000),
            point_a=PatternPoint(10, 50000),
            point_b=PatternPoint(20, 43000),
            point_c=PatternPoint(30, 47000),
            point_d=PatternPoint(40, 42000),
            pattern_type=pattern_type,
            validation_status=PatternValidation.VALID,
            confidence_score=confidence,
            ab_ratio=0.618,
            bc_ratio=0.571,
            cd_ratio=1.272,
            ad_ratio=0.786,
            entry_price=42000,
            stop_loss=38000,
            take_profit_1=44500,
            take_profit_2=46000,
            take_profit_3=48000,
            risk_reward_ratio=2.5,
            max_risk_percent=2.8,
            pattern_strength=0.82,
            fibonacci_confluence=0.87,
            volume_confirmation=0.65,
            completion_time=pd.Timestamp.now(),
            symbol="BTCUSDT",
            timeframe="1h"
        )


class TestPositionSizing:
    """Tests for position sizing calculation."""
    
    def test_position_sizing_calculation(self):
        """Test position sizing calculation."""
        detector = GartleyPattern()
        pattern = TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.8)
        
        account_balance = 10000  # $10k account
        max_risk_percent = 2.0   # 2% risk per trade
        
        position_info = detector.calculate_position_size(
            pattern, account_balance, max_risk_percent
        )
        
        # Check main calculations
        assert position_info['risk_amount'] == 200  # 2% of $10k
        assert position_info['position_size'] > 0
        assert position_info['leverage'] >= 1
        assert position_info['max_loss'] == 200
        
        # Potential profits should be calculated
        assert position_info['potential_profit_tp1'] > 0
        assert position_info['potential_profit_tp2'] > position_info['potential_profit_tp1']
        assert position_info['potential_profit_tp3'] > position_info['potential_profit_tp2']
    
    def test_position_sizing_with_confidence_adjustment(self):
        """Test position size adjustment based on confidence."""
        detector = GartleyPattern()
        
        # High confidence pattern
        high_conf_pattern = TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.9)
        high_conf_position = detector.calculate_position_size(high_conf_pattern, 10000, 2.0)
        
        # Low confidence pattern
        low_conf_pattern = TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.6)
        low_conf_position = detector.calculate_position_size(low_conf_pattern, 10000, 2.0)
        
        # High confidence should have a larger position size
        assert high_conf_position['position_size'] > low_conf_position['position_size']
        assert high_conf_position['confidence_adjustment'] > low_conf_position['confidence_adjustment']
    
    def test_zero_risk_position_sizing(self):
        """Test zero risk handling."""
        detector = GartleyPattern()
        
        # Pattern with entry == stop_loss (zero risk)
        pattern = TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.8)
        pattern = pattern._replace(stop_loss=pattern.entry_price)  # Zero risk
        
        position_info = detector.calculate_position_size(pattern, 10000, 2.0)
        
        assert position_info['position_size'] == 0
        assert position_info['risk_amount'] == 0


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_analyze_pattern_performance(self):
        """Test pattern performance analysis."""
        # Create patterns for analysis
        patterns = [
            TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.8),
            TestEntrySignals()._create_sample_pattern_result(PatternType.BEARISH, 0.75)
        ]
        
        # Create mock future price data
        future_prices = pd.Series([42000, 43000, 44500, 46000, 48000, 47000, 45000] * 10)
        
        performance = analyze_pattern_performance(patterns, future_prices, lookforward_periods=50)
        
        # Check result structure
        assert 'total_patterns' in performance
        assert 'successful_patterns' in performance
        assert 'success_rate' in performance
        assert 'tp1_hit_rate' in performance
        assert 'average_confidence' in performance
        
        assert performance['total_patterns'] == len(patterns)
        assert 0 <= performance['success_rate'] <= 1
    
    def test_filter_patterns_by_quality(self):
        """Test pattern quality filtering."""
        # Create patterns with different quality
        patterns = [
            TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.9),   # High quality
            TestEntrySignals()._create_sample_pattern_result(PatternType.BEARISH, 0.6),  # Low quality
            TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.8),  # Medium quality
        ]
        
        # Set high filtering standards
        filtered = filter_patterns_by_quality(
            patterns,
            min_confidence=0.75,
            min_risk_reward=2.0,
            max_risk_percent=3.0
        )
        
        # Only high-quality patterns should remain
        assert len(filtered) <= len(patterns)
        
        for pattern in filtered:
            assert pattern.confidence_score >= 0.75
            assert pattern.risk_reward_ratio >= 2.0
            assert pattern.max_risk_percent <= 3.0
            assert pattern.validation_status == PatternValidation.VALID
    
    def test_empty_pattern_lists(self):
        """Test handling of empty pattern lists."""
        # Performance analysis with empty list
        performance = analyze_pattern_performance([], pd.Series([]), 50)
        assert performance == {}
        
        # Filter with empty list
        filtered = filter_patterns_by_quality([])
        assert filtered == []


class TestVisualizationData:
    """Tests for visualization data preparation."""
    
    def test_prepare_visualization_data(self):
        """Test visualization data preparation."""
        detector = GartleyPattern()
        pattern = TestEntrySignals()._create_sample_pattern_result(PatternType.BULLISH, 0.85)
        
        viz_data = detector.prepare_visualization_data(pattern)
        
        # Check main components
        assert 'pattern_points' in viz_data
        assert 'pattern_lines' in viz_data
        assert 'fibonacci_levels' in viz_data
        assert 'trading_levels' in viz_data
        assert 'metadata' in viz_data
        
        # Check pattern points
        assert len(viz_data['pattern_points']) == 5  # X, A, B, C, D
        for point in viz_data['pattern_points']:
            assert 'name' in point
            assert 'index' in point
            assert 'price' in point
            assert 'color' in point
        
        # Check trading levels
        trading_levels = viz_data['trading_levels']
        level_names = [level['name'] for level in trading_levels]
        assert 'Entry' in level_names
        assert 'Stop Loss' in level_names
        assert 'TP1' in level_names
        assert 'TP2' in level_names
        assert 'TP3' in level_names
        
        # Check metadata
        metadata = viz_data['metadata']
        assert metadata['pattern_type'] == 'bullish'
        assert 'confidence_score' in metadata
        assert 'risk_reward_ratio' in metadata


class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_invalid_data_handling(self):
        """Test handling of incorrect data."""
        detector = GartleyPattern()
        
        # Test with None
        with pytest.raises((ValueError, AttributeError)):
            detector.detect_patterns(None)
        
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            detector.detect_patterns(empty_df)
    
    def test_corrupted_pattern_data(self):
        """Test handling of corrupted pattern data."""
        detector = GartleyPattern()
        
        # Pattern with None points
        corrupted_points = (None, None, None, None, None)
        data = TestDataValidation()._create_sample_ohlcv_data(100)
        
        result = detector._validate_and_score_pattern(
            corrupted_points, data, "BTCUSDT", "1h"
        )
        
        # Should return None or handle gracefully
        assert result is None
    
    def test_extreme_price_values(self):
        """Test handling of extreme price values."""
        detector = GartleyPattern()
        
        # Create data with extreme values
        extreme_data = pd.DataFrame({
            'open': [1e-10, 1e10, 0, float('inf')],
            'high': [1e-10, 1e10, 0, float('inf')],
            'low': [1e-10, 1e10, 0, float('inf')],
            'close': [1e-10, 1e10, 0, float('inf')],
            'volume': [1000, 1000, 1000, 1000]
        })
        
        # Should not raise an exception, may return empty list
        try:
            patterns = detector.detect_patterns(extreme_data)
            assert isinstance(patterns, list)
        except (ValueError, RuntimeError):
            # Acceptable if extreme values cause a controlled error
            pass


# Performance Tests
class TestPerformance:
    """Performance and scalability tests."""
    
    def test_large_dataset_performance(self):
        """Test performance on large data."""
        detector = GartleyPattern()
        
        # Create a large dataset
        large_data = TestDataValidation()._create_sample_ohlcv_data(5000)
        
        import time
        start_time = time.time()
        
        patterns = detector.detect_patterns(large_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete in reasonable time (< 30 seconds)
        assert execution_time < 30
        assert isinstance(patterns, list)
    
    def test_memory_usage(self):
        """Test memory usage."""
        detector = GartleyPattern()
        
        # Create data
        data = TestDataValidation()._create_sample_ohlcv_data(1000)
        
        # Run detection multiple times
        for _ in range(5):
            patterns = detector.detect_patterns(data, symbol=f"TEST_{_}", timeframe="1h")
        
        # Check that cache does not grow uncontrollably
        cache_stats = detector.get_cache_stats()
        assert cache_stats['patterns_cache_size'] <= 10  # Reasonable limit
        
        # Clear cache and check memory release
        detector.clear_cache()
        final_stats = detector.get_cache_stats()
        assert final_stats['patterns_cache_size'] == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])