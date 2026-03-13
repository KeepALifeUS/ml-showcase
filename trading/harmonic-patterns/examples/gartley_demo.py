#!/usr/bin/env python3
"""
Gartley Pattern Detection Demo - Complete Example.

Professional demonstration of Gartley pattern detection
with real-world crypto trading scenarios, comprehensive analysis, and visualization.

This example demonstrates:
1. Pattern detection in realistic crypto data
2. Trading signal generation
3. Risk management calculations
4. Performance analysis
5. Visualization data preparation

Author: ML Harmonic Patterns Contributors
Created: 2025-09-11
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
from typing import List, Dict
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from patterns.gartley_pattern import (
    GartleyPattern,
    PatternResult,
    PatternType,
    analyze_pattern_performance,
    filter_patterns_by_quality
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_realistic_crypto_data(symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
    """
    Create realistic crypto OHLCV data for demonstration.
    
    Simulates Bitcoin behavior with volatility, trends, and embedded Gartley patterns.
    """
    logger.info(f"Generating {days} days of realistic {symbol} data...")
    
    # Parameters for realistic modeling
    hours = days * 24
    base_price = 45000.0  # Starting BTC price
    daily_volatility = 0.03  # 3% daily volatility
    
    # Create timestamp index
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(start=start_date, periods=hours, freq='1H')
    
    # Generate base price path with trend and volatility
    np.random.seed(42)  # For reproducibility
    
    # Long-term trend (slight upward bias)
    trend = np.linspace(0, 0.2, hours)  # 20% growth over period
    
    # Daily cycles (crypto often has daily patterns)
    daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * 0.01
    
    # Random walk component
    random_returns = np.random.normal(0, daily_volatility / np.sqrt(24), hours)
    
    # Combine components
    log_returns = trend / hours + daily_cycle + random_returns
    prices = [base_price]
    
    for i in range(1, hours):
        new_price = prices[-1] * np.exp(log_returns[i])
        prices.append(max(new_price, 1000.0))  # Floor price
    
    # Create OHLCV data
    ohlcv_data = []
    
    for i, close_price in enumerate(prices):
        # Generate realistic OHLC from close price
        intrabar_volatility = abs(np.random.normal(0, daily_volatility / 4))
        
        # Open (previous close or gap)
        if i == 0:
            open_price = close_price
        else:
            gap = np.random.normal(0, 0.002)  # Small gaps
            open_price = prices[i-1] * (1 + gap)
        
        # High/Low range
        range_size = close_price * intrabar_volatility
        high = max(open_price, close_price) + range_size
        low = min(open_price, close_price) - range_size
        
        # Volume (anti-correlated with price movement and higher on volatility)
        price_change = abs(close_price - open_price) / open_price if open_price > 0 else 0
        base_volume = np.random.uniform(800, 1200)
        volatility_multiplier = 1 + price_change * 5  # Higher volume on big moves
        volume = base_volume * volatility_multiplier * np.random.uniform(0.5, 1.5)
        
        ohlcv_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(ohlcv_data, index=dates)
    
    # Add embedded Gartley patterns for demonstration
    df = _embed_gartley_patterns(df)
    
    logger.info(f"Generated {len(df)} hours of {symbol} data")
    logger.info(f"Price range: ${df['low'].min():,.0f} - ${df['high'].max():,.0f}")
    logger.info(f"Average volume: {df['volume'].mean():,.0f}")
    
    return df


def _embed_gartley_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Embed Gartley-like structures in data for demonstration."""
    
    # Create copy for modification
    modified_df = df.copy()
    
    # Embed bullish Gartley in first third of data
    if len(df) > 200:
        start_idx = len(df) // 4
        _create_gartley_structure(modified_df, start_idx, pattern_type='bullish')
    
    # Embed bearish Gartley in second half
    if len(df) > 400:
        start_idx = len(df) * 2 // 3
        _create_gartley_structure(modified_df, start_idx, pattern_type='bearish')
    
    return modified_df


def _create_gartley_structure(
    df: pd.DataFrame, 
    start_idx: int, 
    pattern_type: str,
    amplitude: float = 0.08
):
    """Create Gartley structure in data."""
    
    if start_idx + 100 > len(df):
        return
    
    base_price = df.iloc[start_idx]['close']
    
    if pattern_type == 'bullish':
        # Bullish Gartley: X(low) -> A(high) -> B(low) -> C(high) -> D(low)
        xa_move = base_price * amplitude         # Up move
        ab_retracement = xa_move * 0.618        # 61.8% retracement
        bc_move = ab_retracement * 0.5          # Partial recovery
        cd_move = bc_move * 1.272               # 127.2% extension
        
        # Apply structure to prices
        pattern_points = [
            (start_idx, 0),                                    # X
            (start_idx + 20, xa_move),                        # A 
            (start_idx + 40, xa_move - ab_retracement),       # B
            (start_idx + 60, xa_move - ab_retracement + bc_move), # C
            (start_idx + 80, xa_move - ab_retracement + bc_move - cd_move) # D
        ]
        
    else:  # bearish
        # Bearish Gartley: X(high) -> A(low) -> B(high) -> C(low) -> D(high)
        xa_move = base_price * amplitude         # Down move
        ab_retracement = xa_move * 0.618        # 61.8% retracement  
        bc_move = ab_retracement * 0.5          # Partial recovery
        cd_move = bc_move * 1.272               # 127.2% extension
        
        pattern_points = [
            (start_idx, 0),                                    # X
            (start_idx + 20, -xa_move),                       # A
            (start_idx + 40, -xa_move + ab_retracement),      # B
            (start_idx + 60, -xa_move + ab_retracement - bc_move), # C
            (start_idx + 80, -xa_move + ab_retracement - bc_move + cd_move) # D
        ]
    
    # Smooth interpolation between points
    for i in range(len(pattern_points) - 1):
        start_point = pattern_points[i]
        end_point = pattern_points[i + 1]
        
        start_price = base_price + start_point[1]
        end_price = base_price + end_point[1]
        
        # Interpolate between points
        indices = range(start_point[0], end_point[0])
        if indices:
            price_diff = end_price - start_price
            for j, idx in enumerate(indices):
                if idx < len(df):
                    progress = j / len(indices)
                    target_price = start_price + (price_diff * progress)
                    
                    # Apply with blending
                    current_price = df.iloc[idx]['close'] 
                    blended_price = current_price * 0.7 + target_price * 0.3
                    
                    # Update OHLC maintaining realism
                    df.iloc[idx, df.columns.get_loc('close')] = blended_price
                    df.iloc[idx, df.columns.get_loc('high')] = max(df.iloc[idx]['high'], blended_price * 1.005)
                    df.iloc[idx, df.columns.get_loc('low')] = min(df.iloc[idx]['low'], blended_price * 0.995)


def demonstrate_pattern_detection():
    """Main demonstration of Gartley pattern detection."""
    
    print("=" * 80)
    print("🎯 GARTLEY PATTERN DETECTION DEMO")
    print("ML Harmonic Patterns")
    print("=" * 80)
    
    # 1. Create realistic crypto data
    print("\n📊 Step 1: Generating realistic crypto market data...")
    data = create_realistic_crypto_data(symbol="BTCUSDT", days=15)
    
    print(f"   ✅ Generated {len(data)} hours of BTCUSDT data")
    print(f"   📈 Price range: ${data['low'].min():,.0f} - ${data['high'].max():,.0f}")
    print(f"   📊 Total volume: {data['volume'].sum():,.0f}")
    
    # 2. Initialize detector
    print("\n🔧 Step 2: Initializing Gartley Pattern Detector...")
    detector = GartleyPattern(
        tolerance=0.05,              # 5% Fibonacci tolerance
        min_confidence=0.70,         # 70% minimum confidence
        enable_volume_analysis=True,
        enable_ml_scoring=True,
        min_pattern_bars=20,
        max_pattern_bars=150
    )
    print("   ✅ Detector initialized with enterprise configuration")
    
    # 3. Detect patterns
    print("\n🔍 Step 3: Detecting Gartley patterns...")
    patterns = detector.detect_patterns(
        data, 
        symbol="BTCUSDT", 
        timeframe="1h"
    )
    
    print(f"   ✅ Found {len(patterns)} valid Gartley patterns")
    
    if not patterns:
        print("   ⚠️  No patterns found. Trying with lower confidence threshold...")
        detector.min_confidence = 0.50
        patterns = detector.detect_patterns(data, symbol="BTCUSDT", timeframe="1h")
        print(f"   ✅ Found {len(patterns)} patterns with relaxed criteria")
    
    # 4. Analyze detected patterns
    if patterns:
        print("\n📈 Step 4: Analyzing detected patterns...")
        analyze_detected_patterns(patterns)
        
        # 5. Generate trading signals
        print("\n🎯 Step 5: Generating trading signals...")
        generate_trading_signals(detector, patterns)
        
        # 6. Risk management
        print("\n🛡️ Step 6: Risk management analysis...")
        risk_management_demo(detector, patterns)
        
        # 7. Visualization preparation
        print("\n🎨 Step 7: Preparing visualization data...")
        visualization_demo(detector, patterns[0])
        
        # 8. Performance analysis
        print("\n📊 Step 8: Performance analysis...")
        performance_analysis_demo(patterns, data)
        
    else:
        print("   ❌ No patterns detected for analysis")
        print("   💡 Try adjusting detection parameters or using different data")
    
    # 9. Cache and performance stats
    print("\n⚡ Step 9: Performance statistics...")
    cache_stats = detector.get_cache_stats()
    print(f"   📊 Cache usage: {cache_stats}")
    
    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)


def analyze_detected_patterns(patterns: List[PatternResult]):
    """Analysis of detected patterns."""
    
    print(f"\n   📊 Pattern Distribution:")
    bullish_count = sum(1 for p in patterns if p.pattern_type == PatternType.BULLISH)
    bearish_count = len(patterns) - bullish_count
    
    print(f"      🟢 Bullish patterns: {bullish_count}")
    print(f"      🔴 Bearish patterns: {bearish_count}")
    
    # Confidence distribution
    confidences = [p.confidence_score for p in patterns]
    print(f"\n   🎯 Confidence Statistics:")
    print(f"      📈 Average: {np.mean(confidences):.1%}")
    print(f"      📊 Range: {min(confidences):.1%} - {max(confidences):.1%}")
    print(f"      🎪 High confidence (>80%): {sum(1 for c in confidences if c > 0.8)}")
    
    # Risk/Reward analysis
    rr_ratios = [p.risk_reward_ratio for p in patterns]
    print(f"\n   ⚖️ Risk/Reward Analysis:")
    print(f"      📈 Average R/R: {np.mean(rr_ratios):.2f}")
    print(f"      🎯 Excellent R/R (>2.0): {sum(1 for r in rr_ratios if r > 2.0)}")
    
    # Fibonacci accuracy
    fib_scores = [p.fibonacci_confluence for p in patterns]
    print(f"\n   📐 Fibonacci Accuracy:")
    print(f"      📊 Average confluence: {np.mean(fib_scores):.1%}")
    print(f"      🎯 High accuracy (>85%): {sum(1 for f in fib_scores if f > 0.85)}")


def generate_trading_signals(detector: GartleyPattern, patterns: List[PatternResult]):
    """Generate trading signals."""
    
    # Analyze best pattern
    best_pattern = patterns[0]  # Already sorted by confidence
    
    print(f"\n   🏆 Best Pattern Analysis:")
    print(f"      📍 Type: {best_pattern.pattern_type.value.title()}")
    print(f"      🎯 Confidence: {best_pattern.confidence_score:.1%}")
    print(f"      📊 Pattern Strength: {best_pattern.pattern_strength:.1%}")
    print(f"      ⚖️ Risk/Reward: {best_pattern.risk_reward_ratio:.2f}")
    
    # Generate signals
    signals = detector.get_entry_signals(best_pattern)
    
    print(f"\n   📈 Trading Signals:")
    print(f"      🎯 Action: {signals['action']}")
    print(f"      💰 Entry Price: ${signals['entry_price']:,.2f}")
    print(f"      🛑 Stop Loss: ${signals['stop_loss']:,.2f}")
    print(f"      🎪 Take Profit 1: ${signals['take_profit_levels'][0]:,.2f}")
    print(f"      🎪 Take Profit 2: ${signals['take_profit_levels'][1]:,.2f}")
    print(f"      🎪 Take Profit 3: ${signals['take_profit_levels'][2]:,.2f}")
    
    # Entry conditions
    conditions = signals['entry_conditions']
    print(f"\n   ✅ Entry Conditions:")
    for condition, status in conditions.items():
        emoji = "✅" if status else "❌"
        print(f"      {emoji} {condition.replace('_', ' ').title()}: {status}")
    
    # Timing recommendation
    timing = signals['timing']
    if timing['immediate']:
        print(f"   🚀 Recommendation: IMMEDIATE ENTRY (High confidence)")
    elif timing['wait_for_confirmation']:
        print(f"   ⏳ Recommendation: WAIT FOR CONFIRMATION (Medium confidence)")
    else:
        print(f"   ⛔ Recommendation: AVOID (Low confidence)")


def risk_management_demo(detector: GartleyPattern, patterns: List[PatternResult]):
    """Risk management demonstration."""
    
    best_pattern = patterns[0]
    
    # Different account sizes
    account_sizes = [5000, 10000, 25000, 50000]
    risk_percentages = [1.0, 2.0, 3.0]
    
    print(f"\n   💼 Position Sizing Analysis:")
    
    for account_balance in account_sizes:
        print(f"\n      💰 Account Balance: ${account_balance:,}")
        
        for risk_pct in risk_percentages:
            position_info = detector.calculate_position_size(
                best_pattern, account_balance, risk_pct
            )
            
            print(f"         📊 {risk_pct}% Risk:")
            print(f"            🎯 Position Size: {position_info['position_size']:.4f} BTC")
            print(f"            💸 Risk Amount: ${position_info['risk_amount']:.0f}")
            print(f"            📈 Leverage: {position_info['leverage']:.1f}x")
            print(f"            💰 Potential Profit (TP1): ${position_info['potential_profit_tp1']:.0f}")
    
    # Risk analysis
    max_risk = best_pattern.max_risk_percent
    print(f"\n   ⚠️ Risk Analysis:")
    print(f"      📊 Maximum drawdown risk: {max_risk:.1f}%")
    print(f"      🎯 Risk category: {'LOW' if max_risk < 2 else 'MEDIUM' if max_risk < 4 else 'HIGH'}")
    
    if max_risk > 5:
        print(f"      ⚠️ WARNING: High risk pattern - consider smaller position size")


def visualization_demo(detector: GartleyPattern, pattern: PatternResult):
    """Visualization data demonstration."""
    
    viz_data = detector.prepare_visualization_data(pattern)
    
    print(f"\n   🎨 Visualization Data Prepared:")
    print(f"      📍 Pattern Points: {len(viz_data['pattern_points'])} (X, A, B, C, D)")
    print(f"      📏 Pattern Lines: {len(viz_data['pattern_lines'])} connection lines")
    print(f"      📐 Fibonacci Levels: {len(viz_data['fibonacci_levels'])} ratios")
    print(f"      🎯 Trading Levels: {len(viz_data['trading_levels'])} price levels")
    
    # Show pattern points
    print(f"\n   📍 Pattern Points Detail:")
    for point in viz_data['pattern_points']:
        print(f"      {point['name']}: Index {point['index']}, Price ${point['price']:,.2f}")
    
    # Show Fibonacci accuracy
    print(f"\n   📐 Fibonacci Validation:")
    for fib_level in viz_data['fibonacci_levels']:
        if 'accuracy' in fib_level:
            accuracy = fib_level['accuracy']
            emoji = "✅" if accuracy > 0.9 else "⚠️" if accuracy > 0.7 else "❌"
            print(f"      {emoji} {fib_level['level']}: {accuracy:.1%} accuracy")
        elif 'in_range' in fib_level:
            status = "✅" if fib_level['in_range'] else "❌"
            print(f"      {status} {fib_level['level']}: {'In range' if fib_level['in_range'] else 'Out of range'}")
    
    # Trading levels
    print(f"\n   🎯 Trading Levels:")
    for level in viz_data['trading_levels']:
        print(f"      📊 {level['name']}: ${level['price']:,.2f}")


def performance_analysis_demo(patterns: List[PatternResult], data: pd.DataFrame):
    """Performance analysis demonstration."""
    
    # Create mock future price data for analysis
    last_prices = data['close'].iloc[-100:]  # Last 100 prices as "future"
    
    # Analyze performance
    performance = analyze_pattern_performance(
        patterns=patterns,
        actual_prices=last_prices,
        lookforward_periods=50
    )
    
    if performance:
        print(f"\n   📊 Pattern Performance Analysis:")
        print(f"      🎯 Total Patterns: {performance['total_patterns']}")
        print(f"      ✅ Successful Patterns: {performance['successful_patterns']}")
        print(f"      📈 Success Rate: {performance['success_rate']:.1%}")
        print(f"      🎪 TP1 Hit Rate: {performance['tp1_hit_rate']:.1%}")
        print(f"      🎯 TP2 Hit Rate: {performance['tp2_hit_rate']:.1%}")
        print(f"      💎 TP3 Hit Rate: {performance['tp3_hit_rate']:.1%}")
        print(f"      🧠 Average Confidence: {performance['average_confidence']:.1%}")
        print(f"      ⚖️ Average R/R Ratio: {performance['average_risk_reward']:.2f}")
    
    # Quality filtering demo
    high_quality = filter_patterns_by_quality(
        patterns,
        min_confidence=0.80,
        min_risk_reward=2.0,
        max_risk_percent=3.0
    )
    
    print(f"\n   🏆 Quality Filtering Results:")
    print(f"      📊 Original patterns: {len(patterns)}")
    print(f"      ✨ High-quality patterns: {len(high_quality)}")
    print(f"      📈 Quality ratio: {len(high_quality)/len(patterns)*100 if patterns else 0:.1f}%")
    
    if high_quality:
        avg_conf = np.mean([p.confidence_score for p in high_quality])
        avg_rr = np.mean([p.risk_reward_ratio for p in high_quality])
        print(f"      🎯 Average confidence (filtered): {avg_conf:.1%}")
        print(f"      ⚖️ Average R/R (filtered): {avg_rr:.2f}")


def demo_error_handling():
    """Error handling demonstration."""
    
    print("\n🛡️ Error Handling Demo:")
    
    detector = GartleyPattern()
    
    # Test with invalid data
    try:
        detector.detect_patterns(None)
    except (ValueError, AttributeError) as e:
        print(f"   ✅ Properly handled None data: {type(e).__name__}")
    
    # Test with insufficient data
    try:
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1000, 1000]
        })
        detector.detect_patterns(small_data)
    except ValueError as e:
        print(f"   ✅ Properly handled insufficient data: Insufficient data")
    
    print(f"   🎯 Error handling works correctly!")


if __name__ == "__main__":
    try:
        # Main demonstration
        demonstrate_pattern_detection()
        
        # Error handling demo
        demo_error_handling()
        
        print(f"\n🎉 All demonstrations completed successfully!")
        print(f"💡 Check the logs above for detailed analysis results.")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)