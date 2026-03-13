"""
Level Validator for Support/Resistance System
ML-Framework-1332 - Validation and backtesting for support/resistance levels

 2025: Statistical validation with backtesting capabilities
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Result of level validation"""
    level_price: float
    is_valid: bool
    accuracy: float
    touches: int
    bounces: int
    breaks: int
    holding_percentage: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Result of backtesting levels"""
    total_levels_tested: int
    valid_levels: int
    accuracy_rate: float
    average_touches: float
    average_holding_time: float
    profit_factor: float
    win_rate: float
    statistics: Dict = field(default_factory=dict)


class LevelValidator:
    """Validate and backtest support/resistance levels"""

    def __init__(
        self,
        min_touches: int = 2,
        bounce_threshold_percent: float = 0.5,
        break_threshold_percent: float = 1.0
    ):
        self.min_touches = min_touches
        self.bounce_threshold_percent = bounce_threshold_percent
        self.break_threshold_percent = break_threshold_percent
        self.logger = logging.getLogger("LevelValidator")

    def validate_level(
        self,
        level,
        price_data: pd.DataFrame
    ) -> ValidationResult:
        """Validate a single support/resistance level"""

        level_price = level.price
        touches = 0
        bounces = 0
        breaks = 0
        holding_periods = []

        bounce_threshold = level_price * (self.bounce_threshold_percent / 100)
        break_threshold = level_price * (self.break_threshold_percent / 100)

        for i, row in price_data.iterrows():
            # Check for touch
            if self._check_touch(level_price, row, bounce_threshold):
                touches += 1

                # Check for bounce
                if i < len(price_data) - 1:
                    next_row = price_data.iloc[i + 1]
                    if self._check_bounce(level, row, next_row):
                        bounces += 1

            # Check for break
            if self._check_break(level, row, break_threshold):
                breaks += 1

        # Calculate accuracy
        accuracy = bounces / touches if touches > 0 else 0

        # Calculate holding percentage
        holding_percentage = (bounces / (bounces + breaks)) * 100 if (bounces + breaks) > 0 else 0

        # Determine if valid
        is_valid = (
            touches >= self.min_touches and
            accuracy >= 0.5 and
            holding_percentage >= 60
        )

        return ValidationResult(
            level_price=level_price,
            is_valid=is_valid,
            accuracy=accuracy,
            touches=touches,
            bounces=bounces,
            breaks=breaks,
            holding_percentage=holding_percentage,
            metadata={
                'level_type': level.level_type.value,
                'detection_method': level.method.value
            }
        )

    def backtest_levels(
        self,
        levels: List,
        price_data: pd.DataFrame,
        test_period_start: Optional[datetime] = None
    ) -> BacktestResult:
        """Backtest multiple levels on historical data"""

        if test_period_start:
            price_data = price_data[price_data['timestamp'] >= test_period_start]

        validation_results = []
        for level in levels:
            result = self.validate_level(level, price_data)
            validation_results.append(result)

        # Calculate statistics
        valid_count = sum(1 for r in validation_results if r.is_valid)
        total_count = len(validation_results)

        accuracy_rate = valid_count / total_count if total_count > 0 else 0
        average_touches = np.mean([r.touches for r in validation_results]) if validation_results else 0
        average_accuracy = np.mean([r.accuracy for r in validation_results]) if validation_results else 0

        # Calculate trading performance
        profit_factor, win_rate = self._calculate_trading_performance(
            validation_results, price_data
        )

        return BacktestResult(
            total_levels_tested=total_count,
            valid_levels=valid_count,
            accuracy_rate=accuracy_rate,
            average_touches=average_touches,
            average_holding_time=0,  # Can be calculated if needed
            profit_factor=profit_factor,
            win_rate=win_rate,
            statistics={
                'average_accuracy': average_accuracy,
                'total_touches': sum(r.touches for r in validation_results),
                'total_bounces': sum(r.bounces for r in validation_results),
                'total_breaks': sum(r.breaks for r in validation_results)
            }
        )

    def _check_touch(
        self,
        level_price: float,
        row: pd.Series,
        threshold: float
    ) -> bool:
        """Check if price touched the level"""
        return (
            abs(row['high'] - level_price) <= threshold or
            abs(row['low'] - level_price) <= threshold
        )

    def _check_bounce(
        self,
        level,
        current_row: pd.Series,
        next_row: pd.Series
    ) -> bool:
        """Check if price bounced from level"""
        level_price = level.price

        if level.level_type.value == 'support':
            # For support, price should bounce up
            return (
                current_row['low'] <= level_price * 1.01 and
                next_row['close'] > current_row['close']
            )
        else:  # resistance
            # For resistance, price should bounce down
            return (
                current_row['high'] >= level_price * 0.99 and
                next_row['close'] < current_row['close']
            )

    def _check_break(
        self,
        level,
        row: pd.Series,
        threshold: float
    ) -> bool:
        """Check if level was broken"""
        level_price = level.price

        if level.level_type.value == 'support':
            return row['close'] < level_price - threshold
        else:  # resistance
            return row['close'] > level_price + threshold

    def _calculate_trading_performance(
        self,
        validation_results: List[ValidationResult],
        price_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate trading performance metrics"""

        if not validation_results or price_data.empty:
            return 1.0, 0.5

        # Simulate simple bounce trading
        wins = sum(r.bounces for r in validation_results)
        losses = sum(r.breaks for r in validation_results)

        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0

        # Simple profit factor (can be enhanced with actual PnL)
        profit_factor = wins / losses if losses > 0 else wins

        return profit_factor, win_rate