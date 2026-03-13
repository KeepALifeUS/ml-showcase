"""
Data validators for Elliott Wave Analyzer.

Comprehensive validation with crypto market
data quality checks, Elliott Wave rule validation, and security validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import re

from .logger import get_logger, error_logger
from .config import config

logger = get_logger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    CRYPTO_ADAPTIVE = "crypto_adaptive"


class DataQuality(str, Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"     # 95%+ quality
    GOOD = "good"              # 85-94% quality
    ACCEPTABLE = "acceptable"   # 70-84% quality
    POOR = "poor"              # 50-69% quality
    UNUSABLE = "unusable"      # <50% quality


@dataclass
class ValidationResult:
    """Validation result container."""
    is_valid: bool
    quality_score: float
    quality_level: DataQuality
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.error_count = len(self.errors)
        self.warning_count = len(self.warnings)
        self.total_issues = self.error_count + self.warning_count
        
    @property
    def has_critical_errors(self) -> bool:
        """Check for critical validation errors."""
        return self.error_count > 0
        
    @property
    def is_production_ready(self) -> bool:
        """Check if data is ready for production use."""
        return (self.is_valid and 
                self.quality_score >= 0.7 and
                self.error_count == 0)


class PriceDataValidator:
    """
    Price data validation for crypto trading.
    
    Comprehensive OHLCV data validation with crypto-specific checks.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """Initialize price data validator."""
        self.validation_level = validation_level
        self.crypto_symbols = self._load_crypto_symbols()
        
        # Validation thresholds based on level
        self.thresholds = self._get_validation_thresholds()
        
    def _load_crypto_symbols(self) -> List[str]:
        """Load list of valid crypto symbols."""
        # In production, this would come from exchange APIs
        common_symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "BNBUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT",
            "TRXUSDT", "ETCUSDT", "XLMUSDT", "ATOMUSDT", "XMRUSDT"
        ]
        return common_symbols
        
    def _get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds based on validation level."""
        thresholds = {
            ValidationLevel.STRICT: {
                "min_data_points": 200,
                "max_gap_minutes": 5,
                "max_price_change_pct": 0.2,  # 20%
                "min_volume": 1000,
                "max_spread_pct": 0.05,       # 5%
                "required_columns": ["open", "high", "low", "close", "volume"],
                "allow_missing_volume": False
            },
            ValidationLevel.MODERATE: {
                "min_data_points": 100,
                "max_gap_minutes": 15,
                "max_price_change_pct": 0.5,  # 50%
                "min_volume": 100,
                "max_spread_pct": 0.1,        # 10%
                "required_columns": ["open", "high", "low", "close"],
                "allow_missing_volume": True
            },
            ValidationLevel.LENIENT: {
                "min_data_points": 50,
                "max_gap_minutes": 60,
                "max_price_change_pct": 1.0,  # 100%
                "min_volume": 0,
                "max_spread_pct": 0.2,        # 20%
                "required_columns": ["high", "low", "close"],
                "allow_missing_volume": True
            },
            ValidationLevel.CRYPTO_ADAPTIVE: {
                "min_data_points": 100,
                "max_gap_minutes": 5,         # Crypto trades 24/7
                "max_price_change_pct": 0.3,  # 30% (crypto volatility)
                "min_volume": 500,
                "max_spread_pct": 0.15,       # 15% (crypto spreads)
                "required_columns": ["open", "high", "low", "close", "volume"],
                "allow_missing_volume": False
            }
        }
        
        return thresholds.get(self.validation_level, thresholds[ValidationLevel.MODERATE])
    
    def validate_price_data(self, 
                          data: pd.DataFrame, 
                          symbol: str = None,
                          timeframe: str = None) -> ValidationResult:
        """
        Comprehensive price data validation.
        
        Multi-layer validation with crypto market adaptations.
        """
        errors = []
        warnings = []
        recommendations = []
        metadata = {}
        
        try:
            # Basic structure validation
            structure_issues = self._validate_structure(data)
            errors.extend(structure_issues.get("errors", []))
            warnings.extend(structure_issues.get("warnings", []))
            
            if not errors:  # Only continue if structure is valid
                # Data quality validation
                quality_issues = self._validate_data_quality(data)
                errors.extend(quality_issues.get("errors", []))
                warnings.extend(quality_issues.get("warnings", []))
                
                # OHLC logic validation
                ohlc_issues = self._validate_ohlc_logic(data)
                errors.extend(ohlc_issues.get("errors", []))
                warnings.extend(ohlc_issues.get("warnings", []))
                
                # Crypto-specific validation
                if self.validation_level == ValidationLevel.CRYPTO_ADAPTIVE:
                    crypto_issues = self._validate_crypto_specific(data, symbol, timeframe)
                    errors.extend(crypto_issues.get("errors", []))
                    warnings.extend(crypto_issues.get("warnings", []))
                    
                # Time series validation
                time_issues = self._validate_time_series(data, timeframe)
                errors.extend(time_issues.get("errors", []))
                warnings.extend(time_issues.get("warnings", []))
                
                # Statistical validation
                stats_issues = self._validate_statistics(data)
                warnings.extend(stats_issues.get("warnings", []))
                recommendations.extend(stats_issues.get("recommendations", []))
                
                # Generate metadata
                metadata = self._generate_metadata(data, symbol, timeframe)
                
        except Exception as e:
            errors.append(f"Validation failed with exception: {str(e)}")
            error_logger.log_exception(e, {"symbol": symbol, "timeframe": timeframe})
            
        # Calculate quality score
        quality_score = self._calculate_quality_score(data, errors, warnings)
        quality_level = self._determine_quality_level(quality_score)
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(errors, warnings, data))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=quality_score,
            quality_level=quality_level,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
            metadata=metadata
        )
        
    def _validate_structure(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate basic DataFrame structure."""
        errors = []
        warnings = []
        
        # Check if data exists
        if data is None or data.empty:
            errors.append("Data is empty or None")
            return {"errors": errors, "warnings": warnings}
            
        # Check minimum data points
        if len(data) < self.thresholds["min_data_points"]:
            errors.append(f"Insufficient data points: {len(data)} < {self.thresholds['min_data_points']}")
            
        # Check required columns
        missing_columns = []
        for col in self.thresholds["required_columns"]:
            if col not in data.columns:
                missing_columns.append(col)
                
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            
        # Check for volume column if required
        if not self.thresholds["allow_missing_volume"] and "volume" not in data.columns:
            errors.append("Volume column is required but missing")
            
        # Check index type
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.append("Index is not DatetimeIndex - timestamp parsing may be needed")
            
        return {"errors": errors, "warnings": warnings}
        
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data quality metrics."""
        errors = []
        warnings = []
        
        # Check for null values
        null_counts = data.isnull().sum()
        for col, count in null_counts.items():
            if count > 0:
                null_pct = (count / len(data)) * 100
                if null_pct > 5:  # More than 5% null values
                    errors.append(f"Column {col} has {null_pct:.1f}% null values")
                else:
                    warnings.append(f"Column {col} has {count} null values")
                    
        # Check for duplicate timestamps
        if isinstance(data.index, pd.DatetimeIndex):
            duplicate_count = data.index.duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Found {duplicate_count} duplicate timestamps")
                
        # Check for negative prices
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in data.columns:
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    errors.append(f"Column {col} has {negative_count} negative or zero values")
                    
        # Check for negative volume
        if "volume" in data.columns:
            negative_volume = (data["volume"] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Volume has {negative_volume} negative values")
                
        return {"errors": errors, "warnings": warnings}
        
    def _validate_ohlc_logic(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate OHLC price logic."""
        errors = []
        warnings = []
        
        required_cols = ["open", "high", "low", "close"]
        if not all(col in data.columns for col in required_cols):
            return {"errors": errors, "warnings": warnings}
            
        # High should be >= Open, Close
        high_violations = ((data["high"] < data["open"]) | 
                          (data["high"] < data["close"])).sum()
        if high_violations > 0:
            errors.append(f"High price logic violation in {high_violations} records")
            
        # Low should be <= Open, Close
        low_violations = ((data["low"] > data["open"]) | 
                         (data["low"] > data["close"])).sum()
        if low_violations > 0:
            errors.append(f"Low price logic violation in {low_violations} records")
            
        # High should be >= Low
        high_low_violations = (data["high"] < data["low"]).sum()
        if high_low_violations > 0:
            errors.append(f"High < Low violation in {high_low_violations} records")
            
        # Check for abnormal spreads
        spreads = (data["high"] - data["low"]) / data["low"]
        abnormal_spreads = (spreads > self.thresholds["max_spread_pct"]).sum()
        if abnormal_spreads > 0:
            warnings.append(f"Abnormally high spreads in {abnormal_spreads} records")
            
        return {"errors": errors, "warnings": warnings}
        
    def _validate_crypto_specific(self, 
                                data: pd.DataFrame, 
                                symbol: str = None,
                                timeframe: str = None) -> Dict[str, List[str]]:
        """Crypto-specific validation checks."""
        errors = []
        warnings = []
        
        # Validate symbol format
        if symbol:
            if not self._is_valid_crypto_symbol(symbol):
                warnings.append(f"Symbol {symbol} may not be a valid crypto pair")
                
        # Check for crypto market characteristics
        if "close" in data.columns:
            # High volatility is normal for crypto
            returns = data["close"].pct_change().dropna()
            volatility = returns.std()
            
            if volatility > 0.1:  # > 10% volatility
                warnings.append(f"High volatility detected: {volatility:.3f}")
            elif volatility < 0.005:  # < 0.5% volatility
                warnings.append("Unusually low volatility for crypto market")
                
        # Check for 24/7 trading (no gaps on weekends)
        if isinstance(data.index, pd.DatetimeIndex) and timeframe:
            gap_issues = self._check_crypto_trading_gaps(data, timeframe)
            errors.extend(gap_issues.get("errors", []))
            warnings.extend(gap_issues.get("warnings", []))
            
        # Volume validation for crypto
        if "volume" in data.columns:
            zero_volume_count = (data["volume"] == 0).sum()
            if zero_volume_count > len(data) * 0.05:  # More than 5% zero volume
                warnings.append(f"High number of zero volume periods: {zero_volume_count}")
                
        return {"errors": errors, "warnings": warnings}
        
    def _validate_time_series(self, 
                            data: pd.DataFrame, 
                            timeframe: str = None) -> Dict[str, List[str]]:
        """Validate time series properties."""
        errors = []
        warnings = []
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return {"errors": errors, "warnings": warnings}
            
        # Check chronological order
        if not data.index.is_monotonic_increasing:
            errors.append("Timestamps are not in chronological order")
            
        # Check for large time gaps
        time_diffs = data.index.to_series().diff().dropna()
        if timeframe:
            expected_interval = self._parse_timeframe_minutes(timeframe)
            if expected_interval:
                large_gaps = time_diffs > timedelta(minutes=expected_interval * 2)
                gap_count = large_gaps.sum()
                
                if gap_count > 0:
                    warnings.append(f"Found {gap_count} large time gaps in data")
                    
        # Check timezone consistency
        if data.index.tz is None:
            warnings.append("Timestamps have no timezone information")
        elif not all(ts.tz == data.index.tz for ts in data.index if ts.tz):
            errors.append("Inconsistent timezone in timestamps")
            
        return {"errors": errors, "warnings": warnings}
        
    def _validate_statistics(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate statistical properties."""
        warnings = []
        recommendations = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Check for outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(data)) * 100
            
            if outlier_pct > 5:  # More than 5% outliers
                warnings.append(f"Column {col} has {outlier_pct:.1f}% outliers")
                recommendations.append(f"Consider outlier treatment for {col}")
                
            # Check for low variance (potential data issues)
            if data[col].std() / data[col].mean() < 0.01:  # CV < 1%
                warnings.append(f"Column {col} has very low variance")
                
        return {"warnings": warnings, "recommendations": recommendations}
        
    def _is_valid_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol looks like a valid crypto pair."""
        # Basic crypto symbol pattern validation
        crypto_pattern = r'^[A-Z]{2,10}USDT?$|^[A-Z]{2,10}BTC$|^[A-Z]{2,10}ETH$'
        return bool(re.match(crypto_pattern, symbol))
        
    def _check_crypto_trading_gaps(self, 
                                 data: pd.DataFrame, 
                                 timeframe: str) -> Dict[str, List[str]]:
        """Check for trading gaps in 24/7 crypto markets."""
        errors = []
        warnings = []
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return {"errors": errors, "warnings": warnings}
            
        expected_interval = self._parse_timeframe_minutes(timeframe)
        if not expected_interval:
            return {"errors": errors, "warnings": warnings}
            
        # Check for gaps longer than expected
        time_diffs = data.index.to_series().diff().dropna()
        max_allowed_gap = timedelta(minutes=expected_interval * 2)
        
        large_gaps = time_diffs > max_allowed_gap
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            warnings.append(f"Found {gap_count} data gaps in 24/7 crypto market")
            
            # Weekend gaps are particularly suspicious for crypto
            weekend_gaps = 0
            for idx, is_gap in large_gaps.items():
                if is_gap and idx.weekday() in [5, 6]:  # Saturday, Sunday
                    weekend_gaps += 1
                    
            if weekend_gaps > 0:
                errors.append(f"Found {weekend_gaps} weekend gaps - crypto markets trade 24/7")
                
        return {"errors": errors, "warnings": warnings}
        
    def _parse_timeframe_minutes(self, timeframe: str) -> Optional[int]:
        """Parse timeframe string to minutes."""
        timeframe_map = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360,
            "12h": 720, "1d": 1440, "3d": 4320, "1w": 10080
        }
        return timeframe_map.get(timeframe.lower())
        
    def _calculate_quality_score(self, 
                               data: pd.DataFrame, 
                               errors: List[str], 
                               warnings: List[str]) -> float:
        """Calculate overall data quality score."""
        if not data.empty:
            base_score = 1.0
            
            # Penalize errors more heavily than warnings
            error_penalty = len(errors) * 0.2
            warning_penalty = len(warnings) * 0.05
            
            # Data completeness bonus
            completeness = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            completeness_bonus = completeness * 0.1
            
            # Size adequacy bonus
            size_bonus = min(len(data) / 1000, 0.1)  # Up to 10% bonus for large datasets
            
            quality_score = base_score - error_penalty - warning_penalty + completeness_bonus + size_bonus
            return max(0.0, min(1.0, quality_score))
        
        return 0.0
        
    def _determine_quality_level(self, score: float) -> DataQuality:
        """Determine quality level from score."""
        if score >= 0.95:
            return DataQuality.EXCELLENT
        elif score >= 0.85:
            return DataQuality.GOOD
        elif score >= 0.70:
            return DataQuality.ACCEPTABLE
        elif score >= 0.50:
            return DataQuality.POOR
        else:
            return DataQuality.UNUSABLE
            
    def _generate_metadata(self, 
                         data: pd.DataFrame, 
                         symbol: str = None,
                         timeframe: str = None) -> Dict[str, Any]:
        """Generate validation metadata."""
        metadata = {
            "data_points": len(data),
            "columns": list(data.columns),
            "date_range": {
                "start": data.index.min() if isinstance(data.index, pd.DatetimeIndex) else None,
                "end": data.index.max() if isinstance(data.index, pd.DatetimeIndex) else None
            },
            "completeness": 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns))),
            "validation_level": self.validation_level.value,
            "symbol": symbol,
            "timeframe": timeframe
        }
        
        # Add price statistics if available
        if "close" in data.columns:
            metadata.update({
                "price_range": {
                    "min": float(data["close"].min()),
                    "max": float(data["close"].max()),
                    "mean": float(data["close"].mean())
                },
                "volatility": float(data["close"].pct_change().std()) if len(data) > 1 else 0.0
            })
            
        # Add volume statistics if available
        if "volume" in data.columns:
            metadata.update({
                "volume_stats": {
                    "total": float(data["volume"].sum()),
                    "mean": float(data["volume"].mean()),
                    "zero_volume_periods": int((data["volume"] == 0).sum())
                }
            })
            
        return metadata
        
    def _generate_recommendations(self, 
                                errors: List[str], 
                                warnings: List[str],
                                data: pd.DataFrame) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Recommendations based on errors
        if any("null" in error.lower() for error in errors):
            recommendations.append("Consider filling null values with interpolation or forward fill")
            
        if any("duplicate" in error.lower() for error in errors):
            recommendations.append("Remove duplicate timestamps or aggregate duplicate periods")
            
        if any("gap" in error.lower() for error in errors):
            recommendations.append("Fill data gaps using market data providers or interpolation")
            
        # Recommendations based on warnings
        if any("volatility" in warning.lower() for warning in warnings):
            recommendations.append("Review volatility levels and consider risk adjustments")
            
        if any("outlier" in warning.lower() for warning in warnings):
            recommendations.append("Implement outlier detection and treatment procedures")
            
        # Data size recommendations
        if len(data) < 200:
            recommendations.append("Increase data sample size for more robust analysis")
        elif len(data) > 10000:
            recommendations.append("Consider data sampling or chunking for performance")
            
        return recommendations


class ElliottWaveValidator:
    """
    Elliott Wave specific validation rules.
    
    Domain-specific validation for wave patterns.
    """
    
    @staticmethod
    def validate_impulse_wave_rules(wave_data: Dict[str, Any]) -> ValidationResult:
        """Validate Elliott Wave impulse rules."""
        errors = []
        warnings = []
        
        # Rule 1: Wave 3 cannot be the shortest
        wave_lengths = [
            wave_data.get("wave_1_length", 0),
            wave_data.get("wave_3_length", 0), 
            wave_data.get("wave_5_length", 0)
        ]
        
        if wave_lengths[1] > 0 and wave_lengths[1] == min(wave_lengths):
            errors.append("Wave 3 cannot be the shortest impulse wave")
            
        # Rule 2: Wave 2 cannot retrace more than 100% of wave 1
        wave_2_retrace = wave_data.get("wave_2_retracement", 0)
        if wave_2_retrace >= 1.0:
            errors.append("Wave 2 cannot retrace more than 100% of wave 1")
            
        # Rule 3: Wave 4 cannot overlap wave 1 price territory
        # This would require actual price data to validate properly
        
        # Guidelines
        if wave_2_retrace < 0.382 or wave_2_retrace > 0.786:
            warnings.append("Wave 2 retracement outside typical 38.2%-78.6% range")
            
        quality_score = 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            quality_score=max(0.0, quality_score),
            quality_level=DataQuality.GOOD if len(errors) == 0 else DataQuality.POOR,
            errors=errors,
            warnings=warnings,
            recommendations=[],
            metadata={"rules_checked": 3, "guidelines_checked": 1}
        )
        
    @staticmethod
    def validate_fibonacci_relationships(ratios: Dict[str, float]) -> ValidationResult:
        """Validate Fibonacci ratio relationships."""
        errors = []
        warnings = []
        
        common_fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        
        for ratio_name, ratio_value in ratios.items():
            if isinstance(ratio_value, (int, float)) and 0.1 < ratio_value < 10.0:
                # Find closest Fibonacci level
                closest_fib = min(common_fib_levels, key=lambda x: abs(x - ratio_value))
                accuracy = 1 - abs(ratio_value - closest_fib) / closest_fib
                
                if accuracy < 0.85:  # More than 15% off
                    warnings.append(f"Ratio {ratio_name} ({ratio_value:.3f}) deviates from Fibonacci level")
                    
        quality_score = 1.0 - (len(warnings) * 0.1)
        
        return ValidationResult(
            is_valid=True,  # Fibonacci validation rarely produces hard errors
            quality_score=quality_score,
            quality_level=DataQuality.GOOD,
            errors=errors,
            warnings=warnings,
            recommendations=[],
            metadata={"ratios_checked": len(ratios)}
        )


# Main validation functions for easy import
def validate_price_data(data: pd.DataFrame, 
                       symbol: str = None,
                       timeframe: str = None,
                       validation_level: ValidationLevel = ValidationLevel.MODERATE) -> ValidationResult:
    """Main price data validation function."""
    validator = PriceDataValidator(validation_level)
    return validator.validate_price_data(data, symbol, timeframe)


def validate_crypto_data(data: pd.DataFrame, 
                        symbol: str, 
                        timeframe: str) -> ValidationResult:
    """Crypto-specific data validation."""
    validator = PriceDataValidator(ValidationLevel.CRYPTO_ADAPTIVE)
    return validator.validate_price_data(data, symbol, timeframe)


# Export main classes and functions
__all__ = [
    'PriceDataValidator',
    'ElliottWaveValidator',
    'ValidationResult',
    'ValidationLevel',
    'DataQuality',
    'validate_price_data',
    'validate_crypto_data'
]