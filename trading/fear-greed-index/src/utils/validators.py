"""
Data Validators for Fear & Greed Index System

Comprehensive validation utilities for data integrity,
format validation, and business logic constraints.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger(__name__)


class ValidationLevel(Enum):
    """Validation levels"""
    BASIC = "basic"           # Basic validation (types, not null)
    STANDARD = "standard"     # Standard validation + ranges
    STRICT = "strict"         # Strict validation + business logic
    ENTERPRISE = "enterprise" # Enterprise level


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_level: ValidationLevel
    timestamp: datetime

    def add_error(self, error: str) -> None:
        """Add an error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add a warning"""
        self.warnings.append(warning)

    def has_errors(self) -> bool:
        """Check for errors"""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check for warnings"""
        return len(self.warnings) > 0


class DataValidator:
    """
    Main data validation class for Fear & Greed Index System

    Implements enterprise patterns for ensuring
    high data quality and system reliability.
    """

    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level

        # Acceptable ranges for financial data
        self.financial_ranges = {
            "price": {"min": 0.000001, "max": 1_000_000},
            "volume": {"min": 0, "max": 1e18},
            "percentage": {"min": -100, "max": 100},
            "ratio": {"min": 0, "max": 1000},
            "fear_greed_score": {"min": 0, "max": 100},
            "sentiment_score": {"min": -1, "max": 1},
            "volatility": {"min": 0, "max": 1000}  # % annual
        }

        # Required columns for OHLCV data
        self.required_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']

        # Valid cryptocurrency symbols
        self.valid_crypto_symbols = {
            'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'COMP',
            'SOL', 'AVAX', 'MATIC', 'ATOM', 'FTM', 'NEAR', 'ALGO', 'XTZ'
        }

        # Regular expressions for validation
        self.regex_patterns = {
            "symbol": r'^[A-Z]{2,10}$',
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "url": r'^https?:\/\/[^\s<>"{}|\\^`\[\]]+$',
            "api_key": r'^[A-Za-z0-9_-]{20,}$'
        }

        logger.info("DataValidator initialized", level=validation_level)

    def create_result(self, is_valid: bool = True) -> ValidationResult:
        """Create a validation result object"""
        return ValidationResult(
            is_valid=is_valid,
            errors=[],
            warnings=[],
            validation_level=self.validation_level,
            timestamp=datetime.utcnow()
        )

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1
    ) -> ValidationResult:
        """
        Validate a pandas DataFrame

        Args:
            df: DataFrame to validate
            required_columns: Required columns
            min_rows: Minimum number of rows

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            # Basic validation
            if df is None:
                result.add_error("DataFrame is None")
                return result

            if not isinstance(df, pd.DataFrame):
                result.add_error(f"Expected pandas DataFrame, got {type(df)}")
                return result

            # Size check
            if len(df) < min_rows:
                result.add_error(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

            # Required columns check
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    result.add_error(f"Missing required columns: {missing_columns}")

            # Null values check
            null_counts = df.isnull().sum()
            null_columns = null_counts[null_counts > 0]
            if not null_columns.empty:
                result.add_warning(f"Columns with null values: {dict(null_columns)}")

            # Duplicate index check
            if df.index.duplicated().any():
                result.add_warning("DataFrame has duplicate index values")

            # Additional checks for strict level
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                # Data type check
                for column in df.columns:
                    if df[column].dtype == 'object':
                        non_string_count = sum(not isinstance(x, (str, type(None))) for x in df[column])
                        if non_string_count > 0:
                            result.add_warning(f"Column {column} has mixed types")

                # Memory check
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                if memory_usage > 100:  # > 100MB
                    result.add_warning(f"DataFrame uses {memory_usage:.2f} MB of memory")

            logger.debug("DataFrame validation completed",
                        shape=df.shape, errors=len(result.errors), warnings=len(result.warnings))

        except Exception as e:
            result.add_error(f"DataFrame validation failed: {str(e)}")
            logger.error("DataFrame validation error", error=str(e))

        return result

    def validate_price_data(self, price_data: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLCV price data

        Args:
            price_data: DataFrame with price data

        Returns:
            ValidationResult
        """
        result = self.validate_dataframe(
            price_data,
            required_columns=self.required_ohlcv_columns,
            min_rows=1
        )

        if not result.is_valid:
            return result

        try:
            # OHLC logic check
            invalid_ohlc = (
                (price_data['high'] < price_data['low']) |
                (price_data['high'] < price_data['open']) |
                (price_data['high'] < price_data['close']) |
                (price_data['low'] > price_data['open']) |
                (price_data['low'] > price_data['close'])
            )

            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                result.add_error(f"OHLC logic violation in {invalid_count} rows")

            # Price range check
            for price_col in ['open', 'high', 'low', 'close']:
                if price_col in price_data.columns:
                    col_data = price_data[price_col]

                    # Positive values check
                    if (col_data <= 0).any():
                        zero_count = (col_data <= 0).sum()
                        result.add_error(f"Column {price_col} has {zero_count} non-positive values")

                    # Reasonable range check
                    price_range = self.financial_ranges["price"]
                    out_of_range = (col_data < price_range["min"]) | (col_data > price_range["max"])
                    if out_of_range.any():
                        result.add_warning(f"Column {price_col} has values outside reasonable range")

            # Volume check
            if 'volume' in price_data.columns:
                volume = price_data['volume']

                if (volume < 0).any():
                    result.add_error("Volume has negative values")

                # Zero volume check
                zero_volume_count = (volume == 0).sum()
                if zero_volume_count > len(price_data) * 0.1:  # > 10% zero volumes
                    result.add_warning(f"High number of zero volume periods: {zero_volume_count}")

            # Timestamp check (if index is datetime)
            if isinstance(price_data.index, pd.DatetimeIndex):
                # Check for time gaps
                time_diff = price_data.index.to_series().diff()
                median_diff = time_diff.median()

                large_gaps = time_diff > median_diff * 3
                if large_gaps.any():
                    result.add_warning(f"Found {large_gaps.sum()} large time gaps in data")

                # Check for future dates
                future_dates = price_data.index > datetime.utcnow()
                if future_dates.any():
                    result.add_error(f"Found {future_dates.sum()} future timestamps")

            # Enterprise level - additional checks
            if self.validation_level == ValidationLevel.ENTERPRISE:
                # Check for suspicious patterns
                self._check_suspicious_patterns(price_data, result)

                # Assess data quality
                self._assess_data_quality(price_data, result)

        except Exception as e:
            result.add_error(f"Price data validation failed: {str(e)}")
            logger.error("Price data validation error", error=str(e))

        return result

    def _check_suspicious_patterns(self, price_data: pd.DataFrame, result: ValidationResult) -> None:
        """Check for suspicious patterns in data"""
        try:
            # Check for identical OHLC values (suspicious)
            same_ohlc = (
                (price_data['open'] == price_data['high']) &
                (price_data['high'] == price_data['low']) &
                (price_data['low'] == price_data['close'])
            )

            if same_ohlc.any():
                same_count = same_ohlc.sum()
                if same_count > len(price_data) * 0.05:  # > 5%
                    result.add_warning(f"High number of periods with identical OHLC values: {same_count}")

            # Check for extreme price changes
            if len(price_data) > 1:
                price_changes = price_data['close'].pct_change().abs()
                extreme_changes = price_changes > 0.5  # > 50% change

                if extreme_changes.any():
                    extreme_count = extreme_changes.sum()
                    result.add_warning(f"Found {extreme_count} extreme price changes (>50%)")

        except Exception as e:
            logger.error("Suspicious pattern check failed", error=str(e))

    def _assess_data_quality(self, price_data: pd.DataFrame, result: ValidationResult) -> None:
        """Assess overall data quality"""
        try:
            quality_score = 100.0

            # Penalty for missing values
            null_percentage = price_data.isnull().sum().sum() / (len(price_data) * len(price_data.columns)) * 100
            quality_score -= null_percentage * 2

            # Penalty for zero volumes
            if 'volume' in price_data.columns:
                zero_volume_pct = (price_data['volume'] == 0).sum() / len(price_data) * 100
                quality_score -= zero_volume_pct

            # Bonus for data completeness
            if len(price_data) > 1000:  # Large data volume
                quality_score += 5

            quality_score = max(0, min(100, quality_score))

            if quality_score < 70:
                result.add_warning(f"Data quality score is low: {quality_score:.1f}/100")
            else:
                result.add_warning(f"Data quality score: {quality_score:.1f}/100")

        except Exception as e:
            logger.error("Data quality assessment failed", error=str(e))

    def validate_fear_greed_score(self, score: float) -> ValidationResult:
        """
        Validate Fear & Greed score

        Args:
            score: Score to validate (0-100)

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            if not isinstance(score, (int, float)):
                result.add_error(f"Score must be numeric, got {type(score)}")
                return result

            if pd.isna(score):
                result.add_error("Score cannot be NaN")
                return result

            score_range = self.financial_ranges["fear_greed_score"]
            if not (score_range["min"] <= score <= score_range["max"]):
                result.add_error(f"Score {score} outside valid range {score_range['min']}-{score_range['max']}")

            # Warnings for extreme values
            if score < 10:
                result.add_warning("Score indicates extreme fear")
            elif score > 90:
                result.add_warning("Score indicates extreme greed")

        except Exception as e:
            result.add_error(f"Score validation failed: {str(e)}")
            logger.error("Score validation error", error=str(e))

        return result

    def validate_symbol(self, symbol: str) -> ValidationResult:
        """
        Validate cryptocurrency symbol

        Args:
            symbol: Symbol to validate

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            if not isinstance(symbol, str):
                result.add_error(f"Symbol must be string, got {type(symbol)}")
                return result

            if not symbol:
                result.add_error("Symbol cannot be empty")
                return result

            # Format check
            if not re.match(self.regex_patterns["symbol"], symbol):
                result.add_error(f"Symbol '{symbol}' has invalid format")

            # Check against known symbols (for strict level)
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                if symbol.upper() not in self.valid_crypto_symbols:
                    result.add_warning(f"Symbol '{symbol}' not in known cryptocurrency list")

        except Exception as e:
            result.add_error(f"Symbol validation failed: {str(e)}")
            logger.error("Symbol validation error", error=str(e))

        return result

    def validate_timeframe(self, timeframe: str) -> ValidationResult:
        """
        Validate time interval

        Args:
            timeframe: Timeframe to validate (e.g., '1h', '1d', '1w')

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            if not isinstance(timeframe, str):
                result.add_error(f"Timeframe must be string, got {type(timeframe)}")
                return result

            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']

            if timeframe not in valid_timeframes:
                result.add_error(f"Timeframe '{timeframe}' not supported. Valid: {valid_timeframes}")

        except Exception as e:
            result.add_error(f"Timeframe validation failed: {str(e)}")
            logger.error("Timeframe validation error", error=str(e))

        return result

    def validate_api_response(
        self,
        response: Dict[str, Any],
        required_fields: Optional[List[str]] = None
    ) -> ValidationResult:
        """
        Validate external API response

        Args:
            response: API response to validate
            required_fields: Required fields

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            if not isinstance(response, dict):
                result.add_error(f"API response must be dict, got {type(response)}")
                return result

            if not response:
                result.add_error("API response is empty")
                return result

            # Required fields check
            if required_fields:
                missing_fields = set(required_fields) - set(response.keys())
                if missing_fields:
                    result.add_error(f"Missing required fields: {missing_fields}")

            # Error check in response
            if 'error' in response:
                result.add_error(f"API returned error: {response['error']}")

            if 'status' in response and response['status'] != 'success':
                result.add_warning(f"API status: {response['status']}")

        except Exception as e:
            result.add_error(f"API response validation failed: {str(e)}")
            logger.error("API response validation error", error=str(e))

        return result

    def validate_sentiment_data(
        self,
        sentiment_score: float,
        confidence: float
    ) -> ValidationResult:
        """
        Validate sentiment analysis data

        Args:
            sentiment_score: Sentiment score (-1 to 1)
            confidence: Confidence score (0 to 1)

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            # Validate sentiment score
            if not isinstance(sentiment_score, (int, float)) or pd.isna(sentiment_score):
                result.add_error("Sentiment score must be numeric and not NaN")
            else:
                sentiment_range = self.financial_ranges["sentiment_score"]
                if not (sentiment_range["min"] <= sentiment_score <= sentiment_range["max"]):
                    result.add_error(f"Sentiment score {sentiment_score} outside valid range [-1, 1]")

            # Validate confidence
            if not isinstance(confidence, (int, float)) or pd.isna(confidence):
                result.add_error("Confidence must be numeric and not NaN")
            else:
                if not (0 <= confidence <= 1):
                    result.add_error(f"Confidence {confidence} outside valid range [0, 1]")

                # Warning for low confidence
                if confidence < 0.3:
                    result.add_warning(f"Low confidence score: {confidence}")

        except Exception as e:
            result.add_error(f"Sentiment data validation failed: {str(e)}")
            logger.error("Sentiment validation error", error=str(e))

        return result

    def validate_batch_data(
        self,
        data_batch: List[Dict[str, Any]],
        batch_size_limit: int = 1000
    ) -> ValidationResult:
        """
        Validate a data batch

        Args:
            data_batch: Data batch to validate
            batch_size_limit: Maximum batch size

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            if not isinstance(data_batch, list):
                result.add_error(f"Data batch must be list, got {type(data_batch)}")
                return result

            if len(data_batch) == 0:
                result.add_error("Data batch is empty")
                return result

            if len(data_batch) > batch_size_limit:
                result.add_error(f"Batch size {len(data_batch)} exceeds limit {batch_size_limit}")

            # Validate each element
            invalid_items = []
            for i, item in enumerate(data_batch):
                if not isinstance(item, dict):
                    invalid_items.append(i)

            if invalid_items:
                result.add_error(f"Non-dict items at indices: {invalid_items}")

        except Exception as e:
            result.add_error(f"Batch validation failed: {str(e)}")
            logger.error("Batch validation error", error=str(e))

        return result

    def validate_all(self, **kwargs) -> ValidationResult:
        """
        Comprehensive validation of all provided data

        Args:
            **kwargs: Data to validate

        Returns:
            ValidationResult
        """
        result = self.create_result()

        try:
            validation_count = 0

            for key, value in kwargs.items():
                if key.endswith('_dataframe') and isinstance(value, pd.DataFrame):
                    sub_result = self.validate_dataframe(value)
                elif key.endswith('_score') and isinstance(value, (int, float)):
                    sub_result = self.validate_fear_greed_score(value)
                elif key == 'symbol' and isinstance(value, str):
                    sub_result = self.validate_symbol(value)
                elif key == 'timeframe' and isinstance(value, str):
                    sub_result = self.validate_timeframe(value)
                else:
                    continue  # Skip unknown types

                validation_count += 1

                # Merge results
                result.errors.extend(sub_result.errors)
                result.warnings.extend(sub_result.warnings)

                if not sub_result.is_valid:
                    result.is_valid = False

            logger.info(f"Completed comprehensive validation",
                       items_validated=validation_count,
                       errors=len(result.errors),
                       warnings=len(result.warnings))

        except Exception as e:
            result.add_error(f"Comprehensive validation failed: {str(e)}")
            logger.error("Comprehensive validation error", error=str(e))

        return result
