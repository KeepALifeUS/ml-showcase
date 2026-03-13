"""
Helper utilities for Prophet forecasting system.

Common utility functions for data validation, formatting, type conversion,
and other helper operations following enterprise patterns.
"""

import re
import math
from typing import Union, Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from .logger import get_logger
from .exceptions import InvalidDataException

logger = get_logger(__name__)

# Supported symbols cryptocurrencies
SUPPORTED_SYMBOLS = {
    "BTC", "ETH", "BNB", "ADA", "SOL", "XRP", "AVAX", "DOT", "MATIC", "LINK",
    "UNI", "AAVE", "SUSHI", "CRV", "YFI", "COMP", "MKR", "SNX", "1INCH", "ALPHA",
    "LTC", "BCH", "XLM", "ALGO", "ATOM", "FTM", "NEAR", "SAND", "MANA", "APE"
}

# Supported timeframes
SUPPORTED_TIMEFRAMES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440, "3d": 4320, "1w": 10080, "1M": 43200
}


def validate_symbol(symbol: str, raise_error: bool = True) -> bool:
    """
    Validation symbol cryptocurrency
    
    Args:
        symbol: Symbol for validation (for example, "BTC")
        raise_error: Call exception when error
        
    Returns:
        True if symbol valid
        
    Raises:
        InvalidDataException: If symbol not is supported (when raise_error=True)
    """
    if not isinstance(symbol, str):
        if raise_error:
            raise InvalidDataException(f"Symbol must be string, got {type(symbol)}")
        return False
    
    symbol_upper = symbol.upper().strip()
    
    # Validation format (2-10 symbols, only letters and digits)
    if not re.match(r'^[A-Z0-9]{2,10}$', symbol_upper):
        if raise_error:
            raise InvalidDataException(f"Invalid symbol format: {symbol}")
        return False
    
    # Validation in list supported
    if symbol_upper not in SUPPORTED_SYMBOLS:
        if raise_error:
            raise InvalidDataException(
                f"Unsupported symbol: {symbol}. Supported: {', '.join(sorted(SUPPORTED_SYMBOLS))}"
            )
        return False
    
    return True


def validate_timeframe(timeframe: str, raise_error: bool = True) -> bool:
    """
    Validation timeframe
    
    Args:
        timeframe: Timeframe for validation (for example, "1h")
        raise_error: Call exception when error
        
    Returns:
        True if timeframe valid
        
    Raises:
        InvalidDataException: If timeframe not is supported
    """
    if not isinstance(timeframe, str):
        if raise_error:
            raise InvalidDataException(f"Timeframe must be string, got {type(timeframe)}")
        return False
    
    timeframe_lower = timeframe.lower().strip()
    
    if timeframe_lower not in SUPPORTED_TIMEFRAMES:
        if raise_error:
            raise InvalidDataException(
                f"Unsupported timeframe: {timeframe}. "
                f"Supported: {', '.join(sorted(SUPPORTED_TIMEFRAMES.keys()))}"
            )
        return False
    
    return True


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """
    Conversion timeframe in minutes
    
    Args:
        timeframe: Timeframe (for example, "1h", "4h", "1d")
        
    Returns:
        Number minutes
        
    Raises:
        InvalidDataException: When incorrect timeframe
    """
    validate_timeframe(timeframe)
    return SUPPORTED_TIMEFRAMES[timeframe.lower().strip()]


def parse_timeframe_to_seconds(timeframe: str) -> int:
    """
    Conversion timeframe in seconds
    
    Args:
        timeframe: Timeframe
        
    Returns:
        Number seconds
    """
    minutes = parse_timeframe_to_minutes(timeframe)
    return minutes * 60


def parse_timeframe_to_timedelta(timeframe: str) -> timedelta:
    """
    Conversion timeframe in timedelta
    
    Args:
        timeframe: Timeframe
        
    Returns:
        timedelta object
    """
    minutes = parse_timeframe_to_minutes(timeframe)
    return timedelta(minutes=minutes)


def normalize_symbol(symbol: str) -> str:
    """
    Normalization symbol cryptocurrency
    
    Args:
        symbol: Original symbol
        
    Returns:
        Normalized symbol (upper register, without spaces)
    """
    if not isinstance(symbol, str):
        raise InvalidDataException(f"Symbol must be string, got {type(symbol)}")
    
    return symbol.upper().strip()


def normalize_timeframe(timeframe: str) -> str:
    """
    Normalization timeframe
    
    Args:
        timeframe: Original timeframe
        
    Returns:
        Normalized timeframe (lower register, without spaces)
    """
    if not isinstance(timeframe, str):
        raise InvalidDataException(f"Timeframe must be string, got {type(timeframe)}")
    
    return timeframe.lower().strip()


def ensure_datetime(
    value: Union[datetime, pd.Timestamp, str, int, float],
    default_format: str = "%Y-%m-%d %H:%M:%S"
) -> datetime:
    """
    Conversion various types in datetime
    
    Args:
        value: Value for conversion
        default_format: Format for strings
        
    Returns:
        datetime object
        
    Raises:
        InvalidDataException: When error conversion
    """
    try:
        if isinstance(value, datetime):
            return value
        elif isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        elif isinstance(value, str):
            # Attempt parsing in ISO format first
            try:
                return pd.to_datetime(value).to_pydatetime()
            except:
                return datetime.strptime(value, default_format)
        elif isinstance(value, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(value)
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    except Exception as e:
        raise InvalidDataException(f"Cannot convert {value} to datetime: {e}")


def safe_divide(
    numerator: Union[int, float, Decimal], 
    denominator: Union[int, float, Decimal],
    default: Union[int, float, None] = None
) -> Union[float, None]:
    """
    Safe division with processing division on zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value by default when division on zero
        
    Returns:
        Result division or default value
    """
    try:
        if denominator == 0:
            return default
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def format_number(
    value: Union[int, float, Decimal],
    precision: int = 2,
    percentage: bool = False,
    thousands_sep: bool = True
) -> str:
    """
    Formatting numbers for mapping
    
    Args:
        value: Number for formatting
        precision: Number characters after comma
        percentage: Format as percent
        thousands_sep: Use separator thousands
        
    Returns:
        Formatted string
    """
    try:
        if value is None or math.isnan(float(value)):
            return "N/A"
        
        if percentage:
            value = float(value)
            if thousands_sep:
                return f"{value:,.{precision}f}%"
            else:
                return f"{value:.{precision}f}%"
        else:
            value = float(value)
            if thousands_sep:
                return f"{value:,.{precision}f}"
            else:
                return f"{value:.{precision}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_price(
    price: Union[int, float, Decimal],
    symbol: str = "USD"
) -> str:
    """
    Formatting price with automatic selection precision
    
    Args:
        price: Price
        symbol: Symbol currencies
        
    Returns:
        Formatted price
    """
    try:
        price = float(price)
        
        # Selection precision on basis size price
        if price >= 1000:
            precision = 2
        elif price >= 1:
            precision = 4
        elif price >= 0.01:
            precision = 6
        else:
            precision = 8
        
        formatted = f"{price:,.{precision}f}"
        return f"${formatted}" if symbol == "USD" else f"{formatted} {symbol}"
    
    except (ValueError, TypeError):
        return "N/A"


def round_to_precision(value: Union[float, Decimal], precision: int = 8) -> Decimal:
    """
    Rounding until specified accuracy with using Decimal
    
    Args:
        value: Value for rounding
        precision: Number characters after comma
        
    Returns:
        Rounded Decimal value
    """
    try:
        decimal_value = Decimal(str(value))
        quantize_exp = Decimal('0.' + '0' * (precision - 1) + '1')
        return decimal_value.quantize(quantize_exp, rounding=ROUND_HALF_UP)
    except:
        return Decimal('0')


def calculate_percentage_change(
    old_value: Union[int, float],
    new_value: Union[int, float]
) -> float:
    """
    Computation percentage changes
    
    Args:
        old_value: Old value
        new_value: New value
        
    Returns:
        Percentage change (-100% until +∞%)
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return ((new_value - old_value) / abs(old_value)) * 100


def validate_ohlcv_data(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """
    Validation OHLCV data
    
    Args:
        df: DataFrame with data
        required_cols: Required columns (by default OHLCV)
        
    Returns:
        True if data valid
        
    Raises:
        InvalidDataException: When invalid data
    """
    if required_cols is None:
        required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Validation presence columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise InvalidDataException(f"Missing required columns: {missing_cols}")
    
    # Validation on void
    if df.empty:
        raise InvalidDataException("DataFrame is empty")
    
    # Validation types data
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise InvalidDataException(f"Column {col} must be numeric")
    
    # Validation logic OHLC
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # High must be >= max(open, close)
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).any()
        if invalid_high:
            raise InvalidDataException("High price must be >= max(open, close)")
        
        # Low must be <= min(open, close)  
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).any()
        if invalid_low:
            raise InvalidDataException("Low price must be <= min(open, close)")
    
    # Validation on negative values
    for col in numeric_cols:
        if col in df.columns:
            if (df[col] < 0).any():
                raise InvalidDataException(f"Column {col} contains negative values")
    
    return True


def detect_outliers_iqr(
    series: pd.Series,
    multiplier: float = 1.5
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Detection outliers method IQR
    
    Args:
        series: Data for analysis
        multiplier: Multiplier for IQR (usually 1.5 or 3.0)
        
    Returns:
        Tuple (mask outliers, statistics)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers_mask = (series < lower_bound) | (series > upper_bound)
    
    stats = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers_count': outliers_mask.sum(),
        'outliers_percentage': (outliers_mask.sum() / len(series)) * 100
    }
    
    return outliers_mask, stats


def clean_numeric_data(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    fill_method: str = 'interpolate',
    remove_outliers: bool = False,
    outlier_method: str = 'iqr'
) -> pd.DataFrame:
    """
    Cleanup numeric data
    
    Args:
        df: DataFrame for cleanup
        columns: Columns for processing (all numeric by default)
        fill_method: Method filling gaps ('interpolate', 'forward', 'backward', 'median')
        remove_outliers: Delete outliers
        outlier_method: Method detection outliers ('iqr', 'zscore')
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
        
        # Filling gaps
        if df_clean[col].isna().any():
            if fill_method == 'interpolate':
                df_clean[col] = df_clean[col].interpolate()
            elif fill_method == 'forward':
                df_clean[col] = df_clean[col].fillna(method='ffill')
            elif fill_method == 'backward':
                df_clean[col] = df_clean[col].fillna(method='bfill')
            elif fill_method == 'median':
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Removal outliers
        if remove_outliers and len(df_clean[col]) > 10:
            if outlier_method == 'iqr':
                outliers_mask, _ = detect_outliers_iqr(df_clean[col])
                df_clean = df_clean[~outliers_mask]
            elif outlier_method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
                outliers_mask = z_scores > 3
                df_clean = df_clean[~outliers_mask]
    
    return df_clean


def create_time_features(
    timestamps: Union[pd.Series, List[datetime]],
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Creation temporal features from timestamp
    
    Args:
        timestamps: Temporal labels
        features: List features for creation
        
    Returns:
        DataFrame with temporal features
    """
    if isinstance(timestamps, list):
        timestamps = pd.Series(timestamps)
    
    if features is None:
        features = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year']
    
    df_features = pd.DataFrame(index=timestamps.index)
    timestamps = pd.to_datetime(timestamps)
    
    for feature in features:
        if feature == 'hour':
            df_features['hour'] = timestamps.dt.hour
        elif feature == 'day_of_week':
            df_features['day_of_week'] = timestamps.dt.dayofweek
        elif feature == 'day_of_month':
            df_features['day_of_month'] = timestamps.dt.day
        elif feature == 'month':
            df_features['month'] = timestamps.dt.month
        elif feature == 'quarter':
            df_features['quarter'] = timestamps.dt.quarter
        elif feature == 'year':
            df_features['year'] = timestamps.dt.year
        elif feature == 'is_weekend':
            df_features['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
        elif feature == 'is_month_start':
            df_features['is_month_start'] = timestamps.dt.is_month_start.astype(int)
        elif feature == 'is_month_end':
            df_features['is_month_end'] = timestamps.dt.is_month_end.astype(int)
        elif feature == 'is_quarter_start':
            df_features['is_quarter_start'] = timestamps.dt.is_quarter_start.astype(int)
        elif feature == 'is_quarter_end':
            df_features['is_quarter_end'] = timestamps.dt.is_quarter_end.astype(int)
    
    return df_features


def memory_usage_mb(df: pd.DataFrame) -> float:
    """
    Computation usage memory DataFrame in megabytes
    
    Args:
        df: DataFrame for analysis
        
    Returns:
        Usage memory in MB
    """
    return df.memory_usage(deep=True).sum() / 1024 / 1024


def optimize_dataframe_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimization usage memory DataFrame
    
    Args:
        df: DataFrame for optimization
        
    Returns:
        Tuple (optimized DataFrame, statistics)
    """
    initial_memory = memory_usage_mb(df)
    df_optimized = df.copy()
    
    stats = {
        'initial_memory_mb': initial_memory,
        'columns_optimized': [],
        'optimization_details': {}
    }
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if pd.api.types.is_integer_dtype(col_type):
            # Optimization integer numbers
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
                
            stats['columns_optimized'].append(col)
            stats['optimization_details'][col] = f"{col_type} -> {df_optimized[col].dtype}"
        
        elif pd.api.types.is_float_dtype(col_type):
            # Optimization numbers with floating point
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df_optimized[col] = df_optimized[col].astype(np.float32)
                stats['columns_optimized'].append(col)
                stats['optimization_details'][col] = f"{col_type} -> {df_optimized[col].dtype}"
    
    final_memory = memory_usage_mb(df_optimized)
    stats['final_memory_mb'] = final_memory
    stats['memory_reduction_mb'] = initial_memory - final_memory
    stats['memory_reduction_percent'] = ((initial_memory - final_memory) / initial_memory) * 100
    
    return df_optimized, stats