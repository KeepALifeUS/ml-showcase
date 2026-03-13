"""
Window Management for StateVector Construction
Time-Series Windowing

Handles 168-hour (7-day) sliding windows with:
- Timestamp alignment across multiple symbols
- Missing data detection and filling
- Window validation
- Memory-efficient sliding

Performance: <1ms for window operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
 """Configuration for WindowManager"""

 # Window parameters
 window_hours: int = 168 # 7 days
 step_hours: int = 1 # 1 hour step for sliding

 # Missing data handling
 fill_method: str = 'forward' # 'forward', 'backward', 'interpolate', 'zero'
 max_missing_ratio: float = 0.10 # Max 10% missing data allowed

 # Timestamp alignment
 align_timestamps: bool = True
 timestamp_tolerance_seconds: int = 60 # 1 minute tolerance

 # Validation
 validate_continuity: bool = True
 validate_ordering: bool = True


class SlidingWindow:
 """
 Sliding window for time-series data

 Manages efficient sliding over historical data for state vector construction.
 """

 def __init__(self, data: pd.DataFrame, window_hours: int = 168, step_hours: int = 1):
 """
 Initialize sliding window

 Args:
 data: DataFrame with timestamp column
 window_hours: Window size in hours
 step_hours: Step size for sliding
 """
 self.data = data
 self.window_hours = window_hours
 self.step_hours = step_hours

 # Ensure timestamp column exists
 if 'timestamp' not in data.columns:
 raise ValueError("DataFrame must have 'timestamp' column")

 # Sort by timestamp
 self.data = self.data.sort_values('timestamp').reset_index(drop=True)

 # Calculate window indices
 self._calculate_window_indices

 def _calculate_window_indices(self) -> None:
 """Calculate valid window start indices"""
 self.window_indices: List[int] = []

 for i in range(len(self.data) - self.window_hours + 1):
 # Check if window is continuous
 window_data = self.data.iloc[i:i+self.window_hours]
 if self._is_continuous_window(window_data):
 self.window_indices.append(i)

 def _is_continuous_window(self, window_data: pd.DataFrame) -> bool:
 """Check if window has continuous hourly data"""
 if len(window_data) != self.window_hours:
 return False

 timestamps = pd.to_datetime(window_data['timestamp'])
 time_diffs = timestamps.diff[1:] # Skip first NaT

 # Check if all gaps are approximately 1 hour
 expected_diff = pd.Timedelta(hours=1)
 tolerance = pd.Timedelta(minutes=5)

 return all(abs(diff - expected_diff) <= tolerance for diff in time_diffs)

 def get_window(self, index: int) -> pd.DataFrame:
 """Get window at specific index"""
 if index >= len(self.window_indices):
 raise IndexError(f"Window index {index} out of range (max {len(self.window_indices)-1})")

 start_idx = self.window_indices[index]
 return self.data.iloc[start_idx:start_idx+self.window_hours].copy

 def __len__(self) -> int:
 """Number of valid windows"""
 return len(self.window_indices)

 def __iter__(self):
 """Iterate over all valid windows"""
 for idx in range(len(self)):
 yield self.get_window(idx)


class WindowManager:
 """
 Window Manager for multi-symbol time-series data

 Handles:
 - Multi-symbol timestamp alignment
 - Missing data detection and filling
 - Window validation
 - Efficient windowing operations
 """

 def __init__(self, config: Optional[WindowConfig] = None):
 self.config = config or WindowConfig
 logger.info(f"WindowManager initialized: {self.config.window_hours}h window, {self.config.step_hours}h step")

 def prepare_windows(
 self,
 ohlcv_data: Dict[str, pd.DataFrame],
 timestamp: Optional[datetime] = None
 ) -> Dict[str, pd.DataFrame]:
 """
 Prepare aligned windows for all symbols

 Args:
 ohlcv_data: Dict of symbol -> DataFrame with OHLCV data
 timestamp: End timestamp for window (defaults to latest data)

 Returns:
 Dict of symbol -> DataFrame with exactly window_hours rows, aligned timestamps

 Raises:
 ValueError: If data quality is insufficient
 """
 # Validate inputs
 self._validate_inputs(ohlcv_data)

 # Align timestamps across symbols
 if self.config.align_timestamps:
 ohlcv_data = align_timestamps(
 ohlcv_data,
 tolerance_seconds=self.config.timestamp_tolerance_seconds
 )

 # Extract windows
 windows = {}
 for symbol, df in ohlcv_data.items:
 # Get last window_hours rows
 if timestamp is None:
 window = df.tail(self.config.window_hours).copy
 else:
 # Filter by timestamp
 window = df[df['timestamp'] <= timestamp].tail(self.config.window_hours).copy

 # Handle missing data
 if len(window) < self.config.window_hours:
 missing_ratio = 1.0 - (len(window) / self.config.window_hours)
 if missing_ratio > self.config.max_missing_ratio:
 raise ValueError(
 f"{symbol}: Insufficient data. "
 f"Got {len(window)}/{self.config.window_hours} rows "
 f"(missing {missing_ratio*100:.1f}%)"
 )

 # Fill missing data
 window = self._fill_missing_data(window, symbol)

 windows[symbol] = window

 # Validate window quality
 self._validate_windows(windows)

 return windows

 def _validate_inputs(self, ohlcv_data: Dict[str, pd.DataFrame]) -> None:
 """Validate input data"""
 if not ohlcv_data:
 raise ValueError("Empty ohlcv_data provided")

 required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
 for symbol, df in ohlcv_data.items:
 missing = set(required_cols) - set(df.columns)
 if missing:
 raise ValueError(f"{symbol}: Missing columns {missing}")

 if len(df) == 0:
 raise ValueError(f"{symbol}: Empty DataFrame")

 def _fill_missing_data(self, window: pd.DataFrame, symbol: str) -> pd.DataFrame:
 """Fill missing data in window"""
 target_rows = self.config.window_hours

 if len(window) >= target_rows:
 return window

 logger.warning(f"{symbol}: Filling {target_rows - len(window)} missing rows")

 # Create full timestamp range
 if len(window) > 0:
 end_time = pd.to_datetime(window['timestamp'].iloc[-1])
 else:
 end_time = datetime.now(timezone.utc)

 start_time = end_time - timedelta(hours=target_rows - 1)
 full_timestamps = pd.date_range(start=start_time, end=end_time, freq='1h')

 # Create full DataFrame
 full_df = pd.DataFrame({'timestamp': full_timestamps})

 # Merge with existing data
 window_filled = full_df.merge(window, on='timestamp', how='left')

 # Fill missing values
 if self.config.fill_method == 'forward':
 window_filled = window_filled.fillna(method='ffill')
 elif self.config.fill_method == 'backward':
 window_filled = window_filled.fillna(method='bfill')
 elif self.config.fill_method == 'zero':
 window_filled = window_filled.fillna(0.0)
 elif self.config.fill_method == 'interpolate':
 window_filled = window_filled.interpolate(method='linear')
 else:
 raise ValueError(f"Unknown fill_method: {self.config.fill_method}")

 # Fill any remaining NaNs with zero
 window_filled = window_filled.fillna(0.0)

 return window_filled

 def _validate_windows(self, windows: Dict[str, pd.DataFrame]) -> None:
 """Validate prepared windows"""
 # Check all windows have same length
 lengths = [len(df) for df in windows.values]
 if len(set(lengths)) > 1:
 raise ValueError(f"Window length mismatch: {dict(zip(windows.keys, lengths))}")

 # Check all have exactly window_hours rows
 if lengths[0] != self.config.window_hours:
 raise ValueError(f"Expected {self.config.window_hours} rows, got {lengths[0]}")

 # Validate timestamp continuity
 if self.config.validate_continuity:
 for symbol, df in windows.items:
 if not self._check_continuity(df):
 logger.warning(f"{symbol}: Timestamp continuity issues detected")

 # Validate timestamp ordering
 if self.config.validate_ordering:
 for symbol, df in windows.items:
 if not df['timestamp'].is_monotonic_increasing:
 raise ValueError(f"{symbol}: Timestamps not in ascending order")

 def _check_continuity(self, df: pd.DataFrame) -> bool:
 """Check if timestamps are continuous (1h gaps)"""
 timestamps = pd.to_datetime(df['timestamp'])
 time_diffs = timestamps.diff[1:] # Skip first NaT

 expected_diff = pd.Timedelta(hours=1)
 tolerance = pd.Timedelta(minutes=5)

 return all(abs(diff - expected_diff) <= tolerance for diff in time_diffs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def align_timestamps(
 ohlcv_data: Dict[str, pd.DataFrame],
 tolerance_seconds: int = 60
) -> Dict[str, pd.DataFrame]:
 """
 Align timestamps across multiple symbols

 Finds common timestamps and filters all DataFrames to those timestamps.

 Args:
 ohlcv_data: Dict of symbol -> DataFrame
 tolerance_seconds: Timestamp matching tolerance

 Returns:
 Dict of symbol -> DataFrame with aligned timestamps
 """
 if not ohlcv_data:
 return ohlcv_data

 # Convert all timestamps to datetime
 for symbol, df in ohlcv_data.items:
 df['timestamp'] = pd.to_datetime(df['timestamp'])

 # Find common timestamps (with tolerance)
 timestamp_sets = []
 for symbol, df in ohlcv_data.items:
 # Round to nearest minute for matching
 rounded = df['timestamp'].dt.round(f'{tolerance_seconds}s')
 timestamp_sets.append(set(rounded))

 # Intersection of all timestamps
 common_timestamps = set.intersection(*timestamp_sets)

 if not common_timestamps:
 logger.warning("No common timestamps found across symbols")
 return ohlcv_data

 # Filter each DataFrame to common timestamps
 aligned_data = {}
 for symbol, df in ohlcv_data.items:
 rounded = df['timestamp'].dt.round(f'{tolerance_seconds}s')
 mask = rounded.isin(common_timestamps)
 aligned_df = df[mask].copy

 if len(aligned_df) < len(df) * 0.9: # Lost >10% of data
 logger.warning(f"{symbol}: Alignment removed {len(df) - len(aligned_df)} rows")

 aligned_data[symbol] = aligned_df

 logger.info(f"Aligned {len(ohlcv_data)} symbols to {len(common_timestamps)} common timestamps")
 return aligned_data


def handle_missing_data(
 df: pd.DataFrame,
 method: str = 'forward',
 max_gap_hours: int = 4
) -> pd.DataFrame:
 """
 Handle missing data in time-series DataFrame

 Args:
 df: DataFrame with OHLCV data
 method: Fill method ('forward', 'backward', 'interpolate', 'zero')
 max_gap_hours: Maximum gap to fill (larger gaps = warning)

 Returns:
 DataFrame with missing data filled
 """
 original_len = len(df)

 # Detect gaps
 if 'timestamp' in df.columns:
 timestamps = pd.to_datetime(df['timestamp'])
 gaps = timestamps.diff[1:]
 large_gaps = gaps[gaps > pd.Timedelta(hours=max_gap_hours)]

 if len(large_gaps) > 0:
 logger.warning(f"Found {len(large_gaps)} gaps > {max_gap_hours}h")

 # Fill based on method
 if method == 'forward':
 df = df.fillna(method='ffill')
 elif method == 'backward':
 df = df.fillna(method='bfill')
 elif method == 'zero':
 df = df.fillna(0.0)
 elif method == 'interpolate':
 numeric_cols = df.select_dtypes(include=[np.number]).columns
 df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
 else:
 raise ValueError(f"Unknown fill method: {method}")

 # Fill any remaining NaNs
 df = df.fillna(0.0)

 filled_count = df.isna.sum.sum
 if filled_count > 0:
 logger.debug(f"Filled {filled_count} missing values using {method}")

 return df
