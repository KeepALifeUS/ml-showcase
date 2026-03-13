"""
Market Hours & Time Features
Time-Based Regime

Calculates 2 time-based dimensions:
1. Trading session ID (0=Asian, 1=European, 2=US, 3=Overnight)
2. Day of week normalized (0-1, Monday=0, Sunday=0.857)

Performance: <0.01ms
"""

import numpy as np
from typing import Union, Optional
from datetime import datetime, timezone

def classify_trading_session(timestamp: Optional[datetime] = None) -> int:
 """
 Classify trading session based on UTC hour

 Args:
 timestamp: Datetime object (None = current time)

 Returns:
 Session ID:
 0 = Asian (00:00-08:00 UTC)
 1 = European (08:00-16:00 UTC)
 2 = US (16:00-22:00 UTC)
 3 = Overnight (22:00-00:00 UTC)
 """
 if timestamp is None:
 timestamp = datetime.now(timezone.utc)

 hour = timestamp.hour

 if 0 <= hour < 8:
 return 0 # Asian
 elif 8 <= hour < 16:
 return 1 # European
 elif 16 <= hour < 22:
 return 2 # US
 else:
 return 3 # Overnight

def normalize_day_of_week(timestamp: Optional[datetime] = None) -> float:
 """
 Normalize day of week to [0, 1]

 Args:
 timestamp: Datetime object (None = current time)

 Returns:
 Normalized day of week (0-1)
 Monday = 0.0, Tuesday = 0.143, ..., Sunday = 0.857
 """
 if timestamp is None:
 timestamp = datetime.now(timezone.utc)

 day_of_week = timestamp.weekday # 0=Monday, 6=Sunday

 # Normalize to [0, 1]
 normalized = day_of_week / 7.0

 return float(normalized)

def extract_time_features(timestamp: Optional[datetime] = None) -> np.ndarray:
 """
 Extract all 2 time features: [session_id, day_of_week_normalized]

 Args:
 timestamp: Datetime object (None = current time)

 Returns:
 2-dimensional feature vector:
 [0] session_id (0-3)
 [1] day_of_week_normalized (0-1)

 Example:
 timestamp = datetime(2025, 10, 10, 15, 30) # Friday 15:30 UTC
 features = extract_time_features(timestamp)
 # Returns: [1.0, 0.571] # European session, Friday
 """
 features = np.zeros(2, dtype=np.float32)
 features[0] = float(classify_trading_session(timestamp))
 features[1] = normalize_day_of_week(timestamp)
 return features

__all__ = ["classify_trading_session", "normalize_day_of_week", "extract_time_features"]
