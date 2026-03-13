"""
Temporal Embeddings - Time-Based Feature Encoding
Time-Series Learning

Encodes temporal information as 10-dimensional features:
1-2. Hour of day (sin, cos) - Cyclic encoding of 0-23
3-4. Day of week (sin, cos) - Cyclic encoding of 0-6
5-6. Week of year (sin, cos) - Cyclic encoding of 1-53
7. Month of year - Normalized 0-1
8. Quarter (one-hot style) - 0, 0.33, 0.67, 1.0
9. Year fraction - Days since year start / 365
10. Is weekend - Binary 0/1

Why cyclic encoding?
- Preserves continuity (23:00 → 00:00 are close)
- Neural networks learn better with sin/cos than raw integers
- Standard technique for time features in ML

Performance: <0.05ms per extraction
"""

import numpy as np
from typing import Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def extract_temporal_embeddings(
 timestamp: Optional[datetime] = None
) -> np.ndarray:
 """
 Extract 10-dimensional temporal embeddings

 Args:
 timestamp: Timestamp to encode (defaults to current UTC time)

 Returns:
 np.ndarray: 10-dimensional temporal feature vector
 [0]: Hour sin encoding
 [1]: Hour cos encoding
 [2]: Day of week sin encoding
 [3]: Day of week cos encoding
 [4]: Week of year sin encoding
 [5]: Week of year cos encoding
 [6]: Month (normalized 0-1)
 [7]: Quarter (0, 0.33, 0.67, 1.0)
 [8]: Year fraction (days / 365)
 [9]: Is weekend (0 or 1)

 Example:
 >>> from datetime import datetime
 >>> ts = datetime(2025, 10, 10, 15, 30, 0) # Friday 3:30 PM
 >>> features = extract_temporal_embeddings(ts)
 >>> features.shape
 (10,)
 """
 if timestamp is None:
 timestamp = datetime.now(timezone.utc)

 features = np.zeros(10, dtype=np.float32)

 # 1-2. Hour of day (0-23) - cyclic encoding
 hour_sin, hour_cos = encode_hour_of_day(timestamp)
 features[0] = hour_sin
 features[1] = hour_cos

 # 3-4. Day of week (0-6) - cyclic encoding
 dow_sin, dow_cos = encode_day_of_week(timestamp)
 features[2] = dow_sin
 features[3] = dow_cos

 # 5-6. Week of year (1-53) - cyclic encoding
 week_sin, week_cos = encode_week_of_year(timestamp)
 features[4] = week_sin
 features[5] = week_cos

 # 7. Month of year (1-12) - normalized
 month = encode_month_of_year(timestamp)
 features[6] = month

 # 8. Quarter (1-4) - categorical as continuous
 quarter = (timestamp.month - 1) // 3
 features[7] = quarter / 3.0 # 0, 0.33, 0.67, 1.0

 # 9. Year fraction (day of year / 365)
 day_of_year = timestamp.timetuple.tm_yday
 features[8] = day_of_year / 365.0

 # 10. Is weekend (Saturday=5, Sunday=6)
 is_weekend = 1.0 if timestamp.weekday >= 5 else 0.0
 features[9] = is_weekend

 return features


def encode_hour_of_day(timestamp: datetime) -> tuple[float, float]:
 """
 Encode hour of day as cyclic features (sin, cos)

 Hour 0-23 mapped to circle:
 - 0:00 → angle 0°
 - 6:00 → angle 90°
 - 12:00 → angle 180°
 - 18:00 → angle 270°

 Returns:
 Tuple of (sin, cos) values
 """
 hour = timestamp.hour + timestamp.minute / 60.0 # Include minutes for precision
 angle = 2 * np.pi * hour / 24.0

 return float(np.sin(angle)), float(np.cos(angle))


def encode_day_of_week(timestamp: datetime) -> tuple[float, float]:
 """
 Encode day of week as cyclic features (sin, cos)

 Monday=0 → Sunday=6 mapped to circle:
 - Monday → angle 0°
 - Thursday → angle 180°
 - Sunday → angle ~309°

 Returns:
 Tuple of (sin, cos) values
 """
 day = timestamp.weekday # 0=Monday, 6=Sunday
 angle = 2 * np.pi * day / 7.0

 return float(np.sin(angle)), float(np.cos(angle))


def encode_week_of_year(timestamp: datetime) -> tuple[float, float]:
 """
 Encode week of year as cyclic features (sin, cos)

 Week 1-53 mapped to circle (ISO calendar week)

 Returns:
 Tuple of (sin, cos) values
 """
 _, week, _ = timestamp.isocalendar
 angle = 2 * np.pi * week / 53.0

 return float(np.sin(angle)), float(np.cos(angle))


def encode_month_of_year(timestamp: datetime) -> float:
 """
 Encode month of year as normalized value 0-1

 January=1 → 0.0
 December=12 → 1.0

 Returns:
 Normalized month value
 """
 return (timestamp.month - 1) / 11.0 # 0 to 1


def encode_time_of_day_category(timestamp: datetime) -> int:
 """
 Encode time of day as categorical (for analysis, not used in state vector)

 Returns:
 0: Night (00:00-06:00)
 1: Morning (06:00-12:00)
 2: Afternoon (12:00-18:00)
 3: Evening (18:00-24:00)
 """
 hour = timestamp.hour
 if 0 <= hour < 6:
 return 0 # Night
 elif 6 <= hour < 12:
 return 1 # Morning
 elif 12 <= hour < 18:
 return 2 # Afternoon
 else:
 return 3 # Evening


def encode_trading_session(timestamp: datetime) -> int:
 """
 Encode trading session (for crypto, 24/7 but with activity patterns)

 Returns:
 0: Asian session (00:00-08:00 UTC)
 1: European session (08:00-16:00 UTC)
 2: US session (16:00-24:00 UTC)
 """
 hour = timestamp.hour
 if 0 <= hour < 8:
 return 0 # Asian
 elif 8 <= hour < 16:
 return 1 # European
 else:
 return 2 # US
