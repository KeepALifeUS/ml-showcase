"""
Specialized XGBoost Models
=========================

Domain-specific XGBoost models for different trading scenarios.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np
from .xgb_time_series import XGBoostTimeSeriesModel


class XGBoostVolatilityModel(XGBoostTimeSeriesModel):
    """Specialized XGBoost model for volatility prediction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Volatility-specific configuration
        self.volatility_focused = True


class XGBoostTrendModel(XGBoostTimeSeriesModel):
    """Specialized XGBoost model for trend prediction"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Trend-specific configuration
        self.trend_focused = True


class XGBoostAnomalyModel(XGBoostTimeSeriesModel):
    """Specialized XGBoost model for anomaly detection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Anomaly-specific configuration
        self.anomaly_focused = True