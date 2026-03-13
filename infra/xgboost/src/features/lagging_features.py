"""
Lagging Features for Time Series Models
=======================================

Time-based lagged feature generators for XGBoost time series models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from .feature_engineer import BaseFeatureGenerator, FeatureType


class LagType(Enum):
    """Types of lag features"""
    SIMPLE = "simple"
    ROLLING = "rolling"
    DIFFERENCE = "difference"


@dataclass
class LagConfig:
    """Lag features configuration"""
    max_lags: int = 20
    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])


class LaggingFeatures(BaseFeatureGenerator):
    """Lag features generator"""
    
    def __init__(self, feature_config, lag_config: Optional[LagConfig] = None):
        self.feature_config = feature_config
        self.config = lag_config or LagConfig()
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
    
    def generate_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate lag features"""
        features = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            for lag in self.config.lag_windows:
                if lag <= self.config.max_lags:
                    features[f'{col}_lag_{lag}'] = data[col].shift(lag)
                    features[f'{col}_diff_{lag}'] = data[col].diff(lag)
        
        result_df = pd.DataFrame(features, index=data.index)
        self.feature_names_ = list(result_df.columns)
        for col in result_df.columns:
            self.feature_types_[col] = FeatureType.NUMERICAL.value
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_
    
    def get_feature_types(self) -> Dict[str, str]:
        return self.feature_types_