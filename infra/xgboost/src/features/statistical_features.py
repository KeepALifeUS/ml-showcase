"""
Statistical Features for XGBoost Models
=======================================

Statistical and mathematical feature generators optimized for tree-based models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from .feature_engineer import BaseFeatureGenerator, FeatureType


class StatFeatureType(Enum):
    """Types of statistical features"""
    MOMENTS = "moments"
    QUANTILES = "quantiles"
    ENTROPY = "entropy"


@dataclass
class StatisticalConfig:
    """Statistical features configuration"""
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    enable_moments: bool = True
    enable_quantiles: bool = True


class StatisticalFeatures(BaseFeatureGenerator):
    """Statistical features generator"""
    
    def __init__(self, feature_config, statistical_config: Optional[StatisticalConfig] = None):
        self.feature_config = feature_config
        self.config = statistical_config or StatisticalConfig()
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
    
    def generate_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate statistical features"""
        features = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            for window in self.config.windows:
                if self.config.enable_moments:
                    features[f'{col}_skew_{window}'] = data[col].rolling(window).skew()
                    features[f'{col}_kurt_{window}'] = data[col].rolling(window).kurt()
                
                if self.config.enable_quantiles:
                    features[f'{col}_q25_{window}'] = data[col].rolling(window).quantile(0.25)
                    features[f'{col}_q75_{window}'] = data[col].rolling(window).quantile(0.75)
        
        result_df = pd.DataFrame(features, index=data.index)
        self.feature_names_ = list(result_df.columns)
        for col in result_df.columns:
            self.feature_types_[col] = FeatureType.NUMERICAL.value
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_
    
    def get_feature_types(self) -> Dict[str, str]:
        return self.feature_types_