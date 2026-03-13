"""
Interaction Features for Tree Models
====================================

Feature interaction generators optimized for XGBoost models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from .feature_engineer import BaseFeatureGenerator, FeatureType


class InteractionType(Enum):
    """Types of interaction features"""
    PRODUCT = "product"
    RATIO = "ratio"
    DIFFERENCE = "difference"


@dataclass
class InteractionConfig:
    """Interaction features configuration"""
    max_degree: int = 2
    correlation_threshold: float = 0.1


class InteractionFeatures(BaseFeatureGenerator):
    """Interaction features generator"""
    
    def __init__(self, feature_config, interaction_config: Optional[InteractionConfig] = None):
        self.feature_config = feature_config
        self.config = interaction_config or InteractionConfig()
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
    
    def generate_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate interaction features"""
        features = {}
        numerical_cols = data.select_dtypes(include=[np.number]).columns[:5]  # Limit for performance
        
        # Simple interactions between the first few features
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                if len(features) < 20:  # Limit the count
                    features[f'{col1}_{col2}_product'] = data[col1] * data[col2]
                    features[f'{col1}_{col2}_ratio'] = data[col1] / (data[col2] + 1e-8)
        
        result_df = pd.DataFrame(features, index=data.index)
        self.feature_names_ = list(result_df.columns)
        for col in result_df.columns:
            self.feature_types_[col] = FeatureType.NUMERICAL.value
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_
    
    def get_feature_types(self) -> Dict[str, str]:
        return self.feature_types_