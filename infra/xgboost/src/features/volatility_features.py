"""
Volatility Features for Crypto Trading
======================================

Volatility-based feature generators for time series.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from .feature_engineer import BaseFeatureGenerator, FeatureType


class VolatilityType(Enum):
    """Types of volatility"""
    REALIZED = "realized"
    GARCH = "garch"
    PARKINSON = "parkinson"


@dataclass
class VolatilityConfig:
    """Volatility features configuration"""
    windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    enable_realized: bool = True
    enable_garch: bool = True


class VolatilityFeatures(BaseFeatureGenerator):
    """Volatility features generator"""
    
    def __init__(self, feature_config, volatility_config: Optional[VolatilityConfig] = None):
        self.feature_config = feature_config
        self.config = volatility_config or VolatilityConfig()
        self.feature_names_: List[str] = []
        self.feature_types_: Dict[str, str] = {}
    
    def generate_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Generate volatility features"""
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close']
            returns = close_prices.pct_change()
            
            for window in self.config.windows:
                if self.config.enable_realized:
                    # Realized volatility
                    realized_vol = returns.rolling(window).std() * np.sqrt(252)
                    features[f'realized_vol_{window}'] = realized_vol
                    
                    # Volatility of volatility
                    features[f'vol_of_vol_{window}'] = realized_vol.rolling(window).std()
        
        if 'high' in data.columns and 'low' in data.columns:
            # Parkinson volatility estimator
            for window in self.config.windows:
                parkinson_vol = np.sqrt(
                    (1 / (4 * np.log(2))) * 
                    (np.log(data['high'] / data['low'])).rolling(window).mean()
                ) * np.sqrt(252)
                features[f'parkinson_vol_{window}'] = parkinson_vol
        
        result_df = pd.DataFrame(features, index=data.index)
        self.feature_names_ = list(result_df.columns)
        for col in result_df.columns:
            self.feature_types_[col] = FeatureType.NUMERICAL.value
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names_
    
    def get_feature_types(self) -> Dict[str, str]:
        return self.feature_types_