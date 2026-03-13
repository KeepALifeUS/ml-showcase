"""
Advanced Feature Engineering for XGBoost Tree Models
====================================================

Comprehensive feature engineering pipeline specifically optimized for tree-based models
and time series data. Implements enterprise patterns for
production-ready feature generation.

Key Features:
- Tree-optimized feature transformations
- Time Series-specific indicators
- Time series feature engineering
- Automated feature selection
- Feature interaction detection
- Performance optimization with caching

Author: ML XGBoost Contributors
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import joblib
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
    KBinsDiscretizer, LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_regression, f_classif,
    mutual_info_regression, mutual_info_classif,
    VarianceThreshold, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA
from sklearn.impute import SimpleImputer, KNNImputer
import pandas_ta as ta
import talib
import scipy.stats as stats
from scipy.signal import find_peaks
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numba
from numba import jit, prange


class FeatureType(Enum):
    """Feature types optimized for tree-based models"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    TEMPORAL = "temporal"
    INTERACTION = "interaction"
    RATIO = "ratio"
    DIFFERENCE = "difference"
    LOG_TRANSFORM = "log_transform"
    BINNED = "binned"


class ScalingMethod(Enum):
    """Data scaling methods"""
    NONE = "none"
    STANDARD = "standard"  # Z-score normalization
    ROBUST = "robust"  # Median and IQR
    MINMAX = "minmax"  # Min-max scaling
    QUANTILE = "quantile"  # Quantile transformation
    LOG = "log"  # Log transformation
    SQRT = "sqrt"  # Square root transformation


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""

    # Main parameters
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_lag_features: bool = True
    enable_interaction_features: bool = True
    enable_volatility_features: bool = True
    
    # Technical indicators
    technical_periods: List[int] = field(default_factory=lambda: [5, 10, 14, 20, 30, 50])
    enable_oscillators: bool = True
    enable_trend_indicators: bool = True
    enable_volume_indicators: bool = True
    
    # Statistical features
    statistical_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    enable_moments: bool = True  # skewness, kurtosis
    enable_quantiles: bool = True
    enable_entropy: bool = True
    
    # Lag features
    max_lags: int = 20
    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 20])
    enable_rolling_lags: bool = True
    
    # Interaction features
    max_interaction_degree: int = 2
    interaction_threshold: float = 0.1  # Minimum correlation for interaction
    auto_detect_interactions: bool = True
    
    # Volatility features  
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 30])
    enable_garch_features: bool = True
    enable_realized_volatility: bool = True
    
    # Feature selection
    enable_feature_selection: bool = True
    feature_selection_method: str = "mutual_info"  # mutual_info, f_score, variance
    feature_selection_k: int = 100  # Top K features
    variance_threshold: float = 0.01
    
    # Data preprocessing
    handle_missing: str = "forward_fill"  # forward_fill, interpolate, knn, drop
    scaling_method: ScalingMethod = ScalingMethod.ROBUST
    remove_outliers: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    
    # Performance optimization
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    
    # Categorical features
    categorical_encoding: str = "target"  # target, onehot, label, binary
    max_cardinality: int = 50
    enable_binning: bool = True
    binning_strategy: str = "quantile"  # quantile, uniform, kmeans


@dataclass
class FeatureMetadata:
    """Metadata about created features"""
    
    feature_names: List[str] = field(default_factory=list)
    feature_types: Dict[str, str] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    creation_time: Dict[str, float] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    selected_features: List[str] = field(default_factory=list)
    removed_features: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'feature_names': self.feature_names,
            'feature_types': self.feature_types,
            'feature_importance': self.feature_importance,
            'creation_time': self.creation_time,
            'feature_stats': self.feature_stats,
            'selected_features': self.selected_features,
            'removed_features': self.removed_features,
        }


class BaseFeatureGenerator(ABC):
    """Base class for feature generators - enterprise pattern"""
    
    @abstractmethod
    def generate_features(
        self, 
        data: pd.DataFrame, 
        target: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Generate features"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of created features"""
        pass

    @abstractmethod
    def get_feature_types(self) -> Dict[str, str]:
        """Get feature types"""
        pass


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Advanced Feature Engineer for XGBoost models

    Implements enterprise patterns:
    - Modular architecture with pluggable generators
    - Caching for performance
    - Parallel data processing
    - Data quality validation and monitoring
    - Automatic optimization for tree-based models
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        
        # Object state
        self.feature_generators_: List[BaseFeatureGenerator] = []
        self.scalers_: Dict[str, Any] = {}
        self.feature_selector_: Optional[Any] = None
        self.metadata_: FeatureMetadata = FeatureMetadata()
        self.is_fitted_: bool = False
        
        # Caching
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._stats_cache: Dict[str, Dict[str, Any]] = {}
        
        self.console = Console()
        self._setup_logging()
        self._initialize_generators()
    
    def _setup_logging(self):
        """Set up logging"""
        logger.add(
            f"logs/feature_engineer_{datetime.now():%Y%m%d}.log",
            rotation="daily",
            retention="30 days", 
            level="INFO"
        )
    
    def _initialize_generators(self):
        """Initialize feature generators"""

        logger.info("🔧 Initializing feature generators...")

        # Import generators
        from .technical_indicators import TechnicalIndicators
        from .statistical_features import StatisticalFeatures
        from .lagging_features import LaggingFeatures
        from .interaction_features import InteractionFeatures
        from .volatility_features import VolatilityFeatures
        
        # Create generators based on configuration
        if self.config.enable_technical_indicators:
            self.feature_generators_.append(TechnicalIndicators(self.config))
        
        if self.config.enable_statistical_features:
            self.feature_generators_.append(StatisticalFeatures(self.config))
        
        if self.config.enable_lag_features:
            self.feature_generators_.append(LaggingFeatures(self.config))
        
        if self.config.enable_interaction_features:
            self.feature_generators_.append(InteractionFeatures(self.config))
        
        if self.config.enable_volatility_features:
            self.feature_generators_.append(VolatilityFeatures(self.config))
        
        logger.info(f"✅ Initialized {len(self.feature_generators_)} generators")
    
    def _validate_input_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""

        logger.info("🔍 Validating input data...")
        
        X = X.copy()
        
        # Check for empty data
        if X.empty:
            raise ValueError("Input data is empty")

        # Check for duplicate columns
        if X.columns.duplicated().any():
            logger.warning("⚠️ Duplicate columns found, removing...")
            X = X.loc[:, ~X.columns.duplicated()]
        
        # Handle missing values
        missing_ratio = X.isnull().sum() / len(X)
        high_missing_cols = missing_ratio[missing_ratio > 0.5].index.tolist()
        
        if high_missing_cols:
            logger.warning(f"⚠️ Removing columns with >50% missing values: {high_missing_cols}")
            X = X.drop(columns=high_missing_cols)
        
        # Handle remaining missing values
        if self.config.handle_missing == "forward_fill":
            X = X.fillna(method='ffill').fillna(method='bfill')
        elif self.config.handle_missing == "interpolate":
            X = X.interpolate().fillna(method='bfill')
        elif self.config.handle_missing == "drop":
            X = X.dropna()
        
        # Handle inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logger.info(f"✅ Data validated: {X.shape}")
        
        return X
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect feature types"""
        
        feature_types = {}
        
        for col in X.columns:
            dtype = X[col].dtype
            unique_values = X[col].nunique()
            unique_ratio = unique_values / len(X)
            
            if dtype in ['object', 'category']:
                if unique_values <= self.config.max_cardinality:
                    feature_types[col] = FeatureType.CATEGORICAL.value
                else:
                    feature_types[col] = FeatureType.CATEGORICAL.value  # Large cardinality
            elif dtype == 'bool':
                feature_types[col] = FeatureType.BINARY.value
            elif unique_values == 2:
                feature_types[col] = FeatureType.BINARY.value
            elif unique_values < 10 and unique_ratio < 0.1:
                feature_types[col] = FeatureType.ORDINAL.value
            else:
                feature_types[col] = FeatureType.NUMERICAL.value
        
        return feature_types
    
    def _remove_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers"""
        
        if not self.config.remove_outliers:
            return X
        
        logger.info(f"🧹 Removing outliers using {self.config.outlier_method} method...")
        
        X_clean = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if self.config.outlier_method == "iqr":
            for col in numerical_cols:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Clipping instead of removal to preserve data
                X_clean[col] = X[col].clip(lower_bound, upper_bound)
        
        elif self.config.outlier_method == "zscore":
            for col in numerical_cols:
                z_scores = np.abs(stats.zscore(X[col]))
                # Clip values beyond 3 standard deviations
                threshold_indices = z_scores > 3
                if threshold_indices.any():
                    median_val = X[col].median()
                    X_clean.loc[threshold_indices, col] = median_val
        
        logger.info("✅ Outliers processed")
        
        return X_clean
    
    def _create_base_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Create base features using all generators"""

        logger.info("🏗️ Creating features...")
        
        all_features = [X]  # Start with original features
        
        if self.config.parallel_processing:
            # Parallel feature generation
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                
                future_to_generator = {
                    executor.submit(generator.generate_features, X, y): generator
                    for generator in self.feature_generators_
                }
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                ) as progress:
                    
                    task = progress.add_task("Generating features...", total=len(self.feature_generators_))
                    
                    for future in as_completed(future_to_generator):
                        generator = future_to_generator[future]
                        
                        try:
                            features = future.result()
                            if features is not None and not features.empty:
                                all_features.append(features)
                                
                                # Update metadata
                                gen_feature_names = generator.get_feature_names()
                                gen_feature_types = generator.get_feature_types()
                                
                                self.metadata_.feature_names.extend(gen_feature_names)
                                self.metadata_.feature_types.update(gen_feature_types)
                                
                            logger.info(f"✅ {generator.__class__.__name__}: {len(features.columns) if features is not None else 0} features")

                        except Exception as e:
                            logger.error(f"❌ Error in {generator.__class__.__name__}: {e}")
                        
                        progress.advance(task)
        else:
            # Sequential generation
            for generator in self.feature_generators_:
                try:
                    features = generator.generate_features(X, y)
                    if features is not None and not features.empty:
                        all_features.append(features)
                        
                        gen_feature_names = generator.get_feature_names()
                        gen_feature_types = generator.get_feature_types()
                        
                        self.metadata_.feature_names.extend(gen_feature_names)
                        self.metadata_.feature_types.update(gen_feature_types)
                        
                    logger.info(f"✅ {generator.__class__.__name__}: {len(features.columns) if features is not None else 0} features")

                except Exception as e:
                    logger.error(f"❌ Error in {generator.__class__.__name__}: {e}")
        
        # Combine all features
        if len(all_features) > 1:
            X_features = pd.concat(all_features, axis=1)
            
            # Remove duplicate columns
            X_features = X_features.loc[:, ~X_features.columns.duplicated()]
        else:
            X_features = X
        
        logger.info(f"🎯 Created {len(X_features.columns)} features (was {len(X.columns)})")
        
        return X_features
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply data scaling"""
        
        if self.config.scaling_method == ScalingMethod.NONE:
            return X
        
        logger.info(f"📏 Applying scaling: {self.config.scaling_method.value}")
        
        X_scaled = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col not in self.scalers_:
                if self.config.scaling_method == ScalingMethod.STANDARD:
                    scaler = StandardScaler()
                elif self.config.scaling_method == ScalingMethod.ROBUST:
                    scaler = RobustScaler()
                elif self.config.scaling_method == ScalingMethod.MINMAX:
                    scaler = MinMaxScaler()
                elif self.config.scaling_method == ScalingMethod.QUANTILE:
                    scaler = QuantileTransformer()
                elif self.config.scaling_method == ScalingMethod.LOG:
                    # Log transform (add 1 to handle zeros)
                    X_scaled[col] = np.log1p(X[col] - X[col].min() + 1)
                    continue
                elif self.config.scaling_method == ScalingMethod.SQRT:
                    # Square root transform
                    X_scaled[col] = np.sqrt(X[col] - X[col].min() + 1)
                    continue
                
                self.scalers_[col] = scaler
                X_scaled[col] = scaler.fit_transform(X[[col]]).flatten()
            else:
                X_scaled[col] = self.scalers_[col].transform(X[[col]]).flatten()
        
        return X_scaled
    
    def _apply_binning(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binning for categorical features"""
        
        if not self.config.enable_binning:
            return X
        
        logger.info("🗂️ Applying binning for categorical features...")
        
        X_binned = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Create binned versions of numerical features
            try:
                if self.config.binning_strategy == "quantile":
                    X_binned[f'{col}_binned'] = pd.qcut(
                        X[col], q=10, labels=False, duplicates='drop'
                    )
                elif self.config.binning_strategy == "uniform":
                    X_binned[f'{col}_binned'] = pd.cut(
                        X[col], bins=10, labels=False, duplicates='drop'
                    )
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to apply binning to {col}: {e}")
        
        return X_binned
    
    def _select_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Select most important features"""
        
        if not self.config.enable_feature_selection or y is None:
            return X
        
        logger.info(f"🎯 Selecting features using {self.config.feature_selection_method} method...")
        
        # Remove features with low variance
        variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
        X_variance_selected = pd.DataFrame(
            variance_selector.fit_transform(X),
            columns=X.columns[variance_selector.get_support()],
            index=X.index
        )
        
        removed_by_variance = set(X.columns) - set(X_variance_selected.columns)
        if removed_by_variance:
            logger.info(f"🗑️ Removed by variance threshold: {len(removed_by_variance)} features")
            self.metadata_.removed_features.extend(list(removed_by_variance))
        
        # Main feature selection
        if self.config.feature_selection_method == "mutual_info":
            if y.dtype == 'object' or y.nunique() < 10:
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
            else:
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
        elif self.config.feature_selection_method == "f_score":
            if y.dtype == 'object' or y.nunique() < 10:
                selector = SelectKBest(
                    score_func=f_classif,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
            else:
                selector = SelectKBest(
                    score_func=f_regression,
                    k=min(self.config.feature_selection_k, len(X_variance_selected.columns))
                )
        else:
            return X_variance_selected
        
        self.feature_selector_ = selector
        X_selected = pd.DataFrame(
            selector.fit_transform(X_variance_selected, y),
            columns=X_variance_selected.columns[selector.get_support()],
            index=X.index
        )
        
        # Update metadata
        selected_features = list(X_selected.columns)
        removed_features = set(X_variance_selected.columns) - set(selected_features)
        
        self.metadata_.selected_features = selected_features
        self.metadata_.removed_features.extend(list(removed_features))
        
        # Save feature importance
        if hasattr(selector, 'scores_'):
            for i, feature in enumerate(X_variance_selected.columns):
                if selector.get_support()[i]:
                    self.metadata_.feature_importance[feature] = selector.scores_[i]
        
        logger.info(f"✅ Selected {len(selected_features)} out of {len(X.columns)} features")
        
        return X_selected
    
    def _calculate_feature_stats(self, X: pd.DataFrame):
        """Calculate feature statistics"""

        logger.info("📊 Calculating feature statistics...")
        
        for col in X.columns:
            stats_dict = {
                'dtype': str(X[col].dtype),
                'missing_ratio': X[col].isnull().sum() / len(X),
                'unique_values': X[col].nunique(),
                'unique_ratio': X[col].nunique() / len(X),
            }
            
            if X[col].dtype in ['int64', 'float64']:
                stats_dict.update({
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'q25': X[col].quantile(0.25),
                    'median': X[col].median(),
                    'q75': X[col].quantile(0.75),
                    'skewness': X[col].skew(),
                    'kurtosis': X[col].kurtosis(),
                })
            
            self.metadata_.feature_stats[col] = stats_dict
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> 'FeatureEngineer':
        """
        Fit the feature engineer

        Args:
            X: Input data
            y: Target variable (optional)
        """

        start_time = time.time()
        logger.info("🚀 Starting Feature Engineer fitting...")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        
        # Data validation
        X = self._validate_input_data(X)

        # Detect feature types
        base_feature_types = self._detect_feature_types(X)
        self.metadata_.feature_types.update(base_feature_types)
        
        # Remove outliers
        X = self._remove_outliers(X)

        # Create features
        X_features = self._create_base_features(X, y)

        # Apply binning
        X_features = self._apply_binning(X_features)

        # Scaling
        X_features = self._apply_scaling(X_features)

        # Feature selection
        X_final = self._select_features(X_features, y)

        # Calculate statistics
        self._calculate_feature_stats(X_final)
        
        # Correlation matrix
        if len(X_final.columns) <= 100:  # Limit for performance
            self.metadata_.correlation_matrix = X_final.corr()
        
        self.is_fitted_ = True
        
        training_time = time.time() - start_time
        logger.info(f"✅ Feature Engineer fitted in {training_time:.2f}s")
        logger.info(f"📊 Total features: {len(X_final.columns)}")
        
        self._display_summary()
        
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Apply feature engineering to new data

        Args:
            X: New data

        Returns:
            Transformed data
        """

        if not self.is_fitted_:
            raise ValueError("Feature Engineer is not fitted. Call fit() first.")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Data validation
        X = self._validate_input_data(X)

        # Remove outliers
        X = self._remove_outliers(X)

        # Create features (without target)
        X_features = self._create_base_features(X, y=None)
        
        # Apply binning
        X_features = self._apply_binning(X_features)

        # Scaling
        X_features = self._apply_scaling(X_features)

        # Feature selection (apply already fitted selector)
        if self.feature_selector_ is not None:
            # Take only columns that were present during fitting
            available_features = [col for col in self.metadata_.selected_features if col in X_features.columns]
            X_final = X_features[available_features]
        else:
            X_final = X_features
        
        return X_final
    
    def fit_transform(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> pd.DataFrame:
        """Fit and transform in a single call"""
        
        self.fit(X, y)
        return self.transform(X)
    
    def _display_summary(self):
        """Display results summary"""
        
        table = Table(title="🎯 FEATURE ENGINEERING SUMMARY")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total features", str(len(self.metadata_.feature_names)))
        table.add_row("Selected features", str(len(self.metadata_.selected_features)))
        table.add_row("Removed features", str(len(self.metadata_.removed_features)))
        table.add_row("Active generators", str(len(self.feature_generators_)))

        # Statistics by feature type
        type_counts = {}
        for feature_type in self.metadata_.feature_types.values():
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1
        
        for ftype, count in type_counts.items():
            table.add_row(f"Type {ftype}", str(count))
        
        self.console.print(table)
    
    def get_feature_importance(self, top_k: Optional[int] = None) -> pd.DataFrame:
        """Get feature importance"""

        if not self.metadata_.feature_importance:
            logger.warning("⚠️ Feature importance not calculated")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in self.metadata_.feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        if top_k:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def plot_feature_importance(
        self, 
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """Visualize feature importance"""
        
        importance_df = self.get_feature_importance(top_k)
        
        if importance_df.empty:
            logger.warning("⚠️ No feature importance data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(
            range(len(importance_df)), 
            importance_df['importance'],
            color='skyblue',
            alpha=0.7
        )
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Feature Importance Score')
        ax.set_title(f'Top {top_k} Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add values
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Feature importance plot saved: {save_path}")
        else:
            plt.show()
    
    def plot_correlation_heatmap(
        self,
        features: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Correlation heatmap"""

        if self.metadata_.correlation_matrix is None:
            logger.warning("⚠️ Correlation matrix not calculated")
            return
        
        corr_matrix = self.metadata_.correlation_matrix
        
        if features:
            available_features = [f for f in features if f in corr_matrix.columns]
            corr_matrix = corr_matrix.loc[available_features, available_features]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(
            corr_matrix,
            annot=False,
            cmap='coolwarm',
            center=0,
            square=True,
            ax=ax
        )
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Correlation matrix saved: {save_path}")
        else:
            plt.show()
    
    def save_engineer(self, path: Union[str, Path]):
        """Save feature engineer"""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'enable_technical_indicators': self.config.enable_technical_indicators,
            'enable_statistical_features': self.config.enable_statistical_features,
            'enable_lag_features': self.config.enable_lag_features,
            'enable_interaction_features': self.config.enable_interaction_features,
            'enable_volatility_features': self.config.enable_volatility_features,
            'technical_periods': self.config.technical_periods,
            'statistical_windows': self.config.statistical_windows,
            'max_lags': self.config.max_lags,
            'scaling_method': self.config.scaling_method.value,
            # ... add other parameters
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save metadata
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata_.to_dict(), f, indent=2, default=str)

        # Save scalers
        if self.scalers_:
            joblib.dump(self.scalers_, path / "scalers.pkl")
        
        # Save feature selector
        if self.feature_selector_:
            joblib.dump(self.feature_selector_, path / "feature_selector.pkl")

        logger.info(f"💾 Feature Engineer saved to {path}")
    
    @classmethod
    def load_engineer(cls, path: Union[str, Path]) -> 'FeatureEngineer':
        """Load feature engineer"""
        
        path = Path(path)
        
        # Load configuration
        with open(path / "config.json", 'r') as f:
            config_dict = json.load(f)
        
        # Create configuration
        config = FeatureConfig(**config_dict)

        # Create instance
        engineer = cls(config)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        
        engineer.metadata_ = FeatureMetadata(**metadata_dict)
        
        # Load scalers
        scalers_path = path / "scalers.pkl"
        if scalers_path.exists():
            engineer.scalers_ = joblib.load(scalers_path)
        
        # Load feature selector
        selector_path = path / "feature_selector.pkl"
        if selector_path.exists():
            engineer.feature_selector_ = joblib.load(selector_path)
        
        engineer.is_fitted_ = True
        
        logger.info(f"📂 Feature Engineer loaded from {path}")
        
        return engineer


# Utility functions for fast computations
@jit(nopython=True, parallel=True)
def fast_rolling_mean(data: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean calculation"""
    
    n = len(data)
    result = np.empty(n)
    
    for i in prange(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            result[i] = np.mean(data[i-window+1:i+1])
    
    return result


@jit(nopython=True)
def fast_percentage_change(data: np.ndarray) -> np.ndarray:
    """Fast percentage change calculation"""
    
    n = len(data)
    result = np.empty(n)
    result[0] = 0.0
    
    for i in range(1, n):
        if data[i-1] != 0:
            result[i] = (data[i] - data[i-1]) / data[i-1]
        else:
            result[i] = 0.0
    
    return result


if __name__ == "__main__":
    # Usage example
    logger.info("🧪 Testing Feature Engineer...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Synthetic time series price data
    price_data = pd.DataFrame(index=dates)
    price_data['open'] = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    price_data['high'] = price_data['open'] + np.random.exponential(0.5, n_samples)
    price_data['low'] = price_data['open'] - np.random.exponential(0.5, n_samples)
    price_data['close'] = price_data['open'] + np.random.randn(n_samples) * 0.5
    price_data['volume'] = np.random.exponential(1000, n_samples)
    
    # Target variable - future price change
    target = price_data['close'].pct_change(periods=5).shift(-5)  # 5-period forward return
    
    # Create feature engineer
    config = FeatureConfig(
        enable_technical_indicators=True,
        enable_statistical_features=True,
        enable_lag_features=True,
        enable_interaction_features=False,  # Disabled for speed
        enable_volatility_features=True,
        feature_selection_k=50
    )
    
    engineer = FeatureEngineer(config)
    
    # Fit and transform
    X_transformed = engineer.fit_transform(price_data, target.dropna())
    
    logger.info(f"📊 Original features: {len(price_data.columns)}")
    logger.info(f"📊 Created features: {len(X_transformed.columns)}")

    # Feature importance
    importance = engineer.get_feature_importance(top_k=10)
    if not importance.empty:
        logger.info(f"🎯 Top-3 features: {importance['feature'].iloc[:3].tolist()}")

    logger.info("✅ Feature Engineer testing completed!")