"""
Advanced Data Preprocessor for Crypto Trading AutoML
Implements enterprise patterns for robust data preprocessing
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, QuantileUniformTransformer,
    LabelEncoder, OneHotEncoder, TargetEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.stats import boxcox, yeojohnson
import pandas_ta as pta
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
import joblib
from pathlib import Path

from .config_manager import AutoMLConfig, DataPreprocessingConfig


@dataclass
class PreprocessingResult:
    """Result preprocessing data"""
    processed_data: pd.DataFrame
    preprocessing_metadata: Dict[str, Any]
    transformers: Dict[str, Any]
    processing_time: float
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]


class DataPreprocessor:
    """
    Advanced preprocessor data for crypto trading
    Implements enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.preprocessing_config = self.config.data_preprocessing
        
        # Save transformers for repeated use
        self.fitted_transformers = {}
        self.preprocessing_pipeline = None
        self.is_fitted = False
        
        # Metadata
        self.preprocessing_metadata = {}
        
        logger.info("🔧 DataPreprocessor initialized")
    
    def preprocess(
        self,
        data: pd.DataFrame,
        fit: bool = True,
        preserve_index: bool = True
    ) -> pd.DataFrame:
        """
        Main method preprocessing data
        
        Args:
            data: Source data
            fit: Train transformers (True for training set)
            preserve_index: Save index
        """
        import time
        start_time = time.time()
        
        logger.info(f"🔄 Start preprocessing: {data.shape}")
        
        original_shape = data.shape
        processed_data = data.copy()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                
                # Stage 1: Base cleanup
                task = progress.add_task("Base cleanup data...", total=None)
                processed_data = self._basic_cleaning(processed_data)
                
                # Stage 2: Processing missing values
                progress.update(task, description="Processing missing values...")
                processed_data = self._handle_missing_values(processed_data, fit)
                
                # Stage 3: Processing outliers
                progress.update(task, description="Processing outliers...")
                processed_data = self._handle_outliers(processed_data, fit)
                
                # Stage 4: Encode categorical features
                progress.update(task, description="Encode categorical features...")
                processed_data = self._encode_categorical(processed_data, fit)
                
                # Stage 5: Scale numeric features
                progress.update(task, description="Scale features...")
                processed_data = self._scale_features(processed_data, fit)
                
                # Stage 6: Remove features with low variance
                progress.update(task, description="Remove features with low variance...")
                processed_data = self._remove_low_variance_features(processed_data, fit)
                
                # Stage 7: Final cleanup
                progress.update(task, description="Final cleanup...")
                processed_data = self._final_cleaning(processed_data)
                
                progress.update(task, description="✅ Preprocessing completed", completed=True)
        
            # Save metadata
            processing_time = time.time() - start_time
            final_shape = processed_data.shape
            
            self.preprocessing_metadata = {
                'original_shape': original_shape,
                'final_shape': final_shape,
                'processing_time': processing_time,
                'rows_removed': original_shape[0] - final_shape[0],
                'columns_removed': original_shape[1] - final_shape[1],
                'missing_values_handled': True,
                'outliers_handled': True,
                'categorical_encoded': True,
                'features_scaled': True
            }
            
            if fit:
                self.is_fitted = True
            
            logger.info(f"✅ Preprocessing completed: {original_shape} → {final_shape} for {processing_time:.2f}with")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing: {e}")
            return data # Return original data in case errors
    
    def preprocess_target(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Preprocessing target variable"""
        logger.info("🎯 Preprocessing target variable...")
        
        processed_target = target.copy()
        
        try:
            # Processing missing values
            if processed_target.isna().any():
                if self.preprocessing_config.missing_value_strategy == 'drop':
                    processed_target = processed_target.dropna()
                else:
                    fill_value = processed_target.mean()
                    processed_target = processed_target.fillna(fill_value)
                    logger.info(f"📝 Filled {target.isna().sum()} missing values target variable")
            
            # Processing outliers in target variable
            if self.preprocessing_config.outlier_handling != 'none':
                processed_target = self._handle_target_outliers(processed_target, fit)
            
            # Scale target variable (if necessary)
            if self.preprocessing_config.scale_target:
                processed_target = self._scale_target(processed_target, fit)
            
            logger.info(f"✅ Target variable : {len(target)} → {len(processed_target)}")
            
            return processed_target
            
        except Exception as e:
            logger.error(f"❌ Error processing target variable: {e}")
            return target
    
    def _basic_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Base cleanup data"""
        logger.info("🧹 Base cleanup data...")
        
        cleaned_data = data.copy()
        
        # Remove completely empty rows and columns
        initial_shape = cleaned_data.shape
        cleaned_data = cleaned_data.dropna(how='all', axis=0)  # Rows
        cleaned_data = cleaned_data.dropna(how='all', axis=1)  # Columns
        
        if cleaned_data.shape != initial_shape:
            logger.info(f"📝 Removed empty rows/columns: {initial_shape} → {cleaned_data.shape}")
        
        # Remove duplicate rows
        duplicates = cleaned_data.duplicated().sum()
        if duplicates > 0:
            cleaned_data = cleaned_data.drop_duplicates()
            logger.info(f"📝 Removed {duplicates} duplicate rows")
        
        # Remove constants (columns with one unique )
        constant_columns = []
        for col in cleaned_data.columns:
            if cleaned_data[col].nunique() <= 1:
                constant_columns.append(col)
        
        if constant_columns:
            cleaned_data = cleaned_data.drop(columns=constant_columns)
            logger.info(f"📝 Removed constant columns: {constant_columns}")
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Processing missing values"""
        logger.info("🕳️ Processing missing values...")
        
        if not data.isna().any().any():
            logger.info("📝 Missing values not detected")
            return data
        
        strategy = self.preprocessing_config.missing_value_strategy
        threshold = self.preprocessing_config.missing_value_threshold
        
        # Remove columns with large number gaps
        missing_ratios = data.isna().sum() / len(data)
        columns_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
        
        if columns_to_drop:
            data = data.drop(columns=columns_to_drop)
            logger.info(f"📝 Removed columns with >({threshold*100}%) gaps: {columns_to_drop}")
        
        # Split on numeric and categorical columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Processing numeric columns
        if numeric_columns:
            if strategy == 'mean':
                imputer_numeric = SimpleImputer(strategy='mean')
            elif strategy == 'median':
                imputer_numeric = SimpleImputer(strategy='median')
            elif strategy == 'forward_fill':
                data[numeric_columns] = data[numeric_columns].fillna(method='ffill')
                imputer_numeric = None
            else:  # KNN imputation
                imputer_numeric = KNNImputer(n_neighbors=5)
            
            if imputer_numeric and fit:
                data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])
                self.fitted_transformers['numeric_imputer'] = imputer_numeric
            elif imputer_numeric and not fit and 'numeric_imputer' in self.fitted_transformers:
                data[numeric_columns] = self.fitted_transformers['numeric_imputer'].transform(data[numeric_columns])
        
        # Processing categorical columns
        if categorical_columns:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            
            if fit:
                data[categorical_columns] = imputer_categorical.fit_transform(data[categorical_columns])
                self.fitted_transformers['categorical_imputer'] = imputer_categorical
            elif 'categorical_imputer' in self.fitted_transformers:
                data[categorical_columns] = self.fitted_transformers['categorical_imputer'].transform(data[categorical_columns])
        
        remaining_missing = data.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"⚠️ Remaining missing values: {remaining_missing}")
            # Final cleanup - filling zeros
            data = data.fillna(0)
        else:
            logger.info("✅ All missing values processed")
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Processing outliers"""
        logger.info("📊 Processing outliers...")
        
        method = self.preprocessing_config.outlier_detection_method
        handling = self.preprocessing_config.outlier_handling
        threshold = self.preprocessing_config.outlier_threshold
        
        if handling == 'none':
            return data
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return data
        
        outliers_detected = 0
        
        for col in numeric_columns:
            try:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(data[col], nan_policy='omit'))
                    outliers_mask = z_scores > threshold
                    
                elif method == 'isolation_forest':
                    if fit:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        outliers_pred = iso_forest.fit_predict(data[col].values.reshape(-1, 1))
                        self.fitted_transformers[f'isolation_forest_{col}'] = iso_forest
                    else:
                        if f'isolation_forest_{col}' in self.fitted_transformers:
                            iso_forest = self.fitted_transformers[f'isolation_forest_{col}']
                            outliers_pred = iso_forest.predict(data[col].values.reshape(-1, 1))
                        else:
                            continue
                    
                    outliers_mask = outliers_pred == -1
                
                outliers_count = outliers_mask.sum()
                if outliers_count > 0:
                    outliers_detected += outliers_count
                    
                    if handling == 'remove':
                        data = data[~outliers_mask]
                    elif handling == 'clip':
                        if method != 'isolation_forest':
                            data.loc[outliers_mask, col] = data[col].clip(lower_bound, upper_bound)
                        else:
                            # For isolation forest use quantiles
                            lower_clip = data[col].quantile(0.01)
                            upper_clip = data[col].quantile(0.99)
                            data.loc[outliers_mask, col] = data[col].clip(lower_clip, upper_clip)
                    elif handling == 'transform':
                        # Logarithmic transformation for positive values
                        if data[col].min() > 0:
                            data.loc[outliers_mask, col] = np.log1p(data.loc[outliers_mask, col])
            
            except Exception as e:
                logger.warning(f"⚠️ Error processing outliers in column {col}: {e}")
                continue
        
        if outliers_detected > 0:
            logger.info(f"📝 Processed {outliers_detected} outliers method {method}")
        else:
            logger.info("📝 Outliers not detected")
        
        return data
    
    def _encode_categorical(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("🔤 Encode categorical features...")
        
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not categorical_columns:
            logger.info("📝 Categorical features not detected")
            return data
        
        encoding_method = self.preprocessing_config.categorical_encoding
        max_categories = self.preprocessing_config.max_categories_onehot
        
        encoded_data = data.copy()
        
        for col in categorical_columns:
            try:
                unique_count = data[col].nunique()
                
                if encoding_method == 'onehot' and unique_count <= max_categories:
                    # One-Hot Encoding
                    if fit:
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoded_features = encoder.fit_transform(data[[col]])
                        self.fitted_transformers[f'onehot_{col}'] = encoder
                    else:
                        if f'onehot_{col}' in self.fitted_transformers:
                            encoder = self.fitted_transformers[f'onehot_{col}']
                            encoded_features = encoder.transform(data[[col]])
                        else:
                            continue
                    
                    # Create names features
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
                    
                    # Replacement original features
                    encoded_data = encoded_data.drop(columns=[col])
                    encoded_data = pd.concat([encoded_data, encoded_df], axis=1)
                    
                elif encoding_method == 'label' or unique_count > max_categories:
                    # Label Encoding
                    if fit:
                        encoder = LabelEncoder()
                        encoded_data[col] = encoder.fit_transform(data[col].astype(str))
                        self.fitted_transformers[f'label_{col}'] = encoder
                    else:
                        if f'label_{col}' in self.fitted_transformers:
                            encoder = self.fitted_transformers[f'label_{col}']
                            # Processing categories
                            try:
                                encoded_data[col] = encoder.transform(data[col].astype(str))
                            except ValueError:
                                # For categories assign -1
                                encoded_values = []
                                for value in data[col].astype(str):
                                    if value in encoder.classes_:
                                        encoded_values.append(encoder.transform([value])[0])
                                    else:
                                        encoded_values.append(-1)
                                encoded_data[col] = encoded_values
            
            except Exception as e:
                logger.warning(f"⚠️ Error encoding features {col}: {e}")
                continue
        
        logger.info(f"✅ Encoded {len(categorical_columns)} categorical features")
        
        return encoded_data
    
    def _scale_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        logger.info("⚖️ Scale numeric features...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            logger.info("📝 Numeric features for scaling not found")
            return data
        
        scaling_method = self.preprocessing_config.scaling_method
        scaled_data = data.copy()
        
        try:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'quantile':
                scaler = QuantileUniformTransformer()
            else:
                logger.warning(f"⚠️ Unknown method scaling: {scaling_method}")
                return data
            
            if fit:
                scaled_data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                self.fitted_transformers['feature_scaler'] = scaler
            else:
                if 'feature_scaler' in self.fitted_transformers:
                    scaler = self.fitted_transformers['feature_scaler']
                    scaled_data[numeric_columns] = scaler.transform(data[numeric_columns])
                else:
                    logger.warning("⚠️ Scaler not found, skip scaling")
            
            logger.info(f"✅ Scaled {len(numeric_columns)} numeric features method {scaling_method}")
            
        except Exception as e:
            logger.error(f"❌ Error scaling: {e}")
            return data
        
        return scaled_data
    
    def _remove_low_variance_features(self, data: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Remove features with low variance"""
        logger.info("📉 Remove features with low variance...")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return data
        
        threshold = self.preprocessing_config.variance_threshold
        
        try:
            if fit:
                variance_selector = VarianceThreshold(threshold=threshold)
                selected_features = variance_selector.fit_transform(data[numeric_columns])
                
                # Get indices selected features
                selected_mask = variance_selector.get_support()
                selected_columns = [col for col, mask in zip(numeric_columns, selected_mask) if mask]
                removed_columns = [col for col, mask in zip(numeric_columns, selected_mask) if not mask]
                
                self.fitted_transformers['variance_selector'] = variance_selector
                self.fitted_transformers['selected_numeric_columns'] = selected_columns
            else:
                if 'selected_numeric_columns' in self.fitted_transformers:
                    selected_columns = self.fitted_transformers['selected_numeric_columns']
                    removed_columns = [col for col in numeric_columns if col not in selected_columns]
                else:
                    return data
            
            # Remove features with low variance
            filtered_data = data.copy()
            if removed_columns:
                filtered_data = filtered_data.drop(columns=removed_columns)
                logger.info(f"📝 Removed {len(removed_columns)} features with low variance")
            else:
                logger.info("📝 All features have sufficient variance")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Error by variance: {e}")
            return data
    
    def _final_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup data"""
        logger.info("🏁 Final cleanup data...")
        
        cleaned_data = data.copy()
        
        # Remove infinite values
        infinite_mask = np.isinf(cleaned_data.select_dtypes(include=[np.number]))
        if infinite_mask.any().any():
            cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
            cleaned_data = cleaned_data.fillna(0)
            logger.info("📝 Processed infinite values")
        
        # Final check on NaN
        nan_count = cleaned_data.isna().sum().sum()
        if nan_count > 0:
            cleaned_data = cleaned_data.fillna(0)
            logger.info(f"📝 Filled {nan_count} remaining NaN values")
        
        return cleaned_data
    
    def _handle_target_outliers(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Processing outliers in target variable"""
        method = self.preprocessing_config.outlier_detection_method
        threshold = self.preprocessing_config.outlier_threshold
        
        if method == 'iqr':
            Q1 = target.quantile(0.25)
            Q3 = target.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (target < lower_bound) | (target > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(target, nan_policy='omit'))
            outliers_mask = z_scores > threshold
        
        else:
            return target
        
        outliers_count = outliers_mask.sum()
        if outliers_count > 0:
            # Trimming outliers
            target_clipped = target.clip(target.quantile(0.01), target.quantile(0.99))
            logger.info(f"📝 Processed {outliers_count} outliers in target variable")
            return target_clipped
        
        return target
    
    def _scale_target(self, target: pd.Series, fit: bool = True) -> pd.Series:
        """Scale target variable"""
        try:
            if fit:
                scaler = StandardScaler()
                scaled_target = scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
                self.fitted_transformers['target_scaler'] = scaler
            else:
                if 'target_scaler' in self.fitted_transformers:
                    scaler = self.fitted_transformers['target_scaler']
                    scaled_target = scaler.transform(target.values.reshape(-1, 1)).flatten()
                else:
                    return target
            
            return pd.Series(scaled_target, index=target.index)
            
        except Exception as e:
            logger.error(f"❌ Error scaling target variable: {e}")
            return target
    
    def inverse_transform_target(self, scaled_target: pd.Series) -> pd.Series:
        """Inverse transformation target variable"""
        if 'target_scaler' not in self.fitted_transformers:
            return scaled_target
        
        try:
            scaler = self.fitted_transformers['target_scaler']
            original_target = scaler.inverse_transform(scaled_target.values.reshape(-1, 1)).flatten()
            return pd.Series(original_target, index=scaled_target.index)
        except Exception as e:
            logger.error(f"❌ Error inverse transformations: {e}")
            return scaled_target
    
    def save_transformers(self, filepath: Union[str, Path]):
        """Save trained transformers"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        transformers_data = {
            'fitted_transformers': self.fitted_transformers,
            'preprocessing_metadata': self.preprocessing_metadata,
            'is_fitted': self.is_fitted,
            'config': self.preprocessing_config.__dict__ if hasattr(self.preprocessing_config, '__dict__') else str(self.preprocessing_config)
        }
        
        joblib.dump(transformers_data, filepath)
        logger.info(f"💾 Transformers saved: {filepath}")
    
    def load_transformers(self, filepath: Union[str, Path]):
        """Load trained transformers"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File transformers not found: {filepath}")
        
        transformers_data = joblib.load(filepath)
        
        self.fitted_transformers = transformers_data['fitted_transformers']
        self.preprocessing_metadata = transformers_data['preprocessing_metadata']
        self.is_fitted = transformers_data['is_fitted']
        
        logger.info(f"📂 Transformers loaded: {filepath}")
    
    def get_preprocessing_report(self) -> str:
        """Create report by """
        if not self.preprocessing_metadata:
            return "Preprocessing yet not completed"
        
        metadata = self.preprocessing_metadata
        
        report = f"""
=== REPORT By PREPROCESSING Data ===

Source data: {metadata.get('original_shape', 'N/A')}
Processed data: {metadata.get('final_shape', 'N/A')}
Time processing: {metadata.get('processing_time', 0):.2f}with

Changes:
- Removed rows: {metadata.get('rows_removed', 0)}
- Removed columns: {metadata.get('columns_removed', 0)}

Completed stages:
- Missing values: {'✅' if metadata.get('missing_values_handled') else '❌'}
- Processing outliers: {'✅' if metadata.get('outliers_handled') else '❌'}
- Encode categorical: {'✅' if metadata.get('categorical_encoded') else '❌'}
- Scale features: {'✅' if metadata.get('features_scaled') else '❌'}

Trained transformers: {len(self.fitted_transformers)}
"""
        
        return report


if __name__ == "__main__":
    # Example use DataPreprocessor
    
    # Create test data
    np.random.seed(42)
    n_samples = 1000
    
    # Create data with various issues
    data = pd.DataFrame({
        'numeric_normal': np.random.randn(n_samples),
        'numeric_with_outliers': np.concatenate([
            np.random.randn(n_samples - 50),
            np.random.randn(50) * 10  # Outliers
        ]),
        'numeric_with_missing': np.random.randn(n_samples),
        'categorical': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'binary': np.random.choice([0, 1], n_samples),
        'constant': [1] * n_samples, # Constant feature
    })
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, size=100, replace=False)
    data.loc[missing_indices, 'numeric_with_missing'] = np.nan
    
    # Target variable
    target = pd.Series(
        data['numeric_normal'] * 2 + 
        data['binary'] * 3 + 
        np.random.randn(n_samples) * 0.5
    )
    
    print("=== ORIGINAL Data ===")
    print(f"Shape data: {data.shape}")
    print(f"Missing values: {data.isna().sum().sum()}")
    print(f"Types data:\n{data.dtypes}")
    
    # Create and usage preprocessor
    config = AutoMLConfig()
    preprocessor = DataPreprocessor(config)
    
    # Preprocessing training data
    processed_data = preprocessor.preprocess(data, fit=True)
    processed_target = preprocessor.preprocess_target(target)
    
    print("\n=== PROCESSED Data ===")
    print(f"Shape data: {processed_data.shape}")
    print(f"Missing values: {processed_data.isna().sum().sum()}")
    print(f"Columns: {list(processed_data.columns)}")
    
    # Report by
    print(preprocessor.get_preprocessing_report())
    
    # Testing on new data (without training transformers)
    test_data = data.iloc[-100:].copy()
    processed_test_data = preprocessor.preprocess(test_data, fit=False)
    
    print(f"\n=== Test Data ===")
    print(f" shape: {test_data.shape}")
    print(f"Processed shape: {processed_test_data.shape}")