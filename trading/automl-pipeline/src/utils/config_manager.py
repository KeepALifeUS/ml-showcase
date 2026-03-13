"""
Configuration Manager for AutoML Pipeline
Implements enterprise patterns for configuration management
"""

import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import os
import json
import yaml
from pathlib import Path
from pydantic import BaseSettings, Field, validator
from loguru import logger


@dataclass
class FeatureGenerationConfig:
    """Configuration generation features"""
    enable_technical_indicators: bool = True
    enable_statistical_features: bool = True
    enable_polynomial_features: bool = True
    enable_tsfresh_features: bool = True
    
    # Parameters technical indicators
    technical_indicators_windows: List[int] = field(default_factory=lambda: [10, 20, 50])
    
    # Parameters statistical features
    statistical_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Parameters polynomial features
    polynomial_degree: int = 2
    polynomial_max_features: int = 50
    polynomial_interaction_only: bool = True
    
    # Parameters TSFresh
    tsfresh_max_features: int = 30
    tsfresh_default_fc_parameters: str = "efficient"
    
    # General parameters
    parallel_generation: bool = True
    max_features_per_generator: int = 100


@dataclass
class FeatureSelectionConfig:
    """Configuration selection features"""
    enable_statistical_selection: bool = True
    enable_model_based_selection: bool = True
    enable_correlation_selection: bool = True
    enable_variance_selection: bool = True
    
    # Parameters statistical selection
    statistical_method: str = "f_regression"  # f_regression, mutual_info_regression
    statistical_k: int = 50
    statistical_percentile: float = 50.0
    
    # Parameters selection
    model_type: str = "random_forest"  # random_forest, xgboost
    model_max_features: int = 100
    
    # Parameters correlation selection
    correlation_threshold: float = 0.95
    target_correlation_min: float = 0.01
    
    # Parameters selection by variance
    variance_threshold: float = 0.0
    
    # Ensemble selection
    ensemble_selection: bool = True
    min_votes_threshold: int = 2


@dataclass
class HyperparameterOptimizationConfig:
    """Configuration optimization hyperparameters"""
    default_optimizer: str = "optuna_tpe"  # optuna_tpe, optuna_random, gaussian_process
    n_trials: int = 100
    n_jobs: int = -1
    random_state: int = 42
    
    # Parameters Optuna
    optuna_study_name_prefix: str = "automl_optimization"
    optuna_sampler_startup_trials: int = 10
    optuna_sampler_n_ei_candidates: int = 24
    
    # Parameters scikit-optimize
    skopt_n_initial_points: int = 10
    skopt_acq_func: str = "EI"  # EI, PI, LCB
    
    # General parameters
    cv_folds: int = 5
    scoring_metric: Optional[str] = None  # Automatic determination
    timeout_per_trial: int = 300 # seconds
    
    # Early stopping
    enable_pruning: bool = True
    pruning_min_trials: int = 20


@dataclass
class ModelSelectionConfig:
    """Configuration selection models"""
    enable_sklearn_models: bool = True
    enable_xgboost: bool = True
    enable_lightgbm: bool = True
    enable_catboost: bool = True
    
    # Model for testing
    sklearn_models: List[str] = field(default_factory=lambda: [
        'linear_regression', 'ridge', 'lasso', 'elasticnet',
        'random_forest', 'gradient_boosting', 'extra_trees'
    ])
    
    gradient_boosting_models: List[str] = field(default_factory=lambda: [
        'xgboost', 'lightgbm', 'catboost'
    ])
    
    # Cross-validation parameters
    cv_folds: int = 5
    time_series_split: bool = True
    shuffle_split: bool = False
    
    # Criteria selection
    scoring_metric: Optional[str] = None
    top_k_models: int = 5
    
    # models
    max_training_time_per_model: int = 600 # seconds
    min_score_threshold: Optional[float] = None


@dataclass
class EnsembleConfig:
    """Configuration ensembles"""
    enable_voting: bool = True
    enable_stacking: bool = True
    enable_blending: bool = True
    enable_bagging: bool = False
    
    # Parameters voting ensemble
    voting_estimators_limit: int = 10
    voting_weights: Optional[List[float]] = None
    
    # Parameters stacking
    stacking_cv_folds: int = 5
    stacking_meta_learner: str = "ridge"  # ridge, linear_regression
    stacking_use_features_in_secondary: bool = True
    
    # Parameters blending
    blending_holdout_size: float = 0.2
    
    # General parameters
    ensemble_size_limit: int = 5
    min_ensemble_diversity: float = 0.1


@dataclass
class DataPreprocessingConfig:
    """Configuration preprocessing data"""
    # Processing missing values
    missing_value_strategy: str = "median"  # mean, median, mode, drop, forward_fill
    missing_value_threshold: float = 0.5 # Threshold for removal columns/rows
    
    # Processing outliers
    outlier_detection_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    outlier_handling: str = "clip"  # clip, remove, transform
    
    # Scale
    scaling_method: str = "standard"  # standard, robust, minmax, quantile
    scale_target: bool = False
    
    # Encode categorical features
    categorical_encoding: str = "onehot"  # onehot, label, target, binary
    max_categories_onehot: int = 10
    
    # Processing temporal series
    handle_seasonality: bool = True
    detrend_method: Optional[str] = None  # linear, polynomial
    
    # General parameters
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ModelEvaluationConfig:
    """Configuration evaluation models"""
    # Metrics for regression
    regression_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'r2', 'mape', 'rmse'
    ])
    
    # Metrics for classification
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'auc'
    ])
    
    # Cross-validation
    cv_folds: int = 5
    cv_scoring: Optional[str] = None
    
    # features
    calculate_feature_importance: bool = True
    feature_importance_method: str = "permutation"  # permutation, shap, built_in
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 300
    
    # Reports
    generate_report: bool = True
    report_format: str = "html"  # html, pdf, markdown


class AutoMLConfig(BaseSettings):
    """
    Main configuration AutoML Pipeline
    Implements enterprise patterns for configuration management
    """
    
    # Main parameters
    project_name: str = Field(default="crypto_trading_automl", env="AUTOML_PROJECT_NAME")
    version: str = "1.0.0"
    random_state: int = Field(default=42, env="AUTOML_RANDOM_STATE")
    n_jobs: int = Field(default=-1, env="AUTOML_N_JOBS")
    
    # Paths
    output_dir: str = Field(default="automl_output", env="AUTOML_OUTPUT_DIR")
    cache_dir: str = Field(default="automl_cache", env="AUTOML_CACHE_DIR")
    models_dir: str = Field(default="automl_models", env="AUTOML_MODELS_DIR")
    logs_dir: str = Field(default="automl_logs", env="AUTOML_LOGS_DIR")
    
    # Regimes work
    debug_mode: bool = Field(default=False, env="AUTOML_DEBUG")
    verbose: bool = Field(default=True, env="AUTOML_VERBOSE")
    enable_caching: bool = Field(default=True, env="AUTOML_CACHE")
    
    # Limits resources
    max_memory_gb: float = Field(default=8.0, env="AUTOML_MAX_MEMORY")
    max_training_time: int = Field(default=3600, env="AUTOML_MAX_TIME") # seconds
    max_models_to_try: int = Field(default=50, env="AUTOML_MAX_MODELS")
    
    # Configuration components
    feature_generation: FeatureGenerationConfig = field(default_factory=FeatureGenerationConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    hyperparameter_optimization: HyperparameterOptimizationConfig = field(
        default_factory=HyperparameterOptimizationConfig
    )
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    data_preprocessing: DataPreprocessingConfig = field(default_factory=DataPreprocessingConfig)
    model_evaluation: ModelEvaluationConfig = field(default_factory=ModelEvaluationConfig)
    
    # Specific for crypto trading parameters
    crypto_specific: Dict[str, Any] = field(default_factory=lambda: {
        'enable_technical_indicators': True,
        'enable_market_regime_detection': True,
        'enable_volatility_features': True,
        'enable_momentum_features': True,
        'lookback_periods': [5, 10, 20, 50],
        'prediction_horizon': 1, # Horizon predictions (periods)
        'risk_adjusted_metrics': True,
        'walk_forward_validation': True
    })
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
    
    @validator('output_dir', 'cache_dir', 'models_dir', 'logs_dir')
    def create_directories(cls, v):
        """Create directories if not exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('max_memory_gb')
    def validate_memory(cls, v):
        """Validation limit memory"""
        if v <= 0:
            raise ValueError("max_memory_gb should be positive ")
        return v
    
    @validator('n_jobs')
    def validate_n_jobs(cls, v):
        """Validation number processes"""
        if v == 0:
            raise ValueError("n_jobs not can be 0")
        return v
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save configuration in file"""
        filepath = Path(filepath)
        
        config_dict = self.dict()
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
        else:
            raise ValueError("Supported only formats .json, .yml, .yaml")
        
        logger.info(f"💾 Configuration saved: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]):
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File configuration not found: {filepath}")
        
        if filepath.suffix.lower() == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Supported only formats .json, .yml, .yaml")
        
        logger.info(f"📂 Configuration loaded: {filepath}")
        
        return cls(**config_dict)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        model_configs = {
            'xgboost': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbosity': 0 if not self.verbose else 1
            },
            'lightgbm': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': -1 if not self.verbose else 1
            },
            'catboost': {
                'random_state': self.random_state,
                'verbose': self.verbose
            },
            'sklearn': {
                'random_state': self.random_state,
                'n_jobs': self.n_jobs if model_name in [
                    'random_forest', 'extra_trees', 'knn'
                ] else None
            }
        }
        
        # Base configuration for sklearn models
        base_config = model_configs.get('sklearn', {})
        
        # Specific configuration
        if model_name.startswith('xgb') or model_name == 'xgboost':
            return {**base_config, **model_configs['xgboost']}
        elif model_name.startswith('lgb') or model_name == 'lightgbm':
            return {**base_config, **model_configs['lightgbm']}
        elif model_name.startswith('cat') or model_name == 'catboost':
            return {**base_config, **model_configs['catboost']}
        else:
            return base_config
    
    def get_crypto_features_config(self) -> Dict[str, Any]:
        """Get configuration for cryptocurrency features"""
        return {
            **self.crypto_specific,
            'technical_windows': self.feature_generation.technical_indicators_windows,
            'statistical_windows': self.feature_generation.statistical_windows,
            'enable_technical': self.feature_generation.enable_technical_indicators,
            'enable_statistical': self.feature_generation.enable_statistical_features
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get configuration for validation"""
        return {
            'cv_folds': self.model_selection.cv_folds,
            'time_series_split': self.model_selection.time_series_split,
            'walk_forward_validation': self.crypto_specific.get('walk_forward_validation', True),
            'random_state': self.random_state
        }
    
    def __str__(self) -> str:
        """String representation configuration"""
        return f"AutoMLConfig(project='{self.project_name}', version='{self.version}')"


# Preset configuration
class PresetConfigs:
    """Preset configuration for different scenarios"""
    
    @staticmethod
    def fast_prototype() -> AutoMLConfig:
        """Fast configuration for prototyping"""
        config = AutoMLConfig()
        
        # Decreasing number iterations
        config.hyperparameter_optimization.n_trials = 20
        config.model_selection.cv_folds = 3
        config.model_evaluation.cv_folds = 3
        
        # Disable complex features
        config.feature_generation.enable_tsfresh_features = False
        config.feature_generation.enable_polynomial_features = False
        
        # Limit model
        config.model_selection.sklearn_models = ['ridge', 'random_forest']
        config.model_selection.gradient_boosting_models = ['xgboost']
        
        return config
    
    @staticmethod
    def production_ready() -> AutoMLConfig:
        """Configuration for production"""
        config = AutoMLConfig()
        
        # Increasing number iterations
        config.hyperparameter_optimization.n_trials = 200
        config.model_selection.cv_folds = 10
        config.model_evaluation.cv_folds = 10
        
        # Enable all functions
        config.feature_generation.enable_tsfresh_features = True
        config.feature_generation.enable_polynomial_features = True
        
        # Enable ensembles
        config.ensemble.enable_stacking = True
        config.ensemble.enable_voting = True
        
        # Enable detailed
        config.model_evaluation.calculate_feature_importance = True
        config.model_evaluation.generate_plots = True
        config.model_evaluation.generate_report = True
        
        return config
    
    @staticmethod
    def crypto_trading() -> AutoMLConfig:
        """Specialized configuration for crypto trading"""
        config = AutoMLConfig()
        
        # Configure under temporal series
        config.model_selection.time_series_split = True
        config.data_preprocessing.handle_seasonality = True
        
        # Cryptocurrency features
        config.feature_generation.enable_technical_indicators = True
        config.feature_generation.technical_indicators_windows = [5, 10, 20, 50, 100]
        
        # Specific parameters
        config.crypto_specific.update({
            'enable_volatility_features': True,
            'enable_momentum_features': True,
            'enable_market_regime_detection': True,
            'lookback_periods': [1, 3, 5, 10, 20],
            'prediction_horizon': 1
        })
        
        # Model suitable for temporal series
        config.model_selection.sklearn_models = [
            'ridge', 'lasso', 'elasticnet', 'random_forest', 'gradient_boosting'
        ]
        config.model_selection.gradient_boosting_models = ['xgboost', 'lightgbm']
        
        return config
    
    @staticmethod
    def high_frequency_trading() -> AutoMLConfig:
        """Configuration for high-frequency trading"""
        config = PresetConfigs.crypto_trading()
        
        # Reducing time training
        config.max_training_time = 1800 # 30 minutes
        config.hyperparameter_optimization.n_trials = 50
        config.hyperparameter_optimization.timeout_per_trial = 60
        
        # Fast model
        config.model_selection.sklearn_models = ['ridge', 'lasso']
        config.model_selection.gradient_boosting_models = ['lightgbm'] # Most fast
        
        # Disable complex generation features
        config.feature_generation.enable_tsfresh_features = False
        config.feature_generation.polynomial_max_features = 20
        
        # Specific parameters for HFT
        config.crypto_specific.update({
            'lookback_periods': [1, 2, 3, 5], # Short periods
            'prediction_horizon': 1, # Only next
            'enable_microstructure_features': True,
            'enable_order_book_features': True
        })
        
        return config


if __name__ == "__main__":
    # Example use
    
    # Create base configuration
    config = AutoMLConfig()
    print(f"Base configuration: {config}")
    
    # Save in file
    config.save_to_file("automl_config.json")
    
    # Load from file
    loaded_config = AutoMLConfig.load_from_file("automl_config.json")
    print(f"Loaded configuration: {loaded_config}")
    
    # Preset configuration
    fast_config = PresetConfigs.fast_prototype()
    print(f"Fast configuration: {fast_config}")
    
    crypto_config = PresetConfigs.crypto_trading()
    print(f"Configuration for crypto trading: {crypto_config}")
    
    # Get configuration model
    xgb_config = config.get_model_config('xgboost')
    print(f"Configuration XGBoost: {xgb_config}")
    
    # Configuration validation
    validation_config = config.get_validation_config()
    print(f"Configuration validation: {validation_config}")