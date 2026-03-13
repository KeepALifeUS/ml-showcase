"""
XGBoost Time Series Models for Time Series Forecasting
=====================================================

Advanced XGBoost implementations optimized for time series time series prediction
with enterprise patterns and production-ready features.

Models are specifically tuned for:
- High-frequency financial market data
- Volatility and regime changes
- Non-stationary time series
- Feature importance tracking
- Model interpretability via SHAP
- Real-time prediction serving

Author: ML XGBoost Contributors
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import pickle
import joblib
import json
from pathlib import Path
import time
from datetime import datetime, timedelta

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_percentage_error, median_absolute_error
)
import shap
import optuna
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


class ModelType(Enum):
    """XGBoost model types"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    RANKING = "ranking"


class Objective(Enum):
    """XGBoost objective functions optimized for time series"""

    # Regression objectives
    SQUARED_ERROR = "reg:squarederror"  # Primary for price prediction
    PSEUDO_HUBER = "reg:pseudohubererror"  # Robust to outliers
    GAMMA = "reg:gamma"  # For positive values (volatility)
    TWEEDIE = "reg:tweedie"  # For zero-inflated data

    # Classification objectives
    LOGISTIC = "binary:logistic"  # Binary classification (up/down)
    HINGE = "binary:hinge"  # SVM-like for trend detection
    MULTICLASS = "multi:softmax"  # Multi-class (strong_buy/buy/hold/sell/strong_sell)
    MULTIPROB = "multi:softprob"  # Class probabilities

    # Ranking objectives
    RANK_PAIRWISE = "rank:pairwise"  # For asset ranking
    RANK_NDCG = "rank:ndcg"  # NDCG optimization

    # Survival analysis (for time-to-event)
    SURVIVAL_COX = "survival:cox"  # Cox regression
    SURVIVAL_AFT = "survival:aft"  # Accelerated failure time


@dataclass
class TrainingConfig:
    """XGBoost model training configuration"""

    # Main parameters
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    colsample_bylevel: float = 0.8
    colsample_bynode: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    gamma: float = 0.0  # Minimum split loss
    min_child_weight: int = 1
    
    # Early stopping
    early_stopping_rounds: int = 50
    eval_metric: Optional[str] = None
    
    # Performance
    n_jobs: int = -1
    random_state: int = 42
    verbosity: int = 0
    
    # Financial-specific
    monotone_constraints: Optional[Dict[str, int]] = None
    interaction_constraints: Optional[List[List[int]]] = None
    feature_weights: Optional[Dict[str, float]] = None
    
    # Validation
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    
    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameters"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'colsample_bylevel': self.colsample_bylevel,
            'colsample_bynode': self.colsample_bynode,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbosity': self.verbosity,
        }
        
        # Add optional parameters
        if self.monotone_constraints:
            params['monotone_constraints'] = self.monotone_constraints
        if self.interaction_constraints:
            params['interaction_constraints'] = self.interaction_constraints
            
        return params


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    
    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    mape: Optional[float] = None
    r2: Optional[float] = None
    median_ae: Optional[float] = None
    
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    
    # Trading specific metrics
    directional_accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Model info
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    feature_count: Optional[int] = None
    best_iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetrics':
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class BaseXGBoostModel(ABC, BaseEstimator):
    """
    Base class for XGBoost models
    Implements enterprise patterns
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        objective: Optional[Union[str, Objective]] = None,
        enable_shap: bool = True,
        enable_feature_tracking: bool = True,
        model_name: Optional[str] = None
    ):
        self.config = config or TrainingConfig()
        self.objective = objective
        self.enable_shap = enable_shap
        self.enable_feature_tracking = enable_feature_tracking
        self.model_name = model_name or self.__class__.__name__
        
        # Model state
        self.model_: Optional[xgb.XGBModel] = None
        self.feature_names_: List[str] = []
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.shap_explainer_: Optional[shap.Explainer] = None
        self.shap_values_: Optional[np.ndarray] = None
        self.metrics_: Optional[ModelMetrics] = None
        self.training_history_: List[Dict[str, Any]] = []
        
        # Caching
        self._prediction_cache: Dict[str, Any] = {}
        self._explanation_cache: Dict[str, Any] = {}
        
        self.console = Console()
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging"""
        logger.add(
            f"logs/xgb_model_{self.model_name}_{datetime.now():%Y%m%d}.log",
            rotation="daily",
            retention="30 days",
            level="INFO"
        )
    
    @abstractmethod
    def _get_default_objective(self) -> str:
        """Get default objective function"""
        pass
    
    @abstractmethod 
    def _create_model(self) -> xgb.XGBModel:
        """Create XGBoost model"""
        pass

    def _prepare_objective(self) -> str:
        """Prepare objective function"""
        if self.objective is None:
            return self._get_default_objective()
        elif isinstance(self.objective, Objective):
            return self.objective.value
        else:
            return self.objective
    
    def _prepare_data(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare data for training"""

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        
        # Clean data
        X = X.copy()
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove inf values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Process y if present
        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            y = y.fillna(method='ffill').fillna(method='bfill')
            y = y.replace([np.inf, -np.inf], y.median())
        
        return X, y
    
    def _create_validation_sets(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Create train/validation split respecting temporal structure"""
        
        n_samples = len(X)
        n_val = int(n_samples * validation_size)
        
        # Temporal split - validation is the most recent data
        X_train = X.iloc[:-n_val]
        y_train = y.iloc[:-n_val]
        X_val = X.iloc[-n_val:]
        y_val = y.iloc[-n_val:]
        
        logger.info(f"📊 Train size: {len(X_train)}, Validation size: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = True
    ) -> 'BaseXGBoostModel':
        """
        Train the XGBoost model

        Args:
            X: Feature matrix
            y: Target variable
            X_val: Validation feature set
            y_val: Validation target variable
            sample_weight: Sample weights
            eval_set: Evaluation set
            verbose: Progress output
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"🚀 Starting model training {self.model_name}")

        # Prepare data
        X, y = self._prepare_data(X, y)
        self.feature_names_ = list(X.columns)
        
        # Create validation set if not provided
        if X_val is None and eval_set is None:
            X_train, y_train, X_val, y_val = self._create_validation_sets(
                X, y, self.config.validation_size
            )
        else:
            X_train, y_train = X, y
            if X_val is not None and y_val is not None:
                X_val, y_val = self._prepare_data(X_val, y_val)
        
        # Create model
        self.model_ = self._create_model()

        # Configure eval_set
        if eval_set is None and X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'validation']
        else:
            eval_names = None
        
        # Training with progress bar
        if verbose:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            ) as progress:
                task = progress.add_task("Training model...", total=100)

                # Callbacks for progress tracking
                callbacks = []
                if hasattr(xgb, 'callback'):
                    def progress_callback(env):
                        progress.update(task, completed=env.iteration)
                    callbacks.append(progress_callback)
                
                self.model_.fit(
                    X_train, y_train,
                    sample_weight=sample_weight,
                    eval_set=eval_set,
                    eval_names=eval_names,
                    early_stopping_rounds=self.config.early_stopping_rounds,
                    verbose=False,
                    callbacks=callbacks
                )
                
                progress.update(task, completed=100)
        else:
            self.model_.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_names=eval_names,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose=False
            )
        
        # Calculate metrics
        training_time = time.time() - start_time
        self.metrics_ = self._calculate_metrics(X_train, y_train, X_val, y_val, training_time)
        
        # Feature importance
        if self.enable_feature_tracking:
            self._calculate_feature_importance()
        
        # SHAP values
        if self.enable_shap:
            self._setup_shap_explainer(X_train)
        
        # Save training history
        self._save_training_history(X_train, y_train, training_time)
        
        if verbose:
            self._display_training_results()
        
        logger.info(f"✅ Training completed in {training_time:.2f}s")
        
        return self
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        use_cache: bool = True
    ) -> np.ndarray:
        """Prediction"""

        if self.model_ is None:
            raise ValueError("Model is not trained. Call fit() first.")

        # Prepare data
        X, _ = self._prepare_data(X)
        
        # Cache predictions
        if use_cache:
            cache_key = f"predict_{hash(str(X.values.tobytes()))}"
            if cache_key in self._prediction_cache:
                return self._prediction_cache[cache_key]
        
        start_time = time.time()
        predictions = self.model_.predict(X)
        prediction_time = time.time() - start_time
        
        # Update prediction time metrics
        if self.metrics_:
            self.metrics_.prediction_time = prediction_time
        
        # Cache result
        if use_cache:
            self._prediction_cache[cache_key] = predictions
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict probabilities (for classification)"""

        if not hasattr(self.model_, 'predict_proba'):
            raise AttributeError("Model does not support predict_proba")
        
        X, _ = self._prepare_data(X)
        return self.model_.predict_proba(X)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance"""
        
        if self.model_ is None:
            return
        
        # Get importance from model
        importance_gain = self.model_.feature_importances_
        
        # Get other importance metrics if available
        booster = self.model_.get_booster()
        importance_weight = booster.get_score(importance_type='weight')
        importance_cover = booster.get_score(importance_type='cover')
        
        # Create DataFrame with importance
        feature_importance_data = []
        for i, feature_name in enumerate(self.feature_names_):
            feature_importance_data.append({
                'feature': feature_name,
                'importance_gain': importance_gain[i] if i < len(importance_gain) else 0,
                'importance_weight': importance_weight.get(f'f{i}', 0),
                'importance_cover': importance_cover.get(f'f{i}', 0),
            })
        
        self.feature_importance_ = pd.DataFrame(feature_importance_data)
        self.feature_importance_ = self.feature_importance_.sort_values(
            'importance_gain', ascending=False
        ).reset_index(drop=True)
    
    def get_feature_importance(
        self, 
        importance_type: str = 'gain',
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            importance_type: Importance type ('gain', 'weight', 'cover')
            top_k: Number of top features
        """
        
        if self.feature_importance_ is None:
            self._calculate_feature_importance()
        
        if self.feature_importance_ is None:
            return pd.DataFrame()
        
        column_name = f'importance_{importance_type}'
        if column_name not in self.feature_importance_.columns:
            column_name = 'importance_gain'  # fallback
        
        result = self.feature_importance_.sort_values(column_name, ascending=False)
        
        if top_k:
            result = result.head(top_k)
        
        return result
    
    def _setup_shap_explainer(self, X_train: pd.DataFrame):
        """Set up SHAP explainer"""
        
        try:
            # Use TreeExplainer for XGBoost
            self.shap_explainer_ = shap.TreeExplainer(self.model_)
            
            # Calculate SHAP values on sample data for initialization
            sample_size = min(100, len(X_train))
            sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_indices]
            
            self.shap_values_ = self.shap_explainer_.shap_values(X_sample)
            
            logger.info("✅ SHAP explainer set up successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Error setting up SHAP: {e}")
            self.enable_shap = False
    
    def get_shap_values(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        max_samples: int = 1000
    ) -> np.ndarray:
        """
        Get SHAP values

        Args:
            X: Data to explain
            max_samples: Maximum number of samples for analysis
        """

        if not self.enable_shap or self.shap_explainer_ is None:
            raise ValueError("SHAP is not set up. Set enable_shap=True when creating the model.")
        
        X, _ = self._prepare_data(X)
        
        # Limit number of samples for performance
        if len(X) > max_samples:
            sample_indices = np.random.choice(len(X), max_samples, replace=False)
            X = X.iloc[sample_indices]
        
        cache_key = f"shap_{hash(str(X.values.tobytes()))}"
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]
        
        shap_values = self.shap_explainer_.shap_values(X)
        
        # Caching
        self._explanation_cache[cache_key] = shap_values
        
        return shap_values
    
    def plot_feature_importance(
        self, 
        importance_type: str = 'gain',
        top_k: int = 20,
        save_path: Optional[str] = None
    ):
        """Visualize feature importance"""
        
        importance_df = self.get_feature_importance(importance_type, top_k)
        
        if importance_df.empty:
            logger.warning("⚠️ No feature importance data available")
            return

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Horizontal bar plot
        bars = ax.barh(
            range(len(importance_df)), 
            importance_df[f'importance_{importance_type}'],
            color='skyblue',
            alpha=0.7
        )
        
        # Configure axes
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel(f'Feature Importance ({importance_type.capitalize()})')
        ax.set_title(f'Top {top_k} Feature Importance - {self.model_name}')
        ax.grid(True, alpha=0.3)
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Feature importance plot saved: {save_path}")
        else:
            plt.show()
    
    def plot_shap_summary(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """SHAP summary plot"""
        
        if not self.enable_shap:
            logger.warning("⚠️ SHAP is not enabled")
            return
        
        X, _ = self._prepare_data(X)
        shap_values = self.get_shap_values(X)
        
        # SHAP summary plot
        shap.summary_plot(
            shap_values, X, 
            max_display=max_display,
            show=save_path is None
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 SHAP summary plot saved: {save_path}")
    
    @abstractmethod
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        training_time: float
    ) -> ModelMetrics:
        """Calculate model metrics"""
        pass
    
    def _save_training_history(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        training_time: float
    ):
        """Save training history"""
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'training_samples': len(X_train),
            'feature_count': len(self.feature_names_),
            'training_time': training_time,
            'config': self.config.__dict__.copy(),
            'metrics': self.metrics_.to_dict() if self.metrics_ else {},
        }
        
        self.training_history_.append(history_entry)
    
    def _display_training_results(self):
        """Display training results"""
        
        if self.metrics_ is None:
            return
        
        # Create results table
        table = Table(title=f"🎯 TRAINING RESULTS - {self.model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        metrics_dict = self.metrics_.to_dict()
        for key, value in metrics_dict.items():
            if value is not None:
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
        
        self.console.print(table)
    
    def save_model(self, path: Union[str, Path], include_shap: bool = True):
        """Save model"""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        if self.model_:
            model_path = path / "xgb_model.json"
            self.model_.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'feature_names': self.feature_names_,
            'config': self.config.__dict__,
            'metrics': self.metrics_.to_dict() if self.metrics_ else {},
            'training_history': self.training_history_,
            'objective': self.objective.value if isinstance(self.objective, Objective) else self.objective,
            'enable_shap': self.enable_shap,
            'enable_feature_tracking': self.enable_feature_tracking,
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature importance
        if self.feature_importance_ is not None:
            self.feature_importance_.to_csv(path / "feature_importance.csv", index=False)
        
        # Save SHAP explainer
        if include_shap and self.shap_explainer_:
            joblib.dump(self.shap_explainer_, path / "shap_explainer.pkl")
        
        logger.info(f"💾 Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: Union[str, Path]) -> 'BaseXGBoostModel':
        """Load model"""
        
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Create configuration
        config = TrainingConfig(**metadata['config'])

        # Create model instance
        model = cls(
            config=config,
            objective=metadata.get('objective'),
            enable_shap=metadata.get('enable_shap', True),
            enable_feature_tracking=metadata.get('enable_feature_tracking', True),
            model_name=metadata.get('model_name')
        )
        
        # Load XGBoost model
        model.model_ = model._create_model()
        if (path / "xgb_model.json").exists():
            model.model_.load_model(str(path / "xgb_model.json"))
        
        # Restore attributes
        model.feature_names_ = metadata.get('feature_names', [])
        model.training_history_ = metadata.get('training_history', [])
        
        # Load metrics
        if metadata.get('metrics'):
            model.metrics_ = ModelMetrics.from_dict(metadata['metrics'])
        
        # Load feature importance
        importance_path = path / "feature_importance.csv"
        if importance_path.exists():
            model.feature_importance_ = pd.read_csv(importance_path)
        
        # Load SHAP explainer
        shap_path = path / "shap_explainer.pkl"
        if shap_path.exists():
            model.shap_explainer_ = joblib.load(shap_path)
        
        logger.info(f"📂 Model loaded from {path}")
        
        return model


class XGBoostRegressor(BaseXGBoostModel, RegressorMixin):
    """XGBoost regressor for time series price prediction"""
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        objective: Optional[Union[str, Objective]] = None,
        enable_shap: bool = True,
        enable_feature_tracking: bool = True,
        model_name: Optional[str] = None
    ):
        super().__init__(
            config=config,
            objective=objective or Objective.SQUARED_ERROR,
            enable_shap=enable_shap,
            enable_feature_tracking=enable_feature_tracking,
            model_name=model_name or "XGBoostRegressor"
        )
    
    def _get_default_objective(self) -> str:
        return Objective.SQUARED_ERROR.value
    
    def _create_model(self) -> xgb.XGBRegressor:
        """Create XGBoost regressor"""
        
        params = self.config.to_xgb_params()
        params['objective'] = self._prepare_objective()
        
        return xgb.XGBRegressor(**params)
    
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        training_time: float
    ) -> ModelMetrics:
        """Calculate regression metrics"""

        # Predictions on train set
        y_train_pred = self.model_.predict(X_train)
        
        # Basic regression metrics
        metrics = ModelMetrics(
            rmse=np.sqrt(mean_squared_error(y_train, y_train_pred)),
            mae=mean_absolute_error(y_train, y_train_pred),
            r2=r2_score(y_train, y_train_pred),
            median_ae=median_absolute_error(y_train, y_train_pred),
            training_time=training_time,
            feature_count=len(self.feature_names_),
            best_iteration=getattr(self.model_, 'best_iteration', None)
        )
        
        # MAPE (if no zero values)
        if not (y_train == 0).any():
            metrics.mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Metrics on validation set
        if X_val is not None and y_val is not None:
            y_val_pred = self.model_.predict(X_val)
            
            # Directional accuracy for trading
            y_train_direction = np.sign(y_train.diff().fillna(0))
            y_train_pred_direction = np.sign(pd.Series(y_train_pred).diff().fillna(0))
            metrics.directional_accuracy = accuracy_score(
                y_train_direction, y_train_pred_direction
            )
            
            # Trading metrics
            returns_actual = y_val.pct_change().fillna(0)
            returns_pred = pd.Series(y_val_pred).pct_change().fillna(0)
            
            if returns_actual.std() != 0:
                metrics.sharpe_ratio = returns_actual.mean() / returns_actual.std() * np.sqrt(252)
            
            # Max drawdown
            cumulative_returns = (1 + returns_actual).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics.max_drawdown = drawdown.min()
        
        return metrics


class XGBoostClassifier(BaseXGBoostModel, ClassifierMixin):
    """XGBoost classifier for predicting price movement direction"""
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        objective: Optional[Union[str, Objective]] = None,
        enable_shap: bool = True,
        enable_feature_tracking: bool = True,
        model_name: Optional[str] = None,
        n_classes: int = 2
    ):
        self.n_classes = n_classes
        
        # Choose objective based on number of classes
        if objective is None:
            objective = (
                Objective.LOGISTIC if n_classes == 2 
                else Objective.MULTICLASS
            )
        
        super().__init__(
            config=config,
            objective=objective,
            enable_shap=enable_shap,
            enable_feature_tracking=enable_feature_tracking,
            model_name=model_name or "XGBoostClassifier"
        )
    
    def _get_default_objective(self) -> str:
        return (
            Objective.LOGISTIC.value if self.n_classes == 2 
            else Objective.MULTICLASS.value
        )
    
    def _create_model(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier"""
        
        params = self.config.to_xgb_params()
        params['objective'] = self._prepare_objective()
        
        if self.n_classes > 2:
            params['num_class'] = self.n_classes
        
        return xgb.XGBClassifier(**params)
    
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        training_time: float
    ) -> ModelMetrics:
        """Calculate classification metrics"""

        # Predictions on train set
        y_train_pred = self.model_.predict(X_train)

        # Basic classification metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_train, y_train_pred),
            precision=precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
            recall=recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
            f1=f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
            training_time=training_time,
            feature_count=len(self.feature_names_),
            best_iteration=getattr(self.model_, 'best_iteration', None)
        )
        
        return metrics


class XGBoostTimeSeriesModel(XGBoostRegressor):
    """
    Specialized XGBoost model for time series

    Optimized for:
    - Temporal data structure
    - Non-stationarity of time series
    - Regime shifts and volatility
    - High-frequency data
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        objective: Optional[Union[str, Objective]] = None,
        enable_shap: bool = True,
        enable_feature_tracking: bool = True,
        model_name: Optional[str] = None,
        # Time series specific parameters
        time_feature_engineering: bool = True,
        lag_features: int = 10,
        rolling_windows: List[int] = None,
        volatility_features: bool = True,
        regime_detection: bool = True
    ):
        # Time series parameters
        self.time_feature_engineering = time_feature_engineering
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.volatility_features = volatility_features
        self.regime_detection = regime_detection
        
        # Special configuration for time series
        if config is None:
            config = TrainingConfig(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.5,
                gamma=0.1,
                early_stopping_rounds=30,
            )
        
        super().__init__(
            config=config,
            objective=objective or Objective.PSEUDO_HUBER,  # Robust to outliers
            enable_shap=enable_shap,
            enable_feature_tracking=enable_feature_tracking,
            model_name=model_name or "XGBoostTimeSeriesModel"
        )
    
    def _prepare_time_series_data(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        create_features: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare time series data with feature creation"""
        
        X, y = self._prepare_data(X, y)
        
        if not create_features or not self.time_feature_engineering:
            return X, y
        
        logger.info("🔧 Creating time series features...")

        # Create copy for modification
        X_enhanced = X.copy()
        
        # Sort by time if datetime index exists
        if isinstance(X.index, pd.DatetimeIndex):
            X_enhanced = X_enhanced.sort_index()
            if y is not None:
                y = y.sort_index()
        
        # Lag features
        if self.lag_features > 0:
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    for lag in range(1, self.lag_features + 1):
                        X_enhanced[f'{col}_lag_{lag}'] = X[col].shift(lag)
        
        # Rolling window features
        if self.rolling_windows:
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    for window in self.rolling_windows:
                        X_enhanced[f'{col}_rolling_mean_{window}'] = X[col].rolling(window).mean()
                        X_enhanced[f'{col}_rolling_std_{window}'] = X[col].rolling(window).std()
                        X_enhanced[f'{col}_rolling_min_{window}'] = X[col].rolling(window).min()
                        X_enhanced[f'{col}_rolling_max_{window}'] = X[col].rolling(window).max()
        
        # Volatility features
        if self.volatility_features:
            for col in X.columns:
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Log returns
                    log_returns = np.log(X[col] / X[col].shift(1))
                    X_enhanced[f'{col}_log_returns'] = log_returns
                    
                    # Realized volatility
                    for window in [5, 10, 20]:
                        X_enhanced[f'{col}_realized_vol_{window}'] = (
                            log_returns.rolling(window).std() * np.sqrt(252)
                        )
                    
                    # GARCH-like volatility proxy
                    X_enhanced[f'{col}_vol_proxy'] = log_returns.rolling(20).std()
        
        # Clean NaN values after feature creation
        X_enhanced = X_enhanced.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if y is not None:
            # Remove corresponding rows from y
            valid_indices = X_enhanced.index
            y = y.reindex(valid_indices).fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ Created {len(X_enhanced.columns)} features (was {len(X.columns)})")
        
        return X_enhanced, y
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_set: Optional[List[Tuple]] = None,
        verbose: bool = True
    ) -> 'XGBoostTimeSeriesModel':
        """Training with time series feature engineering"""

        # Create time series features
        X_enhanced, y_prepared = self._prepare_time_series_data(X, y, create_features=True)
        
        # Prepare validation set with the same features
        if X_val is not None:
            X_val_enhanced, y_val_prepared = self._prepare_time_series_data(
                X_val, y_val, create_features=True
            )
        else:
            X_val_enhanced, y_val_prepared = None, None
        
        # Call parent fit method
        return super().fit(
            X_enhanced, y_prepared,
            X_val=X_val_enhanced, y_val=y_val_prepared,
            sample_weight=sample_weight,
            eval_set=eval_set,
            verbose=verbose
        )
    
    def predict(
        self, 
        X: Union[pd.DataFrame, np.ndarray],
        use_cache: bool = True
    ) -> np.ndarray:
        """Prediction with time series feature creation"""

        # Create features
        X_enhanced, _ = self._prepare_time_series_data(X, create_features=True)
        
        # Prediction
        return super().predict(X_enhanced, use_cache=use_cache)


class XGBoostRanker(BaseXGBoostModel):
    """XGBoost ranker for sorting time series by attractiveness"""
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        objective: Optional[Union[str, Objective]] = None,
        enable_shap: bool = True,
        enable_feature_tracking: bool = True,
        model_name: Optional[str] = None
    ):
        super().__init__(
            config=config,
            objective=objective or Objective.RANK_PAIRWISE,
            enable_shap=enable_shap,
            enable_feature_tracking=enable_feature_tracking,
            model_name=model_name or "XGBoostRanker"
        )
    
    def _get_default_objective(self) -> str:
        return Objective.RANK_PAIRWISE.value
    
    def _create_model(self) -> xgb.XGBRanker:
        """Create XGBoost ranker"""
        
        params = self.config.to_xgb_params()
        params['objective'] = self._prepare_objective()
        
        return xgb.XGBRanker(**params)
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        group: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        group_val: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'XGBoostRanker':
        """
        Train the ranker

        Args:
            group: Group sizes for ranking
        """

        if verbose:
            logger.info(f"🚀 Starting ranker training {self.model_name}")

        # Prepare data
        X, y = self._prepare_data(X, y)
        self.feature_names_ = list(X.columns)
        
        # Create model
        self.model_ = self._create_model()

        # Prepare eval_set for ranker
        eval_set = None
        eval_group = None
        if X_val is not None and y_val is not None and group_val is not None:
            X_val, y_val = self._prepare_data(X_val, y_val)
            eval_set = [(X, y), (X_val, y_val)]
            eval_group = [group, group_val]
        
        # Training
        start_time = time.time()
        
        self.model_.fit(
            X, y, group=group,
            eval_set=eval_set,
            eval_group=eval_group,
            sample_weight=sample_weight,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        
        # Basic metrics (ranker has specific metrics)
        self.metrics_ = ModelMetrics(
            training_time=training_time,
            feature_count=len(self.feature_names_),
            best_iteration=getattr(self.model_, 'best_iteration', None)
        )
        
        # Feature importance
        if self.enable_feature_tracking:
            self._calculate_feature_importance()
        
        # SHAP values
        if self.enable_shap:
            self._setup_shap_explainer(X)
        
        if verbose:
            logger.info(f"✅ Ranker training completed in {training_time:.2f}s")
        
        return self
    
    def _calculate_metrics(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        training_time: float
    ) -> ModelMetrics:
        """For ranker, metrics are calculated separately"""
        return ModelMetrics(
            training_time=training_time,
            feature_count=len(self.feature_names_),
            best_iteration=getattr(self.model_, 'best_iteration', None)
        )


# JIT-compiled utility functions for accelerating computations
@jit(nopython=True, parallel=True)
def calculate_rolling_features(data: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling features calculation with Numba"""
    
    n = len(data)
    result = np.empty((n, 4))  # mean, std, min, max
    
    for i in prange(n):
        if i < window - 1:
            result[i] = np.nan
        else:
            window_data = data[i-window+1:i+1]
            result[i, 0] = np.mean(window_data)  # mean
            result[i, 1] = np.std(window_data)   # std
            result[i, 2] = np.min(window_data)   # min
            result[i, 3] = np.max(window_data)   # max
    
    return result


@jit(nopython=True)
def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Fast log returns calculation"""
    
    n = len(prices)
    log_returns = np.empty(n)
    log_returns[0] = 0.0
    
    for i in range(1, n):
        if prices[i] > 0 and prices[i-1] > 0:
            log_returns[i] = np.log(prices[i] / prices[i-1])
        else:
            log_returns[i] = 0.0
    
    return log_returns


if __name__ == "__main__":
    # Usage example
    logger.info("🧪 Testing XGBoost Time Series Model...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Time index
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
    
    # Create features with temporal dependence
    X = pd.DataFrame(index=dates)
    
    for i in range(n_features):
        # Random walk with trend
        trend = 0.001 * np.arange(n_samples)
        noise = np.random.randn(n_samples) * 0.1
        X[f'feature_{i}'] = trend + noise + np.sin(np.arange(n_samples) * 0.1) * 0.5
    
    # Target variable - combination of features with noise
    y = pd.Series(
        X.iloc[:, :3].sum(axis=1) + 0.2 * np.random.randn(n_samples),
        index=dates,
        name='target'
    )
    
    # Split data
    train_size = int(0.8 * n_samples)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size] 
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    # Create and train model
    model = XGBoostTimeSeriesModel(
        enable_shap=True,
        time_feature_engineering=True,
        lag_features=5,
        rolling_windows=[10, 20],
        volatility_features=True
    )
    
    # Training
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"📊 Test RMSE: {rmse:.4f}")
    logger.info(f"📊 Test MAE: {mae:.4f}") 
    logger.info(f"📊 Test R²: {r2:.4f}")
    
    # Feature importance
    importance = model.get_feature_importance(top_k=10)
    logger.info(f"🎯 Top features: {importance['feature'].tolist()[:5]}")
    
    # SHAP values
    shap_values = model.get_shap_values(X_test.iloc[:100])
    logger.info(f"📈 SHAP values shape: {shap_values.shape}")
    
    logger.info("✅ Testing completed successfully!")