"""
Advanced Model Selection for Crypto Trading AutoML
Implements enterprise patterns for robust model selection
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor, SGDClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.console import Console
import matplotlib.pyplot as plt
import seaborn as sns
import time

from ..utils.config_manager import AutoMLConfig


class TaskType(Enum):
    """Types tasks machine training"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


@dataclass
class ModelSelectionResult:
    """Result selection models"""
    model_scores: Dict[str, float]
    best_models: List[str]
    evaluation_metadata: Dict[str, Any]
    task_type: str
    selection_time: float
    cv_results: Dict[str, List[float]]


class BaseModelProvider(ABC):
    """Base class for providers models - pattern"""
    
    @abstractmethod
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """ model for tasks"""
        pass
    
    @abstractmethod
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """ parameters by default for model"""
        pass


class SklearnModelProvider(BaseModelProvider):
    """Provider models scikit-learn"""
    
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """ model scikit-learn"""
        if task_type == TaskType.REGRESSION:
            return {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(random_state=42),
                'lasso': Lasso(random_state=42, max_iter=2000),
                'elasticnet': ElasticNet(random_state=42, max_iter=2000),
                'random_forest': RandomForestRegressor(random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingRegressor(random_state=42),
                'extra_trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
                'decision_tree': DecisionTreeRegressor(random_state=42),
                'knn': KNeighborsRegressor(n_jobs=-1),
                'svr': SVR(),
                'mlp': MLPRegressor(random_state=42, max_iter=500)
            }
        else:  # Classification
            return {
                'logistic_regression': LogisticRegression(random_state=42, max_iter=2000),
                'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'gradient_boosting': GradientBoostingClassifier(random_state=42),
                'extra_trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'knn': KNeighborsClassifier(n_jobs=-1),
                'svc': SVC(random_state=42, probability=True),
                'mlp': MLPClassifier(random_state=42, max_iter=500)
            }
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Parameters by default for sklearn models"""
        default_params = {
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'gradient_boosting': {'n_estimators': 100, 'max_depth': 6},
            'extra_trees': {'n_estimators': 100, 'max_depth': 10},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'elasticnet': {'alpha': 1.0, 'l1_ratio': 0.5},
            'knn': {'n_neighbors': 5},
            'mlp': {'hidden_layer_sizes': (100,), 'alpha': 0.001}
        }
        return default_params.get(model_name, {})


class GradientBoostingModelProvider(BaseModelProvider):
    """Provider models gradient boosting"""
    
    def get_models(self, task_type: TaskType) -> Dict[str, Any]:
        """ model gradient boosting"""
        if task_type == TaskType.REGRESSION:
            return {
                'xgboost': xgb.XGBRegressor(
                    random_state=42, n_jobs=-1, verbosity=0
                ),
                'lightgbm': lgb.LGBMRegressor(
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'catboost': cb.CatBoostRegressor(
                    random_state=42, verbose=False
                )
            }
        else:  # Classification
            return {
                'xgboost': xgb.XGBClassifier(
                    random_state=42, n_jobs=-1, verbosity=0
                ),
                'lightgbm': lgb.LGBMClassifier(
                    random_state=42, n_jobs=-1, verbose=-1
                ),
                'catboost': cb.CatBoostClassifier(
                    random_state=42, verbose=False
                )
            }
    
    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        """Parameters by default for gradient boosting"""
        default_params = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'catboost': {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1
            }
        }
        return default_params.get(model_name, {})


class ModelSelector:
    """
    Advanced selector models for crypto trading
    Implements enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.model_providers: Dict[str, BaseModelProvider] = {}
        self.console = Console()
        self._setup_providers()
        
    def _setup_providers(self):
        """Configure providers models"""
        logger.info("🔧 Configure providers models...")
        
        self.model_providers['sklearn'] = SklearnModelProvider()
        self.model_providers['gradient_boosting'] = GradientBoostingModelProvider()
        
        logger.info(f"✅ Configured {len(self.model_providers)} providers")
    
    def _detect_task_type(self, y: pd.Series) -> TaskType:
        """Determine type tasks"""
        unique_values = y.nunique()
        
        if y.dtype in ['float64', 'float32'] or unique_values > 20:
            return TaskType.REGRESSION
        elif unique_values == 2:
            return TaskType.BINARY_CLASSIFICATION
        else:
            return TaskType.MULTICLASS_CLASSIFICATION
    
    def _get_scoring_metric(self, task_type: TaskType) -> str:
        """Get metrics for scoring"""
        if task_type == TaskType.REGRESSION:
            return 'neg_mean_squared_error'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            return 'f1'
        else:  # Multiclass
            return 'f1_macro'
    
    def _get_all_models(self, task_type: TaskType, models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get all available models"""
        all_models = {}
        
        for provider_name, provider in self.model_providers.items():
            provider_models = provider.get_models(task_type)
            
            # models if specified specific
            if models:
                provider_models = {
                    name: model for name, model in provider_models.items()
                    if name in models
                }
            
            all_models.update(provider_models)
        
        return all_models
    
    def select_best_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Optional[List[str]] = None,
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        time_series_split: bool = True,
        top_k: int = 5
    ) -> ModelSelectionResult:
        """
        Select best models
        
        Args:
            X: Matrix features
            y: Target variable
            models: List models for testing
            cv_folds: Number folds for cross-validation
            scoring: Metric scoring
            time_series_split: Use split
            top_k: Number best models for return
        """
        start_time = time.time()
        
        logger.info("🤖 Launch selection models...")
        
        # Determine type tasks
        task_type = self._detect_task_type(y)
        logger.info(f"🎯 Type tasks: {task_type.value}")
        
        # Get metrics scoring
        if scoring is None:
            scoring = self._get_scoring_metric(task_type)
        
        # Get models for testing
        all_models = self._get_all_models(task_type, models)
        
        if not all_models:
            logger.error("❌ No models for testing")
            return ModelSelectionResult(
                model_scores={},
                best_models=[],
                evaluation_metadata={'error': 'No models available'},
                task_type=task_type.value,
                selection_time=0.0,
                cv_results={}
            )
        
        logger.info(f"📊 Testing {len(all_models)} models...")
        
        # Configure cross-validation
        if time_series_split and task_type == TaskType.REGRESSION:
            cv = TimeSeriesSplit(n_splits=cv_folds)
            cv_name = f"TimeSeriesSplit({cv_folds})"
        elif task_type != TaskType.REGRESSION:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_name = f"StratifiedKFold({cv_folds})"
        else:
            cv = cv_folds
            cv_name = f"KFold({cv_folds})"
        
        # Evaluate models
        model_scores = {}
        cv_results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Evaluate models...", total=len(all_models))
            
            for model_name, model in all_models.items():
                progress.update(task, description=f"Model: {model_name}")
                
                try:
                    # Preparation data
                    X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
                    y_clean = y.fillna(y.mean()) if y.isna().any() else y
                    
                    # Cross-validation
                    scores = cross_val_score(
                        model, X_clean, y_clean,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1,
                        error_score='raise'
                    )
                    
                    mean_score = np.mean(scores)
                    model_scores[model_name] = mean_score
                    cv_results[model_name] = scores.tolist()
                    
                    logger.debug(f"✅ {model_name}: {mean_score:.4f} ± {np.std(scores):.4f}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error in model {model_name}: {e}")
                    model_scores[model_name] = float('-inf') # Bad score
                    cv_results[model_name] = []
                
                progress.advance(task)
        
        # Sort models by score
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True # For majority metrics more = better
        )
        
        # Correction for metrics where less = better (for example, MSE)
        if scoring.startswith('neg_'):
            sorted_models = sorted(
                model_scores.items(),
                key=lambda x: -x[1], # for neg_ metrics
                reverse=True
            )
        
        best_models = [model[0] for model in sorted_models[:top_k]]
        
        selection_time = time.time() - start_time
        
        result = ModelSelectionResult(
            model_scores=model_scores,
            best_models=best_models,
            evaluation_metadata={
                'task_type': task_type.value,
                'scoring_metric': scoring,
                'cv_strategy': cv_name,
                'models_tested': len(all_models),
                'successful_models': len([s for s in model_scores.values() if s != float('-inf')])
            },
            task_type=task_type.value,
            selection_time=selection_time,
            cv_results=cv_results
        )
        
        # Output results
        self._print_results(result)
        
        logger.info(f"✅ Select models completed for {selection_time:.2f}with")
        
        return result
    
    def _print_results(self, result: ModelSelectionResult):
        """Output results selection models"""
        
        # Create table with
        table = Table(title="🏆 Results SELECTION Models")
        
        table.add_column("Rank", style="cyan", no_wrap=True)
        table.add_column("Model", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Std", style="yellow")
        
        # Sort by score
        sorted_models = sorted(
            result.model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (model_name, score) in enumerate(sorted_models[:10], 1):
            if model_name in result.cv_results and result.cv_results[model_name]:
                std_score = np.std(result.cv_results[model_name])
                std_str = f"±{std_score:.4f}"
            else:
                std_str = "N/A"
            
            table.add_row(
                str(i),
                model_name,
                f"{score:.4f}",
                std_str
            )
        
        self.console.print(table)
    
    def plot_model_comparison(
        self,
        result: ModelSelectionResult,
        top_n: int = 10,
        save_path: Optional[str] = None
    ):
        """Visualization comparison models"""
        try:
            # Top N models
            sorted_models = sorted(
                result.model_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            models, scores = zip(*sorted_models)
            
            # Chart scores
            plt.figure(figsize=(12, 8))
            
            # Main chart
            plt.subplot(2, 1, 1)
            bars = plt.barh(models, scores, color='skyblue', alpha=0.7)
            plt.xlabel('Score model')
            plt.title(f'Comparison top {top_n} models')
            plt.grid(True, alpha=0.3)
            
            # Add values on columns
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center')
            
            # Box plot for top-5 models with CV
            plt.subplot(2, 1, 2)
            top_5_models = [m for m in models[:5] if m in result.cv_results and result.cv_results[m]]
            
            if top_5_models:
                cv_data = [result.cv_results[model] for model in top_5_models]
                plt.boxplot(cv_data, labels=top_5_models)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('CV Score')
                plt.title('Distribution CV scores for top-5 models')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Chart saved: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error creation : {e}")
    
    def get_model_recommendations(
        self,
        result: ModelSelectionResult,
        data_size: int,
        feature_count: int
    ) -> Dict[str, str]:
        """Get recommendations by selection models"""
        
        recommendations = {}
        
        # Recommendations on basis size data
        if data_size < 1000:
            recommendations['data_size'] = "Small dataset: recommended simple model (Linear, Ridge, Lasso)"
        elif data_size < 10000:
            recommendations['data_size'] = "Average dataset: are suitable Random Forest, Gradient Boosting"
        else:
            recommendations['data_size'] = "Large dataset: effective XGBoost, LightGBM, CatBoost"
        
        # Recommendations on basis number features
        if feature_count < 10:
            recommendations['features'] = " features: simple model can be more effective"
        elif feature_count < 100:
            recommendations['features'] = "Moderate number features: are suitable ensembles"
        else:
            recommendations['features'] = "Many features: recommended regularization (Ridge, Lasso)"
        
        # Recommendations on basis type tasks
        task_type = result.task_type
        if task_type == 'regression':
            recommendations['task'] = "Regression: note attention on MSE and R² metrics"
        else:
            recommendations['task'] = "Classification: precision, recall and F1-score"
        
        # Top model
        if result.best_models:
            best_model = result.best_models[0]
            best_score = result.model_scores[best_model]
            recommendations['best_model'] = f"Best model: {best_model} (score: {best_score:.4f})"
        
        return recommendations
    
    def get_selection_report(self, result: ModelSelectionResult) -> str:
        """Create report by models"""
        
        report = f"""
=== REPORT By SELECTION Models ===

Type tasks: {result.task_type}
Time selection: {result.selection_time:.2f}with
Tested models: {len(result.model_scores)}
Successful models: {result.evaluation_metadata.get('successful_models', 0)}

Top-5 models:
"""
        
        sorted_models = sorted(
            result.model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (model_name, score) in enumerate(sorted_models[:5], 1):
            if model_name in result.cv_results and result.cv_results[model_name]:
                std_score = np.std(result.cv_results[model_name])
                report += f"{i}. {model_name}: {score:.4f} ± {std_score:.4f}\n"
            else:
                report += f"{i}. {model_name}: {score:.4f}\n"
        
        report += f"\nMetadata: {result.evaluation_metadata}"
        
        return report


if __name__ == "__main__":
    # Example use
    from ..utils.config_manager import AutoMLConfig
    
    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Regression
    y_reg = pd.Series(
        X.iloc[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
    )
    
    # Classification
    y_clf = pd.Series((y_reg > y_reg.median()).astype(int))
    
    # Create selector
    config = AutoMLConfig()
    selector = ModelSelector(config)
    
    # Testing regression
    print("=== Testing REGRESSION ===")
    result_reg = selector.select_best_models(
        X, y_reg,
        models=['linear_regression', 'ridge', 'random_forest', 'xgboost'],
        cv_folds=3,
        top_k=3
    )
    
    print(selector.get_selection_report(result_reg))
    
    # Testing classification
    print("\n=== Testing CLASSIFICATION ===")
    result_clf = selector.select_best_models(
        X, y_clf,
        models=['logistic_regression', 'random_forest', 'xgboost'],
        cv_folds=3,
        time_series_split=False
    )
    
    print(selector.get_selection_report(result_clf))
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    recommendations = selector.get_model_recommendations(
        result_reg, data_size=len(X), feature_count=len(X.columns)
    )
    
    for key, rec in recommendations.items():
        print(f"{key}: {rec}")