"""
Advanced Ensemble Builder for Crypto Trading AutoML
Implements enterprise patterns for robust ensemble construction
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
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.ensemble import VotingRegressor, VotingClassifier
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

from ..utils.config_manager import AutoMLConfig


class EnsembleMethod(Enum):
    """Methods ensembling"""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAGGING = "bagging"
    DYNAMIC_WEIGHTING = "dynamic_weighting"


@dataclass
class EnsembleResult:
    """Result building ensemble"""
    ensembles: Dict[str, Any]
    ensemble_scores: Dict[str, float]
    best_ensemble_method: str
    best_ensemble_score: float
    base_model_scores: Dict[str, float]
    ensemble_weights: Dict[str, Dict[str, float]]
    ensemble_metadata: Dict[str, Any]
    build_time: float


class BaseEnsembleBuilder(ABC):
    """Base class for builders ensembles - pattern"""
    
    @abstractmethod
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Build ensemble"""
        pass


class VotingEnsembleBuilder(BaseEnsembleBuilder):
    """Builder voting ensemble"""
    
    def __init__(self, voting_type: str = 'soft', weights: Optional[List[float]] = None):
        self.voting_type = voting_type
        self.weights = weights
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Build ensemble"""
        logger.info(f"🗳️ Build voting ensemble ({self.voting_type})")
        
        # Preparation models for ensemble
        estimators = [(name, model) for name, model in models.items()]
        
        try:
            if task_type == 'regression':
                ensemble = VotingRegressor(
                    estimators=estimators,
                    weights=self.weights
                )
            else:
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=self.voting_type,
                    weights=self.weights
                )
            
            # Training ensemble
            ensemble.fit(X, y)
            
            logger.info(f"✅ Voting ensemble built with {len(models)} models")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error building voting ensemble: {e}")
            return None


class StackingEnsembleBuilder(BaseEnsembleBuilder):
    """Builder stacking ensemble"""
    
    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        cv_folds: int = 5,
        use_features_in_secondary: bool = True
    ):
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.use_features_in_secondary = use_features_in_secondary
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Build stacking ensemble"""
        logger.info("🥞 Build stacking ensemble")
        
        try:
            # Configure meta-learning algorithm by default
            if self.meta_learner is None:
                if task_type == 'regression':
                    self.meta_learner = Ridge(alpha=1.0)
                else:
                    self.meta_learner = LogisticRegression(max_iter=1000)
            
            # Create stacking ensemble
            from sklearn.ensemble import StackingRegressor, StackingClassifier
            
            estimators = [(name, model) for name, model in models.items()]
            
            if task_type == 'regression':
                ensemble = StackingRegressor(
                    estimators=estimators,
                    final_estimator=self.meta_learner,
                    cv=self.cv_folds,
                    passthrough=self.use_features_in_secondary,
                    n_jobs=-1
                )
            else:
                ensemble = StackingClassifier(
                    estimators=estimators,
                    final_estimator=self.meta_learner,
                    cv=self.cv_folds,
                    passthrough=self.use_features_in_secondary,
                    n_jobs=-1
                )
            
            # Training ensemble
            ensemble.fit(X, y)
            
            logger.info(f"✅ Stacking ensemble built with {len(models)} base models")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error building stacking ensemble: {e}")
            return None


class BlendingEnsembleBuilder(BaseEnsembleBuilder):
    """Builder blending ensemble"""
    
    def __init__(
        self,
        holdout_size: float = 0.2,
        meta_learner: Optional[Any] = None
    ):
        self.holdout_size = holdout_size
        self.meta_learner = meta_learner
        self.base_models = None
        self.blending_predictions = None
        
    def build(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str = 'regression',
        **kwargs
    ) -> Any:
        """Build blending ensemble"""
        logger.info("🔀 Build blending ensemble")
        
        try:
            from sklearn.model_selection import train_test_split
            
            # Split on training base models and blending
            X_base, X_blend, y_base, y_blend = train_test_split(
                X, y,
                test_size=self.holdout_size,
                random_state=42
            )
            
            # Training base models
            trained_models = {}
            blend_predictions = []
            
            for name, model in models.items():
                # Create copies model for training
                if hasattr(model, 'copy'):
                    trained_model = model.copy()
                else:
                    from sklearn.base import clone
                    trained_model = clone(model)
                
                # Training on base
                trained_model.fit(X_base, y_base)
                trained_models[name] = trained_model
                
                # Predictions on blending
                predictions = trained_model.predict(X_blend)
                blend_predictions.append(predictions)
            
            # Preparation data for meta-learning
            blend_features = np.column_stack(blend_predictions)
            
            # Configure meta-algorithm by default
            if self.meta_learner is None:
                if task_type == 'regression':
                    self.meta_learner = Ridge(alpha=1.0)
                else:
                    self.meta_learner = LogisticRegression(max_iter=1000)
            
            # Training meta-algorithm
            self.meta_learner.fit(blend_features, y_blend)
            
            # Create final ensemble
            ensemble = BlendingEnsemble(
                base_models=trained_models,
                meta_learner=self.meta_learner
            )
            
            logger.info(f"✅ Blending ensemble built with {len(models)} base models")
            
            return ensemble
            
        except Exception as e:
            logger.error(f"❌ Error building blending ensemble: {e}")
            return None


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """Custom blending ensemble"""
    
    def __init__(self, base_models: Dict[str, Any], meta_learner: Any):
        self.base_models = base_models
        self.meta_learner = meta_learner
        
    def fit(self, X, y):
        # Model already trained in BlendingEnsembleBuilder
        return self
        
    def predict(self, X):
        # Get predictions from base models
        base_predictions = []
        for name, model in self.base_models.items():
            predictions = model.predict(X)
            base_predictions.append(predictions)
        
        # Stacking predictions
        stacked_predictions = np.column_stack(base_predictions)
        
        # Final prediction meta-algorithm
        final_predictions = self.meta_learner.predict(stacked_predictions)
        
        return final_predictions


class DynamicWeightingEnsemble(BaseEstimator, RegressorMixin):
    """Ensemble with dynamic weights"""
    
    def __init__(self, models: Dict[str, Any], window_size: int = 100):
        self.models = models
        self.window_size = window_size
        self.weights_history = []
        self.performance_history = {name: [] for name in models.keys()}
        
    def fit(self, X, y):
        # Training all base models
        for name, model in self.models.items():
            model.fit(X, y)
        
        return self
        
    def predict(self, X):
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # If no history, use equal weights
        if not self.weights_history:
            weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            weights = self._calculate_dynamic_weights()
        
        # Weighted averaging predictions
        final_predictions = np.zeros(len(X))
        for name, weight in weights.items():
            final_predictions += weight * predictions[name]
        
        return final_predictions
    
    def _calculate_dynamic_weights(self):
        """Computation dynamic weights on basis recent performance"""
        # Simplified implementation - equal weights
        return {name: 1.0 / len(self.models) for name in self.models.keys()}


class EnsembleBuilder:
    """
    Main class for building ensembles
    Implements enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.ensemble_config = self.config.ensemble
        self.console = Console()
        
        # Builders ensembles
        self.ensemble_builders: Dict[str, BaseEnsembleBuilder] = {}
        self._setup_builders()
        
    def _setup_builders(self):
        """Configure builders ensembles"""
        logger.info("🔧 Configure builders ensembles...")
        
        if self.ensemble_config.enable_voting:
            self.ensemble_builders['voting'] = VotingEnsembleBuilder(
                voting_type='soft',
                weights=self.ensemble_config.voting_weights
            )
        
        if self.ensemble_config.enable_stacking:
            self.ensemble_builders['stacking'] = StackingEnsembleBuilder(
                cv_folds=self.ensemble_config.stacking_cv_folds,
                use_features_in_secondary=self.ensemble_config.stacking_use_features_in_secondary
            )
        
        if self.ensemble_config.enable_blending:
            self.ensemble_builders['blending'] = BlendingEnsembleBuilder(
                holdout_size=self.ensemble_config.blending_holdout_size
            )
        
        logger.info(f"✅ Configured {len(self.ensemble_builders)} builders ensembles")
    
    def build_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        ensemble_methods: Optional[List[str]] = None,
        task_type: str = 'regression'
    ) -> EnsembleResult:
        """
        Main method building ensembles
        
        Args:
            X: Matrix features
            y: Target variable
            models: Dictionary base models
            ensemble_methods: Methods ensembling for use
            task_type: Type tasks (regression/classification)
        """
        start_time = time.time()
        
        logger.info(f"🤝 Launch building ensembles with {len(models)} base models")
        
        if ensemble_methods is None:
            ensemble_methods = list(self.ensemble_builders.keys())
        
        # Limitation number models in ensemble
        if len(models) > self.ensemble_config.ensemble_size_limit:
            # Sort models by performance and selection best
            sorted_models = self._rank_models_by_performance(X, y, models, task_type)
            models = dict(list(sorted_models.items())[:self.ensemble_config.ensemble_size_limit])
            logger.info(f"📝 Limited up to {len(models)} best models for ensemble")
        
        ensembles = {}
        ensemble_scores = {}
        ensemble_weights = {}
        base_model_scores = {}
        
        # Evaluate base models
        base_model_scores = self._evaluate_base_models(X, y, models, task_type)
        
        # Build ensembles
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        ) as progress:
            
            task = progress.add_task("Build ensembles...", total=len(ensemble_methods))
            
            for method in ensemble_methods:
                progress.update(task, description=f"Method: {method}")
                
                if method not in self.ensemble_builders:
                    logger.warning(f"⚠️ Unknown method ensembling: {method}")
                    continue
                
                try:
                    # Build ensemble
                    ensemble = self.ensemble_builders[method].build(
                        X, y, models, task_type=task_type
                    )
                    
                    if ensemble is not None:
                        ensembles[method] = ensemble
                        
                        # Evaluate ensemble
                        score = self._evaluate_ensemble(X, y, ensemble, task_type)
                        ensemble_scores[method] = score
                        
                        # Get weights (if applicable)
                        weights = self._extract_ensemble_weights(ensemble, method)
                        if weights:
                            ensemble_weights[method] = weights
                        
                        logger.info(f"✅ {method} ensemble: score {score:.4f}")
                
                except Exception as e:
                    logger.error(f"❌ Error building {method} ensemble: {e}")
                
                progress.advance(task)
        
        # Determine best ensemble
        if ensemble_scores:
            best_method = max(ensemble_scores.keys(), key=lambda k: ensemble_scores[k])
            best_score = ensemble_scores[best_method]
        else:
            best_method = "none"
            best_score = 0.0
            logger.warning("⚠️ one ensemble not was successfully built")
        
        build_time = time.time() - start_time
        
        result = EnsembleResult(
            ensembles=ensembles,
            ensemble_scores=ensemble_scores,
            best_ensemble_method=best_method,
            best_ensemble_score=best_score,
            base_model_scores=base_model_scores,
            ensemble_weights=ensemble_weights,
            ensemble_metadata={
                'task_type': task_type,
                'base_models_count': len(models),
                'ensemble_methods_tried': len(ensemble_methods),
                'successful_ensembles': len(ensembles)
            },
            build_time=build_time
        )
        
        # Output results
        self._print_ensemble_results(result)
        
        logger.info(f"✅ Build ensembles completed for {build_time:.2f}with")
        
        return result
    
    def _rank_models_by_performance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """Ranking models by performance"""
        logger.info("📊 Ranking models by performance...")
        
        model_scores = {}
        
        for name, model in models.items():
            try:
                # Fast evaluation with 3-fold CV
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1)
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
                
                model_scores[name] = np.mean(scores)
                
            except Exception as e:
                logger.warning(f"⚠️ Error evaluation model {name}: {e}")
                model_scores[name] = 0.0
        
        # Sort by descending score
        ranked_models = dict(
            sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return {name: models[name] for name in ranked_models.keys()}
    
    def _evaluate_base_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        models: Dict[str, Any],
        task_type: str
    ) -> Dict[str, float]:
        """Evaluate base models"""
        logger.info("📏 Evaluate base models...")
        
        base_scores = {}
        
        for name, model in models.items():
            try:
                if task_type == 'regression':
                    scores = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1)
                else:
                    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy', n_jobs=-1)
                
                base_scores[name] = np.mean(scores)
                
            except Exception as e:
                logger.warning(f"⚠️ Error evaluation base model {name}: {e}")
                base_scores[name] = 0.0
        
        return base_scores
    
    def _evaluate_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        ensemble: Any,
        task_type: str
    ) -> float:
        """Evaluate ensemble"""
        try:
            if task_type == 'regression':
                scores = cross_val_score(ensemble, X, y, cv=3, scoring='r2', n_jobs=-1)
            else:
                scores = cross_val_score(ensemble, X, y, cv=3, scoring='accuracy', n_jobs=-1)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"❌ Error evaluation ensemble: {e}")
            return 0.0
    
    def _extract_ensemble_weights(self, ensemble: Any, method: str) -> Optional[Dict[str, float]]:
        """Extraction weights ensemble"""
        try:
            if method == 'voting' and hasattr(ensemble, 'estimators_'):
                if hasattr(ensemble, 'weights') and ensemble.weights is not None:
                    estimator_names = [name for name, _ in ensemble.estimators]
                    return dict(zip(estimator_names, ensemble.weights))
            
            elif method == 'stacking' and hasattr(ensemble, 'final_estimator_'):
                if hasattr(ensemble.final_estimator_, 'coef_'):
                    estimator_names = [name for name, _ in ensemble.estimators]
                    weights = ensemble.final_estimator_.coef_
                    if len(weights) >= len(estimator_names):
                        return dict(zip(estimator_names, weights[:len(estimator_names)]))
            
            return None
            
        except Exception as e:
            logger.debug(f"Not succeeded extract weights for {method}: {e}")
            return None
    
    def _print_ensemble_results(self, result: EnsembleResult):
        """Output results ensembling"""
        
        # Table with ensembles
        table = Table(title="🤝 Results ENSEMBLING")
        
        table.add_column("Method", style="cyan", no_wrap=True)
        table.add_column("Score", style="green")
        table.add_column("Improvement", style="magenta")
        
        # Best base score for comparison
        best_base_score = max(result.base_model_scores.values()) if result.base_model_scores else 0.0
        
        for method, score in sorted(result.ensemble_scores.items(), key=lambda x: x[1], reverse=True):
            improvement = ((score - best_base_score) / best_base_score * 100) if best_base_score > 0 else 0.0
            table.add_row(
                method,
                f"{score:.4f}",
                f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            )
        
        self.console.print(table)
        
        # Information best ensemble
        if result.best_ensemble_method != "none":
            best_info = f"""
🏆 Best ensemble: {result.best_ensemble_method}
📊 Score: {result.best_ensemble_score:.4f}
⏱️ Time building: {result.build_time:.2f}with
🔢 Base models: {result.ensemble_metadata['base_models_count']}
"""
            self.console.print(best_info)
    
    def plot_ensemble_comparison(
        self,
        result: EnsembleResult,
        save_path: Optional[str] = None
    ):
        """Visualization comparison ensembles"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Chart 1: Comparison scores
            all_scores = {**result.base_model_scores, **result.ensemble_scores}
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            
            methods, scores = zip(*sorted_scores)
            colors = ['red' if method in result.ensemble_scores else 'blue' for method in methods]
            
            axes[0].barh(methods, scores, color=colors, alpha=0.7)
            axes[0].set_xlabel('Score')
            axes[0].set_title('Comparison base models and ensembles')
            axes[0].grid(True, alpha=0.3)
            
            # Legend
            axes[0].axvline(x=0, color='blue', alpha=0.7, label='Base model')
            axes[0].axvline(x=0, color='red', alpha=0.7, label='Ensembles')
            axes[0].legend()
            
            # Chart 2: Improvements from ensembling
            if result.ensemble_scores and result.base_model_scores:
                best_base_score = max(result.base_model_scores.values())
                
                improvements = {}
                for method, score in result.ensemble_scores.items():
                    improvement = ((score - best_base_score) / best_base_score * 100) if best_base_score > 0 else 0.0
                    improvements[method] = improvement
                
                if improvements:
                    methods, improve_values = zip(*improvements.items())
                    colors = ['green' if imp > 0 else 'orange' for imp in improve_values]
                    
                    axes[1].bar(methods, improve_values, color=colors, alpha=0.7)
                    axes[1].set_ylabel('Improvement (%)')
                    axes[1].set_title('Improvement from ensembling')
                    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Chart ensembles saved: {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"❌ Error creation ensembles: {e}")
    
    def get_ensemble_report(self, result: EnsembleResult) -> str:
        """Create report by ensembling"""
        
        report = f"""
=== REPORT By ENSEMBLING ===

Base models: {len(result.base_model_scores)}
Methods ensembling: {len(result.ensemble_scores)}
Time building: {result.build_time:.2f}with

Best ensemble: {result.best_ensemble_method}
Best score: {result.best_ensemble_score:.4f}

Results ensembles:
"""
        
        for method, score in sorted(result.ensemble_scores.items(), key=lambda x: x[1], reverse=True):
            report += f"  {method}: {score:.4f}\n"
        
        # Weights ensembles
        if result.ensemble_weights:
            report += "\nWeights in :\n"
            for method, weights in result.ensemble_weights.items():
                report += f"  {method}:\n"
                for model, weight in weights.items():
                    report += f"    {model}: {weight:.3f}\n"
        
        report += f"\nMetadata: {result.ensemble_metadata}"
        
        return report


if __name__ == "__main__":
    # Example use EnsembleBuilder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    import xgboost as xgb
    
    # Create test data
    np.random.seed(42)
    n_samples, n_features = 1000, 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y = pd.Series(
        X.iloc[:, :5].sum(axis=1) + 0.1 * np.random.randn(n_samples)
    )
    
    # Create base models
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=50, random_state=42)
    }
    
    # Create builder ensembles
    config = AutoMLConfig()
    builder = EnsembleBuilder(config)
    
    # Build ensembles
    result = builder.build_ensemble(
        X, y, models,
        ensemble_methods=['voting', 'stacking'],
        task_type='regression'
    )
    
    print("=== Results ENSEMBLING ===")
    print(f"Best ensemble: {result.best_ensemble_method}")
    print(f"Best score: {result.best_ensemble_score:.4f}")
    print(f"Time building: {result.build_time:.2f}with")
    
    # Report
    print(builder.get_ensemble_report(result))