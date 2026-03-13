"""
Comprehensive Model Evaluator for Crypto Trading AutoML
Implements enterprise patterns for thorough model evaluation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, cross_validate, TimeSeriesSplit
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import time
import joblib
from pathlib import Path

from ..utils.config_manager import AutoMLConfig


@dataclass
class EvaluationResult:
    """Result evaluation model"""
    model_name: str
    
    # Metrics for training set
    train_mse: float
    train_mae: float
    train_r2: float
    
    # Metrics for test set
    test_mse: float
    test_mae: float
    test_r2: float
    
    # Cross-validation
    cross_val_scores: List[float]
    
    # features
    feature_importance: Dict[str, float]
    
    # Metadata
    evaluation_metadata: Dict[str, Any]
    evaluation_time: float


class CryptoTradingMetrics:
    """Specialized metrics for crypto trading"""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Computation coefficient Sharpe"""
        try:
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return (mean_return - risk_free_rate) / std_return
        except Exception:
            return 0.0
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Computation coefficient Sortino"""
        try:
            if len(returns) == 0:
                return 0.0
            
            mean_return = np.mean(returns)
            downside_returns = returns[returns < risk_free_rate]
            
            if len(downside_returns) == 0:
                return float('inf') if mean_return > risk_free_rate else 0.0
            
            downside_deviation = np.std(downside_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            return (mean_return - risk_free_rate) / downside_deviation
        except Exception:
            return 0.0
    
    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> float:
        """Computation maximum drawdown"""
        try:
            if len(returns) == 0:
                return 0.0
            
            cumulative_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            
            return np.min(drawdown)
        except Exception:
            return 0.0
    
    @staticmethod
    def win_rate(predictions: np.ndarray, actuals: np.ndarray, threshold: float = 0.0) -> float:
        """Computation profitable trades"""
        try:
            if len(predictions) == 0 or len(actuals) == 0:
                return 0.0
            
            # Determine directions predictions and actual values
            pred_direction = predictions > threshold
            actual_direction = actuals > threshold
            
            # Count correct predictions directions
            correct_predictions = pred_direction == actual_direction
            
            return np.mean(correct_predictions)
        except Exception:
            return 0.0
    
    @staticmethod
    def profit_factor(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Computation profit"""
        try:
            if len(predictions) == 0 or len(actuals) == 0:
                return 0.0
            
            # Assume we enter a position based on prediction
            # and getting return equal current value
            profitable_trades = actuals[predictions > 0]
            losing_trades = actuals[predictions < 0]
            
            gross_profit = np.sum(profitable_trades[profitable_trades > 0])
            gross_loss = -np.sum(losing_trades[losing_trades < 0])
            
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0
            
            return gross_profit / gross_loss
        except Exception:
            return 0.0
    
    @staticmethod
    def information_ratio(predictions: np.ndarray, actuals: np.ndarray, benchmark_return: float = 0.0) -> float:
        """Computation information coefficient"""
        try:
            if len(predictions) == 0 or len(actuals) == 0:
                return 0.0
            
            # Active return (exceeding over benchmark)
            active_returns = actuals - benchmark_return
            
            # Error tracking
            tracking_error = np.std(active_returns)
            
            if tracking_error == 0:
                return 0.0
            
            return np.mean(active_returns) / tracking_error
        except Exception:
            return 0.0


class ModelEvaluator:
    """
    Comprehensive models for crypto trading
    Implements enterprise patterns
    """
    
    def __init__(self, config: Optional[AutoMLConfig] = None):
        self.config = config or AutoMLConfig()
        self.evaluation_config = self.config.model_evaluation
        self.console = Console()
        
        # Cache for SHAP objects (if is used)
        self.shap_cache = {}
        
        logger.info("📊 ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "unknown",
        task_type: str = "regression"
    ) -> EvaluationResult:
        """
        Comprehensive evaluation model
        
        Args:
            model: Trained model
            X_train: Features training set
            y_train: Target variable training set
            X_test: Features test set
            y_test: Target variable test set
            model_name: Name model
            task_type: Type tasks (regression/classification)
        """
        start_time = time.time()
        
        logger.info(f"📊 Evaluate model {model_name}...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
            ) as progress:
                
                task = progress.add_task("Evaluate model...", total=None)
                
                # Predictions
                progress.update(task, description="Get predictions...")
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # Base metrics
                progress.update(task, description="Computation base metrics...")
                if task_type == "regression":
                    metrics = self._calculate_regression_metrics(
                        y_train, y_train_pred, y_test, y_test_pred
                    )
                else:
                    metrics = self._calculate_classification_metrics(
                        y_train, y_train_pred, y_test, y_test_pred
                    )
                
                # Crypto-trading metrics
                progress.update(task, description="Computation crypto trading metrics...")
                crypto_metrics = self._calculate_crypto_metrics(
                    y_test_pred, y_test
                )
                
                # Cross-validation
                progress.update(task, description="Cross-validation...")
                cv_scores = self._perform_cross_validation(
                    model, X_train, y_train, task_type
                )
                
                # features
                progress.update(task, description="Computation features...")
                feature_importance = self._calculate_feature_importance(
                    model, X_train, y_train, model_name
                )
                
                progress.update(task, description="✅ Evaluate completed", completed=True)
            
            evaluation_time = time.time() - start_time
            
            # Create
            if task_type == "regression":
                result = EvaluationResult(
                    model_name=model_name,
                    train_mse=metrics['train_mse'],
                    train_mae=metrics['train_mae'],
                    train_r2=metrics['train_r2'],
                    test_mse=metrics['test_mse'],
                    test_mae=metrics['test_mae'],
                    test_r2=metrics['test_r2'],
                    cross_val_scores=cv_scores,
                    feature_importance=feature_importance,
                    evaluation_metadata={
                        'task_type': task_type,
                        'crypto_metrics': crypto_metrics,
                        'train_samples': len(y_train),
                        'test_samples': len(y_test),
                        'features_count': len(X_train.columns)
                    },
                    evaluation_time=evaluation_time
                )
            else:
                # Adaptation for classification
                result = EvaluationResult(
                    model_name=model_name,
                    train_mse=0.0, # Not applicable
                    train_mae=0.0, # Not applicable
                    train_r2=metrics.get('train_accuracy', 0.0),
                    test_mse=0.0, # Not applicable
                    test_mae=0.0, # Not applicable
                    test_r2=metrics.get('test_accuracy', 0.0),
                    cross_val_scores=cv_scores,
                    feature_importance=feature_importance,
                    evaluation_metadata={
                        'task_type': task_type,
                        'classification_metrics': metrics,
                        'train_samples': len(y_train),
                        'test_samples': len(y_test),
                        'features_count': len(X_train.columns)
                    },
                    evaluation_time=evaluation_time
                )
            
            # Output results
            self._print_evaluation_results(result)
            
            logger.info(f"✅ Evaluate model {model_name} completed for {evaluation_time:.2f}with")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error evaluation model {model_name}: {e}")
            evaluation_time = time.time() - start_time
            
            # Return empty result in case errors
            return EvaluationResult(
                model_name=model_name,
                train_mse=0.0, train_mae=0.0, train_r2=0.0,
                test_mse=0.0, test_mae=0.0, test_r2=0.0,
                cross_val_scores=[], feature_importance={},
                evaluation_metadata={'error': str(e)},
                evaluation_time=evaluation_time
            )
    
    def _calculate_regression_metrics(
        self,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        y_test: pd.Series,
        y_test_pred: np.ndarray
    ) -> Dict[str, float]:
        """Computation metrics regression"""
        
        metrics = {}
        
        # Training sample
        metrics['train_mse'] = mean_squared_error(y_train, y_train_pred)
        metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
        metrics['train_r2'] = r2_score(y_train, y_train_pred)
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        
        # Test sample
        metrics['test_mse'] = mean_squared_error(y_test, y_test_pred)
        metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        
        # MAPE (Mean Absolute Percentage Error)
        try:
            def mape(actual, pred):
                mask = actual != 0
                if mask.sum() == 0:
                    return 0.0
                return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100
            
            metrics['train_mape'] = mape(y_train.values, y_train_pred)
            metrics['test_mape'] = mape(y_test.values, y_test_pred)
        except:
            metrics['train_mape'] = 0.0
            metrics['test_mape'] = 0.0
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        y_train: pd.Series,
        y_train_pred: np.ndarray,
        y_test: pd.Series,
        y_test_pred: np.ndarray
    ) -> Dict[str, float]:
        """Computation metrics classification"""
        
        metrics = {}
        
        # Training sample
        metrics['train_accuracy'] = accuracy_score(y_train, y_train_pred)
        metrics['train_precision'] = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train_recall'] = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
        metrics['train_f1'] = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
        
        # Test sample
        metrics['test_accuracy'] = accuracy_score(y_test, y_test_pred)
        metrics['test_precision'] = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['test_recall'] = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        metrics['test_f1'] = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # AUC (if binary classification)
        try:
            if len(np.unique(y_test)) == 2:
                # Needed probability for AUC
                if hasattr(self, 'predict_proba'):
                    y_test_proba = y_test_pred # Assuming that this already probability
                    metrics['test_auc'] = roc_auc_score(y_test, y_test_proba)
                else:
                    metrics['test_auc'] = 0.0
            else:
                metrics['test_auc'] = 0.0
        except:
            metrics['test_auc'] = 0.0
        
        return metrics
    
    def _calculate_crypto_metrics(
        self,
        predictions: np.ndarray,
        actuals: pd.Series
    ) -> Dict[str, float]:
        """Computation specialized metrics for crypto trading"""
        
        crypto_metrics = {}
        
        try:
            actuals_array = actuals.values
            
            # Coefficient Sharpe
            crypto_metrics['sharpe_ratio'] = CryptoTradingMetrics.sharpe_ratio(actuals_array)
            
            # Coefficient Sortino
            crypto_metrics['sortino_ratio'] = CryptoTradingMetrics.sortino_ratio(actuals_array)
            
            # Maximum drawdown
            crypto_metrics['max_drawdown'] = CryptoTradingMetrics.maximum_drawdown(actuals_array)
            
            # profitable trades
            crypto_metrics['win_rate'] = CryptoTradingMetrics.win_rate(predictions, actuals_array)
            
            # profit
            crypto_metrics['profit_factor'] = CryptoTradingMetrics.profit_factor(predictions, actuals_array)
            
            # Information coefficient
            crypto_metrics['information_ratio'] = CryptoTradingMetrics.information_ratio(
                predictions, actuals_array
            )
            
            # Correlation predictions and actual values
            crypto_metrics['prediction_correlation'] = np.corrcoef(predictions, actuals_array)[0, 1]
            
        except Exception as e:
            logger.warning(f"⚠️ Error computation : {e}")
            crypto_metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'information_ratio': 0.0,
                'prediction_correlation': 0.0
            }
        
        return crypto_metrics
    
    def _perform_cross_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str
    ) -> List[float]:
        """Execute cross-validation"""
        
        try:
            cv_folds = self.evaluation_config.cv_folds
            cv_scoring = self.evaluation_config.cv_scoring
            
            # Determine scoring by default
            if cv_scoring is None:
                if task_type == "regression":
                    cv_scoring = 'r2'
                else:
                    cv_scoring = 'accuracy'
            
            # Usage TimeSeriesSplit for temporal series
            if hasattr(self.config, 'crypto_specific') and self.config.crypto_specific.get('walk_forward_validation', False):
                cv = TimeSeriesSplit(n_splits=cv_folds)
            else:
                cv = cv_folds
            
            # Execute cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring=cv_scoring,
                n_jobs=-1,
                error_score='raise'
            )
            
            return scores.tolist()
            
        except Exception as e:
            logger.warning(f"⚠️ Error cross-validation: {e}")
            return []
    
    def _calculate_feature_importance(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str
    ) -> Dict[str, float]:
        """Computation features"""
        
        try:
            method = self.evaluation_config.feature_importance_method
            
            if method == 'built_in' and hasattr(model, 'feature_importances_'):
                # Built-in features
                importances = model.feature_importances_
                return dict(zip(X.columns, importances))
                
            elif method == 'permutation':
                # Permutation importance
                try:
                    perm_importance = permutation_importance(
                        model, X, y,
                        n_repeats=5,
                        random_state=42,
                        n_jobs=-1
                    )
                    return dict(zip(X.columns, perm_importance.importances_mean))
                except Exception as e:
                    logger.warning(f"⚠️ Error permutation importance: {e}")
                    return {}
                    
            elif method == 'shap' and model_name not in self.shap_cache:
                # SHAP values (more slow, but informative)
                try:
                    # Limit size set for SHAP
                    sample_size = min(100, len(X))
                    X_sample = X.sample(n=sample_size, random_state=42)
                    
                    if hasattr(model, 'predict_proba'):
                        explainer = shap.TreeExplainer(model)
                    else:
                        explainer = shap.Explainer(model, X_sample)
                    
                    shap_values = explainer.shap_values(X_sample)
                    
                    # If classification, take first class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    
                    # Average absolute by all features
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    
                    self.shap_cache[model_name] = dict(zip(X.columns, mean_abs_shap))
                    return self.shap_cache[model_name]
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error SHAP importance: {e}")
                    return {}
            
            # Fallback: equal all features
            return {col: 1.0 / len(X.columns) for col in X.columns}
            
        except Exception as e:
            logger.warning(f"⚠️ Error computation features: {e}")
            return {}
    
    def _print_evaluation_results(self, result: EvaluationResult):
        """Output results evaluation"""
        
        # Main table with metrics
        table = Table(title=f"📊 Evaluation Model: {result.model_name.upper()}")
        
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Training", style="green")
        table.add_column("Test", style="magenta")
        
        if result.evaluation_metadata.get('task_type') == 'regression':
            table.add_row("R²", f"{result.train_r2:.4f}", f"{result.test_r2:.4f}")
            table.add_row("MSE", f"{result.train_mse:.4f}", f"{result.test_mse:.4f}")
            table.add_row("MAE", f"{result.train_mae:.4f}", f"{result.test_mae:.4f}")
            
            if 'train_rmse' in result.evaluation_metadata.get('crypto_metrics', {}):
                table.add_row("RMSE", f"{np.sqrt(result.train_mse):.4f}", f"{np.sqrt(result.test_mse):.4f}")
        else:
            # Classification
            metrics = result.evaluation_metadata.get('classification_metrics', {})
            table.add_row("Accuracy", f"{metrics.get('train_accuracy', 0):.4f}", f"{metrics.get('test_accuracy', 0):.4f}")
            table.add_row("Precision", f"{metrics.get('train_precision', 0):.4f}", f"{metrics.get('test_precision', 0):.4f}")
            table.add_row("Recall", f"{metrics.get('train_recall', 0):.4f}", f"{metrics.get('test_recall', 0):.4f}")
            table.add_row("F1-Score", f"{metrics.get('train_f1', 0):.4f}", f"{metrics.get('test_f1', 0):.4f}")
        
        self.console.print(table)
        
        # Cross-validation
        if result.cross_val_scores:
            cv_mean = np.mean(result.cross_val_scores)
            cv_std = np.std(result.cross_val_scores)
            cv_info = f"📊 Cross-validation: {cv_mean:.4f} ± {cv_std:.4f} (n={len(result.cross_val_scores)})"
            self.console.print(cv_info)
        
        # Crypto-trading metrics
        crypto_metrics = result.evaluation_metadata.get('crypto_metrics', {})
        if crypto_metrics:
            crypto_table = Table(title="💰 CRYPTO-TRADING Metrics")
            crypto_table.add_column("Metric", style="cyan")
            crypto_table.add_column("Value", style="yellow")
            
            for metric, value in crypto_metrics.items():
                if isinstance(value, float):
                    crypto_table.add_row(metric.replace('_', ' ').title(), f"{value:.4f}")
            
            self.console.print(crypto_table)
        
        # Top features
        if result.feature_importance:
            top_features = sorted(
                result.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]
            
            if top_features:
                features_info = "🔍 Top-10 features:\n"
                for i, (feature, importance) in enumerate(top_features, 1):
                    features_info += f"  {i:2d}. {feature}: {importance:.4f}\n"
                
                self.console.print(features_info)
    
    def compare_models(
        self,
        results: List[EvaluationResult],
        save_path: Optional[str] = None
    ):
        """Comparison several models"""
        
        if not results:
            logger.warning("⚠️ No results for comparison")
            return
        
        logger.info(f"📊 Comparison {len(results)} models")
        
        # Table comparison
        comparison_table = Table(title="🏆 COMPARISON Models")
        
        comparison_table.add_column("Rank", style="cyan", no_wrap=True)
        comparison_table.add_column("Model", style="magenta")
        comparison_table.add_column("Test R²", style="green")
        comparison_table.add_column("Test RMSE", style="yellow")
        comparison_table.add_column("CV Score", style="blue")
        
        # Sort by test R²
        sorted_results = sorted(results, key=lambda x: x.test_r2, reverse=True)
        
        for i, result in enumerate(sorted_results, 1):
            cv_mean = np.mean(result.cross_val_scores) if result.cross_val_scores else 0.0
            test_rmse = np.sqrt(result.test_mse) if result.test_mse > 0 else 0.0
            
            comparison_table.add_row(
                str(i),
                result.model_name,
                f"{result.test_r2:.4f}",
                f"{test_rmse:.4f}",
                f"{cv_mean:.4f}"
            )
        
        self.console.print(comparison_table)
        
        # Create comparison
        if save_path:
            self.plot_models_comparison(results, save_path)
    
    def plot_models_comparison(
        self,
        results: List[EvaluationResult],
        save_path: str
    ):
        """Create comparison models"""
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            model_names = [r.model_name for r in results]
            
            # Chart 1: R² scores
            train_r2 = [r.train_r2 for r in results]
            test_r2 = [r.test_r2 for r in results]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, train_r2, width, label='Train', alpha=0.7)
            axes[0, 0].bar(x + width/2, test_r2, width, label='Test', alpha=0.7)
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].set_title('R² Score by models')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Chart 2: RMSE
            train_rmse = [np.sqrt(r.train_mse) for r in results]
            test_rmse = [np.sqrt(r.test_mse) for r in results]
            
            axes[0, 1].bar(x - width/2, train_rmse, width, label='Train', alpha=0.7)
            axes[0, 1].bar(x + width/2, test_rmse, width, label='Test', alpha=0.7)
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].set_title('RMSE by models')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Chart 3: Cross-validation scores
            cv_means = [np.mean(r.cross_val_scores) if r.cross_val_scores else 0 for r in results]
            cv_stds = [np.std(r.cross_val_scores) if r.cross_val_scores else 0 for r in results]
            
            axes[1, 0].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7)
            axes[1, 0].set_xlabel('Model')
            axes[1, 0].set_ylabel('CV Score')
            axes[1, 0].set_title('Cross-validation Scores')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Chart 4: Time evaluation
            eval_times = [r.evaluation_time for r in results]
            
            axes[1, 1].bar(model_names, eval_times, alpha=0.7, color='orange')
            axes[1, 1].set_xlabel('Model')
            axes[1, 1].set_ylabel('Time evaluation (with)')
            axes[1, 1].set_title('Time evaluation models')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Chart comparison saved: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Error creation comparison: {e}")
    
    def generate_evaluation_report(
        self,
        results: List[EvaluationResult],
        save_path: Optional[str] = None
    ) -> str:
        """Generation detailed report by """
        
        if not results:
            return "No results for creation report"
        
        report = f"""
=== DETAILED REPORT By EVALUATION Models ===

Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Number models: {len(results)}

"""
        
        # table
        report += "SUMMARY TABLE:\n"
        report += f"{'Model':<20} {'Test R²':<10} {'Test RMSE':<12} {'CV Mean':<10} {'Time':<8}\n"
        report += "-" * 70 + "\n"
        
        sorted_results = sorted(results, key=lambda x: x.test_r2, reverse=True)
        
        for result in sorted_results:
            cv_mean = np.mean(result.cross_val_scores) if result.cross_val_scores else 0.0
            test_rmse = np.sqrt(result.test_mse)
            
            report += f"{result.model_name:<20} {result.test_r2:<10.4f} {test_rmse:<12.4f} {cv_mean:<10.4f} {result.evaluation_time:<8.2f}\n"
        
        # Detailed information by each model
        report += "\n" + "="*70 + "\n"
        report += "DETAILED INFORMATION By MODELS:\n\n"
        
        for i, result in enumerate(sorted_results, 1):
            report += f"{i}. Model: {result.model_name.upper()}\n"
            report += "-" * 40 + "\n"
            
            # Main metrics
            report += f"Training sample:\n"
            report += f"  R²: {result.train_r2:.4f}\n"
            report += f"  MSE: {result.train_mse:.4f}\n"
            report += f"  MAE: {result.train_mae:.4f}\n"
            
            report += f"\nTest sample:\n"
            report += f"  R²: {result.test_r2:.4f}\n"
            report += f"  MSE: {result.test_mse:.4f}\n"
            report += f"  MAE: {result.test_mae:.4f}\n"
            
            # Cross-validation
            if result.cross_val_scores:
                cv_mean = np.mean(result.cross_val_scores)
                cv_std = np.std(result.cross_val_scores)
                report += f"\nCross-validation:\n"
                report += f"  Average: {cv_mean:.4f}\n"
                report += f" Std. deviation: {cv_std:.4f}\n"
                report += f"  Minimum: {np.min(result.cross_val_scores):.4f}\n"
                report += f" Maximum: {np.max(result.cross_val_scores):.4f}\n"
            
            # Crypto-trading metrics
            crypto_metrics = result.evaluation_metadata.get('crypto_metrics', {})
            if crypto_metrics:
                report += f"\nCrypto-trading metrics:\n"
                for metric, value in crypto_metrics.items():
                    if isinstance(value, float):
                        report += f"  {metric.replace('_', ' ').title()}: {value:.4f}\n"
            
            # Top features
            if result.feature_importance:
                top_features = sorted(
                    result.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                
                report += f"\nTop-5 features:\n"
                for j, (feature, importance) in enumerate(top_features, 1):
                    report += f"  {j}. {feature}: {importance:.4f}\n"
            
            # Metadata
            report += f"\nMetadata:\n"
            report += f" Time evaluation: {result.evaluation_time:.2f}with\n"
            report += f" Training samples: {result.evaluation_metadata.get('train_samples', 'N/A')}\n"
            report += f"  Test samples: {result.evaluation_metadata.get('test_samples', 'N/A')}\n"
            report += f"  Number features: {result.evaluation_metadata.get('features_count', 'N/A')}\n"
            
            report += "\n"
        
        # Recommendations
        report += "="*70 + "\n"
        report += "RECOMMENDATIONS:\n\n"
        
        best_model = sorted_results[0]
        report += f"🏆 Best model: {best_model.model_name}\n"
        report += f"📊 Test R²: {best_model.test_r2:.4f}\n"
        
        if len(results) > 1:
            second_best = sorted_results[1]
            improvement = ((best_model.test_r2 - second_best.test_r2) / second_best.test_r2 * 100) if second_best.test_r2 > 0 else 0
            report += f"📈 Superiority over second : {improvement:.2f}%\n"
        
        # Analysis
        overfitting = best_model.train_r2 - best_model.test_r2
        if overfitting > 0.1:
            report += f"⚠️ Possible (difference Train-Test R²: {overfitting:.4f})\n"
            report += " Recommended boosting regularization or collection additional data\n"
        
        # Save report
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"📝 Report saved: {save_path}")
        
        return report


if __name__ == "__main__":
    # Example use ModelEvaluator
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and training models
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=50, random_state=42)
    }
    
    results = []
    
    # Create evaluator
    config = AutoMLConfig()
    evaluator = ModelEvaluator(config)
    
    # Evaluate each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        result = evaluator.evaluate_model(
            model, X_train, y_train, X_test, y_test,
            model_name=name
        )
        
        results.append(result)
    
    # Comparison models
    evaluator.compare_models(results)
    
    # Generation report
    report = evaluator.generate_evaluation_report(results)
    print("\n" + "="*50)
    print("BRIEF REPORT:")
    print(report[:1000] + "...")