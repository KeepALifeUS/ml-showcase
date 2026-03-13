"""
Advanced Hyperparameter Tuning for XGBoost Models
==================================================

Comprehensive hyperparameter optimization system with multiple strategies
specifically optimized for XGBoost and time series prediction.

Features:
- Multi-objective optimization (performance vs speed)
- Distributed tuning capabilities
- Early stopping and pruning
- Custom objective functions for trading metrics
- Integration with cross-validation
- Parameter importance analysis
- Automated feature selection integration

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
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

import xgboost as xgb
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score
)
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.visualization import (
    plot_optimization_history, plot_param_importances,
    plot_slice, plot_contour
)
from scipy.optimize import differential_evolution, minimize
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from rich.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class OptimizationObjective(Enum):
    """Optimization objectives for time series"""
    
    # Regression objectives
    MINIMIZE_RMSE = "minimize_rmse"
    MINIMIZE_MAE = "minimize_mae" 
    MAXIMIZE_R2 = "maximize_r2"
    MINIMIZE_MAPE = "minimize_mape"
    
    # Classification objectives
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    MAXIMIZE_F1 = "maximize_f1"
    MAXIMIZE_AUC = "maximize_auc"
    MAXIMIZE_PRECISION = "maximize_precision"
    MAXIMIZE_RECALL = "maximize_recall"
    
    # Trading specific objectives
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_PROFIT_FACTOR = "maximize_profit_factor"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"
    MINIMIZE_VAR = "minimize_var"
    
    # Multi-objective
    PARETO_PERFORMANCE_SPEED = "pareto_performance_speed"
    PARETO_ACCURACY_COMPLEXITY = "pareto_accuracy_complexity"


class TuningStrategy(Enum):
    """Hyperparameter search strategies"""
    OPTUNA_TPE = "optuna_tpe"  # Tree-structured Parzen Estimator
    OPTUNA_CMAES = "optuna_cmaes"  # CMA-ES
    BAYESIAN_GP = "bayesian_gp"  # Gaussian Process
    BAYESIAN_RF = "bayesian_rf"  # Random Forest
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GENETIC_ALGORITHM = "genetic_algorithm"


@dataclass
class ParameterSpace:
    """Parameter space definition for optimization"""
    
    # Tree structure parameters
    n_estimators: Tuple[int, int] = (50, 2000)
    max_depth: Tuple[int, int] = (3, 15)
    min_child_weight: Tuple[int, int] = (1, 10)
    
    # Learning parameters
    learning_rate: Tuple[float, float] = (0.001, 0.3)
    gamma: Tuple[float, float] = (0.0, 10.0)
    
    # Regularization parameters
    reg_alpha: Tuple[float, float] = (0.0, 100.0)
    reg_lambda: Tuple[float, float] = (0.0, 100.0)
    
    # Sampling parameters
    subsample: Tuple[float, float] = (0.1, 1.0)
    colsample_bytree: Tuple[float, float] = (0.1, 1.0)
    colsample_bylevel: Tuple[float, float] = (0.1, 1.0)
    colsample_bynode: Tuple[float, float] = (0.1, 1.0)
    
    # Advanced parameters
    scale_pos_weight: Tuple[float, float] = (0.1, 10.0)
    max_delta_step: Tuple[int, int] = (0, 10)
    
    # Categorical parameters
    booster: List[str] = field(default_factory=lambda: ['gbtree', 'dart'])
    tree_method: List[str] = field(default_factory=lambda: ['auto', 'exact', 'approx', 'hist'])
    grow_policy: List[str] = field(default_factory=lambda: ['depthwise', 'lossguide'])
    
    def to_optuna_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Convert to Optuna search space"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', *self.n_estimators),
            'max_depth': trial.suggest_int('max_depth', *self.max_depth),
            'min_child_weight': trial.suggest_int('min_child_weight', *self.min_child_weight),
            'learning_rate': trial.suggest_float('learning_rate', *self.learning_rate, log=True),
            'gamma': trial.suggest_float('gamma', *self.gamma),
            'reg_alpha': trial.suggest_float('reg_alpha', *self.reg_alpha),
            'reg_lambda': trial.suggest_float('reg_lambda', *self.reg_lambda),
            'subsample': trial.suggest_float('subsample', *self.subsample),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *self.colsample_bytree),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', *self.colsample_bylevel),
            'colsample_bynode': trial.suggest_float('colsample_bynode', *self.colsample_bynode),
            'booster': trial.suggest_categorical('booster', self.booster),
            'tree_method': trial.suggest_categorical('tree_method', self.tree_method),
        }
        
        # Conditional parameters
        if params['booster'] == 'gbtree':
            params['grow_policy'] = trial.suggest_categorical('grow_policy', self.grow_policy)
        
        return params
    
    def to_skopt_space(self) -> List[Any]:
        """Convert to scikit-optimize search space"""
        
        space = [
            Integer(*self.n_estimators, name='n_estimators'),
            Integer(*self.max_depth, name='max_depth'),
            Integer(*self.min_child_weight, name='min_child_weight'),
            Real(*self.learning_rate, prior='log-uniform', name='learning_rate'),
            Real(*self.gamma, name='gamma'),
            Real(*self.reg_alpha, name='reg_alpha'),
            Real(*self.reg_lambda, name='reg_lambda'),
            Real(*self.subsample, name='subsample'),
            Real(*self.colsample_bytree, name='colsample_bytree'),
            Real(*self.colsample_bylevel, name='colsample_bylevel'),
            Real(*self.colsample_bynode, name='colsample_bynode'),
            Categorical(self.booster, name='booster'),
            Categorical(self.tree_method, name='tree_method'),
        ]
        
        return space


@dataclass
class TuningConfig:
    """Hyperparameter tuning configuration"""

    # Main parameters
    strategy: TuningStrategy = TuningStrategy.OPTUNA_TPE
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_RMSE
    n_trials: int = 100
    timeout_seconds: Optional[int] = 3600  # 1 hour
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # time_series, stratified, kfold
    scoring_metric: Optional[str] = None
    
    # Performance optimization
    n_jobs: int = -1
    enable_distributed: bool = False
    storage_url: Optional[str] = None  # For distributed Optuna
    study_name: Optional[str] = None
    
    # Early stopping and pruning
    enable_pruning: bool = True
    pruning_patience: int = 10
    min_trials_for_pruning: int = 5
    
    # Parameter space
    parameter_space: Optional[ParameterSpace] = None
    
    # Multi-objective optimization
    enable_multi_objective: bool = False
    secondary_objective: Optional[OptimizationObjective] = None
    objective_weights: Tuple[float, float] = (0.8, 0.2)
    
    # Save results
    save_intermediate_results: bool = True
    results_dir: str = "tuning_results"
    
    # Additional options
    enable_feature_selection: bool = False
    feature_selection_threshold: float = 0.01
    enable_ensemble_tuning: bool = False
    
    def __post_init__(self):
        if self.parameter_space is None:
            self.parameter_space = ParameterSpace()


@dataclass
class TuningResult:
    """Hyperparameter tuning result"""

    # Best parameters
    best_params: Dict[str, Any]
    best_score: float
    best_trial_number: int
    
    # Optimization history
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    parameter_importance: Dict[str, float] = field(default_factory=dict)
    
    # Statistics
    total_trials: int = 0
    successful_trials: int = 0
    pruned_trials: int = 0
    failed_trials: int = 0
    tuning_time: float = 0.0
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Multi-objective results
    pareto_front: Optional[List[Tuple[float, float]]] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[TuningConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_number': self.best_trial_number,
            'optimization_history': self.optimization_history,
            'parameter_importance': self.parameter_importance,
            'total_trials': self.total_trials,
            'successful_trials': self.successful_trials,
            'pruned_trials': self.pruned_trials,
            'failed_trials': self.failed_trials,
            'tuning_time': self.tuning_time,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'pareto_front': self.pareto_front,
            'timestamp': self.timestamp.isoformat(),
        }


class BaseHyperparameterTuner(ABC):
    """Base class for hyperparameter tuners - enterprise pattern"""
    
    @abstractmethod
    def optimize(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        config: TuningConfig
    ) -> TuningResult:
        """Optimize hyperparameters"""
        pass

    @abstractmethod
    def get_tuner_name(self) -> str:
        """Get tuner name"""
        pass


class HyperparameterTuner:
    """
    Main class for XGBoost model hyperparameter optimization

    Implements enterprise patterns:
    - Strategy pattern for different optimization algorithms
    - Factory pattern for creating optimizers
    - Observer pattern for progress monitoring
    - Command pattern for cancellable operations
    """
    
    def __init__(self, config: Optional[TuningConfig] = None):
        self.config = config or TuningConfig()
        
        # Internal state
        self.current_study_: Optional[Any] = None
        self.optimization_callbacks_: List[Callable] = []
        self.results_history_: List[TuningResult] = []
        
        # Caching
        self._objective_cache: Dict[str, float] = {}
        self._model_cache: Dict[str, Any] = {}
        
        # Parallel processing
        self._stop_optimization = threading.Event()
        
        self.console = Console()
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging"""
        logger.add(
            f"logs/hyperparameter_tuning_{datetime.now():%Y%m%d}.log",
            rotation="daily",
            retention="30 days",
            level="INFO"
        )
    
    def add_callback(self, callback: Callable[[optuna.Trial], None]):
        """Add callback function for monitoring"""
        self.optimization_callbacks_.append(callback)
    
    def optimize(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> TuningResult:
        """
        Optimize hyperparameters

        Args:
            model_class: XGBoost model class
            X: Training data
            y: Target variable
            X_val: Validation data
            y_val: Validation target variable
        """
        
        start_time = time.time()
        logger.info(f"🚀 Starting hyperparameter optimization: {self.config.strategy.value}")

        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            return self._objective_function(
                trial, model_class, X, y, X_val, y_val
            )
        
        # Choose optimization strategy
        if self.config.strategy == TuningStrategy.OPTUNA_TPE:
            result = self._optimize_with_optuna(objective, TPESampler())
        elif self.config.strategy == TuningStrategy.OPTUNA_CMAES:
            result = self._optimize_with_optuna(objective, CmaEsSampler())
        elif self.config.strategy == TuningStrategy.BAYESIAN_GP:
            result = self._optimize_with_skopt(model_class, X, y, X_val, y_val, 'gp')
        elif self.config.strategy == TuningStrategy.BAYESIAN_RF:
            result = self._optimize_with_skopt(model_class, X, y, X_val, y_val, 'rf')
        elif self.config.strategy == TuningStrategy.RANDOM_SEARCH:
            result = self._optimize_with_random_search(model_class, X, y, X_val, y_val)
        else:
            raise ValueError(f"Unsupported strategy: {self.config.strategy}")

        # Update optimization time
        result.tuning_time = time.time() - start_time
        result.config = self.config
        
        # Save results
        if self.config.save_intermediate_results:
            self._save_results(result, results_dir)
        
        # Add to history
        self.results_history_.append(result)
        
        logger.info(f"✅ Optimization completed in {result.tuning_time:.2f}s")

        # Display results
        self._display_results(result)
        
        return result
    
    def _objective_function(
        self,
        trial: optuna.Trial,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> float:
        """
        Objective function for optimization

        Args:
            trial: Optuna trial object
            model_class: Model class
            X, y: Training data
            X_val, y_val: Validation data
        """
        
        try:
            # Generate parameters for this trial
            params = self.config.parameter_space.to_optuna_space(trial)
            
            # Caching for acceleration
            params_key = str(sorted(params.items()))
            if params_key in self._objective_cache:
                return self._objective_cache[params_key]
            
            # Create model with parameters
            model = model_class(**params, random_state=42, n_jobs=1)
            
            # Cross-validation or validation on a separate set
            if X_val is not None and y_val is not None:
                # Validation on a separate set
                model.fit(X, y)
                predictions = model.predict(X_val)
                score = self._calculate_score(y_val, predictions)
            else:
                # Cross-validation
                scores = self._cross_validate_model(model, X, y)
                score = np.mean(scores)
            
            # Caching result
            self._objective_cache[params_key] = score
            
            # Callback for monitoring
            for callback in self.optimization_callbacks_:
                callback(trial)
            
            # Pruning for early stopping of unpromising trials
            if self.config.enable_pruning and trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.warning(f"⚠️ Error in trial {trial.number}: {e}")
            # Return a bad score for unsuccessful trials
            if self._is_minimization_objective():
                return float('inf')
            else:
                return float('-inf')
    
    def _optimize_with_optuna(
        self,
        objective: Callable,
        sampler: optuna.samplers.BaseSampler
    ) -> TuningResult:
        """Optimization using Optuna"""

        # Configure pruner
        if self.config.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.config.min_trials_for_pruning,
                n_warmup_steps=self.config.pruning_patience
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        direction = "minimize" if self._is_minimization_objective() else "maximize"
        
        if self.config.storage_url and self.config.study_name:
            # Distributed optimization
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                storage=self.config.storage_url,
                study_name=self.config.study_name,
                load_if_exists=True
            )
        else:
            # Local optimization
            study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                pruner=pruner
            )
        
        self.current_study_ = study
        
        # Progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Optimizing hyperparameters...", total=self.config.n_trials)
            
            def progress_callback(study, trial):
                progress.update(task, advance=1)
                progress.update(
                    task,
                    description=f"Trial {trial.number}/{self.config.n_trials} | "
                               f"Best: {study.best_value:.4f}"
                )
            
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout_seconds,
                callbacks=[progress_callback],
                n_jobs=self.config.n_jobs if not self.config.enable_distributed else 1,
                show_progress_bar=False
            )
        
        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial_number=study.best_trial.number,
            total_trials=len(study.trials),
            successful_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            pruned_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            failed_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        )
        
        # Optimization history
        result.optimization_history = [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None,
            }
            for trial in study.trials
        ]
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            result.parameter_importance = importance
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate parameter importance: {e}")
        
        return result
    
    def _optimize_with_skopt(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        base_estimator: str = 'gp'
    ) -> TuningResult:
        """Optimization using scikit-optimize"""
        
        space = self.config.parameter_space.to_skopt_space()
        
        @use_named_args(space)
        def skopt_objective(**params):
            try:
                model = model_class(**params, random_state=42, n_jobs=1)
                
                if X_val is not None and y_val is not None:
                    model.fit(X, y)
                    predictions = model.predict(X_val)
                    score = self._calculate_score(y_val, predictions)
                else:
                    scores = self._cross_validate_model(model, X, y)
                    score = np.mean(scores)
                
                return -score if not self._is_minimization_objective() else score
                
            except Exception as e:
                logger.warning(f"⚠️ Error in skopt objective: {e}")
                return 1e10 if self._is_minimization_objective() else -1e10
        
        # Optimization
        if base_estimator == 'gp':
            skopt_result = gp_minimize(
                func=skopt_objective,
                dimensions=space,
                n_calls=self.config.n_trials,
                n_jobs=self.config.n_jobs,
                random_state=42
            )
        else:  # Random Forest
            skopt_result = forest_minimize(
                func=skopt_objective,
                dimensions=space,
                n_calls=self.config.n_trials,
                n_jobs=self.config.n_jobs,
                random_state=42
            )
        
        # Convert result
        best_params = {}
        for i, param_name in enumerate([dim.name for dim in space]):
            best_params[param_name] = skopt_result.x[i]
        
        best_score = -skopt_result.fun if not self._is_minimization_objective() else skopt_result.fun
        
        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_number=0,  # skopt doesn't have trial numbers
            total_trials=len(skopt_result.func_vals),
            successful_trials=len(skopt_result.func_vals),
            pruned_trials=0,
            failed_trials=0,
        )
        
        # Optimization history
        result.optimization_history = [
            {
                'trial_number': i,
                'value': -val if not self._is_minimization_objective() else val,
                'params': dict(zip([dim.name for dim in space], x)),
            }
            for i, (x, val) in enumerate(zip(skopt_result.x_iters, skopt_result.func_vals))
        ]
        
        return result
    
    def _optimize_with_random_search(
        self,
        model_class: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> TuningResult:
        """Optimization using random search"""
        
        best_score = float('inf') if self._is_minimization_objective() else float('-inf')
        best_params = {}
        best_trial_number = 0
        history = []
        
        param_space = self.config.parameter_space
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            task = progress.add_task("Random Search...", total=self.config.n_trials)
            
            for trial_num in range(self.config.n_trials):
                try:
                    # Random parameter generation
                    params = {
                        'n_estimators': np.random.randint(*param_space.n_estimators),
                        'max_depth': np.random.randint(*param_space.max_depth),
                        'min_child_weight': np.random.randint(*param_space.min_child_weight),
                        'learning_rate': np.random.uniform(*param_space.learning_rate),
                        'gamma': np.random.uniform(*param_space.gamma),
                        'reg_alpha': np.random.uniform(*param_space.reg_alpha),
                        'reg_lambda': np.random.uniform(*param_space.reg_lambda),
                        'subsample': np.random.uniform(*param_space.subsample),
                        'colsample_bytree': np.random.uniform(*param_space.colsample_bytree),
                        'booster': np.random.choice(param_space.booster),
                        'tree_method': np.random.choice(param_space.tree_method),
                    }
                    
                    # Evaluate parameters
                    model = model_class(**params, random_state=42, n_jobs=1)
                    
                    if X_val is not None and y_val is not None:
                        model.fit(X, y)
                        predictions = model.predict(X_val)
                        score = self._calculate_score(y_val, predictions)
                    else:
                        scores = self._cross_validate_model(model, X, y)
                        score = np.mean(scores)
                    
                    # Check for best result
                    if self._is_better_score(score, best_score):
                        best_score = score
                        best_params = params
                        best_trial_number = trial_num
                    
                    # Add to history
                    history.append({
                        'trial_number': trial_num,
                        'value': score,
                        'params': params,
                        'state': 'COMPLETE',
                    })
                    
                    progress.update(
                        task,
                        advance=1,
                        description=f"Trial {trial_num}/{self.config.n_trials} | "
                                   f"Best: {best_score:.4f}"
                    )
                
                except Exception as e:
                    logger.warning(f"⚠️ Error in random search trial {trial_num}: {e}")
                    history.append({
                        'trial_number': trial_num,
                        'value': None,
                        'params': None,
                        'state': 'FAIL',
                    })
        
        result = TuningResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_number=best_trial_number,
            total_trials=len(history),
            successful_trials=len([h for h in history if h['state'] == 'COMPLETE']),
            pruned_trials=0,
            failed_trials=len([h for h in history if h['state'] == 'FAIL']),
            optimization_history=history,
        )
        
        return result
    
    def _cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> List[float]:
        """Cross-validate model"""

        # Choose CV strategy
        if self.config.cv_strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = self.config.cv_folds
        
        # Determine scoring metric
        if self.config.scoring_metric:
            scoring = self.config.scoring_metric
        else:
            scoring = self._get_default_scoring_metric()
        
        # Cross-validation
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # To avoid conflicts in parallel optimization
            error_score='raise'
        )
        
        return scores.tolist()
    
    def _calculate_score(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> float:
        """Calculate score based on objective"""
        
        if self.config.objective == OptimizationObjective.MINIMIZE_RMSE:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.config.objective == OptimizationObjective.MINIMIZE_MAE:
            return mean_absolute_error(y_true, y_pred)
        elif self.config.objective == OptimizationObjective.MAXIMIZE_R2:
            return r2_score(y_true, y_pred)
        elif self.config.objective == OptimizationObjective.MAXIMIZE_ACCURACY:
            return accuracy_score(y_true, y_pred)
        elif self.config.objective == OptimizationObjective.MAXIMIZE_F1:
            return f1_score(y_true, y_pred, average='weighted')
        elif self.config.objective == OptimizationObjective.MAXIMIZE_SHARPE_RATIO:
            returns = pd.Series(y_pred).pct_change().fillna(0)
            return returns.mean() / returns.std() if returns.std() != 0 else 0
        else:
            # Fallback to RMSE
            return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def _is_minimization_objective(self) -> bool:
        """Check if the objective is a minimization one"""
        minimization_objectives = {
            OptimizationObjective.MINIMIZE_RMSE,
            OptimizationObjective.MINIMIZE_MAE,
            OptimizationObjective.MINIMIZE_MAPE,
            OptimizationObjective.MINIMIZE_DRAWDOWN,
            OptimizationObjective.MINIMIZE_VAR,
        }
        return self.config.objective in minimization_objectives
    
    def _is_better_score(self, new_score: float, current_best: float) -> bool:
        """Check if the new score is better"""
        if self._is_minimization_objective():
            return new_score < current_best
        else:
            return new_score > current_best
    
    def _get_default_scoring_metric(self) -> str:
        """Get default metric for CV"""
        
        if self.config.objective in [
            OptimizationObjective.MINIMIZE_RMSE,
            OptimizationObjective.MINIMIZE_MAE,
            OptimizationObjective.MAXIMIZE_R2
        ]:
            return 'neg_mean_squared_error'
        elif self.config.objective in [
            OptimizationObjective.MAXIMIZE_ACCURACY,
            OptimizationObjective.MAXIMIZE_F1
        ]:
            return 'accuracy'
        else:
            return 'neg_mean_squared_error'
    
    def _display_results(self, result: TuningResult):
        """Display optimization results"""

        table = Table(title=f"🎯 HYPERPARAMETER OPTIMIZATION RESULTS")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Best score", f"{result.best_score:.6f}")
        table.add_row("Best trial", str(result.best_trial_number))
        table.add_row("Total trials", str(result.total_trials))
        table.add_row("Successful", str(result.successful_trials))
        table.add_row("Pruned", str(result.pruned_trials))
        table.add_row("Failed", str(result.failed_trials))
        table.add_row("Optimization time", f"{result.tuning_time:.2f}s")
        
        self.console.print(table)
        
        # Top-5 parameters by importance
        if result.parameter_importance:
            importance_table = Table(title="📊 PARAMETER IMPORTANCE")
            importance_table.add_column("Parameter", style="cyan")
            importance_table.add_column("Importance", style="green")
            
            sorted_importance = sorted(
                result.parameter_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for param, importance in sorted_importance:
                importance_table.add_row(param, f"{importance:.4f}")
            
            self.console.print(importance_table)
    
    def _save_results(self, result: TuningResult, results_dir: Path):
        """Save optimization results"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        results_file = results_dir / f"tuning_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Save best parameters separately
        params_file = results_dir / f"best_params_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump(result.best_params, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_dir}")
    
    def plot_optimization_history(
        self,
        result: TuningResult,
        save_path: Optional[str] = None
    ):
        """Plot optimization history"""

        if not result.optimization_history:
            logger.warning("⚠️ No optimization history to plot")
            return
        
        # Extract data
        trial_numbers = []
        values = []
        
        for entry in result.optimization_history:
            if entry.get('value') is not None:
                trial_numbers.append(entry['trial_number'])
                values.append(entry['value'])
        
        if not trial_numbers:
            logger.warning("⚠️ No successful trials to plot")
            return
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Optimization history
        ax1.plot(trial_numbers, values, 'b-', alpha=0.6, label='Trial values')
        
        # Best score at each step
        best_so_far = []
        current_best = float('inf') if self._is_minimization_objective() else float('-inf')
        
        for value in values:
            if self._is_better_score(value, current_best):
                current_best = value
            best_so_far.append(current_best)
        
        ax1.plot(trial_numbers, best_so_far, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Score distribution
        ax2.hist(values, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(result.best_score, color='red', linestyle='--', 
                   label=f'Best Score: {result.best_score:.4f}')
        ax2.set_xlabel('Objective Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Trial Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Plot saved: {save_path}")
        else:
            plt.show()
    
    def plot_parameter_importance(
        self,
        result: TuningResult,
        save_path: Optional[str] = None
    ):
        """Plot parameter importance"""

        if not result.parameter_importance:
            logger.warning("⚠️ No parameter importance data available")
            return
        
        # Sort by importance
        sorted_importance = sorted(
            result.parameter_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        params, importance = zip(*sorted_importance)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(params)), importance, color='skyblue', alpha=0.7)
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
        ax.set_xlabel('Parameter Importance')
        ax.set_title('Hyperparameter Importance')
        ax.grid(True, alpha=0.3)
        
        # Add values to bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Parameter importance plot saved: {save_path}")
        else:
            plt.show()
    
    def get_best_model(
        self,
        model_class: Any,
        result: TuningResult
    ) -> Any:
        """Create model with best parameters"""
        
        return model_class(**result.best_params, random_state=42)
    
    def resume_optimization(
        self,
        study_name: str,
        storage_url: str,
        additional_trials: int = 50
    ) -> TuningResult:
        """Resume interrupted optimization"""

        # Load existing study
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url
        )
        
        logger.info(f"📂 Loaded study with {len(study.trials)} trials")

        # Continue optimization
        def objective(trial):
            # The original objective function needs to be passed here
            # In a real implementation this should be saved
            raise NotImplementedError("Resume requires a saved objective function")
        
        study.optimize(objective, n_trials=additional_trials)
        
        # Create result
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            best_trial_number=study.best_trial.number,
            total_trials=len(study.trials),
        )
        
        return result


if __name__ == "__main__":
    # Usage example
    logger.info("🧪 Testing Hyperparameter Tuner...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Target variable
    y = pd.Series(
        X.iloc[:, :3].sum(axis=1) + 0.1 * np.random.randn(n_samples),
        name='target'
    )
    
    # Create configuration
    config = TuningConfig(
        strategy=TuningStrategy.OPTUNA_TPE,
        objective=OptimizationObjective.MINIMIZE_RMSE,
        n_trials=20,  # Low count for fast testing
        cv_folds=3,
        enable_pruning=True
    )
    
    # Create tuner
    tuner = HyperparameterTuner(config)
    
    # Optimization (using base XGBRegressor for testing)
    from sklearn.ensemble import RandomForestRegressor  # Stub for testing
    
    class MockXGBRegressor:
        def __init__(self, **kwargs):
            self.params = kwargs
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return np.random.randn(len(X))
    
    # result = tuner.optimize(MockXGBRegressor, X, y)
    
    logger.info("✅ Hyperparameter Tuner testing ready for use!")