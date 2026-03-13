"""
Hyperparameter Tuning Module for XGBoost Models
===============================================

Advanced hyperparameter optimization specifically designed for XGBoost models
and time series prediction with enterprise patterns.

Key Components:
--------------
- HyperparameterTuner: Main tuning orchestrator
- OptunaTuner: Optuna-based Bayesian optimization
- BayesianOptimizer: Custom Bayesian optimization
- GridSearchTuner: Exhaustive grid search
- RandomSearchTuner: Random search optimization

Optimization Strategies:
-----------------------
- Bayesian optimization with Tree-structured Parzen Estimator
- Multi-objective optimization (accuracy vs inference time)
- Early stopping and pruning for efficiency
- Cross-validation integration
- Distributed tuning support
"""

from .hyperparameter_tuner import (
    HyperparameterTuner,
    TuningConfig,
    TuningResult,
    OptimizationObjective,
)

from .optuna_tuner import (
    OptunaTuner,
    OptunaConfig,
    OptunaStudyManager,
)

from .bayesian_optimizer import (
    BayesianOptimizer,
    BayesianConfig,
    AcquisitionFunction,
)

from .grid_search_tuner import (
    GridSearchTuner,
    GridSearchConfig,
)

__all__ = [
    # Main tuner
    "HyperparameterTuner",
    "TuningConfig",
    "TuningResult", 
    "OptimizationObjective",
    
    # Optuna tuner
    "OptunaTuner",
    "OptunaConfig",
    "OptunaStudyManager",
    
    # Bayesian optimizer
    "BayesianOptimizer", 
    "BayesianConfig",
    "AcquisitionFunction",
    
    # Grid search
    "GridSearchTuner",
    "GridSearchConfig",
]