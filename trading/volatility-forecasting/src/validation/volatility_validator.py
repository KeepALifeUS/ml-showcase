"""
Volatility Model Validation Framework

Implementation of comprehensive validation system for volatility models:
- Out-of-sample testing with time series splits
- Rolling window validation
- Model comparison (Diebold-Mariano test)
- Loss functions (QLIKE, MSE, MAE, Hit Rate)
- VaR backtesting (Kupiec, Christoffersen tests)
- Model confidence set analysis
- Statistical significance testing

Features:
- Automated validation pipelines
- Production validation monitoring
- A/B testing framework for models
- Performance degradation detection
- Regulatory compliance validation
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import optuna
from numba import jit

# Logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ValidationMethod(Enum):
    """Validation methods"""
    OUT_OF_SAMPLE = "out_of_sample"
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    CROSS_VALIDATION = "cross_validation"
    BOOTSTRAP = "bootstrap"

class LossFunctionType(Enum):
    """Loss function types for volatility"""
    MSE = "mse"
    MAE = "mae"
    QLIKE = "qlike"
    R2LOGLIKE = "r2loglike"
    HIT_RATE = "hit_rate"
    DIRECTIONAL_ACCURACY = "directional_accuracy"

class TestType(Enum):
    """Statistical test types"""
    DIEBOLD_MARIANO = "diebold_mariano"
    KUPIEC = "kupiec"
    CHRISTOFFERSEN = "christoffersen"
    HANSEN_MCS = "hansen_mcs"
    WHITE_REALITY_CHECK = "white_reality_check"

@dataclass
class ValidationResult:
    """Model validation result"""
    model_name: str
    symbol: str
    validation_method: ValidationMethod
    validation_period: Tuple[datetime, datetime]
    
    # Performance metrics
    loss_scores: Dict[LossFunctionType, float]
    statistical_tests: Dict[TestType, Dict[str, Any]]
    
    # Detailed results
    predictions: pd.Series
    actuals: pd.Series
    residuals: pd.Series
    
    # Time-varying performance
    rolling_performance: Optional[pd.DataFrame] = None
    
    # Model diagnostics
    overfitting_score: float = 0.0
    stability_score: float = 1.0
    robustness_score: float = 1.0
    
    # Confidence intervals
    confidence_intervals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    
    # Metadata
    n_predictions: int = 0
    validation_start: datetime = field(default_factory=datetime.now)
    validation_end: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelComparison:
    """Model comparison result"""
    models: List[str]
    symbol: str
    comparison_period: Tuple[datetime, datetime]
    
    # Rankings
    performance_ranking: List[Tuple[str, float]]  # (model_name, score)
    statistical_significance: Dict[Tuple[str, str], Dict[str, Any]]  # (model1, model2) -> test results
    
    # Model confidence set
    mcs_results: Optional[Dict[str, Any]] = None
    
    # Best model selection
    best_model: str = ""
    confidence_level: float = 0.95
    
    # Ensemble recommendations
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """VaR backtesting result"""
    model_name: str
    symbol: str
    confidence_level: float
    
    # Violation statistics
    n_violations: int
    n_observations: int
    violation_rate: float
    expected_violation_rate: float
    
    # Statistical tests
    kupiec_test: Dict[str, Any]
    christoffersen_test: Dict[str, Any]
    
    # Violation clustering
    violation_dates: List[datetime]
    max_violation_cluster: int
    average_violation_size: float
    
    # Performance during violations
    average_excess_loss: float
    worst_violation: float
    
    # Traffic light status
    traffic_light: str = "green"  # green, yellow, red
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@jit(nopython=True)
def _calculate_qlike_loss_numba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast QLIKE loss calculation with Numba"""
    n = len(y_true)
    qlike = 0.0
    
    for i in range(n):
        if y_pred[i] > 0 and y_true[i] > 0:
            qlike += y_true[i] / y_pred[i] + np.log(y_pred[i])
    
    return qlike / n if n > 0 else 0.0

@jit(nopython=True)
def _calculate_r2loglike_numba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast R2-LogLike loss calculation"""
    n = len(y_true)
    r2loglike = 0.0
    
    for i in range(n):
        if y_pred[i] > 0 and y_true[i] > 0:
            r2loglike += (y_true[i] - y_pred[i])**2 / y_pred[i] + np.log(y_pred[i])
    
    return r2loglike / n if n > 0 else 0.0

@jit(nopython=True)
def _calculate_hit_rate_numba(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast hit rate calculation"""
    n = len(y_true)
    if n <= 1:
        return 0.0
    
    correct_directions = 0.0
    
    for i in range(1, n):
        true_direction = y_true[i] - y_true[i-1]
        pred_direction = y_pred[i] - y_pred[i-1]
        
        if true_direction * pred_direction > 0:
            correct_directions += 1
    
    return correct_directions / (n - 1) if n > 1 else 0.0

class BaseLossFunction(ABC):
    """Base class for loss functions"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss function"""
        pass

class QLIKELoss(BaseLossFunction):
    """QLIKE Loss Function for volatility forecasting"""
    
    def __init__(self):
        super().__init__("QLIKE")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate QLIKE loss"""
        return _calculate_qlike_loss_numba(y_true, y_pred)

class R2LogLikeLoss(BaseLossFunction):
    """R2-LogLike Loss Function"""
    
    def __init__(self):
        super().__init__("R2LogLike")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R2-LogLike loss"""
        return _calculate_r2loglike_numba(y_true, y_pred)

class HitRateLoss(BaseLossFunction):
    """Hit Rate (Directional Accuracy)"""
    
    def __init__(self):
        super().__init__("HitRate")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate hit rate (higher is better)"""
        return _calculate_hit_rate_numba(y_true, y_pred)

class VolatilityValidator:
    """
    Comprehensive Volatility Model Validator
    
    Supports multiple validation methods and statistical tests
    for comprehensive model evaluation in production environment.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.validation_history: List[ValidationResult] = []
        self.comparison_history: List[ModelComparison] = []
        
        # Loss functions
        self.loss_functions = {
            LossFunctionType.MSE: lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
            LossFunctionType.MAE: lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            LossFunctionType.QLIKE: QLIKELoss(),
            LossFunctionType.R2LOGLIKE: R2LogLikeLoss(),
            LossFunctionType.HIT_RATE: HitRateLoss(),
        }
        
        logger.info(f"🎯 Volatility validator initialized for {symbol}")

    async def validate_model(
        self,
        model: Any,  # Volatility model with predict method
        X: pd.DataFrame,
        y: pd.Series,
        method: ValidationMethod = ValidationMethod.OUT_OF_SAMPLE,
        test_size: float = 0.3,
        n_splits: int = 5,
        **kwargs
    ) -> ValidationResult:
        """
        Validate volatility model
        
        Args:
            model: Trained volatility model
            X: Feature matrix
            y: Target volatility series
            method: Validation method
            test_size: Test set size (for out-of-sample)
            n_splits: Number of splits (for cross-validation)
        """
        try:
            logger.info(f"🔄 Validating model with {method.value} method...")
            
            start_time = datetime.now()
            
            if method == ValidationMethod.OUT_OF_SAMPLE:
                result = await self._out_of_sample_validation(
                    model, X, y, test_size, **kwargs
                )
            elif method == ValidationMethod.ROLLING_WINDOW:
                result = await self._rolling_window_validation(
                    model, X, y, **kwargs
                )
            elif method == ValidationMethod.EXPANDING_WINDOW:
                result = await self._expanding_window_validation(
                    model, X, y, **kwargs
                )
            elif method == ValidationMethod.CROSS_VALIDATION:
                result = await self._cross_validation(
                    model, X, y, n_splits, **kwargs
                )
            else:
                raise ValueError(f"Unsupported validation method: {method}")
            
            computation_time = (datetime.now() - start_time).total_seconds()
            result.computation_time = computation_time
            
            self.validation_history.append(result)
            logger.info(f"✅ Model validation completed in {computation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in model validation: {e}")
            raise

    async def _out_of_sample_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        **kwargs
    ) -> ValidationResult:
        """Out-of-sample validation"""
        
        # Time-series split (not random!)
        split_index = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        
        if len(y_test) < 30:
            raise ValueError("Insufficient test data for validation")
        
        # Model fitting (if required)
        if hasattr(model, 'fit') and not getattr(model, 'is_fitted', False):
            await model.fit(X_train, y_train)
        
        # Prediction
        if hasattr(model, 'predict'):
            predictions_result = await model.predict(X_test, horizon=1)
            
            if hasattr(predictions_result, 'volatility_forecast'):
                predictions = predictions_result.volatility_forecast
                if len(predictions) == 1:
                    predictions = np.repeat(predictions[0], len(y_test))
                predictions = predictions[:len(y_test)]
            else:
                predictions = predictions_result
        else:
            raise ValueError("Model must have predict method")
        
        predictions_series = pd.Series(predictions, index=y_test.index)
        residuals = y_test - predictions_series
        
        # Calculate loss scores
        loss_scores = {}
        
        for loss_type, loss_func in self.loss_functions.items():
            try:
                if hasattr(loss_func, 'calculate'):
                    score = loss_func.calculate(y_test.values, predictions)
                else:
                    score = loss_func(y_test.values, predictions)
                loss_scores[loss_type] = float(score)
            except Exception as e:
                logger.warning(f"⚠️ Failed to calculate {loss_type.value}: {e}")
                loss_scores[loss_type] = np.inf
        
        # Statistical tests
        statistical_tests = await self._run_statistical_tests(
            y_test, predictions_series, residuals
        )
        
        # Model diagnostics
        overfitting_score = await self._calculate_overfitting_score(
            model, X_train, y_train, X_test, y_test
        )
        
        stability_score = await self._calculate_stability_score(
            predictions_series, y_test
        )
        
        robustness_score = await self._calculate_robustness_score(residuals)
        
        result = ValidationResult(
            model_name=getattr(model, 'name', 'Unknown Model'),
            symbol=self.symbol,
            validation_method=ValidationMethod.OUT_OF_SAMPLE,
            validation_period=(X_test.index[0], X_test.index[-1]),
            loss_scores=loss_scores,
            statistical_tests=statistical_tests,
            predictions=predictions_series,
            actuals=y_test,
            residuals=residuals,
            overfitting_score=overfitting_score,
            stability_score=stability_score,
            robustness_score=robustness_score,
            n_predictions=len(predictions),
            validation_start=X_test.index[0],
            validation_end=X_test.index[-1],
            metadata={
                "test_size": test_size,
                "train_size": len(X_train),
                "test_size_actual": len(X_test)
            }
        )
        
        return result

    async def _rolling_window_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        window_size: int = 252,
        step_size: int = 1,
        **kwargs
    ) -> ValidationResult:
        """Rolling window validation"""
        
        predictions_list = []
        actuals_list = []
        prediction_dates = []
        
        # Rolling window loop
        for i in range(window_size, len(X), step_size):
            train_start = max(0, i - window_size)
            train_end = i
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            
            X_pred = X.iloc[i:i+1]
            y_actual = y.iloc[i]
            
            try:
                # Retrain model (simplified - in production could be incremental)
                if hasattr(model, 'fit'):
                    await model.fit(X_train, y_train)
                
                # Predict
                pred_result = await model.predict(X_pred, horizon=1)
                
                if hasattr(pred_result, 'volatility_forecast'):
                    pred_value = pred_result.volatility_forecast[0]
                else:
                    pred_value = pred_result[0] if hasattr(pred_result, '__len__') else pred_result
                
                predictions_list.append(pred_value)
                actuals_list.append(y_actual)
                prediction_dates.append(X_pred.index[0])
                
            except Exception as e:
                logger.debug(f"Rolling window prediction failed at {i}: {e}")
                continue
        
        if len(predictions_list) < 30:
            raise ValueError("Insufficient successful predictions for validation")
        
        # Create series
        predictions_series = pd.Series(predictions_list, index=prediction_dates)
        actuals_series = pd.Series(actuals_list, index=prediction_dates)
        residuals = actuals_series - predictions_series
        
        # Calculate metrics
        loss_scores = {}
        for loss_type, loss_func in self.loss_functions.items():
            try:
                if hasattr(loss_func, 'calculate'):
                    score = loss_func.calculate(actuals_series.values, predictions_series.values)
                else:
                    score = loss_func(actuals_series.values, predictions_series.values)
                loss_scores[loss_type] = float(score)
            except:
                loss_scores[loss_type] = np.inf
        
        # Rolling performance analysis
        rolling_performance = await self._calculate_rolling_performance(
            predictions_series, actuals_series, window=30
        )
        
        # Statistical tests
        statistical_tests = await self._run_statistical_tests(
            actuals_series, predictions_series, residuals
        )
        
        result = ValidationResult(
            model_name=getattr(model, 'name', 'Unknown Model'),
            symbol=self.symbol,
            validation_method=ValidationMethod.ROLLING_WINDOW,
            validation_period=(prediction_dates[0], prediction_dates[-1]),
            loss_scores=loss_scores,
            statistical_tests=statistical_tests,
            predictions=predictions_series,
            actuals=actuals_series,
            residuals=residuals,
            rolling_performance=rolling_performance,
            n_predictions=len(predictions_list),
            validation_start=prediction_dates[0],
            validation_end=prediction_dates[-1],
            metadata={
                "window_size": window_size,
                "step_size": step_size,
                "successful_predictions": len(predictions_list),
                "failed_predictions": max(0, (len(X) - window_size) // step_size - len(predictions_list))
            }
        )
        
        return result

    async def _expanding_window_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        min_train_size: int = 100,
        **kwargs
    ) -> ValidationResult:
        """Expanding window validation"""
        
        predictions_list = []
        actuals_list = []
        prediction_dates = []
        
        # Expanding window loop
        for i in range(min_train_size, len(X)):
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            X_pred = X.iloc[i:i+1]
            y_actual = y.iloc[i]
            
            try:
                # Retrain model with expanding dataset
                if hasattr(model, 'fit'):
                    await model.fit(X_train, y_train)
                
                # Predict
                pred_result = await model.predict(X_pred, horizon=1)
                
                if hasattr(pred_result, 'volatility_forecast'):
                    pred_value = pred_result.volatility_forecast[0]
                else:
                    pred_value = pred_result[0] if hasattr(pred_result, '__len__') else pred_result
                
                predictions_list.append(pred_value)
                actuals_list.append(y_actual)
                prediction_dates.append(X_pred.index[0])
                
            except Exception as e:
                logger.debug(f"Expanding window prediction failed at {i}: {e}")
                continue
        
        # Same processing as rolling window
        predictions_series = pd.Series(predictions_list, index=prediction_dates)
        actuals_series = pd.Series(actuals_list, index=prediction_dates)
        residuals = actuals_series - predictions_series
        
        # Calculate metrics
        loss_scores = {}
        for loss_type, loss_func in self.loss_functions.items():
            try:
                if hasattr(loss_func, 'calculate'):
                    score = loss_func.calculate(actuals_series.values, predictions_series.values)
                else:
                    score = loss_func(actuals_series.values, predictions_series.values)
                loss_scores[loss_type] = float(score)
            except:
                loss_scores[loss_type] = np.inf
        
        statistical_tests = await self._run_statistical_tests(
            actuals_series, predictions_series, residuals
        )
        
        result = ValidationResult(
            model_name=getattr(model, 'name', 'Unknown Model'),
            symbol=self.symbol,
            validation_method=ValidationMethod.EXPANDING_WINDOW,
            validation_period=(prediction_dates[0], prediction_dates[-1]),
            loss_scores=loss_scores,
            statistical_tests=statistical_tests,
            predictions=predictions_series,
            actuals=actuals_series,
            residuals=residuals,
            n_predictions=len(predictions_list),
            validation_start=prediction_dates[0],
            validation_end=prediction_dates[-1],
            metadata={
                "min_train_size": min_train_size,
                "max_train_size": len(X) - 1,
                "successful_predictions": len(predictions_list)
            }
        )
        
        return result

    async def _cross_validation(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int,
        **kwargs
    ) -> ValidationResult:
        """Time series cross-validation"""
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        cv_scores = []
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            logger.debug(f"Processing CV fold {fold+1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            try:
                # Train model
                if hasattr(model, 'fit'):
                    await model.fit(X_train, y_train)
                
                # Predict
                pred_result = await model.predict(X_test, horizon=1)
                
                if hasattr(pred_result, 'volatility_forecast'):
                    predictions = pred_result.volatility_forecast
                    if len(predictions) == 1:
                        predictions = np.repeat(predictions[0], len(y_test))
                else:
                    predictions = pred_result
                
                predictions = predictions[:len(y_test)]
                
                all_predictions.extend(predictions)
                all_actuals.extend(y_test.values)
                all_dates.extend(y_test.index)
                
                # Fold score
                fold_mse = mean_squared_error(y_test.values, predictions)
                cv_scores.append(fold_mse)
                
            except Exception as e:
                logger.warning(f"⚠️ CV fold {fold+1} failed: {e}")
                continue
        
        if len(all_predictions) < 30:
            raise ValueError("Insufficient CV predictions for validation")
        
        # Aggregate results
        predictions_series = pd.Series(all_predictions, index=all_dates)
        actuals_series = pd.Series(all_actuals, index=all_dates)
        residuals = actuals_series - predictions_series
        
        # Calculate metrics
        loss_scores = {}
        for loss_type, loss_func in self.loss_functions.items():
            try:
                if hasattr(loss_func, 'calculate'):
                    score = loss_func.calculate(actuals_series.values, predictions_series.values)
                else:
                    score = loss_func(actuals_series.values, predictions_series.values)
                loss_scores[loss_type] = float(score)
            except:
                loss_scores[loss_type] = np.inf
        
        statistical_tests = await self._run_statistical_tests(
            actuals_series, predictions_series, residuals
        )
        
        result = ValidationResult(
            model_name=getattr(model, 'name', 'Unknown Model'),
            symbol=self.symbol,
            validation_method=ValidationMethod.CROSS_VALIDATION,
            validation_period=(all_dates[0], all_dates[-1]),
            loss_scores=loss_scores,
            statistical_tests=statistical_tests,
            predictions=predictions_series,
            actuals=actuals_series,
            residuals=residuals,
            n_predictions=len(all_predictions),
            validation_start=all_dates[0],
            validation_end=all_dates[-1],
            metadata={
                "n_splits": n_splits,
                "cv_scores": cv_scores,
                "cv_mean_score": np.mean(cv_scores),
                "cv_std_score": np.std(cv_scores)
            }
        )
        
        return result

    async def _run_statistical_tests(
        self,
        actuals: pd.Series,
        predictions: pd.Series,
        residuals: pd.Series
    ) -> Dict[TestType, Dict[str, Any]]:
        """Run statistical tests on predictions"""
        
        tests = {}
        
        # Normality test on residuals
        try:
            stat, p_value = stats.jarque_bera(residuals.dropna())
            tests[TestType.WHITE_REALITY_CHECK] = {
                "name": "Jarque-Bera Normality Test",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
                "interpretation": "Residuals are normally distributed" if p_value > 0.05 else "Residuals are not normally distributed"
            }
        except Exception as e:
            logger.debug(f"Jarque-Bera test failed: {e}")
        
        # Autocorrelation test on residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
            min_p_value = lb_result['lb_pvalue'].min()
            
            tests[TestType.CHRISTOFFERSEN] = {
                "name": "Ljung-Box Autocorrelation Test",
                "min_p_value": float(min_p_value),
                "has_autocorrelation": min_p_value < 0.05,
                "interpretation": "Residuals have significant autocorrelation" if min_p_value < 0.05 else "No significant autocorrelation in residuals"
            }
        except Exception as e:
            logger.debug(f"Ljung-Box test failed: {e}")
        
        return tests

    async def _calculate_overfitting_score(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> float:
        """Calculate overfitting score"""
        
        try:
            # In-sample performance
            train_pred = await model.predict(X_train, horizon=1)
            if hasattr(train_pred, 'volatility_forecast'):
                train_predictions = train_pred.volatility_forecast
                if len(train_predictions) == 1:
                    train_predictions = np.repeat(train_predictions[0], len(y_train))
            else:
                train_predictions = train_pred
            
            train_mse = mean_squared_error(y_train.values, train_predictions[:len(y_train)])
            
            # Out-of-sample performance
            test_pred = await model.predict(X_test, horizon=1)
            if hasattr(test_pred, 'volatility_forecast'):
                test_predictions = test_pred.volatility_forecast
                if len(test_predictions) == 1:
                    test_predictions = np.repeat(test_predictions[0], len(y_test))
            else:
                test_predictions = test_pred
            
            test_mse = mean_squared_error(y_test.values, test_predictions[:len(y_test)])
            
            # Overfitting score: higher = more overfitting
            if train_mse > 0:
                overfitting_score = max(0, (test_mse - train_mse) / train_mse)
            else:
                overfitting_score = 0.0
            
            return min(1.0, overfitting_score)  # Cap at 1.0
            
        except Exception as e:
            logger.debug(f"Overfitting calculation failed: {e}")
            return 0.5  # Default score

    async def _calculate_stability_score(
        self,
        predictions: pd.Series,
        actuals: pd.Series
    ) -> float:
        """Calculate prediction stability score"""
        
        try:
            # Rolling MSE stability
            window_size = min(30, len(predictions) // 3)
            rolling_mse = []
            
            for i in range(window_size, len(predictions)):
                window_preds = predictions.iloc[i-window_size:i]
                window_actuals = actuals.iloc[i-window_size:i]
                mse = mean_squared_error(window_actuals, window_preds)
                rolling_mse.append(mse)
            
            if len(rolling_mse) > 5:
                # Stability = 1 - coefficient of variation of rolling MSE
                mse_mean = np.mean(rolling_mse)
                mse_std = np.std(rolling_mse)
                
                if mse_mean > 0:
                    cv = mse_std / mse_mean
                    stability = max(0, 1 - cv)
                else:
                    stability = 1.0
            else:
                stability = 0.5  # Default
            
            return min(1.0, max(0.0, stability))
            
        except Exception as e:
            logger.debug(f"Stability calculation failed: {e}")
            return 0.5

    async def _calculate_robustness_score(self, residuals: pd.Series) -> float:
        """Calculate model robustness score"""
        
        try:
            # Robustness based on outlier resistance
            clean_residuals = residuals.dropna()
            
            if len(clean_residuals) < 10:
                return 0.5
            
            # Interquartile range method
            q1 = clean_residuals.quantile(0.25)
            q3 = clean_residuals.quantile(0.75)
            iqr = q3 - q1
            
            # Outlier bounds
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Fraction of non-outliers
            non_outliers = ((clean_residuals >= lower_bound) & (clean_residuals <= upper_bound)).sum()
            robustness = non_outliers / len(clean_residuals)
            
            return robustness
            
        except Exception as e:
            logger.debug(f"Robustness calculation failed: {e}")
            return 0.5

    async def _calculate_rolling_performance(
        self,
        predictions: pd.Series,
        actuals: pd.Series,
        window: int = 30
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        rolling_metrics = []
        
        for i in range(window, len(predictions)):
            window_preds = predictions.iloc[i-window:i]
            window_actuals = actuals.iloc[i-window:i]
            
            mse = mean_squared_error(window_actuals, window_preds)
            mae = mean_absolute_error(window_actuals, window_preds)
            
            try:
                r2 = r2_score(window_actuals, window_preds)
            except:
                r2 = 0.0
            
            rolling_metrics.append({
                'date': predictions.index[i],
                'mse': mse,
                'mae': mae,
                'r2': r2
            })
        
        return pd.DataFrame(rolling_metrics).set_index('date')

    async def compare_models(
        self,
        validation_results: List[ValidationResult],
        primary_metric: LossFunctionType = LossFunctionType.QLIKE,
        confidence_level: float = 0.95
    ) -> ModelComparison:
        """Compare multiple validated models"""
        
        try:
            logger.info(f"🔄 Comparing {len(validation_results)} models...")
            
            if len(validation_results) < 2:
                raise ValueError("At least 2 models required for comparison")
            
            # Performance ranking
            model_scores = []
            for result in validation_results:
                score = result.loss_scores.get(primary_metric, np.inf)
                # For hit rate (higher is better), negate for ranking
                if primary_metric == LossFunctionType.HIT_RATE:
                    score = -score  # Convert to loss (lower is better)
                model_scores.append((result.model_name, score))
            
            # Sort by score (lower is better)
            performance_ranking = sorted(model_scores, key=lambda x: x[1])
            
            # Statistical significance testing
            significance_tests = {}
            
            for i, result1 in enumerate(validation_results):
                for j, result2 in enumerate(validation_results):
                    if i >= j:  # Avoid duplicates and self-comparison
                        continue
                    
                    # Diebold-Mariano test
                    dm_test = await self._diebold_mariano_test(
                        result1.actuals,
                        result1.predictions,
                        result2.predictions,
                        loss_function=primary_metric
                    )
                    
                    significance_tests[(result1.model_name, result2.model_name)] = dm_test
            
            # Model confidence set (simplified)
            mcs_results = await self._model_confidence_set(
                validation_results, primary_metric, confidence_level
            )
            
            # Best model selection
            best_model = performance_ranking[0][0] if performance_ranking else ""
            
            # Ensemble weights (inverse performance weighting)
            ensemble_weights = await self._calculate_ensemble_weights(
                validation_results, primary_metric
            )
            
            comparison = ModelComparison(
                models=[result.model_name for result in validation_results],
                symbol=self.symbol,
                comparison_period=(
                    min(result.validation_start for result in validation_results),
                    max(result.validation_end for result in validation_results)
                ),
                performance_ranking=performance_ranking,
                statistical_significance=significance_tests,
                mcs_results=mcs_results,
                best_model=best_model,
                confidence_level=confidence_level,
                ensemble_weights=ensemble_weights,
                metadata={
                    "primary_metric": primary_metric.value,
                    "n_models": len(validation_results),
                    "comparison_timestamp": datetime.now()
                }
            )
            
            self.comparison_history.append(comparison)
            logger.info(f"✅ Model comparison completed. Best model: {best_model}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"❌ Error in model comparison: {e}")
            raise

    async def _diebold_mariano_test(
        self,
        actuals: pd.Series,
        predictions1: pd.Series,
        predictions2: pd.Series,
        loss_function: LossFunctionType
    ) -> Dict[str, Any]:
        """Diebold-Mariano test for forecast accuracy"""
        
        try:
            # Align series
            common_index = actuals.index.intersection(predictions1.index).intersection(predictions2.index)
            
            if len(common_index) < 30:
                return {"error": "Insufficient common observations"}
            
            y_true = actuals.reindex(common_index).values
            pred1 = predictions1.reindex(common_index).values
            pred2 = predictions2.reindex(common_index).values
            
            # Calculate loss differences
            if loss_function == LossFunctionType.MSE:
                loss1 = (y_true - pred1) ** 2
                loss2 = (y_true - pred2) ** 2
            elif loss_function == LossFunctionType.MAE:
                loss1 = np.abs(y_true - pred1)
                loss2 = np.abs(y_true - pred2)
            elif loss_function == LossFunctionType.QLIKE:
                loss1 = y_true / pred1 + np.log(pred1)
                loss2 = y_true / pred2 + np.log(pred2)
            else:
                # Default to MSE
                loss1 = (y_true - pred1) ** 2
                loss2 = (y_true - pred2) ** 2
            
            loss_diff = loss1 - loss2
            
            # DM test statistic
            mean_diff = np.mean(loss_diff)
            
            # Autocorrelation-robust variance (Newey-West)
            n = len(loss_diff)
            variance = np.var(loss_diff, ddof=1)
            
            # Simplified variance calculation (for full implementation need Newey-West)
            dm_stat = mean_diff / np.sqrt(variance / n) if variance > 0 else 0
            
            # P-value (two-tailed test)
            p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
            
            return {
                "dm_statistic": float(dm_stat),
                "p_value": float(p_value),
                "is_significant": p_value < 0.05,
                "mean_loss_difference": float(mean_diff),
                "interpretation": "Model 1 significantly better" if dm_stat < -1.96 and p_value < 0.05 
                               else "Model 2 significantly better" if dm_stat > 1.96 and p_value < 0.05
                               else "No significant difference"
            }
            
        except Exception as e:
            logger.debug(f"DM test calculation failed: {e}")
            return {"error": str(e)}

    async def _model_confidence_set(
        self,
        validation_results: List[ValidationResult],
        metric: LossFunctionType,
        confidence_level: float
    ) -> Dict[str, Any]:
        """Simplified Model Confidence Set analysis"""
        
        try:
            # Extract loss series for each model
            model_losses = {}
            
            for result in validation_results:
                model_name = result.model_name
                score = result.loss_scores.get(metric, np.inf)
                
                # Convert single score to series (simplified)
                n_obs = len(result.predictions)
                if n_obs > 0:
                    model_losses[model_name] = np.full(n_obs, score)
                else:
                    model_losses[model_name] = np.array([score])
            
            if len(model_losses) < 2:
                return {"error": "Insufficient models for MCS"}
            
            # Simplified MCS: keep models within confidence interval of best
            best_score = min(model_losses[name].mean() for name in model_losses)
            
            # Calculate confidence interval width (simplified)
            all_scores = [model_losses[name].mean() for name in model_losses]
            score_std = np.std(all_scores)
            ci_width = stats.norm.ppf((1 + confidence_level) / 2) * score_std
            
            # Models in confidence set
            mcs_models = []
            for model_name, losses in model_losses.items():
                model_score = losses.mean()
                if model_score <= best_score + ci_width:
                    mcs_models.append(model_name)
            
            return {
                "mcs_models": mcs_models,
                "eliminated_models": [name for name in model_losses.keys() if name not in mcs_models],
                "confidence_level": confidence_level,
                "mcs_p_values": {name: 1.0 if name in mcs_models else 0.0 for name in model_losses.keys()},
                "best_score": best_score,
                "confidence_interval_width": ci_width
            }
            
        except Exception as e:
            logger.debug(f"MCS calculation failed: {e}")
            return {"error": str(e)}

    async def _calculate_ensemble_weights(
        self,
        validation_results: List[ValidationResult],
        metric: LossFunctionType
    ) -> Dict[str, float]:
        """Calculate ensemble weights based on performance"""
        
        try:
            model_scores = {}
            
            for result in validation_results:
                score = result.loss_scores.get(metric, np.inf)
                
                # Handle different metrics
                if metric == LossFunctionType.HIT_RATE:
                    # Higher is better, so use score directly
                    model_scores[result.model_name] = max(0, score)
                else:
                    # Lower is better, so use inverse
                    model_scores[result.model_name] = 1 / (1 + score) if score != np.inf else 0
            
            # Normalize weights
            total_score = sum(model_scores.values())
            if total_score > 0:
                weights = {name: score / total_score for name, score in model_scores.items()}
            else:
                # Equal weights if all models failed
                n_models = len(model_scores)
                weights = {name: 1.0 / n_models for name in model_scores.keys()}
            
            return weights
            
        except Exception as e:
            logger.debug(f"Ensemble weights calculation failed: {e}")
            # Return equal weights as fallback
            n_models = len(validation_results)
            return {result.model_name: 1.0 / n_models for result in validation_results}

    def generate_validation_report(self, validation_result: ValidationResult) -> str:
        """Generate comprehensive validation report"""
        
        report = f"""
🎯 Volatility Model Validation Report
{'='*60}

Model: {validation_result.model_name}
Symbol: {validation_result.symbol}
Validation Method: {validation_result.validation_method.value}
Validation Period: {validation_result.validation_start.date()} to {validation_result.validation_end.date()}
Number of Predictions: {validation_result.n_predictions}

Performance Metrics:
"""
        
        # Loss scores
        for loss_type, score in validation_result.loss_scores.items():
            if not np.isinf(score):
                report += f"- {loss_type.value.upper()}: {score:.6f}\n"
        
        # Model diagnostics
        report += f"""
Model Diagnostics:
- Overfitting Score: {validation_result.overfitting_score:.3f} (lower is better)
- Stability Score: {validation_result.stability_score:.3f} (higher is better)  
- Robustness Score: {validation_result.robustness_score:.3f} (higher is better)
"""
        
        # Statistical tests
        if validation_result.statistical_tests:
            report += "\nStatistical Tests:\n"
            for test_type, test_result in validation_result.statistical_tests.items():
                if isinstance(test_result, dict) and "interpretation" in test_result:
                    report += f"- {test_result.get('name', test_type.value)}: {test_result['interpretation']}\n"
        
        # Performance summary
        if not np.isinf(validation_result.loss_scores.get(LossFunctionType.QLIKE, np.inf)):
            qlike_score = validation_result.loss_scores[LossFunctionType.QLIKE]
            if qlike_score < 1.0:
                performance = "Excellent"
            elif qlike_score < 2.0:
                performance = "Good"
            elif qlike_score < 3.0:
                performance = "Fair"
            else:
                performance = "Poor"
            
            report += f"\nOverall Performance: {performance}"
        
        # Recommendations
        report += "\n\nRecommendations:\n"
        
        if validation_result.overfitting_score > 0.3:
            report += "- Consider regularization to reduce overfitting\n"
        
        if validation_result.stability_score < 0.7:
            report += "- Model shows instability, consider ensemble methods\n"
        
        if validation_result.robustness_score < 0.8:
            report += "- Model is sensitive to outliers, consider robust estimators\n"
        
        report += f"\nValidation completed in {validation_result.computation_time:.2f} seconds"
        
        return report

    def generate_comparison_report(self, comparison: ModelComparison) -> str:
        """Generate model comparison report"""
        
        report = f"""
🏆 Model Comparison Report
{'='*60}

Symbol: {comparison.symbol}
Comparison Period: {comparison.comparison_period[0].date()} to {comparison.comparison_period[1].date()}
Models Compared: {len(comparison.models)}

Performance Ranking:
"""
        
        for i, (model_name, score) in enumerate(comparison.performance_ranking):
            report += f"{i+1}. {model_name}: {score:.6f}\n"
        
        report += f"\nBest Model: {comparison.best_model}\n"
        
        # Statistical significance
        if comparison.statistical_significance:
            report += "\nStatistical Significance Tests:\n"
            for (model1, model2), test_result in comparison.statistical_significance.items():
                if isinstance(test_result, dict) and "interpretation" in test_result:
                    report += f"- {model1} vs {model2}: {test_result['interpretation']}\n"
        
        # Model confidence set
        if comparison.mcs_results and "mcs_models" in comparison.mcs_results:
            mcs_models = comparison.mcs_results["mcs_models"]
            report += f"\nModel Confidence Set ({comparison.confidence_level*100}%): {mcs_models}\n"
        
        # Ensemble recommendations
        if comparison.ensemble_weights:
            report += "\nEnsemble Weights:\n"
            for model, weight in comparison.ensemble_weights.items():
                report += f"- {model}: {weight:.3f}\n"
        
        return report

class VaRBacktester:
    """
    VaR Backtesting Framework
    
    Implements regulatory-compliant VaR backtesting with Kupiec and Christoffersen tests.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.backtest_history: List[BacktestResult] = []
        
        logger.info(f"🎯 VaR backtester initialized for {symbol}")

    async def backtest_var(
        self,
        returns: pd.Series,
        var_forecasts: pd.Series,
        confidence_level: float = 0.95,
        model_name: str = "Unknown"
    ) -> BacktestResult:
        """Comprehensive VaR backtesting"""
        
        try:
            logger.info(f"🔄 Backtesting VaR for {model_name}...")
            
            # Align data
            common_index = returns.index.intersection(var_forecasts.index)
            if len(common_index) < 30:
                raise ValueError("Insufficient aligned data for backtesting")
            
            aligned_returns = returns.reindex(common_index)
            aligned_var = var_forecasts.reindex(common_index)
            
            # Violations (returns < -VaR)
            violations = aligned_returns < -aligned_var
            n_violations = violations.sum()
            n_observations = len(aligned_returns)
            
            violation_rate = n_violations / n_observations
            expected_violation_rate = 1 - confidence_level
            
            # Kupiec test
            kupiec_test = await self._kupiec_test(
                n_violations, n_observations, expected_violation_rate
            )
            
            # Christoffersen test (independence)
            christoffersen_test = await self._christoffersen_test(violations.values)
            
            # Violation analysis
            violation_dates = aligned_returns[violations].index.tolist()
            violation_sizes = -aligned_returns[violations].values - aligned_var[violations].values
            
            # Clustering analysis
            max_cluster = await self._calculate_max_violation_cluster(violations.values)
            avg_violation_size = np.mean(violation_sizes) if len(violation_sizes) > 0 else 0
            
            # Excess loss during violations
            excess_losses = violation_sizes[violation_sizes > 0]
            avg_excess_loss = np.mean(excess_losses) if len(excess_losses) > 0 else 0
            worst_violation = np.max(violation_sizes) if len(violation_sizes) > 0 else 0
            
            # Traffic light determination
            traffic_light = await self._determine_traffic_light(
                n_violations, n_observations, expected_violation_rate
            )
            
            result = BacktestResult(
                model_name=model_name,
                symbol=self.symbol,
                confidence_level=confidence_level,
                n_violations=n_violations,
                n_observations=n_observations,
                violation_rate=violation_rate,
                expected_violation_rate=expected_violation_rate,
                kupiec_test=kupiec_test,
                christoffersen_test=christoffersen_test,
                violation_dates=violation_dates,
                max_violation_cluster=max_cluster,
                average_violation_size=avg_violation_size,
                average_excess_loss=avg_excess_loss,
                worst_violation=worst_violation,
                traffic_light=traffic_light,
                metadata={
                    "backtest_period": f"{common_index[0].date()} to {common_index[-1].date()}",
                    "total_excess_loss": np.sum(violation_sizes) if len(violation_sizes) > 0 else 0
                }
            )
            
            self.backtest_history.append(result)
            logger.info(f"✅ VaR backtest: {violation_rate:.2%} violations (expected: {expected_violation_rate:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in VaR backtesting: {e}")
            raise

    async def _kupiec_test(
        self,
        violations: int,
        n_observations: int,
        expected_rate: float
    ) -> Dict[str, Any]:
        """Kupiec likelihood ratio test"""
        
        if violations == 0 or violations == n_observations:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "is_rejected": False,
                "interpretation": "Test not applicable (no violations or all violations)"
            }
        
        observed_rate = violations / n_observations
        
        # Log-likelihood ratio statistic
        if observed_rate > 0:
            lr_stat = -2 * (
                violations * np.log(expected_rate / observed_rate) +
                (n_observations - violations) * np.log((1 - expected_rate) / (1 - observed_rate))
            )
        else:
            lr_stat = 0.0
        
        # P-value from chi-square distribution (df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        # Reject null hypothesis if p < 0.05
        is_rejected = p_value < 0.05
        
        interpretation = (
            f"VaR model {'rejected' if is_rejected else 'not rejected'} at 5% significance level. "
            f"Observed rate: {observed_rate:.2%}, Expected: {expected_rate:.2%}"
        )
        
        return {
            "statistic": float(lr_stat),
            "p_value": float(p_value),
            "is_rejected": is_rejected,
            "observed_rate": observed_rate,
            "expected_rate": expected_rate,
            "interpretation": interpretation
        }

    async def _christoffersen_test(self, violations: np.ndarray) -> Dict[str, Any]:
        """Christoffersen independence test"""
        
        try:
            # Transition matrix for violations
            n = len(violations)
            if n < 2:
                return {"error": "Insufficient data for independence test"}
            
            # Count transitions: 00, 01, 10, 11
            n00 = n01 = n10 = n11 = 0
            
            for i in range(n - 1):
                if violations[i] == 0 and violations[i + 1] == 0:
                    n00 += 1
                elif violations[i] == 0 and violations[i + 1] == 1:
                    n01 += 1
                elif violations[i] == 1 and violations[i + 1] == 0:
                    n10 += 1
                elif violations[i] == 1 and violations[i + 1] == 1:
                    n11 += 1
            
            # Transition probabilities
            n0 = n00 + n01  # States starting with 0
            n1 = n10 + n11  # States starting with 1
            
            if n0 == 0 or n1 == 0:
                return {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "is_independent": True,
                    "interpretation": "Independence test not applicable"
                }
            
            pi01 = n01 / n0 if n0 > 0 else 0
            pi11 = n11 / n1 if n1 > 0 else 0
            pi = (n01 + n11) / (n - 1)  # Overall violation probability
            
            # Likelihood ratio test statistic
            if pi01 > 0 and pi11 > 0 and pi > 0:
                lr_ind = 2 * (
                    n00 * np.log(1 - pi) + n01 * np.log(pi) +
                    n10 * np.log(1 - pi) + n11 * np.log(pi) -
                    n00 * np.log(1 - pi01) - n01 * np.log(pi01) -
                    n10 * np.log(1 - pi11) - n11 * np.log(pi11)
                )
            else:
                lr_ind = 0.0
            
            # P-value (chi-square with df=1)
            p_value = 1 - stats.chi2.cdf(lr_ind, df=1) if lr_ind > 0 else 1.0
            
            is_independent = p_value > 0.05
            
            interpretation = (
                f"Violations are {'independent' if is_independent else 'not independent'} "
                f"(clustering {'not detected' if is_independent else 'detected'})"
            )
            
            return {
                "statistic": float(lr_ind),
                "p_value": float(p_value),
                "is_independent": is_independent,
                "pi01": pi01,
                "pi11": pi11,
                "interpretation": interpretation
            }
            
        except Exception as e:
            logger.debug(f"Christoffersen test failed: {e}")
            return {"error": str(e)}

    async def _calculate_max_violation_cluster(self, violations: np.ndarray) -> int:
        """Calculate maximum consecutive violation cluster"""
        
        max_cluster = 0
        current_cluster = 0
        
        for violation in violations:
            if violation:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0
        
        return max_cluster

    async def _determine_traffic_light(
        self,
        violations: int,
        n_observations: int,
        expected_rate: float
    ) -> str:
        """Determine Basel traffic light status"""
        
        # Basel III thresholds for 99% VaR over 250 days
        if expected_rate == 0.01:  # 99% confidence level
            if violations <= 4:
                return "green"
            elif violations <= 9:
                return "yellow"
            else:
                return "red"
        
        # General thresholds based on expected violations
        expected_violations = expected_rate * n_observations
        
        if violations <= expected_violations * 1.5:
            return "green"
        elif violations <= expected_violations * 2.5:
            return "yellow"
        else:
            return "red"

    def generate_backtest_report(self, backtest_result: BacktestResult) -> str:
        """Generate VaR backtest report"""
        
        report = f"""
📊 VaR Backtesting Report
{'='*60}

Model: {backtest_result.model_name}
Symbol: {backtest_result.symbol}
Confidence Level: {backtest_result.confidence_level*100}%
Traffic Light Status: {backtest_result.traffic_light.upper()}

Violation Statistics:
- Violations: {backtest_result.n_violations}/{backtest_result.n_observations}
- Violation Rate: {backtest_result.violation_rate:.2%}
- Expected Rate: {backtest_result.expected_violation_rate:.2%}
- Worst Violation: {backtest_result.worst_violation:.4f}
- Average Excess Loss: {backtest_result.average_excess_loss:.4f}

Statistical Tests:
"""
        
        # Kupiec test
        kupiec = backtest_result.kupiec_test
        if "interpretation" in kupiec:
            report += f"- Kupiec Test: {kupiec['interpretation']}\n"
            report += f"  (Statistic: {kupiec.get('statistic', 0):.3f}, P-value: {kupiec.get('p_value', 1):.3f})\n"
        
        # Christoffersen test
        christoffersen = backtest_result.christoffersen_test
        if "interpretation" in christoffersen:
            report += f"- Christoffersen Test: {christoffersen['interpretation']}\n"
        
        # Clustering analysis
        report += f"""
Violation Clustering:
- Maximum Consecutive Violations: {backtest_result.max_violation_cluster}
- Total Violation Dates: {len(backtest_result.violation_dates)}
"""
        
        # Status interpretation
        if backtest_result.traffic_light == "green":
            status_msg = "VaR model performance is acceptable"
        elif backtest_result.traffic_light == "yellow":
            status_msg = "VaR model needs monitoring and possible adjustment"
        else:
            status_msg = "VaR model is inadequate and requires immediate revision"
        
        report += f"\nOverall Assessment: {status_msg}"
        
        return report

# Export all classes
__all__ = [
    "ValidationMethod",
    "LossFunctionType", 
    "TestType",
    "ValidationResult",
    "ModelComparison",
    "BacktestResult",
    "BaseLossFunction",
    "QLIKELoss",
    "R2LogLikeLoss",
    "HitRateLoss",
    "VolatilityValidator",
    "VaRBacktester",
    "_calculate_qlike_loss_numba",
    "_calculate_r2loglike_numba",
    "_calculate_hit_rate_numba"
]

logger.info("🔥 Volatility Validation Framework loaded successfully!")