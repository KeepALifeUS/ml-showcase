"""
Regression Tester - Enterprise Enterprise ML Regression Testing Framework
 testing on regression for ML models crypto trading

Applies enterprise principles:
- Continuous model validation
- Automated regression detection
- Performance baseline tracking
- Production model monitoring
"""

import os
import pickle
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
import logging
from contextlib import contextmanager
import hashlib

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegressionType(Enum):
    """Types regression testing for Enterprise ML governance"""
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    PREDICTION_DRIFT = "prediction_drift"
    MODEL_BEHAVIOR = "model_behavior"
    FEATURE_IMPORTANCE = "feature_importance"
    LATENCY = "latency"
    MEMORY_USAGE = "memory_usage"


class ComparisonMethod(Enum):
    """Method comparison for Enterprise statistical testing"""
    ABSOLUTE_THRESHOLD = "absolute_threshold"
    RELATIVE_THRESHOLD = "relative_threshold"
    STATISTICAL_TEST = "statistical_test"
    DISTRIBUTION_COMPARISON = "distribution_comparison"


class RegressionSeverity(Enum):
    """Levels severity regression for Enterprise alerting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelBaseline:
    """Baseline models for Enterprise regression testing"""
    model_id: str
    model_version: str
    metrics: Dict[str, float]
    test_data_hash: str
    predictions: np.ndarray
    feature_importances: Optional[Dict[str, float]] = None
    performance_stats: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionResult:
    """Result regression test - Enterprise structured result"""
    test_type: RegressionType
    severity: RegressionSeverity
    passed: bool
    message: str
    baseline_value: Optional[float] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    relative_change: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.test_type.value}: {self.message}"


@dataclass
class RegressionConfig:
    """Configuration regression testing - Enterprise typed configuration"""
    # Performance thresholds
    accuracy_threshold: float = 0.05  # 5% acceptable decrease accuracy
    latency_threshold_ms: float = 10.0  # 10ms acceptable increase latency
    memory_threshold_mb: float = 50.0  # 50MB acceptable increase memory
    
    # Prediction drift thresholds
    prediction_drift_threshold: float = 0.1  # 10% acceptable drift predictions
    distribution_shift_threshold: float = 0.05  # KS test p-value threshold
    
    # Feature importance changes
    feature_importance_threshold: float = 0.2  # 20% change importance features
    
    # Comparison methods
    accuracy_comparison: ComparisonMethod = ComparisonMethod.RELATIVE_THRESHOLD
    latency_comparison: ComparisonMethod = ComparisonMethod.ABSOLUTE_THRESHOLD
    prediction_comparison: ComparisonMethod = ComparisonMethod.STATISTICAL_TEST
    
    # Enterprise enterprise settings
    enable_statistical_tests: bool = True
    confidence_level: float = 0.95
    save_detailed_results: bool = True
    alert_on_regression: bool = True
    
    # Crypto trading specific
    price_prediction_tolerance: float = 0.02  # 2% tolerance for price predictions
 signal_accuracy_min_threshold: float = 0.7 # Minimum 70% accuracy for signal


class RegressionTester:
    """
    Enterprise Enterprise Regression Tester
    
 Comprehensive regression testing framework for ML models crypto trading systems.
 Provides continuous validation automated regression detection.
    """
    
    def __init__(self, config: Optional[RegressionConfig] = None, baseline_storage_path: str = "./baselines"):
        """
 Initialization regression tester Enterprise configuration
        
        Args:
            config: Configuration regression testing (Enterprise typed)
 baseline_storage_path: Path for baselines
        """
        self.config = config or RegressionConfig()
        self.baseline_storage_path = Path(baseline_storage_path)
        self.baseline_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._test_results: List[RegressionResult] = []
        
        # Enterprise statistical testing
        self._statistical_methods = self._init_statistical_methods()
    
    def _init_statistical_methods(self) -> Dict[str, Callable]:
        """Initialization statistical method for Enterprise testing"""
        methods = {}
        
        try:
            from scipy import stats
            methods['ks_test'] = stats.ks_2samp
            methods['ttest'] = stats.ttest_ind
            methods['mannwhitney'] = stats.mannwhitneyu
            methods['chi2_contingency'] = stats.chi2_contingency
        except ImportError:
            self.logger.warning("SciPy not set - tests unavailable")
        
        return methods
    
    def create_baseline(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_id: str,
        model_version: str = "1.0.0"
    ) -> ModelBaseline:
        """
        Create baseline models for future regression testing
        
        Args:
            model: ML model
            X_test: Test data (features)
            y_test: Test data (targets)
 model_id: ID models
 model_version: models
        
        Returns:
 ModelBaseline: baseline
        """
        self.logger.info(f"Create baseline for models {model_id} v{model_version}")
        
        # Generation hash test data
        test_data_hash = self._calculate_data_hash(X_test, y_test)
        
        # Retrieval predictions
        predictions = model.predict(X_test)
        
        # Calculation metrics
        metrics = self._calculate_metrics(y_test, predictions)
        
        # Feature importance (if available)
        feature_importances = self._extract_feature_importances(model, X_test.shape[1])
        
        # Performance statistics
        performance_stats = self._measure_performance_stats(model, X_test)
        
        # Create baseline
        baseline = ModelBaseline(
            model_id=model_id,
            model_version=model_version,
            metrics=metrics,
            test_data_hash=test_data_hash,
            predictions=predictions,
            feature_importances=feature_importances,
            performance_stats=performance_stats,
            metadata={
                'test_data_shape': X_test.shape,
                'prediction_shape': predictions.shape,
                'model_type': type(model).__name__
            }
        )
        
        # Save baseline
        self._save_baseline(baseline)
        
        self.logger.info(f"Baseline saved for {model_id}")
        return baseline
    
    def run_regression_tests(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_id: str,
        baseline_version: Optional[str] = None
    ) -> List[RegressionResult]:
        """
        Execution comprehensive regression testing against baseline
        
        Args:
 model: ML model for testing
            X_test: Test data (features)
            y_test: Test data (targets)
 model_id: ID models for search baseline
 baseline_version: baseline (optional)
        
        Returns:
            List[RegressionResult]: Results all regression tests
        """
        self.logger.info(f"Launch regression tests for models {model_id}")
        self._test_results.clear()
        
        # Load baseline
        baseline = self._load_baseline(model_id, baseline_version)
        if not baseline:
            self.logger.error(f"Baseline not for models {model_id}")
            return []
        
        # Verification compatibility test data
        test_data_hash = self._calculate_data_hash(X_test, y_test)
        if test_data_hash != baseline.test_data_hash:
            self.logger.warning("Test data creation baseline")
        
        # Retrieval current predictions metrics
        current_predictions = model.predict(X_test)
        current_metrics = self._calculate_metrics(y_test, current_predictions)
        current_performance = self._measure_performance_stats(model, X_test)
        
        # Execution regression tests
        try:
            # 1. Accuracy regression
            self._test_accuracy_regression(baseline.metrics, current_metrics)
            
            # 2. Performance regression
            self._test_performance_regression(baseline.performance_stats, current_performance)
            
            # 3. Prediction drift
            self._test_prediction_drift(baseline.predictions, current_predictions)
            
            # 4. Model behavior consistency
            self._test_model_behavior(baseline, model, X_test, y_test)
            
            # 5. Feature importance drift (if available)
            if baseline.feature_importances:
                current_feature_importances = self._extract_feature_importances(model, X_test.shape[1])
                if current_feature_importances:
                    self._test_feature_importance_drift(baseline.feature_importances, current_feature_importances)
            
            # 6. Crypto trading specific tests
            self._test_crypto_trading_specific(baseline, current_predictions, y_test)
            
            # regression
            self._assess_overall_regression()
            
        except Exception as e:
            self.logger.error(f"Error at execution regression tests: {e}")
            self._test_results.append(RegressionResult(
                test_type=RegressionType.MODEL_BEHAVIOR,
                severity=RegressionSeverity.CRITICAL,
                passed=False,
                message=f"Critical error regression testing: {str(e)}",
                details={'error': str(e)}
            ))
        
        self.logger.info(f"Regression tests completed. Result: {len(self._test_results)}")
        return self._test_results.copy()
    
    def _calculate_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """Calculation hash test data for"""
        combined_data = np.column_stack([X, y.reshape(-1, 1)])
        data_bytes = combined_data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculation comprehensive metrics for baseline comparison"""
        metrics = {}
        
        try:
            # Regression metrics
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            try:
                metrics['r2_score'] = r2_score(y_true, y_pred)
            except ValueError:
                metrics['r2_score'] = float('nan')
            
            # Mean Absolute Percentage Error
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    metrics['mape'] = mape
                else:
                    metrics['mape'] = float('inf')
            
            # Classification metrics (if applicable)
            if self._is_classification_task(y_true):
                y_true_class = y_true.astype(int)
                y_pred_class = np.round(y_pred).astype(int)
                
                try:
                    metrics['accuracy'] = accuracy_score(y_true_class, y_pred_class)
                    metrics['precision'] = precision_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
                    metrics['f1_score'] = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
                except Exception as e:
                    self.logger.debug(f"Error at classification metrics: {e}")
            
            # Crypto trading specific metrics
            if self._is_price_prediction_task(y_true, y_pred):
                metrics['directional_accuracy'] = self._calculate_directional_accuracy(y_true, y_pred)
                metrics['profit_factor'] = self._calculate_profit_factor(y_true, y_pred)
        
        except Exception as e:
            self.logger.error(f"Error at metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _is_classification_task(self, y_true: np.ndarray) -> bool:
        """Detection tasks - class or regression"""
        unique_values = np.unique(y_true)
        return len(unique_values) <= 10 and np.all(np.equal(np.mod(unique_values, 1), 0))
    
    def _is_price_prediction_task(self, y_true: np.ndarray, y_pred: np.ndarray) -> bool:
        """Detection tasks predictions prices"""
        # : positive values, range for prices
        return (np.all(y_true > 0) and np.all(y_pred > 0) and 
                np.median(y_true) > 0.01 and np.median(y_true) < 1e6)
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculation directional accuracy for trading predictions"""
        if len(y_true) < 2:
            return 0.0
        
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        return np.mean(true_direction == pred_direction)
    
    def _calculate_profit_factor(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculation profit factor for trading strategy simulation"""
        if len(y_true) < 2:
            return 1.0
        
        # Simulate simple trading strategy
        true_returns = np.diff(y_true) / y_true[:-1]
        predicted_direction = np.diff(y_pred) > 0
        
        # Calculate returns if following predictions
        strategy_returns = np.where(predicted_direction, true_returns, -true_returns)
        
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')
        
        gross_profit = np.sum(positive_returns)
        gross_loss = abs(np.sum(negative_returns))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _extract_feature_importances(self, model: Any, num_features: int) -> Optional[Dict[str, float]]:
        """feature importances from models"""
        try:
            importances = None
            
            # Scikit-learn models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            
            # Tree-based models
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()  # CatBoost
            
            if importances is not None and len(importances) == num_features:
                return {f'feature_{i}': float(imp) for i, imp in enumerate(importances)}
        
        except Exception as e:
            self.logger.debug(f"Not succeeded feature importances: {e}")
        
        return None
    
    def _measure_performance_stats(self, model: Any, X_test: np.ndarray) -> Dict[str, float]:
        """Measurement performance statistics models"""
        import time
        import psutil
        import gc
        
        stats = {}
        
        try:
            # Latency measurement
            latencies = []
            for _ in range(10):
                sample = X_test[:1]
                start_time = time.perf_counter()
                _ = model.predict(sample)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # ms
            
            stats['mean_latency_ms'] = np.mean(latencies)
            stats['p95_latency_ms'] = np.percentile(latencies, 95)
            
            # Memory usage
            gc.collect()
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Large batch prediction for memory measurement
            batch_size = min(100, len(X_test))
            _ = model.predict(X_test[:batch_size])
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            stats['memory_usage_mb'] = memory_after - memory_before
            
            # Throughput
            batch_data = X_test[:min(32, len(X_test))]
            start_time = time.perf_counter()
            _ = model.predict(batch_data)
            end_time = time.perf_counter()
            
            duration = end_time - start_time
            stats['throughput_predictions_per_sec'] = len(batch_data) / duration
        
        except Exception as e:
            self.logger.warning(f"Error at performance stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _test_accuracy_regression(self, baseline_metrics: Dict[str, float], current_metrics: Dict[str, float]) -> None:
        """Testing accuracy regression against baseline"""
        for metric_name in ['mae', 'mse', 'rmse', 'r2_score', 'accuracy', 'f1_score']:
            if metric_name not in baseline_metrics or metric_name not in current_metrics:
                continue
            
            baseline_value = baseline_metrics[metric_name]
            current_value = current_metrics[metric_name]
            
            if np.isnan(baseline_value) or np.isnan(current_value):
                continue
            
            # Detection changes ( = or = )
            higher_is_better = metric_name in ['r2_score', 'accuracy', 'f1_score', 'precision', 'recall']
            
            if higher_is_better:
                relative_change = (current_value - baseline_value) / baseline_value
                regression_detected = relative_change < -self.config.accuracy_threshold
                message = f"{metric_name} {baseline_value:.4f} before {current_value:.4f} ({relative_change:.2%})"
            else:
                relative_change = (current_value - baseline_value) / baseline_value
                regression_detected = relative_change > self.config.accuracy_threshold
                message = f"{metric_name} {baseline_value:.4f} before {current_value:.4f} ({relative_change:.2%})"
            
            # Detection severity
            if abs(relative_change) > 0.5:  # 50% change
                severity = RegressionSeverity.CRITICAL
            elif abs(relative_change) > 0.2:  # 20% change
                severity = RegressionSeverity.HIGH
            elif abs(relative_change) > 0.1:  # 10% change
                severity = RegressionSeverity.MEDIUM
            else:
                severity = RegressionSeverity.LOW
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.ACCURACY,
                severity=severity,
                passed=not regression_detected,
                message=message,
                baseline_value=baseline_value,
                current_value=current_value,
                threshold=self.config.accuracy_threshold,
                relative_change=relative_change,
                details={'metric_name': metric_name, 'higher_is_better': higher_is_better}
            ))
    
    def _test_performance_regression(self, baseline_perf: Dict[str, float], current_perf: Dict[str, float]) -> None:
        """Testing performance regression (latency, memory, throughput)"""
        # Latency regression
        if 'mean_latency_ms' in baseline_perf and 'mean_latency_ms' in current_perf:
            baseline_lat = baseline_perf['mean_latency_ms']
            current_lat = current_perf['mean_latency_ms']
            
            latency_increase = current_lat - baseline_lat
            regression_detected = latency_increase > self.config.latency_threshold_ms
            
            severity = RegressionSeverity.LOW
            if latency_increase > self.config.latency_threshold_ms * 5:
                severity = RegressionSeverity.CRITICAL
            elif latency_increase > self.config.latency_threshold_ms * 2:
                severity = RegressionSeverity.HIGH
            elif latency_increase > self.config.latency_threshold_ms:
                severity = RegressionSeverity.MEDIUM
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.LATENCY,
                severity=severity,
                passed=not regression_detected,
                message=f"Latency {baseline_lat:.2f}ms before {current_lat:.2f}ms (+{latency_increase:.2f}ms)",
                baseline_value=baseline_lat,
                current_value=current_lat,
                threshold=self.config.latency_threshold_ms,
                details={'latency_increase_ms': latency_increase}
            ))
        
        # Memory regression
        if 'memory_usage_mb' in baseline_perf and 'memory_usage_mb' in current_perf:
            baseline_mem = baseline_perf['memory_usage_mb']
            current_mem = current_perf['memory_usage_mb']
            
            memory_increase = current_mem - baseline_mem
            regression_detected = memory_increase > self.config.memory_threshold_mb
            
            severity = RegressionSeverity.LOW
            if memory_increase > self.config.memory_threshold_mb * 4:
                severity = RegressionSeverity.CRITICAL
            elif memory_increase > self.config.memory_threshold_mb * 2:
                severity = RegressionSeverity.HIGH
            elif memory_increase > self.config.memory_threshold_mb:
                severity = RegressionSeverity.MEDIUM
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.MEMORY_USAGE,
                severity=severity,
                passed=not regression_detected,
                message=f"Memory usage {baseline_mem:.1f}MB before {current_mem:.1f}MB (+{memory_increase:.1f}MB)",
                baseline_value=baseline_mem,
                current_value=current_mem,
                threshold=self.config.memory_threshold_mb,
                details={'memory_increase_mb': memory_increase}
            ))
        
        # Throughput regression
        if 'throughput_predictions_per_sec' in baseline_perf and 'throughput_predictions_per_sec' in current_perf:
            baseline_thr = baseline_perf['throughput_predictions_per_sec']
            current_thr = current_perf['throughput_predictions_per_sec']
            
            relative_change = (current_thr - baseline_thr) / baseline_thr
            regression_detected = relative_change < -0.1  # 10% decrease throughput
            
            severity = RegressionSeverity.LOW
            if relative_change < -0.5:
                severity = RegressionSeverity.CRITICAL
            elif relative_change < -0.3:
                severity = RegressionSeverity.HIGH
            elif relative_change < -0.1:
                severity = RegressionSeverity.MEDIUM
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.PERFORMANCE,
                severity=severity,
                passed=not regression_detected,
                message=f"Throughput {baseline_thr:.1f} before {current_thr:.1f} predictions/sec ({relative_change:.1%})",
                baseline_value=baseline_thr,
                current_value=current_thr,
                relative_change=relative_change,
                details={'throughput_change': relative_change}
            ))
    
    def _test_prediction_drift(self, baseline_predictions: np.ndarray, current_predictions: np.ndarray) -> None:
        """Testing prediction drift usage statistical tests"""
        # verification - mean absolute difference
        mean_abs_diff = np.mean(np.abs(baseline_predictions - current_predictions))
        baseline_mean = np.mean(np.abs(baseline_predictions))
        
        if baseline_mean > 0:
            relative_drift = mean_abs_diff / baseline_mean
        else:
            relative_drift = float('inf') if mean_abs_diff > 0 else 0.0
        
        drift_detected = relative_drift > self.config.prediction_drift_threshold
        
        # Detection severity
        if relative_drift > 0.5:
            severity = RegressionSeverity.CRITICAL
        elif relative_drift > 0.2:
            severity = RegressionSeverity.HIGH
        elif relative_drift > 0.1:
            severity = RegressionSeverity.MEDIUM
        else:
            severity = RegressionSeverity.LOW
        
        self._test_results.append(RegressionResult(
            test_type=RegressionType.PREDICTION_DRIFT,
            severity=severity,
            passed=not drift_detected,
            message=f"Prediction drift: change {relative_drift:.2%}",
            current_value=relative_drift,
            threshold=self.config.prediction_drift_threshold,
            details={
                'mean_absolute_difference': mean_abs_diff,
                'baseline_mean': baseline_mean,
                'relative_drift': relative_drift
            }
        ))
        
        # tests (if )
        if 'ks_test' in self._statistical_methods:
            try:
                ks_statistic, ks_p_value = self._statistical_methods['ks_test'](
                    baseline_predictions, current_predictions
                )
                
                # Kolmogorov-Smirnov test for distribution comparison
                distribution_changed = ks_p_value < self.config.distribution_shift_threshold
                
                self._test_results.append(RegressionResult(
                    test_type=RegressionType.PREDICTION_DRIFT,
                    severity=RegressionSeverity.MEDIUM if distribution_changed else RegressionSeverity.LOW,
                    passed=not distribution_changed,
                    message=f"Kolmogorov-Smirnov test: p-value={ks_p_value:.4f}, statistic={ks_statistic:.4f}",
                    current_value=ks_p_value,
                    threshold=self.config.distribution_shift_threshold,
                    details={
                        'test_type': 'kolmogorov_smirnov',
                        'ks_statistic': ks_statistic,
                        'ks_p_value': ks_p_value
                    }
                ))
                
            except Exception as e:
                self.logger.debug(f"KS test failed: {e}")
    
    def _test_model_behavior(self, baseline: ModelBaseline, current_model: Any, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Testing consistency models"""
        # Test on extreme values
        try:
            # extreme test cases
            X_mean = np.mean(X_test, axis=0)
            X_std = np.std(X_test, axis=0)
            
            # Extreme positive values
            X_extreme_pos = X_mean + 3 * X_std
            X_extreme_neg = X_mean - 3 * X_std
            
            extreme_cases = np.vstack([X_extreme_pos.reshape(1, -1), X_extreme_neg.reshape(1, -1)])
            
            current_extreme_preds = current_model.predict(extreme_cases)
            
            # Verification on reasonable predictions for extreme cases
            finite_predictions = np.isfinite(current_extreme_preds).all()
            reasonable_range = (np.abs(current_extreme_preds) < 1e6).all()  # Not more 1M
            
            behavior_consistent = finite_predictions and reasonable_range
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.MODEL_BEHAVIOR,
                severity=RegressionSeverity.HIGH if not behavior_consistent else RegressionSeverity.LOW,
                passed=behavior_consistent,
                message=f"Model behavior on extreme values: finite={finite_predictions}, reasonable={reasonable_range}",
                details={
                    'extreme_predictions': current_extreme_preds.tolist(),
                    'finite_predictions': finite_predictions,
                    'reasonable_range': reasonable_range
                }
            ))
            
        except Exception as e:
            self._test_results.append(RegressionResult(
                test_type=RegressionType.MODEL_BEHAVIOR,
                severity=RegressionSeverity.MEDIUM,
                passed=False,
                message=f"Error at model behavior: {str(e)}",
                details={'error': str(e)}
            ))
    
    def _test_feature_importance_drift(self, baseline_importances: Dict[str, float], current_importances: Dict[str, float]) -> None:
        """Testing drift feature importance"""
        common_features = set(baseline_importances.keys()) & set(current_importances.keys())
        
        if not common_features:
            self._test_results.append(RegressionResult(
                test_type=RegressionType.FEATURE_IMPORTANCE,
                severity=RegressionSeverity.MEDIUM,
                passed=False,
                message="features for comparison importance",
                details={'baseline_features': len(baseline_importances), 'current_features': len(current_importances)}
            ))
            return
        
        # Calculation relative changes importance
        importance_changes = []
        significant_changes = []
        
        for feature in common_features:
            baseline_imp = baseline_importances[feature]
            current_imp = current_importances[feature]
            
            if baseline_imp > 0:
                relative_change = (current_imp - baseline_imp) / baseline_imp
                importance_changes.append(abs(relative_change))
                
                if abs(relative_change) > self.config.feature_importance_threshold:
                    significant_changes.append({
                        'feature': feature,
                        'baseline': baseline_imp,
                        'current': current_imp,
                        'relative_change': relative_change
                    })
        
        # drift
        mean_importance_change = np.mean(importance_changes) if importance_changes else 0.0
        max_importance_change = np.max(importance_changes) if importance_changes else 0.0
        
        significant_drift = len(significant_changes) > 0
        
        severity = RegressionSeverity.LOW
        if len(significant_changes) > len(common_features) * 0.5:  # >50% features changed significantly
            severity = RegressionSeverity.HIGH
        elif len(significant_changes) > len(common_features) * 0.2:  # >20% features changed
            severity = RegressionSeverity.MEDIUM
        
        self._test_results.append(RegressionResult(
            test_type=RegressionType.FEATURE_IMPORTANCE,
            severity=severity,
            passed=not significant_drift,
            message=f"Feature importance drift: {len(significant_changes)}/{len(common_features)} features",
            current_value=mean_importance_change,
            threshold=self.config.feature_importance_threshold,
            details={
                'mean_change': mean_importance_change,
                'max_change': max_importance_change,
                'significant_changes': significant_changes,
                'total_features_compared': len(common_features)
            }
        ))
    
    def _test_crypto_trading_specific(self, baseline: ModelBaseline, current_predictions: np.ndarray, y_test: np.ndarray) -> None:
        """Crypto trading specific regression tests"""
        # Directional accuracy comparison (if this price prediction)
        if 'directional_accuracy' in baseline.metrics:
            current_directional_acc = self._calculate_directional_accuracy(y_test, current_predictions)
            baseline_directional_acc = baseline.metrics['directional_accuracy']
            
            accuracy_drop = baseline_directional_acc - current_directional_acc
            significant_drop = accuracy_drop > self.config.price_prediction_tolerance
            
            severity = RegressionSeverity.LOW
            if accuracy_drop > 0.1:  # 10% drop
                severity = RegressionSeverity.CRITICAL
            elif accuracy_drop > 0.05:  # 5% drop
                severity = RegressionSeverity.HIGH
            elif accuracy_drop > 0.02:  # 2% drop
                severity = RegressionSeverity.MEDIUM
            
            self._test_results.append(RegressionResult(
                test_type=RegressionType.ACCURACY,
                severity=severity,
                passed=not significant_drop,
                message=f"Directional accuracy: {baseline_directional_acc:.3f} -> {current_directional_acc:.3f} (Δ{accuracy_drop:.3f})",
                baseline_value=baseline_directional_acc,
                current_value=current_directional_acc,
                threshold=self.config.price_prediction_tolerance,
                details={'metric_type': 'directional_accuracy', 'accuracy_drop': accuracy_drop}
            ))
        
        # Trading signal accuracy (if applicable)
        if self._is_classification_task(y_test):
            current_metrics = self._calculate_metrics(y_test, current_predictions)
            if 'accuracy' in current_metrics:
                signal_accuracy = current_metrics['accuracy']
                below_threshold = signal_accuracy < self.config.signal_accuracy_min_threshold
                
                self._test_results.append(RegressionResult(
                    test_type=RegressionType.ACCURACY,
                    severity=RegressionSeverity.CRITICAL if below_threshold else RegressionSeverity.LOW,
                    passed=not below_threshold,
                    message=f"Trading signal accuracy: {signal_accuracy:.3f} (minimum: {self.config.signal_accuracy_min_threshold:.3f})",
                    current_value=signal_accuracy,
                    threshold=self.config.signal_accuracy_min_threshold,
                    details={'metric_type': 'trading_signal_accuracy'}
                ))
    
    def _assess_overall_regression(self) -> None:
        """regression by all test"""
        if not self._test_results:
            return
        
        # Counting results by severity
        critical_failures = sum(1 for r in self._test_results if not r.passed and r.severity == RegressionSeverity.CRITICAL)
        high_failures = sum(1 for r in self._test_results if not r.passed and r.severity == RegressionSeverity.HIGH)
        medium_failures = sum(1 for r in self._test_results if not r.passed and r.severity == RegressionSeverity.MEDIUM)
        low_failures = sum(1 for r in self._test_results if not r.passed and r.severity == RegressionSeverity.LOW)
        
        total_tests = len(self._test_results)
        passed_tests = sum(1 for r in self._test_results if r.passed)
        
        # Detection total status
        if critical_failures > 0:
            overall_status = "CRITICAL_REGRESSION"
            overall_severity = RegressionSeverity.CRITICAL
        elif high_failures > 0:
            overall_status = "HIGH_REGRESSION"
            overall_severity = RegressionSeverity.HIGH
        elif medium_failures > 0:
            overall_status = "MEDIUM_REGRESSION"
            overall_severity = RegressionSeverity.MEDIUM
        elif low_failures > 0:
            overall_status = "LOW_REGRESSION"
            overall_severity = RegressionSeverity.LOW
        else:
            overall_status = "NO_REGRESSION"
            overall_severity = RegressionSeverity.LOW
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 1.0
        
        # Addition total result
        self._test_results.append(RegressionResult(
            test_type=RegressionType.MODEL_BEHAVIOR,
            severity=overall_severity,
            passed=(overall_status == "NO_REGRESSION"),
            message=f"status regression: {overall_status} ({passed_tests}/{total_tests} tests )",
            current_value=success_rate,
            details={
                'overall_status': overall_status,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'critical_failures': critical_failures,
                'high_failures': high_failures,
                'medium_failures': medium_failures,
                'low_failures': low_failures,
                'success_rate': success_rate
            }
        ))
    
    def _save_baseline(self, baseline: ModelBaseline) -> None:
        """Save baseline files systems"""
        baseline_path = self.baseline_storage_path / f"{baseline.model_id}_{baseline.model_version}.pkl"
        
        try:
            with open(baseline_path, 'wb') as f:
                pickle.dump(baseline, f)
            self.logger.info(f"Baseline saved: {baseline_path}")
        except Exception as e:
            self.logger.error(f"Error at saving baseline: {e}")
            raise
    
    def _load_baseline(self, model_id: str, version: Optional[str] = None) -> Optional[ModelBaseline]:
        """Load baseline from files system"""
        if version:
            baseline_path = self.baseline_storage_path / f"{model_id}_{version}.pkl"
            if baseline_path.exists():
                try:
                    with open(baseline_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.error(f"Error at loading baseline {baseline_path}: {e}")
                    return None
        else:
            # Search after baseline
            pattern = f"{model_id}_*.pkl"
            baseline_files = list(self.baseline_storage_path.glob(pattern))
            
            if baseline_files:
                # Sorting by time
                latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_baseline, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    self.logger.error(f"Error at loading baseline {latest_baseline}: {e}")
                    return None
        
        self.logger.warning(f"Baseline not for models {model_id}")
        return None
    
    def list_baselines(self) -> List[Dict[str, Any]]:
        """List all baselines"""
        baselines = []
        
        for baseline_file in self.baseline_storage_path.glob("*.pkl"):
            try:
                with open(baseline_file, 'rb') as f:
                    baseline = pickle.load(f)
                    baselines.append({
                        'model_id': baseline.model_id,
                        'version': baseline.model_version,
                        'created_at': baseline.created_at.isoformat(),
                        'file_path': str(baseline_file),
                        'metrics': baseline.metrics
                    })
            except Exception as e:
                self.logger.warning(f"Not succeeded baseline {baseline_file}: {e}")
        
        return sorted(baselines, key=lambda x: x['created_at'], reverse=True)
    
    def generate_regression_report(self) -> Dict[str, Any]:
        """Generation comprehensive regression report"""
        if not self._test_results:
            return {'status': 'no_tests_run'}
        
        # Grouping results by tests
        results_by_type = {}
        for result in self._test_results:
            test_type = result.test_type.value
            if test_type not in results_by_type:
                results_by_type[test_type] = []
            results_by_type[test_type].append(result)
        
        # Statistics by severity levels
        severity_stats = {}
        for severity in RegressionSeverity:
            severity_stats[severity.value] = {
                'total': sum(1 for r in self._test_results if r.severity == severity),
                'passed': sum(1 for r in self._test_results if r.severity == severity and r.passed),
                'failed': sum(1 for r in self._test_results if r.severity == severity and not r.passed)
            }
        
        # statistics
        total_tests = len(self._test_results)
        passed_tests = sum(1 for r in self._test_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Critical issues
        critical_issues = [
            {
                'test_type': r.test_type.value,
                'message': r.message,
                'baseline_value': r.baseline_value,
                'current_value': r.current_value,
                'relative_change': r.relative_change
            }
            for r in self._test_results 
            if not r.passed and r.severity == RegressionSeverity.CRITICAL
        ]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 1.0,
                'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
            },
            'severity_breakdown': severity_stats,
            'critical_issues': critical_issues,
            'results_by_type': {
                test_type: [
                    {
                        'passed': r.passed,
                        'severity': r.severity.value,
                        'message': r.message,
                        'baseline_value': r.baseline_value,
                        'current_value': r.current_value,
                        'threshold': r.threshold,
                        'relative_change': r.relative_change,
                        'details': r.details
                    }
                    for r in results
                ]
                for test_type, results in results_by_type.items()
            },
            'recommendations': self._generate_regression_recommendations()
        }
        
        return report
    
    def _generate_regression_recommendations(self) -> List[str]:
        """Generation by result regression testing"""
        recommendations = []
        
        failed_results = [r for r in self._test_results if not r.passed]
        
        # Accuracy regression recommendations
        accuracy_failures = [r for r in failed_results if r.test_type == RegressionType.ACCURACY]
        if accuracy_failures:
            recommendations.append(
                "Detected decrease accuracy - recommended training data"
                "hyperparameters models"
            )
        
        # Performance regression recommendations
        latency_failures = [r for r in failed_results if r.test_type == RegressionType.LATENCY]
        if latency_failures:
            recommendations.append(
                "Increase latency models - recommended inference pipeline"
            )
        
        memory_failures = [r for r in failed_results if r.test_type == RegressionType.MEMORY_USAGE]
        if memory_failures:
            recommendations.append(
                "Increase memory - recommended analysis memory leaks models"
            )
        
        # Prediction drift recommendations
        drift_failures = [r for r in failed_results if r.test_type == RegressionType.PREDICTION_DRIFT]
        if drift_failures:
            recommendations.append(
                "Detected prediction drift - recommended analysis distribution shift "
                "input data overfitting models"
            )
        
        # Feature importance recommendations
        feature_failures = [r for r in failed_results if r.test_type == RegressionType.FEATURE_IMPORTANCE]
        if feature_failures:
            recommendations.append(
                "changes feature importance - recommended analysis"
                "stability features feature engineering pipeline"
            )
        
        # Critical failures
        critical_failures = [r for r in failed_results if r.severity == RegressionSeverity.CRITICAL]
        if critical_failures:
            recommendations.append(
                ": Detected critical regression - model production"
                "before elimination all critical issues"
            )
        
        if not recommendations:
            recommendations.append("All regression tests - model deployment")
        
        return recommendations
    
    def export_results(self, filepath: str) -> None:
        """Export results regression testing"""
        report = self.generate_regression_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Regression report exported {filepath}")


def create_crypto_trading_regression_tester(baseline_storage_path: str = "./crypto_baselines") -> RegressionTester:
    """
    Factory function for creation regression tester for crypto trading models
    Enterprise pre-configured tester for financial ML systems
    """
    config = RegressionConfig(
        # Strict requirements for crypto trading
        accuracy_threshold=0.02,  # 2% acceptable decrease accuracy
        latency_threshold_ms=5.0,  # 5ms maximum increase latency
        memory_threshold_mb=20.0,  # 20MB maximum increase memory
        
        # Crypto trading specific
        price_prediction_tolerance=0.01,  # 1% tolerance for price predictions
        signal_accuracy_min_threshold=0.75,  # 75% minimum for trading signals
        
        # Prediction drift for financial data
        prediction_drift_threshold=0.05,  # 5% acceptable drift
        distribution_shift_threshold=0.01,  # Strict p-value threshold
        
        # Feature importance for trading models
        feature_importance_threshold=0.15,  # 15% change importance features
        
        # Enterprise enterprise settings
        enable_statistical_tests=True,
 confidence_level=0.99, # High confidence for
        save_detailed_results=True,
        alert_on_regression=True
    )
    
    return RegressionTester(config, baseline_storage_path)