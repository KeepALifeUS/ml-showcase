"""
Metrics Calculator - Enterprise Enterprise ML Metrics Calculation
Comprehensive metrics calculation framework for ML models crypto trading

Applies enterprise principles:
- Enterprise-grade metrics
- Production performance monitoring
- Multi-framework compatibility
- Statistical significance testing
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import logging
import json

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types metrics for Enterprise ML evaluation"""
    # Performance metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PR_AUC = "pr_auc"
    
    # Regression metrics
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    MAPE = "mape"
    R_SQUARED = "r_squared"
    
    # Time series metrics
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    HIT_RATE = "hit_rate"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    PROFIT_FACTOR = "profit_factor"
    
    # Latency metrics
    MEAN_LATENCY = "mean_latency"
    MEDIAN_LATENCY = "median_latency"
    P95_LATENCY = "p95_latency"
    P99_LATENCY = "p99_latency"
    
    # Throughput metrics
    THROUGHPUT = "throughput"
    BATCH_EFFICIENCY = "batch_efficiency"
    
    # Memory metrics
    MEMORY_USAGE = "memory_usage"
    MEMORY_EFFICIENCY = "memory_efficiency"
    
    # Stability metrics
    PREDICTION_STABILITY = "prediction_stability"
    MODEL_CONSISTENCY = "model_consistency"
    
    # Business metrics (crypto trading specific)
    TRADING_RETURN = "trading_return"
    WIN_RATE = "win_rate"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


class MetricCategory(Enum):
    """metrics for Enterprise organization"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    STABILITY = "stability"


@dataclass
class MetricResult:
    """Result metrics - Enterprise structured result"""
    metric_name: str
    metric_type: MetricType
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    statistical_significance: Optional[float] = None
    sample_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialization metrics"""
        return {
            'metric_name': self.metric_name,
            'metric_type': self.metric_type.value,
            'value': self.value,
            'confidence_interval': self.confidence_interval,
            'statistical_significance': self.statistical_significance,
            'sample_size': self.sample_size,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class MetricsConfig:
    """Configuration metrics calculation - Enterprise typed configuration"""
    # Bootstrap settings for confidence intervals
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Significance testing
    enable_significance_testing: bool = True
    significance_threshold: float = 0.05
    
    # Time series settings
    time_series_frequency: str = "1min"  # pandas frequency string
    enable_autocorrelation_adjustment: bool = True
    
    # Business metrics settings
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_cost_bps: float = 5.0  # 5 basis points trading cost
    
    # Performance metrics settings
    latency_unit: str = "ms"  # "ms", "us", "ns"
    memory_unit: str = "MB"  # "KB", "MB", "GB"
    
    # Stability metrics settings
    stability_window_size: int = 100
    consistency_threshold: float = 0.1  # 10% variation threshold
    
    # Enterprise enterprise settings
    enable_detailed_analysis: bool = True
    save_intermediate_results: bool = False
    generate_visualizations: bool = False
    
    # Crypto trading specific
    crypto_return_calculation: str = "log"  # "simple", "log"
    enable_volatility_adjustment: bool = True
    slippage_bps: float = 2.0  # 2 basis points slippage


class MetricsCalculator:
    """
    Enterprise Enterprise Metrics Calculator
    
 Comprehensive metrics calculation framework for ML models crypto trading.
 Provides enterprise-grade metrics statistical significance testing.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
 Initialization metrics calculator Enterprise configuration
        
        Args:
            config: Configuration metrics calculation (Enterprise typed)
        """
        self.config = config or MetricsConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enterprise metrics registry
        self.calculated_metrics: Dict[str, MetricResult] = {}
        self.metric_history: List[MetricResult] = []
        
        # Statistical methods
        self._init_statistical_methods()
    
    def _init_statistical_methods(self) -> None:
        """Initialization statistical method"""
        try:
            from scipy import stats
            from sklearn import metrics
            self.stats_available = True
            self.sklearn_available = True
        except ImportError:
            self.stats_available = False
            self.sklearn_available = False
            self.logger.warning("SciPy or scikit-learn unavailable - metrics will be skipped")
    
    def calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_labels: Optional[List[str]] = None
    ) -> Dict[str, MetricResult]:
        """
 Calculation classification metrics Enterprise statistical analysis
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            class_labels: Class labels (optional)
        
        Returns:
            Dict[str, MetricResult]: Calculated classification metrics
        """
        self.logger.info("Calculating classification metrics")
        results = {}
        
        if not self.sklearn_available:
            self.logger.error("scikit-learn unavailable for classification metrics")
            return results
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, confusion_matrix,
            classification_report
        )
        
        try:
            # Basic classification metrics
            accuracy = accuracy_score(y_true, y_pred)
            results['accuracy'] = MetricResult(
                metric_name='accuracy',
                metric_type=MetricType.ACCURACY,
                value=accuracy,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    lambda y_t, y_p: accuracy_score(y_t, y_p), y_true, y_pred
                ),
                metadata={'metric_category': MetricCategory.CLASSIFICATION.value}
            )
            
            # Multi-class handling
            avg_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
            
            precision = precision_score(y_true, y_pred, average=avg_method, zero_division=0)
            results['precision'] = MetricResult(
                metric_name='precision',
                metric_type=MetricType.PRECISION,
                value=precision,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    lambda y_t, y_p: precision_score(y_t, y_p, average=avg_method, zero_division=0),
                    y_true, y_pred
                ),
                metadata={'average_method': avg_method}
            )
            
            recall = recall_score(y_true, y_pred, average=avg_method, zero_division=0)
            results['recall'] = MetricResult(
                metric_name='recall',
                metric_type=MetricType.RECALL,
                value=recall,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    lambda y_t, y_p: recall_score(y_t, y_p, average=avg_method, zero_division=0),
                    y_true, y_pred
                ),
                metadata={'average_method': avg_method}
            )
            
            f1 = f1_score(y_true, y_pred, average=avg_method, zero_division=0)
            results['f1_score'] = MetricResult(
                metric_name='f1_score',
                metric_type=MetricType.F1_SCORE,
                value=f1,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    lambda y_t, y_p: f1_score(y_t, y_p, average=avg_method, zero_division=0),
                    y_true, y_pred
                ),
                metadata={'average_method': avg_method}
            )
            
            # ROC AUC (if is probabilities or binary classification)
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        if y_pred_proba.ndim == 2:
                            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                        else:
                            roc_auc = roc_auc_score(y_true, y_pred_proba)
                    else:
                        # Multi-class
                        roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    
                    results['roc_auc'] = MetricResult(
                        metric_name='roc_auc',
                        metric_type=MetricType.ROC_AUC,
                        value=roc_auc,
                        sample_size=len(y_true),
                        metadata={'probabilities_used': True}
                    )
                    
                    # Precision-Recall AUC
                    if len(np.unique(y_true)) == 2:
                        pr_auc_scores = y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba
                        pr_auc = average_precision_score(y_true, pr_auc_scores)
                        results['pr_auc'] = MetricResult(
                            metric_name='pr_auc',
                            metric_type=MetricType.PR_AUC,
                            value=pr_auc,
                            sample_size=len(y_true)
                        )
                
                except Exception as e:
                    self.logger.warning(f"AUC calculation failed: {e}")
            
            # Confusion Matrix analysis
            cm = confusion_matrix(y_true, y_pred)
            results['confusion_matrix_analysis'] = MetricResult(
                metric_name='confusion_matrix_analysis',
                metric_type=MetricType.ACCURACY,  # Use accuracy as base type
                value=accuracy,  # Overall accuracy
                sample_size=len(y_true),
                metadata={
                    'confusion_matrix': cm.tolist(),
                    'class_labels': class_labels or list(range(len(cm))),
                    'true_positives': np.diag(cm).tolist(),
                    'false_positives': (cm.sum(axis=0) - np.diag(cm)).tolist(),
                    'false_negatives': (cm.sum(axis=1) - np.diag(cm)).tolist(),
                }
            )
            
        except Exception as e:
            self.logger.error(f"Classification metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """
 Calculation regression metrics Enterprise statistical analysis
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            Dict[str, MetricResult]: Calculated regression metrics
        """
        self.logger.info("Calculating regression metrics")
        results = {}
        
        if not self.sklearn_available:
            self.logger.error("scikit-learn unavailable for regression metrics")
            return results
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        try:
            # Mean Absolute Error
            mae = mean_absolute_error(y_true, y_pred)
            results['mae'] = MetricResult(
                metric_name='mae',
                metric_type=MetricType.MAE,
                value=mae,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    mean_absolute_error, y_true, y_pred
                ),
                metadata={'metric_category': MetricCategory.REGRESSION.value}
            )
            
            # Mean Squared Error
            mse = mean_squared_error(y_true, y_pred)
            results['mse'] = MetricResult(
                metric_name='mse',
                metric_type=MetricType.MSE,
                value=mse,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    mean_squared_error, y_true, y_pred
                )
            )
            
            # Root Mean Squared Error
            rmse = np.sqrt(mse)
            results['rmse'] = MetricResult(
                metric_name='rmse',
                metric_type=MetricType.RMSE,
                value=rmse,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    lambda y_t, y_p: np.sqrt(mean_squared_error(y_t, y_p)), y_true, y_pred
                )
            )
            
            # R-squared
            r2 = r2_score(y_true, y_pred)
            results['r_squared'] = MetricResult(
                metric_name='r_squared',
                metric_type=MetricType.R_SQUARED,
                value=r2,
                sample_size=len(y_true),
                confidence_interval=self._bootstrap_confidence_interval(
                    r2_score, y_true, y_pred
                )
            )
            
            # Mean Absolute Percentage Error
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                non_zero_mask = y_true != 0
                if np.any(non_zero_mask):
                    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
                    results['mape'] = MetricResult(
                        metric_name='mape',
                        metric_type=MetricType.MAPE,
                        value=mape,
                        sample_size=np.sum(non_zero_mask),
                        metadata={'excluded_zero_values': len(y_true) - np.sum(non_zero_mask)}
                    )
            
            # Additional regression metrics
            residuals = y_true - y_pred
            results['residual_analysis'] = MetricResult(
                metric_name='residual_analysis',
                metric_type=MetricType.MSE,  # Use MSE as base type
                value=mse,
                sample_size=len(y_true),
                metadata={
                    'residual_mean': float(np.mean(residuals)),
                    'residual_std': float(np.std(residuals)),
                    'residual_skewness': float(stats.skew(residuals)) if self.stats_available else None,
                    'residual_kurtosis': float(stats.kurtosis(residuals)) if self.stats_available else None,
                    'residual_autocorrelation': self._calculate_autocorrelation(residuals) if self.stats_available else None
                }
            )
            
        except Exception as e:
            self.logger.error(f"Regression metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def calculate_time_series_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculation time series specific metrics for crypto trading
        
        Args:
            y_true: True values (prices/returns)
            y_pred: Predicted values
            timestamps: Temporal labels (optional)
        
        Returns:
            Dict[str, MetricResult]: Time series specific metrics
        """
        self.logger.info("Calculating time series metrics")
        results = {}
        
        try:
            # Directional Accuracy
            if len(y_true) > 1:
                true_direction = np.diff(y_true) > 0
                pred_direction = np.diff(y_pred) > 0
                
                directional_accuracy = np.mean(true_direction == pred_direction)
                results['directional_accuracy'] = MetricResult(
                    metric_name='directional_accuracy',
                    metric_type=MetricType.DIRECTIONAL_ACCURACY,
                    value=directional_accuracy,
                    sample_size=len(true_direction),
                    confidence_interval=self._bootstrap_confidence_interval(
                        lambda y_t, y_p: np.mean(np.diff(y_t) > 0 == np.diff(y_p) > 0),
                        y_true, y_pred
                    ),
                    metadata={'metric_category': MetricCategory.TIME_SERIES.value}
                )
            
            # Hit Rate (percentage of correct predictions within threshold)
            threshold = np.std(y_true) * 0.1  # 10% of std as threshold
            hits = np.abs(y_true - y_pred) <= threshold
            hit_rate = np.mean(hits)
            
            results['hit_rate'] = MetricResult(
                metric_name='hit_rate',
                metric_type=MetricType.HIT_RATE,
                value=hit_rate,
                sample_size=len(y_true),
                metadata={'threshold': threshold}
            )
            
            # Calculate returns for financial metrics
            if self.config.crypto_return_calculation == "log":
                true_returns = np.diff(np.log(y_true + 1e-8))  # Avoid log(0)
                pred_returns = np.diff(np.log(y_pred + 1e-8))
            else:
                true_returns = np.diff(y_true) / (y_true[:-1] + 1e-8)
                pred_returns = np.diff(y_pred) / (y_pred[:-1] + 1e-8)
            
            # Trading simulation metrics
            trading_metrics = self._calculate_trading_metrics(true_returns, pred_returns)
            results.update(trading_metrics)
            
        except Exception as e:
            self.logger.error(f"Time series metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def calculate_performance_metrics(
        self,
        latencies: List[float],
        throughputs: List[float],
        memory_usage: List[float],
        batch_sizes: Optional[List[int]] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculation performance metrics for ML models
        
        Args:
 latencies: Latency measurements ( ms)
            throughputs: Throughput measurements (predictions/sec)
 memory_usage: Memory usage measurements ( MB)
            batch_sizes: Batch sizes for efficiency calculation
        
        Returns:
            Dict[str, MetricResult]: Performance metrics
        """
        self.logger.info("Calculating performance metrics")
        results = {}
        
        try:
            # Latency metrics
            if latencies:
                results['mean_latency'] = MetricResult(
                    metric_name='mean_latency',
                    metric_type=MetricType.MEAN_LATENCY,
                    value=np.mean(latencies),
                    sample_size=len(latencies),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.mean, latencies),
                    metadata={
                        'unit': self.config.latency_unit,
                        'metric_category': MetricCategory.PERFORMANCE.value
                    }
                )
                
                results['median_latency'] = MetricResult(
                    metric_name='median_latency',
                    metric_type=MetricType.MEDIAN_LATENCY,
                    value=np.median(latencies),
                    sample_size=len(latencies),
                    metadata={'unit': self.config.latency_unit}
                )
                
                results['p95_latency'] = MetricResult(
                    metric_name='p95_latency',
                    metric_type=MetricType.P95_LATENCY,
                    value=np.percentile(latencies, 95),
                    sample_size=len(latencies),
                    metadata={'unit': self.config.latency_unit}
                )
                
                results['p99_latency'] = MetricResult(
                    metric_name='p99_latency',
                    metric_type=MetricType.P99_LATENCY,
                    value=np.percentile(latencies, 99),
                    sample_size=len(latencies),
                    metadata={'unit': self.config.latency_unit}
                )
            
            # Throughput metrics
            if throughputs:
                max_throughput = np.max(throughputs)
                results['max_throughput'] = MetricResult(
                    metric_name='max_throughput',
                    metric_type=MetricType.THROUGHPUT,
                    value=max_throughput,
                    sample_size=len(throughputs),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.max, throughputs),
                    metadata={'unit': 'predictions/sec'}
                )
                
                results['mean_throughput'] = MetricResult(
                    metric_name='mean_throughput',
                    metric_type=MetricType.THROUGHPUT,
                    value=np.mean(throughputs),
                    sample_size=len(throughputs),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.mean, throughputs),
                    metadata={'unit': 'predictions/sec'}
                )
            
            # Memory metrics
            if memory_usage:
                results['max_memory_usage'] = MetricResult(
                    metric_name='max_memory_usage',
                    metric_type=MetricType.MEMORY_USAGE,
                    value=np.max(memory_usage),
                    sample_size=len(memory_usage),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.max, memory_usage),
                    metadata={'unit': self.config.memory_unit}
                )
                
                results['mean_memory_usage'] = MetricResult(
                    metric_name='mean_memory_usage',
                    metric_type=MetricType.MEMORY_USAGE,
                    value=np.mean(memory_usage),
                    sample_size=len(memory_usage),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.mean, memory_usage),
                    metadata={'unit': self.config.memory_unit}
                )
            
            # Batch efficiency
            if batch_sizes and throughputs and len(batch_sizes) == len(throughputs):
                # Calculate efficiency as throughput per unit batch size
                efficiencies = [t / b for t, b in zip(throughputs, batch_sizes)]
                batch_efficiency = np.mean(efficiencies)
                
                results['batch_efficiency'] = MetricResult(
                    metric_name='batch_efficiency',
                    metric_type=MetricType.BATCH_EFFICIENCY,
                    value=batch_efficiency,
                    sample_size=len(efficiencies),
                    metadata={
                        'efficiency_values': efficiencies,
                        'batch_sizes': batch_sizes,
                        'throughputs': throughputs
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Performance metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def calculate_stability_metrics(
        self,
        predictions_history: List[np.ndarray],
        model_versions: Optional[List[str]] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculation stability metrics for model consistency
        
        Args:
            predictions_history: History predictions by versions/time
 model_versions: models (optional)
        
        Returns:
            Dict[str, MetricResult]: Stability metrics
        """
        self.logger.info("Calculating stability metrics")
        results = {}
        
        try:
            if len(predictions_history) < 2:
                self.logger.warning("Insufficient data for stability metrics")
                return results
            
            # Prediction stability - consistency across different runs
            prediction_stabilities = []
            
            for i in range(len(predictions_history) - 1):
                pred1 = predictions_history[i]
                pred2 = predictions_history[i + 1]
                
                # Align predictions if different lengths
                min_len = min(len(pred1), len(pred2))
                pred1_aligned = pred1[:min_len]
                pred2_aligned = pred2[:min_len]
                
                # Calculate stability as correlation
                if min_len > 1:
                    correlation = np.corrcoef(pred1_aligned, pred2_aligned)[0, 1]
                    if not np.isnan(correlation):
                        prediction_stabilities.append(correlation)
            
            if prediction_stabilities:
                mean_stability = np.mean(prediction_stabilities)
                results['prediction_stability'] = MetricResult(
                    metric_name='prediction_stability',
                    metric_type=MetricType.PREDICTION_STABILITY,
                    value=mean_stability,
                    sample_size=len(prediction_stabilities),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.mean, prediction_stabilities),
                    metadata={
                        'stability_correlations': prediction_stabilities,
                        'metric_category': MetricCategory.STABILITY.value
                    }
                )
            
            # Model consistency - variability in predictions
            all_predictions = np.concatenate(predictions_history)
            if len(all_predictions) > 0:
                # Coefficient of variation as consistency measure
                cv = np.std(all_predictions) / (abs(np.mean(all_predictions)) + 1e-8)
                consistency_score = 1 / (1 + cv)  # Higher is more consistent
                
                results['model_consistency'] = MetricResult(
                    metric_name='model_consistency',
                    metric_type=MetricType.MODEL_CONSISTENCY,
                    value=consistency_score,
                    sample_size=len(all_predictions),
                    metadata={
                        'coefficient_of_variation': cv,
                        'total_predictions': len(all_predictions),
                        'prediction_mean': np.mean(all_predictions),
                        'prediction_std': np.std(all_predictions)
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Stability metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        trading_costs: Optional[float] = None
    ) -> Dict[str, MetricResult]:
        """
        Calculation business metrics for crypto trading
        
        Args:
            y_true: True values (prices/returns)
            y_pred: Predicted values
 trading_costs: Trading costs basis points
        
        Returns:
            Dict[str, MetricResult]: Business-specific metrics
        """
        self.logger.info("Calculating business metrics for crypto trading")
        results = {}
        
        try:
            trading_costs_bps = trading_costs or self.config.trading_cost_bps
            
            # Simulate trading strategy based on predictions
            if len(y_true) > 1 and len(y_pred) > 1:
                # Calculate returns
                if self.config.crypto_return_calculation == "log":
                    true_returns = np.diff(np.log(y_true + 1e-8))
                else:
                    true_returns = np.diff(y_true) / (y_true[:-1] + 1e-8)
                
                # Trading signals based on predictions
                pred_direction = np.diff(y_pred) > 0
                
                # Strategy returns (go long when pred_direction is positive)
                strategy_returns = np.where(pred_direction, true_returns, -true_returns)
                
                # Apply trading costs
                gross_returns = strategy_returns - (trading_costs_bps / 10000)  # Convert bps to decimal
                
                # Trading metrics
                total_return = np.sum(gross_returns)
                results['trading_return'] = MetricResult(
                    metric_name='trading_return',
                    metric_type=MetricType.TRADING_RETURN,
                    value=total_return,
                    sample_size=len(gross_returns),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.sum, gross_returns),
                    metadata={
                        'trading_costs_bps': trading_costs_bps,
                        'metric_category': MetricCategory.BUSINESS.value
                    }
                )
                
                # Win rate
                winning_trades = gross_returns > 0
                win_rate = np.mean(winning_trades)
                results['win_rate'] = MetricResult(
                    metric_name='win_rate',
                    metric_type=MetricType.WIN_RATE,
                    value=win_rate,
                    sample_size=len(gross_returns),
                    confidence_interval=self._bootstrap_confidence_interval_simple(np.mean, winning_trades),
                    metadata={'total_trades': len(gross_returns)}
                )
                
                # Average win/loss
                if np.any(winning_trades):
                    avg_win = np.mean(gross_returns[winning_trades])
                    results['average_win'] = MetricResult(
                        metric_name='average_win',
                        metric_type=MetricType.AVERAGE_WIN,
                        value=avg_win,
                        sample_size=np.sum(winning_trades),
                        metadata={'winning_trades_count': np.sum(winning_trades)}
                    )
                
                if np.any(~winning_trades):
                    avg_loss = np.mean(gross_returns[~winning_trades])
                    results['average_loss'] = MetricResult(
                        metric_name='average_loss',
                        metric_type=MetricType.AVERAGE_LOSS,
                        value=avg_loss,
                        sample_size=np.sum(~winning_trades),
                        metadata={'losing_trades_count': np.sum(~winning_trades)}
                    )
                
                # Sharpe ratio
                if len(gross_returns) > 1:
                    excess_returns = gross_returns - (self.config.risk_free_rate / 252)  # Daily risk-free rate
                    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252)
                    
                    results['sharpe_ratio'] = MetricResult(
                        metric_name='sharpe_ratio',
                        metric_type=MetricType.SHARPE_RATIO,
                        value=sharpe_ratio,
                        sample_size=len(excess_returns),
                        metadata={'risk_free_rate': self.config.risk_free_rate}
                    )
                
                # Maximum drawdown
                cumulative_returns = np.cumsum(gross_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = cumulative_returns - running_max
                max_drawdown = np.min(drawdown)
                
                results['max_drawdown'] = MetricResult(
                    metric_name='max_drawdown',
                    metric_type=MetricType.MAX_DRAWDOWN,
                    value=max_drawdown,
                    sample_size=len(drawdown),
                    metadata={'drawdown_series': drawdown.tolist()}
                )
                
                # Profit factor
                gross_profit = np.sum(gross_returns[gross_returns > 0]) if np.any(gross_returns > 0) else 0
                gross_loss = abs(np.sum(gross_returns[gross_returns < 0])) if np.any(gross_returns < 0) else 1e-8
                profit_factor = gross_profit / gross_loss
                
                results['profit_factor'] = MetricResult(
                    metric_name='profit_factor',
                    metric_type=MetricType.PROFIT_FACTOR,
                    value=profit_factor,
                    sample_size=len(gross_returns),
                    metadata={
                        'gross_profit': gross_profit,
                        'gross_loss': gross_loss
                    }
                )
                
                # Risk-adjusted return
                volatility = np.std(gross_returns) * np.sqrt(252)  # Annualized volatility
                risk_adjusted_return = (total_return * 252) / (volatility + 1e-8)  # Annualized return / volatility
                
                results['risk_adjusted_return'] = MetricResult(
                    metric_name='risk_adjusted_return',
                    metric_type=MetricType.RISK_ADJUSTED_RETURN,
                    value=risk_adjusted_return,
                    sample_size=len(gross_returns),
                    metadata={
                        'annualized_return': total_return * 252,
                        'annualized_volatility': volatility
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Business metrics calculation failed: {e}")
        
        # Save registry
        self.calculated_metrics.update(results)
        self.metric_history.extend(results.values())
        
        return results
    
    def _calculate_trading_metrics(
        self, 
        true_returns: np.ndarray, 
        pred_returns: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Helper method for trading-specific metrics"""
        trading_results = {}
        
        try:
            # Directional accuracy for returns
            true_direction = true_returns > 0
            pred_direction = pred_returns > 0
            
            if len(true_direction) > 0:
                directional_accuracy = np.mean(true_direction == pred_direction)
                trading_results['return_directional_accuracy'] = MetricResult(
                    metric_name='return_directional_accuracy',
                    metric_type=MetricType.DIRECTIONAL_ACCURACY,
                    value=directional_accuracy,
                    sample_size=len(true_direction)
                )
            
            # Return prediction accuracy (correlation)
            if len(true_returns) > 1:
                return_correlation = np.corrcoef(true_returns, pred_returns)[0, 1]
                if not np.isnan(return_correlation):
                    trading_results['return_correlation'] = MetricResult(
                        metric_name='return_correlation',
                        metric_type=MetricType.R_SQUARED,  # Use R_SQUARED as proxy
                        value=return_correlation,
                        sample_size=len(true_returns)
                    )
        
        except Exception as e:
            self.logger.debug(f"Trading metrics calculation failed: {e}")
        
        return trading_results
    
    def _bootstrap_confidence_interval(
        self,
        metric_func: Callable,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bootstrap: Optional[int] = None
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for metrics arrays"""
        n_bootstrap = n_bootstrap or self.config.bootstrap_samples
        
        try:
            bootstrap_scores = []
            n_samples = len(y_true)
            
            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred[indices]
                
                try:
                    score = metric_func(y_true_boot, y_pred_boot)
                    if not np.isnan(score) and not np.isinf(score):
                        bootstrap_scores.append(score)
                except:
                    continue
            
            if len(bootstrap_scores) > 10:  # Minimum samples for CI
                alpha = 1 - self.config.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = np.percentile(bootstrap_scores, lower_percentile)
                ci_upper = np.percentile(bootstrap_scores, upper_percentile)
                
                return (ci_lower, ci_upper)
        
        except Exception as e:
            self.logger.debug(f"Bootstrap CI calculation failed: {e}")
        
        return None
    
    def _bootstrap_confidence_interval_simple(
        self,
        metric_func: Callable,
        values: List[float],
        n_bootstrap: Optional[int] = None
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for simple metrics"""
        n_bootstrap = n_bootstrap or self.config.bootstrap_samples
        
        try:
            values_array = np.array(values)
            bootstrap_scores = []
            n_samples = len(values_array)
            
            for _ in range(n_bootstrap):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                values_boot = values_array[indices]
                
                try:
                    score = metric_func(values_boot)
                    if not np.isnan(score) and not np.isinf(score):
                        bootstrap_scores.append(score)
                except:
                    continue
            
            if len(bootstrap_scores) > 10:
                alpha = 1 - self.config.confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                ci_lower = np.percentile(bootstrap_scores, lower_percentile)
                ci_upper = np.percentile(bootstrap_scores, upper_percentile)
                
                return (ci_lower, ci_upper)
        
        except Exception as e:
            self.logger.debug(f"Bootstrap CI calculation failed: {e}")
        
        return None
    
    def _calculate_autocorrelation(self, series: np.ndarray, max_lags: int = 10) -> List[float]:
        """Calculation autocorrelation for residual analysis"""
        try:
            from statsmodels.tsa.stattools import acf
            autocorr = acf(series, nlags=max_lags, fft=False)
            return autocorr[1:].tolist()  # Exclude lag 0
        except ImportError:
            # Fallback manual calculation
            n = len(series)
            autocorr = []
            mean = np.mean(series)
            variance = np.var(series)
            
            for lag in range(1, min(max_lags + 1, n)):
                c = np.mean((series[:-lag] - mean) * (series[lag:] - mean))
                autocorr.append(c / variance if variance > 0 else 0)
            
            return autocorr
        except Exception as e:
            self.logger.debug(f"Autocorrelation calculation failed: {e}")
            return []
    
    def compare_metrics(
        self,
        metrics1: Dict[str, MetricResult],
        metrics2: Dict[str, MetricResult],
        comparison_name: str = "comparison"
    ) -> Dict[str, Any]:
        """
 Comparison metrics between models/experiments
        
        Args:
 metrics1: suite metrics
 metrics2: suite metrics
            comparison_name: Name comparison
        
        Returns:
            Dict[str, Any]: Result comparison
        """
        self.logger.info(f"Comparing metrics: {comparison_name}")
        
        comparison_result = {
            'comparison_name': comparison_name,
            'timestamp': datetime.now().isoformat(),
            'metrics_compared': 0,
            'significant_differences': [],
            'metric_comparisons': {},
            'overall_winner': None
        }
        
        # Find common metrics
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        comparison_result['metrics_compared'] = len(common_metrics)
        
        winner_scores = {'model1': 0, 'model2': 0}
        
        for metric_name in common_metrics:
            metric1 = metrics1[metric_name]
            metric2 = metrics2[metric_name]
            
            # Basic comparison
            value_diff = metric2.value - metric1.value
            relative_diff = value_diff / (abs(metric1.value) + 1e-8) * 100
            
            # Determine which is better (higher or lower)
            is_higher_better = metric1.metric_type in [
                MetricType.ACCURACY, MetricType.PRECISION, MetricType.RECALL, MetricType.F1_SCORE,
                MetricType.ROC_AUC, MetricType.PR_AUC, MetricType.R_SQUARED, MetricType.THROUGHPUT,
                MetricType.DIRECTIONAL_ACCURACY, MetricType.HIT_RATE, MetricType.SHARPE_RATIO,
                MetricType.PROFIT_FACTOR, MetricType.WIN_RATE, MetricType.RISK_ADJUSTED_RETURN
            ]
            
            if is_higher_better:
                winner = 'model2' if metric2.value > metric1.value else 'model1'
            else:
                winner = 'model2' if metric2.value < metric1.value else 'model1'
            
            winner_scores[winner] += 1
            
            # Statistical significance test (if is confidence intervals)
            is_significant = False
            p_value = None
            
            if (metric1.confidence_interval and metric2.confidence_interval and 
                self.config.enable_significance_testing and self.stats_available):
                
                # Simple overlap test for confidence intervals
                ci1_lower, ci1_upper = metric1.confidence_interval
                ci2_lower, ci2_upper = metric2.confidence_interval
                
                # No overlap = significant difference
                is_significant = (ci1_upper < ci2_lower) or (ci2_upper < ci1_lower)
                
                if is_significant:
                    comparison_result['significant_differences'].append({
                        'metric': metric_name,
                        'winner': winner,
                        'value_difference': value_diff,
                        'relative_difference_percent': relative_diff
                    })
            
            comparison_result['metric_comparisons'][metric_name] = {
                'model1_value': metric1.value,
                'model2_value': metric2.value,
                'difference': value_diff,
                'relative_difference_percent': relative_diff,
                'winner': winner,
                'is_significant': is_significant,
                'is_higher_better': is_higher_better,
                'model1_ci': metric1.confidence_interval,
                'model2_ci': metric2.confidence_interval
            }
        
        # Overall winner
        if winner_scores['model1'] > winner_scores['model2']:
            comparison_result['overall_winner'] = 'model1'
        elif winner_scores['model2'] > winner_scores['model1']:
            comparison_result['overall_winner'] = 'model2'
        else:
            comparison_result['overall_winner'] = 'tie'
        
        comparison_result['winner_scores'] = winner_scores
        
        return comparison_result
    
    def generate_metrics_report(self, include_history: bool = False) -> Dict[str, Any]:
        """Generation comprehensive metrics report"""
        current_metrics = self.calculated_metrics
        
        if not current_metrics:
            return {'status': 'no_metrics_calculated'}
        
        # Group metrics by category
        metrics_by_category = {}
        for metric_result in current_metrics.values():
            category = metric_result.metadata.get('metric_category', 'other')
            if category not in metrics_by_category:
                metrics_by_category[category] = {}
            metrics_by_category[category][metric_result.metric_name] = metric_result.to_dict()
        
        # Summary statistics
        total_metrics = len(current_metrics)
        metrics_with_ci = sum(1 for m in current_metrics.values() if m.confidence_interval is not None)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_metrics': total_metrics,
                'metrics_with_confidence_intervals': metrics_with_ci,
                'confidence_level': self.config.confidence_level,
                'bootstrap_samples': self.config.bootstrap_samples
            },
            'metrics_by_category': metrics_by_category,
            'configuration': {
                'bootstrap_samples': self.config.bootstrap_samples,
                'confidence_level': self.config.confidence_level,
                'significance_threshold': self.config.significance_threshold,
                'crypto_return_calculation': self.config.crypto_return_calculation,
                'trading_cost_bps': self.config.trading_cost_bps,
                'risk_free_rate': self.config.risk_free_rate
            }
        }
        
        if include_history:
            report['metrics_history'] = {
                'total_historical_calculations': len(self.metric_history),
                'history': [m.to_dict() for m in self.metric_history[-50:]]  # Last 50
            }
        
        return report
    
    def export_metrics(self, filepath: str, include_history: bool = False) -> None:
        """Export metrics report file"""
        report = self.generate_metrics_report(include_history)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Metrics report exported {filepath}")


def create_crypto_trading_metrics_calculator() -> MetricsCalculator:
    """
    Factory function for creation metrics calculator for crypto trading models
    Enterprise pre-configured calculator for financial ML systems
    """
    config = MetricsConfig(
        # Statistical settings for financial data
 bootstrap_samples=2000, # samples for accuracy
 confidence_level=0.99, # High confidence level for
        
        # Significance testing
        enable_significance_testing=True,
 significance_threshold=0.01, # 1% threshold for
        
        # Time series settings for crypto data
        time_series_frequency="1min",
        enable_autocorrelation_adjustment=True,
        
        # Business metrics for crypto trading
        risk_free_rate=0.02,  # 2% annual
        trading_cost_bps=10.0,  # 10 bps for crypto (higher than traditional)
        
        # Performance settings
        latency_unit="ms",
        memory_unit="MB",
        
        # Stability settings
 stability_window_size=200, # window for crypto volatility
        consistency_threshold=0.05,  # 5% variation threshold
        
        # Advanced analysis for enterprise
        enable_detailed_analysis=True,
        save_intermediate_results=True,
        generate_visualizations=False,  # Disable for performance
        
        # Crypto specific settings
        crypto_return_calculation="log",  # Log returns for crypto
        enable_volatility_adjustment=True,
        slippage_bps=5.0  # 5 bps slippage for crypto markets
    )
    
    return MetricsCalculator(config)