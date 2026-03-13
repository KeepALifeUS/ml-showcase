"""
Metrics calculation utilities for Prophet forecasting evaluation.

Comprehensive metrics suite for evaluating forecast accuracy, model performance,
and business impact following enterprise patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

from .logger import get_logger
from .exceptions import ValidationException

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types metrics for forecasting"""
    ACCURACY = "accuracy"  # Metrics accuracy
    DIRECTIONAL = "directional"  # Directionality forecast
    BUSINESS = "business"  # Business-metrics
    STATISTICAL = "statistical"  # Statistical metrics
    RISK = "risk"  # Metrics risk


@dataclass
class MetricResult:
    """
    Result computations metrics
    """
    name: str
    value: float
    metric_type: MetricType
    description: str
    is_percentage: bool = False
    higher_is_better: bool = False
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    
    @property
    def status(self) -> str:
        """Status metrics on basis thresholds"""
        if self.threshold_critical is not None:
            if (self.higher_is_better and self.value < self.threshold_critical) or \
               (not self.higher_is_better and self.value > self.threshold_critical):
                return "critical"
        
        if self.threshold_warning is not None:
            if (self.higher_is_better and self.value < self.threshold_warning) or \
               (not self.higher_is_better and self.value > self.threshold_warning):
                return "warning"
        
        return "good"
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary for serialization"""
        return {
            "name": self.name,
            "value": self.value,
            "metric_type": self.metric_type.value,
            "description": self.description,
            "is_percentage": self.is_percentage,
            "higher_is_better": self.higher_is_better,
            "status": self.status,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical
        }


class ForecastMetrics:
    """
    Comprehensive class for computations metrics forecasting
    
    Supports various types metrics:
    - Accuracy forecasts (MAE, RMSE, MAPE)
    - Directionality (Directional Accuracy)
    - Business-metrics (ROI, Sharpe Ratio)
    - Statistical tests
    - Metrics risk
    """
    
    def __init__(self, symbol: str = "unknown", timeframe: str = "1h"):
        """
        Initialization calculator metrics
        
        Args:
            symbol: Symbol cryptocurrency
            timeframe: Timeframe data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.logger = get_logger(f"{__name__}.{symbol}.{timeframe}")
        
        # Thresholds for metrics (possible configure)
        self.thresholds = {
            "mape": {"warning": 10.0, "critical": 20.0},  # %
            "mae": {"warning": None, "critical": None},    # Depends from price
            "rmse": {"warning": None, "critical": None},   # Depends from price
            "directional_accuracy": {"warning": 55.0, "critical": 50.0},  # %
            "hit_rate": {"warning": 60.0, "critical": 50.0},  # %
        }
    
    def calculate_all_metrics(
        self,
        y_true: Union[pd.Series, np.ndarray, List[float]],
        y_pred: Union[pd.Series, np.ndarray, List[float]],
        timestamps: Optional[Union[pd.Series, List]] = None,
        confidence_lower: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        confidence_upper: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
        prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None
    ) -> Dict[str, MetricResult]:
        """
        Computation all available metrics
        
        Args:
            y_true: Actual values
            y_pred: Forecast values
            timestamps: Temporal labels (optionally)
            confidence_lower: Lower boundary confidence interval
            confidence_upper: Upper boundary confidence interval
            prices: Price for business-metrics (if differ from y_true)
            
        Returns:
            Dictionary with results all metrics
        """
        try:
            # Transformation in numpy arrays
            y_true = self._to_numpy(y_true)
            y_pred = self._to_numpy(y_pred)
            
            # Validation input data
            self._validate_inputs(y_true, y_pred)
            
            metrics = {}
            
            # Metrics accuracy
            accuracy_metrics = self._calculate_accuracy_metrics(y_true, y_pred)
            metrics.update(accuracy_metrics)
            
            # Directed metrics
            if len(y_true) > 1:
                directional_metrics = self._calculate_directional_metrics(y_true, y_pred)
                metrics.update(directional_metrics)
            
            # Metrics confidence intervals
            if confidence_lower is not None and confidence_upper is not None:
                confidence_lower = self._to_numpy(confidence_lower)
                confidence_upper = self._to_numpy(confidence_upper)
                interval_metrics = self._calculate_interval_metrics(
                    y_true, confidence_lower, confidence_upper
                )
                metrics.update(interval_metrics)
            
            # Statistical metrics
            statistical_metrics = self._calculate_statistical_metrics(y_true, y_pred)
            metrics.update(statistical_metrics)
            
            # Business-metrics (if exists data about prices)
            if prices is not None or len(y_true) > 10:
                business_prices = self._to_numpy(prices) if prices is not None else y_true
                business_metrics = self._calculate_business_metrics(
                    y_true, y_pred, business_prices
                )
                metrics.update(business_metrics)
            
            # Metrics risk
            risk_metrics = self._calculate_risk_metrics(y_true, y_pred)
            metrics.update(risk_metrics)
            
            self.logger.info(
                f"Calculated {len(metrics)} metrics",
                extra={
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "samples": len(y_true),
                    "metrics_count": len(metrics)
                }
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            raise ValidationException(f"Failed to calculate metrics: {e}")
    
    def _to_numpy(self, data: Union[pd.Series, np.ndarray, List[float]]) -> np.ndarray:
        """Transformation data in numpy array"""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Validation input data"""
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("Empty input arrays")
        
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
            self.logger.warning("NaN values detected in input data")
            # Removal NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if not np.any(mask):
                raise ValueError("All values are NaN")
    
    def _calculate_accuracy_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation metrics accuracy"""
        metrics = {}
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        metrics["mae"] = MetricResult(
            name="Mean Absolute Error",
            value=mae,
            metric_type=MetricType.ACCURACY,
            description="Average absolute difference between predicted and actual values",
            higher_is_better=False,
            threshold_warning=self.thresholds["mae"]["warning"],
            threshold_critical=self.thresholds["mae"]["critical"]
        )
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        metrics["rmse"] = MetricResult(
            name="Root Mean Square Error",
            value=rmse,
            metric_type=MetricType.ACCURACY,
            description="Square root of average squared differences",
            higher_is_better=False,
            threshold_warning=self.thresholds["rmse"]["warning"],
            threshold_critical=self.thresholds["rmse"]["critical"]
        )
        
        # Mean Absolute Percentage Error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            if np.isfinite(mape):
                metrics["mape"] = MetricResult(
                    name="Mean Absolute Percentage Error",
                    value=mape,
                    metric_type=MetricType.ACCURACY,
                    description="Average absolute percentage difference",
                    is_percentage=True,
                    higher_is_better=False,
                    threshold_warning=self.thresholds["mape"]["warning"],
                    threshold_critical=self.thresholds["mape"]["critical"]
                )
        
        # Symmetric Mean Absolute Percentage Error
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
            if np.isfinite(smape):
                metrics["smape"] = MetricResult(
                    name="Symmetric Mean Absolute Percentage Error",
                    value=smape,
                    metric_type=MetricType.ACCURACY,
                    description="Symmetric version of MAPE",
                    is_percentage=True,
                    higher_is_better=False
                )
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["mse"] = MetricResult(
            name="Mean Squared Error",
            value=mse,
            metric_type=MetricType.ACCURACY,
            description="Average of squared differences",
            higher_is_better=False
        )
        
        # R² Score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot != 0:
            r2 = 1 - (ss_res / ss_tot)
            metrics["r2_score"] = MetricResult(
                name="R² Coefficient of Determination",
                value=r2,
                metric_type=MetricType.STATISTICAL,
                description="Proportion of variance explained by the model",
                higher_is_better=True
            )
        
        return metrics
    
    def _calculate_directional_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation directed metrics"""
        metrics = {}
        
        # Directed accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        
        if len(true_direction) > 0:
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
            metrics["directional_accuracy"] = MetricResult(
                name="Directional Accuracy",
                value=directional_accuracy,
                metric_type=MetricType.DIRECTIONAL,
                description="Percentage of correctly predicted price directions",
                is_percentage=True,
                higher_is_better=True,
                threshold_warning=self.thresholds["directional_accuracy"]["warning"],
                threshold_critical=self.thresholds["directional_accuracy"]["critical"]
            )
            
            # Hit Rate for positive movements
            if np.any(true_direction):
                hit_rate = np.mean(pred_direction[true_direction]) * 100
                metrics["hit_rate_positive"] = MetricResult(
                    name="Hit Rate (Positive Moves)",
                    value=hit_rate,
                    metric_type=MetricType.DIRECTIONAL,
                    description="Accuracy of predicting positive price movements",
                    is_percentage=True,
                    higher_is_better=True
                )
            
            # Hit Rate for negative movements
            if np.any(~true_direction):
                hit_rate_neg = np.mean(~pred_direction[~true_direction]) * 100
                metrics["hit_rate_negative"] = MetricResult(
                    name="Hit Rate (Negative Moves)",
                    value=hit_rate_neg,
                    metric_type=MetricType.DIRECTIONAL,
                    description="Accuracy of predicting negative price movements",
                    is_percentage=True,
                    higher_is_better=True
                )
        
        return metrics
    
    def _calculate_interval_metrics(
        self,
        y_true: np.ndarray,
        confidence_lower: np.ndarray,
        confidence_upper: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation metrics confidence intervals"""
        metrics = {}
        
        # Coverage confidence interval
        in_interval = (y_true >= confidence_lower) & (y_true <= confidence_upper)
        coverage = np.mean(in_interval) * 100
        
        metrics["interval_coverage"] = MetricResult(
            name="Confidence Interval Coverage",
            value=coverage,
            metric_type=MetricType.STATISTICAL,
            description="Percentage of actual values within confidence interval",
            is_percentage=True,
            higher_is_better=True,
            threshold_warning=75.0,
            threshold_critical=60.0
        )
        
        # Average width interval
        interval_width = np.mean(confidence_upper - confidence_lower)
        metrics["mean_interval_width"] = MetricResult(
            name="Mean Interval Width",
            value=interval_width,
            metric_type=MetricType.STATISTICAL,
            description="Average width of confidence intervals",
            higher_is_better=False
        )
        
        # Relative width interval
        relative_width = np.mean((confidence_upper - confidence_lower) / np.abs(y_true)) * 100
        if np.isfinite(relative_width):
            metrics["relative_interval_width"] = MetricResult(
                name="Relative Interval Width",
                value=relative_width,
                metric_type=MetricType.STATISTICAL,
                description="Average interval width relative to actual values",
                is_percentage=True,
                higher_is_better=False
            )
        
        return metrics
    
    def _calculate_statistical_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation statistical metrics"""
        metrics = {}
        
        # Correlation Pearson
        if len(y_true) > 1:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            if np.isfinite(correlation):
                metrics["correlation"] = MetricResult(
                    name="Pearson Correlation",
                    value=correlation,
                    metric_type=MetricType.STATISTICAL,
                    description="Linear correlation between actual and predicted values",
                    higher_is_better=True,
                    threshold_warning=0.7,
                    threshold_critical=0.5
                )
        
        # Average deviation (bias)
        bias = np.mean(y_pred - y_true)
        metrics["bias"] = MetricResult(
            name="Prediction Bias",
            value=bias,
            metric_type=MetricType.STATISTICAL,
            description="Average difference (predicted - actual), positive means overestimation",
            higher_is_better=False  # Close to 0 better
        )
        
        # Standard deviation errors
        residuals = y_pred - y_true
        residuals_std = np.std(residuals)
        metrics["residuals_std"] = MetricResult(
            name="Residuals Standard Deviation",
            value=residuals_std,
            metric_type=MetricType.STATISTICAL,
            description="Standard deviation of prediction errors",
            higher_is_better=False
        )
        
        return metrics
    
    def _calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prices: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation business-metrics"""
        metrics = {}
        
        try:
            # Profitability when following forecasts
            if len(prices) > 1:
                price_returns = np.diff(prices) / prices[:-1]
                pred_directions = np.diff(y_pred) > 0
                
                # Strategy: purchase when forecast growth, sale when forecast drops
                strategy_returns = np.where(pred_directions, price_returns, -price_returns)
                
                # Aggregate profitability
                cumulative_return = (1 + strategy_returns).prod() - 1
                metrics["strategy_return"] = MetricResult(
                    name="Strategy Cumulative Return",
                    value=cumulative_return * 100,
                    metric_type=MetricType.BUSINESS,
                    description="Total return from following forecast signals",
                    is_percentage=True,
                    higher_is_better=True
                )
                
                # Coefficient Sharpe (if sufficient data)
                if len(strategy_returns) > 20:
                    if np.std(strategy_returns) > 0:
                        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                        metrics["sharpe_ratio"] = MetricResult(
                            name="Sharpe Ratio",
                            value=sharpe_ratio,
                            metric_type=MetricType.BUSINESS,
                            description="Risk-adjusted return metric",
                            higher_is_better=True,
                            threshold_warning=1.0,
                            threshold_critical=0.5
                        )
                
                # Maximum drawdown
                cumulative_returns = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdowns) * 100
                
                metrics["max_drawdown"] = MetricResult(
                    name="Maximum Drawdown",
                    value=abs(max_drawdown),
                    metric_type=MetricType.RISK,
                    description="Largest peak-to-trough decline",
                    is_percentage=True,
                    higher_is_better=False,
                    threshold_warning=10.0,
                    threshold_critical=20.0
                )
        
        except Exception as e:
            self.logger.warning(f"Could not calculate business metrics: {e}")
        
        return metrics
    
    def _calculate_risk_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, MetricResult]:
        """Computation metrics risk"""
        metrics = {}
        
        try:
            # Value at Risk (VaR) errors forecast
            errors = y_pred - y_true
            var_95 = np.percentile(np.abs(errors), 95)
            metrics["var_95"] = MetricResult(
                name="Value at Risk 95%",
                value=var_95,
                metric_type=MetricType.RISK,
                description="95th percentile of absolute prediction errors",
                higher_is_better=False
            )
            
            # Maximum error
            max_error = np.max(np.abs(errors))
            metrics["max_absolute_error"] = MetricResult(
                name="Maximum Absolute Error",
                value=max_error,
                metric_type=MetricType.RISK,
                description="Largest absolute prediction error",
                higher_is_better=False
            )
            
            # Asymmetry errors (tendency to re-/underestimation)
            if len(errors) > 3:
                from scipy.stats import skew
                error_skewness = skew(errors)
                metrics["error_skewness"] = MetricResult(
                    name="Error Skewness",
                    value=error_skewness,
                    metric_type=MetricType.STATISTICAL,
                    description="Asymmetry of prediction errors (positive = overestimation bias)",
                    higher_is_better=False  # Close to 0 better
                )
        
        except Exception as e:
            self.logger.warning(f"Could not calculate all risk metrics: {e}")
        
        return metrics
    
    def get_summary_report(
        self, 
        metrics: Dict[str, MetricResult]
    ) -> Dict[str, Any]:
        """
        Creation summary report by metrics
        
        Args:
            metrics: Dictionary with computed metrics
            
        Returns:
            Summary report with analysis
        """
        # Grouping metrics by types
        metrics_by_type = {}
        for metric in metrics.values():
            metric_type = metric.metric_type.value
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(metric)
        
        # Analysis status
        status_counts = {"good": 0, "warning": 0, "critical": 0}
        for metric in metrics.values():
            status_counts[metric.status] += 1
        
        # Key metrics for summary
        key_metrics = {}
        for key in ["mape", "directional_accuracy", "correlation", "sharpe_ratio"]:
            if key in metrics:
                key_metrics[key] = metrics[key].value
        
        # Total status model
        if status_counts["critical"] > 0:
            overall_status = "critical"
        elif status_counts["warning"] > 0:
            overall_status = "warning"
        else:
            overall_status = "good"
        
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "overall_status": overall_status,
            "total_metrics": len(metrics),
            "status_counts": status_counts,
            "key_metrics": key_metrics,
            "metrics_by_type": {
                metric_type: len(metric_list) 
                for metric_type, metric_list in metrics_by_type.items()
            },
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(
        self, 
        metrics: Dict[str, MetricResult]
    ) -> List[str]:
        """Generation recommendations on basis metrics"""
        recommendations = []
        
        # Validation accuracy
        if "mape" in metrics and metrics["mape"].status in ["warning", "critical"]:
            recommendations.append(
                f"MAPE {metrics['mape'].value:.2f}% indicates low accuracy. "
                "Consider tuning hyperparameters or adding more features."
            )
        
        # Validation directionality
        if "directional_accuracy" in metrics:
            da = metrics["directional_accuracy"]
            if da.status == "critical":
                recommendations.append(
                    f"Directional accuracy {da.value:.1f}% is poor. "
                    "Model struggles with trend prediction."
                )
            elif da.value > 70:
                recommendations.append(
                    f"Excellent directional accuracy {da.value:.1f}%. "
                    "Model captures trends well."
                )
        
        # Validation business-metrics
        if "sharpe_ratio" in metrics:
            sr = metrics["sharpe_ratio"]
            if sr.value > 1.5:
                recommendations.append(
                    f"Excellent risk-adjusted returns (Sharpe: {sr.value:.2f}). "
                    "Model shows strong trading potential."
                )
            elif sr.status in ["warning", "critical"]:
                recommendations.append(
                    f"Low Sharpe ratio {sr.value:.2f}. "
                    "Risk-adjusted returns are poor."
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Metrics look good. Continue monitoring model performance.")
        
        return recommendations


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]],
    symbol: str = "unknown",
    timeframe: str = "1h",
    **kwargs
) -> Dict[str, MetricResult]:
    """
    Convenient function for fast computations metrics
    
    Args:
        y_true: Actual values
        y_pred: Forecast values 
        symbol: Symbol cryptocurrency
        timeframe: Timeframe
        **kwargs: Additional parameters for ForecastMetrics
        
    Returns:
        Dictionary with metrics
    """
    calculator = ForecastMetrics(symbol=symbol, timeframe=timeframe)
    return calculator.calculate_all_metrics(y_true, y_pred, **kwargs)