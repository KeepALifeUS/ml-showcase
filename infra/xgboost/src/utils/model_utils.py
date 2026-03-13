"""
Model Utilities for XGBoost
==========================

Utility functions for model saving, loading, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger


class ModelSaver:
    """Utility for saving XGBoost models"""
    
    def __init__(self):
        pass
    
    def save_model(
        self, 
        model: Any, 
        path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save model with metadata"""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, path / "model.pkl")
        
        # Save metadata
        if metadata:
            with open(path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {path}")


class ModelLoader:
    """Utility for loading XGBoost models"""
    
    def __init__(self):
        pass
    
    def load_model(self, path: str) -> Any:
        """Load model from path"""
        
        model_path = Path(path) / "model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {path}")
        
        return model
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """Load model metadata"""
        
        metadata_path = Path(path) / "metadata.json"
        
        if not metadata_path.exists():
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata


class ModelMetrics:
    """Model performance metrics calculator"""
    
    def __init__(self):
        pass
    
    def calculate_regression_metrics(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics"""
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        }
        
        return metrics
    
    def calculate_directional_accuracy(
        self, 
        y_true: pd.Series, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy for trading"""
        
        true_direction = np.sign(y_true.diff().fillna(0))
        pred_direction = np.sign(pd.Series(y_pred).diff().fillna(0))
        
        accuracy = (true_direction == pred_direction).mean()
        return accuracy


class PerformanceAnalyzer:
    """Comprehensive performance analysis"""
    
    def __init__(self):
        pass
    
    def analyze_predictions(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive prediction analysis"""
        
        metrics_calc = ModelMetrics()
        
        # Basic metrics
        regression_metrics = metrics_calc.calculate_regression_metrics(y_true, y_pred)
        directional_accuracy = metrics_calc.calculate_directional_accuracy(y_true, y_pred)
        
        # Residual analysis
        residuals = y_true - y_pred
        
        residual_stats = {
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'residual_skewness': residuals.skew(),
            'residual_kurtosis': residuals.kurtosis(),
        }
        
        # Prediction intervals
        pred_intervals = {
            'q05': np.percentile(y_pred, 5),
            'q25': np.percentile(y_pred, 25),
            'q50': np.percentile(y_pred, 50),
            'q75': np.percentile(y_pred, 75),
            'q95': np.percentile(y_pred, 95),
        }
        
        analysis = {
            'regression_metrics': regression_metrics,
            'directional_accuracy': directional_accuracy,
            'residual_stats': residual_stats,
            'prediction_intervals': pred_intervals,
            'sample_size': len(y_true),
        }
        
        return analysis
    
    def generate_report(
        self,
        analysis: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """Generate performance report"""
        
        report = f"""
XGBoost Model Performance Report
==============================

Sample Size: {analysis['sample_size']}

Regression Metrics:
------------------
RMSE: {analysis['regression_metrics']['rmse']:.4f}
MAE: {analysis['regression_metrics']['mae']:.4f}
R²: {analysis['regression_metrics']['r2']:.4f}
MAPE: {analysis['regression_metrics']['mape']:.2f}%

Trading Metrics:
---------------
Directional Accuracy: {analysis['directional_accuracy']:.2%}

Residual Analysis:
-----------------
Mean: {analysis['residual_stats']['residual_mean']:.4f}
Std: {analysis['residual_stats']['residual_std']:.4f}
Skewness: {analysis['residual_stats']['residual_skewness']:.4f}
Kurtosis: {analysis['residual_stats']['residual_kurtosis']:.4f}

Prediction Intervals:
--------------------
5%: {analysis['prediction_intervals']['q05']:.4f}
25%: {analysis['prediction_intervals']['q25']:.4f}
50%: {analysis['prediction_intervals']['q50']:.4f}
75%: {analysis['prediction_intervals']['q75']:.4f}
95%: {analysis['prediction_intervals']['q95']:.4f}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {save_path}")
        
        return report