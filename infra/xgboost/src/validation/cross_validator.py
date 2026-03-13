"""
Time Series Cross-Validation Framework
======================================

Cross-validation optimized for time series and time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger


class TimeSeriesCrossValidator:
    """Time series cross-validator with walk-forward analysis"""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.cv_results_: List[Dict[str, Any]] = []
    
    def cross_validate(
        self, 
        model: Any, 
        X: pd.DataFrame, 
        y: pd.Series,
        scoring: str = "neg_mean_squared_error"
    ) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate score
            if scoring == "neg_mean_squared_error":
                score = -mean_squared_error(y_val, y_pred)
            elif scoring == "r2":
                score = r2_score(y_val, y_pred)
            else:
                score = -mean_squared_error(y_val, y_pred)  # Default
            
            scores.append(score)
            
            # Store fold results
            fold_result = {
                'fold': fold,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'score': score,
                'train_period': (X_train.index.min(), X_train.index.max()),
                'val_period': (X_val.index.min(), X_val.index.max()),
            }
            
            self.cv_results_.append(fold_result)
        
        return {
            'scores': scores,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'cv_results': self.cv_results_
        }
    
    def get_fold_results(self) -> List[Dict[str, Any]]:
        """Get detailed results for each fold"""
        return self.cv_results_


class WalkForwardValidator:
    """Walk-forward validation for time series models"""
    
    def __init__(self, initial_train_size: int, step_size: int = 1):
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.validation_results_: List[Dict[str, Any]] = []
    
    def validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """Perform walk-forward validation"""
        
        predictions = []
        actual_values = []
        
        for i in range(self.initial_train_size, len(X), self.step_size):
            # Define training and test sets
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i+self.step_size]
            y_test = y.iloc[i:i+self.step_size]
            
            if len(X_test) == 0:
                break
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Store results
            predictions.extend(y_pred)
            actual_values.extend(y_test.values)
            
            # Store step results
            step_result = {
                'step': i,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred) if len(y_test) > 1 else 0,
            }
            
            self.validation_results_.append(step_result)
        
        # Overall metrics
        overall_rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        overall_r2 = r2_score(actual_values, predictions)
        
        return {
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'predictions': predictions,
            'actual_values': actual_values,
            'step_results': self.validation_results_
        }


class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self):
        self.validation_results_: Dict[str, Any] = {}
    
    def validate_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """Comprehensive model validation"""
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Training metrics
        train_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'mae': np.mean(np.abs(y_train - y_train_pred)),
            'r2': r2_score(y_train, y_train_pred),
        }
        
        # Test metrics
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': np.mean(np.abs(y_test - y_test_pred)),
            'r2': r2_score(y_test, y_test_pred),
        }
        
        # Directional accuracy
        train_direction_acc = self._calculate_directional_accuracy(y_train, y_train_pred)
        test_direction_acc = self._calculate_directional_accuracy(y_test, y_test_pred)
        
        results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_directional_accuracy': train_direction_acc,
            'test_directional_accuracy': test_direction_acc,
            'overfitting_ratio': test_metrics['rmse'] / train_metrics['rmse'],
        }
        
        self.validation_results_ = results
        return results
    
    def _calculate_directional_accuracy(
        self, 
        y_actual: pd.Series, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate directional accuracy"""
        
        actual_direction = np.sign(y_actual.diff().fillna(0))
        pred_direction = np.sign(pd.Series(y_pred).diff().fillna(0))
        
        correct_directions = (actual_direction == pred_direction).sum()
        total_directions = len(actual_direction)
        
        return correct_directions / total_directions if total_directions > 0 else 0


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.backtest_results_: Dict[str, Any] = {}
    
    def backtest(
        self,
        predictions: pd.Series,
        actual_prices: pd.Series,
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """Simple backtest based on predictions"""
        
        # Generate signals (1 for buy, -1 for sell, 0 for hold)
        signals = np.where(predictions.diff() > 0, 1,
                          np.where(predictions.diff() < 0, -1, 0))
        
        # Calculate returns
        price_returns = actual_prices.pct_change().fillna(0)
        
        # Strategy returns (simplified)
        strategy_returns = []
        position = 0
        
        for i, signal in enumerate(signals):
            if signal != 0:
                # Transaction cost
                ret = price_returns.iloc[i] * position - abs(signal) * transaction_cost
                position = signal
            else:
                ret = price_returns.iloc[i] * position
            
            strategy_returns.append(ret)
        
        strategy_returns = pd.Series(strategy_returns, index=actual_prices.index)
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (strategy_returns > 0).mean(),
            'strategy_returns': strategy_returns,
        }
        
        self.backtest_results_ = results
        return results