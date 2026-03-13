"""
Model Validator - Enterprise Enterprise ML Testing Framework
Comprehensive validation framework for ML models in crypto trading systems

Applies enterprise principles:
- Enterprise model governance
- Production-ready validation
- Multi-framework compatibility
- Risk-aware testing
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, Tuple, Type
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Levels severity validation for Enterprise compliance"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelFramework(Enum):
    """ML frameworks for Enterprise compatibility"""
    SCIKIT_LEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Result validation models - Enterprise structured output"""
    test_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ModelInterface(Protocol):
    """Enterprise model interface protocol for type safety"""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """method prediction for all models"""
        ...
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ModelInterface':
        """method training"""
        ...


class ValidationConfig(BaseModel):
    """Configuration validation - Enterprise typed configuration"""
    
    # Input/Output shape validation
    check_input_shape: bool = Field(default=True, description="Validation input data")
    check_output_shape: bool = Field(default=True, description="Validation output data")
    expected_input_features: Optional[int] = Field(default=None, description="number features")
    expected_output_shape: Optional[Tuple[int, ...]] = Field(default=None, description="")
    
    # Prediction validation
    check_prediction_range: bool = Field(default=True, description="Verification range predictions")
    min_prediction_value: Optional[float] = Field(default=None, description="value predictions")
    max_prediction_value: Optional[float] = Field(default=None, description="Maximum value predictions")
    
    # Model determinism
    check_determinism: bool = Field(default=True, description="Verification models")
    determinism_tolerance: float = Field(default=1e-10, description="Acceptable for")
    
    # Performance thresholds
    min_accuracy: Optional[float] = Field(default=None, description="Minimum accuracy models")
    max_training_time: Optional[float] = Field(default=None, description="Maximum training ()")
    max_inference_time: Optional[float] = Field(default=None, description="Maximum ()")
    
    # Overfitting detection
    check_overfitting: bool = Field(default=True, description="Verification on overfitting")
    overfitting_threshold: float = Field(default=0.1, description="for detection overfitting")
    
    # Memory usage
    max_memory_usage: Optional[float] = Field(default=None, description="Maximum usage memory (MB)")
    
    # Enterprise enterprise settings
    enable_strict_mode: bool = Field(default=False, description="Strict Enterprise")
    production_ready_check: bool = Field(default=True, description="Verification production")
    security_validation: bool = Field(default=True, description="Validation models")
    
    @validator('overfitting_threshold')
    def validate_overfitting_threshold(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError('overfitting_threshold should be between 0 1')
        return v


class ModelValidator:
    """
    Enterprise Enterprise Model Validator
    
 Comprehensive validation framework for ML models crypto trading system.
 temporal Enterprise patterns for production-ready quality.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
 Initialization Enterprise configuration
        
        Args:
            config: Configuration validation (Enterprise typed)
        """
        self.config = config or ValidationConfig()
        self.results: List[ValidationResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enterprise enterprise metrics
        self._validation_start_time: Optional[datetime] = None
        self._model_framework: Optional[ModelFramework] = None
        self._model_metadata: Dict[str, Any] = {}
    
    def validate_model(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> List[ValidationResult]:
        """
 validation models - Enterprise enterprise validation pipeline
        
        Args:
            model: ML model for validation
 X_train: data (features)
 y_train: data (targets)
            X_test: Test data (features, optional)
            y_test: Test data (targets, optional)
        
        Returns:
            List[ValidationResult]: Results all checks
        """
        self._validation_start_time = datetime.now()
        self.results.clear()
        
        self.logger.info("Starting Enterprise enterprise model validation")
        
        # Detection framework models
        self._detect_model_framework(model)
        
        # Core validation pipeline
        self._validate_model_interface(model)
        self._validate_input_output_shapes(model, X_train, y_train)
        self._validate_prediction_range(model, X_train)
        self._validate_determinism(model, X_train)
        
        # Performance validation (if is test data)
        if X_test is not None and y_test is not None:
            self._validate_performance(model, X_train, y_train, X_test, y_test)
            self._validate_overfitting(model, X_train, y_train, X_test, y_test)
        
        # Enterprise enterprise validations
        if self.config.production_ready_check:
            self._validate_production_readiness(model)
        
        if self.config.security_validation:
            self._validate_security(model)
        
        # Memory usage validation
        if self.config.max_memory_usage:
            self._validate_memory_usage(model, X_train)
        
        validation_time = (datetime.now() - self._validation_start_time).total_seconds()
        self.logger.info(f"Enterprise validation completed in {validation_time:.2f} seconds")
        
        return self.results
    
    def _detect_model_framework(self, model: Any) -> None:
        """Detection ML framework models for Enterprise compatibility"""
        model_class = type(model).__name__
        module_name = type(model).__module__
        
        if hasattr(model, 'predict') and 'sklearn' in module_name:
            self._model_framework = ModelFramework.SCIKIT_LEARN
        elif hasattr(model, 'forward') and 'torch' in module_name:
            self._model_framework = ModelFramework.PYTORCH
        elif hasattr(model, 'call') and 'tensorflow' in module_name:
            self._model_framework = ModelFramework.TENSORFLOW
        elif 'xgb' in module_name or 'xgboost' in module_name:
            self._model_framework = ModelFramework.XGBOOST
        elif 'lightgbm' in module_name or 'lgb' in module_name:
            self._model_framework = ModelFramework.LIGHTGBM
        elif 'catboost' in module_name:
            self._model_framework = ModelFramework.CATBOOST
        else:
            self._model_framework = ModelFramework.CUSTOM
        
        self._model_metadata['framework'] = self._model_framework.value
        self._model_metadata['class_name'] = model_class
        self.logger.info(f"Detected framework: {self._model_framework.value}")
    
    def _validate_model_interface(self, model: Any) -> None:
        """Validation Enterprise model interface"""
        required_methods = ['predict']
        missing_methods = []
        
        for method in required_methods:
            if not hasattr(model, method):
                missing_methods.append(method)
        
        if missing_methods:
            self.results.append(ValidationResult(
                test_name="model_interface",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Model not method: {missing_methods}",
                details={'missing_methods': missing_methods}
            ))
        else:
            self.results.append(ValidationResult(
                test_name="model_interface",
                passed=True,
                severity=ValidationSeverity.LOW,
                message="Enterprise model interface"
            ))
    
    def _validate_input_output_shapes(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> None:
        """Validation input output data"""
        try:
            # Verification input data
            if self.config.check_input_shape and self.config.expected_input_features:
                if X_train.shape[1] != self.config.expected_input_features:
                    self.results.append(ValidationResult(
                        test_name="input_shape",
                        passed=False,
                        severity=ValidationSeverity.HIGH,
                        message=f"number features: {X_train.shape[1]}, : {self.config.expected_input_features}",
                        details={'actual_features': X_train.shape[1], 'expected_features': self.config.expected_input_features}
                    ))
                    return
            
            # Verification output data through predict
            sample_prediction = model.predict(X_train[:1])
            
            if self.config.check_output_shape and self.config.expected_output_shape:
                if sample_prediction.shape != self.config.expected_output_shape:
                    self.results.append(ValidationResult(
                        test_name="output_shape",
                        passed=False,
                        severity=ValidationSeverity.HIGH,
                        message=f"output data: {sample_prediction.shape}, : {self.config.expected_output_shape}",
                        details={'actual_shape': sample_prediction.shape, 'expected_shape': self.config.expected_output_shape}
                    ))
                    return
            
            self.results.append(ValidationResult(
                test_name="input_output_shapes",
                passed=True,
                severity=ValidationSeverity.LOW,
                message="input output data",
                details={'input_shape': X_train.shape, 'output_shape': sample_prediction.shape}
            ))
            
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="input_output_shapes",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Error at validation data: {str(e)}",
                details={'error': str(e)}
            ))
    
    def _validate_prediction_range(self, model: Any, X_train: np.ndarray) -> None:
        """Validation range predictions models"""
        if not self.config.check_prediction_range:
            return
        
        try:
            predictions = model.predict(X_train)
            
            # Verification on NaN infinity
            if np.any(np.isnan(predictions)):
                self.results.append(ValidationResult(
                    test_name="prediction_range",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Model NaN values",
                    details={'nan_count': np.sum(np.isnan(predictions))}
                ))
                return
            
            if np.any(np.isinf(predictions)):
                self.results.append(ValidationResult(
                    test_name="prediction_range",
                    passed=False,
                    severity=ValidationSeverity.CRITICAL,
                    message="Model values",
                    details={'inf_count': np.sum(np.isinf(predictions))}
                ))
                return
            
            # Verification range values
            min_pred, max_pred = predictions.min(), predictions.max()
            
            range_violations = []
            if self.config.min_prediction_value is not None and min_pred < self.config.min_prediction_value:
                range_violations.append(f"{min_pred} {self.config.min_prediction_value}")
            
            if self.config.max_prediction_value is not None and max_pred > self.config.max_prediction_value:
                range_violations.append(f"Maximum {max_pred} {self.config.max_prediction_value}")
            
            if range_violations:
                self.results.append(ValidationResult(
                    test_name="prediction_range",
                    passed=False,
                    severity=ValidationSeverity.HIGH,
 message=f" in acceptable range: {'; '.join(range_violations)}",
                    details={'min_prediction': min_pred, 'max_prediction': max_pred, 'violations': range_violations}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="prediction_range",
                    passed=True,
                    severity=ValidationSeverity.LOW,
                    message="predictions",
                    details={'min_prediction': min_pred, 'max_prediction': max_pred}
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="prediction_range",
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Error at validation range predictions: {str(e)}",
                details={'error': str(e)}
            ))
    
    def _validate_determinism(self, model: Any, X_train: np.ndarray) -> None:
        """Validation models - Enterprise consistency check"""
        if not self.config.check_determinism:
            return
        
        try:
            # predictions on data
 sample_data = X_train[:min(100, len(X_train))] # not more 100 for
            
            pred1 = model.predict(sample_data)
            pred2 = model.predict(sample_data)
            
            # Checking between predictions
            max_diff = np.max(np.abs(pred1 - pred2))
            
            if max_diff > self.config.determinism_tolerance:
                self.results.append(ValidationResult(
                    test_name="model_determinism",
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"Model : maximum {max_diff} acceptable {self.config.determinism_tolerance}",
                    details={'max_difference': max_diff, 'tolerance': self.config.determinism_tolerance}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="model_determinism",
                    passed=True,
                    severity=ValidationSeverity.LOW,
                    message="Model",
                    details={'max_difference': max_diff, 'tolerance': self.config.determinism_tolerance}
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="model_determinism",
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Error at : {str(e)}",
                details={'error': str(e)}
            ))
    
    def _validate_performance(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> None:
        """Validation performance models on test data"""
        try:
            # predictions on test data
            y_pred = model.predict(X_test)
            
            # metrics performance
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # R² score (if applicable)
            try:
                r2 = r2_score(y_test, y_pred)
 accuracy = r2 # For using R² as accuracy
            except:
                accuracy = None
            
            # Verification minimum accuracy
            performance_details = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2_score': accuracy
            }
            
            if self.config.min_accuracy is not None and accuracy is not None:
                if accuracy < self.config.min_accuracy:
                    self.results.append(ValidationResult(
                        test_name="model_performance",
                        passed=False,
                        severity=ValidationSeverity.HIGH,
                        message=f"Performance models : {accuracy:.4f} < {self.config.min_accuracy}",
                        details=performance_details
                    ))
                else:
                    self.results.append(ValidationResult(
                        test_name="model_performance",
                        passed=True,
                        severity=ValidationSeverity.LOW,
                        message=f"Performance models requirements: {accuracy:.4f}",
                        details=performance_details
                    ))
            else:
                self.results.append(ValidationResult(
                    test_name="model_performance",
                    passed=True,
                    severity=ValidationSeverity.LOW,
                    message="Metrics performance",
                    details=performance_details
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="model_performance",
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Error at validation performance: {str(e)}",
                details={'error': str(e)}
            ))
    
    def _validate_overfitting(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> None:
        """Validation on overfitting - Enterprise overfitting detection"""
        if not self.config.check_overfitting:
            return
        
        try:
            # predictions on test data
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # errors on test data
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            # errors
            if train_mae > 0:
                error_ratio = (test_mae - train_mae) / train_mae
            else:
                error_ratio = float('inf') if test_mae > 0 else 0
            
            overfitting_details = {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'error_ratio': error_ratio,
                'threshold': self.config.overfitting_threshold
            }
            
            if error_ratio > self.config.overfitting_threshold:
                self.results.append(ValidationResult(
                    test_name="overfitting_check",
                    passed=False,
                    severity=ValidationSeverity.HIGH,
                    message=f"Detected can overfitting: errors {error_ratio:.3f} > {self.config.overfitting_threshold}",
                    details=overfitting_details
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="overfitting_check",
                    passed=True,
                    severity=ValidationSeverity.LOW,
                    message=f"Overfitting not detected: errors {error_ratio:.3f}",
                    details=overfitting_details
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="overfitting_check",
                passed=False,
                severity=ValidationSeverity.MEDIUM,
                message=f"Error at overfitting: {str(e)}",
                details={'error': str(e)}
            ))
    
    def _validate_production_readiness(self, model: Any) -> None:
        """Enterprise production readiness validation"""
        production_issues = []
        
        # Verification serializability models
        try:
            import pickle
            pickle.dumps(model)
        except Exception as e:
            production_issues.append(f"Model not may be : {str(e)}")
        
        # Verification presence required method for production
        required_production_methods = ['predict']
        for method in required_production_methods:
            if not hasattr(model, method):
                production_issues.append(f"method {method}")
        
        # Verification data models (if applicable)
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                if not params:
                    production_issues.append("Model not parameters")
            except:
                production_issues.append("Error at parameters models")
        
        if production_issues:
            self.results.append(ValidationResult(
                test_name="production_readiness",
                passed=False,
                severity=ValidationSeverity.HIGH,
 message=f"Model not production: {'; '.join(production_issues)}",
                details={'issues': production_issues}
            ))
        else:
            self.results.append(ValidationResult(
                test_name="production_readiness",
                passed=True,
                severity=ValidationSeverity.MEDIUM,
                message="Model production",
                details={'framework': self._model_framework.value}
            ))
    
    def _validate_security(self, model: Any) -> None:
        """Enterprise security validation for ML models"""
        security_issues = []
        
        # Verification on potentially dangerous attributes
        dangerous_attributes = ['__reduce__', '__getstate__', '__setstate__']
        for attr in dangerous_attributes:
            if hasattr(model, attr):
                # Checking, that this methods, not
                method = getattr(model, attr)
                if callable(method) and not method.__name__.startswith('_'):
                    security_issues.append(f"Detected potentially : {attr}")
        
        # Verification size models (protection from DoS attacks)
        try:
            import sys
            model_size = sys.getsizeof(model)
            if model_size > 100 * 1024 * 1024:  # 100MB
                security_issues.append(f"Model ({model_size / 1024 / 1024:.1f}MB), DoS")
        except:
            pass
        
        if security_issues:
            self.results.append(ValidationResult(
                test_name="security_validation",
                passed=False,
                severity=ValidationSeverity.MEDIUM,
 message=f"Detected issues : {'; '.join(security_issues)}",
                details={'issues': security_issues}
            ))
        else:
            self.results.append(ValidationResult(
                test_name="security_validation",
                passed=True,
                severity=ValidationSeverity.LOW,
                message="Verification"
            ))
    
    def _validate_memory_usage(self, model: Any, X_train: np.ndarray) -> None:
        """Validation usage memory model"""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # prediction for usage memory
            _ = model.predict(X_train)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if self.config.max_memory_usage and memory_used > self.config.max_memory_usage:
                self.results.append(ValidationResult(
                    test_name="memory_usage",
                    passed=False,
                    severity=ValidationSeverity.MEDIUM,
                    message=f"maximum usage memory: {memory_used:.1f}MB > {self.config.max_memory_usage}MB",
                    details={'memory_used_mb': memory_used, 'max_allowed_mb': self.config.max_memory_usage}
                ))
            else:
                self.results.append(ValidationResult(
                    test_name="memory_usage",
                    passed=True,
                    severity=ValidationSeverity.LOW,
                    message=f"memory : {memory_used:.1f}MB",
                    details={'memory_used_mb': memory_used}
                ))
                
        except ImportError:
            self.results.append(ValidationResult(
                test_name="memory_usage",
                passed=False,
                severity=ValidationSeverity.LOW,
                message="Not usage memory: psutil not set",
                details={'error': 'psutil_not_found'}
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                test_name="memory_usage",
                passed=False,
                severity=ValidationSeverity.LOW,
                message=f"Error at usage memory: {str(e)}",
                details={'error': str(e)}
            ))
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Retrieval results validation - Enterprise structured reporting"""
        if not self.results:
            return {'status': 'no_validation_run'}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Grouping by severity
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = sum(1 for r in self.results if r.severity == severity and not r.passed)
        
        # Detection total status
        critical_failures = severity_counts.get('critical', 0)
        high_failures = severity_counts.get('high', 0)
        
        if critical_failures > 0:
            overall_status = 'critical_failure'
        elif high_failures > 0:
            overall_status = 'high_risk'
        elif failed_tests > 0:
            overall_status = 'low_risk'
        else:
            overall_status = 'passed'
        
        return {
            'overall_status': overall_status,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'severity_breakdown': severity_counts,
            'model_framework': self._model_framework.value if self._model_framework else 'unknown',
            'validation_timestamp': self._validation_start_time.isoformat() if self._validation_start_time else None,
            'enterprise_compliant': overall_status in ['passed', 'low_risk']
        }
    
    def export_results(self, filepath: str, format: str = 'json') -> None:
        """Export results validation - Enterprise reporting"""
        import json
        
        export_data = {
            'validation_summary': self.get_validation_summary(),
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'severity': r.severity.value,
                    'message': r.message,
                    'details': r.details,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ],
            'model_metadata': self._model_metadata
        }
        
        if format.lower() == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f": {format}")
        
        self.logger.info(f"Results validation exported {filepath}")


def create_crypto_trading_validator() -> ModelValidator:
    """
 Factory function for creation for crypto trading models
    Enterprise pre-configured validator for financial ML systems
    """
    config = ValidationConfig(
        # Crypto trading specific settings
        check_prediction_range=True,
 min_prediction_value=0.0, # Prices not be negative
 max_prediction_value=1e6, # maximum for prices
        
        # requirements for
        check_determinism=True,
        determinism_tolerance=1e-12,
        
        # Performance requirements for trading
 min_accuracy=0.6, # Minimum accuracy for signal
 max_inference_time=100.0, # Maximum 100 on prediction
        
        # Overfitting for financial data
        check_overfitting=True,
 overfitting_threshold=0.05, # Strict
        
        # Enterprise enterprise mode
        enable_strict_mode=True,
        production_ready_check=True,
        security_validation=True,
        
        # Memory constraints for high-frequency trading
        max_memory_usage=500.0  # Maximum 500MB
    )
    
    return ModelValidator(config)