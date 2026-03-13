"""
Custom exceptions for Prophet Forecasting System

Comprehensive exception hierarchy for proper error handling and  
enterprise patterns with detailed logging and error context.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class ProphetForecastingException(Exception):
    """
    Base exception for system forecasting Prophet
    
    All specific exceptions must inherit from of this class
    for ensuring uniform processing errors.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialization base exceptions
        
        Args:
            message: Message about error
            error_code: Code errors for software processing
            details: Additional details errors
            original_exception: Original exception (if exists)
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        
        # Add original exception in details
        if original_exception:
            self.details['original_error'] = str(original_exception)
            self.details['original_type'] = type(original_exception).__name__
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Conversion exceptions in dictionary for JSON serialization
        
        Returns:
            Dictionary with information about error
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'original_exception': str(self.original_exception) if self.original_exception else None
        }
    
    def __str__(self) -> str:
        """String representation errors"""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.details:
            base_msg += f" | Details: {self.details}"
        return base_msg


class ModelNotTrainedException(ProphetForecastingException):
    """
    Exception for cases usage untrained model
    
    Is called when try execute forecasting or other operations
    with model, which still not was trained.
    """
    
    def __init__(
        self, 
        message: str = "Model must be trained before use",
        model_info: Optional[Dict[str, Any]] = None
    ):
        details = {"model_info": model_info} if model_info else {}
        super().__init__(
            message=message,
            error_code="MODEL_NOT_TRAINED",
            details=details
        )


class InsufficientDataException(ProphetForecastingException):
    """
    Exception for cases shortage data
    
    Is called when provided data insufficient for training model
    or execution other operations.
    """
    
    def __init__(
        self, 
        message: str,
        required_samples: Optional[int] = None,
        provided_samples: Optional[int] = None,
        min_period_days: Optional[int] = None
    ):
        details = {}
        if required_samples is not None:
            details['required_samples'] = required_samples
        if provided_samples is not None:
            details['provided_samples'] = provided_samples
        if min_period_days is not None:
            details['min_period_days'] = min_period_days
            
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            details=details
        )


class InvalidDataException(ProphetForecastingException):
    """
    Exception for incorrect input data
    
    Is called when detection incorrect format data, 
    absence required columns, incorrect types data and etc.etc.
    """
    
    def __init__(
        self, 
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        data_info: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if validation_errors:
            details['validation_errors'] = validation_errors
        if data_info:
            details['data_info'] = data_info
            
        super().__init__(
            message=message,
            error_code="INVALID_DATA", 
            details=details
        )


class ModelTrainingException(ProphetForecastingException):
    """
    Exception for errors training model
    
    Is called when errors in process training model Prophet,
    including problems with parameters, convergence and etc.etc.
    """
    
    def __init__(
        self, 
        message: str,
        model_params: Optional[Dict[str, Any]] = None,
        training_stage: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if model_params:
            details['model_params'] = model_params
        if training_stage:
            details['training_stage'] = training_stage
            
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            details=details,
            original_exception=original_exception
        )


class PredictionException(ProphetForecastingException):
    """
    Exception for errors forecasting
    
    Is called when errors in process creation forecasts,
    including problems with input data, parameters forecast and etc.etc.
    """
    
    def __init__(
        self, 
        message: str,
        prediction_params: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if prediction_params:
            details['prediction_params'] = prediction_params
        if model_info:
            details['model_info'] = model_info
            
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details,
            original_exception=original_exception
        )


class ConfigurationException(ProphetForecastingException):
    """
    Exception for errors configuration
    
    Is called when incorrect parameters configuration,
    absence required settings and etc.etc.
    """
    
    def __init__(
        self, 
        message: str,
        config_section: Optional[str] = None,
        invalid_params: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if config_section:
            details['config_section'] = config_section
        if invalid_params:
            details['invalid_params'] = invalid_params
            
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class DataProcessingException(ProphetForecastingException):
    """
    Exception for errors processing data
    
    Is called when errors in preprocessing, feature engineering,
    loading data with exchanges and other operations with data.
    """
    
    def __init__(
        self, 
        message: str,
        processing_stage: Optional[str] = None,
        data_source: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if processing_stage:
            details['processing_stage'] = processing_stage
        if data_source:
            details['data_source'] = data_source
            
        super().__init__(
            message=message,
            error_code="DATA_PROCESSING_ERROR",
            details=details,
            original_exception=original_exception
        )


class APIException(ProphetForecastingException):
    """
    Exception for errors API
    
    Is called when errors in REST API endpoints,
    WebSocket connections and other API operations.
    """
    
    def __init__(
        self, 
        message: str,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint
        if request_data:
            details['request_data'] = request_data
            
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details=details
        )


class ValidationException(ProphetForecastingException):
    """
    Exception for errors validation forecasts
    
    Is called when errors in cross-validation, metrics quality,
    backtesting and other procedures validation.
    """
    
    def __init__(
        self, 
        message: str,
        validation_type: Optional[str] = None,
        failed_metrics: Optional[Dict[str, Any]] = None,
        threshold_violations: Optional[Dict[str, Any]] = None
    ):
        details = {}
        if validation_type:
            details['validation_type'] = validation_type
        if failed_metrics:
            details['failed_metrics'] = failed_metrics
        if threshold_violations:
            details['threshold_violations'] = threshold_violations
            
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class OptimizationException(ProphetForecastingException):
    """
    Exception for errors optimization hyperparameters
    
    Is called when errors in Bayesian optimization, Grid Search
    and other methods optimization parameters model.
    """
    
    def __init__(
        self, 
        message: str,
        optimization_method: Optional[str] = None,
        trial_info: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        details = {}
        if optimization_method:
            details['optimization_method'] = optimization_method
        if trial_info:
            details['trial_info'] = trial_info
            
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            details=details,
            original_exception=original_exception
        )


# Helper function for work with exceptions

def handle_prophet_exception(func):
    """
    Decorator for processing exceptions in functions Prophet
    
    Automatically wraps standard exceptions in specialized
    exceptions system forecasting.
    
    Args:
        func: Function for decoration
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ProphetForecastingException:
            # Already our exception - simply skip
            raise
        except ValueError as e:
            raise InvalidDataException(
                f"Invalid data in {func.__name__}: {e}",
                original_exception=e
            )
        except KeyError as e:
            raise InvalidDataException(
                f"Missing required field in {func.__name__}: {e}",
                original_exception=e
            )
        except FileNotFoundError as e:
            raise DataProcessingException(
                f"File not found in {func.__name__}: {e}",
                original_exception=e
            )
        except Exception as e:
            # Total exception for unforeseen cases
            raise ProphetForecastingException(
                f"Unexpected error in {func.__name__}: {e}",
                original_exception=e
            )
    
    return wrapper


def create_error_response(exception: ProphetForecastingException) -> Dict[str, Any]:
    """
    Creation standardized response about error for API
    
    Args:
        exception: Exception system forecasting
        
    Returns:
        Dictionary with information about error for API response
    """
    return {
        "success": False,
        "error": {
            "type": exception.__class__.__name__,
            "code": exception.error_code,
            "message": exception.message,
            "details": exception.details,
            "timestamp": exception.timestamp.isoformat()
        }
    }


def log_exception(logger, exception: Exception, context: Optional[Dict[str, Any]] = None):
    """
    Logging exceptions with context
    
    Args:
        logger: Object logger
        exception: Exception for logging
        context: Additional context
    """
    context = context or {}
    
    if isinstance(exception, ProphetForecastingException):
        logger.error(
            f"Prophet Exception: {exception.message}",
            extra={
                "error_code": exception.error_code,
                "error_type": exception.__class__.__name__,
                "details": exception.details,
                "timestamp": exception.timestamp.isoformat(),
                **context
            },
            exc_info=exception.original_exception is not None
        )
    else:
        logger.error(
            f"Unexpected exception: {exception}",
            extra={
                "error_type": type(exception).__name__,
                **context
            },
            exc_info=True
        )