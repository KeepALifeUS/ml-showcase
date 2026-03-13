"""
Structured logging utilities for Prophet Forecasting System

Enterprise-grade logging with enterprise patterns, structured output,
and comprehensive observability features for production environments.
"""

import logging
import sys
from typing import Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import json
from enum import Enum

import structlog
from structlog.types import Processor


class LogLevel(str, Enum):
    """Levels logging"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Formats logging"""
    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


# Global configuration logging
_logging_configured = False
_log_level = LogLevel.INFO
_log_format = LogFormat.JSON


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    format_type: LogFormat = LogFormat.JSON,
    log_file: Optional[Union[str, Path]] = None,
    service_name: str = "prophet-forecasting",
    service_version: str = "5.0.0",
    environment: str = "development"
) -> None:
    """
    Configuration structured logging for total application
    
    Args:
        level: Level logging
        format_type: Format output logs
        log_file: Path to file logs (optionally)
        service_name: Name service
        service_version: Version service
        environment: Environment execution
    """
    global _logging_configured, _log_level, _log_format
    
    if _logging_configured:
        return
    
    _log_level = level
    _log_format = format_type
    
    # Processors for structuring logs
    processors = [
        # Addition timestamp
        structlog.processors.TimeStamper(fmt="ISO"),
        
        # Addition level log
        structlog.stdlib.add_log_level,
        
        # Addition information about logger
        structlog.stdlib.add_logger_name,
        
        # Addition context service
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        
        # Processing exceptions
        structlog.processors.format_exc_info,
        
        # Custom processor for additions metadata service
        _add_service_context(service_name, service_version, environment),
    ]
    
    # Selection final processor on basis format
    if format_type == LogFormat.JSON:
        processors.append(structlog.processors.JSONRenderer())
    elif format_type == LogFormat.COLORED:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    else:  # TEXT
        processors.append(
            structlog.processors.KeyValueRenderer(
                key_order=['timestamp', 'level', 'logger', 'event']
            )
        )
    
    # Configuration structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configuration standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.value)
    )
    
    # Configuration file logging
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.value))
        
        # For file always use JSON format
        file_formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
        file_handler.setFormatter(file_formatter)
        
        # Addition to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    # Suppression redundant logs from third-party libraries
    _suppress_noisy_loggers()
    
    _logging_configured = True


def _add_service_context(
    service_name: str, 
    service_version: str, 
    environment: str
) -> Processor:
    """
    Creation processor for additions context service
    
    Args:
        service_name: Name service
        service_version: Version service
        environment: Environment execution
        
    Returns:
        Processor structlog
    """
    def processor(logger, method_name, event_dict):
        event_dict.update({
            'service': service_name,
            'version': service_version,
            'environment': environment,
            'pid': structlog.processors._get_process_id(),
        })
        return event_dict
    
    return processor


def _suppress_noisy_loggers():
    """Suppression excessive logging from third-party libraries"""
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3.connectionpool',
        'matplotlib',
        'PIL',
        'asyncio',
        'concurrent.futures',
        'cmdstanpy'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Retrieval configured structured logger
    
    Args:
        name: Name logger (optionally, by default __name__ caller'and)
        
    Returns:
        Configured structured logger
    """
    # Auto-configuration when first usage
    if not _logging_configured:
        configure_logging()
    
    # Retrieval name from caller'and if not specified
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return structlog.get_logger(name)


def get_request_logger(
    request_id: str,
    user_id: Optional[str] = None,
    endpoint: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Retrieval logger with context HTTP request
    
    Args:
        request_id: ID request for tracing
        user_id: ID user (if exists)
        endpoint: Endpoint API
        
    Returns:
        Logger with bound context request
    """
    logger = get_logger("api")
    
    context = {
        'request_id': request_id,
        'endpoint': endpoint
    }
    
    if user_id:
        context['user_id'] = user_id
    
    return logger.bind(**context)


def get_model_logger(
    symbol: str,
    timeframe: str,
    model_version: Optional[str] = None,
    operation: Optional[str] = None
) -> structlog.BoundLogger:
    """
    Retrieval logger with context model
    
    Args:
        symbol: Symbol cryptocurrency
        timeframe: Timeframe
        model_version: Version model
        operation: Current operation (train, predict, validate)
        
    Returns:
        Logger with bound context model
    """
    logger = get_logger("model")
    
    context = {
        'symbol': symbol,
        'timeframe': timeframe
    }
    
    if model_version:
        context['model_version'] = model_version
    if operation:
        context['operation'] = operation
    
    return logger.bind(**context)


def log_performance_metrics(
    logger: structlog.BoundLogger,
    operation: str,
    duration_seconds: float,
    success: bool = True,
    additional_metrics: Optional[Dict[str, Any]] = None
):
    """
    Logging metrics performance
    
    Args:
        logger: Logger for records
        operation: Name operations
        duration_seconds: Duration in seconds
        success: Success rate operations
        additional_metrics: Additional metrics
    """
    metrics = {
        'operation': operation,
        'duration_seconds': round(duration_seconds, 4),
        'success': success,
        'performance_log': True
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    if success:
        logger.info(f"Performance: {operation} completed", **metrics)
    else:
        logger.error(f"Performance: {operation} failed", **metrics)


def log_forecast_metrics(
    logger: structlog.BoundLogger,
    symbol: str,
    timeframe: str,
    forecast_points: int,
    training_samples: int,
    metrics: Dict[str, float]
):
    """
    Logging metrics forecasting
    
    Args:
        logger: Logger for records
        symbol: Symbol cryptocurrency
        timeframe: Timeframe
        forecast_points: Number points forecast
        training_samples: Number samples for training
        metrics: Metrics quality model
    """
    log_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'forecast_points': forecast_points,
        'training_samples': training_samples,
        'forecast_metrics': metrics,
        'metrics_log': True
    }
    
    logger.info("Forecast metrics computed", **log_data)


def log_model_training(
    logger: structlog.BoundLogger,
    symbol: str,
    timeframe: str,
    training_duration: float,
    samples_count: int,
    model_params: Dict[str, Any],
    validation_metrics: Optional[Dict[str, float]] = None
):
    """
    Logging process training model
    
    Args:
        logger: Logger for records
        symbol: Symbol cryptocurrency
        timeframe: Timeframe
        training_duration: Duration training
        samples_count: Number samples
        model_params: Parameters model
        validation_metrics: Metrics validation
    """
    log_data = {
        'symbol': symbol,
        'timeframe': timeframe,
        'training_duration_seconds': round(training_duration, 4),
        'training_samples': samples_count,
        'model_parameters': model_params,
        'training_log': True
    }
    
    if validation_metrics:
        log_data['validation_metrics'] = validation_metrics
    
    logger.info("Model training completed", **log_data)


class LoggerMixin:
    """
    Mixin class for additions logging in other classes
    
    Provides convenient interface for logging in methods class
    with automatic context class.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None
        self._log_context = {}
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Retrieval logger for class"""
        if self._logger is None:
            class_name = self.__class__.__name__
            module_name = self.__class__.__module__
            logger_name = f"{module_name}.{class_name}"
            
            base_logger = get_logger(logger_name)
            
            # Addition context class
            context = {
                'class': class_name,
                **self._log_context
            }
            
            self._logger = base_logger.bind(**context)
        
        return self._logger
    
    def set_log_context(self, **kwargs):
        """
        Installation additional context for logging
        
        Args:
            **kwargs: Contextual variables
        """
        self._log_context.update(kwargs)
        # Reset logger for recreation with new context
        self._logger = None
    
    def log_operation_start(self, operation: str, **kwargs):
        """Logging beginning operations"""
        self.logger.info(f"Starting {operation}", operation=operation, **kwargs)
    
    def log_operation_end(self, operation: str, success: bool = True, **kwargs):
        """Logging completion operations"""
        if success:
            self.logger.info(f"Completed {operation}", operation=operation, success=success, **kwargs)
        else:
            self.logger.error(f"Failed {operation}", operation=operation, success=success, **kwargs)


# Decorators for automatic logging

def log_function_calls(
    include_args: bool = False,
    include_result: bool = False,
    log_level: LogLevel = LogLevel.DEBUG
):
    """
    Decorator for automatic logging calls functions
    
    Args:
        include_args: Include arguments in log
        include_result: Include result in log
        log_level: Level logging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            log_data = {
                'function': func.__name__,
                'function_call': True
            }
            
            if include_args:
                log_data['args'] = args
                log_data['kwargs'] = kwargs
            
            # Logging beginning
            getattr(logger, log_level.value.lower())(f"Calling {func.__name__}", **log_data)
            
            try:
                result = func(*args, **kwargs)
                
                # Logging success
                success_data = log_data.copy()
                if include_result:
                    success_data['result'] = result
                
                getattr(logger, log_level.value.lower())(f"Completed {func.__name__}", **success_data)
                
                return result
                
            except Exception as e:
                # Logging errors
                error_data = log_data.copy()
                error_data.update({
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                logger.error(f"Failed {func.__name__}", **error_data)
                raise
        
        return wrapper
    return decorator


def timed_operation(operation_name: Optional[str] = None):
    """
    Decorator for measurements time execution operations
    
    Args:
        operation_name: Name operations (by default name function)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            
            logger = get_logger(func.__module__)
            op_name = operation_name or func.__name__
            
            start_time = time.time()
            logger.debug(f"Starting timed operation: {op_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                log_performance_metrics(
                    logger=logger,
                    operation=op_name,
                    duration_seconds=duration,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                log_performance_metrics(
                    logger=logger,
                    operation=op_name,
                    duration_seconds=duration,
                    success=False,
                    additional_metrics={'error': str(e)}
                )
                
                raise
        
        return wrapper
    return decorator