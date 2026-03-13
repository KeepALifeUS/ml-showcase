"""
System logging for Order Flow Detection
Uses enterprise patterns with structured logging
"""

import logging
import structlog
import sys
from typing import Any, Dict, Optional, List
from pathlib import Path
import json
from datetime import datetime
from enum import Enum
import traceback
import asyncio
from contextlib import contextmanager
import threading
from functools import wraps
import time

from .config import get_settings, LogLevel

class LoggerType(str, Enum):
    """Types loggers"""
    ORDER_FLOW = "order_flow"
    PATTERN_DETECTION = "pattern_detection"
    ML_MODELS = "ml_models"
    API = "api"
    REALTIME = "realtime"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"

class LogContext:
    """
    Context for structured logging
    Contextual Logging pattern
    """
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
        self._local = threading.local()
    
    def set(self, **kwargs) -> None:
        """Installation context"""
        if not hasattr(self._local, 'context'):
            self._local.context = {}
        self._local.context.update(kwargs)
    
    def get(self) -> Dict[str, Any]:
        """Retrieval context"""
        if hasattr(self._local, 'context'):
            return self._local.context.copy()
        return {}
    
    def clear(self) -> None:
        """Cleanup context"""
        if hasattr(self._local, 'context'):
            self._local.context.clear()
    
    @contextmanager
    def bind(self, **kwargs):
        """Temporal binding context"""
        old_context = self.get()
        self.set(**kwargs)
        try:
            yield
        finally:
            self.clear()
            self.set(**old_context)

# Global context logging
log_context = LogContext()

def add_correlation_id(logger, method_name, event_dict):
    """Addition correlation ID for tracing requests"""
    context = log_context.get()
    if 'correlation_id' not in event_dict and 'correlation_id' in context:
        event_dict['correlation_id'] = context['correlation_id']
    return event_dict

def add_timestamp(logger, method_name, event_dict):
    """Addition temporal labels"""
    event_dict['timestamp'] = datetime.utcnow().isoformat()
    return event_dict

def add_logger_name(logger, method_name, event_dict):
    """Addition name logger"""
    if hasattr(logger, 'name'):
        event_dict['logger'] = logger.name
    return event_dict

def add_context_info(logger, method_name, event_dict):
    """Addition contextual information"""
    context = log_context.get()
    for key, value in context.items():
        if key not in event_dict:
            event_dict[key] = value
    return event_dict

class PerformanceLogger:
    """
    Logger performance
    Performance Monitoring pattern
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_execution_time(self, operation: str, duration: float, **kwargs):
        """Logging time execution"""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Logging usage memory"""
        self.logger.info(
            "Memory usage",
            operation=operation,
            memory_mb=memory_mb,
            **kwargs
        )
    
    def log_throughput(self, operation: str, items_per_second: float, **kwargs):
        """Logging throughput capabilities"""
        self.logger.info(
            "Throughput metric",
            operation=operation,
            items_per_second=items_per_second,
            **kwargs
        )

class SecurityLogger:
    """
    Security logging
    Security Logging pattern
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip: str = None):
        """Logging attempts authentication"""
        self.logger.warning(
            "Authentication attempt",
            user_id=user_id,
            success=success,
            ip_address=ip,
            event_type="auth_attempt"
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, action: str):
        """Logging failures authorization"""
        self.logger.warning(
            "Authorization failure",
            user_id=user_id,
            resource=resource,
            action=action,
            event_type="auth_failure"
        )
    
    def log_suspicious_activity(self, description: str, severity: str = "medium", **kwargs):
        """Logging suspicious activity"""
        self.logger.error(
            "Suspicious activity detected",
            description=description,
            severity=severity,
            event_type="suspicious_activity",
            **kwargs
        )

class ErrorLogger:
    """
    Logger errors with detailed information
    Error Handling pattern
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
    
    def log_exception(self, exc: Exception, operation: str = None, **kwargs):
        """Logging exceptions with full information"""
        self.logger.error(
            "Exception occurred",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation=operation,
            traceback=traceback.format_exc(),
            **kwargs
        )
    
    def log_api_error(self, endpoint: str, status_code: int, error_message: str, **kwargs):
        """Logging errors API"""
        self.logger.error(
            "API error",
            endpoint=endpoint,
            status_code=status_code,
            error_message=error_message,
            event_type="api_error",
            **kwargs
        )
    
    def log_data_error(self, data_source: str, error_type: str, details: Dict[str, Any]):
        """Logging errors data"""
        self.logger.error(
            "Data error",
            data_source=data_source,
            error_type=error_type,
            details=details,
            event_type="data_error"
        )

class OrderFlowLogger:
    """
    Specialized logger for order flow analysis
    """
    
    def __init__(self, logger: structlog.BoundLogger):
        self.logger = logger
        self.performance = PerformanceLogger(logger)
        self.security = SecurityLogger(logger)
        self.error = ErrorLogger(logger)
    
    def log_pattern_detected(self, pattern_type: str, symbol: str, confidence: float, **kwargs):
        """Logging detection pattern"""
        self.logger.info(
            "Pattern detected",
            pattern_type=pattern_type,
            symbol=symbol,
            confidence=confidence,
            event_type="pattern_detection",
            **kwargs
        )
    
    def log_order_flow_anomaly(self, symbol: str, anomaly_type: str, severity: float, **kwargs):
        """Logging anomalies order flow"""
        self.logger.warning(
            "Order flow anomaly",
            symbol=symbol,
            anomaly_type=anomaly_type,
            severity=severity,
            event_type="flow_anomaly",
            **kwargs
        )
    
    def log_market_manipulation(self, symbol: str, manipulation_type: str, evidence: Dict[str, Any]):
        """Logging suspicions on manipulation market"""
        self.security.log_suspicious_activity(
            f"Market manipulation detected: {manipulation_type}",
            severity="high",
            symbol=symbol,
            manipulation_type=manipulation_type,
            evidence=evidence
        )
    
    def log_model_prediction(self, model_name: str, symbol: str, prediction: Dict[str, Any]):
        """Logging predictions model"""
        self.logger.info(
            "Model prediction",
            model_name=model_name,
            symbol=symbol,
            prediction=prediction,
            event_type="model_prediction"
        )
    
    def log_trade_signal(self, signal_type: str, symbol: str, strength: float, **kwargs):
        """Logging trading signal"""
        self.logger.info(
            "Trade signal generated",
            signal_type=signal_type,
            symbol=symbol,
            strength=strength,
            event_type="trade_signal",
            **kwargs
        )

def setup_logging() -> None:
    """
    Configuration system logging
    Centralized Configuration pattern
    """
    settings = get_settings()
    
    # Configuration structlog
    structlog.configure(
        processors=[
            # Filters and handlers
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            add_correlation_id,
            add_timestamp,
            add_context_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configuration standard logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.value),
        format="%(message)s" if settings.debug else "",
        stream=sys.stdout,
    )
    
    # Creation directory for logs
    if not settings.logs_dir.exists():
        settings.logs_dir.mkdir(parents=True)
    
    # Configuration file logging
    file_handler = logging.FileHandler(settings.logs_dir / "order_flow.log")
    file_handler.setLevel(getattr(logging, settings.log_level.value))
    
    # Configuration formatter for files
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Addition handler to root logger
    logging.getLogger().addHandler(file_handler)

def get_logger(logger_type: LoggerType = LoggerType.SYSTEM, name: str = None) -> OrderFlowLogger:
    """
    Retrieval logger specific type
    Factory pattern for loggers
    """
    logger_name = name or f"order_flow.{logger_type.value}"
    base_logger = structlog.get_logger(logger_name)
    return OrderFlowLogger(base_logger)

def performance_monitor(operation: str = None):
    """
    Decorator for monitoring performance
    Aspect-Oriented Programming pattern
    """
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(LoggerType.PERFORMANCE)
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance.log_execution_time(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.performance.log_execution_time(op_name, duration, error=True)
                logger.error.log_exception(e, operation=op_name)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(LoggerType.PERFORMANCE)
            op_name = operation or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.performance.log_execution_time(op_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.performance.log_execution_time(op_name, duration, error=True)
                logger.error.log_exception(e, operation=op_name)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator

def log_function_call(logger_type: LoggerType = LoggerType.SYSTEM):
    """
    Decorator for logging calls functions
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_type)
            func_name = f"{func.__module__}.{func.__name__}"
            
            logger.logger.debug(
                "Function called",
                function=func_name,
                args=len(args),
                kwargs=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                logger.logger.debug("Function completed", function=func_name)
                return result
            except Exception as e:
                logger.error.log_exception(e, operation=func_name)
                raise
        
        return wrapper
    return decorator

# Context manager for logging blocks code
@contextmanager
def log_block(operation: str, logger_type: LoggerType = LoggerType.SYSTEM, **kwargs):
    """Logging block code"""
    logger = get_logger(logger_type)
    start_time = time.time()
    
    logger.logger.info("Starting operation", operation=operation, **kwargs)
    
    try:
        yield logger
        duration = time.time() - start_time
        logger.logger.info(
            "Operation completed",
            operation=operation,
            duration_ms=duration * 1000
        )
    except Exception as e:
        duration = time.time() - start_time
        logger.logger.error(
            "Operation failed",
            operation=operation,
            duration_ms=duration * 1000,
            error=str(e)
        )
        raise

# Utility functions
def set_correlation_id(correlation_id: str):
    """Installation correlation ID for tracing"""
    log_context.set(correlation_id=correlation_id)

def set_user_context(user_id: str, session_id: str = None):
    """Installation user context"""
    context = {'user_id': user_id}
    if session_id:
        context['session_id'] = session_id
    log_context.set(**context)

def set_trading_context(symbol: str, exchange: str = None):
    """Installation trading context"""
    context = {'symbol': symbol}
    if exchange:
        context['exchange'] = exchange
    log_context.set(**context)

# Export main logging components
__all__ = [
    'LoggerType',
    'OrderFlowLogger',
    'PerformanceLogger',
    'SecurityLogger',
    'ErrorLogger',
    'LogContext',
    'setup_logging',
    'get_logger',
    'performance_monitor',
    'log_function_call',
    'log_block',
    'set_correlation_id',
    'set_user_context',
    'set_trading_context',
    'log_context'
]