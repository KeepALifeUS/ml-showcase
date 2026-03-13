"""
Centralized logging configuration for Elliott Wave Analyzer.

Structured logging with performance monitoring,
distributed tracing, and security audit trails.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger
from datetime import datetime
import asyncio
import inspect
import traceback
from functools import wraps
from .config import config, is_production, is_debug


class LogLevel:
    """Log level constants."""
    TRACE = "TRACE"
    DEBUG = "DEBUG" 
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PerformanceLogger:
    """
    Performance monitoring logger.
    
    Performance observability with metrics collection.
    """
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        
    def log_execution_time(self, func_name: str, duration: float, **kwargs):
        """Log function execution time."""
        logger.info(
            "Performance metric",
            function=func_name,
            duration_ms=duration * 1000,
            **kwargs
        )
        
    def log_memory_usage(self, func_name: str, memory_mb: float):
        """Log memory usage."""
        logger.info(
            "Memory usage",
            function=func_name,
            memory_mb=memory_mb
        )


class SecurityLogger:
    """
    Security audit logger.
    
    Security monitoring and audit trails.
    """
    
    @staticmethod
    def log_auth_attempt(username: str, success: bool, ip_address: str):
        """Log authentication attempt."""
        logger.info(
            "Authentication attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            security_event="auth_attempt"
        )
        
    @staticmethod
    def log_api_access(endpoint: str, method: str, status_code: int, user_id: Optional[str] = None):
        """Log API access."""
        logger.info(
            "API access",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            user_id=user_id,
            security_event="api_access"
        )
        
    @staticmethod
    def log_data_access(resource: str, action: str, user_id: str):
        """Log sensitive data access."""
        logger.warning(
            "Data access",
            resource=resource,
            action=action,
            user_id=user_id,
            security_event="data_access"
        )


class TradingLogger:
    """
    Trading-specific logger for Elliott Wave analysis.
    
    Domain-specific logging with structured data.
    """
    
    @staticmethod
    def log_wave_detection(symbol: str, timeframe: str, wave_type: str, confidence: float, **kwargs):
        """Log wave pattern detection."""
        logger.info(
            "Wave pattern detected",
            symbol=symbol,
            timeframe=timeframe,
            wave_type=wave_type,
            confidence=confidence,
            trading_event="wave_detection",
            **kwargs
        )
        
    @staticmethod
    def log_signal_generation(symbol: str, signal_type: str, entry_price: float, 
                            stop_loss: float, take_profit: float, confidence: float):
        """Log trading signal generation."""
        logger.info(
            "Trading signal generated",
            symbol=symbol,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            trading_event="signal_generation"
        )
        
    @staticmethod
    def log_fibonacci_level(symbol: str, timeframe: str, level_type: str, 
                          price_level: float, support_resistance: str):
        """Log Fibonacci level identification."""
        logger.info(
            "Fibonacci level identified",
            symbol=symbol,
            timeframe=timeframe,
            level_type=level_type,
            price_level=price_level,
            support_resistance=support_resistance,
            trading_event="fibonacci_level"
        )
        
    @staticmethod
    def log_market_structure(symbol: str, timeframe: str, structure_type: str, **kwargs):
        """Log market structure analysis."""
        logger.info(
            "Market structure analysis",
            symbol=symbol,
            timeframe=timeframe,
            structure_type=structure_type,
            trading_event="market_structure",
            **kwargs
        )


class ErrorLogger:
    """
    Enhanced error logging with context and stack traces.
    
    Error observability with actionable insights.
    """
    
    @staticmethod
    def log_exception(exception: Exception, context: Optional[Dict[str, Any]] = None):
        """Log exception with full context."""
        error_context = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "stack_trace": traceback.format_exc(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if context:
            error_context.update(context)
            
        logger.error("Exception occurred", **error_context)
        
    @staticmethod
    def log_validation_error(field: str, value: Any, expected: str, actual: str):
        """Log validation error."""
        logger.error(
            "Validation error",
            field=field,
            value=value,
            expected=expected,
            actual=actual,
            error_type="validation"
        )
        
    @staticmethod
    def log_api_error(endpoint: str, status_code: int, error_message: str, **kwargs):
        """Log API error."""
        logger.error(
            "API error",
            endpoint=endpoint,
            status_code=status_code,
            error_message=error_message,
            error_type="api_error",
            **kwargs
        )


def setup_logging() -> None:
    """
    Setup centralized logging configuration.
    
    Centralized logging with environment-specific configuration.
    """
    # Remove default logger
    logger.remove()
    
    # Console logging
    if is_debug():
        log_level = "DEBUG"
        log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                     "<level>{level: <8}</level> | "
                     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                     "<level>{message}</level>")
    else:
        log_level = config.log_level
        log_format = config.log_format
    
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # File logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Main log file
    logger.add(
        log_dir / "elliott_wave.log",
        format=config.log_format,
        level=log_level,
        rotation=config.log_rotation,
        retention=config.log_retention,
        compression="gz",
        serialize=False
    )
    
    # Error log file
    logger.add(
        log_dir / "errors.log",
        format=config.log_format,
        level="ERROR",
        rotation=config.log_rotation,
        retention=config.log_retention,
        compression="gz",
        serialize=False
    )
    
    # JSON structured log for production
    if is_production():
        logger.add(
            log_dir / "elliott_wave.json",
            format="{time} {level} {message}",
            level=log_level,
            rotation=config.log_rotation,
            retention=config.log_retention,
            compression="gz",
            serialize=True
        )
    
    # Trading-specific logs
    logger.add(
        log_dir / "trading.log",
        format=config.log_format,
        level="INFO",
        rotation=config.log_rotation,
        retention=config.log_retention,
        compression="gz",
        filter=lambda record: "trading_event" in record["extra"]
    )
    
    # Security audit logs
    logger.add(
        log_dir / "security.log",
        format=config.log_format,
        level="INFO",
        rotation=config.log_rotation,
        retention="1 year",  # Keep security logs longer
        compression="gz",
        filter=lambda record: "security_event" in record["extra"]
    )


def performance_monitor(func):
    """
    Decorator for performance monitoring.
    
    Performance observability through decorators.
    """
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(
                    "Function executed",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    success=True
                )
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    success=False,
                    error=str(e)
                )
                raise
        return async_wrapper
    else:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                logger.info(
                    "Function executed",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    success=True
                )
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger.error(
                    "Function failed",
                    function=func.__name__,
                    duration_ms=duration * 1000,
                    success=False,
                    error=str(e)
                )
                raise
        return sync_wrapper


def get_logger(name: str) -> logger:
    """
    Get named logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logger: Configured logger instance
    """
    return logger.bind(name=name)


def log_startup():
    """Log application startup information."""
    logger.info(
        "Elliott Wave Analyzer starting",
        version="1.0.0",
        environment=config.environment,
        debug=is_debug(),
        python_version=sys.version,
        startup_event="application_start"
    )


def log_shutdown():
    """Log application shutdown."""
    logger.info(
        "Elliott Wave Analyzer shutting down",
        shutdown_event="application_stop"
    )


# Initialize logging
setup_logging()

# Create logger instances
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()
trading_logger = TradingLogger()
error_logger = ErrorLogger()

# Export for easy import
__all__ = [
    'logger',
    'get_logger',
    'setup_logging',
    'performance_monitor',
    'performance_logger',
    'security_logger',
    'trading_logger',
    'error_logger',
    'log_startup',
    'log_shutdown',
    'LogLevel'
]