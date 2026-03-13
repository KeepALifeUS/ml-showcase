"""
Enterprise Logging System for NLP Sentiment Analysis

Structured logging with enterprise patterns for comprehensive monitoring,
debugging, and observability in cryptocurrency sentiment analysis.

Author: ML-Framework Team
"""

import logging
import logging.config
import logging.handlers
import sys
import json
import time
import traceback
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import structlog
import asyncio
from functools import wraps
from contextlib import contextmanager

# observability imports
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

try:
    import prometheus_client
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def filter(self, record):
        # Add timestamp
        record.timestamp = datetime.utcnow().isoformat()
        
        # Add trace information if available
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span:
                record.trace_id = format(span.get_span_context().trace_id, '032x')
                record.span_id = format(span.get_span_context().span_id, '016x')
        
        return True


class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def __init__(self, include_timestamp=True, include_trace=True):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_trace = include_trace
    
    def format(self, record):
        """Format log record as JSON"""
        
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if self.include_timestamp:
            log_data["timestamp"] = datetime.utcnow().isoformat()
        
        # Add trace information
        if self.include_trace and hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
            log_data["span_id"] = record.span_id
        
        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        extra_fields = {
            k: v for k, v in record.__dict__.items()
            if k not in ["name", "msg", "args", "levelname", "levelno", 
                        "pathname", "filename", "module", "lineno", 
                        "funcName", "created", "msecs", "relativeCreated",
                        "thread", "threadName", "processName", "process",
                        "message", "exc_info", "exc_text", "stack_info"]
        }
        
        if extra_fields:
            log_data["extra"] = extra_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class MetricsHandler(logging.Handler):
    """Log handler that updates Prometheus metrics"""
    
    def __init__(self):
        super().__init__()
        if PROMETHEUS_AVAILABLE:
            self.log_counter = prometheus_client.Counter(
                'log_messages_total',
                'Total number of log messages',
                ['level', 'logger']
            )
            self.error_counter = prometheus_client.Counter(
                'log_errors_total',
                'Total number of error messages',
                ['logger', 'exception_type']
            )
    
    def emit(self, record):
        """Update metrics based on log record"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Update log counter
            self.log_counter.labels(
                level=record.levelname,
                logger=record.name
            ).inc()
            
            # Update error counter for errors/exceptions
            if record.levelno >= logging.ERROR:
                exception_type = "none"
                if record.exc_info:
                    exception_type = record.exc_info[0].__name__
                
                self.error_counter.labels(
                    logger=record.name,
                    exception_type=exception_type
                ).inc()
        
        except Exception:
            # Don't fail if metrics update fails
            pass


class AsyncLogHandler(logging.Handler):
    """Async handler for non-blocking log processing"""
    
    def __init__(self, target_handler):
        super().__init__()
        self.target_handler = target_handler
        self.queue = asyncio.Queue()
        self.processor_task = None
    
    def emit(self, record):
        """Queue log record for async processing"""
        try:
            if self.queue:
                # Use asyncio.create_task if in async context
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.queue.put(record))
                except RuntimeError:
                    # Not in async context, process synchronously
                    self.target_handler.emit(record)
        except Exception:
            # Fallback to synchronous processing
            self.target_handler.emit(record)
    
    async def process_logs(self):
        """Async log processor"""
        while True:
            try:
                record = await self.queue.get()
                self.target_handler.emit(record)
                self.queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log processing error (avoid infinite recursion)
                print(f"Error processing log: {e}", file=sys.stderr)


class Logger:
    """
    Enterprise Logger with enterprise integration
    
    Features:
    - Structured JSON logging
    - Distributed tracing integration
    - Metrics collection
    - Async processing
    - Context management
    - Performance monitoring
    - Error tracking
    """
    
    def __init__(
        self,
        name: str,
        level: str = "INFO",
        structured: bool = True,
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_async: bool = False,
        log_file: Optional[str] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
    ):
        self.name = name
        self.structured = structured
        self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE
        self.enable_tracing = enable_tracing and OTEL_AVAILABLE
        self.enable_async = enable_async
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers(log_file, max_file_size, backup_count)
        
        # Add context filter
        context_filter = ContextFilter()
        self.logger.addFilter(context_filter)
        
        # Performance tracking
        self.performance_data = {}
    
    def _setup_handlers(self, log_file: Optional[str], max_file_size: int, backup_count: int):
        """Setup log handlers"""
        
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.structured:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        handlers.append(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            if self.structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
            handlers.append(file_handler)
        
        # Metrics handler
        if self.enable_metrics:
            metrics_handler = MetricsHandler()
            handlers.append(metrics_handler)
        
        # Wrap with async handler if enabled
        if self.enable_async:
            for handler in handlers:
                async_handler = AsyncLogHandler(handler)
                self.logger.addHandler(async_handler)
        else:
            for handler in handlers:
                self.logger.addHandler(handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger instance"""
        return self.logger
    
    def debug(self, message: str, **kwargs):
        """Debug level logging"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Info level logging"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Warning level logging"""
        self._log(logging.WARNING, message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Alias for warning"""
        self.warning(message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Error level logging"""
        if exception:
            kwargs["exc_info"] = (type(exception), exception, exception.__traceback__)
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Critical level logging"""
        if exception:
            kwargs["exc_info"] = (type(exception), exception, exception.__traceback__)
        self._log(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method"""
        
        # Add performance data if available
        if hasattr(self, '_current_operation'):
            kwargs.update(self.performance_data.get(self._current_operation, {}))
        
        # Add trace span if available
        if self.enable_tracing and OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span:
                # Add span context to log
                kwargs.update({
                    "trace_id": format(span.get_span_context().trace_id, '032x'),
                    "span_id": format(span.get_span_context().span_id, '016x'),
                })
        
        # Log with extra context
        self.logger.log(level, message, extra=kwargs)
    
    @contextmanager
    def operation(self, operation_name: str, **context):
        """Context manager for operation tracking"""
        
        start_time = time.time()
        self._current_operation = operation_name
        
        # Create span if tracing enabled
        span = None
        if self.enable_tracing and OTEL_AVAILABLE:
            tracer = trace.get_tracer(self.name)
            span = tracer.start_span(operation_name)
            
            # Add context to span
            for key, value in context.items():
                span.set_attribute(key, str(value))
        
        try:
            self.info(f"Starting operation: {operation_name}", **context)
            yield self
            
            # Success
            duration = time.time() - start_time
            self.info(
                f"Completed operation: {operation_name}",
                duration=duration,
                status="success",
                **context
            )
            
            if span:
                span.set_status(Status(StatusCode.OK))
        
        except Exception as e:
            # Error
            duration = time.time() - start_time
            self.error(
                f"Failed operation: {operation_name}",
                exception=e,
                duration=duration,
                status="error",
                **context
            )
            
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            
            raise
        
        finally:
            # Cleanup
            if hasattr(self, '_current_operation'):
                delattr(self, '_current_operation')
            
            if span:
                span.end()
    
    def performance_timer(self, operation_name: str):
        """Decorator for measuring performance"""
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.info(
                        f"Performance: {operation_name}",
                        function=func.__name__,
                        duration=duration,
                        status="success"
                    )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.error(
                        f"Performance: {operation_name} failed",
                        function=func.__name__,
                        duration=duration,
                        status="error",
                        exception=e
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def async_performance_timer(self, operation_name: str):
        """Async decorator for measuring performance"""
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.info(
                        f"Async Performance: {operation_name}",
                        function=func.__name__,
                        duration=duration,
                        status="success"
                    )
                    
                    return result
                
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.error(
                        f"Async Performance: {operation_name} failed",
                        function=func.__name__,
                        duration=duration,
                        status="error",
                        exception=e
                    )
                    
                    raise
            
            return wrapper
        return decorator
    
    def log_model_performance(
        self,
        model_name: str,
        operation: str,
        duration: float,
        input_size: int,
        output_size: Optional[int] = None,
        accuracy: Optional[float] = None,
        **metrics
    ):
        """Log model performance metrics"""
        
        performance_data = {
            "model_name": model_name,
            "operation": operation,
            "duration": duration,
            "input_size": input_size,
            "throughput": input_size / duration if duration > 0 else 0,
        }
        
        if output_size is not None:
            performance_data["output_size"] = output_size
        
        if accuracy is not None:
            performance_data["accuracy"] = accuracy
        
        # Add additional metrics
        performance_data.update(metrics)
        
        self.info("Model Performance", **performance_data)
    
    def log_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        user_id: Optional[str] = None,
        **context
    ):
        """Log API request metrics"""
        
        request_data = {
            "method": method,
            "endpoint": endpoint,
            "status_code": status_code,
            "duration": duration,
            "status": "success" if 200 <= status_code < 400 else "error",
        }
        
        if request_size is not None:
            request_data["request_size"] = request_size
        
        if response_size is not None:
            request_data["response_size"] = response_size
        
        if user_id is not None:
            request_data["user_id"] = user_id
        
        # Add additional context
        request_data.update(context)
        
        level = logging.INFO if 200 <= status_code < 400 else logging.ERROR
        self._log(level, f"API Request: {method} {endpoint}", **request_data)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        **context
    ):
        """Log security events"""
        
        security_data = {
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "category": "security",
        }
        
        if source_ip:
            security_data["source_ip"] = source_ip
        
        if user_id:
            security_data["user_id"] = user_id
        
        # Add additional context
        security_data.update(context)
        
        # Determine log level based on severity
        level_map = {
            "low": logging.INFO,
            "medium": logging.WARNING,
            "high": logging.ERROR,
            "critical": logging.CRITICAL,
        }
        
        level = level_map.get(severity.lower(), logging.WARNING)
        self._log(level, f"Security Event: {event_type}", **security_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        
        return {
            "name": self.name,
            "level": self.logger.level,
            "handlers": len(self.logger.handlers),
            "structured": self.structured,
            "metrics_enabled": self.enable_metrics,
            "tracing_enabled": self.enable_tracing,
            "async_enabled": self.enable_async,
        }


# Global logger registry
_logger_registry: Dict[str, Logger] = {}


def get_logger(
    name: str,
    level: str = "INFO",
    **kwargs
) -> Logger:
    """
    Get or create logger instance
    
    Args:
        name: Logger name
        level: Logging level
        **kwargs: Additional logger configuration
        
    Returns:
        Logger instance
    """
    
    if name not in _logger_registry:
        _logger_registry[name] = Logger(name=name, level=level, **kwargs)
    
    return _logger_registry[name]


def configure_logging(
    level: str = "INFO",
    structured: bool = True,
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    log_file: Optional[str] = None,
    **kwargs
):
    """
    Configure global logging settings
    
    Args:
        level: Default logging level
        structured: Enable structured JSON logging
        enable_metrics: Enable metrics collection
        enable_tracing: Enable distributed tracing
        log_file: Log file path
        **kwargs: Additional configuration
    """
    
    # Update default settings for new loggers
    Logger._default_settings = {
        "level": level,
        "structured": structured,
        "enable_metrics": enable_metrics,
        "enable_tracing": enable_tracing,
        "log_file": log_file,
        **kwargs
    }


# Setup structlog integration
if hasattr(structlog, 'configure'):
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )