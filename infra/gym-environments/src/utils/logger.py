"""
Structured Logging for Trading Environments
enterprise patterns for production logging

Features:
- Structured JSON logging
- Trading event tracking
- Performance monitoring integration
- Error tracking and alerting
- Compliance logging
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import contextvars


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Trading event types"""
    TRADE_EXECUTION = "trade_execution"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_ALERT = "risk_alert"
    MARKET_DATA_UPDATE = "market_data_update"
    ENVIRONMENT_RESET = "environment_reset"
    ENVIRONMENT_STEP = "environment_step"
    MODEL_INFERENCE = "model_inference"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_EVENT = "error_event"


@dataclass
class TradingEvent:
    """Structured trading event"""
    event_type: EventType
    timestamp: float
    environment_id: str
    step: Optional[int] = None
    asset: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class StructuredLogger:
    """
    Advanced structured logger for trading environments
    
    Provides comprehensive logging with trading-specific features
    """
    
    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        environment_id: Optional[str] = None,
        enable_performance_logging: bool = True
    ):
        self.name = name
        self.level = level
        self.environment_id = environment_id or f"env_{int(time.time())}"
        self.enable_performance_logging = enable_performance_logging
        
        # Initialize Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        # Create formatted handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
        
        # Event tracking
        self.events = []
        self.performance_metrics = {}
        
        # Context variables
        self.context = contextvars.ContextVar('trading_context', default={})
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message"""
        self._log(LogLevel.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message"""
        self._log(LogLevel.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """Log error message"""
        if exc_info:
            extra = extra or {}
            extra["traceback"] = traceback.format_exc()
        self._log(LogLevel.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, extra)
    
    def _log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Internal logging method"""
        
        # Build structured log record
        log_data = {
            "timestamp": time.time(),
            "level": level.value,
            "environment_id": self.environment_id,
            "message": message,
            "context": self.context.get({})
        }
        
        if extra:
            log_data.update(extra)
        
        # Log to Python logger
        python_level = getattr(logging, level.value)
        self.logger.log(python_level, message, extra=log_data)
    
    def log_event(self, event: TradingEvent) -> None:
        """Log structured trading event"""
        
        self.events.append(event)
        
        # Convert to log message
        message = f"{event.event_type.value}"
        if event.asset:
            message += f" for {event.asset}"
        if event.step is not None:
            message += f" at step {event.step}"
        
        extra = {
            "event": asdict(event),
            "event_type": event.event_type.value
        }
        
        self.info(message, extra)
    
    def log_trade_execution(
        self,
        asset: str,
        side: str,
        quantity: float,
        price: float,
        fees: float,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log trade execution event"""
        
        event = TradingEvent(
            event_type=EventType.TRADE_EXECUTION,
            timestamp=time.time(),
            environment_id=self.environment_id,
            step=step,
            asset=asset,
            data={
                "side": side,
                "quantity": quantity,
                "price": price,
                "fees": fees,
                "trade_value": quantity * price
            },
            metadata=metadata
        )
        
        self.log_event(event)
    
    def log_portfolio_update(
        self,
        portfolio_value: float,
        balance: float,
        positions: Dict[str, float],
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log portfolio update event"""
        
        event = TradingEvent(
            event_type=EventType.PORTFOLIO_UPDATE,
            timestamp=time.time(),
            environment_id=self.environment_id,
            step=step,
            data={
                "portfolio_value": portfolio_value,
                "balance": balance,
                "positions": positions,
                "total_exposure": sum(abs(pos) for pos in positions.values())
            },
            metadata=metadata
        )
        
        self.log_event(event)
    
    def log_risk_alert(
        self,
        risk_type: str,
        risk_level: str,
        risk_value: float,
        threshold: float,
        step: Optional[int] = None,
        asset: Optional[str] = None
    ) -> None:
        """Log risk alert event"""
        
        event = TradingEvent(
            event_type=EventType.RISK_ALERT,
            timestamp=time.time(),
            environment_id=self.environment_id,
            step=step,
            asset=asset,
            data={
                "risk_type": risk_type,
                "risk_level": risk_level,
                "risk_value": risk_value,
                "threshold": threshold,
                "breach_percentage": (risk_value - threshold) / threshold * 100
            }
        )
        
        # Log as warning or error based on severity
        if risk_level in ["HIGH", "CRITICAL"]:
            self.error(f"Risk alert: {risk_type} = {risk_value:.4f} exceeds threshold {threshold:.4f}")
        else:
            self.warning(f"Risk alert: {risk_type} = {risk_value:.4f} approaching threshold {threshold:.4f}")
        
        self.log_event(event)
    
    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log performance metric"""
        
        # Track metrics history
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append({
            "value": metric_value,
            "timestamp": time.time(),
            "step": step
        })
        
        if self.enable_performance_logging:
            event = TradingEvent(
                event_type=EventType.PERFORMANCE_METRIC,
                timestamp=time.time(),
                environment_id=self.environment_id,
                step=step,
                data={
                    "metric_name": metric_name,
                    "metric_value": metric_value
                },
                metadata=metadata
            )
            
            self.log_event(event)
    
    def set_context(self, **kwargs) -> None:
        """Set logging context"""
        current_context = self.context.get({})
        current_context.update(kwargs)
        self.context.set(current_context)
    
    def clear_context(self) -> None:
        """Clear logging context"""
        self.context.set({})
    
    def get_events(self, event_type: Optional[EventType] = None, limit: Optional[int] = None) -> List[TradingEvent]:
        """Get logged events"""
        
        events = self.events
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        summary = {}
        
        for metric_name, metric_history in self.performance_metrics.items():
            values = [m["value"] for m in metric_history]
            
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5 if len(values) > 1 else 0.0
                }
        
        return summary
    
    def export_events(self, filename: str) -> None:
        """Export events to JSON file"""
        
        events_data = [asdict(event) for event in self.events]
        
        with open(filename, 'w') as f:
            json.dump(events_data, f, indent=2, default=str)
    
    def reset(self) -> None:
        """Reset logger state"""
        
        self.events.clear()
        self.performance_metrics.clear()
        self.clear_context()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logs"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        
        log_data = {
            "timestamp": record.created,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra data if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# Factory functions
def create_environment_logger(
    environment_id: str,
    level: LogLevel = LogLevel.INFO
) -> StructuredLogger:
    """Create logger for trading environment"""
    
    return StructuredLogger(
        name=f"trading_env.{environment_id}",
        level=level,
        environment_id=environment_id,
        enable_performance_logging=True
    )


def create_agent_logger(
    agent_id: str,
    level: LogLevel = LogLevel.INFO
) -> StructuredLogger:
    """Create logger for trading agent"""
    
    return StructuredLogger(
        name=f"trading_agent.{agent_id}",
        level=level,
        environment_id=agent_id,
        enable_performance_logging=True
    )


__all__ = [
    "LogLevel",
    "EventType",
    "TradingEvent",
    "StructuredLogger",
    "StructuredFormatter",
    "create_environment_logger",
    "create_agent_logger"
]