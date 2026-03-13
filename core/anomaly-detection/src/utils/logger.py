"""
📝 Structured Logger

Enterprise logging system with OpenTelemetry integration.
"""

import structlog
from typing import Dict, Any

class Logger:
 @staticmethod
 def configure:
 structlog.configure(
 processors=[
 structlog.stdlib.filter_by_level,
 structlog.stdlib.add_logger_name,
 structlog.stdlib.add_log_level,
 structlog.stdlib.PositionalArgumentsFormatter,
 structlog.processors.TimeStamper(fmt="iso"),
 structlog.processors.StackInfoRenderer,
 structlog.processors.format_exc_info,
 structlog.processors.UnicodeDecoder,
 structlog.processors.JSONRenderer
 ],
 context_class=dict,
 logger_factory=structlog.stdlib.LoggerFactory,
 wrapper_class=structlog.stdlib.BoundLogger,
 cache_logger_on_first_use=True,
 )

 @staticmethod
 def get_logger(name: str):
 return structlog.get_logger(name)