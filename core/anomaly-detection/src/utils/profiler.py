"""
⚡ Performance Profiler

Performance monitoring and profiling utilities.
"""

import time
import psutil
import functools
from typing import Dict, Any, Callable
import structlog

logger = structlog.get_logger(__name__)

class Profiler:
 """Performance profiling utilities."""

 @staticmethod
 def profile_function(func: Callable) -> Callable:
 """Decorator to profile function execution."""
 @functools.wraps(func)
 def wrapper(*args, **kwargs):
 start_time = time.time
 start_memory = psutil.Process.memory_info.rss / 1024 / 1024 # MB

 try:
 result = func(*args, **kwargs)
 success = True
 error = None
 except Exception as e:
 result = None
 success = False
 error = str(e)
 raise
 finally:
 end_time = time.time
 end_memory = psutil.Process.memory_info.rss / 1024 / 1024 # MB

 execution_time = end_time - start_time
 memory_used = end_memory - start_memory

 logger.info(
 f"Function {func.__name__} profiled",
 execution_time_seconds=execution_time,
 memory_used_mb=memory_used,
 success=success,
 error=error
 )

 return result
 return wrapper

 @staticmethod
 def get_system_stats -> Dict[str, Any]:
 """Get current system statistics."""
 return {
 "cpu_percent": psutil.cpu_percent,
 "memory_percent": psutil.virtual_memory.percent,
 "disk_usage_percent": psutil.disk_usage('/').percent,
 "available_memory_gb": psutil.virtual_memory.available / 1024 / 1024 / 1024
 }