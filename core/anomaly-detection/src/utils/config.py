"""
⚙️ Configuration Management

Centralized configuration system for anomaly detection components.
Supports environment-based configs, validation, and hot reloading.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class DatabaseConfig:
 """Database configuration."""
 host: str = "localhost"
 port: int = 5432
 database: str = "ml-framework_anomalies"
 username: str = "ml-framework"
 password: str = ""
 max_connections: int = 10

@dataclass
class RedisConfig:
 """Redis configuration."""
 host: str = "localhost"
 port: int = 6379
 db: int = 0
 password: Optional[str] = None
 max_connections: int = 10

@dataclass
class APIConfig:
 """API configuration."""
 host: str = "0.0.0.0"
 port: int = 8000
 workers: int = 4
 reload: bool = False
 log_level: str = "info"

@dataclass
class MonitoringConfig:
 """Monitoring configuration."""
 enable_prometheus: bool = True
 prometheus_port: int = 9090
 enable_jaeger: bool = True
 jaeger_endpoint: str = "http://localhost:14268/api/traces"

@dataclass
class CryptoConfig:
 """Crypto-specific configuration."""
 supported_exchanges: list = None
 default_contamination: float = 0.05
 volatility_threshold: float = 1.0

 def __post_init__(self):
 if self.supported_exchanges is None:
 self.supported_exchanges = ["binance", "coinbase", "kraken"]

class Config:
 """Central configuration manager."""

 _instance = None
 _config = None

 def __new__(cls):
 if cls._instance is None:
 cls._instance = super.__new__(cls)
 cls._instance._load_config
 return cls._instance

 def _load_config(self):
 """Load configuration from multiple sources."""
 self._config = {
 "database": DatabaseConfig,
 "redis": RedisConfig,
 "api": APIConfig,
 "monitoring": MonitoringConfig,
 "crypto": CryptoConfig
 }

 # Override with environment variables
 self._load_from_env

 # Override with config files
 self._load_from_files

 logger.info("Configuration loaded successfully")

 def _load_from_env(self):
 """Load configuration from environment variables."""
 # Database
 if os.getenv("DB_HOST"):
 self._config["database"].host = os.getenv("DB_HOST")
 if os.getenv("DB_PORT"):
 self._config["database"].port = int(os.getenv("DB_PORT"))
 if os.getenv("DB_NAME"):
 self._config["database"].database = os.getenv("DB_NAME")
 if os.getenv("DB_USER"):
 self._config["database"].username = os.getenv("DB_USER")
 if os.getenv("DB_PASSWORD"):
 self._config["database"].password = os.getenv("DB_PASSWORD")

 # Redis
 if os.getenv("REDIS_HOST"):
 self._config["redis"].host = os.getenv("REDIS_HOST")
 if os.getenv("REDIS_PORT"):
 self._config["redis"].port = int(os.getenv("REDIS_PORT"))
 if os.getenv("REDIS_PASSWORD"):
 self._config["redis"].password = os.getenv("REDIS_PASSWORD")

 # API
 if os.getenv("API_HOST"):
 self._config["api"].host = os.getenv("API_HOST")
 if os.getenv("API_PORT"):
 self._config["api"].port = int(os.getenv("API_PORT"))
 if os.getenv("API_WORKERS"):
 self._config["api"].workers = int(os.getenv("API_WORKERS"))

 def _load_from_files(self):
 """Load configuration from config files."""
 config_paths = [
 Path("config.yaml"),
 Path("config.yml"),
 Path("config.json"),
 Path("/etc/ml-framework/anomaly-detection.yaml")
 ]

 for path in config_paths:
 if path.exists:
 try:
 if path.suffix in [".yaml", ".yml"]:
 with open(path, 'r') as f:
 config_data = yaml.safe_load(f)
 elif path.suffix == ".json":
 with open(path, 'r') as f:
 config_data = json.load(f)
 else:
 continue

 self._merge_config(config_data)
 logger.info(f"Loaded config from {path}")
 break

 except Exception as e:
 logger.warning(f"Failed to load config from {path}: {e}")

 def _merge_config(self, config_data: Dict[str, Any]):
 """Merge external config with current config."""
 for section, data in config_data.items:
 if section in self._config and isinstance(data, dict):
 for key, value in data.items:
 if hasattr(self._config[section], key):
 setattr(self._config[section], key, value)

 @property
 def database(self) -> DatabaseConfig:
 return self._config["database"]

 @property
 def redis(self) -> RedisConfig:
 return self._config["redis"]

 @property
 def api(self) -> APIConfig:
 return self._config["api"]

 @property
 def monitoring(self) -> MonitoringConfig:
 return self._config["monitoring"]

 @property
 def crypto(self) -> CryptoConfig:
 return self._config["crypto"]

 def get(self, key: str, default=None):
 """Get configuration value by key."""
 keys = key.split(".")
 value = self._config

 for k in keys:
 if isinstance(value, dict) and k in value:
 value = value[k]
 elif hasattr(value, k):
 value = getattr(value, k)
 else:
 return default

 return value

 def set(self, key: str, value: Any):
 """Set configuration value by key."""
 keys = key.split(".")
 target = self._config

 for k in keys[:-1]:
 if k in target:
 target = target[k]
 else:
 return False

 if hasattr(target, keys[-1]):
 setattr(target, keys[-1], value)
 return True
 elif isinstance(target, dict):
 target[keys[-1]] = value
 return True

 return False

 def to_dict(self) -> Dict[str, Any]:
 """Convert configuration to dictionary."""
 result = {}
 for key, config_obj in self._config.items:
 result[key] = asdict(config_obj)
 return result

 def update(self, config_dict: Dict[str, Any]):
 """Update configuration with dictionary."""
 self._merge_config(config_dict)
 logger.info("Configuration updated")

 @classmethod
 def reload(cls):
 """Reload configuration from sources."""
 if cls._instance:
 cls._instance._load_config
 logger.info("Configuration reloaded")

# Global config instance
config = Config