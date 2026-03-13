"""
Configuration system for Order Flow Detection
Should enterprise patterns for configuration and DI
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import os
from dataclasses import dataclass, field
from enum import Enum
import json
from pydantic import BaseSettings, validator
from pydantic_settings import BaseSettings as PydanticSettings

class Environment(str, Enum):
    """Environment system"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    """Levels logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Configuration base data"""
    host: str = "localhost"
    port: int = 5432
    database: str = "ml-framework_order_flow"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Configuration Redis"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

@dataclass
class ExchangeConfig:
    """Configuration exchange"""
    name: str
    api_key: str
    api_secret: str
    sandbox: bool = False
    rate_limit: int = 1000
    timeout: int = 30
    enable_rate_limit: bool = True
    verbose: bool = False

@dataclass
class MLModelConfig:
    """Configuration ML models"""
    model_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    batch_size: int = 1024
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    use_gpu: bool = True
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class OrderFlowConfig:
    """Configuration analysis order flow"""
    tick_size: float = 0.01
    depth_levels: int = 20
    imbalance_threshold: float = 0.7
    absorption_ratio: float = 2.0
    exhaustion_threshold: float = 0.3
    iceberg_detection_window: int = 100
    spoofing_timeout: float = 5.0
    min_volume_threshold: float = 1000.0
    update_frequency_ms: int = 100

@dataclass
class VolumeProfileConfig:
    """Configuration volume profile"""
    num_price_levels: int = 100
    value_area_percentage: float = 0.68
    poc_threshold: float = 0.05
    cluster_min_volume: float = 500.0
    vwap_period: int = 1440  # minutes
    tpo_period: int = 30  # minutes
    profile_type: str = "volume"  # volume, ticks, delta

@dataclass
class RealTimeConfig:
    """Configuration real-time processing"""
    websocket_timeout: int = 60
    reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    heartbeat_interval: int = 30
    max_queue_size: int = 10000
    buffer_size: int = 1024 * 1024  # 1MB
    compression: bool = True
    enable_ping_pong: bool = True

@dataclass
class APIConfig:
    """Configuration API servers"""
    rest_host: str = "0.0.0.0"
    rest_port: int = 8000
    websocket_port: int = 8001
    grpc_port: int = 50051
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=list)
    enable_docs: bool = True
    docs_url: str = "/docs"

class OrderFlowSettings(PydanticSettings):
    """
    Main configuration system Order Flow Detection
    Uses Pydantic for validation and enterprise patterns
    """
    
    # Main settings
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    app_name: str = "OrderFlowDetection"
    version: str = "1.0.0"
    
    # Paths
    data_dir: Path = Path("data")
    logs_dir: Path = Path("logs")
    cache_dir: Path = Path("cache")
    temp_dir: Path = Path("temp")
    
    # Components configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml_models: MLModelConfig = field(default_factory=MLModelConfig)
    order_flow: OrderFlowConfig = field(default_factory=OrderFlowConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    realtime: RealTimeConfig = field(default_factory=RealTimeConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Exchanges configuration
    exchanges: Dict[str, ExchangeConfig] = field(default_factory=dict)
    
    # Symbols to track
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
    
    # Security settings
    jwt_secret_key: str = "your-jwt-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    
    # Performance settings
    max_concurrent_requests: int = 100
    worker_threads: int = 4
    enable_profiling: bool = False
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    class Config:
        env_file = ".env"
        env_prefix = "ORDER_FLOW_"
        case_sensitive = False
        
    @validator('data_dir', 'logs_dir', 'cache_dir', 'temp_dir', pre=True)
    def create_directories(cls, v):
        """Creation directories if their no"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator('exchanges')
    def validate_exchanges(cls, v):
        """Validation configuration exchanges"""
        for name, config in v.items():
            if not config.api_key or not config.api_secret:
                raise ValueError(f"API credentials missing for exchange {name}")
        return v
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Retrieval configuration exchange"""
        return self.exchanges.get(exchange_name)
    
    def add_exchange(self, name: str, config: ExchangeConfig) -> None:
        """Addition configuration exchange"""
        self.exchanges[name] = config
    
    def get_database_url(self) -> str:
        """Retrieval URL base data"""
        db = self.database
        return f"postgresql+asyncpg://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    def get_redis_url(self) -> str:
        """Retrieval URL Redis"""
        redis = self.redis
        password_part = f":{redis.password}@" if redis.password else ""
        return f"redis://{password_part}{redis.host}:{redis.port}/{redis.database}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary"""
        return self.dict()
    
    def save_to_file(self, file_path: Path) -> None:
        """Saving configuration in file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> 'OrderFlowSettings':
        """Loading configuration from file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Global instance configuration
_settings: Optional[OrderFlowSettings] = None

def get_settings() -> OrderFlowSettings:
    """
    Retrieval global instance settings (Singleton pattern)
    Dependency Injection pattern
    """
    global _settings
    if _settings is None:
        _settings = OrderFlowSettings()
    return _settings

def update_settings(**kwargs) -> None:
    """Update global settings"""
    global _settings
    if _settings is None:
        _settings = OrderFlowSettings(**kwargs)
    else:
        for key, value in kwargs.items():
            if hasattr(_settings, key):
                setattr(_settings, key, value)

def reset_settings() -> None:
    """Reset global settings (for testing)"""
    global _settings
    _settings = None

#  Configuration Provider pattern
class ConfigurationProvider:
    """
    Provider configuration next enterprise patterns
    """
    
    def __init__(self, settings: OrderFlowSettings):
        self._settings = settings
        self._watchers: List[callable] = []
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Retrieval values configuration"""
        try:
            return getattr(self._settings, key, default)
        except AttributeError:
            return default
    
    def set_value(self, key: str, value: Any) -> None:
        """Installation values configuration"""
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
            self._notify_watchers(key, value)
    
    def watch(self, callback: callable) -> None:
        """Subscription on changes configuration"""
        self._watchers.append(callback)
    
    def _notify_watchers(self, key: str, value: Any) -> None:
        """Notification subscribers about changes"""
        for watcher in self._watchers:
            try:
                watcher(key, value)
            except Exception as e:
                # Log error but don't break other watchers
                print(f"Config watcher error: {e}")

#  Factory pattern for configurations
class ConfigurationFactory:
    """Factory for creation configurations different environments"""
    
    @staticmethod
    def create_development_config() -> OrderFlowSettings:
        """Configuration for development"""
        return OrderFlowSettings(
            environment=Environment.DEVELOPMENT,
            debug=True,
            log_level=LogLevel.DEBUG
        )
    
    @staticmethod
    def create_production_config() -> OrderFlowSettings:
        """Configuration for production"""
        return OrderFlowSettings(
            environment=Environment.PRODUCTION,
            debug=False,
            log_level=LogLevel.WARNING
        )
    
    @staticmethod
    def create_testing_config() -> OrderFlowSettings:
        """Configuration for testing"""
        return OrderFlowSettings(
            environment=Environment.TESTING,
            debug=True,
            log_level=LogLevel.DEBUG,
            database=DatabaseConfig(database="test_order_flow"),
            redis=RedisConfig(database=1)
        )
    
    @staticmethod
    def create_from_environment() -> OrderFlowSettings:
        """Creation configuration on basis variables environment"""
        env = os.getenv("ENVIRONMENT", "development").lower()
        
        if env == "production":
            return ConfigurationFactory.create_production_config()
        elif env == "testing":
            return ConfigurationFactory.create_testing_config()
        else:
            return ConfigurationFactory.create_development_config()

# Utility functions for work with configuration
def is_development() -> bool:
    """Validation on mode development"""
    return get_settings().environment == Environment.DEVELOPMENT

def is_production() -> bool:
    """Validation on production mode"""
    return get_settings().environment == Environment.PRODUCTION

def is_testing() -> bool:
    """Validation on test mode"""
    return get_settings().environment == Environment.TESTING

def get_log_level() -> str:
    """Retrieval level logging"""
    return get_settings().log_level.value

def get_database_url() -> str:
    """Retrieval URL base data"""
    return get_settings().get_database_url()

def get_redis_url() -> str:
    """Retrieval URL Redis"""
    return get_settings().get_redis_url()

# Export main configuration components
__all__ = [
    'OrderFlowSettings',
    'Environment',
    'LogLevel',
    'DatabaseConfig',
    'RedisConfig',
    'ExchangeConfig',
    'MLModelConfig',
    'OrderFlowConfig',
    'VolumeProfileConfig',
    'RealTimeConfig',
    'APIConfig',
    'get_settings',
    'update_settings',
    'reset_settings',
    'ConfigurationProvider',
    'ConfigurationFactory',
    'is_development',
    'is_production',
    'is_testing',
    'get_log_level',
    'get_database_url',
    'get_redis_url'
]