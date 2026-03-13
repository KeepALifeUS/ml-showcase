"""
Configuration Management for Fear & Greed Index System

Centralized configuration with environment variable support
and validation.
"""

import os
from typing import Dict, List, Optional, Any, Union
from datetime import timedelta
from dataclasses import dataclass, field
from enum import Enum
import structlog
from pydantic import BaseSettings, validator, Field
import json

logger = structlog.get_logger(__name__)


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseType(Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    TIMESCALEDB = "timescaledb"
    SQLITE = "sqlite"


@dataclass
class APIRateLimits:
    """External API rate limits"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10


@dataclass
class CacheConfig:
    """Cache settings"""
    redis_url: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "500mb"
    eviction_policy: str = "allkeys-lru"


@dataclass
class DatabaseConfig:
    """Database settings"""
    type: DatabaseType = DatabaseType.POSTGRESQL
    host: str = os.environ.get("DATABASE_HOST", "localhost")
    port: int = int(os.environ.get("DATABASE_PORT", "5432"))
    database: str = os.environ.get("DATABASE_NAME", "fear_greed_index")
    username: str = os.environ.get("DATABASE_USERNAME", "postgres")
    password: str = os.environ.get("DATABASE_PASSWORD", "")
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class MLConfig:
    """ML model settings"""
    model_cache_dir: str = "./models"
    model_update_interval: int = 3600  # seconds
    batch_size: int = 32
    max_sequence_length: int = 512
    ensemble_models: List[str] = field(default_factory=lambda: [
        "finbert", "crypto_sentiment", "general_sentiment"
    ])


@dataclass
class MonitoringConfig:
    """Monitoring settings"""
    prometheus_port: int = 9090
    grafana_url: str = ""
    alert_webhook: str = ""
    log_level: str = "INFO"
    structured_logging: bool = True


class FearGreedConfig(BaseSettings):
    """
    Main configuration class for Fear & Greed Index System

    Uses Pydantic for validation and automatic loading
    from environment variables.
    """

    # === Environment Settings ===
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )

    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )

    # === Application Settings ===
    app_name: str = Field(
        default="ML Fear & Greed Index",
        description="Application name"
    )

    version: str = Field(
        default="1.0.0",
        description="Application version"
    )

    timezone: str = Field(
        default="UTC",
        description="Application timezone"
    )

    # === API Settings ===
    api_host: str = Field(
        default="0.0.0.0",
        description="API host"
    )

    api_port: int = Field(
        default=8000,
        description="API port"
    )

    api_workers: int = Field(
        default=4,
        description="Number of API workers"
    )

    api_rate_limits: APIRateLimits = Field(
        default_factory=APIRateLimits,
        description="API rate limiting configuration"
    )

    # === External API Keys ===
    twitter_bearer_token: Optional[str] = Field(
        default=None,
        description="Twitter API Bearer Token"
    )

    reddit_client_id: Optional[str] = Field(
        default=None,
        description="Reddit API Client ID"
    )

    reddit_client_secret: Optional[str] = Field(
        default=None,
        description="Reddit API Client Secret"
    )

    news_api_key: Optional[str] = Field(
        default=None,
        description="News API Key"
    )

    coingecko_api_key: Optional[str] = Field(
        default=None,
        description="CoinGecko API Key"
    )

    binance_api_key: Optional[str] = Field(
        default=None,
        description="Binance API Key"
    )

    binance_secret_key: Optional[str] = Field(
        default=None,
        description="Binance Secret Key"
    )

    # === Database Configuration ===
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )

    # === Cache Configuration ===
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )

    # === ML Configuration ===
    ml: MLConfig = Field(
        default_factory=MLConfig,
        description="Machine Learning configuration"
    )

    # === Monitoring Configuration ===
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring and observability configuration"
    )

    # === Fear & Greed Index Specific Settings ===

    # Component weights for final index calculation
    component_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "volatility": 0.25,
            "momentum": 0.25,
            "volume": 0.15,
            "social_sentiment": 0.15,
            "dominance": 0.10,
            "search_trends": 0.05,
            "surveys": 0.05
        },
        description="Weights for Fear & Greed Index components"
    )

    # Data collection intervals
    data_collection_intervals: Dict[str, int] = Field(
        default_factory=lambda: {
            "price_data": 60,        # seconds
            "social_data": 300,      # 5 minutes
            "trends_data": 3600,     # 1 hour
            "surveys_data": 14400,   # 4 hours
            "dominance_data": 1800   # 30 minutes
        },
        description="Data collection intervals in seconds"
    )

    # Historical data retention
    data_retention: Dict[str, int] = Field(
        default_factory=lambda: {
            "raw_data": 90,          # days
            "aggregated_hourly": 365, # days
            "aggregated_daily": 3650, # days (10 years)
            "models": 30             # days
        },
        description="Data retention periods in days"
    )

    # Alert thresholds
    alert_thresholds: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "fear_greed_index": {
                "extreme_fear": 20,
                "fear": 40,
                "greed": 60,
                "extreme_greed": 80
            },
            "volatility": {
                "high": 100,     # % annual
                "extreme": 200   # % annual
            },
            "volume": {
                "high_ratio": 3.0,    # 3x average
                "extreme_ratio": 5.0  # 5x average
            }
        },
        description="Alert threshold configurations"
    )

    # Supported cryptocurrencies
    supported_symbols: List[str] = Field(
        default_factory=lambda: [
            "BTC", "ETH", "ADA", "DOT", "LINK", "UNI", "AAVE", "COMP"
        ],
        description="Supported cryptocurrency symbols"
    )

    # Timeframes for analysis
    timeframes: List[str] = Field(
        default_factory=lambda: [
            "1h", "4h", "1d", "1w", "1M"
        ],
        description="Supported timeframes for analysis"
    )

    # Circuit breaker settings
    circuit_breaker: Dict[str, Any] = Field(
        default_factory=lambda: {
            "failure_threshold": 5,
            "recovery_timeout": 30,
            "expected_exception": "Exception"
        },
        description="Circuit breaker configuration"
    )

    # Security settings
    security: Dict[str, Any] = Field(
        default_factory=lambda: {
            "api_key_header": "X-API-Key",
            "rate_limit_header": "X-RateLimit-Remaining",
            "cors_origins": ["*"],
            "cors_methods": ["GET", "POST"],
            "cors_headers": ["*"]
        },
        description="Security configuration"
    )

    class Config:
        """Pydantic configuration"""
        env_prefix = "FEAR_GREED_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True

        @classmethod
        def schema_extra(cls, schema: Dict[str, Any], model_type) -> None:
            """Add examples to schema"""
            schema["examples"] = [
                {
                    "environment": "production",
                    "api_port": 8000,
                    "database": {
                        "type": "postgresql",
                        "host": "localhost",
                        "port": 5432
                    }
                }
            ]

    @validator("component_weights")
    def validate_component_weights(cls, v):
        """Validate component weights"""
        total_weight = sum(v.values())
        if abs(total_weight - 1.0) > 0.001:  # Allow small margin of error
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")

        for component, weight in v.items():
            if not 0 <= weight <= 1:
                raise ValueError(f"Weight for {component} must be between 0 and 1, got {weight}")

        return v

    @validator("data_collection_intervals")
    def validate_intervals(cls, v):
        """Validate data collection intervals"""
        for interval_name, interval_seconds in v.items():
            if interval_seconds < 60:  # Minimum 1 minute
                raise ValueError(f"Interval {interval_name} must be at least 60 seconds")
            if interval_seconds > 86400:  # Maximum 1 day
                raise ValueError(f"Interval {interval_name} must not exceed 86400 seconds")
        return v

    @validator("supported_symbols")
    def validate_symbols(cls, v):
        """Validate supported symbols"""
        if not v:
            raise ValueError("At least one symbol must be supported")

        for symbol in v:
            if not isinstance(symbol, str) or len(symbol) < 2:
                raise ValueError(f"Invalid symbol format: {symbol}")

        return v

    def get_database_url(self) -> str:
        """Get database URL"""
        db = self.database
        if db.type == DatabaseType.POSTGRESQL or db.type == DatabaseType.TIMESCALEDB:
            return f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
        elif db.type == DatabaseType.SQLITE:
            return f"sqlite:///{db.database}.db"
        else:
            raise ValueError(f"Unsupported database type: {db.type}")

    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.cache.redis_url

    def is_production(self) -> bool:
        """Check if production environment"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if development environment"""
        return self.environment == Environment.DEVELOPMENT

    def get_component_weight(self, component_name: str) -> float:
        """Get component weight"""
        return self.component_weights.get(component_name, 0.0)

    def get_data_collection_interval(self, data_type: str) -> int:
        """Get data collection interval in seconds"""
        return self.data_collection_intervals.get(data_type, 3600)

    def get_alert_threshold(self, metric_type: str, threshold_name: str) -> Optional[float]:
        """Get alert threshold"""
        return self.alert_thresholds.get(metric_type, {}).get(threshold_name)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file"""
        try:
            config_dict = self.dict()
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info("Configuration saved", file_path=file_path)
        except Exception as e:
            logger.error("Failed to save configuration", file_path=file_path, error=str(e))
            raise

    @classmethod
    def load_from_file(cls, file_path: str) -> 'FearGreedConfig':
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            logger.info("Configuration loaded", file_path=file_path)
            return cls(**config_dict)
        except Exception as e:
            logger.error("Failed to load configuration", file_path=file_path, error=str(e))
            raise

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Update component weights with validation"""
        # Create a copy of current weights
        updated_weights = self.component_weights.copy()
        updated_weights.update(new_weights)

        # Validate through Pydantic validator
        validated_weights = self.__class__.validate_component_weights(updated_weights)
        self.component_weights = validated_weights

        logger.info("Component weights updated", new_weights=new_weights)

    def get_effective_config(self) -> Dict[str, Any]:
        """Get effective configuration for logging"""
        config = self.dict()

        # Mask sensitive data
        sensitive_fields = [
            'twitter_bearer_token', 'reddit_client_secret', 'binance_secret_key',
            'database.password'
        ]

        for field in sensitive_fields:
            if '.' in field:
                section, key = field.split('.')
                if section in config and key in config[section]:
                    config[section][key] = "***MASKED***"
            else:
                if field in config:
                    config[field] = "***MASKED***"

        return config


# Global config instance
_config_instance: Optional[FearGreedConfig] = None


def get_config() -> FearGreedConfig:
    """
    Get global configuration instance (Singleton pattern)

    Returns:
        FearGreedConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = FearGreedConfig()
        logger.info("Configuration initialized",
                   environment=_config_instance.environment,
                   version=_config_instance.version)

    return _config_instance


def set_config(config: FearGreedConfig) -> None:
    """
    Set global configuration instance

    Args:
        config: FearGreedConfig instance
    """
    global _config_instance
    _config_instance = config
    logger.info("Configuration updated globally")


def reset_config() -> None:
    """Reset global configuration"""
    global _config_instance
    _config_instance = None
    logger.info("Configuration reset")
