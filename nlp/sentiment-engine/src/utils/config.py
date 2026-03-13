"""
Configuration management for ML-Framework ML Sentiment Engine

Enterprise-grade configuration with validation, environment support and .
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field, validator


class Environment(str, Enum):
 """Supported deployment environments"""
 DEVELOPMENT = "development"
 TESTING = "testing"
 STAGING = "staging"
 PRODUCTION = "production"


class LogLevel(str, Enum):
 """Log levels"""
 DEBUG = "DEBUG"
 INFO = "INFO"
 WARNING = "WARNING"
 ERROR = "ERROR"
 CRITICAL = "CRITICAL"


class DatabaseConfig(BaseSettings):
 """Database configuration"""

 host: str = Field(default="localhost", env="DB_HOST")
 port: int = Field(default=5432, env="DB_PORT")
 username: str = Field(env="DB_USERNAME")
 password: str = Field(env="DB_PASSWORD")
 database: str = Field(default="ml-framework_sentiment", env="DB_DATABASE")
 pool_size: int = Field(default=10, env="DB_POOL_SIZE")
 max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")

 @property
 def url(self) -> str:
 """Create database connection URL"""
 return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

 class Config:
 env_prefix = "DB_"


class RedisConfig(BaseSettings):
 """Redis configuration for caching"""

 host: str = Field(default="localhost", env="REDIS_HOST")
 port: int = Field(default=6379, env="REDIS_PORT")
 password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
 db: int = Field(default=0, env="REDIS_DB")
 max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")

 @property
 def url(self) -> str:
 """Create Redis connection URL"""
 if self.password:
 return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
 return f"redis://{self.host}:{self.port}/{self.db}"

 class Config:
 env_prefix = "REDIS_"


class KafkaConfig(BaseSettings):
 """Kafka configuration for real-time streaming"""

 bootstrap_servers: List[str] = Field(default=["localhost:9092"], env="KAFKA_BOOTSTRAP_SERVERS")
 consumer_group_id: str = Field(default="sentiment-engine", env="KAFKA_CONSUMER_GROUP_ID")
 auto_offset_reset: str = Field(default="latest", env="KAFKA_AUTO_OFFSET_RESET")
 enable_auto_commit: bool = Field(default=True, env="KAFKA_ENABLE_AUTO_COMMIT")
 max_poll_records: int = Field(default=500, env="KAFKA_MAX_POLL_RECORDS")

 @validator("bootstrap_servers", pre=True)
 def parse_bootstrap_servers(cls, v):
 """Parse bootstrap servers from string"""
 if isinstance(v, str):
 return v.split(",")
 return v

 class Config:
 env_prefix = "KAFKA_"


class APIConfig(BaseSettings):
 """API endpoint configuration"""

 host: str = Field(default="0.0.0.0", env="API_HOST")
 port: int = Field(default=8003, env="API_PORT")
 workers: int = Field(default=4, env="API_WORKERS")
 reload: bool = Field(default=False, env="API_RELOAD")
 debug: bool = Field(default=False, env="API_DEBUG")

 # Rate limiting
 rate_limit_requests: int = Field(default=1000, env="API_RATE_LIMIT_REQUESTS")
 rate_limit_window: int = Field(default=60, env="API_RATE_LIMIT_WINDOW")

 # CORS
 cors_origins: List[str] = Field(default=["*"], env="API_CORS_ORIGINS")

 @validator("cors_origins", pre=True)
 def parse_cors_origins(cls, v):
 """Parse CORS origins from string"""
 if isinstance(v, str):
 return v.split(",")
 return v

 class Config:
 env_prefix = "API_"


class MLConfig(BaseSettings):
 """ML model configuration"""

 # Model paths
 models_dir: str = Field(default="./models", env="ML_MODELS_DIR")
 cache_dir: str = Field(default="./cache", env="ML_CACHE_DIR")

 # FinBERT configuration
 finbert_model: str = Field(default="ProsusAI/finbert", env="ML_FINBERT_MODEL")
 finbert_max_length: int = Field(default=512, env="ML_FINBERT_MAX_LENGTH")

 # Ensemble configuration
 ensemble_weights: Dict[str, float] = Field(
 default={
 "finbert": 0.4,
 "vader": 0.2,
 "textblob": 0.2,
 "crypto_specific": 0.2
 },
 env="ML_ENSEMBLE_WEIGHTS"
 )

 # Performance settings
 batch_size: int = Field(default=32, env="ML_BATCH_SIZE")
 num_workers: int = Field(default=4, env="ML_NUM_WORKERS")
 use_gpu: bool = Field(default=False, env="ML_USE_GPU")

 @validator("ensemble_weights", pre=True)
 def parse_ensemble_weights(cls, v):
 """Parse ensemble weights from JSON string"""
 if isinstance(v, str):
 import json
 return json.loads(v)
 return v

 class Config:
 env_prefix = "ML_"


class SocialConfig(BaseSettings):
 """Social media API configuration"""

 # Twitter/X API
 twitter_bearer_token: Optional[str] = Field(env="TWITTER_BEARER_TOKEN")
 twitter_api_key: Optional[str] = Field(env="TWITTER_API_KEY")
 twitter_api_secret: Optional[str] = Field(env="TWITTER_API_SECRET")
 twitter_access_token: Optional[str] = Field(env="TWITTER_ACCESS_TOKEN")
 twitter_access_token_secret: Optional[str] = Field(env="TWITTER_ACCESS_TOKEN_SECRET")

 # Reddit API
 reddit_client_id: Optional[str] = Field(env="REDDIT_CLIENT_ID")
 reddit_client_secret: Optional[str] = Field(env="REDDIT_CLIENT_SECRET")
 reddit_user_agent: str = Field(default="ML-Framework-SentimentEngine/1.0", env="REDDIT_USER_AGENT")

 # Telegram API
 telegram_api_id: Optional[str] = Field(env="TELEGRAM_API_ID")
 telegram_api_hash: Optional[str] = Field(env="TELEGRAM_API_HASH")
 telegram_phone: Optional[str] = Field(env="TELEGRAM_PHONE")

 # Discord API
 discord_token: Optional[str] = Field(env="DISCORD_TOKEN")
 discord_channels: List[str] = Field(default=[], env="DISCORD_CHANNELS")

 @validator("discord_channels", pre=True)
 def parse_discord_channels(cls, v):
 """Parse Discord channels from string"""
 if isinstance(v, str):
 return v.split(",") if v else []
 return v

 class Config:
 env_prefix = "SOCIAL_"


class MonitoringConfig(BaseSettings):
 """Monitoring and metrics configuration"""

 # Prometheus metrics
 metrics_port: int = Field(default=8004, env="METRICS_PORT")
 metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")

 # OpenTelemetry tracing
 jaeger_endpoint: Optional[str] = Field(env="JAEGER_ENDPOINT")
 tracing_enabled: bool = Field(default=False, env="TRACING_ENABLED")

 # Alerting
 alert_webhook_url: Optional[str] = Field(env="ALERT_WEBHOOK_URL")
 slack_webhook_url: Optional[str] = Field(env="SLACK_WEBHOOK_URL")

 class Config:
 env_prefix = "MONITORING_"


class SentimentEngineConfig(BaseSettings):
 """Main Sentiment Engine configuration"""

 # Environment
 environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
 debug: bool = Field(default=False, env="DEBUG")

 # Logging
 log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
 log_format: str = Field(default="json", env="LOG_FORMAT")

 # Service configuration
 service_name: str = Field(default="ml-framework-sentiment-engine", env="SERVICE_NAME")
 service_version: str = Field(default="1.0.0", env="SERVICE_VERSION")

 # Component configurations
 database: DatabaseConfig = DatabaseConfig
 redis: RedisConfig = RedisConfig
 kafka: KafkaConfig = KafkaConfig
 api: APIConfig = APIConfig
 ml: MLConfig = MLConfig
 social: SocialConfig = SocialConfig
 monitoring: MonitoringConfig = MonitoringConfig

 # Processing settings
 processing_batch_size: int = Field(default=100, env="PROCESSING_BATCH_SIZE")
 processing_interval: int = Field(default=60, env="PROCESSING_INTERVAL") # seconds
 max_concurrent_tasks: int = Field(default=10, env="MAX_CONCURRENT_TASKS")

 # Circuit breaker settings
 circuit_breaker_failure_threshold: int = Field(default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD")
 circuit_breaker_recovery_timeout: int = Field(default=60, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT")
 circuit_breaker_expected_exception: str = Field(default="Exception", env="CIRCUIT_BREAKER_EXPECTED_EXCEPTION")

 @validator("environment", pre=True)
 def validate_environment(cls, v):
 """Validate environment"""
 if isinstance(v, str):
 return Environment(v.lower)
 return v

 @validator("log_level", pre=True)
 def validate_log_level(cls, v):
 """Validate log level"""
 if isinstance(v, str):
 return LogLevel(v.upper)
 return v

 @property
 def is_development(self) -> bool:
 """Check for development environment"""
 return self.environment == Environment.DEVELOPMENT

 @property
 def is_production(self) -> bool:
 """Check for production environment"""
 return self.environment == Environment.PRODUCTION

 class Config:
 env_file = ".env"
 env_file_encoding = "utf-8"
 case_sensitive = False


# Singleton instance
_config: Optional[SentimentEngineConfig] = None


def get_config -> SentimentEngineConfig:
 """
 Get singleton configuration instance

 Returns:
 SentimentEngineConfig: Sentiment engine configuration
 """
 global _config
 if _config is None:
 _config = SentimentEngineConfig
 return _config


def reload_config -> SentimentEngineConfig:
 """
 Force configuration reload

 Returns:
 SentimentEngineConfig: Updated configuration
 """
 global _config
 _config = SentimentEngineConfig
 return _config


def get_crypto_symbols -> List[str]:
 """
 Get list of supported cryptocurrencies

 Returns:
 List[str]: List of cryptocurrency symbols
 """
 return [
 "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "TRX", "LTC", "AVAX",
 "DOT", "MATIC", "SHIB", "UNI", "LINK", "ATOM", "XLM", "ALGO", "VET", "ICP",
 "FIL", "MANA", "SAND", "AXS", "THETA", "AAVE", "MKR", "COMP", "SNX", "YFI"
 ]


def get_crypto_keywords -> List[str]:
 """
 Get keywords for crypto analysis

 Returns:
 List[str]: List of keywords
 """
 return [
 "bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft", "web3",
 "hodl", "moon", "lambo", "diamond hands", "paper hands", "bull", "bear",
 "pump", "dump", "whale", "ape", "fud", "fomo", "rugpull", "altcoin",
 "staking", "yield farming", "liquidity mining", "metaverse", "dao"
 ]