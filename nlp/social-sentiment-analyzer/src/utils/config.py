"""
Configuration system analysis sentiments

 enterprise-grade configuration with support various
environments and validation parameters.
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class DatabaseConfig:
    """Configuration base data."""
    host: str = "localhost"
    port: int = 5432
    database: str = "crypto_sentiment"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class RedisConfig:
    """Configuration Redis."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: int = 30
    socket_connect_timeout: int = 30

@dataclass
class TwitterConfig:
    """Configuration Twitter API."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_secret: str = ""
    bearer_token: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.bearer_token or (self.api_key and self.api_secret))

@dataclass
class RedditConfig:
    """Configuration Reddit API."""
    client_id: str = ""
    client_secret: str = ""
    username: str = ""
    password: str = ""
    user_agent: str = "CryptoSentimentBot/1.0"
    
    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

@dataclass
class TelegramConfig:
    """Configuration Telegram API."""
    api_id: int = 0
    api_hash: str = ""
    phone: str = ""
    session_string: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_id and self.api_hash)

@dataclass
class DiscordConfig:
    """Configuration Discord Bot."""
    bot_token: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token)

@dataclass
class YouTubeConfig:
    """Configuration YouTube API."""
    api_key: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

@dataclass
class TikTokConfig:
    """Configuration TikTok API."""
    access_token: str = ""
    
    @property
    def is_configured(self) -> bool:
        return bool(self.access_token)

@dataclass
class MLConfig:
    """Configuration ML models."""
    model_cache_dir: str = "./models"
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 32
    max_sequence_length: int = 512
    ensemble_models: List[str] = field(default_factory=lambda: [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "ProsusAI/finbert"
    ])
    use_gpu: bool = True
    fp16: bool = False
    
    @property
    def models_dir(self) -> Path:
        return Path(self.model_cache_dir)

@dataclass
class MonitoringConfig:
    """Configuration monitoring."""
    prometheus_port: int = 9090
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    log_level: str = "INFO"
    structured_logging: bool = True
    
@dataclass
class SecurityConfig:
    """Configuration security."""
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60
    api_key_header: str = "X-API-Key"
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # seconds
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

class Config:
    """
    Main class configuration with enterprise patterns
    
    Features:
    - Environment-based configuration
    - Validation parameters
    - Secret management
    - Hot-reload configuration
    - Health checks
    """
    
    def __init__(self, env: str = "development"):
        self.env = env
        self.logger = logger.bind(component="config", environment=env)
        
        # Loading configuration
        self._load_config()
        
        # Validation
        self._validate_config()
        
        self.logger.info("Configuration loaded successfully", env=env)
    
    def _load_config(self) -> None:
        """Loading configuration from variables environment."""
        
        # Database
        self.database = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "crypto_sentiment"),
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20"))
        )
        
        # Redis
        self.redis = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
        )
        
        # Twitter
        self.twitter = TwitterConfig(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.getenv("TWITTER_ACCESS_SECRET", ""),
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN", "")
        )
        
        # Reddit
        self.reddit = RedditConfig(
            client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            username=os.getenv("REDDIT_USERNAME", ""),
            password=os.getenv("REDDIT_PASSWORD", ""),
            user_agent=os.getenv("REDDIT_USER_AGENT", "CryptoSentimentBot/1.0")
        )
        
        # Telegram
        self.telegram = TelegramConfig(
            api_id=int(os.getenv("TELEGRAM_API_ID", "0")),
            api_hash=os.getenv("TELEGRAM_API_HASH", ""),
            phone=os.getenv("TELEGRAM_PHONE", ""),
            session_string=os.getenv("TELEGRAM_SESSION", "")
        )
        
        # Discord
        self.discord = DiscordConfig(
            bot_token=os.getenv("DISCORD_BOT_TOKEN", "")
        )
        
        # YouTube
        self.youtube = YouTubeConfig(
            api_key=os.getenv("YOUTUBE_API_KEY", "")
        )
        
        # TikTok
        self.tiktok = TikTokConfig(
            access_token=os.getenv("TIKTOK_ACCESS_TOKEN", "")
        )
        
        # ML Models
        self.ml = MLConfig(
            model_cache_dir=os.getenv("ML_MODEL_CACHE_DIR", "./models"),
            device=os.getenv("ML_DEVICE", "auto"),
            batch_size=int(os.getenv("ML_BATCH_SIZE", "32")),
            max_sequence_length=int(os.getenv("ML_MAX_SEQ_LENGTH", "512")),
            use_gpu=os.getenv("ML_USE_GPU", "true").lower() == "true",
            fp16=os.getenv("ML_FP16", "false").lower() == "true"
        )
        
        # Monitoring
        self.monitoring = MonitoringConfig(
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            structured_logging=os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
        )
        
        # Security
        self.security = SecurityConfig(
            jwt_secret=os.getenv("JWT_SECRET", ""),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
            api_key_header=os.getenv("API_KEY_HEADER", "X-API-Key"),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "1000")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "3600")),
            allowed_origins=os.getenv("ALLOWED_ORIGINS", "*").split(",")
        )
        
        # Additional parameters
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8004"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.workers = int(os.getenv("WORKERS", "4"))
        
        # Real-time processing
        self.realtime_batch_size = int(os.getenv("REALTIME_BATCH_SIZE", "32"))
        self.realtime_max_queue_size = int(os.getenv("REALTIME_MAX_QUEUE_SIZE", "1000"))
        self.realtime_timeout = float(os.getenv("REALTIME_TIMEOUT", "1.0"))
        
        # URLs for connections
        self.database_url = self._build_database_url()
        self.redis_url = self._build_redis_url()
    
    def _build_database_url(self) -> str:
        """Construction URL for base data."""
        
        db = self.database
        if db.password:
            auth = f"{db.username}:{db.password}"
        else:
            auth = db.username
        
        return f"postgresql://{auth}@{db.host}:{db.port}/{db.database}"
    
    def _build_redis_url(self) -> str:
        """Construction URL for Redis."""
        
        redis = self.redis
        if redis.password:
            auth = f":{redis.password}@"
        else:
            auth = ""
        
        return f"redis://{auth}{redis.host}:{redis.port}/{redis.database}"
    
    def _validate_config(self) -> None:
        """Validation configuration."""
        
        errors = []
        
        # Validation critical parameters
        if self.env == "production":
            if not self.security.jwt_secret:
                errors.append("JWT_SECRET is required in production")
            
            if self.debug:
                errors.append("DEBUG should be false in production")
            
            if not any([
                self.twitter.is_configured,
                self.reddit.is_configured,
                self.telegram.is_configured,
                self.discord.is_configured,
                self.youtube.is_configured
            ]):
                errors.append("At least one social media platform must be configured")
        
        # Validation ML settings
        if self.ml.batch_size <= 0:
            errors.append("ML batch_size must be positive")
        
        if self.ml.max_sequence_length <= 0:
            errors.append("ML max_sequence_length must be positive")
        
        # Validation ports
        if not (1024 <= self.api_port <= 65535):
            errors.append("API port must be between 1024 and 65535")
        
        # Validation rate limiting
        if self.security.rate_limit_requests <= 0:
            errors.append("Rate limit requests must be positive")
        
        if errors:
            self.logger.error("Configuration validation failed", errors=errors)
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    def get_platform_configs(self) -> Dict[str, Any]:
        """Get configuration platforms."""
        
        return {
            "twitter": self.twitter,
            "reddit": self.reddit,
            "telegram": self.telegram,
            "discord": self.discord,
            "youtube": self.youtube,
            "tiktok": self.tiktok
        }
    
    def get_enabled_platforms(self) -> List[str]:
        """Get list enabled platforms."""
        
        enabled = []
        
        if self.twitter.is_configured:
            enabled.append("twitter")
        if self.reddit.is_configured:
            enabled.append("reddit")
        if self.telegram.is_configured:
            enabled.append("telegram")
        if self.discord.is_configured:
            enabled.append("discord")
        if self.youtube.is_configured:
            enabled.append("youtube")
        if self.tiktok.is_configured:
            enabled.append("tiktok")
        
        return enabled
    
    def health_check(self) -> Dict[str, Any]:
        """Validation state configuration."""
        
        return {
            "environment": self.env,
            "debug": self.debug,
            "enabled_platforms": self.get_enabled_platforms(),
            "api_config": {
                "host": self.api_host,
                "port": self.api_port,
                "workers": self.workers
            },
            "ml_config": {
                "device": self.ml.device,
                "batch_size": self.ml.batch_size,
                "models_count": len(self.ml.ensemble_models)
            },
            "performance_config": {
                "realtime_batch_size": self.realtime_batch_size,
                "max_queue_size": self.realtime_max_queue_size,
                "timeout": self.realtime_timeout
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary (without secrets)."""
        
        return {
            "environment": self.env,
            "debug": self.debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "workers": self.workers,
            "enabled_platforms": self.get_enabled_platforms(),
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "pool_size": self.database.pool_size
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "database": self.redis.database
            },
            "ml": {
                "device": self.ml.device,
                "batch_size": self.ml.batch_size,
                "use_gpu": self.ml.use_gpu,
                "models_count": len(self.ml.ensemble_models)
            }
        }

# Global instance configuration
config = Config(os.getenv("ENVIRONMENT", "development"))

def get_config() -> Config:
    """Get global configuration."""
    return config

def reload_config() -> Config:
    """Restart configuration."""
    global config
    config = Config(os.getenv("ENVIRONMENT", "development"))
    return config