"""
Configuration Management for NLP Sentiment Analysis

Enterprise-grade configuration system with enterprise patterns for
managing application settings, model parameters, and deployment configs.

Author: ML-Framework Team
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field, asdict
from pydantic import BaseSettings, Field, validator
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    model_type: str
    model_name_or_path: str
    num_labels: int = 3
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    
    # Model-specific parameters
    dropout_rate: float = 0.1
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Training parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Inference parameters
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    do_sample: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    # Text cleaning
    lowercase: bool = True
    remove_special_chars: bool = False
    remove_urls: bool = True
    remove_mentions: bool = False
    remove_hashtags: bool = False
    normalize_whitespace: bool = True
    
    # Crypto-specific preprocessing
    normalize_crypto_tickers: bool = True
    normalize_crypto_amounts: bool = True
    normalize_crypto_slang: bool = True
    extract_crypto_entities: bool = True
    
    # Emoji handling
    convert_emojis_to_text: bool = False
    remove_emojis: bool = False
    extract_emoji_sentiment: bool = True
    
    # Language processing
    detect_language: bool = True
    translate_to_english: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "ja", "ko"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "crypto_sentiment"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    def get_connection_string(self) -> str:
        """Get database connection string"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    
    def get_connection_params(self) -> Dict[str, Any]:
        """Get Redis connection parameters"""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.database,
            "password": self.password,
            "max_connections": self.max_connections,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "retry_on_timeout": self.retry_on_timeout,
        }


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    timeout: int = 300
    keep_alive: int = 2
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # CORS
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Authentication
    auth_enabled: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 24 * 60 * 60  # 24 hours


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_to_file: bool = True
    log_file_path: str = "logs/sentiment_analysis.log"
    log_file_max_size: int = 100 * 1024 * 1024  # 100MB
    log_file_backup_count: int = 5
    
    # Structured logging
    structured_logging: bool = True
    json_format: bool = True
    
    # Monitoring integration
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    
    # External logging services
    elasticsearch_enabled: bool = False
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_index: str = "crypto-sentiment-logs"


class Config(BaseSettings):
    """
    Main configuration class with enterprise patterns
    
    Features:
    - Environment-based configuration
    - YAML/JSON config file support
    - Validation and type checking
    - Hot reloading capabilities
    - Secrets management integration
    - Multi-environment support
    """
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    version: str = Field(default="1.0.0", env="VERSION")
    
    # Application settings
    app_name: str = Field(default="Crypto NLP Sentiment Analysis", env="APP_NAME")
    app_description: str = Field(default="Enterprise NLP sentiment analysis for cryptocurrency", env="APP_DESCRIPTION")
    
    # Paths
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    models_dir: Path = Field(default=Path("models"), env="MODELS_DIR")
    cache_dir: Path = Field(default=Path("cache"), env="CACHE_DIR")
    logs_dir: Path = Field(default=Path("logs"), env="LOGS_DIR")
    
    # Model configurations
    default_model: str = Field(default="ensemble", env="DEFAULT_MODEL")
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    # Component configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Security settings
    secret_key: Optional[str] = Field(default=None, env="SECRET_KEY")
    encryption_key: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Performance settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    batch_size: int = Field(default=32, env="BATCH_SIZE")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    # Monitoring and observability
    monitoring_enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # External services
    external_apis: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_all = True
        extra = "allow"
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
            **kwargs: Override configuration values
        """
        
        # Load from file if provided
        file_config = {}
        if config_file:
            file_config = self._load_config_file(config_file)
        
        # Merge configurations: file < environment < kwargs
        merged_config = {**file_config, **kwargs}
        
        super().__init__(**merged_config)
        
        # Initialize default models
        if not self.models:
            self._setup_default_models()
        
        # Create directories
        self._create_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _load_config_file(self, config_file: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file {config_path} not found")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.error(f"Unsupported config file format: {config_path.suffix}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}
    
    def _setup_default_models(self):
        """Setup default model configurations"""
        
        self.models = {
            "bert_sentiment": ModelConfig(
                name="bert_sentiment",
                model_type="bert",
                model_name_or_path="bert-base-uncased",
                num_labels=3,
                batch_size=32,
            ),
            "finbert": ModelConfig(
                name="finbert",
                model_type="finbert",
                model_name_or_path="ProsusAI/finbert",
                num_labels=3,
                batch_size=16,
            ),
            "roberta_sentiment": ModelConfig(
                name="roberta_sentiment",
                model_type="roberta",
                model_name_or_path="cardiffnlp/twitter-roberta-base-sentiment-latest",
                num_labels=3,
                batch_size=32,
            ),
            "distilbert": ModelConfig(
                name="distilbert",
                model_type="distilbert",
                model_name_or_path="distilbert-base-uncased",
                num_labels=3,
                batch_size=64,
            ),
            "crypto_bert": ModelConfig(
                name="crypto_bert",
                model_type="crypto_bert",
                model_name_or_path="bert-base-uncased",
                num_labels=3,
                batch_size=32,
            ),
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        
        directories = [
            self.data_dir,
            self.models_dir,
            self.cache_dir,
            self.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        
        import logging.config
        
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": self.logging.format
                },
                "json": {
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                    "class": "pythonjsonlogger.jsonlogger.JsonFormatter" if self.logging.json_format else "logging.Formatter",
                }
            },
            "handlers": {
                "console": {
                    "level": self.logging.level,
                    "class": "logging.StreamHandler",
                    "formatter": "json" if self.logging.structured_logging else "standard",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console"],
                    "level": self.logging.level,
                    "propagate": False
                }
            }
        }
        
        # Add file handler if enabled
        if self.logging.log_to_file:
            logging_config["handlers"]["file"] = {
                "level": self.logging.level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": self.logging.log_file_path,
                "maxBytes": self.logging.log_file_max_size,
                "backupCount": self.logging.log_file_backup_count,
                "formatter": "json" if self.logging.structured_logging else "standard",
            }
            logging_config["loggers"][""]["handlers"].append("file")
        
        logging.config.dictConfig(logging_config)
    
    @validator("data_dir", "models_dir", "cache_dir", "logs_dir", pre=True)
    def validate_paths(cls, v):
        """Validate and convert path strings to Path objects"""
        return Path(v) if isinstance(v, str) else v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value"""
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        return self.models.get(model_name)
    
    def add_model_config(self, model_config: ModelConfig):
        """Add new model configuration"""
        self.models[model_config.name] = model_config
    
    def update_model_config(self, model_name: str, **kwargs):
        """Update existing model configuration"""
        if model_name in self.models:
            for key, value in kwargs.items():
                if hasattr(self.models[model_name], key):
                    setattr(self.models[model_name], key, value)
    
    def save_config(self, config_file: Union[str, Path], format: str = "yaml"):
        """
        Save current configuration to file
        
        Args:
            config_file: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        
        config_path = Path(config_file)
        config_data = self.dict()
        
        # Convert Path objects to strings for serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        config_data = convert_paths(config_data)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif format.lower() == 'json':
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            raise
    
    def reload_config(self, config_file: Optional[Union[str, Path]] = None):
        """Reload configuration from file"""
        
        if config_file:
            file_config = self._load_config_file(config_file)
            
            # Update current configuration
            for key, value in file_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            logger.info(f"Configuration reloaded from {config_file}")
    
    def get_api_key(self, service_name: str) -> Optional[str]:
        """Get API key for external service"""
        return self.api_keys.get(service_name)
    
    def add_api_key(self, service_name: str, api_key: str):
        """Add API key for external service"""
        self.api_keys[service_name] = api_key
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.dict()
    
    def __repr__(self) -> str:
        return f"Config(environment={self.environment}, models={len(self.models)}, debug={self.debug})"


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        # Try to load from default config file
        config_files = ["config.yaml", "config.yml", "config.json"]
        
        for config_file in config_files:
            if Path(config_file).exists():
                _global_config = Config(config_file=config_file)
                break
        else:
            # Create default configuration
            _global_config = Config()
    
    return _global_config


def set_config(config: Config):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


def load_config(config_file: Union[str, Path]) -> Config:
    """Load configuration from file"""
    return Config(config_file=config_file)