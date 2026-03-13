"""
Configuration management for Prophet forecasting system.

Provides comprehensive configuration management using Pydantic with enterprise patterns
for enterprise-grade deployment, monitoring, and performance optimization.
"""

from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime, timedelta
from pathlib import Path
import os
from enum import Enum

from pydantic import BaseSettings, Field, validator, root_validator
from pydantic.types import PositiveInt, PositiveFloat, confloat


class LogLevel(str, Enum):
    """Levels logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SeasonalityMode(str, Enum):
    """Modes seasonality Prophet"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class GrowthMode(str, Enum):
    """Modes growth Prophet"""
    LINEAR = "linear"
    LOGISTIC = "logistic"
    FLAT = "flat"


class OptimizationMethod(str, Enum):
    """Methods optimization hyperparameters"""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    OPTUNA = "optuna"


class ModelConfig(BaseSettings):
    """
    Configuration model Prophet with enterprise settings
    """
    
    # === Main parameters Prophet ===
    growth: GrowthMode = Field(
        default=GrowthMode.LINEAR,
        description="Type growth trend"
    )
    
    seasonality_mode: SeasonalityMode = Field(
        default=SeasonalityMode.ADDITIVE,
        description="Mode seasonality"
    )
    
    changepoint_prior_scale: PositiveFloat = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Flexibility changes trends"
    )
    
    seasonality_prior_scale: PositiveFloat = Field(
        default=10.0,
        ge=0.01,
        le=100.0,
        description="Force seasonality"
    )
    
    holidays_prior_scale: PositiveFloat = Field(
        default=10.0,
        ge=0.01,
        le=100.0,
        description="Influence holidays"
    )
    
    daily_seasonality: Union[bool, str, int] = Field(
        default="auto",
        description="Daily seasonality"
    )
    
    weekly_seasonality: Union[bool, str, int] = Field(
        default="auto", 
        description="Weekly seasonality"
    )
    
    yearly_seasonality: Union[bool, str, int] = Field(
        default="auto",
        description="Annual seasonality"
    )
    
    # === Points changes ===
    n_changepoints: PositiveInt = Field(
        default=25,
        ge=1,
        le=100,
        description="Number points changes"
    )
    
    changepoint_range: confloat(gt=0, le=1) = Field(
        default=0.8,
        description="Share history for search changepoints"
    )
    
    # === Intervals uncertainty ===
    interval_width: confloat(gt=0, lt=1) = Field(
        default=0.8,
        description="Width confidence interval"
    )
    
    uncertainty_samples: PositiveInt = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number samples for uncertainty"
    )
    
    # === Custom seasonality ===
    custom_seasonalities: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "name": "crypto_hourly",
                "period": 24,
                "fourier_order": 8,
                "mode": "additive"
            },
            {
                "name": "crypto_weekly", 
                "period": 168,  # 24*7 hours
                "fourier_order": 10,
                "mode": "additive"
            },
            {
                "name": "crypto_monthly",
                "period": 720,  # 24*30 hours
                "fourier_order": 5,
                "mode": "additive"
            }
        ],
        description="Custom seasonality for crypto markets"
    )
    
    # === Regressors ===
    additional_regressors: List[str] = Field(
        default_factory=lambda: [
            "volume_ma",
            "volatility", 
            "rsi",
            "macd",
            "sentiment_score",
            "btc_dominance"
        ],
        description="Additional regressors"
    )
    
    # === Validation ===
    
    @validator('daily_seasonality', 'weekly_seasonality', 'yearly_seasonality')
    def validate_seasonality(cls, v):
        """Validation parameters seasonality"""
        if isinstance(v, str) and v not in ["auto"]:
            raise ValueError("String seasonality must be 'auto'")
        if isinstance(v, int) and v < 0:
            raise ValueError("Integer seasonality must be >= 0")
        return v
    
    class Config:
        env_prefix = "PROPHET_MODEL_"
        case_sensitive = False


class DataConfig(BaseSettings):
    """
    Configuration processing data
    """
    
    # === Sources data ===
    default_exchange: str = Field(
        default="binance",
        description="Exchange by default"
    )
    
    supported_exchanges: List[str] = Field(
        default_factory=lambda: ["binance", "coinbase", "kraken", "okx"],
        description="Supported exchange"
    )
    
    # === Parameters data ===
    min_history_days: PositiveInt = Field(
        default=365,
        ge=30,
        le=3650,
        description="Minimum days history"
    )
    
    max_history_days: PositiveInt = Field(
        default=1095,  # 3 year
        ge=365,
        le=3650,
        description="Maximum days history"
    )
    
    forecast_horizon_days: PositiveInt = Field(
        default=30,
        ge=1,
        le=365,
        description="Horizon forecast in days"
    )
    
    # === Processing data ===
    outlier_detection: bool = Field(
        default=True,
        description="Enable detection outliers"
    )
    
    outlier_threshold: PositiveFloat = Field(
        default=3.0,
        ge=1.0,
        le=5.0,
        description="Threshold for outliers (in std)"
    )
    
    missing_data_strategy: Literal["interpolate", "forward_fill", "drop", "median"] = Field(
        default="interpolate",
        description="Strategy for missing data"
    )
    
    data_validation: bool = Field(
        default=True,
        description="Enable validation data"
    )
    
    # === Caching ===
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching data"
    )
    
    cache_ttl_hours: PositiveInt = Field(
        default=6,
        ge=1,
        le=168,
        description="TTL cache in hours"
    )
    
    class Config:
        env_prefix = "PROPHET_DATA_"
        case_sensitive = False


class APIConfig(BaseSettings):
    """
    Configuration API server
    """
    
    # === Server ===
    host: str = Field(default="0.0.0.0", description="Host API")
    port: PositiveInt = Field(default=8000, ge=1000, le=65535, description="Port API")
    debug: bool = Field(default=False, description="Mode debugging")
    
    # === CORS ===
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # === Rate Limiting ===
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: PositiveInt = Field(default=100, description="Requests in minute")
    rate_limit_window: PositiveInt = Field(default=60, description="Window in seconds")
    
    # === WebSocket ===
    websocket_enabled: bool = Field(default=True, description="Enable WebSocket")
    websocket_max_connections: PositiveInt = Field(default=100, description="Max connections")
    
    # === Authentication ===
    auth_enabled: bool = Field(default=False, description="Enable authentication")
    auth_secret_key: Optional[str] = Field(default=None, description="Secret key")
    auth_algorithm: str = Field(default="HS256", description="Algorithm JWT")
    
    class Config:
        env_prefix = "PROPHET_API_"
        case_sensitive = False


class OptimizationConfig(BaseSettings):
    """
    Configuration optimization hyperparameters
    """
    
    method: OptimizationMethod = Field(
        default=OptimizationMethod.BAYESIAN,
        description="Method optimization"
    )
    
    n_trials: PositiveInt = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number trials"
    )
    
    timeout_hours: PositiveFloat = Field(
        default=2.0,
        ge=0.1,
        le=24.0,
        description="Timeout in hours"
    )
    
    cv_folds: PositiveInt = Field(
        default=5,
        ge=2,
        le=10,
        description="Number folds for cross-validation"
    )
    
    # === Parameters for optimization ===
    param_space: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "changepoint_prior_scale": {"low": 0.001, "high": 0.5, "type": "float"},
            "seasonality_prior_scale": {"low": 0.01, "high": 100.0, "type": "float"},
            "holidays_prior_scale": {"low": 0.01, "high": 100.0, "type": "float"},
            "n_changepoints": {"low": 5, "high": 50, "type": "int"},
            "changepoint_range": {"low": 0.6, "high": 0.95, "type": "float"}
        },
        description="Space parameters for optimization"
    )
    
    # === Metrics for optimization ===
    optimization_metric: Literal["mape", "mae", "rmse", "smape"] = Field(
        default="mape",
        description="Metric for optimization"
    )
    
    class Config:
        env_prefix = "PROPHET_OPTIMIZATION_"
        case_sensitive = False


class MonitoringConfig(BaseSettings):
    """
    Configuration monitoring and observability
    """
    
    # === Logging ===
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Level logging")
    log_format: str = Field(default="json", description="Format logs")
    log_file: Optional[str] = Field(default=None, description="File logs")
    
    # === Metrics ===
    metrics_enabled: bool = Field(default=True, description="Enable metrics")
    metrics_port: PositiveInt = Field(default=9090, description="Port metrics")
    
    # === Health checks ===
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_interval: PositiveInt = Field(default=30, description="Interval validation")
    
    # === Tracing ===
    tracing_enabled: bool = Field(default=False, description="Enable tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="Endpoint tracing")
    
    class Config:
        env_prefix = "PROPHET_MONITORING_"
        case_sensitive = False


class ProphetConfig(BaseSettings):
    """
    Main configuration Prophet forecasting system
    
    Combines all aspects configuration with enterprise patterns
    """
    
    # === General settings ===
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Environment execution"
    )
    
    service_name: str = Field(
        default="prophet-forecasting",
        description="Name service"
    )
    
    version: str = Field(
        default="5.0.0",
        description="Version service"
    )
    
    # === Components configuration ===
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # === Database data ===
    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/prophet_db",
        description="URL base data"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="URL Redis"
    )
    
    # === Paths ===
    models_dir: Path = Field(
        default=Path("./models"),
        description="Directory models"
    )
    
    data_dir: Path = Field(
        default=Path("./data"),
        description="Directory data"
    )
    
    logs_dir: Path = Field(
        default=Path("./logs"),
        description="Directory logs"
    )
    
    @root_validator
    def validate_paths(cls, values):
        """Create directory if not exist"""
        for path_key in ["models_dir", "data_dir", "logs_dir"]:
            if path_key in values:
                path = values[path_key]
                if isinstance(path, str):
                    path = Path(path)
                    values[path_key] = path
                path.mkdir(parents=True, exist_ok=True)
        return values
    
    @validator("database_url", "redis_url")
    def validate_urls(cls, v):
        """Base validation URL"""
        if not v or not isinstance(v, str):
            raise ValueError("URL must be a non-empty string")
        return v
    
    def get_model_config_for_crypto(self, symbol: str) -> ModelConfig:
        """
        Get configuration model for specific cryptocurrency
        
        Args:
            symbol: Symbol cryptocurrency
            
        Returns:
            Specialized configuration model
        """
        config = self.model.copy()
        
        # Adaptation parameters for different cryptocurrencies
        if symbol.upper() in ["BTC", "ETH"]:
            # For large cryptocurrencies - more flexibility
            config.changepoint_prior_scale = 0.1
            config.n_changepoints = 35
        elif symbol.upper() in ["DOGE", "SHIB"]:
            # For meme-coins - more seasonality
            config.seasonality_prior_scale = 15.0
            config.daily_seasonality = True
        
        return config
    
    def is_production(self) -> bool:
        """Check production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check development environment"""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_all = True
        extra = "forbid"  # Prohibit excess fields


# Global configuration
_config: Optional[ProphetConfig] = None


def get_config() -> ProphetConfig:
    """
    Get global configuration (singleton pattern)
    
    Returns:
        Instance ProphetConfig
    """
    global _config
    if _config is None:
        _config = ProphetConfig()
    return _config


def reload_config() -> ProphetConfig:
    """
    Restart configuration
    
    Returns:
        New instance ProphetConfig
    """
    global _config
    _config = ProphetConfig()
    return _config


# Helper function
def load_config_from_file(config_path: Union[str, Path]) -> ProphetConfig:
    """
    Load configuration from file
    
    Args:
        config_path: Path to file configuration
        
    Returns:
        Instance ProphetConfig
    """
    import yaml
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    return ProphetConfig(**config_data)


def save_config_to_file(config: ProphetConfig, config_path: Union[str, Path]) -> None:
    """
    Save configuration in file
    
    Args:
        config: Instance configuration
        config_path: Path for saving
    """
    import yaml
    
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, allow_unicode=True)