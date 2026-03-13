"""
Configuration management for Elliott Wave Analyzer.

Configuration as Code with environment-specific overrides
and secure credential management.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseSettings, Field, validator
from enum import Enum
import json


class Environment(str, Enum):
    """Environment types for deployment configuration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging" 
    PRODUCTION = "production"


class WaveDegree(str, Enum):
    """Elliott Wave degree classifications."""
    GRAND_SUPERCYCLE = "grand_supercycle"
    SUPERCYCLE = "supercycle"
    CYCLE = "cycle"
    PRIMARY = "primary"
    INTERMEDIATE = "intermediate"
    MINOR = "minor"
    MINUTE = "minute"
    MINUETTE = "minuette"
    SUBMINUETTE = "subminuette"


class CryptoExchange(str, Enum):
    """Supported crypto exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITFINEX = "bitfinex"
    HUOBI = "huobi"
    KUCOIN = "kucoin"
    FTX = "ftx"
    BYBIT = "bybit"


class ElliottWaveConfig(BaseSettings):
    """
    Main configuration class for Elliott Wave Analyzer.
    
    Uses Pydantic Settings for type validation and environment variable support.
    Configuration validation and security.
    """
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Deployment environment"
    )
    debug: bool = Field(default=False, description="Debug mode flag")
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://localhost:5432/elliott_waves",
        description="Database connection URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of API workers")
    api_rate_limit: int = Field(default=1000, description="API rate limit per minute")
    
    # WebSocket Configuration
    ws_host: str = Field(default="0.0.0.0", description="WebSocket host")
    ws_port: int = Field(default=8001, description="WebSocket port")
    ws_max_connections: int = Field(default=1000, description="Max WebSocket connections")
    
    # Crypto Exchange APIs
    exchanges: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "binance": {
                "api_key": "",
                "api_secret": "",
                "sandbox": True,
                "rate_limit": 1200,
                "timeout": 10.0
            },
            "coinbase": {
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "sandbox": True,
                "rate_limit": 10,
                "timeout": 10.0
            }
        },
        description="Exchange API configurations"
    )
    
    # Elliott Wave Analysis Configuration
    wave_analysis: Dict[str, Any] = Field(
        default_factory=lambda: {
            "min_wave_length": 5,
            "max_wave_length": 1000,
            "fibonacci_tolerance": 0.05,
            "confidence_threshold": 0.7,
            "max_alternative_counts": 5,
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "default_degree": WaveDegree.MINOR.value,
            "enable_multi_timeframe": True,
            "enable_confluence": True
        },
        description="Wave analysis parameters"
    )
    
    # Machine Learning Configuration
    ml_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model_path": "models/",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "early_stopping": True,
            "patience": 10,
            "validation_split": 0.2,
            "test_split": 0.2,
            "random_seed": 42,
            "device": "auto",  # auto, cpu, cuda
            "model_versions": {
                "cnn_wave_detector": "1.0.0",
                "lstm_predictor": "1.0.0", 
                "transformer_analyzer": "1.0.0"
            }
        },
        description="Machine learning configuration"
    )
    
    # Fibonacci Configuration
    fibonacci_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "retracement_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
            "extension_levels": [1.272, 1.414, 1.618, 2.0, 2.618],
            "time_ratios": [1.0, 1.272, 1.618, 2.0, 2.618],
            "cluster_tolerance": 0.02,
            "projection_periods": 100,
            "enable_fan_lines": True,
            "enable_arcs": False
        },
        description="Fibonacci analysis configuration"
    )
    
    # Backtesting Configuration
    backtesting_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0005,
            "margin_req": 0.1,
            "max_positions": 10,
            "risk_per_trade": 0.02,
            "max_drawdown": 0.2,
            "benchmark": "BTC",
            "start_date": "2020-01-01",
            "end_date": "2024-01-01"
        },
        description="Backtesting parameters"
    )
    
    # Real-time Analysis Configuration
    realtime_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "update_interval": 5,  # seconds
            "data_buffer_size": 1000,
            "enable_alerts": True,
            "alert_channels": ["email", "webhook", "telegram"],
            "max_concurrent_streams": 100,
            "stream_timeout": 30,
            "reconnect_attempts": 5,
            "heartbeat_interval": 30
        },
        description="Real-time analysis configuration"
    )
    
    # Visualization Configuration  
    visualization_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "chart_width": 1200,
            "chart_height": 800,
            "theme": "dark",
            "show_volume": True,
            "show_indicators": True,
            "wave_colors": {
                "impulse": "#00ff00",
                "corrective": "#ff0000",
                "diagonal": "#ffff00",
                "triangle": "#ff00ff"
            },
            "fibonacci_colors": {
                "retracement": "#0080ff",
                "extension": "#ff8000", 
                "projection": "#8000ff"
            },
            "export_formats": ["png", "svg", "html", "json"],
            "interactive": True
        },
        description="Visualization configuration"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        description="Log message format"
    )
    log_rotation: str = Field(default="100 MB", description="Log rotation size")
    log_retention: str = Field(default="30 days", description="Log retention period")
    
    # Security Configuration
    jwt_secret: str = Field(default="", description="JWT secret key - set via JWT_SECRET env var")
    jwt_expiry: int = Field(default=3600, description="JWT expiry time in seconds")
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Performance Configuration
    performance_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "enable_caching": True,
            "cache_ttl": 300,  # seconds
            "max_cache_size": 1000,
            "enable_compression": True,
            "worker_pool_size": 4,
            "async_timeout": 30.0,
            "connection_pool_size": 20,
            "max_retries": 3,
            "retry_delay": 1.0
        },
        description="Performance optimization settings"
    )
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('exchanges')
    def validate_exchanges(cls, v):
        """Validate exchange configurations."""
        for exchange, config in v.items():
            if exchange not in [e.value for e in CryptoExchange]:
                raise ValueError(f"Unsupported exchange: {exchange}")
        return v
    
    @validator('wave_analysis')
    def validate_wave_analysis(cls, v):
        """Validate wave analysis configuration."""
        if v['min_wave_length'] >= v['max_wave_length']:
            raise ValueError("min_wave_length must be less than max_wave_length")
        if not 0 < v['confidence_threshold'] < 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global configuration instance
config = ElliottWaveConfig()


def load_config(config_path: Optional[str] = None) -> ElliottWaveConfig:
    """
    Load configuration from file or environment.
    
    Configuration loading with fallbacks.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ElliottWaveConfig: Loaded configuration
    """
    if config_path and Path(config_path).exists():
        # Load from JSON file
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        return ElliottWaveConfig(**config_data)
    
    # Load from environment variables and defaults
    return ElliottWaveConfig()


def save_config(config_obj: ElliottWaveConfig, config_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        config_obj: Configuration object to save
        config_path: Path to save configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config_obj.dict(), f, indent=2, default=str)


def get_exchange_config(exchange: str) -> Dict[str, Any]:
    """
    Get configuration for specific exchange.
    
    Args:
        exchange: Exchange name
        
    Returns:
        Dict[str, Any]: Exchange configuration
        
    Raises:
        ValueError: If exchange is not configured
    """
    if exchange not in config.exchanges:
        raise ValueError(f"Exchange '{exchange}' not configured")
    return config.exchanges[exchange]


def is_production() -> bool:
    """Check if running in production environment."""
    return config.environment == Environment.PRODUCTION


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return config.debug or config.environment == Environment.DEVELOPMENT


# Export main configuration for easy import
__all__ = [
    'config',
    'ElliottWaveConfig',
    'Environment',
    'WaveDegree', 
    'CryptoExchange',
    'load_config',
    'save_config',
    'get_exchange_config',
    'is_production',
    'is_debug'
]