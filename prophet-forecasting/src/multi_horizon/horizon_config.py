"""
Horizon Configuration System
ML-Framework-1329 - Multi-horizon forecasting configuration management

 2025: Adaptive configuration, performance optimization,
enterprise-grade parameter management.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any
import json


class HorizonType(Enum):
    """Types of forecasting horizons"""
    ULTRA_SHORT = "ultra_short"      # Seconds to minutes (scalping)
    SHORT = "short"                  # Minutes to hours (day trading)
    MEDIUM = "medium"                # Hours to days (swing trading)
    LONG = "long"                    # Days to weeks (position trading)
    STRATEGIC = "strategic"          # Weeks to months (investment)


class ModelType(Enum):
    """Available model types for horizons"""
    PROPHET = "prophet"
    PROPHET_EXTENDED = "prophet_extended"
    ENSEMBLE = "ensemble"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ARIMA = "arima"
    CUSTOM = "custom"


class SeasonalityMode(Enum):
    """Seasonality detection modes"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    AUTO = "auto"


@dataclass
class ProphetParameters:
    """Prophet-specific model parameters"""
    growth: str = "linear"                    # 'linear' or 'logistic'
    changepoint_prior_scale: float = 0.05     # Flexibility of trend
    seasonality_prior_scale: float = 10.0     # Strength of seasonality
    holidays_prior_scale: float = 10.0        # Strength of holiday effects
    seasonality_mode: str = "additive"        # 'additive' or 'multiplicative'
    changepoint_range: float = 0.8            # Proportion of history for changepoints
    yearly_seasonality: Union[bool, str, int] = "auto"
    weekly_seasonality: Union[bool, str, int] = "auto"
    daily_seasonality: Union[bool, str, int] = "auto"
    mcmc_samples: int = 0                     # MCMC samples for uncertainty
    interval_width: float = 0.80              # Prediction interval width
    uncertainty_samples: int = 1000           # Samples for uncertainty quantification

    def to_prophet_kwargs(self) -> Dict[str, Any]:
        """Convert to Prophet constructor arguments"""
        return {
            'growth': self.growth,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'seasonality_prior_scale': self.seasonality_prior_scale,
            'holidays_prior_scale': self.holidays_prior_scale,
            'seasonality_mode': self.seasonality_mode,
            'changepoint_range': self.changepoint_range,
            'yearly_seasonality': self.yearly_seasonality,
            'weekly_seasonality': self.weekly_seasonality,
            'daily_seasonality': self.daily_seasonality,
            'mcmc_samples': self.mcmc_samples,
            'interval_width': self.interval_width,
            'uncertainty_samples': self.uncertainty_samples
        }


@dataclass
class PerformanceConstraints:
    """Performance requirements for horizon"""
    max_execution_time_ms: float = 100.0     # Maximum execution time
    max_memory_mb: float = 512.0              # Maximum memory usage
    min_accuracy: float = 0.70                # Minimum required accuracy
    max_cpu_cores: int = 2                    # CPU core limit
    enable_caching: bool = True               # Enable result caching
    cache_ttl_minutes: int = 15               # Cache time-to-live
    priority: int = 1                         # Execution priority (1=highest)


@dataclass
class ValidationSettings:
    """Model validation and testing configuration"""
    enable_cross_validation: bool = True     # Enable cross-validation
    cv_horizon_days: int = 30                # Cross-validation horizon
    cv_period_days: int = 180                # Cross-validation period
    cv_initial_days: int = 365               # Initial training period
    min_training_days: int = 90              # Minimum training data
    outlier_detection: bool = True           # Enable outlier detection
    outlier_threshold: float = 3.0           # Outlier detection threshold
    enable_backtesting: bool = True          # Enable backtesting
    backtest_periods: int = 5                # Number of backtest periods


@dataclass
class DataProcessingConfig:
    """Data preprocessing configuration"""
    resample_frequency: Optional[str] = None  # Resample to frequency
    fill_missing_method: str = "forward"      # Method for missing data
    outlier_handling: str = "clip"            # 'clip', 'remove', 'interpolate'
    smoothing_window: Optional[int] = None    # Smoothing window size
    log_transform: bool = False               # Apply log transformation
    difference_order: int = 0                 # Differencing order
    normalize: bool = False                   # Normalize data
    feature_engineering: List[str] = field(  # Feature engineering methods
        default_factory=lambda: ["returns", "volatility", "volume_profile"]
    )


@dataclass
class HorizonConfig:
    """
    Comprehensive configuration for a single forecasting horizon.

    Defines all parameters needed to configure and optimize
    predictions for a specific time horizon.
    """

    # Basic identification
    name: str                                 # Unique horizon name
    timeframe: str                           # Timeframe string (1m, 5m, 1h, 1d, etc.)
    horizon_type: HorizonType               # Type classification
    model_type: ModelType = ModelType.PROPHET

    # Time configuration
    forecast_periods: int = 30               # Default forecast periods
    min_history_periods: int = 100           # Minimum historical data
    max_history_periods: int = 1000          # Maximum historical data

    # Model-specific parameters
    prophet_params: ProphetParameters = field(default_factory=ProphetParameters)
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Performance and constraints
    performance: PerformanceConstraints = field(default_factory=PerformanceConstraints)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)

    # Advanced features
    enable_ensemble: bool = False            # Use ensemble of models
    ensemble_models: List[ModelType] = field(default_factory=list)
    custom_features: List[str] = field(default_factory=list)
    external_regressors: List[str] = field(default_factory=list)

    # enterprise integration
    adaptive_config: bool = True             # Enable adaptive configuration
    auto_optimization: bool = True           # Enable auto-optimization
    circuit_breaker: bool = True             # Enable circuit breaker
    monitoring: bool = True                  # Enable performance monitoring

    # Metadata
    description: Optional[str] = None        # Human-readable description
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    version: str = "1.0"

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Validate timeframe format
        self._validate_timeframe()

        # Set up horizon-specific defaults
        self._apply_horizon_defaults()

        # Validate configuration consistency
        self._validate_configuration()

    def _validate_timeframe(self):
        """Validate timeframe string format"""
        import re
        pattern = r'^(\d+)([mhd]|min|hour|day)$'
        if not re.match(pattern, self.timeframe.lower()):
            raise ValueError(f"Invalid timeframe format: {self.timeframe}")

    def _apply_horizon_defaults(self):
        """Apply defaults based on horizon type"""
        defaults = {
            HorizonType.ULTRA_SHORT: {
                'forecast_periods': 60,
                'min_history_periods': 200,
                'max_execution_time_ms': 50.0,
                'priority': 1,
                'enable_caching': True,
                'cache_ttl_minutes': 5
            },
            HorizonType.SHORT: {
                'forecast_periods': 24,
                'min_history_periods': 168,
                'max_execution_time_ms': 100.0,
                'priority': 2,
                'enable_caching': True,
                'cache_ttl_minutes': 15
            },
            HorizonType.MEDIUM: {
                'forecast_periods': 30,
                'min_history_periods': 90,
                'max_execution_time_ms': 200.0,
                'priority': 3,
                'enable_caching': True,
                'cache_ttl_minutes': 60
            },
            HorizonType.LONG: {
                'forecast_periods': 14,
                'min_history_periods': 60,
                'max_execution_time_ms': 500.0,
                'priority': 4,
                'enable_caching': True,
                'cache_ttl_minutes': 240
            },
            HorizonType.STRATEGIC: {
                'forecast_periods': 12,
                'min_history_periods': 36,
                'max_execution_time_ms': 1000.0,
                'priority': 5,
                'enable_caching': True,
                'cache_ttl_minutes': 1440  # 24 hours
            }
        }

        if self.horizon_type in defaults:
            horizon_defaults = defaults[self.horizon_type]

            # Apply performance defaults
            if hasattr(self.performance, 'max_execution_time_ms'):
                if self.performance.max_execution_time_ms == 100.0:  # Default value
                    self.performance.max_execution_time_ms = horizon_defaults['max_execution_time_ms']

            if hasattr(self.performance, 'priority'):
                if self.performance.priority == 1:  # Default value
                    self.performance.priority = horizon_defaults['priority']

            if hasattr(self.performance, 'cache_ttl_minutes'):
                if self.performance.cache_ttl_minutes == 15:  # Default value
                    self.performance.cache_ttl_minutes = horizon_defaults['cache_ttl_minutes']

    def _validate_configuration(self):
        """Validate configuration for consistency"""
        # Validate forecast periods
        if self.forecast_periods <= 0:
            raise ValueError("forecast_periods must be positive")

        # Validate history requirements
        if self.min_history_periods <= 0:
            raise ValueError("min_history_periods must be positive")

        if self.max_history_periods < self.min_history_periods:
            raise ValueError("max_history_periods must be >= min_history_periods")

        # Validate performance constraints
        if self.performance.max_execution_time_ms <= 0:
            raise ValueError("max_execution_time_ms must be positive")

        if self.performance.min_accuracy < 0 or self.performance.min_accuracy > 1:
            raise ValueError("min_accuracy must be between 0 and 1")

    @classmethod
    def create_ultra_short_config(cls, name: str, timeframe: str = "1m") -> 'HorizonConfig':
        """Create optimized configuration for ultra-short horizons (scalping)"""
        prophet_params = ProphetParameters(
            changepoint_prior_scale=0.1,       # More flexible for rapid changes
            seasonality_prior_scale=5.0,       # Reduced seasonality importance
            daily_seasonality=True,            # Important for intraday
            weekly_seasonality=False,          # Less relevant for ultra-short
            yearly_seasonality=False,          # Not relevant for ultra-short
            mcmc_samples=0,                    # Speed over precision
            interval_width=0.90                # Higher confidence for trading
        )

        performance = PerformanceConstraints(
            max_execution_time_ms=25.0,        # Ultra-fast execution
            priority=1,                        # Highest priority
            cache_ttl_minutes=1                # Very short cache
        )

        return cls(
            name=name,
            timeframe=timeframe,
            horizon_type=HorizonType.ULTRA_SHORT,
            forecast_periods=60,               # 1 hour ahead
            prophet_params=prophet_params,
            performance=performance,
            description=f"Ultra-short term forecasting for {timeframe} scalping"
        )

    @classmethod
    def create_short_config(cls, name: str, timeframe: str = "5m") -> 'HorizonConfig':
        """Create optimized configuration for short horizons (day trading)"""
        prophet_params = ProphetParameters(
            changepoint_prior_scale=0.05,      # Balanced flexibility
            seasonality_prior_scale=10.0,      # Standard seasonality
            daily_seasonality=True,            # Important for day trading
            weekly_seasonality=True,           # Some weekly patterns
            yearly_seasonality=False,          # Not relevant for short-term
            uncertainty_samples=500            # Moderate uncertainty sampling
        )

        return cls(
            name=name,
            timeframe=timeframe,
            horizon_type=HorizonType.SHORT,
            forecast_periods=48,               # 4 hours ahead
            prophet_params=prophet_params,
            description=f"Short-term forecasting for {timeframe} day trading"
        )

    @classmethod
    def create_medium_config(cls, name: str, timeframe: str = "1h") -> 'HorizonConfig':
        """Create optimized configuration for medium horizons (swing trading)"""
        prophet_params = ProphetParameters(
            changepoint_prior_scale=0.05,      # Standard flexibility
            seasonality_prior_scale=10.0,      # Full seasonality
            daily_seasonality=True,            # Daily patterns important
            weekly_seasonality=True,           # Weekly patterns relevant
            yearly_seasonality="auto",         # Let Prophet decide
            uncertainty_samples=1000           # Full uncertainty sampling
        )

        validation = ValidationSettings(
            enable_cross_validation=True,      # Full validation for medium-term
            cv_horizon_days=7,                 # 1 week validation horizon
            enable_backtesting=True            # Enable backtesting
        )

        return cls(
            name=name,
            timeframe=timeframe,
            horizon_type=HorizonType.MEDIUM,
            forecast_periods=168,              # 1 week ahead
            prophet_params=prophet_params,
            validation=validation,
            description=f"Medium-term forecasting for {timeframe} swing trading"
        )

    @classmethod
    def create_long_config(cls, name: str, timeframe: str = "1d") -> 'HorizonConfig':
        """Create optimized configuration for long horizons (position trading)"""
        prophet_params = ProphetParameters(
            growth="logistic",                 # Logistic growth for long-term
            changepoint_prior_scale=0.01,      # Less flexibility, more stable
            seasonality_prior_scale=15.0,      # Strong seasonality
            yearly_seasonality=True,           # Yearly patterns important
            weekly_seasonality=True,           # Weekly patterns relevant
            daily_seasonality=False,           # Not relevant for daily data
            mcmc_samples=200,                  # MCMC for better uncertainty
            uncertainty_samples=1000           # Full uncertainty sampling
        )

        validation = ValidationSettings(
            cv_horizon_days=30,                # 1 month validation
            cv_period_days=90,                 # 3 months periods
            cv_initial_days=730,               # 2 years initial training
            enable_backtesting=True            # Full backtesting
        )

        return cls(
            name=name,
            timeframe=timeframe,
            horizon_type=HorizonType.LONG,
            forecast_periods=30,               # 1 month ahead
            prophet_params=prophet_params,
            validation=validation,
            enable_ensemble=True,              # Ensemble for long-term accuracy
            description=f"Long-term forecasting for {timeframe} position trading"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def _convert_value(value):
            if hasattr(value, '__dict__'):
                return {k: _convert_value(v) for k, v in value.__dict__.items()}
            elif isinstance(value, Enum):
                return value.value
            elif isinstance(value, list):
                return [_convert_value(v) for v in value]
            else:
                return value

        return _convert_value(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HorizonConfig':
        """Create configuration from dictionary"""
        # Convert enum strings back to enums
        if 'horizon_type' in data:
            data['horizon_type'] = HorizonType(data['horizon_type'])
        if 'model_type' in data:
            data['model_type'] = ModelType(data['model_type'])

        # Reconstruct nested objects
        if 'prophet_params' in data and isinstance(data['prophet_params'], dict):
            data['prophet_params'] = ProphetParameters(**data['prophet_params'])

        if 'performance' in data and isinstance(data['performance'], dict):
            data['performance'] = PerformanceConstraints(**data['performance'])

        if 'validation' in data and isinstance(data['validation'], dict):
            data['validation'] = ValidationSettings(**data['validation'])

        if 'data_processing' in data and isinstance(data['data_processing'], dict):
            data['data_processing'] = DataProcessingConfig(**data['data_processing'])

        return cls(**data)

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'HorizonConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def clone(self, **overrides) -> 'HorizonConfig':
        """Create a copy of this configuration with optional overrides"""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.from_dict(config_dict)

    def get_timeframe_minutes(self) -> int:
        """Convert timeframe string to minutes"""
        import re
        match = re.match(r'(\d+)([mhd]|min|hour|day)', self.timeframe.lower())
        if not match:
            raise ValueError(f"Invalid timeframe format: {self.timeframe}")

        value, unit = match.groups()
        value = int(value)

        if unit in ['m', 'min']:
            return value
        elif unit in ['h', 'hour']:
            return value * 60
        elif unit in ['d', 'day']:
            return value * 60 * 24
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    def __str__(self) -> str:
        """String representation of configuration"""
        return f"HorizonConfig(name='{self.name}', timeframe='{self.timeframe}', type={self.horizon_type.value})"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"HorizonConfig(name='{self.name}', timeframe='{self.timeframe}', "
                f"type={self.horizon_type.value}, model={self.model_type.value}, "
                f"periods={self.forecast_periods})")