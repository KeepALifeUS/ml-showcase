"""
Model Orchestrator System
ML-Framework-1329 - Multi-horizon model lifecycle management

 2025: Adaptive model selection, health monitoring,
performance-optimized model orchestration.
"""

import asyncio
import logging
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json

import pandas as pd
import numpy as np
from prophet import Prophet

from .horizon_config import HorizonConfig, ModelType, ProphetParameters


class ModelState(Enum):
    """Model lifecycle states"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    TRAINING = "training"
    READY = "ready"
    PREDICTING = "predicting"
    RETRAINING = "retraining"
    DEPRECATED = "deprecated"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ModelHealth(Enum):
    """Model health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    model_id: str
    horizon_name: str
    model_type: ModelType
    state: ModelState
    health: ModelHealth
    created_at: datetime
    last_trained: Optional[datetime] = None
    last_used: Optional[datetime] = None
    training_data_hash: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    version: str = "1.0"
    config_hash: Optional[str] = None

    def update_usage(self):
        """Update last used timestamp"""
        self.last_used = datetime.now()

    def is_stale(self, max_age_hours: int = 24) -> bool:
        """Check if model is stale and needs retraining"""
        if not self.last_trained:
            return True
        age = datetime.now() - self.last_trained
        return age > timedelta(hours=max_age_hours)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_id': self.model_id,
            'horizon_name': self.horizon_name,
            'model_type': self.model_type.value,
            'state': self.state.value,
            'health': self.health.value,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'training_data_hash': self.training_data_hash,
            'performance_metrics': self.performance_metrics,
            'version': self.version,
            'config_hash': self.config_hash
        }


@dataclass
class ModelInstance:
    """Model instance with metadata and actual model"""
    metadata: ModelMetadata
    model: Any  # The actual trained model
    config: HorizonConfig
    training_data: Optional[pd.DataFrame] = None

    @property
    def is_ready(self) -> bool:
        """Check if model is ready for predictions"""
        return (self.metadata.state == ModelState.READY and
                self.metadata.health in [ModelHealth.HEALTHY, ModelHealth.WARNING])

    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of training data for change detection"""
        if data.empty:
            return ""

        # Create hash from data shape and key statistics
        data_str = f"{data.shape}_{data.describe().to_string()}"
        return hashlib.md5(data_str.encode()).hexdigest()

    def needs_retraining(self, new_data: pd.DataFrame) -> bool:
        """Check if model needs retraining based on data changes"""
        if self.metadata.training_data_hash is None:
            return True

        new_hash = self.calculate_data_hash(new_data)
        return new_hash != self.metadata.training_data_hash


class ModelOrchestrator:
    """
    Orchestrates model lifecycle across multiple horizons.

    Manages model creation, training, health monitoring, and optimization
    with enterprise patterns for enterprise reliability.
    """

    def __init__(
        self,
        symbol: str,
        model_cache_dir: Optional[str] = None,
        max_concurrent_training: int = 2,
        model_retention_hours: int = 48,
        auto_retrain: bool = True
    ):
        self.symbol = symbol
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path(f"./models/{symbol}")
        self.max_concurrent_training = max_concurrent_training
        self.model_retention_hours = model_retention_hours
        self.auto_retrain = auto_retrain

        # Model storage
        self.models: Dict[str, ModelInstance] = {}
        self.model_registry: Dict[str, ModelMetadata] = {}

        # Performance tracking
        self.training_history: List[Dict] = []
        self.performance_history: Dict[str, List[Dict]] = {}

        # Concurrency control
        self.training_executor = ThreadPoolExecutor(max_workers=max_concurrent_training)
        self.training_locks: Dict[str, asyncio.Lock] = {}

        # Health monitoring
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = datetime.now()

        # Setup logging
        self.logger = logging.getLogger(f"ModelOrchestrator.{symbol}")

        # Ensure model cache directory exists
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"ModelOrchestrator initialized for {symbol}")

    async def initialize(self):
        """Initialize orchestrator and load existing models"""
        try:
            # Load existing model registry
            await self._load_model_registry()

            # Load cached models
            await self._load_cached_models()

            # Start health monitoring
            asyncio.create_task(self._health_monitoring_loop())

            self.logger.info(f"Orchestrator initialized with {len(self.models)} models")

        except Exception as e:
            self.logger.error(f"Orchestrator initialization failed: {e}")
            raise

    async def get_model(
        self,
        horizon_name: str,
        config: HorizonConfig,
        training_data: Optional[pd.DataFrame] = None,
        force_retrain: bool = False
    ) -> Any:
        """
        Get or create model for specific horizon.

        Args:
            horizon_name: Unique horizon identifier
            config: Horizon configuration
            training_data: Training data (required for initial training)
            force_retrain: Force model retraining

        Returns:
            Trained model ready for predictions
        """
        model_id = self._generate_model_id(horizon_name, config)

        # Get or create training lock for this model
        if model_id not in self.training_locks:
            self.training_locks[model_id] = asyncio.Lock()

        async with self.training_locks[model_id]:
            # Check if model exists and is ready
            if model_id in self.models and not force_retrain:
                model_instance = self.models[model_id]

                # Check if retraining is needed
                if training_data is not None and model_instance.needs_retraining(training_data):
                    self.logger.info(f"Data changed, retraining model {model_id}")
                    force_retrain = True
                elif model_instance.is_ready:
                    model_instance.metadata.update_usage()
                    return model_instance.model

            # Create or retrain model
            if force_retrain or model_id not in self.models:
                if training_data is None:
                    raise ValueError(f"Training data required for model {model_id}")

                model_instance = await self._create_and_train_model(
                    model_id, horizon_name, config, training_data
                )
                self.models[model_id] = model_instance

            return self.models[model_id].model

    async def _create_and_train_model(
        self,
        model_id: str,
        horizon_name: str,
        config: HorizonConfig,
        training_data: pd.DataFrame
    ) -> ModelInstance:
        """Create and train a new model instance"""

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            horizon_name=horizon_name,
            model_type=config.model_type,
            state=ModelState.INITIALIZING,
            health=ModelHealth.UNKNOWN,
            created_at=datetime.now(),
            config_hash=self._calculate_config_hash(config)
        )

        # Create model instance
        model_instance = ModelInstance(
            metadata=metadata,
            model=None,
            config=config,
            training_data=training_data.copy()
        )

        try:
            metadata.state = ModelState.TRAINING
            self.logger.info(f"Training model {model_id} for horizon {horizon_name}")

            # Train model based on type
            if config.model_type == ModelType.PROPHET:
                trained_model = await self._train_prophet_model(config, training_data)
            elif config.model_type == ModelType.PROPHET_EXTENDED:
                trained_model = await self._train_prophet_extended_model(config, training_data)
            else:
                raise ValueError(f"Unsupported model type: {config.model_type}")

            # Update model instance
            model_instance.model = trained_model
            metadata.state = ModelState.READY
            metadata.health = ModelHealth.HEALTHY
            metadata.last_trained = datetime.now()
            metadata.training_data_hash = model_instance.calculate_data_hash(training_data)

            # Validate model performance
            performance_metrics = await self._validate_model_performance(
                model_instance, training_data
            )
            metadata.performance_metrics = performance_metrics

            # Update health based on performance
            metadata.health = self._assess_model_health(performance_metrics)

            # Cache model
            await self._cache_model(model_instance)

            # Update registry
            self.model_registry[model_id] = metadata

            # Record training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'model_id': model_id,
                'horizon_name': horizon_name,
                'model_type': config.model_type.value,
                'training_samples': len(training_data),
                'performance_metrics': performance_metrics,
                'training_duration_ms': 0  # Will be updated by caller
            })

            self.logger.info(f"Model {model_id} trained successfully")
            return model_instance

        except Exception as e:
            metadata.state = ModelState.ERROR
            metadata.health = ModelHealth.CRITICAL
            self.logger.error(f"Model training failed for {model_id}: {e}")
            raise

    async def _train_prophet_model(
        self,
        config: HorizonConfig,
        training_data: pd.DataFrame
    ) -> Prophet:
        """Train Prophet model with configuration"""

        # Prepare data for Prophet
        prophet_data = self._prepare_prophet_data(training_data)

        # Create Prophet model with parameters
        prophet_kwargs = config.prophet_params.to_prophet_kwargs()
        model = Prophet(**prophet_kwargs)

        # Add custom seasonalities if configured
        await self._add_custom_seasonalities(model, config)

        # Add external regressors if configured
        if config.external_regressors:
            for regressor in config.external_regressors:
                if regressor in prophet_data.columns:
                    model.add_regressor(regressor)

        # Train model
        model.fit(prophet_data)

        return model

    async def _train_prophet_extended_model(
        self,
        config: HorizonConfig,
        training_data: pd.DataFrame
    ) -> Prophet:
        """Train Prophet model with extended features"""

        # Start with base Prophet training
        model = await self._train_prophet_model(config, training_data)

        # Add extended features
        # TODO: Implement extended features like custom holidays,
        # advanced seasonalities, etc.

        return model

    def _prepare_prophet_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for Prophet training"""
        if 'timestamp' in data.columns and 'close' in data.columns:
            prophet_data = pd.DataFrame({
                'ds': pd.to_datetime(data['timestamp']),
                'y': data['close']
            })
        elif 'ds' in data.columns and 'y' in data.columns:
            prophet_data = data[['ds', 'y']].copy()
        else:
            raise ValueError("Data must contain either (timestamp, close) or (ds, y) columns")

        # Remove any rows with missing values
        prophet_data = prophet_data.dropna()

        return prophet_data

    async def _add_custom_seasonalities(self, model: Prophet, config: HorizonConfig):
        """Add custom seasonalities based on configuration"""

        # Add crypto-specific seasonalities
        if config.horizon_type.value in ['ultra_short', 'short']:
            # Hourly seasonality for short-term predictions
            model.add_seasonality(
                name='hourly',
                period=24,
                fourier_order=5,
                prior_scale=5.0
            )

        if config.timeframe in ['1m', '5m', '15m']:
            # Intraday seasonality
            model.add_seasonality(
                name='intraday',
                period=1,
                fourier_order=10,
                prior_scale=10.0
            )

    async def _validate_model_performance(
        self,
        model_instance: ModelInstance,
        training_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Validate model performance and calculate metrics"""

        try:
            model = model_instance.model
            config = model_instance.config

            # Prepare validation data
            prophet_data = self._prepare_prophet_data(training_data)

            if len(prophet_data) < config.validation.min_training_days:
                return {'validation_error': 1.0, 'insufficient_data': True}

            # Perform cross-validation if enabled
            if config.validation.enable_cross_validation and len(prophet_data) >= 100:
                cv_results = await self._perform_cross_validation(model, prophet_data, config)
                return cv_results
            else:
                # Simple validation on training data
                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)

                # Calculate basic metrics
                actual = prophet_data['y'].values
                predicted = forecast['yhat'].values

                mae = np.mean(np.abs(actual - predicted))
                mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))

                return {
                    'mae': mae,
                    'mape': mape,
                    'rmse': rmse,
                    'training_samples': len(prophet_data)
                }

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return {'validation_error': 1.0, 'error': str(e)}

    async def _perform_cross_validation(
        self,
        model: Prophet,
        data: pd.DataFrame,
        config: HorizonConfig
    ) -> Dict[str, float]:
        """Perform cross-validation for model performance assessment"""

        try:
            from prophet.diagnostics import cross_validation, performance_metrics

            # Configure cross-validation parameters
            cv_horizon = f"{config.validation.cv_horizon_days} days"
            cv_period = f"{config.validation.cv_period_days} days"
            cv_initial = f"{config.validation.cv_initial_days} days"

            # Perform cross-validation
            cv_results = cross_validation(
                model,
                horizon=cv_horizon,
                period=cv_period,
                initial=cv_initial
            )

            # Calculate performance metrics
            metrics = performance_metrics(cv_results)

            # Extract key metrics
            return {
                'cv_mae': metrics['mae'].mean(),
                'cv_mape': metrics['mape'].mean(),
                'cv_rmse': metrics['rmse'].mean(),
                'cv_coverage': metrics['coverage'].mean(),
                'cv_samples': len(cv_results)
            }

        except Exception as e:
            self.logger.warning(f"Cross-validation failed, using simple validation: {e}")
            return {'cv_error': 1.0, 'error': str(e)}

    def _assess_model_health(self, performance_metrics: Dict[str, float]) -> ModelHealth:
        """Assess model health based on performance metrics"""

        if 'validation_error' in performance_metrics or 'cv_error' in performance_metrics:
            return ModelHealth.CRITICAL

        # Check MAPE (Mean Absolute Percentage Error)
        mape = performance_metrics.get('cv_mape', performance_metrics.get('mape', 100))

        if mape < 5:
            return ModelHealth.HEALTHY
        elif mape < 15:
            return ModelHealth.WARNING
        elif mape < 30:
            return ModelHealth.DEGRADED
        else:
            return ModelHealth.CRITICAL

    async def _cache_model(self, model_instance: ModelInstance):
        """Cache trained model to disk"""
        try:
            model_path = self.model_cache_dir / f"{model_instance.metadata.model_id}.pkl"
            metadata_path = self.model_cache_dir / f"{model_instance.metadata.model_id}_metadata.json"

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance.model, f)

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(model_instance.metadata.to_dict(), f, indent=2)

            self.logger.debug(f"Model {model_instance.metadata.model_id} cached")

        except Exception as e:
            self.logger.warning(f"Failed to cache model {model_instance.metadata.model_id}: {e}")

    def _generate_model_id(self, horizon_name: str, config: HorizonConfig) -> str:
        """Generate unique model ID"""
        config_str = f"{horizon_name}_{config.timeframe}_{config.model_type.value}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{self.symbol}_{horizon_name}_{config_hash}"

    def _calculate_config_hash(self, config: HorizonConfig) -> str:
        """Calculate hash of configuration for change detection"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    async def _health_monitoring_loop(self):
        """Background health monitoring for all models"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")

    async def _perform_health_checks(self):
        """Perform health checks on all models"""
        current_time = datetime.now()

        for model_id, model_instance in self.models.items():
            try:
                # Check if model is stale
                if model_instance.metadata.is_stale(self.model_retention_hours):
                    model_instance.metadata.health = ModelHealth.WARNING
                    self.logger.warning(f"Model {model_id} is stale and may need retraining")

                # Check memory usage and performance
                # TODO: Implement memory and performance monitoring

            except Exception as e:
                self.logger.error(f"Health check failed for model {model_id}: {e}")

        self.last_health_check = current_time

    async def _load_model_registry(self):
        """Load existing model registry from disk"""
        registry_path = self.model_cache_dir / "model_registry.json"

        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)

                for model_id, metadata_dict in registry_data.items():
                    # Reconstruct metadata object
                    metadata = ModelMetadata(**metadata_dict)
                    self.model_registry[model_id] = metadata

                self.logger.info(f"Loaded {len(self.model_registry)} models from registry")

            except Exception as e:
                self.logger.error(f"Failed to load model registry: {e}")

    async def _load_cached_models(self):
        """Load cached models from disk"""
        for model_id, metadata in self.model_registry.items():
            try:
                model_path = self.model_cache_dir / f"{model_id}.pkl"

                if model_path.exists():
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    # Create model instance (config will be loaded when needed)
                    model_instance = ModelInstance(
                        metadata=metadata,
                        model=model,
                        config=None  # Will be set when model is requested
                    )

                    self.models[model_id] = model_instance
                    self.logger.debug(f"Loaded cached model {model_id}")

            except Exception as e:
                self.logger.warning(f"Failed to load cached model {model_id}: {e}")

    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            'symbol': self.symbol,
            'total_models': len(self.models),
            'ready_models': sum(1 for m in self.models.values() if m.is_ready),
            'model_health_summary': self._get_health_summary(),
            'recent_training_count': len([h for h in self.training_history if
                                        (datetime.now() - h['timestamp']).hours < 24]),
            'cache_directory': str(self.model_cache_dir),
            'last_health_check': self.last_health_check.isoformat(),
            'auto_retrain_enabled': self.auto_retrain
        }

    def _get_health_summary(self) -> Dict[str, int]:
        """Get summary of model health status"""
        health_counts = {}
        for health in ModelHealth:
            health_counts[health.value] = 0

        for model_instance in self.models.values():
            health = model_instance.metadata.health
            health_counts[health.value] += 1

        return health_counts

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'training_executor'):
            self.training_executor.shutdown(wait=True)

        # Save model registry
        try:
            registry_path = self.model_cache_dir / "model_registry.json"
            registry_data = {
                model_id: metadata.to_dict()
                for model_id, metadata in self.model_registry.items()
            }

            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save model registry: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()