"""
Multi-Horizon Forecasting Engine
ML-Framework-1329 - Enterprise-grade concurrent prediction orchestration

 2025: Adaptive model selection, performance optimization,
sub-100ms targets with fault tolerance.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import time

import pandas as pd
import numpy as np
from prophet import Prophet

from .horizon_config import HorizonConfig, HorizonType
from .ensemble_forecaster import EnsembleForecaster
from .model_orchestrator import ModelOrchestrator
from .horizon_optimizer import HorizonOptimizer
from .performance_monitor import HorizonPerformanceMonitor


class EngineState(Enum):
    """Engine operational states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    READY = "ready"
    PREDICTING = "predicting"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class PredictionStrategy(Enum):
    """Multi-horizon prediction strategies"""
    PARALLEL = "parallel"           # All horizons simultaneously
    SEQUENTIAL = "sequential"       # One by one
    ADAPTIVE = "adaptive"          # Smart scheduling based on complexity
    PRIORITY_BASED = "priority"    # High-frequency first


@dataclass
class HorizonPrediction:
    """Single horizon prediction result"""
    horizon: str
    timeframe: str
    prediction_data: pd.DataFrame
    confidence_score: float
    execution_time_ms: float
    model_metadata: Dict
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None and not self.prediction_data.empty


@dataclass
class MultiHorizonResult:
    """Complete multi-horizon prediction result"""
    symbol: str
    timestamp: datetime
    horizons: Dict[str, HorizonPrediction]
    ensemble_prediction: Optional[pd.DataFrame] = None
    total_execution_time_ms: float = 0.0
    strategy_used: str = "parallel"
    metadata: Dict = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate across horizons"""
        if not self.horizons:
            return 0.0
        successful = sum(1 for h in self.horizons.values() if h.is_valid)
        return successful / len(self.horizons)

    @property
    def average_confidence(self) -> float:
        """Average confidence across valid predictions"""
        valid_horizons = [h for h in self.horizons.values() if h.is_valid]
        if not valid_horizons:
            return 0.0
        return np.mean([h.confidence_score for h in valid_horizons])


class MultiHorizonEngine:
    """
    Enterprise multi-horizon forecasting engine with enterprise patterns.

    Orchestrates concurrent predictions across multiple time horizons
    with adaptive model selection and performance optimization.
    """

    def __init__(
        self,
        symbol: str,
        horizons_config: List[HorizonConfig],
        max_workers: int = 4,
        performance_target_ms: float = 85.0,
        enable_ensemble: bool = True,
        cache_predictions: bool = True
    ):
        self.symbol = symbol
        self.horizons_config = {h.name: h for h in horizons_config}
        self.max_workers = min(max_workers, len(horizons_config))
        self.performance_target_ms = performance_target_ms
        self.enable_ensemble = enable_ensemble
        self.cache_predictions = cache_predictions

        # Core components
        self.model_orchestrator = ModelOrchestrator(symbol)
        self.ensemble_forecaster = EnsembleForecaster()
        self.optimizer = HorizonOptimizer(performance_target_ms)
        self.performance_monitor = HorizonPerformanceMonitor()

        # State management
        self.state = EngineState.IDLE
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.prediction_cache: Dict[str, Tuple[MultiHorizonResult, datetime]] = {}

        # Performance tracking
        self.execution_history: List[Dict] = []
        self.current_strategy = PredictionStrategy.PARALLEL

        # Logging
        self.logger = logging.getLogger(f"MultiHorizonEngine.{symbol}")

        self.logger.info(f"MultiHorizonEngine initialized for {symbol}")
        self.logger.info(f"Horizons: {list(self.horizons_config.keys())}")
        self.logger.info(f"Performance target: {performance_target_ms}ms")

    async def initialize(self) -> bool:
        """Initialize engine and prepare models"""
        try:
            self.state = EngineState.INITIALIZING
            self.logger.info("Initializing multi-horizon engine...")

            # Initialize model orchestrator
            await self.model_orchestrator.initialize()

            # Prepare horizon-specific models
            initialization_tasks = []
            for horizon_name, config in self.horizons_config.items():
                task = self._prepare_horizon_model(horizon_name, config)
                initialization_tasks.append(task)

            # Execute initialization concurrently
            start_time = time.time()
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            init_time = (time.time() - start_time) * 1000

            # Check initialization results
            failed_horizons = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    horizon_name = list(self.horizons_config.keys())[i]
                    failed_horizons.append(horizon_name)
                    self.logger.error(f"Failed to initialize horizon {horizon_name}: {result}")

            if failed_horizons:
                self.logger.warning(f"Failed to initialize horizons: {failed_horizons}")
                # Remove failed horizons from config
                for horizon in failed_horizons:
                    del self.horizons_config[horizon]

            if not self.horizons_config:
                self.state = EngineState.ERROR
                return False

            # Initialize ensemble if enabled
            if self.enable_ensemble:
                await self.ensemble_forecaster.initialize(list(self.horizons_config.keys()))

            self.state = EngineState.READY
            self.logger.info(f"Engine initialized successfully in {init_time:.1f}ms")
            self.logger.info(f"Active horizons: {len(self.horizons_config)}")

            return True

        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error(f"Engine initialization failed: {e}")
            return False

    async def predict(
        self,
        data: pd.DataFrame,
        periods_ahead: int = 30,
        strategy: Optional[PredictionStrategy] = None,
        include_ensemble: bool = True,
        use_cache: bool = True
    ) -> MultiHorizonResult:
        """
        Generate multi-horizon predictions with performance optimization.

        Args:
            data: Historical OHLCV data
            periods_ahead: Number of periods to predict
            strategy: Prediction execution strategy
            include_ensemble: Whether to generate ensemble prediction
            use_cache: Whether to use cached results

        Returns:
            MultiHorizonResult with predictions for all horizons
        """
        if self.state != EngineState.READY:
            raise RuntimeError(f"Engine not ready. Current state: {self.state}")

        # Check cache first
        cache_key = self._generate_cache_key(data, periods_ahead)
        if use_cache and self.cache_predictions:
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info("Returning cached multi-horizon result")
                return cached_result

        self.state = EngineState.PREDICTING
        prediction_start = time.time()

        try:
            # Select optimal prediction strategy
            if strategy is None:
                strategy = await self._select_optimal_strategy(data, periods_ahead)

            self.logger.info(f"Executing multi-horizon prediction with {strategy.value} strategy")

            # Execute predictions based on strategy
            horizon_results = await self._execute_prediction_strategy(
                strategy, data, periods_ahead
            )

            # Generate ensemble prediction if enabled
            ensemble_prediction = None
            if include_ensemble and self.enable_ensemble and len(horizon_results) >= 2:
                ensemble_prediction = await self._generate_ensemble_prediction(horizon_results)

            # Calculate total execution time
            total_time_ms = (time.time() - prediction_start) * 1000

            # Create result object
            result = MultiHorizonResult(
                symbol=self.symbol,
                timestamp=datetime.now(),
                horizons=horizon_results,
                ensemble_prediction=ensemble_prediction,
                total_execution_time_ms=total_time_ms,
                strategy_used=strategy.value,
                metadata={
                    'data_points': len(data),
                    'periods_ahead': periods_ahead,
                    'cache_used': False,
                    'performance_target_met': total_time_ms <= self.performance_target_ms
                }
            )

            # Cache the result
            if self.cache_predictions:
                self._cache_result(cache_key, result)

            # Update performance monitoring
            self.performance_monitor.record_execution(result)

            # Log performance metrics
            self.logger.info(f"Multi-horizon prediction completed in {total_time_ms:.1f}ms")
            self.logger.info(f"Success rate: {result.success_rate:.1%}")
            self.logger.info(f"Average confidence: {result.average_confidence:.3f}")

            # Store execution history
            self.execution_history.append({
                'timestamp': datetime.now(),
                'total_time_ms': total_time_ms,
                'strategy': strategy.value,
                'success_rate': result.success_rate,
                'performance_target_met': result.metadata['performance_target_met']
            })

            # Trigger optimization if performance degraded
            if total_time_ms > self.performance_target_ms * 1.2:  # 20% tolerance
                asyncio.create_task(self._optimize_performance())

            return result

        except Exception as e:
            self.logger.error(f"Multi-horizon prediction failed: {e}")
            raise
        finally:
            self.state = EngineState.READY

    async def _execute_prediction_strategy(
        self,
        strategy: PredictionStrategy,
        data: pd.DataFrame,
        periods_ahead: int
    ) -> Dict[str, HorizonPrediction]:
        """Execute predictions based on selected strategy"""

        if strategy == PredictionStrategy.PARALLEL:
            return await self._predict_parallel(data, periods_ahead)
        elif strategy == PredictionStrategy.SEQUENTIAL:
            return await self._predict_sequential(data, periods_ahead)
        elif strategy == PredictionStrategy.ADAPTIVE:
            return await self._predict_adaptive(data, periods_ahead)
        elif strategy == PredictionStrategy.PRIORITY_BASED:
            return await self._predict_priority_based(data, periods_ahead)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _predict_parallel(
        self,
        data: pd.DataFrame,
        periods_ahead: int
    ) -> Dict[str, HorizonPrediction]:
        """Execute all horizon predictions in parallel"""

        prediction_tasks = []
        for horizon_name, config in self.horizons_config.items():
            task = self._predict_single_horizon(horizon_name, config, data, periods_ahead)
            prediction_tasks.append(task)

        # Execute all predictions concurrently
        results = await asyncio.gather(*prediction_tasks, return_exceptions=True)

        # Process results
        horizon_results = {}
        for i, result in enumerate(results):
            horizon_name = list(self.horizons_config.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Horizon {horizon_name} prediction failed: {result}")
                # Create error prediction
                horizon_results[horizon_name] = HorizonPrediction(
                    horizon=horizon_name,
                    timeframe=self.horizons_config[horizon_name].timeframe,
                    prediction_data=pd.DataFrame(),
                    confidence_score=0.0,
                    execution_time_ms=0.0,
                    model_metadata={},
                    error=str(result)
                )
            else:
                horizon_results[horizon_name] = result

        return horizon_results

    async def _predict_sequential(
        self,
        data: pd.DataFrame,
        periods_ahead: int
    ) -> Dict[str, HorizonPrediction]:
        """Execute horizon predictions one by one"""

        horizon_results = {}
        for horizon_name, config in self.horizons_config.items():
            try:
                result = await self._predict_single_horizon(
                    horizon_name, config, data, periods_ahead
                )
                horizon_results[horizon_name] = result
            except Exception as e:
                self.logger.error(f"Horizon {horizon_name} prediction failed: {e}")
                horizon_results[horizon_name] = HorizonPrediction(
                    horizon=horizon_name,
                    timeframe=config.timeframe,
                    prediction_data=pd.DataFrame(),
                    confidence_score=0.0,
                    execution_time_ms=0.0,
                    model_metadata={},
                    error=str(e)
                )

        return horizon_results

    async def _predict_single_horizon(
        self,
        horizon_name: str,
        config: HorizonConfig,
        data: pd.DataFrame,
        periods_ahead: int
    ) -> HorizonPrediction:
        """Predict for a single horizon with performance monitoring"""

        start_time = time.time()

        try:
            # Get horizon-specific model
            model = await self.model_orchestrator.get_model(horizon_name, config)

            # Prepare data for this specific horizon
            processed_data = await self._prepare_horizon_data(data, config)

            # Execute prediction
            prediction_df = await self._execute_horizon_prediction(
                model, processed_data, periods_ahead, config
            )

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                prediction_df, config, processed_data
            )

            execution_time_ms = (time.time() - start_time) * 1000

            return HorizonPrediction(
                horizon=horizon_name,
                timeframe=config.timeframe,
                prediction_data=prediction_df,
                confidence_score=confidence_score,
                execution_time_ms=execution_time_ms,
                model_metadata={
                    'model_type': type(model).__name__,
                    'data_points_used': len(processed_data),
                    'config': config.__dict__
                }
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Single horizon prediction failed for {horizon_name}: {e}")

            return HorizonPrediction(
                horizon=horizon_name,
                timeframe=config.timeframe,
                prediction_data=pd.DataFrame(),
                confidence_score=0.0,
                execution_time_ms=execution_time_ms,
                model_metadata={},
                error=str(e)
            )

    def get_performance_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if not self.execution_history:
            return {}

        recent_executions = self.execution_history[-10:]  # Last 10 executions

        return {
            'total_executions': len(self.execution_history),
            'average_execution_time_ms': np.mean([e['total_time_ms'] for e in recent_executions]),
            'performance_target_met_rate': np.mean([e['performance_target_met'] for e in recent_executions]),
            'average_success_rate': np.mean([e['success_rate'] for e in recent_executions]),
            'current_strategy': self.current_strategy.value,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'state': self.state.value,
            'active_horizons': list(self.horizons_config.keys()),
            'performance_monitor': self.performance_monitor.get_metrics()
        }

    async def optimize_performance(self) -> Dict:
        """Manual performance optimization trigger"""
        return await self._optimize_performance()

    async def _optimize_performance(self) -> Dict:
        """Optimize engine performance based on historical data"""
        if self.state == EngineState.OPTIMIZING:
            return {'status': 'already_optimizing'}

        self.state = EngineState.OPTIMIZING

        try:
            optimization_result = await self.optimizer.optimize_engine(
                self.execution_history,
                self.horizons_config,
                self.performance_monitor.get_metrics()
            )

            # Apply optimization recommendations
            if 'optimal_strategy' in optimization_result:
                self.current_strategy = PredictionStrategy(optimization_result['optimal_strategy'])

            if 'recommended_max_workers' in optimization_result:
                new_workers = optimization_result['recommended_max_workers']
                if new_workers != self.max_workers:
                    self.max_workers = new_workers
                    self.executor.shutdown(wait=True)
                    self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

            self.logger.info(f"Performance optimization completed: {optimization_result}")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return {'status': 'failed', 'error': str(e)}
        finally:
            self.state = EngineState.READY

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.prediction_cache.clear()
        self.execution_history.clear()
        self.logger.info("MultiHorizonEngine cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()