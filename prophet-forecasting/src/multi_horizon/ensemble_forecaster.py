"""
Ensemble Forecasting System
ML-Framework-1329 - Multi-horizon ensemble prediction aggregation

 2025: Adaptive model combination, confidence weighting,
performance-optimized ensemble strategies.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .horizon_config import HorizonConfig, HorizonType


class EnsembleStrategy(Enum):
    """Ensemble combination strategies"""
    SIMPLE_AVERAGE = "simple_average"           # Equal weights
    WEIGHTED_AVERAGE = "weighted_average"       # Confidence-based weights
    MEDIAN_ENSEMBLE = "median_ensemble"         # Robust median
    BAYESIAN_MODEL_AVERAGE = "bayesian_model_average"  # Bayesian weighting
    DYNAMIC_WEIGHTING = "dynamic_weighting"     # Time-varying weights
    CONFIDENCE_RANKING = "confidence_ranking"   # Rank by confidence
    VOLATILITY_ADJUSTED = "volatility_adjusted" # Volatility-based weighting


class TimeAlignment(Enum):
    """Time alignment strategies for multi-horizon ensemble"""
    NEAREST_NEIGHBOR = "nearest_neighbor"       # Match to nearest time point
    LINEAR_INTERPOLATION = "linear_interpolation"  # Linear interpolation
    SPLINE_INTERPOLATION = "spline_interpolation"  # Spline interpolation
    FORWARD_FILL = "forward_fill"               # Forward fill missing values
    BACKWARD_FILL = "backward_fill"             # Backward fill missing values


@dataclass
class EnsembleWeights:
    """Weights for ensemble combination"""
    horizon_weights: Dict[str, float]           # Per-horizon weights
    time_decay_factor: float = 0.95             # Temporal weight decay
    confidence_scaling: float = 1.0             # Confidence weight scaling
    performance_factor: float = 1.0             # Historical performance factor
    volatility_adjustment: float = 1.0          # Volatility-based adjustment

    def normalize_weights(self):
        """Normalize horizon weights to sum to 1"""
        total = sum(self.horizon_weights.values())
        if total > 0:
            self.horizon_weights = {
                h: w / total for h, w in self.horizon_weights.items()
            }

    def apply_decay(self, time_steps: int):
        """Apply time decay to weights"""
        decay = self.time_decay_factor ** time_steps
        for horizon in self.horizon_weights:
            self.horizon_weights[horizon] *= decay


@dataclass
class EnsemblePrediction:
    """Result of ensemble forecasting"""
    ensemble_forecast: pd.DataFrame             # Combined forecast
    individual_forecasts: Dict[str, pd.DataFrame]  # Individual horizon forecasts
    weights_used: EnsembleWeights              # Weights applied
    strategy_used: str                         # Strategy used
    confidence_score: float                    # Overall confidence
    performance_metrics: Dict[str, float]      # Performance metrics
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def forecast_period_days(self) -> int:
        """Get forecast period in days"""
        if self.ensemble_forecast.empty:
            return 0
        start_date = self.ensemble_forecast['ds'].min()
        end_date = self.ensemble_forecast['ds'].max()
        return (end_date - start_date).days

    @property
    def total_forecasts_combined(self) -> int:
        """Get total number of individual forecasts combined"""
        return len(self.individual_forecasts)


class EnsembleForecaster:
    """
    Advanced ensemble forecasting system for multi-horizon predictions.

    Combines forecasts from different time horizons using various
    strategies optimized for trading applications.
    """

    def __init__(
        self,
        default_strategy: EnsembleStrategy = EnsembleStrategy.WEIGHTED_AVERAGE,
        alignment_method: TimeAlignment = TimeAlignment.LINEAR_INTERPOLATION,
        confidence_threshold: float = 0.5,
        max_workers: int = 2
    ):
        self.default_strategy = default_strategy
        self.alignment_method = alignment_method
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_workers

        # Performance tracking
        self.strategy_performance: Dict[str, List[float]] = {}
        self.execution_history: List[Dict] = []

        # Adaptive components
        self.learned_weights: Dict[str, EnsembleWeights] = {}
        self.performance_decay = 0.95  # Decay factor for historical performance

        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.logger = logging.getLogger("EnsembleForecaster")

        self.logger.info(f"EnsembleForecaster initialized with {default_strategy.value} strategy")

    async def initialize(self, horizon_names: List[str]):
        """Initialize ensemble forecaster with horizon names"""
        self.horizon_names = horizon_names

        # Initialize learned weights for each horizon combination
        for combination in self._generate_horizon_combinations():
            self.learned_weights[combination] = EnsembleWeights(
                horizon_weights={h: 1.0 / len(horizon_names) for h in horizon_names}
            )

        self.logger.info(f"Initialized ensemble for horizons: {horizon_names}")

    def _generate_horizon_combinations(self) -> List[str]:
        """Generate all possible horizon combinations for weight learning"""
        combinations = []
        n = len(self.horizon_names)

        # Generate combinations of different sizes
        for i in range(1, n + 1):
            from itertools import combinations as iter_combinations
            for combo in iter_combinations(self.horizon_names, i):
                combinations.append(",".join(sorted(combo)))

        return combinations

    async def create_ensemble(
        self,
        horizon_predictions: Dict[str, pd.DataFrame],
        strategy: Optional[EnsembleStrategy] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        target_periods: Optional[int] = None
    ) -> EnsemblePrediction:
        """
        Create ensemble prediction from multiple horizon forecasts.

        Args:
            horizon_predictions: Dictionary of horizon name -> forecast DataFrame
            strategy: Ensemble strategy to use (default: self.default_strategy)
            custom_weights: Custom weights for horizons
            target_periods: Target number of forecast periods

        Returns:
            EnsemblePrediction with combined forecast and metadata
        """
        if not horizon_predictions:
            raise ValueError("No horizon predictions provided")

        strategy = strategy or self.default_strategy
        start_time = datetime.now()

        try:
            # Filter valid predictions
            valid_predictions = self._filter_valid_predictions(horizon_predictions)
            if not valid_predictions:
                raise ValueError("No valid predictions available for ensemble")

            # Align time series data
            aligned_predictions = await self._align_time_series(
                valid_predictions, target_periods
            )

            # Calculate ensemble weights
            ensemble_weights = await self._calculate_ensemble_weights(
                aligned_predictions, strategy, custom_weights
            )

            # Create ensemble forecast
            ensemble_forecast = await self._combine_forecasts(
                aligned_predictions, ensemble_weights, strategy
            )

            # Calculate confidence and performance metrics
            confidence_score = self._calculate_ensemble_confidence(
                aligned_predictions, ensemble_weights
            )

            performance_metrics = self._calculate_performance_metrics(
                aligned_predictions, ensemble_forecast
            )

            # Create result
            result = EnsemblePrediction(
                ensemble_forecast=ensemble_forecast,
                individual_forecasts=aligned_predictions,
                weights_used=ensemble_weights,
                strategy_used=strategy.value,
                confidence_score=confidence_score,
                performance_metrics=performance_metrics,
                metadata={
                    'horizons_combined': list(valid_predictions.keys()),
                    'alignment_method': self.alignment_method.value,
                    'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                    'target_periods': target_periods,
                    'custom_weights_used': custom_weights is not None
                }
            )

            # Update performance tracking
            await self._update_performance_tracking(result)

            self.logger.info(
                f"Ensemble created: {len(valid_predictions)} horizons, "
                f"confidence: {confidence_score:.3f}, "
                f"strategy: {strategy.value}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            raise

    def _filter_valid_predictions(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Filter out invalid or empty predictions"""
        valid = {}

        for horizon, forecast_df in predictions.items():
            if forecast_df is None or forecast_df.empty:
                self.logger.warning(f"Skipping empty prediction for horizon: {horizon}")
                continue

            # Check required columns
            required_cols = ['ds', 'yhat']
            if not all(col in forecast_df.columns for col in required_cols):
                self.logger.warning(f"Skipping prediction with missing columns for horizon: {horizon}")
                continue

            # Check for reasonable data
            if len(forecast_df) < 2:
                self.logger.warning(f"Skipping prediction with insufficient data for horizon: {horizon}")
                continue

            valid[horizon] = forecast_df

        return valid

    async def _align_time_series(
        self,
        predictions: Dict[str, pd.DataFrame],
        target_periods: Optional[int]
    ) -> Dict[str, pd.DataFrame]:
        """Align time series from different horizons to common time grid"""

        # Determine common time range
        all_dates = []
        for forecast_df in predictions.values():
            all_dates.extend(forecast_df['ds'].tolist())

        if not all_dates:
            return predictions

        # Create common time grid
        min_date = min(all_dates)
        max_date = max(all_dates)

        # Determine appropriate frequency based on predictions
        freq = self._infer_common_frequency(predictions)

        # Generate target periods if specified
        if target_periods:
            time_grid = pd.date_range(start=min_date, periods=target_periods, freq=freq)
        else:
            time_grid = pd.date_range(start=min_date, end=max_date, freq=freq)

        # Align each prediction to common grid
        aligned_predictions = {}

        alignment_tasks = []
        for horizon, forecast_df in predictions.items():
            task = self._align_single_forecast(horizon, forecast_df, time_grid)
            alignment_tasks.append(task)

        # Execute alignment in parallel
        alignment_results = await asyncio.gather(*alignment_tasks)

        for (horizon, aligned_df) in alignment_results:
            if not aligned_df.empty:
                aligned_predictions[horizon] = aligned_df

        return aligned_predictions

    async def _align_single_forecast(
        self,
        horizon: str,
        forecast_df: pd.DataFrame,
        time_grid: pd.DatetimeIndex
    ) -> Tuple[str, pd.DataFrame]:
        """Align a single forecast to the target time grid"""
        try:
            # Create target DataFrame with time grid
            target_df = pd.DataFrame({'ds': time_grid})

            # Merge with forecast data
            if self.alignment_method == TimeAlignment.NEAREST_NEIGHBOR:
                # Use merge_asof for nearest neighbor
                forecast_df = forecast_df.sort_values('ds')
                target_df = target_df.sort_values('ds')
                aligned = pd.merge_asof(target_df, forecast_df, on='ds', direction='nearest')

            elif self.alignment_method == TimeAlignment.LINEAR_INTERPOLATION:
                # Linear interpolation
                merged = pd.merge(target_df, forecast_df, on='ds', how='left')
                merged = merged.set_index('ds').interpolate(method='linear').reset_index()
                aligned = merged

            elif self.alignment_method == TimeAlignment.SPLINE_INTERPOLATION:
                # Spline interpolation (requires more data points)
                merged = pd.merge(target_df, forecast_df, on='ds', how='left')
                if len(forecast_df) >= 4:  # Minimum for spline
                    merged = merged.set_index('ds').interpolate(method='spline', order=2).reset_index()
                else:
                    merged = merged.set_index('ds').interpolate(method='linear').reset_index()
                aligned = merged

            elif self.alignment_method == TimeAlignment.FORWARD_FILL:
                # Forward fill
                merged = pd.merge(target_df, forecast_df, on='ds', how='left')
                aligned = merged.fillna(method='ffill')

            elif self.alignment_method == TimeAlignment.BACKWARD_FILL:
                # Backward fill
                merged = pd.merge(target_df, forecast_df, on='ds', how='left')
                aligned = merged.fillna(method='bfill')

            else:
                raise ValueError(f"Unknown alignment method: {self.alignment_method}")

            # Remove rows with NaN values in critical columns
            aligned = aligned.dropna(subset=['yhat'])

            return horizon, aligned

        except Exception as e:
            self.logger.error(f"Failed to align forecast for horizon {horizon}: {e}")
            return horizon, pd.DataFrame()

    def _infer_common_frequency(self, predictions: Dict[str, pd.DataFrame]) -> str:
        """Infer common frequency from predictions"""
        frequencies = []

        for forecast_df in predictions.values():
            if len(forecast_df) >= 2:
                time_diff = forecast_df['ds'].iloc[1] - forecast_df['ds'].iloc[0]
                frequencies.append(time_diff)

        if frequencies:
            # Use the most common frequency
            from collections import Counter
            most_common_freq = Counter(frequencies).most_common(1)[0][0]

            # Convert to pandas frequency string
            if most_common_freq <= timedelta(minutes=1):
                return '1min'
            elif most_common_freq <= timedelta(minutes=5):
                return '5min'
            elif most_common_freq <= timedelta(hours=1):
                return '1H'
            elif most_common_freq <= timedelta(days=1):
                return '1D'
            else:
                return '1W'
        else:
            return '1H'  # Default frequency

    async def _calculate_ensemble_weights(
        self,
        predictions: Dict[str, pd.DataFrame],
        strategy: EnsembleStrategy,
        custom_weights: Optional[Dict[str, float]]
    ) -> EnsembleWeights:
        """Calculate weights for ensemble combination"""

        if custom_weights:
            # Use custom weights
            weights = EnsembleWeights(horizon_weights=custom_weights.copy())
            weights.normalize_weights()
            return weights

        # Calculate weights based on strategy
        if strategy == EnsembleStrategy.SIMPLE_AVERAGE:
            weights = self._calculate_simple_average_weights(predictions)
        elif strategy == EnsembleStrategy.WEIGHTED_AVERAGE:
            weights = await self._calculate_weighted_average_weights(predictions)
        elif strategy == EnsembleStrategy.BAYESIAN_MODEL_AVERAGE:
            weights = await self._calculate_bayesian_weights(predictions)
        elif strategy == EnsembleStrategy.DYNAMIC_WEIGHTING:
            weights = await self._calculate_dynamic_weights(predictions)
        elif strategy == EnsembleStrategy.CONFIDENCE_RANKING:
            weights = await self._calculate_confidence_ranking_weights(predictions)
        elif strategy == EnsembleStrategy.VOLATILITY_ADJUSTED:
            weights = await self._calculate_volatility_adjusted_weights(predictions)
        else:
            # Default to simple average
            weights = self._calculate_simple_average_weights(predictions)

        weights.normalize_weights()
        return weights

    def _calculate_simple_average_weights(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> EnsembleWeights:
        """Calculate equal weights for all predictions"""
        n_horizons = len(predictions)
        equal_weight = 1.0 / n_horizons

        return EnsembleWeights(
            horizon_weights={horizon: equal_weight for horizon in predictions.keys()}
        )

    async def _calculate_weighted_average_weights(
        self,
        predictions: Dict[str, pd.DataFrame]
    ) -> EnsembleWeights:
        """Calculate confidence-based weights"""
        weights = {}

        for horizon, forecast_df in predictions.items():
            # Calculate confidence based on prediction interval width
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                interval_width = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
                # Inverse relationship: smaller intervals = higher confidence = higher weight
                confidence = 1.0 / (1.0 + interval_width / forecast_df['yhat'].mean())
            else:
                # Use historical performance if available
                if horizon in self.strategy_performance:
                    recent_performance = self.strategy_performance[horizon][-10:]  # Last 10
                    confidence = np.mean(recent_performance) if recent_performance else 0.5
                else:
                    confidence = 0.5  # Default confidence

            weights[horizon] = max(0.1, confidence)  # Minimum weight of 0.1

        return EnsembleWeights(horizon_weights=weights)

    async def _combine_forecasts(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: EnsembleWeights,
        strategy: EnsembleStrategy
    ) -> pd.DataFrame:
        """Combine multiple forecasts into ensemble prediction"""

        if not predictions:
            return pd.DataFrame()

        # Get reference time series (first prediction)
        reference_horizon = list(predictions.keys())[0]
        reference_df = predictions[reference_horizon].copy()

        if strategy == EnsembleStrategy.MEDIAN_ENSEMBLE:
            # Use median combination
            return await self._combine_with_median(predictions)
        else:
            # Use weighted combination
            return await self._combine_with_weights(predictions, weights, reference_df)

    async def _combine_with_weights(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: EnsembleWeights,
        reference_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine forecasts using weighted average"""

        # Initialize ensemble with reference structure
        ensemble_df = reference_df[['ds']].copy()

        # Combine main prediction (yhat)
        ensemble_df['yhat'] = 0.0
        for horizon, forecast_df in predictions.items():
            weight = weights.horizon_weights.get(horizon, 0.0)
            if 'yhat' in forecast_df.columns:
                # Align by index for safety
                aligned_yhat = forecast_df.set_index('ds')['yhat'].reindex(ensemble_df.set_index('ds').index, method='nearest')
                ensemble_df['yhat'] += weight * aligned_yhat.fillna(0).values

        # Combine confidence intervals if available
        if any('yhat_lower' in df.columns for df in predictions.values()):
            ensemble_df['yhat_lower'] = 0.0
            ensemble_df['yhat_upper'] = 0.0

            for horizon, forecast_df in predictions.items():
                weight = weights.horizon_weights.get(horizon, 0.0)
                if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                    aligned_lower = forecast_df.set_index('ds')['yhat_lower'].reindex(ensemble_df.set_index('ds').index, method='nearest')
                    aligned_upper = forecast_df.set_index('ds')['yhat_upper'].reindex(ensemble_df.set_index('ds').index, method='nearest')
                    ensemble_df['yhat_lower'] += weight * aligned_lower.fillna(ensemble_df['yhat']).values
                    ensemble_df['yhat_upper'] += weight * aligned_upper.fillna(ensemble_df['yhat']).values

        return ensemble_df

    def _calculate_ensemble_confidence(
        self,
        predictions: Dict[str, pd.DataFrame],
        weights: EnsembleWeights
    ) -> float:
        """Calculate overall ensemble confidence score"""

        confidence_scores = []

        for horizon, forecast_df in predictions.items():
            weight = weights.horizon_weights.get(horizon, 0.0)

            # Calculate individual confidence
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                # Based on prediction interval width
                interval_width = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
                prediction_mean = forecast_df['yhat'].mean()
                if prediction_mean != 0:
                    relative_width = interval_width / abs(prediction_mean)
                    individual_confidence = max(0.0, min(1.0, 1.0 - relative_width))
                else:
                    individual_confidence = 0.5
            else:
                individual_confidence = 0.7  # Default confidence

            confidence_scores.append(weight * individual_confidence)

        return sum(confidence_scores) if confidence_scores else 0.5

    def _calculate_performance_metrics(
        self,
        predictions: Dict[str, pd.DataFrame],
        ensemble_forecast: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics for the ensemble"""

        metrics = {
            'horizons_count': len(predictions),
            'forecast_length': len(ensemble_forecast),
            'average_prediction_value': ensemble_forecast['yhat'].mean() if not ensemble_forecast.empty else 0.0,
            'prediction_volatility': ensemble_forecast['yhat'].std() if not ensemble_forecast.empty else 0.0,
        }

        # Calculate prediction spread (disagreement between models)
        if len(predictions) > 1:
            all_predictions = []
            for forecast_df in predictions.values():
                if not forecast_df.empty:
                    all_predictions.append(forecast_df['yhat'].values)

            if all_predictions:
                # Calculate standard deviation across models at each time point
                min_length = min(len(pred) for pred in all_predictions)
                truncated_predictions = [pred[:min_length] for pred in all_predictions]

                if truncated_predictions:
                    prediction_array = np.array(truncated_predictions)
                    model_disagreement = np.mean(np.std(prediction_array, axis=0))
                    metrics['model_disagreement'] = model_disagreement

        return metrics

    async def _update_performance_tracking(self, result: EnsemblePrediction):
        """Update performance tracking for adaptive learning"""

        # Record execution for this strategy
        strategy_name = result.strategy_used
        confidence = result.confidence_score

        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []

        self.strategy_performance[strategy_name].append(confidence)

        # Keep only recent performance (sliding window)
        max_history = 50
        if len(self.strategy_performance[strategy_name]) > max_history:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-max_history:]

        # Update execution history
        self.execution_history.append({
            'timestamp': datetime.now(),
            'strategy': strategy_name,
            'confidence': confidence,
            'horizons_count': result.total_forecasts_combined,
            'execution_time_ms': result.metadata.get('execution_time_ms', 0)
        })

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all strategies"""
        summary = {
            'total_ensembles_created': len(self.execution_history),
            'strategies_used': list(self.strategy_performance.keys()),
            'strategy_performance': {}
        }

        for strategy, scores in self.strategy_performance.items():
            if scores:
                summary['strategy_performance'][strategy] = {
                    'average_confidence': np.mean(scores),
                    'confidence_std': np.std(scores),
                    'executions': len(scores),
                    'recent_trend': np.mean(scores[-5:]) if len(scores) >= 5 else np.mean(scores)
                }

        return summary

    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        self.strategy_performance.clear()
        self.execution_history.clear()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()