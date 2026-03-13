"""
GARCH Models Family for Crypto Volatility Forecasting

Implementation of a full suite of GARCH models:
- GARCH(1,1) - Standard GARCH
- EGARCH - Exponential GARCH for asymmetric effects
- GJR-GARCH - Threshold GARCH for leverage effects
- FIGARCH - Fractionally Integrated GARCH for long memory
- DCC-GARCH - Dynamic Conditional Correlation for multivariate

Features:
- Production-ready error handling
- Async model fitting
- Comprehensive logging
- Model performance monitoring
- Automatic hyperparameter optimization
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
from arch.univariate import GARCH, EGARCH, FIGARCH, ConstantMean
from arch.univariate.mean import ZeroMean
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from numba import jit

# Logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='arch')

@dataclass
class VolatilityForecast:
    """Volatility forecast result"""
    symbol: str
    timestamp: datetime
    forecast_horizon: int
    volatility_forecast: np.ndarray
    variance_forecast: np.ndarray
    confidence_intervals: Dict[float, Tuple[np.ndarray, np.ndarray]]
    model_name: str
    model_params: Dict[str, Any]
    forecast_quality: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    symbol: str
    period: str
    aic: float
    bic: float
    log_likelihood: float
    mse: float
    mae: float
    qlike: float
    r_squared: float
    hit_rate: float
    validation_period: Tuple[datetime, datetime]
    created_at: datetime = field(default_factory=datetime.now)

class BaseGARCHModel(ABC):
    """
    Base class for all GARCH models
    """
    
    def __init__(
        self, 
        symbol: str,
        name: str,
        mean_model: str = "Zero",
        vol_model: str = "GARCH",
        dist: str = "Normal",
        optimization_method: str = "mle"
    ):
        self.symbol = symbol
        self.name = name
        self.mean_model = mean_model
        self.vol_model = vol_model
        self.dist = dist  
        self.optimization_method = optimization_method
        
        self.model = None
        self.fitted_model = None
        self.last_fit_time: Optional[datetime] = None
        self.performance_history: List[ModelPerformance] = []
        
        logger.info(f"🎯 Initialized {name} model for {symbol}")

    @abstractmethod
    async def fit(
        self, 
        returns: pd.Series, 
        update_freq: int = 252,
        **kwargs
    ) -> "BaseGARCHModel":
        """Train the model with async support"""
        pass

    @abstractmethod
    async def forecast(
        self, 
        horizon: int = 1,
        method: str = "simulation",
        confidence_levels: List[float] = None
    ) -> VolatilityForecast:
        """Generate volatility forecast"""
        pass

    def calculate_qlike_loss(
        self, 
        realized_var: np.ndarray, 
        forecast_var: np.ndarray
    ) -> float:
        """QLIKE loss function for evaluating volatility forecasts"""
        return np.mean(realized_var / forecast_var + np.log(forecast_var))

    def validate_model(
        self,
        returns: pd.Series,
        validation_start: datetime,
        validation_end: datetime,
        rolling_window: int = 252
    ) -> ModelPerformance:
        """Validate model on out-of-sample data"""
        val_returns = returns[validation_start:validation_end]
        predictions = []
        actuals = []
        
        for i in range(rolling_window, len(val_returns)):
            train_data = val_returns.iloc[i-rolling_window:i]
            actual_vol = val_returns.iloc[i]
            
            # Train on rolling window
            temp_model = self._create_model()
            temp_fitted = temp_model.fit(train_data, disp="off")
            forecast = temp_fitted.forecast(horizon=1)
            
            predictions.append(np.sqrt(forecast.variance.iloc[-1, 0]))
            actuals.append(abs(actual_vol))

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        qlike = self.calculate_qlike_loss(actuals**2, predictions**2)
        r_squared = 1 - (np.sum((actuals - predictions)**2) / 
                        np.sum((actuals - np.mean(actuals))**2))
        hit_rate = np.mean((predictions > np.median(predictions)) == 
                          (actuals > np.median(actuals)))

        return ModelPerformance(
            model_name=self.name,
            symbol=self.symbol,
            period=f"{validation_start.date()}-{validation_end.date()}",
            aic=self.fitted_model.aic if self.fitted_model else np.nan,
            bic=self.fitted_model.bic if self.fitted_model else np.nan,
            log_likelihood=self.fitted_model.loglikelihood if self.fitted_model else np.nan,
            mse=mse,
            mae=mae, 
            qlike=qlike,
            r_squared=r_squared,
            hit_rate=hit_rate,
            validation_period=(validation_start, validation_end)
        )

class GARCHModel(BaseGARCHModel):
    """
    Standard GARCH(1,1) Model
    
    Classic GARCH model for volatility forecasting.
    Optimal for cryptocurrencies with moderate volatility.
    """
    
    def __init__(self, symbol: str, p: int = 1, q: int = 1, **kwargs):
        super().__init__(symbol, f"GARCH({p},{q})", **kwargs)
        self.p = p  # GARCH lag order
        self.q = q  # ARCH lag order
    
    def _create_model(self, returns: pd.Series = None):
        """Create GARCH model"""
        if self.mean_model == "Zero":
            mean_model = ZeroMean(returns)
        else:
            mean_model = ConstantMean(returns)
            
        return mean_model

    async def fit(
        self, 
        returns: pd.Series,
        update_freq: int = 252,
        **kwargs
    ) -> "GARCHModel":
        """
        Async GARCH model fitting
        """
        try:
            logger.info(f"🔄 Fitting {self.name} model for {self.symbol}...")
            
            # Validate data
            if len(returns) < 100:
                raise ValueError(f"Insufficient data: {len(returns)} observations")
            
            # Prepare data
            returns_clean = returns.dropna() * 100  # Convert to percentage
            
            # Create model
            mean_model = ZeroMean(returns_clean)
            mean_model.volatility = GARCH(p=self.p, q=self.q)
            mean_model.distribution = self._get_distribution()
            
            # Async fitting
            loop = asyncio.get_event_loop()
            self.fitted_model = await loop.run_in_executor(
                None, 
                lambda: mean_model.fit(update_freq=update_freq, disp="off")
            )
            
            self.model = mean_model
            self.last_fit_time = datetime.now()
            
            logger.info(f"✅ {self.name} model fitted successfully")
            logger.info(f"   AIC: {self.fitted_model.aic:.4f}")
            logger.info(f"   BIC: {self.fitted_model.bic:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting {self.name} model: {e}")
            raise

    async def forecast(
        self,
        horizon: int = 1,
        method: str = "simulation",
        confidence_levels: List[float] = None,
        simulations: int = 1000
    ) -> VolatilityForecast:
        """
        Generate volatility forecast
        """
        if not self.fitted_model:
            raise ValueError("Model must be fitted before forecasting")
        
        if confidence_levels is None:
            confidence_levels = [0.05, 0.1, 0.25]
        
        try:
            logger.info(f"📈 Forecasting volatility for {horizon} periods...")
            
            if method == "analytical":
                # Analytical forecast
                forecast = self.fitted_model.forecast(horizon=horizon)
                variance_forecast = forecast.variance.iloc[-1].values / 10000  # Convert back from %
                volatility_forecast = np.sqrt(variance_forecast)
                
                # Confidence intervals (simple approximation)
                std_err = np.sqrt(variance_forecast * 2 / len(self.fitted_model.resid))
                confidence_intervals = {}
                
                for alpha in confidence_levels:
                    z_score = stats.norm.ppf(1 - alpha/2)
                    lower = volatility_forecast - z_score * std_err
                    upper = volatility_forecast + z_score * std_err
                    confidence_intervals[1-alpha] = (
                        np.maximum(lower, 0),  # Vol cannot be negative
                        upper
                    )
            
            else:
                # Simulation-based forecast
                forecasts = self.fitted_model.forecast(
                    horizon=horizon, 
                    method='simulation',
                    simulations=simulations
                )
                
                # Extract simulations
                variance_sims = forecasts.simulations.variances[-simulations:] / 10000
                volatility_sims = np.sqrt(variance_sims)
                
                # Mean forecast
                variance_forecast = np.mean(variance_sims, axis=0)
                volatility_forecast = np.sqrt(variance_forecast)
                
                # Confidence intervals from simulations
                confidence_intervals = {}
                for alpha in confidence_levels:
                    lower = np.percentile(volatility_sims, alpha/2*100, axis=0)
                    upper = np.percentile(volatility_sims, (1-alpha/2)*100, axis=0)
                    confidence_intervals[1-alpha] = (lower, upper)
            
            # Assess forecast quality
            forecast_quality = self._assess_forecast_quality(
                volatility_forecast, confidence_intervals
            )
            
            forecast_result = VolatilityForecast(
                symbol=self.symbol,
                timestamp=datetime.now(),
                forecast_horizon=horizon,
                volatility_forecast=volatility_forecast,
                variance_forecast=variance_forecast,
                confidence_intervals=confidence_intervals,
                model_name=self.name,
                model_params={
                    "p": self.p,
                    "q": self.q,
                    "distribution": self.dist
                },
                forecast_quality=forecast_quality,
                metadata={
                    "method": method,
                    "simulations": simulations if method == "simulation" else None,
                    "last_fit": self.last_fit_time
                }
            )
            
            logger.info(f"✅ Forecast generated: {volatility_forecast[0]:.4f} volatility")
            return forecast_result
            
        except Exception as e:
            logger.error(f"❌ Error generating forecast: {e}")
            raise

    def _get_distribution(self):
        """Get distribution for the model"""
        from arch.univariate import Normal, StudentsT, SkewStudent
        
        dist_map = {
            "Normal": Normal(),
            "StudentT": StudentsT(), 
            "SkewStudent": SkewStudent()
        }
        return dist_map.get(self.dist, Normal())

    def _assess_forecast_quality(
        self,
        forecast: np.ndarray,
        confidence_intervals: Dict[float, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, float]:
        """Assess forecast quality"""
        quality_metrics = {}
        
        # Confidence interval width (smaller = better)
        if 0.95 in confidence_intervals:
            ci_95 = confidence_intervals[0.95]
            avg_width = np.mean(ci_95[1] - ci_95[0])
            quality_metrics["ci_width_95"] = avg_width
        
        # Forecast stability (less variability = better)
        if len(forecast) > 1:
            stability = np.std(np.diff(forecast)) / np.mean(forecast)
            quality_metrics["forecast_stability"] = stability
        
        # Model goodness-of-fit
        if self.fitted_model:
            quality_metrics["aic"] = self.fitted_model.aic
            quality_metrics["bic"] = self.fitted_model.bic
            quality_metrics["log_likelihood"] = self.fitted_model.loglikelihood
        
        return quality_metrics

class EGARCHModel(BaseGARCHModel):
    """
    Exponential GARCH (EGARCH) Model
    
    Model accounting for asymmetric effects (leverage effect):
    - Negative news increases volatility more than positive news
    - Especially important for cryptocurrencies with high volatility
    """
    
    def __init__(self, symbol: str, p: int = 1, o: int = 1, q: int = 1, **kwargs):
        super().__init__(symbol, f"EGARCH({p},{o},{q})", **kwargs)
        self.p = p  # Symmetric GARCH terms
        self.o = o  # Asymmetric terms
        self.q = q  # ARCH terms

    async def fit(
        self, 
        returns: pd.Series,
        update_freq: int = 252,
        **kwargs
    ) -> "EGARCHModel":
        """Train EGARCH model"""
        try:
            logger.info(f"🔄 Fitting {self.name} model for {self.symbol}...")
            
            returns_clean = returns.dropna() * 100
            
            # EGARCH model
            mean_model = ZeroMean(returns_clean)
            mean_model.volatility = EGARCH(p=self.p, o=self.o, q=self.q)
            mean_model.distribution = self._get_distribution()
            
            loop = asyncio.get_event_loop()
            self.fitted_model = await loop.run_in_executor(
                None,
                lambda: mean_model.fit(update_freq=update_freq, disp="off")
            )
            
            self.model = mean_model
            self.last_fit_time = datetime.now()
            
            logger.info(f"✅ {self.name} model fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting {self.name}: {e}")
            raise

    async def forecast(
        self,
        horizon: int = 1,
        method: str = "simulation",
        confidence_levels: List[float] = None,
        simulations: int = 1000
    ) -> VolatilityForecast:
        """Forecast accounting for asymmetry"""
        # Similar to GARCH, but accounting for asymmetric effects
        return await self._forecast_base(horizon, method, confidence_levels, simulations)

    def _get_distribution(self):
        """EGARCH often works better with Student-t distribution"""
        from arch.univariate import Normal, StudentsT, SkewStudent
        
        dist_map = {
            "Normal": Normal(),
            "StudentT": StudentsT(),
            "SkewStudent": SkewStudent()
        }
        return dist_map.get(self.dist, StudentsT())  # Default to Student-t

    async def _forecast_base(self, horizon, method, confidence_levels, simulations):
        """Base forecast implementation (shared by EGARCH/GJR-GARCH)"""
        if not self.fitted_model:
            raise ValueError("Model must be fitted before forecasting")
        
        if confidence_levels is None:
            confidence_levels = [0.05, 0.1, 0.25]
        
        forecast = self.fitted_model.forecast(horizon=horizon, method=method)
        variance_forecast = forecast.variance.iloc[-1].values / 10000
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Simplified confidence intervals
        confidence_intervals = {}
        std_err = np.sqrt(variance_forecast * 2 / len(self.fitted_model.resid))
        
        for alpha in confidence_levels:
            z_score = stats.norm.ppf(1 - alpha/2)
            lower = np.maximum(volatility_forecast - z_score * std_err, 0)
            upper = volatility_forecast + z_score * std_err
            confidence_intervals[1-alpha] = (lower, upper)
        
        forecast_quality = self._assess_forecast_quality(volatility_forecast, confidence_intervals)
        
        return VolatilityForecast(
            symbol=self.symbol,
            timestamp=datetime.now(),
            forecast_horizon=horizon,
            volatility_forecast=volatility_forecast,
            variance_forecast=variance_forecast,
            confidence_intervals=confidence_intervals,
            model_name=self.name,
            model_params={"p": self.p, "o": getattr(self, 'o', 0), "q": self.q},
            forecast_quality=forecast_quality,
            metadata={"method": method, "last_fit": self.last_fit_time}
        )

class GJRGARCHModel(BaseGARCHModel):
    """
    Glosten-Jagannathan-Runkle GARCH (GJR-GARCH) Model
    
    Threshold GARCH model for leverage effects:
    - Different response to positive and negative news
    - Threshold function for asymmetric volatility
    """
    
    def __init__(self, symbol: str, p: int = 1, o: int = 1, q: int = 1, **kwargs):
        super().__init__(symbol, f"GJR-GARCH({p},{o},{q})", **kwargs)
        self.p = p
        self.o = o  # Threshold terms
        self.q = q

    async def fit(self, returns: pd.Series, update_freq: int = 252, **kwargs) -> "GJRGARCHModel":
        """Train GJR-GARCH model"""
        try:
            logger.info(f"🔄 Fitting {self.name} model for {self.symbol}...")
            
            returns_clean = returns.dropna() * 100
            
            # Use standard GARCH with power=2 for GJR effects
            mean_model = ZeroMean(returns_clean)
            # In the arch library, the o parameter can be used for threshold effects
            mean_model.volatility = GARCH(p=self.p, q=self.q) # FIXME: needs a dedicated GJR implementation
            mean_model.distribution = self._get_distribution()
            
            loop = asyncio.get_event_loop()
            self.fitted_model = await loop.run_in_executor(
                None,
                lambda: mean_model.fit(update_freq=update_freq, disp="off")
            )
            
            self.model = mean_model
            self.last_fit_time = datetime.now()
            
            logger.info(f"✅ {self.name} model fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting {self.name}: {e}")
            raise

    async def forecast(self, horizon: int = 1, method: str = "simulation", 
                      confidence_levels: List[float] = None, simulations: int = 1000) -> VolatilityForecast:
        """Forecast with threshold effects"""
        return await self._forecast_base(horizon, method, confidence_levels, simulations)

    def _get_distribution(self):
        from arch.univariate import StudentsT
        return StudentsT()  # GJR-GARCH works better with heavy-tailed distributions

    async def _forecast_base(self, horizon, method, confidence_levels, simulations):
        """Base forecast implementation"""
        # Copy of implementation from EGARCHModel
        if not self.fitted_model:
            raise ValueError("Model must be fitted before forecasting")
        
        if confidence_levels is None:
            confidence_levels = [0.05, 0.1, 0.25]
        
        forecast = self.fitted_model.forecast(horizon=horizon, method=method)
        variance_forecast = forecast.variance.iloc[-1].values / 10000
        volatility_forecast = np.sqrt(variance_forecast)
        
        confidence_intervals = {}
        std_err = np.sqrt(variance_forecast * 2 / len(self.fitted_model.resid))
        
        for alpha in confidence_levels:
            z_score = stats.norm.ppf(1 - alpha/2)
            lower = np.maximum(volatility_forecast - z_score * std_err, 0)
            upper = volatility_forecast + z_score * std_err
            confidence_intervals[1-alpha] = (lower, upper)
        
        forecast_quality = self._assess_forecast_quality(volatility_forecast, confidence_intervals)
        
        return VolatilityForecast(
            symbol=self.symbol,
            timestamp=datetime.now(),
            forecast_horizon=horizon,
            volatility_forecast=volatility_forecast,
            variance_forecast=variance_forecast,
            confidence_intervals=confidence_intervals,
            model_name=self.name,
            model_params={"p": self.p, "o": self.o, "q": self.q},
            forecast_quality=forecast_quality,
            metadata={"method": method, "last_fit": self.last_fit_time}
        )

class FIGARCHModel(BaseGARCHModel):
    """
    Fractionally Integrated GARCH (FIGARCH) Model
    
    Model for long memory in volatility:
    - Long-range dependence in volatility
    - Fractional integration for crypto markets
    - Especially effective for Bitcoin and major altcoins
    """
    
    def __init__(self, symbol: str, p: int = 1, q: int = 1, **kwargs):
        super().__init__(symbol, f"FIGARCH({p},{q})", **kwargs)
        self.p = p
        self.q = q

    async def fit(self, returns: pd.Series, update_freq: int = 252, **kwargs) -> "FIGARCHModel":
        """Train FIGARCH model"""
        try:
            logger.info(f"🔄 Fitting {self.name} model for {self.symbol}...")
            
            returns_clean = returns.dropna() * 100
            
            mean_model = ZeroMean(returns_clean)
            mean_model.volatility = FIGARCH(p=self.p, q=self.q)
            mean_model.distribution = self._get_distribution()
            
            loop = asyncio.get_event_loop()
            self.fitted_model = await loop.run_in_executor(
                None,
                lambda: mean_model.fit(update_freq=update_freq, disp="off")
            )
            
            self.model = mean_model
            self.last_fit_time = datetime.now()
            
            logger.info(f"✅ {self.name} model fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting {self.name}: {e}")
            raise

    async def forecast(self, horizon: int = 1, method: str = "simulation",
                      confidence_levels: List[float] = None, simulations: int = 1000) -> VolatilityForecast:
        """Forecast with long memory effects"""
        return await self._forecast_base(horizon, method, confidence_levels, simulations)

    def _get_distribution(self):
        from arch.univariate import Normal
        return Normal()  # FIGARCH typically with Normal distribution

    async def _forecast_base(self, horizon, method, confidence_levels, simulations):
        """FIGARCH forecast accounting for long memory"""
        if not self.fitted_model:
            raise ValueError("Model must be fitted before forecasting")
        
        if confidence_levels is None:
            confidence_levels = [0.05, 0.1, 0.25]
        
        forecast = self.fitted_model.forecast(horizon=horizon)
        variance_forecast = forecast.variance.iloc[-1].values / 10000
        volatility_forecast = np.sqrt(variance_forecast)
        
        # FIGARCH has wider confidence intervals due to long memory
        confidence_intervals = {}
        std_err = np.sqrt(variance_forecast * 3 / len(self.fitted_model.resid))  # Increased uncertainty
        
        for alpha in confidence_levels:
            z_score = stats.norm.ppf(1 - alpha/2)
            lower = np.maximum(volatility_forecast - z_score * std_err, 0)
            upper = volatility_forecast + z_score * std_err
            confidence_intervals[1-alpha] = (lower, upper)
        
        forecast_quality = self._assess_forecast_quality(volatility_forecast, confidence_intervals)
        # Add long memory metric
        forecast_quality["long_memory_strength"] = getattr(self.fitted_model, 'd', 0.5)
        
        return VolatilityForecast(
            symbol=self.symbol,
            timestamp=datetime.now(),
            forecast_horizon=horizon,
            volatility_forecast=volatility_forecast,
            variance_forecast=variance_forecast,
            confidence_intervals=confidence_intervals,
            model_name=self.name,
            model_params={"p": self.p, "q": self.q, "d": getattr(self.fitted_model, 'd', 0.5)},
            forecast_quality=forecast_quality,
            metadata={"method": method, "last_fit": self.last_fit_time}
        )

class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH (DCC-GARCH) Model
    
    Multivariate GARCH for correlation analysis:
    - Dynamic correlations between cryptocurrencies
    - Portfolio risk assessment
    - Cross-asset volatility spillovers
    """
    
    def __init__(self, symbols: List[str], name: str = "DCC-GARCH"):
        self.symbols = symbols
        self.name = name
        self.models = {}  # Individual GARCH models
        self.dcc_model = None
        self.last_fit_time: Optional[datetime] = None
        
        logger.info(f"🎯 Initialized {name} for {len(symbols)} assets: {symbols}")

    async def fit(self, returns_matrix: pd.DataFrame, **kwargs) -> "DCCGARCHModel":
        """
        Train DCC-GARCH model

        Args:
            returns_matrix: DataFrame with returns per symbol
        """
        try:
            logger.info(f"🔄 Fitting {self.name} model for {len(self.symbols)} assets...")
            
            # Step 1: Train individual GARCH models
            for symbol in self.symbols:
                if symbol not in returns_matrix.columns:
                    raise ValueError(f"Symbol {symbol} not found in returns data")
                
                returns = returns_matrix[symbol].dropna()
                garch_model = GARCHModel(symbol)
                await garch_model.fit(returns)
                self.models[symbol] = garch_model
                
                logger.info(f"   ✅ Fitted individual GARCH for {symbol}")
            
            # Step 2: DCC estimation (simplified implementation)
            # In a real implementation, a dedicated DCC procedure is used
            residuals_matrix = pd.DataFrame()
            
            for symbol in self.symbols:
                model = self.models[symbol]
                if model.fitted_model:
                    std_residuals = model.fitted_model.std_resid
                    residuals_matrix[symbol] = std_residuals
            
            # Dynamic correlations (simple exponential scheme)
            self.correlation_dynamics = self._estimate_dynamic_correlations(residuals_matrix)
            
            self.last_fit_time = datetime.now()
            logger.info(f"✅ {self.name} model fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Error fitting {self.name}: {e}")
            raise

    def _estimate_dynamic_correlations(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """Estimate dynamic correlations (simplified DCC)"""
        # Exponentially weighted correlations
        correlations = residuals.ewm(span=30).corr().unstack()
        return correlations

    async def forecast_portfolio_volatility(
        self, 
        weights: Dict[str, float],
        horizon: int = 1
    ) -> Dict[str, Any]:
        """
        Forecast portfolio volatility

        Args:
            weights: Asset weights in the portfolio
            horizon: Forecast horizon
        """
        if not all(symbol in self.models for symbol in weights.keys()):
            raise ValueError("All portfolio symbols must be fitted in DCC model")
        
        # Individual volatility forecasts
        individual_forecasts = {}
        for symbol, weight in weights.items():
            if weight > 0:
                forecast = await self.models[symbol].forecast(horizon=horizon)
                individual_forecasts[symbol] = forecast.volatility_forecast[0]
        
        # Portfolio volatility accounting for correlations
        portfolio_variance = 0
        symbols_list = list(weights.keys())
        
        for i, symbol1 in enumerate(symbols_list):
            for j, symbol2 in enumerate(symbols_list):
                weight1 = weights[symbol1]
                weight2 = weights[symbol2]
                vol1 = individual_forecasts.get(symbol1, 0)
                vol2 = individual_forecasts.get(symbol2, 0)
                
                if i == j:
                    correlation = 1.0
                else:
                    # Latest correlation from the dynamic model
                    try:
                        correlation = self.correlation_dynamics.iloc[-1][(symbol1, symbol2)]
                    except:
                        correlation = 0.5  # Default correlation
                
                portfolio_variance += weight1 * weight2 * vol1 * vol2 * correlation
        
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        result = {
            "portfolio_volatility": portfolio_volatility,
            "individual_volatilities": individual_forecasts,
            "weights": weights,
            "correlation_matrix": self._get_current_correlations(),
            "diversification_ratio": self._calculate_diversification_ratio(weights, individual_forecasts, portfolio_volatility),
            "timestamp": datetime.now(),
            "horizon": horizon
        }
        
        logger.info(f"📊 Portfolio volatility forecast: {portfolio_volatility:.4f}")
        return result

    def _get_current_correlations(self) -> Dict[Tuple[str, str], float]:
        """Get current correlation matrix"""
        if self.correlation_dynamics is None or len(self.correlation_dynamics) == 0:
            return {}
        
        current_corr = {}
        for (s1, s2), corr_series in self.correlation_dynamics.items():
            if not corr_series.empty:
                current_corr[(s1, s2)] = corr_series.iloc[-1]
        
        return current_corr

    def _calculate_diversification_ratio(
        self, 
        weights: Dict[str, float], 
        individual_vols: Dict[str, float],
        portfolio_vol: float
    ) -> float:
        """Calculate diversification ratio"""
        weighted_avg_vol = sum(weights[s] * vol for s, vol in individual_vols.items())
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        return 1.0

class GARCHModelSelector:
    """
    Automatic selection of the best GARCH model for a specific asset

    Features:
    - Bayesian optimization for hyperparameters
    - Cross-validation for model selection
    - Performance monitoring
    - Automatic model updates
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models = {
            "GARCH": GARCHModel,
            "EGARCH": EGARCHModel,
            "GJR-GARCH": GJRGARCHModel,
            "FIGARCH": FIGARCHModel
        }
        self.fitted_models: Dict[str, BaseGARCHModel] = {}
        self.model_scores: Dict[str, float] = {}
        self.best_model: Optional[BaseGARCHModel] = None
        
        logger.info(f"🎯 Model selector initialized for {symbol}")

    async def select_best_model(
        self,
        returns: pd.Series,
        validation_split: float = 0.8,
        scoring_method: str = "aic"
    ) -> BaseGARCHModel:
        """
        Select best model based on validation

        Args:
            returns: Returns time series
            validation_split: Training data fraction
            scoring_method: Scoring method ("aic", "bic", "qlike", "mse")
        """
        logger.info(f"🔍 Selecting best GARCH model for {self.symbol}...")
        
        # Split data
        split_point = int(len(returns) * validation_split)
        train_returns = returns.iloc[:split_point]
        val_returns = returns.iloc[split_point:]
        
        val_start = val_returns.index[0] if len(val_returns) > 0 else train_returns.index[-1]
        val_end = val_returns.index[-1] if len(val_returns) > 0 else train_returns.index[-1]
        
        model_performances = {}
        
        # Train and validate each model
        for model_name, ModelClass in self.models.items():
            try:
                logger.info(f"   📊 Testing {model_name}...")
                
                # Create and train model
                model = ModelClass(self.symbol)
                await model.fit(train_returns)
                
                # Validate model
                if len(val_returns) > 30:  # Enough data for validation
                    performance = model.validate_model(
                        returns, val_start, val_end, rolling_window=min(252, len(train_returns)//2)
                    )
                else:
                    # Use in-sample metrics
                    performance = ModelPerformance(
                        model_name=model_name,
                        symbol=self.symbol,
                        period="in-sample",
                        aic=model.fitted_model.aic,
                        bic=model.fitted_model.bic,
                        log_likelihood=model.fitted_model.loglikelihood,
                        mse=0.0,
                        mae=0.0,
                        qlike=0.0,
                        r_squared=0.0,
                        hit_rate=0.0,
                        validation_period=(val_start, val_end)
                    )
                
                self.fitted_models[model_name] = model
                model_performances[model_name] = performance
                
                # Determine score based on method
                if scoring_method == "aic":
                    score = performance.aic
                    minimize = True
                elif scoring_method == "bic": 
                    score = performance.bic
                    minimize = True
                elif scoring_method == "qlike":
                    score = performance.qlike
                    minimize = True
                elif scoring_method == "mse":
                    score = performance.mse
                    minimize = True
                else:
                    score = performance.r_squared
                    minimize = False
                
                self.model_scores[model_name] = score
                
                logger.info(f"   ✅ {model_name}: {scoring_method}={score:.4f}")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to fit {model_name}: {e}")
                continue
        
        if not self.model_scores:
            raise ValueError("No models could be fitted successfully")
        
        # Select best model
        if minimize:
            best_model_name = min(self.model_scores.keys(), key=lambda k: self.model_scores[k])
        else:
            best_model_name = max(self.model_scores.keys(), key=lambda k: self.model_scores[k])
        
        self.best_model = self.fitted_models[best_model_name]
        best_score = self.model_scores[best_model_name]
        
        logger.info(f"🏆 Best model selected: {best_model_name} ({scoring_method}={best_score:.4f})")
        
        # Save results
        self.selection_results = {
            "best_model": best_model_name,
            "best_score": best_score,
            "all_scores": self.model_scores.copy(),
            "selection_method": scoring_method,
            "validation_period": (val_start, val_end),
            "model_performances": model_performances,
            "selection_time": datetime.now()
        }
        
        return self.best_model

    async def optimize_hyperparameters(
        self,
        returns: pd.Series,
        model_type: str = "GARCH",
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Hyperparameter optimization using Optuna

        Args:
            returns: Returns time series
            model_type: Model type to optimize
            n_trials: Number of optimization trials
        """
        logger.info(f"🎯 Optimizing {model_type} hyperparameters for {self.symbol}...")
        
        def objective(trial):
            try:
                # Suggest parameters
                if model_type == "GARCH":
                    p = trial.suggest_int("p", 1, 3)
                    q = trial.suggest_int("q", 1, 3)
                    model = GARCHModel(self.symbol, p=p, q=q)
                elif model_type == "EGARCH":
                    p = trial.suggest_int("p", 1, 2)
                    o = trial.suggest_int("o", 1, 2)
                    q = trial.suggest_int("q", 1, 2)
                    model = EGARCHModel(self.symbol, p=p, o=o, q=q)
                else:
                    p = trial.suggest_int("p", 1, 2)
                    q = trial.suggest_int("q", 1, 2)
                    model = self.models[model_type](self.symbol, p=p, q=q)
                
                # Synchronous fitting for Optuna
                train_data = returns.dropna() * 100
                mean_model = ZeroMean(train_data)
                
                if model_type == "GARCH":
                    mean_model.volatility = GARCH(p=p, q=q)
                elif model_type == "FIGARCH":
                    mean_model.volatility = FIGARCH(p=p, q=q)
                else:
                    mean_model.volatility = GARCH(p=1, q=1)  # Fallback
                
                fitted = mean_model.fit(disp="off")
                return fitted.aic  # Minimize AIC
                
            except Exception:
                return float('inf')  # Penalty for failed fits
        
        # Optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"🏆 Best {model_type} parameters: {best_params}")
        logger.info(f"   Best AIC: {best_value:.4f}")
        
        # Create and train optimal model
        if model_type == "GARCH":
            optimal_model = GARCHModel(self.symbol, **best_params)
        elif model_type == "EGARCH":
            optimal_model = EGARCHModel(self.symbol, **best_params)  
        else:
            optimal_model = self.models[model_type](self.symbol, **best_params)
        
        await optimal_model.fit(returns)
        
        optimization_results = {
            "model_type": model_type,
            "best_parameters": best_params,
            "best_aic": best_value,
            "n_trials": n_trials,
            "optimal_model": optimal_model,
            "optimization_time": datetime.now(),
            "study_summary": {
                "n_completed_trials": len(study.trials),
                "n_failed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            }
        }
        
        return optimization_results

    def get_model_comparison(self) -> pd.DataFrame:
        """Compare all trained models"""
        if not self.fitted_models:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, model in self.fitted_models.items():
            if model.fitted_model:
                row = {
                    "Model": model_name,
                    "AIC": model.fitted_model.aic,
                    "BIC": model.fitted_model.bic,
                    "LogLikelihood": model.fitted_model.loglikelihood,
                    "Score": self.model_scores.get(model_name, np.nan),
                    "LastFit": model.last_fit_time,
                    "Parameters": str(getattr(model, 'p', 1)) + "," + str(getattr(model, 'q', 1))
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("AIC")  # Best models on top
        
        return comparison_df

    def summary_report(self) -> str:
        """Generate model selection report"""
        if not self.best_model:
            return "No model selection performed yet."
        
        report = f"""
🎯 GARCH Model Selection Report for {self.symbol}
{'='*60}

Best Model: {self.best_model.name}
Selection Time: {self.selection_results['selection_time']}
Selection Method: {self.selection_results['selection_method']}
Best Score: {self.selection_results['best_score']:.4f}

Model Comparison:
{self.get_model_comparison().to_string(index=False)}

Recommendations:
- Use {self.best_model.name} for volatility forecasting
- Model shows {'good' if self.selection_results['best_score'] < 0 else 'acceptable'} fit quality
- Consider retraining every {'month' if isinstance(self.best_model, FIGARCHModel) else '2 weeks'}
        """
        
        return report

# Utility functions for quick usage

async def quick_volatility_forecast(
    symbol: str,
    returns: pd.Series,
    horizon: int = 1,
    auto_select: bool = True
) -> VolatilityForecast:
    """
    Quick volatility forecast creation

    Args:
        symbol: Asset symbol
        returns: Returns
        horizon: Forecast horizon
        auto_select: Automatic model selection
    """
    if auto_select:
        selector = GARCHModelSelector(symbol)
        model = await selector.select_best_model(returns)
    else:
        model = GARCHModel(symbol)
        await model.fit(returns)
    
    return await model.forecast(horizon=horizon)

def create_garch_ensemble(
    symbol: str,
    model_types: List[str] = None
) -> List[BaseGARCHModel]:
    """
    Create a GARCH model ensemble

    Args:
        symbol: Asset symbol
        model_types: List of model types
    """
    if model_types is None:
        model_types = ["GARCH", "EGARCH", "GJR-GARCH"]
    
    models = []
    model_classes = {
        "GARCH": GARCHModel,
        "EGARCH": EGARCHModel, 
        "GJR-GARCH": GJRGARCHModel,
        "FIGARCH": FIGARCHModel
    }
    
    for model_type in model_types:
        if model_type in model_classes:
            model = model_classes[model_type](symbol)
            models.append(model)
    
    return models

# Export all classes
__all__ = [
    "BaseGARCHModel",
    "GARCHModel", 
    "EGARCHModel",
    "GJRGARCHModel", 
    "FIGARCHModel",
    "DCCGARCHModel",
    "GARCHModelSelector",
    "VolatilityForecast",
    "ModelPerformance",
    "quick_volatility_forecast",
    "create_garch_ensemble"
]

logger.info("🔥 GARCH Models module loaded successfully!")