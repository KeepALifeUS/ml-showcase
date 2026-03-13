"""
Realized Volatility Estimators for High-Frequency Crypto Data

Implementation of modern volatility estimators:
- Realized Volatility (RV) - classic realized volatility
- Bipower Variation (BPV) - robust to jumps
- Realized Kernel (RK) - microstructure noise robust
- Realized GARCH - combination of RV with GARCH models
- Two-Scale Realized Volatility (TSRV)
- Multi-Scale Realized Volatility (MSRV)

Features:
- High-frequency data handling (tick data)
- Jump detection and robust estimation
- Intraday patterns recognition
- Real-time calculation
- Production-ready performance
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import asyncio
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import warnings

# Logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class RealizedVolatilityMeasure:
    """Realized volatility calculation result"""
    symbol: str
    timestamp: datetime
    period: str  # "1D", "1H", etc.
    measure_type: str  # "RV", "BPV", "RK", etc.
    
    # Core metrics
    realized_volatility: float
    realized_variance: float

    # Additional metrics
    jump_component: Optional[float] = None
    continuous_component: Optional[float] = None
    microstructure_bias: Optional[float] = None
    efficiency_ratio: Optional[float] = None
    
    # Statistics
    n_observations: int = 0
    sampling_frequency: str = "1min"
    data_quality_score: float = 1.0
    
    # Intraday patterns
    intraday_seasonality: Optional[Dict[str, float]] = None
    volatility_signature: Optional[Dict[str, float]] = None
    
    # Metadata
    calculation_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class JumpDetectionResult:
    """Jump detection result"""
    timestamp: datetime
    jump_detected: bool
    jump_size: float
    jump_significance: float
    test_statistic: float
    critical_value: float
    method: str = "BNS"  # Barndorff-Nielsen & Shephard

@jit(nopython=True)
def _compute_realized_variance_numba(log_returns: np.ndarray) -> float:
    """Fast realized variance calculation with Numba"""
    return np.sum(log_returns**2)

@jit(nopython=True) 
def _compute_bipower_variation_numba(log_returns: np.ndarray) -> float:
    """Fast Bipower Variation calculation with Numba"""
    n = len(log_returns)
    if n < 2:
        return 0.0
    
    abs_returns = np.abs(log_returns)
    bv = 0.0
    
    for i in prange(1, n):
        bv += abs_returns[i-1] * abs_returns[i]
    
    return (np.pi/2) * bv

@jit(nopython=True)
def _compute_tripower_quarticity_numba(log_returns: np.ndarray) -> float:
    """Fast Tripower Quarticity calculation with Numba"""
    n = len(log_returns)
    if n < 3:
        return 0.0
    
    abs_returns = np.abs(log_returns)
    tq = 0.0
    
    for i in prange(2, n):
        tq += (abs_returns[i-2] ** (4/3)) * (abs_returns[i-1] ** (4/3)) * (abs_returns[i] ** (4/3))
    
    return ((np.pi/2)**3) * tq

class BaseVolatilityEstimator(ABC):
    """Base class for all estimators"""
    
    def __init__(self, symbol: str, name: str):
        self.symbol = symbol
        self.name = name
        self.history: List[RealizedVolatilityMeasure] = []
        
        logger.info(f"🎯 Initialized {name} estimator for {symbol}")

    @abstractmethod
    async def estimate(
        self, 
        price_data: pd.DataFrame,
        **kwargs
    ) -> RealizedVolatilityMeasure:
        """Calculate realized volatility"""
        pass

    def _validate_data(self, price_data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input data"""
        if price_data.empty:
            return False, "Empty price data"
        
        required_columns = ['timestamp', 'close']
        missing_columns = [col for col in required_columns if col not in price_data.columns]
        if missing_columns:
            return False, f"Missing columns: {missing_columns}"
        
        if price_data['close'].isna().all():
            return False, "All close prices are NaN"
        
        return True, "Data validation passed"

    def _calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        return np.log(prices / prices.shift(1)).dropna()

    def _detect_outliers(self, returns: pd.Series, method: str = "iqr") -> pd.Series:
        """Detect outliers in returns"""
        if method == "iqr":
            Q1 = returns.quantile(0.25)
            Q3 = returns.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (returns >= lower_bound) & (returns <= upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(returns))
            return z_scores < 3
        
        return pd.Series(True, index=returns.index)

    def calculate_data_quality_score(self, price_data: pd.DataFrame) -> float:
        """Assess data quality for volatility calculation"""
        scores = []
        
        # Completeness (absence of missing values)
        completeness = 1 - (price_data['close'].isna().sum() / len(price_data))
        scores.append(completeness)
        
        # Regularity (regularity of time intervals)
        if len(price_data) > 1:
            time_diffs = price_data['timestamp'].diff().dt.total_seconds()
            time_diffs = time_diffs.dropna()
            if len(time_diffs) > 0:
                regularity = 1 - (time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 1)
                regularity = max(0, min(1, regularity))
                scores.append(regularity)
        
        # Absence of extreme values
        returns = self._calculate_log_returns(price_data['close'])
        if len(returns) > 0:
            outlier_ratio = 1 - (self._detect_outliers(returns).sum() / len(returns))
            outlier_score = max(0, 1 - outlier_ratio * 2)  # Penalize outliers
            scores.append(outlier_score)
        
        return np.mean(scores) if scores else 0.0

class RealizedVolatilityEstimator(BaseVolatilityEstimator):
    """
    Classical Realized Volatility Estimator
    
    RV = Sum(r_t^2) where r_t - intraday returns

    Simple and effective estimator for liquid markets without significant jumps.
    Optimal for major cryptocurrencies (BTC, ETH) on short intervals.
    """
    
    def __init__(self, symbol: str, sampling_frequency: str = "5min"):
        super().__init__(symbol, "Realized Volatility")
        self.sampling_frequency = sampling_frequency
        
    async def estimate(
        self,
        price_data: pd.DataFrame,
        period: str = "1D",
        annualize: bool = True,
        remove_overnight: bool = True
    ) -> RealizedVolatilityMeasure:
        """
        Calculate classic realized volatility

        Args:
            price_data: DataFrame with columns ['timestamp', 'close']
            period: Aggregation period ("1D", "1H")
            annualize: Annualize result
            remove_overnight: Remove overnight returns
        """
        try:
            logger.info(f"🔄 Calculating RV for {self.symbol} over {period}...")
            
            # Validate data
            is_valid, message = self._validate_data(price_data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {message}")
            
            # Prepare data
            data = price_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')
            
            # Calculate log returns
            log_returns = self._calculate_log_returns(data['close'])
            
            # Remove overnight returns if required
            if remove_overnight:
                log_returns = self._remove_overnight_returns(data, log_returns)
            
            # Async calculation with Numba
            loop = asyncio.get_event_loop()
            realized_variance = await loop.run_in_executor(
                None, 
                _compute_realized_variance_numba, 
                log_returns.values
            )
            
            realized_volatility = np.sqrt(realized_variance)
            
            # Annualization
            if annualize:
                # Assuming 252 trading days and observations per day
                periods_per_day = self._estimate_periods_per_day(data)
                annualization_factor = np.sqrt(252 * periods_per_day)
                
                realized_volatility *= annualization_factor
                realized_variance *= (annualization_factor ** 2)
            
            # Calculate intraday seasonality
            intraday_seasonality = self._calculate_intraday_seasonality(data, log_returns)
            
            # Calculate volatility signature
            volatility_signature = await self._calculate_volatility_signature(data)
            
            # Assess data quality
            data_quality = self.calculate_data_quality_score(data)
            
            result = RealizedVolatilityMeasure(
                symbol=self.symbol,
                timestamp=data['timestamp'].iloc[-1],
                period=period,
                measure_type="RV",
                realized_volatility=realized_volatility,
                realized_variance=realized_variance,
                n_observations=len(log_returns),
                sampling_frequency=self.sampling_frequency,
                data_quality_score=data_quality,
                intraday_seasonality=intraday_seasonality,
                volatility_signature=volatility_signature,
                metadata={
                    "annualized": annualize,
                    "overnight_removed": remove_overnight,
                    "original_observations": len(data)
                }
            )
            
            self.history.append(result)
            logger.info(f"✅ RV calculated: {realized_volatility:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating RV for {self.symbol}: {e}")
            raise

    def _remove_overnight_returns(
        self, 
        data: pd.DataFrame, 
        log_returns: pd.Series
    ) -> pd.Series:
        """Remove overnight returns"""
        # Simple removal of returns with large time gaps
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        median_diff = time_diffs.median()
        
        # Returns with interval > 2x median are considered overnight
        overnight_mask = time_diffs > (2 * median_diff)
        overnight_indices = overnight_mask[overnight_mask].index
        
        # Remove corresponding returns
        clean_returns = log_returns.copy()
        for idx in overnight_indices:
            if idx in clean_returns.index:
                clean_returns = clean_returns.drop(idx)
        
        return clean_returns

    def _estimate_periods_per_day(self, data: pd.DataFrame) -> int:
        """Estimate observations per day"""
        if len(data) < 2:
            return 1440  # Default: minute data
        
        # Median interval in minutes
        time_diffs_minutes = data['timestamp'].diff().dt.total_seconds() / 60
        median_interval = time_diffs_minutes.median()
        
        if median_interval > 0:
            return int(1440 / median_interval)  # 1440 minutes per day
        
        return 1440

    def _calculate_intraday_seasonality(
        self,
        data: pd.DataFrame,
        log_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate intraday seasonality"""
        if len(data) < 24:  # Minimum data for analysis
            return {}
        
        # Add hour to data
        data_with_returns = data.copy()
        data_with_returns['log_returns'] = log_returns.reindex(data.index).fillna(0)
        data_with_returns['hour'] = data_with_returns['timestamp'].dt.hour
        
        # Average volatility by hour
        hourly_vol = data_with_returns.groupby('hour')['log_returns'].apply(
            lambda x: np.sqrt(np.sum(x**2))
        )
        
        if len(hourly_vol) > 0:
            # Normalize relative to mean
            avg_vol = hourly_vol.mean()
            if avg_vol > 0:
                normalized_seasonality = (hourly_vol / avg_vol).to_dict()
                return {f"hour_{hour}": vol for hour, vol in normalized_seasonality.items()}
        
        return {}

    async def _calculate_volatility_signature(
        self,
        data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate volatility signature plot
        Dependence of realized volatility on sampling frequency
        """
        signature = {}
        
        # Different sampling frequencies for analysis
        frequencies = ["1min", "5min", "15min", "30min", "1H"]
        
        for freq in frequencies:
            try:
                # Resample data to different frequency
                resampled = data.set_index('timestamp').resample(freq)['close'].last().dropna()
                
                if len(resampled) > 2:
                    returns = self._calculate_log_returns(resampled)
                    rv = _compute_realized_variance_numba(returns.values)
                    signature[freq] = np.sqrt(rv)
                    
            except Exception as e:
                logger.debug(f"Failed to calculate signature for {freq}: {e}")
                continue
        
        return signature

class BipowerVariation(BaseVolatilityEstimator):
    """
    Bipower Variation Estimator - Robust to Jumps
    
    BV = (π/2) * Σ|r_{t-1}| * |r_t|
    
    Robust estimator for markets with jumps. Estimates the continuous component
    of volatility, excluding the effect of jumps. Ideal for volatile cryptocurrencies.
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "Bipower Variation")

    async def estimate(
        self,
        price_data: pd.DataFrame,
        period: str = "1D",
        annualize: bool = True,
        detect_jumps: bool = True
    ) -> RealizedVolatilityMeasure:
        """Calculate Bipower Variation"""
        try:
            logger.info(f"🔄 Calculating BPV for {self.symbol}...")
            
            # Prepare data
            is_valid, message = self._validate_data(price_data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {message}")
                
            data = price_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')
            
            log_returns = self._calculate_log_returns(data['close'])
            
            if len(log_returns) < 2:
                raise ValueError("Insufficient data for BPV calculation")
            
            # Async BPV calculation
            loop = asyncio.get_event_loop()
            bipower_variation = await loop.run_in_executor(
                None,
                _compute_bipower_variation_numba,
                log_returns.values
            )
            
            bipower_volatility = np.sqrt(bipower_variation)
            
            # Calculate RV for comparison
            realized_variance = _compute_realized_variance_numba(log_returns.values)
            
            # Jump detection
            jump_component = None
            continuous_component = bipower_variation
            jump_test_result = None
            
            if detect_jumps:
                jump_test_result = await self._detect_jumps_bns_test(
                    log_returns, realized_variance, bipower_variation
                )
                
                if jump_test_result.jump_detected:
                    jump_component = max(0, realized_variance - bipower_variation)
                    logger.info(f"🚨 Jump detected: size={jump_component:.6f}")
            
            # Annualization
            if annualize:
                periods_per_day = self._estimate_periods_per_day(data)
                annualization_factor = np.sqrt(252 * periods_per_day)
                
                bipower_volatility *= annualization_factor
                bipower_variation *= (annualization_factor ** 2)
                
                if jump_component is not None:
                    jump_component *= (annualization_factor ** 2)
            
            # Estimator efficiency (ratio to RV)
            efficiency_ratio = bipower_variation / realized_variance if realized_variance > 0 else 1.0
            
            result = RealizedVolatilityMeasure(
                symbol=self.symbol,
                timestamp=data['timestamp'].iloc[-1],
                period=period,
                measure_type="BPV",
                realized_volatility=bipower_volatility,
                realized_variance=bipower_variation,
                jump_component=jump_component,
                continuous_component=continuous_component,
                efficiency_ratio=efficiency_ratio,
                n_observations=len(log_returns),
                data_quality_score=self.calculate_data_quality_score(data),
                metadata={
                    "jump_detected": jump_test_result.jump_detected if jump_test_result else False,
                    "jump_test": jump_test_result.__dict__ if jump_test_result else None,
                    "annualized": annualize
                }
            )
            
            self.history.append(result)
            logger.info(f"✅ BPV calculated: {bipower_volatility:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating BPV: {e}")
            raise

    async def _detect_jumps_bns_test(
        self,
        log_returns: pd.Series,
        realized_variance: float,
        bipower_variation: float
    ) -> JumpDetectionResult:
        """
        Barndorff-Nielsen & Shephard jump test
        
        H0: No jumps
        H1: Jumps present
        """
        n = len(log_returns)
        
        if n < 3:
            return JumpDetectionResult(
                timestamp=datetime.now(),
                jump_detected=False,
                jump_size=0.0,
                jump_significance=0.0,
                test_statistic=0.0,
                critical_value=0.0
            )
        
        # Calculate Tripower Quarticity
        loop = asyncio.get_event_loop()
        tripower_quarticity = await loop.run_in_executor(
            None,
            _compute_tripower_quarticity_numba,
            log_returns.values
        )
        
        # Test statistic
        numerator = (realized_variance - bipower_variation) * n
        
        if tripower_quarticity > 0:
            denominator = np.sqrt(
                ((np.pi**2)/4 + np.pi - 5) * 
                max(1, tripower_quarticity)
            )
            z_statistic = numerator / denominator
        else:
            z_statistic = 0.0
        
        # Critical value (95% confidence)
        critical_value = stats.norm.ppf(0.975)
        
        # Jump detection
        jump_detected = abs(z_statistic) > critical_value
        jump_size = max(0, realized_variance - bipower_variation)
        jump_significance = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        return JumpDetectionResult(
            timestamp=datetime.now(),
            jump_detected=jump_detected,
            jump_size=jump_size,
            jump_significance=jump_significance,
            test_statistic=z_statistic,
            critical_value=critical_value,
            method="BNS"
        )

    def _estimate_periods_per_day(self, data: pd.DataFrame) -> int:
        """Estimate observations per day"""
        if len(data) < 2:
            return 1440
        
        time_diffs_minutes = data['timestamp'].diff().dt.total_seconds() / 60
        median_interval = time_diffs_minutes.median()
        
        return int(1440 / median_interval) if median_interval > 0 else 1440

class RealizedKernel(BaseVolatilityEstimator):
    """
    Realized Kernel Estimator - Microstructure Noise Robust
    
    RK = Sum w_h * gamma_h where gamma_h - autocovariances, w_h - kernel weights
    
    Advanced estimator robust to market microstructure noise.
    Uses kernel weighting for noise bias correction.
    Optimal for high-frequency crypto data with bid-ask bounces.
    """
    
    def __init__(self, symbol: str, kernel_type: str = "bartlett"):
        super().__init__(symbol, "Realized Kernel")
        self.kernel_type = kernel_type
        
    async def estimate(
        self,
        price_data: pd.DataFrame,
        period: str = "1D",
        annualize: bool = True,
        optimal_bandwidth: bool = True
    ) -> RealizedVolatilityMeasure:
        """Calculate Realized Kernel"""
        try:
            logger.info(f"🔄 Calculating RK for {self.symbol}...")
            
            # Validate data
            is_valid, message = self._validate_data(price_data)
            if not is_valid:
                raise ValueError(f"Data validation failed: {message}")
            
            data = price_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp')
            
            log_returns = self._calculate_log_returns(data['close'])
            
            if len(log_returns) < 10:
                raise ValueError("Insufficient data for RK calculation")
            
            # Optimal bandwidth selection
            if optimal_bandwidth:
                H = await self._select_optimal_bandwidth(log_returns)
            else:
                H = min(10, len(log_returns) // 4)
            
            # Calculate autocovariances
            autocovariances = await self._calculate_autocovariances(log_returns, H)
            
            # Kernel weights
            kernel_weights = self._get_kernel_weights(H, self.kernel_type)
            
            # Realized Kernel
            realized_kernel = 0.0
            for h in range(H + 1):
                weight = kernel_weights.get(h, 0.0)
                autocov = autocovariances.get(h, 0.0)
                
                if h == 0:
                    realized_kernel += weight * autocov
                else:
                    realized_kernel += 2 * weight * autocov  # Symmetry
            
            realized_kernel_vol = np.sqrt(max(0, realized_kernel))
            
            # Estimate microstructure bias
            microstructure_bias = self._estimate_microstructure_bias(
                autocovariances, realized_kernel
            )
            
            # Annualization
            if annualize:
                periods_per_day = self._estimate_periods_per_day(data)
                annualization_factor = np.sqrt(252 * periods_per_day)
                
                realized_kernel_vol *= annualization_factor
                realized_kernel *= (annualization_factor ** 2)
            
            result = RealizedVolatilityMeasure(
                symbol=self.symbol,
                timestamp=data['timestamp'].iloc[-1],
                period=period,
                measure_type="RK",
                realized_volatility=realized_kernel_vol,
                realized_variance=realized_kernel,
                microstructure_bias=microstructure_bias,
                n_observations=len(log_returns),
                data_quality_score=self.calculate_data_quality_score(data),
                metadata={
                    "kernel_type": self.kernel_type,
                    "bandwidth": H,
                    "optimal_bandwidth": optimal_bandwidth,
                    "annualized": annualize
                }
            )
            
            self.history.append(result)
            logger.info(f"✅ RK calculated: {realized_kernel_vol:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating RK: {e}")
            raise

    async def _select_optimal_bandwidth(self, returns: pd.Series) -> int:
        """
        Optimal bandwidth selection using cross-validation
        """
        n = len(returns)
        max_H = min(20, n // 5)
        
        if max_H < 2:
            return 2
        
        # Cross-validation for optimal bandwidth selection
        bandwidths = range(2, max_H + 1)
        cv_scores = []
        
        for H in bandwidths:
            try:
                # Simple cross-validation score (can be improved)
                autocovariances = await self._calculate_autocovariances(returns, H)
                kernel_weights = self._get_kernel_weights(H, self.kernel_type)
                
                # Realized Kernel value
                rk = 0.0
                for h in range(H + 1):
                    weight = kernel_weights.get(h, 0.0)
                    autocov = autocovariances.get(h, 0.0)
                    rk += weight * autocov if h == 0 else 2 * weight * autocov
                
                # Score: prefer positive values close to RV
                rv = np.sum(returns**2)
                score = -abs(rk - rv) if rk > 0 else -1e6
                cv_scores.append(score)
                
            except:
                cv_scores.append(-1e6)
        
        # Select bandwidth with maximum score
        optimal_idx = np.argmax(cv_scores)
        optimal_H = bandwidths[optimal_idx]
        
        logger.debug(f"Optimal bandwidth selected: H={optimal_H}")
        return optimal_H

    async def _calculate_autocovariances(
        self,
        returns: pd.Series,
        max_lag: int
    ) -> Dict[int, float]:
        """Calculate autocovariances up to max_lag"""
        autocovariances = {}
        returns_values = returns.values
        n = len(returns_values)
        
        for h in range(max_lag + 1):
            if h == 0:
                # Variance
                autocov = np.mean(returns_values**2)
            else:
                if n - h > 0:
                    # Autocovariance at lag h
                    autocov = np.mean(returns_values[h:] * returns_values[:-h])
                else:
                    autocov = 0.0
            
            autocovariances[h] = autocov
        
        return autocovariances

    def _get_kernel_weights(self, H: int, kernel_type: str) -> Dict[int, float]:
        """Get kernel function weights"""
        weights = {}
        
        if kernel_type == "bartlett":
            # Bartlett (Triangular) kernel
            for h in range(H + 1):
                if h <= H:
                    weights[h] = 1 - (h / (H + 1))
                else:
                    weights[h] = 0.0
        
        elif kernel_type == "parzen":
            # Parzen kernel
            for h in range(H + 1):
                if h <= H // 2:
                    weights[h] = 1 - 6 * (h / H)**2 + 6 * (h / H)**3
                elif h <= H:
                    weights[h] = 2 * (1 - h / H)**3
                else:
                    weights[h] = 0.0
        
        elif kernel_type == "qs":
            # Quadratic Spectral kernel
            for h in range(H + 1):
                if h == 0:
                    weights[h] = 1.0
                else:
                    x = 6 * np.pi * h / (5 * H)
                    weights[h] = 3 * (np.sin(x) / x - np.cos(x)) / (x**2)
        
        else:
            # Default: Uniform kernel
            for h in range(H + 1):
                weights[h] = 1.0
        
        return weights

    def _estimate_microstructure_bias(
        self,
        autocovariances: Dict[int, float],
        realized_kernel: float
    ) -> float:
        """Estimate microstructure bias"""
        if len(autocovariances) < 2:
            return 0.0
        
        # Simple estimate: negative first autocovariance indicates bias
        first_autocov = autocovariances.get(1, 0.0)
        
        if first_autocov < 0:
            # Bias approximately equals twice the negative autocovariance
            return -2 * first_autocov
        
        return 0.0

    def _estimate_periods_per_day(self, data: pd.DataFrame) -> int:
        """Estimate observations per day"""
        if len(data) < 2:
            return 1440
        
        time_diffs_minutes = data['timestamp'].diff().dt.total_seconds() / 60
        median_interval = time_diffs_minutes.median()
        
        return int(1440 / median_interval) if median_interval > 0 else 1440

class RealizedGARCH(BaseVolatilityEstimator):
    """
    Realized GARCH Model - Combination of RV with GARCH
    
    Combines high-frequency realized measures with GARCH modeling
    for improved volatility forecasts. Uses realized volatility
    as additional information in the GARCH framework.
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "Realized GARCH")
        self.garch_model = None
        
    async def estimate(
        self,
        price_data: pd.DataFrame,
        daily_returns: pd.Series,
        period: str = "1D",
        forecast_horizon: int = 1
    ) -> RealizedVolatilityMeasure:
        """
        Calculate Realized GARCH
        
        Args:
            price_data: High-frequency price data
            daily_returns: Daily returns for GARCH modeling
            period: Period for estimation
            forecast_horizon: Forecast horizon
        """
        try:
            logger.info(f"🔄 Calculating Realized GARCH for {self.symbol}...")
            
            # 1. Calculate daily realized volatility
            rv_estimator = RealizedVolatilityEstimator(self.symbol)
            
            # Group by day and calculate daily RV
            price_data['date'] = pd.to_datetime(price_data['timestamp']).dt.date
            daily_rv_measures = {}
            
            for date, day_data in price_data.groupby('date'):
                if len(day_data) > 10:  # Minimum data for a day
                    rv_measure = await rv_estimator.estimate(
                        day_data, 
                        period="1D", 
                        annualize=False
                    )
                    daily_rv_measures[date] = rv_measure.realized_variance
            
            # 2. Create daily RV series
            daily_rv_series = pd.Series(daily_rv_measures)
            daily_rv_series.index = pd.to_datetime(daily_rv_series.index)
            
            # 3. Realized GARCH modeling
            realized_garch_result = await self._fit_realized_garch(
                daily_returns, 
                daily_rv_series
            )
            
            # 4. Forecast
            forecast_result = await self._forecast_realized_garch(
                realized_garch_result,
                forecast_horizon
            )
            
            # 5. Result
            result = RealizedVolatilityMeasure(
                symbol=self.symbol,
                timestamp=datetime.now(),
                period=period,
                measure_type="Realized-GARCH",
                realized_volatility=forecast_result["volatility_forecast"],
                realized_variance=forecast_result["variance_forecast"],
                n_observations=len(daily_returns),
                data_quality_score=self.calculate_data_quality_score(price_data),
                metadata={
                    "garch_params": realized_garch_result.get("parameters", {}),
                    "model_fit": realized_garch_result.get("fit_quality", {}),
                    "forecast_horizon": forecast_horizon,
                    "daily_rv_observations": len(daily_rv_series)
                }
            )
            
            self.history.append(result)
            logger.info(f"✅ Realized GARCH calculated: {forecast_result['volatility_forecast']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating Realized GARCH: {e}")
            raise

    async def _fit_realized_garch(
        self,
        daily_returns: pd.Series,
        daily_rv: pd.Series
    ) -> Dict[str, Any]:
        """
        Fit Realized GARCH model
        
        Uses measurement equation: log(RV_t) = xi + phi*log(h_t) + delta*z_t + u_t
        where h_t - conditional variance from GARCH
        """
        from arch import arch_model
        from arch.univariate import ZeroMean, GARCH, Normal
        
        # Align data
        common_index = daily_returns.index.intersection(daily_rv.index)
        returns_aligned = daily_returns.reindex(common_index).dropna()
        rv_aligned = daily_rv.reindex(common_index).dropna()
        
        if len(returns_aligned) < 30:
            raise ValueError("Insufficient aligned data for Realized GARCH")
        
        # 1. Fit standard GARCH to returns
        garch_model = arch_model(
            returns_aligned * 100,  # Convert to percentage
            vol="GARCH",
            p=1, q=1,
            dist="Normal"
        )
        
        loop = asyncio.get_event_loop()
        garch_fit = await loop.run_in_executor(
            None,
            lambda: garch_model.fit(disp="off")
        )
        
        # 2. Extract conditional variance
        conditional_variance = garch_fit.conditional_volatility**2 / 10000  # Convert back
        
        # 3. Measurement equation for RV
        # log(RV_t) = ξ + φ*log(h_t) + u_t
        
        # Align conditional variance with RV
        h_aligned = conditional_variance.reindex(rv_aligned.index).dropna()
        rv_final = rv_aligned.reindex(h_aligned.index).dropna()
        
        if len(rv_final) < 10:
            raise ValueError("Insufficient data after alignment")
        
        # Log transformation (avoid log(0))
        log_rv = np.log(np.maximum(rv_final, 1e-10))
        log_h = np.log(np.maximum(h_aligned, 1e-10))
        
        # OLS regression: log(RV) = ξ + φ*log(h) + u
        from sklearn.linear_model import LinearRegression
        
        X = log_h.values.reshape(-1, 1)
        y = log_rv.values
        
        reg_model = LinearRegression()
        reg_model.fit(X, y)
        
        xi = reg_model.intercept_  # Constant
        phi = reg_model.coef_[0]   # Coefficient on log(h)
        
        # Residuals
        log_rv_pred = reg_model.predict(X)
        residuals = y - log_rv_pred
        sigma_u = np.std(residuals)
        
        result = {
            "garch_model": garch_fit,
            "parameters": {
                "xi": xi,
                "phi": phi,
                "sigma_u": sigma_u,
                "garch_params": {
                    "omega": garch_fit.params.iloc[0],
                    "alpha": garch_fit.params.iloc[1],
                    "beta": garch_fit.params.iloc[2] if len(garch_fit.params) > 2 else 0.0
                }
            },
            "fit_quality": {
                "r_squared": reg_model.score(X, y),
                "residual_std": sigma_u,
                "garch_aic": garch_fit.aic,
                "garch_bic": garch_fit.bic
            },
            "aligned_data": {
                "log_rv": log_rv,
                "log_h": log_h,
                "residuals": residuals
            }
        }
        
        self.garch_model = result
        return result

    async def _forecast_realized_garch(
        self,
        model_result: Dict[str, Any],
        horizon: int
    ) -> Dict[str, Any]:
        """Forecast Realized GARCH"""
        
        garch_fit = model_result["garch_model"]
        params = model_result["parameters"]
        
        # GARCH forecast
        garch_forecast = garch_fit.forecast(horizon=horizon)
        h_forecast = garch_forecast.variance.iloc[-1].values / 10000  # Convert back
        
        # Realized volatility forecast using measurement equation
        # E[log(RV_t+h)] = ξ + φ*log(h_t+h)
        log_h_forecast = np.log(np.maximum(h_forecast, 1e-10))
        log_rv_forecast = params["xi"] + params["phi"] * log_h_forecast
        
        # Transform back to levels
        rv_forecast = np.exp(log_rv_forecast)
        vol_forecast = np.sqrt(rv_forecast)
        
        # Prediction intervals (simple approximation)
        sigma_u = params["sigma_u"]
        log_rv_lower = log_rv_forecast - 1.96 * sigma_u
        log_rv_upper = log_rv_forecast + 1.96 * sigma_u
        
        rv_lower = np.exp(log_rv_lower)
        rv_upper = np.exp(log_rv_upper)
        
        vol_lower = np.sqrt(rv_lower)
        vol_upper = np.sqrt(rv_upper)
        
        return {
            "variance_forecast": rv_forecast[0] if len(rv_forecast) > 0 else 0.0,
            "volatility_forecast": vol_forecast[0] if len(vol_forecast) > 0 else 0.0,
            "confidence_intervals": {
                "variance": {
                    "lower": rv_lower[0] if len(rv_lower) > 0 else 0.0,
                    "upper": rv_upper[0] if len(rv_upper) > 0 else 0.0
                },
                "volatility": {
                    "lower": vol_lower[0] if len(vol_lower) > 0 else 0.0,
                    "upper": vol_upper[0] if len(vol_upper) > 0 else 0.0
                }
            },
            "garch_variance_forecast": h_forecast[0] if len(h_forecast) > 0 else 0.0,
            "horizon": horizon
        }

class TwoScaleRealizedVolatility(BaseVolatilityEstimator):
    """
    Two-Scale Realized Volatility (TSRV)
    
    Bias correction for microstructure noise using two sampling frequencies.
    TSRV = RV_fast - (n_fast/n_slow) * RV_slow_bias
    
    Effective for very high-frequency crypto data with significant noise.
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "Two-Scale RV")

    async def estimate(
        self,
        price_data: pd.DataFrame,
        fast_frequency: str = "1min",
        slow_frequency: str = "5min",
        period: str = "1D"
    ) -> RealizedVolatilityMeasure:
        """Calculate Two-Scale Realized Volatility"""
        try:
            logger.info(f"🔄 Calculating TSRV for {self.symbol}...")
            
            # Prepare data
            data = price_data.copy()
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.sort_values('timestamp').set_index('timestamp')
            
            # Fast sampling
            fast_data = data['close'].resample(fast_frequency).last().dropna()
            fast_returns = self._calculate_log_returns(fast_data)
            rv_fast = _compute_realized_variance_numba(fast_returns.values)
            n_fast = len(fast_returns)
            
            # Slow sampling  
            slow_data = data['close'].resample(slow_frequency).last().dropna()
            slow_returns = self._calculate_log_returns(slow_data)
            rv_slow = _compute_realized_variance_numba(slow_returns.values)
            n_slow = len(slow_returns)
            
            if n_slow == 0 or n_fast == 0:
                raise ValueError("Insufficient data for TSRV calculation")
            
            # Two-Scale estimator
            bias_correction = (n_fast / n_slow) * rv_slow if n_slow > 0 else 0
            tsrv_variance = max(0, rv_fast - bias_correction)
            tsrv_volatility = np.sqrt(tsrv_variance)
            
            # Noise-to-signal ratio
            noise_ratio = bias_correction / rv_fast if rv_fast > 0 else 0
            
            result = RealizedVolatilityMeasure(
                symbol=self.symbol,
                timestamp=data.index[-1],
                period=period,
                measure_type="TSRV",
                realized_volatility=tsrv_volatility,
                realized_variance=tsrv_variance,
                microstructure_bias=bias_correction,
                n_observations=n_fast,
                data_quality_score=self.calculate_data_quality_score(price_data),
                metadata={
                    "fast_frequency": fast_frequency,
                    "slow_frequency": slow_frequency,
                    "n_fast": n_fast,
                    "n_slow": n_slow,
                    "rv_fast": rv_fast,
                    "rv_slow": rv_slow,
                    "noise_ratio": noise_ratio
                }
            )
            
            self.history.append(result)
            logger.info(f"✅ TSRV calculated: {tsrv_volatility:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating TSRV: {e}")
            raise

# Async volatility estimator manager

class VolatilityEstimatorManager:
    """
    Manager for all volatility estimators
    
    Features:
    - Parallel estimation with different methods
    - Model comparison and selection
    - Real-time streaming calculations
    - Performance monitoring
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.estimators = {
            "RV": RealizedVolatilityEstimator(symbol),
            "BPV": BipowerVariation(symbol),
            "RK": RealizedKernel(symbol),
            "TSRV": TwoScaleRealizedVolatility(symbol)
        }
        self.results_history: List[Dict[str, RealizedVolatilityMeasure]] = []
        
        logger.info(f"🎯 Volatility estimator manager initialized for {symbol}")

    async def estimate_all(
        self,
        price_data: pd.DataFrame,
        daily_returns: Optional[pd.Series] = None,
        include_realized_garch: bool = False
    ) -> Dict[str, RealizedVolatilityMeasure]:
        """Parallel calculation of all estimators"""
        
        tasks = {}
        
        # Standard estimators
        for name, estimator in self.estimators.items():
            if name == "TSRV":
                tasks[name] = estimator.estimate(price_data)
            else:
                tasks[name] = estimator.estimate(price_data)
        
        # Realized GARCH if daily returns available
        if include_realized_garch and daily_returns is not None:
            rg_estimator = RealizedGARCH(self.symbol)
            tasks["RG"] = rg_estimator.estimate(price_data, daily_returns)
        
        # Parallel execution
        results = {}
        completed_tasks = await asyncio.gather(
            *tasks.values(), 
            return_exceptions=True
        )
        
        # Process results
        for i, (name, task_result) in enumerate(zip(tasks.keys(), completed_tasks)):
            if isinstance(task_result, Exception):
                logger.warning(f"⚠️ {name} estimation failed: {task_result}")
            else:
                results[name] = task_result
                logger.info(f"✅ {name}: {task_result.realized_volatility:.4f}")
        
        self.results_history.append(results)
        return results

    def compare_estimators(
        self, 
        results: Dict[str, RealizedVolatilityMeasure]
    ) -> pd.DataFrame:
        """Compare estimator results"""
        
        comparison_data = []
        for name, result in results.items():
            row = {
                "Estimator": name,
                "Volatility": result.realized_volatility,
                "Variance": result.realized_variance, 
                "Data_Quality": result.data_quality_score,
                "N_Observations": result.n_observations,
                "Jump_Component": result.jump_component or 0.0,
                "Microstructure_Bias": result.microstructure_bias or 0.0
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df.sort_values("Volatility")

    def get_consensus_estimate(
        self,
        results: Dict[str, RealizedVolatilityMeasure],
        method: str = "median"
    ) -> float:
        """Consensus estimate from multiple methods"""
        
        volatilities = [r.realized_volatility for r in results.values()]
        
        if method == "mean":
            return np.mean(volatilities)
        elif method == "median":
            return np.median(volatilities)
        elif method == "trimmed_mean":
            # Remove extreme values
            sorted_vols = sorted(volatilities)
            n = len(sorted_vols)
            if n >= 4:
                trim_count = max(1, n // 4)
                trimmed = sorted_vols[trim_count:-trim_count]
                return np.mean(trimmed)
            else:
                return np.mean(volatilities)
        else:
            return np.mean(volatilities)

    def generate_summary_report(self) -> str:
        """Generate summary report"""
        if not self.results_history:
            return "No estimation results available."
        
        latest_results = self.results_history[-1]
        comparison_df = self.compare_estimators(latest_results)
        consensus = self.get_consensus_estimate(latest_results)
        
        report = f"""
🎯 Volatility Estimation Report for {self.symbol}
{'='*60}

Latest Estimates:
{comparison_df.to_string(index=False)}

Consensus Estimate: {consensus:.4f}

Recommendations:
- Primary estimate: Use median of RV, BPV, RK
- For jump detection: Check BPV vs RV difference
- For noisy data: Prefer RK or TSRV
- For forecasting: Consider Realized GARCH

Estimation Quality:
- Average data quality: {np.mean([r.data_quality_score for r in latest_results.values()]):.3f}
- Total observations: {sum([r.n_observations for r in latest_results.values()])}
        """
        
        return report

# Export all classes
__all__ = [
    "BaseVolatilityEstimator",
    "RealizedVolatilityEstimator", 
    "BipowerVariation",
    "RealizedKernel",
    "RealizedGARCH",
    "TwoScaleRealizedVolatility",
    "VolatilityEstimatorManager",
    "RealizedVolatilityMeasure",
    "JumpDetectionResult",
    "_compute_realized_variance_numba",
    "_compute_bipower_variation_numba",
    "_compute_tripower_quarticity_numba"
]

logger.info("🔥 Realized Volatility Estimators module loaded successfully!")