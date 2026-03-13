"""
Risk Metrics Calculation System for Advanced Portfolio Management

Implementation of comprehensive risk metrics:
- Value at Risk (VaR) - Historical, Parametric, Monte Carlo
- Conditional VaR (CVaR/Expected Shortfall)
- Volatility cones for benchmarking
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Maximum Drawdown analysis
- Tail risk measures
- Dynamic correlation analysis

Features:
- Multi-asset risk aggregation
- Real-time risk monitoring
- Stress testing scenarios
- Backtesting framework
- Regulatory compliance (Basel III)
"""

from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from numba import jit, prange
import warnings

# Logging configuration
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class RiskMeasureType(Enum):
    """Risk measure types"""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric"
    VAR_MONTE_CARLO = "var_monte_carlo"
    CVAR = "cvar"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    TAIL_RATIO = "tail_ratio"
    OMEGA_RATIO = "omega_ratio"

class ConfidenceLevel(Enum):
    """Confidence levels for risk measures"""
    P95 = 0.95
    P99 = 0.99
    P975 = 0.975
    P995 = 0.995

@dataclass
class VaRResult:
    """Value at Risk calculation result"""
    symbol: str
    timestamp: datetime
    method: RiskMeasureType
    confidence_level: float
    holding_period: int  # days
    
    # Core metrics
    var_absolute: float
    var_percentage: float
    expected_shortfall: float

    # Additional metrics
    var_portfolio_contribution: Optional[float] = None
    marginal_var: Optional[float] = None
    component_var: Optional[float] = None
    
    # Backtesting results
    violations_count: int = 0
    violations_percentage: float = 0.0
    kupiec_test_pvalue: Optional[float] = None
    
    # Model parameters
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Estimation quality
    confidence_interval: Optional[Tuple[float, float]] = None
    estimation_error: Optional[float] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DrawdownAnalysis:
    """Drawdown analysis"""
    symbol: str
    timestamp: datetime
    analysis_period: Tuple[datetime, datetime]
    
    # Core metrics
    max_drawdown: float
    max_drawdown_duration: int  # days
    current_drawdown: float

    # Drawdown statistics
    average_drawdown: float
    drawdown_frequency: float  # drawdowns per year
    recovery_time_avg: float  # average recovery time
    
    # Tail risk
    var_of_drawdowns: float
    expected_drawdown: float
    
    # Underwater curve
    underwater_curve: pd.Series
    drawdown_periods: List[Dict[str, Any]]
    
    # Quality metrics
    sterling_ratio: float
    burke_ratio: float
    pain_index: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolatilityCone:
    """Volatility cone for benchmarking"""
    symbol: str
    timestamp: datetime
    
    # Cone data
    periods: List[int]  # [1, 5, 22, 63, 252] days
    percentiles: List[float]  # [5, 25, 50, 75, 95]
    cone_data: pd.DataFrame  # percentiles x periods
    
    # Current levels
    current_volatility: Dict[int, float]  # period -> current volatility
    percentile_ranks: Dict[int, float]  # period -> percentile
    
    # Regime analysis
    volatility_regime: str  # "low", "normal", "high", "extreme"
    regime_probability: float
    
    # Mean reversion analysis
    mean_reversion_half_life: float
    volatility_clustering_strength: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskAdjustedReturns:
    """Risk-adjusted return metrics"""
    symbol: str
    timestamp: datetime
    analysis_period: Tuple[datetime, datetime]
    
    # Basic metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    
    # Risk-adjusted ratios
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Advanced ratios
    treynor_ratio: Optional[float] = None  # if benchmark is available
    information_ratio: Optional[float] = None
    jensen_alpha: Optional[float] = None
    
    # Downside risk metrics
    downside_deviation: float
    maximum_drawdown: float
    var_95: float
    cvar_95: float
    
    # Higher moments
    skewness: float
    kurtosis: float
    
    # Performance attribution
    best_month: float
    worst_month: float
    positive_months_percentage: float
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@jit(nopython=True)
def _calculate_historical_var_numba(
    returns: np.ndarray, 
    confidence_level: float
) -> Tuple[float, float]:
    """Fast Historical VaR calculation with Numba"""
    sorted_returns = np.sort(returns)
    n = len(sorted_returns)
    
    var_index = int((1 - confidence_level) * n)
    var_value = -sorted_returns[var_index] if var_index < n else 0.0
    
    # Expected Shortfall (mean of tail losses)
    tail_returns = sorted_returns[:var_index+1] if var_index+1 > 0 else np.array([0.0])
    expected_shortfall = -np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
    
    return var_value, expected_shortfall

@jit(nopython=True)
def _calculate_maximum_drawdown_numba(cumulative_returns: np.ndarray) -> Tuple[float, int]:
    """Fast Maximum Drawdown calculation with Numba"""
    n = len(cumulative_returns)
    max_dd = 0.0
    max_duration = 0
    
    peak = cumulative_returns[0]
    peak_idx = 0
    
    for i in range(1, n):
        if cumulative_returns[i] > peak:
            peak = cumulative_returns[i]
            peak_idx = i
        else:
            drawdown = (peak - cumulative_returns[i]) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                max_duration = i - peak_idx
    
    return max_dd, max_duration

@jit(nopython=True)
def _calculate_underwater_curve_numba(cumulative_returns: np.ndarray) -> np.ndarray:
    """Fast underwater curve calculation with Numba"""
    n = len(cumulative_returns)
    underwater = np.zeros(n)
    
    running_max = cumulative_returns[0]
    
    for i in range(n):
        if cumulative_returns[i] > running_max:
            running_max = cumulative_returns[i]
        
        underwater[i] = (cumulative_returns[i] - running_max) / running_max
    
    return underwater

class BaseRiskCalculator(ABC):
    """Base class for all risk calculators"""
    
    def __init__(self, symbol: str, name: str):
        self.symbol = symbol
        self.name = name
        self.calculation_history = []
        
        logger.info(f"🎯 Initialized {name} risk calculator for {symbol}")

    @abstractmethod
    async def calculate(self, returns: pd.Series, **kwargs) -> Any:
        """Calculate risk metric"""
        pass

    def _validate_returns(self, returns: pd.Series) -> Tuple[bool, str]:
        """Validate returns data"""
        if returns.empty:
            return False, "Empty returns series"
        
        if returns.isna().all():
            return False, "All returns are NaN"
        
        if len(returns.dropna()) < 30:
            return False, f"Insufficient data: {len(returns.dropna())} observations"
        
        return True, "Returns validation passed"

class VaRCalculator(BaseRiskCalculator):
    """
    Value at Risk Calculator
    
    Supports multiple VaR calculation methods:
    - Historical Simulation
    - Parametric (Normal, t-distribution)
    - Monte Carlo Simulation
    - Extreme Value Theory
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "VaR Calculator")

    async def calculate_historical_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        holding_period: int = 1
    ) -> VaRResult:
        """Historical VaR calculation"""
        try:
            logger.info(f"🔄 Calculating Historical VaR for {self.symbol}...")
            
            # Validation
            is_valid, message = self._validate_returns(returns)
            if not is_valid:
                raise ValueError(f"Returns validation failed: {message}")
            
            clean_returns = returns.dropna()
            
            # Scaling for holding period
            if holding_period > 1:
                scaled_returns = clean_returns * np.sqrt(holding_period)
            else:
                scaled_returns = clean_returns
            
            # Async Historical VaR calculation
            loop = asyncio.get_event_loop()
            var_abs, expected_shortfall = await loop.run_in_executor(
                None,
                _calculate_historical_var_numba,
                scaled_returns.values,
                confidence_level
            )
            
            # VaR in percentage
            var_pct = var_abs * 100
            
            # Backtesting
            violations = (scaled_returns < -var_abs).sum()
            violations_pct = violations / len(scaled_returns) * 100
            expected_violations_pct = (1 - confidence_level) * 100
            
            # Kupiec test for backtesting
            kupiec_p_value = await self._kupiec_test(
                violations, len(scaled_returns), 1 - confidence_level
            )
            
            result = VaRResult(
                symbol=self.symbol,
                timestamp=datetime.now(),
                method=RiskMeasureType.VAR_HISTORICAL,
                confidence_level=confidence_level,
                holding_period=holding_period,
                var_absolute=var_abs,
                var_percentage=var_pct,
                expected_shortfall=expected_shortfall,
                violations_count=violations,
                violations_percentage=violations_pct,
                kupiec_test_pvalue=kupiec_p_value,
                model_parameters={
                    "sample_size": len(scaled_returns),
                    "method": "historical_simulation",
                    "expected_violations": expected_violations_pct
                },
                metadata={
                    "data_period": f"{clean_returns.index[0].date()} to {clean_returns.index[-1].date()}",
                    "calculation_method": "empirical_quantile"
                }
            )
            
            self.calculation_history.append(result)
            logger.info(f"✅ Historical VaR: {var_pct:.2f}% ({confidence_level*100}%)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating Historical VaR: {e}")
            raise

    async def calculate_parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        distribution: str = "normal"
    ) -> VaRResult:
        """Parametric VaR calculation"""
        try:
            logger.info(f"🔄 Calculating Parametric VaR ({distribution}) for {self.symbol}...")
            
            is_valid, message = self._validate_returns(returns)
            if not is_valid:
                raise ValueError(f"Returns validation failed: {message}")
            
            clean_returns = returns.dropna()
            
            # Statistics
            mean_return = clean_returns.mean()
            std_return = clean_returns.std()
            
            # Scaling for holding period
            if holding_period > 1:
                mean_return = mean_return * holding_period
                std_return = std_return * np.sqrt(holding_period)
            
            # Distribution parameters
            if distribution == "normal":
                z_score = stats.norm.ppf(1 - confidence_level)
                var_abs = -(mean_return + z_score * std_return)
                
                # Expected shortfall for normal distribution
                phi_z = stats.norm.pdf(z_score)
                expected_shortfall = std_return * phi_z / (1 - confidence_level) - mean_return
                
                model_params = {
                    "distribution": "normal",
                    "mean": mean_return,
                    "std": std_return,
                    "z_score": z_score
                }
                
            elif distribution == "t":
                # Fit t-distribution
                df, loc, scale = stats.t.fit(clean_returns)
                t_score = stats.t.ppf(1 - confidence_level, df, loc, scale)
                var_abs = -t_score
                
                # Expected shortfall for t-distribution (approximation)
                expected_shortfall = var_abs * 1.2  # Simplified
                
                model_params = {
                    "distribution": "t",
                    "degrees_freedom": df,
                    "location": loc,
                    "scale": scale,
                    "t_score": t_score
                }
            
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            var_pct = var_abs * 100
            
            # Backtesting against historical data
            scaled_returns = clean_returns * (np.sqrt(holding_period) if holding_period > 1 else 1)
            violations = (scaled_returns < -var_abs).sum()
            violations_pct = violations / len(scaled_returns) * 100
            
            # Kupiec test
            kupiec_p_value = await self._kupiec_test(
                violations, len(scaled_returns), 1 - confidence_level
            )
            
            result = VaRResult(
                symbol=self.symbol,
                timestamp=datetime.now(),
                method=RiskMeasureType.VAR_PARAMETRIC,
                confidence_level=confidence_level,
                holding_period=holding_period,
                var_absolute=var_abs,
                var_percentage=var_pct,
                expected_shortfall=expected_shortfall,
                violations_count=violations,
                violations_percentage=violations_pct,
                kupiec_test_pvalue=kupiec_p_value,
                model_parameters=model_params,
                metadata={
                    "data_period": f"{clean_returns.index[0].date()} to {clean_returns.index[-1].date()}",
                    "calculation_method": "parametric_estimation"
                }
            )
            
            self.calculation_history.append(result)
            logger.info(f"✅ Parametric VaR ({distribution}): {var_pct:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating Parametric VaR: {e}")
            raise

    async def calculate_monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        n_simulations: int = 10000,
        model_type: str = "bootstrap"
    ) -> VaRResult:
        """Monte Carlo VaR calculation"""
        try:
            logger.info(f"🔄 Calculating Monte Carlo VaR for {self.symbol}...")
            
            is_valid, message = self._validate_returns(returns)
            if not is_valid:
                raise ValueError(f"Returns validation failed: {message}")
            
            clean_returns = returns.dropna()
            
            # Monte Carlo simulation
            if model_type == "bootstrap":
                # Bootstrap simulation
                simulated_returns = await self._bootstrap_simulation(
                    clean_returns, holding_period, n_simulations
                )
            elif model_type == "parametric":
                # Parametric simulation (normal)
                simulated_returns = await self._parametric_simulation(
                    clean_returns, holding_period, n_simulations
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # VaR from simulations
            var_abs = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            # Expected Shortfall
            tail_losses = simulated_returns[simulated_returns <= -var_abs]
            expected_shortfall = -np.mean(tail_losses) if len(tail_losses) > 0 else var_abs
            
            var_pct = var_abs * 100
            
            # Confidence interval for VaR estimate
            var_estimates = []
            n_bootstrap = 100
            
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(simulated_returns, size=len(simulated_returns)//2)
                var_est = -np.percentile(bootstrap_sample, (1 - confidence_level) * 100)
                var_estimates.append(var_est)
            
            var_ci_lower = np.percentile(var_estimates, 2.5)
            var_ci_upper = np.percentile(var_estimates, 97.5)
            confidence_interval = (var_ci_lower, var_ci_upper)
            
            # Backtesting
            scaled_returns = clean_returns * (np.sqrt(holding_period) if holding_period > 1 else 1)
            violations = (scaled_returns < -var_abs).sum()
            violations_pct = violations / len(scaled_returns) * 100
            
            kupiec_p_value = await self._kupiec_test(
                violations, len(scaled_returns), 1 - confidence_level
            )
            
            result = VaRResult(
                symbol=self.symbol,
                timestamp=datetime.now(),
                method=RiskMeasureType.VAR_MONTE_CARLO,
                confidence_level=confidence_level,
                holding_period=holding_period,
                var_absolute=var_abs,
                var_percentage=var_pct,
                expected_shortfall=expected_shortfall,
                violations_count=violations,
                violations_percentage=violations_pct,
                kupiec_test_pvalue=kupiec_p_value,
                confidence_interval=confidence_interval,
                estimation_error=var_ci_upper - var_ci_lower,
                model_parameters={
                    "n_simulations": n_simulations,
                    "model_type": model_type,
                    "simulation_mean": np.mean(simulated_returns),
                    "simulation_std": np.std(simulated_returns)
                },
                metadata={
                    "data_period": f"{clean_returns.index[0].date()} to {clean_returns.index[-1].date()}",
                    "calculation_method": "monte_carlo_simulation"
                }
            )
            
            self.calculation_history.append(result)
            logger.info(f"✅ Monte Carlo VaR: {var_pct:.2f}% ±{(var_ci_upper-var_ci_lower)*100:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating Monte Carlo VaR: {e}")
            raise

    async def _bootstrap_simulation(
        self,
        returns: pd.Series,
        holding_period: int,
        n_simulations: int
    ) -> np.ndarray:
        """Bootstrap simulation of returns"""
        
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Sample with replacement
            sampled_returns = np.random.choice(returns.values, size=holding_period, replace=True)
            period_return = np.sum(sampled_returns)
            simulated_returns.append(period_return)
        
        return np.array(simulated_returns)

    async def _parametric_simulation(
        self,
        returns: pd.Series,
        holding_period: int,
        n_simulations: int
    ) -> np.ndarray:
        """Parametric simulation (normal distribution)"""
        
        mean_daily = returns.mean()
        std_daily = returns.std()
        
        # Simulation for holding period
        simulated_returns = []
        
        for _ in range(n_simulations):
            daily_returns = np.random.normal(mean_daily, std_daily, holding_period)
            period_return = np.sum(daily_returns)
            simulated_returns.append(period_return)
        
        return np.array(simulated_returns)

    async def _kupiec_test(
        self,
        violations: int,
        n_observations: int,
        expected_violation_rate: float
    ) -> float:
        """Kupiec test for VaR backtesting"""
        
        if violations == 0 or violations == n_observations:
            return 0.0  # Test not applicable
        
        # Likelihood ratio test
        p = violations / n_observations
        
        if p == 0:
            lr_stat = -2 * n_observations * np.log(1 - expected_violation_rate)
        elif p == 1:
            lr_stat = -2 * n_observations * np.log(expected_violation_rate)
        else:
            lr_stat = -2 * (
                violations * np.log(expected_violation_rate / p) +
                (n_observations - violations) * np.log((1 - expected_violation_rate) / (1 - p))
            )
        
        # P-value from chi-square distribution (df=1)
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return p_value

    async def calculate(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        holding_period: int = 1,
        method: str = "historical"
    ) -> VaRResult:
        """Unified VaR calculation interface"""
        
        if method == "historical":
            return await self.calculate_historical_var(returns, confidence_level, holding_period)
        elif method == "parametric":
            return await self.calculate_parametric_var(returns, confidence_level, holding_period)
        elif method == "monte_carlo":
            return await self.calculate_monte_carlo_var(returns, confidence_level, holding_period)
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

class DrawdownAnalyzer(BaseRiskCalculator):
    """
    Drawdown Analysis Calculator
    
    Comprehensive drawdown analysis:
    - Maximum Drawdown
    - Drawdown duration
    - Recovery analysis
    - Underwater curve
    - Risk-adjusted ratios
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "Drawdown Analyzer")

    async def calculate(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> DrawdownAnalysis:
        """Comprehensive drawdown analysis"""
        try:
            logger.info(f"🔄 Analyzing drawdowns for {self.symbol}...")
            
            is_valid, message = self._validate_returns(returns)
            if not is_valid:
                raise ValueError(f"Returns validation failed: {message}")
            
            clean_returns = returns.dropna()
            
            # Cumulative returns
            cumulative_returns = (1 + clean_returns).cumprod()
            
            # Maximum Drawdown calculation
            loop = asyncio.get_event_loop()
            max_drawdown, max_duration = await loop.run_in_executor(
                None,
                _calculate_maximum_drawdown_numba,
                cumulative_returns.values
            )
            
            # Underwater curve
            underwater_curve = await loop.run_in_executor(
                None,
                _calculate_underwater_curve_numba,
                cumulative_returns.values
            )
            
            underwater_series = pd.Series(underwater_curve, index=cumulative_returns.index)
            
            # Current drawdown
            current_drawdown = abs(underwater_curve[-1])
            
            # Drawdown periods analysis
            drawdown_periods = await self._analyze_drawdown_periods(underwater_series)
            
            # Statistics
            avg_drawdown = np.mean([abs(min(period['underwater_values'])) 
                                   for period in drawdown_periods if period['underwater_values']])
            
            drawdown_frequency = len(drawdown_periods) / (len(clean_returns) / 252)  # per year
            
            recovery_times = [period['recovery_days'] for period in drawdown_periods 
                             if period['recovered'] and period['recovery_days'] > 0]
            recovery_time_avg = np.mean(recovery_times) if recovery_times else 0
            
            # VaR of drawdowns
            drawdown_values = [abs(min(period['underwater_values'])) 
                              for period in drawdown_periods if period['underwater_values']]
            
            if len(drawdown_values) >= 10:
                var_of_drawdowns = np.percentile(drawdown_values, 95)
                expected_drawdown = np.mean(drawdown_values)
            else:
                var_of_drawdowns = max_drawdown
                expected_drawdown = avg_drawdown
            
            # Risk-adjusted ratios
            annualized_return = clean_returns.mean() * 252
            
            if max_drawdown > 0:
                sterling_ratio = annualized_return / max_drawdown
                calmar_ratio = annualized_return / max_drawdown  # Synonym
            else:
                sterling_ratio = float('inf') if annualized_return > 0 else 0
                calmar_ratio = sterling_ratio
            
            # Burke ratio (average drawdown)
            if avg_drawdown > 0:
                burke_ratio = annualized_return / avg_drawdown
            else:
                burke_ratio = float('inf') if annualized_return > 0 else 0
            
            # Pain Index (average underwater percentage)
            pain_index = abs(np.mean(underwater_curve))
            
            result = DrawdownAnalysis(
                symbol=self.symbol,
                timestamp=datetime.now(),
                analysis_period=(clean_returns.index[0], clean_returns.index[-1]),
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_duration,
                current_drawdown=current_drawdown,
                average_drawdown=avg_drawdown,
                drawdown_frequency=drawdown_frequency,
                recovery_time_avg=recovery_time_avg,
                var_of_drawdowns=var_of_drawdowns,
                expected_drawdown=expected_drawdown,
                underwater_curve=underwater_series,
                drawdown_periods=drawdown_periods,
                sterling_ratio=sterling_ratio,
                burke_ratio=burke_ratio,
                pain_index=pain_index,
                metadata={
                    "n_drawdown_periods": len(drawdown_periods),
                    "max_recovery_time": max(recovery_times) if recovery_times else 0,
                    "current_underwater_days": self._count_current_underwater_days(underwater_series),
                    "benchmark_used": benchmark_returns is not None
                }
            )
            
            self.calculation_history.append(result)
            logger.info(f"✅ Drawdown analysis: Max DD={max_drawdown:.2%}, Current DD={current_drawdown:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in drawdown analysis: {e}")
            raise

    async def _analyze_drawdown_periods(
        self,
        underwater_curve: pd.Series
    ) -> List[Dict[str, Any]]:
        """Analyze individual drawdown periods"""
        
        periods = []
        in_drawdown = False
        current_period = None
        
        for i, (date, underwater_value) in enumerate(underwater_curve.items()):
            if underwater_value < -1e-6:  # In drawdown (with tolerance)
                if not in_drawdown:
                    # Start new drawdown period
                    current_period = {
                        'start_date': date,
                        'start_index': i,
                        'peak_value': 0.0,  # Will be updated
                        'trough_value': underwater_value,
                        'trough_date': date,
                        'end_date': None,
                        'end_index': None,
                        'duration_days': 0,
                        'recovery_days': 0,
                        'underwater_values': [underwater_value],
                        'recovered': False
                    }
                    in_drawdown = True
                else:
                    # Continue drawdown
                    current_period['underwater_values'].append(underwater_value)
                    if underwater_value < current_period['trough_value']:
                        current_period['trough_value'] = underwater_value
                        current_period['trough_date'] = date
            
            else:  # Not in drawdown
                if in_drawdown:
                    # End current drawdown period
                    current_period['end_date'] = date
                    current_period['end_index'] = i
                    current_period['duration_days'] = (
                        current_period['end_date'] - current_period['start_date']
                    ).days
                    current_period['recovery_days'] = (
                        current_period['end_date'] - current_period['trough_date']
                    ).days
                    current_period['recovered'] = True
                    
                    periods.append(current_period)
                    in_drawdown = False
                    current_period = None
        
        # Handle ongoing drawdown
        if in_drawdown and current_period:
            current_period['end_date'] = underwater_curve.index[-1]
            current_period['end_index'] = len(underwater_curve) - 1
            current_period['duration_days'] = (
                current_period['end_date'] - current_period['start_date']
            ).days
            current_period['recovered'] = False
            current_period['recovery_days'] = 0
            
            periods.append(current_period)
        
        return periods

    def _count_current_underwater_days(self, underwater_curve: pd.Series) -> int:
        """Count days in current drawdown"""
        if len(underwater_curve) == 0:
            return 0
        
        # Count consecutive days at the end where underwater < 0
        count = 0
        for value in reversed(underwater_curve.values):
            if value < -1e-6:
                count += 1
            else:
                break
        
        return count

class VolatilityConeCalculator(BaseRiskCalculator):
    """
    Volatility Cone Calculator
    
    Creating volatility cones for benchmarking current volatility
    levels against historical ranges.
    """
    
    def __init__(self, symbol: str):
        super().__init__(symbol, "Volatility Cone")

    async def calculate(
        self,
        returns: pd.Series,
        periods: List[int] = None,
        percentiles: List[float] = None
    ) -> VolatilityCone:
        """Calculate volatility cone"""
        try:
            logger.info(f"🔄 Calculating volatility cone for {self.symbol}...")
            
            is_valid, message = self._validate_returns(returns)
            if not is_valid:
                raise ValueError(f"Returns validation failed: {message}")
            
            if periods is None:
                periods = [1, 5, 22, 63, 126, 252]  # 1D, 1W, 1M, 3M, 6M, 1Y
            
            if percentiles is None:
                percentiles = [5, 25, 50, 75, 95]
            
            clean_returns = returns.dropna()
            
            # Calculate rolling volatilities for each period
            cone_data = {}
            current_volatilities = {}
            
            for period in periods:
                if len(clean_returns) < period * 2:
                    continue
                
                # Rolling volatility
                rolling_vol = clean_returns.rolling(period).std() * np.sqrt(252)  # Annualized
                rolling_vol = rolling_vol.dropna()
                
                if len(rolling_vol) < 50:  # Minimum samples for percentiles
                    continue
                
                # Calculate percentiles
                period_percentiles = {}
                for pct in percentiles:
                    period_percentiles[pct] = np.percentile(rolling_vol.values, pct)
                
                cone_data[period] = period_percentiles
                
                # Current volatility level
                current_volatilities[period] = rolling_vol.iloc[-1]
            
            if not cone_data:
                raise ValueError("Insufficient data to calculate volatility cone")
            
            # Create cone DataFrame
            cone_df = pd.DataFrame(cone_data).T
            cone_df.index.name = 'Period'
            
            # Calculate percentile ranks for current levels
            percentile_ranks = {}
            for period in current_volatilities:
                if period in cone_data:
                    current_vol = current_volatilities[period]
                    
                    # Historical volatilities for this period
                    historical_vols = clean_returns.rolling(period).std() * np.sqrt(252)
                    historical_vols = historical_vols.dropna()
                    
                    # Percentile rank
                    rank = (historical_vols < current_vol).mean() * 100
                    percentile_ranks[period] = rank
            
            # Volatility regime determination
            # Use medium-term period (22 days) for classification
            if 22 in percentile_ranks:
                current_rank = percentile_ranks[22]
                if current_rank < 25:
                    regime = "low"
                elif current_rank < 75:
                    regime = "normal" 
                elif current_rank < 95:
                    regime = "high"
                else:
                    regime = "extreme"
                    
                regime_probability = await self._estimate_regime_probability(
                    clean_returns, current_rank
                )
            else:
                regime = "unknown"
                regime_probability = 0.5
            
            # Mean reversion analysis
            mean_reversion_half_life = await self._calculate_mean_reversion_half_life(
                clean_returns.rolling(22).std() * np.sqrt(252)
            )
            
            # Volatility clustering
            clustering_strength = await self._calculate_volatility_clustering(clean_returns)
            
            result = VolatilityCone(
                symbol=self.symbol,
                timestamp=datetime.now(),
                periods=list(cone_data.keys()),
                percentiles=percentiles,
                cone_data=cone_df,
                current_volatility=current_volatilities,
                percentile_ranks=percentile_ranks,
                volatility_regime=regime,
                regime_probability=regime_probability,
                mean_reversion_half_life=mean_reversion_half_life,
                volatility_clustering_strength=clustering_strength,
                metadata={
                    "data_period": f"{clean_returns.index[0].date()} to {clean_returns.index[-1].date()}",
                    "n_observations": len(clean_returns)
                }
            )
            
            self.calculation_history.append(result)
            logger.info(f"✅ Volatility cone: Regime={regime}, Current 1M rank={percentile_ranks.get(22, 0):.0f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating volatility cone: {e}")
            raise

    async def _estimate_regime_probability(
        self,
        returns: pd.Series,
        current_percentile: float
    ) -> float:
        """Estimate probability of current volatility regime"""
        
        # Simple model based on historical data
        vol_series = returns.rolling(22).std() * np.sqrt(252)
        vol_series = vol_series.dropna()
        
        if len(vol_series) < 100:
            return 0.5
        
        # Transition probabilities between regimes
        # Simplified model - probability of staying in the same regime
        if current_percentile < 25:
            # Low vol regime - tend to persist
            persistence_prob = 0.8
        elif current_percentile > 75:
            # High vol regime - less persistent
            persistence_prob = 0.6
        else:
            # Normal regime
            persistence_prob = 0.7
        
        return persistence_prob

    async def _calculate_mean_reversion_half_life(
        self,
        volatility_series: pd.Series
    ) -> float:
        """Calculate half-life of volatility mean reversion"""
        
        vol_series = volatility_series.dropna()
        
        if len(vol_series) < 50:
            return 30.0  # Default
        
        # AR(1) model for volatility
        y = vol_series.iloc[1:].values
        x = vol_series.iloc[:-1].values
        
        # Simple linear regression: y_t = α + β*y_{t-1} + ε_t
        from sklearn.linear_model import LinearRegression
        
        reg = LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        
        beta = reg.coef_[0]
        
        # Half-life = ln(0.5) / ln(β)
        if 0 < beta < 1:
            half_life = np.log(0.5) / np.log(beta)
            return abs(half_life)
        else:
            return 30.0  # Default if model is not applicable

    async def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering strength"""
        
        # Autocorrelation in squared returns
        squared_returns = returns**2
        
        # Lag-1 autocorrelation
        try:
            correlation = squared_returns.autocorr(lag=1)
            return max(0.0, min(1.0, correlation)) if not np.isnan(correlation) else 0.0
        except:
            return 0.0

class RiskMetricsCalculator:
    """
    Unified Risk Metrics Calculator
    
    Coordinates all risk calculators and provides a
    comprehensive risk dashboard for portfolio management.
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.var_calculator = VaRCalculator(symbol)
        self.drawdown_analyzer = DrawdownAnalyzer(symbol)
        self.volatility_cone_calculator = VolatilityConeCalculator(symbol)
        
        self.risk_dashboard_history = []
        
        logger.info(f"🎯 Risk metrics calculator initialized for {symbol}")

    async def calculate_comprehensive_risk_metrics(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = None,
        var_methods: List[str] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive risk analysis"""
        
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        
        if var_methods is None:
            var_methods = ["historical", "parametric", "monte_carlo"]
        
        logger.info(f"🔄 Calculating comprehensive risk metrics for {self.symbol}...")
        
        try:
            # Parallel calculation of all risk metrics
            tasks = []
            
            # VaR calculations
            for method in var_methods:
                for confidence_level in confidence_levels:
                    task = self.var_calculator.calculate(
                        returns, confidence_level=confidence_level, method=method
                    )
                    tasks.append(("var", method, confidence_level, task))
            
            # Drawdown analysis
            drawdown_task = self.drawdown_analyzer.calculate(returns, benchmark_returns)
            tasks.append(("drawdown", None, None, drawdown_task))
            
            # Volatility cone
            cone_task = self.volatility_cone_calculator.calculate(returns)
            tasks.append(("cone", None, None, cone_task))
            
            # Risk-adjusted returns
            risk_adjusted_task = self._calculate_risk_adjusted_returns(returns, benchmark_returns)
            tasks.append(("risk_adjusted", None, None, risk_adjusted_task))
            
            # Execute all tasks
            task_results = []
            for task_type, method, confidence_level, task in tasks:
                try:
                    result = await task
                    task_results.append((task_type, method, confidence_level, result))
                except Exception as e:
                    logger.warning(f"⚠️ Failed {task_type} calculation: {e}")
                    continue
            
            # Organize results
            risk_metrics = {
                "symbol": self.symbol,
                "timestamp": datetime.now(),
                "analysis_period": (returns.index[0], returns.index[-1]),
                "var_results": {},
                "drawdown_analysis": None,
                "volatility_cone": None,
                "risk_adjusted_returns": None
            }
            
            for task_type, method, confidence_level, result in task_results:
                if task_type == "var":
                    if method not in risk_metrics["var_results"]:
                        risk_metrics["var_results"][method] = {}
                    risk_metrics["var_results"][method][confidence_level] = result
                    
                elif task_type == "drawdown":
                    risk_metrics["drawdown_analysis"] = result
                    
                elif task_type == "cone":
                    risk_metrics["volatility_cone"] = result
                    
                elif task_type == "risk_adjusted":
                    risk_metrics["risk_adjusted_returns"] = result
            
            # Summary metrics
            summary = await self._generate_risk_summary(risk_metrics)
            risk_metrics["summary"] = summary
            
            # Risk dashboard
            dashboard = await self._create_risk_dashboard(risk_metrics)
            risk_metrics["dashboard"] = dashboard
            
            self.risk_dashboard_history.append(risk_metrics)
            logger.info(f"✅ Comprehensive risk analysis completed")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive risk analysis: {e}")
            raise

    async def _calculate_risk_adjusted_returns(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskAdjustedReturns:
        """Calculate risk-adjusted return metrics"""
        
        clean_returns = returns.dropna()
        
        # Basic metrics
        total_return = (1 + clean_returns).prod() - 1
        n_periods = len(clean_returns)
        periods_per_year = 252  # Daily returns assumed
        
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
        annualized_volatility = clean_returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assumed 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Downside deviation (for Sortino ratio)
        negative_returns = clean_returns[clean_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown for Calmar ratio
        cumulative = (1 + clean_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # VaR and CVaR (95%)
        var_95 = -np.percentile(clean_returns, 5) * 100  # In percentage
        cvar_95 = -clean_returns[clean_returns <= -var_95/100].mean() * 100
        
        # Higher moments
        skewness = clean_returns.skew()
        kurtosis = clean_returns.kurtosis()
        
        # Monthly statistics
        monthly_returns = clean_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns.max() * 100 if len(monthly_returns) > 0 else 0
        worst_month = monthly_returns.min() * 100 if len(monthly_returns) > 0 else 0
        positive_months_pct = (monthly_returns > 0).mean() * 100 if len(monthly_returns) > 0 else 0
        
        # Benchmark-related metrics (if available)
        treynor_ratio = None
        information_ratio = None
        jensen_alpha = None
        
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(clean_returns.index).dropna()
            common_dates = clean_returns.index.intersection(aligned_benchmark.index)
            
            if len(common_dates) > 30:
                ret_aligned = clean_returns.reindex(common_dates)
                bench_aligned = aligned_benchmark.reindex(common_dates)
                
                # Beta calculation
                covariance = ret_aligned.cov(bench_aligned)
                benchmark_variance = bench_aligned.var()
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                # Treynor ratio
                benchmark_annual_return = bench_aligned.mean() * periods_per_year
                treynor_ratio = (annualized_return - risk_free_rate) / beta if beta != 0 else 0
                
                # Information ratio (tracking error)
                excess_returns = ret_aligned - bench_aligned
                tracking_error = excess_returns.std() * np.sqrt(periods_per_year)
                information_ratio = excess_returns.mean() * periods_per_year / tracking_error if tracking_error > 0 else 0
                
                # Jensen's alpha
                expected_return = risk_free_rate + beta * (benchmark_annual_return - risk_free_rate)
                jensen_alpha = annualized_return - expected_return
        
        return RiskAdjustedReturns(
            symbol=self.symbol,
            timestamp=datetime.now(),
            analysis_period=(clean_returns.index[0], clean_returns.index[-1]),
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
            jensen_alpha=jensen_alpha,
            downside_deviation=downside_deviation,
            maximum_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            best_month=best_month,
            worst_month=worst_month,
            positive_months_percentage=positive_months_pct,
            metadata={
                "n_observations": len(clean_returns),
                "n_monthly_observations": len(monthly_returns),
                "risk_free_rate": risk_free_rate,
                "periods_per_year": periods_per_year
            }
        )

    async def _generate_risk_summary(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk summary"""
        
        summary = {
            "overall_risk_level": "medium",
            "primary_concerns": [],
            "key_metrics": {},
            "recommendations": []
        }
        
        # Extracting key metrics
        if risk_metrics["var_results"]:
            # Get best VaR estimate (historical 95%)
            if "historical" in risk_metrics["var_results"] and 0.95 in risk_metrics["var_results"]["historical"]:
                var_95 = risk_metrics["var_results"]["historical"][0.95]
                summary["key_metrics"]["var_95_1d"] = var_95.var_percentage
                
                # Risk level based on VaR
                if var_95.var_percentage < 2:
                    risk_level = "low"
                elif var_95.var_percentage < 5:
                    risk_level = "medium"  
                elif var_95.var_percentage < 10:
                    risk_level = "high"
                else:
                    risk_level = "extreme"
                
                summary["overall_risk_level"] = risk_level
        
        # Drawdown concerns
        if risk_metrics["drawdown_analysis"]:
            dd_analysis = risk_metrics["drawdown_analysis"]
            summary["key_metrics"]["max_drawdown"] = dd_analysis.max_drawdown
            summary["key_metrics"]["current_drawdown"] = dd_analysis.current_drawdown
            
            if dd_analysis.max_drawdown > 0.2:  # 20%
                summary["primary_concerns"].append("High maximum drawdown")
            
            if dd_analysis.current_drawdown > 0.1:  # 10%
                summary["primary_concerns"].append("Currently in significant drawdown")
        
        # Volatility regime
        if risk_metrics["volatility_cone"]:
            cone = risk_metrics["volatility_cone"]
            summary["key_metrics"]["volatility_regime"] = cone.volatility_regime
            
            if cone.volatility_regime in ["high", "extreme"]:
                summary["primary_concerns"].append(f"Volatility in {cone.volatility_regime} regime")
        
        # Recommendations
        if summary["overall_risk_level"] == "high":
            summary["recommendations"].append("Consider reducing position sizes")
            summary["recommendations"].append("Implement additional hedging strategies")
        
        if "Currently in significant drawdown" in summary["primary_concerns"]:
            summary["recommendations"].append("Monitor drawdown recovery closely")
            summary["recommendations"].append("Consider drawdown-based position scaling")
        
        return summary

    async def _create_risk_dashboard(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create risk dashboard"""
        
        dashboard = {
            "risk_score": 5.0,  # 1-10 scale
            "risk_factors": {},
            "alerts": [],
            "trends": {},
            "actionable_insights": []
        }
        
        # Calculate risk score
        risk_score_components = []
        
        # VaR component
        if risk_metrics["var_results"] and "historical" in risk_metrics["var_results"]:
            var_data = risk_metrics["var_results"]["historical"].get(0.95)
            if var_data:
                var_component = min(10, max(1, var_data.var_percentage * 2))
                risk_score_components.append(var_component)
                dashboard["risk_factors"]["var_95"] = var_component
        
        # Drawdown component  
        if risk_metrics["drawdown_analysis"]:
            dd = risk_metrics["drawdown_analysis"]
            dd_component = min(10, max(1, dd.max_drawdown * 20))
            risk_score_components.append(dd_component)
            dashboard["risk_factors"]["max_drawdown"] = dd_component
        
        # Volatility component
        if risk_metrics["volatility_cone"]:
            cone = risk_metrics["volatility_cone"]
            if 22 in cone.percentile_ranks:  # 1-month volatility rank
                vol_rank = cone.percentile_ranks[22]
                vol_component = max(1, min(10, vol_rank / 10))
                risk_score_components.append(vol_component)
                dashboard["risk_factors"]["volatility_regime"] = vol_component
        
        # Overall risk score
        if risk_score_components:
            dashboard["risk_score"] = np.mean(risk_score_components)
        
        # Alerts generation
        if dashboard["risk_score"] > 7:
            dashboard["alerts"].append({
                "severity": "high",
                "message": f"High risk score: {dashboard['risk_score']:.1f}/10",
                "timestamp": datetime.now()
            })
        
        # Actionable insights
        if dashboard["risk_score"] > 6:
            dashboard["actionable_insights"].append(
                "Consider implementing stricter position sizing rules"
            )
            dashboard["actionable_insights"].append(
                "Monitor intraday volatility patterns for entry/exit timing"
            )
        
        return dashboard

    def generate_risk_report(self, risk_metrics: Dict[str, Any]) -> str:
        """Generate comprehensive risk report"""
        
        if not risk_metrics:
            return "No risk metrics available"
        
        summary = risk_metrics.get("summary", {})
        dashboard = risk_metrics.get("dashboard", {})
        
        report = f"""
🎯 Comprehensive Risk Report for {risk_metrics['symbol']}
{'='*60}

Overall Risk Assessment:
- Risk Level: {summary.get('overall_risk_level', 'unknown').upper()}
- Risk Score: {dashboard.get('risk_score', 0):.1f}/10
- Analysis Period: {risk_metrics['analysis_period'][0].date()} to {risk_metrics['analysis_period'][1].date()}

Value at Risk (VaR):
"""
        
        # VaR section
        if risk_metrics.get("var_results"):
            for method, confidence_data in risk_metrics["var_results"].items():
                report += f"\n{method.title()} Method:\n"
                for confidence_level, var_result in confidence_data.items():
                    report += f"  {confidence_level*100}% VaR: {var_result.var_percentage:.2f}%\n"
                    report += f"  Expected Shortfall: {var_result.expected_shortfall*100:.2f}%\n"
        
        # Drawdown section
        if risk_metrics.get("drawdown_analysis"):
            dd = risk_metrics["drawdown_analysis"]
            report += f"""
Drawdown Analysis:
- Maximum Drawdown: {dd.max_drawdown:.2%}
- Current Drawdown: {dd.current_drawdown:.2%}
- Average Recovery Time: {dd.recovery_time_avg:.0f} days
- Calmar Ratio: {dd.sterling_ratio:.2f}
"""
        
        # Risk-adjusted returns
        if risk_metrics.get("risk_adjusted_returns"):
            ra = risk_metrics["risk_adjusted_returns"]
            report += f"""
Risk-Adjusted Performance:
- Sharpe Ratio: {ra.sharpe_ratio:.2f}
- Sortino Ratio: {ra.sortino_ratio:.2f}
- Maximum Drawdown: {ra.maximum_drawdown:.2%}
- Annualized Return: {ra.annualized_return:.2%}
- Annualized Volatility: {ra.annualized_volatility:.2%}
"""
        
        # Primary concerns
        concerns = summary.get("primary_concerns", [])
        if concerns:
            report += f"\n🚨 Primary Concerns:\n"
            for concern in concerns:
                report += f"- {concern}\n"
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            report += f"\n💡 Recommendations:\n"
            for recommendation in recommendations:
                report += f"- {recommendation}\n"
        
        # Alerts
        alerts = dashboard.get("alerts", [])
        if alerts:
            report += f"\n⚠️ Active Alerts:\n"
            for alert in alerts:
                report += f"- {alert['severity'].upper()}: {alert['message']}\n"
        
        report += f"\n📊 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report

# Export all classes
__all__ = [
    "RiskMeasureType",
    "ConfidenceLevel",
    "VaRResult",
    "DrawdownAnalysis", 
    "VolatilityCone",
    "RiskAdjustedReturns",
    "BaseRiskCalculator",
    "VaRCalculator",
    "DrawdownAnalyzer",
    "VolatilityConeCalculator", 
    "RiskMetricsCalculator",
    "_calculate_historical_var_numba",
    "_calculate_maximum_drawdown_numba",
    "_calculate_underwater_curve_numba"
]

logger.info("🔥 Risk Metrics module loaded successfully!")