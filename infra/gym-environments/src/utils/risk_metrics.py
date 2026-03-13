"""
Risk Metrics Calculation Utilities
enterprise patterns for comprehensive risk analysis

Advanced risk calculation functions:
- Sharpe, Sortino, Calmar ratios
- VaR and Expected Shortfall
- Maximum Drawdown calculation
- Volatility metrics
- Position sizing algorithms
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings
from scipy import stats
import logging


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics container"""
    
    # Return-based metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Risk measures
    volatility: float = 0.0
    downside_volatility: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    tail_ratio: float = 0.0
    
    # Tracking metrics
    tracking_error: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Period information
    periods: int = 0
    annualization_factor: float = 252.0


class RiskCalculator:
    """
    Advanced risk metrics calculator
    
    Calculates comprehensive risk metrics for trading strategies
    """
    
    def __init__(self, annualization_factor: float = 252.0):
        self.annualization_factor = annualization_factor
        self.logger = logging.getLogger(__name__)
    
    def calculate_risk_metrics(
        self,
        returns: Union[List[float], np.ndarray],
        benchmark_returns: Optional[Union[List[float], np.ndarray]] = None,
        risk_free_rate: float = 0.02
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        returns = np.array(returns)
        
        if len(returns) < 2:
            self.logger.warning("Insufficient data for risk calculations")
            return RiskMetrics()
        
        # Basic metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns) * np.sqrt(self.annualization_factor)
        
        # Drawdown metrics
        max_dd, current_dd = self._calculate_drawdown_metrics(returns)
        
        # Return-based ratios
        sharpe = self._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = self._calculate_sortino_ratio(returns, risk_free_rate)
        calmar = self._calculate_calmar_ratio(returns)
        
        # VaR and Expected Shortfall
        var_95 = self._calculate_var(returns, 0.05)
        var_99 = self._calculate_var(returns, 0.01)
        es_95 = self._calculate_expected_shortfall(returns, 0.05)
        es_99 = self._calculate_expected_shortfall(returns, 0.01)
        
        # Distribution metrics
        skewness = stats.skew(returns) if len(returns) > 2 else 0.0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0.0
        tail_ratio = self._calculate_tail_ratio(returns)
        
        # Downside volatility
        downside_vol = self._calculate_downside_volatility(returns, risk_free_rate)
        
        # Benchmark-relative metrics
        information_ratio = 0.0
        tracking_error = 0.0
        beta = 0.0
        alpha = 0.0
        
        if benchmark_returns is not None:
            benchmark_returns = np.array(benchmark_returns)
            if len(benchmark_returns) == len(returns):
                information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
                tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                beta, alpha = self._calculate_beta_alpha(returns, benchmark_returns, risk_free_rate)
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=information_ratio,
            volatility=volatility,
            downside_volatility=downside_vol,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            tracking_error=tracking_error,
            beta=beta,
            alpha=alpha,
            periods=len(returns),
            annualization_factor=self.annualization_factor
        )
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / self.annualization_factor
        
        if np.std(returns) == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / np.std(returns) * np.sqrt(self.annualization_factor))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target_return: float) -> float:
        """Calculate Sortino ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - target_return / self.annualization_factor
        downside_returns = np.where(excess_returns < 0, excess_returns, 0)
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        
        return float(np.mean(excess_returns) / downside_std * np.sqrt(self.annualization_factor))
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * self.annualization_factor
        max_drawdown, _ = self._calculate_drawdown_metrics(returns)
        
        if max_drawdown == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return float(annual_return / max_drawdown)
    
    def _calculate_information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate Information ratio"""
        
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0
        
        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        return float(np.mean(excess_returns) / tracking_error * np.sqrt(self.annualization_factor))
    
    def _calculate_tracking_error(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate tracking error"""
        
        if len(returns) != len(benchmark_returns):
            return 0.0
        
        excess_returns = returns - benchmark_returns
        return float(np.std(excess_returns) * np.sqrt(self.annualization_factor))
    
    def _calculate_beta_alpha(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        risk_free_rate: float
    ) -> Tuple[float, float]:
        """Calculate beta and alpha"""
        
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            return 0.0, 0.0
        
        # Excess returns
        portfolio_excess = returns - risk_free_rate / self.annualization_factor
        benchmark_excess = benchmark_returns - risk_free_rate / self.annualization_factor
        
        # Beta calculation
        if np.var(benchmark_excess) == 0:
            beta = 0.0
        else:
            beta = np.cov(portfolio_excess, benchmark_excess)[0, 1] / np.var(benchmark_excess)
        
        # Alpha calculation
        alpha = np.mean(portfolio_excess) - beta * np.mean(benchmark_excess)
        alpha *= self.annualization_factor  # Annualize
        
        return float(beta), float(alpha)
    
    def _calculate_drawdown_metrics(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate maximum and current drawdown"""
        
        if len(returns) < 2:
            return 0.0, 0.0
        
        # Calculate cumulative returns
        cumulative = np.cumprod(1 + returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdowns
        drawdowns = (cumulative - running_max) / running_max
        
        max_drawdown = float(np.min(drawdowns))
        current_drawdown = float(drawdowns[-1])
        
        return abs(max_drawdown), abs(current_drawdown)
    
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        
        if len(returns) < 10:
            return 0.0
        
        return float(-np.percentile(returns, confidence_level * 100))
    
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        
        if len(returns) < 10:
            return 0.0
        
        var = self._calculate_var(returns, confidence_level)
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        return float(-np.mean(tail_returns))
    
    def _calculate_downside_volatility(self, returns: np.ndarray, target_return: float) -> float:
        """Calculate downside volatility"""
        
        if len(returns) < 2:
            return 0.0
        
        target_daily = target_return / self.annualization_factor
        downside_returns = returns[returns < target_daily] - target_daily
        
        if len(downside_returns) == 0:
            return 0.0
        
        return float(np.std(downside_returns) * np.sqrt(self.annualization_factor))
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        
        if len(returns) < 20:
            return 0.0
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 == 0:
            return float('inf') if p95 > 0 else 0.0
        
        return float(abs(p95 / p5))


class PositionSizer:
    """
    Advanced position sizing algorithms
    """
    
    def __init__(self, method: str = "kelly", max_risk: float = 0.02):
        self.method = method
        self.max_risk = max_risk
        self.logger = logging.getLogger(__name__)
    
    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """Calculate optimal position size"""
        
        if self.method == "kelly":
            return self._kelly_position_size(
                signal_strength, current_price, portfolio_value,
                win_rate, avg_win, avg_loss
            )
        elif self.method == "fixed_fraction":
            return self._fixed_fraction_position_size(
                signal_strength, current_price, portfolio_value
            )
        elif self.method == "volatility":
            return self._volatility_position_size(
                signal_strength, current_price, portfolio_value, volatility
            )
        else:
            return signal_strength * self.max_risk * portfolio_value / current_price
    
    def _kelly_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> float:
        """Kelly criterion position sizing"""
        
        # Use defaults if parameters not provided
        win_rate = win_rate or 0.5
        avg_win = avg_win or 0.02
        avg_loss = avg_loss or 0.02
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply signal strength and risk limits
        kelly_fraction *= abs(signal_strength)
        kelly_fraction = np.clip(kelly_fraction, 0, self.max_risk * 2)  # Max 2x normal risk
        
        position_value = kelly_fraction * portfolio_value
        return position_value / current_price if current_price > 0 else 0.0
    
    def _fixed_fraction_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float
    ) -> float:
        """Fixed fraction position sizing"""
        
        fraction = self.max_risk * abs(signal_strength)
        position_value = fraction * portfolio_value
        return position_value / current_price if current_price > 0 else 0.0
    
    def _volatility_position_size(
        self,
        signal_strength: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None
    ) -> float:
        """Volatility-based position sizing"""
        
        volatility = volatility or 0.02  # Default 2% daily volatility
        
        # Inverse volatility sizing
        vol_adjusted_risk = self.max_risk / volatility if volatility > 0 else self.max_risk
        
        fraction = vol_adjusted_risk * abs(signal_strength)
        fraction = min(fraction, self.max_risk * 3)  # Cap at 3x normal risk
        
        position_value = fraction * portfolio_value
        return position_value / current_price if current_price > 0 else 0.0


# Convenience functions
def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    calculator = RiskCalculator()
    return calculator._calculate_sharpe_ratio(np.array(returns), risk_free_rate)


def calculate_sortino_ratio(returns: Union[List[float], np.ndarray], target_return: float = 0.0) -> float:
    """Calculate Sortino ratio"""
    calculator = RiskCalculator()
    return calculator._calculate_sortino_ratio(np.array(returns), target_return)


def calculate_max_drawdown(returns: Union[List[float], np.ndarray]) -> float:
    """Calculate maximum drawdown"""
    calculator = RiskCalculator()
    max_dd, _ = calculator._calculate_drawdown_metrics(np.array(returns))
    return max_dd


def calculate_information_ratio(
    returns: Union[List[float], np.ndarray],
    benchmark_returns: Union[List[float], np.ndarray]
) -> float:
    """Calculate Information ratio"""
    calculator = RiskCalculator()
    return calculator._calculate_information_ratio(np.array(returns), np.array(benchmark_returns))


def calculate_calmar_ratio(returns: Union[List[float], np.ndarray]) -> float:
    """Calculate Calmar ratio"""
    calculator = RiskCalculator()
    return calculator._calculate_calmar_ratio(np.array(returns))


__all__ = [
    "RiskMetrics",
    "RiskCalculator", 
    "PositionSizer",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_information_ratio",
    "calculate_calmar_ratio"
]