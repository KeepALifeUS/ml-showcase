"""
Performance Metrics for DQN evaluation with .

Comprehensive metrics for evaluation DQN performance:
- Traditional RL metrics (reward, episode length)
- Financial metrics (Sharpe ratio, max drawdown, Calmar ratio)
- Risk-adjusted performance measures
- Statistical significance testing
- Benchmark comparisons
- Real-time monitoring integration
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import structlog
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = structlog.get_logger(__name__)


@dataclass
class EpisodeMetrics:
 """Metrics a single episode."""
 episode_id: int
 total_reward: float
 episode_length: int
 max_q_value: float
 avg_q_value: float
 actions_taken: List[int]
 timestamp: datetime = field(default_factory=datetime.now)

 def action_distribution(self) -> Dict[int, float]:
 """Distribution actions in episode."""
 if not self.actions_taken:
 return {}

 unique, counts = np.unique(self.actions_taken, return_counts=True)
 total = len(self.actions_taken)
 return {int(action): count / total for action, count in zip(unique, counts)}


@dataclass
class FinancialMetrics:
 """Financial metrics for trading agents."""
 total_return: float
 annualized_return: float
 sharpe_ratio: float
 sortino_ratio: float
 calmar_ratio: float
 max_drawdown: float
 volatility: float
 skewness: float
 kurtosis: float
 var_95: float # Value at Risk 95%
 cvar_95: float # Conditional VaR 95%

 @classmethod
 def from_returns(cls, returns: np.ndarray, risk_free_rate: float = 0.02) -> 'FinancialMetrics':
 """
 Computing financial metrics from array returns.

 Args:
 returns: Array returns
 risk_free_rate: Risk-free rate (annualized)

 Returns:
 FinancialMetrics object
 """
 if len(returns) == 0:
 return cls(**{key: 0.0 for key in cls.__dataclass_fields__.keys})

 # Basic statistics
 total_return = (1 + returns).prod - 1
 annualized_return = (1 + returns.mean) ** 252 - 1
 volatility = returns.std * np.sqrt(252)

 # Risk metrics
 daily_rf = risk_free_rate / 252
 excess_returns = returns - daily_rf

 # Sharpe ratio
 sharpe_ratio = 0.0
 if volatility > 0:
 sharpe_ratio = (annualized_return - risk_free_rate) / volatility

 # Sortino ratio (downside deviation)
 downside_returns = returns[returns < 0]
 sortino_ratio = 0.0
 if len(downside_returns) > 0:
 downside_std = downside_returns.std * np.sqrt(252)
 if downside_std > 0:
 sortino_ratio = (annualized_return - risk_free_rate) / downside_std

 # Maximum drawdown
 cumulative = (1 + returns).cumprod
 running_max = cumulative.expanding.max
 drawdowns = (cumulative - running_max) / running_max
 max_drawdown = abs(drawdowns.min)

 # Calmar ratio
 calmar_ratio = 0.0
 if max_drawdown > 0:
 calmar_ratio = annualized_return / max_drawdown

 # Higher moments
 skewness = stats.skew(returns) if len(returns) > 2 else 0.0
 kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0.0

 # Value at Risk
 var_95 = np.percentile(returns, 5)
 cvar_95 = returns[returns <= var_95].mean if np.any(returns <= var_95) else var_95

 return cls(
 total_return=total_return,
 annualized_return=annualized_return,
 sharpe_ratio=sharpe_ratio,
 sortino_ratio=sortino_ratio,
 calmar_ratio=calmar_ratio,
 max_drawdown=max_drawdown,
 volatility=volatility,
 skewness=skewness,
 kurtosis=kurtosis,
 var_95=var_95,
 cvar_95=cvar_95
 )


class PerformanceMetricsConfig(BaseModel):
 """Configuration performance metrics."""

 # Window sizes for rolling metrics
 short_window: int = Field(default=50, description="Short window for rolling metrics", gt=0)
 long_window: int = Field(default=200, description="Long window for rolling metrics", gt=0)

 # Statistical significance
 confidence_level: float = Field(default=0.95, description="Confidence level", ge=0.9, le=0.99)
 min_samples: int = Field(default=30, description="Minimum samples for statistics", gt=0)

 # Financial metrics
 risk_free_rate: float = Field(default=0.02, description="Risk-free rate (annualized)", ge=0, le=0.1)
 trading_days: int = Field(default=252, description="Trading days in year", gt=0)

 # Benchmarking
 benchmark_returns: Optional[List[float]] = Field(default=None, description="Benchmark returns")

 @validator("long_window")
 def validate_long_window(cls, v, values):
 if "short_window" in values and v <= values["short_window"]:
 raise ValueError("long_window must be more short_window")
 return v


class PerformanceMetrics:
 """
 Comprehensive Performance Metrics System with enterprise functionality.

 Features:
 - Traditional RL metrics (reward, success rate, convergence)
 - Financial performance metrics (Sharpe, Sortino, Calmar)
 - Risk analytics (VaR, CVaR, maximum drawdown)
 - Statistical significance testing
 - Rolling window analysis
 - Benchmark comparisons
 - Real-time monitoring support
 """

 def __init__(self, config: Optional[PerformanceMetricsConfig] = None):
 """
 Initialization Performance Metrics.

 Args:
 config: Configuration metrics
 """
 if config is None:
 config = PerformanceMetricsConfig

 self.config = config

 # Episode data storage
 self.episodes: List[EpisodeMetrics] = []
 self.returns: List[float] = []

 # Rolling statistics
 self.rolling_rewards = []
 self.rolling_lengths = []

 # Financial tracking
 self.portfolio_values = []
 self.portfolio_returns = []

 # Benchmark comparison
 self.benchmark_data = config.benchmark_returns or []

 self.logger = structlog.get_logger(__name__).bind(component="PerformanceMetrics")
 self.logger.info("Performance Metrics initialized")

 def add_episode(self,
 episode_id: int,
 total_reward: float,
 episode_length: int,
 max_q_value: float = 0.0,
 avg_q_value: float = 0.0,
 actions_taken: Optional[List[int]] = None,
 portfolio_value: Optional[float] = None) -> None:
 """
 Adding metrics episode.

 Args:
 episode_id: ID episode
 total_reward: Total reward
 episode_length: Length episode
 max_q_value: Maximum Q-value
 avg_q_value: Average Q-value
 actions_taken: List actions
 portfolio_value: Value portfolio (for trading)
 """
 episode_metrics = EpisodeMetrics(
 episode_id=episode_id,
 total_reward=total_reward,
 episode_length=episode_length,
 max_q_value=max_q_value,
 avg_q_value=avg_q_value,
 actions_taken=actions_taken or []
 )

 self.episodes.append(episode_metrics)

 # Update rolling statistics
 self.rolling_rewards.append(total_reward)
 self.rolling_lengths.append(episode_length)

 # Financial tracking
 if portfolio_value is not None:
 self.portfolio_values.append(portfolio_value)

 if len(self.portfolio_values) > 1:
 prev_value = self.portfolio_values[-2]
 return_value = (portfolio_value - prev_value) / prev_value
 self.portfolio_returns.append(return_value)
 self.returns.append(return_value)
 else:
 # Use reward as return proxy
 self.returns.append(total_reward)

 # Limit memory usage
 max_history = max(self.config.long_window * 2, 1000)
 if len(self.episodes) > max_history:
 self.episodes = self.episodes[-max_history:]
 self.rolling_rewards = self.rolling_rewards[-max_history:]
 self.rolling_lengths = self.rolling_lengths[-max_history:]

 def get_basic_metrics(self) -> Dict[str, float]:
 """Get base RL metrics."""
 if not self.episodes:
 return {}

 rewards = [ep.total_reward for ep in self.episodes]
 lengths = [ep.episode_length for ep in self.episodes]

 metrics = {
 "total_episodes": len(self.episodes),
 "mean_reward": np.mean(rewards),
 "std_reward": np.std(rewards),
 "min_reward": np.min(rewards),
 "max_reward": np.max(rewards),
 "median_reward": np.median(rewards),

 "mean_length": np.mean(lengths),
 "std_length": np.std(lengths),
 "min_length": np.min(lengths),
 "max_length": np.max(lengths),
 }

 # Success rate (episodes with positive reward)
 successful_episodes = sum(1 for reward in rewards if reward > 0)
 metrics["success_rate"] = successful_episodes / len(rewards)

 return metrics

 def get_rolling_metrics(self, window: Optional[int] = None) -> Dict[str, float]:
 """
 Get rolling window metrics.

 Args:
 window: Size window (by default short_window)

 Returns:
 Rolling metrics
 """
 if window is None:
 window = self.config.short_window

 if len(self.rolling_rewards) < window:
 return {"insufficient_data": True, "available_episodes": len(self.rolling_rewards)}

 recent_rewards = self.rolling_rewards[-window:]
 recent_lengths = self.rolling_lengths[-window:]

 return {
 "rolling_mean_reward": np.mean(recent_rewards),
 "rolling_std_reward": np.std(recent_rewards),
 "rolling_mean_length": np.mean(recent_lengths),
 "rolling_trend": self._calculate_trend(recent_rewards),
 "rolling_success_rate": sum(1 for r in recent_rewards if r > 0) / len(recent_rewards),
 "window_size": window,
 }

 def get_financial_metrics(self) -> Dict[str, Any]:
 """Get financial metrics."""
 if len(self.returns) < self.config.min_samples:
 return {"insufficient_data": True, "available_samples": len(self.returns)}

 returns_array = np.array(self.returns)
 financial_metrics = FinancialMetrics.from_returns(returns_array, self.config.risk_free_rate)

 return {
 "total_return": financial_metrics.total_return,
 "annualized_return": financial_metrics.annualized_return,
 "sharpe_ratio": financial_metrics.sharpe_ratio,
 "sortino_ratio": financial_metrics.sortino_ratio,
 "calmar_ratio": financial_metrics.calmar_ratio,
 "max_drawdown": financial_metrics.max_drawdown,
 "volatility": financial_metrics.volatility,
 "skewness": financial_metrics.skewness,
 "kurtosis": financial_metrics.kurtosis,
 "var_95": financial_metrics.var_95,
 "cvar_95": financial_metrics.cvar_95,
 "num_samples": len(returns_array),
 }

 def get_convergence_metrics(self) -> Dict[str, Any]:
 """Analysis convergence training."""
 if len(self.rolling_rewards) < self.config.long_window:
 return {"insufficient_data": True}

 rewards = np.array(self.rolling_rewards)

 # Trend analysis
 x = np.arange(len(rewards))
 slope, intercept, r_value, p_value, std_err = stats.linregress(x, rewards)

 # Stability metrics
 recent_std = np.std(rewards[-self.config.short_window:])
 early_std = np.std(rewards[:self.config.short_window])
 stability_ratio = recent_std / early_std if early_std > 0 else 0

 # Convergence detection
 recent_mean = np.mean(rewards[-self.config.short_window:])
 earlier_mean = np.mean(rewards[-self.config.long_window:-self.config.short_window])
 convergence_test = abs(recent_mean - earlier_mean) / np.std(rewards[-self.config.long_window:])

 return {
 "trend_slope": slope,
 "trend_r_squared": r_value ** 2,
 "trend_p_value": p_value,
 "stability_ratio": stability_ratio,
 "convergence_metric": convergence_test,
 "is_converged": convergence_test < 0.1, # Threshold for convergence
 "recent_improvement": recent_mean > earlier_mean,
 }

 def compare_to_benchmark(self) -> Dict[str, Any]:
 """Comparison with benchmark."""
 if not self.benchmark_data or len(self.returns) != len(self.benchmark_data):
 return {"benchmark_comparison": "unavailable"}

 agent_returns = np.array(self.returns)
 benchmark_returns = np.array(self.benchmark_data)

 # Excess returns
 excess_returns = agent_returns - benchmark_returns

 # Information ratio
 info_ratio = 0.0
 if np.std(excess_returns) > 0:
 info_ratio = np.mean(excess_returns) / np.std(excess_returns)

 # Beta calculation
 cov_matrix = np.cov(agent_returns, benchmark_returns)
 beta = cov_matrix[0, 1] / np.var(benchmark_returns) if np.var(benchmark_returns) > 0 else 0

 # Alpha (Jensen's alpha)
 alpha = np.mean(agent_returns) - beta * np.mean(benchmark_returns)

 # Tracking error
 tracking_error = np.std(excess_returns) * np.sqrt(self.config.trading_days)

 return {
 "information_ratio": info_ratio,
 "beta": beta,
 "alpha": alpha,
 "tracking_error": tracking_error,
 "excess_return": np.mean(excess_returns),
 "outperformance_rate": np.mean(agent_returns > benchmark_returns),
 }

 def get_q_value_analysis(self) -> Dict[str, Any]:
 """Analysis Q-values."""
 if not self.episodes:
 return {}

 episodes_with_q = [ep for ep in self.episodes if ep.max_q_value != 0 or ep.avg_q_value != 0]

 if not episodes_with_q:
 return {"q_values_available": False}

 max_q_values = [ep.max_q_value for ep in episodes_with_q]
 avg_q_values = [ep.avg_q_value for ep in episodes_with_q]

 return {
 "q_values_available": True,
 "max_q_mean": np.mean(max_q_values),
 "max_q_std": np.std(max_q_values),
 "max_q_trend": self._calculate_trend(max_q_values),

 "avg_q_mean": np.mean(avg_q_values),
 "avg_q_std": np.std(avg_q_values),
 "avg_q_trend": self._calculate_trend(avg_q_values),

 "q_value_stability": np.std(max_q_values) / np.mean(max_q_values) if np.mean(max_q_values) > 0 else 0,
 }

 def get_action_analysis(self) -> Dict[str, Any]:
 """Analysis distribution actions."""
 if not self.episodes:
 return {}

 all_actions = []
 for episode in self.episodes:
 all_actions.extend(episode.actions_taken)

 if not all_actions:
 return {"actions_available": False}

 # Overall action distribution
 unique_actions, counts = np.unique(all_actions, return_counts=True)
 action_dist = {int(action): count / len(all_actions) for action, count in zip(unique_actions, counts)}

 # Action diversity (entropy)
 probabilities = counts / len(all_actions)
 entropy = -np.sum(probabilities * np.log2(probabilities + 1e-8))

 # Recent vs early action distribution comparison
 if len(self.episodes) >= 2 * self.config.short_window:
 recent_actions = []
 early_actions = []

 for episode in self.episodes[-self.config.short_window:]:
 recent_actions.extend(episode.actions_taken)

 for episode in self.episodes[:self.config.short_window]:
 early_actions.extend(episode.actions_taken)

 # KL divergence between distributions
 recent_dist = np.bincount(recent_actions, minlength=len(unique_actions))
 early_dist = np.bincount(early_actions, minlength=len(unique_actions))

 recent_dist = recent_dist / np.sum(recent_dist)
 early_dist = early_dist / np.sum(early_dist)

 kl_divergence = stats.entropy(recent_dist, early_dist)
 else:
 kl_divergence = 0.0

 return {
 "actions_available": True,
 "action_distribution": action_dist,
 "action_entropy": entropy,
 "action_diversity": len(unique_actions),
 "exploration_change": kl_divergence,
 }

 def _calculate_trend(self, data: List[float]) -> float:
 """Computing trend in data."""
 if len(data) < 2:
 return 0.0

 x = np.arange(len(data))
 slope, _, _, _, _ = stats.linregress(x, data)
 return slope

 def get_statistical_significance(self) -> Dict[str, Any]:
 """Statistical importance results."""
 if len(self.returns) < self.config.min_samples:
 return {"sufficient_data": False}

 returns = np.array(self.returns)

 # T-test against zero (zero hypothesis: mean return = 0)
 t_stat, p_value = stats.ttest_1samp(returns, 0)

 # Confidence interval for mean return
 confidence_interval = stats.t.interval(
 self.config.confidence_level,
 len(returns) - 1,
 loc=np.mean(returns),
 scale=stats.sem(returns)
 )

 # Normality test
 normality_stat, normality_p = stats.normaltest(returns)

 return {
 "sufficient_data": True,
 "sample_size": len(returns),
 "t_statistic": t_stat,
 "p_value": p_value,
 "is_significant": p_value < (1 - self.config.confidence_level),
 "confidence_interval": confidence_interval,
 "mean_return": np.mean(returns),
 "standard_error": stats.sem(returns),
 "normality_p_value": normality_p,
 "is_normal": normality_p > 0.05,
 }

 def generate_report(self) -> Dict[str, Any]:
 """Generation full report by performance."""
 report = {
 "generated_at": datetime.now.isoformat,
 "basic_metrics": self.get_basic_metrics,
 "rolling_metrics": self.get_rolling_metrics,
 "financial_metrics": self.get_financial_metrics,
 "convergence_analysis": self.get_convergence_metrics,
 "benchmark_comparison": self.compare_to_benchmark,
 "q_value_analysis": self.get_q_value_analysis,
 "action_analysis": self.get_action_analysis,
 "statistical_significance": self.get_statistical_significance,
 }

 self.logger.info("Performance report generated",
 episodes=len(self.episodes),
 returns_samples=len(self.returns))

 return report

 def clear_history(self) -> None:
 """Cleanup history data."""
 self.episodes.clear
 self.returns.clear
 self.rolling_rewards.clear
 self.rolling_lengths.clear
 self.portfolio_values.clear
 self.portfolio_returns.clear

 self.logger.info("Performance metrics history cleared")

 def export_to_dataframe(self) -> pd.DataFrame:
 """Export data in pandas DataFrame."""
 if not self.episodes:
 return pd.DataFrame

 data = []
 for ep in self.episodes:
 row = {
 "episode_id": ep.episode_id,
 "total_reward": ep.total_reward,
 "episode_length": ep.episode_length,
 "max_q_value": ep.max_q_value,
 "avg_q_value": ep.avg_q_value,
 "timestamp": ep.timestamp,
 "num_actions": len(ep.actions_taken),
 }

 # Action distribution
 action_dist = ep.action_distribution
 for action, freq in action_dist.items:
 row[f"action_{action}_freq"] = freq

 data.append(row)

 df = pd.DataFrame(data)

 # Add returns if available
 if len(self.returns) == len(self.episodes):
 df["return"] = self.returns

 return df