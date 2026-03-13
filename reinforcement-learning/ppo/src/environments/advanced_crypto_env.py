"""
Advanced Crypto Trading Environment for PPO
Production-ready trading environment
Supports multi-asset trading, realistic market dynamics, and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import gym
from gym import spaces
import logging
from collections import deque, defaultdict
import json
from datetime import datetime, timedelta

# ====================================================================
# CONFIGURATION & TYPES
# ====================================================================

class MarketRegime(Enum):
 """Market regime types"""
 BULL = "bull"
 BEAR = "bear"
 SIDEWAYS = "sideways"
 VOLATILE = "volatile"
 CRASH = "crash"

class OrderType(Enum):
 """Order types"""
 MARKET = "market"
 LIMIT = "limit"
 STOP_LOSS = "stop_loss"
 TAKE_PROFIT = "take_profit"
 TRAILING_STOP = "trailing_stop"

@dataclass
class TradingEnvironmentConfig:
 """Trading environment configuration"""

 # Asset configuration
 assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "BNB", "SOL", "ADA"])
 base_currency: str = "USDT"
 initial_balance: float = 10000.0

 # Market simulation
 use_real_data: bool = True
 data_source: str = "binance" # binance, coinbase, kraken
 lookback_window: int = 100
 forecast_horizon: int = 10

 # Trading parameters
 max_position_size: float = 0.3 # 30% of portfolio
 min_trade_amount: float = 10.0
 leverage_enabled: bool = True
 max_leverage: float = 3.0

 # Fees and slippage
 maker_fee: float = 0.001 # 0.1%
 taker_fee: float = 0.002 # 0.2%
 slippage_model: str = "linear" # linear, square_root, logarithmic
 max_slippage: float = 0.005 # 0.5%

 # Risk management
 max_drawdown: float = 0.2 # 20%
 position_limit: int = 10
 risk_per_trade: float = 0.02 # 2%
 use_stop_loss: bool = True
 default_stop_loss: float = 0.05 # 5%

 # Reward shaping
 reward_type: str = "sharpe" # returns, sharpe, sortino, calmar
 risk_penalty: float = 0.1
 transaction_penalty: float = 0.001
 holding_penalty: float = 0.0001

 # Market dynamics
 simulate_volatility: bool = True
 volatility_clustering: bool = True
 simulate_liquidity: bool = True
 simulate_market_impact: bool = True

 # Environment parameters
 max_steps: int = 1000
 render_mode: Optional[str] = None
 seed: Optional[int] = None

# ====================================================================
# MARKET SIMULATOR
# ====================================================================

class MarketSimulator:
 """
 Simulates realistic market dynamics
 High-fidelity market simulation
 """

 def __init__(self, config: TradingEnvironmentConfig):
 self.config = config
 self.current_regime = MarketRegime.SIDEWAYS
 self.volatility_state = 1.0
 self.liquidity_state = 1.0

 # Historical patterns
 self.price_history = defaultdict(deque)
 self.volume_history = defaultdict(deque)
 self.volatility_history = defaultdict(deque)

 # Market microstructure
 self.order_book_imbalance = defaultdict(float)
 self.bid_ask_spread = defaultdict(float)

 def update_market_regime(self, prices: Dict[str, np.ndarray]) -> MarketRegime:
 """Detect and update market regime"""
 # Simple regime detection based on price trends and volatility
 returns = {asset: np.diff(np.log(prices[asset])) for asset in prices}

 avg_return = np.mean([np.mean(r) for r in returns.values()])
 avg_volatility = np.mean([np.std(r) for r in returns.values()])

 if avg_return > 0.02 and avg_volatility < 0.03:
 self.current_regime = MarketRegime.BULL
 elif avg_return < -0.02 and avg_volatility < 0.03:
 self.current_regime = MarketRegime.BEAR
 elif abs(avg_return) < 0.01 and avg_volatility < 0.02:
 self.current_regime = MarketRegime.SIDEWAYS
 elif avg_volatility > 0.05:
 self.current_regime = MarketRegime.VOLATILE
 elif avg_return < -0.05 and avg_volatility > 0.04:
 self.current_regime = MarketRegime.CRASH

 return self.current_regime

 def simulate_volatility_clustering(self, current_vol: float) -> float:
 """Simulate GARCH-like volatility clustering"""
 if not self.config.volatility_clustering:
 return current_vol

 # Simple GARCH(1,1) approximation
 omega = 0.00001
 alpha = 0.1
 beta = 0.85

 shock = np.random.randn() * np.sqrt(current_vol)
 new_vol = omega + alpha * shock**2 + beta * current_vol

 return np.clip(new_vol, 0.0001, 0.1)

 def calculate_market_impact(self, asset: str, volume: float, is_buy: bool) -> float:
 """Calculate market impact of large orders"""
 if not self.config.simulate_market_impact:
 return 0.0

 # Square-root market impact model
 avg_volume = np.mean(self.volume_history[asset]) if self.volume_history[asset] else 1000000
 volume_ratio = volume / avg_volume

 impact = np.sqrt(volume_ratio) * 0.001 # 0.1% impact for 1% of daily volume

 return impact if is_buy else -impact

 def calculate_slippage(self, size: float, liquidity: float = 1.0) -> float:
 """Calculate slippage based on order size and liquidity"""
 if self.config.slippage_model == "linear":
 slippage = size * 0.0001 / liquidity
 elif self.config.slippage_model == "square_root":
 slippage = np.sqrt(size) * 0.00001 / liquidity
 else: # logarithmic
 slippage = np.log1p(size) * 0.00001 / liquidity

 return min(slippage, self.config.max_slippage)

# ====================================================================
# PORTFOLIO MANAGER
# ====================================================================

class PortfolioManager:
 """
 Manages portfolio positions and risk
 Professional portfolio management
 """

 def __init__(self, config: TradingEnvironmentConfig):
 self.config = config
 self.positions = {asset: 0.0 for asset in config.assets}
 self.cash = config.initial_balance
 self.initial_balance = config.initial_balance

 # Position tracking
 self.entry_prices = {}
 self.position_times = {}
 self.realized_pnl = 0.0
 self.unrealized_pnl = 0.0

 # Risk metrics
 self.max_portfolio_value = config.initial_balance
 self.current_drawdown = 0.0
 self.var_95 = 0.0
 self.cvar_95 = 0.0

 # Order management
 self.pending_orders = []
 self.executed_orders = []
 self.stop_losses = {}
 self.take_profits = {}

 def execute_trade(
 self,
 asset: str,
 amount: float,
 price: float,
 order_type: OrderType = OrderType.MARKET
 ) -> Dict[str, Any]:
 """Execute a trade with proper risk checks"""
 # Risk checks
 if not self._check_risk_limits(asset, amount, price):
 return {"status": "rejected", "reason": "risk_limit_exceeded"}

 # Calculate costs
 is_buy = amount > 0
 abs_amount = abs(amount)
 fee = abs_amount * price * (self.config.maker_fee if order_type == OrderType.LIMIT else self.config.taker_fee)

 # Check sufficient funds
 if is_buy:
 required_cash = abs_amount * price + fee
 if required_cash > self.cash:
 return {"status": "rejected", "reason": "insufficient_funds"}
 else:
 if abs_amount > self.positions.get(asset, 0):
 return {"status": "rejected", "reason": "insufficient_position"}

 # Execute trade
 if is_buy:
 self.positions[asset] = self.positions.get(asset, 0) + abs_amount
 self.cash -= abs_amount * price + fee
 self.entry_prices[asset] = price
 else:
 # Calculate realized PnL
 entry_price = self.entry_prices.get(asset, price)
 pnl = abs_amount * (price - entry_price) - fee
 self.realized_pnl += pnl

 self.positions[asset] -= abs_amount
 self.cash += abs_amount * price - fee

 if self.positions[asset] == 0:
 del self.entry_prices[asset]

 # Update position time
 self.position_times[asset] = 0

 return {
 "status": "executed",
 "asset": asset,
 "amount": amount,
 "price": price,
 "fee": fee,
 "timestamp": datetime.now()
 }

 def update_portfolio_value(self, prices: Dict[str, float]) -> float:
 """Update and return current portfolio value"""
 portfolio_value = self.cash

 for asset, position in self.positions.items():
 if position > 0 and asset in prices:
 portfolio_value += position * prices[asset]

 # Update unrealized PnL
 if asset in self.entry_prices:
 self.unrealized_pnl = position * (prices[asset] - self.entry_prices[asset])

 # Update drawdown
 self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
 self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value

 return portfolio_value

 def _check_risk_limits(self, asset: str, amount: float, price: float) -> bool:
 """Check if trade violates risk limits"""
 # Check position limit
 if len([p for p in self.positions.values() if p > 0]) >= self.config.position_limit:
 return False

 # Check max position size
 position_value = abs(amount) * price
 portfolio_value = self.update_portfolio_value({asset: price})

 if position_value > portfolio_value * self.config.max_position_size:
 return False

 # Check max drawdown
 if self.current_drawdown >= self.config.max_drawdown:
 return False

 return True

 def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
 """Calculate portfolio risk metrics"""
 if len(returns) < 2:
 return {}

 metrics = {
 "volatility": np.std(returns),
 "downside_deviation": np.std(returns[returns < 0]) if any(returns < 0) else 0,
 "max_drawdown": self.current_drawdown,
 "var_95": np.percentile(returns, 5),
 "cvar_95": np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns) > 20 else 0,
 "sharpe_ratio": np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
 "sortino_ratio": np.mean(returns) / np.std(returns[returns < 0]) if any(returns < 0) and np.std(returns[returns < 0]) > 0 else 0
 }

 return metrics

# ====================================================================
# ADVANCED TRADING ENVIRONMENT
# ====================================================================

class AdvancedCryptoTradingEnv(gym.Env):
 """
 Advanced Crypto Trading Environment for PPO
 Production-ready gym environment
 """

 metadata = {'render.modes': ['human', 'ansi']}

 def __init__(self, config: TradingEnvironmentConfig):
 super().__init__()
 self.config = config
 self.logger = logging.getLogger(__name__)

 # Market components
 self.market_sim = MarketSimulator(config)
 self.portfolio = PortfolioManager(config)

 # Environment state
 self.current_step = 0
 self.episode_returns = []
 self.price_data = {}
 self.volume_data = {}

 # Observation and action spaces
 self._setup_spaces()

 # Load or generate market data
 self._initialize_market_data()

 def _setup_spaces(self):
 """Setup observation and action spaces"""
 # Observation space: [prices, volumes, positions, indicators, portfolio_stats]
 obs_dim = (
 len(self.config.assets) * 10 + # Price features per asset
 len(self.config.assets) * 5 + # Volume features
 len(self.config.assets) + # Current positions
 10 # Portfolio statistics
 )

 self.observation_space = spaces.Box(
 low=-np.inf,
 high=np.inf,
 shape=(obs_dim,),
 dtype=np.float32
 )

 # Action space: [action_type, asset_index, amount]
 # Discrete actions: hold, buy, sell for each asset
 self.action_space = spaces.Discrete(len(self.config.assets) * 3)

 def _initialize_market_data(self):
 """Initialize or load market data"""
 if self.config.use_real_data:
 # Load real market data
 # This would connect to actual data source
 pass
 else:
 # Generate synthetic data for testing
 for asset in self.config.assets:
 # Generate realistic price series using GBM
 returns = np.random.randn(self.config.max_steps + self.config.lookback_window) * 0.02
 prices = 1000 * np.exp(np.cumsum(returns))
 self.price_data[asset] = prices

 # Generate volume data
 volumes = np.random.lognormal(15, 1, len(prices))
 self.volume_data[asset] = volumes

 def reset(self) -> np.ndarray:
 """Reset environment to initial state"""
 self.current_step = 0
 self.episode_returns = []

 # Reset portfolio
 self.portfolio = PortfolioManager(self.config)

 # Reset market simulator
 self.market_sim = MarketSimulator(self.config)

 # Get initial observation
 obs = self._get_observation()

 return obs

 def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
 """Execute one environment step"""
 # Parse action
 action_type, asset, amount = self._parse_action(action)

 # Get current prices
 current_prices = self._get_current_prices()

 # Execute action
 if action_type != "hold":
 trade_result = self.portfolio.execute_trade(
 asset,
 amount if action_type == "buy" else -amount,
 current_prices[asset]
 )
 else:
 trade_result = {"status": "hold"}

 # Update market state
 self.current_step += 1
 self._update_market_state()

 # Calculate reward
 reward = self._calculate_reward(trade_result)
 self.episode_returns.append(reward)

 # Get new observation
 obs = self._get_observation()

 # Check if episode is done
 done = self._is_done()

 # Compile info
 info = self._get_info(trade_result)

 return obs, reward, done, info

 def _parse_action(self, action: int) -> Tuple[str, str, float]:
 """Parse discrete action into trade parameters"""
 num_assets = len(self.config.assets)
 action_type_idx = action // num_assets
 asset_idx = action % num_assets

 action_types = ["hold", "buy", "sell"]
 action_type = action_types[action_type_idx] if action_type_idx < 3 else "hold"
 asset = self.config.assets[asset_idx]

 # Calculate trade amount based on portfolio value
 portfolio_value = self.portfolio.update_portfolio_value(self._get_current_prices())
 amount = min(
 portfolio_value * self.config.risk_per_trade,
 portfolio_value * self.config.max_position_size
 )

 return action_type, asset, amount

 def _get_current_prices(self) -> Dict[str, float]:
 """Get current market prices"""
 prices = {}
 for asset in self.config.assets:
 if asset in self.price_data:
 idx = self.config.lookback_window + self.current_step
 prices[asset] = self.price_data[asset][idx]
 else:
 prices[asset] = 1000.0 # Default price

 return prices

 def _update_market_state(self):
 """Update market dynamics"""
 # Update market regime
 recent_prices = {}
 for asset in self.config.assets:
 start_idx = max(0, self.config.lookback_window + self.current_step - 100)
 end_idx = self.config.lookback_window + self.current_step
 recent_prices[asset] = self.price_data[asset][start_idx:end_idx]

 self.market_sim.update_market_regime(recent_prices)

 # Update volatility clustering
 if self.config.simulate_volatility:
 for asset in self.config.assets:
 current_vol = np.std(np.diff(np.log(recent_prices[asset])))
 new_vol = self.market_sim.simulate_volatility_clustering(current_vol)
 self.market_sim.volatility_history[asset].append(new_vol)

 def _get_observation(self) -> np.ndarray:
 """Construct observation vector"""
 obs = []

 current_prices = self._get_current_prices()

 for asset in self.config.assets:
 # Price features
 start_idx = max(0, self.config.lookback_window + self.current_step - 20)
 end_idx = self.config.lookback_window + self.current_step
 recent_prices = self.price_data[asset][start_idx:end_idx]

 if len(recent_prices) > 1:
 returns = np.diff(np.log(recent_prices))
 obs.extend([
 recent_prices[-1] / recent_prices[0] - 1, # Return
 np.mean(returns), # Mean return
 np.std(returns), # Volatility
 np.min(recent_prices), # Min price
 np.max(recent_prices), # Max price
 (recent_prices[-1] - np.min(recent_prices)) / (np.max(recent_prices) - np.min(recent_prices) + 1e-8), # Price position
 np.mean(recent_prices[-5:]) / np.mean(recent_prices[-20:]) - 1, # MA ratio
 returns[-1] if len(returns) > 0 else 0, # Last return
 np.sum(returns > 0) / len(returns), # Win rate
 self.market_sim.order_book_imbalance.get(asset, 0) # Order book imbalance
 ])
 else:
 obs.extend([0] * 10)

 # Volume features
 recent_volumes = self.volume_data[asset][start_idx:end_idx]
 if len(recent_volumes) > 1:
 obs.extend([
 recent_volumes[-1] / np.mean(recent_volumes), # Volume ratio
 np.std(recent_volumes) / (np.mean(recent_volumes) + 1e-8), # Volume volatility
 np.max(recent_volumes) / (np.mean(recent_volumes) + 1e-8), # Max volume ratio
 np.sum(recent_volumes[-5:]) / np.sum(recent_volumes[-20:]), # Volume trend
 self.market_sim.bid_ask_spread.get(asset, 0.001) # Bid-ask spread
 ])
 else:
 obs.extend([0] * 5)

 # Position features
 position_value = self.portfolio.positions.get(asset, 0) * current_prices[asset]
 portfolio_value = self.portfolio.update_portfolio_value(current_prices)
 obs.append(position_value / portfolio_value if portfolio_value > 0 else 0)

 # Portfolio statistics
 portfolio_value = self.portfolio.update_portfolio_value(current_prices)
 obs.extend([
 (portfolio_value - self.config.initial_balance) / self.config.initial_balance, # Total return
 self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1, # Cash ratio
 len([p for p in self.portfolio.positions.values() if p > 0]) / len(self.config.assets), # Position ratio
 self.portfolio.current_drawdown, # Current drawdown
 self.portfolio.realized_pnl / self.config.initial_balance, # Realized PnL ratio
 self.portfolio.unrealized_pnl / self.config.initial_balance, # Unrealized PnL ratio
 float(self.market_sim.current_regime == MarketRegime.BULL), # Bull market
 float(self.market_sim.current_regime == MarketRegime.BEAR), # Bear market
 float(self.market_sim.current_regime == MarketRegime.VOLATILE), # Volatile market
 self.current_step / self.config.max_steps # Time progress
 ])

 return np.array(obs, dtype=np.float32)

 def _calculate_reward(self, trade_result: Dict[str, Any]) -> float:
 """Calculate step reward"""
 current_prices = self._get_current_prices()
 portfolio_value = self.portfolio.update_portfolio_value(current_prices)

 # Base return
 returns = (portfolio_value - self.config.initial_balance) / self.config.initial_balance

 # Risk-adjusted returns
 if len(self.episode_returns) > 10:
 recent_returns = np.array(self.episode_returns[-10:])

 if self.config.reward_type == "sharpe":
 if np.std(recent_returns) > 0:
 reward = np.mean(recent_returns) / np.std(recent_returns)
 else:
 reward = np.mean(recent_returns)
 elif self.config.reward_type == "sortino":
 downside_returns = recent_returns[recent_returns < 0]
 if len(downside_returns) > 0 and np.std(downside_returns) > 0:
 reward = np.mean(recent_returns) / np.std(downside_returns)
 else:
 reward = np.mean(recent_returns)
 else:
 reward = returns
 else:
 reward = returns

 # Penalties
 if trade_result["status"] == "executed":
 reward -= self.config.transaction_penalty

 # Risk penalty
 reward -= self.config.risk_penalty * self.portfolio.current_drawdown

 # Holding penalty (encourage active trading)
 total_position_value = sum([
 self.portfolio.positions.get(asset, 0) * current_prices[asset]
 for asset in self.config.assets
 ])
 if total_position_value < portfolio_value * 0.1:
 reward -= self.config.holding_penalty

 return reward

 def _is_done(self) -> bool:
 """Check if episode should end"""
 # Time limit
 if self.current_step >= self.config.max_steps:
 return True

 # Bankruptcy
 portfolio_value = self.portfolio.update_portfolio_value(self._get_current_prices())
 if portfolio_value < self.config.initial_balance * 0.1:
 return True

 # Max drawdown breached
 if self.portfolio.current_drawdown > self.config.max_drawdown * 1.5:
 return True

 return False

 def _get_info(self, trade_result: Dict[str, Any]) -> Dict[str, Any]:
 """Compile step information"""
 current_prices = self._get_current_prices()
 portfolio_value = self.portfolio.update_portfolio_value(current_prices)

 info = {
 "step": self.current_step,
 "portfolio_value": portfolio_value,
 "total_return": (portfolio_value - self.config.initial_balance) / self.config.initial_balance,
 "cash": self.portfolio.cash,
 "positions": dict(self.portfolio.positions),
 "realized_pnl": self.portfolio.realized_pnl,
 "unrealized_pnl": self.portfolio.unrealized_pnl,
 "current_drawdown": self.portfolio.current_drawdown,
 "market_regime": self.market_sim.current_regime.value,
 "trade_result": trade_result
 }

 # Add risk metrics
 if len(self.episode_returns) > 10:
 risk_metrics = self.portfolio.calculate_risk_metrics(np.array(self.episode_returns))
 info.update(risk_metrics)

 return info

 def render(self, mode: str = 'human'):
 """Render environment state"""
 if mode == 'ansi':
 return self._get_ansi_string()
 elif mode == 'human':
 print(self._get_ansi_string())

 def _get_ansi_string(self) -> str:
 """Get ANSI string representation"""
 current_prices = self._get_current_prices()
 portfolio_value = self.portfolio.update_portfolio_value(current_prices)

 output = f"\n{'='*50}\n"
 output += f"Step: {self.current_step}/{self.config.max_steps}\n"
 output += f"Portfolio Value: ${portfolio_value:.2f}\n"
 output += f"Total Return: {(portfolio_value - self.config.initial_balance) / self.config.initial_balance:.2%}\n"
 output += f"Cash: ${self.portfolio.cash:.2f}\n"
 output += f"Drawdown: {self.portfolio.current_drawdown:.2%}\n"
 output += f"Market Regime: {self.market_sim.current_regime.value}\n"
 output += f"\nPositions:\n"

 for asset, position in self.portfolio.positions.items():
 if position > 0:
 value = position * current_prices[asset]
 output += f" {asset}: {position:.4f} (${value:.2f})\n"

 output += f"{'='*50}\n"

 return output

 def close(self):
 """Clean up environment"""
 pass