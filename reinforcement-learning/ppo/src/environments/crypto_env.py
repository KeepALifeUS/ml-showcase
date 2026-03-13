"""
Crypto Trading Environment for PPO
for realistic trading simulation

Features:
- Multi-asset trading simulation
- Realistic market dynamics
- Transaction costs and slippage
- Risk metrics integration
- Historical data support
- Real-time market conditions
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import logging
from collections import deque
import random
import math

from ..utils.normalization import RunningMeanStd


@dataclass
class CryptoEnvConfig:
 """Configuration for crypto trading environment"""
 
 # Environment parameters
 initial_balance: float = 10000.0 # Starting capital
 max_steps: int = 1000 # Maximum steps per episode
 
 # Market data
 assets: List[str] = field(default_factory=lambda: ["BTC", "ETH"])
 price_history_length: int = 100 # Lookback window
 
 # Trading parameters
 transaction_cost: float = 0.001 # 0.1% transaction cost
 slippage_factor: float = 0.0001 # Slippage factor
 max_position_size: float = 1.0 # Maximum position (fraction of portfolio)
 
 # Market dynamics
 volatility_factor: float = 1.0 # Volatility multiplier
 trend_strength: float = 0.1 # Trend persistence
 mean_reversion_factor: float = 0.05 # Mean reversion strength
 
 # Risk parameters
 margin_requirement: float = 1.0 # Margin requirement
 liquidation_threshold: float = 0.8 # Liquidation threshold
 max_drawdown_limit: float = 0.5 # Maximum allowed drawdown
 
 # Reward configuration
 reward_scaling: float = 1.0 # Reward scaling factor
 risk_penalty_factor: float = 0.1 # Risk penalty multiplier
 diversification_bonus: float = 0.01 # Diversification reward
 
 # Data source
 data_source: str = "synthetic" # synthetic, historical, live
 data_path: Optional[str] = None # Path to historical data
 
 # Advanced features
 include_technical_indicators: bool = True
 include_order_book: bool = False
 include_news_sentiment: bool = False


class MarketDataGenerator:
 """
 Synthetic market data generator
 
 Generates realistic crypto price movements
 with configurable market conditions
 """
 
 def __init__(self, config: CryptoEnvConfig):
 self.config = config
 self.assets = config.assets
 
 # Market state
 self.current_prices = {asset: 100.0 for asset in self.assets} # Normalized prices
 self.price_trends = {asset: 0.0 for asset in self.assets}
 self.volatilities = {asset: 0.02 for asset in self.assets} # Daily volatility
 
 # Market regime
 self.market_regime = "normal" # bull, bear, normal, volatile
 self.regime_persistence = 0.95
 
 # Technical indicators
 if config.include_technical_indicators:
 self.ma_periods = [5, 20, 50]
 self.price_history = {asset: deque(maxlen=max(self.ma_periods)) for asset in self.assets}
 for asset in self.assets:
 self.price_history[asset].extend([100.0] * max(self.ma_periods))
 
 def generate_next_prices(self) -> Dict[str, float]:
 """Generate next price tick"""
 
 # Update market regime
 self._update_market_regime()
 
 new_prices = {}
 
 for asset in self.assets:
 # Get current state
 current_price = self.current_prices[asset]
 trend = self.price_trends[asset]
 volatility = self.volatilities[asset] * self.config.volatility_factor
 
 # Generate price movement
 random_shock = np.random.normal(0, volatility)
 
 # Trend component
 trend_component = trend * self.config.trend_strength
 
 # Mean reversion
 mean_price = 100.0 # Normalized mean
 mean_reversion = (mean_price - current_price) * self.config.mean_reversion_factor
 
 # Market regime effects
 regime_effect = self._get_regime_effect()
 
 # Combined price change
 price_change = trend_component + random_shock + mean_reversion + regime_effect
 
 # Update price
 new_price = current_price * (1 + price_change)
 new_price = max(new_price, 1.0) # Prevent negative prices
 
 new_prices[asset] = new_price
 self.current_prices[asset] = new_price
 
 # Update price history
 if hasattr(self, 'price_history'):
 self.price_history[asset].append(new_price)
 
 # Update trend (with some persistence)
 self.price_trends[asset] = 0.9 * trend + 0.1 * price_change
 
 return new_prices
 
 def _update_market_regime(self):
 """Update market regime"""
 
 # Regime transition probabilities
 if random.random() > self.regime_persistence:
 regimes = ["bull", "bear", "normal", "volatile"]
 self.market_regime = random.choice(regimes)
 
 def _get_regime_effect(self) -> float:
 """Get market regime effect on prices"""
 
 regime_effects = {
 "bull": 0.001, # Slight upward bias
 "bear": -0.001, # Slight downward bias
 "normal": 0.0, # No bias
 "volatile": 0.0 # No bias but higher volatility handled elsewhere
 }
 
 return regime_effects.get(self.market_regime, 0.0)
 
 def get_technical_indicators(self, asset: str) -> Dict[str, float]:
 """Calculate technical indicators"""
 
 if not hasattr(self, 'price_history') or len(self.price_history[asset]) == 0:
 return {}
 
 prices = list(self.price_history[asset])
 indicators = {}
 
 # Moving averages
 for period in self.ma_periods:
 if len(prices) >= period:
 ma_value = np.mean(prices[-period:])
 indicators[f"ma_{period}"] = ma_value
 else:
 indicators[f"ma_{period}"] = prices[-1] if prices else 100.0
 
 # RSI (simplified)
 if len(prices) >= 14:
 price_changes = np.diff(prices[-14:])
 gains = np.where(price_changes > 0, price_changes, 0)
 losses = np.where(price_changes < 0, -price_changes, 0)
 
 avg_gain = np.mean(gains)
 avg_loss = np.mean(losses)
 
 if avg_loss != 0:
 rs = avg_gain / avg_loss
 rsi = 100 - (100 / (1 + rs))
 else:
 rsi = 100
 
 indicators["rsi"] = rsi
 else:
 indicators["rsi"] = 50.0
 
 # Volatility
 if len(prices) >= 20:
 returns = np.diff(np.log(prices[-20:]))
 volatility = np.std(returns) * np.sqrt(365) # Annualized
 indicators["volatility"] = volatility
 else:
 indicators["volatility"] = 0.2
 
 return indicators


class CryptoTradingEnvironment(gym.Env):
 """
 Crypto trading environment for PPO training
 
 Provides realistic crypto trading simulation
 with proper reward structure and market dynamics
 """
 
 def __init__(self, config: Optional[CryptoEnvConfig] = None):
 super().__init__()
 
 self.config = config or CryptoEnvConfig()
 self.logger = logging.getLogger(__name__)
 
 # Initialize market data generator
 self.market_generator = MarketDataGenerator(self.config)
 
 # Trading state
 self.balance = self.config.initial_balance
 self.initial_balance = self.config.initial_balance
 self.positions = {asset: 0.0 for asset in self.config.assets}
 self.portfolio_value = self.config.initial_balance
 self.max_portfolio_value = self.config.initial_balance
 
 # Episode tracking
 self.current_step = 0
 self.episode_reward = 0.0
 self.done = False
 
 # History tracking
 self.price_history = deque(maxlen=self.config.price_history_length)
 self.portfolio_history = deque(maxlen=1000)
 self.action_history = deque(maxlen=100)
 
 # Performance metrics
 self.total_trades = 0
 self.winning_trades = 0
 self.total_fees = 0.0
 self.max_drawdown = 0.0
 
 # Define action and observation spaces
 self._setup_spaces()
 
 self.logger.info(f"Crypto trading environment initialized with {len(self.config.assets)} assets")
 
 def _setup_spaces(self):
 """Setup action and observation spaces"""
 
 # Action space: [position_sizes...] for each asset
 # Each position size is in [-max_position_size, max_position_size]
 self.action_space = spaces.Box(
 low=-self.config.max_position_size,
 high=self.config.max_position_size,
 shape=(len(self.config.assets),),
 dtype=np.float32
 )
 
 # Observation space: 
 # - Price history for each asset
 # - Technical indicators
 # - Portfolio state
 # - Market regime indicators
 
 obs_size = 0
 
 # Price history
 obs_size += len(self.config.assets) * self.config.price_history_length
 
 # Technical indicators (per asset)
 if self.config.include_technical_indicators:
 obs_size += len(self.config.assets) * 6 # MA5, MA20, MA50, RSI, volatility, price_ratio
 
 # Portfolio state
 obs_size += len(self.config.assets) # Current positions
 obs_size += 4 # balance, portfolio_value, drawdown, step_ratio
 
 # Market state
 obs_size += 3 # market_regime encoded, volatility_regime, trend_strength
 
 self.observation_space = spaces.Box(
 low=-np.inf,
 high=np.inf,
 shape=(obs_size,),
 dtype=np.float32
 )
 
 def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
 """Reset environment to initial state"""
 
 super().reset(seed=seed)
 
 # Reset trading state
 self.balance = self.config.initial_balance
 self.positions = {asset: 0.0 for asset in self.config.assets}
 self.portfolio_value = self.config.initial_balance
 self.max_portfolio_value = self.config.initial_balance
 
 # Reset episode tracking
 self.current_step = 0
 self.episode_reward = 0.0
 self.done = False
 
 # Reset metrics
 self.total_trades = 0
 self.winning_trades = 0
 self.total_fees = 0.0
 self.max_drawdown = 0.0
 
 # Clear history
 self.price_history.clear()
 self.portfolio_history.clear()
 self.action_history.clear()
 
 # Reset market generator
 self.market_generator = MarketDataGenerator(self.config)
 
 # Generate initial price history
 for _ in range(self.config.price_history_length):
 prices = self.market_generator.generate_next_prices()
 self.price_history.append(prices)
 
 # Get initial observation
 observation = self._get_observation()
 
 return observation, {}
 
 def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
 """Execute one step in environment"""
 
 # Convert action to positions
 target_positions = {}
 for i, asset in enumerate(self.config.assets):
 target_positions[asset] = float(action[i])
 
 # Generate new market prices
 new_prices = self.market_generator.generate_next_prices()
 self.price_history.append(new_prices)
 
 # Calculate portfolio value before trading
 old_portfolio_value = self._calculate_portfolio_value(new_prices)
 
 # Execute trades
 trade_info = self._execute_trades(target_positions, new_prices)
 
 # Update portfolio state
 self.portfolio_value = self._calculate_portfolio_value(new_prices)
 
 # Calculate reward
 reward = self._calculate_reward(new_prices, trade_info)
 self.episode_reward += reward
 
 # Update performance metrics
 self._update_metrics(trade_info)
 
 # Check termination conditions
 self.done = self._check_done()
 
 # Get next observation
 observation = self._get_observation()
 
 # Prepare info
 info = {
 "portfolio_value": self.portfolio_value,
 "balance": self.balance,
 "positions": self.positions.copy(),
 "total_trades": self.total_trades,
 "total_fees": self.total_fees,
 "max_drawdown": self.max_drawdown,
 "prices": new_prices.copy(),
 "trade_info": trade_info
 }
 
 # Add episode info if done
 if self.done:
 info["episode_reward"] = self.episode_reward
 info["episode_length"] = self.current_step
 info["final_portfolio_value"] = self.portfolio_value
 info["total_return"] = (self.portfolio_value - self.initial_balance) / self.initial_balance
 info["win_rate"] = self.winning_trades / max(self.total_trades, 1)
 
 self.current_step += 1
 
 return observation, reward, self.done, False, info
 
 def _execute_trades(
 self,
 target_positions: Dict[str, float],
 current_prices: Dict[str, float]
 ) -> Dict[str, Any]:
 """Execute trades based on target positions"""
 
 trade_info = {
 "trades_executed": [],
 "total_fees": 0.0,
 "total_slippage": 0.0
 }
 
 for asset in self.config.assets:
 current_position = self.positions[asset]
 target_position = target_positions[asset]
 
 # Calculate position change
 position_change = target_position - current_position
 
 if abs(position_change) < 1e-6: # No significant change
 continue
 
 current_price = current_prices[asset]
 trade_value = abs(position_change) * current_price
 
 # Check if enough balance for buy orders
 if position_change > 0: # Buying
 required_balance = trade_value * (1 + self.config.transaction_cost)
 if required_balance > self.balance:
 # Partial fill based on available balance
 max_affordable = self.balance / (current_price * (1 + self.config.transaction_cost))
 position_change = min(position_change, max_affordable)
 trade_value = position_change * current_price
 
 if abs(position_change) < 1e-6:
 continue
 
 # Apply slippage
 slippage = self.config.slippage_factor * np.sqrt(abs(position_change))
 if position_change > 0:
 execution_price = current_price * (1 + slippage)
 else:
 execution_price = current_price * (1 - slippage)
 
 # Calculate fees
 fees = trade_value * self.config.transaction_cost
 
 # Execute trade
 self.positions[asset] += position_change
 
 if position_change > 0: # Buying
 self.balance -= trade_value + fees
 else: # Selling
 self.balance += trade_value - fees
 
 # Track trade
 self.total_trades += 1
 if position_change > 0:
 action_type = "buy"
 else:
 action_type = "sell"
 
 trade_info["trades_executed"].append({
 "asset": asset,
 "action": action_type,
 "quantity": abs(position_change),
 "price": execution_price,
 "fees": fees,
 "slippage": slippage * current_price
 })
 
 trade_info["total_fees"] += fees
 trade_info["total_slippage"] += abs(slippage * current_price * abs(position_change))
 
 self.total_fees += trade_info["total_fees"]
 
 return trade_info
 
 def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
 """Calculate current portfolio value"""
 
 total_value = self.balance
 
 for asset in self.config.assets:
 position_value = self.positions[asset] * current_prices[asset]
 total_value += position_value
 
 return total_value
 
 def _calculate_reward(
 self,
 current_prices: Dict[str, float],
 trade_info: Dict[str, Any]
 ) -> float:
 """Calculate step reward"""
 
 # Portfolio return
 if len(self.portfolio_history) > 0:
 previous_value = self.portfolio_history[-1]["portfolio_value"]
 portfolio_return = (self.portfolio_value - previous_value) / previous_value
 else:
 portfolio_return = 0.0
 
 # Base reward from portfolio return
 reward = portfolio_return * self.config.reward_scaling
 
 # Risk penalty
 drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
 risk_penalty = drawdown * self.config.risk_penalty_factor
 reward -= risk_penalty
 
 # Transaction cost penalty
 fee_penalty = trade_info["total_fees"] / self.portfolio_value
 reward -= fee_penalty
 
 # Diversification bonus
 active_positions = sum(1 for pos in self.positions.values() if abs(pos) > 1e-6)
 if active_positions > 1:
 diversification_bonus = self.config.diversification_bonus * (active_positions - 1)
 reward += diversification_bonus
 
 # Update max portfolio value
 if self.portfolio_value > self.max_portfolio_value:
 self.max_portfolio_value = self.portfolio_value
 
 # Store portfolio history
 self.portfolio_history.append({
 "step": self.current_step,
 "portfolio_value": self.portfolio_value,
 "balance": self.balance,
 "positions": self.positions.copy(),
 "reward": reward,
 "prices": current_prices.copy()
 })
 
 return reward
 
 def _get_observation(self) -> np.ndarray:
 """Get current observation"""
 
 observation = []
 
 # Price history (normalized)
 if len(self.price_history) >= self.config.price_history_length:
 recent_prices = list(self.price_history)[-self.config.price_history_length:]
 for asset in self.config.assets:
 asset_prices = [prices[asset] for prices in recent_prices]
 # Normalize relative to first price
 if asset_prices[0] > 0:
 normalized_prices = [p / asset_prices[0] for p in asset_prices]
 else:
 normalized_prices = asset_prices
 observation.extend(normalized_prices)
 else:
 # Pad with zeros if insufficient history
 padding_length = len(self.config.assets) * self.config.price_history_length
 observation.extend([1.0] * padding_length)
 
 # Technical indicators
 if self.config.include_technical_indicators:
 for asset in self.config.assets:
 indicators = self.market_generator.get_technical_indicators(asset)
 current_price = list(self.price_history)[-1][asset] if self.price_history else 100.0
 
 # Add indicators (normalized)
 observation.extend([
 indicators.get("ma_5", current_price) / current_price,
 indicators.get("ma_20", current_price) / current_price,
 indicators.get("ma_50", current_price) / current_price,
 (indicators.get("rsi", 50.0) - 50.0) / 50.0, # Centered around 0
 indicators.get("volatility", 0.2),
 current_price / 100.0 # Price ratio to base
 ])
 
 # Portfolio state
 for asset in self.config.assets:
 observation.append(self.positions[asset] / self.config.max_position_size)
 
 # Portfolio metrics
 observation.extend([
 self.balance / self.initial_balance,
 self.portfolio_value / self.initial_balance,
 (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value, # Drawdown
 self.current_step / self.config.max_steps # Step ratio
 ])
 
 # Market state
 regime_encoded = {
 "bull": [1, 0, 0],
 "bear": [0, 1, 0], 
 "normal": [0, 0, 1],
 "volatile": [1, 1, 0]
 }
 observation.extend(regime_encoded.get(self.market_generator.market_regime, [0, 0, 1]))
 
 return np.array(observation, dtype=np.float32)
 
 def _update_metrics(self, trade_info: Dict[str, Any]):
 """Update performance metrics"""
 
 # Update drawdown
 current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
 self.max_drawdown = max(self.max_drawdown, current_drawdown)
 
 # Update winning trades (simplified)
 if len(self.portfolio_history) > 1:
 if self.portfolio_value > self.portfolio_history[-2]["portfolio_value"]:
 self.winning_trades += len(trade_info["trades_executed"])
 
 def _check_done(self) -> bool:
 """Check if episode should terminate"""
 
 # Maximum steps reached
 if self.current_step >= self.config.max_steps:
 return True
 
 # Liquidation condition
 if self.portfolio_value <= self.initial_balance * self.config.liquidation_threshold:
 return True
 
 # Maximum drawdown exceeded
 if self.max_drawdown >= self.config.max_drawdown_limit:
 return True
 
 return False
 
 def render(self, mode: str = "human") -> Optional[Any]:
 """Render environment state"""
 
 if mode == "human":
 print(f"Step: {self.current_step}")
 print(f"Portfolio Value: ${self.portfolio_value:.2f}")
 print(f"Balance: ${self.balance:.2f}")
 print("Positions:", {asset: f"{pos:.4f}" for asset, pos in self.positions.items()})
 print(f"Total Trades: {self.total_trades}")
 print(f"Max Drawdown: {self.max_drawdown:.2%}")
 print("-" * 50)
 
 elif mode == "rgb_array":
 # Could return a chart/plot as numpy array
 pass
 
 def close(self):
 """Clean up environment"""
 pass


# Factory function
def create_crypto_env(config: Optional[CryptoEnvConfig] = None) -> CryptoTradingEnvironment:
 """Create crypto trading environment"""
 return CryptoTradingEnvironment(config)


# Convenience aliases
CryptoEnv = CryptoTradingEnvironment
CryptoTradingEnv = CryptoTradingEnvironment


# Export classes
__all__ = [
 "CryptoEnvConfig",
 "CryptoTradingEnvironment",
 "CryptoEnv", 
 "CryptoTradingEnv",
 "MarketDataGenerator",
 "create_crypto_env"
]