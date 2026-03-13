"""
DQN Crypto Trading Agent with enterprise patterns.

Specialized DQN agent for crypto trading:
- Crypto-specific state representation (OHLCV, indicators, order book)
- Trading action space (buy/sell/hold with position sizing)
- Risk-adjusted reward shaping (Sharpe ratio, max drawdown)
- Transaction cost modeling
- Multi-asset portfolio management
- Real-time market data integration
- Advanced risk management
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field, validator
import structlog
from datetime import datetime, timedelta

from ..core.dqn import DQN, DQNConfig
from ..extensions.double_dqn import DoubleDQN, DoubleDQNConfig
from ..networks.q_network import QNetworkConfig

logger = structlog.get_logger(__name__)


class TradingAction(int, Enum):
 """Trading actions."""
 STRONG_SELL = 0
 SELL = 1
 HOLD = 2
 BUY = 3
 STRONG_BUY = 4


@dataclass
class MarketData:
 """Market data for state."""
 timestamp: datetime
 symbol: str
 open: float
 high: float
 low: float
 close: float
 volume: float

 # Technical indicators
 rsi: Optional[float] = None
 macd: Optional[float] = None
 bb_upper: Optional[float] = None
 bb_lower: Optional[float] = None
 ema_20: Optional[float] = None
 ema_50: Optional[float] = None

 # Order book data
 bid_price: Optional[float] = None
 ask_price: Optional[float] = None
 bid_size: Optional[float] = None
 ask_size: Optional[float] = None


@dataclass
class PortfolioState:
 """State portfolio."""
 cash_balance: float
 positions: Dict[str, float] # symbol -> quantity
 total_value: float
 unrealized_pnl: float
 realized_pnl: float


@dataclass
class TradingResult:
 """Result trading actions."""
 action: TradingAction
 symbol: str
 quantity: float
 price: float
 transaction_cost: float
 success: bool
 error_message: Optional[str] = None


class TradingEnvironmentConfig(BaseModel):
 """Configuration trading environment."""

 # Market parameters
 symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT"], description="Trading symbols")
 initial_balance: float = Field(default=10000.0, description="Initial balance", gt=0)

 # Transaction costs
 maker_fee: float = Field(default=0.001, description="Maker fee", ge=0, le=0.1)
 taker_fee: float = Field(default=0.0015, description="Taker fee", ge=0, le=0.1)
 slippage: float = Field(default=0.001, description="Slippage factor", ge=0, le=0.05)

 # Position management
 max_position_size: float = Field(default=0.3, description="Max position size (fraction)", ge=0, le=1.0)
 min_trade_size: float = Field(default=10.0, description="Minimum size trades", gt=0)

 # Risk management
 max_drawdown: float = Field(default=0.2, description="Maximum drawdown", ge=0, le=0.5)
 stop_loss: Optional[float] = Field(default=0.05, description="Stop loss level", ge=0, le=0.2)
 take_profit: Optional[float] = Field(default=0.1, description="Take profit level", ge=0, le=0.5)

 # State representation
 lookback_window: int = Field(default=20, description="Lookback window for state", gt=0, le=100)
 include_technical_indicators: bool = Field(default=True, description="Include technical indicators")
 include_order_book: bool = Field(default=False, description="Include order book data")

 # Reward shaping
 reward_type: str = Field(default="sharpe", description="Type reward function")
 risk_aversion: float = Field(default=0.5, description="Risk aversion parameter", ge=0, le=2.0)

 @validator("reward_type")
 def validate_reward_type(cls, v):
 valid_types = ["pnl", "sharpe", "sortino", "calmar", "risk_adjusted"]
 if v not in valid_types:
 raise ValueError(f"Reward type must be one from: {valid_types}")
 return v


class CryptoTradingDQNConfig(DoubleDQNConfig):
 """Configuration DQN for crypto trading."""

 # Trading environment
 trading_config: TradingEnvironmentConfig = Field(default_factory=TradingEnvironmentConfig)

 # DQN customizations for trading
 use_double_dqn: bool = Field(default=True, description="Use Double DQN")
 use_prioritized_replay: bool = Field(default=True, description="Use PER")

 # Portfolio specific
 rebalance_freq: int = Field(default=24, description="Frequency rebalancing (hours)", gt=0)
 risk_budget: float = Field(default=0.02, description="Risk budget per trade", ge=0, le=0.1)


class DQNTrader:
 """
 DQN Trading Agent with enterprise-grade implementation for crypto trading.

 Features:
 - Multi-asset portfolio management with dynamic allocation
 - Risk-adjusted reward functions (Sharpe, Sortino, Calmar)
 - Transaction cost modeling with realistic slippage
 - Advanced risk management (stop-loss, take-profit, drawdown control)
 - Real-time technical indicators integration
 - Portfolio rebalancing strategies
 - Performance attribution analysis
 - Backtesting and forward testing support
 """

 def __init__(self, config: CryptoTradingDQNConfig):
 """
 Initialization DQN Trading Agent.

 Args:
 config: Configuration trading agent
 """
 self.config = config
 self.trading_config = config.trading_config

 # Creating state space sizes
 self._setup_state_space

 # Setup action space
 self.action_size = len(TradingAction) * len(self.trading_config.symbols)

 # Updating network config
 network_config = config.network_config
 network_config.state_size = self.state_size
 network_config.action_size = self.action_size

 # Creating DQN agent
 if config.use_double_dqn:
 self.agent = DoubleDQN(config)
 else:
 self.agent = DQN(config)

 # Portfolio tracking
 self.portfolio_history = []
 self.trade_history = []
 self.performance_metrics = {}

 # Risk management
 self.current_drawdown = 0.0
 self.max_portfolio_value = config.trading_config.initial_balance
 self.last_rebalance_time = None

 # Market data storage
 self.market_data_buffer = {}

 self.logger = structlog.get_logger(__name__).bind(
 component="DQNTrader",
 symbols=self.trading_config.symbols,
 initial_balance=self.trading_config.initial_balance
 )

 self.logger.info("DQN Trader initialized", config=config.dict)

 def _setup_state_space(self) -> None:
 """Setup sizes state space."""
 base_features_per_symbol = 5 # OHLCV

 # Technical indicators
 if self.trading_config.include_technical_indicators:
 base_features_per_symbol += 6 # RSI, MACD, BB_upper, BB_lower, EMA_20, EMA_50

 # Order book data
 if self.trading_config.include_order_book:
 base_features_per_symbol += 4 # bid_price, ask_price, bid_size, ask_size

 # Market features
 market_features = base_features_per_symbol * len(self.trading_config.symbols) * self.trading_config.lookback_window

 # Portfolio features
 portfolio_features = 3 + len(self.trading_config.symbols) # cash, total_value, drawdown + positions

 # Time features
 time_features = 4 # hour, day_of_week, day_of_month, month

 self.state_size = market_features + portfolio_features + time_features

 self.logger.debug("State space configured",
 state_size=self.state_size,
 market_features=market_features,
 portfolio_features=portfolio_features)

 def create_state(self,
 market_data: Dict[str, List[MarketData]],
 portfolio: PortfolioState,
 timestamp: datetime) -> np.ndarray:
 """
 Creating state representation.

 Args:
 market_data: Market data by symbol
 portfolio: Current state portfolio
 timestamp: Current time

 Returns:
 State vector
 """
 state_components = []

 # Market data features
 for symbol in self.trading_config.symbols:
 symbol_data = market_data.get(symbol, [])

 # Getting last lookback_window points
 recent_data = symbol_data[-self.trading_config.lookback_window:]

 for data_point in recent_data:
 # OHLCV features (normalized)
 if len(recent_data) > 1:
 close_prev = recent_data[0].close
 features = [
 (data_point.open - close_prev) / close_prev,
 (data_point.high - close_prev) / close_prev,
 (data_point.low - close_prev) / close_prev,
 (data_point.close - close_prev) / close_prev,
 np.log(data_point.volume + 1) / 20, # Volume normalization
 ]
 else:
 features = [0.0, 0.0, 0.0, 0.0, 0.0]

 # Technical indicators
 if self.trading_config.include_technical_indicators:
 features.extend([
 (data_point.rsi or 50) / 100 - 0.5, # RSI normalized to [-0.5, 0.5]
 np.tanh((data_point.macd or 0) / 100), # MACD
 ((data_point.bb_upper or data_point.close) - data_point.close) / data_point.close,
 ((data_point.bb_lower or data_point.close) - data_point.close) / data_point.close,
 ((data_point.ema_20 or data_point.close) - data_point.close) / data_point.close,
 ((data_point.ema_50 or data_point.close) - data_point.close) / data_point.close,
 ])

 # Order book data
 if self.trading_config.include_order_book:
 spread = ((data_point.ask_price or data_point.close) -
 (data_point.bid_price or data_point.close)) / data_point.close
 features.extend([
 spread,
 np.log((data_point.bid_size or 1) + 1) / 10,
 np.log((data_point.ask_size or 1) + 1) / 10,
 spread * np.log((data_point.bid_size or 1) + (data_point.ask_size or 1) + 1),
 ])

 state_components.extend(features)

 # Padding if insufficient data
 while len(recent_data) < self.trading_config.lookback_window:
 padding_size = (5 +
 (6 if self.trading_config.include_technical_indicators else 0) +
 (4 if self.trading_config.include_order_book else 0))
 state_components.extend([0.0] * padding_size)

 # Portfolio features
 portfolio_features = [
 portfolio.cash_balance / self.trading_config.initial_balance - 1, # Normalized cash
 portfolio.total_value / self.trading_config.initial_balance - 1, # Normalized total value
 self.current_drawdown, # Current drawdown
 ]

 # Position features (normalized by total value)
 for symbol in self.trading_config.symbols:
 position = portfolio.positions.get(symbol, 0.0)
 position_value = position * (market_data.get(symbol, [{}])[-1].close if market_data.get(symbol) else 0)
 normalized_position = position_value / max(portfolio.total_value, 1.0)
 portfolio_features.append(normalized_position)

 state_components.extend(portfolio_features)

 # Time features
 time_features = [
 timestamp.hour / 24.0, # Hour of day
 timestamp.weekday / 7.0, # Day of week
 timestamp.day / 31.0, # Day of month
 timestamp.month / 12.0, # Month
 ]
 state_components.extend(time_features)

 return np.array(state_components, dtype=np.float32)

 def decode_action(self, action_id: int) -> Tuple[str, TradingAction]:
 """
 Decoding action ID in symbol and trading action.

 Args:
 action_id: ID actions

 Returns:
 Tuple of (symbol, trading_action)
 """
 num_actions = len(TradingAction)
 symbol_idx = action_id // num_actions
 action_idx = action_id % num_actions

 symbol = self.trading_config.symbols[symbol_idx]
 trading_action = TradingAction(action_idx)

 return symbol, trading_action

 def calculate_position_size(self,
 symbol: str,
 action: TradingAction,
 portfolio: PortfolioState,
 current_price: float) -> float:
 """
 Calculate sizeand position with accounting risk management.

 Args:
 symbol: Trading symbol
 action: Trading action
 portfolio: Current portfolio state
 current_price: Current price

 Returns:
 Position size
 """
 if action == TradingAction.HOLD:
 return 0.0

 # Available capital
 available_cash = portfolio.cash_balance
 max_position_value = portfolio.total_value * self.trading_config.max_position_size

 # Current position
 current_position = portfolio.positions.get(symbol, 0.0)
 current_position_value = abs(current_position * current_price)

 # Risk-based position sizing
 risk_budget = portfolio.total_value * self.config.risk_budget

 if action in [TradingAction.BUY, TradingAction.STRONG_BUY]:
 # Buy actions
 max_buy_value = min(
 available_cash * 0.95, # Leave some cash buffer
 max_position_value - current_position_value,
 risk_budget * (2 if action == TradingAction.STRONG_BUY else 1)
 )

 quantity = max_buy_value / current_price

 # Apply minimum trade size
 if max_buy_value < self.trading_config.min_trade_size:
 return 0.0

 else: # SELL actions
 # Sell actions
 if current_position <= 0:
 return 0.0 # No position to sell

 sell_fraction = 0.5 if action == TradingAction.SELL else 1.0
 quantity = -current_position * sell_fraction

 # Minimum trade size check
 if abs(quantity * current_price) < self.trading_config.min_trade_size:
 return 0.0

 return quantity

 def execute_trade(self,
 symbol: str,
 action: TradingAction,
 quantity: float,
 current_price: float,
 portfolio: PortfolioState) -> TradingResult:
 """
 Simulation completion trades with transaction costs.

 Args:
 symbol: Trading symbol
 action: Trading action
 quantity: Trade quantity
 current_price: Current price
 portfolio: Current portfolio state

 Returns:
 Trading result
 """
 if quantity == 0:
 return TradingResult(
 action=action,
 symbol=symbol,
 quantity=0,
 price=current_price,
 transaction_cost=0,
 success=True
 )

 # Calculate transaction costs
 trade_value = abs(quantity * current_price)

 # Determine if maker or taker (simplified)
 is_market_order = action in [TradingAction.STRONG_BUY, TradingAction.STRONG_SELL]
 fee_rate = self.trading_config.taker_fee if is_market_order else self.trading_config.maker_fee

 # Slippage simulation
 slippage_factor = 1 + (self.trading_config.slippage if quantity > 0 else -self.trading_config.slippage)
 execution_price = current_price * slippage_factor

 # Total transaction cost
 transaction_cost = trade_value * fee_rate

 # Check if sufficient balance
 if quantity > 0: # Buy
 total_cost = trade_value + transaction_cost
 if total_cost > portfolio.cash_balance:
 return TradingResult(
 action=action,
 symbol=symbol,
 quantity=0,
 price=current_price,
 transaction_cost=0,
 success=False,
 error_message="Insufficient balance"
 )
 else: # Sell
 current_position = portfolio.positions.get(symbol, 0.0)
 if abs(quantity) > current_position:
 return TradingResult(
 action=action,
 symbol=symbol,
 quantity=0,
 price=current_price,
 transaction_cost=0,
 success=False,
 error_message="Insufficient position"
 )

 # Execute trade
 portfolio.positions[symbol] = portfolio.positions.get(symbol, 0.0) + quantity
 portfolio.cash_balance -= quantity * execution_price + transaction_cost

 # Update realized P&L
 if quantity < 0: # Selling
 # Simple P&L calculation (can improve with FIFO/LIFO)
 cost_basis = execution_price # Simplified
 pnl = abs(quantity) * (execution_price - cost_basis)
 portfolio.realized_pnl += pnl

 trade_result = TradingResult(
 action=action,
 symbol=symbol,
 quantity=quantity,
 price=execution_price,
 transaction_cost=transaction_cost,
 success=True
 )

 # Record trade
 self.trade_history.append(trade_result)

 return trade_result

 def calculate_reward(self,
 portfolio_before: PortfolioState,
 portfolio_after: PortfolioState,
 trade_result: TradingResult) -> float:
 """
 Calculate reward based on trading performance.

 Args:
 portfolio_before: Portfolio state to actions
 portfolio_after: Portfolio state after actions
 trade_result: Result trades

 Returns:
 Reward value
 """
 # Base P&L reward
 pnl_change = portfolio_after.total_value - portfolio_before.total_value
 pnl_reward = pnl_change / self.trading_config.initial_balance

 if self.trading_config.reward_type == "pnl":
 return pnl_reward

 # Risk-adjusted rewards
 if len(self.portfolio_history) > 1:
 returns = []
 for i in range(1, len(self.portfolio_history)):
 ret = (self.portfolio_history[i].total_value -
 self.portfolio_history[i-1].total_value) / self.portfolio_history[i-1].total_value
 returns.append(ret)

 if len(returns) > 10: # Minimum history for meaningful calculation
 returns_array = np.array(returns)

 if self.trading_config.reward_type == "sharpe":
 # Sharpe ratio reward
 if np.std(returns_array) > 0:
 sharpe = np.mean(returns_array) / np.std(returns_array)
 return sharpe * 0.1 # Scale factor

 elif self.trading_config.reward_type == "sortino":
 # Sortino ratio (downside deviation)
 negative_returns = returns_array[returns_array < 0]
 if len(negative_returns) > 0 and np.std(negative_returns) > 0:
 sortino = np.mean(returns_array) / np.std(negative_returns)
 return sortino * 0.1

 elif self.trading_config.reward_type == "calmar":
 # Calmar ratio (return / max drawdown)
 if self.current_drawdown > 0:
 calmar = np.mean(returns_array) / self.current_drawdown
 return calmar * 0.1

 # Risk penalty
 risk_penalty = 0.0
 if self.current_drawdown > self.trading_config.max_drawdown * 0.8:
 risk_penalty = -0.1 * (self.current_drawdown / self.trading_config.max_drawdown)

 # Transaction cost penalty
 cost_penalty = -trade_result.transaction_cost / self.trading_config.initial_balance

 # Risk-adjusted reward
 final_reward = (pnl_reward +
 risk_penalty +
 cost_penalty * self.trading_config.risk_aversion)

 return final_reward

 def update_portfolio_metrics(self, portfolio: PortfolioState) -> None:
 """Updating portfolio metrics and risk tracking."""
 # Update max portfolio value and drawdown
 if portfolio.total_value > self.max_portfolio_value:
 self.max_portfolio_value = portfolio.total_value
 self.current_drawdown = 0.0
 else:
 self.current_drawdown = (self.max_portfolio_value - portfolio.total_value) / self.max_portfolio_value

 # Record portfolio history
 self.portfolio_history.append(portfolio)

 # Limit history size
 if len(self.portfolio_history) > 10000:
 self.portfolio_history = self.portfolio_history[-5000:]

 def act(self,
 market_data: Dict[str, List[MarketData]],
 portfolio: PortfolioState,
 timestamp: datetime,
 training: bool = True) -> Tuple[str, TradingAction, float]:
 """
 Selection trading actions.

 Args:
 market_data: Market data
 portfolio: Portfolio state
 timestamp: Current timestamp
 training: Training mode

 Returns:
 Tuple of (symbol, action, quantity)
 """
 # Create state
 state = self.create_state(market_data, portfolio, timestamp)

 # Get action from DQN
 action_id = self.agent.act(state, training)

 # Decode action
 symbol, trading_action = self.decode_action(action_id)

 # Calculate position size
 current_price = market_data.get(symbol, [{}])[-1].close if market_data.get(symbol) else 0.0
 quantity = self.calculate_position_size(symbol, trading_action, portfolio, current_price)

 return symbol, trading_action, quantity

 def train_step(self,
 state_before: np.ndarray,
 action_id: int,
 reward: float,
 state_after: np.ndarray,
 done: bool) -> Dict[str, float]:
 """
 Training step.

 Args:
 state_before: State before actions
 action_id: Action ID
 reward: Reward
 state_after: State after actions
 done: Episode done flag

 Returns:
 Training metrics
 """
 # Store experience
 self.agent.store_experience(state_before, action_id, reward, state_after, done)

 # Train
 metrics = self.agent.train_step

 return metrics

 def get_trading_statistics(self) -> Dict[str, Any]:
 """Get statistics trading."""
 if not self.trade_history:
 return {"trades": 0}

 # Trade statistics
 total_trades = len(self.trade_history)
 successful_trades = sum(1 for trade in self.trade_history if trade.success)

 # P&L statistics
 total_pnl = sum(getattr(trade, 'pnl', 0) for trade in self.trade_history)
 total_costs = sum(trade.transaction_cost for trade in self.trade_history)

 # Portfolio performance
 if len(self.portfolio_history) > 1:
 initial_value = self.portfolio_history[0].total_value
 current_value = self.portfolio_history[-1].total_value
 total_return = (current_value - initial_value) / initial_value

 # Calculate Sharpe ratio
 returns = []
 for i in range(1, len(self.portfolio_history)):
 ret = ((self.portfolio_history[i].total_value - self.portfolio_history[i-1].total_value) /
 self.portfolio_history[i-1].total_value)
 returns.append(ret)

 sharpe_ratio = 0.0
 if len(returns) > 1 and np.std(returns) > 0:
 sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) # Annualized
 else:
 total_return = 0.0
 sharpe_ratio = 0.0

 # Action distribution
 action_counts = {}
 for trade in self.trade_history:
 action_counts[trade.action.name] = action_counts.get(trade.action.name, 0) + 1

 return {
 "total_trades": total_trades,
 "successful_trades": successful_trades,
 "success_rate": successful_trades / total_trades if total_trades > 0 else 0,
 "total_pnl": total_pnl,
 "total_transaction_costs": total_costs,
 "net_pnl": total_pnl - total_costs,
 "total_return": total_return,
 "max_drawdown": self.current_drawdown,
 "sharpe_ratio": sharpe_ratio,
 "action_distribution": action_counts,
 "portfolio_history_length": len(self.portfolio_history),
 }

 def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
 """Saving checkpoint trading agent."""
 trading_metadata = {
 "portfolio_history": self.portfolio_history[-1000:], # Last 1000 records
 "trade_history": self.trade_history[-1000:],
 "trading_statistics": self.get_trading_statistics,
 "current_drawdown": self.current_drawdown,
 "max_portfolio_value": self.max_portfolio_value,
 }

 if metadata:
 trading_metadata.update(metadata)

 self.agent.save_checkpoint(filepath, trading_metadata)

 def __repr__(self) -> str:
 """String representation trading agent."""
 stats = self.get_trading_statistics
 return (
 f"DQNTrader(symbols={self.trading_config.symbols}, "
 f"trades={stats.get('total_trades', 0)}, "
 f"return={stats.get('total_return', 0):.2%}, "
 f"sharpe={stats.get('sharpe_ratio', 0):.2f})"
 )