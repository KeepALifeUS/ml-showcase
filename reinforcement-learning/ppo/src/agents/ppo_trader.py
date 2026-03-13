"""
PPO Crypto Trading Agent
for production crypto trading

Features:
- Real-time trading decisions
- Risk management integration
- Multi-asset support
- Position sizing optimization
- Performance monitoring
- Production-ready deployment
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import logging
from collections import deque, defaultdict
import time
from datetime import datetime, timedelta
import asyncio
import json

from ..core.ppo import PPOAlgorithm, PPOConfig
from ..networks.actor_critic import CryptoActorCritic, ActorCriticConfig
from ..utils.normalization import ObservationNormalizer, RunningMeanStd
from ..environments.crypto_env import CryptoTradingEnvironment


@dataclass
class PPOTraderConfig:
 """Configuration for PPO crypto trader"""
 
 # Trading parameters
 max_position_size: float = 1.0 # Maximum position size (fraction)
 min_position_size: float = 0.01 # Minimum position size
 stop_loss_threshold: float = 0.05 # Stop loss threshold (5%)
 take_profit_threshold: float = 0.10 # Take profit threshold (10%)
 
 # Risk management
 max_drawdown: float = 0.15 # Maximum drawdown (15%)
 portfolio_heat: float = 0.02 # Portfolio heat (2% risk per trade)
 max_correlation: float = 0.8 # Maximum correlation between positions
 
 # Multi-asset trading
 assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "BNB"])
 max_concurrent_positions: int = 3 # Maximum concurrent positions
 position_rebalance_frequency: int = 24 # Hours between rebalancing
 
 # Market data
 lookback_window: int = 100 # Price history window
 technical_indicators: List[str] = field(default_factory=lambda: [
 "rsi", "macd", "bb", "ema", "volume_sma"
 ])
 
 # Model configuration
 model_path: Optional[str] = None # Path to trained model
 normalize_observations: bool = True
 observation_clip: float = 5.0
 
 # Performance tracking
 track_sharpe_ratio: bool = True
 track_max_drawdown: bool = True 
 track_win_rate: bool = True
 performance_window: int = 1000 # Performance calculation window
 
 # Execution
 execution_delay: float = 0.1 # Execution delay (seconds)
 slippage_model: str = "linear" # linear, sqrt, constant
 transaction_costs: float = 0.001 # Transaction costs (0.1%)
 
 # Real-time features
 realtime_updates: bool = True
 update_frequency: int = 60 # Seconds between updates
 
 # Production features
 monitoring_enabled: bool = True
 alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
 "max_loss": -0.10,
 "max_drawdown": -0.15,
 "min_sharpe": 0.5
 })


class PositionManager:
 """
 Position management for crypto trading
 
 Handles position sizing, risk management,
 and portfolio optimization
 """
 
 def __init__(self, config: PPOTraderConfig):
 self.config = config
 
 # Current positions
 self.positions: Dict[str, Dict[str, float]] = {} # {asset: {size, entry_price, timestamp}}
 self.portfolio_value = 1.0 # Starting value
 self.available_capital = 1.0
 
 # Risk tracking
 self.max_drawdown_reached = 0.0
 self.peak_value = 1.0
 self.position_history = deque(maxlen=1000)
 
 # Performance metrics
 self.total_trades = 0
 self.winning_trades = 0
 self.total_pnl = 0.0
 self.sharpe_history = deque(maxlen=config.performance_window)
 
 self.logger = logging.getLogger(__name__)
 
 def can_open_position(self, asset: str, position_size: float) -> bool:
 """Check if can open position"""
 
 # Check maximum concurrent positions
 if len(self.positions) >= self.config.max_concurrent_positions:
 if asset not in self.positions:
 return False
 
 # Check minimum position size
 if abs(position_size) < self.config.min_position_size:
 return False
 
 # Check maximum position size
 if abs(position_size) > self.config.max_position_size:
 return False
 
 # Check available capital
 required_capital = abs(position_size) * self.portfolio_value
 if required_capital > self.available_capital:
 return False
 
 # Check portfolio heat
 portfolio_risk = self._calculate_portfolio_risk(asset, position_size)
 if portfolio_risk > self.config.portfolio_heat:
 return False
 
 return True
 
 def open_position(
 self,
 asset: str,
 position_size: float,
 current_price: float,
 confidence: float = 1.0
 ) -> bool:
 """Open new position or update existing"""
 
 if not self.can_open_position(asset, position_size):
 return False
 
 # Risk-adjusted position sizing
 adjusted_size = self._risk_adjust_position_size(
 position_size, confidence, asset
 )
 
 # Update position
 if asset in self.positions:
 # Update existing position
 current_pos = self.positions[asset]
 weighted_price = (
 current_pos["size"] * current_pos["entry_price"] +
 adjusted_size * current_price
 ) / (current_pos["size"] + adjusted_size)
 
 self.positions[asset] = {
 "size": current_pos["size"] + adjusted_size,
 "entry_price": weighted_price,
 "timestamp": time.time()
 }
 else:
 # New position
 self.positions[asset] = {
 "size": adjusted_size,
 "entry_price": current_price,
 "timestamp": time.time()
 }
 
 # Update available capital
 self.available_capital -= abs(adjusted_size) * current_price
 
 self.logger.info(f"Opened position: {asset} size={adjusted_size:.4f} price={current_price:.4f}")
 
 return True
 
 def close_position(self, asset: str, current_price: float, reason: str = "") -> bool:
 """Close position"""
 
 if asset not in self.positions:
 return False
 
 position = self.positions.pop(asset)
 
 # Calculate PnL
 pnl = position["size"] * (current_price - position["entry_price"])
 self.total_pnl += pnl
 self.total_trades += 1
 
 if pnl > 0:
 self.winning_trades += 1
 
 # Update capital
 position_value = abs(position["size"]) * current_price
 self.available_capital += position_value
 
 self.logger.info(
 f"Closed position: {asset} PnL={pnl:.4f} reason={reason}"
 )
 
 return True
 
 def update_portfolio_value(self, market_prices: Dict[str, float]):
 """Update portfolio value based on current prices"""
 
 # Calculate position values
 position_value = 0.0
 for asset, position in self.positions.items():
 if asset in market_prices:
 current_value = position["size"] * market_prices[asset]
 position_value += current_value
 
 # Total portfolio value
 self.portfolio_value = self.available_capital + position_value
 
 # Update drawdown tracking
 if self.portfolio_value > self.peak_value:
 self.peak_value = self.portfolio_value
 
 current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
 self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)
 
 # Store history
 self.position_history.append({
 "timestamp": time.time(),
 "portfolio_value": self.portfolio_value,
 "drawdown": current_drawdown,
 "num_positions": len(self.positions)
 })
 
 def _calculate_portfolio_risk(self, asset: str, position_size: float) -> float:
 """Calculate portfolio risk with new position"""
 
 # Simple portfolio heat calculation
 position_value = abs(position_size) * self.portfolio_value
 portfolio_risk = position_value / self.portfolio_value
 
 # Add correlation penalty (simplified)
 correlation_penalty = len(self.positions) * 0.001
 
 return portfolio_risk + correlation_penalty
 
 def _risk_adjust_position_size(
 self,
 target_size: float,
 confidence: float,
 asset: str
 ) -> float:
 """Adjust position size based on risk parameters"""
 
 # Confidence adjustment
 confidence_adjusted = target_size * confidence
 
 # Volatility adjustment (simplified)
 volatility_factor = 1.0 # Would calculate from price history
 volatility_adjusted = confidence_adjusted / volatility_factor
 
 # Portfolio heat constraint
 max_size_by_heat = self.config.portfolio_heat * self.portfolio_value
 max_position = min(abs(volatility_adjusted), max_size_by_heat)
 
 # Apply sign
 return max_position * np.sign(target_size)
 
 def get_portfolio_metrics(self) -> Dict[str, float]:
 """Get current portfolio performance metrics"""
 
 # Basic metrics
 win_rate = self.winning_trades / max(self.total_trades, 1)
 
 # Sharpe ratio (simplified)
 if len(self.position_history) > 1:
 returns = []
 for i in range(1, len(self.position_history)):
 ret = (self.position_history[i]["portfolio_value"] / 
 self.position_history[i-1]["portfolio_value"] - 1)
 returns.append(ret)
 
 if returns:
 sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365)
 self.sharpe_history.append(sharpe_ratio)
 else:
 sharpe_ratio = 0.0
 else:
 sharpe_ratio = 0.0
 
 return {
 "portfolio_value": self.portfolio_value,
 "available_capital": self.available_capital,
 "total_pnl": self.total_pnl,
 "max_drawdown": self.max_drawdown_reached,
 "win_rate": win_rate,
 "total_trades": self.total_trades,
 "sharpe_ratio": sharpe_ratio,
 "num_positions": len(self.positions)
 }


class PPOTrader:
 """
 PPO-based crypto trading agent
 
 Features:
 - Multi-asset trading
 - Risk management
 - Real-time decision making
 - Performance monitoring
 - Production deployment
 """
 
 def __init__(
 self,
 config: PPOTraderConfig,
 actor_critic: Optional[nn.Module] = None,
 device: str = "auto"
 ):
 self.config = config
 self.logger = logging.getLogger(__name__)
 
 # Device setup
 if device == "auto":
 self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 else:
 self.device = torch.device(device)
 
 # Initialize actor-critic network
 if actor_critic is None:
 # Create default crypto trading network
 ac_config = ActorCriticConfig(
 obs_dim=len(config.technical_indicators) + config.lookback_window,
 action_dim=3, # position_size, entry_price_offset, stop_loss_ratio
 action_type="crypto_trading"
 )
 self.actor_critic = CryptoActorCritic(
 ac_config,
 price_history_length=config.lookback_window,
 num_technical_indicators=len(config.technical_indicators)
 ).to(self.device)
 else:
 self.actor_critic = actor_critic.to(self.device)
 
 # Load trained model if provided
 if config.model_path:
 self.load_model(config.model_path)
 
 # Initialize observation normalizer
 if config.normalize_observations:
 obs_dim = len(config.technical_indicators) + config.lookback_window
 self.obs_normalizer = ObservationNormalizer((obs_dim,))
 else:
 self.obs_normalizer = None
 
 # Initialize position manager
 self.position_manager = PositionManager(config)
 
 # Market data storage
 self.market_data: Dict[str, deque] = {
 asset: deque(maxlen=config.lookback_window * 2)
 for asset in config.assets
 }
 
 # Performance tracking
 self.trading_history = deque(maxlen=10000)
 self.decision_times = deque(maxlen=1000)
 
 # State
 self.is_trading = False
 self.last_update_time = time.time()
 
 self.logger.info(f"PPO Trader initialized with {len(config.assets)} assets")
 self.logger.info(f"Device: {self.device}")
 
 def load_model(self, model_path: str):
 """Load trained PPO model"""
 
 try:
 checkpoint = torch.load(model_path, map_location=self.device)
 
 if "actor_critic_state_dict" in checkpoint:
 self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
 else:
 self.actor_critic.load_state_dict(checkpoint)
 
 # Load normalizer if available
 if "obs_normalizer_state" in checkpoint and self.obs_normalizer:
 self.obs_normalizer.running_stats.set_state(checkpoint["obs_normalizer_state"])
 
 self.logger.info(f"Model loaded from {model_path}")
 
 except Exception as e:
 self.logger.error(f"Failed to load model: {e}")
 raise
 
 def save_model(self, model_path: str):
 """Save current model state"""
 
 checkpoint = {
 "actor_critic_state_dict": self.actor_critic.state_dict(),
 "config": self.config,
 "portfolio_metrics": self.position_manager.get_portfolio_metrics(),
 "timestamp": time.time()
 }
 
 if self.obs_normalizer:
 checkpoint["obs_normalizer_state"] = self.obs_normalizer.running_stats.get_state()
 
 torch.save(checkpoint, model_path)
 self.logger.info(f"Model saved to {model_path}")
 
 def update_market_data(self, asset: str, market_data: Dict[str, Any]):
 """Update market data for specific asset"""
 
 if asset not in self.market_data:
 return
 
 # Extract required features
 features = {
 "price": market_data.get("price", 0.0),
 "volume": market_data.get("volume", 0.0),
 "timestamp": market_data.get("timestamp", time.time())
 }
 
 # Add technical indicators if available
 for indicator in self.config.technical_indicators:
 features[indicator] = market_data.get(indicator, 0.0)
 
 self.market_data[asset].append(features)
 
 def get_trading_decision(
 self,
 asset: str,
 current_price: float,
 force_decision: bool = False
 ) -> Dict[str, Any]:
 """
 Get trading decision for specific asset
 
 Args:
 asset: Asset symbol
 current_price: Current market price
 force_decision: Force decision even if insufficient data
 
 Returns:
 Trading decision dictionary
 """
 
 decision_start_time = time.time()
 
 try:
 # Check data availability
 if len(self.market_data[asset]) < self.config.lookback_window:
 if not force_decision:
 return self._empty_decision(asset, "Insufficient market data")
 
 # Prepare observations
 observation = self._prepare_observation(asset)
 
 if observation is None:
 return self._empty_decision(asset, "Failed to prepare observation")
 
 # Normalize observation
 if self.obs_normalizer:
 normalized_obs = self.obs_normalizer(observation, update_stats=False)
 else:
 normalized_obs = observation
 
 # Get model prediction
 with torch.no_grad():
 # Extract price history and technical indicators
 obs_tensor = normalized_obs.unsqueeze(0)
 
 if isinstance(self.actor_critic, CryptoActorCritic):
 # Specialized crypto trading forward pass
 price_history = obs_tensor[:, :self.config.lookback_window]
 tech_indicators = obs_tensor[:, self.config.lookback_window:]
 
 action_dist, values, risk_estimates = self.actor_critic(
 price_history, tech_indicators
 )
 else:
 # Standard actor-critic
 action_dist, values = self.actor_critic(obs_tensor)
 risk_estimates = torch.tensor([0.5]) # Default risk
 
 # Sample action (deterministic for production)
 if hasattr(action_dist, 'mean'):
 raw_action = action_dist.mean.squeeze().cpu().numpy()
 else:
 raw_action = action_dist.sample().squeeze().cpu().numpy()
 
 confidence = float(values.squeeze().cpu().numpy())
 risk_level = float(risk_estimates.squeeze().cpu().numpy())
 
 # Interpret action
 trading_action = self._interpret_action(raw_action, current_price, confidence, risk_level)
 
 # Apply risk management
 final_decision = self._apply_risk_management(asset, trading_action, current_price)
 
 # Record decision time
 decision_time = time.time() - decision_start_time
 self.decision_times.append(decision_time)
 
 # Log decision
 self.logger.debug(
 f"Trading decision for {asset}: action={final_decision['action']} "
 f"size={final_decision['position_size']:.4f} "
 f"confidence={confidence:.3f} risk={risk_level:.3f}"
 )
 
 return final_decision
 
 except Exception as e:
 self.logger.error(f"Error generating trading decision for {asset}: {e}")
 return self._empty_decision(asset, f"Error: {str(e)}")
 
 def execute_trading_decision(self, asset: str, decision: Dict[str, Any]) -> bool:
 """Execute trading decision"""
 
 if decision["action"] == "hold":
 return True
 
 current_price = decision["current_price"]
 
 try:
 if decision["action"] == "buy" or decision["action"] == "sell":
 # Open or modify position
 position_size = decision["position_size"]
 if decision["action"] == "sell":
 position_size = -abs(position_size)
 
 success = self.position_manager.open_position(
 asset=asset,
 position_size=position_size,
 current_price=current_price,
 confidence=decision["confidence"]
 )
 
 if success:
 self.trading_history.append({
 "timestamp": time.time(),
 "asset": asset,
 "action": decision["action"],
 "size": position_size,
 "price": current_price,
 "reason": decision.get("reason", "Model decision")
 })
 
 return success
 
 elif decision["action"] == "close":
 # Close position
 success = self.position_manager.close_position(
 asset=asset,
 current_price=current_price,
 reason=decision.get("reason", "Model decision")
 )
 
 if success:
 self.trading_history.append({
 "timestamp": time.time(),
 "asset": asset,
 "action": "close",
 "size": 0.0,
 "price": current_price,
 "reason": decision.get("reason", "Model decision")
 })
 
 return success
 
 except Exception as e:
 self.logger.error(f"Error executing decision for {asset}: {e}")
 return False
 
 return False
 
 def _prepare_observation(self, asset: str) -> Optional[torch.Tensor]:
 """Prepare observation tensor for model input"""
 
 if asset not in self.market_data or len(self.market_data[asset]) == 0:
 return None
 
 try:
 # Get recent market data
 recent_data = list(self.market_data[asset])[-self.config.lookback_window:]
 
 if len(recent_data) < self.config.lookback_window:
 # Pad with earliest available data
 padding_needed = self.config.lookback_window - len(recent_data)
 if recent_data:
 padded_data = [recent_data[0]] * padding_needed + recent_data
 else:
 return None
 recent_data = padded_data
 
 # Extract price history
 prices = [data["price"] for data in recent_data]
 
 # Extract technical indicators
 tech_indicators = []
 for indicator in self.config.technical_indicators:
 values = [data.get(indicator, 0.0) for data in recent_data]
 tech_indicators.append(np.mean(values)) # Use mean for now
 
 # Combine features
 observation = torch.tensor(
 prices + tech_indicators,
 dtype=torch.float32,
 device=self.device
 )
 
 return observation
 
 except Exception as e:
 self.logger.error(f"Error preparing observation for {asset}: {e}")
 return None
 
 def _interpret_action(
 self,
 raw_action: np.ndarray,
 current_price: float,
 confidence: float,
 risk_level: float
 ) -> Dict[str, Any]:
 """Interpret raw model output into trading action"""
 
 if len(raw_action) >= 3:
 position_size = float(raw_action[0])
 price_offset = float(raw_action[1])
 stop_loss_ratio = float(raw_action[2])
 else:
 position_size = float(raw_action[0]) if len(raw_action) > 0 else 0.0
 price_offset = 0.0
 stop_loss_ratio = 0.05
 
 # Determine action type
 if abs(position_size) < self.config.min_position_size:
 action_type = "hold"
 elif position_size > 0:
 action_type = "buy"
 else:
 action_type = "sell"
 
 return {
 "action": action_type,
 "position_size": abs(position_size),
 "entry_price": current_price * (1 + price_offset * 0.01),
 "stop_loss": current_price * (1 - abs(stop_loss_ratio) * 0.05),
 "confidence": confidence,
 "risk_level": risk_level,
 "current_price": current_price
 }
 
 def _apply_risk_management(
 self,
 asset: str,
 trading_action: Dict[str, Any],
 current_price: float
 ) -> Dict[str, Any]:
 """Apply risk management rules to trading action"""
 
 action = trading_action.copy()
 
 # Check drawdown limit
 portfolio_metrics = self.position_manager.get_portfolio_metrics()
 if portfolio_metrics["max_drawdown"] > self.config.max_drawdown:
 if action["action"] in ["buy", "sell"]:
 action["action"] = "hold"
 action["reason"] = "Max drawdown exceeded"
 return action
 
 # Check existing position
 if asset in self.position_manager.positions:
 position = self.position_manager.positions[asset]
 
 # Stop loss check
 current_pnl_ratio = (current_price - position["entry_price"]) / position["entry_price"]
 
 if position["size"] > 0 and current_pnl_ratio < -self.config.stop_loss_threshold:
 action["action"] = "close"
 action["reason"] = "Stop loss triggered"
 return action
 
 if position["size"] < 0 and current_pnl_ratio > self.config.stop_loss_threshold:
 action["action"] = "close"
 action["reason"] = "Stop loss triggered (short)"
 return action
 
 # Take profit check
 if abs(current_pnl_ratio) > self.config.take_profit_threshold:
 action["action"] = "close"
 action["reason"] = "Take profit triggered"
 return action
 
 # Position sizing limits
 if action["action"] in ["buy", "sell"]:
 max_allowed_size = min(
 self.config.max_position_size,
 portfolio_metrics["available_capital"] / current_price
 )
 
 action["position_size"] = min(action["position_size"], max_allowed_size)
 
 # Check minimum size again
 if action["position_size"] < self.config.min_position_size:
 action["action"] = "hold"
 action["reason"] = "Position size too small"
 
 return action
 
 def _empty_decision(self, asset: str, reason: str) -> Dict[str, Any]:
 """Return empty/hold decision"""
 
 return {
 "action": "hold",
 "position_size": 0.0,
 "confidence": 0.0,
 "risk_level": 0.5,
 "reason": reason,
 "asset": asset,
 "timestamp": time.time()
 }
 
 def get_performance_metrics(self) -> Dict[str, Any]:
 """Get comprehensive performance metrics"""
 
 portfolio_metrics = self.position_manager.get_portfolio_metrics()
 
 # Trading performance
 avg_decision_time = np.mean(self.decision_times) if self.decision_times else 0.0
 
 # Recent trading activity
 recent_trades = len([t for t in self.trading_history 
 if t["timestamp"] > time.time() - 86400]) # Last 24h
 
 return {
 **portfolio_metrics,
 "avg_decision_time_ms": avg_decision_time * 1000,
 "recent_trades_24h": recent_trades,
 "total_decisions": len(self.trading_history),
 "assets_monitored": len(self.config.assets),
 "active_positions": len(self.position_manager.positions),
 "last_update": self.last_update_time,
 "model_device": str(self.device)
 }
 
 async def run_realtime_trading(self, market_data_stream: Any):
 """Run real-time trading loop"""
 
 self.is_trading = True
 self.logger.info("Starting real-time trading")
 
 try:
 while self.is_trading:
 # Get market data updates
 market_updates = await market_data_stream.get_updates()
 
 for asset, data in market_updates.items():
 if asset in self.config.assets:
 # Update market data
 self.update_market_data(asset, data)
 
 # Get trading decision
 decision = self.get_trading_decision(asset, data["price"])
 
 # Execute decision
 self.execute_trading_decision(asset, decision)
 
 # Update portfolio metrics
 current_prices = {asset: data["price"] for asset, data in market_updates.items()}
 self.position_manager.update_portfolio_value(current_prices)
 
 # Performance monitoring
 if self.config.monitoring_enabled:
 await self._check_alert_conditions()
 
 # Wait for next update
 await asyncio.sleep(self.config.update_frequency)
 
 except Exception as e:
 self.logger.error(f"Real-time trading error: {e}")
 finally:
 self.is_trading = False
 self.logger.info("Real-time trading stopped")
 
 async def _check_alert_conditions(self):
 """Check alert conditions"""
 
 metrics = self.get_performance_metrics()
 
 for condition, threshold in self.config.alert_thresholds.items():
 if condition in metrics:
 if metrics[condition] < threshold:
 self.logger.warning(
 f"Alert: {condition} = {metrics[condition]:.4f} < {threshold:.4f}"
 )
 
 def stop_trading(self):
 """Stop real-time trading"""
 self.is_trading = False


# Export main class
__all__ = [
 "PPOTraderConfig",
 "PPOTrader",
 "PositionManager"
]