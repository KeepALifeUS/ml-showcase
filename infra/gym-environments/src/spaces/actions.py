"""
Action Spaces for Crypto Trading Environments
enterprise patterns for sophisticated trading actions

Features:
- Multi-asset trading actions
- Discrete and continuous action spaces
- Order type support (market/limit/stop)
- Position sizing with risk constraints
- Portfolio rebalancing actions
- Advanced order management
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from gymnasium import spaces
from enum import Enum
import warnings

from ..utils.risk_metrics import PositionSizer


class ActionMode(Enum):
    """Action space modes"""
    DISCRETE = "discrete"           # Buy/Sell/Hold for each asset
    CONTINUOUS = "continuous"       # Continuous position sizing
    MIXED = "mixed"                # Discrete actions + continuous sizing
    ORDERS = "orders"              # Advanced order management
    PORTFOLIO = "portfolio"        # Portfolio allocation targets


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"


@dataclass
class ActionConfig:
    """Configuration for action spaces"""
    
    # Action mode
    action_mode: ActionMode = ActionMode.CONTINUOUS
    
    # Position sizing
    max_position_size: float = 1.0      # Maximum position per asset
    min_position_size: float = 0.001    # Minimum position size
    position_granularity: int = 100     # Granularity for discrete sizing
    
    # Portfolio constraints
    max_portfolio_weight: float = 0.3   # Maximum weight per asset
    min_portfolio_weight: float = 0.0   # Minimum weight per asset
    max_leverage: float = 1.0           # Maximum total leverage
    max_correlation_exposure: float = 0.8  # Max exposure to correlated assets
    
    # Order management
    enable_limit_orders: bool = True
    enable_stop_orders: bool = True
    price_offset_range: float = 0.05    # Max price offset for limit orders
    
    # Risk management
    enable_position_sizing: bool = True
    position_sizing_method: str = "kelly"  # kelly, fixed_fraction, volatility
    risk_per_trade: float = 0.02        # Max risk per trade
    
    # Advanced features
    enable_portfolio_rebalancing: bool = False
    enable_multi_timeframe: bool = False
    enable_pair_trading: bool = False
    
    # Performance optimization
    clip_actions: bool = True
    normalize_actions: bool = True


class DiscreteActionSpace:
    """Discrete action space for crypto trading"""
    
    def __init__(self, config: ActionConfig, assets: List[str]):
        self.config = config
        self.assets = assets
        self.num_assets = len(assets)
        
        # Action mapping
        self.actions_per_asset = 3  # Buy, Sell, Hold
        self.total_actions = self.actions_per_asset ** self.num_assets
        
        # Create action mapping
        self._create_action_mapping()
    
    def _create_action_mapping(self) -> None:
        """Create mapping from action index to asset actions"""
        
        self.action_mapping = {}
        
        for action_idx in range(self.total_actions):
            asset_actions = []
            remaining = action_idx
            
            for _ in range(self.num_assets):
                asset_action = remaining % self.actions_per_asset
                asset_actions.append(asset_action)
                remaining //= self.actions_per_asset
            
            self.action_mapping[action_idx] = asset_actions
    
    def create_space(self) -> spaces.Discrete:
        """Create gymnasium discrete space"""
        return spaces.Discrete(self.total_actions)
    
    def parse_action(self, action: int) -> Dict[str, Any]:
        """Parse discrete action to trading orders"""
        
        if action not in self.action_mapping:
            warnings.warn(f"Invalid action index: {action}")
            action = 0  # Default to all hold
        
        asset_actions = self.action_mapping[action]
        
        orders = []
        for i, asset in enumerate(self.assets):
            asset_action = asset_actions[i]
            
            if asset_action == 0:  # Hold
                continue
            elif asset_action == 1:  # Buy
                orders.append({
                    "asset": asset,
                    "side": "buy",
                    "order_type": OrderType.MARKET,
                    "quantity": self.config.max_position_size / self.num_assets,
                    "price": None
                })
            elif asset_action == 2:  # Sell
                orders.append({
                    "asset": asset,
                    "side": "sell",
                    "order_type": OrderType.MARKET,
                    "quantity": self.config.max_position_size / self.num_assets,
                    "price": None
                })
        
        return {"orders": orders, "action_type": "discrete"}


class ContinuousActionSpace:
    """Continuous action space for sophisticated position sizing"""
    
    def __init__(self, config: ActionConfig, assets: List[str]):
        self.config = config
        self.assets = assets
        self.num_assets = len(assets)
        
        # Position sizer for risk management
        if config.enable_position_sizing:
            self.position_sizer = PositionSizer(
                method=config.position_sizing_method,
                max_risk=config.risk_per_trade
            )
        
        # Action dimensions
        self.action_dim = self._calculate_action_dimensions()
    
    def _calculate_action_dimensions(self) -> int:
        """Calculate total action dimensions"""
        
        dim = 0
        
        # Base position targets for each asset
        dim += self.num_assets
        
        # Optional price offsets for limit orders
        if self.config.enable_limit_orders:
            dim += self.num_assets  # Price offsets
        
        # Optional stop loss levels
        if self.config.enable_stop_orders:
            dim += self.num_assets  # Stop levels
        
        return dim
    
    def create_space(self) -> spaces.Box:
        """Create gymnasium continuous space"""
        
        # Position targets: [-max_position, +max_position]
        low = np.full(self.action_dim, -self.config.max_position_size, dtype=np.float32)
        high = np.full(self.action_dim, self.config.max_position_size, dtype=np.float32)
        
        # Adjust bounds for price offsets and stops
        if self.config.enable_limit_orders or self.config.enable_stop_orders:
            offset_start = self.num_assets
            offset_end = offset_start + self.num_assets
            
            # Price offsets: [-price_offset_range, +price_offset_range]
            if self.config.enable_limit_orders:
                low[offset_start:offset_end] = -self.config.price_offset_range
                high[offset_start:offset_end] = self.config.price_offset_range
                offset_start = offset_end
                offset_end = offset_start + self.num_assets
            
            # Stop levels: [0, price_offset_range] (percentage below current price)
            if self.config.enable_stop_orders:
                low[offset_start:offset_end] = 0.0
                high[offset_start:offset_end] = self.config.price_offset_range
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def parse_action(
        self, 
        action: np.ndarray,
        current_prices: Optional[Dict[str, float]] = None,
        current_positions: Optional[Dict[str, float]] = None,
        portfolio_value: Optional[float] = None
    ) -> Dict[str, Any]:
        """Parse continuous action to trading orders"""
        
        if self.config.clip_actions:
            action = np.clip(action, -self.config.max_position_size, self.config.max_position_size)
        
        orders = []
        action_idx = 0
        
        # Parse position targets
        position_targets = action[:self.num_assets]
        action_idx += self.num_assets
        
        # Parse price offsets
        price_offsets = None
        if self.config.enable_limit_orders:
            price_offsets = action[action_idx:action_idx + self.num_assets]
            action_idx += self.num_assets
        
        # Parse stop levels
        stop_levels = None
        if self.config.enable_stop_orders:
            stop_levels = action[action_idx:action_idx + self.num_assets]
            action_idx += self.num_assets
        
        # Create orders for each asset
        for i, asset in enumerate(self.assets):
            target_position = position_targets[i]
            current_position = current_positions.get(asset, 0.0) if current_positions else 0.0
            
            # Calculate position change
            position_change = target_position - current_position
            
            if abs(position_change) < self.config.min_position_size:
                continue
            
            # Apply position sizing
            if hasattr(self, 'position_sizer') and current_prices and portfolio_value:
                current_price = current_prices.get(asset, 1.0)
                position_change = self.position_sizer.calculate_position_size(
                    position_change, current_price, portfolio_value
                )
            
            # Determine order side
            side = "buy" if position_change > 0 else "sell"
            quantity = abs(position_change)
            
            # Create base order
            order = {
                "asset": asset,
                "side": side,
                "quantity": quantity,
                "order_type": OrderType.MARKET,
                "price": None
            }
            
            # Add limit order pricing
            if self.config.enable_limit_orders and price_offsets is not None:
                price_offset = price_offsets[i]
                if abs(price_offset) > 1e-6 and current_prices:  # Use limit order
                    current_price = current_prices.get(asset, 1.0)
                    limit_price = current_price * (1 + price_offset)
                    
                    order["order_type"] = OrderType.LIMIT
                    order["price"] = limit_price
            
            # Add stop loss
            if self.config.enable_stop_orders and stop_levels is not None:
                stop_level = stop_levels[i]
                if stop_level > 1e-6 and current_prices:
                    current_price = current_prices.get(asset, 1.0)
                    stop_price = current_price * (1 - stop_level)
                    
                    # Create separate stop order
                    stop_order = {
                        "asset": asset,
                        "side": "sell" if side == "buy" else "buy",
                        "quantity": quantity,
                        "order_type": OrderType.STOP,
                        "price": stop_price
                    }
                    orders.append(stop_order)
            
            orders.append(order)
        
        return {"orders": orders, "action_type": "continuous"}


class PortfolioActionSpace:
    """Portfolio allocation action space"""
    
    def __init__(self, config: ActionConfig, assets: List[str]):
        self.config = config
        self.assets = assets
        self.num_assets = len(assets)
    
    def create_space(self) -> spaces.Box:
        """Create portfolio allocation space"""
        
        # Portfolio weights: [0, 1] for each asset
        # Will be normalized to sum to 1 in parse_action
        low = np.zeros(self.num_assets, dtype=np.float32)
        high = np.ones(self.num_assets, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def parse_action(
        self,
        action: np.ndarray,
        current_positions: Optional[Dict[str, float]] = None,
        portfolio_value: Optional[float] = None,
        current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Parse portfolio allocation action"""
        
        # Normalize weights to sum to 1
        action = np.clip(action, 0.0, 1.0)
        if np.sum(action) > 0:
            weights = action / np.sum(action)
        else:
            weights = np.ones(self.num_assets) / self.num_assets
        
        # Apply portfolio constraints
        weights = np.clip(weights, 
                         self.config.min_portfolio_weight,
                         self.config.max_portfolio_weight)
        
        # Renormalize after clipping
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        orders = []
        
        if portfolio_value and current_prices and current_positions:
            for i, asset in enumerate(self.assets):
                target_weight = weights[i]
                target_value = portfolio_value * target_weight
                current_price = current_prices.get(asset, 1.0)
                target_position = target_value / current_price
                
                current_position = current_positions.get(asset, 0.0)
                position_change = target_position - current_position
                
                if abs(position_change) > self.config.min_position_size:
                    side = "buy" if position_change > 0 else "sell"
                    quantity = abs(position_change)
                    
                    orders.append({
                        "asset": asset,
                        "side": side,
                        "quantity": quantity,
                        "order_type": OrderType.MARKET,
                        "price": None,
                        "target_weight": target_weight
                    })
        
        return {
            "orders": orders,
            "action_type": "portfolio",
            "target_weights": dict(zip(self.assets, weights))
        }


class CryptoActionSpace:
    """Main action space builder for crypto trading"""
    
    def __init__(self, config: ActionConfig, assets: List[str]):
        self.config = config
        self.assets = assets
        
        # Create appropriate action space
        if config.action_mode == ActionMode.DISCRETE:
            self.action_space_impl = DiscreteActionSpace(config, assets)
        elif config.action_mode == ActionMode.CONTINUOUS:
            self.action_space_impl = ContinuousActionSpace(config, assets)
        elif config.action_mode == ActionMode.PORTFOLIO:
            self.action_space_impl = PortfolioActionSpace(config, assets)
        else:
            raise ValueError(f"Unsupported action mode: {config.action_mode}")
    
    def create_space(self) -> Union[spaces.Discrete, spaces.Box]:
        """Create gymnasium action space"""
        return self.action_space_impl.create_space()
    
    def parse_action(
        self,
        action: Union[int, np.ndarray],
        current_prices: Optional[Dict[str, float]] = None,
        current_positions: Optional[Dict[str, float]] = None,
        portfolio_value: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Parse action to trading orders"""
        
        return self.action_space_impl.parse_action(
            action=action,
            current_prices=current_prices,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
            **kwargs
        )
    
    def get_action_info(self) -> Dict[str, Any]:
        """Get information about action space"""
        
        return {
            "action_mode": self.config.action_mode.value,
            "assets": self.assets.copy(),
            "config": self.config.__dict__,
            "space_type": type(self.action_space_impl).__name__
        }
    
    def sample_action(self) -> Union[int, np.ndarray]:
        """Sample random action from space"""
        
        space = self.create_space()
        return space.sample()
    
    def validate_action(self, action: Union[int, np.ndarray]) -> bool:
        """Validate action is within space bounds"""
        
        space = self.create_space()
        return space.contains(action)


# Advanced action spaces

class MultiTimeframeActionSpace:
    """Multi-timeframe trading actions"""
    
    def __init__(self, config: ActionConfig, assets: List[str], timeframes: List[str]):
        self.config = config
        self.assets = assets
        self.timeframes = timeframes
        self.num_assets = len(assets)
        self.num_timeframes = len(timeframes)
        
        # Create separate action space for each timeframe
        self.timeframe_actions = {}
        for tf in timeframes:
            tf_config = ActionConfig(**config.__dict__)
            self.timeframe_actions[tf] = ContinuousActionSpace(tf_config, assets)
    
    def create_space(self) -> spaces.Box:
        """Create multi-timeframe action space"""
        
        # Combine action dimensions from all timeframes
        total_dim = 0
        for tf_action in self.timeframe_actions.values():
            total_dim += tf_action.action_dim
        
        low = np.full(total_dim, -self.config.max_position_size, dtype=np.float32)
        high = np.full(total_dim, self.config.max_position_size, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def parse_action(self, action: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Parse multi-timeframe action"""
        
        all_orders = []
        action_idx = 0
        
        for tf, tf_action_space in self.timeframe_actions.items():
            tf_action = action[action_idx:action_idx + tf_action_space.action_dim]
            tf_result = tf_action_space.parse_action(tf_action, **kwargs)
            
            # Add timeframe information to orders
            for order in tf_result["orders"]:
                order["timeframe"] = tf
            
            all_orders.extend(tf_result["orders"])
            action_idx += tf_action_space.action_dim
        
        return {
            "orders": all_orders,
            "action_type": "multi_timeframe"
        }


class PairTradingActionSpace:
    """Pair trading action space"""
    
    def __init__(self, config: ActionConfig, asset_pairs: List[Tuple[str, str]]):
        self.config = config
        self.asset_pairs = asset_pairs
        self.num_pairs = len(asset_pairs)
    
    def create_space(self) -> spaces.Box:
        """Create pair trading action space"""
        
        # Each pair has: pair_ratio, hedge_ratio
        action_dim = self.num_pairs * 2
        
        low = np.full(action_dim, -1.0, dtype=np.float32)
        high = np.full(action_dim, 1.0, dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)
    
    def parse_action(self, action: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Parse pair trading action"""
        
        orders = []
        
        for i, (asset1, asset2) in enumerate(self.asset_pairs):
            pair_signal = action[i * 2]      # [-1, 1] Long/short pair
            hedge_ratio = action[i * 2 + 1]  # [-1, 1] Hedge ratio
            
            if abs(pair_signal) > 0.1:  # Threshold for action
                # Long first asset, short second asset (or vice versa)
                if pair_signal > 0:
                    orders.append({
                        "asset": asset1,
                        "side": "buy",
                        "quantity": abs(pair_signal) * self.config.max_position_size,
                        "order_type": OrderType.MARKET,
                        "pair_trade": True,
                        "pair_id": f"{asset1}_{asset2}"
                    })
                    
                    orders.append({
                        "asset": asset2,
                        "side": "sell",
                        "quantity": abs(pair_signal * hedge_ratio) * self.config.max_position_size,
                        "order_type": OrderType.MARKET,
                        "pair_trade": True,
                        "pair_id": f"{asset1}_{asset2}"
                    })
                else:
                    orders.append({
                        "asset": asset1,
                        "side": "sell",
                        "quantity": abs(pair_signal) * self.config.max_position_size,
                        "order_type": OrderType.MARKET,
                        "pair_trade": True,
                        "pair_id": f"{asset1}_{asset2}"
                    })
                    
                    orders.append({
                        "asset": asset2,
                        "side": "buy", 
                        "quantity": abs(pair_signal * hedge_ratio) * self.config.max_position_size,
                        "order_type": OrderType.MARKET,
                        "pair_trade": True,
                        "pair_id": f"{asset1}_{asset2}"
                    })
        
        return {
            "orders": orders,
            "action_type": "pair_trading"
        }


__all__ = [
    "ActionMode",
    "OrderType", 
    "ActionConfig",
    "DiscreteActionSpace",
    "ContinuousActionSpace",
    "PortfolioActionSpace",
    "CryptoActionSpace",
    "MultiTimeframeActionSpace",
    "PairTradingActionSpace"
]