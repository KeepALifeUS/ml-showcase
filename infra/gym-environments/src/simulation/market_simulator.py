"""
Market Simulation Engine
enterprise patterns for realistic market dynamics

Advanced market simulation with sophisticated features:
- Market impact modeling
- Order matching engine
- Liquidity modeling
- Slippage simulation
- Latency simulation
- Market regime effects
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import logging
import random

from ..spaces.actions import OrderType


class MarketImpactModel(Enum):
    """Market impact models"""
    LINEAR = "linear"
    SQRT = "sqrt"
    LOGARITHMIC = "logarithmic"
    ALMGREN_CHRISS = "almgren_chriss"


class LiquidityRegime(Enum):
    """Liquidity regime types"""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    STRESSED = "stressed"


@dataclass
class MarketSimulatorConfig:
    """Configuration for market simulator"""
    
    # Market impact parameters
    impact_model: MarketImpactModel = MarketImpactModel.SQRT
    base_impact_coefficient: float = 0.001      # Base market impact
    temporary_impact_decay: float = 0.9         # Temporary impact decay rate
    permanent_impact_factor: float = 0.3        # Fraction of temporary impact that becomes permanent
    
    # Liquidity modeling
    base_liquidity: Dict[str, float] = field(default_factory=lambda: {})  # Base liquidity per asset
    liquidity_regeneration_rate: float = 0.1   # How fast liquidity regenerates
    min_liquidity_factor: float = 0.1          # Minimum liquidity as fraction of base
    
    # Slippage modeling
    base_slippage: float = 0.0001              # Base slippage factor
    volatility_slippage_multiplier: float = 2.0  # Volatility-based slippage scaling
    size_slippage_exponent: float = 0.5         # Order size effect on slippage
    
    # Order matching
    enable_partial_fills: bool = True           # Allow partial fills
    min_fill_ratio: float = 0.1                # Minimum fill ratio for orders
    max_order_wait_time: float = 30.0          # Max time orders wait (seconds)
    
    # Latency simulation
    enable_latency_simulation: bool = False
    base_latency_ms: float = 1.0               # Base latency in milliseconds
    latency_variance: float = 0.5              # Latency variance
    
    # Market regime effects
    regime_impact_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "bull": 0.8,      # Lower impact in bull markets
        "bear": 1.3,      # Higher impact in bear markets
        "volatile": 1.5,  # Much higher impact in volatile markets
        "crisis": 2.0     # Extreme impact during crisis
    })
    
    # Advanced features
    enable_inventory_effects: bool = True       # Market maker inventory effects
    enable_clustering: bool = True              # Transaction clustering
    enable_momentum_effects: bool = True        # Short-term momentum from large trades


@dataclass
class OrderExecution:
    """Order execution result"""
    order_id: str
    asset: str
    side: str
    requested_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_fees: float
    slippage: float
    market_impact: float
    execution_time: float
    status: str  # filled, partial, rejected, pending


class MarketSimulator:
    """
    Advanced market simulator with realistic trading dynamics
    
    Simulates sophisticated market effects:
    - Market impact from large orders
    - Dynamic liquidity conditions
    - Realistic slippage modeling
    - Order matching with partial fills
    - Latency effects
    """
    
    def __init__(self, assets: List[str], config: Optional[MarketSimulatorConfig] = None):
        self.assets = assets
        self.config = config or MarketSimulatorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Market state
        self._initialize_market_state()
        
        # Order management
        self.pending_orders = {}
        self.order_history = []
        self.order_id_counter = 0
        
        # Market impact tracking
        self.temporary_impacts = defaultdict(lambda: deque(maxlen=100))
        self.permanent_impacts = defaultdict(float)
        
        # Liquidity tracking
        self.current_liquidity = {}
        self.liquidity_shocks = defaultdict(float)
        
        # Performance metrics
        self.total_volume = defaultdict(float)
        self.impact_costs = defaultdict(float)
        self.execution_stats = defaultdict(list)
        
        self.logger.info("Market simulator initialized", extra={
            "assets": assets,
            "impact_model": self.config.impact_model.value,
            "enable_latency": self.config.enable_latency_simulation
        })
    
    def _initialize_market_state(self) -> None:
        """Initialize market state variables"""
        
        # Base liquidity levels
        for asset in self.assets:
            base_liquidity = self.config.base_liquidity.get(asset, 1000.0)
            self.current_liquidity[asset] = base_liquidity
        
        # Market regime
        self.current_regime = "normal"
        
        # Volatility tracking
        self.asset_volatilities = {asset: 0.02 for asset in self.assets}
        self.price_momentum = {asset: 0.0 for asset in self.assets}
    
    def execute_trades(
        self,
        orders: List[Dict[str, Any]],
        current_prices: Dict[str, float],
        portfolio: Dict[str, float],
        balance: float,
        order_books: Optional[Dict[str, Dict]] = None,
        market_regime: str = "normal"
    ) -> Dict[str, Any]:
        """
        Execute list of trading orders with realistic market simulation
        
        Returns comprehensive execution results
        """
        
        self.current_regime = market_regime
        execution_results = []
        total_fees = 0.0
        total_slippage = 0.0
        total_impact = 0.0
        
        # Process orders
        for order in orders:
            try:
                # Validate order
                if not self._validate_order(order, current_prices, portfolio, balance):
                    execution_results.append(self._create_rejected_execution(order, "validation_failed"))
                    continue
                
                # Execute order
                execution = self._execute_single_order(
                    order, current_prices, order_books
                )
                
                execution_results.append(execution)
                
                # Update totals
                total_fees += execution.total_fees
                total_slippage += execution.slippage
                total_impact += execution.market_impact
                
                # Update market state
                self._update_market_state(execution, current_prices)
                
            except Exception as e:
                self.logger.error(f"Error executing order: {e}", exc_info=True)
                execution_results.append(self._create_rejected_execution(order, f"execution_error: {e}"))
        
        # Calculate portfolio updates
        portfolio_updates = self._calculate_portfolio_updates(execution_results, portfolio, balance)
        
        result = {
            "filled_orders": [e.__dict__ for e in execution_results if e.status == "filled"],
            "rejected_orders": [e.__dict__ for e in execution_results if e.status == "rejected"],
            "partial_orders": [e.__dict__ for e in execution_results if e.status == "partial"],
            "total_fees": total_fees,
            "total_slippage": total_slippage,
            "total_market_impact": total_impact,
            "new_balance": portfolio_updates["new_balance"],
            "position_changes": portfolio_updates["position_changes"],
            "execution_summary": self._create_execution_summary(execution_results)
        }
        
        return result
    
    def _validate_order(
        self,
        order: Dict[str, Any],
        current_prices: Dict[str, float],
        portfolio: Dict[str, float],
        balance: float
    ) -> bool:
        """Validate order before execution"""
        
        asset = order.get("asset")
        side = order.get("side")
        quantity = order.get("quantity", 0.0)
        
        # Basic validation
        if not asset or asset not in self.assets:
            self.logger.warning(f"Invalid asset: {asset}")
            return False
        
        if side not in ["buy", "sell"]:
            self.logger.warning(f"Invalid side: {side}")
            return False
        
        if quantity <= 0:
            self.logger.warning(f"Invalid quantity: {quantity}")
            return False
        
        current_price = current_prices.get(asset, 0.0)
        if current_price <= 0:
            self.logger.warning(f"Invalid price for {asset}: {current_price}")
            return False
        
        # Position validation for sell orders
        if side == "sell":
            current_position = portfolio.get(asset, 0.0)
            if current_position < quantity:
                self.logger.warning(f"Insufficient position for sell: {current_position} < {quantity}")
                return False
        
        # Balance validation for buy orders
        elif side == "buy":
            estimated_cost = quantity * current_price * 1.1  # 10% buffer for fees/slippage
            if balance < estimated_cost:
                self.logger.warning(f"Insufficient balance for buy: {balance} < {estimated_cost}")
                return False
        
        return True
    
    def _execute_single_order(
        self,
        order: Dict[str, Any],
        current_prices: Dict[str, float],
        order_books: Optional[Dict[str, Dict]] = None
    ) -> OrderExecution:
        """Execute single order with market simulation"""
        
        start_time = time.time()
        
        # Generate order ID
        order_id = f"order_{self.order_id_counter}"
        self.order_id_counter += 1
        
        asset = order["asset"]
        side = order["side"]
        quantity = float(order["quantity"])
        order_type = order.get("order_type", OrderType.MARKET)
        limit_price = order.get("price")
        
        current_price = current_prices[asset]
        
        # Simulate latency
        if self.config.enable_latency_simulation:
            latency = self._simulate_latency()
            time.sleep(latency / 1000.0)  # Convert ms to seconds
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(asset, quantity, side)
        
        # Calculate slippage
        slippage = self._calculate_slippage(asset, quantity, current_price)
        
        # Determine execution price
        if order_type == OrderType.MARKET:
            execution_price = self._calculate_market_execution_price(
                current_price, market_impact, slippage, side
            )
        elif order_type == OrderType.LIMIT:
            if not self._check_limit_order_execution(current_price, limit_price, side):
                return self._create_rejected_execution(order, "limit_not_reached")
            execution_price = limit_price
        else:
            execution_price = current_price
        
        # Determine fill quantity (can be partial)
        fill_quantity = self._determine_fill_quantity(asset, quantity)
        
        # Calculate fees
        fees = self._calculate_trading_fees(asset, fill_quantity, execution_price)
        
        # Create execution result
        execution = OrderExecution(
            order_id=order_id,
            asset=asset,
            side=side,
            requested_quantity=quantity,
            filled_quantity=fill_quantity,
            avg_fill_price=execution_price,
            total_fees=fees,
            slippage=slippage,
            market_impact=market_impact,
            execution_time=time.time() - start_time,
            status="filled" if fill_quantity == quantity else ("partial" if fill_quantity > 0 else "rejected")
        )
        
        self.order_history.append(execution)
        
        return execution
    
    def _calculate_market_impact(self, asset: str, quantity: float, side: str) -> float:
        """Calculate market impact based on configured model"""
        
        base_coefficient = self.config.base_impact_coefficient
        
        # Adjust for regime
        regime_multiplier = self.config.regime_impact_multipliers.get(self.current_regime, 1.0)
        
        # Adjust for liquidity
        current_liquidity = self.current_liquidity[asset]
        liquidity_factor = max(self.config.min_liquidity_factor, current_liquidity / 1000.0)
        
        # Base impact calculation
        if self.config.impact_model == MarketImpactModel.LINEAR:
            impact = base_coefficient * quantity
        elif self.config.impact_model == MarketImpactModel.SQRT:
            impact = base_coefficient * np.sqrt(quantity)
        elif self.config.impact_model == MarketImpactModel.LOGARITHMIC:
            impact = base_coefficient * np.log1p(quantity)
        else:  # Almgren-Chriss
            volatility = self.asset_volatilities[asset]
            impact = base_coefficient * quantity * volatility
        
        # Apply adjustments
        impact *= regime_multiplier / liquidity_factor
        
        # Ensure reasonable bounds
        impact = np.clip(impact, 0.0, 0.1)  # Max 10% impact
        
        return float(impact)
    
    def _calculate_slippage(self, asset: str, quantity: float, current_price: float) -> float:
        """Calculate realistic slippage"""
        
        # Base slippage
        base_slippage = self.config.base_slippage
        
        # Size-based slippage
        size_factor = np.power(quantity / 100.0, self.config.size_slippage_exponent)
        
        # Volatility-based slippage
        volatility = self.asset_volatilities[asset]
        volatility_factor = 1.0 + volatility * self.config.volatility_slippage_multiplier
        
        # Random component
        random_factor = np.random.normal(1.0, 0.2)
        
        total_slippage = base_slippage * size_factor * volatility_factor * random_factor
        
        # Convert to price terms
        slippage_amount = total_slippage * current_price
        
        return float(slippage_amount)
    
    def _calculate_market_execution_price(
        self,
        current_price: float,
        market_impact: float,
        slippage: float,
        side: str
    ) -> float:
        """Calculate final execution price"""
        
        if side == "buy":
            # Buy orders push price up
            execution_price = current_price * (1 + market_impact) + slippage
        else:
            # Sell orders push price down
            execution_price = current_price * (1 - market_impact) - slippage
        
        # Ensure positive price
        execution_price = max(execution_price, current_price * 0.5)
        
        return float(execution_price)
    
    def _determine_fill_quantity(self, asset: str, requested_quantity: float) -> float:
        """Determine how much of order gets filled"""
        
        if not self.config.enable_partial_fills:
            return requested_quantity
        
        # Liquidity-based fill ratio
        available_liquidity = self.current_liquidity[asset]
        liquidity_ratio = min(1.0, available_liquidity / (requested_quantity * 10.0))
        
        # Random execution variation
        execution_randomness = np.random.uniform(0.8, 1.0)
        
        # Calculate fill ratio
        fill_ratio = liquidity_ratio * execution_randomness
        fill_ratio = max(self.config.min_fill_ratio, fill_ratio)
        fill_ratio = min(1.0, fill_ratio)
        
        return float(requested_quantity * fill_ratio)
    
    def _calculate_trading_fees(self, asset: str, quantity: float, price: float) -> float:
        """Calculate trading fees"""
        
        trade_value = quantity * price
        
        # Simple fee structure - could be made more sophisticated
        fee_rate = 0.001  # 0.1% fee
        return float(trade_value * fee_rate)
    
    def _check_limit_order_execution(self, current_price: float, limit_price: float, side: str) -> bool:
        """Check if limit order should be executed"""
        
        if side == "buy":
            return current_price <= limit_price
        else:
            return current_price >= limit_price
    
    def _simulate_latency(self) -> float:
        """Simulate execution latency"""
        
        base_latency = self.config.base_latency_ms
        variance = self.config.latency_variance
        
        # Log-normal distribution for realistic latency
        latency = np.random.lognormal(np.log(base_latency), variance)
        
        return float(latency)
    
    def _update_market_state(self, execution: OrderExecution, current_prices: Dict[str, float]) -> None:
        """Update market state after execution"""
        
        asset = execution.asset
        quantity = execution.filled_quantity
        
        # Update liquidity
        liquidity_impact = quantity * 0.1  # Simple liquidity consumption
        self.current_liquidity[asset] -= liquidity_impact
        
        # Regenerate liquidity
        base_liquidity = self.config.base_liquidity.get(asset, 1000.0)
        regeneration = (base_liquidity - self.current_liquidity[asset]) * self.config.liquidity_regeneration_rate
        self.current_liquidity[asset] += regeneration
        
        # Update temporary impact
        if execution.market_impact > 0:
            self.temporary_impacts[asset].append(execution.market_impact)
        
        # Update permanent impact
        permanent_component = execution.market_impact * self.config.permanent_impact_factor
        self.permanent_impacts[asset] += permanent_component
        
        # Decay temporary impacts
        for impact_list in self.temporary_impacts.values():
            if impact_list:
                for i in range(len(impact_list)):
                    impact_list[i] *= self.config.temporary_impact_decay
        
        # Update volume tracking
        self.total_volume[asset] += execution.filled_quantity
    
    def _calculate_portfolio_updates(
        self,
        executions: List[OrderExecution],
        current_portfolio: Dict[str, float],
        current_balance: float
    ) -> Dict[str, Any]:
        """Calculate portfolio changes from executions"""
        
        new_portfolio = current_portfolio.copy()
        new_balance = current_balance
        position_changes = defaultdict(float)
        
        for execution in executions:
            if execution.status in ["filled", "partial"]:
                asset = execution.asset
                quantity = execution.filled_quantity
                total_cost = quantity * execution.avg_fill_price + execution.total_fees
                
                if execution.side == "buy":
                    new_portfolio[asset] = new_portfolio.get(asset, 0.0) + quantity
                    new_balance -= total_cost
                    position_changes[asset] += quantity
                else:  # sell
                    new_portfolio[asset] = new_portfolio.get(asset, 0.0) - quantity
                    new_balance += total_cost - execution.total_fees  # Fees already included
                    position_changes[asset] -= quantity
        
        return {
            "new_portfolio": new_portfolio,
            "new_balance": new_balance,
            "position_changes": dict(position_changes)
        }
    
    def _create_execution_summary(self, executions: List[OrderExecution]) -> Dict[str, Any]:
        """Create execution summary statistics"""
        
        if not executions:
            return {}
        
        total_executed = len([e for e in executions if e.status == "filled"])
        total_partial = len([e for e in executions if e.status == "partial"])
        total_rejected = len([e for e in executions if e.status == "rejected"])
        
        avg_execution_time = np.mean([e.execution_time for e in executions])
        avg_slippage = np.mean([e.slippage for e in executions if e.status != "rejected"])
        avg_impact = np.mean([e.market_impact for e in executions if e.status != "rejected"])
        
        return {
            "total_orders": len(executions),
            "filled_orders": total_executed,
            "partial_orders": total_partial,
            "rejected_orders": total_rejected,
            "fill_rate": (total_executed + total_partial) / len(executions),
            "avg_execution_time": float(avg_execution_time),
            "avg_slippage": float(avg_slippage) if not np.isnan(avg_slippage) else 0.0,
            "avg_market_impact": float(avg_impact) if not np.isnan(avg_impact) else 0.0
        }
    
    def _create_rejected_execution(self, order: Dict[str, Any], reason: str) -> OrderExecution:
        """Create rejected order execution"""
        
        return OrderExecution(
            order_id=f"rejected_{self.order_id_counter}",
            asset=order.get("asset", "unknown"),
            side=order.get("side", "unknown"),
            requested_quantity=order.get("quantity", 0.0),
            filled_quantity=0.0,
            avg_fill_price=0.0,
            total_fees=0.0,
            slippage=0.0,
            market_impact=0.0,
            execution_time=0.0,
            status="rejected"
        )
    
    def get_market_state(self) -> Dict[str, Any]:
        """Get current market state"""
        
        return {
            "current_liquidity": self.current_liquidity.copy(),
            "temporary_impacts": {k: list(v) for k, v in self.temporary_impacts.items()},
            "permanent_impacts": self.permanent_impacts.copy(),
            "total_volume": dict(self.total_volume),
            "asset_volatilities": self.asset_volatilities.copy(),
            "current_regime": self.current_regime,
            "orders_processed": len(self.order_history),
            "avg_execution_time": np.mean([o.execution_time for o in self.order_history]) if self.order_history else 0.0
        }
    
    def reset(self) -> None:
        """Reset simulator state"""
        
        self._initialize_market_state()
        self.pending_orders.clear()
        self.order_history.clear()
        self.order_id_counter = 0
        self.temporary_impacts.clear()
        self.permanent_impacts.clear()
        self.total_volume.clear()
        self.impact_costs.clear()
        self.execution_stats.clear()


__all__ = [
    "MarketImpactModel",
    "LiquidityRegime",
    "MarketSimulatorConfig",
    "OrderExecution", 
    "MarketSimulator"
]