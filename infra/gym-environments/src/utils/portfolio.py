"""
Portfolio Management Utilities
enterprise patterns for sophisticated portfolio management

Advanced portfolio management features:
- Position tracking and validation
- Risk-based position sizing
- Portfolio rebalancing algorithms
- Performance attribution analysis
- Real-time portfolio monitoring
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import time
from enum import Enum

from .risk_metrics import RiskCalculator, PositionSizer


class PortfolioStatus(Enum):
    """Portfolio status states"""
    NORMAL = "normal"
    WARNING = "warning"  
    CRITICAL = "critical"
    LIQUIDATION = "liquidation"


@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    
    # Basic metrics
    total_value: float = 0.0
    balance: float = 0.0
    invested_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Performance metrics
    total_return: float = 0.0
    daily_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk metrics
    portfolio_beta: float = 0.0
    tracking_error: float = 0.0
    var_95: float = 0.0
    expected_shortfall: float = 0.0
    
    # Concentration metrics
    largest_position_weight: float = 0.0
    top_5_concentration: float = 0.0
    effective_diversification: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0


@dataclass
class PortfolioConfig:
    """Portfolio management configuration"""
    
    # Risk limits
    max_position_weight: float = 0.2        # Max 20% per position
    max_sector_weight: float = 0.4          # Max 40% per sector
    max_correlation_exposure: float = 0.6   # Max 60% in correlated assets
    max_leverage: float = 1.0               # Max leverage ratio
    max_drawdown_limit: float = 0.15        # Max 15% drawdown
    
    # Rebalancing
    rebalancing_threshold: float = 0.05     # 5% threshold for rebalancing
    rebalancing_frequency: int = 24         # Hours between rebalancing checks
    min_trade_size: float = 10.0            # Minimum trade size ($)
    
    # Position sizing
    enable_position_sizing: bool = True
    position_sizing_method: str = "kelly"
    base_position_size: float = 0.02        # 2% risk per position
    
    # Portfolio optimization
    enable_portfolio_optimization: bool = False
    optimization_method: str = "mean_variance"  # mean_variance, risk_parity, black_litterman
    rebalancing_costs: float = 0.001        # 0.1% rebalancing cost
    
    # Monitoring
    enable_real_time_monitoring: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "drawdown": 0.10,        # Alert at 10% drawdown
        "position_weight": 0.25,  # Alert if position > 25%
        "correlation": 0.8,       # Alert if correlation > 0.8
        "leverage": 0.9          # Alert if leverage > 90%
    })


class PortfolioManager:
    """
    Enterprise-grade portfolio management system
    
    Manages portfolio state, positions, risk metrics, and optimization
    """
    
    def __init__(
        self,
        initial_balance: float,
        assets: Optional[List[str]] = None,
        config: Optional[PortfolioConfig] = None
    ):
        self.initial_balance = initial_balance
        self.assets = assets or []
        self.config = config or PortfolioConfig()
        self.logger = logging.getLogger(__name__)
        
        # Portfolio state
        self.balance = initial_balance
        self.positions = {}  # asset -> quantity
        self.avg_entry_prices = {}  # asset -> average entry price
        self.position_history = deque(maxlen=1000)
        
        # Performance tracking
        self.portfolio_value_history = deque(maxlen=1000)
        self.trade_history = []
        self.rebalancing_history = []
        
        # Risk management
        self.risk_calculator = RiskCalculator()
        if config and config.enable_position_sizing:
            self.position_sizer = PositionSizer(
                method=config.position_sizing_method,
                max_risk=config.base_position_size
            )
        
        # Monitoring state
        self.last_rebalance_time = time.time()
        self.alerts_triggered = []
        self.portfolio_status = PortfolioStatus.NORMAL
        
        self.logger.info(f"Portfolio manager initialized with balance ${initial_balance:,.2f}")
    
    def update_position(
        self,
        asset: str,
        quantity_change: float,
        price: float,
        fees: float = 0.0,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Update position after trade execution"""
        
        timestamp = timestamp or time.time()
        
        # Current position
        current_position = self.positions.get(asset, 0.0)
        new_position = current_position + quantity_change
        
        # Calculate trade value
        trade_value = abs(quantity_change) * price
        
        # Update balance
        if quantity_change > 0:  # Buy
            self.balance -= trade_value + fees
        else:  # Sell
            self.balance += trade_value - fees
        
        # Update position
        if abs(new_position) < 1e-8:  # Close to zero
            self.positions.pop(asset, None)
            self.avg_entry_prices.pop(asset, None)
        else:
            self.positions[asset] = new_position
            
            # Update average entry price
            if quantity_change > 0:  # Adding to position
                if asset in self.avg_entry_prices:
                    total_cost = (current_position * self.avg_entry_prices[asset] + 
                                quantity_change * price)
                    self.avg_entry_prices[asset] = total_cost / new_position
                else:
                    self.avg_entry_prices[asset] = price
        
        # Record trade
        trade_record = {
            "timestamp": timestamp,
            "asset": asset,
            "quantity": quantity_change,
            "price": price,
            "fees": fees,
            "balance_after": self.balance,
            "position_after": new_position
        }
        
        self.trade_history.append(trade_record)
        
        # Update position history
        self.position_history.append({
            "timestamp": timestamp,
            "positions": self.positions.copy(),
            "balance": self.balance
        })
        
        return trade_record
    
    def update_market_data(
        self,
        current_prices: Dict[str, float],
        timestamp: Optional[float] = None
    ) -> None:
        """Update with current market prices"""
        
        timestamp = timestamp or time.time()
        
        # Calculate current portfolio value
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        # Store in history
        self.portfolio_value_history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "prices": current_prices.copy()
        })
        
        # Check for alerts
        if self.config.enable_real_time_monitoring:
            self._check_alerts(current_prices, portfolio_value)
        
        # Check for rebalancing
        if self._should_rebalance(timestamp):
            self._suggest_rebalancing(current_prices)
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        
        portfolio_value = self.balance
        
        for asset, quantity in self.positions.items():
            if asset in current_prices:
                portfolio_value += quantity * current_prices[asset]
        
        return portfolio_value
    
    def calculate_portfolio_weights(self, current_prices: Dict[str, float]) -> Dict[str, float]:
        """Calculate portfolio weights"""
        
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        if portfolio_value <= 0:
            return {}
        
        weights = {}
        for asset, quantity in self.positions.items():
            if asset in current_prices:
                position_value = quantity * current_prices[asset]
                weights[asset] = position_value / portfolio_value
        
        return weights
    
    def calculate_portfolio_metrics(
        self,
        current_prices: Dict[str, float],
        benchmark_returns: Optional[List[float]] = None
    ) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if not self.portfolio_value_history:
            return PortfolioMetrics()
        
        # Basic calculations
        current_value = self.calculate_portfolio_value(current_prices)
        invested_value = current_value - self.balance
        unrealized_pnl = self._calculate_unrealized_pnl(current_prices)
        realized_pnl = self._calculate_realized_pnl()
        
        # Performance calculations
        returns = self._calculate_returns()
        
        if len(returns) > 1:
            total_return = (current_value - self.initial_balance) / self.initial_balance
            daily_return = returns[-1] if returns else 0.0
            annualized_return = np.mean(returns) * 252 if returns else 0.0
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0
            
            # Risk-adjusted metrics
            risk_metrics = self.risk_calculator.calculate_risk_metrics(
                returns, benchmark_returns
            )
            
            sharpe_ratio = risk_metrics.sharpe_ratio
            max_drawdown = risk_metrics.max_drawdown
            var_95 = risk_metrics.var_95
            expected_shortfall = risk_metrics.expected_shortfall_95
        else:
            total_return = daily_return = annualized_return = 0.0
            volatility = sharpe_ratio = max_drawdown = 0.0
            var_95 = expected_shortfall = 0.0
        
        # Concentration metrics
        weights = self.calculate_portfolio_weights(current_prices)
        largest_position = max(weights.values()) if weights else 0.0
        top_5_positions = sorted(weights.values(), reverse=True)[:5]
        top_5_concentration = sum(top_5_positions)
        
        # Effective diversification (Herfindahl index)
        herfindahl_index = sum(w**2 for w in weights.values()) if weights else 1.0
        effective_diversification = 1.0 / herfindahl_index if herfindahl_index > 0 else 1.0
        
        # Trading metrics
        winning_trades = [t for t in self.trade_history if self._is_winning_trade(t, current_prices)]
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0.0
        
        return PortfolioMetrics(
            total_value=current_value,
            balance=self.balance,
            invested_value=invested_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
            total_return=total_return,
            daily_return=daily_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            largest_position_weight=largest_position,
            top_5_concentration=top_5_concentration,
            effective_diversification=effective_diversification,
            total_trades=len(self.trade_history),
            win_rate=win_rate
        )
    
    def suggest_position_size(
        self,
        asset: str,
        signal_strength: float,
        current_price: float,
        current_prices: Dict[str, float]
    ) -> float:
        """Suggest optimal position size"""
        
        if not hasattr(self, 'position_sizer'):
            # Simple position sizing
            portfolio_value = self.calculate_portfolio_value(current_prices)
            max_position_value = portfolio_value * self.config.max_position_weight
            return max_position_value / current_price * abs(signal_strength)
        
        # Advanced position sizing
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        return self.position_sizer.calculate_position_size(
            signal_strength=signal_strength,
            current_price=current_price,
            portfolio_value=portfolio_value
        )
    
    def check_risk_constraints(
        self,
        proposed_trade: Dict[str, Any],
        current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if proposed trade violates risk constraints"""
        
        asset = proposed_trade["asset"]
        quantity = proposed_trade["quantity"]
        
        # Simulate trade
        temp_positions = self.positions.copy()
        temp_positions[asset] = temp_positions.get(asset, 0.0) + quantity
        
        # Calculate new portfolio value
        temp_portfolio_value = self.balance
        for a, q in temp_positions.items():
            temp_portfolio_value += q * current_prices.get(a, 0.0)
        
        violations = []
        
        # Check position weight limit
        if asset in current_prices:
            position_value = temp_positions[asset] * current_prices[asset]
            position_weight = position_value / temp_portfolio_value
            
            if position_weight > self.config.max_position_weight:
                violations.append({
                    "type": "position_weight",
                    "limit": self.config.max_position_weight,
                    "actual": position_weight,
                    "severity": "high"
                })
        
        # Check leverage limit
        total_exposure = sum(abs(q * current_prices.get(a, 0.0)) for a, q in temp_positions.items())
        leverage_ratio = total_exposure / temp_portfolio_value
        
        if leverage_ratio > self.config.max_leverage:
            violations.append({
                "type": "leverage",
                "limit": self.config.max_leverage,
                "actual": leverage_ratio,
                "severity": "high"
            })
        
        return {
            "allowed": len(violations) == 0,
            "violations": violations,
            "risk_score": len([v for v in violations if v["severity"] == "high"])
        }
    
    def _calculate_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L"""
        
        unrealized = 0.0
        
        for asset, quantity in self.positions.items():
            if asset in current_prices and asset in self.avg_entry_prices:
                current_price = current_prices[asset]
                entry_price = self.avg_entry_prices[asset]
                unrealized += quantity * (current_price - entry_price)
        
        return unrealized
    
    def _calculate_realized_pnl(self) -> float:
        """Calculate realized P&L from closed positions"""
        
        # Simplified calculation - would need more sophisticated tracking in production
        realized = 0.0
        
        for trade in self.trade_history:
            if trade["quantity"] < 0:  # Sell trade
                # This is a simplification - real calculation needs position tracking
                realized += abs(trade["quantity"]) * trade["price"]
        
        return realized - sum(trade["fees"] for trade in self.trade_history)
    
    def _calculate_returns(self) -> List[float]:
        """Calculate portfolio returns"""
        
        if len(self.portfolio_value_history) < 2:
            return []
        
        returns = []
        values = [entry["portfolio_value"] for entry in self.portfolio_value_history]
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(ret)
        
        return returns
    
    def _is_winning_trade(self, trade: Dict[str, Any], current_prices: Dict[str, float]) -> bool:
        """Check if trade is winning (simplified)"""
        
        asset = trade["asset"]
        if asset not in current_prices:
            return False
        
        entry_price = trade["price"]
        current_price = current_prices[asset]
        
        if trade["quantity"] > 0:  # Long position
            return current_price > entry_price
        else:  # Short position
            return current_price < entry_price
    
    def _check_alerts(self, current_prices: Dict[str, float], portfolio_value: float) -> None:
        """Check for risk alerts"""
        
        current_time = time.time()
        
        # Drawdown alert
        if self.portfolio_value_history:
            max_value = max(entry["portfolio_value"] for entry in self.portfolio_value_history)
            drawdown = (max_value - portfolio_value) / max_value
            
            if drawdown > self.config.alert_thresholds["drawdown"]:
                self._trigger_alert("drawdown", drawdown, current_time)
        
        # Position concentration alert
        weights = self.calculate_portfolio_weights(current_prices)
        max_weight = max(weights.values()) if weights else 0.0
        
        if max_weight > self.config.alert_thresholds["position_weight"]:
            self._trigger_alert("position_concentration", max_weight, current_time)
        
        # Update portfolio status
        if drawdown > 0.20:
            self.portfolio_status = PortfolioStatus.CRITICAL
        elif drawdown > 0.15:
            self.portfolio_status = PortfolioStatus.WARNING
        else:
            self.portfolio_status = PortfolioStatus.NORMAL
    
    def _trigger_alert(self, alert_type: str, value: float, timestamp: float) -> None:
        """Trigger risk alert"""
        
        alert = {
            "type": alert_type,
            "value": value,
            "timestamp": timestamp,
            "message": f"Alert: {alert_type} = {value:.4f}"
        }
        
        self.alerts_triggered.append(alert)
        self.logger.warning(f"Risk alert triggered: {alert['message']}")
    
    def _should_rebalance(self, current_time: float) -> bool:
        """Check if portfolio should be rebalanced"""
        
        time_since_rebalance = current_time - self.last_rebalance_time
        return time_since_rebalance >= (self.config.rebalancing_frequency * 3600)
    
    def _suggest_rebalancing(self, current_prices: Dict[str, float]) -> None:
        """Suggest portfolio rebalancing"""
        
        # Simple rebalancing logic - equal weight all positions
        weights = self.calculate_portfolio_weights(current_prices)
        
        if len(weights) <= 1:
            return
        
        target_weight = 1.0 / len(weights)
        rebalancing_trades = []
        
        portfolio_value = self.calculate_portfolio_value(current_prices)
        
        for asset, current_weight in weights.items():
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > self.config.rebalancing_threshold:
                target_value = portfolio_value * weight_diff
                target_quantity = target_value / current_prices[asset]
                
                rebalancing_trades.append({
                    "asset": asset,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "quantity_change": target_quantity
                })
        
        if rebalancing_trades:
            self.rebalancing_history.append({
                "timestamp": time.time(),
                "trades": rebalancing_trades
            })
            
            self.logger.info(f"Rebalancing suggested: {len(rebalancing_trades)} trades")
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        
        metrics = self.calculate_portfolio_metrics(current_prices)
        
        return {
            "portfolio_value": metrics.total_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "weights": self.calculate_portfolio_weights(current_prices),
            "metrics": metrics.__dict__,
            "status": self.portfolio_status.value,
            "active_alerts": len([a for a in self.alerts_triggered if time.time() - a["timestamp"] < 3600]),
            "last_rebalance": self.last_rebalance_time,
            "trade_count": len(self.trade_history)
        }
    
    def reset(self) -> None:
        """Reset portfolio manager state"""
        
        self.balance = self.initial_balance
        self.positions.clear()
        self.avg_entry_prices.clear()
        self.position_history.clear()
        self.portfolio_value_history.clear()
        self.trade_history.clear()
        self.rebalancing_history.clear()
        self.alerts_triggered.clear()
        self.portfolio_status = PortfolioStatus.NORMAL
        self.last_rebalance_time = time.time()
    
    def close(self) -> None:
        """Cleanup portfolio manager"""
        
        self.logger.info("Portfolio manager closed")


__all__ = [
    "PortfolioStatus",
    "PortfolioMetrics", 
    "PortfolioConfig",
    "PortfolioManager"
]