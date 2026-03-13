"""
Advanced Crypto Trading Environment with Sentiment Analysis
enterprise patterns for production trading systems

Features:
- Multi-asset trading simulation with realistic market dynamics
- Sentiment-based trading signals integration
- Order book simulation and market microstructure
- Advanced risk management with circuit breakers
- Real-time data streaming support
- Multi-exchange integration ready
- ML-powered market regime detection
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import deque
from enum import Enum
import numpy as np
import pandas as pd
from gymnasium import spaces

from .base_trading_env import BaseTradingEnvironment, BaseTradingConfig
from ..spaces.observations import CryptoObservationSpace
from ..spaces.actions import CryptoActionSpace
from ..simulation.market_simulator import MarketSimulator
from ..simulation.order_book import OrderBookSimulator
from ..data.data_stream import CryptoDataStream
from ..utils.indicators import TechnicalIndicators
from ..utils.sentiment import SentimentAnalyzer
from ..utils.regime_detection import MarketRegimeDetector


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class CryptoTradingConfig(BaseTradingConfig):
    """Configuration for crypto trading environment"""
    
    # Assets configuration
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "BNB", "ADA", "SOL"])
    base_currency: str = "USDT"
    
    # Market data
    data_source: str = "synthetic"  # synthetic, historical, binance, coinbase
    historical_data_path: Optional[str] = None
    real_time_data: bool = False
    tick_frequency: str = "1m"  # 1s, 1m, 5m, 1h
    
    # Market simulation
    enable_order_book: bool = True
    order_book_depth: int = 10
    market_impact_model: str = "sqrt"  # linear, sqrt, logarithmic
    
    # Sentiment analysis
    enable_sentiment_signals: bool = True
    sentiment_sources: List[str] = field(default_factory=lambda: [
        "twitter", "reddit", "news", "fear_greed_index"
    ])
    sentiment_lag: int = 0  # Steps delay for sentiment data
    sentiment_weight: float = 0.2
    
    # Advanced features
    enable_futures_trading: bool = False
    enable_options_trading: bool = False
    enable_margin_trading: bool = True
    enable_staking: bool = False
    
    # Market regime detection
    regime_detection_window: int = 50
    regime_features: List[str] = field(default_factory=lambda: [
        "volatility", "volume", "price_momentum", "correlation"
    ])
    
    # Risk management
    position_limits: Dict[str, float] = field(default_factory=dict)
    correlation_limit: float = 0.8
    sector_exposure_limit: float = 0.6
    
    # Performance optimization
    parallel_asset_processing: bool = True
    cache_indicators: bool = True
    prefetch_data: bool = True


class CryptoTradingEnvironment(BaseTradingEnvironment):
    """
    Advanced crypto trading environment with sentiment analysis
    
    Implements sophisticated trading simulation for crypto markets
    with enterprise-grade patterns and  best practices
    """
    
    def __init__(
        self,
        config: Optional[CryptoTradingConfig] = None,
        **kwargs
    ):
        self.crypto_config = config or CryptoTradingConfig()
        super().__init__(self.crypto_config, **kwargs)
        
        # Initialize components
        self._setup_crypto_components()
        
        # Market state
        self.current_prices = {}
        self.price_changes = {}
        self.volumes = {}
        self.market_regime = MarketRegime.SIDEWAYS
        
        # Sentiment state
        self.sentiment_scores = {}
        self.sentiment_history = deque(maxlen=100)
        
        # Technical indicators
        self.indicators = {}
        self.indicator_history = {}
        
        # Market microstructure
        self.order_books = {}
        self.trade_flows = {}
        
        self.logger.info("Crypto trading environment initialized", extra={
            "assets": self.crypto_config.assets,
            "enable_sentiment": self.crypto_config.enable_sentiment_signals,
            "enable_order_book": self.crypto_config.enable_order_book,
            "data_source": self.crypto_config.data_source
        })
    
    def _setup_crypto_components(self) -> None:
        """Setup crypto-specific components"""
        
        # Market simulator
        self.market_simulator = MarketSimulator(
            assets=self.crypto_config.assets,
            impact_model=self.crypto_config.market_impact_model
        )
        
        # Order book simulator
        if self.crypto_config.enable_order_book:
            self.order_book_simulator = OrderBookSimulator(
                assets=self.crypto_config.assets,
                depth=self.crypto_config.order_book_depth
            )
        
        # Data stream
        self.data_stream = CryptoDataStream(
            assets=self.crypto_config.assets,
            source=self.crypto_config.data_source,
            frequency=self.crypto_config.tick_frequency
        )
        
        # Technical indicators
        self.technical_indicators = TechnicalIndicators(
            indicators=self.crypto_config.technical_indicators,
            cache_enabled=self.crypto_config.cache_indicators
        )
        
        # Sentiment analyzer
        if self.crypto_config.enable_sentiment_signals:
            self.sentiment_analyzer = SentimentAnalyzer(
                sources=self.crypto_config.sentiment_sources,
                assets=self.crypto_config.assets
            )
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(
            window_size=self.crypto_config.regime_detection_window,
            features=self.crypto_config.regime_features
        )
    
    def _setup_spaces(self) -> None:
        """Setup observation and action spaces"""
        
        # Create observation space
        self.obs_space_builder = CryptoObservationSpace(self.crypto_config)
        self.observation_space = self.obs_space_builder.create_space()
        
        # Create action space
        self.action_space_builder = CryptoActionSpace(self.crypto_config)
        self.action_space = self.action_space_builder.create_space()
        
        self.logger.debug("Spaces configured", extra={
            "observation_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.shape[0] if hasattr(self.action_space, 'shape') else 'discrete'
        })
    
    def _initialize_market_data(self) -> None:
        """Initialize market data"""
        
        try:
            # Initialize data stream
            initial_data = self.data_stream.initialize()
            
            # Setup initial prices
            for asset in self.crypto_config.assets:
                self.current_prices[asset] = initial_data.get(f"{asset}_price", 100.0)
                self.price_changes[asset] = 0.0
                self.volumes[asset] = initial_data.get(f"{asset}_volume", 1000.0)
            
            # Initialize order books
            if hasattr(self, 'order_book_simulator'):
                self.order_book_simulator.initialize(self.current_prices)
            
            # Initialize indicators
            self._initialize_indicators()
            
            # Initialize sentiment
            if hasattr(self, 'sentiment_analyzer'):
                self._initialize_sentiment()
            
            self.logger.debug("Market data initialized", extra={
                "initial_prices": self.current_prices,
                "num_assets": len(self.crypto_config.assets)
            })
            
        except Exception as e:
            self.logger.error(f"Error initializing market data: {e}", exc_info=True)
            # Fallback to synthetic data
            self._initialize_synthetic_data()
    
    def _initialize_synthetic_data(self) -> None:
        """Initialize with synthetic market data"""
        
        base_prices = {
            "BTC": 50000.0, "ETH": 3000.0, "BNB": 400.0,
            "ADA": 1.0, "SOL": 100.0, "DOT": 30.0, "MATIC": 1.5
        }
        
        for asset in self.crypto_config.assets:
            self.current_prices[asset] = base_prices.get(asset, 100.0)
            self.price_changes[asset] = 0.0
            self.volumes[asset] = np.random.uniform(1000, 10000)
    
    def _initialize_indicators(self) -> None:
        """Initialize technical indicators"""
        
        for asset in self.crypto_config.assets:
            # Generate initial price history
            initial_price = self.current_prices[asset]
            price_history = []
            
            # Create realistic price history
            price = initial_price
            for _ in range(100):
                change = np.random.normal(0, 0.02)
                price *= (1 + change)
                price_history.append(price)
            
            # Calculate initial indicators
            self.indicators[asset] = self.technical_indicators.calculate(
                prices=price_history,
                volumes=[self.volumes[asset]] * 100
            )
            
            self.indicator_history[asset] = deque(maxlen=200)
    
    def _initialize_sentiment(self) -> None:
        """Initialize sentiment analysis"""
        
        try:
            # Get initial sentiment scores
            sentiment_data = self.sentiment_analyzer.get_current_sentiment()
            
            for asset in self.crypto_config.assets:
                self.sentiment_scores[asset] = sentiment_data.get(asset, {
                    "overall": 0.0,
                    "twitter": 0.0,
                    "reddit": 0.0,
                    "news": 0.0,
                    "fear_greed": 50.0
                })
            
        except Exception as e:
            self.logger.warning(f"Error initializing sentiment: {e}")
            # Use neutral sentiment
            for asset in self.crypto_config.assets:
                self.sentiment_scores[asset] = {
                    "overall": 0.0, "twitter": 0.0, "reddit": 0.0,
                    "news": 0.0, "fear_greed": 50.0
                }
    
    def _update_market_data(self) -> Dict[str, Any]:
        """Update market data for current step"""
        
        try:
            # Get new market data
            market_data = self.data_stream.get_next_tick()
            
            # Update prices and volumes
            price_updates = {}
            for asset in self.crypto_config.assets:
                old_price = self.current_prices[asset]
                new_price = market_data.get(f"{asset}_price", old_price)
                
                self.current_prices[asset] = new_price
                self.price_changes[asset] = (new_price - old_price) / old_price
                self.volumes[asset] = market_data.get(f"{asset}_volume", self.volumes[asset])
                
                price_updates[asset] = new_price
            
            # Update order books
            if hasattr(self, 'order_book_simulator'):
                self.order_book_simulator.update(price_updates)
                self.order_books = self.order_book_simulator.get_order_books()
            
            # Update technical indicators
            self._update_indicators()
            
            # Update sentiment (with lag if configured)
            if hasattr(self, 'sentiment_analyzer') and self.current_step % 10 == 0:
                self._update_sentiment()
            
            # Update market regime
            self._update_market_regime()
            
            return {
                "prices": self.current_prices.copy(),
                "price_changes": self.price_changes.copy(),
                "volumes": self.volumes.copy(),
                "market_regime": self.market_regime.value,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _update_indicators(self) -> None:
        """Update technical indicators"""
        
        for asset in self.crypto_config.assets:
            try:
                # Get recent price history
                if len(self.price_history) >= 20:
                    recent_prices = [p[asset] for p in list(self.price_history)[-50:]]
                    recent_volumes = [self.volumes[asset]] * len(recent_prices)
                    
                    # Calculate indicators
                    new_indicators = self.technical_indicators.calculate(
                        prices=recent_prices,
                        volumes=recent_volumes
                    )
                    
                    self.indicators[asset] = new_indicators
                    self.indicator_history[asset].append(new_indicators)
                
            except Exception as e:
                self.logger.warning(f"Error updating indicators for {asset}: {e}")
    
    def _update_sentiment(self) -> None:
        """Update sentiment scores"""
        
        try:
            if hasattr(self, 'sentiment_analyzer'):
                sentiment_data = self.sentiment_analyzer.get_current_sentiment()
                
                for asset in self.crypto_config.assets:
                    self.sentiment_scores[asset] = sentiment_data.get(asset, 
                                                                    self.sentiment_scores[asset])
                
                # Store sentiment history
                self.sentiment_history.append({
                    "step": self.current_step,
                    "sentiment": self.sentiment_scores.copy(),
                    "timestamp": time.time()
                })
                
        except Exception as e:
            self.logger.warning(f"Error updating sentiment: {e}")
    
    def _update_market_regime(self) -> None:
        """Update market regime detection"""
        
        try:
            if len(self.price_history) >= self.crypto_config.regime_detection_window:
                # Prepare data for regime detection
                regime_data = self._prepare_regime_data()
                
                # Detect regime
                detected_regime = self.regime_detector.detect_regime(regime_data)
                
                # Update regime with smoothing
                if detected_regime != self.market_regime:
                    self.market_regime = detected_regime
                    self.logger.info(f"Market regime changed to: {detected_regime.value}")
                
        except Exception as e:
            self.logger.warning(f"Error updating market regime: {e}")
    
    def _prepare_regime_data(self) -> Dict[str, np.ndarray]:
        """Prepare data for regime detection"""
        
        recent_history = list(self.price_history)[-self.crypto_config.regime_detection_window:]
        
        regime_data = {}
        
        # Price volatility
        for asset in self.crypto_config.assets:
            prices = [h[asset] for h in recent_history]
            returns = np.diff(np.log(prices))
            regime_data[f"{asset}_volatility"] = np.std(returns)
        
        # Volume data
        regime_data["avg_volume"] = np.mean([self.volumes[asset] 
                                           for asset in self.crypto_config.assets])
        
        # Price momentum
        for asset in self.crypto_config.assets:
            prices = [h[asset] for h in recent_history]
            if len(prices) >= 10:
                momentum = (prices[-1] - prices[-10]) / prices[-10]
                regime_data[f"{asset}_momentum"] = momentum
        
        # Cross-correlation
        if len(self.crypto_config.assets) >= 2:
            asset1, asset2 = self.crypto_config.assets[0], self.crypto_config.assets[1]
            prices1 = [h[asset1] for h in recent_history]
            prices2 = [h[asset2] for h in recent_history]
            
            if len(prices1) >= 20 and len(prices2) >= 20:
                returns1 = np.diff(np.log(prices1))
                returns2 = np.diff(np.log(prices2))
                correlation = np.corrcoef(returns1, returns2)[0, 1]
                regime_data["correlation"] = correlation if not np.isnan(correlation) else 0.0
        
        return regime_data
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with sentiment and regime data"""
        
        return self.obs_space_builder.build_observation(
            prices=self.current_prices,
            price_changes=self.price_changes,
            volumes=self.volumes,
            indicators=self.indicators,
            sentiment_scores=self.sentiment_scores if hasattr(self, 'sentiment_analyzer') else {},
            order_books=getattr(self, 'order_books', {}),
            market_regime=self.market_regime,
            portfolio_state={
                "balance": self.balance,
                "positions": self.positions,
                "portfolio_value": self.portfolio_value,
                "max_drawdown": self.max_drawdown
            },
            step=self.current_step,
            max_steps=self.crypto_config.max_steps
        )
    
    def _execute_action(self, action: Union[np.ndarray, int]) -> Dict[str, Any]:
        """Execute crypto trading action"""
        
        try:
            # Parse action
            parsed_action = self.action_space_builder.parse_action(action)
            
            # Execute trades through market simulator
            execution_result = self.market_simulator.execute_trades(
                orders=parsed_action["orders"],
                current_prices=self.current_prices,
                portfolio=self.positions,
                balance=self.balance,
                order_books=getattr(self, 'order_books', {})
            )
            
            # Update positions and balance
            self.balance = execution_result["new_balance"]
            self.positions.update(execution_result["position_changes"])
            
            # Track execution metrics
            result = {
                "trades_executed": len(execution_result["filled_orders"]),
                "total_fees": execution_result["total_fees"],
                "total_slippage": execution_result["total_slippage"],
                "filled_orders": execution_result["filled_orders"],
                "rejected_orders": execution_result["rejected_orders"],
                "market_impact": execution_result.get("market_impact", 0.0)
            }
            
            # Log significant trades
            if result["trades_executed"] > 0:
                self.logger.debug("Trades executed", extra={
                    "step": self.current_step,
                    "trades": result["trades_executed"],
                    "fees": result["total_fees"],
                    "slippage": result["total_slippage"]
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing action: {e}", exc_info=True)
            return {
                "trades_executed": 0,
                "total_fees": 0.0,
                "total_slippage": 0.0,
                "filled_orders": [],
                "rejected_orders": [],
                "error": str(e)
            }
    
    def _calculate_reward(self, action_result: Dict[str, Any]) -> float:
        """Calculate reward with sentiment and regime factors"""
        
        # Base portfolio return reward
        if len(self.portfolio_history) > 0:
            prev_value = self.portfolio_history[-1]["portfolio_value"]
            portfolio_return = (self.portfolio_value - prev_value) / prev_value
        else:
            portfolio_return = 0.0
        
        reward = portfolio_return * 100.0  # Scale up
        
        # Risk-adjusted reward
        if self.max_drawdown > 0:
            risk_penalty = self.max_drawdown * 10.0
            reward -= risk_penalty
        
        # Transaction costs
        fees_penalty = action_result.get("total_fees", 0.0) / self.portfolio_value
        reward -= fees_penalty * 100.0
        
        # Sentiment alignment bonus
        if hasattr(self, 'sentiment_analyzer') and self.crypto_config.enable_sentiment_signals:
            sentiment_bonus = self._calculate_sentiment_bonus(action_result)
            reward += sentiment_bonus * self.crypto_config.sentiment_weight
        
        # Market regime alignment
        regime_bonus = self._calculate_regime_bonus(action_result)
        reward += regime_bonus
        
        # Diversification bonus
        active_positions = sum(1 for pos in self.positions.values() if abs(pos) > 1e-6)
        if active_positions > 1:
            diversification_bonus = 0.1 * (active_positions - 1)
            reward += diversification_bonus
        
        # Liquidity provision bonus (if order book trading)
        if hasattr(self, 'order_book_simulator'):
            liquidity_bonus = self._calculate_liquidity_bonus(action_result)
            reward += liquidity_bonus
        
        return float(reward)
    
    def _calculate_sentiment_bonus(self, action_result: Dict[str, Any]) -> float:
        """Calculate sentiment alignment bonus"""
        
        if not action_result.get("filled_orders"):
            return 0.0
        
        sentiment_bonus = 0.0
        
        for order in action_result["filled_orders"]:
            asset = order["asset"]
            side = order["side"]  # buy/sell
            quantity = order["quantity"]
            
            if asset in self.sentiment_scores:
                sentiment = self.sentiment_scores[asset]["overall"]
                
                # Reward alignment with positive sentiment
                if side == "buy" and sentiment > 0:
                    sentiment_bonus += sentiment * quantity * 0.1
                elif side == "sell" and sentiment < 0:
                    sentiment_bonus += abs(sentiment) * quantity * 0.1
                # Penalize contrarian moves
                elif side == "buy" and sentiment < -0.5:
                    sentiment_bonus -= abs(sentiment) * quantity * 0.05
                elif side == "sell" and sentiment > 0.5:
                    sentiment_bonus -= sentiment * quantity * 0.05
        
        return sentiment_bonus
    
    def _calculate_regime_bonus(self, action_result: Dict[str, Any]) -> float:
        """Calculate market regime alignment bonus"""
        
        if not action_result.get("filled_orders"):
            return 0.0
        
        regime_bonus = 0.0
        total_trade_value = sum(order["quantity"] * order["price"] 
                              for order in action_result["filled_orders"])
        
        # Reward regime-appropriate strategies
        if self.market_regime == MarketRegime.BULL:
            # Reward long positions in bull market
            long_value = sum(order["quantity"] * order["price"] 
                           for order in action_result["filled_orders"] 
                           if order["side"] == "buy")
            regime_bonus += (long_value / max(total_trade_value, 1)) * 0.2
            
        elif self.market_regime == MarketRegime.BEAR:
            # Reward defensive positions in bear market
            short_value = sum(order["quantity"] * order["price"] 
                            for order in action_result["filled_orders"] 
                            if order["side"] == "sell")
            regime_bonus += (short_value / max(total_trade_value, 1)) * 0.2
            
        elif self.market_regime == MarketRegime.VOLATILE:
            # Reward active trading in volatile markets
            if len(action_result["filled_orders"]) > 2:
                regime_bonus += 0.1
                
        return regime_bonus
    
    def _calculate_liquidity_bonus(self, action_result: Dict[str, Any]) -> float:
        """Calculate liquidity provision bonus"""
        
        # Simple liquidity bonus for market making
        liquidity_bonus = 0.0
        
        for order in action_result.get("filled_orders", []):
            if order.get("order_type") == "limit":
                # Small bonus for limit orders (providing liquidity)
                liquidity_bonus += 0.01
            elif order.get("order_type") == "market":
                # Small penalty for market orders (taking liquidity)
                liquidity_bonus -= 0.005
        
        return liquidity_bonus
    
    async def async_step(
        self, 
        action: Union[np.ndarray, int, List[float]]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Async version for real-time trading"""
        
        # Update sentiment asynchronously
        if hasattr(self, 'sentiment_analyzer'):
            sentiment_task = asyncio.create_task(
                self.sentiment_analyzer.async_get_sentiment()
            )
        
        # Process step
        result = self.step(action)
        
        # Wait for sentiment update
        if hasattr(self, 'sentiment_analyzer'):
            try:
                await sentiment_task
            except Exception as e:
                self.logger.warning(f"Async sentiment update failed: {e}")
        
        return result
    
    def get_market_info(self) -> Dict[str, Any]:
        """Get comprehensive market information"""
        
        return {
            "current_prices": self.current_prices.copy(),
            "price_changes": self.price_changes.copy(),
            "volumes": self.volumes.copy(),
            "market_regime": self.market_regime.value,
            "sentiment_scores": getattr(self, 'sentiment_scores', {}),
            "technical_indicators": self.indicators.copy(),
            "order_books": getattr(self, 'order_books', {}),
            "portfolio_summary": {
                "balance": self.balance,
                "positions": self.positions.copy(),
                "portfolio_value": self.portfolio_value,
                "total_return": self.total_return,
                "max_drawdown": self.max_drawdown,
                "sharpe_ratio": self.sharpe_ratio
            }
        }


# Factory functions
def create_crypto_env(
    config: Optional[CryptoTradingConfig] = None,
    **kwargs
) -> CryptoTradingEnvironment:
    """Create crypto trading environment"""
    return CryptoTradingEnvironment(config, **kwargs)


def create_sentiment_crypto_env(
    assets: List[str] = None,
    sentiment_sources: List[str] = None,
    **kwargs
) -> CryptoTradingEnvironment:
    """Create crypto environment with sentiment analysis"""
    
    config = CryptoTradingConfig(
        assets=assets or ["BTC", "ETH", "BNB"],
        enable_sentiment_signals=True,
        sentiment_sources=sentiment_sources or ["twitter", "reddit", "news"],
        **kwargs
    )
    
    return CryptoTradingEnvironment(config)


__all__ = [
    "MarketRegime",
    "CryptoTradingConfig",
    "CryptoTradingEnvironment",
    "create_crypto_env",
    "create_sentiment_crypto_env"
]