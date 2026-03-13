"""
Meta Policy Implementation for Hierarchical RL
Implements high-level policy for selection strategies and coordination subtasks.

enterprise Pattern:
- Strategic decision making with adaptive strategy selection
- Production-ready meta-learning for dynamic trading environments
- Multi-objective optimization for complex trading scenarios
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import logging
import asyncio
from collections import deque, defaultdict
import pickle
import json

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Types trading strategies"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    PAIRS_TRADING = "pairs_trading"
    BREAKOUT = "breakout"
    SWING_TRADING = "swing_trading"


@dataclass
class StrategyContext:
    """Context for selection strategies"""
    market_state: np.ndarray
    volatility: float
    volume: float
    trend_strength: float
    spread: float
    time_of_day: int
    market_regime: str
    risk_level: float
    portfolio_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySelection:
    """Result selection strategies"""
    strategy_type: StrategyType
    confidence: float
    parameters: Dict[str, Any]
    expected_horizon: int
    risk_assessment: float
    reasoning: str


class MetaPolicyNetwork(nn.Module):
    """Neural network for meta-policy"""
    
    def __init__(self,
                 state_dim: int,
                 num_strategies: int,
                 context_dim: int = 32,
                 hidden_dims: List[int] = [512, 256, 128]):
        super().__init__()
        
        self.num_strategies = num_strategies
        
        # Encoder for state market
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], context_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for important features
        self.attention = nn.MultiheadAttention(context_dim, num_heads=4, batch_first=True)
        
        # Policy head for selection strategies
        self.policy_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], num_strategies)
        )
        
        # Value head for estimation state
        self.value_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        # Confidence head for estimation confidence
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], num_strategies),
            nn.Sigmoid()
        )
        
        # Risk assessment head
        self.risk_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (strategy_logits, value, confidence, risk)
        """
        # Encode state
        encoded = self.state_encoder(state)
        
        # Self-attention for allocation important patterns
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)  # Remove sequence dimension
        
        # Compute outputs
        strategy_logits = self.policy_head(attended)
        value = self.value_head(attended)
        confidence = self.confidence_head(attended)
        risk = self.risk_head(attended)
        
        return strategy_logits, value, confidence, risk


class MetaPolicy:
    """
    Meta Policy for high-level strategic decisions
    Coordinates selection and switching between various trading strategies
    """
    
    def __init__(self,
                 state_dim: int,
                 strategy_registry: Dict[str, Any],
                 learning_rate: float = 1e-4,
                 exploration_epsilon: float = 0.1,
                 confidence_threshold: float = 0.7,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.strategy_registry = strategy_registry
        self.num_strategies = len(strategy_registry)
        self.exploration_epsilon = exploration_epsilon
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device)
        
        # Create network
        self.network = MetaPolicyNetwork(
            state_dim=state_dim,
            num_strategies=self.num_strategies
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Mapping strategies
        self.strategy_names = list(strategy_registry.keys())
        self.strategy_id_to_name = {i: name for i, name in enumerate(self.strategy_names)}
        self.strategy_name_to_id = {name: i for i, name in enumerate(self.strategy_names)}
        
        # History making decisions
        self.decision_history: deque = deque(maxlen=10000)
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'total_reward': 0.0,
            'executions': 0,
            'success_rate': 0.0,
            'avg_duration': 0.0,
            'risk_adjusted_return': 0.0
        })
        
        # Current state
        self.current_strategy: Optional[str] = None
        self.strategy_start_time: Optional[float] = None
        self.episode_rewards: List[float] = []
        
        # Adaptive parameters
        self.adaptation_rate = 0.01
        self.regime_detector = MarketRegimeDetector()
        
    def select_strategy(self, context: StrategyContext, deterministic: bool = False) -> StrategySelection:
        """Selects strategy on basis context market"""
        state_tensor = torch.FloatTensor(context.market_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            strategy_logits, value, confidence, risk = self.network(state_tensor)
            strategy_probs = F.softmax(strategy_logits, dim=-1)
        
        # Selection strategies
        if deterministic or np.random.random() > self.exploration_epsilon:
            # Greedy selection with considering confidence
            strategy_scores = strategy_probs.squeeze() * confidence.squeeze()
            strategy_id = torch.argmax(strategy_scores).item()
        else:
            # Exploration
            strategy_id = torch.multinomial(strategy_probs, 1).item()
        
        strategy_name = self.strategy_id_to_name[strategy_id]
        strategy_type = StrategyType(strategy_name)
        
        # Parameters strategies on basis current context
        parameters = self._generate_strategy_parameters(strategy_type, context)
        
        # Expected horizon on basis type strategies and market conditions
        expected_horizon = self._estimate_strategy_horizon(strategy_type, context)
        
        selection = StrategySelection(
            strategy_type=strategy_type,
            confidence=confidence[0, strategy_id].item(),
            parameters=parameters,
            expected_horizon=expected_horizon,
            risk_assessment=risk.item(),
            reasoning=self._generate_reasoning(strategy_type, context, value.item())
        )
        
        # Save decision
        self.decision_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'context': context,
            'selection': selection,
            'market_regime': self.regime_detector.detect_regime(context.market_state)
        })
        
        logger.info(f"Selected strategy: {strategy_type.value} (confidence: {selection.confidence:.3f})")
        return selection
    
    def _generate_strategy_parameters(self, 
                                    strategy_type: StrategyType, 
                                    context: StrategyContext) -> Dict[str, Any]:
        """Generates parameters for selected strategies"""
        base_params = self.strategy_registry.get(strategy_type.value, {})
        
        # Adapt parameters under current context
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return {
                **base_params,
                'lookback_period': min(50, max(10, int(20 / context.volatility))),
                'momentum_threshold': 0.02 * context.volatility,
                'stop_loss': 0.03 * (1 + context.risk_level),
                'take_profit': 0.05 * (1 + context.trend_strength)
            }
        
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return {
                **base_params,
                'deviation_threshold': 2.0 * context.volatility,
                'reversion_period': int(30 / context.volatility),
                'position_size': 0.1 / (1 + context.risk_level),
                'max_holding_time': int(100 / context.volatility)
            }
        
        elif strategy_type == StrategyType.ARBITRAGE:
            return {
                **base_params,
                'min_spread': max(0.001, context.spread * 0.5),
                'max_position_size': 0.2 / (1 + context.risk_level),
                'execution_timeout': 5,  # seconds
                'slippage_tolerance': 0.001
            }
        
        elif strategy_type == StrategyType.MARKET_MAKING:
            return {
                **base_params,
                'bid_ask_spread': max(0.002, context.spread * 1.2),
                'inventory_target': 0.0,
                'max_inventory': 0.1 / (1 + context.risk_level),
                'quote_frequency': max(1, int(context.volume / 1000000))
            }
        
        else:
            return base_params
    
    def _estimate_strategy_horizon(self, 
                                 strategy_type: StrategyType, 
                                 context: StrategyContext) -> int:
        """Evaluates expected temporal horizon strategies"""
        base_horizons = {
            StrategyType.ARBITRAGE: 10,
            StrategyType.MARKET_MAKING: 30,
            StrategyType.MOMENTUM: 50,
            StrategyType.MEAN_REVERSION: 100,
            StrategyType.TREND_FOLLOWING: 200,
            StrategyType.SWING_TRADING: 500,
            StrategyType.PAIRS_TRADING: 150,
            StrategyType.BREAKOUT: 80
        }
        
        base_horizon = base_horizons.get(strategy_type, 100)
        
        # Adapt on basis volatility and volume
        volatility_factor = 1.0 / (1.0 + context.volatility)
        volume_factor = min(2.0, context.volume / 1000000)
        
        adjusted_horizon = int(base_horizon * volatility_factor * volume_factor)
        return max(5, min(1000, adjusted_horizon))
    
    def _generate_reasoning(self, 
                          strategy_type: StrategyType, 
                          context: StrategyContext, 
                          value_estimate: float) -> str:
        """Generates explanation selection strategies"""
        reasons = []
        
        if strategy_type == StrategyType.TREND_FOLLOWING:
            if context.trend_strength > 0.5:
                reasons.append(f"Strong trend (force: {context.trend_strength:.2f})")
            if context.volume > 1000000:
                reasons.append(f"High volume trading ({context.volume/1e6:.1f}M)")
        
        elif strategy_type == StrategyType.MEAN_REVERSION:
            if context.volatility > 0.05:
                reasons.append(f"High volatility ({context.volatility:.3f})")
            if abs(context.trend_strength) < 0.3:
                reasons.append("Absence explicit trend")
        
        elif strategy_type == StrategyType.ARBITRAGE:
            if context.spread > 0.003:
                reasons.append(f"Wide spread ({context.spread:.4f})")
            if context.volatility < 0.02:
                reasons.append("Low volatility")
        
        if value_estimate > 0:
            reasons.append(f"Positive estimation value ({value_estimate:.3f})")
        
        return "; ".join(reasons) if reasons else "Base selection strategies"
    
    def update_strategy_performance(self, 
                                  strategy_name: str, 
                                  reward: float, 
                                  duration: int, 
                                  success: bool) -> None:
        """Updates statistics performance strategies"""
        perf = self.strategy_performance[strategy_name]
        
        perf['total_reward'] += reward
        perf['executions'] += 1
        
        # Exponential moving average for other metrics
        alpha = self.adaptation_rate
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * float(success)
        perf['avg_duration'] = (1 - alpha) * perf['avg_duration'] + alpha * duration
        
        # Risk-adjusted return (Sharpe-like ratio)
        if perf['executions'] > 10:
            avg_reward = perf['total_reward'] / perf['executions']
            perf['risk_adjusted_return'] = avg_reward / max(0.01, np.std(self.episode_rewards[-10:]))
    
    def should_switch_strategy(self, 
                             current_context: StrategyContext,
                             current_performance: float,
                             time_since_start: float) -> bool:
        """Determines necessity change strategies"""
        if not self.current_strategy:
            return True
        
        # Check performance
        expected_performance = self.strategy_performance[self.current_strategy]['total_reward'] / \
                             max(1, self.strategy_performance[self.current_strategy]['executions'])
        
        if current_performance < expected_performance * 0.5:
            logger.info(f"Low performance strategies {self.current_strategy}")
            return True
        
        # Check change mode market
        current_regime = self.regime_detector.detect_regime(current_context.market_state)
        if hasattr(self, '_last_regime') and current_regime != self._last_regime:
            logger.info(f"Change mode market: {self._last_regime} -> {current_regime}")
            return True
        
        # Check maximum time execution
        max_duration = self._estimate_strategy_horizon(
            StrategyType(self.current_strategy), current_context
        )
        
        if time_since_start > max_duration * 1.5:
            logger.info(f"Exceeded maximum time execution strategies")
            return True
        
        return False
    
    def train_step(self, 
                  states: torch.Tensor, 
                  actions: torch.Tensor, 
                  rewards: torch.Tensor, 
                  next_states: torch.Tensor,
                  dones: torch.Tensor) -> Dict[str, float]:
        """Training step meta-policy"""
        strategy_logits, values, confidences, risks = self.network(states)
        next_strategy_logits, next_values, _, _ = self.network(next_states)
        
        # Policy loss
        log_probs = F.log_softmax(strategy_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Advantage estimation
        td_targets = rewards + 0.99 * next_values.squeeze() * (1 - dones)
        advantages = td_targets - values.squeeze()
        
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), td_targets.detach())
        
        # Confidence loss (binary classification success)
        success_targets = (rewards > 0).float()
        confidence_loss = F.binary_cross_entropy(
            confidences.gather(1, actions.unsqueeze(1)).squeeze(),
            success_targets
        )
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * confidence_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save_policy(self, filepath: str) -> None:
        """Saves meta-policy"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'strategy_performance': dict(self.strategy_performance),
            'decision_history': list(self.decision_history)
        }, filepath)
    
    def load_policy(self, filepath: str) -> None:
        """Loads meta-policy"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.strategy_performance = defaultdict(lambda: {
            'total_reward': 0.0,
            'executions': 0,
            'success_rate': 0.0,
            'avg_duration': 0.0,
            'risk_adjusted_return': 0.0
        }, checkpoint['strategy_performance'])
        self.decision_history = deque(checkpoint['decision_history'], maxlen=10000)
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Returns statistics by strategies"""
        return {
            'strategy_performance': dict(self.strategy_performance),
            'current_strategy': self.current_strategy,
            'total_decisions': len(self.decision_history),
            'exploration_epsilon': self.exploration_epsilon
        }


class MarketRegimeDetector:
    """Detector modes market for adaptation meta-policy"""
    
    def __init__(self, lookback_period: int = 50):
        self.lookback_period = lookback_period
        self.price_history: deque = deque(maxlen=lookback_period)
        self.volume_history: deque = deque(maxlen=lookback_period)
        
    def detect_regime(self, market_state: np.ndarray) -> str:
        """Determines current mode market"""
        if len(market_state) > 0:
            self.price_history.append(market_state[0])
        if len(market_state) > 1:
            self.volume_history.append(market_state[1])
        
        if len(self.price_history) < 10:
            return "insufficient_data"
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history) if self.volume_history else np.ones_like(prices)
        
        # Compute characteristics
        volatility = np.std(np.diff(prices))
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
        
        # Classification mode
        if volatility > 0.05:
            return "high_volatility"
        elif abs(trend) > 0.01:
            return "trending"
        elif volume_trend > 0.1:
            return "increasing_volume"
        elif volatility < 0.01:
            return "low_volatility"
        else:
            return "sideways"


# Factory for creation meta-policies

def create_crypto_meta_policy(state_dim: int = 15, device: str = "cpu") -> MetaPolicy:
    """Creates meta-policy for crypto trading"""
    strategy_registry = {
        "trend_following": {
            "base_lookback": 20,
            "base_threshold": 0.02
        },
        "mean_reversion": {
            "base_deviation": 2.0,
            "base_period": 30
        },
        "momentum": {
            "base_window": 14,
            "base_threshold": 0.01
        },
        "arbitrage": {
            "min_spread": 0.001,
            "max_exposure": 0.1
        },
        "market_making": {
            "base_spread": 0.002,
            "inventory_limit": 0.05
        }
    }
    
    return MetaPolicy(
        state_dim=state_dim,
        strategy_registry=strategy_registry,
        learning_rate=1e-4,
        exploration_epsilon=0.1,
        confidence_threshold=0.7,
        device=device
    )