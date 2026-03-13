"""
Curious Trader Agent for autonomous strategy discovery.

Implements advanced trading agent with curiosity-driven exploration
for automatic discovery new trading strategies in crypto markets.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from collections import deque

from ..curiosity.icm import ICMTrainer, ICMConfig
from ..curiosity.rnd import RNDTrainer, RNDConfig
from ..curiosity.ngu import NGUTrainer, NGUConfig
from ..exploration.exploration_bonus import ExplorationBonusManager, ExplorationBonusConfig
from ..memory.curiosity_buffer import CuriosityReplayBuffer, CuriosityBufferConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CuriousTraderConfig:
    """Configuration for curious trader agent."""
    
    # State and action spaces
    state_dim: int = 256
    action_dim: int = 10
    
    # Curiosity components
    use_icm: bool = True
    use_rnd: bool = True
    use_ngu: bool = False
    
    # Exploration settings
    exploration_weight: float = 0.1
    curiosity_decay: float = 0.99
    
    # Trading specific
    risk_tolerance: float = 0.5
    profit_target: float = 0.02
    max_position_size: float = 1.0
    
    #  enterprise settings
    distributed_training: bool = True
    real_time_execution: bool = True


class CuriousTrader:
    """
    Advanced trading agent with curiosity-driven exploration.
    
    Applies design pattern "Autonomous Agent" for
    intelligent trading strategy discovery.
    """
    
    def __init__(self, config: CuriousTraderConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Policy network
        self.policy_net = self._build_policy_network().to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        
        # Curiosity components
        self.curiosity_systems = {}
        
        if config.use_icm:
            icm_config = ICMConfig(
                state_dim=config.state_dim,
                action_dim=config.action_dim
            )
            self.curiosity_systems['icm'] = ICMTrainer(icm_config, device)
        
        if config.use_rnd:
            rnd_config = RNDConfig(
                state_dim=config.state_dim
            )
            self.curiosity_systems['rnd'] = RNDTrainer(rnd_config, device)
        
        # Exploration bonus manager
        bonus_config = ExplorationBonusConfig()
        self.exploration_manager = ExplorationBonusManager(bonus_config)
        
        # Experience replay
        buffer_config = CuriosityBufferConfig()
        self.replay_buffer = CuriosityReplayBuffer(buffer_config)
        
        # Trading state
        self.portfolio_value = 10000.0
        self.position = 0.0
        self.trade_history = deque(maxlen=1000)
        
        logger.info("Curious trader agent initialized")
    
    def _build_policy_network(self) -> nn.Module:
        """Build policy network."""
        return nn.Sequential(
            nn.Linear(self.config.state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.action_dim),
            nn.Tanh()
        )
    
    def select_action(
        self,
        state: np.ndarray,
        exploration: bool = True
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Action selection with curiosity-driven exploration."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_logits = self.policy_net(state_tensor)
            action = action_logits.squeeze(0).cpu().numpy()
        
        # Curiosity-driven exploration
        if exploration:
            # Get exploration bonus
            exploration_bonus, bonus_breakdown = self.exploration_manager.compute_exploration_bonus(
                state=state,
                action=action,
                context={'portfolio_value': self.portfolio_value}
            )
            
            # Action noise based on exploration bonus
            noise_scale = min(exploration_bonus * self.config.exploration_weight, 0.5)
            action += np.random.normal(0, noise_scale, action.shape)
            
            action = np.clip(action, -1, 1)
            
            return action, bonus_breakdown
        
        return action, {}
    
    def execute_trade(
        self,
        action: np.ndarray,
        market_price: float,
        market_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute trading ."""
        # Decode action
        position_change = action[0] * self.config.max_position_size
        
        # Risk management
        if abs(self.position + position_change) > self.config.max_position_size:
            position_change = np.sign(position_change) * (
                self.config.max_position_size - abs(self.position)
            )
        
        # Execute trade
        old_position = self.position
        self.position += position_change
        
        # Calculate trade cost and profit
        trade_cost = abs(position_change) * market_price * 0.001  # 0.1% fee
        self.portfolio_value -= trade_cost
        
        # Record trade
        trade_record = {
            'timestamp': market_data.get('timestamp', 0),
            'price': market_price,
            'position_change': position_change,
            'new_position': self.position,
            'portfolio_value': self.portfolio_value,
            'trade_cost': trade_cost
        }
        
        self.trade_history.append(trade_record)
        
        return trade_record
    
    def calculate_reward(
        self,
        prev_portfolio_value: float,
        current_portfolio_value: float,
        risk_metrics: Dict[str, float]
    ) -> float:
        """Computation trading reward."""
        # Portfolio return
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Risk-adjusted return
        volatility = risk_metrics.get('volatility', 0.1)
        risk_penalty = volatility * self.config.risk_tolerance
        
        # Position penalty for large positions
        position_penalty = abs(self.position) * 0.01
        
        reward = portfolio_return - risk_penalty - position_penalty
        
        return reward
    
    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """Update agent with new experience."""
        # Compute curiosity rewards
        curiosity_rewards = {}
        total_curiosity_reward = 0.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        for name, system in self.curiosity_systems.items():
            if name == 'icm':
                curiosity_reward = system.get_curiosity_reward(
                    state_tensor, action_tensor, next_state_tensor
                ).item()
            elif name == 'rnd':
                curiosity_reward = system.compute_intrinsic_reward(next_state_tensor).item()
            else:
                curiosity_reward = 0.0
            
            curiosity_rewards[name] = curiosity_reward
            total_curiosity_reward += curiosity_reward
        
        # Total reward
        total_reward = reward + self.config.exploration_weight * total_curiosity_reward
        
        # Store experience
        self.replay_buffer.add(
            state=state,
            action=action,
            reward=total_reward,
            next_state=next_state,
            done=done,
            curiosity_reward=total_curiosity_reward
        )
        
        # Update curiosity systems
        update_metrics = {}
        for name, system in self.curiosity_systems.items():
            if name == 'icm':
                metrics = system.train_step(state_tensor, action_tensor, next_state_tensor)
            elif name == 'rnd':
                metrics = system.train_step(next_state_tensor)
            else:
                metrics = {}
            
            update_metrics[f'{name}_metrics'] = metrics
        
        # Policy update (simplified)
        if self.replay_buffer.size >= 64:
            self._update_policy()
        
        update_metrics.update({
            'total_reward': total_reward,
            'extrinsic_reward': reward,
            'curiosity_reward': total_curiosity_reward,
            'curiosity_breakdown': curiosity_rewards,
            'portfolio_value': self.portfolio_value,
            'position': self.position
        })
        
        return update_metrics
    
    def _update_policy(self) -> None:
        """Policy network update."""
        batch = self.replay_buffer.sample(64)
        if not batch:
            return
        
        # Simple policy gradient update
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        
        self.optimizer.zero_grad()
        
        predicted_actions = self.policy_net(states)
        loss = nn.MSELoss()(predicted_actions, actions)
        
        loss.backward()
        self.optimizer.step()
    
    def get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get statistics portfolio."""
        if not self.trade_history:
            return {}
        
        trades = list(self.trade_history)
        portfolio_values = [trade['portfolio_value'] for trade in trades]
        
        return {
            'current_portfolio_value': self.portfolio_value,
            'current_position': self.position,
            'total_trades': len(trades),
            'portfolio_return': (self.portfolio_value - 10000.0) / 10000.0,
            'max_portfolio_value': max(portfolio_values),
            'min_portfolio_value': min(portfolio_values),
            'total_trade_costs': sum(trade['trade_cost'] for trade in trades)
        }


if __name__ == "__main__":
    config = CuriousTraderConfig(
        state_dim=128,
        action_dim=5,
        use_icm=True,
        use_rnd=True
    )
    
    trader = CuriousTrader(config)
    
    # Simulation loop
    for episode in range(10):
        state = np.random.randn(128)
        
        for step in range(100):
            # Select action
            action, exploration_info = trader.select_action(state, exploration=True)
            
            # Simulate market step
            next_state = np.random.randn(128)
            market_price = 1000 + np.random.randn() * 10
            reward = np.random.randn() * 0.01
            
            # Execute trade
            trade_info = trader.execute_trade(
                action, market_price, {'timestamp': step}
            )
            
            # Update agent
            update_metrics = trader.update(
                state, action, reward, next_state, done=(step == 99)
            )
            
            state = next_state
            
            if step % 50 == 0:
                print(f"Episode {episode}, Step {step}: "
                      f"Portfolio: {trader.portfolio_value:.2f}, "
                      f"Position: {trader.position:.3f}")
    
    # Final statistics
    stats = trader.get_portfolio_statistics()
    print("\nFinal Portfolio Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")