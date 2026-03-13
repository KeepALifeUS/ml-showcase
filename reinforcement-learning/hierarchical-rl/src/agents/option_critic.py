"""
Option-Critic Architecture Implementation
Implementation Option-Critic for end-to-end training options in trading strategies.

enterprise Pattern:
- End-to-end option learning for adaptive trading strategies
- Production-ready option-critic training with distributed execution
- Scalable temporal abstraction learning for complex market dynamics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import logging
import asyncio
from collections import deque, namedtuple
import copy
import random

logger = logging.getLogger(__name__)


class OptionCriticMode(Enum):
    """Modes work Option-Critic"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    PRODUCTION = "production"


@dataclass
class OptionCriticConfig:
    """Configuration Option-Critic agent"""
    state_dim: int
    num_options: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    beta_reg: float = 0.01  # Regularization for termination
    epsilon: float = 0.1    # Exploration for options
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    device: str = "cpu"


Experience = namedtuple('Experience', [
    'state', 'option', 'action', 'reward', 'next_state', 'terminated', 'done'
])


class OptionCriticNetwork(nn.Module):
    """Neural network for Option-Critic architectures"""
    
    def __init__(self, config: OptionCriticConfig):
        super().__init__()
        
        self.config = config
        
        # Shared representation
        self.shared_layers = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU()
        )
        
        # Option-value function Q(s, ω)
        self.option_values = nn.Linear(config.hidden_dim, config.num_options)
        
        # Intra-option policies π(a|s, ω)
        self.intra_option_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, config.action_dim)
            ) for _ in range(config.num_options)
        ])
        
        # Termination functions β(s, ω)
        self.termination_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(config.num_options)
        ])
        
        # Option-over-option policy Ω(ω|s)
        self.option_policy = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_options)
        )
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returns all components"""
        shared_features = self.shared_layers(state)
        
        # Option values
        option_values = self.option_values(shared_features)
        
        # Intra-option action probabilities
        intra_option_logits = torch.stack([
            policy(shared_features) for policy in self.intra_option_policies
        ], dim=1)  # [batch_size, num_options, action_dim]
        
        # Termination probabilities
        termination_probs = torch.cat([
            term_func(shared_features) for term_func in self.termination_functions
        ], dim=1)  # [batch_size, num_options]
        
        # Option selection logits
        option_logits = self.option_policy(shared_features)
        
        return {
            'option_values': option_values,
            'intra_option_logits': intra_option_logits,
            'termination_probs': termination_probs,
            'option_logits': option_logits
        }


class OptionCriticAgent:
    """
    Option-Critic agent for hierarchical training with reinforcement
    Trains options end-to-end without predefined subgoals
    """
    
    def __init__(self, config: OptionCriticConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create networks
        self.network = OptionCriticNetwork(config).to(self.device)
        self.target_network = copy.deepcopy(self.network)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=config.buffer_size)
        
        # State agent
        self.current_option: Optional[int] = None
        self.steps_since_option_start = 0
        self.mode = OptionCriticMode.TRAINING
        
        # Statistics
        self.total_steps = 0
        self.episode_count = 0
        self.option_lengths: List[int] = []
        self.option_rewards: List[float] = []
        self.option_usage_counts = np.zeros(config.num_options)
        
        # Metrics training
        self.losses = {
            'q_loss': [],
            'policy_loss': [],
            'termination_loss': [],
            'total_loss': []
        }
        
        logger.info(f"Initialized Option-Critic agent with {config.num_options} options")
    
    def select_option(self, state: np.ndarray, force_new: bool = False) -> int:
        """Selects option for execution"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(state_tensor)
            option_logits = outputs['option_logits']
            
            if self.mode == OptionCriticMode.TRAINING:
                # Epsilon-greedy selection for exploration
                if np.random.random() < self.config.epsilon or force_new:
                    option = np.random.randint(self.config.num_options)
                else:
                    option_probs = F.softmax(option_logits, dim=-1)
                    option = torch.multinomial(option_probs, 1).item()
            else:
                # Greedy selection for evaluation/production
                option = option_logits.argmax(dim=-1).item()
        
        return option
    
    def select_action(self, state: np.ndarray, option: int) -> int:
        """Selects action in within current options"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(state_tensor)
            intra_option_logits = outputs['intra_option_logits']
            option_action_logits = intra_option_logits[0, option]  # [action_dim]
            
            if self.mode == OptionCriticMode.TRAINING:
                # Stochastic action selection
                action_probs = F.softmax(option_action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
            else:
                # Deterministic action selection
                action = option_action_logits.argmax().item()
        
        return action
    
    def should_terminate_option(self, state: np.ndarray, option: int) -> bool:
        """Determines, should whether complete current option"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.network(state_tensor)
            termination_probs = outputs['termination_probs']
            termination_prob = termination_probs[0, option].item()
            
            # Stochastic termination
            return np.random.random() < termination_prob
    
    def step(self, state: np.ndarray) -> Tuple[int, int, bool]:
        """
        Executes one step agent
        Returns (option, action, option_terminated)
        """
        option_terminated = False
        
        # Check, needed whether select new option
        if (self.current_option is None or 
            self.should_terminate_option(state, self.current_option)):
            
            if self.current_option is not None:
                option_terminated = True
                # Record statistics completed options
                self.option_lengths.append(self.steps_since_option_start)
                self.option_usage_counts[self.current_option] += 1
            
            # Select new option
            self.current_option = self.select_option(state, force_new=option_terminated)
            self.steps_since_option_start = 0
        
        # Select action in within current options
        action = self.select_action(state, self.current_option)
        
        self.steps_since_option_start += 1
        self.total_steps += 1
        
        return self.current_option, action, option_terminated
    
    def store_experience(self, 
                        state: np.ndarray,
                        option: int,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        terminated: bool,
                        done: bool) -> None:
        """Saves experience in replay buffer"""
        experience = Experience(
            state=state,
            option=option,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            done=done
        )
        self.replay_buffer.append(experience)
    
    def train_step(self) -> Dict[str, float]:
        """Executes one step training"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        
        # Convert in tensors
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        options = torch.LongTensor([exp.option for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        terminated = torch.BoolTensor([exp.terminated for exp in batch]).to(self.device)
        done = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Forward pass
        outputs = self.network(states)
        next_outputs = self.target_network(next_states)
        
        # 1. Option-value loss (Q-learning for options)
        q_loss = self._compute_q_loss(
            outputs, next_outputs, options, rewards, terminated, done
        )
        
        # 2. Intra-option policy loss
        policy_loss = self._compute_policy_loss(
            outputs, options, actions, rewards, next_states, terminated, done
        )
        
        # 3. Termination loss
        termination_loss = self._compute_termination_loss(
            outputs, next_outputs, options, rewards, terminated, done
        )
        
        # Total loss
        total_loss = q_loss + policy_loss + termination_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.total_steps % self.config.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        # Save losses
        losses = {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'termination_loss': termination_loss.item(),
            'total_loss': total_loss.item()
        }
        
        for key, value in losses.items():
            self.losses[key].append(value)
        
        return losses
    
    def _compute_q_loss(self, 
                       outputs: Dict[str, torch.Tensor],
                       next_outputs: Dict[str, torch.Tensor],
                       options: torch.Tensor,
                       rewards: torch.Tensor,
                       terminated: torch.Tensor,
                       done: torch.Tensor) -> torch.Tensor:
        """Computes Q-learning loss for option-values"""
        option_values = outputs['option_values']
        next_option_values = next_outputs['option_values']
        next_termination_probs = next_outputs['termination_probs']
        
        # Current Q-values
        current_q_values = option_values.gather(1, options.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            # For completed options: maximum value among all options
            # For ongoing options: value current options
            next_q_terminated = next_option_values.max(dim=1)[0]
            next_q_continued = next_option_values.gather(1, options.unsqueeze(1)).squeeze(1)
            
            # Weighted combination on basis probability completion
            next_termination = next_termination_probs.gather(1, options.unsqueeze(1)).squeeze(1)
            next_q_values = (next_termination * next_q_terminated + 
                           (1 - next_termination) * next_q_continued)
            
            target_q_values = rewards + self.config.gamma * next_q_values * (~done)
        
        q_loss = F.mse_loss(current_q_values, target_q_values)
        return q_loss
    
    def _compute_policy_loss(self,
                           outputs: Dict[str, torch.Tensor],
                           options: torch.Tensor,
                           actions: torch.Tensor,
                           rewards: torch.Tensor,
                           next_states: torch.Tensor,
                           terminated: torch.Tensor,
                           done: torch.Tensor) -> torch.Tensor:
        """Computes policy gradient loss for intra-option policies"""
        option_values = outputs['option_values']
        intra_option_logits = outputs['intra_option_logits']
        
        batch_size = options.size(0)
        
        # Extract logits for selected options
        selected_logits = intra_option_logits[range(batch_size), options]  # [batch_size, action_dim]
        
        # Log probabilities selected actions
        log_probs = F.log_softmax(selected_logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Advantage estimation
        with torch.no_grad():
            # Use option-values as baseline
            current_values = option_values.gather(1, options.unsqueeze(1)).squeeze(1)
            
            # Simple TD-error as advantage
            next_values = self.target_network(next_states)['option_values']
            next_values_selected = next_values.gather(1, options.unsqueeze(1)).squeeze(1)
            
            td_targets = rewards + self.config.gamma * next_values_selected * (~done)
            advantages = td_targets - current_values
        
        # Policy gradient loss
        policy_loss = -(action_log_probs * advantages).mean()
        
        return policy_loss
    
    def _compute_termination_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                next_outputs: Dict[str, torch.Tensor],
                                options: torch.Tensor,
                                rewards: torch.Tensor,
                                terminated: torch.Tensor,
                                done: torch.Tensor) -> torch.Tensor:
        """Computes loss for termination functions"""
        termination_probs = outputs['termination_probs']
        option_values = outputs['option_values']
        next_option_values = next_outputs['option_values']
        
        batch_size = options.size(0)
        
        # Probability completion for selected options
        selected_term_probs = termination_probs.gather(1, options.unsqueeze(1)).squeeze(1)
        
        # Advantage from completion options
        with torch.no_grad():
            current_option_values = option_values.gather(1, options.unsqueeze(1)).squeeze(1)
            max_next_option_values = next_option_values.max(dim=1)[0]
            
            # Advantage completion = maximum value - current value options
            termination_advantage = max_next_option_values - current_option_values
        
        # Termination loss with regularization
        # Negative gradient, in order to increase probability completion when positive advantage
        termination_loss = -(selected_term_probs * termination_advantage).mean()
        
        # Entropy regularization for prevention too early/late completions
        entropy_loss = -(selected_term_probs * torch.log(selected_term_probs + 1e-8) + 
                        (1 - selected_term_probs) * torch.log(1 - selected_term_probs + 1e-8)).mean()
        
        total_termination_loss = termination_loss + self.config.beta_reg * entropy_loss
        
        return total_termination_loss
    
    def episode_done(self, total_reward: float) -> None:
        """Is called when completion episode"""
        self.episode_count += 1
        
        # Record statistics
        if self.current_option is not None:
            self.option_lengths.append(self.steps_since_option_start)
            self.option_usage_counts[self.current_option] += 1
        
        self.option_rewards.append(total_reward)
        
        # Reset state
        self.current_option = None
        self.steps_since_option_start = 0
        
        logger.debug(f"Episode {self.episode_count} completed with reward {total_reward:.3f}")
    
    def set_mode(self, mode: OptionCriticMode) -> None:
        """Sets mode work agent"""
        self.mode = mode
        if mode == OptionCriticMode.TRAINING:
            self.network.train()
        else:
            self.network.eval()
    
    def get_option_statistics(self) -> Dict[str, Any]:
        """Returns statistics by options"""
        if not self.option_lengths:
            return {}
        
        return {
            'avg_option_length': np.mean(self.option_lengths),
            'option_usage_distribution': (self.option_usage_counts / 
                                         max(self.option_usage_counts.sum(), 1)).tolist(),
            'total_episodes': self.episode_count,
            'avg_episode_reward': np.mean(self.option_rewards) if self.option_rewards else 0.0,
            'option_diversity': len(np.where(self.option_usage_counts > 0)[0]) / self.config.num_options
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Returns statistics training"""
        stats = {}
        
        for loss_name, loss_values in self.losses.items():
            if loss_values:
                stats[f'avg_{loss_name}'] = np.mean(loss_values[-100:])  # Recent 100 values
        
        stats.update({
            'total_steps': self.total_steps,
            'replay_buffer_size': len(self.replay_buffer),
            'current_epsilon': self.config.epsilon
        })
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """Saves model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'episode_count': self.episode_count,
            'option_usage_counts': self.option_usage_counts,
            'losses': self.losses
        }, filepath)
        
        logger.info(f"Model saved in {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Loads model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_steps = checkpoint.get('total_steps', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.option_usage_counts = checkpoint.get('option_usage_counts', 
                                                 np.zeros(self.config.num_options))
        self.losses = checkpoint.get('losses', {key: [] for key in self.losses.keys()})
        
        logger.info(f"Model loaded from {filepath}")
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01) -> None:
        """Decreases epsilon for exploration"""
        self.config.epsilon = max(min_epsilon, self.config.epsilon * decay_rate)


class TradingOptionCriticAgent(OptionCriticAgent):
    """
    Specialized Option-Critic agent for trading
    Adds trading-specific function
    """
    
    def __init__(self, 
                 state_dim: int,
                 num_options: int = 8,
                 action_dim: int = 3,
                 device: str = "cpu"):
        
        config = OptionCriticConfig(
            state_dim=state_dim,
            num_options=num_options,
            action_dim=action_dim,
            hidden_dim=256,
            learning_rate=3e-4,
            gamma=0.99,
            beta_reg=0.01,
            epsilon=0.1,
            device=device
        )
        
        super().__init__(config)
        
        # Trading metrics
        self.portfolio_value = 1.0
        self.trades_executed = 0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # History trading
        self.portfolio_history: List[float] = [1.0]
        self.trade_returns: List[float] = []
        
        # Mapping options on trading strategies
        self.option_names = [
            "trend_following", "mean_reversion", "momentum", "arbitrage",
            "breakout", "support_resistance", "volatility", "hold"
        ]
    
    def execute_trade(self, action: int, market_data: Dict[str, float]) -> float:
        """Executes trading action and returns reward"""
        # Simple simulation trading
        price_change = market_data.get('price_change', 0.0)
        
        if action == 0:  # Buy
            trade_return = price_change * 0.95  # Consider slippage
        elif action == 2:  # Sell
            trade_return = -price_change * 0.95
        else:  # Hold
            trade_return = 0.0
        
        # Update portfolio
        self.portfolio_value *= (1 + trade_return)
        self.portfolio_history.append(self.portfolio_value)
        
        if action != 1:  # If not hold
            self.trades_executed += 1
            self.trade_returns.append(trade_return)
            
            # Update metrics
            self._update_trading_metrics()
        
        # Reward on basis trading result
        reward = trade_return
        
        # Penalty for too frequent trading
        if len(self.trade_returns) > 10:
            recent_trades = sum(1 for r in self.trade_returns[-10:] if r != 0)
            if recent_trades > 7:  # More 7 trading from recent 10
                reward -= 0.001
        
        return reward
    
    def _update_trading_metrics(self) -> None:
        """Updates trading metrics"""
        if len(self.trade_returns) > 0:
            # Win rate
            wins = sum(1 for r in self.trade_returns if r > 0)
            self.win_rate = wins / len(self.trade_returns)
            
            # Sharpe ratio
            if len(self.trade_returns) > 1:
                returns_array = np.array(self.trade_returns)
                self.sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
            
            # Max drawdown
            if len(self.portfolio_history) > 1:
                peak = np.maximum.accumulate(self.portfolio_history)
                drawdowns = (self.portfolio_history - peak) / peak
                self.max_drawdown = abs(np.min(drawdowns))
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        """Returns trading statistics"""
        base_stats = self.get_option_statistics()
        
        trading_stats = {
            'portfolio_value': self.portfolio_value,
            'total_return': self.portfolio_value - 1.0,
            'trades_executed': self.trades_executed,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_trade_return': np.mean(self.trade_returns) if self.trade_returns else 0.0
        }
        
        # Add statistics by named options
        if 'option_usage_distribution' in base_stats:
            usage_dist = base_stats['option_usage_distribution']
            named_usage = {
                self.option_names[i]: usage_dist[i] 
                for i in range(min(len(self.option_names), len(usage_dist)))
            }
            trading_stats['option_usage_by_name'] = named_usage
        
        return {**base_stats, **trading_stats}


# Factory for creation trading Option-Critic agents

def create_crypto_option_critic(state_dim: int = 20, 
                               num_options: int = 8,
                               device: str = "cpu") -> TradingOptionCriticAgent:
    """Creates Option-Critic agent for crypto trading"""
    return TradingOptionCriticAgent(
        state_dim=state_dim,
        num_options=num_options,
        action_dim=3,  # buy, hold, sell
        device=device
    )


def create_forex_option_critic(state_dim: int = 25,
                              num_options: int = 6,
                              device: str = "cpu") -> TradingOptionCriticAgent:
    """Creates Option-Critic agent for forex trading"""
    agent = TradingOptionCriticAgent(
        state_dim=state_dim,
        num_options=num_options,
        action_dim=3,
        device=device
    )
    
    # Specific settings for forex
    agent.config.gamma = 0.95  # Smaller discount factor
    agent.config.learning_rate = 1e-4  # More slow training
    agent.option_names = [
        "carry_trade", "trend_following", "range_trading",
        "news_trading", "scalping", "hold"
    ]
    
    return agent