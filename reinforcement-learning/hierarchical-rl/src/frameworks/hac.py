"""
Hierarchical Actor-Critic (HAC) Implementation
Implementation hierarchical actor-critic for multi-level trading strategies.

enterprise Pattern:
- Hierarchical Actor-Critic with temporal abstraction for complex strategies
- Production-ready multi-level policy learning with hindsight experience replay
- Scalable goal-conditioned learning for crypto trading environments
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
from torch.distributions import Normal, Categorical
import logging
import asyncio
import random
from collections import deque, namedtuple
import copy

logger = logging.getLogger(__name__)


@dataclass
class HACTransition:
    """Transition in HAC system"""
    state: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    level: int
    intrinsic_reward: float = 0.0
    achieved_goal: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HACLevel(Enum):
    """Levels hierarchy in HAC"""
    HIGH = "high"          # High level (strategies)
    MIDDLE = "middle"      # Average level (tactics)  
    LOW = "low"           # Low level (actions)


class Actor(nn.Module):
    """Actor network for continuous actions"""
    
    def __init__(self, 
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 max_action: float = 1.0,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        self.max_action = max_action
        input_dim = state_dim + goal_dim
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer for average and logarithm variance
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        self.network = nn.Sequential(*layers)
        
        # Initialization weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns average and standard deviation actions"""
        x = torch.cat([state, goal], dim=-1)
        features = self.network(x)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)  # Limit for stability
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples action and returns log_prob"""
        mean, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t) * self.max_action
        
        # Compute log probability with correction on tanh
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, goal: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Returns action (deterministic or stochastic)"""
        mean, log_std = self.forward(state, goal)
        
        if deterministic:
            action = torch.tanh(mean) * self.max_action
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t) * self.max_action
        
        return action


class Critic(nn.Module):
    """Critic network for estimation state-action values"""
    
    def __init__(self, 
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256]):
        super().__init__()
        
        # Two Q-networks for Double Q-learning
        self.q1 = self._build_network(state_dim + goal_dim + action_dim, hidden_dims)
        self.q2 = self._build_network(state_dim + goal_dim + action_dim, hidden_dims)
        
        self.apply(self._init_weights)
    
    def _build_network(self, input_dim: int, hidden_dims: List[int]) -> nn.Module:
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns Q-values from both networks"""
        x = torch.cat([state, goal, action], dim=-1)
        q1 = self.q1(x)
        q2 = self.q2(x)
        return q1, q2


class HACAgent:
    """
    Hierarchical Actor-Critic Agent
    Implements multi-level training with goal-conditioned policies
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 goal_dim: int,
                 max_action: float = 1.0,
                 num_levels: int = 3,
                 subgoal_freq: int = 10,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 noise_std: float = 0.1,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.max_action = max_action
        self.num_levels = num_levels
        self.subgoal_freq = subgoal_freq
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.device = torch.device(device)
        
        # Create actor-critic for of each level
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for level in range(num_levels):
            # Actor and Critic for of each level
            actor = Actor(state_dim, goal_dim, action_dim if level == 0 else goal_dim, max_action).to(self.device)
            critic = Critic(state_dim, goal_dim, action_dim if level == 0 else goal_dim).to(self.device)
            
            # Target networks
            target_actor = copy.deepcopy(actor)
            target_critic = copy.deepcopy(critic)
            
            # Optimizers
            actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
            critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_optimizers.append(actor_optimizer)
            self.critic_optimizers.append(critic_optimizer)
        
        # Replay buffers for of each level
        self.replay_buffers = [deque(maxlen=100000) for _ in range(num_levels)]
        
        # Counters for subgoal frequency
        self.subgoal_counters = [0] * num_levels
        
        # Statistics training
        self.training_stats = {
            'actor_losses': [[] for _ in range(num_levels)],
            'critic_losses': [[] for _ in range(num_levels)],
            'intrinsic_rewards': [[] for _ in range(num_levels)],
            'goal_achievements': [0] * num_levels
        }
    
    def select_action(self, 
                     state: np.ndarray, 
                     goal: np.ndarray, 
                     level: int = 0,
                     deterministic: bool = False,
                     add_noise: bool = True) -> np.ndarray:
        """Selects action on specified level"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        goal_tensor = torch.FloatTensor(goal).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actors[level].get_action(state_tensor, goal_tensor, deterministic)
            action = action.cpu().numpy().flatten()
        
        # Add noise for exploration
        if add_noise and not deterministic:
            action += np.random.normal(0, self.noise_std, size=action.shape)
            
            # Clamp for low level (primitive actions)
            if level == 0:
                action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def hierarchical_action_selection(self, 
                                    state: np.ndarray, 
                                    goal: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Hierarchical selection actions
        Returns primitive action and list subgoals
        """
        subgoals = []
        current_goal = goal.copy()
        
        # Traverse hierarchy from above down (except low level)
        for level in range(self.num_levels - 1, 0, -1):
            # Update subgoal if needed
            if self.subgoal_counters[level] % self.subgoal_freq == 0:
                subgoal = self.select_action(state, current_goal, level)
                subgoals.append(subgoal)
                current_goal = subgoal
            else:
                # Use previous subgoal
                if subgoals:
                    current_goal = subgoals[-1]
            
            self.subgoal_counters[level] += 1
        
        # Select primitive action on low level
        primitive_action = self.select_action(state, current_goal, level=0)
        
        return primitive_action, subgoals
    
    def compute_intrinsic_reward(self, 
                                state: np.ndarray,
                                next_state: np.ndarray,
                                goal: np.ndarray,
                                level: int) -> float:
        """
        Computes intrinsic reward for achievement subgoal
        Uses distance-based reward
        """
        if level == 0:
            # For low level use environment reward
            return 0.0
        
        # For high levels use negative distance to goal
        current_distance = np.linalg.norm(state - goal)
        next_distance = np.linalg.norm(next_state - goal)
        
        # Reward = progress to goals
        intrinsic_reward = current_distance - next_distance
        
        # Bonus for reaching goals
        if next_distance < 0.1:  # Threshold achievement goals
            intrinsic_reward += 1.0
            self.training_stats['goal_achievements'][level] += 1
        
        return intrinsic_reward
    
    def store_transition(self, 
                        transition: HACTransition) -> None:
        """Saves transition in replay buffer corresponding level"""
        self.replay_buffers[transition.level].append(transition)
    
    def hindsight_experience_replay(self, 
                                   episode_transitions: List[HACTransition]) -> List[HACTransition]:
        """
        Hindsight Experience Replay (HER)
        Creates additional transitions with alternative goals
        """
        her_transitions = []
        
        for transition in episode_transitions:
            # Original transition
            her_transitions.append(transition)
            
            # HER transitions (use future state as goals)
            if transition.level > 0:  # Only for high levels
                # Randomly select future state as goal
                future_idx = random.randint(0, len(episode_transitions) - 1)
                new_goal = episode_transitions[future_idx].achieved_goal
                
                if new_goal is not None:
                    # Recalculate reward with new purpose
                    new_intrinsic_reward = self.compute_intrinsic_reward(
                        transition.state,
                        transition.next_state,
                        new_goal,
                        transition.level
                    )
                    
                    her_transition = HACTransition(
                        state=transition.state,
                        goal=new_goal,
                        action=transition.action,
                        reward=transition.reward,
                        next_state=transition.next_state,
                        done=transition.done,
                        level=transition.level,
                        intrinsic_reward=new_intrinsic_reward,
                        achieved_goal=transition.achieved_goal
                    )
                    her_transitions.append(her_transition)
        
        return her_transitions
    
    def update_networks(self, level: int, batch_size: int = 256) -> Tuple[float, float]:
        """Updates actor and critic networks for specified level"""
        if len(self.replay_buffers[level]) < batch_size:
            return 0.0, 0.0
        
        # Sample batch
        batch = random.sample(self.replay_buffers[level], batch_size)
        
        states = torch.FloatTensor([t.state for t in batch]).to(self.device)
        goals = torch.FloatTensor([t.goal for t in batch]).to(self.device)
        actions = torch.FloatTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward + t.intrinsic_reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t.next_state for t in batch]).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions, _ = self.target_actors[level].sample(next_states, goals)
            target_q1, target_q2 = self.target_critics[level](next_states, goals, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critics[level](states, goals, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizers[level].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[level].step()
        
        # Update Actor
        new_actions, log_probs = self.actors[level].sample(states, goals)
        q1, q2 = self.critics[level](states, goals, new_actions)
        q = torch.min(q1, q2)
        
        # SAC-style actor loss with entropy regularization
        alpha = 0.2  # Temperature parameter
        actor_loss = (alpha * log_probs - q).mean()
        
        self.actor_optimizers[level].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[level].step()
        
        # Soft update target networks
        self._soft_update(self.critics[level], self.target_critics[level])
        self._soft_update(self.actors[level], self.target_actors[level])
        
        # Save statistics
        self.training_stats['actor_losses'][level].append(actor_loss.item())
        self.training_stats['critic_losses'][level].append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def train_episode(self, 
                     episode_transitions: List[HACTransition],
                     training_iterations: int = 40) -> Dict[str, List[float]]:
        """Trains on episode with HER"""
        # Apply HER
        her_transitions = self.hindsight_experience_replay(episode_transitions)
        
        # Distribute transitions by levels
        for transition in her_transitions:
            self.store_transition(transition)
        
        # Update networks for of each level
        losses = {'actor_losses': [], 'critic_losses': []}
        
        for _ in range(training_iterations):
            for level in range(self.num_levels):
                actor_loss, critic_loss = self.update_networks(level)
                if actor_loss > 0:  # If was update
                    losses['actor_losses'].append(actor_loss)
                    losses['critic_losses'].append(critic_loss)
        
        return losses
    
    def save_models(self, filename_prefix: str) -> None:
        """Saves all model"""
        for level in range(self.num_levels):
            torch.save(self.actors[level].state_dict(), f"{filename_prefix}_actor_level_{level}.pth")
            torch.save(self.critics[level].state_dict(), f"{filename_prefix}_critic_level_{level}.pth")
    
    def load_models(self, filename_prefix: str) -> None:
        """Loads all model"""
        for level in range(self.num_levels):
            self.actors[level].load_state_dict(
                torch.load(f"{filename_prefix}_actor_level_{level}.pth", map_location=self.device)
            )
            self.critics[level].load_state_dict(
                torch.load(f"{filename_prefix}_critic_level_{level}.pth", map_location=self.device)
            )
            
            # Synchronize target networks
            self.target_actors[level].load_state_dict(self.actors[level].state_dict())
            self.target_critics[level].load_state_dict(self.critics[level].state_dict())
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Returns statistics training"""
        stats = {
            'num_levels': self.num_levels,
            'replay_buffer_sizes': [len(buffer) for buffer in self.replay_buffers],
            'goal_achievements': self.training_stats['goal_achievements'].copy(),
            'avg_losses': {}
        }
        
        for level in range(self.num_levels):
            actor_losses = self.training_stats['actor_losses'][level]
            critic_losses = self.training_stats['critic_losses'][level]
            
            stats['avg_losses'][f'level_{level}'] = {
                'actor_loss': np.mean(actor_losses[-100:]) if actor_losses else 0.0,
                'critic_loss': np.mean(critic_losses[-100:]) if critic_losses else 0.0
            }
        
        return stats


class HACTradingEnvironment:
    """
    Wrapper for trading environment with support HAC
    Implements goal-conditioned environment for crypto trading
    """
    
    def __init__(self, base_env, goal_space_dim: int = 4):
        self.base_env = base_env
        self.goal_space_dim = goal_space_dim
        self.current_goal = None
        self.episode_steps = 0
        self.max_episode_steps = 1000
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resets environment and generates new goal"""
        state = self.base_env.reset() if hasattr(self.base_env, 'reset') else np.random.randn(10)
        self.current_goal = self._generate_goal()
        self.episode_steps = 0
        return state, self.current_goal
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Executes step in environment"""
        # Simulation for demonstration
        next_state = np.random.randn(10)
        reward = np.random.normal(0.001, 0.01)
        self.episode_steps += 1
        
        done = (self.episode_steps >= self.max_episode_steps or 
                self._is_goal_achieved(next_state, self.current_goal))
        
        info = {
            'achieved_goal': self._get_achieved_goal(next_state),
            'desired_goal': self.current_goal.copy(),
            'episode_steps': self.episode_steps
        }
        
        return next_state, reward, done, info
    
    def _generate_goal(self) -> np.ndarray:
        """Generates random goal"""
        # Goal can be: [target_price, target_profit, max_drawdown, time_horizon]
        return np.random.uniform(-1, 1, self.goal_space_dim)
    
    def _get_achieved_goal(self, state: np.ndarray) -> np.ndarray:
        """Extracts achieved goal from state"""
        # Simple projection state in space goals
        if len(state) >= self.goal_space_dim:
            return state[:self.goal_space_dim]
        else:
            return np.pad(state, (0, self.goal_space_dim - len(state)), 'constant')
    
    def _is_goal_achieved(self, state: np.ndarray, goal: np.ndarray) -> bool:
        """Checks reaching goals"""
        achieved = self._get_achieved_goal(state)
        distance = np.linalg.norm(achieved - goal)
        return distance < 0.1  # Threshold achievement goals
    
    def compute_reward(self, 
                      achieved_goal: np.ndarray, 
                      desired_goal: np.ndarray, 
                      info: Dict[str, Any]) -> float:
        """Computes reward on basis achievement goals (for HER)"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return -distance + (1.0 if distance < 0.1 else 0.0)


# Factory for creation HAC agents for crypto trading

def create_crypto_hac_agent(state_dim: int = 10, 
                           action_dim: int = 3,
                           goal_dim: int = 4,
                           device: str = "cpu") -> HACAgent:
    """Creates HAC agent for crypto trading"""
    return HACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        max_action=1.0,
        num_levels=3,  # Strategic, tactical, executive
        subgoal_freq=10,  # Update subgoals every 10 steps
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        noise_std=0.1,
        device=device
    )


def create_arbitrage_hac_agent(state_dim: int = 15,
                              action_dim: int = 5,
                              goal_dim: int = 3,
                              device: str = "cpu") -> HACAgent:
    """Creates HAC agent for arbitrage"""
    return HACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        goal_dim=goal_dim,
        max_action=1.0,
        num_levels=2,  # Search capabilities, execution
        subgoal_freq=5,  # Fast updates for arbitrage
        actor_lr=1e-3,  # More high learning rate
        critic_lr=1e-3,
        gamma=0.95,  # Smaller discount for fast strategies
        tau=0.01,   # More fast update target networks
        noise_std=0.05,  # Smaller noise for exact execution
        device=device
    )