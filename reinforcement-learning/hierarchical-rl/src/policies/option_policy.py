"""
Option Policy Implementation for Hierarchical RL
Implements policy options for temporally extended actions in trading strategies.

enterprise Pattern:
- Option-based temporal abstraction for complex trading sequences
- Production-ready option learning with adaptive termination
- Hierarchical option composition for scalable strategy execution
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
from torch.distributions import Normal, Categorical, Bernoulli
import logging
import asyncio
from collections import deque, defaultdict
import copy
import pickle

logger = logging.getLogger(__name__)


class OptionPhase(Enum):
    """Phases execution options"""
    INITIATION = "initiation"
    EXECUTION = "execution"  
    TERMINATION = "termination"
    COMPLETED = "completed"


@dataclass
class OptionState:
    """State options"""
    environment_state: np.ndarray
    option_id: str
    phase: OptionPhase
    steps_executed: int
    cumulative_reward: float
    termination_confidence: float
    internal_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptionExecution:
    """Result execution options"""
    option_id: str
    success: bool
    total_steps: int
    total_reward: float
    termination_reason: str
    final_state: np.ndarray
    execution_trace: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptionPolicyNetwork(nn.Module):
    """Neural network for policy options"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 option_id: str,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.option_id = option_id
        self.action_dim = action_dim
        
        # Encoder for state
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Policy head (actor)
        self.policy_mean = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim)
        )
        
        self.policy_log_std = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], action_dim)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1)
        )
        
        # Initiation head (determines when possible start option)
        self.initiation_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # Termination head (determines when complete option)
        self.termination_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
        
        # Progress head (evaluates progress execution)
        self.progress_head = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returns all outputs"""
        features = self.state_encoder(state)
        
        outputs = {
            'policy_mean': self.policy_mean(features),
            'policy_log_std': torch.clamp(self.policy_log_std(features), -20, 2),
            'value': self.value_head(features),
            'initiation': self.initiation_head(features),
            'termination': self.termination_head(features),
            'progress': self.progress_head(features)
        }
        
        return outputs
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples action and returns log_prob"""
        outputs = self.forward(state)
        
        mean = outputs['policy_mean']
        log_std = outputs['policy_log_std']
        std = log_std.exp()
        
        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Returns action (deterministic or stochastic)"""
        outputs = self.forward(state)
        
        if deterministic:
            return outputs['policy_mean']
        else:
            action, _ = self.sample_action(state)
            return action


class OptionPolicy:
    """
    Policy options with support initiation, execution, and termination
    Implements temporally extended actions for trading strategies
    """
    
    def __init__(self,
                 option_id: str,
                 state_dim: int,
                 action_dim: int,
                 initiation_conditions: Optional[Callable] = None,
                 termination_conditions: Optional[Callable] = None,
                 learning_rate: float = 3e-4,
                 device: str = "cpu"):
        
        self.option_id = option_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Create network
        self.network = OptionPolicyNetwork(
            state_dim, action_dim, option_id
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Target network
        self.target_network = copy.deepcopy(self.network)
        self.target_update_freq = 100
        self.update_counter = 0
        
        # Conditions initiation and completion
        self.initiation_conditions = initiation_conditions
        self.termination_conditions = termination_conditions
        
        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        
        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.average_duration = 0.0
        
        # Current execution
        self.current_state: Optional[OptionState] = None
        self.execution_trace: List[Dict[str, Any]] = []
        
        # Adaptive parameters
        self.initiation_threshold = 0.5
        self.termination_threshold = 0.8
        self.max_option_length = 100
        
    def can_initiate(self, state: np.ndarray) -> bool:
        """Checks capability initiation options"""
        # Use external conditions if exists
        if self.initiation_conditions:
            return self.initiation_conditions(state)
        
        # Use neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.network(state_tensor)
            initiation_prob = outputs['initiation'].item()
        
        return initiation_prob > self.initiation_threshold
    
    def should_terminate(self, state: np.ndarray, steps_executed: int) -> bool:
        """Determines necessity completion options"""
        # Maximum duration
        if steps_executed >= self.max_option_length:
            return True
        
        # External conditions
        if self.termination_conditions:
            if self.termination_conditions(state):
                return True
        
        # Neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.network(state_tensor)
            termination_prob = outputs['termination'].item()
        
        return termination_prob > self.termination_threshold
    
    async def execute_option(self, 
                           initial_state: np.ndarray,
                           environment_step_fn: Callable,
                           max_steps: int = 100) -> OptionExecution:
        """Executes option until completion"""
        if not self.can_initiate(initial_state):
            return OptionExecution(
                option_id=self.option_id,
                success=False,
                total_steps=0,
                total_reward=0.0,
                termination_reason="cannot_initiate",
                final_state=initial_state,
                execution_trace=[]
            )
        
        # Initialize state options
        self.current_state = OptionState(
            environment_state=initial_state.copy(),
            option_id=self.option_id,
            phase=OptionPhase.INITIATION,
            steps_executed=0,
            cumulative_reward=0.0,
            termination_confidence=0.0
        )
        
        self.execution_trace = []
        
        try:
            # Phase execution
            self.current_state.phase = OptionPhase.EXECUTION
            
            while (self.current_state.steps_executed < max_steps and 
                   self.current_state.phase != OptionPhase.COMPLETED):
                
                # Retrieve action
                state_tensor = torch.FloatTensor(
                    self.current_state.environment_state
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.network(state_tensor)
                    action, log_prob = self.network.sample_action(state_tensor)
                    action_np = action.cpu().numpy().flatten()
                
                # Execute action in environment
                next_state, reward, done, info = await environment_step_fn(action_np)
                
                # Update state options
                self.current_state.environment_state = next_state
                self.current_state.steps_executed += 1
                self.current_state.cumulative_reward += reward
                self.current_state.termination_confidence = outputs['termination'].item()
                
                # Save in trace
                step_info = {
                    'step': self.current_state.steps_executed,
                    'action': action_np.tolist(),
                    'reward': reward,
                    'state': next_state.tolist(),
                    'termination_prob': self.current_state.termination_confidence,
                    'progress': outputs['progress'].item(),
                    'value_estimate': outputs['value'].item()
                }
                self.execution_trace.append(step_info)
                
                # Save experience for training
                self.store_experience(
                    self.current_state.environment_state,
                    action_np,
                    reward,
                    next_state,
                    done
                )
                
                # Check completion
                if (done or 
                    self.should_terminate(next_state, self.current_state.steps_executed)):
                    self.current_state.phase = OptionPhase.TERMINATION
                    break
                
                await asyncio.sleep(0.001)  # Asynchrony
            
            # Completion
            self.current_state.phase = OptionPhase.COMPLETED
            success = self.current_state.cumulative_reward > 0  # Simple condition success
            
            # Update statistics
            self.execution_count += 1
            if success:
                self.success_count += 1
            
            self.total_reward += self.current_state.cumulative_reward
            self.average_duration = (
                (self.average_duration * (self.execution_count - 1) + 
                 self.current_state.steps_executed) / self.execution_count
            )
            
            execution = OptionExecution(
                option_id=self.option_id,
                success=success,
                total_steps=self.current_state.steps_executed,
                total_reward=self.current_state.cumulative_reward,
                termination_reason="success" if success else "max_steps_or_termination",
                final_state=self.current_state.environment_state,
                execution_trace=self.execution_trace.copy(),
                metadata={
                    'average_progress': np.mean([t['progress'] for t in self.execution_trace]),
                    'final_termination_confidence': self.current_state.termination_confidence
                }
            )
            
            logger.info(f"Option {self.option_id} completed: {success}, steps: {self.current_state.steps_executed}, reward: {self.current_state.cumulative_reward:.4f}")
            return execution
            
        except Exception as e:
            logger.error(f"Error execution options {self.option_id}: {e}")
            return OptionExecution(
                option_id=self.option_id,
                success=False,
                total_steps=self.current_state.steps_executed if self.current_state else 0,
                total_reward=self.current_state.cumulative_reward if self.current_state else 0.0,
                termination_reason="error",
                final_state=self.current_state.environment_state if self.current_state else initial_state,
                execution_trace=self.execution_trace
            )
        
        finally:
            self.current_state = None
    
    def store_experience(self,
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool) -> None:
        """Saves experience for training"""
        self.replay_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """Training step policy options"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        import random
        batch = random.sample(self.replay_buffer, batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Forward pass
        outputs = self.network(states)
        next_outputs = self.target_network(next_states)
        
        # Actor loss
        _, log_probs = self.network.sample_action(states)
        values = outputs['value']
        
        # TD targets
        with torch.no_grad():
            next_values = next_outputs['value']
            td_targets = rewards.unsqueeze(1) + 0.99 * next_values * (1 - dones.unsqueeze(1))
        
        # Advantages
        advantages = td_targets - values
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values, td_targets)
        
        # Initiation loss (train predict when option can begin)
        # Simple heuristic: option well is initiated if reward positive
        initiation_targets = (rewards > 0).float()
        initiation_loss = F.binary_cross_entropy(
            outputs['initiation'].squeeze(), initiation_targets
        )
        
        # Termination loss (train predict when option must complete)
        # Option must complete when done=True or when accumulated sufficient reward
        termination_targets = torch.logical_or(
            dones.bool(), 
            rewards > np.percentile([exp['reward'] for exp in batch], 75)
        ).float()
        termination_loss = F.binary_cross_entropy(
            outputs['termination'].squeeze(), termination_targets
        )
        
        # Progress loss (train evaluate progress)
        # Simple heuristic: progress = normalized reward
        max_reward = max(rewards.max().item(), 0.01)
        progress_targets = torch.clamp(rewards / max_reward, 0, 1)
        progress_loss = F.mse_loss(outputs['progress'].squeeze(), progress_targets)
        
        # Total loss
        total_loss = (actor_loss + 0.5 * critic_loss + 
                     0.1 * initiation_loss + 0.1 * termination_loss + 
                     0.1 * progress_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'initiation_loss': initiation_loss.item(),
            'termination_loss': termination_loss.item(),
            'progress_loss': progress_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def adapt_thresholds(self, success_rate: float) -> None:
        """Adapts thresholds on basis performance"""
        if success_rate < 0.3:
            # Reduce threshold initiation (more conservatively)
            self.initiation_threshold = min(0.8, self.initiation_threshold + 0.05)
            # Increase threshold completion (complete earlier)
            self.termination_threshold = max(0.5, self.termination_threshold - 0.05)
        elif success_rate > 0.7:
            # Reduce threshold initiation (more aggressively)
            self.initiation_threshold = max(0.2, self.initiation_threshold - 0.05)
            # Reduce threshold completion (execute longer)
            self.termination_threshold = min(0.9, self.termination_threshold + 0.05)
    
    def get_success_rate(self) -> float:
        """Returns share successful executions"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics options"""
        return {
            'option_id': self.option_id,
            'execution_count': self.execution_count,
            'success_rate': self.get_success_rate(),
            'average_reward': self.total_reward / max(1, self.execution_count),
            'average_duration': self.average_duration,
            'initiation_threshold': self.initiation_threshold,
            'termination_threshold': self.termination_threshold,
            'replay_buffer_size': len(self.replay_buffer)
        }
    
    def save_option(self, filepath: str) -> None:
        """Saves option"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'option_id': self.option_id,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'total_reward': self.total_reward,
            'average_duration': self.average_duration,
            'initiation_threshold': self.initiation_threshold,
            'termination_threshold': self.termination_threshold
        }, filepath)
    
    def load_option(self, filepath: str) -> None:
        """Loads option"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.execution_count = checkpoint.get('execution_count', 0)
        self.success_count = checkpoint.get('success_count', 0)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        self.average_duration = checkpoint.get('average_duration', 0.0)
        self.initiation_threshold = checkpoint.get('initiation_threshold', 0.5)
        self.termination_threshold = checkpoint.get('termination_threshold', 0.8)


# Specialized options for crypto trading

def create_trend_following_option(state_dim: int = 15, action_dim: int = 3, device: str = "cpu") -> OptionPolicy:
    """Creates option for following trend"""
    
    def trend_initiation(state: np.ndarray) -> bool:
        """Condition initiation: strong trend"""
        if len(state) < 3:
            return False
        price_change = state[0]
        volume = state[1] 
        volatility = state[2]
        
        return abs(price_change) > 0.01 and volume > 1000000 and volatility > 0.02
    
    def trend_termination(state: np.ndarray) -> bool:
        """Condition completion: attenuation trend or reaching goals"""
        if len(state) < 4:
            return False
        
        pnl = state[3]
        return pnl > 0.03 or pnl < -0.015  # Take profit or stop loss
    
    return OptionPolicy(
        option_id="trend_following",
        state_dim=state_dim,
        action_dim=action_dim,
        initiation_conditions=trend_initiation,
        termination_conditions=trend_termination,
        device=device
    )


def create_arbitrage_option(state_dim: int = 15, action_dim: int = 3, device: str = "cpu") -> OptionPolicy:
    """Creates option for arbitrage"""
    
    def arbitrage_initiation(state: np.ndarray) -> bool:
        """Condition initiation: detected spread"""
        if len(state) < 5:
            return False
        
        spread = abs(state[4])  # Difference prices between exchanges
        return spread > 0.005  # Minimum spread 0.5%
    
    def arbitrage_termination(state: np.ndarray) -> bool:
        """Condition completion: spread closed"""
        if len(state) < 5:
            return True
        
        spread = abs(state[4])
        return spread < 0.001  # Spread closed
    
    option = OptionPolicy(
        option_id="arbitrage",
        state_dim=state_dim,
        action_dim=action_dim,
        initiation_conditions=arbitrage_initiation,
        termination_conditions=arbitrage_termination,
        device=device
    )
    
    # More short options for arbitrage
    option.max_option_length = 20
    option.termination_threshold = 0.6
    
    return option


def create_mean_reversion_option(state_dim: int = 15, action_dim: int = 3, device: str = "cpu") -> OptionPolicy:
    """Creates option for return to average"""
    
    def reversion_initiation(state: np.ndarray) -> bool:
        """Condition initiation: deviation from average"""
        if len(state) < 6:
            return False
        
        price_deviation = abs(state[5])  # Deviation from sliding average
        return price_deviation > 0.02  # 2% deviation
    
    def reversion_termination(state: np.ndarray) -> bool:
        """Condition completion: return to average"""
        if len(state) < 6:
            return True
        
        price_deviation = abs(state[5])
        return price_deviation < 0.005  # Return to average
    
    return OptionPolicy(
        option_id="mean_reversion",
        state_dim=state_dim,
        action_dim=action_dim,
        initiation_conditions=reversion_initiation,
        termination_conditions=reversion_termination,
        device=device
    )


class OptionComposer:
    """
    Compositor options for creation complex strategies
    Coordinates execution and switching between options
    """
    
    def __init__(self, options: Dict[str, OptionPolicy]):
        self.options = options
        self.active_option: Optional[str] = None
        self.option_history: List[Tuple[str, OptionExecution]] = []
        self.option_transitions: Dict[str, List[str]] = {}
        
    def add_transition(self, from_option: str, to_options: List[str]) -> None:
        """Adds possible transitions between options"""
        self.option_transitions[from_option] = to_options
    
    def select_next_option(self, current_state: np.ndarray) -> Optional[str]:
        """Selects next option for execution"""
        available_options = []
        
        # Check all options on capability initiation
        for option_id, option in self.options.items():
            if option.can_initiate(current_state):
                available_options.append(option_id)
        
        if not available_options:
            return None
        
        # Simple heuristic selection: by success rate
        best_option = max(available_options, 
                         key=lambda x: self.options[x].get_success_rate())
        
        return best_option
    
    async def execute_option_sequence(self, 
                                    initial_state: np.ndarray,
                                    environment_step_fn: Callable,
                                    max_options: int = 5) -> List[OptionExecution]:
        """Executes sequence options"""
        results = []
        current_state = initial_state.copy()
        
        for _ in range(max_options):
            option_id = self.select_next_option(current_state)
            if not option_id:
                break
            
            option = self.options[option_id]
            execution = await option.execute_option(current_state, environment_step_fn)
            results.append(execution)
            
            self.option_history.append((option_id, execution))
            current_state = execution.final_state
            
            if not execution.success:
                break
        
        return results
    
    def get_option_statistics(self) -> Dict[str, Any]:
        """Returns statistics by all options"""
        stats = {}
        for option_id, option in self.options.items():
            stats[option_id] = option.get_statistics()
        
        return {
            'option_stats': stats,
            'total_executions': len(self.option_history),
            'active_option': self.active_option
        }