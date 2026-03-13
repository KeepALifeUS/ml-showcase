"""
Skill Policy Implementation for Hierarchical RL
Implements low-level skills for execution specific trading actions.

enterprise Pattern:
- Modular skill composition for reusable trading behaviors
- Production-ready skill transfer learning for adaptive execution
- Hierarchical skill coordination with multi-level abstraction
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
from collections import deque, defaultdict
import copy

logger = logging.getLogger(__name__)


class SkillType(Enum):
    """Types trading skills"""
    ORDER_EXECUTION = "order_execution"
    POSITION_MANAGEMENT = "position_management"
    RISK_CONTROL = "risk_control"
    MARKET_SCANNING = "market_scanning"
    PRICE_PREDICTION = "price_prediction"
    LIQUIDITY_PROVISION = "liquidity_provision"
    SLIPPAGE_MINIMIZATION = "slippage_minimization"
    TIMING_OPTIMIZATION = "timing_optimization"


class SkillStatus(Enum):
    """Statuses execution skills"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class SkillExecution:
    """Result execution skill"""
    skill_type: SkillType
    success: bool
    execution_time: float
    actions_taken: List[np.ndarray]
    final_state: np.ndarray
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    """Context for execution skill"""
    current_state: np.ndarray
    target_goal: np.ndarray
    constraints: Dict[str, Any]
    environment_info: Dict[str, Any]
    urgency_level: float = 0.5
    risk_tolerance: float = 0.5


class SkillNetwork(nn.Module):
    """Neural network for skill"""
    
    def __init__(self,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 skill_type: SkillType,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.skill_type = skill_type
        self.action_dim = action_dim
        input_dim = state_dim + goal_dim
        
        # Feature extractor specific for type skill
        self.feature_extractor = self._build_feature_extractor(input_dim, skill_type)
        
        # Actor head
        self.actor = self._build_actor_head(hidden_dims[-1], action_dim)
        
        # Critic head
        self.critic = self._build_critic_head(hidden_dims[-1])
        
        # Termination head (determines when skill completed)
        self.termination = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Progress head (evaluates progress to goals)
        self.progress = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _build_feature_extractor(self, input_dim: int, skill_type: SkillType) -> nn.Module:
        """Creates feature extractor specific for type skill"""
        if skill_type == SkillType.ORDER_EXECUTION:
            # Focus on price action and order book
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        
        elif skill_type == SkillType.RISK_CONTROL:
            # Focus on risk metrics and portfolio state
            return nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
        
        elif skill_type == SkillType.MARKET_SCANNING:
            # Focus on market indicators and patterns
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        
        else:
            # Total extractor
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
    
    def _build_actor_head(self, input_dim: int, action_dim: int) -> nn.Module:
        """Creates actor head"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim * 2)  # mean and log_std for continuous actions
        )
    
    def _build_critic_head(self, input_dim: int) -> nn.Module:
        """Creates critic head"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: (action_params, value, termination_prob, progress)
        """
        x = torch.cat([state, goal], dim=-1)
        features = self.feature_extractor(x)
        
        action_params = self.actor(features)
        value = self.critic(features)
        termination_prob = self.termination(features)
        progress = self.progress(features)
        
        return action_params, value, termination_prob, progress
    
    def sample_action(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples action and returns log_prob"""
        action_params, _, _, _ = self.forward(state, goal)
        
        # Split on mean and log_std
        mean, log_std = torch.chunk(action_params, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        
        # Sample action
        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(-1, keepdim=True)
        
        return action, log_prob


class Skill:
    """
    Base class for trading skill
    Encapsulates specific behavior and training
    """
    
    def __init__(self,
                 skill_type: SkillType,
                 state_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 device: str = "cpu"):
        
        self.skill_type = skill_type
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Create network
        self.network = SkillNetwork(
            state_dim, goal_dim, action_dim, skill_type
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Target network for stable training
        self.target_network = copy.deepcopy(self.network)
        self.target_update_freq = 100
        self.update_counter = 0
        
        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=10000)
        
        # Statistics execution
        self.execution_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.execution_times: List[float] = []
        
        # State execution
        self.status = SkillStatus.INACTIVE
        self.current_context: Optional[SkillContext] = None
        self.start_time: Optional[float] = None
        
    async def execute(self, 
                     context: SkillContext, 
                     max_steps: int = 100) -> SkillExecution:
        """Executes skill until completion"""
        self.status = SkillStatus.ACTIVE
        self.current_context = context
        self.start_time = asyncio.get_event_loop().time()
        
        current_state = context.current_state.copy()
        actions_taken = []
        total_reward = 0.0
        
        try:
            for step in range(max_steps):
                # Retrieve action from policy
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(self.device)
                goal_tensor = torch.FloatTensor(context.target_goal).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_params, value, termination_prob, progress = self.network(state_tensor, goal_tensor)
                    mean, log_std = torch.chunk(action_params, 2, dim=-1)
                    std = log_std.exp()
                    
                    action = torch.normal(mean, std)
                    actions_taken.append(action.cpu().numpy().flatten())
                
                # Execute action and get new state
                next_state, reward = await self._execute_action(
                    current_state, action.cpu().numpy().flatten(), context
                )
                
                total_reward += reward
                current_state = next_state
                
                # Check condition completion
                if termination_prob.item() > 0.8 or self._is_goal_achieved(current_state, context.target_goal):
                    self.status = SkillStatus.COMPLETED
                    break
                
                await asyncio.sleep(0.001)  # Asynchrony
            
            execution_time = asyncio.get_event_loop().time() - self.start_time
            success = self.status == SkillStatus.COMPLETED
            
            # Update statistics
            self.execution_count += 1
            if success:
                self.success_count += 1
            self.total_reward += total_reward
            self.execution_times.append(execution_time)
            
            execution = SkillExecution(
                skill_type=self.skill_type,
                success=success,
                execution_time=execution_time,
                actions_taken=actions_taken,
                final_state=current_state,
                reward=total_reward,
                metadata={
                    'steps_taken': len(actions_taken),
                    'avg_progress': progress.item() if 'progress' in locals() else 0.0
                }
            )
            
            return execution
            
        except Exception as e:
            logger.error(f"Error execution skill {self.skill_type.value}: {e}")
            self.status = SkillStatus.FAILED
            return SkillExecution(
                skill_type=self.skill_type,
                success=False,
                execution_time=asyncio.get_event_loop().time() - self.start_time,
                actions_taken=actions_taken,
                final_state=current_state,
                reward=total_reward
            )
        
        finally:
            self.status = SkillStatus.INACTIVE
            self.current_context = None
    
    async def _execute_action(self, 
                             state: np.ndarray, 
                             action: np.ndarray, 
                             context: SkillContext) -> Tuple[np.ndarray, float]:
        """Executes action in environment (is overridden in subclasses)"""
        # Base simulation
        next_state = state.copy()
        
        # Simple change state on basis actions
        if len(next_state) > 0 and len(action) > 0:
            next_state[0] += action[0] * 0.001
        
        # Simple reward function
        progress = self._compute_progress(state, next_state, context.target_goal)
        reward = progress - 0.001  # Penalty for each step
        
        return next_state, reward
    
    def _compute_progress(self, 
                         state: np.ndarray, 
                         next_state: np.ndarray, 
                         goal: np.ndarray) -> float:
        """Computes progress to goals"""
        current_distance = np.linalg.norm(state - goal)
        next_distance = np.linalg.norm(next_state - goal)
        return max(0, current_distance - next_distance)
    
    def _is_goal_achieved(self, state: np.ndarray, goal: np.ndarray, tolerance: float = 0.1) -> bool:
        """Checks reaching goals"""
        return np.linalg.norm(state - goal) < tolerance
    
    def store_experience(self, 
                        state: np.ndarray,
                        goal: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool) -> None:
        """Saves experience in replay buffer"""
        self.replay_buffer.append({
            'state': state,
            'goal': goal,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train_step(self, batch_size: int = 64) -> Dict[str, float]:
        """Training step skill"""
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        import random
        batch = random.sample(self.replay_buffer, batch_size)
        
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        goals = torch.FloatTensor([exp['goal'] for exp in batch]).to(self.device)
        actions = torch.FloatTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.FloatTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Forward pass
        action_params, values, term_probs, progress = self.network(states, goals)
        mean, log_std = torch.chunk(action_params, 2, dim=-1)
        
        # Target values
        with torch.no_grad():
            _, next_values, _, _ = self.target_network(next_states, goals)
            targets = rewards.unsqueeze(1) + 0.99 * next_values * (1 - dones.unsqueeze(1))
        
        # Losses
        value_loss = F.mse_loss(values, targets)
        
        # Policy loss (REINFORCE with baseline)
        advantages = (targets - values).detach()
        std = log_std.exp()
        normal = Normal(mean, std)
        log_probs = normal.log_prob(actions).sum(-1, keepdim=True)
        policy_loss = -(log_probs * advantages).mean()
        
        # Progress loss (train predict progress)
        actual_progress = torch.FloatTensor([
            self._compute_progress(batch[i]['state'], batch[i]['next_state'], batch[i]['goal'])
            for i in range(len(batch))
        ]).unsqueeze(1).to(self.device)
        progress_loss = F.mse_loss(progress, actual_progress)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * progress_loss
        
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
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'progress_loss': progress_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_success_rate(self) -> float:
        """Returns share successful executions"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_average_execution_time(self) -> float:
        """Returns average time execution"""
        if not self.execution_times:
            return 0.0
        return np.mean(self.execution_times)
    
    def save_skill(self, filepath: str) -> None:
        """Saves skill"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'skill_type': self.skill_type.value,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'total_reward': self.total_reward
        }, filepath)
    
    def load_skill(self, filepath: str) -> None:
        """Loads skill"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.execution_count = checkpoint.get('execution_count', 0)
        self.success_count = checkpoint.get('success_count', 0)
        self.total_reward = checkpoint.get('total_reward', 0.0)


# Specialized skills for crypto trading

class OrderExecutionSkill(Skill):
    """Skill execution orders with minimization slippage"""
    
    def __init__(self, state_dim: int, goal_dim: int, device: str = "cpu"):
        super().__init__(
            SkillType.ORDER_EXECUTION, 
            state_dim, 
            goal_dim, 
            action_dim=3,  # [price_offset, quantity_ratio, timing_delay]
            device=device
        )
    
    async def _execute_action(self, 
                             state: np.ndarray, 
                             action: np.ndarray, 
                             context: SkillContext) -> Tuple[np.ndarray, float]:
        """Execution trading order"""
        # Extract parameters actions
        price_offset = action[0]  # Deviation from current price
        quantity_ratio = torch.sigmoid(torch.tensor(action[1])).item()  # Share from target volume
        timing_delay = max(0, action[2])  # Delay in seconds
        
        # Simulation execution order
        await asyncio.sleep(min(timing_delay * 0.001, 0.1))  # Scaled delay
        
        next_state = state.copy()
        
        # Update position
        if len(next_state) > 2:
            next_state[2] += quantity_ratio * context.target_goal[0]  # Target position
        
        # Compute slippage and market impact
        slippage = abs(price_offset) * 0.1  # Simple model slippage
        market_impact = quantity_ratio * 0.05  # Market impact from size order
        
        # Reward = negative slippage and market impact
        reward = -(slippage + market_impact) + 0.01  # Base reward for execution
        
        return next_state, reward


class RiskControlSkill(Skill):
    """Skill management risks"""
    
    def __init__(self, state_dim: int, goal_dim: int, device: str = "cpu"):
        super().__init__(
            SkillType.RISK_CONTROL,
            state_dim,
            goal_dim,
            action_dim=4,  # [position_adjustment, stop_loss, take_profit, hedge_ratio]
            device=device
        )
    
    async def _execute_action(self, 
                             state: np.ndarray, 
                             action: np.ndarray, 
                             context: SkillContext) -> Tuple[np.ndarray, float]:
        """Management risks positions"""
        position_adjustment = action[0]
        stop_loss = torch.sigmoid(torch.tensor(action[1])).item()
        take_profit = torch.sigmoid(torch.tensor(action[2])).item()
        hedge_ratio = torch.sigmoid(torch.tensor(action[3])).item()
        
        next_state = state.copy()
        
        # Apply risk controls
        current_risk = abs(state[3]) if len(state) > 3 else 0.0  # PnL as measure risk
        target_risk = context.risk_tolerance
        
        # Adjust position for reduction risk
        if current_risk > target_risk:
            if len(next_state) > 2:
                next_state[2] *= (1 + position_adjustment * 0.1)  # Reduce position
        
        # Reward = reduction risk
        risk_reduction = max(0, current_risk - abs(next_state[3]) if len(next_state) > 3 else 0)
        reward = risk_reduction * 10 - 0.001  # Penalty for each action
        
        return next_state, reward


class MarketScanningSkill(Skill):
    """Skill scanning market on capabilities"""
    
    def __init__(self, state_dim: int, goal_dim: int, device: str = "cpu"):
        super().__init__(
            SkillType.MARKET_SCANNING,
            state_dim,
            goal_dim,
            action_dim=5,  # [scan_focus, timeframe, filters, threshold, priority]
            device=device
        )
    
    async def _execute_action(self, 
                             state: np.ndarray, 
                             action: np.ndarray, 
                             context: SkillContext) -> Tuple[np.ndarray, float]:
        """Scanning market"""
        # Simulation scanning
        await asyncio.sleep(0.01)  # Time on scanning
        
        next_state = state.copy()
        
        # Update market indicators
        if len(next_state) > 5:
            # Simulate detection patterns
            pattern_strength = np.random.exponential(0.1)
            next_state[5] = pattern_strength
        
        # Reward = quality detected capabilities
        reward = pattern_strength if 'pattern_strength' in locals() else 0.0
        
        return next_state, reward


# Factory for creation skills

def create_skill_library(state_dim: int = 15, goal_dim: int = 5, device: str = "cpu") -> Dict[str, Skill]:
    """Creates library skills for crypto trading"""
    skills = {
        SkillType.ORDER_EXECUTION.value: OrderExecutionSkill(state_dim, goal_dim, device),
        SkillType.RISK_CONTROL.value: RiskControlSkill(state_dim, goal_dim, device),
        SkillType.MARKET_SCANNING.value: MarketScanningSkill(state_dim, goal_dim, device),
        SkillType.POSITION_MANAGEMENT.value: Skill(
            SkillType.POSITION_MANAGEMENT, state_dim, goal_dim, 4, device=device
        ),
        SkillType.LIQUIDITY_PROVISION.value: Skill(
            SkillType.LIQUIDITY_PROVISION, state_dim, goal_dim, 3, device=device
        )
    }
    
    return skills


class SkillComposer:
    """
    Compositor skills for creation complex behavior
    Coordinates execution multiple skills
    """
    
    def __init__(self, skills: Dict[str, Skill]):
        self.skills = skills
        self.active_skills: List[Skill] = []
        self.skill_dependencies: Dict[str, List[str]] = {}
        
    def add_dependency(self, skill_name: str, dependencies: List[str]) -> None:
        """Adds dependencies between skills"""
        self.skill_dependencies[skill_name] = dependencies
    
    async def execute_skill_sequence(self, 
                                   skill_sequence: List[str], 
                                   context: SkillContext) -> List[SkillExecution]:
        """Executes sequence skills"""
        results = []
        
        for skill_name in skill_sequence:
            if skill_name in self.skills:
                skill = self.skills[skill_name]
                result = await skill.execute(context)
                results.append(result)
                
                # Update context for next skill
                context.current_state = result.final_state
            else:
                logger.warning(f"Skill {skill_name} not found")
        
        return results
    
    async def execute_parallel_skills(self, 
                                    skill_names: List[str], 
                                    context: SkillContext) -> List[SkillExecution]:
        """Executes skills in parallel"""
        tasks = []
        
        for skill_name in skill_names:
            if skill_name in self.skills:
                skill = self.skills[skill_name]
                task = asyncio.create_task(skill.execute(context))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if isinstance(r, SkillExecution)]
        
        return []