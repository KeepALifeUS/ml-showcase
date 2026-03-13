"""
Hierarchical Training System Implementation
Comprehensive system training for all hierarchical components.

enterprise Pattern:
- Multi-level training orchestration for complex hierarchical systems
- Production-ready distributed training with adaptive curriculum
- Scalable training pipeline with monitoring and optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import asyncio
import time
import json
import pickle
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import threading
import queue
import wandb  # For logging experiments

# Imports agents and components
from ..agents.hierarchical_trader import HierarchicalTradingAgent, TradingState, TradingDecision
from ..agents.option_critic import OptionCriticAgent, TradingOptionCriticAgent, OptionCriticConfig
from ..frameworks.hac import HACAgent, HACTransition
from ..policies.meta_policy import MetaPolicy
from ..policies.skill_policy import Skill, SkillContext
from ..policies.option_policy import OptionPolicy

logger = logging.getLogger(__name__)


class TrainingPhase(Enum):
    """Phases training"""
    EXPLORATION = "exploration"      # Research and collection data
    SKILL_DISCOVERY = "skill_discovery"  # Detection skills
    HIERARCHICAL = "hierarchical"    # Hierarchical training
    FINE_TUNING = "fine_tuning"     # Final configuration
    EVALUATION = "evaluation"        # Estimation performance


class TrainingMode(Enum):
    """Modes training"""
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    DISTRIBUTED = "distributed"
    CURRICULUM = "curriculum"


@dataclass
class TrainingConfig:
    """Configuration training"""
    # Main parameters
    total_episodes: int = 10000
    max_episode_steps: int = 1000
    eval_frequency: int = 100
    save_frequency: int = 500
    
    # Parameters environment
    state_dim: int = 20
    action_dim: int = 3
    goal_dim: int = 4
    
    # Parameters training
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Hierarchical parameters
    num_options: int = 8
    num_skills: int = 10
    subgoal_freq: int = 10
    
    # Modes
    training_mode: TrainingMode = TrainingMode.SINGLE_AGENT
    training_phases: List[TrainingPhase] = field(default_factory=lambda: [
        TrainingPhase.EXPLORATION,
        TrainingPhase.SKILL_DISCOVERY,
        TrainingPhase.HIERARCHICAL,
        TrainingPhase.FINE_TUNING
    ])
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cpu"
    num_workers: int = 4


class TradingEnvironment:
    """Simulation trading environment"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_step = 0
        self.max_steps = config.max_episode_steps
        
        # Market data (simulation)
        self.price_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=1000)
        
        # State portfolio
        self.portfolio_value = 1.0
        self.position = 0.0
        self.cash = 1.0
        
        # Market conditions
        self.volatility = 0.02
        self.trend_strength = 0.0
        self.market_regime = "normal"
        
        self.reset()
    
    def reset(self) -> TradingState:
        """Resets environment"""
        self.current_step = 0
        self.portfolio_value = 1.0
        self.position = 0.0
        self.cash = 1.0
        
        # Generate initial market data
        self._generate_market_data()
        
        return self._get_current_state()
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[TradingState, float, bool, Dict[str, Any]]:
        """Executes step in environment"""
        self.current_step += 1
        
        # Process action
        reward = self._execute_action(action)
        
        # Update market data
        self._generate_market_data()
        
        # Check completion episode
        done = (self.current_step >= self.max_steps or 
                self.portfolio_value <= 0.5 or  # Stop loss
                self.portfolio_value >= 2.0)    # Take profit
        
        next_state = self._get_current_state()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'step': self.current_step
        }
        
        return next_state, reward, done, info
    
    def _execute_action(self, action: Union[int, np.ndarray]) -> float:
        """Executes trading action"""
        if isinstance(action, np.ndarray):
            # Continuous actions (for HAC)
            action_type = np.argmax(action[:3])  # First 3 component - discrete actions
            intensity = np.clip(action[3] if len(action) > 3 else 1.0, 0.1, 1.0)
        else:
            # Discrete actions
            action_type = action
            intensity = 1.0
        
        # Retrieve current price
        current_price = self.price_history[-1] if self.price_history else 1.0
        
        # Execute action
        if action_type == 0:  # Buy
            max_buy = self.cash / current_price
            quantity = max_buy * intensity * 0.1  # Maximum 10% for times
            
            if quantity > 0:
                cost = quantity * current_price * 1.001  # Commission 0.1%
                if cost <= self.cash:
                    self.position += quantity
                    self.cash -= cost
                    reward = 0.001  # Small reward for successful purchase
                else:
                    reward = -0.001  # Penalty for unsuccessful attempt
            else:
                reward = 0.0
                
        elif action_type == 2:  # Sell
            quantity = self.position * intensity * 0.1  # Maximum 10% positions
            
            if quantity > 0:
                revenue = quantity * current_price * 0.999  # Commission 0.1%
                self.position -= quantity
                self.cash += revenue
                reward = 0.001  # Small reward for successful sale
            else:
                reward = -0.001  # Penalty for unsuccessful attempt
                
        else:  # Hold
            reward = 0.0001  # Small reward for retention
        
        # Update value portfolio
        self.portfolio_value = self.cash + self.position * current_price
        
        # Additional reward/penalty for change value portfolio
        if len(self.price_history) > 1:
            price_change = (current_price - self.price_history[-2]) / self.price_history[-2]
            portfolio_change = price_change * (self.position * current_price / self.portfolio_value)
            reward += portfolio_change
        
        return reward
    
    def _generate_market_data(self) -> None:
        """Generates market data"""
        if not self.price_history:
            price = 1.0
        else:
            # Simple model random wandering with trend
            prev_price = self.price_history[-1]
            
            # Update trend
            self.trend_strength += np.random.normal(0, 0.01)
            self.trend_strength = np.clip(self.trend_strength, -0.1, 0.1)
            
            # Update volatility
            self.volatility += np.random.normal(0, 0.001)
            self.volatility = np.clip(self.volatility, 0.005, 0.1)
            
            # Generate change price
            trend_component = self.trend_strength
            random_component = np.random.normal(0, self.volatility)
            price_change = trend_component + random_component
            
            price = prev_price * (1 + price_change)
            price = max(price, 0.01)  # Minimum price
        
        self.price_history.append(price)
        
        # Generate volume trading
        base_volume = 1000000
        volume_change = np.random.exponential(0.5) - 0.5
        volume = base_volume * (1 + volume_change)
        self.volume_history.append(max(volume, 10000))
    
    def _get_current_state(self) -> TradingState:
        """Returns current state"""
        # Create market_data
        market_data = np.zeros(self.config.state_dim)
        
        if len(self.price_history) >= 10:
            # Price data
            recent_prices = list(self.price_history)[-10:]
            price_returns = [
                (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                for i in range(1, len(recent_prices))
            ]
            market_data[:len(price_returns)] = price_returns
            
            # Technical indicators
            if len(price_returns) >= 5:
                market_data[10] = np.mean(price_returns[-5:])  # SMA
                market_data[11] = np.std(price_returns[-5:])   # Volatility
                market_data[12] = price_returns[-1]            # Last return
        
        # State portfolio
        market_data[13] = self.position / max(self.portfolio_value, 0.01)
        market_data[14] = self.cash / max(self.portfolio_value, 0.01)
        market_data[15] = (self.portfolio_value - 1.0)  # Total return
        
        # Market conditions
        market_data[16] = self.volatility
        market_data[17] = self.trend_strength
        market_data[18] = self.current_step / self.max_steps  # Progress
        
        # Additional features
        if len(self.volume_history) > 0:
            market_data[19] = np.log(self.volume_history[-1] / 1000000)  # Volume indicator
        
        return TradingState(
            market_data=market_data,
            portfolio_state={
                'value': self.portfolio_value,
                'position': self.position,
                'cash': self.cash
            },
            risk_metrics={
                'volatility': self.volatility,
                'trend_strength': self.trend_strength,
                'risk_level': abs(self.position) / max(self.portfolio_value, 0.01)
            },
            execution_context={
                'volume': self.volume_history[-1] if self.volume_history else 1000000,
                'spread': 0.001,
                'timestamp': time.time()
            },
            timestamp=time.time()
        )


class HierarchicalTrainer:
    """
    Main class for training hierarchical agents
    Coordinates training all components
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialization environment
        self.env = TradingEnvironment(config)
        
        # Initialization agents
        self.agents: Dict[str, Any] = {}
        self.current_agent = None
        
        # Statistics training
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': [],
            'success_rates': [],
            'phase_progress': {},
            'component_losses': defaultdict(list)
        }
        
        # Current phase training
        self.current_phase = TrainingPhase.EXPLORATION
        self.phase_episode = 0
        self.total_episodes = 0
        
        # Saving and logging
        self.best_performance = -float('inf')
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Multithreading
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        logger.info(f"Initialized HierarchicalTrainer with configuration: {config}")
    
    def add_agent(self, name: str, agent: Any) -> None:
        """Adds agent for training"""
        self.agents[name] = agent
        logger.info(f"Added agent: {name}")
    
    def train(self, agent_name: str) -> Dict[str, Any]:
        """Launches full loop training"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        self.current_agent = self.agents[agent_name]
        
        logger.info(f"Begin training agent {agent_name}")
        start_time = time.time()
        
        try:
            # Traverse through all phases training
            for phase in self.config.training_phases:
                logger.info(f"Begin phase: {phase.value}")
                self.current_phase = phase
                self.phase_episode = 0
                
                phase_stats = self._train_phase(phase)
                self.training_stats['phase_progress'][phase.value] = phase_stats
                
                # Save checkpoint after of each phases
                self.checkpoint_manager.save_checkpoint(
                    self.current_agent, self.training_stats, f"{agent_name}_{phase.value}"
                )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed for {training_time:.2f} seconds")
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Error when training: {e}")
            raise
    
    def _train_phase(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Trains in within one phases"""
        phase_stats = {
            'episodes': 0,
            'avg_reward': 0.0,
            'avg_length': 0.0,
            'success_rate': 0.0,
            'phase_duration': 0.0
        }
        
        start_time = time.time()
        episodes_in_phase = self.config.total_episodes // len(self.config.training_phases)
        
        for episode in range(episodes_in_phase):
            episode_stats = self._train_episode(phase)
            
            # Update statistics
            self.training_stats['episode_rewards'].append(episode_stats['total_reward'])
            self.training_stats['episode_lengths'].append(episode_stats['episode_length'])
            self.training_stats['portfolio_values'].append(episode_stats['final_portfolio_value'])
            
            self.phase_episode += 1
            self.total_episodes += 1
            
            # Logging
            if episode % 100 == 0:
                logger.info(
                    f"Phase {phase.value}, Episode {episode}, "
                    f"Reward: {episode_stats['total_reward']:.3f}, "
                    f"Portfolio: {episode_stats['final_portfolio_value']:.3f}"
                )
            
            # Estimation and saving
            if episode % self.config.eval_frequency == 0:
                eval_stats = self._evaluate_agent()
                if eval_stats['avg_reward'] > self.best_performance:
                    self.best_performance = eval_stats['avg_reward']
                    self.checkpoint_manager.save_best_model(self.current_agent)
            
            if episode % self.config.save_frequency == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.current_agent, self.training_stats, 
                    f"episode_{self.total_episodes}"
                )
        
        phase_stats['episodes'] = episodes_in_phase
        phase_stats['phase_duration'] = time.time() - start_time
        
        # Compute final statistics phases
        recent_rewards = self.training_stats['episode_rewards'][-episodes_in_phase:]
        recent_lengths = self.training_stats['episode_lengths'][-episodes_in_phase:]
        recent_portfolios = self.training_stats['portfolio_values'][-episodes_in_phase:]
        
        phase_stats['avg_reward'] = np.mean(recent_rewards)
        phase_stats['avg_length'] = np.mean(recent_lengths)
        phase_stats['success_rate'] = sum(1 for p in recent_portfolios if p > 1.0) / len(recent_portfolios)
        
        return phase_stats
    
    def _train_episode(self, phase: TrainingPhase) -> Dict[str, Any]:
        """Trains one episode"""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Trajectory for training
        trajectory = []
        
        while not done and episode_length < self.config.max_episode_steps:
            # Retrieve action from agent
            action = self._get_agent_action(state, phase)
            
            # Execute step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Save experience
            trajectory.append({
                'state': state.market_data,
                'action': action,
                'reward': reward,
                'next_state': next_state.market_data,
                'done': done,
                'info': info
            })
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        # Train agent on trajectory
        self._train_agent_on_trajectory(trajectory, phase)
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'final_portfolio_value': info.get('portfolio_value', 1.0),
            'trajectory_length': len(trajectory)
        }
    
    def _get_agent_action(self, state: TradingState, phase: TrainingPhase) -> Union[int, np.ndarray]:
        """Gets action from current agent"""
        if isinstance(self.current_agent, HierarchicalTradingAgent):
            # For hierarchical agent get decisions
            decisions = asyncio.run(self.current_agent.make_decision(state))
            
            # Select executive decision
            exec_decisions = [d for d in decisions if d.level.value == "execution"]
            if exec_decisions:
                decision = exec_decisions[0]
                if decision.action_type == "buy_market":
                    return 0
                elif decision.action_type == "sell_market":
                    return 2
                else:
                    return 1
            else:
                return 1  # Hold by default
        
        elif isinstance(self.current_agent, (OptionCriticAgent, TradingOptionCriticAgent)):
            # For Option-Critic agent
            option, action, terminated = self.current_agent.step(state.market_data)
            return action
        
        elif isinstance(self.current_agent, HACAgent):
            # For HAC agent
            goal = np.array([0.02, 0.0, 0.0, 0.01])  # Simple goal
            action = self.current_agent.select_action(state.market_data, goal, level=0)
            return action
        
        else:
            # For other types agents - random action
            return np.random.randint(0, self.config.action_dim)
    
    def _train_agent_on_trajectory(self, trajectory: List[Dict[str, Any]], phase: TrainingPhase) -> None:
        """Trains agent on trajectory"""
        try:
            if isinstance(self.current_agent, HierarchicalTradingAgent):
                # For hierarchical agent use trajectory for discovery
                if phase == TrainingPhase.SKILL_DISCOVERY:
                    states = [step['state'] for step in trajectory]
                    actions = [step['action'] for step in trajectory]
                    rewards = [step['reward'] for step in trajectory]
                    
                    asyncio.run(self.current_agent.learn_from_trajectories(
                        [states], [actions], [rewards]
                    ))
            
            elif isinstance(self.current_agent, (OptionCriticAgent, TradingOptionCriticAgent)):
                # For Option-Critic save experience and train
                for i, step in enumerate(trajectory):
                    if i < len(trajectory) - 1:
                        self.current_agent.store_experience(
                            state=step['state'],
                            option=0,  # Simplification - use option 0
                            action=step['action'],
                            reward=step['reward'],
                            next_state=step['next_state'],
                            terminated=False,
                            done=step['done']
                        )
                
                # Training step
                losses = self.current_agent.train_step()
                if losses:
                    for loss_name, loss_value in losses.items():
                        self.training_stats['component_losses'][f'option_critic_{loss_name}'].append(loss_value)
            
            elif isinstance(self.current_agent, HACAgent):
                # For HAC agent
                hac_transitions = []
                goal = np.array([0.02, 0.0, 0.0, 0.01])
                
                for step in trajectory:
                    transition = HACTransition(
                        state=step['state'],
                        goal=goal,
                        action=np.array([step['action']]) if isinstance(step['action'], int) else step['action'],
                        reward=step['reward'],
                        next_state=step['next_state'],
                        done=step['done'],
                        level=0,
                        intrinsic_reward=0.0
                    )
                    hac_transitions.append(transition)
                
                if hac_transitions:
                    losses = self.current_agent.train_episode(hac_transitions)
                    for loss_name, loss_values in losses.items():
                        if loss_values:
                            self.training_stats['component_losses'][f'hac_{loss_name}'].extend(loss_values)
        
        except Exception as e:
            logger.error(f"Error when training agent on trajectory: {e}")
    
    def _evaluate_agent(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluates performance agent"""
        eval_rewards = []
        eval_portfolios = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            episode_length = 0
            
            while not done and episode_length < self.config.max_episode_steps:
                action = self._get_agent_action(state, TrainingPhase.EVALUATION)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            eval_rewards.append(episode_reward)
            eval_portfolios.append(info.get('portfolio_value', 1.0))
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_portfolio': np.mean(eval_portfolios),
            'success_rate': sum(1 for p in eval_portfolios if p > 1.0) / len(eval_portfolios)
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Returns summary training"""
        if not self.training_stats['episode_rewards']:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'final_performance': {
                'avg_reward': np.mean(self.training_stats['episode_rewards'][-100:]),
                'avg_portfolio': np.mean(self.training_stats['portfolio_values'][-100:]),
                'success_rate': sum(1 for p in self.training_stats['portfolio_values'][-100:] if p > 1.0) / min(100, len(self.training_stats['portfolio_values']))
            },
            'best_performance': self.best_performance,
            'phase_progress': self.training_stats['phase_progress'],
            'component_losses': {
                key: np.mean(values[-100:]) if values else 0.0
                for key, values in self.training_stats['component_losses'].items()
            }
        }


class CheckpointManager:
    """Manager for saving and loading checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, agent: Any, stats: Dict[str, Any], name: str) -> None:
        """Saves checkpoint agent"""
        checkpoint_path = f"{self.checkpoint_dir}/{name}.pkl"
        
        try:
            if hasattr(agent, 'save_agent'):
                agent.save_agent(checkpoint_path)
            elif hasattr(agent, 'save_model'):
                agent.save_model(checkpoint_path)
            else:
                # Universal saving
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({'agent': agent, 'stats': stats}, f)
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error when saving checkpoint: {e}")
    
    def save_best_model(self, agent: Any) -> None:
        """Saves best model"""
        self.save_checkpoint(agent, {}, "best_model")
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[Any, Dict[str, Any]]:
        """Loads checkpoint"""
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict) and 'agent' in data:
                return data['agent'], data.get('stats', {})
            else:
                return data, {}
                
        except Exception as e:
            logger.error(f"Error when loading checkpoint: {e}")
            raise


class MultiAgentTrainer(HierarchicalTrainer):
    """Trainer for multiple agents"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.agent_performances: Dict[str, float] = {}
    
    def train_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Trains all registered agents"""
        results = {}
        
        for agent_name in self.agents.keys():
            logger.info(f"Begin training agent {agent_name}")
            
            try:
                agent_stats = self.train(agent_name)
                results[agent_name] = agent_stats
                
                # Estimation final performance
                eval_stats = self._evaluate_agent()
                self.agent_performances[agent_name] = eval_stats['avg_reward']
                
            except Exception as e:
                logger.error(f"Error when training agent {agent_name}: {e}")
                results[agent_name] = {'error': str(e)}
        
        return results
    
    def get_best_agent(self) -> Tuple[str, Any]:
        """Returns best agent"""
        if not self.agent_performances:
            raise ValueError("No trained agents")
        
        best_agent_name = max(self.agent_performances, key=self.agent_performances.get)
        return best_agent_name, self.agents[best_agent_name]


# Factory for creation trainers

def create_option_critic_trainer(state_dim: int = 20, 
                                num_options: int = 8,
                                device: str = "cpu") -> HierarchicalTrainer:
    """Creates trainer for Option-Critic agent"""
    config = TrainingConfig(
        state_dim=state_dim,
        action_dim=3,
        num_options=num_options,
        device=device,
        training_phases=[
            TrainingPhase.EXPLORATION,
            TrainingPhase.HIERARCHICAL,
            TrainingPhase.FINE_TUNING
        ]
    )
    
    trainer = HierarchicalTrainer(config)
    
    # Create and add Option-Critic agent
    from ..agents.option_critic import create_crypto_option_critic
    agent = create_crypto_option_critic(state_dim, num_options, device)
    trainer.add_agent("option_critic", agent)
    
    return trainer


def create_hierarchical_trader_trainer(state_dim: int = 20,
                                     device: str = "cpu") -> HierarchicalTrainer:
    """Creates trainer for hierarchical trading agent"""
    config = TrainingConfig(
        state_dim=state_dim,
        action_dim=3,
        device=device,
        training_phases=[
            TrainingPhase.EXPLORATION,
            TrainingPhase.SKILL_DISCOVERY,
            TrainingPhase.HIERARCHICAL,
            TrainingPhase.FINE_TUNING
        ]
    )
    
    trainer = HierarchicalTrainer(config)
    
    # Create and add hierarchical agent
    from ..agents.hierarchical_trader import create_trend_following_agent
    agent = create_trend_following_agent(state_dim, device)
    trainer.add_agent("hierarchical_trader", agent)
    
    return trainer


def create_multi_agent_trainer(state_dim: int = 20, device: str = "cpu") -> MultiAgentTrainer:
    """Creates trainer for multiple agents"""
    config = TrainingConfig(
        state_dim=state_dim,
        device=device,
        training_mode=TrainingMode.MULTI_AGENT
    )
    
    trainer = MultiAgentTrainer(config)
    
    # Add various agents
    from ..agents.hierarchical_trader import create_trend_following_agent, create_arbitrage_agent
    from ..agents.option_critic import create_crypto_option_critic
    
    trainer.add_agent("trend_following", create_trend_following_agent(state_dim, device))
    trainer.add_agent("arbitrage", create_arbitrage_agent(state_dim, device))
    trainer.add_agent("option_critic", create_crypto_option_critic(state_dim, device=device))
    
    return trainer


# Utility function for training

def run_training_experiment(trainer: HierarchicalTrainer, 
                          agent_name: str,
                          experiment_name: str = "hierarchical_rl_experiment") -> Dict[str, Any]:
    """Launches full experiment with logging"""
    
    # Initialize wandb for logging (optionally)
    try:
        wandb.init(project=experiment_name, name=f"{agent_name}_training")
        use_wandb = True
    except:
        use_wandb = False
        logger.warning("Wandb unavailable, skip logging")
    
    # Run training
    start_time = time.time()
    results = trainer.train(agent_name)
    training_time = time.time() - start_time
    
    # Retrieve summary
    summary = trainer.get_training_summary()
    summary['training_time'] = training_time
    
    # Log results
    if use_wandb:
        wandb.log(summary)
        wandb.finish()
    
    logger.info(f"Experiment completed for {training_time:.2f} seconds")
    logger.info(f"Final performance: {summary.get('final_performance', {})}")
    
    return summary