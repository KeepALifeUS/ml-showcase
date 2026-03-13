"""
Hierarchical Trading Agent Implementation
Integration all hierarchical components for creation production-ready trading agent.

enterprise Pattern:
- Multi-level decision architecture for complex trading strategies
- Production-ready agent orchestration with real-time adaptation
- Enterprise scalability with distributed execution and monitoring
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import logging
import asyncio
from collections import deque, defaultdict
import time
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Imports hierarchical components
from ..frameworks.options import OptionsFramework, Option, create_trend_following_option
from ..frameworks.ham import HAMFramework, HAMMachine, create_trend_following_ham
from ..frameworks.maxq import MAXQHierarchy, create_trading_maxq_hierarchy
from ..frameworks.hac import HACAgent, create_crypto_hac_agent
from ..policies.meta_policy import MetaPolicy, create_crypto_meta_policy, StrategyContext
from ..policies.skill_policy import SkillComposer, create_skill_library
from ..policies.option_policy import OptionComposer, create_trend_following_option as create_trend_option
from ..discovery.subgoal_discovery import SubgoalDiscoveryEngine, create_crypto_subgoal_discovery
from ..discovery.skill_discovery import SkillDiscoveryEngine, create_crypto_skill_discovery
from ..discovery.bottleneck_detection import BottleneckDetectionEngine, create_crypto_bottleneck_detector

logger = logging.getLogger(__name__)


class HierarchicalLevel(Enum):
    """Levels hierarchy in trading agent"""
    STRATEGIC = "strategic"      # Selection strategies (hours/days)
    TACTICAL = "tactical"        # Tactical decisions (minutes/hours)
    OPERATIONAL = "operational"  # Operational actions (seconds/minutes)
    EXECUTION = "execution"      # Execution orders (milliseconds/seconds)


class TradingMode(Enum):
    """Modes trading"""
    TRAINING = "training"
    TESTING = "testing"
    PRODUCTION = "production"
    SIMULATION = "simulation"


@dataclass
class TradingState:
    """State trading agent"""
    market_data: np.ndarray
    portfolio_state: Dict[str, float]
    risk_metrics: Dict[str, float]
    execution_context: Dict[str, Any]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingDecision:
    """Decision trading agent"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    expected_return: float
    risk_level: float
    reasoning: str
    level: HierarchicalLevel
    execution_priority: int = 1


@dataclass
class ExecutionResult:
    """Result execution trading decisions"""
    decision_id: str
    success: bool
    actual_return: float
    execution_time: float
    slippage: float
    fees: float
    market_impact: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalTradingAgent:
    """
    Main hierarchical trading agent
    Integrates all levels making decisions
    """
    
    def __init__(self,
                 state_dim: int = 20,
                 action_dim: int = 5,
                 frameworks: Optional[List[str]] = None,
                 trading_mode: TradingMode = TradingMode.SIMULATION,
                 device: str = "cpu"):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.trading_mode = trading_mode
        self.device = device
        
        if frameworks is None:
            frameworks = ['options', 'hac', 'meta_policy', 'skills']
        
        self.active_frameworks = frameworks
        
        # Initialization components
        self._initialize_frameworks()
        self._initialize_discovery_engines()
        
        # State agent
        self.current_state: Optional[TradingState] = None
        self.active_decisions: Dict[str, TradingDecision] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Performance
        self.total_return = 0.0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        
        # Asynchronous execution
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.decision_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        
        # Monitoring
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.decision_latencies: List[float] = []
        
        logger.info(f"Initialized hierarchical trading agent with frameworks: {frameworks}")
    
    def _initialize_frameworks(self) -> None:
        """Initializes hierarchical frameworks"""
        # Meta Policy (Strategic Level)
        if 'meta_policy' in self.active_frameworks:
            self.meta_policy = create_crypto_meta_policy(self.state_dim, self.device)
            logger.info("Initialized Meta Policy")
        
        # Options Framework (Tactical Level)
        if 'options' in self.active_frameworks:
            self.options_framework = OptionsFramework()
            
            # Add predefined options
            trend_option = create_trend_following_option(self.state_dim, self.action_dim)
            self.options_framework.register_option(trend_option)
            logger.info("Initialized Options Framework")
        
        # HAM Framework (Operational Level)
        if 'ham' in self.active_frameworks:
            self.ham_framework = HAMFramework()
            
            # Add predefined machines
            trend_ham = create_trend_following_ham(self.state_dim, self.action_dim)
            self.ham_framework.register_machine(trend_ham)
            logger.info("Initialized HAM Framework")
        
        # MAXQ Hierarchy (Multi-level Value Functions)
        if 'maxq' in self.active_frameworks:
            self.maxq_hierarchy = create_trading_maxq_hierarchy(self.state_dim)
            logger.info("Initialized MAXQ Hierarchy")
        
        # HAC Agent (Continuous Control)
        if 'hac' in self.active_frameworks:
            self.hac_agent = create_crypto_hac_agent(
                self.state_dim, self.action_dim, goal_dim=4, device=self.device
            )
            logger.info("Initialized HAC Agent")
        
        # Skill Library (Operational Skills)
        if 'skills' in self.active_frameworks:
            self.skill_library = create_skill_library(self.state_dim, goal_dim=4, device=self.device)
            self.skill_composer = SkillComposer(self.skill_library)
            logger.info("Initialized Skill Library")
        
        # Option Policies (Temporal Abstraction)
        if 'option_policies' in self.active_frameworks:
            option_policies = {
                'trend_following': create_trend_option(self.state_dim, self.action_dim, self.device)
            }
            self.option_composer = OptionComposer(option_policies)
            logger.info("Initialized Option Policies")
    
    def _initialize_discovery_engines(self) -> None:
        """Initializes engines detection"""
        # Subgoal Discovery
        self.subgoal_discovery = create_crypto_subgoal_discovery(self.state_dim)
        
        # Skill Discovery  
        self.skill_discovery = create_crypto_skill_discovery(self.action_dim, self.state_dim)
        
        # Bottleneck Detection
        self.bottleneck_detector = create_crypto_bottleneck_detector()
        
        logger.info("Initialized engines detection")
    
    async def make_decision(self, 
                           trading_state: TradingState,
                           urgency: float = 0.5) -> List[TradingDecision]:
        """Accepts hierarchical trading decisions"""
        start_time = time.time()
        decisions = []
        
        try:
            self.current_state = trading_state
            
            # Strategic Level - Meta Policy
            if hasattr(self, 'meta_policy'):
                strategic_decision = await self._make_strategic_decision(trading_state)
                if strategic_decision:
                    decisions.append(strategic_decision)
            
            # Tactical Level - Options or HAC
            if urgency > 0.7:  # High urgency - use HAC
                if hasattr(self, 'hac_agent'):
                    tactical_decisions = await self._make_hac_decision(trading_state)
                    decisions.extend(tactical_decisions)
            else:  # Regular urgency - use Options
                if hasattr(self, 'options_framework'):
                    tactical_decisions = await self._make_option_decisions(trading_state)
                    decisions.extend(tactical_decisions)
            
            # Operational Level - Skills or HAM
            if hasattr(self, 'skill_composer'):
                operational_decisions = await self._make_skill_decisions(trading_state)
                decisions.extend(operational_decisions)
            
            # Execution Level - Direct Actions
            execution_decisions = await self._make_execution_decisions(trading_state, decisions)
            decisions.extend(execution_decisions)
            
            # Save active decisions
            for decision in decisions:
                decision_id = f"{decision.level.value}_{time.time()}"
                self.active_decisions[decision_id] = decision
            
            decision_time = time.time() - start_time
            self.decision_latencies.append(decision_time)
            
            logger.info(f"Accepted {len(decisions)} decisions for {decision_time:.3f}with")
            return decisions
            
        except Exception as e:
            logger.error(f"Error when making decisions: {e}")
            return []
    
    async def _make_strategic_decision(self, trading_state: TradingState) -> Optional[TradingDecision]:
        """Accepts strategic decisions"""
        try:
            # Create context for meta-policy
            context = StrategyContext(
                market_state=trading_state.market_data,
                volatility=trading_state.risk_metrics.get('volatility', 0.02),
                volume=trading_state.execution_context.get('volume', 1000000),
                trend_strength=trading_state.risk_metrics.get('trend_strength', 0.0),
                spread=trading_state.execution_context.get('spread', 0.001),
                time_of_day=int(trading_state.timestamp % 86400 / 3600),  # Hour day
                market_regime=trading_state.metadata.get('market_regime', 'normal'),
                risk_level=trading_state.risk_metrics.get('risk_level', 0.5),
                portfolio_state=trading_state.portfolio_state
            )
            
            # Select strategy
            strategy_selection = self.meta_policy.select_strategy(context)
            
            decision = TradingDecision(
                action_type=f"strategy_{strategy_selection.strategy_type.value}",
                parameters=strategy_selection.parameters,
                confidence=strategy_selection.confidence,
                expected_return=0.0,  # Will be updated
                risk_level=strategy_selection.risk_assessment,
                reasoning=strategy_selection.reasoning,
                level=HierarchicalLevel.STRATEGIC,
                execution_priority=1
            )
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in strategic decision: {e}")
            return None
    
    async def _make_option_decisions(self, trading_state: TradingState) -> List[TradingDecision]:
        """Accepts decisions on basis options"""
        decisions = []
        
        try:
            # Retrieve available options
            available_options = self.options_framework.get_available_options(trading_state.market_data)
            
            for option in available_options[:2]:  # Maximum 2 options simultaneously
                # Execute option asynchronously
                result = await self.options_framework.execute_option(option, trading_state.market_data)
                
                if result.success:
                    decision = TradingDecision(
                        action_type=f"option_{option.option_id}",
                        parameters={
                            'option_id': option.option_id,
                            'expected_steps': result.steps_taken,
                            'confidence': option.get_success_rate()
                        },
                        confidence=option.get_success_rate(),
                        expected_return=result.total_reward,
                        risk_level=0.3,  # Options usually less risky
                        reasoning=f"Execution options {option.option_id}",
                        level=HierarchicalLevel.TACTICAL
                    )
                    decisions.append(decision)
                
        except Exception as e:
            logger.error(f"Error in making decisions by options: {e}")
        
        return decisions
    
    async def _make_hac_decision(self, trading_state: TradingState) -> List[TradingDecision]:
        """Accepts decisions with help HAC"""
        decisions = []
        
        try:
            # Simple goal for demonstration
            goal = np.array([0.02, 0.0, 0.0, 0.01])  # target_return, risk_limit, drawdown_limit, time_horizon
            
            # Retrieve hierarchical actions
            primitive_action, subgoals = self.hac_agent.hierarchical_action_selection(
                trading_state.market_data, goal
            )
            
            # Create decisions for of each level
            for level, subgoal in enumerate(subgoals):
                decision = TradingDecision(
                    action_type=f"hac_subgoal_level_{level}",
                    parameters={
                        'subgoal': subgoal.tolist(),
                        'level': level,
                        'primitive_action': primitive_action.tolist()
                    },
                    confidence=0.7,  # HAC usually quite confident
                    expected_return=goal[0],  # Expected profitability
                    risk_level=goal[1],  # Level risk
                    reasoning=f"HAC decision level {level}",
                    level=HierarchicalLevel.TACTICAL
                )
                decisions.append(decision)
            
        except Exception as e:
            logger.error(f"Error in HAC decision: {e}")
        
        return decisions
    
    async def _make_skill_decisions(self, trading_state: TradingState) -> List[TradingDecision]:
        """Accepts decisions on basis skills"""
        decisions = []
        
        try:
            # Define required skills on basis context
            required_skills = self._determine_required_skills(trading_state)
            
            for skill_name in required_skills:
                if skill_name in self.skill_library:
                    skill = self.skill_library[skill_name]
                    
                    decision = TradingDecision(
                        action_type=f"skill_{skill_name}",
                        parameters={
                            'skill_type': skill.skill_type.value,
                            'success_rate': skill.get_success_rate(),
                            'avg_execution_time': skill.get_average_execution_time()
                        },
                        confidence=skill.get_success_rate(),
                        expected_return=0.001,  # Skills give small, but stable profit
                        risk_level=0.1,  # Skills usually low-risk
                        reasoning=f"Execution skill {skill_name}",
                        level=HierarchicalLevel.OPERATIONAL
                    )
                    decisions.append(decision)
            
        except Exception as e:
            logger.error(f"Error in making decisions by skills: {e}")
        
        return decisions
    
    def _determine_required_skills(self, trading_state: TradingState) -> List[str]:
        """Determines required skills on basis state"""
        required_skills = []
        
        # Analyze market conditions
        volatility = trading_state.risk_metrics.get('volatility', 0.02)
        volume = trading_state.execution_context.get('volume', 1000000)
        spread = trading_state.execution_context.get('spread', 0.001)
        
        # Skill execution orders almost always needed
        required_skills.append('order_execution')
        
        # Management risks when high volatility
        if volatility > 0.03:
            required_skills.append('risk_control')
        
        # Scanning market when low volume
        if volume < 500000:
            required_skills.append('market_scanning')
        
        # Management position
        if trading_state.portfolio_state.get('total_exposure', 0) > 0.1:
            required_skills.append('position_management')
        
        return required_skills
    
    async def _make_execution_decisions(self, 
                                      trading_state: TradingState,
                                      higher_level_decisions: List[TradingDecision]) -> List[TradingDecision]:
        """Accepts decisions by execution on basis decisions higher levels"""
        execution_decisions = []
        
        try:
            for decision in higher_level_decisions:
                if decision.level in [HierarchicalLevel.STRATEGIC, HierarchicalLevel.TACTICAL]:
                    # Translate high-level decisions in specific actions
                    concrete_actions = self._translate_to_execution(decision, trading_state)
                    execution_decisions.extend(concrete_actions)
            
        except Exception as e:
            logger.error(f"Error in making decisions by execution: {e}")
        
        return execution_decisions
    
    def _translate_to_execution(self, 
                               decision: TradingDecision,
                               trading_state: TradingState) -> List[TradingDecision]:
        """Translates high-level decisions in specific actions"""
        execution_actions = []
        
        if decision.action_type.startswith('strategy_'):
            strategy_type = decision.action_type.replace('strategy_', '')
            
            if strategy_type == 'trend_following':
                # Create actions for following trend
                if trading_state.risk_metrics.get('trend_strength', 0) > 0:
                    action = TradingDecision(
                        action_type='buy_market',
                        parameters={
                            'quantity': decision.parameters.get('position_size', 0.1),
                            'stop_loss': decision.parameters.get('stop_loss', 0.02),
                            'take_profit': decision.parameters.get('take_profit', 0.05)
                        },
                        confidence=decision.confidence * 0.8,
                        expected_return=decision.expected_return,
                        risk_level=decision.risk_level,
                        reasoning=f"Execution strategies {strategy_type}",
                        level=HierarchicalLevel.EXECUTION,
                        execution_priority=2
                    )
                    execution_actions.append(action)
            
            elif strategy_type == 'arbitrage':
                # Create actions for arbitrage
                action = TradingDecision(
                    action_type='arbitrage_execute',
                    parameters={
                        'min_spread': decision.parameters.get('min_spread', 0.001),
                        'max_position': decision.parameters.get('max_position_size', 0.2)
                    },
                    confidence=decision.confidence,
                    expected_return=decision.expected_return,
                    risk_level=decision.risk_level,
                    reasoning=f"Execution arbitrage",
                    level=HierarchicalLevel.EXECUTION,
                    execution_priority=3
                )
                execution_actions.append(action)
        
        return execution_actions
    
    async def execute_decisions(self, decisions: List[TradingDecision]) -> List[ExecutionResult]:
        """Executes accepted decisions"""
        results = []
        
        # Sort by priority execution
        sorted_decisions = sorted(decisions, key=lambda d: d.execution_priority, reverse=True)
        
        for decision in sorted_decisions:
            try:
                result = await self._execute_single_decision(decision)
                results.append(result)
                
                # Update metrics performance
                self._update_performance_metrics(result)
                
            except Exception as e:
                logger.error(f"Error when execution decisions {decision.action_type}: {e}")
                
                # Create result with error
                error_result = ExecutionResult(
                    decision_id=f"{decision.level.value}_{time.time()}",
                    success=False,
                    actual_return=0.0,
                    execution_time=0.0,
                    slippage=0.0,
                    fees=0.0,
                    market_impact=0.0,
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        self.execution_history.extend(results)
        return results
    
    async def _execute_single_decision(self, decision: TradingDecision) -> ExecutionResult:
        """Executes one decision"""
        start_time = time.time()
        
        # Simulation execution (in real system here was would integration with exchange)
        await asyncio.sleep(0.01)  # Imitation time execution
        
        # Simple simulation results
        success = np.random.random() > 0.1  # 90% success rate
        actual_return = decision.expected_return * (0.8 + 0.4 * np.random.random())
        slippage = abs(np.random.normal(0, 0.001))
        fees = 0.001 * abs(actual_return)  # 0.1% commission
        market_impact = slippage * 0.5
        
        execution_time = time.time() - start_time
        
        result = ExecutionResult(
            decision_id=f"{decision.level.value}_{start_time}",
            success=success,
            actual_return=actual_return if success else -abs(actual_return) * 0.1,
            execution_time=execution_time,
            slippage=slippage,
            fees=fees,
            market_impact=market_impact,
            metadata={
                'action_type': decision.action_type,
                'confidence': decision.confidence,
                'parameters': decision.parameters
            }
        )
        
        return result
    
    def _update_performance_metrics(self, result: ExecutionResult) -> None:
        """Updates metrics performance"""
        self.total_return += result.actual_return
        
        # Update win rate
        self.performance_metrics['returns'].append(result.actual_return)
        wins = sum(1 for r in self.performance_metrics['returns'] if r > 0)
        self.win_rate = wins / len(self.performance_metrics['returns'])
        
        # Update Sharpe ratio
        if len(self.performance_metrics['returns']) > 1:
            returns_array = np.array(self.performance_metrics['returns'])
            self.sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        
        # Update maximum drawdown
        cumulative_returns = np.cumsum(self.performance_metrics['returns'])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        self.max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Save additional metrics
        self.performance_metrics['execution_times'].append(result.execution_time)
        self.performance_metrics['slippages'].append(result.slippage)
        self.performance_metrics['fees'].append(result.fees)
    
    async def learn_from_trajectories(self, 
                                    trajectories: List[List[np.ndarray]],
                                    actions_list: List[List[np.ndarray]],
                                    rewards_list: List[List[float]]) -> None:
        """Trains on basis trajectories"""
        logger.info(f"Begin training on {len(trajectories)} trajectories")
        
        try:
            # Detection subgoals
            subgoals = self.subgoal_discovery.discover_subgoals(trajectories, rewards_list)
            logger.info(f"Detected {len(subgoals)} subgoals")
            
            # Detection skills
            skills = self.skill_discovery.discover_skills(trajectories, actions_list, rewards_list)
            logger.info(f"Detected {len(skills)} skills")
            
            # Detection narrow places
            bottlenecks = self.bottleneck_detector.detect_bottlenecks(
                trajectories, actions_list, rewards_list
            )
            logger.info(f"Detected {len(bottlenecks)} narrow places")
            
            # Training components
            if hasattr(self, 'hac_agent'):
                await self._train_hac_agent(trajectories, actions_list, rewards_list)
            
            if hasattr(self, 'meta_policy'):
                await self._train_meta_policy(trajectories, rewards_list)
            
        except Exception as e:
            logger.error(f"Error when training: {e}")
    
    async def _train_hac_agent(self, 
                             trajectories: List[List[np.ndarray]],
                             actions_list: List[List[np.ndarray]],
                             rewards_list: List[List[float]]) -> None:
        """Trains HAC agent"""
        # Convert trajectory in HAC format
        from ..frameworks.hac import HACTransition
        
        for traj_idx, (states, actions, rewards) in enumerate(
            zip(trajectories, actions_list, rewards_list)
        ):
            episode_transitions = []
            
            for i in range(len(states) - 1):
                transition = HACTransition(
                    state=states[i],
                    goal=np.array([0.02, 0.0, 0.0, 0.01]),  # Simple goal
                    action=actions[i] if i < len(actions) else np.zeros(self.action_dim),
                    reward=rewards[i] if i < len(rewards) else 0.0,
                    next_state=states[i + 1],
                    done=i == len(states) - 2,
                    level=0,
                    intrinsic_reward=0.0
                )
                episode_transitions.append(transition)
            
            # Train on episode
            if episode_transitions:
                losses = self.hac_agent.train_episode(episode_transitions)
                logger.debug(f"HAC training on episode {traj_idx}: {losses}")
    
    async def _train_meta_policy(self, 
                               trajectories: List[List[np.ndarray]],
                               rewards_list: List[List[float]]) -> None:
        """Trains meta-policy"""
        # Create training data for meta-policy
        states = []
        actions = []
        rewards = []
        
        for trajectory, trajectory_rewards in zip(trajectories, rewards_list):
            for state, reward in zip(trajectory, trajectory_rewards):
                states.append(state)
                rewards.append(reward)
                # Simple strategy for training
                action = 0 if reward > 0 else 1  # trend_following vs mean_reversion
                actions.append(action)
        
        if len(states) > 10:
            # Convert in tensors
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)
            next_states_tensor = torch.FloatTensor(states[1:] + [states[-1]])  # Simple approximation
            dones_tensor = torch.zeros(len(states))
            
            # Training step
            losses = self.meta_policy.train_step(
                states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor
            )
            logger.debug(f"Meta Policy training: {losses}")
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive statistics agent"""
        stats = {
            'performance': {
                'total_return': self.total_return,
                'win_rate': self.win_rate,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown,
                'total_trades': len(self.execution_history),
                'avg_decision_latency': np.mean(self.decision_latencies) if self.decision_latencies else 0.0
            },
            'frameworks': {}
        }
        
        # Statistics frameworks
        if hasattr(self, 'options_framework'):
            stats['frameworks']['options'] = self.options_framework.get_statistics()
        
        if hasattr(self, 'hac_agent'):
            stats['frameworks']['hac'] = self.hac_agent.get_training_statistics()
        
        if hasattr(self, 'meta_policy'):
            stats['frameworks']['meta_policy'] = self.meta_policy.get_strategy_statistics()
        
        if hasattr(self, 'skill_library'):
            stats['frameworks']['skills'] = {
                skill_name: skill.get_success_rate() 
                for skill_name, skill in self.skill_library.items()
            }
        
        # Statistics detection
        stats['discovery'] = {
            'subgoals': self.subgoal_discovery.get_statistics(),
            'skills': self.skill_discovery.get_statistics(),
            'bottlenecks': self.bottleneck_detector.get_statistics()
        }
        
        return stats
    
    def save_agent(self, filepath: str) -> None:
        """Saves state agent"""
        agent_data = {
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'active_frameworks': self.active_frameworks,
                'trading_mode': self.trading_mode.value
            },
            'performance': {
                'total_return': self.total_return,
                'win_rate': self.win_rate,
                'sharpe_ratio': self.sharpe_ratio,
                'max_drawdown': self.max_drawdown
            },
            'execution_history': [
                {
                    'decision_id': result.decision_id,
                    'success': result.success,
                    'actual_return': result.actual_return,
                    'execution_time': result.execution_time,
                    'metadata': result.metadata
                }
                for result in self.execution_history[-1000:]  # Recent 1000 results
            ]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
        
        # Save components separately
        base_path = filepath.replace('.pkl', '')
        
        if hasattr(self, 'meta_policy'):
            self.meta_policy.save_policy(f"{base_path}_meta_policy.pkl")
        
        if hasattr(self, 'hac_agent'):
            self.hac_agent.save_models(f"{base_path}_hac")
        
        self.subgoal_discovery.save_subgoals(f"{base_path}_subgoals.pkl")
        self.skill_discovery.save_skills(f"{base_path}_skills.pkl")
        self.bottleneck_detector.save_bottlenecks(f"{base_path}_bottlenecks.pkl")
        
        logger.info(f"Agent saved in {filepath}")
    
    def load_agent(self, filepath: str) -> None:
        """Loads state agent"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        # Restore configuration
        config = agent_data['config']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.active_frameworks = config['active_frameworks']
        self.trading_mode = TradingMode(config['trading_mode'])
        
        # Restore performance
        performance = agent_data['performance']
        self.total_return = performance['total_return']
        self.win_rate = performance['win_rate']
        self.sharpe_ratio = performance['sharpe_ratio']
        self.max_drawdown = performance['max_drawdown']
        
        # Load components
        base_path = filepath.replace('.pkl', '')
        
        try:
            if hasattr(self, 'meta_policy'):
                self.meta_policy.load_policy(f"{base_path}_meta_policy.pkl")
            
            if hasattr(self, 'hac_agent'):
                self.hac_agent.load_models(f"{base_path}_hac")
            
            self.subgoal_discovery.load_subgoals(f"{base_path}_subgoals.pkl")
            self.skill_discovery.load_skills(f"{base_path}_skills.pkl")
            
            logger.info(f"Agent loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Some components not succeeded load: {e}")


# Factory for creation specialized agents

def create_trend_following_agent(state_dim: int = 20, device: str = "cpu") -> HierarchicalTradingAgent:
    """Creates agent for following trend"""
    return HierarchicalTradingAgent(
        state_dim=state_dim,
        action_dim=3,  # buy, hold, sell
        frameworks=['meta_policy', 'options', 'hac', 'skills'],
        trading_mode=TradingMode.SIMULATION,
        device=device
    )


def create_arbitrage_agent(state_dim: int = 25, device: str = "cpu") -> HierarchicalTradingAgent:
    """Creates agent for arbitrage"""
    return HierarchicalTradingAgent(
        state_dim=state_dim,
        action_dim=5,  # buy_exchange1, sell_exchange1, buy_exchange2, sell_exchange2, hold
        frameworks=['hac', 'skills', 'options'],
        trading_mode=TradingMode.SIMULATION,
        device=device
    )


def create_market_making_agent(state_dim: int = 30, device: str = "cpu") -> HierarchicalTradingAgent:
    """Creates agent for market-making"""
    return HierarchicalTradingAgent(
        state_dim=state_dim,
        action_dim=7,  # bid_price, bid_size, ask_price, ask_size, cancel_orders, adjust_spread, hold
        frameworks=['meta_policy', 'skills', 'ham'],
        trading_mode=TradingMode.SIMULATION,
        device=device
    )