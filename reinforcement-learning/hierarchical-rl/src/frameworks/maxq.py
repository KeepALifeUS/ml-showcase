"""
MAXQ Value Function Decomposition Implementation
Implementation hierarchical decomposition value functions for complex trading strategies.

enterprise Pattern:
- Hierarchical value decomposition for scalable training
- Production-ready MAXQ learning with adaptive subtask selection
- Value function approximation with neural networks for complex state spaces
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import asyncio
import pickle
import json
from collections import defaultdict, deque
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class MAXQNodeType(Enum):
    """Types nodes in MAXQ decomposition"""
    COMPOSITE = "composite"    # Composite task
    PRIMITIVE = "primitive"    # Primitive action
    MAX = "max"               # MAX node (selection subtasks)


@dataclass
class MAXQState:
    """State in MAXQ hierarchy"""
    environment_state: np.ndarray
    task_stack: List[str]
    subtask_completion: Dict[str, bool]
    variables: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'MAXQState':
        """Creates copy state"""
        return MAXQState(
            environment_state=self.environment_state.copy(),
            task_stack=self.task_stack.copy(),
            subtask_completion=self.subtask_completion.copy(),
            variables=self.variables.copy()
        )


@dataclass
class MAXQTransition:
    """Transition in MAXQ system"""
    from_state: MAXQState
    action: Union[str, int]  # Subtask or primitive action
    to_state: MAXQState
    reward: float
    terminal: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAXQValueFunction(nn.Module):
    """Neural network for approximation value function"""
    
    def __init__(self, 
                 state_dim: int,
                 task_embedding_dim: int = 32,
                 hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.task_embedding_dim = task_embedding_dim
        
        # Embeddings for tasks
        self.task_embeddings = nn.Embedding(100, task_embedding_dim)  # Until 100 tasks
        
        # Main network
        input_dim = state_dim + task_embedding_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))  # Scalar value
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """Computes Q-value for state and tasks"""
        task_emb = self.task_embeddings(task_id)
        combined = torch.cat([state, task_emb], dim=-1)
        return self.network(combined)


class MAXQNode(ABC):
    """Abstract base class for nodes MAXQ"""
    
    def __init__(self, node_id: str, node_type: MAXQNodeType, parent: Optional['MAXQNode'] = None):
        self.node_id = node_id
        self.node_type = node_type
        self.parent = parent
        self.children: List['MAXQNode'] = []
        self.task_id = hash(node_id) % 100  # Simple ID for embeddings
        
        # Statistics
        self.execution_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        
    @abstractmethod
    def is_terminal(self, state: MAXQState) -> bool:
        """Checks, is whether state terminal for of this tasks"""
        pass
    
    @abstractmethod
    def get_policy_action(self, state: MAXQState) -> Union[str, int]:
        """Returns action according to policy"""
        pass
    
    def add_child(self, child: 'MAXQNode') -> None:
        """Adds child node"""
        child.parent = self
        self.children.append(child)
        
    def get_success_rate(self) -> float:
        """Returns share successful executions"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count


class CompositeNode(MAXQNode):
    """Composite task in MAXQ"""
    
    def __init__(self, 
                 node_id: str,
                 termination_condition: Callable[[MAXQState], bool],
                 value_function: Optional[MAXQValueFunction] = None):
        super().__init__(node_id, MAXQNodeType.COMPOSITE)
        self.termination_condition = termination_condition
        self.value_function = value_function
        self.subtask_rewards: Dict[str, float] = {}
        
    def is_terminal(self, state: MAXQState) -> bool:
        """Checks condition completion"""
        return self.termination_condition(state)
    
    def get_policy_action(self, state: MAXQState) -> str:
        """Selects subtask with highest Q-value"""
        if not self.children:
            return "no_action"
        
        if self.value_function is None:
            # Random selection, if no value function
            return np.random.choice([child.node_id for child in self.children])
        
        # Compute Q-values for all subtasks
        best_subtask = None
        best_value = float('-inf')
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.environment_state).unsqueeze(0)
            
            for child in self.children:
                task_tensor = torch.LongTensor([child.task_id])
                q_value = self.value_function(state_tensor, task_tensor).item()
                
                if q_value > best_value:
                    best_value = q_value
                    best_subtask = child.node_id
        
        return best_subtask or self.children[0].node_id


class MaxNode(MAXQNode):
    """MAX node for selection optimal subtasks"""
    
    def __init__(self, node_id: str, value_function: MAXQValueFunction):
        super().__init__(node_id, MAXQNodeType.MAX)
        self.value_function = value_function
        
    def is_terminal(self, state: MAXQState) -> bool:
        """MAX node not has own conditions completion"""
        return False
    
    def get_policy_action(self, state: MAXQState) -> str:
        """Selects subtask with maximum Q-value"""
        if not self.children:
            return "no_action"
        
        best_subtask = None
        best_value = float('-inf')
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.environment_state).unsqueeze(0)
            
            for child in self.children:
                if child.is_terminal(state):
                    continue  # Skip already completed tasks
                
                task_tensor = torch.LongTensor([child.task_id])
                q_value = self.value_function(state_tensor, task_tensor).item()
                
                if q_value > best_value:
                    best_value = q_value
                    best_subtask = child.node_id
        
        return best_subtask or self.children[0].node_id


class PrimitiveNode(MAXQNode):
    """Primitive action in MAXQ"""
    
    def __init__(self, node_id: str, action_id: int, action_function: Optional[Callable] = None):
        super().__init__(node_id, MAXQNodeType.PRIMITIVE)
        self.action_id = action_id
        self.action_function = action_function
        
    def is_terminal(self, state: MAXQState) -> bool:
        """Primitive actions always terminal after execution"""
        return True
    
    def get_policy_action(self, state: MAXQState) -> int:
        """Returns ID primitive actions"""
        return self.action_id
    
    async def execute(self, state: MAXQState) -> Tuple[MAXQState, float]:
        """Executes primitive action"""
        if self.action_function:
            try:
                new_env_state, reward = await self.action_function(
                    state.environment_state, self.action_id
                )
            except Exception as e:
                logger.error(f"Error execution actions {self.action_id}: {e}")
                new_env_state = state.environment_state
                reward = -0.01  # Penalty for error
        else:
            # Simple simulation
            new_env_state = state.environment_state.copy()
            if len(new_env_state) > 0:
                new_env_state[0] += (self.action_id - 1) * 0.001 + np.random.normal(0, 0.005)
            reward = np.random.normal(0.001, 0.01)
        
        new_state = state.copy()
        new_state.environment_state = new_env_state
        
        return new_state, reward


class MAXQHierarchy:
    """
    MAXQ hierarchy tasks
    Implements decomposition value function and hierarchical training
    """
    
    def __init__(self, root_task: str):
        self.root_task = root_task
        self.nodes: Dict[str, MAXQNode] = {}
        self.hierarchy_graph = nx.DiGraph()
        
        # Components training
        self.replay_buffer: deque = deque(maxlen=10000)
        self.learning_rates: Dict[str, float] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.cumulative_rewards: List[float] = []
        
    def add_node(self, node: MAXQNode) -> None:
        """Adds node in hierarchy"""
        self.nodes[node.node_id] = node
        self.hierarchy_graph.add_node(node.node_id, type=node.node_type.value)
        
        # Configure training for composite tasks
        if isinstance(node, (CompositeNode, MaxNode)) and node.value_function:
            self.learning_rates[node.node_id] = 0.001
            self.optimizers[node.node_id] = optim.Adam(
                node.value_function.parameters(), 
                lr=self.learning_rates[node.node_id]
            )
        
        logger.debug(f"Added MAXQ node: {node.node_id}")
    
    def add_hierarchy_edge(self, parent_id: str, child_id: str) -> None:
        """Adds connection parent-child in hierarchy"""
        if parent_id in self.nodes and child_id in self.nodes:
            parent = self.nodes[parent_id]
            child = self.nodes[child_id]
            parent.add_child(child)
            self.hierarchy_graph.add_edge(parent_id, child_id)
            logger.debug(f"Added connection: {parent_id} -> {child_id}")
    
    async def execute_task(self, 
                          task_id: str, 
                          initial_state: MAXQState, 
                          max_steps: int = 1000) -> Tuple[List[MAXQTransition], float]:
        """
        Executes task in hierarchy
        Returns trajectory and total reward
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} not found in hierarchy")
        
        task_node = self.nodes[task_id]
        current_state = initial_state.copy()
        trajectory: List[MAXQTransition] = []
        total_reward = 0.0
        steps = 0
        
        # Add task in stack
        current_state.task_stack.append(task_id)
        
        try:
            while steps < max_steps and not task_node.is_terminal(current_state):
                action = task_node.get_policy_action(current_state)
                
                if isinstance(task_node, PrimitiveNode):
                    # Execute primitive action
                    new_state, reward = await task_node.execute(current_state)
                    
                    transition = MAXQTransition(
                        from_state=current_state.copy(),
                        action=action,
                        to_state=new_state,
                        reward=reward,
                        terminal=True
                    )
                    trajectory.append(transition)
                    total_reward += reward
                    current_state = new_state
                    break
                    
                else:
                    # Execute subtask
                    if action in self.nodes:
                        subtask_trajectory, subtask_reward = await self.execute_task(
                            action, current_state, max_steps - steps
                        )
                        
                        # Update state and reward
                        if subtask_trajectory:
                            final_state = subtask_trajectory[-1].to_state
                            
                            transition = MAXQTransition(
                                from_state=current_state.copy(),
                                action=action,
                                to_state=final_state,
                                reward=subtask_reward,
                                terminal=self.nodes[action].is_terminal(final_state)
                            )
                            trajectory.append(transition)
                            trajectory.extend(subtask_trajectory)
                            
                            current_state = final_state
                            total_reward += subtask_reward
                            steps += len(subtask_trajectory)
                    else:
                        logger.warning(f"Subtask {action} not found")
                        break
                
                steps += 1
                await asyncio.sleep(0.001)  # Asynchrony
        
        except Exception as e:
            logger.error(f"Error execution tasks {task_id}: {e}")
        
        finally:
            # Remove task from stack
            if current_state.task_stack and current_state.task_stack[-1] == task_id:
                current_state.task_stack.pop()
        
        return trajectory, total_reward
    
    def compute_q_value(self, state: MAXQState, task_id: str, subtask_id: str) -> float:
        """
        Computes Q-value for execution subtasks in context tasks
        Q(s, task, subtask) = V(s', subtask) + C(s, s', task)
        """
        task_node = self.nodes.get(task_id)
        subtask_node = self.nodes.get(subtask_id)
        
        if not task_node or not subtask_node:
            return 0.0
        
        # If exists value function, use its
        if hasattr(task_node, 'value_function') and task_node.value_function:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state.environment_state).unsqueeze(0)
                task_tensor = torch.LongTensor([subtask_node.task_id])
                q_value = task_node.value_function(state_tensor, task_tensor).item()
                return q_value
        
        # Otherwise use heuristic estimation
        return np.random.normal(0.0, 0.1)
    
    def update_value_functions(self, trajectory: List[MAXQTransition], discount: float = 0.99) -> None:
        """Updates value function on basis trajectory"""
        if not trajectory:
            return
        
        # Compute returns for of each step
        returns = []
        cumulative_return = 0.0
        
        for transition in reversed(trajectory):
            cumulative_return = transition.reward + discount * cumulative_return
            returns.append(cumulative_return)
        
        returns.reverse()
        
        # Update value function
        for i, transition in enumerate(trajectory):
            if isinstance(transition.action, str) and transition.action in self.nodes:
                task_node = self.nodes[transition.action]
                
                if hasattr(task_node, 'value_function') and task_node.value_function:
                    optimizer = self.optimizers.get(transition.action)
                    if optimizer:
                        # Prepare data
                        state_tensor = torch.FloatTensor(
                            transition.from_state.environment_state
                        ).unsqueeze(0)
                        task_tensor = torch.LongTensor([task_node.task_id])
                        target = torch.FloatTensor([returns[i]])
                        
                        # Compute losses and update
                        predicted = task_node.value_function(state_tensor, task_tensor)
                        loss = F.mse_loss(predicted, target)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
    
    def store_trajectory(self, trajectory: List[MAXQTransition]) -> None:
        """Saves trajectory in replay buffer"""
        for transition in trajectory:
            self.replay_buffer.append(transition)
    
    def train_episode(self, initial_state: MAXQState, max_steps: int = 1000) -> float:
        """Trains on one episode"""
        self.episode_count += 1
        
        async def run_episode():
            trajectory, total_reward = await self.execute_task(
                self.root_task, initial_state, max_steps
            )
            
            # Save trajectory
            self.store_trajectory(trajectory)
            
            # Update value function
            self.update_value_functions(trajectory)
            
            # Update statistics
            self.total_steps += len(trajectory)
            self.cumulative_rewards.append(total_reward)
            
            return total_reward
        
        # Run asynchronously
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(run_episode())
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Returns statistics hierarchy"""
        node_stats = {}
        for node_id, node in self.nodes.items():
            node_stats[node_id] = {
                'type': node.node_type.value,
                'execution_count': node.execution_count,
                'success_rate': node.get_success_rate(),
                'total_reward': node.total_reward
            }
        
        return {
            'total_nodes': len(self.nodes),
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.cumulative_rewards) if self.cumulative_rewards else 0.0,
            'replay_buffer_size': len(self.replay_buffer),
            'node_statistics': node_stats
        }
    
    def visualize_hierarchy(self, filename: Optional[str] = None) -> None:
        """Creates visualization hierarchy"""
        plt.figure(figsize=(12, 8))
        
        # Use hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(self.hierarchy_graph, prog='dot')
        
        # Colors for different types nodes
        node_colors = []
        for node_id in self.hierarchy_graph.nodes():
            node_type = self.hierarchy_graph.nodes[node_id]['type']
            if node_type == 'composite':
                node_colors.append('lightblue')
            elif node_type == 'max':
                node_colors.append('lightgreen')
            elif node_type == 'primitive':
                node_colors.append('lightyellow')
            else:
                node_colors.append('lightgray')
        
        # Draw graph
        nx.draw(self.hierarchy_graph, pos,
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray')
        
        # Add signatures
        nx.draw_networkx_labels(self.hierarchy_graph, pos)
        
        plt.title("MAXQ Task Hierarchy")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_hierarchy(self, filename: str) -> None:
        """Saves hierarchy in file"""
        data = {
            'root_task': self.root_task,
            'hierarchy_graph': nx.node_link_data(self.hierarchy_graph),
            'statistics': self.get_hierarchy_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# Predefined MAXQ hierarchy for crypto trading

def create_trading_maxq_hierarchy(state_dim: int = 10) -> MAXQHierarchy:
    """Creates MAXQ hierarchy for crypto trading"""
    hierarchy = MAXQHierarchy("trading_root")
    
    # Value function
    main_value_fn = MAXQValueFunction(state_dim)
    entry_value_fn = MAXQValueFunction(state_dim)
    exit_value_fn = MAXQValueFunction(state_dim)
    
    # Conditions completion
    def trading_complete(state: MAXQState) -> bool:
        return len(state.environment_state) > 3 and abs(state.environment_state[3]) > 0.05
    
    def position_entered(state: MAXQState) -> bool:
        return state.variables.get('position_size', 0) > 0
    
    def position_exited(state: MAXQState) -> bool:
        return state.variables.get('position_size', 0) == 0
    
    # Nodes
    root = CompositeNode("trading_root", trading_complete, main_value_fn)
    entry_task = CompositeNode("entry_strategy", position_entered, entry_value_fn)
    exit_task = CompositeNode("exit_strategy", position_exited, exit_value_fn)
    
    # Primitive actions
    buy_action = PrimitiveNode("buy", 1)
    sell_action = PrimitiveNode("sell", 2)
    hold_action = PrimitiveNode("hold", 0)
    
    # Add nodes
    hierarchy.add_node(root)
    hierarchy.add_node(entry_task)
    hierarchy.add_node(exit_task)
    hierarchy.add_node(buy_action)
    hierarchy.add_node(sell_action)
    hierarchy.add_node(hold_action)
    
    # Build hierarchy
    hierarchy.add_hierarchy_edge("trading_root", "entry_strategy")
    hierarchy.add_hierarchy_edge("trading_root", "exit_strategy")
    hierarchy.add_hierarchy_edge("entry_strategy", "buy")
    hierarchy.add_hierarchy_edge("entry_strategy", "hold")
    hierarchy.add_hierarchy_edge("exit_strategy", "sell")
    hierarchy.add_hierarchy_edge("exit_strategy", "hold")
    
    return hierarchy


def create_arbitrage_maxq_hierarchy(state_dim: int = 10) -> MAXQHierarchy:
    """Creates MAXQ hierarchy for arbitrage"""
    hierarchy = MAXQHierarchy("arbitrage_root")
    
    # Value function
    main_value_fn = MAXQValueFunction(state_dim)
    scan_value_fn = MAXQValueFunction(state_dim)
    execute_value_fn = MAXQValueFunction(state_dim)
    
    # Conditions completion
    def arbitrage_complete(state: MAXQState) -> bool:
        return state.variables.get('arbitrage_profit', 0) > 0.01
    
    def opportunity_found(state: MAXQState) -> bool:
        return len(state.environment_state) > 4 and abs(state.environment_state[4]) > 0.005
    
    def arbitrage_executed(state: MAXQState) -> bool:
        return state.variables.get('arbitrage_active', False)
    
    # Nodes
    root = CompositeNode("arbitrage_root", arbitrage_complete, main_value_fn)
    scan_task = CompositeNode("scan_opportunities", opportunity_found, scan_value_fn)
    execute_task = CompositeNode("execute_arbitrage", arbitrage_executed, execute_value_fn)
    
    # Primitive actions
    scan_action = PrimitiveNode("scan", 0)
    buy_low_action = PrimitiveNode("buy_low", 1)
    sell_high_action = PrimitiveNode("sell_high", 2)
    
    # Add nodes
    hierarchy.add_node(root)
    hierarchy.add_node(scan_task)
    hierarchy.add_node(execute_task)
    hierarchy.add_node(scan_action)
    hierarchy.add_node(buy_low_action)
    hierarchy.add_node(sell_high_action)
    
    # Build hierarchy
    hierarchy.add_hierarchy_edge("arbitrage_root", "scan_opportunities")
    hierarchy.add_hierarchy_edge("arbitrage_root", "execute_arbitrage")
    hierarchy.add_hierarchy_edge("scan_opportunities", "scan")
    hierarchy.add_hierarchy_edge("execute_arbitrage", "buy_low")
    hierarchy.add_hierarchy_edge("execute_arbitrage", "sell_high")
    
    return hierarchy