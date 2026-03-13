"""
Hierarchical Abstract Machines (HAM) Implementation
Implementation hierarchical abstract machines for structured making decisions.

enterprise Pattern:
- Hierarchical state machines for complex trading strategies
- Production-ready finite state automata with asynchronous execution
- Formal verification and debugging capabilities for reliability
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import logging
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
import json
import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class HAMNodeType(Enum):
    """Types nodes in HAM"""
    CHOICE = "choice"           # Node selection actions
    CALL = "call"              # Call submachines
    ACTION = "action"          # Primitive action
    STOP = "stop"              # Stopping execution


class HAMTransitionType(Enum):
    """Types transitions in HAM"""
    DETERMINISTIC = "deterministic"    # Deterministic transition
    STOCHASTIC = "stochastic"         # Stochastic transition
    CONDITIONAL = "conditional"        # Conditional transition


@dataclass
class HAMTransition:
    """Transition between nodes HAM"""
    from_node: str
    to_node: str
    condition: Optional[Callable[[np.ndarray], bool]]
    probability: float = 1.0
    transition_type: HAMTransitionType = HAMTransitionType.DETERMINISTIC
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HAMExecutionContext:
    """Context execution HAM machines"""
    machine_id: str
    current_node: str
    state: np.ndarray
    step_count: int
    total_reward: float
    execution_stack: List[str]
    variables: Dict[str, Any]
    start_time: float
    
    def copy(self) -> 'HAMExecutionContext':
        """Creates copy context"""
        return HAMExecutionContext(
            machine_id=self.machine_id,
            current_node=self.current_node,
            state=self.state.copy(),
            step_count=self.step_count,
            total_reward=self.total_reward,
            execution_stack=self.execution_stack.copy(),
            variables=self.variables.copy(),
            start_time=self.start_time
        )


class HAMNode(ABC):
    """Abstract base class for nodes HAM"""
    
    def __init__(self, node_id: str, node_type: HAMNodeType):
        self.node_id = node_id
        self.node_type = node_type
        self.incoming_transitions: List[HAMTransition] = []
        self.outgoing_transitions: List[HAMTransition] = []
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    async def execute(self, context: HAMExecutionContext) -> Tuple[Any, str]:
        """
        Executes node and returns (result, next_node)
        """
        pass
    
    def add_transition(self, transition: HAMTransition) -> None:
        """Adds transition from of this node"""
        self.outgoing_transitions.append(transition)
    
    def can_transition_to(self, target_node: str, state: np.ndarray) -> bool:
        """Checks capability transition to target node"""
        for transition in self.outgoing_transitions:
            if (transition.to_node == target_node and
                (transition.condition is None or transition.condition(state))):
                return True
        return False


class ChoiceNode(HAMNode):
    """Node selection actions with using policy"""
    
    def __init__(self, node_id: str, policy: nn.Module, action_space: int):
        super().__init__(node_id, HAMNodeType.CHOICE)
        self.policy = policy
        self.action_space = action_space
        
    async def execute(self, context: HAMExecutionContext) -> Tuple[int, str]:
        """Selects action with help policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(context.state).unsqueeze(0)
            action_probs = self.policy(state_tensor)
            action = torch.multinomial(action_probs, 1).item()
        
        # Select next node on basis actions and transitions
        valid_transitions = [t for t in self.outgoing_transitions 
                           if t.condition is None or t.condition(context.state)]
        
        if valid_transitions:
            # If exists conditional transitions, use their
            next_node = valid_transitions[action % len(valid_transitions)].to_node
        else:
            # Otherwise remain in that same node (loop)
            next_node = self.node_id
            
        logger.debug(f"ChoiceNode {self.node_id}: selected action {action}, transition to {next_node}")
        return action, next_node


class CallNode(HAMNode):
    """Node call submachines"""
    
    def __init__(self, node_id: str, called_machine_id: str):
        super().__init__(node_id, HAMNodeType.CALL)
        self.called_machine_id = called_machine_id
        
    async def execute(self, context: HAMExecutionContext) -> Tuple[Any, str]:
        """Calls submachine"""
        # Add call in stack
        context.execution_stack.append(self.called_machine_id)
        
        # Return ID called machines as next node
        logger.debug(f"CallNode {self.node_id}: call machines {self.called_machine_id}")
        return self.called_machine_id, "start"


class ActionNode(HAMNode):
    """Node primitive actions"""
    
    def __init__(self, node_id: str, action_id: int, action_function: Optional[Callable] = None):
        super().__init__(node_id, HAMNodeType.ACTION)
        self.action_id = action_id
        self.action_function = action_function
        
    async def execute(self, context: HAMExecutionContext) -> Tuple[int, str]:
        """Executes primitive action"""
        if self.action_function:
            try:
                result = await self.action_function(context.state, self.action_id)
            except Exception as e:
                logger.error(f"Error execution actions {self.action_id}: {e}")
                result = self.action_id
        else:
            result = self.action_id
        
        # Select next node
        if self.outgoing_transitions:
            next_node = self.outgoing_transitions[0].to_node
        else:
            next_node = "stop"
            
        logger.debug(f"ActionNode {self.node_id}: completed action {self.action_id}")
        return result, next_node


class StopNode(HAMNode):
    """Node stopping execution"""
    
    def __init__(self, node_id: str = "stop"):
        super().__init__(node_id, HAMNodeType.STOP)
        
    async def execute(self, context: HAMExecutionContext) -> Tuple[None, str]:
        """Stops execution"""
        logger.debug(f"StopNode {self.node_id}: stopping execution")
        return None, "stop"


class HAMMachine:
    """
    Hierarchical abstract machine
    Implements finite automaton with capability call submachines
    """
    
    def __init__(self, machine_id: str, start_node: str = "start"):
        self.machine_id = machine_id
        self.start_node = start_node
        self.nodes: Dict[str, HAMNode] = {}
        self.transitions: List[HAMTransition] = []
        self.submachines: Dict[str, 'HAMMachine'] = {}
        
        # Statistics execution
        self.execution_count = 0
        self.success_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        
        # Graph for visualization
        self.graph = nx.DiGraph()
        
    def add_node(self, node: HAMNode) -> None:
        """Adds node in machine"""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, type=node.node_type.value)
        logger.debug(f"Added node {node.node_id} type {node.node_type.value}")
        
    def add_transition(self, transition: HAMTransition) -> None:
        """Adds transition between nodes"""
        self.transitions.append(transition)
        
        # Add transition to outgoing node
        if transition.from_node in self.nodes:
            self.nodes[transition.from_node].add_transition(transition)
            
        # Add in graph
        self.graph.add_edge(
            transition.from_node, 
            transition.to_node,
            condition=str(transition.condition) if transition.condition else "always",
            probability=transition.probability
        )
        
        logger.debug(f"Added transition {transition.from_node} -> {transition.to_node}")
        
    def add_submachine(self, submachine: 'HAMMachine') -> None:
        """Adds submachine"""
        self.submachines[submachine.machine_id] = submachine
        logger.debug(f"Added submachine {submachine.machine_id}")
        
    async def execute(self, initial_state: np.ndarray, max_steps: int = 1000) -> HAMExecutionContext:
        """
        Executes machine until completion or achievement limit steps
        """
        context = HAMExecutionContext(
            machine_id=self.machine_id,
            current_node=self.start_node,
            state=initial_state.copy(),
            step_count=0,
            total_reward=0.0,
            execution_stack=[],
            variables={},
            start_time=asyncio.get_event_loop().time()
        )
        
        self.execution_count += 1
        
        try:
            while context.step_count < max_steps and context.current_node != "stop":
                current_node = self.nodes.get(context.current_node)
                
                if current_node is None:
                    # Possibly, this call submachines
                    if context.current_node in self.submachines:
                        submachine = self.submachines[context.current_node]
                        sub_context = await submachine.execute(context.state, max_steps - context.step_count)
                        
                        # Update context results submachines
                        context.state = sub_context.state
                        context.step_count += sub_context.step_count
                        context.total_reward += sub_context.total_reward
                        
                        # Return to calling machine
                        if context.execution_stack:
                            calling_machine = context.execution_stack.pop()
                            context.current_node = calling_machine
                        else:
                            context.current_node = "stop"
                    else:
                        logger.error(f"Unknown node: {context.current_node}")
                        break
                else:
                    # Execute current node
                    result, next_node = await current_node.execute(context)
                    context.current_node = next_node
                    context.step_count += 1
                    
                    # Update state (in reality from environment)
                    if isinstance(result, int):  # If this action
                        context.state = self._simulate_state_transition(context.state, result)
                        context.total_reward += np.random.normal(0.001, 0.01)
                
                # Small delay for asynchrony
                await asyncio.sleep(0.001)
                
            # Update statistics
            self.total_steps += context.step_count
            self.total_reward += context.total_reward
            
            if context.current_node == "stop":
                self.success_count += 1
                
            logger.info(f"HAM machine {self.machine_id} completed for {context.step_count} steps")
            return context
            
        except Exception as e:
            logger.error(f"Error execution HAM machines {self.machine_id}: {e}")
            return context
    
    def _simulate_state_transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """Simulates transition state (stub)"""
        new_state = state.copy()
        # Simple simulation
        if len(new_state) > 0:
            new_state[0] += (action - 1) * 0.001 + np.random.normal(0, 0.005)
        return new_state
    
    def validate(self) -> List[str]:
        """Validates structure machines"""
        errors = []
        
        # Check presence starting node
        if self.start_node not in self.nodes:
            errors.append(f"Starting node '{self.start_node}' not found")
            
        # Check reachability all nodes
        reachable = set()
        self._dfs_reachability(self.start_node, reachable)
        
        for node_id in self.nodes:
            if node_id not in reachable:
                errors.append(f"Node '{node_id}' unreachable from starting node")
                
        # Check presence nodes in transitions
        for transition in self.transitions:
            if transition.from_node not in self.nodes:
                errors.append(f"Node source '{transition.from_node}' not exists")
            if transition.to_node not in self.nodes and transition.to_node not in self.submachines:
                errors.append(f"Target node '{transition.to_node}' not exists")
                
        return errors
    
    def _dfs_reachability(self, node: str, visited: Set[str]) -> None:
        """DFS for validation reachability nodes"""
        if node in visited or node not in self.nodes:
            return
            
        visited.add(node)
        
        for transition in self.transitions:
            if transition.from_node == node:
                self._dfs_reachability(transition.to_node, visited)
    
    def visualize(self, filename: Optional[str] = None) -> None:
        """Creates visualization machines"""
        plt.figure(figsize=(12, 8))
        
        # Configure positions nodes
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Colors for different types nodes
        node_colors = []
        for node_id in self.graph.nodes():
            node_type = self.graph.nodes[node_id]['type']
            if node_type == 'choice':
                node_colors.append('lightblue')
            elif node_type == 'call':
                node_colors.append('lightgreen')
            elif node_type == 'action':
                node_colors.append('lightyellow')
            elif node_type == 'stop':
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightgray')
        
        # Draw graph
        nx.draw(self.graph, pos, 
                node_color=node_colors,
                node_size=1500,
                font_size=10,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray')
        
        # Add signatures nodes
        nx.draw_networkx_labels(self.graph, pos)
        
        # Add signatures edges
        edge_labels = {}
        for edge in self.graph.edges(data=True):
            if edge[2]['probability'] < 1.0:
                edge_labels[(edge[0], edge[1])] = f"{edge[2]['probability']:.2f}"
        
        if edge_labels:
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=8)
        
        plt.title(f"HAM Machine: {self.machine_id}")
        plt.axis('off')
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes machine in dictionary"""
        return {
            'machine_id': self.machine_id,
            'start_node': self.start_node,
            'nodes': {node_id: {
                'type': node.node_type.value,
                'metadata': node.metadata
            } for node_id, node in self.nodes.items()},
            'transitions': [{
                'from_node': t.from_node,
                'to_node': t.to_node,
                'probability': t.probability,
                'transition_type': t.transition_type.value,
                'metadata': t.metadata
            } for t in self.transitions],
            'statistics': {
                'execution_count': self.execution_count,
                'success_count': self.success_count,
                'success_rate': self.success_count / max(self.execution_count, 1),
                'avg_steps': self.total_steps / max(self.execution_count, 1),
                'avg_reward': self.total_reward / max(self.execution_count, 1)
            }
        }
    
    def save(self, filename: str) -> None:
        """Saves machine in file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class HAMFramework:
    """
    Framework for management hierarchical abstract machines
    Implements design pattern for enterprise system
    """
    
    def __init__(self):
        self.machines: Dict[str, HAMMachine] = {}
        self.execution_history: List[HAMExecutionContext] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def register_machine(self, machine: HAMMachine) -> None:
        """Registers machine in framework"""
        errors = machine.validate()
        if errors:
            raise ValueError(f"Errors validation machines {machine.machine_id}: {errors}")
            
        self.machines[machine.machine_id] = machine
        logger.info(f"Registered HAM machine: {machine.machine_id}")
    
    async def execute_machine(self, machine_id: str, initial_state: np.ndarray) -> HAMExecutionContext:
        """Executes machine asynchronously"""
        if machine_id not in self.machines:
            raise ValueError(f"Machine {machine_id} not registered")
            
        machine = self.machines[machine_id]
        context = await machine.execute(initial_state)
        self.execution_history.append(context)
        
        return context
    
    def get_machine_statistics(self) -> Dict[str, Any]:
        """Returns statistics by all machines"""
        stats = {
            'total_machines': len(self.machines),
            'total_executions': len(self.execution_history),
            'machine_stats': {}
        }
        
        for machine_id, machine in self.machines.items():
            stats['machine_stats'][machine_id] = machine.to_dict()['statistics']
            
        return stats


# Predefined HAM machines for crypto trading

def create_trend_following_ham(state_dim: int = 10, action_dim: int = 3) -> HAMMachine:
    """Creates HAM machine for strategies following trend"""
    machine = HAMMachine("trend_following_ham", "check_trend")
    
    # Create policy
    policy = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, action_dim),
        nn.Softmax(dim=-1)
    )
    
    # Nodes
    check_trend = ChoiceNode("check_trend", policy, action_dim)
    enter_position = ActionNode("enter_position", 1)  # Action "buy"
    hold_position = ActionNode("hold_position", 0)    # Action "hold"
    exit_position = ActionNode("exit_position", 2)    # Action "sell"
    stop = StopNode("stop")
    
    # Add nodes
    machine.add_node(check_trend)
    machine.add_node(enter_position)
    machine.add_node(hold_position)
    machine.add_node(exit_position)
    machine.add_node(stop)
    
    # Conditions transitions
    def uptrend_condition(state: np.ndarray) -> bool:
        return len(state) > 0 and state[0] > 0.01  # Growth price > 1%
    
    def downtrend_condition(state: np.ndarray) -> bool:
        return len(state) > 0 and state[0] < -0.01  # Drop price > 1%
    
    def profit_condition(state: np.ndarray) -> bool:
        return len(state) > 3 and state[3] > 0.02  # Profit > 2%
    
    # Transitions
    machine.add_transition(HAMTransition("check_trend", "enter_position", uptrend_condition))
    machine.add_transition(HAMTransition("check_trend", "exit_position", downtrend_condition))
    machine.add_transition(HAMTransition("check_trend", "hold_position", None))  # By default
    
    machine.add_transition(HAMTransition("enter_position", "hold_position", None))
    machine.add_transition(HAMTransition("hold_position", "exit_position", profit_condition))
    machine.add_transition(HAMTransition("hold_position", "check_trend", None))
    machine.add_transition(HAMTransition("exit_position", "stop", None))
    
    return machine


def create_arbitrage_ham(state_dim: int = 10, action_dim: int = 3) -> HAMMachine:
    """Creates HAM machine for arbitrage strategies"""
    machine = HAMMachine("arbitrage_ham", "scan_opportunities")
    
    policy = nn.Sequential(
        nn.Linear(state_dim, 32),
        nn.ReLU(),
        nn.Linear(32, action_dim),
        nn.Softmax(dim=-1)
    )
    
    # Nodes
    scan_opportunities = ChoiceNode("scan_opportunities", policy, action_dim)
    execute_arbitrage = ActionNode("execute_arbitrage", 1)
    monitor_position = ActionNode("monitor_position", 0)
    close_arbitrage = ActionNode("close_arbitrage", 2)
    stop = StopNode("stop")
    
    machine.add_node(scan_opportunities)
    machine.add_node(execute_arbitrage)
    machine.add_node(monitor_position)
    machine.add_node(close_arbitrage)
    machine.add_node(stop)
    
    # Conditions
    def arbitrage_opportunity(state: np.ndarray) -> bool:
        return len(state) > 4 and abs(state[4]) > 0.005  # Spread > 0.5%
    
    def position_profitable(state: np.ndarray) -> bool:
        return len(state) > 3 and state[3] > 0.001  # Profit > 0.1%
    
    # Transitions
    machine.add_transition(HAMTransition("scan_opportunities", "execute_arbitrage", arbitrage_opportunity))
    machine.add_transition(HAMTransition("scan_opportunities", "scan_opportunities", None))
    machine.add_transition(HAMTransition("execute_arbitrage", "monitor_position", None))
    machine.add_transition(HAMTransition("monitor_position", "close_arbitrage", position_profitable))
    machine.add_transition(HAMTransition("monitor_position", "monitor_position", None))
    machine.add_transition(HAMTransition("close_arbitrage", "stop", None))
    
    return machine


def create_hierarchical_trading_ham() -> HAMMachine:
    """Creates hierarchical machine with submachines"""
    main_machine = HAMMachine("hierarchical_trading", "strategy_selection")
    
    # Main machine selection strategies
    policy = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
        nn.Softmax(dim=-1)
    )
    
    strategy_selection = ChoiceNode("strategy_selection", policy, 3)
    trend_call = CallNode("trend_call", "trend_following_ham")
    arbitrage_call = CallNode("arbitrage_call", "arbitrage_ham")
    stop = StopNode("stop")
    
    main_machine.add_node(strategy_selection)
    main_machine.add_node(trend_call)
    main_machine.add_node(arbitrage_call)
    main_machine.add_node(stop)
    
    # Transitions
    def high_volatility(state: np.ndarray) -> bool:
        return len(state) > 2 and state[2] > 0.05
    
    def low_volatility(state: np.ndarray) -> bool:
        return len(state) > 2 and state[2] < 0.02
    
    main_machine.add_transition(HAMTransition("strategy_selection", "trend_call", high_volatility))
    main_machine.add_transition(HAMTransition("strategy_selection", "arbitrage_call", low_volatility))
    main_machine.add_transition(HAMTransition("strategy_selection", "stop", None))
    
    # Add submachines
    main_machine.add_submachine(create_trend_following_ham())
    main_machine.add_submachine(create_arbitrage_ham())
    
    return main_machine