"""
Hierarchy Utilities for Hierarchical RL System
Helper utilities for work with hierarchical structures and components.

enterprise Pattern:
- Utility functions for hierarchical system management
- Production-ready helper classes with performance optimization
- Scalable data structures for complex hierarchy operations
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import logging
import json
import pickle
from collections import defaultdict, deque
import time
import hashlib
import uuid

logger = logging.getLogger(__name__)


class HierarchyType(Enum):
    """Types hierarchical structures"""
    OPTIONS = "options"
    HAM = "ham"
    MAXQ = "maxq"
    HAC = "hac"
    SKILLS = "skills"


@dataclass
class HierarchyMetrics:
    """Metrics hierarchical system"""
    depth: int
    breadth: int
    avg_branching_factor: float
    complexity_score: float
    efficiency_ratio: float
    coverage_percentage: float
    reusability_index: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'depth': self.depth,
            'breadth': self.breadth,
            'avg_branching_factor': self.avg_branching_factor,
            'complexity_score': self.complexity_score,
            'efficiency_ratio': self.efficiency_ratio,
            'coverage_percentage': self.coverage_percentage,
            'reusability_index': self.reusability_index
        }


class HierarchyAnalyzer:
    """Analyzer hierarchical structures"""
    
    def __init__(self):
        self.analysis_cache: Dict[str, Any] = {}
        
    def analyze_hierarchy(self, hierarchy_graph: nx.Graph) -> HierarchyMetrics:
        """Analyzes hierarchical structure"""
        if len(hierarchy_graph.nodes()) == 0:
            return HierarchyMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Compute depth hierarchy
        depth = self._compute_depth(hierarchy_graph)
        
        # Compute width hierarchy
        breadth = len(hierarchy_graph.nodes())
        
        # Compute average coefficient branching
        avg_branching = self._compute_avg_branching_factor(hierarchy_graph)
        
        # Compute complexity
        complexity = self._compute_complexity_score(hierarchy_graph)
        
        # Compute efficiency
        efficiency = self._compute_efficiency_ratio(hierarchy_graph)
        
        # Compute coverage
        coverage = self._compute_coverage_percentage(hierarchy_graph)
        
        # Compute reusability
        reusability = self._compute_reusability_index(hierarchy_graph)
        
        return HierarchyMetrics(
            depth=depth,
            breadth=breadth,
            avg_branching_factor=avg_branching,
            complexity_score=complexity,
            efficiency_ratio=efficiency,
            coverage_percentage=coverage,
            reusability_index=reusability
        )
    
    def _compute_depth(self, graph: nx.Graph) -> int:
        """Computes depth hierarchy"""
        if graph.is_directed():
            try:
                # For directed graph search most long path
                return nx.dag_longest_path_length(graph)
            except:
                # If graph not is DAG
                return len(nx.shortest_path(graph, list(graph.nodes())[0], 
                                          list(graph.nodes())[-1])) - 1 if len(graph.nodes()) > 1 else 0
        else:
            # For undirected graph search diameter
            if nx.is_connected(graph):
                return nx.diameter(graph)
            else:
                return max([nx.diameter(graph.subgraph(c)) 
                           for c in nx.connected_components(graph)] or [0])
    
    def _compute_avg_branching_factor(self, graph: nx.Graph) -> float:
        """Computes average coefficient branching"""
        if len(graph.nodes()) == 0:
            return 0.0
        
        degrees = [graph.degree(node) for node in graph.nodes()]
        return np.mean(degrees)
    
    def _compute_complexity_score(self, graph: nx.Graph) -> float:
        """Computes score complexity hierarchy"""
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        
        if num_nodes == 0:
            return 0.0
        
        # Normalized density graph
        max_edges = num_nodes * (num_nodes - 1) / 2 if not graph.is_directed() else num_nodes * (num_nodes - 1)
        density = num_edges / max(max_edges, 1)
        
        # Combine size and density
        complexity = (np.log(num_nodes + 1) * density) / 10.0
        return min(complexity, 1.0)
    
    def _compute_efficiency_ratio(self, graph: nx.Graph) -> float:
        """Computes coefficient efficiency"""
        if len(graph.nodes()) <= 1:
            return 1.0
        
        # Ratio real connections to minimally necessary
        min_edges = len(graph.nodes()) - 1  # Minimum for connectivity
        actual_edges = len(graph.edges())
        
        if actual_edges == 0:
            return 0.0
        
        efficiency = min_edges / actual_edges
        return min(efficiency, 1.0)
    
    def _compute_coverage_percentage(self, graph: nx.Graph) -> float:
        """Computes percent coverage states"""
        # Simple heuristic: percent connected nodes
        if len(graph.nodes()) == 0:
            return 0.0
        
        if graph.is_directed():
            # For directed graph check weak connectivity
            if nx.is_weakly_connected(graph):
                return 100.0
            else:
                largest_component = max(nx.weakly_connected_components(graph), key=len)
                return (len(largest_component) / len(graph.nodes())) * 100.0
        else:
            # For undirected graph
            if nx.is_connected(graph):
                return 100.0
            else:
                largest_component = max(nx.connected_components(graph), key=len)
                return (len(largest_component) / len(graph.nodes())) * 100.0
    
    def _compute_reusability_index(self, graph: nx.Graph) -> float:
        """Computes index reusability"""
        if len(graph.nodes()) == 0:
            return 0.0
        
        # Nodes with high degree are considered more reusable
        degrees = [graph.degree(node) for node in graph.nodes()]
        max_degree = max(degrees) if degrees else 0
        
        if max_degree == 0:
            return 0.0
        
        # Normalized index on basis distribution degrees
        normalized_degrees = [d / max_degree for d in degrees]
        reusability = np.mean(normalized_degrees)
        
        return reusability


class StateSpaceMapper:
    """Mapper for work with spaces states"""
    
    def __init__(self, state_dim: int, discretization_levels: int = 50):
        self.state_dim = state_dim
        self.discretization_levels = discretization_levels
        self.state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.state_history: deque = deque(maxlen=10000)
        
    def update_bounds(self, states: List[np.ndarray]) -> None:
        """Updates boundaries space states"""
        if not states:
            return
        
        states_array = np.array(states)
        
        if self.state_bounds is None:
            self.state_bounds = (states_array.min(axis=0), states_array.max(axis=0))
        else:
            min_bounds, max_bounds = self.state_bounds
            new_min = np.minimum(min_bounds, states_array.min(axis=0))
            new_max = np.maximum(max_bounds, states_array.max(axis=0))
            self.state_bounds = (new_min, new_max)
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretizes state"""
        if self.state_bounds is None:
            # Use simple discretization
            return tuple((state * self.discretization_levels).astype(int))
        
        min_bounds, max_bounds = self.state_bounds
        
        # Normalize and discretize
        normalized = np.zeros_like(state)
        for i in range(len(state)):
            if max_bounds[i] > min_bounds[i]:
                normalized[i] = (state[i] - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
            else:
                normalized[i] = 0.5
        
        discretized = (normalized * (self.discretization_levels - 1)).astype(int)
        discretized = np.clip(discretized, 0, self.discretization_levels - 1)
        
        return tuple(discretized)
    
    def continuous_from_discrete(self, discrete_state: Tuple[int, ...]) -> np.ndarray:
        """Restores continuous state from discrete"""
        if self.state_bounds is None:
            return np.array(discrete_state, dtype=float) / self.discretization_levels
        
        min_bounds, max_bounds = self.state_bounds
        
        # Denormalize
        normalized = np.array(discrete_state) / (self.discretization_levels - 1)
        continuous = min_bounds + normalized * (max_bounds - min_bounds)
        
        return continuous
    
    def compute_state_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Computes similarity states"""
        if len(state1) != len(state2):
            return 0.0
        
        # Normalized Euclidean distance
        distance = np.linalg.norm(state1 - state2)
        
        # Convert in similarity (0-1)
        max_distance = np.sqrt(len(state1))  # Maximum distance for normalized states
        similarity = 1.0 - min(distance / max_distance, 1.0)
        
        return similarity
    
    def get_state_density(self, state: np.ndarray, radius: float = 0.1) -> float:
        """Computes density states in neighborhood"""
        if not self.state_history:
            return 0.0
        
        # Count number states in radius
        count = 0
        for hist_state in self.state_history:
            if np.linalg.norm(state - hist_state) <= radius:
                count += 1
        
        # Normalize on total number states
        density = count / len(self.state_history)
        return density
    
    def add_state_to_history(self, state: np.ndarray) -> None:
        """Adds state in history"""
        self.state_history.append(state.copy())


class HierarchyBuilder:
    """Constructor hierarchical structures"""
    
    def __init__(self):
        self.built_hierarchies: Dict[str, Any] = {}
        
    def build_from_trajectories(self, 
                              trajectories: List[List[np.ndarray]],
                              hierarchy_type: HierarchyType,
                              **kwargs) -> Any:
        """Builds hierarchy from trajectories"""
        hierarchy_id = self._generate_hierarchy_id(hierarchy_type, kwargs)
        
        if hierarchy_id in self.built_hierarchies:
            return self.built_hierarchies[hierarchy_id]
        
        if hierarchy_type == HierarchyType.OPTIONS:
            hierarchy = self._build_options_hierarchy(trajectories, **kwargs)
        elif hierarchy_type == HierarchyType.HAM:
            hierarchy = self._build_ham_hierarchy(trajectories, **kwargs)
        elif hierarchy_type == HierarchyType.MAXQ:
            hierarchy = self._build_maxq_hierarchy(trajectories, **kwargs)
        elif hierarchy_type == HierarchyType.SKILLS:
            hierarchy = self._build_skills_hierarchy(trajectories, **kwargs)
        else:
            raise ValueError(f"Unsupported type hierarchy: {hierarchy_type}")
        
        self.built_hierarchies[hierarchy_id] = hierarchy
        return hierarchy
    
    def _generate_hierarchy_id(self, hierarchy_type: HierarchyType, kwargs: Dict[str, Any]) -> str:
        """Generates unique ID for hierarchy"""
        content = f"{hierarchy_type.value}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _build_options_hierarchy(self, trajectories: List[List[np.ndarray]], **kwargs) -> Any:
        """Builds hierarchy options"""
        from ..frameworks.options import OptionsFramework, create_trend_following_option
        
        framework = OptionsFramework()
        
        # Create standard options
        for i in range(kwargs.get('num_options', 5)):
            option_id = f"discovered_option_{i}"
            # Here possible add logic for creation options on basis trajectories
            option = create_trend_following_option()
            option.option_id = option_id
            framework.register_option(option)
        
        return framework
    
    def _build_ham_hierarchy(self, trajectories: List[List[np.ndarray]], **kwargs) -> Any:
        """Builds HAM hierarchy"""
        from ..frameworks.ham import HAMFramework, create_trend_following_ham
        
        framework = HAMFramework()
        
        # Create machines on basis analysis trajectories
        machine = create_trend_following_ham()
        framework.register_machine(machine)
        
        return framework
    
    def _build_maxq_hierarchy(self, trajectories: List[List[np.ndarray]], **kwargs) -> Any:
        """Builds MAXQ hierarchy"""
        from ..frameworks.maxq import create_trading_maxq_hierarchy
        
        state_dim = len(trajectories[0][0]) if trajectories and trajectories[0] else 10
        hierarchy = create_trading_maxq_hierarchy(state_dim)
        
        return hierarchy
    
    def _build_skills_hierarchy(self, trajectories: List[List[np.ndarray]], **kwargs) -> Any:
        """Builds hierarchy skills"""
        from ..policies.skill_policy import create_skill_library
        
        state_dim = len(trajectories[0][0]) if trajectories and trajectories[0] else 15
        skills = create_skill_library(state_dim)
        
        return skills


class PerformanceProfiler:
    """Profiler performance for hierarchical systems"""
    
    def __init__(self):
        self.profiles: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'memory_usage': []
        })
        
    def profile_function(self, func_name: str):
        """Decorator for profiling functions"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Update statistics
                    profile = self.profiles[func_name]
                    profile['call_count'] += 1
                    profile['total_time'] += execution_time
                    profile['min_time'] = min(profile['min_time'], execution_time)
                    profile['max_time'] = max(profile['max_time'], execution_time)
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    profile = self.profiles[func_name]
                    profile['call_count'] += 1
                    profile['total_time'] += execution_time
                    raise e
                    
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Returns report about performance"""
        report = {}
        
        for func_name, profile in self.profiles.items():
            if profile['call_count'] > 0:
                avg_time = profile['total_time'] / profile['call_count']
                
                report[func_name] = {
                    'call_count': profile['call_count'],
                    'total_time': profile['total_time'],
                    'avg_time': avg_time,
                    'min_time': profile['min_time'],
                    'max_time': profile['max_time'],
                    'calls_per_second': profile['call_count'] / profile['total_time'] if profile['total_time'] > 0 else 0
                }
        
        return report
    
    def reset_profiles(self) -> None:
        """Resets profiles"""
        self.profiles.clear()


class HierarchyValidator:
    """Validator hierarchical structures"""
    
    def __init__(self):
        self.validation_rules = {
            'max_depth': 10,
            'max_breadth': 1000,
            'min_coverage': 0.5,
            'max_complexity': 0.9
        }
    
    def validate_hierarchy(self, hierarchy: Any, hierarchy_type: HierarchyType) -> List[str]:
        """Validates hierarchical structure"""
        errors = []
        
        try:
            # General validation
            if hierarchy is None:
                errors.append("Hierarchy not can be None")
                return errors
            
            # Specific validation by type
            if hierarchy_type == HierarchyType.OPTIONS:
                errors.extend(self._validate_options_hierarchy(hierarchy))
            elif hierarchy_type == HierarchyType.HAM:
                errors.extend(self._validate_ham_hierarchy(hierarchy))
            elif hierarchy_type == HierarchyType.MAXQ:
                errors.extend(self._validate_maxq_hierarchy(hierarchy))
            
        except Exception as e:
            errors.append(f"Error validation: {str(e)}")
        
        return errors
    
    def _validate_options_hierarchy(self, hierarchy) -> List[str]:
        """Validates hierarchy options"""
        errors = []
        
        if not hasattr(hierarchy, 'options'):
            errors.append("Options framework must have attribute 'options'")
            return errors
        
        if len(hierarchy.options) == 0:
            errors.append("Options framework not contains options")
        
        # Check each option
        for option_id, option in hierarchy.options.items():
            if not hasattr(option, 'initiation_set'):
                errors.append(f"Option {option_id} not has initiation_set")
            
            if not hasattr(option, 'policy'):
                errors.append(f"Option {option_id} not has policy")
            
            if not hasattr(option, 'termination_condition'):
                errors.append(f"Option {option_id} not has termination_condition")
        
        return errors
    
    def _validate_ham_hierarchy(self, hierarchy) -> List[str]:
        """Validates HAM hierarchy"""
        errors = []
        
        if not hasattr(hierarchy, 'machines'):
            errors.append("HAM framework must have attribute 'machines'")
            return errors
        
        # Check each machine
        for machine_id, machine in hierarchy.machines.items():
            machine_errors = machine.validate() if hasattr(machine, 'validate') else []
            errors.extend([f"Machine {machine_id}: {error}" for error in machine_errors])
        
        return errors
    
    def _validate_maxq_hierarchy(self, hierarchy) -> List[str]:
        """Validates MAXQ hierarchy"""
        errors = []
        
        if not hasattr(hierarchy, 'nodes'):
            errors.append("MAXQ hierarchy must have attribute 'nodes'")
            return errors
        
        if len(hierarchy.nodes) == 0:
            errors.append("MAXQ hierarchy not contains nodes")
        
        return errors


class HierarchySerializer:
    """Serializer for hierarchical structures"""
    
    @staticmethod
    def serialize_hierarchy(hierarchy: Any, hierarchy_type: HierarchyType) -> Dict[str, Any]:
        """Serializes hierarchy in dictionary"""
        try:
            if hierarchy_type == HierarchyType.OPTIONS:
                return HierarchySerializer._serialize_options(hierarchy)
            elif hierarchy_type == HierarchyType.HAM:
                return HierarchySerializer._serialize_ham(hierarchy)
            elif hierarchy_type == HierarchyType.MAXQ:
                return HierarchySerializer._serialize_maxq(hierarchy)
            else:
                return {'type': hierarchy_type.value, 'data': str(hierarchy)}
                
        except Exception as e:
            logger.error(f"Error serialization hierarchy: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _serialize_options(hierarchy) -> Dict[str, Any]:
        """Serializes Options framework"""
        return {
            'type': 'options',
            'options_count': len(hierarchy.options),
            'option_ids': list(hierarchy.options.keys()),
            'statistics': hierarchy.get_statistics() if hasattr(hierarchy, 'get_statistics') else {}
        }
    
    @staticmethod
    def _serialize_ham(hierarchy) -> Dict[str, Any]:
        """Serializes HAM framework"""
        return {
            'type': 'ham',
            'machines_count': len(hierarchy.machines),
            'machine_ids': list(hierarchy.machines.keys()),
            'statistics': hierarchy.get_machine_statistics() if hasattr(hierarchy, 'get_machine_statistics') else {}
        }
    
    @staticmethod
    def _serialize_maxq(hierarchy) -> Dict[str, Any]:
        """Serializes MAXQ hierarchy"""
        return {
            'type': 'maxq',
            'nodes_count': len(hierarchy.nodes),
            'root_task': hierarchy.root_task,
            'statistics': hierarchy.get_hierarchy_statistics() if hasattr(hierarchy, 'get_hierarchy_statistics') else {}
        }
    
    @staticmethod
    def save_hierarchy(hierarchy: Any, 
                      hierarchy_type: HierarchyType, 
                      filepath: str) -> None:
        """Saves hierarchy in file"""
        try:
            serialized = HierarchySerializer.serialize_hierarchy(hierarchy, hierarchy_type)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Hierarchy saved in {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving hierarchy: {e}")
            raise


# Global utilities

def create_hierarchy_hash(components: List[str], parameters: Dict[str, Any]) -> str:
    """Creates hash for hierarchical configuration"""
    content = f"{sorted(components)}_{str(sorted(parameters.items()))}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def compare_hierarchies(hierarchy1: Any, hierarchy2: Any) -> float:
    """Compares two hierarchy and returns coefficient similarity"""
    try:
        # Simple comparison on basis structural characteristics
        if type(hierarchy1) != type(hierarchy2):
            return 0.0
        
        # Compare number components
        size1 = len(getattr(hierarchy1, 'options', getattr(hierarchy1, 'machines', getattr(hierarchy1, 'nodes', {}))))
        size2 = len(getattr(hierarchy2, 'options', getattr(hierarchy2, 'machines', getattr(hierarchy2, 'nodes', {}))))
        
        if size1 == 0 and size2 == 0:
            return 1.0
        
        size_similarity = 1.0 - abs(size1 - size2) / max(size1, size2, 1)
        
        return size_similarity
        
    except Exception as e:
        logger.warning(f"Error comparison hierarchies: {e}")
        return 0.0


def optimize_hierarchy_performance(hierarchy: Any, target_metrics: Dict[str, float]) -> Any:
    """Optimizes performance hierarchy"""
    try:
        # Base optimization - removal unused components
        if hasattr(hierarchy, 'options'):
            # For Options framework
            unused_options = []
            for option_id, option in hierarchy.options.items():
                if hasattr(option, 'execution_count') and option.execution_count == 0:
                    unused_options.append(option_id)
            
            for option_id in unused_options:
                del hierarchy.options[option_id]
                logger.info(f"Deleted unused option: {option_id}")
        
        return hierarchy
        
    except Exception as e:
        logger.error(f"Error optimization hierarchy: {e}")
        return hierarchy


# Export main classes and functions
__all__ = [
    'HierarchyAnalyzer',
    'StateSpaceMapper', 
    'HierarchyBuilder',
    'PerformanceProfiler',
    'HierarchyValidator',
    'HierarchySerializer',
    'HierarchyMetrics',
    'HierarchyType',
    'create_hierarchy_hash',
    'compare_hierarchies',
    'optimize_hierarchy_performance'
]