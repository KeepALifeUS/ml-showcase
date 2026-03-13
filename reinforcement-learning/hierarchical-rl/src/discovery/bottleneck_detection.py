"""
Bottleneck Detection Implementation for Hierarchical RL
Detection narrow places in space states for hierarchical decomposition.

enterprise Pattern:
- Graph-based bottleneck analysis for large-scale state spaces
- Production-ready centrality measures with distributed computation
- Scalable state space decomposition for complex trading environments
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import logging
import asyncio
from collections import deque, defaultdict, Counter
import networkx as nx
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """Types narrow places"""
    BRIDGE = "bridge"                    # Bridge nodes
    ARTICULATION = "articulation"        # Points articulations
    HIGH_BETWEENNESS = "high_betweenness" # High betweenness centrality
    FLOW_CHOKE = "flow_choke"           # Limitations flow
    DENSITY_BOUNDARY = "density_boundary" # Boundaries density


@dataclass
class Bottleneck:
    """Representation narrow places"""
    bottleneck_id: str
    state_representation: np.ndarray
    bottleneck_type: BottleneckType
    centrality_score: float
    flow_capacity: float
    visit_frequency: int
    criticality: float  # Importance for navigation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateTransition:
    """Transition between states"""
    from_state: np.ndarray
    to_state: np.ndarray
    action: np.ndarray
    reward: float
    probability: float = 1.0
    frequency: int = 1


class StateSpaceGraph:
    """Graph space states for analysis bottlenecks"""
    
    def __init__(self, 
                 state_discretization: int = 50,
                 similarity_threshold: float = 0.1):
        self.state_discretization = state_discretization
        self.similarity_threshold = similarity_threshold
        
        # Graph transitions
        self.transition_graph = nx.DiGraph()
        self.undirected_graph = nx.Graph()  # For some algorithms
        
        # Mapping states
        self.state_to_node: Dict[Tuple, str] = {}
        self.node_to_state: Dict[str, np.ndarray] = {}
        self.node_visits: Dict[str, int] = defaultdict(int)
        self.node_rewards: Dict[str, List[float]] = defaultdict(list)
        
        # Statistics
        self.total_transitions = 0
        self.state_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
    def add_trajectory(self, 
                      states: List[np.ndarray],
                      actions: List[np.ndarray],
                      rewards: List[float]) -> None:
        """Adds trajectory in graph"""
        if not states or len(states) < 2:
            return
        
        # Update boundaries states
        self._update_state_bounds(states)
        
        # Add transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            action = actions[i] if i < len(actions) else np.zeros(3)
            reward = rewards[i] if i < len(rewards) else 0.0
            
            # Discretize state
            current_node = self._discretize_state(current_state)
            next_node = self._discretize_state(next_state)
            
            # Add nodes and edges
            self._add_transition(current_node, next_node, current_state, next_state, action, reward)
    
    def _update_state_bounds(self, states: List[np.ndarray]) -> None:
        """Updates boundaries space states"""
        states_array = np.array(states)
        
        if self.state_bounds is None:
            self.state_bounds = (states_array.min(axis=0), states_array.max(axis=0))
        else:
            min_bounds, max_bounds = self.state_bounds
            new_min = np.minimum(min_bounds, states_array.min(axis=0))
            new_max = np.maximum(max_bounds, states_array.max(axis=0))
            self.state_bounds = (new_min, new_max)
    
    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretizes state in node graph"""
        if self.state_bounds is None:
            return f"state_{hash(tuple(state)) % 10000}"
        
        min_bounds, max_bounds = self.state_bounds
        
        # Normalize and discretize
        normalized = np.zeros_like(state)
        for i in range(len(state)):
            if max_bounds[i] > min_bounds[i]:
                normalized[i] = (state[i] - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
            else:
                normalized[i] = 0.5
        
        # Create discrete grid
        discretized = (normalized * (self.state_discretization - 1)).astype(int)
        discretized = np.clip(discretized, 0, self.state_discretization - 1)
        
        node_id = '_'.join(map(str, discretized))
        
        # Save mapping
        state_key = tuple(discretized)
        if state_key not in self.state_to_node:
            self.state_to_node[state_key] = node_id
            self.node_to_state[node_id] = state.copy()
        
        return node_id
    
    def _add_transition(self, 
                       from_node: str,
                       to_node: str,
                       from_state: np.ndarray,
                       to_state: np.ndarray,
                       action: np.ndarray,
                       reward: float) -> None:
        """Adds transition in graph"""
        # Add nodes
        if from_node not in self.transition_graph:
            self.transition_graph.add_node(from_node)
            self.undirected_graph.add_node(from_node)
        
        if to_node not in self.transition_graph:
            self.transition_graph.add_node(to_node)
            self.undirected_graph.add_node(to_node)
        
        # Update statistics nodes
        self.node_visits[from_node] += 1
        self.node_rewards[from_node].append(reward)
        
        # Add/update edge
        if self.transition_graph.has_edge(from_node, to_node):
            # Update weight edges
            self.transition_graph[from_node][to_node]['weight'] += 1
            self.transition_graph[from_node][to_node]['total_reward'] += reward
        else:
            # Create new edge
            self.transition_graph.add_edge(
                from_node, to_node,
                weight=1,
                total_reward=reward,
                action=action.copy()
            )
            
            # Add in undirected graph
            self.undirected_graph.add_edge(from_node, to_node, weight=1)
        
        self.total_transitions += 1
    
    def get_node_properties(self, node: str) -> Dict[str, Any]:
        """Returns properties node"""
        return {
            'visits': self.node_visits[node],
            'avg_reward': np.mean(self.node_rewards[node]) if self.node_rewards[node] else 0.0,
            'state': self.node_to_state.get(node, np.array([])),
            'out_degree': self.transition_graph.out_degree(node),
            'in_degree': self.transition_graph.in_degree(node)
        }


class CentralityAnalyzer:
    """Analyzer centrality for detection bottlenecks"""
    
    def __init__(self):
        self.centrality_cache: Dict[str, Dict[str, float]] = {}
        
    def compute_centralities(self, graph: StateSpaceGraph) -> Dict[str, Dict[str, float]]:
        """Computes various measures centrality"""
        centralities = {}
        
        # Betweenness centrality (most important for bottlenecks)
        try:
            betweenness = nx.betweenness_centrality(
                graph.undirected_graph, 
                weight='weight',
                normalized=True
            )
            centralities['betweenness'] = betweenness
        except Exception as e:
            logger.warning(f"Not succeeded compute betweenness centrality: {e}")
            centralities['betweenness'] = {}
        
        # Closeness centrality
        try:
            closeness = nx.closeness_centrality(
                graph.undirected_graph,
                distance='weight'
            )
            centralities['closeness'] = closeness
        except Exception as e:
            logger.warning(f"Not succeeded compute closeness centrality: {e}")
            centralities['closeness'] = {}
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(
                graph.undirected_graph,
                max_iter=1000,
                weight='weight'
            )
            centralities['eigenvector'] = eigenvector
        except Exception as e:
            logger.warning(f"Not succeeded compute eigenvector centrality: {e}")
            centralities['eigenvector'] = {}
        
        # PageRank (for directed graph)
        try:
            pagerank = nx.pagerank(
                graph.transition_graph,
                weight='weight',
                max_iter=1000
            )
            centralities['pagerank'] = pagerank
        except Exception as e:
            logger.warning(f"Not succeeded compute PageRank: {e}")
            centralities['pagerank'] = {}
        
        # Load centrality (for bottlenecks especially important)
        try:
            load = nx.load_centrality(
                graph.undirected_graph,
                weight='weight'
            )
            centralities['load'] = load
        except Exception as e:
            logger.warning(f"Not succeeded compute load centrality: {e}")
            centralities['load'] = {}
        
        self.centrality_cache = centralities
        return centralities
    
    def identify_high_centrality_nodes(self, 
                                     centralities: Dict[str, Dict[str, float]],
                                     threshold_percentile: float = 90) -> Dict[str, List[str]]:
        """Identifies nodes with high centrality"""
        high_centrality_nodes = {}
        
        for centrality_type, values in centralities.items():
            if not values:
                continue
            
            # Compute threshold
            threshold = np.percentile(list(values.values()), threshold_percentile)
            
            # Find nodes above threshold
            high_nodes = [node for node, value in values.items() if value >= threshold]
            high_centrality_nodes[centrality_type] = high_nodes
        
        return high_centrality_nodes


class BridgeDetector:
    """Detector bridges and points articulations"""
    
    def find_bridges(self, graph: StateSpaceGraph) -> List[Tuple[str, str]]:
        """Finds bridges in graph"""
        try:
            bridges = list(nx.bridges(graph.undirected_graph))
            logger.info(f"Found {len(bridges)} bridges")
            return bridges
        except Exception as e:
            logger.error(f"Error when search bridges: {e}")
            return []
    
    def find_articulation_points(self, graph: StateSpaceGraph) -> List[str]:
        """Finds points articulations"""
        try:
            articulation_points = list(nx.articulation_points(graph.undirected_graph))
            logger.info(f"Found {len(articulation_points)} points articulations")
            return articulation_points
        except Exception as e:
            logger.error(f"Error when search points articulations: {e}")
            return []
    
    def analyze_bridge_importance(self, 
                                 graph: StateSpaceGraph,
                                 bridges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        """Analyzes importance bridges"""
        bridge_importance = {}
        
        for bridge in bridges:
            node1, node2 = bridge
            
            # Importance is based on:
            # 1. Number transitions through bridge
            weight = 0
            if graph.transition_graph.has_edge(node1, node2):
                weight += graph.transition_graph[node1][node2].get('weight', 0)
            if graph.transition_graph.has_edge(node2, node1):
                weight += graph.transition_graph[node2][node1].get('weight', 0)
            
            # 2. Number visits nodes
            visits = graph.node_visits[node1] + graph.node_visits[node2]
            
            # 3. Average reward
            rewards1 = graph.node_rewards[node1]
            rewards2 = graph.node_rewards[node2]
            avg_reward = 0
            if rewards1 or rewards2:
                all_rewards = rewards1 + rewards2
                avg_reward = np.mean(all_rewards)
            
            # Combined score importance
            importance = (weight * 0.4 + visits * 0.4 + max(0, avg_reward) * 0.2)
            bridge_importance[bridge] = importance
        
        return bridge_importance


class FlowAnalyzer:
    """Analyzer flows for detection choke points"""
    
    def __init__(self):
        self.flow_capacity_cache: Dict[str, float] = {}
    
    def compute_max_flow_centrality(self, graph: StateSpaceGraph) -> Dict[str, float]:
        """Computes centrality on basis maximum flow"""
        flow_centralities = {}
        
        nodes = list(graph.transition_graph.nodes())
        if len(nodes) < 3:
            return flow_centralities
        
        # Select representative sources and sinks
        high_visit_nodes = sorted(nodes, key=lambda n: graph.node_visits[n], reverse=True)
        sources = high_visit_nodes[:min(5, len(nodes)//3)]
        sinks = high_visit_nodes[-min(5, len(nodes)//3):]
        
        for node in nodes:
            total_flow_through = 0
            count = 0
            
            # Compute flow through node for various pairs source-sink
            for source in sources:
                for sink in sinks:
                    if source == sink or source == node or sink == node:
                        continue
                    
                    try:
                        # Maximum flow from source to sink
                        flow_value = nx.maximum_flow_value(
                            graph.transition_graph, source, sink, capacity='weight'
                        )
                        
                        # Flow from source to node
                        if nx.has_path(graph.transition_graph, source, node):
                            flow_to_node = nx.maximum_flow_value(
                                graph.transition_graph, source, node, capacity='weight'
                            )
                        else:
                            flow_to_node = 0
                        
                        # Flow from node to sink
                        if nx.has_path(graph.transition_graph, node, sink):
                            flow_from_node = nx.maximum_flow_value(
                                graph.transition_graph, node, sink, capacity='weight'
                            )
                        else:
                            flow_from_node = 0
                        
                        # Node is bottleneck if flow through it less total flow
                        if flow_value > 0:
                            bottleneck_factor = min(flow_to_node, flow_from_node) / flow_value
                            total_flow_through += bottleneck_factor
                            count += 1
                            
                    except Exception as e:
                        continue  # Skip problematic pairs
            
            if count > 0:
                flow_centralities[node] = total_flow_through / count
            else:
                flow_centralities[node] = 0.0
        
        return flow_centralities
    
    def identify_choke_points(self, 
                            graph: StateSpaceGraph,
                            flow_centralities: Dict[str, float],
                            threshold: float = 0.1) -> List[str]:
        """Identifies choke points"""
        choke_points = []
        
        # Find nodes with low flow centrality (they limit flow)
        for node, centrality in flow_centralities.items():
            if centrality < threshold and graph.node_visits[node] > 1:
                choke_points.append(node)
        
        logger.info(f"Found {len(choke_points)} choke points")
        return choke_points


class DensityAnalyzer:
    """Analyzer density for detection boundaries areas"""
    
    def __init__(self, eps: float = 0.1, min_samples: int = 3):
        self.eps = eps
        self.min_samples = min_samples
        
    def detect_density_boundaries(self, graph: StateSpaceGraph) -> List[str]:
        """Detects boundaries density states"""
        if len(graph.node_to_state) < self.min_samples * 2:
            return []
        
        # Extract state and their frequencies visits
        states = []
        visits = []
        node_ids = []
        
        for node_id, state in graph.node_to_state.items():
            states.append(state)
            visits.append(graph.node_visits[node_id])
            node_ids.append(node_id)
        
        states_array = np.array(states)
        
        # Normalize state
        scaler = StandardScaler()
        states_normalized = scaler.fit_transform(states_array)
        
        # Clustering DBSCAN for search dense areas
        try:
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            cluster_labels = dbscan.fit_predict(states_normalized)
            
            # Find nodes on boundaries clusters
            boundary_nodes = []
            
            for i, (node_id, label) in enumerate(zip(node_ids, cluster_labels)):
                if label == -1:  # Noise points (potential boundaries)
                    boundary_nodes.append(node_id)
                    continue
                
                # Check, exists whether neighbors from other clusters
                neighbors = list(graph.transition_graph.neighbors(node_id))
                neighbor_labels = []
                
                for neighbor in neighbors:
                    if neighbor in node_ids:
                        neighbor_idx = node_ids.index(neighbor)
                        neighbor_labels.append(cluster_labels[neighbor_idx])
                
                # If exists neighbors from other clusters - this boundary
                if neighbor_labels and any(nl != label for nl in neighbor_labels):
                    boundary_nodes.append(node_id)
            
            logger.info(f"Found {len(boundary_nodes)} nodes on boundaries density")
            return boundary_nodes
            
        except Exception as e:
            logger.error(f"Error when analysis density: {e}")
            return []


class BottleneckDetectionEngine:
    """
    Main engine for detection narrow places
    Combines various methods analysis
    """
    
    def __init__(self, 
                 state_discretization: int = 50,
                 min_centrality_threshold: float = 0.1):
        self.state_discretization = state_discretization
        self.min_centrality_threshold = min_centrality_threshold
        
        # Components analysis
        self.state_graph = StateSpaceGraph(state_discretization)
        self.centrality_analyzer = CentralityAnalyzer()
        self.bridge_detector = BridgeDetector()
        self.flow_analyzer = FlowAnalyzer()
        self.density_analyzer = DensityAnalyzer()
        
        # Detected bottlenecks
        self.discovered_bottlenecks: List[Bottleneck] = []
        
    def detect_bottlenecks(self, 
                          trajectories: List[List[np.ndarray]],
                          actions_list: List[List[np.ndarray]],
                          rewards_list: List[List[float]]) -> List[Bottleneck]:
        """Detects narrow places in trajectories"""
        if not trajectories:
            return []
        
        logger.info(f"Begin detection bottlenecks from {len(trajectories)} trajectories")
        
        # Step 1: Construction graph space states
        for trajectory, actions, rewards in zip(trajectories, actions_list, rewards_list):
            self.state_graph.add_trajectory(trajectory, actions, rewards)
        
        logger.info(f"Built graph with {len(self.state_graph.transition_graph.nodes())} nodes and {len(self.state_graph.transition_graph.edges())} edges")
        
        all_bottlenecks = []
        
        # Step 2: Analysis centrality
        centralities = self.centrality_analyzer.compute_centralities(self.state_graph)
        high_centrality_nodes = self.centrality_analyzer.identify_high_centrality_nodes(centralities)
        
        # Create bottlenecks from nodes with high betweenness centrality
        for node in high_centrality_nodes.get('betweenness', []):
            centrality_score = centralities['betweenness'].get(node, 0)
            if centrality_score >= self.min_centrality_threshold:
                bottleneck = self._create_bottleneck(
                    node, BottleneckType.HIGH_BETWEENNESS, centrality_score
                )
                all_bottlenecks.append(bottleneck)
        
        # Step 3: Search bridges and points articulations
        bridges = self.bridge_detector.find_bridges(self.state_graph)
        articulation_points = self.bridge_detector.find_articulation_points(self.state_graph)
        
        # Create bottlenecks from bridges
        bridge_importance = self.bridge_detector.analyze_bridge_importance(self.state_graph, bridges)
        for bridge, importance in bridge_importance.items():
            node1, node2 = bridge
            # Select node with large number visits
            selected_node = node1 if self.state_graph.node_visits[node1] >= self.state_graph.node_visits[node2] else node2
            
            bottleneck = self._create_bottleneck(
                selected_node, BottleneckType.BRIDGE, importance
            )
            all_bottlenecks.append(bottleneck)
        
        # Create bottlenecks from points articulations
        for node in articulation_points:
            visits = self.state_graph.node_visits[node]
            bottleneck = self._create_bottleneck(
                node, BottleneckType.ARTICULATION, visits / max(self.state_graph.total_transitions, 1)
            )
            all_bottlenecks.append(bottleneck)
        
        # Step 4: Analysis flows
        try:
            flow_centralities = self.flow_analyzer.compute_max_flow_centrality(self.state_graph)
            choke_points = self.flow_analyzer.identify_choke_points(
                self.state_graph, flow_centralities
            )
            
            for node in choke_points:
                flow_capacity = flow_centralities.get(node, 0)
                bottleneck = self._create_bottleneck(
                    node, BottleneckType.FLOW_CHOKE, 1.0 - flow_capacity
                )
                all_bottlenecks.append(bottleneck)
        except Exception as e:
            logger.warning(f"Error when analysis flows: {e}")
        
        # Step 5: Analysis density
        try:
            density_boundaries = self.density_analyzer.detect_density_boundaries(self.state_graph)
            
            for node in density_boundaries:
                visits = self.state_graph.node_visits[node]
                boundary_score = visits / max(self.state_graph.total_transitions, 1)
                bottleneck = self._create_bottleneck(
                    node, BottleneckType.DENSITY_BOUNDARY, boundary_score
                )
                all_bottlenecks.append(bottleneck)
        except Exception as e:
            logger.warning(f"Error when analysis density: {e}")
        
        # Step 6: Filtering and ranking
        filtered_bottlenecks = self._filter_and_rank_bottlenecks(all_bottlenecks)
        
        self.discovered_bottlenecks = filtered_bottlenecks
        logger.info(f"Detected {len(filtered_bottlenecks)} bottlenecks")
        
        return filtered_bottlenecks
    
    def _create_bottleneck(self, 
                          node: str,
                          bottleneck_type: BottleneckType,
                          score: float) -> Bottleneck:
        """Creates object bottleneck from node graph"""
        state = self.state_graph.node_to_state.get(node, np.array([]))
        visits = self.state_graph.node_visits[node]
        rewards = self.state_graph.node_rewards[node]
        
        # Compute criticality
        criticality = self._compute_criticality(node, bottleneck_type, score)
        
        return Bottleneck(
            bottleneck_id=f"{bottleneck_type.value}_{node}",
            state_representation=state,
            bottleneck_type=bottleneck_type,
            centrality_score=score,
            flow_capacity=1.0 - score if bottleneck_type == BottleneckType.FLOW_CHOKE else score,
            visit_frequency=visits,
            criticality=criticality,
            metadata={
                'node_id': node,
                'avg_reward': np.mean(rewards) if rewards else 0.0,
                'out_degree': self.state_graph.transition_graph.out_degree(node),
                'in_degree': self.state_graph.transition_graph.in_degree(node)
            }
        )
    
    def _compute_criticality(self, 
                           node: str,
                           bottleneck_type: BottleneckType,
                           score: float) -> float:
        """Computes criticality bottleneck"""
        # Base criticality from type
        type_weights = {
            BottleneckType.BRIDGE: 0.9,
            BottleneckType.ARTICULATION: 0.8,
            BottleneckType.HIGH_BETWEENNESS: 0.7,
            BottleneckType.FLOW_CHOKE: 0.6,
            BottleneckType.DENSITY_BOUNDARY: 0.4
        }
        
        base_criticality = type_weights.get(bottleneck_type, 0.5)
        
        # Adjustment on basis frequencies visits
        visits = self.state_graph.node_visits[node]
        visit_factor = min(visits / max(self.state_graph.total_transitions / len(self.state_graph.transition_graph.nodes()), 1), 2.0)
        
        # Adjustment on basis degree node
        degree = self.state_graph.transition_graph.degree(node)
        degree_factor = min(degree / 10.0, 1.0)
        
        # Final criticality
        criticality = base_criticality * score * visit_factor * degree_factor
        return min(criticality, 1.0)
    
    def _filter_and_rank_bottlenecks(self, 
                                   bottlenecks: List[Bottleneck],
                                   max_bottlenecks: int = 15,
                                   min_criticality: float = 0.1) -> List[Bottleneck]:
        """Filters and ranks bottlenecks"""
        # Remove duplicates (one node can be bottleneck by several criteria)
        unique_bottlenecks = {}
        
        for bottleneck in bottlenecks:
            node_id = bottleneck.metadata.get('node_id')
            if node_id not in unique_bottlenecks:
                unique_bottlenecks[node_id] = bottleneck
            else:
                # Keep bottleneck with more high criticality
                if bottleneck.criticality > unique_bottlenecks[node_id].criticality:
                    unique_bottlenecks[node_id] = bottleneck
        
        filtered = list(unique_bottlenecks.values())
        
        # Filter by minimum criticality
        filtered = [b for b in filtered if b.criticality >= min_criticality]
        
        # Sort by criticality
        filtered.sort(key=lambda b: b.criticality, reverse=True)
        
        return filtered[:max_bottlenecks]
    
    def visualize_bottlenecks(self, filename: Optional[str] = None) -> None:
        """Creates visualization detected bottlenecks"""
        if not self.discovered_bottlenecks:
            logger.warning("No bottlenecks for visualization")
            return
        
        # Extract state for visualization
        states = np.array([b.state_representation for b in self.discovered_bottlenecks])
        
        # Reduce dimensionality until 2D
        if states.shape[1] > 2:
            if states.shape[0] > 3:
                tsne = TSNE(n_components=2, random_state=42)
                states_2d = tsne.fit_transform(states)
            else:
                states_2d = states[:, :2]  # Take first 2 components
        else:
            states_2d = states
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Colors for different types bottlenecks
        type_colors = {
            BottleneckType.BRIDGE: 'red',
            BottleneckType.ARTICULATION: 'orange',
            BottleneckType.HIGH_BETWEENNESS: 'green',
            BottleneckType.FLOW_CHOKE: 'blue',
            BottleneckType.DENSITY_BOUNDARY: 'purple'
        }
        
        for i, bottleneck in enumerate(self.discovered_bottlenecks):
            color = type_colors.get(bottleneck.bottleneck_type, 'gray')
            size = 50 + bottleneck.criticality * 300  # Size proportional criticality
            
            plt.scatter(states_2d[i, 0], states_2d[i, 1],
                       c=color, s=size, alpha=0.7,
                       label=bottleneck.bottleneck_type.value if i == 0 else "")
            
            # Signatures
            plt.annotate(f"{bottleneck.bottleneck_id}\n({bottleneck.criticality:.2f})",
                        (states_2d[i, 0], states_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Discovered Bottlenecks Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_bottlenecks(self, filepath: str) -> None:
        """Saves detected bottlenecks"""
        data = {
            'bottlenecks': [
                {
                    'bottleneck_id': b.bottleneck_id,
                    'state_representation': b.state_representation.tolist(),
                    'bottleneck_type': b.bottleneck_type.value,
                    'centrality_score': b.centrality_score,
                    'flow_capacity': b.flow_capacity,
                    'visit_frequency': b.visit_frequency,
                    'criticality': b.criticality,
                    'metadata': b.metadata
                }
                for b in self.discovered_bottlenecks
            ],
            'graph_info': {
                'nodes': len(self.state_graph.transition_graph.nodes()),
                'edges': len(self.state_graph.transition_graph.edges()),
                'total_transitions': self.state_graph.total_transitions
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Bottlenecks saved in {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics detection bottlenecks"""
        if not self.discovered_bottlenecks:
            return {}
        
        type_counts = Counter([b.bottleneck_type for b in self.discovered_bottlenecks])
        
        return {
            'total_bottlenecks': len(self.discovered_bottlenecks),
            'bottlenecks_by_type': {k.value: v for k, v in type_counts.items()},
            'avg_criticality': np.mean([b.criticality for b in self.discovered_bottlenecks]),
            'avg_visit_frequency': np.mean([b.visit_frequency for b in self.discovered_bottlenecks]),
            'graph_nodes': len(self.state_graph.transition_graph.nodes()),
            'graph_edges': len(self.state_graph.transition_graph.edges()),
            'total_transitions': self.state_graph.total_transitions
        }


# Factory for creation bottleneck detection engine

def create_crypto_bottleneck_detector(state_discretization: int = 50) -> BottleneckDetectionEngine:
    """Creates detector bottlenecks for crypto trading"""
    return BottleneckDetectionEngine(
        state_discretization=state_discretization,
        min_centrality_threshold=0.05  # Low threshold for financial data
    )