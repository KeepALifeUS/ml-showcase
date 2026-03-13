"""
Subgoal Discovery Implementation for Hierarchical RL
Automatic detection subgoals for creation hierarchical strategies.

enterprise Pattern:
- Automatic subgoal identification for complex trading environments  
- Production-ready pattern mining with temporal sequence analysis
- Scalable graph-based decomposition for large state spaces
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import asyncio
from collections import deque, defaultdict, Counter
import networkx as nx
import sklearn.cluster as cluster
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

logger = logging.getLogger(__name__)


class SubgoalType(Enum):
    """Types subgoals"""
    BOTTLENECK = "bottleneck"           # Narrow places in state space
    FREQUENT_STATE = "frequent_state"   # Often visited state
    REWARD_PEAK = "reward_peak"         # Peaks reward function
    SKILL_BOUNDARY = "skill_boundary"   # Boundaries between skills
    TEMPORAL_LANDMARK = "temporal_landmark"  # Temporal landmarks


@dataclass
class Subgoal:
    """Representation subgoals"""
    subgoal_id: str
    state_representation: np.ndarray
    subgoal_type: SubgoalType
    confidence: float
    visit_frequency: int
    avg_reward: float
    reachability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySegment:
    """Segment trajectory between subgoals"""
    start_state: np.ndarray
    end_state: np.ndarray
    actions: List[np.ndarray]
    rewards: List[float]
    states: List[np.ndarray]
    duration: int
    success: bool


class StateVisitCounter:
    """Counter visits states with discretization"""
    
    def __init__(self, state_dim: int, discretization_bins: int = 50):
        self.state_dim = state_dim
        self.discretization_bins = discretization_bins
        self.visit_counts: Dict[Tuple, int] = defaultdict(int)
        self.state_ranges: List[Tuple[float, float]] = []
        self.total_visits = 0
        
    def update_ranges(self, states: List[np.ndarray]) -> None:
        """Updates ranges for discretization"""
        if not states:
            return
            
        states_array = np.array(states)
        self.state_ranges = [
            (float(states_array[:, i].min()), float(states_array[:, i].max()))
            for i in range(self.state_dim)
        ]
    
    def discretize_state(self, state: np.ndarray) -> Tuple[int, ...]:
        """Discretizes state"""
        if not self.state_ranges:
            return tuple([0] * len(state))
        
        discretized = []
        for i, (value, (min_val, max_val)) in enumerate(zip(state, self.state_ranges)):
            if max_val > min_val:
                bin_idx = int((value - min_val) / (max_val - min_val) * (self.discretization_bins - 1))
                bin_idx = max(0, min(self.discretization_bins - 1, bin_idx))
            else:
                bin_idx = 0
            discretized.append(bin_idx)
        
        return tuple(discretized)
    
    def add_visit(self, state: np.ndarray) -> None:
        """Adds visit state"""
        discrete_state = self.discretize_state(state)
        self.visit_counts[discrete_state] += 1
        self.total_visits += 1
    
    def get_frequency(self, state: np.ndarray) -> float:
        """Returns frequency visits state"""
        discrete_state = self.discretize_state(state)
        if self.total_visits == 0:
            return 0.0
        return self.visit_counts[discrete_state] / self.total_visits
    
    def get_most_frequent_states(self, top_k: int = 10) -> List[Tuple[Tuple, float]]:
        """Returns most often visited state"""
        if self.total_visits == 0:
            return []
        
        sorted_states = sorted(
            self.visit_counts.items(),
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(state, count / self.total_visits) for state, count in sorted_states[:top_k]]


class BottleneckDetector:
    """Detector narrow places in space states"""
    
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.transition_graph = nx.DiGraph()
        self.state_embeddings: Dict[str, np.ndarray] = {}
        
    def build_transition_graph(self, trajectories: List[List[np.ndarray]]) -> None:
        """Builds graph transitions between states"""
        self.transition_graph.clear()
        
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):
                current_state = trajectory[i]
                next_state = trajectory[i + 1]
                
                # Discretize state for creation nodes graph
                current_node = self._state_to_node(current_state)
                next_node = self._state_to_node(next_state)
                
                # Add edge in graph
                if self.transition_graph.has_edge(current_node, next_node):
                    self.transition_graph[current_node][next_node]['weight'] += 1
                else:
                    self.transition_graph.add_edge(current_node, next_node, weight=1)
                
                # Save embeddings states
                self.state_embeddings[current_node] = current_state
                self.state_embeddings[next_node] = next_state
    
    def _state_to_node(self, state: np.ndarray) -> str:
        """Converts state in node graph"""
        # Simple discretization for creation nodes
        discretized = np.round(state * 10).astype(int)
        return '_'.join(map(str, discretized))
    
    def detect_bottlenecks(self, min_betweenness: float = 0.1) -> List[Subgoal]:
        """Detects narrow places using betweenness centrality"""
        if len(self.transition_graph.nodes()) < 3:
            return []
        
        try:
            # Compute betweenness centrality
            betweenness = nx.betweenness_centrality(self.transition_graph, weight='weight')
            
            bottlenecks = []
            for node, centrality in betweenness.items():
                if centrality >= min_betweenness:
                    state = self.state_embeddings.get(node, np.zeros(self.state_dim))
                    
                    bottleneck = Subgoal(
                        subgoal_id=f"bottleneck_{node}",
                        state_representation=state,
                        subgoal_type=SubgoalType.BOTTLENECK,
                        confidence=centrality,
                        visit_frequency=self.transition_graph.degree(node),
                        avg_reward=0.0,  # Will be updated later
                        reachability_score=centrality,
                        metadata={'betweenness_centrality': centrality}
                    )
                    bottlenecks.append(bottleneck)
            
            logger.info(f"Detected {len(bottlenecks)} narrow places")
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error when detection narrow places: {e}")
            return []


class ClusteringBasedDiscovery:
    """Detection subgoals on basis clustering states"""
    
    def __init__(self, state_dim: int, n_clusters: int = 10):
        self.state_dim = state_dim
        self.n_clusters = n_clusters
        self.clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        
    def discover_subgoals(self, 
                         states: List[np.ndarray], 
                         rewards: List[float]) -> List[Subgoal]:
        """Detects subgoals through clustering high-rewardtion states"""
        if len(states) < self.n_clusters:
            return []
        
        states_array = np.array(states)
        rewards_array = np.array(rewards)
        
        # Filter state with high reward
        high_reward_threshold = np.percentile(rewards_array, 75)
        high_reward_mask = rewards_array >= high_reward_threshold
        
        if np.sum(high_reward_mask) < 3:
            # If few high-rewardtion states, use all
            filtered_states = states_array
            filtered_rewards = rewards_array
        else:
            filtered_states = states_array[high_reward_mask]
            filtered_rewards = rewards_array[high_reward_mask]
        
        try:
            # Execute clustering
            self.cluster_labels = self.clusterer.fit_predict(filtered_states)
            self.cluster_centers = self.clusterer.cluster_centers_
            
            subgoals = []
            for cluster_id in range(self.n_clusters):
                cluster_mask = self.cluster_labels == cluster_id
                
                if np.sum(cluster_mask) == 0:
                    continue
                
                # Statistics cluster
                cluster_states = filtered_states[cluster_mask]
                cluster_rewards = filtered_rewards[cluster_mask]
                
                center = self.cluster_centers[cluster_id]
                avg_reward = np.mean(cluster_rewards)
                visit_frequency = len(cluster_states)
                
                # Estimation quality cluster
                intra_cluster_distance = np.mean([
                    np.linalg.norm(state - center) for state in cluster_states
                ])
                confidence = 1.0 / (1.0 + intra_cluster_distance)
                
                subgoal = Subgoal(
                    subgoal_id=f"cluster_{cluster_id}",
                    state_representation=center,
                    subgoal_type=SubgoalType.REWARD_PEAK,
                    confidence=confidence,
                    visit_frequency=visit_frequency,
                    avg_reward=avg_reward,
                    reachability_score=confidence,
                    metadata={
                        'cluster_size': visit_frequency,
                        'intra_cluster_distance': intra_cluster_distance
                    }
                )
                subgoals.append(subgoal)
            
            logger.info(f"Detected {len(subgoals)} subgoals through clustering")
            return subgoals
            
        except Exception as e:
            logger.error(f"Error when clustering: {e}")
            return []


class TemporalLandmarkDetector:
    """Detector temporal landmarks in trajectories"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
    def detect_landmarks(self, 
                        trajectories: List[List[np.ndarray]],
                        rewards_list: List[List[float]]) -> List[Subgoal]:
        """Detects temporal landmarks"""
        landmarks = []
        landmark_candidates = defaultdict(list)
        
        for traj_idx, (trajectory, rewards) in enumerate(zip(trajectories, rewards_list)):
            # Search local maximums reward
            reward_peaks = self._find_reward_peaks(rewards)
            
            for peak_idx in reward_peaks:
                if peak_idx < len(trajectory):
                    state = trajectory[peak_idx]
                    reward = rewards[peak_idx]
                    
                    # Group similar state
                    state_key = self._discretize_state(state)
                    landmark_candidates[state_key].append({
                        'state': state,
                        'reward': reward,
                        'trajectory': traj_idx,
                        'step': peak_idx
                    })
        
        # Create landmarks from often occurring patterns
        landmark_id = 0
        for state_key, candidates in landmark_candidates.items():
            if len(candidates) >= 3:  # Minimum 3 trajectory must pass through through this landmark
                
                avg_state = np.mean([c['state'] for c in candidates], axis=0)
                avg_reward = np.mean([c['reward'] for c in candidates])
                confidence = len(candidates) / len(trajectories)  # Share trajectories
                
                landmark = Subgoal(
                    subgoal_id=f"landmark_{landmark_id}",
                    state_representation=avg_state,
                    subgoal_type=SubgoalType.TEMPORAL_LANDMARK,
                    confidence=confidence,
                    visit_frequency=len(candidates),
                    avg_reward=avg_reward,
                    reachability_score=confidence,
                    metadata={
                        'peak_occurrences': len(candidates),
                        'trajectory_coverage': confidence
                    }
                )
                landmarks.append(landmark)
                landmark_id += 1
        
        logger.info(f"Detected {len(landmarks)} temporal landmarks")
        return landmarks
    
    def _find_reward_peaks(self, rewards: List[float]) -> List[int]:
        """Finds local maximums in sequences rewards"""
        if len(rewards) < 3:
            return []
        
        peaks = []
        for i in range(1, len(rewards) - 1):
            if (rewards[i] > rewards[i-1] and 
                rewards[i] > rewards[i+1] and 
                rewards[i] > np.mean(rewards)):  # More average
                peaks.append(i)
        
        return peaks
    
    def _discretize_state(self, state: np.ndarray, bins: int = 20) -> Tuple[int, ...]:
        """Discretizes state for grouping"""
        return tuple((state * bins).astype(int))


class SubgoalDiscoveryEngine:
    """
    Main engine for detection subgoals
    Combines various methods detection
    """
    
    def __init__(self, 
                 state_dim: int,
                 discovery_methods: Optional[List[str]] = None):
        self.state_dim = state_dim
        
        if discovery_methods is None:
            discovery_methods = ['clustering', 'bottleneck', 'temporal', 'frequent']
        
        self.discovery_methods = discovery_methods
        
        # Initialize detectors
        self.visit_counter = StateVisitCounter(state_dim)
        self.bottleneck_detector = BottleneckDetector(state_dim)
        self.clustering_discovery = ClusteringBasedDiscovery(state_dim)
        self.temporal_detector = TemporalLandmarkDetector()
        
        # Storage detected subgoals
        self.discovered_subgoals: List[Subgoal] = []
        self.subgoal_graph = nx.Graph()
        
    def discover_subgoals(self, 
                         trajectories: List[List[np.ndarray]],
                         rewards_list: List[List[float]]) -> List[Subgoal]:
        """Detects subgoals using all available methods"""
        if not trajectories:
            return []
        
        all_subgoals = []
        
        # Preparation data
        all_states = [state for traj in trajectories for state in traj]
        all_rewards = [reward for rewards in rewards_list for reward in rewards]
        
        # Update counter visits
        self.visit_counter.update_ranges(all_states)
        for state in all_states:
            self.visit_counter.add_visit(state)
        
        # Method 1: Clustering
        if 'clustering' in self.discovery_methods:
            clustering_subgoals = self.clustering_discovery.discover_subgoals(
                all_states, all_rewards
            )
            all_subgoals.extend(clustering_subgoals)
        
        # Method 2: Narrow places
        if 'bottleneck' in self.discovery_methods:
            self.bottleneck_detector.build_transition_graph(trajectories)
            bottleneck_subgoals = self.bottleneck_detector.detect_bottlenecks()
            all_subgoals.extend(bottleneck_subgoals)
        
        # Method 3: Temporal landmarks
        if 'temporal' in self.discovery_methods:
            temporal_subgoals = self.temporal_detector.detect_landmarks(
                trajectories, rewards_list
            )
            all_subgoals.extend(temporal_subgoals)
        
        # Method 4: Often visited state
        if 'frequent' in self.discovery_methods:
            frequent_subgoals = self._discover_frequent_states()
            all_subgoals.extend(frequent_subgoals)
        
        # Filtering and ranking
        filtered_subgoals = self._filter_and_rank_subgoals(all_subgoals)
        
        # Build graph connections between subgoals
        self._build_subgoal_graph(filtered_subgoals)
        
        self.discovered_subgoals = filtered_subgoals
        logger.info(f"Total detected {len(filtered_subgoals)} subgoals")
        
        return filtered_subgoals
    
    def _discover_frequent_states(self) -> List[Subgoal]:
        """Detects often visited state"""
        frequent_states = self.visit_counter.get_most_frequent_states(top_k=10)
        subgoals = []
        
        for i, (discrete_state, frequency) in enumerate(frequent_states):
            if frequency < 0.01:  # Minimum frequency 1%
                continue
            
            # Restore continuous state from discrete
            continuous_state = self._discrete_to_continuous_state(discrete_state)
            
            subgoal = Subgoal(
                subgoal_id=f"frequent_{i}",
                state_representation=continuous_state,
                subgoal_type=SubgoalType.FREQUENT_STATE,
                confidence=frequency,
                visit_frequency=int(frequency * self.visit_counter.total_visits),
                avg_reward=0.0,  # Will be updated
                reachability_score=frequency,
                metadata={'visit_frequency': frequency}
            )
            subgoals.append(subgoal)
        
        return subgoals
    
    def _discrete_to_continuous_state(self, discrete_state: Tuple[int, ...]) -> np.ndarray:
        """Restores continuous state from discrete"""
        if not self.visit_counter.state_ranges:
            return np.zeros(self.state_dim)
        
        continuous = []
        for i, (bin_idx, (min_val, max_val)) in enumerate(
            zip(discrete_state, self.visit_counter.state_ranges)
        ):
            if max_val > min_val:
                value = min_val + (bin_idx / (self.visit_counter.discretization_bins - 1)) * (max_val - min_val)
            else:
                value = min_val
            continuous.append(value)
        
        return np.array(continuous)
    
    def _filter_and_rank_subgoals(self, 
                                 subgoals: List[Subgoal],
                                 max_subgoals: int = 20,
                                 min_distance: float = 0.1) -> List[Subgoal]:
        """Filters and ranks subgoals"""
        if not subgoals:
            return []
        
        # Remove duplicates (too close subgoals)
        filtered = []
        for subgoal in subgoals:
            is_duplicate = False
            for existing in filtered:
                distance = np.linalg.norm(
                    subgoal.state_representation - existing.state_representation
                )
                if distance < min_distance:
                    # Keep subgoal with more high confidence
                    if subgoal.confidence > existing.confidence:
                        filtered.remove(existing)
                        filtered.append(subgoal)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(subgoal)
        
        # Rank by combined score
        def subgoal_score(subgoal: Subgoal) -> float:
            return (subgoal.confidence * 0.4 + 
                   subgoal.reachability_score * 0.3 +
                   min(subgoal.visit_frequency / 100, 1.0) * 0.3)
        
        filtered.sort(key=subgoal_score, reverse=True)
        
        return filtered[:max_subgoals]
    
    def _build_subgoal_graph(self, subgoals: List[Subgoal]) -> None:
        """Builds graph connections between subgoals"""
        self.subgoal_graph.clear()
        
        # Add nodes
        for subgoal in subgoals:
            self.subgoal_graph.add_node(
                subgoal.subgoal_id,
                state=subgoal.state_representation,
                type=subgoal.subgoal_type.value
            )
        
        # Add edges on basis distances
        for i, subgoal1 in enumerate(subgoals):
            for j, subgoal2 in enumerate(subgoals):
                if i >= j:
                    continue
                
                distance = np.linalg.norm(
                    subgoal1.state_representation - subgoal2.state_representation
                )
                
                # Add edge if subgoals sufficient close
                if distance < 0.5:  # Threshold proximity
                    self.subgoal_graph.add_edge(
                        subgoal1.subgoal_id,
                        subgoal2.subgoal_id,
                        weight=1.0 / (distance + 0.01)
                    )
    
    def get_subgoal_hierarchy(self) -> Dict[str, List[str]]:
        """Returns hierarchy subgoals"""
        if len(self.subgoal_graph.nodes()) == 0:
            return {}
        
        try:
            # Use community for creation hierarchy
            import networkx.algorithms.community as nx_comm
            communities = list(nx_comm.greedy_modularity_communities(self.subgoal_graph))
            
            hierarchy = {}
            for i, community in enumerate(communities):
                hierarchy[f"cluster_{i}"] = list(community)
            
            return hierarchy
            
        except Exception as e:
            logger.warning(f"Not succeeded build hierarchy: {e}")
            return {"all_subgoals": [sg.subgoal_id for sg in self.discovered_subgoals]}
    
    def visualize_subgoals(self, filename: Optional[str] = None) -> None:
        """Creates visualization detected subgoals"""
        if not self.discovered_subgoals:
            logger.warning("No subgoals for visualization")
            return
        
        # Extract state for visualization
        states = np.array([sg.state_representation for sg in self.discovered_subgoals])
        
        # Reduce dimensionality until 2D for visualization
        if states.shape[1] > 2:
            if states.shape[0] > 3:
                tsne = TSNE(n_components=2, random_state=42)
                states_2d = tsne.fit_transform(states)
            else:
                # For small number points use PCA
                pca = PCA(n_components=2)
                states_2d = pca.fit_transform(states)
        else:
            states_2d = states
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Colors for different types subgoals
        type_colors = {
            SubgoalType.BOTTLENECK: 'red',
            SubgoalType.FREQUENT_STATE: 'blue', 
            SubgoalType.REWARD_PEAK: 'green',
            SubgoalType.TEMPORAL_LANDMARK: 'orange',
            SubgoalType.SKILL_BOUNDARY: 'purple'
        }
        
        for i, subgoal in enumerate(self.discovered_subgoals):
            color = type_colors.get(subgoal.subgoal_type, 'gray')
            size = 50 + subgoal.confidence * 200  # Size proportional confidence
            
            plt.scatter(states_2d[i, 0], states_2d[i, 1], 
                       c=color, s=size, alpha=0.7, 
                       label=subgoal.subgoal_type.value if i == 0 else "")
            
            # Signatures
            plt.annotate(subgoal.subgoal_id, 
                        (states_2d[i, 0], states_2d[i, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8)
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title('Discovered Subgoals Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_subgoals(self, filepath: str) -> None:
        """Saves detected subgoals"""
        data = {
            'subgoals': [
                {
                    'subgoal_id': sg.subgoal_id,
                    'state_representation': sg.state_representation.tolist(),
                    'subgoal_type': sg.subgoal_type.value,
                    'confidence': sg.confidence,
                    'visit_frequency': sg.visit_frequency,
                    'avg_reward': sg.avg_reward,
                    'reachability_score': sg.reachability_score,
                    'metadata': sg.metadata
                }
                for sg in self.discovered_subgoals
            ],
            'subgoal_graph': nx.node_link_data(self.subgoal_graph),
            'discovery_params': {
                'state_dim': self.state_dim,
                'discovery_methods': self.discovery_methods
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Subgoals saved in {filepath}")
    
    def load_subgoals(self, filepath: str) -> None:
        """Loads detected subgoals"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore subgoals
        self.discovered_subgoals = []
        for sg_data in data['subgoals']:
            subgoal = Subgoal(
                subgoal_id=sg_data['subgoal_id'],
                state_representation=np.array(sg_data['state_representation']),
                subgoal_type=SubgoalType(sg_data['subgoal_type']),
                confidence=sg_data['confidence'],
                visit_frequency=sg_data['visit_frequency'],
                avg_reward=sg_data['avg_reward'],
                reachability_score=sg_data['reachability_score'],
                metadata=sg_data['metadata']
            )
            self.discovered_subgoals.append(subgoal)
        
        # Restore graph
        self.subgoal_graph = nx.node_link_graph(data['subgoal_graph'])
        
        logger.info(f"Loaded {len(self.discovered_subgoals)} subgoals")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics detection subgoals"""
        if not self.discovered_subgoals:
            return {}
        
        type_counts = Counter([sg.subgoal_type for sg in self.discovered_subgoals])
        
        return {
            'total_subgoals': len(self.discovered_subgoals),
            'subgoals_by_type': {k.value: v for k, v in type_counts.items()},
            'avg_confidence': np.mean([sg.confidence for sg in self.discovered_subgoals]),
            'avg_visit_frequency': np.mean([sg.visit_frequency for sg in self.discovered_subgoals]),
            'subgoal_graph_nodes': len(self.subgoal_graph.nodes()),
            'subgoal_graph_edges': len(self.subgoal_graph.edges()),
            'discovery_methods': self.discovery_methods
        }


# Factory for creation discovery engine

def create_crypto_subgoal_discovery(state_dim: int = 15) -> SubgoalDiscoveryEngine:
    """Creates engine detection subgoals for crypto trading"""
    return SubgoalDiscoveryEngine(
        state_dim=state_dim,
        discovery_methods=['clustering', 'bottleneck', 'temporal', 'frequent']
    )