"""
Skill Discovery Implementation for Hierarchical RL
Automatic detection skills from trajectories for creation reusable behaviors.

enterprise Pattern:
- Automatic skill segmentation for modular trading strategies
- Production-ready behavior clustering with transfer learning
- Scalable skill composition for complex multi-step strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import asyncio
from collections import deque, defaultdict, Counter
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import networkx as nx
import pickle
import json

logger = logging.getLogger(__name__)


class SkillPattern(Enum):
    """Types patterns skills"""
    SEQUENTIAL = "sequential"       # Sequential actions
    REPETITIVE = "repetitive"      # Repeating actions
    CONDITIONAL = "conditional"    # Conditional actions
    REACTIVE = "reactive"          # Reactive actions
    EXPLORATORY = "exploratory"    # Research actions


@dataclass
class DiscoveredSkill:
    """Detected skill"""
    skill_id: str
    pattern_type: SkillPattern
    action_sequence: List[np.ndarray]
    state_conditions: List[np.ndarray]
    success_rate: float
    frequency: int
    avg_reward: float
    duration: int
    transferability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillSegment:
    """Segment trajectory as potential skill"""
    start_idx: int
    end_idx: int
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    total_reward: float
    success: bool


class TrajectorySegmenter:
    """Segmentation trajectories on potential skills"""
    
    def __init__(self, 
                 min_segment_length: int = 3,
                 max_segment_length: int = 50,
                 reward_threshold: float = 0.01):
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.reward_threshold = reward_threshold
        
    def segment_trajectories(self, 
                           trajectories: List[List[np.ndarray]],
                           actions_list: List[List[np.ndarray]],
                           rewards_list: List[List[float]]) -> List[SkillSegment]:
        """Segments trajectory on potential skills"""
        all_segments = []
        
        for traj_idx, (states, actions, rewards) in enumerate(
            zip(trajectories, actions_list, rewards_list)
        ):
            segments = self._segment_single_trajectory(states, actions, rewards)
            all_segments.extend(segments)
        
        logger.info(f"Created {len(all_segments)} segments from {len(trajectories)} trajectories")
        return all_segments
    
    def _segment_single_trajectory(self, 
                                  states: List[np.ndarray],
                                  actions: List[np.ndarray],
                                  rewards: List[float]) -> List[SkillSegment]:
        """Segments one trajectory"""
        segments = []
        
        # Method 1: Segmentation by changes in reward
        reward_change_points = self._find_reward_change_points(rewards)
        
        # Method 2: Segmentation by changes in actions
        action_change_points = self._find_action_change_points(actions)
        
        # Combine points changes
        change_points = sorted(set(reward_change_points + action_change_points))
        change_points = [0] + change_points + [len(states)]
        
        # Create segments
        for i in range(len(change_points) - 1):
            start_idx = change_points[i]
            end_idx = min(change_points[i + 1], len(states) - 1)
            
            if end_idx - start_idx >= self.min_segment_length:
                segment_states = states[start_idx:end_idx]
                segment_actions = actions[start_idx:end_idx-1] if end_idx > start_idx else []
                segment_rewards = rewards[start_idx:end_idx]
                
                total_reward = sum(segment_rewards)
                success = total_reward > self.reward_threshold
                
                segment = SkillSegment(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    states=segment_states,
                    actions=segment_actions,
                    rewards=segment_rewards,
                    total_reward=total_reward,
                    success=success
                )
                segments.append(segment)
        
        return segments
    
    def _find_reward_change_points(self, rewards: List[float]) -> List[int]:
        """Finds points significant changes in reward"""
        if len(rewards) < 3:
            return []
        
        change_points = []
        window_size = 3
        
        for i in range(window_size, len(rewards) - window_size):
            # Compare average until and after points
            before_mean = np.mean(rewards[i-window_size:i])
            after_mean = np.mean(rewards[i:i+window_size])
            
            # If change more threshold
            if abs(after_mean - before_mean) > self.reward_threshold * 2:
                change_points.append(i)
        
        return change_points
    
    def _find_action_change_points(self, actions: List[np.ndarray]) -> List[int]:
        """Finds points significant changes in actions"""
        if len(actions) < 3:
            return []
        
        change_points = []
        
        for i in range(1, len(actions)):
            # Compute difference between neighboring actions
            action_diff = np.linalg.norm(actions[i] - actions[i-1])
            
            # If difference more threshold
            if action_diff > 0.5:  # Threshold value
                change_points.append(i)
        
        return change_points


class ActionSequenceEncoder:
    """Encoder sequences actions for clustering"""
    
    def __init__(self, action_dim: int, sequence_length: int = 10):
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def encode_segments(self, segments: List[SkillSegment]) -> np.ndarray:
        """Encodes segments in vectors features"""
        features = []
        
        for segment in segments:
            feature_vector = self._encode_single_segment(segment)
            features.append(feature_vector)
        
        if features:
            features_array = np.array(features)
            # Normalize features
            features_normalized = self.scaler.fit_transform(features_array)
            return features_normalized
        
        return np.array([])
    
    def _encode_single_segment(self, segment: SkillSegment) -> np.ndarray:
        """Encodes one segment in vector features"""
        features = []
        
        # 1. Statistics actions
        if segment.actions:
            actions_array = np.array(segment.actions)
            
            # Average actions
            features.extend(np.mean(actions_array, axis=0))
            
            # Standard deviations actions
            features.extend(np.std(actions_array, axis=0))
            
            # Range actions
            features.extend(np.max(actions_array, axis=0) - np.min(actions_array, axis=0))
        else:
            # Fill zeros if no actions
            features.extend([0.0] * (self.action_dim * 3))
        
        # 2. Characteristics sequences
        features.append(len(segment.actions))  # Length sequences
        features.append(segment.total_reward)  # Total reward
        features.append(np.mean(segment.rewards) if segment.rewards else 0.0)  # Average reward
        features.append(np.std(segment.rewards) if len(segment.rewards) > 1 else 0.0)  # Spread rewards
        
        # 3. Patterns in actions
        if len(segment.actions) > 1:
            action_changes = [
                np.linalg.norm(segment.actions[i] - segment.actions[i-1]) 
                for i in range(1, len(segment.actions))
            ]
            features.append(np.mean(action_changes))  # Average variability
            features.append(np.std(action_changes) if len(action_changes) > 1 else 0.0)  # Stability
        else:
            features.extend([0.0, 0.0])
        
        # 4. Success rate
        features.append(float(segment.success))
        
        return np.array(features)


class SkillClusterer:
    """Clustering segments for detection skills"""
    
    def __init__(self, 
                 min_clusters: int = 3,
                 max_clusters: int = 20,
                 min_cluster_size: int = 3):
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.optimal_clusters = None
        self.clusterer = None
        
    def cluster_segments(self, 
                        encoded_segments: np.ndarray,
                        segments: List[SkillSegment]) -> List[List[SkillSegment]]:
        """Clusters segments for detection skills"""
        if len(encoded_segments) < self.min_clusters:
            logger.warning("Insufficient segments for clustering")
            return []
        
        # Find optimal number clusters
        self.optimal_clusters = self._find_optimal_clusters(encoded_segments)
        
        # Execute clustering
        self.clusterer = cluster.KMeans(
            n_clusters=self.optimal_clusters, 
            random_state=42,
            n_init=10
        )
        
        cluster_labels = self.clusterer.fit_predict(encoded_segments)
        
        # Group segments by clusters
        clustered_segments = [[] for _ in range(self.optimal_clusters)]
        
        for segment, label in zip(segments, cluster_labels):
            clustered_segments[label].append(segment)
        
        # Filter clusters by minimum size
        filtered_clusters = [
            cluster for cluster in clustered_segments 
            if len(cluster) >= self.min_cluster_size
        ]
        
        logger.info(f"Created {len(filtered_clusters)} clusters skills")
        return filtered_clusters
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Finds optimal number clusters"""
        max_clusters = min(self.max_clusters, len(data) // 2)
        
        if max_clusters < self.min_clusters:
            return self.min_clusters
        
        silhouette_scores = []
        cluster_range = range(self.min_clusters, max_clusters + 1)
        
        for n_clusters in cluster_range:
            try:
                clusterer = cluster.KMeans(n_clusters=n_clusters, random_state=42)
                labels = clusterer.fit_predict(data)
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            except Exception as e:
                logger.warning(f"Error when clustering with {n_clusters} clusters: {e}")
                silhouette_scores.append(-1)
        
        if silhouette_scores:
            best_idx = np.argmax(silhouette_scores)
            optimal = cluster_range[best_idx]
            logger.info(f"Optimal number clusters: {optimal} (silhouette: {silhouette_scores[best_idx]:.3f})")
            return optimal
        
        return self.min_clusters


class SkillPatternAnalyzer:
    """Analyzer patterns in detected skills"""
    
    def analyze_skill_clusters(self, 
                              skill_clusters: List[List[SkillSegment]]) -> List[DiscoveredSkill]:
        """Analyzes clusters segments and creates skills"""
        discovered_skills = []
        
        for cluster_idx, cluster in enumerate(skill_clusters):
            if not cluster:
                continue
                
            skill = self._analyze_single_cluster(cluster, cluster_idx)
            if skill:
                discovered_skills.append(skill)
        
        logger.info(f"Analyzed {len(discovered_skills)} skills")
        return discovered_skills
    
    def _analyze_single_cluster(self, 
                               cluster: List[SkillSegment], 
                               cluster_idx: int) -> Optional[DiscoveredSkill]:
        """Analyzes one cluster segments"""
        if not cluster:
            return None
        
        # Define type pattern
        pattern_type = self._identify_pattern_type(cluster)
        
        # Extract typical sequence actions
        action_sequence = self._extract_action_sequence(cluster)
        
        # Extract conditions states
        state_conditions = self._extract_state_conditions(cluster)
        
        # Compute metrics
        success_rate = sum(1 for seg in cluster if seg.success) / len(cluster)
        avg_reward = np.mean([seg.total_reward for seg in cluster])
        avg_duration = np.mean([len(seg.actions) for seg in cluster])
        
        # Estimation portability
        transferability_score = self._compute_transferability(cluster)
        
        skill = DiscoveredSkill(
            skill_id=f"skill_{cluster_idx}_{pattern_type.value}",
            pattern_type=pattern_type,
            action_sequence=action_sequence,
            state_conditions=state_conditions,
            success_rate=success_rate,
            frequency=len(cluster),
            avg_reward=avg_reward,
            duration=int(avg_duration),
            transferability_score=transferability_score,
            metadata={
                'cluster_size': len(cluster),
                'reward_std': np.std([seg.total_reward for seg in cluster]),
                'duration_std': np.std([len(seg.actions) for seg in cluster])
            }
        )
        
        return skill
    
    def _identify_pattern_type(self, cluster: List[SkillSegment]) -> SkillPattern:
        """Determines type pattern skill"""
        # Analyze characteristics cluster
        action_sequences = [seg.actions for seg in cluster if seg.actions]
        
        if not action_sequences:
            return SkillPattern.EXPLORATORY
        
        # Check on repeating patterns
        if self._has_repetitive_patterns(action_sequences):
            return SkillPattern.REPETITIVE
        
        # Check on sequential patterns
        if self._has_sequential_patterns(action_sequences):
            return SkillPattern.SEQUENTIAL
        
        # Check on conditional patterns
        if self._has_conditional_patterns(cluster):
            return SkillPattern.CONDITIONAL
        
        # Check on reactive patterns
        if self._has_reactive_patterns(cluster):
            return SkillPattern.REACTIVE
        
        return SkillPattern.EXPLORATORY
    
    def _has_repetitive_patterns(self, action_sequences: List[List[np.ndarray]]) -> bool:
        """Checks presence repeating patterns"""
        for actions in action_sequences:
            if len(actions) < 4:
                continue
            
            # Search repeating subsequences
            for pattern_length in range(2, len(actions) // 2 + 1):
                for start in range(len(actions) - pattern_length * 2 + 1):
                    pattern = actions[start:start + pattern_length]
                    next_pattern = actions[start + pattern_length:start + 2 * pattern_length]
                    
                    # Check similarity patterns
                    if self._sequences_similar(pattern, next_pattern, threshold=0.1):
                        return True
        
        return False
    
    def _has_sequential_patterns(self, action_sequences: List[List[np.ndarray]]) -> bool:
        """Checks presence sequential patterns"""
        # Simple heuristic: sequential changes in actions
        for actions in action_sequences:
            if len(actions) < 3:
                continue
            
            # Check gradients changes
            gradients = []
            for i in range(1, len(actions)):
                diff = actions[i] - actions[i-1]
                gradients.append(np.linalg.norm(diff))
            
            # If gradients relatively stable - this sequential pattern
            if len(gradients) > 2 and np.std(gradients) < np.mean(gradients) * 0.5:
                return True
        
        return False
    
    def _has_conditional_patterns(self, cluster: List[SkillSegment]) -> bool:
        """Checks presence conditional patterns"""
        # Simple heuristic: different actions in similar states
        if len(cluster) < 3:
            return False
        
        # Group by initial states
        state_groups = defaultdict(list)
        for seg in cluster:
            if seg.states:
                state_key = tuple(np.round(seg.states[0], 2))  # Round for grouping
                state_groups[state_key].append(seg.actions)
        
        # Check diversity actions in groups
        for actions_list in state_groups.values():
            if len(actions_list) > 1:
                action_diversity = self._compute_action_diversity(actions_list)
                if action_diversity > 0.3:  # Threshold diversity
                    return True
        
        return False
    
    def _has_reactive_patterns(self, cluster: List[SkillSegment]) -> bool:
        """Checks presence reactive patterns"""
        # Simple heuristic: fast changes actions in response on changes states
        reactive_count = 0
        
        for seg in cluster:
            if len(seg.states) < 3 or len(seg.actions) < 2:
                continue
            
            # Search correlation between changes states and actions
            state_changes = []
            action_changes = []
            
            for i in range(1, min(len(seg.states), len(seg.actions) + 1)):
                if i < len(seg.states):
                    state_change = np.linalg.norm(seg.states[i] - seg.states[i-1])
                    state_changes.append(state_change)
                
                if i-1 < len(seg.actions) and i-2 >= 0 and i-2 < len(seg.actions):
                    action_change = np.linalg.norm(seg.actions[i-1] - seg.actions[i-2])
                    action_changes.append(action_change)
            
            # Check correlation
            if len(state_changes) > 2 and len(action_changes) > 2:
                correlation = np.corrcoef(state_changes[:len(action_changes)], action_changes)[0, 1]
                if not np.isnan(correlation) and correlation > 0.5:
                    reactive_count += 1
        
        return reactive_count > len(cluster) * 0.3  # 30% segments must be reactive
    
    def _sequences_similar(self, 
                          seq1: List[np.ndarray], 
                          seq2: List[np.ndarray], 
                          threshold: float = 0.1) -> bool:
        """Checks similarity two sequences actions"""
        if len(seq1) != len(seq2):
            return False
        
        total_distance = 0
        for a1, a2 in zip(seq1, seq2):
            total_distance += np.linalg.norm(a1 - a2)
        
        avg_distance = total_distance / len(seq1)
        return avg_distance < threshold
    
    def _compute_action_diversity(self, actions_list: List[List[np.ndarray]]) -> float:
        """Computes diversity actions"""
        if len(actions_list) < 2:
            return 0.0
        
        all_actions = []
        for actions in actions_list:
            all_actions.extend(actions)
        
        if len(all_actions) < 2:
            return 0.0
        
        # Compute pairwise distances
        distances = []
        for i in range(len(all_actions)):
            for j in range(i + 1, len(all_actions)):
                distance = np.linalg.norm(all_actions[i] - all_actions[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _extract_action_sequence(self, cluster: List[SkillSegment]) -> List[np.ndarray]:
        """Extracts typical sequence actions"""
        # Find median length sequences
        lengths = [len(seg.actions) for seg in cluster if seg.actions]
        if not lengths:
            return []
        
        median_length = int(np.median(lengths))
        
        # Average actions by positions
        action_sequence = []
        for pos in range(median_length):
            position_actions = []
            for seg in cluster:
                if pos < len(seg.actions):
                    position_actions.append(seg.actions[pos])
            
            if position_actions:
                avg_action = np.mean(position_actions, axis=0)
                action_sequence.append(avg_action)
        
        return action_sequence
    
    def _extract_state_conditions(self, cluster: List[SkillSegment]) -> List[np.ndarray]:
        """Extracts conditions states for skill"""
        # Extract initial state
        initial_states = [seg.states[0] for seg in cluster if seg.states]
        
        if not initial_states:
            return []
        
        # Compute typical initial state and range
        mean_state = np.mean(initial_states, axis=0)
        std_state = np.std(initial_states, axis=0)
        
        # Return average state as condition
        return [mean_state]
    
    def _compute_transferability(self, cluster: List[SkillSegment]) -> float:
        """Evaluates portability skill"""
        if len(cluster) < 2:
            return 0.0
        
        # Factors portability:
        # 1. Stability reward
        rewards = [seg.total_reward for seg in cluster]
        reward_stability = 1.0 / (1.0 + np.std(rewards))
        
        # 2. Stability duration
        durations = [len(seg.actions) for seg in cluster]
        duration_stability = 1.0 / (1.0 + np.std(durations) / max(np.mean(durations), 1))
        
        # 3. Success rate
        success_rate = sum(1 for seg in cluster if seg.success) / len(cluster)
        
        # Combined score
        transferability = (reward_stability * 0.3 + 
                          duration_stability * 0.3 + 
                          success_rate * 0.4)
        
        return min(transferability, 1.0)


class SkillDiscoveryEngine:
    """
    Main engine for detection skills
    Combines segmentation, clustering and analysis patterns
    """
    
    def __init__(self, 
                 action_dim: int,
                 state_dim: int,
                 min_skill_frequency: int = 3,
                 min_success_rate: float = 0.3):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.min_skill_frequency = min_skill_frequency
        self.min_success_rate = min_success_rate
        
        # Components pipeline
        self.segmenter = TrajectorySegmenter()
        self.encoder = ActionSequenceEncoder(action_dim)
        self.clusterer = SkillClusterer()
        self.analyzer = SkillPatternAnalyzer()
        
        # Detected skills
        self.discovered_skills: List[DiscoveredSkill] = []
        self.skill_graph = nx.Graph()
        
    def discover_skills(self, 
                       trajectories: List[List[np.ndarray]],
                       actions_list: List[List[np.ndarray]],
                       rewards_list: List[List[float]]) -> List[DiscoveredSkill]:
        """Detects skills from trajectories"""
        if not trajectories or not actions_list:
            logger.warning("No data for detection skills")
            return []
        
        logger.info(f"Begin detection skills from {len(trajectories)} trajectories")
        
        # Step 1: Segmentation trajectories
        segments = self.segmenter.segment_trajectories(
            trajectories, actions_list, rewards_list
        )
        
        if len(segments) < self.min_skill_frequency:
            logger.warning("Insufficient segments for detection skills")
            return []
        
        # Step 2: Encoding segments
        encoded_segments = self.encoder.encode_segments(segments)
        
        if len(encoded_segments) == 0:
            logger.warning("Not succeeded encode segments")
            return []
        
        # Step 3: Clustering
        skill_clusters = self.clusterer.cluster_segments(encoded_segments, segments)
        
        # Step 4: Analysis patterns
        discovered_skills = self.analyzer.analyze_skill_clusters(skill_clusters)
        
        # Step 5: Filtering by quality
        filtered_skills = self._filter_skills(discovered_skills)
        
        # Step 6: Construction graph skills
        self._build_skill_graph(filtered_skills)
        
        self.discovered_skills = filtered_skills
        logger.info(f"Detected {len(filtered_skills)} qualitative skills")
        
        return filtered_skills
    
    def _filter_skills(self, skills: List[DiscoveredSkill]) -> List[DiscoveredSkill]:
        """Filters skills by quality"""
        filtered = []
        
        for skill in skills:
            # Check minimum requirements
            if (skill.frequency >= self.min_skill_frequency and
                skill.success_rate >= self.min_success_rate and
                skill.transferability_score > 0.2):
                filtered.append(skill)
        
        # Sort by quality
        filtered.sort(key=lambda s: s.transferability_score * s.success_rate, reverse=True)
        
        return filtered
    
    def _build_skill_graph(self, skills: List[DiscoveredSkill]) -> None:
        """Builds graph connections between skills"""
        self.skill_graph.clear()
        
        # Add nodes
        for skill in skills:
            self.skill_graph.add_node(
                skill.skill_id,
                pattern_type=skill.pattern_type.value,
                transferability=skill.transferability_score
            )
        
        # Add edges on basis similarity actions
        for i, skill1 in enumerate(skills):
            for j, skill2 in enumerate(skills):
                if i >= j:
                    continue
                
                similarity = self._compute_skill_similarity(skill1, skill2)
                
                # Add edge if skills similar
                if similarity > 0.5:
                    self.skill_graph.add_edge(
                        skill1.skill_id,
                        skill2.skill_id,
                        similarity=similarity
                    )
    
    def _compute_skill_similarity(self, 
                                 skill1: DiscoveredSkill, 
                                 skill2: DiscoveredSkill) -> float:
        """Computes similarity between skills"""
        # Similarity types patterns
        type_similarity = 1.0 if skill1.pattern_type == skill2.pattern_type else 0.3
        
        # Similarity sequences actions
        action_similarity = self._compute_action_sequence_similarity(
            skill1.action_sequence, skill2.action_sequence
        )
        
        # Similarity duration
        duration_diff = abs(skill1.duration - skill2.duration)
        duration_similarity = 1.0 / (1.0 + duration_diff / max(skill1.duration, skill2.duration, 1))
        
        # Combined similarity
        total_similarity = (type_similarity * 0.4 + 
                           action_similarity * 0.4 + 
                           duration_similarity * 0.2)
        
        return total_similarity
    
    def _compute_action_sequence_similarity(self, 
                                          seq1: List[np.ndarray], 
                                          seq2: List[np.ndarray]) -> float:
        """Computes similarity sequences actions"""
        if not seq1 or not seq2:
            return 0.0
        
        # Use Dynamic Time Warping for comparison sequences
        return self._dtw_similarity(seq1, seq2)
    
    def _dtw_similarity(self, seq1: List[np.ndarray], seq2: List[np.ndarray]) -> float:
        """Simplified DTW for comparison sequences"""
        n, m = len(seq1), len(seq2)
        
        if n == 0 or m == 0:
            return 0.0
        
        # Matrix distances
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Normalize and convert in similarity
        max_len = max(n, m)
        normalized_distance = dtw_matrix[n, m] / max_len
        similarity = 1.0 / (1.0 + normalized_distance)
        
        return similarity
    
    def get_skill_composition_graph(self) -> nx.Graph:
        """Returns graph composition skills"""
        return self.skill_graph.copy()
    
    def save_skills(self, filepath: str) -> None:
        """Saves detected skills"""
        data = {
            'skills': [
                {
                    'skill_id': skill.skill_id,
                    'pattern_type': skill.pattern_type.value,
                    'action_sequence': [action.tolist() for action in skill.action_sequence],
                    'state_conditions': [state.tolist() for state in skill.state_conditions],
                    'success_rate': skill.success_rate,
                    'frequency': skill.frequency,
                    'avg_reward': skill.avg_reward,
                    'duration': skill.duration,
                    'transferability_score': skill.transferability_score,
                    'metadata': skill.metadata
                }
                for skill in self.discovered_skills
            ],
            'skill_graph': nx.node_link_data(self.skill_graph),
            'discovery_params': {
                'action_dim': self.action_dim,
                'state_dim': self.state_dim,
                'min_skill_frequency': self.min_skill_frequency,
                'min_success_rate': self.min_success_rate
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Skills saved in {filepath}")
    
    def load_skills(self, filepath: str) -> None:
        """Loads detected skills"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore skills
        self.discovered_skills = []
        for skill_data in data['skills']:
            skill = DiscoveredSkill(
                skill_id=skill_data['skill_id'],
                pattern_type=SkillPattern(skill_data['pattern_type']),
                action_sequence=[np.array(action) for action in skill_data['action_sequence']],
                state_conditions=[np.array(state) for state in skill_data['state_conditions']],
                success_rate=skill_data['success_rate'],
                frequency=skill_data['frequency'],
                avg_reward=skill_data['avg_reward'],
                duration=skill_data['duration'],
                transferability_score=skill_data['transferability_score'],
                metadata=skill_data['metadata']
            )
            self.discovered_skills.append(skill)
        
        # Restore graph
        self.skill_graph = nx.node_link_graph(data['skill_graph'])
        
        logger.info(f"Loaded {len(self.discovered_skills)} skills")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics detection skills"""
        if not self.discovered_skills:
            return {}
        
        pattern_counts = Counter([skill.pattern_type for skill in self.discovered_skills])
        
        return {
            'total_skills': len(self.discovered_skills),
            'skills_by_pattern': {k.value: v for k, v in pattern_counts.items()},
            'avg_success_rate': np.mean([skill.success_rate for skill in self.discovered_skills]),
            'avg_transferability': np.mean([skill.transferability_score for skill in self.discovered_skills]),
            'avg_frequency': np.mean([skill.frequency for skill in self.discovered_skills]),
            'skill_graph_nodes': len(self.skill_graph.nodes()),
            'skill_graph_edges': len(self.skill_graph.edges())
        }


# Factory for creation skill discovery engine

def create_crypto_skill_discovery(action_dim: int = 3, state_dim: int = 15) -> SkillDiscoveryEngine:
    """Creates engine detection skills for crypto trading"""
    return SkillDiscoveryEngine(
        action_dim=action_dim,
        state_dim=state_dim,
        min_skill_frequency=3,
        min_success_rate=0.3
    )