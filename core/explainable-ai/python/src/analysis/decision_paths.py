"""
Decision Path Analysis Crypto Trading Bot v5.0

 comprehensive analysis
 support tree-based models enterprise patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import graphviz
from collections import defaultdict, deque

# Advanced tree analysis imports
try:
    from sklearn.tree import _tree
    from sklearn.inspection import partial_dependence
    ADVANCED_TREE_AVAILABLE = True
except ImportError:
    ADVANCED_TREE_AVAILABLE = False

# XGBoost and LightGBM support
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DecisionPathConfig:
    """Configuration decision path analysis enterprise patterns"""
    max_depth_analysis: int = 10
    min_samples_leaf_analysis: int = 5
    max_paths_per_tree: int = 100
    path_similarity_threshold: float = 0.8
    # Tree visualization
    visualize_trees: bool = True
    max_tree_visualization_depth: int = 5
    # Path analysis
    analyze_feature_interactions: bool = True
    identify_decision_rules: bool = True
    # Enterprise performance
    parallel_workers: int = 4
    cache_results: bool = True
    # Trading specific
    focus_on_trading_decisions: bool = True
    min_path_confidence: float = 0.7
    analyze_risk_paths: bool = True


@dataclass
class DecisionRule:
    """Single decision rule trading metadata"""
    conditions: List[str]  # List of conditions in the path
    prediction: Union[int, float]
    confidence: float
    support: int  # Number of samples following this path
    feature_path: List[str]  # Features used in the path
    path_id: str
    # Trading specific
    trading_signal: Optional[str] = None
    risk_level: Optional[str] = None
    market_conditions: Optional[List[str]] = None
    profitability_estimate: Optional[float] = None
    # Path statistics
    path_length: int = 0
    average_feature_importance: float = 0.0


@dataclass
class DecisionPathResult:
    """ analysis metadata"""
    decision_rules: List[DecisionRule]
    path_statistics: Dict[str, Any]
    feature_interaction_analysis: Dict[str, Any]
    tree_structure_analysis: Dict[str, Any]
    model_type: str
    timestamp: datetime = datetime.now()
    # Trading specific analysis
    trading_decision_analysis: Optional[Dict[str, Any]] = None
    risk_path_analysis: Optional[Dict[str, Any]] = None
    # Model interpretability metrics
    model_complexity_score: float = 0.0
    decision_boundary_clarity: float = 0.0
    rule_consistency_score: float = 0.0
    #  enterprise metadata
    total_paths_analyzed: int = 0
    significant_paths: int = 0
    validation_metrics: Optional[Dict[str, float]] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingDecisionPathAnalyzer:
    """
    Enterprise-grade decision path analyzer crypto trading models
    
    Provides comprehensive decision path analysis :
    - Tree-based trading models (RF, XGBoost, LightGBM)
    - Decision rule extraction
    - Trading strategy interpretation
    - Risk path identification
    - Model complexity analysis
    
    enterprise patterns:
    - Tree-agnostic path extraction
    - Trading-aware rule interpretation
    - Enterprise rule validation
    - Async processing large ensembles
    - Regulatory compliance tracking
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        config: Optional[DecisionPathConfig] = None,
        feature_names: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize decision path analyzer"""
        self.model = model
        self.config = config or DecisionPathConfig()
        self.feature_names = feature_names or []
        self.cache_dir = cache_dir or Path("./cache/decision_paths")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model type detection
        self.model_type = self._detect_model_type()
        
        # Trading feature categorization
        self.trading_feature_categories = self._init_trading_categories()
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"Initialized decision path analyzer for {self.model_type} model")
    
    def _detect_model_type(self) -> str:
        """Detect the type of tree-based model"""
        model_name = type(self.model).__name__.lower()
        
        if 'randomforest' in model_name:
            return 'random_forest'
        elif 'decisiontree' in model_name:
            return 'decision_tree'
        elif 'xgb' in model_name or 'xgboost' in model_name:
            return 'xgboost'
        elif 'lgb' in model_name or 'lightgbm' in model_name:
            return 'lightgbm'
        elif 'gradientboosting' in model_name:
            return 'gradient_boosting'
        else:
            return 'unknown'
    
    def _init_trading_categories(self) -> Dict[str, List[str]]:
        """Initialize trading-specific feature categories"""
        return {
            'price_signals': ['price', 'close', 'open', 'high', 'low', 'sma', 'ema'],
            'momentum_indicators': ['rsi', 'macd', 'momentum', 'roc', 'stoch'],
            'volume_indicators': ['volume', 'vol', 'obv', 'ad_line', 'mfi'],
            'volatility_indicators': ['volatility', 'atr', 'bb', 'std', 'vix'],
            'trend_indicators': ['trend', 'adx', 'cci', 'dmi', 'aroon'],
            'risk_indicators': ['var', 'drawdown', 'sharpe', 'sortino', 'calmar']
        }
    
    async def analyze_async(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> DecisionPathResult:
        """
        Async decision path analysis large models
        
        Args:
            X: Feature matrix path analysis
            y: Target variable (optional, validation)
            feature_names: Feature names
            
        Returns:
            Comprehensive decision path analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.analyze,
            X,
            y,
            feature_names
        )
    
    def analyze(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_names: Optional[List[str]] = None
    ) -> DecisionPathResult:
        """
        Comprehensive decision path analysis
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            feature_names: Feature names
            
        Returns:
            Decision path analysis results
        """
        try:
            # Prepare data
            if isinstance(X, pd.DataFrame):
                self.feature_names = list(X.columns)
                X_array = X.values
            else:
                X_array = X
                self.feature_names = feature_names or self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y_array = y.values if y is not None else None
            else:
                y_array = y
            
            logger.info(f"Analyzing decision paths for {self.model_type} model")
            
            # Extract decision rules from model
            decision_rules = self._extract_decision_rules(X_array, y_array)
            
            # Analyze path statistics
            path_statistics = self._analyze_path_statistics(decision_rules)
            
            # Feature interaction analysis
            interaction_analysis = self._analyze_feature_interactions(decision_rules) if self.config.analyze_feature_interactions else {}
            
            # Tree structure analysis
            structure_analysis = self._analyze_tree_structure()
            
            # Trading specific analysis
            trading_analysis = self._analyze_trading_decisions(decision_rules, X_array) if self.config.focus_on_trading_decisions else None
            
            # Risk path analysis
            risk_analysis = self._analyze_risk_paths(decision_rules) if self.config.analyze_risk_paths else None
            
            # Calculate model complexity metrics
            complexity_metrics = self._calculate_complexity_metrics(decision_rules)
            
            # Validation metrics
            validation_metrics = self._calculate_validation_metrics(decision_rules, X_array, y_array) if y_array is not None else None
            
            # Create comprehensive result
            result = DecisionPathResult(
                decision_rules=decision_rules,
                path_statistics=path_statistics,
                feature_interaction_analysis=interaction_analysis,
                tree_structure_analysis=structure_analysis,
                model_type=self.model_type,
                timestamp=datetime.now(),
                trading_decision_analysis=trading_analysis,
                risk_path_analysis=risk_analysis,
                total_paths_analyzed=len(decision_rules),
                significant_paths=len([r for r in decision_rules if r.confidence >= self.config.min_path_confidence]),
                validation_metrics=validation_metrics,
                **complexity_metrics
            )
            
            logger.info(f"Decision path analysis completed. Found {len(decision_rules)} decision rules.")
            return result
            
        except Exception as e:
            logger.error(f"Error in decision path analysis: {e}")
            raise
    
    def _extract_decision_rules(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract decision rules from the model"""
        
        if self.model_type == 'decision_tree':
            return self._extract_from_single_tree(self.model, X, y)
        elif self.model_type == 'random_forest':
            return self._extract_from_random_forest(X, y)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            return self._extract_from_xgboost(X, y)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            return self._extract_from_lightgbm(X, y)
        elif self.model_type == 'gradient_boosting':
            return self._extract_from_gradient_boosting(X, y)
        else:
            # Fallback: train surrogate decision tree
            return self._extract_from_surrogate_tree(X, y)
    
    def _extract_from_single_tree(
        self,
        tree_model: BaseEstimator,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from single decision tree"""
        try:
            decision_rules = []
            tree = tree_model.tree_
            
            def extract_paths(node_id: int, path_conditions: List[str], path_features: List[str]):
                # Check if leaf node
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # Leaf node - create decision rule
                    if tree.n_outputs == 1:
                        prediction = tree.value[node_id][0][0] if tree.n_classes[0] > 1 else tree.value[node_id][0][0]
                    else:
                        prediction = np.argmax(tree.value[node_id][0])
                    
                    samples = tree.n_node_samples[node_id]
                    confidence = samples / tree.n_node_samples[0]  # Relative to root
                    
                    # Create trading signal
                    trading_signal = self._determine_trading_signal(prediction, tree.n_classes[0] if hasattr(tree, 'n_classes') else None)
                    
                    # Generate path ID
                    path_id = f"tree_0_path_{len(decision_rules)}"
                    
                    rule = DecisionRule(
                        conditions=path_conditions.copy(),
                        prediction=float(prediction),
                        confidence=float(confidence),
                        support=int(samples),
                        feature_path=path_features.copy(),
                        path_id=path_id,
                        trading_signal=trading_signal,
                        path_length=len(path_conditions),
                        average_feature_importance=0.0  # Will be calculated later
                    )
                    
                    # Enrich with trading metadata
                    self._enrich_rule_with_trading_info(rule)
                    
                    decision_rules.append(rule)
                    return
                
                # Internal node - continue traversal
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                
                # Left child (<=)
                left_condition = f"{feature_name} <= {threshold:.4f}"
                extract_paths(
                    tree.children_left[node_id],
                    path_conditions + [left_condition],
                    path_features + [feature_name]
                )
                
                # Right child (>)
                right_condition = f"{feature_name} > {threshold:.4f}"
                extract_paths(
                    tree.children_right[node_id],
                    path_conditions + [right_condition],
                    path_features + [feature_name]
                )
            
            # Start extraction from root
            extract_paths(0, [], [])
            
            return decision_rules[:self.config.max_paths_per_tree]
            
        except Exception as e:
            logger.warning(f"Error extracting from single tree: {e}")
            return []
    
    def _extract_from_random_forest(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from Random Forest"""
        try:
            all_rules = []
            
            # Extract from subset of trees for performance
            n_trees_to_analyze = min(10, len(self.model.estimators_))
            
            for tree_idx in range(n_trees_to_analyze):
                tree = self.model.estimators_[tree_idx]
                tree_rules = self._extract_from_single_tree(tree, X, y)
                
                # Update path IDs to include tree index
                for rule in tree_rules:
                    rule.path_id = f"tree_{tree_idx}_" + rule.path_id.split('_', 2)[-1]
                
                all_rules.extend(tree_rules)
            
            # Sort by confidence and support
            all_rules.sort(key=lambda r: (r.confidence, r.support), reverse=True)
            
            return all_rules[:self.config.max_paths_per_tree]
            
        except Exception as e:
            logger.warning(f"Error extracting from Random Forest: {e}")
            return []
    
    def _extract_from_xgboost(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from XGBoost model"""
        try:
            decision_rules = []
            
            # Get booster trees
            if hasattr(self.model, 'get_booster'):
                booster = self.model.get_booster()
                
                # Get tree dump
                tree_dump = booster.get_dump()
                
                for tree_idx, tree_str in enumerate(tree_dump[:10]):  # Analyze first 10 trees
                    tree_rules = self._parse_xgboost_tree_string(tree_str, tree_idx)
                    decision_rules.extend(tree_rules)
            
            return decision_rules[:self.config.max_paths_per_tree]
            
        except Exception as e:
            logger.warning(f"Error extracting from XGBoost: {e}")
            return self._extract_from_surrogate_tree(X, y)
    
    def _extract_from_lightgbm(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from LightGBM model"""
        try:
            decision_rules = []
            
            # Get tree information
            if hasattr(self.model, 'booster_'):
                tree_info = self.model.booster_.dump_model()
                
                for tree_idx, tree_data in enumerate(tree_info['tree_info'][:10]):  # First 10 trees
                    tree_rules = self._parse_lightgbm_tree_data(tree_data, tree_idx)
                    decision_rules.extend(tree_rules)
            
            return decision_rules[:self.config.max_paths_per_tree]
            
        except Exception as e:
            logger.warning(f"Error extracting from LightGBM: {e}")
            return self._extract_from_surrogate_tree(X, y)
    
    def _extract_from_gradient_boosting(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from Gradient Boosting"""
        try:
            decision_rules = []
            
            # Extract from subset of estimators
            n_estimators_to_analyze = min(10, len(self.model.estimators_))
            
            for estimator_idx in range(n_estimators_to_analyze):
                if hasattr(self.model.estimators_[estimator_idx], '__len__'):
                    # Multi-output estimator
                    for tree_idx, tree in enumerate(self.model.estimators_[estimator_idx]):
                        tree_rules = self._extract_from_single_tree(tree, X, y)
                        for rule in tree_rules:
                            rule.path_id = f"estimator_{estimator_idx}_tree_{tree_idx}_" + rule.path_id.split('_', 2)[-1]
                        decision_rules.extend(tree_rules)
                else:
                    # Single tree estimator
                    tree = self.model.estimators_[estimator_idx]
                    tree_rules = self._extract_from_single_tree(tree, X, y)
                    for rule in tree_rules:
                        rule.path_id = f"estimator_{estimator_idx}_" + rule.path_id
                    decision_rules.extend(tree_rules)
            
            return decision_rules[:self.config.max_paths_per_tree]
            
        except Exception as e:
            logger.warning(f"Error extracting from Gradient Boosting: {e}")
            return []
    
    def _extract_from_surrogate_tree(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> List[DecisionRule]:
        """Extract rules from surrogate decision tree"""
        try:
            logger.info("Using surrogate decision tree for path extraction")
            
            # Generate predictions from original model
            if y is None:
                if hasattr(self.model, 'predict'):
                    y_surrogate = self.model.predict(X)
                else:
                    logger.error("Cannot generate surrogate targets")
                    return []
            else:
                y_surrogate = y
            
            # Train surrogate tree
            if len(np.unique(y_surrogate)) <= 20:  # Classification
                surrogate_tree = DecisionTreeClassifier(
                    max_depth=self.config.max_depth_analysis,
                    min_samples_leaf=self.config.min_samples_leaf_analysis,
                    random_state=42
                )
            else:  # Regression
                surrogate_tree = DecisionTreeRegressor(
                    max_depth=self.config.max_depth_analysis,
                    min_samples_leaf=self.config.min_samples_leaf_analysis,
                    random_state=42
                )
            
            surrogate_tree.fit(X, y_surrogate)
            
            # Extract rules from surrogate
            return self._extract_from_single_tree(surrogate_tree, X, y_surrogate)
            
        except Exception as e:
            logger.error(f"Error extracting from surrogate tree: {e}")
            return []
    
    def _parse_xgboost_tree_string(self, tree_str: str, tree_idx: int) -> List[DecisionRule]:
        """Parse XGBoost tree string to extract decision rules"""
        # Simplified XGBoost tree parsing
        # This would need more sophisticated parsing for production use
        try:
            rules = []
            lines = tree_str.split('\n')
            
            # Simple parsing - would need enhancement for full XGBoost support
            for i, line in enumerate(lines):
                if 'leaf=' in line:
                    # Extract leaf value
                    leaf_value = float(line.split('leaf=')[1])
                    
                    # Create simplified rule
                    rule = DecisionRule(
                        conditions=[f"XGBoost tree {tree_idx} path {i}"],
                        prediction=leaf_value,
                        confidence=0.5,  # Simplified
                        support=10,  # Simplified
                        feature_path=[],
                        path_id=f"xgb_tree_{tree_idx}_leaf_{i}",
                        path_length=1
                    )
                    
                    rules.append(rule)
            
            return rules[:5] # Limit performance
            
        except Exception as e:
            logger.warning(f"Error parsing XGBoost tree: {e}")
            return []
    
    def _parse_lightgbm_tree_data(self, tree_data: Dict, tree_idx: int) -> List[DecisionRule]:
        """Parse LightGBM tree data to extract decision rules"""
        try:
            rules = []
            
            def traverse_lgb_tree(node_data: Dict, path_conditions: List[str], path_features: List[str]):
                if 'leaf_value' in node_data:
                    # Leaf node
                    leaf_value = node_data['leaf_value']
                    
                    rule = DecisionRule(
                        conditions=path_conditions.copy(),
                        prediction=float(leaf_value),
                        confidence=0.5,  # Simplified
                        support=node_data.get('leaf_count', 10),
                        feature_path=path_features.copy(),
                        path_id=f"lgb_tree_{tree_idx}_path_{len(rules)}",
                        path_length=len(path_conditions)
                    )
                    
                    rules.append(rule)
                    return
                
                # Internal node
                if 'split_feature' in node_data:
                    feature_idx = node_data['split_feature']
                    threshold = node_data['threshold']
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                    
                    # Left child
                    if 'left_child' in node_data:
                        left_condition = f"{feature_name} <= {threshold:.4f}"
                        traverse_lgb_tree(
                            node_data['left_child'],
                            path_conditions + [left_condition],
                            path_features + [feature_name]
                        )
                    
                    # Right child
                    if 'right_child' in node_data:
                        right_condition = f"{feature_name} > {threshold:.4f}"
                        traverse_lgb_tree(
                            node_data['right_child'],
                            path_conditions + [right_condition],
                            path_features + [feature_name]
                        )
            
            # Start traversal
            traverse_lgb_tree(tree_data['tree_structure'], [], [])
            
            return rules[:10] # Limit performance
            
        except Exception as e:
            logger.warning(f"Error parsing LightGBM tree: {e}")
            return []
    
    def _determine_trading_signal(
        self,
        prediction: Union[int, float],
        n_classes: Optional[int] = None
    ) -> str:
        """Determine trading signal from prediction"""
        if n_classes is not None:
            # Classification
            if n_classes == 2:
                return "BUY" if prediction >= 0.5 else "SELL"
            elif n_classes == 3:
                if prediction == 0:
                    return "SELL"
                elif prediction == 1:
                    return "HOLD"
                else:
                    return "BUY"
            else:
                return f"CLASS_{int(prediction)}"
        else:
            # Regression
            if prediction > 0.1:
                return "BUY"
            elif prediction < -0.1:
                return "SELL"
            else:
                return "HOLD"
    
    def _enrich_rule_with_trading_info(self, rule: DecisionRule) -> None:
        """Enrich decision rule with trading-specific information"""
        try:
            # Determine risk level based on path characteristics
            if rule.confidence >= 0.8 and rule.support >= 100:
                rule.risk_level = "LOW"
            elif rule.confidence >= 0.6 and rule.support >= 50:
                rule.risk_level = "MEDIUM"
            else:
                rule.risk_level = "HIGH"
            
            # Analyze market conditions from features in path
            market_conditions = []
            for feature in rule.feature_path:
                feature_lower = feature.lower()
                
                if any(indicator in feature_lower for indicator in ['rsi', 'macd', 'momentum']):
                    market_conditions.append("MOMENTUM_DRIVEN")
                elif any(indicator in feature_lower for indicator in ['volume', 'vol']):
                    market_conditions.append("VOLUME_DRIVEN")
                elif any(indicator in feature_lower for indicator in ['volatility', 'atr']):
                    market_conditions.append("VOLATILITY_DRIVEN")
                elif any(indicator in feature_lower for indicator in ['sma', 'ema', 'trend']):
                    market_conditions.append("TREND_FOLLOWING")
            
            rule.market_conditions = list(set(market_conditions)) if market_conditions else ["GENERAL"]
            
            # Estimate profitability (simplified)
            base_profit = 0.5
            confidence_bonus = (rule.confidence - 0.5) * 0.4
            support_bonus = min(rule.support / 1000, 0.2)
            
            rule.profitability_estimate = max(0.1, base_profit + confidence_bonus + support_bonus)
            
        except Exception as e:
            logger.warning(f"Error enriching rule with trading info: {e}")
    
    def _analyze_path_statistics(self, decision_rules: List[DecisionRule]) -> Dict[str, Any]:
        """Analyze statistics of decision paths"""
        try:
            if not decision_rules:
                return {}
            
            statistics = {
                'total_rules': len(decision_rules),
                'average_path_length': float(np.mean([rule.path_length for rule in decision_rules])),
                'average_confidence': float(np.mean([rule.confidence for rule in decision_rules])),
                'average_support': float(np.mean([rule.support for rule in decision_rules])),
                'max_path_length': max(rule.path_length for rule in decision_rules),
                'min_path_length': min(rule.path_length for rule in decision_rules),
                'confidence_distribution': {},
                'support_distribution': {},
                'path_length_distribution': {}
            }
            
            # Confidence distribution
            confidence_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]
            confidence_hist, _ = np.histogram([rule.confidence for rule in decision_rules], bins=confidence_bins)
            statistics['confidence_distribution'] = {
                f"{confidence_bins[i]:.1f}-{confidence_bins[i+1]:.1f}": int(count)
                for i, count in enumerate(confidence_hist)
            }
            
            # Support distribution
            support_values = [rule.support for rule in decision_rules]
            if support_values:
                support_percentiles = np.percentile(support_values, [25, 50, 75, 90])
                statistics['support_distribution'] = {
                    'q25': float(support_percentiles[0]),
                    'median': float(support_percentiles[1]),
                    'q75': float(support_percentiles[2]),
                    'q90': float(support_percentiles[3])
                }
            
            # Path length distribution
            length_counts = defaultdict(int)
            for rule in decision_rules:
                length_counts[rule.path_length] += 1
            statistics['path_length_distribution'] = dict(length_counts)
            
            return statistics
            
        except Exception as e:
            logger.warning(f"Error analyzing path statistics: {e}")
            return {}
    
    def _analyze_feature_interactions(self, decision_rules: List[DecisionRule]) -> Dict[str, Any]:
        """Analyze feature interactions in decision paths"""
        try:
            interaction_analysis = {
                'feature_co_occurrence': {},
                'feature_usage_frequency': {},
                'interaction_patterns': {},
                'critical_feature_pairs': []
            }
            
            # Feature usage frequency
            feature_counts = defaultdict(int)
            for rule in decision_rules:
                for feature in rule.feature_path:
                    feature_counts[feature] += 1
            
            total_rules = len(decision_rules)
            interaction_analysis['feature_usage_frequency'] = {
                feature: {'count': count, 'frequency': count / total_rules}
                for feature, count in feature_counts.items()
            }
            
            # Feature co-occurrence analysis
            feature_pairs = defaultdict(int)
            for rule in decision_rules:
                features = rule.feature_path
                for i in range(len(features)):
                    for j in range(i + 1, len(features)):
                        pair = tuple(sorted([features[i], features[j]]))
                        feature_pairs[pair] += 1
            
            # Convert to co-occurrence probabilities
            co_occurrence = {}
            for (feat1, feat2), count in feature_pairs.items():
                co_occurrence[f"{feat1}_{feat2}"] = {
                    'count': count,
                    'probability': count / total_rules,
                    'lift': count / (feature_counts[feat1] * feature_counts[feat2] / total_rules)
                }
            
            # Sort by lift (interestingness measure)
            sorted_pairs = sorted(co_occurrence.items(), key=lambda x: x[1]['lift'], reverse=True)
            interaction_analysis['feature_co_occurrence'] = dict(sorted_pairs[:20])
            
            # Critical feature pairs (appear together frequently with high confidence)
            critical_pairs = []
            for rule in decision_rules:
                if rule.confidence >= 0.8 and len(rule.feature_path) >= 2:
                    for i in range(len(rule.feature_path) - 1):
                        pair = (rule.feature_path[i], rule.feature_path[i + 1])
                        critical_pairs.append({
                            'features': pair,
                            'rule_confidence': rule.confidence,
                            'trading_signal': rule.trading_signal
                        })
            
            interaction_analysis['critical_feature_pairs'] = critical_pairs[:10]
            
            return interaction_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing feature interactions: {e}")
            return {}
    
    def _analyze_tree_structure(self) -> Dict[str, Any]:
        """Analyze overall tree structure characteristics"""
        try:
            structure_analysis = {
                'model_type': self.model_type,
                'estimated_complexity': 0,
                'tree_characteristics': {}
            }
            
            if hasattr(self.model, 'tree_'):
                # Single tree
                tree = self.model.tree_
                structure_analysis['tree_characteristics'] = {
                    'max_depth': int(tree.max_depth),
                    'node_count': int(tree.node_count),
                    'leaf_count': int(np.sum(tree.children_left == tree.children_right)),
                    'feature_count': len(np.unique(tree.feature[tree.feature >= 0]))
                }
                structure_analysis['estimated_complexity'] = tree.node_count / (tree.max_depth + 1)
                
            elif hasattr(self.model, 'estimators_'):
                # Ensemble
                if hasattr(self.model.estimators_[0], 'tree_'):
                    # Forest of trees
                    depths = []
                    node_counts = []
                    leaf_counts = []
                    
                    for estimator in self.model.estimators_[:10]:  # Analyze first 10
                        tree = estimator.tree_
                        depths.append(tree.max_depth)
                        node_counts.append(tree.node_count)
                        leaf_counts.append(np.sum(tree.children_left == tree.children_right))
                    
                    structure_analysis['tree_characteristics'] = {
                        'average_depth': float(np.mean(depths)),
                        'average_nodes': float(np.mean(node_counts)),
                        'average_leaves': float(np.mean(leaf_counts)),
                        'total_estimators': len(self.model.estimators_)
                    }
                    structure_analysis['estimated_complexity'] = np.mean(node_counts) * len(self.model.estimators_)
            
            return structure_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing tree structure: {e}")
            return {'model_type': self.model_type}
    
    def _analyze_trading_decisions(
        self,
        decision_rules: List[DecisionRule],
        X: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze trading-specific decision patterns"""
        try:
            trading_analysis = {
                'signal_distribution': {},
                'risk_level_distribution': {},
                'market_condition_analysis': {},
                'profitability_analysis': {},
                'feature_category_usage': {}
            }
            
            # Signal distribution
            signal_counts = defaultdict(int)
            for rule in decision_rules:
                if rule.trading_signal:
                    signal_counts[rule.trading_signal] += 1
            trading_analysis['signal_distribution'] = dict(signal_counts)
            
            # Risk level distribution
            risk_counts = defaultdict(int)
            for rule in decision_rules:
                if rule.risk_level:
                    risk_counts[rule.risk_level] += 1
            trading_analysis['risk_level_distribution'] = dict(risk_counts)
            
            # Market conditions analysis
            condition_counts = defaultdict(int)
            for rule in decision_rules:
                if rule.market_conditions:
                    for condition in rule.market_conditions:
                        condition_counts[condition] += 1
            trading_analysis['market_condition_analysis'] = dict(condition_counts)
            
            # Profitability analysis
            profit_estimates = [rule.profitability_estimate for rule in decision_rules if rule.profitability_estimate]
            if profit_estimates:
                trading_analysis['profitability_analysis'] = {
                    'mean_profitability': float(np.mean(profit_estimates)),
                    'std_profitability': float(np.std(profit_estimates)),
                    'high_profit_rules': len([p for p in profit_estimates if p > 0.7]),
                    'low_risk_high_profit': len([
                        rule for rule in decision_rules
                        if rule.risk_level == "LOW" and rule.profitability_estimate and rule.profitability_estimate > 0.7
                    ])
                }
            
            # Feature category usage
            category_usage = defaultdict(int)
            for rule in decision_rules:
                for feature in rule.feature_path:
                    for category, keywords in self.trading_feature_categories.items():
                        if any(keyword in feature.lower() for keyword in keywords):
                            category_usage[category] += 1
                            break
            
            trading_analysis['feature_category_usage'] = dict(category_usage)
            
            return trading_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing trading decisions: {e}")
            return {}
    
    def _analyze_risk_paths(self, decision_rules: List[DecisionRule]) -> Dict[str, Any]:
        """Analyze risk-related decision paths"""
        try:
            risk_analysis = {
                'high_risk_paths': [],
                'risk_factors': {},
                'risk_mitigation_rules': [],
                'risk_concentration': {}
            }
            
            # Identify high-risk paths
            high_risk_rules = [rule for rule in decision_rules if rule.risk_level == "HIGH"]
            risk_analysis['high_risk_paths'] = [
                {
                    'path_id': rule.path_id,
                    'conditions': rule.conditions[:3],  # First 3 conditions
                    'confidence': rule.confidence,
                    'trading_signal': rule.trading_signal,
                    'profitability_estimate': rule.profitability_estimate
                }
                for rule in high_risk_rules[:10]
            ]
            
            # Analyze risk factors (features appearing in high-risk paths)
            risk_factor_counts = defaultdict(int)
            for rule in high_risk_rules:
                for feature in rule.feature_path:
                    risk_factor_counts[feature] += 1
            
            # Sort by frequency
            sorted_risk_factors = sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)
            risk_analysis['risk_factors'] = {
                factor: {'frequency': count, 'risk_contribution': count / len(high_risk_rules)}
                for factor, count in sorted_risk_factors[:10]
            }
            
            # Risk mitigation rules (low risk, high confidence)
            mitigation_rules = [
                rule for rule in decision_rules
                if rule.risk_level == "LOW" and rule.confidence >= 0.8
            ]
            
            risk_analysis['risk_mitigation_rules'] = [
                {
                    'path_id': rule.path_id,
                    'conditions': rule.conditions[:3],
                    'confidence': rule.confidence,
                    'support': rule.support,
                    'trading_signal': rule.trading_signal
                }
                for rule in mitigation_rules[:10]
            ]
            
            # Risk concentration by trading signal
            signal_risk = defaultdict(lambda: {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0})
            for rule in decision_rules:
                if rule.trading_signal and rule.risk_level:
                    signal_risk[rule.trading_signal][rule.risk_level] += 1
            
            risk_analysis['risk_concentration'] = dict(signal_risk)
            
            return risk_analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing risk paths: {e}")
            return {}
    
    def _calculate_complexity_metrics(self, decision_rules: List[DecisionRule]) -> Dict[str, float]:
        """Calculate model complexity and interpretability metrics"""
        try:
            if not decision_rules:
                return {
                    'model_complexity_score': 0.0,
                    'decision_boundary_clarity': 0.0,
                    'rule_consistency_score': 0.0
                }
            
            # Model complexity (normalized by number of features and rules)
            avg_path_length = np.mean([rule.path_length for rule in decision_rules])
            total_rules = len(decision_rules)
            complexity_score = 1.0 / (1.0 + (avg_path_length * total_rules) / 100)
            
            # Decision boundary clarity (based on confidence distribution)
            confidences = [rule.confidence for rule in decision_rules]
            high_confidence_ratio = len([c for c in confidences if c >= 0.8]) / len(confidences)
            clarity_score = high_confidence_ratio
            
            # Rule consistency (similar rules should have similar predictions)
            consistency_score = self._calculate_rule_consistency(decision_rules)
            
            return {
                'model_complexity_score': float(complexity_score),
                'decision_boundary_clarity': float(clarity_score),
                'rule_consistency_score': float(consistency_score)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating complexity metrics: {e}")
            return {
                'model_complexity_score': 0.0,
                'decision_boundary_clarity': 0.0,
                'rule_consistency_score': 0.0
            }
    
    def _calculate_rule_consistency(self, decision_rules: List[DecisionRule]) -> float:
        """Calculate consistency score for decision rules"""
        try:
            if len(decision_rules) < 2:
                return 1.0
            
            # Group rules by similar feature sets
            feature_groups = defaultdict(list)
            for rule in decision_rules:
                feature_signature = tuple(sorted(rule.feature_path))
                feature_groups[feature_signature].append(rule)
            
            # Calculate consistency within each group
            consistency_scores = []
            for group_rules in feature_groups.values():
                if len(group_rules) < 2:
                    consistency_scores.append(1.0)
                    continue
                
                # Compare predictions within the group
                predictions = [rule.prediction for rule in group_rules]
                if all(isinstance(p, (int, np.integer)) for p in predictions):
                    # Classification: fraction of rules with same prediction
                    most_common = max(set(predictions), key=predictions.count)
                    consistency = predictions.count(most_common) / len(predictions)
                else:
                    # Regression: inverse of coefficient of variation
                    if np.std(predictions) == 0:
                        consistency = 1.0
                    else:
                        cv = np.std(predictions) / (np.mean(predictions) if np.mean(predictions) != 0 else 1)
                        consistency = 1.0 / (1.0 + cv)
                
                consistency_scores.append(consistency)
            
            return float(np.mean(consistency_scores))
            
        except Exception as e:
            logger.warning(f"Error calculating rule consistency: {e}")
            return 0.5
    
    def _calculate_validation_metrics(
        self,
        decision_rules: List[DecisionRule],
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Calculate validation metrics for decision paths"""
        try:
            validation_metrics = {}
            
            # Rule coverage (fraction of samples covered by high-confidence rules)
            high_conf_rules = [rule for rule in decision_rules if rule.confidence >= 0.8]
            total_support = sum(rule.support for rule in high_conf_rules)
            coverage = total_support / len(X) if len(X) > 0 else 0
            
            validation_metrics['high_confidence_coverage'] = float(coverage)
            
            # Rule precision (weighted by support)
            weighted_confidence = sum(rule.confidence * rule.support for rule in decision_rules)
            total_weighted_support = sum(rule.support for rule in decision_rules)
            avg_weighted_confidence = weighted_confidence / total_weighted_support if total_weighted_support > 0 else 0
            
            validation_metrics['weighted_average_confidence'] = float(avg_weighted_confidence)
            
            # Feature utilization (fraction of features used in rules)
            used_features = set()
            for rule in decision_rules:
                used_features.update(rule.feature_path)
            
            feature_utilization = len(used_features) / len(self.feature_names) if self.feature_names else 0
            validation_metrics['feature_utilization'] = float(feature_utilization)
            
            return validation_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating validation metrics: {e}")
            return {}
    
    def create_decision_tree_visualization(
        self,
        result: DecisionPathResult,
        max_rules: int = 10,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create decision path visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Decision Path Analysis', fontsize=16)
            
            # 1. Path length distribution
            if result.decision_rules:
                path_lengths = [rule.path_length for rule in result.decision_rules]
                axes[0, 0].hist(path_lengths, bins=max(1, len(set(path_lengths))), alpha=0.7)
                axes[0, 0].set_xlabel('Path Length')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Decision Path Length Distribution')
            
            # 2. Confidence vs Support scatter
            if result.decision_rules:
                confidences = [rule.confidence for rule in result.decision_rules]
                supports = [rule.support for rule in result.decision_rules]
                colors = ['red' if rule.risk_level == 'HIGH' else 'yellow' if rule.risk_level == 'MEDIUM' else 'green' 
                         for rule in result.decision_rules]
                
                scatter = axes[0, 1].scatter(confidences, supports, c=colors, alpha=0.6)
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Support')
                axes[0, 1].set_title('Rule Confidence vs Support')
                
                # Legend
                red_patch = mpatches.Patch(color='red', label='High Risk')
                yellow_patch = mpatches.Patch(color='yellow', label='Medium Risk')
                green_patch = mpatches.Patch(color='green', label='Low Risk')
                axes[0, 1].legend(handles=[red_patch, yellow_patch, green_patch])
            
            # 3. Feature usage frequency
            if result.feature_interaction_analysis.get('feature_usage_frequency'):
                feature_usage = result.feature_interaction_analysis['feature_usage_frequency']
                top_features = sorted(feature_usage.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
                
                feature_names = [item[0] for item in top_features]
                feature_counts = [item[1]['count'] for item in top_features]
                
                axes[1, 0].barh(range(len(feature_names)), feature_counts)
                axes[1, 0].set_yticks(range(len(feature_names)))
                axes[1, 0].set_yticklabels(feature_names, fontsize=8)
                axes[1, 0].set_xlabel('Usage Count')
                axes[1, 0].set_title('Feature Usage in Decision Paths')
                axes[1, 0].invert_yaxis()
            
            # 4. Trading signal distribution
            if result.trading_decision_analysis and 'signal_distribution' in result.trading_decision_analysis:
                signals = list(result.trading_decision_analysis['signal_distribution'].keys())
                counts = list(result.trading_decision_analysis['signal_distribution'].values())
                
                axes[1, 1].pie(counts, labels=signals, autopct='%1.1f%%')
                axes[1, 1].set_title('Trading Signal Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved decision path visualization to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            # Return simple figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f"Visualization Error: {e}", ha='center', va='center')
            return fig
    
    def export_decision_rules(
        self,
        result: DecisionPathResult,
        export_path: Path,
        format: str = 'json',
        max_rules: int = 100
    ) -> None:
        """Export decision rules in various formats"""
        try:
            # Select top rules
            top_rules = sorted(result.decision_rules, key=lambda r: (r.confidence, r.support), reverse=True)[:max_rules]
            
            if format == 'json':
                export_data = {
                    'decision_rules': [
                        {
                            'path_id': rule.path_id,
                            'conditions': rule.conditions,
                            'prediction': rule.prediction,
                            'confidence': rule.confidence,
                            'support': rule.support,
                            'trading_signal': rule.trading_signal,
                            'risk_level': rule.risk_level,
                            'market_conditions': rule.market_conditions,
                            'profitability_estimate': rule.profitability_estimate,
                            'path_length': rule.path_length
                        }
                        for rule in top_rules
                    ],
                    'summary': {
                        'total_rules': len(result.decision_rules),
                        'exported_rules': len(top_rules),
                        'model_type': result.model_type,
                        'timestamp': result.timestamp.isoformat(),
                        'complexity_metrics': {
                            'model_complexity_score': result.model_complexity_score,
                            'decision_boundary_clarity': result.decision_boundary_clarity,
                            'rule_consistency_score': result.rule_consistency_score
                        }
                    }
                }
                
                with open(export_path.with_suffix('.json'), 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                    
            elif format == 'csv':
                # Export as CSV
                rule_data = []
                for rule in top_rules:
                    rule_data.append({
                        'path_id': rule.path_id,
                        'conditions': ' AND '.join(rule.conditions),
                        'prediction': rule.prediction,
                        'confidence': rule.confidence,
                        'support': rule.support,
                        'trading_signal': rule.trading_signal,
                        'risk_level': rule.risk_level,
                        'path_length': rule.path_length
                    })
                
                df = pd.DataFrame(rule_data)
                df.to_csv(export_path.with_suffix('.csv'), index=False)
            
            elif format == 'text':
                # Export as human-readable text
                with open(export_path.with_suffix('.txt'), 'w') as f:
                    f.write(f"Decision Rules Analysis\n")
                    f.write(f"Model Type: {result.model_type}\n")
                    f.write(f"Generated: {result.timestamp}\n")
                    f.write(f"Total Rules: {len(result.decision_rules)}\n\n")
                    
                    for i, rule in enumerate(top_rules, 1):
                        f.write(f"Rule {i}: {rule.path_id}\n")
                        f.write(f"  Conditions: {' AND '.join(rule.conditions)}\n")
                        f.write(f"  Prediction: {rule.prediction}\n")
                        f.write(f"  Confidence: {rule.confidence:.3f}\n")
                        f.write(f"  Support: {rule.support}\n")
                        f.write(f"  Trading Signal: {rule.trading_signal}\n")
                        f.write(f"  Risk Level: {rule.risk_level}\n\n")
            
            logger.info(f"Exported {len(top_rules)} decision rules to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting decision rules: {e}")
            raise