"""
Feature Importance Analysis Crypto Trading Bot v5.0

 comprehensive analysis
 support multiple methods enterprise patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

# Advanced feature selection imports
try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression
    ADVANCED_SELECTION_AVAILABLE = True
except ImportError:
    ADVANCED_SELECTION_AVAILABLE = False

# Crypto-specific feature analysis
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportanceConfig:
    """Configuration feature importance analysis enterprise patterns"""
    methods: List[str] = None  # ['permutation', 'shap', 'mutual_info', 'rfe', 'univariate']
    n_repeats: int = 10  # For permutation importance
    random_state: int = 42
    scoring_metric: str = 'accuracy'  # or 'roc_auc', 'f1', 'r2', 'neg_mean_squared_error'
    cv_folds: int = 5
    # Feature selection parameters
    k_best_features: int = 20
    rfe_step: float = 0.1
    rfe_cv_folds: int = 3
    # Enterprise performance
    parallel_workers: int = 4
    cache_results: bool = True
    # Trading specific
    focus_on_trading_features: bool = True
    min_importance_threshold: float = 0.01
    stability_analysis: bool = True

    def __post_init__(self):
        if self.methods is None:
            self.methods = ['permutation', 'mutual_info', 'univariate']


@dataclass
class FeatureImportanceResult:
    """ analysis metadata"""
    feature_scores: Dict[str, Dict[str, float]]  # method -> feature -> score
    feature_rankings: Dict[str, List[Tuple[str, float]]]  # method -> [(feature, score), ...]
    consensus_ranking: List[Tuple[str, float]]  # Consensus across methods
    stability_scores: Dict[str, float]  # Feature -> stability score
    method_correlations: Dict[Tuple[str, str], float]  # Method pair -> correlation
    model_type: str
    timestamp: datetime = datetime.now()
    # Trading specific analysis
    trading_feature_analysis: Optional[Dict[str, Any]] = None
    feature_categories: Optional[Dict[str, str]] = None
    # Statistical metadata
    total_features: int = 0
    significant_features: int = 0
    redundant_features: Optional[List[str]] = None
    #  enterprise metadata
    validation_metrics: Optional[Dict[str, float]] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingFeatureImportanceAnalyzer:
    """
    Enterprise-grade feature importance analyzer crypto trading models
    
    Provides comprehensive feature analysis :
    - Trading signal prediction models
    - Risk assessment models
    - Portfolio optimization
    - Market regime detection
    - Feature selection dimensionality reduction
    
    enterprise patterns:
    - Multi-method consensus analysis
    - Trading-aware feature categorization
    - Enterprise caching monitoring
    - Async processing large datasets
    - Statistical validation stability analysis
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        config: Optional[FeatureImportanceConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize feature importance analyzer"""
        self.model = model
        self.config = config or FeatureImportanceConfig()
        self.cache_dir = cache_dir or Path("./cache/feature_importance")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature categorization trading
        self.trading_feature_categories = self._init_trading_feature_categories()
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"Initialized feature importance analyzer with methods: {self.config.methods}")
    
    def _init_trading_feature_categories(self) -> Dict[str, List[str]]:
        """Initialize trading-specific feature categories"""
        return {
            'price_features': ['price', 'close', 'open', 'high', 'low', 'adj_close'],
            'volume_features': ['volume', 'vol', 'turnover', 'market_cap'],
            'technical_indicators': ['rsi', 'macd', 'sma', 'ema', 'bb', 'stoch', 'adx', 'cci'],
            'volatility_features': ['volatility', 'atr', 'std', 'var', 'vix'],
            'momentum_features': ['momentum', 'roc', 'williams', 'trix'],
            'trend_features': ['trend', 'slope', 'direction', 'channel'],
            'market_structure': ['support', 'resistance', 'fibonacci', 'pivot'],
            'fundamental_features': ['pe_ratio', 'pb_ratio', 'roe', 'debt_ratio'],
            'sentiment_features': ['sentiment', 'fear_greed', 'social_volume'],
            'macro_features': ['interest_rate', 'inflation', 'gdp', 'unemployment']
        }
    
    async def analyze_async(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> FeatureImportanceResult:
        """
        Async feature importance analysis large datasets
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Feature names interpretability
            
        Returns:
            Comprehensive feature importance results
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
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> FeatureImportanceResult:
        """
        Comprehensive feature importance analysis
        
        Args:
            X: Feature matrix
            y: Target variable  
            feature_names: Feature names
            
        Returns:
            Feature importance analysis results
        """
        try:
            # Prepare data
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
                X_array = X.values
            else:
                X_array = X
                feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            if isinstance(y, pd.Series):
                y_array = y.values
            else:
                y_array = y
            
            logger.info(f"Analyzing {len(feature_names)} features with {len(self.config.methods)} methods")
            
            # Initialize results
            feature_scores = {}
            feature_rankings = {}
            
            # Apply each method
            for method in self.config.methods:
                try:
                    logger.info(f"Computing importance using {method}")
                    scores = self._compute_importance_method(X_array, y_array, feature_names, method)
                    
                    if scores is not None:
                        feature_scores[method] = scores
                        # Create ranking
                        sorted_features = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
                        feature_rankings[method] = sorted_features
                        
                except Exception as e:
                    logger.warning(f"Failed to compute {method} importance: {e}")
                    continue
            
            if not feature_scores:
                raise ValueError("No importance methods succeeded")
            
            # Calculate consensus ranking
            consensus_ranking = self._calculate_consensus_ranking(feature_rankings)
            
            # Calculate stability scores
            stability_scores = self._calculate_stability_scores(feature_scores) if self.config.stability_analysis else {}
            
            # Calculate method correlations
            method_correlations = self._calculate_method_correlations(feature_scores)
            
            # Analyze trading features
            trading_analysis = self._analyze_trading_features(feature_scores, feature_names) if self.config.focus_on_trading_features else None
            
            # Categorize features
            feature_categories = self._categorize_features(feature_names)
            
            # Find redundant features
            redundant_features = self._find_redundant_features(X_array, feature_names, consensus_ranking)
            
            # Validation metrics
            validation_metrics = self._calculate_validation_metrics(X_array, y_array, consensus_ranking)
            
            # Create comprehensive result
            result = FeatureImportanceResult(
                feature_scores=feature_scores,
                feature_rankings=feature_rankings,
                consensus_ranking=consensus_ranking,
                stability_scores=stability_scores,
                method_correlations=method_correlations,
                model_type=type(self.model).__name__,
                timestamp=datetime.now(),
                trading_feature_analysis=trading_analysis,
                feature_categories=feature_categories,
                total_features=len(feature_names),
                significant_features=len([f for f, s in consensus_ranking if abs(s) > self.config.min_importance_threshold]),
                redundant_features=redundant_features,
                validation_metrics=validation_metrics
            )
            
            logger.info(f"Feature importance analysis completed. Found {result.significant_features} significant features.")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            raise
    
    def _compute_importance_method(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str
    ) -> Optional[Dict[str, float]]:
        """Compute feature importance using specific method"""
        
        try:
            if method == 'permutation':
                return self._compute_permutation_importance(X, y, feature_names)
            elif method == 'shap' and SHAP_AVAILABLE:
                return self._compute_shap_importance(X, y, feature_names)
            elif method == 'mutual_info' and ADVANCED_SELECTION_AVAILABLE:
                return self._compute_mutual_info_importance(X, y, feature_names)
            elif method == 'rfe' and ADVANCED_SELECTION_AVAILABLE:
                return self._compute_rfe_importance(X, y, feature_names)
            elif method == 'univariate' and ADVANCED_SELECTION_AVAILABLE:
                return self._compute_univariate_importance(X, y, feature_names)
            elif method == 'model_based':
                return self._compute_model_based_importance(X, y, feature_names)
            else:
                logger.warning(f"Method {method} not available or recognized")
                return None
                
        except Exception as e:
            logger.warning(f"Error computing {method} importance: {e}")
            return None
    
    def _compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute permutation importance"""
        try:
            # Determine scoring metric based on problem type
            scoring = self._get_scoring_metric(y)
            
            perm_importance = permutation_importance(
                self.model,
                X,
                y,
                n_repeats=self.config.n_repeats,
                random_state=self.config.random_state,
                scoring=scoring,
                n_jobs=-1
            )
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(perm_importance.importances_mean[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing permutation importance: {e}")
            return {}
    
    def _compute_shap_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute SHAP-based importance"""
        try:
            # Use subset for performance
            n_samples = min(1000, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[indices]
            
            # Initialize appropriate SHAP explainer
            if hasattr(self.model, 'feature_importances_'):  # Tree models
                explainer = shap.TreeExplainer(self.model)
            else:  # Other models
                explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    X_subset[:100]  # Background data
                )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_subset)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Calculate mean absolute SHAP values
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(mean_abs_shap[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing SHAP importance: {e}")
            return {}
    
    def _compute_mutual_info_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute mutual information importance"""
        try:
            # Determine if classification or regression
            is_classification = self._is_classification_problem(y)
            
            if is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=self.config.random_state)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.config.random_state)
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(mi_scores[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing mutual info importance: {e}")
            return {}
    
    def _compute_rfe_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute RFE-based importance"""
        try:
            # Create appropriate estimator for RFE
            if self._is_classification_problem(y):
                estimator = RandomForestClassifier(n_estimators=50, random_state=self.config.random_state)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=self.config.random_state)
            
            # Use RFECV for automatic feature selection
            rfe = RFECV(
                estimator,
                step=self.config.rfe_step,
                cv=self.config.rfe_cv_folds,
                scoring=self._get_scoring_metric(y),
                n_jobs=-1
            )
            
            rfe.fit(X, y)
            
            # RFE ranking (lower is better, convert to importance score)
            max_rank = np.max(rfe.ranking_)
            importance_scores = {}
            
            for i, feature_name in enumerate(feature_names):
                # Convert ranking to importance (higher = better)
                importance = 1.0 / rfe.ranking_[i] if rfe.ranking_[i] > 0 else 0.0
                importance_scores[feature_name] = float(importance)
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing RFE importance: {e}")
            return {}
    
    def _compute_univariate_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute univariate statistical importance"""
        try:
            # Determine appropriate test
            if self._is_classification_problem(y):
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
            
            selector.fit(X, y)
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(selector.scores_[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing univariate importance: {e}")
            return {}
    
    def _compute_model_based_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute model intrinsic feature importance"""
        try:
            # Check if model has feature_importances_
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                # Linear models: use coefficient magnitude
                importances = np.abs(self.model.coef_)
                if importances.ndim > 1:
                    importances = np.mean(importances, axis=0)
            else:
                # Train surrogate model for importance
                if self._is_classification_problem(y):
                    surrogate = RandomForestClassifier(n_estimators=100, random_state=self.config.random_state)
                else:
                    surrogate = RandomForestRegressor(n_estimators=100, random_state=self.config.random_state)
                
                surrogate.fit(X, y)
                importances = surrogate.feature_importances_
            
            importance_scores = {}
            for i, feature_name in enumerate(feature_names):
                importance_scores[feature_name] = float(importances[i])
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Error computing model-based importance: {e}")
            return {}
    
    def _calculate_consensus_ranking(
        self,
        feature_rankings: Dict[str, List[Tuple[str, float]]]
    ) -> List[Tuple[str, float]]:
        """Calculate consensus ranking across methods"""
        try:
            # Collect all features
            all_features = set()
            for rankings in feature_rankings.values():
                all_features.update([f for f, _ in rankings])
            
            # Calculate average rank for each feature
            feature_avg_ranks = {}
            feature_scores = {}
            
            for feature in all_features:
                ranks = []
                scores = []
                
                for method, rankings in feature_rankings.items():
                    feature_dict = dict(rankings)
                    if feature in feature_dict:
                        # Find rank (position in sorted list)
                        rank = next(i for i, (f, _) in enumerate(rankings) if f == feature)
                        ranks.append(rank)
                        scores.append(abs(feature_dict[feature]))  # Use absolute value
                    else:
                        # Feature not in this method's ranking
                        ranks.append(len(rankings))  # Worst possible rank
                        scores.append(0.0)
                
                feature_avg_ranks[feature] = np.mean(ranks)
                feature_scores[feature] = np.mean(scores)
            
            # Sort by average rank (lower is better)
            consensus_ranking = sorted(
                [(feature, feature_scores[feature]) for feature in all_features],
                key=lambda x: feature_avg_ranks[x[0]]
            )
            
            return consensus_ranking
            
        except Exception as e:
            logger.warning(f"Error calculating consensus ranking: {e}")
            return []
    
    def _calculate_stability_scores(
        self,
        feature_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate stability of feature importance across methods"""
        try:
            if len(feature_scores) < 2:
                return {}
            
            # Get all features
            all_features = set()
            for scores in feature_scores.values():
                all_features.update(scores.keys())
            
            stability_scores = {}
            
            for feature in all_features:
                feature_importances = []
                
                for method_scores in feature_scores.values():
                    importance = method_scores.get(feature, 0.0)
                    feature_importances.append(importance)
                
                # Calculate coefficient of variation as stability measure
                if len(feature_importances) > 1 and np.mean(feature_importances) > 0:
                    cv = np.std(feature_importances) / np.mean(feature_importances)
                    stability = 1.0 / (1.0 + cv)  # Higher CV = lower stability
                else:
                    stability = 0.0
                
                stability_scores[feature] = float(stability)
            
            return stability_scores
            
        except Exception as e:
            logger.warning(f"Error calculating stability scores: {e}")
            return {}
    
    def _calculate_method_correlations(
        self,
        feature_scores: Dict[str, Dict[str, float]]
    ) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between importance methods"""
        try:
            methods = list(feature_scores.keys())
            if len(methods) < 2:
                return {}
            
            # Get common features
            common_features = set(feature_scores[methods[0]].keys())
            for method in methods[1:]:
                common_features.intersection_update(feature_scores[method].keys())
            
            if len(common_features) < 3:  # Need at least 3 features for correlation
                return {}
            
            correlations = {}
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    scores1 = [feature_scores[method1][f] for f in common_features]
                    scores2 = [feature_scores[method2][f] for f in common_features]
                    
                    correlation = float(np.corrcoef(scores1, scores2)[0, 1])
                    correlations[(method1, method2)] = correlation
            
            return correlations
            
        except Exception as e:
            logger.warning(f"Error calculating method correlations: {e}")
            return {}
    
    def _analyze_trading_features(
        self,
        feature_scores: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze trading-specific features"""
        try:
            analysis = {
                'category_importance': {},
                'top_features_by_category': {},
                'trading_signals': {},
                'risk_features': {}
            }
            
            # Categorize features
            feature_categories = self._categorize_features(feature_names)
            
            # Calculate average importance by category
            for category, category_features in self.trading_feature_categories.items():
                category_scores = []
                category_feature_scores = {}
                
                for feature_name in feature_names:
                    if any(keyword in feature_name.lower() for keyword in category_features):
                        # Calculate average score across methods
                        feature_avg_scores = []
                        for method_scores in feature_scores.values():
                            if feature_name in method_scores:
                                feature_avg_scores.append(method_scores[feature_name])
                        
                        if feature_avg_scores:
                            avg_score = np.mean(feature_avg_scores)
                            category_scores.append(avg_score)
                            category_feature_scores[feature_name] = avg_score
                
                if category_scores:
                    analysis['category_importance'][category] = {
                        'average_importance': float(np.mean(category_scores)),
                        'max_importance': float(np.max(category_scores)),
                        'feature_count': len(category_scores)
                    }
                    
                    # Top features in this category
                    top_features = sorted(category_feature_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                    analysis['top_features_by_category'][category] = top_features
            
            # Identify potential trading signals
            signal_features = []
            for feature_name in feature_names:
                if any(keyword in feature_name.lower() for keyword in ['signal', 'prediction', 'forecast', 'target']):
                    avg_scores = []
                    for method_scores in feature_scores.values():
                        if feature_name in method_scores:
                            avg_scores.append(method_scores[feature_name])
                    if avg_scores:
                        signal_features.append((feature_name, np.mean(avg_scores)))
            
            analysis['trading_signals'] = sorted(signal_features, key=lambda x: x[1], reverse=True)[:10]
            
            # Identify risk features
            risk_features = []
            for feature_name in feature_names:
                if any(keyword in feature_name.lower() for keyword in ['risk', 'volatility', 'var', 'drawdown']):
                    avg_scores = []
                    for method_scores in feature_scores.values():
                        if feature_name in method_scores:
                            avg_scores.append(method_scores[feature_name])
                    if avg_scores:
                        risk_features.append((feature_name, np.mean(avg_scores)))
            
            analysis['risk_features'] = sorted(risk_features, key=lambda x: x[1], reverse=True)[:10]
            
            return analysis
            
        except Exception as e:
            logger.warning(f"Error analyzing trading features: {e}")
            return {}
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, str]:
        """Categorize features based on trading domain knowledge"""
        feature_categories = {}
        
        for feature_name in feature_names:
            feature_lower = feature_name.lower()
            category = 'other'
            
            for cat_name, keywords in self.trading_feature_categories.items():
                if any(keyword in feature_lower for keyword in keywords):
                    category = cat_name
                    break
            
            feature_categories[feature_name] = category
        
        return feature_categories
    
    def _find_redundant_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        consensus_ranking: List[Tuple[str, float]],
        correlation_threshold: float = 0.95
    ) -> List[str]:
        """Find redundant features based on correlation"""
        try:
            if len(feature_names) < 2:
                return []
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(X.T)
            redundant_features = []
            
            # Find highly correlated feature pairs
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs(correlation_matrix[i, j]) > correlation_threshold:
                        # Keep the more important feature
                        feature_i_rank = next((idx for idx, (name, _) in enumerate(consensus_ranking) if name == feature_names[i]), len(consensus_ranking))
                        feature_j_rank = next((idx for idx, (name, _) in enumerate(consensus_ranking) if name == feature_names[j]), len(consensus_ranking))
                        
                        # Remove the less important one
                        if feature_i_rank > feature_j_rank:  # Higher rank = less important
                            if feature_names[i] not in redundant_features:
                                redundant_features.append(feature_names[i])
                        else:
                            if feature_names[j] not in redundant_features:
                                redundant_features.append(feature_names[j])
            
            return redundant_features
            
        except Exception as e:
            logger.warning(f"Error finding redundant features: {e}")
            return []
    
    def _calculate_validation_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        consensus_ranking: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """Calculate validation metrics feature importance"""
        try:
            validation_metrics = {}
            
            # Test performance with different numbers of top features
            for n_features in [5, 10, 20, min(50, len(consensus_ranking))]:
                if n_features > len(consensus_ranking):
                    continue
                
                # Get top N features
                top_features = [name for name, _ in consensus_ranking[:n_features]]
                feature_indices = [i for i, name in enumerate(self._get_feature_names_from_array(X)) if name in top_features]
                
                if not feature_indices:
                    continue
                
                X_subset = X[:, feature_indices]
                
                # Cross-validation score
                scoring = self._get_scoring_metric(y)
                cv_scores = cross_val_score(self.model, X_subset, y, cv=self.config.cv_folds, scoring=scoring)
                
                validation_metrics[f'cv_score_top_{n_features}'] = float(np.mean(cv_scores))
                validation_metrics[f'cv_std_top_{n_features}'] = float(np.std(cv_scores))
            
            return validation_metrics
            
        except Exception as e:
            logger.warning(f"Error calculating validation metrics: {e}")
            return {}
    
    def _get_feature_names_from_array(self, X: np.ndarray) -> List[str]:
        """Get feature names for array (fallback method)"""
        return [f"feature_{i}" for i in range(X.shape[1])]
    
    def _is_classification_problem(self, y: np.ndarray) -> bool:
        """Determine if problem is classification or regression"""
        unique_values = np.unique(y)
        return len(unique_values) <= 20 and np.all(unique_values == np.round(unique_values))
    
    def _get_scoring_metric(self, y: np.ndarray) -> str:
        """Get appropriate scoring metric based on problem type"""
        if self._is_classification_problem(y):
            unique_classes = len(np.unique(y))
            if unique_classes == 2:
                return 'roc_auc'
            else:
                return 'f1_macro'
        else:
            return 'neg_mean_squared_error'
    
    def create_importance_visualization(
        self,
        result: FeatureImportanceResult,
        top_n: int = 20,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Create comprehensive importance visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Feature Importance Analysis', fontsize=16)
            
            # 1. Consensus ranking bar plot
            consensus_features = [name for name, _ in result.consensus_ranking[:top_n]]
            consensus_scores = [score for _, score in result.consensus_ranking[:top_n]]
            
            axes[0, 0].barh(range(len(consensus_features)), consensus_scores)
            axes[0, 0].set_yticks(range(len(consensus_features)))
            axes[0, 0].set_yticklabels(consensus_features)
            axes[0, 0].set_xlabel('Importance Score')
            axes[0, 0].set_title('Consensus Feature Ranking')
            axes[0, 0].invert_yaxis()
            
            # 2. Method comparison heatmap
            if len(result.feature_scores) > 1:
                methods = list(result.feature_scores.keys())
                features_for_heatmap = consensus_features[:min(15, len(consensus_features))]
                
                heatmap_data = []
                for method in methods:
                    method_scores = []
                    for feature in features_for_heatmap:
                        score = result.feature_scores[method].get(feature, 0.0)
                        method_scores.append(score)
                    heatmap_data.append(method_scores)
                
                sns.heatmap(
                    heatmap_data,
                    xticklabels=features_for_heatmap,
                    yticklabels=methods,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    ax=axes[0, 1]
                )
                axes[0, 1].set_title('Method Comparison')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Stability scores
            if result.stability_scores:
                stable_features = sorted(result.stability_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                stability_names = [name for name, _ in stable_features]
                stability_values = [score for _, score in stable_features]
                
                axes[1, 0].barh(range(len(stability_names)), stability_values)
                axes[1, 0].set_yticks(range(len(stability_names)))
                axes[1, 0].set_yticklabels(stability_names)
                axes[1, 0].set_xlabel('Stability Score')
                axes[1, 0].set_title('Feature Stability')
                axes[1, 0].invert_yaxis()
            
            # 4. Trading category analysis
            if result.trading_feature_analysis and 'category_importance' in result.trading_feature_analysis:
                categories = list(result.trading_feature_analysis['category_importance'].keys())
                category_scores = [result.trading_feature_analysis['category_importance'][cat]['average_importance'] 
                                 for cat in categories]
                
                axes[1, 1].pie(category_scores, labels=categories, autopct='%1.1f%%')
                axes[1, 1].set_title('Importance by Trading Category')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved feature importance visualization to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            # Return simple figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f"Visualization Error: {e}", ha='center', va='center')
            return fig
    
    def export_results(
        self,
        result: FeatureImportanceResult,
        export_path: Path,
        format: str = 'json'
    ) -> None:
        """Export feature importance results"""
        try:
            export_data = {
                'consensus_ranking': result.consensus_ranking,
                'feature_scores': result.feature_scores,
                'stability_scores': result.stability_scores,
                'method_correlations': {f"{k[0]}_{k[1]}": v for k, v in result.method_correlations.items()},
                'trading_feature_analysis': result.trading_feature_analysis,
                'feature_categories': result.feature_categories,
                'validation_metrics': result.validation_metrics,
                'metadata': {
                    'model_type': result.model_type,
                    'timestamp': result.timestamp.isoformat(),
                    'total_features': result.total_features,
                    'significant_features': result.significant_features,
                    'redundant_features': result.redundant_features
                }
            }
            
            if format == 'json':
                with open(export_path.with_suffix('.json'), 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format == 'csv':
                # Export consensus ranking as CSV
                df = pd.DataFrame(result.consensus_ranking, columns=['feature', 'importance'])
                df.to_csv(export_path.with_suffix('.csv'), index=False)
            
            logger.info(f"Exported feature importance results to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise