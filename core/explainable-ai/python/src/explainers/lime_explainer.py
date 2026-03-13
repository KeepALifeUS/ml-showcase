"""
LIME (Local Interpretable Model-agnostic Explanations) Explainer Crypto Trading Bot v5.0

 LIME-based local explanations ML .
enterprise patterns model-agnostic interpretability.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import pandas as pd
from lime import lime_tabular, lime_text, lime_image
from lime.discretize import QuartileDiscretizer, DecileDiscretizer, EntropyDiscretizer
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances

# Crypto trading specific imports
try:
    import torch
    import tensorflow as tf
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class LIMEConfig:
    """Configuration LIME explainer enterprise patterns"""
    mode: str = 'tabular'  # 'tabular', 'text', 'image'
    num_features: int = 10
    num_samples: int = 1000
    distance_metric: str = 'euclidean'
    model_regressor: Optional[str] = 'linear'  # 'linear', 'tree', 'lasso'
    discretize_continuous: bool = True
    discretizer: str = 'quartile'  # 'quartile', 'decile', 'entropy'
    sample_around_instance: bool = True
    random_state: int = 42
    # Enterprise performance settings
    cache_explanations: bool = True
    parallel_workers: int = 4
    timeout_seconds: int = 180
    batch_size: int = 16
    # Trading specific
    include_feature_values: bool = True
    explain_top_labels: int = 2


@dataclass
class LIMEExplanation:
    """Structured LIME explanation metadata"""
    local_explanation: Dict[str, Any]
    feature_importance: Dict[str, float]
    prediction_probabilities: np.ndarray
    intercept: float
    r2_score: float
    model_type: str
    explanation_type: str = 'lime'
    timestamp: datetime = datetime.now()
    # Crypto trading context
    symbol: Optional[str] = None
    prediction_confidence: Optional[float] = None
    trade_signal: Optional[str] = None
    feature_values: Optional[Dict[str, float]] = None
    # Local explanation metadata
    num_features_used: int = 0
    samples_generated: int = 0
    explanation_fidelity: float = 0.0
    #  enterprise metadata
    model_version: Optional[str] = None
    data_version: Optional[str] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingLIMEExplainer:
    """
    Enterprise-grade LIME explainer crypto trading models
    
    Provides local, model-agnostic interpretability :
    - Individual trading decisions
    - Specific market conditions
    - Real-time prediction explanations
    - A/B testing model variants
    - Black-box model understanding
    
    enterprise patterns:
    - Model-agnostic architecture any ML framework
    - High-performance local explanations
    - Enterprise caching monitoring
    - Async processing real-time trading
    - Compliance tracking regulatory audit
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, Callable],
        config: Optional[LIMEConfig] = None,
        cache_dir: Optional[Path] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        categorical_names: Optional[Dict[int, List[str]]] = None
    ):
        """Initialize LIME explainer enterprise configuration"""
        self.model = model
        self.config = config or LIMEConfig()
        self.cache_dir = cache_dir or Path("./cache/lime_explanations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model wrapper consistent interface
        self._model_predict_fn = self._create_model_predict_function()
        
        # LIME explainer components
        self._explainer: Optional[lime_tabular.LimeTabularExplainer] = None
        self._feature_names = feature_names or []
        self._categorical_features = categorical_features or []
        self._categorical_names = categorical_names or {}
        
        # Training data LIME background
        self._training_data: Optional[np.ndarray] = None
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"Initialized LIME explainer with config: {self.config}")
    
    def _create_model_predict_function(self) -> Callable:
        """Create consistent model prediction function"""
        if hasattr(self.model, 'predict_proba'):
            return lambda x: self.model.predict_proba(x)
        elif hasattr(self.model, 'predict'):
            # Wrap single predictions probability-like format
            def predict_wrapper(x):
                predictions = self.model.predict(x)
                if predictions.ndim == 1:
                    # Binary classification: convert to probabilities
                    predictions = np.column_stack([1 - predictions, predictions])
                return predictions
            return predict_wrapper
        else:
            # Custom callable model
            return self.model
    
    def fit(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        categorical_names: Optional[Dict[int, List[str]]] = None
    ) -> None:
        """
        Fit LIME explainer on training data
        
        Args:
            training_data: Background data LIME sampling
            feature_names: Feature names interpretability
            categorical_features: Indices of categorical features
            categorical_names: Names categorical feature values
        """
        try:
            # Process training data
            if isinstance(training_data, pd.DataFrame):
                self._feature_names = list(training_data.columns)
                self._training_data = training_data.values
            else:
                self._feature_names = feature_names or self._feature_names or [f"feature_{i}" for i in range(training_data.shape[1])]
                self._training_data = training_data
            
            # Update categorical information
            if categorical_features is not None:
                self._categorical_features = categorical_features
            if categorical_names is not None:
                self._categorical_names = categorical_names
            
            # Initialize discretizer
            discretizer = self._create_discretizer()
            
            # Initialize LIME explainer
            self._explainer = lime_tabular.LimeTabularExplainer(
                training_data=self._training_data,
                feature_names=self._feature_names,
                categorical_features=self._categorical_features,
                categorical_names=self._categorical_names,
                discretize_continuous=self.config.discretize_continuous,
                discretizer=discretizer,
                random_state=self.config.random_state,
                mode='classification'
            )
            
            logger.info(f"LIME explainer fitted with {len(self._training_data)} training samples")
            
        except Exception as e:
            logger.error(f"Error fitting LIME explainer: {e}")
            raise
    
    def _create_discretizer(self):
        """Create appropriate discretizer based on configuration"""
        if self.config.discretizer == 'quartile':
            return QuartileDiscretizer
        elif self.config.discretizer == 'decile':
            return DecileDiscretizer
        elif self.config.discretizer == 'entropy':
            return EntropyDiscretizer
        else:
            logger.warning(f"Unknown discretizer {self.config.discretizer}, using quartile")
            return QuartileDiscretizer
    
    async def explain_async(
        self,
        instance: Union[np.ndarray, pd.Series],
        symbol: Optional[str] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Async LIME explanation high-frequency trading
        
        Args:
            instance: Single instance explanation
            symbol: Trading symbol context
            num_features: Number of features explanation
            num_samples: Number of perturbed samples
            
        Returns:
            Structured LIME explanation trading metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.explain,
            instance,
            symbol,
            num_features,
            num_samples
        )
    
    def explain(
        self,
        instance: Union[np.ndarray, pd.Series],
        symbol: Optional[str] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> LIMEExplanation:
        """
        Generate LIME explanation single instance
        
        Args:
            instance: Input instance explanation
            symbol: Trading symbol context
            num_features: Number of top features
            num_samples: Number of perturbed samples
            
        Returns:
            Comprehensive LIME explanation
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        try:
            # Prepare instance data
            if isinstance(instance, pd.Series):
                instance_array = instance.values
            else:
                instance_array = instance
                
            if instance_array.ndim > 1:
                instance_array = instance_array.flatten()
            
            # Configuration
            num_features = num_features or self.config.num_features
            num_samples = num_samples or self.config.num_samples
            
            # Check cache
            cache_key = self._get_cache_key(instance_array, symbol, num_features, num_samples)
            if self.config.cache_explanations:
                cached_explanation = self._load_from_cache(cache_key)
                if cached_explanation is not None:
                    logger.debug("Loaded LIME explanation from cache")
                    return cached_explanation
            
            # Generate LIME explanation
            lime_explanation = self._explainer.explain_instance(
                data_row=instance_array,
                predict_fn=self._model_predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric=self.config.distance_metric,
                model_regressor=self.config.model_regressor,
                top_labels=self.config.explain_top_labels,
                labels=None  # Explain all labels
            )
            
            # Extract explanation components
            feature_importance = {}
            local_explanation = {}
            
            # Get explanation each label
            for label in lime_explanation.available_labels():
                label_explanation = lime_explanation.as_list(label=label)
                local_explanation[f'label_{label}'] = label_explanation
                
                # Extract feature importance this label
                for feature_name, importance in label_explanation:
                    if f'label_{label}' not in feature_importance:
                        feature_importance[f'label_{label}'] = {}
                    feature_importance[f'label_{label}'][feature_name] = importance
            
            # Get prediction probabilities
            prediction_probs = self._model_predict_fn(instance_array.reshape(1, -1))[0]
            
            # Calculate explanation quality metrics
            explanation_metrics = self._calculate_explanation_metrics(
                lime_explanation, instance_array, num_samples
            )
            
            # Extract trading metadata
            trade_metadata = self._extract_trade_metadata(
                instance_array, symbol, prediction_probs
            )
            
            # Create comprehensive explanation
            explanation = LIMEExplanation(
                local_explanation=local_explanation,
                feature_importance=feature_importance,
                prediction_probabilities=prediction_probs,
                intercept=float(lime_explanation.intercept[lime_explanation.available_labels()[0]]),
                r2_score=explanation_metrics['r2_score'],
                model_type=type(self.model).__name__,
                timestamp=datetime.now(),
                symbol=symbol,
                num_features_used=num_features,
                samples_generated=num_samples,
                explanation_fidelity=explanation_metrics['fidelity'],
                **trade_metadata
            )
            
            # Cache future use
            if self.config.cache_explanations:
                self._save_to_cache(cache_key, explanation)
            
            logger.info(f"Generated LIME explanation for {symbol or 'unknown symbol'}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}")
            raise
    
    def explain_batch(
        self,
        instances: Union[List[np.ndarray], np.ndarray, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None
    ) -> List[LIMEExplanation]:
        """
        Batch LIME explanations multiple instances
        
        Args:
            instances: Multiple instances explanation
            symbols: Trading symbols each instance
            num_features: Number of features per explanation
            num_samples: Number of samples per explanation
            
        Returns:
            List of LIME explanations
        """
        # Prepare instances
        if isinstance(instances, pd.DataFrame):
            instance_arrays = [instances.iloc[i].values for i in range(len(instances))]
        elif isinstance(instances, np.ndarray):
            if instances.ndim == 1:
                instance_arrays = [instances]
            else:
                instance_arrays = [instances[i] for i in range(len(instances))]
        else:
            instance_arrays = instances
        
        symbols = symbols or [None] * len(instance_arrays)
        
        # Process in batches performance
        batch_size = self.config.batch_size
        explanations = []
        
        for i in range(0, len(instance_arrays), batch_size):
            batch_instances = instance_arrays[i:i + batch_size]
            batch_symbols = symbols[i:i + batch_size]
            
            batch_explanations = []
            for instance, symbol in zip(batch_instances, batch_symbols):
                explanation = self.explain(instance, symbol, num_features, num_samples)
                batch_explanations.append(explanation)
            
            explanations.extend(batch_explanations)
            logger.debug(f"Processed batch {i // batch_size + 1}/{(len(instance_arrays) + batch_size - 1) // batch_size}")
        
        return explanations
    
    def _calculate_explanation_metrics(
        self,
        lime_explanation,
        instance: np.ndarray,
        num_samples: int
    ) -> Dict[str, float]:
        """Calculate explanation quality metrics"""
        try:
            metrics = {}
            
            # Get the local model's R² score
            label = lime_explanation.available_labels()[0]
            if hasattr(lime_explanation.local_model, 'score'):
                # Get generated samples for scoring
                neighborhood_data, neighborhood_labels = lime_explanation.get_neighborhood_data_and_labels(
                    instance,
                    num_samples,
                    distance_metric=self.config.distance_metric,
                    sampling_method='gaussian'
                )
                
                r2_score = lime_explanation.local_model.score(
                    neighborhood_data, 
                    neighborhood_labels
                )
                metrics['r2_score'] = float(r2_score)
            else:
                metrics['r2_score'] = 0.0
            
            # Calculate explanation fidelity (consistency)
            original_pred = self._model_predict_fn(instance.reshape(1, -1))[0]
            
            # Generate small perturbations check consistency
            n_fidelity_samples = min(100, num_samples // 10)
            fidelity_scores = []
            
            for _ in range(n_fidelity_samples):
                # Small perturbation
                noise = np.random.normal(0, 0.01, instance.shape)
                perturbed_instance = instance + noise
                
                # Get explanations both
                try:
                    perturbed_explanation = self._explainer.explain_instance(
                        perturbed_instance,
                        self._model_predict_fn,
                        num_features=5, # Reduced speed
                        num_samples=num_samples // 10
                    )
                    
                    # Compare feature rankings
                    orig_features = [f for f, _ in lime_explanation.as_list(label=label)]
                    pert_features = [f for f, _ in perturbed_explanation.as_list(label=label)]
                    
                    # Calculate ranking similarity
                    common_features = set(orig_features[:5]).intersection(set(pert_features[:5]))
                    similarity = len(common_features) / 5.0
                    fidelity_scores.append(similarity)
                    
                except Exception:
                    # Skip if perturbation fails
                    continue
            
            if fidelity_scores:
                metrics['fidelity'] = float(np.mean(fidelity_scores))
            else:
                metrics['fidelity'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating explanation metrics: {e}")
            return {'r2_score': 0.0, 'fidelity': 0.0}
    
    def _extract_trade_metadata(
        self,
        instance: np.ndarray,
        symbol: Optional[str],
        prediction_probs: np.ndarray
    ) -> Dict[str, Any]:
        """Extract trading specific metadata"""
        metadata = {}
        
        try:
            # Prediction confidence
            confidence = float(np.max(prediction_probs))
            predicted_class = int(np.argmax(prediction_probs))
            
            # Map to trading signals
            if len(prediction_probs) == 2:
                trade_signal = "BUY" if predicted_class == 1 else "SELL"
            elif len(prediction_probs) == 3:
                trade_signal = ["SELL", "HOLD", "BUY"][predicted_class]
            else:
                trade_signal = f"CLASS_{predicted_class}"
            
            metadata.update({
                'prediction_confidence': confidence,
                'trade_signal': trade_signal
            })
            
            # Feature values context
            if self.config.include_feature_values and self._feature_names:
                feature_values = {}
                for i, feature_name in enumerate(self._feature_names):
                    if i < len(instance):
                        feature_values[feature_name] = float(instance[i])
                metadata['feature_values'] = feature_values
            
        except Exception as e:
            logger.warning(f"Error extracting trade metadata: {e}")
        
        return metadata
    
    def analyze_feature_perturbations(
        self,
        instance: np.ndarray,
        feature_indices: Optional[List[int]] = None,
        perturbation_sizes: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how feature perturbations affect predictions
        
        Args:
            instance: Base instance perturbation
            feature_indices: Features to perturb (default: all)
            perturbation_sizes: Sizes of perturbations
            
        Returns:
            Perturbation analysis results
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted")
        
        feature_indices = feature_indices or list(range(len(instance)))
        perturbation_sizes = perturbation_sizes or [0.01, 0.05, 0.1, 0.2]
        
        results = {}
        base_prediction = self._model_predict_fn(instance.reshape(1, -1))[0]
        
        for feature_idx in feature_indices:
            feature_name = self._feature_names[feature_idx] if feature_idx < len(self._feature_names) else f"feature_{feature_idx}"
            feature_results = {
                'base_value': float(instance[feature_idx]),
                'perturbation_effects': {}
            }
            
            for perturbation_size in perturbation_sizes:
                # Positive perturbation
                perturbed_instance = instance.copy()
                perturbed_instance[feature_idx] += perturbation_size
                pos_prediction = self._model_predict_fn(perturbed_instance.reshape(1, -1))[0]
                
                # Negative perturbation  
                perturbed_instance = instance.copy()
                perturbed_instance[feature_idx] -= perturbation_size
                neg_prediction = self._model_predict_fn(perturbed_instance.reshape(1, -1))[0]
                
                # Calculate impact
                pos_impact = float(np.linalg.norm(pos_prediction - base_prediction))
                neg_impact = float(np.linalg.norm(neg_prediction - base_prediction))
                
                feature_results['perturbation_effects'][f'size_{perturbation_size}'] = {
                    'positive_impact': pos_impact,
                    'negative_impact': neg_impact,
                    'sensitivity': (pos_impact + neg_impact) / 2
                }
            
            results[feature_name] = feature_results
        
        return results
    
    def compare_explanations(
        self,
        explanations: List[LIMEExplanation],
        comparison_metric: str = 'feature_overlap'
    ) -> Dict[str, Any]:
        """
        Compare multiple LIME explanations consistency analysis
        
        Args:
            explanations: List of explanations comparison
            comparison_metric: Metric comparison
            
        Returns:
            Comparison analysis results
        """
        if len(explanations) < 2:
            raise ValueError("Need at least 2 explanations comparison")
        
        comparison_results = {
            'total_explanations': len(explanations),
            'comparison_metric': comparison_metric,
            'analysis': {}
        }
        
        if comparison_metric == 'feature_overlap':
            # Analyze feature overlap across explanations
            all_features = set()
            explanation_features = []
            
            for explanation in explanations:
                # Get top features from first label
                first_label = list(explanation.feature_importance.keys())[0]
                features = set(explanation.feature_importance[first_label].keys())
                explanation_features.append(features)
                all_features.update(features)
            
            # Calculate pairwise overlaps
            overlaps = []
            for i in range(len(explanation_features)):
                for j in range(i + 1, len(explanation_features)):
                    overlap = len(explanation_features[i].intersection(explanation_features[j]))
                    union = len(explanation_features[i].union(explanation_features[j]))
                    jaccard = overlap / union if union > 0 else 0
                    overlaps.append(jaccard)
            
            comparison_results['analysis'] = {
                'mean_jaccard_similarity': float(np.mean(overlaps)),
                'std_jaccard_similarity': float(np.std(overlaps)),
                'total_unique_features': len(all_features),
                'average_features_per_explanation': float(np.mean([len(f) for f in explanation_features]))
            }
            
        elif comparison_metric == 'prediction_consistency':
            # Analyze prediction consistency
            confidences = [exp.prediction_confidence for exp in explanations if exp.prediction_confidence]
            signals = [exp.trade_signal for exp in explanations if exp.trade_signal]
            
            comparison_results['analysis'] = {
                'confidence_mean': float(np.mean(confidences)) if confidences else 0,
                'confidence_std': float(np.std(confidences)) if confidences else 0,
                'signal_consistency': len(set(signals)) / len(signals) if signals else 0,
                'most_common_signal': max(set(signals), key=signals.count) if signals else None
            }
        
        return comparison_results
    
    def _get_cache_key(
        self,
        instance: np.ndarray,
        symbol: Optional[str],
        num_features: int,
        num_samples: int
    ) -> str:
        """Generate cache key explanation"""
        instance_hash = hash(instance.tobytes())
        model_hash = hash(str(type(self.model)))
        symbol_hash = hash(symbol or "")
        config_hash = hash((num_features, num_samples))
        return f"lime_{model_hash}_{symbol_hash}_{instance_hash}_{config_hash}"
    
    def _save_to_cache(self, cache_key: str, explanation: LIMEExplanation) -> None:
        """Save explanation to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(explanation, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[LIMEExplanation]:
        """Load explanation from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
        return None
    
    def generate_summary_report(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Generate comprehensive summary report for multiple explanations"""
        if not explanations:
            return {}
        
        try:
            # Collect all feature importance data
            all_feature_importance = {}
            for explanation in explanations:
                for label, features in explanation.feature_importance.items():
                    if label not in all_feature_importance:
                        all_feature_importance[label] = []
                    all_feature_importance[label].append(features)
            
            # Aggregate feature importance
            aggregated_importance = {}
            for label, feature_lists in all_feature_importance.items():
                # Combine all feature dictionaries
                all_features = set()
                for feature_dict in feature_lists:
                    all_features.update(feature_dict.keys())
                
                label_importance = {}
                for feature in all_features:
                    importance_values = [
                        feature_dict.get(feature, 0.0) 
                        for feature_dict in feature_lists
                    ]
                    label_importance[feature] = {
                        'mean': float(np.mean(importance_values)),
                        'std': float(np.std(importance_values)),
                        'frequency': sum(1 for val in importance_values if val != 0.0) / len(importance_values)
                    }
                
                aggregated_importance[label] = label_importance
            
            # Calculate summary statistics
            summary = {
                'total_explanations': len(explanations),
                'time_range': {
                    'start': min(exp.timestamp for exp in explanations).isoformat(),
                    'end': max(exp.timestamp for exp in explanations).isoformat()
                },
                'model_performance': {
                    'mean_r2_score': float(np.mean([exp.r2_score for exp in explanations])),
                    'mean_fidelity': float(np.mean([exp.explanation_fidelity for exp in explanations])),
                    'mean_confidence': float(np.mean([exp.prediction_confidence for exp in explanations if exp.prediction_confidence]))
                },
                'feature_analysis': aggregated_importance,
                'trading_analysis': self._analyze_trading_patterns(explanations),
                'consistency_analysis': self.compare_explanations(explanations),
                'compliance_metrics': self._calculate_compliance_metrics(explanations)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            return {}
    
    def _analyze_trading_patterns(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Analyze trading patterns from explanations"""
        patterns = {}
        
        try:
            # Signal distribution
            signals = [exp.trade_signal for exp in explanations if exp.trade_signal]
            if signals:
                from collections import Counter
                signal_counts = Counter(signals)
                patterns['signal_distribution'] = dict(signal_counts)
            
            # Symbol analysis
            symbols = [exp.symbol for exp in explanations if exp.symbol]
            if symbols:
                from collections import Counter
                symbol_counts = Counter(symbols)
                patterns['symbol_frequency'] = dict(symbol_counts)
            
            # Confidence trends
            confidences = [exp.prediction_confidence for exp in explanations if exp.prediction_confidence]
            timestamps = [exp.timestamp for exp in explanations if exp.prediction_confidence]
            
            if len(confidences) > 1:
                # Simple trend analysis
                time_diffs = [(t - timestamps[0]).total_seconds() for t in timestamps]
                correlation = np.corrcoef(time_diffs, confidences)[0, 1] if len(time_diffs) > 1 else 0
                patterns['confidence_trend'] = {
                    'correlation_with_time': float(correlation),
                    'mean_confidence': float(np.mean(confidences)),
                    'confidence_volatility': float(np.std(confidences))
                }
            
        except Exception as e:
            logger.warning(f"Error analyzing trading patterns: {e}")
        
        return patterns
    
    def _calculate_compliance_metrics(self, explanations: List[LIMEExplanation]) -> Dict[str, Any]:
        """Calculate compliance regulatory metrics"""
        compliance = {
            'explainability_coverage': len(explanations),
            'model_transparency': {
                'average_r2_score': float(np.mean([exp.r2_score for exp in explanations])),
                'average_fidelity': float(np.mean([exp.explanation_fidelity for exp in explanations])),
                'explanation_consistency': float(np.std([exp.r2_score for exp in explanations]))
            },
            'decision_auditability': {
                'traced_decisions': len([exp for exp in explanations if exp.trade_signal]),
                'feature_coverage': len(set().union(*[
                    list(exp.feature_importance.get(list(exp.feature_importance.keys())[0], {}).keys())
                    for exp in explanations
                    if exp.feature_importance
                ])),
                'temporal_coverage_hours': (
                    max(exp.timestamp for exp in explanations) - 
                    min(exp.timestamp for exp in explanations)
                ).total_seconds() / 3600 if len(explanations) > 1 else 0
            }
        }
        
        return compliance
    
    def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Cleanup old cached explanations"""
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old LIME cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0