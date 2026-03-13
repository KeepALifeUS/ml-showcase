"""
Counterfactual Explainer Crypto Trading Bot v5.0

 counterfactual explanations understanding decision boundaries
     .
enterprise patterns scenario-based interpretability.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize, differential_evolution
import pickle
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Advanced optimization imports
try:
    from alibi.explainers import CounterFactual, CounterFactualProto
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualConfig:
    """Configuration counterfactual explainer enterprise patterns"""
    method: str = 'optimization'  # 'optimization', 'prototype', 'genetic', 'alibi'
    target_class: Optional[int] = None
    distance_metric: str = 'euclidean'
    feature_range: Optional[Tuple[float, float]] = None
    categorical_features: Optional[List[int]] = None
    immutable_features: Optional[List[int]] = None
    # Optimization parameters
    max_iterations: int = 1000
    learning_rate: float = 0.01
    tolerance: float = 1e-6
    regularization_strength: float = 0.01
    # Enterprise performance
    cache_results: bool = True
    parallel_workers: int = 4
    timeout_seconds: int = 300
    # Trading specific
    preserve_market_constraints: bool = True
    min_price_change: float = 0.001
    max_price_change: float = 0.1


@dataclass
class CounterfactualExplanation:
    """Structured counterfactual explanation metadata"""
    original_instance: np.ndarray
    counterfactual_instance: np.ndarray
    original_prediction: Union[int, float, np.ndarray]
    counterfactual_prediction: Union[int, float, np.ndarray]
    feature_changes: Dict[str, Dict[str, float]]
    distance_to_original: float
    validity_score: float
    plausibility_score: float
    model_type: str
    explanation_type: str = 'counterfactual'
    timestamp: datetime = datetime.now()
    # Crypto trading context
    symbol: Optional[str] = None
    original_trade_signal: Optional[str] = None
    counterfactual_trade_signal: Optional[str] = None
    market_feasibility: Optional[Dict[str, bool]] = None
    # Optimization metadata
    optimization_converged: bool = False
    optimization_iterations: int = 0
    optimization_method: str = ''
    #  enterprise metadata
    model_version: Optional[str] = None
    scenario_id: Optional[str] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingCounterfactualExplainer:
    """
    Enterprise-grade counterfactual explainer crypto trading models
    
    Provides counterfactual analysis :
    - Understanding decision boundaries
    - "What-if" scenario analysis  
    - Alternative trading strategies
    - Risk scenario modeling
    - Model robustness testing
    
    enterprise patterns:
    - Advanced optimization algorithms
    - Market constraint preservation
    - High-performance async processing
    - Enterprise caching monitoring
    - Regulatory compliance tracking
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, Callable],
        config: Optional[CounterfactualConfig] = None,
        feature_names: Optional[List[str]] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize counterfactual explainer enterprise configuration"""
        self.model = model
        self.config = config or CounterfactualConfig()
        self.feature_names = feature_names or []
        self.cache_dir = cache_dir or Path("./cache/counterfactual_explanations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model prediction function
        self._predict_fn = self._create_predict_function()
        self._predict_proba_fn = self._create_predict_proba_function()
        
        # Training data boundaries constraint validation
        self._feature_bounds: Optional[Dict[int, Tuple[float, float]]] = None
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        # Initialize advanced explainers if available
        self._alibi_explainer = None
        if ALIBI_AVAILABLE and self.config.method == 'alibi':
            self._init_alibi_explainer()
        
        logger.info(f"Initialized counterfactual explainer with method: {self.config.method}")
    
    def _create_predict_function(self) -> Callable:
        """Create consistent prediction function"""
        if hasattr(self.model, 'predict'):
            return self.model.predict
        else:
            return self.model
    
    def _create_predict_proba_function(self) -> Callable:
        """Create probability prediction function"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba
        elif hasattr(self.model, 'predict'):
            # Binary classification wrapper
            def predict_proba_wrapper(x):
                predictions = self.model.predict(x)
                if predictions.ndim == 1:
                    predictions = predictions.reshape(-1, 1)
                    predictions = np.column_stack([1 - predictions, predictions])
                return predictions
            return predict_proba_wrapper
        else:
            return self.model
    
    def fit(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit counterfactual explainer on training data
        
        Args:
            training_data: Training data constraint learning
            feature_names: Feature names interpretability
        """
        try:
            # Process training data
            if isinstance(training_data, pd.DataFrame):
                self.feature_names = list(training_data.columns)
                training_array = training_data.values
            else:
                self.feature_names = feature_names or self.feature_names
                training_array = training_data
            
            # Calculate feature bounds constraints
            self._feature_bounds = {}
            for i in range(training_array.shape[1]):
                if self.config.categorical_features and i in self.config.categorical_features:
                    # Categorical feature: use unique values
                    unique_values = np.unique(training_array[:, i])
                    self._feature_bounds[i] = (float(np.min(unique_values)), float(np.max(unique_values)))
                else:
                    # Continuous feature: use min/max margin
                    margin = (np.max(training_array[:, i]) - np.min(training_array[:, i])) * 0.1
                    self._feature_bounds[i] = (
                        float(np.min(training_array[:, i]) - margin),
                        float(np.max(training_array[:, i]) + margin)
                    )
            
            # Initialize Alibi explainer if needed
            if ALIBI_AVAILABLE and self.config.method == 'alibi':
                self._init_alibi_explainer()
            
            logger.info(f"Counterfactual explainer fitted on {len(training_array)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting counterfactual explainer: {e}")
            raise
    
    def _init_alibi_explainer(self) -> None:
        """Initialize Alibi counterfactual explainer"""
        try:
            if ALIBI_AVAILABLE:
                # Basic Alibi counterfactual explainer
                self._alibi_explainer = CounterFactual(
                    self._predict_fn,
                    shape=None,  # Will be set during explanation
                    distance_fn='l2',
                    target_class=self.config.target_class
                )
                logger.info("Initialized Alibi counterfactual explainer")
        except Exception as e:
            logger.warning(f"Failed to initialize Alibi explainer: {e}")
            self._alibi_explainer = None
    
    async def explain_async(
        self,
        instance: Union[np.ndarray, pd.Series],
        target_class: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> CounterfactualExplanation:
        """
        Async counterfactual explanation high-frequency trading
        
        Args:
            instance: Instance counterfactual generation
            target_class: Target class counterfactual
            symbol: Trading symbol context
            
        Returns:
            Counterfactual explanation trading metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.explain,
            instance,
            target_class,
            symbol
        )
    
    def explain(
        self,
        instance: Union[np.ndarray, pd.Series],
        target_class: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation instance
        
        Args:
            instance: Input instance counterfactual
            target_class: Desired prediction class
            symbol: Trading symbol context
            
        Returns:
            Comprehensive counterfactual explanation
        """
        try:
            # Prepare instance
            if isinstance(instance, pd.Series):
                instance_array = instance.values
            else:
                instance_array = instance.copy()
            
            if instance_array.ndim > 1:
                instance_array = instance_array.flatten()
            
            # Get original prediction
            original_pred = self._predict_fn(instance_array.reshape(1, -1))[0]
            original_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                original_pred_proba = self._predict_proba_fn(instance_array.reshape(1, -1))[0]
            
            # Determine target class
            if target_class is None:
                if isinstance(original_pred, (int, np.integer)):
                    # For classification, flip to opposite class
                    target_class = 1 - original_pred if original_pred in [0, 1] else (original_pred + 1) % 2
                else:
                    # For regression, target higher/lower value
                    target_class = original_pred + 0.1 if np.random.random() > 0.5 else original_pred - 0.1
            
            # Check cache
            cache_key = self._get_cache_key(instance_array, target_class, symbol)
            if self.config.cache_results:
                cached_explanation = self._load_from_cache(cache_key)
                if cached_explanation is not None:
                    logger.debug("Loaded counterfactual from cache")
                    return cached_explanation
            
            # Generate counterfactual based on method
            counterfactual_result = self._generate_counterfactual(
                instance_array, target_class, original_pred
            )
            
            counterfactual_instance = counterfactual_result['counterfactual']
            optimization_info = counterfactual_result['optimization_info']
            
            # Get counterfactual prediction
            cf_pred = self._predict_fn(counterfactual_instance.reshape(1, -1))[0]
            cf_pred_proba = None
            if hasattr(self.model, 'predict_proba'):
                cf_pred_proba = self._predict_proba_fn(counterfactual_instance.reshape(1, -1))[0]
            
            # Calculate metrics
            distance = float(np.linalg.norm(counterfactual_instance - instance_array))
            
            # Calculate validity (did we achieve target?)
            if isinstance(target_class, (int, np.integer)):
                validity = 1.0 if cf_pred == target_class else 0.0
            else:
                validity = 1.0 - abs(cf_pred - target_class) / max(abs(target_class), 1.0)
            
            # Calculate plausibility (realistic feature values?)
            plausibility = self._calculate_plausibility(counterfactual_instance)
            
            # Analyze feature changes
            feature_changes = self._analyze_feature_changes(
                instance_array, counterfactual_instance
            )
            
            # Extract trading metadata
            trade_metadata = self._extract_trade_metadata(
                instance_array, counterfactual_instance, symbol,
                original_pred_proba, cf_pred_proba
            )
            
            # Create comprehensive explanation
            explanation = CounterfactualExplanation(
                original_instance=instance_array,
                counterfactual_instance=counterfactual_instance,
                original_prediction=original_pred,
                counterfactual_prediction=cf_pred,
                feature_changes=feature_changes,
                distance_to_original=distance,
                validity_score=validity,
                plausibility_score=plausibility,
                model_type=type(self.model).__name__,
                timestamp=datetime.now(),
                symbol=symbol,
                optimization_converged=optimization_info['converged'],
                optimization_iterations=optimization_info['iterations'],
                optimization_method=self.config.method,
                **trade_metadata
            )
            
            # Cache
            if self.config.cache_results:
                self._save_to_cache(cache_key, explanation)
            
            logger.info(f"Generated counterfactual for {symbol or 'unknown symbol'}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating counterfactual explanation: {e}")
            raise
    
    def _generate_counterfactual(
        self,
        instance: np.ndarray,
        target_class: Union[int, float],
        original_pred: Union[int, float]
    ) -> Dict[str, Any]:
        """Generate counterfactual using specified method"""
        
        if self.config.method == 'optimization':
            return self._generate_via_optimization(instance, target_class)
        elif self.config.method == 'genetic':
            return self._generate_via_genetic_algorithm(instance, target_class)
        elif self.config.method == 'prototype' and self._feature_bounds:
            return self._generate_via_prototype(instance, target_class)
        elif self.config.method == 'alibi' and self._alibi_explainer:
            return self._generate_via_alibi(instance, target_class)
        else:
            # Fallback to optimization
            logger.warning(f"Method {self.config.method} not available, using optimization")
            return self._generate_via_optimization(instance, target_class)
    
    def _generate_via_optimization(
        self,
        instance: np.ndarray,
        target_class: Union[int, float]
    ) -> Dict[str, Any]:
        """Generate counterfactual via gradient-based optimization"""
        
        def objective(x):
            """Objective function optimization"""
            # Prediction loss
            pred = self._predict_fn(x.reshape(1, -1))[0]
            if isinstance(target_class, (int, np.integer)):
                # Classification: maximize probability of target class
                if hasattr(self.model, 'predict_proba'):
                    proba = self._predict_proba_fn(x.reshape(1, -1))[0]
                    pred_loss = -proba[target_class] if target_class < len(proba) else abs(pred - target_class)
                else:
                    pred_loss = abs(pred - target_class)
            else:
                # Regression: minimize distance to target
                pred_loss = abs(pred - target_class)
            
            # Distance penalty
            distance_loss = np.linalg.norm(x - instance)
            
            # Constraint penalties
            constraint_penalty = self._calculate_constraint_penalty(x)
            
            return pred_loss + self.config.regularization_strength * distance_loss + constraint_penalty
        
        # Set bounds
        bounds = []
        for i in range(len(instance)):
            if self._feature_bounds and i in self._feature_bounds:
                bounds.append(self._feature_bounds[i])
            else:
                # Default bounds
                bounds.append((instance[i] - abs(instance[i]) * 0.5, instance[i] + abs(instance[i]) * 0.5))
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0=instance,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': self.config.max_iterations, 'ftol': self.config.tolerance}
            )
            
            return {
                'counterfactual': result.x,
                'optimization_info': {
                    'converged': result.success,
                    'iterations': result.nit,
                    'final_cost': result.fun
                }
            }
            
        except Exception as e:
            logger.warning(f"Optimization failed: {e}")
            # Return perturbed instance as fallback
            noise = np.random.normal(0, 0.1, instance.shape)
            return {
                'counterfactual': instance + noise,
                'optimization_info': {
                    'converged': False,
                    'iterations': 0,
                    'final_cost': float('inf')
                }
            }
    
    def _generate_via_genetic_algorithm(
        self,
        instance: np.ndarray,
        target_class: Union[int, float]
    ) -> Dict[str, Any]:
        """Generate counterfactual via genetic algorithm"""
        
        def objective(x):
            # Same objective as optimization
            pred = self._predict_fn(x.reshape(1, -1))[0]
            if isinstance(target_class, (int, np.integer)):
                if hasattr(self.model, 'predict_proba'):
                    proba = self._predict_proba_fn(x.reshape(1, -1))[0]
                    pred_loss = -proba[target_class] if target_class < len(proba) else abs(pred - target_class)
                else:
                    pred_loss = abs(pred - target_class)
            else:
                pred_loss = abs(pred - target_class)
            
            distance_loss = np.linalg.norm(x - instance)
            constraint_penalty = self._calculate_constraint_penalty(x)
            
            return pred_loss + self.config.regularization_strength * distance_loss + constraint_penalty
        
        # Set bounds for genetic algorithm
        bounds = []
        for i in range(len(instance)):
            if self._feature_bounds and i in self._feature_bounds:
                bounds.append(self._feature_bounds[i])
            else:
                bounds.append((instance[i] - abs(instance[i]) * 0.5, instance[i] + abs(instance[i]) * 0.5))
        
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=self.config.max_iterations // 10,  # Genetic algorithms need fewer iterations
                seed=42,
                atol=self.config.tolerance
            )
            
            return {
                'counterfactual': result.x,
                'optimization_info': {
                    'converged': result.success,
                    'iterations': result.nit,
                    'final_cost': result.fun
                }
            }
            
        except Exception as e:
            logger.warning(f"Genetic algorithm failed: {e}")
            noise = np.random.normal(0, 0.1, instance.shape)
            return {
                'counterfactual': instance + noise,
                'optimization_info': {
                    'converged': False,
                    'iterations': 0,
                    'final_cost': float('inf')
                }
            }
    
    def _generate_via_prototype(
        self,
        instance: np.ndarray,
        target_class: Union[int, float]
    ) -> Dict[str, Any]:
        """Generate counterfactual via prototype selection"""
        # Simple prototype method: random sampling with constraints
        best_counterfactual = None
        best_score = float('inf')
        iterations = 0
        
        for _ in range(self.config.max_iterations // 10):
            # Generate random candidate
            candidate = instance.copy()
            
            # Randomly perturb features
            n_features_to_change = np.random.randint(1, len(instance) // 2 + 1)
            features_to_change = np.random.choice(len(instance), n_features_to_change, replace=False)
            
            for feature_idx in features_to_change:
                if self._feature_bounds and feature_idx in self._feature_bounds:
                    low, high = self._feature_bounds[feature_idx]
                    candidate[feature_idx] = np.random.uniform(low, high)
                else:
                    candidate[feature_idx] *= np.random.uniform(0.5, 1.5)
            
            # Evaluate candidate
            pred = self._predict_fn(candidate.reshape(1, -1))[0]
            
            if isinstance(target_class, (int, np.integer)):
                if pred == target_class:
                    distance = np.linalg.norm(candidate - instance)
                    if distance < best_score:
                        best_counterfactual = candidate
                        best_score = distance
            else:
                score = abs(pred - target_class) + 0.1 * np.linalg.norm(candidate - instance)
                if score < best_score:
                    best_counterfactual = candidate
                    best_score = score
            
            iterations += 1
        
        if best_counterfactual is None:
            # Fallback: simple perturbation
            noise = np.random.normal(0, 0.1, instance.shape)
            best_counterfactual = instance + noise
        
        return {
            'counterfactual': best_counterfactual,
            'optimization_info': {
                'converged': best_counterfactual is not None and best_score < float('inf'),
                'iterations': iterations,
                'final_cost': best_score
            }
        }
    
    def _generate_via_alibi(
        self,
        instance: np.ndarray,
        target_class: Union[int, float]
    ) -> Dict[str, Any]:
        """Generate counterfactual via Alibi library"""
        try:
            if not self._alibi_explainer:
                raise ValueError("Alibi explainer not initialized")
            
            # Alibi expects specific format
            explanation = self._alibi_explainer.explain(
                instance.reshape(1, -1),
                target_class=int(target_class) if isinstance(target_class, (int, np.integer)) else None
            )
            
            if explanation.cf is not None:
                return {
                    'counterfactual': explanation.cf[0],
                    'optimization_info': {
                        'converged': True,
                        'iterations': getattr(explanation, 'iterations', 0),
                        'final_cost': getattr(explanation, 'cost', 0.0)
                    }
                }
            else:
                raise ValueError("Alibi failed to generate counterfactual")
                
        except Exception as e:
            logger.warning(f"Alibi counterfactual generation failed: {e}")
            # Fallback to optimization
            return self._generate_via_optimization(instance, target_class)
    
    def _calculate_constraint_penalty(self, x: np.ndarray) -> float:
        """Calculate penalty constraint violations"""
        penalty = 0.0
        
        try:
            # Feature bounds constraints
            if self._feature_bounds:
                for i, (low, high) in self._feature_bounds.items():
                    if i < len(x):
                        if x[i] < low:
                            penalty += (low - x[i]) ** 2
                        elif x[i] > high:
                            penalty += (x[i] - high) ** 2
            
            # Immutable features constraint
            if self.config.immutable_features:
                for feature_idx in self.config.immutable_features:
                    if feature_idx < len(x):
                        # These features should not change significantly
                        penalty += 100 * abs(x[feature_idx])  # Assuming original was passed as reference
            
            # Trading specific constraints
            if self.config.preserve_market_constraints:
                # Price-related features should change within realistic bounds
                price_features = [i for i, name in enumerate(self.feature_names) 
                                if name and 'price' in name.lower()]
                for feature_idx in price_features:
                    if feature_idx < len(x):
                        # Price changes should be within reasonable bounds
                        relative_change = abs(x[feature_idx]) / max(abs(x[feature_idx]), 1e-6)
                        if relative_change > self.config.max_price_change:
                            penalty += 10 * (relative_change - self.config.max_price_change) ** 2
            
        except Exception as e:
            logger.warning(f"Error calculating constraint penalty: {e}")
        
        return penalty
    
    def _calculate_plausibility(self, counterfactual: np.ndarray) -> float:
        """Calculate plausibility score counterfactual"""
        try:
            plausibility_factors = []
            
            # Feature bounds plausibility
            if self._feature_bounds:
                bounds_violations = 0
                for i, (low, high) in self._feature_bounds.items():
                    if i < len(counterfactual):
                        if not (low <= counterfactual[i] <= high):
                            bounds_violations += 1
                
                bounds_plausibility = 1.0 - (bounds_violations / len(self._feature_bounds))
                plausibility_factors.append(bounds_plausibility)
            
            # Feature correlation plausibility (simplified)
            if len(counterfactual) > 1:
                # Check if feature relationships are preserved
                correlation_score = 1.0  # Simplified for now
                plausibility_factors.append(correlation_score)
            
            # Trading specific plausibility
            if self.config.preserve_market_constraints:
                market_plausibility = self._calculate_market_plausibility(counterfactual)
                plausibility_factors.append(market_plausibility)
            
            return float(np.mean(plausibility_factors)) if plausibility_factors else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating plausibility: {e}")
            return 0.5
    
    def _calculate_market_plausibility(self, counterfactual: np.ndarray) -> float:
        """Calculate market-specific plausibility"""
        try:
            # Check for realistic trading feature values
            plausibility_score = 1.0
            
            # Price features should be positive
            price_features = [i for i, name in enumerate(self.feature_names) 
                            if name and any(keyword in name.lower() for keyword in ['price', 'close', 'open', 'high', 'low'])]
            
            for feature_idx in price_features:
                if feature_idx < len(counterfactual) and counterfactual[feature_idx] <= 0:
                    plausibility_score -= 0.2
            
            # Volume features should be non-negative
            volume_features = [i for i, name in enumerate(self.feature_names) 
                             if name and 'volume' in name.lower()]
            
            for feature_idx in volume_features:
                if feature_idx < len(counterfactual) and counterfactual[feature_idx] < 0:
                    plausibility_score -= 0.1
            
            # Technical indicators should be within expected ranges
            rsi_features = [i for i, name in enumerate(self.feature_names) 
                          if name and 'rsi' in name.lower()]
            
            for feature_idx in rsi_features:
                if feature_idx < len(counterfactual):
                    rsi_value = counterfactual[feature_idx]
                    if not (0 <= rsi_value <= 100):
                        plausibility_score -= 0.1
            
            return max(0.0, plausibility_score)
            
        except Exception as e:
            logger.warning(f"Error calculating market plausibility: {e}")
            return 0.5
    
    def _analyze_feature_changes(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Analyze changes between original and counterfactual instances"""
        changes = {}
        
        try:
            for i in range(len(original)):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
                
                original_value = float(original[i])
                counterfactual_value = float(counterfactual[i])
                
                absolute_change = counterfactual_value - original_value
                relative_change = (absolute_change / original_value) if original_value != 0 else float('inf')
                
                changes[feature_name] = {
                    'original_value': original_value,
                    'counterfactual_value': counterfactual_value,
                    'absolute_change': absolute_change,
                    'relative_change': relative_change,
                    'magnitude': abs(absolute_change)
                }
            
        except Exception as e:
            logger.warning(f"Error analyzing feature changes: {e}")
        
        return changes
    
    def _extract_trade_metadata(
        self,
        original: np.ndarray,
        counterfactual: np.ndarray,
        symbol: Optional[str],
        original_proba: Optional[np.ndarray],
        cf_proba: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Extract trading specific metadata"""
        metadata = {}
        
        try:
            # Original trade signal
            if original_proba is not None:
                orig_class = int(np.argmax(original_proba))
                if len(original_proba) == 2:
                    orig_signal = "BUY" if orig_class == 1 else "SELL"
                elif len(original_proba) == 3:
                    orig_signal = ["SELL", "HOLD", "BUY"][orig_class]
                else:
                    orig_signal = f"CLASS_{orig_class}"
                metadata['original_trade_signal'] = orig_signal
            
            # Counterfactual trade signal
            if cf_proba is not None:
                cf_class = int(np.argmax(cf_proba))
                if len(cf_proba) == 2:
                    cf_signal = "BUY" if cf_class == 1 else "SELL"
                elif len(cf_proba) == 3:
                    cf_signal = ["SELL", "HOLD", "BUY"][cf_class]
                else:
                    cf_signal = f"CLASS_{cf_class}"
                metadata['counterfactual_trade_signal'] = cf_signal
            
            # Market feasibility analysis
            market_feasibility = {
                'realistic_price_changes': True,
                'feasible_volume_changes': True,
                'reasonable_indicator_changes': True
            }
            
            # Check price change feasibility
            price_features = [i for i, name in enumerate(self.feature_names) 
                            if name and 'price' in name.lower()]
            
            for feature_idx in price_features:
                if feature_idx < len(original):
                    price_change = abs(counterfactual[feature_idx] - original[feature_idx])
                    relative_change = price_change / max(abs(original[feature_idx]), 1e-6)
                    
                    if relative_change > self.config.max_price_change:
                        market_feasibility['realistic_price_changes'] = False
                        break
            
            metadata['market_feasibility'] = market_feasibility
            
        except Exception as e:
            logger.warning(f"Error extracting trade metadata: {e}")
        
        return metadata
    
    def analyze_decision_boundary(
        self,
        instance: np.ndarray,
        n_samples: int = 100,
        feature_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze decision boundary around instance
        
        Args:
            instance: Central instance analysis
            n_samples: Number of samples boundary exploration
            feature_indices: Features perturbation
            
        Returns:
            Decision boundary analysis
        """
        try:
            feature_indices = feature_indices or list(range(len(instance)))
            boundary_analysis = {
                'samples': [],
                'predictions': [],
                'distances': [],
                'feature_importance': {}
            }
            
            # Generate samples around instance
            for _ in range(n_samples):
                # Random perturbation
                sample = instance.copy()
                for feature_idx in feature_indices:
                    if self._feature_bounds and feature_idx in self._feature_bounds:
                        low, high = self._feature_bounds[feature_idx]
                        sample[feature_idx] = np.random.uniform(low, high)
                    else:
                        perturbation = np.random.normal(0, abs(instance[feature_idx]) * 0.1)
                        sample[feature_idx] += perturbation
                
                # Get prediction
                prediction = self._predict_fn(sample.reshape(1, -1))[0]
                distance = float(np.linalg.norm(sample - instance))
                
                boundary_analysis['samples'].append(sample.copy())
                boundary_analysis['predictions'].append(prediction)
                boundary_analysis['distances'].append(distance)
            
            # Analyze feature importance boundary crossing
            original_pred = self._predict_fn(instance.reshape(1, -1))[0]
            
            for feature_idx in feature_indices:
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                
                # Count how often changing this feature leads to prediction change
                changes = 0
                total = 0
                
                for sample, pred in zip(boundary_analysis['samples'], boundary_analysis['predictions']):
                    if abs(sample[feature_idx] - instance[feature_idx]) > 1e-6:  # Feature was changed
                        total += 1
                        if (isinstance(original_pred, (int, np.integer)) and pred != original_pred) or \
                           (isinstance(original_pred, (float, np.floating)) and abs(pred - original_pred) > 0.1):
                            changes += 1
                
                boundary_importance = changes / total if total > 0 else 0
                boundary_analysis['feature_importance'][feature_name] = boundary_importance
            
            return boundary_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing decision boundary: {e}")
            return {}
    
    def generate_multiple_counterfactuals(
        self,
        instance: Union[np.ndarray, pd.Series],
        n_counterfactuals: int = 5,
        diversity_weight: float = 0.5,
        symbol: Optional[str] = None
    ) -> List[CounterfactualExplanation]:
        """
        Generate multiple diverse counterfactuals comprehensive analysis
        
        Args:
            instance: Input instance
            n_counterfactuals: Number of counterfactuals to generate
            diversity_weight: Weight diversity selection
            symbol: Trading symbol
            
        Returns:
            List of diverse counterfactual explanations
        """
        counterfactuals = []
        
        try:
            # Generate more counterfactuals than needed
            candidates = []
            
            for i in range(n_counterfactuals * 3):
                # Try different target classes/values
                if hasattr(self.model, 'classes_'):
                    classes = self.model.classes_
                    target_class = classes[i % len(classes)]
                else:
                    # For regression unknown model
                    original_pred = self._predict_fn(instance.reshape(1, -1) if instance.ndim == 1 else instance)[0]
                    target_class = original_pred + np.random.uniform(-1, 1)
                
                try:
                    cf_explanation = self.explain(instance, target_class, symbol)
                    candidates.append(cf_explanation)
                except Exception as e:
                    logger.warning(f"Failed to generate counterfactual {i}: {e}")
                    continue
            
            if not candidates:
                logger.warning("No counterfactuals generated")
                return []
            
            # Select diverse counterfactuals
            selected = [candidates[0]]  # Start with first candidate
            
            for _ in range(min(n_counterfactuals - 1, len(candidates) - 1)):
                best_candidate = None
                best_score = -float('inf')
                
                for candidate in candidates:
                    if candidate in selected:
                        continue
                    
                    # Calculate diversity score
                    diversity_score = min([
                        np.linalg.norm(candidate.counterfactual_instance - selected_cf.counterfactual_instance)
                        for selected_cf in selected
                    ])
                    
                    # Calculate validity score
                    validity_score = candidate.validity_score
                    
                    # Combined score
                    combined_score = (1 - diversity_weight) * validity_score + diversity_weight * diversity_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected.append(best_candidate)
            
            return selected
            
        except Exception as e:
            logger.error(f"Error generating multiple counterfactuals: {e}")
            return []
    
    def _get_cache_key(
        self,
        instance: np.ndarray,
        target_class: Union[int, float],
        symbol: Optional[str]
    ) -> str:
        """Generate cache key counterfactual"""
        instance_hash = hash(instance.tobytes())
        model_hash = hash(str(type(self.model)))
        target_hash = hash(str(target_class))
        symbol_hash = hash(symbol or "")
        return f"counterfactual_{model_hash}_{symbol_hash}_{instance_hash}_{target_hash}"
    
    def _save_to_cache(self, cache_key: str, explanation: CounterfactualExplanation) -> None:
        """Save explanation to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(explanation, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[CounterfactualExplanation]:
        """Load explanation from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
        return None
    
    def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Cleanup old cached explanations"""
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old counterfactual cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0