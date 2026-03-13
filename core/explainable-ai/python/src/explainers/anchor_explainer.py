"""
Anchor Explainer Crypto Trading Bot v5.0

 Anchor-based explanations high-precision rule-based
interpretability .
enterprise patterns rule-based model understanding.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances
import pickle
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import itertools
from collections import defaultdict

# Advanced anchor explainer imports
try:
    from anchor import anchor_tabular
    ANCHOR_AVAILABLE = True
except ImportError:
    ANCHOR_AVAILABLE = False

try:
    from alibi.explainers import AnchorTabular
    ALIBI_ANCHOR_AVAILABLE = True
except ImportError:
    ALIBI_ANCHOR_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class AnchorConfig:
    """Configuration anchor explainer enterprise patterns"""
    threshold: float = 0.95 # Precision threshold anchors
    delta: float = 0.1  # Confidence level
    tau: float = 0.15  # Margin around threshold
    batch_size: int = 100
    coverage_samples: int = 10000
    beam_size: int = 1
    epsilon: float = 0.1  # For epsilon-greedy exploration
    # Feature discretization
    discretize_continuous: bool = True
    discretization_method: str = 'quartile'  # 'quartile', 'decile', 'custom'
    # Enterprise performance
    cache_anchors: bool = True
    parallel_workers: int = 4
    timeout_seconds: int = 600
    # Trading specific
    include_trading_rules: bool = True
    min_support_samples: int = 50
    max_rule_length: int = 5


@dataclass
class AnchorRule:
    """Single anchor rule trading metadata"""
    features: List[str]
    conditions: List[str] 
    precision: float
    coverage: float
    support: int
    confidence_interval: Tuple[float, float]
    # Trading specific
    trading_signal: Optional[str] = None
    market_conditions: Optional[List[str]] = None
    risk_level: Optional[str] = None
    profitability_estimate: Optional[float] = None


@dataclass
class AnchorExplanation:
    """Structured anchor explanation metadata"""
    anchor_rules: List[AnchorRule]
    primary_anchor: AnchorRule
    instance_prediction: Union[int, float, np.ndarray]
    instance_confidence: float
    model_type: str
    explanation_type: str = 'anchor'
    timestamp: datetime = datetime.now()
    # Crypto trading context
    symbol: Optional[str] = None
    trade_signal: Optional[str] = None
    market_regime: Optional[str] = None
    # Rule quality metrics
    total_rules_found: int = 0
    average_precision: float = 0.0
    average_coverage: float = 0.0
    rule_diversity_score: float = 0.0
    #  enterprise metadata
    model_version: Optional[str] = None
    rule_validation_score: Optional[float] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingAnchorExplainer:
    """
    Enterprise-grade anchor explainer crypto trading models
    
    Provides rule-based interpretability :
    - High-precision trading rules
    - Market condition identification
    - Strategy rule discovery
    - Risk constraint validation
    - Regulatory compliance rules
    
    enterprise patterns:
    - High-precision rule generation
    - Market-aware rule validation
    - Enterprise rule management
    - Async rule discovery
    - Regulatory compliance tracking
    """
    
    def __init__(
        self,
        model: Union[BaseEstimator, Callable],
        config: Optional[AnchorConfig] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize anchor explainer enterprise configuration"""
        self.model = model
        self.config = config or AnchorConfig()
        self.feature_names = feature_names or []
        self.categorical_features = categorical_features or []
        self.cache_dir = cache_dir or Path("./cache/anchor_explanations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model prediction function
        self._predict_fn = self._create_predict_function()
        self._predict_proba_fn = self._create_predict_proba_function()
        
        # Training data discretization rule validation
        self._training_data: Optional[np.ndarray] = None
        self._feature_discretizers: Dict[int, Any] = {}
        
        # Anchor explainer
        self._anchor_explainer = None
        
        # Async executor
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"Initialized anchor explainer with threshold: {self.config.threshold}")
    
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
        else:
            # Fallback non-probabilistic models
            def predict_proba_wrapper(x):
                predictions = self._predict_fn(x)
                if predictions.ndim == 1:
                    # Binary: convert to probabilities
                    return np.column_stack([1 - predictions, predictions])
                return predictions
            return predict_proba_wrapper
    
    def fit(
        self,
        training_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ) -> None:
        """
        Fit anchor explainer on training data
        
        Args:
            training_data: Training data discretization
            feature_names: Feature names interpretability
            categorical_features: Indices of categorical features
        """
        try:
            # Process training data
            if isinstance(training_data, pd.DataFrame):
                self.feature_names = list(training_data.columns)
                self._training_data = training_data.values
            else:
                self.feature_names = feature_names or self.feature_names
                self._training_data = training_data
            
            if categorical_features is not None:
                self.categorical_features = categorical_features
            
            # Setup feature discretization
            self._setup_discretization()
            
            # Initialize anchor explainer
            if ANCHOR_AVAILABLE:
                self._init_anchor_explainer()
            elif ALIBI_ANCHOR_AVAILABLE:
                self._init_alibi_anchor_explainer()
            else:
                logger.warning("No anchor library available, using custom implementation")
                self._init_custom_anchor_explainer()
            
            logger.info(f"Anchor explainer fitted on {len(self._training_data)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting anchor explainer: {e}")
            raise
    
    def _setup_discretization(self) -> None:
        """Setup feature discretization continuous features"""
        try:
            for feature_idx in range(self._training_data.shape[1]):
                if feature_idx not in self.categorical_features:
                    feature_data = self._training_data[:, feature_idx]
                    
                    if self.config.discretization_method == 'quartile':
                        # Quartile-based discretization
                        quartiles = np.percentile(feature_data, [25, 50, 75])
                        self._feature_discretizers[feature_idx] = {
                            'method': 'quartile',
                            'thresholds': quartiles
                        }
                    elif self.config.discretization_method == 'decile':
                        # Decile-based discretization
                        deciles = np.percentile(feature_data, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                        self._feature_discretizers[feature_idx] = {
                            'method': 'decile',
                            'thresholds': deciles
                        }
                    else:
                        # Custom trading-aware discretization
                        self._feature_discretizers[feature_idx] = self._create_trading_discretizer(
                            feature_data, feature_idx
                        )
            
        except Exception as e:
            logger.warning(f"Error setting up discretization: {e}")
    
    def _create_trading_discretizer(self, feature_data: np.ndarray, feature_idx: int) -> Dict[str, Any]:
        """Create trading-specific discretizer για feature"""
        feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
        
        # Trading-specific thresholds
        if 'rsi' in feature_name.lower():
            # RSI: oversold/neutral/overbought
            return {
                'method': 'trading_rsi',
                'thresholds': [30, 70],
                'labels': ['oversold', 'neutral', 'overbought']
            }
        elif 'macd' in feature_name.lower():
            # MACD: negative/positive
            return {
                'method': 'trading_macd',
                'thresholds': [0],
                'labels': ['negative', 'positive']
            }
        elif 'volume' in feature_name.lower():
            # Volume: low/normal/high based on historical data
            volume_50 = np.percentile(feature_data, 50)
            volume_90 = np.percentile(feature_data, 90)
            return {
                'method': 'trading_volume',
                'thresholds': [volume_50, volume_90],
                'labels': ['low', 'normal', 'high']
            }
        elif any(keyword in feature_name.lower() for keyword in ['price', 'close', 'open']):
            # Price: below/above moving averages
            price_mean = np.mean(feature_data)
            price_std = np.std(feature_data)
            return {
                'method': 'trading_price',
                'thresholds': [price_mean - price_std, price_mean, price_mean + price_std],
                'labels': ['low', 'below_mean', 'above_mean', 'high']
            }
        else:
            # Default quartile discretization
            quartiles = np.percentile(feature_data, [25, 50, 75])
            return {
                'method': 'quartile',
                'thresholds': quartiles,
                'labels': ['low', 'medium_low', 'medium_high', 'high']
            }
    
    def _init_anchor_explainer(self) -> None:
        """Initialize original anchor explainer"""
        try:
            # Create categorical names mapping
            categorical_names = {}
            for feature_idx in range(len(self.feature_names)):
                if feature_idx in self._feature_discretizers:
                    discretizer = self._feature_discretizers[feature_idx]
                    if 'labels' in discretizer:
                        categorical_names[feature_idx] = discretizer['labels']
                    else:
                        categorical_names[feature_idx] = [f"bin_{i}" for i in range(len(discretizer['thresholds']) + 1)]
            
            self._anchor_explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=['negative', 'positive'] if not hasattr(self.model, 'classes_') else self.model.classes_,
                feature_names=self.feature_names,
                train_data=self._training_data,
                categorical_names=categorical_names,
                discretize_continuous=self.config.discretize_continuous
            )
            
            logger.info("Initialized original anchor explainer")
            
        except Exception as e:
            logger.warning(f"Failed to initialize anchor explainer: {e}")
            self._anchor_explainer = None
    
    def _init_alibi_anchor_explainer(self) -> None:
        """Initialize Alibi anchor explainer"""
        try:
            self._anchor_explainer = AnchorTabular(
                self._predict_fn,
                feature_names=self.feature_names,
                categorical_names={},  # Will be populated during explain
                seed=42
            )
            
            # Fit on training data
            self._anchor_explainer.fit(self._training_data)
            
            logger.info("Initialized Alibi anchor explainer")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Alibi anchor explainer: {e}")
            self._anchor_explainer = None
    
    def _init_custom_anchor_explainer(self) -> None:
        """Initialize custom anchor explainer implementation"""
        logger.info("Using custom anchor explainer implementation")
        self._anchor_explainer = "custom"
    
    async def explain_async(
        self,
        instance: Union[np.ndarray, pd.Series],
        symbol: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> AnchorExplanation:
        """
        Async anchor explanation high-frequency trading
        
        Args:
            instance: Instance anchor generation
            symbol: Trading symbol context
            threshold: Precision threshold anchors
            
        Returns:
            Anchor explanation trading rules
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.explain,
            instance,
            symbol,
            threshold
        )
    
    def explain(
        self,
        instance: Union[np.ndarray, pd.Series],
        symbol: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> AnchorExplanation:
        """
        Generate anchor explanation για instance
        
        Args:
            instance: Input instance για anchor rules
            symbol: Trading symbol για context
            threshold: Precision threshold
            
        Returns:
            Comprehensive anchor explanation
        """
        try:
            # Prepare instance
            if isinstance(instance, pd.Series):
                instance_array = instance.values
            else:
                instance_array = instance.copy()
            
            if instance_array.ndim > 1:
                instance_array = instance_array.flatten()
            
            threshold = threshold or self.config.threshold
            
            # Get original prediction
            instance_pred = self._predict_fn(instance_array.reshape(1, -1))[0]
            instance_proba = self._predict_proba_fn(instance_array.reshape(1, -1))[0]
            instance_confidence = float(np.max(instance_proba))
            
            # Check cache
            cache_key = self._get_cache_key(instance_array, threshold, symbol)
            if self.config.cache_anchors:
                cached_explanation = self._load_from_cache(cache_key)
                if cached_explanation is not None:
                    logger.debug("Loaded anchor explanation from cache")
                    return cached_explanation
            
            # Generate anchor rules
            anchor_rules = self._generate_anchor_rules(
                instance_array, instance_pred, threshold
            )
            
            if not anchor_rules:
                logger.warning("No anchor rules found, creating fallback explanation")
                anchor_rules = [self._create_fallback_anchor(instance_array, instance_pred)]
            
            # Select primary anchor (highest precision)
            primary_anchor = max(anchor_rules, key=lambda r: r.precision)
            
            # Extract trading metadata
            trade_metadata = self._extract_trade_metadata(
                instance_array, symbol, instance_pred, instance_proba, anchor_rules
            )
            
            # Calculate rule quality metrics
            quality_metrics = self._calculate_rule_quality_metrics(anchor_rules)
            
            # Create comprehensive explanation
            explanation = AnchorExplanation(
                anchor_rules=anchor_rules,
                primary_anchor=primary_anchor,
                instance_prediction=instance_pred,
                instance_confidence=instance_confidence,
                model_type=type(self.model).__name__,
                timestamp=datetime.now(),
                symbol=symbol,
                total_rules_found=len(anchor_rules),
                **quality_metrics,
                **trade_metadata
            )
            
            # Cache results
            if self.config.cache_anchors:
                self._save_to_cache(cache_key, explanation)
            
            logger.info(f"Generated {len(anchor_rules)} anchor rules for {symbol or 'unknown symbol'}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating anchor explanation: {e}")
            raise
    
    def _generate_anchor_rules(
        self,
        instance: np.ndarray,
        instance_pred: Union[int, float],
        threshold: float
    ) -> List[AnchorRule]:
        """Generate anchor rules for instance"""
        
        if self._anchor_explainer and self._anchor_explainer != "custom":
            return self._generate_with_library(instance, instance_pred, threshold)
        else:
            return self._generate_with_custom_method(instance, instance_pred, threshold)
    
    def _generate_with_library(
        self,
        instance: np.ndarray,
        instance_pred: Union[int, float],
        threshold: float
    ) -> List[AnchorRule]:
        """Generate anchors using external library"""
        try:
            if ANCHOR_AVAILABLE and hasattr(self._anchor_explainer, 'explain_instance'):
                # Original anchor library
                explanation = self._anchor_explainer.explain_instance(
                    instance,
                    self._predict_fn,
                    threshold=threshold,
                    delta=self.config.delta,
                    tau=self.config.tau,
                    batch_size=self.config.batch_size,
                    coverage_samples=self.config.coverage_samples,
                    beam_size=self.config.beam_size
                )
                
                # Convert to our format
                anchor_rule = AnchorRule(
                    features=[self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}" 
                             for i in explanation.features()],
                    conditions=list(explanation.names()),
                    precision=float(explanation.precision()),
                    coverage=float(explanation.coverage()),
                    support=int(explanation.coverage() * self.config.coverage_samples),
                    confidence_interval=(
                        float(explanation.precision() - explanation.precision() * 0.1),
                        float(explanation.precision() + explanation.precision() * 0.1)
                    )
                )
                
                return [anchor_rule]
                
            elif ALIBI_ANCHOR_AVAILABLE and hasattr(self._anchor_explainer, 'explain'):
                # Alibi anchor explainer
                explanation = self._anchor_explainer.explain(
                    instance.reshape(1, -1),
                    threshold=threshold,
                    delta=self.config.delta,
                    tau=self.config.tau,
                    batch_size=self.config.batch_size,
                    coverage_samples=self.config.coverage_samples,
                    beam_size=self.config.beam_size,
                    epsilon=self.config.epsilon
                )
                
                if hasattr(explanation, 'anchor'):
                    anchor_features = explanation.anchor
                    anchor_names = [f"{self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'}" 
                                   for i in anchor_features]
                    
                    anchor_rule = AnchorRule(
                        features=anchor_names,
                        conditions=[f"{name} in valid range" for name in anchor_names],
                        precision=float(getattr(explanation, 'precision', threshold)),
                        coverage=float(getattr(explanation, 'coverage', 0.1)),
                        support=int(getattr(explanation, 'coverage', 0.1) * self.config.coverage_samples),
                        confidence_interval=(threshold - 0.05, threshold + 0.05)
                    )
                    
                    return [anchor_rule]
            
            return []
            
        except Exception as e:
            logger.warning(f"Error generating anchors with library: {e}")
            return []
    
    def _generate_with_custom_method(
        self,
        instance: np.ndarray,
        instance_pred: Union[int, float],
        threshold: float
    ) -> List[AnchorRule]:
        """Generate anchors using custom beam search method"""
        try:
            anchor_rules = []
            
            # Generate candidate feature combinations
            feature_indices = list(range(len(instance)))
            
            # Try different rule lengths
            for rule_length in range(1, min(self.config.max_rule_length + 1, len(feature_indices) + 1)):
                for feature_combo in itertools.combinations(feature_indices, rule_length):
                    try:
                        anchor_rule = self._evaluate_feature_combination(
                            instance, instance_pred, feature_combo, threshold
                        )
                        
                        if anchor_rule and anchor_rule.precision >= threshold:
                            anchor_rules.append(anchor_rule)
                        
                        # Limit number of rules performance
                        if len(anchor_rules) >= 20:
                            break
                            
                    except Exception as e:
                        logger.debug(f"Error evaluating combination {feature_combo}: {e}")
                        continue
                
                # If we found good rules, prioritize shorter ones
                if anchor_rules:
                    break
            
            # Sort by precision and return top rules
            anchor_rules.sort(key=lambda r: r.precision, reverse=True)
            return anchor_rules[:10]  # Return top 10 rules
            
        except Exception as e:
            logger.warning(f"Error generating anchors with custom method: {e}")
            return []
    
    def _evaluate_feature_combination(
        self,
        instance: np.ndarray,
        instance_pred: Union[int, float],
        feature_combo: Tuple[int, ...],
        threshold: float
    ) -> Optional[AnchorRule]:
        """Evaluate specific feature combination anchor rule"""
        try:
            # Generate conditions for this combination
            conditions = []
            feature_names = []
            
            for feature_idx in feature_combo:
                feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                feature_names.append(feature_name)
                
                # Create condition based on discretization
                condition = self._create_feature_condition(instance, feature_idx)
                conditions.append(condition)
            
            # Sample instances that satisfy these conditions
            matching_samples = self._sample_matching_instances(instance, feature_combo, conditions)
            
            if len(matching_samples) < self.config.min_support_samples:
                return None
            
            # Calculate precision
            predictions = []
            for sample in matching_samples:
                pred = self._predict_fn(sample.reshape(1, -1))[0]
                predictions.append(pred)
            
            if isinstance(instance_pred, (int, np.integer)):
                # Classification: fraction with same class
                correct_predictions = sum(1 for p in predictions if p == instance_pred)
                precision = correct_predictions / len(predictions)
            else:
                # Regression: fraction within tolerance
                tolerance = abs(instance_pred) * 0.1 if instance_pred != 0 else 0.1
                correct_predictions = sum(1 for p in predictions if abs(p - instance_pred) <= tolerance)
                precision = correct_predictions / len(predictions)
            
            # Calculate coverage (simplified)
            coverage = len(matching_samples) / self.config.coverage_samples
            
            # Calculate confidence interval (simplified)
            ci_margin = 1.96 * np.sqrt(precision * (1 - precision) / len(matching_samples))
            confidence_interval = (max(0, precision - ci_margin), min(1, precision + ci_margin))
            
            if precision >= threshold:
                anchor_rule = AnchorRule(
                    features=feature_names,
                    conditions=conditions,
                    precision=float(precision),
                    coverage=float(coverage),
                    support=len(matching_samples),
                    confidence_interval=confidence_interval
                )
                
                # Add trading metadata
                self._enrich_anchor_with_trading_info(anchor_rule, matching_samples, predictions)
                
                return anchor_rule
            
            return None
            
        except Exception as e:
            logger.debug(f"Error evaluating feature combination: {e}")
            return None
    
    def _create_feature_condition(self, instance: np.ndarray, feature_idx: int) -> str:
        """Create condition string feature"""
        feature_value = instance[feature_idx]
        feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
        
        if feature_idx in self._feature_discretizers:
            discretizer = self._feature_discretizers[feature_idx]
            
            # Find which bin this value falls into
            if 'thresholds' in discretizer:
                thresholds = discretizer['thresholds']
                bin_idx = sum(1 for t in thresholds if feature_value > t)
                
                if 'labels' in discretizer and bin_idx < len(discretizer['labels']):
                    label = discretizer['labels'][bin_idx]
                    return f"{feature_name} is {label}"
                else:
                    if bin_idx == 0:
                        return f"{feature_name} <= {thresholds[0]:.4f}"
                    elif bin_idx >= len(thresholds):
                        return f"{feature_name} > {thresholds[-1]:.4f}"
                    else:
                        return f"{thresholds[bin_idx-1]:.4f} < {feature_name} <= {thresholds[bin_idx]:.4f}"
        
        # Fallback: simple range condition
        tolerance = abs(feature_value) * 0.1 if feature_value != 0 else 0.1
        return f"{feature_name} ∈ [{feature_value - tolerance:.4f}, {feature_value + tolerance:.4f}]"
    
    def _sample_matching_instances(
        self,
        instance: np.ndarray,
        feature_combo: Tuple[int, ...],
        conditions: List[str]
    ) -> List[np.ndarray]:
        """Sample instances that match the anchor conditions"""
        matching_samples = []
        max_attempts = self.config.coverage_samples * 2
        
        for _ in range(max_attempts):
            # Start with random sample from training data
            if self._training_data is not None and len(self._training_data) > 0:
                base_idx = np.random.randint(0, len(self._training_data))
                sample = self._training_data[base_idx].copy()
            else:
                # Fallback: perturb original instance
                sample = instance.copy()
                noise = np.random.normal(0, 0.1, sample.shape)
                sample += noise
            
            # Modify features outside the combo to create variation
            for i in range(len(sample)):
                if i not in feature_combo:
                    # Add some randomness
                    if self._training_data is not None:
                        feature_data = self._training_data[:, i]
                        sample[i] = np.random.choice(feature_data)
                    else:
                        sample[i] += np.random.normal(0, abs(sample[i]) * 0.2)
            
            # Keep anchor features from original instance (approximately)
            for feature_idx in feature_combo:
                if self._matches_condition(sample[feature_idx], instance[feature_idx], feature_idx):
                    continue
                else:
                    # Adjust to match condition
                    sample[feature_idx] = instance[feature_idx] + np.random.normal(0, abs(instance[feature_idx]) * 0.05)
            
            matching_samples.append(sample)
            
            if len(matching_samples) >= self.config.coverage_samples:
                break
        
        return matching_samples
    
    def _matches_condition(self, value: float, target_value: float, feature_idx: int) -> bool:
        """Check if value matches the condition for feature"""
        if feature_idx in self._feature_discretizers:
            discretizer = self._feature_discretizers[feature_idx]
            
            if 'thresholds' in discretizer:
                # Same discretization bin
                thresholds = discretizer['thresholds']
                value_bin = sum(1 for t in thresholds if value > t)
                target_bin = sum(1 for t in thresholds if target_value > t)
                return value_bin == target_bin
        
        # Fallback: within tolerance
        tolerance = abs(target_value) * 0.1 if target_value != 0 else 0.1
        return abs(value - target_value) <= tolerance
    
    def _enrich_anchor_with_trading_info(
        self,
        anchor_rule: AnchorRule,
        matching_samples: List[np.ndarray],
        predictions: List[Union[int, float]]
    ) -> None:
        """Enrich anchor rule with trading-specific information"""
        try:
            # Determine trading signal
            if isinstance(predictions[0], (int, np.integer)):
                # Classification
                prediction_counts = defaultdict(int)
                for pred in predictions:
                    prediction_counts[pred] += 1
                
                most_common_pred = max(prediction_counts, key=prediction_counts.get)
                
                if len(prediction_counts) == 2:
                    anchor_rule.trading_signal = "BUY" if most_common_pred == 1 else "SELL"
                elif len(prediction_counts) == 3:
                    anchor_rule.trading_signal = ["SELL", "HOLD", "BUY"][most_common_pred]
                else:
                    anchor_rule.trading_signal = f"CLASS_{most_common_pred}"
            else:
                # Regression
                avg_prediction = np.mean(predictions)
                if avg_prediction > 0.1:
                    anchor_rule.trading_signal = "BUY"
                elif avg_prediction < -0.1:
                    anchor_rule.trading_signal = "SELL"
                else:
                    anchor_rule.trading_signal = "HOLD"
            
            # Analyze market conditions from features
            market_conditions = []
            for feature_name in anchor_rule.features:
                if 'rsi' in feature_name.lower():
                    market_conditions.append("RSI_based")
                elif 'volume' in feature_name.lower():
                    market_conditions.append("Volume_based")
                elif any(word in feature_name.lower() for word in ['price', 'close', 'sma', 'ema']):
                    market_conditions.append("Price_based")
                elif 'macd' in feature_name.lower():
                    market_conditions.append("Momentum_based")
            
            anchor_rule.market_conditions = market_conditions if market_conditions else ["General"]
            
            # Risk level based on precision coverage
            if anchor_rule.precision >= 0.95 and anchor_rule.coverage >= 0.1:
                anchor_rule.risk_level = "LOW"
            elif anchor_rule.precision >= 0.85 and anchor_rule.coverage >= 0.05:
                anchor_rule.risk_level = "MEDIUM"
            else:
                anchor_rule.risk_level = "HIGH"
            
            # Profitability estimate (simplified)
            if anchor_rule.precision >= 0.9:
                anchor_rule.profitability_estimate = 0.7
            elif anchor_rule.precision >= 0.8:
                anchor_rule.profitability_estimate = 0.5
            else:
                anchor_rule.profitability_estimate = 0.3
            
        except Exception as e:
            logger.warning(f"Error enriching anchor with trading info: {e}")
    
    def _create_fallback_anchor(
        self,
        instance: np.ndarray,
        instance_pred: Union[int, float]
    ) -> AnchorRule:
        """Create fallback anchor when no good rules found"""
        # Use most important feature (if available)
        if hasattr(self.model, 'feature_importances_'):
            most_important_idx = np.argmax(self.model.feature_importances_)
        else:
            # Random feature as fallback
            most_important_idx = np.random.randint(0, len(instance))
        
        feature_name = self.feature_names[most_important_idx] if most_important_idx < len(self.feature_names) else f"feature_{most_important_idx}"
        condition = self._create_feature_condition(instance, most_important_idx)
        
        return AnchorRule(
            features=[feature_name],
            conditions=[condition],
            precision=0.5,  # Low precision fallback
            coverage=0.1,
            support=50,
            confidence_interval=(0.4, 0.6),
            trading_signal="HOLD",
            market_conditions=["Fallback"],
            risk_level="HIGH",
            profitability_estimate=0.3
        )
    
    def _extract_trade_metadata(
        self,
        instance: np.ndarray,
        symbol: Optional[str],
        instance_pred: Union[int, float],
        instance_proba: np.ndarray,
        anchor_rules: List[AnchorRule]
    ) -> Dict[str, Any]:
        """Extract trading-specific metadata"""
        metadata = {}
        
        try:
            # Primary trade signal
            if isinstance(instance_pred, (int, np.integer)):
                if len(instance_proba) == 2:
                    metadata['trade_signal'] = "BUY" if instance_pred == 1 else "SELL"
                elif len(instance_proba) == 3:
                    metadata['trade_signal'] = ["SELL", "HOLD", "BUY"][instance_pred]
                else:
                    metadata['trade_signal'] = f"CLASS_{instance_pred}"
            else:
                if instance_pred > 0.1:
                    metadata['trade_signal'] = "BUY"
                elif instance_pred < -0.1:
                    metadata['trade_signal'] = "SELL"
                else:
                    metadata['trade_signal'] = "HOLD"
            
            # Market regime analysis
            regime_indicators = []
            for i, feature_name in enumerate(self.feature_names):
                if i < len(instance):
                    if 'volatility' in feature_name.lower() and instance[i] > 0.5:
                        regime_indicators.append("HIGH_VOLATILITY")
                    elif 'volume' in feature_name.lower() and instance[i] > np.mean(self._training_data[:, i]) if self._training_data is not None else True:
                        regime_indicators.append("HIGH_VOLUME")
                    elif 'rsi' in feature_name.lower():
                        if instance[i] < 30:
                            regime_indicators.append("OVERSOLD")
                        elif instance[i] > 70:
                            regime_indicators.append("OVERBOUGHT")
            
            if regime_indicators:
                metadata['market_regime'] = "_".join(regime_indicators[:3])  # Limit to 3 indicators
            else:
                metadata['market_regime'] = "NORMAL"
            
        except Exception as e:
            logger.warning(f"Error extracting trade metadata: {e}")
        
        return metadata
    
    def _calculate_rule_quality_metrics(self, anchor_rules: List[AnchorRule]) -> Dict[str, float]:
        """Calculate overall rule quality metrics"""
        if not anchor_rules:
            return {
                'average_precision': 0.0,
                'average_coverage': 0.0,
                'rule_diversity_score': 0.0
            }
        
        # Average precision coverage
        avg_precision = float(np.mean([rule.precision for rule in anchor_rules]))
        avg_coverage = float(np.mean([rule.coverage for rule in anchor_rules]))
        
        # Rule diversity (based on different feature combinations)
        unique_feature_sets = set()
        for rule in anchor_rules:
            feature_set = frozenset(rule.features)
            unique_feature_sets.add(feature_set)
        
        diversity_score = len(unique_feature_sets) / len(anchor_rules) if anchor_rules else 0.0
        
        return {
            'average_precision': avg_precision,
            'average_coverage': avg_coverage,
            'rule_diversity_score': float(diversity_score)
        }
    
    def analyze_rule_consistency(
        self,
        explanations: List[AnchorExplanation]
    ) -> Dict[str, Any]:
        """Analyze consistency of anchor rules across explanations"""
        if not explanations:
            return {}
        
        try:
            consistency_analysis = {
                'total_explanations': len(explanations),
                'rule_overlap_analysis': {},
                'feature_frequency': defaultdict(int),
                'signal_consistency': defaultdict(int),
                'average_rule_quality': {}
            }
            
            # Collect all rules
            all_rules = []
            for explanation in explanations:
                all_rules.extend(explanation.anchor_rules)
            
            # Feature frequency analysis
            for rule in all_rules:
                for feature in rule.features:
                    consistency_analysis['feature_frequency'][feature] += 1
            
            # Signal consistency
            for rule in all_rules:
                if rule.trading_signal:
                    consistency_analysis['signal_consistency'][rule.trading_signal] += 1
            
            # Rule overlap (simplified)
            feature_combinations = []
            for rule in all_rules:
                feature_combinations.append(frozenset(rule.features))
            
            unique_combinations = set(feature_combinations)
            overlap_score = 1.0 - (len(unique_combinations) / len(feature_combinations)) if feature_combinations else 0
            
            consistency_analysis['rule_overlap_analysis'] = {
                'total_rules': len(all_rules),
                'unique_feature_combinations': len(unique_combinations),
                'overlap_score': float(overlap_score)
            }
            
            # Average rule quality
            if all_rules:
                consistency_analysis['average_rule_quality'] = {
                    'precision': float(np.mean([rule.precision for rule in all_rules])),
                    'coverage': float(np.mean([rule.coverage for rule in all_rules])),
                    'support': float(np.mean([rule.support for rule in all_rules]))
                }
            
            return consistency_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing rule consistency: {e}")
            return {}
    
    def generate_rule_summary(self, anchor_rules: List[AnchorRule]) -> Dict[str, Any]:
        """Generate human-readable summary of anchor rules"""
        if not anchor_rules:
            return {}
        
        try:
            summary = {
                'total_rules': len(anchor_rules),
                'high_precision_rules': len([r for r in anchor_rules if r.precision >= 0.9]),
                'trading_signals': {},
                'market_conditions': {},
                'risk_levels': {},
                'top_features': {},
                'rule_descriptions': []
            }
            
            # Aggregate trading signals
            signal_counts = defaultdict(int)
            for rule in anchor_rules:
                if rule.trading_signal:
                    signal_counts[rule.trading_signal] += 1
            summary['trading_signals'] = dict(signal_counts)
            
            # Aggregate market conditions
            condition_counts = defaultdict(int)
            for rule in anchor_rules:
                if rule.market_conditions:
                    for condition in rule.market_conditions:
                        condition_counts[condition] += 1
            summary['market_conditions'] = dict(condition_counts)
            
            # Aggregate risk levels
            risk_counts = defaultdict(int)
            for rule in anchor_rules:
                if rule.risk_level:
                    risk_counts[rule.risk_level] += 1
            summary['risk_levels'] = dict(risk_counts)
            
            # Top features
            feature_counts = defaultdict(int)
            for rule in anchor_rules:
                for feature in rule.features:
                    feature_counts[feature] += 1
            
            # Sort by frequency
            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
            summary['top_features'] = dict(sorted_features[:10])
            
            # Rule descriptions (top 5 rules)
            top_rules = sorted(anchor_rules, key=lambda r: r.precision, reverse=True)[:5]
            for i, rule in enumerate(top_rules):
                description = {
                    'rank': i + 1,
                    'precision': rule.precision,
                    'coverage': rule.coverage,
                    'conditions': rule.conditions,
                    'trading_signal': rule.trading_signal,
                    'risk_level': rule.risk_level
                }
                summary['rule_descriptions'].append(description)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating rule summary: {e}")
            return {}
    
    def _get_cache_key(
        self,
        instance: np.ndarray,
        threshold: float,
        symbol: Optional[str]
    ) -> str:
        """Generate cache key για anchor explanation"""
        instance_hash = hash(instance.tobytes())
        model_hash = hash(str(type(self.model)))
        threshold_hash = hash(str(threshold))
        symbol_hash = hash(symbol or "")
        return f"anchor_{model_hash}_{symbol_hash}_{instance_hash}_{threshold_hash}"
    
    def _save_to_cache(self, cache_key: str, explanation: AnchorExplanation) -> None:
        """Save explanation to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(explanation, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[AnchorExplanation]:
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
            
            logger.info(f"Cleaned up {deleted_count} old anchor cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0