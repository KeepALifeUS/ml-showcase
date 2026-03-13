"""
SHAP (SHapley Additive exPlanations) Explainer Crypto Trading Bot v5.0

 comprehensive SHAP-based ML .
enterprise patterns interpretability transparency.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Crypto trading specific imports
try:
    import torch
    import tensorflow as tf
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SHAPConfig:
    """Configuration SHAP explainer enterprise patterns"""
    explainer_type: str = 'tree'  # 'tree', 'deep', 'linear', 'kernel', 'permutation'
    max_evals: int = 100
    feature_perturbation: str = 'interventional'
    model_output: str = 'probability'
    batch_size: int = 32
    background_size: int = 100
    check_additivity: bool = False
    algorithm: str = 'auto'
    # Enterprise patterns
    cache_explanations: bool = True
    enable_gpu: bool = True
    parallel_workers: int = 4
    timeout_seconds: int = 300


@dataclass 
class SHAPExplanation:
    """Structured SHAP explanation metadata"""
    shap_values: np.ndarray
    base_values: Union[float, np.ndarray]
    data: np.ndarray
    feature_names: List[str]
    expected_value: float
    model_type: str
    explanation_type: str
    timestamp: datetime
    # Crypto trading specific
    symbol: Optional[str] = None
    prediction_confidence: Optional[float] = None
    trade_signal: Optional[str] = None
    risk_factors: Optional[Dict[str, float]] = None
    #  enterprise metadata
    model_version: Optional[str] = None
    data_version: Optional[str] = None
    compliance_flags: Optional[List[str]] = None


class CryptoTradingSHAPExplainer:
    """
    Enterprise-grade SHAP explainer crypto trading models
    
    Provides comprehensive model interpretability :
    - Trading signal predictions
    - Risk assessment models  
    - Portfolio optimization
    - Market regime detection
    - Anomaly detection in trading patterns
    
    enterprise patterns:
    - Async processing high-frequency explanations
    - Caching performance optimization
    - Enterprise monitoring logging
    - Compliance tracking regulatory requirements
    """
    
    def __init__(
        self,
        model: BaseEstimator,
        config: Optional[SHAPConfig] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize SHAP explainer enterprise configuration"""
        self.model = model
        self.config = config or SHAPConfig()
        self.cache_dir = cache_dir or Path("./cache/shap_explanations")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize explainer based on model type
        self._explainer: Optional[shap.Explainer] = None
        self._background_data: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        logger.info(f"Initialized SHAP explainer with config: {self.config}")
    
    def fit(
        self,
        background_data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit SHAP explainer on background data
        
        Args:
            background_data: Baseline SHAP calculations
            feature_names: features interpretability
        """
        try:
            # Prepare background data
            if isinstance(background_data, pd.DataFrame):
                self._feature_names = list(background_data.columns)
                background_array = background_data.values
            else:
                self._feature_names = feature_names or [f"feature_{i}" for i in range(background_data.shape[1])]
                background_array = background_data
            
            # Sample background data if too large performance
            if len(background_array) > self.config.background_size:
                indices = np.random.choice(
                    len(background_array), 
                    size=self.config.background_size,
                    replace=False
                )
                background_array = background_array[indices]
            
            self._background_data = background_array
            
            # Initialize appropriate SHAP explainer
            self._init_explainer(background_array)
            
            logger.info(f"SHAP explainer fitted with {len(background_array)} background samples")
            
        except Exception as e:
            logger.error(f"Error fitting SHAP explainer: {e}")
            raise
    
    def _init_explainer(self, background_data: np.ndarray) -> None:
        """Initialize appropriate SHAP explainer based on model type"""
        model_type = type(self.model).__name__
        
        try:
            # Tree-based models (XGBoost, LightGBM, RandomForest, etc.)
            if hasattr(self.model, 'feature_importances_') or 'Tree' in model_type or 'Forest' in model_type:
                logger.info("Using TreeExplainer for tree-based model")
                self._explainer = shap.TreeExplainer(
                    self.model,
                    feature_perturbation=self.config.feature_perturbation,
                    model_output=self.config.model_output,
                    check_additivity=self.config.check_additivity
                )
            
            # Deep learning models
            elif DEEP_LEARNING_AVAILABLE and (hasattr(self.model, 'layers') or 'torch' in str(type(self.model))):
                logger.info("Using DeepExplainer for deep learning model") 
                self._explainer = shap.DeepExplainer(self.model, background_data)
            
            # Linear models
            elif hasattr(self.model, 'coef_'):
                logger.info("Using LinearExplainer for linear model")
                self._explainer = shap.LinearExplainer(self.model, background_data)
            
            # Fallback to KernelExplainer for other models
            else:
                logger.info("Using KernelExplainer as fallback")
                self._explainer = shap.KernelExplainer(
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    background_data,
                    algorithm=self.config.algorithm
                )
                
        except Exception as e:
            logger.warning(f"Error initializing specific explainer: {e}. Falling back to KernelExplainer")
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                background_data
            )
    
    async def explain_async(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        symbol: Optional[str] = None,
        max_evals: Optional[int] = None
    ) -> SHAPExplanation:
        """
        Async SHAP explanation high-frequency trading
        
        Args:
            data: Input data
            symbol: Trading symbol context
            max_evals: Maximum evaluations performance control
            
        Returns:
            Structured SHAP explanation trading metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.explain,
            data,
            symbol,
            max_evals
        )
    
    def explain(
        self,
        data: Union[np.ndarray, pd.DataFrame], 
        symbol: Optional[str] = None,
        max_evals: Optional[int] = None
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation data
        
        Args:
            data: Input data
            symbol: Trading symbol context
            max_evals: Maximum evaluations
            
        Returns:
            Comprehensive SHAP explanation
        """
        if self._explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        try:
            # Prepare input data
            if isinstance(data, pd.DataFrame):
                data_array = data.values
            else:
                data_array = data
            
            # Ensure correct shape
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            
            # Check cache first performance
            cache_key = self._get_cache_key(data_array, symbol)
            if self.config.cache_explanations:
                cached_explanation = self._load_from_cache(cache_key)
                if cached_explanation is not None:
                    logger.debug("Loaded explanation from cache")
                    return cached_explanation
            
            # Calculate SHAP values
            max_evals = max_evals or self.config.max_evals
            
            if hasattr(self._explainer, 'expected_value'):
                # TreeExplainer, LinearExplainer
                shap_values = self._explainer.shap_values(data_array)
                expected_value = self._explainer.expected_value
            else:
                # KernelExplainer, DeepExplainer
                shap_values = self._explainer.shap_values(
                    data_array, 
                    nsamples=max_evals
                )
                expected_value = getattr(self._explainer, 'expected_value', 0.0)
            
            # Handle multi-class outputs
            if isinstance(shap_values, list):
                # Multi-class: use positive class class with highest prediction
                prediction = self.model.predict_proba(data_array)[0] if hasattr(self.model, 'predict_proba') else self.model.predict(data_array)
                if isinstance(prediction, np.ndarray) and len(prediction) > 1:
                    class_idx = np.argmax(prediction)
                    shap_values = shap_values[class_idx]
                    if isinstance(expected_value, list):
                        expected_value = expected_value[class_idx]
                else:
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                    expected_value = expected_value[1] if isinstance(expected_value, list) and len(expected_value) > 1 else expected_value
            
            # Calculate base values
            if hasattr(self._explainer, 'expected_value'):
                base_values = self._explainer.expected_value
            else:
                base_values = expected_value
            
            # Get trading specific metadata
            trade_metadata = self._get_trade_metadata(data_array, symbol)
            
            # Create comprehensive explanation
            explanation = SHAPExplanation(
                shap_values=shap_values,
                base_values=base_values,
                data=data_array,
                feature_names=self._feature_names.copy(),
                expected_value=float(expected_value),
                model_type=type(self.model).__name__,
                explanation_type="shap",
                timestamp=datetime.now(),
                symbol=symbol,
                **trade_metadata
            )
            
            # Cache future requests
            if self.config.cache_explanations:
                self._save_to_cache(cache_key, explanation)
            
            logger.info(f"Generated SHAP explanation for {symbol or 'unknown symbol'}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            raise
    
    def explain_batch(
        self,
        data_batch: Union[List[np.ndarray], np.ndarray, pd.DataFrame],
        symbols: Optional[List[str]] = None,
        batch_size: Optional[int] = None
    ) -> List[SHAPExplanation]:
        """
        Batch SHAP explanations high-throughput processing
        
        Args:
            data_batch: Batch of input data
            symbols: Trading symbols each sample
            batch_size: Processing batch size
            
        Returns:
            List of SHAP explanations
        """
        if isinstance(data_batch, (pd.DataFrame, np.ndarray)):
            if data_batch.ndim == 1:
                data_batch = [data_batch]
            elif data_batch.ndim == 2:
                data_batch = [data_batch[i] for i in range(len(data_batch))]
        
        batch_size = batch_size or self.config.batch_size
        symbols = symbols or [None] * len(data_batch)
        
        explanations = []
        for i in range(0, len(data_batch), batch_size):
            batch = data_batch[i:i + batch_size]
            batch_symbols = symbols[i:i + batch_size]
            
            for data, symbol in zip(batch, batch_symbols):
                explanation = self.explain(data, symbol)
                explanations.append(explanation)
        
        return explanations
    
    def _get_trade_metadata(self, data: np.ndarray, symbol: Optional[str]) -> Dict[str, Any]:
        """Extract trading specific metadata from prediction"""
        metadata = {}
        
        try:
            # Get model prediction confidence
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(data.reshape(1, -1))[0]
                prediction_confidence = float(np.max(prediction_proba))
                prediction_class = int(np.argmax(prediction_proba))
                
                # Map to trading signals
                if len(prediction_proba) == 2:  # Binary classification
                    trade_signal = "BUY" if prediction_class == 1 else "SELL"
                elif len(prediction_proba) == 3:  # Three-class
                    trade_signal = ["SELL", "HOLD", "BUY"][prediction_class]
                else:
                    trade_signal = f"CLASS_{prediction_class}"
                
                metadata.update({
                    'prediction_confidence': prediction_confidence,
                    'trade_signal': trade_signal
                })
            
            # Extract risk factors from feature values
            if len(self._feature_names) > 0:
                risk_features = [name for name in self._feature_names if 'risk' in name.lower() or 'volatility' in name.lower()]
                if risk_features:
                    risk_factors = {}
                    for i, feature_name in enumerate(self._feature_names):
                        if feature_name in risk_features and i < len(data[0]):
                            risk_factors[feature_name] = float(data[0][i])
                    metadata['risk_factors'] = risk_factors
            
        except Exception as e:
            logger.warning(f"Error extracting trade metadata: {e}")
        
        return metadata
    
    def _get_cache_key(self, data: np.ndarray, symbol: Optional[str]) -> str:
        """Generate unique cache key explanation"""
        data_hash = hash(data.tobytes())
        model_hash = hash(str(type(self.model)))
        symbol_hash = hash(symbol or "")
        return f"shap_{model_hash}_{symbol_hash}_{data_hash}"
    
    def _save_to_cache(self, cache_key: str, explanation: SHAPExplanation) -> None:
        """Save explanation to cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(explanation, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[SHAPExplanation]:
        """Load explanation from cache"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")
        return None
    
    def get_feature_importance(
        self,
        explanations: List[SHAPExplanation],
        aggregation: str = 'mean_abs'
    ) -> pd.DataFrame:
        """
        Calculate global feature importance across explanations
        
        Args:
            explanations: List of SHAP explanations
            aggregation: Aggregation method ('mean_abs', 'mean', 'median_abs', 'std')
            
        Returns:
            DataFrame feature importance rankings
        """
        if not explanations:
            raise ValueError("No explanations provided")
        
        # Collect all SHAP values
        all_shap_values = []
        for explanation in explanations:
            if explanation.shap_values.ndim == 1:
                all_shap_values.append(explanation.shap_values)
            else:
                # Handle batch explanations
                for i in range(explanation.shap_values.shape[0]):
                    all_shap_values.append(explanation.shap_values[i])
        
        shap_matrix = np.array(all_shap_values)
        feature_names = explanations[0].feature_names
        
        # Calculate importance based on aggregation method
        if aggregation == 'mean_abs':
            importance = np.mean(np.abs(shap_matrix), axis=0)
        elif aggregation == 'mean':
            importance = np.mean(shap_matrix, axis=0)
        elif aggregation == 'median_abs':
            importance = np.median(np.abs(shap_matrix), axis=0)
        elif aggregation == 'std':
            importance = np.std(shap_matrix, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Create results DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'rank': range(1, len(feature_names) + 1)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def analyze_feature_interactions(
        self,
        data: np.ndarray,
        feature_pairs: Optional[List[Tuple[int, int]]] = None,
        max_pairs: int = 10
    ) -> Dict[Tuple[str, str], float]:
        """
        Analyze feature interactions using SHAP interaction values
        
        Args:
            data: Input data analysis
            feature_pairs: Specific feature pairs analysis
            max_pairs: Maximum number of top interactions
            
        Returns:
            Dictionary of feature pair interactions
        """
        if not hasattr(self._explainer, 'shap_interaction_values'):
            logger.warning("Interaction values not available for this explainer type")
            return {}
        
        try:
            interaction_values = self._explainer.shap_interaction_values(data)
            
            # Calculate interaction strengths
            interactions = {}
            n_features = len(self._feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if feature_pairs is None or (i, j) in feature_pairs:
                        # Average absolute interaction value
                        interaction_strength = np.mean(np.abs(interaction_values[:, i, j]))
                        feature_pair = (self._feature_names[i], self._feature_names[j])
                        interactions[feature_pair] = float(interaction_strength)
            
            # Return top interactions
            sorted_interactions = dict(
                sorted(interactions.items(), key=lambda x: x[1], reverse=True)[:max_pairs]
            )
            
            return sorted_interactions
            
        except Exception as e:
            logger.error(f"Error analyzing feature interactions: {e}")
            return {}
    
    def generate_model_summary(self, explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Generate comprehensive model interpretability summary"""
        if not explanations:
            return {}
        
        try:
            # Feature importance
            feature_importance = self.get_feature_importance(explanations)
            
            # Model behavior statistics
            all_shap_values = np.vstack([exp.shap_values for exp in explanations])
            
            summary = {
                'model_type': explanations[0].model_type,
                'total_explanations': len(explanations),
                'explanation_period': {
                    'start': min(exp.timestamp for exp in explanations).isoformat(),
                    'end': max(exp.timestamp for exp in explanations).isoformat()
                },
                'feature_statistics': {
                    'total_features': len(explanations[0].feature_names),
                    'top_features': feature_importance.head(10).to_dict('records'),
                    'shap_value_stats': {
                        'mean_abs_shap': float(np.mean(np.abs(all_shap_values))),
                        'std_shap': float(np.std(all_shap_values)),
                        'max_abs_shap': float(np.max(np.abs(all_shap_values))),
                        'feature_sparsity': float(np.mean(all_shap_values == 0))
                    }
                },
                'trading_metrics': self._calculate_trading_metrics(explanations),
                'compliance_summary': self._generate_compliance_summary(explanations)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating model summary: {e}")
            return {}
    
    def _calculate_trading_metrics(self, explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Calculate trading specific metrics from explanations"""
        metrics = {}
        
        try:
            # Trading signal distribution
            signals = [exp.trade_signal for exp in explanations if exp.trade_signal]
            if signals:
                from collections import Counter
                signal_counts = Counter(signals)
                metrics['signal_distribution'] = dict(signal_counts)
            
            # Confidence statistics
            confidences = [exp.prediction_confidence for exp in explanations if exp.prediction_confidence]
            if confidences:
                metrics['confidence_stats'] = {
                    'mean_confidence': float(np.mean(confidences)),
                    'std_confidence': float(np.std(confidences)),
                    'min_confidence': float(np.min(confidences)),
                    'max_confidence': float(np.max(confidences))
                }
            
            # Symbol distribution
            symbols = [exp.symbol for exp in explanations if exp.symbol]
            if symbols:
                from collections import Counter
                symbol_counts = Counter(symbols)
                metrics['symbol_distribution'] = dict(symbol_counts)
            
        except Exception as e:
            logger.warning(f"Error calculating trading metrics: {e}")
        
        return metrics
    
    def _generate_compliance_summary(self, explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Generate compliance and regulatory summary"""
        compliance = {
            'explanation_coverage': len(explanations),
            'model_transparency_score': self._calculate_transparency_score(explanations),
            'bias_detection': self._detect_bias_indicators(explanations),
            'auditability': {
                'cached_explanations': len(list(self.cache_dir.glob("*.pkl"))),
                'traceable_decisions': len([exp for exp in explanations if exp.trade_signal]),
                'metadata_completeness': len([exp for exp in explanations if exp.compliance_flags])
            }
        }
        
        return compliance
    
    def _calculate_transparency_score(self, explanations: List[SHAPExplanation]) -> float:
        """Calculate model transparency score (0-100)"""
        try:
            # Factors: feature interpretability, explanation consistency, coverage
            factors = []
            
            # Feature interpretability (more human-readable features = higher score)
            readable_features = len([name for name in self._feature_names 
                                   if any(keyword in name.lower() for keyword in ['price', 'volume', 'rsi', 'macd', 'sma'])])
            interpretability_score = min(readable_features / len(self._feature_names) * 100, 100) if self._feature_names else 0
            factors.append(interpretability_score)
            
            # Explanation consistency (lower std deviation = higher consistency)
            if len(explanations) > 1:
                all_shap_values = np.vstack([exp.shap_values for exp in explanations])
                consistency_score = max(0, 100 - np.std(all_shap_values) * 10)
                factors.append(consistency_score)
            
            # Coverage (percentage of features with non-zero importance)
            all_shap_values = np.vstack([exp.shap_values for exp in explanations])
            active_features = np.sum(np.any(all_shap_values != 0, axis=0))
            coverage_score = active_features / len(self._feature_names) * 100 if self._feature_names else 0
            factors.append(coverage_score)
            
            return float(np.mean(factors))
            
        except Exception as e:
            logger.warning(f"Error calculating transparency score: {e}")
            return 0.0
    
    def _detect_bias_indicators(self, explanations: List[SHAPExplanation]) -> Dict[str, Any]:
        """Detect potential bias indicators explanations"""
        bias_indicators = {
            'feature_dominance': {},
            'symbol_bias': {},
            'temporal_drift': False
        }
        
        try:
            # Feature dominance (one feature contributing too much)
            feature_importance = self.get_feature_importance(explanations, 'mean_abs')
            if len(feature_importance) > 0:
                max_importance = feature_importance.iloc[0]['importance']
                total_importance = feature_importance['importance'].sum()
                dominance_ratio = max_importance / total_importance if total_importance > 0 else 0
                
                bias_indicators['feature_dominance'] = {
                    'dominant_feature': feature_importance.iloc[0]['feature'],
                    'dominance_ratio': float(dominance_ratio),
                    'is_concerning': dominance_ratio > 0.5
                }
            
            # Symbol bias (predictions vary significantly by symbol)
            symbol_predictions = {}
            for exp in explanations:
                if exp.symbol and exp.prediction_confidence:
                    if exp.symbol not in symbol_predictions:
                        symbol_predictions[exp.symbol] = []
                    symbol_predictions[exp.symbol].append(exp.prediction_confidence)
            
            if len(symbol_predictions) > 1:
                symbol_stds = {symbol: np.std(confidences) for symbol, confidences in symbol_predictions.items()}
                max_std = max(symbol_stds.values())
                min_std = min(symbol_stds.values())
                
                bias_indicators['symbol_bias'] = {
                    'max_std_symbol': max(symbol_stds, key=symbol_stds.get),
                    'std_difference': float(max_std - min_std),
                    'is_concerning': (max_std - min_std) > 0.2
                }
            
            # Temporal drift (explanation patterns change over time)
            if len(explanations) > 10:
                timestamps = [exp.timestamp for exp in explanations]
                sorted_explanations = sorted(zip(timestamps, explanations))
                
                # Compare first and last quartiles
                n_quarter = len(sorted_explanations) // 4
                first_quarter = [exp for _, exp in sorted_explanations[:n_quarter]]
                last_quarter = [exp for _, exp in sorted_explanations[-n_quarter:]]
                
                first_importance = self.get_feature_importance(first_quarter, 'mean_abs')
                last_importance = self.get_feature_importance(last_quarter, 'mean_abs')
                
                # Calculate feature ranking correlation
                merged = first_importance.merge(last_importance, on='feature', suffixes=('_first', '_last'))
                if len(merged) > 3:
                    correlation = merged['rank_first'].corr(merged['rank_last'])
                    bias_indicators['temporal_drift'] = float(correlation) < 0.7
            
        except Exception as e:
            logger.warning(f"Error detecting bias indicators: {e}")
        
        return bias_indicators
    
    def cleanup_cache(self, max_age_days: int = 30) -> int:
        """Cleanup old cached explanations"""
        try:
            deleted_count = 0
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old cache files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0