"""
Novelty Detection System for crypto trading environments.

Implements advanced methods for detection novel states and patterns
with enterprise patterns for real-time anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque, defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NoveltyDetectionConfig:
    """Configuration for novelty detection system."""
    
    # Detection methods
    detection_methods: List[str] = field(default_factory=lambda: [
        "autoencoder", "isolation_forest", "one_class_svm", "lof"
    ])
    method_weights: Dict[str, float] = field(default_factory=lambda: {
        "autoencoder": 0.4,
        "isolation_forest": 0.3,
        "one_class_svm": 0.2,
        "lof": 0.1
    })
    
    # Autoencoder parameters
    autoencoder_latent_dim: int = 32
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    reconstruction_threshold_percentile: float = 95.0
    
    # Ensemble parameters
    ensemble_voting: str = "weighted"  # "majority", "weighted", "average"
    confidence_threshold: float = 0.7
    
    # Crypto-specific parameters
    market_regime_aware: bool = True
    temporal_context_length: int = 10
    volatility_adjustment: bool = True
    portfolio_novelty_weight: float = 0.3
    
    # Adaptive parameters
    adaptive_thresholds: bool = True
    threshold_update_frequency: int = 1000
    false_positive_tolerance: float = 0.05
    
    # Performance optimization
    batch_processing: bool = True
    max_batch_size: int = 1000
    parallel_detection: bool = True
    caching_enabled: bool = True
    
    #  enterprise settings
    real_time_detection: bool = True
    distributed_processing: bool = True
    anomaly_storage: bool = True
    alert_system: bool = True


class NoveltyDetector(ABC):
    """
    Abstract base class for novelty detection methods.
    
    Applies design pattern "Strategy Pattern" for
    flexible novelty detection approaches.
    """
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Training detector on normal data."""
        pass
    
    @abstractmethod
    def predict_novelty(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prediction novelty scores.
        
        Returns:
            Tuple (novelty_scores, is_novel_binary)
        """
        pass
    
    @abstractmethod
    def update(self, data: np.ndarray, is_novel: Optional[np.ndarray] = None) -> None:
        """Online update detector."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        pass


class AutoencoderNoveltyDetector(NoveltyDetector):
    """
    Autoencoder-based novelty detection.
    
    Uses design pattern "Representation Learning" for
    detection through reconstruction error.
    """
    
    def __init__(self, config: NoveltyDetectionConfig, input_dim: int, device: str = 'cuda'):
        self.config = config
        self.input_dim = input_dim
        self.device = device
        
        # Autoencoder architecture
        self.autoencoder = self._build_autoencoder()
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-4)
        
        # Threshold tracking
        self.reconstruction_errors = deque(maxlen=10000)
        self.novelty_threshold = 0.0
        self.threshold_percentile = config.reconstruction_threshold_percentile
        
        # Training statistics
        self.training_losses = deque(maxlen=1000)
        self.is_fitted = False
        
        logger.info(f"Autoencoder novelty detector initialized: {input_dim}D input")
    
    def _build_autoencoder(self) -> nn.Module:
        """Build autoencoder architecture."""
        encoder_layers = []
        decoder_layers = []
        
        # Encoder
        prev_dim = self.input_dim
        for hidden_dim in self.config.autoencoder_hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Bottleneck
        encoder_layers.append(nn.Linear(prev_dim, self.config.autoencoder_latent_dim))
        
        # Decoder (reverse of encoder)
        decoder_input_dim = self.config.autoencoder_latent_dim
        hidden_dims_reversed = list(reversed(self.config.autoencoder_hidden_dims))
        
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(decoder_input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            decoder_input_dim = hidden_dim
        
        # Output layer
        decoder_layers.append(nn.Linear(decoder_input_dim, self.input_dim))
        
        # Combine encoder and decoder
        autoencoder = nn.Sequential(
            nn.Sequential(*encoder_layers),  # Encoder
            nn.Sequential(*decoder_layers)   # Decoder
        )
        
        return autoencoder.to(self.device)
    
    def fit(self, data: np.ndarray) -> None:
        """Training autoencoder on normal data."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Training loop
        self.autoencoder.train()
        num_epochs = 100
        batch_size = min(64, len(data))
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            # Mini-batch training
            for i in range(0, len(data), batch_size):
                batch_data = data_tensor[i:i + batch_size]
                
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed = self.autoencoder(batch_data)
                loss = F.mse_loss(reconstructed, batch_data)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = np.mean(epoch_losses)
            self.training_losses.append(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Autoencoder training epoch {epoch}: loss = {avg_loss:.6f}")
        
        # Computation threshold on training data
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(data_tensor)
            reconstruction_errors = F.mse_loss(
                reconstructed, data_tensor, reduction='none'
            ).mean(dim=1).cpu().numpy()
            
            self.reconstruction_errors.extend(reconstruction_errors)
            self.novelty_threshold = np.percentile(
                reconstruction_errors, self.threshold_percentile
            )
        
        self.is_fitted = True
        logger.info(f"Autoencoder fitted. Threshold: {self.novelty_threshold:.6f}")
    
    def predict_novelty(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction novelty through reconstruction error."""
        if not self.is_fitted:
            logger.warning("Autoencoder not fitted. Using default threshold.")
            return np.zeros(len(data)), np.zeros(len(data), dtype=bool)
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder(data_tensor)
            reconstruction_errors = F.mse_loss(
                reconstructed, data_tensor, reduction='none'
            ).mean(dim=1).cpu().numpy()
        
        # Normalization by threshold
        novelty_scores = reconstruction_errors / (self.novelty_threshold + 1e-8)
        is_novel = reconstruction_errors > self.novelty_threshold
        
        # Update reconstruction errors history
        self.reconstruction_errors.extend(reconstruction_errors)
        
        return novelty_scores, is_novel
    
    def update(self, data: np.ndarray, is_novel: Optional[np.ndarray] = None) -> None:
        """Online update autoencoder."""
        if not self.is_fitted:
            return
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        # Update only on normal data (if )
        if is_novel is not None:
            normal_mask = ~is_novel
            if np.any(normal_mask):
                normal_data = data_tensor[normal_mask]
                
                self.autoencoder.train()
                self.optimizer.zero_grad()
                
                reconstructed = self.autoencoder(normal_data)
                loss = F.mse_loss(reconstructed, normal_data)
                
                loss.backward()
                self.optimizer.step()
                
                self.training_losses.append(loss.item())
        
        # Adaptive threshold update
        if self.config.adaptive_thresholds and len(self.reconstruction_errors) > 100:
            recent_errors = list(self.reconstruction_errors)[-1000:]
            self.novelty_threshold = np.percentile(recent_errors, self.threshold_percentile)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get autoencoder statistics."""
        stats = {
            'is_fitted': self.is_fitted,
            'novelty_threshold': self.novelty_threshold,
            'latent_dim': self.config.autoencoder_latent_dim,
            'num_parameters': sum(p.numel() for p in self.autoencoder.parameters())
        }
        
        if self.training_losses:
            stats['training_loss'] = {
                'mean': np.mean(list(self.training_losses)),
                'latest': list(self.training_losses)[-1]
            }
        
        if self.reconstruction_errors:
            errors = list(self.reconstruction_errors)
            stats['reconstruction_errors'] = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'count': len(errors)
            }
        
        return stats


class IsolationForestDetector(NoveltyDetector):
    """
    Isolation Forest-based novelty detection.
    
    Applies design pattern "Ensemble Learning" for
    robust anomaly detection.
    """
    
    def __init__(self, config: NoveltyDetectionConfig):
        self.config = config
        self.detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
            n_jobs=-1 if config.parallel_detection else 1
        )
        self.is_fitted = False
        self.decision_scores = deque(maxlen=10000)
        
        logger.info("Isolation Forest novelty detector initialized")
    
    def fit(self, data: np.ndarray) -> None:
        """Training Isolation Forest."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        self.detector.fit(data)
        self.is_fitted = True
        
        # Get decision scores for threshold calibration
        scores = self.detector.decision_function(data)
        self.decision_scores.extend(scores)
        
        logger.info(f"Isolation Forest fitted on {len(data)} samples")
    
    def predict_novelty(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction novelty through isolation scores."""
        if not self.is_fitted:
            logger.warning("Isolation Forest not fitted.")
            return np.zeros(len(data)), np.zeros(len(data), dtype=bool)
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Decision function scores (than less, that more anomalous)
        decision_scores = self.detector.decision_function(data)
        
        # Binary predictions
        predictions = self.detector.predict(data)
        is_novel = predictions == -1
        
        # Normalize scores (invert so higher = more novel)
        novelty_scores = -decision_scores
        
        # Save scores
        self.decision_scores.extend(decision_scores)
        
        return novelty_scores, is_novel
    
    def update(self, data: np.ndarray, is_novel: Optional[np.ndarray] = None) -> None:
        """Isolation Forest not supports online updates."""
        # on new data if enough
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Isolation Forest statistics."""
        stats = {
            'is_fitted': self.is_fitted,
            'n_estimators': self.detector.n_estimators if hasattr(self.detector, 'n_estimators') else 0
        }
        
        if self.decision_scores:
            scores = list(self.decision_scores)
            stats['decision_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }
        
        return stats


class OneClassSVMDetector(NoveltyDetector):
    """One-Class SVM novelty detector."""
    
    def __init__(self, config: NoveltyDetectionConfig):
        self.config = config
        self.detector = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=0.1
        )
        self.is_fitted = False
        self.decision_scores = deque(maxlen=10000)
        
        logger.info("One-Class SVM novelty detector initialized")
    
    def fit(self, data: np.ndarray) -> None:
        """Training One-Class SVM."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        self.detector.fit(data)
        self.is_fitted = True
        
        # Decision scores for calibration
        scores = self.detector.decision_function(data)
        self.decision_scores.extend(scores)
        
        logger.info(f"One-Class SVM fitted on {len(data)} samples")
    
    def predict_novelty(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction through SVM decision function."""
        if not self.is_fitted:
            logger.warning("One-Class SVM not fitted.")
            return np.zeros(len(data)), np.zeros(len(data), dtype=bool)
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        decision_scores = self.detector.decision_function(data)
        predictions = self.detector.predict(data)
        is_novel = predictions == -1
        
        # Normalize scores (invert for consistency)
        novelty_scores = -decision_scores
        
        self.decision_scores.extend(decision_scores)
        
        return novelty_scores, is_novel
    
    def update(self, data: np.ndarray, is_novel: Optional[np.ndarray] = None) -> None:
        """SVM not supports online updates."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """SVM statistics."""
        stats = {
            'is_fitted': self.is_fitted,
            'kernel': self.detector.kernel,
            'support_vectors': len(self.detector.support_vectors_) if self.is_fitted else 0
        }
        
        if self.decision_scores:
            scores = list(self.decision_scores)
            stats['decision_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return stats


class LOFDetector(NoveltyDetector):
    """Local Outlier Factor detector."""
    
    def __init__(self, config: NoveltyDetectionConfig):
        self.config = config
        self.detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
            n_jobs=-1 if config.parallel_detection else 1
        )
        self.is_fitted = False
        self.outlier_scores = deque(maxlen=10000)
        
        logger.info("LOF novelty detector initialized")
    
    def fit(self, data: np.ndarray) -> None:
        """Training LOF."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        self.detector.fit(data)
        self.is_fitted = True
        
        logger.info(f"LOF fitted on {len(data)} samples")
    
    def predict_novelty(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LOF prediction."""
        if not self.is_fitted:
            logger.warning("LOF not fitted.")
            return np.zeros(len(data)), np.zeros(len(data), dtype=bool)
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        decision_scores = self.detector.decision_function(data)
        predictions = self.detector.predict(data)
        is_novel = predictions == -1
        
        novelty_scores = -decision_scores
        
        self.outlier_scores.extend(decision_scores)
        
        return novelty_scores, is_novel
    
    def update(self, data: np.ndarray, is_novel: Optional[np.ndarray] = None) -> None:
        """LOF not supports online updates."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """LOF statistics."""
        stats = {
            'is_fitted': self.is_fitted,
            'n_neighbors': self.detector.n_neighbors
        }
        
        if self.outlier_scores:
            scores = list(self.outlier_scores)
            stats['outlier_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        return stats


class CryptoNoveltyDetector:
    """
    Comprehensive novelty detection system for crypto trading.
    
    Uses design pattern "Ensemble Strategy" for
    robust novelty detection with multiple methods.
    """
    
    def __init__(self, config: NoveltyDetectionConfig, input_dim: int, device: str = 'cuda'):
        self.config = config
        self.input_dim = input_dim
        self.device = device
        
        # Initialize detectors
        self.detectors = {}
        if "autoencoder" in config.detection_methods:
            self.detectors["autoencoder"] = AutoencoderNoveltyDetector(config, input_dim, device)
        if "isolation_forest" in config.detection_methods:
            self.detectors["isolation_forest"] = IsolationForestDetector(config)
        if "one_class_svm" in config.detection_methods:
            self.detectors["one_class_svm"] = OneClassSVMDetector(config)
        if "lof" in config.detection_methods:
            self.detectors["lof"] = LOFDetector(config)
        
        # Ensemble weights
        self.method_weights = config.method_weights.copy()
        
        # Crypto-specific tracking
        self.market_regimes_data = defaultdict(list)
        self.temporal_contexts = deque(maxlen=config.temporal_context_length)
        self.volatility_history = deque(maxlen=1000)
        
        # Performance tracking
        self.detection_history = deque(maxlen=10000)
        self.false_positive_rate = 0.0
        self.true_positive_rate = 0.0
        
        # Adaptive thresholds
        self.ensemble_threshold = config.confidence_threshold
        
        logger.info(f"Crypto novelty detector initialized with {len(self.detectors)} methods")
    
    def fit(self, data: np.ndarray, market_regimes: Optional[np.ndarray] = None) -> None:
        """
        Training all detectors.
        
        Args:
            data: Training data [n_samples, input_dim]
            market_regimes: Market regime labels for each sample
        """
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        logger.info(f"Fitting novelty detectors on {len(data)} samples")
        
        # Split data by market regimes if available
        if market_regimes is not None and self.config.market_regime_aware:
            unique_regimes = np.unique(market_regimes)
            for regime in unique_regimes:
                regime_mask = market_regimes == regime
                regime_data = data[regime_mask]
                self.market_regimes_data[regime] = regime_data
                logger.info(f"Regime {regime}: {len(regime_data)} samples")
        
        # Training each detector
        for name, detector in self.detectors.items():
            try:
                start_time = time.time()
                detector.fit(data)
                fit_time = time.time() - start_time
                logger.info(f"Detector {name} fitted in {fit_time:.2f}s")
            except Exception as e:
                logger.error(f"Error fitting detector {name}: {e}")
    
    def detect_novelty(
        self,
        data: np.ndarray,
        market_regime: Optional[str] = None,
        portfolio_volatility: Optional[float] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Ensemble novelty detection.
        
        Args:
            data: Input data for detection
            market_regime: Current market regime
            portfolio_volatility: Portfolio volatility level
            
        Returns:
            Tuple (ensemble_novelty_score, detection_breakdown)
        """
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Get predictions from each detector
        detector_results = {}
        detector_scores = []
        detector_decisions = []
        
        for name, detector in self.detectors.items():
            try:
                scores, decisions = detector.predict_novelty(data)
                detector_results[name] = {
                    'scores': scores,
                    'decisions': decisions,
                    'weight': self.method_weights.get(name, 0.0)
                }
                
                # For ensemble voting
                detector_scores.append(scores[0] if len(scores) > 0 else 0.0)
                detector_decisions.append(decisions[0] if len(decisions) > 0 else False)
                
            except Exception as e:
                logger.warning(f"Error in detector {name}: {e}")
                detector_results[name] = {
                    'scores': np.array([0.0]),
                    'decisions': np.array([False]),
                    'weight': 0.0
                }
                detector_scores.append(0.0)
                detector_decisions.append(False)
        
        # Ensemble combination
        if self.config.ensemble_voting == "weighted":
            weighted_score = 0.0
            total_weight = 0.0
            
            for name, result in detector_results.items():
                weight = result['weight']
                score = result['scores'][0] if len(result['scores']) > 0 else 0.0
                weighted_score += weight * score
                total_weight += weight
            
            ensemble_score = weighted_score / (total_weight + 1e-8)
            
        elif self.config.ensemble_voting == "average":
            ensemble_score = np.mean(detector_scores)
            
        elif self.config.ensemble_voting == "majority":
            ensemble_score = 1.0 if sum(detector_decisions) > len(detector_decisions) / 2 else 0.0
        
        # Crypto-specific adjustments
        if portfolio_volatility is not None and self.config.volatility_adjustment:
            # Higher volatility = higher chance of novelty
            volatility_multiplier = 1.0 + self.config.portfolio_novelty_weight * portfolio_volatility
            ensemble_score *= volatility_multiplier
        
        # Temporal context consideration
        self.temporal_contexts.append(data[0] if len(data) > 0 else np.zeros(self.input_dim))
        if len(self.temporal_contexts) >= 2:
            # Temporal consistency check
            prev_context = np.array(list(self.temporal_contexts)[-2])
            curr_context = data[0] if len(data) > 0 else np.zeros(self.input_dim)
            temporal_change = np.linalg.norm(curr_context - prev_context)
            
            if self.config.temporal_context_length > 0:
                # Weight novelty by temporal change
                temporal_weight = min(temporal_change / 10.0, 2.0)  # Normalize and cap
                ensemble_score *= temporal_weight
        
        # Binary decision
        is_novel = ensemble_score > self.ensemble_threshold
        
        # Save for history
        self.detection_history.append({
            'score': ensemble_score,
            'is_novel': is_novel,
            'market_regime': market_regime,
            'portfolio_volatility': portfolio_volatility
        })
        
        # Detection breakdown
        breakdown = {
            'ensemble_score': ensemble_score,
            'is_novel': is_novel,
            'ensemble_threshold': self.ensemble_threshold,
            'detector_results': detector_results,
            'market_regime': market_regime,
            'portfolio_volatility': portfolio_volatility,
            'temporal_change': temporal_change if len(self.temporal_contexts) >= 2 else 0.0
        }
        
        return ensemble_score, breakdown
    
    def update_detectors(
        self,
        data: np.ndarray,
        true_novelty: Optional[np.ndarray] = None,
        market_regime: Optional[str] = None
    ) -> Dict[str, Any]:
        """Online update all detectors."""
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        update_stats = {}
        
        # Update each detector
        for name, detector in self.detectors.items():
            try:
                detector.update(data, true_novelty)
                update_stats[f'{name}_updated'] = True
            except Exception as e:
                logger.warning(f"Error updating detector {name}: {e}")
                update_stats[f'{name}_updated'] = False
        
        # Adaptive threshold adjustment
        if (self.config.adaptive_thresholds and 
            len(self.detection_history) > self.config.threshold_update_frequency):
            self._update_ensemble_threshold()
        
        # Performance evaluation if there is ground truth
        if true_novelty is not None:
            self._evaluate_performance(true_novelty)
        
        update_stats['ensemble_threshold'] = self.ensemble_threshold
        update_stats['detection_history_size'] = len(self.detection_history)
        
        return update_stats
    
    def _update_ensemble_threshold(self) -> None:
        """Adaptive update ensemble threshold."""
        recent_detections = list(self.detection_history)[-self.config.threshold_update_frequency:]
        scores = [d['score'] for d in recent_detections]
        
        if scores:
            # Adjust threshold based on score distribution
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            
            # Target false positive rate
            target_fpr = self.config.false_positive_tolerance
            
            # Empirical threshold adjustment
            new_threshold = score_mean + score_std * 2.0  # 2-sigma rule
            
            # Smooth update
            alpha = 0.1
            self.ensemble_threshold = (
                (1 - alpha) * self.ensemble_threshold + 
                alpha * new_threshold
            )
            
            logger.info(f"Updated ensemble threshold: {self.ensemble_threshold:.4f}")
    
    def _evaluate_performance(self, true_novelty: np.ndarray) -> None:
        """Evaluation performance metrics."""
        if len(self.detection_history) == 0:
            return
        
        recent_decisions = [d['is_novel'] for d in self.detection_history[-len(true_novelty):]]
        
        if len(recent_decisions) == len(true_novelty):
            # Compute metrics
            true_positives = sum(1 for i, j in zip(recent_decisions, true_novelty) if i and j)
            false_positives = sum(1 for i, j in zip(recent_decisions, true_novelty) if i and not j)
            true_negatives = sum(1 for i, j in zip(recent_decisions, true_novelty) if not i and not j)
            false_negatives = sum(1 for i, j in zip(recent_decisions, true_novelty) if not i and j)
            
            # Update rates
            self.true_positive_rate = true_positives / max(1, true_positives + false_negatives)
            self.false_positive_rate = false_positives / max(1, false_positives + true_negatives)
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive detection statistics."""
        stats = {
            'ensemble_threshold': self.ensemble_threshold,
            'true_positive_rate': self.true_positive_rate,
            'false_positive_rate': self.false_positive_rate,
            'detection_history_size': len(self.detection_history),
            'method_weights': self.method_weights.copy()
        }
        
        # Individual detector statistics
        for name, detector in self.detectors.items():
            stats[f'{name}_stats'] = detector.get_statistics()
        
        # Detection history statistics
        if self.detection_history:
            scores = [d['score'] for d in self.detection_history]
            novelty_decisions = [d['is_novel'] for d in self.detection_history]
            
            stats['detection_scores'] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': min(scores),
                'max': max(scores)
            }
            
            stats['novelty_rate'] = sum(novelty_decisions) / len(novelty_decisions)
        
        # Market regime statistics
        if self.market_regimes_data:
            regime_stats = {}
            for regime, data in self.market_regimes_data.items():
                regime_stats[regime] = {
                    'sample_count': len(data),
                    'mean_features': np.mean(data, axis=0).tolist() if len(data) > 0 else []
                }
            stats['market_regime_stats'] = regime_stats
        
        return stats


def create_novelty_detection_system(
    config: NoveltyDetectionConfig,
    input_dim: int
) -> CryptoNoveltyDetector:
    """
    Factory function for creation novelty detection system.
    
    Args:
        config: Novelty detection configuration
        input_dim: Input dimension
        
    Returns:
        Configured crypto novelty detector
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = CryptoNoveltyDetector(config, input_dim, device)
    
    logger.info("Novelty detection system created successfully")
    logger.info(f"Detection methods: {list(detector.detectors.keys())}")
    logger.info(f"Input dimension: {input_dim}")
    
    return detector


if __name__ == "__main__":
    # Example use novelty detection
    config = NoveltyDetectionConfig(
        detection_methods=["autoencoder", "isolation_forest"],
        ensemble_voting="weighted",
        adaptive_thresholds=True
    )
    
    input_dim = 128
    detector = create_novelty_detection_system(config, input_dim)
    
    # Create synthetic training data
    normal_data = np.random.randn(1000, input_dim)
    detector.fit(normal_data)
    
    # Testing on novel data
    novel_data = np.random.randn(100, input_dim) * 3  # More extreme values
    
    for i in range(10):
        test_sample = novel_data[i:i+1]
        novelty_score, breakdown = detector.detect_novelty(
            test_sample,
            market_regime="volatile",
            portfolio_volatility=0.8
        )
        
        print(f"Sample {i}: Novelty Score = {novelty_score:.4f}, "
              f"Is Novel = {breakdown['is_novel']}")
    
    # Statistics
    stats = detector.get_detection_statistics()
    print("\nDetection Statistics:")
    for key, value in stats.items():
        if not isinstance(value, dict):
            print(f"{key}: {value}")