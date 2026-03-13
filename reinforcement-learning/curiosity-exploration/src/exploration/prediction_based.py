"""
Prediction-Based Exploration for crypto trading environments.

Implements exploration strategies based on prediction uncertainty
with enterprise patterns for intelligent strategy discovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque
from abc import ABC, abstractmethod
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionBasedConfig:
    """Configuration for prediction-based exploration."""
    
    # Model architecture
    state_dim: int = 256
    action_dim: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    ensemble_size: int = 5
    
    # Uncertainty estimation
    uncertainty_method: str = "ensemble"  # "ensemble", "dropout", "bayesian"
    dropout_rate: float = 0.2
    mc_samples: int = 10
    epistemic_weight: float = 0.7
    aleatoric_weight: float = 0.3
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 256
    prediction_horizon: int = 1
    multi_step_prediction: bool = True
    max_prediction_steps: int = 5
    
    # Exploration bonuses
    uncertainty_bonus_coeff: float = 0.1
    information_gain_coeff: float = 0.05
    max_uncertainty_bonus: float = 2.0
    min_uncertainty_bonus: float = 0.001
    
    # Crypto-specific parameters
    market_prediction_weight: float = 0.4
    portfolio_prediction_weight: float = 0.3
    risk_prediction_weight: float = 0.3
    volatility_bonus_multiplier: float = 1.5
    
    # Advanced features
    adaptive_uncertainty: bool = True
    temporal_consistency_weight: float = 0.2
    prediction_diversity_bonus: bool = True
    confidence_calibration: bool = True
    
    #  enterprise settings
    distributed_ensemble: bool = True
    model_compression: bool = True
    uncertainty_caching: bool = True
    real_time_inference: bool = True


class UncertaintyEstimator(ABC):
    """
    Abstract base class for uncertainty estimation.
    
    Applies design pattern "Strategy Pattern" for
    flexible uncertainty quantification methods.
    """
    
    @abstractmethod
    def predict_with_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with uncertainty estimation.
        
        Returns:
            Tuple (predictions, epistemic_uncertainty, aleatoric_uncertainty)
        """
        pass
    
    @abstractmethod
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """Update model predictions."""
        pass


class EnsemblePredictor(nn.Module):
    """
    Ensemble-based uncertainty estimation.
    
    Uses design pattern "Ensemble Methods" for
    robust uncertainty quantification in financial predictions.
    """
    
    def __init__(self, config: PredictionBasedConfig):
        super().__init__()
        self.config = config
        self.ensemble_size = config.ensemble_size
        
        # Create ensemble models
        self.ensemble_models = nn.ModuleList()
        for _ in range(self.ensemble_size):
            model = self._create_prediction_model()
            self.ensemble_models.append(model)
        
        # optimizers for each model ensemble
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            for model in self.ensemble_models
        ]
        
        # Uncertainty calibration
        if config.confidence_calibration:
            self.calibration_model = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            self.calibration_optimizer = torch.optim.Adam(
                self.calibration_model.parameters(), lr=1e-3
            )
        
        logger.info(f"Ensemble predictor initialized with {self.ensemble_size} models")
    
    def _create_prediction_model(self) -> nn.Module:
        """Create single prediction model for ensemble."""
        layers = []
        input_dim = self.config.state_dim + self.config.action_dim
        
        # Encoder layers
        prev_dim = input_dim
        for hidden_dim in self.config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Multi-component output for crypto trading
        market_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, self.config.state_dim // 2)
        )
        
        portfolio_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 4),
            nn.ReLU(),
            nn.Linear(prev_dim // 4, self.config.state_dim // 4)
        )
        
        risk_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 4),
            nn.ReLU(),
            nn.Linear(prev_dim // 4, self.config.state_dim // 4)
        )
        
        # Aleatoric uncertainty heads
        market_uncertainty = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 4),
            nn.ReLU(),
            nn.Linear(prev_dim // 4, self.config.state_dim // 2),
            nn.Softplus() # Ensures values
        )
        
        portfolio_uncertainty = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 8),
            nn.ReLU(),
            nn.Linear(prev_dim // 8, self.config.state_dim // 4),
            nn.Softplus()
        )
        
        risk_uncertainty = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 8),
            nn.ReLU(),
            nn.Linear(prev_dim // 8, self.config.state_dim // 4),
            nn.Softplus()
        )
        
        # Merging in one model
        model = nn.ModuleDict({
            'encoder': nn.Sequential(*layers[:-1]), # Without last dropout
            'market_head': market_head,
            'portfolio_head': portfolio_head,
            'risk_head': risk_head,
            'market_uncertainty': market_uncertainty,
            'portfolio_uncertainty': portfolio_uncertainty,
            'risk_uncertainty': risk_uncertainty
        })
        
        return model
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble with prediction and uncertainty.
        
        Args:
            states: Input states [batch_size, state_dim]
            actions: Input actions [batch_size, action_dim]
            
        Returns:
            Dictionary with predictions and uncertainties from all models
        """
        batch_size = states.size(0)
        
        # Merging states and actions
        inputs = torch.cat([states, actions], dim=1)
        
        # Predictions from each model in ensemble
        ensemble_predictions = {
            'market': [],
            'portfolio': [],
            'risk': [],
            'market_uncertainty': [],
            'portfolio_uncertainty': [],
            'risk_uncertainty': []
        }
        
        for model in self.ensemble_models:
            # Encoder
            encoded = model['encoder'](inputs)
            
            # Predictions for each component
            market_pred = model['market_head'](encoded)
            portfolio_pred = model['portfolio_head'](encoded)
            risk_pred = model['risk_head'](encoded)
            
            # Aleatoric uncertainties
            market_unc = model['market_uncertainty'](encoded)
            portfolio_unc = model['portfolio_uncertainty'](encoded)
            risk_unc = model['risk_uncertainty'](encoded)
            
            # Save predictions
            ensemble_predictions['market'].append(market_pred)
            ensemble_predictions['portfolio'].append(portfolio_pred)
            ensemble_predictions['risk'].append(risk_pred)
            ensemble_predictions['market_uncertainty'].append(market_unc)
            ensemble_predictions['portfolio_uncertainty'].append(portfolio_unc)
            ensemble_predictions['risk_uncertainty'].append(risk_unc)
        
        # acking predictions
        for key in ensemble_predictions:
            ensemble_predictions[key] = torch.stack(ensemble_predictions[key], dim=0)
        
        return ensemble_predictions
    
    def predict_with_uncertainty(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prediction with full uncertainty quantification.
        
        Args:
            states: Input states
            actions: Input actions
            
        Returns:
            Tuple (mean_predictions, epistemic_uncertainty, aleatoric_uncertainty)
        """
        with torch.no_grad():
            ensemble_outputs = self.forward(states, actions)
            
            # Mean predictions
            market_mean = ensemble_outputs['market'].mean(dim=0)
            portfolio_mean = ensemble_outputs['portfolio'].mean(dim=0)
            risk_mean = ensemble_outputs['risk'].mean(dim=0)
            
            mean_predictions = torch.cat([market_mean, portfolio_mean, risk_mean], dim=1)
            
            # Epistemic uncertainty (variance across ensemble)
            market_epistemic = ensemble_outputs['market'].var(dim=0)
            portfolio_epistemic = ensemble_outputs['portfolio'].var(dim=0)
            risk_epistemic = ensemble_outputs['risk'].var(dim=0)
            
            epistemic_uncertainty = torch.cat([
                market_epistemic, portfolio_epistemic, risk_epistemic
            ], dim=1)
            
            # Aleatoric uncertainty (mean of predicted uncertainties)
            market_aleatoric = ensemble_outputs['market_uncertainty'].mean(dim=0)
            portfolio_aleatoric = ensemble_outputs['portfolio_uncertainty'].mean(dim=0)
            risk_aleatoric = ensemble_outputs['risk_uncertainty'].mean(dim=0)
            
            aleatoric_uncertainty = torch.cat([
                market_aleatoric, portfolio_aleatoric, risk_aleatoric
            ], dim=1)
            
            return mean_predictions, epistemic_uncertainty, aleatoric_uncertainty
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update ensemble models.
        
        Args:
            states: Input states
            actions: Input actions
            targets: Target next states
            
        Returns:
            Training metrics
        """
        batch_size = states.size(0)
        
        # Split targets on components
        market_targets = targets[:, :self.config.state_dim // 2]
        portfolio_targets = targets[:, 
            self.config.state_dim // 2:3 * self.config.state_dim // 4]
        risk_targets = targets[:, 3 * self.config.state_dim // 4:]
        
        total_loss = 0.0
        component_losses = {'market': 0.0, 'portfolio': 0.0, 'risk': 0.0}
        
        # Training each model in ensemble
        for i, (model, optimizer) in enumerate(zip(self.ensemble_models, self.optimizers)):
            optimizer.zero_grad()
            
            # Forward pass
            inputs = torch.cat([states, actions], dim=1)
            encoded = model['encoder'](inputs)
            
            # Predictions
            market_pred = model['market_head'](encoded)
            portfolio_pred = model['portfolio_head'](encoded)
            risk_pred = model['risk_head'](encoded)
            
            # Uncertainty predictions
            market_unc = model['market_uncertainty'](encoded)
            portfolio_unc = model['portfolio_uncertainty'](encoded)
            risk_unc = model['risk_uncertainty'](encoded)
            
            # Heteroscedastic loss (uncertainty-aware)
            market_loss = self._heteroscedastic_loss(market_pred, market_targets, market_unc)
            portfolio_loss = self._heteroscedastic_loss(portfolio_pred, portfolio_targets, portfolio_unc)
            risk_loss = self._heteroscedastic_loss(risk_pred, risk_targets, risk_unc)
            
            # Weighted combination
            model_loss = (
                self.config.market_prediction_weight * market_loss +
                self.config.portfolio_prediction_weight * portfolio_loss +
                self.config.risk_prediction_weight * risk_loss
            )
            
            # Regularization for diversity ensemble
            if self.config.prediction_diversity_bonus and len(self.ensemble_models) > 1:
                diversity_loss = self._compute_diversity_loss(i, states, actions)
                model_loss += 0.01 * diversity_loss
            
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += model_loss.item()
            component_losses['market'] += market_loss.item()
            component_losses['portfolio'] += portfolio_loss.item()
            component_losses['risk'] += risk_loss.item()
        
        # Averaging losses
        avg_loss = total_loss / self.ensemble_size
        for key in component_losses:
            component_losses[key] /= self.ensemble_size
        
        metrics = {
            'total_loss': avg_loss,
            'market_loss': component_losses['market'],
            'portfolio_loss': component_losses['portfolio'],
            'risk_loss': component_losses['risk'],
            'ensemble_size': self.ensemble_size
        }
        
        return metrics
    
    def _heteroscedastic_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Heteroscedastic loss predicted uncertainty."""
        # Precision = 1 / variance
        precisions = 1.0 / (uncertainties + 1e-8)
        
        # Weighted MSE loss
        mse_loss = (predictions - targets) ** 2
        weighted_loss = 0.5 * (precisions * mse_loss + torch.log(uncertainties + 1e-8))
        
        return weighted_loss.mean()
    
    def _compute_diversity_loss(
        self,
        model_idx: int,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Diversity loss for encouraging diversity in ensemble."""
        current_model = self.ensemble_models[model_idx]
        inputs = torch.cat([states, actions], dim=1)
        
        with torch.no_grad():
            current_encoded = current_model['encoder'](inputs)
            current_market = current_model['market_head'](current_encoded)
            
            # Comparison with other models
            diversity_losses = []
            for other_idx, other_model in enumerate(self.ensemble_models):
                if other_idx != model_idx:
                    other_encoded = other_model['encoder'](inputs)
                    other_market = other_model['market_head'](other_encoded)
                    
                    # Negative correlation loss (encourage diversity)
                    correlation = F.cosine_similarity(
                        current_market.view(current_market.size(0), -1),
                        other_market.view(other_market.size(0), -1),
                        dim=1
                    )
                    diversity_losses.append(-correlation.mean())
        
        if diversity_losses:
            return torch.stack(diversity_losses).mean()
        else:
            return torch.tensor(0.0, device=states.device)


class PredictionBasedExplorer:
    """
    Prediction-based exploration system with uncertainty quantification.
    
    Uses design pattern "Uncertainty-Aware Exploration" for
    intelligent discovery trading strategies.
    """
    
    def __init__(self, config: PredictionBasedConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize uncertainty estimator
        if config.uncertainty_method == "ensemble":
            self.predictor = EnsemblePredictor(config).to(device)
        else:
            raise NotImplementedError(f"Uncertainty method {config.uncertainty_method} not implemented")
        
        # Uncertainty tracking and calibration
        self.uncertainty_history = deque(maxlen=10000)
        self.prediction_errors = deque(maxlen=10000)
        self.information_gains = deque(maxlen=10000)
        
        # Running statistics for normalization
        self.uncertainty_stats = {'mean': 0.0, 'std': 1.0}
        self.error_stats = {'mean': 0.0, 'std': 1.0}
        
        # Multi-step prediction tracking
        if config.multi_step_prediction:
            self.multi_step_errors = {
                i: deque(maxlen=5000) 
                for i in range(1, config.max_prediction_steps + 1)
            }
        
        # Crypto-specific tracking
        self.market_volatility_history = deque(maxlen=1000)
        self.portfolio_uncertainty_history = deque(maxlen=1000)
        
        # Performance optimization
        self.prediction_cache = {} if config.uncertainty_caching else None
        self.cache_hit_rate = 0.0
        self.cache_requests = 0
        
        logger.info(f"Prediction-based explorer initialized on {device}")
    
    def get_uncertainty_bonus(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        market_volatility: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computation uncertainty-based exploration bonus.
        
        Args:
            state: Current state
            action: Proposed action
            market_volatility: Current market volatility
            
        Returns:
            Tuple (total_bonus, component_breakdown)
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Check cache
        cache_key = None
        if self.prediction_cache is not None:
            state_hash = hash(state.cpu().numpy().tobytes())
            action_hash = hash(action.cpu().numpy().tobytes())
            cache_key = (state_hash, action_hash)
            
            self.cache_requests += 1
            if cache_key in self.prediction_cache:
                self.cache_hit_rate = 0.99 * self.cache_hit_rate + 0.01 * 1.0
                cached_result = self.prediction_cache[cache_key]
                return cached_result['bonus'], cached_result['breakdown']
            else:
                self.cache_hit_rate = 0.99 * self.cache_hit_rate + 0.01 * 0.0
        
        # Prediction with uncertainty
        predictions, epistemic_unc, aleatoric_unc = self.predictor.predict_with_uncertainty(
            state, action
        )
        
        # Combining uncertainties
        total_epistemic = epistemic_unc.mean(dim=1)
        total_aleatoric = aleatoric_unc.mean(dim=1)
        
        # Weighted uncertainty
        combined_uncertainty = (
            self.config.epistemic_weight * total_epistemic +
            self.config.aleatoric_weight * total_aleatoric
        )
        
        # Normalization uncertainty
        if len(self.uncertainty_history) > 100:
            uncertainty_mean = np.mean(list(self.uncertainty_history)[-1000:])
            uncertainty_std = np.std(list(self.uncertainty_history)[-1000:])
            normalized_uncertainty = (combined_uncertainty.item() - uncertainty_mean) / (uncertainty_std + 1e-8)
        else:
            normalized_uncertainty = combined_uncertainty.item()
        
        # Base uncertainty bonus
        uncertainty_bonus = self.config.uncertainty_bonus_coeff * normalized_uncertainty
        uncertainty_bonus = np.clip(
            uncertainty_bonus, 
            self.config.min_uncertainty_bonus,
            self.config.max_uncertainty_bonus
        )
        
        # Information gain bonus
        information_gain = self._compute_information_gain(
            epistemic_unc, aleatoric_unc
        )
        info_gain_bonus = self.config.information_gain_coeff * information_gain
        
        # Market volatility adjustment
        volatility_multiplier = 1.0
        if market_volatility is not None:
            # More high bonus in volatile periods
            volatility_multiplier = 1.0 + self.config.volatility_bonus_multiplier * market_volatility
            self.market_volatility_history.append(market_volatility)
        
        # Total bonus
        total_bonus = (uncertainty_bonus + info_gain_bonus) * volatility_multiplier
        
        # Save for statistics
        self.uncertainty_history.append(combined_uncertainty.item())
        self.information_gains.append(information_gain)
        
        # components
        component_breakdown = {
            'epistemic_uncertainty': total_epistemic.item(),
            'aleatoric_uncertainty': total_aleatoric.item(),
            'combined_uncertainty': combined_uncertainty.item(),
            'uncertainty_bonus': uncertainty_bonus,
            'information_gain': information_gain,
            'info_gain_bonus': info_gain_bonus,
            'volatility_multiplier': volatility_multiplier,
            'total_bonus': total_bonus
        }
        
        # Save in cache
        if cache_key is not None:
            self.prediction_cache[cache_key] = {
                'bonus': total_bonus,
                'breakdown': component_breakdown
            }
            
            # Limitation size cache
            if len(self.prediction_cache) > 10000:
                # Remove old entries
                keys_to_remove = list(self.prediction_cache.keys())[:2000]
                for key in keys_to_remove:
                    del self.prediction_cache[key]
        
        return total_bonus, component_breakdown
    
    def _compute_information_gain(
        self,
        epistemic_uncertainty: torch.Tensor,
        aleatoric_uncertainty: torch.Tensor
    ) -> float:
        """Computation information gain from observation."""
        # Information gain proportional to epistemic uncertainty
        # (reducible uncertainty through more data)
        epistemic_mean = epistemic_uncertainty.mean().item()
        
        # Mutual information approximation
        information_gain = np.log(1 + epistemic_mean)
        
        return information_gain
    
    def update_predictions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        multi_step_targets: Optional[Dict[int, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Update prediction models with new experiences.
        
        Args:
            states: Current states
            actions: Executed actions
            next_states: Resulting next states
            multi_step_targets: Multi-step prediction targets
            
        Returns:
            Training metrics
        """
        # update predictor
        metrics = self.predictor.update(states, actions, next_states)
        
        # Computation prediction errors for calibration
        with torch.no_grad():
            predictions, epistemic_unc, aleatoric_unc = self.predictor.predict_with_uncertainty(
                states, actions
            )
            
            # Prediction error
            prediction_error = F.mse_loss(predictions, next_states, reduction='none').mean(dim=1)
            
            # Save errors for analysis
            for error in prediction_error:
                self.prediction_errors.append(error.item())
        
        # Multi-step prediction updates
        if self.config.multi_step_prediction and multi_step_targets is not None:
            multi_step_metrics = self._update_multi_step_predictions(
                states, actions, multi_step_targets
            )
            metrics.update(multi_step_metrics)
        
        # Update statistics for normalization
        self._update_normalization_stats()
        
        # Add exploration-specific metrics
        metrics.update({
            'avg_uncertainty': np.mean(list(self.uncertainty_history)[-100:]) if self.uncertainty_history else 0.0,
            'avg_prediction_error': np.mean(list(self.prediction_errors)[-100:]) if self.prediction_errors else 0.0,
            'avg_information_gain': np.mean(list(self.information_gains)[-100:]) if self.information_gains else 0.0,
            'cache_hit_rate': self.cache_hit_rate,
            'uncertainty_calibration': self._compute_calibration_score()
        })
        
        return metrics
    
    def _update_multi_step_predictions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        multi_step_targets: Dict[int, torch.Tensor]
    ) -> Dict[str, float]:
        """Update multi-step prediction capabilities."""
        multi_step_losses = {}
        
        current_state = states
        for step in range(1, self.config.max_prediction_steps + 1):
            if step in multi_step_targets:
                # Prediction on step steps
                predictions, _, _ = self.predictor.predict_with_uncertainty(
                    current_state, actions
                )
                
                # Loss for given step
                target = multi_step_targets[step]
                step_loss = F.mse_loss(predictions, target)
                
                multi_step_losses[f'step_{step}_loss'] = step_loss.item()
                
                # Save error for analysis
                step_error = F.mse_loss(predictions, target, reduction='none').mean(dim=1)
                for error in step_error:
                    if step in self.multi_step_errors:
                        self.multi_step_errors[step].append(error.item())
                
                # state for prediction
                current_state = predictions.detach()
        
        return multi_step_losses
    
    def _update_normalization_stats(self) -> None:
        """Update running statistics for normalization."""
        if len(self.uncertainty_history) > 100:
            recent_uncertainties = list(self.uncertainty_history)[-1000:]
            self.uncertainty_stats = {
                'mean': np.mean(recent_uncertainties),
                'std': np.std(recent_uncertainties)
            }
        
        if len(self.prediction_errors) > 100:
            recent_errors = list(self.prediction_errors)[-1000:]
            self.error_stats = {
                'mean': np.mean(recent_errors),
                'std': np.std(recent_errors)
            }
    
    def _compute_calibration_score(self) -> float:
        """Computation calibration score for uncertainty estimates."""
        if len(self.uncertainty_history) < 100 or len(self.prediction_errors) < 100:
            return 0.0
        
        uncertainties = np.array(list(self.uncertainty_history)[-1000:])
        errors = np.array(list(self.prediction_errors)[-1000:])
        
        # Correlation between predicted uncertainty and actual error
        if len(uncertainties) == len(errors):
            correlation = np.corrcoef(uncertainties, errors)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics exploration."""
        stats = {
            'uncertainty_statistics': {
                'mean': np.mean(list(self.uncertainty_history)) if self.uncertainty_history else 0.0,
                'std': np.std(list(self.uncertainty_history)) if self.uncertainty_history else 0.0,
                'count': len(self.uncertainty_history)
            },
            'prediction_error_statistics': {
                'mean': np.mean(list(self.prediction_errors)) if self.prediction_errors else 0.0,
                'std': np.std(list(self.prediction_errors)) if self.prediction_errors else 0.0,
                'count': len(self.prediction_errors)
            },
            'information_gain_statistics': {
                'mean': np.mean(list(self.information_gains)) if self.information_gains else 0.0,
                'std': np.std(list(self.information_gains)) if self.information_gains else 0.0,
                'count': len(self.information_gains)
            },
            'calibration_score': self._compute_calibration_score(),
            'cache_performance': {
                'hit_rate': self.cache_hit_rate,
                'total_requests': self.cache_requests,
                'cache_size': len(self.prediction_cache) if self.prediction_cache else 0
            }
        }
        
        # Multi-step prediction statistics
        if self.config.multi_step_prediction:
            multi_step_stats = {}
            for step, errors in self.multi_step_errors.items():
                if errors:
                    multi_step_stats[f'step_{step}'] = {
                        'mean_error': np.mean(list(errors)),
                        'std_error': np.std(list(errors)),
                        'count': len(errors)
                    }
            stats['multi_step_statistics'] = multi_step_stats
        
        # Market volatility statistics
        if self.market_volatility_history:
            stats['market_volatility_statistics'] = {
                'mean': np.mean(list(self.market_volatility_history)),
                'std': np.std(list(self.market_volatility_history)),
                'count': len(self.market_volatility_history)
            }
        
        return stats
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint exploration system."""
        checkpoint = {
            'predictor_state': self.predictor.state_dict(),
            'config': self.config,
            'uncertainty_stats': self.uncertainty_stats,
            'error_stats': self.error_stats,
            'cache_hit_rate': self.cache_hit_rate,
            'cache_requests': self.cache_requests
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Prediction-based explorer checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint exploration system."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.predictor.load_state_dict(checkpoint['predictor_state'])
        self.uncertainty_stats = checkpoint['uncertainty_stats']
        self.error_stats = checkpoint['error_stats']
        self.cache_hit_rate = checkpoint['cache_hit_rate']
        self.cache_requests = checkpoint['cache_requests']
        
        logger.info(f"Prediction-based explorer checkpoint loaded from {filepath}")


def create_prediction_based_system(config: PredictionBasedConfig) -> PredictionBasedExplorer:
    """
    Factory function for creation prediction-based exploration system.
    
    Args:
        config: Prediction-based configuration
        
    Returns:
        Configured prediction-based explorer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    explorer = PredictionBasedExplorer(config, device)
    
    logger.info("Prediction-based exploration system created successfully")
    logger.info(f"Uncertainty method: {config.uncertainty_method}")
    logger.info(f"Ensemble size: {config.ensemble_size}")
    
    return explorer


if __name__ == "__main__":
    # Example use prediction-based exploration
    config = PredictionBasedConfig(
        state_dim=128,
        action_dim=5,
        ensemble_size=3,
        uncertainty_method="ensemble",
        multi_step_prediction=True
    )
    
    explorer = create_prediction_based_system(config)
    
    # Simulation exploration
    batch_size = 32
    states = torch.randn(batch_size, config.state_dim)
    actions = torch.randn(batch_size, config.action_dim)
    next_states = torch.randn(batch_size, config.state_dim)
    
    # Training
    for step in range(100):
        metrics = explorer.update_predictions(states, actions, next_states)
        
        if step % 20 == 0:
            print(f"Step {step}: Loss={metrics['total_loss']:.4f}, "
                  f"Calibration={metrics['uncertainty_calibration']:.4f}")
    
    # Get exploration bonus
    single_state = torch.randn(1, config.state_dim)
    single_action = torch.randn(1, config.action_dim)
    
    bonus, breakdown = explorer.get_uncertainty_bonus(
        single_state, single_action, market_volatility=0.5
    )
    
    print(f"\nExploration bonus: {bonus:.4f}")
    print("Breakdown:", breakdown)
    
    # Statistics
    stats = explorer.get_exploration_statistics()
    print("\nExploration Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")