"""
GNN Ensemble System for Crypto Trading
=======================================

Enterprise-grade ensemble of Graph Neural Networks combining GCN, GAT, 
GraphSAGE, and MPNN for robust crypto market predictions with enterprise patterns.

Features:
- Multi-model ensemble architecture
- Adaptive model weighting based on performance
- Cross-validation ensemble training
- Uncertainty quantification
- Production-ready inference pipeline
- Real-time model performance monitoring

Author: ML-Framework ML Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle
import json
import warnings

# GNN models
from .gcn import GraphConvolutionalNetwork, GCNConfig
from .gat import GraphAttentionNetwork, GATConfig
from .graphsage import GraphSAGE, GraphSAGEConfig
from .mpnn import MessagePassingNeuralNetwork, MPNNConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnsembleConfig:
    """
    Configuration for GNN Ensemble
    
    Comprehensive Ensemble Configuration
    """
    # Model configurations
    gcn_config: Optional[GCNConfig] = None
    gat_config: Optional[GATConfig] = None
    graphsage_config: Optional[GraphSAGEConfig] = None
    mpnn_config: Optional[MPNNConfig] = None
    
    # Ensemble parameters
    ensemble_method: str = 'weighted_average'  # weighted_average, voting, stacking, dynamic_weighting
    initial_weights: Optional[Dict[str, float]] = None
    
    # Adaptive weighting parameters
    use_adaptive_weights: bool = True
    weight_update_frequency: int = 100  # steps
    performance_window_size: int = 50 # performance window for adaptation
    min_weight: float = 0.1  # Minimum weight model
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    uncertainty_method: str = 'ensemble_variance'  # ensemble_variance, monte_carlo_dropout
    mc_dropout_samples: int = 10
    
    # Performance monitoring
    enable_monitoring: bool = True
    monitoring_metrics: List[str] = None
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Meta-learning parameters
    use_meta_learning: bool = False
    meta_learning_lr: float = 0.01
    
    def __post_init__(self):
        if self.initial_weights is None:
            self.initial_weights = {
                'gcn': 0.25,
                'gat': 0.25, 
                'graphsage': 0.25,
                'mpnn': 0.25
            }
        
        if self.monitoring_metrics is None:
            self.monitoring_metrics = ['mae', 'mse', 'mape', 'directional_accuracy']

class ModelPerformanceTracker:
    """
    Tracker performance models for adaptive weights
    
    Real-time Performance Analytics
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history = defaultdict(list)
        self.current_weights = {}
        
    def update_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Update metrics performance model"""
        for metric_name, value in metrics.items():
            key = f"{model_name}_{metric_name}"
            self.performance_history[key].append(value)
            
            # Limitation size windows
            if len(self.performance_history[key]) > self.window_size:
                self.performance_history[key] = self.performance_history[key][-self.window_size:]
    
    def get_recent_performance(self, model_name: str, metric: str) -> List[float]:
        """Get metrics performance"""
        key = f"{model_name}_{metric}"
        return self.performance_history.get(key, [])
    
    def compute_adaptive_weights(self, models: List[str], metric: str = 'mae', min_weight: float = 0.1) -> Dict[str, float]:
        """
        Computation adaptive weights on basis performance
        
        Model with best weight
        """
        weights = {}
        performance_scores = {}
        
        # Computing averages performance
        for model_name in models:
            recent_performance = self.get_recent_performance(model_name, metric)
            if recent_performance:
                # For MAE and MSE - the smaller, the better
                if metric in ['mae', 'mse']:
                    avg_performance = np.mean(recent_performance)
                    performance_scores[model_name] = 1.0 / (avg_performance + 1e-8)
                else:
                    # For accuracy metrics - than more, that better
                    performance_scores[model_name] = np.mean(recent_performance)
            else:
                # If no history - equal weight
                performance_scores[model_name] = 1.0
        
        # Normalization weights
        total_score = sum(performance_scores.values())
        for model_name in models:
            weight = performance_scores[model_name] / total_score
            # Applying minimum weight
            weights[model_name] = max(weight, min_weight)
        
        # after minimum weights
        total_weight = sum(weights.values())
        for model_name in weights:
            weights[model_name] /= total_weight
        
        self.current_weights = weights
        return weights
    
    def get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary by performance all models"""
        summary = {}
        
        for key, values in self.performance_history.items():
            if '_' in key:
                model_name, metric_name = key.rsplit('_', 1)
                if model_name not in summary:
                    summary[model_name] = {}
                
                if values:
                    summary[model_name][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'recent': values[-5:] if len(values) >= 5 else values,
                        'trend': 'improving' if len(values) > 1 and values[-1] < values[0] else 'stable'
                    }
        
        return summary

class UncertaintyQuantifier:
    """
     for uncertainty in ensemble predictions
    
    Risk Assessment and Uncertainty Management
    """
    
    def __init__(self, method: str = 'ensemble_variance'):
        self.method = method
        
    def compute_uncertainty(
        self, 
        predictions: List[torch.Tensor], 
        models: Optional[List[nn.Module]] = None,
        data: Optional[Data] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computation uncertainty predictions
        
        Args:
            predictions: List predictions from different models
            models: List models (for MC dropout)
            data: Input data (for MC dropout)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (mean_prediction, uncertainty)
        """
        if self.method == 'ensemble_variance':
            return self._ensemble_variance_uncertainty(predictions)
        elif self.method == 'monte_carlo_dropout':
            if models is None or data is None:
                raise ValueError("Models and data for MC dropout")
            return self._monte_carlo_dropout_uncertainty(models, data)
        else:
            raise ValueError(f"Unknown method uncertainty: {self.method}")
    
    def _ensemble_variance_uncertainty(self, predictions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ on basis variance between models"""
        # predictions
        stacked_predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, output_dim]
        
        # Average prediction
        mean_prediction = torch.mean(stacked_predictions, dim=0)
        
        # Variance as measure uncertainty
        uncertainty = torch.var(stacked_predictions, dim=0)
        
        return mean_prediction, uncertainty
    
    def _monte_carlo_dropout_uncertainty(
        self, 
        models: List[nn.Module], 
        data: Data, 
        num_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout uncertainty for each model"""
        all_predictions = []
        
        for model in models:
            model.train()  # Enable dropout
            model_predictions = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    pred = model(data)
                    model_predictions.append(pred)
            
            # Average prediction from this model
            model_mean = torch.mean(torch.stack(model_predictions), dim=0)
            all_predictions.append(model_mean)
            
            model.eval()  # Return in eval mode
        
        # Ensemble uncertainty
        return self._ensemble_variance_uncertainty(all_predictions)
    
    def confidence_interval(
        self, 
        mean_prediction: torch.Tensor, 
        uncertainty: torch.Tensor, 
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computation
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (lower_bound, upper_bound)
        """
        # Z-score for specified level
        from scipy.stats import norm
        z_score = norm.ppf((1 + confidence_level) / 2)
        
        std_dev = torch.sqrt(uncertainty)
        margin = z_score * std_dev
        
        lower_bound = mean_prediction - margin
        upper_bound = mean_prediction + margin
        
        return lower_bound, upper_bound

class GraphNeuralNetworkEnsemble(nn.Module):
    """
    Production-Ready GNN Ensemble for crypto trading
    
    Scalable Ensemble Learning Architecture
    """
    
    def __init__(self, config: EnsembleConfig):
        super().__init__()
        self.config = config
        
        # Initialize individual models
        self.models = nn.ModuleDict()
        self._initialize_models()
        
        # System weights ensemble
        self.model_weights = nn.Parameter(
            torch.tensor(list(self.config.initial_weights.values()), dtype=torch.float32),
            requires_grad=self.config.use_meta_learning
        )
        self.model_names = list(self.config.initial_weights.keys())
        
        # Tracker performance
        self.performance_tracker = ModelPerformanceTracker(self.config.performance_window_size)
        
        # Quantifier uncertainty
        if self.config.enable_uncertainty:
            self.uncertainty_quantifier = UncertaintyQuantifier(self.config.uncertainty_method)
        
        # Meta-learning for weights (optionally)
        if self.config.use_meta_learning:
            self.meta_optimizer = torch.optim.Adam(
                [self.model_weights], 
                lr=self.config.meta_learning_lr
            )
        
        # Monitor
        self.step_counter = 0
        self.monitoring_data = defaultdict(list)
        
        logger.info(f"Initialized GNN Ensemble with {len(self.models)} models")
    
    def _initialize_models(self) -> None:
        """Initialize individual GNN models"""
        
        # GCN
        if self.config.gcn_config is not None:
            self.models['gcn'] = GraphConvolutionalNetwork(self.config.gcn_config)
        
        # GAT
        if self.config.gat_config is not None:
            self.models['gat'] = GraphAttentionNetwork(self.config.gat_config)
        
        # GraphSAGE
        if self.config.graphsage_config is not None:
            self.models['graphsage'] = GraphSAGE(self.config.graphsage_config)
        
        # MPNN
        if self.config.mpnn_config is not None:
            self.models['mpnn'] = MessagePassingNeuralNetwork(self.config.mpnn_config)
        
        if len(self.models) == 0:
            raise ValueError("Configuration for at least one model must be specified")
        
        # Updating weights if model from configuration
        active_models = list(self.models.keys())
        if set(active_models) != set(self.config.initial_weights.keys()):
            # Recalculating weights for active models
            equal_weight = 1.0 / len(active_models)
            new_weights = {model: equal_weight for model in active_models}
            self.config.initial_weights = new_weights
            self.model_names = active_models
    
    def forward(self, data: Data, return_individual: bool = False, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through ensemble
        
        Args:
            data: Input data graph
            return_individual: Return predictions
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            Union[torch.Tensor, Tuple]: Ensemble prediction and optionally information
        """
        individual_predictions = []
        individual_results = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'gat' and hasattr(model, 'forward'):
                    # GAT returns also attention weights
                    pred, _ = model(data) # attention weights in ensemble
                else:
                    pred = model(data)
                
                individual_predictions.append(pred)
                individual_results[model_name] = pred
                
            except Exception as e:
                logger.warning(f"Error in model {model_name}: {e}")
                # Create dummy prediction not ensemble
                dummy_pred = torch.zeros_like(individual_predictions[0] if individual_predictions else torch.zeros((data.x.size(0) if hasattr(data, 'batch') else 1, 1)))
                individual_predictions.append(dummy_pred)
                individual_results[model_name] = dummy_pred
        
        # Ensemble aggregation
        ensemble_prediction = self._aggregate_predictions(individual_predictions)
        
        # Results for return
        results = [ensemble_prediction]
        
        if return_individual:
            results.append(individual_results)
        
        if return_uncertainty and self.config.enable_uncertainty:
            _, uncertainty = self.uncertainty_quantifier.compute_uncertainty(individual_predictions)
            results.append(uncertainty)
        
        # Update step counter
        self.step_counter += 1
        
        return results[0] if len(results) == 1 else tuple(results)
    
    def _aggregate_predictions(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Aggregation predictions according to ensemble method"""
        
        if self.config.ensemble_method == 'weighted_average':
            # Weighted average
            weights = F.softmax(self.model_weights, dim=0)  # Normalization weights
            
            weighted_predictions = []
            for i, pred in enumerate(predictions):
                weighted_predictions.append(weights[i] * pred)
            
            return torch.sum(torch.stack(weighted_predictions), dim=0)
        
        elif self.config.ensemble_method == 'voting':
            # Majority voting (for classification)
            # For regression use median
            stacked_predictions = torch.stack(predictions, dim=0)
            return torch.median(stacked_predictions, dim=0)[0]
        
        elif self.config.ensemble_method == 'dynamic_weighting':
            # Adaptive weights on basis recent performance
            if hasattr(self, 'performance_tracker') and self.performance_tracker.current_weights:
                weights = []
                for model_name in self.model_names:
                    weight = self.performance_tracker.current_weights.get(model_name, 1.0/len(predictions))
                    weights.append(weight)
                
                weights = torch.tensor(weights, device=predictions[0].device)
                weights = weights / torch.sum(weights)  # Normalization
                
                weighted_predictions = []
                for i, pred in enumerate(predictions):
                    weighted_predictions.append(weights[i] * pred)
                
                return torch.sum(torch.stack(weighted_predictions), dim=0)
            else:
                # Fallback on averaging
                return torch.mean(torch.stack(predictions), dim=0)
        
        else:
            # Default: averaging
            return torch.mean(torch.stack(predictions), dim=0)
    
    def update_model_weights(self, performance_metrics: Dict[str, Dict[str, float]]) -> None:
        """Update weights models on basis performance"""
        
        # Updating performance
        for model_name, metrics in performance_metrics.items():
            if model_name in self.model_names:
                self.performance_tracker.update_performance(model_name, metrics)
        
        # Recalculating weights if use
        if self.config.use_adaptive_weights and self.step_counter % self.config.weight_update_frequency == 0:
            new_weights = self.performance_tracker.compute_adaptive_weights(
                self.model_names, 
                metric='mae', # Possible make configurable
                min_weight=self.config.min_weight
            )
            
            logger.info(f" weights models: {new_weights}")
    
    def predict_with_uncertainty(self, data: Union[Data, List[Data]]) -> Dict[str, np.ndarray]:
        """
        Prediction with uncertainty
        
        Returns:
            Dict prediction, uncertainty, confidence_intervals
        """
        self.eval()
        
        # Preparation data
        if isinstance(data, list):
            batch = Batch.from_data_list(data)
        else:
            batch = data
        
        with torch.no_grad():
            # Get individual predictions
            individual_predictions = []
            for model_name, model in self.models.items():
                try:
                    if model_name == 'gat':
                        pred, _ = model(batch)
                    else:
                        pred = model(batch)
                    individual_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error in model {model_name}: {e}")
                    continue
            
            if not individual_predictions:
                raise RuntimeError("All model in ensemble errors")
            
            # Ensemble prediction
            ensemble_pred = self._aggregate_predictions(individual_predictions)
            
            # Uncertainty quantification
            if self.config.enable_uncertainty:
                mean_pred, uncertainty = self.uncertainty_quantifier.compute_uncertainty(individual_predictions)
                
                #
                lower_bound, upper_bound = self.uncertainty_quantifier.confidence_interval(
                    mean_pred, uncertainty, confidence_level=0.95
                )
                
                return {
                    'prediction': ensemble_pred.cpu().numpy(),
                    'mean_prediction': mean_pred.cpu().numpy(),
                    'uncertainty': uncertainty.cpu().numpy(),
                    'confidence_interval_lower': lower_bound.cpu().numpy(),
                    'confidence_interval_upper': upper_bound.cpu().numpy(),
                    'individual_predictions': {
                        f'model_{i}': pred.cpu().numpy() 
                        for i, pred in enumerate(individual_predictions)
                    }
                }
            else:
                return {
                    'prediction': ensemble_pred.cpu().numpy(),
                    'individual_predictions': {
                        f'model_{i}': pred.cpu().numpy()
                        for i, pred in enumerate(individual_predictions)
                    }
                }
    
    def get_model_importance(self) -> Dict[str, float]:
        """Get each model in ensemble"""
        if self.config.use_meta_learning:
            weights = F.softmax(self.model_weights, dim=0)
            return {name: weight.item() for name, weight in zip(self.model_names, weights)}
        else:
            return self.performance_tracker.current_weights if self.performance_tracker.current_weights else self.config.initial_weights
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance"""
        return {
            'model_weights': self.get_model_importance(),
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'ensemble_config': {
                'method': self.config.ensemble_method,
                'uncertainty_enabled': self.config.enable_uncertainty,
                'adaptive_weights': self.config.use_adaptive_weights,
                'meta_learning': self.config.use_meta_learning
            },
            'monitoring_data': dict(self.monitoring_data) if self.config.enable_monitoring else {}
        }

class CryptoGNNEnsembleTrainer:
    """
    Specialized trainer for GNN Ensemble
    
    Enterprise Ensemble Training Pipeline
    """
    
    def __init__(self, ensemble: GraphNeuralNetworkEnsemble, config: EnsembleConfig):
        self.ensemble = ensemble
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizers for different components
        self.model_optimizers = {}
        for model_name, model in self.ensemble.models.items():
            self.model_optimizers[model_name] = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Scheduler for ensemble
        self.schedulers = {}
        for model_name, optimizer in self.model_optimizers.items():
            self.schedulers[model_name] = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.8, patience=10
            )
        
        self.ensemble.to(self.device)
        
        # History training
        self.history = {
            'ensemble_loss': [], 'ensemble_mae': [],
            'individual_losses': {name: [] for name in self.ensemble.models.keys()},
            'model_weights_history': [],
            'uncertainty_metrics': []
        }
        
        logger.info(f"Ensemble trainer ready with {len(self.ensemble.models)} models")
    
    def train_step(self, batch: Data) -> Dict[str, float]:
        """Step training ensemble"""
        batch = batch.to(self.device)
        
        # Training each model
        individual_losses = {}
        individual_metrics = {}
        
        for model_name, model in self.ensemble.models.items():
            optimizer = self.model_optimizers[model_name]
            
            model.train()
            optimizer.zero_grad()
            
            try:
                # Forward pass
                if model_name == 'gat':
                    predictions, _ = model(batch)
                else:
                    predictions = model(batch)
                
                targets = batch.y.view(-1, 1).float()
                
                # Loss computation
                loss = F.mse_loss(predictions, targets)
                mae = F.l1_loss(predictions, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                individual_losses[model_name] = loss.item()
                individual_metrics[model_name] = {'mae': mae.item(), 'mse': loss.item()}
                
            except Exception as e:
                logger.warning(f"Error in training model {model_name}: {e}")
                individual_losses[model_name] = float('inf')
                individual_metrics[model_name] = {'mae': float('inf'), 'mse': float('inf')}
        
        # Update weights ensemble
        self.ensemble.update_model_weights(individual_metrics)
        
        # Ensemble prediction for metrics
        self.ensemble.eval()
        with torch.no_grad():
            ensemble_pred = self.ensemble(batch)
            targets = batch.y.view(-1, 1).float()
            ensemble_loss = F.mse_loss(ensemble_pred, targets)
            ensemble_mae = F.l1_loss(ensemble_pred, targets)
        
        self.ensemble.train()
        
        return {
            'ensemble_loss': ensemble_loss.item(),
            'ensemble_mae': ensemble_mae.item(),
            'individual_losses': individual_losses,
            'model_weights': self.ensemble.get_model_importance()
        }
    
    def validate_step(self, batch: Data) -> Dict[str, float]:
        """Validation ensemble"""
        self.ensemble.eval()
        batch = batch.to(self.device)
        
        with torch.no_grad():
            # Ensemble prediction
            if self.config.enable_uncertainty:
                results = self.ensemble.predict_with_uncertainty(batch)
                ensemble_pred = torch.from_numpy(results['prediction']).to(self.device)
                uncertainty = torch.from_numpy(results['uncertainty']).to(self.device)
            else:
                ensemble_pred = self.ensemble(batch)
                uncertainty = None
            
            targets = batch.y.view(-1, 1).float()
            
            # Metrics
            ensemble_loss = F.mse_loss(ensemble_pred, targets)
            ensemble_mae = F.l1_loss(ensemble_pred, targets)
            
            # Directional accuracy (for financial data)
            pred_direction = torch.sign(ensemble_pred)
            target_direction = torch.sign(targets)
            directional_accuracy = (pred_direction == target_direction).float().mean()
            
            metrics = {
                'loss': ensemble_loss.item(),
                'mae': ensemble_mae.item(),
                'directional_accuracy': directional_accuracy.item()
            }
            
            if uncertainty is not None:
                avg_uncertainty = torch.mean(uncertainty).item()
                metrics['avg_uncertainty'] = avg_uncertainty
            
            return metrics
    
    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Training one epochs ensemble"""
        train_metrics = {
            'ensemble_loss': [], 'ensemble_mae': [],
            'individual_losses': {name: [] for name in self.ensemble.models.keys()}
        }
        
        for batch in train_loader:
            metrics = self.train_step(batch)
            
            train_metrics['ensemble_loss'].append(metrics['ensemble_loss'])
            train_metrics['ensemble_mae'].append(metrics['ensemble_mae'])
            
            for model_name, loss in metrics['individual_losses'].items():
                train_metrics['individual_losses'][model_name].append(loss)
        
        # Aggregation metrics
        epoch_metrics = {
            'train_ensemble_loss': np.mean(train_metrics['ensemble_loss']),
            'train_ensemble_mae': np.mean(train_metrics['ensemble_mae'])
        }
        
        for model_name, losses in train_metrics['individual_losses'].items():
            epoch_metrics[f'train_{model_name}_loss'] = np.mean([l for l in losses if l != float('inf')])
        
        # Validation
        if val_loader is not None:
            val_metrics = {'loss': [], 'mae': [], 'directional_accuracy': [], 'avg_uncertainty': []}
            
            for batch in val_loader:
                metrics = self.validate_step(batch)
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
            
            for key, values in val_metrics.items():
                if values:  # If there is values
                    epoch_metrics[f'val_{key}'] = np.mean(values)
            
            # Update schedulers
            for model_name, scheduler in self.schedulers.items():
                scheduler.step(epoch_metrics.get('val_loss', epoch_metrics['train_ensemble_loss']))
        
        # Save current weights models
        current_weights = self.ensemble.get_model_importance()
        epoch_metrics['model_weights'] = current_weights
        
        # Update history
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
        
        self.history['model_weights_history'].append(current_weights)
        
        return epoch_metrics
    
    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble"""
        save_dict = {
            'ensemble_state_dict': self.ensemble.state_dict(),
            'model_optimizers': {name: opt.state_dict() for name, opt in self.model_optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'config': self.config,
            'history': self.history,
            'performance_tracker': self.ensemble.performance_tracker,
            'model_weights': self.ensemble.get_model_importance()
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Ensemble in {filepath}")
    
    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.ensemble.load_state_dict(checkpoint['ensemble_state_dict'])
        
        for name, state_dict in checkpoint['model_optimizers'].items():
            if name in self.model_optimizers:
                self.model_optimizers[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['schedulers'].items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(state_dict)
        
        self.history = checkpoint.get('history', self.history)
        self.ensemble.performance_tracker = checkpoint.get('performance_tracker', self.ensemble.performance_tracker)
        
        logger.info(f"Ensemble from {filepath}")

def create_crypto_gnn_ensemble(
    input_dim: int,
    output_dim: int = 1,
    hidden_dim: int = 128,
    enable_all_models: bool = True,
    **kwargs
) -> Tuple[GraphNeuralNetworkEnsemble, CryptoGNNEnsembleTrainer]:
    """
    Factory function for creation GNN Ensemble
    
    Factory with Full Configuration
    """
    # Create configurations for each model
    model_configs = {}
    
    if enable_all_models:
        # GCN
        model_configs['gcn_config'] = GCNConfig(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2]
        )
        
        # GAT
        model_configs['gat_config'] = GATConfig(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            num_heads=[4, 2, 1]
        )
        
        # GraphSAGE
        model_configs['graphsage_config'] = GraphSAGEConfig(
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2]
        )
        
        # MPNN
        model_configs['mpnn_config'] = MPNNConfig(
            node_input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim
        )
    
    # Ensemble configuration
    ensemble_config = EnsembleConfig(
        **model_configs,
        **kwargs
    )
    
    ensemble = GraphNeuralNetworkEnsemble(ensemble_config)
    trainer = CryptoGNNEnsembleTrainer(ensemble, ensemble_config)
    
    return ensemble, trainer

# Export for use
__all__ = [
    'GraphNeuralNetworkEnsemble',
    'EnsembleConfig',
    'CryptoGNNEnsembleTrainer',
    'ModelPerformanceTracker',
    'UncertaintyQuantifier',
    'create_crypto_gnn_ensemble'
]