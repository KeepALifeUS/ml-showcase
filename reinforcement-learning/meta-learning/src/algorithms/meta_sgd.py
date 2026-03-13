"""
Meta-SGD Implementation
Learnable Learning Rates for Crypto Trading

Implementation Meta-SGD algorithm with trainable learning rates for of each parameter.
Adaptive optimization for cryptocurrency trading strategies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
from collections import OrderedDict
import copy

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class MetaSGDConfig:
    """Configuration for Meta-SGD algorithm"""
    
    # Main parameters
    meta_lr: float = 0.001  # Meta-speed training
    num_inner_steps: int = 5  # Number inner steps
    
    # Parameters tasks
    num_support: int = 5  # Size support set
    num_query: int = 15  # Size query set
    
    # Initialization learning rates
    init_inner_lr: float = 0.01  # Initial values inner learning rates
    lr_init_range: Tuple[float, float] = (0.001, 0.1)  # Range initialization
    
    # Optimization
    outer_lr: float = 0.001  # Speed training for meta-parameters
    grad_clip: Optional[float] = 1.0  # Trimming gradients
    weight_decay: float = 0.0001  # L2 regularization
    
    # Regularization learning rates
    lr_regularization: float = 0.001  # Regularization for learning rates
    lr_bounds: Tuple[float, float] = (1e-5, 1.0)  # Boundaries for learning rates
    
    # Monitoring
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MetaSGD:
    """
    Meta-SGD: Learning to Learn by Gradient Descent by Gradient Descent
    
    Adaptive Meta-Learning System
    - Learnable per-parameter learning rates
    - Second-order gradient optimization
    - Automatic hyperparameter tuning
    - Crypto market specialization
    
    Meta-SGD trains as weights model, so and learning rates for fast adaptation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MetaSGDConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialization Meta-SGD
        
        Args:
            model: Base model for meta-training
            config: Configuration Meta-SGD
            logger: Logger for monitoring
        """
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialization trainable learning rates
        self.inner_lrs = self._initialize_learning_rates()
        
        # Meta-optimizer for model and learning rates
        self.meta_optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.inner_lrs.values()),
            lr=config.outer_lr,
            weight_decay=config.weight_decay
        )
        
        # Utilities
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # State
        self.global_step = 0
        self.best_meta_loss = float('inf')
        
        self.logger.info(f"Meta-SGD initialized with config: {config}")
        self.logger.info(f"Learning rates shape: {len(self.inner_lrs)} parameters")
    
    def _initialize_learning_rates(self) -> OrderedDict:
        """
        Initialization trainable learning rates for of each parameter model
        
        Returns:
            OrderedDict with learning rates for of each parameter
        """
        inner_lrs = OrderedDict()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Initialize learning rate with that same form that and parameter
                lr_param = torch.full_like(
                    param.data,
                    self.config.init_inner_lr,
                    requires_grad=True,
                    device=self.config.device
                )
                
                # Add small noise for diversity
                with torch.no_grad():
                    noise = torch.randn_like(lr_param) * 0.001
                    lr_param.data.add_(noise)
                    
                    # Trim in allowed boundaries
                    lr_param.data.clamp_(
                        self.config.lr_bounds[0],
                        self.config.lr_bounds[1]
                    )
                
                inner_lrs[name] = lr_param
        
        return inner_lrs
    
    def inner_loop(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_state: OrderedDict,
        create_graph: bool = True
    ) -> Tuple[OrderedDict, List[float], Dict[str, torch.Tensor]]:
        """
        Inner loop with trainable learning rates
        
        Args:
            support_data: Data for training on task
            support_labels: Labels for training
            model_state: Current state model
            create_graph: Create whether graph computations
            
        Returns:
            Tuple from adapted parameters, losses and used learning rates
        """
        adapted_params = OrderedDict()
        for name, param in model_state.items():
            adapted_params[name] = param.clone()
        
        inner_losses = []
        lr_history = {name: [] for name in self.inner_lrs.keys()}
        
        for step in range(self.config.num_inner_steps):
            # Forward pass
            predictions = self._forward_with_params(
                support_data, adapted_params
            )
            
            # Compute loss
            loss = nn.functional.mse_loss(predictions, support_labels)
            inner_losses.append(loss.item())
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=create_graph,
                allow_unused=True
            )
            
            # Update parameters using learnable learning rates
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None and name in self.inner_lrs:
                    # Apply per-parameter learning rate
                    current_lr = torch.clamp(
                        self.inner_lrs[name],
                        self.config.lr_bounds[0],
                        self.config.lr_bounds[1]
                    )
                    
                    adapted_params[name] = param - current_lr * grad
                    
                    # Save history learning rates
                    lr_history[name].append(current_lr.detach().clone())
        
        return adapted_params, inner_losses, lr_history
    
    def _forward_with_params(
        self,
        data: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """Forward pass with specified parameters"""
        # Create functional copy model
        def fmodel(x):
            # Simple implementation for demonstration
            # IN reality needed more complex functional replacement
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
                if name in params:
                    param.data = params[name]
            
            try:
                output = self.model(x)
            finally:
                # Restore parameters
                for name, param in self.model.named_parameters():
                    param.data = original_params[name]
            
            return output
        
        return fmodel(data)
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        One step meta-training Meta-SGD
        
        Args:
            task_batch: Batch tasks for meta-training
            
        Returns:
            Dictionary with metrics
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        adaptation_losses = []
        query_accuracies = []
        lr_stats = []
        
        # Retrieve current parameters model
        model_state = OrderedDict(self.model.named_parameters())
        
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Inner loop with trainable learning rates
            adapted_params, inner_losses, lr_history = self.inner_loop(
                support_data,
                support_labels,
                model_state,
                create_graph=True
            )
            
            # Query loss for outer loop
            query_predictions = self._forward_with_params(
                query_data, adapted_params
            )
            meta_loss = nn.functional.mse_loss(query_predictions, query_labels)
            meta_losses.append(meta_loss)
            
            # Metrics
            adaptation_losses.extend(inner_losses)
            query_accuracy = self._compute_accuracy(
                query_predictions, query_labels
            )
            query_accuracies.append(query_accuracy)
            
            # Statistics learning rates
            for name, lr_hist in lr_history.items():
                if lr_hist:
                    avg_lr = torch.stack(lr_hist).mean()
                    lr_stats.append(avg_lr.item())
        
        # Aggregate meta-loss
        total_meta_loss = torch.stack(meta_losses).mean()
        
        # Add regularization for learning rates
        lr_regularization = self._compute_lr_regularization()
        total_loss = total_meta_loss + lr_regularization
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip:
            # Trimming gradients model
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
            # Trimming gradients learning rates
            torch.nn.utils.clip_grad_norm_(
                self.inner_lrs.values(), self.config.grad_clip
            )
        
        # Update meta-parameters
        self.meta_optimizer.step()
        
        # Limit learning rates in allowed boundaries
        self._clamp_learning_rates()
        
        # Collect metrics
        metrics = {
            'meta_loss': total_meta_loss.item(),
            'lr_regularization': lr_regularization.item(),
            'total_loss': total_loss.item(),
            'adaptation_loss': np.mean(adaptation_losses),
            'query_accuracy': np.mean(query_accuracies),
            'avg_inner_lr': np.mean(lr_stats) if lr_stats else 0.0,
            'lr_std': np.std(lr_stats) if lr_stats else 0.0,
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.model.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def _compute_lr_regularization(self) -> torch.Tensor:
        """
        Computes regularization for learning rates
        
        Returns:
            Tensor with value regularization
        """
        if self.config.lr_regularization <= 0:
            return torch.tensor(0.0, device=self.config.device)
        
        lr_penalty = torch.tensor(0.0, device=self.config.device)
        
        for lr_param in self.inner_lrs.values():
            # L2 regularization
            lr_penalty += torch.sum(lr_param ** 2)
            
            # Penalty for extreme values
            too_large = torch.sum(torch.relu(lr_param - 0.1) ** 2)
            too_small = torch.sum(torch.relu(0.001 - lr_param) ** 2)
            lr_penalty += too_large + too_small
        
        return self.config.lr_regularization * lr_penalty
    
    def _clamp_learning_rates(self) -> None:
        """Limits learning rates in allowed boundaries"""
        with torch.no_grad():
            for lr_param in self.inner_lrs.values():
                lr_param.data.clamp_(
                    self.config.lr_bounds[0],
                    self.config.lr_bounds[1]
                )
    
    def meta_validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Validation meta-model"""
        all_metrics = []
        
        with torch.no_grad():
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Adaptation without gradients for outer loop
                adapted_params, adaptation_losses, _ = self.inner_loop(
                    support_data,
                    support_labels,
                    OrderedDict(self.model.named_parameters()),
                    create_graph=False
                )
                
                # Validation on query set
                query_predictions = self._forward_with_params(
                    query_data, adapted_params
                )
                query_loss = nn.functional.mse_loss(
                    query_predictions, query_labels
                ).item()
                
                query_accuracy = self._compute_accuracy(
                    query_predictions, query_labels
                )
                
                all_metrics.append({
                    'query_loss': query_loss,
                    'query_accuracy': query_accuracy,
                    'adaptation_loss': np.mean(adaptation_losses)
                })
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def few_shot_adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        num_adaptation_steps: Optional[int] = None
    ) -> nn.Module:
        """
        Fast adaptation with using trained learning rates
        
        Args:
            support_data: Data for adaptation
            support_labels: Labels for adaptation
            num_adaptation_steps: Number steps adaptation
            
        Returns:
            Adapted model
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.num_inner_steps
        
        # Create copy model
        adapted_model = copy.deepcopy(self.model)
        
        # Adaptation with trained learning rates
        for step in range(num_adaptation_steps):
            # Forward pass
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=False,
                allow_unused=True
            )
            
            # Update parameters with trained learning rates
            with torch.no_grad():
                for (name, param), grad in zip(
                    adapted_model.named_parameters(), grads
                ):
                    if grad is not None and name in self.inner_lrs:
                        current_lr = torch.clamp(
                            self.inner_lrs[name],
                            self.config.lr_bounds[0],
                            self.config.lr_bounds[1]
                        )
                        param.data -= current_lr * grad
            
            if step % 5 == 0:
                self.logger.debug(f"Adaptation step {step}, loss: {loss.item():.4f}")
        
        return adapted_model
    
    def get_learning_rates_stats(self) -> Dict[str, Any]:
        """Gets statistics trained learning rates"""
        stats = {}
        
        with torch.no_grad():
            all_lrs = []
            for name, lr_param in self.inner_lrs.items():
                lr_values = lr_param.flatten()
                all_lrs.extend(lr_values.cpu().numpy())
                
                stats[f'{name}_mean'] = lr_values.mean().item()
                stats[f'{name}_std'] = lr_values.std().item()
                stats[f'{name}_min'] = lr_values.min().item()
                stats[f'{name}_max'] = lr_values.max().item()
            
            # Total statistics
            all_lrs = np.array(all_lrs)
            stats['global_lr_mean'] = np.mean(all_lrs)
            stats['global_lr_std'] = np.std(all_lrs)
            stats['global_lr_min'] = np.min(all_lrs)
            stats['global_lr_max'] = np.max(all_lrs)
        
        return stats
    
    def _compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.1
    ) -> float:
        """Computes accuracy for regression"""
        with torch.no_grad():
            errors = torch.abs(predictions - labels)
            correct = (errors < threshold).float()
            return correct.mean().item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saving checkpoint model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'inner_lrs': {name: lr.data for name, lr in self.inner_lrs.items()},
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_meta_loss': self.best_meta_loss
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Meta-SGD checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loading checkpoint model"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore learning rates
        for name, lr_data in checkpoint['inner_lrs'].items():
            if name in self.inner_lrs:
                self.inner_lrs[name].data = lr_data.to(self.config.device)
        
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_meta_loss = checkpoint['best_meta_loss']
        
        self.logger.info(f"Meta-SGD checkpoint loaded from {filepath}")


class MetaSGDTrainer:
    """
    Trainer for Meta-SGD with enterprise patterns
    
    Features:
    - Adaptive learning rate monitoring
    - Learning rate visualization
    - Advanced checkpoint management
    """
    
    def __init__(
        self,
        meta_sgd: MetaSGD,
        train_loader: Any,
        val_loader: Any,
        config: MetaSGDConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.meta_sgd = meta_sgd
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics_history = []
        self.lr_stats_history = []
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        lr_analysis_interval: int = 50
    ) -> Dict[str, List[float]]:
        """Main loop training Meta-SGD"""
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self.meta_sgd.meta_validate(self.val_loader)
            
            # Learning rates analysis
            if epoch % lr_analysis_interval == 0:
                lr_stats = self.meta_sgd.get_learning_rates_stats()
                self.lr_stats_history.append(lr_stats)
                self._log_lr_stats(epoch, lr_stats)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_metrics(epoch, epoch_metrics)
            
            # Checkpoint saving
            if epoch % 100 == 0:
                self.meta_sgd.save_checkpoint(
                    f"{save_dir}/meta_sgd_epoch_{epoch}.pt"
                )
        
        return self._compile_metrics_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Training one epoch"""
        epoch_metrics = []
        
        for batch in tqdm(self.train_loader, desc="Meta-SGD Training"):
            metrics = self.meta_sgd.meta_train_step(batch)
            epoch_metrics.append(metrics)
        
        return {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logging metrics"""
        self.logger.info(f"Meta-SGD Epoch {epoch}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _log_lr_stats(self, epoch: int, lr_stats: Dict[str, Any]) -> None:
        """Logging statistics learning rates"""
        self.logger.info(f"Learning Rates Stats at Epoch {epoch}:")
        self.logger.info(f"  Global LR Mean: {lr_stats['global_lr_mean']:.6f}")
        self.logger.info(f"  Global LR Std: {lr_stats['global_lr_std']:.6f}")
        self.logger.info(f"  Global LR Range: [{lr_stats['global_lr_min']:.6f}, {lr_stats['global_lr_max']:.6f}]")
    
    def _compile_metrics_history(self) -> Dict[str, List[float]]:
        """Compilation history metrics"""
        if not self.metrics_history:
            return {}
        
        compiled = {}
        for key in self.metrics_history[0].keys():
            compiled[key] = [m[key] for m in self.metrics_history]
        return compiled