"""
MAML (Model-Agnostic Meta-Learning) Implementation
Scalable Meta-Learning for Crypto Trading

Implementation algorithm MAML for fast adaptation to new cryptocurrency markets.
Is based on principles gradient-based meta-learning with support higher derivatives.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import higher
from collections import OrderedDict

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics
from ..models.meta_model import MetaModel


@dataclass
class MAMLConfig:
    """Configuration for MAML algorithm"""
    
    # Main parameters
    inner_lr: float = 0.01  # Speed training on inner loop
    outer_lr: float = 0.001  # Speed training on external loop
    num_inner_steps: int = 5  # Number steps gradient descent on task
    
    # Parameters tasks
    num_support: int = 5  # Number examples in support set
    num_query: int = 15  # Number examples in query set
    
    # Regularization
    first_order: bool = False  # Use whether first order (Reptile-like)
    allow_unused: bool = True  # Resolve unused parameters
    allow_nograd: bool = True  # Resolve parameters without gradient
    
    # Optimization
    grad_clip: Optional[float] = 1.0  # Trimming gradients
    weight_decay: float = 0.0001  # L2 regularization
    
    # Monitoring
    log_interval: int = 10  # Interval logging
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML) Implementation
    
    Enterprise Meta-Learning System
    - Scalable gradient computation
    - Memory-efficient implementation
    - Production-ready monitoring
    - Crypto market specialization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialization MAML
        
        Args:
            model: Base model for meta-training
            config: Configuration MAML
            logger: Logger for monitoring
        """
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Optimizer for outer loop
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.outer_lr,
            weight_decay=config.weight_decay
        )
        
        # Utilities
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # State
        self.global_step = 0
        self.best_meta_loss = float('inf')
        
        self.logger.info(f"MAML initialized with config: {config}")
    
    def inner_loop(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        model_state: OrderedDict,
        create_graph: bool = True
    ) -> Tuple[OrderedDict, List[float]]:
        """
        Inner loop training on specific task
        
        Args:
            support_data: Data for training on task
            support_labels: Labels for training
            model_state: Current state model
            create_graph: Create whether graph computations for second derivative
            
        Returns:
            Tuple from adapted parameters and losses
        """
        # Create copy model for inner loop
        adapted_params = OrderedDict()
        for name, param in model_state.items():
            adapted_params[name] = param.clone()
        
        inner_losses = []
        
        for step in range(self.config.num_inner_steps):
            # Forward pass with current parameters
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
                allow_unused=self.config.allow_unused
            )
            
            # Update parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.config.inner_lr * grad
        
        return adapted_params, inner_losses
    
    def _forward_with_params(
        self,
        data: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """
        Forward pass with specified parameters
        
        Args:
            data: Input data
            params: Parameters model
            
        Returns:
            Predictions model
        """
        # Temporary replace model parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            if name in params:
                param.data = params[name]
        
        try:
            predictions = self.model(data)
        finally:
            # Restore original parameters
            for name, param in self.model.named_parameters():
                param.data = original_params[name]
        
        return predictions
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        One step meta-training on batch tasks
        
        Args:
            task_batch: Batch tasks with support/query sets
            
        Returns:
            Dictionary with metrics
        """
        self.meta_optimizer.zero_grad()
        
        meta_losses = []
        adaptation_losses = []
        query_accuracies = []
        
        # Retrieve current parameters model
        model_state = OrderedDict(self.model.named_parameters())
        
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Inner loop - adaptation to task
            adapted_params, inner_losses = self.inner_loop(
                support_data,
                support_labels,
                model_state,
                create_graph=not self.config.first_order
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
        
        # Aggregate meta-loss
        total_meta_loss = torch.stack(meta_losses).mean()
        
        # Backward pass
        total_meta_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )
        
        # Update meta-parameters
        self.meta_optimizer.step()
        
        # Collect metrics
        metrics = {
            'meta_loss': total_meta_loss.item(),
            'adaptation_loss': np.mean(adaptation_losses),
            'query_accuracy': np.mean(query_accuracies),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.model.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def meta_validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Validation meta-model on validation tasks
        
        Args:
            validation_tasks: Tasks for validation
            
        Returns:
            Dictionary with metrics validation
        """
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Adaptation without gradients for outer loop
                adapted_params, adaptation_losses = self.inner_loop(
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
        
        self.model.train()
        
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
        Fast adaptation to new task (few-shot learning)
        
        Args:
            support_data: Data for adaptation
            support_labels: Labels for adaptation
            num_adaptation_steps: Number steps adaptation
            
        Returns:
            Adapted model
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.num_inner_steps
        
        # Create copy model for adaptation
        adapted_model = type(self.model)(
            **self.model.config.__dict__ if hasattr(self.model, 'config') else {}
        ).to(self.config.device)
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Optimizer for adaptation
        adaptation_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )
        
        # Adaptation
        adapted_model.train()
        for step in range(num_adaptation_steps):
            adaptation_optimizer.zero_grad()
            
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            
            loss.backward()
            adaptation_optimizer.step()
            
            if step % 5 == 0:
                self.logger.debug(f"Adaptation step {step}, loss: {loss.item():.4f}")
        
        return adapted_model
    
    def _compute_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.1
    ) -> float:
        """
        Computes accuracy for regression (percent predictions in within threshold)
        
        Args:
            predictions: Predictions model
            labels: True labels
            threshold: Threshold for counting predictions correct
            
        Returns:
            Accuracy value
        """
        with torch.no_grad():
            errors = torch.abs(predictions - labels)
            correct = (errors < threshold).float()
            return correct.mean().item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saving checkpoint model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_meta_loss': self.best_meta_loss
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loading checkpoint model"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_meta_loss = checkpoint['best_meta_loss']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")


class MAMLTrainer:
    """
    Trainer class for MAML with enterprise patterns
    
    Features:
    - Automated checkpoint management
    - Comprehensive metrics tracking
    - Early stopping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        maml: MAML,
        train_loader: Any,
        val_loader: Any,
        config: MAMLConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.maml = maml
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Scheduler for learning rate
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            maml.meta_optimizer,
            mode='min',
            factor=0.8,
            patience=10,
            verbose=True
        )
        
        self.metrics_history = []
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Main loop training MAML
        
        Args:
            num_epochs: Number epochs
            save_dir: Directory for saving checkpoint'
            early_stopping_patience: Patience for early stopping
            
        Returns:
            History metrics training
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self.maml.meta_validate(self.val_loader)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_query_loss'])
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_metrics(epoch, epoch_metrics)
            
            # Checkpoint saving
            current_val_loss = val_metrics['val_query_loss']
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                self.maml.best_meta_loss = best_val_loss
                self.maml.save_checkpoint(f"{save_dir}/best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        return self._compile_metrics_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Training one epoch"""
        epoch_metrics = []
        
        for batch in tqdm(self.train_loader, desc="Training"):
            metrics = self.maml.meta_train_step(batch)
            epoch_metrics.append(metrics)
        
        # Aggregate metrics epoch
        return {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logging metrics"""
        self.logger.info(f"Epoch {epoch}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _compile_metrics_history(self) -> Dict[str, List[float]]:
        """Compilation history metrics"""
        compiled = {}
        for key in self.metrics_history[0].keys():
            compiled[key] = [m[key] for m in self.metrics_history]
        return compiled