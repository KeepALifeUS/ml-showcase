"""
Reptile Algorithm Implementation
First-Order Meta-Learning for Crypto Trading

Implementation algorithm Reptile - simplified version MAML without second derivatives.
Especially effective for fast adaptation to new cryptocurrency assets.
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
class ReptileConfig:
    """Configuration for Reptile algorithm"""
    
    # Main parameters
    inner_lr: float = 0.01  # Speed training on task
    meta_lr: float = 0.001  # Speed meta-training
    num_inner_steps: int = 5  # Number steps on task
    
    # Parameters tasks
    num_support: int = 5  # Size support set
    num_query: int = 15  # Size query set
    
    # Optimization
    meta_batch_size: int = 32  # Number tasks in meta-batch
    gradient_clip: Optional[float] = 1.0  # Trimming gradients
    weight_decay: float = 0.0001  # L2 regularization
    
    # Monitoring
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Reptile:
    """
    Reptile Meta-Learning Algorithm
    
    Simplified Meta-Learning System
    - First-order optimization only
    - Memory efficient
    - Fast convergence
    - Production scalable
    
    Reptile uses simple rule updates:
    θ' = θ + ε * (φ - θ)
    where φ - parameters after adaptation on task
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ReptileConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialization Reptile
        
        Args:
            model: Base model for meta-training
            config: Configuration Reptile
            logger: Logger for monitoring
        """
        self.model = model.to(config.device)
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Meta-optimizer not needed, use direct update
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # State
        self.global_step = 0
        self.best_meta_loss = float('inf')
        
        self.logger.info(f"Reptile initialized with config: {config}")
    
    def inner_adaptation(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[OrderedDict, List[float]]:
        """
        Adaptation model to specific task
        
        Args:
            support_data: Data for training on task
            support_labels: Labels for training
            
        Returns:
            Tuple from adapted parameters and losses
        """
        # Create copy model for adaptation
        adapted_model = copy.deepcopy(self.model)
        
        # Optimizer for adaptation
        inner_optimizer = optim.SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr
        )
        
        adaptation_losses = []
        
        for step in range(self.config.num_inner_steps):
            inner_optimizer.zero_grad()
            
            # Forward pass
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            adaptation_losses.append(loss.item())
            
            # Backward pass
            loss.backward()
            inner_optimizer.step()
        
        # Return adapted parameters
        adapted_params = OrderedDict(adapted_model.named_parameters())
        return adapted_params, adaptation_losses
    
    def meta_train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        One step meta-training Reptile
        
        Args:
            task_batch: Batch tasks for meta-training
            
        Returns:
            Dictionary with metrics
        """
        # Save original parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        all_adaptation_losses = []
        query_losses = []
        query_accuracies = []
        adapted_params_list = []
        
        # Process each task in batch
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Adaptation to task
            adapted_params, adaptation_losses = self.inner_adaptation(
                support_data, support_labels
            )
            adapted_params_list.append(adapted_params)
            all_adaptation_losses.extend(adaptation_losses)
            
            # Estimation on query set
            with torch.no_grad():
                # Temporarily apply adapted parameters
                self._apply_params(adapted_params)
                
                query_predictions = self.model(query_data)
                query_loss = nn.functional.mse_loss(
                    query_predictions, query_labels
                ).item()
                query_losses.append(query_loss)
                
                query_accuracy = self._compute_accuracy(
                    query_predictions, query_labels
                )
                query_accuracies.append(query_accuracy)
                
                # Restore original parameters
                self._apply_params(original_params)
        
        # Reptile meta-update
        self._reptile_meta_update(adapted_params_list, original_params)
        
        # Metrics
        metrics = {
            'adaptation_loss': np.mean(all_adaptation_losses),
            'query_loss': np.mean(query_losses),
            'query_accuracy': np.mean(query_accuracies),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.model.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def _reptile_meta_update(
        self,
        adapted_params_list: List[OrderedDict],
        original_params: OrderedDict
    ) -> None:
        """
        Main update Reptile
        
        Args:
            adapted_params_list: List adapted parameters
            original_params: Original parameters model
        """
        # Compute average direction updates
        meta_gradients = OrderedDict()
        
        for name in original_params.keys():
            # Average difference between adapted and original parameters
            param_diffs = []
            for adapted_params in adapted_params_list:
                diff = adapted_params[name].data - original_params[name]
                param_diffs.append(diff)
            
            # Average by all tasks
            meta_gradients[name] = torch.stack(param_diffs).mean(dim=0)
        
        # Apply meta-update
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in meta_gradients:
                    # Reptile update: θ = θ + α * (φ_avg - θ)
                    param.data.add_(
                        meta_gradients[name], alpha=self.config.meta_lr
                    )
                    
                    # Weight decay
                    if self.config.weight_decay > 0:
                        param.data.mul_(1 - self.config.weight_decay)
        
        # Gradient clipping on meta-gradients
        if self.config.gradient_clip:
            total_norm = 0
            for grad in meta_gradients.values():
                total_norm += grad.norm().item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > self.config.gradient_clip:
                clip_coef = self.config.gradient_clip / (total_norm + 1e-6)
                for name, param in self.model.named_parameters():
                    if name in meta_gradients:
                        param.data.add_(
                            meta_gradients[name] * (clip_coef - 1),
                            alpha=self.config.meta_lr
                        )
    
    def _apply_params(self, params: OrderedDict) -> None:
        """Applies parameters to model"""
        for name, param in self.model.named_parameters():
            if name in params:
                param.data = params[name].data
    
    def meta_validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Validation meta-model
        
        Args:
            validation_tasks: Tasks for validation
            
        Returns:
            Dictionary with metrics validation
        """
        # Save current parameters
        original_params = OrderedDict()
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        all_metrics = []
        
        try:
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Adaptation to task
                adapted_params, adaptation_losses = self.inner_adaptation(
                    support_data, support_labels
                )
                
                # Apply adapted parameters
                self._apply_params(adapted_params)
                
                # Estimation on query set
                with torch.no_grad():
                    query_predictions = self.model(query_data)
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
                
                # Restore original parameters for next tasks
                self._apply_params(original_params)
        
        finally:
            # Guaranteed restore parameters
            self._apply_params(original_params)
        
        # Aggregate metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[f'val_{key}'] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def few_shot_adapt(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        num_adaptation_steps: Optional[int] = None,
        return_copy: bool = True
    ) -> nn.Module:
        """
        Fast adaptation to new task
        
        Args:
            support_data: Data for adaptation
            support_labels: Labels for adaptation
            num_adaptation_steps: Number steps adaptation
            return_copy: Return copy or change original model
            
        Returns:
            Adapted model
        """
        if num_adaptation_steps is None:
            num_adaptation_steps = self.config.num_inner_steps
        
        if return_copy:
            adapted_model = copy.deepcopy(self.model)
        else:
            adapted_model = self.model
        
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
        """Computes accuracy for regression"""
        with torch.no_grad():
            errors = torch.abs(predictions - labels)
            correct = (errors < threshold).float()
            return correct.mean().item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saving checkpoint model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_meta_loss': self.best_meta_loss
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Reptile checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loading checkpoint model"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_meta_loss = checkpoint['best_meta_loss']
        
        self.logger.info(f"Reptile checkpoint loaded from {filepath}")


class ReptileTrainer:
    """
    Trainer for Reptile with enterprise patterns
    
    Features:
    - Efficient memory usage
    - Fast training cycles
    - Robust checkpoint management
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        reptile: Reptile,
        train_loader: Any,
        val_loader: Any,
        config: ReptileConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.reptile = reptile
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics_history = []
        self.best_val_accuracy = 0.0
    
    def train(
        self,
        num_iterations: int,
        save_dir: str = "./checkpoints",
        validation_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        Main loop training Reptile
        
        Args:
            num_iterations: Number iterations training
            save_dir: Directory for checkpoint'
            validation_interval: Interval validation
            
        Returns:
            History metrics training
        """
        iteration = 0
        
        for epoch in range(num_iterations // len(self.train_loader) + 1):
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                if iteration >= num_iterations:
                    break
                
                # Training step
                train_metrics = self.reptile.meta_train_step(batch)
                
                # Validation
                if iteration % validation_interval == 0:
                    val_metrics = self.reptile.meta_validate(self.val_loader)
                    
                    # Combine metrics
                    combined_metrics = {**train_metrics, **val_metrics}
                    self.metrics_history.append(combined_metrics)
                    
                    # Logging
                    if iteration % self.config.log_interval == 0:
                        self._log_metrics(iteration, combined_metrics)
                    
                    # Checkpoint saving
                    current_val_accuracy = val_metrics.get('val_query_accuracy', 0)
                    if current_val_accuracy > self.best_val_accuracy:
                        self.best_val_accuracy = current_val_accuracy
                        self.reptile.save_checkpoint(f"{save_dir}/best_reptile_model.pt")
                
                iteration += 1
        
        return self._compile_metrics_history()
    
    def _log_metrics(self, iteration: int, metrics: Dict[str, float]) -> None:
        """Logging metrics"""
        self.logger.info(f"Iteration {iteration}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def _compile_metrics_history(self) -> Dict[str, List[float]]:
        """Compilation history metrics"""
        if not self.metrics_history:
            return {}
        
        compiled = {}
        for key in self.metrics_history[0].keys():
            compiled[key] = [m[key] for m in self.metrics_history]
        return compiled