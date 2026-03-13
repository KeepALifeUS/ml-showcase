"""
Elastic Weight Consolidation (EWC) for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade implementation EWC strategies for prevention
catastrophic forgetting in crypto trading with integration.
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from copy import deepcopy
import logging
from datetime import datetime

from ..core.continual_learner import ContinualLearner, LearnerConfig, TaskMetadata
from ..core.memory_buffer import MemorySample, BaseMemoryBuffer, MemoryBufferFactory, SamplingStrategy


class EWCLearner(ContinualLearner):
    """
    Elastic Weight Consolidation Learner for crypto trading
    
    EWC catastrophic forgetting addition
    regularization term that penalizes changes to important weights.
    
    enterprise Features:
    - Adaptive importance computation
    - Memory-efficient Fisher matrix
    - Market regime aware regularization
    - Performance degradation detection
    """
    
    def __init__(self, model: nn.Module, config: LearnerConfig):
        super().__init__(model, config)
        
        # EWC specific components
        self.fisher_matrices: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params: Dict[int, Dict[str, torch.Tensor]] = {}
        self.ewc_lambda = 0.4 # Strength regularization EWC
        self.fisher_estimation_samples = 500 # Number samples for evaluation Fisher matrix
        
        # enterprise settings
        self.adaptive_lambda = True  # Adaptive regularization
        self.market_regime_scaling = True # Scale by market regime
        self.performance_tracking = True # Track degradation performance
        
        # Memory for samples
        self.memory_buffer = MemoryBufferFactory.create_crypto_trading_buffer(
            max_size=config.memory_budget,
            strategy=SamplingStrategy.RESERVOIR
        )
        
        # Optimizer settings
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_lambda
        )
        
        # Loss function
        self.criterion = nn.MSELoss() # For regression (price/return)
        
        # Monitor 
        self.ewc_losses: List[float] = []
        self.fisher_computation_times: List[float] = []
        self.regularization_strengths: Dict[int, float] = {}
    
    def learn_task(self, task_data: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Learning new tasks with EWC regularization
        
        Args:
            task_data: Data for training (features, targets, etc.)
            task_metadata: Metadata tasks
            
        Returns:
            Dict with metrics performance
        """
        self.logger.info(f"Starting EWC learning for task {task_metadata.task_id}: {task_metadata.name}")
        
        # Preparation data
        features = torch.tensor(task_data["features"], dtype=torch.float32)
        targets = torch.tensor(task_data["targets"], dtype=torch.float32)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        # Create DataLoader
        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Adaptive regularization on basis market regime
        if self.market_regime_scaling:
            self.ewc_lambda = self._compute_regime_adaptive_lambda(task_metadata.market_regime)
            self.regularization_strengths[task_metadata.task_id] = self.ewc_lambda
        
        # Training model
        metrics = self._train_with_ewc(dataloader, task_metadata)
        
        # Computation Fisher Information Matrix for new tasks
        self._compute_fisher_matrix(dataloader, task_metadata.task_id)
        
        # Save optimal parameters
        self._save_optimal_params(task_metadata.task_id)
        
        # Update history tasks
        self.task_history.append(task_metadata)
        self.current_task = task_metadata.task_id
        self.performance_history[task_metadata.task_id] = metrics
        
        # Add samples in memory for replay
        self._add_samples_to_memory(features, targets, task_metadata)
        
        # Checkpoint saving
        if self.config.enable_checkpointing:
            checkpoint_name = f"ewc_task_{task_metadata.task_id}_{task_metadata.market_regime}"
            self.save_checkpoint(checkpoint_name)
        
        self.logger.info(f"Completed EWC learning for task {task_metadata.task_id}")
        return metrics
    
    def _compute_regime_adaptive_lambda(self, market_regime: str) -> float:
        """
        Compute adaptive regularization strength based on market regime
        
        Adaptive regularization for various market conditions
        
        Args:
            market_regime: Market regime (bull, bear, sideways, volatile)
            
        Returns:
            Adaptive coefficient lambda
        """
        base_lambda = 0.4
        
        # Scale by regime
        regime_scaling = {
            "volatile": 0.8,     # High regularization in volatility
            "bear": 0.6, # Average- in bear market
            "bull": 0.3, # Low in bull market (more plasticity)
            "sideways": 0.5 # Average in
        }
        
        scaling_factor = regime_scaling.get(market_regime, 0.4)
        adaptive_lambda = base_lambda * scaling_factor
        
        self.logger.info(f"Adaptive EWC lambda for {market_regime}: {adaptive_lambda}")
        return adaptive_lambda
    
    def _train_with_ewc(self, dataloader: DataLoader, task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Training model with EWC regularization
        
        Args:
            dataloader: DataLoader with training data
            task_metadata: Metadata tasks
            
        Returns:
            Metrics training
        """
        self.model.train()
        total_loss = 0.0
        task_loss = 0.0
        ewc_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Training cycle
        for epoch in range(10): # Possible make configurable
            epoch_loss = 0.0
            
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.config.device)
                batch_targets = batch_targets.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_features)
                
                # Task loss
                current_task_loss = self.criterion(predictions, batch_targets)
                
                # EWC regularization loss
                current_ewc_loss = self._compute_ewc_loss()
                
                # Total loss
                total_batch_loss = current_task_loss + current_ewc_loss
                
                # Backward pass
                total_batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_batch_loss.item()
                task_loss += current_task_loss.item()
                ewc_loss += current_ewc_loss.item()
                num_batches += 1
                
                # Accuracy calculation (for classification directions prices)
                if predictions.dim() == batch_targets.dim():
                    pred_direction = (predictions > 0).float()
                    target_direction = (batch_targets > 0).float()
                    correct_predictions += (pred_direction == target_direction).sum().item()
                    total_predictions += batch_targets.numel()
            
            self.logger.debug(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Computation final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_task_loss = task_loss / num_batches if num_batches > 0 else 0.0
        avg_ewc_loss = ewc_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            "total_loss": avg_loss,
            "task_loss": avg_task_loss,
            "ewc_loss": avg_ewc_loss,
            "accuracy": accuracy,
            "ewc_lambda": self.ewc_lambda,
            "num_batches": num_batches,
            "predictions_made": total_predictions
        }
        
        # Save EWC loss for monitoring
        self.ewc_losses.append(avg_ewc_loss)
        
        return metrics
    
    def _compute_ewc_loss(self) -> torch.Tensor:
        """
        Computation EWC regularization loss
        
        Returns:
            EWC regularization term
        """
        ewc_loss = torch.tensor(0.0, device=self.config.device)
        
        # Iterate through all previous tasks
        for task_id in self.fisher_matrices:
            fisher_matrix = self.fisher_matrices[task_id]
            optimal_params = self.optimal_params[task_id]
            
            for name, param in self.model.named_parameters():
                if name in fisher_matrix and name in optimal_params:
                    # EWC penalty: F_i * (theta - theta*_i)^2
                    fisher = fisher_matrix[name].to(self.config.device)
                    optimal = optimal_params[name].to(self.config.device)
                    
                    penalty = fisher * (param - optimal) ** 2
                    ewc_loss += self.ewc_lambda * penalty.sum()
        
        return ewc_loss
    
    def _compute_fisher_matrix(self, dataloader: DataLoader, task_id: int) -> None:
        """
        Computation Fisher Information Matrix for tasks
        
        Args:
            dataloader: DataLoader with data tasks
            task_id: ID tasks
        """
        start_time = datetime.now()
        self.logger.info(f"Computing Fisher matrix for task {task_id}")
        
        self.model.eval()
        fisher_matrix = {}
        
        # Initialize Fisher matrix
        for name, param in self.model.named_parameters():
            fisher_matrix[name] = torch.zeros_like(param)
        
        # Sampling for evaluation Fisher matrix
        num_samples = min(self.fisher_estimation_samples, len(dataloader.dataset))
        sample_count = 0
        
        for batch_features, batch_targets in dataloader:
            if sample_count >= num_samples:
                break
            
            batch_features = batch_features.to(self.config.device)
            batch_targets = batch_targets.to(self.config.device)
            
            for i in range(min(batch_features.size(0), num_samples - sample_count)):
                sample_input = batch_features[i:i+1]
                sample_target = batch_targets[i:i+1]
                
                self.model.zero_grad()
                output = self.model(sample_input)
                loss = self.criterion(output, sample_target)
                
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_matrix[name] += param.grad.detach() ** 2
                
                sample_count += 1
        
        # Normalize Fisher matrix
        for name in fisher_matrix:
            fisher_matrix[name] /= sample_count
        
        # Save Fisher matrix
        self.fisher_matrices[task_id] = {
            name: matrix.clone().detach() for name, matrix in fisher_matrix.items()
        }
        
        # Monitor time computation
        computation_time = (datetime.now() - start_time).total_seconds()
        self.fisher_computation_times.append(computation_time)
        
        self.logger.info(f"Fisher matrix computed for task {task_id} in {computation_time:.2f}s")
    
    def _save_optimal_params(self, task_id: int) -> None:
        """
        Save optimal parameters for tasks
        
        Args:
            task_id: ID tasks
        """
        self.optimal_params[task_id] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }
        
        self.logger.debug(f"Optimal parameters saved for task {task_id}")
    
    def _add_samples_to_memory(self, features: torch.Tensor, targets: torch.Tensor, task_metadata: TaskMetadata) -> None:
        """
        Add samples in buffer memory for replay
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
        """
        # Create MemorySample objects
        samples = []
        num_samples = min(features.size(0), 50) # Limitation for efficiency
        
        indices = torch.randperm(features.size(0))[:num_samples]
        
        for i in indices:
            sample = MemorySample(
                features=features[i],
                target=targets[i],
                task_id=task_metadata.task_id,
                timestamp=task_metadata.start_time,
                market_regime=task_metadata.market_regime,
                asset=task_metadata.assets[0] if task_metadata.assets else "UNKNOWN",
                timeframe=task_metadata.timeframe,
                uncertainty_score=0.5, # Possible improve
                gradient_magnitude=0.0,
                loss_value=0.0,
                diversity_score=0.5,
                quality_score=0.8
            )
            samples.append(sample)
        
        self.memory_buffer.add_samples(samples)
        self.logger.debug(f"Added {len(samples)} samples to memory buffer")
    
    def evaluate_task(self, task_id: int, test_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate performance on specific task
        
        Args:
            task_id: ID tasks for evaluation
            test_data: Test data
            
        Returns:
            Dict with metrics evaluation
        """
        self.model.eval()
        
        features = torch.tensor(test_data["features"], dtype=torch.float32).to(self.config.device)
        targets = torch.tensor(test_data["targets"], dtype=torch.float32).to(self.config.device)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            # Accuracy for directions movements prices
            pred_direction = (predictions > 0).float()
            target_direction = (targets > 0).float()
            accuracy = (pred_direction == target_direction).float().mean().item()
            
            # MAE for regression
            mae = torch.abs(predictions - targets).mean().item()
            
            # RMSE
            rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        metrics = {
            "task_id": task_id,
            "test_loss": loss.item(),
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "num_samples": features.size(0)
        }
        
        self.logger.info(f"Task {task_id} evaluation - Accuracy: {accuracy:.3f}, Loss: {loss.item():.4f}")
        return metrics
    
    def get_ewc_statistics(self) -> Dict[str, Any]:
        """
        Get statistics EWC training
        
        Returns:
            Dict with statistics EWC
        """
        stats = {
            "strategy": "ewc",
            "ewc_lambda": self.ewc_lambda,
            "num_fisher_matrices": len(self.fisher_matrices),
            "num_optimal_params": len(self.optimal_params),
            "adaptive_lambda_enabled": self.adaptive_lambda,
            "market_regime_scaling_enabled": self.market_regime_scaling,
            "regularization_strengths": self.regularization_strengths
        }
        
        if self.ewc_losses:
            stats.update({
                "avg_ewc_loss": np.mean(self.ewc_losses),
                "min_ewc_loss": np.min(self.ewc_losses),
                "max_ewc_loss": np.max(self.ewc_losses),
                "ewc_loss_trend": "increasing" if self.ewc_losses[-1] > self.ewc_losses[0] else "decreasing"
            })
        
        if self.fisher_computation_times:
            stats.update({
                "avg_fisher_computation_time": np.mean(self.fisher_computation_times),
                "total_fisher_computation_time": sum(self.fisher_computation_times)
            })
        
        # Memory buffer statistics
        if self.memory_buffer:
            memory_stats = self.memory_buffer.get_memory_statistics()
            stats["memory_buffer"] = memory_stats
        
        return stats
    
    def clear_task_data(self, task_id: int) -> None:
        """
        Cleanup data specific tasks for memory
        
        Args:
            task_id: ID tasks for cleanup
        """
        if task_id in self.fisher_matrices:
            del self.fisher_matrices[task_id]
        
        if task_id in self.optimal_params:
            del self.optimal_params[task_id]
        
        if task_id in self.regularization_strengths:
            del self.regularization_strengths[task_id]
        
        self.logger.info(f"Cleared data for task {task_id}")
    
    def __repr__(self) -> str:
        return (
            f"EWCLearner("
            f"lambda={self.ewc_lambda}, "
            f"tasks={len(self.fisher_matrices)}, "
            f"memory_size={len(self.memory_buffer) if self.memory_buffer else 0})"
        )