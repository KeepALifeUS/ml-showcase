"""
PackNet for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade implementation PackNet strategies for prevention
catastrophic forgetting through pruning
and parameters with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
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


class PackNetMask:
    """
    Mask for PackNet weights available for each tasks
    
    Ensures structured parameter partitioning between tasks
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self.free_capacity: Dict[str, torch.Tensor] = {}
        self.total_capacity: Dict[str, int] = {}
        
        # Initialize masks free capacity
        self._initialize_capacity()
    
    def _initialize_capacity(self) -> None:
        """Initialize general capacity for each layers"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.free_capacity[name] = torch.ones_like(param, dtype=torch.bool)
                self.total_capacity[name] = param.numel()
    
    def allocate_capacity_for_task(
        self, 
        task_id: int, 
        allocation_ratio: float = 0.2,
        importance_scores: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extraction capacity for new tasks
        
        Args:
            task_id: ID tasks
            allocation_ratio: Fraction free capacity for extraction
            importance_scores: Scores parameters
            
        Returns:
            Masks for new tasks
        """
        if task_id in self.task_masks:
            return self.task_masks[task_id]
        
        new_masks = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad or name not in self.free_capacity:
                continue
            
            free_mask = self.free_capacity[name]
            free_indices = torch.nonzero(free_mask, as_tuple=True)
            
            if len(free_indices[0]) == 0:
                # No free capacity - create empty mask
                new_masks[name] = torch.zeros_like(param, dtype=torch.bool)
                continue
            
            # Determine number parameters for extraction
            available_count = len(free_indices[0])
            target_allocation = int(available_count * allocation_ratio)
            
            if target_allocation == 0:
                new_masks[name] = torch.zeros_like(param, dtype=torch.bool)
                continue
            
            # Select parameters for extraction
            if importance_scores and name in importance_scores:
                # On basis
                selected_indices = self._select_by_importance(
                    free_indices, importance_scores[name], target_allocation
                )
            else:
                # Random selection
                perm = torch.randperm(available_count)[:target_allocation]
                selected_indices = tuple(idx[perm] for idx in free_indices)
            
            # Create masks for tasks
            task_mask = torch.zeros_like(param, dtype=torch.bool)
            task_mask[selected_indices] = True
            new_masks[name] = task_mask
            
            # Update free capacity
            self.free_capacity[name][selected_indices] = False
        
        self.task_masks[task_id] = new_masks
        return new_masks
    
    def _select_by_importance(
        self, 
        free_indices: Tuple[torch.Tensor, ...], 
        importance: torch.Tensor, 
        count: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Select parameters on basis
        
        Args:
            free_indices: Indices free parameters
            importance: Scores
            count: Number parameters for selection
            
        Returns:
            Indices parameters
        """
        # Get for free parameters
        free_importance = importance[free_indices]
        
        # Select top parameters
        _, top_indices = torch.topk(free_importance, min(count, len(free_importance)))
        
        # Convert back in original indices
        selected_indices = tuple(idx[top_indices] for idx in free_indices)
        
        return selected_indices
    
    def get_task_mask(self, task_id: int) -> Optional[Dict[str, torch.Tensor]]:
        """Get masks for tasks"""
        return self.task_masks.get(task_id)
    
    def apply_mask(self, task_id: int) -> None:
        """
        Apply masks to gradients model
        
        Args:
            task_id: ID tasks
        """
        if task_id not in self.task_masks:
            return
        
        masks = self.task_masks[task_id]
        
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in masks:
                param.grad.data *= masks[name].float()
    
    def freeze_task_parameters(self, task_id: int) -> None:
        """
        Freezing parameters tasks after training
        
        Args:
            task_id: ID tasks
        """
        if task_id not in self.task_masks:
            return
        
        # PackNet parameters after training tasks
        # This through gradients in apply_mask
        pass
    
    def get_capacity_statistics(self) -> Dict[str, Any]:
        """
        Statistics use capacity
        
        Returns:
            Statistics capacity
        """
        stats = {
            "total_tasks": len(self.task_masks),
            "layer_statistics": {}
        }
        
        for name in self.total_capacity:
            total = self.total_capacity[name]
            free_count = self.free_capacity[name].sum().item()
            used_count = total - free_count
            
            stats["layer_statistics"][name] = {
                "total_parameters": total,
                "used_parameters": used_count,
                "free_parameters": free_count,
                "utilization": used_count / total if total > 0 else 0.0
            }
        
        # Total statistics
        total_params = sum(self.total_capacity.values())
        total_free = sum(mask.sum().item() for mask in self.free_capacity.values())
        total_used = total_params - total_free
        
        stats.update({
            "total_parameters": total_params,
            "total_used": total_used,
            "total_free": total_free,
            "overall_utilization": total_used / total_params if total_params > 0 else 0.0
        })
        
        return stats


class PackNetLearner(ContinualLearner):
    """
    PackNet Learner for crypto trading
    
    PackNet prevents catastrophic forgetting through structured
    separation parameters model between tasks through pruning.
    
    enterprise Features:
    - Importance-guided parameter allocation
    - Dynamic capacity management
    - Market regime aware pruning
    - Memory-efficient parameter sharing
    - Adaptive compression ratios
    """
    
    def __init__(self, model: nn.Module, config: LearnerConfig):
        super().__init__(model, config)
        
        # PackNet specific components
        self.packnet_mask = PackNetMask(model)
        self.task_allocation_ratios: Dict[int, float] = {}
        self.current_task_mask: Optional[Dict[str, torch.Tensor]] = None
        
        # enterprise settings
        self.importance_driven_allocation = True # Extraction on basis
        self.adaptive_compression = True # Adaptive compression
        self.market_regime_pruning = True  # Pruning with consideration market regime
        self.dynamic_capacity_management = True #
        
        # Settings pruning
        self.base_allocation_ratio = 0.2 # Base fraction for extraction
        self.min_allocation_ratio = 0.1 # Minimal fraction
        self.max_allocation_ratio = 0.4 # Maximum fraction
        self.importance_threshold = 0.01 # Threshold for pruning
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_lambda
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Memory buffer (small, so as in architecture)
        self.memory_buffer = MemoryBufferFactory.create_crypto_trading_buffer(
            max_size=config.memory_budget // 3,
            strategy=SamplingStrategy.RESERVOIR
        )
        
        # Monitor 
        self.pruning_statistics: List[Dict[str, Any]] = []
        self.capacity_history: List[Dict[str, Any]] = []
        self.importance_scores_history: Dict[int, Dict[str, float]] = {}
        self.compression_ratios: Dict[int, float] = {}
    
    def learn_task(self, task_data: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Learning new tasks with PackNet
        
        Args:
            task_data: Data for training (features, targets, etc.)
            task_metadata: Metadata tasks
            
        Returns:
            Dict with metrics performance
        """
        self.logger.info(f"Starting PackNet learning for task {task_metadata.task_id}: {task_metadata.name}")
        
        # Preparation data
        features = torch.tensor(task_data["features"], dtype=torch.float32)
        targets = torch.tensor(task_data["targets"], dtype=torch.float32)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        # capacity on basis market regime
        allocation_ratio = self._compute_adaptive_allocation_ratio(task_metadata.market_regime)
        self.task_allocation_ratios[task_metadata.task_id] = allocation_ratio
        
        # Computation parameters before
        importance_scores = None
        if self.importance_driven_allocation:
            importance_scores = self._compute_parameter_importance(features, targets)
        
        # Extraction capacity for new tasks
        self.current_task_mask = self.packnet_mask.allocate_capacity_for_task(
            task_metadata.task_id,
            allocation_ratio,
            importance_scores
        )
        
        # Create DataLoader
        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training with PackNet masks
        metrics = self._train_with_packnet(dataloader, task_metadata)
        
        # Freezing parameters tasks
        self.packnet_mask.freeze_task_parameters(task_metadata.task_id)
        
        # Pruning model after training
        if self.market_regime_pruning:
            pruning_stats = self._perform_regime_aware_pruning(task_metadata)
            metrics.update(pruning_stats)
        
        # Update history tasks
        self.task_history.append(task_metadata)
        self.current_task = task_metadata.task_id
        self.performance_history[task_metadata.task_id] = metrics
        
        # Add samples in memory
        self._add_samples_to_memory(features, targets, task_metadata)
        
        # Save statistics
        capacity_stats = self.packnet_mask.get_capacity_statistics()
        self.capacity_history.append({
            "task_id": task_metadata.task_id,
            "timestamp": datetime.now().isoformat(),
            **capacity_stats
        })
        
        # Checkpoint saving
        if self.config.enable_checkpointing:
            checkpoint_name = f"packnet_task_{task_metadata.task_id}_{task_metadata.market_regime}"
            self.save_checkpoint(checkpoint_name)
        
        self.logger.info(f"Completed PackNet learning for task {task_metadata.task_id}")
        return metrics
    
    def _compute_adaptive_allocation_ratio(self, market_regime: str) -> float:
        """
        Compute adaptive allocation coefficient based on market regime
        
        Market regime aware resource allocation
        
        Args:
            market_regime: Market regime
            
        Returns:
            Coefficient extraction capacity
        """
        if not self.adaptive_compression:
            return self.base_allocation_ratio
        
        # Adaptation under market conditions
        regime_ratios = {
            "volatile": 0.35, # More parameters for complex patterns
            "bear": 0.25, # Average-high for bear market
            "bull": 0.15, # Fewer for stable market
            "sideways": 0.20 # Average for movements
        }
        
        # current loading capacity
        capacity_stats = self.packnet_mask.get_capacity_statistics()
        current_utilization = capacity_stats.get("overall_utilization", 0.0)
        
        base_ratio = regime_ratios.get(market_regime, self.base_allocation_ratio)
        
        # Adaptation on basis loading
        if current_utilization > 0.8:  # High loading
            adjusted_ratio = base_ratio * 0.7
        elif current_utilization < 0.3:  # Low loading
            adjusted_ratio = base_ratio * 1.3
        else:
            adjusted_ratio = base_ratio
        
        # Limitation in
        final_ratio = np.clip(adjusted_ratio, self.min_allocation_ratio, self.max_allocation_ratio)
        
        self.compression_ratios[len(self.task_history)] = final_ratio
        self.logger.info(f"Adaptive allocation ratio for {market_regime}: {final_ratio:.3f} (utilization: {current_utilization:.3f})")
        
        return final_ratio
    
    def _compute_parameter_importance(
        self, 
        features: torch.Tensor, 
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Computation parameters on basis gradients
        
        Args:
            features: Input features
            targets: Target values
            
        Returns:
            Dictionary with for each parameter
        """
        self.model.train()
        importance_scores = {}
        
        # Initialize
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance_scores[name] = torch.zeros_like(param)
        
        # for evaluation
        num_importance_samples = min(len(features), 100)
        indices = torch.randperm(len(features))[:num_importance_samples]
        
        for idx in indices:
            sample_features = features[idx:idx+1].to(self.config.device)
            sample_targets = targets[idx:idx+1].to(self.config.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sample_features)
            loss = self.criterion(predictions, sample_targets)
            
            # Backward pass
            loss.backward()
            
            # gradients as measure
            for name, param in self.model.named_parameters():
                if param.grad is not None and name in importance_scores:
                    importance_scores[name] += param.grad.detach() ** 2
        
        # Normalization
        for name in importance_scores:
            if num_importance_samples > 0:
                importance_scores[name] /= num_importance_samples
                
                # Additional normalization for stability
                max_importance = importance_scores[name].max()
                if max_importance > 0:
                    importance_scores[name] /= max_importance
        
        self.logger.debug(f"Computed parameter importance for {len(importance_scores)} layers")
        return importance_scores
    
    def _train_with_packnet(self, dataloader: DataLoader, task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Training model with PackNet masks
        
        Args:
            dataloader: DataLoader with training data
            task_metadata: Metadata tasks
            
        Returns:
            Metrics training
        """
        if self.current_task_mask is None:
            raise ValueError("No task mask allocated for training")
        
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Statistics use parameters
        active_params = 0
        total_params = 0
        
        for name, mask in self.current_task_mask.items():
            active_params += mask.sum().item()
            total_params += mask.numel()
        
        utilization = active_params / total_params if total_params > 0 else 0.0
        
        # Training cycle
        num_epochs = 12 # Possible configure
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.config.device)
                batch_targets = batch_targets.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_features)
                loss = self.criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                
                # Apply PackNet masks to gradients
                self.packnet_mask.apply_mask(task_metadata.task_id)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate metrics
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
                
                # Accuracy calculation
                if predictions.dim() == batch_targets.dim():
                    pred_direction = (predictions > 0).float()
                    target_direction = (batch_targets > 0).float()
                    correct_predictions += (pred_direction == target_direction).sum().item()
                    total_predictions += batch_targets.numel()
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            self.logger.debug(f"Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Final metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "parameter_utilization": utilization,
            "active_parameters": active_params,
            "total_parameters": total_params,
            "allocation_ratio": self.task_allocation_ratios.get(task_metadata.task_id, 0.0),
            "num_batches": num_batches,
            "predictions_made": total_predictions
        }
        
        return metrics
    
    def _perform_regime_aware_pruning(self, task_metadata: TaskMetadata) -> Dict[str, Any]:
        """
        Pruning with consideration market regime after training tasks
        
        Market regime aware model compression
        
        Args:
            task_metadata: Metadata tasks
            
        Returns:
            Statistics pruning
        """
        if not self.current_task_mask:
            return {}
        
        pruning_stats = {
            "pruned_parameters": 0,
            "pruning_ratio": 0.0,
            "market_regime": task_metadata.market_regime
        }
        
        # Determine pruning on basis market regime
        regime_pruning_ratios = {
            "volatile": 0.1, # pruning in volatility
            "bear": 0.2,         # Average pruning in bear market
            "bull": 0.3, # pruning in market
            "sideways": 0.25 # Average- pruning
        }
        
        target_pruning_ratio = regime_pruning_ratios.get(
            task_metadata.market_regime, 0.2
        )
        
        # Pruning parameters with low
        total_pruned = 0
        total_params = 0
        
        for name, mask in self.current_task_mask.items():
            if name not in self.model.state_dict():
                continue
            
            param = self.model.state_dict()[name]
            active_mask = mask
            
            # Computation active parameters
            param_magnitudes = torch.abs(param) * active_mask.float()
            active_magnitudes = param_magnitudes[active_mask]
            
            if len(active_magnitudes) == 0:
                continue
            
            # Determine threshold for pruning
            num_to_prune = int(len(active_magnitudes) * target_pruning_ratio)
            if num_to_prune > 0:
                threshold_value = torch.kthvalue(active_magnitudes, num_to_prune).values
                
                # Create masks for pruning
                prune_mask = (param_magnitudes < threshold_value) & active_mask
                
                # Update masks tasks ( pruned parameters)
                self.current_task_mask[name] = active_mask & (~prune_mask)
                
                pruned_count = prune_mask.sum().item()
                total_pruned += pruned_count
                total_params += active_mask.sum().item()
        
        if total_params > 0:
            actual_pruning_ratio = total_pruned / total_params
            pruning_stats.update({
                "pruned_parameters": total_pruned,
                "pruning_ratio": actual_pruning_ratio,
                "target_pruning_ratio": target_pruning_ratio
            })
        
        # Save statistics pruning
        self.pruning_statistics.append({
            "task_id": task_metadata.task_id,
            "timestamp": datetime.now().isoformat(),
            **pruning_stats
        })
        
        self.logger.info(
            f"Performed regime-aware pruning for {task_metadata.market_regime}: "
            f"{total_pruned} parameters pruned ({pruning_stats['pruning_ratio']:.3f} ratio)"
        )
        
        return pruning_stats
    
    def _add_samples_to_memory(self, features: torch.Tensor, targets: torch.Tensor, task_metadata: TaskMetadata) -> None:
        """
        Add samples in buffer memory
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
        """
        # Minimum number samples (main information in architecture)
        num_samples = min(features.size(0) // 5, 20)
        indices = torch.randperm(features.size(0))[:num_samples]
        
        samples = []
        for i in indices:
            sample = MemorySample(
                features=features[i],
                target=targets[i],
                task_id=task_metadata.task_id,
                timestamp=task_metadata.start_time,
                market_regime=task_metadata.market_regime,
                asset=task_metadata.assets[0] if task_metadata.assets else "UNKNOWN",
                timeframe=task_metadata.timeframe,
                uncertainty_score=0.5,
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
        # Setup masks for tasks
        task_mask = self.packnet_mask.get_task_mask(task_id)
        if task_mask is None:
            self.logger.warning(f"No mask found for task {task_id}")
            return {"error": "mask_not_found", "task_id": task_id}
        
        self.model.eval()
        
        features = torch.tensor(test_data["features"], dtype=torch.float32).to(self.config.device)
        targets = torch.tensor(test_data["targets"], dtype=torch.float32).to(self.config.device)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        with torch.no_grad():
            # Apply masks to model for inference
            original_params = {}
            for name, param in self.model.named_parameters():
                if name in task_mask:
                    original_params[name] = param.data.clone()
                    # parameters for evaluation
                    param.data *= task_mask[name].float()
            
            # Prediction
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
            
            # Restore original parameters
            for name, original_param in original_params.items():
                self.model.state_dict()[name].copy_(original_param)
            
            # Metrics
            pred_direction = (predictions > 0).float()
            target_direction = (targets > 0).float()
            accuracy = (pred_direction == target_direction).float().mean().item()
            
            mae = torch.abs(predictions - targets).mean().item()
            rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        # Statistics use parameters for this tasks
        active_params = sum(mask.sum().item() for mask in task_mask.values())
        total_params = sum(mask.numel() for mask in task_mask.values())
        utilization = active_params / total_params if total_params > 0 else 0.0
        
        metrics = {
            "task_id": task_id,
            "test_loss": loss.item(),
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "parameter_utilization": utilization,
            "active_parameters": active_params,
            "num_samples": features.size(0)
        }
        
        self.logger.info(
            f"Task {task_id} evaluation - Accuracy: {accuracy:.3f}, "
            f"Loss: {loss.item():.4f}, Utilization: {utilization:.3f}"
        )
        return metrics
    
    def get_packnet_statistics(self) -> Dict[str, Any]:
        """
        Get statistics PackNet training
        
        Returns:
            Dict with statistics PackNet
        """
        stats = {
            "strategy": "packnet",
            "importance_driven_allocation": self.importance_driven_allocation,
            "adaptive_compression": self.adaptive_compression,
            "market_regime_pruning": self.market_regime_pruning,
            "base_allocation_ratio": self.base_allocation_ratio,
            "task_allocation_ratios": dict(self.task_allocation_ratios),
            "compression_ratios": dict(self.compression_ratios)
        }
        
        # Statistics capacity
        capacity_stats = self.packnet_mask.get_capacity_statistics()
        stats.update(capacity_stats)
        
        # Statistics pruning
        if self.pruning_statistics:
            total_pruned = sum(stat.get("pruned_parameters", 0) for stat in self.pruning_statistics)
            avg_pruning_ratio = np.mean([stat.get("pruning_ratio", 0) for stat in self.pruning_statistics])
            
            stats.update({
                "total_pruned_parameters": total_pruned,
                "average_pruning_ratio": avg_pruning_ratio,
                "pruning_operations": len(self.pruning_statistics)
            })
        
        # History capacity
        if self.capacity_history:
            latest_capacity = self.capacity_history[-1]
            stats["latest_capacity_snapshot"] = latest_capacity
        
        # Memory buffer statistics
        if self.memory_buffer:
            memory_stats = self.memory_buffer.get_memory_statistics()
            stats["memory_buffer"] = memory_stats
        
        return stats
    
    def optimize_model_size(self) -> Dict[str, Any]:
        """
        Optimization size model through pruning
        
        Adaptive model compression
        
        Returns:
            Statistics optimization
        """
        initial_capacity = self.packnet_mask.get_capacity_statistics()
        
        # pruning unused parameters
        total_removed = 0
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Check use parameter in from tasks
            used_mask = torch.zeros_like(param, dtype=torch.bool)
            
            for task_mask in self.packnet_mask.task_masks.values():
                if name in task_mask:
                    used_mask |= task_mask[name]
            
            # Remove unused parameters from free capacity
            if name in self.packnet_mask.free_capacity:
                unused_free = self.packnet_mask.free_capacity[name] & (~used_mask)
                removed_count = unused_free.sum().item()
                
                if removed_count > 0:
                    # Update free capacity
                    self.packnet_mask.free_capacity[name] &= (~unused_free)
                    total_removed += removed_count
        
        final_capacity = self.packnet_mask.get_capacity_statistics()
        
        optimization_stats = {
            "removed_parameters": total_removed,
            "initial_utilization": initial_capacity["overall_utilization"],
            "final_utilization": final_capacity["overall_utilization"],
            "optimization_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Model size optimization completed: {optimization_stats}")
        return optimization_stats
    
    def __repr__(self) -> str:
        capacity_stats = self.packnet_mask.get_capacity_statistics()
        return (
            f"PackNetLearner("
            f"tasks={len(self.packnet_mask.task_masks)}, "
            f"utilization={capacity_stats['overall_utilization']:.3f}, "
            f"allocation_ratio={self.base_allocation_ratio})"
        )