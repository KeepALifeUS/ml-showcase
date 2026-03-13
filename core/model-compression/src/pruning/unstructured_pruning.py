"""
Module unstructured pruning for crypto trading models.
Removes individual weights with high granularity for maximum compression.

Fine-grained optimization patterns for edge deployment
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from enum import Enum
import copy
import math

logger = logging.getLogger(__name__)

class UnstructuredPruningStrategy(Enum):
    """Strategies unstructured pruning"""
    MAGNITUDE = "magnitude"              # By absolute magnitude weights
    RANDOM = "random"                   # Random pruning
    SNIP = "snip"                      # SNIP (Single-shot Network Pruning)
    GRASP = "grasp"                    # GraSP (Gradient Signal Preservation)
    LOTTERY_TICKET = "lottery_ticket"   # Lottery Ticket Hypothesis
    MAGNITUDE_STRUCTURED = "magnitude_structured"  # Hybrid approach

class PruningSchedule(Enum):
    """Schedule for gradual pruning"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    COSINE = "cosine"

class SparsityPattern(Enum):
    """Patterns sparsity"""
    RANDOM = "random"
    BLOCK = "block"          # Block sparsity
    STRUCTURED = "structured"  # Structured sparsity
    N_M = "n_m"             # N:M sparsity (for example 2:4)

class UnstructuredPruner:
    """
    Base class unstructured pruning with support various strategies
    and schedules for crypto trading models
    """
    
    def __init__(self, 
                 target_sparsity: float = 0.9,
                 strategy: UnstructuredPruningStrategy = UnstructuredPruningStrategy.MAGNITUDE,
                 schedule: PruningSchedule = PruningSchedule.EXPONENTIAL,
                 sparsity_pattern: SparsityPattern = SparsityPattern.RANDOM):
        """
        Args:
            target_sparsity: Target sparsity (0.9 = 90% weights zeroed)
            strategy: Strategy pruning
            schedule: Schedule gradual pruning
            sparsity_pattern: Pattern sparsity
        """
        self.target_sparsity = target_sparsity
        self.strategy = strategy
        self.schedule = schedule
        self.sparsity_pattern = sparsity_pattern
        
        self.logger = logging.getLogger(f"{__name__}.UnstructuredPruner")
        self.pruning_stats = {}
        self.sparsity_history = []
        
    def prune_model(self, 
                   model: nn.Module,
                   data_loader: Optional[torch.utils.data.DataLoader] = None,
                   num_steps: int = 10,
                   fine_tune_fn: Optional[Callable] = None) -> nn.Module:
        """
        Main method unstructured pruning
        
        Args:
            model: Model for pruning
            data_loader: Data for gradients (for SNIP/GraSP)
            num_steps: Number steps gradual pruning
            fine_tune_fn: Function fine-tuning between steps
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Begin unstructured pruning "
                        f"until {self.target_sparsity*100:.1f}% sparsity")
        
        pruned_model = copy.deepcopy(model)
        
        if self.strategy in [UnstructuredPruningStrategy.SNIP, UnstructuredPruningStrategy.GRASP]:
            # Single-shot methods
            pruned_model = self._single_shot_pruning(pruned_model, data_loader)
        else:
            # Gradual pruning
            pruned_model = self._gradual_pruning(
                pruned_model, data_loader, num_steps, fine_tune_fn
            )
        
        # Finalization and analysis results
        final_sparsity = self._calculate_global_sparsity(pruned_model)
        
        self.pruning_stats = {
            "final_sparsity": final_sparsity,
            "target_sparsity": self.target_sparsity,
            "strategy": self.strategy.value,
            "schedule": self.schedule.value,
            "sparsity_pattern": self.sparsity_pattern.value,
            "num_parameters_pruned": self._count_pruned_parameters(pruned_model),
            "total_parameters": self._count_total_parameters(pruned_model)
        }
        
        self.logger.info(f"Unstructured pruning completed. "
                        f"Achieved sparsity: {final_sparsity*100:.1f}%")
        
        return pruned_model
    
    def _gradual_pruning(self, 
                        model: nn.Module,
                        data_loader: Optional[torch.utils.data.DataLoader],
                        num_steps: int,
                        fine_tune_fn: Optional[Callable]) -> nn.Module:
        """Gradual pruning with using schedule"""
        current_sparsity = 0.0
        
        for step in range(num_steps):
            # Compute target sparsity for current step
            step_sparsity = self._calculate_step_sparsity(step, num_steps)
            
            if step_sparsity <= current_sparsity:
                continue
            
            self.logger.info(f"Step {step + 1}/{num_steps}: "
                           f"sparsity {current_sparsity*100:.1f}% -> {step_sparsity*100:.1f}%")
            
            # Apply pruning for current step
            model = self._apply_pruning_step(model, step_sparsity, data_loader)
            
            # Fine-tuning if provided function
            if fine_tune_fn and step < num_steps - 1:  # Not fine-tune on last step
                model = fine_tune_fn(model)
            
            current_sparsity = step_sparsity
            
            # Save history
            actual_sparsity = self._calculate_global_sparsity(model)
            self.sparsity_history.append({
                "step": step,
                "target_sparsity": step_sparsity,
                "actual_sparsity": actual_sparsity,
                "model_size_mb": self._calculate_model_size(model)
            })
        
        return model
    
    def _single_shot_pruning(self, 
                            model: nn.Module,
                            data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Single-shot pruning methods (SNIP, GraSP)"""
        if data_loader is None:
            raise ValueError(f"DataLoader required for {self.strategy.value}")
        
        if self.strategy == UnstructuredPruningStrategy.SNIP:
            return self._snip_pruning(model, data_loader)
        elif self.strategy == UnstructuredPruningStrategy.GRASP:
            return self._grasp_pruning(model, data_loader)
        else:
            raise ValueError(f"Unsupported single-shot strategy: {self.strategy.value}")
    
    def _calculate_step_sparsity(self, step: int, total_steps: int) -> float:
        """Computation sparsity for current step on basis schedule"""
        progress = (step + 1) / total_steps
        
        if self.schedule == PruningSchedule.LINEAR:
            return progress * self.target_sparsity
        
        elif self.schedule == PruningSchedule.EXPONENTIAL:
            # Exponential schedule: fast growth in beginning
            return self.target_sparsity * (1 - (1 - progress) ** 3)
        
        elif self.schedule == PruningSchedule.POLYNOMIAL:
            # Polynomial schedule
            return self.target_sparsity * (progress ** 2)
        
        elif self.schedule == PruningSchedule.COSINE:
            # Cosine annealing schedule
            return self.target_sparsity * (1 - math.cos(progress * math.pi / 2))
        
        else:
            return progress * self.target_sparsity
    
    def _apply_pruning_step(self, 
                           model: nn.Module,
                           target_sparsity: float,
                           data_loader: Optional[torch.utils.data.DataLoader]) -> nn.Module:
        """Application one step pruning"""
        if self.strategy == UnstructuredPruningStrategy.MAGNITUDE:
            return self._magnitude_pruning(model, target_sparsity)
        
        elif self.strategy == UnstructuredPruningStrategy.RANDOM:
            return self._random_pruning(model, target_sparsity)
        
        elif self.strategy == UnstructuredPruningStrategy.LOTTERY_TICKET:
            return self._lottery_ticket_pruning(model, target_sparsity)
        
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy.value}")
    
    def _magnitude_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """Pruning by absolute magnitude weights"""
        # Collect all weights for global ranking
        all_weights = []
        layer_info = []
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                weights = module.weight.data.abs().flatten()
                all_weights.append(weights)
                layer_info.append((name, module, len(weights)))
        
        if not all_weights:
            return model
        
        # Global threshold
        all_weights_tensor = torch.cat(all_weights)
        threshold_idx = int(target_sparsity * len(all_weights_tensor))
        threshold_value = torch.kthvalue(all_weights_tensor, threshold_idx + 1).values
        
        # Apply pruning to each layer
        for name, module, _ in layer_info:
            mask = (module.weight.data.abs() >= threshold_value).float()
            
            if self.sparsity_pattern == SparsityPattern.BLOCK:
                mask = self._apply_block_sparsity(mask)
            elif self.sparsity_pattern == SparsityPattern.N_M:
                mask = self._apply_n_m_sparsity(mask, n=2, m=4)
            
            # Apply mask
            prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _random_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """Random pruning"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Create random mask
                mask = torch.rand_like(module.weight.data) > target_sparsity
                mask = mask.float()
                
                if self.sparsity_pattern == SparsityPattern.BLOCK:
                    mask = self._apply_block_sparsity(mask)
                elif self.sparsity_pattern == SparsityPattern.N_M:
                    mask = self._apply_n_m_sparsity(mask, n=2, m=4)
                
                prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _lottery_ticket_pruning(self, model: nn.Module, target_sparsity: float) -> nn.Module:
        """
        Lottery Ticket Hypothesis pruning
        Requires saved initial weights
        """
        if not hasattr(self, 'initial_weights'):
            self.logger.warning("Initial weights not saved. Use magnitude pruning.")
            return self._magnitude_pruning(model, target_sparsity)
        
        # Implementation lottery ticket: find "winning ticket"
        # This simplified version - in production needed full implementation
        return self._magnitude_pruning(model, target_sparsity)
    
    def _snip_pruning(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        SNIP (Single-shot Network Pruning)
        Is based on gradients immediately after initialization
        """
        model.train()
        
        # Compute gradients on one batch
        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, targets)
        else:
            outputs = model(batch)
            loss = nn.MSELoss()(outputs, batch)
        
        loss.backward()
        
        # Compute SNIP scores for of each parameter
        snip_scores = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.weight.grad is not None:
                    # SNIP score = |weight * gradient|
                    score = (module.weight.data * module.weight.grad).abs()
                    snip_scores[name] = score
        
        # Apply pruning on basis SNIP scores
        model = self._apply_score_based_pruning(model, snip_scores, self.target_sparsity)
        
        # Clear gradients
        model.zero_grad()
        
        return model
    
    def _grasp_pruning(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> nn.Module:
        """
        GraSP (Gradient Signal Preservation)
        Saves gradient flow through network
        """
        model.train()
        
        # Compute Hessian-gradient product for GraSP
        batch = next(iter(data_loader))
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch, batch
        
        # First forward-backward pass
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        first_grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        # Second pass for GraSP score
        grad_norm = sum(g.pow(2).sum() for g in first_grads)
        second_grads = torch.autograd.grad(grad_norm, model.parameters())
        
        # Compute GraSP scores
        grasp_scores = {}
        param_idx = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # GraSP score = |weight * hessian_gradient|
                hessian_grad = second_grads[param_idx]
                score = (module.weight.data * hessian_grad).abs()
                grasp_scores[name] = score
                param_idx += 1
        
        # Apply pruning
        model = self._apply_score_based_pruning(model, grasp_scores, self.target_sparsity)
        model.zero_grad()
        
        return model
    
    def _apply_score_based_pruning(self, 
                                  model: nn.Module,
                                  scores: Dict[str, torch.Tensor],
                                  target_sparsity: float) -> nn.Module:
        """Application pruning on basis importance scores"""
        # Collect all scores for global ranking
        all_scores = []
        layer_info = []
        
        for name, score in scores.items():
            flat_scores = score.flatten()
            all_scores.append(flat_scores)
            module = dict(model.named_modules())[name]
            layer_info.append((name, module, score.shape))
        
        # Global threshold
        all_scores_tensor = torch.cat(all_scores)
        threshold_idx = int(target_sparsity * len(all_scores_tensor))
        threshold_value = torch.kthvalue(all_scores_tensor, threshold_idx + 1).values
        
        # Apply pruning
        for name, module, original_shape in layer_info:
            score = scores[name]
            mask = (score >= threshold_value).float()
            
            if self.sparsity_pattern == SparsityPattern.BLOCK:
                mask = self._apply_block_sparsity(mask)
            elif self.sparsity_pattern == SparsityPattern.N_M:
                mask = self._apply_n_m_sparsity(mask, n=2, m=4)
            
            prune.custom_from_mask(module, name='weight', mask=mask)
        
        return model
    
    def _apply_block_sparsity(self, mask: torch.Tensor, block_size: int = 4) -> torch.Tensor:
        """Application block sparsity pattern"""
        # Simple implementation block sparsity
        original_shape = mask.shape
        
        if len(original_shape) >= 2:
            # Group in blocks and accept decision on level block
            h, w = original_shape[-2], original_shape[-1]
            
            # Padding until multiplicity block_size
            pad_h = (block_size - h % block_size) % block_size
            pad_w = (block_size - w % block_size) % block_size
            
            if pad_h > 0 or pad_w > 0:
                mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h))
            
            new_h, new_w = mask.shape[-2], mask.shape[-1]
            
            # Split on blocks
            blocks = mask.unfold(-2, block_size, block_size).unfold(-1, block_size, block_size)
            
            # For of each block: if sum > half elements, save entire block
            block_mask = blocks.sum(dim=(-2, -1)) > (block_size * block_size / 2)
            
            # Restore full mask
            expanded_mask = block_mask.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
            
            # Trim until original size
            mask = expanded_mask[..., :h, :w]
            
            # Restore original form
            if len(original_shape) > 2:
                mask = mask.view(original_shape)
        
        return mask
    
    def _apply_n_m_sparsity(self, mask: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
        """
        Application N:M sparsity pattern
        From every M weights keep N largest
        """
        original_shape = mask.shape
        flat_mask = mask.flatten()
        
        # Group by M elements
        num_groups = len(flat_mask) // m
        remainder = len(flat_mask) % m
        
        # Process complete groups
        grouped = flat_mask[:num_groups * m].view(num_groups, m)
        
        # For of each groups keep only n largest values
        topk_values, topk_indices = torch.topk(grouped, n, dim=1)
        
        # Create new mask for groups
        group_mask = torch.zeros_like(grouped)
        group_mask.scatter_(1, topk_indices, 1.0)
        
        # Restore flat mask
        new_flat_mask = torch.cat([
            group_mask.flatten(),
            flat_mask[num_groups * m:]  # Remainder remains without changes
        ])
        
        return new_flat_mask.view(original_shape)
    
    def _calculate_global_sparsity(self, model: nn.Module) -> float:
        """Computation global level sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    # Use mask if exists
                    total_params += module.weight_mask.numel()
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    # Count directly by weights
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def _count_pruned_parameters(self, model: nn.Module) -> int:
        """Counting number zeroed parameters"""
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    zero_params += (module.weight == 0).sum().item()
        
        return zero_params
    
    def _count_total_parameters(self, model: nn.Module) -> int:
        """Counting total number parameters"""
        total_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                total_params += module.weight.numel()
        
        return total_params
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / 1024 / 1024
    
    def save_initial_weights(self, model: nn.Module) -> None:
        """Saving initial weights for Lottery Ticket"""
        self.initial_weights = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                self.initial_weights[name] = module.weight.data.clone()
    
    def restore_initial_weights(self, model: nn.Module) -> nn.Module:
        """Recovery initial weights"""
        if not hasattr(self, 'initial_weights'):
            self.logger.warning("Initial weights not saved")
            return model
        
        for name, module in model.named_modules():
            if name in self.initial_weights:
                module.weight.data = self.initial_weights[name].clone()
        
        return model
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Retrieval detailed statistics pruning"""
        stats = {
            "global_stats": self.pruning_stats,
            "sparsity_history": self.sparsity_history,
            "layer_wise_sparsity": self._get_layerwise_sparsity_stats()
        }
        
        return stats
    
    def _get_layerwise_sparsity_stats(self) -> Dict[str, Dict[str, float]]:
        """Retrieval statistics sparsity by layers"""
        # This method must be called after pruning
        # IN given implementation return stub
        return {
            "note": "Statistics by layers available after application pruning to model"
        }

class CryptoTradingUnstructuredPruner:
    """
    Specialized unstructured pruner for crypto trading models
    with adaptive strategies and crypto-specific optimizations
    """
    
    def __init__(self, 
                 target_compression_ratio: float = 10.0,
                 accuracy_threshold: float = 0.95,
                 latency_target_ms: float = 0.5):
        """
        Args:
            target_compression_ratio: Target coefficient compression
            accuracy_threshold: Minimum accuracy
            latency_target_ms: Target latency in ms
        """
        self.target_compression_ratio = target_compression_ratio
        self.accuracy_threshold = accuracy_threshold
        self.latency_target_ms = latency_target_ms
        
        # Compute target sparsity from compression ratio
        self.target_sparsity = 1.0 - (1.0 / target_compression_ratio)
        
        self.logger = logging.getLogger(f"{__name__}.CryptoTradingUnstructuredPruner")
        self.optimization_results = {}
        
    def adaptive_pruning(self, 
                        model: nn.Module,
                        training_data: torch.utils.data.DataLoader,
                        validation_data: torch.utils.data.DataLoader,
                        fine_tune_fn: Optional[Callable] = None) -> nn.Module:
        """
        Adaptive pruning with automatic selection optimal strategies
        
        Args:
            model: Model for pruning
            training_data: Training data
            validation_data: Validation data  
            fine_tune_fn: Function fine-tuning
            
        Returns:
            Optimally pruned model
        """
        self.logger.info(f"Begin adaptive crypto pruning until {self.target_sparsity*100:.1f}% sparsity")
        
        # Save initial weights for lottery ticket
        initial_model = copy.deepcopy(model)
        
        # Test various strategies
        strategies_to_test = [
            (UnstructuredPruningStrategy.MAGNITUDE, PruningSchedule.EXPONENTIAL),
            (UnstructuredPruningStrategy.SNIP, PruningSchedule.LINEAR),
            (UnstructuredPruningStrategy.GRASP, PruningSchedule.POLYNOMIAL),
            (UnstructuredPruningStrategy.RANDOM, PruningSchedule.COSINE)
        ]
        
        best_model = None
        best_score = -float('inf')
        best_config = None
        
        for strategy, schedule in strategies_to_test:
            try:
                self.logger.info(f"Test strategy: {strategy.value} with schedule {schedule.value}")
                
                # Create copy model for testing
                test_model = copy.deepcopy(initial_model)
                
                # Create pruner with current configuration
                pruner = UnstructuredPruner(
                    target_sparsity=self.target_sparsity,
                    strategy=strategy,
                    schedule=schedule,
                    sparsity_pattern=SparsityPattern.RANDOM
                )
                
                # Apply pruning
                pruned_model = pruner.prune_model(
                    test_model,
                    data_loader=training_data,
                    num_steps=5,  # Less steps for fast testing
                    fine_tune_fn=fine_tune_fn
                )
                
                # Evaluate result
                score = self._evaluate_crypto_model(pruned_model, validation_data)
                
                self.logger.info(f"Strategy {strategy.value}: score = {score:.4f}")
                
                if score > best_score:
                    best_model = pruned_model
                    best_score = score
                    best_config = (strategy, schedule)
                    
            except Exception as e:
                self.logger.warning(f"Error with strategy {strategy.value}: {e}")
                continue
        
        if best_model is None:
            self.logger.error("Not succeeded find suitable strategy pruning")
            return model
        
        # Apply best strategy with complete settings
        self.logger.info(f"Best strategy: {best_config[0].value} with {best_config[1].value}")
        
        final_pruner = UnstructuredPruner(
            target_sparsity=self.target_sparsity,
            strategy=best_config[0],
            schedule=best_config[1],
            sparsity_pattern=SparsityPattern.N_M  # Use N:M for best hardware efficiency
        )
        
        final_model = final_pruner.prune_model(
            copy.deepcopy(initial_model),
            data_loader=training_data,
            num_steps=10,
            fine_tune_fn=fine_tune_fn
        )
        
        # Additional optimization for crypto trading
        optimized_model = self._apply_crypto_optimizations(final_model)
        
        # Final estimation
        final_score = self._evaluate_crypto_model(optimized_model, validation_data)
        final_sparsity = self._calculate_global_sparsity(optimized_model)
        
        # Save results
        self.optimization_results = {
            "best_strategy": best_config[0].value,
            "best_schedule": best_config[1].value,
            "final_sparsity": final_sparsity,
            "final_score": final_score,
            "compression_ratio": 1.0 / (1.0 - final_sparsity),
            "meets_accuracy_threshold": final_score >= self.accuracy_threshold,
            "pruning_stats": final_pruner.pruning_stats
        }
        
        self.logger.info(f"Adaptive crypto pruning completed. "
                        f"Sparsity: {final_sparsity*100:.1f}%, Score: {final_score:.4f}")
        
        return optimized_model
    
    def _evaluate_crypto_model(self, 
                              model: nn.Module, 
                              validation_data: torch.utils.data.DataLoader) -> float:
        """Estimation model for crypto trading tasks"""
        model.eval()
        
        total_mse = 0.0
        total_directional_acc = 0.0
        total_sharpe = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = model(inputs)
                    
                    # MSE Loss
                    mse = nn.MSELoss()(outputs, targets)
                    total_mse += mse.item()
                    
                    # Directional Accuracy for trading
                    dir_acc = self._calculate_directional_accuracy(outputs, targets)
                    total_directional_acc += dir_acc.item()
                    
                    # Simulated Sharpe ratio
                    sharpe = self._calculate_simulated_sharpe(outputs, targets)
                    total_sharpe += sharpe
                    
                    num_batches += 1
                    
                    if num_batches >= 50:  # Limit for speed
                        break
        
        if num_batches == 0:
            return 0.0
        
        # Combined score for crypto trading
        avg_mse = total_mse / num_batches
        avg_dir_acc = total_directional_acc / num_batches
        avg_sharpe = total_sharpe / num_batches
        
        # Weighted score: priority directional accuracy and sharpe ratio
        score = (
            0.3 * (1.0 / (1.0 + avg_mse)) +  # Back proportionally MSE
            0.4 * avg_dir_acc +               # Directional accuracy
            0.3 * max(0.0, avg_sharpe)        # Sharpe ratio (only positive)
        )
        
        return score
    
    def _calculate_directional_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Directional accuracy for trading signals"""
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            pred_direction = torch.sign(predictions[1:, 0] - predictions[:-1, 0])
            target_direction = torch.sign(targets[1:, 0] - targets[:-1, 0])
        else:
            pred_direction = torch.sign(predictions[1:] - predictions[:-1])
            target_direction = torch.sign(targets[1:] - targets[:-1])
        
        correct = (pred_direction == target_direction).float()
        return correct.mean()
    
    def _calculate_simulated_sharpe(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Approximate computation Sharpe ratio"""
        if len(predictions) < 2:
            return 0.0
        
        # Simulate returns on basis predictions
        pred_returns = predictions[1:] - predictions[:-1]
        actual_returns = targets[1:] - targets[:-1]
        
        # Correlation between predicted and real returns
        correlation = torch.corrcoef(torch.stack([pred_returns.flatten(), actual_returns.flatten()]))[0, 1]
        
        if torch.isnan(correlation):
            return 0.0
        
        # Simplified Sharpe: correlation * volatility adjustment
        volatility = torch.std(pred_returns)
        sharpe_approx = correlation.item() / (volatility.item() + 1e-8)
        
        return max(-2.0, min(2.0, sharpe_approx))  # Limit range
    
    def _apply_crypto_optimizations(self, model: nn.Module) -> nn.Module:
        """Crypto-specific optimization after pruning"""
        optimized_model = model
        
        try:
            # 1. Removal masks for compact representation
            optimized_model = self._finalize_sparse_model(optimized_model)
            
            # 2. Memory layout optimization
            optimized_model = self._optimize_sparse_memory_layout(optimized_model)
            
            # 3. JIT optimization if possibly
            optimized_model = self._apply_jit_optimization(optimized_model)
            
        except Exception as e:
            self.logger.warning(f"Some crypto optimization not succeeded: {e}")
        
        return optimized_model
    
    def _finalize_sparse_model(self, model: nn.Module) -> nn.Module:
        """Finalization sparse model - removal masks"""
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                # Apply mask finally
                module.weight.data = module.weight.data * module.weight_mask
                # Remove mask
                prune.remove(module, 'weight')
        
        return model
    
    def _optimize_sparse_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimization memory layout for sparse matrices"""
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Ensure contiguous layout
                if not module.weight.is_contiguous():
                    module.weight.data = module.weight.data.contiguous()
                
                # For very sparse weights possible consider sparse tensor format
                sparsity = (module.weight == 0).float().mean()
                if sparsity > 0.9:  # If >90% zeros
                    # IN production here possible convert in sparse format
                    pass
        
        return model
    
    def _apply_jit_optimization(self, model: nn.Module) -> nn.Module:
        """JIT optimization for sparse model"""
        try:
            if hasattr(torch, 'jit'):
                # Create dummy input for tracing
                dummy_input = torch.randn(1, self._estimate_input_size(model))
                traced_model = torch.jit.trace(model, dummy_input)
                return torch.jit.optimize_for_inference(traced_model)
            
        except Exception as e:
            self.logger.warning(f"JIT optimization not succeeded: {e}")
        
        return model
    
    def _estimate_input_size(self, model: nn.Module) -> int:
        """Estimation size input data"""
        first_layer = next(iter(model.modules()))
        if isinstance(first_layer, nn.Linear):
            return first_layer.in_features
        elif isinstance(first_layer, (nn.Conv1d, nn.Conv2d)):
            # For crypto data usually temporal series
            return 100  # Approximate size
        else:
            return 100  # Default value
    
    def _calculate_global_sparsity(self, model: nn.Module) -> float:
        """Computation global level sparsity"""
        total_params = 0
        zero_params = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                total_params += module.weight.numel()
                zero_params += (module.weight == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Retrieval detailed report about optimization"""
        return {
            "optimization_results": self.optimization_results,
            "configuration": {
                "target_compression_ratio": self.target_compression_ratio,
                "target_sparsity": self.target_sparsity,
                "accuracy_threshold": self.accuracy_threshold,
                "latency_target_ms": self.latency_target_ms
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> Dict[str, str]:
        """Recommendations by further optimization"""
        recommendations = {}
        
        if not self.optimization_results:
            return {"general": "Run adaptive pruning for retrieval recommendations"}
        
        final_score = self.optimization_results.get("final_score", 0.0)
        meets_threshold = self.optimization_results.get("meets_accuracy_threshold", False)
        
        if not meets_threshold:
            recommendations["accuracy"] = "Consider less aggressive pruning or fine-tuning"
        
        if final_score > 0.8:
            recommendations["next_step"] = "Try combination with quantization for further compression"
        
        if self.optimization_results.get("compression_ratio", 1.0) < self.target_compression_ratio * 0.8:
            recommendations["compression"] = "Try more aggressive settings or structured pruning"
        
        return recommendations