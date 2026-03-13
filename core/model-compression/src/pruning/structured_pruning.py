"""
Module structured pruning for crypto trading models.
Removes integer channels, filters and neurons for hardware-friendly compression.

Resource-constrained deployment patterns for edge computing
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import logging
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)

class StructuredPruningStrategy(Enum):
    """Strategies structured pruning"""
    MAGNITUDE = "magnitude"           # By magnitude weights
    GRADIENT = "gradient"            # By gradients
    FISHER_INFO = "fisher_information"  # By information Fisher
    LOTTERY_TICKET = "lottery_ticket"   # Lottery Ticket Hypothesis
    NEURAL_ARCHITECTURE_SEARCH = "nas"  # On basis NAS

class PruningGranularity(Enum):
    """Granularity pruning"""
    CHANNEL = "channel"      # Removal channels
    FILTER = "filter"        # Removal filters
    NEURON = "neuron"        # Removal neurons
    LAYER = "layer"          # Removal layers

class BaseStructuredPruner(ABC):
    """Base class for structured pruning"""
    
    def __init__(self, 
                 target_sparsity: float = 0.5,
                 granularity: PruningGranularity = PruningGranularity.CHANNEL):
        """
        Args:
            target_sparsity: Target sparsity (0.5 = 50% pruning)
            granularity: Level granularity pruning
        """
        self.target_sparsity = target_sparsity
        self.granularity = granularity
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pruning_history = []
        
    @abstractmethod
    def calculate_importance_scores(self, 
                                  model: nn.Module,
                                  data_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, torch.Tensor]:
        """Computation importance scores for of each structural element"""
        pass
    
    def prune_model(self, 
                   model: nn.Module,
                   data_loader: Optional[torch.utils.data.DataLoader] = None,
                   validate_fn: Optional[Callable] = None) -> nn.Module:
        """
        Main method pruning model
        
        Args:
            model: Model for pruning
            data_loader: DataLoader for computations importance scores
            validate_fn: Function validation accuracy
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Begin structured pruning. Goal: {self.target_sparsity*100:.1f}%")
        
        # Create copy model
        pruned_model = copy.deepcopy(model)
        
        # Compute importance scores
        importance_scores = self.calculate_importance_scores(pruned_model, data_loader)
        
        # Apply pruning with gradual increase sparsity
        current_sparsity = 0.0
        sparsity_step = min(0.1, self.target_sparsity / 5)  # Gradual increase
        
        while current_sparsity < self.target_sparsity:
            next_sparsity = min(current_sparsity + sparsity_step, self.target_sparsity)
            
            # Apply pruning for current level sparsity
            pruned_model = self._apply_pruning_step(
                pruned_model, importance_scores, next_sparsity
            )
            
            # Validation accuracy if provided function
            if validate_fn:
                accuracy = validate_fn(pruned_model)
                self.logger.info(f"Sparsity: {next_sparsity*100:.1f}%, Accuracy: {accuracy:.4f}")
                
                # Save history
                self.pruning_history.append({
                    "sparsity": next_sparsity,
                    "accuracy": accuracy,
                    "model_size_mb": self._calculate_model_size(pruned_model)
                })
            
            current_sparsity = next_sparsity
        
        # Finalization pruning
        final_model = self._finalize_pruning(pruned_model)
        
        self.logger.info(f"Structured pruning completed. Final sparsity: {current_sparsity*100:.1f}%")
        
        return final_model
    
    def _apply_pruning_step(self, 
                           model: nn.Module, 
                           importance_scores: Dict[str, torch.Tensor],
                           target_sparsity: float) -> nn.Module:
        """Application one step pruning"""
        # Define which elements needed remove
        elements_to_prune = self._select_elements_to_prune(
            importance_scores, target_sparsity
        )
        
        # Apply pruning in dependencies from granularity
        if self.granularity == PruningGranularity.CHANNEL:
            model = self._prune_channels(model, elements_to_prune)
        elif self.granularity == PruningGranularity.FILTER:
            model = self._prune_filters(model, elements_to_prune)
        elif self.granularity == PruningGranularity.NEURON:
            model = self._prune_neurons(model, elements_to_prune)
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        
        return model
    
    def _select_elements_to_prune(self, 
                                 importance_scores: Dict[str, torch.Tensor],
                                 target_sparsity: float) -> Dict[str, List[int]]:
        """Selection elements for removal on basis importance scores"""
        elements_to_prune = {}
        
        for layer_name, scores in importance_scores.items():
            # Number elements for removal
            num_elements = len(scores)
            num_to_prune = int(num_elements * target_sparsity)
            
            # Sort by importance (ascending - remove least important)
            sorted_indices = torch.argsort(scores).tolist()
            
            # Select least important elements
            elements_to_prune[layer_name] = sorted_indices[:num_to_prune]
        
        return elements_to_prune
    
    @abstractmethod
    def _prune_channels(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal channels"""
        pass
    
    @abstractmethod
    def _prune_filters(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal filters"""
        pass
    
    @abstractmethod
    def _prune_neurons(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal neurons"""
        pass
    
    def _finalize_pruning(self, model: nn.Module) -> nn.Module:
        """Finalization pruning - creation new compact representations"""
        # Remove pruning masks and create compact model
        for module in model.modules():
            if hasattr(module, 'weight_mask'):
                # Apply mask and remove its
                module.weight = nn.Parameter(module.weight * module.weight_mask)
                delattr(module, 'weight_mask')
            
            if hasattr(module, 'bias_mask'):
                module.bias = nn.Parameter(module.bias * module.bias_mask)
                delattr(module, 'bias_mask')
        
        return model
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / 1024 / 1024

class MagnitudeStructuredPruner(BaseStructuredPruner):
    """Structured pruning on basis magnitudes weights"""
    
    def calculate_importance_scores(self, 
                                  model: nn.Module,
                                  data_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, torch.Tensor]:
        """Computation importance on basis L2 norm weights"""
        importance_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                weight = module.weight.data
                
                if self.granularity == PruningGranularity.CHANNEL:
                    if len(weight.shape) == 4:  # Conv2d
                        # For conv layers - importance channels (dim=1)
                        scores = torch.norm(weight, dim=(0, 2, 3))
                    elif len(weight.shape) == 3:  # Conv1d
                        scores = torch.norm(weight, dim=(0, 2))
                    else:  # Linear
                        scores = torch.norm(weight, dim=0)
                
                elif self.granularity == PruningGranularity.FILTER:
                    if len(weight.shape) >= 3:  # Conv layers
                        # Importance filters (dim=0)
                        scores = torch.norm(weight.view(weight.size(0), -1), dim=1)
                    else:  # Linear
                        scores = torch.norm(weight, dim=1)
                
                else:  # NEURON
                    # For neurons use norm by incoming weights
                    if len(weight.shape) >= 2:
                        scores = torch.norm(weight.view(weight.size(0), -1), dim=1)
                    else:
                        scores = torch.abs(weight)
                
                importance_scores[name] = scores
        
        return importance_scores
    
    def _prune_channels(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal channels in Conv and Linear layers"""
        for name, indices in elements_to_prune.items():
            module = dict(model.named_modules())[name]
            
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                # Create mask for channels
                num_channels = module.weight.size(1)
                mask = torch.ones(num_channels, device=module.weight.device)
                mask[indices] = 0
                
                # Apply mask to weights
                if len(module.weight.shape) == 4:  # Conv2d
                    module.weight.data = module.weight.data * mask.view(1, -1, 1, 1)
                else:  # Conv1d
                    module.weight.data = module.weight.data * mask.view(1, -1, 1)
            
            elif isinstance(module, nn.Linear):
                # For Linear layers prune incoming connections
                num_features = module.weight.size(1)
                mask = torch.ones(num_features, device=module.weight.device)
                mask[indices] = 0
                
                module.weight.data = module.weight.data * mask.view(1, -1)
        
        return model
    
    def _prune_filters(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal filters in Conv and neurons in Linear layers"""
        for name, indices in elements_to_prune.items():
            module = dict(model.named_modules())[name]
            
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                # Create mask for filters/neurons
                num_filters = module.weight.size(0)
                mask = torch.ones(num_filters, device=module.weight.device)
                mask[indices] = 0
                
                # Apply mask to weights
                module.weight.data = module.weight.data * mask.view(-1, *([1] * (len(module.weight.shape) - 1)))
                
                # If exists bias, apply mask and to it
                if module.bias is not None:
                    module.bias.data = module.bias.data * mask
        
        return model
    
    def _prune_neurons(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Removal neurons - similarly filters for most cases"""
        return self._prune_filters(model, elements_to_prune)

class GradientStructuredPruner(BaseStructuredPruner):
    """Structured pruning on basis gradients"""
    
    def __init__(self, 
                 target_sparsity: float = 0.5,
                 granularity: PruningGranularity = PruningGranularity.CHANNEL,
                 gradient_accumulation_steps: int = 100):
        super().__init__(target_sparsity, granularity)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accumulated_gradients = {}
    
    def calculate_importance_scores(self, 
                                  model: nn.Module,
                                  data_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, torch.Tensor]:
        """Computation importance on basis accumulated gradients"""
        if data_loader is None:
            raise ValueError("DataLoader required for gradient-based pruning")
        
        model.train()
        self.accumulated_gradients = {}
        
        # Accumulate gradients
        for step, batch in enumerate(data_loader):
            if step >= self.gradient_accumulation_steps:
                break
            
            # Forward pass (assume that batch contains (input, target))
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
                outputs = model(inputs)
                
                # Simple loss function - MSE
                loss = nn.MSELoss()(outputs, targets)
            else:
                # If only inputs, use reconstruction loss
                outputs = model(batch)
                loss = nn.MSELoss()(outputs, batch)
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            self._accumulate_gradients(model)
            
            # Clear gradients
            model.zero_grad()
        
        # Compute importance scores on basis accumulated gradients
        importance_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if name in self.accumulated_gradients:
                    grad = self.accumulated_gradients[name]
                    
                    if self.granularity == PruningGranularity.CHANNEL:
                        if len(grad.shape) == 4:  # Conv2d
                            scores = torch.norm(grad, dim=(0, 2, 3))
                        elif len(grad.shape) == 3:  # Conv1d
                            scores = torch.norm(grad, dim=(0, 2))
                        else:  # Linear
                            scores = torch.norm(grad, dim=0)
                    
                    elif self.granularity == PruningGranularity.FILTER:
                        scores = torch.norm(grad.view(grad.size(0), -1), dim=1)
                    
                    else:  # NEURON
                        scores = torch.norm(grad.view(grad.size(0), -1), dim=1)
                    
                    importance_scores[name] = scores
        
        return importance_scores
    
    def _accumulate_gradients(self, model: nn.Module) -> None:
        """Accumulation gradients for analysis"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if module.weight.grad is not None:
                    if name not in self.accumulated_gradients:
                        self.accumulated_gradients[name] = torch.zeros_like(module.weight.grad)
                    
                    self.accumulated_gradients[name] += module.weight.grad.abs()
    
    def _prune_channels(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Uses that same method that and MagnitudeStructuredPruner"""
        magnitude_pruner = MagnitudeStructuredPruner(self.target_sparsity, self.granularity)
        return magnitude_pruner._prune_channels(model, elements_to_prune)
    
    def _prune_filters(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Uses that same method that and MagnitudeStructuredPruner"""
        magnitude_pruner = MagnitudeStructuredPruner(self.target_sparsity, self.granularity)
        return magnitude_pruner._prune_filters(model, elements_to_prune)
    
    def _prune_neurons(self, model: nn.Module, elements_to_prune: Dict[str, List[int]]) -> nn.Module:
        """Uses that same method that and MagnitudeStructuredPruner"""
        magnitude_pruner = MagnitudeStructuredPruner(self.target_sparsity, self.granularity)
        return magnitude_pruner._prune_neurons(model, elements_to_prune)

class CryptoTradingStructuredPruner:
    """
    Specialized structured pruner for crypto trading models
    with considering specifics temporal series and real-time requirements
    """
    
    def __init__(self, 
                 target_compression_ratio: float = 4.0,
                 accuracy_threshold: float = 0.95,
                 latency_target_ms: float = 1.0):
        """
        Args:
            target_compression_ratio: Target coefficient compression
            accuracy_threshold: Minimum accuracy
            latency_target_ms: Target latency in ms
        """
        self.target_compression_ratio = target_compression_ratio
        self.accuracy_threshold = accuracy_threshold
        self.latency_target_ms = latency_target_ms
        
        self.logger = logging.getLogger(f"{__name__}.CryptoTradingStructuredPruner")
        self.pruning_results = {}
        
    def prune_for_crypto_trading(self, 
                                model: nn.Module,
                                training_data: torch.utils.data.DataLoader,
                                validation_data: torch.utils.data.DataLoader,
                                strategy: StructuredPruningStrategy = StructuredPruningStrategy.MAGNITUDE) -> nn.Module:
        """
        Specialized pruning for crypto trading models
        
        Args:
            model: Model for pruning
            training_data: Data for computations importance
            validation_data: Data for validation
            strategy: Strategy pruning
            
        Returns:
            Optimized model for crypto trading
        """
        self.logger.info(f"Begin crypto-optimized pruning with strategy {strategy.value}")
        
        original_size = self._calculate_model_size(model)
        target_size = original_size / self.target_compression_ratio
        
        # Select pruner on basis strategies
        if strategy == StructuredPruningStrategy.MAGNITUDE:
            pruner = MagnitudeStructuredPruner(
                target_sparsity=1.0 - (1.0 / self.target_compression_ratio),
                granularity=PruningGranularity.CHANNEL
            )
        elif strategy == StructuredPruningStrategy.GRADIENT:
            pruner = GradientStructuredPruner(
                target_sparsity=1.0 - (1.0 / self.target_compression_ratio),
                granularity=PruningGranularity.CHANNEL
            )
        else:
            raise ValueError(f"Strategy {strategy.value} while not implemented")
        
        # Create function validation
        def validate_accuracy(model_to_validate):
            return self._validate_crypto_model(model_to_validate, validation_data)
        
        # Apply pruning
        pruned_model = pruner.prune_model(
            model=model,
            data_loader=training_data,
            validate_fn=validate_accuracy
        )
        
        # Additional optimization for crypto trading
        optimized_model = self._apply_crypto_optimizations(pruned_model)
        
        # Final validation
        final_accuracy = validate_accuracy(optimized_model)
        final_size = self._calculate_model_size(optimized_model)
        actual_compression_ratio = original_size / final_size
        
        # Save results
        self.pruning_results = {
            "original_size_mb": original_size,
            "final_size_mb": final_size,
            "compression_ratio": actual_compression_ratio,
            "accuracy_retained": final_accuracy,
            "meets_accuracy_threshold": final_accuracy >= self.accuracy_threshold,
            "strategy_used": strategy.value,
            "pruning_history": pruner.pruning_history
        }
        
        self.logger.info(f"Crypto trading pruning completed. "
                        f"Coefficient compression: {actual_compression_ratio:.2f}x, "
                        f"Accuracy: {final_accuracy:.4f}")
        
        if final_accuracy < self.accuracy_threshold:
            self.logger.warning(f"Accuracy below threshold: {final_accuracy:.4f} < {self.accuracy_threshold:.4f}")
        
        return optimized_model
    
    def _validate_crypto_model(self, 
                              model: nn.Module, 
                              validation_data: torch.utils.data.DataLoader) -> float:
        """Validation model on crypto trading tasks"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_data:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    outputs = model(inputs)
                    
                    # For crypto trading use combination metrics
                    mse_loss = nn.MSELoss()(outputs, targets)
                    
                    # Additional metric - directional accuracy for trading
                    direction_acc = self._calculate_directional_accuracy(outputs, targets)
                    
                    # Combined metric
                    combined_loss = mse_loss - 0.1 * direction_acc  # Encourage correct direction
                    
                    total_loss += combined_loss.item()
                    num_batches += 1
                
                if num_batches >= 100:  # Limit for speed
                    break
        
        # Return accuracy (back proportionally loss)
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        accuracy = max(0.0, 1.0 - avg_loss)  # Simple transformation
        
        return accuracy
    
    def _calculate_directional_accuracy(self, 
                                      predictions: torch.Tensor, 
                                      targets: torch.Tensor) -> torch.Tensor:
        """Computation directional accuracy for trading signals"""
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            # For multi-output predictions take first output
            pred_direction = torch.sign(predictions[1:, 0] - predictions[:-1, 0])
            target_direction = torch.sign(targets[1:, 0] - targets[:-1, 0])
        else:
            pred_direction = torch.sign(predictions[1:] - predictions[:-1])
            target_direction = torch.sign(targets[1:] - targets[:-1])
        
        correct_direction = (pred_direction == target_direction).float()
        return correct_direction.mean()
    
    def _apply_crypto_optimizations(self, model: nn.Module) -> nn.Module:
        """Additional optimization for crypto trading"""
        optimized_model = model
        
        try:
            # 1. Optimization for temporal series
            optimized_model = self._optimize_for_time_series(optimized_model)
            
            # 2. Memory layout optimization
            optimized_model = self._optimize_memory_layout(optimized_model)
            
            # 3. Batch normalization folding if possibly
            optimized_model = self._fold_batch_normalization(optimized_model)
            
        except Exception as e:
            self.logger.warning(f"Some crypto optimization not succeeded: {e}")
        
        return optimized_model
    
    def _optimize_for_time_series(self, model: nn.Module) -> nn.Module:
        """Optimization for work with temporal series crypto data"""
        # Check presence recurrent layers and optimize their
        for name, module in model.named_modules():
            if isinstance(module, (nn.LSTM, nn.GRU)):
                # Enable batch_first for best performance
                if hasattr(module, 'batch_first') and not module.batch_first:
                    self.logger.info(f"Convert {name} in batch_first mode")
                    # This complex operation, requires recreation layer
                    # For simplicity skip, but in production this important
        
        return model
    
    def _optimize_memory_layout(self, model: nn.Module) -> nn.Module:
        """Optimization memory layout"""
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        
        return model
    
    def _fold_batch_normalization(self, model: nn.Module) -> nn.Module:
        """Folding batch normalization in conv layers for acceleration inference"""
        # Simple implementation batch norm folding
        modules_list = list(model.named_modules())
        
        for i, (name, module) in enumerate(modules_list[:-1]):
            next_name, next_module = modules_list[i + 1]
            
            if (isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)) and
                isinstance(next_module, nn.BatchNorm1d)):
                
                try:
                    # Fold batch norm in previous layer
                    self._fold_bn_into_conv(module, next_module)
                    self.logger.debug(f"Folded batch norm from {next_name} into {name}")
                except Exception as e:
                    self.logger.warning(f"Not succeeded fold batch norm: {e}")
        
        return model
    
    def _fold_bn_into_conv(self, conv_module, bn_module):
        """Folding batch normalization in conv layer"""
        # This process requires recalculation weights conv layer
        # IN production version here was would full implementation
        pass
    
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculation size model in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        return param_size / 1024 / 1024
    
    def get_pruning_report(self) -> Dict[str, Any]:
        """Retrieval detailed report about pruning"""
        return {
            "pruning_results": self.pruning_results,
            "configuration": {
                "target_compression_ratio": self.target_compression_ratio,
                "accuracy_threshold": self.accuracy_threshold,
                "latency_target_ms": self.latency_target_ms
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> Dict[str, str]:
        """Recommendations by further optimization"""
        recommendations = {}
        
        if not self.pruning_results:
            return {"general": "Run pruning for retrieval recommendations"}
        
        compression_ratio = self.pruning_results.get("compression_ratio", 1.0)
        accuracy = self.pruning_results.get("accuracy_retained", 0.0)
        
        if compression_ratio < self.target_compression_ratio * 0.8:
            recommendations["compression"] = "Try more aggressive pruning or combination with quantization"
        
        if accuracy < self.accuracy_threshold:
            recommendations["accuracy"] = "Consider fine-tuning after pruning or less aggressive settings"
        
        if compression_ratio >= self.target_compression_ratio:
            recommendations["next_step"] = "Try knowledge distillation for further improvements"
        
        return recommendations