"""
Gradient Utilities for Meta-Learning
Advanced Gradient Management

Utilities for work with gradients in meta-training: computation norms,
modification, analysis and optimization gradient operations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Iterator
import numpy as np
import logging
from collections import defaultdict, OrderedDict
import math


class GradientManager:
    """
    Manager for work with gradients in meta-training
    
    Gradient Management System
    - Efficient gradient computation
    - Memory optimization
    - Gradient analysis and debugging
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics gradients
        self.gradient_stats = defaultdict(list)
        self.computation_stats = defaultdict(list)
        
    def compute_gradient_norm(
        self,
        parameters: Iterator[nn.Parameter],
        norm_type: float = 2.0
    ) -> float:
        """
        Computes norm gradients
        
        Args:
            parameters: Iterator parameters model
            norm_type: Type norms (1, 2, inf)
            
        Returns:
            Value norms gradients
        """
        parameters = list(parameters)
        
        if len(parameters) == 0:
            return 0.0
        
        device = parameters[0].device
        
        if norm_type == float('inf'):
            # L-infinity norm
            total_norm = max(
                param.grad.detach().abs().max().to(device) 
                for param in parameters 
                if param.grad is not None
            )
        else:
            # L-p norm
            total_norm = torch.norm(
                torch.stack([
                    torch.norm(param.grad.detach(), norm_type).to(device)
                    for param in parameters
                    if param.grad is not None
                ]),
                norm_type
            )
        
        return total_norm.item()
    
    def compute_gradient_stats(
        self,
        parameters: Iterator[nn.Parameter]
    ) -> Dict[str, float]:
        """
        Computes statistics gradients
        
        Args:
            parameters: Iterator parameters model
            
        Returns:
            Dictionary with statistics
        """
        parameters = list(parameters)
        
        if not any(param.grad is not None for param in parameters):
            return {
                'mean': 0.0,
                'std': 0.0,
                'max': 0.0,
                'min': 0.0,
                'norm_l1': 0.0,
                'norm_l2': 0.0,
                'norm_inf': 0.0
            }
        
        # Collect all gradients
        all_grads = torch.cat([
            param.grad.detach().flatten()
            for param in parameters
            if param.grad is not None
        ])
        
        stats = {
            'mean': float(torch.mean(all_grads).item()),
            'std': float(torch.std(all_grads).item()),
            'max': float(torch.max(all_grads).item()),
            'min': float(torch.min(all_grads).item()),
            'norm_l1': float(torch.norm(all_grads, p=1).item()),
            'norm_l2': float(torch.norm(all_grads, p=2).item()),
            'norm_inf': float(torch.norm(all_grads, p=float('inf')).item())
        }
        
        # Additional statistics
        stats['zero_ratio'] = float((all_grads == 0).float().mean().item())
        stats['positive_ratio'] = float((all_grads > 0).float().mean().item())
        
        return stats
    
    def clip_gradients(
        self,
        parameters: Iterator[nn.Parameter],
        max_norm: float,
        norm_type: float = 2.0
    ) -> float:
        """
        Trims gradients and returns original norm
        
        Args:
            parameters: Iterator parameters model
            max_norm: Maximum norm gradients
            norm_type: Type norms
            
        Returns:
            Original norm gradients until trimming
        """
        parameters = list(parameters)
        
        # Compute current norm
        total_norm = self.compute_gradient_norm(parameters, norm_type)
        
        # Trim if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for param in parameters:
                if param.grad is not None:
                    param.grad.detach().mul_(clip_coef)
        
        return total_norm
    
    def scale_gradients(
        self,
        parameters: Iterator[nn.Parameter],
        scale_factor: float
    ) -> None:
        """
        Scales gradients
        
        Args:
            parameters: Iterator parameters model
            scale_factor: Coefficient scaling
        """
        for param in parameters:
            if param.grad is not None:
                param.grad.detach().mul_(scale_factor)
    
    def add_gradient_noise(
        self,
        parameters: Iterator[nn.Parameter],
        noise_std: float,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Adds noise to gradients for regularization
        
        Args:
            parameters: Iterator parameters model
            noise_std: Standard deviation noise
            device: Device for computations
        """
        for param in parameters:
            if param.grad is not None:
                if device is None:
                    device = param.grad.device
                
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.detach().add_(noise)
    
    def compute_gradient_similarity(
        self,
        grads1: List[torch.Tensor],
        grads2: List[torch.Tensor],
        similarity_type: str = "cosine"
    ) -> float:
        """
        Computes similarity between two sets gradients
        
        Args:
            grads1: First set gradients
            grads2: Second set gradients
            similarity_type: Type similarity (cosine, l2, pearson)
            
        Returns:
            Value similarity
        """
        if len(grads1) != len(grads2):
            raise ValueError("Gradient lists must have the same length")
        
        # Merge all gradients in one tensor
        flat_grads1 = torch.cat([g.flatten() for g in grads1])
        flat_grads2 = torch.cat([g.flatten() for g in grads2])
        
        if similarity_type == "cosine":
            # Cosine similarity
            dot_product = torch.dot(flat_grads1, flat_grads2)
            norm1 = torch.norm(flat_grads1)
            norm2 = torch.norm(flat_grads2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity.item())
        
        elif similarity_type == "l2":
            # L2 distance (inverted for similarity)
            l2_distance = torch.norm(flat_grads1 - flat_grads2)
            similarity = 1.0 / (1.0 + l2_distance.item())
            return float(similarity)
        
        elif similarity_type == "pearson":
            # Correlation Pearson
            mean1 = torch.mean(flat_grads1)
            mean2 = torch.mean(flat_grads2)
            
            numerator = torch.sum((flat_grads1 - mean1) * (flat_grads2 - mean2))
            denominator = torch.sqrt(
                torch.sum((flat_grads1 - mean1) ** 2) * 
                torch.sum((flat_grads2 - mean2) ** 2)
            )
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return float(correlation.item())
        
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    def analyze_gradient_flow(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyzes flow gradients through layers model
        
        Args:
            model: Model for analysis
            layer_names: Names layers for analysis (all, if None)
            
        Returns:
            Dictionary with analysis gradient flow
        """
        gradient_flow = {}
        
        for name, param in model.named_parameters():
            if layer_names is None or any(layer_name in name for layer_name in layer_names):
                if param.grad is not None:
                    grad_stats = {
                        'mean': float(torch.mean(param.grad).item()),
                        'std': float(torch.std(param.grad).item()),
                        'max': float(torch.max(param.grad).item()),
                        'min': float(torch.min(param.grad).item()),
                        'norm': float(torch.norm(param.grad).item()),
                        'shape': list(param.grad.shape),
                        'numel': int(param.grad.numel())
                    }
                    
                    # Check on problems
                    grad_stats['has_nan'] = bool(torch.isnan(param.grad).any().item())
                    grad_stats['has_inf'] = bool(torch.isinf(param.grad).any().item())
                    grad_stats['is_zero'] = bool(torch.all(param.grad == 0).item())
                    
                    gradient_flow[name] = grad_stats
        
        return gradient_flow
    
    def detect_gradient_problems(
        self,
        model: nn.Module,
        gradient_threshold: float = 1e-7
    ) -> Dict[str, Any]:
        """
        Detects problems with gradients
        
        Args:
            model: Model for analysis
            gradient_threshold: Threshold for detection vanishing gradients
            
        Returns:
            Dictionary with detected problems
        """
        problems = {
            'vanishing_gradients': [],
            'exploding_gradients': [],
            'nan_gradients': [],
            'inf_gradients': [],
            'zero_gradients': []
        }
        
        gradient_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                gradient_norms[name] = grad_norm
                
                # Vanishing gradients
                if grad_norm < gradient_threshold:
                    problems['vanishing_gradients'].append({
                        'layer': name,
                        'norm': grad_norm
                    })
                
                # Exploding gradients
                if grad_norm > 100:  # Threshold for exploding gradients
                    problems['exploding_gradients'].append({
                        'layer': name,
                        'norm': grad_norm
                    })
                
                # NaN gradients
                if torch.isnan(param.grad).any():
                    problems['nan_gradients'].append(name)
                
                # Inf gradients
                if torch.isinf(param.grad).any():
                    problems['inf_gradients'].append(name)
                
                # Zero gradients
                if torch.all(param.grad == 0):
                    problems['zero_gradients'].append(name)
        
        # Add statistics
        if gradient_norms:
            problems['statistics'] = {
                'mean_norm': float(np.mean(list(gradient_norms.values()))),
                'std_norm': float(np.std(list(gradient_norms.values()))),
                'max_norm': float(np.max(list(gradient_norms.values()))),
                'min_norm': float(np.min(list(gradient_norms.values())))
            }
        
        return problems
    
    def track_gradient_updates(
        self,
        model: nn.Module,
        step_name: str
    ) -> None:
        """
        Tracks updates gradients for analysis
        
        Args:
            model: Model for tracking
            step_name: Name step for identification
        """
        gradient_stats = self.compute_gradient_stats(model.parameters())
        
        # Record statistics
        for stat_name, value in gradient_stats.items():
            self.gradient_stats[f"{step_name}_{stat_name}"].append(value)
        
        # Record time
        import time
        self.computation_stats[f"{step_name}_timestamp"].append(time.time())
    
    def get_gradient_statistics_summary(self) -> Dict[str, Any]:
        """
        Returns summary statistics gradients
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for stat_name, values in self.gradient_stats.items():
            if values:
                summary[stat_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return summary
    
    def clear_statistics(self) -> None:
        """Clears accumulated statistics"""
        self.gradient_stats.clear()
        self.computation_stats.clear()


class HigherOrderGradients:
    """
    Utilities for work with gradients higher orders
    
    Higher-Order Gradient Computation
    - Second-order gradient computation
    - Hessian approximations
    - Memory-efficient implementation
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_hessian_vector_product(
        self,
        loss: torch.Tensor,
        parameters: List[nn.Parameter],
        vector: List[torch.Tensor],
        retain_graph: bool = True
    ) -> List[torch.Tensor]:
        """
        Computes product matrix Hessian on vector
        
        Args:
            loss: Function losses
            parameters: Parameters model
            vector: Vector for multiplication
            retain_graph: Save whether graph computations
            
        Returns:
            Result products Hv
        """
        # First derivatives
        grads = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True
        )
        
        # Scalar product gradients with vector
        grad_vector_product = torch.sum(
            torch.stack([
                torch.sum(g * v) for g, v in zip(grads, vector)
            ])
        )
        
        # Second derivatives
        hv_grads = torch.autograd.grad(
            grad_vector_product, parameters, retain_graph=retain_graph
        )
        
        return list(hv_grads)
    
    def approximate_hessian_diagonal(
        self,
        loss: torch.Tensor,
        parameters: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        """
        Approximates diagonal matrix Hessian
        
        Args:
            loss: Function losses
            parameters: Parameters model
            
        Returns:
            Diagonal elements Hessian
        """
        # First derivatives
        grads = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True
        )
        
        diagonal_elements = []
        
        for grad in grads:
            # Second derivatives by diagonal
            grad_sum = torch.sum(grad)
            second_grads = torch.autograd.grad(
                grad_sum, grad, retain_graph=True
            )[0]
            
            diagonal_elements.append(second_grads)
        
        return diagonal_elements
    
    def compute_gradient_of_gradient_norm(
        self,
        loss: torch.Tensor,
        parameters: List[nn.Parameter]
    ) -> List[torch.Tensor]:
        """
        Computes gradient norms gradient (for analysis convergence)
        
        Args:
            loss: Function losses
            parameters: Parameters model
            
        Returns:
            Gradient norms gradient
        """
        # First derivatives
        grads = torch.autograd.grad(
            loss, parameters, create_graph=True, retain_graph=True
        )
        
        # Norm gradient
        grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]))
        
        # Gradient norms gradient
        grad_norm_grads = torch.autograd.grad(
            grad_norm, parameters, retain_graph=True
        )
        
        return list(grad_norm_grads)


class GradientAccumulator:
    """
    Accumulator gradients for mini-batch training
    
    Gradient Accumulation
    - Memory-efficient large batch simulation
    - Stable gradient computation
    - Flexible accumulation strategies
    """
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        logger: Optional[logging.Logger] = None
    ):
        self.accumulation_steps = accumulation_steps
        self.logger = logger or logging.getLogger(__name__)
        
        self.current_step = 0
        self.accumulated_gradients = {}
        
    def accumulate_gradients(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        normalize: bool = True
    ) -> bool:
        """
        Accumulates gradients
        
        Args:
            model: Model
            loss: Function losses
            normalize: Normalize whether gradients on number steps
            
        Returns:
            True if accumulation completed and possible do step optimization
        """
        # Normalize loss on number steps accumulation
        if normalize:
            loss = loss / self.accumulation_steps
        
        # Compute gradients
        loss.backward()
        
        self.current_step += 1
        
        # Check, completed whether accumulation
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        
        return False
    
    def zero_grad(self, model: nn.Module) -> None:
        """Resets to zero gradients after step optimization"""
        for param in model.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    def get_effective_batch_size(self, base_batch_size: int) -> int:
        """Returns effective size batch"""
        return base_batch_size * self.accumulation_steps


class GradientProfiler:
    """
    Profiler for analysis performance gradient computations
    
    Performance Profiling
    - Gradient computation timing
    - Memory usage tracking
    - Performance optimization hints
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.timing_data = defaultdict(list)
        self.memory_data = defaultdict(list)
        
    def profile_gradient_computation(
        self,
        func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Profiles computation gradients
        
        Args:
            func: Function for profiling
            *args: Arguments function
            **kwargs: Key arguments function
            
        Returns:
            Result function
        """
        import time
        
        # Measure memory until execution
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = 0
        
        # Measure time
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Measure time after
        end_time = time.time()
        
        # Measure memory after
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_after = torch.cuda.memory_allocated()
        else:
            memory_after = 0
        
        # Record data
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        self.timing_data[func_name].append(end_time - start_time)
        self.memory_data[func_name].append(memory_after - memory_before)
        
        return result
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Returns summary profiling"""
        summary = {}
        
        for func_name in self.timing_data.keys():
            timing_values = self.timing_data[func_name]
            memory_values = self.memory_data[func_name]
            
            summary[func_name] = {
                'timing': {
                    'mean': float(np.mean(timing_values)),
                    'std': float(np.std(timing_values)),
                    'min': float(np.min(timing_values)),
                    'max': float(np.max(timing_values)),
                    'total': float(np.sum(timing_values)),
                    'count': len(timing_values)
                },
                'memory': {
                    'mean': float(np.mean(memory_values)),
                    'std': float(np.std(memory_values)),
                    'min': float(np.min(memory_values)),
                    'max': float(np.max(memory_values)),
                    'total': float(np.sum(memory_values)),
                    'count': len(memory_values)
                }
            }
        
        return summary
    
    def clear_profiling_data(self) -> None:
        """Clears data profiling"""
        self.timing_data.clear()
        self.memory_data.clear()