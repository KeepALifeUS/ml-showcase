"""
Attention utilities for analysis, optimization and debugging attention mechanisms.
Includes performance profiling, attention pattern analysis and memory optimization.

Production utilities for attention mechanism monitoring and optimization.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttentionStats:
    """Statistics about attention patterns."""
    entropy: float
    sparsity: float
    max_attention: float
    attention_variance: float
    head_diversity: float
    temporal_consistency: float
    memory_usage_mb: float
    compute_time_ms: float


class AttentionAnalyzer:
    """
    Analyzer for attention patterns and performance.
    
    Features:
    - Attention pattern analysis
    - Head diversity measurement
    - Sparsity analysis
    - Performance profiling
    - Memory usage tracking
    """
    
    def __init__(self):
        self.attention_history = []
        self.performance_history = []
    
    def analyze_attention_weights(
        self,
        attention_weights: torch.Tensor,
        head_dim: Optional[int] = None
    ) -> AttentionStats:
        """
        Analyze attention weight patterns.
        
        Args:
            attention_weights: Attention weights [batch, num_heads, seq_len, seq_len]
            head_dim: Head dimension (for memory calculation)
            
        Returns:
            AttentionStats object with analysis results
        """
        device = attention_weights.device
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        with torch.no_grad():
            # Entropy calculation (attention focus)
            # Higher entropy = more distributed attention
            entropy = self._calculate_entropy(attention_weights)
            
            # Sparsity (percentage of near-zero weights)
            sparsity = self._calculate_sparsity(attention_weights)
            
            # Maximum attention value
            max_attention = torch.max(attention_weights).item()
            
            # Attention variance (stability measure)
            attention_variance = torch.var(attention_weights).item()
            
            # Head diversity (how different are attention heads)
            head_diversity = self._calculate_head_diversity(attention_weights)
            
            # Temporal consistency (how consistent attention patterns over time)
            temporal_consistency = self._calculate_temporal_consistency(attention_weights)
            
            # Memory usage estimation
            memory_usage_mb = self._estimate_memory_usage(attention_weights, head_dim)
        
        return AttentionStats(
            entropy=entropy,
            sparsity=sparsity,
            max_attention=max_attention,
            attention_variance=attention_variance,
            head_diversity=head_diversity,
            temporal_consistency=temporal_consistency,
            memory_usage_mb=memory_usage_mb,
            compute_time_ms=0.0  # Will be set by profiler
        )
    
    def _calculate_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate attention entropy."""
        # Add small epsilon for numerical stability
        eps = 1e-8
        log_weights = torch.log(attention_weights + eps)
        entropy = -torch.sum(attention_weights * log_weights, dim=-1).mean().item()
        return entropy
    
    def _calculate_sparsity(self, attention_weights: torch.Tensor, threshold: float = 1e-3) -> float:
        """Calculate attention sparsity."""
        sparse_mask = attention_weights < threshold
        sparsity = sparse_mask.float().mean().item()
        return sparsity
    
    def _calculate_head_diversity(self, attention_weights: torch.Tensor) -> float:
        """Calculate diversity between attention heads."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Flatten sequence dimensions
        flattened = attention_weights.view(batch_size, num_heads, -1)
        
        # Calculate pairwise cosine similarities between heads
        similarities = []
        for i in range(num_heads):
            for j in range(i + 1, num_heads):
                sim = F.cosine_similarity(
                    flattened[:, i, :], 
                    flattened[:, j, :], 
                    dim=-1
                ).mean().item()
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            diversity = 1.0 - abs(avg_similarity)
        else:
            diversity = 1.0
        
        return diversity
    
    def _calculate_temporal_consistency(self, attention_weights: torch.Tensor) -> float:
        """Calculate temporal consistency of attention patterns."""
        seq_len = attention_weights.shape[-1]
        
        if seq_len < 2:
            return 1.0
        
        # Calculate consistency across sequence positions
        consistencies = []
        for i in range(seq_len - 1):
            curr_pattern = attention_weights[:, :, i, :]
            next_pattern = attention_weights[:, :, i + 1, :]
            
            consistency = F.cosine_similarity(
                curr_pattern.flatten(1), 
                next_pattern.flatten(1), 
                dim=-1
            ).mean().item()
            
            consistencies.append(abs(consistency))
        
        return sum(consistencies) / len(consistencies) if consistencies else 1.0
    
    def _estimate_memory_usage(
        self,
        attention_weights: torch.Tensor,
        head_dim: Optional[int] = None
    ) -> float:
        """Estimate memory usage in MB."""
        # Base memory for attention weights
        weights_memory = attention_weights.numel() * 4  # 4 bytes per float32
        
        # Additional memory for Q, K, V if head_dim provided
        if head_dim is not None:
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * 4  # Q, K, V
            total_memory = weights_memory + qkv_memory
        else:
            total_memory = weights_memory
        
        return total_memory / (1024 * 1024)  # Convert to MB
    
    def profile_attention_layer(
        self,
        attention_layer: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3
    ) -> Dict[str, float]:
        """
        Profile attention layer performance.
        
        Args:
            attention_layer: Attention module to profile
            input_tensor: Input tensor for profiling
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Performance statistics dictionary
        """
        device = input_tensor.device
        
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = attention_layer(input_tensor)
        
        # Synchronize GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Profile forward pass
        forward_times = []
        memory_usage = []
        
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                start_memory = torch.cuda.memory_allocated()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = attention_layer(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            forward_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if device.type == 'cuda':
                peak_memory = torch.cuda.max_memory_allocated()
                memory_usage.append((peak_memory - start_memory) / 1024 / 1024)  # MB
        
        # Calculate statistics
        stats = {
            'forward_time_mean_ms': np.mean(forward_times),
            'forward_time_std_ms': np.std(forward_times),
            'forward_time_min_ms': np.min(forward_times),
            'forward_time_max_ms': np.max(forward_times),
        }
        
        if memory_usage:
            stats.update({
                'memory_usage_mean_mb': np.mean(memory_usage),
                'memory_usage_max_mb': np.max(memory_usage),
            })
        
        return stats


class AttentionMemoryOptimizer:
    """
    Memory optimization utilities for attention mechanisms.
    
    Features:
    - Gradient checkpointing
    - Attention chunking
    - Memory-efficient attention
    - Dynamic sequence length adjustment
    """
    
    @staticmethod
    def chunk_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int = 512,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention in chunks to save memory.
        
        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            chunk_size: Size of each chunk
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch, heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = head_dim ** -0.5
        
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            
            # Chunk query
            q_chunk = query[:, :, i:end_i, :]
            
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                
                # Chunk key and value
                k_chunk = key[:, :, j:end_j, :]
                v_chunk = value[:, :, j:end_j, :]
                
                # Compute attention scores
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
                
                # Apply mask if provided
                if attention_mask is not None:
                    mask_chunk = attention_mask[:, :, i:end_i, j:end_j]
                    scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
                
                # Apply softmax and compute output
                attn_weights = F.softmax(scores, dim=-1)
                chunk_output = torch.matmul(attn_weights, v_chunk)
                
                # Accumulate output (weighted by attention strength)
                if j == 0:
                    output[:, :, i:end_i, :] = chunk_output
                else:
                    # Weight chunks by attention strength
                    attn_strength = torch.sum(attn_weights, dim=-1, keepdim=True)
                    total_strength = torch.sum(torch.softmax(scores, dim=-1), dim=-1, keepdim=True)
                    
                    weight = attn_strength / (total_strength + 1e-8)
                    output[:, :, i:end_i, :] += chunk_output * weight
        
        return output
    
    @staticmethod
    def gradient_checkpoint_attention(
        attention_fn: Callable,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply gradient checkpointing to attention function.
        
        Args:
            attention_fn: Attention function to checkpoint
            *args: Arguments for attention function
            **kwargs: Keyword arguments for attention function
        """
        return torch.utils.checkpoint.checkpoint(attention_fn, *args, **kwargs)
    
    @staticmethod
    def dynamic_sequence_truncation(
        input_tensor: torch.Tensor,
        max_memory_mb: float = 1000.0,
        head_dim: int = 64,
        num_heads: int = 8
    ) -> torch.Tensor:
        """
        Dynamically truncate sequence to fit memory constraints.
        
        Args:
            input_tensor: Input tensor [batch, seq_len, d_model]
            max_memory_mb: Maximum memory usage in MB
            head_dim: Attention head dimension
            num_heads: Number of attention heads
            
        Returns:
            Truncated tensor
        """
        batch_size, seq_len, d_model = input_tensor.shape
        
        # Estimate memory usage for attention
        attention_memory = batch_size * num_heads * seq_len * seq_len * 4  # bytes
        qkv_memory = 3 * batch_size * seq_len * d_model * 4  # Q, K, V
        total_memory_mb = (attention_memory + qkv_memory) / 1024 / 1024
        
        if total_memory_mb <= max_memory_mb:
            return input_tensor
        
        # Calculate maximum sequence length
        # Solving: batch_size * num_heads * max_seq^2 * 4 + 3 * batch_size * max_seq * d_model * 4 <= max_memory_mb * 1024^2
        # Simplified: max_seq^2 * A + max_seq * B <= C
        A = batch_size * num_heads * 4
        B = 3 * batch_size * d_model * 4
        C = max_memory_mb * 1024 * 1024
        
        # Quadratic formula: max_seq = (-B + sqrt(B^2 + 4*A*C)) / (2*A)
        discriminant = B * B + 4 * A * C
        if discriminant < 0:
            max_seq = 1
        else:
            max_seq = int((-B + math.sqrt(discriminant)) / (2 * A))
        
        max_seq = max(1, min(max_seq, seq_len))
        
        if max_seq < seq_len:
            logger.warning(f"Truncating sequence from {seq_len} to {max_seq} due to memory constraints")
            return input_tensor[:, :max_seq, :]
        
        return input_tensor


class AttentionPatternVisualizer:
    """
    Visualization utilities for attention patterns.
    
    Features:
    - Attention heatmap generation
    - Head comparison visualization
    - Temporal attention patterns
    - Cross-attention analysis
    """
    
    @staticmethod
    def extract_attention_patterns(
        attention_weights: torch.Tensor,
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention patterns for visualization.
        
        Args:
            attention_weights: Attention weights tensor or list of tensors
            layer_names: Names of attention layers
            
        Returns:
            Dictionary with attention patterns
        """
        patterns = {}
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = [attention_weights]
        
        if layer_names is None:
            layer_names = [f"layer_{i}" for i in range(len(attention_weights))]
        
        for i, (weights, name) in enumerate(zip(attention_weights, layer_names)):
            if weights is None:
                continue
            
            # Convert to numpy
            if isinstance(weights, torch.Tensor):
                weights_np = weights.detach().cpu().numpy()
            else:
                weights_np = weights
            
            # Average over batch dimension
            if weights_np.ndim == 4:  # [batch, heads, seq, seq]
                weights_np = weights_np.mean(axis=0)
            
            patterns[name] = weights_np
        
        return patterns
    
    @staticmethod
    def compute_attention_statistics(
        attention_patterns: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for attention patterns.
        
        Args:
            attention_patterns: Dictionary with attention patterns
            
        Returns:
            Dictionary with statistics for each layer
        """
        stats = {}
        
        for layer_name, pattern in attention_patterns.items():
            layer_stats = {}
            
            # Basic statistics
            layer_stats['mean'] = float(np.mean(pattern))
            layer_stats['std'] = float(np.std(pattern))
            layer_stats['max'] = float(np.max(pattern))
            layer_stats['min'] = float(np.min(pattern))
            
            # Sparsity
            threshold = 0.01
            sparsity = np.mean(pattern < threshold)
            layer_stats['sparsity'] = float(sparsity)
            
            # Entropy (if pattern is probability distribution)
            if pattern.ndim >= 2:
                # Compute entropy for each head
                entropies = []
                for head_idx in range(pattern.shape[0]):
                    head_pattern = pattern[head_idx]
                    # Normalize to probabilities
                    head_pattern_norm = head_pattern / (np.sum(head_pattern, axis=-1, keepdims=True) + 1e-8)
                    entropy = -np.sum(head_pattern_norm * np.log(head_pattern_norm + 1e-8), axis=-1)
                    entropies.append(np.mean(entropy))
                
                layer_stats['mean_entropy'] = float(np.mean(entropies))
                layer_stats['std_entropy'] = float(np.std(entropies))
            
            stats[layer_name] = layer_stats
        
        return stats


class AttentionDebugger:
    """
    Debugging utilities for attention mechanisms.
    
    Features:
    - NaN/Inf detection
    - Gradient flow analysis
    - Attention weight distribution analysis
    - Performance bottleneck detection
    """
    
    def __init__(self):
        self.debug_info = {}
        self.hooks = []
    
    def register_attention_hooks(self, model: nn.Module):
        """Register hooks for attention debugging."""
        
        def attention_hook(module, input, output):
            module_name = f"{module.__class__.__name__}_{id(module)}"
            
            # Check for NaN/Inf
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
            elif isinstance(output, (list, tuple)) and len(output) > 0:
                has_nan = any(torch.isnan(o).any().item() for o in output if isinstance(o, torch.Tensor))
                has_inf = any(torch.isinf(o).any().item() for o in output if isinstance(o, torch.Tensor))
            else:
                has_nan = False
                has_inf = False
            
            self.debug_info[module_name] = {
                'has_nan': has_nan,
                'has_inf': has_inf,
                'input_shape': input[0].shape if input and isinstance(input[0], torch.Tensor) else None,
                'output_shape': output.shape if isinstance(output, torch.Tensor) else None
            }
            
            if has_nan:
                logger.warning(f"NaN detected in {module_name}")
            if has_inf:
                logger.warning(f"Inf detected in {module_name}")
        
        # Register hooks for attention modules
        for name, module in model.named_modules():
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn']):
                hook = module.register_forward_hook(attention_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_debug_report(self) -> Dict[str, Any]:
        """Get comprehensive debug report."""
        report = {
            'num_modules_checked': len(self.debug_info),
            'modules_with_nan': sum(1 for info in self.debug_info.values() if info['has_nan']),
            'modules_with_inf': sum(1 for info in self.debug_info.values() if info['has_inf']),
            'module_details': self.debug_info
        }
        
        return report
    
    @staticmethod
    def validate_attention_weights(
        attention_weights: torch.Tensor,
        tolerance: float = 1e-6
    ) -> Dict[str, bool]:
        """
        Validate attention weights for common issues.
        
        Args:
            attention_weights: Attention weights tensor
            tolerance: Numerical tolerance
            
        Returns:
            Validation results dictionary
        """
        validation = {}
        
        with torch.no_grad():
            # Check for NaN/Inf
            validation['has_nan'] = torch.isnan(attention_weights).any().item()
            validation['has_inf'] = torch.isinf(attention_weights).any().item()
            
            # Check if weights sum to 1 (approximately)
            weight_sums = torch.sum(attention_weights, dim=-1)
            sum_diff = torch.abs(weight_sums - 1.0)
            validation['weights_sum_to_one'] = (sum_diff < tolerance).all().item()
            
            # Check for negative weights
            validation['has_negative_weights'] = (attention_weights < 0).any().item()
            
            # Check for extremely small or large weights
            validation['has_extreme_weights'] = (
                (attention_weights > 1.0).any().item() or 
                (attention_weights < 1e-8).any().item()
            )
            
            # Check weight distribution
            weight_std = torch.std(attention_weights).item()
            validation['weight_std'] = weight_std
            validation['is_uniform'] = weight_std < tolerance  # Very uniform distribution
            validation['is_peaked'] = weight_std > 0.5  # Very peaked distribution
        
        return validation


def benchmark_attention_implementations(
    implementations: Dict[str, nn.Module],
    input_tensor: torch.Tensor,
    num_runs: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different attention implementations.
    
    Args:
        implementations: Dictionary with attention implementations
        input_tensor: Test input tensor
        num_runs: Number of benchmark runs
        
    Returns:
        Benchmark results
    """
    analyzer = AttentionAnalyzer()
    results = {}
    
    for name, implementation in implementations.items():
        logger.info(f"Benchmarking {name}...")
        
        try:
            # Profile performance
            perf_stats = analyzer.profile_attention_layer(
                implementation, input_tensor, num_runs=num_runs
            )
            
            # Test forward pass for additional analysis
            with torch.no_grad():
                output = implementation(input_tensor)
                
            # Extract attention weights if available
            attention_weights = None
            if hasattr(implementation, 'get_attention_weights'):
                attention_weights = implementation.get_attention_weights()
            
            # Analyze attention patterns if weights available
            if attention_weights is not None:
                attention_stats = analyzer.analyze_attention_weights(attention_weights)
                perf_stats.update({
                    'entropy': attention_stats.entropy,
                    'sparsity': attention_stats.sparsity,
                    'head_diversity': attention_stats.head_diversity,
                    'memory_usage_mb': attention_stats.memory_usage_mb
                })
            
            results[name] = perf_stats
            
        except Exception as e:
            logger.error(f"Error benchmarking {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    # Test attention utilities
    print("Testing Attention Utilities:")
    
    # Create test attention weights
    batch_size, num_heads, seq_len = 2, 4, 64
    attention_weights = torch.softmax(
        torch.randn(batch_size, num_heads, seq_len, seq_len),
        dim=-1
    )
    
    # Test analyzer
    analyzer = AttentionAnalyzer()
    stats = analyzer.analyze_attention_weights(attention_weights)
    
    print(f"Attention Analysis:")
    print(f"  Entropy: {stats.entropy:.4f}")
    print(f"  Sparsity: {stats.sparsity:.4f}")
    print(f"  Max attention: {stats.max_attention:.4f}")
    print(f"  Head diversity: {stats.head_diversity:.4f}")
    print(f"  Temporal consistency: {stats.temporal_consistency:.4f}")
    print(f"  Memory usage: {stats.memory_usage_mb:.2f} MB")
    
    # Test memory optimizer
    print(f"\nTesting Memory Optimization:")
    
    # Create large tensors
    large_seq_len = 512
    d_model = 256
    head_dim = 32
    
    query = torch.randn(1, 8, large_seq_len, head_dim)
    key = torch.randn(1, 8, large_seq_len, head_dim)
    value = torch.randn(1, 8, large_seq_len, head_dim)
    
    # Standard attention
    start_time = time.time()
    standard_output = torch.matmul(
        torch.softmax(torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1),
        value
    )
    standard_time = (time.time() - start_time) * 1000
    
    # Chunked attention
    start_time = time.time()
    chunked_output = AttentionMemoryOptimizer.chunk_attention(
        query, key, value, chunk_size=128
    )
    chunked_time = (time.time() - start_time) * 1000
    
    print(f"  Standard attention: {standard_time:.2f}ms")
    print(f"  Chunked attention: {chunked_time:.2f}ms")
    print(f"  Output difference: {torch.abs(standard_output - chunked_output).max().item():.6f}")
    
    # Test pattern visualizer
    print(f"\nTesting Pattern Visualizer:")
    
    visualizer = AttentionPatternVisualizer()
    patterns = visualizer.extract_attention_patterns(attention_weights, ["test_layer"])
    pattern_stats = visualizer.compute_attention_statistics(patterns)
    
    for layer_name, stats in pattern_stats.items():
        print(f"  {layer_name}:")
        for stat_name, stat_value in stats.items():
            print(f"    {stat_name}: {stat_value:.4f}")
    
    # Test debugger
    print(f"\nTesting Attention Debugger:")
    
    debugger = AttentionDebugger()
    validation = debugger.validate_attention_weights(attention_weights)
    
    print(f"  Validation results:")
    for check, result in validation.items():
        if isinstance(result, bool):
            status = "✓" if result else "✗"
            print(f"    {check}: {status}")
        else:
            print(f"    {check}: {result:.6f}")
    
    # Test sequence truncation
    print(f"\nTesting Dynamic Sequence Truncation:")
    
    input_tensor = torch.randn(4, 2048, 512)
    truncated = AttentionMemoryOptimizer.dynamic_sequence_truncation(
        input_tensor, max_memory_mb=100.0
    )
    
    print(f"  Original shape: {input_tensor.shape}")
    print(f"  Truncated shape: {truncated.shape}")
    
    print(f"\n✅ Attention utilities testing complete!")