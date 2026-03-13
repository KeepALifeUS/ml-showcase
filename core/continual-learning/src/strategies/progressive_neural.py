"""
Progressive Neural Networks for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade implementation Progressive Neural Networks for prevention
catastrophic forgetting through addition of new columns for each task
with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from copy import deepcopy
import logging
from datetime import datetime
from collections import OrderedDict

from ..core.continual_learner import ContinualLearner, LearnerConfig, TaskMetadata
from ..core.memory_buffer import MemorySample, BaseMemoryBuffer, MemoryBufferFactory, SamplingStrategy


class LateralConnection(nn.Module):
    """
    Lateral connection between in Progressive Neural Network
    
    Ensures knowledge transfer from previous tasks to the new task
    """
    
    def __init__(self, prev_layer_size: int, current_layer_size: int, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Scaling factor for lateral connections
        self.adapter = nn.Linear(prev_layer_size, current_layer_size)
        self.activation = nn.ReLU()
        
        # Initialize weights
        nn.init.xavier_uniform_(self.adapter.weight)
        nn.init.zeros_(self.adapter.bias)
    
    def forward(self, prev_activations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through lateral connection
        
        Args:
            prev_activations: Activation from previous columns
            
        Returns:
             activation
        """
        adapted = self.adapter(prev_activations)
        return self.alpha * self.activation(adapted)


class ProgressiveColumn(nn.Module):
    """
    Column in Progressive Neural Network for one tasks
    
     column contains:
    - Main layers for current tasks
    - Lateral connections from previous columns
    - Specialized for different types predictions
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        task_id: int,
        prev_columns: Optional[List['ProgressiveColumn']] = None,
        market_regime: str = "unknown"
    ):
        super().__init__()
        self.task_id = task_id
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.market_regime = market_regime
        
        # Main layers columns
        self.layers = nn.ModuleList()
        
        # Input layer
        current_input = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(current_input, hidden_size))
            current_input = hidden_size
        
        # Output head
        self.output_head = nn.Linear(current_input, output_size)
        
        # Lateral connections from previous columns
        self.lateral_connections = nn.ModuleDict()
        if prev_columns:
            self._create_lateral_connections(prev_columns)
        
        # Activation functions
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Batch normalization for stability
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(size) for size in hidden_sizes
        ])
    
    def _create_lateral_connections(self, prev_columns: List['ProgressiveColumn']) -> None:
        """
        Create lateral connections from previous columns
        
        Args:
            prev_columns: List previous columns
        """
        for i, prev_column in enumerate(prev_columns):
            column_connections = nn.ModuleList()
            
            # Connections for each layers
            for layer_idx in range(len(self.hidden_sizes)):
                if layer_idx < len(prev_column.hidden_sizes):
                    prev_size = prev_column.hidden_sizes[layer_idx]
                    current_size = self.hidden_sizes[layer_idx]
                    
                    # Adaptive alpha on basis market regimes
                    alpha = self._compute_adaptive_alpha(
                        prev_column.market_regime, 
                        self.market_regime
                    )
                    
                    connection = LateralConnection(prev_size, current_size, alpha)
                    column_connections.append(connection)
                else:
                    column_connections.append(None)
            
            self.lateral_connections[f"column_{prev_column.task_id}"] = column_connections
    
    def _compute_adaptive_alpha(self, prev_regime: str, current_regime: str) -> float:
        """
        Computation adaptive alpha for lateral connections
        on basis similarity market regimes
        
        Args:
            prev_regime: Market regime previous tasks
            current_regime: Market regime current tasks
            
        Returns:
            Adaptive coefficient alpha
        """
        # Matrix similarity market regimes
        regime_similarity = {
            ("bull", "bull"): 1.0,
            ("bear", "bear"): 1.0,
            ("sideways", "sideways"): 1.0,
            ("volatile", "volatile"): 1.0,
            ("bull", "sideways"): 0.7,
            ("sideways", "bull"): 0.7,
            ("bear", "volatile"): 0.8,
            ("volatile", "bear"): 0.8,
            ("bull", "bear"): 0.3,
            ("bear", "bull"): 0.3,
        }
        
        similarity = regime_similarity.get((prev_regime, current_regime), 0.5)
        return similarity * 1.2  # Base scaling
    
    def forward(
        self, 
        x: torch.Tensor, 
        prev_activations: Optional[Dict[str, List[torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through column
        
        Args:
            x: Input data
            prev_activations: Activation from previous columns
            
        Returns:
            (output, current_activations)
        """
        current_activations = []
        current_input = x
        
        # Forward pass through layers with lateral connections
        for layer_idx, layer in enumerate(self.layers):
            # Main activation
            layer_output = layer(current_input)
            
            # Add lateral connections
            if prev_activations:
                for column_name, connections in self.lateral_connections.items():
                    if (column_name in prev_activations and 
                        layer_idx < len(connections) and 
                        connections[layer_idx] is not None and
                        layer_idx < len(prev_activations[column_name])):
                        
                        prev_activation = prev_activations[column_name][layer_idx]
                        lateral_contribution = connections[layer_idx](prev_activation)
                        layer_output = layer_output + lateral_contribution
            
            # Batch normalization
            if layer_idx < len(self.batch_norms):
                layer_output = self.batch_norms[layer_idx](layer_output)
            
            # Activation and dropout
            layer_output = self.activation(layer_output)
            layer_output = self.dropout(layer_output)
            
            current_activations.append(layer_output)
            current_input = layer_output
        
        # Output head
        output = self.output_head(current_input)
        
        return output, current_activations


class ProgressiveNetworkLearner(ContinualLearner):
    """
    Progressive Neural Network Learner for crypto trading
    
    Prevents catastrophic forgetting through addition of new columns
    for each tasks with lateral connections for knowledge.
    
    enterprise Features:
    - Market regime-aware column architecture
    - Adaptive lateral connections
    - Dynamic column pruning
    - Memory-efficient inference
    - Performance-based column selection
    """
    
    def __init__(self, model: nn.Module, config: LearnerConfig):
        super().__init__(model, config)
        
        # Progressive Network specific components
        self.columns: List[ProgressiveColumn] = []
        self.column_optimizers: List[optim.Optimizer] = []
        self.active_column_idx = -1
        
        # parameters
        self.hidden_sizes = [64, 32, 16] # Possible configure
        self.input_size = self._infer_input_size()
        self.output_size = self._infer_output_size()
        
        # enterprise settings
        self.adaptive_architecture = True  # Adaptive architecture columns
        self.performance_pruning = True # Remove inefficient columns
        self.memory_efficient_inference = True  # Optimized inference
        self.column_selection_strategy = "best_performance" # Strategy selection columns
        
        # Monitor performance by
        self.column_performances: Dict[int, Dict[str, float]] = {}
        self.lateral_connection_weights: Dict[str, List[float]] = {}
        self.column_usage_counts: Dict[int, int] = {}
        
        # Memory buffer for samples ( than in other strategies)
        self.memory_buffer = MemoryBufferFactory.create_crypto_trading_buffer(
            max_size=config.memory_budget // 2, # Fewer memory, more in architecture
            strategy=SamplingStrategy.RESERVOIR
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Adaptive pruning settings
        self.min_column_performance = 0.3 # Minimal performance for saving
        self.max_columns = 20 # Maximum columns for management complexity
        self.pruning_interval = 5  # Every N tasks check pruning
    
    def _infer_input_size(self) -> int:
        """Automatic determination size """
        # Attempt from model
        if hasattr(self.model, 'fc') or hasattr(self.model, 'linear'):
            first_layer = next(self.model.parameters())
            return first_layer.shape[1] if len(first_layer.shape) > 1 else first_layer.shape[0]
        
        return 50  # Value by default for crypto trading
    
    def _infer_output_size(self) -> int:
        """Automatic determination size output"""
        # For regression prices 1 output
        return 1
    
    def learn_task(self, task_data: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Learning new tasks through addition new columns
        
        Args:
            task_data: Data for training (features, targets, etc.)
            task_metadata: Metadata tasks
            
        Returns:
            Dict with metrics performance
        """
        self.logger.info(f"Starting Progressive Network learning for task {task_metadata.task_id}: {task_metadata.name}")
        
        # Preparation data
        features = torch.tensor(task_data["features"], dtype=torch.float32)
        targets = torch.tensor(task_data["targets"], dtype=torch.float32)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        # Adaptation architecture under market regime
        if self.adaptive_architecture:
            hidden_sizes = self._adapt_architecture_to_regime(task_metadata.market_regime)
        else:
            hidden_sizes = self.hidden_sizes
        
        # Create new columns for tasks
        new_column = self._create_new_column(task_metadata, hidden_sizes)
        self.columns.append(new_column)
        self.active_column_idx = len(self.columns) - 1
        
        # Create optimizer for new columns
        column_optimizer = optim.Adam(
            new_column.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_lambda
        )
        self.column_optimizers.append(column_optimizer)
        
        # Create DataLoader
        dataset = TensorDataset(features, targets)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training new columns
        metrics = self._train_new_column(dataloader, task_metadata)
        
        # Update history tasks
        self.task_history.append(task_metadata)
        self.current_task = task_metadata.task_id
        self.performance_history[task_metadata.task_id] = metrics
        self.column_performances[task_metadata.task_id] = metrics
        
        # Add samples in memory
        self._add_samples_to_memory(features, targets, task_metadata)
        
        # Pruning inefficient columns
        if self.performance_pruning and len(self.columns) >= self.pruning_interval:
            self._prune_inefficient_columns()
        
        # Checkpoint saving
        if self.config.enable_checkpointing:
            checkpoint_name = f"progressive_task_{task_metadata.task_id}_{task_metadata.market_regime}"
            self.save_checkpoint(checkpoint_name)
        
        self.logger.info(f"Completed Progressive Network learning for task {task_metadata.task_id}")
        return metrics
    
    def _adapt_architecture_to_regime(self, market_regime: str) -> List[int]:
        """
        Adaptation architecture columns under market regime
        
        Adaptive architecture
        
        Args:
            market_regime: Market regime
            
        Returns:
             sizes hidden layers
        """
        base_sizes = self.hidden_sizes.copy()
        
        # Adaptation under regime
        if market_regime == "volatile":
            # More neurons for complex patterns in volatility
            base_sizes = [int(size * 1.5) for size in base_sizes]
        elif market_regime == "sideways":
            # Fewer neurons for patterns
            base_sizes = [int(size * 0.8) for size in base_sizes]
        elif market_regime in ["bull", "bear"]:
            # Average complexity for
            base_sizes = [int(size * 1.1) for size in base_sizes]
        
        self.logger.info(f"Adapted architecture for {market_regime}: {base_sizes}")
        return base_sizes
    
    def _create_new_column(self, task_metadata: TaskMetadata, hidden_sizes: List[int]) -> ProgressiveColumn:
        """
        Create new columns for tasks
        
        Args:
            task_metadata: Metadata tasks
            hidden_sizes: Sizes hidden layers
            
        Returns:
            New column
        """
        prev_columns = self.columns.copy() if self.columns else None
        
        new_column = ProgressiveColumn(
            input_size=self.input_size,
            hidden_sizes=hidden_sizes,
            output_size=self.output_size,
            task_id=task_metadata.task_id,
            prev_columns=prev_columns,
            market_regime=task_metadata.market_regime
        ).to(self.config.device)
        
        self.logger.info(f"Created new column for task {task_metadata.task_id} with architecture: {hidden_sizes}")
        return new_column
    
    def _train_new_column(self, dataloader: DataLoader, task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Training new columns
        
        Args:
            dataloader: DataLoader with training data
            task_metadata: Metadata tasks
            
        Returns:
            Metrics training
        """
        current_column = self.columns[self.active_column_idx]
        current_optimizer = self.column_optimizers[self.active_column_idx]
        
        current_column.train()
        
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # Training cycle
        num_epochs = 15 # More epochs for best learning new columns
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_targets in dataloader:
                batch_features = batch_features.to(self.config.device)
                batch_targets = batch_targets.to(self.config.device)
                
                current_optimizer.zero_grad()
                
                # Get activations from previous columns
                prev_activations = self._get_previous_activations(batch_features)
                
                # Forward pass through current column
                predictions, _ = current_column(batch_features, prev_activations)
                
                # Loss calculation
                loss = self.criterion(predictions, batch_targets)
                
                # Backward pass only for current columns
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(current_column.parameters(), max_norm=1.0)
                
                current_optimizer.step()
                
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
        
        # Count use columns
        task_id = task_metadata.task_id
        if task_id not in self.column_usage_counts:
            self.column_usage_counts[task_id] = 0
        self.column_usage_counts[task_id] += num_batches
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "column_id": self.active_column_idx,
            "num_columns": len(self.columns),
            "architecture": current_column.hidden_sizes,
            "lateral_connections": len(current_column.lateral_connections),
            "num_batches": num_batches,
            "predictions_made": total_predictions
        }
        
        return metrics
    
    def _get_previous_activations(self, inputs: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Get activations from all previous columns
        
        Args:
            inputs: Input data
            
        Returns:
            Dict with activations from each previous columns
        """
        if not self.columns or len(self.columns) <= 1:
            return {}
        
        prev_activations = {}
        
        # Get activations from all previous columns (except current)
        for i, column in enumerate(self.columns[:-1]):
            column.eval()  # Regime evaluation for previous columns
            
            with torch.no_grad():
                # Get activations previous columns for this columns
                column_prev_activations = {}
                for j in range(i):
                    prev_column = self.columns[j]
                    _, activations = prev_column(inputs, column_prev_activations)
                    column_prev_activations[f"column_{prev_column.task_id}"] = activations
                
                # Forward pass through current column
                _, activations = column(inputs, column_prev_activations)
                prev_activations[f"column_{column.task_id}"] = activations
        
        return prev_activations
    
    def _prune_inefficient_columns(self) -> None:
        """
        Remove inefficient columns for management complexity
        
        Performance-driven pruning
        """
        if len(self.columns) <= 2:  # Minimum 2 columns
            return
        
        # Analysis performance columns
        column_scores = []
        
        for task_id, performance in self.column_performances.items():
            accuracy = performance.get("accuracy", 0.0)
            usage_count = self.column_usage_counts.get(task_id, 0)
            
            # Composite score: performance + frequency use
            score = accuracy * 0.7 + min(usage_count / 1000, 1.0) * 0.3
            column_scores.append((task_id, score))
        
        # Sort by descending quality
        column_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Remove columns
        columns_to_remove = []
        for task_id, score in column_scores:
            if (score < self.min_column_performance and 
                len(self.columns) > len(columns_to_remove) + 2):
                columns_to_remove.append(task_id)
        
        # removal
        removed_count = 0
        for task_id in columns_to_remove[:3]: # Maximum 3 for time
            column_idx = next((i for i, col in enumerate(self.columns) if col.task_id == task_id), None)
            if column_idx is not None:
                removed_column = self.columns.pop(column_idx)
                removed_optimizer = self.column_optimizers.pop(column_idx)
                
                # Cleanup related data
                if task_id in self.column_performances:
                    del self.column_performances[task_id]
                if task_id in self.column_usage_counts:
                    del self.column_usage_counts[task_id]
                
                removed_count += 1
                self.logger.info(f"Pruned inefficient column for task {task_id} (score: {column_scores[task_id][1]:.3f})")
        
        if removed_count > 0:
            # Update index columns
            self.active_column_idx = len(self.columns) - 1
            self.logger.info(f"Pruned {removed_count} inefficient columns. Total columns: {len(self.columns)}")
    
    def _add_samples_to_memory(self, features: torch.Tensor, targets: torch.Tensor, task_metadata: TaskMetadata) -> None:
        """
        Add samples in buffer memory
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
        """
        # Fewer samples than in other strategies (architecture )
        num_samples = min(features.size(0) // 4, 25)
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
                quality_score=0.7
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
        # Search corresponding columns
        target_column = next((col for col in self.columns if col.task_id == task_id), None)
        
        if target_column is None:
            self.logger.warning(f"No column found for task {task_id}")
            return {"error": "column_not_found", "task_id": task_id}
        
        target_column.eval()
        
        features = torch.tensor(test_data["features"], dtype=torch.float32).to(self.config.device)
        targets = torch.tensor(test_data["targets"], dtype=torch.float32).to(self.config.device)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        
        with torch.no_grad():
            # Get activations from previous columns
            prev_activations = self._get_activations_up_to_column(features, task_id)
            
            # Prediction through target column
            predictions, _ = target_column(features, prev_activations)
            loss = self.criterion(predictions, targets)
            
            # Accuracy
            pred_direction = (predictions > 0).float()
            target_direction = (targets > 0).float()
            accuracy = (pred_direction == target_direction).float().mean().item()
            
            # Additional metrics
            mae = torch.abs(predictions - targets).mean().item()
            rmse = torch.sqrt(((predictions - targets) ** 2).mean()).item()
        
        metrics = {
            "task_id": task_id,
            "test_loss": loss.item(),
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "column_used": target_column.task_id,
            "num_samples": features.size(0)
        }
        
        self.logger.info(f"Task {task_id} evaluation - Accuracy: {accuracy:.3f}, Loss: {loss.item():.4f}")
        return metrics
    
    def _get_activations_up_to_column(self, inputs: torch.Tensor, target_task_id: int) -> Dict[str, List[torch.Tensor]]:
        """
        Get activations all columns up to target columns
        
        Args:
            inputs: Input data
            target_task_id: ID target tasks
            
        Returns:
            Dict with activations
        """
        target_column_idx = next((i for i, col in enumerate(self.columns) if col.task_id == target_task_id), -1)
        
        if target_column_idx <= 0:
            return {}
        
        activations = {}
        
        for i in range(target_column_idx):
            column = self.columns[i]
            column.eval()
            
            # Get previous activations for this columns
            column_prev_activations = {}
            for j in range(i):
                prev_column = self.columns[j]
                if f"column_{prev_column.task_id}" in activations:
                    column_prev_activations[f"column_{prev_column.task_id}"] = activations[f"column_{prev_column.task_id}"]
            
            with torch.no_grad():
                _, column_activations = column(inputs, column_prev_activations)
                activations[f"column_{column.task_id}"] = column_activations
        
        return activations
    
    def get_progressive_statistics(self) -> Dict[str, Any]:
        """
        Get statistics Progressive Network training
        
        Returns:
            Dict with statistics Progressive Network
        """
        stats = {
            "strategy": "progressive_neural_networks",
            "num_columns": len(self.columns),
            "active_column": self.active_column_idx,
            "max_columns": self.max_columns,
            "adaptive_architecture_enabled": self.adaptive_architecture,
            "performance_pruning_enabled": self.performance_pruning,
            "min_column_performance": self.min_column_performance,
            "column_usage_counts": dict(self.column_usage_counts)
        }
        
        if self.columns:
            # statistics
            architectures = [col.hidden_sizes for col in self.columns]
            total_parameters = sum(sum(p.numel() for p in col.parameters()) for col in self.columns)
            
            stats.update({
                "column_architectures": architectures,
                "total_parameters": total_parameters,
                "avg_parameters_per_column": total_parameters / len(self.columns)
            })
            
            # Lateral connections statistics
            lateral_stats = {}
            for col in self.columns:
                lateral_stats[f"column_{col.task_id}"] = len(col.lateral_connections)
            
            stats["lateral_connections"] = lateral_stats
        
        # Performance statistics
        if self.column_performances:
            accuracies = [perf.get("accuracy", 0) for perf in self.column_performances.values()]
            stats.update({
                "avg_column_accuracy": np.mean(accuracies),
                "best_column_accuracy": np.max(accuracies),
                "worst_column_accuracy": np.min(accuracies),
                "column_performance_variance": np.var(accuracies)
            })
        
        # Memory buffer statistics
        if self.memory_buffer:
            memory_stats = self.memory_buffer.get_memory_statistics()
            stats["memory_buffer"] = memory_stats
        
        return stats
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Analysis complexity model
        
        Returns:
            Metrics complexity model
        """
        if not self.columns:
            return {"total_parameters": 0, "total_columns": 0}
        
        complexity = {
            "total_columns": len(self.columns),
            "total_parameters": 0,
            "parameters_per_column": {},
            "lateral_connections_count": 0,
            "memory_usage_mb": 0
        }
        
        for i, column in enumerate(self.columns):
            param_count = sum(p.numel() for p in column.parameters())
            complexity["total_parameters"] += param_count
            complexity["parameters_per_column"][f"column_{column.task_id}"] = param_count
            
            # Count lateral connections
            for connections in column.lateral_connections.values():
                complexity["lateral_connections_count"] += len([c for c in connections if c is not None])
        
        # Evaluate use memory (approximate)
        complexity["memory_usage_mb"] = complexity["total_parameters"] * 4 / (1024 * 1024)  # 4 bytes per float32
        
        return complexity
    
    def __repr__(self) -> str:
        return (
            f"ProgressiveNetworkLearner("
            f"columns={len(self.columns)}, "
            f"active_column={self.active_column_idx}, "
            f"total_params={sum(sum(p.numel() for p in col.parameters()) for col in self.columns)})"
        )