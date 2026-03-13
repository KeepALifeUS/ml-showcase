"""
Prototypical Networks Implementation
Prototype-Based Few-Shot Learning for Crypto Trading

Implementation Prototypical Networks for classification and regression in context
cryptocurrency trading strategies. Is based on studying prototypes classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
from collections import defaultdict

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class ProtoNetConfig:
    """Configuration for Prototypical Networks"""
    
    # Architecture
    embedding_dim: int = 128  # Dimensionality embedding space
    hidden_dims: List[int] = None  # Hidden layers encoder
    
    # Parameters training
    learning_rate: float = 0.001  # Speed training
    num_support: int = 5  # Examples on class in support set
    num_query: int = 15  # Examples on class in query set
    num_classes: int = 5  # Number classes in task
    
    # Distance and prototypes
    distance_metric: str = "euclidean"  # euclidean, cosine, manhattan
    temperature: float = 1.0  # Temperature for softmax
    prototype_aggregation: str = "mean"  # mean, median, weighted
    
    # Optimization
    weight_decay: float = 0.0001  # L2 regularization
    grad_clip: Optional[float] = 1.0  # Trimming gradients
    dropout_rate: float = 0.1  # Dropout for regularization
    
    # Regression (for prices cryptocurrencies)
    regression_mode: bool = False  # Mode regression instead classification
    regression_loss: str = "mse"  # mse, mae, huber
    
    # Monitoring
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class ProtoNetEncoder(nn.Module):
    """
    Encoder for Prototypical Networks
    
    Configurable Deep Encoder
    - Modular architecture
    - Dropout regularization
    - Batch normalization
    - Residual connections
    """
    
    def __init__(self, input_dim: int, config: ProtoNetConfig):
        super().__init__()
        
        self.config = config
        layers = []
        
        # Input layer
        current_dim = input_dim
        
        # Hidden layers
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output embedding layer
        layers.append(nn.Linear(current_dim, config.embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Initialization weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialization weights model"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder
        
        Args:
            x: Input data [batch_size, input_dim]
            
        Returns:
            Embedding vectors [batch_size, embedding_dim]
        """
        return self.encoder(x)


class PrototypicalNetworks:
    """
    Prototypical Networks for few-shot learning
    
    Prototype-Based Meta-Learning
    - Efficient prototype computation
    - Multiple distance metrics
    - Support for both classification and regression
    - Crypto market specialization
    """
    
    def __init__(
        self,
        input_dim: int,
        config: ProtoNetConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialization Prototypical Networks
        
        Args:
            input_dim: Dimensionality input data
            config: Configuration model
            logger: Logger for monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Encoder
        self.encoder = ProtoNetEncoder(input_dim, config).to(config.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=10
        )
        
        # Utilities
        self.gradient_manager = GradientManager()
        self.metrics = MetaLearningMetrics()
        
        # State
        self.global_step = 0
        self.best_accuracy = 0.0
        
        self.logger.info(f"ProtoNet initialized with config: {config}")
    
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes prototypes classes from support set
        
        Args:
            support_embeddings: Embeddings support examples [n_support, embedding_dim]
            support_labels: Labels support examples [n_support]
            
        Returns:
            Prototypes classes [n_classes, embedding_dim]
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        prototypes = torch.zeros(
            n_classes, 
            self.config.embedding_dim,
            device=self.config.device
        )
        
        for idx, label in enumerate(unique_labels):
            # Find all examples given class
            class_mask = (support_labels == label)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute prototype class
            if self.config.prototype_aggregation == "mean":
                prototype = class_embeddings.mean(dim=0)
            elif self.config.prototype_aggregation == "median":
                prototype = class_embeddings.median(dim=0)[0]
            elif self.config.prototype_aggregation == "weighted":
                # Weighted average (possible add weights)
                weights = torch.ones(len(class_embeddings)) / len(class_embeddings)
                weights = weights.to(self.config.device)
                prototype = (class_embeddings * weights.unsqueeze(1)).sum(dim=0)
            else:
                prototype = class_embeddings.mean(dim=0)
            
            prototypes[idx] = prototype
        
        return prototypes
    
    def compute_distances(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes distances between query embeddings and prototypes
        
        Args:
            query_embeddings: Query embeddings [n_query, embedding_dim]
            prototypes: Prototypes classes [n_classes, embedding_dim]
            
        Returns:
            Distances [n_query, n_classes]
        """
        if self.config.distance_metric == "euclidean":
            # Euclidean distance
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        elif self.config.distance_metric == "cosine":
            # Cosine distance (1 - cosine similarity)
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            cosine_sim = torch.mm(query_norm, proto_norm.t())
            distances = 1.0 - cosine_sim
        
        elif self.config.distance_metric == "manhattan":
            # Manhattan distance
            distances = torch.cdist(query_embeddings, prototypes, p=1)
        
        else:
            raise ValueError(f"Unknown distance metric: {self.config.distance_metric}")
        
        return distances
    
    def predict_classification(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predictions for classification
        
        Args:
            query_embeddings: Query embeddings
            prototypes: Prototypes classes
            
        Returns:
            Tuple from logits and probabilities
        """
        # Compute distances
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert distances in logits (than less distance, the more logit)
        logits = -distances / self.config.temperature
        
        # Compute probability
        probabilities = F.softmax(logits, dim=1)
        
        return logits, probabilities
    
    def predict_regression(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        support_targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Predictions for regression on basis nearest neighbors
        
        Args:
            query_embeddings: Query embeddings
            support_embeddings: Support embeddings
            support_targets: Support target values
            
        Returns:
            Predicted values
        """
        # Compute distances until all support examples
        distances = self.compute_distances(query_embeddings, support_embeddings)
        
        # Weights on basis reverse distances
        weights = 1.0 / (distances + 1e-8)  # Avoid division on zero
        weights = F.softmax(weights / self.config.temperature, dim=1)
        
        # Weighted average target values
        predictions = torch.mm(weights, support_targets.unsqueeze(1)).squeeze(1)
        
        return predictions
    
    def train_step(
        self,
        task_batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        One step training on batch tasks
        
        Args:
            task_batch: Batch tasks for training
            
        Returns:
            Dictionary with metrics
        """
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        total_accuracy = 0.0
        batch_size = len(task_batch)
        
        for task in task_batch:
            support_data = task['support_data'].to(self.config.device)
            support_labels = task['support_labels'].to(self.config.device)
            query_data = task['query_data'].to(self.config.device)
            query_labels = task['query_labels'].to(self.config.device)
            
            # Retrieve embeddings
            support_embeddings = self.encoder(support_data)
            query_embeddings = self.encoder(query_data)
            
            if self.config.regression_mode:
                # Mode regression
                predictions = self.predict_regression(
                    query_embeddings, support_embeddings, support_labels
                )
                
                # Loss for regression
                if self.config.regression_loss == "mse":
                    loss = F.mse_loss(predictions, query_labels)
                elif self.config.regression_loss == "mae":
                    loss = F.l1_loss(predictions, query_labels)
                elif self.config.regression_loss == "huber":
                    loss = F.smooth_l1_loss(predictions, query_labels)
                else:
                    loss = F.mse_loss(predictions, query_labels)
                
                # Accuracy for regression (in within threshold)
                with torch.no_grad():
                    threshold = 0.1 * torch.std(query_labels)
                    errors = torch.abs(predictions - query_labels)
                    accuracy = (errors < threshold).float().mean()
            
            else:
                # Mode classification
                prototypes = self.compute_prototypes(
                    support_embeddings, support_labels
                )
                
                logits, _ = self.predict_classification(
                    query_embeddings, prototypes
                )
                
                # Cross-entropy loss
                loss = F.cross_entropy(logits, query_labels.long())
                
                # Accuracy
                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_labels).float().mean()
            
            total_loss += loss
            total_accuracy += accuracy
        
        # Average by batch
        avg_loss = total_loss / batch_size
        avg_accuracy = total_accuracy / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.encoder.parameters(), self.config.grad_clip
            )
        
        # Optimization step
        self.optimizer.step()
        
        # Metrics
        metrics = {
            'loss': avg_loss.item(),
            'accuracy': avg_accuracy.item(),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                self.encoder.parameters()
            )
        }
        
        self.global_step += 1
        return metrics
    
    def validate(
        self,
        validation_tasks: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Validation model"""
        self.encoder.eval()
        
        all_losses = []
        all_accuracies = []
        
        with torch.no_grad():
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Retrieve embeddings
                support_embeddings = self.encoder(support_data)
                query_embeddings = self.encoder(query_data)
                
                if self.config.regression_mode:
                    # Regression
                    predictions = self.predict_regression(
                        query_embeddings, support_embeddings, support_labels
                    )
                    loss = F.mse_loss(predictions, query_labels)
                    
                    # Accuracy for regression
                    threshold = 0.1 * torch.std(query_labels)
                    errors = torch.abs(predictions - query_labels)
                    accuracy = (errors < threshold).float().mean()
                
                else:
                    # Classification
                    prototypes = self.compute_prototypes(
                        support_embeddings, support_labels
                    )
                    logits, _ = self.predict_classification(
                        query_embeddings, prototypes
                    )
                    loss = F.cross_entropy(logits, query_labels.long())
                    
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == query_labels).float().mean()
                
                all_losses.append(loss.item())
                all_accuracies.append(accuracy.item())
        
        self.encoder.train()
        
        return {
            'val_loss': np.mean(all_losses),
            'val_accuracy': np.mean(all_accuracies)
        }
    
    def few_shot_predict(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Few-shot prediction on new task
        
        Args:
            support_data: Support examples
            support_labels: Support labels
            query_data: Query examples
            
        Returns:
            Predictions and confidence scores
        """
        self.encoder.eval()
        
        with torch.no_grad():
            # Retrieve embeddings
            support_embeddings = self.encoder(support_data)
            query_embeddings = self.encoder(query_data)
            
            if self.config.regression_mode:
                # Regression
                predictions = self.predict_regression(
                    query_embeddings, support_embeddings, support_labels
                )
                # For regression confidence possible compute as reverse distance
                distances = self.compute_distances(query_embeddings, support_embeddings)
                confidence = 1.0 / (distances.min(dim=1)[0] + 1e-8)
            
            else:
                # Classification
                prototypes = self.compute_prototypes(
                    support_embeddings, support_labels
                )
                logits, probabilities = self.predict_classification(
                    query_embeddings, prototypes
                )
                predictions = torch.argmax(logits, dim=1)
                confidence = torch.max(probabilities, dim=1)[0]
        
        self.encoder.train()
        return predictions, confidence
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saving checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"ProtoNet checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loading checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_accuracy = checkpoint['best_accuracy']
        
        self.logger.info(f"ProtoNet checkpoint loaded from {filepath}")


class ProtoNetTrainer:
    """
    Trainer for Prototypical Networks with enterprise patterns
    
    Features:
    - Prototype visualization
    - Distance metric analysis
    - Embedding space monitoring
    """
    
    def __init__(
        self,
        protonet: PrototypicalNetworks,
        train_loader: Any,
        val_loader: Any,
        config: ProtoNetConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.protonet = protonet
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics_history = []
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """Main loop training"""
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self.protonet.validate(self.val_loader)
            
            # Learning rate scheduling
            self.protonet.scheduler.step(val_metrics['val_loss'])
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_metrics(epoch, epoch_metrics)
            
            # Checkpoint saving
            current_val_accuracy = val_metrics['val_accuracy']
            if current_val_accuracy > best_val_accuracy:
                best_val_accuracy = current_val_accuracy
                self.protonet.best_accuracy = best_val_accuracy
                self.protonet.save_checkpoint(f"{save_dir}/best_protonet.pt")
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
        
        for batch in tqdm(self.train_loader, desc="ProtoNet Training"):
            metrics = self.protonet.train_step(batch)
            epoch_metrics.append(metrics)
        
        return {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logging metrics"""
        self.logger.info(f"ProtoNet Epoch {epoch}:")
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