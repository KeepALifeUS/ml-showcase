"""
Matching Networks Implementation
Attention-Based Few-Shot Learning for Crypto Trading

Implementation Matching Networks with mechanism attention for fast adaptation
to new cryptocurrency trading pairs and strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from tqdm import tqdm
import math

from ..utils.gradient_utils import GradientManager
from ..utils.meta_utils import MetaLearningMetrics


@dataclass
class MatchingNetConfig:
    """Configuration for Matching Networks"""
    
    # Architecture
    embedding_dim: int = 128  # Dimensionality embedding space
    hidden_dims: List[int] = None  # Hidden layers encoder
    
    # Attention mechanism
    attention_type: str = "cosine"  # cosine, dot, mlp
    attention_heads: int = 8  # Number attention heads
    attention_dropout: float = 0.1  # Dropout for attention
    
    # LSTM for context encoding
    lstm_layers: int = 1  # Number LSTM layers
    lstm_bidirectional: bool = True  # Bidirectional LSTM
    
    # Parameters training
    learning_rate: float = 0.001  # Speed training
    num_support: int = 5  # Examples on class in support set
    num_query: int = 15  # Examples on class in query set
    num_classes: int = 5  # Number classes in task
    
    # FCE (Full Context Embeddings)
    use_fce: bool = True  # Use whether FCE
    fce_steps: int = 3  # Number steps FCE
    
    # Optimization
    weight_decay: float = 0.0001  # L2 regularization
    grad_clip: Optional[float] = 1.0  # Trimming gradients
    dropout_rate: float = 0.1  # Dropout for regularization
    
    # Temperature for softmax
    temperature: float = 1.0
    
    # Monitoring
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


class AttentionModule(nn.Module):
    """
    Attention mechanism for Matching Networks
    
    Multi-Head Attention
    - Scalable attention computation
    - Multiple attention types
    - Efficient memory usage
    """
    
    def __init__(self, config: MatchingNetConfig):
        super().__init__()
        
        self.config = config
        self.attention_type = config.attention_type
        self.embedding_dim = config.embedding_dim
        self.num_heads = config.attention_heads
        self.head_dim = config.embedding_dim // config.attention_heads
        
        if config.attention_type == "mlp":
            # MLP-based attention
            self.attention_mlp = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                nn.ReLU(),
                nn.Dropout(config.attention_dropout),
                nn.Linear(config.embedding_dim, 1)
            )
        
        elif config.attention_type == "dot" or config.attention_type == "cosine":
            # Multi-head attention
            self.query_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.key_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.value_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
            
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes attention between query and support examples
        
        Args:
            query: Query embeddings [n_query, embedding_dim]
            support: Support embeddings [n_support, embedding_dim]
            support_labels: Support labels [n_support]
            
        Returns:
            Attended features and attention weights
        """
        if self.attention_type == "cosine":
            return self._cosine_attention(query, support, support_labels)
        elif self.attention_type == "dot":
            return self._dot_attention(query, support, support_labels)
        elif self.attention_type == "mlp":
            return self._mlp_attention(query, support, support_labels)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
    
    def _cosine_attention(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cosine similarity attention"""
        # Normalize vectors
        query_norm = F.normalize(query, p=2, dim=1)  # [n_query, embedding_dim]
        support_norm = F.normalize(support, p=2, dim=1)  # [n_support, embedding_dim]
        
        # Compute cosine similarity
        attention_weights = torch.mm(query_norm, support_norm.t())  # [n_query, n_support]
        attention_weights = attention_weights / self.config.temperature
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted combination of support labels
        attended_features = torch.mm(attention_weights, support)  # [n_query, embedding_dim]
        
        return attended_features, attention_weights
    
    def _dot_attention(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head dot product attention"""
        batch_size = query.size(0)
        support_size = support.size(0)
        
        # Project to query, key, value
        Q = self.query_proj(query).view(batch_size, self.num_heads, self.head_dim)
        K = self.key_proj(support).view(support_size, self.num_heads, self.head_dim)
        V = self.value_proj(support).view(support_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.einsum('qhd,khd->qhk', Q, K) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=2)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.einsum('qhk,khd->qhd', attention_weights, V)
        attended = attended.view(batch_size, self.embedding_dim)
        
        # Output projection
        attended_features = self.output_proj(attended)
        
        # Average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1)
        
        return attended_features, avg_attention
    
    def _mlp_attention(
        self,
        query: torch.Tensor,
        support: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MLP-based attention"""
        n_query = query.size(0)
        n_support = support.size(0)
        
        # Expand for pairwise computation
        query_expanded = query.unsqueeze(1).expand(n_query, n_support, -1)
        support_expanded = support.unsqueeze(0).expand(n_query, -1, -1)
        
        # Concatenate query and support
        combined = torch.cat([query_expanded, support_expanded], dim=2)
        combined = combined.view(-1, self.embedding_dim * 2)
        
        # Compute attention scores
        attention_scores = self.attention_mlp(combined)
        attention_scores = attention_scores.view(n_query, n_support)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted combination
        attended_features = torch.bmm(
            attention_weights.unsqueeze(1),
            support_expanded
        ).squeeze(1)
        
        return attended_features, attention_weights


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for contextual encoding in Matching Networks
    
    Bidirectional Sequence Encoding
    - Captures sequential dependencies
    - Bidirectional processing
    - Proper initialization
    """
    
    def __init__(self, config: MatchingNetConfig):
        super().__init__()
        
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.embedding_dim // 2 if config.lstm_bidirectional else config.embedding_dim,
            num_layers=config.lstm_layers,
            bidirectional=config.lstm_bidirectional,
            dropout=config.dropout_rate if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, embedding_dim]
            
        Returns:
            Contextualized embeddings [batch_size, seq_len, embedding_dim]
        """
        lstm_out, _ = self.lstm(embeddings)
        return self.dropout(lstm_out)


class MatchingNetworkEncoder(nn.Module):
    """
    Main encoder for Matching Networks
    
    Deep Feature Encoder
    - Modular architecture
    - Proper regularization
    - Efficient computation
    """
    
    def __init__(self, input_dim: int, config: MatchingNetConfig):
        super().__init__()
        
        self.config = config
        
        # Feature encoder
        layers = []
        current_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, config.embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # LSTM for contextual encoding (if is used FCE)
        if config.use_fce:
            self.lstm_encoder = LSTMEncoder(config)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialization weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor, use_fce: bool = False) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            use_fce: Use whether Full Context Embeddings
            
        Returns:
            Embeddings [batch_size, embedding_dim] or [batch_size, seq_len, embedding_dim]
        """
        if x.dim() == 3 and use_fce:
            # FCE mode: process sequences
            batch_size, seq_len, input_dim = x.shape
            x_flat = x.view(-1, input_dim)
            embeddings_flat = self.encoder(x_flat)
            embeddings = embeddings_flat.view(batch_size, seq_len, self.config.embedding_dim)
            
            if self.config.use_fce:
                embeddings = self.lstm_encoder(embeddings)
            
            return embeddings
        else:
            # Regular mode
            return self.encoder(x)


class MatchingNetworks:
    """
    Matching Networks for few-shot learning
    
    Attention-Based Meta-Learning
    - Attention-based matching
    - Full Context Embeddings (FCE)
    - Memory-efficient processing
    - Crypto trading specialization
    """
    
    def __init__(
        self,
        input_dim: int,
        config: MatchingNetConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialization Matching Networks
        
        Args:
            input_dim: Dimensionality input data
            config: Configuration model
            logger: Logger for monitoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Encoder
        self.encoder = MatchingNetworkEncoder(input_dim, config).to(config.device)
        
        # Attention module
        self.attention = AttentionModule(config).to(config.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.attention.parameters()),
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
        
        self.logger.info(f"MatchingNetworks initialized with config: {config}")
    
    def full_context_embeddings(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Full Context Embeddings (FCE) for improvements support representations
        
        Args:
            support_data: Support examples [n_support, input_dim]
            support_labels: Support labels [n_support]
            
        Returns:
            Contextualized support embeddings [n_support, embedding_dim]
        """
        if not self.config.use_fce:
            return self.encoder(support_data)
        
        # Initial embeddings
        current_embeddings = self.encoder(support_data)
        
        # Iterative improvement through self-attention
        for step in range(self.config.fce_steps):
            # Self-attention over support set
            attended_features, _ = self.attention(
                current_embeddings, current_embeddings, support_labels
            )
            
            # Residual connection
            current_embeddings = current_embeddings + attended_features
            
            # Optional: add layer norm
            current_embeddings = F.layer_norm(
                current_embeddings, [self.config.embedding_dim]
            )
        
        return current_embeddings
    
    def predict(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction on basis attention-based matching
        
        Args:
            support_data: Support examples
            support_labels: Support labels
            query_data: Query examples
            
        Returns:
            Logits and attention weights
        """
        # Retrieve embeddings
        query_embeddings = self.encoder(query_data)
        support_embeddings = self.full_context_embeddings(support_data, support_labels)
        
        # Attention-based matching
        attended_features, attention_weights = self.attention(
            query_embeddings, support_embeddings, support_labels
        )
        
        # Compute probability classes through weighted voting
        num_classes = len(torch.unique(support_labels))
        n_query = query_embeddings.size(0)
        n_support = support_embeddings.size(0)
        
        # One-hot encoding support labels
        support_labels_one_hot = F.one_hot(
            support_labels.long(), num_classes
        ).float()  # [n_support, num_classes]
        
        # Weighted voting through attention
        logits = torch.mm(
            attention_weights, support_labels_one_hot
        )  # [n_query, num_classes]
        
        return logits, attention_weights
    
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
            
            # Predictions
            logits, attention_weights = self.predict(
                support_data, support_labels, query_data
            )
            
            # Loss
            loss = F.cross_entropy(logits, query_labels.long())
            total_loss += loss
            
            # Accuracy
            with torch.no_grad():
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean()
                total_accuracy += accuracy
        
        # Average by batch
        avg_loss = total_loss / batch_size
        avg_accuracy = total_accuracy / batch_size
        
        # Backward pass
        avg_loss.backward()
        
        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.attention.parameters()),
                self.config.grad_clip
            )
        
        # Optimization step
        self.optimizer.step()
        
        # Metrics
        metrics = {
            'loss': avg_loss.item(),
            'accuracy': avg_accuracy.item(),
            'gradient_norm': self.gradient_manager.compute_gradient_norm(
                list(self.encoder.parameters()) + list(self.attention.parameters())
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
        self.attention.eval()
        
        all_losses = []
        all_accuracies = []
        
        with torch.no_grad():
            for task in validation_tasks:
                support_data = task['support_data'].to(self.config.device)
                support_labels = task['support_labels'].to(self.config.device)
                query_data = task['query_data'].to(self.config.device)
                query_labels = task['query_labels'].to(self.config.device)
                
                # Predictions
                logits, _ = self.predict(
                    support_data, support_labels, query_data
                )
                
                # Metrics
                loss = F.cross_entropy(logits, query_labels.long())
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_labels).float().mean()
                
                all_losses.append(loss.item())
                all_accuracies.append(accuracy.item())
        
        self.encoder.train()
        self.attention.train()
        
        return {
            'val_loss': np.mean(all_losses),
            'val_accuracy': np.mean(all_accuracies)
        }
    
    def few_shot_predict(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        query_data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Few-shot prediction with attention weights
        
        Args:
            support_data: Support examples
            support_labels: Support labels
            query_data: Query examples
            
        Returns:
            Predictions, probabilities, and attention weights
        """
        self.encoder.eval()
        self.attention.eval()
        
        with torch.no_grad():
            logits, attention_weights = self.predict(
                support_data, support_labels, query_data
            )
            
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        self.encoder.train()
        self.attention.train()
        
        return predictions, probabilities, attention_weights
    
    def save_checkpoint(self, filepath: str) -> None:
        """Saving checkpoint"""
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'attention_state_dict': self.attention.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"MatchingNetworks checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Loading checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.attention.load_state_dict(checkpoint['attention_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_accuracy = checkpoint['best_accuracy']
        
        self.logger.info(f"MatchingNetworks checkpoint loaded from {filepath}")


class MatchingNetTrainer:
    """
    Trainer for Matching Networks with enterprise patterns
    
    Features:
    - Attention visualization
    - Support-query matching analysis
    - Memory-efficient training
    """
    
    def __init__(
        self,
        matching_net: MatchingNetworks,
        train_loader: Any,
        val_loader: Any,
        config: MatchingNetConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.matching_net = matching_net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        self.metrics_history = []
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "./checkpoints",
        attention_analysis_interval: int = 50
    ) -> Dict[str, List[float]]:
        """Main loop training"""
        
        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            val_metrics = self.matching_net.validate(self.val_loader)
            
            # Learning rate scheduling
            self.matching_net.scheduler.step(val_metrics['val_loss'])
            
            # Attention analysis
            if epoch % attention_analysis_interval == 0:
                self._analyze_attention_patterns(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.metrics_history.append(epoch_metrics)
            
            # Logging
            if epoch % self.config.log_interval == 0:
                self._log_metrics(epoch, epoch_metrics)
            
            # Checkpoint saving
            if val_metrics['val_accuracy'] > self.matching_net.best_accuracy:
                self.matching_net.best_accuracy = val_metrics['val_accuracy']
                self.matching_net.save_checkpoint(f"{save_dir}/best_matching_net.pt")
        
        return self._compile_metrics_history()
    
    def _train_epoch(self) -> Dict[str, float]:
        """Training one epoch"""
        epoch_metrics = []
        
        for batch in tqdm(self.train_loader, desc="MatchingNet Training"):
            metrics = self.matching_net.train_step(batch)
            epoch_metrics.append(metrics)
        
        return {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
    
    def _analyze_attention_patterns(self, epoch: int) -> None:
        """Analysis patterns attention"""
        # Possible add visualization attention patterns
        self.logger.info(f"Attention analysis at epoch {epoch} - placeholder for visualization")
    
    def _log_metrics(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Logging metrics"""
        self.logger.info(f"MatchingNet Epoch {epoch}:")
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