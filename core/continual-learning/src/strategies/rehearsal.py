"""
Experience Replay (Rehearsal) for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade implementation Experience Replay strategies for prevention
catastrophic forgetting through on samples
with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import numpy as np
from datetime import datetime
import logging
import random

from ..core.continual_learner import ContinualLearner, LearnerConfig, TaskMetadata
from ..core.memory_buffer import (
    MemorySample, BaseMemoryBuffer, MemoryBufferFactory, 
    SamplingStrategy, SelectionCriteria
)


class RehearsalLearner(ContinualLearner):
    """
    Experience Replay (Rehearsal) Learner for crypto trading
    
    Rehearsal prevents catastrophic forgetting through storage
    and retraining on samples from previous tasks.
    
    enterprise Features:
    - Intelligent sample selection
    - Market regime balanced replay
    - Adaptive replay frequency
    - Memory-efficient storage
    - Performance-driven sample management
    """
    
    def __init__(self, model: nn.Module, config: LearnerConfig):
        super().__init__(model, config)
        
        # Rehearsal specific components
        self.replay_ratio = 0.5 # Fraction replay samples in batch
        self.replay_frequency = 1 # As often replay (each N batches)
        self.min_samples_per_task = 10  # Minimum samples on task
        
        # enterprise settings
        self.intelligent_selection = True # selection samples
        self.market_regime_balancing = True # Balancing by market regimes
        self.adaptive_replay = True # Adaptive frequency replay
        self.performance_driven_selection = True # Select on basis performance
        
        # Memory for samples with
        self.memory_buffer = MemoryBufferFactory.create_crypto_trading_buffer(
            max_size=config.memory_budget,
            strategy=SamplingStrategy.K_CENTER,  # More diverse samples
            config_overrides={
                "selection_criteria": [
                    SelectionCriteria.UNCERTAINTY,
                    SelectionCriteria.DIVERSITY,
                    SelectionCriteria.MARKET_REGIME,
                    SelectionCriteria.GRADIENT_MAGNITUDE
                ],
                "market_regime_balance": True,
                "asset_diversity": True
            }
        )
        
        # Optimizer settings
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.l2_lambda
        )
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.replay_criterion = nn.MSELoss() # Can be
        
        # Monitor 
        self.replay_losses: List[float] = []
        self.replay_accuracies: List[float] = []
        self.sample_selection_times: List[float] = []
        self.task_replay_counts: Dict[int, int] = {}
        self.performance_based_selections: int = 0
        
        # Adaptive configuration replay
        self.recent_performances: List[float] = []
        self.performance_threshold = 0.1 # Threshold degradation for replay
    
    def learn_task(self, task_data: Dict[str, Any], task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Learning new tasks with Experience Replay
        
        Args:
            task_data: Data for training (features, targets, etc.)
            task_metadata: Metadata tasks
            
        Returns:
            Dict with metrics performance
        """
        self.logger.info(f"Starting Rehearsal learning for task {task_metadata.task_id}: {task_metadata.name}")
        
        # Preparation data new tasks
        new_features = torch.tensor(task_data["features"], dtype=torch.float32)
        new_targets = torch.tensor(task_data["targets"], dtype=torch.float32)
        
        if new_features.dim() == 1:
            new_features = new_features.unsqueeze(0)
        if new_targets.dim() == 1:
            new_targets = new_targets.unsqueeze(0)
        
        # Adaptive configuration replay on basis performance
        if self.adaptive_replay:
            self._adapt_replay_parameters(task_metadata)
        
        # Create DataLoader for new tasks
        new_dataset = TensorDataset(new_features, new_targets)
        new_dataloader = DataLoader(
            new_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Training with replay
        metrics = self._train_with_replay(new_dataloader, task_metadata)
        
        # Add best samples new tasks in memory
        self._add_intelligent_samples_to_memory(new_features, new_targets, task_metadata, metrics)
        
        # Update history tasks
        self.task_history.append(task_metadata)
        self.current_task = task_metadata.task_id
        self.performance_history[task_metadata.task_id] = metrics
        
        # Update list performance for adaptation
        self.recent_performances.append(metrics.get("accuracy", 0.0))
        if len(self.recent_performances) > 10: # 10 tasks
            self.recent_performances.pop(0)
        
        # Checkpoint saving
        if self.config.enable_checkpointing:
            checkpoint_name = f"rehearsal_task_{task_metadata.task_id}_{task_metadata.market_regime}"
            self.save_checkpoint(checkpoint_name)
        
        self.logger.info(f"Completed Rehearsal learning for task {task_metadata.task_id}")
        return metrics
    
    def _adapt_replay_parameters(self, task_metadata: TaskMetadata) -> None:
        """
        Adaptation parameters replay on basis performance
        
        Adaptive replay for improvements performance
        
        Args:
            task_metadata: Metadata current tasks
        """
        if len(self.recent_performances) < 2:
            return
        
        # Check degradation performance
        recent_avg = np.mean(self.recent_performances[-3:]) if len(self.recent_performances) >= 3 else self.recent_performances[-1]
        earlier_avg = np.mean(self.recent_performances[:-3]) if len(self.recent_performances) > 3 else self.recent_performances[0]
        
        performance_drop = earlier_avg - recent_avg
        
        if performance_drop > self.performance_threshold:
            # Increasing replay at degradation
            old_ratio = self.replay_ratio
            self.replay_ratio = min(0.8, self.replay_ratio * 1.2)
            self.replay_frequency = max(1, self.replay_frequency - 1)
            
            self.logger.info(
                f"Performance degradation detected ({performance_drop:.3f}). "
                f"Increased replay ratio: {old_ratio:.2f} -> {self.replay_ratio:.2f}"
            )
        elif performance_drop < -0.05: # Performance
            # Decreasing replay at
            old_ratio = self.replay_ratio
            self.replay_ratio = max(0.2, self.replay_ratio * 0.9)
            
            self.logger.info(
                f"Performance improvement detected. "
                f"Decreased replay ratio: {old_ratio:.2f} -> {self.replay_ratio:.2f}"
            )
        
        # Adaptation on basis market regime
        if task_metadata.market_regime == "volatile":
            self.replay_ratio = min(0.7, self.replay_ratio * 1.1) # More replay in volatility
        elif task_metadata.market_regime == "bull":
            self.replay_ratio = max(0.3, self.replay_ratio * 0.95) # Fewer replay in bull market
    
    def _train_with_replay(self, new_dataloader: DataLoader, task_metadata: TaskMetadata) -> Dict[str, float]:
        """
        Training model with Experience Replay
        
        Args:
            new_dataloader: DataLoader with data new tasks
            task_metadata: Metadata tasks
            
        Returns:
            Metrics training
        """
        self.model.train()
        total_loss = 0.0
        new_task_loss = 0.0
        replay_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        # replay
        replay_batches = 0
        total_replay_samples = 0
        
        # Training cycle
        num_epochs = 10 # Possible make configurable
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, (batch_features, batch_targets) in enumerate(new_dataloader):
                batch_features = batch_features.to(self.config.device)
                batch_targets = batch_targets.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass for new tasks
                new_predictions = self.model(batch_features)
                current_new_loss = self.criterion(new_predictions, batch_targets)
                
                total_batch_loss = current_new_loss
                current_replay_loss = torch.tensor(0.0)
                
                # Experience Replay
                if (batch_idx + 1) % self.replay_frequency == 0 and len(self.memory_buffer) > 0:
                    replay_batch_size = int(batch_features.size(0) * self.replay_ratio)
                    
                    if replay_batch_size > 0:
                        current_replay_loss = self._perform_replay(replay_batch_size, task_metadata.task_id)
                        total_batch_loss = current_new_loss + current_replay_loss
                        replay_batches += 1
                        total_replay_samples += replay_batch_size
                
                # Backward pass
                total_batch_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_batch_loss.item()
                new_task_loss += current_new_loss.item()
                replay_loss += current_replay_loss.item()
                num_batches += 1
                epoch_batches += 1
                
                # Accuracy calculation
                if new_predictions.dim() == batch_targets.dim():
                    pred_direction = (new_predictions > 0).float()
                    target_direction = (batch_targets > 0).float()
                    correct_predictions += (pred_direction == target_direction).sum().item()
                    total_predictions += batch_targets.numel()
            
            avg_epoch_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0.0
            self.logger.debug(f"Epoch {epoch + 1}, Avg Loss: {avg_epoch_loss:.4f}")
        
        # Computation final metrics
        avg_total_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_new_loss = new_task_loss / num_batches if num_batches > 0 else 0.0
        avg_replay_loss = replay_loss / replay_batches if replay_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            "total_loss": avg_total_loss,
            "new_task_loss": avg_new_loss,
            "replay_loss": avg_replay_loss,
            "accuracy": accuracy,
            "replay_ratio": self.replay_ratio,
            "replay_frequency": self.replay_frequency,
            "replay_batches": replay_batches,
            "total_replay_samples": total_replay_samples,
            "num_batches": num_batches,
            "predictions_made": total_predictions
        }
        
        # Save replay metrics for monitoring
        self.replay_losses.append(avg_replay_loss)
        self.replay_accuracies.append(accuracy)
        
        # Update replay by tasks
        if task_metadata.task_id not in self.task_replay_counts:
            self.task_replay_counts[task_metadata.task_id] = 0
        self.task_replay_counts[task_metadata.task_id] += replay_batches
        
        return metrics
    
    def _perform_replay(self, replay_batch_size: int, current_task_id: int) -> torch.Tensor:
        """
        Execute Experience Replay on samples from memory
        
        Args:
            replay_batch_size: Size batch for replay
            current_task_id: ID current tasks
            
        Returns:
            Replay loss
        """
        # Excluding current task from replay
        exclude_tasks = {current_task_id}
        
        # Get samples from memory
        replay_samples = self.memory_buffer.sample_batch(
            replay_batch_size,
            exclude_task_ids=exclude_tasks
        )
        
        if not replay_samples:
            return torch.tensor(0.0, device=self.config.device)
        
        # Preparation data replay
        replay_features = torch.stack([sample.features for sample in replay_samples])
        replay_targets = torch.stack([sample.target for sample in replay_samples])
        
        replay_features = replay_features.to(self.config.device)
        replay_targets = replay_targets.to(self.config.device)
        
        # Forward pass for replay
        replay_predictions = self.model(replay_features)
        replay_loss = self.replay_criterion(replay_predictions, replay_targets)
        
        return replay_loss
    
    def _add_intelligent_samples_to_memory(
        self, 
        features: torch.Tensor, 
        targets: torch.Tensor, 
        task_metadata: TaskMetadata,
        training_metrics: Dict[str, float]
    ) -> None:
        """
        Intelligent sample addition to memory based on importance
        
        Performance-driven sample selection
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
            training_metrics: Metrics training for determining
        """
        start_time = datetime.now()
        
        # Determine number samples for saving
        max_samples_from_task = min(
            features.size(0) // 2, # Maximum from all samples
            self.config.memory_budget // (len(self.task_history) + 1), #
            100 # maximum
        )
        max_samples_from_task = max(max_samples_from_task, self.min_samples_per_task)
        
        if self.intelligent_selection:
            selected_samples = self._select_important_samples(
                features, targets, task_metadata, max_samples_from_task, training_metrics
            )
        else:
            # Random selection
            indices = torch.randperm(features.size(0))[:max_samples_from_task]
            selected_samples = self._create_memory_samples(
                features[indices], targets[indices], task_metadata
            )
        
        # Add in memory
        self.memory_buffer.add_samples(selected_samples)
        
        # Monitor time selection
        selection_time = (datetime.now() - start_time).total_seconds()
        self.sample_selection_times.append(selection_time)
        
        self.logger.info(
            f"Added {len(selected_samples)} intelligent samples from task {task_metadata.task_id} "
            f"in {selection_time:.2f}s"
        )
    
    def _select_important_samples(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        task_metadata: TaskMetadata,
        num_samples: int,
        training_metrics: Dict[str, float]
    ) -> List[MemorySample]:
        """
        Select samples on basis various
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
            num_samples: Number samples for selection
            training_metrics: Metrics training
            
        Returns:
            List selected samples
        """
        self.model.eval()
        
        sample_scores = []
        
        with torch.no_grad():
            for i in range(features.size(0)):
                sample_feature = features[i:i+1].to(self.config.device)
                sample_target = targets[i:i+1].to(self.config.device)
                
                # Prediction uncertainty
                prediction = self.model(sample_feature)
                prediction_confidence = torch.sigmoid(torch.abs(prediction)).item()
                uncertainty_score = 1.0 - prediction_confidence
                
                # Loss magnitude
                loss = self.criterion(prediction, sample_target).item()
                
                # Diversity score ( up to )
                feature_np = sample_feature.cpu().numpy().flatten()
                feature_mean = features.mean(dim=0).cpu().numpy().flatten()
                diversity_score = np.linalg.norm(feature_np - feature_mean)
                
                # Market volatility (if available)
                volatility_score = self._compute_market_volatility_score(task_metadata, i)
                
                # Composite importance score
                importance_score = (
                    0.3 * uncertainty_score +
                    0.3 * min(loss, 10.0) / 10.0 +  # Normalization loss
                    0.2 * min(diversity_score, 5.0) / 5.0 +  # Normalization diversity
                    0.2 * volatility_score
                )
                
                sample_scores.append((i, importance_score, uncertainty_score, loss, diversity_score))
        
        # Sort by
        sample_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top samples
        selected_indices = [score[0] for score in sample_scores[:num_samples]]
        
        # Create MemorySample objects
        selected_samples = []
        for idx in selected_indices:
            _, importance, uncertainty, loss, diversity = sample_scores[idx]
            
            sample = MemorySample(
                features=features[idx],
                target=targets[idx],
                task_id=task_metadata.task_id,
                timestamp=task_metadata.start_time,
                market_regime=task_metadata.market_regime,
                asset=task_metadata.assets[0] if task_metadata.assets else "UNKNOWN",
                timeframe=task_metadata.timeframe,
                uncertainty_score=float(uncertainty),
                gradient_magnitude=0.0, # Possible improve
                loss_value=float(loss),
                diversity_score=float(diversity),
                prediction_confidence=1.0 - float(uncertainty),
                quality_score=float(importance)
            )
            selected_samples.append(sample)
        
        self.performance_based_selections += 1
        return selected_samples
    
    def _compute_market_volatility_score(self, task_metadata: TaskMetadata, sample_idx: int) -> float:
        """
        Computation evaluation volatility for samples
        
        Args:
            task_metadata: Metadata tasks
            sample_idx: Index samples
            
        Returns:
            Evaluate volatility (0-1)
        """
        # Simple evaluation on basis market regime
        volatility_scores = {
            "volatile": 0.9,
            "bear": 0.6,
            "bull": 0.4,
            "sideways": 0.3
        }
        
        return volatility_scores.get(task_metadata.market_regime, 0.5)
    
    def _create_memory_samples(
        self, 
        features: torch.Tensor, 
        targets: torch.Tensor, 
        task_metadata: TaskMetadata
    ) -> List[MemorySample]:
        """
        Create MemorySample objects
        
        Args:
            features: Input features
            targets: Target values
            task_metadata: Metadata tasks
            
        Returns:
            List MemorySample objects
        """
        samples = []
        
        for i in range(features.size(0)):
            sample = MemorySample(
                features=features[i],
                target=targets[i],
                task_id=task_metadata.task_id,
                timestamp=task_metadata.start_time,
                market_regime=task_metadata.market_regime,
                asset=task_metadata.assets[0] if task_metadata.assets else "UNKNOWN",
                timeframe=task_metadata.timeframe,
                uncertainty_score=0.5,
                gradient_magnitude=0.0,
                loss_value=0.0,
                diversity_score=0.5,
                quality_score=0.7
            )
            samples.append(sample)
        
        return samples
    
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
            
            # Confidence metrics
            prediction_confidence = torch.sigmoid(torch.abs(predictions)).mean().item()
        
        metrics = {
            "task_id": task_id,
            "test_loss": loss.item(),
            "accuracy": accuracy,
            "mae": mae,
            "rmse": rmse,
            "prediction_confidence": prediction_confidence,
            "num_samples": features.size(0)
        }
        
        self.logger.info(
            f"Task {task_id} evaluation - Accuracy: {accuracy:.3f}, "
            f"Loss: {loss.item():.4f}, Confidence: {prediction_confidence:.3f}"
        )
        return metrics
    
    def get_rehearsal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics Rehearsal training
        
        Returns:
            Dict with statistics Rehearsal
        """
        stats = {
            "strategy": "rehearsal",
            "replay_ratio": self.replay_ratio,
            "replay_frequency": self.replay_frequency,
            "min_samples_per_task": self.min_samples_per_task,
            "intelligent_selection_enabled": self.intelligent_selection,
            "market_regime_balancing_enabled": self.market_regime_balancing,
            "adaptive_replay_enabled": self.adaptive_replay,
            "performance_based_selections": self.performance_based_selections,
            "task_replay_counts": dict(self.task_replay_counts)
        }
        
        if self.replay_losses:
            stats.update({
                "avg_replay_loss": np.mean(self.replay_losses),
                "min_replay_loss": np.min(self.replay_losses),
                "max_replay_loss": np.max(self.replay_losses),
                "replay_loss_trend": "increasing" if len(self.replay_losses) > 1 and self.replay_losses[-1] > self.replay_losses[0] else "decreasing"
            })
        
        if self.replay_accuracies:
            stats.update({
                "avg_replay_accuracy": np.mean(self.replay_accuracies),
                "min_replay_accuracy": np.min(self.replay_accuracies),
                "max_replay_accuracy": np.max(self.replay_accuracies)
            })
        
        if self.sample_selection_times:
            stats.update({
                "avg_selection_time": np.mean(self.sample_selection_times),
                "total_selection_time": sum(self.sample_selection_times)
            })
        
        if self.recent_performances:
            stats.update({
                "recent_avg_performance": np.mean(self.recent_performances),
                "performance_trend": "improving" if len(self.recent_performances) > 1 and self.recent_performances[-1] > self.recent_performances[0] else "declining"
            })
        
        # Memory buffer statistics
        if self.memory_buffer:
            memory_stats = self.memory_buffer.get_memory_statistics()
            stats["memory_buffer"] = memory_stats
        
        return stats
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimization use memory buffer
        
        Adaptive memory management
        
        Returns:
            Statistics optimization
        """
        initial_size = len(self.memory_buffer)
        
        # Remove obsolete samples
        removed_stale = self.memory_buffer.remove_stale_samples()
        
        # Balancing by market regimes
        self.memory_buffer.balance_market_regimes()
        
        final_size = len(self.memory_buffer)
        
        optimization_stats = {
            "initial_size": initial_size,
            "final_size": final_size,
            "removed_stale_samples": removed_stale,
            "size_reduction": initial_size - final_size,
            "optimization_timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Memory optimization completed: {optimization_stats}")
        return optimization_stats
    
    def __repr__(self) -> str:
        return (
            f"RehearsalLearner("
            f"replay_ratio={self.replay_ratio}, "
            f"memory_size={len(self.memory_buffer) if self.memory_buffer else 0}, "
            f"tasks={len(self.task_history)})"
        )