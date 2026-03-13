"""
Variational Inference for Bayesian Neural Networks
Implements ELBO optimization for crypto trading uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


@dataclass
class VIConfig:
    """Configuration for Variational Inference"""
    n_samples: int = 5  # Number of weight samples per forward pass
    kl_weight: float = 1.0  # KL divergence weight in ELBO
    learning_rate: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 100
    warmup_epochs: int = 10  # KL annealing warmup
    
    # enterprise patterns
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'exponential'
    monitor_frequency: int = 10
    checkpoint_frequency: int = 20
    enable_amp: bool = True  # Automatic mixed precision


class VariationalInference:
    """
    Variational Inference trainer for Bayesian Neural Networks
    Optimizes Evidence Lower Bound (ELBO) for crypto trading models
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: VIConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.enable_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_nll': [],
            'train_kl': [],
            'val_nll': [],
            'val_metrics': []
        }
        
        # Best model state for early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler based on config"""
        if self.config.lr_scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.n_epochs
            )
        elif self.config.lr_scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.lr_scheduler == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            return optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
    
    def elbo_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        kl_divergence: torch.Tensor,
        n_batches: int,
        epoch: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Evidence Lower Bound (ELBO) loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            kl_divergence: KL divergence of the model
            n_batches: Total number of batches in dataset
            epoch: Current epoch for KL annealing
        
        Returns:
            Tuple of (total_loss, nll_loss, kl_loss)
        """
        # Negative log-likelihood (reconstruction loss)
        if predictions.shape[-1] > 1:  # Classification
            nll = nn.functional.cross_entropy(predictions, targets)
        else:  # Regression
            nll = nn.functional.mse_loss(predictions, targets)
        
        # KL annealing for stable training
        kl_weight = self._kl_annealing_weight(epoch)
        
        # Scale KL by dataset size for minibatch training
        kl_loss = kl_divergence / n_batches * kl_weight
        
        # ELBO = -NLL + KL
        total_loss = nll + kl_loss
        
        return total_loss, nll, kl_loss
    
    def _kl_annealing_weight(self, epoch: int) -> float:
        """
        KL annealing schedule for stable training
        Gradually increases KL weight during warmup
        """
        if epoch < self.config.warmup_epochs:
            return (epoch + 1) / self.config.warmup_epochs * self.config.kl_weight
        return self.config.kl_weight
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        total_nll = 0
        total_kl = 0
        n_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.n_epochs}')
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.config.enable_amp):
                # Monte Carlo sampling for expectation
                batch_loss = 0
                batch_nll = 0
                batch_kl = 0
                
                for _ in range(self.config.n_samples):
                    # Forward pass with weight sampling
                    predictions = self.model(data, sample=True)
                    
                    # Compute KL divergence
                    kl = self._compute_model_kl()
                    
                    # ELBO loss
                    loss, nll, kl_loss = self.elbo_loss(
                        predictions, targets, kl, n_batches, epoch
                    )
                    
                    batch_loss += loss / self.config.n_samples
                    batch_nll += nll / self.config.n_samples
                    batch_kl += kl_loss / self.config.n_samples
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(batch_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += batch_loss.item()
            total_nll += batch_nll.item()
            total_kl += batch_kl.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': batch_loss.item(),
                'nll': batch_nll.item(),
                'kl': batch_kl.item()
            })
        
        # Average metrics
        metrics = {
            'train_loss': total_loss / n_batches,
            'train_nll': total_nll / n_batches,
            'train_kl': total_kl / n_batches
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate model performance
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_nll = 0
        n_batches = len(val_loader)
        
        # Additional metrics for crypto trading
        predictions_list = []
        targets_list = []
        uncertainties_list = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Multiple forward passes for uncertainty estimation
                pred_samples = []
                for _ in range(self.config.n_samples * 2):  # More samples for validation
                    pred = self.model(data, sample=True)
                    pred_samples.append(pred)
                
                pred_samples = torch.stack(pred_samples)
                
                # Mean prediction
                predictions = pred_samples.mean(dim=0)
                
                # Uncertainty (std)
                uncertainty = pred_samples.std(dim=0)
                
                # Compute losses
                kl = self._compute_model_kl()
                loss, nll, _ = self.elbo_loss(
                    predictions, targets, kl, n_batches, epoch
                )
                
                total_loss += loss.item()
                total_nll += nll.item()
                
                # Store for metrics calculation
                predictions_list.append(predictions.cpu())
                targets_list.append(targets.cpu())
                uncertainties_list.append(uncertainty.cpu())
        
        # Calculate trading-specific metrics
        all_predictions = torch.cat(predictions_list)
        all_targets = torch.cat(targets_list)
        all_uncertainties = torch.cat(uncertainties_list)
        
        trading_metrics = self._calculate_trading_metrics(
            all_predictions, all_targets, all_uncertainties
        )
        
        metrics = {
            'val_loss': total_loss / n_batches,
            'val_nll': total_nll / n_batches,
            **trading_metrics
        }
        
        return metrics
    
    def _compute_model_kl(self) -> torch.Tensor:
        """Compute total KL divergence for all Bayesian layers"""
        kl = 0
        for module in self.model.modules():
            if hasattr(module, 'kl_divergence'):
                kl += module.kl_divergence()
        return kl
    
    def _calculate_trading_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate crypto trading specific metrics
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            uncertainties: Prediction uncertainties
        
        Returns:
            Dictionary of trading metrics
        """
        metrics = {}
        
        # Regression metrics
        if predictions.shape[-1] == 1:
            predictions = predictions.squeeze()
            targets = targets.squeeze()
            uncertainties = uncertainties.squeeze()
            
            # Mean Absolute Error
            mae = torch.abs(predictions - targets).mean().item()
            metrics['mae'] = mae
            
            # Directional accuracy (for price prediction)
            if len(predictions) > 1:
                pred_direction = (predictions[1:] > predictions[:-1]).float()
                true_direction = (targets[1:] > targets[:-1]).float()
                direction_acc = (pred_direction == true_direction).float().mean().item()
                metrics['direction_accuracy'] = direction_acc
            
            # Calibration error (uncertainty quality)
            # Check if uncertainties capture the errors
            errors = torch.abs(predictions - targets)
            in_confidence = (errors <= 2 * uncertainties).float().mean().item()
            metrics['uncertainty_calibration'] = in_confidence
            
            # Sharpe-like ratio (return/risk)
            if predictions.std() > 0:
                sharpe = predictions.mean() / predictions.std()
                metrics['prediction_sharpe'] = sharpe.item()
        
        # Classification metrics
        else:
            # Accuracy
            _, pred_classes = predictions.max(dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            metrics['accuracy'] = accuracy
            
            # Entropy of predictions (uncertainty measure)
            probs = torch.softmax(predictions, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()
            metrics['prediction_entropy'] = entropy
        
        # Average uncertainty
        metrics['mean_uncertainty'] = uncertainties.mean().item()
        metrics['uncertainty_std'] = uncertainties.std().item()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation and early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Optional callbacks to execute after each epoch
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting Variational Inference training on {self.device}")
        logger.info(f"Config: {self.config}")
        
        for epoch in range(self.config.n_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate(val_loader, epoch)
                
                # Early stopping check
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        break
            
            # Update learning rate
            self.scheduler.step()
            
            # Update history
            for key, value in train_metrics.items():
                self.history[key].append(value)
            for key, value in val_metrics.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            
            # Logging
            epoch_time = time.time() - start_time
            if (epoch + 1) % self.config.monitor_frequency == 0:
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_time)
            
            # Checkpointing
            if (epoch + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(epoch)
            
            # Execute callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self.model, epoch, train_metrics, val_metrics)
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"Loaded best model with val_loss: {self.best_val_loss:.4f}")
        
        return self.history
    
    def _log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float
    ):
        """Log metrics for current epoch"""
        log_str = f"Epoch {epoch+1}/{self.config.n_epochs} ({epoch_time:.2f}s) - "
        log_str += f"Train Loss: {train_metrics['train_loss']:.4f}, "
        log_str += f"NLL: {train_metrics['train_nll']:.4f}, "
        log_str += f"KL: {train_metrics['train_kl']:.4f}"
        
        if val_metrics:
            log_str += f", Val Loss: {val_metrics['val_loss']:.4f}"
            if 'accuracy' in val_metrics:
                log_str += f", Acc: {val_metrics['accuracy']:.4f}"
            if 'direction_accuracy' in val_metrics:
                log_str += f", Dir Acc: {val_metrics['direction_accuracy']:.4f}"
        
        logger.info(log_str)
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config
        }
        
        filename = f"bnn_checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, filename)
        logger.info(f"Saved checkpoint: {filename}")
    
    def predict_with_uncertainty(
        self,
        data_loader: DataLoader,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions with uncertainty estimates
        
        Args:
            data_loader: Data loader for prediction
            n_samples: Number of forward passes for uncertainty
        
        Returns:
            Tuple of (predictions, uncertainties, raw_samples)
        """
        self.model.eval()
        all_predictions = []
        all_uncertainties = []
        all_samples = []
        
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="Generating predictions"):
                data = data.to(self.device)
                
                # Multiple forward passes
                samples = []
                for _ in range(n_samples):
                    pred = self.model(data, sample=True)
                    samples.append(pred.cpu())
                
                samples = torch.stack(samples)  # [n_samples, batch, output]
                
                # Calculate statistics
                mean_pred = samples.mean(dim=0)
                std_pred = samples.std(dim=0)
                
                all_predictions.append(mean_pred.numpy())
                all_uncertainties.append(std_pred.numpy())
                all_samples.append(samples.numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        uncertainties = np.concatenate(all_uncertainties, axis=0)
        raw_samples = np.concatenate(all_samples, axis=1)
        
        return predictions, uncertainties, raw_samples