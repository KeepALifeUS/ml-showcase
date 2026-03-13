"""
Intrinsic Curiosity Module (ICM) for exploration trading strategies.

Implements curiosity-driven exploration through forward/inverse dynamics model
with enterprise patterns for system exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMConfig:
    """Configuration for Intrinsic Curiosity Module."""
    
    state_dim: int = 256
    action_dim: int = 10
    feature_dim: int = 128
    hidden_dim: int = 256
    
    # Weights loss
    forward_loss_weight: float = 0.2
    inverse_loss_weight: float = 0.8
    curiosity_reward_weight: float = 1.0
    
    # Parameters training
    learning_rate: float = 1e-4
    batch_size: int = 256
    
    # Crypto-trading specific parameters
    market_features: int = 50 # Technical indicators
    portfolio_features: int = 20 # State portfolio
    risk_features: int = 10  # Risk metrics
    
    #  cloud-native settings
    distributed_training: bool = True
    checkpoint_interval: int = 1000
    metrics_enabled: bool = True


class FeatureEncoder(nn.Module):
    """
     states for ICM with support crypto-trading data.
    
    Applies design pattern "Feature Representation Learning"
    for efficient representations states market.
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # architecture for different types data
        self.market_encoder = nn.Sequential(
            nn.Linear(config.market_features, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(config.portfolio_features, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(0.1)
        )
        
        self.risk_encoder = nn.Sequential(
            nn.Linear(config.risk_features, config.hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 4)
        )
        
        # layer
        total_features = config.hidden_dim // 2 + config.hidden_dim // 2 + config.hidden_dim // 4
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_features, config.feature_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.feature_dim)
        )
        
        logger.info(f"Feature encoder initialized with {total_features} input features")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state in compact feature representation.
        
        Args:
            state: Tensor [batch_size, state_dim]
            
        Returns:
            Encoded features: [batch_size, feature_dim]
        """
        batch_size = state.size(0)
        
        # Split state on components
        market_data = state[:, :self.config.market_features]
        portfolio_data = state[:, 
            self.config.market_features:self.config.market_features + self.config.portfolio_features
        ]
        risk_data = state[:, -self.config.risk_features:]
        
        # Encode each component
        market_features = self.market_encoder(market_data)
        portfolio_features = self.portfolio_encoder(portfolio_data)
        risk_features = self.risk_encoder(risk_data)
        
        # Merging and encoding
        combined_features = torch.cat([market_features, portfolio_features, risk_features], dim=1)
        encoded_state = self.fusion_layer(combined_features)
        
        return encoded_state


class ForwardModel(nn.Module):
    """
    Forward Dynamics Model for predictions next state.
    
    Uses design pattern "Predictive Modeling" for
      market.
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # Architecture with residual connections for stable training
        self.action_encoder = nn.Linear(config.action_dim, config.hidden_dim // 4)
        
        self.forward_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.feature_dim + config.hidden_dim // 4, config.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(config.hidden_dim),
                nn.Dropout(0.1)
            ),
            nn.Linear(config.hidden_dim, config.feature_dim)
        ])
        
        logger.info("Forward model initialized for next state prediction")
    
    def forward(self, state_features: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state from current state and action.
        
        Args:
            state_features: Encoded state features [batch_size, feature_dim]
            action: Action vector [batch_size, action_dim]
            
        Returns:
            Predicted next state features [batch_size, feature_dim]
        """
        # Encode actions
        action_encoded = F.relu(self.action_encoder(action))
        
        # Merging state and actions
        state_action = torch.cat([state_features, action_encoded], dim=1)
        
        # through forward network with residual connection
        x = state_action
        for i, layer in enumerate(self.forward_net[:-1]):
            residual = x if i > 0 else None
            x = layer(x)
            if residual is not None and residual.shape == x.shape:
                x = x + residual
        
        # Final layer without residual
        predicted_next_state = self.forward_net[-1](x)
        
        return predicted_next_state


class InverseModel(nn.Module):
    """
    Inverse Dynamics Model for predictions actions between states.
    
    Applies design pattern "Action Understanding" for learning
     aspects .
    """
    
    def __init__(self, config: ICMConfig):
        super().__init__()
        self.config = config
        
        # Symmetric architecture for processing states
        self.inverse_net = nn.Sequential(
            nn.Linear(config.feature_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(0.1),
            
            nn.Linear(config.hidden_dim // 2, config.action_dim)
        )
        
        logger.info("Inverse model initialized for action prediction")
    
    def forward(self, state_features: torch.Tensor, next_state_features: torch.Tensor) -> torch.Tensor:
        """
        Prediction actions between two states.
        
        Args:
            state_features: Current state features [batch_size, feature_dim]
            next_state_features: Next state features [batch_size, feature_dim]
            
        Returns:
            Predicted action [batch_size, action_dim]
        """
        # Merging states
        state_pair = torch.cat([state_features, next_state_features], dim=1)
        
        # Prediction actions
        predicted_action = self.inverse_net(state_pair)
        
        return predicted_action


class CuriosityRewardCalculator:
    """
    Calculator intrinsic reward on basis prediction error.
    
    Uses design pattern "Reward Engineering" for
      .
    """
    
    def __init__(self, config: ICMConfig):
        self.config = config
        self.prediction_errors = []
        self.running_mean = 0.0
        self.running_var = 1.0
        self.alpha = 0.01 # Coefficient for
        
        logger.info("Curiosity reward calculator initialized")
    
    def calculate_curiosity_reward(
        self,
        predicted_next_state: torch.Tensor,
        actual_next_state: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Computation curiosity reward on basis prediction error.
        
        Args:
            predicted_next_state: next state
            actual_next_state: next state
            normalize: normalization
            
        Returns:
            Curiosity rewards for each sample in batch
        """
        # Computation L2 prediction error
        prediction_error = F.mse_loss(
            predicted_next_state, 
            actual_next_state, 
            reduction='none'
        ).mean(dim=1)
        
        if normalize:
            # Update running statistics
            current_mean = prediction_error.mean().item()
            current_var = prediction_error.var().item()
            
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * current_mean
            self.running_var = (1 - self.alpha) * self.running_var + self.alpha * current_var
            
            # Normalization reward
            std = (self.running_var + 1e-8) ** 0.5
            normalized_error = (prediction_error - self.running_mean) / std
            curiosity_reward = torch.clamp(normalized_error, 0, 5) # Limitation
        else:
            curiosity_reward = prediction_error
        
        # Save for statistics
        self.prediction_errors.extend(prediction_error.detach().cpu().numpy())
        if len(self.prediction_errors) > 10000:
            self.prediction_errors = self.prediction_errors[-5000:]
        
        return curiosity_reward * self.config.curiosity_reward_weight


class ICMTrainer:
    """
    Trainer for Intrinsic Curiosity Module with advanced optimization.
    
    Implements design pattern "Distributed Learning" for
    efficient training on large volumes data.
    """
    
    def __init__(self, config: ICMConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize models
        self.feature_encoder = FeatureEncoder(config).to(device)
        self.forward_model = ForwardModel(config).to(device)
        self.inverse_model = InverseModel(config).to(device)
        self.curiosity_calculator = CuriosityRewardCalculator(config)
        
        # Optimizers with different learning rates
        self.optimizer = torch.optim.Adam([
            {'params': self.feature_encoder.parameters(), 'lr': config.learning_rate},
            {'params': self.forward_model.parameters(), 'lr': config.learning_rate * 0.5},
            {'params': self.inverse_model.parameters(), 'lr': config.learning_rate * 1.5}
        ], weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=2
        )
        
        self.training_step = 0
        self.metrics = {
            'forward_loss': [],
            'inverse_loss': [],
            'total_loss': [],
            'curiosity_rewards': []
        }
        
        logger.info(f"ICM trainer initialized on device: {device}")
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Execute one step training ICM.
        
        Args:
            states: Batch of current states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
            
        Returns:
            Dictionary with metrics training
        """
        self.optimizer.zero_grad()
        
        # Encode states
        state_features = self.feature_encoder(states)
        next_state_features = self.feature_encoder(next_states)
        
        # Forward model prediction
        predicted_next_features = self.forward_model(state_features, actions)
        
        # Inverse model prediction
        predicted_actions = self.inverse_model(state_features, next_state_features)
        
        # Computation loss
        forward_loss = F.mse_loss(predicted_next_features, next_state_features.detach())
        inverse_loss = F.mse_loss(predicted_actions, actions)
        
        # Total
        total_loss = (
            self.config.forward_loss_weight * forward_loss +
            self.config.inverse_loss_weight * inverse_loss
        )
        
        # Backpropagation
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_encoder.parameters()) +
            list(self.forward_model.parameters()) +
            list(self.inverse_model.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        self.scheduler.step()
        
        # Computation curiosity reward
        with torch.no_grad():
            curiosity_rewards = self.curiosity_calculator.calculate_curiosity_reward(
                predicted_next_features, next_state_features
            )
        
        # Update metrics
        metrics = {
            'forward_loss': forward_loss.item(),
            'inverse_loss': inverse_loss.item(),
            'total_loss': total_loss.item(),
            'curiosity_reward_mean': curiosity_rewards.mean().item(),
            'curiosity_reward_std': curiosity_rewards.std().item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        for key, value in metrics.items():
            if key != 'learning_rate':
                self.metrics[key.replace('_mean', '').replace('_std', '')].append(value)
        
        self.training_step += 1
        
        return metrics
    
    def get_curiosity_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Get curiosity reward for evaluation.
        
        Args:
            state: Current state
            action: Executed action
            next_state: Resulting state
            
        Returns:
            Curiosity reward
        """
        with torch.no_grad():
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)
            predicted_next_features = self.forward_model(state_features, action)
            
            curiosity_reward = self.curiosity_calculator.calculate_curiosity_reward(
                predicted_next_features, next_state_features
            )
            
        return curiosity_reward
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint model."""
        checkpoint = {
            'feature_encoder': self.feature_encoder.state_dict(),
            'forward_model': self.forward_model.state_dict(),
            'inverse_model': self.inverse_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'metrics': self.metrics
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.feature_encoder.load_state_dict(checkpoint['feature_encoder'])
        self.forward_model.load_state_dict(checkpoint['forward_model'])
        self.inverse_model.load_state_dict(checkpoint['inverse_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']
        self.metrics = checkpoint['metrics']
        
        logger.info(f"Checkpoint loaded from {filepath}")


class CryptoICMEnvironment:
    """
    Specialized for integration ICM with crypto trading environment.
    
    Applies design pattern "Environment Adaptation" for
    seamless integration with systems.
    """
    
    def __init__(self, base_env, icm_trainer: ICMTrainer, reward_mix: float = 0.1):
        self.base_env = base_env
        self.icm_trainer = icm_trainer
        self.reward_mix = reward_mix # Fraction intrinsic reward in reward
        
        self.last_state = None
        self.last_action = None
        
        logger.info(f"Crypto ICM environment initialized with reward mix: {reward_mix}")
    
    def step(self, action):
        """
        Execute step with curiosity reward.
        
        Args:
            action:
            
        Returns:
            Tuple (next_state, total_reward, done, info)
        """
        # Execute actions in base environment
        next_state, extrinsic_reward, done, info = self.base_env.step(action)
        
        # Computation curiosity reward if there is state
        intrinsic_reward = 0.0
        if self.last_state is not None and self.last_action is not None:
            state_tensor = torch.FloatTensor(self.last_state).unsqueeze(0).to(self.icm_trainer.device)
            action_tensor = torch.FloatTensor(self.last_action).unsqueeze(0).to(self.icm_trainer.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.icm_trainer.device)
            
            curiosity_reward = self.icm_trainer.get_curiosity_reward(
                state_tensor, action_tensor, next_state_tensor
            )
            intrinsic_reward = curiosity_reward.item()
        
        # Merging extrinsic and intrinsic rewards
        total_reward = extrinsic_reward + self.reward_mix * intrinsic_reward
        
        # Update state for next step
        self.last_state = next_state.copy() if hasattr(next_state, 'copy') else next_state
        self.last_action = action.copy() if hasattr(action, 'copy') else action
        
        # Add curiosity info
        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = extrinsic_reward
        info['reward_mix'] = self.reward_mix
        
        return next_state, total_reward, done, info
    
    def reset(self):
        """Reset ."""
        state = self.base_env.reset()
        self.last_state = state.copy() if hasattr(state, 'copy') else state
        self.last_action = None
        return state


def create_icm_system(config: ICMConfig) -> Tuple[ICMTrainer, CryptoICMEnvironment]:
    """
    Factory function for creation complete ICM system.
    
    Args:
        config: Configuration ICM
        
    Returns:
        Tuple (ICM trainer, ICM-wrapped environment)
    """
    # Initialize ICM trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    icm_trainer = ICMTrainer(config, device)
    
    logger.info("ICM system created successfully")
    logger.info(f"Feature encoder parameters: {sum(p.numel() for p in icm_trainer.feature_encoder.parameters())}")
    logger.info(f"Forward model parameters: {sum(p.numel() for p in icm_trainer.forward_model.parameters())}")
    logger.info(f"Inverse model parameters: {sum(p.numel() for p in icm_trainer.inverse_model.parameters())}")
    
    return icm_trainer


if __name__ == "__main__":
    # Example use ICM for crypto trading
    config = ICMConfig(
        state_dim=80,  # 50 market + 20 portfolio + 10 risk
        action_dim=5, # Buy/Sell/Hold for different assets
        feature_dim=64,
        hidden_dim=128
    )
    
    icm_trainer = create_icm_system(config)
    
    # Create synthetic data for demonstration
    batch_size = 32
    states = torch.randn(batch_size, config.state_dim)
    actions = torch.randn(batch_size, config.action_dim)
    next_states = torch.randn(batch_size, config.state_dim)
    
    # Training ICM
    metrics = icm_trainer.train_step(states, actions, next_states)
    print("Training metrics:", metrics)
    
    # Get curiosity reward
    curiosity_reward = icm_trainer.get_curiosity_reward(
        states[:1], actions[:1], next_states[:1]
    )
    print(f"Curiosity reward: {curiosity_reward.item():.4f}")