"""
Random Network Distillation (RND) for exploration through .

Implements advanced exploration mechanism through prediction error from random network
with enterprise patterns for scalable curiosity systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RNDConfig:
    """Configuration for Random Network Distillation."""
    
    state_dim: int = 256
    target_network_dim: int = 512
    predictor_network_dim: int = 512
    hidden_layers: List[int] = None
    
    # Parameters training
    learning_rate: float = 1e-4
    batch_size: int = 256
    target_update_frequency: int = 1000
    
    # Reward scaling
    intrinsic_reward_coeff: float = 1.0
    reward_normalization: bool = True
    reward_clip_max: float = 5.0
    
    # Running statistics
    obs_normalize: bool = True
    obs_clip_max: float = 5.0
    
    # Crypto-specific parameters
    market_volatility_weight: float = 0.3
    portfolio_diversity_weight: float = 0.4
    risk_novelty_weight: float = 0.3
    
    #  enterprise settings
    distributed_mode: bool = True
    checkpoint_interval: int = 2000
    metrics_tracking: bool = True
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [512, 256, 128]


class RunningMeanStd:
    """
    Running mean and standard deviation calculator for normalization.
    
    Uses design pattern "Streaming Statistics" for
     processing continuous data streams.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        
    def update(self, x: np.ndarray) -> None:
        """Update statistics new data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        """Update from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalization data."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class RandomNetwork(nn.Module):
    """
    Random target network for RND.
    
    Applies design pattern "Fixed Random Features" for
    creation stable exploration targets.
    """
    
    def __init__(self, config: RNDConfig):
        super().__init__()
        self.config = config
        
        # Build network with architecture
        layers = []
        prev_dim = config.state_dim
        
        for hidden_dim in config.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim) # for random network
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, config.target_network_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Freeze parameters - network random
        for param in self.parameters():
            param.requires_grad = False
        
        logger.info(f"Random target network initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(x)


class PredictorNetwork(nn.Module):
    """
    Predictor network for training random network output.
    
    Uses design pattern "Adaptive Learning" for
    efficient training representation.
    """
    
    def __init__(self, config: RNDConfig):
        super().__init__()
        self.config = config
        
        # More architecture for predictor network
        layers = []
        prev_dim = config.state_dim
        
        # Encoder part with attention mechanism for crypto data
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_layers[0]),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_layers[0]),
            nn.Dropout(0.2)
        )
        
        # Multi-head attention for extraction features
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_layers[0],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Predictor layers
        prev_dim = config.hidden_layers[0]
        for i, hidden_dim in enumerate(config.hidden_layers[1:], 1):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1 if i < len(config.hidden_layers) - 1 else 0.05)
            ])
            prev_dim = hidden_dim
        
        # Output layer with regularization
        layers.extend([
            nn.Linear(prev_dim, config.predictor_network_dim),
            nn.Tanh() # Limitation output values
        ])
        
        self.predictor = nn.Sequential(*layers)
        
        logger.info(f"Predictor network initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Input state tensor [batch_size, state_dim]
            
        Returns:
            Predicted random network output [batch_size, predictor_network_dim]
        """
        batch_size = x.size(0)
        
        # Encode state
        encoded = self.state_encoder(x)
        
        # Self-attention for extraction patterns
        # Add sequence dimension for attention
        encoded_seq = encoded.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attended, _ = self.attention(encoded_seq, encoded_seq, encoded_seq)
        attended = attended.squeeze(1)  # [batch_size, hidden_dim]
        
        # Residual connection
        attended = attended + encoded
        
        # Predictor network
        prediction = self.predictor(attended)
        
        return prediction


class CryptoStateProcessor:
    """
     states for crypto trading.
    
    Implements design pattern "Domain-Specific Processing" for
     processing financial data.
    """
    
    def __init__(self, config: RNDConfig):
        self.config = config
        
        # Weights for different components state
        self.market_weight = config.market_volatility_weight
        self.portfolio_weight = config.portfolio_diversity_weight
        self.risk_weight = config.risk_novelty_weight
        
        # Statistics for normalization
        self.market_stats = RunningMeanStd()
        self.portfolio_stats = RunningMeanStd()
        self.risk_stats = RunningMeanStd()
        
        logger.info("Crypto state processor initialized")
    
    def process_state(self, state: np.ndarray) -> np.ndarray:
        """
        Processing state with consideration crypto-specifics.
        
        Args:
            state: Raw state from trading environment
            
        Returns:
            Processed state for RND
        """
        # Split state on components ( )
        total_features = state.shape[-1]
        market_end = int(total_features * 0.6)  # 60% - market data
        portfolio_end = int(total_features * 0.85)  # 25% - portfolio
        # Rest - metrics
        
        market_data = state[..., :market_end]
        portfolio_data = state[..., market_end:portfolio_end]
        risk_data = state[..., portfolio_end:]
        
        # Update statistics and normalization
        if len(state.shape) == 2:  # Batch
            self.market_stats.update(market_data)
            self.portfolio_stats.update(portfolio_data)
            self.risk_stats.update(risk_data)
        
        market_normalized = self.market_stats.normalize(market_data)
        portfolio_normalized = self.portfolio_stats.normalize(portfolio_data)
        risk_normalized = self.risk_stats.normalize(risk_data)
        
        # Weighted with crypto-specific weights
        processed_state = np.concatenate([
            market_normalized * self.market_weight,
            portfolio_normalized * self.portfolio_weight,
            risk_normalized * self.risk_weight
        ], axis=-1)
        
        return processed_state


class RNDTrainer:
    """
    Trainer for Random Network Distillation with advanced optimization.
    
    Applies design pattern "Distributed Training" for
    scalable learning on large volumes trading data.
    """
    
    def __init__(self, config: RNDConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Initialize networks
        self.target_network = RandomNetwork(config).to(device)
        self.predictor_network = PredictorNetwork(config).to(device)
        self.state_processor = CryptoStateProcessor(config)
        
        # Optimizer only for predictor network
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-6
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=100000,
            pct_start=0.1
        )
        
        # Statistics for intrinsic reward
        self.reward_stats = RunningMeanStd()
        self.obs_stats = RunningMeanStd(shape=(config.state_dim,))
        
        # Metrics
        self.training_step = 0
        self.prediction_errors = deque(maxlen=10000)
        self.intrinsic_rewards = deque(maxlen=10000)
        
        # Performance tracking
        self.last_update_time = time.time()
        self.updates_per_second = 0.0
        
        logger.info(f"RND trainer initialized on device: {device}")
        logger.info(f"Target network frozen with {sum(p.numel() for p in self.target_network.parameters())} parameters")
    
    def normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Normalization observations."""
        if self.config.obs_normalize:
            # Update statistics
            obs_np = observations.detach().cpu().numpy()
            self.obs_stats.update(obs_np)
            
            # Normalization
            normalized = self.obs_stats.normalize(obs_np)
            normalized = np.clip(normalized, -self.config.obs_clip_max, self.config.obs_clip_max)
            
            return torch.FloatTensor(normalized).to(self.device)
        
        return observations
    
    def compute_intrinsic_reward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Computation intrinsic reward through RND prediction error.
        
        Args:
            observations: Batch of observations [batch_size, state_dim]
            
        Returns:
            Intrinsic rewards [batch_size]
        """
        with torch.no_grad():
            # Normalization observations
            normalized_obs = self.normalize_observations(observations)
            
            # Get targets and predictions
            target_output = self.target_network(normalized_obs)
            predicted_output = self.predictor_network(normalized_obs)
            
            # Computation MSE error as intrinsic reward
            prediction_error = F.mse_loss(
                predicted_output, target_output, reduction='none'
            ).mean(dim=1)
            
            # Save for statistics
            self.prediction_errors.extend(prediction_error.cpu().numpy())
            
            if self.config.reward_normalization:
                # Update reward statistics
                error_np = prediction_error.cpu().numpy()
                self.reward_stats.update(error_np)
                
                # Normalization reward
                normalized_reward = self.reward_stats.normalize(error_np)
                normalized_reward = np.clip(
                    normalized_reward, 0, self.config.reward_clip_max
                )
                
                intrinsic_reward = torch.FloatTensor(normalized_reward).to(self.device)
            else:
                intrinsic_reward = prediction_error
            
            # Scale
            intrinsic_reward = intrinsic_reward * self.config.intrinsic_reward_coeff
            
            # Save for statistics
            self.intrinsic_rewards.extend(intrinsic_reward.cpu().numpy())
            
            return intrinsic_reward
    
    def train_step(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Execute one step training RND.
        
        Args:
            observations: Batch of observations for training
            
        Returns:
            Dictionary with metrics training
        """
        start_time = time.time()
        
        # Processing states for crypto trading
        if isinstance(observations, np.ndarray):
            processed_obs = self.state_processor.process_state(observations)
            observations = torch.FloatTensor(processed_obs).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Normalization observations
        normalized_obs = self.normalize_observations(observations)
        
        # Forward pass
        target_output = self.target_network(normalized_obs)
        predicted_output = self.predictor_network(normalized_obs)
        
        # Computation
        prediction_loss = F.mse_loss(predicted_output, target_output.detach())
        
        # Backpropagation
        prediction_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.predictor_network.parameters(), max_norm=1.0
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update statistics
        self.training_step += 1
        update_time = time.time() - start_time
        self.updates_per_second = 0.9 * self.updates_per_second + 0.1 / update_time
        
        # Computation current intrinsic rewards
        with torch.no_grad():
            current_rewards = self.compute_intrinsic_reward(observations)
        
        # Metrics
        metrics = {
            'prediction_loss': prediction_loss.item(),
            'intrinsic_reward_mean': current_rewards.mean().item(),
            'intrinsic_reward_std': current_rewards.std().item(),
            'prediction_error_mean': np.mean(list(self.prediction_errors)[-100:]) if self.prediction_errors else 0.0,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'updates_per_second': self.updates_per_second,
            'training_step': self.training_step
        }
        
        return metrics
    
    def get_exploration_bonus(self, observation: torch.Tensor) -> float:
        """
        Get exploration bonus for single observation.
        
        Args:
            observation: Single observation
            
        Returns:
            Exploration bonus value
        """
        if len(observation.shape) == 1:
            observation = observation.unsqueeze(0)
        
        intrinsic_reward = self.compute_intrinsic_reward(observation)
        return intrinsic_reward.item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint."""
        checkpoint = {
            'predictor_network': self.predictor_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'reward_stats': self.reward_stats,
            'obs_stats': self.obs_stats,
            'state_processor': self.state_processor
        }
        torch.save(checkpoint, filepath)
        logger.info(f"RND checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.predictor_network.load_state_dict(checkpoint['predictor_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']
        self.reward_stats = checkpoint['reward_stats']
        self.obs_stats = checkpoint['obs_stats']
        self.state_processor = checkpoint['state_processor']
        
        logger.info(f"RND checkpoint loaded from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics RND."""
        return {
            'training_step': self.training_step,
            'updates_per_second': self.updates_per_second,
            'prediction_errors': {
                'mean': np.mean(list(self.prediction_errors)) if self.prediction_errors else 0.0,
                'std': np.std(list(self.prediction_errors)) if self.prediction_errors else 0.0,
                'count': len(self.prediction_errors)
            },
            'intrinsic_rewards': {
                'mean': np.mean(list(self.intrinsic_rewards)) if self.intrinsic_rewards else 0.0,
                'std': np.std(list(self.intrinsic_rewards)) if self.intrinsic_rewards else 0.0,
                'count': len(self.intrinsic_rewards)
            },
            'reward_normalization': {
                'mean': self.reward_stats.mean,
                'var': self.reward_stats.var,
                'count': self.reward_stats.count
            },
            'observation_normalization': {
                'mean': self.obs_stats.mean,
                'var': self.obs_stats.var,
                'count': self.obs_stats.count
            }
        }


class CryptoRNDEnvironment:
    """
    Crypto trading environment with RND-based exploration.
    
    Integrates design pattern "Environment Augmentation" for
    enhanced exploration in financial .
    """
    
    def __init__(
        self,
        base_env,
        rnd_trainer: RNDTrainer,
        intrinsic_reward_weight: float = 0.1,
        exploration_bonus_decay: float = 0.99
    ):
        self.base_env = base_env
        self.rnd_trainer = rnd_trainer
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.exploration_bonus_decay = exploration_bonus_decay
        
        self.episode_step = 0
        self.total_episodes = 0
        self.episode_intrinsic_rewards = []
        
        logger.info(f"Crypto RND environment initialized with intrinsic weight: {intrinsic_reward_weight}")
    
    def step(self, action):
        """Step with RND exploration bonus."""
        # Execute actions in base environment
        next_state, extrinsic_reward, done, info = self.base_env.step(action)
        
        # Get RND exploration bonus
        state_tensor = torch.FloatTensor(next_state).to(self.rnd_trainer.device)
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        intrinsic_reward = self.rnd_trainer.compute_intrinsic_reward(state_tensor).item()
        
        # Decay exploration bonus by episode
        decayed_intrinsic_reward = intrinsic_reward * (self.exploration_bonus_decay ** self.episode_step)
        
        # Total reward
        total_reward = extrinsic_reward + self.intrinsic_reward_weight * decayed_intrinsic_reward
        
        # Update statistics
        self.episode_step += 1
        self.episode_intrinsic_rewards.append(intrinsic_reward)
        
        # Information for analysis
        info.update({
            'intrinsic_reward': intrinsic_reward,
            'decayed_intrinsic_reward': decayed_intrinsic_reward,
            'extrinsic_reward': extrinsic_reward,
            'total_reward': total_reward,
            'intrinsic_weight': self.intrinsic_reward_weight,
            'episode_step': self.episode_step
        })
        
        if done:
            info['episode_intrinsic_reward_sum'] = sum(self.episode_intrinsic_rewards)
            info['episode_intrinsic_reward_mean'] = np.mean(self.episode_intrinsic_rewards)
        
        return next_state, total_reward, done, info
    
    def reset(self):
        """Reset environment."""
        state = self.base_env.reset()
        
        # Reset episode statistics
        self.episode_step = 0
        self.total_episodes += 1
        self.episode_intrinsic_rewards = []
        
        return state
    
    def update_rnd(self, batch_observations: np.ndarray) -> Dict[str, float]:
        """Update RND with batch observations."""
        return self.rnd_trainer.train_step(batch_observations)


def create_rnd_system(config: RNDConfig) -> RNDTrainer:
    """
    Factory function for creation RND system.
    
    Args:
        config: RND configuration
        
    Returns:
        Configured RND trainer
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rnd_trainer = RNDTrainer(config, device)
    
    logger.info("RND system created successfully")
    logger.info(f"Target network parameters: {sum(p.numel() for p in rnd_trainer.target_network.parameters())}")
    logger.info(f"Predictor network parameters: {sum(p.numel() for p in rnd_trainer.predictor_network.parameters())}")
    
    return rnd_trainer


if __name__ == "__main__":
    # Example use RND for crypto trading exploration
    config = RNDConfig(
        state_dim=128,  # Crypto market state
        target_network_dim=256,
        predictor_network_dim=256,
        hidden_layers=[512, 256, 128],
        learning_rate=1e-4,
        intrinsic_reward_coeff=0.1
    )
    
    rnd_trainer = create_rnd_system(config)
    
    # Create synthetic crypto trading data
    batch_size = 64
    observations = np.random.randn(batch_size, config.state_dim)
    
    # Training RND
    for step in range(100):
        metrics = rnd_trainer.train_step(observations)
        
        if step % 20 == 0:
            print(f"Step {step}: Loss={metrics['prediction_loss']:.4f}, "
                  f"Intrinsic Reward={metrics['intrinsic_reward_mean']:.4f}")
    
    # Get exploration bonus
    single_obs = torch.randn(1, config.state_dim)
    exploration_bonus = rnd_trainer.get_exploration_bonus(single_obs)
    print(f"Exploration bonus: {exploration_bonus:.4f}")
    
    # Statistics
    stats = rnd_trainer.get_statistics()
    print("RND Statistics:", stats)