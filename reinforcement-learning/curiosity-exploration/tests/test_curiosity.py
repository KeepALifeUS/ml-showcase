"""
Comprehensive tests for curiosity-driven exploration system.

Implements thorough testing all components with enterprise patterns
for reliable and robust system validation.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import for testing
from src.curiosity.icm import ICMConfig, ICMTrainer
from src.curiosity.rnd import RNDConfig, RNDTrainer
from src.curiosity.ngu import NGUConfig, NGUTrainer
from src.exploration.count_based import CountBasedConfig, CountBasedExplorer
from src.exploration.prediction_based import PredictionBasedConfig, PredictionBasedExplorer
from src.exploration.exploration_bonus import ExplorationBonusConfig, ExplorationBonusManager
from src.novelty.novelty_detector import NoveltyDetectionConfig, CryptoNoveltyDetector
from src.novelty.state_visitor import StateVisitorConfig, StateVisitor
from src.memory.episodic_memory import EpisodicMemoryConfig, EpisodicMemorySystem
from src.memory.curiosity_buffer import CuriosityBufferConfig, CuriosityReplayBuffer
from src.agents.curious_trader import CuriousTraderConfig, CuriousTrader
from src.agents.exploration_agent import ExplorationAgentConfig, ExplorationAgent
from src.utils.state_encoder import StateEncoderConfig, CryptoStateEncoder
from src.utils.reward_shaping import RewardShapingConfig, CryptoRewardShaper


class TestICMSystem:
    """Test suite for ICM system."""
    
    def test_icm_initialization(self):
        """Test ICM initialization."""
        config = ICMConfig(state_dim=128, action_dim=5)
        icm = ICMTrainer(config, device='cpu')
        
        assert icm.config.state_dim == 128
        assert icm.config.action_dim == 5
        assert icm.device == 'cpu'
    
    def test_icm_training_step(self):
        """Test ICM training step."""
        config = ICMConfig(state_dim=64, action_dim=3)
        icm = ICMTrainer(config, device='cpu')
        
        batch_size = 32
        states = torch.randn(batch_size, 64)
        actions = torch.randn(batch_size, 3)
        next_states = torch.randn(batch_size, 64)
        
        metrics = icm.train_step(states, actions, next_states)
        
        assert 'forward_loss' in metrics
        assert 'inverse_loss' in metrics
        assert 'total_loss' in metrics
        assert isinstance(metrics['forward_loss'], float)
    
    def test_icm_curiosity_reward(self):
        """Test ICM curiosity reward computation."""
        config = ICMConfig(state_dim=32, action_dim=2)
        icm = ICMTrainer(config, device='cpu')
        
        state = torch.randn(1, 32)
        action = torch.randn(1, 2)
        next_state = torch.randn(1, 32)
        
        reward = icm.get_curiosity_reward(state, action, next_state)
        
        assert isinstance(reward, torch.Tensor)
        assert reward.numel() == 1
        assert reward.item() >= 0


class TestRNDSystem:
    """Test suite for RND system."""
    
    def test_rnd_initialization(self):
        """Test RND initialization."""
        config = RNDConfig(state_dim=128)
        rnd = RNDTrainer(config, device='cpu')
        
        assert rnd.config.state_dim == 128
        assert rnd.target_network is not None
        assert rnd.predictor_network is not None
    
    def test_rnd_training(self):
        """Test RND training step."""
        config = RNDConfig(state_dim=64, batch_size=16)
        rnd = RNDTrainer(config, device='cpu')
        
        observations = np.random.randn(16, 64)
        metrics = rnd.train_step(observations)
        
        assert 'prediction_loss' in metrics
        assert isinstance(metrics['prediction_loss'], float)
    
    def test_rnd_intrinsic_reward(self):
        """Test RND intrinsic reward computation."""
        config = RNDConfig(state_dim=32)
        rnd = RNDTrainer(config, device='cpu')
        
        observation = torch.randn(1, 32)
        reward = rnd.compute_intrinsic_reward(observation)
        
        assert isinstance(reward, torch.Tensor)
        assert reward.numel() == 1


class TestNGUSystem:
    """Test suite for NGU system."""
    
    def test_ngu_initialization(self):
        """Test NGU initialization."""
        config = NGUConfig(state_dim=128, action_dim=5)
        ngu = NGUTrainer(config, device='cpu')
        
        assert ngu.config.state_dim == 128
        assert ngu.state_embedder is not None
        assert ngu.episodic_memory is not None
    
    def test_ngu_intrinsic_reward(self):
        """Test NGU intrinsic reward computation."""
        config = NGUConfig(state_dim=64, action_dim=3)
        ngu = NGUTrainer(config, device='cpu')
        
        state = torch.randn(1, 64)
        reward, breakdown = ngu.compute_intrinsic_reward(
            state, episode_id=0, market_regime=1
        )
        
        assert isinstance(reward, float)
        assert isinstance(breakdown, dict)
        assert 'rnd_reward' in breakdown
        assert 'episodic_bonus' in breakdown


class TestCountBasedExploration:
    """Test suite for count-based exploration."""
    
    def test_count_based_initialization(self):
        """Test count-based explorer initialization."""
        config = CountBasedConfig()
        explorer = CountBasedExplorer(config, state_dim=128)
        
        assert explorer.state_dim == 128
        assert explorer.total_visits == 0
    
    def test_count_based_bonus(self):
        """Test count-based exploration bonus."""
        config = CountBasedConfig()
        explorer = CountBasedExplorer(config, state_dim=64)
        
        state = np.random.randn(64)
        
        # First visit should give high bonus
        bonus1 = explorer.get_count_bonus(state)
        explorer.update_counts(state)
        
        # Second visit should give lower bonus
        bonus2 = explorer.get_count_bonus(state)
        
        assert bonus1 > bonus2
        assert bonus1 > 0
        assert bonus2 > 0


class TestPredictionBasedExploration:
    """Test suite for prediction-based exploration."""
    
    def test_prediction_based_initialization(self):
        """Test prediction-based explorer initialization."""
        config = PredictionBasedConfig(state_dim=128, action_dim=5)
        explorer = PredictionBasedExplorer(config, device='cpu')
        
        assert explorer.config.state_dim == 128
        assert explorer.predictor is not None
    
    def test_uncertainty_bonus(self):
        """Test uncertainty-based exploration bonus."""
        config = PredictionBasedConfig(state_dim=64, action_dim=3, ensemble_size=2)
        explorer = PredictionBasedExplorer(config, device='cpu')
        
        state = torch.randn(1, 64)
        action = torch.randn(1, 3)
        
        bonus, breakdown = explorer.get_uncertainty_bonus(state, action)
        
        assert isinstance(bonus, float)
        assert isinstance(breakdown, dict)
        assert bonus >= 0


class TestNoveltyDetection:
    """Test suite for novelty detection."""
    
    def test_novelty_detector_initialization(self):
        """Test novelty detector initialization."""
        config = NoveltyDetectionConfig(detection_methods=["autoencoder"])
        detector = CryptoNoveltyDetector(config, input_dim=128, device='cpu')
        
        assert len(detector.detectors) > 0
        assert detector.input_dim == 128
    
    def test_novelty_detection(self):
        """Test novelty detection."""
        config = NoveltyDetectionConfig(detection_methods=["autoencoder"])
        detector = CryptoNoveltyDetector(config, input_dim=64, device='cpu')
        
        # Fit on normal data
        normal_data = np.random.randn(100, 64)
        detector.fit(normal_data)
        
        # Detect novelty
        test_data = np.random.randn(1, 64) * 3  # More extreme
        novelty_score, breakdown = detector.detect_novelty(test_data)
        
        assert isinstance(novelty_score, float)
        assert isinstance(breakdown, dict)


class TestStateVisitor:
    """Test suite for state visitor system."""
    
    def test_state_visitor_initialization(self):
        """Test state visitor initialization."""
        config = StateVisitorConfig()
        visitor = StateVisitor(config, state_dim=128)
        
        assert visitor.state_dim == 128
        assert visitor.total_visits == 0
    
    def test_state_visit_tracking(self):
        """Test state visit tracking."""
        config = StateVisitorConfig()
        visitor = StateVisitor(config, state_dim=64)
        
        state = np.random.randn(64)
        
        stats = visitor.visit_state(state, market_regime="bull")
        
        assert stats['is_first_visit'] == True
        assert stats['new_visit_count'] == 1
        assert visitor.total_visits == 1


class TestMemorySystems:
    """Test suite for memory systems."""
    
    def test_episodic_memory(self):
        """Test episodic memory system."""
        config = EpisodicMemoryConfig(memory_capacity=100, embedding_dim=32)
        memory = EpisodicMemorySystem(config)
        
        # Add experience
        embedding = np.random.randn(32)
        memory.add_experience(embedding, reward=0.5, episode_id=0)
        
        assert memory.size == 1
        
        # Retrieve similar
        query = np.random.randn(32)
        distances, indices = memory.retrieve_similar(query, k=1)
        
        assert len(distances) == 1
        assert len(indices) == 1
    
    def test_curiosity_buffer(self):
        """Test curiosity replay buffer."""
        config = CuriosityBufferConfig(buffer_capacity=100)
        buffer = CuriosityReplayBuffer(config)
        
        # Add experience
        state = np.random.randn(10)
        action = np.random.randn(3)
        reward = 0.1
        next_state = np.random.randn(10)
        curiosity_reward = 0.05
        
        buffer.add(state, action, reward, next_state, False, curiosity_reward)
        
        assert buffer.size == 1
        
        # Add more for sampling
        for _ in range(20):
            buffer.add(
                np.random.randn(10), np.random.randn(3), np.random.randn(),
                np.random.randn(10), False, np.random.randn()
            )
        
        # Sample batch
        batch = buffer.sample(5)
        
        assert 'states' in batch
        assert batch['states'].shape[0] == 5


class TestAgents:
    """Test suite for trading agents."""
    
    def test_curious_trader(self):
        """Test curious trader agent."""
        config = CuriousTraderConfig(state_dim=64, action_dim=3)
        trader = CuriousTrader(config, device='cpu')
        
        assert trader.config.state_dim == 64
        assert trader.policy_net is not None
        
        # Test action selection
        state = np.random.randn(64)
        action, exploration_info = trader.select_action(state)
        
        assert len(action) == 3
        assert isinstance(exploration_info, dict)
    
    def test_exploration_agent(self):
        """Test exploration agent."""
        config = ExplorationAgentConfig(state_dim=64, action_dim=3)
        agent = ExplorationAgent(config, device='cpu')
        
        assert agent.config.state_dim == 64
        
        # Test exploration action
        state = np.random.randn(64)
        action, info = agent.select_exploration_action(state)
        
        assert len(action) == 3
        assert 'strategy' in info


class TestUtilities:
    """Test suite for utility functions."""
    
    def test_state_encoder(self):
        """Test state encoder."""
        config = StateEncoderConfig(input_dim=256, encoding_dim=64)
        encoder = CryptoStateEncoder(config)
        
        batch_size = 16
        states = torch.randn(batch_size, 256)
        
        with torch.no_grad():
            encoded = encoder(states)
        
        assert encoded.shape == (batch_size, 64)
    
    def test_reward_shaper(self):
        """Test reward shaper."""
        config = RewardShapingConfig()
        shaper = CryptoRewardShaper(config)
        
        breakdown = shaper.shape_reward(
            extrinsic_reward=0.01,
            intrinsic_reward=0.05,
            curiosity_bonus=0.02,
            exploration_bonus=0.01,
            risk_penalty=0.005
        )
        
        assert 'shaped_reward' in breakdown
        assert isinstance(breakdown['shaped_reward'], float)


class TestIntegration:
    """Integration tests for full system."""
    
    def test_full_exploration_pipeline(self):
        """Test complete exploration pipeline."""
        # Initialize components
        icm_config = ICMConfig(state_dim=64, action_dim=3)
        icm = ICMTrainer(icm_config, device='cpu')
        
        count_config = CountBasedConfig()
        count_explorer = CountBasedExplorer(count_config, state_dim=64)
        
        bonus_config = ExplorationBonusConfig()
        bonus_manager = ExplorationBonusManager(bonus_config)
        
        # Simulate exploration loop
        for step in range(10):
            state = np.random.randn(64)
            action = np.random.randn(3)
            next_state = np.random.randn(64)
            
            # Get curiosity reward
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            
            curiosity_reward = icm.get_curiosity_reward(
                state_tensor, action_tensor, next_state_tensor
            ).item()
            
            # Get count-based bonus
            count_bonus = count_explorer.get_count_bonus(state)
            count_explorer.update_counts(state)
            
            # Combine bonuses
            total_bonus, breakdown = bonus_manager.compute_exploration_bonus(
                state=state,
                action=action,
                context={'prediction_error': curiosity_reward}
            )
            
            assert curiosity_reward >= 0
            assert count_bonus >= 0
            assert total_bonus >= 0
            assert isinstance(breakdown, dict)
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test ICM checkpointing
            config = ICMConfig(state_dim=32, action_dim=2)
            icm = ICMTrainer(config, device='cpu')
            
            # Train for changes
            states = torch.randn(16, 32)
            actions = torch.randn(16, 2)
            next_states = torch.randn(16, 32)
            
            icm.train_step(states, actions, next_states)
            
            # Save checkpoint
            checkpoint_path = os.path.join(temp_dir, 'icm_checkpoint.pth')
            icm.save_checkpoint(checkpoint_path)
            
            # Load checkpoint
            icm2 = ICMTrainer(config, device='cpu')
            icm2.load_checkpoint(checkpoint_path)
            
            # Verify models have same parameters
            for p1, p2 in zip(icm.feature_encoder.parameters(), icm2.feature_encoder.parameters()):
                assert torch.allclose(p1, p2)


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])