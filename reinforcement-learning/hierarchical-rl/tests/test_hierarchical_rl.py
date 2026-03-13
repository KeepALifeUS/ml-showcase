"""
Comprehensive Test Suite for Hierarchical RL System
Testing all components hierarchical system training with reinforcement.

enterprise Pattern:
- Production-ready testing suite with comprehensive coverage
- Performance benchmarking for real-time trading systems
- Integration testing for complex hierarchical components
"""

import pytest
import numpy as np
import torch
import asyncio
import tempfile
import os
from typing import Dict, List, Any
import time

# Imports tested components
from ..src.frameworks.options import (
    OptionsFramework, Option, create_trend_following_option,
    TradingInitiationSet, TradingTerminationCondition
)
from ..src.frameworks.ham import (
    HAMFramework, HAMMachine, create_trend_following_ham,
    ChoiceNode, ActionNode, StopNode
)
from ..src.frameworks.maxq import (
    MAXQHierarchy, create_trading_maxq_hierarchy,
    CompositeNode, PrimitiveNode
)
from ..src.frameworks.hac import (
    HACAgent, create_crypto_hac_agent,
    HACTradingEnvironment
)
from ..src.policies.meta_policy import (
    MetaPolicy, create_crypto_meta_policy,
    StrategyContext
)
from ..src.policies.skill_policy import (
    create_skill_library, SkillComposer,
    OrderExecutionSkill, RiskControlSkill
)
from ..src.policies.option_policy import (
    OptionPolicy, create_trend_following_option as create_trend_option,
    OptionComposer
)
from ..src.discovery.subgoal_discovery import (
    SubgoalDiscoveryEngine, create_crypto_subgoal_discovery
)
from ..src.discovery.skill_discovery import (
    SkillDiscoveryEngine, create_crypto_skill_discovery
)
from ..src.discovery.bottleneck_detection import (
    BottleneckDetectionEngine, create_crypto_bottleneck_detector
)
from ..src.agents.hierarchical_trader import (
    HierarchicalTradingAgent, create_trend_following_agent,
    TradingState
)
from ..src.agents.option_critic import (
    OptionCriticAgent, TradingOptionCriticAgent,
    create_crypto_option_critic, OptionCriticConfig
)
from ..src.training.hierarchical_trainer import (
    HierarchicalTrainer, TrainingConfig,
    create_option_critic_trainer, TradingEnvironment
)


class TestOptionsFramework:
    """Tests for Options Framework"""
    
    def test_option_creation(self):
        """Test creation options"""
        option = create_trend_following_option()
        
        assert option.option_id == "trend_following"
        assert option.status.value == "inactive"
        assert hasattr(option, 'initiation_set')
        assert hasattr(option, 'policy')
        assert hasattr(option, 'termination_condition')
    
    def test_initiation_set(self):
        """Test set initiation"""
        initiation = TradingInitiationSet()
        
        # Test with valid conditions
        state = np.array([0.03, 2000000, 0.04, 0.01])  # price_change, volume, volatility, pnl
        assert initiation.can_initiate(state) == True
        
        # Test with invalid conditions
        state_invalid = np.array([0.001, 500000, 0.15, 0.01])
        assert initiation.can_initiate(state_invalid) == False
        
        # Test probability initiation
        prob = initiation.initiation_probability(state)
        assert 0.0 <= prob <= 1.0
    
    def test_termination_condition(self):
        """Test conditions completion"""
        termination = TradingTerminationCondition()
        
        # Test on reaching profit target
        state_profit = np.array([0.01, 1000000, 0.02, 0.06])
        assert termination.should_terminate(state_profit, 10, 0.06) == True
        
        # Test on stop loss
        state_loss = np.array([0.01, 1000000, 0.02, -0.03])
        assert termination.should_terminate(state_loss, 10, -0.03) == True
        
        # Test on timeout
        state_timeout = np.array([0.01, 1000000, 0.02, 0.01])
        assert termination.should_terminate(state_timeout, 150, 0.01) == True
    
    @pytest.mark.asyncio
    async def test_option_execution(self):
        """Test execution options"""
        option = create_trend_following_option()
        state = np.array([0.03, 2000000, 0.04, 0.0])
        
        # Initiate option
        success = option.initiate(state)
        assert success == True
        assert option.status.value == "active"
        
        # Execute several steps
        for _ in range(5):
            action, terminated = option.execute_step(state)
            assert isinstance(action, (int, np.integer))
            if terminated:
                break
        
        # Complete option
        result = option.terminate(0.02, True)
        assert result.success == True
        assert result.total_reward > 0
    
    def test_options_framework(self):
        """Test framework options"""
        framework = OptionsFramework()
        option = create_trend_following_option()
        
        # Register option
        framework.register_option(option)
        assert len(framework.options) == 1
        
        # Retrieve available options
        state = np.array([0.03, 2000000, 0.04, 0.0])
        available = framework.get_available_options(state)
        assert len(available) >= 1
        
        # Select option
        selected = framework.select_option(state)
        assert selected is not None


class TestHAMFramework:
    """Tests for HAM Framework"""
    
    def test_ham_machine_creation(self):
        """Test creation HAM machines"""
        machine = create_trend_following_ham()
        
        assert machine.machine_id == "trend_following_ham"
        assert machine.start_node == "check_trend"
        assert len(machine.nodes) > 0
        assert len(machine.transitions) > 0
    
    def test_ham_node_types(self):
        """Test various types nodes"""
        # Choice Node
        policy = torch.nn.Sequential(
            torch.nn.Linear(10, 3),
            torch.nn.Softmax(dim=-1)
        )
        choice_node = ChoiceNode("test_choice", policy, 3)
        assert choice_node.node_type.value == "choice"
        
        # Action Node
        action_node = ActionNode("test_action", 1)
        assert action_node.node_type.value == "action"
        
        # Stop Node
        stop_node = StopNode()
        assert stop_node.node_type.value == "stop"
    
    @pytest.mark.asyncio
    async def test_ham_execution(self):
        """Test execution HAM machines"""
        machine = create_trend_following_ham()
        
        # Validate machine
        errors = machine.validate()
        assert len(errors) == 0
        
        # Execute machine
        initial_state = np.random.randn(10)
        context = await machine.execute(initial_state, max_steps=50)
        
        assert context.step_count > 0
        assert context.machine_id == "trend_following_ham"
    
    def test_ham_framework(self):
        """Test HAM framework"""
        framework = HAMFramework()
        machine = create_trend_following_ham()
        
        # Register machine
        framework.register_machine(machine)
        assert len(framework.machines) == 1
        
        # Retrieve statistics
        stats = framework.get_machine_statistics()
        assert "total_machines" in stats
        assert stats["total_machines"] == 1


class TestMAXQHierarchy:
    """Tests for MAXQ Hierarchy"""
    
    def test_maxq_hierarchy_creation(self):
        """Test creation MAXQ hierarchy"""
        hierarchy = create_trading_maxq_hierarchy()
        
        assert hierarchy.root_task == "trading_root"
        assert len(hierarchy.nodes) > 0
    
    def test_composite_node(self):
        """Test composite node"""
        def always_false(state):
            return False
        
        node = CompositeNode("test_composite", always_false)
        assert node.node_type.value == "composite"
        assert not node.is_terminal(None)
    
    def test_primitive_node(self):
        """Test primitive node"""
        node = PrimitiveNode("test_primitive", 1)
        assert node.node_type.value == "primitive"
        assert node.action_id == 1
    
    @pytest.mark.asyncio
    async def test_maxq_execution(self):
        """Test execution MAXQ hierarchy"""
        hierarchy = create_trading_maxq_hierarchy()
        
        # Create initial state
        from ..src.frameworks.maxq import MAXQState
        initial_state = MAXQState(
            environment_state=np.random.randn(10),
            task_stack=[],
            subtask_completion={}
        )
        
        # Execute task
        trajectory, reward = await hierarchy.execute_task("trading_root", initial_state, max_steps=20)
        
        assert len(trajectory) > 0
        assert isinstance(reward, float)


class TestHACAgent:
    """Tests for HAC Agent"""
    
    def test_hac_agent_creation(self):
        """Test creation HAC agent"""
        agent = create_crypto_hac_agent()
        
        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.num_levels == 3
        assert len(agent.actors) == 3
        assert len(agent.critics) == 3
    
    def test_hac_action_selection(self):
        """Test selection actions"""
        agent = create_crypto_hac_agent()
        state = np.random.randn(10)
        goal = np.array([0.02, 0.0, 0.0, 0.01])
        
        # Test hierarchical selection actions
        primitive_action, subgoals = agent.hierarchical_action_selection(state, goal)
        
        assert isinstance(primitive_action, np.ndarray)
        assert len(primitive_action) == agent.action_dim
        assert isinstance(subgoals, list)
    
    def test_hac_training_environment(self):
        """Test trading environment for HAC"""
        env = HACTradingEnvironment(None)
        
        state, goal = env.reset()
        assert len(state) > 0
        assert len(goal) == env.goal_space_dim
        
        # Test step
        action = np.random.randn(3)
        next_state, reward, done, info = env.step(action)
        
        assert len(next_state) == len(state)
        assert isinstance(reward, float)
        assert isinstance(done, bool)


class TestMetaPolicy:
    """Tests for Meta Policy"""
    
    def test_meta_policy_creation(self):
        """Test creation meta-policy"""
        policy = create_crypto_meta_policy()
        
        assert policy.state_dim == 15
        assert len(policy.strategy_registry) > 0
        assert hasattr(policy, 'network')
    
    def test_strategy_selection(self):
        """Test selection strategies"""
        policy = create_crypto_meta_policy()
        
        context = StrategyContext(
            market_state=np.random.randn(15),
            volatility=0.03,
            volume=1500000,
            trend_strength=0.2,
            spread=0.002,
            time_of_day=14,
            market_regime="trending",
            risk_level=0.3,
            portfolio_state={"cash": 0.5, "position": 0.5}
        )
        
        selection = policy.select_strategy(context)
        
        assert hasattr(selection, 'strategy_type')
        assert hasattr(selection, 'confidence')
        assert 0.0 <= selection.confidence <= 1.0
        assert len(selection.parameters) > 0


class TestSkillPolicies:
    """Tests for Skill Policies"""
    
    def test_skill_library_creation(self):
        """Test creation libraries skills"""
        skills = create_skill_library()
        
        assert len(skills) > 0
        assert "order_execution" in skills
        assert "risk_control" in skills
    
    @pytest.mark.asyncio
    async def test_skill_execution(self):
        """Test execution skill"""
        skills = create_skill_library()
        order_skill = skills["order_execution"]
        
        from ..src.policies.skill_policy import SkillContext
        context = SkillContext(
            current_state=np.random.randn(15),
            target_goal=np.array([0.02, 0.0, 0.0, 0.01, 0.0]),
            constraints={},
            environment_info={}
        )
        
        result = await order_skill.execute(context)
        
        assert hasattr(result, 'skill_type')
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_time')
    
    def test_skill_composer(self):
        """Test compositor skills"""
        skills = create_skill_library()
        composer = SkillComposer(skills)
        
        assert len(composer.skills) > 0


class TestOptionPolicies:
    """Tests for Option Policies"""
    
    def test_option_policy_creation(self):
        """Test creation policy options"""
        option = create_trend_option()
        
        assert option.option_id == "trend_following"
        assert hasattr(option, 'network')
        assert option.state_dim == 15
        assert option.action_dim == 3
    
    @pytest.mark.asyncio
    async def test_option_policy_execution(self):
        """Test execution policy options"""
        option = create_trend_option()
        
        async def mock_environment_step(action):
            return (
                np.random.randn(15),  # next_state
                np.random.normal(0.001, 0.01),  # reward
                False,  # done
                {}  # info
            )
        
        initial_state = np.random.randn(15)
        result = await option.execute_option(initial_state, mock_environment_step, max_steps=10)
        
        assert hasattr(result, 'success')
        assert hasattr(result, 'total_steps')
        assert result.total_steps > 0


class TestDiscoveryEngines:
    """Tests for Discovery Engines"""
    
    def test_subgoal_discovery(self):
        """Test detection subgoals"""
        discovery = create_crypto_subgoal_discovery()
        
        # Create test trajectory
        trajectories = [
            [np.random.randn(15) for _ in range(20)]
            for _ in range(5)
        ]
        rewards_list = [
            [np.random.normal(0.001, 0.01) for _ in range(20)]
            for _ in range(5)
        ]
        
        subgoals = discovery.discover_subgoals(trajectories, rewards_list)
        
        assert isinstance(subgoals, list)
        # Can be empty for random data
    
    def test_skill_discovery(self):
        """Test detection skills"""
        discovery = create_crypto_skill_discovery()
        
        # Create test data
        trajectories = [
            [np.random.randn(15) for _ in range(20)]
            for _ in range(5)
        ]
        actions_list = [
            [np.random.randint(0, 3, 3) for _ in range(19)]
            for _ in range(5)
        ]
        rewards_list = [
            [np.random.normal(0.001, 0.01) for _ in range(20)]
            for _ in range(5)
        ]
        
        skills = discovery.discover_skills(trajectories, actions_list, rewards_list)
        
        assert isinstance(skills, list)
    
    def test_bottleneck_detection(self):
        """Test detection narrow places"""
        detector = create_crypto_bottleneck_detector()
        
        # Create test data
        trajectories = [
            [np.random.randn(15) for _ in range(15)]
            for _ in range(5)
        ]
        actions_list = [
            [np.random.randint(0, 3, 3) for _ in range(14)]
            for _ in range(5)
        ]
        rewards_list = [
            [np.random.normal(0.001, 0.01) for _ in range(15)]
            for _ in range(5)
        ]
        
        bottlenecks = detector.detect_bottlenecks(trajectories, actions_list, rewards_list)
        
        assert isinstance(bottlenecks, list)


class TestHierarchicalAgents:
    """Tests for hierarchical agents"""
    
    def test_hierarchical_trader_creation(self):
        """Test creation hierarchical trader"""
        agent = create_trend_following_agent()
        
        assert agent.state_dim == 20
        assert agent.action_dim == 3
        assert len(agent.active_frameworks) > 0
        assert hasattr(agent, 'meta_policy')
    
    @pytest.mark.asyncio
    async def test_hierarchical_decision_making(self):
        """Test making hierarchical decisions"""
        agent = create_trend_following_agent()
        
        # Create test state
        trading_state = TradingState(
            market_data=np.random.randn(20),
            portfolio_state={"value": 1.0, "position": 0.1, "cash": 0.9},
            risk_metrics={"volatility": 0.02, "trend_strength": 0.1, "risk_level": 0.2},
            execution_context={"volume": 1000000, "spread": 0.001, "timestamp": time.time()},
            timestamp=time.time()
        )
        
        decisions = await agent.make_decision(trading_state)
        
        assert isinstance(decisions, list)
        # Can be empty in dependencies from conditions
    
    def test_option_critic_agent(self):
        """Test Option-Critic agent"""
        agent = create_crypto_option_critic()
        
        assert isinstance(agent, TradingOptionCriticAgent)
        assert agent.config.num_options == 8
        assert agent.config.action_dim == 3
        
        # Test selection options
        state = np.random.randn(20)
        option = agent.select_option(state)
        assert 0 <= option < agent.config.num_options
        
        # Test selection actions
        action = agent.select_action(state, option)
        assert 0 <= action < agent.config.action_dim


class TestTrainingSystem:
    """Tests for system training"""
    
    def test_training_environment(self):
        """Test trading environment"""
        config = TrainingConfig()
        env = TradingEnvironment(config)
        
        # Test reset
        state = env.reset()
        assert hasattr(state, 'market_data')
        assert len(state.market_data) == config.state_dim
        
        # Test step
        action = 1  # Hold
        next_state, reward, done, info = env.step(action)
        
        assert hasattr(next_state, 'market_data')
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_hierarchical_trainer_creation(self):
        """Test creation trainer"""
        trainer = create_option_critic_trainer()
        
        assert isinstance(trainer, HierarchicalTrainer)
        assert len(trainer.agents) == 1
        assert "option_critic" in trainer.agents
    
    def test_training_config(self):
        """Test configuration training"""
        config = TrainingConfig(
            total_episodes=100,
            state_dim=15,
            action_dim=3
        )
        
        assert config.total_episodes == 100
        assert config.state_dim == 15
        assert config.action_dim == 3
        assert len(config.training_phases) > 0


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_training_pipeline(self):
        """Test full pipeline training"""
        # Create minimum configuration for fast test
        config = TrainingConfig(
            total_episodes=10,
            max_episode_steps=20,
            eval_frequency=5,
            save_frequency=5
        )
        
        trainer = HierarchicalTrainer(config)
        
        # Add simple agent
        agent = create_crypto_option_critic(device="cpu")
        trainer.add_agent("test_agent", agent)
        
        # Run short training
        results = trainer.train("test_agent")
        
        assert isinstance(results, dict)
        assert len(trainer.training_stats['episode_rewards']) > 0
    
    def test_model_saving_loading(self):
        """Test saving and loading models"""
        agent = create_crypto_option_critic()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            
            # Save model
            agent.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Create new agent and load model
            new_agent = create_crypto_option_critic()
            new_agent.load_model(model_path)
            
            # Check, that parameters loaded
            assert new_agent.total_steps == agent.total_steps


class TestPerformance:
    """Tests performance"""
    
    def test_decision_latency(self):
        """Test latency making decisions"""
        agent = create_trend_following_agent()
        
        trading_state = TradingState(
            market_data=np.random.randn(20),
            portfolio_state={"value": 1.0},
            risk_metrics={"volatility": 0.02},
            execution_context={"volume": 1000000},
            timestamp=time.time()
        )
        
        start_time = time.time()
        
        # Measure time making decisions
        async def measure_decision():
            decisions = await agent.make_decision(trading_state)
            return decisions
        
        decisions = asyncio.run(measure_decision())
        decision_time = time.time() - start_time
        
        # Check, that time making decisions reasonable (< 100ms)
        assert decision_time < 0.1
    
    def test_memory_usage(self):
        """Test usage memory"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create several agents
        agents = [create_crypto_option_critic() for _ in range(5)]
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Check, that increase memory reasonable (< 500MB)
        assert memory_increase < 500 * 1024 * 1024
    
    def test_training_speed(self):
        """Test speed training"""
        config = TrainingConfig(
            total_episodes=5,
            max_episode_steps=10
        )
        
        trainer = HierarchicalTrainer(config)
        agent = create_crypto_option_critic()
        trainer.add_agent("speed_test", agent)
        
        start_time = time.time()
        trainer.train("speed_test")
        training_time = time.time() - start_time
        
        # Check, that training completes for reasonable time
        episodes_per_second = config.total_episodes / training_time
        assert episodes_per_second > 0.1  # Minimum 0.1 episodes in second


# Configuration pytest
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Configuration test environment"""
    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Configuration logging for tests
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce level logging


@pytest.fixture
def mock_market_data():
    """Fixture for creation test market data"""
    return {
        'prices': np.random.randn(100) * 0.01 + 1.0,
        'volumes': np.random.exponential(1000000, 100),
        'timestamps': np.arange(100)
    }


if __name__ == "__main__":
    # Launch tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])