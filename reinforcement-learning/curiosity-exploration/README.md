# ML Curiosity Exploration System

Enterprise-grade curiosity-driven exploration system for autonomous cryptocurrency trading strategy discovery, built with enterprise patterns for production deployment.

## 🚀 Features

### Core Curiosity Mechanisms

- **Intrinsic Curiosity Module (ICM)** - Forward/inverse dynamics models for controllable exploration
- **Random Network Distillation (RND)** - Exploration through prediction error uncertainty
- **Never Give Up (NGU)** - Persistent exploration with episodic memory

### Advanced Exploration Strategies

- **Count-Based Exploration** - State visitation frequency-based bonuses
- **Prediction-Based Exploration** - Uncertainty quantification with ensemble methods
- **Unified Exploration Bonus System** - Adaptive combination of multiple exploration signals

### Novelty Detection

- **Multi-Method Novelty Detection** - Autoencoder, Isolation Forest, One-Class SVM, LOF
- **Crypto-Specific Novelty** - Market regime-aware anomaly detection
- **Real-Time Detection** - Streaming novelty detection for live trading

### Memory Systems

- **Episodic Memory** - FAISS-based similarity search for experience retrieval
- **Curiosity Replay Buffer** - Prioritized replay based on curiosity rewards
- **Temporal Context Tracking** - Sequential pattern memory

### Trading Agents

- **Curious Trader** - Autonomous strategy discovery through curiosity
- **Exploration Agent** - Pure exploration for strategy space coverage
- **Multi-Strategy Integration** - Adaptive strategy selection and weighting

## 🏗️ Architecture

```

ml-curiosity-exploration/
├── src/
│   ├── curiosity/           # Core curiosity mechanisms
│   │   ├── icm.py          # Intrinsic Curiosity Module
│   │   ├── rnd.py          # Random Network Distillation
│   │   └── ngu.py          # Never Give Up agent
│   ├── exploration/         # Exploration strategies
│   │   ├── count_based.py  # Count-based exploration
│   │   ├── prediction_based.py # Prediction-based exploration
│   │   └── exploration_bonus.py # Unified bonus system
│   ├── novelty/            # Novelty detection
│   │   ├── novelty_detector.py # Multi-method novelty detection
│   │   └── state_visitor.py    # State visitation tracking
│   ├── memory/             # Memory systems
│   │   ├── episodic_memory.py  # Episodic memory with FAISS
│   │   └── curiosity_buffer.py # Prioritized curiosity replay
│   ├── agents/             # Trading agents
│   │   ├── curious_trader.py   # Curious trading agent
│   │   └── exploration_agent.py # Pure exploration agent
│   └── utils/              # Utilities
│       ├── state_encoder.py    # State encoding utilities
│       └── reward_shaping.py   # Reward shaping utilities
├── tests/                  # Comprehensive tests
│   └── test_curiosity.py  # Full test suite
├── README.md              # This file
├── package.json           # Node.js package configuration
└── pyproject.toml         # Python package configuration

```

## 🎯 enterprise patterns

### Scalable Architecture

- **Distributed Training** - Multi-GPU and multi-node support
- **Real-Time Inference** - Low-latency exploration for live trading
- **Memory Management** - Efficient storage and retrieval systems
- **Performance Optimization** - Vectorized operations and caching

### Production Deployment

- **Checkpoint Management** - Robust model saving/loading
- **Metrics Collection** - Comprehensive performance tracking
- **Error Handling** - Graceful degradation and recovery
- **Resource Monitoring** - Memory and compute optimization

### Cloud-Native Design

- **Horizontal Scaling** - Auto-scaling exploration components
- **Service Mesh Integration** - Microservices architecture
- **Event-Driven Processing** - Asynchronous exploration updates
- **Observability** - Distributed tracing and logging

## 🚀 Quick Start

### Installation

```bash
# Install package dependencies
npm install

# Install Python dependencies
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

```

### Basic Usage

```python
from src.curiosity.icm import ICMConfig, ICMTrainer
from src.exploration.exploration_bonus import ExplorationBonusConfig, ExplorationBonusManager
from src.agents.curious_trader import CuriousTraderConfig, CuriousTrader

# Initialize curiosity system
icm_config = ICMConfig(state_dim=256, action_dim=10)
icm_trainer = ICMTrainer(icm_config)

# Initialize exploration bonus manager
bonus_config = ExplorationBonusConfig()
bonus_manager = ExplorationBonusManager(bonus_config)

# Initialize curious trader
trader_config = CuriousTraderConfig(state_dim=256, action_dim=10)
trader = CuriousTrader(trader_config)

# Training loop
for episode in range(1000):
    state = env.reset()

    for step in range(200):
        # Select action with curiosity-driven exploration
        action, exploration_info = trader.select_action(state, exploration=True)

        # Execute action in environment
        next_state, reward, done, info = env.step(action)

        # Update trader with experience
        update_metrics = trader.update(state, action, reward, next_state, done)

        state = next_state
        if done:
            break

```

### Advanced Configuration

```python
# Advanced ICM configuration
icm_config = ICMConfig(
    state_dim=512,
    action_dim=20,
    feature_dim=128,
    hidden_dim=256,
    forward_loss_weight=0.2,
    inverse_loss_weight=0.8,
    curiosity_reward_weight=1.0,
    market_features=100,  # Crypto-specific
    portfolio_features=40,
    risk_features=20
)

# Advanced exploration configuration
exploration_config = ExplorationBonusConfig(
    strategy_weights={
        "count_based": 0.2,
        "prediction_based": 0.3,
        "curiosity_driven": 0.5
    },
    adaptive_weights=True,
    market_regime_bonus_scaling={
        "bull": 1.0,
        "bear": 1.5,
        "volatile": 2.0
    },
    risk_adjusted_exploration=True
)

```

## 📊 Performance Metrics

### Exploration Efficiency

- **Coverage Rate** - Unique states discovered per episode
- **Exploration Diversity** - Action space coverage uniformity
- **Novelty Detection Rate** - Novel state identification accuracy
- **Strategy Discovery** - New profitable strategies found

### Trading Performance

- **Portfolio Return** - Risk-adjusted returns
- **Sharpe Ratio** - Risk-adjusted performance metric
- **Max Drawdown** - Maximum portfolio decline
- **Trade Frequency** - Number of trades executed

### System Performance

- **Inference Latency** - Real-time decision making speed
- **Memory Usage** - Efficient resource utilization
- **Throughput** - Exploration steps per second
- **Scalability** - Performance with system size

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_curiosity.py::TestICMSystem -v
pytest tests/test_curiosity.py::TestNoveltyDetection -v
pytest tests/test_curiosity.py::TestIntegration -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance tests
pytest tests/ -m "not slow"

```

## 🔧 Configuration

### Environment Variables

```bash
export CUDA_DEVICE="0"                    # GPU device
export CURIOSITY_LOG_LEVEL="INFO"        # Logging level
export EXPLORATION_CACHE_SIZE="10000"    # Cache size
export FAISS_GPU_ENABLED="true"         # Enable GPU FAISS

```

### Configuration Files

```yaml
# config/curiosity.yaml
curiosity:
  icm:
    state_dim: 256
    action_dim: 10
    learning_rate: 1e-4

  rnd:
    target_network_dim: 512
    predictor_network_dim: 512
    intrinsic_reward_coeff: 0.1

  exploration:
    strategy_weights:
      count_based: 0.3
      prediction_based: 0.4
      curiosity_driven: 0.3

```

## 📈 Monitoring & Observability

### Metrics Collection

- **Curiosity Rewards** - ICM, RND prediction errors
- **Exploration Coverage** - State space exploration metrics
- **Model Performance** - Training loss and convergence
- **System Resources** - Memory, CPU, GPU utilization

### Visualization

```python
# Built-in visualization
from src.novelty.state_visitor import StateVisitor

visitor = StateVisitor(config, state_dim=256)
# ... collect exploration data ...
visitor.visualize_exploration_patterns("exploration_analysis.png")

```

### Logging

```python
import logging
from src.curiosity.icm import ICMTrainer

# Structured logging
logger = logging.getLogger("curiosity.exploration")
logger.setLevel(logging.INFO)

# ICM training with logging
icm = ICMTrainer(config)
metrics = icm.train_step(states, actions, next_states)
logger.info("ICM training step", extra=metrics)

```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-cuda

WORKDIR /app
COPY . .

RUN pip install -e .
CMD ["python", "-m", "src.agents.curious_trader"]

```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: curiosity-exploration
spec:
  replicas: 3
  selector:
    matchLabels:
      app: curiosity-exploration
  template:
    metadata:
      labels:
        app: curiosity-exploration
    spec:
      containers:
        - name: curiosity
          image: ml-framework/curiosity-exploration:latest
          resources:
            requests:
              memory: '4Gi'
              cpu: '2'
              nvidia.com/gpu: '1'

```

### Scaling Configuration

```python
# Distributed training setup
from src.curiosity.icm import ICMTrainer

config = ICMConfig(
    distributed_training=True,
    checkpoint_interval=1000,
    metrics_enabled=True
)

# Multi-GPU training
trainer = ICMTrainer(config, device='cuda')
trainer.setup_distributed()

```

## 🔗 Integration

### With Existing Trading Systems

```python
# Integration with existing RL agents
class CuriosityEnhancedAgent(BaseAgent):
    def __init__(self, base_agent, curiosity_config):
        self.base_agent = base_agent
        self.curiosity_system = ICMTrainer(curiosity_config)

    def select_action(self, state):
        base_action = self.base_agent.select_action(state)
        curiosity_bonus = self.curiosity_system.get_curiosity_reward(...)

        # Combine base policy with curiosity exploration
        return self.combine_policies(base_action, curiosity_bonus)

```

### API Integration

```python
from fastapi import FastAPI
from src.agents.curious_trader import CuriousTrader

app = FastAPI()
trader = CuriousTrader(config)

@app.post("/explore")
async def explore_action(state: StateRequest):
    action, exploration_info = trader.select_action(
        state.data, exploration=True
    )
    return {"action": action.tolist(), "info": exploration_info}

```

## 📚 Research Background

This system implements state-of-the-art curiosity-driven exploration methods:

- **Intrinsic Curiosity Module** - Pathak et al. (2017)
- **Random Network Distillation** - Burda et al. (2018)
- **Never Give Up** - Badia et al. (2020)
- **Count-Based Exploration** - Strehl & Littman (2008)
- **Ensemble Uncertainty** - Osband et al. (2016)

Enhanced for cryptocurrency trading with:

- Market regime awareness
- Portfolio risk management
- Real-time processing capabilities
- Enterprise-grade scalability

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-exploration`)
3. Make changes with tests
4. Run test suite (`pytest tests/ -v`)
5. Submit pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run type checking
mypy src/

```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- enterprise framework for enterprise patterns
- PyTorch team for deep learning framework
- FAISS team for efficient similarity search
- OpenAI for reinforcement learning research

## Support

For questions and support, please open an issue on GitHub.
