# ML-PPO: Enterprise Proximal Policy Optimization

[![CI](https://github.com/KeepALifeUS/ml-ppo/actions/workflows/ci.yml/badge.svg)](https://github.com/KeepALifeUS/ml-ppo/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Comprehensive PPO implementation for crypto trading.

## Features

### Core PPO Algorithm

- **Standard PPO**: Clipped surrogate objective with GAE
- **PPO2**: Enhanced version with improvements
- **Adaptive Clipping**: Dynamic clipping parameters
- **KL Penalty**: Alternative to clipping for policy updates
- **Natural Gradients**: More principled policy updates

### Network Architectures

- **Actor-Critic**: Shared and separate network options
- **Multi-Modal**: Support for different input types
- **CNN Integration**: For price chart processing
- **LSTM Support**: For sequential data
- **Attention Mechanisms**: Multi-head attention for complex patterns
- **Crypto-Specific**: Specialized networks for trading

### Advantage Estimation

- **GAE (Generalized Advantage Estimation)**: λ-return advantages
- **TD(λ)**: Temporal difference learning with eligibility traces
- **Multi-Step**: N-step returns for robust estimation
- **Adaptive**: Dynamic λ parameter adjustment
- **Risk-Adjusted**: Risk-aware advantage computation

### Training Infrastructure

- **Distributed Training**: Multi-process data collection
- **Asynchronous Rollouts**: Parallel environment interaction
- **Checkpointing**: Automatic model saving and recovery
- **Monitoring**: Comprehensive performance tracking
- **Fault Tolerance**: Error recovery and worker management

### Crypto Trading Integration

- **Multi-Asset**: Support for multiple cryptocurrencies
- **Risk Management**: Position sizing and drawdown control
- **Real-Time**: Live trading decision making
- **Backtesting**: Historical data simulation
- **Performance Analytics**: Comprehensive trading metrics

## Installation

```bash
# Install core dependencies
pip install torch torchvision gymnasium numpy pandas

# Install additional dependencies
pip install tensorboard wandb ray[rllib] psutil

# For development
pip install pytest black flake8 mypy isort pre-commit

```

## Architecture

```

ml-ppo/
├── src/
│ ├── core/ # Core PPO algorithms
│ │ ├── ppo.py # Standard PPO
│ │ └── ppo2.py # Enhanced PPO2
│ ├── networks/ # Neural network architectures
│ │ ├── actor_critic.py # Actor-critic networks
│ │ ├── policy_network.py # Policy networks
│ │ └── value_network.py # Value networks
│ ├── advantages/ # Advantage estimation
│ │ ├── gae.py # GAE implementation
│ │ └── td_lambda.py # TD(λ) methods
│ ├── optimization/ # Optimization techniques
│ │ ├── clipped_objective.py # Clipped objectives
│ │ └── kl_penalty.py # KL penalty methods
│ ├── buffers/ # Data storage
│ │ ├── rollout_buffer.py # Rollout buffer
│ │ └── trajectory_buffer.py # Trajectory buffer
│ ├── training/ # Training infrastructure
│ │ ├── ppo_trainer.py # Main trainer
│ │ └── distributed_ppo.py # Distributed training
│ ├── agents/ # Trading agents
│ │ └── ppo_trader.py # Crypto trading agent
│ ├── environments/ # Trading environments
│ │ └── crypto_env.py # Crypto trading env
│ └── utils/ # Utilities
│ ├── normalization.py # Data normalization
│ └── scheduling.py # Parameter scheduling
├── tests/ # Comprehensive tests
└── docs/ # Documentation

```

## Quick Start

### Basic PPO Training

```python
from src.core.ppo import PPOAlgorithm, PPOConfig
from src.networks.actor_critic import ActorCriticNetwork, ActorCriticConfig
from src.environments.crypto_env import CryptoTradingEnvironment

# Configure environment
env = CryptoTradingEnvironment()

# Configure network
network_config = ActorCriticConfig(
 obs_dim=env.observation_space.shape[0],
 action_dim=env.action_space.shape[0],
 action_type="continuous"
)
actor_critic = ActorCriticNetwork(network_config)

# Configure PPO
ppo_config = PPOConfig(
 learning_rate=3e-4,
 gamma=0.99,
 gae_lambda=0.95,
 clip_range=0.2
)
ppo = PPOAlgorithm(actor_critic, ppo_config)

# Training loop
for episode in range(1000):
 obs, _ = env.reset()
 done = False

 while not done:
 with torch.no_grad():
 action_dist, value = actor_critic(torch.tensor(obs).unsqueeze(0))
 action = action_dist.sample()

 obs, reward, done, _, info = env.step(action.numpy())

```

### Comprehensive Training with Trainer

```python
from src.training.ppo_trainer import PPOTrainer, PPOTrainerConfig

# Configure training
config = PPOTrainerConfig(
 total_timesteps=1_000_000,
 rollout_steps=2048,
 batch_size=64,
 num_envs=4,

 # Algorithm
 algorithm="ppo",
 learning_rate=3e-4,

 # Environment
 env_type="crypto_trading",

 # Monitoring
 use_wandb=True,
 wandb_project="ppo-crypto-trading",

 # Checkpointing
 save_interval=100,
 checkpoint_dir="./checkpoints"
)

# Create trainer
trainer = PPOTrainer(config)

# Start training
results = trainer.train()

print(f"Training completed!")
print(f"Final reward: {results['final_reward']:.2f}")
print(f"Training time: {results['total_time']:.1f} seconds")

```

### Distributed Training

```python
from src.training.distributed_ppo import DistributedPPOTrainer, DistributedPPOConfig

# Configure distributed training
config = DistributedPPOConfig(
 total_timesteps=5_000_000,
 world_size=4,
 rollout_workers=8,
 async_rollouts=True,

 # Performance optimization
 use_gpu=True,
 gradient_compression=True,

 # Fault tolerance
 auto_restart_failed_workers=True,
 max_worker_failures=3
)

# Create distributed trainer
trainer = DistributedPPOTrainer(config)

# Start distributed training
results = trainer.train()

```

### Crypto Trading Agent

```python
from src.agents.ppo_trader import PPOTrader, PPOTraderConfig

# Configure trading agent
config = PPOTraderConfig(
 assets=["BTC", "ETH", "BNB"],
 max_position_size=0.3,
 stop_loss_threshold=0.05,
 take_profit_threshold=0.10,

 # Risk management
 max_drawdown=0.15,
 portfolio_heat=0.02,

 # Model
 model_path="./models/ppo_crypto_model.pt",

 # Real-time trading
 realtime_updates=True,
 update_frequency=60
)

# Create trader
trader = PPOTrader(config)

# Get trading decision
decision = trader.get_trading_decision("BTC", current_price=50000.0)

if decision["action"] != "hold":
 success = trader.execute_trading_decision("BTC", decision)
 if success:
 print(f"Executed {decision['action']} for BTC")

```

### Advanced GAE Configuration

```python
from src.advantages.gae import GAE, GAEConfig

# Configure GAE
gae_config = GAEConfig(
 gamma=0.99,
 gae_lambda=0.95,
 normalize_advantages=True,
 adaptive_lambda=True # Dynamic λ adjustment
)

gae = GAE(gae_config)

# Compute advantages
advantages, returns = gae.compute_advantages_and_returns(
 rewards=torch.tensor(episode_rewards),
 values=torch.tensor(episode_values),
 dones=torch.tensor(episode_dones)
)

```

## Configuration

### PPO Configuration

```python
@dataclass
class PPOConfig:
 # Core parameters
 learning_rate: float = 3e-4
 gamma: float = 0.99
 gae_lambda: float = 0.95
 clip_range: float = 0.2

 # Training
 n_epochs: int = 10
 batch_size: int = 64
 max_grad_norm: float = 0.5

 # Regularization
 ent_coef: float = 0.01
 vf_coef: float = 0.5

 # Advanced
 target_kl: Optional[float] = 0.01
 normalize_advantage: bool = True

```

### Network Configuration

```python
@dataclass
class ActorCriticConfig:
 # Architecture
 shared_backbone: bool = True
 hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
 activation: str = "tanh"

 # Input/Output
 obs_dim: int = 64
 action_dim: int = 4
 action_type: str = "continuous"

 # CNN (for price charts)
 use_cnn: bool = False
 cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64])

 # LSTM (for sequences)
 use_lstm: bool = False
 lstm_hidden_size: int = 128

 # Attention
 use_attention: bool = False
 attention_heads: int = 8

 # Multi-asset
 multi_asset: bool = False
 num_assets: int = 1

```

### Crypto Environment Configuration

```python
@dataclass
class CryptoEnvConfig:
 # Trading parameters
 initial_balance: float = 10000.0
 assets: List[str] = field(default_factory=lambda: ["BTC", "ETH"])
 transaction_cost: float = 0.001
 max_position_size: float = 1.0

 # Market dynamics
 volatility_factor: float = 1.0
 trend_strength: float = 0.1

 # Risk management
 liquidation_threshold: float = 0.8
 max_drawdown_limit: float = 0.5

 # Features
 include_technical_indicators: bool = True

```

## Monitoring and Logging

### Weights & Biases Integration

```python
config = PPOTrainerConfig(
 use_wandb=True,
 wandb_project="crypto-ppo-trading",
 wandb_entity="your-team",
 wandb_tags=["ppo", "crypto", "production"]
)

```

### TensorBoard Logging

```python
# Automatic logging of training metrics
# View with: tensorboard --logdir ./runs

```

### Performance Metrics

The system tracks comprehensive metrics:

- **Training**: Loss, KL divergence, clip fraction, entropy
- **Environment**: Episode reward, length, success rate
- **Trading**: Portfolio value, Sharpe ratio, max drawdown
- **System**: FPS, memory usage, GPU utilization

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_ppo.py::TestPPOCore -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Performance benchmarks
python -m pytest tests/test_ppo.py::TestPerformance -v

```

## Advanced Features

### Adaptive PPO

```python
from src.core.ppo2 import PPO2Algorithm, PPO2Config

config = PPO2Config(
 adaptive_clipping=True,
 use_multi_step=True,
 dynamic_lambda=True,
 mixed_precision=True # Faster training
)

ppo2 = PPO2Algorithm(actor_critic, config)

```

### Risk-Adjusted GAE

```python
from src.advantages.gae import RiskAdjustedGAE

risk_gae = RiskAdjustedGAE(
 config=gae_config,
 risk_adjustment_factor=0.1,
 volatility_window=20
)

advantages, returns = risk_gae.compute_advantages_and_returns(
 rewards, values, dones,
 prices=price_data,
 positions=position_data
)

```

### Natural Policy Gradients

```python
from src.optimization.kl_penalty import NaturalPolicyGradientKL

natural_ppo = NaturalPolicyGradientKL(kl_config)
natural_gradient = natural_ppo.compute_natural_policy_gradient(
 policy_gradient, action_dist
)

```

## Performance Optimization

### GPU Acceleration

```python
config = PPOTrainerConfig(
 device="cuda",
 pin_memory=True,
 non_blocking_transfer=True
)

```

### Mixed Precision Training

```python
config = PPO2Config(
 mixed_precision=True,
 gradient_accumulation_steps=4
)

```

### Distributed Training

```python
config = DistributedPPOConfig(
 world_size=8,
 backend="nccl",
 gradient_compression=True
)

```

## Production Deployment

### Model Serving

```python
# Save trained model
trainer.save_model("./models/ppo_crypto.pt")

# Load for inference
trader = PPOTrader(config)
trader.load_model("./models/ppo_crypto.pt")

# Real-time trading
async def trading_loop():
 while True:
 for asset in assets:
 decision = trader.get_trading_decision(asset, get_current_price(asset))
 if decision["action"] != "hold":
 execute_trade(asset, decision)

 await asyncio.sleep(60) # Update every minute

```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["python", "-m", "src.agents.ppo_trader"]

```

### Monitoring in Production

```python
config = PPOTraderConfig(
 monitoring_enabled=True,
 alert_thresholds={
 "max_loss": -0.10,
 "max_drawdown": -0.15,
 "min_sharpe": 0.5
 }
)

```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/`
5. Run linting: `black src/ tests/ && flake8 src/ tests/`
6. Commit changes: `git commit -am 'Add new feature'`
7. Push to branch: `git push origin feature/new-feature`
8. Create a Pull Request

## References

### Academic Papers

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) - Schulman et al., 2017
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - Schulman et al., 2016
- [Implementation Matters in Deep RL](https://arxiv.org/abs/2005.12729) - Engstrom et al., 2020

### Patterns

- Enterprise-grade error handling and recovery
- Production monitoring and alerting
- Scalable distributed training
- Performance optimization techniques

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Version History

- **v1.0.0** - Initial release with core PPO implementation
- **v1.1.0** - Added PPO2 with advanced features
- **v1.2.0** - Crypto trading integration
- **v1.3.0** - Distributed training support
- **v1.4.0** - Production deployment features

## Tips and Best Practices

### Training Tips

1. Start with smaller networks and scale up
2. Use learning rate scheduling for better convergence
3. Monitor KL divergence to detect training instability
4. Adjust clip range based on policy update magnitude
5. Use GAE λ parameter to balance bias/variance

### Crypto Trading Tips

1. Start with paper trading to validate strategies
2. Implement proper risk management from day one
3. Monitor portfolio heat and correlation
4. Use multiple timeframes for better decisions
5. Backtest thoroughly before live deployment

### Performance Tips

1. Use GPU acceleration for faster training
2. Enable mixed precision for memory efficiency
3. Use distributed training for large-scale experiments
4. Profile your code to identify bottlenecks
5. Cache frequently accessed data

---

Built for the crypto trading community.

## Support

For questions and support, please open an issue on GitHub.
