# ML Gym Environments for Crypto Trading

Enterprise-grade OpenAI Gym environments for cryptocurrency trading with enterprise patterns and comprehensive sentiment analysis integration.

## Overview

This package implements sophisticated trading environments for reinforcement learning agents, featuring:

- **Multi-Asset Trading**: Support for multiple cryptocurrency pairs
- **Sentiment Analysis**: Integration of market sentiment from multiple sources
- **Market Microstructure**: Realistic order book simulation and market impact
- **Advanced Risk Management**: Comprehensive risk metrics and position sizing
- **Enterprise Patterns**: Production-ready logging, monitoring, and error handling
- **Cloud-Native Compatibility**: Modern cloud-native architectural patterns

## Installation

```bash
# From project root
cd packages/ml-gym-environments
pip install -e .

# Or install dependencies directly
pip install gymnasium numpy pandas scipy
```

## Architecture

```
ml-gym-environments/
├── src/
│   ├── environments/          # Core trading environments
│   │   ├── base_trading_env.py       # Abstract base class
│   │   └── crypto_trading_env.py     # Crypto-specific environment
│   ├── spaces/                # Observation & action spaces
│   │   ├── observations.py           # Multi-modal observations
│   │   └── actions.py                # Trading action definitions
│   ├── rewards/               # Reward function strategies
│   │   ├── profit_reward.py          # Profit-based rewards
│   │   └── sharpe_reward.py          # Risk-adjusted rewards
│   ├── simulation/            # Market simulation engine
│   │   ├── market_simulator.py       # Advanced market dynamics
│   │   └── order_book.py             # Order book simulation
│   ├── data/                  # Data management
│   │   ├── data_stream.py            # Real-time data streaming
│   │   └── data_preprocessor.py      # Feature engineering
│   ├── utils/                 # Utility functions
│   │   ├── logger.py                 # Structured logging
│   │   ├── risk_metrics.py           # Risk calculations
│   │   └── indicators.py             # Technical indicators
│   └── wrappers/              # Environment wrappers
└── tests/                     # Comprehensive test suite
```

## Quick Start

### Basic Environment

```python
from ml_gym_environments import create_crypto_env, CryptoTradingConfig

# Create basic crypto environment
config = CryptoTradingConfig(
    assets=["BTC", "ETH", "BNB"],
    initial_balance=10000.0,
    max_steps=1000
)

env = create_crypto_env(config)

# Standard Gym interface
observation, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

### Sentiment-Enhanced Environment

```python
from ml_gym_environments import create_sentiment_crypto_env

# Create environment with sentiment analysis
env = create_sentiment_crypto_env(
    assets=["BTC", "ETH"],
    sentiment_sources=["twitter", "reddit", "news", "fear_greed_index"],
    sentiment_weight=0.2
)

# Environment includes sentiment in observations
observation, info = env.reset()
print(f"Observation shape: {observation.shape}")
print(f"Market info: {env.get_market_info()}")
```

### Advanced Configuration

```python
from ml_gym_environments import (
    CryptoTradingEnvironment,
    CryptoTradingConfig,
    ObservationConfig,
    ActionConfig,
    ActionMode
)

# Comprehensive configuration
obs_config = ObservationConfig(
    include_sentiment=True,
    include_order_book=True,
    include_technical_indicators=True,
    technical_indicators=["sma_20", "rsi_14", "macd", "bb_upper"]
)

action_config = ActionConfig(
    action_mode=ActionMode.CONTINUOUS,
    max_position_size=0.3,
    enable_limit_orders=True,
    enable_position_sizing=True,
    position_sizing_method="kelly"
)

trading_config = CryptoTradingConfig(
    assets=["BTC", "ETH", "BNB", "ADA"],
    enable_sentiment_signals=True,
    enable_order_book=True,
    enable_futures_trading=True,
    data_source="binance",
    observation_config=obs_config,
    action_config=action_config
)

env = CryptoTradingEnvironment(trading_config)
```

## Key Features

### 1. Multi-Modal Observations

The environments provide rich, multi-modal observations including:

- **Price Data**: OHLCV data with configurable history length
- **Technical Indicators**: 20+ technical indicators
- **Sentiment Data**: Multi-source sentiment analysis
- **Market Microstructure**: Order book depth, bid-ask spreads
- **Portfolio State**: Current positions, balance, metrics
- **Market Regime**: Detected market conditions

### 2. Sophisticated Action Spaces

Multiple action space modes:

- **Discrete**: Simple buy/sell/hold actions
- **Continuous**: Position sizing with risk constraints
- **Portfolio**: Target portfolio allocation
- **Orders**: Advanced order management (limit, stop, etc.)

### 3. Advanced Reward Functions

Enterprise-grade reward strategies:

```python
from ml_gym_environments import (
    create_simple_profit_reward,
    create_risk_adjusted_profit_reward,
    create_sharpe_reward,
    create_sortino_reward
)

# Profit-based reward
profit_reward = create_simple_profit_reward(
    profit_scale=100.0,
    enable_risk_penalty=True
)

# Risk-adjusted reward
sharpe_reward = create_sharpe_reward(
    lookback_window=50,
    target_sharpe=1.5
)
```

### 4. Market Simulation

Realistic market simulation with:

- **Market Impact**: Sophisticated impact models
- **Slippage**: Volatility and size-based slippage
- **Liquidity**: Dynamic liquidity modeling
- **Latency**: Execution latency simulation
- **Partial Fills**: Realistic order execution

### 5. Enterprise Logging

Structured logging for production:

```python
from ml_gym_environments import create_environment_logger

logger = create_environment_logger("crypto_env_001")

# Automatic trade logging
logger.log_trade_execution(
    asset="BTC",
    side="buy",
    quantity=0.1,
    price=45000.0,
    fees=45.0
)

# Performance metrics
logger.log_performance_metric("sharpe_ratio", 1.25)

# Export for analysis
logger.export_events("trading_events.json")
```

## Observation Space

The observation space includes multiple feature categories:

| Category           | Features              | Description                  |
| ------------------ | --------------------- | ---------------------------- |
| **Prices**         | OHLCV history         | Historical price data        |
| **Technical**      | SMA, EMA, RSI, etc.   | Technical indicators         |
| **Sentiment**      | Twitter, Reddit, News | Sentiment scores             |
| **Microstructure** | Order book, trades    | Market microstructure        |
| **Portfolio**      | Positions, balance    | Current portfolio state      |
| **Regime**         | Bull/Bear/Volatile    | Market regime classification |

## Performance Features

- **Async Support**: Real-time trading compatibility
- **Vectorized Environments**: Parallel execution
- **Memory Efficient**: Optimized data structures
- **Configurable**: Extensive configuration options
- **Monitoring**: Built-in performance monitoring

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_environments.py -v
pytest tests/test_rewards.py -v
pytest tests/test_simulation.py -v

# Run with coverage
pytest --cov=ml_gym_environments tests/
```

## Example Training Loop

```python
import gymnasium as gym
from ml_gym_environments import create_sentiment_crypto_env
from stable_baselines3 import PPO

# Create environment
env = create_sentiment_crypto_env(
    assets=["BTC", "ETH"],
    sentiment_sources=["twitter", "reddit", "news"]
)

# Train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode finished. Portfolio value: ${info['portfolio_value']:,.2f}")
        obs, _ = env.reset()
```

## Configuration Options

### Environment Configuration

- **Assets**: List of crypto assets to trade
- **Data Source**: Historical, synthetic, or live data
- **Market Features**: Order books, sentiment, technical indicators
- **Risk Management**: Position limits, stop losses, drawdown limits

### Observation Configuration

- **History Length**: Number of historical periods
- **Normalization**: Data normalization methods
- **Feature Selection**: Which features to include
- **Update Frequency**: How often to update features

### Action Configuration

- **Action Mode**: Discrete, continuous, or portfolio
- **Position Sizing**: Fixed, Kelly, volatility-based
- **Order Types**: Market, limit, stop orders
- **Risk Constraints**: Maximum position sizes, leverage

## Advanced Usage

### Custom Reward Functions

```python
from ml_gym_environments import BaseTradingEnvironment

class CustomReward:
    def calculate_reward(self, portfolio_value, previous_value, trade_info):
        # Custom reward logic
        profit = portfolio_value - previous_value
        volume_penalty = trade_info.get("total_fees", 0) * 0.5
        return profit - volume_penalty

# Use with environment
reward_fn = CustomReward()
# Integrate in training loop...
```

### Multi-Environment Training

```python
from ml_gym_environments import create_crypto_env
from gymnasium.vector import AsyncVectorEnv

# Create multiple environments
def make_env():
    return create_crypto_env(CryptoTradingConfig(assets=["BTC", "ETH"]))

envs = AsyncVectorEnv([make_env for _ in range(4)])

# Train with vectorized environments
# ... training code ...
```

## Documentation

- **[API Reference](docs/api_reference.md)**: Complete API documentation
- **[Examples](examples/)**: Usage examples and tutorials
- **[Configuration Guide](docs/configuration.md)**: Detailed configuration options
- **[Architecture](docs/architecture.md)**: System architecture overview

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `pytest tests/`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Create Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version

Current version: **1.0.0**

## Related Packages

- **[ml-ppo](../ml-ppo/)**: PPO implementation for crypto trading
- **[trading-engine](../trading-engine/)**: Core trading engine
- **[risk-manager](../risk-manager/)**: Risk management system
- **[common](../common/)**: Shared utilities and patterns

## Support

For support and questions:

- Create an issue in the repository
- Check the [documentation](docs/)
- Review existing [examples](examples/)

---

**Built for Enterprise Crypto Trading Applications**

_Enterprise-grade trading environments powered by cloud-native patterns_
