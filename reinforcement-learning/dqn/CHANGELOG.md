# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2024-02-01

### Added

#### DQN Algorithms
- **Core DQN** - Basic implementation with epsilon-greedy exploration
- **Double DQN** - Decoupled action selection and evaluation
- **Dueling DQN** - Separate value and advantage streams
- **Noisy Networks** - Parameter space exploration
- **Rainbow DQN** - Combined state-of-the-art improvements

#### Experience Replay
- Standard replay buffer with O(1) operations
- Prioritized Experience Replay (PER) with sum-tree
- Multi-step returns (N-step bootstrapping)
- Distributional DQN (C51)

#### Crypto Trading Integration
- Multi-asset portfolio management
- Advanced state representation (OHLCV, indicators, order book)
- Risk-adjusted rewards (Sharpe, Sortino, Calmar)
- Transaction cost modeling
- Risk management (stop-loss, take-profit, drawdown control)

#### Enterprise Infrastructure
- TensorBoard and Weights & Biases integration
- Distributed training support (multi-GPU)
- Model versioning and checkpointing
- Comprehensive performance analytics

### Performance
- Achieves state-of-the-art on CartPole-v1 in <100 episodes
- Competitive performance on LunarLander-v2
- Production-tested on cryptocurrency markets

## [Unreleased]

### Planned
- QR-DQN (Quantile Regression)
- IQN (Implicit Quantile Networks)
- Additional crypto exchange integrations
- Real-time inference optimization
