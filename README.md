# ML Portfolio

Production-grade machine learning systems for cryptocurrency trading and research. 27 modules covering reinforcement learning, quantitative trading, NLP, and core ML infrastructure.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Reinforcement Learning

| Module | Description |
|--------|-------------|
| [`dqn`](reinforcement-learning/dqn/) | Rainbow DQN with prioritized replay, dueling networks, noisy nets |
| [`ppo`](reinforcement-learning/ppo/) | Proximal Policy Optimization for continuous trading actions |
| [`hierarchical-rl`](reinforcement-learning/hierarchical-rl/) | Options framework with temporal abstraction for multi-timeframe strategies |
| [`curiosity-exploration`](reinforcement-learning/curiosity-exploration/) | ICM-based curiosity-driven exploration for novel market regimes |
| [`meta-learning`](reinforcement-learning/meta-learning/) | MAML/Reptile for fast adaptation to new trading pairs |

## Quantitative Trading

| Module | Description |
|--------|-------------|
| [`order-flow-detection`](trading/order-flow-detection/) | Real-time order flow imbalance detection and whale tracking |
| [`harmonic-patterns`](trading/harmonic-patterns/) | Gartley, Butterfly, Bat, Crab pattern detection with ML validation |
| [`elliott-wave`](trading/elliott-wave/) | CNN-based Elliott Wave pattern recognition |
| [`volatility-forecasting`](trading/volatility-forecasting/) | GARCH family models + ML-based realized volatility prediction |
| [`fear-greed-index`](trading/fear-greed-index/) | Multi-signal Fear & Greed Index for market sentiment |
| [`automl-pipeline`](trading/automl-pipeline/) | Automated feature engineering, model selection, and hyperparameter tuning |

## NLP & Sentiment

| Module | Description |
|--------|-------------|
| [`sentiment-engine`](nlp/sentiment-engine/) | Multi-source sentiment aggregation (news, social, on-chain) |
| [`social-sentiment-analyzer`](nlp/social-sentiment-analyzer/) | Twitter/Reddit/Discord sentiment with influence weighting |
| [`nlp-sentiment`](nlp/nlp-sentiment/) | Transformer-based crypto-specific sentiment classification |

## Core ML

| Module | Description |
|--------|-------------|
| [`anomaly-detection`](core/anomaly-detection/) | Isolation Forest + Autoencoder ensemble for market anomalies |
| [`generative-models`](core/generative-models/) | GAN/VAE for synthetic trading data augmentation |
| [`graph-networks`](core/graph-networks/) | GNN for crypto correlation and dependency modeling |
| [`attention-mechanisms`](core/attention-mechanisms/) | Multi-head attention for temporal pattern recognition |
| [`bayesian-networks`](core/bayesian-networks/) | Uncertainty-aware predictions with calibrated confidence |
| [`continual-learning`](core/continual-learning/) | EWC/PackNet for non-stationary market adaptation |
| [`explainable-ai`](core/explainable-ai/) | SHAP/LIME integration for trade decision explanations |
| [`model-compression`](core/model-compression/) | Quantization, pruning, distillation for low-latency inference |

## Infrastructure

| Module | Description |
|--------|-------------|
| [`gym-environments`](infra/gym-environments/) | OpenAI Gym environments for crypto trading simulation |
| [`testing`](infra/testing/) | ML-specific testing: data drift, model regression, A/B validation |
| [`common`](infra/common/) | Shared utilities, data loaders, feature store connectors |
| [`xgboost`](infra/xgboost/) | Gradient boosting baselines and feature importance analysis |

## Other

| Module | Description |
|--------|-------------|
| [`prophet-forecasting`](prophet-forecasting/) | Facebook Prophet for price and volume time series forecasting |

---

## Quick Start

```bash
git clone https://github.com/KeepALifeUS/ml-showcase.git
cd ml-showcase

# Install a specific module
cd reinforcement-learning/dqn
pip install -r requirements.txt
python train.py
```

## Architecture

```
ml-showcase/
├── reinforcement-learning/   # RL agents for trading
├── trading/                  # Quantitative analysis modules
├── nlp/                      # Sentiment analysis pipeline
├── core/                     # Foundational ML components
├── infra/                    # Testing, utilities, environments
└── prophet-forecasting/      # Time series forecasting
```

Each module is self-contained with its own `README.md`, requirements, and tests.

## License

MIT
