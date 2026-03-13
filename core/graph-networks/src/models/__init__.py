"""
Graph Neural Network Models for Crypto Trading Bot v5.0
========================================================

This module provides enterprise-grade implementations of various Graph Neural Network
architectures optimized for cryptocurrency market analysis and trading.

Available Models:
- GCN: Graph Convolutional Network for node classification and regression
- GAT: Graph Attention Network with multi-head attention mechanisms
- GraphSAGE: Scalable sampling-based GNN for large graphs
- MPNN: Message Passing Neural Network with customizable functions
- Ensemble: Multi-model ensemble system with uncertainty quantification

All models follow cloud-native patterns for production deployment.

Author: ML-Framework ML Team
Version: 1.0.0
"""

from .gcn import (
    GraphConvolutionalNetwork,
    GCNConfig,
    CryptoGCNTrainer,
    create_crypto_gcn_model
)

from .gat import (
    GraphAttentionNetwork,
    GATConfig,
    CryptoGATTrainer,
    MultiHeadAttention,
    create_crypto_gat_model
)

from .graphsage import (
    GraphSAGE,
    GraphSAGEConfig,
    CryptoGraphSAGETrainer,
    AdvancedSampler,
    CryptoAggregator,
    create_crypto_graphsage_model
)

from .mpnn import (
    MessagePassingNeuralNetwork,
    MPNNConfig,
    CryptoMPNNTrainer,
    MPNNLayer,
    MessageFunction,
    UpdateFunction,
    ReadoutFunction,
    create_crypto_mpnn_model
)

from .gnn_ensemble import (
    GraphNeuralNetworkEnsemble,
    EnsembleConfig,
    CryptoGNNEnsembleTrainer,
    ModelPerformanceTracker,
    UncertaintyQuantifier,
    create_crypto_gnn_ensemble
)

# Model registry for dynamic model creation
MODEL_REGISTRY = {
    'gcn': GraphConvolutionalNetwork,
    'gat': GraphAttentionNetwork,
    'graphsage': GraphSAGE,
    'mpnn': MessagePassingNeuralNetwork,
    'ensemble': GraphNeuralNetworkEnsemble
}

# Config registry
CONFIG_REGISTRY = {
    'gcn': GCNConfig,
    'gat': GATConfig,
    'graphsage': GraphSAGEConfig,
    'mpnn': MPNNConfig,
    'ensemble': EnsembleConfig
}

# Factory functions registry
FACTORY_REGISTRY = {
    'gcn': create_crypto_gcn_model,
    'gat': create_crypto_gat_model,
    'graphsage': create_crypto_graphsage_model,
    'mpnn': create_crypto_mpnn_model,
    'ensemble': create_crypto_gnn_ensemble
}

def get_model_class(model_type: str):
    """Get model class by name"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

def get_config_class(model_type: str):
    """Get config class by name"""
    if model_type not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config type: {model_type}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[model_type]

def create_model(model_type: str, **kwargs):
    """Factory function for creating models with their trainers"""
    if model_type not in FACTORY_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(FACTORY_REGISTRY.keys())}")
    return FACTORY_REGISTRY[model_type](**kwargs)

# Export all public components
__all__ = [
    # GCN
    'GraphConvolutionalNetwork',
    'GCNConfig', 
    'CryptoGCNTrainer',
    'create_crypto_gcn_model',
    
    # GAT
    'GraphAttentionNetwork',
    'GATConfig',
    'CryptoGATTrainer', 
    'MultiHeadAttention',
    'create_crypto_gat_model',
    
    # GraphSAGE
    'GraphSAGE',
    'GraphSAGEConfig',
    'CryptoGraphSAGETrainer',
    'AdvancedSampler',
    'CryptoAggregator',
    'create_crypto_graphsage_model',
    
    # MPNN
    'MessagePassingNeuralNetwork',
    'MPNNConfig',
    'CryptoMPNNTrainer',
    'MPNNLayer',
    'MessageFunction',
    'UpdateFunction', 
    'ReadoutFunction',
    'create_crypto_mpnn_model',
    
    # Ensemble
    'GraphNeuralNetworkEnsemble',
    'EnsembleConfig',
    'CryptoGNNEnsembleTrainer',
    'ModelPerformanceTracker',
    'UncertaintyQuantifier',
    'create_crypto_gnn_ensemble',
    
    # Utilities
    'MODEL_REGISTRY',
    'CONFIG_REGISTRY',
    'FACTORY_REGISTRY',
    'get_model_class',
    'get_config_class',
    'create_model'
]