"""
ML Graph Networks Package for Crypto Trading Bot v5.0
======================================================

Enterprise Graph Neural Networks implementation for cryptocurrency 
market analysis and trading with cloud-native patterns.

This package provides:
- GCN, GAT, GraphSAGE, MPNN implementations
- GNN ensemble system with uncertainty quantification  
- Graph construction algorithms for crypto markets
- Production-ready training and inference pipelines
- Real-time price prediction applications

Key Features:
- enterprise patterns for production deployment
- Scalable graph processing for large crypto datasets
- Advanced sampling strategies for memory efficiency
- Uncertainty quantification for risk management
- Real-time model performance monitoring
- Comprehensive testing and validation

Author: ML-Framework ML Team
Version: 1.0.0
License: MIT
"""

# Core models
from .models import (
    # GCN
    GraphConvolutionalNetwork,
    GCNConfig,
    CryptoGCNTrainer,
    create_crypto_gcn_model,
    
    # GAT
    GraphAttentionNetwork, 
    GATConfig,
    CryptoGATTrainer,
    create_crypto_gat_model,
    
    # GraphSAGE
    GraphSAGE,
    GraphSAGEConfig,
    CryptoGraphSAGETrainer,
    create_crypto_graphsage_model,
    
    # MPNN
    MessagePassingNeuralNetwork,
    MPNNConfig,
    CryptoMPNNTrainer,
    create_crypto_mpnn_model,
    
    # Ensemble
    GraphNeuralNetworkEnsemble,
    EnsembleConfig,
    CryptoGNNEnsembleTrainer,
    create_crypto_gnn_ensemble,
    
    # Utilities
    get_model_class,
    get_config_class,
    create_model
)

# Graph construction
from .graph_construction import (
    CorrelationGraphConfig,
    CorrelationCalculator,
    CorrelationGraphBuilder,
    create_correlation_graph
)

# Custom layers
from .layers.graph_layers import (
    TemporalGraphConv,
    CryptoAttentionLayer,
    DynamicEdgeWeightLearner
)

# Applications
from .applications.price_prediction import (
    PricePredictionConfig,
    CryptoPricePredictor,
    create_price_prediction_system
)

# Version info
__version__ = "1.0.0"
__author__ = "ML-Framework ML Team"
__email__ = "ml@ml-framework.dev"

# Package metadata
PACKAGE_INFO = {
    'name': 'ml-graph-networks',
    'version': __version__,
    'description': 'Enterprise Graph Neural Networks for Crypto Trading',
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'keywords': ['graph-neural-networks', 'cryptocurrency', 'trading', 'machine-learning'],
    'python_requires': '>=3.9',
    'dependencies': [
        'torch>=2.1.0',
        'torch-geometric>=2.4.0',
        'numpy>=1.24.0',
        'pandas>=2.1.0',
        'scikit-learn>=1.3.0',
        'networkx>=3.2.0'
    ]
}

# Model registry for dynamic creation
AVAILABLE_MODELS = {
    'gcn': 'Graph Convolutional Network',
    'gat': 'Graph Attention Network', 
    'graphsage': 'GraphSAGE',
    'mpnn': 'Message Passing Neural Network',
    'ensemble': 'GNN Ensemble System'
}

AVAILABLE_APPLICATIONS = {
    'price_prediction': 'Cryptocurrency Price Prediction',
    'portfolio_optimization': 'Portfolio Optimization (Future)',
    'anomaly_detection': 'Market Anomaly Detection (Future)',
    'risk_assessment': 'Risk Assessment (Future)'
}

def get_package_info() -> dict:
    """Get package information"""
    return PACKAGE_INFO.copy()

def list_available_models() -> dict:
    """List all available GNN models"""
    return AVAILABLE_MODELS.copy()

def list_available_applications() -> dict:
    """List all available applications"""
    return AVAILABLE_APPLICATIONS.copy()

# Export all public components
__all__ = [
    # Models
    'GraphConvolutionalNetwork',
    'GCNConfig',
    'CryptoGCNTrainer',
    'create_crypto_gcn_model',
    
    'GraphAttentionNetwork',
    'GATConfig', 
    'CryptoGATTrainer',
    'create_crypto_gat_model',
    
    'GraphSAGE',
    'GraphSAGEConfig',
    'CryptoGraphSAGETrainer',
    'create_crypto_graphsage_model',
    
    'MessagePassingNeuralNetwork',
    'MPNNConfig',
    'CryptoMPNNTrainer',
    'create_crypto_mpnn_model',
    
    'GraphNeuralNetworkEnsemble',
    'EnsembleConfig',
    'CryptoGNNEnsembleTrainer',
    'create_crypto_gnn_ensemble',
    
    # Graph construction
    'CorrelationGraphConfig',
    'CorrelationCalculator',
    'CorrelationGraphBuilder', 
    'create_correlation_graph',
    
    # Layers
    'TemporalGraphConv',
    'CryptoAttentionLayer',
    'DynamicEdgeWeightLearner',
    
    # Applications
    'PricePredictionConfig',
    'CryptoPricePredictor',
    'create_price_prediction_system',
    
    # Utilities
    'get_model_class',
    'get_config_class',
    'create_model',
    'get_package_info',
    'list_available_models',
    'list_available_applications'
]