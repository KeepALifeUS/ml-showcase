"""
Enterprise NLP Sentiment Analysis for Crypto Trading - enterprise integration

Comprehensive sentiment analysis system optimized for cryptocurrency text analysis,
featuring transformer models, multi-language support, and real-time inference.

Key Components:
- Transformer Models: BERT, FinBERT, RoBERTa, DistilBERT variants
- Crypto-specific Features: Ticker detection, hashtag sentiment, meme scoring
- Multi-language Support: Translation services and multilingual models
- Enterprise Patterns: Model versioning, A/B testing, monitoring
- Real-time Inference: Streaming predictions with caching
- Explainability: LIME, SHAP, attention visualization

Author: ML-Framework Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "ML-Framework Team"
__email__ = "team@ml-framework.dev"
__license__ = "MIT"

# Core imports
from .models import (
    BERTSentiment,
    FinBERTModel,
    RoBERTaSentiment,
    DistilBERTModel,
    CryptoBERT,
    EnsembleModel,
)

from .preprocessing import (
    TextCleaner,
    CryptoTokenizer,
    EmojiHandler,
    SlangNormalizer,
    EntityExtractor,
    AbbreviationExpander,
)

from .features import (
    TFIDFFeatures,
    WordEmbeddings,
    SentimentLexicons,
    SyntacticFeatures,
    ContextualEmbeddings,
    CryptoSpecificFeatures,
)

from .inference import (
    BatchPredictor,
    StreamingPredictor,
    ModelServer,
    OptimizationEngine,
    CachingLayer,
)

from .api import (
    SentimentAPI,
    GRPCServer,
    WebSocketAPI,
    BatchAPI,
)

from .utils import (
    Config,
    Logger,
    ModelRegistry,
    DataValidator,
    Profiler,
)

# Enterprise enterprise patterns
from .explainability import (
    AttentionVisualizer,
    LIMEExplainer,
    SHAPExplainer,
    IntegratedGradients,
)

from .evaluation import (
    SentimentMetrics,
    CrossValidator,
    Benchmark,
    ErrorAnalysis,
    ConfusionMatrix,
)

# Export main classes for easy access
__all__ = [
    # Core Models
    "BERTSentiment",
    "FinBERTModel", 
    "RoBERTaSentiment",
    "DistilBERTModel",
    "CryptoBERT",
    "EnsembleModel",
    
    # Preprocessing
    "TextCleaner",
    "CryptoTokenizer",
    "EmojiHandler",
    "SlangNormalizer",
    "EntityExtractor",
    "AbbreviationExpander",
    
    # Features
    "TFIDFFeatures",
    "WordEmbeddings",
    "SentimentLexicons",
    "SyntacticFeatures",
    "ContextualEmbeddings",
    "CryptoSpecificFeatures",
    
    # Inference
    "BatchPredictor",
    "StreamingPredictor",
    "ModelServer",
    "OptimizationEngine",
    "CachingLayer",
    
    # API
    "SentimentAPI",
    "GRPCServer",
    "WebSocketAPI",
    "BatchAPI",
    
    # Utils
    "Config",
    "Logger",
    "ModelRegistry",
    "DataValidator",
    "Profiler",
    
    # Explainability
    "AttentionVisualizer",
    "LIMEExplainer",
    "SHAPExplainer",
    "IntegratedGradients",
    
    # Evaluation
    "SentimentMetrics",
    "CrossValidator",
    "Benchmark",
    "ErrorAnalysis",
    "ConfusionMatrix",
]

# Version info
version_info = tuple(map(int, __version__.split('.')))

# enterprise Metadata
ENTERPRISE_METADATA = {
    "component_type": "ml_nlp_sentiment",
    "architecture_pattern": "microservice_ml_pipeline",
    "scalability_tier": "enterprise",
    "deployment_ready": True,
    "monitoring_enabled": True,
    "observability_level": "full",
    "security_compliance": "fintech_grade",
    "performance_tier": "high_throughput",
}

# Package health check
def health_check() -> dict:
    """Validation state package and dependencies"""
    import sys
    import torch
    
    return {
        "status": "healthy",
        "version": __version__,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "_integration": "enabled",
        "enterprise_features": "active",
    }

# Initialize logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())