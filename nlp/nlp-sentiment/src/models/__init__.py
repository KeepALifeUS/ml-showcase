"""
Transformer Models Module for Crypto Sentiment Analysis

Comprehensive collection of transformer-based sentiment analysis models
optimized for cryptocurrency text analysis with enterprise patterns.

Models:
- BERTSentiment: Standard BERT for general sentiment analysis
- FinBERTModel: FinBERT specialized for financial text
- RoBERTaSentiment: RoBERTa model for robust sentiment classification
- DistilBERTModel: Lightweight DistilBERT for fast inference
- CryptoBERT: Crypto-specific BERT fine-tuned on crypto data
- EnsembleModel: Ensemble of multiple models for better accuracy

Author: ML-Framework Team
"""

from .bert_sentiment import BERTSentiment
from .finbert_model import FinBERTModel
from .roberta_sentiment import RoBERTaSentiment
from .distilbert_model import DistilBERTModel
from .crypto_bert import CryptoBERT
from .ensemble_model import EnsembleModel

__all__ = [
    "BERTSentiment",
    "FinBERTModel",
    "RoBERTaSentiment", 
    "DistilBERTModel",
    "CryptoBERT",
    "EnsembleModel",
]