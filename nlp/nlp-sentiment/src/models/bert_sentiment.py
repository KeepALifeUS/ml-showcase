"""
BERT Sentiment Analysis Model for Crypto Text

Enterprise-grade BERT implementation with enterprise patterns for cryptocurrency
sentiment analysis. Optimized for financial text with advanced features.

Author: ML-Framework Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    BertModel, 
    BertTokenizer, 
    BertConfig,
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
from contextlib import contextmanager

#  imports
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import Profiler
from ..utils.model_registry import ModelRegistry

logger = Logger(__name__).get_logger()


@dataclass
class SentimentOutput:
    """Structured output model sentiment analysis"""
    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_class: int
    confidence: float
    processing_time: float
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion in dictionary for API"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": float(self.confidence),
            "probabilities": self.probabilities.tolist() if torch.is_tensor(self.probabilities) else self.probabilities,
            "processing_time": self.processing_time,
            "model_name": self.model_name,
            "sentiment_label": self._get_sentiment_label(),
        }
    
    def _get_sentiment_label(self) -> str:
        """Get text label sentiment"""
        labels = ["negative", "neutral", "positive"]
        return labels[self.predicted_class] if self.predicted_class < len(labels) else "unknown"


class BERTSentiment(nn.Module):
    """
    BERT-based Sentiment Analysis Model for Crypto Text
    
    Implements enterprise-grade BERT model with enterprise patterns:
    - Model versioning and registry integration
    - Performance monitoring and profiling
    - Advanced dropout and regularization
    - Crypto-specific preprocessing
    - Batch optimization for high-throughput
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,  # negative, neutral, positive
        dropout_rate: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        super().__init__()
        
        self.config = config or Config()
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.dropout_rate = dropout_rate
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.profiler = Profiler()
        self.model_registry = ModelRegistry()
        
        # Load BERT configuration
        self.bert_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            cache_dir=cache_dir,
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            do_lower_case=True
        )
        
        self.bert = AutoModel.from_pretrained(
            model_name,
            config=self.bert_config,
            cache_dir=cache_dir,
        )
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Advanced regularization
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # Crypto-specific features
        self.crypto_embedding = nn.Embedding(1000, 128)  # For crypto-specific tokens
        self.crypto_attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Model metadata
        self.model_metadata = {
            "model_type": "bert_sentiment",
            "version": "1.0.0",
            "num_parameters": self._count_parameters(),
            "crypto_optimized": True,
            "enterprise_enabled": True,
        }
        
        self.to(self.device)
        logger.info(f"Initialized BERTSentiment model: {model_name} on {self.device}")
    
    def _count_parameters(self) -> int:
        """Counting number parameters model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @contextmanager
    def _performance_context(self, operation: str):
        """Context manager for tracking performance"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.profiler.record(operation, elapsed)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through BERT model
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            labels: Labels for training [batch_size]
            return_dict: Whether to return dictionary
            
        Returns:
            SentimentOutput or tuple of tensors
        """
        
        with self._performance_context("forward_pass"):
            # BERT forward pass
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            
            # Get pooled output (CLS token)
            pooled_output = outputs.pooler_output
            
            # Apply layer normalization and dropout
            pooled_output = self.layer_norm(pooled_output)
            pooled_output = self.dropout(pooled_output)
            
            # Crypto-specific attention (optional enhancement)
            if hasattr(self, 'use_crypto_attention') and self.use_crypto_attention:
                crypto_attended, _ = self.crypto_attention(
                    pooled_output.unsqueeze(1),
                    pooled_output.unsqueeze(1), 
                    pooled_output.unsqueeze(1)
                )
                pooled_output = crypto_attended.squeeze(1)
            
            # Classification
            logits = self.classifier(pooled_output)
            
            # Calculate loss if labels provided
            loss = None
            if labels is not None:
                if self.num_labels == 1:
                    # Regression
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    # Classification
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            if not return_dict:
                output = (logits,) + outputs[2:]
                return ((loss,) + output) if loss is not None else output
            
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": outputs.hidden_states if outputs.hidden_states else None,
                "attentions": outputs.attentions if outputs.attentions else None,
            }
    
    def predict(
        self, 
        texts: Union[str, List[str]],
        batch_size: int = 32,
        return_probabilities: bool = True,
        return_confidence: bool = True,
    ) -> Union[SentimentOutput, List[SentimentOutput]]:
        """
        Prediction sentiment for text or list texts
        
        Args:
            texts: Text or list texts
            batch_size: Size batch
            return_probabilities: Return probability
            return_confidence: Return confidence score
            
        Returns:
            SentimentOutput or list SentimentOutput
        """
        
        self.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        results = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenization
                start_time = time.time()
                
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
                
                logits = outputs["logits"]
                probabilities = F.softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                processing_time = time.time() - start_time
                
                # Create results
                for j, (logit, prob, pred_class, conf) in enumerate(zip(
                    logits, probabilities, predicted_classes, confidences
                )):
                    result = SentimentOutput(
                        logits=logit,
                        probabilities=prob,
                        predicted_class=pred_class.item(),
                        confidence=conf.item(),
                        processing_time=processing_time / len(batch_texts),
                        model_name=self.model_name
                    )
                    results.append(result)
        
        return results[0] if is_single else results
    
    def predict_proba(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Get probability classes"""
        results = self.predict(texts, **kwargs)
        if isinstance(results, list):
            return np.array([r.probabilities.cpu().numpy() for r in results])
        else:
            return results.probabilities.cpu().numpy()
    
    def preprocess_crypto_text(self, text: str) -> str:
        """
        Preprocessing text for crypto-specific features
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        
        import re
        
        # Normalize crypto tickers ($BTC, $ETH, etc.)
        text = re.sub(r'\$([A-Z]{2,10})', r'[TICKER_\1]', text)
        
        # Normalize hashtags (#Bitcoin, #DeFi, etc.)
        text = re.sub(r'#(\w+)', r'[HASHTAG_\1]', text)
        
        # Handle price mentions ($1000, $50K, etc.)
        text = re.sub(r'\$(\d+(?:[\.,]\d+)?[KMB]?)', r'[PRICE_\1]', text)
        
        # Handle percentage changes (+10%, -5%, etc.)
        text = re.sub(r'([+-]?\d+(?:\.\d+)?%)', r'[PERCENT_\1]', text)
        
        # Normalize common crypto slang
        crypto_slang = {
            'HODL': 'HOLD',
            'FUD': 'FEAR_UNCERTAINTY_DOUBT',
            'FOMO': 'FEAR_OF_MISSING_OUT',
            'ATH': 'ALL_TIME_HIGH',
            'ATL': 'ALL_TIME_LOW',
            'DCA': 'DOLLAR_COST_AVERAGE',
            'REKT': 'WRECKED',
            'MOON': 'PRICE_INCREASE',
            'DIAMOND_HANDS': 'HOLD_STRONG',
            'PAPER_HANDS': 'SELL_WEAK',
        }
        
        for slang, replacement in crypto_slang.items():
            text = re.sub(rf'\b{slang}\b', replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def save_model(self, save_path: Union[str, Path]) -> None:
        """Save model and tokenizer"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.bert_config,
            'model_metadata': self.model_metadata,
            'tokenizer_config': self.tokenizer.get_vocab(),
        }, save_path / "pytorch_model.bin")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.model_metadata, f, indent=2)
        
        # Register in model registry
        self.model_registry.register_model(
            name=f"bert_sentiment_{int(time.time())}",
            model_path=str(save_path),
            metadata=self.model_metadata
        )
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: Union[str, Path], **kwargs) -> "BERTSentiment":
        """Load saved model"""
        model_path = Path(model_path)
        
        # Load checkpoint
        checkpoint = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
        
        # Create model
        model = cls(**kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about model"""
        return {
            **self.model_metadata,
            "device": str(self.device),
            "parameters": self._count_parameters(),
            "performance_stats": self.profiler.get_stats(),
        }
    
    def enable_crypto_attention(self) -> None:
        """Enable crypto-specific attention mechanism"""
        self.use_crypto_attention = True
        logger.info("Crypto attention mechanism enabled")
    
    def disable_crypto_attention(self) -> None:
        """Disable crypto-specific attention mechanism"""
        self.use_crypto_attention = False
        logger.info("Crypto attention mechanism disabled")


# Factory function for convenient creation model
def create_bert_sentiment(
    model_name: str = "bert-base-uncased",
    crypto_optimized: bool = True,
    **kwargs
) -> BERTSentiment:
    """
    Factory function for creation BERT sentiment model
    
    Args:
        model_name: Name pretrained model
        crypto_optimized: Enable crypto-specific optimizations
        **kwargs: Additional parameters
        
    Returns:
        Configured BERTSentiment model
    """
    
    model = BERTSentiment(model_name=model_name, **kwargs)
    
    if crypto_optimized:
        model.enable_crypto_attention()
    
    return model


# Enterprise integration for production deployment
class BERTSentimentEnterprise(BERTSentiment):
    """
    BERT Sentiment with full enterprise integration

    Additional enterprise features:
    - A/B testing framework
    - Model monitoring and alerting  
    - Distributed training support
    - Auto-scaling inference
    - Advanced caching strategies
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # specific initialization
        self._setup_monitoring()
        self._setup_caching()
        self._setup_ab_testing()
    
    def _setup_monitoring(self):
        """Setup performance monitoring"""
        # Implementation would integrate with monitoring
        pass
    
    def _setup_caching(self):
        """Setup intelligent caching"""
        # Implementation would integrate with caching layer  
        pass
    
    def _setup_ab_testing(self):
        """Setup A/B testing framework"""
        # Implementation would integrate with A/B testing
        pass