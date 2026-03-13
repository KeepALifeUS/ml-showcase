"""
FinBERT Model for Financial Sentiment Analysis

Specialized BERT model trained on financial data for enhanced performance
on cryptocurrency and financial text sentiment analysis.

Author: ML-Framework Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BertTokenizer,
    BertModel,
)
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
import re

from .bert_sentiment import BERTSentiment, SentimentOutput
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import Profiler

logger = Logger(__name__).get_logger()


class FinBERTModel(BERTSentiment):
    """
    FinBERT Model for Financial Text Sentiment Analysis
    
    Extends BERTSentiment with financial domain-specific optimizations:
    - Pre-trained on financial text corpus
    - Financial terminology understanding
    - Market sentiment classification
    - Risk assessment capabilities
    - Regulatory text processing
    """
    
    FINBERT_MODEL_NAMES = {
        "base": "ProsusAI/finbert",
        "large": "nlpaueb/sec-bert-base", 
        "domain": "AdaptLLM/finance-chat",
        "sentiment": "ProsusAI/finbert-sentiment",
        "esg": "nlpaueb/legal-bert-base-uncased",
    }
    
    def __init__(
        self,
        model_variant: str = "sentiment",  # base, large, domain, sentiment, esg
        num_labels: int = 3,
        financial_vocab_size: int = 5000,
        sector_embedding_dim: int = 64,
        market_condition_awareness: bool = True,
        regulatory_compliance: bool = True,
        **kwargs
    ):
        
        # Get appropriate FinBERT model name
        model_name = self.FINBERT_MODEL_NAMES.get(model_variant, "ProsusAI/finbert")
        
        # Initialize parent class
        super().__init__(model_name=model_name, num_labels=num_labels, **kwargs)
        
        self.model_variant = model_variant
        self.financial_vocab_size = financial_vocab_size
        self.sector_embedding_dim = sector_embedding_dim
        self.market_condition_awareness = market_condition_awareness
        self.regulatory_compliance = regulatory_compliance
        
        # Financial domain-specific components
        self._setup_financial_components()
        
        # Update metadata
        self.model_metadata.update({
            "model_type": "finbert_sentiment",
            "variant": model_variant,
            "financial_optimized": True,
            "regulatory_compliant": regulatory_compliance,
            "market_aware": market_condition_awareness,
        })
        
        logger.info(f"Initialized FinBERT model: {model_variant} variant")
    
    def _setup_financial_components(self):
        """Setup financial domain-specific neural components"""
        
        hidden_size = self.bert.config.hidden_size
        
        # Financial terminology embeddings
        self.financial_embedding = nn.Embedding(
            self.financial_vocab_size, 
            hidden_size // 4
        )
        
        # Sector-specific embeddings
        self.sector_embedding = nn.Embedding(50, self.sector_embedding_dim)  # 50 sectors
        
        # Market condition embeddings
        if self.market_condition_awareness:
            self.market_condition_embedding = nn.Embedding(10, 32)  # bull, bear, sideways, etc.
        
        # Financial attention mechanism
        self.financial_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=12,
            dropout=0.1,
            batch_first=True
        )
        
        # Risk assessment head
        self.risk_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5)  # very_low, low, medium, high, very_high
        )
        
        # Volatility prediction head  
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)  # continuous volatility score
        )
        
        # Financial entity recognition
        self.entity_tagger = nn.Linear(hidden_size, 9)  # BIO tagging for financial entities
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        sector_ids: Optional[torch.Tensor] = None,
        market_condition_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        risk_labels: Optional[torch.Tensor] = None,
        volatility_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Enhanced forward pass with financial domain features
        """
        
        with self._performance_context("finbert_forward"):
            # Get BERT outputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            
            # Apply financial attention
            financial_attended, attention_weights = self.financial_attention(
                sequence_output, sequence_output, sequence_output,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            
            # Pool financial attention output
            financial_pooled = financial_attended.mean(dim=1)
            
            # Combine outputs
            combined_output = pooled_output + financial_pooled
            
            # Add sector information if provided
            if sector_ids is not None:
                sector_emb = self.sector_embedding(sector_ids)
                combined_output = torch.cat([combined_output, sector_emb], dim=-1)
                # Adjust classifier input size dynamically
                if not hasattr(self, '_classifier_adjusted'):
                    self.classifier = nn.Linear(
                        combined_output.shape[-1], 
                        self.num_labels
                    ).to(self.device)
                    self._classifier_adjusted = True
            
            # Add market condition awareness
            if self.market_condition_awareness and market_condition_ids is not None:
                market_emb = self.market_condition_embedding(market_condition_ids)
                combined_output = torch.cat([combined_output, market_emb], dim=-1)
            
            # Apply normalization and dropout
            combined_output = self.layer_norm(combined_output)
            combined_output = self.dropout(combined_output)
            
            # Main sentiment classification
            sentiment_logits = self.classifier(combined_output)
            
            # Risk assessment
            risk_logits = self.risk_classifier(pooled_output)
            
            # Volatility prediction
            volatility_score = self.volatility_predictor(pooled_output).squeeze(-1)
            
            # Entity tagging (token-level)
            entity_logits = self.entity_tagger(sequence_output)
            
            # Calculate losses
            total_loss = None
            sentiment_loss = None
            risk_loss = None
            volatility_loss = None
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                sentiment_loss = loss_fct(sentiment_logits, labels)
                total_loss = sentiment_loss
            
            if risk_labels is not None:
                risk_loss_fct = nn.CrossEntropyLoss()
                risk_loss = risk_loss_fct(risk_logits, risk_labels)
                total_loss = (total_loss + risk_loss) if total_loss else risk_loss
            
            if volatility_labels is not None:
                vol_loss_fct = nn.MSELoss()
                volatility_loss = vol_loss_fct(volatility_score, volatility_labels)
                total_loss = (total_loss + volatility_loss) if total_loss else volatility_loss
            
            if not return_dict:
                return (sentiment_logits, risk_logits, volatility_score, entity_logits)
            
            return {
                "loss": total_loss,
                "sentiment_logits": sentiment_logits,
                "risk_logits": risk_logits,
                "volatility_score": volatility_score,
                "entity_logits": entity_logits,
                "attention_weights": attention_weights,
                "hidden_states": outputs.hidden_states,
                "sentiment_loss": sentiment_loss,
                "risk_loss": risk_loss,
                "volatility_loss": volatility_loss,
            }
    
    def predict_financial(
        self,
        texts: Union[str, List[str]],
        include_risk: bool = True,
        include_volatility: bool = True,
        include_entities: bool = False,
        market_condition: Optional[str] = None,
        sector: Optional[str] = None,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Enhanced prediction with financial domain features
        
        Args:
            texts: Input text(s)
            include_risk: Include risk assessment
            include_volatility: Include volatility prediction
            include_entities: Include entity recognition
            market_condition: Current market condition (bull, bear, etc.)
            sector: Financial sector (tech, energy, etc.)
            
        Returns:
            Dictionary with financial predictions
        """
        
        self.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        results = []
        
        # Prepare market condition and sector IDs
        market_condition_id = self._get_market_condition_id(market_condition)
        sector_id = self._get_sector_id(sector)
        
        with torch.no_grad():
            # Tokenize
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Prepare additional inputs
            batch_size = input_ids.shape[0]
            
            sector_ids = None
            if sector_id is not None:
                sector_ids = torch.tensor([sector_id] * batch_size).to(self.device)
            
            market_condition_ids = None
            if market_condition_id is not None:
                market_condition_ids = torch.tensor([market_condition_id] * batch_size).to(self.device)
            
            # Forward pass
            start_time = time.time()
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sector_ids=sector_ids,
                market_condition_ids=market_condition_ids,
                return_dict=True
            )
            processing_time = time.time() - start_time
            
            # Process outputs
            sentiment_probs = F.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            
            risk_probs = F.softmax(outputs["risk_logits"], dim=-1) if include_risk else None
            risk_preds = torch.argmax(risk_probs, dim=-1) if include_risk else None
            
            volatility_scores = outputs["volatility_score"] if include_volatility else None
            
            # Create results
            for i in range(batch_size):
                result = {
                    "sentiment": {
                        "predicted_class": sentiment_preds[i].item(),
                        "probabilities": sentiment_probs[i].tolist(),
                        "confidence": torch.max(sentiment_probs[i]).item(),
                        "label": self._get_sentiment_label(sentiment_preds[i].item()),
                    },
                    "processing_time": processing_time / batch_size,
                    "model_name": self.model_name,
                    "model_variant": self.model_variant,
                }
                
                if include_risk and risk_probs is not None:
                    result["risk"] = {
                        "predicted_class": risk_preds[i].item(),
                        "probabilities": risk_probs[i].tolist(),
                        "confidence": torch.max(risk_probs[i]).item(),
                        "label": self._get_risk_label(risk_preds[i].item()),
                    }
                
                if include_volatility and volatility_scores is not None:
                    result["volatility"] = {
                        "score": volatility_scores[i].item(),
                        "level": self._get_volatility_level(volatility_scores[i].item()),
                    }
                
                if include_entities:
                    entities = self._extract_financial_entities(
                        texts[i], 
                        outputs["entity_logits"][i],
                        encoded["input_ids"][i]
                    )
                    result["entities"] = entities
                
                results.append(result)
        
        return results[0] if is_single else results
    
    def _get_market_condition_id(self, condition: Optional[str]) -> Optional[int]:
        """Convert market condition to ID"""
        if condition is None:
            return None
        
        conditions = {
            "bull": 0, "bullish": 0,
            "bear": 1, "bearish": 1, 
            "sideways": 2, "neutral": 2,
            "volatile": 3,
            "crash": 4,
            "recovery": 5,
            "bubble": 6,
            "correction": 7,
            "rally": 8,
            "consolidation": 9,
        }
        
        return conditions.get(condition.lower(), 2)  # default to neutral
    
    def _get_sector_id(self, sector: Optional[str]) -> Optional[int]:
        """Convert sector to ID"""
        if sector is None:
            return None
        
        sectors = {
            "technology": 0, "tech": 0,
            "financial": 1, "finance": 1, "fintech": 1,
            "healthcare": 2, "health": 2,
            "energy": 3,
            "consumer": 4,
            "industrial": 5,
            "materials": 6,
            "utilities": 7,
            "telecom": 8,
            "real_estate": 9, "realestate": 9,
            "crypto": 10, "cryptocurrency": 10, "blockchain": 10,
            "defi": 11, "decentralized_finance": 11,
            "nft": 12, "non_fungible_tokens": 12,
            "gaming": 13, "metaverse": 13,
            "ai": 14, "artificial_intelligence": 14,
        }
        
        return sectors.get(sector.lower(), 0)  # default to technology
    
    def _get_sentiment_label(self, class_id: int) -> str:
        """Get sentiment label from class ID"""
        labels = ["negative", "neutral", "positive"]
        return labels[class_id] if class_id < len(labels) else "unknown"
    
    def _get_risk_label(self, class_id: int) -> str:
        """Get risk label from class ID"""
        labels = ["very_low", "low", "medium", "high", "very_high"]
        return labels[class_id] if class_id < len(labels) else "unknown"
    
    def _get_volatility_level(self, score: float) -> str:
        """Convert volatility score to level"""
        if score < 0.2:
            return "very_low"
        elif score < 0.4:
            return "low"
        elif score < 0.6:
            return "medium"
        elif score < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _extract_financial_entities(
        self, 
        text: str, 
        entity_logits: torch.Tensor, 
        input_ids: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """Extract financial entities from text"""
        
        # Convert logits to predictions
        entity_preds = torch.argmax(entity_logits, dim=-1)
        
        # Convert input IDs back to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        entities = []
        entity_labels = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        
        current_entity = None
        current_tokens = []
        
        for token, pred in zip(tokens, entity_preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            label = entity_labels[pred.item()]
            
            if label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                        "label": current_entity,
                        "confidence": 0.8,  # placeholder
                    })
                
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith("I-") and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
            else:
                # End current entity
                if current_entity:
                    entities.append({
                        "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                        "label": current_entity,
                        "confidence": 0.8,  # placeholder
                    })
                    current_entity = None
                    current_tokens = []
        
        # Handle final entity
        if current_entity:
            entities.append({
                "text": self.tokenizer.convert_tokens_to_string(current_tokens),
                "label": current_entity,
                "confidence": 0.8,
            })
        
        return entities
    
    def preprocess_financial_text(self, text: str) -> str:
        """
        Enhanced preprocessing for financial text
        """
        
        # Start with crypto preprocessing
        text = self.preprocess_crypto_text(text)
        
        # Financial-specific preprocessing
        
        # Normalize financial amounts (1.5M, 2B, 500K, etc.)
        text = re.sub(r'\b(\d+(?:\.\d+)?)\s*([KMB])\b', r'[AMOUNT_\1\2]', text, flags=re.IGNORECASE)
        
        # Normalize financial ratios (P/E, D/E, etc.)
        text = re.sub(r'\b([A-Z]+/[A-Z]+)\b', r'[RATIO_\1]', text)
        
        # Normalize financial periods (Q1, Q2, FY2023, etc.)
        text = re.sub(r'\b(Q[1-4]|FY\d{4}|H[1-2])\b', r'[PERIOD_\1]', text)
        
        # Normalize currency codes (USD, EUR, GBP, etc.)
        text = re.sub(r'\b([A-Z]{3})\b(?=\s|$)', r'[CURRENCY_\1]', text)
        
        # Financial institutions
        bank_keywords = ['bank', 'credit', 'financial', 'investment', 'fund', 'capital']
        for keyword in bank_keywords:
            text = re.sub(rf'\b(\w+\s+{keyword}|\w+{keyword})\b', r'[FINANCIAL_INST_\1]', text, flags=re.IGNORECASE)
        
        # Stock exchanges
        exchanges = ['NYSE', 'NASDAQ', 'LSE', 'TSE', 'HKEX', 'BSE', 'NSE']
        for exchange in exchanges:
            text = re.sub(rf'\b{exchange}\b', f'[EXCHANGE_{exchange}]', text)
        
        # Financial instruments
        instruments = ['bond', 'stock', 'option', 'future', 'derivative', 'ETF', 'mutual fund', 'REIT']
        for instrument in instruments:
            text = re.sub(rf'\b{instrument}s?\b', f'[INSTRUMENT_{instrument.upper()}]', text, flags=re.IGNORECASE)
        
        # Economic indicators
        indicators = ['GDP', 'CPI', 'PPI', 'unemployment', 'inflation', 'interest rate', 'yield']
        for indicator in indicators:
            text = re.sub(rf'\b{indicator}\b', f'[INDICATOR_{indicator.upper().replace(" ", "_")}]', text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_market_sentiment(self, texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Analyze overall market sentiment from multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Aggregated market sentiment analysis
        """
        
        if not texts:
            return {"error": "No texts provided"}
        
        # Get predictions for all texts
        predictions = self.predict_financial(texts, **kwargs)
        
        # Aggregate sentiment
        sentiment_counts = {"negative": 0, "neutral": 0, "positive": 0}
        risk_counts = {"very_low": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}
        volatility_scores = []
        
        total_confidence = 0
        
        for pred in predictions:
            sentiment = pred["sentiment"]
            sentiment_counts[sentiment["label"]] += 1
            total_confidence += sentiment["confidence"]
            
            if "risk" in pred:
                risk_counts[pred["risk"]["label"]] += 1
            
            if "volatility" in pred:
                volatility_scores.append(pred["volatility"]["score"])
        
        # Calculate overall sentiment
        total_texts = len(texts)
        sentiment_distribution = {k: v/total_texts for k, v in sentiment_counts.items()}
        
        overall_sentiment = max(sentiment_distribution, key=sentiment_distribution.get)
        avg_confidence = total_confidence / total_texts
        
        # Calculate market outlook
        positive_ratio = sentiment_distribution["positive"]
        negative_ratio = sentiment_distribution["negative"]
        
        if positive_ratio > 0.6:
            market_outlook = "bullish"
        elif negative_ratio > 0.6:
            market_outlook = "bearish"
        elif positive_ratio > negative_ratio:
            market_outlook = "cautiously_optimistic"
        elif negative_ratio > positive_ratio:
            market_outlook = "cautiously_pessimistic"
        else:
            market_outlook = "neutral"
        
        result = {
            "overall_sentiment": overall_sentiment,
            "market_outlook": market_outlook,
            "sentiment_distribution": sentiment_distribution,
            "average_confidence": avg_confidence,
            "total_texts_analyzed": total_texts,
        }
        
        if risk_counts[list(risk_counts.keys())[0]] > 0:  # Check if risk data available
            risk_distribution = {k: v/total_texts for k, v in risk_counts.items()}
            result["risk_distribution"] = risk_distribution
            result["dominant_risk_level"] = max(risk_distribution, key=risk_distribution.get)
        
        if volatility_scores:
            avg_volatility = sum(volatility_scores) / len(volatility_scores)
            result["average_volatility"] = avg_volatility
            result["volatility_level"] = self._get_volatility_level(avg_volatility)
        
        return result


# Factory function for creation FinBERT model
def create_finbert_model(
    variant: str = "sentiment",
    financial_optimized: bool = True,
    **kwargs
) -> FinBERTModel:
    """
    Factory function for creation FinBERT model
    
    Args:
        variant: Model variant (base, large, domain, sentiment, esg)
        financial_optimized: Enable financial optimizations
        **kwargs: Additional parameters
        
    Returns:
        Configured FinBERTModel
    """
    
    model = FinBERTModel(
        model_variant=variant,
        market_condition_awareness=financial_optimized,
        regulatory_compliance=financial_optimized,
        **kwargs
    )
    
    return model