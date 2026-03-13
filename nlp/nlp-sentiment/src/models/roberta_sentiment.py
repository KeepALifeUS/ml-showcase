"""
RoBERTa Sentiment Analysis Model for Crypto Text

RoBERTa (Robustly Optimized BERT Pretraining Approach) implementation
optimized for cryptocurrency sentiment analysis with enhanced robustness.

Author: ML-Framework Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
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
import re

from .bert_sentiment import BERTSentiment, SentimentOutput
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import Profiler

logger = Logger(__name__).get_logger()


class RoBERTaSentiment(BERTSentiment):
    """
    RoBERTa-based Sentiment Analysis Model for Crypto Text
    
    Enhanced version of BERT with:
    - No Next Sentence Prediction (NSP) task
    - Dynamic masking during training
    - Larger training corpus
    - Improved tokenization with byte-pair encoding
    - Better handling of social media text
    - Enhanced robustness to adversarial examples
    """
    
    ROBERTA_MODEL_NAMES = {
        "base": "roberta-base",
        "large": "roberta-large", 
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "emotion": "cardiffnlp/twitter-roberta-base-emotion-latest",
        "offensive": "cardiffnlp/twitter-roberta-base-offensive-latest",
        "irony": "cardiffnlp/twitter-roberta-base-irony-latest",
        "financial": "ProsusAI/finbert-roberta",
        "crypto": "ElKulako/cryptobert",
        "social": "martin-ha/toxic-comment-model"
    }
    
    def __init__(
        self,
        model_variant: str = "sentiment",  # base, large, sentiment, emotion, etc.
        num_labels: int = 3,
        social_media_optimized: bool = True,
        adversarial_robustness: bool = True,
        emoji_aware: bool = True,
        sarcasm_detection: bool = True,
        **kwargs
    ):
        
        # Get appropriate RoBERTa model name
        model_name = self.ROBERTA_MODEL_NAMES.get(model_variant, "roberta-base")
        
        # Initialize parent class (will use RoBERTa through AutoModel)
        super().__init__(model_name=model_name, num_labels=num_labels, **kwargs)
        
        self.model_variant = model_variant
        self.social_media_optimized = social_media_optimized
        self.adversarial_robustness = adversarial_robustness
        self.emoji_aware = emoji_aware
        self.sarcasm_detection = sarcasm_detection
        
        # RoBERTa-specific components
        self._setup_roberta_components()
        
        # Update metadata
        self.model_metadata.update({
            "model_type": "roberta_sentiment",
            "variant": model_variant,
            "social_media_optimized": social_media_optimized,
            "adversarial_robust": adversarial_robustness,
            "emoji_aware": emoji_aware,
            "sarcasm_detection": sarcasm_detection,
        })
        
        logger.info(f"Initialized RoBERTa model: {model_variant} variant")
    
    def _setup_roberta_components(self):
        """Setup RoBERTa-specific neural components"""
        
        hidden_size = self.bert.config.hidden_size
        
        # Social media text components
        if self.social_media_optimized:
            self.hashtag_embedding = nn.Embedding(10000, 64)  # Hashtag embeddings
            self.mention_embedding = nn.Embedding(5000, 32)   # @mention embeddings
            self.url_embedding = nn.Embedding(100, 16)        # URL type embeddings
        
        # Emoji processing
        if self.emoji_aware:
            self.emoji_embedding = nn.Embedding(2000, 128)    # Emoji embeddings
            self.emoji_sentiment_classifier = nn.Linear(128, 3)  # emoji sentiment
        
        # Sarcasm detection
        if self.sarcasm_detection:
            self.sarcasm_detector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 2)  # sarcastic vs non-sarcastic
            )
            
            # Sarcasm-aware attention
            self.sarcasm_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Adversarial robustness components
        if self.adversarial_robustness:
            self.noise_layer = nn.Dropout2d(0.1)  # Adversarial noise
            self.gradient_reversal = GradientReversalLayer()
            self.domain_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 5)  # Different text domains
            )
        
        # Enhanced attention mechanisms
        self.multi_scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=heads,
                dropout=0.1,
                batch_first=True
            ) for heads in [4, 8, 12]  # Different attention scales
        ])
        
        # Contextual understanding
        self.context_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Advanced pooling strategies
        self.adaptive_pooling = nn.AdaptiveAvgPool1d(1)
        self.max_pooling = nn.AdaptiveMaxPool1d(1)
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            batch_first=True
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        emoji_ids: Optional[torch.Tensor] = None,
        hashtag_ids: Optional[torch.Tensor] = None,
        mention_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sarcasm_labels: Optional[torch.Tensor] = None,
        domain_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        Enhanced RoBERTa forward pass with social media features
        """
        
        with self._performance_context("roberta_forward"):
            # Get RoBERTa outputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            sequence_output = outputs.last_hidden_state
            pooled_output = self._get_pooled_output(sequence_output, attention_mask)
            
            # Apply multi-scale attention
            multi_scale_outputs = []
            for attention_layer in self.multi_scale_attention:
                attended_output, _ = attention_layer(
                    sequence_output, 
                    sequence_output, 
                    sequence_output,
                    key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
                )
                multi_scale_outputs.append(attended_output.mean(dim=1))
            
            # Combine multi-scale features
            multi_scale_pooled = torch.stack(multi_scale_outputs, dim=1).mean(dim=1)
            enhanced_pooled = pooled_output + multi_scale_pooled
            
            # Add social media features if provided
            if self.social_media_optimized:
                if hashtag_ids is not None:
                    hashtag_emb = self.hashtag_embedding(hashtag_ids).mean(dim=1)
                    enhanced_pooled = enhanced_pooled + hashtag_emb
                
                if mention_ids is not None:
                    mention_emb = self.mention_embedding(mention_ids).mean(dim=1)
                    enhanced_pooled = enhanced_pooled + mention_emb
            
            # Add emoji features
            emoji_sentiment_logits = None
            if self.emoji_aware and emoji_ids is not None:
                emoji_emb = self.emoji_embedding(emoji_ids).mean(dim=1)
                enhanced_pooled = enhanced_pooled + emoji_emb
                emoji_sentiment_logits = self.emoji_sentiment_classifier(emoji_emb)
            
            # Sarcasm detection
            sarcasm_logits = None
            if self.sarcasm_detection:
                # Apply sarcasm-aware attention
                sarcasm_attended, _ = self.sarcasm_attention(
                    sequence_output,
                    sequence_output,
                    sequence_output,
                    key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
                )
                sarcasm_pooled = sarcasm_attended.mean(dim=1)
                sarcasm_logits = self.sarcasm_detector(sarcasm_pooled)
                
                # Adjust sentiment based on sarcasm
                sarcasm_prob = F.softmax(sarcasm_logits, dim=-1)[:, 1:2]  # Probability of sarcasm
                enhanced_pooled = enhanced_pooled * (1 - sarcasm_prob) + sarcasm_pooled * sarcasm_prob
            
            # Adversarial robustness
            domain_logits = None
            if self.adversarial_robustness:
                # Apply gradient reversal for domain adaptation
                domain_features = self.gradient_reversal(enhanced_pooled)
                domain_logits = self.domain_classifier(domain_features)
            
            # Apply normalization and dropout
            enhanced_pooled = self.layer_norm(enhanced_pooled)
            enhanced_pooled = self.dropout(enhanced_pooled)
            
            # Main sentiment classification
            sentiment_logits = self.classifier(enhanced_pooled)
            
            # Calculate losses
            total_loss = None
            sentiment_loss = None
            sarcasm_loss = None
            domain_loss = None
            emoji_loss = None
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                sentiment_loss = loss_fct(sentiment_logits, labels)
                total_loss = sentiment_loss
            
            if self.sarcasm_detection and sarcasm_labels is not None and sarcasm_logits is not None:
                sarcasm_loss_fct = nn.CrossEntropyLoss()
                sarcasm_loss = sarcasm_loss_fct(sarcasm_logits, sarcasm_labels)
                total_loss = (total_loss + 0.3 * sarcasm_loss) if total_loss else sarcasm_loss
            
            if self.adversarial_robustness and domain_labels is not None and domain_logits is not None:
                domain_loss_fct = nn.CrossEntropyLoss()
                domain_loss = domain_loss_fct(domain_logits, domain_labels)
                total_loss = (total_loss + 0.2 * domain_loss) if total_loss else domain_loss
            
            if not return_dict:
                outputs_tuple = (sentiment_logits,)
                if sarcasm_logits is not None:
                    outputs_tuple += (sarcasm_logits,)
                if domain_logits is not None:
                    outputs_tuple += (domain_logits,)
                return ((total_loss,) + outputs_tuple) if total_loss else outputs_tuple
            
            return {
                "loss": total_loss,
                "sentiment_logits": sentiment_logits,
                "sarcasm_logits": sarcasm_logits,
                "domain_logits": domain_logits,
                "emoji_sentiment_logits": emoji_sentiment_logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
                "sentiment_loss": sentiment_loss,
                "sarcasm_loss": sarcasm_loss,
                "domain_loss": domain_loss,
            }
    
    def _get_pooled_output(
        self, 
        sequence_output: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Advanced pooling strategy combining multiple pooling methods
        """
        
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        # 1. CLS token (first token)
        cls_output = sequence_output[:, 0, :]
        
        # 2. Mean pooling with attention mask
        if attention_mask is not None:
            masked_output = sequence_output * attention_mask.unsqueeze(-1).float()
            mean_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).float()
        else:
            mean_output = sequence_output.mean(dim=1)
        
        # 3. Max pooling
        max_output = sequence_output.max(dim=1)[0]
        
        # 4. Attention pooling
        query = cls_output.unsqueeze(1)
        attention_output, _ = self.attention_pooling(query, sequence_output, sequence_output)
        attention_output = attention_output.squeeze(1)
        
        # Combine all pooling strategies
        combined_output = (cls_output + mean_output + max_output + attention_output) / 4
        
        return combined_output
    
    def predict_enhanced(
        self,
        texts: Union[str, List[str]],
        include_sarcasm: bool = True,
        include_emoji_sentiment: bool = True,
        include_social_features: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Enhanced prediction with RoBERTa-specific features
        
        Args:
            texts: Input text(s)
            include_sarcasm: Include sarcasm detection
            include_emoji_sentiment: Include emoji sentiment
            include_social_features: Extract social media features
            
        Returns:
            Enhanced prediction results
        """
        
        self.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        results = []
        
        with torch.no_grad():
            # Extract social media features
            processed_texts = []
            hashtag_features = []
            mention_features = []
            emoji_features = []
            
            for text in texts:
                processed_text, features = self._extract_social_features(text)
                processed_texts.append(processed_text)
                
                if include_social_features:
                    hashtag_features.append(features.get("hashtags", []))
                    mention_features.append(features.get("mentions", []))
                    emoji_features.append(features.get("emojis", []))
            
            # Tokenize
            encoded = self.tokenizer(
                processed_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Prepare social media features as tensors (simplified)
            batch_size = input_ids.shape[0]
            
            # Forward pass
            start_time = time.time()
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            processing_time = time.time() - start_time
            
            # Process outputs
            sentiment_probs = F.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            
            sarcasm_probs = None
            sarcasm_preds = None
            if include_sarcasm and outputs["sarcasm_logits"] is not None:
                sarcasm_probs = F.softmax(outputs["sarcasm_logits"], dim=-1)
                sarcasm_preds = torch.argmax(sarcasm_probs, dim=-1)
            
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
                
                if include_sarcasm and sarcasm_probs is not None:
                    result["sarcasm"] = {
                        "is_sarcastic": bool(sarcasm_preds[i].item()),
                        "probability": sarcasm_probs[i][1].item(),
                        "confidence": torch.max(sarcasm_probs[i]).item(),
                    }
                
                if include_social_features:
                    result["social_features"] = {
                        "hashtags": hashtag_features[i],
                        "mentions": mention_features[i],
                        "emojis": emoji_features[i],
                    }
                
                if include_emoji_sentiment and emoji_features[i]:
                    result["emoji_sentiment"] = self._analyze_emoji_sentiment(emoji_features[i])
                
                results.append(result)
        
        return results[0] if is_single else results
    
    def _extract_social_features(self, text: str) -> Tuple[str, Dict[str, List[str]]]:
        """Extract social media features from text"""
        
        import re
        
        features = {
            "hashtags": [],
            "mentions": [],
            "emojis": [],
            "urls": [],
        }
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', text)
        features["hashtags"] = [h.lower() for h in hashtags]
        
        # Extract mentions
        mentions = re.findall(r'@\w+', text)
        features["mentions"] = [m.lower() for m in mentions]
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        features["urls"] = urls
        
        # Extract emojis (simplified - would need emoji library for full implementation)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+"
        )
        emojis = emoji_pattern.findall(text)
        features["emojis"] = emojis
        
        # Clean text (replace features with tokens)
        processed_text = text
        for hashtag in features["hashtags"]:
            processed_text = processed_text.replace(hashtag, "[HASHTAG]")
        for mention in features["mentions"]:
            processed_text = processed_text.replace(mention, "[MENTION]")
        for url in features["urls"]:
            processed_text = processed_text.replace(url, "[URL]")
        for emoji in features["emojis"]:
            processed_text = processed_text.replace(emoji, "[EMOJI]")
        
        return processed_text, features
    
    def _analyze_emoji_sentiment(self, emojis: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of emojis"""
        
        # Simplified emoji sentiment mapping
        positive_emojis = ["😀", "😃", "😄", "😁", "😆", "😍", "🥰", "😘", "🤗", "😊", "☺️", "😌", "👍", "💯", "🎉", "🚀", "💰", "💎"]
        negative_emojis = ["😠", "😡", "🤬", "😤", "😖", "😞", "😟", "😢", "😭", "😰", "😨", "👎", "💔", "📉", "🔻"]
        neutral_emojis = ["😐", "😑", "🤔", "🙄", "😶", "😏"]
        
        sentiment_scores = {"positive": 0, "negative": 0, "neutral": 0}
        
        for emoji in emojis:
            if emoji in positive_emojis:
                sentiment_scores["positive"] += 1
            elif emoji in negative_emojis:
                sentiment_scores["negative"] += 1
            else:
                sentiment_scores["neutral"] += 1
        
        total = sum(sentiment_scores.values())
        if total == 0:
            return {"sentiment": "neutral", "confidence": 0.0, "distribution": sentiment_scores}
        
        # Normalize scores
        distribution = {k: v/total for k, v in sentiment_scores.items()}
        dominant_sentiment = max(distribution, key=distribution.get)
        confidence = distribution[dominant_sentiment]
        
        return {
            "sentiment": dominant_sentiment,
            "confidence": confidence,
            "distribution": distribution,
            "emoji_count": total,
        }
    
    def preprocess_social_media_text(self, text: str) -> str:
        """
        Enhanced preprocessing for social media text
        """
        
        # Start with financial preprocessing
        text = self.preprocess_crypto_text(text)
        
        # Social media specific preprocessing
        
        # Normalize repeated characters (sooooo good -> so good)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Normalize repeated punctuation (!!!! -> !!)
        text = re.sub(r'([!?.])\1{2,}', r'\1\1', text)
        
        # Handle common social media abbreviations
        social_abbreviations = {
            'lol': 'laugh out loud',
            'lmao': 'laugh my ass off',
            'omg': 'oh my god',
            'wtf': 'what the fuck',
            'tbh': 'to be honest',
            'imo': 'in my opinion',
            'imho': 'in my humble opinion',
            'fyi': 'for your information',
            'btw': 'by the way',
            'afaik': 'as far as i know',
            'idk': 'i dont know',
            'nvm': 'never mind',
            'smh': 'shaking my head',
        }
        
        for abbrev, expansion in social_abbreviations.items():
            text = re.sub(rf'\b{abbrev}\b', expansion, text, flags=re.IGNORECASE)
        
        # Normalize contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am",
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text, flags=re.IGNORECASE)
        
        return text


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer for adversarial training"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class GradientReversalFunction(torch.autograd.Function):
    """Function for gradient reversal"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


# Factory function for creation RoBERTa model
def create_roberta_sentiment(
    variant: str = "sentiment",
    social_media_optimized: bool = True,
    adversarial_robust: bool = True,
    **kwargs
) -> RoBERTaSentiment:
    """
    Factory function for creation RoBERTa sentiment model
    
    Args:
        variant: Model variant (base, large, sentiment, emotion, etc.)
        social_media_optimized: Enable social media optimizations
        adversarial_robust: Enable adversarial robustness
        **kwargs: Additional parameters
        
    Returns:
        Configured RoBERTaSentiment model
    """
    
    model = RoBERTaSentiment(
        model_variant=variant,
        social_media_optimized=social_media_optimized,
        adversarial_robustness=adversarial_robust,
        emoji_aware=social_media_optimized,
        sarcasm_detection=social_media_optimized,
        **kwargs
    )
    
    return model