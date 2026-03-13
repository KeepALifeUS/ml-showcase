"""
CryptoBERT - Crypto-Specific BERT Model

BERT model fine-tuned specifically for cryptocurrency text analysis
with domain-specific vocabulary and understanding.

Author: ML-Framework Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import re
from collections import defaultdict

from .bert_sentiment import BERTSentiment, SentimentOutput
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import Profiler

logger = Logger(__name__).get_logger()


class CryptoBERT(BERTSentiment):
    """
    CryptoBERT - Cryptocurrency-Specific BERT Model
    
    Features:
    - Crypto-specific vocabulary expansion
    - Multi-asset sentiment analysis
    - Market condition awareness
    - Technical analysis integration
    - Social sentiment aggregation
    - Meme and slang detection
    - Influencer sentiment tracking
    - Market manipulation detection
    """
    
    CRYPTO_MODEL_VARIANTS = {
        "base": "bert-base-uncased",  # Base model for crypto fine-tuning
        "financial": "ProsusAI/finbert",
        "social": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "crypto": "ElKulako/cryptobert",  # Pre-trained on crypto data
        "defi": "kylelix7/nlp-crypto-analysis-bert",
    }
    
    # Comprehensive crypto vocabulary
    CRYPTO_VOCABULARY = {
        # Major cryptocurrencies
        "bitcoin": ["btc", "bitcoin", "₿", "sats", "satoshi", "satoshis"],
        "ethereum": ["eth", "ethereum", "ether", "gwei", "wei"],
        "binance_coin": ["bnb", "binance coin", "binance smart chain", "bsc"],
        "cardano": ["ada", "cardano"],
        "solana": ["sol", "solana"],
        "polkadot": ["dot", "polkadot"],
        "dogecoin": ["doge", "dogecoin", "doge coin", "such coin"],
        "shiba_inu": ["shib", "shiba", "shiba inu"],
        "avalanche": ["avax", "avalanche"],
        "polygon": ["matic", "polygon"],
        
        # DeFi terms
        "defi": ["defi", "decentralized finance", "yield farming", "liquidity mining"],
        "dex": ["dex", "decentralized exchange", "uniswap", "sushiswap", "pancakeswap"],
        "liquidity": ["liquidity", "liquidity pool", "lp token", "impermanent loss"],
        "staking": ["staking", "stake", "staked", "validator", "delegation"],
        "yield": ["yield", "apy", "apr", "yield farming", "farming"],
        
        # NFT terms
        "nft": ["nft", "non fungible token", "opensea", "nft collection"],
        "metaverse": ["metaverse", "virtual world", "vr", "ar"],
        
        # Trading terms
        "trading": ["trade", "trading", "long", "short", "leverage", "margin"],
        "hodl": ["hodl", "hold", "diamond hands", "paper hands"],
        "fud": ["fud", "fear uncertainty doubt", "fomo", "fear of missing out"],
        "pump": ["pump", "moon", "to the moon", "bullish", "bull run"],
        "dump": ["dump", "crash", "bearish", "bear market", "correction"],
        
        # Technical analysis
        "ta": ["resistance", "support", "breakout", "pattern", "chart", "candles"],
        "indicators": ["rsi", "macd", "bollinger bands", "moving average", "fibonacci"],
        
        # Market terms
        "market_cap": ["market cap", "mcap", "marketcap"],
        "volume": ["volume", "trading volume", "24h volume"],
        "volatility": ["volatile", "volatility", "vix"],
    }
    
    # Market sentiment keywords
    SENTIMENT_KEYWORDS = {
        "bullish": {
            "moon": 0.9, "bullish": 0.8, "pump": 0.7, "rally": 0.8,
            "breakout": 0.6, "uptrend": 0.7, "gains": 0.6, "profit": 0.6,
            "diamond hands": 0.8, "hodl": 0.5, "buy the dip": 0.6,
            "accumulate": 0.5, "long": 0.4, "calls": 0.5,
        },
        "bearish": {
            "crash": 0.9, "dump": 0.8, "bearish": 0.8, "correction": 0.6,
            "breakdown": 0.7, "downtrend": 0.7, "loss": 0.6, "rekt": 0.9,
            "paper hands": 0.7, "sell": 0.4, "short": 0.5, "puts": 0.5,
            "bubble": 0.6, "overvalued": 0.5, "panic": 0.8,
        },
        "neutral": {
            "consolidation": 0.0, "sideways": 0.0, "range": 0.0,
            "wait": 0.0, "watch": 0.0, "uncertain": 0.0,
        }
    }
    
    def __init__(
        self,
        model_variant: str = "crypto",
        num_labels: int = 3,
        crypto_vocab_size: int = 10000,
        asset_embedding_dim: int = 128,
        enable_multi_asset: bool = True,
        market_condition_aware: bool = True,
        technical_analysis_integration: bool = True,
        social_sentiment_aggregation: bool = True,
        meme_detection: bool = True,
        influencer_tracking: bool = True,
        manipulation_detection: bool = True,
        **kwargs
    ):
        
        # Get appropriate model name
        model_name = self.CRYPTO_MODEL_VARIANTS.get(model_variant, "bert-base-uncased")
        
        # Initialize parent class
        super().__init__(model_name=model_name, num_labels=num_labels, **kwargs)
        
        self.model_variant = model_variant
        self.crypto_vocab_size = crypto_vocab_size
        self.asset_embedding_dim = asset_embedding_dim
        self.enable_multi_asset = enable_multi_asset
        self.market_condition_aware = market_condition_aware
        self.technical_analysis_integration = technical_analysis_integration
        self.social_sentiment_aggregation = social_sentiment_aggregation
        self.meme_detection = meme_detection
        self.influencer_tracking = influencer_tracking
        self.manipulation_detection = manipulation_detection
        
        # Setup crypto-specific components
        self._setup_crypto_components()
        self._build_crypto_vocabulary()
        
        # Update metadata
        self.model_metadata.update({
            "model_type": "crypto_bert",
            "variant": model_variant,
            "crypto_optimized": True,
            "multi_asset_support": enable_multi_asset,
            "market_aware": market_condition_aware,
            "ta_integration": technical_analysis_integration,
            "meme_detection": meme_detection,
            "manipulation_detection": manipulation_detection,
        })
        
        logger.info(f"Initialized CryptoBERT model: {model_variant} variant")
        logger.info(f"Crypto vocabulary size: {len(self.crypto_vocab)}")
    
    def _setup_crypto_components(self):
        """Setup cryptocurrency-specific neural components"""
        
        hidden_size = self.bert.config.hidden_size
        
        # Crypto asset embeddings
        if self.enable_multi_asset:
            self.asset_embedding = nn.Embedding(200, self.asset_embedding_dim)  # 200 assets
            self.asset_attention = nn.MultiheadAttention(
                embed_dim=self.asset_embedding_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
        
        # Market condition embeddings
        if self.market_condition_aware:
            self.market_embedding = nn.Embedding(20, 64)  # Market conditions
            self.timeframe_embedding = nn.Embedding(10, 32)  # Different timeframes
        
        # Technical analysis integration
        if self.technical_analysis_integration:
            self.ta_encoder = nn.Sequential(
                nn.Linear(20, hidden_size // 4),  # 20 TA indicators
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, hidden_size // 8)
            )
        
        # Social sentiment components
        if self.social_sentiment_aggregation:
            self.social_aggregator = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=12,
                dropout=0.1,
                batch_first=True
            )
            self.influence_score_head = nn.Linear(hidden_size, 1)
        
        # Meme detection
        if self.meme_detection:
            self.meme_detector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 5)  # Different meme categories
            )
        
        # Market manipulation detection
        if self.manipulation_detection:
            self.manipulation_detector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 3)  # pump_and_dump, wash_trading, none
            )
        
        # Multi-head crypto attention
        self.crypto_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )
        
        # Price movement prediction head
        self.price_movement_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 5)  # strong_down, down, neutral, up, strong_up
        )
        
        # Volatility prediction
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)  # Volatility score 0-1
        )
    
    def _build_crypto_vocabulary(self):
        """Build comprehensive cryptocurrency vocabulary"""
        
        self.crypto_vocab = {}
        self.crypto_entities = defaultdict(list)
        
        # Add crypto vocabulary
        for category, terms in self.CRYPTO_VOCABULARY.items():
            for term in terms:
                self.crypto_vocab[term.lower()] = category
                self.crypto_entities[category].append(term.lower())
        
        # Add sentiment keywords
        for sentiment, keywords in self.SENTIMENT_KEYWORDS.items():
            for keyword, score in keywords.items():
                self.crypto_vocab[keyword.lower()] = {
                    "category": "sentiment",
                    "sentiment": sentiment,
                    "score": score
                }
        
        logger.info(f"Built crypto vocabulary with {len(self.crypto_vocab)} terms")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        asset_ids: Optional[torch.Tensor] = None,
        market_condition_ids: Optional[torch.Tensor] = None,
        ta_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        price_movement_labels: Optional[torch.Tensor] = None,
        meme_labels: Optional[torch.Tensor] = None,
        manipulation_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ):
        """
        CryptoBERT forward pass with crypto-specific features
        """
        
        with self._performance_context("crypto_bert_forward"):
            # Get BERT outputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            
            sequence_output = outputs.last_hidden_state
            pooled_output = outputs.pooler_output
            
            # Apply crypto-specific attention
            crypto_attended, attention_weights = self.crypto_attention(
                sequence_output, 
                sequence_output, 
                sequence_output,
                key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
            )
            
            # Pool crypto attention output
            crypto_pooled = crypto_attended.mean(dim=1)
            
            # Combine with standard pooled output
            enhanced_pooled = pooled_output + crypto_pooled
            
            # Add asset-specific information
            if self.enable_multi_asset and asset_ids is not None:
                asset_emb = self.asset_embedding(asset_ids)
                if asset_emb.dim() == 3:  # Multiple assets per sample
                    asset_emb = asset_emb.mean(dim=1)
                enhanced_pooled = torch.cat([enhanced_pooled, asset_emb], dim=-1)
                
                # Adjust classifier if needed
                if not hasattr(self, '_classifier_adjusted_crypto'):
                    self.classifier = nn.Linear(
                        enhanced_pooled.shape[-1],
                        self.num_labels
                    ).to(self.device)
                    self._classifier_adjusted_crypto = True
            
            # Add market condition information
            if self.market_condition_aware and market_condition_ids is not None:
                market_emb = self.market_embedding(market_condition_ids)
                enhanced_pooled = torch.cat([enhanced_pooled, market_emb], dim=-1)
            
            # Add technical analysis features
            if self.technical_analysis_integration and ta_features is not None:
                ta_encoded = self.ta_encoder(ta_features)
                enhanced_pooled = torch.cat([enhanced_pooled, ta_encoded], dim=-1)
            
            # Apply normalization and dropout
            enhanced_pooled = self.layer_norm(enhanced_pooled)
            enhanced_pooled = self.dropout(enhanced_pooled)
            
            # Main sentiment classification
            sentiment_logits = self.classifier(enhanced_pooled)
            
            # Price movement prediction
            price_movement_logits = self.price_movement_predictor(pooled_output)
            
            # Volatility prediction
            volatility_score = self.volatility_predictor(pooled_output).squeeze(-1)
            
            # Meme detection
            meme_logits = None
            if self.meme_detection:
                meme_logits = self.meme_detector(pooled_output)
            
            # Manipulation detection
            manipulation_logits = None
            if self.manipulation_detection:
                manipulation_logits = self.manipulation_detector(pooled_output)
            
            # Social influence score
            influence_score = None
            if self.social_sentiment_aggregation:
                influence_score = self.influence_score_head(pooled_output).squeeze(-1)
            
            # Calculate losses
            total_loss = None
            sentiment_loss = None
            price_loss = None
            meme_loss = None
            manipulation_loss = None
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                sentiment_loss = loss_fct(sentiment_logits, labels)
                total_loss = sentiment_loss
            
            if price_movement_labels is not None:
                price_loss_fct = nn.CrossEntropyLoss()
                price_loss = price_loss_fct(price_movement_logits, price_movement_labels)
                total_loss = (total_loss + 0.5 * price_loss) if total_loss else price_loss
            
            if self.meme_detection and meme_labels is not None and meme_logits is not None:
                meme_loss_fct = nn.CrossEntropyLoss()
                meme_loss = meme_loss_fct(meme_logits, meme_labels)
                total_loss = (total_loss + 0.3 * meme_loss) if total_loss else meme_loss
            
            if self.manipulation_detection and manipulation_labels is not None and manipulation_logits is not None:
                manip_loss_fct = nn.CrossEntropyLoss()
                manipulation_loss = manip_loss_fct(manipulation_logits, manipulation_labels)
                total_loss = (total_loss + 0.4 * manipulation_loss) if total_loss else manipulation_loss
            
            if not return_dict:
                outputs_tuple = (sentiment_logits, price_movement_logits, volatility_score)
                if meme_logits is not None:
                    outputs_tuple += (meme_logits,)
                if manipulation_logits is not None:
                    outputs_tuple += (manipulation_logits,)
                return ((total_loss,) + outputs_tuple) if total_loss else outputs_tuple
            
            return {
                "loss": total_loss,
                "sentiment_logits": sentiment_logits,
                "price_movement_logits": price_movement_logits,
                "volatility_score": volatility_score,
                "meme_logits": meme_logits,
                "manipulation_logits": manipulation_logits,
                "influence_score": influence_score,
                "attention_weights": attention_weights,
                "hidden_states": outputs.hidden_states,
                "sentiment_loss": sentiment_loss,
                "price_loss": price_loss,
                "meme_loss": meme_loss,
                "manipulation_loss": manipulation_loss,
            }
    
    def predict_crypto(
        self,
        texts: Union[str, List[str]],
        assets: Optional[Union[str, List[str]]] = None,
        market_condition: Optional[str] = None,
        ta_features: Optional[np.ndarray] = None,
        include_price_prediction: bool = True,
        include_meme_detection: bool = True,
        include_manipulation_detection: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Comprehensive crypto sentiment prediction
        
        Args:
            texts: Input text(s)
            assets: Asset symbol(s) (BTC, ETH, etc.)
            market_condition: Current market condition
            ta_features: Technical analysis features
            include_price_prediction: Include price movement prediction
            include_meme_detection: Include meme detection
            include_manipulation_detection: Include manipulation detection
            
        Returns:
            Comprehensive crypto sentiment analysis
        """
        
        self.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        results = []
        
        with torch.no_grad():
            # Preprocess texts for crypto-specific features
            processed_texts = []
            extracted_assets = []
            
            for i, text in enumerate(texts):
                processed_text = self.preprocess_crypto_text_advanced(text)
                processed_texts.append(processed_text)
                
                # Extract assets from text if not provided
                if assets is None:
                    text_assets = self._extract_crypto_assets(text)
                    extracted_assets.append(text_assets)
                else:
                    if isinstance(assets, str):
                        extracted_assets.append([assets])
                    else:
                        extracted_assets.append(assets if isinstance(assets, list) else [assets])
            
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
            
            # Prepare additional features
            batch_size = input_ids.shape[0]
            
            asset_ids = None
            if extracted_assets[0]:  # If any assets found
                asset_ids = self._convert_assets_to_ids(extracted_assets)
                asset_ids = torch.tensor(asset_ids).to(self.device)
            
            market_condition_id = None
            if market_condition:
                market_condition_id = self._get_market_condition_id(market_condition)
                market_condition_ids = torch.tensor([market_condition_id] * batch_size).to(self.device)
            else:
                market_condition_ids = None
            
            ta_features_tensor = None
            if ta_features is not None:
                ta_features_tensor = torch.tensor(ta_features).float().to(self.device)
                if ta_features_tensor.dim() == 1:
                    ta_features_tensor = ta_features_tensor.unsqueeze(0).repeat(batch_size, 1)
            
            # Forward pass
            start_time = time.time()
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                asset_ids=asset_ids,
                market_condition_ids=market_condition_ids,
                ta_features=ta_features_tensor,
                return_dict=True
            )
            processing_time = time.time() - start_time
            
            # Process outputs
            sentiment_probs = F.softmax(outputs["sentiment_logits"], dim=-1)
            sentiment_preds = torch.argmax(sentiment_probs, dim=-1)
            
            price_probs = F.softmax(outputs["price_movement_logits"], dim=-1)
            price_preds = torch.argmax(price_probs, dim=-1)
            
            volatility_scores = outputs["volatility_score"]
            
            meme_probs = None
            meme_preds = None
            if include_meme_detection and outputs["meme_logits"] is not None:
                meme_probs = F.softmax(outputs["meme_logits"], dim=-1)
                meme_preds = torch.argmax(meme_probs, dim=-1)
            
            manipulation_probs = None
            manipulation_preds = None
            if include_manipulation_detection and outputs["manipulation_logits"] is not None:
                manipulation_probs = F.softmax(outputs["manipulation_logits"], dim=-1)
                manipulation_preds = torch.argmax(manipulation_probs, dim=-1)
            
            # Create results
            for i in range(batch_size):
                result = {
                    "sentiment": {
                        "predicted_class": sentiment_preds[i].item(),
                        "probabilities": sentiment_probs[i].tolist(),
                        "confidence": torch.max(sentiment_probs[i]).item(),
                        "label": self._get_sentiment_label(sentiment_preds[i].item()),
                    },
                    "assets_detected": extracted_assets[i],
                    "processing_time": processing_time / batch_size,
                    "model_name": self.model_name,
                    "model_variant": self.model_variant,
                }
                
                if include_price_prediction:
                    result["price_movement"] = {
                        "predicted_class": price_preds[i].item(),
                        "probabilities": price_probs[i].tolist(),
                        "confidence": torch.max(price_probs[i]).item(),
                        "label": self._get_price_movement_label(price_preds[i].item()),
                    }
                    
                    result["volatility"] = {
                        "score": volatility_scores[i].item(),
                        "level": self._get_volatility_level(volatility_scores[i].item()),
                    }
                
                if include_meme_detection and meme_probs is not None:
                    result["meme_analysis"] = {
                        "predicted_class": meme_preds[i].item(),
                        "probabilities": meme_probs[i].tolist(),
                        "confidence": torch.max(meme_probs[i]).item(),
                        "category": self._get_meme_category(meme_preds[i].item()),
                    }
                
                if include_manipulation_detection and manipulation_probs is not None:
                    result["manipulation_risk"] = {
                        "predicted_class": manipulation_preds[i].item(),
                        "probabilities": manipulation_probs[i].tolist(),
                        "confidence": torch.max(manipulation_probs[i]).item(),
                        "type": self._get_manipulation_type(manipulation_preds[i].item()),
                    }
                
                # Add social influence score if available
                if outputs["influence_score"] is not None:
                    result["influence_score"] = outputs["influence_score"][i].item()
                
                results.append(result)
        
        return results[0] if is_single else results
    
    def _extract_crypto_assets(self, text: str) -> List[str]:
        """Extract cryptocurrency assets from text"""
        
        import re
        
        assets = set()
        text_lower = text.lower()
        
        # Look for ticker symbols ($BTC, $ETH, etc.)
        ticker_matches = re.findall(r'\$([A-Z]{2,10})', text)
        assets.update([ticker.lower() for ticker in ticker_matches])
        
        # Look for known crypto terms
        for term, category in self.crypto_vocab.items():
            if isinstance(category, str) and term in text_lower:
                # Map terms to standard asset symbols
                asset_mapping = {
                    "bitcoin": "btc", "ethereum": "eth", "dogecoin": "doge",
                    "cardano": "ada", "solana": "sol", "polkadot": "dot",
                    "binance coin": "bnb", "shiba inu": "shib",
                    "avalanche": "avax", "polygon": "matic"
                }
                
                if category in asset_mapping:
                    assets.add(asset_mapping[category])
                elif category in ["bitcoin", "ethereum", "dogecoin"]:  # Direct mappings
                    assets.add(category[:3] if len(category) > 3 else category)
        
        return list(assets)
    
    def _convert_assets_to_ids(self, asset_lists: List[List[str]]) -> List[List[int]]:
        """Convert asset symbols to IDs"""
        
        # Asset symbol to ID mapping (simplified)
        asset_to_id = {
            "btc": 1, "eth": 2, "bnb": 3, "ada": 4, "sol": 5,
            "dot": 6, "doge": 7, "shib": 8, "avax": 9, "matic": 10,
            # Add more as needed
        }
        
        result = []
        for asset_list in asset_lists:
            ids = [asset_to_id.get(asset, 0) for asset in asset_list]  # 0 for unknown
            if not ids:
                ids = [0]  # Default if no assets
            result.append(ids[0])  # Take first asset for simplicity
        
        return result
    
    def _get_market_condition_id(self, condition: str) -> int:
        """Convert market condition to ID"""
        
        conditions = {
            "bull": 1, "bullish": 1, "bull_run": 1,
            "bear": 2, "bearish": 2, "bear_market": 2,
            "sideways": 3, "crab": 3, "consolidation": 3,
            "volatile": 4, "high_volatility": 4,
            "crash": 5, "panic": 5,
            "recovery": 6, "bounce": 6,
            "bubble": 7, "euphoria": 7,
            "correction": 8, "pullback": 8,
        }
        
        return conditions.get(condition.lower(), 0)
    
    def _get_price_movement_label(self, class_id: int) -> str:
        """Get price movement label"""
        labels = ["strong_down", "down", "neutral", "up", "strong_up"]
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
    
    def _get_meme_category(self, class_id: int) -> str:
        """Get meme category"""
        categories = ["none", "doge_memes", "diamond_hands", "to_moon", "ape_together"]
        return categories[class_id] if class_id < len(categories) else "unknown"
    
    def _get_manipulation_type(self, class_id: int) -> str:
        """Get manipulation type"""
        types = ["none", "pump_and_dump", "wash_trading"]
        return types[class_id] if class_id < len(types) else "unknown"
    
    def preprocess_crypto_text_advanced(self, text: str) -> str:
        """
        Advanced crypto text preprocessing
        """
        
        # Start with basic crypto preprocessing
        text = self.preprocess_crypto_text(text)
        
        # Advanced crypto-specific preprocessing
        
        # Normalize crypto addresses (simplified)
        text = re.sub(r'\b0x[a-fA-F0-9]{40}\b', '[CRYPTO_ADDRESS]', text)
        
        # Normalize transaction hashes
        text = re.sub(r'\b0x[a-fA-F0-9]{64}\b', '[TX_HASH]', text)
        
        # Handle emoji sequences common in crypto
        moon_emojis = r'🚀+|🌙+|💎+|💰+|📈+|📉+'
        text = re.sub(moon_emojis, '[CRYPTO_EMOJI]', text)
        
        # Normalize common crypto phrases
        crypto_phrases = {
            'to the moon': '[TO_MOON]',
            'diamond hands': '[DIAMOND_HANDS]',
            'paper hands': '[PAPER_HANDS]',
            'buy the dip': '[BUY_DIP]',
            'when lambo': '[WHEN_LAMBO]',
            'this is the way': '[THIS_WAY]',
            'number go up': '[NUMBER_UP]',
            'have fun staying poor': '[HFSP]',
            'we\'re all gonna make it': '[WAGMI]',
        }
        
        for phrase, replacement in crypto_phrases.items():
            text = re.sub(rf'\b{phrase}\b', replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def analyze_crypto_sentiment_comprehensive(
        self,
        texts: List[str],
        assets: Optional[List[str]] = None,
        timeframe: str = "1h",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Comprehensive crypto sentiment analysis
        
        Args:
            texts: List of texts to analyze
            assets: List of assets to focus on
            timeframe: Analysis timeframe
            
        Returns:
            Comprehensive market analysis
        """
        
        if not texts:
            return {"error": "No texts provided"}
        
        # Get predictions
        predictions = self.predict_crypto(texts, assets=assets, **kwargs)
        
        # Aggregate by asset
        asset_sentiment = defaultdict(lambda: {
            "positive": 0, "neutral": 0, "negative": 0,
            "price_up": 0, "price_down": 0, "price_neutral": 0,
            "texts_count": 0, "total_confidence": 0
        })
        
        overall_stats = {
            "total_texts": len(texts),
            "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
            "price_prediction": {"up": 0, "down": 0, "neutral": 0},
            "manipulation_risk": {"high": 0, "medium": 0, "low": 0},
            "meme_activity": {"high": 0, "medium": 0, "low": 0},
            "average_volatility": 0.0,
        }
        
        total_volatility = 0
        
        for pred in predictions:
            sentiment = pred["sentiment"]["label"]
            overall_stats["sentiment_distribution"][sentiment] += 1
            
            if "price_movement" in pred:
                price_label = pred["price_movement"]["label"]
                if price_label in ["up", "strong_up"]:
                    overall_stats["price_prediction"]["up"] += 1
                elif price_label in ["down", "strong_down"]:
                    overall_stats["price_prediction"]["down"] += 1
                else:
                    overall_stats["price_prediction"]["neutral"] += 1
            
            if "volatility" in pred:
                total_volatility += pred["volatility"]["score"]
            
            # Asset-specific aggregation
            for asset in pred.get("assets_detected", ["general"]):
                asset_stats = asset_sentiment[asset]
                asset_stats[sentiment] += 1
                asset_stats["texts_count"] += 1
                asset_stats["total_confidence"] += pred["sentiment"]["confidence"]
        
        # Calculate averages
        overall_stats["average_volatility"] = total_volatility / len(predictions) if predictions else 0
        
        # Calculate asset-specific metrics
        asset_analysis = {}
        for asset, stats in asset_sentiment.items():
            if stats["texts_count"] > 0:
                total = stats["texts_count"]
                asset_analysis[asset] = {
                    "sentiment_distribution": {
                        k: v/total for k, v in stats.items() 
                        if k in ["positive", "neutral", "negative"]
                    },
                    "average_confidence": stats["total_confidence"] / total,
                    "texts_analyzed": total,
                    "dominant_sentiment": max(
                        ["positive", "neutral", "negative"],
                        key=lambda x: stats[x]
                    )
                }
        
        # Market outlook
        pos_ratio = overall_stats["sentiment_distribution"]["positive"] / len(texts)
        neg_ratio = overall_stats["sentiment_distribution"]["negative"] / len(texts)
        
        if pos_ratio > 0.6:
            market_outlook = "very_bullish"
        elif pos_ratio > 0.4:
            market_outlook = "bullish"
        elif neg_ratio > 0.6:
            market_outlook = "very_bearish"
        elif neg_ratio > 0.4:
            market_outlook = "bearish"
        else:
            market_outlook = "neutral"
        
        return {
            "market_outlook": market_outlook,
            "overall_statistics": overall_stats,
            "asset_analysis": asset_analysis,
            "timeframe": timeframe,
            "analysis_timestamp": time.time(),
        }


# Factory function
def create_crypto_bert(
    variant: str = "crypto",
    full_crypto_features: bool = True,
    **kwargs
) -> CryptoBERT:
    """
    Factory function for creation CryptoBERT model
    
    Args:
        variant: Model variant
        full_crypto_features: Enable all crypto features
        **kwargs: Additional parameters
        
    Returns:
        Configured CryptoBERT model
    """
    
    model = CryptoBERT(
        model_variant=variant,
        enable_multi_asset=full_crypto_features,
        market_condition_aware=full_crypto_features,
        technical_analysis_integration=full_crypto_features,
        social_sentiment_aggregation=full_crypto_features,
        meme_detection=full_crypto_features,
        influencer_tracking=full_crypto_features,
        manipulation_detection=full_crypto_features,
        **kwargs
    )
    
    return model