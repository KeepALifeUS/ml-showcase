"""
Ensemble Model for Enhanced Sentiment Analysis

Combines multiple transformer models for improved accuracy and robustness
in cryptocurrency sentiment analysis.

Author: ML-Framework Team
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .bert_sentiment import BERTSentiment, SentimentOutput
from .finbert_model import FinBERTModel
from .roberta_sentiment import RoBERTaSentiment
from .distilbert_model import DistilBERTModel
from .crypto_bert import CryptoBERT
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.profiler import Profiler

logger = Logger(__name__).get_logger()


@dataclass
class EnsembleOutput:
    """Structured output for ensemble predictions"""
    ensemble_sentiment: Dict[str, Any]
    individual_predictions: Dict[str, Dict[str, Any]]
    confidence_scores: Dict[str, float]
    model_agreements: Dict[str, float]
    processing_times: Dict[str, float]
    ensemble_confidence: float
    total_processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API"""
        return {
            "ensemble_sentiment": self.ensemble_sentiment,
            "individual_predictions": self.individual_predictions,
            "confidence_scores": self.confidence_scores,
            "model_agreements": self.model_agreements,
            "processing_times": self.processing_times,
            "ensemble_confidence": self.ensemble_confidence,
            "total_processing_time": self.total_processing_time,
            "recommendation": self._get_recommendation(),
        }
    
    def _get_recommendation(self) -> str:
        """Get trading recommendation based on ensemble"""
        sentiment = self.ensemble_sentiment["label"]
        confidence = self.ensemble_confidence
        
        if sentiment == "positive" and confidence > 0.8:
            return "strong_buy"
        elif sentiment == "positive" and confidence > 0.6:
            return "buy"
        elif sentiment == "negative" and confidence > 0.8:
            return "strong_sell"
        elif sentiment == "negative" and confidence > 0.6:
            return "sell"
        else:
            return "hold"


class EnsembleModel(nn.Module):
    """
    Ensemble Model for Enhanced Sentiment Analysis
    
    Combines multiple transformer models with sophisticated aggregation:
    - Weighted voting based on model confidence
    - Dynamic weight adjustment based on performance
    - Uncertainty quantification
    - Model agreement analysis
    - Robust prediction aggregation
    - Real-time model selection
    """
    
    ENSEMBLE_STRATEGIES = {
        "weighted_voting": "Weighted voting based on confidence scores",
        "majority_voting": "Simple majority voting",
        "confidence_weighted": "Weight by individual model confidence",
        "performance_weighted": "Weight by historical performance",
        "dynamic_selection": "Dynamic model selection based on input",
        "stacking": "Meta-learner stacking approach",
        "bayesian_aggregation": "Bayesian model averaging",
    }
    
    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        ensemble_strategy: str = "weighted_voting",
        model_weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.5,
        agreement_threshold: float = 0.7,
        enable_uncertainty_quantification: bool = True,
        enable_model_selection: bool = True,
        parallel_inference: bool = True,
        cache_predictions: bool = True,
        device: Optional[str] = None,
        config: Optional[Config] = None,
        **kwargs
    ):
        super().__init__()
        
        self.config = config or Config()
        self.ensemble_strategy = ensemble_strategy
        self.confidence_threshold = confidence_threshold
        self.agreement_threshold = agreement_threshold
        self.enable_uncertainty_quantification = enable_uncertainty_quantification
        self.enable_model_selection = enable_model_selection
        self.parallel_inference = parallel_inference
        self.cache_predictions = cache_predictions
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.profiler = Profiler()
        self.model_performance_history = {}
        
        # Initialize models
        if models is None:
            models = self._create_default_models()
        
        self.models = nn.ModuleDict(models)
        self.model_names = list(self.models.keys())
        self.num_models = len(self.models)
        
        # Initialize weights
        if model_weights is None:
            model_weights = {name: 1.0 for name in self.model_names}
        
        self.model_weights = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(weight, dtype=torch.float32))
            for name, weight in model_weights.items()
        })
        
        # Ensemble components
        self._setup_ensemble_components()
        
        # Cache for predictions
        if self.cache_predictions:
            self.prediction_cache = {}
            self.cache_size_limit = 1000
        
        # Model metadata
        self.ensemble_metadata = {
            "ensemble_type": "multi_transformer",
            "strategy": ensemble_strategy,
            "num_models": self.num_models,
            "model_types": [type(model).__name__ for model in self.models.values()],
            "parallel_inference": parallel_inference,
            "uncertainty_quantification": enable_uncertainty_quantification,
        }
        
        self.to(self.device)
        logger.info(f"Initialized Ensemble with {self.num_models} models using {ensemble_strategy}")
    
    def _create_default_models(self) -> Dict[str, nn.Module]:
        """Create default ensemble of models"""
        
        logger.info("Creating default ensemble models...")
        
        models = {}
        
        try:
            # BERT-based models
            models["bert_sentiment"] = BERTSentiment(
                model_name="bert-base-uncased",
                num_labels=3,
                device=self.device
            )
            
            models["finbert"] = FinBERTModel(
                model_variant="sentiment",
                num_labels=3,
                device=self.device
            )
            
            models["roberta"] = RoBERTaSentiment(
                model_variant="sentiment",
                num_labels=3,
                device=self.device
            )
            
            models["distilbert"] = DistilBERTModel(
                model_variant="base",
                num_labels=3,
                device=self.device
            )
            
            models["crypto_bert"] = CryptoBERT(
                model_variant="crypto",
                num_labels=3,
                device=self.device
            )
            
        except Exception as e:
            logger.warning(f"Error creating some models: {e}")
            # Fallback to available models
            models["bert_base"] = BERTSentiment(device=self.device)
        
        logger.info(f"Created {len(models)} models for ensemble")
        return models
    
    def _setup_ensemble_components(self):
        """Setup ensemble-specific neural components"""
        
        # Meta-learner for stacking
        if self.ensemble_strategy == "stacking":
            self.meta_learner = nn.Sequential(
                nn.Linear(self.num_models * 3, 64),  # 3 classes per model
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 3)  # Final 3 classes
            )
        
        # Uncertainty quantification components
        if self.enable_uncertainty_quantification:
            self.uncertainty_estimator = nn.Sequential(
                nn.Linear(self.num_models, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        # Model selection components
        if self.enable_model_selection:
            self.model_selector = nn.Sequential(
                nn.Linear(768, 128),  # Assuming 768-dim input features
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_models),
                nn.Softmax(dim=-1)
            )
        
        # Attention-based aggregation
        self.ensemble_attention = nn.MultiheadAttention(
            embed_dim=self.num_models,
            num_heads=min(4, self.num_models),
            dropout=0.1,
            batch_first=True
        )
        
        # Confidence calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(self.num_models, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_individual: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass through ensemble
        """
        
        individual_outputs = {}
        individual_logits = []
        individual_losses = []
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                with self._performance_context(f"model_{name}"):
                    # Get model prediction
                    if hasattr(model, 'bert'):  # Transformer models
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True,
                            **kwargs
                        )
                    else:  # Other model types
                        outputs = model(input_ids, attention_mask, labels, **kwargs)
                    
                    if isinstance(outputs, dict):
                        logits = outputs.get("logits", outputs.get("sentiment_logits"))
                        loss = outputs.get("loss")
                    else:
                        logits = outputs[0]
                        loss = outputs[0] if labels is not None else None
                    
                    individual_outputs[name] = outputs
                    individual_logits.append(logits)
                    
                    if loss is not None:
                        individual_losses.append(loss)
            
            except Exception as e:
                logger.warning(f"Error in model {name}: {e}")
                # Use zero logits as fallback
                batch_size = input_ids.shape[0]
                fallback_logits = torch.zeros(batch_size, 3, device=self.device)
                individual_logits.append(fallback_logits)
        
        # Stack individual logits
        if individual_logits:
            stacked_logits = torch.stack(individual_logits, dim=1)  # [batch, num_models, num_classes]
        else:
            batch_size = input_ids.shape[0]
            stacked_logits = torch.zeros(batch_size, self.num_models, 3, device=self.device)
        
        # Apply ensemble strategy
        ensemble_logits = self._apply_ensemble_strategy(stacked_logits, **kwargs)
        
        # Calculate ensemble loss
        ensemble_loss = None
        if labels is not None and individual_losses:
            # Weighted combination of individual losses
            weights = F.softmax(torch.stack([self.model_weights[name] for name in self.model_names]), dim=0)
            ensemble_loss = sum(w * loss for w, loss in zip(weights, individual_losses))
        
        # Uncertainty quantification
        uncertainty_score = None
        if self.enable_uncertainty_quantification:
            model_agreements = self._calculate_model_agreement(stacked_logits)
            uncertainty_score = self.uncertainty_estimator(model_agreements.unsqueeze(0))
        
        return {
            "ensemble_logits": ensemble_logits,
            "individual_logits": stacked_logits if return_individual else None,
            "individual_outputs": individual_outputs if return_individual else None,
            "uncertainty_score": uncertainty_score,
            "ensemble_loss": ensemble_loss,
            "model_weights": {name: weight.item() for name, weight in self.model_weights.items()},
        }
    
    def _apply_ensemble_strategy(
        self, 
        stacked_logits: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """Apply selected ensemble strategy"""
        
        if self.ensemble_strategy == "weighted_voting":
            return self._weighted_voting(stacked_logits)
        elif self.ensemble_strategy == "majority_voting":
            return self._majority_voting(stacked_logits)
        elif self.ensemble_strategy == "confidence_weighted":
            return self._confidence_weighted_voting(stacked_logits)
        elif self.ensemble_strategy == "stacking":
            return self._stacking_aggregation(stacked_logits)
        elif self.ensemble_strategy == "bayesian_aggregation":
            return self._bayesian_aggregation(stacked_logits)
        else:
            # Default to weighted voting
            return self._weighted_voting(stacked_logits)
    
    def _weighted_voting(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Weighted voting ensemble"""
        
        # Get normalized weights
        weights = F.softmax(torch.stack([self.model_weights[name] for name in self.model_names]), dim=0)
        weights = weights.unsqueeze(0).unsqueeze(-1)  # [1, num_models, 1]
        
        # Weighted sum
        ensemble_logits = (stacked_logits * weights).sum(dim=1)
        
        return ensemble_logits
    
    def _majority_voting(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Simple majority voting"""
        
        # Convert logits to predictions
        predictions = torch.argmax(stacked_logits, dim=-1)  # [batch, num_models]
        
        # Count votes for each class
        batch_size, num_models = predictions.shape
        ensemble_predictions = []
        
        for b in range(batch_size):
            votes = torch.bincount(predictions[b], minlength=3)
            ensemble_predictions.append(torch.argmax(votes))
        
        # Convert back to logits (one-hot)
        ensemble_predictions = torch.stack(ensemble_predictions)
        ensemble_logits = F.one_hot(ensemble_predictions, num_classes=3).float()
        
        return ensemble_logits
    
    def _confidence_weighted_voting(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Confidence-weighted voting"""
        
        # Calculate confidence for each model
        confidences = torch.max(F.softmax(stacked_logits, dim=-1), dim=-1)[0]  # [batch, num_models]
        
        # Normalize confidences as weights
        weights = F.softmax(confidences, dim=-1).unsqueeze(-1)  # [batch, num_models, 1]
        
        # Weighted sum
        ensemble_logits = (stacked_logits * weights).sum(dim=1)
        
        return ensemble_logits
    
    def _stacking_aggregation(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Meta-learner stacking"""
        
        # Flatten individual predictions
        batch_size = stacked_logits.shape[0]
        flattened = stacked_logits.view(batch_size, -1)  # [batch, num_models * num_classes]
        
        # Apply meta-learner
        ensemble_logits = self.meta_learner(flattened)
        
        return ensemble_logits
    
    def _bayesian_aggregation(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Bayesian model averaging"""
        
        # Convert to probabilities
        probs = F.softmax(stacked_logits, dim=-1)  # [batch, num_models, num_classes]
        
        # Calculate model uncertainties (simplified)
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [batch, num_models]
        
        # Weight by inverse uncertainty
        weights = F.softmax(-entropies, dim=-1).unsqueeze(-1)  # [batch, num_models, 1]
        
        # Weighted average of probabilities
        ensemble_probs = (probs * weights).sum(dim=1)
        
        # Convert back to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        return ensemble_logits
    
    def predict_ensemble(
        self,
        texts: Union[str, List[str]],
        return_individual: bool = True,
        return_uncertainty: bool = True,
        enable_caching: bool = True,
        **kwargs
    ) -> Union[EnsembleOutput, List[EnsembleOutput]]:
        """
        Ensemble prediction with comprehensive analysis
        
        Args:
            texts: Input text(s)
            return_individual: Return individual model predictions
            return_uncertainty: Return uncertainty estimates
            enable_caching: Use prediction caching
            
        Returns:
            Comprehensive ensemble prediction
        """
        
        self.eval()
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        results = []
        
        with torch.no_grad():
            start_time = time.time()
            
            if self.parallel_inference:
                # Parallel processing
                results = self._predict_parallel(
                    texts, return_individual, return_uncertainty, enable_caching, **kwargs
                )
            else:
                # Sequential processing
                results = self._predict_sequential(
                    texts, return_individual, return_uncertainty, enable_caching, **kwargs
                )
            
            total_time = time.time() - start_time
            
            # Update processing times
            for result in results:
                result.total_processing_time = total_time / len(texts)
        
        return results[0] if is_single else results
    
    def _predict_parallel(
        self,
        texts: List[str],
        return_individual: bool,
        return_uncertainty: bool,
        enable_caching: bool,
        **kwargs
    ) -> List[EnsembleOutput]:
        """Parallel prediction using ThreadPoolExecutor"""
        
        def predict_single_model(model_name: str, model: nn.Module, texts: List[str]) -> Dict[str, Any]:
            """Predict with single model"""
            try:
                start_time = time.time()
                
                if hasattr(model, 'predict'):
                    predictions = model.predict(texts, **kwargs)
                elif hasattr(model, 'predict_crypto'):
                    predictions = model.predict_crypto(texts, **kwargs)
                elif hasattr(model, 'predict_financial'):
                    predictions = model.predict_financial(texts, **kwargs)
                else:
                    # Fallback to tokenization and forward pass
                    predictions = self._fallback_predict(model, texts)
                
                processing_time = time.time() - start_time
                
                return {
                    "model_name": model_name,
                    "predictions": predictions,
                    "processing_time": processing_time,
                    "success": True,
                }
                
            except Exception as e:
                logger.warning(f"Error in parallel prediction for {model_name}: {e}")
                return {
                    "model_name": model_name,
                    "predictions": None,
                    "processing_time": 0.0,
                    "success": False,
                    "error": str(e),
                }
        
        # Execute predictions in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(4, self.num_models)) as executor:
            future_to_model = {
                executor.submit(predict_single_model, name, model, texts): name
                for name, model in self.models.items()
            }
            
            model_results = {}
            for future in as_completed(future_to_model):
                result = future.result()
                model_results[result["model_name"]] = result
        
        # Aggregate results
        for i, text in enumerate(texts):
            individual_preds = {}
            confidences = {}
            processing_times = {}
            agreements = {}
            
            all_probs = []
            all_confidences = []
            
            for model_name, result in model_results.items():
                if result["success"] and result["predictions"]:
                    if isinstance(result["predictions"], list):
                        pred = result["predictions"][i] if i < len(result["predictions"]) else result["predictions"][0]
                    else:
                        pred = result["predictions"]
                    
                    individual_preds[model_name] = pred
                    processing_times[model_name] = result["processing_time"] / len(texts)
                    
                    if isinstance(pred, dict) and "sentiment" in pred:
                        sentiment_info = pred["sentiment"]
                        confidences[model_name] = sentiment_info.get("confidence", 0.0)
                        probs = sentiment_info.get("probabilities", [0.33, 0.33, 0.33])
                    else:
                        # Handle SentimentOutput or other formats
                        confidences[model_name] = getattr(pred, "confidence", 0.0)
                        probs = getattr(pred, "probabilities", [0.33, 0.33, 0.33])
                        if torch.is_tensor(probs):
                            probs = probs.tolist()
                    
                    all_probs.append(probs)
                    all_confidences.append(confidences[model_name])
            
            # Calculate ensemble prediction
            if all_probs:
                ensemble_probs = np.mean(all_probs, axis=0)
                ensemble_class = int(np.argmax(ensemble_probs))
                ensemble_confidence = float(np.mean(all_confidences))
                
                # Calculate model agreement
                predictions = [np.argmax(probs) for probs in all_probs]
                agreement = np.mean([pred == ensemble_class for pred in predictions])
                
                ensemble_sentiment = {
                    "predicted_class": ensemble_class,
                    "probabilities": ensemble_probs.tolist(),
                    "confidence": ensemble_confidence,
                    "label": self._get_sentiment_label(ensemble_class),
                }
            else:
                # Fallback if no successful predictions
                ensemble_sentiment = {
                    "predicted_class": 1,  # neutral
                    "probabilities": [0.33, 0.34, 0.33],
                    "confidence": 0.0,
                    "label": "neutral",
                }
                agreement = 0.0
            
            # Create ensemble output
            result = EnsembleOutput(
                ensemble_sentiment=ensemble_sentiment,
                individual_predictions=individual_preds if return_individual else {},
                confidence_scores=confidences,
                model_agreements={"overall_agreement": agreement},
                processing_times=processing_times,
                ensemble_confidence=ensemble_confidence if 'ensemble_confidence' in locals() else 0.0,
                total_processing_time=0.0,  # Will be updated later
            )
            
            results.append(result)
        
        return results
    
    def _predict_sequential(
        self,
        texts: List[str],
        return_individual: bool,
        return_uncertainty: bool,
        enable_caching: bool,
        **kwargs
    ) -> List[EnsembleOutput]:
        """Sequential prediction processing"""
        
        results = []
        
        for text in texts:
            individual_preds = {}
            confidences = {}
            processing_times = {}
            
            all_probs = []
            all_confidences = []
            
            # Get predictions from each model
            for model_name, model in self.models.items():
                try:
                    start_time = time.time()
                    
                    if hasattr(model, 'predict'):
                        pred = model.predict(text, **kwargs)
                    else:
                        # Fallback prediction
                        pred = self._fallback_predict(model, [text])[0]
                    
                    processing_time = time.time() - start_time
                    
                    individual_preds[model_name] = pred
                    processing_times[model_name] = processing_time
                    
                    # Extract probabilities and confidence
                    if isinstance(pred, dict) and "sentiment" in pred:
                        sentiment_info = pred["sentiment"]
                        confidences[model_name] = sentiment_info.get("confidence", 0.0)
                        probs = sentiment_info.get("probabilities", [0.33, 0.33, 0.33])
                    else:
                        confidences[model_name] = getattr(pred, "confidence", 0.0)
                        probs = getattr(pred, "probabilities", [0.33, 0.33, 0.33])
                        if torch.is_tensor(probs):
                            probs = probs.tolist()
                    
                    all_probs.append(probs)
                    all_confidences.append(confidences[model_name])
                    
                except Exception as e:
                    logger.warning(f"Error in sequential prediction for {model_name}: {e}")
                    processing_times[model_name] = 0.0
                    confidences[model_name] = 0.0
            
            # Calculate ensemble prediction
            if all_probs:
                ensemble_probs = np.mean(all_probs, axis=0)
                ensemble_class = int(np.argmax(ensemble_probs))
                ensemble_confidence = float(np.mean(all_confidences))
                
                ensemble_sentiment = {
                    "predicted_class": ensemble_class,
                    "probabilities": ensemble_probs.tolist(),
                    "confidence": ensemble_confidence,
                    "label": self._get_sentiment_label(ensemble_class),
                }
            else:
                ensemble_sentiment = {
                    "predicted_class": 1,
                    "probabilities": [0.33, 0.34, 0.33],
                    "confidence": 0.0,
                    "label": "neutral",
                }
                ensemble_confidence = 0.0
            
            # Create result
            result = EnsembleOutput(
                ensemble_sentiment=ensemble_sentiment,
                individual_predictions=individual_preds if return_individual else {},
                confidence_scores=confidences,
                model_agreements={"overall_agreement": 0.0},  # Calculate if needed
                processing_times=processing_times,
                ensemble_confidence=ensemble_confidence,
                total_processing_time=0.0,
            )
            
            results.append(result)
        
        return results
    
    def _fallback_predict(self, model: nn.Module, texts: List[str]) -> List[Dict[str, Any]]:
        """Fallback prediction method"""
        
        # Simple tokenization and forward pass
        try:
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            else:
                # Use a default tokenizer
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            encoded = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("sentiment_logits"))
                else:
                    logits = outputs[0]
                
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
            
            results = []
            for i in range(len(texts)):
                results.append({
                    "sentiment": {
                        "predicted_class": preds[i].item(),
                        "probabilities": probs[i].tolist(),
                        "confidence": confidences[i].item(),
                        "label": self._get_sentiment_label(preds[i].item()),
                    }
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            # Return neutral predictions
            return [{
                "sentiment": {
                    "predicted_class": 1,
                    "probabilities": [0.33, 0.34, 0.33],
                    "confidence": 0.0,
                    "label": "neutral",
                }
            }] * len(texts)
    
    def _calculate_model_agreement(self, stacked_logits: torch.Tensor) -> torch.Tensor:
        """Calculate agreement between models"""
        
        # Convert to predictions
        predictions = torch.argmax(stacked_logits, dim=-1)  # [batch, num_models]
        
        agreements = []
        for b in range(predictions.shape[0]):
            batch_preds = predictions[b]
            # Calculate pairwise agreement
            agreement = torch.mean((batch_preds.unsqueeze(0) == batch_preds.unsqueeze(1)).float())
            agreements.append(agreement)
        
        return torch.stack(agreements)
    
    def _get_sentiment_label(self, class_id: int) -> str:
        """Get sentiment label from class ID"""
        labels = ["negative", "neutral", "positive"]
        return labels[class_id] if class_id < len(labels) else "unknown"
    
    def update_model_weights(self, performance_scores: Dict[str, float]) -> None:
        """Update model weights based on performance"""
        
        logger.info("Updating model weights based on performance...")
        
        for name, score in performance_scores.items():
            if name in self.model_weights:
                # Update weight (simple exponential moving average)
                current_weight = self.model_weights[name].item()
                new_weight = 0.9 * current_weight + 0.1 * score
                self.model_weights[name].data = torch.tensor(new_weight, dtype=torch.float32)
                
                logger.debug(f"Updated {name} weight: {current_weight:.3f} -> {new_weight:.3f}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get ensemble model statistics"""
        
        total_params = sum(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            for model in self.models.values()
        )
        
        model_stats = {}
        for name, model in self.models.items():
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_stats[name] = {
                "parameters": model_params,
                "weight": self.model_weights[name].item(),
                "type": type(model).__name__,
            }
        
        return {
            "ensemble_metadata": self.ensemble_metadata,
            "total_parameters": total_params,
            "model_statistics": model_stats,
            "current_weights": {name: weight.item() for name, weight in self.model_weights.items()},
            "performance_history": self.model_performance_history,
        }
    
    @contextmanager
    def _performance_context(self, operation: str):
        """Context manager for performance tracking"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.profiler.record(operation, elapsed)


# Factory function
def create_ensemble_model(
    model_types: Optional[List[str]] = None,
    ensemble_strategy: str = "weighted_voting",
    parallel_inference: bool = True,
    **kwargs
) -> EnsembleModel:
    """
    Factory function for creating ensemble model
    
    Args:
        model_types: List of model types to include
        ensemble_strategy: Ensemble aggregation strategy
        parallel_inference: Enable parallel model inference
        **kwargs: Additional parameters
        
    Returns:
        Configured EnsembleModel
    """
    
    if model_types is None:
        model_types = ["bert", "finbert", "roberta", "distilbert", "crypto_bert"]
    
    models = {}
    
    # Create specified models
    for model_type in model_types:
        try:
            if model_type == "bert":
                models["bert"] = BERTSentiment(**kwargs)
            elif model_type == "finbert":
                models["finbert"] = FinBERTModel(**kwargs)
            elif model_type == "roberta":
                models["roberta"] = RoBERTaSentiment(**kwargs)
            elif model_type == "distilbert":
                models["distilbert"] = DistilBERTModel(**kwargs)
            elif model_type == "crypto_bert":
                models["crypto_bert"] = CryptoBERT(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to create {model_type}: {e}")
    
    # Ensure at least one model
    if not models:
        logger.warning("No models created, using default BERT")
        models["bert_default"] = BERTSentiment(**kwargs)
    
    ensemble = EnsembleModel(
        models=models,
        ensemble_strategy=ensemble_strategy,
        parallel_inference=parallel_inference,
        **kwargs
    )
    
    return ensemble