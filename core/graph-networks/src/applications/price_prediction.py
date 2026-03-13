"""
Price Prediction Application using Graph Neural Networks
=========================================================

Production-ready price prediction system using GNN ensemble for 
cryptocurrency market forecasting with enterprise patterns.

Features:
- Multi-step price forecasting
- Uncertainty quantification
- Real-time prediction pipeline
- Model interpretability
- Production deployment ready

Author: ML-Framework ML Team  
Version: 1.0.0
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
import warnings

# Imports from our modules
from ..models import create_crypto_gnn_ensemble, GraphNeuralNetworkEnsemble
from ..graph_construction import create_correlation_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PricePredictionConfig:
    """Configuration for crypto price prediction system"""
    
    # Prediction parameters
    prediction_horizon: int = 24  # Hours ahead to predict
    prediction_steps: int = 1     # Multi-step prediction
    confidence_intervals: bool = True
    
    # Model parameters
    ensemble_models: List[str] = None
    input_features: int = 64
    hidden_dim: int = 128
    
    # Data parameters
    lookback_window: int = 168   # Hours of historical data
    correlation_window: int = 48  # Hours for correlation calculation
    min_correlation: float = 0.3
    
    # Production parameters
    batch_prediction: bool = True
    real_time_update: bool = True
    model_retraining_interval: int = 24  # Hours
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['gcn', 'gat', 'graphsage', 'mpnn']

class CryptoPricePredictor:
    """
    Main class for crypto price prediction using GNN ensemble
    
    Production ML Pipeline
    """
    
    def __init__(self, config: PricePredictionConfig):
        self.config = config
        
        # Initialize GNN ensemble
        self.ensemble, self.trainer = create_crypto_gnn_ensemble(
            input_dim=config.input_features,
            hidden_dim=config.hidden_dim,
            enable_all_models=True,
            enable_uncertainty=True
        )
        
        # State tracking
        self.is_trained = False
        self.last_training_time = None
        self.prediction_history = []
        
        logger.info("Initialized CryptoPricePredictor")
    
    def prepare_data(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        external_features: Optional[pd.DataFrame] = None
    ) -> List[Data]:
        """
        Preparation data for training/predictions
        
        Args:
            price_data: Historical price data [timestamp, assets]
            volume_data: Volume data [timestamp, assets] 
            external_features: Additional features [timestamp, features]
            
        Returns:
            List[Data]: Prepared graph data for each timestamp
        """
        graphs = []
        timestamps = price_data.index
        
        for i, timestamp in enumerate(timestamps[self.config.lookback_window:]):
            # Historical window for this timestamp
            start_idx = i
            end_idx = i + self.config.lookback_window
            
            window_price_data = price_data.iloc[start_idx:end_idx]
            window_volume_data = volume_data.iloc[start_idx:end_idx] if volume_data is not None else None
            
            # Create correlation graph for this window
            try:
                graph = create_correlation_graph(
                    price_data=window_price_data,
                    volume_data=window_volume_data,
                    correlation_method='pearson',
                    min_correlation=self.config.min_correlation,
                    time_window=self.config.correlation_window
                )
                
                # Add target (next price change)
                if i + self.config.lookback_window + self.config.prediction_horizon < len(price_data):
                    future_prices = price_data.iloc[i + self.config.lookback_window + self.config.prediction_horizon]
                    current_prices = price_data.iloc[i + self.config.lookback_window]
                    
                    # Price change percentage as target
                    price_changes = ((future_prices - current_prices) / current_prices).values
                    graph.y = torch.tensor(price_changes.mean(), dtype=torch.float32).unsqueeze(0)
                    
                    # Add timestamp info
                    graph.timestamp = timestamp
                    graphs.append(graph)
                    
            except Exception as e:
                logger.warning(f"Error creation graph for {timestamp}: {e}")
                continue
        
        logger.info(f" {len(graphs)} graphs for training/predictions")
        return graphs
    
    def train(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Training GNN ensemble for price prediction
        """
        logger.info("Start training model predictions prices")
        
        # Preparation data
        graphs = self.prepare_data(price_data, volume_data)
        
        if len(graphs) < 10:
            raise ValueError("Insufficient data for training")
        
        # Train/validation split
        split_idx = int(len(graphs) * (1 - validation_split))
        train_graphs = graphs[:split_idx]
        val_graphs = graphs[split_idx:]
        
        # Create data loaders
        from torch_geometric.loader import DataLoader
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
        for epoch in range(epochs):
            # Training
            epoch_metrics = self.trainer.train_epoch(train_loader, val_loader)
            
            # Update history
            for key, value in epoch_metrics.items():
                if key in history:
                    history[key].append(value)
            
            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={epoch_metrics.get('train_ensemble_loss', 0):.4f}, "
                    f"Val Loss={epoch_metrics.get('val_loss', 0):.4f}"
                )
        
        self.is_trained = True
        self.last_training_time = pd.Timestamp.now()
        
        logger.info("Training completed successfully")
        return history
    
    def predict(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        return_uncertainty: bool = True
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Prediction prices with uncertainty quantification
        
        Returns:
            Dict with predictions, uncertainty, and confidence intervals
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. train() first.")
        
        # Prepare latest data for prediction
        graphs = self.prepare_data(price_data, volume_data)
        
        if not graphs:
            raise ValueError("Not succeeded data for predictions")
        
        # Use latest graph for prediction
        latest_graph = graphs[-1]
        
        # Prediction with uncertainty
        if return_uncertainty:
            results = self.ensemble.predict_with_uncertainty([latest_graph])
            
            return {
                'prediction': results['prediction'][0],
                'uncertainty': results['uncertainty'][0],
                'confidence_interval_lower': results['confidence_interval_lower'][0],
                'confidence_interval_upper': results['confidence_interval_upper'][0],
                'individual_predictions': {
                    k: v[0] for k, v in results['individual_predictions'].items()
                },
                'prediction_timestamp': pd.Timestamp.now(),
                'data_timestamp': latest_graph.timestamp if hasattr(latest_graph, 'timestamp') else None
            }
        else:
            # Simple prediction
            self.ensemble.eval()
            with torch.no_grad():
                prediction = self.ensemble(latest_graph)
            
            return {
                'prediction': prediction.cpu().numpy()[0],
                'prediction_timestamp': pd.Timestamp.now(),
                'data_timestamp': latest_graph.timestamp if hasattr(latest_graph, 'timestamp') else None
            }
    
    def predict_multi_step(
        self, 
        price_data: pd.DataFrame,
        steps: int,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Multi-step ahead prediction
        """
        predictions = []
        uncertainties = []
        
        current_data = price_data.copy()
        
        for step in range(steps):
            # Predict next step
            result = self.predict(current_data, volume_data, return_uncertainty=True)
            
            predictions.append(result['prediction'])
            uncertainties.append(result['uncertainty'])
            
            # Update data with prediction for next step
            # (Simplified approach - in production would need more sophisticated data augmentation)
            last_timestamp = current_data.index[-1]
            next_timestamp = last_timestamp + pd.Timedelta(hours=self.config.prediction_horizon)
            
            # Create new row with predicted values
            new_row = current_data.iloc[-1].copy()
            # Apply predicted change
            new_row = new_row * (1 + result['prediction'])
            
            # Add to data
            current_data.loc[next_timestamp] = new_row
        
        return {
            'predictions': np.array(predictions),
            'uncertainties': np.array(uncertainties),
            'prediction_horizons': list(range(1, steps + 1))
        }
    
    def get_feature_importance(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Feature importance analysis using model attention weights
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        graphs = self.prepare_data(price_data)
        if not graphs:
            return {}
        
        latest_graph = graphs[-1]
        
        # Get model importance from ensemble
        model_importance = self.ensemble.get_model_importance()
        
        # Get attention weights from GAT model if available
        attention_importance = {}
        try:
            if hasattr(self.ensemble.models, 'gat'):
                gat_model = self.ensemble.models['gat']
                attention_weights = gat_model.get_attention_weights(latest_graph)
                
                # Aggregate attention weights across layers and heads
                for i, att_weights in enumerate(attention_weights):
                    attention_importance[f'layer_{i}_attention'] = att_weights.mean().item()
                    
        except Exception as e:
            logger.warning(f"Not succeeded attention weights: {e}")
        
        return {
            'model_importance': model_importance,
            'attention_importance': attention_importance,
            'graph_statistics': {
                'num_nodes': latest_graph.num_nodes,
                'num_edges': latest_graph.edge_index.size(1),
                'avg_node_degree': latest_graph.edge_index.size(1) / latest_graph.num_nodes
            }
        }
    
    def evaluate_performance(
        self, 
        test_price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        # Prepare test data
        test_graphs = self.prepare_data(test_price_data, volume_data)
        
        if not test_graphs:
            raise ValueError("No test data")
        
        predictions = []
        actuals = []
        uncertainties = []
        
        for graph in test_graphs:
            # Prediction
            with torch.no_grad():
                pred_result = self.ensemble.predict_with_uncertainty([graph])
                predictions.append(pred_result['prediction'][0])
                uncertainties.append(pred_result['uncertainty'][0])
                actuals.append(graph.y.item())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        uncertainties = np.array(uncertainties)
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # Uncertainty calibration
        avg_uncertainty = np.mean(uncertainties)
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'avg_uncertainty': avg_uncertainty,
            'num_predictions': len(predictions)
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model"""
        self.trainer.save_ensemble(filepath)
        logger.info(f"Model saved in {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load trained model"""
        self.trainer.load_ensemble(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")

def create_price_prediction_system(
    input_features: int = 64,
    prediction_horizon: int = 24,
    **kwargs
) -> CryptoPricePredictor:
    """
    Factory function for creating price prediction system
    
    Factory for ML Applications
    """
    config = PricePredictionConfig(
        input_features=input_features,
        prediction_horizon=prediction_horizon,
        **kwargs
    )
    
    return CryptoPricePredictor(config)

__all__ = [
    'PricePredictionConfig',
    'CryptoPricePredictor', 
    'create_price_prediction_system'
]