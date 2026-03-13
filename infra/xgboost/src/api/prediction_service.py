"""
Prediction Service API for XGBoost Models
========================================

FastAPI-based prediction service for real-time inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from loguru import logger


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: Dict[str, float]
    model_name: Optional[str] = "default"


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    confidence: Optional[float] = None
    model_used: str
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features: List[Dict[str, float]]
    model_name: Optional[str] = "default"


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[float]
    model_used: str
    count: int


class PredictionService:
    """Core prediction service"""
    
    def __init__(self):
        self.models_: Dict[str, Any] = {}
        self.feature_names_: List[str] = []
    
    def load_model(self, model: Any, model_name: str = "default"):
        """Load a trained model"""
        self.models_[model_name] = model
        logger.info(f"Model '{model_name}' loaded successfully")
    
    def predict_single(
        self, 
        features: Dict[str, float], 
        model_name: str = "default"
    ) -> float:
        """Single prediction"""
        
        if model_name not in self.models_:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([features])
        
        # Predict
        model = self.models_[model_name]
        prediction = model.predict(feature_df)[0]
        
        return float(prediction)
    
    def predict_batch(
        self, 
        features_list: List[Dict[str, float]], 
        model_name: str = "default"
    ) -> List[float]:
        """Batch prediction"""
        
        if model_name not in self.models_:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Predict
        model = self.models_[model_name]
        predictions = model.predict(features_df)
        
        return predictions.tolist()


class ModelServer:
    """FastAPI model server"""
    
    def __init__(self, prediction_service: PredictionService):
        self.app = FastAPI(title="XGBoost Model Server", version="1.0.0")
        self.prediction_service = prediction_service
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "models_loaded": len(self.prediction_service.models_)}
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            try:
                prediction = self.prediction_service.predict_single(
                    request.features, request.model_name
                )
                
                return PredictionResponse(
                    prediction=prediction,
                    model_used=request.model_name,
                    timestamp=pd.Timestamp.now().isoformat()
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/predict_batch", response_model=BatchPredictionResponse)
        async def predict_batch(request: BatchPredictionRequest):
            try:
                predictions = self.prediction_service.predict_batch(
                    request.features, request.model_name
                )
                
                return BatchPredictionResponse(
                    predictions=predictions,
                    model_used=request.model_name,
                    count=len(predictions)
                )
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/models")
        async def list_models():
            return {"models": list(self.prediction_service.models_.keys())}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        uvicorn.run(self.app, host=host, port=port)


class BatchPredictor:
    """Batch prediction utility"""
    
    def __init__(self, model: Any):
        self.model = model
    
    def predict_csv(self, input_path: str, output_path: str):
        """Predict on CSV file"""
        
        # Load data
        data = pd.read_csv(input_path)
        
        # Predict
        predictions = self.model.predict(data)
        
        # Save results
        results = data.copy()
        results['prediction'] = predictions
        results.to_csv(output_path, index=False)
        
        logger.info(f"Batch predictions saved to {output_path}")
    
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on DataFrame"""
        
        predictions = self.model.predict(df)
        
        result_df = df.copy()
        result_df['prediction'] = predictions
        
        return result_df