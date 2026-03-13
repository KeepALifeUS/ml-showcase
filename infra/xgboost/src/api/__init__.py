"""
FastAPI Integration for XGBoost Model Serving
============================================

REST API endpoints for XGBoost model inference and management.
"""

from .prediction_service import (
    PredictionService,
    ModelServer,
    BatchPredictor,
)

__all__ = [
    "PredictionService",
    "ModelServer",
    "BatchPredictor",
]