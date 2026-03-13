"""
API module for Prophet forecasting system.

Provides FastAPI-based REST API and WebSocket endpoints for cryptocurrency
price forecasting with enterprise patterns.
"""

from .forecast_api import create_forecast_app, run_server

__all__ = [
    "create_forecast_app",
    "run_server"
]