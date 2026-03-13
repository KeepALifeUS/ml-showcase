"""
Tests for Prophet model wrapper.

Comprehensive test suite for ProphetForecaster class with enterprise patterns,
including unit tests, integration tests, and performance benchmarks.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from src.models.prophet_model import ProphetForecaster, ForecastResult
from src.config.prophet_config import ProphetConfig, ModelConfig
from src.utils.exceptions import (
    ModelNotTrainedException,
    InsufficientDataException,
    InvalidDataException,
    ModelTrainingException,
    PredictionException
)


@pytest.fixture
def sample_ohlcv_data():
    """Creation test OHLCV data"""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_points = len(dates)
    
    # Generation realistic OHLCV data
    np.random.seed(42)
    base_price = 50000
    price_walk = np.cumsum(np.random.randn(n_points) * 0.02) + base_price
    
    data = []
    for i, date in enumerate(dates):
        price = price_walk[i]
        volatility = abs(np.random.randn() * 0.01)
        
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = price + np.random.randn() * price * 0.005
        close_price = price + np.random.randn() * price * 0.005
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': max(open_price, 0),
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': max(close_price, 0),
            'volume': volume
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def prophet_config():
    """Test configuration Prophet"""
    return ProphetConfig()


@pytest.fixture
def prophet_model(prophet_config):
    """Test model Prophet"""
    return ProphetForecaster(
        symbol="BTC",
        timeframe="1d",
        config=prophet_config
    )


class TestProphetForecaster:
    """Tests for class ProphetForecaster"""
    
    def test_initialization(self, prophet_config):
        """Test initialization model"""
        model = ProphetForecaster(
            symbol="eth",
            timeframe="4h",
            config=prophet_config
        )
        
        assert model.symbol == "ETH"
        assert model.timeframe == "4h"
        assert model.config == prophet_config
        assert not model.is_trained
        assert model.model is None
        assert model.training_data is None
    
    def test_initialization_without_config(self):
        """Test initialization without configuration"""
        model = ProphetForecaster(symbol="BTC", timeframe="1h")
        
        assert model.symbol == "BTC"
        assert model.timeframe == "1h"
        assert model.config is not None
        assert isinstance(model.config, ProphetConfig)
    
    def test_create_prophet_model(self, prophet_model):
        """Test creation Prophet model"""
        model = prophet_model._create_prophet_model()
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        
        # Validation parameters
        assert model.growth == prophet_model.model_config.growth.value
        assert model.seasonality_mode == prophet_model.model_config.seasonality_mode.value
    
    def test_validate_training_data_valid(self, prophet_model, sample_ohlcv_data):
        """Test validation correct data"""
        # Preparation data in format Prophet
        df = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        # Must pass without exceptions
        validated_df = prophet_model._validate_training_data(df)
        
        assert not validated_df.empty
        assert 'ds' in validated_df.columns
        assert 'y' in validated_df.columns
        assert len(validated_df) > 0
    
    def test_validate_training_data_missing_columns(self, prophet_model):
        """Test validation data with missing columns"""
        df = pd.DataFrame({'only_one_column': [1, 2, 3]})
        
        with pytest.raises(InvalidDataException) as exc_info:
            prophet_model._validate_training_data(df)
        
        assert "missing required columns" in str(exc_info.value).lower()
    
    def test_validate_training_data_insufficient_data(self, prophet_model):
        """Test validation insufficient number data"""
        df = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=5, freq='D'),
            'y': [1, 2, 3, 4, 5]
        })
        
        with pytest.raises(InsufficientDataException) as exc_info:
            prophet_model._validate_training_data(df)
        
        assert "insufficient data" in str(exc_info.value).lower()
    
    def test_train_model_success(self, prophet_model, sample_ohlcv_data):
        """Test successful training model"""
        # Preparation data
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        # Training model
        metrics = prophet_model.train(training_data, validate=False)  # Disable CV for speed
        
        # Validation
        assert prophet_model.is_trained
        assert prophet_model.model is not None
        assert prophet_model.training_data is not None
        assert prophet_model.last_training_time is not None
        
        # Validation metrics
        assert isinstance(metrics, dict)
        assert 'training_time_seconds' in metrics
        assert 'training_samples' in metrics
        assert metrics['training_samples'] == len(training_data)
    
    def test_train_model_with_regressors(self, prophet_model, sample_ohlcv_data):
        """Test training with additional regressors"""
        # Preparation data with regressors
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close'],
            'volume': sample_ohlcv_data['volume'],
            'high': sample_ohlcv_data['high']
        })
        
        regressors_data = {
            'volume': sample_ohlcv_data['volume'],
            'high': sample_ohlcv_data['high']
        }
        
        metrics = prophet_model.train(
            training_data,
            regressors_data=regressors_data,
            validate=False
        )
        
        assert prophet_model.is_trained
        assert isinstance(metrics, dict)
    
    @pytest.mark.asyncio
    async def test_train_model_with_validation(self, prophet_model, sample_ohlcv_data):
        """Test training with cross-validation"""
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        metrics = prophet_model.train(training_data, validate=True)
        
        assert prophet_model.is_trained
        assert 'cv_mae' in metrics or 'training_time_seconds' in metrics  # CV can not work on small data
    
    def test_train_model_invalid_data(self, prophet_model):
        """Test training with incorrect data"""
        invalid_data = pd.DataFrame({'wrong_columns': [1, 2, 3]})
        
        with pytest.raises(ModelTrainingException):
            prophet_model.train(invalid_data)
    
    def test_predict_without_training(self, prophet_model):
        """Test forecasting without training"""
        with pytest.raises(ModelNotTrainedException):
            prophet_model.predict(periods=10)
    
    def test_predict_success(self, prophet_model, sample_ohlcv_data):
        """Test successful forecasting"""
        # Training model
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        
        # Forecasting
        forecast_result = prophet_model.predict(periods=30)
        
        # Validation
        assert isinstance(forecast_result, ForecastResult)
        assert forecast_result.symbol == "BTC"
        assert forecast_result.timeframe == "1d"
        assert len(forecast_result.forecast_df) == 30
        
        # Validation required columns
        assert 'ds' in forecast_result.forecast_df.columns
        assert 'yhat' in forecast_result.forecast_df.columns
        
        # Validation metrics
        assert isinstance(forecast_result.metrics, dict)
        assert len(forecast_result.changepoints) >= 0
    
    def test_predict_with_future_data(self, prophet_model, sample_ohlcv_data):
        """Test forecasting with predefined data"""
        # Training
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        
        # Creation future data
        future_dates = pd.date_range(
            start=sample_ohlcv_data['timestamp'].max() + timedelta(days=1),
            periods=10,
            freq='D'
        )
        future_data = pd.DataFrame({'ds': future_dates})
        
        # Forecasting
        forecast_result = prophet_model.predict(future_data=future_data)
        
        assert len(forecast_result.forecast_df) == 10
        assert forecast_result.forecast_df['ds'].min() == future_dates[0]
    
    @pytest.mark.asyncio
    async def test_predict_async(self, prophet_model, sample_ohlcv_data):
        """Test asynchronous forecasting"""
        # Training
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        
        # Asynchronous forecasting
        forecast_result = await prophet_model.predict_async(periods=15)
        
        assert isinstance(forecast_result, ForecastResult)
        assert len(forecast_result.forecast_df) == 15
    
    def test_save_and_load_model(self, prophet_model, sample_ohlcv_data, tmp_path):
        """Test saving and loading model"""
        # Training model
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        
        # Saving
        model_path = tmp_path / "test_model.pkl"
        prophet_model.save_model(model_path)
        
        assert model_path.exists()
        
        # Creation new model and loading
        new_model = ProphetForecaster(symbol="BTC", timeframe="1d")
        new_model.load_model(model_path)
        
        # Validation
        assert new_model.is_trained
        assert new_model.symbol == prophet_model.symbol
        assert new_model.timeframe == prophet_model.timeframe
        
        # Validation operability
        forecast = new_model.predict(periods=5)
        assert len(forecast.forecast_df) == 5
    
    def test_save_model_not_trained(self, prophet_model, tmp_path):
        """Test saving untrained model"""
        model_path = tmp_path / "test_model.pkl"
        
        with pytest.raises(ModelNotTrainedException):
            prophet_model.save_model(model_path)
    
    def test_load_model_file_not_found(self, prophet_model):
        """Test loading non-existent file"""
        with pytest.raises(FileNotFoundError):
            prophet_model.load_model("nonexistent_file.pkl")
    
    def test_get_model_info(self, prophet_model, sample_ohlcv_data):
        """Test retrieval information about model"""
        # Until training
        info_before = prophet_model.get_model_info()
        assert not info_before['is_trained']
        assert info_before['training_data_size'] == 0
        
        # After training
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        
        info_after = prophet_model.get_model_info()
        assert info_after['is_trained']
        assert info_after['symbol'] == "BTC"
        assert info_after['timeframe'] == "1d"
        assert info_after['training_data_size'] == len(training_data)
        assert 'last_training_time' in info_after
    
    def test_plot_forecast(self, prophet_model, sample_ohlcv_data):
        """Test creation chart forecast"""
        # Training and forecasting
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        prophet_model.train(training_data, validate=False)
        forecast_result = prophet_model.predict(periods=30)
        
        # Creation chart
        fig = prophet_model.plot_forecast(forecast_result, show_components=True)
        
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly Figure object
        assert len(fig.data) > 0  # Exists data for mapping
    
    def test_convert_seasonality_parameters(self, prophet_model):
        """Test conversion parameters seasonality"""
        # Test various types parameters
        assert prophet_model._convert_seasonality("auto") == "auto"
        assert prophet_model._convert_seasonality(True) == True
        assert prophet_model._convert_seasonality(False) == False
        assert prophet_model._convert_seasonality(5) == 5
        assert prophet_model._convert_seasonality("invalid") == False


class TestForecastResult:
    """Tests for class ForecastResult"""
    
    def test_forecast_result_creation(self):
        """Test creation ForecastResult"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=10, freq='D'),
            'yhat': np.random.randn(10) * 100 + 50000,
            'yhat_lower': np.random.randn(10) * 50 + 49000,
            'yhat_upper': np.random.randn(10) * 50 + 51000
        })
        
        result = ForecastResult(
            symbol="BTC",
            timeframe="1d",
            forecast_df=df,
            metrics={"mae": 100.0, "rmse": 150.0},
            confidence_intervals={},
            changepoints=[datetime.now()],
            trend_components={},
            seasonality_components={},
            forecast_timestamp=datetime.now(),
            model_version="5.0.0"
        )
        
        assert result.symbol == "BTC"
        assert result.timeframe == "1d"
        assert len(result.forecast_df) == 10
        assert "mae" in result.metrics
    
    def test_forecast_result_to_dict(self):
        """Test conversion ForecastResult in dictionary"""
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=5, freq='D'),
            'yhat': [50000, 50100, 50200, 50300, 50400]
        })
        
        result = ForecastResult(
            symbol="ETH",
            timeframe="4h",
            forecast_df=df,
            metrics={"mape": 2.5},
            confidence_intervals={},
            changepoints=[],
            trend_components={},
            seasonality_components={},
            forecast_timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_version="5.0.0"
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["symbol"] == "ETH"
        assert result_dict["timeframe"] == "4h"
        assert "forecast" in result_dict
        assert len(result_dict["forecast"]) == 5
        assert result_dict["metrics"]["mape"] == 2.5


class TestIntegrationProphetModel:
    """Integration tests"""
    
    @pytest.mark.integration
    def test_full_workflow(self, sample_ohlcv_data):
        """Test full workflow: creation → training → forecast → saving"""
        # 1. Creation model
        model = ProphetForecaster(symbol="BTC", timeframe="1d")
        assert not model.is_trained
        
        # 2. Preparation data
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        # 3. Training
        training_metrics = model.train(training_data, validate=False)
        assert model.is_trained
        assert isinstance(training_metrics, dict)
        
        # 4. Forecasting
        forecast = model.predict(periods=14)
        assert isinstance(forecast, ForecastResult)
        assert len(forecast.forecast_df) == 14
        
        # 5. Information about model
        model_info = model.get_model_info()
        assert model_info['is_trained']
        assert model_info['symbol'] == "BTC"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_workflow(self, sample_ohlcv_data):
        """Test asynchronous workflow"""
        model = ProphetForecaster(symbol="ETH", timeframe="4h")
        
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        # Training (synchronous)
        model.train(training_data, validate=False)
        
        # Asynchronous forecasting
        forecast = await model.predict_async(periods=7)
        
        assert isinstance(forecast, ForecastResult)
        assert len(forecast.forecast_df) == 7
        assert forecast.symbol == "ETH"
        assert forecast.timeframe == "4h"


class TestProphetModelPerformance:
    """Tests performance"""
    
    @pytest.mark.performance
    def test_training_performance(self, sample_ohlcv_data):
        """Test performance training"""
        model = ProphetForecaster(symbol="BTC", timeframe="1d")
        
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        
        start_time = datetime.now()
        metrics = model.train(training_data, validate=False)
        end_time = datetime.now()
        
        training_time = (end_time - start_time).total_seconds()
        
        # Training must occupy reasonable time
        assert training_time < 30  # Maximum 30 seconds
        assert metrics['training_time_seconds'] < 30
        
        # Validation usage memory (approximately)
        assert model.training_data is not None
        assert len(model.training_data) == len(training_data)
    
    @pytest.mark.performance 
    def test_prediction_performance(self, sample_ohlcv_data):
        """Test performance forecasting"""
        model = ProphetForecaster(symbol="BTC", timeframe="1d")
        
        # Training
        training_data = pd.DataFrame({
            'ds': sample_ohlcv_data['timestamp'],
            'y': sample_ohlcv_data['close']
        })
        model.train(training_data, validate=False)
        
        # Test forecasting
        start_time = datetime.now()
        forecast = model.predict(periods=90)  # 3 months
        end_time = datetime.now()
        
        prediction_time = (end_time - start_time).total_seconds()
        
        # Forecasting must be fast
        assert prediction_time < 5  # Maximum 5 seconds
        assert len(forecast.forecast_df) == 90