"""
Basic usage example for Prophet forecasting system.

This example demonstrates the fundamental workflow:
1. Data preparation
2. Model training 
3. Forecasting
4. Visualization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the forecasting components
from src.models.prophet_model import ProphetForecaster
from src.preprocessing.data_processor import CryptoDataProcessor
from src.config.prophet_config import ProphetConfig


def generate_sample_data():
    """Generate sample OHLCV data for demonstration"""
    print("📊 Generating sample OHLCV data...")
    
    # Create date range for 1 year of daily data
    dates = pd.date_range(
        start='2023-01-01', 
        end='2023-12-31', 
        freq='D'
    )
    
    # Generate realistic cryptocurrency price data
    np.random.seed(42)
    n_points = len(dates)
    
    # Base price with random walk
    base_price = 50000
    price_changes = np.random.randn(n_points) * 0.03  # 3% daily volatility
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate OHLCV data
    data = []
    for i, date in enumerate(dates):
        price = prices[i]
        daily_volatility = abs(np.random.randn() * 0.02)
        
        # Open price (previous close + small gap)
        open_price = price + np.random.randn() * price * 0.01
        
        # High and low based on volatility
        high_price = price * (1 + daily_volatility)
        low_price = price * (1 - daily_volatility)
        
        # Close price
        close_price = price + np.random.randn() * price * 0.01
        
        # Ensure OHLC logic
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        # Volume (realistic range for crypto)
        volume = np.random.uniform(100000, 1000000)
        
        data.append({
            'timestamp': date,
            'open': max(open_price, 0.01),
            'high': max(high_price, 0.01),
            'low': max(low_price, 0.01), 
            'close': max(close_price, 0.01),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    print(f"✅ Generated {len(df)} data points from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def basic_forecasting_example():
    """Basic forecasting example with Prophet"""
    print("\n🔮 Basic Prophet Forecasting Example")
    print("=" * 50)
    
    # 1. Generate sample data
    sample_data = generate_sample_data()
    
    # 2. Create and configure Prophet forecaster
    print("\n📈 Creating Prophet forecaster for BTC/1d...")
    forecaster = ProphetForecaster(
        symbol="BTC",
        timeframe="1d"
    )
    
    print(f"Model info: {forecaster.symbol} {forecaster.timeframe}")
    print(f"Model trained: {forecaster.is_trained}")
    
    # 3. Process data for Prophet
    print("\n🔧 Processing data for Prophet...")
    processor = CryptoDataProcessor(
        symbol="BTC",
        timeframe="1d"
    )
    
    processed_data = processor.process_ohlcv_data(
        data=sample_data,
        target_column="close",
        include_features=True
    )
    
    print(f"✅ Data processed: {processed_data.prophet_df.shape}")
    print(f"Features created: {processed_data.features_df.shape}")
    print(f"Processing metadata: {len(processed_data.metadata)} items")
    
    # 4. Train the model
    print("\n🎯 Training Prophet model...")
    training_start = datetime.now()
    
    training_metrics = forecaster.train(
        data=processed_data.prophet_df,
        validate=True  # Enable cross-validation
    )
    
    training_time = (datetime.now() - training_start).total_seconds()
    
    print(f"✅ Model trained in {training_time:.2f} seconds")
    print(f"Training samples: {training_metrics.get('training_samples', 'N/A')}")
    print(f"Training metrics: {list(training_metrics.keys())}")
    
    if 'cv_mae' in training_metrics:
        print(f"Cross-validation MAE: {training_metrics['cv_mae']:.2f}")
        print(f"Cross-validation MAPE: {training_metrics.get('cv_mape', 'N/A'):.2f}%")
    
    # 5. Create forecast
    print("\n🔍 Generating forecast for next 30 days...")
    forecast_start = datetime.now()
    
    forecast_result = forecaster.predict(
        periods=30,  # 30 days ahead
        include_history=False
    )
    
    forecast_time = (datetime.now() - forecast_start).total_seconds()
    
    print(f"✅ Forecast completed in {forecast_time:.2f} seconds")
    print(f"Forecast points: {len(forecast_result.forecast_df)}")
    print(f"Forecast period: {forecast_result.forecast_df['ds'].min()} to {forecast_result.forecast_df['ds'].max()}")
    
    # 6. Display forecast summary
    print("\n📊 Forecast Summary:")
    print("-" * 30)
    
    forecast_df = forecast_result.forecast_df
    first_prediction = forecast_df.iloc[0]
    last_prediction = forecast_df.iloc[-1]
    
    print(f"First prediction: ${first_prediction['yhat']:,.2f} on {first_prediction['ds'].date()}")
    print(f"Last prediction: ${last_prediction['yhat']:,.2f} on {last_prediction['ds'].date()}")
    
    price_change = last_prediction['yhat'] - first_prediction['yhat']
    price_change_pct = (price_change / first_prediction['yhat']) * 100
    
    print(f"Total change: ${price_change:,.2f} ({price_change_pct:+.2f}%)")
    print(f"Trend: {'📈 Upward' if price_change > 0 else '📉 Downward'}")
    
    # Confidence intervals
    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
        avg_interval_width = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
        print(f"Average confidence interval: ±${avg_interval_width/2:,.2f}")
    
    # 7. Model diagnostics
    print("\n🔍 Model Diagnostics:")
    print("-" * 30)
    
    model_info = forecaster.get_model_info()
    print(f"Model version: {model_info.get('model_version', 'N/A')}")
    print(f"Changepoints detected: {len(forecast_result.changepoints)}")
    print(f"Trend components: {len(forecast_result.trend_components)}")
    print(f"Seasonality components: {len(forecast_result.seasonality_components)}")
    
    # 8. Visualization (optional - requires display)
    try:
        print("\n📈 Creating forecast visualization...")
        fig = forecaster.plot_forecast(
            forecast_result,
            show_components=True
        )
        
        # Save plot to file
        fig.write_html("forecast_plot.html")
        print("✅ Forecast plot saved to 'forecast_plot.html'")
        print("Open this file in a web browser to view the interactive chart")
        
    except Exception as e:
        print(f"⚠️ Visualization not available: {e}")
    
    print(f"\n🎉 Forecasting example completed successfully!")
    
    return forecast_result


def configuration_example():
    """Example of custom configuration usage"""
    print("\n⚙️ Configuration Example")
    print("=" * 30)
    
    # Create custom configuration
    config = ProphetConfig()
    
    # Modify model parameters
    config.model.changepoint_prior_scale = 0.1  # More flexible trends
    config.model.seasonality_prior_scale = 15.0  # Stronger seasonality
    config.model.daily_seasonality = True
    config.model.weekly_seasonality = True
    
    # Modify data processing
    config.data.outlier_detection = True
    config.data.outlier_threshold = 2.5
    config.data.missing_data_strategy = "interpolate"
    
    print("Custom configuration created:")
    print(f"- Changepoint prior scale: {config.model.changepoint_prior_scale}")
    print(f"- Seasonality prior scale: {config.model.seasonality_prior_scale}")
    print(f"- Daily seasonality: {config.model.daily_seasonality}")
    print(f"- Outlier detection: {config.data.outlier_detection}")
    print(f"- Outlier threshold: {config.data.outlier_threshold}")
    
    # Create model with custom config
    custom_forecaster = ProphetForecaster(
        symbol="ETH",
        timeframe="4h", 
        config=config
    )
    
    print(f"\n✅ Custom forecaster created: {custom_forecaster.symbol} {custom_forecaster.timeframe}")
    
    return custom_forecaster


def error_handling_example():
    """Example of error handling"""
    print("\n⚠️ Error Handling Example")
    print("=" * 30)
    
    from src.utils.exceptions import (
        ModelNotTrainedException,
        InvalidDataException,
        InsufficientDataException
    )
    
    # 1. Try to predict with untrained model
    print("1. Testing untrained model prediction...")
    try:
        untrained_model = ProphetForecaster("BTC", "1h")
        untrained_model.predict(periods=10)
    except ModelNotTrainedException as e:
        print(f"✅ Caught expected error: {e}")
    
    # 2. Try to train with invalid data
    print("\n2. Testing invalid training data...")
    try:
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        model = ProphetForecaster("BTC", "1h")
        model.train(invalid_data)
    except Exception as e:
        print(f"✅ Caught expected error: {type(e).__name__}: {e}")
    
    # 3. Try to train with insufficient data
    print("\n3. Testing insufficient training data...")
    try:
        insufficient_data = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=5, freq='D'),
            'y': [1, 2, 3, 4, 5]
        })
        model = ProphetForecaster("BTC", "1h")
        model.train(insufficient_data)
    except InsufficientDataException as e:
        print(f"✅ Caught expected error: {e}")
    
    print("\n✅ Error handling examples completed")


if __name__ == "__main__":
    """Main execution"""
    print("🔮 Prophet Forecasting - Basic Usage Examples")
    print("=" * 60)
    print(f"Execution started at: {datetime.now()}")
    
    try:
        # Run basic forecasting example
        forecast = basic_forecasting_example()
        
        # Run configuration example
        configuration_example()
        
        # Run error handling example
        error_handling_example()
        
        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print(f"Execution finished at: {datetime.now()}")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nNext steps:")
    print("- Try the advanced_usage.py example")
    print("- Explore API examples in api_examples.py")
    print("- Check the validation examples")
    print("- Read the full documentation in README.md")