"""
Pytest Fixtures - Enterprise Enterprise ML Testing Integration
Comprehensive pytest fixtures for ML testing crypto trading systems

Applies enterprise principles:
- Enterprise test infrastructure
- Production-ready fixtures
- Comprehensive test data generation
- ML model mocking and testing
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
from pathlib import Path
import tempfile
import logging
from unittest.mock import Mock, MagicMock
import warnings

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings test environment
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """
    Session-level test configuration for ML testing
    
    Returns:
        Dict[str, Any]: Test configuration
    """
    return {
        'random_seed': 42,
        'test_data_size': 1000,
        'batch_sizes': [1, 8, 32, 128],
        'num_features': 10,
        'num_classes': 3,
        'crypto_symbols': ['BTC', 'ETH', 'ADA', 'DOT', 'SOL'],
        'test_duration_seconds': 60,
        'confidence_level': 0.95,
        'significance_threshold': 0.05,
        'trading_cost_bps': 10.0,
        'max_latency_ms': 100.0,
        'min_throughput': 100.0,
        'max_memory_mb': 512.0
    }


@pytest.fixture(scope="session")
def temp_directory() -> Generator[Path, None, None]:
    """
 for test artifacts
    
    Yields:
        Path: Temporary directory path
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info(f"Created temporary test directory: {temp_path}")
        yield temp_path


@pytest.fixture
def random_seed(test_config: Dict[str, Any]) -> int:
    """
    Random seed for reproducible testing
    
    Args:
        test_config: Test configuration
    
    Returns:
        int: Random seed
    """
    seed = test_config['random_seed']
    np.random.seed(seed)
    return seed


@pytest.fixture
def sample_classification_data(test_config: Dict[str, Any], random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample classification data for testing
    
    Args:
        test_config: Test configuration
        random_seed: Random seed
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) classification data
    """
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=test_config['test_data_size'],
        n_features=test_config['num_features'],
        n_classes=test_config['num_classes'],
        n_informative=test_config['num_features'] // 2,
        n_redundant=test_config['num_features'] // 4,
        random_state=random_seed
    )
    
    return X, y


@pytest.fixture
def sample_regression_data(test_config: Dict[str, Any], random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample regression data for testing
    
    Args:
        test_config: Test configuration
        random_seed: Random seed
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) regression data
    """
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=test_config['test_data_size'],
        n_features=test_config['num_features'],
        n_informative=test_config['num_features'] // 2,
        noise=0.1,
        random_state=random_seed
    )
    
    return X, y


@pytest.fixture
def crypto_price_data(test_config: Dict[str, Any], random_seed: int) -> pd.DataFrame:
    """
    Synthetic crypto price data for testing
    
    Args:
        test_config: Test configuration
        random_seed: Random seed
    
    Returns:
        pd.DataFrame: Crypto price data
    """
    np.random.seed(random_seed)
    
    # Generate time series data
    timestamps = pd.date_range(
        start=datetime.now() - timedelta(days=100),
        end=datetime.now(),
        freq='1H'
    )
    
    symbols = test_config['crypto_symbols']
    crypto_data = []
    
    for symbol in symbols:
        # Geometric Brownian Motion for price simulation
        n_steps = len(timestamps)
        dt = 1/24/365  # Hourly steps
        mu = 0.0001  # Small drift
        sigma = 0.02  # 2% volatility per hour
        
        # Initial price
        initial_price = np.random.uniform(1000, 50000)
        
        # Generate random walk
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        price_changes = (mu - 0.5 * sigma**2) * dt + sigma * dW
        
        # Cumulative price
        log_prices = np.log(initial_price) + np.cumsum(price_changes)
        prices = np.exp(log_prices)
        
        # Add volume data
        volumes = np.random.lognormal(mean=5, sigma=1, size=n_steps)
        
        for i, (timestamp, price, volume) in enumerate(zip(timestamps, prices, volumes)):
            crypto_data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'open': price * np.random.uniform(0.999, 1.001),
                'high': price * np.random.uniform(1.000, 1.002),
                'low': price * np.random.uniform(0.998, 1.000),
                'close': price,
                'volume': volume
            })
    
    df = pd.DataFrame(crypto_data)
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    return df


@pytest.fixture
def crypto_features_data(crypto_price_data: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """
    Crypto features derived from price data for ML testing
    
    Args:
        crypto_price_data: Crypto price data
        random_seed: Random seed
    
    Returns:
        pd.DataFrame: Feature data
    """
    np.random.seed(random_seed)
    
    features_data = []
    
    for symbol in crypto_price_data['symbol'].unique():
        symbol_data = crypto_price_data[crypto_price_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('timestamp').reset_index(drop=True)
        
        # Technical indicators
        symbol_data['returns'] = symbol_data['close'].pct_change()
        symbol_data['log_returns'] = np.log(symbol_data['close'] / symbol_data['close'].shift(1))
        
        # Moving averages
        symbol_data['sma_10'] = symbol_data['close'].rolling(10).mean()
        symbol_data['sma_30'] = symbol_data['close'].rolling(30).mean()
        symbol_data['ema_10'] = symbol_data['close'].ewm(span=10).mean()
        
        # Volatility
        symbol_data['volatility_10'] = symbol_data['returns'].rolling(10).std()
        symbol_data['volatility_30'] = symbol_data['returns'].rolling(30).std()
        
        # RSI
        delta = symbol_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        symbol_data['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        symbol_data['volume_sma_10'] = symbol_data['volume'].rolling(10).mean()
        symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_sma_10']
        
        # Price momentum
        symbol_data['momentum_5'] = symbol_data['close'] / symbol_data['close'].shift(5) - 1
        symbol_data['momentum_20'] = symbol_data['close'] / symbol_data['close'].shift(20) - 1
        
        # Add noise features for testing feature selection
        for i in range(5):
            symbol_data[f'noise_{i}'] = np.random.normal(0, 1, len(symbol_data))
        
        features_data.append(symbol_data)
    
    combined_data = pd.concat(features_data, ignore_index=True)
    combined_data = combined_data.dropna()  # Remove NaN values from indicators
    
    return combined_data


@pytest.fixture
def mock_ml_model() -> Mock:
    """
    Mock ML model for testing without real model dependencies
    
    Returns:
        Mock: Mocked ML model
    """
    model = Mock()
    
    # Mock predict method
    def mock_predict(X):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Return realistic predictions
        n_samples = X.shape[0]
        predictions = np.random.normal(0, 1, n_samples)
        return predictions
    
    model.predict = Mock(side_effect=mock_predict)
    
    # Mock predict_proba method for classification
    def mock_predict_proba(X):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        n_classes = 3
        
        # Generate random probabilities that sum to 1
        probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        return probs
    
    model.predict_proba = Mock(side_effect=mock_predict_proba)
    
    # Mock fit method
    def mock_fit(X, y):
        return model
    
    model.fit = Mock(side_effect=mock_fit)
    
    # Mock other common methods
    model.score = Mock(return_value=0.85)
    model.get_params = Mock(return_value={'param1': 'value1', 'param2': 'value2'})
    
    # Add model metadata
    model.__class__.__name__ = 'MockMLModel'
    model.__module__ = 'test_module'
    
    return model


@pytest.fixture
def mock_slow_model(test_config: Dict[str, Any]) -> Mock:
    """
    Mock slow ML model for performance testing
    
    Args:
        test_config: Test configuration
    
    Returns:
        Mock: Slow mocked ML model
    """
    import time
    
    model = Mock()
    
    def slow_predict(X):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Simulate slow prediction
        batch_size = X.shape[0]
        latency_per_sample = 0.01  # 10ms per sample
        time.sleep(latency_per_sample * batch_size)
        
        predictions = np.random.normal(0, 1, batch_size)
        return predictions
    
    model.predict = Mock(side_effect=slow_predict)
    model.__class__.__name__ = 'MockSlowModel'
    
    return model


@pytest.fixture
def mock_memory_intensive_model() -> Mock:
    """
    Mock memory-intensive ML model for memory testing
    
    Returns:
        Mock: Memory-intensive mocked ML model
    """
    model = Mock()
    
    # Store large arrays to simulate memory usage
    model._large_arrays = []
    
    def memory_intensive_predict(X):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        batch_size = X.shape[0]
        
        # Allocate memory proportional to batch size
        large_array = np.random.random((batch_size, 10000))  # ~80MB per 1000 samples
        model._large_arrays.append(large_array)
        
        # Keep only recent arrays (memory leak simulation)
        if len(model._large_arrays) > 3:
            model._large_arrays.pop(0)
        
        predictions = np.random.normal(0, 1, batch_size)
        return predictions
    
    model.predict = Mock(side_effect=memory_intensive_predict)
    model.__class__.__name__ = 'MockMemoryIntensiveModel'
    
    return model


@pytest.fixture
def sklearn_classification_models() -> Dict[str, Any]:
    """
    Real scikit-learn classification models for integration testing
    
    Returns:
        Dict[str, Any]: Dictionary of sklearn models
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=200),
            'svm': SVC(random_state=42, probability=True),
            'naive_bayes': GaussianNB()
        }
        
        return models
    
    except ImportError:
        pytest.skip("scikit-learn not available for sklearn models fixture")


@pytest.fixture
def sklearn_regression_models() -> Dict[str, Any]:
    """
    Real scikit-learn regression models for integration testing
    
    Returns:
        Dict[str, Any]: Dictionary of sklearn regression models
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.svm import SVR
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=42),
            'svr': SVR()
        }
        
        return models
    
    except ImportError:
        pytest.skip("scikit-learn not available for sklearn regression models fixture")


@pytest.fixture
def trained_classification_model(sklearn_classification_models, sample_classification_data):
    """
    Pre-trained classification model for testing
    
    Args:
        sklearn_classification_models: Available sklearn models
        sample_classification_data: Sample data for training
    
    Returns:
        Any: Trained classification model
    """
    X, y = sample_classification_data
    model = sklearn_classification_models['random_forest']
    
    # Split data for training
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def trained_regression_model(sklearn_regression_models, sample_regression_data):
    """
    Pre-trained regression model for testing
    
    Args:
        sklearn_regression_models: Available sklearn models
        sample_regression_data: Sample data for training
    
    Returns:
        Any: Trained regression model
    """
    X, y = sample_regression_data
    model = sklearn_regression_models['random_forest']
    
    # Split data for training
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def validation_config() -> Dict[str, Any]:
    """
    Validation configuration for testing
    
    Returns:
        Dict[str, Any]: Validation configuration
    """
    return {
        'check_input_shape': True,
        'check_output_shape': True,
        'check_prediction_range': True,
        'check_determinism': True,
        'check_overfitting': True,
        'min_accuracy': 0.6,
        'max_latency_ms': 100.0,
        'max_memory_usage_mb': 512.0,
        'accuracy_threshold': 0.05,
        'latency_threshold_ms': 10.0,
        'memory_threshold_mb': 50.0,
        'prediction_drift_threshold': 0.1,
        'enable_strict_mode': False,
        'production_ready_check': True,
        'security_validation': True
    }


@pytest.fixture
def benchmark_config(test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Benchmark configuration for testing
    
    Args:
        test_config: Test configuration
    
    Returns:
        Dict[str, Any]: Benchmark configuration
    """
    return {
        'duration_seconds': test_config['test_duration_seconds'],
        'warmup_iterations': 5,
        'measurement_iterations': 20,
        'batch_sizes': test_config['batch_sizes'],
        'max_concurrent_requests': 10,
        'thread_pool_size': 2,
        'max_latency_ms': test_config['max_latency_ms'],
        'min_throughput': test_config['min_throughput'],
        'max_memory_mb': test_config['max_memory_mb'],
        'monitor_resources': True,
 'monitor_gpu': False, # Disable GPU monitoring tests
 'enable_mlflow_tracking': False, # Disable MLflow tests
        'save_detailed_logs': False
    }


@pytest.fixture
def metrics_config(test_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Metrics configuration for testing
    
    Args:
        test_config: Test configuration
    
    Returns:
        Dict[str, Any]: Metrics configuration
    """
    return {
        'bootstrap_samples': 100,  # Reduced for faster tests
        'confidence_level': test_config['confidence_level'],
        'enable_significance_testing': True,
        'significance_threshold': test_config['significance_threshold'],
        'time_series_frequency': "1H",
        'risk_free_rate': 0.02,
        'trading_cost_bps': test_config['trading_cost_bps'],
        'latency_unit': "ms",
        'memory_unit': "MB",
        'enable_detailed_analysis': False,  # Disable for faster tests
        'save_intermediate_results': False,
        'generate_visualizations': False,
        'crypto_return_calculation': "log",
        'enable_volatility_adjustment': True,
        'slippage_bps': 2.0
    }


@pytest.fixture
def data_quality_config() -> Dict[str, Any]:
    """
    Data quality configuration for testing
    
    Returns:
        Dict[str, Any]: Data quality configuration
    """
    return {
        'max_missing_percentage': 5.0,
        'critical_missing_percentage': 20.0,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'max_outliers_percentage': 5.0,
        'check_exact_duplicates': True,
        'check_near_duplicates': False,
        'validate_ranges': True,
        'check_temporal_order': True,
        'check_temporal_gaps': True,
        'max_temporal_gap_hours': 24.0,
        'check_distribution_shift': True,
        'distribution_p_value_threshold': 0.05,
        'check_correlation_stability': True,
        'correlation_change_threshold': 0.3,
        'enable_advanced_checks': False,  # Disable for faster tests
        'save_quality_reports': False,
        'generate_recommendations': True,
        'validate_price_data': True,
        'min_price_value': 0.0001,
        'max_price_value': 1000000.0,
        'validate_volume_data': True,
        'min_volume_value': 0.0
    }


@pytest.fixture
def feature_validation_config() -> Dict[str, Any]:
    """
    Feature validation configuration for testing
    
    Returns:
        Dict[str, Any]: Feature validation configuration
    """
    return {
        'constant_threshold': 0.95,
        'min_unique_values': 2,
        'high_correlation_threshold': 0.95,
        'check_target_correlation': True,
        'min_target_correlation': 0.01,
        'min_variance_threshold': 1e-6,
        'normalize_variance': True,
        'max_missing_percentage': 5.0,
        'critical_missing_threshold': 20.0,
        'outlier_method': 'iqr',
        'outlier_threshold': 1.5,
        'max_outlier_percentage': 5.0,
        'check_normality': False,
        'check_skewness': True,
        'max_skewness': 5.0,
        'max_kurtosis': 10.0,
        'enable_drift_detection': False,  # Disable for faster tests
        'drift_threshold': 0.1,
        'check_information_leakage': True,
        'perfect_correlation_threshold': 0.999,
        'check_feature_scaling': True,
        'scaling_tolerance': 100.0,
        'check_temporal_consistency': True,
        'temporal_gap_tolerance_hours': 24.0,
        'enable_advanced_analysis': False,
        'generate_feature_profiles': False,  # Disable for faster tests
        'save_validation_history': False,
        'validate_trading_features': True
    }


@pytest.fixture
def drift_detection_config() -> Dict[str, Any]:
    """
    Drift detection configuration for testing
    
    Returns:
        Dict[str, Any]: Drift detection configuration
    """
    return {
        'p_value_threshold': 0.05,
        'drift_score_threshold': 0.1,
        'psi_threshold_low': 0.1,
        'psi_threshold_medium': 0.2,
        'psi_threshold_high': 0.25,
        'ks_threshold': 0.05,
        'wasserstein_threshold': 0.1,
        'jensen_shannon_threshold': 0.1,
        'concept_drift_window_size': 50,
        'concept_drift_threshold': 0.05,
        'prediction_drift_threshold': 0.1,
        'performance_degradation_threshold': 0.05,
        'minimum_samples_for_detection': 50,
        'enable_temporal_drift': False,  # Disable for faster tests
        'temporal_window_hours': 24.0,
        'enable_advanced_methods': False,  # Disable for faster tests
        'save_drift_history': False,
        'auto_generate_alerts': False,
        'crypto_volatility_adjustment': True,
        'price_drift_sensitivity': 0.02,
        'volume_drift_sensitivity': 0.05
    }


# Parametrized fixtures for testing multiple scenarios

@pytest.fixture(params=[1, 8, 32, 128])
def batch_size(request) -> int:
    """
    Parametrized batch size fixture for testing different batch sizes
    
    Returns:
        int: Batch size
    """
    return request.param


@pytest.fixture(params=['classification', 'regression'])
def ml_task_type(request) -> str:
    """
    Parametrized ML task type for testing different types
    
    Returns:
        str: ML task type
    """
    return request.param


@pytest.fixture(params=['BTC', 'ETH', 'ADA'])
def crypto_symbol(request) -> str:
    """
    Parametrized crypto symbol for testing different cryptocurrencies
    
    Returns:
        str: Crypto symbol
    """
    return request.param


# Helper fixtures for common test patterns

@pytest.fixture
def assert_approximately_equal():
    """
    Helper fixture for approximate equality assertions
    
    Returns:
        Callable: Function for approximate assertions
    """
    def _assert_approximately_equal(actual, expected, tolerance=1e-6, message=""):
        """
        Assert that two values are approximately equal
        
        Args:
            actual: Actual value
            expected: Expected value
            tolerance: Tolerance for comparison
            message: Optional message
        """
        if isinstance(actual, (list, tuple, np.ndarray)):
            actual = np.array(actual)
            expected = np.array(expected)
            assert np.allclose(actual, expected, rtol=tolerance), \
                f"{message} Arrays not approximately equal: {actual} vs {expected}"
        else:
            assert abs(actual - expected) <= tolerance, \
                f"{message} Values not approximately equal: {actual} vs {expected} (tolerance: {tolerance})"
    
    return _assert_approximately_equal


@pytest.fixture
def assert_performance_metrics():
    """
    Helper fixture for performance assertions
    
    Returns:
        Callable: Function for performance assertions
    """
    def _assert_performance_metrics(metrics, config):
        """
        Assert performance metrics meet requirements
        
        Args:
            metrics: Performance metrics dictionary
            config: Test configuration with thresholds
        """
        if 'mean_latency_ms' in metrics and 'max_latency_ms' in config:
            assert metrics['mean_latency_ms'] <= config['max_latency_ms'], \
                f"Latency too high: {metrics['mean_latency_ms']}ms > {config['max_latency_ms']}ms"
        
        if 'max_throughput' in metrics and 'min_throughput' in config:
            assert metrics['max_throughput'] >= config['min_throughput'], \
                f"Throughput too low: {metrics['max_throughput']} < {config['min_throughput']}"
        
        if 'max_memory_usage_mb' in metrics and 'max_memory_mb' in config:
            assert metrics['max_memory_usage_mb'] <= config['max_memory_mb'], \
                f"Memory usage too high: {metrics['max_memory_usage_mb']}MB > {config['max_memory_mb']}MB"
    
    return _assert_performance_metrics


@pytest.fixture
def create_test_model():
    """
    Factory fixture for creating test models
    
    Returns:
        Callable: Function for creating test models
    """
    def _create_test_model(model_type='mock', **kwargs):
        """
        Create test model of specified type
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments
        
        Returns:
            Any: Test model
        """
        if model_type == 'mock':
            return Mock(**kwargs)
        
        elif model_type == 'sklearn_classification':
            try:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=5, random_state=42, **kwargs)
            except ImportError:
                pytest.skip("scikit-learn not available")
        
        elif model_type == 'sklearn_regression':
            try:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=5, random_state=42, **kwargs)
            except ImportError:
                pytest.skip("scikit-learn not available")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return _create_test_model


# Cleanup fixtures

@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """
    Auto-cleanup fixture for test artifacts
    """
    # Setup - not
    yield
    
    # Cleanup
    import gc
    import os
    import tempfile
    
    # Force garbage collection
    gc.collect()
    
    # Clean up temporary files
    temp_dir = tempfile.gettempdir()
    for file in os.listdir(temp_dir):
        if file.startswith('ml_test_') or file.startswith('pytest_'):
            try:
                os.remove(os.path.join(temp_dir, file))
            except:
                pass  # Ignore cleanup errors


@pytest.fixture(autouse=True)
def suppress_warnings():
    """
 Auto-suppress warnings test environment
    """
    import warnings
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# Context managers for testing

@pytest.fixture
def timing_context():
    """
    Context manager for timing test operations
    
    Returns:
        Callable: Timing context manager
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def _timing_context():
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            logger.info(f"Test operation took {duration:.4f} seconds")
    
    return _timing_context