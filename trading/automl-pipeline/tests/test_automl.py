"""
Comprehensive Test Suite for ML AutoML Pipeline
Tests core functionality with enterprise patterns
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path

# Import modules to test
from src.utils.config_manager import AutoMLConfig, PresetConfigs
from src.utils.data_preprocessor import DataPreprocessor
from src.feature_engineering.auto_feature_generator import AutoFeatureGenerator
from src.feature_engineering.feature_selector import AdvancedFeatureSelector
from src.model_selection.model_selector import ModelSelector
from src.optimization.bayesian_optimizer import CryptoMLHyperparameterOptimizer
from src.model_selection.ensemble_builder import EnsembleBuilder
from src.evaluation.model_evaluator import ModelEvaluator, CryptoTradingMetrics
from src.pipeline.automl_pipeline import AutoMLPipeline


class TestDataFixtures:
    """Fixtures test data"""
    
    @staticmethod
    def create_crypto_data(n_samples: int = 1000) -> pd.DataFrame:
        """Create synthetic cryptocurrency data"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
        
        # Base OHLCV data
        base_price = 50000
        returns = np.random.randn(n_samples) * 0.02
        
        data = pd.DataFrame({
            'open': base_price + np.cumsum(returns * base_price * 0.01),
            'high': base_price + np.cumsum(returns * base_price * 0.01) * 1.02,
            'low': base_price + np.cumsum(returns * base_price * 0.01) * 0.98,
            'close': base_price + np.cumsum(returns * base_price * 0.01),
            'volume': np.random.exponential(1000, n_samples),
            'future_return': np.random.randn(n_samples) * 0.01  # Target variable
        }, index=dates)
        
        return data
    
    @staticmethod
    def create_regression_data(n_samples: int = 500, n_features: int = 10) -> tuple:
        """Create data for regression"""
        np.random.seed(42)
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Target variable as linear combination first 3 features + noise
        y = pd.Series(
            X.iloc[:, :3].sum(axis=1) + 0.1 * np.random.randn(n_samples)
        )
        
        return X, y
    
    @staticmethod
    def create_classification_data(n_samples: int = 500, n_features: int = 10) -> tuple:
        """Create data for classification"""
        np.random.seed(42)
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Binary classification
        threshold = X.iloc[:, :3].sum(axis=1).median()
        y = pd.Series((X.iloc[:, :3].sum(axis=1) > threshold).astype(int))
        
        return X, y


@pytest.fixture
def temp_output_dir():
    """Temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def crypto_data():
    """Fixture cryptocurrency data"""
    return TestDataFixtures.create_crypto_data()


@pytest.fixture
def regression_data():
    """Fixture data regression"""
    return TestDataFixtures.create_regression_data()


@pytest.fixture
def classification_data():
    """Fixture data classification"""
    return TestDataFixtures.create_classification_data()


class TestAutoMLConfig:
    """Tests configuration AutoML"""
    
    def test_default_config_creation(self):
        """Test creation configuration by default"""
        config = AutoMLConfig()
        
        assert config.project_name == "crypto_trading_automl"
        assert config.random_state == 42
        assert config.n_jobs == -1
        assert config.verbose is True
    
    def test_preset_configs(self):
        """Test preset configurations"""
        # Fast prototype
        fast_config = PresetConfigs.fast_prototype()
        assert fast_config.hyperparameter_optimization.n_trials == 20
        assert fast_config.model_selection.cv_folds == 3
        
        # Production ready
        prod_config = PresetConfigs.production_ready()
        assert prod_config.hyperparameter_optimization.n_trials == 200
        assert prod_config.model_selection.cv_folds == 10
        
        # Crypto trading
        crypto_config = PresetConfigs.crypto_trading()
        assert crypto_config.model_selection.time_series_split is True
        assert crypto_config.feature_generation.enable_technical_indicators is True
    
    def test_config_serialization(self, temp_output_dir):
        """Test saving and loading configuration"""
        config = AutoMLConfig()
        config_path = Path(temp_output_dir) / "test_config.json"
        
        # Save
        config.save_to_file(config_path)
        assert config_path.exists()
        
        # Load
        loaded_config = AutoMLConfig.load_from_file(config_path)
        assert loaded_config.project_name == config.project_name
        assert loaded_config.random_state == config.random_state


class TestDataPreprocessor:
    """Tests preprocessing data"""
    
    def test_basic_preprocessing(self, regression_data):
        """Test base preprocessing"""
        X, y = regression_data
        
        config = AutoMLConfig()
        preprocessor = DataPreprocessor(config)
        
        # Add missing values for testing
        X_with_missing = X.copy()
        X_with_missing.loc[10:20, 'feature_0'] = np.nan
        
        # Preprocessing
        processed_X = preprocessor.preprocess(X_with_missing)
        processed_y = preprocessor.preprocess_target(y)
        
        assert processed_X.isna().sum().sum() == 0  # No missing values
        assert processed_y.isna().sum() == 0
        assert len(processed_X) == len(X)
    
    def test_preprocessor_fit_transform(self, regression_data):
        """Test fit/transform regime"""
        X, y = regression_data
        
        config = AutoMLConfig()
        preprocessor = DataPreprocessor(config)
        
        # Split data
        X_train, X_test = X.iloc[:400], X.iloc[400:]
        
        # Training on train
        processed_X_train = preprocessor.preprocess(X_train, fit=True)
        
        # Apply to test (without training)
        processed_X_test = preprocessor.preprocess(X_test, fit=False)
        
        assert preprocessor.is_fitted is True
        assert processed_X_train.shape[1] == processed_X_test.shape[1]
    
    def test_outlier_handling(self, regression_data):
        """Test processing outliers"""
        X, y = regression_data
        
        # Add outliers
        X_with_outliers = X.copy()
        X_with_outliers.loc[0:5, 'feature_0'] = 100  # Extreme values
        
        config = AutoMLConfig()
        config.data_preprocessing.outlier_handling = 'clip'
        
        preprocessor = DataPreprocessor(config)
        processed_X = preprocessor.preprocess(X_with_outliers)
        
        # Check that outliers processed
        assert processed_X['feature_0'].max() < 100


class TestFeatureEngineering:
    """Tests generation and selection features"""
    
    def test_feature_generation(self, crypto_data):
        """Test generation features"""
        config = AutoMLConfig()
        generator = AutoFeatureGenerator(config)
        
        # Use only OHLCV columns
        ohlcv_data = crypto_data[['open', 'high', 'low', 'close', 'volume']].iloc[:100]
        
        result = generator.generate_features(ohlcv_data)
        
        assert len(result.feature_names) > 0
        assert result.processing_time > 0
        assert not result.features.empty
    
    def test_feature_selection(self, regression_data):
        """Test selection features"""
        X, y = regression_data
        
        config = AutoMLConfig()
        selector = AdvancedFeatureSelector(config)
        
        result = selector.select_features(X, y, ensemble_selection=False)
        
        assert len(result.selected_features) > 0
        assert len(result.selected_features) <= len(X.columns)
        assert result.selection_time > 0
    
    def test_feature_importance_ranking(self, regression_data):
        """Test ranking features"""
        X, y = regression_data
        
        config = AutoMLConfig()
        generator = AutoFeatureGenerator(config)
        
        # Create simple result for testing
        from src.feature_engineering.auto_feature_generator import FeatureGenerationResult
        
        result = FeatureGenerationResult(
            features=X,
            feature_names=list(X.columns),
            feature_importance={col: np.random.random() for col in X.columns},
            generation_metadata={},
            processing_time=1.0
        )
        
        ranking = generator.get_feature_importance_ranking(result)
        
        assert len(ranking) == len(X.columns)
        assert all(isinstance(item, tuple) for item in ranking)
        assert all(len(item) == 2 for item in ranking)


class TestModelSelection:
    """Tests selection models"""
    
    def test_model_selector_regression(self, regression_data):
        """Test selection models for regression"""
        X, y = regression_data
        
        config = AutoMLConfig()
        selector = ModelSelector(config)
        
        result = selector.select_best_models(
            X, y,
            models=['ridge', 'random_forest'],
            cv_folds=3,
            time_series_split=False,
            top_k=2
        )
        
        assert len(result.best_models) <= 2
        assert result.task_type == 'regression'
        assert len(result.model_scores) > 0
    
    def test_model_selector_classification(self, classification_data):
        """Test selection models for classification"""
        X, y = classification_data
        
        config = AutoMLConfig()
        selector = ModelSelector(config)
        
        result = selector.select_best_models(
            X, y,
            models=['logistic_regression', 'random_forest'],
            cv_folds=3,
            time_series_split=False
        )
        
        assert result.task_type in ['binary_classification', 'multiclass_classification']
        assert len(result.model_scores) > 0


class TestHyperparameterOptimization:
    """Tests optimization hyperparameters"""
    
    def test_bayesian_optimization(self, regression_data):
        """Test Bayesian optimization"""
        X, y = regression_data
        
        config = AutoMLConfig()
        optimizer = CryptoMLHyperparameterOptimizer(config)
        
        # Test with small number iterations
        result = optimizer.optimize_model(
            X, y,
            model_name='ridge',
            optimizer_method='optuna_tpe',
            n_calls=5 # for speed tests
        )
        
        assert result.model_name == 'ridge'
        assert result.best_params is not None
        assert result.optimization_time > 0
    
    def test_multiple_models_optimization(self, regression_data):
        """Test optimization several models"""
        X, y = regression_data
        
        config = AutoMLConfig()
        optimizer = CryptoMLHyperparameterOptimizer(config)
        
        results = optimizer.optimize_multiple_models(
            X, y,
            models=['ridge', 'random_forest'],
            n_calls=3
        )
        
        assert len(results) <= 2
        assert all(result.model_name in ['ridge', 'random_forest'] for result in results.values())


class TestEnsembleMethods:
    """Tests ensemble methods"""
    
    def test_ensemble_builder(self, regression_data):
        """Test building ensembles"""
        X, y = regression_data
        
        # Create simple models
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        # Training models
        for model in models.values():
            model.fit(X, y)
        
        config = AutoMLConfig()
        builder = EnsembleBuilder(config)
        
        result = builder.build_ensemble(
            X, y, models,
            ensemble_methods=['voting'],
            task_type='regression'
        )
        
        assert len(result.ensembles) > 0
        assert 'voting' in result.ensembles
        assert result.build_time > 0


class TestModelEvaluation:
    """Tests evaluation models"""
    
    def test_crypto_trading_metrics(self):
        """Test crypto trading metrics"""
        # Test data returns
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        predictions = np.array([0.02, -0.01, 0.03, 0.01, 0.01])
        
        # Test Sharpe ratio
        sharpe = CryptoTradingMetrics.sharpe_ratio(returns)
        assert isinstance(sharpe, float)
        
        # Test Sortino ratio
        sortino = CryptoTradingMetrics.sortino_ratio(returns)
        assert isinstance(sortino, float)
        
        # Test maximum drawdown
        max_dd = CryptoTradingMetrics.maximum_drawdown(returns)
        assert isinstance(max_dd, float)
        assert max_dd <= 0 # Drawdown must be negative
        
        # Test win rate
        win_rate = CryptoTradingMetrics.win_rate(predictions, returns)
        assert 0 <= win_rate <= 1
    
    def test_model_evaluator(self, regression_data):
        """Test evaluator models"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import Ridge
        
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Training model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        config = AutoMLConfig()
        evaluator = ModelEvaluator(config)
        
        result = evaluator.evaluate_model(
            model, X_train, y_train, X_test, y_test,
            model_name='ridge'
        )
        
        assert result.model_name == 'ridge'
        assert result.test_r2 is not None
        assert result.test_mse >= 0
        assert result.evaluation_time > 0


class TestAutoMLPipeline:
    """Tests full AutoML pipeline"""
    
    def test_pipeline_basic_run(self, crypto_data, temp_output_dir):
        """Test basic launch pipeline"""
        # Use fast configuration
        config = PresetConfigs.fast_prototype()
        config.hyperparameter_optimization.n_trials = 3  # Yet faster
        config.model_selection.cv_folds = 2
        
        pipeline = AutoMLPipeline(config, output_dir=temp_output_dir)
        
        # Use small dataset for speed
        small_data = crypto_data.iloc[:200].copy()
        
        result = pipeline.run(
            data=small_data,
            target_column='future_return',
            test_size=0.3,
            stages=['data_preprocessing', 'model_selection']  # Only main stages
        )
        
        assert result.best_model_name is not None
        assert result.total_time > 0
        assert len(result.stages_completed) > 0
    
    def test_pipeline_with_all_stages(self, regression_data, temp_output_dir):
        """Test pipeline with ( version)"""
        X, y = regression_data
        
        # Combine in dataframe
        data = X.copy()
        data['target'] = y
        
        # Minimal configuration for speed
        config = PresetConfigs.fast_prototype()
        config.hyperparameter_optimization.n_trials = 2
        config.model_selection.sklearn_models = ['ridge'] # Only model
        config.feature_generation.enable_tsfresh_features = False  # Disable slow features
        
        pipeline = AutoMLPipeline(config, output_dir=temp_output_dir)
        
        result = pipeline.run(
            data=data,
            target_column='target',
            test_size=0.3
        )
        
        assert result.best_model is not None
        assert result.total_time > 0
    
    def test_pipeline_error_handling(self, temp_output_dir):
        """Test processing errors in """
        config = AutoMLConfig()
        pipeline = AutoMLPipeline(config, output_dir=temp_output_dir)
        
        # Invalid data
        bad_data = pd.DataFrame({'a': [1, 2, 3]})
        
        # Should error without
        with pytest.raises(Exception):
            pipeline.run(bad_data, target_column='nonexistent_column')


class TestIntegrationScenarios:
    """Integration tests various scenarios"""
    
    def test_crypto_trading_scenario(self, crypto_data, temp_output_dir):
        """Test full scenario crypto trading"""
        # Cryptocurrency configuration
        config = PresetConfigs.crypto_trading()
        config.hyperparameter_optimization.n_trials = 5  # Fast tests
        config.model_selection.cv_folds = 3
        config.feature_generation.enable_tsfresh_features = False  # Disable for speed
        
        pipeline = AutoMLPipeline(config, output_dir=temp_output_dir)
        
        # Limit data for speed
        test_data = crypto_data.iloc[:300].copy()
        
        result = pipeline.run(
            data=test_data,
            target_column='future_return',
            time_series_split=True,
            stages=['data_preprocessing', 'feature_generation', 'model_selection']
        )
        
        assert result.best_model is not None
        assert 'crypto_metrics' in result.evaluation_result.evaluation_metadata
    
    def test_high_frequency_trading_scenario(self, crypto_data, temp_output_dir):
        """Test scenario high-frequency trading"""
        config = PresetConfigs.high_frequency_trading()
        config.hyperparameter_optimization.n_trials = 3
        
        pipeline = AutoMLPipeline(config, output_dir=temp_output_dir)
        
        # HFT requires fast processing
        hft_data = crypto_data.iloc[:150].copy()
        
        result = pipeline.run(
            data=hft_data,
            target_column='future_return',
            stages=['data_preprocessing', 'model_selection']  # Minimum stages for speed
        )
        
        assert result.total_time < 60  # HFT should be fast
        assert result.best_model is not None


# Additional utility tests
class TestUtilities:
    """Tests utility functions"""
    
    def test_config_validation(self):
        """Test validation configuration"""
        config = AutoMLConfig()
        
        # Check validators
        assert config.max_memory_gb > 0
        assert config.n_jobs != 0
    
    def test_data_fixtures_quality(self):
        """Test quality test data"""
        # Cryptocurrency data
        crypto_data = TestDataFixtures.create_crypto_data(100)
        assert len(crypto_data) == 100
        assert 'open' in crypto_data.columns
        assert 'future_return' in crypto_data.columns
        
        # Data regression
        X, y = TestDataFixtures.create_regression_data(50, 5)
        assert X.shape == (50, 5)
        assert len(y) == 50
        
        # Data classification
        X, y = TestDataFixtures.create_classification_data(50, 5)
        assert X.shape == (50, 5)
        assert set(y.unique()).issubset({0, 1})


if __name__ == "__main__":
    # Launch tests
    pytest.main([__file__, "-v", "--tb=short"])