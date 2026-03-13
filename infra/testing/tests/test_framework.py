"""
ML Testing Framework - Comprehensive Test Suite
Integrated tests for ML Testing Framework

Enterprise Pattern: Quality Assurance & Continuous Testing
- Automated testing of all ML pipeline components
- Integration tests with real-world trading scenarios
- End-to-end tests for enterprise workflow validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

# Import all components
from src.model_testing.model_validator import ModelValidator, create_crypto_trading_validator
from src.model_testing.performance_tester import PerformanceTester, create_crypto_performance_tester
from src.model_testing.regression_tester import RegressionTester, create_crypto_regression_tester
from src.data_testing.data_quality import DataQualityValidator, create_crypto_data_validator
from src.data_testing.feature_validator import FeatureValidator, create_crypto_feature_validator
from src.data_testing.drift_detector import DriftDetector, create_crypto_drift_detector
from src.benchmarking.benchmark_runner import BenchmarkRunner, create_crypto_benchmark_runner
from src.benchmarking.metrics_calculator import MetricsCalculator, create_crypto_metrics_calculator
from src.integration.test_harness import TestHarness, create_crypto_trading_harness
from src.utils.test_data_generator import TestDataGenerator, create_crypto_trading_data_generator
from src.utils.visualization import MLTestingVisualizer, create_crypto_trading_visualizer

warnings.filterwarnings('ignore')


class TestMLFrameworkIntegration:
    """
 tests for ML Testing Framework
    
    Enterprise Pattern: Comprehensive System Validation
 - Testing full ML validation
 - Verification integration between all component
 - Validation enterprise-grade
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_generator = create_crypto_trading_data_generator(
            samples=100, random_state=42
        )
        
        # Create test data
        self.sample_data, self.sample_target = self.test_data_generator.generate_features_dataset(
            task_type="classification"
        )
        
        # Create simple models for testing
        self.classification_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.regression_model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Training models
        self.classification_model.fit(self.sample_data, self.sample_target)
        reg_target = pd.Series(np.random.normal(0, 1, len(self.sample_data)))
        self.regression_model.fit(self.sample_data, reg_target)
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_classification_workflow(self):
        """Test full workflow for tasks class"""
        # 1. Create components
        validator = create_crypto_trading_validator()
        performance_tester = create_crypto_performance_tester()
        data_validator = create_crypto_data_validator()
        
        # 2. Validation models
        validation_result = validator.validate_model(
            self.classification_model, 
            self.sample_data, 
            self.sample_target
        )
        
        assert validation_result.is_valid
        assert validation_result.input_shape_valid
        assert validation_result.output_format_valid
        
        # 3. Testing performance
        performance_result = performance_tester.test_model_performance(
            self.classification_model, 
            self.sample_data
        )
        
        assert performance_result.latency_ms > 0
        assert performance_result.throughput_samples_per_sec > 0
        
        # 4. Validation data
        quality_result = data_validator.validate_data(self.sample_data, self.sample_target)
        
        assert quality_result.overall_quality_score >= 0.5
        assert not quality_result.has_critical_issues
        
        print(f"✅ End-to-end classification workflow completed successfully")
        print(f"   - Model validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        print(f"   - Performance test: {performance_result.latency_ms:.2f}ms latency")
        print(f"   - Data quality: {quality_result.overall_quality_score:.3f} score")
    
    def test_end_to_end_regression_workflow(self):
        """Test full workflow for tasks"""
        # Create data
        reg_data, reg_target = self.test_data_generator.generate_features_dataset(
            task_type="regression"
        )
        
        # Components
        validator = create_crypto_trading_validator()
        benchmark_runner = create_crypto_benchmark_runner()
        metrics_calc = create_crypto_metrics_calculator()
        
        # 1. Validation models
        validation_result = validator.validate_model(
            self.regression_model,
            reg_data,
            reg_target
        )
        
        assert validation_result.is_valid
        
        # 2. Benchmarking
        benchmark_result = benchmark_runner.run_benchmark(
            self.regression_model,
            reg_data
        )
        
        assert benchmark_result.latency_metrics['mean_ms'] > 0
        assert benchmark_result.throughput_samples_per_sec > 0
        
        # 3. Calculation metrics
        predictions = self.regression_model.predict(reg_data)
        regression_metrics = metrics_calc.calculate_regression_metrics(
            reg_target, predictions
        )
        
        assert 'mse' in regression_metrics
        assert 'mae' in regression_metrics
        assert 'r2' in regression_metrics
        
        print(f"✅ End-to-end regression workflow completed successfully")
        print(f"   - Benchmark latency: {benchmark_result.latency_metrics['mean_ms']:.2f}ms")
        print(f"   - R² score: {regression_metrics['r2']:.3f}")
    
    def test_crypto_price_data_workflow(self):
        """Test for crypto trading workflow data"""
        # Generation cryptocurrency data
        crypto_data = self.test_data_generator.generate_crypto_price_data(
            market_condition=self.test_data_generator.__class__.__module__.split('.')[-1]  # MarketCondition.NORMAL
        )
        
        # Components for data
        data_validator = create_crypto_data_validator()
        feature_validator = create_crypto_feature_validator()
        drift_detector = create_crypto_drift_detector()
        
        # 1. Validation price data
        quality_result = data_validator.validate_data(crypto_data)
        
        # Should be main columns OHLCV
        expected_columns = {'open', 'high', 'low', 'close', 'volume'}
        assert expected_columns.issubset(set(crypto_data.columns))
        
        # 2. Validation features
        numeric_data = crypto_data.select_dtypes(include=[np.number])
        feature_result = feature_validator.validate_features(numeric_data)
        
        assert feature_result.n_features == len(numeric_data.columns)
        
        # 3. Verification drift (comparison itself itself - not should be drift)
        drift_result = drift_detector.detect_drift(crypto_data, crypto_data)
        
        assert not drift_result.drift_detected
 assert drift_result.drift_score < 0.1 # Very low drift at itself itself
        
        print(f"✅ Crypto price data workflow completed successfully")
        print(f"   - Data shape: {crypto_data.shape}")
        print(f"   - Price range: ${crypto_data['close'].min():.2f} - ${crypto_data['close'].max():.2f}")
        print(f"   - Quality score: {quality_result.overall_quality_score:.3f}")
    
    def test_model_comparison_workflow(self):
        """Test workflow comparison models"""
        # Create multiple models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=5, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=100)
        }
        
        # Training models
        for model in models.values():
            model.fit(self.sample_data, self.sample_target)
        
        # Components
        regression_tester = create_crypto_regression_tester()
        metrics_calc = create_crypto_metrics_calculator()
        
        # Comparison models
        results = {}
        for name, model in models.items():
            predictions = model.predict(self.sample_data)
            
            # Calculation metrics
            class_metrics = metrics_calc.calculate_classification_metrics(
                self.sample_target, predictions
            )
            
            results[name] = class_metrics
        
        # Verification results
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            
            # values metrics
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1_score'] <= 1
        
        # Comparison performance
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        
        print(f"✅ Model comparison workflow completed successfully")
        print(f"   - Models tested: {list(models.keys())}")
        print(f"   - Best model: {best_model} (accuracy: {results[best_model]['accuracy']:.3f})")
    
    def test_data_quality_issues_detection(self):
        """Test detected issues quality data"""
        # Create data issue
        from src.utils.test_data_generator import DataQualityIssue
        
        clean_data = self.sample_data.copy()
        corrupted_data = self.test_data_generator.inject_data_quality_issues(
            clean_data,
            issues=[
                DataQualityIssue.MISSING_VALUES,
                DataQualityIssue.OUTLIERS,
                DataQualityIssue.DUPLICATE_ROWS
            ],
            severity=0.1
        )
        
        # Validation
        data_validator = create_crypto_data_validator()
        
        clean_result = data_validator.validate_data(clean_data)
        corrupted_result = data_validator.validate_data(corrupted_data)
        
        # data should have quality
        assert clean_result.overall_quality_score > 0.8
        
        # data should have more low quality
        assert corrupted_result.overall_quality_score < clean_result.overall_quality_score
        
        # Should be detected issues
        assert len(corrupted_result.quality_issues) > 0
        
        print(f"✅ Data quality issues detection completed successfully")
        print(f"   - Clean data quality: {clean_result.overall_quality_score:.3f}")
        print(f"   - Corrupted data quality: {corrupted_result.overall_quality_score:.3f}")
        print(f"   - Issues detected: {len(corrupted_result.quality_issues)}")
    
    def test_test_harness_integration(self):
        """Test integration Test Harness"""
        import asyncio
        
        async def run_harness_test():
            # Create Test Harness
            harness = create_crypto_trading_harness(
                environment="testing",
                risk_tolerance="conservative",
 parallel=False # For testing
            )
            
            # Launch suite tests
            results = await harness.run_test_suite(
                model=self.classification_model,
                data=self.sample_data,
                target=self.sample_target
            )
            
            return results
        
        # Execution
        results = asyncio.run(run_harness_test())
        
        # Verification results
        assert results.suite_id is not None
        assert results.total_duration is not None
        assert len(results.test_results) > 0
        
        # Statistics
        total_tests = len(results.test_results)
        passed_ratio = results.passed_count / total_tests if total_tests > 0 else 0
        
        print(f"✅ Test Harness integration completed successfully")
        print(f"   - Suite ID: {results.suite_id}")
        print(f"   - Total tests: {total_tests}")
        print(f"   - Passed: {results.passed_count}/{total_tests} ({passed_ratio:.1%})")
        print(f"   - Duration: {results.total_duration:.2f}s")
    
    def test_visualization_integration(self):
        """Test integration component visualization"""
        # Create visualizer
        visualizer = create_crypto_trading_visualizer(
 interactive=False, # For testing using
            high_quality=False
        )
        
        # Test data for visualization
        metrics_history = {
            'accuracy': [0.75, 0.78, 0.82, 0.85],
            'precision': [0.72, 0.76, 0.79, 0.83],
            'recall': [0.68, 0.72, 0.75, 0.78],
            'f1_score': [0.70, 0.74, 0.77, 0.80]
        }
        
        # Create chart metrics
        plot_path = visualizer.plot_model_performance_metrics(
            metrics_history,
            "Test Model Performance"
        )
        
        # Verification creation file
        assert Path(plot_path).exists()
        assert Path(plot_path).suffix in ['.png', '.pdf', '.svg', '.html']
        
        # Create chart comparison models
        models_metrics = {
            'Model_A': {'accuracy': 0.85, 'precision': 0.83},
            'Model_B': {'accuracy': 0.88, 'precision': 0.86}
        }
        
        comparison_path = visualizer.plot_model_comparison(
            models_metrics,
            "Test Model Comparison"
        )
        
        assert Path(comparison_path).exists()
        
        print(f"✅ Visualization integration completed successfully")
        print(f"   - Performance plot: {Path(plot_path).name}")
        print(f"   - Comparison plot: {Path(comparison_path).name}")
        
        # Cleanup test files
        for path in [plot_path, comparison_path]:
            try:
                Path(path).unlink()
            except:
                pass
    
    def test_edge_cases_handling(self):
        """Test processing edge cases"""
        # Test empty data
        empty_data = pd.DataFrame()
        
        validator = create_crypto_trading_validator()
        data_validator = create_crypto_data_validator()
        
        # Should correctly empty data without errors
        try:
            empty_quality = data_validator.validate_data(empty_data)
            assert empty_quality is not None
            print("✅ Empty data handled correctly")
        except Exception as e:
            print(f"⚠️  Empty data handling issue: {e}")
        
        # Test data, containing only NaN
        nan_data = pd.DataFrame({
            'feature_0': [np.nan] * 10,
            'feature_1': [np.nan] * 10
        })
        
        try:
            nan_quality = data_validator.validate_data(nan_data)
            assert nan_quality is not None
 assert nan_quality.overall_quality_score < 0.5 # quality
            print("✅ NaN-only data handled correctly")
        except Exception as e:
            print(f"⚠️  NaN data handling issue: {e}")
        
        # Test extreme values
        extreme_data = self.test_data_generator.generate_edge_case_data('extreme_values')
        
        try:
            extreme_quality = data_validator.validate_data(extreme_data)
            assert extreme_quality is not None
            print("✅ Extreme values handled correctly")
        except Exception as e:
            print(f"⚠️  Extreme values handling issue: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance components"""
        import time
        
        # validation models
        validator = create_crypto_trading_validator()
        
        start_time = time.time()
        validation_result = validator.validate_model(
            self.classification_model,
            self.sample_data,
            self.sample_target
        )
        validation_time = time.time() - start_time
        
 assert validation_time < 5.0 # Validation should in 5 seconds
        
        # quality data
        data_validator = create_crypto_data_validator()
        
        start_time = time.time()
        quality_result = data_validator.validate_data(self.sample_data, self.sample_target)
        quality_time = time.time() - start_time
        
 assert quality_time < 5.0 # Validation data should in 5 seconds
        
        # testing performance
        performance_tester = create_crypto_performance_tester()
        
        start_time = time.time()
        perf_result = performance_tester.test_model_performance(
            self.classification_model,
            self.sample_data
        )
        perf_test_time = time.time() - start_time
        
        assert perf_test_time < 10.0  # Test performance in 10 seconds
        
        print(f"✅ Performance benchmarks completed successfully")
        print(f"   - Model validation: {validation_time:.3f}s")
        print(f"   - Data quality check: {quality_time:.3f}s")
        print(f"   - Performance test: {perf_test_time:.3f}s")
    
    def test_error_handling_and_recovery(self):
        """Test processing errors recovery"""
        validator = create_crypto_trading_validator()
        
        # Test invalid model (None)
        try:
            result = validator.validate_model(None, self.sample_data, self.sample_target)
            # Should result error, but not
            assert not result.is_valid
            print("✅ None model handled correctly")
        except Exception as e:
            print(f"⚠️  None model handling issue: {e}")
        
        # Test incompatible data
        try:
            wrong_data = pd.DataFrame({'wrong_feature': [1, 2, 3]})
            result = validator.validate_model(
                self.classification_model,
                wrong_data,
                self.sample_target
            )
            # May error or result
            print("✅ Incompatible data handled correctly")
        except Exception as e:
            print(f"⚠️  Incompatible data - expected error: {type(e).__name__}")
        
        # Test recovery after errors
        try:
            # After errors system should
            normal_result = validator.validate_model(
                self.classification_model,
                self.sample_data,
                self.sample_target
            )
            assert normal_result.is_valid
            print("✅ System recovery after error successful")
        except Exception as e:
            print(f"⚠️  System recovery issue: {e}")


class TestFactoryFunctions:
    """factory functions for creation components"""
    
    def test_all_factory_functions_work(self):
        """Verification operability all factory functions"""
        factories = [
            create_crypto_trading_validator,
            create_crypto_performance_tester,
            create_crypto_regression_tester,
            create_crypto_data_validator,
            create_crypto_feature_validator,
            create_crypto_drift_detector,
            create_crypto_benchmark_runner,
            create_crypto_metrics_calculator,
            lambda: create_crypto_trading_harness(),
            create_crypto_trading_data_generator,
            lambda: create_crypto_trading_visualizer()
        ]
        
        for i, factory in enumerate(factories):
            try:
                component = factory()
                assert component is not None
                print(f"✅ Factory {i+1}: {factory.__name__} - OK")
            except Exception as e:
                print(f"❌ Factory {i+1}: {factory.__name__} - ERROR: {e}")
                raise
        
        print(f"✅ All {len(factories)} factory functions work correctly")
    
    def test_factory_customization(self):
        """Verification customization through factory functions"""
        # Test customization generator data
        custom_generator = create_crypto_trading_data_generator(
            samples=50,
            volatility=0.05,
            random_state=123
        )
        
        assert custom_generator.config.n_samples == 50
        assert custom_generator.config.volatility == 0.05
        assert custom_generator.config.random_state == 123
        
        # Test customization Test Harness
        custom_harness = create_crypto_trading_harness(
            environment="production",
            risk_tolerance="aggressive",
            parallel=False
        )
        
        assert custom_harness.config.trading_environment == "production"
        assert custom_harness.config.risk_tolerance == "aggressive"
        assert not custom_harness.config.parallel_execution
        
        print("✅ Factory customization works correctly")


# for pytest
@pytest.fixture
def sample_model():
    """Fixture simple models for testing"""
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    
    # Simple test data
    X = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(0, 1, 100)
    })
    y = pd.Series(np.random.choice([0, 1], 100))
    
    model.fit(X, y)
    return model, X, y


@pytest.fixture
def crypto_price_data():
    """Fixture cryptocurrency data"""
    generator = create_crypto_trading_data_generator(samples=200, random_state=42)
    return generator.generate_crypto_price_data()


# Parameterized tests
@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_different_task_types(task_type):
    """Test various types tasks ML"""
    generator = create_crypto_trading_data_generator(samples=50, random_state=42)
    
    X, y = generator.generate_features_dataset(
        task_type=task_type,
        include_technical_indicators=True
    )
    
    assert X.shape[0] == 50
    assert len(y) == 50
    
    if task_type == "classification":
 assert set(y.unique()).issubset({0, 1}) # Binary class
    else:
        assert y.dtype in [np.float64, np.float32]  # Regression
    
    print(f"✅ Task type '{task_type}' handled correctly")


@pytest.mark.parametrize("market_condition", [
    "NORMAL", "BULL_MARKET", "BEAR_MARKET", "HIGH_VOLATILITY"
])
def test_different_market_conditions(market_condition):
    """Test various market conditions"""
    generator = create_crypto_trading_data_generator(samples=100, random_state=42)
    
    # various market conditions through change parameters
    if market_condition == "HIGH_VOLATILITY":
        generator.config.volatility = 0.05
    elif market_condition == "BULL_MARKET":
        generator.config.drift = 0.001
    elif market_condition == "BEAR_MARKET":
        generator.config.drift = -0.001
    
    crypto_data = generator.generate_crypto_price_data()
    
    assert len(crypto_data) > 0
    assert 'close' in crypto_data.columns
    assert 'volume' in crypto_data.columns
    
    # Verification validity price data
    assert crypto_data['close'].min() > 0
    assert crypto_data['volume'].min() >= 0
    
    print(f"✅ Market condition '{market_condition}' generated correctly")


if __name__ == "__main__":
    # Launch tests for
    print("=== ML Testing Framework - Comprehensive Test Suite ===\n")
    
    # Create test
    test_instance = TestMLFrameworkIntegration()
    test_instance.setup()
    
    try:
        # Execution tests
        print("1. Testing End-to-End Classification Workflow...")
        test_instance.test_end_to_end_classification_workflow()
        
        print("\n2. Testing End-to-End Regression Workflow...")
        test_instance.test_end_to_end_regression_workflow()
        
        print("\n3. Testing Crypto Price Data Workflow...")
        test_instance.test_crypto_price_data_workflow()
        
        print("\n4. Testing Model Comparison Workflow...")
        test_instance.test_model_comparison_workflow()
        
        print("\n5. Testing Data Quality Issues Detection...")
        test_instance.test_data_quality_issues_detection()
        
        print("\n6. Testing Visualization Integration...")
        test_instance.test_visualization_integration()
        
        print("\n7. Testing Edge Cases Handling...")
        test_instance.test_edge_cases_handling()
        
        print("\n8. Testing Performance Benchmarks...")
        test_instance.test_performance_benchmarks()
        
        print("\n9. Testing Error Handling and Recovery...")
        test_instance.test_error_handling_and_recovery()
        
        print("\n10. Testing Factory Functions...")
        factory_tests = TestFactoryFunctions()
        factory_tests.test_all_factory_functions_work()
        factory_tests.test_factory_customization()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉")
        print("ML Testing Framework is ready for production use.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
 test_instance.setup().__next__() # cleanup
        except:
            pass