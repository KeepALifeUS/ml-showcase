"""
ML Testing Framework - Test Harness
Main tests harness for execution comprehensive ML tests

Enterprise Pattern: Test Orchestration & Automation
- execution all types ML tests
- Enterprise report CI/CD
- for enterprise ML pipelines
"""

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union

import mlflow
import pandas as pd
from sklearn.base import BaseEstimator

from ..model_testing.model_validator import ModelValidator, ValidationResult
from ..model_testing.performance_tester import PerformanceTester, PerformanceResult
from ..model_testing.regression_tester import RegressionTester, RegressionResult
from ..data_testing.data_quality import DataQualityValidator, QualityResult
from ..data_testing.feature_validator import FeatureValidator, FeatureValidationResult
from ..data_testing.drift_detector import DriftDetector, DriftResult
from ..benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkResult
from ..benchmarking.metrics_calculator import MetricsCalculator


class TestType(Enum):
    """Types tests ML Testing Framework"""
    MODEL_VALIDATION = "model_validation"
    PERFORMANCE_TESTING = "performance_testing"
    REGRESSION_TESTING = "regression_testing"
    DATA_QUALITY = "data_quality"
    FEATURE_VALIDATION = "feature_validation"
    DRIFT_DETECTION = "drift_detection"
    BENCHMARKING = "benchmarking"
    FULL_SUITE = "full_suite"


class TestStatus(Enum):
    """Status execution tests"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfiguration:
    """Configuration for execution tests"""
    test_types: List[TestType] = field(default_factory=lambda: [TestType.FULL_SUITE])
    parallel_execution: bool = True
    max_workers: int = 4
 timeout_seconds: int = 3600 # 1
    fail_fast: bool = False
    generate_report: bool = True
    report_format: str = "html"  # html, json, pdf
    mlflow_tracking: bool = True
    log_level: str = "INFO"
    
    # Enterprise Settings
    governance_mode: bool = True
    compliance_checks: bool = True
    audit_logging: bool = True
    quality_gates: bool = True
    
    # Crypto Trading Specific
    trading_environment: str = "staging"  # staging, production
    risk_tolerance: str = "conservative"  # conservative, moderate, aggressive
    market_conditions: str = "normal"  # bull, bear, normal, volatile


@dataclass
class TestResult:
    """Result execution one test"""
    test_type: TestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and self.duration_seconds is None:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class TestSuiteResult:
    """Result execution suite tests"""
    suite_id: str
    configuration: TestConfiguration
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    test_results: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    passed_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    
    def __post_init__(self):
        if self.end_time and self.total_duration is None:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Counting statistics
        self.passed_count = sum(1 for r in self.test_results if r.status == TestStatus.PASSED)
        self.failed_count = sum(1 for r in self.test_results if r.status == TestStatus.FAILED)
        self.error_count = sum(1 for r in self.test_results if r.status == TestStatus.ERROR)
        self.skipped_count = sum(1 for r in self.test_results if r.status == TestStatus.SKIPPED)


class TestHarness:
    """
 class for ML tests
    
    Enterprise Pattern: Centralized Test Orchestration
 - all ML test
 - execution
 - system monitoring
    """
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        self.config = config or TestConfiguration()
        self.logger = self._setup_logging()
        
        # Initialization test components
        self.model_validator = ModelValidator()
        self.performance_tester = PerformanceTester()
        self.regression_tester = RegressionTester()
        self.data_quality_validator = DataQualityValidator()
        self.feature_validator = FeatureValidator()
        self.drift_detector = DriftDetector()
        self.benchmark_runner = BenchmarkRunner()
        self.metrics_calculator = MetricsCalculator()
        
        # MLflow setup
        if self.config.mlflow_tracking:
            self._setup_mlflow()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup"""
        logger = logging.getLogger("ml_test_harness")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking"""
        try:
            mlflow.set_experiment(f"ml_testing_framework_{self.config.trading_environment}")
            self.logger.info("MLflow tracking")
        except Exception as e:
            self.logger.warning(f"Not succeeded MLflow: {e}")
    
    async def run_test_suite(
        self,
        model: Optional[BaseEstimator] = None,
        data: Optional[pd.DataFrame] = None,
        target: Optional[pd.Series] = None,
        baseline_model: Optional[BaseEstimator] = None,
        **kwargs
    ) -> TestSuiteResult:
        """
        Launch full suite tests
        
        Args:
            model: ML model for testing
            data: Data for testing
 target: variable
 baseline_model: model for comparison
            **kwargs: Additional parameters
        
        Returns:
            TestSuiteResult: Results execution all tests
        """
        suite_id = f"test_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        self.logger.info(f"Launch suite tests {suite_id}")
        
        if self.config.mlflow_tracking:
            mlflow.start_run(run_name=suite_id)
        
        try:
            # Preparation tests
            test_tasks = self._prepare_test_tasks(
                model, data, target, baseline_model, **kwargs
            )
            
            # Execution tests
            if self.config.parallel_execution:
                test_results = await self._run_tests_parallel(test_tasks)
            else:
                test_results = await self._run_tests_sequential(test_tasks)
            
            # Create result
            suite_result = TestSuiteResult(
                suite_id=suite_id,
                configuration=self.config,
                start_time=start_time,
                end_time=datetime.now(),
                test_results=test_results
            )
            
            # Generation report
            if self.config.generate_report:
                await self._generate_report(suite_result)
            
            # MLflow logging
            if self.config.mlflow_tracking:
                self._log_to_mlflow(suite_result)
            
            self.logger.info(f"tests completed: {suite_result.passed_count} ,"
                           f"{suite_result.failed_count}")
            
            return suite_result
            
        except Exception as e:
            self.logger.error(f"Critical error at execution tests: {e}")
            self.logger.error(traceback.format_exc())
            
            error_result = TestResult(
                test_type=TestType.FULL_SUITE,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
            
            return TestSuiteResult(
                suite_id=suite_id,
                configuration=self.config,
                start_time=start_time,
                end_time=datetime.now(),
                test_results=[error_result]
            )
        
        finally:
            if self.config.mlflow_tracking and mlflow.active_run():
                mlflow.end_run()
    
    def _prepare_test_tasks(
        self,
        model: Optional[BaseEstimator],
        data: Optional[pd.DataFrame],
        target: Optional[pd.Series],
        baseline_model: Optional[BaseEstimator],
        **kwargs
    ) -> List[Callable]:
        """Preparation tasks for execution tests"""
        tasks = []
        
        for test_type in self.config.test_types:
            if test_type == TestType.FULL_SUITE:
                # all types tests
                tasks.extend([
                    lambda: self._run_model_validation(model, data, target),
                    lambda: self._run_performance_testing(model, data),
                    lambda: self._run_regression_testing(model, baseline_model, data, target),
                    lambda: self._run_data_quality(data, target),
                    lambda: self._run_feature_validation(data),
                    lambda: self._run_drift_detection(data, kwargs.get('reference_data')),
                    lambda: self._run_benchmarking(model, data)
                ])
            else:
                task_func = self._get_test_task_function(test_type)
                if task_func:
                    tasks.append(lambda t=test_type: task_func(model, data, target, **kwargs))
        
        return tasks
    
    def _get_test_task_function(self, test_type: TestType) -> Optional[Callable]:
        """Retrieval functions for execution test"""
        task_mapping = {
            TestType.MODEL_VALIDATION: self._run_model_validation,
            TestType.PERFORMANCE_TESTING: self._run_performance_testing,
            TestType.REGRESSION_TESTING: self._run_regression_testing,
            TestType.DATA_QUALITY: self._run_data_quality,
            TestType.FEATURE_VALIDATION: self._run_feature_validation,
            TestType.DRIFT_DETECTION: self._run_drift_detection,
            TestType.BENCHMARKING: self._run_benchmarking
        }
        return task_mapping.get(test_type)
    
    async def _run_tests_parallel(self, test_tasks: List[Callable]) -> List[TestResult]:
        """execution tests"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Launch tasks
            future_to_task = {
                executor.submit(self._execute_test_task, task): task 
                for task in test_tasks
            }
            
            # results
            for future in as_completed(future_to_task, timeout=self.config.timeout_seconds):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if self.config.fail_fast and result.status == TestStatus.FAILED:
                        # tasks
                        for remaining_future in future_to_task:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error at execution test: {e}")
                    results.append(TestResult(
                        test_type=TestType.FULL_SUITE,
                        status=TestStatus.ERROR,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(e)
                    ))
        
        return results
    
    async def _run_tests_sequential(self, test_tasks: List[Callable]) -> List[TestResult]:
        """After execution tests"""
        results = []
        
        for task in test_tasks:
            result = self._execute_test_task(task)
            results.append(result)
            
            if self.config.fail_fast and result.status == TestStatus.FAILED:
                break
        
        return results
    
    def _execute_test_task(self, task: Callable) -> TestResult:
        """Execution one test tasks"""
        start_time = datetime.now()
        
        try:
            result_data = task()
            
            return TestResult(
 test_type=TestType.FULL_SUITE, # method
                status=TestStatus.PASSED if result_data else TestStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                result_data=result_data
            )
            
        except Exception as e:
            self.logger.error(f"Error at execution test: {e}")
            return TestResult(
                test_type=TestType.FULL_SUITE,
                status=TestStatus.ERROR,
                start_time=start_time,
                end_time=datetime.now(),
                error_message=str(e)
            )
    
    # methods execution tests
    def _run_model_validation(
        self, 
        model: BaseEstimator, 
        data: pd.DataFrame, 
        target: pd.Series
    ) -> ValidationResult:
        """Execution validation models"""
        return self.model_validator.validate_model(model, data, target)
    
    def _run_performance_testing(
        self, 
        model: BaseEstimator, 
        data: pd.DataFrame
    ) -> PerformanceResult:
        """Execution testing performance"""
        return self.performance_tester.test_model_performance(model, data)
    
    def _run_regression_testing(
        self,
        model: BaseEstimator,
        baseline_model: Optional[BaseEstimator],
        data: pd.DataFrame,
        target: pd.Series
    ) -> RegressionResult:
        """Execution testing"""
        if baseline_model is None:
            return RegressionResult(
                accuracy_changed=False,
                performance_changed=False,
                prediction_drift_score=0.0,
                comparison_metrics={}
            )
        
        return self.regression_tester.compare_models(
            current_model=model,
            baseline_model=baseline_model,
            test_data=data,
            test_target=target
        )
    
    def _run_data_quality(
        self, 
        data: pd.DataFrame, 
        target: Optional[pd.Series]
    ) -> QualityResult:
        """Execution verification quality data"""
        return self.data_quality_validator.validate_data(data, target)
    
    def _run_feature_validation(self, data: pd.DataFrame) -> FeatureValidationResult:
        """Execution validation features"""
        return self.feature_validator.validate_features(data)
    
    def _run_drift_detection(
        self, 
        data: pd.DataFrame, 
        reference_data: Optional[pd.DataFrame]
    ) -> DriftResult:
        """Execution detected drift"""
        if reference_data is None:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                feature_drifts={},
                test_statistics={}
            )
        
        return self.drift_detector.detect_drift(reference_data, data)
    
    def _run_benchmarking(
        self, 
        model: BaseEstimator, 
        data: pd.DataFrame
    ) -> BenchmarkResult:
        """Execution benchmark"""
        return self.benchmark_runner.run_benchmark(model, data)
    
    async def _generate_report(self, suite_result: TestSuiteResult) -> None:
        """Generation report result testing"""
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        report_file = report_dir / f"{suite_result.suite_id}.{self.config.report_format}"
        
        if self.config.report_format == "json":
            import json
            with open(report_file, 'w') as f:
                # Serialization results for JSON
                json.dump(self._serialize_results(suite_result), f, indent=2)
        
        elif self.config.report_format == "html":
            html_content = self._generate_html_report(suite_result)
            with open(report_file, 'w') as f:
                f.write(html_content)
        
        self.logger.info(f"saved: {report_file}")
    
    def _serialize_results(self, suite_result: TestSuiteResult) -> Dict[str, Any]:
        """Serialization results for saving"""
        return {
            "suite_id": suite_result.suite_id,
            "start_time": suite_result.start_time.isoformat(),
            "end_time": suite_result.end_time.isoformat() if suite_result.end_time else None,
            "total_duration": suite_result.total_duration,
            "passed_count": suite_result.passed_count,
            "failed_count": suite_result.failed_count,
            "error_count": suite_result.error_count,
            "skipped_count": suite_result.skipped_count,
            "test_results": [
                {
                    "test_type": result.test_type.value,
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                    "error_message": result.error_message,
                    "warnings": result.warnings,
                    "metrics": result.metrics
                }
                for result in suite_result.test_results
            ]
        }
    
    def _generate_html_report(self, suite_result: TestSuiteResult) -> str:
        """Generation HTML report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Testing Framework Report - {suite_result.suite_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .test-result {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; }}
                .passed {{ border-left: 5px solid #28a745; }}
                .failed {{ border-left: 5px solid #dc3545; }}
                .error {{ border-left: 5px solid #fd7e14; }}
                .metrics {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ML Testing Framework Report</h1>
                <p>Suite ID: {suite_result.suite_id}</p>
                <p>Duration: {suite_result.total_duration:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>✅ Passed: {suite_result.passed_count}</p>
                <p>❌ Failed: {suite_result.failed_count}</p>
                <p>🚨 Errors: {suite_result.error_count}</p>
                <p>⏭️ Skipped: {suite_result.skipped_count}</p>
            </div>
            
            <div class="tests">
                <h2>Test Results</h2>
                {''.join([self._format_test_result_html(result) for result in suite_result.test_results])}
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_test_result_html(self, result: TestResult) -> str:
        """result test for HTML"""
        status_class = result.status.value.lower()
        return f"""
        <div class="test-result {status_class}">
            <h3>{result.test_type.value.replace('_', ' ').title()}</h3>
            <p>Status: {result.status.value}</p>
            <p>Duration: {result.duration_seconds:.3f}s</p>
            {f'<p>Error: {result.error_message}</p>' if result.error_message else ''}
            {f'<div class="metrics">Metrics: {result.metrics}</div>' if result.metrics else ''}
        </div>
        """
    
    def _log_to_mlflow(self, suite_result: TestSuiteResult) -> None:
        """Logging results MLflow"""
        try:
            # Main metrics
            mlflow.log_metric("total_tests", len(suite_result.test_results))
            mlflow.log_metric("passed_tests", suite_result.passed_count)
            mlflow.log_metric("failed_tests", suite_result.failed_count)
            mlflow.log_metric("error_tests", suite_result.error_count)
            mlflow.log_metric("duration_seconds", suite_result.total_duration or 0)
            
            # Parameters configuration
            mlflow.log_param("parallel_execution", self.config.parallel_execution)
            mlflow.log_param("max_workers", self.config.max_workers)
            mlflow.log_param("trading_environment", self.config.trading_environment)
            mlflow.log_param("risk_tolerance", self.config.risk_tolerance)
            
            if self.config.generate_report:
                report_file = f"reports/{suite_result.suite_id}.{self.config.report_format}"
                if Path(report_file).exists():
                    mlflow.log_artifact(report_file)
            
        except Exception as e:
            self.logger.warning(f"Not succeeded MLflow: {e}")


def create_crypto_trading_harness(
    environment: str = "staging",
    risk_tolerance: str = "conservative",
    parallel: bool = True
) -> TestHarness:
    """
    Factory function for creation Test Harness for crypto trading
    
    Args:
 environment: Trading (staging/production)
 risk_tolerance:
 parallel: execution tests
    
    Returns:
 TestHarness: Configured tests harness
    """
    config = TestConfiguration(
        test_types=[TestType.FULL_SUITE],
        parallel_execution=parallel,
        max_workers=4 if parallel else 1,
        trading_environment=environment,
        risk_tolerance=risk_tolerance,
        governance_mode=True,
        compliance_checks=True,
        mlflow_tracking=True,
        generate_report=True
    )
    
    return TestHarness(config)


# Example usage
if __name__ == "__main__":
    async def main():
        # Create harness for crypto trading
        harness = create_crypto_trading_harness(
            environment="staging",
            risk_tolerance="conservative"
        )
        
        # Example models data
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feature_{i}' for i in range(5)])
        y = pd.Series(np.random.choice([0, 1], 100))
        
        model.fit(X, y)
        
        # Launch tests
        results = await harness.run_test_suite(model=model, data=X, target=y)
        
        print(f"Testing completed: {results.passed_count}/{len(results.test_results)} tests")
    
    asyncio.run(main())