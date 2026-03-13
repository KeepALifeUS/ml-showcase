"""
Data Quality Framework - Enterprise Enterprise Data Validation
Comprehensive data quality validation for ML datasets crypto trading

Applies enterprise principles:
- Enterprise data governance
- Production data monitoring
- Statistical data validation
- Automated quality assurance
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import logging
import json

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityIssue(Enum):
    """Types issues quality data for Enterprise governance"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    INCONSISTENT_FORMAT = "inconsistent_format"
    INVALID_RANGE = "invalid_range"
    DATA_TYPE_MISMATCH = "data_type_mismatch"
    SCHEMA_VIOLATION = "schema_violation"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    CORRELATION_ANOMALY = "correlation_anomaly"
    DISTRIBUTION_SHIFT = "distribution_shift"


class QualityCheckSeverity(Enum):
    """Levels severity for Enterprise alerting"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityCheckResult:
    """Result verification quality data - Enterprise structured result"""
    check_name: str
    issue_type: DataQualityIssue
    severity: QualityCheckSeverity
    passed: bool
    message: str
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.check_name}: {self.message}"


@dataclass
class DataQualityConfig:
    """Configuration data quality validation - Enterprise typed configuration"""
    # Missing values thresholds
    max_missing_percentage: float = 5.0  # 5% maximum missing values
    critical_missing_percentage: float = 20.0  # 20% critical level
    
    # Outlier detection settings
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 1.5  # IQR multiplier or Z-score threshold
    max_outliers_percentage: float = 5.0  # 5% maximum outliers
    
    # Duplicate detection
    check_exact_duplicates: bool = True
    check_near_duplicates: bool = False
    near_duplicate_threshold: float = 0.95  # Similarity threshold
    
    # Data range validation
    validate_ranges: bool = True
    custom_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Temporal consistency (for time series)
    check_temporal_order: bool = True
    check_temporal_gaps: bool = True
    max_temporal_gap_hours: float = 24.0
    
    # Statistical validation
    check_distribution_shift: bool = True
    distribution_test_method: str = "ks_test"  # "ks_test", "chi2_test"
    distribution_p_value_threshold: float = 0.05
    
    # Correlation monitoring
    check_correlation_stability: bool = True
    correlation_change_threshold: float = 0.3  # 30% change threshold
    
    # Enterprise enterprise settings
    enable_advanced_checks: bool = True
    save_quality_reports: bool = True
    generate_recommendations: bool = True
    
    # Crypto trading specific
    validate_price_data: bool = True
 min_price_value: float = 0.0001 # Minimum price
 max_price_value: float = 1000000.0 # price
    validate_volume_data: bool = True
    min_volume_value: float = 0.0
    
    # Feature engineering validation
    check_feature_stability: bool = True
    feature_drift_threshold: float = 0.1  # 10% change threshold


class DataQualityValidator:
    """
    Enterprise Enterprise Data Quality Validator
    
 Comprehensive data quality framework for ML datasets crypto trading systems.
 Provides enterprise-grade data governance automated quality assurance.
    """
    
    def __init__(self, config: Optional[DataQualityConfig] = None):
        """
 Initialization data quality validator Enterprise configuration
        
        Args:
            config: Configuration data quality validation (Enterprise typed)
        """
        self.config = config or DataQualityConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.results: List[QualityCheckResult] = []
        
        # Enterprise historical data for comparison
        self._baseline_stats: Dict[str, Dict[str, Any]] = {}
        self._validation_history: List[Dict[str, Any]] = []
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> List[QualityCheckResult]:
        """
 validation dataset - Enterprise enterprise validation pipeline
        
        Args:
            df: DataFrame for validation
            dataset_name: Name dataset for reports
            target_column: Name target column (optional)
            feature_columns: List feature columns (optional)
        
        Returns:
            List[QualityCheckResult]: Results all checks quality
        """
        self.logger.info(f"Starting Enterprise data quality validation for {dataset_name}")
        self.results.clear()
        
        validation_start_time = datetime.now()
        
        try:
            # Basic dataset info
            self._log_dataset_info(df, dataset_name)
            
            # 1. Basic Data Quality Checks
            self._check_missing_values(df)
            self._check_duplicates(df)
            self._check_data_types(df)
            
            # 2. Statistical Data Quality Checks
            self._check_outliers(df, feature_columns)
            self._check_data_ranges(df)
            
            # 3. Temporal Consistency (if is datetime columns)
            datetime_columns = df.select_dtypes(include=['datetime64', 'datetimetz']).columns.tolist()
            if datetime_columns and self.config.check_temporal_order:
                self._check_temporal_consistency(df, datetime_columns)
            
            # 4. Feature-specific checks
            if feature_columns:
                self._check_feature_quality(df, feature_columns)
                
                if self.config.check_feature_stability:
                    self._check_feature_stability(df, feature_columns, dataset_name)
            
            # 5. Target variable checks (if )
            if target_column and target_column in df.columns:
                self._check_target_variable_quality(df, target_column)
            
            # 6. Correlation analysis
            if self.config.check_correlation_stability and len(df.select_dtypes(include=[np.number]).columns) > 1:
                self._check_correlation_stability(df, dataset_name)
            
            # 7. Distribution shift detection (if is baseline)
            if self.config.check_distribution_shift and dataset_name in self._baseline_stats:
                self._check_distribution_shift(df, dataset_name)
            
            # 8. Crypto trading specific checks
            self._check_crypto_specific_quality(df)
            
            # 9. Advanced enterprise checks
            if self.config.enable_advanced_checks:
                self._check_advanced_quality_issues(df)
            
            # Update baseline statistics
            self._update_baseline_stats(df, dataset_name)
            
            validation_time = (datetime.now() - validation_start_time).total_seconds()
            self.logger.info(f"Data quality validation completed in {validation_time:.2f} seconds")
            
            # Generation one report
            self._generate_summary_result(df, dataset_name)
            
        except Exception as e:
            self.logger.error(f"Error at data quality validation: {e}")
            self.results.append(QualityCheckResult(
                check_name="validation_error",
                issue_type=DataQualityIssue.SCHEMA_VIOLATION,
                severity=QualityCheckSeverity.CRITICAL,
                passed=False,
                message=f"Critical error validation: {str(e)}",
                details={'error': str(e)}
            ))
        
        return self.results.copy()
    
    def _log_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Logging base information dataset"""
        self.logger.info(f"Dataset {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    def _check_missing_values(self, df: pd.DataFrame) -> None:
        """Verification missing values - Enterprise data completeness validation"""
        total_cells = df.shape[0] * df.shape[1]
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            # Detection severity level
            if missing_percentage == 0:
 continue # missing values
            elif missing_percentage <= self.config.max_missing_percentage:
                severity = QualityCheckSeverity.INFO
                passed = True
                message = f"level missing values {column}: {missing_percentage:.1f}%"
            elif missing_percentage <= self.config.critical_missing_percentage:
                severity = QualityCheckSeverity.WARNING
                passed = False
                message = f"High level missing values {column}: {missing_percentage:.1f}%"
            else:
                severity = QualityCheckSeverity.CRITICAL
                passed = False
                message = f"level missing values {column}: {missing_percentage:.1f}%"
            
            recommendations = []
            if not passed:
                if missing_percentage < 10:
                    recommendations.extend([
                        "Consider imputation values (mean/median/mode)",
                        "Verify correlation missing values other"
                    ])
                else:
                    recommendations.extend([
                        "Consider removal column or data",
                        "Analyze reasons missing values",
                        "Verify quality source data"
                    ])
            
            self.results.append(QualityCheckResult(
                check_name="missing_values",
                issue_type=DataQualityIssue.MISSING_VALUES,
                severity=severity,
                passed=passed,
                message=message,
                affected_rows=int(missing_count),
                affected_columns=[column],
                details={
                    'missing_count': int(missing_count),
                    'missing_percentage': missing_percentage,
                    'total_rows': len(df)
                },
                recommendations=recommendations
            ))
    
    def _check_duplicates(self, df: pd.DataFrame) -> None:
        """Verification records - Enterprise data uniqueness validation"""
        if not self.config.check_exact_duplicates:
            return
        
        # Exact duplicates
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        if duplicate_count == 0:
            self.results.append(QualityCheckResult(
                check_name="exact_duplicates",
                issue_type=DataQualityIssue.DUPLICATES,
                severity=QualityCheckSeverity.INFO,
                passed=True,
                message="Exact duplicates not detected",
                details={'duplicate_count': 0, 'duplicate_percentage': 0.0}
            ))
        else:
            severity = QualityCheckSeverity.WARNING if duplicate_percentage < 5 else QualityCheckSeverity.ERROR
            
            recommendations = [
                "Remove duplicate records df.drop_duplicates()",
                "Analyze reasons appearance",
                "Consider unique for records"
            ]
            
            self.results.append(QualityCheckResult(
                check_name="exact_duplicates",
                issue_type=DataQualityIssue.DUPLICATES,
                severity=severity,
                passed=False,
                message=f"Detected {duplicate_count} exact duplicates ({duplicate_percentage:.1f}%)",
                affected_rows=int(duplicate_count),
                details={
                    'duplicate_count': int(duplicate_count),
                    'duplicate_percentage': duplicate_percentage
                },
                recommendations=recommendations
            ))
        
        # Near duplicates (if enabled)
        if self.config.check_near_duplicates:
            self._check_near_duplicates(df)
    
    def _check_near_duplicates(self, df: pd.DataFrame) -> None:
        """Verification near-duplicates using similarity metrics"""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            from sklearn.preprocessing import StandardScaler
            
            # Using only numeric columns for similarity
            numeric_df = df.select_dtypes(include=[np.number])
            
 if numeric_df.empty or len(numeric_df) > 10000: # Skip for datasets
                return
            
            # Normalization data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df.fillna(0))
            
            # Calculation cosine similarity
            similarity_matrix = cosine_similarity(scaled_data)
            
            # Search near duplicates
            near_duplicates = 0
            threshold = self.config.near_duplicate_threshold
            
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i][j] > threshold:
                        near_duplicates += 1
            
            if near_duplicates > 0:
                percentage = (near_duplicates / len(df)) * 100
                self.results.append(QualityCheckResult(
                    check_name="near_duplicates",
                    issue_type=DataQualityIssue.DUPLICATES,
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"Detected {near_duplicates} near-duplicate pairs ({percentage:.1f}%)",
                    details={
                        'near_duplicate_pairs': near_duplicates,
                        'similarity_threshold': threshold
                    },
                    recommendations=[
                        "Analyze near-duplicate records",
                        "Consider records",
                        "Verify quality feature engineering"
                    ]
                ))
        
        except ImportError:
            self.logger.warning("scikit-learn not set - near-duplicate detection unavailable")
    
    def _check_data_types(self, df: pd.DataFrame) -> None:
        """Verification correctly types data - Enterprise schema validation"""
        type_issues = []
        
        for column in df.columns:
            column_dtype = df[column].dtype
            sample_values = df[column].dropna().head(100)
            
            # Verification on mixed types data
            if column_dtype == 'object':
                # Analysis object columns
                numeric_count = 0
                string_count = 0
                
                for value in sample_values:
                    try:
                        float(value)
                        numeric_count += 1
                    except (ValueError, TypeError):
                        string_count += 1
                
                if numeric_count > 0 and string_count > 0:
                    type_issues.append({
                        'column': column,
                        'issue': 'mixed_types',
                        'details': f'{numeric_count} numeric, {string_count} string values',
                        'recommendation': 'Convert type data or clean data'
                    })
                elif numeric_count > string_count * 0.8:  # >80% numeric
                    type_issues.append({
                        'column': column,
                        'issue': 'should_be_numeric',
                        'details': f'{numeric_count}/{len(sample_values)} values are numeric',
                        'recommendation': 'Consider numeric type'
                    })
        
        if type_issues:
            for issue in type_issues:
                self.results.append(QualityCheckResult(
                    check_name="data_types",
                    issue_type=DataQualityIssue.DATA_TYPE_MISMATCH,
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
 message=f"Issue data {issue['column']}: {issue['details']}",
                    affected_columns=[issue['column']],
                    recommendations=[issue['recommendation']]
                ))
        else:
            self.results.append(QualityCheckResult(
                check_name="data_types",
                issue_type=DataQualityIssue.DATA_TYPE_MISMATCH,
                severity=QualityCheckSeverity.INFO,
                passed=True,
                message="Types data"
            ))
    
    def _check_outliers(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> None:
        """Verification outliers - Enterprise statistical quality validation"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if feature_columns:
            numeric_columns = [col for col in feature_columns if col in numeric_columns]
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) < 10:  # Insufficient data for outlier detection
                continue
            
            outlier_count = 0
            outlier_method = self.config.outlier_method
            
            try:
                if outlier_method == "iqr":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR
                    
                    outliers = (series < lower_bound) | (series > upper_bound)
                    outlier_count = outliers.sum()
                
                elif outlier_method == "zscore":
                    z_scores = np.abs(stats.zscore(series))
                    outliers = z_scores > self.config.outlier_threshold
                    outlier_count = outliers.sum()
                
                elif outlier_method == "isolation_forest":
                    try:
                        from sklearn.ensemble import IsolationForest
                        iso_forest = IsolationForest(contamination='auto', random_state=42)
                        outliers = iso_forest.fit_predict(series.values.reshape(-1, 1)) == -1
                        outlier_count = outliers.sum()
                    except ImportError:
                        self.logger.warning("scikit-learn unavailable - using IQR method")
                        continue
                
                outlier_percentage = (outlier_count / len(series)) * 100
                
                # Detection severity
                if outlier_percentage <= self.config.max_outliers_percentage:
                    severity = QualityCheckSeverity.INFO
                    passed = True
                    message = f"level outliers {column}: {outlier_percentage:.1f}%"
                else:
                    severity = QualityCheckSeverity.WARNING
                    passed = False
                    message = f"High level outliers {column}: {outlier_percentage:.1f}%"
                
                recommendations = []
                if not passed:
                    recommendations.extend([
                        "Analyze nature outliers - error data or real extreme values",
                        "Consider methods processing outliers (winsorization, transformation)",
                        "For crypto data - outliers be (volatility )"
                    ])
                
                self.results.append(QualityCheckResult(
                    check_name="outliers",
                    issue_type=DataQualityIssue.OUTLIERS,
                    severity=severity,
                    passed=passed,
                    message=message,
                    affected_rows=int(outlier_count),
                    affected_columns=[column],
                    details={
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': outlier_percentage,
                        'method': outlier_method,
                        'threshold': self.config.outlier_threshold
                    },
                    recommendations=recommendations
                ))
            
            except Exception as e:
                self.logger.warning(f"Error at outlier detection for {column}: {e}")
    
    def _check_data_ranges(self, df: pd.DataFrame) -> None:
        """Verification ranges values - Enterprise business rules validation"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Custom ranges from configuration
        for column, (min_val, max_val) in self.config.custom_ranges.items():
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            violations = ((series < min_val) | (series > max_val)).sum()
            
            if violations > 0:
                violation_percentage = (violations / len(series)) * 100
                severity = QualityCheckSeverity.ERROR if violation_percentage > 5 else QualityCheckSeverity.WARNING
                
                self.results.append(QualityCheckResult(
                    check_name="data_ranges",
                    issue_type=DataQualityIssue.INVALID_RANGE,
                    severity=severity,
                    passed=False,
                    message=f"{column} in acceptable range [{min_val}, {max_val}]: {violations} ({violation_percentage:.1f}%)",
                    affected_rows=int(violations),
                    affected_columns=[column],
                    details={
                        'min_expected': min_val,
                        'max_expected': max_val,
                        'min_actual': float(series.min()),
                        'max_actual': float(series.max()),
                        'violations': int(violations)
                    },
                    recommendations=[
                        "Verify source data on errors",
                        "Apply clipping or filtering",
                        "Update business rules if range"
                    ]
                ))
        
        # verification for numeric columns
        for column in numeric_columns:
            series = df[column].dropna()
            
            # Verification on infinite values
            inf_count = np.isinf(series).sum()
            if inf_count > 0:
                self.results.append(QualityCheckResult(
                    check_name="infinite_values",
                    issue_type=DataQualityIssue.INVALID_RANGE,
                    severity=QualityCheckSeverity.ERROR,
                    passed=False,
                    message=f"Detected infinite values {column}: {inf_count}",
                    affected_rows=int(inf_count),
                    affected_columns=[column],
                    recommendations=[
                        "Remove or infinite values",
                        "Verify , on"
                    ]
                ))
    
    def _check_temporal_consistency(self, df: pd.DataFrame, datetime_columns: List[str]) -> None:
        """Verification temporal - Enterprise time series validation"""
        for column in datetime_columns:
            series = df[column].dropna().sort_values()
            
            if len(series) < 2:
                continue
            
            # Verification monotonicity (ascending order)
            if not series.is_monotonic_increasing:
                non_monotonic_count = (series.diff() < timedelta(0)).sum()
                self.results.append(QualityCheckResult(
                    check_name="temporal_order",
                    issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"temporal order {column}: {non_monotonic_count}",
                    affected_rows=int(non_monotonic_count),
                    affected_columns=[column],
                    recommendations=[
                        "Sort data by temporal",
                        "Verify correctly temporal",
                        "Remove duplicate temporal labels"
                    ]
                ))
            
            # Verification temporal
            if self.config.check_temporal_gaps:
                time_diffs = series.diff().dropna()
                max_gap_threshold = timedelta(hours=self.config.max_temporal_gap_hours)
                
                large_gaps = time_diffs > max_gap_threshold
                gap_count = large_gaps.sum()
                
                if gap_count > 0:
                    max_gap = time_diffs.max()
                    self.results.append(QualityCheckResult(
                        check_name="temporal_gaps",
                        issue_type=DataQualityIssue.TEMPORAL_INCONSISTENCY,
                        severity=QualityCheckSeverity.WARNING,
                        passed=False,
                        message=f"Detected temporal {column}: {gap_count} gaps, maximum gap: {max_gap}",
                        affected_rows=int(gap_count),
                        affected_columns=[column],
                        details={
                            'max_gap_hours': max_gap.total_seconds() / 3600,
                            'threshold_hours': self.config.max_temporal_gap_hours,
                            'gap_count': int(gap_count)
                        },
                        recommendations=[
                            "Analyze reasons temporal gaps",
                            "Consider interpolation or forward/backward fill",
                            "Verify stability source data"
                        ]
                    ))
    
    def _check_feature_quality(self, df: pd.DataFrame, feature_columns: List[str]) -> None:
        """Verification quality feature columns - Enterprise ML feature validation"""
        for column in feature_columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            
            # Verification on constant features
            if len(series.unique()) == 1:
                self.results.append(QualityCheckResult(
                    check_name="constant_feature",
                    issue_type=DataQualityIssue.INVALID_RANGE,
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"Feature {column} only value: {series.iloc[0]}",
                    affected_columns=[column],
                    recommendations=[
                        "Remove constant feature from models",
                        "Verify correctness feature engineering",
                        "Analyze source data"
                    ]
                ))
            
            # Verification on quasi-constant features (>95% values)
            elif len(series) > 0:
                most_frequent_value = series.value_counts().iloc[0]
                quasi_constant_threshold = 0.95
                
                if most_frequent_value / len(series) > quasi_constant_threshold:
                    dominant_value = series.value_counts().index[0]
                    percentage = (most_frequent_value / len(series)) * 100
                    
                    self.results.append(QualityCheckResult(
                        check_name="quasi_constant_feature",
                        issue_type=DataQualityIssue.INVALID_RANGE,
                        severity=QualityCheckSeverity.INFO,
                        passed=False,
                        message=f"Feature {column} is quasi-constant: {percentage:.1f}% values = {dominant_value}",
                        affected_columns=[column],
                        details={
                            'dominant_value': str(dominant_value),
                            'dominant_percentage': percentage,
                            'unique_values': len(series.unique())
                        },
                        recommendations=[
                            "Consider removal quasi-constant feature",
                            "Analyze feature",
                            "Verify feature engineering pipeline"
                        ]
                    ))
    
    def _check_feature_stability(self, df: pd.DataFrame, feature_columns: List[str], dataset_name: str) -> None:
        """Verification stability features against baseline - Enterprise drift detection"""
        if dataset_name not in self._baseline_stats:
 return # baseline for comparison
        
        baseline_stats = self._baseline_stats[dataset_name]
        
        for column in feature_columns:
            if column not in df.columns or column not in baseline_stats:
                continue
            
            current_series = df[column].dropna()
            baseline_info = baseline_stats[column]
            
            # Comparison main
            current_mean = current_series.mean() if len(current_series) > 0 else np.nan
            baseline_mean = baseline_info.get('mean', np.nan)
            
            if not (np.isnan(current_mean) or np.isnan(baseline_mean)):
                relative_change = abs((current_mean - baseline_mean) / baseline_mean)
                
                if relative_change > self.config.feature_drift_threshold:
                    severity = QualityCheckSeverity.WARNING if relative_change < 0.3 else QualityCheckSeverity.ERROR
                    
                    self.results.append(QualityCheckResult(
                        check_name="feature_stability",
                        issue_type=DataQualityIssue.DISTRIBUTION_SHIFT,
                        severity=severity,
                        passed=False,
                        message=f"Feature {column} drift: mean on {relative_change:.1%}",
                        affected_columns=[column],
                        details={
                            'baseline_mean': baseline_mean,
                            'current_mean': current_mean,
                            'relative_change': relative_change,
                            'threshold': self.config.feature_drift_threshold
                        },
                        recommendations=[
                            "Analyze reasons feature drift",
                            "Consider overfitting models",
                            "Verify stability source data"
                        ]
                    ))
    
    def _check_target_variable_quality(self, df: pd.DataFrame, target_column: str) -> None:
        """Verification quality target variable - Enterprise supervised learning validation"""
        target_series = df[target_column].dropna()
        
        if len(target_series) == 0:
            self.results.append(QualityCheckResult(
                check_name="target_variable",
                issue_type=DataQualityIssue.MISSING_VALUES,
                severity=QualityCheckSeverity.CRITICAL,
                passed=False,
                message=f"Target variable {target_column}",
                affected_columns=[target_column]
            ))
            return
        
        # Verification on class imbalance (for class)
        unique_values = target_series.unique()
        
        if len(unique_values) <= 10 and target_series.dtype in ['int64', 'int32', 'bool', 'object']:
            # Classification task
            value_counts = target_series.value_counts()
            class_proportions = value_counts / len(target_series)
            
            # Verification on class imbalance
            min_class_proportion = class_proportions.min()
            max_class_proportion = class_proportions.max()
            
 if min_class_proportion < 0.05: # 5% for class
                imbalance_ratio = max_class_proportion / min_class_proportion
                
                self.results.append(QualityCheckResult(
                    check_name="class_imbalance",
                    issue_type=DataQualityIssue.DISTRIBUTION_SHIFT,
                    severity=QualityCheckSeverity.WARNING if imbalance_ratio < 10 else QualityCheckSeverity.ERROR,
                    passed=False,
                    message=f"Detected class imbalance {target_column}: {imbalance_ratio:.1f}:1",
                    affected_columns=[target_column],
                    details={
                        'class_distribution': value_counts.to_dict(),
                        'min_class_proportion': min_class_proportion,
                        'max_class_proportion': max_class_proportion,
                        'imbalance_ratio': imbalance_ratio
                    },
                    recommendations=[
                        "Consider methods balancing (SMOTE, undersampling, oversampling)",
                        "Use appropriate metrics (F1, AUC, precision/recall)",
                        "class weights models"
                    ]
                ))
        
        else:
            # Regression task
            # Verification on normality distribution
            try:
                from scipy.stats import shapiro
 if len(target_series) <= 5000: # Shapiro-Wilk test
                    stat, p_value = shapiro(target_series.sample(min(1000, len(target_series))))
                    
                    if p_value < 0.05:
                        self.results.append(QualityCheckResult(
                            check_name="target_distribution",
                            issue_type=DataQualityIssue.DISTRIBUTION_SHIFT,
                            severity=QualityCheckSeverity.INFO,
                            passed=True,
                            message=f"Target variable {target_column} not is normal (p={p_value:.4f})",
                            affected_columns=[target_column],
                            details={'shapiro_p_value': p_value},
                            recommendations=[
                                "Consider transformation target variable (log, sqrt, Box-Cox)",
                                "Use models, non-normal distributions"
                            ]
                        ))
            except ImportError:
                self.logger.debug("SciPy unavailable - normality test")
    
    def _check_correlation_stability(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Verification stability correlations between features - Enterprise relationship monitoring"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return
        
        current_corr = numeric_df.corr()
        
        if dataset_name in self._baseline_stats and 'correlation_matrix' in self._baseline_stats[dataset_name]:
            baseline_corr = pd.DataFrame(self._baseline_stats[dataset_name]['correlation_matrix'])
            
            # Comparison correlations
            common_columns = current_corr.columns.intersection(baseline_corr.columns)
            
            if len(common_columns) >= 2:
                correlation_changes = []
                
                for i in range(len(common_columns)):
                    for j in range(i + 1, len(common_columns)):
                        col1, col2 = common_columns[i], common_columns[j]
                        
                        current_val = current_corr.loc[col1, col2]
                        baseline_val = baseline_corr.loc[col1, col2]
                        
                        if not (np.isnan(current_val) or np.isnan(baseline_val)):
                            change = abs(current_val - baseline_val)
                            
                            if change > self.config.correlation_change_threshold:
                                correlation_changes.append({
                                    'feature_pair': f"{col1} - {col2}",
                                    'baseline_corr': baseline_val,
                                    'current_corr': current_val,
                                    'change': change
                                })
                
                if correlation_changes:
                    max_change = max(change['change'] for change in correlation_changes)
                    
                    self.results.append(QualityCheckResult(
                        check_name="correlation_stability",
                        issue_type=DataQualityIssue.CORRELATION_ANOMALY,
                        severity=QualityCheckSeverity.WARNING if max_change < 0.5 else QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"Detected changes correlation: {len(correlation_changes)} features",
                        details={
                            'max_correlation_change': max_change,
                            'threshold': self.config.correlation_change_threshold,
 'changed_pairs': correlation_changes[:5] # Show only -5
                        },
                        recommendations=[
                            "Analyze reasons changes correlations",
                            "Verify stability feature engineering",
                            "Consider overfitting models"
                        ]
                    ))
    
    def _check_distribution_shift(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Verification distribution shift against baseline - Enterprise statistical monitoring"""
        if not self.config.check_distribution_shift:
            return
        
        baseline_stats = self._baseline_stats.get(dataset_name, {})
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        try:
            from scipy.stats import ks_2samp, chi2_contingency
            
            for column in numeric_columns:
                if column not in baseline_stats:
                    continue
                
                current_series = df[column].dropna()
                baseline_info = baseline_stats[column]
                
                # For distribution on baseline stats
                baseline_mean = baseline_info.get('mean', 0)
                baseline_std = baseline_info.get('std', 1)
                
                if baseline_std > 0:
                    # baseline distribution for comparison
                    baseline_sample = np.random.normal(baseline_mean, baseline_std, len(current_series))
                    
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p_value = ks_2samp(current_series, baseline_sample)
                    
                    if ks_p_value < self.config.distribution_p_value_threshold:
                        self.results.append(QualityCheckResult(
                            check_name="distribution_shift",
                            issue_type=DataQualityIssue.DISTRIBUTION_SHIFT,
                            severity=QualityCheckSeverity.WARNING if ks_p_value < 0.01 else QualityCheckSeverity.INFO,
                            passed=False,
                            message=f"Distribution shift detected {column}: KS p-value = {ks_p_value:.4f}",
                            affected_columns=[column],
                            details={
                                'ks_statistic': ks_stat,
                                'ks_p_value': ks_p_value,
                                'threshold': self.config.distribution_p_value_threshold
                            },
                            recommendations=[
                                "Analyze reasons distribution shift",
                                "Consider or transformation data",
                                "Update baseline statistics if changes"
                            ]
                        ))
        
        except ImportError:
            self.logger.warning("SciPy unavailable - distribution shift detection")
    
    def _check_crypto_specific_quality(self, df: pd.DataFrame) -> None:
        """Crypto trading specific data quality checks - Enterprise domain validation"""
        if not (self.config.validate_price_data or self.config.validate_volume_data):
            return
        
        # Price data validation
        price_columns = [col for col in df.columns if 
                        any(price_keyword in col.lower() for price_keyword in 
                            ['price', 'close', 'open', 'high', 'low', 'last'])]
        
        if self.config.validate_price_data and price_columns:
            for column in price_columns:
                series = df[column].dropna()
                
                if len(series) == 0:
                    continue
                
                # Verification range prices
                invalid_prices = ((series < self.config.min_price_value) | 
                                (series > self.config.max_price_value)).sum()
                
                if invalid_prices > 0:
                    percentage = (invalid_prices / len(series)) * 100
                    
                    self.results.append(QualityCheckResult(
                        check_name="crypto_price_validation",
                        issue_type=DataQualityIssue.INVALID_RANGE,
                        severity=QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"prices {column}: {invalid_prices} values ({percentage:.1f}%) in range [{self.config.min_price_value}, {self.config.max_price_value}]",
                        affected_rows=int(invalid_prices),
                        affected_columns=[column],
                        details={
                            'min_valid_price': self.config.min_price_value,
                            'max_valid_price': self.config.max_price_value,
                            'min_actual': float(series.min()),
                            'max_actual': float(series.max()),
                            'invalid_count': int(invalid_prices)
                        },
                        recommendations=[
                            "Verify source price data",
                            "Remove or prices",
                            "Verify correctness currency pairs"
                        ]
                    ))
                
                # Verification on zero prices
                zero_prices = (series == 0).sum()
                if zero_prices > 0:
                    self.results.append(QualityCheckResult(
                        check_name="zero_prices",
                        issue_type=DataQualityIssue.INVALID_RANGE,
                        severity=QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"Detected zero prices {column}: {zero_prices} values",
                        affected_rows=int(zero_prices),
                        affected_columns=[column],
                        recommendations=[
                            "Remove records zero",
                            "Verify quality market data feed",
                            "Consider forward/backward fill for short-term gaps"
                        ]
                    ))
        
        # Volume data validation
        volume_columns = [col for col in df.columns if 
                         any(vol_keyword in col.lower() for vol_keyword in 
                             ['volume', 'vol', 'quantity', 'amount'])]
        
        if self.config.validate_volume_data and volume_columns:
            for column in volume_columns:
                series = df[column].dropna()
                
                if len(series) == 0:
                    continue
                
                # Verification on negative volumes
                negative_volumes = (series < self.config.min_volume_value).sum()
                if negative_volumes > 0:
                    self.results.append(QualityCheckResult(
                        check_name="negative_volumes",
                        issue_type=DataQualityIssue.INVALID_RANGE,
                        severity=QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"Detected negative volumes {column}: {negative_volumes} values",
                        affected_rows=int(negative_volumes),
                        affected_columns=[column],
                        recommendations=[
                            "Remove records negative",
                            "Verify correctness processing buy/sell orders"
                        ]
                    ))
    
    def _check_advanced_quality_issues(self, df: pd.DataFrame) -> None:
        """Advanced quality checks - Enterprise enterprise-grade validation"""
        # Verification on encoding issues
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for column in text_columns:
            series = df[column].dropna().astype(str)
            
            # Search or encoding issues
            encoding_issues = 0
 for value in series.head(1000): # Checking 1000 values
                try:
                    value.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    encoding_issues += 1
            
            if encoding_issues > 0:
                self.results.append(QualityCheckResult(
                    check_name="encoding_issues",
                    issue_type=DataQualityIssue.INCONSISTENT_FORMAT,
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"Detected issues {column}: {encoding_issues} values",
                    affected_rows=encoding_issues,
                    affected_columns=[column],
                    recommendations=[
                        "Verify encoding input data",
                        "Apply encoding at files",
                        "data from"
                    ]
                ))
    
    def _update_baseline_stats(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Update baseline statistics for future comparison"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        baseline_stats = {}
        
        # Statistics by each numeric column
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            
            if len(series) > 0:
                baseline_stats[column] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'median': float(series.median()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'q1': float(series.quantile(0.25)),
                    'q3': float(series.quantile(0.75)),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                }
        
        # Correlation matrix
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            baseline_stats['correlation_matrix'] = corr_matrix.to_dict()
        
        # Shape info
        baseline_stats['shape'] = {
            'rows': df.shape[0],
            'columns': df.shape[1],
            'numeric_columns': len(numeric_df.columns)
        }
        
        # Missing values stats
        baseline_stats['missing_stats'] = {
            column: int(df[column].isnull().sum())
            for column in df.columns
        }
        
        self._baseline_stats[dataset_name] = baseline_stats
        
        # history validation
        self._validation_history.append({
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'stats': baseline_stats
        })
    
    def _generate_summary_result(self, df: pd.DataFrame, dataset_name: str) -> None:
        """Generation one result validation"""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Counting by severity levels
        critical_issues = sum(1 for r in self.results if not r.passed and r.severity == QualityCheckSeverity.CRITICAL)
        error_issues = sum(1 for r in self.results if not r.passed and r.severity == QualityCheckSeverity.ERROR)
        warning_issues = sum(1 for r in self.results if not r.passed and r.severity == QualityCheckSeverity.WARNING)
        
        # status quality data
        if critical_issues > 0:
            overall_quality = "CRITICAL"
            quality_severity = QualityCheckSeverity.CRITICAL
        elif error_issues > 0:
            overall_quality = "POOR"
            quality_severity = QualityCheckSeverity.ERROR
        elif warning_issues > 0:
            overall_quality = "ACCEPTABLE"
            quality_severity = QualityCheckSeverity.WARNING
        else:
            overall_quality = "EXCELLENT"
            quality_severity = QualityCheckSeverity.INFO
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        summary_message = (f"Dataset {dataset_name} - Quality data: {overall_quality} "
                          f"({passed_checks}/{total_checks} checks , {success_rate:.1f}%)")
        
        self.results.append(QualityCheckResult(
            check_name="data_quality_summary",
            issue_type=DataQualityIssue.SCHEMA_VIOLATION,
            severity=quality_severity,
            passed=(critical_issues == 0 and error_issues == 0),
            message=summary_message,
            details={
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': success_rate,
                'critical_issues': critical_issues,
                'error_issues': error_issues,
                'warning_issues': warning_issues,
                'dataset_shape': df.shape,
                'overall_quality': overall_quality
            }
        ))
    
    def generate_quality_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """Generation comprehensive data quality report - Enterprise reporting"""
        if not self.results:
            return {'status': 'no_validation_run'}
        
        # Grouping results by issues
        issues_by_type = {}
        for result in self.results:
            issue_type = result.issue_type.value
            if issue_type not in issues_by_type:
                issues_by_type[issue_type] = []
            issues_by_type[issue_type].append(result)
        
        # Severity statistics
        severity_stats = {}
        for severity in QualityCheckSeverity:
            severity_results = [r for r in self.results if r.severity == severity]
            severity_stats[severity.value] = {
                'total': len(severity_results),
                'passed': sum(1 for r in severity_results if r.passed),
                'failed': sum(1 for r in severity_results if not r.passed)
            }
        
        # Top issues (most critical)
        critical_issues = [
            {
                'check_name': r.check_name,
                'issue_type': r.issue_type.value,
                'severity': r.severity.value,
                'message': r.message,
                'affected_rows': r.affected_rows,
                'affected_columns': r.affected_columns,
                'recommendations': r.recommendations if include_recommendations else []
            }
            for r in self.results 
            if not r.passed and r.severity in [QualityCheckSeverity.CRITICAL, QualityCheckSeverity.ERROR]
        ]
        
        # Overall statistics
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': total_checks - passed_checks,
                'success_rate': success_rate,
                'overall_status': 'PASS' if len(critical_issues) == 0 else 'FAIL'
            },
            'severity_breakdown': severity_stats,
            'issues_by_type': {
                issue_type: len(results) for issue_type, results in issues_by_type.items()
            },
            'critical_issues': critical_issues,
            'detailed_results': [
                {
                    'check_name': r.check_name,
                    'issue_type': r.issue_type.value,
                    'severity': r.severity.value,
                    'passed': r.passed,
                    'message': r.message,
                    'affected_rows': r.affected_rows,
                    'affected_columns': r.affected_columns,
                    'details': r.details,
                    'recommendations': r.recommendations if include_recommendations else [],
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        if include_recommendations:
            report['general_recommendations'] = self._generate_general_recommendations()
        
        return report
    
    def _generate_general_recommendations(self) -> List[str]:
        """Generation by result data quality validation"""
        recommendations = []
        failed_results = [r for r in self.results if not r.passed]
        
        # Missing values recommendations
        missing_issues = [r for r in failed_results if r.issue_type == DataQualityIssue.MISSING_VALUES]
        if missing_issues:
            recommendations.append(
                "Detected issues missing values - implement strategy"
                "or data"
            )
        
        # Outliers recommendations
        outlier_issues = [r for r in failed_results if r.issue_type == DataQualityIssue.OUTLIERS]
        if outlier_issues:
            recommendations.append(
                "High level outliers - analysis their nature"
                "appropriate methods processing"
            )
        
        # Duplicates recommendations
        duplicate_issues = [r for r in failed_results if r.issue_type == DataQualityIssue.DUPLICATES]
        if duplicate_issues:
            recommendations.append(
                "Detected duplicates - clean dataset for quality models"
            )
        
        # Distribution shift recommendations
        drift_issues = [r for r in failed_results if r.issue_type == DataQualityIssue.DISTRIBUTION_SHIFT]
        if drift_issues:
            recommendations.append(
                "Distribution drift detected - consider overfitting models "
                "or baseline statistics"
            )
        
        # Critical issues
        critical_issues = [r for r in failed_results if r.severity == QualityCheckSeverity.CRITICAL]
        if critical_issues:
            recommendations.append(
                ": data for training models before elimination"
                "all critical issues quality"
            )
        
        if not recommendations:
            recommendations.append("Data quality - for ML pipeline")
        
        return recommendations
    
    def export_report(self, filepath: str, include_recommendations: bool = True) -> None:
        """Export data quality report file - Enterprise reporting"""
        report = self.generate_quality_report(include_recommendations)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Data quality report exported {filepath}")
    
    def load_baseline_stats(self, filepath: str) -> None:
        """Load baseline statistics from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self._baseline_stats = json.load(f)
            self.logger.info(f"Baseline statistics from {filepath}")
        except Exception as e:
            self.logger.error(f"Error at loading baseline statistics: {e}")
    
    def save_baseline_stats(self, filepath: str) -> None:
        """Save baseline statistics file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self._baseline_stats, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Baseline statistics saved {filepath}")
        except Exception as e:
            self.logger.error(f"Error at saving baseline statistics: {e}")


def create_crypto_trading_data_validator() -> DataQualityValidator:
    """
    Factory function for creation data quality validator for crypto trading data
    Enterprise pre-configured validator for financial datasets
    """
    config = DataQualityConfig(
        # Strict requirements for financial data
        max_missing_percentage=2.0,  # 2% maximum missing values
        critical_missing_percentage=10.0,  # 10% critical level
        
        # Crypto market settings
        validate_price_data=True,
 min_price_value=1e-8, # very prices ()
        max_price_value=10000000.0,  # 10M maximum for BTC
        
        validate_volume_data=True,
        min_volume_value=0.0,
        
        # Outlier detection for volatile crypto markets
        outlier_method="iqr",
 outlier_threshold=2.0, # strict threshold for crypto volatility
        max_outliers_percentage=10.0,  # 10% outliers normal for crypto
        
        # Temporal consistency for trading data
        check_temporal_order=True,
        check_temporal_gaps=True,
 max_temporal_gap_hours=1.0, # 1 maximum gap for crypto data
        
        # Advanced monitoring for trading systems
        check_distribution_shift=True,
        distribution_p_value_threshold=0.01,  # Strict threshold
        
        check_correlation_stability=True,
        correlation_change_threshold=0.2,  # 20% change threshold
        
        check_feature_stability=True,
        feature_drift_threshold=0.05,  # 5% feature drift threshold
        
        # Enterprise enterprise settings
        enable_advanced_checks=True,
        save_quality_reports=True,
        generate_recommendations=True
    )
    
    return DataQualityValidator(config)