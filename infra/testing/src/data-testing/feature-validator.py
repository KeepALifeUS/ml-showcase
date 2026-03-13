"""
Feature Validator - Enterprise Enterprise Feature Engineering Validation
Comprehensive validation framework for ML features crypto trading systems

Applies enterprise principles:
- Enterprise feature governance
- Production feature monitoring
- Statistical feature validation
- Automated feature quality assurance
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import logging
import json
from pathlib import Path

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureIssueType(Enum):
    """Types issues features for Enterprise governance"""
    CONSTANT_FEATURE = "constant_feature"
    QUASI_CONSTANT_FEATURE = "quasi_constant_feature"
    HIGH_CORRELATION = "high_correlation"
    LOW_VARIANCE = "low_variance"
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    FEATURE_DRIFT = "feature_drift"
    INFORMATION_LEAKAGE = "information_leakage"
    SCALING_ISSUE = "scaling_issue"
    DISTRIBUTION_ANOMALY = "distribution_anomaly"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"


class FeatureValidationSeverity(Enum):
    """Levels severity for Enterprise alerting"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FeatureValidationResult:
    """Result validation feature - Enterprise structured result"""
    feature_name: str
    issue_type: FeatureIssueType
    severity: FeatureValidationSeverity
    passed: bool
    message: str
 impact_score: float = 0.0 # 0-1 score on model
    affected_samples: int = 0
    statistical_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.feature_name}: {self.message}"


@dataclass
class FeatureValidationConfig:
    """Configuration feature validation - Enterprise typed configuration"""
    # Constant feature detection
 constant_threshold: float = 0.95 # 95% values = quasi-constant
    min_unique_values: int = 2  # Minimum unique values
    
    # Correlation analysis
    high_correlation_threshold: float = 0.95  # High correlation between features
    check_target_correlation: bool = True
 min_target_correlation: float = 0.01 # Minimum correlation target
    
    # Variance analysis
    min_variance_threshold: float = 1e-6  # Minimum variance
 normalize_variance: bool = True # by scale
    
    # Missing values tolerance
    max_missing_percentage: float = 5.0  # 5% maximum missing values
    critical_missing_threshold: float = 20.0  # 20% critical level
    
    # Outlier detection
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation_forest"
    outlier_threshold: float = 1.5
    max_outlier_percentage: float = 5.0
    
    # Distribution analysis
 check_normality: bool = False # Not normality for crypto data
    check_skewness: bool = True
 max_skewness: float = 5.0 # skewness
 max_kurtosis: float = 10.0 #
    
    # Feature drift detection
    enable_drift_detection: bool = True
 drift_threshold: float = 0.1 # 10% change statistics
    drift_detection_methods: List[str] = field(default_factory=lambda: ["mean", "std", "quantiles"])
    
    # Information leakage detection
    check_information_leakage: bool = True
 perfect_correlation_threshold: float = 0.999 # on leakage
    
    # Scaling validation
    check_feature_scaling: bool = True
 scaling_tolerance: float = 100.0 #
    
    # Temporal consistency (for time series features)
    check_temporal_consistency: bool = True
    temporal_gap_tolerance_hours: float = 24.0
    
    # Enterprise enterprise settings
    enable_advanced_analysis: bool = True
    generate_feature_profiles: bool = True
    save_validation_history: bool = True
    
    # Crypto trading specific
    validate_trading_features: bool = True
    price_feature_keywords: List[str] = field(default_factory=lambda: 
        ["price", "close", "open", "high", "low", "last", "ask", "bid"])
    volume_feature_keywords: List[str] = field(default_factory=lambda: 
        ["volume", "vol", "quantity", "amount", "size"])
    technical_indicator_keywords: List[str] = field(default_factory=lambda: 
        ["rsi", "macd", "sma", "ema", "bollinger", "stoch", "atr", "adx"])


class FeatureProfile:
    """Enterprise feature profile for monitoring comparison"""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.created_at = datetime.now()
        self.statistics: Dict[str, Any] = {}
        self.distribution_info: Dict[str, Any] = {}
        self.quality_metrics: Dict[str, float] = {}
        self.validation_history: List[Dict[str, Any]] = []
    
    def update_profile(self, series: pd.Series) -> None:
        """Update profile on basis data"""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return
        
        # Basic statistics
        self.statistics = {
            'count': len(series),
            'non_null_count': len(clean_series),
            'null_count': series.isnull().sum(),
            'null_percentage': (series.isnull().sum() / len(series)) * 100,
            'unique_count': clean_series.nunique(),
            'unique_percentage': (clean_series.nunique() / len(clean_series)) * 100,
            'mean': float(clean_series.mean()) if clean_series.dtype.kind in 'biufc' else None,
            'std': float(clean_series.std()) if clean_series.dtype.kind in 'biufc' else None,
            'min': float(clean_series.min()) if clean_series.dtype.kind in 'biufc' else str(clean_series.min()),
            'max': float(clean_series.max()) if clean_series.dtype.kind in 'biufc' else str(clean_series.max()),
            'median': float(clean_series.median()) if clean_series.dtype.kind in 'biufc' else None,
            'q1': float(clean_series.quantile(0.25)) if clean_series.dtype.kind in 'biufc' else None,
            'q3': float(clean_series.quantile(0.75)) if clean_series.dtype.kind in 'biufc' else None
        }
        
        # Distribution info for numeric features
        if clean_series.dtype.kind in 'biufc' and len(clean_series) > 1:
            self.distribution_info = {
                'skewness': float(clean_series.skew()),
                'kurtosis': float(clean_series.kurtosis()),
                'variance': float(clean_series.var()),
                'coefficient_of_variation': float(clean_series.std() / clean_series.mean()) if clean_series.mean() != 0 else float('inf')
            }
        
        # Quality metrics
        self.quality_metrics = {
            'completeness': (len(clean_series) / len(series)) * 100,
            'uniqueness': (clean_series.nunique() / len(clean_series)) * 100 if len(clean_series) > 0 else 0,
            'consistency': 100.0 - (self._calculate_inconsistency_score(clean_series) * 100)
        }
        
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'statistics': self.statistics.copy(),
            'distribution_info': self.distribution_info.copy(),
            'quality_metrics': self.quality_metrics.copy()
        })
    
    def _calculate_inconsistency_score(self, series: pd.Series) -> float:
        """Calculation inconsistency score (0 = consistent, 1 = inconsistent)"""
        if series.dtype.kind not in 'biufc':
 return 0.0 # For categorical data inconsistency
        
        if len(series) < 2:
            return 0.0
        
        # Using coefficient of variation as inconsistency
        cv = series.std() / series.mean() if series.mean() != 0 else 0
        # CV range [0, 1]
        return min(cv / 2.0, 1.0)  # Arbitrary normalization
    
    def compare_with_baseline(self, baseline_profile: 'FeatureProfile') -> Dict[str, float]:
        """Comparison baseline profile"""
        changes = {}
        
        for metric, current_value in self.statistics.items():
            if metric in baseline_profile.statistics and current_value is not None:
                baseline_value = baseline_profile.statistics[metric]
                if baseline_value is not None and baseline_value != 0:
                    relative_change = abs((current_value - baseline_value) / baseline_value)
                    changes[f"{metric}_relative_change"] = relative_change
        
        return changes


class FeatureValidator:
    """
    Enterprise Enterprise Feature Validator
    
 Comprehensive feature validation framework for ML features crypto trading systems.
 Provides enterprise-grade feature governance automated quality assurance.
    """
    
    def __init__(self, config: Optional[FeatureValidationConfig] = None):
        """
 Initialization feature validator Enterprise configuration
        
        Args:
            config: Configuration feature validation (Enterprise typed)
        """
        self.config = config or FeatureValidationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.results: List[FeatureValidationResult] = []
        
        # Enterprise feature profiles for monitoring
        self.feature_profiles: Dict[str, FeatureProfile] = {}
        self.baseline_profiles: Dict[str, FeatureProfile] = {}
        
        # Statistical methods
        self._init_statistical_methods()
    
    def _init_statistical_methods(self) -> None:
        """Initialization statistical method"""
        try:
            from scipy import stats
            self.stats_available = True
        except ImportError:
            self.stats_available = False
            self.logger.warning("SciPy unavailable - tests will be skipped")
        
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            self.logger.warning("scikit-learn unavailable - advanced methods will be skipped")
    
    def validate_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None,
        dataset_name: str = "dataset"
    ) -> List[FeatureValidationResult]:
        """
 validation features - Enterprise enterprise validation pipeline
        
        Args:
 X: DataFrame features
            y: Target variable (optional)
            feature_names: List feature names for validation (optional)
 dataset_name: Name dataset for
        
        Returns:
            List[FeatureValidationResult]: Results all feature validations
        """
        self.logger.info(f"Starting Enterprise feature validation for {dataset_name}")
        self.results.clear()
        
        validation_start_time = datetime.now()
        
        # Detection features for validation
        features_to_validate = feature_names or X.columns.tolist()
        
        try:
            # 1. Individual Feature Validation
            for feature_name in features_to_validate:
                if feature_name not in X.columns:
                    self._add_result(
                        feature_name, FeatureIssueType.MISSING_VALUES, 
                        FeatureValidationSeverity.ERROR, False,
                        f"Feature {feature_name} dataset"
                    )
                    continue
                
                series = X[feature_name]
                
                # Create/ feature profile
                if feature_name not in self.feature_profiles:
                    self.feature_profiles[feature_name] = FeatureProfile(feature_name)
                
                self.feature_profiles[feature_name].update_profile(series)
                
                # Individual feature checks
                self._validate_single_feature(feature_name, series, y)
            
            # 2. Cross-Feature Validation
            if len(features_to_validate) > 1:
                self._validate_feature_correlations(X[features_to_validate])
                self._validate_feature_scaling(X[features_to_validate])
            
            # 3. Target-Feature Relationships (if target provided)
            if y is not None:
                self._validate_target_relationships(X[features_to_validate], y)
            
            # 4. Drift Detection (if is baseline)
            if self.config.enable_drift_detection:
                self._detect_feature_drift(X[features_to_validate], dataset_name)
            
            # 5. Crypto Trading Specific Validation
            if self.config.validate_trading_features:
                self._validate_trading_features(X[features_to_validate])
            
            # 6. Advanced Enterprise Validation
            if self.config.enable_advanced_analysis:
                self._advanced_feature_analysis(X[features_to_validate], y)
            
            validation_time = (datetime.now() - validation_start_time).total_seconds()
            self.logger.info(f"Feature validation completed in {validation_time:.2f} seconds")
            
            # Generation one result
            self._generate_validation_summary(X[features_to_validate], dataset_name)
            
        except Exception as e:
            self.logger.error(f"Error at feature validation: {e}")
            self._add_result(
                "validation_error", FeatureIssueType.DISTRIBUTION_ANOMALY,
                FeatureValidationSeverity.CRITICAL, False,
                f"Critical error feature validation: {str(e)}",
                statistical_details={'error': str(e)}
            )
        
        return self.results.copy()
    
    def _validate_single_feature(self, feature_name: str, series: pd.Series, y: Optional[pd.Series] = None) -> None:
        """Validation individual feature"""
        clean_series = series.dropna()
        
        # 1. Missing Values Check
        self._check_missing_values(feature_name, series)
        
        # 2. Constant/Quasi-Constant Check
        self._check_constant_feature(feature_name, clean_series)
        
        # 3. Variance Check (for numeric features)
        if clean_series.dtype.kind in 'biufc' and len(clean_series) > 1:
            self._check_feature_variance(feature_name, clean_series)
            
            # 4. Outlier Detection
            self._check_feature_outliers(feature_name, clean_series)
            
            # 5. Distribution Analysis
            self._check_feature_distribution(feature_name, clean_series)
        
        # 6. Temporal Consistency (if feature contains information)
        if self.config.check_temporal_consistency and self._is_temporal_feature(feature_name, series):
            self._check_temporal_consistency(feature_name, series)
    
    def _check_missing_values(self, feature_name: str, series: pd.Series) -> None:
        """Verification missing values feature"""
        missing_count = series.isnull().sum()
        missing_percentage = (missing_count / len(series)) * 100
        
        if missing_percentage == 0:
            return  # No missing values
        
        # Detection severity
        if missing_percentage <= self.config.max_missing_percentage:
            severity = FeatureValidationSeverity.INFO
            passed = True
        elif missing_percentage <= self.config.critical_missing_threshold:
            severity = FeatureValidationSeverity.WARNING
            passed = False
        else:
            severity = FeatureValidationSeverity.CRITICAL
            passed = False
        
        impact_score = min(missing_percentage / 100, 1.0)  # Higher missing % = higher impact
        
        recommendations = []
        if not passed:
            if missing_percentage < 10:
                recommendations.extend([
                    "Consider imputation missing values (mean/median/mode)",
                    "Analyze missing values",
                    "Create indicator variable for missingness"
                ])
            else:
                recommendations.extend([
                    "Consider removal feature or",
                    "Verify quality source data",
                    "Analyze correlation missingness target variable"
                ])
        
        self._add_result(
            feature_name, FeatureIssueType.MISSING_VALUES, severity, passed,
            f"Missing values: {missing_count} ({missing_percentage:.1f}%)",
            impact_score=impact_score,
            affected_samples=int(missing_count),
            statistical_details={
                'missing_count': int(missing_count),
                'missing_percentage': missing_percentage,
                'total_samples': len(series)
            },
            recommendations=recommendations
        )
    
    def _check_constant_feature(self, feature_name: str, series: pd.Series) -> None:
        """Verification on constant quasi-constant features"""
        if len(series) == 0:
            return
        
        unique_count = series.nunique()
        total_count = len(series)
        
        # Constant feature check
        if unique_count == 1:
            self._add_result(
                feature_name, FeatureIssueType.CONSTANT_FEATURE,
                FeatureValidationSeverity.ERROR, False,
                f"Feature is constant: all values = {series.iloc[0]}",
                impact_score=1.0,  # Maximum impact - useless feature
                statistical_details={
                    'unique_values': unique_count,
                    'constant_value': str(series.iloc[0])
                },
                recommendations=[
                    "Remove constant feature from models",
                    "Verify correctly feature engineering",
                    "Analyze source data on errors"
                ]
            )
            return
        
        # Quasi-constant feature check
        most_frequent_count = series.value_counts().iloc[0]
        quasi_constant_ratio = most_frequent_count / total_count
        
        if quasi_constant_ratio > self.config.constant_threshold:
            dominant_value = series.value_counts().index[0]
            
            severity = FeatureValidationSeverity.WARNING if quasi_constant_ratio < 0.99 else FeatureValidationSeverity.ERROR
            impact_score = quasi_constant_ratio  # Higher ratio = higher impact
            
            self._add_result(
                feature_name, FeatureIssueType.QUASI_CONSTANT_FEATURE, severity, False,
                f"Feature is quasi-constant: {quasi_constant_ratio:.1%} values = {dominant_value}",
                impact_score=impact_score,
                statistical_details={
                    'unique_values': unique_count,
                    'dominant_value': str(dominant_value),
                    'dominant_ratio': quasi_constant_ratio,
                    'threshold': self.config.constant_threshold
                },
                recommendations=[
                    "Consider removal quasi-constant feature",
                    "Analyze feature",
                    "Verify feature engineering pipeline on correctness"
                ]
            )
    
    def _check_feature_variance(self, feature_name: str, series: pd.Series) -> None:
        """Verification variance feature"""
        if len(series) < 2:
            return
        
        variance = series.var()
        
        # Normalization variance (if enabled)
        if self.config.normalize_variance:
            mean_abs = abs(series.mean())
            normalized_variance = variance / (mean_abs ** 2) if mean_abs > 0 else variance
        else:
            normalized_variance = variance
        
        if normalized_variance < self.config.min_variance_threshold:
            severity = FeatureValidationSeverity.WARNING
            impact_score = 1.0 - (normalized_variance / self.config.min_variance_threshold)
            
            self._add_result(
                feature_name, FeatureIssueType.LOW_VARIANCE, severity, False,
                f"Low variance: {variance:.6f} (: {normalized_variance:.6f})",
                impact_score=impact_score,
                statistical_details={
                    'variance': variance,
                    'normalized_variance': normalized_variance,
                    'threshold': self.config.min_variance_threshold,
                    'std': series.std(),
                    'coefficient_of_variation': series.std() / series.mean() if series.mean() != 0 else float('inf')
                },
                recommendations=[
                    "Consider removal low-variance feature",
                    "Verify scaling normalization data",
                    "Analyze information content feature"
                ]
            )
    
    def _check_feature_outliers(self, feature_name: str, series: pd.Series) -> None:
        """Verification outliers feature"""
        outlier_count = 0
        outlier_indices = []
        
        try:
            if self.config.outlier_method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                outliers_mask = (series < lower_bound) | (series > upper_bound)
                outlier_count = outliers_mask.sum()
                outlier_indices = series[outliers_mask].index.tolist()
            
            elif self.config.outlier_method == "zscore":
                z_scores = np.abs(stats.zscore(series))
                outliers_mask = z_scores > self.config.outlier_threshold
                outlier_count = outliers_mask.sum()
                outlier_indices = series[outliers_mask].index.tolist()
            
            elif self.config.outlier_method == "isolation_forest" and self.sklearn_available:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination='auto', random_state=42)
                outliers_mask = iso_forest.fit_predict(series.values.reshape(-1, 1)) == -1
                outlier_count = outliers_mask.sum()
                outlier_indices = series[outliers_mask].index.tolist()
            
            outlier_percentage = (outlier_count / len(series)) * 100
            
            if outlier_percentage <= self.config.max_outlier_percentage:
                severity = FeatureValidationSeverity.INFO
                passed = True
            else:
                severity = FeatureValidationSeverity.WARNING
                passed = False
            
            impact_score = min(outlier_percentage / 100, 1.0)
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    "Analyze nature outliers - error data or real values",
                    "Consider methods processing outliers (winsorization, transformation)",
                    "For crypto features - outliers be (market volatility)"
                ])
            
            self._add_result(
                feature_name, FeatureIssueType.OUTLIERS, severity, passed,
                f"Outliers: {outlier_count} ({outlier_percentage:.1f}%)",
                impact_score=impact_score,
                affected_samples=int(outlier_count),
                statistical_details={
                    'outlier_count': int(outlier_count),
                    'outlier_percentage': outlier_percentage,
                    'method': self.config.outlier_method,
                    'threshold': self.config.outlier_threshold,
 'outlier_indices': outlier_indices[:100] # number for report
                },
                recommendations=recommendations
            )
        
        except Exception as e:
            self.logger.debug(f"Error at outlier detection for {feature_name}: {e}")
    
    def _check_feature_distribution(self, feature_name: str, series: pd.Series) -> None:
        """Analysis distribution feature"""
        if len(series) < 3:  # Insufficient data for analysis distribution
            return
        
        try:
            skewness = series.skew()
            kurtosis = series.kurtosis()
            
            issues = []
            recommendations = []
            
            # Verification skewness
            if abs(skewness) > self.config.max_skewness:
                issues.append(f"high skewness ({skewness:.2f})")
                recommendations.extend([
                    "Consider logarithmic transformation",
                    "Box-Cox or Yeo-Johnson transformation",
                    "For crypto data high skewness may be normal"
                ])
            
            # Verification kurtosis
            if abs(kurtosis) > self.config.max_kurtosis:
                issues.append(f"high ({kurtosis:.2f})")
                recommendations.extend([
                    "Verify on outliers",
                    "Consider robust scaling methods"
                ])
            
            if issues:
                severity = FeatureValidationSeverity.INFO if len(issues) == 1 else FeatureValidationSeverity.WARNING
                impact_score = min((abs(skewness) + abs(kurtosis)) / 20, 1.0)  # Normalized impact
                
                self._add_result(
                    feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY, severity, False,
 f" distribution: {', '.join(issues)}",
                    impact_score=impact_score,
                    statistical_details={
                        'skewness': skewness,
                        'kurtosis': kurtosis,
                        'max_skewness_threshold': self.config.max_skewness,
                        'max_kurtosis_threshold': self.config.max_kurtosis
                    },
                    recommendations=recommendations
                )
            
            # Normality test (if enabled)
            if self.config.check_normality and self.stats_available:
                from scipy.stats import shapiro
 if len(series) <= 5000: # Shapiro-Wilk test
                    try:
                        stat, p_value = shapiro(series.sample(min(1000, len(series))))
                        if p_value < 0.05:
                            self._add_result(
                                feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY,
                                FeatureValidationSeverity.INFO, True,
                                f"Feature not is normal (Shapiro p={p_value:.4f})",
                                statistical_details={'shapiro_p_value': p_value},
                                recommendations=[
                                    "Normality not for ML",
                                    "Consider transformation if linear models"
                                ]
                            )
                    except Exception as e:
                        self.logger.debug(f"Normality test failed for {feature_name}: {e}")
        
        except Exception as e:
            self.logger.debug(f"Distribution analysis failed for {feature_name}: {e}")
    
    def _is_temporal_feature(self, feature_name: str, series: pd.Series) -> bool:
        """Detection is feature temporal"""
        # Verification on datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Verification on temporal keywords
        temporal_keywords = ['time', 'date', 'timestamp', 'hour', 'day', 'month', 'year']
        return any(keyword in feature_name.lower() for keyword in temporal_keywords)
    
    def _check_temporal_consistency(self, feature_name: str, series: pd.Series) -> None:
        """Verification temporal feature"""
        if not pd.api.types.is_datetime64_any_dtype(series):
            return
        
        clean_series = series.dropna().sort_values()
        
        if len(clean_series) < 2:
            return
        
        # Verification monotonicity
        if not clean_series.is_monotonic_increasing:
            non_monotonic_count = (clean_series.diff() < timedelta(0)).sum()
            
            self._add_result(
                feature_name, FeatureIssueType.TEMPORAL_INCONSISTENCY,
                FeatureValidationSeverity.WARNING, False,
                f"temporal order: {non_monotonic_count}",
                impact_score=min(non_monotonic_count / len(clean_series), 1.0),
                affected_samples=int(non_monotonic_count),
                recommendations=[
                    "Sort data by temporal feature",
                    "Verify correctly temporal",
                    "Remove duplicate temporal labels"
                ]
            )
        
        # Verification temporal
        time_diffs = clean_series.diff().dropna()
        max_gap_threshold = timedelta(hours=self.config.temporal_gap_tolerance_hours)
        
        large_gaps = time_diffs > max_gap_threshold
        gap_count = large_gaps.sum()
        
        if gap_count > 0:
            max_gap = time_diffs.max()
            
            self._add_result(
                feature_name, FeatureIssueType.TEMPORAL_INCONSISTENCY,
                FeatureValidationSeverity.WARNING, False,
                f"temporal : {gap_count} gaps, maximum: {max_gap}",
                impact_score=min(gap_count / len(clean_series), 1.0),
                affected_samples=int(gap_count),
                statistical_details={
                    'max_gap_hours': max_gap.total_seconds() / 3600,
                    'threshold_hours': self.config.temporal_gap_tolerance_hours,
                    'gap_count': int(gap_count)
                },
                recommendations=[
                    "Analyze reasons temporal gaps",
                    "Consider interpolation data",
                    "Verify stability source data"
                ]
            )
    
    def _validate_feature_correlations(self, X: pd.DataFrame) -> None:
        """Validation correlations between features"""
        numeric_features = X.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) < 2:
            return
        
        try:
            corr_matrix = numeric_features.corr()
            
            # Search highly correlated features
            high_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    correlation = abs(corr_matrix.iloc[i, j])
                    
                    if correlation > self.config.high_correlation_threshold:
                        high_correlations.append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'correlation': correlation
                        })
            
            if high_correlations:
                for corr_info in high_correlations:
                    severity = (FeatureValidationSeverity.ERROR if corr_info['correlation'] > 0.99 
                              else FeatureValidationSeverity.WARNING)
                    
                    # Check for potential information leakage (perfect correlation)
                    issue_type = (FeatureIssueType.INFORMATION_LEAKAGE if corr_info['correlation'] > self.config.perfect_correlation_threshold
                                 else FeatureIssueType.HIGH_CORRELATION)
                    
                    feature_pair = f"{corr_info['feature1']} & {corr_info['feature2']}"
                    
                    self._add_result(
                        feature_pair, issue_type, severity, False,
                        f"High correlation: {corr_info['correlation']:.3f}",
                        impact_score=corr_info['correlation'],
                        statistical_details={
                            'correlation': corr_info['correlation'],
                            'threshold': self.config.high_correlation_threshold,
                            'feature1': corr_info['feature1'],
                            'feature2': corr_info['feature2']
                        },
                        recommendations=[
                            "Consider removal one from features",
                            "Apply PCA or dimensionality reduction methods",
                            "If correlation = 1.0, on information leakage"
                        ]
                    )
        
        except Exception as e:
            self.logger.debug(f"Correlation analysis failed: {e}")
    
    def _validate_feature_scaling(self, X: pd.DataFrame) -> None:
        """Validation scaling features"""
        numeric_features = X.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) < 2:
            return
        
        # Calculation scale each feature
        feature_scales = {}
        for column in numeric_features.columns:
            series = numeric_features[column].dropna()
            if len(series) > 0:
                feature_scales[column] = {
                    'mean': abs(series.mean()),
                    'std': series.std(),
                    'range': series.max() - series.min(),
                    'scale': max(abs(series.min()), abs(series.max()))
                }
        
        if len(feature_scales) < 2:
            return
        
        # Search features
        scales = [info['scale'] for info in feature_scales.values()]
        max_scale = max(scales)
 min_scale = min([s for s in scales if s > 0]) # zero scales
        
        if min_scale > 0 and max_scale / min_scale > self.config.scaling_tolerance:
            problematic_features = []
            
            for feature, info in feature_scales.items():
                if info['scale'] == max_scale or info['scale'] == min_scale:
                    problematic_features.append({
                        'feature': feature,
                        'scale': info['scale'],
                        'mean': info['mean'],
                        'std': info['std']
                    })
            
            impact_score = min(np.log10(max_scale / min_scale) / 3, 1.0)  # Normalized log impact
            
            self._add_result(
                "feature_scaling", FeatureIssueType.SCALING_ISSUE,
                FeatureValidationSeverity.WARNING, False,
                f"features: ratio = {max_scale/min_scale:.1f}",
                impact_score=impact_score,
                statistical_details={
                    'max_scale': max_scale,
                    'min_scale': min_scale,
                    'scale_ratio': max_scale / min_scale,
                    'tolerance': self.config.scaling_tolerance,
                    'problematic_features': problematic_features
                },
                recommendations=[
                    "Apply StandardScaler or MinMaxScaler",
                    "Consider RobustScaler for data outliers",
                    "Verify units of measurement for features"
                ]
            )
    
    def _validate_target_relationships(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validation relationships features target variable"""
        if not self.config.check_target_correlation:
            return
        
        numeric_features = X.select_dtypes(include=[np.number])
        
        if len(numeric_features.columns) == 0:
            return
        
        # Calculation correlation target
        target_correlations = []
        
        for column in numeric_features.columns:
            series = numeric_features[column].dropna()
            
            # Align series and target
            common_index = series.index.intersection(y.dropna().index)
            if len(common_index) < 10:  # Insufficient data
                continue
            
            aligned_series = series[common_index]
            aligned_target = y[common_index]
            
            try:
                correlation = aligned_series.corr(aligned_target)
                if not np.isnan(correlation):
                    target_correlations.append({
                        'feature': column,
                        'correlation': abs(correlation),
                        'raw_correlation': correlation
                    })
            except Exception as e:
                self.logger.debug(f"Target correlation calculation failed for {column}: {e}")
        
        # Verification features target
        for corr_info in target_correlations:
            if corr_info['correlation'] < self.config.min_target_correlation:
                self._add_result(
                    corr_info['feature'], FeatureIssueType.LOW_VARIANCE,
                    FeatureValidationSeverity.INFO, False,
 f"Low correlation target: {corr_info['raw_correlation']:.4f}",
                    impact_score=1.0 - corr_info['correlation'],  # Lower correlation = higher impact
                    statistical_details={
                        'target_correlation': corr_info['raw_correlation'],
                        'abs_correlation': corr_info['correlation'],
                        'min_threshold': self.config.min_target_correlation
                    },
                    recommendations=[
                        "Consider removal feature predictive power",
                        "feature engineering or",
                        "Verify target"
                    ]
                )
        
        # Verification on potential information leakage ( high correlation)
        for corr_info in target_correlations:
            if corr_info['correlation'] > self.config.perfect_correlation_threshold:
                self._add_result(
                    corr_info['feature'], FeatureIssueType.INFORMATION_LEAKAGE,
                    FeatureValidationSeverity.CRITICAL, False,
 f" on information leakage: correlation target = {corr_info['raw_correlation']:.4f}",
                    impact_score=corr_info['correlation'],
                    statistical_details={
                        'target_correlation': corr_info['raw_correlation'],
                        'leakage_threshold': self.config.perfect_correlation_threshold
                    },
                    recommendations=[
                        ": Verify feature on information leakage",
                        "Ensure that feature on prediction",
                        "Remove feature if leakage"
                    ]
                )
    
    def _detect_feature_drift(self, X: pd.DataFrame, dataset_name: str) -> None:
        """Detected feature drift baseline"""
        if dataset_name not in self.baseline_profiles:
 return # baseline for comparison
        
        for feature_name in X.columns:
            if feature_name not in self.feature_profiles or feature_name not in self.baseline_profiles:
                continue
            
            current_profile = self.feature_profiles[feature_name]
            baseline_profile = self.baseline_profiles[dataset_name].get(feature_name)
            
            if not baseline_profile:
                continue
            
            # Comparison baseline
            changes = current_profile.compare_with_baseline(baseline_profile)
            
            significant_changes = []
            for metric, change in changes.items():
                if change > self.config.drift_threshold:
                    significant_changes.append({
                        'metric': metric,
                        'change': change,
                        'threshold': self.config.drift_threshold
                    })
            
            if significant_changes:
                max_change = max(change['change'] for change in significant_changes)
                severity = (FeatureValidationSeverity.ERROR if max_change > 0.5 
                           else FeatureValidationSeverity.WARNING)
                
                self._add_result(
                    feature_name, FeatureIssueType.FEATURE_DRIFT, severity, False,
                    f"Feature drift detected: {len(significant_changes)} metrics",
                    impact_score=min(max_change, 1.0),
                    statistical_details={
                        'significant_changes': significant_changes,
                        'max_change': max_change,
                        'drift_threshold': self.config.drift_threshold
                    },
                    recommendations=[
                        "Analyze reasons feature drift",
                        "Consider overfitting models",
                        "Verify stability feature engineering pipeline",
                        "Update baseline if drift expected"
                    ]
                )
    
    def _validate_trading_features(self, X: pd.DataFrame) -> None:
        """Crypto trading specific feature validation"""
        # Price feature validation
        price_features = [col for col in X.columns if 
                         any(keyword in col.lower() for keyword in self.config.price_feature_keywords)]
        
        for feature_name in price_features:
            series = X[feature_name].dropna()
            
            if len(series) == 0:
                continue
            
            # Verification on negative prices
            negative_prices = (series < 0).sum()
            if negative_prices > 0:
                self._add_result(
                    feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY,
                    FeatureValidationSeverity.ERROR, False,
                    f"prices detected: {negative_prices} values",
                    impact_score=1.0,
                    affected_samples=int(negative_prices),
                    recommendations=[
                        "Remove records negative",
                        "Verify correctness processing price data",
                        "Analyze source price feeds"
                    ]
                )
            
            # Verification on zero prices
            zero_prices = (series == 0).sum()
            if zero_prices > 0:
                zero_percentage = (zero_prices / len(series)) * 100
                severity = (FeatureValidationSeverity.ERROR if zero_percentage > 1 
                           else FeatureValidationSeverity.WARNING)
                
                self._add_result(
                    feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY, severity, False,
                    f"prices detected: {zero_prices} values ({zero_percentage:.1f}%)",
                    impact_score=min(zero_percentage / 100, 1.0),
                    affected_samples=int(zero_prices),
                    recommendations=[
                        "Investigate reasons zero prices",
                        "Consider forward/backward fill for short-term gaps",
                        "Verify connectivity market data feeds"
                    ]
                )
        
        # Volume feature validation
        volume_features = [col for col in X.columns if 
                          any(keyword in col.lower() for keyword in self.config.volume_feature_keywords)]
        
        for feature_name in volume_features:
            series = X[feature_name].dropna()
            
            if len(series) == 0:
                continue
            
            # Verification on negative volumes
            negative_volumes = (series < 0).sum()
            if negative_volumes > 0:
                self._add_result(
                    feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY,
                    FeatureValidationSeverity.ERROR, False,
                    f"volumes detected: {negative_volumes} values",
                    impact_score=1.0,
                    affected_samples=int(negative_volumes),
                    recommendations=[
                        "Remove records negative",
                        "Verify correctness processing buy/sell orders"
                    ]
                )
        
        # Technical indicator validation
        tech_features = [col for col in X.columns if 
                        any(keyword in col.lower() for keyword in self.config.technical_indicator_keywords)]
        
        for feature_name in tech_features:
            series = X[feature_name].dropna()
            
            if len(series) == 0:
                continue
            
            # Verification ranges for specific indicators
            if 'rsi' in feature_name.lower():
                out_of_range = ((series < 0) | (series > 100)).sum()
                if out_of_range > 0:
                    self._add_result(
                        feature_name, FeatureIssueType.DISTRIBUTION_ANOMALY,
                        FeatureValidationSeverity.ERROR, False,
                        f"RSI range [0, 100]: {out_of_range} values",
                        impact_score=min(out_of_range / len(series), 1.0),
                        affected_samples=int(out_of_range),
                        recommendations=[
                            "Verify correctness RSI",
                            "RSI should be range [0, 100]"
                        ]
                    )
    
    def _advanced_feature_analysis(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """Advanced enterprise feature analysis"""
        try:
            # Feature importance estimation (if is target sklearn)
            if y is not None and self.sklearn_available and len(X.select_dtypes(include=[np.number]).columns) > 0:
                self._estimate_feature_importance(X, y)
            
            # Multicollinearity analysis
            if self.sklearn_available:
                self._analyze_multicollinearity(X)
            
        except Exception as e:
            self.logger.debug(f"Advanced analysis failed: {e}")
    
    def _estimate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """importance features using Random Forest"""
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Preparation data
            numeric_features = X.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) < 2:
                return
            
            # Detection tasks
            if y.nunique() <= 10 and y.dtype in ['int64', 'int32', 'bool', 'object']:
                # Classification
                model = RandomForestClassifier(n_estimators=50, random_state=42)
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(y.astype(str))
                else:
                    y_encoded = y
            else:
                # Regression
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                y_encoded = y
            
            # Training models
            X_clean = numeric_features.fillna(numeric_features.mean())
            common_index = X_clean.index.intersection(y.dropna().index)
            
            if len(common_index) < 10:
                return
            
            X_aligned = X_clean.loc[common_index]
            y_aligned = y_encoded[common_index] if hasattr(y_encoded, 'index') else y_encoded
            
            model.fit(X_aligned, y_aligned)
            
            # Analysis feature importance
            feature_importance = model.feature_importances_
            feature_names = X_aligned.columns
            
            # Search features very importance
            low_importance_threshold = 0.01  # 1% from total importance
            
            for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                if importance < low_importance_threshold:
                    self._add_result(
                        feature, FeatureIssueType.LOW_VARIANCE,
                        FeatureValidationSeverity.INFO, False,
                        f"Low importance feature: {importance:.4f}",
                        impact_score=1.0 - importance * 10,  # Scale to [0,1]
                        statistical_details={
                            'feature_importance': importance,
                            'importance_threshold': low_importance_threshold,
                            'model_type': type(model).__name__
                        },
                        recommendations=[
                            "Consider removal low-importance feature",
                            "feature engineering for importance",
                            "Verify correlation other features"
                        ]
                    )
        
        except ImportError:
            self.logger.debug("scikit-learn unavailable for feature importance analysis")
        except Exception as e:
            self.logger.debug(f"Feature importance estimation failed: {e}")
    
    def _analyze_multicollinearity(self, X: pd.DataFrame) -> None:
        """Analysis multicollinearity using VIF"""
        try:
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            numeric_features = X.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) < 3:  # Insufficient features for VIF
                return
            
            # Preparation data
            X_clean = numeric_features.fillna(numeric_features.mean())
            
            if len(X_clean) < 10:
                return
            
            # Standardization
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
            
            # Simplified VIF calculation (correlation other features)
            correlations = np.corrcoef(X_scaled.T)
            
            high_multicollinearity_features = []
            
            for i, feature in enumerate(numeric_features.columns):
                # correlation other features
                feature_correlations = np.abs(correlations[i])
 feature_correlations[i] = 0 # correlation itself itself
                max_correlation = np.max(feature_correlations)
                
                if max_correlation > 0.9:  # High multicollinearity
                    correlated_feature_idx = np.argmax(feature_correlations)
                    correlated_feature = numeric_features.columns[correlated_feature_idx]
                    
                    high_multicollinearity_features.append({
                        'feature': feature,
                        'correlated_with': correlated_feature,
                        'correlation': max_correlation
                    })
            
            for multi_info in high_multicollinearity_features:
                self._add_result(
                    multi_info['feature'], FeatureIssueType.HIGH_CORRELATION,
                    FeatureValidationSeverity.WARNING, False,
 f"High multicollinearity {multi_info['correlated_with']}: {multi_info['correlation']:.3f}",
                    impact_score=multi_info['correlation'],
                    statistical_details={
                        'correlation': multi_info['correlation'],
                        'correlated_feature': multi_info['correlated_with'],
                        'multicollinearity_threshold': 0.9
                    },
                    recommendations=[
                        "Consider removal one from features",
                        "Apply Ridge regression or regularized methods",
                        "Use PCA for dimensionality reduction"
                    ]
                )
        
        except Exception as e:
            self.logger.debug(f"Multicollinearity analysis failed: {e}")
    
    def _add_result(
        self,
        feature_name: str,
        issue_type: FeatureIssueType,
        severity: FeatureValidationSeverity,
        passed: bool,
        message: str,
        impact_score: float = 0.0,
        affected_samples: int = 0,
        statistical_details: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ) -> None:
        """Addition result validation"""
        self.results.append(FeatureValidationResult(
            feature_name=feature_name,
            issue_type=issue_type,
            severity=severity,
            passed=passed,
            message=message,
            impact_score=impact_score,
            affected_samples=affected_samples,
            statistical_details=statistical_details or {},
            recommendations=recommendations or []
        ))
    
    def _generate_validation_summary(self, X: pd.DataFrame, dataset_name: str) -> None:
        """Generation one result feature validation"""
        total_features = len(X.columns)
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        # Counting by severity levels
        critical_issues = sum(1 for r in self.results if not r.passed and r.severity == FeatureValidationSeverity.CRITICAL)
        error_issues = sum(1 for r in self.results if not r.passed and r.severity == FeatureValidationSeverity.ERROR)
        warning_issues = sum(1 for r in self.results if not r.passed and r.severity == FeatureValidationSeverity.WARNING)
        
        # Overall feature quality
        if critical_issues > 0:
            overall_quality = "CRITICAL"
            quality_severity = FeatureValidationSeverity.CRITICAL
        elif error_issues > 0:
            overall_quality = "POOR"
            quality_severity = FeatureValidationSeverity.ERROR
        elif warning_issues > 0:
            overall_quality = "ACCEPTABLE"
            quality_severity = FeatureValidationSeverity.WARNING
        else:
            overall_quality = "EXCELLENT"
            quality_severity = FeatureValidationSeverity.INFO
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 100
        
        # Average impact score
        impact_scores = [r.impact_score for r in self.results if not r.passed]
        avg_impact = np.mean(impact_scores) if impact_scores else 0.0
        
        summary_message = (f"Feature validation for {dataset_name} - Quality: {overall_quality} "
                          f"({passed_checks}/{total_checks} checks , {success_rate:.1f}%)")
        
        self._add_result(
            "feature_validation_summary", FeatureIssueType.DISTRIBUTION_ANOMALY,
            quality_severity, (critical_issues == 0 and error_issues == 0),
            summary_message,
            impact_score=avg_impact,
            statistical_details={
                'total_features': total_features,
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'failed_checks': failed_checks,
                'success_rate': success_rate,
                'critical_issues': critical_issues,
                'error_issues': error_issues,
                'warning_issues': warning_issues,
                'average_impact_score': avg_impact,
                'overall_quality': overall_quality
            }
        )
    
    def generate_feature_report(self, include_profiles: bool = False) -> Dict[str, Any]:
        """Generation comprehensive feature validation report"""
        if not self.results:
            return {'status': 'no_validation_run'}
        
        # Grouping results by features
        results_by_feature = {}
        for result in self.results:
            if result.feature_name not in results_by_feature:
                results_by_feature[result.feature_name] = []
            results_by_feature[result.feature_name].append(result)
        
        # Severity statistics
        severity_stats = {}
        for severity in FeatureValidationSeverity:
            severity_results = [r for r in self.results if r.severity == severity]
            severity_stats[severity.value] = {
                'total': len(severity_results),
                'passed': sum(1 for r in severity_results if r.passed),
                'failed': sum(1 for r in severity_results if not r.passed)
            }
        
        # Issue type statistics
        issue_stats = {}
        for issue_type in FeatureIssueType:
            issue_results = [r for r in self.results if r.issue_type == issue_type]
            issue_stats[issue_type.value] = len(issue_results)
        
        # Top problematic features
        feature_scores = {}
        for feature, results in results_by_feature.items():
            if feature == "feature_validation_summary":
                continue
            
            failed_results = [r for r in results if not r.passed]
            if failed_results:
                # Weighted score on severity impact
                severity_weights = {
                    FeatureValidationSeverity.CRITICAL: 4,
                    FeatureValidationSeverity.ERROR: 3,
                    FeatureValidationSeverity.WARNING: 2,
                    FeatureValidationSeverity.INFO: 1
                }
                
                total_score = sum(
                    severity_weights[r.severity] * (1 + r.impact_score)
                    for r in failed_results
                )
                feature_scores[feature] = total_score
        
        # Top 10 most problematic features
        top_problematic = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_features': len(results_by_feature) - 1,  # Exclude summary
                'total_checks': len(self.results),
                'passed_checks': sum(1 for r in self.results if r.passed),
                'failed_checks': sum(1 for r in self.results if not r.passed),
                'success_rate': (sum(1 for r in self.results if r.passed) / len(self.results)) * 100,
                'overall_status': 'PASS' if all(r.passed for r in self.results if r.feature_name != "feature_validation_summary") else 'FAIL'
            },
            'severity_breakdown': severity_stats,
            'issue_type_breakdown': issue_stats,
            'top_problematic_features': [
                {
                    'feature': feature,
                    'problem_score': score,
                    'issues': [r.issue_type.value for r in results_by_feature[feature] if not r.passed]
                }
                for feature, score in top_problematic
            ],
            'detailed_results': {
                feature: [
                    {
                        'issue_type': r.issue_type.value,
                        'severity': r.severity.value,
                        'passed': r.passed,
                        'message': r.message,
                        'impact_score': r.impact_score,
                        'affected_samples': r.affected_samples,
                        'statistical_details': r.statistical_details,
                        'recommendations': r.recommendations,
                        'timestamp': r.timestamp.isoformat()
                    }
                    for r in results
                ]
                for feature, results in results_by_feature.items()
                if feature != "feature_validation_summary"
            }
        }
        
        if include_profiles and self.feature_profiles:
            report['feature_profiles'] = {
                name: {
                    'statistics': profile.statistics,
                    'distribution_info': profile.distribution_info,
                    'quality_metrics': profile.quality_metrics,
                    'created_at': profile.created_at.isoformat()
                }
                for name, profile in self.feature_profiles.items()
            }
        
        return report
    
    def save_baseline_profiles(self, dataset_name: str) -> None:
        """Save current profiles as baseline for future comparison"""
        self.baseline_profiles[dataset_name] = self.feature_profiles.copy()
        self.logger.info(f"Baseline profiles saved for {dataset_name}")
    
    def load_baseline_profiles(self, filepath: str) -> None:
        """Load baseline profiles from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Recovery baseline profiles
            for dataset_name, profiles_data in data.items():
                self.baseline_profiles[dataset_name] = {}
                
                for feature_name, profile_data in profiles_data.items():
                    profile = FeatureProfile(feature_name)
                    profile.statistics = profile_data['statistics']
                    profile.distribution_info = profile_data['distribution_info']
                    profile.quality_metrics = profile_data['quality_metrics']
                    profile.created_at = datetime.fromisoformat(profile_data['created_at'])
                    self.baseline_profiles[dataset_name][feature_name] = profile
            
            self.logger.info(f"Baseline profiles from {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error at loading baseline profiles: {e}")
    
    def save_baseline_profiles_to_file(self, filepath: str) -> None:
        """Save baseline profiles file"""
        try:
            data = {}
            
            for dataset_name, profiles in self.baseline_profiles.items():
                data[dataset_name] = {}
                
                for feature_name, profile in profiles.items():
                    data[dataset_name][feature_name] = {
                        'statistics': profile.statistics,
                        'distribution_info': profile.distribution_info,
                        'quality_metrics': profile.quality_metrics,
                        'created_at': profile.created_at.isoformat()
                    }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Baseline profiles saved {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error at saving baseline profiles: {e}")
    
    def export_report(self, filepath: str, include_profiles: bool = False) -> None:
        """Export feature validation report file"""
        report = self.generate_feature_report(include_profiles)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Feature validation report exported {filepath}")


def create_crypto_trading_feature_validator() -> FeatureValidator:
    """
    Factory function for creation feature validator for crypto trading features
    Enterprise pre-configured validator for financial ML features
    """
    config = FeatureValidationConfig(
        # Strict requirements for crypto trading features
        constant_threshold=0.98,  # 98% threshold for quasi-constant
        min_unique_values=3,  # Minimum 3 unique values
        
        # Correlation thresholds
        high_correlation_threshold=0.90,  # 90% for crypto volatility
        check_target_correlation=True,
        min_target_correlation=0.005,  # 0.5% minimum for crypto signals
        
        # Variance requirements
 min_variance_threshold=1e-8, # Very for crypto precision
        normalize_variance=True,
        
        # Missing values tolerance
        max_missing_percentage=1.0,  # 1% maximum for trading data
        critical_missing_threshold=5.0,  # 5% critical level
        
        # Outlier detection for volatile markets
        outlier_method="iqr",
 outlier_threshold=2.0, # strict for crypto volatility
        max_outlier_percentage=15.0,  # 15% outliers normal for crypto
        
        # Distribution tolerance
 check_normality=False, # Crypto data
        check_skewness=True,
 max_skewness=8.0, # High skewness for crypto
 max_kurtosis=20.0, # High for crypto
        
        # Feature drift detection
        enable_drift_detection=True,
        drift_threshold=0.05,  # 5% threshold for trading features
        
        # Information leakage protection
        check_information_leakage=True,
 perfect_correlation_threshold=0.995, # Very
        
        # Scaling validation
        check_feature_scaling=True,
 scaling_tolerance=1000.0, # tolerance for crypto ranges
        
        # Temporal consistency
        check_temporal_consistency=True,
 temporal_gap_tolerance_hours=1.0, # 1 for crypto data
        
        # Advanced analysis
        enable_advanced_analysis=True,
        generate_feature_profiles=True,
        save_validation_history=True,
        
        # Crypto trading specific
        validate_trading_features=True,
        price_feature_keywords=["price", "close", "open", "high", "low", "last", "ask", "bid", "vwap"],
        volume_feature_keywords=["volume", "vol", "quantity", "amount", "size", "notional"],
        technical_indicator_keywords=[
            "rsi", "macd", "sma", "ema", "bollinger", "stoch", "atr", "adx",
            "obv", "mfi", "cci", "williams", "momentum", "roc", "trix"
        ]
    )
    
    return FeatureValidator(config)