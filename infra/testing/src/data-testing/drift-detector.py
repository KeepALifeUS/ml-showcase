"""
Drift Detector - Enterprise Enterprise Data & Model Drift Detection
Comprehensive drift detection framework for ML systems crypto trading

Applies enterprise principles:
- Enterprise drift monitoring
- Production-ready alerting
- Statistical drift detection
- Automated model retraining triggers
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
import hashlib

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types drift for Enterprise monitoring"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"
    TARGET_DRIFT = "target_drift"
    COVARIATE_SHIFT = "covariate_shift"
    PRIOR_PROBABILITY_SHIFT = "prior_probability_shift"


class DriftSeverity(Enum):
    """Levels severity drift for Enterprise alerting"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetectionMethod(Enum):
    """Method detected drift for Enterprise statistical testing"""
    KOLMOGOROV_SMIRNOV = "ks_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    JENSEN_SHANNON = "jensen_shannon"
    WASSERSTEIN = "wasserstein"
    POPULATION_STABILITY_INDEX = "psi"
    STATISTICAL_DISTANCE = "statistical_distance"
    ADVERSARIAL_DRIFT = "adversarial_drift"


@dataclass
class DriftDetectionResult:
    """Result detected drift - Enterprise structured result"""
    drift_type: DriftType
    severity: DriftSeverity
    detected: bool
    method: DriftDetectionMethod
    feature_name: Optional[str]
    drift_score: float
    p_value: Optional[float]
    threshold: float
    message: str
    statistical_details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    affected_samples: int = 0
    
    def __str__(self) -> str:
        status = "DETECTED" if self.detected else "NOT_DETECTED"
        return f"[{status}] {self.drift_type.value}: {self.message}"


@dataclass
class DriftDetectionConfig:
    """Configuration drift detection - Enterprise typed configuration"""
    # Statistical test thresholds
    p_value_threshold: float = 0.05  # Significance level
    drift_score_threshold: float = 0.1  # General drift score threshold
    
    # Population Stability Index thresholds
    psi_threshold_low: float = 0.1  # No significant drift
    psi_threshold_medium: float = 0.2  # Medium drift
    psi_threshold_high: float = 0.25  # High drift
    
    # Statistical distance thresholds
    ks_threshold: float = 0.05  # Kolmogorov-Smirnov p-value
    wasserstein_threshold: float = 0.1  # Wasserstein distance
    jensen_shannon_threshold: float = 0.1  # JS divergence
    
    # Concept drift detection
    concept_drift_window_size: int = 100  # Sliding window size
    concept_drift_threshold: float = 0.05  # Performance degradation threshold
    
    # Feature drift settings
    feature_drift_methods: List[DriftDetectionMethod] = field(default_factory=lambda: [
        DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
        DriftDetectionMethod.POPULATION_STABILITY_INDEX
    ])
    
    # Prediction drift settings
    prediction_drift_threshold: float = 0.1  # 10% change in predictions
    prediction_drift_methods: List[DriftDetectionMethod] = field(default_factory=lambda: [
        DriftDetectionMethod.WASSERSTEIN,
        DriftDetectionMethod.JENSEN_SHANNON
    ])
    
    # Performance monitoring
    performance_degradation_threshold: float = 0.05  # 5% performance drop
    minimum_samples_for_detection: int = 100  # Minimum samples for reliable detection
    
    # Time-based drift detection
    enable_temporal_drift: bool = True
    temporal_window_hours: float = 24.0  # 24-hour windows
    
    # Enterprise enterprise settings
    enable_advanced_methods: bool = True
    save_drift_history: bool = True
    auto_generate_alerts: bool = True
    
    # Crypto trading specific
    crypto_volatility_adjustment: bool = True  # Adjust thresholds for crypto volatility
    price_drift_sensitivity: float = 0.02  # 2% price drift sensitivity
    volume_drift_sensitivity: float = 0.05  # 5% volume drift sensitivity


@dataclass
class BaselineDistribution:
    """Baseline distribution for Enterprise drift comparison"""
    feature_name: str
    data_hash: str
    sample_size: int
    statistics: Dict[str, float]
    histogram_bins: Optional[np.ndarray] = None
    histogram_values: Optional[np.ndarray] = None
    percentiles: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialization baseline for saving"""
        return {
            'feature_name': self.feature_name,
            'data_hash': self.data_hash,
            'sample_size': self.sample_size,
            'statistics': self.statistics,
            'histogram_bins': self.histogram_bins.tolist() if self.histogram_bins is not None else None,
            'histogram_values': self.histogram_values.tolist() if self.histogram_values is not None else None,
            'percentiles': self.percentiles.tolist() if self.percentiles is not None else None,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineDistribution':
        """serialization baseline from"""
        baseline = cls(
            feature_name=data['feature_name'],
            data_hash=data['data_hash'],
            sample_size=data['sample_size'],
            statistics=data['statistics'],
            created_at=datetime.fromisoformat(data['created_at']),
            metadata=data.get('metadata', {})
        )
        
        if data.get('histogram_bins'):
            baseline.histogram_bins = np.array(data['histogram_bins'])
        if data.get('histogram_values'):
            baseline.histogram_values = np.array(data['histogram_values'])
        if data.get('percentiles'):
            baseline.percentiles = np.array(data['percentiles'])
        
        return baseline


class DriftDetector:
    """
    Enterprise Enterprise Drift Detector
    
 Comprehensive drift detection framework for ML systems crypto trading.
 Provides enterprise-grade monitoring automated retraining triggers.
    """
    
    def __init__(self, config: Optional[DriftDetectionConfig] = None):
        """
 Initialization drift detector Enterprise configuration
        
        Args:
            config: Configuration drift detection (Enterprise typed)
        """
        self.config = config or DriftDetectionConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Enterprise baseline storage
        self.baseline_distributions: Dict[str, BaselineDistribution] = {}
        self.drift_history: List[DriftDetectionResult] = []
        
        # Statistical methods initialization
        self._init_statistical_methods()
        
        # Performance tracking for concept drift
        self.performance_history: List[Dict[str, Any]] = []
    
    def _init_statistical_methods(self) -> None:
        """Initialization statistical method"""
        try:
            from scipy import stats
            from scipy.spatial.distance import wasserstein_distance
            self.stats_available = True
            self.scipy_methods = {
                'ks_2samp': stats.ks_2samp,
                'mannwhitneyu': stats.mannwhitneyu,
                'chi2_contingency': stats.chi2_contingency,
                'wasserstein_distance': wasserstein_distance
            }
        except ImportError:
            self.stats_available = False
            self.scipy_methods = {}
            self.logger.warning("SciPy unavailable - methods drift detection will be skipped")
        
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.model_selection import cross_val_score
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            self.logger.warning("scikit-learn unavailable - advanced methods will be skipped")
    
    def create_baseline(
        self,
        data: Union[pd.DataFrame, pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None,
        baseline_name: str = "default"
    ) -> Dict[str, BaselineDistribution]:
        """
        Create baseline distributions for future drift detection
        
        Args:
            data: Baseline data
 feature_names: features (if data = DataFrame)
 baseline_name: Name baseline for
        
        Returns:
 Dict[str, BaselineDistribution]: data baselines by features
        """
        self.logger.info(f"Create baseline distributions: {baseline_name}")
        
        baselines = {}
        
        if isinstance(data, pd.DataFrame):
            features_to_process = feature_names or data.columns.tolist()
            
            for feature_name in features_to_process:
                if feature_name not in data.columns:
                    continue
                
                series = data[feature_name].dropna()
                baseline = self._create_feature_baseline(feature_name, series)
                baselines[feature_name] = baseline
                self.baseline_distributions[f"{baseline_name}_{feature_name}"] = baseline
        
        elif isinstance(data, pd.Series):
            feature_name = feature_names[0] if feature_names else "feature"
            clean_series = data.dropna()
            baseline = self._create_feature_baseline(feature_name, clean_series)
            baselines[feature_name] = baseline
            self.baseline_distributions[f"{baseline_name}_{feature_name}"] = baseline
        
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                # 1D array
                feature_name = feature_names[0] if feature_names else "feature"
                clean_data = data[~np.isnan(data)]
                series = pd.Series(clean_data, name=feature_name)
                baseline = self._create_feature_baseline(feature_name, series)
                baselines[feature_name] = baseline
                self.baseline_distributions[f"{baseline_name}_{feature_name}"] = baseline
            
            elif len(data.shape) == 2:
                # 2D array
                features_to_process = feature_names or [f"feature_{i}" for i in range(data.shape[1])]
                
                for i, feature_name in enumerate(features_to_process):
                    if i >= data.shape[1]:
                        break
                    
                    feature_data = data[:, i]
                    clean_data = feature_data[~np.isnan(feature_data)]
                    series = pd.Series(clean_data, name=feature_name)
                    baseline = self._create_feature_baseline(feature_name, series)
                    baselines[feature_name] = baseline
                    self.baseline_distributions[f"{baseline_name}_{feature_name}"] = baseline
        
        self.logger.info(f"{len(baselines)} baseline distributions")
        return baselines
    
    def _create_feature_baseline(self, feature_name: str, series: pd.Series) -> BaselineDistribution:
        """Create baseline for individual feature"""
        # Data hash for
        data_hash = hashlib.md5(series.values.tobytes()).hexdigest()
        
        # Basic statistics
        statistics = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'median': float(series.median()),
            'min': float(series.min()),
            'max': float(series.max()),
            'q1': float(series.quantile(0.25)),
            'q3': float(series.quantile(0.75)),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'count': len(series)
        }
        
        # Histogram for distribution comparison
        if len(series) > 10:
            hist_values, hist_bins = np.histogram(series, bins=min(50, len(series) // 10))
        else:
            hist_values, hist_bins = None, None
        
        # Percentiles for detailed comparison
        percentiles = np.percentile(series, np.linspace(0, 100, 101)) if len(series) > 0 else None
        
        baseline = BaselineDistribution(
            feature_name=feature_name,
            data_hash=data_hash,
            sample_size=len(series),
            statistics=statistics,
            histogram_bins=hist_bins,
            histogram_values=hist_values,
            percentiles=percentiles,
            metadata={
                'data_type': str(series.dtype),
                'unique_values': series.nunique(),
                'null_count': series.isnull().sum()
            }
        )
        
        return baseline
    
    def detect_drift(
        self,
        current_data: Union[pd.DataFrame, pd.Series, np.ndarray],
        baseline_name: str = "default",
        feature_names: Optional[List[str]] = None,
        methods: Optional[List[DriftDetectionMethod]] = None
    ) -> List[DriftDetectionResult]:
        """
        Comprehensive drift detection against baseline
        
        Args:
 current_data: data for verification drift
            baseline_name: Name baseline for comparison
 feature_names: features (if data = DataFrame)
 methods: Method drift detection for usage
        
        Returns:
            List[DriftDetectionResult]: Results drift detection
        """
        self.logger.info(f"Starting drift detection against baseline: {baseline_name}")
        
        results = []
        methods_to_use = methods or self.config.feature_drift_methods
        
        if isinstance(current_data, pd.DataFrame):
            features_to_check = feature_names or current_data.columns.tolist()
            
            for feature_name in features_to_check:
                if feature_name not in current_data.columns:
                    continue
                
                baseline_key = f"{baseline_name}_{feature_name}"
                if baseline_key not in self.baseline_distributions:
                    self.logger.warning(f"Baseline not for {baseline_key}")
                    continue
                
                series = current_data[feature_name].dropna()
                baseline = self.baseline_distributions[baseline_key]
                
                # Verification size sample
                if len(series) < self.config.minimum_samples_for_detection:
                    results.append(DriftDetectionResult(
                        drift_type=DriftType.DATA_DRIFT,
                        severity=DriftSeverity.LOW,
                        detected=False,
                        method=DriftDetectionMethod.STATISTICAL_DISTANCE,
                        feature_name=feature_name,
                        drift_score=0.0,
                        p_value=None,
                        threshold=0.0,
                        message=f"Insufficient data for drift detection: {len(series)} < {self.config.minimum_samples_for_detection}",
                        affected_samples=len(series)
                    ))
                    continue
                
                # all method drift detection
                for method in methods_to_use:
                    drift_result = self._detect_feature_drift(series, baseline, method)
                    if drift_result:
                        results.append(drift_result)
        
        elif isinstance(current_data, pd.Series):
            feature_name = feature_names[0] if feature_names else "feature"
            baseline_key = f"{baseline_name}_{feature_name}"
            
            if baseline_key in self.baseline_distributions:
                clean_series = current_data.dropna()
                baseline = self.baseline_distributions[baseline_key]
                
                for method in methods_to_use:
                    drift_result = self._detect_feature_drift(clean_series, baseline, method)
                    if drift_result:
                        results.append(drift_result)
        
        # Save results history
        self.drift_history.extend(results)
        
        # Generation drift
        overall_result = self._generate_overall_drift_assessment(results, baseline_name)
        if overall_result:
            results.append(overall_result)
        
        self.logger.info(f"Drift detection completed. {len(results)} results")
        return results
    
    def _detect_feature_drift(
        self,
        current_series: pd.Series,
        baseline: BaselineDistribution,
        method: DriftDetectionMethod
    ) -> Optional[DriftDetectionResult]:
        """Detected drift for individual feature using specific method"""
        try:
            if method == DriftDetectionMethod.KOLMOGOROV_SMIRNOV:
                return self._ks_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.MANN_WHITNEY_U:
                return self._mann_whitney_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.POPULATION_STABILITY_INDEX:
                return self._psi_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.WASSERSTEIN:
                return self._wasserstein_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.JENSEN_SHANNON:
                return self._jensen_shannon_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.CHI_SQUARE:
                return self._chi_square_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.STATISTICAL_DISTANCE:
                return self._statistical_distance_drift_test(current_series, baseline)
            
            elif method == DriftDetectionMethod.ADVERSARIAL_DRIFT and self.sklearn_available:
                return self._adversarial_drift_test(current_series, baseline)
            
        except Exception as e:
            self.logger.error(f"Error drift detection method {method.value}: {e}")
            return None
        
        return None
    
    def _ks_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Kolmogorov-Smirnov drift test"""
        if not self.stats_available:
            return self._create_error_result("KS test unavailable - SciPy not installed", baseline.feature_name)
        
        # Generation baseline sample for comparison
        baseline_sample = np.random.normal(
            baseline.statistics['mean'],
            baseline.statistics['std'],
            size=len(current_series)
        )
        
        # KS test
        ks_statistic, p_value = self.scipy_methods['ks_2samp'](current_series.values, baseline_sample)
        
        # Detection drift
        drift_detected = p_value < self.config.ks_threshold
        
        # Severity assessment
        if p_value < 0.001:
            severity = DriftSeverity.CRITICAL
        elif p_value < 0.01:
            severity = DriftSeverity.HIGH
        elif p_value < 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        recommendations = []
        if drift_detected:
            recommendations.extend([
                "Analyze changes source data",
                "Consider overfitting models",
                "Verify data preprocessing pipeline"
            ])
        
        return DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            detected=drift_detected,
            method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            feature_name=baseline.feature_name,
            drift_score=ks_statistic,
            p_value=p_value,
            threshold=self.config.ks_threshold,
            message=f"KS test: statistic={ks_statistic:.4f}, p-value={p_value:.4f}",
            statistical_details={
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'baseline_mean': baseline.statistics['mean'],
                'current_mean': float(current_series.mean()),
                'baseline_std': baseline.statistics['std'],
                'current_std': float(current_series.std())
            },
            recommendations=recommendations,
            affected_samples=len(current_series)
        )
    
    def _mann_whitney_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Mann-Whitney U drift test"""
        if not self.stats_available:
            return self._create_error_result("Mann-Whitney test unavailable - SciPy not installed", baseline.feature_name)
        
        # Generation baseline sample
        baseline_sample = np.random.normal(
            baseline.statistics['mean'],
            baseline.statistics['std'],
 size=min(len(current_series), 1000) # size for performance
        )
        
        # Mann-Whitney U test
        statistic, p_value = self.scipy_methods['mannwhitneyu'](
            current_series.values, baseline_sample, alternative='two-sided'
        )
        
        drift_detected = p_value < self.config.p_value_threshold
        
        # Severity based on p-value
        if p_value < 0.001:
            severity = DriftSeverity.CRITICAL
        elif p_value < 0.01:
            severity = DriftSeverity.HIGH
        elif p_value < 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        return DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            detected=drift_detected,
            method=DriftDetectionMethod.MANN_WHITNEY_U,
            feature_name=baseline.feature_name,
            drift_score=float(statistic),
            p_value=p_value,
            threshold=self.config.p_value_threshold,
            message=f"Mann-Whitney U test: statistic={statistic}, p-value={p_value:.4f}",
            statistical_details={
                'u_statistic': float(statistic),
                'p_value': p_value,
                'test_type': 'two-sided'
            },
            affected_samples=len(current_series)
        )
    
    def _psi_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Population Stability Index drift test"""
        try:
            # Create on basis baseline percentiles
            if baseline.percentiles is not None and len(baseline.percentiles) > 10:
 bins = np.percentile(baseline.percentiles, np.linspace(0, 100, 11)) # 10
            else:
                # Fallback
                bins = np.linspace(
                    baseline.statistics['min'], 
                    baseline.statistics['max'], 
                    11
                )
            
            # for values
            bins = np.unique(bins)
            if len(bins) < 2:
                return DriftDetectionResult(
                    drift_type=DriftType.DATA_DRIFT,
                    severity=DriftSeverity.LOW,
                    detected=False,
                    method=DriftDetectionMethod.POPULATION_STABILITY_INDEX,
                    feature_name=baseline.feature_name,
                    drift_score=0.0,
                    p_value=None,
                    threshold=self.config.psi_threshold_low,
                    message="PSI: Insufficient unique values for binning",
                    affected_samples=len(current_series)
                )
            
            # Calculation baseline distribution
            baseline_counts, _ = np.histogram(
                np.random.normal(baseline.statistics['mean'], baseline.statistics['std'], 10000),
                bins=bins
            )
            baseline_props = baseline_counts / baseline_counts.sum()
            baseline_props = np.clip(baseline_props, 1e-10, 1.0)  # Avoid zero probabilities
            
            # Calculation current distribution
            current_counts, _ = np.histogram(current_series.values, bins=bins)
            current_props = current_counts / current_counts.sum()
            current_props = np.clip(current_props, 1e-10, 1.0)  # Avoid zero probabilities
            
            # PSI calculation
            psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))
            
            # Drift assessment
            if psi < self.config.psi_threshold_low:
                drift_detected = False
                severity = DriftSeverity.LOW
                message = f"PSI: {psi:.4f} - No significant drift"
            elif psi < self.config.psi_threshold_medium:
                drift_detected = True
                severity = DriftSeverity.MEDIUM
                message = f"PSI: {psi:.4f} - Medium drift detected"
            elif psi < self.config.psi_threshold_high:
                drift_detected = True
                severity = DriftSeverity.HIGH
                message = f"PSI: {psi:.4f} - High drift detected"
            else:
                drift_detected = True
                severity = DriftSeverity.CRITICAL
                message = f"PSI: {psi:.4f} - Critical drift detected"
            
            recommendations = []
            if drift_detected:
                recommendations.extend([
                    "PSI change distribution",
                    "Verify changes data collection process",
                    "Consider baseline or overfitting models"
                ])
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.POPULATION_STABILITY_INDEX,
                feature_name=baseline.feature_name,
                drift_score=psi,
                p_value=None,
                threshold=self.config.psi_threshold_medium,
                message=message,
                statistical_details={
                    'psi_score': psi,
                    'baseline_props': baseline_props.tolist(),
                    'current_props': current_props.tolist(),
                    'bins': bins.tolist(),
                    'psi_thresholds': {
                        'low': self.config.psi_threshold_low,
                        'medium': self.config.psi_threshold_medium,
                        'high': self.config.psi_threshold_high
                    }
                },
                recommendations=recommendations,
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            self.logger.error(f"PSI calculation failed: {e}")
            return self._create_error_result(f"PSI calculation failed: {str(e)}", baseline.feature_name)
    
    def _wasserstein_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Wasserstein distance drift test"""
        if not self.stats_available or 'wasserstein_distance' not in self.scipy_methods:
            return self._create_error_result("Wasserstein distance unavailable - SciPy not installed", baseline.feature_name)
        
        try:
            # Generation baseline sample
            baseline_sample = np.random.normal(
                baseline.statistics['mean'],
                baseline.statistics['std'],
                size=min(len(current_series), 5000)
            )
            
            # Wasserstein distance calculation
            wasserstein_dist = self.scipy_methods['wasserstein_distance'](
                current_series.values, baseline_sample
            )
            
            # Normalization distance baseline scale
            baseline_range = baseline.statistics['max'] - baseline.statistics['min']
            normalized_distance = wasserstein_dist / baseline_range if baseline_range > 0 else wasserstein_dist
            
            drift_detected = normalized_distance > self.config.wasserstein_threshold
            
            # Severity assessment
            if normalized_distance > self.config.wasserstein_threshold * 4:
                severity = DriftSeverity.CRITICAL
            elif normalized_distance > self.config.wasserstein_threshold * 2:
                severity = DriftSeverity.HIGH
            elif normalized_distance > self.config.wasserstein_threshold:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.WASSERSTEIN,
                feature_name=baseline.feature_name,
                drift_score=normalized_distance,
                p_value=None,
                threshold=self.config.wasserstein_threshold,
                message=f"Wasserstein distance: {wasserstein_dist:.4f} (normalized: {normalized_distance:.4f})",
                statistical_details={
                    'wasserstein_distance': wasserstein_dist,
                    'normalized_distance': normalized_distance,
                    'baseline_range': baseline_range,
                    'baseline_mean': baseline.statistics['mean'],
                    'current_mean': float(current_series.mean())
                },
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            return self._create_error_result(f"Wasserstein distance calculation failed: {str(e)}", baseline.feature_name)
    
    def _jensen_shannon_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Jensen-Shannon divergence drift test"""
        try:
            # Create for comparison
            common_min = min(baseline.statistics['min'], current_series.min())
            common_max = max(baseline.statistics['max'], current_series.max())
            
            bins = np.linspace(common_min, common_max, 50)
            
            # Baseline histogram
            baseline_sample = np.random.normal(
                baseline.statistics['mean'],
                baseline.statistics['std'],
                size=10000
            )
            baseline_hist, _ = np.histogram(baseline_sample, bins=bins, density=True)
            baseline_hist = baseline_hist + 1e-10  # Avoid zero probabilities
            baseline_hist = baseline_hist / baseline_hist.sum()
            
            # Current histogram
            current_hist, _ = np.histogram(current_series.values, bins=bins, density=True)
            current_hist = current_hist + 1e-10  # Avoid zero probabilities
            current_hist = current_hist / current_hist.sum()
            
            # Jensen-Shannon divergence calculation
            m = 0.5 * (baseline_hist + current_hist)
            js_divergence = 0.5 * np.sum(baseline_hist * np.log(baseline_hist / m)) + \
                           0.5 * np.sum(current_hist * np.log(current_hist / m))
            
            # JS distance (square root of divergence)
            js_distance = np.sqrt(js_divergence)
            
            drift_detected = js_distance > self.config.jensen_shannon_threshold
            
            # Severity assessment
            if js_distance > self.config.jensen_shannon_threshold * 3:
                severity = DriftSeverity.CRITICAL
            elif js_distance > self.config.jensen_shannon_threshold * 2:
                severity = DriftSeverity.HIGH
            elif js_distance > self.config.jensen_shannon_threshold:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.JENSEN_SHANNON,
                feature_name=baseline.feature_name,
                drift_score=js_distance,
                p_value=None,
                threshold=self.config.jensen_shannon_threshold,
                message=f"Jensen-Shannon distance: {js_distance:.4f}",
                statistical_details={
                    'js_divergence': js_divergence,
                    'js_distance': js_distance,
                    'bins_count': len(bins) - 1,
                    'baseline_entropy': -np.sum(baseline_hist * np.log(baseline_hist + 1e-10)),
                    'current_entropy': -np.sum(current_hist * np.log(current_hist + 1e-10))
                },
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            return self._create_error_result(f"Jensen-Shannon calculation failed: {str(e)}", baseline.feature_name)
    
    def _chi_square_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Chi-square drift test for categorical features"""
        if not self.stats_available:
            return self._create_error_result("Chi-square test unavailable - SciPy not installed", baseline.feature_name)
        
        try:
            # Verification on categorical data
 if current_series.nunique() > 50: # for chi-square
                return self._create_error_result("Too many unique values for chi-square test", baseline.feature_name)
            
            # Retrieval value counts for current data
            current_counts = current_series.value_counts().sort_index()
            
            # Create expected counts on basis baseline
            # Using from baseline for expected counts
            total_current = len(current_series)
            
            # baseline proportions
            # ( implementation need baseline value counts)
            expected_uniform = total_current / current_counts.nunique()
            expected_counts = np.full(len(current_counts), expected_uniform)
            
            # Chi-square test
            chi2_stat, p_value = stats.chisquare(current_counts.values, expected_counts)
            
            drift_detected = p_value < self.config.p_value_threshold
            
            # Severity assessment
            if p_value < 0.001:
                severity = DriftSeverity.CRITICAL
            elif p_value < 0.01:
                severity = DriftSeverity.HIGH
            elif p_value < 0.05:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.CHI_SQUARE,
                feature_name=baseline.feature_name,
                drift_score=chi2_stat,
                p_value=p_value,
                threshold=self.config.p_value_threshold,
                message=f"Chi-square test: statistic={chi2_stat:.4f}, p-value={p_value:.4f}",
                statistical_details={
                    'chi2_statistic': chi2_stat,
                    'p_value': p_value,
                    'degrees_of_freedom': len(current_counts) - 1,
                    'observed_counts': current_counts.to_dict(),
                    'unique_values': current_series.nunique()
                },
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            return self._create_error_result(f"Chi-square test failed: {str(e)}", baseline.feature_name)
    
    def _statistical_distance_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Statistical distance drift test (composite method)"""
        try:
            # Composite drift score on basis multiple statistics
            current_stats = {
                'mean': float(current_series.mean()),
                'std': float(current_series.std()),
                'median': float(current_series.median()),
                'skewness': float(current_series.skew()),
                'kurtosis': float(current_series.kurtosis())
            }
            
            # Normalized differences
            differences = {}
            composite_score = 0.0
            
            for stat_name, current_value in current_stats.items():
                if stat_name in baseline.statistics:
                    baseline_value = baseline.statistics[stat_name]
                    if baseline_value != 0:
                        relative_diff = abs((current_value - baseline_value) / baseline_value)
                    else:
                        relative_diff = abs(current_value)
                    
                    differences[f"{stat_name}_diff"] = relative_diff
                    composite_score += relative_diff
            
            # Normalize composite score
            composite_score = composite_score / len(differences) if differences else 0.0
            
            drift_detected = composite_score > self.config.drift_score_threshold
            
            # Severity assessment
            if composite_score > self.config.drift_score_threshold * 4:
                severity = DriftSeverity.CRITICAL
            elif composite_score > self.config.drift_score_threshold * 2:
                severity = DriftSeverity.HIGH
            elif composite_score > self.config.drift_score_threshold:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.STATISTICAL_DISTANCE,
                feature_name=baseline.feature_name,
                drift_score=composite_score,
                p_value=None,
                threshold=self.config.drift_score_threshold,
                message=f"Statistical distance: {composite_score:.4f}",
                statistical_details={
                    'composite_score': composite_score,
                    'baseline_stats': baseline.statistics,
                    'current_stats': current_stats,
                    'individual_differences': differences
                },
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            return self._create_error_result(f"Statistical distance calculation failed: {str(e)}", baseline.feature_name)
    
    def _adversarial_drift_test(self, current_series: pd.Series, baseline: BaselineDistribution) -> DriftDetectionResult:
        """Adversarial drift detection using classifier approach"""
        if not self.sklearn_available:
            return self._create_error_result("Adversarial drift test unavailable - scikit-learn not installed", baseline.feature_name)
        
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import StandardScaler
            
            # Generation baseline sample
            baseline_sample = np.random.normal(
                baseline.statistics['mean'],
                baseline.statistics['std'],
                size=len(current_series)
            )
            
            # Create dataset for classification
            X = np.concatenate([baseline_sample, current_series.values]).reshape(-1, 1)
            y = np.concatenate([
                np.zeros(len(baseline_sample)),  # Baseline = 0
                np.ones(len(current_series))     # Current = 1
            ])
            
            # Scaling features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train classifier to distinguish baseline from current
            classifier = RandomForestClassifier(n_estimators=50, random_state=42)
            
            # Cross-validation accuracy
            cv_scores = cross_val_score(classifier, X_scaled, y, cv=5, scoring='accuracy')
            mean_accuracy = cv_scores.mean()
            
            # Drift assessment: if classifier may baseline from current,
            # is drift. Random baseline = 0.5 accuracy
            drift_score = abs(mean_accuracy - 0.5) * 2  # Normalize to [0, 1]
            drift_detected = mean_accuracy > 0.6  # 60% accuracy indicates drift
            
            # Severity assessment
            if mean_accuracy > 0.8:
                severity = DriftSeverity.CRITICAL
            elif mean_accuracy > 0.7:
                severity = DriftSeverity.HIGH
            elif mean_accuracy > 0.6:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW
            
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=severity,
                detected=drift_detected,
                method=DriftDetectionMethod.ADVERSARIAL_DRIFT,
                feature_name=baseline.feature_name,
                drift_score=drift_score,
                p_value=None,
                threshold=0.1,  # 10% drift score threshold
                message=f"Adversarial drift: classifier accuracy={mean_accuracy:.3f}",
                statistical_details={
                    'classifier_accuracy': mean_accuracy,
                    'cv_scores': cv_scores.tolist(),
                    'drift_score': drift_score,
                    'baseline_distinguishability': mean_accuracy > 0.6
                },
                affected_samples=len(current_series)
            )
            
        except Exception as e:
            return self._create_error_result(f"Adversarial drift test failed: {str(e)}", baseline.feature_name)
    
    def detect_concept_drift(
        self,
        y_true_history: List[np.ndarray],
        y_pred_history: List[np.ndarray],
        performance_metric: Callable[[np.ndarray, np.ndarray], float],
        window_size: Optional[int] = None
    ) -> List[DriftDetectionResult]:
        """
 Detected concept drift on basis performance degradation
        
        Args:
 y_true_history: History true labels by
 y_pred_history: History predictions by
 performance_metric: for performance (onexample, accuracy_score)
 window_size: sliding window
        
        Returns:
            List[DriftDetectionResult]: Results concept drift detection
        """
        if len(y_true_history) != len(y_pred_history):
            raise ValueError("y_true_history y_pred_history should have")
        
        if len(y_true_history) < 2:
            return []  # Insufficient data for drift detection
        
        window_size = window_size or self.config.concept_drift_window_size
        results = []
        
        # Calculation performance for each
        performance_scores = []
        for y_true, y_pred in zip(y_true_history, y_pred_history):
            if len(y_true) > 0 and len(y_pred) > 0:
                try:
                    score = performance_metric(y_true, y_pred)
                    performance_scores.append(score)
                except Exception as e:
                    self.logger.warning(f"Performance metric calculation failed: {e}")
                    continue
        
        if len(performance_scores) < 2:
            return results
        
        # Sliding window analysis for concept drift
        for i in range(1, len(performance_scores)):
            current_performance = performance_scores[i]
            baseline_performance = np.mean(performance_scores[:i]) if i > 0 else performance_scores[0]
            
            # Performance degradation
            performance_drop = baseline_performance - current_performance
            relative_drop = performance_drop / baseline_performance if baseline_performance > 0 else 0
            
            # Concept drift detection
            drift_detected = relative_drop > self.config.concept_drift_threshold
            
            if drift_detected:
                # Severity assessment
                if relative_drop > self.config.concept_drift_threshold * 4:
                    severity = DriftSeverity.CRITICAL
                elif relative_drop > self.config.concept_drift_threshold * 2:
                    severity = DriftSeverity.HIGH
                else:
                    severity = DriftSeverity.MEDIUM
                
                results.append(DriftDetectionResult(
                    drift_type=DriftType.CONCEPT_DRIFT,
                    severity=severity,
                    detected=True,
                    method=DriftDetectionMethod.STATISTICAL_DISTANCE,
                    feature_name=None,
                    drift_score=relative_drop,
                    p_value=None,
                    threshold=self.config.concept_drift_threshold,
                    message=f"Concept drift: performance drop {relative_drop:.1%} at window {i}",
                    statistical_details={
                        'window_index': i,
                        'baseline_performance': baseline_performance,
                        'current_performance': current_performance,
                        'absolute_drop': performance_drop,
                        'relative_drop': relative_drop,
                        'performance_history': performance_scores
                    },
                    recommendations=[
                        "Model overfitting on data",
                        "Analyze changes target distribution",
                        "Consider online learning or incremental updates"
                    ],
                    affected_samples=len(y_true_history[i])
                ))
        
        return results
    
    def detect_prediction_drift(
        self,
        baseline_predictions: np.ndarray,
        current_predictions: np.ndarray,
        methods: Optional[List[DriftDetectionMethod]] = None
    ) -> List[DriftDetectionResult]:
        """
 Detected drift model predictions
        
        Args:
            baseline_predictions: Baseline predictions
            current_predictions: Current predictions
 methods: Method for detection
        
        Returns:
            List[DriftDetectionResult]: Results prediction drift detection
        """
        results = []
        methods_to_use = methods or self.config.prediction_drift_methods
        
        # Create baseline for predictions
        baseline_series = pd.Series(baseline_predictions, name="predictions")
        temp_baseline = self._create_feature_baseline("predictions", baseline_series)
        
        current_series = pd.Series(current_predictions, name="predictions")
        
        # drift detection methods
        for method in methods_to_use:
            drift_result = self._detect_feature_drift(current_series, temp_baseline, method)
            if drift_result:
                drift_result.drift_type = DriftType.PREDICTION_DRIFT
                results.append(drift_result)
        
        return results
    
    def _generate_overall_drift_assessment(
        self, 
        results: List[DriftDetectionResult], 
        baseline_name: str
    ) -> Optional[DriftDetectionResult]:
        """Generation drift by all features"""
        if not results:
            return None
        
        detected_drifts = [r for r in results if r.detected]
        
        if not detected_drifts:
            return DriftDetectionResult(
                drift_type=DriftType.DATA_DRIFT,
                severity=DriftSeverity.LOW,
                detected=False,
                method=DriftDetectionMethod.STATISTICAL_DISTANCE,
                feature_name=None,
                drift_score=0.0,
                p_value=None,
                threshold=0.0,
                message=f"Overall assessment: No drift detected in {baseline_name}",
                statistical_details={
                    'total_features_checked': len(results),
                    'features_with_drift': 0,
                    'baseline_name': baseline_name
                }
            )
        
        # Severity aggregation
        severity_weights = {
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 3,
            DriftSeverity.CRITICAL: 4
        }
        
        total_severity_score = sum(severity_weights[r.severity] for r in detected_drifts)
        avg_severity_score = total_severity_score / len(detected_drifts)
        
        # Overall severity
        if avg_severity_score >= 3.5:
            overall_severity = DriftSeverity.CRITICAL
        elif avg_severity_score >= 2.5:
            overall_severity = DriftSeverity.HIGH
        elif avg_severity_score >= 1.5:
            overall_severity = DriftSeverity.MEDIUM
        else:
            overall_severity = DriftSeverity.LOW
        
        # Overall drift score
        avg_drift_score = np.mean([r.drift_score for r in detected_drifts])
        
        # Features with drift
        features_with_drift = [r.feature_name for r in detected_drifts if r.feature_name]
        
        return DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=overall_severity,
            detected=True,
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            feature_name=None,
            drift_score=avg_drift_score,
            p_value=None,
            threshold=self.config.drift_score_threshold,
            message=f"Overall drift assessment: {len(detected_drifts)}/{len(results)} features show drift",
            statistical_details={
                'total_features_checked': len(results),
                'features_with_drift': len(detected_drifts),
                'drift_percentage': len(detected_drifts) / len(results) * 100,
                'features_with_drift_names': features_with_drift,
                'avg_severity_score': avg_severity_score,
                'severity_distribution': {
                    severity.value: sum(1 for r in detected_drifts if r.severity == severity)
                    for severity in DriftSeverity
                },
                'baseline_name': baseline_name
            },
            recommendations=[
                "drift detected multiple features",
                "overfitting models",
                "Analyze changes data pipeline",
                "Update baseline distributions"
            ]
        )
    
    def _create_error_result(self, error_message: str, feature_name: Optional[str]) -> DriftDetectionResult:
        """Create error result for drift detection"""
        return DriftDetectionResult(
            drift_type=DriftType.DATA_DRIFT,
            severity=DriftSeverity.LOW,
            detected=False,
            method=DriftDetectionMethod.STATISTICAL_DISTANCE,
            feature_name=feature_name,
            drift_score=0.0,
            p_value=None,
            threshold=0.0,
            message=f"Error: {error_message}",
            statistical_details={'error': error_message}
        )
    
    def generate_drift_report(
        self, 
        results: Optional[List[DriftDetectionResult]] = None,
        include_history: bool = False
    ) -> Dict[str, Any]:
        """Generation comprehensive drift detection report"""
        results_to_report = results or self.drift_history
        
        if not results_to_report:
            return {'status': 'no_drift_detection_run'}
        
        # Grouping by drift
        drift_by_type = {}
        for result in results_to_report:
            drift_type = result.drift_type.value
            if drift_type not in drift_by_type:
                drift_by_type[drift_type] = []
            drift_by_type[drift_type].append(result)
        
        # Severity statistics
        severity_stats = {}
        for severity in DriftSeverity:
            severity_results = [r for r in results_to_report if r.severity == severity]
            severity_stats[severity.value] = {
                'total': len(severity_results),
                'detected': sum(1 for r in severity_results if r.detected),
                'not_detected': sum(1 for r in severity_results if not r.detected)
            }
        
        # Method statistics
        method_stats = {}
        for method in DriftDetectionMethod:
            method_results = [r for r in results_to_report if r.method == method]
            method_stats[method.value] = {
                'total': len(method_results),
                'detected': sum(1 for r in method_results if r.detected)
            }
        
        # Critical drift alerts
        critical_drifts = [
            {
                'feature_name': r.feature_name,
                'drift_type': r.drift_type.value,
                'method': r.method.value,
                'drift_score': r.drift_score,
                'message': r.message,
                'recommendations': r.recommendations
            }
            for r in results_to_report 
            if r.detected and r.severity == DriftSeverity.CRITICAL
        ]
        
        # Overall statistics
        total_checks = len(results_to_report)
        detected_drifts = sum(1 for r in results_to_report if r.detected)
        detection_rate = detected_drifts / total_checks * 100 if total_checks > 0 else 0
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'detected_drifts': detected_drifts,
                'detection_rate_percent': detection_rate,
                'overall_status': 'DRIFT_DETECTED' if detected_drifts > 0 else 'NO_DRIFT'
            },
            'drift_by_type': {
                drift_type: {
                    'total_checks': len(results),
                    'detected': sum(1 for r in results if r.detected),
                    'detection_rate': sum(1 for r in results if r.detected) / len(results) * 100
                }
                for drift_type, results in drift_by_type.items()
            },
            'severity_breakdown': severity_stats,
            'method_performance': method_stats,
            'critical_drift_alerts': critical_drifts,
            'detailed_results': [
                {
                    'drift_type': r.drift_type.value,
                    'severity': r.severity.value,
                    'detected': r.detected,
                    'method': r.method.value,
                    'feature_name': r.feature_name,
                    'drift_score': r.drift_score,
                    'p_value': r.p_value,
                    'threshold': r.threshold,
                    'message': r.message,
                    'statistical_details': r.statistical_details,
                    'recommendations': r.recommendations,
                    'timestamp': r.timestamp.isoformat(),
                    'affected_samples': r.affected_samples
                }
                for r in results_to_report
            ]
        }
        
        if include_history:
            report['drift_history'] = {
                'total_historical_checks': len(self.drift_history),
                'baseline_count': len(self.baseline_distributions)
            }
        
        return report
    
    def save_baselines(self, filepath: str) -> None:
        """Save baseline distributions file"""
        try:
            baselines_data = {
                name: baseline.to_dict()
                for name, baseline in self.baseline_distributions.items()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(baselines_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Baseline distributions saved {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error at saving baselines: {e}")
    
    def load_baselines(self, filepath: str) -> None:
        """Load baseline distributions from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                baselines_data = json.load(f)
            
            self.baseline_distributions = {}
            for name, baseline_dict in baselines_data.items():
                baseline = BaselineDistribution.from_dict(baseline_dict)
                self.baseline_distributions[name] = baseline
            
            self.logger.info(f"{len(self.baseline_distributions)} baseline distributions from {filepath}")
        
        except Exception as e:
            self.logger.error(f"Error at loading baselines: {e}")
    
    def export_report(self, filepath: str, results: Optional[List[DriftDetectionResult]] = None) -> None:
        """Export drift detection report file"""
        report = self.generate_drift_report(results, include_history=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Drift detection report exported {filepath}")


def create_crypto_trading_drift_detector() -> DriftDetector:
    """
    Factory function for creation drift detector for crypto trading systems
    Enterprise pre-configured detector for financial ML systems
    """
    config = DriftDetectionConfig(
        # Strict thresholds for crypto trading
        p_value_threshold=0.01,  # 1% significance level
        drift_score_threshold=0.05,  # 5% drift threshold
        
        # PSI thresholds for crypto volatility
 psi_threshold_low=0.05, # 5% for drift
 psi_threshold_medium=0.15, # 15% for drift
 psi_threshold_high=0.25, # 25% for drift
        
        # Statistical distance thresholds
        ks_threshold=0.01,  # Strict KS threshold
        wasserstein_threshold=0.05,  # 5% Wasserstein threshold
        jensen_shannon_threshold=0.05,  # 5% JS threshold
        
        # Concept drift for trading performance
        concept_drift_window_size=50,  # 50 trades window
        concept_drift_threshold=0.03,  # 3% performance degradation
        
        # Comprehensive feature drift methods
        feature_drift_methods=[
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX,
            DriftDetectionMethod.WASSERSTEIN,
            DriftDetectionMethod.STATISTICAL_DISTANCE
        ],
        
        # Prediction drift for trading signals
        prediction_drift_threshold=0.05,  # 5% change in predictions
        prediction_drift_methods=[
            DriftDetectionMethod.WASSERSTEIN,
            DriftDetectionMethod.JENSEN_SHANNON,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX
        ],
        
        # Performance monitoring
        performance_degradation_threshold=0.02,  # 2% performance drop
        minimum_samples_for_detection=50,  # 50 samples minimum
        
        # Temporal drift for market changes
        enable_temporal_drift=True,
        temporal_window_hours=4.0,  # 4-hour windows for crypto markets
        
        # Advanced methods for enterprise detection
        enable_advanced_methods=True,
        save_drift_history=True,
        auto_generate_alerts=True,
        
        # Crypto trading specific
        crypto_volatility_adjustment=True,
        price_drift_sensitivity=0.01,  # 1% price drift sensitivity
        volume_drift_sensitivity=0.03   # 3% volume drift sensitivity
    )
    
    return DriftDetector(config)