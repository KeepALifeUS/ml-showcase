"""
Few-Shot Evaluation System
Comprehensive Meta-Learning Evaluation

System estimation few-shot learning for cryptocurrency trading strategies
with metrics performance, statistical analysis and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.meta_utils import MetaLearningMetrics
from ..algorithms.maml import MAML
from ..algorithms.reptile import Reptile
from ..algorithms.proto_net import PrototypicalNetworks
from ..algorithms.matching_net import MatchingNetworks


@dataclass
class EvaluationConfig:
    """Configuration for few-shot evaluation"""
    
    # Parameters experiment
    num_episodes: int = 100  # Number episodes for estimation
    num_runs: int = 5  # Number launches for statistics
    
    # Few-shot settings
    support_shots: List[int] = field(default_factory=lambda: [1, 5, 10])  # K-shot
    query_shots: int = 15  # Number query examples
    num_ways: List[int] = field(default_factory=lambda: [3, 5])  # N-way
    
    # Adaptation
    adaptation_steps: List[int] = field(default_factory=lambda: [1, 5, 10])
    adaptation_lr: float = 0.01
    
    # Metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc"
    ])
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "mape", "sharpe_ratio"
    ])
    
    # Statistics
    confidence_level: float = 0.95  # Confidence interval
    significance_threshold: float = 0.05  # Level significance
    
    # Crypto-specific metrics
    include_trading_metrics: bool = True
    trading_metrics: List[str] = field(default_factory=lambda: [
        "total_return", "max_drawdown", "win_rate", "profit_factor"
    ])
    
    # Performance
    use_gpu: bool = True
    batch_evaluation: bool = True
    
    # Visualization
    save_plots: bool = True
    plot_dir: str = "./evaluation_plots"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEvaluator(ABC):
    """
    Base class for few-shot evaluators
    
    Abstract Evaluation Framework
    - Consistent evaluation interface
    - Extensible metric system
    - Statistical analysis support
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Results estimation
        self.evaluation_results = defaultdict(list)
        self.episode_results = []
        
        # Utilities
        self.metrics_calculator = MetaLearningMetrics()
        
        # Statistics
        self.timing_stats = defaultdict(list)
        
    @abstractmethod
    def evaluate_episode(
        self,
        model: nn.Module,
        task_data: Dict[str, torch.Tensor],
        num_shots: int,
        num_ways: int,
        adaptation_steps: int
    ) -> Dict[str, float]:
        """Evaluates one episode few-shot learning"""
        pass
    
    def run_evaluation(
        self,
        model: nn.Module,
        task_generator: Callable,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Launches full estimation few-shot performance
        
        Args:
            model: Model for estimation
            task_generator: Generator tasks for estimation
            model_name: Name model for logging
            
        Returns:
            Dictionary with results estimation
        """
        self.logger.info(f"Starting few-shot evaluation for {model_name}")
        
        all_results = defaultdict(lambda: defaultdict(list))
        
        # Run several runs for statistical reliability
        for run in range(self.config.num_runs):
            self.logger.info(f"Run {run + 1}/{self.config.num_runs}")
            
            run_results = self._evaluate_single_run(model, task_generator)
            
            # Collect results
            for setting, metrics in run_results.items():
                for metric_name, value in metrics.items():
                    all_results[setting][metric_name].append(value)
        
        # Aggregate results by all runs
        aggregated_results = self._aggregate_results(all_results)
        
        # Statistical analysis
        statistical_analysis = self._perform_statistical_analysis(all_results)
        
        # Final report
        final_report = {
            'model_name': model_name,
            'config': self.config,
            'aggregated_results': aggregated_results,
            'statistical_analysis': statistical_analysis,
            'timing_stats': dict(self.timing_stats)
        }
        
        self.logger.info(f"Evaluation completed for {model_name}")
        return final_report
    
    def _evaluate_single_run(
        self,
        model: nn.Module,
        task_generator: Callable
    ) -> Dict[str, Dict[str, float]]:
        """Evaluates one run experiment"""
        
        run_results = defaultdict(lambda: defaultdict(list))
        
        # Iterate all combinations parameters
        for num_shots in self.config.support_shots:
            for num_ways in self.config.num_ways:
                for adaptation_steps in self.config.adaptation_steps:
                    
                    setting_name = f"{num_shots}shot_{num_ways}way_{adaptation_steps}adapt"
                    
                    # Evaluate several episodes for given settings
                    for episode in range(self.config.num_episodes):
                        start_time = time.time()
                        
                        # Generate task
                        task_data = task_generator()
                        
                        # Evaluate episode
                        episode_metrics = self.evaluate_episode(
                            model, task_data, num_shots, num_ways, adaptation_steps
                        )
                        
                        # Record time
                        episode_time = time.time() - start_time
                        self.timing_stats[setting_name].append(episode_time)
                        
                        # Collect metrics
                        for metric_name, value in episode_metrics.items():
                            run_results[setting_name][metric_name].append(value)
        
        # Average by episodes
        averaged_results = {}
        for setting_name, metrics_dict in run_results.items():
            averaged_results[setting_name] = {}
            for metric_name, values in metrics_dict.items():
                averaged_results[setting_name][metric_name] = np.mean(values)
        
        return averaged_results
    
    def _aggregate_results(
        self,
        all_results: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Aggregates results by all runs"""
        
        aggregated = {}
        
        for setting_name, metrics_dict in all_results.items():
            aggregated[setting_name] = {}
            
            for metric_name, values in metrics_dict.items():
                values_array = np.array(values)
                
                aggregated[setting_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array)),
                    'q25': float(np.percentile(values_array, 25)),
                    'q75': float(np.percentile(values_array, 75))
                }
                
                # Confidence interval
                confidence_interval = self._compute_confidence_interval(
                    values_array, self.config.confidence_level
                )
                aggregated[setting_name][metric_name]['ci_lower'] = confidence_interval[0]
                aggregated[setting_name][metric_name]['ci_upper'] = confidence_interval[1]
        
        return aggregated
    
    def _compute_confidence_interval(
        self,
        data: np.ndarray,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Computes confidence interval"""
        from scipy import stats
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of mean
        
        # t-distribution for small samples
        dof = len(data) - 1
        confidence_interval = stats.t.interval(
            confidence_level, dof, loc=mean, scale=sem
        )
        
        return float(confidence_interval[0]), float(confidence_interval[1])
    
    def _perform_statistical_analysis(
        self,
        all_results: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Any]:
        """Executes statistical analysis results"""
        from scipy import stats
        
        analysis = {}
        
        # Test on normality distribution
        normality_tests = {}
        for setting_name, metrics_dict in all_results.items():
            normality_tests[setting_name] = {}
            for metric_name, values in metrics_dict.items():
                if len(values) >= 3:  # Minimum for test
                    stat, p_value = stats.shapiro(values)
                    normality_tests[setting_name][metric_name] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': p_value > self.config.significance_threshold
                    }
        
        analysis['normality_tests'] = normality_tests
        
        # Comparison between settings (if exists several)
        if len(all_results) > 1:
            pairwise_comparisons = self._perform_pairwise_comparisons(all_results)
            analysis['pairwise_comparisons'] = pairwise_comparisons
        
        # Effect size analysis
        effect_sizes = self._compute_effect_sizes(all_results)
        analysis['effect_sizes'] = effect_sizes
        
        return analysis
    
    def _perform_pairwise_comparisons(
        self,
        all_results: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Executes pairwise comparison between settings"""
        from scipy import stats
        
        comparisons = {}
        setting_names = list(all_results.keys())
        
        for i, setting1 in enumerate(setting_names):
            for j, setting2 in enumerate(setting_names[i+1:], i+1):
                comparison_key = f"{setting1}_vs_{setting2}"
                comparisons[comparison_key] = {}
                
                # Compare each metric
                for metric_name in all_results[setting1].keys():
                    if metric_name in all_results[setting2]:
                        values1 = all_results[setting1][metric_name]
                        values2 = all_results[setting2][metric_name]
                        
                        # T-test (assume normality)
                        t_stat, t_p = stats.ttest_ind(values1, values2)
                        
                        # Mann-Whitney U test (nonparametric)
                        u_stat, u_p = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                        
                        comparisons[comparison_key][metric_name] = {
                            't_test': {'statistic': float(t_stat), 'p_value': float(t_p)},
                            'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_p)},
                            'significant': min(t_p, u_p) < self.config.significance_threshold
                        }
        
        return comparisons
    
    def _compute_effect_sizes(
        self,
        all_results: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, Dict[str, float]]:
        """Computes dimensions effect (Cohen's d)"""
        
        effect_sizes = {}
        setting_names = list(all_results.keys())
        
        for i, setting1 in enumerate(setting_names):
            for j, setting2 in enumerate(setting_names[i+1:], i+1):
                comparison_key = f"{setting1}_vs_{setting2}"
                effect_sizes[comparison_key] = {}
                
                for metric_name in all_results[setting1].keys():
                    if metric_name in all_results[setting2]:
                        values1 = np.array(all_results[setting1][metric_name])
                        values2 = np.array(all_results[setting2][metric_name])
                        
                        # Cohen's d
                        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                                            (len(values2) - 1) * np.var(values2, ddof=1)) / 
                                           (len(values1) + len(values2) - 2))
                        
                        if pooled_std > 0:
                            cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std
                            effect_sizes[comparison_key][metric_name] = float(cohens_d)
        
        return effect_sizes


class ClassificationEvaluator(BaseEvaluator):
    """
    Evaluator for tasks classification
    
    Classification Evaluation
    - Standard classification metrics
    - Confusion matrix analysis
    - Class-wise performance
    """
    
    def evaluate_episode(
        self,
        model: nn.Module,
        task_data: Dict[str, torch.Tensor],
        num_shots: int,
        num_ways: int,
        adaptation_steps: int
    ) -> Dict[str, float]:
        """Evaluates episode classification"""
        
        support_data = task_data['support_data'][:num_shots * num_ways]
        support_labels = task_data['support_labels'][:num_shots * num_ways]
        query_data = task_data['query_data']
        query_labels = task_data['query_labels']
        
        # Adaptation model (simplified version)
        adapted_model = self._adapt_model(
            model, support_data, support_labels, adaptation_steps
        )
        
        # Predictions on query set
        with torch.no_grad():
            adapted_model.eval()
            query_predictions = adapted_model(query_data)
            
            if hasattr(adapted_model, 'predict_classification'):
                # For specialized models (for example, ProtoNet)
                predictions, probabilities = adapted_model.few_shot_predict(
                    support_data, support_labels, query_data
                )
            else:
                # For regular models
                probabilities = torch.softmax(query_predictions, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
        
        # Convert in numpy for computations metrics
        y_true = query_labels.cpu().numpy()
        y_pred = predictions.cpu().numpy()
        y_proba = probabilities.cpu().numpy()
        
        # Base metrics
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1'] = float(f1)
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        for i in range(min(len(precision_per_class), num_ways)):
            metrics[f'precision_class_{i}'] = float(precision_per_class[i])
            metrics[f'recall_class_{i}'] = float(recall_per_class[i])
            metrics[f'f1_class_{i}'] = float(f1_per_class[i])
        
        # Confusion matrix metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix_trace'] = float(np.trace(cm))
        metrics['confusion_matrix_off_diagonal_sum'] = float(np.sum(cm) - np.trace(cm))
        
        # AUC (for multi-class tasks)
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) > 1 and y_proba.shape[1] > 1:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
                metrics['auc'] = float(auc)
        except:
            metrics['auc'] = 0.0
        
        # Entropy predictions (measure confidence)
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        metrics['prediction_entropy'] = float(np.mean(entropy))
        
        return metrics
    
    def _adapt_model(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        adaptation_steps: int
    ) -> nn.Module:
        """Adapts model to support set"""
        
        # Check, exists whether at model special method adaptation
        if hasattr(model, 'few_shot_adapt'):
            return model.few_shot_adapt(support_data, support_labels, adaptation_steps)
        
        # Otherwise use standard fine-tuning procedure
        adapted_model = type(model)(
            **model.get_config() if hasattr(model, 'get_config') else {}
        )
        adapted_model.load_state_dict(model.state_dict())
        adapted_model.train()
        
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.config.adaptation_lr)
        
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(support_data)
            loss = nn.functional.cross_entropy(predictions, support_labels.long())
            loss.backward()
            optimizer.step()
        
        return adapted_model


class RegressionEvaluator(BaseEvaluator):
    """
    Evaluator for tasks regression
    
    Regression Evaluation
    - Standard regression metrics
    - Crypto trading specific metrics
    - Risk-adjusted performance
    """
    
    def evaluate_episode(
        self,
        model: nn.Module,
        task_data: Dict[str, torch.Tensor],
        num_shots: int,
        num_ways: int,  # Not is used in regression
        adaptation_steps: int
    ) -> Dict[str, float]:
        """Evaluates episode regression"""
        
        support_data = task_data['support_data'][:num_shots]
        support_labels = task_data['support_labels'][:num_shots]
        query_data = task_data['query_data']
        query_labels = task_data['query_labels']
        
        # Adaptation model
        adapted_model = self._adapt_model(
            model, support_data, support_labels, adaptation_steps
        )
        
        # Predictions on query set
        with torch.no_grad():
            adapted_model.eval()
            query_predictions = adapted_model(query_data)
        
        # Convert in numpy
        y_true = query_labels.cpu().numpy()
        y_pred = query_predictions.cpu().numpy().flatten()
        
        # Base metrics regression
        metrics = {}
        
        # MSE, MAE, R²
        metrics['mse'] = float(mean_squared_error(y_true, y_pred))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
        
        try:
            metrics['r2'] = float(r2_score(y_true, y_pred))
        except:
            metrics['r2'] = 0.0
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['mape'] = float(mape)
        
        # Correlation
        if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['correlation'] = float(correlation)
        else:
            metrics['correlation'] = 0.0
        
        # Crypto trading specific metrics
        if self.config.include_trading_metrics:
            trading_metrics = self._compute_trading_metrics(y_true, y_pred)
            metrics.update(trading_metrics)
        
        # Statistics errors
        errors = y_true - y_pred
        metrics['error_mean'] = float(np.mean(errors))
        metrics['error_std'] = float(np.std(errors))
        metrics['error_skewness'] = float(self._compute_skewness(errors))
        metrics['error_kurtosis'] = float(self._compute_kurtosis(errors))
        
        # Percentile metrics
        abs_errors = np.abs(errors)
        metrics['error_percentile_50'] = float(np.percentile(abs_errors, 50))
        metrics['error_percentile_90'] = float(np.percentile(abs_errors, 90))
        metrics['error_percentile_95'] = float(np.percentile(abs_errors, 95))
        
        return metrics
    
    def _compute_trading_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Computes metrics trading strategies"""
        
        trading_metrics = {}
        
        # Generate trading signals
        # If prediction > 0, then BUY, otherwise SELL
        signals = np.where(y_pred > 0, 1, -1)
        
        # Profitability
        returns = y_true * signals  # Assume, that y_true - this profitability
        
        # Total return
        total_return = np.sum(returns)
        trading_metrics['total_return'] = float(total_return)
        
        # Win rate
        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        trading_metrics['win_rate'] = float(win_rate)
        
        # Sharpe ratio
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
            trading_metrics['sharpe_ratio'] = float(sharpe_ratio)
        else:
            trading_metrics['sharpe_ratio'] = 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)
        trading_metrics['max_drawdown'] = float(max_drawdown)
        
        # Profit factor
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = -np.sum(returns[returns < 0])
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
            trading_metrics['profit_factor'] = float(profit_factor)
        else:
            trading_metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(returns) / np.std(downside_returns)
            trading_metrics['sortino_ratio'] = float(sortino_ratio)
        else:
            trading_metrics['sortino_ratio'] = 0.0
        
        return trading_metrics
    
    def _compute_skewness(self, data: np.ndarray) -> float:
        """Computes coefficient asymmetry"""
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return np.mean(((data - mean) / std) ** 3)
        return 0.0
    
    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Computes coefficient kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return np.mean(((data - mean) / std) ** 4) - 3
        return 0.0
    
    def _adapt_model(
        self,
        model: nn.Module,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        adaptation_steps: int
    ) -> nn.Module:
        """Adapts model for regression"""
        
        if hasattr(model, 'few_shot_adapt'):
            return model.few_shot_adapt(support_data, support_labels, adaptation_steps)
        
        # Standard fine-tuning procedure
        adapted_model = type(model)(
            **model.get_config() if hasattr(model, 'get_config') else {}
        )
        adapted_model.load_state_dict(model.state_dict())
        adapted_model.train()
        
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.config.adaptation_lr)
        
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            predictions = adapted_model(support_data)
            loss = nn.functional.mse_loss(predictions, support_labels)
            loss.backward()
            optimizer.step()
        
        return adapted_model


class FewShotBenchmark:
    """
    Comprehensive system benchmarking for few-shot learning
    
    Comprehensive Benchmarking System
    - Multi-algorithm comparison
    - Statistical significance testing
    - Automated reporting
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Results benchmark
        self.benchmark_results = {}
        
    def run_benchmark(
        self,
        models: Dict[str, nn.Module],
        task_generator: Callable,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Launches comprehensive benchmark models
        
        Args:
            models: Dictionary models for comparison
            task_generator: Generator tasks
            task_type: Type tasks (classification/regression)
            
        Returns:
            Results benchmark
        """
        
        self.logger.info(f"Starting benchmark for {len(models)} models")
        
        # Select evaluator
        if task_type == "classification":
            evaluator = ClassificationEvaluator(self.config, self.logger)
        else:
            evaluator = RegressionEvaluator(self.config, self.logger)
        
        # Evaluate each model
        for model_name, model in models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            
            model_results = evaluator.run_evaluation(
                model, task_generator, model_name
            )
            
            self.benchmark_results[model_name] = model_results
        
        # Comparative analysis
        comparison_analysis = self._perform_comparison_analysis()
        
        # Generate report
        final_report = {
            'benchmark_config': self.config,
            'task_type': task_type,
            'individual_results': self.benchmark_results,
            'comparison_analysis': comparison_analysis,
            'summary': self._generate_summary()
        }
        
        # Save results
        if self.config.save_plots:
            self._generate_visualizations(final_report)
        
        self.logger.info("Benchmark completed")
        return final_report
    
    def _perform_comparison_analysis(self) -> Dict[str, Any]:
        """Executes comparative analysis models"""
        
        if len(self.benchmark_results) < 2:
            return {"note": "Need at least 2 models for comparison"}
        
        analysis = {}
        
        # Collect all metrics
        all_metrics = set()
        for model_results in self.benchmark_results.values():
            for setting_results in model_results['aggregated_results'].values():
                all_metrics.update(setting_results.keys())
        
        # Rating models by of each metric
        rankings = {}
        for metric in all_metrics:
            metric_rankings = self._rank_models_by_metric(metric)
            if metric_rankings:
                rankings[metric] = metric_rankings
        
        analysis['rankings'] = rankings
        
        # Statistical significance differences
        significance_tests = self._test_statistical_significance()
        analysis['significance_tests'] = significance_tests
        
        # Total rating
        overall_ranking = self._compute_overall_ranking(rankings)
        analysis['overall_ranking'] = overall_ranking
        
        return analysis
    
    def _rank_models_by_metric(self, metric_name: str) -> Dict[str, List[str]]:
        """Ranks model by specific metric"""
        
        rankings = {}
        
        # Collect values metrics for all models and settings
        for setting in self.config.support_shots:
            setting_key = f"{setting}shot"  # Simplified key
            
            model_scores = {}
            for model_name, results in self.benchmark_results.items():
                # Search corresponding setting
                for setting_name, setting_results in results['aggregated_results'].items():
                    if f"{setting}shot" in setting_name and metric_name in setting_results:
                        model_scores[model_name] = setting_results[metric_name]['mean']
                        break
            
            if model_scores:
                # Sort (for accuracy, f1 - by descending, for loss - by ascending)
                reverse = metric_name in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'correlation']
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse)
                rankings[setting_key] = [model_name for model_name, _ in sorted_models]
        
        return rankings
    
    def _test_statistical_significance(self) -> Dict[str, Any]:
        """Tests statistical significance differences"""
        # Simplified version - full implementation will require access to raw data
        return {"note": "Statistical significance testing requires access to raw episode results"}
    
    def _compute_overall_ranking(self, rankings: Dict[str, Dict[str, List[str]]]) -> List[str]:
        """Computes total rating models"""
        
        if not rankings:
            return []
        
        # Count points for of each model (1 place = n points, 2 place = n-1 points, etc.)
        model_scores = defaultdict(int)
        total_rankings = 0
        
        for metric_rankings in rankings.values():
            for setting_rankings in metric_rankings.values():
                total_rankings += 1
                for i, model_name in enumerate(setting_rankings):
                    points = len(setting_rankings) - i
                    model_scores[model_name] += points
        
        # Normalize and sort
        if total_rankings > 0:
            for model_name in model_scores:
                model_scores[model_name] /= total_rankings
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return [model_name for model_name, _ in sorted_models]
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generates summary information"""
        
        summary = {
            'num_models_evaluated': len(self.benchmark_results),
            'total_episodes_per_model': self.config.num_episodes * self.config.num_runs,
            'configurations_tested': len(self.config.support_shots) * len(self.config.num_ways) * len(self.config.adaptation_steps)
        }
        
        # Best results by of each main metric
        key_metrics = ['accuracy', 'f1', 'mse', 'sharpe_ratio']
        best_results = {}
        
        for metric in key_metrics:
            best_score = None
            best_model = None
            best_setting = None
            
            for model_name, results in self.benchmark_results.items():
                for setting_name, setting_results in results['aggregated_results'].items():
                    if metric in setting_results:
                        score = setting_results[metric]['mean']
                        
                        # Define, better whether this result
                        is_better = False
                        if best_score is None:
                            is_better = True
                        elif metric in ['accuracy', 'f1', 'r2', 'correlation', 'sharpe_ratio']:
                            is_better = score > best_score
                        else:  # For loss metrics
                            is_better = score < best_score
                        
                        if is_better:
                            best_score = score
                            best_model = model_name
                            best_setting = setting_name
            
            if best_score is not None:
                best_results[metric] = {
                    'score': best_score,
                    'model': best_model,
                    'setting': best_setting
                }
        
        summary['best_results'] = best_results
        
        return summary
    
    def _generate_visualizations(self, report: Dict[str, Any]) -> None:
        """Generates visualization results"""
        
        import os
        os.makedirs(self.config.plot_dir, exist_ok=True)
        
        # Comparison models by main metrics
        self._plot_model_comparison(report)
        
        # Influence number shots on performance
        self._plot_shots_effect(report)
        
        # Heatmap performance
        self._plot_performance_heatmap(report)
        
        self.logger.info(f"Visualizations saved to {self.config.plot_dir}")
    
    def _plot_model_comparison(self, report: Dict[str, Any]) -> None:
        """Creates chart comparison models"""
        # Placeholder for visualization
        pass
    
    def _plot_shots_effect(self, report: Dict[str, Any]) -> None:
        """Creates chart influence number shots"""
        # Placeholder for visualization
        pass
    
    def _plot_performance_heatmap(self, report: Dict[str, Any]) -> None:
        """Creates heatmap performance"""
        # Placeholder for visualization
        pass