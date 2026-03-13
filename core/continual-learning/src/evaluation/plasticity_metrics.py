"""
Plasticity Metrics for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system metrics for dimensions plasticity model
( learn new tasks) with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json
from collections import defaultdict


class PlasticityMeasure(Enum):
    """Types metrics plasticity"""
    FORWARD_TRANSFER = "forward_transfer"           # FWT - Forward Transfer
    LEARNING_EFFICIENCY = "learning_efficiency" # Speed learning new tasks
    ADAPTATION_RATE = "adaptation_rate" # Speed adaptation to new
    KNOWLEDGE_RETENTION = "knowledge_retention" # Save knowledge
    TRANSFER_ABILITY = "transfer_ability" # to transfer knowledge
    LEARNING_CURVE_ANALYSIS = "learning_curve" # Analysis curve training


@dataclass
class LearningEvent:
    """Event training for analysis plasticity"""
    task_id: int
    initial_performance: float
    final_performance: float
    learning_speed: float # Speed achieving performance
    convergence_epochs: int
    market_regime: str
    timestamp: datetime
    transfer_benefit: float  # Benefit from previous tasks
    learning_difficulty: str  # easy, medium, hard
    knowledge_sources: List[int] # Tasks, from which benefit


class PlasticityMetrics:
    """
    System metrics for analysis plasticity in continual training
    
    enterprise Features:
    - Multi-dimensional plasticity analysis
    - Learning efficiency tracking
    - Knowledge transfer quantification
    - Market regime adaptation analysis
    - Predictive learning difficulty assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("PlasticityMetrics")
        
        # Settings analysis
        self.transfer_threshold = 0.05 # Minimum threshold for transfer
        self.learning_speed_baseline = 0.1 # Base speed training
        self.convergence_patience = 10 # for determining convergence
        
        # enterprise settings
        self.market_regime_analysis = True
        self.predictive_analysis = True
        self.learning_efficiency_tracking = True
        self.knowledge_source_tracking = True
        
        # History training
        self.learning_history: List[LearningEvent] = []
        self.task_learning_curves: Dict[int, List[Tuple[int, float]]] = {}  # epoch -> performance
        self.baseline_learning_times: Dict[str, float] = {} # By market regimes
        
        # Statistics transfer knowledge
        self.transfer_matrix: Dict[Tuple[int, int], float] = {}  # (source, target) -> transfer_score
        self.knowledge_sources: Dict[int, List[int]] = {}  # task -> source_tasks
    
    def calculate_forward_transfer(
        self,
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> float:
        """
        Calculation Forward Transfer (FWT) - influence previous knowledge on new tasks
        
        FWT = (1/T-1) * Σ(R_i,i - b_i) for i = 2 up to T
        where R_i,i - initial performance on task i,
        b_i - performance random initialization
        
        Args:
            performance_history: History performance by tasks
            reference_metric: Metric for analysis
            
        Returns:
            FWT score (positive = positive transfer)
        """
        if len(performance_history) < 2:
            return 0.0
        
        task_ids = sorted(performance_history.keys())
        transfer_effects = []
        
        # Evaluate baseline performance (random initialization)
        baseline_performance = self._estimate_baseline_performance(performance_history, reference_metric)
        
        for task_id in task_ids[1:]: # Starting with second tasks
            # Initial performance on new task
            if task_id in self.learning_history:
                event = next(e for e in self.learning_history if e.task_id == task_id)
                initial_perf = event.initial_performance
            else:
                # If no in history, use current performance
                initial_perf = performance_history[task_id].get(reference_metric, baseline_performance)
            
            # Transfer effect = initial - baseline
            transfer_effect = initial_perf - baseline_performance
            transfer_effects.append(transfer_effect)
        
        # Average forward transfer
        fwt = np.mean(transfer_effects) if transfer_effects else 0.0
        
        self.logger.debug(f"Forward Transfer calculated: {fwt:.4f}")
        return float(fwt)
    
    def _estimate_baseline_performance(
        self,
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str
    ) -> float:
        """
        Evaluate base performance random initialization
        
        Args:
            performance_history: History performance
            reference_metric: Metric for analysis
            
        Returns:
            Evaluate base performance
        """
        # For crypto trading possible use
        if reference_metric == "accuracy":
            return 0.5  # Random classification directions
        elif reference_metric in ["mae", "rmse", "loss"]:
            # For regression take average across all tasks as approximate baseline
            all_values = []
            for metrics in performance_history.values():
                if reference_metric in metrics:
                    all_values.append(metrics[reference_metric])
            return np.mean(all_values) * 1.5 if all_values else 1.0 # Assuming performance
        else:
            return 0.5
    
    def calculate_learning_efficiency(
        self,
        task_id: int,
        learning_curve: List[Tuple[int, float]],
        target_performance: float = 0.8
    ) -> Dict[str, Any]:
        """
        Calculation efficiency training for specific tasks
        
        Args:
            task_id: ID tasks
            learning_curve: Curve training [(epoch, performance), ...]
            target_performance: Target performance
            
        Returns:
            Metrics efficiency training
        """
        if not learning_curve:
            return {"error": "empty_learning_curve"}
        
        epochs, performances = zip(*learning_curve)
        
        # Initial and performance
        initial_performance = performances[0]
        final_performance = performances[-1]
        max_performance = max(performances)
        
        # Speed training (improvement for epoch)
        if len(performances) > 1:
            learning_speed = (final_performance - initial_performance) / len(performances)
        else:
            learning_speed = 0.0
        
        # Time up to achieving target performance
        convergence_epoch = None
        for epoch, perf in learning_curve:
            if perf >= target_performance:
                convergence_epoch = epoch
                break
        
        # Stability training ( in last 20% epochs)
        stability_window = max(1, len(performances) // 5)
        recent_performances = performances[-stability_window:]
        learning_stability = 1.0 / (1.0 + np.var(recent_performances))
        
        # = speed * * goal
        if convergence_epoch is not None:
            convergence_efficiency = 1.0 / convergence_epoch if convergence_epoch > 0 else 1.0
        else:
            convergence_efficiency = 0.0
        
        overall_efficiency = (learning_speed * 0.4 + 
                             learning_stability * 0.3 + 
                             convergence_efficiency * 0.3)
        
        efficiency_metrics = {
            "task_id": task_id,
            "initial_performance": initial_performance,
            "final_performance": final_performance,
            "max_performance": max_performance,
            "learning_speed": learning_speed,
            "convergence_epoch": convergence_epoch,
            "learning_stability": learning_stability,
            "overall_efficiency": overall_efficiency,
            "epochs_trained": len(performances),
            "target_achieved": final_performance >= target_performance
        }
        
        # Save curve training
        self.task_learning_curves[task_id] = learning_curve
        
        return efficiency_metrics
    
    def analyze_adaptation_rate(
        self,
        task_id: int,
        market_regime: str,
        learning_curve: List[Tuple[int, float]]
    ) -> Dict[str, Any]:
        """
        Analysis speed adaptation to new market regime
        
        Market regime adaptation analysis
        
        Args:
            task_id: ID tasks
            market_regime: Market regime
            learning_curve: Curve training
            
        Returns:
            Analysis adaptation
        """
        if not learning_curve or not self.market_regime_analysis:
            return {}
        
        epochs, performances = zip(*learning_curve)
        
        # Basic time adaptation for regime
        if market_regime not in self.baseline_learning_times:
            # Initialize basic time
            self.baseline_learning_times[market_regime] = len(performances)
        
        baseline_time = self.baseline_learning_times[market_regime]
        current_time = len(performances)
        
        # Speed adaptation baseline
        if baseline_time > 0:
            adaptation_speedup = baseline_time / current_time
        else:
            adaptation_speedup = 1.0
        
        # Quality adaptation
        final_performance = performances[-1]
        performance_improvement = final_performance - performances[0]
        
        # Comparison with other regimes
        regime_comparison = {}
        for regime, time in self.baseline_learning_times.items():
            if regime != market_regime:
                regime_comparison[regime] = current_time / time if time > 0 else 1.0
        
        adaptation_analysis = {
            "task_id": task_id,
            "market_regime": market_regime,
            "adaptation_time": current_time,
            "baseline_time": baseline_time,
            "adaptation_speedup": adaptation_speedup,
            "performance_improvement": performance_improvement,
            "final_performance": final_performance,
            "regime_comparison": regime_comparison,
            "adaptation_quality": self._classify_adaptation_quality(
                adaptation_speedup, performance_improvement
            )
        }
        
        # Update basic time (moving average)
        self.baseline_learning_times[market_regime] = (
            self.baseline_learning_times[market_regime] * 0.8 + current_time * 0.2
        )
        
        return adaptation_analysis
    
    def _classify_adaptation_quality(self, speedup: float, improvement: float) -> str:
        """
        Classification quality adaptation
        
        Args:
            speedup: adaptation
            improvement: Improvement performance
            
        Returns:
            Quality adaptation: excellent, good, fair, poor
        """
        if speedup > 1.5 and improvement > 0.3:
            return "excellent"
        elif speedup > 1.2 and improvement > 0.2:
            return "good"
        elif speedup > 0.8 and improvement > 0.1:
            return "fair"
        else:
            return "poor"
    
    def track_knowledge_transfer(
        self,
        source_task_id: int,
        target_task_id: int,
        transfer_score: float
    ) -> None:
        """
        Track transfer knowledge between tasks
        
        Args:
            source_task_id: ID original tasks
            target_task_id: ID target tasks
            transfer_score: Evaluate transfer
        """
        if not self.knowledge_source_tracking:
            return
        
        # Save evaluation transfer
        self.transfer_matrix[(source_task_id, target_task_id)] = transfer_score
        
        # Update sources knowledge for target tasks
        if target_task_id not in self.knowledge_sources:
            self.knowledge_sources[target_task_id] = []
        
        if transfer_score > self.transfer_threshold:
            if source_task_id not in self.knowledge_sources[target_task_id]:
                self.knowledge_sources[target_task_id].append(source_task_id)
        
        self.logger.debug(f"Knowledge transfer tracked: {source_task_id} -> {target_task_id} = {transfer_score:.4f}")
    
    def analyze_knowledge_transfer_patterns(self) -> Dict[str, Any]:
        """
        Analysis patterns transfer knowledge
        
        Returns:
            Analysis patterns transfer
        """
        if not self.transfer_matrix:
            return {"status": "no_transfer_data"}
        
        # Statistics by transfer
        transfer_scores = list(self.transfer_matrix.values())
        positive_transfers = [score for score in transfer_scores if score > self.transfer_threshold]
        negative_transfers = [score for score in transfer_scores if score < -self.transfer_threshold]
        
        analysis = {
            "total_transfers": len(transfer_scores),
            "positive_transfers": len(positive_transfers),
            "negative_transfers": len(negative_transfers),
            "average_transfer": np.mean(transfer_scores),
            "best_transfer": max(transfer_scores) if transfer_scores else 0.0,
            "worst_transfer": min(transfer_scores) if transfer_scores else 0.0
        }
        
        # Analysis sources knowledge
        if self.knowledge_sources:
            source_popularity = defaultdict(int)
            for target_task, sources in self.knowledge_sources.items():
                for source in sources:
                    source_popularity[source] += 1
            
            # Top knowledge
            top_sources = sorted(source_popularity.items(), key=lambda x: x[1], reverse=True)
            analysis["top_knowledge_sources"] = top_sources[:5]
            
            # Tasks with highest number sources
            knowledge_rich_tasks = sorted(
                [(task, len(sources)) for task, sources in self.knowledge_sources.items()],
                key=lambda x: x[1], reverse=True
            )
            analysis["knowledge_rich_tasks"] = knowledge_rich_tasks[:5]
        
        # Matrix transfer for visualization
        if len(self.transfer_matrix) <= 100: # Only for matrices
            analysis["transfer_matrix"] = dict(self.transfer_matrix)
        
        return analysis
    
    def predict_learning_difficulty(
        self,
        task_metadata: Dict[str, Any],
        historical_performance: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Prediction complexity learning new tasks
        
        Predictive difficulty assessment
        
        Args:
            task_metadata: Metadata new tasks
            historical_performance: Historical performance
            
        Returns:
            Forecast complexity learning
        """
        if not self.predictive_analysis:
            return {}
        
        market_regime = task_metadata.get("market_regime", "unknown")
        
        # Analysis historical data for tasks
        similar_tasks_performance = []
        similar_tasks_times = []
        
        for event in self.learning_history:
            if event.market_regime == market_regime:
                similar_tasks_performance.append(event.final_performance)
                similar_tasks_times.append(event.convergence_epochs)
        
        if not similar_tasks_performance:
            # No historical data for this regime
            return {
                "difficulty_prediction": "unknown",
                "confidence": 0.0,
                "reason": f"no_historical_data_for_{market_regime}"
            }
        
        # Forecast on basis historical data
        expected_performance = np.mean(similar_tasks_performance)
        expected_time = np.mean(similar_tasks_times)
        performance_variance = np.var(similar_tasks_performance)
        
        # Classification complexity
        if expected_performance > 0.8 and expected_time < 20:
            difficulty = "easy"
        elif expected_performance > 0.6 and expected_time < 50:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        # on basis number data and
        confidence = min(len(similar_tasks_performance) / 10, 1.0) * (1.0 - performance_variance)
        confidence = max(0.0, min(1.0, confidence))
        
        prediction = {
            "difficulty_prediction": difficulty,
            "expected_performance": expected_performance,
            "expected_convergence_time": expected_time,
            "confidence": confidence,
            "similar_tasks_count": len(similar_tasks_performance),
            "performance_variance": performance_variance,
            "market_regime": market_regime
        }
        
        # Recommendations on basis forecast
        prediction["recommendations"] = self._generate_difficulty_recommendations(difficulty, prediction)
        
        return prediction
    
    def _generate_difficulty_recommendations(
        self, 
        difficulty: str, 
        prediction: Dict[str, Any]
    ) -> List[str]:
        """
        Generation recommendations on basis complexity
        
        Args:
            difficulty: complexity
            prediction: Details forecast
            
        Returns:
            List recommendations
        """
        recommendations = []
        
        if difficulty == "easy":
            recommendations.extend([
                "Standard learning parameters should work well",
                "Consider faster learning rate for quicker convergence"
            ])
        elif difficulty == "medium":
            recommendations.extend([
                "Monitor convergence carefully",
                "Consider experience replay if available",
                "Prepare for moderate training time"
            ])
        else:  # hard
            recommendations.extend([
                "Reduce learning rate for stability",
                "Increase regularization to prevent overfitting",
                "Allocate more training time",
                "Consider transfer learning from similar tasks",
                "Monitor for catastrophic forgetting closely"
            ])
        
        # Additional recommendations on basis confidence
        if prediction["confidence"] < 0.5:
            recommendations.append("Low confidence prediction - monitor closely and adjust as needed")
        
        return recommendations
    
    def record_learning_event(
        self,
        task_id: int,
        initial_performance: float,
        final_performance: float,
        convergence_epochs: int,
        market_regime: str,
        knowledge_sources: Optional[List[int]] = None
    ) -> None:
        """
        Record events training
        
        Args:
            task_id: ID tasks
            initial_performance: Initial performance
            final_performance: Final performance
            convergence_epochs: Number epochs up to convergence
            market_regime: Market regime
            knowledge_sources: knowledge
        """
        if convergence_epochs > 0:
            learning_speed = (final_performance - initial_performance) / convergence_epochs
        else:
            learning_speed = 0.0
        
        # Determine complexity training
        if final_performance > 0.8 and convergence_epochs < 20:
            difficulty = "easy"
        elif final_performance > 0.6 and convergence_epochs < 50:
            difficulty = "medium"
        else:
            difficulty = "hard"
        
        # Calculation from transfer
        transfer_benefit = 0.0
        if knowledge_sources:
            for source_id in knowledge_sources:
                transfer_score = self.transfer_matrix.get((source_id, task_id), 0.0)
                transfer_benefit += max(0.0, transfer_score)
        
        event = LearningEvent(
            task_id=task_id,
            initial_performance=initial_performance,
            final_performance=final_performance,
            learning_speed=learning_speed,
            convergence_epochs=convergence_epochs,
            market_regime=market_regime,
            timestamp=datetime.now(),
            transfer_benefit=transfer_benefit,
            learning_difficulty=difficulty,
            knowledge_sources=knowledge_sources or []
        )
        
        self.learning_history.append(event)
        self.logger.debug(f"Learning event recorded for task {task_id}: {difficulty} difficulty")
    
    def get_comprehensive_plasticity_analysis(
        self,
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis plasticity
        
        Args:
            performance_history: History performance
            reference_metric: Metric for analysis
            
        Returns:
            Full analysis plasticity
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "metric_used": reference_metric,
            "num_tasks": len(performance_history),
            "num_learning_events": len(self.learning_history)
        }
        
        # Main metrics
        analysis["forward_transfer"] = self.calculate_forward_transfer(
            performance_history, reference_metric
        )
        
        # Analysis efficiency training by tasks
        learning_efficiencies = {}
        for task_id, curve in self.task_learning_curves.items():
            efficiency = self.calculate_learning_efficiency(task_id, curve)
            learning_efficiencies[task_id] = efficiency
        
        analysis["learning_efficiencies"] = learning_efficiencies
        
        # Statistics by complexity tasks
        if self.learning_history:
            difficulty_stats = defaultdict(int)
            regime_stats = defaultdict(list)
            
            for event in self.learning_history:
                difficulty_stats[event.learning_difficulty] += 1
                regime_stats[event.market_regime].append(event.final_performance)
            
            analysis["difficulty_distribution"] = dict(difficulty_stats)
            analysis["regime_performance"] = {
                regime: {
                    "count": len(performances),
                    "avg_performance": np.mean(performances),
                    "std_performance": np.std(performances)
                }
                for regime, performances in regime_stats.items()
            }
        
        # Analysis transfer knowledge
        analysis["knowledge_transfer_analysis"] = self.analyze_knowledge_transfer_patterns()
        
        # General trends
        analysis["plasticity_trends"] = self._analyze_plasticity_trends()
        
        return analysis
    
    def _analyze_plasticity_trends(self) -> Dict[str, Any]:
        """
        Analysis trends plasticity time
        
        Returns:
            Analysis trends
        """
        if len(self.learning_history) < 3:
            return {"status": "insufficient_data"}
        
        # Sort by time
        sorted_events = sorted(self.learning_history, key=lambda e: e.timestamp)
        
        # Analysis trends performance
        performances = [event.final_performance for event in sorted_events]
        learning_speeds = [event.learning_speed for event in sorted_events]
        
        # trends
        x = np.arange(len(performances))
        
        if len(performances) >= 2:
            perf_trend = np.polyfit(x, performances, 1)[0]  # Slope
            speed_trend = np.polyfit(x, learning_speeds, 1)[0]
        else:
            perf_trend = 0.0
            speed_trend = 0.0
        
        trends = {
            "performance_trend": "improving" if perf_trend > 0.01 else "declining" if perf_trend < -0.01 else "stable",
            "learning_speed_trend": "accelerating" if speed_trend > 0.001 else "decelerating" if speed_trend < -0.001 else "stable",
            "performance_slope": perf_trend,
            "learning_speed_slope": speed_trend,
            "recent_avg_performance": np.mean(performances[-5:]) if len(performances) >= 5 else np.mean(performances),
            "early_avg_performance": np.mean(performances[:5]) if len(performances) >= 10 else np.mean(performances[:len(performances)//2])
        }
        
        return trends
    
    def export_plasticity_report(self, filepath: str) -> bool:
        """
        Export report plasticity
        
        Args:
            filepath: Path for saving
            
        Returns:
            True if export successful
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_learning_events": len(self.learning_history),
                "learning_curves_recorded": len(self.task_learning_curves),
                "baseline_learning_times": self.baseline_learning_times,
                "transfer_relationships": len(self.transfer_matrix),
                "recent_learning_events": [
                    {
                        "task_id": event.task_id,
                        "final_performance": event.final_performance,
                        "learning_speed": event.learning_speed,
                        "difficulty": event.learning_difficulty,
                        "market_regime": event.market_regime,
                        "transfer_benefit": event.transfer_benefit,
                        "timestamp": event.timestamp.isoformat()
                    }
                    for event in self.learning_history[-20:]
                ],
                "knowledge_transfer_summary": self.analyze_knowledge_transfer_patterns(),
                "configuration": {
                    "transfer_threshold": self.transfer_threshold,
                    "learning_speed_baseline": self.learning_speed_baseline,
                    "convergence_patience": self.convergence_patience
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Plasticity report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting plasticity report: {e}")
            return False
    
    def __repr__(self) -> str:
        return (
            f"PlasticityMetrics("
            f"events={len(self.learning_history)}, "
            f"curves={len(self.task_learning_curves)}, "
            f"transfers={len(self.transfer_matrix)})"
        )