"""
Forgetting Metrics for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system metrics for dimensions catastrophic forgetting
in crypto trading with integration.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import json


class ForgettingMeasure(Enum):
    """Types metrics forgetting"""
    BACKWARD_TRANSFER = "backward_transfer"          # BWT - Backward Transfer
    FORGETTING_MEASURE = "forgetting_measure"       # FM - Forgetting Measure 
    RETENTION_RATE = "retention_rate" # Fraction performance
    CATASTROPHIC_FORGETTING = "catastrophic_forgetting" # performance
    INCREMENTAL_ACCURACY = "incremental_accuracy"   # Accuracy after each new tasks
    MEMORY_STABILITY = "memory_stability" # Stability memory tasks


@dataclass
class ForgettingEvent:
    """Event forgetting for analysis"""
    task_id: int
    previous_performance: float
    current_performance: float
    forgetting_magnitude: float
    timestamp: datetime
    market_regime: str
    new_task_introduced: int # ID tasks, which forgetting
    recovery_possible: bool
    severity_level: str  # mild, moderate, severe, catastrophic


class ForgettingMetrics:
    """
    Metrics system for forgetting analysis in continual learning
    
    enterprise Features:
    - Multi-dimensional forgetting analysis
    - Market regime aware evaluation
    - Temporal forgetting patterns
    - Recovery tracking and prediction
    - Performance degradation alerts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger("ForgettingMetrics")
        
        # Settings evaluation
        self.catastrophic_threshold = 0.3  # Threshold catastrophic forgetting
        self.mild_forgetting_threshold = 0.05 # Threshold forgetting
        self.recovery_tracking_enabled = True
        self.temporal_analysis_enabled = True
        
        # enterprise settings
        self.market_regime_analysis = True
        self.performance_alerts = True
        self.forgetting_prediction = True
        
        # History forgetting
        self.forgetting_history: List[ForgettingEvent] = []
        self.task_trajectories: Dict[int, List[Tuple[datetime, float]]] = {}
        self.baseline_performances: Dict[int, float] = {}
        
        # Statistics by market regimes
        self.regime_forgetting_stats: Dict[str, Dict[str, float]] = {}
    
    def calculate_backward_transfer(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> float:
        """
        Calculation Backward Transfer (BWT) - average learning new tasks on old
        
        BWT = (1/(T-1)) * Σ(R_i,T - R_i,i) for i = 1 up to T-1
        where R_i,j - performance on task i after learning j tasks
        
        Args:
            performance_history: History performance by tasks
            reference_metric: Metric for analysis (accuracy, loss, etc.)
            
        Returns:
            BWT score (positive = positive transfer, = forgetting)
        """
        if len(performance_history) < 2:
            return 0.0
        
        task_ids = sorted(performance_history.keys())
        transfer_effects = []
        
        for i, task_id in enumerate(task_ids[:-1]): # Excluding last task
            # Base performance immediately after learning tasks
            if task_id not in self.baseline_performances:
                baseline_perf = performance_history[task_id].get(reference_metric, 0.0)
                self.baseline_performances[task_id] = baseline_perf
            else:
                baseline_perf = self.baseline_performances[task_id]
            
            # Current performance on this task
            current_perf = performance_history[task_id].get(reference_metric, 0.0)
            
            # Transfer effect (positive = improvement, negative = forgetting)
            transfer_effect = current_perf - baseline_perf
            transfer_effects.append(transfer_effect)
        
        # Average backward transfer
        bwt = np.mean(transfer_effects) if transfer_effects else 0.0
        
        self.logger.debug(f"Backward Transfer calculated: {bwt:.4f}")
        return float(bwt)
    
    def calculate_forgetting_measure(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> float:
        """
        Calculation Forgetting Measure - forgetting by all tasks
        
        FM = (1/T-1) * Σ max(0, f_k) for k = 1 up to T-1
        where f_k = max(R_k,j for j=k up to T-1) - R_k,T
        
        Args:
            performance_history: History performance by tasks
            reference_metric: Metric for analysis
            
        Returns:
            Forgetting measure (0 = no forgetting, more = more forgetting)
        """
        if len(performance_history) < 2:
            return 0.0
        
        task_ids = sorted(performance_history.keys())
        forgetting_scores = []
        
        for task_id in task_ids[:-1]: # Excluding last task
            # Maximum performance, achieved on this task
            max_performance = self.baseline_performances.get(task_id, 0.0)
            
            # Current performance
            current_performance = performance_history[task_id].get(reference_metric, 0.0)
            
            # Forgetting = max(0, maximum - current)
            forgetting = max(0.0, max_performance - current_performance)
            forgetting_scores.append(forgetting)
        
        # Average forgetting
        fm = np.mean(forgetting_scores) if forgetting_scores else 0.0
        
        self.logger.debug(f"Forgetting Measure calculated: {fm:.4f}")
        return float(fm)
    
    def calculate_retention_rate(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> Dict[int, float]:
        """
        Calculation coefficient saving for each tasks
        
        Retention Rate = Current Performance / Baseline Performance
        
        Args:
            performance_history: History performance by tasks
            reference_metric: Metric for analysis
            
        Returns:
            Dict with saving for each tasks
        """
        retention_rates = {}
        
        for task_id, metrics in performance_history.items():
            baseline = self.baseline_performances.get(task_id)
            current = metrics.get(reference_metric, 0.0)
            
            if baseline is not None and baseline > 0:
                retention_rate = current / baseline
                retention_rates[task_id] = min(retention_rate, 1.0)  # Limit up to 1.0
            else:
                retention_rates[task_id] = 1.0 # If no baseline, full saving
        
        self.logger.debug(f"Retention rates calculated for {len(retention_rates)} tasks")
        return retention_rates
    
    def detect_catastrophic_forgetting(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> List[ForgettingEvent]:
        """
        Detect events catastrophic forgetting
        
        Args:
            performance_history: History performance by tasks
            reference_metric: Metric for analysis
            
        Returns:
            List events catastrophic forgetting
        """
        catastrophic_events = []
        current_time = datetime.now()
        
        retention_rates = self.calculate_retention_rate(performance_history, reference_metric)
        
        for task_id, retention_rate in retention_rates.items():
            if retention_rate < (1.0 - self.catastrophic_threshold):
                # Determine severity level
                if retention_rate < 0.3:
                    severity = "catastrophic"
                elif retention_rate < 0.5:
                    severity = "severe"
                elif retention_rate < 0.7:
                    severity = "moderate"
                else:
                    severity = "mild"
                
                # Search tasks, which cause forgetting
                causing_task = max(performance_history.keys()) if performance_history else task_id
                
                # Get base performance
                baseline = self.baseline_performances.get(task_id, 1.0)
                current_perf = performance_history[task_id].get(reference_metric, 0.0)
                
                event = ForgettingEvent(
                    task_id=task_id,
                    previous_performance=baseline,
                    current_performance=current_perf,
                    forgetting_magnitude=baseline - current_perf,
                    timestamp=current_time,
                    market_regime=self._infer_market_regime(task_id, performance_history),
                    new_task_introduced=causing_task,
                    recovery_possible=retention_rate > 0.1, #
                    severity_level=severity
                )
                
                catastrophic_events.append(event)
                self.forgetting_history.append(event)
        
        if catastrophic_events and self.performance_alerts:
            self._send_forgetting_alert(catastrophic_events)
        
        return catastrophic_events
    
    def _infer_market_regime(self, task_id: int, performance_history: Dict[int, Dict[str, float]]) -> str:
        """
        Output market regime for tasks ()
        
        Args:
            task_id: ID tasks
            performance_history: History performance
            
        Returns:
             market regime
        """
        # Simple on basis performance
        task_performance = performance_history.get(task_id, {}).get("accuracy", 0.5)
        
        if task_performance > 0.8:
            return "bull" # High performance = market
        elif task_performance < 0.4:
            return "volatile" # Low performance = market
        elif 0.6 <= task_performance <= 0.8:
            return "sideways" # Average performance = trend
        else:
            return "bear" # Rest = bear market
    
    def _send_forgetting_alert(self, events: List[ForgettingEvent]) -> None:
        """
            forgetting
        
        Performance degradation alerting
        
        Args:
            events: List events forgetting
        """
        severity_counts = {}
        for event in events:
            severity_counts[event.severity_level] = severity_counts.get(event.severity_level, 0) + 1
        
        alert_message = f"Forgetting Alert: {len(events)} events detected. "
        alert_message += ", ".join([f"{count} {severity}" for severity, count in severity_counts.items()])
        
        self.logger.warning(alert_message)
        
        # possible with systems monitoring
        # : send_to_monitoring_system(alert_message, events)
    
    def analyze_temporal_forgetting_patterns(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Analysis temporal patterns forgetting
        
        Temporal performance analysis
        
        Args:
            performance_history: History performance
            time_window_hours: Temporal window for analysis (hours)
            
        Returns:
            Analysis temporal patterns
        """
        if not self.temporal_analysis_enabled:
            return {}
        
        current_time = datetime.now()
        time_window = timedelta(hours=time_window_hours)
        
        # events forgetting by time
        recent_events = [
            event for event in self.forgetting_history
            if (current_time - event.timestamp) <= time_window
        ]
        
        if not recent_events:
            return {"status": "no_recent_forgetting", "window_hours": time_window_hours}
        
        # Analysis patterns
        patterns = {
            "total_events": len(recent_events),
            "severity_distribution": {},
            "affected_tasks": list(set(event.task_id for event in recent_events)),
            "market_regime_impact": {},
            "average_forgetting_magnitude": 0.0,
            "recovery_potential": 0.0,
            "trend": "stable"
        }
        
        # Distribution by severity
        for event in recent_events:
            severity = event.severity_level
            patterns["severity_distribution"][severity] = patterns["severity_distribution"].get(severity, 0) + 1
        
        # market regimes
        for event in recent_events:
            regime = event.market_regime
            if regime not in patterns["market_regime_impact"]:
                patterns["market_regime_impact"][regime] = {"count": 0, "avg_magnitude": 0.0}
            
            patterns["market_regime_impact"][regime]["count"] += 1
            patterns["market_regime_impact"][regime]["avg_magnitude"] += event.forgetting_magnitude
        
        # Normalization average values
        for regime_data in patterns["market_regime_impact"].values():
            if regime_data["count"] > 0:
                regime_data["avg_magnitude"] /= regime_data["count"]
        
        # General metrics
        magnitudes = [event.forgetting_magnitude for event in recent_events]
        recoverable_events = [event for event in recent_events if event.recovery_possible]
        
        patterns["average_forgetting_magnitude"] = np.mean(magnitudes) if magnitudes else 0.0
        patterns["recovery_potential"] = len(recoverable_events) / len(recent_events) if recent_events else 0.0
        
        # Trend ( analysis)
        if len(recent_events) >= 3:
            recent_magnitudes = [event.forgetting_magnitude for event in recent_events[-3:]]
            early_magnitudes = [event.forgetting_magnitude for event in recent_events[:3]]
            
            if np.mean(recent_magnitudes) > np.mean(early_magnitudes) * 1.2:
                patterns["trend"] = "worsening"
            elif np.mean(recent_magnitudes) < np.mean(early_magnitudes) * 0.8:
                patterns["trend"] = "improving"
            else:
                patterns["trend"] = "stable"
        
        return patterns
    
    def calculate_market_regime_specific_forgetting(
        self, 
        performance_history: Dict[int, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analysis forgetting by market regimes
        
        Market regime aware evaluation
        
        Args:
            performance_history: History performance
            
        Returns:
            Statistics forgetting by market regimes
        """
        if not self.market_regime_analysis:
            return {}
        
        regime_stats = {
            "bull": {"count": 0, "avg_forgetting": 0.0, "max_forgetting": 0.0},
            "bear": {"count": 0, "avg_forgetting": 0.0, "max_forgetting": 0.0},
            "sideways": {"count": 0, "avg_forgetting": 0.0, "max_forgetting": 0.0},
            "volatile": {"count": 0, "avg_forgetting": 0.0, "max_forgetting": 0.0}
        }
        
        # Grouping events by regimes
        for event in self.forgetting_history:
            regime = event.market_regime
            if regime in regime_stats:
                regime_stats[regime]["count"] += 1
                regime_stats[regime]["avg_forgetting"] += event.forgetting_magnitude
                regime_stats[regime]["max_forgetting"] = max(
                    regime_stats[regime]["max_forgetting"],
                    event.forgetting_magnitude
                )
        
        # Normalization average values
        for regime, stats in regime_stats.items():
            if stats["count"] > 0:
                stats["avg_forgetting"] /= stats["count"]
        
        # Save for monitoring
        self.regime_forgetting_stats = regime_stats
        
        return regime_stats
    
    def predict_future_forgetting_risk(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        look_ahead_tasks: int = 3
    ) -> Dict[str, Any]:
        """
        Prediction risk forgetting for future tasks
        
        Predictive performance analysis
        
        Args:
            performance_history: History performance
            look_ahead_tasks: Number tasks for forecast
            
        Returns:
            Forecast risk forgetting
        """
        if not self.forgetting_prediction:
            return {}
        
        # Simple model forecast on basis historical data
        if len(self.forgetting_history) < 3:
            return {"status": "insufficient_data", "minimum_required": 3}
        
        # Analysis trend forgetting
        recent_events = self.forgetting_history[-5:] # Latest 5 events
        forgetting_magnitudes = [event.forgetting_magnitude for event in recent_events]
        
        # Linear regression for trend
        if len(forgetting_magnitudes) >= 2:
            x = np.arange(len(forgetting_magnitudes))
            coeffs = np.polyfit(x, forgetting_magnitudes, 1)
            slope = coeffs[0]  # Trend
            
            # Forecast for future tasks
            predictions = []
            base_magnitude = forgetting_magnitudes[-1]
            
            for i in range(1, look_ahead_tasks + 1):
                predicted_magnitude = base_magnitude + slope * i
                predicted_magnitude = max(0.0, predicted_magnitude) # Not can be negative
                predictions.append(predicted_magnitude)
            
            # Classification risk
            avg_predicted_magnitude = np.mean(predictions)
            if avg_predicted_magnitude > 0.3:
                risk_level = "high"
            elif avg_predicted_magnitude > 0.1:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            prediction_result = {
                "risk_level": risk_level,
                "predicted_magnitudes": predictions,
                "average_predicted_magnitude": avg_predicted_magnitude,
                "trend_slope": slope,
                "confidence": min(len(self.forgetting_history) / 10, 1.0), # More data = more confidence
                "recommendations": self._generate_risk_recommendations(risk_level)
            }
        else:
            prediction_result = {
                "status": "insufficient_trend_data",
                "available_events": len(recent_events)
            }
        
        return prediction_result
    
    def _generate_risk_recommendations(self, risk_level: str) -> List[str]:
        """
        Generation recommendations on basis level risk
        
        Args:
            risk_level: Level risk (low, medium, high)
            
        Returns:
            List recommendations
        """
        recommendations = []
        
        if risk_level == "high":
            recommendations.extend([
                "Consider increasing EWC lambda or replay ratio",
                "Reduce learning rate for new tasks",
                "Implement more aggressive experience replay",
                "Monitor performance closely after each new task"
            ])
        elif risk_level == "medium":
            recommendations.extend([
                "Monitor forgetting metrics more frequently",
                "Consider adjusting continual learning strategy parameters",
                "Prepare rollback plans for critical tasks"
            ])
        else:  # low
            recommendations.extend([
                "Current continual learning strategy is performing well",
                "Continue monitoring but no immediate changes needed"
            ])
        
        return recommendations
    
    def export_forgetting_report(self, filepath: str) -> bool:
        """
        Export detailed report forgetting
        
        Args:
            filepath: Path for saving report
            
        Returns:
            True if export successful
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_forgetting_events": len(self.forgetting_history),
                "baseline_performances": self.baseline_performances,
                "regime_forgetting_stats": self.regime_forgetting_stats,
                "recent_events": [
                    {
                        "task_id": event.task_id,
                        "forgetting_magnitude": event.forgetting_magnitude,
                        "severity_level": event.severity_level,
                        "market_regime": event.market_regime,
                        "timestamp": event.timestamp.isoformat(),
                        "recovery_possible": event.recovery_possible
                    }
                    for event in self.forgetting_history[-20:] # Latest 20 events
                ],
                "configuration": {
                    "catastrophic_threshold": self.catastrophic_threshold,
                    "mild_forgetting_threshold": self.mild_forgetting_threshold,
                    "market_regime_analysis": self.market_regime_analysis,
                    "performance_alerts": self.performance_alerts
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Forgetting report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting forgetting report: {e}")
            return False
    
    def get_comprehensive_forgetting_analysis(
        self, 
        performance_history: Dict[int, Dict[str, float]],
        reference_metric: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis all aspects forgetting
        
        Args:
            performance_history: History performance
            reference_metric: Metric for analysis
            
        Returns:
            Full analysis forgetting
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "metric_used": reference_metric,
            "num_tasks": len(performance_history)
        }
        
        # Main metrics
        analysis["backward_transfer"] = self.calculate_backward_transfer(
            performance_history, reference_metric
        )
        analysis["forgetting_measure"] = self.calculate_forgetting_measure(
            performance_history, reference_metric
        )
        analysis["retention_rates"] = self.calculate_retention_rate(
            performance_history, reference_metric
        )
        
        # analysis
        catastrophic_events = self.detect_catastrophic_forgetting(
            performance_history, reference_metric
        )
        analysis["catastrophic_events"] = len(catastrophic_events)
        analysis["catastrophic_event_details"] = [
            {
                "task_id": event.task_id,
                "severity": event.severity_level,
                "magnitude": event.forgetting_magnitude
            }
            for event in catastrophic_events
        ]
        
        # Temporal analysis
        analysis["temporal_patterns"] = self.analyze_temporal_forgetting_patterns(
            performance_history
        )
        
        # analysis
        analysis["market_regime_analysis"] = self.calculate_market_regime_specific_forgetting(
            performance_history
        )
        
        # Forecast
        analysis["future_risk_prediction"] = self.predict_future_forgetting_risk(
            performance_history
        )
        
        # General recommendations
        analysis["overall_assessment"] = self._generate_overall_assessment(analysis)
        
        return analysis
    
    def _generate_overall_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generation general evaluation state forgetting
        
        Args:
            analysis: Results analysis
            
        Returns:
            Total evaluation
        """
        assessment = {"status": "unknown", "priority": "low", "actions": []}
        
        # Analysis main metrics
        bwt = analysis.get("backward_transfer", 0.0)
        fm = analysis.get("forgetting_measure", 0.0)
        catastrophic_count = analysis.get("catastrophic_events", 0)
        
        # Determine status
        if bwt < -0.2 or fm > 0.3 or catastrophic_count > 0:
            assessment["status"] = "critical"
            assessment["priority"] = "high"
        elif bwt < -0.1 or fm > 0.15:
            assessment["status"] = "concerning"
            assessment["priority"] = "medium"
        elif bwt > 0.0 and fm < 0.05:
            assessment["status"] = "excellent"
            assessment["priority"] = "low"
        else:
            assessment["status"] = "acceptable"
            assessment["priority"] = "low"
        
        # Generation
        if assessment["status"] in ["critical", "concerning"]:
            assessment["actions"].extend([
                "Review continual learning strategy",
                "Consider increasing regularization",
                "Analyze task interference patterns",
                "Implement recovery procedures"
            ])
        
        return assessment
    
    def __repr__(self) -> str:
        return (
            f"ForgettingMetrics("
            f"events={len(self.forgetting_history)}, "
            f"tasks_tracked={len(self.baseline_performances)}, "
            f"catastrophic_threshold={self.catastrophic_threshold})"
        )