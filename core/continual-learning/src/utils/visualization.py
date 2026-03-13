"""
Visualization utilities for Continual Learning in Crypto Trading Bot v5.0

Enterprise-grade system visualization metrics continual training
with integration for monitoring.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from dataclasses import asdict

# Attempt plotly for
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# enterprise colors
THEME_COLORS = {
    'primary': '#2E86C1',
    'success': '#28B463', 
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'info': '#17A2B8',
    'dark': '#343A40',
    'light': '#F8F9FA',
    'accent': '#8E44AD'
}

# Market regime colors
REGIME_COLORS = {
    'bull': '#28B463',    # Green
    'bear': '#E74C3C',    # Red
    'sideways': '#F39C12', # Orange
    'volatile': '#8E44AD'  # Purple
}


class ContinualLearningVisualizer:
    """
    System visualization for continual training
    
    enterprise Features:
    - Interactive performance dashboards
    - Market regime aware visualizations
    - Real-time monitoring charts
    - Export capabilities for reports
    - Mobile-responsive layouts
    """
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("ContinualLearningVisualizer")
        
        # Configure
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # style for matplotlib
        self.setup_matplotlib_style()
    
    def setup_matplotlib_style(self) -> None:
        """Configure for matplotlib"""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 16,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'font.family': 'sans-serif',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def plot_learning_curve(
        self,
        learning_history: List[Tuple[int, float]],
        task_name: str = "Task",
        metric_name: str = "Accuracy",
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Optional[str]:
        """
        Build curve training
        
        Args:
            learning_history: History training [(epoch, metric), ...]
            task_name: Name tasks
            metric_name: Name metrics
            save_path: Path for saving
            interactive: Use interactive chart
            
        Returns:
            Path to saved chart
        """
        if not learning_history:
            self.logger.warning("Empty learning history provided")
            return None
        
        epochs, metrics = zip(*learning_history)
        
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_learning_curve(
                epochs, metrics, task_name, metric_name, save_path
            )
        else:
            return self._plot_static_learning_curve(
                epochs, metrics, task_name, metric_name, save_path
            )
    
    def _plot_interactive_learning_curve(
        self,
        epochs: List[int],
        metrics: List[float],
        task_name: str,
        metric_name: str,
        save_path: Optional[str]
    ) -> str:
        """ curve training with Plotly"""
        fig = go.Figure()
        
        # Main curve
        fig.add_trace(go.Scatter(
            x=epochs,
            y=metrics,
            mode='lines+markers',
            name=f'{task_name} {metric_name}',
            line=dict(color=THEME_COLORS['primary'], width=2),
            marker=dict(size=6)
        ))
        
        # Smoothed curve (moving average)
        if len(metrics) > 5:
            window_size = min(5, len(metrics) // 3)
            smoothed = pd.Series(metrics).rolling(window=window_size).mean()
            fig.add_trace(go.Scatter(
                x=epochs,
                y=smoothed,
                mode='lines',
                name=f'Smoothed {metric_name}',
                line=dict(color=THEME_COLORS['success'], width=3, dash='dash'),
                opacity=0.7
            ))
        
        # Configure layout
        fig.update_layout(
            title=f'{task_name} Learning Curve',
            xaxis_title='Epoch',
            yaxis_title=metric_name,
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / f"{task_name}_learning_curve.html")
        
        fig.write_html(save_path)
        self.logger.info(f"Interactive learning curve saved to {save_path}")
        return save_path
    
    def _plot_static_learning_curve(
        self,
        epochs: List[int],
        metrics: List[float],
        task_name: str,
        metric_name: str,
        save_path: Optional[str]
    ) -> str:
        """ curve training with matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Main curve
        ax.plot(epochs, metrics, 'o-', color=THEME_COLORS['primary'], 
                linewidth=2, markersize=6, label=f'{task_name} {metric_name}')
        
        # Smoothed curve
        if len(metrics) > 5:
            window_size = min(5, len(metrics) // 3)
            smoothed = pd.Series(metrics).rolling(window=window_size).mean()
            ax.plot(epochs, smoothed, '--', color=THEME_COLORS['success'], 
                   linewidth=3, alpha=0.7, label=f'Smoothed {metric_name}')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{task_name} Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / f"{task_name}_learning_curve.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static learning curve saved to {save_path}")
        return save_path
    
    def plot_forgetting_analysis(
        self,
        forgetting_data: Dict[str, Any],
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Optional[str]:
        """
        Visualization analysis forgetting
        
        Args:
            forgetting_data: Data analysis forgetting
            save_path: Path for saving
            interactive: Use interactive chart
            
        Returns:
            Path to saved chart
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._plot_interactive_forgetting_analysis(forgetting_data, save_path)
        else:
            return self._plot_static_forgetting_analysis(forgetting_data, save_path)
    
    def _plot_interactive_forgetting_analysis(
        self,
        forgetting_data: Dict[str, Any],
        save_path: Optional[str]
    ) -> str:
        """Interactive analysis forgetting"""
        # Create
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Backward Transfer Over Time',
                'Retention Rates by Task',
                'Catastrophic Events Timeline',
                'Market Regime Impact'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Backward Transfer
        bwt_history = forgetting_data.get('backward_transfer_history', [])
        if bwt_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(bwt_history))),
                    y=bwt_history,
                    mode='lines+markers',
                    name='Backward Transfer',
                    line=dict(color=THEME_COLORS['primary'])
                ),
                row=1, col=1
            )
        
        # 2. Retention Rates
        retention_rates = forgetting_data.get('retention_rates', {})
        if retention_rates:
            tasks = list(retention_rates.keys())
            rates = list(retention_rates.values())
            
            fig.add_trace(
                go.Bar(
                    x=[f"Task {t}" for t in tasks],
                    y=rates,
                    name='Retention Rate',
                    marker_color=THEME_COLORS['success']
                ),
                row=1, col=2
            )
        
        # 3. Catastrophic Events
        catastrophic_events = forgetting_data.get('catastrophic_event_details', [])
        if catastrophic_events:
            tasks = [event['task_id'] for event in catastrophic_events]
            magnitudes = [event['magnitude'] for event in catastrophic_events]
            severities = [event['severity'] for event in catastrophic_events]
            
            color_map = {'mild': THEME_COLORS['info'], 'moderate': THEME_COLORS['warning'], 
                        'severe': THEME_COLORS['danger'], 'catastrophic': '#8B0000'}
            colors = [color_map.get(sev, THEME_COLORS['dark']) for sev in severities]
            
            fig.add_trace(
                go.Scatter(
                    x=tasks,
                    y=magnitudes,
                    mode='markers',
                    name='Catastrophic Events',
                    marker=dict(color=colors, size=10),
                    text=severities,
                    hovertemplate='Task: %{x}<br>Magnitude: %{y}<br>Severity: %{text}'
                ),
                row=2, col=1
            )
        
        # 4. Market Regime Impact
        regime_analysis = forgetting_data.get('market_regime_analysis', {})
        if regime_analysis:
            regimes = list(regime_analysis.keys())
            avg_forgetting = [regime_analysis[regime]['avg_forgetting'] for regime in regimes]
            
            colors = [REGIME_COLORS.get(regime, THEME_COLORS['dark']) for regime in regimes]
            
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=avg_forgetting,
                    name='Avg Forgetting',
                    marker_color=colors
                ),
                row=2, col=2
            )
        
        # Layout
        fig.update_layout(
            height=800,
            title_text="Continual Learning Forgetting Analysis",
            template='plotly_white',
            showlegend=True
        )
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / "forgetting_analysis.html")
        
        fig.write_html(save_path)
        self.logger.info(f"Interactive forgetting analysis saved to {save_path}")
        return save_path
    
    def _plot_static_forgetting_analysis(
        self,
        forgetting_data: Dict[str, Any],
        save_path: Optional[str]
    ) -> str:
        """Static analysis forgetting"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Continual Learning Forgetting Analysis', fontsize=16)
        
        # 1. Backward Transfer
        bwt_history = forgetting_data.get('backward_transfer_history', [])
        if bwt_history:
            axes[0, 0].plot(bwt_history, 'o-', color=THEME_COLORS['primary'])
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Backward Transfer Over Time')
            axes[0, 0].set_ylabel('BWT Score')
        
        # 2. Retention Rates
        retention_rates = forgetting_data.get('retention_rates', {})
        if retention_rates:
            tasks = list(retention_rates.keys())
            rates = list(retention_rates.values())
            
            bars = axes[0, 1].bar([f"Task {t}" for t in tasks], rates, 
                                 color=THEME_COLORS['success'], alpha=0.7)
            axes[0, 1].set_title('Retention Rates by Task')
            axes[0, 1].set_ylabel('Retention Rate')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Catastrophic Events
        catastrophic_events = forgetting_data.get('catastrophic_event_details', [])
        if catastrophic_events:
            tasks = [event['task_id'] for event in catastrophic_events]
            magnitudes = [event['magnitude'] for event in catastrophic_events]
            severities = [event['severity'] for event in catastrophic_events]
            
            color_map = {'mild': THEME_COLORS['info'], 'moderate': THEME_COLORS['warning'], 
                        'severe': THEME_COLORS['danger'], 'catastrophic': '#8B0000'}
            colors = [color_map.get(sev, THEME_COLORS['dark']) for sev in severities]
            
            axes[1, 0].scatter(tasks, magnitudes, c=colors, s=100, alpha=0.7)
            axes[1, 0].set_title('Catastrophic Events')
            axes[1, 0].set_xlabel('Task ID')
            axes[1, 0].set_ylabel('Forgetting Magnitude')
        
        # 4. Market Regime Impact
        regime_analysis = forgetting_data.get('market_regime_analysis', {})
        if regime_analysis:
            regimes = list(regime_analysis.keys())
            avg_forgetting = [regime_analysis[regime]['avg_forgetting'] for regime in regimes]
            
            colors = [REGIME_COLORS.get(regime, THEME_COLORS['dark']) for regime in regimes]
            axes[1, 1].bar(regimes, avg_forgetting, color=colors, alpha=0.7)
            axes[1, 1].set_title('Market Regime Impact')
            axes[1, 1].set_ylabel('Avg Forgetting')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / "forgetting_analysis.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static forgetting analysis saved to {save_path}")
        return save_path
    
    def plot_plasticity_dashboard(
        self,
        plasticity_data: Dict[str, Any],
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Optional[str]:
        """
        Create dashboard plasticity
        
        Args:
            plasticity_data: Data analysis plasticity
            save_path: Path for saving
            interactive: Use interactive chart
            
        Returns:
            Path to saved dashboard
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_plasticity_dashboard(plasticity_data, save_path)
        else:
            return self._create_static_plasticity_dashboard(plasticity_data, save_path)
    
    def _create_interactive_plasticity_dashboard(
        self,
        plasticity_data: Dict[str, Any],
        save_path: Optional[str]
    ) -> str:
        """Interactive dashboard plasticity"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Forward Transfer Trend',
                'Learning Efficiency by Task',
                'Knowledge Transfer Matrix',
                'Difficulty Prediction Accuracy',
                'Task Completion Times',
                'Performance by Market Regime'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Forward Transfer Trend
        fwt_history = plasticity_data.get('forward_transfer_history', [])
        if fwt_history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(fwt_history))),
                    y=fwt_history,
                    mode='lines+markers',
                    name='Forward Transfer',
                    line=dict(color=THEME_COLORS['success'])
                ),
                row=1, col=1
            )
        
        # 2. Learning Efficiency
        learning_efficiencies = plasticity_data.get('learning_efficiencies', {})
        if learning_efficiencies:
            tasks = list(learning_efficiencies.keys())
            efficiencies = [learning_efficiencies[task].get('overall_efficiency', 0) for task in tasks]
            
            fig.add_trace(
                go.Bar(
                    x=[f"Task {t}" for t in tasks],
                    y=efficiencies,
                    name='Learning Efficiency',
                    marker_color=THEME_COLORS['info']
                ),
                row=1, col=2
            )
        
        # 3. Knowledge Transfer Matrix ( version)
        transfer_analysis = plasticity_data.get('knowledge_transfer_analysis', {})
        if 'top_knowledge_sources' in transfer_analysis:
            sources = transfer_analysis['top_knowledge_sources'][:5] # Top 5
            if sources:
                source_ids, counts = zip(*sources)
                
                fig.add_trace(
                    go.Bar(
                        x=[f"Task {sid}" for sid in source_ids],
                        y=counts,
                        name='Knowledge Sources',
                        marker_color=THEME_COLORS['accent']
                    ),
                    row=2, col=1
                )
        
        # 4. Difficulty Distribution
        difficulty_dist = plasticity_data.get('difficulty_distribution', {})
        if difficulty_dist:
            difficulties = list(difficulty_dist.keys())
            counts = list(difficulty_dist.values())
            
            colors = [THEME_COLORS['success'] if d == 'easy' else 
                     THEME_COLORS['warning'] if d == 'medium' else 
                     THEME_COLORS['danger'] for d in difficulties]
            
            fig.add_trace(
                go.Pie(
                    labels=difficulties,
                    values=counts,
                    name="Difficulty Distribution",
                    marker_colors=colors
                ),
                row=2, col=2
            )
        
        # 5. Completion Times (if there is data time)
        # Placeholder - possible data time execution
        
        # 6. Performance by Market Regime
        regime_performance = plasticity_data.get('regime_performance', {})
        if regime_performance:
            regimes = list(regime_performance.keys())
            avg_perfs = [regime_performance[regime]['avg_performance'] for regime in regimes]
            
            colors = [REGIME_COLORS.get(regime, THEME_COLORS['dark']) for regime in regimes]
            
            fig.add_trace(
                go.Bar(
                    x=regimes,
                    y=avg_perfs,
                    name='Avg Performance',
                    marker_color=colors
                ),
                row=3, col=2
            )
        
        # Layout
        fig.update_layout(
            height=1200,
            title_text="Continual Learning Plasticity Dashboard",
            template='plotly_white',
            showlegend=True
        )
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / "plasticity_dashboard.html")
        
        fig.write_html(save_path)
        self.logger.info(f"Interactive plasticity dashboard saved to {save_path}")
        return save_path
    
    def _create_static_plasticity_dashboard(
        self,
        plasticity_data: Dict[str, Any],
        save_path: Optional[str]
    ) -> str:
        """Static dashboard plasticity"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Continual Learning Plasticity Dashboard', fontsize=16)
        
        # 1. Forward Transfer Trend
        fwt_history = plasticity_data.get('forward_transfer_history', [])
        if fwt_history:
            axes[0, 0].plot(fwt_history, 'o-', color=THEME_COLORS['success'])
            axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Forward Transfer Trend')
            axes[0, 0].set_ylabel('FWT Score')
        
        # 2. Learning Efficiency
        learning_efficiencies = plasticity_data.get('learning_efficiencies', {})
        if learning_efficiencies:
            tasks = list(learning_efficiencies.keys())
            efficiencies = [learning_efficiencies[task].get('overall_efficiency', 0) for task in tasks]
            
            axes[0, 1].bar([f"T{t}" for t in tasks], efficiencies, 
                          color=THEME_COLORS['info'], alpha=0.7)
            axes[0, 1].set_title('Learning Efficiency by Task')
            axes[0, 1].set_ylabel('Efficiency Score')
        
        # 3. Knowledge Sources
        transfer_analysis = plasticity_data.get('knowledge_transfer_analysis', {})
        if 'top_knowledge_sources' in transfer_analysis:
            sources = transfer_analysis['top_knowledge_sources'][:5]
            if sources:
                source_ids, counts = zip(*sources)
                axes[1, 0].bar([f"Task {sid}" for sid in source_ids], counts, 
                              color=THEME_COLORS['accent'], alpha=0.7)
                axes[1, 0].set_title('Top Knowledge Sources')
                axes[1, 0].set_ylabel('Usage Count')
        
        # 4. Difficulty Distribution
        difficulty_dist = plasticity_data.get('difficulty_distribution', {})
        if difficulty_dist:
            difficulties = list(difficulty_dist.keys())
            counts = list(difficulty_dist.values())
            
            colors = [THEME_COLORS['success'] if d == 'easy' else 
                     THEME_COLORS['warning'] if d == 'medium' else 
                     THEME_COLORS['danger'] for d in difficulties]
            
            axes[1, 1].pie(counts, labels=difficulties, colors=colors, autopct='%1.1f%%')
            axes[1, 1].set_title('Task Difficulty Distribution')
        
        # 5. Plasticity Trends
        trends = plasticity_data.get('plasticity_trends', {})
        if trends:
            recent_avg = trends.get('recent_avg_performance', 0)
            early_avg = trends.get('early_avg_performance', 0)
            
            axes[2, 0].bar(['Early Tasks', 'Recent Tasks'], [early_avg, recent_avg],
                          color=[THEME_COLORS['primary'], THEME_COLORS['success']])
            axes[2, 0].set_title('Performance Evolution')
            axes[2, 0].set_ylabel('Average Performance')
        
        # 6. Market Regime Performance
        regime_performance = plasticity_data.get('regime_performance', {})
        if regime_performance:
            regimes = list(regime_performance.keys())
            avg_perfs = [regime_performance[regime]['avg_performance'] for regime in regimes]
            
            colors = [REGIME_COLORS.get(regime, THEME_COLORS['dark']) for regime in regimes]
            axes[2, 1].bar(regimes, avg_perfs, color=colors, alpha=0.7)
            axes[2, 1].set_title('Performance by Market Regime')
            axes[2, 1].set_ylabel('Average Performance')
        
        plt.tight_layout()
        
        # Save
        if save_path is None:
            save_path = str(self.output_dir / "plasticity_dashboard.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Static plasticity dashboard saved to {save_path}")
        return save_path
    
    def create_comprehensive_report(
        self,
        continual_learning_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Create report by training
        
        Args:
            continual_learning_data: All data continual training
            save_path: Path for saving HTML report
            
        Returns:
            Path to
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.output_dir / f"continual_learning_report_{timestamp}.html")
        
        # Create individual visualizations
        visualizations = {}
        
        # Learning curves for each tasks
        if 'task_learning_curves' in continual_learning_data:
            for task_id, curve_data in continual_learning_data['task_learning_curves'].items():
                viz_path = self.plot_learning_curve(
                    curve_data, f"Task_{task_id}", interactive=True
                )
                visualizations[f"learning_curve_task_{task_id}"] = viz_path
        
        # Forgetting analysis
        if 'forgetting_analysis' in continual_learning_data:
            viz_path = self.plot_forgetting_analysis(
                continual_learning_data['forgetting_analysis'], interactive=True
            )
            visualizations["forgetting_analysis"] = viz_path
        
        # Plasticity dashboard
        if 'plasticity_analysis' in continual_learning_data:
            viz_path = self.plot_plasticity_dashboard(
                continual_learning_data['plasticity_analysis'], interactive=True
            )
            visualizations["plasticity_dashboard"] = viz_path
        
        # Create HTML report
        html_content = self._generate_html_report(continual_learning_data, visualizations)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Comprehensive report saved to {save_path}")
        return save_path
    
    def _generate_html_report(
        self,
        data: Dict[str, Any],
        visualizations: Dict[str, str]
    ) -> str:
        """Generation HTML report"""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Continual Learning Report - Crypto Trading Bot v5.0</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #343a40;
                }}
                
                .header {{
                    background: linear-gradient(135deg, {THEME_COLORS['primary']}, {THEME_COLORS['accent']});
                    color: white;
                    padding: 30px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                    text-align: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                .section {{
                    background: white;
                    margin: 20px 0;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .metric-card {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid {THEME_COLORS['primary']};
                }}
                
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: {THEME_COLORS['primary']};
                    margin-bottom: 5px;
                }}
                
                .metric-label {{
                    color: #6c757d;
                    font-weight: 500;
                }}
                
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                
                .visualization iframe {{
                    width: 100%;
                    height: 600px;
                    border: none;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Continual Learning Report</h1>
                    <p>Crypto Trading Bot v5.0 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>enterprise Machine Learning Analytics</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-value">{data.get('num_tasks', 'N/A')}</div>
                            <div class="metric-label">Tasks Completed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{data.get('forgetting_analysis', {}).get('backward_transfer', 0):.3f}</div>
                            <div class="metric-label">Backward Transfer</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{data.get('plasticity_analysis', {}).get('forward_transfer', 0):.3f}</div>
                            <div class="metric-label">Forward Transfer</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{data.get('forgetting_analysis', {}).get('catastrophic_events', 0)}</div>
                            <div class="metric-label">Catastrophic Events</div>
                        </div>
                    </div>
                </div>
        """
        
        # Add visualizations
        if "plasticity_dashboard" in visualizations:
            html_template += f"""
                <div class="section">
                    <h2>Plasticity Analysis Dashboard</h2>
                    <div class="visualization">
                        <iframe src="{Path(visualizations['plasticity_dashboard']).name}"></iframe>
                    </div>
                </div>
            """
        
        if "forgetting_analysis" in visualizations:
            html_template += f"""
                <div class="section">
                    <h2>Forgetting Analysis</h2>
                    <div class="visualization">
                        <iframe src="{Path(visualizations['forgetting_analysis']).name}"></iframe>
                    </div>
                </div>
            """
        
        # Learning curves for tasks
        learning_curve_sections = ""
        for viz_key, viz_path in visualizations.items():
            if viz_key.startswith("learning_curve_task_"):
                task_id = viz_key.split("_")[-1]
                learning_curve_sections += f"""
                    <div class="section">
                        <h3>Task {task_id} Learning Curve</h3>
                        <div class="visualization">
                            <iframe src="{Path(viz_path).name}"></iframe>
                        </div>
                    </div>
                """
        
        if learning_curve_sections:
            html_template += f"""
                <div class="section">
                    <h2>Individual Task Performance</h2>
                    {learning_curve_sections}
                </div>
            """
        
        html_template += """
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def export_metrics_to_csv(
        self,
        metrics_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> str:
        """
        Export metrics in CSV format
        
        Args:
            metrics_data: Data metrics
            save_path: Path for saving
            
        Returns:
            Path to CSV file
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = str(self.output_dir / f"continual_learning_metrics_{timestamp}.csv")
        
        # Transform data in DataFrame
        rows = []
        
        # Main metrics
        if 'forgetting_analysis' in metrics_data:
            fa = metrics_data['forgetting_analysis']
            rows.append({
                'timestamp': datetime.now().isoformat(),
                'metric_type': 'forgetting',
                'metric_name': 'backward_transfer',
                'value': fa.get('backward_transfer', 0),
                'task_id': None,
                'market_regime': None
            })
            
            rows.append({
                'timestamp': datetime.now().isoformat(),
                'metric_type': 'forgetting',
                'metric_name': 'forgetting_measure',
                'value': fa.get('forgetting_measure', 0),
                'task_id': None,
                'market_regime': None
            })
        
        if 'plasticity_analysis' in metrics_data:
            pa = metrics_data['plasticity_analysis']
            rows.append({
                'timestamp': datetime.now().isoformat(),
                'metric_type': 'plasticity',
                'metric_name': 'forward_transfer',
                'value': pa.get('forward_transfer', 0),
                'task_id': None,
                'market_regime': None
            })
        
        # Metrics by tasks
        if 'learning_efficiencies' in metrics_data:
            for task_id, efficiency_data in metrics_data['learning_efficiencies'].items():
                for metric_name, value in efficiency_data.items():
                    if isinstance(value, (int, float)):
                        rows.append({
                            'timestamp': datetime.now().isoformat(),
                            'metric_type': 'efficiency',
                            'metric_name': metric_name,
                            'value': value,
                            'task_id': task_id,
                            'market_regime': None
                        })
        
        # Create DataFrame and saving
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
        self.logger.info(f"Metrics exported to CSV: {save_path}")
        return save_path
    
    def __repr__(self) -> str:
        return (
            f"ContinualLearningVisualizer("
            f"output_dir='{self.output_dir}', "
            f"plotly_available={PLOTLY_AVAILABLE})"
        )