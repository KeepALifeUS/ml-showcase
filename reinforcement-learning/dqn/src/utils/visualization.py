"""
Training Visualization utilities for DQN with .

Comprehensive visualization system for DQN training:
- Real-time training curves
- Q-value evolution analysis
- Action distribution plots
- Performance comparison charts
- Interactive exploration plots
- Financial metrics visualization
- Distribution analysis for categorical DQN
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import structlog
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = structlog.get_logger(__name__)


class TrainingVisualizer:
 """
 Comprehensive training visualization system with enterprise functionality.

 Features:
 - Real-time training curves with multiple metrics
 - Q-value analysis and distribution plots
 - Action exploration visualization
 - Performance comparison between algorithms
 - Financial metrics plots for trading
 - Interactive Plotly dashboards
 - Automatic report generation
 - Export in multiple formats
 """

 def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
 """
 Initialization visualizer.

 Args:
 style: Matplotlib style
 figsize: Default figure size
 """
 self.style = style
 self.figsize = figsize

 # Set plotting style
 plt.style.use('default') # Use default since seaborn styles might not be available
 sns.set_palette("husl")

 # Color schemes
 self.colors = {
 'primary': '#1f77b4',
 'secondary': '#ff7f0e',
 'success': '#2ca02c',
 'danger': '#d62728',
 'warning': '#ff9800',
 'info': '#17a2b8',
 'muted': '#6c757d'
 }

 self.logger = structlog.get_logger(__name__).bind(component="TrainingVisualizer")
 self.logger.info("Training Visualizer initialized")

 def plot_training_curves(self,
 metrics_history: Dict[str, List[float]],
 title: str = "Training Curves",
 save_path: Optional[str] = None,
 show_plot: bool = True) -> plt.Figure:
 """
 Building training curves.

 Args:
 metrics_history: History metrics {metric_name: [values]}
 title: Headers chart
 save_path: Path for saving
 show_plot: Show chart

 Returns:
 Matplotlib figure
 """
 fig, axes = plt.subplots(2, 2, figsize=(15, 10))
 fig.suptitle(title, fontsize=16, fontweight='bold')

 # Reward curve
 if 'episode_rewards' in metrics_history:
 rewards = metrics_history['episode_rewards']
 episodes = range(len(rewards))

 axes[0, 0].plot(episodes, rewards, color=self.colors['primary'], alpha=0.7)

 # Moving average
 if len(rewards) > 10:
 window_size = min(50, len(rewards) // 10)
 ma_rewards = pd.Series(rewards).rolling(window=window_size).mean
 axes[0, 0].plot(episodes, ma_rewards, color=self.colors['danger'],
 linewidth=2, label=f'MA({window_size})')
 axes[0, 0].legend

 axes[0, 0].set_title('Episode Rewards')
 axes[0, 0].set_xlabel('Episode')
 axes[0, 0].set_ylabel('Reward')
 axes[0, 0].grid(True, alpha=0.3)

 # Loss curve
 if 'losses' in metrics_history:
 losses = metrics_history['losses']
 steps = range(len(losses))

 axes[0, 1].plot(steps, losses, color=self.colors['warning'], alpha=0.8)
 axes[0, 1].set_title('Training Loss')
 axes[0, 1].set_xlabel('Training Step')
 axes[0, 1].set_ylabel('Loss')
 axes[0, 1].grid(True, alpha=0.3)

 # Log scale if loss values strongly differ
 if max(losses) / min([l for l in losses if l > 0] + [1]) > 100:
 axes[0, 1].set_yscale('log')

 # Epsilon decay
 if 'epsilon' in metrics_history:
 epsilons = metrics_history['epsilon']
 steps = range(len(epsilons))

 axes[1, 0].plot(steps, epsilons, color=self.colors['success'])
 axes[1, 0].set_title('Epsilon Decay')
 axes[1, 0].set_xlabel('Training Step')
 axes[1, 0].set_ylabel('Epsilon')
 axes[1, 0].grid(True, alpha=0.3)

 # Q-values
 if 'q_values' in metrics_history:
 q_values = metrics_history['q_values']
 steps = range(len(q_values))

 axes[1, 1].plot(steps, q_values, color=self.colors['info'])
 axes[1, 1].set_title('Average Q-Values')
 axes[1, 1].set_xlabel('Training Step')
 axes[1, 1].set_ylabel('Q-Value')
 axes[1, 1].grid(True, alpha=0.3)

 plt.tight_layout

 if save_path:
 fig.savefig(save_path, dpi=300, bbox_inches='tight')
 self.logger.info("Training curves saved", path=save_path)

 if show_plot:
 plt.show

 return fig

 def plot_performance_comparison(self,
 results: Dict[str, Dict[str, List[float]]],
 metric: str = 'episode_rewards',
 title: Optional[str] = None,
 save_path: Optional[str] = None) -> plt.Figure:
 """
 Comparison performance different algorithms.

 Args:
 results: {algorithm_name: {metric: [values]}}
 metric: Metric for comparison
 title: Headers
 save_path: Path for saving

 Returns:
 Figure
 """
 if title is None:
 title = f"Performance Comparison - {metric.replace('_', ' ').title}"

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 fig.suptitle(title, fontsize=16, fontweight='bold')

 colors = plt.cm.Set3(np.linspace(0, 1, len(results)))

 # Curves comparison
 for i, (algo_name, metrics) in enumerate(results.items):
 if metric in metrics:
 values = metrics[metric]
 episodes = range(len(values))

 ax1.plot(episodes, values, label=algo_name,
 color=colors[i], alpha=0.7, linewidth=2)

 # Moving average
 if len(values) > 20:
 window_size = min(50, len(values) // 10)
 ma_values = pd.Series(values).rolling(window=window_size).mean
 ax1.plot(episodes, ma_values, color=colors[i],
 linewidth=3, alpha=0.9)

 ax1.set_title(f'{metric.replace("_", " ").title} Over Time')
 ax1.set_xlabel('Episode')
 ax1.set_ylabel(metric.replace('_', ' ').title)
 ax1.legend
 ax1.grid(True, alpha=0.3)

 # Box plot comparison
 box_data = []
 labels = []

 for algo_name, metrics in results.items:
 if metric in metrics:
 values = metrics[metric]
 if len(values) > 0:
 # Take last 20% values for final performance
 final_values = values[-max(1, len(values) // 5):]
 box_data.append(final_values)
 labels.append(algo_name)

 if box_data:
 bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)

 # Color boxes
 for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)

 ax2.set_title('Final Performance Distribution')
 ax2.set_ylabel(metric.replace('_', ' ').title)
 ax2.grid(True, alpha=0.3)

 plt.tight_layout

 if save_path:
 fig.savefig(save_path, dpi=300, bbox_inches='tight')
 self.logger.info("Performance comparison saved", path=save_path)

 return fig

 def plot_action_distribution(self,
 action_history: List[int],
 action_names: Optional[List[str]] = None,
 title: str = "Action Distribution",
 save_path: Optional[str] = None) -> plt.Figure:
 """
 Visualization distribution actions.

 Args:
 action_history: History actions
 action_names: Names actions
 title: Headers
 save_path: Path saving

 Returns:
 Figure
 """
 if not action_history:
 self.logger.warning("Empty action history")
 return plt.figure

 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
 fig.suptitle(title, fontsize=16, fontweight='bold')

 unique_actions, counts = np.unique(action_history, return_counts=True)
 total_actions = len(action_history)

 # Action frequency bar plot
 if action_names:
 labels = [action_names[i] if i < len(action_names) else f"Action {i}"
 for i in unique_actions]
 else:
 labels = [f"Action {i}" for i in unique_actions]

 bars = ax1.bar(labels, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique_actions))))

 # Add percentage labels on bars
 for bar, count in zip(bars, counts):
 height = bar.get_height
 ax1.text(bar.get_x + bar.get_width/2., height + 0.01*max(counts),
 f'{count}\n({100*count/total_actions:.1f}%)',
 ha='center', va='bottom', fontweight='bold')

 ax1.set_title('Action Frequency')
 ax1.set_ylabel('Count')
 ax1.grid(True, alpha=0.3, axis='y')

 # Action evolution over time
 window_size = max(100, len(action_history) // 50)
 action_evolution = []

 for i in range(window_size, len(action_history), window_size):
 window_actions = action_history[i-window_size:i]
 action_dist = np.zeros(max(unique_actions) + 1)

 for action in window_actions:
 action_dist[action] += 1

 action_dist = action_dist / len(window_actions)
 action_evolution.append(action_dist)

 if action_evolution:
 action_evolution = np.array(action_evolution)
 time_steps = range(len(action_evolution))

 for action_idx in unique_actions:
 ax2.plot(time_steps, action_evolution[:, action_idx],
 label=labels[list(unique_actions).index(action_idx)],
 linewidth=2)

 ax2.set_title('Action Distribution Evolution')
 ax2.set_xlabel('Time Window')
 ax2.set_ylabel('Action Probability')
 ax2.legend
 ax2.grid(True, alpha=0.3)

 plt.tight_layout

 if save_path:
 fig.savefig(save_path, dpi=300, bbox_inches='tight')
 self.logger.info("Action distribution saved", path=save_path)

 return fig

 def plot_financial_metrics(self,
 returns: List[float],
 portfolio_values: Optional[List[float]] = None,
 benchmark_returns: Optional[List[float]] = None,
 title: str = "Financial Performance",
 save_path: Optional[str] = None) -> plt.Figure:
 """
 Financial performance visualization.

 Args:
 returns: Portfolio returns
 portfolio_values: Portfolio values over time
 benchmark_returns: Benchmark returns for comparison
 title: Chart title
 save_path: Save path

 Returns:
 Figure
 """
 fig = plt.figure(figsize=(16, 12))
 gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

 fig.suptitle(title, fontsize=16, fontweight='bold')

 if not returns:
 self.logger.warning("No returns data provided")
 return fig

 returns_array = np.array(returns)

 # Cumulative returns
 ax1 = fig.add_subplot(gs[0, :])
 cumulative_returns = np.cumprod(1 + returns_array)
 time_index = range(len(cumulative_returns))

 ax1.plot(time_index, cumulative_returns,
 color=self.colors['primary'], linewidth=2, label='Portfolio')

 if benchmark_returns:
 bench_cumulative = np.cumprod(1 + np.array(benchmark_returns))
 ax1.plot(time_index, bench_cumulative,
 color=self.colors['muted'], linewidth=2,
 linestyle='--', label='Benchmark')

 ax1.set_title('Cumulative Returns')
 ax1.set_ylabel('Cumulative Return')
 ax1.legend
 ax1.grid(True, alpha=0.3)

 # Portfolio value (if provided)
 ax2 = fig.add_subplot(gs[1, 0])
 if portfolio_values:
 ax2.plot(range(len(portfolio_values)), portfolio_values,
 color=self.colors['success'], linewidth=2)
 ax2.set_title('Portfolio Value')
 ax2.set_ylabel('Value')
 ax2.grid(True, alpha=0.3)
 else:
 ax2.text(0.5, 0.5, 'No Portfolio Values',
 transform=ax2.transAxes, ha='center', va='center')
 ax2.set_title('Portfolio Value (N/A)')

 # Drawdown analysis
 ax3 = fig.add_subplot(gs[1, 1])
 running_max = np.maximum.accumulate(cumulative_returns)
 drawdowns = (cumulative_returns - running_max) / running_max

 ax3.fill_between(time_index, drawdowns, 0,
 color=self.colors['danger'], alpha=0.7)
 ax3.set_title(f'Drawdown (Max: {np.min(drawdowns):.2%})')
 ax3.set_ylabel('Drawdown')
 ax3.grid(True, alpha=0.3)

 # Returns distribution
 ax4 = fig.add_subplot(gs[2, 0])
 ax4.hist(returns_array, bins=50, density=True, alpha=0.7,
 color=self.colors['info'], edgecolor='black')

 # Add normal distribution overlay
 mu, sigma = np.mean(returns_array), np.std(returns_array)
 x_norm = np.linspace(np.min(returns_array), np.max(returns_array), 100)
 y_norm = ((2 * np.pi * sigma**2) ** -0.5) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
 ax4.plot(x_norm, y_norm, 'r--', linewidth=2, label='Normal Dist')

 ax4.set_title('Returns Distribution')
 ax4.set_xlabel('Return')
 ax4.set_ylabel('Density')
 ax4.legend
 ax4.grid(True, alpha=0.3)

 # Risk metrics table
 ax5 = fig.add_subplot(gs[2, 1])
 ax5.axis('off')

 # Calculate metrics
 total_return = (cumulative_returns[-1] - 1) * 100
 volatility = np.std(returns_array) * np.sqrt(252) * 100 # Annualized
 sharpe_ratio = (np.mean(returns_array) / np.std(returns_array)) * np.sqrt(252) if np.std(returns_array) > 0 else 0
 max_drawdown = np.min(drawdowns) * 100

 metrics_text = f"""
 Financial Metrics:

 Total Return: {total_return:.2f}%
 Volatility: {volatility:.2f}%
 Sharpe Ratio: {sharpe_ratio:.3f}
 Max Drawdown: {max_drawdown:.2f}%

 Win Rate: {np.mean(returns_array > 0) * 100:.1f}%
 Best Day: {np.max(returns_array) * 100:.2f}%
 Worst Day: {np.min(returns_array) * 100:.2f}%
 """

 ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
 fontsize=11, verticalalignment='top', fontfamily='monospace',
 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

 if save_path:
 fig.savefig(save_path, dpi=300, bbox_inches='tight')
 self.logger.info("Financial metrics saved", path=save_path)

 return fig

 def plot_distributional_analysis(self,
 q_distributions: np.ndarray,
 support_values: np.ndarray,
 actions: Optional[List[str]] = None,
 title: str = "Q-Value Distributions",
 save_path: Optional[str] = None) -> plt.Figure:
 """
 Visualization for distributional DQN.

 Args:
 q_distributions: Q-value distributions [num_actions, num_atoms]
 support_values: Support values [num_atoms]
 actions: Action names
 title: Title
 save_path: Save path

 Returns:
 Figure
 """
 num_actions, num_atoms = q_distributions.shape

 if actions is None:
 actions = [f"Action {i}" for i in range(num_actions)]

 fig, axes = plt.subplots(2, 2, figsize=(15, 10))
 fig.suptitle(title, fontsize=16, fontweight='bold')

 colors = plt.cm.viridis(np.linspace(0, 1, num_actions))

 # Individual distributions
 ax1 = axes[0, 0]
 for i in range(num_actions):
 ax1.plot(support_values, q_distributions[i],
 color=colors[i], linewidth=2, label=actions[i])

 ax1.set_title('Q-Value Distributions')
 ax1.set_xlabel('Value')
 ax1.set_ylabel('Probability')
 ax1.legend
 ax1.grid(True, alpha=0.3)

 # Expected values (mean)
 ax2 = axes[0, 1]
 expected_values = np.sum(q_distributions * support_values, axis=1)
 bars = ax2.bar(actions, expected_values, color=colors)

 ax2.set_title('Expected Q-Values')
 ax2.set_ylabel('Q-Value')
 ax2.grid(True, alpha=0.3, axis='y')

 # Add value labels on bars
 for bar, value in zip(bars, expected_values):
 height = bar.get_height
 ax2.text(bar.get_x + bar.get_width/2., height + 0.01*max(expected_values),
 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

 # Uncertainty (standard deviation)
 ax3 = axes[1, 0]
 uncertainties = np.sqrt(np.sum(q_distributions * (support_values - expected_values.reshape(-1, 1))**2, axis=1))
 bars = ax3.bar(actions, uncertainties, color=colors, alpha=0.7)

 ax3.set_title('Q-Value Uncertainty (Std Dev)')
 ax3.set_ylabel('Standard Deviation')
 ax3.grid(True, alpha=0.3, axis='y')

 # Entropy
 ax4 = axes[1, 1]
 entropies = -np.sum(q_distributions * np.log(q_distributions + 1e-8), axis=1)
 bars = ax4.bar(actions, entropies, color=colors, alpha=0.7)

 ax4.set_title('Distribution Entropy')
 ax4.set_ylabel('Entropy')
 ax4.grid(True, alpha=0.3, axis='y')

 plt.tight_layout

 if save_path:
 fig.savefig(save_path, dpi=300, bbox_inches='tight')
 self.logger.info("Distributional analysis saved", path=save_path)

 return fig

 def create_interactive_dashboard(self,
 training_data: Dict[str, Any],
 save_path: Optional[str] = None) -> str:
 """
 Creating interactive Plotly dashboard.

 Args:
 training_data: Data training
 save_path: Path for saving HTML

 Returns:
 HTML string or path to file
 """
 # Create subplots
 fig = make_subplots(
 rows=3, cols=2,
 subplot_titles=('Episode Rewards', 'Training Loss', 'Epsilon Decay',
 'Q-Values', 'Action Distribution', 'Performance Metrics'),
 specs=[[{"secondary_y": False}, {"secondary_y": False}],
 [{"secondary_y": False}, {"secondary_y": False}],
 [{"type": "pie"}, {"type": "table"}]]
 )

 # Episode rewards
 if 'episode_rewards' in training_data:
 rewards = training_data['episode_rewards']
 fig.add_trace(
 go.Scatter(x=list(range(len(rewards))), y=rewards,
 mode='lines', name='Rewards',
 line=dict(color='blue', width=2)),
 row=1, col=1
 )

 # Training loss
 if 'losses' in training_data:
 losses = training_data['losses']
 fig.add_trace(
 go.Scatter(x=list(range(len(losses))), y=losses,
 mode='lines', name='Loss',
 line=dict(color='red', width=2)),
 row=1, col=2
 )

 # Epsilon decay
 if 'epsilon' in training_data:
 epsilons = training_data['epsilon']
 fig.add_trace(
 go.Scatter(x=list(range(len(epsilons))), y=epsilons,
 mode='lines', name='Epsilon',
 line=dict(color='green', width=2)),
 row=2, col=1
 )

 # Q-values
 if 'q_values' in training_data:
 q_values = training_data['q_values']
 fig.add_trace(
 go.Scatter(x=list(range(len(q_values))), y=q_values,
 mode='lines', name='Q-Values',
 line=dict(color='purple', width=2)),
 row=2, col=2
 )

 # Action distribution pie chart
 if 'action_distribution' in training_data:
 action_dist = training_data['action_distribution']
 labels = list(action_dist.keys)
 values = list(action_dist.values)

 fig.add_trace(
 go.Pie(labels=labels, values=values, name="Actions"),
 row=3, col=1
 )

 # Performance metrics table
 if 'performance_metrics' in training_data:
 metrics = training_data['performance_metrics']

 fig.add_trace(
 go.Table(
 header=dict(values=['Metric', 'Value']),
 cells=dict(values=[list(metrics.keys),
 [f"{v:.4f}" if isinstance(v, float) else str(v)
 for v in metrics.values]])
 ),
 row=3, col=2
 )

 # Update layout
 fig.update_layout(
 height=900,
 showlegend=True,
 title_text="DQN Training Dashboard",
 title_x=0.5
 )

 # Save or return
 if save_path:
 fig.write_html(save_path)
 self.logger.info("Interactive dashboard saved", path=save_path)
 return save_path
 else:
 return fig.to_html

 def generate_training_report(self,
 training_data: Dict[str, Any],
 save_dir: str = "./training_report") -> str:
 """
 Generate comprehensive training report.

 Args:
 training_data: All data training
 save_dir: Directory for saving

 Returns:
 Path to generated reportat
 """
 save_path = Path(save_dir)
 save_path.mkdir(parents=True, exist_ok=True)

 timestamp = datetime.now.strftime("%Y%m%d_%H%M%S")

 # Generate plots
 plots_generated = []

 # Training curves
 if any(key in training_data for key in ['episode_rewards', 'losses', 'epsilon', 'q_values']):
 training_fig = self.plot_training_curves(
 training_data,
 save_path=str(save_path / f"training_curves_{timestamp}.png"),
 show_plot=False
 )
 plots_generated.append("training_curves")
 plt.close(training_fig)

 # Financial metrics
 if 'returns' in training_data:
 financial_fig = self.plot_financial_metrics(
 training_data['returns'],
 portfolio_values=training_data.get('portfolio_values'),
 save_path=str(save_path / f"financial_metrics_{timestamp}.png"),
 show_plot=False
 )
 plots_generated.append("financial_metrics")
 plt.close(financial_fig)

 # Action distribution
 if 'action_history' in training_data:
 action_fig = self.plot_action_distribution(
 training_data['action_history'],
 save_path=str(save_path / f"action_distribution_{timestamp}.png"),
 show_plot=False
 )
 plots_generated.append("action_distribution")
 plt.close(action_fig)

 # Interactive dashboard
 dashboard_path = str(save_path / f"dashboard_{timestamp}.html")
 self.create_interactive_dashboard(training_data, dashboard_path)

 # Generate summary report
 report_path = str(save_path / f"training_report_{timestamp}.md")
 self._generate_markdown_report(training_data, plots_generated, report_path)

 self.logger.info("Training report generated",
 report_path=report_path,
 plots_count=len(plots_generated))

 return report_path

 def _generate_markdown_report(self,
 training_data: Dict[str, Any],
 plots_generated: List[str],
 report_path: str) -> None:
 """Generate markdown training report."""
 timestamp = datetime.now.strftime("%Y-%m-%d %H:%M:%S")

 with open(report_path, 'w') as f:
 f.write(f"# DQN Training Report\n\n")
 f.write(f"**Generated:** {timestamp}\n\n")

 # Training summary
 if 'episode_rewards' in training_data:
 rewards = training_data['episode_rewards']
 f.write("## Training Summary\n\n")
 f.write(f"- **Total Episodes:** {len(rewards)}\n")
 f.write(f"- **Final Reward:** {rewards[-1]:.3f}\n")
 f.write(f"- **Best Reward:** {max(rewards):.3f}\n")
 f.write(f"- **Average Reward:** {np.mean(rewards):.3f}\n\n")

 # Performance metrics
 if 'performance_metrics' in training_data:
 metrics = training_data['performance_metrics']
 f.write("## Performance Metrics\n\n")
 for key, value in metrics.items:
 if isinstance(value, float):
 f.write(f"- **{key.replace('_', ' ').title}:** {value:.4f}\n")
 else:
 f.write(f"- **{key.replace('_', ' ').title}:** {value}\n")
 f.write("\n")

 # Generated plots
 if plots_generated:
 f.write("## Generated Visualizations\n\n")
 for plot_name in plots_generated:
 f.write(f"- {plot_name.replace('_', ' ').title}\n")
 f.write("\n")

 # Configuration
 if 'config' in training_data:
 f.write("## Training Configuration\n\n")
 f.write("```yaml\n")
 config = training_data['config']
 for key, value in config.items:
 f.write(f"{key}: {value}\n")
 f.write("```\n\n")

 def __del__(self):
 """Cleanup matplotlib resources."""
 plt.close('all')