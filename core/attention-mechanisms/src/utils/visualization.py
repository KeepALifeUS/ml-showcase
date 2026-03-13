"""
Visualization utilities for attention mechanisms in crypto trading models.
Creates heatmaps, attention flow diagrams and interactive visualizations.

Production visualization tools for attention analysis and model debugging.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import numpy as np
import logging
from dataclasses import dataclass
from pathlib import Path

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Some visualization features disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Interactive visualization features disabled.")

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    cmap: str = "viridis"
    save_format: str = "png"
    save_dir: str = "./attention_plots"
    show_plots: bool = True
    interactive: bool = True


class AttentionHeatmapVisualizer:
    """
    Creates heatmap visualizations for attention weights.
    
    Features:
    - Multi-head attention heatmaps
    - Layer comparison heatmaps
    - Temporal attention patterns
    - Cross-attention visualizations
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available. Heatmap visualization disabled.")
    
    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        title: str = "Attention Heatmap",
        token_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Create attention heatmap visualization.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len] or [num_heads, seq_len, seq_len]
            title: Plot title
            token_labels: Labels for sequence positions
            save_path: Path for saving plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for heatmap visualization")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        # Handle different dimensions
        if weights_np.ndim == 3:  # [num_heads, seq_len, seq_len]
            num_heads, seq_len, _ = weights_np.shape
            fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
            
            if num_heads == 1:
                axes = [axes]
            
            for head_idx, ax in enumerate(axes):
                im = ax.imshow(
                    weights_np[head_idx],
                    cmap=self.config.cmap,
                    aspect='auto',
                    interpolation='nearest'
                )
                ax.set_title(f'Head {head_idx + 1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
                # Set token labels if provided
                if token_labels:
                    ax.set_xticks(range(len(token_labels)))
                    ax.set_yticks(range(len(token_labels)))
                    ax.set_xticklabels(token_labels, rotation=45)
                    ax.set_yticklabels(token_labels)
            
            fig.suptitle(title)
            
        elif weights_np.ndim == 2:  # [seq_len, seq_len]
            fig, ax = plt.subplots(figsize=self.config.figsize)
            
            im = ax.imshow(
                weights_np,
                cmap=self.config.cmap,
                aspect='auto',
                interpolation='nearest'
            )
            
            ax.set_title(title)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Set token labels if provided
            if token_labels:
                ax.set_xticks(range(len(token_labels)))
                ax.set_yticks(range(len(token_labels)))
                ax.set_xticklabels(token_labels, rotation=45)
                ax.set_yticklabels(token_labels)
        
        else:
            logger.error(f"Unsupported attention weights shape: {weights_np.shape}")
            return None
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Attention heatmap saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_multi_layer_attention(
        self,
        attention_weights_list: List[torch.Tensor],
        layer_names: Optional[List[str]] = None,
        title: str = "Multi-Layer Attention Comparison",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot attention weights from multiple layers for comparison.
        
        Args:
            attention_weights_list: List of attention weight tensors
            layer_names: Names of layers
            title: Plot title
            save_path: Path for saving plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for multi-layer visualization")
            return None
        
        num_layers = len(attention_weights_list)
        
        if layer_names is None:
            layer_names = [f"Layer {i+1}" for i in range(num_layers)]
        
        fig, axes = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
        
        if num_layers == 1:
            axes = [axes]
        
        for layer_idx, (weights, layer_name, ax) in enumerate(zip(attention_weights_list, layer_names, axes)):
            # Convert to numpy and take average over heads if needed
            if isinstance(weights, torch.Tensor):
                weights_np = weights.detach().cpu().numpy()
            else:
                weights_np = weights
            
            if weights_np.ndim == 3:  # Average over heads
                weights_np = weights_np.mean(axis=0)
            
            im = ax.imshow(
                weights_np,
                cmap=self.config.cmap,
                aspect='auto',
                interpolation='nearest'
            )
            
            ax.set_title(layer_name)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Multi-layer attention plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_attention_evolution(
        self,
        attention_sequence: List[torch.Tensor],
        timestamps: Optional[List[str]] = None,
        title: str = "Attention Pattern Evolution",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot evolution of attention patterns over time.
        
        Args:
            attention_sequence: Sequence of attention weight tensors
            timestamps: Timestamps for each attention pattern
            title: Plot title
            save_path: Path for saving plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for evolution visualization")
            return None
        
        num_steps = len(attention_sequence)
        cols = min(5, num_steps)
        rows = math.ceil(num_steps / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        
        if num_steps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for step_idx, weights in enumerate(attention_sequence):
            row = step_idx // cols
            col = step_idx % cols
            
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            else:
                ax = axes[row, col]
            
            # Convert to numpy and average over heads
            if isinstance(weights, torch.Tensor):
                weights_np = weights.detach().cpu().numpy()
            else:
                weights_np = weights
            
            if weights_np.ndim == 3:
                weights_np = weights_np.mean(axis=0)
            
            im = ax.imshow(
                weights_np,
                cmap=self.config.cmap,
                aspect='auto',
                interpolation='nearest'
            )
            
            # Set title
            if timestamps:
                ax.set_title(f"t={timestamps[step_idx]}")
            else:
                ax.set_title(f"Step {step_idx + 1}")
            
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide empty subplots
        for step_idx in range(num_steps, rows * cols):
            row = step_idx // cols
            col = step_idx % cols
            if rows == 1:
                ax = axes[col] if cols > 1 else axes
            else:
                ax = axes[row, col]
            ax.set_visible(False)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Attention evolution plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        
        return fig


class InteractiveAttentionVisualizer:
    """
    Interactive attention visualizations using Plotly.
    
    Features:
    - Interactive heatmaps with zoom/pan
    - 3D attention visualizations  
    - Animated attention evolution
    - Multi-modal attention analysis
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Interactive visualization disabled.")
    
    def create_interactive_heatmap(
        self,
        attention_weights: torch.Tensor,
        title: str = "Interactive Attention Heatmap",
        token_labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create interactive attention heatmap.
        
        Args:
            attention_weights: Attention weights tensor
            title: Plot title  
            token_labels: Labels for tokens
            save_path: Path for saving HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for interactive visualization")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        # Handle multi-head attention
        if weights_np.ndim == 3:
            num_heads = weights_np.shape[0]
            
            # Create subplots for each head
            fig = make_subplots(
                rows=1, cols=num_heads,
                subplot_titles=[f'Head {i+1}' for i in range(num_heads)],
                horizontal_spacing=0.05
            )
            
            for head_idx in range(num_heads):
                heatmap = go.Heatmap(
                    z=weights_np[head_idx],
                    colorscale='Viridis',
                    showscale=(head_idx == num_heads - 1),  # Show colorbar only for last head
                    x=token_labels,
                    y=token_labels
                )
                
                fig.add_trace(heatmap, row=1, col=head_idx + 1)
            
            fig.update_layout(
                title=title,
                height=400,
                width=300 * num_heads
            )
            
        else:  # 2D attention
            fig = go.Figure(data=go.Heatmap(
                z=weights_np,
                colorscale='Viridis',
                x=token_labels,
                y=token_labels
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Key Position',
                yaxis_title='Query Position',
                width=600,
                height=600
            )
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Interactive heatmap saved to {save_path}")
        
        if self.config.show_plots:
            fig.show()
        
        return fig
    
    def create_3d_attention_visualization(
        self,
        attention_weights: torch.Tensor,
        title: str = "3D Attention Visualization",
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create 3D visualization of attention patterns.
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            title: Plot title
            save_path: Path for saving HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for 3D visualization")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        if weights_np.ndim != 3:
            logger.error("3D visualization requires 3D attention weights [num_heads, seq_len, seq_len]")
            return None
        
        num_heads, seq_len, _ = weights_np.shape
        
        # Create 3D surface plot
        fig = go.Figure()
        
        # Create coordinate grids
        x = np.arange(seq_len)
        y = np.arange(seq_len)
        
        for head_idx in range(num_heads):
            fig.add_trace(go.Surface(
                z=weights_np[head_idx],
                x=x,
                y=y,
                name=f'Head {head_idx + 1}',
                colorscale='Viridis',
                opacity=0.7,
                visible=(head_idx == 0)  # Show only first head initially
            ))
        
        # Add buttons for switching between heads
        buttons = []
        for head_idx in range(num_heads):
            visibility = [False] * num_heads
            visibility[head_idx] = True
            
            buttons.append(dict(
                args=[{"visible": visibility}],
                label=f"Head {head_idx + 1}",
                method="restyle"
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Key Position',
                yaxis_title='Query Position',
                zaxis_title='Attention Weight'
            ),
            updatemenus=[dict(
                type="buttons",
                direction="left",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.02,
                yanchor="top"
            )]
        )
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"3D attention visualization saved to {save_path}")
        
        if self.config.show_plots:
            fig.show()
        
        return fig
    
    def create_attention_flow_diagram(
        self,
        attention_weights: torch.Tensor,
        token_labels: List[str],
        threshold: float = 0.1,
        title: str = "Attention Flow Diagram",
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create attention flow diagram showing connections between tokens.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len]
            token_labels: Labels for tokens
            threshold: Minimum attention weight to show connection
            title: Plot title
            save_path: Path for saving HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for flow diagram")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        if weights_np.ndim == 3:  # Average over heads
            weights_np = weights_np.mean(axis=0)
        
        seq_len = weights_np.shape[0]
        
        # Create node positions (circular layout)
        angles = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)
        node_x = np.cos(angles)
        node_y = np.sin(angles)
        
        # Create edges for strong connections
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j and weights_np[i, j] > threshold:
                    # Add edge
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
                    edge_weights.append(weights_np[i, j])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=token_labels,
            textposition="middle center",
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=20,
                color=np.sum(weights_np, axis=1),  # Node color based on total attention received
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Total Attention"
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Attention connections above threshold",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Attention flow diagram saved to {save_path}")
        
        if self.config.show_plots:
            fig.show()
        
        return fig


class CryptoAttentionVisualizer:
    """
    Specialized visualization for crypto trading attention patterns.
    
    Features:
    - Price-attention correlation plots
    - Temporal attention patterns
    - Multi-asset attention analysis
    - Risk-attention relationship visualization
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.heatmap_viz = AttentionHeatmapVisualizer(config)
        self.interactive_viz = InteractiveAttentionVisualizer(config)
    
    def plot_price_attention_correlation(
        self,
        attention_weights: torch.Tensor,
        price_data: torch.Tensor,
        timestamps: List[str],
        title: str = "Price-Attention Correlation",
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot correlation between attention patterns and price movements.
        
        Args:
            attention_weights: Attention weights [seq_len, seq_len]
            price_data: Price data [seq_len]
            timestamps: Timestamps for each data point
            title: Plot title
            save_path: Path for saving plot
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available for correlation plot")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        if isinstance(price_data, torch.Tensor):
            prices_np = price_data.detach().cpu().numpy()
        else:
            prices_np = price_data
        
        # Calculate attention focus (entropy)
        attention_entropy = []
        for i in range(weights_np.shape[0]):
            row = weights_np[i]
            entropy = -np.sum(row * np.log(row + 1e-8))
            attention_entropy.append(entropy)
        
        attention_entropy = np.array(attention_entropy)
        
        # Calculate price changes
        price_changes = np.diff(prices_np, prepend=prices_np[0])
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Price data
        ax1.plot(prices_np, label='Price', color='blue')
        ax1.set_title('Price Movement')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attention entropy
        ax2.plot(attention_entropy, label='Attention Entropy', color='red')
        ax2.set_title('Attention Focus (Entropy)')
        ax2.set_ylabel('Entropy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Correlation
        ax3.scatter(attention_entropy, price_changes, alpha=0.6)
        ax3.set_xlabel('Attention Entropy')
        ax3.set_ylabel('Price Change')
        ax3.set_title('Price Change vs Attention Entropy')
        ax3.grid(True, alpha=0.3)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(attention_entropy, price_changes)[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Price-attention correlation plot saved to {save_path}")
        
        if self.config.show_plots:
            plt.show()
        
        return fig
    
    def plot_multi_asset_attention(
        self,
        attention_weights: torch.Tensor,
        asset_names: List[str],
        title: str = "Multi-Asset Attention Patterns",
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Visualize attention patterns across multiple crypto assets.
        
        Args:
            attention_weights: Cross-asset attention weights [num_assets, num_assets]
            asset_names: Names of crypto assets
            title: Plot title
            save_path: Path for saving HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for multi-asset visualization")
            return None
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            weights_np = attention_weights.detach().cpu().numpy()
        else:
            weights_np = attention_weights
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=weights_np,
            x=asset_names,
            y=asset_names,
            colorscale='RdYlBu_r',
            hoverongaps=False,
            hovertemplate='<b>%{y}</b> → <b>%{x}</b><br>Attention: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Target Asset',
            yaxis_title='Source Asset',
            width=600,
            height=600
        )
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Multi-asset attention plot saved to {save_path}")
        
        if self.config.show_plots:
            fig.show()
        
        return fig
    
    def create_comprehensive_attention_dashboard(
        self,
        attention_data: Dict[str, torch.Tensor],
        price_data: Optional[torch.Tensor] = None,
        timestamps: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> Optional[go.Figure]:
        """
        Create comprehensive dashboard with multiple attention visualizations.
        
        Args:
            attention_data: Dictionary with different attention patterns
            price_data: Price data for correlation analysis
            timestamps: Timestamps for data
            save_path: Path for saving HTML file
        """
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available for dashboard")
            return None
        
        # Calculate number of subplots needed
        num_plots = len(attention_data)
        if price_data is not None:
            num_plots += 1
        
        rows = math.ceil(num_plots / 2)
        cols = min(2, num_plots)
        
        # Create subplots
        subplot_titles = list(attention_data.keys())
        if price_data is not None:
            subplot_titles.append("Price Data")
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Add attention heatmaps
        plot_idx = 0
        for name, weights in attention_data.items():
            row = (plot_idx // cols) + 1
            col = (plot_idx % cols) + 1
            
            # Convert to numpy
            if isinstance(weights, torch.Tensor):
                weights_np = weights.detach().cpu().numpy()
            else:
                weights_np = weights
            
            # Handle multi-dimensional weights
            if weights_np.ndim == 3:
                weights_np = weights_np.mean(axis=0)  # Average over heads
            
            heatmap = go.Heatmap(
                z=weights_np,
                colorscale='Viridis',
                showscale=False
            )
            
            fig.add_trace(heatmap, row=row, col=col)
            plot_idx += 1
        
        # Add price data if provided
        if price_data is not None:
            row = (plot_idx // cols) + 1
            col = (plot_idx % cols) + 1
            
            if isinstance(price_data, torch.Tensor):
                prices_np = price_data.detach().cpu().numpy()
            else:
                prices_np = price_data
            
            price_trace = go.Scatter(
                y=prices_np,
                mode='lines',
                name='Price',
                x=timestamps if timestamps else None
            )
            
            fig.add_trace(price_trace, row=row, col=col)
        
        fig.update_layout(
            title="Comprehensive Attention Analysis Dashboard",
            height=400 * rows,
            showlegend=False
        )
        
        # Save if requested
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            logger.info(f"Attention dashboard saved to {save_path}")
        
        if self.config.show_plots:
            fig.show()
        
        return fig


def create_visualization_report(
    attention_data: Dict[str, torch.Tensor],
    output_dir: str = "./attention_report",
    report_title: str = "Attention Mechanisms Analysis Report"
) -> str:
    """
    Create comprehensive visualization report.
    
    Args:
        attention_data: Dictionary with attention patterns
        output_dir: Output directory for report
        report_title: Title of the report
        
    Returns:
        Path to generated report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualizers
    config = VisualizationConfig(save_dir=str(output_path), show_plots=False)
    heatmap_viz = AttentionHeatmapVisualizer(config)
    interactive_viz = InteractiveAttentionVisualizer(config)
    
    # Generate visualizations
    generated_files = []
    
    for name, weights in attention_data.items():
        # Static heatmap
        static_path = output_path / f"{name}_heatmap.png"
        heatmap_viz.plot_attention_heatmap(
            weights, 
            title=f"{name} Attention Pattern",
            save_path=str(static_path)
        )
        generated_files.append(static_path)
        
        # Interactive heatmap
        interactive_path = output_path / f"{name}_interactive.html"
        interactive_viz.create_interactive_heatmap(
            weights,
            title=f"Interactive {name} Attention",
            save_path=str(interactive_path)
        )
        generated_files.append(interactive_path)
    
    # Create HTML report
    report_path = output_path / "attention_report.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .section {{ margin-bottom: 40px; }}
            .visualization {{ margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>{report_title}</h1>
        <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>Attention Pattern Analysis</h2>
    """
    
    for name, weights in attention_data.items():
        html_content += f"""
            <div class="visualization">
                <h3>{name} Attention Pattern</h3>
                <img src="{name}_heatmap.png" alt="{name} Heatmap">
                <p><a href="{name}_interactive.html" target="_blank">View Interactive Version</a></p>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Attention analysis report generated: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing Attention Visualization:")
    
    # Create test attention data
    batch_size, num_heads, seq_len = 2, 4, 32
    
    # Multi-head attention weights
    attention_weights = torch.softmax(
        torch.randn(num_heads, seq_len, seq_len),
        dim=-1
    )
    
    # Test static heatmap visualization
    if MATPLOTLIB_AVAILABLE:
        print("Testing static heatmap visualization:")
        config = VisualizationConfig(show_plots=False)
        heatmap_viz = AttentionHeatmapVisualizer(config)
        
        token_labels = [f"Token_{i}" for i in range(seq_len)]
        fig = heatmap_viz.plot_attention_heatmap(
            attention_weights,
            title="Test Attention Heatmap",
            token_labels=token_labels
        )
        
        if fig is not None:
            print("  ✓ Static heatmap created successfully")
        else:
            print("  ✗ Failed to create static heatmap")
    
    # Test interactive visualization
    if PLOTLY_AVAILABLE:
        print("Testing interactive visualization:")
        interactive_viz = InteractiveAttentionVisualizer(config)
        
        fig = interactive_viz.create_interactive_heatmap(
            attention_weights,
            title="Test Interactive Heatmap",
            token_labels=token_labels
        )
        
        if fig is not None:
            print("  ✓ Interactive heatmap created successfully")
        else:
            print("  ✗ Failed to create interactive heatmap")
        
        # Test 3D visualization
        fig_3d = interactive_viz.create_3d_attention_visualization(
            attention_weights.unsqueeze(0),  # Add batch dimension
            title="Test 3D Visualization"
        )
        
        if fig_3d is not None:
            print("  ✓ 3D visualization created successfully")
        else:
            print("  ✗ Failed to create 3D visualization")
    
    # Test crypto-specific visualization
    print("Testing crypto-specific visualization:")
    crypto_viz = CryptoAttentionVisualizer(config)
    
    # Test price-attention correlation
    if MATPLOTLIB_AVAILABLE:
        price_data = torch.cumsum(torch.randn(seq_len) * 0.01, dim=0) + 100
        timestamps = [f"t_{i}" for i in range(seq_len)]
        
        fig = crypto_viz.plot_price_attention_correlation(
            attention_weights.mean(0),  # Average over heads
            price_data,
            timestamps,
            title="Test Price-Attention Correlation"
        )
        
        if fig is not None:
            print("  ✓ Price-attention correlation plot created successfully")
        else:
            print("  ✗ Failed to create price-attention correlation plot")
    
    # Test multi-asset attention
    if PLOTLY_AVAILABLE:
        num_assets = 5
        asset_names = ["BTC", "ETH", "ADA", "DOT", "LINK"]
        multi_asset_attention = torch.softmax(torch.randn(num_assets, num_assets), dim=-1)
        
        fig = crypto_viz.plot_multi_asset_attention(
            multi_asset_attention,
            asset_names,
            title="Test Multi-Asset Attention"
        )
        
        if fig is not None:
            print("  ✓ Multi-asset attention plot created successfully")
        else:
            print("  ✗ Failed to create multi-asset attention plot")
    
    # Test comprehensive dashboard
    if PLOTLY_AVAILABLE:
        attention_data = {
            "Self-Attention": attention_weights.mean(0),
            "Cross-Attention": torch.softmax(torch.randn(seq_len, seq_len), dim=-1),
            "Temporal-Attention": torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        }
        
        fig = crypto_viz.create_comprehensive_attention_dashboard(
            attention_data,
            price_data=price_data,
            timestamps=timestamps
        )
        
        if fig is not None:
            print("  ✓ Comprehensive dashboard created successfully")
        else:
            print("  ✗ Failed to create comprehensive dashboard")
    
    print(f"\n✅ Attention visualization testing complete!")
    
    # Print availability status
    print(f"\nVisualization Libraries Status:")
    print(f"  Matplotlib: {'✓ Available' if MATPLOTLIB_AVAILABLE else '✗ Not Available'}")
    print(f"  Plotly: {'✓ Available' if PLOTLY_AVAILABLE else '✗ Not Available'}")