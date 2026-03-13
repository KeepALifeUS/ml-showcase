"""
ML Testing Framework - Visualization Utilities
Visualization results ML testing for crypto trading

Enterprise Pattern: Data-Driven Decision Support
- Enterprise visualization for stakeholders technical teams
- Interactive dashboard for monitoring quality ML models
- Automated generation reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
from datetime import datetime
import base64
from io import BytesIO

# For statistical tests
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Setup charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "seaborn"
    color_palette: str = "husl"
    save_format: str = "png"  # png, pdf, svg, html
    interactive: bool = True
    show_plots: bool = False
    save_plots: bool = True
    output_dir: str = "plots"
    
    # Enterprise Settings
    enterprise_theme: bool = True
    include_branding: bool = True
    high_quality: bool = True
    accessibility: bool = True


class MLTestingVisualizer:
    """
    Class for visualization results ML testing
    
    Enterprise Pattern: Executive Reporting & Analytics
    - Create publication-ready charts for reports
 - Interactive dashboard for real-time monitoring
 - Automated visualization KPI metrics quality
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup matplotlib
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['savefig.dpi'] = self.config.dpi
        
        if self.config.enterprise_theme:
            self._setup_enterprise_theme()
    
    def _setup_enterprise_theme(self) -> None:
        """Setup for charts"""
        corporate_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E']
        
        plt.rcParams.update({
            'axes.prop_cycle': plt.cycler(color=corporate_colors),
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.linewidth': 0.8,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10
        })
    
    def plot_model_performance_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        title: str = "Model Performance Metrics Over Time"
    ) -> str:
        """
 Visualization metrics performance models time
        
        Args:
            metrics_history: History metrics {metric_name: [values]}
            title: Header chart
        
        Returns:
 str: Path saved file
        """
        if self.config.interactive:
            return self._plot_interactive_metrics(metrics_history, title)
        else:
            return self._plot_static_metrics(metrics_history, title)
    
    def _plot_interactive_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        title: str
    ) -> str:
        """visualization metrics Plotly"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(metrics_history.keys())[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (metric_name, values) in enumerate(list(metrics_history.items())[:4]):
            row, col = positions[i]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
            
            # Addition trend
            if len(values) > 1:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(values))),
                        y=p(range(len(values))),
                        mode='lines',
                        name=f'{metric_name} trend',
                        line=dict(color=colors[i], dash='dash', width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save
        filename = f"interactive_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def _plot_static_metrics(
        self,
        metrics_history: Dict[str, List[float]],
        title: str
    ) -> str:
        """visualization metrics matplotlib"""
        n_metrics = len(metrics_history)
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (metric_name, values) in enumerate(metrics_history.items()):
            ax = axes[i]
            
            ax.plot(values, marker='o', linewidth=2, markersize=4, label=metric_name)
            
            if len(values) > 1:
                z = np.polyfit(range(len(values)), values, 1)
                p = np.poly1d(z)
                ax.plot(range(len(values)), p(range(len(values))), 
                       '--', alpha=0.7, label=f'{metric_name} trend')
            
            ax.axhline(y=np.mean(values), color='red', linestyle=':', 
                      alpha=0.7, label=f'Mean: {np.mean(values):.3f}')
            
            ax.set_title(f'{metric_name}')
            ax.set_xlabel('Iteration/Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Deletion subplot'
        for i in range(len(metrics_history), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save
        filename = f"static_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.save_format}"
        filepath = self.output_dir / filename
        plt.savefig(str(filepath), dpi=self.config.dpi, bbox_inches='tight')
        
        if not self.config.show_plots:
            plt.close()
        
        return str(filepath)
    
    def plot_data_quality_report(
        self,
        data: pd.DataFrame,
        quality_issues: Dict[str, Any],
        title: str = "Data Quality Assessment"
    ) -> str:
        """
 Visualization report data
        
        Args:
 data: data
 quality_issues: Detected issues quality
            title: Header report
        
        Returns:
 str: Path saved file
        """
        if self.config.interactive:
            return self._plot_interactive_data_quality(data, quality_issues, title)
        else:
            return self._plot_static_data_quality(data, quality_issues, title)
    
    def _plot_interactive_data_quality(
        self,
        data: pd.DataFrame,
        quality_issues: Dict[str, Any],
        title: str
    ) -> str:
        """visualization quality data"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Missing Values", "Data Types", "Outliers", "Correlations"],
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # 1. Missing Values
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        
        if not missing_data.empty:
            fig.add_trace(
                go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    name="Missing Values",
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # 2. Data Types Distribution
        dtype_counts = data.dtypes.astype(str).value_counts()
        fig.add_trace(
            go.Pie(
                labels=dtype_counts.index,
                values=dtype_counts.values,
                name="Data Types"
            ),
            row=1, col=2
        )
        
        # 3. Outliers (for numeric )
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col_name = numeric_cols[0]
            values = data[col_name].dropna()
            
            # Z-score for detection outliers
            z_scores = np.abs(stats.zscore(values))
            outliers = values[z_scores > 3]
            
            fig.add_trace(
                go.Scatter(
                    x=range(len(values)),
                    y=values,
                    mode='markers',
                    name="Normal",
                    marker=dict(color='blue', size=4)
                ),
                row=2, col=1
            )
            
            if len(outliers) > 0:
                outlier_indices = np.where(z_scores > 3)[0]
                fig.add_trace(
                    go.Scatter(
                        x=outlier_indices,
                        y=outliers,
                        mode='markers',
                        name="Outliers",
                        marker=dict(color='red', size=8)
                    ),
                    row=2, col=1
                )
        
        # 4. Correlation Heatmap
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        filename = f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def plot_model_comparison(
        self,
        models_metrics: Dict[str, Dict[str, float]],
        title: str = "Model Comparison"
    ) -> str:
        """
 Comparison multiple models by metric
        
        Args:
            models_metrics: {model_name: {metric: value}}
            title: Header chart
        
        Returns:
 str: Path saved file
        """
        # Transform DataFrame for
        df = pd.DataFrame(models_metrics).T
        
        if self.config.interactive:
            return self._plot_interactive_comparison(df, title)
        else:
            return self._plot_static_comparison(df, title)
    
    def _plot_interactive_comparison(self, df: pd.DataFrame, title: str) -> str:
        """comparison models"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Metrics Radar", "Bar Chart", "Scatter Matrix", "Ranking"],
            specs=[[{"type": "scatterpolar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Radar Chart for first models
        if len(df) > 0:
            first_model = df.iloc[0]
            fig.add_trace(
                go.Scatterpolar(
                    r=first_model.values,
                    theta=first_model.index,
                    fill='toself',
                    name=first_model.name
                ),
                row=1, col=1
            )
        
        # 2. Bar Chart comparison
        for metric in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df[metric],
                    name=metric
                ),
                row=1, col=2
            )
        
        # 3. Scatter plot for metrics
        if len(df.columns) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=df.iloc[:, 0],
                    y=df.iloc[:, 1],
                    mode='markers+text',
                    text=df.index,
                    textposition="top center",
                    marker=dict(size=12),
                    name="Models"
                ),
                row=2, col=1
            )
        
        # 4. Ranking by average indicator
        df_mean = df.mean(axis=1).sort_values(ascending=False)
        fig.add_trace(
            go.Bar(
                x=df_mean.values,
                y=df_mean.index,
                orientation='h',
                name="Average Score"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def plot_drift_analysis(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_scores: Dict[str, float],
        title: str = "Data Drift Analysis"
    ) -> str:
        """
        Visualization analysis drift data
        
        Args:
 reference_data: data
 current_data: data
 drift_scores: drift by feature
            title: Header chart
        
        Returns:
 str: Path saved file
        """
        if self.config.interactive:
            return self._plot_interactive_drift(reference_data, current_data, drift_scores, title)
        else:
            return self._plot_static_drift(reference_data, current_data, drift_scores, title)
    
    def _plot_interactive_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        drift_scores: Dict[str, float],
        title: str
    ) -> str:
        """visualization drift"""
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        n_cols = min(4, len(numeric_cols))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Drift Scores", "Distribution Comparison", 
                           "Feature Evolution", "PCA Comparison"]
        )
        
        # 1. Drift Scores
        drift_df = pd.DataFrame(list(drift_scores.items()), columns=['Feature', 'Drift_Score'])
        fig.add_trace(
            go.Bar(
                x=drift_df['Feature'],
                y=drift_df['Drift_Score'],
                name="Drift Score",
                marker_color=['red' if score > 0.1 else 'green' for score in drift_df['Drift_Score']]
            ),
            row=1, col=1
        )
        
        # 2. Distribution comparison for feature
        if len(numeric_cols) > 0:
            col_name = numeric_cols[0]
            
            fig.add_trace(
                go.Histogram(
                    x=reference_data[col_name],
                    name="Reference",
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Histogram(
                    x=current_data[col_name],
                    name="Current",
                    opacity=0.7,
                    nbinsx=30
                ),
                row=1, col=2
            )
        
        # 3. Feature Evolution (if is temporal labels)
        if 'timestamp' in reference_data.columns and len(numeric_cols) > 0:
            col_name = numeric_cols[0]
            
            # Grouping by time calculation
            ref_time_series = reference_data.groupby(
                pd.Grouper(key='timestamp', freq='D')
            )[col_name].mean()
            
            curr_time_series = current_data.groupby(
                pd.Grouper(key='timestamp', freq='D')
            )[col_name].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=ref_time_series.index,
                    y=ref_time_series.values,
                    name="Reference Trend",
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=curr_time_series.index,
                    y=curr_time_series.values,
                    name="Current Trend",
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # 4. PCA Comparison
        if len(numeric_cols) >= 2:
            pca = PCA(n_components=2)
            
            ref_pca = pca.fit_transform(reference_data[numeric_cols].fillna(0))
            curr_pca = pca.transform(current_data[numeric_cols].fillna(0))
            
            fig.add_trace(
                go.Scatter(
                    x=ref_pca[:, 0],
                    y=ref_pca[:, 1],
                    mode='markers',
                    name="Reference PCA",
                    marker=dict(color='blue', opacity=0.6)
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=curr_pca[:, 0],
                    y=curr_pca[:, 1],
                    mode='markers',
                    name="Current PCA",
                    marker=dict(color='red', opacity=0.6)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        filename = f"drift_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: List[float],
        title: str = "Feature Importance Analysis"
    ) -> str:
        """
        Visualization importance features
        
        Args:
 feature_names: features
 importance_scores: importance
            title: Header chart
        
        Returns:
 str: Path saved file
        """
        # Sorting by importance
        sorted_indices = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_indices]
        sorted_scores = [importance_scores[i] for i in sorted_indices]
        
        if self.config.interactive:
            fig = go.Figure()
            
            # Main bar chart
            fig.add_trace(
                go.Bar(
                    y=sorted_features,
                    x=sorted_scores,
                    orientation='h',
                    marker_color=px.colors.sequential.Viridis[:len(sorted_features)]
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, len(feature_names) * 20),
                template='plotly_white'
            )
            
            filename = f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            filepath = self.output_dir / filename
            fig.write_html(str(filepath))
            
        else:
            plt.figure(figsize=(10, max(6, len(feature_names) * 0.3)))
            bars = plt.barh(range(len(sorted_features)), sorted_scores)
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_features)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.yticks(range(len(sorted_features)), sorted_features)
            plt.xlabel('Importance Score')
            plt.title(title)
            plt.grid(axis='x', alpha=0.3)
            
            # Addition values on bars
            for i, (feature, score) in enumerate(zip(sorted_features, sorted_scores)):
                plt.text(score + 0.01 * max(sorted_scores), i, f'{score:.3f}', 
                        va='center', fontsize=9)
            
            plt.tight_layout()
            
            filename = f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.save_format}"
            filepath = self.output_dir / filename
            plt.savefig(str(filepath), dpi=self.config.dpi, bbox_inches='tight')
            
            if not self.config.show_plots:
                plt.close()
        
        return str(filepath)
    
    def create_ml_testing_dashboard(
        self,
        test_results: Dict[str, Any],
        title: str = "ML Testing Dashboard"
    ) -> str:
        """
 Create dashboard result all tests
        
        Args:
            test_results: Results various tests
            title: Header dashboard
        
        Returns:
 str: Path HTML file dashboard
        """
        # Create dashboard
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                "Model Performance", "Data Quality", "Feature Importance",
                "Drift Detection", "Model Comparison", "Resource Usage",
                "Test Coverage", "Error Analysis", "Recommendations"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # Addition various charts on basis results tests
        # (here can from test_results)
        
        fig.update_layout(
            title=title,
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Save dashboard
        filename = f"ml_testing_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = self.output_dir / filename
        fig.write_html(str(filepath))
        
        return str(filepath)
    
    def export_plots_to_pdf(self, plot_files: List[str], output_name: str = None) -> str:
        """
 multiple charts PDF report
        
        Args:
 plot_files: List file charts
 output_name: Name one PDF file
        
        Returns:
 str: Path PDF file
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        if output_name is None:
            output_name = f"ml_testing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        pdf_path = self.output_dir / output_name
        
        with PdfPages(str(pdf_path)) as pdf:
            for plot_file in plot_files:
                if Path(plot_file).suffix == '.png':
                    img = plt.imread(plot_file)
                    fig, ax = plt.subplots(figsize=self.config.figure_size)
                    ax.imshow(img)
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        return str(pdf_path)
    
    def get_plot_as_base64(self, plot_function, *args, **kwargs) -> str:
        """
 Retrieval chart base64 for report
        
        Args:
            plot_function: Function for creation chart
 *args, **kwargs: for functions
        
        Returns:
 str: Base64
        """
        # saved
        temp_path = plot_function(*args, **kwargs)
        
        if temp_path.endswith('.html'):
            # For HTML files path
            return temp_path
        else:
            # For base64
            with open(temp_path, 'rb') as img_file:
                base64_string = base64.b64encode(img_file.read()).decode()
            
            # file
            Path(temp_path).unlink()
            
            return f"data:image/{self.config.save_format};base64,{base64_string}"


def create_crypto_trading_visualizer(
    interactive: bool = True,
    high_quality: bool = True,
    enterprise_theme: bool = True
) -> MLTestingVisualizer:
    """
    Factory function for creation visualizer for crypto trading
    
    Args:
 interactive: Interactive
 high_quality: quality charts
 enterprise_theme: Enterprise
    
    Returns:
        MLTestingVisualizer: Configured visualizer
    """
    config = VisualizationConfig(
        interactive=interactive,
        high_quality=high_quality,
        enterprise_theme=enterprise_theme,
        figure_size=(14, 10) if high_quality else (10, 6),
        dpi=300 if high_quality else 150,
        save_plots=True,
        show_plots=False
    )
    
    return MLTestingVisualizer(config)


# Example usage
if __name__ == "__main__":
    # Create visualizer for crypto trading
    visualizer = create_crypto_trading_visualizer(
        interactive=True,
        high_quality=True,
        enterprise_theme=True
    )
    
    # Example data
    print("=== Create example ===")
    
    # 1. Metrics performance
    metrics_history = {
        'accuracy': [0.75, 0.78, 0.82, 0.85, 0.87, 0.89],
        'precision': [0.72, 0.76, 0.79, 0.83, 0.85, 0.88],
        'recall': [0.68, 0.72, 0.75, 0.78, 0.81, 0.84],
        'f1_score': [0.70, 0.74, 0.77, 0.80, 0.83, 0.86]
    }
    
    plot1 = visualizer.plot_model_performance_metrics(
        metrics_history,
        "Crypto Trading Model Performance"
    )
    print(f"Performance metrics plot saved: {plot1}")
    
    # 2. Comparison models
    models_metrics = {
        'RandomForest': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1_score': 0.85},
        'XGBoost': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.90, 'f1_score': 0.88},
        'LSTM': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1_score': 0.82},
        'Ensemble': {'accuracy': 0.91, 'precision': 0.89, 'recall': 0.93, 'f1_score': 0.91}
    }
    
    plot2 = visualizer.plot_model_comparison(
        models_metrics,
        "Crypto Trading Models Comparison"
    )
    print(f"Model comparison plot saved: {plot2}")
    
    # 3. features
    feature_names = ['price_change', 'volume', 'rsi', 'macd', 'bollinger', 'sma_20', 'ema_12', 'volatility']
    importance_scores = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    plot3 = visualizer.plot_feature_importance(
        feature_names,
        importance_scores,
        "Feature Importance for Crypto Prediction"
    )
    print(f"Feature importance plot saved: {plot3}")
    
    # 4. Example data for analysis quality
    sample_data = pd.DataFrame({
        'price': np.random.lognormal(10, 0.1, 1000),
        'volume': np.random.exponential(100000, 1000),
        'rsi': np.random.uniform(0, 100, 1000),
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000)
    })
    
    # Addition some NaN values
    sample_data.loc[np.random.choice(1000, 50, replace=False), 'price'] = np.nan
    
    plot4 = visualizer.plot_data_quality_report(
        sample_data,
        {'missing_values': 50, 'outliers': 25},
        "Crypto Data Quality Assessment"
    )
    print(f"Data quality plot saved: {plot4}")
    
    print(f"\nAll saved : {visualizer.output_dir}")
    print(f"{4} for analysis ML system crypto trading")
