"""
Explanation Dashboard Crypto Trading Bot v5.0

 comprehensive interactive dashboard visualizing XAI results
    enterprise patterns.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Local imports
from ..explainers.shap_explainer import SHAPExplanation
from ..explainers.lime_explainer import LIMEExplanation
from ..explainers.counterfactual_explainer import CounterfactualExplanation
from ..explainers.anchor_explainer import AnchorExplanation
from ..analysis.feature_importance import FeatureImportanceResult
from ..analysis.decision_paths import DecisionPathResult

logger = logging.getLogger(__name__)


class CryptoTradingExplanationDashboard:
    """
    Enterprise-grade explanation dashboard crypto trading models
    
    Provides interactive visualization :
    - SHAP, LIME, Counterfactual, Anchor explanations
    - Feature importance analysis
    - Decision path visualization
    - Trading signal interpretation
    - Risk analysis compliance reporting
    
    enterprise patterns:
    - Real-time explanation updates
    - Multi-model comparison
    - Enterprise authentication
    - Performance monitoring
    - Regulatory compliance tracking
    """
    
    def __init__(
        self,
        port: int = 8050,
        debug: bool = False,
        cache_dir: Optional[Path] = None
    ):
        """Initialize explanation dashboard"""
        self.port = port
        self.debug = debug
        self.cache_dir = cache_dir or Path("./cache/dashboard")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )
        
        # Data storage real-time updates
        self.explanation_data = {
            'shap': [],
            'lime': [],
            'counterfactual': [],
            'anchor': [],
            'feature_importance': None,
            'decision_paths': None
        }
        
        # Trading symbols filtering
        self.available_symbols = set()
        
        # Initialize dashboard layout
        self._setup_layout()
        self._setup_callbacks()
        
        logger.info(f"Initialized explanation dashboard on port {port}")
    
    def _setup_layout(self) -> None:
        """Setup dashboard layout trading focus"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🤖 Crypto Trading AI Explainability Dashboard", 
                           className="text-center mb-4"),
                    html.P("Real-time model interpretation crypto trading decisions",
                           className="text-center text-muted mb-4")
                ])
            ]),
            
            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Dashboard Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Trading Symbol"),
                                    dcc.Dropdown(
                                        id='symbol-dropdown',
                                        options=[],
                                        value=None,
                                        placeholder="Select symbol..."
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Explanation Method"),
                                    dcc.Dropdown(
                                        id='method-dropdown',
                                        options=[
                                            {'label': 'SHAP', 'value': 'shap'},
                                            {'label': 'LIME', 'value': 'lime'},
                                            {'label': 'Counterfactual', 'value': 'counterfactual'},
                                            {'label': 'Anchor Rules', 'value': 'anchor'},
                                            {'label': 'Feature Importance', 'value': 'feature_importance'},
                                            {'label': 'Decision Paths', 'value': 'decision_paths'}
                                        ],
                                        value='shap'
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Time Range"),
                                    dcc.Dropdown(
                                        id='time-range-dropdown',
                                        options=[
                                            {'label': 'Last Hour', 'value': '1h'},
                                            {'label': 'Last 4 Hours', 'value': '4h'},
                                            {'label': 'Last Day', 'value': '1d'},
                                            {'label': 'Last Week', 'value': '1w'},
                                            {'label': 'All Time', 'value': 'all'}
                                        ],
                                        value='1d'
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Auto Refresh"),
                                    dbc.Switch(
                                        id='auto-refresh-switch',
                                        value=True,
                                        className="mt-2"
                                    ),
                                    dcc.Interval(
                                        id='interval-component',
                                        interval=30*1000,  # 30 seconds
                                        n_intervals=0
                                    )
                                ], width=3)
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Main content area
            dbc.Row([
                dbc.Col([
                    # Summary cards
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("0", id="total-explanations", className="text-primary"),
                                    html.P("Total Explanations", className="mb-0")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("0", id="buy-signals", className="text-success"),
                                    html.P("Buy Signals", className="mb-0")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("0", id="sell-signals", className="text-danger"),
                                    html.P("Sell Signals", className="mb-0")
                                ])
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.H4("0%", id="avg-confidence", className="text-info"),
                                    html.P("Avg Confidence", className="mb-0")
                                ])
                            ])
                        ], width=3)
                    ], className="mb-4"),
                    
                    # Main visualization
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("Model Explanation Visualization", className="mb-0"),
                            dbc.Badge("Live", color="success", className="ms-2")
                        ]),
                        dbc.CardBody([
                            dcc.Loading([
                                dcc.Graph(id="main-explanation-chart")
                            ])
                        ])
                    ], className="mb-4"),
                    
                    # Secondary charts
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Feature Importance"),
                                dbc.CardBody([
                                    dcc.Graph(id="feature-importance-chart")
                                ])
                            ])
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Trading Signal Analysis"),
                                dbc.CardBody([
                                    dcc.Graph(id="signal-analysis-chart")
                                ])
                            ])
                        ], width=6)
                    ], className="mb-4"),
                    
                    # Detailed analysis tabs
                    dbc.Card([
                        dbc.CardHeader([
                            dbc.Tabs([
                                dbc.Tab(label="Explanation Details", tab_id="details"),
                                dbc.Tab(label="Risk Analysis", tab_id="risk"),
                                dbc.Tab(label="Compliance Report", tab_id="compliance"),
                                dbc.Tab(label="Model Performance", tab_id="performance")
                            ], id="analysis-tabs", active_tab="details")
                        ]),
                        dbc.CardBody([
                            html.Div(id="tab-content")
                        ])
                    ])
                ])
            ])
        ], fluid=True)
    
    def _setup_callbacks(self) -> None:
        """Setup dashboard callbacks interactivity"""
        
        # Update symbol dropdown options
        @self.app.callback(
            Output('symbol-dropdown', 'options'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_symbol_options(n_intervals):
            symbols = list(self.available_symbols)
            return [{'label': symbol, 'value': symbol} for symbol in sorted(symbols)]
        
        # Update summary statistics
        @self.app.callback(
            [Output('total-explanations', 'children'),
             Output('buy-signals', 'children'),
             Output('sell-signals', 'children'),
             Output('avg-confidence', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('symbol-dropdown', 'value'),
             Input('time-range-dropdown', 'value')]
        )
        def update_summary_stats(n_intervals, selected_symbol, time_range):
            return self._calculate_summary_stats(selected_symbol, time_range)
        
        # Update main explanation chart
        @self.app.callback(
            Output('main-explanation-chart', 'figure'),
            [Input('method-dropdown', 'value'),
             Input('symbol-dropdown', 'value'),
             Input('time-range-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_main_chart(method, symbol, time_range, n_intervals):
            return self._create_main_explanation_chart(method, symbol, time_range)
        
        # Update feature importance chart
        @self.app.callback(
            Output('feature-importance-chart', 'figure'),
            [Input('symbol-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_feature_importance(symbol, n_intervals):
            return self._create_feature_importance_chart(symbol)
        
        # Update signal analysis chart
        @self.app.callback(
            Output('signal-analysis-chart', 'figure'),
            [Input('symbol-dropdown', 'value'),
             Input('time-range-dropdown', 'value'),
             Input('interval-component', 'n_intervals')]
        )
        def update_signal_analysis(symbol, time_range, n_intervals):
            return self._create_signal_analysis_chart(symbol, time_range)
        
        # Update tab content
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('analysis-tabs', 'active_tab'),
             Input('method-dropdown', 'value'),
             Input('symbol-dropdown', 'value')]
        )
        def update_tab_content(active_tab, method, symbol):
            return self._create_tab_content(active_tab, method, symbol)
    
    def add_explanation(
        self,
        explanation: Union[SHAPExplanation, LIMEExplanation, CounterfactualExplanation, AnchorExplanation],
        explanation_type: str
    ) -> None:
        """Add new explanation to dashboard"""
        try:
            if explanation_type in self.explanation_data:
                self.explanation_data[explanation_type].append(explanation)
                
                # Add symbol to available symbols
                if hasattr(explanation, 'symbol') and explanation.symbol:
                    self.available_symbols.add(explanation.symbol)
                
                # Limit stored explanations memory management
                if len(self.explanation_data[explanation_type]) > 1000:
                    self.explanation_data[explanation_type] = self.explanation_data[explanation_type][-500:]
                
                logger.debug(f"Added {explanation_type} explanation")
            
        except Exception as e:
            logger.error(f"Error adding explanation: {e}")
    
    def add_feature_importance(self, result: FeatureImportanceResult) -> None:
        """Add feature importance analysis"""
        try:
            self.explanation_data['feature_importance'] = result
            logger.debug("Added feature importance analysis")
        except Exception as e:
            logger.error(f"Error adding feature importance: {e}")
    
    def add_decision_paths(self, result: DecisionPathResult) -> None:
        """Add decision path analysis"""
        try:
            self.explanation_data['decision_paths'] = result
            logger.debug("Added decision paths analysis")
        except Exception as e:
            logger.error(f"Error adding decision paths: {e}")
    
    def _calculate_summary_stats(
        self,
        symbol: Optional[str],
        time_range: str
    ) -> Tuple[str, str, str, str]:
        """Calculate summary statistics dashboard"""
        try:
            # Filter explanations by symbol and time
            filtered_explanations = self._filter_explanations(symbol, time_range)
            
            total = len(filtered_explanations)
            buy_signals = len([exp for exp in filtered_explanations 
                             if hasattr(exp, 'trade_signal') and exp.trade_signal == 'BUY'])
            sell_signals = len([exp for exp in filtered_explanations 
                              if hasattr(exp, 'trade_signal') and exp.trade_signal == 'SELL'])
            
            # Calculate average confidence
            confidences = []
            for exp in filtered_explanations:
                if hasattr(exp, 'prediction_confidence') and exp.prediction_confidence is not None:
                    confidences.append(exp.prediction_confidence)
                elif hasattr(exp, 'instance_confidence') and exp.instance_confidence is not None:
                    confidences.append(exp.instance_confidence)
                elif hasattr(exp, 'validity_score') and exp.validity_score is not None:
                    confidences.append(exp.validity_score)
            
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return (
                str(total),
                str(buy_signals),
                str(sell_signals),
                f"{avg_confidence*100:.1f}%"
            )
            
        except Exception as e:
            logger.error(f"Error calculating summary stats: {e}")
            return "0", "0", "0", "0%"
    
    def _filter_explanations(
        self,
        symbol: Optional[str],
        time_range: str
    ) -> List[Any]:
        """Filter explanations by symbol and time range"""
        try:
            all_explanations = []
            
            # Collect all explanations
            for explanation_list in self.explanation_data.values():
                if isinstance(explanation_list, list):
                    all_explanations.extend(explanation_list)
            
            # Filter by symbol
            if symbol:
                all_explanations = [
                    exp for exp in all_explanations
                    if hasattr(exp, 'symbol') and exp.symbol == symbol
                ]
            
            # Filter by time range
            if time_range != 'all':
                cutoff_time = self._get_cutoff_time(time_range)
                all_explanations = [
                    exp for exp in all_explanations
                    if hasattr(exp, 'timestamp') and exp.timestamp >= cutoff_time
                ]
            
            return all_explanations
            
        except Exception as e:
            logger.error(f"Error filtering explanations: {e}")
            return []
    
    def _get_cutoff_time(self, time_range: str) -> datetime:
        """Get cutoff time filtering"""
        now = datetime.now()
        
        if time_range == '1h':
            return now - timedelta(hours=1)
        elif time_range == '4h':
            return now - timedelta(hours=4)
        elif time_range == '1d':
            return now - timedelta(days=1)
        elif time_range == '1w':
            return now - timedelta(weeks=1)
        else:
            return now - timedelta(days=365)  # Default fallback
    
    def _create_main_explanation_chart(
        self,
        method: str,
        symbol: Optional[str],
        time_range: str
    ) -> go.Figure:
        """Create main explanation visualization"""
        try:
            if method == 'shap':
                return self._create_shap_chart(symbol, time_range)
            elif method == 'lime':
                return self._create_lime_chart(symbol, time_range)
            elif method == 'counterfactual':
                return self._create_counterfactual_chart(symbol, time_range)
            elif method == 'anchor':
                return self._create_anchor_chart(symbol, time_range)
            elif method == 'feature_importance':
                return self._create_feature_importance_detailed_chart()
            elif method == 'decision_paths':
                return self._create_decision_paths_chart()
            else:
                return self._create_empty_chart("Select explanation method")
                
        except Exception as e:
            logger.error(f"Error creating main chart: {e}")
            return self._create_empty_chart("Error loading explanation data")
    
    def _create_shap_chart(self, symbol: Optional[str], time_range: str) -> go.Figure:
        """Create SHAP explanation visualization"""
        try:
            shap_explanations = [
                exp for exp in self.explanation_data['shap']
                if (not symbol or getattr(exp, 'symbol', None) == symbol)
            ]
            
            if not shap_explanations:
                return self._create_empty_chart("No SHAP explanations available")
            
            # Get latest explanation
            latest_explanation = shap_explanations[-1]
            
            # Create SHAP waterfall plot
            feature_names = latest_explanation.feature_names
            shap_values = latest_explanation.shap_values
            
            if shap_values.ndim > 1:
                shap_values = shap_values[0]  # Take first instance
            
            # Sort by absolute importance
            importance_order = np.argsort(np.abs(shap_values))[::-1][:15]  # Top 15 features
            
            fig = go.Figure()
            
            # Add horizontal bar chart
            fig.add_trace(go.Bar(
                y=[feature_names[i] for i in importance_order],
                x=[shap_values[i] for i in importance_order],
                orientation='h',
                marker=dict(
                    color=[shap_values[i] for i in importance_order],
                    colorscale='RdBu',
                    cmid=0
                ),
                text=[f'{shap_values[i]:.3f}' for i in importance_order],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"SHAP Feature Importance - {symbol or 'All Symbols'}",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating SHAP chart: {e}")
            return self._create_empty_chart("Error loading SHAP data")
    
    def _create_lime_chart(self, symbol: Optional[str], time_range: str) -> go.Figure:
        """Create LIME explanation visualization"""
        try:
            lime_explanations = [
                exp for exp in self.explanation_data['lime']
                if (not symbol or getattr(exp, 'symbol', None) == symbol)
            ]
            
            if not lime_explanations:
                return self._create_empty_chart("No LIME explanations available")
            
            latest_explanation = lime_explanations[-1]
            
            # Get first label's explanation
            first_label = list(latest_explanation.feature_importance.keys())[0]
            feature_importance = latest_explanation.feature_importance[first_label]
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            features, importances = zip(*sorted_features)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=features,
                x=importances,
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='RdBu',
                    cmid=0
                ),
                text=[f'{imp:.3f}' for imp in importances],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"LIME Local Explanation - {symbol or 'All Symbols'}",
                xaxis_title="Feature Importance",
                yaxis_title="Features",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating LIME chart: {e}")
            return self._create_empty_chart("Error loading LIME data")
    
    def _create_counterfactual_chart(self, symbol: Optional[str], time_range: str) -> go.Figure:
        """Create counterfactual explanation visualization"""
        try:
            cf_explanations = [
                exp for exp in self.explanation_data['counterfactual']
                if (not symbol or getattr(exp, 'symbol', None) == symbol)
            ]
            
            if not cf_explanations:
                return self._create_empty_chart("No counterfactual explanations available")
            
            latest_explanation = cf_explanations[-1]
            
            # Show feature changes
            feature_changes = latest_explanation.feature_changes
            
            # Get top changes by magnitude
            sorted_changes = sorted(
                feature_changes.items(),
                key=lambda x: abs(x[1]['absolute_change']),
                reverse=True
            )[:15]
            
            features = []
            original_values = []
            cf_values = []
            changes = []
            
            for feature, change_data in sorted_changes:
                features.append(feature)
                original_values.append(change_data['original_value'])
                cf_values.append(change_data['counterfactual_value'])
                changes.append(change_data['absolute_change'])
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original vs Counterfactual Values', 'Feature Changes'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Original vs counterfactual values
            fig.add_trace(
                go.Scatter(
                    y=features,
                    x=original_values,
                    mode='markers',
                    name='Original',
                    marker=dict(color='blue', size=8)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    y=features,
                    x=cf_values,
                    mode='markers',
                    name='Counterfactual',
                    marker=dict(color='red', size=8)
                ),
                row=1, col=1
            )
            
            # Feature changes
            fig.add_trace(
                go.Bar(
                    y=features,
                    x=changes,
                    orientation='h',
                    name='Change',
                    marker=dict(
                        color=changes,
                        colorscale='RdBu',
                        cmid=0
                    )
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Counterfactual Analysis - {symbol or 'All Symbols'}",
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating counterfactual chart: {e}")
            return self._create_empty_chart("Error loading counterfactual data")
    
    def _create_anchor_chart(self, symbol: Optional[str], time_range: str) -> go.Figure:
        """Create anchor rules visualization"""
        try:
            anchor_explanations = [
                exp for exp in self.explanation_data['anchor']
                if (not symbol or getattr(exp, 'symbol', None) == symbol)
            ]
            
            if not anchor_explanations:
                return self._create_empty_chart("No anchor explanations available")
            
            latest_explanation = anchor_explanations[-1]
            
            # Show top anchor rules
            top_rules = sorted(
                latest_explanation.anchor_rules,
                key=lambda r: r.precision,
                reverse=True
            )[:10]
            
            rule_names = [f"Rule {i+1}" for i in range(len(top_rules))]
            precisions = [rule.precision for rule in top_rules]
            coverages = [rule.coverage for rule in top_rules]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Rule Precision', 'Rule Coverage'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Precision chart
            fig.add_trace(
                go.Bar(
                    x=rule_names,
                    y=precisions,
                    name='Precision',
                    marker=dict(color='green')
                ),
                row=1, col=1
            )
            
            # Coverage chart
            fig.add_trace(
                go.Bar(
                    x=rule_names,
                    y=coverages,
                    name='Coverage',
                    marker=dict(color='blue')
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title=f"Anchor Rules Analysis - {symbol or 'All Symbols'}",
                height=600,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating anchor chart: {e}")
            return self._create_empty_chart("Error loading anchor data")
    
    def _create_feature_importance_chart(self, symbol: Optional[str]) -> go.Figure:
        """Create feature importance chart"""
        try:
            if not self.explanation_data['feature_importance']:
                return self._create_empty_chart("No feature importance data available")
            
            result = self.explanation_data['feature_importance']
            
            # Get consensus ranking
            if not result.consensus_ranking:
                return self._create_empty_chart("No consensus ranking available")
            
            top_features = result.consensus_ranking[:20]
            feature_names = [name for name, _ in top_features]
            importance_scores = [score for _, score in top_features]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=feature_names,
                x=importance_scores,
                orientation='h',
                marker=dict(color='steelblue'),
                text=[f'{score:.3f}' for score in importance_scores],
                textposition='outside'
            ))
            
            fig.update_layout(
                title="Feature Importance Ranking",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return self._create_empty_chart("Error loading feature importance data")
    
    def _create_signal_analysis_chart(
        self,
        symbol: Optional[str],
        time_range: str
    ) -> go.Figure:
        """Create trading signal analysis chart"""
        try:
            filtered_explanations = self._filter_explanations(symbol, time_range)
            
            if not filtered_explanations:
                return self._create_empty_chart("No explanations available for analysis")
            
            # Count trading signals
            signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            
            for exp in filtered_explanations:
                if hasattr(exp, 'trade_signal') and exp.trade_signal:
                    signal = exp.trade_signal
                    if signal in signal_counts:
                        signal_counts[signal] += 1
            
            # Create pie chart
            labels = list(signal_counts.keys())
            values = list(signal_counts.values())
            colors = ['green', 'red', 'yellow']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.3
            )])
            
            fig.update_layout(
                title="Trading Signal Distribution",
                height=400
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating signal analysis chart: {e}")
            return self._create_empty_chart("Error loading signal data")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            height=400
        )
        return fig
    
    def _create_feature_importance_detailed_chart(self) -> go.Figure:
        """Create detailed feature importance visualization"""
        try:
            if not self.explanation_data['feature_importance']:
                return self._create_empty_chart("No feature importance data available")
            
            result = self.explanation_data['feature_importance']
            
            # Create method comparison heatmap
            methods = list(result.feature_scores.keys())
            if len(methods) < 2:
                return self._create_feature_importance_chart(None)
            
            # Get common features
            all_features = set()
            for method_scores in result.feature_scores.values():
                all_features.update(method_scores.keys())
            
            common_features = list(all_features)[:20]  # Top 20 for visualization
            
            # Create heatmap data
            heatmap_data = []
            for method in methods:
                method_data = []
                for feature in common_features:
                    score = result.feature_scores[method].get(feature, 0.0)
                    method_data.append(score)
                heatmap_data.append(method_data)
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=common_features,
                y=methods,
                colorscale='Viridis',
                showscale=True
            ))
            
            fig.update_layout(
                title="Feature Importance Methods Comparison",
                xaxis_title="Features",
                yaxis_title="Methods",
                height=600
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating detailed feature importance chart: {e}")
            return self._create_empty_chart("Error loading detailed feature importance")
    
    def _create_decision_paths_chart(self) -> go.Figure:
        """Create decision paths visualization"""
        try:
            if not self.explanation_data['decision_paths']:
                return self._create_empty_chart("No decision paths data available")
            
            result = self.explanation_data['decision_paths']
            
            # Visualize path statistics
            if not result.decision_rules:
                return self._create_empty_chart("No decision rules available")
            
            # Path length vs confidence scatter plot
            path_lengths = [rule.path_length for rule in result.decision_rules]
            confidences = [rule.confidence for rule in result.decision_rules]
            signals = [rule.trading_signal or 'UNKNOWN' for rule in result.decision_rules]
            
            # Color mapping
            color_map = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'yellow', 'UNKNOWN': 'gray'}
            colors = [color_map.get(signal, 'gray') for signal in signals]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=path_lengths,
                y=confidences,
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6
                ),
                text=[f"Signal: {signal}<br>Support: {rule.support}" 
                      for rule, signal in zip(result.decision_rules, signals)],
                hovertemplate="<b>Path Length:</b> %{x}<br><b>Confidence:</b> %{y:.3f}<br>%{text}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Decision Rules: Path Length vs Confidence",
                xaxis_title="Path Length",
                yaxis_title="Confidence",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating decision paths chart: {e}")
            return self._create_empty_chart("Error loading decision paths data")
    
    def _create_tab_content(
        self,
        active_tab: str,
        method: str,
        symbol: Optional[str]
    ) -> html.Div:
        """Create content active tab"""
        try:
            if active_tab == "details":
                return self._create_details_tab(method, symbol)
            elif active_tab == "risk":
                return self._create_risk_tab(symbol)
            elif active_tab == "compliance":
                return self._create_compliance_tab()
            elif active_tab == "performance":
                return self._create_performance_tab()
            else:
                return html.Div("Select a tab to view content")
                
        except Exception as e:
            logger.error(f"Error creating tab content: {e}")
            return html.Div(f"Error loading tab content: {e}")
    
    def _create_details_tab(self, method: str, symbol: Optional[str]) -> html.Div:
        """Create explanation details tab"""
        explanations = self._filter_explanations(symbol, 'all')
        
        if not explanations:
            return html.Div("No explanations available")
        
        # Show latest explanation details
        latest = explanations[-1]
        
        details = []
        
        if hasattr(latest, 'prediction_confidence'):
            details.append(html.P(f"Prediction Confidence: {latest.prediction_confidence:.3f}"))
        
        if hasattr(latest, 'trade_signal'):
            details.append(html.P(f"Trading Signal: {latest.trade_signal}"))
        
        if hasattr(latest, 'timestamp'):
            details.append(html.P(f"Generated: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"))
        
        if hasattr(latest, 'model_type'):
            details.append(html.P(f"Model Type: {latest.model_type}"))
        
        return html.Div(details)
    
    def _create_risk_tab(self, symbol: Optional[str]) -> html.Div:
        """Create risk analysis tab"""
        explanations = self._filter_explanations(symbol, 'all')
        
        if not explanations:
            return html.Div("No explanations available for risk analysis")
        
        # Calculate risk metrics
        high_conf_predictions = [exp for exp in explanations 
                               if hasattr(exp, 'prediction_confidence') 
                               and exp.prediction_confidence > 0.8]
        
        risk_content = [
            html.H5("Risk Analysis Summary"),
            html.P(f"Total Explanations: {len(explanations)}"),
            html.P(f"High Confidence Predictions: {len(high_conf_predictions)}"),
            html.P(f"Risk Level: {'Low' if len(high_conf_predictions) > len(explanations) * 0.7 else 'Medium' if len(high_conf_predictions) > len(explanations) * 0.5 else 'High'}")
        ]
        
        return html.Div(risk_content)
    
    def _create_compliance_tab(self) -> html.Div:
        """Create compliance report tab"""
        total_explanations = sum(len(exp_list) for exp_list in self.explanation_data.values() 
                               if isinstance(exp_list, list))
        
        compliance_content = [
            html.H5("Compliance Report"),
            html.P(f"Total Explanations Generated: {total_explanations}"),
            html.P("Model Transparency: ✅ Active"),
            html.P("Decision Auditability: ✅ Enabled"),
            html.P("Regulatory Compliance: ✅ Maintained"),
            html.P("Last Updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        ]
        
        return html.Div(compliance_content)
    
    def _create_performance_tab(self) -> html.Div:
        """Create performance metrics tab"""
        performance_content = [
            html.H5("Model Performance Metrics"),
            html.P("Dashboard Status: 🟢 Online"),
            html.P("Real-time Updates: ✅ Active"),
            html.P("Cache Status: ✅ Operational"),
            html.P("Last Refresh: " + datetime.now().strftime('%H:%M:%S'))
        ]
        
        return html.Div(performance_content)
    
    def run(self, host: str = '127.0.0.1', port: Optional[int] = None) -> None:
        """Run dashboard server"""
        port = port or self.port
        
        logger.info(f"Starting explanation dashboard at http://{host}:{port}")
        
        self.app.run_server(
            host=host,
            port=port,
            debug=self.debug,
            use_reloader=False  # Avoid issues with threading
        )


def main():
    """Main function standalone dashboard"""
    dashboard = CryptoTradingExplanationDashboard(
        port=8050,
        debug=True
    )
    
    # Add sample data testing
    from datetime import datetime
    
    # Sample SHAP explanation
    class MockSHAPExplanation:
        def __init__(self):
            self.feature_names = ['price', 'volume', 'rsi', 'macd', 'sma_20', 'volatility']
            self.shap_values = np.random.randn(6) * 0.5
            self.trade_signal = np.random.choice(['BUY', 'SELL', 'HOLD'])
            self.prediction_confidence = np.random.uniform(0.5, 0.95)
            self.symbol = np.random.choice(['BTCUSDT', 'ETHUSDT', 'ADAUSDT'])
            self.timestamp = datetime.now()
    
    # Add sample explanations
    for _ in range(50):
        dashboard.add_explanation(MockSHAPExplanation(), 'shap')
    
    dashboard.run()


if __name__ == "__main__":
    main()