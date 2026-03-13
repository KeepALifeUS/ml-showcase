"""
Correlation-based Graph Construction for Crypto Markets
========================================================

Enterprise-grade graph construction algorithms based on price correlations,
volume relationships, and market dynamics for cryptocurrency trading analysis.

Features:
- Pearson, Spearman, and Kendall correlation graphs
- Dynamic correlation tracking over time windows
- Market regime-aware correlation graphs
- Volatility-adjusted correlations
- Cross-asset correlation networks
- Production-ready scalability with enterprise patterns

Author: ML-Framework ML Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import defaultdict
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationGraphConfig:
    """
    Configuration for building correlation-based graphs
    
    Comprehensive Configuration Management
    """
    # Correlation parameters
    correlation_method: str = 'pearson'  # pearson, spearman, kendall, distance_correlation
    time_window: int = 30 # Temporal window for correlation (days)
    min_correlation: float = 0.3  # Minimal correlation for creation edges
    
    # Dynamic correlation parameters
    use_rolling_correlation: bool = True
    rolling_window: int = 20 # Window for rolling correlation
    correlation_decay: float = 0.95 # Exponential decay for dynamic correlations
    
    # Market regime awareness
    use_market_regimes: bool = True
    volatility_threshold: float = 0.02 # value volatility
    regime_window: int = 10 # Window for determining regime market
    
    # Graph construction parameters
    max_edges_per_node: int = 10 # number edges on
    edge_weight_transform: str = 'absolute'  # absolute, squared, tanh
    use_edge_attributes: bool = True
    
    # Network filtering
    apply_threshold_filtering: bool = True
    apply_topological_filtering: bool = False
    minimum_spanning_tree: bool = False # MST full graph
    
    # Advanced features
    adjust_for_volatility: bool = True
    use_partial_correlations: bool = False # Partial correlation
    include_lag_correlations: bool = False # Lagged correlation
    max_lag: int = 5 # Maximum for correlations

class CorrelationCalculator:
    """
    Calculator various types correlations for crypto assets
    
    Strategy Pattern for correlation methods
    """
    
    def __init__(self, method: str = 'pearson'):
        self.method = method
        self.correlation_functions = {
            'pearson': self._pearson_correlation,
            'spearman': self._spearman_correlation,
            'kendall': self._kendall_correlation,
            'distance_correlation': self._distance_correlation,
            'mutual_information': self._mutual_information_correlation
        }
    
    def compute_correlation_matrix(
        self, 
        data: pd.DataFrame, 
        method: Optional[str] = None
    ) -> np.ndarray:
        """
        Computation correlation matrices
        
        Args:
            data: DataFrame with price data [time, assets]
            method: Method correlation (if None - is used self.method)
            
        Returns:
            np.ndarray: matrix [n_assets, n_assets]
        """
        method = method or self.method
        
        if method not in self.correlation_functions:
            raise ValueError(f"Unknown method correlation: {method}")
        
        return self.correlation_functions[method](data)
    
    def _pearson_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Pearson correlation"""
        # Removing NaN values
        data_clean = data.dropna()
        if data_clean.empty:
            logger.warning("No data after removal NaN")
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='pearson').values
    
    def _spearman_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Spearman correlation"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='spearman').values
    
    def _kendall_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Kendall tau correlation"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        return data_clean.corr(method='kendall').values
    
    def _distance_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Distance correlation ( )"""
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        n_assets = data_clean.shape[1]
        dcorr_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # version distance correlation
                x = data_clean.iloc[:, i].values
                y = data_clean.iloc[:, j].values
                
                # data
                x_centered = x - np.mean(x)
                y_centered = y - np.mean(y)
                
                # Distance correlation approximation
                dcorr = np.corrcoef(np.abs(x_centered), np.abs(y_centered))[0, 1]
                dcorr = dcorr if not np.isnan(dcorr) else 0.0
                
                dcorr_matrix[i, j] = dcorr
                dcorr_matrix[j, i] = dcorr
        
        return dcorr_matrix
    
    def _mutual_information_correlation(self, data: pd.DataFrame) -> np.ndarray:
        """Mutual information based correlation"""
        from sklearn.feature_selection import mutual_info_regression
        
        data_clean = data.dropna()
        if data_clean.empty:
            return np.eye(data.shape[1])
        
        n_assets = data_clean.shape[1]
        mi_matrix = np.eye(n_assets)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                try:
                    x = data_clean.iloc[:, i].values.reshape(-1, 1)
                    y = data_clean.iloc[:, j].values
                    
                    mi = mutual_info_regression(x, y)[0]
                    mi_normalized = 2 * mi / (np.var(data_clean.iloc[:, i]) + np.var(data_clean.iloc[:, j]))
                    mi_normalized = min(mi_normalized, 1.0)  # Limit [0, 1]
                    
                    mi_matrix[i, j] = mi_normalized
                    mi_matrix[j, i] = mi_normalized
                    
                except Exception as e:
                    logger.warning(f"Error in mutual information for {i}, {j}: {e}")
                    mi_matrix[i, j] = 0.0
                    mi_matrix[j, i] = 0.0
        
        return mi_matrix
    
    def compute_rolling_correlations(
        self, 
        data: pd.DataFrame, 
        window: int,
        method: Optional[str] = None
    ) -> List[np.ndarray]:
        """
        Rolling correlation by temporal
        
        Returns:
            List[np.ndarray]: List correlation matrices for each temporal step
        """
        method = method or self.method
        rolling_correlations = []
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            corr_matrix = self.compute_correlation_matrix(window_data, method)
            rolling_correlations.append(corr_matrix)
        
        return rolling_correlations
    
    def compute_lagged_correlations(
        self, 
        data: pd.DataFrame, 
        max_lag: int = 5,
        method: Optional[str] = None
    ) -> Dict[int, np.ndarray]:
        """
        Lagged correlation between assets
        
        Returns:
            Dict[int, np.ndarray]: {lag: correlation_matrix}
        """
        method = method or self.method
        lagged_correlations = {}
        
        for lag in range(1, max_lag + 1):
            lagged_data = data.copy()
            
            # Create lagged data
            for col in data.columns:
                lagged_data[f"{col}_lag{lag}"] = data[col].shift(lag)
            
            # Computing correlation between and data
            original_cols = data.columns
            lagged_cols = [f"{col}_lag{lag}" for col in original_cols]
            
            cross_corr_data = pd.concat([
                lagged_data[original_cols], 
                lagged_data[lagged_cols]
            ], axis=1).dropna()
            
            if not cross_corr_data.empty:
                full_corr = self.compute_correlation_matrix(cross_corr_data, method)
                # Extracting cross-correlations (original vs lagged)
                n_assets = len(original_cols)
                cross_correlations = full_corr[:n_assets, n_assets:]
                lagged_correlations[lag] = cross_correlations
            else:
                lagged_correlations[lag] = np.zeros((len(original_cols), len(original_cols)))
        
        return lagged_correlations

class MarketRegimeDetector:
    """
     market regimes for adaptive correlations
    
    Market Intelligence Module
    """
    
    def __init__(self, volatility_threshold: float = 0.02, window: int = 10):
        self.volatility_threshold = volatility_threshold
        self.window = window
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Determine market regime (low/high volatility)
        
        Args:
            data: DataFrame with price data
            
        Returns:
            pd.Series: Regimes market ('low_vol', 'high_vol')
        """
        # Computing volatility (rolling std )
        returns = data.pct_change().dropna()
        volatility = returns.rolling(window=self.window).std().mean(axis=1)
        
        # Classification regimes
        regimes = pd.Series(index=volatility.index, dtype=str)
        regimes[volatility <= self.volatility_threshold] = 'low_vol'
        regimes[volatility > self.volatility_threshold] = 'high_vol'
        
        return regimes
    
    def get_regime_correlations(
        self, 
        data: pd.DataFrame, 
        calculator: CorrelationCalculator
    ) -> Dict[str, np.ndarray]:
        """
        Correlation in different market regimes
        
        Returns:
            Dict[str, np.ndarray]: {'low_vol': corr_matrix, 'high_vol': corr_matrix}
        """
        regimes = self.detect_regime(data)
        regime_correlations = {}
        
        for regime in ['low_vol', 'high_vol']:
            regime_mask = regimes == regime
            regime_data = data[regime_mask]
            
            if len(regime_data) > 10:  # Enough data
                regime_correlations[regime] = calculator.compute_correlation_matrix(regime_data)
            else:
                logger.warning(f"Insufficient data for regime {regime}")
                regime_correlations[regime] = np.eye(data.shape[1])
        
        return regime_correlations

class CorrelationGraphBuilder:
    """
    Main class for building correlation-based graphs
    
    Builder Pattern for graph construction
    """
    
    def __init__(self, config: CorrelationGraphConfig):
        self.config = config
        self.calculator = CorrelationCalculator(config.correlation_method)
        
        if config.use_market_regimes:
            self.regime_detector = MarketRegimeDetector(
                config.volatility_threshold, 
                config.regime_window
            )
        
        logger.info(f"Initialized CorrelationGraphBuilder with method {config.correlation_method}")
    
    def build_correlation_graph(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        node_features: Optional[np.ndarray] = None,
        asset_names: Optional[List[str]] = None
    ) -> Data:
        """
        Build correlation graph from price data
        
        Args:
            price_data: DataFrame with assets [time, assets]
            volume_data: data by volumes
            node_features: Additional features nodes
            asset_names: assets
            
        Returns:
            Data: PyTorch Geometric Data object
        """
        if asset_names is None:
            asset_names = price_data.columns.tolist()
        
        # Computation correlation matrices
        correlation_matrix = self._compute_adaptive_correlations(price_data, volume_data)
        
        # Build graph from correlation matrices
        edge_index, edge_weights, edge_attr = self._matrix_to_graph(
            correlation_matrix, 
            asset_names
        )
        
        # Node features
        if node_features is None:
            node_features = self._extract_node_features(price_data, volume_data)
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weights,
            edge_attr=edge_attr if self.config.use_edge_attributes else None
        )
        
        # Add metadata
        data.asset_names = asset_names
        data.correlation_matrix = correlation_matrix
        data.num_nodes = len(asset_names)
        
        return data
    
    def _compute_adaptive_correlations(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Computation adaptive correlations with taking into account configuration"""
        
        if self.config.use_rolling_correlation:
            # Rolling correlations
            rolling_corrs = self.calculator.compute_rolling_correlations(
                price_data, 
                self.config.rolling_window,
                self.config.correlation_method
            )
            
            if rolling_corrs:
                # average correlations
                weights = np.array([
                    self.config.correlation_decay ** (len(rolling_corrs) - i - 1)
                    for i in range(len(rolling_corrs))
                ])
                weights = weights / weights.sum()
                
                correlation_matrix = np.average(rolling_corrs, axis=0, weights=weights)
            else:
                correlation_matrix = self.calculator.compute_correlation_matrix(price_data)
        
        elif self.config.use_market_regimes:
            # - correlation
            regime_correlations = self.regime_detector.get_regime_correlations(
                price_data, self.calculator
            )
            
            # Determining current regime by data
            current_regime = self.regime_detector.detect_regime(
                price_data.tail(self.config.regime_window)
            ).mode().iloc[0]
            
            correlation_matrix = regime_correlations.get(current_regime, 
                                                       self.calculator.compute_correlation_matrix(price_data))
        
        else:
            # Standard correlation
            correlation_matrix = self.calculator.compute_correlation_matrix(price_data)
        
        # Adjustment for volatility
        if self.config.adjust_for_volatility and volume_data is not None:
            correlation_matrix = self._adjust_for_volatility(
                correlation_matrix, price_data, volume_data
            )
        
        # Partial correlations
        if self.config.use_partial_correlations:
            correlation_matrix = self._compute_partial_correlations(
                price_data, correlation_matrix
            )
        
        return correlation_matrix
    
    def _adjust_for_volatility(
        self, 
        correlation_matrix: np.ndarray,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame
    ) -> np.ndarray:
        """ correlations on volatility"""
        returns = price_data.pct_change().dropna()
        volatilities = returns.std().values
        
        # Weighted correlation by inverse volatility
        vol_weights = 1.0 / (volatilities + 1e-8)
        vol_weights = vol_weights / vol_weights.sum()
        
        # Apply volatility adjustment
        adjusted_corr = correlation_matrix.copy()
        for i in range(len(vol_weights)):
            for j in range(len(vol_weights)):
                if i != j:
                    vol_factor = np.sqrt(vol_weights[i] * vol_weights[j])
                    adjusted_corr[i, j] *= vol_factor
        
        return adjusted_corr
    
    def _compute_partial_correlations(
        self, 
        price_data: pd.DataFrame,
        correlation_matrix: np.ndarray
    ) -> np.ndarray:
        """Partial correlation ( influence other variables)"""
        try:
            # Precision matrix (inverse of covariance)
            returns = price_data.pct_change().dropna()
            cov_matrix = returns.cov().values
            
            # Regularization for stability
            reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
            precision_matrix = np.linalg.inv(reg_cov)
            
            # Partial correlations from precision matrix
            partial_corr = np.zeros_like(precision_matrix)
            for i in range(len(precision_matrix)):
                for j in range(len(precision_matrix)):
                    if i != j:
                        partial_corr[i, j] = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
                    else:
                        partial_corr[i, j] = 1.0
            
            return partial_corr
            
        except np.linalg.LinAlgError:
            logger.warning("Not succeeded correlation, use ")
            return correlation_matrix
    
    def _matrix_to_graph(
        self, 
        correlation_matrix: np.ndarray,
        asset_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Transform correlation matrices in graph"""
        
        n_assets = len(asset_names)
        edges = []
        weights = []
        edge_attributes = []
        
        # Create edges on basis correlations
        for i in range(n_assets):
            # Getting correlation for node i
            node_correlations = []
            for j in range(n_assets):
                if i != j:
                    corr_value = correlation_matrix[i, j]
                    # Applying weights
                    weight = self._transform_edge_weight(corr_value)
                    
                    if abs(weight) >= self.config.min_correlation:
                        node_correlations.append((j, weight, corr_value))
            
            # Sort by descending weights and take top-K
            node_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top_connections = node_correlations[:self.config.max_edges_per_node]
            
            for j, weight, raw_corr in top_connections:
                edges.append([i, j])
                weights.append(weight)
                
                if self.config.use_edge_attributes:
                    # Additional attributes edges
                    edge_attr = [
                        raw_corr, # correlation
                        abs(raw_corr), # Absolute correlation
                        1.0 if raw_corr > 0 else -1.0, # correlation
                        weight # weight
                    ]
                    edge_attributes.append(edge_attr)
        
        # Minimum Spanning Tree if
        if self.config.minimum_spanning_tree:
            edges, weights, edge_attributes = self._build_mst(
                correlation_matrix, edges, weights, edge_attributes
            )
        
        # Convert in PyTorch tensors
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(weights, dtype=torch.float32)
            
            if self.config.use_edge_attributes and edge_attributes:
                edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
            else:
                edge_attr = None
        else:
            # Empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_weights = torch.empty(0, dtype=torch.float32)
            edge_attr = None
        
        return edge_index, edge_weights, edge_attr
    
    def _transform_edge_weight(self, correlation: float) -> float:
        """Transformation weights edges"""
        if self.config.edge_weight_transform == 'absolute':
            return abs(correlation)
        elif self.config.edge_weight_transform == 'squared':
            return correlation ** 2
        elif self.config.edge_weight_transform == 'tanh':
            return np.tanh(abs(correlation))
        else:
            return correlation
    
    def _build_mst(
        self, 
        correlation_matrix: np.ndarray,
        edges: List[List[int]], 
        weights: List[float],
        edge_attributes: List[List[float]]
    ) -> Tuple[List[List[int]], List[float], List[List[float]]]:
        """Build Minimum Spanning Tree"""
        # Create NetworkX graph
        G = nx.Graph()
        
        for i, (edge, weight) in enumerate(zip(edges, weights)):
            G.add_edge(edge[0], edge[1], weight=1.0 - abs(weight)) # for MST
        
        # Building MST
        mst = nx.minimum_spanning_tree(G)
        
        # Extracting edges MST
        mst_edges = []
        mst_weights = []
        mst_attributes = []
        
        for i, (u, v) in enumerate(mst.edges()):
            # Finding weight
            original_weight = abs(correlation_matrix[u, v])
            mst_edges.append([u, v])
            mst_weights.append(original_weight)
            
            if self.config.use_edge_attributes:
                raw_corr = correlation_matrix[u, v]
                edge_attr = [
                    raw_corr,
                    abs(raw_corr),
                    1.0 if raw_corr > 0 else -1.0,
                    original_weight
                ]
                mst_attributes.append(edge_attr)
        
        return mst_edges, mst_weights, mst_attributes
    
    def _extract_node_features(
        self, 
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Extraction features nodes from data"""
        features = []
        
        for asset in price_data.columns:
            asset_prices = price_data[asset].dropna()
            asset_returns = asset_prices.pct_change().dropna()
            
            asset_features = [
                # Statistical characteristics
                asset_returns.mean(),  # Average return
                asset_returns.std(),   # Volatility
                asset_returns.skew(), # Skewness
                asset_returns.kurtosis(), #
                
                # Risk metrics
                asset_returns.quantile(0.05),  # VaR 5%
                asset_returns.quantile(0.95),  # VaR 95%
                
                # Technical indicators
                asset_prices.iloc[-1] / asset_prices.iloc[0] - 1,  # Total return
                len(asset_returns[asset_returns > 0]) / len(asset_returns), # % positive days
            ]
            
            # Add characteristics if available
            if volume_data is not None and asset in volume_data.columns:
                asset_volumes = volume_data[asset].dropna()
                asset_features.extend([
                    asset_volumes.mean(), # Average volume
                    asset_volumes.std(), # deviation
                    asset_volumes.iloc[-1] / asset_volumes.mean() # Relative current volume
                ])
            else:
                # Fill zeros if no data by volumes
                asset_features.extend([0.0, 0.0, 1.0])
            
            features.append(asset_features)
        
        # Normalization features
        features_array = np.array(features)
        
        # Replacing NaN and inf values
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Standardization
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        return features_normalized.astype(np.float32)
    
    def build_dynamic_correlation_graph(
        self, 
        price_data: pd.DataFrame,
        timestamps: pd.DatetimeIndex,
        node_features: Optional[np.ndarray] = None
    ) -> List[Data]:
        """
        Build dynamic correlation graphs
        
        Returns:
            List[Data]: List graphs for each temporal step
        """
        dynamic_graphs = []
        
        for i, timestamp in enumerate(timestamps):
            # Determining window
            start_idx = max(0, i - self.config.time_window)
            end_idx = i + 1
            
            window_data = price_data.iloc[start_idx:end_idx]
            
            if len(window_data) < 5: # Minimum number
                continue
            
            # Building graph for this windows
            graph = self.build_correlation_graph(
                window_data,
                node_features=node_features[i:i+1] if node_features is not None else None
            )
            
            # Add label
            graph.timestamp = timestamp
            graph.window_start = window_data.index[0]
            graph.window_end = window_data.index[-1]
            
            dynamic_graphs.append(graph)
        
        logger.info(f" {len(dynamic_graphs)} dynamic graphs")
        return dynamic_graphs

def create_correlation_graph(
    price_data: pd.DataFrame,
    volume_data: Optional[pd.DataFrame] = None,
    correlation_method: str = 'pearson',
    min_correlation: float = 0.3,
    **kwargs
) -> Data:
    """
    Factory function for fast creation correlation graph
    
    Simple Factory for graph creation
    """
    config = CorrelationGraphConfig(
        correlation_method=correlation_method,
        min_correlation=min_correlation,
        **kwargs
    )
    
    builder = CorrelationGraphBuilder(config)
    return builder.build_correlation_graph(price_data, volume_data)

# Export main classes
__all__ = [
    'CorrelationGraphConfig',
    'CorrelationCalculator',
    'MarketRegimeDetector',
    'CorrelationGraphBuilder',
    'create_correlation_graph'
]