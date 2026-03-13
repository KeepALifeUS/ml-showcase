"""
Crypto-Specific Meta-Learning Tasks
Domain-Specialized Task Generation

Specialized tasks for cryptocurrency trading with considering specifics
cryptocurrency markets, volatility and specific patterns.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import random
from enum import Enum
import math

from .task_distribution import BaseTaskDistribution, TaskConfig, TaskMetadata


class MarketRegime(Enum):
    """Modes cryptocurrency market"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_vol"
    LOW_VOLATILITY = "low_vol"
    CRASH = "crash"
    RECOVERY = "recovery"


class TradingSignalType(Enum):
    """Types trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class CryptoTaskConfig(TaskConfig):
    """Extended configuration for cryptocurrency tasks"""
    
    # Crypto-specific parameters
    trading_pairs: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT"
    ])
    exchanges: List[str] = field(default_factory=lambda: [
        "binance", "coinbase", "kraken", "huobi"
    ])
    timeframes: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d"
    ])
    
    # Market conditions
    market_regimes: List[MarketRegime] = field(default_factory=lambda: [
        MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS
    ])
    
    # Technical indicators
    use_technical_indicators: bool = True
    indicators: List[str] = field(default_factory=lambda: [
        "sma", "ema", "rsi", "macd", "bollinger", "stochastic", "atr"
    ])
    
    # Price prediction
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 5, 15, 60, 240])  # minutes
    price_change_thresholds: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05, 0.1])
    
    # Portfolio management
    include_portfolio_tasks: bool = True
    max_assets_in_portfolio: int = 10
    rebalancing_frequencies: List[str] = field(default_factory=lambda: ["daily", "weekly", "monthly"])
    
    # Risk management
    include_risk_tasks: bool = True
    risk_metrics: List[str] = field(default_factory=lambda: [
        "sharpe", "sortino", "max_drawdown", "var", "cvar"
    ])
    
    # Arbitrage
    include_arbitrage_tasks: bool = True
    min_arbitrage_opportunity: float = 0.001  # 0.1%
    
    # DeFi specific
    include_defi_tasks: bool = False
    defi_protocols: List[str] = field(default_factory=lambda: [
        "uniswap", "compound", "aave", "curve"
    ])


class CryptoMarketSimulator:
    """
    Simulator cryptocurrency market for generation realistic data
    
    Market Simulation Engine
    - Realistic price movements
    - Multi-factor modeling
    - Various market regimes
    """
    
    def __init__(self, config: CryptoTaskConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Parameters for various modes market
        self.regime_parameters = {
            MarketRegime.BULL: {"drift": 0.0002, "volatility": 0.02, "jump_prob": 0.01},
            MarketRegime.BEAR: {"drift": -0.0001, "volatility": 0.025, "jump_prob": 0.015},
            MarketRegime.SIDEWAYS: {"drift": 0.0, "volatility": 0.015, "jump_prob": 0.005},
            MarketRegime.HIGH_VOLATILITY: {"drift": 0.0, "volatility": 0.04, "jump_prob": 0.02},
            MarketRegime.LOW_VOLATILITY: {"drift": 0.0, "volatility": 0.008, "jump_prob": 0.002},
            MarketRegime.CRASH: {"drift": -0.01, "volatility": 0.08, "jump_prob": 0.05},
            MarketRegime.RECOVERY: {"drift": 0.005, "volatility": 0.035, "jump_prob": 0.03}
        }
    
    def generate_price_series(
        self,
        initial_price: float,
        length: int,
        regime: MarketRegime,
        dt: float = 1/1440  # 1 minute in days
    ) -> np.ndarray:
        """
        Generates temporal series prices for specified mode market
        
        Args:
            initial_price: Initial price
            length: Length series
            regime: Mode market
            dt: Temporal step
            
        Returns:
            Array prices
        """
        params = self.regime_parameters[regime]
        
        # Geometric Brownian movement with jumps
        prices = [initial_price]
        
        for i in range(1, length):
            # Main movement (GBM)
            drift = params["drift"] * dt
            diffusion = params["volatility"] * np.sqrt(dt) * np.random.normal()
            
            # Jumps (jump diffusion)
            if np.random.random() < params["jump_prob"] * dt:
                jump_size = np.random.normal(0, 0.1)  # Jump magnitude
                diffusion += jump_size
            
            # Update price
            log_return = drift + diffusion
            new_price = prices[-1] * np.exp(log_return)
            
            # Limitations on minimum price
            new_price = max(new_price, initial_price * 0.001)
            
            prices.append(new_price)
        
        return np.array(prices)
    
    def generate_volume_series(self, price_series: np.ndarray, base_volume: float = 1000000) -> np.ndarray:
        """Generates volumes trading correlated with price movements"""
        returns = np.diff(np.log(price_series))
        volatility = np.abs(returns)
        
        # Volume increases with volatility
        volume_multiplier = 1 + 2 * volatility  # More volatility -> more volume
        
        # Add random component
        random_factor = np.random.lognormal(0, 0.3, len(volatility))
        
        volumes = base_volume * volume_multiplier * random_factor
        
        # Add first element
        volumes = np.concatenate([[base_volume], volumes])
        
        return volumes
    
    def compute_technical_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes technical indicators"""
        indicators = {}
        
        # Simple Moving Average
        if "sma" in self.config.indicators:
            indicators["sma_20"] = self._compute_sma(prices, 20)
            indicators["sma_50"] = self._compute_sma(prices, 50)
        
        # Exponential Moving Average
        if "ema" in self.config.indicators:
            indicators["ema_12"] = self._compute_ema(prices, 12)
            indicators["ema_26"] = self._compute_ema(prices, 26)
        
        # RSI
        if "rsi" in self.config.indicators:
            indicators["rsi"] = self._compute_rsi(prices, 14)
        
        # MACD
        if "macd" in self.config.indicators:
            macd_line, signal_line, histogram = self._compute_macd(prices)
            indicators["macd"] = macd_line
            indicators["macd_signal"] = signal_line
            indicators["macd_histogram"] = histogram
        
        # Bollinger Bands
        if "bollinger" in self.config.indicators:
            bb_upper, bb_middle, bb_lower = self._compute_bollinger_bands(prices, 20, 2)
            indicators["bb_upper"] = bb_upper
            indicators["bb_middle"] = bb_middle
            indicators["bb_lower"] = bb_lower
        
        # Stochastic
        if "stochastic" in self.config.indicators:
            stoch_k, stoch_d = self._compute_stochastic(prices, 14, 3)
            indicators["stoch_k"] = stoch_k
            indicators["stoch_d"] = stoch_d
        
        # Average True Range
        if "atr" in self.config.indicators:
            # For ATR needed high, low, close. Use simplified version
            indicators["atr"] = self._compute_atr_simplified(prices, 14)
        
        return indicators
    
    def _compute_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Simple Moving Average"""
        sma = np.full_like(prices, np.nan)
        for i in range(window-1, len(prices)):
            sma[i] = np.mean(prices[i-window+1:i+1])
        return sma
    
    def _compute_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Exponential Moving Average"""
        ema = np.full_like(prices, np.nan)
        alpha = 2.0 / (window + 1)
        
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _compute_rsi(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.full(len(prices), np.nan)
        avg_losses = np.full(len(prices), np.nan)
        
        # First value
        if len(gains) >= window:
            avg_gains[window] = np.mean(gains[:window])
            avg_losses[window] = np.mean(losses[:window])
            
            # Exponential smoothing
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (window-1) + gains[i-1]) / window
                avg_losses[i] = (avg_losses[i-1] * (window-1) + losses[i-1]) / window
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD Indicator"""
        ema_12 = self._compute_ema(prices, 12)
        ema_26 = self._compute_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        signal_line = self._compute_ema(macd_line, 9)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _compute_bollinger_bands(
        self, 
        prices: np.ndarray, 
        window: int, 
        std_dev: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands"""
        sma = self._compute_sma(prices, window)
        
        rolling_std = np.full_like(prices, np.nan)
        for i in range(window-1, len(prices)):
            rolling_std[i] = np.std(prices[i-window+1:i+1])
        
        upper_band = sma + (rolling_std * std_dev)
        lower_band = sma - (rolling_std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _compute_stochastic(
        self, 
        prices: np.ndarray, 
        k_window: int, 
        d_window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator"""
        # Simplified version: use price as high/low/close
        stoch_k = np.full_like(prices, np.nan)
        
        for i in range(k_window-1, len(prices)):
            window_prices = prices[i-k_window+1:i+1]
            highest_high = np.max(window_prices)
            lowest_low = np.min(window_prices)
            
            if highest_high != lowest_low:
                stoch_k[i] = 100 * (prices[i] - lowest_low) / (highest_high - lowest_low)
            else:
                stoch_k[i] = 50
        
        stoch_d = self._compute_sma(stoch_k, d_window)
        
        return stoch_k, stoch_d
    
    def _compute_atr_simplified(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Simplified ATR (Average True Range)"""
        # Use absolute changes prices as approximation
        price_changes = np.abs(np.diff(prices))
        
        atr = np.full_like(prices, np.nan)
        for i in range(window, len(prices)):
            atr[i] = np.mean(price_changes[i-window:i])
        
        return atr


class CryptoPriceDirectionTask:
    """
    Task predictions directions price cryptocurrency
    
    Classification Task Generator
    - Multi-class price direction prediction
    - Various time horizons
    - Technical indicator integration
    """
    
    def __init__(self, config: CryptoTaskConfig, simulator: CryptoMarketSimulator):
        self.config = config
        self.simulator = simulator
    
    def generate_task(
        self,
        trading_pair: str,
        regime: MarketRegime,
        prediction_horizon: int
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Generates task predictions directions price"""
        
        # Generate historical data
        initial_price = random.uniform(0.1, 50000)  # Random initial price
        series_length = 2000
        
        prices = self.simulator.generate_price_series(
            initial_price, series_length, regime
        )
        volumes = self.simulator.generate_volume_series(prices)
        
        # Compute technical indicators
        indicators = self.simulator.compute_technical_indicators(prices, volumes)
        
        # Create features and labels
        features, labels = self._create_direction_features_labels(
            prices, volumes, indicators, prediction_horizon
        )
        
        # Split on support and query
        support_data, support_labels, query_data, query_labels = self._split_data(
            features, labels
        )
        
        task_data = {
            'support_data': torch.FloatTensor(support_data),
            'support_labels': torch.LongTensor(support_labels),
            'query_data': torch.FloatTensor(query_data),
            'query_labels': torch.LongTensor(query_labels)
        }
        
        # Metadata
        metadata = TaskMetadata(
            task_id=f"price_direction_{trading_pair}_{regime.value}_{prediction_horizon}",
            task_type="classification",
            difficulty=self._compute_direction_difficulty(labels),
            source_domain="crypto_price_direction",
            target_variable="price_direction",
            feature_names=self._get_feature_names(indicators),
            data_quality_score=0.9,
            created_timestamp=time.time(),
            trading_pair=trading_pair,
            timeframe=f"{prediction_horizon}m"
        )
        
        return task_data, metadata
    
    def _create_direction_features_labels(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        indicators: Dict[str, np.ndarray],
        horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates features and labels for tasks directions price"""
        
        # Window for features
        feature_window = 60  # 60 periods history
        
        features_list = []
        labels_list = []
        
        # Begin with sufficient margin for indicators
        start_idx = max(100, feature_window)
        end_idx = len(prices) - horizon - 10
        
        for i in range(start_idx, end_idx):
            # Features: price, volumes, indicators
            feature_vector = []
            
            # Price features (normalized)
            price_window = prices[i-feature_window:i]
            normalized_prices = (price_window - price_window.mean()) / (price_window.std() + 1e-8)
            feature_vector.extend(normalized_prices)
            
            # Volume features
            volume_window = volumes[i-feature_window:i]
            normalized_volumes = (volume_window - volume_window.mean()) / (volume_window.std() + 1e-8)
            feature_vector.extend(normalized_volumes)
            
            # Technical indicators
            for indicator_name, indicator_values in indicators.items():
                if not np.isnan(indicator_values[i-1]):  # Check availability
                    # Take recent values indicator
                    indicator_window = indicator_values[i-feature_window:i]
                    # Fill NaN average value
                    indicator_window = np.nan_to_num(indicator_window, nan=np.nanmean(indicator_window))
                    # Normalize
                    if indicator_window.std() > 1e-8:
                        indicator_window = (indicator_window - indicator_window.mean()) / indicator_window.std()
                    feature_vector.extend(indicator_window)
            
            # Label: direction price through horizon periods
            current_price = prices[i]
            future_price = prices[i + horizon]
            price_change_pct = (future_price - current_price) / current_price
            
            # Classes directions
            if price_change_pct < -0.02:  # Drop > 2%
                label = 0  # SELL
            elif price_change_pct > 0.02:  # Growth > 2%
                label = 2  # BUY
            else:
                label = 1  # HOLD
            
            features_list.append(feature_vector)
            labels_list.append(label)
        
        return np.array(features_list), np.array(labels_list)
    
    def _split_data(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separates data on support and query sets"""
        
        # Ensure presence all classes
        unique_labels = np.unique(labels)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        
        for class_label in unique_labels:
            class_indices = np.where(labels == class_label)[0]
            
            if len(class_indices) < self.config.num_support + self.config.num_query:
                # Duplicate examples if not enough
                class_indices = np.tile(class_indices, 
                    (self.config.num_support + self.config.num_query) // len(class_indices) + 1
                )[:self.config.num_support + self.config.num_query]
            
            # Randomly select examples
            selected_indices = np.random.choice(
                class_indices, 
                self.config.num_support + self.config.num_query, 
                replace=False
            )
            
            support_indices = selected_indices[:self.config.num_support]
            query_indices = selected_indices[self.config.num_support:]
            
            support_data.append(features[support_indices])
            support_labels.extend([class_label] * len(support_indices))
            query_data.append(features[query_indices])
            query_labels.extend([class_label] * len(query_indices))
        
        return (
            np.vstack(support_data),
            np.array(support_labels),
            np.vstack(query_data),
            np.array(query_labels)
        )
    
    def _compute_direction_difficulty(self, labels: np.ndarray) -> float:
        """Computes complexity tasks directions"""
        unique, counts = np.unique(labels, return_counts=True)
        
        # Entropy as measure complexity
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalize to [0, 1]
        max_entropy = np.log2(len(unique))
        difficulty = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return difficulty
    
    def _get_feature_names(self, indicators: Dict[str, np.ndarray]) -> List[str]:
        """Returns names features"""
        feature_names = []
        
        # Price features
        feature_names.extend([f"price_{i}" for i in range(60)])
        
        # Volume features
        feature_names.extend([f"volume_{i}" for i in range(60)])
        
        # Indicators
        for indicator_name in indicators.keys():
            feature_names.extend([f"{indicator_name}_{i}" for i in range(60)])
        
        return feature_names


class CryptoPortfolioOptimizationTask:
    """
    Task optimization cryptocurrency portfolio
    
    Portfolio Optimization Task
    - Multi-asset portfolio construction
    - Risk-return optimization
    - Dynamic rebalancing
    """
    
    def __init__(self, config: CryptoTaskConfig, simulator: CryptoMarketSimulator):
        self.config = config
        self.simulator = simulator
    
    def generate_task(
        self,
        assets: List[str],
        regime: MarketRegime,
        rebalancing_frequency: str
    ) -> Tuple[Dict[str, torch.Tensor], TaskMetadata]:
        """Generates task optimization portfolio"""
        
        # Generate data for all assets
        series_length = 1000
        asset_data = {}
        
        for asset in assets:
            initial_price = random.uniform(1, 1000)
            prices = self.simulator.generate_price_series(
                initial_price, series_length, regime
            )
            returns = np.diff(np.log(prices))
            asset_data[asset] = {
                'prices': prices,
                'returns': returns
            }
        
        # Create features and targets for portfolio
        features, targets = self._create_portfolio_features_targets(
            asset_data, rebalancing_frequency
        )
        
        # Split data
        support_data, support_targets, query_data, query_targets = self._split_portfolio_data(
            features, targets
        )
        
        task_data = {
            'support_data': torch.FloatTensor(support_data),
            'support_labels': torch.FloatTensor(support_targets),
            'query_data': torch.FloatTensor(query_data),
            'query_labels': torch.FloatTensor(query_targets)
        }
        
        metadata = TaskMetadata(
            task_id=f"portfolio_opt_{'_'.join(assets)}_{regime.value}_{rebalancing_frequency}",
            task_type="regression",
            difficulty=self._compute_portfolio_difficulty(targets),
            source_domain="crypto_portfolio",
            target_variable="optimal_weights",
            feature_names=self._get_portfolio_feature_names(assets),
            data_quality_score=0.85,
            created_timestamp=time.time(),
            trading_pair=f"PORTFOLIO_{len(assets)}assets"
        )
        
        return task_data, metadata
    
    def _create_portfolio_features_targets(
        self,
        asset_data: Dict[str, Dict[str, np.ndarray]],
        rebalancing_frequency: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates features and targets for portfolio"""
        
        # Define period rebalancing
        rebalance_periods = {
            "daily": 1,
            "weekly": 7,
            "monthly": 30
        }
        rebalance_period = rebalance_periods.get(rebalancing_frequency, 7)
        
        # Window for analysis
        lookback_window = 60
        
        features_list = []
        targets_list = []
        
        assets = list(asset_data.keys())
        returns_matrix = np.column_stack([asset_data[asset]['returns'] for asset in assets])
        
        for i in range(lookback_window, len(returns_matrix) - rebalance_period, rebalance_period):
            # Features: statistics profitabilities for lookback period
            feature_vector = []
            
            window_returns = returns_matrix[i-lookback_window:i]
            
            # Average profitability
            mean_returns = np.mean(window_returns, axis=0)
            feature_vector.extend(mean_returns)
            
            # Volatility
            volatilities = np.std(window_returns, axis=0)
            feature_vector.extend(volatilities)
            
            # Correlation matrix (upper triangle)
            corr_matrix = np.corrcoef(window_returns.T)
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            feature_vector.extend(upper_triangle)
            
            # Coefficients Sharpe
            sharpe_ratios = mean_returns / (volatilities + 1e-8)
            feature_vector.extend(sharpe_ratios)
            
            # Target: optimal weights portfolio for next period
            future_returns = returns_matrix[i:i+rebalance_period]
            optimal_weights = self._compute_optimal_weights(future_returns)
            
            features_list.append(feature_vector)
            targets_list.append(optimal_weights)
        
        return np.array(features_list), np.array(targets_list)
    
    def _compute_optimal_weights(self, future_returns: np.ndarray) -> np.ndarray:
        """Computes optimal weights portfolio"""
        # Simplified optimization: maximization Sharpe with constraints
        
        mean_returns = np.mean(future_returns, axis=0)
        cov_matrix = np.cov(future_returns.T)
        
        # Add regularization to covariance matrix
        cov_matrix += np.eye(len(mean_returns)) * 1e-4
        
        try:
            # Compute weights by Markowitz (simplified)
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(len(mean_returns))
            
            # Weights for minimization risk
            risk_min_weights = inv_cov @ ones
            risk_min_weights /= np.sum(risk_min_weights)
            
            # Weights for maximization profitability
            return_max_weights = inv_cov @ mean_returns
            if np.sum(return_max_weights) > 0:
                return_max_weights /= np.sum(return_max_weights)
            else:
                return_max_weights = np.ones(len(mean_returns)) / len(mean_returns)
            
            # Combine (50/50)
            optimal_weights = 0.5 * risk_min_weights + 0.5 * return_max_weights
            
            # Normalize and ensure positivity
            optimal_weights = np.clip(optimal_weights, 0, 1)
            optimal_weights /= np.sum(optimal_weights) if np.sum(optimal_weights) > 0 else 1
            
        except np.linalg.LinAlgError:
            # If matrix irreversible, use equal weights
            optimal_weights = np.ones(len(mean_returns)) / len(mean_returns)
        
        return optimal_weights
    
    def _split_portfolio_data(
        self,
        features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Separates data portfolio on support and query"""
        
        n_total = len(features)
        n_support = min(self.config.num_support * 5, n_total // 2)  # More data for regression
        n_query = min(self.config.num_query * 5, n_total - n_support)
        
        indices = np.random.permutation(n_total)
        support_indices = indices[:n_support]
        query_indices = indices[n_support:n_support + n_query]
        
        return (
            features[support_indices],
            targets[support_indices],
            features[query_indices],
            targets[query_indices]
        )
    
    def _compute_portfolio_difficulty(self, targets: np.ndarray) -> float:
        """Computes complexity tasks portfolio"""
        # Complexity = variability optimal weights
        weight_std = np.std(targets, axis=0).mean()
        return np.clip(weight_std * 10, 0, 1)  # Normalize
    
    def _get_portfolio_feature_names(self, assets: List[str]) -> List[str]:
        """Returns names features for portfolio"""
        feature_names = []
        
        # Average profitability
        feature_names.extend([f"mean_return_{asset}" for asset in assets])
        
        # Volatility
        feature_names.extend([f"volatility_{asset}" for asset in assets])
        
        # Correlation
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets[i+1:], i+1):
                feature_names.append(f"corr_{asset1}_{asset2}")
        
        # Coefficients Sharpe
        feature_names.extend([f"sharpe_{asset}" for asset in assets])
        
        return feature_names


class CryptoTaskDistribution(BaseTaskDistribution):
    """
    Main distribution cryptocurrency tasks
    
    Comprehensive Crypto Task Distribution
    - Multiple task types
    - Market regime awareness
    - Realistic market simulation
    """
    
    def __init__(
        self,
        config: CryptoTaskConfig,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        
        self.crypto_config = config
        self.simulator = CryptoMarketSimulator(config, logger)
        
        # Generators tasks
        self.price_direction_task = CryptoPriceDirectionTask(config, self.simulator)
        self.portfolio_task = CryptoPortfolioOptimizationTask(config, self.simulator)
        
        # Statistics
        self.task_type_counts = defaultdict(int)
        
        self.logger.info(f"CryptoTaskDistribution initialized with config: {config}")
    
    def sample_task(self) -> Dict[str, torch.Tensor]:
        """Samples cryptocurrency task"""
        
        # Select type tasks
        task_types = ["price_direction"]
        if self.crypto_config.include_portfolio_tasks:
            task_types.append("portfolio_optimization")
        
        task_type = random.choice(task_types)
        
        # Select mode market
        regime = random.choice(self.crypto_config.market_regimes)
        
        if task_type == "price_direction":
            trading_pair = random.choice(self.crypto_config.trading_pairs)
            horizon = random.choice(self.crypto_config.prediction_horizons)
            
            task_data, metadata = self.price_direction_task.generate_task(
                trading_pair, regime, horizon
            )
        
        elif task_type == "portfolio_optimization":
            # Select random assets
            num_assets = random.randint(3, min(self.crypto_config.max_assets_in_portfolio, 
                                             len(self.crypto_config.trading_pairs)))
            assets = random.sample(self.crypto_config.trading_pairs, num_assets)
            
            rebalancing_freq = random.choice(self.crypto_config.rebalancing_frequencies)
            
            task_data, metadata = self.portfolio_task.generate_task(
                assets, regime, rebalancing_freq
            )
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Register task
        self.register_task(metadata.task_id, task_data, metadata)
        self.task_type_counts[task_type] += 1
        
        return task_data
    
    def sample_batch(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Samples batch cryptocurrency tasks"""
        return [self.sample_task() for _ in range(batch_size)]
    
    def get_task_difficulty(self, task_data: Dict[str, torch.Tensor]) -> float:
        """Evaluates complexity cryptocurrency tasks"""
        
        support_labels = task_data['support_labels']
        
        if support_labels.dtype == torch.long:
            # Classification
            unique, counts = torch.unique(support_labels, return_counts=True)
            probabilities = counts.float() / len(support_labels)
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
            max_entropy = math.log2(len(unique))
            return (entropy / max_entropy).item() if max_entropy > 0 else 0.5
        else:
            # Regression
            return torch.std(support_labels).item()
    
    def get_crypto_statistics(self) -> Dict[str, Any]:
        """Returns statistics by cryptocurrency tasks"""
        return {
            'task_type_counts': dict(self.task_type_counts),
            'available_pairs': self.crypto_config.trading_pairs,
            'market_regimes': [regime.value for regime in self.crypto_config.market_regimes],
            'total_tasks': sum(self.task_type_counts.values())
        }