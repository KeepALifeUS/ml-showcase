"""
ML Testing Framework - Test Data Generator
 test data for ML models

Enterprise Pattern: Data Quality & Testing Infrastructure
- Generation data for testing
- various market conditions edge cases
- Enterprise reproducibility versioning data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from dataclasses import dataclass, field
import random
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class MarketCondition(Enum):
    """Types market conditions for crypto trading"""
 BULL_MARKET = "bull_market" # market
 BEAR_MARKET = "bear_market" # market
 SIDEWAYS = "sideways" # trend
    HIGH_VOLATILITY = "high_volatility"   # High volatility
    LOW_VOLATILITY = "low_volatility"     # Low volatility
 FLASH_CRASH = "flash_crash" #
 PUMP_AND_DUMP = "pump_and_dump" #
 NORMAL = "normal" #


class DataQualityIssue(Enum):
    """Types issues quality data"""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATE_ROWS = "duplicate_rows"
    INCONSISTENT_TIMESTAMPS = "inconsistent_timestamps"
    ZERO_VOLUME = "zero_volume"
    NEGATIVE_PRICES = "negative_prices"
    GAPS_IN_DATA = "gaps_in_data"
    CONSTANT_VALUES = "constant_values"


@dataclass
class DataGenerationConfig:
    """Configuration for data"""
    n_samples: int = 1000
    n_features: int = 10
    n_classes: int = 2
    random_state: int = 42
    noise_level: float = 0.1
    
    # Temporal
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = field(default_factory=datetime.now)
 frequency: str = "1H" # data (1H, 5min, 1D)
    
    # parameters
 initial_price: float = 50000.0 # price (onexample, BTC)
 volatility: float = 0.02 # volatility
 drift: float = 0.0001 # (trend)
    
    # Quality data
    missing_data_ratio: float = 0.05
    outlier_ratio: float = 0.02
    duplicate_ratio: float = 0.01
    
    # Enterprise Settings
    reproducible: bool = True
    validate_output: bool = True
    include_metadata: bool = True


class TestDataGenerator:
    """
 test data for ML Testing Framework
    
    Enterprise Pattern: Comprehensive Test Data Management
 - Generation financial temporal
 - Create data
 - various issues quality data for testing
    """
    
    def __init__(self, config: Optional[DataGenerationConfig] = None):
        self.config = config or DataGenerationConfig()
        if self.config.reproducible:
            self._set_random_seeds(self.config.random_state)
    
    def _set_random_seeds(self, seed: int) -> None:
        """seed for"""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_crypto_price_data(
        self,
        market_condition: MarketCondition = MarketCondition.NORMAL,
        symbol: str = "BTCUSDT"
    ) -> pd.DataFrame:
        """
 Generation data
        
        Args:
            market_condition: Type market conditions
 symbol: trading
        
        Returns:
 DataFrame : timestamp, open, high, low, close, volume
        """
        # Create temporal labels
        date_range = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq=self.config.frequency
        )
        
        n_points = len(date_range)
        prices = np.zeros(n_points)
        volumes = np.zeros(n_points)
        
        # price
        price = self.config.initial_price
        prices[0] = price
        
        # Parameters from market conditions
        vol, drift, volume_multiplier = self._get_market_parameters(market_condition)
        
        # Generation prices usage
        for i in range(1, n_points):
            # Main trend +
            random_change = np.random.normal(drift, vol)
            
            # for various
            random_change = self._apply_market_specific_changes(
                random_change, market_condition, i, n_points
            )
            
            price *= (1 + random_change)
            prices[i] = price
            
            # Generation
            base_volume = 1000000 * volume_multiplier
            volume_noise = np.random.lognormal(0, 0.5)
            volumes[i] = base_volume * volume_noise
        
        # Create OHLC data from close prices
        df = self._create_ohlc_from_close(prices, volumes, date_range, symbol)
        
        return df
    
    def _get_market_parameters(self, market_condition: MarketCondition) -> Tuple[float, float, float]:
        """Retrieval parameters for various market conditions"""
        base_vol = self.config.volatility
        base_drift = self.config.drift
        
        params = {
            MarketCondition.BULL_MARKET: (base_vol * 0.8, base_drift * 5, 1.2),
            MarketCondition.BEAR_MARKET: (base_vol * 1.2, -base_drift * 3, 0.8),
            MarketCondition.SIDEWAYS: (base_vol * 0.6, base_drift * 0.1, 0.7),
            MarketCondition.HIGH_VOLATILITY: (base_vol * 2.5, base_drift, 1.5),
            MarketCondition.LOW_VOLATILITY: (base_vol * 0.3, base_drift, 0.6),
            MarketCondition.FLASH_CRASH: (base_vol * 1.5, base_drift, 1.0),
            MarketCondition.PUMP_AND_DUMP: (base_vol * 2.0, base_drift, 2.0),
            MarketCondition.NORMAL: (base_vol, base_drift, 1.0)
        }
        
        return params.get(market_condition, params[MarketCondition.NORMAL])
    
    def _apply_market_specific_changes(
        self,
        change: float,
        market_condition: MarketCondition,
        index: int,
        total_points: int
    ) -> float:
        """changes for market conditions"""
        
        if market_condition == MarketCondition.FLASH_CRASH:
            if 0.4 * total_points <= index <= 0.6 * total_points:
 if np.random.random() < 0.1: # 10%
                    change -= 0.1  # -10% change
        
        elif market_condition == MarketCondition.PUMP_AND_DUMP:
 if index < 0.3 * total_points: # Pump
                change += np.random.exponential(0.02)
 elif index > 0.7 * total_points: # Dump
                change -= np.random.exponential(0.03)
        
        return change
    
    def _create_ohlc_from_close(
        self,
        close_prices: np.ndarray,
        volumes: np.ndarray,
        timestamps: pd.DatetimeIndex,
        symbol: str
    ) -> pd.DataFrame:
        """Create OHLC data from closing prices"""
        n_points = len(close_prices)
        
        # Generation prices ( )
        open_prices = np.zeros(n_points)
        open_prices[0] = close_prices[0]
        for i in range(1, n_points):
 gap = np.random.normal(0, 0.001) # gap
            open_prices[i] = close_prices[i-1] * (1 + gap)
        
        # Generation maximum minimum
        high_prices = np.maximum(open_prices, close_prices)
        low_prices = np.minimum(open_prices, close_prices)
        
        # Addition
        for i in range(n_points):
 wick_size = np.random.exponential(0.005) #
            high_prices[i] *= (1 + wick_size)
            low_prices[i] *= (1 - wick_size)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': symbol,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        return df
    
    def generate_features_dataset(
        self,
        task_type: str = "classification",
        include_technical_indicators: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generation suite features for ML models
        
        Args:
            task_type: Type tasks ('classification' or 'regression')
 include_technical_indicators:
        
        Returns:
 Tuple[DataFrame, Series]: Features target variable
        """
        if task_type == "classification":
            X, y = make_classification(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                n_classes=self.config.n_classes,
                n_informative=max(2, self.config.n_features // 2),
                n_redundant=self.config.n_features // 4,
                noise=self.config.noise_level,
                random_state=self.config.random_state
            )
        else:
            X, y = make_regression(
                n_samples=self.config.n_samples,
                n_features=self.config.n_features,
                noise=self.config.noise_level * 10,
                random_state=self.config.random_state
            )
        
        # Create DataFrame
        feature_names = [f'feature_{i}' for i in range(self.config.n_features)]
        
        if include_technical_indicators:
            # Addition " "
            tech_features = self._generate_technical_features(X)
            X = np.column_stack([X, tech_features])
            feature_names.extend(['rsi', 'macd', 'bollinger_upper', 'bollinger_lower', 'volume_sma'])
        
        df = pd.DataFrame(X, columns=feature_names)
        target = pd.Series(y, name='target')
        
        return df, target
    
    def _generate_technical_features(self, base_features: np.ndarray) -> np.ndarray:
        """Generation -"""
        n_samples = base_features.shape[0]
        
        # RSI (0-100)
        rsi = np.random.uniform(20, 80, n_samples)
        
        # MACD (-1 to 1)
        macd = np.random.normal(0, 0.5, n_samples)
        
        # Bollinger Bands (relative values)
        bollinger_upper = np.random.uniform(0.8, 1.2, n_samples)
        bollinger_lower = np.random.uniform(-1.2, -0.8, n_samples)
        
        # Volume SMA (positive values)
        volume_sma = np.random.lognormal(0, 1, n_samples)
        
        return np.column_stack([rsi, macd, bollinger_upper, bollinger_lower, volume_sma])
    
    def inject_data_quality_issues(
        self,
        df: pd.DataFrame,
        issues: List[DataQualityIssue],
        severity: float = 0.1
    ) -> pd.DataFrame:
        """
 issues quality data for testing
        
        Args:
 df: DataFrame
 issues: List issues for
 severity: issues (0.0 - 1.0)
        
        Returns:
 DataFrame issue quality data
        """
        df_corrupted = df.copy()
        n_samples, n_features = df_corrupted.shape
        
        for issue in issues:
            if issue == DataQualityIssue.MISSING_VALUES:
                mask = np.random.random((n_samples, n_features)) < (severity * 0.5)
                df_corrupted = df_corrupted.mask(mask)
            
            elif issue == DataQualityIssue.OUTLIERS:
                # Extreme outliers
                for col in df_corrupted.select_dtypes(include=[np.number]).columns:
                    n_outliers = int(n_samples * severity * 0.1)
                    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
                    
                    col_std = df_corrupted[col].std()
                    col_mean = df_corrupted[col].mean()
                    
                    # outliers (±5 )
                    outlier_values = np.random.choice(
                        [col_mean + 5*col_std, col_mean - 5*col_std], 
                        n_outliers
                    )
                    df_corrupted.loc[outlier_indices, col] = outlier_values
            
            elif issue == DataQualityIssue.DUPLICATE_ROWS:
                # Duplicate
                n_duplicates = int(n_samples * severity * 0.2)
                if n_duplicates > 0:
                    duplicate_indices = np.random.choice(n_samples, n_duplicates, replace=False)
                    duplicate_rows = df_corrupted.iloc[duplicate_indices].copy()
                    df_corrupted = pd.concat([df_corrupted, duplicate_rows], ignore_index=True)
            
            elif issue == DataQualityIssue.CONSTANT_VALUES:
                # values
                n_constant_cols = max(1, int(n_features * severity * 0.3))
                constant_cols = np.random.choice(df_corrupted.columns, n_constant_cols, replace=False)
                
                for col in constant_cols:
                    if col in df_corrupted.select_dtypes(include=[np.number]).columns:
                        df_corrupted[col] = df_corrupted[col].iloc[0]
            
            elif issue == DataQualityIssue.INCONSISTENT_TIMESTAMPS:
                # For data
                if 'timestamp' in df_corrupted.columns:
                    n_inconsistent = int(n_samples * severity * 0.1)
                    if n_inconsistent > 0:
                        inconsistent_indices = np.random.choice(n_samples, n_inconsistent, replace=False)
                        # temporal labels
                        random_dates = pd.date_range(
                            start=self.config.start_date - timedelta(days=365),
                            end=self.config.start_date,
                            periods=n_inconsistent
                        )
                        df_corrupted.loc[inconsistent_indices, 'timestamp'] = random_dates
            
            elif issue == DataQualityIssue.ZERO_VOLUME:
                # volumes (for financial data)
                if 'volume' in df_corrupted.columns:
                    n_zero_volume = int(n_samples * severity * 0.2)
                    zero_indices = np.random.choice(n_samples, n_zero_volume, replace=False)
                    df_corrupted.loc[zero_indices, 'volume'] = 0
            
            elif issue == DataQualityIssue.NEGATIVE_PRICES:
                # prices
                price_cols = [col for col in df_corrupted.columns 
                             if any(price_word in col.lower() for price_word in ['price', 'open', 'high', 'low', 'close'])]
                
                for col in price_cols:
                    n_negative = int(n_samples * severity * 0.05)
                    if n_negative > 0:
                        negative_indices = np.random.choice(n_samples, n_negative, replace=False)
                        df_corrupted.loc[negative_indices, col] *= -1
            
            elif issue == DataQualityIssue.GAPS_IN_DATA:
                # ( gaps data)
                if 'timestamp' in df_corrupted.columns:
                    n_gaps = int(n_samples * severity * 0.1)
                    if n_gaps > 0:
                        gap_indices = np.random.choice(n_samples, n_gaps, replace=False)
                        df_corrupted = df_corrupted.drop(gap_indices).reset_index(drop=True)
        
        return df_corrupted
    
    def generate_time_series_with_patterns(
        self,
        patterns: List[str] = None,
        length: int = 1000
    ) -> pd.DataFrame:
        """
 Generation temporal
        
        Args:
 patterns: List ('trend', 'seasonality', 'cycles', 'noise')
 length:
        
        Returns:
 DataFrame
        """
        if patterns is None:
            patterns = ['trend', 'seasonality', 'noise']
        
        timestamps = pd.date_range(
            start=self.config.start_date,
            periods=length,
            freq=self.config.frequency
        )
        
        values = np.zeros(length)
        
        # value
        base_value = 100
        
        # Addition
        for pattern in patterns:
            if pattern == 'trend':
                # trend
                trend = np.linspace(0, 50, length)
                values += trend
            
            elif pattern == 'seasonality':
                daily_season = 10 * np.sin(2 * np.pi * np.arange(length) / 24)
                weekly_season = 5 * np.sin(2 * np.pi * np.arange(length) / (24 * 7))
                values += daily_season + weekly_season
            
            elif pattern == 'cycles':
                cycle1 = 15 * np.sin(2 * np.pi * np.arange(length) / 200)
                cycle2 = 8 * np.sin(2 * np.pi * np.arange(length) / 50)
                values += cycle1 + cycle2
            
            elif pattern == 'noise':
                noise = np.random.normal(0, 5, length)
                values += noise
        
        values += base_value
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values
        })
        
        return df
    
    def generate_multivariate_dataset(
        self,
        n_variables: int = 5,
        correlation_structure: str = "mixed"
    ) -> pd.DataFrame:
        """
 Generation suite data
        
        Args:
 n_variables:
            correlation_structure: Type correlation ('independent', 'high_corr', 'mixed')
        
        Returns:
 DataFrame data
        """
        n_samples = self.config.n_samples
        
        if correlation_structure == "independent":
            data = np.random.normal(0, 1, (n_samples, n_variables))
        
        elif correlation_structure == "high_corr":
            base_var = np.random.normal(0, 1, (n_samples, 1))
            noise = np.random.normal(0, 0.1, (n_samples, n_variables))
            data = np.repeat(base_var, n_variables, axis=1) + noise
        
        else:  # mixed
            group1_size = n_variables // 2
            base1 = np.random.normal(0, 1, (n_samples, 1))
            noise1 = np.random.normal(0, 0.3, (n_samples, group1_size))
            group1 = np.repeat(base1, group1_size, axis=1) + noise1
            
            group2_size = n_variables - group1_size
            group2 = np.random.normal(0, 1, (n_samples, group2_size))
            
            data = np.column_stack([group1, group2])
        
        # Create DataFrame
        columns = [f'var_{i}' for i in range(n_variables)]
        df = pd.DataFrame(data, columns=columns)
        
        # Addition temporal labels
        timestamps = pd.date_range(
            start=self.config.start_date,
            periods=n_samples,
            freq=self.config.frequency
        )
        df['timestamp'] = timestamps
        
        return df
    
    def generate_edge_case_data(self, case_type: str) -> pd.DataFrame:
        """
        Generation edge case data for testing
        
        Args:
            case_type: Type edge case
 - 'empty':
 - 'single_row':
 - 'single_column':
                - 'all_nan': All values NaN
 - 'all_zeros': All
                - 'extreme_values': Extreme values
        
        Returns:
 DataFrame edge case data
        """
        if case_type == 'empty':
            return pd.DataFrame()
        
        elif case_type == 'single_row':
            return pd.DataFrame({
                'feature_0': [1.0],
                'feature_1': [2.0],
                'target': [0]
            })
        
        elif case_type == 'single_column':
            return pd.DataFrame({
                'feature_0': np.random.normal(0, 1, 100)
            })
        
        elif case_type == 'all_nan':
            df = pd.DataFrame(
                np.full((50, 5), np.nan),
                columns=[f'feature_{i}' for i in range(5)]
            )
            return df
        
        elif case_type == 'all_zeros':
            return pd.DataFrame(
                np.zeros((100, 5)),
                columns=[f'feature_{i}' for i in range(5)]
            )
        
        elif case_type == 'extreme_values':
            data = np.random.normal(0, 1, (100, 5))
            # Extreme values
 data[0, :] = 1e10 # Very
 data[1, :] = -1e10 # Very
 data[2, :] = np.inf #
 data[3, :] = -np.inf # infinity
            
            return pd.DataFrame(
                data,
                columns=[f'feature_{i}' for i in range(5)]
            )
        
        else:
            raise ValueError(f"type edge case: {case_type}")
    
    def generate_test_datasets_suite(self) -> Dict[str, pd.DataFrame]:
        """
 Generation full suite test
        
        Returns:
 Dictionary test data
        """
        datasets = {}
        
        # Main types data
        datasets['crypto_normal'] = self.generate_crypto_price_data(MarketCondition.NORMAL)
        datasets['crypto_volatile'] = self.generate_crypto_price_data(MarketCondition.HIGH_VOLATILITY)
        datasets['crypto_crash'] = self.generate_crypto_price_data(MarketCondition.FLASH_CRASH)
        
        # Data issue quality
        clean_data = self.generate_crypto_price_data(MarketCondition.NORMAL)
        datasets['data_with_missing'] = self.inject_data_quality_issues(
            clean_data, [DataQualityIssue.MISSING_VALUES]
        )
        datasets['data_with_outliers'] = self.inject_data_quality_issues(
            clean_data, [DataQualityIssue.OUTLIERS]
        )
        
        # Feature datasets
        features_class, target_class = self.generate_features_dataset('classification')
        datasets['features_classification'] = features_class
        
        features_reg, target_reg = self.generate_features_dataset('regression')
        datasets['features_regression'] = features_reg
        
        # Edge cases
        datasets['empty_data'] = self.generate_edge_case_data('empty')
        datasets['single_row'] = self.generate_edge_case_data('single_row')
        datasets['extreme_values'] = self.generate_edge_case_data('extreme_values')
        
        # Temporal
        datasets['time_series_trend'] = self.generate_time_series_with_patterns(['trend', 'noise'])
        datasets['time_series_seasonal'] = self.generate_time_series_with_patterns(['seasonality', 'noise'])
        
        # data
        datasets['multivariate_mixed'] = self.generate_multivariate_dataset(
            n_variables=8, correlation_structure="mixed"
        )
        
        return datasets


def create_crypto_trading_data_generator(
    samples: int = 1000,
    volatility: float = 0.02,
    random_state: int = 42
) -> TestDataGenerator:
    """
    Factory function for creation generator data for crypto trading
    
    Args:
 samples:
 volatility: Level
 random_state: Seed for
    
    Returns:
        TestDataGenerator: Configured generator data
    """
    config = DataGenerationConfig(
        n_samples=samples,
        volatility=volatility,
        random_state=random_state,
        reproducible=True,
        validate_output=True,
        include_metadata=True
    )
    
    return TestDataGenerator(config)


# Example usage
if __name__ == "__main__":
    # Create generator for crypto trading
    generator = create_crypto_trading_data_generator(
        samples=1000,
        volatility=0.025,
        random_state=42
    )
    
    # Generation various types data
    print("=== Generation cryptocurrency data ===")
    crypto_data = generator.generate_crypto_price_data(
        market_condition=MarketCondition.BULL_MARKET,
        symbol="ETHUSDT"
    )
    print(f"Crypto data shape: {crypto_data.shape}")
    print(f"Columns: {list(crypto_data.columns)}")
    print(f"Price range: {crypto_data['close'].min():.2f} - {crypto_data['close'].max():.2f}")
    
    print("\n=== Generation features ===")
    features, target = generator.generate_features_dataset(
        task_type="classification",
        include_technical_indicators=True
    )
    print(f"Features shape: {features.shape}")
    print(f"Target distribution: {target.value_counts().to_dict()}")
    
    print("\n=== Data issue quality ===")
    corrupted_data = generator.inject_data_quality_issues(
        features,
        issues=[DataQualityIssue.MISSING_VALUES, DataQualityIssue.OUTLIERS],
        severity=0.1
    )
    missing_ratio = corrupted_data.isnull().sum().sum() / (corrupted_data.shape[0] * corrupted_data.shape[1])
    print(f"Missing values ratio: {missing_ratio:.3f}")
    
    print("\n=== suite ===")
    datasets_suite = generator.generate_test_datasets_suite()
    for name, dataset in datasets_suite.items():
        print(f"{name}: {dataset.shape}")
    
    print("\n=== Edge Cases ===")
    edge_cases = ['empty', 'single_row', 'extreme_values']
    for case in edge_cases:
        try:
            edge_data = generator.generate_edge_case_data(case)
            print(f"{case}: {edge_data.shape}")
        except Exception as e:
            print(f"{case}: Error - {e}")