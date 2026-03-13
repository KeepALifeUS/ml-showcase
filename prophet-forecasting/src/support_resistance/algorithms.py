"""
Support/Resistance Detection Algorithms
ML-Framework-1332 - Multiple detection algorithms for comprehensive level identification

 2025: Advanced technical analysis algorithms with performance optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LevelData:
    """Data structure for detected level"""
    price: float
    type: str  # 'support', 'resistance', 'pivot'
    strength: float
    metadata: Dict[str, Any]


class PivotPointCalculator:
    """
    Calculate pivot points using various methods.

    Supports Classical, Fibonacci, Camarilla, and Woodie pivot calculations.
    """

    def __init__(self):
        self.logger = logging.getLogger("PivotPointCalculator")

    async def calculate_pivots(
        self,
        data: pd.DataFrame,
        method: str = "classical"
    ) -> List[Dict[str, Any]]:
        """Calculate pivot points using specified method"""

        if len(data) == 0:
            return []

        # Get last period's data
        last_high = data['high'].iloc[-1]
        last_low = data['low'].iloc[-1]
        last_close = data['close'].iloc[-1]

        if method == "classical":
            return self._calculate_classical_pivots(last_high, last_low, last_close)
        elif method == "fibonacci":
            return self._calculate_fibonacci_pivots(last_high, last_low, last_close)
        elif method == "camarilla":
            return self._calculate_camarilla_pivots(last_high, last_low, last_close)
        elif method == "woodie":
            return self._calculate_woodie_pivots(data)
        else:
            return self._calculate_classical_pivots(last_high, last_low, last_close)

    def _calculate_classical_pivots(
        self,
        high: float,
        low: float,
        close: float
    ) -> List[Dict[str, Any]]:
        """Calculate classical pivot points"""
        pivot = (high + low + close) / 3

        # Resistance levels
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        # Support levels
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        levels = [
            {'price': pivot, 'type': 'pivot', 'strength': 0.9, 'label': 'PP'},
            {'price': r1, 'type': 'resistance', 'strength': 0.8, 'label': 'R1'},
            {'price': r2, 'type': 'resistance', 'strength': 0.7, 'label': 'R2'},
            {'price': r3, 'type': 'resistance', 'strength': 0.6, 'label': 'R3'},
            {'price': s1, 'type': 'support', 'strength': 0.8, 'label': 'S1'},
            {'price': s2, 'type': 'support', 'strength': 0.7, 'label': 'S2'},
            {'price': s3, 'type': 'support', 'strength': 0.6, 'label': 'S3'}
        ]

        return levels

    def _calculate_fibonacci_pivots(
        self,
        high: float,
        low: float,
        close: float
    ) -> List[Dict[str, Any]]:
        """Calculate Fibonacci pivot points"""
        pivot = (high + low + close) / 3
        range_hl = high - low

        # Fibonacci ratios
        r1 = pivot + (0.382 * range_hl)
        r2 = pivot + (0.618 * range_hl)
        r3 = pivot + (1.000 * range_hl)

        s1 = pivot - (0.382 * range_hl)
        s2 = pivot - (0.618 * range_hl)
        s3 = pivot - (1.000 * range_hl)

        levels = [
            {'price': pivot, 'type': 'pivot', 'strength': 0.9, 'label': 'PP'},
            {'price': r1, 'type': 'resistance', 'strength': 0.75, 'label': 'R1 (38.2%)'},
            {'price': r2, 'type': 'resistance', 'strength': 0.85, 'label': 'R2 (61.8%)'},
            {'price': r3, 'type': 'resistance', 'strength': 0.7, 'label': 'R3 (100%)'},
            {'price': s1, 'type': 'support', 'strength': 0.75, 'label': 'S1 (38.2%)'},
            {'price': s2, 'type': 'support', 'strength': 0.85, 'label': 'S2 (61.8%)'},
            {'price': s3, 'type': 'support', 'strength': 0.7, 'label': 'S3 (100%)'}
        ]

        return levels

    def _calculate_camarilla_pivots(
        self,
        high: float,
        low: float,
        close: float
    ) -> List[Dict[str, Any]]:
        """Calculate Camarilla pivot points"""
        range_hl = high - low

        # Camarilla equations
        r4 = close + range_hl * 1.1 / 2
        r3 = close + range_hl * 1.1 / 4
        r2 = close + range_hl * 1.1 / 6
        r1 = close + range_hl * 1.1 / 12

        s1 = close - range_hl * 1.1 / 12
        s2 = close - range_hl * 1.1 / 6
        s3 = close - range_hl * 1.1 / 4
        s4 = close - range_hl * 1.1 / 2

        pivot = (high + low + close) / 3

        levels = [
            {'price': pivot, 'type': 'pivot', 'strength': 0.85, 'label': 'PP'},
            {'price': r1, 'type': 'resistance', 'strength': 0.6, 'label': 'R1'},
            {'price': r2, 'type': 'resistance', 'strength': 0.7, 'label': 'R2'},
            {'price': r3, 'type': 'resistance', 'strength': 0.8, 'label': 'R3'},
            {'price': r4, 'type': 'resistance', 'strength': 0.9, 'label': 'R4'},
            {'price': s1, 'type': 'support', 'strength': 0.6, 'label': 'S1'},
            {'price': s2, 'type': 'support', 'strength': 0.7, 'label': 'S2'},
            {'price': s3, 'type': 'support', 'strength': 0.8, 'label': 'S3'},
            {'price': s4, 'type': 'support', 'strength': 0.9, 'label': 'S4'}
        ]

        return levels

    def _calculate_woodie_pivots(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate Woodie pivot points"""
        if len(data) < 2:
            return []

        # Woodie uses current open
        high = data['high'].iloc[-1]
        low = data['low'].iloc[-1]
        close = data['close'].iloc[-1]
        open_price = data['open'].iloc[-1] if 'open' in data.columns else close

        # Woodie formula gives more weight to close
        pivot = (high + low + 2 * close) / 4

        r1 = 2 * pivot - low
        r2 = pivot + (high - low)

        s1 = 2 * pivot - high
        s2 = pivot - (high - low)

        levels = [
            {'price': pivot, 'type': 'pivot', 'strength': 0.85, 'label': 'PP (Woodie)'},
            {'price': r1, 'type': 'resistance', 'strength': 0.75, 'label': 'R1'},
            {'price': r2, 'type': 'resistance', 'strength': 0.65, 'label': 'R2'},
            {'price': s1, 'type': 'support', 'strength': 0.75, 'label': 'S1'},
            {'price': s2, 'type': 'support', 'strength': 0.65, 'label': 'S2'}
        ]

        return levels


class PeakTroughDetector:
    """
    Detect peaks and troughs in price data using fractal analysis.
    """

    def __init__(self, window_size: int = 5, min_prominence: float = 0.01):
        self.window_size = window_size
        self.min_prominence = min_prominence
        self.logger = logging.getLogger("PeakTroughDetector")

    async def detect(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect peaks and troughs in price data"""

        if len(data) < self.window_size * 2:
            return []

        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values

        # Smooth the data slightly to reduce noise
        smoothed_close = gaussian_filter1d(close_prices, sigma=1)

        # Find peaks (resistance levels)
        peaks = await self._find_peaks(high_prices, smoothed_close)

        # Find troughs (support levels)
        troughs = await self._find_troughs(low_prices, smoothed_close)

        # Calculate fractal dimension for validation
        fractal_dim = self._calculate_fractal_dimension(close_prices)

        # Combine and validate levels
        levels = []

        for peak_idx, peak_price, peak_strength in peaks:
            levels.append({
                'price': peak_price,
                'type': 'peak',
                'strength': peak_strength,
                'index': peak_idx,
                'touch_count': self._count_touches(peak_price, high_prices, tolerance=0.001),
                'fractal_dimension': fractal_dim
            })

        for trough_idx, trough_price, trough_strength in troughs:
            levels.append({
                'price': trough_price,
                'type': 'trough',
                'strength': trough_strength,
                'index': trough_idx,
                'touch_count': self._count_touches(trough_price, low_prices, tolerance=0.001),
                'fractal_dimension': fractal_dim
            })

        return levels

    async def _find_peaks(
        self,
        high_prices: np.ndarray,
        smoothed_prices: np.ndarray
    ) -> List[Tuple[int, float, float]]:
        """Find peaks in price data"""

        # Use scipy signal to find peaks
        peak_indices, properties = signal.find_peaks(
            smoothed_prices,
            distance=self.window_size,
            prominence=smoothed_prices.std() * self.min_prominence
        )

        peaks = []
        for idx in peak_indices:
            # Use actual high price at peak
            peak_price = high_prices[idx]

            # Calculate strength based on prominence and sharpness
            prominence = properties['prominences'][list(peak_indices).index(idx)]
            strength = min(1.0, prominence / (smoothed_prices.std() * 2))

            peaks.append((idx, peak_price, strength))

        return peaks

    async def _find_troughs(
        self,
        low_prices: np.ndarray,
        smoothed_prices: np.ndarray
    ) -> List[Tuple[int, float, float]]:
        """Find troughs in price data"""

        # Invert for trough detection
        inverted = -smoothed_prices

        trough_indices, properties = signal.find_peaks(
            inverted,
            distance=self.window_size,
            prominence=smoothed_prices.std() * self.min_prominence
        )

        troughs = []
        for idx in trough_indices:
            # Use actual low price at trough
            trough_price = low_prices[idx]

            # Calculate strength
            prominence = properties['prominences'][list(trough_indices).index(idx)]
            strength = min(1.0, prominence / (smoothed_prices.std() * 2))

            troughs.append((idx, trough_price, strength))

        return troughs

    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Normalize prices
            normalized = (prices - prices.min()) / (prices.max() - prices.min())

            # Calculate differences
            diffs = np.abs(np.diff(normalized))

            # Box-counting approximation
            n = len(diffs)
            if n < 2:
                return 1.5  # Default

            # Calculate fractal dimension
            log_n = np.log(n)
            log_range = np.log(diffs.max() - diffs.min() + 1e-10)

            if log_range != 0:
                fractal_dim = 1 + log_n / log_range
                return np.clip(fractal_dim, 1.0, 2.0)
            else:
                return 1.5

        except Exception as e:
            self.logger.warning(f"Fractal dimension calculation failed: {e}")
            return 1.5

    def _count_touches(
        self,
        level_price: float,
        prices: np.ndarray,
        tolerance: float = 0.001
    ) -> int:
        """Count how many times price touched a level"""
        touches = 0
        tolerance_range = level_price * tolerance

        for price in prices:
            if abs(price - level_price) <= tolerance_range:
                touches += 1

        return touches


class VolumeProfileAnalyzer:
    """
    Analyze volume profile to identify high-volume price levels.
    """

    def __init__(self, num_bins: int = 50, min_volume_percentile: float = 70):
        self.num_bins = num_bins
        self.min_volume_percentile = min_volume_percentile
        self.logger = logging.getLogger("VolumeProfileAnalyzer")

    async def analyze_profile(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze volume profile to find significant levels"""

        if len(data) < 10:
            return []

        levels = []

        # Calculate volume profile
        profile = self._calculate_volume_profile(data)

        if profile is None:
            return []

        # Find Point of Control (POC) - highest volume price
        poc_price, poc_volume = self._find_poc(profile)
        if poc_price:
            levels.append({
                'price': poc_price,
                'type': 'poc',
                'strength': 1.0,
                'volume_concentration': poc_volume,
                'is_poc': True,
                'position': self._determine_position(poc_price, data['close'].iloc[-1])
            })

        # Find Value Area (70% of volume)
        value_area = self._calculate_value_area(profile)
        if value_area:
            vah, val = value_area
            levels.append({
                'price': vah,
                'type': 'vah',
                'strength': 0.8,
                'volume_concentration': profile.get(vah, 0),
                'is_poc': False,
                'position': self._determine_position(vah, data['close'].iloc[-1])
            })
            levels.append({
                'price': val,
                'type': 'val',
                'strength': 0.8,
                'volume_concentration': profile.get(val, 0),
                'is_poc': False,
                'position': self._determine_position(val, data['close'].iloc[-1])
            })

        # Find high volume nodes (HVN)
        hvn_levels = self._find_high_volume_nodes(profile)
        for hvn_price, hvn_volume in hvn_levels:
            levels.append({
                'price': hvn_price,
                'type': 'hvn',
                'strength': min(1.0, hvn_volume / poc_volume) if poc_volume else 0.7,
                'volume_concentration': hvn_volume,
                'is_poc': False,
                'position': self._determine_position(hvn_price, data['close'].iloc[-1])
            })

        return levels

    def _calculate_volume_profile(self, data: pd.DataFrame) -> Optional[Dict[float, float]]:
        """Calculate volume distribution across price levels"""
        try:
            # Create price bins
            price_min = data['low'].min()
            price_max = data['high'].max()
            bins = np.linspace(price_min, price_max, self.num_bins)

            # Calculate volume at each price level
            profile = {}

            for _, row in data.iterrows():
                # Distribute volume across the candle's range
                candle_low = row['low']
                candle_high = row['high']
                candle_volume = row['volume']

                # Find bins that overlap with this candle
                for i in range(len(bins) - 1):
                    bin_low = bins[i]
                    bin_high = bins[i + 1]

                    # Check for overlap
                    overlap_low = max(bin_low, candle_low)
                    overlap_high = min(bin_high, candle_high)

                    if overlap_high > overlap_low:
                        # Calculate proportion of volume for this bin
                        overlap_range = overlap_high - overlap_low
                        candle_range = candle_high - candle_low

                        if candle_range > 0:
                            volume_proportion = overlap_range / candle_range
                            bin_volume = candle_volume * volume_proportion

                            # Add to profile
                            bin_price = (bin_low + bin_high) / 2
                            profile[bin_price] = profile.get(bin_price, 0) + bin_volume

            return profile

        except Exception as e:
            self.logger.error(f"Volume profile calculation failed: {e}")
            return None

    def _find_poc(self, profile: Dict[float, float]) -> Tuple[Optional[float], Optional[float]]:
        """Find Point of Control (highest volume price)"""
        if not profile:
            return None, None

        poc_price = max(profile.keys(), key=lambda k: profile[k])
        poc_volume = profile[poc_price]

        return poc_price, poc_volume

    def _calculate_value_area(
        self,
        profile: Dict[float, float]
    ) -> Optional[Tuple[float, float]]:
        """Calculate Value Area High (VAH) and Value Area Low (VAL)"""
        if not profile:
            return None

        # Sort by price
        sorted_prices = sorted(profile.keys())
        total_volume = sum(profile.values())
        target_volume = total_volume * 0.7  # 70% of volume

        # Find POC
        poc_price, _ = self._find_poc(profile)
        if not poc_price:
            return None

        # Expand from POC to find value area
        accumulated_volume = profile[poc_price]
        upper_idx = sorted_prices.index(poc_price)
        lower_idx = upper_idx

        while accumulated_volume < target_volume:
            # Check which direction to expand
            upper_volume = profile[sorted_prices[upper_idx + 1]] if upper_idx < len(sorted_prices) - 1 else 0
            lower_volume = profile[sorted_prices[lower_idx - 1]] if lower_idx > 0 else 0

            if upper_volume >= lower_volume and upper_idx < len(sorted_prices) - 1:
                upper_idx += 1
                accumulated_volume += upper_volume
            elif lower_idx > 0:
                lower_idx -= 1
                accumulated_volume += lower_volume
            else:
                break

        vah = sorted_prices[upper_idx]
        val = sorted_prices[lower_idx]

        return vah, val

    def _find_high_volume_nodes(
        self,
        profile: Dict[float, float]
    ) -> List[Tuple[float, float]]:
        """Find High Volume Nodes (HVN)"""
        if not profile:
            return []

        # Calculate volume threshold
        volumes = list(profile.values())
        threshold = np.percentile(volumes, self.min_volume_percentile)

        # Find prices above threshold
        hvn_levels = []
        for price, volume in profile.items():
            if volume >= threshold:
                hvn_levels.append((price, volume))

        # Sort by volume
        hvn_levels.sort(key=lambda x: x[1], reverse=True)

        # Return top levels (excluding POC)
        poc_price, _ = self._find_poc(profile)
        hvn_levels = [(p, v) for p, v in hvn_levels if p != poc_price]

        return hvn_levels[:5]  # Top 5 HVN levels

    def _determine_position(self, level_price: float, current_price: float) -> str:
        """Determine if level is above or below current price"""
        return "below_price" if level_price < current_price else "above_price"


class PsychologicalLevelDetector:
    """
    Detect psychological price levels (round numbers, quarters, etc.)
    """

    def __init__(self):
        self.logger = logging.getLogger("PsychologicalLevelDetector")

    async def detect_levels(
        self,
        current_price: float,
        price_range: float
    ) -> List[Dict[str, Any]]:
        """Detect psychological levels near current price"""

        levels = []

        # Determine appropriate round number intervals based on price
        intervals = self._get_intervals(current_price)

        # Calculate search range
        search_min = current_price - price_range / 2
        search_max = current_price + price_range / 2

        for interval, strength, level_type in intervals:
            # Find round numbers within range
            start = int(search_min / interval) * interval
            end = int(search_max / interval) * interval + interval

            current = start
            while current <= end:
                if search_min <= current <= search_max:
                    levels.append({
                        'price': current,
                        'strength': strength,
                        'type': level_type
                    })
                current += interval

        # Sort by distance from current price
        levels.sort(key=lambda x: abs(x['price'] - current_price))

        return levels[:20]  # Return top 20 closest levels

    def _get_intervals(self, price: float) -> List[Tuple[float, float, str]]:
        """Get appropriate intervals based on price magnitude"""

        if price < 1:
            return [
                (0.01, 0.5, 'cent'),
                (0.05, 0.6, 'nickel'),
                (0.10, 0.7, 'dime'),
                (0.25, 0.8, 'quarter'),
                (0.50, 0.85, 'half'),
                (1.00, 0.9, 'dollar')
            ]
        elif price < 10:
            return [
                (0.25, 0.5, 'quarter'),
                (0.50, 0.6, 'half'),
                (1.00, 0.8, 'whole'),
                (5.00, 0.9, 'five')
            ]
        elif price < 100:
            return [
                (1, 0.5, 'whole'),
                (5, 0.6, 'five'),
                (10, 0.8, 'ten'),
                (25, 0.85, 'quarter_hundred'),
                (50, 0.9, 'half_hundred')
            ]
        elif price < 1000:
            return [
                (10, 0.5, 'ten'),
                (25, 0.6, 'quarter_hundred'),
                (50, 0.7, 'half_hundred'),
                (100, 0.9, 'hundred')
            ]
        elif price < 10000:
            return [
                (50, 0.5, 'fifty'),
                (100, 0.7, 'hundred'),
                (250, 0.75, 'quarter_thousand'),
                (500, 0.8, 'half_thousand'),
                (1000, 0.9, 'thousand')
            ]
        else:
            return [
                (100, 0.5, 'hundred'),
                (500, 0.6, 'five_hundred'),
                (1000, 0.8, 'thousand'),
                (5000, 0.85, 'five_thousand'),
                (10000, 0.9, 'ten_thousand')
            ]


class MLLevelDetector:
    """
    Machine learning-based support/resistance detection.
    Uses historical patterns to predict future levels.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"MLLevelDetector.{symbol}")
        self.model = None
        self.feature_names = []

    async def predict_levels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict support/resistance levels using ML"""

        if len(data) < 50:
            return []  # Need sufficient data for ML

        try:
            # Extract features
            features = self._extract_features(data)

            # Use statistical methods as fallback (no pre-trained model)
            levels = await self._statistical_prediction(data, features)

            return levels

        except Exception as e:
            self.logger.error(f"ML prediction failed: {e}")
            return []

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML prediction"""
        features = pd.DataFrame()

        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()

        # Technical indicators
        features['rsi'] = self._calculate_rsi(data['close'])
        features['bb_position'] = self._bollinger_position(data['close'])

        # Price patterns
        features['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
        features['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)

        # Volume patterns
        features['volume_spike'] = (data['volume'] > data['volume'].rolling(20).mean() * 1.5).astype(int)

        return features.fillna(0)

    async def _statistical_prediction(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Statistical approach to predict levels"""
        levels = []

        # Use clustering to find price levels
        from sklearn.cluster import DBSCAN

        # Prepare price data for clustering
        prices = data[['high', 'low', 'close']].values.flatten()
        prices = prices.reshape(-1, 1)

        # Cluster to find price levels
        clustering = DBSCAN(eps=data['close'].std() * 0.02, min_samples=3)
        clusters = clustering.fit_predict(prices)

        # Extract cluster centers as potential levels
        unique_clusters = set(clusters)
        unique_clusters.discard(-1)  # Remove noise

        for cluster_id in unique_clusters:
            cluster_prices = prices[clusters == cluster_id]
            level_price = np.mean(cluster_prices)

            # Determine type based on current price
            current_price = data['close'].iloc[-1]
            level_type = 'support' if level_price < current_price else 'resistance'

            # Calculate confidence based on cluster size
            confidence = min(1.0, len(cluster_prices) / (len(prices) * 0.05))

            levels.append({
                'price': float(level_price),
                'type': level_type,
                'confidence': confidence,
                'feature_importance': {
                    'cluster_size': len(cluster_prices),
                    'std_dev': float(np.std(cluster_prices))
                }
            })

        return levels

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _bollinger_position(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)

        position = (prices - lower_band) / (upper_band - lower_band)

        return position