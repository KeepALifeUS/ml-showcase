"""
Portfolio State Feature Engineering
Position Tracking

Extracts 20-dimensional features representing current portfolio state:
- Position quantities (4 symbols)
- Position values (4 symbols)
- Position weights (4 symbols)
- Exposure metrics (8 features)

Look-ahead bias protection: Uses only current state, no future data
Performance: <0.5ms per extraction
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def extract_position_features(
 portfolio: Dict[str, Any],
 current_prices: Dict[str, float],
 symbols: Optional[List[str]] = None
) -> np.ndarray:
 """
 Extract 20-dimensional position state features

 Args:
 portfolio: Portfolio state {positions: {symbol: quantity}, cash: float, total_value: float}
 current_prices: Dict of symbol -> current price
 symbols: List of symbols (defaults to ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'])

 Returns:
 np.ndarray: 20-dimensional feature vector
 [0-3]: Position quantities (BTC, ETH, BNB, SOL)
 [4-7]: Position values (BTC, ETH, BNB, SOL)
 [8-11]: Position weights (BTC, ETH, BNB, SOL)
 [12-19]: Exposure metrics (8 features)

 Example:
 portfolio = {
 'positions': {'BTCUSDT': 0.5, 'ETHUSDT': 2.0},
 'cash': 5000.0,
 'total_value': 100000.0
 }
 prices = {'BTCUSDT': 50000, 'ETHUSDT': 3000, 'BNBUSDT': 400, 'SOLUSDT': 100}
 features = extract_position_features(portfolio, prices)
 """
 if symbols is None:
 symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT']

 features = np.zeros(20, dtype=np.float32)

 positions = portfolio.get('positions', {})
 cash = portfolio.get('cash', 0.0)
 total_value = portfolio.get('total_value', 0.0)

 # 1. Position quantities (4 dims) - always populate, even if total_value <= 0
 for i, symbol in enumerate(symbols):
 features[i] = positions.get(symbol, 0.0)

 # Return early with only position quantities if total_value is invalid
 if total_value <= 0:
 logger.warning("Portfolio total_value is 0 or negative, returning position quantities only")
 return features

 # 2. Position values (4 dims)
 position_values = []
 for i, symbol in enumerate(symbols):
 quantity = positions.get(symbol, 0.0)
 price = current_prices.get(symbol, 0.0)
 value = quantity * price
 features[4 + i] = value
 position_values.append(value)

 # 3. Position weights (4 dims)
 weights = calculate_position_weights(positions, current_prices, total_value)
 for i, symbol in enumerate(symbols):
 features[8 + i] = weights.get(symbol, 0.0)

 # 4. Exposure metrics (8 dims)
 exposure_metrics = calculate_exposure_metrics(positions, current_prices, cash, total_value)
 features[12] = exposure_metrics['total_exposure']
 features[13] = exposure_metrics['cash_ratio']
 features[14] = exposure_metrics['long_exposure']
 features[15] = exposure_metrics['short_exposure']
 features[16] = exposure_metrics['net_exposure']
 features[17] = exposure_metrics['gross_exposure']
 features[18] = exposure_metrics['leverage']

 # 5. Concentration (1 dim)
 concentration = calculate_concentration_metrics(position_values)
 features[19] = concentration['herfindahl_index']

 return features


def calculate_position_weights(
 positions: Dict[str, float],
 current_prices: Dict[str, float],
 total_value: float
) -> Dict[str, float]:
 """
 Calculate position weights (% of portfolio)

 Args:
 positions: Dict of symbol -> quantity
 current_prices: Dict of symbol -> price
 total_value: Total portfolio value

 Returns:
 Dict of symbol -> weight (0 to 1)
 """
 if total_value <= 0:
 return {symbol: 0.0 for symbol in positions.keys}

 weights = {}
 for symbol, quantity in positions.items:
 price = current_prices.get(symbol, 0.0)
 value = quantity * price
 weights[symbol] = value / total_value

 return weights


def calculate_exposure_metrics(
 positions: Dict[str, float],
 current_prices: Dict[str, float],
 cash: float,
 total_value: float
) -> Dict[str, float]:
 """
 Calculate portfolio exposure metrics

 Returns:
 Dict with keys:
 - total_exposure: Sum of all position values
 - cash_ratio: Cash / Total value
 - long_exposure: Sum of long positions
 - short_exposure: Absolute sum of short positions
 - net_exposure: Long - Short
 - gross_exposure: Long + Short
 - leverage: Gross / Total value
 """
 long_exposure = 0.0
 short_exposure = 0.0

 for symbol, quantity in positions.items:
 price = current_prices.get(symbol, 0.0)
 value = quantity * price

 if value > 0:
 long_exposure += value
 else:
 short_exposure += abs(value)

 total_exposure = long_exposure + short_exposure
 net_exposure = long_exposure - short_exposure
 gross_exposure = total_exposure

 cash_ratio = cash / total_value if total_value > 0 else 0.0
 leverage = gross_exposure / total_value if total_value > 0 else 0.0

 return {
 'total_exposure': total_exposure,
 'cash_ratio': cash_ratio,
 'long_exposure': long_exposure,
 'short_exposure': short_exposure,
 'net_exposure': net_exposure,
 'gross_exposure': gross_exposure,
 'leverage': leverage,
 }


def calculate_concentration_metrics(position_values: List[float]) -> Dict[str, float]:
 """
 Calculate portfolio concentration metrics

 Args:
 position_values: List of position values (can be negative)

 Returns:
 Dict with keys:
 - herfindahl_index: H = sum(weight_i^2), measures concentration
 - max_weight: Maximum position weight
 - diversification_ratio: 1/H, number of effective positions
 """
 # Calculate absolute values and total
 abs_values = [abs(v) for v in position_values]
 total = sum(abs_values)

 if total <= 0:
 return {
 'herfindahl_index': 0.0,
 'max_weight': 0.0,
 'diversification_ratio': 0.0,
 }

 # Calculate weights
 weights = [v / total for v in abs_values]

 # Herfindahl-Hirschman Index (HHI)
 herfindahl = sum(w ** 2 for w in weights)

 # Maximum weight
 max_weight = max(weights) if weights else 0.0

 # Diversification ratio (number of effective positions)
 diversification_ratio = 1.0 / herfindahl if herfindahl > 0 else 0.0

 return {
 'herfindahl_index': herfindahl,
 'max_weight': max_weight,
 'diversification_ratio': diversification_ratio,
 }
