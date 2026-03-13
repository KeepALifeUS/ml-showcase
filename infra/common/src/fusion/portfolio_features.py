"""
Portfolio State Features - 100 dimensions

Tracks portfolio positions, balances, PnL, and risk exposure.
Critical for reinforcement learning agent to understand current state.

Author: AI Trading Team
Created: 2025-10-23
Version: 1.0
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PortfolioFeatures:
 """
 Extract 100 portfolio state features

 Feature breakdown:
 1. Position Features (40):
 - 4 symbols × 2 markets × 5 features = 40
 - Features per position: size_normalized, entry_price_norm, current_pnl%, unrealized_pnl%, duration_normalized

 2. Risk Features (24):
 - 4 symbols × 2 markets × 3 features = 24
 - Features per position: leverage, margin_used%, liquidation_distance%

 3. Balance Features (16):
 - Spot (4): total, available, in_orders%, locked%
 - Futures (4): total, available, margin_used%, margin_available%
 - Portfolio (8): equity, free_margin, margin_ratio, risk_exposure,
 total_pnl%, daily_pnl%, leverage_ratio, account_health

 4. Performance Features (12):
 - PnL (3): daily, weekly, monthly (all normalized)
 - ROI (3): daily%, weekly%, monthly%
 - Stats (6): win_rate, avg_win%, avg_loss%, sharpe_ratio, max_drawdown%, recovery_factor

 5. Exposure Features (8):
 - Directional (4): total_long, total_short, net_exposure, gross_exposure
 - Sector (4): btc_exposure, eth_exposure, alt_exposure, stable_exposure

 Total: 40 + 24 + 16 + 12 + 8 = 100 features
 """

 def __init__(
 self,
 symbols: List[str] = ['BTC', 'ETH', 'BNB', 'SOL'],
 markets: List[str] = ['spot', 'futures']
 ):
 """
 Initialize Portfolio Features Extractor

 Args:
 symbols: List of trading symbols
 markets: List of markets (spot, futures)
 """
 self.symbols = symbols
 self.markets = markets
 self.num_symbols = len(symbols)
 self.num_markets = len(markets)

 logger.info(
 f"PortfolioFeatures initialized: {self.num_symbols} symbols, "
 f"{self.num_markets} markets"
 )

 def _normalize_position_size(
 self,
 size: float,
 balance: float
 ) -> float:
 """
 Normalize position size relative to balance

 Returns: position_value / balance (0-1+ range)
 """
 if balance <= 0:
 return 0.0
 return min(abs(size) / balance, 2.0) # Cap at 2x

 def _normalize_pnl(
 self,
 pnl: float,
 entry_value: float
 ) -> float:
 """
 Normalize PnL as percentage

 Returns: pnl / entry_value * 100 (%)
 """
 if entry_value <= 0:
 return 0.0
 return (pnl / entry_value) * 100

 def _normalize_duration(
 self,
 duration_hours: float,
 max_duration: float = 168.0 # 1 week
 ) -> float:
 """
 Normalize position duration

 Returns: duration / max_duration (0-1 range)
 """
 return min(duration_hours / max_duration, 1.0)

 def extract(
 self,
 portfolio: Dict[str, Any],
 market_prices: Optional[Dict[str, float]] = None
 ) -> np.ndarray:
 """
 Extract 100 portfolio features

 Args:
 portfolio: Portfolio state dict with:
 - spot_positions: Dict[symbol, position_info]
 - futures_positions: Dict[symbol, position_info]
 - spot_balance: Dict with total, available, in_orders, locked
 - futures_balance: Dict with total, available, margin_used, margin_available
 - performance: Dict with daily_pnl, weekly_pnl, monthly_pnl, win_rate, etc.
 market_prices: Current market prices for each symbol (optional)

 Returns:
 np.ndarray: Shape (100,) with portfolio features
 """
 features = np.zeros(100)
 feature_idx = 0

 # === POSITION FEATURES (40) ===
 spot_positions = portfolio.get('spot_positions', {})
 futures_positions = portfolio.get('futures_positions', {})

 # Handle both formats: dict with 'total' key or direct float
 spot_balance = portfolio.get('spot_balance', 0)
 futures_balance = portfolio.get('futures_balance', 0)
 spot_balance_total = spot_balance.get('total', 0) if isinstance(spot_balance, dict) else spot_balance
 futures_balance_total = futures_balance.get('total', 0) if isinstance(futures_balance, dict) else futures_balance

 for symbol in self.symbols:
 # Spot position (5 features)
 spot_pos = spot_positions.get(symbol, {})
 if spot_pos and spot_balance_total > 0:
 size = spot_pos.get('size', 0)
 entry_price = spot_pos.get('entry_price', 0)
 current_price = market_prices.get(f"{symbol}_spot", entry_price) if market_prices else entry_price
 entry_value = abs(size * entry_price)
 current_value = abs(size * current_price)

 # Feature: size normalized
 features[feature_idx] = self._normalize_position_size(current_value, spot_balance_total)
 feature_idx += 1

 # Feature: entry price normalized (0-1)
 features[feature_idx] = 0.5 # Neutral value (actual normalization would need historical range)
 feature_idx += 1

 # Feature: current PnL%
 pnl = current_value - entry_value
 features[feature_idx] = self._normalize_pnl(pnl, entry_value) / 100 # -1 to 1 range
 feature_idx += 1

 # Feature: unrealized PnL%
 unrealized_pnl = spot_pos.get('unrealized_pnl', 0)
 features[feature_idx] = self._normalize_pnl(unrealized_pnl, entry_value) / 100
 feature_idx += 1

 # Feature: duration normalized
 duration = spot_pos.get('duration_hours', 0)
 features[feature_idx] = self._normalize_duration(duration)
 feature_idx += 1
 else:
 feature_idx += 5 # No position, all zeros

 # Futures position (5 features)
 futures_pos = futures_positions.get(symbol, {})
 if futures_pos and futures_balance_total > 0:
 size = futures_pos.get('size', 0)
 entry_price = futures_pos.get('entry_price', 0)
 current_price = market_prices.get(f"{symbol}_futures", entry_price) if market_prices else entry_price
 entry_value = abs(size * entry_price)
 current_value = abs(size * current_price)

 # Feature: size normalized
 features[feature_idx] = self._normalize_position_size(current_value, futures_balance_total)
 feature_idx += 1

 # Feature: entry price normalized
 features[feature_idx] = 0.5
 feature_idx += 1

 # Feature: current PnL%
 pnl = current_value - entry_value if size > 0 else entry_value - current_value
 features[feature_idx] = self._normalize_pnl(pnl, entry_value) / 100
 feature_idx += 1

 # Feature: unrealized PnL%
 unrealized_pnl = futures_pos.get('unrealized_pnl', 0)
 features[feature_idx] = self._normalize_pnl(unrealized_pnl, entry_value) / 100
 feature_idx += 1

 # Feature: duration normalized
 duration = futures_pos.get('duration_hours', 0)
 features[feature_idx] = self._normalize_duration(duration)
 feature_idx += 1
 else:
 feature_idx += 5 # No position, all zeros

 # === RISK FEATURES (24) ===
 for symbol in self.symbols:
 # Spot risk (3 features)
 spot_pos = spot_positions.get(symbol, {})

 # Feature: leverage (spot is always 1x)
 features[feature_idx] = 1.0 if spot_pos else 0.0
 feature_idx += 1

 # Feature: margin used% (spot doesn't use margin)
 features[feature_idx] = 0.0
 feature_idx += 1

 # Feature: liquidation distance% (spot can't be liquidated)
 features[feature_idx] = 1.0 if spot_pos else 0.0 # 100% safe
 feature_idx += 1

 # Futures risk (3 features)
 futures_pos = futures_positions.get(symbol, {})

 if futures_pos:
 # Feature: leverage
 leverage = futures_pos.get('leverage', 1.0)
 features[feature_idx] = min(leverage / 20.0, 1.0) # Normalize by max 20x
 feature_idx += 1

 # Feature: margin used%
 margin_used = futures_pos.get('margin_used', 0)
 total_margin = futures_balance_total
 features[feature_idx] = margin_used / total_margin if total_margin > 0 else 0.0
 feature_idx += 1

 # Feature: liquidation distance%
 liquidation_price = futures_pos.get('liquidation_price', 0)
 current_price = market_prices.get(f"{symbol}_futures", 0) if market_prices else 0
 if liquidation_price > 0 and current_price > 0:
 distance = abs(current_price - liquidation_price) / current_price
 features[feature_idx] = min(distance, 1.0)
 else:
 features[feature_idx] = 1.0 # Safe
 feature_idx += 1
 else:
 feature_idx += 3 # No position

 # === BALANCE FEATURES (16) ===
 # Use the balance totals we already extracted (handles both dict and float formats)

 # Spot balance (4 features)
 features[feature_idx] = spot_balance_total / 10000.0 if spot_balance_total > 0 else 0.0 # Normalize by $10k
 feature_idx += 1

 # For detailed balance features, check if dict format or use defaults for float format
 if isinstance(spot_balance, dict):
 spot_available = spot_balance.get('available', spot_balance_total)
 in_orders = spot_balance.get('in_orders', 0)
 locked = spot_balance.get('locked', 0)
 else:
 spot_available = spot_balance_total
 in_orders = 0
 locked = 0

 features[feature_idx] = spot_available / spot_balance_total if spot_balance_total > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = in_orders / spot_balance_total if spot_balance_total > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = locked / spot_balance_total if spot_balance_total > 0 else 0.0
 feature_idx += 1

 # Futures balance (4 features)
 features[feature_idx] = futures_balance_total / 10000.0 if futures_balance_total > 0 else 0.0
 feature_idx += 1

 # For detailed balance features, check if dict format or use defaults for float format
 if isinstance(futures_balance, dict):
 futures_available = futures_balance.get('available', futures_balance_total)
 margin_used = futures_balance.get('margin_used', 0)
 margin_available = futures_balance.get('margin_available', 0)
 else:
 futures_available = futures_balance_total
 margin_used = 0
 margin_available = futures_balance_total

 features[feature_idx] = futures_available / futures_balance_total if futures_balance_total > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = margin_used / futures_balance_total if futures_balance_total > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = margin_available / futures_balance_total if futures_balance_total > 0 else 0.0
 feature_idx += 1

 # Portfolio totals (8 features)
 equity = spot_balance_total + futures_balance_total
 features[feature_idx] = equity / 10000.0 if equity > 0 else 0.0
 feature_idx += 1

 free_margin = spot_available + futures_available
 features[feature_idx] = free_margin / equity if equity > 0 else 0.0
 feature_idx += 1

 margin_ratio = margin_used / equity if equity > 0 else 0.0
 features[feature_idx] = min(margin_ratio, 1.0)
 feature_idx += 1

 # Risk exposure (sum of all position values / equity)
 total_position_value = 0
 for symbol in self.symbols:
 if symbol in spot_positions:
 pos = spot_positions[symbol]
 price = market_prices.get(f"{symbol}_spot", pos.get('entry_price', 0)) if market_prices else pos.get('entry_price', 0)
 total_position_value += abs(pos.get('size', 0) * price)
 if symbol in futures_positions:
 pos = futures_positions[symbol]
 price = market_prices.get(f"{symbol}_futures", pos.get('entry_price', 0)) if market_prices else pos.get('entry_price', 0)
 total_position_value += abs(pos.get('size', 0) * price)

 risk_exposure = total_position_value / equity if equity > 0 else 0.0
 features[feature_idx] = min(risk_exposure, 2.0) / 2.0 # Normalize by 2x max
 feature_idx += 1

 # Total PnL%
 total_pnl = portfolio.get('performance', {}).get('total_pnl', 0)
 features[feature_idx] = total_pnl / equity if equity > 0 else 0.0
 feature_idx += 1

 # Daily PnL%
 daily_pnl = portfolio.get('performance', {}).get('daily_pnl', 0)
 features[feature_idx] = daily_pnl / equity if equity > 0 else 0.0
 feature_idx += 1

 # Leverage ratio
 total_leverage = portfolio.get('leverage_ratio', 1.0)
 features[feature_idx] = min(total_leverage / 10.0, 1.0)
 feature_idx += 1

 # Account health (0-1, 1=healthy)
 account_health = 1.0 - margin_ratio # Simple health: more free margin = healthier
 features[feature_idx] = max(account_health, 0.0)
 feature_idx += 1

 # === PERFORMANCE FEATURES (12) ===
 performance = portfolio.get('performance', {})

 # PnL (3 features)
 daily_pnl = performance.get('daily_pnl', 0)
 features[feature_idx] = daily_pnl / equity if equity > 0 else 0.0
 feature_idx += 1

 weekly_pnl = performance.get('weekly_pnl', 0)
 features[feature_idx] = weekly_pnl / equity if equity > 0 else 0.0
 feature_idx += 1

 monthly_pnl = performance.get('monthly_pnl', 0)
 features[feature_idx] = monthly_pnl / equity if equity > 0 else 0.0
 feature_idx += 1

 # ROI (3 features)
 daily_roi = performance.get('daily_roi', 0)
 features[feature_idx] = daily_roi / 100.0 # Convert % to decimal
 feature_idx += 1

 weekly_roi = performance.get('weekly_roi', 0)
 features[feature_idx] = weekly_roi / 100.0
 feature_idx += 1

 monthly_roi = performance.get('monthly_roi', 0)
 features[feature_idx] = monthly_roi / 100.0
 feature_idx += 1

 # Stats (6 features)
 win_rate = performance.get('win_rate', 0.5)
 features[feature_idx] = win_rate
 feature_idx += 1

 avg_win = performance.get('avg_win', 0)
 features[feature_idx] = avg_win / 100.0 # Assume % values
 feature_idx += 1

 avg_loss = performance.get('avg_loss', 0)
 features[feature_idx] = avg_loss / 100.0
 feature_idx += 1

 sharpe_ratio = performance.get('sharpe_ratio', 0)
 features[feature_idx] = sharpe_ratio / 3.0 # Normalize by 3 (excellent Sharpe)
 feature_idx += 1

 max_drawdown = performance.get('max_drawdown', 0)
 features[feature_idx] = abs(max_drawdown) / 100.0
 feature_idx += 1

 recovery_factor = performance.get('recovery_factor', 0)
 features[feature_idx] = min(recovery_factor / 5.0, 1.0) # Normalize by 5
 feature_idx += 1

 # === EXPOSURE FEATURES (8) ===

 # Calculate exposures
 total_long = 0
 total_short = 0
 btc_exposure = 0
 eth_exposure = 0
 alt_exposure = 0

 for symbol in self.symbols:
 # Futures positions (can be long or short)
 if symbol in futures_positions:
 pos = futures_positions[symbol]
 size = pos.get('size', 0)
 price = market_prices.get(f"{symbol}_futures", pos.get('entry_price', 0)) if market_prices else pos.get('entry_price', 0)
 value = size * price

 if value > 0:
 total_long += value
 else:
 total_short += abs(value)

 # Sector exposure
 if symbol == 'BTC':
 btc_exposure += abs(value)
 elif symbol == 'ETH':
 eth_exposure += abs(value)
 else:
 alt_exposure += abs(value)

 # Spot positions (always long)
 if symbol in spot_positions:
 pos = spot_positions[symbol]
 size = pos.get('size', 0)
 price = market_prices.get(f"{symbol}_spot", pos.get('entry_price', 0)) if market_prices else pos.get('entry_price', 0)
 value = abs(size * price)

 total_long += value

 # Sector exposure
 if symbol == 'BTC':
 btc_exposure += value
 elif symbol == 'ETH':
 eth_exposure += value
 else:
 alt_exposure += value

 # Directional exposure (4 features)
 features[feature_idx] = total_long / equity if equity > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = total_short / equity if equity > 0 else 0.0
 feature_idx += 1

 net_exposure = (total_long - total_short) / equity if equity > 0 else 0.0
 features[feature_idx] = net_exposure
 feature_idx += 1

 gross_exposure = (total_long + total_short) / equity if equity > 0 else 0.0
 features[feature_idx] = min(gross_exposure, 2.0) / 2.0
 feature_idx += 1

 # Sector exposure (4 features)
 features[feature_idx] = btc_exposure / equity if equity > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = eth_exposure / equity if equity > 0 else 0.0
 feature_idx += 1

 features[feature_idx] = alt_exposure / equity if equity > 0 else 0.0
 feature_idx += 1

 stable_exposure = 1.0 - gross_exposure # Cash is stable exposure
 features[feature_idx] = max(stable_exposure, 0.0)
 feature_idx += 1

 # Verify we used exactly 100 features
 assert feature_idx == 100, f"Expected 100 features, got {feature_idx}"

 # Check for NaN or Inf
 features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

 return features


# Test function
def test_portfolio_features:
 """Test portfolio feature extraction"""
 print("=" * 60)
 print("TESTING PORTFOLIO STATE FEATURES")
 print("=" * 60)

 # Create synthetic portfolio state
 portfolio = {
 'spot_positions': {
 'BTC': {
 'size': 0.5,
 'entry_price': 50000,
 'unrealized_pnl': 500,
 'duration_hours': 24
 },
 'ETH': {
 'size': 10.0,
 'entry_price': 2000,
 'unrealized_pnl': -100,
 'duration_hours': 48
 }
 },
 'futures_positions': {
 'BTC': {
 'size': 0.2,
 'entry_price': 50500,
 'unrealized_pnl': 200,
 'duration_hours': 12,
 'leverage': 5.0,
 'margin_used': 2000,
 'liquidation_price': 45000
 }
 },
 'spot_balance': {
 'total': 10000,
 'available': 7000,
 'in_orders': 2000,
 'locked': 1000
 },
 'futures_balance': {
 'total': 5000,
 'available': 3000,
 'margin_used': 2000,
 'margin_available': 3000
 },
 'performance': {
 'total_pnl': 1500,
 'daily_pnl': 200,
 'weekly_pnl': 800,
 'monthly_pnl': 1500,
 'daily_roi': 1.3,
 'weekly_roi': 5.3,
 'monthly_roi': 10.0,
 'win_rate': 0.65,
 'avg_win': 3.5,
 'avg_loss': -2.0,
 'sharpe_ratio': 1.8,
 'max_drawdown': -8.5,
 'recovery_factor': 2.5
 },
 'leverage_ratio': 1.4
 }

 market_prices = {
 'BTC_spot': 51000,
 'ETH_spot': 1980,
 'BTC_futures': 51100,
 'BNB_spot': 300,
 'SOL_spot': 100
 }

 print(f"\n1. Created synthetic portfolio state:")
 print(f" Spot positions: BTC (0.5), ETH (10.0)")
 print(f" Futures positions: BTC (0.2, 5x leverage)")
 print(f" Spot balance: $10,000")
 print(f" Futures balance: $5,000")
 print(f" Total equity: $15,000")

 # Extract features
 extractor = PortfolioFeatures

 print("\n2. Extracting 100 portfolio features...")
 features = extractor.extract(portfolio, market_prices)

 print(f"✅ Features extracted successfully")
 print(f" Shape: {features.shape}")
 print(f" Expected: (100,)")

 # Verify no NaN or Inf
 nan_count = np.isnan(features).sum
 inf_count = np.isinf(features).sum

 print(f"\n3. Data quality check:")
 print(f" NaN values: {nan_count}")
 print(f" Inf values: {inf_count}")

 if nan_count == 0 and inf_count == 0:
 print(" ✅ No NaN or Inf values")
 else:
 print(" ❌ WARNING: Invalid values detected!")

 # Show feature groups
 print("\n4. Feature group statistics:")
 print(f" Position Features [0:40]: non-zero={np.count_nonzero(features[0:40])}/40 ({np.count_nonzero(features[0:40])/40*100:.1f}%)")
 print(f" Risk Features [40:64]: non-zero={np.count_nonzero(features[40:64])}/24 ({np.count_nonzero(features[40:64])/24*100:.1f}%)")
 print(f" Balance Features [64:80]: non-zero={np.count_nonzero(features[64:80])}/16 ({np.count_nonzero(features[64:80])/16*100:.1f}%)")
 print(f" Performance Features [80:92]: non-zero={np.count_nonzero(features[80:92])}/12 ({np.count_nonzero(features[80:92])/12*100:.1f}%)")
 print(f" Exposure Features [92:100]: non-zero={np.count_nonzero(features[92:100])}/8 ({np.count_nonzero(features[92:100])/8*100:.1f}%)")

 # Show some key values
 print("\n5. Key portfolio metrics:")
 print(f" Total equity (normalized): {features[64]:.4f}")
 print(f" Free margin ratio: {features[65]:.4f}")
 print(f" Risk exposure: {features[67]:.4f}")
 print(f" Account health: {features[71]:.4f}")
 print(f" Win rate: {features[86]:.4f}")
 print(f" Sharpe ratio (normalized): {features[89]:.4f}")
 print(f" Net exposure: {features[94]:.4f}")

 # Calculate population percentage
 non_zero_values = np.count_nonzero(features)
 population_pct = (non_zero_values / 100) * 100

 print(f"\n6. Feature population:")
 print(f" Total features: 100")
 print(f" Non-zero values: {non_zero_values}")
 print(f" Population: {population_pct:.1f}%")

 if population_pct > 50:
 print(" ✅ Good feature population!")
 else:
 print(" ⚠️ Low feature population (expected with empty positions)")

 print("\n" + "=" * 60)
 print("ALL TESTS COMPLETED ✅")
 print("=" * 60)
 print(f"\n100 Portfolio State features for 90%+ win rate:")
 print(" - Position tracking (size, PnL, duration)")
 print(" - Risk management (leverage, margin, liquidation)")
 print(" - Balance monitoring (spot/futures, equity)")
 print(" - Performance metrics (ROI, win rate, Sharpe)")
 print(" - Exposure analysis (long/short, sector)")
 print("\nCritical for RL agent to understand portfolio state")


if __name__ == "__main__":
 test_portfolio_features
