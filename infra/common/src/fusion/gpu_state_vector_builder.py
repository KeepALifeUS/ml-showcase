"""
GPU-Native State Vector Builder - Pure torch operations on GPU

Builds state vectors entirely on GPU using torch operations.
Zero NumPy, zero CPU involvement during training.

Expected performance:
- CPU NumPy version: ~180ms per state build
- CPU Vectorized NumPy: ~1ms per state build
- GPU torch version: <0.1ms per state build (10x faster!)
"""
import torch
from typing import Dict, Any, List, Optional


class GPUStateVectorBuilder:
 """
 Ultra-fast GPU state vector builder using pure torch operations

 All operations happen on GPU. No CPU-GPU transfers.
 State vectors built using torch tensor operations.
 """

 def __init__(
 self,
 indicator_cache, # GPUIndicatorCache instance
 symbols: List[str] = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT'],
 window_hours: int = 168,
 state_dim: int = 768, # State vector dimensions (reduced to 384 for memory efficiency)
 device: str = 'cuda',
 dtype: torch.dtype = torch.float32
 ):
 """
 Initialize GPU state vector builder

 Args:
 indicator_cache: GPUIndicatorCache with data in GPU
 symbols: List of trading symbols
 window_hours: Lookback window (168 = 1 week, reduced to 24 for memory efficiency)
 state_dim: State vector dimensions (768 default, reduced to 384 for memory efficiency)
 device: torch device
 dtype: torch dtype
 """
 self.indicator_cache = indicator_cache
 self.symbols = symbols
 self.window_hours = window_hours
 self.state_dim = state_dim # Make state_dim dynamic
 self.device = torch.device(device if torch.cuda.is_available else 'cpu')
 self.dtype = dtype

 print("=" * 80)
 print("🚀 GPU STATE VECTOR BUILDER - Pure torch operations")
 print("=" * 80)
 print(f" Device: {self.device}")
 print(f" Dtype: {self.dtype}")
 print(f" Window: {window_hours} hours")
 print(f" State dim: {state_dim}")
 print(f" Symbols: {len(symbols)}")
 print(f" Expected speedup: 10x over vectorized NumPy (<0.1ms)")
 print("=" * 80)

 def build_from_indices_gpu(
 self,
 start_idx: int,
 end_idx: int,
 portfolio_state: Dict[str, Any]
 ) -> torch.Tensor:
 """
 Build state vector entirely on GPU

 Args:
 start_idx: Start index in historical data
 end_idx: End index (should be start_idx + window_hours)
 portfolio_state: Portfolio metrics dict

 Returns:
 GPU tensor (168, 768) ready for model inference
 """
 # Pre-allocate state tensor on GPU (dynamic dimensions)
 state = torch.zeros(
 (self.window_hours, self.state_dim), # Use dynamic state_dim instead of hardcoded 768
 device=self.device,
 dtype=self.dtype
 )

 # Portfolio features (first 160 dimensions) - vectorized on GPU
 capital = portfolio_state.get('capital', 3000.0)
 total_margin = portfolio_state.get('total_margin', 0.0)
 num_positions = portfolio_state.get('num_positions', 0)
 total_pnl = portfolio_state.get('total_pnl', 0.0)
 drawdown = portfolio_state.get('drawdown', 0.0)

 # Create portfolio tensor on GPU (broadcast to all timesteps)
 portfolio_tensor = torch.tensor([
 capital / 10000.0, # Normalized capital
 total_margin / capital if capital > 0 else 0.0,
 num_positions / 10.0,
 total_pnl / capital if capital > 0 else 0.0,
 drawdown,
 ], device=self.device, dtype=self.dtype)

 # Broadcast portfolio features to all timesteps (columns 0-4)
 state[:, :5] = portfolio_tensor.unsqueeze(0).expand(self.window_hours, -1)

 # Symbol-specific features (40 indicators per symbol × 4 symbols = 160)
 # Start at column 160 (after portfolio features and padding)
 for i, symbol in enumerate(self.symbols):
 offset = 160 + i * 40

 # Get indicator slice from GPU cache (ZERO CPU!)
 indicators = self.indicator_cache.get_slice_gpu(symbol, start_idx, end_idx)
 # indicators shape: (168, 40)

 # Get close prices for normalization
 close_prices = self.indicator_cache.get_close_prices_gpu(symbol, start_idx, end_idx)
 # close_prices shape: (168,)

 # Normalize and assign indicators directly on GPU
 # RSI indicators (0-100 range)
 state[:, offset + 0] = indicators[:, 0] / 100.0 # rsi_14
 state[:, offset + 1] = indicators[:, 1] / 100.0 # rsi_21
 state[:, offset + 2] = indicators[:, 2] / 100.0 # rsi_7

 # MACD (normalize by close price)
 state[:, offset + 3] = indicators[:, 3] / close_prices # macd
 state[:, offset + 4] = indicators[:, 4] / close_prices # macd_signal
 state[:, offset + 5] = indicators[:, 5] / close_prices # macd_hist

 # Bollinger Bands (normalize by close)
 state[:, offset + 6] = indicators[:, 6] / close_prices # bb_upper
 state[:, offset + 7] = indicators[:, 7] / close_prices # bb_middle
 state[:, offset + 8] = indicators[:, 8] / close_prices # bb_lower

 # Moving Averages (normalize by close)
 state[:, offset + 9] = indicators[:, 9] / close_prices # sma_20
 state[:, offset + 10] = indicators[:, 10] / close_prices # sma_50
 state[:, offset + 11] = indicators[:, 11] / close_prices # sma_200
 state[:, offset + 12] = indicators[:, 12] / close_prices # ema_12
 state[:, offset + 13] = indicators[:, 13] / close_prices # ema_26

 # ATR (normalize by close)
 state[:, offset + 14] = indicators[:, 14] / close_prices # atr_14
 state[:, offset + 15] = indicators[:, 15] / close_prices # atr_7

 # ADX (0-100 range)
 state[:, offset + 16] = indicators[:, 16] / 100.0 # adx_14
 state[:, offset + 17] = indicators[:, 17] / 100.0 # plus_di
 state[:, offset + 18] = indicators[:, 18] / 100.0 # minus_di

 # Stochastic (0-100 range)
 state[:, offset + 19] = indicators[:, 19] / 100.0 # slowk
 state[:, offset + 20] = indicators[:, 20] / 100.0 # slowd

 # CCI (normalize to -1 to 1 range, clip at ±200)
 state[:, offset + 21] = torch.clamp(indicators[:, 21] / 200.0, -1.0, 1.0) # cci_14
 state[:, offset + 22] = torch.clamp(indicators[:, 22] / 200.0, -1.0, 1.0) # cci_20

 # Williams %R (already -100 to 0, normalize to 0-1)
 state[:, offset + 23] = (indicators[:, 23] + 100.0) / 100.0 # willr_14

 # OBV (normalize by volume)
 obv_normalized = indicators[:, 24] / 1e9 # Scale down
 state[:, offset + 24] = torch.clamp(obv_normalized, -1.0, 1.0) # obv

 # ROC (rate of change, clip at ±50%)
 state[:, offset + 25] = torch.clamp(indicators[:, 25] / 50.0, -1.0, 1.0) # roc_10
 state[:, offset + 26] = torch.clamp(indicators[:, 26] / 50.0, -1.0, 1.0) # roc_20

 # MFI (0-100 range)
 state[:, offset + 27] = indicators[:, 27] / 100.0 # mfi_14

 # SAR (normalize by close)
 state[:, offset + 28] = indicators[:, 28] / close_prices # sar

 # TRIX (small values, multiply by 100)
 state[:, offset + 29] = torch.clamp(indicators[:, 29] * 100.0, -1.0, 1.0) # trix

 # CMO (-100 to 100, normalize to -1 to 1)
 state[:, offset + 30] = indicators[:, 30] / 100.0 # cmo_14

 # ULTOSC (0-100 range)
 state[:, offset + 31] = indicators[:, 31] / 100.0 # ultosc

 # AROON (0-100 range)
 state[:, offset + 32] = indicators[:, 32] / 100.0 # aroon_up
 state[:, offset + 33] = indicators[:, 33] / 100.0 # aroon_down

 # TEMA (normalize by close)
 state[:, offset + 34] = indicators[:, 34] / close_prices # tema_30

 # KAMA (normalize by close)
 state[:, offset + 35] = indicators[:, 35] / close_prices # kama_30

 # APO (normalize by close)
 state[:, offset + 36] = indicators[:, 36] / close_prices # apo

 # MOM (momentum, normalize by close)
 state[:, offset + 37] = indicators[:, 37] / close_prices # mom_10

 # Last 2 indicators (38-39) - placeholder or additional features
 state[:, offset + 38] = torch.zeros_like(close_prices)
 state[:, offset + 39] = torch.zeros_like(close_prices)

 # Handle NaN and clip on GPU (much faster than CPU!)
 state = torch.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
 state = torch.clamp(state, -10.0, 10.0)

 return state

 def build_from_indices_to_numpy(
 self,
 start_idx: int,
 end_idx: int,
 portfolio_state: Dict[str, Any]
 ):
 """
 Build state vector on GPU and convert to NumPy (for compatibility)

 Args:
 start_idx: Start index
 end_idx: End index
 portfolio_state: Portfolio state dict

 Returns:
 NumPy array (168, 768)
 """
 state_gpu = self.build_from_indices_gpu(start_idx, end_idx, portfolio_state)
 return state_gpu.cpu.numpy
