"""
GPU-Native Indicator Cache - ALL data in GPU VRAM

Pre-loads ALL indicator data into GPU memory to eliminate CPU bottleneck.
State vector building happens entirely on GPU using torch operations.

Memory usage: ~30 MB for 4 symbols × 50k candles × 40 indicators
GPU has 16 GB VRAM - this is only 0.18% usage!
"""
import torch
import numpy as np
from typing import Dict, List, Optional
import talib


class GPUIndicatorCache:
 """
 Ultra-fast GPU-native indicator cache

 All data stored in GPU VRAM as torch tensors.
 All operations (slicing, indexing) happen on GPU.
 Zero CPU-GPU transfers during training!
 """

 def __init__(
 self,
 candles_df,
 symbols: List[str],
 device: str = 'cuda',
 dtype: torch.dtype = torch.float32,
 verbose: bool = True
 ):
 """
 Initialize GPU indicator cache

 Args:
 candles_df: DataFrame with OHLCV data
 symbols: List of symbols to process
 device: torch device ('cuda' or 'cpu')
 dtype: torch dtype (float32 or bfloat16)
 verbose: Print initialization info
 """
 self.device = torch.device(device if torch.cuda.is_available else 'cpu')
 self.dtype = dtype
 self.symbols = symbols

 if verbose:
 print("=" * 80)
 print("🚀 GPU INDICATOR CACHE - Pre-loading ALL data to GPU VRAM")
 print("=" * 80)

 # Store indicators as GPU tensors
 self.indicators_gpu: Dict[str, torch.Tensor] = {}
 self.ohlcv_gpu: Dict[str, torch.Tensor] = {}
 self.timestamps_gpu: Dict[str, torch.Tensor] = {}

 total_gpu_mb = 0

 for symbol in symbols:
 if verbose:
 print(f" Processing {symbol}...")

 symbol_df = candles_df[candles_df['symbol'] == symbol].copy
 n_candles = len(symbol_df)

 # Calculate indicators on CPU (one-time cost)
 indicators_np = self._calculate_all_indicators(symbol_df)

 # Transfer to GPU immediately!
 indicators_tensor = torch.from_numpy(
 indicators_np.drop(columns=['timestamp']).values.astype(np.float32)
 ).to(device=self.device, dtype=self.dtype)

 self.indicators_gpu[symbol] = indicators_tensor

 # Also store OHLCV on GPU
 ohlcv_np = symbol_df[['open', 'high', 'low', 'close', 'volume']].values.astype(np.float32)
 ohlcv_tensor = torch.from_numpy(ohlcv_np).to(device=self.device, dtype=self.dtype)
 self.ohlcv_gpu[symbol] = ohlcv_tensor

 # Timestamps as int64 for indexing (convert datetime64 to int64 Unix timestamp)
 timestamps = symbol_df['timestamp'].values
 # Convert datetime64 to int64 Unix timestamp if needed
 if timestamps.dtype.kind == 'M': # 'M' = datetime64
 timestamps = timestamps.astype('datetime64[s]').astype(np.int64)
 elif timestamps.dtype != np.int64:
 timestamps = timestamps.astype(np.int64)
 timestamps_tensor = torch.from_numpy(timestamps).to(device=self.device, dtype=torch.int64)
 self.timestamps_gpu[symbol] = timestamps_tensor

 # Calculate memory usage
 indicator_mb = (indicators_tensor.element_size * indicators_tensor.nelement) / 1024 / 1024
 ohlcv_mb = (ohlcv_tensor.element_size * ohlcv_tensor.nelement) / 1024 / 1024
 timestamps_mb = (timestamps_tensor.element_size * timestamps_tensor.nelement) / 1024 / 1024
 total_gpu_mb += indicator_mb + ohlcv_mb + timestamps_mb

 if verbose:
 print(f" ✅ {symbol}: {n_candles} candles → GPU ({indicator_mb:.1f} MB indicators)")

 if verbose:
 print("=" * 80)
 print(f"✅ GPU INDICATOR CACHE READY")
 print("=" * 80)
 print(f"📊 Total GPU memory: {total_gpu_mb:.1f} MB")
 print(f"📊 Symbols: {len(symbols)}")
 print(f"📊 Device: {self.device}")
 print(f"📊 Dtype: {self.dtype}")
 print(f"⚡ ALL DATA IN GPU VRAM - Zero CPU transfers during training!")
 print("=" * 80)

 def _calculate_all_indicators(self, df):
 """Calculate all indicators (CPU one-time cost)"""
 import pandas as pd

 close = df['close'].values
 high = df['high'].values
 low = df['low'].values
 volume = df['volume'].values

 indicators = pd.DataFrame({
 'timestamp': df['timestamp'].values,

 # RSI variants
 'rsi_14': talib.RSI(close, timeperiod=14),
 'rsi_21': talib.RSI(close, timeperiod=21),
 'rsi_7': talib.RSI(close, timeperiod=7),

 # MACD
 'macd': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],
 'macd_signal': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1],
 'macd_hist': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2],

 # Bollinger Bands
 'bb_upper': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[0],
 'bb_middle': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[1],
 'bb_lower': talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)[2],

 # Moving Averages
 'sma_20': talib.SMA(close, timeperiod=20),
 'sma_50': talib.SMA(close, timeperiod=50),
 'sma_200': talib.SMA(close, timeperiod=200),
 'ema_12': talib.EMA(close, timeperiod=12),
 'ema_26': talib.EMA(close, timeperiod=26),

 # ATR
 'atr_14': talib.ATR(high, low, close, timeperiod=14),
 'atr_7': talib.ATR(high, low, close, timeperiod=7),

 # ADX
 'adx_14': talib.ADX(high, low, close, timeperiod=14),
 'plus_di': talib.PLUS_DI(high, low, close, timeperiod=14),
 'minus_di': talib.MINUS_DI(high, low, close, timeperiod=14),

 # Stochastic
 'slowk': talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)[0],
 'slowd': talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)[1],

 # CCI
 'cci_14': talib.CCI(high, low, close, timeperiod=14),
 'cci_20': talib.CCI(high, low, close, timeperiod=20),

 # Williams %R
 'willr_14': talib.WILLR(high, low, close, timeperiod=14),

 # OBV
 'obv': talib.OBV(close, volume),

 # ROC
 'roc_10': talib.ROC(close, timeperiod=10),
 'roc_20': talib.ROC(close, timeperiod=20),

 # MFI
 'mfi_14': talib.MFI(high, low, close, volume, timeperiod=14),

 # SAR
 'sar': talib.SAR(high, low, acceleration=0.02, maximum=0.2),

 # TRIX
 'trix': talib.TRIX(close, timeperiod=30),

 # CMO
 'cmo_14': talib.CMO(close, timeperiod=14),

 # ULTOSC
 'ultosc': talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28),

 # AROON
 'aroon_up': talib.AROON(high, low, timeperiod=25)[0],
 'aroon_down': talib.AROON(high, low, timeperiod=25)[1],

 # TEMA
 'tema_30': talib.TEMA(close, timeperiod=30),

 # KAMA
 'kama_30': talib.KAMA(close, timeperiod=30),

 # APO
 'apo': talib.APO(close, fastperiod=12, slowperiod=26, matype=0),

 # MOM
 'mom_10': talib.MOM(close, timeperiod=10),
 })

 # Fill NaN with neutral values
 indicators = indicators.fillna(method='bfill').fillna(method='ffill').fillna(50.0)

 return indicators

 def get_slice_gpu(
 self,
 symbol: str,
 start_idx: int,
 end_idx: int
 ) -> torch.Tensor:
 """
 Get indicator slice on GPU (ZERO CPU involvement!)

 Args:
 symbol: Trading symbol
 start_idx: Start index
 end_idx: End index

 Returns:
 GPU tensor (end_idx - start_idx, 40)
 """
 return self.indicators_gpu[symbol][start_idx:end_idx, :]

 def get_ohlcv_slice_gpu(
 self,
 symbol: str,
 start_idx: int,
 end_idx: int
 ) -> torch.Tensor:
 """
 Get OHLCV slice on GPU

 Returns:
 GPU tensor (end_idx - start_idx, 5)
 """
 return self.ohlcv_gpu[symbol][start_idx:end_idx, :]

 def get_close_prices_gpu(
 self,
 symbol: str,
 start_idx: int,
 end_idx: int
 ) -> torch.Tensor:
 """
 Get close prices on GPU

 Returns:
 GPU tensor (end_idx - start_idx,)
 """
 return self.ohlcv_gpu[symbol][start_idx:end_idx, 3] # Column 3 is close

 def get_memory_stats(self) -> Dict[str, float]:
 """Get GPU memory statistics"""
 if not torch.cuda.is_available:
 return {}

 allocated_mb = torch.cuda.memory_allocated / 1024 / 1024
 reserved_mb = torch.cuda.memory_reserved / 1024 / 1024
 max_allocated_mb = torch.cuda.max_memory_allocated / 1024 / 1024

 return {
 'allocated_mb': allocated_mb,
 'reserved_mb': reserved_mb,
 'max_allocated_mb': max_allocated_mb,
 }
