"""
Temporal Encoding specialized for temporal series in crypto trading.
Accounts for market cyclicality, time zones, and trading activity patterns.

Production temporal encodings for real-time trading systems.
"""

import math
import warnings
from typing import Optional, Tuple, Union, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalEncodingConfig:
    """Configuration for temporal encoding."""
    d_model: int = 512
    max_seq_len: int = 10000
    dropout: float = 0.1
    
    # Time-based encoding parameters
    use_cyclical_encoding: bool = True
    use_trading_hours: bool = True
    use_market_sessions: bool = True
    use_seasonal_patterns: bool = True
    
    # Cyclical patterns
    minute_cycles: int = 60       # Minutes in hour
    hour_cycles: int = 24         # Hours in day  
    day_cycles: int = 7           # Days in week
    month_cycles: int = 12        # Months in year
    
    # Trading session parameters
    session_overlap_penalty: float = 0.8  # Penalty for overlapping sessions
    weekend_penalty: float = 0.5          # Penalty for weekend trading
    
    # Seasonal patterns
    seasonal_periods: List[int] = None     # Custom seasonal periods in hours
    seasonal_strength: float = 0.3        # Strength of seasonal effects
    
    # Crypto-specific patterns
    use_crypto_cycles: bool = True
    halving_cycle_years: float = 4.0       # Bitcoin halving cycle
    use_funding_rates: bool = True         # 8-hour funding rate cycles
    
    # Market microstructure
    use_market_open_close: bool = True
    use_volume_cycles: bool = True
    use_volatility_cycles: bool = True
    
    def __post_init__(self):
        if self.seasonal_periods is None:
            # Default seasonal periods: daily, weekly, monthly (in hours)
            self.seasonal_periods = [24, 168, 720]


class CyclicalTimeEncoding(nn.Module):
    """Cyclical encoding for various temporal patterns."""
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Cyclical projections for each temporal cycle
        if config.use_cyclical_encoding:
            self.minute_proj = nn.Linear(2, config.d_model // 16)  # sin, cos
            self.hour_proj = nn.Linear(2, config.d_model // 8)
            self.day_proj = nn.Linear(2, config.d_model // 8)
            self.month_proj = nn.Linear(2, config.d_model // 16)
        
        # Seasonal pattern projections
        if config.use_seasonal_patterns:
            self.seasonal_projections = nn.ModuleDict({
                f'period_{period}': nn.Linear(2, config.d_model // len(config.seasonal_periods))
                for period in config.seasonal_periods
            })
        
        # Crypto-specific cycle projections
        if config.use_crypto_cycles:
            self.halving_proj = nn.Linear(2, config.d_model // 16)
            
        if config.use_funding_rates:
            self.funding_proj = nn.Linear(2, config.d_model // 32)  # 8-hour cycles
        
        # Combination layer
        total_features = self._calculate_total_features()
        self.combination = nn.Linear(total_features, config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def _calculate_total_features(self) -> int:
        """Calculate total number of cyclical features."""
        total = 0
        
        if self.config.use_cyclical_encoding:
            total += self.config.d_model // 16 * 2  # minute + month
            total += self.config.d_model // 8 * 2   # hour + day
        
        if self.config.use_seasonal_patterns:
            total += self.config.d_model  # All seasonal periods combined
        
        if self.config.use_crypto_cycles:
            total += self.config.d_model // 16
        
        if self.config.use_funding_rates:
            total += self.config.d_model // 32
        
        return total
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Apply cyclical time encoding.
        
        Args:
            timestamps: Unix timestamps [batch_size, seq_len]
            
        Returns:
            encoded_time: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = timestamps.shape
        features = []
        
        # Extract time components
        if self.config.use_cyclical_encoding:
            # Minutes, hours, days, months
            minutes = (timestamps // 60) % self.config.minute_cycles
            hours = (timestamps // 3600) % self.config.hour_cycles
            days = ((timestamps // (24 * 3600)) + 4) % self.config.day_cycles  # +4 for Monday=0
            months = ((timestamps // (30 * 24 * 3600)) % self.config.month_cycles)
            
            # Cyclical encoding
            minute_cycle = self._create_cyclical_features(minutes, self.config.minute_cycles)
            hour_cycle = self._create_cyclical_features(hours, self.config.hour_cycles)
            day_cycle = self._create_cyclical_features(days, self.config.day_cycles)
            month_cycle = self._create_cyclical_features(months, self.config.month_cycles)
            
            # Project cyclical features
            minute_emb = self.minute_proj(minute_cycle)
            hour_emb = self.hour_proj(hour_cycle)
            day_emb = self.day_proj(day_cycle)
            month_emb = self.month_proj(month_cycle)
            
            features.extend([minute_emb, hour_emb, day_emb, month_emb])
        
        # Seasonal patterns
        if self.config.use_seasonal_patterns:
            seasonal_features = []
            for period in self.config.seasonal_periods:
                seasonal_phase = (timestamps / (3600 * period)) % 1
                seasonal_cycle = self._create_cyclical_features(seasonal_phase, 1.0)
                seasonal_proj = self.seasonal_projections[f'period_{period}']
                seasonal_emb = seasonal_proj(seasonal_cycle)
                seasonal_features.append(seasonal_emb)
            
            combined_seasonal = torch.cat(seasonal_features, dim=-1)
            features.append(combined_seasonal)
        
        # Crypto-specific cycles
        if self.config.use_crypto_cycles:
            # Bitcoin halving cycle (4 years)
            halving_period = self.config.halving_cycle_years * 365 * 24 * 3600
            halving_phase = (timestamps % halving_period) / halving_period
            halving_cycle = self._create_cyclical_features(halving_phase, 1.0)
            halving_emb = self.halving_proj(halving_cycle)
            features.append(halving_emb)
        
        if self.config.use_funding_rates:
            # 8-hour funding rate cycles
            funding_period = 8 * 3600  # 8 hours
            funding_phase = (timestamps % funding_period) / funding_period
            funding_cycle = self._create_cyclical_features(funding_phase, 1.0)
            funding_emb = self.funding_proj(funding_cycle)
            features.append(funding_emb)
        
        # Combine all features
        if features:
            combined_features = torch.cat(features, dim=-1)
            encoded_time = self.combination(combined_features)
        else:
            encoded_time = torch.zeros(batch_size, seq_len, self.config.d_model, device=timestamps.device)
        
        return self.dropout(encoded_time)
    
    def _create_cyclical_features(
        self, 
        values: torch.Tensor, 
        cycle_length: float
    ) -> torch.Tensor:
        """Create sin/cos cyclical features."""
        angle = 2 * math.pi * values / cycle_length
        sin_component = torch.sin(angle).unsqueeze(-1)
        cos_component = torch.cos(angle).unsqueeze(-1)
        return torch.cat([sin_component, cos_component], dim=-1)


class MarketSessionEncoding(nn.Module):
    """Market session aware encoding for crypto trading."""
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Market session embeddings
        # 0: Asian (00:00-08:00 UTC), 1: European (08:00-16:00 UTC), 2: US (16:00-24:00 UTC)
        self.session_embedding = nn.Embedding(3, config.d_model // 8)
        
        # Session overlap embeddings
        self.overlap_embedding = nn.Embedding(4, config.d_model // 16)  # 0-3 overlapping sessions
        
        # Weekend penalty
        self.weekend_gate = nn.Sequential(
            nn.Linear(config.d_model // 4, config.d_model // 8),
            nn.ReLU(),
            nn.Linear(config.d_model // 8, 1),
            nn.Sigmoid()
        )
        
        # Market activity prediction
        self.activity_predictor = nn.Sequential(
            nn.Linear(config.d_model // 4, config.d_model // 8),
            nn.ReLU(),
            nn.Linear(config.d_model // 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Apply market session encoding.
        
        Args:
            timestamps: Unix timestamps [batch_size, seq_len]
            
        Returns:
            session_encoding: [batch_size, seq_len, d_model//4]
        """
        # Extract time components
        hours = (timestamps // 3600) % 24
        dow = ((timestamps // (24 * 3600)) + 4) % 7  # Day of week
        
        # Determine market sessions
        sessions = self._get_market_sessions(hours)
        
        # Session overlaps (crypto is traded 24/7, but there are more active periods)
        overlaps = self._get_session_overlaps(hours)
        
        # Weekend detection
        is_weekend = (dow >= 5).float()  # Saturday and Sunday
        
        # Get embeddings
        session_emb = self.session_embedding(sessions)
        overlap_emb = self.overlap_embedding(overlaps)
        
        # Combine session features
        session_features = torch.cat([session_emb, overlap_emb], dim=-1)
        
        # Apply weekend penalty
        weekend_penalty = self.weekend_gate(session_features)
        weekend_factor = 1 - is_weekend.unsqueeze(-1) * (1 - self.config.weekend_penalty)
        session_features = session_features * weekend_penalty * weekend_factor
        
        # Predict market activity
        activity_score = self.activity_predictor(session_features)
        session_features = session_features * activity_score
        
        return session_features
    
    def _get_market_sessions(self, hours: torch.Tensor) -> torch.Tensor:
        """Get primary market session for each hour."""
        # Asian: 0-8, European: 8-16, US: 16-24
        sessions = (hours // 8) % 3
        return sessions.long()
    
    def _get_session_overlaps(self, hours: torch.Tensor) -> torch.Tensor:
        """Get number of overlapping active sessions."""
        # Simplified: assume some hours have multiple active markets
        overlaps = torch.zeros_like(hours)
        
        # Peak overlaps: European open (8-9), US open (16-17), Asia-Europe (7-8)
        peak_times = torch.tensor([7, 8, 9, 16, 17])
        for peak in peak_times:
            overlaps += (hours == peak).long()
        
        return overlaps.clamp(0, 3).long()


class SeasonalityEncoding(nn.Module):
    """Seasonality patterns for crypto markets."""
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Seasonal decomposition components
        self.trend_embedding = nn.Linear(1, config.d_model // 8)
        self.seasonal_embedding = nn.Linear(len(config.seasonal_periods) * 2, config.d_model // 4)
        
        # Learnable seasonal patterns
        self.seasonal_weights = nn.Parameter(torch.ones(len(config.seasonal_periods)))
        
        # Calendar effects
        self.calendar_effects = nn.ModuleDict({
            'month_effect': nn.Embedding(12, config.d_model // 16),
            'quarter_effect': nn.Embedding(4, config.d_model // 16),
            'year_effect': nn.Linear(1, config.d_model // 16)
        })
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Apply seasonality encoding."""
        batch_size, seq_len = timestamps.shape
        
        # Extract calendar components
        months = ((timestamps // (30 * 24 * 3600)) % 12).long()
        quarters = (months // 3).long()
        years = (timestamps // (365 * 24 * 3600)).float().unsqueeze(-1)
        
        # Seasonal components
        seasonal_features = []
        weights = F.softmax(self.seasonal_weights, dim=0)
        
        for i, period in enumerate(self.config.seasonal_periods):
            period_hours = period
            phase = (timestamps / (3600 * period_hours)) % 1
            sin_comp = torch.sin(2 * math.pi * phase)
            cos_comp = torch.cos(2 * math.pi * phase)
            
            # Weight seasonal components
            seasonal_features.extend([sin_comp * weights[i], cos_comp * weights[i]])
        
        seasonal_tensor = torch.stack(seasonal_features, dim=-1)
        seasonal_emb = self.seasonal_embedding(seasonal_tensor)
        
        # Calendar effects
        month_emb = self.calendar_effects['month_effect'](months)
        quarter_emb = self.calendar_effects['quarter_effect'](quarters)
        year_emb = self.calendar_effects['year_effect'](years)
        
        # Trend component (linear trend over time)
        normalized_time = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
        trend_emb = self.trend_embedding(normalized_time.unsqueeze(-1))
        
        # Combine seasonality features
        seasonality_features = torch.cat([
            seasonal_emb, month_emb, quarter_emb, year_emb, trend_emb
        ], dim=-1)
        
        return seasonality_features


class CryptoTemporalEncoding(nn.Module):
    """
    Comprehensive temporal encoding for crypto trading.
    
    Features:
    - Cyclical time patterns
    - Market session awareness  
    - Seasonal effects
    - Crypto-specific cycles
    - Market microstructure timing
    """
    
    def __init__(self, config: TemporalEncodingConfig):
        super().__init__()
        self.config = config
        
        # Component encodings
        self.cyclical_encoding = CyclicalTimeEncoding(config)
        
        if config.use_market_sessions:
            self.session_encoding = MarketSessionEncoding(config)
        
        if config.use_seasonal_patterns:
            self.seasonality_encoding = SeasonalityEncoding(config)
        
        # Volume and volatility cycle encodings
        if config.use_volume_cycles:
            self.volume_cycle_proj = nn.Linear(2, config.d_model // 16)
            
        if config.use_volatility_cycles:
            self.volatility_cycle_proj = nn.Linear(2, config.d_model // 16)
        
        # Market open/close effects
        if config.use_market_open_close:
            self.open_close_embedding = nn.Embedding(4, config.d_model // 16)  # Pre-open, Open, Close, After-hours
        
        # Final combination layer
        self.final_projection = nn.Linear(config.d_model * 2, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        timestamps: torch.Tensor,
        volume_data: Optional[torch.Tensor] = None,
        volatility_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply comprehensive temporal encoding.
        
        Args:
            timestamps: Unix timestamps [batch_size, seq_len]
            volume_data: Volume patterns [batch_size, seq_len] (optional)
            volatility_data: Volatility patterns [batch_size, seq_len] (optional)
            
        Returns:
            temporal_encoding: [batch_size, seq_len, d_model]
        """
        encoding_components = []
        
        # Cyclical encoding (base component)
        cyclical_enc = self.cyclical_encoding(timestamps)
        encoding_components.append(cyclical_enc)
        
        # Market session encoding
        if hasattr(self, 'session_encoding'):
            session_enc = self.session_encoding(timestamps)
            # Pad to match d_model
            if session_enc.shape[-1] < cyclical_enc.shape[-1]:
                padding = torch.zeros(
                    *session_enc.shape[:-1], 
                    cyclical_enc.shape[-1] - session_enc.shape[-1],
                    device=session_enc.device
                )
                session_enc = torch.cat([session_enc, padding], dim=-1)
            encoding_components.append(session_enc)
        
        # Seasonality encoding
        if hasattr(self, 'seasonality_encoding'):
            seasonal_enc = self.seasonality_encoding(timestamps)
            # Pad to match d_model
            if seasonal_enc.shape[-1] < cyclical_enc.shape[-1]:
                padding = torch.zeros(
                    *seasonal_enc.shape[:-1], 
                    cyclical_enc.shape[-1] - seasonal_enc.shape[-1],
                    device=seasonal_enc.device
                )
                seasonal_enc = torch.cat([seasonal_enc, padding], dim=-1)
            encoding_components.append(seasonal_enc)
        
        # Volume cycles
        if self.config.use_volume_cycles and volume_data is not None:
            # Create volume-based temporal cycles
            volume_normalized = (volume_data - volume_data.mean()) / (volume_data.std() + 1e-8)
            volume_phase = volume_normalized % 1
            volume_cycle = self._create_cycle_features(volume_phase)
            volume_enc = self.volume_cycle_proj(volume_cycle)
            
            # Pad to match d_model
            volume_enc_padded = torch.zeros(*volume_enc.shape[:-1], cyclical_enc.shape[-1], device=volume_enc.device)
            volume_enc_padded[..., :volume_enc.shape[-1]] = volume_enc
            encoding_components.append(volume_enc_padded)
        
        # Volatility cycles
        if self.config.use_volatility_cycles and volatility_data is not None:
            volatility_normalized = (volatility_data - volatility_data.mean()) / (volatility_data.std() + 1e-8)
            volatility_phase = volatility_normalized % 1
            volatility_cycle = self._create_cycle_features(volatility_phase)
            volatility_enc = self.volatility_cycle_proj(volatility_cycle)
            
            # Pad to match d_model
            volatility_enc_padded = torch.zeros(*volatility_enc.shape[:-1], cyclical_enc.shape[-1], device=volatility_enc.device)
            volatility_enc_padded[..., :volatility_enc.shape[-1]] = volatility_enc
            encoding_components.append(volatility_enc_padded)
        
        # Market open/close timing
        if self.config.use_market_open_close:
            hours = (timestamps // 3600) % 24
            # Simplified market timing: 0=pre-open (6-9), 1=open (9-16), 2=close (16-17), 3=after-hours (17-6)
            market_timing = torch.zeros_like(hours).long()
            market_timing[(hours >= 6) & (hours < 9)] = 0   # Pre-open
            market_timing[(hours >= 9) & (hours < 16)] = 1  # Open
            market_timing[(hours >= 16) & (hours < 17)] = 2 # Close
            market_timing[(hours >= 17) | (hours < 6)] = 3  # After-hours
            
            timing_enc = self.open_close_embedding(market_timing)
            # Pad to match d_model
            timing_enc_padded = torch.zeros(*timing_enc.shape[:-1], cyclical_enc.shape[-1], device=timing_enc.device)
            timing_enc_padded[..., :timing_enc.shape[-1]] = timing_enc
            encoding_components.append(timing_enc_padded)
        
        # Combine all encoding components
        if len(encoding_components) > 1:
            # Stack and average components
            stacked_encodings = torch.stack(encoding_components, dim=0)
            combined_encoding = stacked_encodings.mean(dim=0)
        else:
            combined_encoding = encoding_components[0]
        
        # Final processing
        # Concatenate with original cyclical encoding for richer representation
        enhanced_encoding = torch.cat([cyclical_enc, combined_encoding], dim=-1)
        final_encoding = self.final_projection(enhanced_encoding)
        final_encoding = self.layer_norm(final_encoding)
        
        return self.dropout(final_encoding)
    
    def _create_cycle_features(self, phase: torch.Tensor) -> torch.Tensor:
        """Create sin/cos features from phase values."""
        sin_comp = torch.sin(2 * math.pi * phase).unsqueeze(-1)
        cos_comp = torch.cos(2 * math.pi * phase).unsqueeze(-1)
        return torch.cat([sin_comp, cos_comp], dim=-1)


def create_temporal_encoding(config: TemporalEncodingConfig) -> nn.Module:
    """
    Factory function for creation temporal encoding.
    
    Args:
        config: Temporal encoding configuration
        
    Returns:
        Temporal encoding module
    """
    return CryptoTemporalEncoding(config)


if __name__ == "__main__":
    # Test temporal encoding
    config = TemporalEncodingConfig(
        d_model=512,
        max_seq_len=1000,
        use_cyclical_encoding=True,
        use_market_sessions=True,
        use_seasonal_patterns=True,
        use_crypto_cycles=True,
        use_funding_rates=True,
        use_volume_cycles=True,
        use_volatility_cycles=True
    )
    
    # Create test data
    batch_size, seq_len = 4, 256
    
    # Generate realistic timestamps (1 week of 1-minute data)
    start_timestamp = 1640995200  # 2022-01-01
    timestamps = torch.arange(start_timestamp, start_timestamp + seq_len * 60, 60)
    timestamps = timestamps.unsqueeze(0).repeat(batch_size, 1)
    
    # Mock volume and volatility data
    volume_data = torch.rand(batch_size, seq_len) * 1000 + 100
    volatility_data = torch.rand(batch_size, seq_len) * 0.1 + 0.01
    
    # Test complete temporal encoding
    temporal_enc = CryptoTemporalEncoding(config)
    
    temporal_output = temporal_enc(
        timestamps=timestamps,
        volume_data=volume_data,
        volatility_data=volatility_data
    )
    
    print(f"Temporal encoding output: {temporal_output.shape}")
    
    # Test individual components
    print("\nTesting individual components:")
    
    # Cyclical encoding
    cyclical_enc = CyclicalTimeEncoding(config)
    cyclical_output = cyclical_enc(timestamps)
    print(f"Cyclical encoding: {cyclical_output.shape}")
    
    # Market session encoding
    session_enc = MarketSessionEncoding(config)
    session_output = session_enc(timestamps)
    print(f"Market session encoding: {session_output.shape}")
    
    # Seasonality encoding
    seasonality_enc = SeasonalityEncoding(config)
    seasonality_output = seasonality_enc(timestamps)
    print(f"Seasonality encoding: {seasonality_output.shape}")
    
    # Model parameters
    print(f"\nModel parameters:")
    print(f"Complete temporal encoding: {sum(p.numel() for p in temporal_enc.parameters())}")
    print(f"Cyclical only: {sum(p.numel() for p in cyclical_enc.parameters())}")
    print(f"Session only: {sum(p.numel() for p in session_enc.parameters())}")
    print(f"Seasonality only: {sum(p.numel() for p in seasonality_enc.parameters())}")
    
    # Test with various time periods
    print(f"\nTesting various temporal periods:")
    
    # Intraday (5-minute data)
    intraday_timestamps = torch.arange(start_timestamp, start_timestamp + 288 * 300, 300)  # 1 day, 5-min intervals
    intraday_timestamps = intraday_timestamps.unsqueeze(0).repeat(2, 1)
    intraday_output = temporal_enc(timestamps=intraday_timestamps)
    print(f"Intraday (5-min): {intraday_output.shape}")
    
    # Daily data
    daily_timestamps = torch.arange(start_timestamp, start_timestamp + 365 * 24 * 3600, 24 * 3600)  # 1 year, daily
    daily_timestamps = daily_timestamps.unsqueeze(0).repeat(2, 1)
    daily_output = temporal_enc(timestamps=daily_timestamps)
    print(f"Daily: {daily_output.shape}")
    
    # Verify temporal consistency
    print(f"\nTemporal consistency check:")
    
    # Same timestamp should produce same encoding
    single_timestamp = torch.tensor([[start_timestamp]])
    encoding1 = temporal_enc(timestamps=single_timestamp)
    encoding2 = temporal_enc(timestamps=single_timestamp)
    
    consistency_error = torch.abs(encoding1 - encoding2).max().item()
    print(f"Consistency error: {consistency_error:.8f} (should be 0.0)")
    
    # Different timestamps should produce different encodings
    timestamp1 = torch.tensor([[start_timestamp]])
    timestamp2 = torch.tensor([[start_timestamp + 3600]])  # +1 hour
    
    enc1 = temporal_enc(timestamps=timestamp1)
    enc2 = temporal_enc(timestamps=timestamp2)
    
    difference = torch.abs(enc1 - enc2).mean().item()
    print(f"Temporal difference (1 hour apart): {difference:.6f} (should be > 0)")
    
    print(f"\n✅ Temporal encoding implementation complete!")