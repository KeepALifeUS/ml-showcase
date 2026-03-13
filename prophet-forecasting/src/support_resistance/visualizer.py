"""
Level Visualizer for Support/Resistance System
ML-Framework-1332 - Visualization components for support/resistance levels

 2025: Interactive charts with multiple visualization styles
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class ChartStyle(Enum):
    """Available chart styles"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    HEATMAP = "heatmap"
    VOLUME_PROFILE = "volume_profile"


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    chart_style: ChartStyle = ChartStyle.CANDLESTICK
    show_volume: bool = True
    show_confluence_zones: bool = True
    level_opacity: float = 0.7
    zone_opacity: float = 0.3
    color_support: str = "#00FF00"
    color_resistance: str = "#FF0000"
    color_pivot: str = "#0000FF"
    color_confluence: str = "#FFD700"


class LevelVisualizer:
    """Visualize support/resistance levels on charts"""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger("LevelVisualizer")

    def create_chart_data(
        self,
        price_data: pd.DataFrame,
        levels: List,
        confluence_zones: Optional[List] = None
    ) -> Dict[str, Any]:
        """Create chart data structure for visualization"""

        chart_data = {
            'price_data': self._prepare_price_data(price_data),
            'levels': self._prepare_levels(levels),
            'zones': self._prepare_zones(confluence_zones) if confluence_zones else [],
            'config': self._config_to_dict()
        }

        return chart_data

    def _prepare_price_data(self, data: pd.DataFrame) -> List[Dict]:
        """Prepare price data for charting"""
        prepared = []

        for _, row in data.iterrows():
            prepared.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return prepared

    def _prepare_levels(self, levels: List) -> List[Dict]:
        """Prepare levels for visualization"""
        prepared = []

        for level in levels:
            color = self._get_level_color(level.level_type.value)

            prepared.append({
                'price': float(level.price),
                'type': level.level_type.value,
                'strength': float(level.strength),
                'color': color,
                'opacity': self.config.level_opacity * level.strength,
                'label': f"{level.level_type.value.upper()} ({level.price:.2f})",
                'metadata': {
                    'method': level.method.value,
                    'touches': level.touch_count,
                    'reliability': level.reliability_score
                }
            })

        return prepared

    def _prepare_zones(self, zones: List) -> List[Dict]:
        """Prepare confluence zones for visualization"""
        prepared = []

        for zone in zones:
            prepared.append({
                'min_price': float(zone.min_price),
                'max_price': float(zone.max_price),
                'center_price': float(zone.center_price),
                'strength': float(zone.combined_strength),
                'color': self.config.color_confluence,
                'opacity': self.config.zone_opacity * zone.combined_strength,
                'label': f"Confluence Zone ({zone.levels_count} levels)"
            })

        return prepared

    def _get_level_color(self, level_type: str) -> str:
        """Get color for level type"""
        if level_type == 'support':
            return self.config.color_support
        elif level_type == 'resistance':
            return self.config.color_resistance
        elif level_type == 'pivot':
            return self.config.color_pivot
        else:
            return self.config.color_confluence

    def _config_to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            'chart_style': self.config.chart_style.value,
            'show_volume': self.config.show_volume,
            'show_confluence_zones': self.config.show_confluence_zones,
            'colors': {
                'support': self.config.color_support,
                'resistance': self.config.color_resistance,
                'pivot': self.config.color_pivot,
                'confluence': self.config.color_confluence
            }
        }