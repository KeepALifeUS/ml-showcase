"""
Weighted Calculator for Fear & Greed Index

Main calculator for combining index components
with configurable weights.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import structlog
from dataclasses import dataclass

from ..components import (
    VolatilityComponent, MomentumComponent, VolumeComponent,
    SocialSentimentComponent, DominanceComponent,
    SearchTrendsComponent, SurveysComponent
)
from ..utils.config import FearGreedConfig
from ..utils.validators import DataValidator
from ..utils.metrics import ComponentMetrics

logger = structlog.get_logger(__name__)


@dataclass
class FearGreedResult:
    """Fear & Greed Index calculation result"""
    timestamp: datetime
    symbol: str
    final_score: float              # 0-100
    interpretation: str             # "Extreme Fear", "Fear", etc.
    confidence: float               # 0-1
    components: Dict[str, float]    # Scores from each component
    weights_used: Dict[str, float]  # Weights used
    metadata: Dict[str, Any]        # Additional information


class WeightedCalculator:
    """
    Weighted calculator for Fear & Greed Index

    Combines all components with configurable weights
    and produces the final index with interpretation.
    """

    def __init__(self, config: FearGreedConfig):
        self.config = config
        self.validator = DataValidator()
        self.metrics = ComponentMetrics("weighted_calculator")

        # Initialize components
        self.volatility = VolatilityComponent(config)
        self.momentum = MomentumComponent(config)
        self.volume = VolumeComponent(config)
        self.social_sentiment = SocialSentimentComponent(config)
        self.dominance = DominanceComponent(config)
        self.search_trends = SearchTrendsComponent(config)
        self.surveys = SurveysComponent(config)

        # Index interpretations
        self.interpretations = {
            (0, 20): "Extreme Fear",
            (20, 40): "Fear",
            (40, 60): "Neutral",
            (60, 80): "Greed",
            (80, 100): "Extreme Greed"
        }

        logger.info("WeightedCalculator initialized", weights=config.component_weights)

    def get_interpretation(self, score: float) -> str:
        """Get text interpretation of the index"""
        for (min_val, max_val), interpretation in self.interpretations.items():
            if min_val <= score < max_val:
                return interpretation
        return "Extreme Greed"  # For score = 100

    async def calculate_index(
        self,
        symbol: str,
        price_data: Optional[Any] = None,
        **kwargs
    ) -> FearGreedResult:
        """
        Calculate Fear & Greed Index

        Args:
            symbol: Cryptocurrency symbol
            price_data: OHLCV data (optional)
            **kwargs: Additional parameters

        Returns:
            FearGreedResult with the final index
        """
        try:
            components = {}
            component_confidences = {}

            # Collect data from all components in parallel
            tasks = []

            if price_data is not None:
                tasks.extend([
                    self.volatility.get_fear_greed_component(symbol, price_data),
                    self.momentum.get_fear_greed_component(symbol, price_data),
                    self.volume.get_fear_greed_component(symbol, price_data)
                ])

            tasks.extend([
                self.social_sentiment.get_fear_greed_component(symbol),
                self.dominance.get_fear_greed_component(),
                self.search_trends.get_fear_greed_component(symbol),
                self.surveys.get_fear_greed_component(symbol)
            ])

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            component_names = ['volatility', 'momentum', 'volume', 'social_sentiment',
                             'dominance', 'search_trends', 'surveys']

            if price_data is None:
                component_names = component_names[3:]  # Skip price-based components

            for name, result in zip(component_names, results):
                if isinstance(result, Exception):
                    logger.error(f"Component {name} failed", error=str(result))
                    components[name] = 50.0  # Neutral value
                    component_confidences[name] = 0.0
                else:
                    components[name] = result.fear_greed_score
                    component_confidences[name] = result.confidence

            # Calculate weighted index
            weights = self.config.component_weights
            total_score = 0.0
            total_weight = 0.0
            total_confidence = 0.0

            for component_name, score in components.items():
                weight = weights.get(component_name, 0.0)
                confidence = component_confidences.get(component_name, 0.0)

                # Weight by confidence
                effective_weight = weight * confidence
                total_score += score * effective_weight
                total_weight += effective_weight
                total_confidence += confidence

            # Final index
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 50.0  # Neutral value if no data available

            # Average confidence
            avg_confidence = total_confidence / len(components) if components else 0.0

            # Interpretation
            interpretation = self.get_interpretation(final_score)

            result = FearGreedResult(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                final_score=final_score,
                interpretation=interpretation,
                confidence=avg_confidence,
                components=components,
                weights_used=weights,
                metadata={
                    'total_components': len(components),
                    'effective_weight': total_weight,
                    'calculation_method': 'weighted_average'
                }
            )

            logger.info("Fear & Greed Index calculated",
                       symbol=symbol,
                       score=final_score,
                       interpretation=interpretation,
                       confidence=avg_confidence)

            self.metrics.record_calculation("index_calculation", 1)

            return result

        except Exception as e:
            logger.error("Error calculating Fear & Greed Index",
                        symbol=symbol, error=str(e))
            self.metrics.record_error("index_calculation_error")
            raise

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        return self.metrics.get_metrics()
