"""
Level Manager for Support/Resistance System
ML-Framework-1332 - Manage, update, and track support/resistance levels

 2025: Adaptive level management with time decay and strength calculation
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


@dataclass
class LevelStrength:
    """Strength metrics for a support/resistance level"""
    base_strength: float  # Initial strength from detection
    touch_strength: float  # Strength from price touches
    volume_strength: float  # Strength from volume
    time_decay: float  # Decay factor based on age
    confluence_boost: float  # Boost from being in confluence zone

    @property
    def total_strength(self) -> float:
        """Calculate total strength combining all factors"""
        return min(1.0, (
            self.base_strength * 0.3 +
            self.touch_strength * 0.25 +
            self.volume_strength * 0.2 +
            self.time_decay * 0.15 +
            self.confluence_boost * 0.1
        ))


@dataclass
class ConfluenceZone:
    """Zone where multiple levels cluster together"""
    min_price: float
    max_price: float
    center_price: float
    levels_count: int
    combined_strength: float
    level_ids: Set[str] = field(default_factory=set)

    @property
    def zone_width(self) -> float:
        """Width of the confluence zone"""
        return self.max_price - self.min_price

    @property
    def zone_width_percent(self) -> float:
        """Width as percentage of center price"""
        return (self.zone_width / self.center_price) * 100 if self.center_price > 0 else 0

    def contains_price(self, price: float) -> bool:
        """Check if price is within the zone"""
        return self.min_price <= price <= self.max_price


@dataclass
class LevelUpdate:
    """Update event for a level"""
    level_id: str
    update_type: str  # 'touch', 'break', 'validate', 'expire'
    timestamp: datetime
    price_at_update: float
    metadata: Dict = field(default_factory=dict)


class LevelManager:
    """
    Manages support/resistance levels with enterprise patterns.

    Handles level lifecycle, strength calculation, confluence detection,
    and real-time updates.
    """

    def __init__(
        self,
        confluence_threshold: float = 0.02,  # 2% price range
        max_levels: int = 100,
        time_decay_days: int = 30,
        min_touches_for_validation: int = 2
    ):
        self.confluence_threshold = confluence_threshold
        self.max_levels = max_levels
        self.time_decay_days = time_decay_days
        self.min_touches_for_validation = min_touches_for_validation

        # Level storage
        self.levels: Dict[str, Any] = {}  # level_id -> SupportResistanceLevel
        self.level_strengths: Dict[str, LevelStrength] = {}
        self.confluence_zones: List[ConfluenceZone] = []

        # Update tracking
        self.update_history: List[LevelUpdate] = []
        self.level_touches: Dict[str, List[datetime]] = defaultdict(list)
        self.level_breaks: Dict[str, List[datetime]] = defaultdict(list)

        # Performance metrics
        self.validated_levels: Set[str] = set()
        self.expired_levels: Set[str] = set()

        self.logger = logging.getLogger("LevelManager")

    def update_levels(self, new_levels: List):
        """Update managed levels with new detections"""
        for level in new_levels:
            level_id = self._generate_level_id(level)

            if level_id in self.levels:
                # Update existing level
                self._update_existing_level(level_id, level)
            else:
                # Add new level
                self._add_new_level(level_id, level)

        # Update confluence zones
        self._update_confluence_zones()

        # Apply time decay
        self._apply_time_decay()

        # Clean up old levels if needed
        if len(self.levels) > self.max_levels:
            self._cleanup_old_levels()

    def _generate_level_id(self, level) -> str:
        """Generate unique ID for a level"""
        return f"{level.level_type.value}_{level.price:.2f}_{level.method.value}"

    def _add_new_level(self, level_id: str, level):
        """Add new level to management"""
        self.levels[level_id] = level

        # Initialize strength
        self.level_strengths[level_id] = LevelStrength(
            base_strength=level.strength,
            touch_strength=0.0,
            volume_strength=0.0,
            time_decay=1.0,
            confluence_boost=0.0
        )

        # Record update
        self.update_history.append(LevelUpdate(
            level_id=level_id,
            update_type='create',
            timestamp=datetime.now(),
            price_at_update=level.price
        ))

        self.logger.debug(f"Added new level: {level_id}")

    def _update_existing_level(self, level_id: str, new_level):
        """Update existing level with new detection"""
        existing_level = self.levels[level_id]

        # Update last validated time
        existing_level.last_validated = datetime.now()

        # Increase base strength slightly
        if level_id in self.level_strengths:
            self.level_strengths[level_id].base_strength = min(
                1.0,
                self.level_strengths[level_id].base_strength * 1.1
            )

        # Record validation
        self.update_history.append(LevelUpdate(
            level_id=level_id,
            update_type='validate',
            timestamp=datetime.now(),
            price_at_update=new_level.price
        ))

    def update_touches(self, price_data: pd.DataFrame, tolerance_percent: float = 0.5):
        """Update touch counts based on price data"""
        for level_id, level in self.levels.items():
            touches_found = 0
            level_price = level.price
            tolerance = level_price * (tolerance_percent / 100)

            for _, row in price_data.iterrows():
                # Check if high or low touched the level
                if (abs(row['high'] - level_price) <= tolerance or
                    abs(row['low'] - level_price) <= tolerance):
                    touches_found += 1
                    self.level_touches[level_id].append(datetime.now())

            if touches_found > 0:
                # Update touch count
                level.touch_count += touches_found

                # Update touch strength
                if level_id in self.level_strengths:
                    self.level_strengths[level_id].touch_strength = min(
                        1.0,
                        level.touch_count / 10  # Max strength at 10 touches
                    )

                # Check for validation
                if level.touch_count >= self.min_touches_for_validation:
                    self.validated_levels.add(level_id)

                # Record update
                self.update_history.append(LevelUpdate(
                    level_id=level_id,
                    update_type='touch',
                    timestamp=datetime.now(),
                    price_at_update=price_data['close'].iloc[-1],
                    metadata={'touches_found': touches_found}
                ))

    def check_level_breaks(
        self,
        current_price: float,
        high_price: float,
        low_price: float,
        break_threshold_percent: float = 1.0
    ) -> List[str]:
        """Check if any levels have been broken"""
        broken_levels = []

        for level_id, level in self.levels.items():
            level_price = level.price
            threshold = level_price * (break_threshold_percent / 100)

            if level.level_type.value == 'support':
                # Support broken if price goes below
                if low_price < level_price - threshold:
                    broken_levels.append(level_id)
                    level.break_count += 1
                    self.level_breaks[level_id].append(datetime.now())

                    # Reduce strength on break
                    if level_id in self.level_strengths:
                        self.level_strengths[level_id].base_strength *= 0.8

            elif level.level_type.value == 'resistance':
                # Resistance broken if price goes above
                if high_price > level_price + threshold:
                    broken_levels.append(level_id)
                    level.break_count += 1
                    self.level_breaks[level_id].append(datetime.now())

                    # Reduce strength on break
                    if level_id in self.level_strengths:
                        self.level_strengths[level_id].base_strength *= 0.8

            # Record break
            if level_id in broken_levels:
                self.update_history.append(LevelUpdate(
                    level_id=level_id,
                    update_type='break',
                    timestamp=datetime.now(),
                    price_at_update=current_price
                ))

        return broken_levels

    def _update_confluence_zones(self):
        """Detect and update confluence zones"""
        self.confluence_zones.clear()

        # Sort levels by price
        sorted_levels = sorted(
            self.levels.items(),
            key=lambda x: x[1].price
        )

        if not sorted_levels:
            return

        # Find clusters
        i = 0
        while i < len(sorted_levels):
            zone_levels = [(sorted_levels[i][0], sorted_levels[i][1])]
            zone_min = sorted_levels[i][1].price
            zone_max = sorted_levels[i][1].price

            # Find all levels within threshold
            j = i + 1
            while j < len(sorted_levels):
                price_diff = abs(sorted_levels[j][1].price - zone_min) / zone_min

                if price_diff <= self.confluence_threshold:
                    zone_levels.append((sorted_levels[j][0], sorted_levels[j][1]))
                    zone_max = max(zone_max, sorted_levels[j][1].price)
                    j += 1
                else:
                    break

            # Create confluence zone if multiple levels
            if len(zone_levels) >= 2:
                # Calculate combined strength
                combined_strength = np.mean([
                    self.level_strengths[lid].total_strength
                    for lid, _ in zone_levels
                    if lid in self.level_strengths
                ])

                zone = ConfluenceZone(
                    min_price=zone_min,
                    max_price=zone_max,
                    center_price=(zone_min + zone_max) / 2,
                    levels_count=len(zone_levels),
                    combined_strength=min(1.0, combined_strength * 1.2),
                    level_ids={lid for lid, _ in zone_levels}
                )

                self.confluence_zones.append(zone)

                # Boost strength for levels in confluence
                for lid, _ in zone_levels:
                    if lid in self.level_strengths:
                        self.level_strengths[lid].confluence_boost = 0.2

            i = j if j > i + 1 else i + 1

    def _apply_time_decay(self):
        """Apply time decay to level strengths"""
        current_time = datetime.now()

        for level_id, level in self.levels.items():
            age_days = (current_time - level.first_detected).days

            if age_days > 0:
                # Calculate decay factor
                decay_factor = max(0.3, 1.0 - (age_days / self.time_decay_days))

                if level_id in self.level_strengths:
                    self.level_strengths[level_id].time_decay = decay_factor

                # Mark as expired if too old
                if age_days > self.time_decay_days * 2:
                    self.expired_levels.add(level_id)

    def _cleanup_old_levels(self):
        """Remove weakest levels if exceeding max_levels"""
        # Calculate total strength for each level
        level_scores = []

        for level_id, level in self.levels.items():
            if level_id in self.level_strengths:
                score = self.level_strengths[level_id].total_strength
            else:
                score = level.strength

            level_scores.append((level_id, score))

        # Sort by score
        level_scores.sort(key=lambda x: x[1], reverse=True)

        # Keep only top levels
        levels_to_keep = set(lid for lid, _ in level_scores[:self.max_levels])

        # Remove weak levels
        levels_to_remove = []
        for level_id in self.levels:
            if level_id not in levels_to_keep:
                levels_to_remove.append(level_id)

        for level_id in levels_to_remove:
            del self.levels[level_id]
            if level_id in self.level_strengths:
                del self.level_strengths[level_id]

            # Record removal
            self.update_history.append(LevelUpdate(
                level_id=level_id,
                update_type='expire',
                timestamp=datetime.now(),
                price_at_update=0
            ))

        if levels_to_remove:
            self.logger.info(f"Removed {len(levels_to_remove)} weak levels")

    def get_all_levels(self) -> List:
        """Get all managed levels with updated strengths"""
        levels_list = []

        for level_id, level in self.levels.items():
            # Update reliability score if strength is tracked
            if level_id in self.level_strengths:
                level.strength = self.level_strengths[level_id].total_strength

            levels_list.append(level)

        return levels_list

    def get_validated_levels(self) -> List:
        """Get only validated levels"""
        return [
            level for lid, level in self.levels.items()
            if lid in self.validated_levels
        ]

    def get_confluence_zones(self) -> List[ConfluenceZone]:
        """Get current confluence zones"""
        return self.confluence_zones

    def get_level_by_price(
        self,
        price: float,
        tolerance_percent: float = 0.5
    ) -> Optional:
        """Find level closest to given price"""
        tolerance = price * (tolerance_percent / 100)
        closest_level = None
        min_distance = float('inf')

        for level in self.levels.values():
            distance = abs(level.price - price)
            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                closest_level = level

        return closest_level

    def get_zone_by_price(self, price: float) -> Optional[ConfluenceZone]:
        """Find confluence zone containing price"""
        for zone in self.confluence_zones:
            if zone.contains_price(price):
                return zone
        return None

    def get_statistics(self) -> Dict:
        """Get management statistics"""
        return {
            'total_levels': len(self.levels),
            'validated_levels': len(self.validated_levels),
            'expired_levels': len(self.expired_levels),
            'confluence_zones': len(self.confluence_zones),
            'total_touches': sum(level.touch_count for level in self.levels.values()),
            'total_breaks': sum(level.break_count for level in self.levels.values()),
            'average_strength': np.mean([
                self.level_strengths[lid].total_strength
                for lid in self.level_strengths
            ]) if self.level_strengths else 0,
            'strongest_level': max(
                self.levels.values(),
                key=lambda l: l.strength
            ).price if self.levels else None
        }

    def reset(self):
        """Reset all managed levels"""
        self.levels.clear()
        self.level_strengths.clear()
        self.confluence_zones.clear()
        self.update_history.clear()
        self.level_touches.clear()
        self.level_breaks.clear()
        self.validated_levels.clear()
        self.expired_levels.clear()

        self.logger.info("Level manager reset")