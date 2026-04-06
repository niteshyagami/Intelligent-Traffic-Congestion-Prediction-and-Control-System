"""
Lane-Aware Adaptive Signal Controller.

Allocates green time proportionally to vehicle count per lane.
Busier lane → longer green. Less busy → shorter green.

Novel contribution:
- Per-LANE control, not per-intersection (fills Gap 2 from research strategy)
- Dynamic cycle adaptation based on total traffic volume
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import math


@dataclass
class LaneSignalState:
    """Signal state for a single lane :-"""
    lane_id: str
    green_duration: int  # seconds
    red_duration: int
    vehicle_count: int = 0
    phase: str = "RED"  # RED or GREEN


@dataclass
class IntersectionState:
    """Complete signal state for a 4-lane intersection."""
    intersection_id: str
    intersection_name: str = ""
    total_cycle: int = 120
    current_green_lane: str = "Lane_N"
    lanes: Dict[str, LaneSignalState] = field(default_factory=dict)
    last_updated: str = ""
    total_vehicles: int = 0
    congestion_level: str = "Low"


class AdaptiveSignalController:
    """
    Proportional green-time allocation based on per-lane vehicle counts.

    Algorithm:
        1. Get vehicle count per lane from detector
        2. Calculate total vehicles across all lanes
        3. Allocate green time proportionally:
           green_time[lane] = (count[lane] / total_count) * available_green_time
        4. Enforce min/max green constraints (15s-45s)
        5. If emergency (any lane >30 vehicles), extend its green to max

    Total cycle time adapts to traffic volume:
        - Light traffic (< 30 total): 80s cycle
        - Normal traffic (30-80): 120s cycle
        - Heavy traffic (> 80): 160s cycle
    """

    MIN_GREEN = 15   # seconds
    MAX_GREEN = 60   # seconds
    YELLOW_TIME = 3  # seconds per transition
    EMERGENCY_THRESHOLD = 50  # vehicles per lane

    DEFAULT_LANES = ["Lane_N", "Lane_S", "Lane_E", "Lane_W"]

    def __init__(self):
        self.intersections: Dict[str, IntersectionState] = {}

    def _get_cycle_time(self, total_vehicles: int) -> int:
        """Dynamic cycle time based on total traffic volume."""
        if total_vehicles < 30:
            return 80
        elif total_vehicles < 80:
            return 120
        return 160

    def _classify_congestion(self, total_vehicles: int) -> str:
        """Classify intersection congestion level."""
        if total_vehicles > 80:
            return "High"
        elif total_vehicles > 40:
            return "Medium"
        return "Low"

    def update_signals(
        self,
        intersection_id: str,
        lane_counts: Dict[str, int],
        intersection_name: str = "",
    ) -> IntersectionState:
        """
        Recalculate signal timings based on lane vehicle counts.

        Args:
            intersection_id: Unique ID for the intersection
            lane_counts: Dict of lane_id → vehicle_count (e.g., {"Lane_N": 12, ...})
            intersection_name: Human-readable name

        Returns:
            Updated IntersectionState with per-lane green durations
        """
        total = sum(lane_counts.values())
        
        # Perfect Dynamic Calculation
        lane_signals = {}
        green_times = {}
        active_lanes = 0

        for lane_id, count in lane_counts.items():
            if count == 0:
                green_times[lane_id] = 0
            else:
                # 2.0 seconds clearance time per vehicle to distribute scale better
                raw_green = int(round(count * 2.0))
                green = max(self.MIN_GREEN, min(self.MAX_GREEN, raw_green))
                
                # Emergency override
                if count >= self.EMERGENCY_THRESHOLD:
                    green = self.MAX_GREEN
                    
                green_times[lane_id] = green
                active_lanes += 1

        # Dynamic Cycle: sum of perfectly allocated greens + necessary yellows
        if active_lanes == 0:
            cycle = 60
            total_yellow = 0
        else:
            total_yellow = self.YELLOW_TIME * active_lanes
            cycle = sum(green_times.values()) + total_yellow

        # Determine which lane gets green first (busiest goes first)
        busiest = max(lane_counts, key=lambda k: lane_counts[k])

        # Build lane signal states
        for lane_id, count in lane_counts.items():
            green = green_times.get(lane_id, self.MIN_GREEN)
            if green == 0:
                red = cycle
            else:
                red = cycle - green - self.YELLOW_TIME
                
            lane_signals[lane_id] = LaneSignalState(
                lane_id=lane_id,
                green_duration=green,
                red_duration=max(0, red),
                vehicle_count=count,
                phase="GREEN" if lane_id == busiest else "RED",
            )

        state = IntersectionState(
            intersection_id=intersection_id,
            intersection_name=intersection_name,
            total_cycle=cycle,
            current_green_lane=busiest,
            lanes=lane_signals,
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_vehicles=total,
            congestion_level=self._classify_congestion(total),
        )

        self.intersections[intersection_id] = state
        return state

    def get_state(self, intersection_id: str) -> Optional[IntersectionState]:
        return self.intersections.get(intersection_id)

    def get_all_states(self) -> List[IntersectionState]:
        return list(self.intersections.values())

    def get_signal_summary(self, intersection_id: str) -> Dict:
        """Return a JSON-friendly summary of signal states for one intersection."""
        state = self.intersections.get(intersection_id)
        if not state:
            return {"error": "Intersection not found"}
        return {
            "intersection_id": state.intersection_id,
            "intersection_name": state.intersection_name,
            "total_cycle": state.total_cycle,
            "congestion_level": state.congestion_level,
            "total_vehicles": state.total_vehicles,
            "current_green_lane": state.current_green_lane,
            "last_updated": state.last_updated,
            "lanes": {
                lid: {
                    "green_duration": ls.green_duration,
                    "red_duration": ls.red_duration,
                    "vehicle_count": ls.vehicle_count,
                    "phase": ls.phase,
                }
                for lid, ls in state.lanes.items()
            },
        }
