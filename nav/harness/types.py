"""
Data structures for multi-agent navigation system.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class GlobalPlannerInput:
    """Input for Global Planner Agent."""

    minimap_annotated: (
        np.ndarray
    )  # Shape (H, W, 3), with agent (red △) and target (green ●)
    history_trajectory: Optional[np.ndarray] = (
        None  # Shape (H, W, 3), optional trajectory overlay
    )

    # Metadata
    current_position: Tuple[int, int] = (0, 0)  # (x, y) on minimap
    target_position: Tuple[int, int] = (0, 0)  # (x, y) on minimap
    distance_to_target: float = 0.0  # Euclidean distance in pixels

    # Agent state
    agent_rotation: float = (
        0.0  # rot_y in degrees (0=North, 90=East, 180=South, 270=West)
    )

    # Context
    allowed_actions: List[str] = field(default_factory=list)


@dataclass
class GlobalPlannerOutput:
    """Output from Global Planner Agent."""

    strategic_direction: str  # Cardinal/ordinal: "north", "northeast", "east", etc.
    recommended_actions: List[str]  # Ordered preferences
    confidence: float  # 0.0 to 1.0
    reasoning: str


@dataclass
class LocalPlannerInput:
    """Input for Local Planner Agent."""

    # Images (RGB uint8)
    egocentric_view: np.ndarray  # Shape (H, W, 3), first-person camera (REQUIRED)
    bev_view: Optional[np.ndarray] = None  # Shape (H, W, 3), bird's eye view (OPTIONAL)

    # Metadata
    current_heading: float = 0.0  # Agent's rotation in degrees

    # Context
    allowed_actions: List[str] = field(default_factory=list)

    target_bearing: float = (
        0.0  # Relative angle to target (-180 to 180, 0=straight ahead)
    )
    target_distance: float = 0.0  # Distance to target (for urgency assessment)


@dataclass
class LocalPlannerOutput:
    """Output from Local Planner Agent."""

    immediate_obstacles: List[Dict] = field(
        default_factory=list
    )  # [{"type": str, "direction": str, "distance": str}]
    safe_directions: List[str] = field(
        default_factory=list
    )  # Actions that are obstacle-free
    unsafe_actions: List[str] = field(
        default_factory=list
    )  # Actions that would cause collision

    path_clearance: str = "unknown"  # "clear", "narrow", "blocked"
    recommended_actions: List[str] = field(
        default_factory=list
    )  # Ordered by preference
    confidence: float = 0.5
    reasoning: str = ""

    emergency: bool = False  # True if immediate danger


@dataclass
class DecisionMakerInput:
    """Input for Decision Maker Agent."""

    # Agent recommendations
    global_plan: GlobalPlannerOutput
    local_plan: LocalPlannerOutput

    # Current state
    current_position: Tuple[int, int] = (0, 0)  # Minimap coordinates
    target_position: Tuple[int, int] = (0, 0)
    distance_to_target: float = 0.0

    # Agent physical state
    agent_world_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    agent_rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Context
    allowed_actions: List[str] = field(default_factory=list)
    reach_threshold: float = 3.0  # Stop if within this distance (pixels)


@dataclass
class DecisionMakerOutput:
    """Output from Decision Maker Agent."""

    action: str  # Final decision from allowed_actions
    reasoning: str  # Full explanation
    confidence: float = 0.5

    # Flags
    is_emergency_stop: bool = False
    is_goal_reached: bool = False
    is_stuck: bool = False
