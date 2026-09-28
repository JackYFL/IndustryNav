"""Online collision detection from commanded and observed displacement."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from nav.config import (
    EVAL_COLLISION_MIN_FORWARD_RATIO,
    EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M,
)
from nav.safety.types import SafetyAssessment


class CollisionDetector:
    """Detect blocked positive movement using Unity's physical step scale."""

    def __init__(
        self,
        min_forward_ratio: float = EVAL_COLLISION_MIN_FORWARD_RATIO,
        forward_distance_per_move_unit_m: float = (
            EVAL_FORWARD_DISTANCE_PER_MOVE_UNIT_M
        ),
    ) -> None:
        self.min_forward_ratio = float(min_forward_ratio)
        self.forward_distance_per_move_unit_m = float(
            forward_distance_per_move_unit_m
        )
        if not 0.0 < self.min_forward_ratio <= 1.0:
            raise ValueError("min_forward_ratio must be in (0, 1]")
        if self.forward_distance_per_move_unit_m <= 0.0:
            raise ValueError(
                "forward_distance_per_move_unit_m must be positive"
            )

    def detect(
        self,
        move_command: float,
        previous_position: Optional[Tuple[float, float]],
        current_position: Optional[Tuple[float, float]],
    ) -> SafetyAssessment:
        """Compare actual X/Z displacement against the commanded distance."""
        move = float(move_command)
        eligible = (
            math.isfinite(move)
            and move > 0.0
            and previous_position is not None
            and current_position is not None
        )
        if not eligible:
            return SafetyAssessment(triggered=False, eligible=False)
        actual = math.hypot(
            current_position[0] - previous_position[0],
            current_position[1] - previous_position[1],
        )
        expected = move * self.forward_distance_per_move_unit_m
        threshold = expected * self.min_forward_ratio
        return SafetyAssessment(
            triggered=actual < threshold,
            actual_distance_m=actual,
            expected_distance_m=expected,
            threshold_m=threshold,
        )
