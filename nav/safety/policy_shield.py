"""Small action shield for depth-based PointGoal policies."""

from __future__ import annotations

import numpy as np

from nav.safety.warning import WarningDetector


class ReactiveSafetyShield:
    """Recover from physical contact with a consistent clearance-seeking turn.

    The shield does not plan or use the minimap. Callers may trigger recovery
    after physical contact or after a configurable streak of pre-contact depth
    warnings. Left/right warning-region occupancy selects a short recovery
    turn, while the PointGoal bearing breaks near-ties.
    """

    def __init__(
        self,
        warning_detector: WarningDetector | None = None,
        *,
        max_clearance_turns: int | None = None,
    ) -> None:
        if max_clearance_turns is not None and max_clearance_turns < 0:
            raise ValueError("max_clearance_turns must be nonnegative or None")
        self.warning_detector = warning_detector or WarningDetector()
        self.max_clearance_turns = max_clearance_turns
        self.last_turn: str | None = None
        self.interventions = 0
        self.recovery_turns_remaining = 0
        self.recovery_forwards_remaining = 0
        self.clearance_turns_remaining = 0

    def reset(self) -> None:
        self.last_turn = None
        self.interventions = 0
        self.recovery_turns_remaining = 0
        self.recovery_forwards_remaining = 0
        self.clearance_turns_remaining = 0

    @staticmethod
    def _near_fraction(
        depth_m: np.ndarray,
        mask: np.ndarray,
        threshold_m: float,
    ) -> float:
        values = depth_m[mask]
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            return 1.0
        return float(np.mean(values < threshold_m))

    def begin_collision_recovery(
        self,
        depth_m: np.ndarray,
        *,
        goal_bearing_sin: float,
        move_command: float,
        turns: int = 2,
        escape_forwards: int = 0,
    ) -> None:
        """Choose and schedule a turn after a new physical collision."""
        if turns <= 0 or escape_forwards < 0:
            return
        verdict = self.warning_detector.detect(
            depth_m,
            move_command=move_command,
        )
        roi = self.warning_detector.create_roi_mask(depth_m.shape) == 1
        midpoint = depth_m.shape[1] // 2
        left_mask = roi.copy()
        left_mask[:, midpoint:] = False
        right_mask = roi.copy()
        right_mask[:, :midpoint] = False
        threshold = float(verdict["threshold_m"])
        left_blocked = self._near_fraction(depth_m, left_mask, threshold)
        right_blocked = self._near_fraction(depth_m, right_mask, threshold)

        if left_blocked + 0.01 < right_blocked:
            selected = "turn left"
        elif right_blocked + 0.01 < left_blocked:
            selected = "turn right"
        elif self.last_turn is not None:
            selected = self.last_turn
        else:
            selected = "turn right" if goal_bearing_sin >= 0.0 else "turn left"
        self.last_turn = selected
        self.recovery_turns_remaining = max(
            self.recovery_turns_remaining, int(turns)
        )
        self.recovery_forwards_remaining = max(
            self.recovery_forwards_remaining, int(escape_forwards)
        )
        if self.max_clearance_turns is not None:
            self.clearance_turns_remaining = max(
                self.clearance_turns_remaining,
                int(self.max_clearance_turns),
            )

    def filter_action(
        self,
        action_name: str,
        *,
        depth_m: np.ndarray | None = None,
        move_command: float = 7.5,
    ) -> tuple[str, bool]:
        """Apply a pending contact-recovery maneuver.

        Optional escape-forward actions prevent a goal-seeking policy from
        immediately undoing the clearance turn.  They are executed only when
        the current warning ROI is clear; otherwise the shield keeps turning
        consistently until a safe opening appears.
        """
        if self.recovery_turns_remaining > 0 and self.last_turn is not None:
            self.recovery_turns_remaining -= 1
            self.interventions += 1
            return self.last_turn, True
        if self.recovery_forwards_remaining <= 0:
            return action_name, False
        if depth_m is not None:
            blocked = self.warning_detector.detect(
                depth_m,
                move_command=move_command,
            )["warning"] == "yes"
            if blocked and self.last_turn is not None:
                if self.max_clearance_turns is not None:
                    if self.clearance_turns_remaining <= 0:
                        # A very wide/static obstacle can remain in the ROI
                        # after a full clearance maneuver.  Relinquish control
                        # instead of turning for the rest of the episode.
                        self.recovery_forwards_remaining = 0
                        return action_name, False
                    self.clearance_turns_remaining -= 1
                self.interventions += 1
                return self.last_turn, True
        self.recovery_forwards_remaining -= 1
        self.interventions += 1
        return "forward", True


__all__ = ["ReactiveSafetyShield"]
