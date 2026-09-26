"""Stateful PointGoal shield matching the fixed canonical evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from nav.safety.policy_shield import ReactiveSafetyShield


@dataclass(frozen=True)
class PointGoalShieldConfig:
    collision_streak: int = 8
    recovery_turns: int = 6
    recovery_forward_steps: int = 4
    max_clearance_turns: int = 8
    warning_streak: int = 8
    warning_turns: int = 2
    turn_streak: int = 12
    turn_escape_steps: int = 4
    move_command: float = 7.5

    def __post_init__(self):
        if self.collision_streak <= 0 or self.recovery_turns <= 0 or self.warning_turns <= 0:
            raise ValueError("contact thresholds and recovery turns must be positive")
        for name in (
            "recovery_forward_steps", "max_clearance_turns", "warning_streak",
            "turn_streak", "turn_escape_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not math.isfinite(self.move_command) or self.move_command <= 0:
            raise ValueError("move_command must be finite and positive")


class PointGoalSafetyProtocol:
    """Contact, warning, and turn-escape handling without a planner or expert.

    ``filter_action`` runs before the physical step; ``observe_step`` runs
    afterwards using the new depth and goal. No terminal homing or fallback
    policy is enabled. Regression tests compare this state machine directly
    against the existing benchmark evaluator, which remains unchanged.
    """

    def __init__(self, config: PointGoalShieldConfig | None = None):
        self.config = config or PointGoalShieldConfig()
        self.reactive = ReactiveSafetyShield(
            max_clearance_turns=self.config.max_clearance_turns,
        )
        self.reset()

    def reset(self) -> None:
        self.reactive.reset()
        self.collision_streak = 0
        self.warning_streak = 0
        self.turn_streak = 0
        self.turn_escape_remaining = 0
        self.turn_escape_interventions = 0
        self.proactive_warning_recoveries = 0

    @property
    def interventions(self) -> int:
        return self.reactive.interventions + self.turn_escape_interventions

    def filter_action(
        self, action_name: str, *, depth_m: np.ndarray, goal_bearing_sin: float,
    ) -> tuple[str, bool]:
        config = self.config
        forward_warning = self.reactive.warning_detector.detect(
            depth_m, move_command=config.move_command,
        )["warning"] == "yes"
        warned_forward = action_name == "forward" and forward_warning
        self.warning_streak = self.warning_streak + 1 if warned_forward else 0
        if config.warning_streak > 0 and self.warning_streak >= config.warning_streak:
            self.reactive.begin_collision_recovery(
                depth_m, goal_bearing_sin=goal_bearing_sin,
                move_command=config.move_command, turns=config.warning_turns,
                escape_forwards=config.recovery_forward_steps,
            )
            self.warning_streak = 0
            self.proactive_warning_recoveries += 1
        action_name, intervened = self.reactive.filter_action(
            action_name, depth_m=depth_m, move_command=config.move_command,
        )
        if not intervened and self.turn_escape_remaining > 0:
            if forward_warning:
                self.turn_escape_remaining = 0
            else:
                action_name, intervened = "forward", True
                self.turn_escape_remaining -= 1
                self.turn_escape_interventions += 1
        prospective_turn_streak = (
            self.turn_streak + 1 if action_name in {"turn left", "turn right"} else 0
        )
        if (
            not intervened and config.turn_streak > 0
            and prospective_turn_streak >= config.turn_streak and not forward_warning
        ):
            action_name, intervened = "forward", True
            self.turn_escape_remaining = max(config.turn_escape_steps - 1, 0)
            self.turn_escape_interventions += 1
        self.turn_streak = (
            self.turn_streak + 1 if action_name in {"turn left", "turn right"} else 0
        )
        return action_name, intervened

    def observe_step(
        self, *, collision: bool, depth_m: np.ndarray, goal_bearing_sin: float,
    ) -> None:
        self.collision_streak = self.collision_streak + 1 if collision else 0
        if self.collision_streak >= self.config.collision_streak:
            self.reactive.begin_collision_recovery(
                depth_m, goal_bearing_sin=goal_bearing_sin,
                move_command=self.config.move_command,
                turns=self.config.recovery_turns,
                escape_forwards=self.config.recovery_forward_steps,
            )
            self.collision_streak = 0
