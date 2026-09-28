"""Reinforcement-learning components for learned navigation baselines."""

from nav.baselines.rl.actions import (
    PPO_ACTIONS,
    PPO_ACTION_TO_LABEL,
    PPO_BOS_LABEL,
    append_action_history,
    initial_action_history,
    pointgoal_homing_action,
)
from nav.baselines.rl.reward import (
    collision_started,
    compute_pointgoal_reward,
    is_turn_reversal,
    should_trigger_stuck_recovery,
)

__all__ = [
    "PPO_ACTIONS",
    "PPO_ACTION_TO_LABEL",
    "PPO_BOS_LABEL",
    "append_action_history",
    "initial_action_history",
    "pointgoal_homing_action",
    "PPOPointGoalAgent",
    "PPOPointGoalController",
    "compute_pointgoal_reward",
    "collision_started",
    "is_turn_reversal",
    "should_trigger_stuck_recovery",
    "SceneWaypointController",
    "RouteGraphController",
]


def __getattr__(name: str):
    if name in {"PPOPointGoalAgent", "PPOPointGoalController"}:
        from nav.baselines.rl.agent import (
            PPOPointGoalAgent,
            PPOPointGoalController,
        )

        return {
            "PPOPointGoalAgent": PPOPointGoalAgent,
            "PPOPointGoalController": PPOPointGoalController,
        }[name]
    if name == "SceneWaypointController":
        from nav.baselines.rl.waypoint import SceneWaypointController

        return SceneWaypointController
    if name == "RouteGraphController":
        from nav.baselines.rl.route_graph import RouteGraphController

        return RouteGraphController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
