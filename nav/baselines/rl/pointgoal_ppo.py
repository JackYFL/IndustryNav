"""Compatibility facade for the original PointGoal PPO import path.

Implementations now live in their owning layers: model architecture under
``nav.models``, task geometry under ``nav.core``, optimization math under
``nav.train``, and PPO-specific action/reward logic under this baseline.
"""

from nav.baselines.rl.actions import (
    PPO_ACTIONS,
    PPO_ACTION_TO_LABEL,
    PPO_BOS_LABEL,
    append_action_history,
    initial_action_history,
)
from nav.baselines.rl.reward import (
    collision_started,
    compute_pointgoal_reward,
    is_turn_reversal,
    should_trigger_stuck_recovery,
)
from nav.baselines.rl.tasks import (
    RandomSceneBlockTaskSampler,
    round_robin_scene_tasks,
)
from nav.core.pointgoal import encode_rl_goal, encode_rl_goal_state
from nav.data.preprocessing import depth_observation_uint8
from nav.models.policies import (
    PointGoalActorCritic,
    PointGoalPPOConfig,
    append_visual_history,
    initial_visual_history,
    load_compatible_pointgoal_state_dict,
)
from nav.train.advantages import generalized_advantage_estimate
from nav.train.initialization import load_bc_depth_encoder

__all__ = [
    "PPO_ACTIONS",
    "PPO_ACTION_TO_LABEL",
    "PPO_BOS_LABEL",
    "append_action_history",
    "initial_action_history",
    "PointGoalActorCritic",
    "PointGoalPPOConfig",
    "append_visual_history",
    "initial_visual_history",
    "load_compatible_pointgoal_state_dict",
    "compute_pointgoal_reward",
    "collision_started",
    "is_turn_reversal",
    "should_trigger_stuck_recovery",
    "depth_observation_uint8",
    "encode_rl_goal",
    "encode_rl_goal_state",
    "generalized_advantage_estimate",
    "load_bc_depth_encoder",
    "round_robin_scene_tasks",
    "RandomSceneBlockTaskSampler",
]
