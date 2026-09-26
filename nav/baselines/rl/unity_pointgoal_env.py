"""Backward-compatible PPO-configured Unity environment import."""

from nav.baselines.rl.actions import PPO_ACTIONS
from nav.baselines.rl.reward import compute_pointgoal_reward
from nav.envs.unity_pointgoal import UnityPointGoalEnv as _UnityPointGoalEnv


class UnityPointGoalEnv(_UnityPointGoalEnv):
    """Legacy wrapper that supplies the PPO action vocabulary and reward."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("action_names", PPO_ACTIONS)
        kwargs.setdefault("reward_fn", compute_pointgoal_reward)
        super().__init__(*args, **kwargs)


__all__ = ["UnityPointGoalEnv"]
