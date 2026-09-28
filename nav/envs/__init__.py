"""Environment adapters used by training baselines."""

from nav.envs.base import NavigationEnvironment

__all__ = ["NavigationEnvironment", "UnityPointGoalEnv"]


def __getattr__(name: str):
    if name == "UnityPointGoalEnv":
        from nav.envs.unity_pointgoal import UnityPointGoalEnv

        return UnityPointGoalEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
