"""Reusable navigation model architectures.

The cnn / resnet / dino "bases" are backbone strings (resolved via
``nav.config.BC_BASE_PRESETS``), not separate model classes — so this package
splits by *policy head*, not by backbone:

- :mod:`nav.models.encoders` — shared visual encoders.
- :mod:`nav.models.policies` — BC and recurrent actor-critic policy heads.
"""

from nav.models.encoders import TimmEncoder, build_encoder_pair, is_vit_like
from nav.models.policies import (
    NavPolicy,
    NavPolicyDiffusion,
    NavPolicyRNN,
    NavPolicyTransformer,
    PointGoalActorCritic,
    PointGoalPPOConfig,
    MemoryDaggerTransformerActorCritic,
    SceneWaypointConfig,
    SceneWaypointPlanner,
    append_visual_history,
    build_policy,
    initial_visual_history,
    load_compatible_pointgoal_state_dict,
)

__all__ = [
    "TimmEncoder",
    "build_encoder_pair",
    "is_vit_like",
    "NavPolicy",
    "NavPolicyRNN",
    "NavPolicyTransformer",
    "NavPolicyDiffusion",
    "build_policy",
    "PointGoalActorCritic",
    "PointGoalPPOConfig",
    "MemoryDaggerTransformerActorCritic",
    "SceneWaypointConfig",
    "SceneWaypointPlanner",
    "append_visual_history",
    "initial_visual_history",
    "load_compatible_pointgoal_state_dict",
]
