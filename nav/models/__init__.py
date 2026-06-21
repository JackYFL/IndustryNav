"""Behavior-cloning navigation models.

The cnn / resnet / dino "bases" are backbone strings (resolved via
``nav.config.BC_BASE_PRESETS``), not separate model classes — so this package
splits by *policy head*, not by backbone:

- :mod:`nav.models.encoder` — the shared ``TimmEncoder`` + construction helpers.
- :mod:`nav.models.policy` — the four policy heads + :func:`build_policy`.
"""

from nav.models.encoder import TimmEncoder, build_encoder_pair, is_vit_like
from nav.models.policy import (
    NavPolicy,
    NavPolicyDiffusion,
    NavPolicyRNN,
    NavPolicyTransformer,
    build_policy,
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
]
