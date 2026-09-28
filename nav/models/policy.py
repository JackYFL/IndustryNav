"""Backward-compatible BC policy imports.

New code should import from :mod:`nav.models.policies`.
"""

from nav.models.policies.bc import (
    NavPolicy,
    NavPolicyDiffusion,
    NavPolicyRNN,
    NavPolicyTransformer,
    build_policy,
)

__all__ = [
    "NavPolicy",
    "NavPolicyRNN",
    "NavPolicyTransformer",
    "NavPolicyDiffusion",
    "build_policy",
]
