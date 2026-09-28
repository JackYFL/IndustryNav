"""Online safety detectors shared by environments, evaluation, and galleries."""

from nav.safety.types import SafetyAssessment

__all__ = [
    "CollisionDetector",
    "ReactiveSafetyShield",
    "SafetyAssessment",
    "WarningDetector",
]


def __getattr__(name: str):
    if name == "CollisionDetector":
        from nav.safety.collision import CollisionDetector

        return CollisionDetector
    if name == "WarningDetector":
        from nav.safety.warning import WarningDetector

        return WarningDetector
    if name == "ReactiveSafetyShield":
        from nav.safety.policy_shield import ReactiveSafetyShield

        return ReactiveSafetyShield
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
