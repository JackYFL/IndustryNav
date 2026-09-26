"""Common safety-detector result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyAssessment:
    """Normalized boolean result plus detector-specific measurements."""

    triggered: bool
    eligible: bool = True
    actual_distance_m: float = 0.0
    expected_distance_m: float = 0.0
    threshold_m: float = 0.0
