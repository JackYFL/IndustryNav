"""Runtime perception helpers used inside the agent decision loop.

Currently holds the red-dot detector that locates the agent's position
marker on the minimap each step. This is in-the-loop perception, distinct
from the post-hoc analysis in :mod:`nav.eval`.
"""

from nav.harness.perception.red_detector import detect_red_point

__all__ = ["detect_red_point"]
