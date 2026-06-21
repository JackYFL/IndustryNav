"""Classical / non-VLLM navigation baselines.

- :class:`AStarBaseline` — minimap A* pathfinding → egocentric actions.
- :class:`NaVidBaseline` — adapter for the external NaVid VLN-CE checkpoint.

These are pure decision classes (``.decide(...)``); the benchmark runner +
``nav.harness.routing`` own "which baseline gets which input". The NaVid
adapter imports torch + the external NaVid repo lazily inside its
constructor, so importing this package does not require those to be present.
"""

from nav.baselines.astar import AStarBaseline
from nav.baselines.navid import NaVidBaseline

__all__ = ["AStarBaseline", "NaVidBaseline"]
