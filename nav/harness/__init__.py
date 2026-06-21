"""Runtime agent harness — the loop that drives the Unity client during a benchmark cell.

Contents:

- :mod:`nav.harness.llm_provider` — OpenRouter client (landed PR 2).
- :mod:`nav.harness.perception` — in-loop perception (red-dot detector).
- The multi-agent planner trio (:class:`GlobalPlannerAgent`,
  :class:`LocalPlannerAgent`, :class:`DecisionMakerAgent`) + their typed
  I/O dataclasses, re-exported here. This experimental planner system is
  currently exercised only by a deprecated editor-mode script; it's kept
  for a future multi-agent revival.

The single-call benchmark decision loop extracted from
``run_headless_benchmark.py`` lands here in PR 5b.
"""

from nav.harness.decision_maker import DecisionMakerAgent
from nav.harness.global_planner import GlobalPlannerAgent
from nav.harness.local_planner import LocalPlannerAgent
from nav.harness.types import (
    DecisionMakerInput,
    DecisionMakerOutput,
    GlobalPlannerInput,
    GlobalPlannerOutput,
    LocalPlannerInput,
    LocalPlannerOutput,
)

__all__ = [
    "GlobalPlannerAgent",
    "LocalPlannerAgent",
    "DecisionMakerAgent",
    "GlobalPlannerInput",
    "GlobalPlannerOutput",
    "LocalPlannerInput",
    "LocalPlannerOutput",
    "DecisionMakerInput",
    "DecisionMakerOutput",
]
