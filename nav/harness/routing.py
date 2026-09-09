"""Per-step decision routing: dispatch one decision by ``baseline``.

:func:`execute_decision` is the single dispatcher the benchmark loop runs
(in a worker thread, so the Unity step loop never blocks on a slow LLM
call). It writes its result into the caller-supplied ``result_container``
dict — ``action`` / ``reasoning`` always, ``error`` on hard failure, and
``finished`` in a ``finally`` so the loop can poll for completion.

Supported baselines (see :data:`nav.config.BENCHMARK_BASELINES`):

- ``"random"`` — uniform choice over the action space.
- ``"llm"`` — LLM via :func:`nav.harness.llm_provider.llm_generate_decision`.
- ``"bc"`` — a behavior-cloning controller's ``predict_action``.
- ``"astar"`` — minimap A* planner (:class:`nav.baselines.astar.AStarBaseline`).

``astar`` is synchronous, but routes through the same dispatcher so the entry
loop has one code path for every baseline.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from nav.config import LLM_ERROR_SENTINELS
from nav.harness.llm_provider import llm_generate_decision

logger = logging.getLogger(__name__)

# Seconds between in-episode LLM re-queries; scaled by the attempt index.
LLM_DECISION_RETRY_BACKOFF_SEC = 5.0


def _llm_decision_failed(decision: dict) -> bool:
    """True when a decision carries the provider error flag or an error sentinel.

    Only explicit sentinels count: a bare "failed" substring trips on
    legitimate chain-of-thought ("forward failed to change position").
    """
    if decision.get("error"):
        return True
    r_lower = str(decision.get("reasoning", "")).lower()
    return any(s in r_lower for s in LLM_ERROR_SENTINELS)


def astar_path_length_m(path, point_to_world) -> float | None:
    """Return the world-space length of an A* pixel path when calibrated."""
    if point_to_world is None or len(path) < 2:
        return None
    world_points = [point_to_world(point) for point in path]
    if any(point is None for point in world_points):
        return None
    return float(
        sum(
            np.hypot(
                float(current[0]) - float(previous[0]),
                float(current[1]) - float(previous[1]),
            )
            for previous, current in zip(world_points, world_points[1:])
        )
    )


def execute_decision(baseline: str, payload: dict, result_container: dict) -> None:
    """Run one decision and record it into ``result_container``.

    Any exception is caught and surfaced as ``action="stop"`` +
    ``error=True`` so a single bad decision degrades to a safe stop rather
    than crashing the run. ``result_container["finished"]`` is always set.
    """
    try:
        if baseline == "random":
            result_container["action"] = str(np.random.choice(payload["action_space"]))
            result_container["reasoning"] = "Random action."
        elif baseline == "llm":
            # A single bad reply (transport timeout, truncated/empty content,
            # malformed JSON) used to abort the whole episode. Re-query the
            # same observation up to ``llm_decision_retries`` extra times
            # before surfacing the error; the retry count is logged so the
            # per-step log stays auditable.
            retries = max(0, int(payload.get("llm_decision_retries", 0)))
            decision = None
            for attempt in range(retries + 1):
                decision = llm_generate_decision(
                    prompt=payload["prompt"],
                    images=payload["images"],
                    model=payload["model_id"],
                    provider=payload.get("llm_provider", "openrouter"),
                    max_tokens=payload["max_tokens"],
                    min_request_interval_sec=payload.get(
                        "llm_min_request_interval_sec", 0.0
                    ),
                    allowed_actions=payload["allowed_actions"],
                )
                if not _llm_decision_failed(decision):
                    break
                if attempt < retries:
                    delay = LLM_DECISION_RETRY_BACKOFF_SEC * (attempt + 1)
                    logger.warning(
                        "LLM decision failed (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, retries + 1, delay,
                        str(decision.get("reasoning", ""))[:200],
                    )
                    time.sleep(delay)
            result_container.update(decision)
            result_container["decision_attempts"] = attempt + 1
            result_container["prompt"] = payload["prompt"]
            result_container["history_entry"] = payload.get("history_entry", {})
            if _llm_decision_failed(decision):
                result_container["error"] = True
        elif baseline == "bc":
            result_container["action"] = payload["bc_controller"].predict_action(
                ego_obs=payload["ego_obs"],
                depth_obs=payload["depth_obs"],
                curr_world_x=payload["curr_world_x"],
                curr_world_z=payload["curr_world_z"],
                curr_yaw_deg=payload["curr_yaw_deg"],
                target_world_x=payload["target_world_x"],
                target_world_z=payload["target_world_z"],
            )
            result_container["reasoning"] = "BC model decision."
        elif baseline == "astar":
            action, reasoning, _path = payload["astar_planner"].decide(
                minimap_rgb=payload["minimap_rgb"],
                curr_xy=payload["curr_xy"],
                target_xy=payload["target_xy"],
                agent_theta=payload["agent_theta"],
                reach_m=payload["reach_m"],
                step=payload["step"],
                curr_world_xz=payload["curr_world_xz"],
                target_world_xz=payload.get("target_world_xz"),
                point_to_world=payload.get("point_to_world"),
            )
            result_container["action"] = action
            result_container["reasoning"] = reasoning
            result_container["astar_path_length_m"] = astar_path_length_m(
                _path,
                payload.get("point_to_world"),
            )
            result_container["astar_path"] = [
                [int(point[0]), int(point[1])] for point in _path
            ]
            result_container["astar_step"] = int(payload["step"])
            result_container["astar_plan_revision"] = (
                payload["astar_planner"].plan_revision
            )
            result_container["astar_plan_status"] = (
                payload["astar_planner"].last_plan_status
            )
            result_container["astar_tracking"] = dict(
                payload["astar_planner"].last_tracking_metrics
            )
        else:
            raise ValueError(f"Unsupported baseline: {baseline}")
    except Exception as e:  # noqa: BLE001 — degrade any failure to a safe stop
        result_container["action"] = "stop"
        result_container["reasoning"] = f"Decision error: {e}"
        result_container["error"] = True
    finally:
        result_container["finished"] = True
