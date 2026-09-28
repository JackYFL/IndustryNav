"""Audit a complete fixed-protocol 96-point report, without running navigation.

This verifies reported evidence, not the provenance of an untrusted report.
Checkpoint/source hashes and training-data audits must be retained separately.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


PROTOCOL = {
    "eval_seed": 0, "reach_m": 2.0, "persistent_policy": True,
    "safety_shield": True, "shield_collision_streak": 8,
    "shield_recovery_turns": 6, "shield_recovery_forward_steps": 4,
    "shield_max_clearance_turns": 8, "shield_warning_streak": 8,
    "shield_warning_turns": 2, "shield_turn_streak": 12,
    "shield_turn_escape_steps": 4, "shield_terminal_homing_distance_m": 0.0,
    "fallback_ppo_checkpoint": None, "fallback_bc_checkpoint": None,
    "second_fallback_bc_checkpoint": None,
    "waypoint_planner_checkpoint": None, "route_graph_checkpoint": None,
}
EXPECTED_KEYS = {(f"scene{s}", f"point{p}") for s in range(1, 25) for p in range(1, 5)}


def validate_report(report, *, expected_keys=None):
    expected_keys = EXPECTED_KEYS if expected_keys is None else set(expected_keys)
    count = len(expected_keys)
    if count <= 0:
        raise ValueError("Expected task set must be nonempty")
    for name, expected in PROTOCOL.items():
        if name not in report or report[name] != expected:
            raise ValueError(f"Unexpected or missing protocol field: {name}")
    episodes = report.get("episodes", [])
    keys = [(e["scene_name"], e["episode_id"]) for e in episodes]
    if len(keys) != count or set(keys) != expected_keys:
        raise ValueError("Expected each specified task exactly once")
    for key, episode in zip(keys, episodes):
        if episode.get("returncode") != 0:
            raise ValueError(f"Failed process: {key}")
        distance = episode["distance_world"]
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid final distance: {key}")
        if type(episode["success"]) is not bool or episode["success"] != (distance <= 2.0):
            raise ValueError(f"Success/distance mismatch: {key}")
        steps, budget = episode["steps_taken"], episode["step_budget"]
        if type(steps) is not int or type(budget) is not int or not (0 < steps <= budget <= 320 and budget >= 80):
            raise ValueError(f"Invalid step count or budget: {key}")
        if not episode["success"] and steps != budget:
            raise ValueError(f"Unsuccessful episode terminated before its budget: {key}")
        expected_reason = "reached_vicinity" if episode["success"] else "max_steps"
        if episode["stop_reason"] != expected_reason:
            raise ValueError(f"Unexpected termination reason: {key}")
        for field in ("terminal_homing_interventions", "fallback_activated", "second_fallback_activated"):
            if field not in episode or episode[field]:
                raise ValueError(f"Missing/active non-protocol intervention {field}: {key}")
        for field in ("collision_steps", "collision_events", "warning_steps", "safety_shield_interventions"):
            if type(episode[field]) is not int or not 0 <= episode[field] <= steps:
                raise ValueError(f"Invalid counter {field}: {key}")
        for field in ("predicted_action_counts", "executed_action_counts"):
            counts = episode[field]
            if set(counts) - {"forward", "turn left", "turn right"} or any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) != steps:
                raise ValueError(f"Action counts do not match navigation steps: {key}")
    successes = sum(e["success"] for e in episodes)
    if report.get("successes") != successes or not math.isclose(report.get("success_rate", -1), successes / count, abs_tol=1e-12):
        raise ValueError("Aggregate score does not match individual episodes")
    return dict(zip(keys, episodes))


def audit_result(report, reference):
    episodes, baseline = validate_report(report), validate_report(reference)
    for key, episode in episodes.items():
        if episode["step_budget"] != baseline[key]["step_budget"]:
            raise ValueError(f"Per-point budget differs from reference: {key}")
    successes = sum(e["success"] for e in episodes.values())
    return {
        "episodes": 96, "successes": successes, "success_rate": successes / 96,
        "score_above_60_percent": successes >= 58,
        "score_above_80_percent": successes >= 77,
        "gained_tasks": [list(k) for k in episodes if episodes[k]["success"] and not baseline[k]["success"]],
        "lost_tasks": [list(k) for k in episodes if not episodes[k]["success"] and baseline[k]["success"]],
        **{field: sum(e[field] for e in episodes.values()) for field in (
            "steps_taken", "collision_steps", "collision_events", "warning_steps",
            "safety_shield_interventions",
        )},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--input-points", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    args = parser.parse_args()
    input_hash = hashlib.sha256(args.input_points.read_bytes()).hexdigest()
    if input_hash != args.expected_input_sha256:
        raise SystemExit("Canonical input_points hash mismatch")
    result = audit_result(json.loads(args.summary.read_text()), json.loads(args.reference_summary.read_text()))
    result["input_points_sha256"] = input_hash
    result["summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
