"""Pin evaluation horizons to audited task budgets, not reset-time pose jitter."""

import hashlib
import json
from pathlib import Path


def load_reference_budgets(path, expected_sha256, task_keys):
    """Read only task IDs/budgets; outcome fields never choose a horizon.

    A full report may cover a subset rerun, but every requested key must exist.
    Callers must separately audit that the pinned report is the intended protocol.
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("Step-budget reference hash mismatch")
    records = json.loads(raw)["episodes"]
    budgets = {}
    for row in records:
        key = (row["scene_name"], row["episode_id"])
        if not all(isinstance(k, str) and k for k in key) or key in budgets:
            raise ValueError("Invalid or duplicate task key in budget reference")
        budget = row["step_budget"]
        if type(budget) is not int or not 80 <= budget <= 320:
            raise ValueError("Reference budgets must be integers in [80, 320]")
        budgets[key] = budget
    keys = list(task_keys)
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(budgets):
        raise ValueError("Requested tasks are missing or duplicated in budget selection")
    return {key: budgets[key] for key in keys}, digest


def validate_completed_budgets(rows, budgets):
    """Never silently retain a resumed row evaluated under a different horizon."""
    if budgets is None:
        return
    seen = set()
    for row in rows:
        key = (row["scene_name"], row["episode_id"])
        if (key in seen or key not in budgets or type(row["step_budget"]) is not int
                or row["step_budget"] != budgets[key]):
            raise ValueError(f"Completed task has duplicate/unknown/different budget: {key}")
        seen.add(key)


def apply_reference_budget(env, task, budgets):
    """Apply the exact reference horizon immediately after reset, before actions."""
    if budgets is None:
        return None
    if env.episode_steps != 0:
        raise ValueError("A reference horizon may only be applied before the first action")
    key = (task["scene_name"], task["episode_id"])
    original = int(env.episode_max_steps)
    fixed = budgets[key]
    if type(fixed) is not int or not 80 <= fixed <= 320:
        raise ValueError("Invalid reference horizon")
    env.episode_max_steps = fixed
    return original
