"""Regression tests for action-aware A*-normalized step efficiency."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from nav.eval.efficiency import (
    DEFAULT_ASTAR_RESULTS_DIR,
    AStarStepReference,
    attach_step_efficiency,
    compute_efficiency_step_budget_max,
    compute_step_efficiency,
    load_astar_step_references,
)


class StepEfficiencyTest(unittest.TestCase):
    def test_inverse_minmax_endpoints_and_midpoint(self) -> None:
        self.assertEqual(compute_step_efficiency(40, 40, 140), 1.0)
        self.assertEqual(compute_step_efficiency(140, 40, 140), 0.0)
        self.assertEqual(compute_step_efficiency(90, 40, 140), 0.5)

    def test_score_is_clipped_and_success_weighted(self) -> None:
        self.assertEqual(compute_step_efficiency(20, 40, 140), 1.0)
        self.assertEqual(compute_step_efficiency(180, 40, 140), 0.0)
        self.assertEqual(compute_step_efficiency(40, 40, 140, success=0), 0.0)
        self.assertEqual(compute_step_efficiency(90, 40, 140, success=0.5), 0.25)

    def test_invalid_or_degenerate_bounds_are_nan(self) -> None:
        self.assertTrue(math.isnan(compute_step_efficiency(50, None, 160)))
        self.assertTrue(math.isnan(compute_step_efficiency(50, 160, 160)))
        self.assertTrue(math.isnan(compute_step_efficiency(50, 170, 160)))

    def test_agent_conversion_accounts_for_larger_atomic_turns(self) -> None:
        reference = AStarStepReference(7, 30.0, 90.0, 1)
        self.assertEqual(reference.agent_steps(), 4)

    def test_loads_successful_astar_control_effort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "scene1" / "point1" / DEFAULT_ASTAR_RESULTS_DIR
            run_dir.mkdir(parents=True)
            with (run_dir / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "scene_name", "point_id", "distance_world", "reach_m",
                        "steps_taken", "sim_steps_per_decision",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "scene_name": "scene1", "point_id": "point1",
                    "distance_world": 1.5, "reach_m": 2.0,
                    "steps_taken": 3, "sim_steps_per_decision": 2,
                })
            with (run_dir / "astar_actions.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["action", "move", "look"]
                )
                writer.writeheader()
                writer.writerows([
                    {"action": "forward", "move": 15, "look": 0},
                    {"action": "astar turn right", "move": 0, "look": 9},
                    {"action": "stop", "move": 0, "look": 0},
                ])

            reference = load_astar_step_references(root)[("scene1", "point1")]
            self.assertEqual(reference.astar_steps, 3)
            self.assertEqual(reference.forward_signal_ticks, 30.0)
            self.assertEqual(reference.turn_signal_ticks, 18.0)
            self.assertEqual(reference.agent_steps(), 3)

    def test_step_budget_max_is_optimal_plus_configurable_k(self) -> None:
        self.assertEqual(compute_efficiency_step_budget_max(40), 140)
        self.assertEqual(compute_efficiency_step_budget_max(40, 25), 65)
        with self.assertRaises(ValueError):
            compute_efficiency_step_budget_max(40, 0)

    def test_attaches_different_optima_for_agent_and_astar(self) -> None:
        rows = [
            {
                "scene_name": "scene1", "point_id": "point1",
                "model": "gpt", "steps_taken": 14, "success": 1,
                "sim_steps_per_decision": 2,
            },
            {
                "scene_name": "scene1", "point_id": "point1",
                "exec_mode": "astar", "steps_taken": 7, "success": 1,
            },
        ]
        reference = AStarStepReference(7, 30.0, 90.0, 1)
        agent, astar = attach_step_efficiency(
            rows, {("scene1", "point1"): reference}, step_margin=20,
        )
        self.assertEqual(agent["optimal_steps"], 4)
        self.assertEqual(astar["optimal_steps"], 7)
        self.assertEqual(agent["efficiency_max_steps"], 24)
        self.assertEqual(astar["efficiency_max_steps"], 27)
        self.assertEqual(agent["efficiency"], 0.5)
        self.assertEqual(astar["efficiency"], 1.0)


if __name__ == "__main__":
    unittest.main()
