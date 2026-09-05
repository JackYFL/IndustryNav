"""Regression tests for shared benchmark step-budget initialization."""

from __future__ import annotations

import unittest

from nav.scripts.run_benchmark_cell import (
    astar_step_allowance,
    resolve_dynamic_step_budget,
)


class StepBudgetTest(unittest.TestCase):
    def test_llm_defaults_to_dynamic_initialization(self) -> None:
        self.assertTrue(resolve_dynamic_step_budget("llm", None))

    def test_other_baselines_keep_legacy_default(self) -> None:
        for baseline in ("astar", "bc", "random"):
            with self.subTest(baseline=baseline):
                self.assertFalse(resolve_dynamic_step_budget(baseline, None))

    def test_explicit_toggle_overrides_baseline_default(self) -> None:
        self.assertFalse(resolve_dynamic_step_budget("llm", False))
        self.assertTrue(resolve_dynamic_step_budget("bc", True))

    def test_distance_budget_matches_historical_protocol(self) -> None:
        self.assertEqual(astar_step_allowance(40.39, 50, 160, 1.25, 20), 71)
        self.assertEqual(astar_step_allowance(0, 50, 160, 1.25, 20), 50)
        self.assertEqual(astar_step_allowance(200, 50, 160, 1.25, 20), 160)


if __name__ == "__main__":
    unittest.main()
