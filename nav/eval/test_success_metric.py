"""Regression tests for fixed-radius success metrics."""

from __future__ import annotations

import unittest

from nav.eval.aggregate import summarize_by
from nav.eval.metrics import compute_success_at_thresholds


class SuccessThresholdMetricTest(unittest.TestCase):
    def test_thresholds_are_inclusive(self) -> None:
        self.assertEqual(
            compute_success_at_thresholds(2.0),
            {
                "success_at_2m": 1,
                "success_at_5m": 1,
                "success_at_10m": 1,
            },
        )

    def test_each_threshold_uses_final_world_distance(self) -> None:
        self.assertEqual(
            compute_success_at_thresholds(4.0),
            {
                "success_at_2m": 0,
                "success_at_5m": 1,
                "success_at_10m": 1,
            },
        )

    def test_missing_distance_is_not_success(self) -> None:
        expected = {
            "success_at_2m": 0,
            "success_at_5m": 0,
            "success_at_10m": 0,
        }
        self.assertEqual(compute_success_at_thresholds(None), expected)
        self.assertEqual(compute_success_at_thresholds(float("nan")), expected)

    def test_aggregate_summary_reports_each_threshold(self) -> None:
        rows = [
            {
                "scene_name": "scene1",
                "success_ratio": 0,
                **compute_success_at_thresholds(2.0),
                "distance_ratio": 0.5,
                "warning_rate": 0.1,
                "collision_rate": 0.2,
                "efficiency_steps": 10,
                "error": "",
            },
            {
                "scene_name": "scene1",
                "success_ratio": 0,
                **compute_success_at_thresholds(4.0),
                "distance_ratio": 0.25,
                "warning_rate": 0.2,
                "collision_rate": 0.3,
                "efficiency_steps": 20,
                "error": "",
            },
        ]

        summary = summarize_by(rows, "scene_name")[0]

        self.assertEqual(summary["success_at_2m_%"], 50.0)
        self.assertEqual(summary["success_at_5m_%"], 100.0)
        self.assertEqual(summary["success_at_10m_%"], 100.0)


if __name__ == "__main__":
    unittest.main()
