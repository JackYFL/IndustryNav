"""Regression tests for the world-displacement collision metric."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from nav.eval.collision import compute_collision_rate


class CollisionRateTest(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        rows: list[dict],
    ) -> Path:
        actions_csv = root / "agent_actions.csv"
        with open(actions_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "action",
                    "move",
                    "curr_world_x",
                    "curr_world_z",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        return actions_csv

    def test_uses_actual_over_theoretical_world_displacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions_csv = self._write_run(
                Path(tmp),
                [
                    {
                        "step": 1,
                        "action": "forward",
                        "move": 15,
                        "curr_world_x": 0.0,
                        "curr_world_z": 0.0,
                    },
                    {
                        "step": 2,
                        "action": "forward",
                        "move": 15,
                        "curr_world_x": 1.5,
                        "curr_world_z": 0.0,
                    },
                    {
                        "step": 3,
                        "action": "turn left",
                        "move": 0,
                        "curr_world_x": 2.0,
                        "curr_world_z": 0.0,
                    },
                ],
            )

            self.assertEqual(compute_collision_rate(actions_csv), (2, 1, 0.5))

    def test_counts_compound_positive_move_and_allows_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions_csv = self._write_run(
                Path(tmp),
                [
                    {
                        "step": 1,
                        "action": "forward right",
                        "move": 7.5,
                        "curr_world_x": 0.0,
                        "curr_world_z": 0.0,
                    },
                    {
                        "step": 2,
                        "action": "stop",
                        "move": 0,
                        "curr_world_x": 0.74,
                        "curr_world_z": 0.0,
                    },
                ],
            )

            self.assertEqual(compute_collision_rate(actions_csv), (1, 0, 0.0))

    def test_uses_unity_controller_distance_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions_csv = self._write_run(
                Path(tmp),
                [
                    {
                        "step": 1,
                        "action": "forward",
                        "move": 15,
                        "curr_world_x": 0.0,
                        "curr_world_z": 0.0,
                    },
                    {
                        "step": 2,
                        "action": "stop",
                        "move": 0,
                        "curr_world_x": 0.74,
                        "curr_world_z": 0.0,
                    },
                ],
            )

            self.assertEqual(compute_collision_rate(actions_csv), (1, 1, 1.0))

    def test_rejects_invalid_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            actions_csv = self._write_run(Path(tmp), [])
            with self.assertRaises(ValueError):
                compute_collision_rate(actions_csv, min_forward_ratio=0.0)


if __name__ == "__main__":
    unittest.main()
