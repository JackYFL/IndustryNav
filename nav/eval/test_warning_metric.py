"""Regression tests for action-aware, resolution-normalized warning rate."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nav.eval.warning import WarningDetector, compute_warning_rate


class WarningDetectorTest(unittest.TestCase):
    def test_positive_move_extends_warning_distance(self) -> None:
        depth = np.full((240, 320), 1.0, dtype=np.float32)
        detector = WarningDetector()

        stationary = detector.detect(depth, move_command=0.0)
        moving = detector.detect(depth, move_command=15.0)

        self.assertEqual(stationary["warning"], "no")
        self.assertEqual(moving["warning"], "yes")
        self.assertAlmostEqual(stationary["threshold_m"], 0.4)
        self.assertAlmostEqual(moving["threshold_m"], 1.9)

    def test_single_near_pixel_does_not_trigger_warning(self) -> None:
        depth = np.full((240, 320), 5.0, dtype=np.float32)
        detector = WarningDetector()
        mask = detector.create_roi_mask(depth.shape)
        first_roi_pixel = tuple(np.argwhere(mask == 1)[0])
        depth[first_roi_pixel] = 0.1

        verdict = detector.detect(depth)

        self.assertEqual(verdict["warning"], "no")
        self.assertEqual(verdict["warning_pixels"], 1)
        self.assertLess(
            verdict["warning_pixel_ratio"],
            verdict["min_warning_pixel_ratio"],
        )

    def test_warning_is_consistent_across_current_sensor_resolutions(self) -> None:
        low_resolution = np.full((240, 320), 5.0, dtype=np.float32)
        detector = WarningDetector()
        mask = detector.create_roi_mask(low_resolution.shape)
        roi_pixels = np.argwhere(mask == 1)
        near_pixel_count = int(np.ceil(len(roi_pixels) * 0.01))
        selected = roi_pixels[:near_pixel_count]
        low_resolution[selected[:, 0], selected[:, 1]] = 0.1
        high_resolution = np.repeat(
            np.repeat(low_resolution, 2, axis=0),
            2,
            axis=1,
        )

        low = detector.detect(low_resolution)
        high = detector.detect(high_resolution)

        self.assertEqual(low["warning"], "yes")
        self.assertEqual(high["warning"], "yes")
        self.assertAlmostEqual(
            low["warning_pixel_ratio"],
            high["warning_pixel_ratio"],
        )

    def test_run_rate_uses_move_from_matching_action_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            depth_dir = run_dir / "agent_depth"
            depth_dir.mkdir()
            depth = np.full((240, 320), 1.0, np.float32)
            np.save(depth_dir / "1.npy", depth)
            np.save(depth_dir / "2.npy", depth)
            with open(run_dir / "agent_actions.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "move"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"step": 1, "move": 0.0},
                        {"step": 2, "move": 15.0},
                    ]
                )

            self.assertEqual(compute_warning_rate(run_dir), (2, 1, 0.5))


if __name__ == "__main__":
    unittest.main()
