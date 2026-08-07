import csv
import tempfile
import unittest
from pathlib import Path

from nav.config import EVAL_SUCCESS_DIST_PX
from nav.eval.metrics import compute_success_efficiency_distance


class EvalMetricsTest(unittest.TestCase):
    def test_default_matches_paper_and_legacy_threshold_is_explicit(self):
        self.assertEqual(EVAL_SUCCESS_DIST_PX, 20)
        with tempfile.TemporaryDirectory() as tmp_dir:
            actions_csv = Path(tmp_dir) / "actions.csv"
            with actions_csv.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "curr_px",
                        "curr_py",
                        "target_px",
                        "target_py",
                        "distance_px",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "step": 1,
                    "curr_px": 0,
                    "curr_py": 0,
                    "target_px": 100,
                    "target_py": 0,
                    "distance_px": 30,
                })

            default_success, _, _ = compute_success_efficiency_distance(actions_csv)
            legacy_success, _, _ = compute_success_efficiency_distance(
                actions_csv, success_dist_px=65
            )

        self.assertEqual(default_success, 0)
        self.assertEqual(legacy_success, 1)


if __name__ == "__main__":
    unittest.main()
