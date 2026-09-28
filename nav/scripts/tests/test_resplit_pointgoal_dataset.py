from __future__ import annotations

import unittest
from collections import Counter

from nav.scripts.tools.resplit_pointgoal_dataset import (
    assign_scene_balanced_splits,
)


class ResplitPointGoalDatasetTest(unittest.TestCase):
    def test_every_scene_contributes_to_every_split(self):
        records = [
            {
                "scene_name": f"scene{scene}",
                "episode_dir": f"scene{scene}/point{episode}",
                "split": "train" if scene <= 16 else "test",
            }
            for scene in range(1, 25)
            for episode in range(8)
        ]
        assigned = assign_scene_balanced_splits(
            records, train_per_scene=6, val_per_scene=1, seed=17
        )
        by_scene = {
            f"scene{scene}": Counter(
                record["split"]
                for record in assigned
                if record["scene_name"] == f"scene{scene}"
            )
            for scene in range(1, 25)
        }
        self.assertTrue(
            all(counts == {"train": 6, "val": 1, "test": 1} for counts in by_scene.values())
        )
        self.assertTrue(all("source_split" in record for record in assigned))

    def test_split_is_deterministic(self):
        records = [
            {"scene_name": "scene1", "episode_dir": f"scene1/p{i}", "split": "train"}
            for i in range(8)
        ]
        first = assign_scene_balanced_splits(
            records, train_per_scene=6, val_per_scene=1, seed=5
        )
        second = assign_scene_balanced_splits(
            records, train_per_scene=6, val_per_scene=1, seed=5
        )
        self.assertEqual(first, second)

    def test_fixed_holdouts_use_all_remaining_episodes_for_training(self):
        records = [
            {"scene_name": "scene1", "episode_dir": f"scene1/p{i}", "split": "test"}
            for i in range(12)
        ]
        assigned = assign_scene_balanced_splits(
            records, val_per_scene=1, test_per_scene=1, seed=5
        )
        self.assertEqual(
            Counter(record["split"] for record in assigned),
            {"train": 10, "val": 1, "test": 1},
        )


if __name__ == "__main__":
    unittest.main()
