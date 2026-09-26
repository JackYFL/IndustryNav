"""Distance curricula retain validation and cross-role spatial exclusions."""

import unittest
import contextlib
import io
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from nav.data.navigation_map import NavigationMap
from nav.scripts.tools.sample_independent_pointgoal_pairs import (
    sample_scene, validate_training_start_clearance, validate_training_clearance, main,
)


class IndependentSamplingTest(unittest.TestCase):
    def test_cli_reads_all_exclusion_manifests_and_preserves_single_file_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.json"
            canonical.write_text(json.dumps({"scene1": []}))
            paths = [root / "old1.jsonl", root / "old2.jsonl"]
            old = [dict(scene_name="scene1", episode_id=f"old_{i}") for i in range(2)]
            for path, row in zip(paths, old):
                path.write_text(json.dumps(row)+"\n")
            records = [dict(scene_name="scene1", split="val", difficulty=kind,
                            episode_id=f"val_{i}")
                       for i, kind in enumerate(("short", "medium", "long", "detour"))]
            records.append(dict(scene_name="scene1", split="train", episode_id="new"))
            for count in (1, 2):
                output = root / f"result_{count}"
                argv = ["sampler", "--map-dir", str(root), "--input-points", str(canonical),
                        "--output-dir", str(output), "--scenes", "1"]
                for path in paths[:count]:
                    argv.extend(["--excluded-training-manifest", str(path)])
                if count == 2:
                    argv.extend(["--training-goal-clearance-m", ".25", ".75"])
                with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                     patch("nav.scripts.tools.sample_independent_pointgoal_pairs.NavigationMap.load"), \
                     patch("nav.scripts.tools.sample_independent_pointgoal_pairs.sample_scene",
                           return_value=(records, {})) as sampler:
                    main()
                self.assertEqual(sampler.call_args.kwargs["excluded_pair_records"], old[:count])
                audit = json.loads((output / "sampling_audit.json").read_text())
                if count == 1:
                    self.assertEqual(audit["excluded_training_sha256"], hashlib.sha256(paths[0].read_bytes()).hexdigest())
                    self.assertNotIn("excluded_training_manifests_sha256", audit)
                else:
                    self.assertIsNone(audit["excluded_training_sha256"])
                    self.assertEqual(audit["excluded_training_manifests_sha256"],
                                     {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
                    self.assertEqual(sampler.call_args.kwargs["training_goal_clearance_m"], [.25, .75])
                    self.assertEqual(audit["training_goal_clearance_range_m"], [.25, .75])

    def clutter_fixture(self, scale=1.):
        side = int(75 / scale)
        free = np.ones((side, side), bool)
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        geometry = NavigationMap(free, {
            "scene_name": "scene1", "scene_id": 0, "world_origin": [0, 0],
            "world_per_cell": [[scale, 0], [0, scale]], "grid_step_px": [2, 2],
            "minimap_size": [2*side, 2*side], "margin": [], "projector": {},
            "obstacle_clearance_m": .45,
        })
        canonical = [dict(start={"x": 2., "z": 2.}, target={"x": 141, "y": 5})]
        validation = [dict(scene_name="scene1", episode_id=f"independent_val_{i:04d}",
            split="val", difficulty="long", init_world_x=10., init_world_z=z,
            sampled_target_world_x=65., sampled_target_world_z=gz,
            euclidean_distance_m=float(np.hypot(55, z-gz)))
            for i, (z, gz) in enumerate(((10., 65.), (60., 10.)))]
        options = dict(seed=723, train_count=12, validation_count=2,
            benchmark_clearance_m=4., split_clearance_m=3.,
            reserved_validation_records=validation, training_min_distance_m=30.,
            training_long_fraction=.5, training_episode_prefix="clutter_train")
        conversion = patch("nav.scripts.tools.sample_independent_pointgoal_pairs.visual_to_world_coords",
                           side_effect=lambda _m, p, *_a, **_k: (np.asarray(p)/2-.5)*scale)
        return geometry, canonical, options, conversion

    def test_clutter_starts_keep_goals_validation_exclusions_and_distance_quotas(self):
        geometry, canonical, options, conversion = self.clutter_fixture()
        with conversion:
            legacy, legacy_audit = sample_scene(geometry, canonical, **options)
            explicit_default, explicit_audit = sample_scene(geometry, canonical,
                                                           training_start_clearance_m=None, **options)
            first, audit = sample_scene(geometry, canonical, training_start_clearance_m=(1., 1.01), **options)
            repeat, _ = sample_scene(geometry, canonical, training_start_clearance_m=(1., 1.01), **options)
            second, second_audit = sample_scene(geometry, canonical, training_start_clearance_m=(1., 1.01),
                excluded_pair_records=[r for r in first if r["split"] == "train"], **options)
        self.assertEqual(legacy, explicit_default)
        self.assertEqual(legacy_audit, explicit_audit)
        self.assertEqual(first, repeat)
        self.assertEqual(first[:2], options["reserved_validation_records"])
        train = [r for r in first if r["split"] == "train"]
        self.assertEqual(len(train), 12)
        self.assertEqual(sum(r["euclidean_distance_m"] >= 40 for r in train), 6)
        self.assertTrue(all(r["euclidean_distance_m"] >= 30 for r in train))
        self.assertTrue(all(r["sampling_start_grid_clearance_m"] == 1. for r in train))
        self.assertTrue(all(r["budget_probe_forward_step_m"] == .75 for r in train))
        # Goal cells retain the old >=1.8-cell eligibility, even though starts do not.
        for r in train:
            goal = (r["sampled_target_world_x"], r["sampled_target_world_z"])
            self.assertGreaterEqual(min(*goal, 74-goal[0], 74-goal[1]), 2.)
        for result in (audit, second_audit):
            self.assertGreaterEqual(result["min_benchmark_endpoint_clearance_m"], 4.)
            self.assertGreaterEqual(result["min_train_val_endpoint_clearance_m"], 3.)
            self.assertEqual(result["unique_unordered_pairs"], 14)
            self.assertEqual(result["sampling_map_obstacle_inflation_m"], .45)
        def pair(r):
            return tuple(sorted(((r["init_world_x"], r["init_world_z"]),
                                 (r["sampled_target_world_x"], r["sampled_target_world_z"]))))
        self.assertFalse({pair(r) for r in train} & {pair(r) for r in second})
        self.assertTrue(all("sampling_start_grid_clearance_m" not in r for r in legacy))

    def test_clutter_range_is_in_world_meters_at_two_resolutions(self):
        for scale in (1., .5):
            geometry, canonical, options, conversion = self.clutter_fixture(scale)
            with conversion:
                records, audit = sample_scene(geometry, canonical, training_start_clearance_m=(1., 1.01), **options)
            self.assertEqual(audit["training_start_grid_clearance_m"], {"min": 1., "max": 1.})
            for r in records[2:]:
                x, y = geometry.world_to_cell((r["init_world_x"], r["init_world_z"]))
                self.assertAlmostEqual(min(x, y, geometry.width-1-x, geometry.height-1-y)*scale, 1.)

    def test_empty_clutter_range_does_not_relax_or_return_partial_data(self):
        geometry, canonical, options, conversion = self.clutter_fixture()
        with conversion, self.assertRaisesRegex(ValueError, "No eligible.*do not relax"):
            sample_scene(geometry, canonical, training_start_clearance_m=(.1, .2), **options)

    def test_goal_clutter_keeps_default_starts_and_reserved_validation(self):
        geometry, canonical, options, conversion = self.clutter_fixture()
        with conversion:
            default, default_audit = sample_scene(geometry, canonical, **options)
            explicit, explicit_audit = sample_scene(geometry, canonical, training_goal_clearance_m=None, **options)
            records, audit = sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01), **options)
            repeat, _ = sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01), **options)
            excluded, excluded_audit = sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01),
                excluded_pair_records=records[2:], **options)
        self.assertEqual((default, default_audit), (explicit, explicit_audit))
        self.assertEqual(records, repeat)
        self.assertEqual(records[:2], options["reserved_validation_records"])
        self.assertTrue(all("sampling_goal_grid_clearance_m" not in row for row in default))
        self.assertEqual(audit["training_goal_grid_clearance_m"], {"min": 1., "max": 1.})
        self.assertEqual(sum(r["euclidean_distance_m"] >= 40 for r in records[2:]), 6)
        for row in records[2:]:
            self.assertEqual(row["sampling_goal_grid_clearance_m"], 1.)
            x, y = row["init_world_x"], row["init_world_z"]
            self.assertGreaterEqual(min(x, y, 74-x, 74-y), 2.)
            self.assertNotIn("sampling_start_grid_clearance_m", row)
        for result in (audit, excluded_audit):
            self.assertGreaterEqual(result["min_benchmark_endpoint_clearance_m"], 4.)
            self.assertGreaterEqual(result["min_train_val_endpoint_clearance_m"], 3.)
            self.assertEqual(result["unique_unordered_pairs"], 14)
        def pair(row):
            return tuple(sorted(((row["init_world_x"], row["init_world_z"]),
                                 (row["sampled_target_world_x"], row["sampled_target_world_z"]))))
        self.assertFalse({pair(r) for r in records[2:]} & {pair(r) for r in excluded})

    def test_goal_and_both_endpoint_bands_use_meters_at_two_resolutions(self):
        for scale in (1., .5):
            for both in (False, True):
                with self.subTest(scale=scale, both=both):
                    geometry, canonical, options, conversion = self.clutter_fixture(scale)
                    with conversion:
                        rows, audit = sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01),
                            training_start_clearance_m=(1., 1.01) if both else None, **options)
                    self.assertEqual(rows[:2], options["reserved_validation_records"])
                    for row in rows[2:]:
                        goal = (row["sampled_target_world_x"], row["sampled_target_world_z"])
                        x, y = geometry.world_to_cell(goal)
                        self.assertAlmostEqual(min(x, y, geometry.width-1-x, geometry.height-1-y)*scale, 1.)
                        if both:
                            self.assertEqual(row["sampling_start_grid_clearance_m"], 1.)
                    self.assertEqual(audit["sampling_map_obstacle_inflation_m"], .45)

    def test_goal_options_preserve_newly_sampled_validation_too(self):
        geometry, canonical, options, conversion = self.clutter_fixture()
        options["reserved_validation_records"] = None
        with conversion:
            original, _ = sample_scene(geometry, canonical, **options)
            for start_range in (None, (1., 1.01)):
                rows, _ = sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01),
                    training_start_clearance_m=start_range, **options)
                self.assertEqual(rows[:2], original[:2])
                self.assertTrue(all("sampling_goal_grid_clearance_m" not in r for r in rows[:2]))

    def test_empty_goal_range_does_not_relax(self):
        geometry, canonical, options, conversion = self.clutter_fixture()
        with conversion, self.assertRaisesRegex(ValueError, "No eligible.*training goals.*do not relax"):
            sample_scene(geometry, canonical, training_goal_clearance_m=(.1, .2), **options)

    def test_goal_range_rejects_pixel_projection_to_a_different_cell(self):
        geometry, canonical, options, _ = self.clutter_fixture()
        with patch("nav.scripts.tools.sample_independent_pointgoal_pairs.visual_to_world_coords",
                   side_effect=lambda _m, p, *_a, **_k: np.asarray(p)/2-.5+np.asarray([3., 0.])), \
             self.assertRaisesRegex(ValueError, "Could not meet.*do not silently relax"):
            sample_scene(geometry, canonical, training_goal_clearance_m=(1., 1.01), max_attempts=3, **options)

    def test_invalid_goal_ranges_fail_before_cli_output_creation(self):
        for value in ((-1., 1.), (2., 1.), (1., 1.), (0., float("inf")), (float("nan"), 1.), (1.,)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Training goal clearance"):
                validate_training_clearance(value, role="goal")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            argv = ["sampler", "--map-dir", "not_read", "--output-dir", str(output),
                    "--training-goal-clearance-m", "2", "1"]
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertFalse(output.exists())

    def test_invalid_clutter_ranges_and_cli_fail_before_output_creation(self):
        for value in ((-1., 1.), (2., 1.), (1., 1.), (0., float("inf")), (float("nan"), 1.), (1.,)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_training_start_clearance(value)
        self.assertIsNone(validate_training_start_clearance(None))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist"
            argv = ["sampler", "--map-dir", "not_read", "--output-dir", str(output),
                    "--training-start-clearance-m", "2", "1"]
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()
            self.assertFalse(output.exists())

    def test_distance_curriculum_preserves_validation_and_excludes_prior_pairs(self):
        navmap = NavigationMap(np.ones((75, 75), bool), {
            "scene_name": "scene1", "scene_id": 0, "world_origin": [0, 0],
            "world_per_cell": [[1, 0], [0, 1]], "grid_step_px": [2, 2],
            "minimap_size": [150, 150], "margin": [], "projector": {},
        })
        canonical = [dict(start={"x": 2., "z": 2.}, target={"x": 141, "y": 5})]
        validation = [dict(scene_name="scene1", episode_id=f"independent_val_{i:04d}",
            split="val", difficulty="long", init_world_x=10., init_world_z=z,
            sampled_target_world_x=65., sampled_target_world_z=gz,
            euclidean_distance_m=float(np.hypot(55, z-gz)))
            for i, (z, gz) in enumerate(((10., 65.), (60., 10.)))]
        options = dict(seed=723, train_count=12, validation_count=2,
            benchmark_clearance_m=4., split_clearance_m=3.,
            reserved_validation_records=validation, training_min_distance_m=30.,
            training_long_fraction=.5, training_episode_prefix="long_train")
        def pair(record):
            return tuple(sorted(((record["init_world_x"],record["init_world_z"]),
                (record["sampled_target_world_x"],record["sampled_target_world_z"]))))
        with patch("nav.scripts.tools.sample_independent_pointgoal_pairs.visual_to_world_coords",
                   side_effect=lambda _m, p, *_a, **_k: np.asarray(p)/2-.5):
            first, _ = sample_scene(navmap, canonical, **options)
            records, audit = sample_scene(navmap, canonical,
                excluded_pair_records=[r for r in first if r["split"] == "train"], **options)
            repeated, _ = sample_scene(navmap, canonical,
                excluded_pair_records=[r for r in first if r["split"] == "train"], **options)
        self.assertEqual(records, repeated)
        self.assertEqual(records[:2], validation)
        train = [r for r in records if r["split"] == "train"]
        self.assertEqual(len(train), 12)
        self.assertEqual(sum(r["euclidean_distance_m"] >= 40 for r in train), 6)
        self.assertTrue(all(30 <= r["euclidean_distance_m"] <= 70 for r in train))
        self.assertTrue(all(r["budget_probe_forward_step_m"] == .75 for r in train))
        self.assertTrue(all(r["episode_id"].startswith("long_train_") for r in train))
        self.assertFalse({pair(r) for r in train} & {pair(r) for r in first})
        self.assertGreaterEqual(audit["min_benchmark_endpoint_clearance_m"], 4.)
        self.assertGreaterEqual(audit["min_train_val_endpoint_clearance_m"], 3.)
        self.assertEqual(audit["unique_unordered_pairs"], 14)
        self.assertTrue(audit["reserved_validation_reused"])


if __name__ == "__main__":
    unittest.main()
