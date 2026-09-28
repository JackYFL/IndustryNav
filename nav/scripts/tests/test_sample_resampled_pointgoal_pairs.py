from __future__ import annotations

import unittest

from nav.scripts.tools.sample_resampled_pointgoal_pairs import (
    Route,
    RoutePoint,
    build_manifest,
    candidate_pairs_for_scene,
    load_manifest_exclusions,
)


def route(scene: str, episode: str, offset: float = 0.0) -> Route:
    points = tuple(
        RoutePoint(
            step=index,
            world_x=offset + float(index),
            world_z=float(index % 3),
            pixel_x=100 + index * 5,
            pixel_y=100 + index,
            cumulative_m=float(index) * 2.0,
        )
        for index in range(21)
    )
    return Route(scene_name=scene, episode_id=episode, points=points)


class ResampledPointGoalPairsTest(unittest.TestCase):
    def test_candidates_exclude_route_and_benchmark_endpoints(self):
        source = route("scene1", "source")
        benchmark = [{
            "start": {"x": -100, "z": -100},
            "target": {"x": 150, "y": 110},
        }]
        candidates = candidate_pairs_for_scene(
            [source], benchmark,
            min_route_distance_m=8,
            max_route_distance_m=30,
            min_euclidean_distance_m=3,
            benchmark_clearance_m=2,
            benchmark_target_clearance_px=3,
            route_endpoint_clearance_m=4,
            candidate_spacing_m=1,
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            for point in (candidate.start, candidate.target):
                self.assertGreaterEqual(point.cumulative_m, 4)
                self.assertGreaterEqual(source.points[-1].cumulative_m - point.cumulative_m, 4)
                self.assertNotEqual((point.pixel_x, point.pixel_y), (150, 110))

    def test_manifest_has_new_pairs_for_every_scene(self):
        from nav.config import SCENE_CODES

        routes = {
            scene: [route(scene, f"{scene}-a"), route(scene, f"{scene}-b", 0.25)]
            for scene in SCENE_CODES
        }
        benchmark = {scene: [] for scene in SCENE_CODES}
        manifest = build_manifest(
            routes,
            benchmark,
            pairs_per_scene=2,
            seed=7,
            min_route_distance_m=8,
            max_route_distance_m=30,
            min_euclidean_distance_m=3,
            benchmark_clearance_m=1,
            benchmark_target_clearance_px=1,
            route_endpoint_clearance_m=2,
            candidate_spacing_m=1,
            endpoint_separation_m=0,
        )
        self.assertEqual(len(manifest), 48)
        self.assertEqual({task["scene_name"] for task in manifest}, set(SCENE_CODES))
        self.assertTrue(all(task["pairing"] == "resampled_route_interior" for task in manifest))
        self.assertTrue(all(task["source_start_step"] not in {0, 20} for task in manifest))
        self.assertTrue(all(task["source_target_step"] not in {0, 20} for task in manifest))

    def test_prior_manifest_endpoints_are_excluded(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text(
                json.dumps({
                    "scene_name": "scene1",
                    "init_world_x": 5,
                    "init_world_z": 2,
                    "target_x": 150,
                    "target_y": 110,
                    "sampled_target_world_x": 10,
                    "sampled_target_world_z": 1,
                }) + "\n",
                encoding="utf-8",
            )
            world, pixels = load_manifest_exclusions([path])
        self.assertEqual(world["scene1"], [(5.0, 2.0), (10.0, 1.0)])
        self.assertEqual(pixels["scene1"], [(150.0, 110.0)])

        candidates = candidate_pairs_for_scene(
            [route("scene1", "source")],
            [],
            min_route_distance_m=8,
            max_route_distance_m=30,
            min_euclidean_distance_m=3,
            benchmark_clearance_m=0,
            benchmark_target_clearance_px=0,
            route_endpoint_clearance_m=2,
            candidate_spacing_m=1,
            excluded_world_endpoints=world["scene1"],
            excluded_target_pixels=pixels["scene1"],
            excluded_world_clearance_m=1,
            excluded_target_clearance_px=3,
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            for point in (candidate.start, candidate.target):
                self.assertGreaterEqual(
                    min(((point.world_x - x) ** 2 + (point.world_z - z) ** 2) ** 0.5 for x, z in world["scene1"]),
                    1,
                )
                self.assertGreaterEqual(
                    ((point.pixel_x - 150) ** 2 + (point.pixel_y - 110) ** 2) ** 0.5,
                    3,
                )

    def test_selection_relaxes_source_cap_before_returning_short(self):
        from nav.scripts.tools.sample_resampled_pointgoal_pairs import (
            candidate_pairs_for_scene,
            select_pairs,
        )

        candidates = candidate_pairs_for_scene(
            [route("scene1", "only-source")],
            [],
            min_route_distance_m=8,
            max_route_distance_m=30,
            min_euclidean_distance_m=3,
            benchmark_clearance_m=0,
            benchmark_target_clearance_px=0,
            route_endpoint_clearance_m=2,
            candidate_spacing_m=1,
        )
        selected = select_pairs(candidates, 8, seed=3, endpoint_separation_m=0)
        self.assertEqual(len(selected), 8)


if __name__ == "__main__":
    unittest.main()
