"""Regression tests for the static top-down comparison gallery."""

from __future__ import annotations

import unittest

from PIL import Image

from nav.scripts.gallery.export_topdown_comparison_gallery import (
    COLLISION_COLOR,
    START_COLOR,
    TARGET_COLOR,
    WARNING_COLOR,
    exclude_models,
    gallery_html,
    group_manifest,
    model_color_map,
    render_map,
)


class TopdownComparisonGalleryTest(unittest.TestCase):
    def test_gpt4o_mini_can_be_excluded_without_changing_source_items(self) -> None:
        items = [
            {"model": "openai/gpt-4o-mini"},
            {"model": "gpt-5-mini"},
        ]
        filtered = exclude_models(items, ["openai/gpt-4o-mini"])
        self.assertEqual(filtered, [{"model": "gpt-5-mini"}])
        self.assertEqual(len(items), 2)

    def test_manifest_is_grouped_into_exactly_96_tasks(self) -> None:
        items = [
            {"scene": "scene1", "point": "point1", "model": "z"},
            {"scene": "scene1", "point": "point1", "model": "a"},
            {"scene": "scene24", "point": "point4", "model": "a"},
        ]
        grouped = group_manifest(items)
        self.assertEqual(len(grouped), 96)
        self.assertEqual(grouped[0][:2], ("scene1", "point1"))
        self.assertEqual([item["model"] for item in grouped[0][2]], ["a", "z"])
        self.assertEqual(grouped[-1][:2], ("scene24", "point4"))

    def test_marker_shapes_and_colors(self) -> None:
        trajectories = [{
            "model": "agent",
            "points": [(20, 20), (100, 100)],
            "warning_points": [(200, 200)],
            "collision_points": [(300, 300)],
            "target": (400, 400),
        }]
        image = render_map(
            Image.new("RGB", (862, 512), "black"),
            trajectories,
            {"agent": (14, 165, 233)},
        )
        warning = image.getpixel((200, 200))
        collision = image.getpixel((300, 300))
        self.assertNotEqual(warning, WARNING_COLOR)
        self.assertNotEqual(collision, COLLISION_COLOR)
        self.assertGreater(warning[0], warning[2])
        self.assertGreater(warning[1], warning[2])
        self.assertGreater(collision[0], collision[1])
        self.assertEqual(image.getpixel((400, 400)), TARGET_COLOR)
        self.assertEqual(image.getpixel((20, 20)), START_COLOR)
        # Warning is a triangle and collision is a cross, so areas that would
        # be filled by circular markers remain untouched.
        self.assertEqual(image.getpixel((196, 196)), (0, 0, 0))
        self.assertEqual(image.getpixel((300, 294)), (0, 0, 0))
        # Start and target use their original circular markers.
        self.assertEqual(image.getpixel((10, 20)), START_COLOR)
        self.assertEqual(image.getpixel((390, 400)), TARGET_COLOR)

    def test_trajectory_is_translucent_over_the_map(self) -> None:
        color = (14, 165, 233)
        image = render_map(
            Image.new("RGB", (862, 512), "black"),
            [{
                "model": "agent",
                "points": [(100, 256), (762, 256)],
                "warning_points": [],
                "collision_points": [],
                "target": None,
            }],
            {"agent": color},
        )
        center = image.getpixel((431, 256))
        self.assertNotEqual(center, color)
        self.assertGreater(sum(center), 0)

    def test_gallery_is_english_and_reports_96_cards(self) -> None:
        items = [
            {
                "scene": f"scene{scene}",
                "scene_number": scene,
                "point": f"point{point}",
                "point_number": point,
                "image": f"scene{scene}_point{point}.png",
                "agent_count": 2,
                "warning_events": 1,
                "collision_events": 0,
            }
            for scene in range(1, 25)
            for point in range(1, 5)
        ]
        html = gallery_html(items, {"agent": (14, 165, 233)})
        self.assertIn('<html lang="en">', html)
        self.assertIn("All Scenes", html)
        self.assertIn("Showing ${shown.length} / ${items.length} comparisons", html)
        self.assertEqual(html.count('"image": "scene'), 96)
        self.assertNotRegex(html, r"[\u4e00-\u9fff]")

    def test_model_colors_are_global_and_stable(self) -> None:
        first = model_color_map([{"model": "b"}, {"model": "a"}])
        second = model_color_map([{"model": "a"}, {"model": "b"}])
        self.assertEqual(first, second)
        self.assertNotEqual(first["a"], first["b"])


if __name__ == "__main__":
    unittest.main()
