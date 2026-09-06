"""Regression tests for trajectory rendering in the GIF gallery."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from nav.scripts.export_llm_gallery import (
    COLLISION_COLOR,
    CURRENT_POSITION_RADIUS_CANONICAL,
    TRAJECTORY_COLOR,
    TRAJECTORY_LINE_WIDTH_CANONICAL,
    TRAJECTORY_POINT_RADIUS_CANONICAL,
    WARNING_COLOR,
    add_trajectory,
    allocate_gallery_gif,
    event_points,
    gallery_html,
    main,
    pending_gallery_runs,
    rotation_step_count,
    run_identity,
    scale_trajectory_points,
    target_point,
)


class TrajectoryScalingTest(unittest.TestCase):
    def test_half_resolution_cli_agent_minimap(self) -> None:
        self.assertEqual(
            scale_trajectory_points([(168, 184), (670, 390)], (431, 256)),
            [(84, 92), (335, 195)],
        )

    def test_canonical_minimap_is_unchanged(self) -> None:
        points = [(168, 184), (670, 390)]
        self.assertEqual(scale_trajectory_points(points, (862, 512)), points)

    def test_trajectory_style_scales_with_minimap_resolution(self) -> None:
        canonical = add_trajectory(
            Image.new("RGB", (862, 512), "black"),
            [(100, 256), (762, 256)],
        )
        half_resolution = add_trajectory(
            Image.new("RGB", (431, 256), "black"),
            [(100, 256), (762, 256)],
        )

        canonical_half_width = TRAJECTORY_LINE_WIDTH_CANONICAL // 2
        half_resolution_half_width = max(
            1,
            round(TRAJECTORY_LINE_WIDTH_CANONICAL * 0.5) // 2,
        )
        self.assertEqual(canonical.getpixel((431, 256)), TRAJECTORY_COLOR)
        self.assertEqual(
            half_resolution.getpixel((216, 128)),
            TRAJECTORY_COLOR,
        )
        self.assertEqual(
            canonical.getpixel((431, 256 + canonical_half_width + 1)),
            (0, 0, 0),
        )
        self.assertEqual(
            half_resolution.getpixel(
                (216, 128 + half_resolution_half_width + 1)
            ),
            (0, 0, 0),
        )
        self.assertEqual(TRAJECTORY_POINT_RADIUS_CANONICAL, 6)
        self.assertEqual(CURRENT_POSITION_RADIUS_CANONICAL, 12)

    def test_target_point_from_csv_row(self) -> None:
        self.assertEqual(
            target_point({"target_px": "480.0", "target_py": "50"}),
            (480, 50),
        )
        self.assertIsNone(target_point({}))

    def test_event_points_are_cumulative_and_skip_invalid_rows(self) -> None:
        rows = [
            {"curr_px": "10", "curr_py": "20"},
            {"curr_px": "bad", "curr_py": "30"},
            {"curr_px": "40.0", "curr_py": "50.0"},
            {"curr_px": "60", "curr_py": "70"},
        ]
        self.assertEqual(event_points(rows, {0, 1, 2, 9}, 2), [(10, 20), (40, 50)])

    def test_warning_and_collision_markers_can_share_a_position(self) -> None:
        image = Image.new("RGB", (862, 512), "black")
        rendered = add_trajectory(
            image,
            [],
            warning_points=[(100, 100), (200, 200)],
            collision_points=[(100, 100), (300, 300)],
        )
        self.assertEqual(rendered.getpixel((100, 100)), COLLISION_COLOR)
        self.assertEqual(rendered.getpixel((107, 100)), WARNING_COLOR)
        self.assertEqual(rendered.getpixel((200, 200)), WARNING_COLOR)
        self.assertEqual(rendered.getpixel((300, 300)), COLLISION_COLOR)

    def test_run_identity_supports_seeded_cli_agent_layout(self) -> None:
        self.assertEqual(
            run_identity(
                Path("outputs/scene23/point4/cli_agent_gpt-5.6-terra/seed0")
            ),
            ("scene23", "point4", "cli_agent_gpt-5.6-terra"),
        )

    def test_run_identity_supports_unseeded_astar_layout(self) -> None:
        self.assertEqual(
            run_identity(Path("outputs/scene23/point4/astar_fix_all_points")),
            ("scene23", "point4", "astar_fix_all_points"),
        )

    def test_gallery_ui_is_english(self) -> None:
        html = gallery_html([], "Test Gallery")
        self.assertIn('<html lang="en">', html)
        self.assertIn("All Models", html)
        self.assertIn("All Scenes", html)
        self.assertIn("All Results", html)
        self.assertIn("Rotation Ratio", html)
        self.assertIn("Showing ", html)
        self.assertNotRegex(html, r"[\u4e00-\u9fff]")

    def test_gallery_title_is_escaped(self) -> None:
        html = gallery_html([], 'Models <A & B>')
        self.assertIn("Models &lt;A &amp; B&gt;", html)
        self.assertNotIn("Models <A & B>", html)

    @patch("nav.scripts.export_llm_gallery.ProcessPoolExecutor")
    def test_html_only_refresh_preserves_manifest_and_gifs(self, executor) -> None:
        item = {
            "model": "test-model", "scene": "scene1", "scene_number": 1,
            "point": "point1", "success": 0, "success_at_2m": 0,
            "success_at_5m": 0, "success_at_10m": 0, "forward_steps": 1,
            "collision_steps": 0, "warning_frames": 1, "warning_steps": 0,
            "action_counts": {"forward": 1}, "gif": "scene1_point1.gif",
            "final_distance_m": 20, "distance_ratio": 0.2,
            "collision_rate": 0, "warning_rate": 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps([item]), encoding="utf-8")
            (root / "gifs").mkdir()
            gif = root / "gifs" / item["gif"]
            Image.new("RGB", (2, 2)).save(gif)
            original_manifest, original_gif = manifest.read_bytes(), gif.read_bytes()
            with patch("sys.argv", ["export-gallery", "--output-dir", tmpdir, "--html-only"]):
                main()
            html = (root / "index.html").read_text(encoding="utf-8")
            self.assertIn('<html lang="en">', html)
            for label in ("All Models", "All Scenes", "Success@2m", "Success@5m",
                          "Success@10m", "Warning Rate", "Highest Collision Rate",
                          "No trajectories match the current filters."):
                self.assertIn(label, html)
            self.assertNotRegex(html, r"[\u4e00-\u9fff]")
            self.assertEqual(manifest.read_bytes(), original_manifest)
            self.assertEqual(gif.read_bytes(), original_gif)
            executor.assert_not_called()

    def test_html_only_rejects_new_input_selection(self) -> None:
        with patch("sys.argv", ["export-gallery", "--html-only", "--limit", "1"]):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 2)

    def test_rotation_steps_use_look_signal_instead_of_action_name(self) -> None:
        rows = [
            {"action": "astar turn right", "look": "9.0"},
            {"action": "astar forward turn left", "look": "-2.25"},
            {"action": "turn left", "look": "-22.5"},
            {"action": "forward", "look": "0.0"},
            {"action": "malformed", "look": "not-a-number"},
        ]
        self.assertEqual(rotation_step_count(rows), 3)

    def test_append_skips_existing_and_missing_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            old, new, partial, empty = [root / name for name in ("old", "new", "partial", "empty")]
            for folder in (old, new, partial, empty):
                folder.mkdir()
            (new / "results.csv").write_text("stop_reason\nmax_steps\n")
            (empty / "results.csv").write_text("stop_reason\n")
            existing = [{"run_dir": str(old)}]
            pending, skipped = pending_gallery_runs([old, new, partial, empty], existing, True)
            self.assertEqual(pending, [new])
            self.assertEqual(len(skipped), 3)
            self.assertEqual(existing, [{"run_dir": str(old)}])

    def test_normal_only_rejects_provider_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reached, exhausted, failed, active = [
                root / name for name in ("reached", "exhausted", "failed", "active")
            ]
            for folder in (reached, exhausted, failed, active):
                folder.mkdir()
            (reached / "results.csv").write_text("stop_reason\nreached_vicinity\n")
            (exhausted / "results.csv").write_text("stop_reason\nmax_steps\n")
            (failed / "results.csv").write_text("stop_reason\ndecision_error\n")
            pending, skipped = pending_gallery_runs(
                [reached, exhausted, failed, active], [], True, normal_only=True,
            )
            self.assertEqual(pending, [reached, exhausted])
            self.assertEqual({folder for folder, _ in skipped}, {failed, active})

    def test_append_never_overwrites_an_existing_gif(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            gifs = Path(tmpdir)
            run = Path("outputs/scene1/point1/gemini-3.8-flash/seed0")
            reserved = set()
            existing = allocate_gallery_gif(run, gifs, reserved, qualified=True, append=True)
            Image.new("RGB", (2, 2)).save(existing)
            original = existing.read_bytes()
            new = allocate_gallery_gif(run, gifs, set(), qualified=True, append=True)
            self.assertNotEqual(existing, new)
            self.assertEqual(existing.read_bytes(), original)
            self.assertIn("gemini-3.8-flash", new.name)

    def test_gif_filenames_distinguish_seeds(self) -> None:
        reserved = set()
        paths = [allocate_gallery_gif(
            Path(f"outputs/scene1/point1/model/seed{seed}"), Path("gifs"), reserved,
            qualified=True, append=True,
        ) for seed in (0, 1)]
        self.assertNotEqual(paths[0], paths[1])


if __name__ == "__main__":
    unittest.main()
