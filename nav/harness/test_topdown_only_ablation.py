"""Offline checks for the nav_minimap_only.txt ablation; no paid API calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from nav.config import ACTION_SPACE_AGENTS, DEFAULT_PROMPT_TOPDOWN_ONLY
from nav.harness.navigation_protocol import navigation_run_config, resolve_navigation_sensors
from nav.harness.prompt_assembly import add_api_observation_contract, render_nav_prompt
from nav.harness.routing import execute_decision
from nav.scripts.agent import run_benchmark_cell as cell_runner
from nav.scripts.agent import run_benchmark_grid as grid
from nav.scripts.evaluation import summarize_llm_ablation as report
from nav.utils import load_prompt_template


class TopdownOnlyAblationTest(unittest.TestCase):
    def test_prompt_and_transport_contract_match_the_single_image(self):
        template = load_prompt_template(DEFAULT_PROMPT_TOPDOWN_ONLY)
        self.assertEqual(Path(DEFAULT_PROMPT_TOPDOWN_ONLY).name, "nav_minimap_only.txt")
        self.assertNotIn("{history}", template)
        prompt = render_nav_prompt(
            template, (20, 30), (80, 90), 180, 0, list(ACTION_SPACE_AGENTS),
            "THIS HISTORY MUST NOT BE SENT", curr_world_xz=(1, 2),
            target_world_xz=(5, 6), distance_m=5.66,
        )
        prompt = add_api_observation_contract(prompt, ["topdown"])
        self.assertIn("Inspect the attached image", prompt)
        self.assertIn("No egocentric camera image is provided.", prompt)
        self.assertNotIn("Egocentric RGB camera", prompt)
        self.assertNotIn("THIS HISTORY MUST NOT BE SENT", prompt)
        self.assertIn("yaw by about 45°", prompt)
        self.assertIn("moves about 1.50 m", prompt)

    @patch("nav.harness.llm_provider.call_openrouter")
    def test_provider_receives_only_minimap_and_retains_map_observation(self, call):
        call.return_value = json.dumps({
            "action": "forward", "reasoning": "Clear route on the minimap.",
            "observation": "A wall is on the left.",
        })
        image = np.zeros((256, 431, 3), dtype=np.uint8)
        result = {}
        execute_decision("llm", {
            "prompt": "navigate", "images": [("topdown", image)],
            "model_id": "test-model", "llm_provider": "openrouter",
            "max_tokens": 60000, "allowed_actions": list(ACTION_SPACE_AGENTS),
            "history_entry": {},
        }, result)
        sent = call.call_args.args[1]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], "topdown")
        self.assertIs(sent[0][1], image)
        self.assertFalse(result["error"])
        self.assertEqual(result["observation"], "A wall is on the left.")

    def test_grid_geometry_is_identical_and_output_paths_are_isolated(self):
        scenes = list(grid.SCENE_CODES)
        points = grid.load_input_points()
        tasks = [(scene, points[scene][0]["point_id"]) for scene in scenes]
        tasks += [(scene, points[scene][1]["point_id"]) for scene in scenes[:8]]
        models = [m[0] for m in report.MODELS]
        only = grid.build_grid(models, scenes, ["0"], [False], [0],
                               topdown_modes=[True], tasks=tasks)
        paired = grid.build_grid(models, scenes, ["0"], [True], [0],
                                 topdown_modes=[True], tasks=tasks)
        self.assertEqual(len(only), 64)
        self.assertEqual(len({c.scene_name for c in only}), 24)
        for a, b in zip(only, paired):
            for field in ("scene_name", "point_id", "seed_id", "init_world_x",
                          "init_world_z", "init_direction", "target_x", "target_y"):
                self.assertEqual(getattr(a, field), getattr(b, field))
            self.assertNotEqual(a.frame_save_dir, b.frame_save_dir)
            self.assertEqual(grid.cell_prompt_file(argparse.Namespace(), a), DEFAULT_PROMPT_TOPDOWN_ONLY)
            reported = report._cell_dir(
                report.TOPDOWN_ONLY_CONDITION, a.scene_name, a.point_id,
                a.model_short, Path("/unused"), grid.REPO_ROOT / "outputs",
            )
            self.assertEqual(reported, a.frame_save_dir)

    @patch("nav.scripts.agent.run_benchmark_grid.free_tcp_port", return_value=55555)
    @patch("nav.scripts.agent.run_benchmark_grid.run_cell_subprocess")
    def test_worker_sends_correct_flags_and_manifest(self, run, port):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("sys.argv", ["grid", "--models", "test/model", "--seeds", "0",
                                     "--vision_input", "off", "--topdown_input", "on",
                                     "--history_sizes", "0", "--output_root", tmpdir]):
                args = grid.parse_args()
            resolve_navigation_sensors(args)
            cell = grid.build_grid(["test/model"], ["scene1"], ["0"], [False], [0],
                                   ["point1"], output_root=tmpdir, topdown_modes=[True])[0]
            expected = grid.cell_run_config(args, cell, "/tmp/fake-unity")
            self.assertEqual(expected["input_modalities"], ["topdown"])
            self.assertFalse(expected["settings"]["vision_input"])
            self.assertEqual(expected["settings"]["history_size"], 0)

            def fake(cmd, **kwargs):
                with patch("sys.argv", ["cell", *cmd[3:]]):
                    actual_args = cell_runner.parse_args()
                resolve_navigation_sensors(actual_args)
                actual = navigation_run_config(actual_args, load_prompt_template(actual_args.prompt_file))
                self.assertEqual(actual, expected)
                cell.results_csv.write_text("stop_reason\nmax_steps\n")
                return argparse.Namespace(returncode=0)

            run.side_effect = fake
            result = grid.run_cell({**vars(args), "cell": asdict(cell),
                                    "file_name": "/tmp/fake-unity", "use_xvfb": False})
            self.assertTrue(result["ok"], result)

    def test_extension_is_opt_in_and_strictly_validates_modality_and_prompt(self):
        self.assertEqual(len(report.selected_conditions(argparse.Namespace())), 4)
        self.assertEqual(len(report.selected_conditions(argparse.Namespace(include_topdown_only=True))), 5)
        with tempfile.TemporaryDirectory() as tmpdir:
            folder = Path(tmpdir)
            args = argparse.Namespace(vision_input=False, topdown_input=True, history_size=0,
                                      model_id="test/model", scene_name="scene1", point_id="point1")
            config = navigation_run_config(args, load_prompt_template(DEFAULT_PROMPT_TOPDOWN_ONLY))
            path = folder / "run_config.json"
            path.write_text(json.dumps(config))
            check = lambda: report._config_error(folder, report.TOPDOWN_ONLY_CONDITION,
                                                 "test/model", "scene1", "point1")
            self.assertEqual(check(), "")
            config["input_modalities"] = ["ego", "topdown"]
            path.write_text(json.dumps(config))
            self.assertIn("input_modalities", check())
            config["input_modalities"] = ["topdown"]
            config["prompt_sha256"] = hashlib.sha256(b"different prompt").hexdigest()
            path.write_text(json.dumps(config))
            self.assertIn("nav_minimap_only.txt", check())

    def test_summary_includes_extension_without_changing_default_condition_count(self):
        base = dict(model_id=report.MODELS[0][0], status="complete", distance_world=1.0,
                    steps_taken=10, success_at_2m=True, success_at_5m=True, success_at_10m=True)
        rows = [{**base, "condition": c.key} for c in report.CONDITIONS]
        self.assertEqual(len(report.summarize(rows)), 4)
        rows.append({**base, "condition": report.TOPDOWN_ONLY_CONDITION.key})
        summaries = report.summarize(rows)
        self.assertEqual(len(summaries), 5)
        self.assertFalse(summaries[-1]["egocentric"])
        self.assertFalse(summaries[-1]["memory"])


if __name__ == "__main__":
    unittest.main()
